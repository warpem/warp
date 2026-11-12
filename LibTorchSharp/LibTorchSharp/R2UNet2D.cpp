#include "CustomModules.h"

#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include <torch/nn/init.h>
#include <torch/serialize.h>
#include <ATen/autocast_mode.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "MultiGPUModule.h"

// Noise2Mic's R2-U-Net (noise2mic/src/model.py, class R2UNet), translated layer for layer.
// Submodule names follow the PyTorch attribute names, so a PyTorch state dict loads unchanged
// and a model saved here loads in PyTorch.

namespace
{
    const double R2Slope = 0.01; // nn.LeakyReLU() default

    // RRCU.CONV_GROUP_SIZE = 128: the largest divisor of channels not above channels / 128.
    int64_t r2_conv_groups(int64_t channels)
    {
        const int64_t size = 128;
        if (channels <= size)
            return 1;
        for (int64_t groups = channels / size; groups > 0; groups--)
            if (channels % groups == 0)
                return groups;
        return 1;
    }

    int64_t r2_norm_groups(int64_t channels)
    {
        int64_t groups = channels >= 32 ? std::min<int64_t>(32, channels) : std::max<int64_t>(1, channels);
        while (channels % groups != 0 && groups > 1)
            groups /= 2;
        if (groups == 0)
            groups = 1;
        return std::min<int64_t>(8, groups);
    }

    torch::nn::Conv2dOptions r2_conv(int64_t in, int64_t out, int64_t kernel, int64_t stride = 1, int64_t padding = 0, bool bias = false, int64_t groups = 1)
    {
        return torch::nn::Conv2dOptions(in, out, kernel).stride(stride).padding(padding).bias(bias).groups(groups);
    }
}

struct R2RRCUImpl : torch::nn::Module
{
    int64_t t;
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::GroupNorm norm1{ nullptr }, norm2{ nullptr };

    R2RRCUImpl(int64_t channels, int64_t steps) : t(steps)
    {
        int64_t groups = r2_conv_groups(channels), norms = r2_norm_groups(channels);
        conv1 = register_module("conv1", torch::nn::Conv2d(r2_conv(channels, channels, 3, 1, 1, false, groups)));
        norm1 = register_module("norm1", torch::nn::GroupNorm(torch::nn::GroupNormOptions(norms, channels)));
        conv2 = register_module("conv2", torch::nn::Conv2d(r2_conv(channels, channels, 3, 1, 1, false, groups)));
        norm2 = register_module("norm2", torch::nn::GroupNorm(torch::nn::GroupNormOptions(norms, channels)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor out = torch::leaky_relu(norm1->forward(conv1->forward(x)), R2Slope);
        torch::Tensor recurrent = out;
        for (int64_t i = 0; i < t; i++)
            recurrent = torch::leaky_relu(norm2->forward(conv2->forward(recurrent + out)), R2Slope);
        return x + recurrent;
    }
};
TORCH_MODULE(R2RRCU);

struct R2EncoderBlockImpl : torch::nn::Module
{
    torch::nn::Conv2d conv_in{ nullptr }, downsample{ nullptr };
    torch::nn::Sequential res_blocks{ nullptr };

    R2EncoderBlockImpl(int64_t in, int64_t out, int64_t blocks, int64_t t)
    {
        conv_in = register_module("conv_in", torch::nn::Conv2d(r2_conv(in, out, 1)));
        res_blocks = torch::nn::Sequential();
        for (int64_t i = 0; i < blocks; i++)
            res_blocks->push_back(R2RRCU(out, t));
        register_module("res_blocks", res_blocks);
        downsample = register_module("downsample", torch::nn::Conv2d(r2_conv(out, out, 4, 2, 1)));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = torch::leaky_relu(conv_in->forward(x), R2Slope);
        x = res_blocks->forward(x);
        torch::Tensor skip = x;
        x = torch::leaky_relu(downsample->forward(x), R2Slope);
        return std::make_tuple(x, skip);
    }
};
TORCH_MODULE(R2EncoderBlock);

struct R2DecoderBlockImpl : torch::nn::Module
{
    torch::nn::Sequential upsample{ nullptr }, res_blocks{ nullptr };
    torch::nn::Conv2d conv_in{ nullptr };

    R2DecoderBlockImpl(int64_t in, int64_t skip, int64_t out, int64_t blocks, int64_t t)
    {
        upsample = torch::nn::Sequential(
            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)),
            torch::nn::Conv2d(r2_conv(in, in / 2, 1)));
        register_module("upsample", upsample);
        conv_in = register_module("conv_in", torch::nn::Conv2d(r2_conv(in / 2 + skip, out, 1)));
        res_blocks = torch::nn::Sequential();
        for (int64_t i = 0; i < blocks; i++)
            res_blocks->push_back(R2RRCU(out, t));
        register_module("res_blocks", res_blocks);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor skip)
    {
        x = torch::leaky_relu(upsample->forward(x), R2Slope);
        if (x.sizes().slice(2) != skip.sizes().slice(2))
            x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>({ skip.size(2), skip.size(3) })).mode(torch::kBilinear).align_corners(true));
        x = torch::cat({ x, skip }, 1);
        x = torch::leaky_relu(conv_in->forward(x), R2Slope);
        return res_blocks->forward(x);
    }
};
TORCH_MODULE(R2DecoderBlock);

struct R2UNet2DImpl : MultiGPUModule
{
    torch::nn::Conv2d conv_in{ nullptr }, conv_out{ nullptr };
    torch::nn::ModuleList encoders{ nullptr }, decoders{ nullptr };
    torch::nn::Sequential bottleneck{ nullptr };

    R2UNet2DImpl(int64_t patch_size, int64_t features_base, int64_t blocks, int64_t t, int64_t min_feature_size)
    {
        if (patch_size < 1 || (patch_size & (patch_size - 1)) != 0 || min_feature_size < 1 || (min_feature_size & (min_feature_size - 1)) != 0 || patch_size < min_feature_size || t < 1)
            throw std::invalid_argument("R2UNet2D: patch and minimum feature sizes must be powers of 2, patch >= minimum, t >= 1.");
        int64_t levels = std::max<int64_t>(1, (int64_t)std::log2((double)patch_size / min_feature_size));

        conv_in = register_module("conv_in", torch::nn::Conv2d(r2_conv(1, features_base, 5, 1, 2)));

        // Python keeps doubling an unclamped width while each block clamps its own channels to 512.
        encoders = torch::nn::ModuleList();
        std::vector<int64_t> encoder_out;
        int64_t current = features_base;
        for (int64_t level = 0; level < levels; level++)
        {
            int64_t out = std::min<int64_t>(512, current * 2);
            encoders->push_back(R2EncoderBlock(std::min<int64_t>(512, current), out, blocks, t));
            encoder_out.push_back(out);
            current *= 2;
        }
        register_module("encoders", encoders);

        bottleneck = torch::nn::Sequential(R2RRCU(std::min<int64_t>(512, current), t));
        register_module("bottleneck", bottleneck);

        decoders = torch::nn::ModuleList();
        for (int64_t level = levels - 1; level >= 0; level--)
        {
            int64_t out = current / 2;
            decoders->push_back(R2DecoderBlock(std::min<int64_t>(512, current), encoder_out[level], std::min<int64_t>(512, out), blocks, t));
            current = out;
        }
        register_module("decoders", decoders);

        conv_out = register_module("conv_out", torch::nn::Conv2d(r2_conv(current, 1, 1, 1, 0, true)));

        // _init_weights: Kaiming normal (a = 0.2, fan_in, leaky ReLU) on convolutions, zero biases; GroupNorm 1/0.
        torch::NoGradGuard guard;
        for (auto& module : modules(false))
        {
            if (auto* conv = module->as<torch::nn::Conv2dImpl>())
            {
                torch::nn::init::kaiming_normal_(conv->weight, 0.2, torch::kFanIn, torch::kLeakyReLU);
                if (conv->bias.defined())
                    torch::nn::init::constant_(conv->bias, 0);
            }
            else if (auto* norm = module->as<torch::nn::GroupNormImpl>())
            {
                torch::nn::init::constant_(norm->weight, 1);
                torch::nn::init::constant_(norm->bias, 0);
            }
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor features = conv_in->forward(x);
        std::vector<torch::Tensor> skips;
        for (const auto& encoder : *encoders)
        {
            auto result = encoder->as<R2EncoderBlockImpl>()->forward(features);
            features = std::get<0>(result);
            skips.push_back(std::get<1>(result));
        }
        features = bottleneck->forward(features);
        int64_t i = 0;
        for (const auto& decoder : *decoders)
            features = decoder->as<R2DecoderBlockImpl>()->forward(features, skips[skips.size() - 1 - i++]);
        return x + conv_out->forward(features);
    }
};
TORCH_MODULE(R2UNet2D);

namespace
{
    NNModule r2_create(int64_t patch_size, int64_t features_base, int64_t blocks, int64_t t, int64_t min_feature_size, NNAnyModule* outAsAnyModule)
    {
        at::globalContext().setBenchmarkCuDNN(true);
        auto mod = std::make_shared<R2UNet2DImpl>(patch_size, features_base, blocks, t, min_feature_size);
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<R2UNet2DImpl>(mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        return new std::shared_ptr<torch::nn::Module>(mod);
    }

    typedef c10::Dict<c10::IValue, c10::IValue> GenericDict;

    // PyTorch checkpoints are torch.save'd dicts: either a bare state dict or one under "model_state_dict"
    // (Noise2Mic's training checkpoints). DataParallel ("module.") and torch.compile ("_orig_mod.") prefixes are dropped.
    void r2_load(const NNModule module, const char* location)
    {
        std::ifstream file(location, std::ios::binary);
        if (!file)
            throw std::runtime_error(std::string("Cannot open ") + location);
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        c10::IValue loaded = torch::pickle_load(bytes);
        if (!loaded.isGenericDict())
            throw std::runtime_error("Checkpoint is not a dictionary.");
        GenericDict dict = loaded.toGenericDict();
        c10::IValue key(std::string("model_state_dict"));
        if (dict.contains(key))
            dict = dict.at(key).toGenericDict();

        auto parameters = (*module)->named_parameters(true);
        auto buffers = (*module)->named_buffers(true);
        const std::vector<std::string> prefixes = { "module.", "_orig_mod." };
        size_t assigned = 0;
        torch::NoGradGuard guard;
        for (const auto& entry : dict)
        {
            std::string name = entry.key().toStringRef();
            for (const auto& prefix : prefixes)
                while (name.rfind(prefix, 0) == 0)
                    name = name.substr(prefix.size());
            torch::Tensor* target = parameters.find(name);
            if (target == nullptr)
                target = buffers.find(name);
            if (target == nullptr)
                throw std::runtime_error("Checkpoint has a tensor this network does not: " + name);
            torch::Tensor value = entry.value().toTensor();
            if (value.sizes() != target->sizes())
                throw std::runtime_error("Shape mismatch for " + name);
            target->copy_(value.to(target->device(), target->scalar_type()));
            assigned++;
        }
        if (assigned != parameters.size() + buffers.size())
            throw std::runtime_error("Checkpoint is missing tensors for this network.");
    }

    // Writes a dict PyTorch's torch.load reads, with the metadata Noise2Mic's inference expects.
    void r2_save(const NNModule module, const char* location, int64_t patch_size, int64_t features, int64_t depth, int64_t t,
                 double deconv_strength, double deconv_falloff, double deconv_highpass)
    {
        GenericDict state(c10::StringType::get(), c10::TensorType::get());
        for (const auto& entry : (*module)->named_parameters(true))
            state.insert(entry.key(), entry.value().detach().to(torch::kCPU));
        for (const auto& entry : (*module)->named_buffers(true))
            state.insert(entry.key(), entry.value().detach().to(torch::kCPU));
        GenericDict checkpoint(c10::StringType::get(), c10::AnyType::get());
        checkpoint.insert(std::string("model_state_dict"), state);
        checkpoint.insert(std::string("patch_size"), patch_size);
        checkpoint.insert(std::string("model_features"), features);
        checkpoint.insert(std::string("model_depth"), depth);
        checkpoint.insert(std::string("t"), t);
        checkpoint.insert(std::string("deconv_strength"), deconv_strength);
        checkpoint.insert(std::string("deconv_falloff"), deconv_falloff);
        checkpoint.insert(std::string("deconv_highpass_a"), deconv_highpass);
        std::vector<char> bytes = torch::pickle_save(c10::IValue(checkpoint));
        std::ofstream file(location, std::ios::binary);
        if (!file)
            throw std::runtime_error(std::string("Cannot write ") + location);
        file.write(bytes.data(), (std::streamsize)bytes.size());
    }
}

NNModule THSNN_R2UNet2D_ctor(const int64_t patch_size, const int64_t features_base, const int64_t blocks, const int64_t t, const int64_t min_feature_size, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(res = r2_create(patch_size, features_base, blocks, t, min_feature_size, outAsAnyModule));
}

Tensor THSNN_R2UNet2D_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<R2UNet2DImpl>()->forward(*input));
}

namespace
{
    // Python's torch.autocast sets these same thread-local dispatcher switches; only GradScaler has no C++
    // counterpart, and bfloat16 does not need one. precision: 0 float32, 1 float16, 2 bfloat16.
    torch::Tensor r2_forward_autocast(const NNModule module, const torch::Tensor& input, int64_t precision)
    {
        auto* network = (*module)->as<R2UNet2DImpl>();
        if (precision == 0)
            return network->forward(input);
        bool enabled = at::autocast::is_autocast_enabled(at::kCUDA);
        at::ScalarType dtype = at::autocast::get_autocast_dtype(at::kCUDA);
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        at::autocast::set_autocast_dtype(at::kCUDA, precision == 1 ? at::kHalf : at::kBFloat16);
        at::autocast::increment_nesting();
        torch::Tensor output;
        try { output = network->forward(input).to(torch::kFloat); }
        catch (...)
        {
            if (at::autocast::decrement_nesting() == 0) at::autocast::clear_cache();
            at::autocast::set_autocast_enabled(at::kCUDA, enabled); at::autocast::set_autocast_dtype(at::kCUDA, dtype);
            throw;
        }
        if (at::autocast::decrement_nesting() == 0) at::autocast::clear_cache();
        at::autocast::set_autocast_enabled(at::kCUDA, enabled);
        at::autocast::set_autocast_dtype(at::kCUDA, dtype);
        return output;
    }
}

Tensor THSNN_R2UNet2D_forward_autocast(const NNModule module, const Tensor input, const int64_t precision)
{
    CATCH_TENSOR(r2_forward_autocast(module, *input, precision));
}

// Peak bytes the caching allocator handed out and reserved since the last reset; reserved is what the GPU sees.
void THSTorch_cuda_peak_memory(const int64_t device, int64_t* allocated, int64_t* reserved, const bool reset)
{
    CATCH(
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats((c10::DeviceIndex)device);
        *allocated = stats.allocated_bytes[0].peak;
        *reserved = stats.reserved_bytes[0].peak;
        if (reset) c10::cuda::CUDACachingAllocator::resetPeakStats((c10::DeviceIndex)device);
    );
}

// bfloat16 tensor cores arrived with compute capability 8.0 (Ampere).
int THSTorch_cuda_supports_bf16(const int64_t device)
{
    int result = 0;
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, (int)device) == cudaSuccess)
        result = major >= 8 ? 1 : 0;
    return result;
}

void THSNN_Module_load_state_dict(const NNModule module, const char* location)
{
    CATCH(r2_load(module, location););
}

void THSNN_Module_save_state_dict(const NNModule module, const char* location, const int64_t patch_size, const int64_t features, const int64_t depth, const int64_t t,
                                  const double deconv_strength, const double deconv_falloff, const double deconv_highpass)
{
    CATCH(r2_save(module, location, patch_size, features, depth, t, deconv_strength, deconv_falloff, deconv_highpass););
}
