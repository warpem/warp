#include "CustomModules.h"

#include <ATen/ATen.h>

// JunkNet: a ConvNeXt-Tiny encoder, whole-image attention evaluated twice with tied weights, and a decoder whose
// fixed-width cells recur once, predicting one binary junk logit per pixel of a 6 Å/pixel micrograph (carbon, gold,
// ice). Translated layer for layer from the PyTorch model; submodule names follow its attribute names, so its state
// dict loads unchanged.

namespace
{
    namespace F = torch::nn::functional;

    torch::nn::Conv2dOptions jn_conv(int64_t in, int64_t out, int64_t kernel, int64_t stride = 1, int64_t padding = 0, int64_t groups = 1)
    {
        return torch::nn::Conv2dOptions(in, out, kernel).stride(stride).padding(padding).groups(groups);
    }

    torch::Tensor jn_resize(const torch::Tensor& x, int64_t height, int64_t width)
    {
        return F::interpolate(x, F::InterpolateFuncOptions().size(std::vector<int64_t>({ height, width })).mode(torch::kBilinear).align_corners(false));
    }
}

// torchvision's LayerNorm2d: a LayerNorm over channels at every pixel.
struct JNLayerNorm2dImpl : torch::nn::Module
{
    torch::Tensor weight, bias;

    explicit JNLayerNorm2dImpl(int64_t channels)
    {
        weight = register_parameter("weight", torch::ones({ channels }));
        bias = register_parameter("bias", torch::zeros({ channels }));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return torch::layer_norm(x.permute({ 0, 2, 3, 1 }), { x.size(1) }, weight, bias, 1e-6).permute({ 0, 3, 1, 2 });
    }
};
TORCH_MODULE(JNLayerNorm2d);

// CNBlock.block: depthwise 7×7, (permute), LayerNorm, Linear 4×, GELU, Linear, (permute). Children keep the
// Sequential's indices; the permutes and GELU have no parameters.
struct JNBlockBodyImpl : torch::nn::Module
{
    torch::nn::Conv2d depthwise{ nullptr };
    torch::nn::LayerNorm norm{ nullptr };
    torch::nn::Linear expand{ nullptr }, contract{ nullptr };

    explicit JNBlockBodyImpl(int64_t dim)
    {
        depthwise = register_module("0", torch::nn::Conv2d(jn_conv(dim, dim, 7, 1, 3, dim)));
        norm = register_module("2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim }).eps(1e-6)));
        expand = register_module("3", torch::nn::Linear(dim, 4 * dim));
        contract = register_module("5", torch::nn::Linear(4 * dim, dim));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = depthwise->forward(x).permute({ 0, 2, 3, 1 });
        x = contract->forward(torch::gelu(expand->forward(norm->forward(x))));
        return x.permute({ 0, 3, 1, 2 });
    }
};
TORCH_MODULE(JNBlockBody);

// CNBlock: x + layer_scale · block(x); stochastic depth is the identity at inference.
struct JNCNBlockImpl : torch::nn::Module
{
    JNBlockBody block{ nullptr };
    torch::Tensor layer_scale;

    explicit JNCNBlockImpl(int64_t dim)
    {
        block = register_module("block", JNBlockBody(dim));
        layer_scale = register_parameter("layer_scale", torch::full({ dim, 1, 1 }, 1e-6));
    }

    torch::Tensor forward(torch::Tensor x) { return x + layer_scale * block->forward(x); }
};
TORCH_MODULE(JNCNBlock);

// torchvision ConvNeXt features are a Sequential of stem, stages and downsampling layers; each part is one of these.
struct JNEncoderPart : torch::nn::Module
{
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};

// Stem: Conv 4×4 stride 4, LayerNorm2d.
struct JNStem : JNEncoderPart
{
    torch::nn::Conv2d conv{ nullptr };
    JNLayerNorm2d norm{ nullptr };

    JNStem()
    {
        conv = register_module("0", torch::nn::Conv2d(jn_conv(3, 96, 4, 4)));
        norm = register_module("1", JNLayerNorm2d(96));
    }

    torch::Tensor forward(torch::Tensor x) override { return norm->forward(conv->forward(x)); }
};

// Downsampling: LayerNorm2d, Conv 2×2 stride 2.
struct JNDownsample : JNEncoderPart
{
    JNLayerNorm2d norm{ nullptr };
    torch::nn::Conv2d conv{ nullptr };

    JNDownsample(int64_t in, int64_t out)
    {
        norm = register_module("0", JNLayerNorm2d(in));
        conv = register_module("1", torch::nn::Conv2d(jn_conv(in, out, 2, 2)));
    }

    torch::Tensor forward(torch::Tensor x) override { return conv->forward(norm->forward(x)); }
};

struct JNStage : JNEncoderPart
{
    std::vector<JNCNBlock> blocks;

    JNStage(int64_t dim, int64_t depth)
    {
        for (int64_t i = 0; i < depth; i++)
            blocks.push_back(register_module(std::to_string(i), JNCNBlock(dim)));
    }

    torch::Tensor forward(torch::Tensor x) override
    {
        for (auto& block : blocks)
            x = block->forward(x);
        return x;
    }
};

// Two Conv/GroupNorm/GELU cells (children 0,1 and 3,4). The second has fixed width, so the recurrent decoder applies it
// again on its own output.
struct JNDecoderBlockImpl : torch::nn::Module
{
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::GroupNorm norm1{ nullptr }, norm2{ nullptr };

    JNDecoderBlockImpl(int64_t in, int64_t out)
    {
        conv1 = register_module("0", torch::nn::Conv2d(jn_conv(in, out, 3, 1, 1)));
        norm1 = register_module("1", torch::nn::GroupNorm(torch::nn::GroupNormOptions(8, out)));
        conv2 = register_module("3", torch::nn::Conv2d(jn_conv(out, out, 3, 1, 1)));
        norm2 = register_module("4", torch::nn::GroupNorm(torch::nn::GroupNormOptions(8, out)));
    }

    torch::Tensor second(torch::Tensor x) { return torch::gelu(norm2->forward(conv2->forward(x))); }
    torch::Tensor forward(torch::Tensor x) { return second(torch::gelu(norm1->forward(conv1->forward(x)))); }
};
TORCH_MODULE(JNDecoderBlock);

// The stride-2 detail path: Conv 3×3 stride 2, GroupNorm, GELU.
struct JNDetailImpl : torch::nn::Module
{
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::GroupNorm norm{ nullptr };

    JNDetailImpl()
    {
        conv = register_module("0", torch::nn::Conv2d(jn_conv(1, 24, 3, 2, 1)));
        norm = register_module("1", torch::nn::GroupNorm(torch::nn::GroupNormOptions(8, 24)));
    }

    torch::Tensor forward(torch::Tensor x) { return torch::gelu(norm->forward(conv->forward(x))); }
};
TORCH_MODULE(JNDetail);

// nn.MultiheadAttention's projections, with scaled dot-product attention over all tokens.
struct JNAttentionImpl : torch::nn::Module
{
    int64_t heads;
    torch::Tensor in_proj_weight, in_proj_bias;
    torch::nn::Linear out_proj{ nullptr };

    JNAttentionImpl(int64_t dim, int64_t heads_) : heads(heads_)
    {
        in_proj_weight = register_parameter("in_proj_weight", torch::empty({ 3 * dim, dim }));
        in_proj_bias = register_parameter("in_proj_bias", torch::zeros({ 3 * dim }));
        out_proj = register_module("out_proj", torch::nn::Linear(dim, dim));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        int64_t batch = x.size(0), tokens = x.size(1), dim = x.size(2);
        auto qkv = torch::linear(x, in_proj_weight, in_proj_bias).chunk(3, -1);
        auto split = [&](const torch::Tensor& t) { return t.reshape({ batch, tokens, heads, dim / heads }).transpose(1, 2); };
        auto attended = at::scaled_dot_product_attention(split(qkv[0]), split(qkv[1]), split(qkv[2]));
        return out_proj->forward(attended.transpose(1, 2).reshape({ batch, tokens, dim }));
    }
};
TORCH_MODULE(JNAttention);

// nn.TransformerEncoderLayer(norm_first=True, activation=ReLU, dropout=0).
struct JNEncoderLayerImpl : torch::nn::Module
{
    JNAttention self_attn{ nullptr };
    torch::nn::Linear linear1{ nullptr }, linear2{ nullptr };
    torch::nn::LayerNorm norm1{ nullptr }, norm2{ nullptr };

    JNEncoderLayerImpl(int64_t dim, int64_t heads, int64_t feedforward)
    {
        self_attn = register_module("self_attn", JNAttention(dim, heads));
        linear1 = register_module("linear1", torch::nn::Linear(dim, feedforward));
        linear2 = register_module("linear2", torch::nn::Linear(feedforward, dim));
        norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));
        norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = x + self_attn->forward(norm1->forward(x));
        return x + linear2->forward(torch::relu(linear1->forward(norm2->forward(x))));
    }
};
TORCH_MODULE(JNEncoderLayer);

struct JNContextImpl : torch::nn::Module
{
    torch::nn::ModuleList layers{ nullptr };

    JNContextImpl()
    {
        layers = register_module("layers", torch::nn::ModuleList(JNEncoderLayer(192, 6, 768), JNEncoderLayer(192, 6, 768)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        for (const auto& layer : *layers)
            x = layer->as<JNEncoderLayerImpl>()->forward(x);
        return x;
    }
};
TORCH_MODULE(JNContext);

// BoxNet's body without its heads (JunkNet removed them).
struct JNBackboneImpl : torch::nn::Module
{
    torch::nn::ModuleList encoder{ nullptr };
    std::vector<std::shared_ptr<JNEncoderPart>> parts;
    JNDetail detail{ nullptr };
    torch::nn::Conv2d context_in{ nullptr }, position{ nullptr };
    JNContext context{ nullptr };
    JNDecoderBlock up16{ nullptr }, up8{ nullptr }, up4{ nullptr }, up2{ nullptr };
    torch::nn::Embedding context_condition{ nullptr }, detail_condition{ nullptr };
    torch::Tensor rgb_mean, rgb_std;

    JNBackboneImpl()
    {
        // torchvision convnext_tiny().features: stem, then stages of 3, 3, 9, 3 blocks with downsampling between them.
        const int64_t widths[] = { 96, 192, 384, 768 }, depths[] = { 3, 3, 9, 3 };
        parts.push_back(std::make_shared<JNStem>());
        for (int s = 0; s < 4; s++)
        {
            parts.push_back(std::make_shared<JNStage>(widths[s], depths[s]));
            if (s < 3)
                parts.push_back(std::make_shared<JNDownsample>(widths[s], widths[s + 1]));
        }
        encoder = torch::nn::ModuleList();
        for (const auto& part : parts)
            encoder->push_back(part);
        register_module("encoder", encoder);
        detail = register_module("detail", JNDetail());
        context_in = register_module("context_in", torch::nn::Conv2d(jn_conv(768, 192, 1)));
        position = register_module("position", torch::nn::Conv2d(jn_conv(2, 192, 1)));
        context = register_module("context", JNContext());
        up16 = register_module("up16", JNDecoderBlock(192 + 384, 192));
        up8 = register_module("up8", JNDecoderBlock(192 + 192, 96));
        up4 = register_module("up4", JNDecoderBlock(96 + 96, 64));
        up2 = register_module("up2", JNDecoderBlock(64 + 24, 32));
        context_condition = register_module("context_condition", torch::nn::Embedding(2, 384));
        detail_condition = register_module("detail_condition", torch::nn::Embedding(2, 64));
        rgb_mean = register_buffer("rgb_mean", torch::tensor({ 0.485f, 0.456f, 0.406f }).view({ 1, 3, 1, 1 }));
        rgb_std = register_buffer("rgb_std", torch::tensor({ 0.229f, 0.224f, 0.225f }).view({ 1, 3, 1, 1 }));
    }
};
TORCH_MODULE(JNBackbone);

struct JunkNetImpl : torch::nn::Module
{
    JNBackbone backbone{ nullptr };
    torch::nn::Conv2d classifier{ nullptr };
    std::shared_ptr<torch::nn::Module> refinement_gates;
    torch::Tensor gate_context, gate_up16, gate_up8, gate_up4, gate_up2;

    JunkNetImpl()
    {
        backbone = register_module("backbone", JNBackbone());
        classifier = register_module("classifier", torch::nn::Conv2d(jn_conv(32, 2, 1)));
        refinement_gates = register_module("refinement_gates", std::make_shared<torch::nn::Module>());
        gate_context = refinement_gates->register_parameter("context", torch::zeros({}));
        gate_up16 = refinement_gates->register_parameter("up16", torch::zeros({}));
        gate_up8 = refinement_gates->register_parameter("up8", torch::zeros({}));
        gate_up4 = refinement_gates->register_parameter("up4", torch::zeros({}));
        gate_up2 = refinement_gates->register_parameter("up2", torch::zeros({}));
    }

    static torch::Tensor refine(const torch::Tensor& gate, const torch::Tensor& first, const torch::Tensor& second)
    {
        return first + gate.tanh().to(first.scalar_type()) * (second - first);
    }

    // The "particles" row of a mode embedding, as (scale, shift) for broadcasting over N, H, W.
    static std::pair<torch::Tensor, torch::Tensor> condition(const torch::nn::Embedding& embedding, int64_t channels)
    {
        auto row = embedding->weight[0];
        return { row.slice(0, 0, channels).view({ 1, channels, 1, 1 }), row.slice(0, channels, 2 * channels).view({ 1, channels, 1, 1 }) };
    }

    torch::Tensor forward(torch::Tensor image)
    {
        auto& b = *backbone;
        int64_t height = image.size(2), width = image.size(3);
        int64_t padHeight = (32 - height % 32) % 32, padWidth = (32 - width % 32) % 32;
        image = F::pad(image, F::PadFuncOptions({ 0, padWidth, 0, padHeight }).mode(torch::kReplicate));

        auto detail = b.detail->forward(image);
        auto x = ((image / 6 + 0.5).clamp(0, 1).expand({ -1, 3, -1, -1 }) - b.rgb_mean) / b.rgb_std;
        std::vector<torch::Tensor> skips;
        int64_t index = 0;
        for (const auto& part : b.parts)
        {
            x = part->forward(x);
            if (index++ % 2 == 1)
                skips.push_back(x);
        }

        x = b.context_in->forward(x);
        auto [scale, shift] = condition(b.context_condition, 192);
        x = x * (1 + scale) + shift;
        int64_t contextHeight = x.size(2), contextWidth = x.size(3);
        auto grid = torch::meshgrid({ torch::linspace(-1, 1, contextHeight, x.options()), torch::linspace(-1, 1, contextWidth, x.options()) }, "ij");
        x = x + b.position->forward(torch::stack({ grid[1], grid[0] }).unsqueeze(0));
        auto tokens = b.context->forward(x.flatten(2).transpose(1, 2));
        tokens = refine(gate_context, tokens, b.context->forward(tokens));
        x = tokens.transpose(1, 2).reshape({ -1, 192, contextHeight, contextWidth });

        const std::pair<JNDecoderBlock*, torch::Tensor*> stages[] = { { &b.up16, &skips[2] }, { &b.up8, &skips[1] }, { &b.up4, &skips[0] }, { &b.up2, &detail } };
        const torch::Tensor* gates[] = { &gate_up16, &gate_up8, &gate_up4, &gate_up2 };
        for (int i = 0; i < 4; i++)
        {
            auto& decoder = *stages[i].first;
            const auto& skip = *stages[i].second;
            auto first = decoder->forward(torch::cat({ jn_resize(x, skip.size(2), skip.size(3)), skip }, 1));
            x = refine(*gates[i], first, decoder->second(first));
        }
        auto [detailScale, detailShift] = condition(b.detail_condition, 32);
        x = x * (1 + detailScale) + detailShift;

        auto latent = jn_resize(classifier->forward(x), height + padHeight, width + padWidth).slice(2, 0, height).slice(3, 0, width);
        auto u = latent.slice(1, 0, 1), v = latent.slice(1, 1, 2);
        return torch::logsumexp(torch::cat({ u, v, u + v }, 1), 1, true);
    }
};
TORCH_MODULE(JunkNet);

namespace
{
    NNModule jn_create(NNAnyModule* outAsAnyModule)
    {
        at::globalContext().setBenchmarkCuDNN(true);
        auto mod = std::make_shared<JunkNetImpl>();
        if (outAsAnyModule != NULL)
        {
            auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<JunkNetImpl>(mod));
            *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
        }
        return new std::shared_ptr<torch::nn::Module>(mod);
    }

    // Inference only: no autograd state is recorded, and the output is contiguous for copying to the host.
    torch::Tensor jn_forward(const NNModule module, const torch::Tensor& input)
    {
        c10::InferenceMode inference;
        torch::NoGradGuard noGrad;
        auto* network = (*module)->as<JunkNetImpl>();
        network->eval();
        return network->forward(input).contiguous();
    }
}

NNModule THSNN_JunkNet_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(res = jn_create(outAsAnyModule));
}

Tensor THSNN_JunkNet_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR(jn_forward(module, *input));
}
