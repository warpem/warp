#include "CustomModules.h"

#include <torch/nn/init.h>
#include <torch/nn/pimpl.h>
#include <torch/fft.h>
#include <torch/nn/parallel/data_parallel.h>

#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>

#include "MultiGPUModule.h"

torch::nn::Conv2dOptions conv_options2d(int64_t in_planes,
    int64_t out_planes,
    int64_t kernel_size = 3,
    int64_t stride = 1,
    int64_t padding = 1,
    bool with_bias = false)
{
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kernel_size)
        .stride(stride)
        .padding(padding)
        .bias(with_bias);

    return conv_options;
}

struct UNet2DResidualBlock : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;
    bool _dochannelattn;
    bool _dospatialattn;

    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::BatchNorm2d bn2{ nullptr };

    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };

    torch::nn::Conv2d convsa{ nullptr };

    UNet2DResidualBlock(int64_t channels, bool donorm = true, bool dochannelattn = false, bool dospatialattn = false)
    {
        _channels = channels;
        _donorm = donorm;
        _dochannelattn = dochannelattn;
        _dospatialattn = dospatialattn;

        int ngroups = 32;
        while (ngroups > 1 && _channels % ngroups != 0)
            ngroups /= 2;

        conv1 = register_module("conv1", torch::nn::Conv2d(conv_options2d(_channels, _channels, 3, 1, 1).groups(std::min((int64_t)1, channels / 32))));
        if (donorm)
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(_channels)));

        conv2 = register_module("conv2", torch::nn::Conv2d(conv_options2d(_channels, _channels, 3, 1, 1).groups(std::min((int64_t)1, channels / 32))));
        if (donorm)
            bn2 = register_module("bn2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(_channels)));

        if (dochannelattn)
        {
            fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(_channels * 2, _channels / 8)));
            fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(_channels / 8, _channels)));
        }

        if (dospatialattn)
            convsa = register_module("convsa", torch::nn::Conv2d(conv_options2d(2, 1, 7, 1, 3)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor residual(x.clone());

        x = conv1->forward(x);
        if (_donorm)
            x = bn1->forward(x);
        x = torch::leaky_relu(x, 0.2);

        x = conv2->forward(x);
        if (_donorm)
            x = bn2->forward(x);

        if (_dochannelattn)
        {
            torch::Tensor ca = x.flatten(2, 3);
            ca = torch::cat({ ca.mean(2, false), std::get<0>(ca.max(2, false)) }, 1);

            ca = fc1->forward(ca);
            ca = torch::relu(ca);

            ca = fc2->forward(ca);
            ca = torch::sigmoid(ca);

            ca = ca.unsqueeze(2);
            ca = ca.unsqueeze(3);

            x = x.mul(ca);
        }

        if (x.size(1) <= 128 && _dospatialattn)
        {
            torch::Tensor sa = torch::cat({ x.mean(1, true), std::get<0>(x.max(1, true)) }, 1);

            sa = convsa->forward(sa);
            sa = sa.sigmoid();

            x = x.mul(sa);
        }

        x += residual;
        x = torch::leaky_relu(x, 0.2);

        return x;
    }
};

struct UNet2DImpl : MultiGPUModule
{
    int64_t _depth_block;
    int64_t _width_block;
    int64_t _input_channels;
    int64_t _final_channels;
    int64_t _final_kernel;

    torch::nn::Conv2d convprefilter{ nullptr };

    torch::nn::Sequential encoder1{ nullptr };
    torch::nn::Sequential encoder2{ nullptr };
    torch::nn::Sequential encoder3{ nullptr };
    torch::nn::Sequential encoder4{ nullptr };
    torch::nn::Sequential encoder5{ nullptr };
    torch::nn::Sequential encoder6{ nullptr };

    torch::nn::Sequential decoder5{ nullptr };
    torch::nn::Sequential decoder4{ nullptr };
    torch::nn::Sequential decoder3{ nullptr };
    torch::nn::Sequential decoder2{ nullptr };
    torch::nn::Sequential decoder1{ nullptr };
    torch::nn::Sequential decoder0{ nullptr };

    torch::nn::Upsample upsampler6{ nullptr };
    torch::nn::Upsample upsampler5{ nullptr };
    torch::nn::Upsample upsampler4{ nullptr };
    torch::nn::Upsample upsampler3{ nullptr };
    torch::nn::Upsample upsampler2{ nullptr };
    torch::nn::Upsample upsampler1{ nullptr };

    torch::nn::Sequential final_conv;

    UNet2DImpl(int64_t depth_block, int64_t width_block, int64_t input_channels, int64_t final_channels, int64_t final_kernel, bool dochannelattn, bool dospatialattn)
    {
        _depth_block = depth_block;
        _width_block = width_block;
        _input_channels = input_channels;
        _final_channels = final_channels;
        _final_kernel = final_kernel;

        convprefilter = register_module("convprefilter", torch::nn::Conv2d(conv_options2d(_input_channels, 32 * _width_block, 5, 1, 2)));

        encoder1 = register_module("encoder1", make_encoder(32 * _width_block, 32 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        encoder2 = register_module("encoder2", make_encoder(32 * _width_block, 64 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        encoder3 = register_module("encoder3", make_encoder(64 * _width_block, 128 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        encoder4 = register_module("encoder4", make_encoder(128 * _width_block, 256 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        encoder5 = register_module("encoder5", make_encoder(256 * _width_block, 512 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        encoder6 = register_module("encoder6", make_encoder(512 * _width_block, 1024 * _width_block, _depth_block, true, dochannelattn, dospatialattn));

        decoder5 = register_module("decoder5", make_decoder(1536 * _width_block, 512 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        decoder4 = register_module("decoder4", make_decoder(768 * _width_block, 256 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        decoder3 = register_module("decoder3", make_decoder(384 * _width_block, 128 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        decoder2 = register_module("decoder2", make_decoder(192 * _width_block, 64 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        decoder1 = register_module("decoder1", make_decoder(96 * _width_block, 32 * _width_block, _depth_block, true, dochannelattn, dospatialattn));
        decoder0 = register_module("decoder0", make_decoder(64 * _width_block, 32 * _width_block, _depth_block, false, dochannelattn, dospatialattn));

        upsampler6 = register_module("upsampler6", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler5 = register_module("upsampler5", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler4 = register_module("upsampler4", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler3 = register_module("upsampler3", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler2 = register_module("upsampler2", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler1 = register_module("upsampler1", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));

        final_conv = torch::nn::Sequential();
        for (int i = 0; i < depth_block; i++)
            final_conv->push_back(UNet2DResidualBlock(32 * _width_block, false, dochannelattn, dospatialattn));
        final_conv->push_back(torch::nn::Conv2d(conv_options2d(32 * _width_block, _final_channels, _final_kernel, 1, _final_kernel / 2)));
        register_module("final_conv", final_conv);

        // Initializing weights
        /*for (auto m : this->modules(false))
            if (m->name() == "torch::nn::Conv2dImpl")
                for (auto p : m->parameters())
                    torch::nn::init::xavier_normal_(p);
            else if (m->name() == "torch::nn::BatchNorm2dImpl")
                for (auto p : m->named_parameters())
                    if (p.key() == "weight")
                        torch::nn::init::constant_(p.value(), 1);
                    else if (p.key() == "bias")
                        torch::nn::init::constant_(p.value(), 0);*/
    }

    torch::Tensor forward(torch::Tensor input)
    {
        //x.print();
        at::Tensor prefilter = convprefilter->forward(input);                       //   32.256
        //prefilter.print();

        torch::Tensor enc1 = encoder1->forward(prefilter);                          //   32.256 ->   32.128
        //enc1.print();
        torch::Tensor enc2 = encoder2->forward(enc1);                               //   32.128 ->   64. 64
        //enc2.print();
        torch::Tensor enc3 = encoder3->forward(enc2);                               //   64. 64 ->  128. 32
        //enc3.print();
        torch::Tensor enc4 = encoder4->forward(enc3);                               //  128. 32 ->  256. 16
        //enc4.print();
        torch::Tensor enc5 = encoder5->forward(enc4);                               //  256. 16 ->  512.  8
        //enc5.print();
        torch::Tensor enc6 = encoder6->forward(enc5);                               //  512.  8 -> 1024.  4
        //enc6.print();

        torch::Tensor up6 = upsampler6->forward(enc6);                              // 1024.  4 -> 1024.  8
        //up6.print();
        torch::Tensor dec5 = decoder5->forward(torch::cat({ up6, enc5 }, 1));       // 1536.  8 ->  512.  8
        //dec5.print();

        torch::Tensor up5 = upsampler5->forward(dec5);                              //  512.  8 ->  512. 16
        //up5.print();
        torch::Tensor dec4 = decoder4->forward(torch::cat({ up5, enc4 }, 1));       //  768. 16 ->  256. 16
        //dec4.print();

        torch::Tensor up4 = upsampler4->forward(dec4);                              //  256. 16 ->  256. 32
        torch::Tensor dec3 = decoder3->forward(torch::cat({ up4, enc3 }, 1));       //  384. 32 ->  128. 32
        //dec3.print();

        torch::Tensor up3 = upsampler3->forward(dec3);                              //  128. 32 ->  128. 64
        torch::Tensor dec2 = decoder2->forward(torch::cat({ up3, enc2 }, 1));       //  192. 64 ->   64. 64
        //dec2.print();

        torch::Tensor up2 = upsampler2->forward(dec2);                              //   64. 64 ->   64.128
        torch::Tensor dec1 = decoder1->forward(torch::cat({ up2, enc1 }, 1));       //   96.128 ->   32.128
        //dec1.print();

        torch::Tensor up1 = upsampler1->forward(dec1);                              //   32.128 ->   32.256
        torch::Tensor dec0 = decoder0->forward(torch::cat({ up1, prefilter }, 1));  //   33.256 ->   16.256
        //dec0.print();

        torch::Tensor result = final_conv->forward(dec0);

        return result;
    }


private:
    torch::nn::Sequential make_encoder(int64_t inchannels, int64_t outchannels, int64_t blocks, bool donorm, bool dochannelattn, bool dospatialattn)
    {
        torch::nn::Sequential layers;

        layers->push_back(torch::nn::Conv2d(conv_options2d(inchannels, outchannels, 1, 1, 0)));
        //layers->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, outchannels)));
        //layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

        for (int64_t i = 0; i < blocks; i++)
            layers->push_back(UNet2DResidualBlock(outchannels, donorm, dochannelattn, dospatialattn));

        layers->push_back(torch::nn::Conv2d(conv_options2d(outchannels, outchannels, 2, 2, 0)));
        //layers->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, outchannels)));

        return layers;
    }

    torch::nn::Sequential make_decoder(int64_t inchannels, int64_t outchannels, int64_t blocks, bool donorm, bool dochannelattn, bool dospatialattn)
    {
        torch::nn::Sequential layers;

        layers->push_back(torch::nn::Conv2d(conv_options2d(inchannels, outchannels, 1, 1, 0)));
        //if (donorm)
        //    layers->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(ngroups, outchannels)));

        for (int64_t i = 0; i < blocks; i++)
            layers->push_back(UNet2DResidualBlock(outchannels, donorm, dochannelattn, dospatialattn));

        return layers;
    }
};

TORCH_MODULE(UNet2D);


NNModule THSNN_UNet2D_ctor(const int64_t depth_block, const int64_t width_block, const int64_t input_channels, const int64_t final_channels, const int64_t final_kernel, const bool dochannelattn, const bool dospatialattn, NNAnyModule* outAsAnyModule)
{
    at::globalContext().setBenchmarkCuDNN(true);

    //CATCH_RETURN_NNModule
    //(
    UNet2DImpl Net(depth_block, width_block, input_channels, final_channels, final_kernel, dochannelattn, dospatialattn);
    auto mod = std::make_shared<UNet2DImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<UNet2DImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_UNet2D_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<UNet2DImpl>()->forward(*input));
}