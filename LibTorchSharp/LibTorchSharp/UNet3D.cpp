#pragma once

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

#include "ATen/autocast_mode.h"

#include "MultiGPUModule.h"

namespace F = torch::nn::functional;

F::PadFuncOptions::mode_t _get_pad_mode_from_conv_padding_mode(torch::nn::detail::conv_padding_mode_t conv_padding_mode) {
    F::PadFuncOptions::mode_t pad_mode;
    if (c10::get_if<torch::enumtype::kReflect>(&conv_padding_mode)) {
        pad_mode = torch::kReflect;
    }
    else if (c10::get_if<torch::enumtype::kReplicate>(&conv_padding_mode)) {
        pad_mode = torch::kReplicate;
    }
    else if (c10::get_if<torch::enumtype::kCircular>(&conv_padding_mode)) {
        pad_mode = torch::kCircular;
    }
    else {
        TORCH_CHECK(false, "Unsupported conv padding mode: ", torch::enumtype::get_enum_name(conv_padding_mode));
    }
    return pad_mode;
}

namespace torch
{
    namespace nn
    {
        struct Conv3dStdImpl : torch::nn::Conv3dImpl
        {
        public:
            Conv3dStdImpl(int64_t input_channels,
                int64_t output_channels,
                ExpandingArray<3> kernel_size)
                : Conv3dImpl(Conv3dOptions(input_channels, output_channels, kernel_size))
            {
            }

            Conv3dStdImpl(Conv3dOptions options_) : Conv3dImpl(options_)
            {
            }

            Tensor forward(const Tensor& input)
            {
                auto wshape = weight.sizes();
                int cout = wshape[0];
                Tensor wflat = weight.view(std::vector<int64_t>({ cout, -1 }));
                //wflat.print();
                Tensor wmean = wflat.mean(1, true);
                //Tensor wstd = wflat.var(1, true);
                wflat = (wflat - wmean);// / (wstd.sqrt() + 1e-5);
                //wmean.print();
                wflat = wflat.view(wshape);

                if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
                    return F::detail::conv3d(
                        F::pad(input, F::PadFuncOptions(_reversed_padding_repeated_twice).mode(_get_pad_mode_from_conv_padding_mode(options.padding_mode()))),
                        wflat, bias,
                        options.stride(),
                        /*padding=*/0,
                        options.dilation(),
                        options.groups());
                }
                return F::detail::conv3d(
                    input,
                    wflat,
                    bias,
                    options.stride(),
                    options.padding(),
                    options.dilation(),
                    options.groups());
            }
        };

        TORCH_MODULE(Conv3dStd);
    }
}

torch::nn::Conv3dOptions conv_options(int64_t in_planes,
    int64_t out_planes,
    int64_t kernel_size = 3,
    int64_t stride = 1,
    int64_t padding = 1,
    int64_t dilation = 1,
    bool with_bias = false)
{
    torch::nn::Conv3dOptions conv_options = torch::nn::Conv3dOptions(in_planes, out_planes, kernel_size)
        .stride(stride)
        .padding(padding)
        .bias(with_bias)
        .dilation(dilation);

    return conv_options;
}

struct UNet3DResidualBlock : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;
    bool _usegroupnorm;

    torch::nn::Conv3d conv1{ nullptr };
    torch::nn::Conv3d conv2{ nullptr };
    torch::nn::Conv3d resconv{ nullptr };

    torch::nn::BatchNorm3d bn1{ nullptr };
    torch::nn::BatchNorm3d bn2{ nullptr };
    torch::nn::GroupNorm gn1{ nullptr };
    torch::nn::GroupNorm gn2{ nullptr };

    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };

    torch::nn::Conv3d convsa{ nullptr };

    UNet3DResidualBlock(int64_t channels, bool donorm, bool usegroupnorm)
    {
        _channels = channels;
        _donorm = donorm;
        _usegroupnorm = usegroupnorm;

        int ngroups = 32;
        while (ngroups > 1 && _channels % ngroups != 0)
            ngroups /= 2;

        conv1 = register_module("conv1", torch::nn::Conv3d(conv_options(_channels, _channels, 3, 1, 1)));
        conv2 = register_module("conv2", torch::nn::Conv3d(conv_options(_channels, _channels, 3, 1, 1)));
        resconv = register_module("resconv", torch::nn::Conv3d(conv_options(_channels, _channels, 1, 1, 0)));

        if (_donorm)
        {
            if (usegroupnorm)
            {
                gn1 = register_module("bn1", torch::nn::GroupNorm(torch::nn::GroupNormOptions(std::max(1, (int)_channels / 8), _channels)));
                gn2 = register_module("bn2", torch::nn::GroupNorm(torch::nn::GroupNormOptions(std::max(1, (int)_channels / 8), _channels)));
            }
            else
            {
                bn1 = register_module("bn1", torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(_channels)));
                bn2 = register_module("bn2", torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(_channels)));
            }
        }

        //fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(_channels * 2, _channels / 8)));
        //fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(_channels / 8, _channels)));

        //convsa = register_module("convsa", torch::nn::Conv3d(conv_options(2, 1, 7, 1, 3)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor residual = resconv->forward(x);

        if (_donorm)
        {
            if (_usegroupnorm)
                x = gn1->forward(x);
            else
                x = bn1->forward(x);
        }
        //x = torch::leaky_relu(x, 0.2);
        x = torch::silu(x);
        x = conv1->forward(x);

        if (_donorm)
        {
            if (_usegroupnorm)
                x = gn2->forward(x);
            else
                x = bn2->forward(x);
        }
        x = torch::silu(x);
        x = conv2->forward(x);

        /*torch::Tensor ca = x.flatten(2, 4);
        ca = torch::cat({ ca.mean(2, false), std::get<0>(ca.max(2, false)) }, 1);

        ca = fc1->forward(ca);
        ca = torch::relu(ca);

        ca = fc2->forward(ca);
        ca = torch::sigmoid(ca);

        ca = ca.unsqueeze(2);
        ca = ca.unsqueeze(3);
        ca = ca.unsqueeze(4);

        x = x.mul(ca);

        if (x.size(1) <= 64)
        {
            torch::Tensor sa = torch::cat({ x.mean(1, true), std::get<0>(x.max(1, true)) }, 1);

            sa = convsa->forward(sa);
            sa = sa.sigmoid();

            x = x.mul(sa);
        }*/

        x += residual;
        //x = torch::leaky_relu(x, 0.2); 

        return x;
    }
};

struct UNet3DImpl : MultiGPUModule
{
    int64_t _depth_block;
    int64_t _width_block;
    int64_t _max_width;
    int64_t _input_channels;
    int64_t _final_channels;
    bool _residual_u;

    torch::nn::Conv3d convprefilter{ nullptr };

    torch::nn::Sequential encoder1{ nullptr };
    torch::nn::Sequential encoder2{ nullptr };
    torch::nn::Sequential encoder3{ nullptr };
    torch::nn::Sequential encoder4{ nullptr };
    torch::nn::Sequential encoder5{ nullptr };

    torch::nn::Sequential bypass1{ nullptr };
    torch::nn::Sequential bypass2{ nullptr };
    torch::nn::Sequential bypass3{ nullptr };
    torch::nn::Sequential bypass4{ nullptr };

    torch::nn::Sequential decoder4{ nullptr };
    torch::nn::Sequential decoder3{ nullptr };
    torch::nn::Sequential decoder2{ nullptr };
    torch::nn::Sequential decoder1{ nullptr };
    torch::nn::Sequential decoder0{ nullptr };

    /*torch::nn::Upsample upsampler5{ nullptr };
    torch::nn::Upsample upsampler4{ nullptr };
    torch::nn::Upsample upsampler3{ nullptr };
    torch::nn::Upsample upsampler2{ nullptr };
    torch::nn::Upsample upsampler1{ nullptr };*/
    torch::nn::ConvTranspose3d upsampler5{ nullptr };
    torch::nn::ConvTranspose3d upsampler4{ nullptr };
    torch::nn::ConvTranspose3d upsampler3{ nullptr };
    torch::nn::ConvTranspose3d upsampler2{ nullptr };
    torch::nn::ConvTranspose3d upsampler1{ nullptr };

    torch::nn::Sequential final_conv;

    UNet3DImpl(int64_t depth_block, int64_t width_block, int64_t max_width, int64_t input_channels, int64_t final_channels, bool residual_u, bool usegroupnorm, bool progressivedepth)
    {
        _depth_block = depth_block;
        _width_block = width_block;
        _max_width = max_width;
        _input_channels = input_channels;
        _final_channels = final_channels;
        _residual_u = residual_u;

        int rmult = _residual_u ? 1 : 0;
        int pd = progressivedepth ? 1 : 0;

        int64_t width0 = 32 * _width_block;
		int64_t width1 = std::min(32 * _width_block, _max_width);
		int64_t width2 = std::min(64 * _width_block, _max_width);
		int64_t width3 = std::min(128 * _width_block, _max_width);
		int64_t width4 = std::min(256 * _width_block, _max_width);
		int64_t width5 = std::min(512 * _width_block, _max_width);

        convprefilter = register_module("convprefilter", torch::nn::Conv3d(conv_options(_input_channels, width0, 5, 1, 2)));

        encoder1 = register_module("encoder1", make_encoder(width0, width1, _depth_block, true, usegroupnorm));
        encoder2 = register_module("encoder2", make_encoder(width1, width2, 1 * pd + _depth_block, true, usegroupnorm));
        encoder3 = register_module("encoder3", make_encoder(width2, width3, 2 * pd + _depth_block, true, usegroupnorm));
        encoder4 = register_module("encoder4", make_encoder(width3, width4, 3 * pd + _depth_block, true, usegroupnorm));
        encoder5 = register_module("encoder5", make_encoder(width4, width5, 4 * pd + _depth_block, true, usegroupnorm));

        /*bypass1 = register_module("bypass1", make_bypass(32 * _width_block, 1, true));
        bypass2 = register_module("bypass2", make_bypass(64 * _width_block, 1, true));
        bypass3 = register_module("bypass3", make_bypass(128 * _width_block, 1, true));
        bypass4 = register_module("bypass4", make_bypass(256 * _width_block, 1, true));*/

        decoder4 = register_module("decoder4", make_decoder(width5 + width4 * rmult, width4, 3 * pd + _depth_block, true, usegroupnorm));
        decoder3 = register_module("decoder3", make_decoder(width4 + width3 * rmult, width3, 2 * pd + _depth_block, true, usegroupnorm));
        decoder2 = register_module("decoder2", make_decoder(width3 + width2 * rmult, width2, 1 * pd + _depth_block, true, usegroupnorm));
        decoder1 = register_module("decoder1", make_decoder(width2 + width1 * rmult, width1, _depth_block, true, usegroupnorm));
        decoder0 = register_module("decoder0", make_decoder(width1 + width0 * rmult, width0, _depth_block, false, false));

        /*upsampler5 = register_module("upsampler5", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2, 2 })).mode(torch::kTrilinear).align_corners(true)));
        upsampler4 = register_module("upsampler4", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2, 2 })).mode(torch::kTrilinear).align_corners(true)));
        upsampler3 = register_module("upsampler3", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2, 2 })).mode(torch::kTrilinear).align_corners(true)));
        upsampler2 = register_module("upsampler2", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2, 2 })).mode(torch::kTrilinear).align_corners(true)));
        upsampler1 = register_module("upsampler1", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2, 2 })).mode(torch::kTrilinear).align_corners(true)));*/
        upsampler5 = register_module("upsampler5", torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(width5, width5, 2).stride(2)));
        upsampler4 = register_module("upsampler4", torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(width4, width4, 2).stride(2)));
        upsampler3 = register_module("upsampler3", torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(width3, width3, 2).stride(2)));
        upsampler2 = register_module("upsampler2", torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(width2, width2, 2).stride(2)));
        upsampler1 = register_module("upsampler1", torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(width1, width1, 2).stride(2)));

        final_conv = torch::nn::Sequential();
        for (int64_t i = 0; i < _depth_block; i++)
            final_conv->push_back(UNet3DResidualBlock(width0, false, false));
        final_conv->push_back(torch::nn::Conv3d(conv_options(width0, _final_channels, 1, 1, 0)));
        register_module("final_conv", final_conv);

        // Initializing weights
        /*for (auto m : this->modules(false))
            if (m->name() == "torch::nn::Conv3dImpl")
                for (auto p : m->named_parameters())
                    if (p.key() == "weight")
                    {
                        p.value().print();
                        //torch::nn::init::xavier_normal_(p);
                    }
            else if (m->name() == "torch::nn::BatchNorm3dImpl")
                for (auto p : m->named_parameters())
                    if (p.key() == "weight")
                        torch::nn::init::constant_(p.value(), 1);
                    else if (p.key() == "bias")
                        torch::nn::init::constant_(p.value(), 0);*/
    }

    torch::Tensor forward(torch::Tensor input)
    {
        //x.print();
        at::Tensor prefilter = convprefilter->forward(input);                       //  32.64
        //prefilter.print();

        torch::Tensor enc1 = encoder1->forward(prefilter);                          //  32.64 ->  32.32
        //enc1.print();
        torch::Tensor enc2 = encoder2->forward(enc1);                               //  32.32 ->  64.16
        //enc2.print();
        torch::Tensor enc3 = encoder3->forward(enc2);                               //  64.16 -> 128. 8
        //enc3.print();
        torch::Tensor enc4 = encoder4->forward(enc3);                               // 128. 8 -> 256. 4
        //enc4.print();
        torch::Tensor enc5 = encoder5->forward(enc4);                               // 256. 4 -> 512. 2
        //enc5.print();

        torch::Tensor up5 = upsampler5->forward(enc5);                              // 512. 2 -> 512. 4
        //up5.print();
        if (_residual_u)
            up5 = torch::cat({ up5, (enc4) }, 1);
        torch::Tensor dec4 = decoder4->forward(up5);                                // 768. 4 -> 256. 4
        //dec4.print();

        torch::Tensor up4 = upsampler4->forward(dec4);                              // 256. 4 -> 256. 8
        if (_residual_u)
            up4 = torch::cat({ up4, (enc3) }, 1);
        torch::Tensor dec3 = decoder3->forward(up4);                                // 384. 8 -> 128. 8
        //dec3.print();

        torch::Tensor up3 = upsampler3->forward(dec3);                              // 128. 8 -> 128.16
        if (_residual_u)
            up3 = torch::cat({ up3, (enc2) }, 1);
        torch::Tensor dec2 = decoder2->forward(up3);                                // 192.16 ->  64.16
        //dec2.print();

        torch::Tensor up2 = upsampler2->forward(dec2);                              //  64.16 ->  64.32
        if (_residual_u)
            up2 = torch::cat({ up2, (enc1) }, 1);
        torch::Tensor dec1 = decoder1->forward(up2);                                //  96.32 ->  32.32
        //dec1.print();

        torch::Tensor up1 = upsampler1->forward(dec1);                              //  32.32 ->  32.64
        if (_residual_u)
            up1 = torch::cat({ up1, (prefilter) }, 1);
        torch::Tensor dec0 = decoder0->forward(up1);  //  64.64 ->  16.64
        //dec0.print();

        torch::Tensor result = final_conv->forward(dec0);

        //result += input;

        return result;
    }

    void standardize_weights()
    {

    }


private:
    torch::nn::Sequential make_encoder(int64_t inchannels, int64_t outchannels, int64_t blocks, bool donorm, bool usegroupnorm)
    {
        torch::nn::Sequential layers;

        if (inchannels != outchannels)
            layers->push_back(torch::nn::Conv3d(conv_options(inchannels, outchannels, 3, 1, 1)));
        //layers->push_back(torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(outchannels)));
        //layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

        for (int64_t i = 0; i < blocks; i++)
            layers->push_back(UNet3DResidualBlock(outchannels, donorm, usegroupnorm));

        layers->push_back(torch::nn::Conv3d(conv_options(outchannels, outchannels, 4, 2, 1)));
        //layers->push_back(torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(outchannels)));

        return layers;
    }

    torch::nn::Sequential make_decoder(int64_t inchannels, int64_t outchannels, int64_t blocks, bool donorm, bool usegroupnorm)
    {
        torch::nn::Sequential layers;

        /*int ngroups = 32;
        while (ngroups > 1 && outchannels % ngroups != 0)
            ngroups /= 2;*/

        if (inchannels != outchannels)
            layers->push_back(torch::nn::Conv3d(conv_options(inchannels, outchannels, 3, 1, 1)));
        //layers->push_back(torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(outchannels)));

        for (int64_t i = 0; i < blocks; i++)
            layers->push_back(UNet3DResidualBlock(outchannels, donorm, usegroupnorm));

        return layers;
    }

    torch::nn::Sequential make_bypass(int64_t inchannels, int64_t blocks, bool donorm, bool usegroupnorm)
    {
        torch::nn::Sequential layers;

        for (int64_t i = 0; i < blocks; i++)
            layers->push_back(UNet3DResidualBlock(inchannels, donorm, usegroupnorm));

        return layers;
    }
};

TORCH_MODULE(UNet3D);


NNModule THSNN_UNet3D_ctor(const int64_t depth_block, const int64_t width_block, const int64_t max_width, const int64_t input_channels, const int64_t final_channels, const bool residual_u, const bool usegroupnorm, const bool progressivedepth, NNAnyModule* outAsAnyModule)
{
    at::globalContext().setBenchmarkCuDNN(true);

    //CATCH_RETURN_NNModule
    //(
    UNet3DImpl Net(depth_block, width_block, max_width, input_channels, final_channels, residual_u, usegroupnorm, progressivedepth);
    auto mod = std::make_shared<UNet3DImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<UNet3DImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_UNet3D_forward(const NNModule module, const Tensor input)
{
    //CATCH_TENSOR((*module)->as<UNet3DImpl>()->forward((* input).toType(torch::kBFloat16)).toType(torch::kFloat));
    //CATCH_TENSOR((*module)->as<UNet3DImpl>()->forward((*input)));
    //at::autocast::set_enabled(true);
    at::Tensor res = (*module)->as<UNet3DImpl>()->forward((*input));
    //at::autocast::clear_cache();
    //at::autocast::set_enabled(false);
    return ResultTensor(res);
}