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

#include "MultiGPUModule.h"

torch::nn::Conv2dOptions BoxNetMulti_conv_options2d(int64_t in_planes,
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

struct BoxNetMultiResidualBlock : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;

    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::BatchNorm2d bn2{ nullptr };

	BoxNetMultiResidualBlock(int64_t channels, bool donorm = true)
    {
        _channels = channels;
        _donorm = donorm;

        int ngroups = 32;
        while (ngroups > 1 && _channels % ngroups != 0)
            ngroups /= 2;

        conv1 = register_module("conv1", torch::nn::Conv2d(BoxNetMulti_conv_options2d(_channels, _channels, 3, 1, 1).groups(std::min((int64_t)1, channels / 32))));
        if (donorm)
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(_channels)));

        conv2 = register_module("conv2", torch::nn::Conv2d(BoxNetMulti_conv_options2d(_channels, _channels, 3, 1, 1).groups(std::min((int64_t)1, channels / 32))));
        if (donorm)
            bn2 = register_module("bn2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(_channels)));
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

        x += residual;
        x = torch::leaky_relu(x, 0.2);

        return x;
    }
};

struct BoxNetMultiImpl : MultiGPUModule
{
    int64_t _depth_block;
    int64_t _width_block;
    int64_t _input_channels;

    torch::nn::Conv2d convprefilter{ nullptr };

    torch::nn::Sequential encoder1{ nullptr };
    torch::nn::Sequential encoder2{ nullptr };
    torch::nn::Sequential encoder3{ nullptr };
    torch::nn::Sequential encoder4{ nullptr };
    torch::nn::Sequential encoder5{ nullptr };
    torch::nn::Sequential encoder6{ nullptr };

    torch::nn::Sequential pick_decoder5{ nullptr };
    torch::nn::Sequential pick_decoder4{ nullptr };
    torch::nn::Sequential pick_decoder3{ nullptr };
    torch::nn::Sequential pick_decoder2{ nullptr };
    torch::nn::Sequential pick_decoder1{ nullptr };
    torch::nn::Sequential pick_decoder0{ nullptr };

	torch::nn::Sequential denoise_decoder5{ nullptr };
	torch::nn::Sequential denoise_decoder4{ nullptr };
	torch::nn::Sequential denoise_decoder3{ nullptr };
	torch::nn::Sequential denoise_decoder2{ nullptr };
	torch::nn::Sequential denoise_decoder1{ nullptr };
	torch::nn::Sequential denoise_decoder0{ nullptr };

    torch::nn::Upsample upsampler6{ nullptr };
    torch::nn::Upsample upsampler5{ nullptr };
    torch::nn::Upsample upsampler4{ nullptr };
    torch::nn::Upsample upsampler3{ nullptr };
    torch::nn::Upsample upsampler2{ nullptr };
    torch::nn::Upsample upsampler1{ nullptr };

    torch::nn::Sequential pick_final_conv;
	torch::nn::Sequential denoise_final_conv;

	BoxNetMultiImpl(int64_t depth_block, int64_t width_block, int64_t input_channels)
    {
        _depth_block = depth_block;
        _width_block = width_block;
        _input_channels = input_channels;

        convprefilter = register_module("convprefilter", torch::nn::Conv2d(BoxNetMulti_conv_options2d(_input_channels, 32 * _width_block, 5, 1, 2)));

        encoder1 = register_module("encoder1", make_encoder(32 * _width_block, 32 * _width_block, _depth_block, true));
        encoder2 = register_module("encoder2", make_encoder(32 * _width_block, 64 * _width_block, _depth_block, true));
        encoder3 = register_module("encoder3", make_encoder(64 * _width_block, 128 * _width_block, _depth_block, true));
        encoder4 = register_module("encoder4", make_encoder(128 * _width_block, 256 * _width_block, _depth_block, true));
        encoder5 = register_module("encoder5", make_encoder(256 * _width_block, 512 * _width_block, _depth_block, true));
        encoder6 = register_module("encoder6", make_encoder(512 * _width_block, 1024 * _width_block, _depth_block, true));

        pick_decoder5 = register_module("decoder5", make_decoder(1536 * _width_block, 512 * _width_block, _depth_block, true));
        pick_decoder4 = register_module("decoder4", make_decoder(768 * _width_block, 256 * _width_block, _depth_block, true));
        pick_decoder3 = register_module("decoder3", make_decoder(384 * _width_block, 128 * _width_block, _depth_block, true));
        pick_decoder2 = register_module("decoder2", make_decoder(192 * _width_block, 64 * _width_block, _depth_block, true));
        pick_decoder1 = register_module("decoder1", make_decoder(96 * _width_block, 32 * _width_block, _depth_block, true));
        pick_decoder0 = register_module("decoder0", make_decoder(64 * _width_block, 32 * _width_block, _depth_block, false));

		denoise_decoder5 = register_module("denoise_decoder5", make_decoder(1536 * _width_block, 512 * _width_block, _depth_block, true));
		denoise_decoder4 = register_module("denoise_decoder4", make_decoder(768 * _width_block, 256 * _width_block, _depth_block, true));
		denoise_decoder3 = register_module("denoise_decoder3", make_decoder(384 * _width_block, 128 * _width_block, _depth_block, true));
		denoise_decoder2 = register_module("denoise_decoder2", make_decoder(192 * _width_block, 64 * _width_block, _depth_block, true));
		denoise_decoder1 = register_module("denoise_decoder1", make_decoder(96 * _width_block, 32 * _width_block, _depth_block, true));
		denoise_decoder0 = register_module("denoise_decoder0", make_decoder(64 * _width_block, 32 * _width_block, _depth_block, false));

        upsampler6 = register_module("upsampler6", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler5 = register_module("upsampler5", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler4 = register_module("upsampler4", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler3 = register_module("upsampler3", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler2 = register_module("upsampler2", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
        upsampler1 = register_module("upsampler1", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));

        pick_final_conv = torch::nn::Sequential();
        for (int i = 0; i < depth_block; i++)
            pick_final_conv->push_back(BoxNetMultiResidualBlock(32 * _width_block, false));
        pick_final_conv->push_back(torch::nn::Conv2d(BoxNetMulti_conv_options2d(32 * _width_block, 3, 1, 1, 0)));
        register_module("final_conv", pick_final_conv);

		denoise_final_conv = torch::nn::Sequential();
		for (int i = 0; i < depth_block; i++)
			denoise_final_conv->push_back(BoxNetMultiResidualBlock(32 * _width_block, false));
		denoise_final_conv->push_back(torch::nn::Conv2d(BoxNetMulti_conv_options2d(32 * _width_block, 1, 1, 1, 0)));
		register_module("denoise_final_conv", denoise_final_conv);

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
		return input;
	}

    torch::Tensor pick_forward(torch::Tensor input)
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
        torch::Tensor dec5 = pick_decoder5->forward(torch::cat({ up6, enc5 }, 1));       // 1536.  8 ->  512.  8
        //dec5.print();

        torch::Tensor up5 = upsampler5->forward(dec5);                              //  512.  8 ->  512. 16
        //up5.print();
        torch::Tensor dec4 = pick_decoder4->forward(torch::cat({ up5, enc4 }, 1));       //  768. 16 ->  256. 16
        //dec4.print();

        torch::Tensor up4 = upsampler4->forward(dec4);                              //  256. 16 ->  256. 32
        torch::Tensor dec3 = pick_decoder3->forward(torch::cat({ up4, enc3 }, 1));       //  384. 32 ->  128. 32
        //dec3.print();

        torch::Tensor up3 = upsampler3->forward(dec3);                              //  128. 32 ->  128. 64
        torch::Tensor dec2 = pick_decoder2->forward(torch::cat({ up3, enc2 }, 1));       //  192. 64 ->   64. 64
        //dec2.print();

        torch::Tensor up2 = upsampler2->forward(dec2);                              //   64. 64 ->   64.128
        torch::Tensor dec1 = pick_decoder1->forward(torch::cat({ up2, enc1 }, 1));       //   96.128 ->   32.128
        //dec1.print();

        torch::Tensor up1 = upsampler1->forward(dec1);                              //   32.128 ->   32.256
        torch::Tensor dec0 = pick_decoder0->forward(torch::cat({ up1, prefilter }, 1));  //   33.256 ->   16.256
        //dec0.print();

        torch::Tensor result = pick_final_conv->forward(dec0);

        return result;
    }

	torch::Tensor denoise_forward(torch::Tensor input)
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
		torch::Tensor dec5 = denoise_decoder5->forward(torch::cat({ up6, enc5 }, 1));       // 1536.  8 ->  512.  8
		//dec5.print();

		torch::Tensor up5 = upsampler5->forward(dec5);                              //  512.  8 ->  512. 16
		//up5.print();
		torch::Tensor dec4 = denoise_decoder4->forward(torch::cat({ up5, enc4 }, 1));       //  768. 16 ->  256. 16
		//dec4.print();

		torch::Tensor up4 = upsampler4->forward(dec4);                              //  256. 16 ->  256. 32
		torch::Tensor dec3 = denoise_decoder3->forward(torch::cat({ up4, enc3 }, 1));       //  384. 32 ->  128. 32
		//dec3.print();

		torch::Tensor up3 = upsampler3->forward(dec3);                              //  128. 32 ->  128. 64
		torch::Tensor dec2 = denoise_decoder2->forward(torch::cat({ up3, enc2 }, 1));       //  192. 64 ->   64. 64
		//dec2.print();

		torch::Tensor up2 = upsampler2->forward(dec2);                              //   64. 64 ->   64.128
		torch::Tensor dec1 = denoise_decoder1->forward(torch::cat({ up2, enc1 }, 1));       //   96.128 ->   32.128
		//dec1.print();

		torch::Tensor up1 = upsampler1->forward(dec1);                              //   32.128 ->   32.256
		torch::Tensor dec0 = denoise_decoder0->forward(torch::cat({ up1, prefilter }, 1));  //   33.256 ->   16.256
		//dec0.print();

		torch::Tensor result = denoise_final_conv->forward(dec0);

		return result;
	}


private:
    torch::nn::Sequential make_encoder(int64_t inchannels, int64_t outchannels, int64_t blocks, bool donorm)
    {
        torch::nn::Sequential layers;

        layers->push_back(torch::nn::Conv2d(BoxNetMulti_conv_options2d(inchannels, outchannels, 1, 1, 0)));
        //layers->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, outchannels)));
        //layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

        for (int64_t i = 0; i < blocks; i++)
            layers->push_back(BoxNetMultiResidualBlock(outchannels, donorm));

        layers->push_back(torch::nn::Conv2d(BoxNetMulti_conv_options2d(outchannels, outchannels, 2, 2, 0)));
        //layers->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, outchannels)));

        return layers;
    }

    torch::nn::Sequential make_decoder(int64_t inchannels, int64_t outchannels, int64_t blocks, bool donorm)
    {
        torch::nn::Sequential layers;

        layers->push_back(torch::nn::Conv2d(BoxNetMulti_conv_options2d(inchannels, outchannels, 1, 1, 0)));
        //if (donorm)
        //    layers->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(ngroups, outchannels)));

        for (int64_t i = 0; i < blocks; i++)
            layers->push_back(BoxNetMultiResidualBlock(outchannels, donorm));

        return layers;
    }
};

TORCH_MODULE(BoxNetMulti);


NNModule THSNN_BoxNetMulti_ctor(const int64_t depth_block, const int64_t width_block, const int64_t input_channels, NNAnyModule* outAsAnyModule)
{
    at::globalContext().setBenchmarkCuDNN(true);

    //CATCH_RETURN_NNModule
    //(
    BoxNetMultiImpl Net(depth_block, width_block, input_channels);
    auto mod = std::make_shared<BoxNetMultiImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<BoxNetMultiImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_BoxNetMulti_pick_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<BoxNetMultiImpl>()->pick_forward(*input));
}

Tensor THSNN_BoxNetMulti_denoise_forward(const NNModule module, const Tensor input)
{
	CATCH_TENSOR((*module)->as<BoxNetMultiImpl>()->denoise_forward(*input));
}