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

struct ParticleWGANResidualBlock : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;

    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::BatchNorm2d bn2{ nullptr };

    ParticleWGANResidualBlock(int64_t channels, bool donorm = true)
    {
        _channels = channels;
        _donorm = donorm;

        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(_channels)));

        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
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

struct ParticleWGANResidualBlockInstanceNorm : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;

    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::InstanceNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::InstanceNorm2d bn2{ nullptr };

    ParticleWGANResidualBlockInstanceNorm(int64_t channels, bool donorm = true)
    {
        _channels = channels;
        _donorm = donorm;

        int ngroups = 32;
        while (ngroups > 1 && _channels % ngroups != 0)
            ngroups /= 2;

        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn1 = register_module("bn1", torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(_channels).affine(false)));

        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn2 = register_module("bn2", torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(_channels).affine(false)));
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

struct ParticleWGANGeneratorImpl : MultiGPUModule
{
    torch::nn::Sequential ProcessorImage;

    torch::nn::Sequential ParticleDecoder;

    torch::nn::Sequential CrapDecoder0;
    torch::nn::Sequential CrapDecoder;

    torch::nn::Sequential ProcessorAdd;
    torch::nn::Sequential ProcessorMul;

    int64_t _codelength;

    int64_t _boxsize;
    int64_t _smallestsize;
    int64_t _smallestchannels;

    ParticleWGANGeneratorImpl(int64_t boxsize, int64_t codelength)
    {
        _boxsize = boxsize;
        _codelength = codelength;

        int currentchannels = 4;
        int currentsize = _boxsize;
        while (currentsize / 2 >= 4)
        {
            currentchannels *= 2;
            currentsize /= 2;
        }

        _smallestchannels = currentchannels;
        _smallestsize = currentsize;

        // Particle decoder
        {
            ParticleDecoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(_codelength, _codelength * 2)));
            ParticleDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            ParticleDecoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(_codelength * 2, _codelength * 4)));
            ParticleDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            ParticleDecoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(_codelength * 4, _boxsize * _boxsize).bias(true)));

            //ParticleDecoder->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(codelength, currentchannels, currentsize).stride(1)));
            //ParticleDecoder->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
            //ParticleDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            //ParticleDecoder->push_back(ParticleWGANResidualBlock(currentchannels, true));

            //while (currentsize * 2 <= boxsize)
            //{
            //    currentchannels /= 2;
            //    currentsize *= 2;

            //    //decoder_conv->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));

            //    //decoder_conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels * 2, currentchannels, 3).padding(1)));
            //    ParticleDecoder->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(currentchannels * 2, currentchannels, 5).stride(2).padding(2).output_padding(1)));
            //    ParticleDecoder->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
            //    ParticleDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            //    //decoder_conv->push_back(C2DNetResidualBlock(currentchannels, true, false));
            //}

            //ParticleDecoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels, 1, 1).padding(0)));

            register_module("particle_decoder", ParticleDecoder);
        }

        // Crapcoder
        {
            //CrapDecoder0->push_back(torch::nn::Linear(torch::nn::LinearOptions(_codelength, 64).bias(true)));
            //CrapDecoder0->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            //CrapDecoder0->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 128).bias(true)));
            //CrapDecoder0->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            //CrapDecoder0->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 64 * 32 * 32).bias(true)));

            //CrapDecoder->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
            //CrapDecoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));                        // 32

            //CrapDecoder->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
            //CrapDecoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 3).padding(1)));
            //CrapDecoder->push_back(ParticleWGANResidualBlock(32, true));                                                            // 64

            //CrapDecoder->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));
            //CrapDecoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 3).padding(1)));
            //CrapDecoder->push_back(ParticleWGANResidualBlock(16, true));                                                            // 128

            //CrapDecoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 1, 5).padding(2)));

            //register_module("crap_decoder0", CrapDecoder0);
            //register_module("crap_decoder", CrapDecoder);
        }

        // Image
        /*{
            ProcessorImage->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 1).bias(true)));

            register_module("processor_image", ProcessorImage);
        }*/

        // Additive noise
        {
            ProcessorAdd->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
            ProcessorAdd->push_back(ParticleWGANResidualBlock(32, true));
            ProcessorAdd->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorAdd->push_back(ParticleWGANResidualBlock(32, true));
            //ProcessorAdd->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            //ProcessorAdd->push_back(ParticleWGANResidualBlock(32, true));
            ProcessorAdd->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1, 5).padding(2)));

            register_module("processor_add", ProcessorAdd);
        }

        // Multiplicative noise
        {
            ProcessorMul->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
            ProcessorMul->push_back(ParticleWGANResidualBlock(32, true));
            ProcessorMul->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorMul->push_back(ParticleWGANResidualBlock(32, true));
            //ProcessorMul->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            //ProcessorMul->push_back(ParticleWGANResidualBlock(32, true));
            ProcessorMul->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1, 5).padding(2)));

            register_module("processor_mul", ProcessorMul);
        }
    }

    torch::Tensor forward_particle(torch::Tensor code, bool transform, double sigmashift)
    {
        //torch::Tensor result = ParticleDecoder->forward(code.reshape({ -1, _codelength, 1, 1 }));
        torch::Tensor result = ParticleDecoder->forward(code).reshape({ -1, 1, _boxsize, _boxsize });

        if (transform)
        {
            torch::Tensor angles = torch::rand({ code.size(0), 1, 1 }, code.options()) * (M_PI * 2);
            torch::Tensor shifts = torch::randn({ code.size(0), 2, 1 }, code.options()) * sigmashift;

            torch::Tensor dirx = angles.cos();
            torch::Tensor diry = angles.sin();

            torch::Tensor rot_mat1 = torch::cat({ dirx, -diry }, 2);
            torch::Tensor rot_mat2 = torch::cat({ diry,  dirx }, 2);

            torch::Tensor trans_matrices = torch::cat({ rot_mat1, rot_mat2 }, 1);
            trans_matrices = torch::cat({ trans_matrices, shifts }, 2);
            torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { result.size(0), result.size(1), result.size(2), result.size(3) });

            result = torch::nn::functional::grid_sample(result, trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kReflection));
        }

        return result;
    }

    torch::Tensor forward(torch::Tensor crapcode, torch::Tensor fakeimages, torch::Tensor ctf)
    {
        ///*torch::Tensor crap = CrapDecoder0->forward(crapcode);
        //crap = crap.view({ -1, 64, 32, 32 });
        //crap = CrapDecoder->forward(crap);*/

        torch::Tensor noise_add = torch::randn(fakeimages.sizes(), fakeimages.options());
        //noise_add = noise_add.add(crap);
        noise_add = ProcessorAdd->forward(noise_add);
        noise_add = torch::fft::rfftn(noise_add, c10::nullopt, std::vector<int64_t>({ 2, 3 }), "forward");
        noise_add = noise_add.mul(ctf);
        noise_add = torch::fft::irfftn(noise_add, c10::nullopt, std::vector<int64_t>({ 2, 3 }), "forward");

        torch::Tensor noise_mul = torch::randn(fakeimages.sizes(), fakeimages.options());
        noise_mul = ProcessorMul->forward(noise_mul);

        torch::Tensor allnoise = noise_add + noise_mul;
        torch::Tensor noisestd = torch::std(allnoise.flatten(1, 3), 1, true, true).unsqueeze(2).unsqueeze(3);
        torch::Tensor noisemean = torch::mean(allnoise.flatten(1, 3), 1, true).unsqueeze(2).unsqueeze(3);
        allnoise = (allnoise - noisemean) / (noisestd + 1e-4f);

        //fakeimages = ProcessorImage->forward(fakeimages);
        fakeimages = fakeimages.add(allnoise);

        return fakeimages;
    }
};

TORCH_MODULE(ParticleWGANGenerator);

struct ParticleWGANDiscriminatorImpl : MultiGPUModule
{
    torch::nn::Sequential Discriminator;
    torch::nn::Sequential DiscriminatorSpectral;

    torch::nn::Sequential DiscriminatorPooled;

    ParticleWGANDiscriminatorImpl()
    {
        // Spatial
        {
            Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 7).stride(1).padding(3)));              // 256
            Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(64).affine(false)));
            Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            //Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));            // 128
            Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(128).affine(false)));
            Discriminator->push_back(ParticleWGANResidualBlockInstanceNorm(128, true));

            //Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));           // 64
            Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(256).affine(false)));
            Discriminator->push_back(ParticleWGANResidualBlockInstanceNorm(256, true));

            //Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1)));          // 32
            Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(512).affine(false)));
            Discriminator->push_back(ParticleWGANResidualBlockInstanceNorm(512, true));

            //Discriminator->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(2).padding(1)));          // 16
            //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(1024).affine(false)));
            Discriminator->push_back(ParticleWGANResidualBlockInstanceNorm(1024, false));

            Discriminator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 2048, 4).stride(1).padding(0)));          // 16
            //Discriminator->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(1024).affine(false)));

            //Discriminator->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })));

            Discriminator->push_back(torch::nn::Flatten());

            Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(2048 * 1 * 1, 256)));
            Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));
            Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, 64)));
            Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));
            //Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 1)));
            //Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));

            DiscriminatorPooled->push_back(torch::nn::Linear(torch::nn::LinearOptions(64 * 1, 64)));
            DiscriminatorPooled->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));
            DiscriminatorPooled->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 64)));
            DiscriminatorPooled->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.0)));
            DiscriminatorPooled->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 1)));

            register_module("discriminator", Discriminator);
            register_module("discriminator_pooled", DiscriminatorPooled);
        }

        // Spectral
        {
            //DiscriminatorSpectral->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 5).stride(1).padding(2)));              // 256
            //DiscriminatorSpectral->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(64).affine(false)));
            //DiscriminatorSpectral->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            //DiscriminatorSpectral->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)));            // 128
            //DiscriminatorSpectral->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(64).affine(false)));
            //DiscriminatorSpectral->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            //DiscriminatorSpectral->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));           // 64
            //DiscriminatorSpectral->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(128).affine(false)));
            //DiscriminatorSpectral->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            //DiscriminatorSpectral->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));          // 32
            //DiscriminatorSpectral->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(256).affine(false)));
            //DiscriminatorSpectral->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            //DiscriminatorSpectral->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1)));          // 16
            //DiscriminatorSpectral->push_back(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(512).affine(false)));
            //DiscriminatorSpectral->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            //DiscriminatorSpectral->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })));

            //DiscriminatorSpectral->push_back(torch::nn::Flatten());
            //DiscriminatorSpectral->push_back(torch::nn::Linear(torch::nn::LinearOptions(512, 128).bias(true)));
            //DiscriminatorSpectral->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            //DiscriminatorSpectral->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 32).bias(true)));
            //DiscriminatorSpectral->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            //DiscriminatorSpectral->push_back(torch::nn::Linear(torch::nn::LinearOptions(32, 1)));

            //register_module("discriminator_spectral", DiscriminatorSpectral);
        }
    }

    torch::Tensor forward(torch::Tensor image)
    {
        /*torch::Tensor imageft = torch::fft::rfftn(image, c10::nullopt, std::vector<int64_t>({ 2, 3 }), "forward");
        imageft = imageft.slice(3, 0, imageft.size(3) - 1, 1);
        imageft = imageft.abs();
        imageft = imageft.roll(imageft.size(2) / 2, 2);
        torch::Tensor decision_spectral = DiscriminatorSpectral->forward(imageft);*/

        /*image = image.slice(3, 8, image.size(3) - 8, 1);
        image = image.slice(2, 8, image.size(2) - 8, 1);

        image = torch::nn::functional::pad(image, torch::nn::functional::PadFuncOptions(std::vector<int64_t>({ 8, 8, 8, 8 })).mode(torch::kConstant).value(0));*/

        std::vector<torch::Tensor> decisions;
        decisions.push_back(Discriminator->forward(image)/*.unsqueeze(1)*/);
        //decisions.push_back(Discriminator->forward(image.rot90(1, { 2, 3 }))/*.unsqueeze(1)*/);
        //decisions.push_back(Discriminator->forward(image.rot90(2, { 2, 3 }))/*.unsqueeze(1)*/);
        //decisions.push_back(Discriminator->forward(image.rot90(3, { 2, 3 }))/*.unsqueeze(1)*/);

        torch::Tensor decision_spatial = DiscriminatorPooled->forward(/*std::get<0>(*/torch::cat(decisions, 1)/*.max(1))*/);

        //x = x.squeeze(3);
        //x = x.squeeze(2);

        return /*decision_spectral +*/ decision_spatial;
    }

    void ClipWeights(double min, double max)
    {
        auto this_params = this->parameters(true);

        for (int64_t i = 0; i < this_params.size(); i++)
            this_params[i].data().clamp_(min, max);
    }

    torch::Tensor PenalizeGradient(torch::Tensor real, torch::Tensor fake, float lambda)
    {
        //fake = fake.to(real.device());

        torch::Tensor eta = torch::rand({ real.size(0), 1, 1, 1 }, real.options());
        eta = eta.expand(real.sizes());
        //eta.print();

        torch::Tensor interp = eta * real + ((1.0f - eta) * fake);
        interp = interp.detach();
        interp.set_requires_grad(true);

        torch::Tensor pred_interp = forward(interp);
        //pred_interp.print();

        torch::Tensor gradients = torch::autograd::grad({ pred_interp }, { interp }, { torch::ones(pred_interp.sizes(), real.options()) }, true, true)[0];
        //gradients.print();
        gradients = gradients.view({ gradients.size(0), -1 });
        //gradients.print();

        //torch::Tensor norm = gradients.norm(2, 1);
        //norm.print();
        torch::Tensor grad_penalty = (gradients.norm(2, 1) - 1).square().mean() * lambda;
        //grad_penalty.print();

        return grad_penalty;
    }
};

TORCH_MODULE(ParticleWGANDiscriminator);


NNModule THSNN_ParticleWGANGenerator_ctor(int64_t boxsize, int64_t codelength, NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    ParticleWGANGeneratorImpl Net(boxsize, codelength);
    auto mod = std::make_shared<ParticleWGANGeneratorImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<ParticleWGANGeneratorImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_ParticleWGANGenerator_forward_particle(const NNModule module, const Tensor code, const bool transform, const double sigmashift)
{
    CATCH_TENSOR((*module)->as<ParticleWGANGeneratorImpl>()->forward_particle(*code, transform, sigmashift));
}

Tensor THSNN_ParticleWGANGenerator_forward_noise(const NNModule module, const Tensor crapcode, const Tensor fakeimages, const Tensor ctf)
{
    CATCH_TENSOR((*module)->as<ParticleWGANGeneratorImpl>()->forward(*crapcode, *fakeimages, *ctf));
}


NNModule THSNN_ParticleWGANDiscriminator_ctor(NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    ParticleWGANDiscriminatorImpl Net;
    auto mod = std::make_shared<ParticleWGANDiscriminatorImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<ParticleWGANDiscriminatorImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_ParticleWGANDiscriminator_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<ParticleWGANDiscriminatorImpl>()->forward(*input));
}

void THSNN_ParticleWGANDiscriminator_clipweights(const NNModule module, const double clip)
{
    (*module)->as<ParticleWGANDiscriminatorImpl>()->ClipWeights(-clip, clip);
}

Tensor THSNN_ParticleWGANDiscriminator_penalizegradient(const NNModule module, const Tensor real, const Tensor fake, const float lambda)
{
    CATCH_TENSOR((*module)->as<ParticleWGANDiscriminatorImpl>()->PenalizeGradient(*real, *fake, lambda));
}