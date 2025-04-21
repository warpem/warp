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

struct Discriminator3DResidualBlock : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;

    torch::nn::Conv3d conv1{ nullptr };
    torch::nn::InstanceNorm3d bn1{ nullptr };
    torch::nn::Conv3d conv2{ nullptr };
    torch::nn::InstanceNorm3d bn2{ nullptr };

    Discriminator3DResidualBlock(int64_t channels, bool donorm = true)
    {
        _channels = channels;
        _donorm = donorm;

        int ngroups = 32;
        while (ngroups > 1 && _channels % ngroups != 0)
            ngroups /= 2;

        conv1 = register_module("conv1", torch::nn::Conv3d(torch::nn::Conv3dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn1 = register_module("bn1", torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(_channels).affine(false)));

        conv2 = register_module("conv2", torch::nn::Conv3d(torch::nn::Conv3dOptions(_channels, _channels, 3).stride(1).padding(1)));
        bn2 = register_module("bn2", torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(_channels).affine(false)));
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

struct Discriminator3DImpl : MultiGPUModule
{
    torch::nn::Sequential Discriminator;
    torch::nn::Sequential DiscriminatorSpectral;

    torch::nn::Sequential DiscriminatorPooled;

    Discriminator3DImpl()
    {
        // Spatial
        {
            Discriminator->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(1, 64, 5).stride(1).padding(2)));              // 128x32x128
            //Discriminator->push_back(torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(64).affine(false)));
            Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            Discriminator->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(64, 128, 3).stride(2).padding(1)));            // 64x16x64
            //Discriminator->push_back(torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(128).affine(false)));
            Discriminator->push_back(Discriminator3DResidualBlock(128, true));

            Discriminator->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(128, 256, 3).stride(2).padding(1)));           // 32x8x32
            //Discriminator->push_back(torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(256).affine(false)));
            Discriminator->push_back(Discriminator3DResidualBlock(256, true));

            Discriminator->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(256, 512, 3).stride(2).padding(1)));          // 16x4x16
            //Discriminator->push_back(torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(512).affine(false)));
            Discriminator->push_back(Discriminator3DResidualBlock(512, true));

            Discriminator->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(512, 1024, 3).stride(2).padding(1)));          // 8x2x8
            //Discriminator->push_back(torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(1024).affine(false)));
            Discriminator->push_back(Discriminator3DResidualBlock(1024, false));

            Discriminator->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(1024, 2048, 2).stride(2).padding(0)));         // 4x1x4
            //Discriminator->push_back(torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(1024).affine(false)));


            Discriminator->push_back(torch::nn::Flatten());

            Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(2048 * 4 * 1 * 4, 256)));
            Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, 64)));
            Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            /*Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 64)));
            Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 64)));
            Discriminator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));*/
            Discriminator->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 1)));

            register_module("discriminator", Discriminator);
            register_module("discriminator_pooled", DiscriminatorPooled);
        }
    }

    torch::Tensor forward(torch::Tensor volume)
    {
        torch::Tensor decision = Discriminator->forward(volume);

        return decision;
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

TORCH_MODULE(Discriminator3D);


NNModule THSNN_Discriminator3D_ctor(NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    Discriminator3DImpl Net;
    auto mod = std::make_shared<Discriminator3DImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<Discriminator3DImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_Discriminator3D_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<Discriminator3DImpl>()->forward(*input));
}

Tensor THSNN_Discriminator3D_penalizegradient(const NNModule module, const Tensor real, const Tensor fake, const float lambda)
{
    CATCH_TENSOR((*module)->as<Discriminator3DImpl>()->PenalizeGradient(*real, *fake, lambda));
}