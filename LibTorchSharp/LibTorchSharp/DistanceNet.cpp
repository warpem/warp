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

struct DistanceNetResidualBlock : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;

    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::BatchNorm2d bn2{ nullptr };

    DistanceNetResidualBlock(int64_t channels, bool donorm = true)
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

struct DistanceNetImpl : MultiGPUModule
{
    torch::nn::Sequential ProcessorReference;
    torch::nn::Sequential ProcessorData;

    torch::nn::Sequential ProcessorCombined;

    torch::nn::Sequential ProcessorDistance;

    int64_t _codelength;

    DistanceNetImpl()
    {
        // Ref
        {
            ProcessorReference->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 64, 5).stride(1).padding(2)));
            ProcessorReference->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
            ProcessorReference->push_back(DistanceNetResidualBlock(64, true));
            ProcessorReference->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
            ProcessorReference->push_back(DistanceNetResidualBlock(64, true));
            ProcessorReference->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));

            register_module("processor_reference", ProcessorReference);
        }

        // Data
        {
            ProcessorData->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 5).stride(1).padding(2)));
            ProcessorData->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorData->push_back(DistanceNetResidualBlock(32, true));
            ProcessorData->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            ProcessorData->push_back(DistanceNetResidualBlock(32, true));
            ProcessorData->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));

            register_module("processor_data", ProcessorData);
        }

        // Combined
        {
            ProcessorCombined->push_back(DistanceNetResidualBlock(64, true));
            ProcessorCombined->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
            ProcessorCombined->push_back(DistanceNetResidualBlock(64, true));
            ProcessorCombined->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));

            register_module("processor_combined", ProcessorCombined);
        }

        // Distance
        {
            //ProcessorDistance->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 64, 3).stride(1).padding(1)));     // 128
            //ProcessorDistance->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
            //ProcessorDistance->push_back(DistanceNetResidualBlock(32, true));

            ProcessorDistance->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));     // 64
            ProcessorDistance->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128)));
            ProcessorDistance->push_back(DistanceNetResidualBlock(128, true));

            ProcessorDistance->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));    // 32
            ProcessorDistance->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)));
            ProcessorDistance->push_back(DistanceNetResidualBlock(256, true));

            ProcessorDistance->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1)));    // 16
            ProcessorDistance->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            ProcessorDistance->push_back(DistanceNetResidualBlock(512, true));

            ProcessorDistance->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(2).padding(1)));   // 8
            ProcessorDistance->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1024)));
            ProcessorDistance->push_back(DistanceNetResidualBlock(1024, true));

            ProcessorDistance->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 1024, 3).stride(2).padding(1)));  // 4
            ProcessorDistance->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1024)));
            ProcessorDistance->push_back(DistanceNetResidualBlock(1024, true));

            //ProcessorDistance->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 1024, 3).stride(2).padding(1)));  // 2
            //ProcessorDistance->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1024)));
            //ProcessorDistance->push_back(DistanceNetResidualBlock(1024, true));

            ProcessorDistance->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })));      // 1

            ProcessorDistance->push_back(torch::nn::Flatten());
            ProcessorDistance->push_back(torch::nn::Linear(torch::nn::LinearOptions(1024 * 1 * 1, 128)));
            ProcessorDistance->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            ProcessorDistance->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 64)));
            ProcessorDistance->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            ProcessorDistance->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 1)));

            register_module("processor_distance", ProcessorDistance);
        }
    }

    torch::Tensor forward(torch::Tensor reference, torch::Tensor data, void* d_reference, void* d_data)
    {
        reference = ProcessorReference->forward(torch::cat({ reference, data }, 1));
        //data = ProcessorData->forward(data);

        cudaMemcpy(d_reference, reference.slice(1, 0, 1, 1).data_ptr(), 128 * 128 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_data, reference.slice(1, 1, 2, 1).data_ptr(), 128 * 128 * sizeof(float), cudaMemcpyDeviceToDevice);

        //torch::Tensor combined = ProcessorCombined->forward(torch::cat({ reference, data }, 1));
        //torch::Tensor multiplied = combined.slice(1, 0, 32, 1).mul(combined.slice(1, 32, 64, 1));

        torch::Tensor distance = ProcessorDistance->forward(reference);// .square().neg().exp();

        return distance;
    }
};

TORCH_MODULE(DistanceNet);


NNModule THSNN_DistanceNet_ctor(NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    DistanceNetImpl Net;
    auto mod = std::make_shared<DistanceNetImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<DistanceNetImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_DistanceNet_forward(const NNModule module, const Tensor reference, const Tensor data, void* d_reference, void* d_data)
{
    CATCH_TENSOR((*module)->as<DistanceNetImpl>()->forward(*reference, *data, d_reference, d_data));
}