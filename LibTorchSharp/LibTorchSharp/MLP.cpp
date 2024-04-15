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


struct MLPBlock : MultiGPUModule
{
    int64_t _features_in;
    int64_t _features_out;
    bool _residual;
    bool _useactivation;

    torch::nn::Linear fc{ nullptr };

    MLPBlock(int64_t features_in, int64_t features_out, bool residual = false, bool useactivation = true)
    {
        _features_in = features_in;
        _features_out = features_out;
        _residual = residual;
        _useactivation = useactivation;

        fc = register_module("fc", torch::nn::Linear(torch::nn::LinearOptions(_features_in, _features_out)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor result = fc->forward(x);

        if (_useactivation)
            result = torch::leaky_relu(result, 0.2);

        if (_features_in == _features_out && _residual)
            result += x;

        return result;
    }
};

struct MLPImpl : MultiGPUModule
{
    torch::nn::Sequential blocks;

    MLPImpl(std::vector<int64_t> block_widths, bool residual)
    {
        for (int i = 0; i < block_widths.size() - 1; i++)
            blocks->push_back(MLPBlock(block_widths[i], block_widths[i + 1], residual, i < block_widths.size() - 2));
    }

    torch::Tensor forward(torch::Tensor input)
    {
        return blocks->forward(input);
    }
};

TORCH_MODULE(MLP);


NNModule THSNN_MLP_ctor(int64_t* block_widths, int nblocks, bool residual, NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    MLPImpl Net(std::vector<int64_t>(block_widths, block_widths + nblocks), residual);
    auto mod = std::make_shared<MLPImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<MLPImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_MLP_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<MLPImpl>()->forward(*input));
}