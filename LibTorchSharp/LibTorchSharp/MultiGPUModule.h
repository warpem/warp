#pragma once

#include <torch/nn/init.h>
#include <torch/nn/pimpl.h>

#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>

#include "../Stdafx.h"
#include "Utils.h"

class MultiGPUModule : public virtual torch::nn::Module
{
public:

    void syncronize_with(MultiGPUModule* replica, torch::Device device);

    void gather_grad(MultiGPUModule* replica);

    void scale_grad(const float factor);
};

EXPORT_API(void) THSNN_module_synchronize_with(const NNModule module, const NNModule replica, const int64_t device);

EXPORT_API(void) THSNN_module_gather_grad(const NNModule module, const NNModule replica);

EXPORT_API(void) THSNN_module_scale_grad(const NNModule module, const float factor);