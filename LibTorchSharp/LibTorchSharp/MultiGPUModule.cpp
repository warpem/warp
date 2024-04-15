#pragma once

#include "MultiGPUModule.h"

void MultiGPUModule::syncronize_with(MultiGPUModule* replica, torch::Device device)
{
    auto replica_params = replica->parameters(true);
    auto this_params = this->parameters(true);

    TORCH_CHECK(
        replica_params.size() == this_params.size(),
        "The cloned module does not have the same number of "
        "parameters as the original module after calling reset(). "
        "Are you sure you called register_parameter() inside reset() "
        "and not the constructor?")

	#pragma omp parallel for num_threads(8)
    for (int64_t i = 0; i < this_params.size(); i++)
    {
        auto& tensor = this_params[i];
        //tensor.print();
        auto data = tensor.device() != device ? tensor.to(device) : torch::autograd::Variable(tensor).clone();
        replica_params[i].set_data(data);
    }

    //std::cout << "now buffers" << std::endl;

    auto replica_buffers = replica->buffers(true);
    auto this_buffers = this->buffers(true);

    TORCH_CHECK(
        replica_buffers.size() == this_buffers.size(),
        "The cloned module does not have the same number of "
        "buffers as the original module after calling reset(). "
        "Are you sure you called register_buffer() inside reset() "
        "and not the constructor?")

	#pragma omp parallel for num_threads(8)
    for (int64_t i = 0; i < this_buffers.size(); i++)
    {
        auto& tensor = this_buffers[i];
        //tensor.print();
        auto data = tensor.device() != device ? tensor.to(device) : torch::autograd::Variable(tensor).clone();
        replica_buffers[i].set_data(data);
    }
}

void MultiGPUModule::gather_grad(MultiGPUModule* replica)
{
    //std::cout << "gradient tensors:" << std::endl;

    auto replica_params = replica->parameters(true);
    auto this_params = this->parameters(true);

    TORCH_CHECK(
        replica_params.size() == this_params.size(),
        "The cloned module does not have the same number of "
        "parameters as the original module after calling reset(). "
        "Are you sure you called register_parameter() inside reset() "
        "and not the constructor?")

	#pragma omp parallel for num_threads(8)
    for (int64_t i = 0; i < this_params.size(); i++)
    {
        auto& grad = this_params[i].grad();
        //this_params[i].print();
        if (grad.defined())
        {
            auto replica_grad = replica_params[i].grad();
            TORCH_CHECK(
                replica_grad.defined(),
                "Original grad defined, but replica is not!");

            auto data = grad.device() != replica_grad.device() ? replica_grad.to(grad.device()) : replica_grad;
            //std::cout << "has gradient:" << std::endl;
            //data.print();

            grad.add_(data);
        }
        else
        {
            //std::cout << "has no gradient" << std::endl;
        }
        //std::cout << std::endl;
    }
}

void MultiGPUModule::scale_grad(const float factor)
{
    auto this_params = this->parameters(true);

	#pragma omp parallel for num_threads(8)
    for (int64_t i = 0; i < this_params.size(); i++)
    {
        auto& grad = this_params[i].grad();
        //this_params[i].print();
        if (grad.defined())
        {
            grad.mul_(factor);
        }
        else
        {
            //std::cout << "has no gradient" << std::endl;
        }
        //std::cout << std::endl;
    }
}

void THSNN_module_synchronize_with(const NNModule module, const NNModule replica, const int64_t device)
{
    (*module)->as<MultiGPUModule>()->syncronize_with((*replica)->as<MultiGPUModule>(),
        torch::Device(torch::kCUDA, device));
}

void THSNN_module_gather_grad(const NNModule module, const NNModule replica)
{
    (*module)->as<MultiGPUModule>()->gather_grad((*replica)->as<MultiGPUModule>());
}

void THSNN_module_scale_grad(const NNModule module, const float factor)
{
    (*module)->as<MultiGPUModule>()->scale_grad(factor);
}