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


struct C2DNetResidualBlock : MultiGPUModule
{
    int64_t _channels;
    bool _donorm;
    bool _doattention;

    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::BatchNorm2d bn2{ nullptr };

    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };

    torch::nn::Conv2d convsa{ nullptr };

    C2DNetResidualBlock(int64_t channels, bool donorm = true, bool doattention = true)
    {
        _channels = channels;
        _donorm = donorm;
        _doattention = doattention;


        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1).groups(32)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(_channels)));

        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(_channels, _channels, 3).stride(1).padding(1).groups(32)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(_channels)));

        if (_doattention)
        {
            fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(_channels * 2, _channels / 8)));
            fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(_channels / 8, _channels)));

            convsa = register_module("convsa", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 1, 7).stride(1).padding(3)));
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor residual(x.clone());

        if (_donorm)
            x = bn1->forward(x);
        x = torch::relu(x);
        x = conv1->forward(x);

        if (_donorm)
            x = bn2->forward(x);
        x = torch::relu(x);
        x = conv2->forward(x);

        if (_doattention)
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

            {
                torch::Tensor sa = torch::cat({ x.mean(1, true), std::get<0>(x.max(1, true)) }, 1);

                sa = convsa->forward(sa);
                sa = sa.sigmoid();

                x = x.mul(sa);
            }
        }

        x += residual;
        //x = torch::leaky_relu(x, 0.2);

        return x;
    }
};

struct C2DNetEncoderImpl : MultiGPUModule
{
    int64_t _boxsize;
    int64_t _codelength;

    torch::nn::Sequential poseestimator;
    torch::nn::Sequential poseestimatorfine;

    torch::nn::Sequential encoder;

    torch::nn::Sequential posepooler;

    C2DNetEncoderImpl(int64_t boxsize, int64_t codelength)
    {
        _boxsize = boxsize;
        _codelength = codelength;

        encoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).stride(1).padding(2)));
        encoder->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
        encoder->push_back(C2DNetResidualBlock(64, true, false));

        poseestimator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).stride(1).padding(2)));
        poseestimator->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
        poseestimator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

        poseestimatorfine->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 64, 5).stride(1).padding(2)));
        poseestimatorfine->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
        poseestimatorfine->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

        int currentchannels = 64;
        int currentsize = _boxsize;

        while (currentsize / 2 >= 2)
        {
            currentchannels *= 2;
            currentsize /= 2;

            encoder->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
            encoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels / 2, currentchannels, 1).padding(0).stride(1)));

            //encoder->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels / 2, currentchannels, 3).padding(1).stride(2)));
            encoder->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
            //encoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            encoder->push_back(C2DNetResidualBlock(currentchannels, true, false));
            encoder->push_back(C2DNetResidualBlock(currentchannels, true, false));


            poseestimator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels / 2, currentchannels, 3).padding(1).stride(2)));
            poseestimator->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
            poseestimator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));


            poseestimatorfine->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels / 2, currentchannels, 3).padding(1).stride(2)));
            poseestimatorfine->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
            poseestimatorfine->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
        }

        encoder->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })));

        encoder->push_back(torch::nn::Flatten());
        encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(currentchannels, codelength * 2)));
        //encoder->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
        //encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(std::max(128, (int)_codelength), codelength)));

        register_module("encoder", encoder);
        /*size_t nc = encoder->children().size();
        std::cout << nc;
        for (auto p : encoder->children().back()->named_parameters())
            if (p.key() == "weight")
                torch::nn::init::constant_(p.value(), 0);
            else if (p.key() == "bias")
                torch::nn::init::constant_(p.value(), 0);*/

        poseestimator->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels, currentchannels * 2 * 2, 2).padding(0).stride(1)));
        //poseestimator->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels * 2 * 2)));
        poseestimator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));

        poseestimator->push_back(torch::nn::Flatten());
        poseestimator->push_back(torch::nn::Linear(torch::nn::LinearOptions(currentchannels * 2 * 2, 512)));
        poseestimator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
        poseestimator->push_back(torch::nn::Linear(torch::nn::LinearOptions(512, 256)));
        poseestimator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
        poseestimator->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, 128)));
        poseestimator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
        poseestimator->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 64)));
        poseestimator->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
        poseestimator->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 4)));

        register_module("pose_estimator", poseestimator);


        poseestimatorfine->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels, currentchannels * 2, 2).padding(0).stride(1)));
        poseestimatorfine->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
        poseestimatorfine->push_back(torch::nn::Flatten());
        poseestimatorfine->push_back(torch::nn::Linear(torch::nn::LinearOptions(currentchannels * 2, 256)));
        poseestimatorfine->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
        poseestimatorfine->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, 128)));
        poseestimatorfine->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
        poseestimatorfine->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 3)));

        register_module("pose_estimator_fine", poseestimatorfine);

        posepooler->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 4, 4 })));
        register_module("pose_pooler", posepooler);
    }

    torch::Tensor forward(torch::Tensor input)
    {
        torch::Tensor result = encoder->forward(input);
        result = result.reshape({ -1, 2, _codelength });

        return result;
    }

    torch::Tensor forward_pose(torch::Tensor input)
    {
        torch::Tensor result = poseestimator->forward(input);

        return result;
    }

    torch::Tensor forward_pose_fine(torch::Tensor decoded, torch::Tensor target)
    {
        torch::Tensor result = poseestimatorfine->forward(torch::cat({ decoded, target }, 1));

        return result;
    }

    torch::Tensor apply_pose(torch::Tensor input, torch::Tensor pose)
    {
        pose = pose.unsqueeze(2);
        pose.print();

        torch::Tensor dirnorm = (pose.slice(1, 0, 2).norm(2, 1, true) + 1e-8);
        dirnorm.print();

        torch::Tensor dirx = pose.slice(1, 0, 1) / dirnorm;
        torch::Tensor diry = pose.slice(1, 1, 2) / dirnorm;
        dirx.print();

        torch::Tensor rot_mat1 = torch::cat({ dirx, -diry }, 2);
        torch::Tensor rot_mat2 = torch::cat({ diry,  dirx }, 2);

        torch::Tensor trans_matrices = torch::cat({ rot_mat1, rot_mat2 }, 1);
        trans_matrices.print();
        trans_matrices = torch::cat({ trans_matrices, pose.slice(1, 2, 4) / (input.size(2) / 8) }, 2);
        torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { input.size(0), input.size(1), input.size(2), input.size(3) });

        torch::Tensor result = torch::nn::functional::grid_sample(input, trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kReflection));

        return result;
    }

    torch::Tensor pose_loss(torch::Tensor pose, torch::Tensor refdotprods)
    {
        //torch::Tensor dirnorm = (1 - pose.slice(1, 0, 2).norm(2, 1, false)).square().mean();
        ////dirnorm = (1 - dirnorm).abs();

        ///*torch::Tensor dirstd = (pose.slice(1, 0, 1).reshape({ -1 }).std(0) + pose.slice(1, 1, 2).reshape({ -1 }).std(0)) * 0.5;
        //dirstd = (0.7066 - dirstd).abs();*/

        ///*torch::Tensor angles = torch::atan2(pose.slice(1, 1, 2), pose.slice(1, 0, 1)) / M_PI;

        //torch::Tensor anglemean = angles.mean().abs();
        //torch::Tensor anglestd = angles.reshape({ -1 }).std(0);
        //anglestd = (0.5773 - anglestd).abs();*/

        //torch::Tensor posmean = pose.slice(1, 2, 4).mean().abs();

        ////torch::Tensor posstd = pose.slice(1, 2, 4).reshape({ -1 }).std(0);

        //return (dirnorm + posmean);

        /*torch::Tensor posed1 = posepooler->forward(pose.slice(0, 0, pose.size(0) - 1, 2));
        torch::Tensor posed2 = posepooler->forward(pose.slice(0, 1, pose.size(0) - 0, 2));

        return (posed1 - posed2).square().mean();*/

        pose = pose.slice(1, 0, 2);
        torch::Tensor dirnorm = pose.norm(2, 1, true) + 1e-8;
        pose = pose / dirnorm;

        /*torch::Tensor dirx = pose.slice(1, 0, 1);
        torch::Tensor diry = pose.slice(1, 1, 2);
        torch::Tensor mats = torch::cat({ dirx, -diry, diry, dirx }, 1);

        torch::Tensor matprods = (mats.slice(0, 0, mats.size(0) - 1, 2) + mats.slice(0, 1, mats.size(0) - 0, 2));*/

        torch::Tensor dotprods = (pose.slice(0, 0, pose.size(0) - 1, 2) * pose.slice(0, 1, pose.size(0) - 0, 2)).sum(1);

        return (refdotprods - dotprods).abs().mean() * 2;
    }

    torch::Tensor apply_pose_fine(torch::Tensor input, torch::Tensor pose)
    {
        pose = pose.unsqueeze(2);

        torch::Tensor angle = pose.slice(1, 0, 1);

        torch::Tensor dirx = angle.cos();
        torch::Tensor diry = angle.sin();

        torch::Tensor rot_mat1 = torch::cat({ dirx, -diry }, 2);
        torch::Tensor rot_mat2 = torch::cat({ diry,  dirx }, 2);

        torch::Tensor trans_matrices = torch::cat({ rot_mat1, rot_mat2 }, 1);
        trans_matrices.print();
        trans_matrices = torch::cat({ trans_matrices, pose.slice(1, 2, 4) / (input.size(2) / 2) }, 2);
        torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { input.size(0), input.size(1), input.size(2), input.size(3) });

        torch::Tensor result = torch::nn::functional::grid_sample(input, trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kReflection));
        
        return result;
    }
};

TORCH_MODULE(C2DNetEncoder);

struct C2DNetDecoderImpl : MultiGPUModule
{
    int64_t _boxsize;
    int64_t _codelength;
    int64_t _smallestsize;
    int64_t _smallestchannels;

    torch::nn::Sequential decoder_lin;
    torch::nn::Sequential decoder_conv;
    torch::nn::Sequential decoder_conv_final;

    C2DNetDecoderImpl(int64_t boxsize, int64_t codelength)
    {
        _boxsize = boxsize;
        _codelength = codelength;

        int currentchannels = 4;
        int currentsize = _boxsize;
        while (currentsize / 2 >= 4)
        {
            currentchannels *= 4;
            currentsize /= 2;
        }

        _smallestchannels = currentchannels;
        _smallestsize = currentsize;
        decoder_lin->push_back(torch::nn::Linear(torch::nn::LinearOptions(codelength, 128)));
        decoder_lin->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm2dOptions(128)));
        decoder_lin->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
        decoder_lin->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 128)));
        decoder_lin->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm2dOptions(128)));
        decoder_lin->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
        decoder_lin->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 128)));
        decoder_lin->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm2dOptions(128)));
        decoder_lin->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
        decoder_lin->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 1 * _boxsize * _boxsize).bias(true)));

        for (auto m : decoder_lin->modules(false))
            if (m->name() == "torch::nn::Linear")
                for (auto p : m->named_parameters())
                    if (p.key() == "weight")
                        torch::nn::init::xavier_uniform_(p.value());
                    else if (p.key() == "bias")
                        torch::nn::init::constant_(p.value(), 0);

        //decoder_lin->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
        /*decoder_lin->push_back(torch::nn::Linear(torch::nn::LinearOptions(codelength, std::max(128, (int)_codelength))));
        decoder_lin->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
        decoder_lin->push_back(torch::nn::Linear(torch::nn::LinearOptions(std::max(128, (int)_codelength), std::max(128, (int)_codelength))));
        decoder_lin->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
        decoder_lin->push_back(torch::nn::Linear(torch::nn::LinearOptions(std::max(128, (int)_codelength), _smallestchannels * _smallestsize * _smallestsize)));*/

        decoder_conv->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(codelength, currentchannels, currentsize).stride(1)));
        decoder_conv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
        decoder_conv->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));
        decoder_conv->push_back(C2DNetResidualBlock(currentchannels, true, false));

        while (currentsize * 2 <= boxsize)
        {
            currentchannels /= 4;
            currentsize *= 2;

            //decoder_conv->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2, 2 })).mode(torch::kBilinear).align_corners(true)));

            //decoder_conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels * 2, currentchannels, 3).padding(1)));
            decoder_conv->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(currentchannels * 4, currentchannels, 5).stride(2).padding(2).output_padding(1)));
            decoder_conv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
            decoder_conv->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)));

            //decoder_conv->push_back(C2DNetResidualBlock(currentchannels, true, false));
        }

        //decoder_conv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(currentchannels)));
        //decoder_conv->push_back(C2DNetResidualBlock(currentchannels, true, false));
        //decoder_conv->push_back(C2DNetResidualBlock(currentchannels, true, false));

        decoder_conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(currentchannels, 1, 1).padding(0)));

        register_module("decoder_lin", decoder_lin);
        register_module("decoder_conv", decoder_conv);
        //register_module("decoder_conv_final", decoder_conv_final);
    }

    torch::Tensor forward(torch::Tensor input, bool usekl)
    {
        torch::Tensor code = input.slice(1, 0, 1).squeeze(1);

        if (usekl)
        {
            torch::Tensor logvar = input.slice(1, 1, 2).squeeze(1);
            code = code + torch::randn_like(logvar, logvar.options()) * torch::exp(0.5 * logvar);
        }

        //torch::Tensor result = decoder_lin->forward(code).reshape({ -1, 1, _boxsize, _boxsize });
        //torch::Tensor result = decoder_conv->forward(smallest);

        torch::Tensor result = decoder_conv->forward(code.reshape({ -1, _codelength, 1, 1 }));

        return result;

        //double anglestep = 360.0 / nrotations / 180.0 * M_PI;
        //std::vector<torch::Tensor> results;

        //torch::Tensor angle = input.slice(1, 2, 3, 1).unsqueeze(2);// .tanh()* (anglestep / 1.0);
        ////direction = direction / (direction.norm(2, 1, true) + 1e-8);
        ////direction = direction.unsqueeze(2);

        //for (int i = 0; i < nrotations; i++)
        //{
        //    torch::Tensor currangle = angle + i * anglestep;
        //    torch::Tensor dirx = currangle.cos();
        //    torch::Tensor diry = currangle.sin();

        //    torch::Tensor rot_mat1 = torch::cat({ dirx, -diry }, 2);
        //    torch::Tensor rot_mat2 = torch::cat({ diry,  dirx }, 2);

        //    torch::Tensor trans_matrices = torch::cat({ rot_mat1, rot_mat2 }, 1);
        //    trans_matrices = torch::cat({ trans_matrices, torch::zeros({input.size(0), 2, 1}, input.options()) }, 2);
        //    torch::Tensor trans_grid = torch::nn::functional::affine_grid(trans_matrices, { result.size(0), result.size(1), result.size(2), result.size(3) });

        //    torch::Tensor rotated = torch::nn::functional::grid_sample(result, trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kReflection));
        //    results.push_back(rotated);

        //    /*trans_matrices = torch::cat({ torch::eye(2, input.options()).repeat({input.size(0), 1, 1}), input.slice(1, 0, 2, 1).unsqueeze(2).tanh() * 0.2 }, 2);
        //    trans_grid = torch::nn::functional::affine_grid(trans_matrices, { result.size(0), result.size(1), result.size(2), result.size(3) });

        //    torch::Tensor translated = torch::nn::functional::grid_sample(rotated, trans_grid, torch::nn::functional::GridSampleFuncOptions().padding_mode(torch::kReflection));
        //    results.push_back(translated);*/
        //}   


        //return torch::cat(results, 1);
    }
};

TORCH_MODULE(C2DNetDecoder);


NNModule THSNN_C2DNetEncoder_ctor(const int64_t boxsize, const int64_t codelength, NNAnyModule* outAsAnyModule)
{
    at::globalContext().setBenchmarkCuDNN(true);
    //CATCH_RETURN_NNModule
    //(
    C2DNetEncoderImpl Net(boxsize, codelength);
    auto mod = std::make_shared<C2DNetEncoderImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<C2DNetEncoderImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_C2DNetEncoder_forward(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<C2DNetEncoderImpl>()->forward(*input));
}

Tensor THSNN_C2DNetEncoder_forward_pose(const NNModule module, const Tensor input)
{
    CATCH_TENSOR((*module)->as<C2DNetEncoderImpl>()->forward_pose(*input));
}

Tensor THSNN_C2DNetEncoder_apply_pose(const NNModule module, const Tensor input, const Tensor pose)
{
    CATCH_TENSOR((*module)->as<C2DNetEncoderImpl>()->apply_pose(*input, *pose));
}

Tensor THSNN_C2DNetEncoder_pose_loss(const NNModule module, const Tensor pose, const Tensor refdotprods)
{
    CATCH_TENSOR((*module)->as<C2DNetEncoderImpl>()->pose_loss(*pose, *refdotprods));
}


NNModule THSNN_C2DNetDecoder_ctor(const int64_t boxsize, const int64_t codelength, NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    C2DNetDecoderImpl Net(boxsize, codelength);
    auto mod = std::make_shared<C2DNetDecoderImpl>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<C2DNetDecoderImpl>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_C2DNetDecoder_forward(const NNModule module, const Tensor input, const bool usekl)
{
    CATCH_TENSOR((*module)->as<C2DNetDecoderImpl>()->forward(*input, usekl));
}

Tensor THSNN_C2DNetDecoder_minmse(const Tensor decoded, const Tensor data)
{
    torch::Tensor mse = (*decoded - *data).square().mean({ 2,3 }, false);
    auto minmse = mse.min(1, false);
    CATCH_TENSOR(std::get<0>(minmse).mean());
}

Tensor THSNN_C2DNetDecoder_kld(const NNModule module, const Tensor input, const double weight)
{
    torch::Tensor mu = input->slice(1, 0, 1).squeeze(1);
    torch::Tensor logvar = input->slice(1, 1, 2).squeeze(1);

    CATCH_TENSOR(torch::mean(-0.5 * torch::sum(1 + logvar - mu.square() - logvar.exp(), 1), 0) * weight);
}