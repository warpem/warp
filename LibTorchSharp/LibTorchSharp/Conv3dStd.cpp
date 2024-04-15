//#pragma once
//
//#include "CustomModules.h"
//
//#include <torch/nn/init.h>
//#include <torch/nn/pimpl.h>
//#include <torch/fft.h>
//#include <torch/nn/parallel/data_parallel.h>
//
//#include <torch/nn/module.h>
//#include <torch/types.h>
//#include <torch/utils.h>
//
//#include <c10/core/TensorOptions.h>
//#include <c10/util/Exception.h>
//
//namespace F = torch::nn::functional;
//
//F::PadFuncOptions::mode_t _get_pad_mode_from_conv_padding_mode(torch::nn::detail::conv_padding_mode_t conv_padding_mode) {
//    F::PadFuncOptions::mode_t pad_mode;
//    if (c10::get_if<torch::enumtype::kReflect>(&conv_padding_mode)) {
//        pad_mode = torch::kReflect;
//    }
//    else if (c10::get_if<torch::enumtype::kReplicate>(&conv_padding_mode)) {
//        pad_mode = torch::kReplicate;
//    }
//    else if (c10::get_if<torch::enumtype::kCircular>(&conv_padding_mode)) {
//        pad_mode = torch::kCircular;
//    }
//    else {
//        TORCH_CHECK(false, "Unsupported conv padding mode: ", torch::enumtype::get_enum_name(conv_padding_mode));
//    }
//    return pad_mode;
//}
//
//namespace torch
//{
//    namespace nn
//    {
//        Tensor Conv3dStdImpl::forward(const Tensor& input)
//        {
//            auto wshape = weight.sizes();
//            int cout = wshape[0];
//            Tensor wflat = weight.view(std::vector<int64_t>({ cout, -1 }));
//            Tensor wmean = wflat.mean();
//            Tensor wstd = wflat.std();
//            wflat = (wflat - wmean) / (wstd + 1e-5);
//            wflat = wflat.view(wshape);
//
//            if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
//                return F::detail::conv3d(
//                    F::pad(input, F::PadFuncOptions(_reversed_padding_repeated_twice).mode(_get_pad_mode_from_conv_padding_mode(options.padding_mode()))),
//                    wflat, bias,
//                    options.stride(),
//                    /*padding=*/0,
//                    options.dilation(),
//                    options.groups());
//            }
//            return F::detail::conv3d(
//                input,
//                wflat,
//                bias,
//                options.stride(),
//                options.padding(),
//                options.dilation(),
//                options.groups());
//        }
//    }
//}