#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

EXPORT_API(NNModule) THSNN_ResNet_ctor(const int64_t size_input, const int64_t blocks1, const int64_t blocks2, const int64_t blocks3, const int64_t blocks4, const int64_t num_classes, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ResNet_forward(const NNModule module, const Tensor tensor);

EXPORT_API(NNModule) THSNN_UNet3D_ctor(const int64_t depth_block, const int64_t width_block, const int64_t max_width, const int64_t input_channels, const int64_t final_channels, const bool residual_u, const bool usegroupnorm, const bool progressivedepth, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_UNet3D_forward(const NNModule module, const Tensor input);

EXPORT_API(NNModule) THSNN_UNet2D_ctor(const int64_t depth_block, const int64_t width_block, const int64_t input_channels, const int64_t final_channels, const int64_t final_kernel, const bool dochannelattn, const bool dospatialattn, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_UNet2D_forward(const NNModule module, const Tensor input);

EXPORT_API(NNModule) THSNN_BoxNetMulti_ctor(const int64_t depth_block, const int64_t width_block, const int64_t input_channels, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_BoxNetMulti_pick_forward(const NNModule module, const Tensor input);
EXPORT_API(Tensor)   THSNN_BoxNetMulti_denoise_forward(const NNModule module, const Tensor input);

EXPORT_API(NNModule) THSNN_ParticleWGANGenerator_ctor(int64_t boxsize, int64_t codelength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ParticleWGANGenerator_forward_particle(const NNModule module, const Tensor code, const bool transform, const double sigmashift);
EXPORT_API(Tensor)   THSNN_ParticleWGANGenerator_forward_noise(const NNModule module, const Tensor crapcode, const Tensor fakeimages, const Tensor ctf);

EXPORT_API(NNModule) THSNN_ParticleWGANDiscriminator_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_ParticleWGANDiscriminator_forward(const NNModule module, const Tensor input);
EXPORT_API(void)     THSNN_ParticleWGANDiscriminator_clipweights(const NNModule module, const double clip);
EXPORT_API(Tensor)   THSNN_ParticleWGANDiscriminator_penalizegradient(const NNModule module, const Tensor real, const Tensor fake, const float lambda);

EXPORT_API(NNModule) THSNN_DistanceNet_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_DistanceNet_forward(const NNModule module, const Tensor reference, const Tensor data, void* d_reference, void* d_data);

EXPORT_API(NNModule) THSNN_C2DNetEncoder_ctor(const int64_t boxsize, const int64_t codelength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_C2DNetEncoder_forward(const NNModule module, const Tensor input);
EXPORT_API(Tensor)   THSNN_C2DNetEncoder_forward_pose(const NNModule module, const Tensor input);
EXPORT_API(Tensor)   THSNN_C2DNetEncoder_apply_pose(const NNModule module, const Tensor input, const Tensor pose);
EXPORT_API(NNModule) THSNN_C2DNetDecoder_ctor(const int64_t boxsize, const int64_t codelength, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_C2DNetDecoder_forward(const NNModule module, const Tensor input, const bool usekl);
EXPORT_API(Tensor)   THSNN_C2DNetDecoder_kld(const NNModule module, const Tensor input, const double weight);
EXPORT_API(Tensor)   THSNN_C2DNetDecoder_minmse(const Tensor decoded, const Tensor data);

EXPORT_API(NNModule) THSNN_MLP_ctor(int64_t* block_widths, int nblocks, bool residual, NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_MLP_forward(const NNModule module, const Tensor input);

EXPORT_API(NNModule) THSNN_Discriminator3D_ctor(NNAnyModule* outAsAnyModule);
EXPORT_API(Tensor)   THSNN_Discriminator3D_forward(const NNModule module, const Tensor input);
EXPORT_API(Tensor)   THSNN_Discriminator3D_penalizegradient(const NNModule module, const Tensor real, const Tensor fake, const float lambda);

