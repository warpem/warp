#include "Functions.h"
#include "CtfGpuCommon.h"
#include <cufft.h>

using namespace gtom;

namespace
{
// ReduceMeanKernel's arithmetic, one frame at a time: the group's running sum starts from 0 and adds the frames in
// order, and the last frame's sum is divided by the frame count.
__global__ void AccumulateMean(const float* frame, float* sum, size_t elements, bool first, bool last, int count)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < elements; i += (size_t)gridDim.x * blockDim.x)
    {
        float total = __fadd_rn(first ? 0.f : sum[i], frame[i]);
        sum[i] = last ? total / (float)count : total;
    }
}
}

// The mean of each group's frames [starts[g], ends[g]) after Fourier-shifting every frame by its own shift (pixels,
// two floats per frame of the stack), into consecutive slices of d_output. Each frame goes through gtom's d_Shift,
// exactly as ShiftStackMassive shifts it, and the shifted frames are averaged as ReduceMean averages them, so the
// result is identical to copying each group's frames out, shifting them with ShiftStackMassive and reducing them
// with ReduceMean; it just reads the frames in place, accumulates as it goes, and makes its plans and buffers once.
extern "C" __declspec(dllexport) int CtfExposureGroupAverage(const float* d_stack, int width, int height, int frames, int groups,
    const int* h_starts, const int* h_ends, const float* h_shifts, float* d_output)
{
    cufftHandle planForward = 0, planBackward = 0;
    float* d_shifted = nullptr;
    tcomplex* d_spectrum = nullptr;
    int result = cudaSuccess;
    try
    {
        if (width < 2 || height < 2 || frames < 1 || groups < 1 || !d_stack || !d_output) return cudaErrorInvalidValue;
        for (int g = 0; g < groups; g++)
            if (h_starts[g] < 0 || h_ends[g] <= h_starts[g] || h_ends[g] > frames) return cudaErrorInvalidValue;
        int3 dims = make_int3(width, height, 1);
        size_t pixels = (size_t)width * height;
        planForward = d_FFTR2CGetPlan(DimensionCount(dims), dims);
        planBackward = d_IFFTC2RGetPlan(DimensionCount(dims), dims);
        warp_ctf::Check(cudaMalloc((void**)&d_shifted, pixels * sizeof(float)));
        warp_ctf::Check(cudaMalloc((void**)&d_spectrum, ElementsFFT(dims) * sizeof(tcomplex)));
        int blocks = (int)std::min<size_t>((pixels + 255) / 256, 8192);
        for (int g = 0; g < groups; g++)
        {
            int count = h_ends[g] - h_starts[g];
            for (int z = h_starts[g]; z < h_ends[g]; z++)
            {
                tfloat3 shift = make_float3(h_shifts[2 * z], h_shifts[2 * z + 1], 0);
                d_Shift((tfloat*)(d_stack + pixels * z), d_shifted, dims, &shift, &planForward, &planBackward, d_spectrum);
                AccumulateMean<<<blocks, 256>>>(d_shifted, d_output + pixels * g, pixels, z == h_starts[g], z == h_ends[g] - 1, count);
            }
        }
        warp_ctf::Check(cudaGetLastError());
        warp_ctf::Check(cudaStreamSynchronize(cudaStreamLegacy));
    }
    catch (cudaError_t e) { result = e; } catch (...) { result = cudaErrorUnknown; }
    cudaFree(d_shifted); cudaFree(d_spectrum);
    if (planForward) cufftDestroy(planForward);
    if (planBackward) cufftDestroy(planBackward);
    return result;
}
