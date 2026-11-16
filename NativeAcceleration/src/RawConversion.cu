#include "include/Functions.h"

// Raw detector samples to float on the device, optionally flipping rows, as ReadTIFF does on the host:
// uploading a movie frame in its file type moves a quarter of the bytes for 8-bit data.
namespace
{
template<class T> __global__ void RawToFloatKernel(const T* raw, float* result, int width, int height, bool flipy)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)width * height) return;
    int y = (int)(i / width), x = (int)(i % width);
    int source = flipy ? height - 1 - y : y;
    result[i] = (float)raw[(size_t)source * width + x];
}
template<class T> void Launch(const void* raw, float* result, int width, int height, bool flipy)
{
    size_t elements = (size_t)width * height;
    RawToFloatKernel<T><<<(unsigned)((elements + 255) / 256), 256>>>((const T*)raw, result, width, height, flipy);
}
}

__declspec(dllexport) void RawToFloat(const void* d_raw, int type, float* d_result, int width, int height, bool flipy)
{
    switch (type)
    {
    case 0: Launch<unsigned char>(d_raw, d_result, width, height, flipy); break;
    case 1: Launch<signed char>(d_raw, d_result, width, height, flipy); break;
    case 2: Launch<unsigned short>(d_raw, d_result, width, height, flipy); break;
    case 3: Launch<short>(d_raw, d_result, width, height, flipy); break;
    case 4: Launch<float>(d_raw, d_result, width, height, flipy); break;
    }
}
