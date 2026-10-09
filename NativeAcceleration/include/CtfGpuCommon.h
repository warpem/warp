#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace warp_ctf
{
inline void Check(cudaError_t error) { if (error != cudaSuccess) throw error; }
struct DeviceScope
{
    int previous;
    explicit DeviceScope(int device) { Check(cudaGetDevice(&previous)); Check(cudaSetDevice(device)); }
    ~DeviceScope() { cudaSetDevice(previous); }
};
template<class T> struct Buffer
{
    T* p = nullptr;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (p) cudaFree(p); }
    void Allocate(size_t n) { if(p) { Check(cudaFree(p)); p=nullptr; } Check(cudaMalloc((void**)&p, n * sizeof(T))); }
    void Upload(const T* source, size_t n) { Check(cudaMemcpy(p, source, n*sizeof(T), cudaMemcpyHostToDevice)); }
    void Download(T* target, size_t n) { Check(cudaMemcpy(target, p, n*sizeof(T), cudaMemcpyDeviceToHost)); }
    template<class U> void UploadConverted(const U* source, size_t n)
    {
        std::vector<T> converted(source, source+n);
        Upload(converted.data(), n);
    }
    template<class U> void DownloadConverted(U* target, size_t n)
    {
        std::vector<T> converted(n);
        Download(converted.data(), n);
        std::copy(converted.begin(), converted.end(), target);
    }
    void Clear(size_t n) { Check(cudaMemset(p, 0, n*sizeof(T))); }
};
template<class T> __device__ inline T WarpSum(T value)
{
    for (int offset=16; offset; offset>>=1) value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}
}
