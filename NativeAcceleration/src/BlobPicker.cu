#include "include/BlobPicker.h"
#include <cufft.h>
#include <math_constants.h>
#include <climits>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>

namespace {
constexpr int Threads = 256;
void Check(cudaError_t status) { if (status != cudaSuccess) throw int(status); }
void FFTCheck(cufftResult status) { if (status != CUFFT_SUCCESS) throw -3; }
struct Plan {
    cufftHandle handle = 0;
    ~Plan() { if (handle) cufftDestroy(handle); }
};
template<class T> T* Ptr(thrust::device_vector<T>& v) { return thrust::raw_pointer_cast(v.data()); }
struct Finite { __host__ __device__ int operator()(float x) const { return isfinite(x) ? 0 : 1; } };
struct Double { __host__ __device__ double operator()(float x) const { return double(x); } };

__global__ void Prepare(const float* input, float* centered, float* squared,
                        float* support, float* blob, int2 dims, float outer2, float inner2,
                        float mean, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    float v = input[id] - mean;
    centered[id] = v;
    squared[id] = v * v;
    int x = id % dims.x, y = id / dims.x;
    if (x > dims.x / 2) x -= dims.x;
    if (y > dims.y / 2) y -= dims.y;
    float r2 = float(x) * x + float(y) * y;
    support[id] = r2 <= 4 * outer2 ? 1.f : 0.f;
    blob[id] = r2 <= outer2 && (inner2 == 0 || r2 >= inner2) ? 1.f : 0.f;
}
__global__ void Template(float* blob, const float* support, float fraction, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) blob[id] = fraction * support[id] - blob[id];
}
__global__ void Product(const cufftComplex* a, const cufftComplex* b,
                        cufftComplex* result, float scale, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        cufftComplex x = a[id], y = b[id];
        result[id] = make_float2((x.x*y.x + x.y*y.y)*scale, (x.y*y.x - x.x*y.y)*scale);
    }
}
__global__ void Normalize(float* correlation, float* power, const float* mean,
                          int2 dims, int margin, float samples, float templatePower, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int x = id % dims.x, y = id / dims.x;
    float variance = fmaxf(0.f, power[id] / samples - (mean[id]/samples)*(mean[id]/samples));
    float score = correlation[id] / sqrtf(samples * variance * templatePower);
    bool valid = x >= margin && y >= margin && x < dims.x-margin && y < dims.y-margin &&
                 variance > 0 && isfinite(score);
    correlation[id] = valid ? fminf(1.f, fmaxf(-1.f, score)) : -CUDART_INF_F;
    power[id] = valid ? variance : 0.f;
}
struct Peak {
    const float* score;
    int2 dims;
    float separation2;
    int extent;
    __device__ bool operator()(int id) const {
        float v = score[id];
        if (!isfinite(v) || v <= 0) return false;
        int x = id % dims.x, y = id / dims.x;
        for (int yy = max(0,y-extent); yy <= min(dims.y-1,y+extent); ++yy)
            for (int xx = max(0,x-extent); xx <= min(dims.x-1,x+extent); ++xx) {
                float dx = xx-x, dy = yy-y;
                if (dx*dx + dy*dy > separation2) continue;
                int other = yy*dims.x+xx;
                float w = score[other];
                if (isfinite(w) && (w > v || (w == v && other < id))) return false;
            }
        return true;
    }
};
__global__ void Gather(const int* ids, const float* score, const float* power,
                       float4* result, int width, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { int id = ids[i]; result[i] = make_float4(id%width,id/width,score[id],power[id]); }
}
bool ValidDims(int2 dims) { return dims.x > 0 && dims.y > 0 && size_t(dims.x)*dims.y <= INT_MAX; }
}

int BlobPickerScore(const float* input, int2 dims, float radius, float innerRadius,
                    float* correlation, float* power) {
    if (!input || !correlation || !power || !ValidDims(dims) || !std::isfinite(radius) ||
        !std::isfinite(innerRadius) || radius <= 0 || innerRadius < 0 || innerRadius >= radius ||
        2*std::ceil(2*radius) >= std::min(dims.x,dims.y)) return -1;
    try {
        int n = dims.x*dims.y, nf = dims.y*(dims.x/2+1), blocks = (n+Threads-1)/Threads;
        auto begin = thrust::device_pointer_cast(input);
        if (thrust::transform_reduce(begin, begin+n, Finite(), 0, thrust::plus<int>())) return -1;
        float globalMean = float(thrust::transform_reduce(begin, begin+n, Double(), 0.0, thrust::plus<double>())/n);
        thrust::device_vector<float> centered(n), squared(n), support(n), blob(n), mean(n);
        Prepare<<<blocks,Threads>>>(input,Ptr(centered),Ptr(squared),Ptr(support),Ptr(blob),dims,
                                    radius*radius,innerRadius*innerRadius,globalMean,n);
        double samples = thrust::transform_reduce(support.begin(),support.end(),Double(),0.0,thrust::plus<double>());
        double signal = thrust::transform_reduce(blob.begin(),blob.end(),Double(),0.0,thrust::plus<double>());
        if (signal <= 0 || signal >= samples) return -1;
        Template<<<blocks,Threads>>>(Ptr(blob),Ptr(support),float(signal/samples),n);
        thrust::device_vector<cufftComplex> imageFT(nf), supportFT(nf), blobFT(nf), workFT(nf);
        Plan forward, backward;
        FFTCheck(cufftPlan2d(&forward.handle,dims.y,dims.x,CUFFT_R2C));
        FFTCheck(cufftPlan2d(&backward.handle,dims.y,dims.x,CUFFT_C2R));
        FFTCheck(cufftExecR2C(forward.handle,Ptr(centered),Ptr(imageFT)));
        FFTCheck(cufftExecR2C(forward.handle,Ptr(support),Ptr(supportFT)));
        FFTCheck(cufftExecR2C(forward.handle,Ptr(blob),Ptr(blobFT)));
        auto convolve = [&](cufftComplex* a, cufftComplex* b, float* output) {
            Product<<<(nf+Threads-1)/Threads,Threads>>>(a,b,Ptr(workFT),1.f/n,nf);
            FFTCheck(cufftExecC2R(backward.handle,Ptr(workFT),output));
        };
        convolve(Ptr(imageFT),Ptr(blobFT),correlation);
        convolve(Ptr(imageFT),Ptr(supportFT),Ptr(mean));
        FFTCheck(cufftExecR2C(forward.handle,Ptr(squared),Ptr(imageFT)));
        convolve(Ptr(imageFT),Ptr(supportFT),power);
        Normalize<<<blocks,Threads>>>(correlation,power,Ptr(mean),dims,int(std::ceil(2*radius)),
                                      float(samples),float(signal-signal*signal/samples),n);
        Check(cudaGetLastError());
        Check(cudaDeviceSynchronize());
        return 0;
    } catch (int status) { return status; }
      catch (const std::bad_alloc&) { return -2; }
      catch (...) { return -2; }
}

int BlobPickerPeaks(const float* correlation, const float* power, int2 dims, float separation,
                    float4** picks, int* count) {
    if (!picks || !count) return -1;
    *picks = nullptr; *count = 0;
    if (!correlation || !power || !ValidDims(dims) || !std::isfinite(separation) || separation <= 0) return -1;
    try {
        int n = dims.x*dims.y;
        thrust::device_vector<int> ids(n);
        auto begin = thrust::make_counting_iterator(0);
        int extent = int(std::ceil(std::min(double(separation),double(std::max(dims.x,dims.y)))));
        auto end = thrust::copy_if(begin,begin+n,ids.begin(),Peak{correlation,dims,separation*separation,extent});
        int found = int(end-ids.begin());
        if (!found) return 0;
        thrust::device_vector<float4> packed(found);
        Gather<<<(found+Threads-1)/Threads,Threads>>>(Ptr(ids),correlation,power,Ptr(packed),dims.x,found);
        Check(cudaGetLastError());
        float4* result = static_cast<float4*>(std::malloc(size_t(found)*sizeof(float4)));
        if (!result) return -2;
        cudaError_t status = cudaMemcpy(result,Ptr(packed),size_t(found)*sizeof(float4),cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) { std::free(result); return int(status); }
        *picks = result; *count = found;
        return 0;
    } catch (int status) { return status; }
      catch (...) { return -2; }
}
