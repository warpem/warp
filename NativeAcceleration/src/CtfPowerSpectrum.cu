#include "Functions.h"
#include "CtfGpuCommon.h"
#include <cufft.h>

namespace
{
using namespace warp_ctf;
// Extraction, cuFFT, power accumulation and display generation are all FP32.
constexpr int Threads = 256;
struct PowerContext
{
    int device, width, height, window, fft, batch, patches, bins, displaySize;
    cufftHandle plan = 0;
    Buffer<float> frame, hann, padded, display;
    Buffer<float2> fourier;
    Buffer<int3> origins;
    Buffer<int> starts, indices, displayIndices, invalid;
    Buffer<float> power;
    ~PowerContext() { if(plan) cufftDestroy(plan); }
};

__global__ void Extract(const float* frame, float* padded, const float* hann, const int3* origins,
                        int width, int window, int fft, int first, int count, int* invalid)
{
    __shared__ float sums[Threads], norms[Threads];
    int p=blockIdx.x, lane=threadIdx.x;
    if(p>=count) return;
    int3 origin=origins[first+p];
    float sum=0,norm=0;
    for(int i=lane;i<window*window;i+=Threads)
    {
        int y=i/window,x=i%window;
        float v=frame[(origin.y+y)*width+origin.x+x];
        float w=hann[x]*hann[y];
        if(!isfinite(v)) atomicExch(invalid,1);
        sum+=v*w; norm+=w;
    }
    sums[lane]=sum;norms[lane]=norm;__syncthreads();
    for(int stride=Threads/2;stride;stride>>=1)
    { if(lane<stride){sums[lane]+=sums[lane+stride];norms[lane]+=norms[lane+stride];} __syncthreads(); }
    float mean=sums[0]/norms[0]; int offset=(fft-window)/2;
    for(int i=lane;i<window*window;i+=Threads)
    {
        int y=i/window,x=i%window;
        padded[(size_t)p*fft*fft+(offset+y)*fft+offset+x]=(float)((frame[(origin.y+y)*width+origin.x+x]-mean)*hann[x]*hann[y]);
    }
}

__global__ void Bin(const float2* fourier, float* power, const int* starts, const int* indices,
                    int fftElements,int bins,int first)
{
    int bin=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32,p=blockIdx.y;
    if(bin>=bins) return;
    float sum=0;
    for(int j=starts[bin]+lane;j<starts[bin+1];j+=32)
    {
        float2 v=fourier[(size_t)p*fftElements+indices[j]];
        sum+=v.x*v.x+v.y*v.y;
    }
    sum=WarpSum(sum);
    if(!lane) power[(size_t)(first+p)*bins+bin]+=sum;
}
__global__ void Display(const float2* fourier,float* output,const int* indices,int length,int fftElements,int count)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=length)return;
    float sum=output[i];
    for(int p=0;p<count;p++){float2 v=fourier[(size_t)p*fftElements+indices[i]];sum+=v.x*v.x+v.y*v.y;}
    output[i]=sum;
}
}

extern "C" __declspec(dllexport) int CtfPowerCreate(int width,int height,int window,int fft,int batch,int patches,int bins,
    const int3* origins,const float* hann,const int* starts,const int* indices,const int* displayIndices,void** result)
{
    *result=nullptr;
    try
    {
        if(width<window || height<window || window<1 || fft<window || batch<1 || patches<1 || bins<1) return cudaErrorInvalidValue;
        std::unique_ptr<PowerContext> c(new PowerContext());Check(cudaGetDevice(&c->device));
        c->width=width;c->height=height;c->window=window;c->fft=fft;c->batch=batch;c->patches=patches;c->bins=bins;c->displaySize=window*window/2;
        c->frame.Allocate((size_t)width*height);c->hann.Allocate(window);c->hann.Upload(hann,window);
        c->origins.Allocate(patches);c->origins.Upload(origins,patches);
        c->starts.Allocate(bins+1);c->starts.Upload(starts,bins+1);c->indices.Allocate(starts[bins]);c->indices.Upload(indices,starts[bins]);
        c->displayIndices.Allocate(c->displaySize);c->displayIndices.Upload(displayIndices,c->displaySize);
        c->padded.Allocate((size_t)batch*fft*fft);c->fourier.Allocate((size_t)batch*fft*(fft/2+1));
        c->power.Allocate((size_t)patches*bins);c->display.Allocate(c->displaySize);c->invalid.Allocate(1);
        int dims[2]={fft,fft};
        if(cufftPlanMany(&c->plan,2,dims,nullptr,1,fft*fft,nullptr,1,fft*(fft/2+1),CUFFT_R2C,batch)!=CUFFT_SUCCESS) return cudaErrorUnknown;
        c->display.Clear(c->displaySize);c->power.Clear((size_t)patches*bins);c->invalid.Clear(1);
        *result=c.release();return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfPowerBegin(void* context,int resetDisplay)
{
    try {auto& c=*(PowerContext*)context;DeviceScope scope(c.device);c.power.Clear((size_t)c.patches*c.bins);c.invalid.Clear(1);if(resetDisplay)c.display.Clear(c.displaySize);return cudaSuccess;}
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfPowerAdd(void* context,const float* frame)
{
    try
    {
        auto& c=*(PowerContext*)context;DeviceScope scope(c.device);c.frame.Upload(frame,(size_t)c.width*c.height);
        int elements=c.fft*(c.fft/2+1);
        for(int first=0;first<c.patches;first+=c.batch)
        {
            int count=std::min(c.batch,c.patches-first);
            c.padded.Clear((size_t)c.batch*c.fft*c.fft);
            Extract<<<count,Threads>>>(c.frame.p,c.padded.p,c.hann.p,c.origins.p,c.width,c.window,c.fft,first,count,c.invalid.p);
            if(cufftExecR2C(c.plan,c.padded.p,c.fourier.p)!=CUFFT_SUCCESS)return cudaErrorUnknown;
            Bin<<<dim3((c.bins+7)/8,count),Threads>>>(c.fourier.p,c.power.p,c.starts.p,c.indices.p,elements,c.bins,first);
            Display<<<(c.displaySize+Threads-1)/Threads,Threads>>>(c.fourier.p,c.display.p,c.displayIndices.p,c.displaySize,elements,count);
        }
        Check(cudaGetLastError());return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfPowerRead(void* context,double* power,float* display)
{
    try
    {
        auto& c=*(PowerContext*)context;DeviceScope scope(c.device);int invalid;c.invalid.Download(&invalid,1);
        if(invalid)return cudaErrorInvalidValue;
        c.power.DownloadConverted(power,(size_t)c.patches*c.bins);c.display.Download(display,c.displaySize);return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) void CtfPowerDestroy(void* context)
{
    if(!context)return;auto* c=(PowerContext*)context;int previous;cudaGetDevice(&previous);cudaSetDevice(c->device);delete c;cudaSetDevice(previous);
}
