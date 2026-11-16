#include "Functions.h"
#include "CtfGpuCommon.h"
#include <cufft.h>

namespace
{
using namespace warp_ctf;
// Extraction, cuFFT, power accumulation and display generation are all FP32.
constexpr int Threads = 256;
// A periodogram on a fine Fourier grid only interpolates the windowed patch's autocorrelation, whose support
// is below twice the window. With small > 0, frames are transformed on that small grid and their power summed
// per patch; the summed autocorrelation is then zero-padded to the fine grid once per patch at read time.
// This is the same sum of fine-grid periodograms (exactly, up to rounding) at a fraction of the cost.
struct PowerContext
{
    int device, width, height, window, fft, small, batch, smallBatch, patches, bins, displaySize;
    cufftHandle plan = 0, smallForward = 0, smallInverse = 0;
    // Work runs on the legacy default stream, or on the context's own stream (CtfPowerUseOwnStream) so it can overlap
    // other work on the device, e.g. full-spectrum diagnostics read while the fit is refined.
    cudaStream_t stream = cudaStreamLegacy;
    bool ownStream = false;
    Buffer<float> frame, hann, padded, display;
    Buffer<float2> fourier;
    Buffer<int3> origins;
    Buffer<int> starts, indices, displayIndices, invalid;
    Buffer<float> power;
    // Small grid: frame sums and their scratch (allocated by the first frame), and resampling scratch.
    Buffer<float> accumulated, smallPadded, lags;
    Buffer<float2> smallFourier, resampleFourier;
    ~PowerContext()
    {
        if(ownStream) cudaStreamSynchronize(stream); // its work must be done before the buffers are freed
        if(plan) cufftDestroy(plan);
        if(smallForward) cufftDestroy(smallForward);
        if(smallInverse) cufftDestroy(smallInverse);
    }
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

// Power of a complex spectrum, or the real part of a resampled periodogram (clamped: the true value is nonnegative).
template<bool Resampled> __device__ inline float Power(float2 v) { return Resampled ? fmaxf(v.x,0.f) : v.x*v.x+v.y*v.y; }

template<bool Resampled>
__global__ void Bin(const float2* fourier, float* power, const int* starts, const int* indices,
                    int fftElements,int bins,int first)
{
    int bin=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32,p=blockIdx.y;
    if(bin>=bins) return;
    float sum=0;
    for(int j=starts[bin]+lane;j<starts[bin+1];j+=32)
        sum+=Power<Resampled>(fourier[(size_t)p*fftElements+indices[j]]);
    sum=WarpSum(sum);
    if(!lane) power[(size_t)(first+p)*bins+bin]+=sum;
}
template<bool Resampled>
__global__ void Display(const float2* fourier,float* output,const int* indices,int length,int fftElements,int count)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=length)return;
    float sum=output[i];
    for(int p=0;p<count;p++)sum+=Power<Resampled>(fourier[(size_t)p*fftElements+indices[i]]);
    output[i]=sum;
}
__global__ void Accumulate(const float2* fourier,float* accumulated,size_t elements,int first,int count)
{
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;if(i>=elements*count)return;
    float2 v=fourier[i];
    accumulated[(size_t)first*elements+i]+=v.x*v.x+v.y*v.y;
}
__global__ void ToComplex(const float* accumulated,float2* fourier,size_t elements)
{
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;if(i>=elements)return;
    fourier[i]=make_float2(accumulated[i],0.f);
}
// Autocorrelation lags from the small circular grid (exact, as its support is below small) onto the fine grid.
// Every call writes the same positions, so the rest of the fine grid stays zero from its single clearing.
__global__ void Embed(const float* lags,float* padded,int small,int fft,int count,float scale)
{
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,area=(size_t)small*small;if(i>=area*count)return;
    int p=(int)(i/area),k=(int)(i%area),y=k/small,x=k%small;
    if(y==small/2||x==small/2)return;
    int dy=y<small/2?y:y-small,dx=x<small/2?x:x-small;
    padded[(size_t)p*fft*fft+(size_t)((dy+fft)%fft)*fft+(dx+fft)%fft]=lags[i]*scale;
}
inline int Blocks(size_t n){return (int)((n+Threads-1)/Threads);}
inline size_t SmallElements(const PowerContext& c){return (size_t)c.small*(c.small/2+1);}

// Frame sums and their transform scratch exist only in contexts that see frames; a context that reads another's
// sums (CtfPowerReadShared) never needs them.
void EnsureAccumulation(PowerContext& c)
{
    if(c.smallForward)return; // the plan comes last: a call that failed part-way is completed by the next one
    size_t elements=SmallElements(c);
    if(!c.accumulated.p){c.accumulated.Allocate((size_t)c.patches*elements);c.accumulated.Clear((size_t)c.patches*elements);}
    // Extract writes the same window positions for every batch, so the zero padding is set once.
    if(!c.smallPadded.p){c.smallPadded.Allocate((size_t)c.smallBatch*c.small*c.small);c.smallPadded.Clear((size_t)c.smallBatch*c.small*c.small);}
    if(!c.smallFourier.p)c.smallFourier.Allocate((size_t)c.smallBatch*elements);
    int dims[2]={c.small,c.small};cufftHandle plan=0;
    if(cufftPlanMany(&plan,2,dims,nullptr,1,c.small*c.small,nullptr,1,(int)elements,CUFFT_R2C,c.smallBatch)!=CUFFT_SUCCESS)throw cudaErrorUnknown;
    if(c.ownStream){cufftSetStream(plan,c.stream);Check(cudaStreamSynchronize(cudaStreamLegacy));}
    c.smallForward=plan;
}

void AddFrame(PowerContext& c,const float* frame)
{
    if(c.small)
    {
        EnsureAccumulation(c);
        size_t elements=SmallElements(c);
        for(int first=0;first<c.patches;first+=c.smallBatch)
        {
            int count=std::min(c.smallBatch,c.patches-first);
            Extract<<<count,Threads,0,c.stream>>>(frame,c.smallPadded.p,c.hann.p,c.origins.p,c.width,c.window,c.small,first,count,c.invalid.p);
            if(cufftExecR2C(c.smallForward,c.smallPadded.p,c.smallFourier.p)!=CUFFT_SUCCESS)throw cudaErrorUnknown;
            Accumulate<<<Blocks(elements*count),Threads,0,c.stream>>>(c.smallFourier.p,c.accumulated.p,elements,first,count);
        }
    }
    else
    {
        int elements=c.fft*(c.fft/2+1);
        for(int first=0;first<c.patches;first+=c.batch)
        {
            int count=std::min(c.batch,c.patches-first);
            Extract<<<count,Threads,0,c.stream>>>(frame,c.padded.p,c.hann.p,c.origins.p,c.width,c.window,c.fft,first,count,c.invalid.p);
            if(cufftExecR2C(c.plan,c.padded.p,c.fourier.p)!=CUFFT_SUCCESS)throw cudaErrorUnknown;
            Bin<false><<<dim3((c.bins+7)/8,count),Threads,0,c.stream>>>(c.fourier.p,c.power.p,c.starts.p,c.indices.p,elements,c.bins,first);
            Display<false><<<(c.displaySize+Threads-1)/Threads,Threads,0,c.stream>>>(c.fourier.p,c.display.p,c.displayIndices.p,c.displaySize,elements,count);
        }
    }
    Check(cudaGetLastError());
}

// Resample frame-summed small-grid periodograms onto the fine grid, then bin and display them as AddFrame would.
void Resample(PowerContext& c,const float* accumulated)
{
    size_t elements=SmallElements(c);int fineElements=c.fft*(c.fft/2+1);
    float scale=1.f/((float)c.small*c.small);
    c.power.Clear((size_t)c.patches*c.bins,c.stream);c.display.Clear(c.displaySize,c.stream);
    for(int first=0;first<c.patches;first+=c.batch)
    {
        int count=std::min(c.batch,c.patches-first);
        ToComplex<<<Blocks(elements*count),Threads,0,c.stream>>>(accumulated+(size_t)first*elements,c.resampleFourier.p,elements*count);
        if(cufftExecC2R(c.smallInverse,c.resampleFourier.p,c.lags.p)!=CUFFT_SUCCESS)throw cudaErrorUnknown;
        Embed<<<Blocks((size_t)c.small*c.small*count),Threads,0,c.stream>>>(c.lags.p,c.padded.p,c.small,c.fft,count,scale);
        if(cufftExecR2C(c.plan,c.padded.p,c.fourier.p)!=CUFFT_SUCCESS)throw cudaErrorUnknown;
        Bin<true><<<dim3((c.bins+7)/8,count),Threads,0,c.stream>>>(c.fourier.p,c.power.p,c.starts.p,c.indices.p,fineElements,c.bins,first);
        Display<true><<<(c.displaySize+Threads-1)/Threads,Threads,0,c.stream>>>(c.fourier.p,c.display.p,c.displayIndices.p,c.displaySize,fineElements,count);
    }
    Check(cudaGetLastError());
}
}

extern "C" __declspec(dllexport) int CtfPowerCreate(int width,int height,int window,int fft,int small,int batch,int patches,int bins,
    const int3* origins,const float* hann,const int* starts,const int* indices,const int* displayIndices,void** result)
{
    *result=nullptr;
    try
    {
        if(width<window || height<window || window<1 || fft<window || batch<1 || patches<1 || bins<1) return cudaErrorInvalidValue;
        // Embed skips index small/2 as the (zero) lag of ±window, which holds only for an even grid of at least 2×window.
        if(small!=0 && (small<2*window || small%2 || small>=fft)) return cudaErrorInvalidValue;
        std::unique_ptr<PowerContext> c(new PowerContext());Check(cudaGetDevice(&c->device));
        c->width=width;c->height=height;c->window=window;c->fft=fft;c->small=small;c->batch=batch;c->patches=patches;c->bins=bins;c->displaySize=window*window/2;
        c->smallBatch=small?std::max(1,std::min(patches,32*1024*1024/(small*small))):1;
        c->hann.Allocate(window);c->hann.Upload(hann,window);
        c->origins.Allocate(patches);c->origins.Upload(origins,patches);
        c->starts.Allocate(bins+1);c->starts.Upload(starts,bins+1);c->indices.Allocate(starts[bins]);c->indices.Upload(indices,starts[bins]);
        c->displayIndices.Allocate(c->displaySize);c->displayIndices.Upload(displayIndices,c->displaySize);
        // Direct transforms (and Embed) write the same positions for every batch, so the zero padding is set once.
        c->padded.Allocate((size_t)batch*fft*fft);c->padded.Clear((size_t)batch*fft*fft);c->fourier.Allocate((size_t)batch*fft*(fft/2+1));
        c->power.Allocate((size_t)patches*bins);c->display.Allocate(c->displaySize);c->invalid.Allocate(1);
        int dims[2]={fft,fft};
        if(cufftPlanMany(&c->plan,2,dims,nullptr,1,fft*fft,nullptr,1,fft*(fft/2+1),CUFFT_R2C,batch)!=CUFFT_SUCCESS) return cudaErrorUnknown;
        if(small)
        {
            size_t elements=SmallElements(*c);
            c->resampleFourier.Allocate((size_t)batch*elements);c->lags.Allocate((size_t)batch*small*small);
            int smallDims[2]={small,small};
            if(cufftPlanMany(&c->smallInverse,2,smallDims,nullptr,1,(int)elements,nullptr,1,small*small,CUFFT_C2R,batch)!=CUFFT_SUCCESS) return cudaErrorUnknown;
        }
        c->display.Clear(c->displaySize);c->power.Clear((size_t)patches*bins);c->invalid.Clear(1);
        *result=c.release();return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfPowerBegin(void* context)
{
    try
    {
        auto& c=*(PowerContext*)context;DeviceScope scope(c.device);c.power.Clear((size_t)c.patches*c.bins,c.stream);c.invalid.Clear(1,c.stream);c.display.Clear(c.displaySize,c.stream);
        if(c.accumulated.p)c.accumulated.Clear((size_t)c.patches*SmallElements(c),c.stream);
        return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfPowerAdd(void* context,const float* frame)
{
    try
    {
        auto& c=*(PowerContext*)context;DeviceScope scope(c.device);
        if(!c.frame.p)c.frame.Allocate((size_t)c.width*c.height);
        if(c.ownStream)Check(cudaStreamSynchronize(c.stream)); // the previous frame is no longer read
        c.frame.Upload(frame,(size_t)c.width*c.height);AddFrame(c,c.frame.p);return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
// A frame already on the device (width × height floats), e.g. a slice of a resident movie stack.
extern "C" __declspec(dllexport) int CtfPowerAddDevice(void* context,const float* deviceFrame)
{
    try {auto& c=*(PowerContext*)context;DeviceScope scope(c.device);AddFrame(c,deviceFrame);return cudaSuccess;}
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfPowerRead(void* context,double* power,float* display)
{
    try
    {
        auto& c=*(PowerContext*)context;DeviceScope scope(c.device);int invalid;c.invalid.Download(&invalid,1,c.stream);
        if(invalid)return cudaErrorInvalidValue;
        if(c.small){EnsureAccumulation(c);Resample(c,c.accumulated.p);}
        c.power.DownloadConverted(power,(size_t)c.patches*c.bins,c.stream);c.display.Download(display,c.displaySize,c.stream);return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
// Read this context's Fourier grid from the frame sums another context made of the same patches (window, origins
// and small grid must match), e.g. full-spectrum diagnostics after band-limited fitting, without another pass over the frames.
extern "C" __declspec(dllexport) int CtfPowerReadShared(void* context,void* source,double* power,float* display)
{
    try
    {
        auto& c=*(PowerContext*)context;auto& s=*(PowerContext*)source;
        if(!c.small||s.small!=c.small||s.patches!=c.patches||s.window!=c.window||s.width!=c.width||s.height!=c.height||s.device!=c.device||!s.smallForward)
            return cudaErrorInvalidValue;
        DeviceScope scope(c.device);int invalid;s.invalid.Download(&invalid,1,c.stream);
        if(invalid)return cudaErrorInvalidValue;
        Resample(c,s.accumulated.p);
        c.power.DownloadConverted(power,(size_t)c.patches*c.bins,c.stream);c.display.Download(display,c.displaySize,c.stream);return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
// Moves the context's work to a stream of its own (non-blocking: it does not wait for the legacy default stream).
// Results are the same; other work on the device can run meanwhile. Call before the context is used.
extern "C" __declspec(dllexport) int CtfPowerUseOwnStream(void* context)
{
    try
    {
        auto& c=*(PowerContext*)context;DeviceScope scope(c.device);
        if(c.ownStream)return cudaSuccess;
        Check(cudaStreamCreateWithFlags(&c.stream,cudaStreamNonBlocking));c.ownStream=true;
        for(cufftHandle plan:{c.plan,c.smallForward,c.smallInverse})
            if(plan&&cufftSetStream(plan,c.stream)!=CUFFT_SUCCESS)return cudaErrorUnknown;
        // The buffers were cleared on the legacy stream, which this stream does not wait for.
        Check(cudaStreamSynchronize(cudaStreamLegacy));
        return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) void CtfPowerDestroy(void* context)
{
    if(!context)return;auto* c=(PowerContext*)context;int previous;cudaGetDevice(&previous);cudaSetDevice(c->device);
    cudaStream_t stream=c->ownStream?c->stream:nullptr;delete c;if(stream)cudaStreamDestroy(stream);cudaSetDevice(previous);
}
