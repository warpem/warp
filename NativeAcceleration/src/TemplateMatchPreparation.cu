#include "Functions.h"

namespace {
__device__ float CtfAt(const float* c,int n,int x,int y) {
    if(x<0){x=-x;y=-y;}if(x>n/2)return 0;
    return c[((y%n+n)%n)*(n/2+1)+x];
}
__global__ void Transfer(const float* c,float* o,int n,int tilts,const float* r,const float* noise) {
    size_t count=size_t(n)*(n/2+1)*n, slice=size_t(n)*(n/2+1);
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<count;i+=gridDim.x*size_t(blockDim.x)) {
        int x=i%(n/2+1),y=(i/(n/2+1))%n,z=i/(size_t(n)*(n/2+1));
        if(y>n/2)y-=n;if(z>n/2)z-=n;float sum=0;
        for(int t=0;t<tilts;t++) {
            const float* a=r+t*9;
            float u=a[0]*x+a[3]*y+a[6]*z,v=a[1]*x+a[4]*y+a[7]*z,w=a[2]*x+a[5]*y+a[8]*z;
            float hat=fmaxf(0,1-fabsf(w));if(hat==0)continue;
            int ix=floorf(u),iy=floorf(v);float fx=u-ix,fy=v-iy;const float* ct=c+t*slice;
            float h=(1-fy)*((1-fx)*CtfAt(ct,n,ix,iy)+fx*CtfAt(ct,n,ix+1,iy))+
                fy*((1-fx)*CtfAt(ct,n,ix,iy+1)+fx*CtfAt(ct,n,ix+1,iy+1));
            sum+=hat*h*h*noise[t];
        }
        o[i]=(x==0 && y==0 && z==0) ? 0 : sum;
    }
}
__global__ void Hybrid(float* w, int box, int particles, int tilts, const float* r,
                       const float* noise, float pixel, float diameter)
{
    size_t f = size_t(box) * (box / 2 + 1), total = f * particles * tilts;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += gridDim.x * size_t(blockDim.x)) {
        if (!(w[i] > 0)) continue;
        int view = int(i / f), t = view % tilts, p = view / tilts;
        int x = int(i % f) % (box / 2 + 1), y = int(i % f) / (box / 2 + 1);
        if (y > box / 2) y -= box;
        if (x == 0 && y == 0) { w[i] = 0; continue; }
        const float* a = r + view * 9; // column-major image-to-specimen rotation
        float kx = (a[0]*x + a[3]*y)/(box*pixel);
        float ky = (a[1]*x + a[4]*y)/(box*pixel);
        float kz = (a[2]*x + a[5]*y)/(box*pixel);
        float c = 0;
        for (int u = 0; u < tilts; ++u) {
            const float* b = r + (p*tilts+u)*9;
            // Missing views have an all-zero rotation; never count them as evidence.
            if (b[6]*b[6]+b[7]*b[7]+b[8]*b[8] < .5f) continue;
            c += fmaxf(0, 1-diameter*fabsf(kx*b[6]+ky*b[7]+kz*b[8]));
        }
        float P = 2 / w[i], N = noise[t];
        w[i] = 2 / fmaxf(1e-30f, N + fmaxf(1,c)*fmaxf(0,P-N));
    }
}
__global__ void Window(float* a, int3 d, float scale) {
    size_t n = size_t(d.x)*d.y*d.z;
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<n;i+=gridDim.x*size_t(blockDim.x)) {
        int x=i%d.x,y=(i/d.x)%d.y,z=i/(size_t(d.x)*d.y);
        x=min(x,d.x-x); y=min(y,d.y-y); z=min(z,d.z-z);
        a[i] *= expf(-scale*(float(x)*x+float(y)*y+float(z)*z));
    }
}
__device__ float At(const float* a,int3 d,int x,int y,int z) {
    if(x<0) {x=-x;y=-y;z=-z;}
    x=min(x,d.x/2);y=(y%d.y+d.y)%d.y;z=(z%d.z+d.z)%d.z;
    return a[(size_t(z)*d.y+y)*(d.x/2+1)+x];
}
__global__ void Resample(const float* a,int3 d,float* o,int n) {
    size_t count=size_t(n)*(n/2+1)*n;
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<count;i+=gridDim.x*size_t(blockDim.x)) {
        int x=i%(n/2+1),y=(i/(n/2+1))%n,z=i/(size_t(n)*(n/2+1));
        if(y>n/2)y-=n;if(z>n/2)z-=n;
        float fx=float(x)*d.x/n,fy=float(y)*d.y/n,fz=float(z)*d.z/n;
        int ix=floorf(fx),iy=floorf(fy),iz=floorf(fz);fx-=ix;fy-=iy;fz-=iz;
        float v=0;
        for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int h=0;h<2;++h)
            v+=At(a,d,ix+h,iy+j,iz+k)*(h?fx:1-fx)*(j?fy:1-fy)*(k?fz:1-fz);
        o[i]=v;
    }
}
__global__ void Backproject(const float* image,int2 im,float* volume,int3 d,
                            const float3* geom,int3 grid,float spacing,float df,float step) {
    size_t n=size_t(d.x)*d.y*d.z;
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<n;i+=gridDim.x*size_t(blockDim.x)) {
        int x=i%d.x,y=(i/d.x)%d.y,z=i/(size_t(d.x)*d.y);
        float fx=x/spacing,fy=y/spacing,fz=z/spacing;
        int ix=min(int(fx),grid.x-2),iy=min(int(fy),grid.y-2),iz=min(int(fz),grid.z-2);
        fx-=ix;fy-=iy;fz-=iz;float3 pos=make_float3(0,0,0);
        for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int h=0;h<2;++h) {
            float w=(h?fx:1-fx)*(j?fy:1-fy)*(k?fz:1-fz);
            float3 g=geom[(size_t(iz+k)*grid.y+iy+j)*grid.x+ix+h];
            pos.x+=w*g.x;pos.y+=w*g.y;pos.z+=w*g.z;
        }
        float w=fmaxf(0,1-fabsf(pos.z-df)/step);if(w==0)continue;
        if(pos.x<0||pos.y<0||pos.x>=im.x-1||pos.y>=im.y-1)continue;
        int u=int(pos.x),v=int(pos.y);float a=pos.x-u,b=pos.y-v;
        volume[i]+=w*((1-b)*((1-a)*image[v*im.x+u]+a*image[v*im.x+u+1])+
                       b*((1-a)*image[(v+1)*im.x+u]+a*image[(v+1)*im.x+u+1]));
    }
}
}
extern "C" __declspec(dllexport) void MatchTransfer(const float* c,float* o,int n,int tilts,const float* rotations,const float* noise) {
    float *r,*w;cudaMalloc(&r,tilts*9*sizeof(float));cudaMalloc(&w,tilts*sizeof(float));
    cudaMemcpy(r,rotations,tilts*9*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(w,noise,tilts*sizeof(float),cudaMemcpyHostToDevice);
    Transfer<<<512,128>>>(c,o,n,tilts,r,w);cudaFree(r);cudaFree(w);
}
extern "C" __declspec(dllexport) void MatchHybridWeights(float* w,int box,int particles,int tilts,
    const float* rotations,const float* noise,float pixel,float diameter) {
    float *r,*n;cudaMalloc(&r,size_t(particles)*tilts*9*sizeof(float));cudaMalloc(&n,tilts*sizeof(float));
    cudaMemcpy(r,rotations,size_t(particles)*tilts*9*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(n,noise,tilts*sizeof(float),cudaMemcpyHostToDevice);
    Hybrid<<<512,128>>>(w,box,particles,tilts,r,n,pixel,diameter);cudaFree(r);cudaFree(n);
}
extern "C" __declspec(dllexport) void MatchAutocorrelationWindow(float* a,int3 d,float pixel,float length) {
    Window<<<512,128>>>(a,d,.5f*pixel*pixel/(length*length));
}
extern "C" __declspec(dllexport) void MatchSpectrumResample(const float* a,int3 d,float* o,int n) {
    Resample<<<512,128>>>(a,d,o,n);
}
extern "C" __declspec(dllexport) void MatchBackproject(const float* image,int2 im,float* volume,int3 d,
    const float3* geom,int3 grid,float spacing,float df,float step) {
    Backproject<<<512,128>>>(image,im,volume,d,geom,grid,spacing,df,step);
}
