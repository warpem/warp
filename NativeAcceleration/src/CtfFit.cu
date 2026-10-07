#include "Functions.h"
#include "CtfGpuCommon.h"
#include <math_constants.h>
#include <type_traits>

namespace
{
using namespace warp_ctf;
// Frequency-sized buffers and the ordinary per-sample path use FP32. Only the
// small constrained spline systems use FP64; ill-conditioned FP32 Gram matrices
// are selectively reassembled in FP64 on the device without a host round trip.
constexpr int Threads=256;
constexpr double Ridge=1e-7;
struct FitView
{
    int records,samples,knots,size,features,stride;
    const float *moments,*basis,*data,*terms;
    const int *starts,*indices,*destinations,*powers,*firstBasis;
    float *baseWeights,*weights,*poses,*model,*derivative,*slabDerivative;
    const double* preciseTerms;
    int* precise;
    double *matrix,*coefficients,*output,*workspace;
};
struct FitContext
{
    int device, sharedBytes;
    FitView v;
    Buffer<float> moments,basis,data,baseWeights,terms,weights,poses,model,derivative,slabDerivative;
    Buffer<double> preciseTerms,matrix,coefficients,output,workspace;
    Buffer<int> starts,indices,destinations,powers,firstBasis,precise;
};
__device__ void Sinc(float x,float& value,float& derivative)
{
    float z=x*x;
    if(fabsf(x)<.05f){value=1-z/6+z*z/120-z*z*z/5040;derivative=-x/3+x*z/30-x*z*z/840;return;}
    float sn,co;sincosf(x,&sn,&co);value=sn/x;derivative=(x*co-sn)/z;
}
__device__ void HannPower(float x,float& value,float& derivative)
{
    if(x==0){value=1;derivative=0;return;}
    float ax=fabsf(x),pi2=CUDART_PI_F*CUDART_PI_F,z=x*x;
    // The five shifted sincs reduce to one rational factor times sin(x).
    // Use the regular sum only near its removable singularities.
    if(ax>.1f && fabsf(ax-CUDART_PI_F)>.1f && fabsf(ax-2*CUDART_PI_F)>.1f)
    {
        float sn,co;sincosf(x,&sn,&co);
        float factor=4*pi2*pi2/(x*(z-pi2)*(z-4*pi2));
        value=sn*factor;
        derivative=factor*(co-sn*(1/x+2*x/(z-pi2)+2*x/(z-4*pi2)));return;
    }
    float a,da,b,db,c,dc,d,dd,e,de;
    Sinc(x,a,da);Sinc(x-CUDART_PI_F,b,db);Sinc(x+CUDART_PI_F,c,dc);
    Sinc(x-2*CUDART_PI_F,d,dd);Sinc(x+2*CUDART_PI_F,e,de);
    value=a+(2.f/3)*(b+c)+(d+e)/6;derivative=da+(2.f/3)*(db+dc)+(dd+de)/6;
}
__global__ void Model(FitView c)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x,p=blockIdx.y;if(i>=c.samples)return;
    const float* m=c.moments+4*i;const float* x=c.poses+7*p;
    float gamma=m[0]*x[0]+m[1]*x[1]+m[2]*x[2]+m[3]+x[3],sn,co;
    sincosf(2*gamma,&sn,&co);
    float z=m[0]*m[0]*fmaxf(0,x[4]),root=sqrtf(z),slab,ds,hx,dx,hy,dy;
    Sinc(root,slab,ds);HannPower(m[0]*x[5],hx,dx);HannPower(m[0]*x[6],hy,dy);
    float dt=m[0]*m[0]*(z<.0025f?-1.f/6+z/60-z*z/1680:ds/(2*root));
    float modulation=slab*hx*hy;
    size_t index=(size_t)p*c.samples+i;
    c.model[index]=.5f-.5f*co*modulation;c.derivative[index]=sn*modulation;
    c.slabDerivative[3*index]=-.5f*co*dt*hx*hy;
    c.slabDerivative[3*index+1]=-.5f*co*slab*dx*m[0]*hy;
    c.slabDerivative[3*index+2]=-.5f*co*slab*hx*dy*m[0];
}
// One warp per sparse normal-equation entry. The design matrix has only four active
// spline functions at each frequency; its fixed products/indices are cached in CSR form.
template<bool Precise> __global__ void NormalEquations(FitView c)
{
    int feature=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32,p=blockIdx.y;
    if(feature>=c.features || (Precise && !c.precise[p]))return;
    using Accumulator=typename std::conditional<Precise,double,float>::type;
    int destination=c.destinations[feature],power=c.powers[feature];
    bool rhs=destination>=c.size*c.size;Accumulator sum=0;
    for(int j=c.starts[feature]+lane;j<c.starts[feature+1];j+=32)
    {
        size_t i=(size_t)p*c.samples+c.indices[j];
        Accumulator v=(Precise?Accumulator(c.preciseTerms[j]):Accumulator(c.terms[j]))*c.weights[i];
        if(power>0) v*=c.model[i];
        if(power>1) v*=c.model[i];
        if(rhs) v*=c.data[i];
        sum+=v;
    }
    sum=WarpSum(sum);
    if(!lane)
    {
        double value=sum;
        double* matrix=c.matrix+(size_t)p*c.stride;
        if(!rhs)
        {
            int row=destination/c.size,col=destination%c.size;
            if(row==col)value+=Ridge;
            matrix[col*c.size+row]=value;
        }
        matrix[destination]=value;
    }
}
__device__ bool SolveSubset(const double* a,const double* b,double* output,const double* active,
    double* l,double* work,double* ids,int n,bool checkCondition)
{
    int m=0;for(int i=0;i<n;i++)if(active[i])ids[m++]=i;
    double maximumDiagonal=0;
    for(int i=0;i<m;i++)maximumDiagonal=fmax(maximumDiagonal,a[(int)ids[i]*n+(int)ids[i]]);
    for(int i=0;i<m;i++)for(int j=0;j<=i;j++)
    {
        double sum=a[(int)ids[i]*n+(int)ids[j]];
        for(int k=0;k<j;k++)sum-=l[i*n+k]*l[j*n+k];
        // FP32 Gram rounding must not overwhelm the small ridge or a weak pivot.
        if(i==j && (sum<=0 || !isfinite(sum) || (checkCondition && sum<1e-4*maximumDiagonal)))return false;
        l[i*n+j]=i==j?sqrt(fmax(1e-20,sum)):sum/l[j*n+j];
    }
    for(int i=0;i<m;i++)
    {work[i]=b[(int)ids[i]];for(int j=0;j<i;j++)work[i]-=l[i*n+j]*work[j];work[i]/=l[i*n+i];}
    for(int i=m-1;i>=0;i--)
    {for(int j=i+1;j<m;j++)work[i]-=l[j*n+i]*work[j];work[i]/=l[i*n+i];}
    for(int i=0;i<n;i++)output[i]=0;
    for(int i=0;i<m;i++)output[(int)ids[i]]=work[i];
    return true;
}
// Each patch owns a small active-set solve. Double precision is retained here to
// resolve nearly dependent background/envelope columns without changing the ridge.
template<bool Precise, bool BackgroundOnly=false> __global__ void SolveNuisance(FitView c,bool sharedWorkspace)
{
    if(threadIdx.x)return;
    extern __shared__ double storage[];
    int p=blockIdx.x,n=c.size;
    if(Precise && !c.precise[p])return;
    if(!Precise)c.precise[p]=0;
    double* l=sharedWorkspace?storage:c.workspace+(size_t)p*(n*n+4*n);
    double *work=l+n*n,*ids=work+n,*active=ids+n,*z=active+n;
    const double* a=c.matrix+(size_t)p*c.stride;const double* b=a+n*n;
    double* x=c.coefficients+(size_t)p*n;
    // Reassemble a rounded, ill-conditioned Gram matrix before attempting this patch again.
#define SOLVE(target) if(!SolveSubset(a,b,target,active,l,work,ids,n,!Precise)) { \
    if(!Precise)c.precise[p]=1; else for(int i=0;i<n;i++)x[i]=CUDART_NAN; return; }
    for(int i=0;i<n;i++)active[i]=!BackgroundOnly || i<c.knots;
    SOLVE(x)
    if(BackgroundOnly)return;
    bool feasible=true;for(int i=c.knots;i<n;i++)feasible&=x[i]>=0;
    if(feasible)return;
    for(int i=c.knots;i<n;i++)active[i]=0;
    SOLVE(x)
    for(int iteration=0;iteration<8*n*n;iteration++)
    {
        int enter=-1;double best=1e-9;
        for(int i=c.knots;i<n;i++)if(!active[i])
        {double w=b[i];for(int j=0;j<n;j++)w-=a[i*n+j]*x[j];if(w>best){best=w;enter=i;}}
        if(enter<0)return;
        active[enter]=1;
        for(int inner=0;inner<=n;inner++)
        {
            SOLVE(z)
            double alpha=1;
            for(int i=c.knots;i<n;i++)if(active[i]&&z[i]<=0)alpha=fmin(alpha,x[i]/fmax(1e-30,x[i]-z[i]));
            if(alpha==1){for(int i=0;i<n;i++)x[i]=z[i];break;}
            for(int i=0;i<n;i++)x[i]+=alpha*(z[i]-x[i]);
            for(int i=c.knots;i<n;i++)if(active[i]&&x[i]<=1e-12){active[i]=0;x[i]=0;}
        }
    }
    if(!Precise)c.precise[p]=1; else for(int i=0;i<n;i++)x[i]=CUDART_NAN;
#undef SOLVE
}
// Fit smooth total power with count weights, then derive fixed inverse variances.
__global__ void InitializeWeights(FitView c)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x,p=blockIdx.y;if(i>=c.samples)return;
    float smooth=0;int first=c.firstBasis[i],last=min(c.knots,first+4);
    for(int j=first;j<last;j++)smooth+=c.basis[(size_t)i*c.knots+j]*(float)c.coefficients[(size_t)p*c.size+j];
    size_t index=(size_t)p*c.samples+i;
    smooth=fmaxf(.02f,smooth);
    float base=c.weights[index]/(smooth*smooth);
    // Negative values denote an untouched spectrum. Otherwise preserve an earlier IRLS pass.
    float previous=c.baseWeights[index];
    c.baseWeights[index]=base;c.weights[index]=previous<0?base:previous;
}
void BackgroundFit(FitContext& c)
{
    auto v=c.v;
    Check(cudaMemset(c.model.p,0,(size_t)v.records*v.samples*sizeof(float)));
    NormalEquations<false><<<dim3((v.features+7)/8,v.records),Threads>>>(v);
    SolveNuisance<false,true><<<v.records,32,c.sharedBytes>>>(v,c.sharedBytes>0);
    NormalEquations<true><<<dim3((v.features+7)/8,v.records),Threads>>>(v);
    SolveNuisance<true,true><<<v.records,32,c.sharedBytes>>>(v,c.sharedBytes>0);
}
__global__ void BackgroundResidual(FitView c)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x,p=blockIdx.y;if(i>=c.samples)return;
    float bg=0;int first=c.firstBasis[i],last=min(c.knots,first+4);
    for(int j=first;j<last;j++)bg+=c.basis[(size_t)i*c.knots+j]*(float)c.coefficients[(size_t)p*c.size+j];
    size_t index=(size_t)p*c.samples+i;c.derivative[index]=c.data[index]-bg;
}
__global__ void RankTrials(FitView c,const float* trials,const float* offsets,double* scores)
{
    __shared__ float cross[Threads],power[Threads];
    int trial=blockIdx.x,p=blockIdx.y,lane=threadIdx.x;float sum=0,norm=0;
    float df=trials[2*trial]+offsets[p],phase=trials[2*trial+1];
    for(int i=lane;i<c.samples;i+=Threads)
    {
        size_t index=(size_t)p*c.samples+i;
        float model=-.5f*cosf(2*(c.moments[4*i]*df+c.moments[4*i+3]+phase)),w=c.weights[index];
        sum+=w*c.derivative[index]*model;norm+=w*model*model;
    }
    cross[lane]=sum;power[lane]=norm;__syncthreads();
    for(int stride=Threads/2;stride;stride>>=1)
    {if(lane<stride){cross[lane]+=cross[lane+stride];power[lane]+=power[lane+stride];}__syncthreads();}
    if(!lane)scores[(size_t)trial*c.records+p]=cross[0]/sqrtf(fmaxf(1e-30f,power[0]));
}
__global__ void Score(FitView c,bool reweight)
{
    __shared__ float reduction[9][Threads];
    int p=blockIdx.x,lane=threadIdx.x;float sums[9]={0,0,0,0,0,0,0,0,0};
    const double* coefficients=c.coefficients+(size_t)p*c.size;
    for(int i=lane;i<c.samples;i+=Threads)
    {
        size_t index=(size_t)p*c.samples+i;float a=0,e=0;
        int first=c.firstBasis[i],last=min(c.knots,first+4);
        for(int j=first;j<last;j++){float b=c.basis[(size_t)i*c.knots+j];a+=b*(float)coefficients[j];e+=b*(float)coefficients[j+c.knots];}
        float residual=a+e*c.model[index]-c.data[index],w=c.weights[index];
        sums[0]+=.5f*w*residual*residual;
        float derivative=w*residual*e*c.derivative[index];
        for(int j=0;j<3;j++)sums[j+1]+=derivative*c.moments[4*i+j];sums[4]+=derivative;
        for(int j=0;j<3;j++)sums[5+j]+=w*residual*e*c.slabDerivative[3*index+j];
        if(reweight)
        {
            float base=c.baseWeights[index],factor=fminf(1.f,9.f/(8.f+residual*residual*base));
            sums[8]=fmaxf(sums[8],fabsf(factor-w/base));c.weights[index]=base*factor;
        }
    }
    for(int j=lane;j<c.size;j+=Threads)sums[0]+=.5*Ridge*coefficients[j]*coefficients[j];
    for(int k=0;k<9;k++)reduction[k][lane]=sums[k];__syncthreads();
    for(int stride=Threads/2;stride;stride>>=1)
    {
        if(lane<stride){for(int k=0;k<8;k++)reduction[k][lane]+=reduction[k][lane+stride];reduction[8][lane]=fmaxf(reduction[8][lane],reduction[8][lane+stride]);}
        __syncthreads();
    }
    if(lane<9)c.output[p*9+lane]=reduction[lane][0];
}
}

extern "C" __declspec(dllexport) int CtfFitCreate(int records,int samples,int knots,const double* moments,const double* basis,
    const double* data,const double* counts,const double* currentWeights,void** result)
{
    *result=nullptr;
    try
    {
        if(records<1||samples<16||knots<4)return cudaErrorInvalidValue;
        std::unique_ptr<FitContext> c(new FitContext());Check(cudaGetDevice(&c->device));auto& v=c->v;
        v.records=records;v.samples=samples;v.knots=knots;v.size=knots*2;v.stride=v.size*v.size+v.size;
        std::vector<int> starts(1,0),indices,destinations,powers,first(samples,0);std::vector<double> terms;
        auto Feature=[&](int j,int k,int destination,int power)
        {
            destinations.push_back(destination);powers.push_back(power);
            for(int i=0;i<samples;i++)
            {
                // Exact products of the FP32 design coefficients preserve positive definiteness
                // in the selective FP64 assembly. A rounded product would defeat the retry.
                double value=(float)basis[(size_t)i*knots+j];if(k>=0)value*=(float)basis[(size_t)i*knots+k];
                if(value!=0){indices.push_back(i);terms.push_back(value);}
            }
            starts.push_back((int)terms.size());
        };
        for(int j=0;j<v.size;j++)for(int k=0;k<=j;k++)Feature(j%knots,k%knots,j*v.size+k,(j>=knots)+(k>=knots));
        for(int j=0;j<v.size;j++)Feature(j%knots,-1,v.size*v.size+j,j>=knots);
        for(int i=0;i<samples;i++)while(first[i]<knots-1&&basis[(size_t)i*knots+first[i]]==0)first[i]++;
        v.features=(int)destinations.size();
#define UPLOAD(member,source,count) c->member.Allocate(count);c->member.UploadConverted(source,count);v.member=c->member.p
        UPLOAD(moments,moments,(size_t)samples*4);UPLOAD(basis,basis,(size_t)samples*knots);
        UPLOAD(data,data,(size_t)records*samples);UPLOAD(weights,counts,(size_t)records*samples);UPLOAD(baseWeights,currentWeights,(size_t)records*samples);
        UPLOAD(starts,starts.data(),starts.size());UPLOAD(indices,indices.data(),indices.size());UPLOAD(terms,terms.data(),terms.size());UPLOAD(preciseTerms,terms.data(),terms.size());
        UPLOAD(destinations,destinations.data(),destinations.size());UPLOAD(powers,powers.data(),powers.size());UPLOAD(firstBasis,first.data(),first.size());
#undef UPLOAD
#define ALLOC(member,count) c->member.Allocate(count);v.member=c->member.p
        ALLOC(precise,records);
        ALLOC(poses,(size_t)records*7);ALLOC(model,(size_t)records*samples);ALLOC(derivative,(size_t)records*samples);ALLOC(slabDerivative,(size_t)records*samples*3);
        ALLOC(matrix,(size_t)records*v.stride);ALLOC(coefficients,(size_t)records*v.size);ALLOC(output,(size_t)records*9);
        size_t bytes=(size_t)(v.size*v.size+4*v.size)*sizeof(double);
        c->sharedBytes=bytes<=48*1024?(int)bytes:0;
        v.workspace=nullptr;if(!c->sharedBytes){ALLOC(workspace,(size_t)records*(v.size*v.size+4*v.size));}
#undef ALLOC
        BackgroundFit(*c);
        InitializeWeights<<<dim3((samples+Threads-1)/Threads,records),Threads>>>(v);
        Check(cudaGetLastError());
        *result=c.release();return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfFitSearch(void* context,const double* trials,const double* offsets,int trialCount,double* scores)
{
    try
    {
        if(!context || trialCount<1)return cudaErrorInvalidValue;
        auto& c=*(FitContext*)context;DeviceScope scope(c.device);auto v=c.v;
        Buffer<float> deviceTrials,deviceOffsets;Buffer<double> deviceScores;
        deviceTrials.Allocate((size_t)trialCount*2);deviceTrials.UploadConverted(trials,(size_t)trialCount*2);
        deviceOffsets.Allocate(v.records);deviceOffsets.UploadConverted(offsets,v.records);
        deviceScores.Allocate((size_t)trialCount*v.records);
        BackgroundFit(c);
        BackgroundResidual<<<dim3((v.samples+Threads-1)/Threads,v.records),Threads>>>(v);
        RankTrials<<<dim3(trialCount,v.records),Threads>>>(v,deviceTrials.p,deviceOffsets.p,deviceScores.p);
        Check(cudaGetLastError());deviceScores.Download(scores,(size_t)trialCount*v.records);return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfFitEvaluate(void* context,const double* poses,int reweight,double* output)
{
    try
    {
        auto& c=*(FitContext*)context;DeviceScope scope(c.device);auto v=c.v;c.poses.UploadConverted(poses,(size_t)v.records*7);
        Model<<<dim3((v.samples+Threads-1)/Threads,v.records),Threads>>>(v);
        NormalEquations<false><<<dim3((v.features+7)/8,v.records),Threads>>>(v);
        SolveNuisance<false><<<v.records,32,c.sharedBytes>>>(v,c.sharedBytes>0);
        NormalEquations<true><<<dim3((v.features+7)/8,v.records),Threads>>>(v);
        SolveNuisance<true><<<v.records,32,c.sharedBytes>>>(v,c.sharedBytes>0);
        Score<<<v.records,Threads>>>(v,reweight!=0);Check(cudaGetLastError());
        c.output.Download(output,(size_t)v.records*9);return cudaSuccess;
    }
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfFitReadWeights(void* context,double* weights)
{
    try{auto& c=*(FitContext*)context;DeviceScope scope(c.device);c.weights.DownloadConverted(weights,(size_t)c.v.records*c.v.samples);return cudaSuccess;}
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) int CtfFitReadCoefficients(void* context,float* coefficients)
{
    try{auto& c=*(FitContext*)context;DeviceScope scope(c.device);c.coefficients.DownloadConverted(coefficients,(size_t)c.v.records*c.v.size);return cudaSuccess;}
    catch(cudaError_t e){return e;}catch(...){return cudaErrorUnknown;}
}
extern "C" __declspec(dllexport) void CtfFitDestroy(void* context)
{
    if(!context)return;auto* c=(FitContext*)context;int previous;cudaGetDevice(&previous);cudaSetDevice(c->device);delete c;cudaSetDevice(previous);
}
