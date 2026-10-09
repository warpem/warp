// Shared spectral shape with one nonnegative scalar per patch. Backgrounds remain
// local. Eliminate backgrounds once per pose, then solve the small reduced problem;
// no sample-sized work is repeated during alternating envelope/amplitude updates.
template<bool Precise> __global__ void EliminateBackground(FitView c)
{
    int p=blockIdx.x,k=c.knots,n=c.size;if(threadIdx.x || (Precise&&!c.precise[p]))return;
    if(!Precise)c.precise[p]=0;
    double* l=c.sharedWork+(size_t)p*(k*k+4*k);
    double* work=l+k*k;double* active=work+k;double* ids=active+k;double* scratch=ids+k;
    const double* a=c.matrix+(size_t)p*c.stride;const double* b=a+n*n;
    double* solved=c.backgroundSolutions+(size_t)p*k*(k+1);
    double* reduced=c.reduced+(size_t)p*k*(k+1);
    double maxdiag=0;for(int i=0;i<k;i++)maxdiag=fmax(maxdiag,a[i*n+i]);
    for(int i=0;i<k;i++)for(int j=0;j<=i;j++)
    {
        double v=a[i*n+j];for(int h=0;h<j;h++)v-=l[i*k+h]*l[j*k+h];
        if(i==j && (!(v>0)||(!Precise&&v<1e-4*maxdiag)))
        {c.precise[p]=1;reduced[0]=CUDART_NAN;return;}
        l[i*k+j]=i==j?sqrt(v):v/l[j*k+j];
    }
    for(int column=0;column<=k;column++)
    {
        for(int i=0;i<k;i++)
        {work[i]=column?a[i*n+k+column-1]:b[i];for(int j=0;j<i;j++)work[i]-=l[i*k+j]*work[j];work[i]/=l[i*k+i];}
        for(int i=k-1;i>=0;i--)
        {for(int j=i+1;j<k;j++)work[i]-=l[j*k+i]*work[j];work[i]/=l[i*k+i];}
        for(int i=0;i<k;i++)solved[column*k+i]=work[i];
    }
    for(int i=0;i<k;i++)
    {
        double rhs=b[k+i];for(int h=0;h<k;h++)rhs-=a[(k+i)*n+h]*solved[h];reduced[k*k+i]=rhs;
        for(int j=0;j<=i;j++)
        {
            double v=a[(k+i)*n+k+j];for(int h=0;h<k;h++)v-=a[(k+i)*n+h]*solved[(j+1)*k+h];
            reduced[i*k+j]=reduced[j*k+i]=v;
        }
    }
    for(int i=0;i<k;i++)active[i]=1;
    if(!SolveSubset(reduced,reduced+k*k,scratch,active,l,work,ids,k,!Precise))
    {c.precise[p]=1;reduced[0]=CUDART_NAN;}
}
__global__ void SharedNormal(FitView c)
{
    int group=blockIdx.y,n=c.anchors*c.knots,feature=blockIdx.x*blockDim.x+threadIdx.x;
    if(feature>=n*n+n)return;
    bool rhs=feature>=n*n;int i=rhs?feature-n*n:feature/n,j=rhs?0:feature%n;
    int ki=i%c.knots,kj=j%c.knots;double sum=0;
    for(int entry=c.groupStarts[group];entry<c.groupStarts[group+1];entry++)
    {
        int p=c.groupRecords[entry];double amplitude=c.amplitudes[p];
        double wi=c.blends[p*c.anchors+i/c.knots];
        const double* reduced=c.reduced+(size_t)p*c.knots*(c.knots+1);
        sum+=rhs?wi*amplitude*reduced[c.knots*c.knots+ki]:wi*c.blends[p*c.anchors+j/c.knots]*amplitude*amplitude*reduced[ki*c.knots+kj];
    }
    // Unobserved anchors have zero RHS. This numerical floor does not impose a
    // spectral penalty; the physical coefficient ridge is already in the Schur system.
    if(!rhs&&i==j&&sum==0)sum=1;
    c.envelopeMatrix[(size_t)group*(n*n+n)+feature]=sum;
}
__global__ void SolveShared(FitView c)
{
    int group=blockIdx.x,n=c.anchors*c.knots;if(threadIdx.x)return;
    double* a=c.envelopeMatrix+(size_t)group*(n*n+n);double* b=a+n*n;
    double* x=c.envelopes+(size_t)group*n;
    double* l=c.envelopeWork+(size_t)group*(n*n+4*n);
    double *work=l+n*n,*ids=work+n,*active=ids+n,*z=active+n;
    // The previous inner update usually has the correct active set. Reuse it
    // instead of rebuilding the factorization once for every positive knot.
    for(int i=0;i<n;i++)active[i]=x[i]>0;
    for(int iteration=0;iteration<8*n*n;iteration++)
    {
        if(iteration>0)
        {
            int enter=-1;double best=1e-10;
            for(int i=0;i<n;i++)if(!active[i])
            {double w=b[i];for(int j=0;j<n;j++)w-=a[i*n+j]*x[j];if(w>best){best=w;enter=i;}}
            if(enter<0)return;active[enter]=1;
        }
        for(int inner=0;inner<=n;inner++)
        {
            if(!SolveSubset(a,b,z,active,l,work,ids,n,false)){x[0]=CUDART_NAN;return;}
            double alpha=1;
            for(int i=0;i<n;i++)if(active[i]&&z[i]<=0)alpha=fmin(alpha,x[i]/fmax(1e-30,x[i]-z[i]));
            if(alpha==1){for(int i=0;i<n;i++)x[i]=z[i];break;}
            for(int i=0;i<n;i++)x[i]+=alpha*(z[i]-x[i]);
            for(int i=0;i<n;i++)if(active[i]&&x[i]<=1e-12){active[i]=0;x[i]=0;}
        }
    }
    x[0]=CUDART_NAN;
}
__global__ void UpdateAmplitudes(FitView c)
{
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=c.records)return;
    int k=c.knots,group=c.envelopeGroups[p];
    const double* e=c.envelopes+(size_t)group*c.anchors*k;
    const double* reduced=c.reduced+(size_t)p*k*(k+1);
    double* coefficients=c.coefficients+(size_t)p*c.size;
    double* shape=c.sharedWork+(size_t)p*(k*k+4*k);
    for(int i=0;i<k;i++)
    {double v=0;for(int a=0;a<c.anchors;a++)v+=c.blends[p*c.anchors+a]*e[a*k+i];shape[i]=v;}
    double numerator=0,denominator=0;
    for(int i=0;i<k;i++)
    {numerator+=shape[i]*reduced[k*k+i];for(int j=0;j<k;j++)denominator+=shape[i]*reduced[i*k+j]*shape[j];}
    double amplitude=denominator>1e-20?fmax(0.,numerator/denominator):0;
    for(int i=0;i<k;i++)
    {
        coefficients[k+i]=amplitude*shape[i];
    }
    c.amplitudes[p]=amplitude;
    c.reducedLoss[p]=-.5*amplitude*numerator;
    const double* solved=c.backgroundSolutions+(size_t)p*k*(k+1);
    for(int i=0;i<k;i++)
    {double v=solved[i];for(int j=0;j<k;j++)v-=solved[(j+1)*k+i]*coefficients[k+j];coefficients[i]=v;}
}
__global__ void NormalizeAmplitudes(FitView c)
{
    if(threadIdx.x)return;int g=blockIdx.x;double sum=0,loss=0;
    int start=c.groupStarts[g],end=c.groupStarts[g+1];
    for(int j=start;j<end;j++){int p=c.groupRecords[j];sum+=c.amplitudes[p];loss+=c.reducedLoss[p];}
    double mean=sum/fmax(1.,(double)(end-start));
    if(mean>1e-20)
    {
        for(int j=start;j<end;j++)c.amplitudes[c.groupRecords[j]]/=mean;
        for(int i=0;i<c.anchors*c.knots;i++)c.envelopes[g*c.anchors*c.knots+i]*=mean;
    }
    c.objective[g]=loss;
}
// Squared extrapolation of two normalized alternating updates. Accept it only
// when the profiled reduced objective improves; otherwise resume the ordinary
// update. This avoids very slow amplitude/shape trades in nearly collinear data.
__global__ void ExtrapolateAmplitudes(FitView c)
{
    if(threadIdx.x)return;int g=blockIdx.x,start=c.groupStarts[g],end=c.groupStarts[g+1];
    double rr=0,vv=0;
    for(int j=start;j<end;j++)
    {int p=c.groupRecords[j];double r=c.amplitude1[p]-c.amplitude0[p],v=c.amplitude2[p]-2*c.amplitude1[p]+c.amplitude0[p];rr+=r*r;vv+=v*v;}
    double alpha=-fmin(100.,fmax(1.,sqrt(rr/fmax(1e-30,vv)))),sum=0;
    for(int j=start;j<end;j++)
    {
        int p=c.groupRecords[j];double r=c.amplitude1[p]-c.amplitude0[p],v=c.amplitude2[p]-2*c.amplitude1[p]+c.amplitude0[p];
        c.amplitudes[p]=fmax(0.,c.amplitude0[p]-2*alpha*r+alpha*alpha*v);sum+=c.amplitudes[p];
    }
    if(sum>1e-20)for(int j=start;j<end;j++)c.amplitudes[c.groupRecords[j]]*=(end-start)/sum;
    else for(int j=start;j<end;j++)c.amplitudes[c.groupRecords[j]]=c.amplitude2[c.groupRecords[j]];
}
__global__ void RejectExtrapolation(FitView c)
{
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=c.records)return;int g=c.envelopeGroups[p];
    if(!isfinite(c.objective[g]) || c.objective[g]>c.previousObjective[g]+1e-10*fmax(1.,fabs(c.previousObjective[g])))c.amplitudes[p]=c.amplitude2[p];
}
// Amplitudes are analytically optimal after each step. Check the remaining
// envelope KKT residual in diagonal-curvature units, relative to the signal
// improvement. Coefficient changes are misleading in nearly flat directions.
__global__ void SharedStationarity(FitView c)
{
    if(threadIdx.x)return;int g=blockIdx.x,n=c.anchors*c.knots;
    const double* a=c.envelopeMatrix+(size_t)g*(n*n+n);const double* b=a+n*n;
    const double* x=c.envelopes+(size_t)g*n;double residual=0;
    for(int i=0;i<n;i++)
    {
        double gradient=-b[i];for(int j=0;j<n;j++)gradient+=a[i*n+j]*x[j];
        if(x[i]<=0 && gradient>0)gradient=0;
        residual+=gradient*gradient/fmax(1e-30,a[i*n+i]);
    }
    c.convergence[g]=isfinite(residual)?sqrt(residual/fmax(1.,-2*c.objective[g])):CUDART_INF;
}
void ConfigureShared(FitContext& c,int groups,int anchors,const int* ids,const double* blends)
{
    auto& v=c.v;if(groups<1||anchors<1||anchors>3)throw cudaErrorInvalidValue;
    if(c.profileGraph){Check(cudaGraphExecDestroy(c.profileGraph));c.profileGraph=nullptr;}
    if(!c.profileStream)Check(cudaStreamCreateWithFlags(&c.profileStream,cudaStreamNonBlocking));
    std::vector<int> starts(groups+1,0),records(v.records);
    for(int p=0;p<v.records;p++){if(ids[p]<0||ids[p]>=groups)throw cudaErrorInvalidValue;starts[ids[p]+1]++;}
    for(int g=1;g<=groups;g++)starts[g]+=starts[g-1];auto next=starts;
    for(int p=0;p<v.records;p++)records[next[ids[p]]++]=p;
    v.envelopeCount=groups;v.anchors=anchors;
#define UP(member,source,count) c.member.Allocate(count);c.member.UploadConverted(source,count);v.member=c.member.p
    UP(envelopeGroups,ids,v.records);UP(blends,blends,(size_t)v.records*anchors);
    UP(groupStarts,starts.data(),starts.size());UP(groupRecords,records.data(),records.size());
#undef UP
    int n=anchors*v.knots,k=v.knots;
#define AL(member,count) c.member.Allocate(count);v.member=c.member.p
    AL(envelopeMatrix,(size_t)groups*(n*n+n));AL(envelopes,(size_t)groups*n);AL(envelopeWork,(size_t)groups*(n*n+4*n));
    AL(reduced,(size_t)v.records*k*(k+1));AL(backgroundSolutions,(size_t)v.records*k*(k+1));AL(sharedWork,(size_t)v.records*(k*k+4*k));
    AL(amplitude0,v.records);AL(amplitude1,v.records);AL(amplitude2,v.records);AL(reducedLoss,v.records);AL(objective,groups);AL(previousObjective,groups);
    AL(amplitudes,v.records);AL(convergence,groups);
#undef AL
}
void ProfileShared(FitContext& c)
{
    auto v=c.v;
    NormalEquations<false><<<dim3((v.features+7)/8,v.records),Threads>>>(v);
    EliminateBackground<false><<<v.records,1>>>(v);
    NormalEquations<true><<<dim3((v.features+7)/8,v.records),Threads>>>(v);
    EliminateBackground<true><<<v.records,1>>>(v);
    // Deterministic restart makes the profiled objective independent of line-search
    // history. All inner iterations touch only the small Schur systems.
    std::vector<double> ones(v.records,1),changes(v.envelopeCount);
    c.amplitudes.UploadConverted(ones.data(),ones.size());
    int n=v.anchors*v.knots;
    Check(cudaMemset(v.envelopes,0,(size_t)v.envelopeCount*n*sizeof(double)));
    if(!c.profileGraph)
    {
        // Capture the fixed inner update once per layout. The spectra and Schur
        // systems change between poses, but their device addresses remain stable.
        auto stream=c.profileStream;
        Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        auto Step=[&]()
        {
            SharedNormal<<<dim3((n*n+n+127)/128,v.envelopeCount),128,0,stream>>>(v);
            SolveShared<<<v.envelopeCount,1,0,stream>>>(v);
            UpdateAmplitudes<<<(v.records+127)/128,128,0,stream>>>(v);
            NormalizeAmplitudes<<<v.envelopeCount,1,0,stream>>>(v);
        };
        Check(cudaMemcpyAsync(v.amplitude0,v.amplitudes,v.records*sizeof(double),cudaMemcpyDeviceToDevice,stream));
        Step();
        Check(cudaMemcpyAsync(v.amplitude1,v.amplitudes,v.records*sizeof(double),cudaMemcpyDeviceToDevice,stream));
        Step();
        Check(cudaMemcpyAsync(v.amplitude2,v.amplitudes,v.records*sizeof(double),cudaMemcpyDeviceToDevice,stream));
        Check(cudaMemcpyAsync(v.previousObjective,v.objective,v.envelopeCount*sizeof(double),cudaMemcpyDeviceToDevice,stream));
        ExtrapolateAmplitudes<<<v.envelopeCount,1,0,stream>>>(v);
        Step();RejectExtrapolation<<<(v.records+127)/128,128,0,stream>>>(v);Step();
        SharedNormal<<<dim3((n*n+n+127)/128,v.envelopeCount),128,0,stream>>>(v);
        SharedStationarity<<<v.envelopeCount,1,0,stream>>>(v);
        cudaGraph_t graph;
        Check(cudaStreamEndCapture(stream,&graph));
        auto error=cudaGraphInstantiate(&c.profileGraph,graph,nullptr,nullptr,0);
        cudaGraphDestroy(graph);Check(error);
    }
    // Complete the model/normal equations and deterministic reset on the default
    // stream before using the nonblocking graph stream.
    Check(cudaStreamSynchronize(0));
    // Relative first-order tolerance on the small nuisance objective, not on
    // arbitrary amplitude/shape scales. Scores themselves use FP32 reductions.
    constexpr double StationarityTolerance=1e-4;
    for(int iteration=0;iteration<512;iteration++)
    {
        Check(cudaGraphLaunch(c.profileGraph,c.profileStream));
        Check(cudaStreamSynchronize(c.profileStream));
        c.convergence.Download(changes.data(),changes.size());
        double maximum=0;for(double change:changes)maximum=fmax(maximum,change);
        if(maximum<StationarityTolerance)return;
    }
    fprintf(stderr,"CTF shared profile did not converge: records=%d anchors=%d groups=%d stationarity=%g\n",v.records,v.anchors,v.envelopeCount,*std::max_element(changes.begin(),changes.end()));
    throw cudaErrorUnknown; // Do not return a nonstationary profile with envelope-theorem gradients.
}
