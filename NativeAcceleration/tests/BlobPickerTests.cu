// Standalone: nvcc -std=c++17 -I. src/BlobPicker.cu tests/BlobPickerTests.cu -lcufft -o blob_picker_tests
#include "include/BlobPicker.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static void Check(cudaError_t status) { if (status) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status)); std::abort(); } }
static void Near(double a, double b, double tolerance) {
    if (std::abs(a-b) > tolerance) { std::fprintf(stderr,"Mismatch %.9g vs %.9g\n",a,b); std::abort(); }
}
// The blob profile BlobPickerScore correlates with: a flat annulus, or for a disk a difference of Gaussians
// (centre sigma radius/2, surround sigma radius), both within the 2*radius support.
static double Blob(double r2, double radius, double inner) {
    double outer2=radius*radius;
    if (inner>0) return r2<=outer2 && r2>=inner*inner ? 1 : 0;
    double centre2=.25*outer2, surround2=outer2;
    return std::exp(-r2/(2*centre2))/centre2-std::exp(-r2/(2*surround2))/surround2;
}
int main() {
    int2 dims = make_int2(75,69); // Rectangular, odd-sized FFT.
    int n=dims.x*dims.y;
    float *input, *score, *power;
    Check(cudaMalloc(&input,n*sizeof(float))); Check(cudaMalloc(&score,n*sizeof(float))); Check(cudaMalloc(&power,n*sizeof(float)));
    for (float inner : {0.f,3.f}) {
        float radius=5.25f;
        std::vector<float> image(n), scores(n), powers(n);
        for (int y=0;y<dims.y;++y) for(int x=0;x<dims.x;++x) {
            int id=y*dims.x+x;
            image[id]=100.f+0.02f*std::sin(id*.731f);
            // Dark particles shaped like the template, so a perfect match scores ~1.
            for (int cx : {22,52}) {
                float r2=float((x-cx)*(x-cx)+(y-34)*(y-34));
                if (r2<=4*radius*radius) image[id]-=float(4*Blob(r2,radius,inner)/Blob(inner>0?inner*inner:0,radius,inner));
            }
        }
        Check(cudaMemcpy(input,image.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        assert(BlobPickerScore(input,dims,radius,inner,score,power)==0);
        Check(cudaMemcpy(scores.data(),score,n*sizeof(float),cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(powers.data(),power,n*sizeof(float),cudaMemcpyDeviceToHost));
        // Independent direct spatial Pearson/variance reference at every valid pixel.
        int margin=int(std::ceil(2*radius));
        for (int y=0;y<dims.y;++y) for(int x=0;x<dims.x;++x) {
            int id=y*dims.x+x;
            if (x<margin || y<margin || x>=dims.x-margin || y>=dims.y-margin) {
                assert(std::isinf(scores[id]) && scores[id]<0); assert(powers[id]==0); continue;
            }
            // The template is the zero-mean (over the support) negative of the blob profile b.
            double sum=0,sum2=0,signalSum=0,count=0,signal=0,energy=0;
            for(int dy=-margin;dy<=margin;++dy) for(int dx=-margin;dx<=margin;++dx) {
                float r2=float(dx*dx+dy*dy);
                if(r2>4*radius*radius) continue;
                double v=image[(y+dy)*dims.x+x+dx]-100., b=Blob(r2,radius,inner);
                sum+=v;sum2+=v*v;count++;
                signal+=b;energy+=b*b;signalSum+=v*b;
            }
            double variance=sum2/count-(sum/count)*(sum/count);
            double expected=(sum*signal/count-signalSum)/std::sqrt((sum2-sum*sum/count)*(energy-signal*signal/count));
            Near(powers[id],variance,2e-5);
            Near(scores[id],expected,3e-3); // Background variance is ~2e-4.
        }
        float4* picks=nullptr;int count=0;
        assert(BlobPickerPeaks(score,power,dims,2*radius,&picks,&count)==0);
        for(int cx : {22,52}) {
            bool found=false;
            for(int i=0;i<count;++i) if(picks[i].x==cx && picks[i].y==34) { found=true;assert(picks[i].z>.99f); }
            assert(found);
        }
        for(int i=0;i<count;++i) for(int j=0;j<i;++j) {
            float dx=picks[i].x-picks[j].x,dy=picks[i].y-picks[j].y;
            assert(dx*dx+dy*dy>4*radius*radius);
        }
        std::free(picks);
        // Affine invariance of correlation and square scaling of power.
        for(float& v:image) v=3*v-219;
        Check(cudaMemcpy(input,image.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        assert(BlobPickerScore(input,dims,radius,inner,score,power)==0);
        std::vector<float> transformed(n), transformedPower(n);
        Check(cudaMemcpy(transformed.data(),score,n*sizeof(float),cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(transformedPower.data(),power,n*sizeof(float),cudaMemcpyDeviceToHost));
        for(int cx : {22,52}) { int id=34*dims.x+cx; Near(scores[id],transformed[id],1e-5); Near(9*powers[id],transformedPower[id],2e-4); }
    }
    // Flat input gives no peaks; NaN input and degenerate sampled annuli are rejected.
    std::vector<float> flat(n,7);
    Check(cudaMemcpy(input,flat.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    assert(BlobPickerScore(input,dims,5,0,score,power)==0);
    float4* picks=nullptr;int count=0;
    assert(BlobPickerPeaks(score,power,dims,10,&picks,&count)==0);assert(count==0 && picks==nullptr);
    flat[17]=std::numeric_limits<float>::quiet_NaN();
    Check(cudaMemcpy(input,flat.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    assert(BlobPickerScore(input,dims,5,0,score,power)==-1);
    assert(BlobPickerScore(input,dims,5,5,score,power)==-1);
    // Fractional Euclidean distance, deterministic plateaus and invalid neighbors.
    std::vector<float> values(n,-std::numeric_limits<float>::infinity());
    values[10*dims.x+10]=.5f; values[10*dims.x+12]=.5f;
    values[12*dims.x+10]=.7f; values[30*dims.x+30]=std::numeric_limits<float>::infinity();
    Check(cudaMemcpy(score,values.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    assert(BlobPickerPeaks(score,power,dims,1.9f,&picks,&count)==0); assert(count==3);std::free(picks);
    assert(BlobPickerPeaks(score,power,dims,2.1f,&picks,&count)==0); assert(count==1);assert(picks[0].x==10 && picks[0].y==12);std::free(picks);
    Check(cudaFree(input));Check(cudaFree(score));Check(cudaFree(power));
    std::puts("BlobPicker CUDA tests passed (difference-of-Gaussians disk, ring, CPU reference, power scaling, boundaries, separation, flat/nonfinite).");
}
