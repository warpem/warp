using System;
using System.Linq;
using System.Collections.Generic;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfSharedEnvelopeTests
{
    [Fact]
    public void AngularLayoutSeparatesHypothesesAndInterpolatesBothSides()
    {
        var layout=CtfEnvelopeLayout.Create(6,new[]{7,7,7,7,7,9},new[]{-60.0,-30,0,30,60,0});
        Assert.Equal(2,layout.Groups);Assert.Equal(3,layout.Anchors);
        Assert.Equal(new[]{1.0,0,0,.5,.5,0,0,1,0,0,.5,.5,0,0,1,0,1,0},layout.Blends);
        Assert.Equal(new[]{0,0,0,0,0,1},layout.Ids);
        Assert.Equal(1,CtfEnvelopeLayout.Create(4).Anchors);
        Assert.Throws<ArgumentException>(()=>CtfEnvelopeLayout.Create(2,angles:new[]{double.NaN,0.0}));
    }

    [CtfCudaFact]
    public void RestrictedCoarseFitRetainsBroadLowDefocusRingsAboveCurvedBackground()
    {
        lock(GPU.Sync)
        {
            GPU.SetDevice(0);
            const double truth=.55,qmax=.14;
            double lambda=CtfSpectrumFit.Wavelength(300),kd=Math.PI*lambda*1e4;
            double[] offsets={-.12,0,.12};
            var spectra=offsets.Select((offset,p)=>
            {
                var samples=new List<CtfSpectrumFit.Sample>();var random=new Random(782+p);
                for(int r=20;r<=140;r++)
                {
                    double q=r*.001,q2=q*q,t=q2/(qmax*qmax);
                    double phase=kd*q2*(truth+offset)-.5*Math.PI*2.7e7*Math.Pow(lambda,3)*q2*q2+Math.Asin(.07);
                    double background=(p+1)*(5+3*t+2*t*t);
                    double power=background+1.7*Math.Exp(-4*t)*Math.Pow(Math.Sin(phase),2)+.002*(random.NextDouble()-.5);
                    samples.Add(new(q2,q2*q2,q2,0,power,100));
                }
                return new CtfSpectrumFit(samples.ToArray(),300,2.7,.07);
            }).ToArray();
            using var batch=new CtfGpuFitBatch(spectra);
            var defocus=Enumerable.Range(0,561).Select(i=>.2+.005*i).ToArray();
            var scores=batch.Search(defocus.SelectMany(df=>new[]{df,0.0}).ToArray(),offsets);
            int best=Enumerable.Range(0,defocus.Length).OrderByDescending(i=>Enumerable.Range(0,3).Sum(p=>scores[3*i+p])).First();
            Assert.InRange(Math.Abs(defocus[best]-truth),0,.01);
        }
    }

    static CtfSpectrumFit Spectrum(double amplitude,double angle,double contamination=0)
    {
        var samples=new List<CtfSpectrumFit.Sample>();double k=Math.PI*CtfSpectrumFit.Wavelength(300)*1e4;
        for(int r=16;r<100;r++)for(int a=0;a<8;a++)
        {
            double q=r/640.0,q2=q*q,az=(a+.3)*Math.PI/8;
            double gamma=k*(1.25*q2+.025*q2*Math.Cos(2*az))-.5*Math.PI*2.7e7*Math.Pow(CtfSpectrumFit.Wavelength(300),3)*q2*q2+Math.Asin(.07);
            double envelope=(1+.8*q)*(1+angle/300)+(angle>0?angle/60*.5*q:0);
            double power=2+q+amplitude*envelope*Math.Pow(Math.Sin(gamma),2)+contamination*Math.Sin(90*q)*Math.Sin(90*q);
            samples.Add(new(q2,q2*q2,q2*Math.Cos(2*az),q2*Math.Sin(2*az),power,30));
        }
        return new(samples.ToArray(),300,2.7,.07);
    }
    [CtfCudaFact]
    public void SharedShapeHasPatchScalarsAndCorrectProfiledGradient()
    {
        lock(GPU.Sync)
        {
            GPU.SetDevice(0);
            double[] angles={-60,-60,-30,-30,0,0,30,30,60,60};
            var spectra=angles.Select((a,i)=>Spectrum(i%2==0?.5:1.8,a)).ToArray();
            using var batch=new CtfGpuFitBatch(spectra,tiltAngles:angles);
            var poses=new double[70];for(int i=0;i<10;i++){poses[7*i]=1.251;poses[7*i+1]=.024;}
            batch.Evaluate(poses);var coefficients=batch.ReadCoefficients();int k=coefficients.Length/20;
            for(int pair=0;pair<5;pair++)
            {
                int first=pair*4*k+k,second=first+2*k;
                double sumA=Enumerable.Range(0,k).Sum(j=>coefficients[first+j]),sumB=Enumerable.Range(0,k).Sum(j=>coefficients[second+j]);
                Assert.True(sumA>0&&sumB>0);
                for(int j=0;j<k;j++)Assert.InRange(Math.Abs(coefficients[first+j]/sumA-coefficients[second+j]/sumB),0,2e-6);
            }
            double Loss(){var values=batch.Evaluate(poses);return Enumerable.Range(0,10).Sum(i=>values[9*i]);}
            foreach(double offset in new[]{0.0,.17})
            {
                for(int i=0;i<10;i++)poses[7*i]=1.251+offset;
                var output=(double[])batch.Evaluate(poses).Clone();
                foreach(int parameter in new[]{0,1,2,3})
                {
                    const double h=1e-4;poses[parameter]+=h;double plus=Loss();poses[parameter]-=2*h;double minus=Loss();poses[parameter]+=h;
                    double numerical=(plus-minus)/(2*h),analytic=output[parameter+1];
                    Assert.InRange(Math.Abs(numerical-analytic),0,.02*Math.Max(1,Math.Abs(analytic)));
                }
            }
        }
    }
    [CtfCudaFact]
    public void IndependentCandidateGroupsCannotBorrowEachOthersEnvelope()
    {
        lock(GPU.Sync)
        {
            GPU.SetDevice(0);
            double[] Evaluate(double pollution)
            {
                using var batch=new CtfGpuFitBatch(new[]{Spectrum(.5,0),Spectrum(1.8,0),Spectrum(1,0,pollution)},new[]{0,0,1});
                var poses=new double[21];for(int i=0;i<3;i++){poses[7*i]=1.251;poses[7*i+1]=.024;}
                var values=batch.Evaluate(poses);return values.Take(18).ToArray();
            }
            var first=Evaluate(0);var second=Evaluate(10);
            for(int i=0;i<first.Length;i++)Assert.InRange(Math.Abs(first[i]-second[i]),0,1e-4*Math.Max(1,Math.Abs(first[i])));
        }
    }
}
