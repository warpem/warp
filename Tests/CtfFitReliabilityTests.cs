using System;
using System.Linq;
using System.Reflection;
using Warp;
using Warp.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Tests;
public class CtfFitReliabilityTests
{
    readonly ITestOutputHelper output;
    public CtfFitReliabilityTests(ITestOutputHelper output) { this.output=output; }
    static (float[][] Spectra,(double X,double Y,double Width)[] Positions,double[] Frequencies) Data(int seed,int good,bool noiseOnly=false)
    {
        var rng=new Random(seed);int n=64,bins=640;
        var q=Enumerable.Range(0,bins).Select(b=>Math.Sqrt(b/16.0/360)).ToArray();
        var pos=Enumerable.Range(0,n).Select(i=>((double)(i%8),(double)(i/8),1.0)).ToArray();
        var spectra=Enumerable.Range(0,n).Select(i=>Enumerable.Range(0,bins).Select(b=>
        {
            double noise=Math.Sqrt(-2*Math.Log(Math.Max(1e-10,rng.NextDouble())))*Math.Cos(2*Math.PI*rng.NextDouble());
            double signal=!noiseOnly&&i<good&&q[b]<.125 ? -2*Math.Cos(2*Math.PI*b/16) : 0;
            // Independent broad contamination in bad regions, including very large edges.
            if(i>=good)noise+=10*Math.Sin(b*.007+i);
            return (float)(signal+noise);
        }).ToArray()).ToArray();
        return(spectra,pos,q);
    }
    [Fact]
    public void GoodIceInAMinorityOfPatchesCanEstablishSupport()
    {
        var d=Data(41,16);var r=CtfFitReliability.Measure(Data(100041,16).Spectra,d.Spectra,d.Positions,d.Frequencies);
        double low=r.Curve.Weight.Where((v,i)=>d.Frequencies[i]>.08&&d.Frequencies[i]<.10).Average();
        double high=r.Curve.Weight.Where((v,i)=>d.Frequencies[i]>.18&&d.Frequencies[i]<.24).Average();
        output.WriteLine($"Independent {r.Curve.IndependentPatches}; low {low}; high {high}; good {r.PatchWeight.Take(16).Average()}; bad {r.PatchWeight.Skip(16).Average()}");
        Assert.True(low>.8);Assert.True(high<.1);
        Assert.True(r.PatchWeight.Take(16).Average()>5*r.PatchWeight.Skip(16).Average());
    }
    [Fact]
    public void SignalFreeSpectraDoNotAcquireHighFrequencySupportFromLowFrequencySelection()
    {
        double sum=0,max=0;
        for(int seed=0;seed<100;seed++)
        {
            var d=Data(seed,64,true);var r=CtfFitReliability.Measure(Data(seed+1000,64,true).Spectra,d.Spectra,d.Positions,d.Frequencies);
            if(r.Curve.IndependentPatches<4)continue;
            double support=r.Curve.Weight.Where((v,i)=>d.Frequencies[i]>.15&&d.Frequencies[i]<.25).Average();
            sum+=support;max=Math.Max(max,support);
        }
        output.WriteLine($"Null mean {sum/100}; maximum {max}");
        Assert.True(sum/100<.05);Assert.True(max<.15);
    }
    [Fact]
    public void OverlappingPatchesDoNotCountAsIndependentReplication()
    {
        var d=Data(1,64);var pos=d.Positions.Select(_=>(0.0,0.0,1.0)).ToArray();
        var r=CtfFitReliability.Measure(d.Spectra,d.Spectra,pos,d.Frequencies);
        Assert.Equal(1,r.Curve.IndependentPatches);
        Assert.All(r.Curve.Agreement,v=>Assert.True(float.IsNaN(v)));
    }
    [Fact]
    public void RepeatingTheSamePatchesDoesNotIncreaseEvidence()
    {
        var d=Data(301,64,true);var train=Data(1301,64,true);
        var original=CtfFitReliability.Measure(train.Spectra,d.Spectra,d.Positions,d.Frequencies);
        var repeated=CtfFitReliability.Measure(
            train.Spectra.SelectMany(row=>Enumerable.Repeat(row,4)).ToArray(),
            d.Spectra.SelectMany(row=>Enumerable.Repeat(row,4)).ToArray(),
            d.Positions.SelectMany(p=>Enumerable.Repeat(p,4)).ToArray(),d.Frequencies);
        Assert.Equal(original.Curve.IndependentPatches,repeated.Curve.IndependentPatches);
        for(int i=0;i<d.Frequencies.Length;i++)Assert.Equal(original.Curve.Weight[i],repeated.Curve.Weight[i],5);
    }

    [Fact]
    public void CorrelatedNoiseDoesNotCreateAUsableHighResolutionBand()
    {
        double average=0,maximum=0;
        for(int seed=0;seed<100;seed++)
        {
            var d=Data(seed,64,true);
            var kernel=Enumerable.Range(-9,19).Select(i=>Math.Exp(-i*i/18.0)).ToArray();
            double norm=Math.Sqrt(kernel.Sum(v=>v*v));
            float[][] Smooth(float[][] rows) => rows.Select(row=>row.Select((_,b)=>(float)kernel.Select((v,k)=>v*row[Math.Clamp(b+k-9,0,row.Length-1)]).Sum()/ (float)norm).ToArray()).ToArray();
            var r=CtfFitReliability.Measure(Smooth(Data(seed+1000,64,true).Spectra),Smooth(d.Spectra),d.Positions,d.Frequencies);
            if(r.Curve.IndependentPatches<4)continue;
            double support=r.Curve.Weight.Where((v,i)=>d.Frequencies[i]>.15&&d.Frequencies[i]<.25).Average();
            average+=support;maximum=Math.Max(maximum,support);
        }
        output.WriteLine($"Correlated null mean {average/100}; maximum {maximum}");
        Assert.True(average/100<.05);Assert.True(maximum<.15);
    }

    [Fact]
    public void CoherentRingsSurviveAnImperfectInitialCtfPhase()
    {
        var d=Data(71,64);
        for(int i=0;i<d.Spectra.Length;i++)for(int b=0;b<d.Frequencies.Length;b++)
            if(d.Frequencies[b]>.15)d.Spectra[i][b]+=5*(float)Math.Sin(2*Math.PI*b/16);
        var r=CtfFitReliability.Measure(Data(1071,64).Spectra,d.Spectra,d.Positions,d.Frequencies);
        Assert.True(r.Curve.Agreement.Where((v,i)=>d.Frequencies[i]>.18&&d.Frequencies[i]<.24).Average()>.8);
        Assert.True(r.Curve.Weight.Where((v,i)=>d.Frequencies[i]>.18&&d.Frequencies[i]<.24).Average()>.8);
    }

    [Fact]
    public void SlabContrastReversalRemainsEvidenceForOscillations()
    {
        var d=Data(73,64);
        for(int i=0;i<d.Spectra.Length;i++)for(int b=0;b<d.Frequencies.Length;b++)
            if(d.Frequencies[b]>.15)d.Spectra[i][b]+=2*(float)Math.Cos(2*Math.PI*b/16);
        var r=CtfFitReliability.Measure(Data(1073,64).Spectra,d.Spectra,d.Positions,d.Frequencies);
        Assert.True(r.Curve.Weight.Where((v,i)=>d.Frequencies[i]>.18&&d.Frequencies[i]<.24).Average()>.8);
    }

    [Fact]
    public void SharedSmoothBackgroundIsNotEvidenceOfCtfOscillations()
    {
        var train=Data(119,64,true);var validation=Data(209,64,true);
        foreach(var d in new[]{train,validation})for(int i=0;i<d.Spectra.Length;i++)for(int b=0;b<d.Frequencies.Length;b++)
        {
            double bg=1/Math.Pow(.003+d.Frequencies[b]*d.Frequencies[b],2);
            d.Spectra[i][b]=(float)(bg*(1+.1*d.Spectra[i][b]));
        }
        var r=CtfFitReliability.Measure(train.Spectra,validation.Spectra,train.Positions,train.Frequencies);
        Assert.True(r.Curve.Weight.Where((v,i)=>train.Frequencies[i]>.15&&train.Frequencies[i]<.25).Average()<.05);
    }

    [Fact]
    public void IsolatedHighFrequencyBumpDoesNotExtendTheReportedLimit()
    {
        var q=Enumerable.Range(1,160).Select(i=>i*.001).ToArray();
        var weight=Enumerable.Range(0,160).Select(i=>i<64||i>120&&i<126?1f:0f).ToArray();
        var c=new CtfFitReliability.Curve(q,weight,weight,12);
        Assert.Equal(1/q[63],c.HalfWeightResolution,8);
        Assert.Equal(0,(c with{IndependentPatches=2}).HalfWeightResolution);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1.2)]
    public void EstimatedFrequencyAxisStaysOnTheLowFrequencyBranch(double phase)
    {
        var samples=Enumerable.Range(1,100).SelectMany(i=>Enumerable.Range(0,24).Select(a=>
        {
            double q=.25*i/100,angle=Math.PI*(a+.5)/24;
            return new CtfSpectrumFit.Sample(q*q,Math.Pow(q,4),q*q*Math.Cos(2*angle),q*q*Math.Sin(2*angle),1,10);
        })).ToArray();
        var spectrum=new CtfSpectrumFit(samples,300,2.7,.1);
        var records=new[]{new CtfPowerSpectrum.Observation(spectrum,new float3(.5f),0)};
        var geometry=new[]{new CtfFitGeometry(new[]{1.0},new[]{1.0},patchWidth:.1)};
        var result=CtfFitReliability.Estimate(records,geometry,new[]{1.8,0,0,phase,.01});
        var frequencies=result.Groups[0].Frequency;
        Assert.All(frequencies,q=>Assert.InRange(q,0,.26));
        for(int i=1;i<frequencies.Length;i++)Assert.True(frequencies[i]>=frequencies[i-1]);
    }

    [CtfCudaFact]
    public void ExcludedSamplesCannotChangeNormalizationLikelihoodOrGradient()
    {
        lock(GPU.Sync)
        {
            GPU.SetDevice(0);
            var method=typeof(CtfSpectrumFit).GetMethod("WithFitWeights",BindingFlags.NonPublic|BindingFlags.Instance);
            var samples=Enumerable.Range(1,200).Select(i=>
            {
                double q=i/800.0;return new CtfSpectrumFit.Sample(q*q,Math.Pow(q,4),q*q,0,3+Math.Sin(i*.21),100);
            }).ToArray();
            var mask=samples.Select((_,i)=>i%2==0?1f:0f).ToArray();
            var changed=samples.Select((s,i)=>i%2==0?s:s with{Power=1e12}).ToArray();
            var a=(CtfSpectrumFit)method.Invoke(new CtfSpectrumFit(samples,300,2.7,.07),new object[]{mask});
            var b=(CtfSpectrumFit)method.Invoke(new CtfSpectrumFit(changed,300,2.7,.07),new object[]{mask});
            Assert.Equal(a.PowerScale,b.PowerScale);
            using var batch=new CtfGpuFitBatch(new[]{a,b});
            double[] pose={2.1,.02,-.01,0,.01,0,0};var poses=pose.Concat(pose).ToArray();
            for(int pass=0;pass<3;pass++)
            {
                var value=batch.Evaluate(poses,true);
                for(int j=0;j<9;j++){Assert.True(double.IsFinite(value[j]));Assert.Equal(value[j],value[j+9],6);}
            }
            var zero=(CtfSpectrumFit)method.Invoke(new CtfSpectrumFit(changed,300,2.7,.07),new object[]{new float[mask.Length]});
            using var empty=new CtfGpuFitBatch(new[]{zero});
            Assert.All(empty.Evaluate(pose,true),v=>Assert.Equal(0,v));
        }
    }

    [Fact]
    public void EachAngularFoldSeparatesDefocusFromAstigmatism()
    {
        var s=Enumerable.Range(0,24).Select(i=>
        {
            double angle=(i+.5)*Math.PI/24;
            return new CtfSpectrumFit.Sample(.01,.0001,.01*Math.Cos(2*angle),.01*Math.Sin(2*angle),1,1);
        }).ToArray();
        foreach(int fold in new[]{-1,1})
        {
            var selected=s.Where(v=>CtfFitReliability.AngularFold(v)==fold).ToArray();
            Assert.Equal(8,selected.Length);
            Assert.InRange(Math.Abs(selected.Average(v=>v.AstigX/v.Q2)),0,1e-12);
            Assert.InRange(Math.Abs(selected.Average(v=>v.AstigY/v.Q2)),0,1e-12);
            double xx=selected.Average(v=>Math.Pow(v.AstigX/v.Q2,2)),yy=selected.Average(v=>Math.Pow(v.AstigY/v.Q2,2));
            double xy=selected.Average(v=>v.AstigX*v.AstigY/(v.Q2*v.Q2));
            Assert.True(xx*yy-xy*xy>.05);
        }
    }

}
