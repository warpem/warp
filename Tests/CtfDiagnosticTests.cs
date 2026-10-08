using System;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfDiagnosticTests
{
    [Fact]
    public void ClosedFormAlignmentPreservesPhaseOnBothBranchesAndWithoutCs()
    {
        foreach (float cs in new[] { 0f, -300f })
            foreach (float df in new[] { 0f, 30f, 1400f })
                foreach (float q2 in new[] { 1e-6f, .001f, .02f, .15f })
                {
                    if (cs == 0 && df == 0) continue;
                    float target = df * q2 + cs * q2 * q2;
                    float result = CtfFitDiagnostics.AlignFrequencySquared(target, df, cs, q2);
                    Assert.InRange(MathF.Abs(result - q2), 0, Math.Max(1e-8f, q2 * 2e-5f));
                }
        Assert.True(float.IsNaN(CtfFitDiagnostics.AlignFrequencySquared(10, 1, -1, .1f)));
        Assert.True(float.IsNaN(CtfFitDiagnostics.AlignFrequencySquared(1, 0, 0, .1f)));
    }
    static (float2[] Spectrum, float[] Support) Curve(CTF ctf, int fft, bool invert = false, bool cutoff = false)
    {
        var spectrum = new float2[fft/2]; var support = new float[fft/2];
        for (int i=0;i<spectrum.Length;i++)
        {
            double f=(double)i/fft, q=f/(double)ctf.PixelSize;
            double model=Math.Pow(ctf.Get1DDouble(q,false,true,true),2);
            double value=cutoff && q>.2 ? .5+.3*Math.Sin(1100*q) : (invert ? 1-model : model);
            spectrum[i]=new((float)f,(float)value);
            support[i]=q>=.035 && q<=.35 ? 1 : 0;
        }
        return (spectrum,support);
    }
    static CTF Reference(decimal defocus=2.5M) => new(){PixelSize=1.5M,Voltage=300,Cs=2.7M,Amplitude=.07M,Defocus=defocus};

    [Fact]
    public void QualityRetainsSignAndMarksMissingOrFlatDataUnknown()
    {
        var ctf=Reference();var (s,w)=Curve(ctf,2048);
        var good=CtfFitDiagnostics.CalculateQuality(s,w,ctf,512);
        Assert.All(good.Where(p=>float.IsFinite(p.Y)),p=>Assert.InRange(p.Y,.9999f,1));
        Assert.True(good.Count(p=>float.IsFinite(p.Y))>200);
        var (inverse,_) = Curve(ctf,2048,true);
        Assert.All(CtfFitDiagnostics.CalculateQuality(inverse,w,ctf,512).Where(p=>float.IsFinite(p.Y)),p=>Assert.InRange(p.Y,-1,-.9999f));
        Assert.Equal(0,CtfFitDiagnostics.EstimateResolution(CtfFitDiagnostics.CalculateQuality(inverse,w,ctf,512),ctf));
        for(int i=0;i<s.Length;i++)s[i].Y=.5f;
        Assert.All(CtfFitDiagnostics.CalculateQuality(s,w,ctf,512),p=>Assert.True(float.IsNaN(p.Y)));
        Assert.All(CtfFitDiagnostics.CalculateQuality(inverse,new float[w.Length],ctf,512),p=>Assert.True(float.IsNaN(p.Y)));
        Assert.True(float.IsNaN(good[0].Y));
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(4.5)]
    public void QualityAndResolutionAreStableUnderFourierPadding(double defocus)
    {
        var ctf=Reference((decimal)defocus);
        var (s,w)=Curve(ctf,2048,cutoff:true);var (s2,w2)=Curve(ctf,4096,cutoff:true);
        var q=CtfFitDiagnostics.CalculateQuality(s,w,ctf,512);
        var q2=CtfFitDiagnostics.CalculateQuality(s2,w2,ctf,512);
        var differences=Enumerable.Range(0,q.Length).Where(i=>float.IsFinite(q[i].Y)&&float.IsFinite(q2[i*2].Y)).Select(i=>Math.Abs(q[i].Y-q2[i*2].Y)).ToArray();
        Assert.InRange(differences.Average(),0,.01);
        Assert.InRange(Math.Abs(CtfFitDiagnostics.EstimateResolution(q,ctf)-CtfFitDiagnostics.EstimateResolution(q2,ctf)),0,.1M);
        Assert.InRange(CtfFitDiagnostics.EstimateResolution(q,ctf),4.5M,5.5M);
        // Missing support in the middle must not turn zero-filled data into a valid correlation.
        for(int i=0;i<w.Length;i++)if(s[i].X>.2f && s[i].X<.25f)w[i]=0;
        var gap=CtfFitDiagnostics.CalculateQuality(s,w,ctf,512);
        Assert.All(gap.Where(p=>p.X>.2f&&p.X<.25f),p=>Assert.True(float.IsNaN(p.Y)));
    }

    [Fact]
    public void QualityIsInvariantToPhysicalPowerUnits()
    {
        var ctf=Reference();var (s,w)=Curve(ctf,2048,cutoff:true);
        var baseline=CtfFitDiagnostics.CalculateQuality(s,w,ctf,512);
        foreach(float scale in new[]{1e-8f,1e8f})
        {
            var scaled=s.Select(p=>new float2(p.X,p.Y*scale)).ToArray();
            var result=CtfFitDiagnostics.CalculateQuality(scaled,w,ctf,512,Enumerable.Repeat(scale,s.Length).ToArray());
            for(int i=0;i<result.Length;i++)
                if(float.IsFinite(baseline[i].Y))Assert.InRange(Math.Abs(result[i].Y-baseline[i].Y),0,1e-5);
                else Assert.True(float.IsNaN(result[i].Y));
        }
    }

    static void SameCurve(float2[] expected,float2[] actual)
    {
        Assert.Equal(expected.Length,actual.Length);
        for(int i=0;i<expected.Length;i++){Assert.Equal(expected[i].X,actual[i].X);Assert.Equal(expected[i].Y,actual[i].Y);}
    }

    [Fact]
    public void QualityMetadataRoundTripsGlobalAndPerTiltIncludingUnsupportedBins()
    {
        string dir=Path.Combine(Path.GetTempPath(),"warp-quality-"+Guid.NewGuid());Directory.CreateDirectory(dir);
        try
        {
            float2[] curve={new(0,float.NaN),new(.1f,.95f),new(.2f,-.3f)};
            string path=Path.Combine(dir,"movie.mrc");var movie=new Movie(path){CTFQuality=curve};movie.SaveMeta();
            SameCurve(curve,new Movie(path).CTFQuality);
            string series=Path.Combine(dir,"series.tomostar");
            File.WriteAllText(series,"data_\n\nloop_\n_wrpMovieName #1\n_wrpAngleTilt #2\n_wrpDose #3\na.mrc -30 1\nb.mrc 30 2\n");
            var ts=new TiltSeries(series){CTFQuality=curve};ts.TiltCTFQuality.Add(curve);ts.TiltCTFQuality.Add(new[]{new float2(.1f,.4f)});ts.SaveMeta();
            var loaded=new TiltSeries(series);SameCurve(curve,loaded.CTFQuality);SameCurve(curve,loaded.TiltCTFQuality[0]);Assert.Equal(.4f,loaded.TiltCTFQuality[1][0].Y);
        }
        finally{Directory.Delete(dir,true);}
    }
}
