using System;
using System.Linq;
using System.Reflection;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfFullDiagnosticTests
{
    [Fact]
    public void TransferZeroDoesNotAmplifyOrEraseBackgroundSubtractedPower()
    {
        const int fft=1024, window=256, zeroBin=300;
        const double pixel=1.5, df=2.3;
        double lambda=CtfSpectrumFit.Wavelength(300), kd=Math.PI*lambda*1e4, kc=-.5*Math.PI*2.7e7*Math.Pow(lambda,3);
        double qzero=zeroBin/(fft*pixel), thickness=Math.PI/(kd*qzero*qzero);
        var samples=Enumerable.Range(1,fft/2-1).Select(b=>
        {
            double q=b/(fft*pixel), gamma=kd*q*q*df+kc*Math.Pow(q,4)+Math.Asin(.07);
            double power=3+2*CtfSlabModel.Power(gamma,kd*q*q,thickness*thickness)+(b==zeroBin?-.02:0);
            return new CtfSpectrumFit.Sample(q*q,Math.Pow(q,4),0,0,power,100);
        }).ToArray();
        var spectrum=new CtfSpectrumFit(samples,300,2.7,.07);
        int knots=(int)typeof(CtfSpectrumFit).GetProperty("KnotCount",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(spectrum);
        var coefficients=Enumerable.Repeat((float)(3/spectrum.PowerScale),knots).Concat(Enumerable.Repeat((float)(2/spectrum.PowerScale),knots)).ToArray();
        double[] parameters={df,0,0,0,thickness*thickness};
        var reference=new CTF{PixelSize=(decimal)pixel,Voltage=300,Cs=2.7M,Amplitude=.07M,Defocus=(decimal)df};
        var fit=new CtfFitEngine.Fit(parameters,0,0,coefficients);
        var records=new[]{new CtfPowerSpectrum.Observation(spectrum,new float3(.5f),0)};
        var geometry=new[]{new CtfFitGeometry(new[]{1.0},new[]{1.0})};
        var actual=CtfFitDiagnostics.Create(records,geometry,fit,new[]{0},new[]{reference},reference,fft,window,new[]{new float[window*window/2]}).Global;
        Assert.InRange(actual.Spectrum[zeroBin].Y,-.0201f,-.0199f);
        Assert.All(actual.Spectrum,p=>Assert.InRange(p.Y,-2.01f,2.01f));
        Assert.InRange(Math.Abs(actual.Envelope.Interp((float)zeroBin/fft)),0,1e-5);
        // The physical signed transfer reverses after its zero; the displayed model must too.
        Assert.True(actual.Envelope.Interp((float)(zeroBin+30)/fft)<0);
    }

    [CtfCudaFact]
    public void FullDiagnosticsPreservePowerBeyondTheFitBandWithoutChangingParameters()
    {
        lock(GPU.Sync)
        {
            GPU.SetDevice(0);
            using var image=new Image(new int3(128,128,3));var rng=new Random(8931);
            foreach(var frame in image.GetHost(Intent.Write))for(int i=0;i<frame.Length;i++)frame[i]=(float)(rng.NextDouble()-.5);
            var options=new ProcessingOptionsMovieCTF{Window=64,PixelSize=3,RangeMin=.2M,RangeMax=.4M,ZMin=.1M,ZMax=1,Voltage=300,Cs=2.7M,Amplitude=.07M};
            var band=CtfPowerSpectrum.Extract(image,options);
            Assert.All(band.Observations[0].Spectrum.Samples,s=>Assert.InRange(Math.Sqrt(s.Q2),.2/6,.4/6));
            using var extractor=new CtfPowerSpectrum.Extractor(new int2(128),options,fullSpectrum:true);
            var full=extractor.Extract(image);
            Assert.True(full.Observations[0].Spectrum.Samples.Min(s=>s.Q2)<Math.Pow(.2/6,2));
            Assert.True(full.Observations[0].Spectrum.Samples.Max(s=>s.Q2)>Math.Pow(.95/6,2));
            double[] parameters={.7,0,0,0,0};var before=(double[])parameters.Clone();
            var geometry=band.Observations.Select(_=>new CtfFitGeometry(new[]{1.0},new[]{1.0})).ToArray();
            var reference=CtfFitEngine.MakeCtf(options,.7,0,0,0);
            var display=new float[64*32];
            var result=CtfFitDiagnostics.CreateFullSpectrum(extractor,new[]{new CtfFitDiagnostics.Input(image,geometry,0)},new(parameters,0,0),new[]{reference},reference,64,new[]{display});
            Assert.Equal(before,parameters);
            Assert.Contains(result.Global.Spectrum,p=>p.X>.35f&&p.Y!=0);
            Assert.Contains(result.Global.Spectrum,p=>p.X<.1f&&p.Y!=0);
            // Display pixels at r=20..28 are outside the fitting radius (12.8).
            Assert.Contains(Enumerable.Range(0,display.Length).Where(i=>
            {int x=i%64-32,y=31-i/64;return x*x+y*y>=400&&x*x+y*y<784;}).Select(i=>display[i]),v=>v!=0);
        }
    }
}
