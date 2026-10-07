using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfThicknessTests
{
    [Fact]
    public void SlabPowerEqualsDepthAverageOfSquaredCtfAndHannIntegral()
    {
        foreach (double q in new[] { .03,.1,.2,.3,.4 })
        foreach (double thickness in new[] { 0.0,.001,.1,.3 })
        {
            double k=Math.PI*CtfSpectrumFit.Wavelength(300)*1e4*q*q, gamma=1.31;
            const int n=10000;double mean=0;
            for(int j=0;j<n;j++)mean+=Math.Pow(Math.Sin(gamma+k*thickness*((j+.5)/n-.5)),2)/n;
            Assert.InRange(Math.Abs(mean-CtfSlabModel.Power(gamma,k,thickness*thickness)),0,1e-7);
            double argument=k*.13, weight=0, integral=0;
            for(int j=0;j<n;j++)
            {
                double u=(j+.5)/n-.5, w=Math.Pow(.5+.5*Math.Cos(2*Math.PI*u),2);
                integral+=w*Math.Cos(2*argument*u);weight+=w;
            }
            Assert.InRange(Math.Abs(integral/weight-CtfSlabModel.HannPower(argument).Value),0,1e-10);
        }
        // The first depth-averaging zero for 1000 Å at 300 kV lies at ~4.44 Å.
        double lambda=CtfSpectrumFit.Wavelength(300), qzero=1/Math.Sqrt(lambda*1000);
        Assert.InRange(Math.Abs(CtfSlabModel.Modulation(Math.PI*lambda*1e4*qzero*qzero,.01,0,0).Value),0,1e-12);
    }
    [Fact]
    public void SquaredThicknessAndTiltPlaneDerivativesRemainFiniteAtZeroThickness()
    {
        foreach(double t in new[] { 0.0,.01 })
        {
            var geometry=new CtfFitGeometry(new[] { 1.0 },new[] { 1.0 },.17,-.12,Matrix3.Euler(.23f,.91f,-.41f),.23);
            double[] p={2.3,.04,-.02,.2,.05,-.08,t};
            double[] localGradient={.7,-.3,.4,.9,1.3,-.6,.2};
            var gradient=new double[p.Length];geometry.AccumulateVolume(gradient,localGradient,p);
            double Value(){var pose=new double[7];geometry.WritePose(p,pose,0);return pose.Zip(localGradient,(a,b)=>a*b).Sum();}
            for(int j=0;j<p.Length;j++)
            {
                double h=1e-6;p[j]+=h;double a=Value();p[j]-=2*h;double b=Value();p[j]+=h;
                Assert.InRange(Math.Abs((a-b)/(2*h)-gradient[j]),0,1e-7);
            }
            double k=27,h2=1e-7;
            var m=CtfSlabModel.Modulation(k,t,.13,-.07);
            double fd=t==0 ? (CtfSlabModel.Modulation(k,h2,.13,-.07).Value-m.Value)/h2 :
                (CtfSlabModel.Modulation(k,t+h2,.13,-.07).Value-CtfSlabModel.Modulation(k,t-h2,.13,-.07).Value)/(2*h2);
            Assert.InRange(Math.Abs(fd-m.ThicknessSquared)/Math.Max(1,Math.Abs(fd)),0,1e-5);
        }
    }
    [Fact]
    public void MovieAndTiltSeriesThicknessMetadataRoundTrips()
    {
        string dir=Path.Combine(Path.GetTempPath(),"warp-thickness-"+Guid.NewGuid());Directory.CreateDirectory(dir);
        try
        {
            string movie=Path.Combine(dir,"movie.mrc");
            var m=new Movie(movie){CTFSpecimenThicknessAngstrom=1000.125M};m.SaveMeta();
            Assert.Equal(1000.125M,new Movie(movie).CTFSpecimenThicknessAngstrom);
            string series=Path.Combine(dir,"series.tomostar");
            File.WriteAllText(series,"data_\n\nloop_\n_wrpMovieName #1\n_wrpAngleTilt #2\n_wrpDose #3\na.mrc -30 1\nb.mrc 30 2\n");
            var ts=new TiltSeries(series){CTFSpecimenThicknessAngstrom=987.25M};ts.SaveMeta();
            Assert.Equal(987.25M,new TiltSeries(series).CTFSpecimenThicknessAngstrom);
        }
        finally { Directory.Delete(dir,true); }
    }
    static CtfSpectrumFit.Sample[] Samples(double[] pose, int seed=1)
    {
        var samples=new List<CtfSpectrumFit.Sample>();var rng=new Random(seed);
        double lambda=CtfSpectrumFit.Wavelength(300),kd=Math.PI*lambda*1e4,kc=-.5*Math.PI*2.7e7*Math.Pow(lambda,3);
        for(int r=30;r<280;r++)for(int sector=0;sector<8;sector++)
        {
            double q=r/768.0,q2=q*q,angle=(sector+.37)*Math.PI/8,ax=q2*Math.Cos(2*angle),ay=q2*Math.Sin(2*angle);
            double gamma=kd*(q2*pose[0]+ax*pose[1]+ay*pose[2])+kc*q2*q2+Math.Asin(.07)+pose[3];
            double model=CtfSlabModel.Power(gamma,kd*q2,pose[4],pose[5],pose[6]);
            double power=3+2*q+(1+q)*model+.001*(rng.NextDouble()-.5);
            samples.Add(new(q2,q2*q2,ax,ay,power,1000));
        }
        return samples.ToArray();
    }
    [CtfCudaFact]
    public void SlabGpuObjectiveAndAllSevenDerivativesMatchReferenceAndFiniteDifferences()
    {
        lock(GPU.Sync)
        {
            GPU.SetDevice(0);
            double[] pose={2.3007,.0403,-.0249,.203,.0101,.076,-.041};
            var samples=Samples(new[] {2.3,.04,-.025,.2,.01,.075,-.04});
            using var gpu=new CtfGpuFitBatch(new[] {new CtfSpectrumFit(samples,300,2.7,.07)});
            var cpu=new CtfCpuSpectrum(samples,300,2.7,.07).Evaluate(pose[0],pose[1],pose[2],pose[3],false,pose[4],pose[5],pose[6]);
            var actual=(double[])gpu.Evaluate(pose).Clone();
            Assert.InRange(Math.Abs(actual[0]-cpu.Loss)/Math.Max(1,cpu.Loss),0,2e-4);
            for(int j=0;j<7;j++)
            {
                Assert.InRange(Math.Abs(actual[j+1]-cpu.Gradient[j])/Math.Max(1,Math.Abs(cpu.Gradient[j])),0,5e-3);
                // Aperture-width gradients are small relative to the total FP32 loss.
                double h=j==4 ? 1e-5 : j>=5 ? 1e-3 : 1e-4;
                pose[j]+=h;double plus=gpu.Evaluate(pose)[0];pose[j]-=2*h;double minus=gpu.Evaluate(pose)[0];pose[j]+=h;
                double fd=(plus-minus)/(2*h);
                Assert.True(Math.Abs(fd-actual[j+1])/Math.Max(1,Math.Abs(fd)) <= .02, $"parameter {j}, loss {actual[0]}, FD {fd}, analytic {actual[j+1]}");
            }
        }
    }
    [CtfCudaFact]
    public void JointMovieAndTiltSeriesFitsRecoverPhysicalThickness()
    {
        lock(GPU.Sync)
        {
            GPU.SetDevice(0);
            foreach(bool tilted in new[] {false,true})
            {
                var options=new ProcessingOptionsMovieCTF{Voltage=300,Cs=2.7M,Amplitude=.07M,ZMin=1,ZMax=4,DoPhase=true};
                double[] truth=tilted ? new[] {2.3,.025,-.015,.2,.035,-.025,.01} : new[] {2.3,.025,-.015,.2,.01};
                var records=new List<CtfPowerSpectrum.Observation>();var geometry=new List<CtfFitGeometry>();
                for(int i=0;i<5;i++)
                {
                    Matrix3? rotation=tilted ? Matrix3.Euler(0,(float)((i-2)*.24),.17f) : null;
                    var g=new CtfFitGeometry(new[] {1.0},new[] {1.0},(i-2)*.07,.04,rotation,tilted?.13:0);
                    var pose=new double[7];g.WritePose(truth,pose,0);
                    records.Add(new(new CtfSpectrumFit(Samples(pose,i+1),300,2.7,.07),new float3(.5f),i));geometry.Add(g);
                }
                var initial=(double[])truth.Clone();initial[0]+=.008;initial[1]=initial[2]=0;initial[3]+=.015;initial[^1]=0;
                var fit=CtfFitEngine.Refine(records.ToArray(),geometry.ToArray(),initial,options);
                double thickness=Math.Sqrt(fit.Parameters[^1])*1e4;
                Assert.True(thickness >= 970 && thickness <= 1030, $"tilted {tilted}: thickness {thickness}, pose {string.Join(",",fit.Parameters)}, loss {fit.Loss}");
                Assert.InRange(Math.Abs(fit.Parameters[0]-truth[0])*1e4,0,1);
                if(tilted){Assert.InRange(Math.Abs(fit.Parameters[4]-truth[4]),0,.005);Assert.InRange(Math.Abs(fit.Parameters[5]-truth[5]),0,.005);}
            }
        }
    }
}
