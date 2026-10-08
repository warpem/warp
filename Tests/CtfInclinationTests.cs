using System;
using System.Collections.Generic;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Tests;

public class CtfInclinationTests
{
    readonly ITestOutputHelper output;
    public CtfInclinationTests(ITestOutputHelper output) { this.output=output; }
    [Fact]
    public void InclinationDomainIsCircularAndUsesNormalizedBeamIncidence()
    {
        var geometry=new CtfFitGeometry(new[]{1.0},new[]{1.0},rotation:Matrix3.Euler(0,0,0));
        foreach(double azimuth in new[]{0.0,.3,1.5,2.7})
        {
            double[] p={2,.02,0,0,Math.Tan(70*Math.PI/180)*Math.Cos(azimuth),Math.Tan(70*Math.PI/180)*Math.Sin(azimuth),.01};
            Assert.True(geometry.IsValidPlane(p));
            p[4]=Math.Tan(87*Math.PI/180)*Math.Cos(azimuth);p[5]=Math.Tan(87*Math.PI/180)*Math.Sin(azimuth);
            Assert.False(geometry.IsValidPlane(p));
        }
        var tilted=new CtfFitGeometry(new[]{1.0},new[]{1.0},rotation:Matrix3.Euler(0,60*Helper.ToRad,0));
        double[] grazing={2,.02,0,0,-Math.Tan(27*Math.PI/180),0,.01};
        Assert.False(tilted.IsValidPlane(grazing));
    }
    static CtfSpectrumFit Spectrum(double[] pose,int seed,double noise,bool wrongPatch,double contrast=1)
    {
        var rng=new Random(seed);var samples=new List<CtfSpectrumFit.Sample>();
        double lambda=CtfSpectrumFit.Wavelength(300),kd=Math.PI*lambda*1e4,kc=-.5*Math.PI*2.7e7*Math.Pow(lambda,3);
        for(int r=27;r<240;r++)for(int sector=0;sector<8;sector++)
        {
            double q=r/640.0,q2=q*q,angle=(sector+.29)*Math.PI/8,ax=q2*Math.Cos(2*angle),ay=q2*Math.Sin(2*angle);
            double df=pose[0]+(wrongPatch?.31:0),phase=pose[3]+(wrongPatch?.8:0);
            double gamma=kd*(q2*df+ax*pose[1]+ay*pose[2])+kc*q2*q2+Math.Asin(.07)+phase;
            double model=CtfSlabModel.Power(gamma,kd*q2,pose[4],pose[5],pose[6]);
            double power=3+4*q+contrast*(1+q)*model;
            power*=1+noise*Math.Sqrt(-2*Math.Log(Math.Max(1e-12,rng.NextDouble())))*Math.Cos(2*Math.PI*rng.NextDouble());
            samples.Add(new(q2,q2*q2,ax,ay,Math.Max(.01,power),300));
        }
        return new(samples.ToArray(),300,2.7,.07);
    }
    [CtfCudaFact]
    public void GlobalPlaneInitializationRecoversLargeInclinationsFromWrongDefocusBasins()
    {
        lock(GPU.Sync)
        {
            GPU.SetDevice(0);
            var cases=new (double Angle,double Azimuth,double Noise,bool Uneven,bool Outliers)[]{(0.0,0.0,.005,false,false),
                (25.0,0.0,.005,false,false),(35.0,45.0,.005,false,false),
                (45.0,90.0,.005,false,false),(30.0,100.0,.08,true,false),
                (20.0,135.0,.04,true,true)};
            foreach(var c in cases)
            {
                const int tilts=5;int plane=tilts+3;
                double a=c.Angle*Math.PI/180,az=c.Azimuth*Math.PI/180;
                var truth=new double[tilts+6];
                for(int t=0;t<tilts;t++)truth[t]=2.1+.13*t;
                truth[tilts]=.025;truth[tilts+1]=-.015;truth[tilts+2]=.17;
                truth[plane]=Math.Tan(a)*Math.Cos(az);truth[plane+1]=Math.Tan(a)*Math.Sin(az);truth[^1]=.01;
                var records=new List<CtfPowerSpectrum.Observation>();var geometry=new List<CtfFitGeometry>();
                int side=c.Angle==45?9:3;
                for(int t=0;t<tilts;t++)for(int j=0;j<side*side;j++)
                {
                    double x=(j%side-(side-1)*.5)*.3/(side-1)+(c.Uneven?.19:0),y=(j/side-(side-1)*.5)*.26/(side-1);
                    if(c.Uneven && j==6)continue;
                    var weights=new double[tilts];weights[t]=1;
                    var g=new CtfFitGeometry(weights,new[]{1.0},x,y,Matrix3.Euler(0,(t-2)*25*Helper.ToRad,0),.09);
                    Assert.True(g.IsValidPlane(truth));
                    var pose=new double[7];g.WritePose(truth,pose,0);
                    records.Add(new(Spectrum(pose,100*t+j,c.Noise,c.Outliers&&j==4),new float3(.5f),t));geometry.Add(g);
                }
                var initial=new double[truth.Length];Array.Fill(initial,4.2,0,tilts);initial[tilts+2]=1.1;
                var options=new ProcessingOptionsMovieCTF{Voltage=300,Cs=2.7M,Amplitude=.07M,ZMin=1,ZMax=5,DoPhase=true};
                var fit=CtfFitEngine.Refine(records.ToArray(),geometry.ToArray(),initial,options);var p=fit.Parameters;
                double norm=Math.Sqrt(1+p[plane]*p[plane]+p[plane+1]*p[plane+1]);
                double cosine=(1+p[plane]*truth[plane]+p[plane+1]*truth[plane+1])/norm/Math.Sqrt(1+truth[plane]*truth[plane]+truth[plane+1]*truth[plane+1]);
                double error=Math.Acos(Math.Clamp(cosine,-1,1))*180/Math.PI;
                double dfRms=Math.Sqrt(Enumerable.Range(0,tilts).Average(t=>Math.Pow((p[t]-truth[t])*1e4,2)));
                output.WriteLine($"inclination {c.Angle} azimuth {c.Azimuth} noise {c.Noise} uneven {c.Uneven} outliers {c.Outliers}: normal error {error:F4} deg, defocus RMS {dfRms:F3} A, thickness {Math.Sqrt(p[^1])*1e4:F2} A, initialization {fit.PlaneInitializationSeconds:F3}s");
                Assert.True(error<(c.Noise>.01?2:.2),$"Normal error {error} degrees for {c}");
                Assert.InRange(dfRms,0,c.Noise>.01?60:5);
                Assert.False(fit.PlaneAtBoundary);
            }
        }
    }
    [CtfCudaFact]
    public void WeakHighTiltBorrowsSupportWithoutSuppressingARealFocusJump()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            double[] angles = { -60, -25, -15, 0, 15, 25, 60 };
            int tilts = angles.Length, plane = tilts + 3;
            var truth = new double[tilts + 6];
            for (int t = 0; t < tilts; t++) truth[t] = 1.8 + .08 * Math.Sin(angles[t] * Math.PI / 180);
            truth[tilts - 1] = 3.0; // A genuine jump with clear rings must remain possible.
            truth[tilts] = .02; truth[tilts + 1] = -.01;
            truth[plane] = .1; truth[plane + 1] = -.06; truth[^1] = .01;
            var records = new List<CtfPowerSpectrum.Observation>();
            var geometry = new List<CtfFitGeometry>();
            for (int t = 0; t < tilts; t++) for (int j = 0; j < 9; j++)
            {
                var weights = new double[tilts]; weights[t] = 1;
                var g = new CtfFitGeometry(weights, new[] { 1.0 }, (j%3-1)*.15, (j/3-1)*.13,
                    Matrix3.Euler(0, (float)(angles[t]*Math.PI/180), 0), .09);
                var pose = new double[7]; g.WritePose(truth, pose, 0);
                records.Add(new(Spectrum(pose, 300*t+j, .015, false, t == 0 ? 0 : 1), new float3(.5f), t));
                geometry.Add(g);
            }
            var initial = new double[truth.Length]; Array.Fill(initial, 4.2, 0, tilts);
            var options = new ProcessingOptionsMovieCTF { Voltage=300, Cs=2.7M, Amplitude=.07M, ZMin=.5M, ZMax=5, DoPhase=false };
            var fit = CtfFitEngine.Refine(records.ToArray(), geometry.ToArray(), initial, options);
            output.WriteLine($"Weak tilt: {fit.Parameters[0]:F5} um; real jump: {fit.Parameters[tilts-1]:F5} um");
            Assert.InRange(Math.Abs(fit.Parameters[0]-truth[0]), 0, .15);
            for (int t = 1; t < tilts; t++) Assert.InRange(Math.Abs(fit.Parameters[t]-truth[t]), 0, .01);
        }
    }

    [CtfCudaFact]
    public void DenseMovieGridKeepsLocalDefocusWithFewerPatchesThanNodes()
    {
        lock(GPU.Sync)
        {
            GPU.SetDevice(0);
            var positions=Enumerable.Range(0,16).Select(i=>new float3((i%4+.5f)/4,(i/4+.5f)/4,.5f)).ToArray();
            var dw=CtfFitGeometry.GridWeights(new int3(6,6,1),positions);
            var geometry=dw.Select(w=>new CtfFitGeometry(w,new[]{1.0})).ToArray();
            var truth=new double[40];
            for(int i=0;i<36;i++)truth[i]=2+.04*(i%6)/5+.03*(i/6)/5;
            truth[36]=.02;truth[37]=-.01;truth[^1]=.01;
            var records=geometry.Select((g,i)=>
            {
                var pose=new double[7];g.WritePose(truth,pose,0);
                return new CtfPowerSpectrum.Observation(Spectrum(pose,300+i,.005,false),positions[i],0);
            }).ToArray();
            var initial=(double[])truth.Clone();Array.Fill(initial,2.035,0,36);
            var options=new ProcessingOptionsMovieCTF{Voltage=300,Cs=2.7M,Amplitude=.07M,ZMin=1,ZMax=4};
            var fit=CtfFitEngine.Refine(records,geometry,initial,options);
            Assert.Null(fit.Reliability);
            foreach(var g in geometry)Assert.InRange(Math.Abs(g.Evaluate(fit.Parameters).Defocus-g.Evaluate(truth).Defocus),0,.002);
            Assert.True(geometry.Max(g=>g.Evaluate(fit.Parameters).Defocus)-geometry.Min(g=>g.Evaluate(fit.Parameters).Defocus)>.03);
        }
    }

}
