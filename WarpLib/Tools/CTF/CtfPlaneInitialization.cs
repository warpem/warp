using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools;

/// <summary>Global specimen-plane initialization from spatial CTF ring shifts.
/// Each plane profiles over an independent defocus and phase for every tilt before
/// low-frequency joint refinement. Thickness is released only after this stage.</summary>
internal static class CtfPlaneInitialization
{
    sealed record Patch(CtfSpectrumFit Spectrum, CtfFitGeometry Geometry);
    sealed record Candidate(double X, double Y, double Score, double[] Defocus, double[] Phase, CtfDefocusPrior Prior = null);
    public sealed record Result(double[] Parameters, int Evaluations, CtfDefocusPrior Prior);

    public static Result Initialize(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry,
        double[] initial, ProcessingOptionsMovieCTF options)
    {
        var groups = Enumerable.Range(0,records.Length).GroupBy(i => records[i].Group).Select(g => g.ToArray()).ToArray();
        var patches = new List<Patch>(); var starts = new int[groups.Length+1];
        // Ring positions constrain inclination without relying on high-frequency envelope attenuation.
        double maximumQ2 = Math.Min(Math.Max(.01,2*records[0].Spectrum.Samples.Min(s => s.Q2)),records[0].Spectrum.Samples.Max(s => s.Q2));
        for(int group=0;group<groups.Length;group++)
        {
            starts[group]=patches.Count;var ids=groups[group];int count=Math.Min(9,ids.Length);
            // Index strides can select a diagonal of a regular patch grid, leaving one
            // component of the normal unconstrained. Cover the actual image coordinates.
            double cx=ids.Average(i=>geometry[i].X),cy=ids.Average(i=>geometry[i].Y);
            var selected=new List<int>();
            int next=ids.OrderBy(i=>Math.Pow(geometry[i].X-cx,2)+Math.Pow(geometry[i].Y-cy,2)).First();
            for(int j=0;j<count;j++)
            {
                int id=next;selected.Add(id);
                if(j+1<count)next=ids.Where(i=>!selected.Contains(i)).OrderByDescending(i=>selected.Min(k=>
                    Math.Pow(geometry[i].X-geometry[k].X,2)+Math.Pow(geometry[i].Y-geometry[k].Y,2))).First();
                var source=records[id].Spectrum;
                var samples=source.Samples.Where(s => s.Q2<=maximumQ2).ToArray();
                if(samples.Length<16) throw new ArgumentException("Inclination fitting requires at least 16 spectrum samples at 10 Å or lower frequency. Extend the low-frequency fitting range.");
                patches.Add(new(new CtfSpectrumFit(samples,source.VoltageKV,source.CsMM,source.Amplitude),geometry[id]));
            }
        }
        starts[^1]=patches.Count;
        var radial=patches.Select(p =>
        {
            double qmax=Math.Sqrt(maximumQ2);
            var samples=p.Spectrum.Samples.GroupBy(s => (int)(Math.Sqrt(s.Q2)/qmax*256)).Select(g =>
            {
                double n=g.Sum(s=>s.Count);
                return new CtfSpectrumFit.Sample(g.Sum(s=>s.Q2*s.Count)/n,g.Sum(s=>s.Q4*s.Count)/n,0,0,g.Sum(s=>s.Power*s.Count)/n,n);
            }).OrderBy(s=>s.Q2).ToArray();
            var source=p.Spectrum;return new CtfSpectrumFit(samples,source.VoltageKV,source.CsMM,source.Amplitude);
        }).ToArray();
        double zmin=(double)options.ZMin,zmax=(double)options.ZMax;
        double k=Math.PI*CtfSpectrumFit.Wavelength((double)options.Voltage)*1e4;
        double step=Math.Min(.02,Math.PI/(8*k*maximumQ2));
        int centralSteps=Math.Max(1,(int)Math.Ceiling((zmax-zmin)/step));step=(zmax-zmin)/centralSteps;
        // Cover all local defoci allowed by the geometric incidence limit, including overfocus.
        double margin=patches.Max(p => Math.Sqrt(p.Geometry.X*p.Geometry.X+p.Geometry.Y*p.Geometry.Y))*CtfFitGeometry.MaximumSlope;
        int extra=(int)Math.Ceiling(margin/step)+2, nz=centralSteps+2*extra+1;
        int phases=options.DoPhase?12:1;double tableMin=zmin-extra*step;
        var trials=new double[checked(nz*phases*2)];
        for(int z=0;z<nz;z++)for(int phase=0;phase<phases;phase++)
        { int j=z*phases+phase;trials[2*j]=tableMin+z*step;trials[2*j+1]=phase*Math.PI/phases; }
        double[] raw;
        using(var search=new CtfGpuFitBatch(radial))raw=search.Search(trials,new double[radial.Length]);
        // Patch-major tables make a full defocus/phase profile contiguous in the hot loop.
        var tables=new float[patches.Count][];
        for(int i=0;i<patches.Count;i++)
        {
            tables[i]=new float[nz*phases];
            for(int j=0;j<tables[i].Length;j++)tables[i][j]=(float)raw[j*patches.Count+i];
        }
        raw=null;
        int plane=geometry[0].ThicknessIndex-2;
        Candidate Evaluate(double x,double y,bool regularize=false)
        {
            double angle=Math.Sqrt(x*x+y*y);
            if(angle>CtfFitGeometry.MaximumInclination) return new(x,y,double.NegativeInfinity,null,null);
            double multiplier=angle>0?Math.Tan(angle)/angle:1;
            var p=(double[])initial.Clone();p[plane]=x*multiplier;p[plane+1]=y*multiplier;
            for(int j=0;j<geometry[0].DefocusWeights.Length;j++)p[j]=0;
            var shifts=new int[patches.Count];var fractions=new float[patches.Count];
            for(int i=0;i<patches.Count;i++)
            {
                if(!patches[i].Geometry.IsValidPlane(p))return new(x,y,double.NegativeInfinity,null,null);
                double shift=patches[i].Geometry.Evaluate(p).Defocus/step+extra;
                shifts[i]=(int)Math.Floor(shift);fractions[i]=(float)(shift-shifts[i]);
            }
            double total=0;var dfs=new double[groups.Length];var phaseValues=new double[groups.Length];
            var profiles=regularize?groups.Select(_=>Enumerable.Repeat(double.NegativeInfinity,centralSteps+1).ToArray()).ToArray():null;
            var phaseAt=regularize?groups.Select(_=>new int[centralSteps+1]).ToArray():null;
            for(int group=0;group<groups.Length;group++)
            {
                double best=double.NegativeInfinity;int bestZ=0,bestPhase=0;
                for(int z=0;z<=centralSteps;z++)for(int phase=0;phase<phases;phase++)
                {
                    double sum=0;
                    for(int i=starts[group];i<starts[group+1];i++)
                    {
                        int j=(z+shifts[i])*phases+phase;float f=fractions[i];
                        sum+=tables[i][j]*(1-f)+tables[i][j+phases]*f;
                    }
                    if(regularize && sum>profiles[group][z]){profiles[group][z]=sum;phaseAt[group][z]=phase;}
                    if(sum>best){best=sum;bestZ=z;bestPhase=phase;}
                }
                total+=best;dfs[group]=zmin+bestZ*step;phaseValues[group]=bestPhase*Math.PI/phases;
            }
            CtfDefocusPrior prior=null;
            if(regularize && geometry[0].DefocusWeights.Length==groups.Length)
            {
                var groupGeometry=groups.Select(g=>geometry[g[0]]).ToArray();
                int[] nodes=groupGeometry.Select(g=>Array.IndexOf(g.DefocusWeights,1.0)).ToArray();
                if(nodes.All(j=>j>=0))
                {
                    prior=CtfDefocusPrior.FromProfiles(profiles,groupGeometry.Select(g=>Math.Atan2(g.Rotation.Value.M31,g.Rotation.Value.M33)).ToArray(),nodes,zmin,step);
                    if(prior!=null)for(int g=0;g<groups.Length;g++)
                    {int z=prior.Select(g);dfs[g]=zmin+z*step;phaseValues[g]=phaseAt[g][z]*Math.PI/phases;}
                }
            }
            return new(x,y,total,dfs,phaseValues,prior);
        }
        var coordinates=new List<(double X,double Y)>{(0,0)};
        const double coarseStep=Math.PI/18;
        for(double angle=coarseStep;angle<=CtfFitGeometry.MaximumInclination;angle+=coarseStep)
        {
            int azimuths=Math.Max(6,(int)Math.Ceiling(2*Math.PI*Math.Sin(angle)/coarseStep));
            for(int j=0;j<azimuths;j++)coordinates.Add((angle*Math.Cos(j*2*Math.PI/azimuths),angle*Math.Sin(j*2*Math.PI/azimuths)));
        }
        Candidate[] Rank(List<(double X,double Y)> points)
        {
            var candidates=new Candidate[points.Count];
            Parallel.For(0,points.Count,i=>candidates[i]=Evaluate(points[i].X,points[i].Y));
            return candidates.Where(c=>double.IsFinite(c.Score)).OrderByDescending(c=>c.Score).ToArray();
        }
        List<Candidate> Select(Candidate[] ranked,double separation)
        {
            var selected=new List<Candidate>();
            foreach(var c in ranked)
            {
                if(selected.Any(s=>Math.Pow(s.X-c.X,2)+Math.Pow(s.Y-c.Y,2)<separation*separation))continue;
                selected.Add(c);if(selected.Count==4)break;
            }
            return selected;
        }
        var bestPlanes=Select(Rank(coordinates),coarseStep*.8);
        if(bestPlanes.Count==0)throw new InvalidOperationException("No specimen plane satisfies the CTF incidence limit for these tilt angles.");
        for(double spacing=coarseStep/2;spacing>=coarseStep/16;spacing/=2)
        {
            coordinates.Clear();
            foreach(var center in bestPlanes)for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)
                coordinates.Add((center.X+x*spacing,center.Y+y*spacing));
            bestPlanes=Select(Rank(coordinates),spacing*.8);
        }
        bestPlanes=bestPlanes.Select(c=>Evaluate(c.X,c.Y,true)).ToList();
        var seeds=bestPlanes.Select(c=>
        {
            var p=(double[])initial.Clone();int nd=geometry[0].DefocusWeights.Length,np=geometry[0].PhaseWeights.Length;
            double angle=Math.Sqrt(c.X*c.X+c.Y*c.Y),factor=angle>0?Math.Tan(angle)/angle:1;
            p[plane]=c.X*factor;p[plane+1]=c.Y*factor;p[^1]=0;
            for(int j=0;j<nd;j++)
            {
                double sum=0,weight=0;
                for(int g=0;g<groups.Length;g++){double w=geometry[groups[g][0]].DefocusWeights[j];sum+=w*c.Defocus[g];weight+=w;}
                if(weight>0)p[j]=sum/weight;
            }
            for(int j=0;j<np;j++)
            {
                double x=0,y=0;
                for(int g=0;g<groups.Length;g++){double w=geometry[groups[g][0]].PhaseWeights[j];x+=w*Math.Cos(2*c.Phase[g]);y+=w*Math.Sin(2*c.Phase[g]);}
                double phase=.5*Math.Atan2(y,x);p[nd+2+j]=phase<0?phase+Math.PI:phase;
            }
            return p;
        }).ToArray();
        // Independently refine the retained planes, preserving their separate L-BFGS histories.
        var spectra=seeds.SelectMany(_=>patches.Select(p=>p.Spectrum)).ToArray();
        using var batch=new CtfGpuFitBatch(spectra,
            Enumerable.Range(0,seeds.Length).SelectMany(s=>Enumerable.Repeat(s,patches.Count)).ToArray(),
            seeds.SelectMany(_=>CtfEnvelopeLayout.Angles(patches.Select(p=>p.Geometry).ToArray())).ToArray());var poses=new double[spectra.Length*7];
        var (scales,lower,upper)=CtfFitEngine.ParameterBounds(geometry[0],options);
        upper[^1]=0;
        var fits=CtfFitOptimizer.MinimizeMany(parameters=>
        {
            var valid=new bool[parameters.Length];
            for(int s=0;s<parameters.Length;s++)
            {
                valid[s]=patches.All(p=>p.Geometry.IsValidPlane(parameters[s]));
                for(int i=0;i<patches.Count;i++)patches[i].Geometry.WritePose(valid[s]?parameters[s]:seeds[s],poses,(s*patches.Count+i)*7);
            }
            var output=batch.Evaluate(poses);var result=new (double Loss,double[] Gradient)[parameters.Length];
            for(int s=0;s<parameters.Length;s++)
            {
                double loss=0;var gradient=new double[initial.Length];var local=new double[7];
                if(valid[s])for(int i=0;i<patches.Count;i++)
                {
                    int j=(s*patches.Count+i)*9;loss+=output[j];Array.Copy(output,j+1,local,0,7);
                    patches[i].Geometry.AccumulateVolume(gradient,local,parameters[s]);
                }
                if(valid[s] && bestPlanes[s].Prior!=null)loss+=bestPlanes[s].Prior.Evaluate(parameters[s],gradient);
                result[s]=(valid[s]?loss/patches.Count:double.PositiveInfinity,gradient.Select(v=>v/patches.Count).ToArray());
            }
            return result;
        },seeds,scales,lower,upper,80);
        var bestFit=fits.OrderBy(f=>f.Loss).First();
        var bestPlane=bestPlanes[Array.IndexOf(fits,bestFit)];
        var bestPrior=bestPlane.Prior;
        bestPrior?.Update(bestFit.Parameters);
        int rescueEvaluations=0;
        if(bestPrior!=null)
        {
            // Search-profile contrast is only a seed heuristic. Compare the independent
            // maximum against the consensus basin using the actual spectral objective,
            // separately for each tilt, so a real focus jump is not lost in initialization.
            var independent=Evaluate(bestPlane.X,bestPlane.Y);
            rescueEvaluations=CompareDefocusBasins(patches,starts,bestFit.Parameters,independent.Defocus,bestPrior,options);
            bestPrior.Update(bestFit.Parameters);
        }
        return new(bestFit.Parameters,fits.Sum(f=>f.Evaluations)+rescueEvaluations,bestPrior);
    }
    static int CompareDefocusBasins(List<Patch> patches,int[] starts,double[] parameters,double[] independent,
        CtfDefocusPrior prior,ProcessingOptionsMovieCTF options)
    {
        int groups=starts.Length-1;
        int[] nodes=Enumerable.Range(0,groups).Select(g=>Array.IndexOf(patches[starts[g]].Geometry.DefocusWeights,1.0)).ToArray();
        var centers=prior.Centers;var sigma=prior.Scales;
        var spectra=Enumerable.Range(0,2).SelectMany(_=>patches.Select(p=>p.Spectrum)).ToArray();
        using var batch=new CtfGpuFitBatch(spectra,Enumerable.Range(0,2*groups).SelectMany(s=>Enumerable.Repeat(s,starts[s%groups+1]-starts[s%groups])).ToArray());
        var poses=new double[spectra.Length*7];
        var seeds=Enumerable.Range(0,groups*2).Select(s=>new[]{s<groups?parameters[nodes[s]]:independent[s-groups]}).ToArray();
        var fits=CtfFitOptimizer.MinimizeMany(values=>
        {
            for(int s=0;s<values.Length;s++)
            {
                int group=s%groups,offset=s/groups*patches.Count;
                var p=(double[])parameters.Clone();p[nodes[group]]=values[s][0];
                for(int i=starts[group];i<starts[group+1];i++)patches[i].Geometry.WritePose(p,poses,7*(offset+i));
            }
            var output=batch.Evaluate(poses);
            return values.Select((v,s)=>
            {
                int group=s%groups,offset=s/groups*patches.Count;
                double residual=v[0]-centers[group],precision=1/(sigma[group]*sigma[group]);
                double loss=.5*residual*residual*precision,gradient=residual*precision;
                for(int i=starts[group];i<starts[group+1];i++)
                {loss+=output[9*(offset+i)];gradient+=output[9*(offset+i)+1];}
                return (loss,new[]{gradient});
            }).ToArray();
        },seeds,new[]{.02},new[]{(double)options.ZMin},new[]{(double)options.ZMax},40);
        for(int group=0;group<groups;group++)
        {
            var best=fits[group].Loss<=fits[group+groups].Loss?fits[group]:fits[group+groups];
            parameters[nodes[group]]=best.Parameters[0];
        }
        return fits.Sum(f=>f.Evaluations);
    }

}
