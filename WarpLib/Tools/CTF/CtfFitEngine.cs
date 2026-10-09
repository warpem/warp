using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools;

public static class CtfFitEngine
{
    public sealed record Fit(double[] Parameters, double Loss, int Evaluations, float[] Coefficients = null, double PlaneInitializationSeconds = 0, bool PlaneAtBoundary = false, CtfFitReliability.Curve[] Reliability = null);
    // A radial, geometry-aware search locates several defocus basins before the full angular fit.
    public static (double Defocus, double Phase) Initialize(CtfPowerSpectrum.Observation[] records, double[] offsets, ProcessingOptionsMovieCTF options)
        => InitializeMany(new[] { records }, new[] { offsets }, options)[0];

    public static (double Defocus, double Phase)[] InitializeMany(CtfPowerSpectrum.Observation[][] groups, double[][] offsets, ProcessingOptionsMovieCTF options)
    {
        var radial = new List<CtfSpectrumFit>(); var delta = new List<double>();
        var starts = new int[groups.Length + 1];
        for (int group = 0; group < groups.Length; group++)
        {
            var records = groups[group];
            int selected = Math.Min(9, records.Length);
            starts[group] = radial.Count;
            for (int k = 0; k < selected; k++)
            {
                int index = selected == 1 ? records.Length / 2 : k * (records.Length - 1) / (selected - 1);
                var s = records[index].Spectrum;
                double qmax = Math.Sqrt(s.Samples.Max(v => v.Q2));
                var samples = s.Samples.GroupBy(v => (int)(Math.Sqrt(v.Q2) / qmax * 256)).Select(g =>
                {
                    double n = g.Sum(v => v.Count);
                    return new CtfSpectrumFit.Sample(g.Sum(v => v.Q2 * v.Count) / n, g.Sum(v => v.Q4 * v.Count) / n, 0, 0, g.Sum(v => v.Power * v.Count) / n, n);
                }).OrderBy(v => v.Q2).ToArray();
                radial.Add(new(samples, (double)options.Voltage, (double)options.Cs, (double)options.Amplitude));
                delta.Add(offsets[group][index]);
            }
        }
        starts[^1] = radial.Count;
        double q2max = groups[0][0].Spectrum.Samples.Max(s => s.Q2);
        double step = Math.Min(.02, 1 / (8 * CtfSpectrumFit.Wavelength((double)options.Voltage) * 1e4 * q2max));
        int nz = Math.Max(1, (int)Math.Ceiling((double)(options.ZMax - options.ZMin) / step));
        int phases = options.DoPhase ? 12 : 1, trialCount = (nz + 1) * phases;
        var trials = new double[trialCount * 2];
        for (int index = 0; index < trialCount; index++)
        {
            trials[2*index] = (double)options.ZMin + (double)(options.ZMax - options.ZMin) * (index / phases) / nz;
            trials[2*index+1] = (index % phases) * Math.PI / phases;
        }
        double[] scores;
        using (var search = new CtfGpuFitBatch(radial.ToArray())) scores = search.Search(trials, delta.ToArray());
        var seeds = new List<(int Group, double Df, double Phase)>();
        for (int group = 0; group < groups.Length; group++)
        {
            var ranked = new (double Df, double Phase, double Score)[trialCount];
            for (int index = 0; index < trialCount; index++)
            {
                double score = 0;
                for (int k = starts[group]; k < starts[group+1]; k++) score += scores[index * radial.Count + k];
                ranked[index] = (trials[2*index], trials[2*index+1], score);
            }
            int first = seeds.Count;
            foreach (var t in ranked.OrderByDescending(t => t.Score))
            {
                if (seeds.Skip(first).Any(s => Math.Abs(s.Df-t.Df) < step*3 && Math.Abs(s.Phase-t.Phase) < .4)) continue;
                seeds.Add((group, t.Df, t.Phase));
                if (seeds.Count-first == 4) break;
            }
        }
        var spectra = new List<CtfSpectrumFit>(); var localOffsets = new List<double>();
        var seedStarts = new int[seeds.Count+1];
        for (int seed = 0; seed < seeds.Count; seed++)
        {
            seedStarts[seed] = spectra.Count;
            for (int k = starts[seeds[seed].Group]; k < starts[seeds[seed].Group+1]; k++)
            { spectra.Add(radial[k]); localOffsets.Add(delta[k]); }
        }
        seedStarts[^1] = spectra.Count;
        using var batch = new CtfGpuFitBatch(spectra.ToArray(), Enumerable.Range(0,seeds.Count).SelectMany(i=>Enumerable.Repeat(i,seedStarts[i+1]-seedStarts[i])).ToArray());
        var poses = new double[spectra.Count*7];
        var fits = CtfFitOptimizer.MinimizeMany(parameters =>
        {
            for (int seed = 0; seed < seeds.Count; seed++)
                for (int k = seedStarts[seed]; k < seedStarts[seed+1]; k++)
                { poses[7*k] = parameters[seed][0] + localOffsets[k]; poses[7*k+3] = parameters[seed][1]; }
            var output = batch.Evaluate(poses);
            var results = new (double Loss, double[] Gradient)[seeds.Count];
            for (int seed = 0; seed < seeds.Count; seed++)
            {
                double loss = 0; var gradient = new double[2];
                for (int k = seedStarts[seed]; k < seedStarts[seed+1]; k++)
                { loss += output[9*k]; gradient[0] += output[9*k+1]; gradient[1] += output[9*k+4]; }
                results[seed] = (loss, gradient);
            }
            return results;
        }, seeds.Select(s => new[] { s.Df, s.Phase }).ToArray(), new[] { .02, .1 },
            new[] { (double)options.ZMin, 0 }, new[] { (double)options.ZMax, options.DoPhase ? Math.PI : 0 }, 35);
        var best = Enumerable.Repeat(double.PositiveInfinity, groups.Length).ToArray();
        var result = new (double Defocus, double Phase)[groups.Length];
        for (int i = 0; i < seeds.Count; i++) if (fits[i].Loss < best[seeds[i].Group])
        { int g = seeds[i].Group; best[g] = fits[i].Loss; result[g] = (fits[i].Parameters[0], fits[i].Parameters[1]); }
        return result;
    }

    public static Fit Refine(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry, double[] initial, ProcessingOptionsMovieCTF options)
    {
        if (initial.Length != geometry[0].ThicknessIndex+1) throw new ArgumentException("CTF parameters must include squared specimen thickness.");
        initial = (double[])initial.Clone();
        int planeEvaluations=0;double planeSeconds=0;CtfDefocusPrior prior=null;
        if(geometry[0].Rotation.HasValue)
        {
            var timer=System.Diagnostics.Stopwatch.StartNew();
            var plane=CtfPlaneInitialization.Initialize(records,geometry,initial,options);
            initial=plane.Parameters;prior=plane.Prior;planeEvaluations=plane.Evaluations;planeSeconds=timer.Elapsed.TotalSeconds;
        }
        var envelopeAngles=CtfEnvelopeLayout.Angles(geometry);
        using var batch = new CtfGpuFitBatch(records.Select(r => r.Spectrum).ToArray(),tiltAngles:envelopeAngles);
        int searchEvaluations = SeedThickness(records,geometry,initial,batch);
        // Shared geometry, astigmatism and thickness use the joint, full-band fit.
        // A narrow reliable band in a weak tilt cannot separately identify thickness
        // and envelope, so the subsequent local correction must retain these estimates.
        var result = RefineCore(records, geometry, initial, options, batch, prior);
        CtfFitReliability.Result reliability=null;
        if(prior!=null && geometry.All(g=>g.PatchWidth>0))
        {
            searchEvaluations+=result.Evaluations;
            using var training=new CtfGpuFitBatch(records.Select(r=>r.Spectrum.WithFitWeights(
                r.Spectrum.Samples.Select(s=>CtfFitReliability.AngularFold(s)>0?1f:0f).ToArray())).ToArray(),tiltAngles:envelopeAngles);
            var trainingFit=RefineCore(records,geometry,(double[])result.Parameters.Clone(),options,training,prior,true);
            searchEvaluations+=trainingFit.Evaluations;
            reliability=CtfFitReliability.Estimate(records,geometry,trainingFit.Parameters);
        }
        // SPA grids may have more defocus nodes than patches. They do not enter this
        // tilt-specific correction and are not required to have spatial replication.
        var supportedSpectra=reliability==null?null:records.Select((r,i)=>
            r.Spectrum.WithFitWeights(reliability.FrequencyWeight[i].Select(w=>w*reliability.PatchWeight[i]).ToArray())).ToArray();
        using var supported=supportedSpectra==null?null:new CtfGpuFitBatch(supportedSpectra,tiltAngles:envelopeAngles);
        var fittingBatch=supported??batch;
        if(supported!=null)result=RefineCore(records,geometry,(double[])result.Parameters.Clone(),options,supported,prior,true);
        result=result with { Reliability=reliability?.Groups };
        result = result with { Evaluations = result.Evaluations + searchEvaluations + planeEvaluations, PlaneInitializationSeconds = planeSeconds,
            PlaneAtBoundary = geometry.Any(g => !g.IsValidPlane(result.Parameters,CtfFitGeometry.MinimumBeamCosine*1.01)) };
        var poses = new double[records.Length * 7];
        for (int i = 0; i < records.Length; i++)
        {
            geometry[i].WritePose(result.Parameters, poses, 7*i);
        }
        fittingBatch.Evaluate(poses);
        var coefficients = fittingBatch.ReadCoefficients();
        if(supportedSpectra!=null)
        {
            int stride=records[0].Spectrum.KnotCount*2;
            for(int i=0;i<records.Length;i++)for(int j=0;j<stride;j++)
                coefficients[i*stride+j]*=(float)(supportedSpectra[i].PowerScale/records[i].Spectrum.PowerScale);
        }
        fittingBatch.SynchronizeWeights();
        return result with { Coefficients = coefficients };
    }

    // Search thickness before IRLS can mistake a sinc reversal for contaminated bins.
    // The quarter-period grid covers 0–1 µm; the optimizer then refines continuously.
    static int SeedThickness(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry, double[] initial, CtfGpuFitBatch batch)
    {
        int t = geometry[0].ThicknessIndex;
        double k = Math.PI*CtfSpectrumFit.Wavelength(records[0].Spectrum.VoltageKV)*1e4;
        double q2 = records[0].Spectrum.Samples.Max(s => s.Q2);
        double factor = geometry.Max(g => Math.Sqrt(g.SlabGeometry(initial).Factor));
        int steps = Math.Max(16, (int)Math.Ceiling(4*k*q2*factor/Math.PI));
        var poses = new double[records.Length*7];
        double best = double.PositiveInfinity, bestSquared = initial[t];
        for (int step = -1; step <= steps; step++)
        {
            initial[t] = step < 0 ? bestSquared : Math.Pow((double)step/steps,2);
            for (int i = 0; i < records.Length; i++) geometry[i].WritePose(initial,poses,7*i);
            var output = batch.Evaluate(poses); double loss = 0;
            for (int i = 0; i < records.Length; i++) loss += output[9*i];
            if (loss < best) { best = loss; bestSquared = initial[t]; }
        }
        initial[t] = bestSquared;
        return steps+2;
    }

    internal static (double[] Scale,double[] Lower,double[] Upper) ParameterBounds(CtfFitGeometry geometry,ProcessingOptionsMovieCTF options)
    {
        int nd=geometry.DefocusWeights.Length,np=geometry.PhaseWeights.Length,n=geometry.ThicknessIndex+1;
        var scale=new double[n];var lo=new double[n];var hi=new double[n];
        for(int j=0;j<nd;j++){scale[j]=.02;lo[j]=(double)options.ZMin;hi[j]=(double)options.ZMax;}
        for(int j=nd;j<nd+2;j++){scale[j]=.02;lo[j]=-.5;hi[j]=.5;}
        for(int j=nd+2;j<nd+2+np;j++){scale[j]=.1;lo[j]=0;hi[j]=options.DoPhase?Math.PI:0;}
        // IsValidPlane enforces the circular/geometric domain inside these enclosing bounds.
        for(int j=nd+2+np;j<n-1;j++){scale[j]=.01;lo[j]=-CtfFitGeometry.MaximumSlope;hi[j]=CtfFitGeometry.MaximumSlope;}
        scale[n-1]=.0025;lo[n-1]=0;hi[n-1]=1;
        return (scale,lo,hi);
    }

    static Fit RefineCore(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry, double[] initial, ProcessingOptionsMovieCTF options, CtfGpuFitBatch batch, CtfDefocusPrior prior, bool defocusOnly=false)
    {
        int n=initial.Length;
        var (scale,lo,hi)=ParameterBounds(geometry[0],options);
        if(defocusOnly)for(int j=geometry[0].DefocusWeights.Length;j<n;j++)lo[j]=hi[j]=initial[j];
        var poses = new double[records.Length * 7];
        bool UpdatePoses(double[] p)
        {
            for (int i = 0; i < records.Length; i++)
            {
                geometry[i].WritePose(p,poses,7*i);
                if (!double.IsFinite(poses[7*i])) return false;
            }
            return true;
        }
        CtfFitOptimizer.Result result = default;
        int evaluations = 0;
        double weightChange = double.PositiveInfinity;
        var changes = new double[records.Length];
        for (int pass = 0; pass < 7; pass++)
        {
            result = CtfFitOptimizer.Minimize(p =>
            {
                if (!UpdatePoses(p)) return (double.PositiveInfinity, new double[n]);
                double loss = 0; var gradient = new double[n];
                double[] output = batch.Evaluate(poses);
                var g = new double[7];
                for (int i = 0; i < records.Length; i++)
                {
                    loss += output[9 * i]; Array.Copy(output, 9 * i + 1, g, 0, 7);
                    geometry[i].AccumulateVolume(gradient, g, p);
                }
                if(prior!=null)loss+=prior.Evaluate(p,gradient);
                return (loss / records.Length, gradient.Select(v => v / records.Length).ToArray());
            }, initial, scale, lo, hi, 100);
            evaluations += result.Evaluations;
            initial = result.Parameters;
            // Reprofile once after astigmatism/phase/defocus have settled. Their initial
            // errors can otherwise make the thickness grid select the wrong sinc lobe.
            if (pass == 0) { prior?.Update(initial); if(!defocusOnly)evaluations += SeedThickness(records,geometry,initial,batch); continue; }
            if (pass == 6 || weightChange < .01) break;
            UpdatePoses(initial);
            double[] output = batch.Evaluate(poses, true);
            for (int i = 0; i < records.Length; i++) changes[i] = output[9 * i + 8];
            weightChange = changes.Max();
        }
        return new(result.Parameters, result.Loss, evaluations);
    }

    public static CTF MakeCtf(ProcessingOptionsMovieCTF options, double defocus, double ax, double ay, double phase) => new()
    {
        PixelSize = options.BinnedPixelSizeMean,
        Voltage = options.Voltage,
        Cs = options.Cs,
        Amplitude = options.Amplitude,
        Defocus = (decimal)defocus,
        DefocusDelta = (decimal)(2 * Math.Sqrt(ax * ax + ay * ay)),
        DefocusAngle = (decimal)(.5 * Math.Atan2(ay, ax) * 180 / Math.PI),
        PhaseShift = (decimal)(phase / Math.PI)
    };
}
