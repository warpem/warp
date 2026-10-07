using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools;

public static class CtfFitEngine
{
    public sealed record Fit(double[] Parameters, double Loss, int Evaluations, float[] Coefficients = null);
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
        using var batch = new CtfGpuFitBatch(spectra.ToArray());
        var poses = new double[spectra.Count*4];
        var fits = CtfFitOptimizer.MinimizeMany(parameters =>
        {
            for (int seed = 0; seed < seeds.Count; seed++)
                for (int k = seedStarts[seed]; k < seedStarts[seed+1]; k++)
                { poses[4*k] = parameters[seed][0] + localOffsets[k]; poses[4*k+3] = parameters[seed][1]; }
            var output = batch.Evaluate(poses);
            var results = new (double Loss, double[] Gradient)[seeds.Count];
            for (int seed = 0; seed < seeds.Count; seed++)
            {
                double loss = 0; var gradient = new double[2];
                for (int k = seedStarts[seed]; k < seedStarts[seed+1]; k++)
                { loss += output[6*k]; gradient[0] += output[6*k+1]; gradient[1] += output[6*k+4]; }
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
        using var batch = new CtfGpuFitBatch(records.Select(r => r.Spectrum).ToArray());
        var result = RefineCore(records, geometry, initial, options, batch);
        int nd = geometry[0].DefocusWeights.Length;
        var poses = new double[records.Length * 4];
        for (int i = 0; i < records.Length; i++)
        {
            var local = geometry[i].Evaluate(result.Parameters);
            poses[4 * i] = local.Defocus; poses[4 * i + 1] = result.Parameters[nd];
            poses[4 * i + 2] = result.Parameters[nd + 1]; poses[4 * i + 3] = local.Phase;
        }
        batch.Evaluate(poses);
        var coefficients = batch.ReadCoefficients();
        batch.SynchronizeWeights();
        return result with { Coefficients = coefficients };
    }

    static Fit RefineCore(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry, double[] initial, ProcessingOptionsMovieCTF options, CtfGpuFitBatch batch)
    {
        int nd = geometry[0].DefocusWeights.Length, np = geometry[0].PhaseWeights.Length, n = initial.Length;
        double[] scale = new double[n], lo = new double[n], hi = new double[n];
        for (int j = 0; j < nd; j++) { scale[j] = .02; lo[j] = (double)options.ZMin; hi[j] = (double)options.ZMax; }
        for (int j = nd; j < nd + 2; j++) { scale[j] = .02; lo[j] = -.5; hi[j] = .5; }
        for (int j = nd + 2; j < nd + 2 + np; j++) { scale[j] = .1; lo[j] = 0; hi[j] = options.DoPhase ? Math.PI : 0; }
        for (int j = nd + 2 + np; j < n; j++) { scale[j] = .01; lo[j] = -.3; hi[j] = .3; }
        var local = new (double Defocus, double Phase, double SlopeX, double SlopeY)[records.Length];
        var poses = new double[records.Length * 4];
        bool UpdatePoses(double[] p)
        {
            for (int i = 0; i < records.Length; i++)
            {
                local[i] = geometry[i].Evaluate(p);
                if (!double.IsFinite(local[i].Defocus)) return false;
                poses[4 * i] = local[i].Defocus; poses[4 * i + 1] = p[nd]; poses[4 * i + 2] = p[nd + 1]; poses[4 * i + 3] = local[i].Phase;
            }
            return true;
        }
        CtfFitOptimizer.Result result = default;
        int evaluations = 0;
        double weightChange = double.PositiveInfinity;
        var changes = new double[records.Length];
        for (int pass = 0; pass < 6; pass++)
        {
            result = CtfFitOptimizer.Minimize(p =>
            {
                if (!UpdatePoses(p)) return (double.PositiveInfinity, new double[n]);
                double loss = 0; var gradient = new double[n];
                double[] output = batch.Evaluate(poses);
                var g = new double[4];
                for (int i = 0; i < records.Length; i++)
                {
                    loss += output[6 * i]; Array.Copy(output, 6 * i + 1, g, 0, 4);
                    geometry[i].Accumulate(gradient, g, local[i].SlopeX, local[i].SlopeY);
                }
                return (loss / records.Length, gradient.Select(v => v / records.Length).ToArray());
            }, initial, scale, lo, hi, 100);
            evaluations += result.Evaluations;
            initial = result.Parameters;
            if (pass == 5 || weightChange < .01) break;
            UpdatePoses(initial);
            double[] output = batch.Evaluate(poses, true);
            for (int i = 0; i < records.Length; i++) changes[i] = output[6 * i + 5];
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
