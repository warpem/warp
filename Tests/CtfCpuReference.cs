using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace Tests;

// Independent CPU oracle for GPU optimizer regressions; not shipped in WarpLib.
internal static class CtfCpuReference
{
    public static (double Defocus, double Phase) Initialize(CtfPowerSpectrum.Observation[] records, double[] offsets, ProcessingOptionsMovieCTF options)
    {
        int selected = Math.Min(9, records.Length);
        var radial = new CtfSpectrumFit[selected]; var delta = new double[selected];
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
            radial[k] = new(samples, (double)options.Voltage, (double)options.Cs, (double)options.Amplitude); delta[k] = offsets[index];
        }
        double q2max = records[0].Spectrum.Samples.Max(s => s.Q2);
        double step = Math.Min(.02, 1 / (8 * CtfSpectrumFit.Wavelength((double)options.Voltage) * 1e4 * q2max));
        int nz = Math.Max(1, (int)Math.Ceiling((double)(options.ZMax - options.ZMin) / step));
        int phases = options.DoPhase ? 12 : 1;
        var trials = new (double Df, double Phase, double Score)[(nz + 1) * phases];
        Parallel.For(0, trials.Length, index =>
        {
            double df = (double)options.ZMin + (double)(options.ZMax - options.ZMin) * (index / phases) / nz;
            double phase = (index % phases) * Math.PI / phases, score = 0;
            for (int k = 0; k < selected; k++) score += radial[k].QuickScore(df + delta[k], phase);
            trials[index] = (df, phase, score);
        });
        var seeds = new List<(double Df, double Phase, double Score)>();
        foreach (var t in trials.OrderByDescending(t => t.Score))
        {
            if (seeds.Any(s => Math.Abs(s.Df - t.Df) < step * 3 && Math.Abs(s.Phase - t.Phase) < .4)) continue;
            seeds.Add(t); if (seeds.Count == 4) break;
        }
        double best = double.PositiveInfinity, bestDf = seeds[0].Df, bestPhase = seeds[0].Phase;
        foreach (var seed in seeds)
        {
            var fit = CtfFitOptimizer.Minimize(p =>
            {
                double loss = 0; var gradient = new double[2];
                for (int k = 0; k < selected; k++) { var e = radial[k].Evaluate(p[0] + delta[k], 0, 0, p[1]); loss += e.Loss; gradient[0] += e.Gradient[0]; gradient[1] += e.Gradient[3]; }
                return (loss, gradient);
            }, new[] { seed.Df, seed.Phase }, new[] { .02, .1 }, new[] { (double)options.ZMin, 0 }, new[] { (double)options.ZMax, options.DoPhase ? Math.PI : 0 }, 35);
            if (fit.Loss < best) { best = fit.Loss; bestDf = fit.Parameters[0]; bestPhase = fit.Parameters[1]; }
        }
        return (bestDf, bestPhase);
    }

    public static CtfFitEngine.Fit Refine(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry, double[] initial, ProcessingOptionsMovieCTF options)
    {
        int nd = geometry[0].DefocusWeights.Length, np = geometry[0].PhaseWeights.Length, n = initial.Length;
        double[] scale = new double[n], lo = new double[n], hi = new double[n];
        for (int j = 0; j < nd; j++) { scale[j] = .02; lo[j] = (double)options.ZMin; hi[j] = (double)options.ZMax; }
        for (int j = nd; j < nd + 2; j++) { scale[j] = .02; lo[j] = -.5; hi[j] = .5; }
        for (int j = nd + 2; j < nd + 2 + np; j++) { scale[j] = .1; lo[j] = 0; hi[j] = options.DoPhase ? Math.PI : 0; }
        for (int j = nd + 2 + np; j < n; j++) { scale[j] = .01; lo[j] = -.3; hi[j] = .3; }
        var values = new CtfCpuSpectrum.Evaluation[records.Length];
        var local = new (double Defocus, double Phase, double SlopeX, double SlopeY)[records.Length];
        bool UpdatePoses(double[] p)
        {
            for (int i = 0; i < records.Length; i++)
            {
                local[i] = geometry[i].Evaluate(p);
                if (!double.IsFinite(local[i].Defocus)) return false;
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
                Parallel.For(0, records.Length, i => values[i] = records[i].Spectrum.Evaluate(local[i].Defocus, p[nd], p[nd + 1], local[i].Phase));
                for (int i = 0; i < records.Length; i++) { loss += values[i].Loss; geometry[i].Accumulate(gradient, values[i].Gradient, local[i].SlopeX, local[i].SlopeY); }
                return (loss / records.Length, gradient.Select(v => v / records.Length).ToArray());
            }, initial, scale, lo, hi, 100);
            evaluations += result.Evaluations;
            initial = result.Parameters;
            if (pass == 5 || weightChange < .01) break;
            UpdatePoses(initial);
            Parallel.For(0, records.Length, i => changes[i] = records[i].Spectrum.Reweight(local[i].Defocus, initial[nd], initial[nd + 1], local[i].Phase));
            weightChange = changes.Max();
        }
        return new(result.Parameters, result.Loss, evaluations);
    }

}
