using System;
using System.Linq;
using Warp;
using Warp.Tools;

namespace Tests;

// Fixed zero-thickness regressions for the original CTF objective, outside production.
internal static class CtfThinReference
{
    public static CtfFitEngine.Fit Refine(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry, double[] initial, ProcessingOptionsMovieCTF options)
    {
        using var batch = new CtfThinGpuBatch(records.Select(r => r.Spectrum).ToArray());
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
        return result with { Coefficients = coefficients, Parameters = result.Parameters.Concat(new[] { 0.0 }).ToArray() };
    }

    static CtfFitEngine.Fit RefineCore(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry, double[] initial, ProcessingOptionsMovieCTF options, CtfThinGpuBatch batch)
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

}
