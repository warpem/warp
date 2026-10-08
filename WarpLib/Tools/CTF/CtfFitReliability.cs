using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools;

/// <summary>Spatial replication of CTF oscillations. This is independent of the number
/// of defocus nodes: only tilt-series groups with a shared defocus are supported.</summary>
public static class CtfFitReliability
{
    public sealed record Curve(double[] Frequency, float[] Agreement, float[] Weight, int IndependentPatches)
    {
        // End of the first sustained supported band, at half of full fit weight.
        // Zero denotes unknown/no sustained support; it is not a resolution claim.
        public double HalfWeightResolution
        {
            get
            {
                if (IndependentPatches < 4) return 0;
                int start = -1, last = -1, bad = 0;
                for (int i = 0; i < Frequency.Length; i++)
                {
                    if (Frequency[i] > 0 && float.IsFinite(Agreement[i]) && Weight[i] >= .5f)
                    { if (start < 0) start = i; if (i - start >= BinsPerCycle) last = i; bad = 0; }
                    else { start = -1; if (last >= 0 && ++bad >= BinsPerCycle) break; }
                }
                return last >= 0 ? 1 / Frequency[last] : 0;
            }
        }
    }
    public sealed record Result(Curve[] Groups, float[] PatchWeight, float[][] FrequencyWeight);

    // Equal-phase bins make an oscillation window meaningful across the entire spectrum.
    const int BinsPerCycle = 16;

    // Interleaved angular wedges keep mean cos(2 theta) and sin(2 theta) near zero
    // in each fold. Splitting into two quadrants would confound defocus with astigmatism.
    public static int AngularFold(CtfSpectrumFit.Sample s)
    {
        float q2 = (float)s.Q2;
        float sine4 = 2 * (float)s.AstigX * (float)s.AstigY / (q2 * q2);
        return sine4 > .5f ? 1 : sine4 < -.5f ? -1 : 0;
    }

    public static Result Estimate(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry,
        double[] parameters)
    {
        int nd = geometry[0].DefocusWeights.Length;
        var groups = Enumerable.Range(0, records.Length).GroupBy(i => records[i].Group).OrderBy(g => g.Key).ToArray();
        var curves = new Curve[groups.Length];
        var patchWeights = new float[records.Length];
        var frequencyWeights = new float[records.Length][];
        double lambda = CtfSpectrumFit.Wavelength(records[0].Spectrum.VoltageKV);
        double kd = Math.PI * lambda * 1e4, kc = -.5 * Math.PI * records[0].Spectrum.CsMM * 1e7 * Math.Pow(lambda, 3);
        Parallel.For(0, groups.Length, group =>
        {
            int[] ids = groups[group].ToArray();
            double centerDf = ids.Average(i => geometry[i].DefocusWeights.Select((w, j) => w * parameters[j]).Sum());
            double phase = ids.Average(i => geometry[i].Evaluate(parameters).Phase) + Math.Asin(records[ids[0]].Spectrum.Amplitude);
            double qmax = Math.Sqrt(records[ids[0]].Spectrum.Samples.Max(s => s.Q2));
            int bins = Math.Max(32, (int)Math.Ceiling((kd * centerDf * qmax * qmax + kc * Math.Pow(qmax, 4) + phase) / Math.PI * BinsPerCycle) + 2);
            var radial = new float[ids.Length][];
            var support = new float[ids.Length][];
            var training = new float[ids.Length][];
            var trainingSupport = new float[ids.Length][];
            var positions = new (double X, double Y, double Width)[ids.Length];
            for (int j = 0; j < ids.Length; j++)
            {
                int id = ids[j]; var g = geometry[id]; var s = records[id].Spectrum; var local = g.Evaluate(parameters);
                radial[j] = new float[bins]; support[j] = new float[bins];
                training[j] = new float[bins]; trainingSupport[j] = new float[bins];
                positions[j] = (g.X, g.Y, g.PatchWidth);
                for (int k = 0; k < s.Samples.Length; k++)
                {
                    var sample = s.Samples[k];
                    // Separate angular sectors for patch selection and validation,
                    // with guards around their boundaries. Ranking patches
                    // on the same noise used for validation would fabricate support.
                    int fold = AngularFold(sample); if (fold == 0) continue;
                    var values = fold > 0 ? training[j] : radial[j];
                    var counts = fold > 0 ? trainingSupport[j] : support[j];
                    float gamma = (float)kd * ((float)sample.Q2 * (float)local.Defocus + (float)sample.AstigX * (float)parameters[nd] + (float)sample.AstigY * (float)parameters[nd + 1]) + (float)kc * (float)sample.Q4 + (float)(local.Phase + Math.Asin(s.Amplitude));
                    float coordinate = gamma / MathF.PI * BinsPerCycle;
                    int b = (int)MathF.Floor(coordinate); if (b < 0 || b >= bins - 1) continue;
                    float f = coordinate - b, count = (float)sample.Count;
                    // Use observed power only; log-power detrending below removes the smooth
                    // background. Fitted nuisance curves could leak training noise.
                    float value = (float)(sample.Power / s.PowerScale);
                    values[b] += value * count * (1 - f); values[b + 1] += value * count * f;
                    counts[b] += count * (1 - f); counts[b + 1] += count * f;
                }
                for (int b = 0; b < bins; b++)
                {
                    radial[j][b] = support[j][b] > 0 ? MathF.Log(MathF.Max(1e-20f, radial[j][b] / support[j][b])) : float.NaN;
                    training[j][b] = trainingSupport[j][b] > 0 ? MathF.Log(MathF.Max(1e-20f, training[j][b] / trainingSupport[j][b])) : float.NaN;
                }
            }
            var frequencies = Enumerable.Range(0, bins).Select(b =>
            {
                double target = b * Math.PI / BinsPerCycle - phase;
                // Before the first phase bin, no low-frequency root exists. The
                // generic solver would otherwise return the distant Cs branch.
                double u = target > 0 ? CtfFitDiagnostics.AlignFrequencySquared((float)target, (float)(kd * centerDf), (float)kc, 0) : 0;
                return u > 0 ? Math.Sqrt(u) : 0;
            }).ToArray();
            // Thickness and the tilted patch aperture can reverse the oscillatory
            // contrast. Do not use the current thickness to decide which frequencies
            // are trustworthy: that would suppress evidence contradicting that fit.
            var measured = Measure(training, radial, positions, frequencies);
            curves[group] = measured.Curve;
            for (int j = 0; j < ids.Length; j++)
            {
                int id = ids[j]; patchWeights[id] = measured.PatchWeight[j];
                frequencyWeights[id] = records[id].Spectrum.Samples.Select(s =>
                {
                    double coordinate = (kd * centerDf * s.Q2 + kc * s.Q4 + phase) / Math.PI * BinsPerCycle;
                    int b = Math.Clamp((int)coordinate, 0, bins - 2); double f = Math.Clamp(coordinate - b, 0, 1);
                    return (float)((1 - f) * measured.Curve.Weight[b] + f * measured.Curve.Weight[b + 1]);
                }).ToArray();
            }
        });
        return new(curves, patchWeights, frequencyWeights);
    }

    public sealed record Measurement(Curve Curve, float[] PatchWeight);

    /// <summary>Separated for calibration with signal-free and contaminated spectra.
    /// Selection and validation rows must use independent samples, in equal CTF-phase bins (16 per cycle).</summary>
    public static Measurement Measure(float[][] selectionSpectra, float[][] spectra, (double X, double Y, double Width)[] positions, double[] frequencies)
    {
        int bins = frequencies.Length;
        var quality = new float[spectra.Length];
        int[] anchor = Enumerable.Range(0, bins).Where(b => frequencies[b] >= 1.0 / 30 && frequencies[b] <= 1.0 / 10).ToArray();
        var cosine = Enumerable.Range(0, bins).Select(b => -(float)Math.Cos(2 * Math.PI * b / BinsPerCycle)).ToArray();
        var sine = Enumerable.Range(0, bins).Select(b => (float)Math.Sin(2 * Math.PI * b / BinsPerCycle)).ToArray();
        for (int i = 0; i < spectra.Length; i++)
        {
            double rc = Correlation(selectionSpectra[i], cosine, anchor), rs = Correlation(selectionSpectra[i], sine, anchor);
            double r = Math.Sqrt(rc * rc + rs * rs);
            quality[i] = double.IsFinite(r) ? (float)Math.Pow(Math.Clamp((r - .15) / .45, 0, 1), 2) : 0;
        }
        // Select actual non-overlapping patches. Fixed image quadrants would reject a
        // useful island of ice surrounded by a grid bar, or mix good and empty regions.
        var chosen = new List<int>();
        foreach (int i in Enumerable.Range(0, spectra.Length).OrderByDescending(i => quality[i]))
        {
            if (quality[i] <= .01) continue;
            if (chosen.Any(j => Math.Abs(positions[i].X - positions[j].X) < .5 * (positions[i].Width + positions[j].Width) &&
                               Math.Abs(positions[i].Y - positions[j].Y) < .5 * (positions[i].Width + positions[j].Width))) continue;
            chosen.Add(i);
        }
        var agreement = Enumerable.Repeat(float.NaN, bins).ToArray();
        var weight = new float[bins];
        // With insufficient spatial replication, support is unknown, not evidence
        // of absence. Retain the supplied band; callers can report the missing check.
        if (chosen.Count < 4)
        {
            Array.Fill(weight, 1);
            return new(new(frequencies, agreement, weight, chosen.Count), quality);
        }
        // Balance good patches between four pools; no patch has an unbounded vote.
        var pools = Enumerable.Range(0, 4).Select(_ => new List<int>()).ToArray();
        var totals = new double[4];
        foreach (int i in chosen)
        {
            int pool = Array.IndexOf(totals, totals.Min()); pools[pool].Add(i); totals[pool] += quality[i];
        }
        var pooled = pools.Select(ids => Enumerable.Range(0, bins).Select(b =>
        {
            var valid = ids.Where(i => float.IsFinite(spectra[i][b])).ToArray();
            if (valid.Length == 0) return float.NaN;
            // Winsorize excursions before pooling. Patch power is normalized before
            // this step, so a high-contrast edge cannot dominate by its absolute scale.
            var values = valid.Select(i => (double)spectra[i][b]).OrderBy(v => v).ToArray();
            double median = values[values.Length / 2];
            var deviations = values.Select(v => Math.Abs(v - median)).OrderBy(v => v).ToArray();
            double width = Math.Max(1e-8, 4 * 1.4826 * deviations[deviations.Length / 2]);
            return (float)(valid.Sum(i => quality[i] * Math.Clamp(spectra[i][b], median - width, median + width)) / valid.Sum(i => quality[i]));
        }).ToArray()).ToArray();
        // Use every usable patch for evidence, correcting effective sample size for
        // Hann-window overlap. Throwing away all overlapping patches wastes most of
        // the information: at half-window spacing their power correlation is only 1/36.
        var usable = Enumerable.Range(0, spectra.Length).Where(i => quality[i] > .01).ToArray();
        var overlap = new double[spectra.Length, spectra.Length];
        foreach (int i in usable) foreach (int j in usable)
            {
                double width = .5 * (positions[i].Width + positions[j].Width);
                double c = HannOverlap(Math.Abs(positions[i].X - positions[j].X) / width) *
                         HannOverlap(Math.Abs(positions[i].Y - positions[j].Y) / width);
                overlap[i, j] = c * c;
            }
        double fullCovarianceWeight = usable.Sum(i => usable.Sum(j => (double)quality[i] * quality[j] * overlap[i, j]));
        for (int b = 0; b < bins; b++)
        {
            if (b % 4 != 0 && b != bins - 1) continue;
            int[] window = Enumerable.Range(Math.Max(0, b - BinsPerCycle), Math.Min(bins, b + BinsPerCycle + 1) - Math.Max(0, b - BinsPerCycle)).ToArray();
            var correlations = new List<double>();
            for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++)
                {
                    double pair = Correlation(pooled[i], pooled[j], window); if (double.IsFinite(pair)) correlations.Add(pair);
                }
            if (correlations.Count < 3) continue;
            correlations.Sort();
            double r = correlations[correlations.Count / 2]; agreement[b] = (float)r;
            // Test both quadratures of coherent CTF oscillations. The phase of a
            // still-imperfect defocus fit must not veto the rings needed to correct it. A fixed
            // correlation cutoff would wrongly discard weak but reproducible rings
            // when the field contains many usable patches. Spatial variance accounts
            // for frequency-bin correlation without pretending padded bins are independent.
            var projections = usable.Select(i => (Index: i, Value: Correlation(spectra[i], cosine, window), Quadrature: Correlation(spectra[i], sine, window), Weight: (double)quality[i]))
                .Where(v => double.IsFinite(v.Value) && double.IsFinite(v.Quadrature) && v.Weight > 0).ToArray();
            double sw = projections.Sum(v => v.Weight);
            double sw2 = projections.Length == usable.Length ? fullCovarianceWeight :
                projections.Sum(v => projections.Sum(u => v.Weight * u.Weight * overlap[v.Index, u.Index]));
            double neff = sw2 > 0 ? sw * sw / sw2 : 0;
            if (neff <= 3) continue;
            double mx = projections.Sum(v => v.Weight * v.Value) / sw;
            double my = projections.Sum(v => v.Weight * v.Quadrature) / sw;
            double denominator = sw - sw2 / sw;
            double vx = projections.Sum(v => v.Weight * Math.Pow(v.Value - mx, 2)) / denominator + 1e-8;
            double vy = projections.Sum(v => v.Weight * Math.Pow(v.Quadrature - my, 2)) / denominator + 1e-8;
            double cov = projections.Sum(v => v.Weight * (v.Value - mx) * (v.Quadrature - my)) / denominator;
            double statistic = Math.Sqrt(Math.Max(0, neff * (vy * mx * mx - 2 * cov * mx * my + vx * my * my) / Math.Max(1e-16, vx * vy - cov * cov)));
            // Conservative finite-sample evidence threshold, calibrated on noise controls.
            // This taper is not a p-value: initial geometry and coarse basin selection
            // precede the angular split, and neighboring frequency windows overlap.
            double threshold = 2.5 * Math.Sqrt((neff - 1) / (neff - 3));
            weight[b] = (float)Math.Pow(Math.Clamp((statistic - threshold) / 2, 0, 1), 2);
        }
        for (int b = 0; b < bins - 1; b += 4)
        {
            int end = Math.Min(b + 4, bins - 1);
            for (int i = b + 1; i < end; i++)
            { float f = (float)(i - b) / (end - b); agreement[i] = (1 - f) * agreement[b] + f * agreement[end]; weight[i] = (1 - f) * weight[b] + f * weight[end]; }
        }
        return new(new(frequencies, agreement, weight, chosen.Count), quality);
    }

    internal static double HannOverlap(double separation)
    {
        if (separation >= 1) return 0;
        return (1 - separation) * (2 + Math.Cos(2 * Math.PI * separation)) / 3 + Math.Sin(2 * Math.PI * separation) / (2 * Math.PI);
    }

    // Remove a smooth local trend before measuring oscillation agreement. NaN bins
    // are excluded rather than counted as shared zeroes.
    static double Correlation(float[] a, float[] b, int[] indices)
    {
        var ids = indices.Where(i => float.IsFinite(a[i]) && float.IsFinite(b[i])).ToArray();
        if (ids.Length < 16) return double.NaN;
        double[] x = ids.Select(i => (double)a[i]).ToArray(), y = ids.Select(i => (double)b[i]).ToArray();
        double mean = ids.Average(), scale = Math.Max(1, ids[^1] - ids[0]);
        double[] t = ids.Select(i => (i - mean) / scale).ToArray();
        // Orthogonalize [1,t,t²] on the actual support (which may contain gaps).
        var basis = new[] { Enumerable.Repeat(1.0, ids.Length).ToArray(), t, t.Select(v => v * v).ToArray() };
        foreach (var v in basis)
        {
            double norm = v.Sum(z => z * z); if (norm < 1e-20) continue;
            double ax = x.Select((z, i) => z * v[i]).Sum() / norm, ay = y.Select((z, i) => z * v[i]).Sum() / norm;
            for (int i = 0; i < x.Length; i++) { x[i] -= ax * v[i]; y[i] -= ay * v[i]; }
            foreach (var w in basis.Skip(Array.IndexOf(basis, v) + 1))
            { double projection = w.Select((z, i) => z * v[i]).Sum() / norm; for (int i = 0; i < w.Length; i++) w[i] -= projection * v[i]; }
        }
        double xx = x.Sum(v => v * v), yy = y.Sum(v => v * v);
        return xx > 1e-20 && yy > 1e-20 ? Math.Clamp(x.Select((v, i) => v * y[i]).Sum() / Math.Sqrt(xx * yy), -1, 1) : double.NaN;
    }
}
