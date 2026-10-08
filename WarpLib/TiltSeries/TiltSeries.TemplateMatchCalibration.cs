using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ZLinq;
using Warp.Tools;

namespace Warp;

public partial class TiltSeries
{
    private void CalibrateMatchSeries(ProcessingOptionsTomoFullMatch options, ParticlePeak[] peaks,
        List<MatchSolution>[] solutions, Image[] tilts, Image[] masks, float[][] weights, Projector projector,
        Image coords, int box, float pixel, float maxShift, string prefix, Func<float, string, bool> progress)
    {
        List<(int Peak, MatchSolution Solution)> selected = new();
        foreach (var item in solutions.Select((group, p) => (Peak: p, Solution: group.Where(s => s.MergedIntoStart == -1 && s.Z > 0)
                    .OrderByDescending(s => s.Z).FirstOrDefault())).Where(p => p.Solution != null).OrderByDescending(p => p.Solution.Z))
        {
            if (selected.Any(p => (p.Solution.Position - item.Solution.Position).LengthSq() < (float)(options.PeakDistance * options.PeakDistance))) continue;
            selected.Add(item);
            if (selected.Count == 300) break;
        }
        if (selected.Count < 8) return;
        MatchProgress(progress, 0, $"Calibrating shared dose damage and tilt scales from {selected.Count} strong candidates...");
        int iterations = options.RefineIterations;
        bool export = options.RefineExportTiltSpectra;
        decimal highpass = options.RefineFitHighpass;
        double[] baseB = Enumerable.Range(0, NTilts).Select(t => (double)GetCTFParamsForOneTilt(pixel, [0],
            [VolumeDimensionsPhysical / 2], t, weighted: true)[0].Bfactor).ToArray();
        double[] slopes = [0, 1, 2, 3, 4, 5, 6, 8, 10];
        double[] scores = new double[slopes.Length];
        double[][] crosses = slopes.Select(_ => new double[NTilts]).ToArray();
        double[][] powers = slopes.Select(_ => new double[NTilts]).ToArray();
        int count = 0;
        // Every candidate uses the same frequency grid and dose envelopes. Compute
        // their exponentials once instead of repeating them for all 300 candidates.
        var envelopeFactors = new Dictionary<(float MaximumQ2, int Bins, double DeltaB), double[]>();
        try
        {
            options.RefineIterations = 0; options.RefineExportTiltSpectra = true; options.RefineFitHighpass = 0;
            // Fold each particle into the slope objectives and pooled tilt statistics immediately:
            // retaining all 300 x 41 x 8192-bin spectra would otherwise cost about 800 MiB.
            foreach (var item in selected)
            {
                var evaluated = RefineMatchBatch(options, [peaks[item.Peak].PositionF], [[item.Solution]], tilts, masks, weights,
                    projector, coords, box, pixel, maxShift, false, out _, true)[0];
                var best = evaluated.FirstOrDefault(s => s.MergedIntoStart == -1 && s.TiltEnvelopeSpectra != null);
                if (best == null) continue;
                count++;
                for (int k = 0; k < slopes.Length; k++)
                {
                    double cross = 0, power = 0;
                    for (int t = 0; t < NTilts; t++)
                    {
                        var spectrum = best.TiltEnvelopeSpectra[t];
                        double deltaB = -slopes[k] * Dose[t] - baseB[t];
                        var key = (spectrum.MaximumFrequencySquared, spectrum.Cross.Length, deltaB);
                        if (!envelopeFactors.TryGetValue(key, out var factors))
                        {
                            factors = TemplateMatchStatistics.EnvelopeFactors(key.MaximumFrequencySquared, key.Length, deltaB);
                            envelopeFactors.Add(key, factors);
                        }
                        var terms = TemplateMatchStatistics.Reweight(spectrum, factors);
                        cross += terms.Cross; power += terms.Power;
                        crosses[k][t] += terms.Cross; powers[k][t] += terms.Power;
                    }
                    if (power > 0) scores[k] += cross / Math.Sqrt(power);
                }
            }
        }
        finally { options.RefineIterations = iterations; options.RefineExportTiltSpectra = export; options.RefineFitHighpass = highpass; }
        if (count < 8) return;
        int bestIndex = 0;
        using StreamWriter report = new(prefix + "_calibration.tsv");
        report.WriteLine("dose_B_slope\tmean_projection_score");
        for (int k = 0; k < slopes.Length; k++)
        {
            report.WriteLine(FormattableString.Invariant($"{slopes[k]:R}\t{scores[k] / count:R}"));
            if (scores[k] > scores[bestIndex]) bestIndex = k;
        }
        double bestSlope = slopes[bestIndex];
        matchDoseSlope = bestSlope;
        double globalAmplitude = crosses[bestIndex].Sum() / powers[bestIndex].Sum();
        matchTiltScales = Enumerable.Repeat(1.0, NTilts).ToArray();
        report.WriteLine("tilt\tscale_correction\tselected_dose_B_slope\tcalibration_particles");
        for (int t = 0; t < NTilts; t++)
        {
            if (powers[bestIndex][t] > 0 && globalAmplitude > 0)
                matchTiltScales[t] = Math.Max(0, crosses[bestIndex][t] / powers[bestIndex][t] / globalAmplitude);
            report.WriteLine(FormattableString.Invariant($"{t}\t{matchTiltScales[t]:R}\t{bestSlope:R}\t{count}"));
        }
        MatchProgress(progress, 1, $"Selected dose B slope {bestSlope:F1} A²/(e/A²); continuing with calibrated tilt scales");
    }
}
