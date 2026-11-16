using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class TiltSeries
{
    // One target search owns the calibration and the observation caches. Decoys may
    // choose orientations/poses, but cannot estimate their own observation model.
    private sealed class MatchComparison : IDisposable
    {
        public ParticlePeak[] Anchors;
        public Image CoarseVolume, BackgroundRootPower;
        public float2 CoarseImageDimensionsPhysical;
        public float3 CoarseSizeRoundingFactors;
        public readonly Dictionary<(float Pixel, int Box), float[][]> NoiseWeights = new();
        public (double Lower, double Upper)? AmplitudeBounds;
        public readonly Dictionary<int, MatchSolution> TargetSolutions = new();
        public int FinalBox;
        public float[][] FinalNoiseWeights;
        public double[] DetectorVariance, TiltScales;
        public double? DoseSlope;

        public void Dispose()
        {
            CoarseVolume?.Dispose();
            BackgroundRootPower?.Dispose();
        }
    }

    private ParticlePeak[] CompareMatchDecoys(ProcessingOptionsTomoFullMatch options, Image target,
        ParticlePeak[] targets, MatchComparison comparison, Func<float, string, bool> progress)
    {
        string targetName = options.TemplateName, targetSuffix = options.OverrideSuffix;
        bool saveCorr = options.DontSaveCorrVolume, saveAngles = options.DontSaveAngleIDVolume;
        bool fitB = options.RefineFitBfactor, exportSpectra = options.RefineExportTiltSpectra;
        var decoys = new Dictionary<int, ParticlePeak>[options.DecoyTemplatePaths.Length];
        comparison.DetectorVariance = matchDetectorVariance.ToArray();
        comparison.TiltScales = matchTiltScales?.ToArray();
        comparison.DoseSlope = matchDoseSlope;
        try
        {
            for (int d = 0; d < decoys.Length; d++)
            {
                MatchProgress(progress, 0, $"Paired decoy {d + 1}/{decoys.Length}: {options.DecoyTemplatePaths[d]}");
                using Image original = Image.FromFile(options.DecoyTemplatePaths[d]);
                if (original.Dims != target.Dims)
                    throw new InvalidDataException("Paired target and decoy templates must have identical dimensions and sampling.");
                using Image flipped = options.DecoyFlipX ? original.AsFlippedX() : null;
                string tag = $"__decoy_{d + 1:D3}_{System.IO.Path.GetFileNameWithoutExtension(options.DecoyTemplatePaths[d])}";
                options.TemplateName = targetName + tag;
                options.OverrideSuffix = string.IsNullOrWhiteSpace(targetSuffix) ? "" : targetSuffix + tag;
                options.DontSaveCorrVolume = options.DontSaveAngleIDVolume = true;
                options.RefineFitBfactor = options.RefineExportTiltSpectra = false;
                matchDetectorVariance = comparison.DetectorVariance;
                matchTiltScales = comparison.TiltScales;
                matchDoseSlope = comparison.DoseSlope;
                decoys[d] = MatchLargeVolumeCore(options, flipped ?? original, progress, comparison, true)
                    .ToDictionary(p => p.AnchorIndex);
            }
        }
        finally
        {
            options.TemplateName = targetName;
            options.OverrideSuffix = targetSuffix;
            options.DontSaveCorrVolume = saveCorr;
            options.DontSaveAngleIDVolume = saveAngles;
            options.RefineFitBfactor = fitB;
            options.RefineExportTiltSpectra = exportSpectra;
            matchDetectorVariance = comparison.DetectorVariance;
            matchTiltScales = comparison.TiltScales;
            matchDoseSlope = comparison.DoseSlope;
        }

        string name = ToTomogramWithPixelSize(Path, options.BinnedPixelSizeMean);
        string suffix = string.IsNullOrWhiteSpace(targetSuffix) ? "_" + targetName : targetSuffix;
        using StreamWriter diagnostics = new(System.IO.Path.Combine(MatchingDir, name + suffix + "_paired_decoys.tsv"));
        diagnostics.WriteLine("# Target-selected coarse anchors; each decoy searches its own orientations, with shared noise weights and imaging parameters.");
        diagnostics.WriteLine("# Rows are valid target fits before final specificity-ranked spatial suppression; missing decoy poses are excluded from the final STAR.");
        diagnostics.WriteLine(options.RefineBoundedScore
            ? "# Bounded likelihood gains evaluated at GPU profile-correlation-refined poses. Specificity is not a calibrated Z, p-value or FDR."
            : "# Projection scores evaluated at independently refined poses. Specificity is not a calibrated Z, p-value or FDR.");
        if (comparison.AmplitudeBounds is { } bounds)
            diagnostics.WriteLine(FormattableString.Invariant($"# amplitude_lower={bounds.Lower:R}; amplitude_upper={bounds.Upper:R}"));
        for (int d = 0; d < decoys.Length; d++)
            diagnostics.WriteLine($"# decoy_{d + 1}={options.DecoyTemplatePaths[d]}");
        diagnostics.WriteLine("anchor\tx_A\ty_A\tz_A\ttarget_score\tdecoy_mean\tspecificity\tstatus\t" +
            string.Join("\t", Enumerable.Range(1, decoys.Length).Select(d => $"decoy_{d}").ToArray()));

        List<ParticlePeak> paired = new();
        int unavailable = 0;
        foreach (ParticlePeak targetPeak in targets.OrderBy(p => p.AnchorIndex))
        {
            ParticlePeak peak = targetPeak;
            peak.TargetScore = peak.Score;
            peak.DecoyScores = decoys.Select(search => search.TryGetValue(peak.AnchorIndex, out var value)
                ? value.Score : float.NaN).ToArray();
            bool valid = peak.DecoyScores.All(float.IsFinite);
            if (valid)
            {
                peak.DecoyMeanScore = (float)peak.DecoyScores.Average(v => (double)v);
                peak.Score = (float)TemplateMatchSpecificity.Score(peak.TargetScore, peak.DecoyScores);
                paired.Add(peak);
            }
            else unavailable++;
            diagnostics.WriteLine(string.Join("\t", new[]
            {
                peak.AnchorIndex.ToString(CultureInfo.InvariantCulture),
                peak.PositionF.X.ToString("R", CultureInfo.InvariantCulture),
                peak.PositionF.Y.ToString("R", CultureInfo.InvariantCulture),
                peak.PositionF.Z.ToString("R", CultureInfo.InvariantCulture),
                peak.TargetScore.ToString("R", CultureInfo.InvariantCulture),
                valid ? peak.DecoyMeanScore.ToString("R", CultureInfo.InvariantCulture) : "NaN",
                valid ? peak.Score.ToString("R", CultureInfo.InvariantCulture) : "NaN",
                valid ? "paired" : "missing_decoy_pose"
            }.Concat(peak.DecoyScores.Select(v => v.ToString("R", CultureInfo.InvariantCulture))).ToArray()));
        }
        // A failed local decoy fit must never be replaced by zero evidence.
        if (targets.Length > 0 && paired.Count == 0)
            throw new InvalidOperationException("No target anchors have valid scores for every decoy. See paired-decoy diagnostics.");
        List<ParticlePeak> kept = SuppressMatchPeaks(paired, (float)options.PeakDistance);
        if (fitB)
            WriteFinalMatchEnvelopes(options, target, comparison.Anchors,
                kept.Select(p => (p.AnchorIndex, comparison.TargetSolutions[p.AnchorIndex])).ToList(),
                comparison.FinalNoiseWeights, comparison.FinalBox, progress);
        MatchProgress(progress, 1, $"Paired specificity retained {kept.Count}/{targets.Length} target anchors after spatial suppression; {unavailable} excluded for missing decoy poses");
        return kept.ToArray();
    }

    private static List<ParticlePeak> SuppressMatchPeaks(IEnumerable<ParticlePeak> peaks, float distance)
    {
        List<ParticlePeak> kept = new();
        float distanceSquared = distance * distance;
        foreach (ParticlePeak peak in peaks.OrderByDescending(p => p.Score).ThenBy(p => p.AnchorIndex))
            if (kept.All(other => (other.PositionF - peak.PositionF).LengthSq() >= distanceSquared))
                kept.Add(peak);
        return kept;
    }
}
