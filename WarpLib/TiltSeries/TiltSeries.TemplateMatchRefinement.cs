using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class TiltSeries
{
    private static void MaskMatchTemplate(Image template, float diameterAngstrom, float pixelSize, bool coarse)
    {
        // Image.MaskSpherically takes a diameter, and starts its outward cosine taper at D/2.
        template.MaskSpherically(diameterAngstrom / pixelSize, Math.Max(coarse ? 5 : 3, 20 / pixelSize), true);
    }

    private static void SubtractMatchTemplateBackground(Image template, float diameter, float pixel)
    {
        // Potential maps can carry a nonzero solvent baseline. Masking that baseline creates a false sphere.
        var background = new List<double>();
        float[][] data = template.GetHost(Intent.ReadWrite);
        float radius2 = MathF.Pow(.6f * diameter / pixel, 2);
        int stride = Math.Max(1, (int)(template.ElementsReal / 8192));
        long index = 0;
        for(int z=0;z<template.Dims.Z;z++) for(int y=0;y<template.Dims.Y;y++) for(int x=0;x<template.Dims.X;x++,index++)
        {
            if(index % stride != 0) continue;
            float dx=x-template.Dims.X/2,dy=y-template.Dims.Y/2,dz=z-template.Dims.Z/2;
            if(dx*dx+dy*dy+dz*dz > radius2) background.Add(data[z][y*template.Dims.X+x]);
        }
        if(background.Count == 0) return;
        float level=(float)TemplateMatchStatistics.Quantile(background,.5);
        foreach(float[] slice in data) for(int i=0;i<slice.Length;i++) slice[i]-=level;
    }

    private delegate int MatchBatchRefiner(ulong textureRe, ulong textureIm,
        int dim, int box, int views, int particles, int hypotheses,
        IntPtr data, IntPtr ctf, IntPtr quadrature, IntPtr inverseNoise, IntPtr phaseRadii,
        float[] geometry, float[] bounds, float[] symmetry, int symmetryCount,
        float[] poses, int[] seedIds, float pixel, float cutoff, float diameter,
        int maxIterations, float mergeDistance, float mergeAngle,
        double[] summary, int[] diagnostics, double[] tiltStatistics);

    private sealed class MatchGeometry
    {
        public float3 ImagePosition; // X/Y Angstrom, Z defocus in micrometers.
        public float3[] PositionDerivatives = new float3[3]; // per Angstrom in specimen XYZ
        public Matrix3 Rotation;
        public Matrix3[] RotationDerivatives = new Matrix3[3];
    }

    private static readonly Matrix3 MatchGeneratorX = new(0, 0, 0, 0, 0, 1, 0, -1, 0);
    private static readonly Matrix3 MatchGeneratorY = new(0, 0, -1, 0, 0, 0, 1, 0, 0);
    private static readonly Matrix3 MatchGeneratorZ = new(0, 1, 0, -1, 0, 0, 0, 0, 0);
    private static float MatchComponent(float3 v, int axis) => axis == 0 ? v.X : axis == 1 ? v.Y : v.Z;
    private static Matrix3 MatchTimes(Matrix3 m, float scale) => new(
        m.M11 * scale, m.M21 * scale, m.M31 * scale,
        m.M12 * scale, m.M22 * scale, m.M32 * scale,
        m.M13 * scale, m.M23 * scale, m.M33 * scale);

    // Matrix3.ToArray historically has a third-column typo; pack all nine fields explicitly.
    private static void PackMatchMatrix(Matrix3 m, float[] destination, int offset, float scale = 1)
    {
        destination[offset] = m.M11 * scale;
        destination[offset + 1] = m.M21 * scale;
        destination[offset + 2] = m.M31 * scale;
        destination[offset + 3] = m.M12 * scale;
        destination[offset + 4] = m.M22 * scale;
        destination[offset + 5] = m.M32 * scale;
        destination[offset + 6] = m.M13 * scale;
        destination[offset + 7] = m.M23 * scale;
        destination[offset + 8] = m.M33 * scale;
    }

    /// <summary>Warp's position/defocus/local-angle geometry with analytic chain-rule derivatives.
    /// The image rotation follows GetAnglesInOneTilt. The position follows GetPositionsInOneTilt,
    /// including 4D specimen warps, stage movement, defocus hand and image-size rounding.</summary>
    private MatchGeometry GetTemplateMatchGeometry(float3 position, int tilt)
    {
        float time = NTilts > 1 ? (float)tilt / (NTilts - 1) : 0;
        float doseTime = MaxDose > MinDose ? (Dose[tilt] - MinDose) / (MaxDose - MinDose) : 0;
        float3 normalized = position / VolumeDimensionsPhysical;
        float3 gridPosition = new(normalized.X, normalized.Y, time);
        float4 warpPosition = new(normalized.X, normalized.Y, normalized.Z, doseTime);
        float wx = GridVolumeWarpX.GetInterpolatedWithGradient(warpPosition, out float3 dwx);
        float wy = GridVolumeWarpY.GetInterpolatedWithGradient(warpPosition, out float3 dwy);
        float wz = GridVolumeWarpZ.GetInterpolatedWithGradient(warpPosition, out float3 dwz);
        dwx /= VolumeDimensionsPhysical;
        dwy /= VolumeDimensionsPhysical;
        dwz /= VolumeDimensionsPhysical;
        Matrix3 warpJacobian = new(1 + dwx.X, dwy.X, dwz.X,
                                      dwx.Y, 1 + dwy.Y, dwz.Y,
                                      dwx.Z, dwy.Z, 1 + dwz.Z);
        float3 centered = position - VolumeDimensionsPhysical / 2 + new float3(wx, wy, wz);
        Matrix3 tiltRotation = Matrix3.Euler(0, (Angles[tilt] + LevelAngleY) * Helper.ToRad,
            -TiltAxisAngles[tilt] * Helper.ToRad) * Matrix3.RotateX(LevelAngleX * Helper.ToRad);
        float3 projected = tiltRotation * centered;
        projected.X += TiltAxisOffsetX[tilt] + ImageDimensionsPhysical.X / 2;
        projected.Y += TiltAxisOffsetY[tilt] + ImageDimensionsPhysical.Y / 2;
        Matrix3 projectedJacobian = tiltRotation * warpJacobian;
        float3 movementAt = new(projected.X / ImageDimensionsPhysical.X, projected.Y / ImageDimensionsPhysical.Y, time);
        float mx = GridMovementX.GetInterpolatedWithGradient(movementAt, out float3 dmx);
        float my = GridMovementY.GetInterpolatedWithGradient(movementAt, out float3 dmy);
        float defocus = GridCTFDefocus.GetInterpolatedWithGradient(gridPosition, out float3 ddefocus);
        float3 spatialDefocus = new(ddefocus.X / VolumeDimensionsPhysical.X, ddefocus.Y / VolumeDimensionsPhysical.Y, 0);

        Matrix3 depthRotation = AreAnglesInverted
            ? Matrix3.Euler(0, -(Angles[tilt] + LevelAngleY) * Helper.ToRad, -TiltAxisAngles[tilt] * Helper.ToRad)
                * Matrix3.RotateX(-LevelAngleX * Helper.ToRad) * Matrix3.Scale(1, 1, -1)
            : tiltRotation;
        Matrix3 depthJacobian = depthRotation * warpJacobian;
        MatchGeometry result = new()
        {
            ImagePosition = new float3(projected.X - mx, projected.Y - my,
                defocus + 1e-4f * (depthRotation * centered).Z) * SizeRoundingFactors
        };
        float3[] columns = [projectedJacobian.C1, projectedJacobian.C2, projectedJacobian.C3];
        float3[] depthColumns = [depthJacobian.C1, depthJacobian.C2, depthJacobian.C3];
        for (int axis = 0; axis < 3; axis++)
        {
            float3 d = columns[axis];
            result.PositionDerivatives[axis] = new float3(
                d.X - dmx.X * d.X / ImageDimensionsPhysical.X - dmx.Y * d.Y / ImageDimensionsPhysical.Y,
                d.Y - dmy.X * d.X / ImageDimensionsPhysical.X - dmy.Y * d.Y / ImageDimensionsPhysical.Y,
                MatchComponent(spatialDefocus, axis) + 1e-4f * depthColumns[axis].Z) * SizeRoundingFactors;
        }

        float ax = GridAngleX.GetInterpolatedWithGradient(gridPosition, out float3 dax) * Helper.ToRad;
        float ay = GridAngleY.GetInterpolatedWithGradient(gridPosition, out float3 day) * Helper.ToRad;
        float az = GridAngleZ.GetInterpolatedWithGradient(gridPosition, out float3 daz) * Helper.ToRad;
        dax = new float3(dax.X / VolumeDimensionsPhysical.X, dax.Y / VolumeDimensionsPhysical.Y, 0) * Helper.ToRad;
        day = new float3(day.X / VolumeDimensionsPhysical.X, day.Y / VolumeDimensionsPhysical.Y, 0) * Helper.ToRad;
        daz = new float3(daz.X / VolumeDimensionsPhysical.X, daz.Y / VolumeDimensionsPhysical.Y, 0) * Helper.ToRad;
        Matrix3 rx = Matrix3.RotateX(ax), ry = Matrix3.RotateY(ay), rz = Matrix3.RotateZ(az);
        result.Rotation = rz * ry * rx * tiltRotation;
        Matrix3 gx = rz * ry * rx * MatchGeneratorX * tiltRotation;
        Matrix3 gy = rz * ry * MatchGeneratorY * rx * tiltRotation;
        Matrix3 gz = rz * MatchGeneratorZ * ry * rx * tiltRotation;
        for (int axis = 0; axis < 3; axis++)
            result.RotationDerivatives[axis] = MatchTimes(gx, MatchComponent(dax, axis))
                + MatchTimes(gy, MatchComponent(day, axis)) + MatchTimes(gz, MatchComponent(daz, axis));
        return result;
    }

    private double matchPreparationSeconds, matchOptimizationSeconds, matchResultSeconds;

    private sealed class MatchSolution
    {
        public float3 Position;
        public Matrix3 Rotation;
        public double Z, Gain, Amplitude;
        public int StartIndex, Iterations, Evaluations, MergedIntoStart = -1;
        public double InitialZ;
        public bool Converged;
        public TemplateMatchTerminationReason TerminationReason;
        public double[] TiltStatistics;
        public TemplateMatchEnvelopeSpectrum EnvelopeSpectrum;
        public TemplateMatchEnvelopeSpectrum[] TiltEnvelopeSpectra;
    }

    private ParticlePeak[] RefineTemplateMatches(ProcessingOptionsTomoFullMatch options, Image template,
        ParticlePeak[] peaks, TemplateMatchStart[][] starts, Func<float, string, bool> progress)
    {
        decimal originalBinTimes = options.BinTimes;
        decimal originalLowpass = options.Lowpass;
        if (options.RefineExportTiltSpectra && !options.RefineFitBfactor)
            throw new ArgumentException("Per-tilt spectrum export requires amplitude/B fitting.");
        decimal coarseSampling = options.BinnedPixelSizeMean;
        float coarsePixel = (float)coarseSampling;
        float maxShift = options.RefineMaxShift > 0 ? (float)options.RefineMaxShift : 3 * coarsePixel;
        decimal finalPixel = options.OptimizePosesAngPix ?? options.BinnedPixelSizeMean;
        if (finalPixel <= 0 || finalPixel > options.BinnedPixelSizeMean || originalLowpass <= 0 || originalLowpass > 1
            || options.RefineMergeFraction < 0 || options.RefineMergeFraction > 0.5M)
            throw new ArgumentException("Invalid refinement resolution, merge threshold or low-pass cutoff.");
        float[] resolutions = TemplateMatchStatistics.ResolutionSchedule(
            2 * coarsePixel / (float)originalLowpass, 2 * (float)finalPixel / (float)originalLowpass);
        // Even a single requested band needs a pass after estimating the shared envelope.
        if (resolutions.Length == 1) resolutions = [resolutions[0], resolutions[0]];
        int stages = resolutions.Length;
        if (options.RefineFitHighpass < 0 || (options.RefineFitBfactor && options.RefineFitHighpass > 0 &&
            options.RefineFitHighpass <= 2 * finalPixel / Math.Min(1M, options.Lowpass)))
            throw new ArgumentException("The amplitude/B high-pass must be nonnegative and coarser than the final refinement resolution.");
        List<MatchSolution>[] solutions = starts.Select(group => group.Select((start, index) => new MatchSolution
        {
            Position = start.Position, Rotation = Matrix3.Euler(start.Angles), StartIndex = index
        }).ToList()).ToArray();
        string name = ToTomogramWithPixelSize(Path, options.BinnedPixelSizeMean);
        string suffix = string.IsNullOrWhiteSpace(options.OverrideSuffix) ? "_" + options.TemplateName : options.OverrideSuffix;
        string diagnosticsPath = System.IO.Path.Combine(MatchingDir, name + suffix + "_refinement.tsv");
        // Preserve sparse proposals and anchors so score/optimizer comparisons do not depend
        // on reconstructing peak selection from large, rounded diagnostic MRC volumes.
        using (StreamWriter proposals = new(System.IO.Path.Combine(MatchingDir, name + suffix + "_starting_poses.tsv")))
        {
            proposals.WriteLine("peak\tstart\tanchor_x_A\tanchor_y_A\tanchor_z_A\tx_A\ty_A\tz_A\trot_deg\ttilt_deg\tpsi_deg\tangle_id\trank\tproposal_score");
            for (int p = 0; p < starts.Length; p++)
                for (int h = 0; h < starts[p].Length; h++)
                {
                    TemplateMatchStart start = starts[p][h];
                    float3 anchor = peaks[p].PositionF, angles = start.Angles * Helper.ToDeg;
                    proposals.WriteLine(FormattableString.Invariant(
                        $"{p}\t{h}\t{anchor.X:R}\t{anchor.Y:R}\t{anchor.Z:R}\t{start.Position.X:R}\t{start.Position.Y:R}\t{start.Position.Z:R}\t{angles.X:R}\t{angles.Y:R}\t{angles.Z:R}\t{start.AngleId}\t{start.Rank}\t{start.ProposalScore:R}"));
                }
        }
        using StreamWriter diagnostics = new(diagnosticsPath);
        diagnostics.WriteLine("stage\tpeak\tstart\tx_A\ty_A\tz_A\trot_deg\ttilt_deg\tpsi_deg\tprojection_z\tprofile_gain\tamplitude\titerations\tconverged\tusable_tilts\ttermination_reason\tevaluations\tinitial_z\tmerged_into_start");
        try
        {
            for (int stage = 0; stage < stages; stage++)
            {
                matchPreparationSeconds = matchOptimizationSeconds = matchResultSeconds = 0;
                var stageTimer = System.Diagnostics.Stopwatch.StartNew();
                int stageBudget = TemplateMatchStatistics.ContinuationHypotheses(starts.Max(g => g.Length), resolutions[0], resolutions[stage]);
                if (stage > 0)
                    for (int p = 0; p < solutions.Length; p++)
                        solutions[p] = solutions[p].OrderByDescending(s => s.Z).ThenBy(s => s.StartIndex).Take(stageBudget).ToList();
                MatchProgress(progress, 0, $"Refinement stage {stage+1}/{stages}: {solutions.Count(g => g.Count > 0)}/{peaks.Length} spatial peaks, {solutions.Sum(g => g.Count)} pose hypotheses total (up to {stageBudget} per peak)");
                decimal stagePixel = Math.Min(coarseSampling, (decimal)resolutions[stage] * originalLowpass / 2);
                if (stage == stages - 1) stagePixel = finalPixel;
                options.BinTimes = (decimal)Math.Log2((double)(stagePixel / options.PixelSizeMean));
                float pixel = (float)options.BinnedPixelSizeMean;
                options.Lowpass = (decimal)(2 * pixel / resolutions[stage]);
                MatchProgress(progress, 0, $"Preparing batched GPU BFGS pose refinement at {pixel:F3} A/px...");
                LoadMovieData(options, out _, out Image[] tilts, false, out _, out _);
                LoadMovieMasks(options, out Image[] masks);
                // Both loaders return shared caches. Borrow these buffers; do not Dispose them.
                try
                {
                    if (!options.DontInvert)
                        foreach (Image tilt in tilts) tilt.Multiply(-1);
                    CTF worstCTF = CTF.GetCopy();
                    worstCTF.PixelSize = (decimal)pixel;
                    worstCTF.Defocus = (decimal)(GridCTFDefocus.Values.Max(v => Math.Abs(v))
                        + 1e-4f * (VolumeDimensionsPhysical.Length() / 2 + maxShift * 2));
                    int support = worstCTF.GetAliasingFreeSize(2 * pixel / (float)options.Lowpass, (float)options.TemplateDiameter / pixel);
                    int requiredBox = (int)Math.Ceiling(Math.Max(2 * (float)options.TemplateDiameter / pixel,
                        support + 2 * maxShift * 2 / pixel + 8));
                    int box = 2 * MathHelper.NextFFTFriendlySize((requiredBox + 1) / 2);
                    if (tilts.Any(t => t.Dims.X < box || t.Dims.Y < box))
                        throw new InvalidOperationException($"The CTF-padded refinement box ({box}) exceeds a tilt image. Use a coarser refinement pixel size.");
                    MatchProgress(progress, 0, $"Estimating fixed background spectra ({options.RefineNoisePatches} patches/tilt, box {box})...");
                    float[][] noiseWeights = EstimateMatchNoise(tilts, masks, box, options.RefineNoisePatches);
                    int scaledSize = Math.Max(2, (int)Math.Round(template.Dims.X * (float)options.TemplatePixel / pixel / 2) * 2);
                    using Image scaled = template.AsScaled(new int3(scaledSize));
                    SubtractMatchTemplateBackground(scaled, (float)options.TemplateDiameter, pixel);
                    MaskMatchTemplate(scaled, (float)options.TemplateDiameter, pixel, false);
                    using Image padded = scaled.AsPadded(new int3(box));
                    using Projector projector = new(padded, 2, true);
                    using Image ctfCoordinates = CTF.GetCTFCoords(box, box, MagnificationCorrection);
                    // Keep observations shared by all hypotheses. Bound transient FFT/data storage to
                    // a quarter of free device memory (at most 2 GiB), including FFT scratch headroom.
                    long bytesPerParticle = (long)NTilts * (box * (box / 2 + 1) * 32L + box * box * 8L);
                    if (options.RefineExportTiltSpectra && stage == stages - 1)
                        bytesPerParticle += (long)NTilts * 2 * TemplateMatchEnvelope.SpectrumBins * sizeof(float);
                    long budget = Math.Min(2L << 30, Math.Max(1, GPU.GetFreeMemory(GPU.GetDevice())) * (1L << 18));
                    int batchSize = (int)Math.Clamp(budget / bytesPerParticle, 1, 64);
                    for (int first = 0; first < peaks.Length; first += batchSize)
                    {
                        int count = Math.Min(batchSize, peaks.Length - first);
                        MatchProgress(progress, (float)first / peaks.Length,
                            $"Refining peaks {first + 1}–{first + count}/{peaks.Length} on GPU, stage {stage + 1}/{stages}");
                        var refined = RefineMatchBatch(options,
                            peaks.Skip(first).Take(count).Select(p => p.PositionF).ToArray(),
                            solutions.Skip(first).Take(count).ToArray(), tilts, masks, noiseWeights,
                            projector, ctfCoordinates, box, pixel, maxShift, true, out int[] usableTilts,
                            options.RefineFitBfactor && stage == stages - 1);
                        for (int local = 0; local < count; local++)
                        {
                            int p = first + local;
                            foreach (MatchSolution solution in refined[local])
                            {
                                float3 angles = Matrix3.EulerFromMatrix(solution.Rotation) * Helper.ToDeg;
                                diagnostics.WriteLine(FormattableString.Invariant(
                                    $"{stage}\t{p}\t{solution.StartIndex}\t{solution.Position.X:R}\t{solution.Position.Y:R}\t{solution.Position.Z:R}\t{angles.X:R}\t{angles.Y:R}\t{angles.Z:R}\t{solution.Z:R}\t{solution.Gain:R}\t{solution.Amplitude:R}\t{solution.Iterations}\t{solution.Converged}\t{usableTilts[local]}\t{solution.TerminationReason}\t{solution.Evaluations}\t{solution.InitialZ:R}\t{solution.MergedIntoStart}"));
                            }
                            solutions[p] = refined[local].Where(s => s.MergedIntoStart == -1).ToList();
                        }
                    }
                    diagnostics.Flush();
                    MatchProgress(progress, 1, $"Stage {stage+1}: {stageTimer.Elapsed.TotalSeconds:F2}s; prepare {matchPreparationSeconds:F2}s, optimize {matchOptimizationSeconds:F2}s, results {matchResultSeconds:F2}s; {solutions.Sum(g=>g.Count)} surviving hypotheses");
                    if (stage == Math.Max(0, stages - 2))
                        CalibrateMatchSeries(options, peaks, solutions, tilts, masks, noiseWeights, projector,
                            ctfCoordinates, box, pixel, maxShift, System.IO.Path.Combine(MatchingDir, name + suffix), progress);
                }
                finally
                {
                    foreach (Image tilt in tilts) tilt?.FreeDevice();
                    foreach (Image mask in masks) mask?.FreeDevice();
                }
            }
        }
        finally
        {
            options.BinTimes = originalBinTimes;
            options.Lowpass = originalLowpass;
        }
        List<(ParticlePeak Peak, int OriginalPeak, MatchSolution Solution)> output = new();
        string perTiltPath = System.IO.Path.Combine(MatchingDir, name + suffix + "_tilt_scores.tsv");
        using StreamWriter perTilt = new(perTiltPath);
        perTilt.WriteLine("peak\ttilt\tcross\tmodel_power");
        for (int p = 0; p < peaks.Length; p++)
        {
            if (solutions[p].Count == 0) continue;
            // Multiplicity of starts is never extra evidence. Final spatial NMS also
            // merges solutions from different original peaks that land on one particle.
            MatchSolution best = solutions[p].OrderByDescending(s => s.Z).First();
            if (!double.IsFinite(best.Gain) || best.Z <= 0) continue;
            ParticlePeak peak = peaks[p];
            peak.PositionF = best.Position;
            peak.Position = new int3((int)Math.Round(best.Position.X / coarsePixel),
                (int)Math.Round(best.Position.Y / coarsePixel), (int)Math.Round(best.Position.Z / coarsePixel));
            peak.Angles = Matrix3.EulerFromMatrix(best.Rotation) * Helper.ToDeg;
            peak.Score = (float)best.Gain;
            peak.ProjectionZ = (float)best.Z;
            peak.FittedAmplitude = (float)best.Amplitude;
            output.Add((peak, p, best));
            for (int t = 0; t < NTilts; t++)
                perTilt.WriteLine(FormattableString.Invariant($"{p}\t{t}\t{best.TiltStatistics[t * 14]:R}\t{best.TiltStatistics[t * 14 + 1]:R}"));
        }
        // Estimate the amplitude interval from distinct strong detections; no labels or expected particle count.
        List<MatchSolution> calibration = new();
        foreach (var item in output.OrderByDescending(p => p.Solution.Z))
        {
            if (calibration.Any(other => (other.Position-item.Solution.Position).LengthSq() < (float)(options.PeakDistance*options.PeakDistance))) continue;
            calibration.Add(item.Solution);
            if (calibration.Count == 300) break;
        }
        if (calibration.Count >= 8)
        {
            double lower = Math.Max(0, TemplateMatchStatistics.Quantile(calibration.Select(s=>s.Amplitude).ToArray(),.02));
            double upper = TemplateMatchStatistics.Quantile(calibration.Select(s=>s.Amplitude).ToArray(),.98);
            for (int i=0;i<output.Count;i++)
            {
                var item = output[i];
                double cross = 0, power = 0;
                for(int t=0;t<NTilts;t++) { cross += item.Solution.TiltStatistics[t*14]; power += item.Solution.TiltStatistics[t*14+1]; }
                item.Peak.Score = (float)TemplateMatchStatistics.BoundedGain(cross,power,lower,upper);
                output[i] = item;
            }
            File.WriteAllText(System.IO.Path.Combine(MatchingDir,name+suffix+"_amplitude_bounds.tsv"),
                FormattableString.Invariant($"lower\tupper\tcalibration_particles\n{lower:R}\t{upper:R}\t{calibration.Count}\n"));
        }
        List<ParticlePeak> kept = new();
        float separation2 = (float)(options.PeakDistance * options.PeakDistance);
        var envelopeRows = new List<(int OriginalPeak, MatchSolution Solution)>();
        foreach (var item in output.OrderByDescending(p => p.Peak.Score))
            if (kept.All(other => (other.PositionF - item.Peak.PositionF).LengthSq() >= separation2))
            {
                kept.Add(item.Peak);
                envelopeRows.Add((item.OriginalPeak, item.Solution));
            }
        if (options.RefineFitBfactor)
        {
            MatchProgress(progress, 1, $"Fitting joint amplitude/B envelopes for {kept.Count} final candidates...");
            WriteMatchEnvelopes(System.IO.Path.Combine(MatchingDir, name + suffix), envelopeRows);
            if (options.RefineExportTiltSpectra)
                WriteMatchTiltEnvelopeSpectra(System.IO.Path.Combine(MatchingDir, name + suffix), envelopeRows, (float)finalPixel);
        }
        MatchProgress(progress, 1, $"GPU refinement retained {kept.Count}/{peaks.Length} peaks after final spatial suppression");
        return kept.ToArray();
    }

    private static void MatchProgress(Func<float, string, bool> callback, float fraction, string message)
    {
        if (callback?.Invoke(fraction, message) == true) throw new OperationCanceledException();
    }

    private static bool MatchPatchIsUsable(Image image, Image mask, int x, int y, int box, bool allowPadding = false)
    {
        if (!allowPadding && (x < 0 || y < 0 || x + box > image.Dims.X || y + box > image.Dims.Y)) return false;
        int left = Math.Max(0, x), top = Math.Max(0, y);
        int right = Math.Min(image.Dims.X, x + box), bottom = Math.Min(image.Dims.Y, y + box);
        if (left >= right || top >= bottom) return false;
        if (mask == null) return true;
        float[] values = mask.GetHost(Intent.Read)[0];
        for (int row = top; row < bottom; row++)
            for (int col = left; col < right; col++)
                if (values[row * mask.Dims.X + col] > 0.5f) return false;
        return true;
    }

    private float[][] EstimateMatchNoise(Image[] tilts, Image[] masks, int box, int samples)
    {
        int elements = box * (box / 2 + 1);
        float[][] weights = Helper.ArrayOfFunction(_ => new float[elements], NTilts);
        for (int t = 0; t < NTilts; t++)
        {
            if (!UseTilt[t]) continue;
            List<int2> origins = new();
            HashSet<(int X, int Y)> used = new();
            for (int attempt = 0; origins.Count < samples && attempt < samples * 16; attempt++)
            {
                int x = (int)(((attempt + 0.5) * 0.6180339887498949 % 1) * (tilts[t].Dims.X - box + 1));
                int y = (int)(((attempt + 0.5) * 0.4142135623730951 % 1) * (tilts[t].Dims.Y - box + 1));
                if (MatchPatchIsUsable(tilts[t], masks[t], x, y, box) && used.Add((x,y))) origins.Add(new int2(x,y));
            }
            if (origins.Count < 2) continue;
            using Image accumulated = new(new int3(box,box,1), true);
            accumulated.Fill(0);
            const int noiseBatch = 32;
            for (int first = 0; first < origins.Count; first += noiseBatch)
            {
                int count = Math.Min(noiseBatch, origins.Count - first);
                using Image patches = new(new int3(box, box, count));
                int3[] patchOrigins = origins.Skip(first).Take(count).Select((o,i)=>new int3(o.X,o.Y,i)).ToArray();
                int extractionStatus = GPU.MatchExtractCentered(tilts[t].GetDevice(Intent.Read),new int2(tilts[t].Dims),patchOrigins,box,count,patches.GetDevice(Intent.Write));
                if (extractionStatus != 0) throw new InvalidOperationException($"Background extraction failed with CUDA status {extractionStatus}.");
                using Image ft = patches.AsFFT();
                float normalization = (float)(1.0 / ((double)box * box * box * box * origins.Count));
                GPU.MatchAccumulatePower(ft.GetDevice(Intent.Read), accumulated.GetDevice(Intent.ReadWrite), elements, count, normalization);
            }
            double[] power = accumulated.GetHost(Intent.Read)[0].Select(v => (double)v).ToArray();
            if (power.Any(v => !double.IsFinite(v))) throw new InvalidDataException("Nonfinite background patch.");
            double floor = Math.Max(1e-30, TemplateMatchStatistics.Quantile(power, .5) * 1e-4);
            for (int y = 0; y < box; y++)
                for (int x = 0; x <= box/2; x++)
                {
                    double smoothed = 0;
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int xx = x + dx, yy = y + dy;
                            if (xx < 0) { xx = -xx; yy = -yy; }
                            xx = Math.Min(xx, box/2); yy = (yy % box + box) % box;
                            smoothed += power[yy*(box/2+1)+xx] / 9;
                        }
                    weights[t][y*(box/2+1)+x] = x == 0 && y == 0 ? 0 : (float)(2/Math.Max(floor, smoothed));
                }
        }
        if (weights.Count(w => w.Any(v => v > 0)) < 3)
            throw new InvalidOperationException("Fewer than three tilts have usable unmasked background patches for refinement.");
        return weights;
    }

    // Small single-particle entry point used by the independent forward-model integration tests.
    // Production batches particles and performs symmetry-aware merging between stages.
    private List<MatchSolution> RefineMatchCandidate(ProcessingOptionsTomoFullMatch options, float3 anchor,
        List<MatchSolution> starts, Image[] tilts, Image[] masks, float[][] noiseWeights, Projector projector,
        Image ctfCoordinates, int box, float pixel, float maxShift, out int usableTilts)
    {
        var result = RefineMatchBatch(options, [anchor], [starts], tilts, masks, noiseWeights, projector,
            ctfCoordinates, box, pixel, maxShift, false, out int[] usable, options.RefineFitBfactor);
        usableTilts = usable[0];
        return result[0];
    }

    private List<MatchSolution>[] RefineMatchBatch(ProcessingOptionsTomoFullMatch options, float3[] anchors,
        List<MatchSolution>[] starts, Image[] tilts, Image[] masks, float[][] noiseWeights, Projector projector,
        Image ctfCoordinates, int box, float pixel, float maxShift, bool merge, out int[] usableTilts, bool fitEnvelope = false)
    {
        var batchTimer = System.Diagnostics.Stopwatch.StartNew();
        int particles = anchors.Length;
        int hypotheses = Math.Max(1, starts.Max(s => s.Count));
        int views = particles * NTilts;
        int elements = box * (box / 2 + 1);
        usableTilts = new int[particles];
        var result = Enumerable.Range(0, particles).Select(_ => new List<MatchSolution>()).ToArray();
        using Image patches = new(new int3(box, box, views));
        using Image inverseNoise = new(new int3(box, box, views), true);
        using Image radiusSquared = new(new int3(box, box, views), true);
        float[][] weights = inverseNoise.GetHost(Intent.Write);
        var patchOrigins = Enumerable.Range(0, NTilts).Select(_ => new List<int3>()).ToArray();
        float[][] phaseRadii = radiusSquared.GetHost(Intent.Write);
        float3[] centerShifts = new float3[views];
        CTFStruct[] ctfParams = new CTFStruct[views], quadParams = new CTFStruct[views];
        float[] weightRotations = new float[views * 9];
        float[] geometry = new float[views * 18], bounds = new float[particles * 6];
        float[] poses = new float[particles * hypotheses * 12];
        int[] seedIds = Enumerable.Repeat(-1, particles * hypotheses).ToArray();
        float[] coordinates = ctfCoordinates.GetHost(Intent.Read)[0];
        float voltage = (float)CTF.Voltage * 1000;
        float wavelength = (float)(12.2643247 / Math.Sqrt(voltage * (1 + voltage * 0.978466e-6)));
        float defocusPhaseScale = -MathF.PI * wavelength * 10000;
        Matrix3 magnification = new(MagnificationCorrection.M11, MagnificationCorrection.M21, 0,
            MagnificationCorrection.M12, MagnificationCorrection.M22, 0, 0, 0, 1);
        bool[] hasNoise = noiseWeights.Select(w => w.Any(v => v > 0)).ToArray();
        var radiusGrids = new Dictionary<(float Pixel, float Delta, float Angle), float[]>();
        for (int p = 0; p < particles; p++)
        {
            float3 anchor = anchors[p];
            for (int axis = 0; axis < 3; axis++)
            {
                bounds[p * 6 + axis] = -Math.Min(maxShift, MatchComponent(anchor, axis));
                bounds[p * 6 + axis + 3] = Math.Min(maxShift,
                    MatchComponent(VolumeDimensionsPhysical - anchor, axis));
            }
            for (int t = 0; t < NTilts; t++)
            {
                int view = p * NTilts + t;
                MatchGeometry local = GetTemplateMatchGeometry(anchor, t);
                // Freeze the local deformation and tilt rotation at the original coarse anchor.
                // Particle orientation remains an exact rotation: q = Rparticle^T * G * frequency.
                PackMatchMatrix(local.Rotation.Transposed() * magnification, geometry, view * 18, projector.Oversampling);
                for (int axis = 0; axis < 3; axis++)
                {
                    geometry[view * 18 + 9 + axis * 2] = local.PositionDerivatives[axis].X / pixel;
                    geometry[view * 18 + 10 + axis * 2] = local.PositionDerivatives[axis].Y / pixel;
                    geometry[view * 18 + 15 + axis] = defocusPhaseScale * local.PositionDerivatives[axis].Z;
                }
                Array.Clear(weights[view]);
                float3 center = local.ImagePosition / pixel;
                int x = (int)Math.Floor(center.X - box / 2f), y = (int)Math.Floor(center.Y - box / 2f);
                // Missing padding is not a missing particle. Keep the target core in view, but permit
                // CTF/motion padding outside the detector as in the research implementation.
                float coreMargin = (float)options.TemplateDiameter / (2 * pixel) + MathF.Sqrt(3) * maxShift / pixel;
                bool coreVisible = center.X >= coreMargin && center.Y >= coreMargin &&
                    center.X < tilts[t].Dims.X - coreMargin && center.Y < tilts[t].Dims.Y - coreMargin;
                if (UseTilt[t] && hasNoise[t] && coreVisible && MatchPatchIsUsable(tilts[t], masks[t], x, y, box, true))
                {
                    usableTilts[p]++;
                    patchOrigins[t].Add(new int3(x, y, view));
                    Array.Copy(noiseWeights[t], weights[view], elements);
                    PackMatchMatrix(local.Rotation.Transposed(), weightRotations, view * 9);
                    centerShifts[view] = new float3(-(center.X - box / 2f - x) + box / 2f,
                        -(center.Y - box / 2f - y) + box / 2f, 0);
                }
                CTF parameters = GetCTFParamsForOneTilt(pixel, [local.ImagePosition.Z], [anchor], t,
                    weighted: true, weightsonly: false)[0];
                if (matchDoseSlope.HasValue) parameters.Bfactor = (decimal)(-matchDoseSlope.Value * Dose[t]);
                if (matchTiltScales != null) parameters.Scale *= (decimal)matchTiltScales[t];
                if (!UseTilt[t]) parameters.Scale = 0;
                ctfParams[view] = parameters.ToStruct();
                parameters.PhaseShift -= 0.5M; // -sin(gamma) -> -cos(gamma)
                quadParams[view] = parameters.ToStruct();
                var radiusKey = (ctfParams[view].PixelSize, ctfParams[view].PixelSizeDelta, ctfParams[view].PixelSizeAngle);
                if (!radiusGrids.TryGetValue(radiusKey, out var grid))
                {
                    grid = phaseRadii[view];
                    for (int f = 0; f < elements; f++)
                    {
                        double effectivePixel = radiusKey.PixelSize * 1e10
                            + radiusKey.PixelSizeDelta * 0.5e10 * Math.Cos(2 * (coordinates[2 * f + 1] - radiusKey.PixelSizeAngle));
                        double frequency = coordinates[2 * f] / effectivePixel;
                        grid[f] = (float)(frequency * frequency);
                    }
                    radiusGrids.Add(radiusKey, grid);
                }
                else Array.Copy(grid, phaseRadii[view], elements);
            }
            if (usableTilts[p] < 3) continue;
            for (int h = 0; h < starts[p].Count; h++)
            {
                int slot = p * hypotheses + h;
                MatchSolution start = starts[p][h];
                float3 delta = start.Position - anchor;
                for (int axis = 0; axis < 3; axis++)
                    poses[slot * 12 + axis] = Math.Clamp(MatchComponent(delta, axis), bounds[p * 6 + axis], bounds[p * 6 + axis + 3]);
                PackMatchMatrix(start.Rotation, poses, slot * 12 + 3);
                seedIds[slot] = start.StartIndex;
            }
        }
        MatchSolution Rejected(MatchSolution start, TemplateMatchTerminationReason reason) => new()
        {
            Position = start.Position, Rotation = start.Rotation, StartIndex = start.StartIndex,
            Z = double.NaN, Gain = double.NaN, Amplitude = double.NaN, InitialZ = double.NaN,
            MergedIntoStart = -2, TerminationReason = reason
        };
        if (seedIds.All(id => id < 0))
        {
            for (int p = 0; p < particles; p++)
                result[p].AddRange(starts[p].Select(s => Rejected(s, TemplateMatchTerminationReason.InsufficientTilts)).ToArray());
            return result;
        }
        patches.Fill(0);
        for (int t = 0; t < NTilts; t++)
        {
            if (patchOrigins[t].Count == 0) continue;
            int extractionStatus = GPU.MatchExtractCentered(tilts[t].GetDevice(Intent.Read), new int2(tilts[t].Dims),
                patchOrigins[t].ToArray(), box, patchOrigins[t].Count, patches.GetDevice(Intent.ReadWrite));
            if (extractionStatus != 0) throw new InvalidDataException($"Candidate extraction failed at tilt {t} with CUDA status {extractionStatus} (check for nonfinite image data).");
        }
        using Image observed = patches.AsFFT();
        observed.Multiply(1f / (box * box));
        observed.ShiftSlices(centerShifts);
        patches.FreeDevice();
        using Image ctf = new(IntPtr.Zero, new int3(box, box, views), true);
        using Image quadrature = new(IntPtr.Zero, new int3(box, box, views), true);
        GPU.CreateCTF(ctf.GetDevice(Intent.Write), ctfCoordinates.GetDevice(Intent.Read), IntPtr.Zero,
            (uint)elements, ctfParams, false, (uint)views);
        GPU.CreateCTF(quadrature.GetDevice(Intent.Write), ctfCoordinates.GetDevice(Intent.Read), IntPtr.Zero,
            (uint)elements, quadParams, false, (uint)views);
        Matrix3[] symmetries = new Symmetry(options.Symmetry ?? "C1").GetRotationMatrices();
        float[] symmetry = new float[symmetries.Length * 9];
        for (int i = 0; i < symmetries.Length; i++) PackMatchMatrix(symmetries[i], symmetry, i * 9);
        float cutoff = box * 0.5f * Math.Min(1, (float)options.Lowpass);
        float diameter = options.TemplateDiameter > 0 ? (float)options.TemplateDiameter : box * pixel / 2;
        float bandPixel = box * pixel / (2 * cutoff);
        // A coarse-score winner need not win after the bandwidth increases. Merge only
        // near-identical poses; Beam's half-pixel criterion discarded useful fine-band basins.
        float mergeDistance = merge ? (float)options.RefineMergeFraction * bandPixel : 0;
        double[] summary = new double[particles * hypotheses * 4];
        int[] diagnostic = new int[particles * hypotheses * 4];
        double[] statistics = new double[particles * hypotheses * NTilts * 2];
        if (matchDetectorVariance != null)
        {
            float[] detector = matchDetectorVariance.Select(v => (float)(v * Math.Pow((double)options.PixelSizeMean / pixel, 2) / (box * box))).ToArray();
            GPU.MatchHybridWeights(inverseNoise.GetDevice(Intent.ReadWrite), box, particles, NTilts,
                weightRotations, detector, pixel, Math.Max(400, 3 * diameter));
        }
        GPU.CopyDeviceToHost(inverseNoise.GetDevice(Intent.Read), new float[1], 1);
        matchPreparationSeconds += batchTimer.Elapsed.TotalSeconds; batchTimer.Restart();
        MatchBatchRefiner refine = GPU.TemplateMatchRefineBatchBfgs;
        int status = refine(projector.t_DataRe, projector.t_DataIm, projector.Data.Dims.X,
            box, NTilts, particles, hypotheses, observed.GetDevice(Intent.Read), ctf.GetDevice(Intent.Read),
            quadrature.GetDevice(Intent.Read), inverseNoise.GetDevice(Intent.Read), radiusSquared.GetDevice(Intent.Read),
            geometry, bounds, symmetry, symmetries.Length, poses, seedIds, pixel, cutoff, diameter,
            options.RefineIterations, mergeDistance,
            MathF.Asin(Math.Min(1, 2 * mergeDistance / diameter)), summary, diagnostic, statistics);
        matchOptimizationSeconds += batchTimer.Elapsed.TotalSeconds; batchTimer.Restart();
        if (status != 0) throw new InvalidOperationException($"Batched GPU BFGS pose refinement failed with status {status}.");
        for (int p = 0; p < particles; p++)
            for (int h = 0; h < starts[p].Count; h++)
            {
                int slot = p * hypotheses + h, d = slot * 4;
                if (diagnostic[d + 2] >= 3 || !(summary[d + 1] > 0))
                {
                    result[p].Add(Rejected(starts[p][h], usableTilts[p] < 3
                        ? TemplateMatchTerminationReason.InsufficientTilts : TemplateMatchTerminationReason.InvalidPose));
                    continue;
                }
                double cross = summary[d], power = summary[d + 1], z = cross / Math.Sqrt(power);
                if (!double.IsFinite(z))
                {
                    result[p].Add(Rejected(starts[p][h], TemplateMatchTerminationReason.InvalidPose));
                    continue;
                }
                // Preserve the established diagnostic layout; derivatives are no longer copied to host.
                double[] tiltStatistics = new double[NTilts * 14];
                for (int t = 0; t < NTilts; t++)
                {
                    tiltStatistics[t * 14] = statistics[(slot * NTilts + t) * 2];
                    tiltStatistics[t * 14 + 1] = statistics[(slot * NTilts + t) * 2 + 1];
                }
                int mergedSlot = diagnostic[d + 3];
                result[p].Add(new MatchSolution
                {
                    Position = anchors[p] + new float3(poses[slot * 12], poses[slot * 12 + 1], poses[slot * 12 + 2]),
                    Rotation = new Matrix3(poses.Skip(slot * 12 + 3).Take(9).ToArray()),
                    Z = z, Gain = 0.5 * Math.Pow(Math.Max(z, 0), 2), Amplitude = Math.Max(cross, 0) / power,
                    StartIndex = starts[p][h].StartIndex, Iterations = diagnostic[d], Evaluations = diagnostic[d + 1],
                    Converged = diagnostic[d + 2] == 1,
                    TerminationReason = diagnostic[d + 2] == 1 ? TemplateMatchTerminationReason.StepTolerance
                        : diagnostic[d + 2] == 2 ? TemplateMatchTerminationReason.LineSearchStalled : TemplateMatchTerminationReason.IterationLimit,
                    MergedIntoStart = mergedSlot >= 0 ? starts[p][mergedSlot].StartIndex : -1,
                    InitialZ = summary[d + 2], TiltStatistics = tiltStatistics
                });
            }
        if (fitEnvelope)
        {
            // Only the winning surviving pose is needed. Keep its original native pose
            // values (rather than subtracting large absolute coordinates again).
            var winners = result.Select(group => group.Where(s => s.MergedIntoStart == -1 && s.Z > 0)
                .OrderByDescending(s => s.Z).FirstOrDefault()).ToArray();
            float[] envelopePoses = new float[particles * 12];
            int[] active = Enumerable.Repeat(-1, particles).ToArray();
            for (int p = 0; p < particles; p++)
                if (winners[p] != null)
                {
                    int h = starts[p].FindIndex(s => s.StartIndex == winners[p].StartIndex);
                    Array.Copy(poses, (p * hypotheses + h) * 12, envelopePoses, p * 12, 12);
                    active[p] = winners[p].StartIndex;
                }
            if (active.Any(id => id >= 0))
            {
                const int bins = TemplateMatchEnvelope.SpectrumBins;
                float maximumQ2 = phaseRadii.Max(view => view.Max());
                double minimumQ2 = options.RefineFitHighpass > 0 ? 1 / Math.Pow((double)options.RefineFitHighpass, 2) : 0;
                float[] spectra = new float[particles * 2 * bins];
                status = GPU.TemplateMatchEnvelopeSpectra(projector.t_DataRe, projector.t_DataIm, projector.Data.Dims.X,
                    box, NTilts, particles, observed.GetDevice(Intent.Read), ctf.GetDevice(Intent.Read),
                    quadrature.GetDevice(Intent.Read), inverseNoise.GetDevice(Intent.Read), radiusSquared.GetDevice(Intent.Read),
                    geometry, envelopePoses, active, cutoff, bins, (float)minimumQ2, maximumQ2, spectra);
                if (status != 0) throw new InvalidOperationException($"GPU envelope spectra failed with status {status}.");
                for (int p = 0; p < particles; p++)
                    if (active[p] >= 0)
                        winners[p].EnvelopeSpectrum = new TemplateMatchEnvelopeSpectrum(maximumQ2,
                            spectra.AsSpan(p * 2 * bins, bins).ToArray(), spectra.AsSpan((p * 2 + 1) * bins, bins).ToArray(), minimumQ2);
                if (options.RefineExportTiltSpectra)
                {
                    float[] byTilt = new float[checked(particles * NTilts * 2 * bins)];
                    status = GPU.TemplateMatchEnvelopeSpectraByTilt(projector.t_DataRe, projector.t_DataIm, projector.Data.Dims.X,
                        box, NTilts, particles, observed.GetDevice(Intent.Read), ctf.GetDevice(Intent.Read),
                        quadrature.GetDevice(Intent.Read), inverseNoise.GetDevice(Intent.Read), radiusSquared.GetDevice(Intent.Read),
                        geometry, envelopePoses, active, cutoff, bins, (float)minimumQ2, maximumQ2, byTilt);
                    if (status != 0) throw new InvalidOperationException($"GPU per-tilt envelope spectra failed with status {status}.");
                    for (int p = 0; p < particles; p++)
                        if (active[p] >= 0)
                            winners[p].TiltEnvelopeSpectra = Enumerable.Range(0, NTilts).Select(t =>
                                new TemplateMatchEnvelopeSpectrum(maximumQ2,
                                    byTilt.AsSpan((p * NTilts + t) * 2 * bins, bins).ToArray(),
                                    byTilt.AsSpan(((p * NTilts + t) * 2 + 1) * bins, bins).ToArray(), minimumQ2)).ToArray();
                }
            }
        }
        matchResultSeconds += batchTimer.Elapsed.TotalSeconds;
        return result;
    }

    private void WriteMatchTiltEnvelopeSpectra(string prefix, List<(int OriginalPeak, MatchSolution Solution)> rows, float pixel)
    {
        // Diagnostic export only: fitting one shared scale per tilt happens from
        // these sufficient statistics, without altering stored image/CTF metadata.
        using BinaryWriter binary = new(File.Create(prefix + "_tilt_envelope_spectra.bin"));
        binary.Write(System.Text.Encoding.ASCII.GetBytes("WRPTENV1"));
        binary.Write(1); binary.Write(rows.Count); binary.Write(NTilts); binary.Write(TemplateMatchEnvelope.SpectrumBins);
        binary.Write(rows.Count > 0 ? rows[0].Solution.EnvelopeSpectrum.MinimumFrequencySquared : 0);
        foreach (var row in rows)
        {
            var spectra = row.Solution.TiltEnvelopeSpectra;
            if (spectra == null || spectra.Length != NTilts) throw new InvalidOperationException("Missing per-tilt spectra.");
            binary.Write(row.OriginalPeak); binary.Write(row.Solution.EnvelopeSpectrum.MaximumFrequencySquared);
            foreach (var spectrum in spectra)
            {
                foreach (float value in spectrum.Cross) binary.Write(value);
                foreach (float value in spectrum.Power) binary.Write(value);
            }
        }
        using StreamWriter table = new(prefix + "_tilt_model.tsv");
        table.WriteLine("tilt\tangle_deg\tdose_e_A2\tused\tmodel_scale_at_center\tmodel_bfactor_at_center_A2\tmovie");
        for (int t = 0; t < NTilts; t++)
        {
            CTF ctf = GetCTFParamsForOneTilt(pixel, [0], [VolumeDimensionsPhysical / 2], t, weighted: true)[0];
            table.WriteLine(FormattableString.Invariant(
                $"{t}\t{Angles[t]:R}\t{Dose[t]:R}\t{UseTilt[t]}\t{ctf.Scale}\t{ctf.Bfactor}\t{TiltMoviePaths[t]}"));
        }
    }

    private static void WriteMatchEnvelopes(string prefix, List<(int OriginalPeak, MatchSolution Solution)> rows)
    {
        if (rows.Any(row => row.Solution.EnvelopeSpectrum == null))
            throw new InvalidOperationException("Missing envelope spectrum for a retained candidate.");
        var fits = new TemplateMatchEnvelopeFit[rows.Count];
        System.Threading.Tasks.Parallel.For(0, rows.Count, i =>
            fits[i] = TemplateMatchEnvelope.Fit(rows[i].Solution.EnvelopeSpectrum, 0));
        // Give every fitted candidate equal weight when choosing one common
        // reporting pivot near the signal that survives its envelope. This is
        // an exact reparameterization: B, the prediction and gain do not change.
        double referenceQ2 = fits.Select(fit => fit.MeanFrequencySquared).Where(double.IsFinite).DefaultIfEmpty(0).Average();
        double minimumQ2 = rows.Count > 0 ? rows[0].Solution.EnvelopeSpectrum.MinimumFrequencySquared : 0;
        double highpass = minimumQ2 > 0 ? 1 / Math.Sqrt(minimumQ2) : 0;
        foreach (var fit in fits) fit.SetReferenceFrequencySquared(referenceQ2);
        using StreamWriter table = new(prefix + "_envelope.tsv");
        table.WriteLine("star_row\tpeak\tx_A\ty_A\tz_A\traw_z\traw_amplitude\treference_q2_A-2\tb_A2\tamplitude\tlog_amplitude\tfitted_z\tprofile_gain\tdelta_gain\tsigma_b_A2\tsigma_log_amplitude\tcorr_log_amplitude_b\tstatus\tfit_highpass_A\tfit_z_at_b0\tfit_amplitude_at_b0");
        // Reproducible sufficient statistics, little endian: magic (8 bytes),
        // version/count/bins (int32), common qref² and minimum q² (float64); then per row:
        // original peak (int32), max q² (float32), C[bins], P[bins] (float32).
        using BinaryWriter binary = new(File.Create(prefix + "_envelope_spectra.bin"));
        binary.Write(System.Text.Encoding.ASCII.GetBytes("WRPENV01"));
        binary.Write(2); binary.Write(rows.Count); binary.Write(TemplateMatchEnvelope.SpectrumBins); binary.Write(referenceQ2); binary.Write(minimumQ2);
        for (int i = 0; i < rows.Count; i++)
        {
            var row = rows[i];
            MatchSolution s = row.Solution;
            TemplateMatchEnvelopeFit f = fits[i];
            table.WriteLine(FormattableString.Invariant(
                $"{i + 1}\t{row.OriginalPeak}\t{s.Position.X:R}\t{s.Position.Y:R}\t{s.Position.Z:R}\t{s.Z:R}\t{s.Amplitude:R}\t{referenceQ2:R}\t{f.B:R}\t{f.Amplitude:R}\t{f.LogAmplitude:R}\t{f.Z:R}\t{f.Gain:R}\t{f.Gain - f.GainAtZeroB:R}\t{f.SigmaB:R}\t{f.SigmaLogAmplitude:R}\t{f.CorrelationLogAmplitudeB:R}\t{f.Status}\t{highpass:R}\t{f.ZAtZeroB:R}\t{f.AmplitudeAtZeroB:R}"));
            binary.Write(row.OriginalPeak); binary.Write(s.EnvelopeSpectrum.MaximumFrequencySquared);
            foreach (float value in s.EnvelopeSpectrum.Cross) binary.Write(value);
            foreach (float value in s.EnvelopeSpectrum.Power) binary.Write(value);
        }
    }
}
