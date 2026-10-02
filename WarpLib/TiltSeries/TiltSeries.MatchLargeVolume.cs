using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using Accord;
using MathNet.Numerics.Statistics;
using SkiaSharp;
using Warp.Tools;
using ZLinq;
using IOPath = System.IO.Path;

namespace Warp;

public partial class TiltSeries
{
    public void MatchLargeVolume(ProcessingOptionsTomoFullMatch options, Image template, Func<float, string, bool> progressCallback)
    {
        if (options.MatchTopK < 1 || options.RefineStarts < 1 || options.RefineIterations < 1 || options.RefineNoisePatches < 2)
            throw new ArgumentOutOfRangeException(nameof(options), "Top-K, starts, iterations must be positive; at least two noise patches are required.");
        if (options.RefineMaxShift < 0 || options.BatchAngles < 1 || options.NResults < 1)
            throw new ArgumentOutOfRangeException(nameof(options));
        if (options.OptimizePoses && options.ReuseCorrVolumes)
            throw new ArgumentException("Pose refinement requires the top-K orientation lists. Re-run without --reuse_results; legacy volumes contain only one orientation.");
        if (options.OptimizePoses && (options.OptimizePosesSteps < 1 || options.Lowpass <= 0 ||
            options.OptimizePosesAngPix.HasValue && (options.OptimizePosesAngPix <= 0 || options.OptimizePosesAngPix > options.BinnedPixelSizeMean)))
            throw new ArgumentException("Pose refinement requires positive resolution steps and low-pass cutoff, and a positive pixel size no coarser than the tomogram.");
        if (!Directory.Exists(MatchingDir))
            Directory.CreateDirectory(MatchingDir);

        string NameWithRes = TiltSeries.ToTomogramWithPixelSize(Path, options.BinnedPixelSizeMean);

        float3[] HealpixAngles = Helper.GetHealpixAngles(options.HealpixOrder, options.Symmetry).Select(a => a * Helper.ToRad).ToArray();
        if (options.TiltRange > 0)
        {
            float Limit = MathF.Sin((float)options.TiltRange * Helper.ToRad);
            HealpixAngles = HealpixAngles.Where(a => MathF.Abs(Matrix3.Euler(a).C3.Z) <= Limit).ToArray();
        }

        if (HealpixAngles.Length == 0 || HealpixAngles.Length > 16777216)
            throw new ArgumentException("The orientation grid must contain between 1 and 16777216 entries.");
        int KeepPoses = options.OptimizePoses ? Math.Min(options.MatchTopK, HealpixAngles.Length) : 1;
        progressCallback?.Invoke(0, $"Using {HealpixAngles.Length} orientations, retaining {KeepPoses} per voxel");

        LoadMovieSizes();

        Image CorrVolume = null, AngleIDVolume = null, TomoRec = null;
        try
        {
            float[][] CorrData;
            float[][] AngleIDData;

            #region Dimensions

            int SizeSub = options.SubVolumeSize;
            int SizeParticle = (int)(options.TemplateDiameter / options.BinnedPixelSizeMean);
            int PeakDistance = (int)(options.PeakDistance / options.BinnedPixelSizeMean);

            int3 DimsVolumeScaled = new int3((int)Math.Round(options.DimensionsPhysical.X / (float)options.BinnedPixelSizeMean / 2) * 2,
                                             (int)Math.Round(options.DimensionsPhysical.Y / (float)options.BinnedPixelSizeMean / 2) * 2,
                                             (int)Math.Round(options.DimensionsPhysical.Z / (float)options.BinnedPixelSizeMean / 2) * 2);

            VolumeDimensionsPhysical = options.DimensionsPhysical;

            int3 DimsVolumePadded = new int3(MathHelper.NextFFTFriendlySize(DimsVolumeScaled.X),
                                             MathHelper.NextFFTFriendlySize(DimsVolumeScaled.Y),
                                             MathHelper.NextFFTFriendlySize(DimsVolumeScaled.Z + SizeParticle));
            int3 DimsVolumeCube = new int3(DimsVolumePadded.Z);

            // Estimate computational waste
            progressCallback?.Invoke(0, $"Using {DimsVolumePadded} volume size for matching, resulting in " +
                                        $"{((float)DimsVolumePadded.Elements() / DimsVolumeScaled.Elements() * 100 - 100):F0} % overhead");

            #endregion

            // Rank-major GPU leaderboards survive only until sparse peak-neighborhood gathering.
            // Never transfer K full volumes to host memory.
            int3 LeaderboardDims = new(DimsVolumePadded.X, DimsVolumePadded.Y, checked(DimsVolumePadded.Z * KeepPoses));
            using Image RankedScores = options.OptimizePoses ? new Image(IntPtr.Zero, LeaderboardDims) : null;
            using Image RankedAngles = options.OptimizePoses ? new Image(IntPtr.Zero, LeaderboardDims) : null;
            TemplateMatchStart[][] StartingPoses = null;

            #region Get correlation and angles either by calculating them from scratch, or by loading precalculated volumes

            string CorrVolumePath = IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_corr.mrc");
            string AngleIDVolumePath = IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_angleid.mrc");

            if (!File.Exists(IOPath.Combine(ReconstructionDir, NameWithRes + ".mrc")))
                throw new FileNotFoundException("A reconstruction at the desired resolution was not found.");

            if (!File.Exists(CorrVolumePath) || !options.ReuseCorrVolumes)
            {
                progressCallback?.Invoke(0, "Loading...");

                TomoRec = Image.FromFile(IOPath.Combine(ReconstructionDir, NameWithRes + ".mrc"));

                TomoRec = TomoRec.AsPadded(DimsVolumePadded).AndDisposeParent();

                CorrVolume = new Image(DimsVolumePadded);
                CorrVolume.Fill(float.MinValue);
                AngleIDVolume = new Image(DimsVolumePadded);

                float[] SpectrumWhitening = new float[128];

                if (options.Lowpass < 0.999M)
                {
                    TomoRec.BandpassGauss(0, (float)options.Lowpass, true, (float)options.LowpassSigma);
                    //TomoRec.WriteMRC("d_tomorec_lowpass.mrc", true);
                }

                TomoRec.Bandpass(2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 1.5f, 2, true, 2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 0.5f);
                //TomoRec.WriteMRC("d_tomorec_highpass.mrc", true);


                #region Scale and pad/crop the template to the right size, create projector

                progressCallback?.Invoke(0, "Preparing template...");

                Projector ProjectorReference;
                {
                    int SizeBinned = (int)Math.Round(template.Dims.X * (options.TemplatePixel / options.BinnedPixelSizeMean) / 2) * 2;

                    Image TemplateScaled = template.AsScaled(new int3(SizeBinned));
                    template.FreeDevice();

                    MaskMatchTemplate(TemplateScaled, (float)options.TemplateDiameter, (float)options.BinnedPixelSizeMean, true);

                    Image TemplatePadded = TemplateScaled.AsPadded(DimsVolumeCube).AndDisposeParent();
                    //TemplatePadded.WriteMRC("d_template.mrc", true);

                    ProjectorReference = new Projector(TemplatePadded, 2, true, 3);
                    TemplatePadded.Dispose();
                    ProjectorReference.PutTexturesOnDevice();
                }
                using var ReferenceLifetime = ProjectorReference;

                #endregion

                #region Make CTF

                Image TemplateCTF = null;
                {
                    Image CTFCoords = CTF.GetCTFCoords(DimsVolumeCube.X, DimsVolumeCube.X);

                    Image CTFs = GetCTFsForOneParticle(options, VolumeDimensionsPhysical * 0.5f, CTFCoords, null, true, false, false);
                    Image CTFsAbs = GetCTFsForOneParticle(options, VolumeDimensionsPhysical * 0.5f, CTFCoords, null, true, false, false);
                    CTFsAbs.Abs();

                    // CTF has to be converted to complex numbers with imag = 0, and weighted by itself

                    Image CTFsComplex = new Image(CTFs.Dims, true, true);
                    CTFsComplex.Fill(new float2(1, 0));
                    CTFsComplex.Multiply(CTFs);
                    CTFsComplex.Multiply(CTFs);
                    //if (b == 0)
                    //    CTFsComplex.AsAmplitudes().WriteMRC("d_ctfs.mrc", true);

                    // Back-project and reconstruct
                    Projector ProjCTF = new Projector(new int3(DimsVolumeCube), 1);

                    ProjCTF.BackProject(CTFsComplex, CTFsAbs, GetAngleInAllTilts(VolumeDimensionsPhysical * 0.5f), MagnificationCorrection);
                    ProjCTF.Weights.Max(0.01f);

                    CTFsComplex.Dispose();

                    TemplateCTF = ProjCTF.Reconstruct(true, "C1", null, -1, -1, -1, 0);
                    ProjCTF.Dispose();
                    //TemplateCTF.WriteMRC("d_ctf.mrc", true);

                    CTFs.Dispose();
                    CTFsAbs.Dispose();
                    CTFCoords.Dispose();
                }
                using var CTFLifetime = TemplateCTF;

                #endregion

                #region Match

                progressCallback?.Invoke(0, "Matching...");

                float[] ProgressFraction = new float[1];
                {
                    #region Perform correlation

                    using Image TomoRecFT = TomoRec.AsFFT(true);
                    TomoRec.FreeDevice();

                    Timer ProgressTimer = new Timer((a) =>
                                                        progressCallback?.Invoke(ProgressFraction[0], "Matching..."), null, 1000, 1000);

                    try
                    {
                        if (options.OptimizePoses)
                        {
                            GPU.CorrelateLargeVolumeTopK(ProjectorReference.t_DataRe,
                                ProjectorReference.t_DataIm, ProjectorReference.Oversampling,
                                ProjectorReference.Data.Dims, TomoRecFT.GetDevice(Intent.Read),
                                TemplateCTF.GetDevice(Intent.Read), DimsVolumePadded,
                                Helper.ToInterleaved(HealpixAngles), (uint)HealpixAngles.Length,
                                (uint)options.BatchAngles, SizeParticle / 2f, (uint)KeepPoses,
                                RankedScores.GetDevice(Intent.Write), RankedAngles.GetDevice(Intent.Write), ProgressFraction);
                            GPU.CopyDeviceToDevice(RankedScores.GetDevice(Intent.Read), CorrVolume.GetDevice(Intent.Write), DimsVolumePadded.Elements());
                            GPU.CopyDeviceToDevice(RankedAngles.GetDevice(Intent.Read), AngleIDVolume.GetDevice(Intent.Write), DimsVolumePadded.Elements());
                        }
                        else
                        {
                            GPU.CorrelateLargeVolume(ProjectorReference.t_DataRe,
                                ProjectorReference.t_DataIm, ProjectorReference.Oversampling,
                                ProjectorReference.Data.Dims, TomoRecFT.GetDevice(Intent.Read),
                                TemplateCTF.GetDevice(Intent.Read), DimsVolumePadded,
                                Helper.ToInterleaved(HealpixAngles), (uint)HealpixAngles.Length,
                                (uint)options.BatchAngles, SizeParticle / 2f,
                                CorrVolume.GetDevice(Intent.Write), AngleIDVolume.GetDevice(Intent.Write), ProgressFraction);
                        }
                    }
                    finally
                    {
                        ProgressTimer.Dispose();
                    }

                    #endregion

                    TomoRecFT.Dispose();

                    if (options.UseTophat > 0)
                        CorrVolume = CorrVolume.AsTophatFiltered(options.UseTophat).AndDisposeParent();

                    if (progressCallback != null && progressCallback(1.0f, "Matching..."))
                        throw new OperationCanceledException();
                }

                #endregion

                #region Postflight

                ProjectorReference.Dispose();
                TemplateCTF.Dispose();

                #region Normalize by local standard deviation of TomoRec

                if (true)
                {
                    using Image LocalStd = new Image(IntPtr.Zero, TomoRec.Dims);
                    GPU.LocalStd(TomoRec.GetDevice(Intent.Read),
                                 TomoRec.Dims,
                                 SizeParticle / 2,
                                 LocalStd.GetDevice(Intent.Write),
                                 IntPtr.Zero,
                                 0,
                                 0);

                    Image Center = LocalStd.AsPadded(LocalStd.Dims / 2);
                    float Median = Center.GetHost(Intent.Read)[Center.Dims.Z / 2].Median();
                    Center.Dispose();

                    LocalStd.Max(MathF.Max(1e-10f, Median));

                    //LocalStd.WriteMRC("d_localstd.mrc", true);

                    CorrVolume.Divide(LocalStd);
                    if (RankedScores != null)
                        GPU.DivideSlices(RankedScores.GetDevice(Intent.Read), LocalStd.GetDevice(Intent.Read),
                            RankedScores.GetDevice(Intent.Write), DimsVolumePadded.Elements(), (uint)KeepPoses);

                    LocalStd.Dispose();
                }

                #endregion

                CorrVolume = CorrVolume.AsPadded(DimsVolumeScaled).AndDisposeParent();
                CorrData = CorrVolume.GetHost(Intent.Read);
                AngleIDVolume = AngleIDVolume.AsPadded(DimsVolumeScaled).AndDisposeParent();
                AngleIDData = AngleIDVolume.GetHost(Intent.Read);

                #region Normalize by background correlation std

                if (options.NormalizeScores)
                {
                    Image Center = CorrVolume.AsPadded(CorrVolume.Dims / 2);
                    Center.Abs();
                    float[] Sorted = ArrayPool<float>.Rent((int)Center.ElementsReal);
                    for (int z = 0; z < Center.Dims.Z; z++)
                        Array.Copy(Center.GetHost(Intent.Read)[z], 0,
                                   Sorted, z * Center.Dims.Y * Center.Dims.X,
                                   Center.Dims.Y * Center.Dims.X);
                    float Percentile = Sorted.OrderBy(v => v).Skip((int)(Sorted.Length * 0.68f)).First();

                    CorrVolume.Multiply(1f / MathF.Max(1e-20f, Percentile));

                    Center.Dispose();
                    ArrayPool<float>.Return(Sorted);
                }

                #endregion

                #region Zero out correlation values not fully covered by desired number of tilts

                if (options.MaxMissingTilts >= 0)
                {
                    progressCallback?.Invoke(0, "Trimming...");

                    float BinnedAngPix = (float)options.BinnedPixelSizeMean;
                    float Margin = (float)options.TemplateDiameter;

                    int Undersample = 4;
                    int3 DimsUndersampled = (DimsVolumeScaled + Undersample - 1) / Undersample;

                    float3[] ImagePositions = new float3[DimsUndersampled.ElementsSlice() * NTilts];
                    float3[] VolumePositions = new float3[DimsUndersampled.ElementsSlice()];
                    for (int y = 0; y < DimsUndersampled.Y; y++)
                        for (int x = 0; x < DimsUndersampled.X; x++)
                            VolumePositions[y * DimsUndersampled.X + x] = new float3((x + 0.5f) * Undersample * BinnedAngPix,
                                                                                     (y + 0.5f) * Undersample * BinnedAngPix,
                                                                                     0);

                    float[][] OccupancyMask = ArrayPool<float>.RentMultiple(VolumePositions.Length, DimsUndersampled.Z);
                    foreach (var slice in OccupancyMask)
                        for (int i = 0; i < slice.Length; i++)
                            slice[i] = 1;

                    float WidthNoMargin = ImageDimensionsPhysical.X - BinnedAngPix - Margin;
                    float HeightNoMargin = ImageDimensionsPhysical.Y - BinnedAngPix - Margin;

                    for (int z = 0; z < DimsUndersampled.Z; z++)
                    {
                        float ZCoord = (z + 0.5f) * Undersample * BinnedAngPix;
                        for (int i = 0; i < VolumePositions.Length; i++)
                            VolumePositions[i].Z = ZCoord;

                        ImagePositions = GetPositionInAllTiltsNoLocalWarp(VolumePositions, ImagePositions);

                        for (int p = 0; p < VolumePositions.Length; p++)
                        {
                            int Missing = 0;

                            for (int t = 0; t < NTilts; t++)
                            {
                                int i = p * NTilts + t;

                                if (UseTilt[t] &&
                                    (ImagePositions[i].X < Margin || ImagePositions[i].Y < Margin ||
                                     ImagePositions[i].X > WidthNoMargin ||
                                     ImagePositions[i].Y > HeightNoMargin))
                                {
                                    Missing++;

                                    if (Missing > options.MaxMissingTilts)
                                    {
                                        OccupancyMask[z][p] = 0;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    CorrData = CorrVolume.GetHost(Intent.ReadWrite);
                    AngleIDData = AngleIDVolume.GetHost(Intent.ReadWrite);

                    for (int z = 0; z < DimsVolumeScaled.Z; z++)
                    {
                        int zz = z / Undersample;
                        for (int y = 0; y < DimsVolumeScaled.Y; y++)
                        {
                            int yy = y / Undersample;
                            for (int x = 0; x < DimsVolumeScaled.X; x++)
                            {
                                int xx = x / Undersample;
                                if (OccupancyMask[zz][yy * DimsUndersampled.X + xx] == 0)
                                {
                                    CorrData[z][y * DimsVolumeScaled.X + x] = float.NegativeInfinity;
                                    AngleIDData[z][y * DimsVolumeScaled.X + x] = -1;
                                }
                            }
                        }
                    }

                    ArrayPool<float>.ReturnMultiple(OccupancyMask);
                }

                #endregion

                progressCallback?.Invoke(0, "Saving global scores...");

                // Store correlation values and angle IDs for re-use later
                if (!options.DontSaveCorrVolume)
                    CorrVolume.WriteMRC16b(CorrVolumePath, (float)options.BinnedPixelSizeMean, true);
                if (!options.DontSaveAngleIDVolume)
                    AngleIDVolume.WriteMRC(AngleIDVolumePath, (float)options.BinnedPixelSizeMean, true);

                #endregion
            }
            else
            {
                progressCallback?.Invoke(0, "Loading...");

                TomoRec = Image.FromFile(System.IO.Path.Combine(ReconstructionDir, NameWithRes + ".mrc"));

                if (!File.Exists(CorrVolumePath))
                    throw new FileNotFoundException("Pre-existing correlation volume not found.");

                if (!File.Exists(AngleIDVolumePath))
                    throw new FileNotFoundException("Pre-existing angle ID volume not found.");

                CorrVolume = Image.FromFile(CorrVolumePath);
                CorrData = CorrVolume.GetHost(Intent.Read);

                AngleIDVolume = Image.FromFile(AngleIDVolumePath);
                AngleIDData = AngleIDVolume.GetHost(Intent.Read);
            }

            //CorrImage?.Dispose();

            #endregion

            #region Get up to NResults spatial peaks

            progressCallback?.Invoke(0, "Extracting best peaks...");
            // Normalization can leave the device newer than the earlier host views,
            // including when score output and coverage trimming are both disabled.
            CorrData = CorrVolume.GetHost(Intent.Read);
            AngleIDData = AngleIDVolume.GetHost(Intent.Read);

            ParticlePeak[] Peaks;
            {
                int3[] InitialPeaks = CorrVolume.GetLocalPeaks(Math.Max(1, PeakDistance), -float.MaxValue);

                List<ParticlePeak> PeakList = new(InitialPeaks.Length);

                for (int i = 0; i < InitialPeaks.Length; i++)
                {
                    int z = InitialPeaks[i].Z;
                    int xy = InitialPeaks[i].Y * DimsVolumeScaled.X + InitialPeaks[i].X;
                    float storedAngle = AngleIDData[z][xy];
                    int angleId = float.IsFinite(storedAngle) ? (int)Math.Round(storedAngle) : -1;
                    float Score = CorrData[z][xy];
                    if (!float.IsFinite(Score) || angleId < 0 || angleId >= HealpixAngles.Length)
                        continue;
                    float3 Angles = HealpixAngles[angleId] * Helper.ToDeg;

                    PeakList.Add(new(InitialPeaks[i],
                                     new float3(InitialPeaks[i]) * (float)options.BinnedPixelSizeMean,
                                     Angles,
                                     Score));
                }

                PeakList = PeakList.OrderByDescending(p => p.Score).Take(options.NResults).ToList();
                Peaks = PeakList.OrderBy(p => p.Position.Z).ThenBy(p => p.Position.Y).ThenBy(p => p.Position.X).ToArray();
            }
            GPU.CheckGPUExceptions();

            #endregion

            if (options.OptimizePoses && Peaks.Length > 0)
            {
                int3[] Offsets = [new(0, 0, 0), new(-1, 0, 0), new(1, 0, 0), new(0, -1, 0), new(0, 1, 0), new(0, 0, -1), new(0, 0, 1)];
                int3 PadOffset = (DimsVolumePadded - DimsVolumeScaled) / 2;
                int3[] GatherPositions = Peaks.SelectMany(p => Offsets.Select(o => p.Position + o + PadOffset)).ToArray();
                float[] GatherScores = new float[checked(GatherPositions.Length * KeepPoses)];
                float[] GatherAngles = new float[GatherScores.Length];
                int GatherStatus = GPU.GatherTemplateMatchTopK(RankedScores.GetDevice(Intent.Read), RankedAngles.GetDevice(Intent.Read),
                    DimsVolumePadded, GatherPositions, GatherPositions.Length, KeepPoses, GatherScores, GatherAngles);
                if (GatherStatus != 0)
                    throw new InvalidOperationException($"Gathering top-K poses failed with CUDA status {GatherStatus}.");
                StartingPoses = new TemplateMatchStart[Peaks.Length][];
                for (int p = 0; p < Peaks.Length; p++)
                {
                    int3[] Voxels = Offsets.Select(o => Peaks[p].Position + o).ToArray();
                    float[] Scores = GatherScores.Skip(p * 7 * KeepPoses).Take(7 * KeepPoses).ToArray();
                    float[] Angles = GatherAngles.Skip(p * 7 * KeepPoses).Take(7 * KeepPoses).ToArray();
                    for (int v = 0; v < Voxels.Length; v++)
                    {
                        int3 q = Voxels[v];
                        bool Inside = q.X >= 0 && q.Y >= 0 && q.Z >= 0 && q.X < DimsVolumeScaled.X && q.Y < DimsVolumeScaled.Y && q.Z < DimsVolumeScaled.Z;
                        if (!Inside || !float.IsFinite(CorrData[q.Z][q.Y * DimsVolumeScaled.X + q.X]))
                            Array.Fill(Scores, float.NegativeInfinity, v * KeepPoses, KeepPoses);
                    }
                    StartingPoses[p] = TemplateMatching.GatherStarts(Voxels, Scores, Angles, KeepPoses,
                        (float)options.BinnedPixelSizeMean, HealpixAngles, options.RefineStarts);
                }
                // FreeDevice would first copy all K dirty volumes to host memory.
                RankedScores.Dispose();
                RankedAngles.Dispose();
                CorrVolume.FreeDevice();
                AngleIDVolume.FreeDevice();
                TomoRec.FreeDevice();
                Peaks = RefineTemplateMatches(options, template, Peaks, StartingPoses, progressCallback);
            }
            RankedScores?.Dispose();
            RankedAngles?.Dispose();

            #region Write out images for quickly assessing different thresholds for picking

            progressCallback?.Invoke(0, "Preparing visualizations...");
            // Coarse matching padded the reconstruction; picks use the original grid.
            if (TomoRec.Dims != DimsVolumeScaled)
                TomoRec = TomoRec.AsPadded(DimsVolumeScaled).AndDisposeParent();

            // extract projection over central slices of tomogram
            int ZThickness = Math.Clamp((int)((float)options.TemplateDiameter / TomoRec.PixelSize), 1, TomoRec.Dims.Z);
            int _ZMin = (TomoRec.Dims.Z - ZThickness) / 2;
            int _ZMax = _ZMin + ZThickness - 1;
            using Image TomogramSlice = TomoRec.AsRegion(
                origin: new int3(0, 0, _ZMin),
                dimensions: new int3(TomoRec.Dims.X, TomoRec.Dims.Y, ZThickness)
            ).AsReducedAlongZ().AndDisposeParent();

            // write images showing particle picks at different thresholds
            float[] Thresholds = { 3f, 4f, 5f, 6f, 7f, 8f, 9f };
            if (options.OptimizePoses && Peaks.Length > 0)
            {
                float[] SortedScores = Peaks.Select(p => p.Score).OrderBy(s => s).ToArray();
                Thresholds = new[] { 0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99 }
                    .Select(q => SortedScores[(int)(q * (SortedScores.Length - 1))]).Distinct().ToArray();
            }
            string PickingImageDirectory = System.IO.Path.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_picks");
            Directory.CreateDirectory(PickingImageDirectory);

            float2 MeanStd;
            {
                Image CentralQuarter = TomogramSlice.AsPadded(new int2(TomogramSlice.Dims) / 2);
                MeanStd = MathHelper.MeanAndStd(CentralQuarter.GetHost(Intent.Read)[0]);
                CentralQuarter.Dispose();
            }
            float SliceMin = MeanStd.X - MeanStd.Y * 3;
            float SliceMax = MeanStd.X + MeanStd.Y * 3;
            TomogramSlice.TransformValues(v => (v - SliceMin) / (SliceMax - SliceMin) * 255);

            foreach (float threshold in Thresholds)
            {
                var filteredPositions = Peaks.Where(p => p.Score >= threshold && p.Position.Z >= _ZMin && p.Position.Z <= _ZMax)
                                             .Select(p => p.Position)
                                             .ToArray();

                // write PNG with image and draw particle circles
                using (SKBitmap SliceImage = new SKBitmap(TomogramSlice.Dims.X, TomogramSlice.Dims.Y, SKColorType.Bgra8888, SKAlphaType.Opaque))
                {
                    float[] SliceData = TomogramSlice.GetHost(Intent.Read)[0];

                    for (int y = 0; y < TomogramSlice.Dims.Y; y++)
                    {
                        for (int x = 0; x < TomogramSlice.Dims.X; x++)
                        {
                            int i = y * TomogramSlice.Dims.X + x;
                            byte PixelValue = (byte)Math.Max(0, Math.Min(255, SliceData[(TomogramSlice.Dims.Y - 1 - y) * TomogramSlice.Dims.X + x]));
                            var color = new SKColor(PixelValue, PixelValue, PixelValue, 255); // Alpha is set to 255 for opaque
                            SliceImage.SetPixel(x, y, color);
                        }
                    }

                    using (SKCanvas canvas = new SKCanvas(SliceImage))
                    {
                        SKPaint paint = new SKPaint
                        {
                            Color = SKColors.Yellow,
                            IsAntialias = true,
                            Style = SKPaintStyle.Stroke, // Change to Fill for filled circles
                            StrokeWidth = 1.25f
                        };

                        foreach (var position in filteredPositions)
                        {
                            float radius = (((float)options.TemplateDiameter / 2f) * 1.0f) / TomoRec.PixelSize;
                            canvas.DrawCircle(position.X, TomogramSlice.Dims.Y - position.Y, radius: radius, paint);
                        }
                    }

                    string ThresholdedPicksImagePath = Helper.PathCombine(PickingImageDirectory, $"{NameWithRes}_{options.TemplateName}_threshold_{threshold}.png");
                    using (Stream s = File.Create(ThresholdedPicksImagePath))
                    {
                        SliceImage.Encode(s, SKEncodedImageFormat.Png, 100);
                    }
                }
            }

            TomogramSlice.Dispose();

            progressCallback?.Invoke(0, "Done...");

            #endregion

            TomoRec.Dispose();

            #region Write peak positions and angles into table

            Star TableOut = new Star(new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnCoordinateZ",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnMicrographName",
                "rlnAutopickFigureOfMerit",
                "wrpTemplateMatchProjectionZ",
                "wrpTemplateMatchAmplitude"
            });

            {
                for (int n = 0; n < Peaks.Length; n++)
                {
                    //float3 Position = RefinedPositions[n] / new float3(DimsVolumeCropped);
                    //float Score = RefinedScores[n];
                    //float3 Angle = RefinedAngles[n] * Helper.ToDeg;

                    float3 Position = Peaks[n].PositionF;
                    float Score = Peaks[n].Score;
                    float3 Angle = Peaks[n].Angles;
                    float3 PositionF = Position / VolumeDimensionsPhysical;

                    TableOut.AddRow(new string[]
                    {
                        PositionF.X.ToString(CultureInfo.InvariantCulture),
                        PositionF.Y.ToString(CultureInfo.InvariantCulture),
                        PositionF.Z.ToString(CultureInfo.InvariantCulture),
                        Angle.X.ToString(CultureInfo.InvariantCulture),
                        Angle.Y.ToString(CultureInfo.InvariantCulture),
                        Angle.Z.ToString(CultureInfo.InvariantCulture),
                        RootName + ".tomostar",
                        Score.ToString(CultureInfo.InvariantCulture),
                        Peaks[n].ProjectionZ.ToString(CultureInfo.InvariantCulture),
                        Peaks[n].FittedAmplitude.ToString(CultureInfo.InvariantCulture)
                    });
                }
            }

            CorrVolume?.Dispose();
            AngleIDVolume?.Dispose();

            var TableName = string.IsNullOrWhiteSpace(options.OverrideSuffix) ?
                                $"{NameWithRes}_{options.TemplateName}.star" :
                                $"{NameWithRes}{options.OverrideSuffix ?? ""}.star";
            TableOut.Save(IOPath.Combine(MatchingDir, TableName));

            progressCallback?.Invoke(0, "Done.");

            #endregion
        }
        finally
        {
            CorrVolume?.Dispose();
            AngleIDVolume?.Dispose();
            TomoRec?.Dispose();
        }
    }
}
