using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using Accord;
using Accord.Math.Optimization;
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
        bool IsCanceled = false;
        if (!Directory.Exists(MatchingDir))
            Directory.CreateDirectory(MatchingDir);

        string NameWithRes = TiltSeries.ToTomogramWithPixelSize(Path, options.BinnedPixelSizeMean);

        float3[] HealpixAngles = Helper.GetHealpixAngles(options.HealpixOrder, options.Symmetry).Select(a => a * Helper.ToRad).ToArray();
        if (options.TiltRange > 0)
        {
            float Limit = MathF.Sin((float)options.TiltRange * Helper.ToRad);
            HealpixAngles = HealpixAngles.Where(a => MathF.Abs(Matrix3.Euler(a).C3.Z) <= Limit).ToArray();
        }

        progressCallback?.Invoke(0, $"Using {HealpixAngles.Length} orientations for matching");

        LoadMovieSizes();

        Image CorrVolume = null, AngleIDVolume = null;
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

        #region Get correlation and angles either by calculating them from scratch, or by loading precalculated volumes

        string CorrVolumePath = IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_corr.mrc");
        string AngleIDVolumePath = IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_angleid.mrc");

        if (!File.Exists(IOPath.Combine(ReconstructionDir, NameWithRes + ".mrc")))
            throw new FileNotFoundException("A reconstruction at the desired resolution was not found.");

        Image TomoRec = null;

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
            Image TemplateMask;
            int TemplateMaskSum = 0;
            {
                int SizeBinned = (int)Math.Round(template.Dims.X * (options.TemplatePixel / options.BinnedPixelSizeMean) / 2) * 2;

                Image TemplateScaled = template.AsScaled(new int3(SizeBinned));
                template.FreeDevice();

                TemplateScaled.MaskSpherically(SizeParticle / 2, Math.Max(5, 20 / (float)options.BinnedPixelSizeMean), true);

                Image TemplatePadded = TemplateScaled.AsPadded(DimsVolumeCube).AndDisposeParent();
                //TemplatePadded.WriteMRC("d_template.mrc", true);

                ProjectorReference = new Projector(TemplatePadded, 2, true, 3);
                TemplatePadded.Dispose();
                ProjectorReference.PutTexturesOnDevice();
            }

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

            #endregion

            #region Match

            progressCallback?.Invoke(0, "Matching...");

            float[] ProgressFraction = new float[1];
            {
                #region Perform correlation

                Image TomoRecFT = TomoRec.AsFFT(true);
                TomoRec.FreeDevice();

                Timer ProgressTimer = new Timer((a) =>
                                                    progressCallback?.Invoke(ProgressFraction[0], "Matching..."), null, 1000, 1000);

                GPU.CorrelateLargeVolume(ProjectorReference.t_DataRe,
                                      ProjectorReference.t_DataIm,
                                      ProjectorReference.Oversampling,
                                      ProjectorReference.Data.Dims,
                                      TomoRecFT.GetDevice(Intent.Read),
                                      TemplateCTF.GetDevice(Intent.Read),
                                      DimsVolumePadded,
                                      Helper.ToInterleaved(HealpixAngles),
                                      (uint)HealpixAngles.Length,
                                      (uint)options.BatchAngles,
                                      SizeParticle / 2,
                                      CorrVolume.GetDevice(Intent.Write),
                                      AngleIDVolume.GetDevice(Intent.Write),
                                      ProgressFraction);

                #endregion

                TomoRecFT.Dispose();

                if (options.UseTophat > 0)
                    CorrVolume = CorrVolume.AsTophatFiltered(options.UseTophat).AndDisposeParent();

                ProgressTimer.Dispose();
                if (progressCallback != null)
                    IsCanceled = progressCallback(1.0f, "Matching...");
            }

            #endregion

            #region Postflight

            ProjectorReference.Dispose();
            TemplateCTF.Dispose();

            #region Normalize by local standard deviation of TomoRec

            if (true)
            {
                Image LocalStd = new Image(IntPtr.Zero, TomoRec.Dims);
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
                            CorrData[z][y * DimsVolumeScaled.X + x] *= OccupancyMask[zz][yy * DimsUndersampled.X + xx];
                            AngleIDData[z][y * DimsVolumeScaled.X + x] *= OccupancyMask[zz][yy * DimsUndersampled.X + xx];
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

        #region Get peak list that has at least NResults values

        progressCallback?.Invoke(0, "Extracting best peaks...");

        ParticlePeak[] Peaks;
        {
            int3[] InitialPeaks = new int3[0];
            float Max = CorrVolume.GetHost(Intent.Read).Select(a => a.Max()).Max();

            for (float s = Max * 0.9f; s > Max * 0.1f; s -= Max * 0.05f)
            {
                float Threshold = s;
                InitialPeaks = CorrVolume.GetLocalPeaks(PeakDistance, Threshold);

                if (InitialPeaks.Length >= options.NResults)
                    break;
            }

            List<ParticlePeak> PeakList = new(InitialPeaks.Length);

            for (int i = 0; i < InitialPeaks.Length; i++)
            {
                int z = InitialPeaks[i].Z;
                int xy = InitialPeaks[i].Y * DimsVolumeScaled.X + InitialPeaks[i].X;
                int angleId = (int)(AngleIDData[z][xy] + 0.5f);
                float3 Angles = HealpixAngles[angleId] * Helper.ToDeg;
                float Score = CorrData[z][xy];

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

        #region Optionally refine peak positions and angles using gradient descent

        if (options.OptimizePoses)
        {
            VolumeDimensionsPhysical = options.DimensionsPhysical;

            if (options.OptimizePosesAngPix == null)
            { 
                options.OptimizePosesAngPix = options.BinnedPixelSizeMean;
                options.OptimizePosesSteps = 1;
            }

            decimal[] PixelSizes = Helper.ArrayOfFunction(i => MathHelper.Lerp(options.BinnedPixelSizeMean,
                                                                               options.OptimizePosesAngPix.Value,
                                                                               (decimal)i / Math.Max(1, options.OptimizePosesSteps - 1)),
                                                          options.OptimizePosesSteps);

            foreach (var angpix in PixelSizes)
            {
                Console.WriteLine($"Optimizing peaks at {angpix:F3} Å/px...");

                options.BinTimes = (decimal)Math.Log2((double)(angpix / options.PixelSizeMean));

                #region Dimensions

                int SizeRegion = (int)((float)options.TemplateDiameter * 2f / (float)options.BinnedPixelSizeMean + 1) / 2 * 2;
                float SizeParticleF = (float)options.TemplateDiameter / (float)options.BinnedPixelSizeMean;

                #endregion

                #region Load and preprocess data

                Image[] TiltData;
                Image[] TiltMasks;
                LoadMovieData(options, out _, out TiltData, false, out _, out _);
                LoadMovieMasks(options, out TiltMasks);
                for (int z = 0; z < NTilts; z++)
                {
                    EraseDirt(TiltData[z], TiltMasks[z]);
                    TiltMasks[z]?.FreeDevice();

                    TiltData[z].SubtractMeanGrid(new int2(1));
                    TiltData[z] = TiltData[z].AsPaddedClampedSoft(new int2(TiltData[z].Dims) * 2, 32).AndDisposeParent();
                    TiltData[z].MaskRectangularly((TiltData[z].Dims / 2).Slice(), MathF.Min(TiltData[z].Dims.X / 4, TiltData[z].Dims.Y / 4), false);
                    //TiltData[z].WriteMRC("d_tiltdata.mrc", true);
                    TiltData[z].Bandpass(1f / (SizeParticleF / 2), 1f, false, 1f / (SizeParticleF / 2));
                    TiltData[z] = TiltData[z].AsPadded(new int2(TiltData[z].Dims) / 2).AndDisposeParent();
                    //TiltData[z].WriteMRC(IOPath.Combine(IOPath.GetDirectoryName(Path), $"d_tiltdatabp_{RootName}_{z}.mrc"), true);

                    if (!options.DontInvert)
                        TiltData[z].Multiply(-1f);

                    TiltData[z].Normalize();

                    //float2 Stats = MathHelper.MeanAndStd(TiltData[z].GetHost(Intent.Read)[0]);
                    //Console.WriteLine($"Tilt {z}: {Stats.X} +- {Stats.Y}");
                }
                GPU.CheckGPUExceptions();

                #endregion

                #region Projector

                Projector Projector;
                {
                    int SizeTemplatePadded = Math.Max(template.Dims.X,
                                                      (int)Math.Round(options.TemplateDiameter * 2 / options.TemplatePixel / 2) * 2);
                    using Image TemplatePadded = template.Dims.X != SizeTemplatePadded ?
                                                    template.AsPadded(new int3(SizeTemplatePadded)) :
                                                    template.GetCopy();

                    int SizeTemplateScaled = (int)Math.Round(SizeTemplatePadded * options.TemplatePixel / options.BinnedPixelSizeMean / 2) * 2;
                    using Image TemplateScaled = TemplatePadded.Dims.X != SizeTemplateScaled ?
                                                       TemplatePadded.AsScaled(new int3(SizeTemplateScaled)) :
                                                       TemplatePadded.GetCopy();

                    using Image TemplateCropped = TemplateScaled.Dims.X != SizeRegion ?
                                                       TemplateScaled.AsPadded(new int3(SizeRegion)) :
                                                       TemplateScaled.GetCopy();

                    {
                        float2 Stats = MathHelper.MeanAndStd(TemplateCropped.GetHost(Intent.Read));
                        Console.WriteLine($"Template dims: {TemplateCropped.Dims}");
                        Console.WriteLine($"Template stats: {Stats.X} +- {Stats.Y}");
                    }

                    Projector = new Projector(TemplateCropped, 2, true);
                    GPU.CheckGPUExceptions();

                    {
                        using Image Proj = Projector.ProjectToRealspace(new int2(SizeRegion), [new float3(0, 0, 0)]);
                        float2 Stats = MathHelper.MeanAndStd(Proj.GetHost(Intent.Read));
                        Console.WriteLine($"Proj stats: {Stats.X} +- {Stats.Y}");
                    }
                }

                #endregion

                Func<ParticlePeak[], float, int, ParticlePeak[]> OptimizeParticles = (peaks, shiftSigma, iterations) =>
                {
                    int NParticles = peaks.Length;
                    var positions = peaks.Select(p => p.PositionF).ToArray();
                    var angles = peaks.Select(p => p.Angles).ToArray();

                    #region Memory and FFT plan allocation

                    int PlanForwParticles = GPU.CreateFFTPlan(new int3(SizeRegion, SizeRegion, 1), (uint)NParticles);
                    int PlanBackParticles = GPU.CreateIFFTPlan(new int3(SizeRegion, SizeRegion, 1), (uint)NParticles);

                    using Image Images = new(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles));
                    using Image ImagesFT = new(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles), true, true);

                    using Image CTFCoords = CTF.GetCTFCoords(SizeRegion, SizeRegion, Matrix2.Identity());
                    using Image CTFs = new Image(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles), true);
                    GPU.CheckGPUExceptions();

                    #endregion

                    float3[] PositionsOri = positions.ToArray();
                    float3[] AnglesOri = angles.ToArray();

                    int NParamsParticles = NParticles * 6;

                    double BestScore = double.NegativeInfinity;
                    double[] BestInput = null;

                    var SetPositions = (double[] input) =>
                    {
                        for (int p = 0; p < NParticles; p++)
                        {
                            positions[p].X = (float)input[p * 6 + 0] + PositionsOri[p].X;
                            positions[p].Y = (float)input[p * 6 + 1] + PositionsOri[p].Y;
                            positions[p].Z = (float)input[p * 6 + 2] + PositionsOri[p].Z;

                            var UpdatedRotation = Matrix3.Euler(AnglesOri[p] * Helper.ToRad) *
                                                    Matrix3.RotateX((float)input[p * 6 + 3] * Helper.ToRad) *
                                                    Matrix3.RotateY((float)input[p * 6 + 4] * Helper.ToRad) *
                                                    Matrix3.RotateZ((float)input[p * 6 + 5] * Helper.ToRad);
                            var UpdatedAngles = Matrix3.EulerFromMatrix(UpdatedRotation) * Helper.ToDeg;

                            angles[p].X = UpdatedAngles.X;
                            angles[p].Y = UpdatedAngles.Y;
                            angles[p].Z = UpdatedAngles.Z;
                        }
                    };

                    Image[] ParticleCTFs = new Image[NTilts];
                    Image[] ParticleWeights = new Image[NTilts];
                    CTF[] Weights = new CTF[NTilts];
                    {
                        for (int t = 0; t < NTilts; t++)
                        {
                            float3[] ParticlePositions = positions.ToArray();
                            float3[] ParticleAngles = angles.ToArray();

                            float3[] ParticlePositionsInImage = GetPositionsInOneTilt(ParticlePositions, t);
                            float3[] ParticleAnglesInImage = GetAnglesInOneTilt(ParticlePositions, ParticleAngles, t);

                            ParticleCTFs[t] = new Image(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles), true);
                            GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                              ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                                              ParticlePositionsInImage,
                                              CTFCoords,
                                              null,
                                              t,
                                              ParticleCTFs[t]);

                            ParticleWeights[t] = new Image(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles), true);
                            GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                              ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                                              ParticlePositionsInImage,
                                              CTFCoords,
                                              null,
                                              t,
                                              ParticleWeights[t],
                                              weighted: true,
                                              weightsonly: true);

                            Weights[t] = GetCTFParamsForOneTilt((float)options.BinnedPixelSizeMean,
                                                                 [1f],
                                                                 [VolumeDimensionsPhysical * 0.5f],
                                                                 t,
                                                                 weighted: true,
                                                                 weightsonly: true)[0];
                        }
                    }

                    Func<double[], float, double[]> EvalParticles = (input, shiftSigma) =>
                    {
                        SetPositions(input);

                        double[] Result = new double[NParticles];

                        for (int t = 0; t < NTilts; t++)
                        {
                            float3[] ParticlePositions = positions.ToArray();
                            float3[] ParticleAngles = angles.ToArray();

                            float3[] ParticlePositionsInImage = GetPositionsInOneTilt(ParticlePositions, t);
                            float3[] ParticleAnglesInImage = GetAnglesInOneTilt(ParticlePositions, ParticleAngles, t);

                            GetParticleImagesFromOneTilt(options,
                                                         TiltData,
                                                         t,
                                                         SizeRegion,
                                                         ParticlePositions,
                                                         PlanForwParticles,
                                                         true,
                                                         Images,
                                                         ImagesFT);
                            GPU.CheckGPUExceptions();

                            using Image References = Projector.Project(new int2(SizeRegion), ParticleAnglesInImage);
                            GPU.CheckGPUExceptions();

                            References.Multiply(ParticleCTFs[t]);
                            ImagesFT.Multiply(ParticleWeights[t]);
                            GPU.CheckGPUExceptions();

                            GPU.IFFT(ImagesFT.GetDevice(Intent.ReadWrite),
                                     Images.GetDevice(Intent.Write),
                                     new int3(SizeRegion).Slice(),
                                     (uint)NParticles,
                                     PlanBackParticles,
                                     normalize: false);
                            Images.Normalize();
                            Images.MaskSpherically((float)(options.TemplateDiameter / options.BinnedPixelSizeMean),
                                                   (float)(60 / options.BinnedPixelSizeMean),
                                                   false,
                                                   true);
                            Images.Normalize();

                            using Image ReferencesIFT = References.AsIFFT(false, PlanBackParticles);
                            ReferencesIFT.Normalize();

                            //Console.WriteLine($"Images stats:");
                            //foreach (var layer in Images.GetHost(Intent.Read))
                            //{
                            //    float2 Stats = MathHelper.MeanAndStd(layer);
                            //    Console.WriteLine($"{Stats.X} +- {Stats.Y}");
                            //}

                            //Console.WriteLine($"ReferencesIFT stats:");
                            //foreach (var layer in ReferencesIFT.GetHost(Intent.Read))
                            //{
                            //    float2 Stats = MathHelper.MeanAndStd(layer);
                            //    Console.WriteLine($"{Stats.X} +- {Stats.Y}");
                            //}

                            Images.Multiply(ReferencesIFT);
                            using var Sums = Images.AsSum2D();
                            var SumsData = Sums.GetHost(Intent.Read)[0];
                            for (int p = 0; p < NParticles; p++)
                                Result[p] += SumsData[p] * (float)Weights[t].Scale;
                        }

                        float[] Priors = Enumerable.Repeat(1f, NParticles).ToArray();
                        if (shiftSigma > 0)
                        {
                            for (int p = 0; p < NParticles; p++)
                            {
                                float Shift2 = (positions[p] - PositionsOri[p]).LengthSq();
                                Priors[p] = MathF.Exp(-Shift2 / (2 * shiftSigma * shiftSigma));
                            }
                        }

                        return Result.Select((v, i) => v / (SizeRegion * SizeRegion * NTilts * NParticles) * Priors[i] * 100).ToArray();
                    };

                    Func<double[], double> Eval = (input) =>
                    {
                        double[] Indiv = EvalParticles(input, shiftSigma);
                        double Result = Indiv.Sum();
                        Console.WriteLine($"Current score: {Result}");

                        if (Result > BestScore)
                        {
                            BestScore = Result;
                            BestInput = input.ToArray();
                        }

                        return Result;
                    };

                    int OptIterations = 0;
                    Func<double[], double[]> Grad = (input) =>
                    {
                        double[] Result = new double[input.Length];
                        double Delta = 0.05;

                        if (++OptIterations > iterations)
                            return Result;

                        for (int icomp = 0; icomp < 6; icomp++)
                        {
                            double[] InputPlus = input.ToArray();
                            for (int p = 0; p < NParticles; p++)
                                InputPlus[p * 6 + icomp] += Delta;
                            double[] EvalPlus = EvalParticles(InputPlus, shiftSigma);

                            double[] InputMinus = input.ToArray();
                            for (int p = 0; p < NParticles; p++)
                                InputMinus[p * 6 + icomp] -= Delta;
                            double[] EvalMinus = EvalParticles(InputMinus, shiftSigma);

                            for (int p = 0; p < NParticles; p++)
                                Result[p * 6 + icomp] = (EvalPlus[p] - EvalMinus[p]) / (2 * Delta);
                        }

                        return Result;
                    };

                    double[] InitialScores = EvalParticles(new double[NParamsParticles], 0);

                    if (iterations > 0)
                    {
                        double[] StartParams = new double[NParamsParticles];
                        BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                        Optimizer.Maximize(StartParams);

                        SetPositions(BestInput);
                    }

                    double[] FinalScores = iterations > 0 ?
                                               EvalParticles(BestInput, 0).Select(v => v * NParticles).ToArray() :
                                               InitialScores.ToArray();
                    if (iterations <= 0)
                        BestInput = new double[NParamsParticles];

                    var Shifts = new float3[NParticles];
                    for (int p = 0; p < NParticles; p++)
                        Shifts[p] = new float3((float)BestInput[p * 6 + 0],
                                               (float)BestInput[p * 6 + 1],
                                               (float)BestInput[p * 6 + 2]);
                    float ShiftRMS = MathF.Sqrt(Shifts.Select(v => v.LengthSq()).Average());
                    Console.WriteLine($"Particle shift RMS: {ShiftRMS.ToString("F2", CultureInfo.InvariantCulture)} A");

                    #region Teardown

                    GPU.DestroyFFTPlan(PlanForwParticles);
                    GPU.DestroyFFTPlan(PlanBackParticles);

                    foreach (var img in ParticleCTFs)
                        img.Dispose();
                    foreach (var img in ParticleWeights)
                        img.Dispose();

                    #endregion

                    for (int p = 0; p < NParticles; p++)
                    {
                        if (!double.IsFinite(FinalScores[p]) || FinalScores[p] == 0 ||
                            !double.IsFinite(InitialScores[p]) || InitialScores[p] == 0)
                            throw new Exception("Non-finite score encountered during optimization.");

                        peaks[p].Score = (float)FinalScores[p];
                        peaks[p].PositionF = positions[p];
                        peaks[p].Angles = angles[p];
                    }

                    return peaks;
                };

                int BatchSize = Math.Min(512, Peaks.Length);
                for (int b = 0; b < Peaks.Length; b += BatchSize)
                {
                    int CurrentBatch = Math.Min(BatchSize, Peaks.Length - b);
                    var Batch = Peaks.Skip(b).Take(CurrentBatch).ToArray();

                    var Optimized = OptimizeParticles(Batch, (float)options.BinnedPixelSizeMean * 3, 15);

                    for (int n = 0; n < CurrentBatch; n++)
                        Peaks[b + n] = Optimized[n];
                }

                List<ParticlePeak> BackgroundPeaks = new();
                for (int b = 0; b < 10; b++)
                {
                    List<int3> RandomPositions = new();
                    int i = 0;
                    while (RandomPositions.Count < BatchSize && i < BatchSize * 100)
                    {
                        i++;
                        int z = Random.Shared.Next(DimsVolumeScaled.Z);
                        int y = Random.Shared.Next(DimsVolumeScaled.Y);
                        int x = Random.Shared.Next(DimsVolumeScaled.X);
                        if (CorrData[z][y * DimsVolumeScaled.X + x] != 0)
                            RandomPositions.Add(new(x, y, z));
                    }

                    var InitialPeaks = RandomPositions.Select(p => new ParticlePeak(position: p,
                                                                                    positionf: new float3(p) * (float)options.BinnedPixelSizeMean,
                                                                                    angles: HealpixAngles[(int)(AngleIDData[p.Z][p.Y * DimsVolumeScaled.X + p.X] + 0.5f)],
                                                                                    score: 0)).ToArray();
                    var OptimizedPeaks = OptimizeParticles(InitialPeaks,
                                                           (float)options.BinnedPixelSizeMean * 1,
                                                           0);
                    BackgroundPeaks.AddRange(OptimizedPeaks);
                }

                float2 MedianStd = MathHelper.MedianAndStd(BackgroundPeaks.Select(p => p.Score).ToArray());

                for (int n = 0; n < Peaks.Length; n++)
                {
                    float ZScore = (Peaks[n].Score - MedianStd.X) / MedianStd.Y;
                    Peaks[n].Score = ZScore;
                }

                #region Teardown

                Projector.Dispose();
                foreach (var img in TiltData)
                    img.Dispose();
                foreach (var img in TiltMasks)
                    img?.Dispose();

                GPU.CheckGPUExceptions();

                #endregion
            }
        }

        #endregion

        #region Write out images for quickly assessing different thresholds for picking

        progressCallback?.Invoke(0, "Preparing visualizations...");

        int TemplateThicknessPixels = (int)((float)options.TemplateDiameter / TomoRec.PixelSize);

        // extract projection over central slices of tomogram
        int ZThickness = Math.Max(1, (int)((float)options.TemplateDiameter / TomoRec.PixelSize));
        int ZCenter = (int)(TomoRec.Dims.Z / 2);
        int _ZMin = (int)(ZCenter - (int)((float)ZThickness / 2));
        int _ZMax = (int)(ZCenter + (int)((float)ZThickness / 2));
        Image TomogramSlice = TomoRec.AsRegion(
            origin: new int3(0, 0, _ZMin),
            dimensions: new int3(TomoRec.Dims.X, TomoRec.Dims.Y, ZThickness)
        ).AsReducedAlongZ().AndDisposeParent();

        // write images showing particle picks at different thresholds
        float[] Thresholds = { 3f, 4f, 5f, 6f, 7f, 8f, 9f };
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
            "rlnAutopickFigureOfMerit"
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
                    Score.ToString(CultureInfo.InvariantCulture)
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
}