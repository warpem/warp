using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Accord;
using Warp.Tools;

namespace Warp;

public partial class Movie
{
    public void MatchFull(Image originalStack, ProcessingOptionsFullMatch options, Image template, Func<int3, int, string, bool> progressCallback)
    {
        bool IsCanceled = false;
        if (!Directory.Exists(MatchingDir))
            Directory.CreateDirectory(MatchingDir);

        string NameWithRes = RootName + $"_{options.BinnedPixelSizeMean:F2}Apx";

        float3[] HealpixAngles = Helper.GetHealpixAngles(options.HealpixOrder, options.Symmetry).Select(a => a * Helper.ToRad).ToArray();

        Image CorrImage = null;
        Image AngleImage = null;
        float[] CorrData;
        float[] AngleData;

        GPU.CheckGPUExceptions();

        #region Dimensions

        int SizeSub = options.SubPatchSize;
        int SizeSubPadded = SizeSub * 2;
        int SizeParticle = (int)(options.TemplateDiameter / options.BinnedPixelSizeMean);
        int SizeUseful = Math.Min(SizeSub / 2, SizeSub - SizeParticle * 2); // Math.Min(SizeSub - SizeParticle, SizeSub / 2);
        if (SizeUseful < 2)
            throw new DimensionMismatchException("Particle diameter is bigger than the box.");

        int3 DimsMicrographCropped = new int3(originalStack.Dims.X, originalStack.Dims.Y, 1);

        int3 Grid = (DimsMicrographCropped - SizeParticle + SizeUseful - 1) / SizeUseful;
        Grid.Z = 1;
        List<float3> GridCoords = new List<float3>();
        for (int y = 0; y < Grid.Y; y++)
        for (int x = 0; x < Grid.X; x++)
            GridCoords.Add(new float3(x * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                y * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                0));

        int3 DimsExtraction = new int3(SizeSubPadded, SizeSubPadded, GridCoords.Count);
        int3 DimsParticle = new int3(SizeSub, SizeSub, GridCoords.Count);

        progressCallback?.Invoke(Grid, 0, "Preparing...");

        #endregion

        #region Figure out where to extract, and how much to shift afterwards

        float3[] ParticleCenters = GridCoords.Select(p => new float3(p)).ToArray();

        float3[][] ParticleOrigins = Helper.ArrayOfFunction(z =>
        {
            float Z = z / (float)Math.Max(1, originalStack.Dims.Z - 1);
            return ParticleCenters.Select(p =>
            {
                float2 LocalShift = GetShiftFromPyramid(new float3(p.X / DimsMicrographCropped.X, p.Y / DimsMicrographCropped.Y, Z)) / (float)options.BinnedPixelSizeMean; // Shifts are in physical Angstrom, convert to binned pixels
                return new float3(p.X - LocalShift.X - SizeSubPadded / 2, p.Y - LocalShift.Y - SizeSubPadded / 2, 0);
            }).ToArray();
        }, originalStack.Dims.Z);

        int3[][] ParticleIntegerOrigins = ParticleOrigins.Select(a => a.Select(p => new int3(p.Floor())).ToArray()).ToArray();
        float3[][] ParticleResidualShifts = Helper.ArrayOfFunction(z =>
                Helper.ArrayOfFunction(i =>
                        new float3(ParticleIntegerOrigins[z][i]) - ParticleOrigins[z][i],
                    GridCoords.Count),
            ParticleOrigins.Length);

        #endregion

        #region CTF, phase flipping & dose weighting

        Image CTFCoords = CTF.GetCTFCoords(SizeSub, (int)(SizeSub * options.DownsampleFactor));
        Image GammaCorrection = CTF.GetGammaCorrection((float)options.BinnedPixelSizeMean, SizeSub);
        Image CTFCoordsPadded = CTF.GetCTFCoords(SizeSub * 2, (int)(SizeSub * 2 * options.DownsampleFactor));
        Image[] DoseWeights = null;

        #region Dose

        if (options.DosePerAngstromFrame != 0)
        {
            float DosePerFrame = (float)options.DosePerAngstromFrame;
            if (options.DosePerAngstromFrame < 0)
                DosePerFrame = -(float)options.DosePerAngstromFrame / originalStack.Dims.Z;

            float3 NikoConst = new float3(0.245f, -1.665f, 2.81f);
            Image CTFFreq = CTFCoordsPadded.AsReal();

            GPU.CheckGPUExceptions();

            DoseWeights = Helper.ArrayOfFunction(z =>
            {
                Image Weights = new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, 1), true);

                GPU.DoseWeighting(CTFFreq.GetDevice(Intent.Read),
                    Weights.GetDevice(Intent.Write),
                    (uint)Weights.ElementsSliceComplex,
                    new[] { DosePerFrame * z, DosePerFrame * (z + 1) },
                    NikoConst,
                    options.Voltage > 250 ? 1 : 0.8f, // It's only defined for 300 and 200 kV, but let's not throw an exception
                    1);

                return Weights;
            }, originalStack.Dims.Z);

            GPU.CheckGPUExceptions();

            Image WeightSum = new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, 1), true);
            WeightSum.Fill(1e-15f);
            for (int nframe = 0; nframe < originalStack.Dims.Z; nframe++)
                WeightSum.Add(DoseWeights[nframe]);
            WeightSum.Multiply(1f / originalStack.Dims.Z);

            GPU.CheckGPUExceptions();

            for (int nframe = 0; nframe < originalStack.Dims.Z; nframe++)
            {
                DoseWeights[nframe].Divide(WeightSum);
                //DoseWeights[nframe].WriteMRC($"d_doseweights_{GPU.GetDevice()}_{nframe}.mrc", true);
            }

            GPU.CheckGPUExceptions();

            WeightSum.Dispose();
            CTFFreq.Dispose();

            GPU.CheckGPUExceptions();
        }

        GPU.CheckGPUExceptions();

        #endregion

        #region Create CTF for themplate and padded phase flipping

        Image ExperimentalCTF = new Image(new int3(SizeSub, SizeSub, 1), true);
        Image ExperimentalCTFPadded = new Image(new int3(SizeSub * 2, SizeSub * 2, 1), true);
        CTF CTFParams = CTF.GetCopy();

        GPU.CreateCTF(ExperimentalCTF.GetDevice(Intent.Write),
            CTFCoords.GetDevice(Intent.Read),
            GammaCorrection.GetDevice(Intent.Read),
            (uint)CTFCoords.ElementsComplex,
            new[] { CTFParams.ToStruct() },
            false,
            1);
        ExperimentalCTF.Abs();

        GPU.CreateCTF(ExperimentalCTFPadded.GetDevice(Intent.Write),
            CTFCoordsPadded.GetDevice(Intent.Read),
            GammaCorrection.GetDevice(Intent.Read),
            (uint)CTFCoordsPadded.ElementsComplex,
            new[] { CTFParams.ToStruct() },
            false,
            1);
        ExperimentalCTFPadded.Sign();

        #endregion

        #endregion

        #region Whiten spectrum in images

        Image FlatteningFactors = new Image(new int3(SizeSub, SizeSub, 1), true);
        FlatteningFactors.Fill(1f);

        if (options.WhitenSpectrum)
        {
            progressCallback?.Invoke(Grid, 0, "Whitening spectral noise...");

            Image OriginalStackFlat = originalStack.AsSpectrumFlattened(false, 0.99f, 256);
            float[] AS1D = originalStack.AsAmplitudes1D(false, 0.99f, SizeSub / 2);
            originalStack.FreeDevice();
            originalStack = OriginalStackFlat;

            float[] FlatteningFactorsData = FlatteningFactors.GetHost(Intent.Write)[0];
            Helper.ForEachElementFT(new int2(FlatteningFactors.Dims), (x, y, xx, yy) =>
            {
                int R = (int)((float)Math.Sqrt(xx * xx + yy * yy) / (SizeSub / 2) * AS1D.Length);
                R = Math.Min(AS1D.Length - 1, R);

                FlatteningFactorsData[y * (SizeSub / 2 + 1) + x] = AS1D[R] > 0 ? 1 / AS1D[R] : 0;
            });
        }

        ExperimentalCTF.MultiplySlices(FlatteningFactors);
        FlatteningFactors.Dispose();

        #endregion

        #region Extract and preprocess all patches

        Image AllPatchesFT;
        {
            Image AverageFT = new Image(DimsExtraction, true, true);
            Image Extracted = new Image(IntPtr.Zero, DimsExtraction);
            Image ExtractedFT = new Image(IntPtr.Zero, DimsExtraction, true, true);

            int PlanForw = GPU.CreateFFTPlan(DimsExtraction.Slice(), (uint)GridCoords.Count);
            int PlanBack = GPU.CreateIFFTPlan(DimsExtraction.Slice(), (uint)GridCoords.Count);

            for (int nframe = 0; nframe < originalStack.Dims.Z; nframe++)
            {
                GPU.Extract(originalStack.GetDeviceSlice(nframe, Intent.Read),
                    Extracted.GetDevice(Intent.Write),
                    originalStack.Dims.Slice(),
                    DimsExtraction.Slice(),
                    Helper.ToInterleaved(ParticleIntegerOrigins[nframe]),
                    false,
                    (uint)GridCoords.Count);

                GPU.FFT(Extracted.GetDevice(Intent.Read),
                    ExtractedFT.GetDevice(Intent.Write),
                    DimsExtraction.Slice(),
                    (uint)GridCoords.Count,
                    PlanForw);

                ExtractedFT.MultiplySlices(ExperimentalCTFPadded);

                ExtractedFT.ShiftSlices(ParticleResidualShifts[nframe]);

                if (options.DosePerAngstromFrame != 0)
                    ExtractedFT.MultiplySlices(DoseWeights[nframe]);

                AverageFT.Add(ExtractedFT);
            }

            Image Average = AverageFT.AsIFFT(false, PlanBack, true).AndDisposeParent();

            Image AllPatches = Average.AsPadded(new int2(DimsParticle));
            Average.Dispose();

            GPU.Normalize(AllPatches.GetDevice(Intent.Read),
                AllPatches.GetDevice(Intent.Write),
                (uint)AllPatches.ElementsSliceReal,
                (uint)GridCoords.Count);
            if (options.Invert)
                AllPatches.Multiply(-1f);

            AllPatchesFT = AllPatches.AsFFT();
            AllPatches.Dispose();

            GPU.DestroyFFTPlan(PlanBack);
            GPU.DestroyFFTPlan(PlanForw);
            Extracted.Dispose();
            ExtractedFT.Dispose();
        }

        #endregion

        originalStack.FreeDevice();

        #region Get correlation and angles

        //if (false)
        {
            #region Scale and pad/crop the template to the right size, create projector

            progressCallback?.Invoke(Grid, 0, "Preparing template...");

            Projector ProjectorReference;
            {
                int SizeBinned = (int)Math.Round(template.Dims.X * (options.TemplatePixel / options.BinnedPixelSizeMean) / 2) * 2;

                Image TemplateScaled = template.AsScaled(new int3(SizeBinned));
                template.FreeDevice();

                GPU.SphereMask(TemplateScaled.GetDevice(Intent.Read),
                    TemplateScaled.GetDevice(Intent.Write),
                    TemplateScaled.Dims,
                    SizeParticle / 2,
                    Math.Max(5, 20 / (float)options.BinnedPixelSizeMean),
                    false,
                    1);

                Image TemplatePadded = TemplateScaled.AsPadded(new int3(SizeSub));
                TemplateScaled.Dispose();

                ProjectorReference = new Projector(TemplatePadded, 1, true, 2);
                TemplatePadded.Dispose();
                ProjectorReference.PutTexturesOnDevice();
            }

            #endregion

            CorrData = new float[DimsMicrographCropped.ElementsSlice()];
            AngleData = new float[DimsMicrographCropped.ElementsSlice()];

            progressCallback?.Invoke(Grid, 0, "Matching...");

            int BatchSize = 1;
            for (int b = 0; b < GridCoords.Count; b += BatchSize)
            {
                int CurBatch = Math.Min(BatchSize, GridCoords.Count - b);

                #region Perform correlation

                Image BestCorrelation = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, CurBatch));
                Image BestAngle = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, CurBatch));

                GPU.CorrelateSubTomos(ProjectorReference.t_DataRe,
                    ProjectorReference.t_DataIm,
                    ProjectorReference.Oversampling,
                    ProjectorReference.Data.Dims,
                    AllPatchesFT.GetDeviceSlice(b, Intent.Read),
                    ExperimentalCTF.GetDevice(Intent.Read),
                    new int3(SizeSub, SizeSub, 1),
                    (uint)CurBatch,
                    Helper.ToInterleaved(HealpixAngles),
                    (uint)HealpixAngles.Length,
                    (uint)128,
                    SizeParticle / 2,
                    BestCorrelation.GetDevice(Intent.Write),
                    BestAngle.GetDevice(Intent.Write),
                    null);

                #endregion

                #region Put correlation values and best angle IDs back into the large volume

                for (int st = 0; st < CurBatch; st++)
                {
                    Image ThisCorrelation = new Image(BestCorrelation.GetDeviceSlice(st, Intent.Read), new int3(SizeSub, SizeSub, 1));
                    Image CroppedCorrelation = ThisCorrelation.AsPadded(new int2(SizeUseful));

                    Image ThisAngle = new Image(BestAngle.GetDeviceSlice(st, Intent.Read), new int3(SizeSub, SizeSub, 1));
                    Image CroppedAngle = ThisAngle.AsPadded(new int2(SizeUseful));

                    float[] SubCorr = CroppedCorrelation.GetHostContinuousCopy();
                    float[] SubAngle = CroppedAngle.GetHostContinuousCopy();
                    int3 Origin = new int3(GridCoords[b + st]) - SizeUseful / 2;
                    for (int y = 0; y < SizeUseful; y++)
                    {
                        int yVol = Origin.Y + y;
                        if (yVol >= DimsMicrographCropped.Y - SizeParticle / 2)
                            continue;

                        for (int x = 0; x < SizeUseful; x++)
                        {
                            int xVol = Origin.X + x;
                            if (xVol >= DimsMicrographCropped.X - SizeParticle / 2)
                                continue;

                            CorrData[yVol * DimsMicrographCropped.X + xVol] = SubCorr[y * SizeUseful + x]; // / (SizeSub * SizeSub);
                            AngleData[yVol * DimsMicrographCropped.X + xVol] = SubAngle[y * SizeUseful + x];
                        }
                    }

                    CroppedCorrelation.Dispose();
                    ThisCorrelation.Dispose();
                    CroppedAngle.Dispose();
                    ThisAngle.Dispose();
                }

                #endregion

                BestCorrelation.Dispose();
                BestAngle.Dispose();

                if (progressCallback != null)
                    IsCanceled = progressCallback(Grid, b + CurBatch, "Matching...");
            }

            #region Postflight

            CTFCoords.Dispose();
            GammaCorrection.Dispose();
            CTFCoordsPadded.Dispose();
            ExperimentalCTF.Dispose();
            ExperimentalCTFPadded.Dispose();
            AllPatchesFT.Dispose();
            ProjectorReference.Dispose();

            if (options.Supersample > 1)
            {
                progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Looking for sub-pixel peaks...");

                Image NormalSampled = new Image(CorrData, DimsMicrographCropped);
                Image SuperSampled = new Image(NormalSampled.GetDevice(Intent.Read), NormalSampled.Dims);

                GPU.SubpixelMax(NormalSampled.GetDevice(Intent.Read),
                    SuperSampled.GetDevice(Intent.Write),
                    NormalSampled.Dims,
                    options.Supersample);

                CorrData = SuperSampled.GetHost(Intent.Read)[0];

                NormalSampled.Dispose();
                SuperSampled.Dispose();
            }

            //if (options.KeepOnlyFullVoxels)
            {
                progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Trimming...");

                float Margin = (float)options.TemplateDiameter / (float)options.BinnedPixelSizeMean;

                for (int y = 0; y < DimsMicrographCropped.Y; y++)
                for (int x = 0; x < DimsMicrographCropped.X; x++)
                {
                    if (x < Margin || y < Margin ||
                        x > DimsMicrographCropped.X - Margin ||
                        y > DimsMicrographCropped.Y - Margin)
                    {
                        CorrData[y * DimsMicrographCropped.X + x] = 0;
                    }
                }
            }

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Saving global scores...");

            // Store correlation values and angle IDs for re-use later
            CorrImage = new Image(CorrData, DimsMicrographCropped);
            CorrImage.WriteMRC(System.IO.Path.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_corr.mrc"), (float)options.BinnedPixelSizeMean, true);

            #endregion
        }

        #endregion

        #region Get peak list that has at most nPeaks values

        progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Extracting best peaks...");

        int3[] InitialPeaks = new int3[0];
        {
            float2 MeanAndStd = MathHelper.MeanAndStdNonZero(CorrImage.GetHostContinuousCopy());

            for (float s = 4; s > 0.5f; s -= 0.05f)
            {
                float Threshold = MeanAndStd.X + MeanAndStd.Y * s;
                InitialPeaks = CorrImage.GetLocalPeaks(SizeParticle * 2 / 3, Threshold);

                if (InitialPeaks.Length >= options.NResults)
                    break;
            }
        }

        CorrImage.Dispose();

        #endregion

        #region Write peak positions and angles into table

        Star TableOut = new Star(new string[]
        {
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnMicrographName",
            "rlnAutopickFigureOfMerit"
        });

        {
            for (int n = 0; n < Math.Min(InitialPeaks.Length, options.NResults); n++)
            {
                float3 Position = new float3(InitialPeaks[n]) * (float)options.DownsampleFactor;
                float Score = CorrData[DimsMicrographCropped.ElementFromPosition(InitialPeaks[n])];
                float3 Angle = HealpixAngles[(int)AngleData[DimsMicrographCropped.ElementFromPosition(InitialPeaks[n])]] * Helper.ToDeg;

                TableOut.AddRow(new string[]
                {
                    Position.X.ToString(CultureInfo.InvariantCulture),
                    Position.Y.ToString(CultureInfo.InvariantCulture),
                    Angle.X.ToString(CultureInfo.InvariantCulture),
                    Angle.Y.ToString(CultureInfo.InvariantCulture),
                    Angle.Z.ToString(CultureInfo.InvariantCulture),
                    RootName + ".mrc",
                    Score.ToString(CultureInfo.InvariantCulture)
                });
            }
        }

        TableOut.Save(System.IO.Path.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + ".star"));
        UpdateParticleCount("_" + options.TemplateName);

        progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Done.");

        #endregion
    }
}

[Serializable]
public class ProcessingOptionsFullMatch : ProcessingOptionsBase
{
    [WarpSerializable] public bool OverwriteFiles { get; set; }
    [WarpSerializable] public bool Invert { get; set; }
    [WarpSerializable] public int SubPatchSize { get; set; }
    [WarpSerializable] public string TemplateName { get; set; }
    [WarpSerializable] public decimal TemplatePixel { get; set; }
    [WarpSerializable] public decimal TemplateDiameter { get; set; }
    [WarpSerializable] public decimal TemplateFraction { get; set; }
    [WarpSerializable] public bool WhitenSpectrum { get; set; }
    [WarpSerializable] public decimal DosePerAngstromFrame { get; set; }
    [WarpSerializable] public int Voltage { get; set; }
    [WarpSerializable] public string Symmetry { get; set; }
    [WarpSerializable] public int HealpixOrder { get; set; }
    [WarpSerializable] public int Supersample { get; set; }
    [WarpSerializable] public int NResults { get; set; }
}