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

namespace Warp;

public partial class TiltSeries
{
    public void MatchFull(ProcessingOptionsTomoFullMatch options, Image template, Func<int3, float, string, bool> progressCallback)
    {
        bool IsCanceled = false;
        if (!Directory.Exists(MatchingDir))
            Directory.CreateDirectory(MatchingDir);

        string NameWithRes = TiltSeries.ToTomogramWithPixelSize(Path, options.BinnedPixelSizeMean);

        float3[] HealpixAngles = Helper.GetHealpixAngles(options.HealpixOrder, options.Symmetry).Select(a => a * Helper.ToRad).ToArray();
        if (options.TiltRange >= 0)
        {
            float Limit = MathF.Sin((float)options.TiltRange * Helper.ToRad);
            HealpixAngles = HealpixAngles.Where(a => MathF.Abs(Matrix3.Euler(a).C3.Z) <= Limit).ToArray();
        }

        progressCallback?.Invoke(new int3(1), 0, $"Using {HealpixAngles.Length} orientations for matching");

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

        // Find optimal box size for matching
        {
            int BestSizeSub = 0;
            long BestVoxels = long.MaxValue;

            for (int testSizeSub = (SizeParticle * 2 + 31) / 32 * 32; testSizeSub <= options.SubVolumeSize; testSizeSub += 32)
            {
                int TestSizeUseful = Math.Max(1, testSizeSub - SizeParticle * 2);
                int3 TestGrid = (DimsVolumeScaled - SizeParticle + TestSizeUseful - 1) / TestSizeUseful;
                long TestVoxels = TestGrid.Elements() * testSizeSub * testSizeSub * testSizeSub;

                if (TestVoxels < BestVoxels)
                {
                    BestVoxels = TestVoxels;
                    BestSizeSub = testSizeSub;
                }
            }

            SizeSub = BestSizeSub;

            progressCallback?.Invoke(new int3(1), 0, $"Using {BestSizeSub} sub-volumes for matching, resulting in {((float)BestVoxels / DimsVolumeScaled.Elements() * 100 - 100):F0} % overhead");
        }

        int SizeSubPadded = SizeSub * 2;
        int SizeUseful = SizeSub - SizeParticle * 2; // Math.Min(SizeSub / 2, SizeSub - SizeParticle * 2);// Math.Min(SizeSub - SizeParticle, SizeSub / 2);

        int3 Grid = (DimsVolumeScaled - SizeParticle + SizeUseful - 1) / SizeUseful;
        List<float3> GridCoords = new List<float3>();
        for (int z = 0; z < Grid.Z; z++)
        for (int x = 0; x < Grid.X; x++)
        for (int y = 0; y < Grid.Y; y++)
            GridCoords.Add(new float3(x * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                y * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                z * SizeUseful + SizeUseful / 2 + SizeParticle / 2));

        progressCallback?.Invoke(Grid, (int)Grid.Elements(), $"Using {Grid} sub-volumes");

        #endregion

        #region Get correlation and angles either by calculating them from scratch, or by loading precalculated volumes

        string CorrVolumePath = System.IO.Path.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_corr.mrc");
        string AngleIDVolumePath = System.IO.Path.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_angleid.tif");

        if (!File.Exists(System.IO.Path.Combine(ReconstructionDir, NameWithRes + ".mrc")))
            throw new FileNotFoundException("A reconstruction at the desired resolution was not found.");

        Image TomoRec = null;

        if (!File.Exists(CorrVolumePath) || !options.ReuseCorrVolumes)
        {
            progressCallback?.Invoke(Grid, 0, "Loading...");

            TomoRec = Image.FromFile(System.IO.Path.Combine(ReconstructionDir, NameWithRes + ".mrc"));

            CorrVolume = new Image(DimsVolumeScaled);
            CorrData = CorrVolume.GetHost(Intent.ReadWrite);

            AngleIDVolume = new Image(DimsVolumeScaled);
            AngleIDData = AngleIDVolume.GetHost(Intent.ReadWrite);

            float[] SpectrumWhitening = new float[128];

            if (options.WhitenSpectrum)
            {
                Image CTFZero;
                {
                    Projector Reconstructor = new Projector(new int3(256), 1);
                    Image OnesComplex = new Image(IntPtr.Zero, new int3(256, 256, 1), true, true);
                    OnesComplex.Fill(new float2(1, 0));
                    Image Ones = OnesComplex.AsReal();
                    Reconstructor.BackProject(OnesComplex,
                        Ones,
                        GetAnglesInOneTilt([VolumeDimensionsPhysical * 0.5f], [new float3(0)], IndicesSortedDose[0]),
                        Matrix2.Identity());
                    OnesComplex.Dispose();
                    Ones.Dispose();
                    Reconstructor.Weights.Fill(1);
                    CTFZero = Reconstructor.Reconstruct(true, "C1", null, -1, -1, -1, 0);
                    Reconstructor.Dispose();

                    CTFZero = CTFZero.AsScaledCTF(TomoRec.Dims).AndDisposeParent();
                }

                Image TomoAmps = TomoRec.GetCopyGPU();
                TomoAmps.MaskRectangularly(TomoAmps.Dims - 64, 32, true);
                TomoAmps = TomoAmps.AsFFT(true).AndDisposeParent().AsAmplitudes().AndDisposeParent();

                int NBins = 128; // Math.Max(SizeSub / 2, TomoRec.Dims.Max() / 2);
                double[] Sums = new double[NBins];
                double[] Samples = new double[NBins];

                float[][] TomoData = TomoAmps.GetHost(Intent.Read);
                float[][] CTFData = CTFZero.GetHost(Intent.Read);
                Helper.ForEachElementFT(TomoAmps.Dims, (x, y, z, xx, yy, zz) =>
                {
                    float CTF = MathF.Abs(CTFData[z][y * (CTFZero.Dims.X / 2 + 1) + x]);
                    if (CTF < 1e-2f)
                        return;

                    float xnorm = (float)xx / TomoAmps.Dims.X * 2;
                    float ynorm = (float)yy / TomoAmps.Dims.Y * 2;
                    float znorm = (float)zz / TomoAmps.Dims.Z * 2;
                    float R = MathF.Sqrt(xnorm * xnorm + ynorm * ynorm + znorm * znorm);
                    if (R >= 1)
                        return;

                    R *= Sums.Length;
                    int ID = (int)R;
                    float W1 = R - ID;
                    float W0 = 1f - W1;

                    float Val = TomoData[z][y * (TomoAmps.Dims.X / 2 + 1) + x];
                    Val *= Val;

                    if (W0 > 0)
                    {
                        Sums[ID] += W0 * Val * CTF;
                        Samples[ID] += W0 * CTF;
                    }

                    if (ID < Sums.Length - 1 && W1 > 0)
                    {
                        Sums[ID + 1] += W1 * Val * CTF;
                        Samples[ID + 1] += W1 * CTF;
                    }
                });

                TomoAmps.Dispose();
                CTFZero.Dispose();

                for (int i = 0; i < Sums.Length; i++)
                    Sums[i] = Math.Sqrt(Sums[i] / Math.Max(1e-6, Samples[i]));

                Sums[Sums.Length - 1] = Sums[Sums.Length - 3];
                Sums[Sums.Length - 2] = Sums[Sums.Length - 3];

                SpectrumWhitening = Sums.Select(v => 1 / MathF.Max(1e-10f, (float)v)).ToArray();
                float Max = MathF.Max(1e-10f, SpectrumWhitening.Max());
                SpectrumWhitening = SpectrumWhitening.Select(v => v / Max).ToArray();

                TomoRec = TomoRec.AsSpectrumMultiplied(true, SpectrumWhitening).AndDisposeParent();
                //TomoRec.WriteMRC("d_tomorec_whitened.mrc", true);
            }

            if (options.Lowpass < 0.999M)
            {
                TomoRec.BandpassGauss(0, (float)options.Lowpass, true, (float)options.LowpassSigma);
                //TomoRec.WriteMRC("d_tomorec_lowpass.mrc", true);
            }

            TomoRec.Bandpass(2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 1.5f, 2, true, 2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 0.5f);
            //TomoRec.WriteMRC("d_tomorec_highpass.mrc", true);

            #region Scale and pad/crop the template to the right size, create projector

            progressCallback?.Invoke(Grid, 0, "Preparing template...");

            Projector ProjectorReference, ProjectorMask, ProjectorRandom;
            Image TemplateMask;
            int TemplateMaskSum = 0;
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

                float TemplateMax = TemplateScaled.GetHost(Intent.Read).Select(a => a.Max()).Max();
                TemplateMask = TemplateScaled.GetCopyGPU();
                TemplateMask.Binarize(TemplateMax * 0.2f);
                TemplateMask = TemplateMask.AsDilatedMask(1, true).AndDisposeParent();
                //TemplateMask.WriteMRC("d_templatemask.mrc", true);

                TemplateScaled.Multiply(TemplateMask);
                TemplateScaled.NormalizeWithinMask(TemplateMask, false);
                //TemplateScaled.WriteMRC("d_template_norm.mrc", true);

                #region Make phase-randomized template

                if (false)
                {
                    Random Rng = new Random(123);
                    RandomNormal RngN = new RandomNormal(123);
                    Image TemplateRandomFT = TemplateScaled.AsFFT(true);
                    TemplateRandomFT.TransformComplexValues(v =>
                    {
                        float Amp = v.Length() / TemplateRandomFT.Dims.Elements();
                        float Phase = Rng.NextSingle() * MathF.PI * 2;
                        return new float2(Amp * MathF.Cos(Phase), Amp * MathF.Sin(Phase));
                    });
                    TemplateRandomFT.Bandpass(0, 1, true, 0.01f);
                    //GPU.SymmetrizeFT(TemplateRandomFT.GetDevice(Intent.ReadWrite), TemplateRandomFT.Dims, options.Symmetry);
                    Image TemplateRandom = TemplateRandomFT.AsIFFT(true).AndDisposeParent();
                    TemplateRandom.TransformValues(v => RngN.NextSingle(0, 1));
                    TemplateRandomFT = TemplateRandom.AsFFT(true).AndDisposeParent();
                    GPU.SymmetrizeFT(TemplateRandomFT.GetDevice(Intent.ReadWrite), TemplateRandomFT.Dims, options.Symmetry);
                    TemplateRandom = TemplateRandomFT.AsIFFT(true).AndDisposeParent();
                    TemplateRandom.Multiply(TemplateMask);
                    TemplateRandom.WriteMRC("d_templaterandom.mrc", true);

                    {
                        Image TemplateAmps = TemplateScaled.AsFFT(true).AsAmplitudes().AndDisposeParent();
                        Image RandomAmps = TemplateRandom.AsFFT(true).AsAmplitudes().AndDisposeParent();
                        RandomAmps.Max(1e-16f);
                        Image RandomPhases = TemplateRandom.AsFFT(true);
                        RandomPhases.Divide(RandomAmps);
                        RandomAmps.Dispose();

                        RandomPhases.Multiply(TemplateAmps);
                        TemplateAmps.Dispose();

                        TemplateRandom = RandomPhases.AsIFFT(true).AndDisposeParent();
                        TemplateRandom.Multiply(TemplateMask);
                    }

                    TemplateRandom.NormalizeWithinMask(TemplateMask, true);
                    //TemplateRandom.WriteMRC("d_templaterandom_norm.mrc", true);

                    Image TemplateRandomPadded = TemplateRandom.AsPadded(new int3(SizeSub)).AndDisposeParent();

                    if (options.WhitenSpectrum)
                        TemplateRandomPadded = TemplateRandomPadded.AsSpectrumMultiplied(true, SpectrumWhitening).AndDisposeParent();

                    TemplateRandomPadded.Bandpass(2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 1.5f,
                        2, true,
                        2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 0.5f);
                }

                #endregion

                Image TemplatePadded = TemplateScaled.AsPadded(new int3(SizeSub)).AndDisposeParent();
                //TemplatePadded.WriteMRC("d_template.mrc", true);

                Image TemplateMaskPadded = TemplateMask.AsPadded(new int3(SizeSub));

                TemplateMaskSum = (int)TemplateMask.GetHost(Intent.Read).Select(a => a.Sum()).Sum();
                TemplateMask.Multiply(1f / TemplateMaskSum);

                if (options.WhitenSpectrum)
                {
                    TemplatePadded = TemplatePadded.AsSpectrumMultiplied(true, SpectrumWhitening).AndDisposeParent();
                    //TemplatePadded.WriteMRC("d_template_whitened.mrc", true);
                }

                if (options.Lowpass < 0.999M)
                    TemplatePadded.BandpassGauss(0, (float)options.Lowpass, true, (float)options.LowpassSigma);

                TemplatePadded.Bandpass(2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 1.5f, 2, true, 2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 0.5f);
                //TemplatePadded.WriteMRC("d_template_highpass.mrc", true);

                //TemplateRandomPadded = TemplateRandomPadded.AsSpectrumMultiplied(true, Sinc2).AndDisposeParent();
                //TemplateRandomPadded.WriteMRC("d_templaterandom_filtered.mrc");

                //new Star(TemplatePadded.AsAmplitudes1D(true, 1, 64), "wrpAmplitudes").Save("d_template_amplitudes.star");
                //new Star(TemplateRandomPadded.AsAmplitudes1D(true, 1, 64), "wrpAmplitudes").Save("d_templaterandom_amplitudes.star");

                ProjectorReference = new Projector(TemplatePadded, 2, true, 3);
                TemplatePadded.Dispose();
                ProjectorReference.PutTexturesOnDevice();

                //ProjectorMask = new Projector(TemplateMaskPadded, 2, true, 3);
                //TemplateMaskPadded.Dispose();
                //ProjectorMask.PutTexturesOnDevice();

                //ProjectorRandom = new Projector(TemplateRandomPadded, 2, true, 3);
                //TemplateRandomPadded.Dispose();
                //ProjectorRandom.PutTexturesOnDevice();
            }

            #endregion

            #region Preflight

            if (TomoRec.Dims != DimsVolumeScaled)
                throw new DimensionMismatchException($"Tomogram dimensions ({TomoRec.Dims}) don't match expectation ({DimsVolumeScaled})");

            //if (options.WhitenSpectrum)
            //{
            //    progressCallback?.Invoke(Grid, 0, "Whitening tomogram spectrum...");

            //    TomoRec.WriteMRC("d_tomorec.mrc", true);
            //    TomoRec = TomoRec.AsSpectrumFlattened(true, 0.99f).AndDisposeParent();
            //    TomoRec.WriteMRC("d_tomorec_whitened.mrc", true);
            //}

            float[][] TomoRecData = TomoRec.GetHost(Intent.Read);

            int PlanForw, PlanBack, PlanForwCTF;
            Projector.GetPlans(new int3(SizeSub), 3, out PlanForw, out PlanBack, out PlanForwCTF);

            Image CTFCoords = CTF.GetCTFCoords(SizeSubPadded, SizeSubPadded);

            #endregion

            #region Match

            progressCallback?.Invoke(Grid, 0, "Matching...");

            int BatchSize = Grid.Y;
            float[] ProgressFraction = new float[1];
            for (int b = 0; b < GridCoords.Count; b += BatchSize)
            {
                int CurBatch = Math.Min(BatchSize, GridCoords.Count - b);

                Image Subtomos = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch), true, true);

                #region Create CTF for this column of subvolumes (X = const, Z = const)

                Image SubtomoCTF;
                {
                    Image CTFs = GetCTFsForOneParticle(options, GridCoords[b], CTFCoords, null, true, false, false);
                    //CTFs.Fill(1);
                    Image CTFsAbs = GetCTFsForOneParticle(options, GridCoords[b], CTFCoords, null, false, false, false);
                    CTFsAbs.Abs();

                    // CTF has to be converted to complex numbers with imag = 0, and weighted by itself

                    Image CTFsComplex = new Image(CTFs.Dims, true, true);
                    CTFsComplex.Fill(new float2(1, 0));
                    CTFsComplex.Multiply(CTFs);
                    CTFsComplex.Multiply(CTFs);
                    //if (b == 0)
                    //    CTFsComplex.AsAmplitudes().WriteMRC("d_ctfs.mrc", true);

                    // Back-project and reconstruct
                    Projector ProjCTF = new Projector(new int3(SizeSubPadded), 1);
                    Projector ProjCTFWeights = new Projector(new int3(SizeSubPadded), 1);

                    //ProjCTF.Weights.Fill(0.01f);

                    ProjCTF.BackProject(CTFsComplex, CTFsAbs, GetAngleInAllTilts(GridCoords[b]), MagnificationCorrection);

                    CTFsAbs.Fill(1);
                    ProjCTFWeights.BackProject(CTFsComplex, CTFsAbs, GetAngleInAllTilts(GridCoords[b]), MagnificationCorrection);
                    ProjCTFWeights.Weights.Min(1);
                    ProjCTF.Data.Multiply(ProjCTFWeights.Weights);
                    //ProjCTF.Weights.Fill(1);

                    CTFsComplex.Dispose();
                    ProjCTFWeights.Dispose();

                    Image PSF = ProjCTF.Reconstruct(false, "C1", null, -1, -1, -1, 0);
                    //PSF.WriteMRC("d_psf.mrc", true);
                    PSF.RemapToFT(true);
                    ProjCTF.Dispose();

                    SubtomoCTF = PSF.AsPadded(new int3(SizeSub), true).AndDisposeParent().AsFFT(true).AndDisposeParent().AsReal().AndDisposeParent();
                    SubtomoCTF.Multiply(1f / (SizeSubPadded * SizeSubPadded));

                    CTFs.Dispose();
                    CTFsAbs.Dispose();
                }
                //SubtomoCTF.WriteMRC("d_ctf.mrc", true);

                #endregion

                #region Extract subvolumes and store their FFTs

                for (int st = 0; st < CurBatch; st++)
                {
                    float[][] SubtomoData = new float[SizeSub][];

                    int XStart = (int)GridCoords[b + st].X - SizeSub / 2;
                    int YStart = (int)GridCoords[b + st].Y - SizeSub / 2;
                    int ZStart = (int)GridCoords[b + st].Z - SizeSub / 2;
                    for (int z = 0; z < SizeSub; z++)
                    {
                        SubtomoData[z] = new float[SizeSub * SizeSub];
                        int zz = ZStart + z;

                        for (int y = 0; y < SizeSub; y++)
                        {
                            int yy = YStart + y;
                            for (int x = 0; x < SizeSub; x++)
                            {
                                int xx = XStart + x;
                                if (xx >= 0 && xx < TomoRec.Dims.X &&
                                    yy >= 0 && yy < TomoRec.Dims.Y &&
                                    zz >= 0 && zz < TomoRec.Dims.Z)
                                    SubtomoData[z][y * SizeSub + x] = TomoRecData[zz][yy * TomoRec.Dims.X + xx];
                                else
                                    SubtomoData[z][y * SizeSub + x] = 0;
                            }
                        }
                    }

                    Image Subtomo = new Image(SubtomoData, new int3(SizeSub));

                    // Re-use FFT plan created previously for CTF reconstruction since it has the right size
                    GPU.FFT(Subtomo.GetDevice(Intent.Read),
                        Subtomos.GetDeviceSlice(SizeSub * st, Intent.Write),
                        Subtomo.Dims,
                        1,
                        PlanForwCTF);

                    Subtomo.Dispose();
                }
                //Subtomos.Multiply(1f / (SizeSub * SizeSub * SizeSub));

                #endregion

                #region Perform correlation

                //if (b == 0)
                //    SubtomoCTF.WriteMRC16b("d_ctf.mrc", true);

                Timer ProgressTimer = new Timer((a) =>
                    progressCallback?.Invoke(Grid, b + ProgressFraction[0] * CurBatch, "Matching..."), null, 1000, 1000);

                Image BestCorrelation = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));
                Image BestAngle = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));

                GPU.CorrelateSubTomos(ProjectorReference.t_DataRe,
                    ProjectorReference.t_DataIm,
                    ProjectorReference.Oversampling,
                    ProjectorReference.Data.Dims,
                    Subtomos.GetDevice(Intent.Read),
                    SubtomoCTF.GetDevice(Intent.Read),
                    new int3(SizeSub),
                    (uint)CurBatch,
                    Helper.ToInterleaved(HealpixAngles),
                    (uint)HealpixAngles.Length,
                    (uint)options.BatchAngles,
                    SizeParticle / 2,
                    BestCorrelation.GetDevice(Intent.Write),
                    BestAngle.GetDevice(Intent.Write),
                    ProgressFraction);


                //Image BestCorrelationRandom = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));
                //Image BestAngleRandom = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));

                //GPU.CorrelateSubTomos(ProjectorRandom.t_DataRe,
                //                      ProjectorRandom.t_DataIm,
                //                      ProjectorMask.t_DataRe,
                //                      ProjectorMask.t_DataIm,
                //                      ProjectorRandom.Oversampling,
                //                      ProjectorRandom.Data.Dims,
                //                      Subtomos.GetDevice(Intent.Read),
                //                      SubtomoCTF.GetDevice(Intent.Read),
                //                      new int3(SizeSub),
                //                      (uint)CurBatch,
                //                      Helper.ToInterleaved(HealpixAngles),
                //                      (uint)HealpixAngles.Length,
                //                      (uint)options.BatchAngles,
                //                      SizeParticle / 2,
                //                      BestCorrelationRandom.GetDevice(Intent.Write),
                //                      BestAngleRandom.GetDevice(Intent.Write),
                //                      IntPtr.Zero,
                //                      ProgressFraction);

                //BestCorrelation.WriteMRC($"d_bestcorr_{b:D2}.mrc", true);
                //BestCorrelationRandom.WriteMRC($"d_bestcorr_random_{b:D2}.mrc", true);

                //BestCorrelation.Subtract(BestCorrelationRandom);

                #endregion

                #region Put correlation values and best angle IDs back into the large volume

                for (int st = 0; st < CurBatch; st++)
                {
                    Image ThisCorrelation = new Image(BestCorrelation.GetDeviceSlice(SizeSub * st, Intent.Read), new int3(SizeSub));
                    Image CroppedCorrelation = ThisCorrelation.AsPadded(new int3(SizeUseful)).AndDisposeParent();

                    Image ThisAngle = new Image(BestAngle.GetDeviceSlice(SizeSub * st, Intent.Read), new int3(SizeSub));
                    Image CroppedAngle = ThisAngle.AsPadded(new int3(SizeUseful)).AndDisposeParent();

                    float[] SubCorr = CroppedCorrelation.GetHostContinuousCopy();
                    float[] SubAngle = CroppedAngle.GetHostContinuousCopy();
                    int3 Origin = new int3(GridCoords[b + st]) - SizeUseful / 2;
                    float Norm = 1f; // / (SizeSub * SizeSub * SizeSub * SizeSub);
                    for (int z = 0; z < SizeUseful; z++)
                    {
                        int zVol = Origin.Z + z;
                        if (zVol >= DimsVolumeScaled.Z - SizeParticle / 2)
                            continue;

                        for (int y = 0; y < SizeUseful; y++)
                        {
                            int yVol = Origin.Y + y;
                            if (yVol >= DimsVolumeScaled.Y - SizeParticle / 2)
                                continue;

                            for (int x = 0; x < SizeUseful; x++)
                            {
                                int xVol = Origin.X + x;
                                if (xVol >= DimsVolumeScaled.X - SizeParticle / 2)
                                    continue;

                                CorrData[zVol][yVol * DimsVolumeScaled.X + xVol] = SubCorr[(z * SizeUseful + y) * SizeUseful + x] * Norm;
                                AngleIDData[zVol][yVol * DimsVolumeScaled.X + xVol] = SubAngle[(z * SizeUseful + y) * SizeUseful + x];
                            }
                        }
                    }

                    CroppedCorrelation.Dispose();
                    CroppedAngle.Dispose();
                }

                #endregion

                Subtomos.Dispose();
                SubtomoCTF.Dispose();

                BestCorrelation.Dispose();
                BestAngle.Dispose();
                //BestCorrelationRandom.Dispose();
                //BestAngleRandom.Dispose();

                ProgressTimer.Dispose();
                if (progressCallback != null)
                    IsCanceled = progressCallback(Grid, b + CurBatch, "Matching...");
            }

            #endregion

            #region Postflight

            GPU.DestroyFFTPlan(PlanForw);
            GPU.DestroyFFTPlan(PlanBack);
            GPU.DestroyFFTPlan(PlanForwCTF);

            CTFCoords.Dispose();
            ProjectorReference.Dispose();
            //ProjectorRandom.Dispose();
            //ProjectorMask.Dispose();

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
                float Median = Center.GetHostContinuousCopy().Median();
                Center.Dispose();

                LocalStd.Max(MathF.Max(1e-10f, Median));

                //LocalStd.WriteMRC("d_localstd.mrc", true);

                CorrVolume.Divide(LocalStd);

                LocalStd.Dispose();
            }

            #endregion

            #region Normalize by background correlation std

            if (options.NormalizeScores)
            {
                Image Center = CorrVolume.AsPadded(CorrVolume.Dims / 2);
                Center.Abs();
                float[] Sorted = Center.GetHostContinuousCopy().OrderBy(v => v).ToArray();
                float Percentile = Sorted[(int)(Sorted.Length * 0.68f)];
                Center.Dispose();

                CorrVolume.Multiply(1f / MathF.Max(1e-20f, Percentile));
            }

            #endregion

            #region Zero out correlation values not fully covered by desired number of tilts

            if (options.MaxMissingTilts >= 0)
            {
                progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Trimming...");

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

                float[][] OccupancyMask = Helper.ArrayOfFunction(z => Helper.ArrayOfConstant(1f, VolumePositions.Length), DimsUndersampled.Z);

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
            }

            #endregion

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Saving global scores...");

            // Store correlation values and angle IDs for re-use later
            CorrVolume.WriteMRC16b(CorrVolumePath, (float)options.BinnedPixelSizeMean, true);
            AngleIDVolume.WriteTIFF(AngleIDVolumePath, (float)options.BinnedPixelSizeMean, typeof(float));

            #endregion
        }
        else
        {
            progressCallback?.Invoke(Grid, 0, "Loading...");

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

        progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Extracting best peaks...");

        int3[] InitialPeaks = new int3[0];
        {
            float Max = CorrVolume.GetHostContinuousCopy().Max();

            for (float s = Max * 0.9f; s > Max * 0.1f; s -= Max * 0.05f)
            {
                float Threshold = s;
                InitialPeaks = CorrVolume.GetLocalPeaks(PeakDistance, Threshold);

                if (InitialPeaks.Length >= options.NResults)
                    break;
            }
        }

        #endregion

        #region Write out images for quickly assessing different thresholds for picking

        progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Preparing visualizations...");

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
        float[] AllPeakScores = new float[InitialPeaks.Count()];
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
            // get positions with score >= thresold
            for (int i = 0; i < InitialPeaks.Count(); i++)
            {
                float3 Position = new float3(InitialPeaks[i]);
                AllPeakScores[i] = CorrData[InitialPeaks[i].Z][InitialPeaks[i].Y * DimsVolumeScaled.X + InitialPeaks[i].X];
            }

            var filteredPositions = InitialPeaks.Zip(AllPeakScores, (position, score) => new { Position = position, Score = score })
                .Where(item => item.Score >= threshold)
                .Where(item => (item.Position.Z >= _ZMin && item.Position.Z <= _ZMax))
                .Select(item => item.Position)
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

        progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Done...");

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
            for (int n = 0; n < InitialPeaks.Length; n++)
            {
                //float3 Position = RefinedPositions[n] / new float3(DimsVolumeCropped);
                //float Score = RefinedScores[n];
                //float3 Angle = RefinedAngles[n] * Helper.ToDeg;

                float3 Position = new float3(InitialPeaks[n]);
                float Score = CorrData[(int)Position.Z][(int)Position.Y * DimsVolumeScaled.X + (int)Position.X];
                float3 Angle = HealpixAngles[(int)(AngleIDData[(int)Position.Z][(int)Position.Y * DimsVolumeScaled.X + (int)Position.X] + 0.5f)] * Helper.ToDeg;
                Position /= new float3(DimsVolumeScaled);

                TableOut.AddRow(new string[]
                {
                    Position.X.ToString(CultureInfo.InvariantCulture),
                    Position.Y.ToString(CultureInfo.InvariantCulture),
                    Position.Z.ToString(CultureInfo.InvariantCulture),
                    Angle.X.ToString(CultureInfo.InvariantCulture),
                    Angle.Y.ToString(CultureInfo.InvariantCulture),
                    Angle.Z.ToString(CultureInfo.InvariantCulture),
                    RootName + ".tomostar",
                    Score.ToString(CultureInfo.InvariantCulture)
                });
            }
        }

        CorrVolume?.Dispose();

        var TableName = string.IsNullOrWhiteSpace(options.OverrideSuffix) ?
                            $"{NameWithRes}_{options.TemplateName}.star" :
                            $"{NameWithRes}{options.OverrideSuffix ?? ""}.star";
        TableOut.Save(System.IO.Path.Combine(MatchingDir, TableName));

        progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Done.");

        #endregion
    }
}

[Serializable]
public class ProcessingOptionsTomoFullMatch : TomoProcessingOptionsBase
{
    [WarpSerializable] public bool OverwriteFiles { get; set; }
    [WarpSerializable] public int SubVolumeSize { get; set; }
    [WarpSerializable] public string TemplateName { get; set; }
    [WarpSerializable] public decimal TemplatePixel { get; set; }
    [WarpSerializable] public decimal TemplateDiameter { get; set; }
    [WarpSerializable] public decimal PeakDistance { get; set; }
    [WarpSerializable] public decimal TemplateFraction { get; set; }
    [WarpSerializable] public int MaxMissingTilts { get; set; }
    [WarpSerializable] public bool WhitenSpectrum { get; set; }
    [WarpSerializable] public decimal Lowpass { get; set; }
    [WarpSerializable] public decimal LowpassSigma { get; set; }
    [WarpSerializable] public string Symmetry { get; set; }
    [WarpSerializable] public int HealpixOrder { get; set; }
    [WarpSerializable] public decimal TiltRange { get; set; }
    [WarpSerializable] public int BatchAngles { get; set; }
    [WarpSerializable] public int Supersample { get; set; }
    [WarpSerializable] public int NResults { get; set; }
    [WarpSerializable] public bool NormalizeScores { get; set; }
    [WarpSerializable] public bool ReuseCorrVolumes { get; set; }
    [WarpSerializable] public string OverrideSuffix { get; set; }
}