using Accord.Math.Optimization;
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
    public void PeakAlign(ProcessingOptionsTomoPeakAlign options, Image template, float3[] positions, float3[] angles)
    {
        #region Dimensions

        VolumeDimensionsPhysical = options.DimensionsPhysical;

        int SizeRegion = (int)(template.Dims.X * (float)options.TemplatePixel / (float)options.BinnedPixelSizeMean / 2 + 1) * 2;

        int NParticles = positions.Length / NTilts;

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
            TiltData[z].Bandpass(1f / (SizeRegion / 2), 1f, false, 1f / (SizeRegion / 2));
            TiltData[z] = TiltData[z].AsPadded(new int2(TiltData[z].Dims) / 2).AndDisposeParent();
            //TiltData[z].WriteMRC("d_tiltdatabp.mrc", true);

            if (options.Invert)
                TiltData[z].Multiply(-1f);

            if (options.Normalize)
                TiltData[z].Normalize();
        }
        GPU.CheckGPUExceptions();

        int2 DimsImage = new int2(TiltData[0].Dims);

        #endregion

        #region Projector

        if (template.Dims.X != SizeRegion)
            template = template.AsScaled(new int3(SizeRegion)).AndDisposeParent();

        Projector Projector = new Projector(template, 2);
        GPU.CheckGPUExceptions();

        #endregion

        #region Memory and FFT plan allocation

        int PlanForwParticles = GPU.CreateFFTPlan(new int3(SizeRegion, SizeRegion, 1), (uint)NParticles);
        int PlanBackParticles = GPU.CreateIFFTPlan(new int3(SizeRegion, SizeRegion, 1), (uint)NParticles);

        Image Images = new(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles));
        Image ImagesFT = new(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles), true, true);

        Image CTFCoords = CTF.GetCTFCoords(SizeRegion, SizeRegion, Matrix2.Identity());
        Image CTFs = new Image(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles), true);
        GPU.CheckGPUExceptions();

        #endregion

        var OptimizeParticles = (bool optimizeTilts) =>
        {
            float3[] PositionsOri = positions.ToArray();
            float3[] AnglesOri = angles.ToArray();
            float[] MovementXOri = GridMovementX?.Values.ToArray();
            float[] MovementYOri = GridMovementY?.Values.ToArray();

            int NParamsParticles = NParticles * 6;
            int NParamsTilts = optimizeTilts ? NTilts * 2 : 0;

            double BestScore = double.NegativeInfinity;
            double[] BestInput = null;

            var SetPositions = (double[] input) =>
            {
                for (int p = 0; p < NParticles; p++)
                    for (int t = 0; t < NTilts; t++)
                    {
                        positions[p * NTilts + t].X = (float)input[p * 6 + 0] + PositionsOri[p * NTilts + t].X;
                        positions[p * NTilts + t].Y = (float)input[p * 6 + 1] + PositionsOri[p * NTilts + t].Y;
                        positions[p * NTilts + t].Z = (float)input[p * 6 + 2] + PositionsOri[p * NTilts + t].Z;

                        var UpdatedRotation = Matrix3.Euler(AnglesOri[p * NTilts + t] * Helper.ToRad) * 
                                              Matrix3.RotateX((float)input[p * 6 + 3] * Helper.ToRad) *
                                              Matrix3.RotateY((float)input[p * 6 + 4] * Helper.ToRad) *
                                              Matrix3.RotateZ((float)input[p * 6 + 5] * Helper.ToRad);
                        var UpdatedAngles = Matrix3.EulerFromMatrix(UpdatedRotation) * Helper.ToDeg;

                        angles[p * NTilts + t].X = UpdatedAngles.X;
                        angles[p * NTilts + t].Y = UpdatedAngles.Y;
                        angles[p * NTilts + t].Z = UpdatedAngles.Z;
                    }
            };

            var SetTilts = (double[] input) =>
            {
                if (optimizeTilts)
                {
                    input = input.Skip(NParamsParticles).ToArray();

                    GridMovementX = new CubicGrid(new int3(1, 1, NTilts), 
                                                  Enumerable.Range(0, NTilts)
                                                            .Select(i => (float)input[i * 2 + 0] + MovementXOri[i])
                                                            .ToArray());
                    GridMovementY = new CubicGrid(new int3(1, 1, NTilts),
                                                  Enumerable.Range(0, NTilts)
                                                            .Select(i => (float)input[i * 2 + 1] + MovementYOri[i])
                                                            .ToArray());
                }
            };

            Func<double[], double[]> EvalParticles = (input) =>
            {
                SetPositions(input);
                SetTilts(input);

                double[] Result = new double[NParticles];

                for (int t = 0; t < NTilts; t++)
                {
                    float3[] ParticlePositions = Enumerable.Range(0, NParticles).Select(p => positions[p * NTilts + t]).ToArray();
                    float3[] ParticleAngles = Enumerable.Range(0, NParticles).Select(p => angles[p * NTilts + t]).ToArray();

                    float3[] ParticlePositionsInImage = GetPositionsInOneTilt(ParticlePositions, t);
                    float3[] ParticleAnglesInImage = GetAnglesInOneTilt(ParticlePositions, ParticleAngles, t);

                    ImagesFT = GetParticleImagesFromOneTilt(options,
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

                    GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                      ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                                      ParticlePositionsInImage,
                                      CTFCoords,
                                      null,
                                      t,
                                      CTFs);
                    References.Multiply(CTFs);
                    GPU.CheckGPUExceptions();

                    GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                      ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                                      ParticlePositionsInImage,
                                      CTFCoords,
                                      null,
                                      t,
                                      CTFs,
                                      weighted: true,
                                      weightsonly: true);
                    ImagesFT.Multiply(CTFs);

                    GPU.IFFT(ImagesFT.GetDevice(Intent.ReadWrite), 
                             Images.GetDevice(Intent.Write), 
                             new int3(SizeRegion).Slice(), 
                             (uint)NParticles, 
                             PlanBackParticles, 
                             normalize: false);
                    Images.Normalize();
                    Images.MaskSpherically((float)(options.TemplateDiameter / options.BinnedPixelSizeMean), 
                                              (float)(40 / options.BinnedPixelSizeMean), 
                                              false,
                                              true);
                    Images.Normalize();

                    using Image ReferencesIFT = References.AsIFFT(false, PlanBackParticles);
                    ReferencesIFT.Normalize();

                    Images.Multiply(ReferencesIFT);
                    var Sums = Images.AsSum2D();
                    var SumsData = Sums.GetHostContinuousCopy();
                    for (int p = 0; p < NParticles; p++)
                        Result[p] += SumsData[p];
                }

                return Result.Select(v => v / (SizeRegion * SizeRegion * NTilts)).ToArray();
            };

            Func<double[], double[]> EvalTilts = (input) =>
            {
                SetPositions(input);
                SetTilts(input);

                double[] Result = new double[NTilts];

                for (int t = 0; t < NTilts; t++)
                {
                    float3[] ParticlePositions = Enumerable.Range(0, NParticles).Select(p => positions[p * NTilts + t]).ToArray();
                    float3[] ParticleAngles = Enumerable.Range(0, NParticles).Select(p => angles[p * NTilts + t]).ToArray();

                    float3[] ParticlePositionsInImage = GetPositionsInOneTilt(ParticlePositions, t);
                    float3[] ParticleAnglesInImage = GetAnglesInOneTilt(ParticlePositions, ParticleAngles, t);

                    ImagesFT = GetParticleImagesFromOneTilt(options,
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

                    GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                      ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                                      ParticlePositionsInImage,
                                      CTFCoords,
                                      null,
                                      t,
                                      CTFs);
                    References.Multiply(CTFs);
                    GPU.CheckGPUExceptions();

                    GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                      ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                                      ParticlePositionsInImage,
                                      CTFCoords,
                                      null,
                                      t,
                                      CTFs,
                                      weighted: true,
                                      weightsonly: true);
                    ImagesFT.Multiply(CTFs);

                    GPU.IFFT(ImagesFT.GetDevice(Intent.ReadWrite),
                             Images.GetDevice(Intent.Write),
                             new int3(SizeRegion).Slice(),
                             (uint)NParticles,
                             PlanBackParticles,
                             normalize: false);
                    Images.Normalize();
                    Images.MaskSpherically((float)(options.TemplateDiameter / options.BinnedPixelSizeMean),
                                              (float)(40 / options.BinnedPixelSizeMean),
                                              false,
                                              true);
                    Images.Normalize();

                    using Image ReferencesIFT = References.AsIFFT(false, PlanBackParticles);
                    ReferencesIFT.Normalize();

                    Images.Multiply(ReferencesIFT);
                    var Sums = Images.AsSum3D();
                    var SumsData = Sums.GetHostContinuousCopy();
                    Result[t] = SumsData[0];
                }

                return Result.Select(v => v / (SizeRegion * SizeRegion * NTilts)).ToArray();
            };

            Func<double[], double> Eval = (input) =>
            {
                double[] Indiv = EvalParticles(input);
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

                if (++OptIterations > 15)
                    return Result;

                for (int icomp = 0; icomp < 6; icomp++)
                {
                    double[] InputPlus = input.ToArray();
                    for (int p = 0; p < NParticles; p++)
                        InputPlus[p * 6 + icomp] += Delta;
                    double[] EvalPlus = EvalParticles(InputPlus);

                    double[] InputMinus = input.ToArray();
                    for (int p = 0; p < NParticles; p++)
                        InputMinus[p * 6 + icomp] -= Delta;
                    double[] EvalMinus = EvalParticles(InputMinus);

                    for (int p = 0; p < NParticles; p++)
                        Result[p * 6 + icomp] = (EvalPlus[p] - EvalMinus[p]) / (2 * Delta);
                }

                if (optimizeTilts)
                {
                    for (int icomp = 0; icomp < 2; icomp++)
                    {
                        double[] InputPlus = input.ToArray();
                        for (int t = 0; t < NTilts; t++)
                            InputPlus[NParamsParticles + t * 2 + icomp] += Delta;
                        double[] EvalPlus = EvalTilts(InputPlus);

                        double[] InputMinus = input.ToArray();
                        for (int t = 0; t < NTilts; t++)
                            InputMinus[NParamsParticles + t * 2 + icomp] -= Delta;
                        double[] EvalMinus = EvalTilts(InputMinus);

                        for (int t = 0; t < NTilts; t++)
                            Result[NParamsParticles + t * 2 + icomp] = (EvalPlus[t] - EvalMinus[t]) / (2 * Delta);
                    }
                }

                return Result;
            };

            double[] StartParams = new double[NParamsParticles + NParamsTilts];
            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
            Optimizer.Maximize(StartParams);

            SetPositions(BestInput);
            
            var Shifts = new float3[NParticles];
            for (int p = 0; p < NParticles; p++)
                Shifts[p] = new float3((float)BestInput[p * NParamsParticles + 0],
                                       (float)BestInput[p * NParamsParticles + 1],
                                       (float)BestInput[p * NParamsParticles + 2]);
            float ShiftRMS = MathF.Sqrt(Shifts.Select(v => v.LengthSq()).Average());
            Console.WriteLine($"Particle shift RMS: {ShiftRMS.ToString("F2", CultureInfo.InvariantCulture)} A");
        };

        var OptimizeTilts = () =>
        {
            Image TiltPeaks = new(new int3(SizeRegion, SizeRegion, NTilts));

            for (int t = 0; t < NTilts; t++)
            {
                float3[] ParticlePositions = Enumerable.Range(0, NParticles).Select(p => positions[p * NTilts + t]).ToArray();
                float3[] ParticleAngles = Enumerable.Range(0, NParticles).Select(p => angles[p * NTilts + t]).ToArray();

                float3[] ParticlePositionsInImage = GetPositionsInOneTilt(ParticlePositions, t);
                float3[] ParticleAnglesInImage = GetAnglesInOneTilt(ParticlePositions, ParticleAngles, t);

                ImagesFT = GetParticleImagesFromOneTilt(options,
                                                        TiltData,
                                                        t,
                                                        SizeRegion,
                                                        ParticlePositions,
                                                        PlanForwParticles,
                                                        false,
                                                        Images,
                                                        ImagesFT);
                GPU.CheckGPUExceptions();

                using Image References = Projector.Project(new int2(SizeRegion), ParticleAnglesInImage);
                GPU.CheckGPUExceptions();

                GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                  ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                                  ParticlePositionsInImage,
                                  CTFCoords,
                                  null,
                                  t,
                                  CTFs);
                References.Multiply(CTFs);
                GPU.CheckGPUExceptions();

                ImagesFT.MultiplyConj(References);
                GPU.CheckGPUExceptions();

                GPU.IFFT(ImagesFT.GetDevice(Intent.Read), Images.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)NParticles, PlanBackParticles, normalize: false);
                GPU.CheckGPUExceptions();

                using Image Average = Images.AsReducedAlongZ();
                GPU.CheckGPUExceptions();
                GPU.CopyDeviceToHost(Average.GetDevice(Intent.Read), TiltPeaks.GetHost(Intent.Write)[t], Average.ElementsReal);
            }

            TiltPeaks.WriteMRC($"d_tiltpeaks_{RootName}.mrc", true);

            int SubpixelFactor = 10;
            TiltPeaks = TiltPeaks.AsScaled(new int2(TiltPeaks.Dims) * SubpixelFactor).AndDisposeParent();
            float2[] Corrections = new float2[NTilts];

            for (int t = 0; t < NTilts; t++)
            {
                float[] PeakData = TiltPeaks.GetHost(Intent.Read)[t];
                int MaxValueIndex = 0;
                float MaxValue = float.MinValue;
                for (int i = 0; i < PeakData.Length; i++)
                    if (PeakData[i] > MaxValue)
                    {
                        MaxValue = PeakData[i];
                        MaxValueIndex = i;
                    }

                float2 PeakPos = new(MaxValueIndex % TiltPeaks.Dims.X, MaxValueIndex / TiltPeaks.Dims.X);
                PeakPos -= new float2(TiltPeaks.Dims.X, TiltPeaks.Dims.Y) / 2;

                Corrections[t] = -PeakPos * ((float)options.BinnedPixelSizeMean / SubpixelFactor);
            }
            
            Console.WriteLine("Tilt shift corrections (X, Y):");
            for (int t = 0; t < NTilts; t++)
                Console.WriteLine($"Tilt {t}: " +
                                  $"{Corrections[t].X.ToString("F2", CultureInfo.InvariantCulture)}, " +
                                  $"{Corrections[t].Y.ToString("F2", CultureInfo.InvariantCulture)} A");

            if (GridMovementX != null)
                GridMovementX = GridMovementX.Resize(new int3(1, 1, NTilts));
            else
                GridMovementX = new CubicGrid(new int3(1, 1, NTilts));

            if (GridMovementY != null)
                GridMovementY = GridMovementY.Resize(new int3(1, 1, NTilts));
            else
                GridMovementY = new CubicGrid(new int3(1, 1, NTilts));

            GridMovementX = new CubicGrid(new int3(1, 1, NTilts), Corrections.Select((v, i) => v.X + GridMovementX.Values[i]).ToArray());
            GridMovementY = new CubicGrid(new int3(1, 1, NTilts), Corrections.Select((v, i) => v.Y + GridMovementY.Values[i]).ToArray());

            #region Teardown

            TiltPeaks.Dispose();

            #endregion
        };

        if (options.OptimizeParticlePoses)
            OptimizeParticles(false);
        OptimizeTilts();

        //OptimizeParticles(true);

        #region Teardown

        Images.Dispose();
        ImagesFT.Dispose();
        CTFs.Dispose();
        CTFCoords.Dispose();

        GPU.DestroyFFTPlan(PlanForwParticles);
        GPU.DestroyFFTPlan(PlanBackParticles);

        Projector.Dispose();
        foreach (var img in TiltData)
            img.Dispose();
        foreach (var img in TiltMasks)
            img?.Dispose();

        GPU.CheckGPUExceptions();

        #endregion
    }
}

[Serializable]
public class ProcessingOptionsTomoPeakAlign : TomoProcessingOptionsBase
{
    [WarpSerializable] public bool Invert { get; set; }
    [WarpSerializable] public bool Normalize { get; set; }
    [WarpSerializable] public decimal TemplatePixel { get; set; }
    [WarpSerializable] public decimal TemplateDiameter { get; set; }
    [WarpSerializable] public bool OptimizeParticlePoses { get; set; }
    [WarpSerializable] public bool WhitenSpectrum { get; set; }
    [WarpSerializable] public decimal Lowpass { get; set; }
    [WarpSerializable] public decimal LowpassSigma { get; set; }
}