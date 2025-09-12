using System;
using System.Collections.Generic;
using System.Linq;
using Accord.Math.Optimization;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class TiltSeries
{
    public void AlignLocallyWithoutReferences(ProcessingOptionsTomoFullReconstruction options)
    {
        VolumeDimensionsPhysical = options.DimensionsPhysical;
        int SizeRegion = options.SubVolumeSize;

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
            TiltData[z] = TiltData[z].AsPaddedClamped(new int2(TiltData[z].Dims) * 2).AndDisposeParent();
            TiltData[z].MaskRectangularly((TiltData[z].Dims / 2).Slice(), MathF.Min(TiltData[z].Dims.X / 4, TiltData[z].Dims.Y / 4), false);
            //TiltData[z].WriteMRC("d_tiltdata.mrc", true);
            TiltData[z].Bandpass(1f / (SizeRegion / 2), 1f, false, 1f / (SizeRegion / 2));
            TiltData[z] = TiltData[z].AsPadded(new int2(TiltData[z].Dims) / 2).AndDisposeParent();
            //TiltData[z].WriteMRC("d_tiltdatabp.mrc", true);

            GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                TiltData[z].GetDevice(Intent.Write),
                (uint)TiltData[z].ElementsReal,
                1);

            if (options.Invert)
                TiltData[z].Multiply(-1f);

            TiltData[z].FreeDevice();
        }

        int2 DimsImage = new int2(TiltData[0].Dims);
        int SizeReconstruction = Math.Max(DimsImage.X, DimsImage.Y);
        int SizeReconstructionPadded = SizeReconstruction * 2;

        #endregion

        int2 DimsPositionGrid;
        int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage - 64,
            new int2(SizeRegion),
            0.5f,
            out DimsPositionGrid).Select(v => new int3(v.X + 32 + SizeRegion / 2,
            v.Y + 32 + SizeRegion / 2,
            0)).ToArray();
        float3[] PositionGridPhysical = PositionGrid.Select(v => new float3(v.X * (float)options.BinnedPixelSizeMean,
            v.Y * (float)options.BinnedPixelSizeMean,
            VolumeDimensionsPhysical.Z / 2)).ToArray();

        Image RegionMask = new Image(new int3(SizeRegion, SizeRegion, 1));
        RegionMask.Fill(1);
        RegionMask.MaskRectangularly(new int3(SizeRegion / 2, SizeRegion / 2, 1), SizeRegion / 4, false);
        RegionMask.WriteMRC("d_mask.mrc", true);

        GridMovementX = new CubicGrid(new int3(1, 1, NTilts));
        GridMovementY = new CubicGrid(new int3(1, 1, NTilts));

        GridVolumeWarpX = new LinearGrid4D(new int4(1));
        GridVolumeWarpY = new LinearGrid4D(new int4(1));
        GridVolumeWarpZ = new LinearGrid4D(new int4(1));

        Image Extracted1 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length)),
            Extracted2 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length)),
            Extracted3 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length));
        Image ExtractedFT1 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length), true, true),
            ExtractedFT2 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length), true, true),
            ExtractedFT3 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length), true, true);
        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length);
        int PlanBack = GPU.CreateIFFTPlan(new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length);

        Image TiltDataFT = new Image(Helper.Combine(TiltData.Select(v => v.GetHost(Intent.Read)[0]).ToArray()), new int3(DimsImage.X, DimsImage.Y, NTilts)).AsFFT().AndDisposeParent();
        TiltDataFT.Multiply(1f / (DimsImage.Elements()));
        Image TiltDataFTFiltered = TiltDataFT.GetCopyGPU();
        Image TiltDataFiltered = new Image(new int3(DimsImage.X, DimsImage.Y, NTilts));
        int PlanBackTiltData = GPU.CreateIFFTPlan(TiltDataFiltered.Dims.Slice(), (uint)NTilts);

        Projector ProjectorCommonLine;
        {
            Image ZeroTilt = new Image(new int3(DimsImage.X));
            ZeroTilt.Fill(1);
            ZeroTilt.MaskRectangularly(new int3(1, 1, SizeReconstruction), 0, true);
            ZeroTilt.MaskRectangularly(new int3(SizeReconstruction - 32, SizeReconstruction - 32, SizeReconstruction / 4), 16, true);

            ProjectorCommonLine = new Projector(ZeroTilt, 1);
            //Image Slice = ProjectorCommonLine.Project(new int2(SizeReconstruction), new[] { new float3(0, 3, -TiltAxisAngles[0]) * Helper.ToRad });
            //Slice.AsReal().WriteMRC("d_slicetest.mrc");
        }

        // Figure out global tilt angle offset
        if (true)
        {
            float[] OriAngles = Angles.ToArray();
            float[] OriAxisAngles = TiltAxisAngles.ToArray();

            Action<double[]> SetAngles = (input) =>
            {
                for (int t = 0; t < NTilts; t++)
                    Angles[t] = OriAngles[t] + (float)input[0];

                for (int t = 0; t < NTilts; t++)
                    TiltAxisAngles[t] = 84.05f; // OriAxisAngles[t] + (float)input[1];
            };

            Func<double[], double> Eval = (input) =>
            {
                SetAngles(input);

                double Result = 0;
                bool FromScratch = true;

                Image CommonLines = ProjectorCommonLine.Project(new int2(SizeReconstruction), TiltAxisAngles.Select(a => new float3(0, 3, -a) * Helper.ToRad).ToArray());
                Image CommonLinesReal = CommonLines.AsReal().AndDisposeParent();
                GPU.MultiplyComplexSlicesByScalar(TiltDataFT.GetDevice(Intent.Read),
                    CommonLinesReal.GetDevice(Intent.Read),
                    TiltDataFTFiltered.GetDevice(Intent.Write),
                    TiltDataFT.ElementsSliceComplex,
                    (uint)TiltDataFT.Dims.Z);
                GPU.IFFT(TiltDataFTFiltered.GetDevice(Intent.Read),
                    TiltDataFiltered.GetDevice(Intent.Write),
                    TiltDataFiltered.Dims.Slice(),
                    (uint)TiltDataFiltered.Dims.Z,
                    PlanBackTiltData,
                    false);
                TiltDataFiltered.Normalize();
                CommonLinesReal.Dispose();
                //TiltDataFiltered.WriteMRC("d_tiltdatafiltered.mrc", true);

                for (int t = NTilts / 2 - 6; t <= NTilts / 2 + 6; t++)
                {
                    if (FromScratch)
                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t - 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        GPU.Extract(TiltDataFiltered.GetDeviceSlice(t - 1, Intent.Read),
                            Extracted1.GetDevice(Intent.Write),
                            TiltData[t - 1].Dims,
                            Extracted1.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Extracted1.GetDevice(Intent.Read), ExtractedFT1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT1.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT1.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                    }
                    else
                    {
                        GPU.CopyDeviceToDevice(Extracted2.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), Extracted1.ElementsReal);
                    }

                    if (FromScratch)
                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        GPU.Extract(TiltDataFiltered.GetDeviceSlice(t, Intent.Read),
                            Extracted2.GetDevice(Intent.Write),
                            TiltData[t].Dims,
                            Extracted2.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Extracted2.GetDevice(Intent.Read), ExtractedFT2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT2.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT2.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                    }
                    else
                    {
                        GPU.CopyDeviceToDevice(Extracted3.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), Extracted2.ElementsReal);
                    }

                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t + 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        GPU.Extract(TiltDataFiltered.GetDeviceSlice(t + 1, Intent.Read),
                            Extracted3.GetDevice(Intent.Write),
                            TiltData[t + 1].Dims,
                            Extracted1.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Extracted3.GetDevice(Intent.Read), ExtractedFT3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT3.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT3.GetDevice(Intent.Read), Extracted3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                    }

                    Extracted1.Add(Extracted3);
                    Extracted1.Multiply(0.5f);

                    Extracted1.Multiply(Extracted2);
                    Extracted1.MultiplySlices(RegionMask);

                    Image Diff = Extracted1.AsSum3D();
                    Result += Diff.GetHost(Intent.Read)[0][0] * MathF.Pow(MathF.Cos(Angles[t] * Helper.ToRad), 1);
                    Diff.Dispose();

                    FromScratch = false;
                }

                return Result;
            };

            int OptIterations = 0;
            Func<double[], double[]> Grad = (input) =>
            {
                double Delta = 0.1;
                double[] Result = new double[input.Length];

                if (OptIterations++ > 12)
                    return Result;

                for (int i = 0; i < input.Length - 1; i++)
                {
                    double[] InputPlus = input.ToArray();
                    InputPlus[i] += Delta;
                    double ScorePlus = Eval(InputPlus);

                    double[] InputMinus = input.ToArray();
                    InputMinus[i] -= Delta;
                    double ScoreMinus = Eval(InputMinus);

                    Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                }

                Console.WriteLine(Eval(input));

                return Result;
            };

            double[] StartParams = new double[2];
            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
            Optimizer.Maximize(StartParams);
        }

        // Patch Z position against raw patch CC
        if (false)
        {
            GridVolumeWarpX = new LinearGrid4D(new int4(1));
            GridVolumeWarpY = new LinearGrid4D(new int4(1));
            GridVolumeWarpZ = new LinearGrid4D(new int4(1));

            CubicGrid GridPatchZ = new CubicGrid(new int3(5, 5, 1));

            float[] OriWarping = Helper.ArrayOfConstant(VolumeDimensionsPhysical.Z / 2, GridPatchZ.Values.Length);

            Action<double[]> SetWarping = (input) =>
            {
                float Mean = input.Select(v => (float)v).Average();
                GridPatchZ = new CubicGrid(GridPatchZ.Dimensions, input.Select((v, i) => OriWarping[i] + (float)v - Mean).ToArray());

                float3[] InterpCoords = PositionGridPhysical.Select(v => new float3(v.X / VolumeDimensionsPhysical.X, v.Y / VolumeDimensionsPhysical.Y, 0.5f)).ToArray();
                float[] InterpVals = GridPatchZ.GetInterpolated(InterpCoords);

                for (int i = 0; i < PositionGridPhysical.Length; i++)
                    PositionGridPhysical[i].Z = InterpVals[i];
            };

            Func<double[], double> Eval = (input) =>
            {
                SetWarping(input);

                double Result = 0;
                bool FromScratch = true;

                for (int t = 1; t < NTilts - 1; t++)
                {
                    if (FromScratch)
                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t - 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        GPU.Extract(TiltDataFiltered.GetDeviceSlice(t - 1, Intent.Read),
                            Extracted1.GetDevice(Intent.Write),
                            TiltData[t - 1].Dims,
                            Extracted1.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Extracted1.GetDevice(Intent.Read), ExtractedFT1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT1.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT1.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                    }
                    else
                    {
                        GPU.CopyDeviceToDevice(Extracted2.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), Extracted1.ElementsReal);
                    }

                    if (FromScratch)
                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        GPU.Extract(TiltDataFiltered.GetDeviceSlice(t, Intent.Read),
                            Extracted2.GetDevice(Intent.Write),
                            TiltData[t].Dims,
                            Extracted2.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Extracted2.GetDevice(Intent.Read), ExtractedFT2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT2.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT2.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                    }
                    else
                    {
                        GPU.CopyDeviceToDevice(Extracted3.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), Extracted2.ElementsReal);
                    }

                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t + 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        GPU.Extract(TiltDataFiltered.GetDeviceSlice(t + 1, Intent.Read),
                            Extracted3.GetDevice(Intent.Write),
                            TiltData[t + 1].Dims,
                            Extracted1.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Extracted3.GetDevice(Intent.Read), ExtractedFT3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT3.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT3.GetDevice(Intent.Read), Extracted3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                    }

                    Extracted1.Add(Extracted3);
                    Extracted1.Multiply(0.5f);

                    Extracted1.Multiply(Extracted2);
                    Extracted1.MultiplySlices(RegionMask);

                    Image Diff = Extracted1.AsSum3D();
                    Result += Diff.GetHost(Intent.Read)[0][0] * MathF.Pow(MathF.Cos(Angles[t] * Helper.ToRad), 1);
                    Diff.Dispose();

                    FromScratch = false;
                }

                return Result;
            };

            int OptIterations = 0;
            Func<double[], double[]> Grad = (input) =>
            {
                double Delta = 0.1;
                double[] Result = new double[input.Length];

                if (OptIterations++ > 12)
                    return Result;

                for (int i = 0; i < input.Length; i++)
                {
                    double[] InputPlus = input.ToArray();
                    InputPlus[i] += Delta;
                    double ScorePlus = Eval(InputPlus);

                    double[] InputMinus = input.ToArray();
                    InputMinus[i] -= Delta;
                    double ScoreMinus = Eval(InputMinus);

                    Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                }

                Console.WriteLine(Eval(input));

                return Result;
            };

            double[] StartParams = new double[OriWarping.Length];
            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
            Optimizer.Maximize(StartParams);

            SetWarping(StartParams);
            Console.WriteLine(Eval(StartParams));

            new Image(PositionGridPhysical.Select(v => v.Z).ToArray(), new int3((int)Math.Sqrt(PositionGridPhysical.Length)).Slice()).WriteMRC("d_heightfield.mrc", true);
        }

        // Volume warp grid against raw patch CC
        if (false)
        {
            GridVolumeWarpX = new LinearGrid4D(new int4(3, 3, 1, 2));
            GridVolumeWarpY = new LinearGrid4D(GridVolumeWarpX.Dimensions);
            GridVolumeWarpZ = new LinearGrid4D(GridVolumeWarpX.Dimensions);

            float[] OriWarping = GridVolumeWarpZ.Values.Skip((int)GridVolumeWarpZ.Dimensions.ElementsSlice()).ToArray();

            Action<double[]> SetWarping = (input) =>
            {
                float[] NewValues = new float[GridVolumeWarpZ.Values.Length];
                float Mean = input.Select(v => (float)v).Average();
                for (int i = 0; i < GridVolumeWarpZ.Dimensions.Elements() - GridVolumeWarpZ.Dimensions.ElementsSlice(); i++)
                    NewValues[GridVolumeWarpZ.Dimensions.ElementsSlice() + i] = (float)input[i] - Mean;

                GridVolumeWarpZ = new LinearGrid4D(GridVolumeWarpZ.Dimensions, NewValues);
            };

            Func<double[], double> Eval = (input) =>
            {
                SetWarping(input);

                double Result = 0;
                bool FromScratch = true;

                for (int t = 1; t < NTilts - 1; t++)
                {
                    if (FromScratch)
                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t - 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        GPU.Extract(TiltData[t - 1].GetDevice(Intent.Read),
                            Extracted1.GetDevice(Intent.Write),
                            TiltData[t - 1].Dims,
                            Extracted1.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Extracted1.GetDevice(Intent.Read), ExtractedFT1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT1.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT1.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                    }
                    else
                    {
                        GPU.CopyDeviceToDevice(Extracted2.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), Extracted1.ElementsReal);
                    }

                    if (FromScratch)
                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                            Extracted2.GetDevice(Intent.Write),
                            TiltData[t].Dims,
                            Extracted2.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Extracted2.GetDevice(Intent.Read), ExtractedFT2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT2.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT2.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                    }
                    else
                    {
                        GPU.CopyDeviceToDevice(Extracted3.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), Extracted2.ElementsReal);
                    }

                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t + 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        GPU.Extract(TiltData[t + 1].GetDevice(Intent.Read),
                            Extracted3.GetDevice(Intent.Write),
                            TiltData[t + 1].Dims,
                            Extracted1.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Extracted3.GetDevice(Intent.Read), ExtractedFT3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT3.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT3.GetDevice(Intent.Read), Extracted3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                    }

                    Extracted1.Add(Extracted3);
                    Extracted1.Multiply(0.5f);

                    Extracted1.Multiply(Extracted2);
                    Extracted1.MultiplySlices(RegionMask);

                    Image Diff = Extracted1.AsSum3D();
                    Result += Diff.GetHost(Intent.Read)[0][0] * MathF.Pow(MathF.Cos(Angles[t] * Helper.ToRad), 1);
                    Diff.Dispose();

                    FromScratch = false;
                }

                return Result;
            };

            int OptIterations = 0;
            Func<double[], double[]> Grad = (input) =>
            {
                double Delta = 0.1;
                double[] Result = new double[input.Length];

                if (OptIterations++ > 12)
                    return Result;

                for (int i = 0; i < input.Length; i++)
                {
                    double[] InputPlus = input.ToArray();
                    InputPlus[i] += Delta;
                    double ScorePlus = Eval(InputPlus);

                    double[] InputMinus = input.ToArray();
                    InputMinus[i] -= Delta;
                    double ScoreMinus = Eval(InputMinus);

                    Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                }

                Console.WriteLine(Eval(input));

                return Result;
            };

            double[] StartParams = new double[OriWarping.Length];
            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
            Optimizer.Maximize(StartParams);

            SetWarping(StartParams);
        }

        // In-plane shift alignment
        if (true)
        {
            //CubicGrid GridWarp = new CubicGrid(new int3(3, 3, 1), new float[] { 0, 0, 0, 0, 10, 0, 0, 0, 0 });
            //{
            //    float3[] Coords = new float3[49];
            //    for (int y = 0; y < 7; y++)
            //        for (int x = 0; x < 7; x++)
            //            Coords[y * 7 + x] = new float3(x / 6f, y / 6f, 0);
            //    float[] Interpolated = GridWarp.GetInterpolated(Coords);

            //    //GridVolumeWarpZ = new LinearGrid4D(new int4(7, 7, 1, 1), Interpolated);
            //    GridVolumeWarpZ = new LinearGrid4D(new int4(1, 1, 1, 1), new[] { 10f });

            //    List<float2[]> TargetWarp = new List<float2[]>();
            //    for (int t = 0; t < NTilts; t++)
            //        TargetWarp.Add(GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => new float2(v.X, v.Y)).ToArray());

            //    float2 OffsetWarped = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, 0).Select(v => new float2(v.X, v.Y)).First();

            //    GridVolumeWarpZ = new LinearGrid4D(GridVolumeWarpX.Dimensions);

            //    float2 OffsetDefault = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, 0).Select(v => new float2(v.X, v.Y)).First();
            //    float2 Relative = OffsetWarped - OffsetDefault;

            //    Matrix3 TiltMatrix = Matrix3.Euler(0, Angles[0] * Helper.ToRad, -TiltAxisAngles[0] * Helper.ToRad);
            //    float3 Transformed = TiltMatrix * new float3(0, 0, 10);

            //    GridMovementX = new CubicGrid(new int3(1), new[] { -Transformed.X });
            //    GridMovementY = new CubicGrid(new int3(1), new[] { -Transformed.Y });
            //    OffsetWarped = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, 0).Select(v => new float2(v.X, v.Y)).First();

            //    GridMovementX = new CubicGrid(new int3(1), new[] { 0f });
            //    GridMovementY = new CubicGrid(new int3(1), new[] { 0f });
            //    OffsetDefault = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, 0).Select(v => new float2(v.X, v.Y)).First();

            //    Relative = OffsetWarped - OffsetDefault;
            //    //double[] StartParams

            //    //for (int t = 0; t < NTilts; t++)
            //    //{

            //    //}

            //}
            {
            }

            List<Image> CorrectedTilts = new List<Image>
            {
                TiltData[IndicesSortedAbsoluteAngle[0]]
            };
            List<int> TiltsProcessed = new List<int>()
            {
                IndicesSortedAbsoluteAngle[0]
            };

            List<double> Scores = new List<double>();

            var FindClosestProcessedTilt = (int currentTilt) =>
            {
                int Closest = TiltsProcessed.First();
                float ClosestDist = MathF.Abs(Angles[currentTilt] - Angles[Closest]);

                for (int i = 1; i < TiltsProcessed.Count; i++)
                {
                    float Dist = MathF.Abs(Angles[currentTilt] - Angles[TiltsProcessed[i]]);
                    if (Dist < ClosestDist)
                    {
                        Closest = TiltsProcessed[i];
                        ClosestDist = Dist;
                    }
                }

                return Closest;
            };

            float[] FinalShiftsX = new float[GridMovementX.Values.Length];
            float[] FinalShiftsY = new float[GridMovementY.Values.Length];

            var MakeWarpedImage = (int t, Image warped) =>
            {
                int2 DimsWarp = new int2(16);
                float StepZ = 1f / Math.Max(NTilts - 1, 1);

                float3[] InterpPoints = new float3[DimsWarp.Elements()];
                for (int y = 0; y < DimsWarp.Y; y++)
                for (int x = 0; x < DimsWarp.X; x++)
                    InterpPoints[y * DimsWarp.X + x] = new float3((float)x / (DimsWarp.X - 1), (float)y / (DimsWarp.Y - 1), t * StepZ);

                float2[] WarpXY = Helper.Zip(GridMovementX.GetInterpolated(InterpPoints), GridMovementY.GetInterpolated(InterpPoints));
                float[] WarpX = WarpXY.Select(v => v.X / (float)options.BinnedPixelSizeMean).ToArray();
                float[] WarpY = WarpXY.Select(v => v.Y / (float)options.BinnedPixelSizeMean).ToArray();

                Image TiltImagePrefiltered = TiltData[t].GetCopyGPU();
                GPU.PrefilterForCubic(TiltImagePrefiltered.GetDevice(Intent.ReadWrite), TiltImagePrefiltered.Dims);

                GPU.WarpImage(TiltImagePrefiltered.GetDevice(Intent.Read),
                    warped.GetDevice(Intent.Write),
                    DimsImage,
                    WarpX,
                    WarpY,
                    DimsWarp,
                    IntPtr.Zero);

                TiltImagePrefiltered.Dispose();
            };

            for (int itilt = 1; itilt < NTilts; itilt++)
            {
                int t = IndicesSortedAbsoluteAngle[itilt];
                if (!UseTilt[t])
                    continue;

                //if (t == 0 || t == NTilts - 1)
                //    continue;

                #region Make global reconstruction

                Projector Reconstructor = new Projector(new int3(SizeReconstructionPadded), 1);
                Projector Sampler = new Projector(new int3(SizeReconstructionPadded), 1);

                Image CTFCoords = CTF.GetCTFCoords(SizeReconstructionPadded, SizeReconstructionPadded);
                Image CTFExtracted = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, 1), true);

                //int[] TwoClosest;
                //{
                //    List<int> AllTilts = Helper.ArrayOfSequence(0, NTilts, 1).ToList();
                //    AllTilts.RemoveAll(v => !UseTilt[v] || v == t);
                //    AllTilts.Sort((a, b) => Math.Abs(Angles[a] - Angles[t]).CompareTo(Math.Abs(Angles[b] - Angles[t])));
                //    TwoClosest = AllTilts.Take(2).ToArray();
                //}

                for (int i = 0; i < itilt; i++)
                {
                    int TiltID = IndicesSortedAbsoluteAngle[i];

                    if (i == itilt || !UseTilt[TiltID]) // || (MathF.Sign(Angles[t]) != MathF.Sign(Angles[TiltID]) && i != 0))
                        continue;

                    float3 PositionInImage = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, TiltID).First();
                    PositionInImage.X /= (float)options.BinnedPixelSizeMean;
                    PositionInImage.Y /= (float)options.BinnedPixelSizeMean;
                    int3 IntPosition = new int3(PositionInImage);
                    float2 Residual = new float2(-(PositionInImage.X - IntPosition.X), -(PositionInImage.Y - IntPosition.Y));
                    IntPosition.X -= DimsImage.X / 2;
                    IntPosition.Y -= DimsImage.Y / 2;
                    IntPosition.Z = 0;

                    Image Extracted = new Image(new int3(DimsImage));
                    GPU.Extract(CorrectedTilts[i].GetDevice(Intent.Read),
                        Extracted.GetDevice(Intent.Write),
                        new int3(DimsImage),
                        new int3(DimsImage),
                        Helper.ToInterleaved(new int3[] { IntPosition }),
                        true,
                        1);

                    Extracted.Multiply(1f / (SizeReconstructionPadded * SizeReconstructionPadded));
                    Image ExtractedPadded = Extracted.AsPadded(new int2(SizeReconstructionPadded)).AndDisposeParent();
                    ExtractedPadded.ShiftSlices(new[] { new float3(Residual.X - DimsImage.X, Residual.Y - DimsImage.Y, 0) });
                    Image ExtractedFT = ExtractedPadded.AsFFT().AndDisposeParent();

                    GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                        new[] { PositionInImage.Z },
                        new[] { VolumeDimensionsPhysical / 2 },
                        CTFCoords,
                        null,
                        TiltID,
                        CTFExtracted,
                        false);

                    ExtractedFT.Multiply(CTFExtracted);
                    CTFExtracted.Abs();

                    Reconstructor.BackProject(ExtractedFT,
                        CTFExtracted,
                        GetAnglesInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, new[] { new float3() }, TiltID),
                        Matrix2.Identity());

                    ExtractedFT.Fill(new float2(1, 0));
                    //ExtractedFT.Multiply(CTFExtracted);
                    CTFExtracted.Fill(1);

                    Sampler.BackProject(ExtractedFT,
                        CTFExtracted,
                        GetAnglesInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, new[] { new float3() }, TiltID),
                        Matrix2.Identity());

                    ExtractedFT.Dispose();
                }

                int ClosestTilt = t; // FindClosestProcessedTilt(t);

                Image Weights = Sampler.Weights.GetCopyGPU();
                Weights.Min(1);

                Reconstructor.Data.Multiply(Weights);
                Sampler.Data.Multiply(Weights);
                Weights.Dispose();

                //Reconstructor.Weights.Fill(1);
                //Sampler.Weights.Fill(1);
                //Reconstructor.Weights.Max(1);
                //Sampler.Weights.Max(1);

                Image Reconstruction = Reconstructor.Reconstruct(false, "C1", null, -1, -1, -1, 0).AsPadded(new int3(SizeReconstruction)).AndDisposeParent();
                Reconstructor.Dispose();
                Reconstruction.MaskRectangularly(new int3(SizeReconstruction - 32, SizeReconstruction - 32, SizeReconstruction / 4), 16, true);
                Reconstruction.WriteMRC("d_rec.mrc", true);

                Image Samples = Sampler.Reconstruct(false, "C1", null, -1, -1, -1, 0).AsPadded(new int3(SizeReconstruction)).AndDisposeParent();
                Sampler.Dispose();
                Samples.MaskSpherically(SizeReconstruction - 32, 16, true);
                Samples.MaskRectangularly(new int3(SizeReconstruction - 32, SizeReconstruction - 32, SizeReconstruction / 4), 16, true);
                //Samples.WriteMRC("d_samples.mrc", true);

                #endregion

                #region Project average and filter currently missing tilt

                Projector RecProjector = new Projector(Reconstruction, 2);
                Projector SamplesProjector = new Projector(Samples, 1);
                Reconstruction.Dispose();
                Samples.Dispose();

                Image NextTiltFull = RecProjector.ProjectToRealspace(new int2(SizeReconstruction), GetAnglesInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, new[] { new float3() }, ClosestTilt));
                NextTiltFull.ShiftSlices(new float3[] { new float3(TiltAxisOffsetX[ClosestTilt], TiltAxisOffsetY[ClosestTilt], 0) / (float)options.BinnedPixelSizeMean });
                NextTiltFull = NextTiltFull.AsPadded(DimsImage).AndDisposeParent();
                NextTiltFull.Normalize();
                RecProjector.Dispose();
                NextTiltFull.WriteMRC($"d_nexttilt_{t:D2}.mrc", true);

                Image NextTiltSamples = SamplesProjector.ProjectToRealspace(new int2(SizeReconstruction), GetAnglesInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, new[] { new float3() }, ClosestTilt));
                NextTiltSamples = NextTiltSamples.AsPadded(DimsImage * 2).AndDisposeParent().AsFFT().AndDisposeParent().AsAmplitudes().AndDisposeParent();
                NextTiltSamples.Multiply(1f / (DimsImage.Elements()));
                SamplesProjector.Dispose();
                //NextTiltSamples.WriteMRC("d_nexttiltsamples.mrc", true);

                GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                    new[] { GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, t).First().Z },
                    new[] { VolumeDimensionsPhysical / 2 },
                    CTFCoords,
                    null,
                    t,
                    CTFExtracted,
                    false);
                CTFExtracted.Sign();

                Image MissingTilt = TiltData[t].GetCopyGPU();
                MissingTilt = MissingTilt.AsPaddedClamped(DimsImage * 2).AndDisposeParent();
                MissingTilt.MaskRectangularly(new int3(DimsImage), DimsImage.X / 2, false);
                MissingTilt = MissingTilt.AsFFT().AndDisposeParent();
                MissingTilt.Multiply(CTFExtracted);
                MissingTilt.Multiply(NextTiltSamples);
                MissingTilt = MissingTilt.AsIFFT().AndDisposeParent().AsPadded(DimsImage).AndDisposeParent();
                MissingTilt.Multiply(1f / (DimsImage.Elements()));
                MissingTilt.Normalize();
                MissingTilt.WriteMRC($"d_missingtilt_{t:D2}.mrc", true);

                CTFExtracted.Dispose();
                CTFCoords.Dispose();

                #endregion

                #region Make references from global projection

                Image Refs;
                {
                    float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, ClosestTilt).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                    int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                    float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                    IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                    Refs = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length));
                    GPU.Extract(NextTiltFull.GetDevice(Intent.Read),
                        Refs.GetDevice(Intent.Write),
                        NextTiltFull.Dims,
                        Refs.Dims.Slice(),
                        Helper.ToInterleaved(IntPositions),
                        false,
                        (uint)PositionGrid.Length);
                    Refs.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                    //Refs.Normalize();
                    //Refs.WriteMRC("d_refs.mrc", true);
                }

                #endregion

                #region Perform optimization

                {
                    float[] OriValuesX = new float[GridMovementX.Values.Length]; // GridMovementX.Values.ToArray();
                    float[] OriValuesY = new float[GridMovementX.Values.Length]; //GridMovementY.Values.ToArray();
                    int ParamsPerTilt = (int)GridMovementX.Dimensions.ElementsSlice();

                    Action<double[]> SetGrids = (input) =>
                    {
                        float[] NewValuesX = OriValuesX.ToArray();
                        float[] NewValuesY = OriValuesY.ToArray();
                        for (int i = 0; i < ParamsPerTilt; i++)
                        {
                            if (itilt > 10)
                            {
                                if (i != ParamsPerTilt / 2)
                                {
                                    NewValuesX[t * ParamsPerTilt + i] += (float)input[0 * 2 + 0];
                                    NewValuesY[t * ParamsPerTilt + i] += (float)input[0 * 2 + 1];
                                }
                                else
                                {
                                    NewValuesX[t * ParamsPerTilt + i] += (float)input[1 * 2 + 0];
                                    NewValuesY[t * ParamsPerTilt + i] += (float)input[1 * 2 + 1];
                                }
                            }
                            else
                            {
                                NewValuesX[t * ParamsPerTilt + i] += (float)input[1 * 2 + 0];
                                NewValuesY[t * ParamsPerTilt + i] += (float)input[1 * 2 + 1];
                            }
                        }

                        GridMovementX = new CubicGrid(GridMovementX.Dimensions, NewValuesX);
                        GridMovementY = new CubicGrid(GridMovementY.Dimensions, NewValuesY);
                    };

                    Func<double[], double> Eval = (input) =>
                    {
                        SetGrids(input);

                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        Image Raws = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length));
                        GPU.Extract(MissingTilt.GetDevice(Intent.Read),
                            Raws.GetDevice(Intent.Write),
                            MissingTilt.Dims,
                            Raws.Dims.Slice(),
                            Helper.ToInterleaved(IntPositions),
                            false,
                            (uint)PositionGrid.Length);

                        GPU.FFT(Raws.GetDevice(Intent.Read), ExtractedFT1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                        ExtractedFT1.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        GPU.IFFT(ExtractedFT1.GetDevice(Intent.Read), Raws.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        //Raws.Normalize();
                        //Raws.WriteMRC("d_raws.mrc", true);

                        Raws.Multiply(Refs);
                        //Raws.Multiply(Raws);
                        Raws.MultiplySlices(RegionMask);

                        Image Diff = Raws.AsSum3D().AndDisposeParent();
                        double Result = Diff.GetHost(Intent.Read)[0][0];
                        Diff.Dispose();

                        return Result;
                    };

                    int OptIterations = 0;
                    Func<double[], double[]> Grad = (input) =>
                    {
                        double Delta = 0.1;
                        double[] Result = new double[input.Length];

                        if (OptIterations++ > 8)
                            return Result;

                        for (int i = 0; i < input.Length; i++)
                        {
                            double[] InputPlus = input.ToArray();
                            InputPlus[i] += Delta;
                            double ScorePlus = Eval(InputPlus);

                            double[] InputMinus = input.ToArray();
                            InputMinus[i] -= Delta;
                            double ScoreMinus = Eval(InputMinus);

                            Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                        }

                        Console.WriteLine(Eval(input));

                        return Result;
                    };

                    double[] StartValues = new double[2 * 2];
                    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartValues.Length, Eval, Grad);
                    Optimizer.Maximize(StartValues);

                    SetGrids(StartValues);
                    Scores.Add(Eval(StartValues));

                    Image CorrectedTilt = new Image(new int3(DimsImage));
                    MakeWarpedImage(t, CorrectedTilt);
                    CorrectedTilts.Add(CorrectedTilt);
                    TiltsProcessed.Add(t);

                    for (int i = 0; i < ParamsPerTilt; i++)
                    {
                        if (itilt > 10)
                        {
                            if (i != ParamsPerTilt / 2)
                            {
                                FinalShiftsX[t * ParamsPerTilt + i] = OriValuesX[t * ParamsPerTilt + i] + (float)StartValues[0 * 2 + 0];
                                FinalShiftsY[t * ParamsPerTilt + i] = OriValuesY[t * ParamsPerTilt + i] + (float)StartValues[0 * 2 + 1];
                            }
                            else
                            {
                                FinalShiftsX[t * ParamsPerTilt + i] = OriValuesX[t * ParamsPerTilt + i] + (float)StartValues[1 * 2 + 0];
                                FinalShiftsY[t * ParamsPerTilt + i] = OriValuesY[t * ParamsPerTilt + i] + (float)StartValues[1 * 2 + 1];
                            }
                        }
                        else
                        {
                            FinalShiftsX[t * ParamsPerTilt + i] = OriValuesX[t * ParamsPerTilt + i] + (float)StartValues[1 * 2 + 0];
                            FinalShiftsY[t * ParamsPerTilt + i] = OriValuesY[t * ParamsPerTilt + i] + (float)StartValues[1 * 2 + 1];
                        }
                    }

                    CorrectedTilt.WriteMRC($"d_corrected_{t:D2}.mrc", true);
                }

                #endregion

                Refs.Dispose();
                NextTiltFull.Dispose();
                NextTiltSamples.Dispose();
                MissingTilt.Dispose();

                Console.WriteLine(GPU.GetFreeMemory(0) + " MB");
            }

            Console.WriteLine(Scores.Sum() / Scores.Count);

            GridMovementX = new CubicGrid(GridMovementX.Dimensions, FinalShiftsX);
            GridMovementY = new CubicGrid(GridMovementY.Dimensions, FinalShiftsY);
        }

        GPU.DestroyFFTPlan(PlanForw);
        GPU.DestroyFFTPlan(PlanBack);
        Extracted1.Dispose();
        Extracted2.Dispose();
        Extracted3.Dispose();
        ExtractedFT1.Dispose();
        ExtractedFT2.Dispose();
        ExtractedFT3.Dispose();

        foreach (var data in TiltData)
            data.Dispose();

        SaveMeta();
    }
}