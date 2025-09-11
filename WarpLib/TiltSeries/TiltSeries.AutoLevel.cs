using Accord.Math.Optimization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class TiltSeries
{
    public void AutoLevel(ProcessingOptionsTomoAutoLevel options)
    {
        VolumeDimensionsPhysical = options.DimensionsPhysical;
        int SizeRegion = options.RegionSize;

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
        }

        int2 DimsImage = new int2(TiltData[0].Dims);
        int SizeReconstruction = Math.Max(DimsImage.X, DimsImage.Y);
        int SizeReconstructionPadded = SizeReconstruction * 2;

        #endregion

        #region Prep

        int2 DimsPositionGrid;
        int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage - (SizeRegion * 2),
                                                         new int2(SizeRegion),
                                                         0.75f,
                                                         out DimsPositionGrid)
                              .Select(v => new int3(v.X + 32 + SizeRegion / 2,
                                                    v.Y + 32 + SizeRegion / 2,
                                                    0))
                              .ToArray();
        float3[] PositionGridPhysical = PositionGrid.Select(v => new float3(v.X * (float)options.BinnedPixelSizeMean,
                                                                            v.Y * (float)options.BinnedPixelSizeMean,
                                                                            VolumeDimensionsPhysical.Z / 2))
                                                    .ToArray();

        Image RegionMask = new Image(new int3(SizeRegion, SizeRegion, 1));
        RegionMask.Fill(1);
        RegionMask.MaskRectangularly(new int3(SizeRegion / 2, SizeRegion / 2, 1), SizeRegion / 4 - 2, false);
        //RegionMask.WriteMRC("d_mask.mrc", true);

        int[] RelevantTiltIDs = Enumerable.Range(0, NTilts).Where(t => MathF.Abs(Angles[t]) < 30).ToArray();
        int NRelevantTilts = RelevantTiltIDs.Length;

        Image[] ExtractedRegions = Helper.ArrayOfFunction(i => new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length)), NTilts);
        Image[] ExtractedRegionsFT = Helper.ArrayOfFunction(i => new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length), true, true), NTilts);
        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length);
        int PlanBack = GPU.CreateIFFTPlan(new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length);

        #endregion

        #region Angle optimization

        var OptimizeAngles = () =>
        {
            foreach (float lowpassFraction in new float[] { 1.0f })
            {
                float OriginalLevelAngleX = LevelAngleX;
                float OriginalLevelAngleY = LevelAngleY;

                float LowpassFraction = lowpassFraction;

                double BestScore = double.NegativeInfinity;
                double[] BestInput = null;

                Action<double[]> SetAngles = (input) =>
                {
                    LevelAngleX = OriginalLevelAngleX + (float)input[0];
                    LevelAngleY = OriginalLevelAngleY + (float)input[1];
                };

                Func<double[], double> Eval = (input) =>
                {
                    SetAngles(input);

                    double Result = 0;

                    for (int irel = 0; irel < NRelevantTilts; irel++)
                    {
                        int t = RelevantTiltIDs[irel];

                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t)
                                                    .Select(v => v / (float)options.BinnedPixelSizeMean)
                                                    .ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X),
                                                                                     -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2,
                                                                         v.Y - SizeRegion / 2,
                                                                         0)).ToArray();

                        GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                    ExtractedRegions[irel].GetDevice(Intent.Write),
                                    TiltData[t].Dims,
                                    ExtractedRegions[irel].Dims.Slice(),
                                    Helper.ToInterleaved(IntPositions),
                                    false,
                                    (uint)PositionGrid.Length);

                        GPU.FFT(ExtractedRegions[irel].GetDevice(Intent.Read),
                                ExtractedRegionsFT[irel].GetDevice(Intent.Write),
                                new int3(SizeRegion).Slice(),
                                (uint)PositionGridPhysical.Length,
                                PlanForw);

                        ExtractedRegionsFT[irel].ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        ExtractedRegionsFT[irel].Bandpass(0, LowpassFraction, false, LowpassFraction * 0.1f);

                        GPU.IFFT(ExtractedRegionsFT[irel].GetDevice(Intent.Read),
                                 ExtractedRegions[irel].GetDevice(Intent.Write),
                                 new int3(SizeRegion).Slice(),
                                 (uint)PositionGridPhysical.Length,
                                 PlanBack,
                                 false);

                        ExtractedRegions[irel].Normalize();
                        ExtractedRegions[irel].MultiplySlices(RegionMask);
                    }

                    for (int irel = 0; irel < NRelevantTilts - 1; irel++)
                    {
                        ExtractedRegions[irel].Multiply(ExtractedRegions[irel + 1]);
                        Image Sum = ExtractedRegions[irel].AsSum3D();
                        Result += Sum.GetHost(Intent.Read)[0][0] / (PositionGridPhysical.Length * SizeRegion * SizeRegion);

                        Sum.Dispose();
                    }

                    Result = Result / (NRelevantTilts - 1) * 1000;

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
                    double Delta = 0.01;
                    double[] Result = new double[input.Length];

                    if (OptIterations++ > 9)
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

                    SetAngles(input);

                    Console.WriteLine($"Lowpass = {lowpassFraction:F1}*Ny, Score = {Eval(input):F3}, LevelAngleX = {LevelAngleX:F3}, LevelAngleY = {LevelAngleY:F3}");

                    return Result;
                };

                double[] StartParams = new double[2];
                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                Optimizer.Maximize(StartParams);

                SetAngles(BestInput);
            }
        };

        #endregion

        #region Elevation correction

        var OptimizeElevation = () =>
        {
            foreach (var lowpassFraction in new float[] { 1f })
            {
                var OriginalPositions = PositionGridPhysical.ToArray();
                var LowpassFraction = lowpassFraction;

                double BestScore = double.NegativeInfinity;
                double[] BestInput = null;

                Action<double[]> SetPositions = (input) =>
                {
                    for (int i = 0; i < PositionGridPhysical.Length; i++)
                        PositionGridPhysical[i] = new float3(OriginalPositions[i].X,
                                                             OriginalPositions[i].Y,
                                                             OriginalPositions[i].Z + (float)input[i]);
                };

                Func<double[], double[]> EvalIndividual = (input) =>
                {
                    SetPositions(input);

                    var Result = new double[input.Length];

                    for (int irel = 0; irel < NRelevantTilts; irel++)
                    {
                        int t = RelevantTiltIDs[irel];

                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t)
                                                    .Select(v => v / (float)options.BinnedPixelSizeMean)
                                                    .ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X),
                                                                                     -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2,
                                                                         v.Y - SizeRegion / 2,
                                                                         0)).ToArray();

                        GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                    ExtractedRegions[irel].GetDevice(Intent.Write),
                                    TiltData[t].Dims,
                                    ExtractedRegions[irel].Dims.Slice(),
                                    Helper.ToInterleaved(IntPositions),
                                    false,
                                    (uint)PositionGrid.Length);

                        GPU.FFT(ExtractedRegions[irel].GetDevice(Intent.Read),
                                ExtractedRegionsFT[irel].GetDevice(Intent.Write),
                                new int3(SizeRegion).Slice(),
                                (uint)PositionGridPhysical.Length,
                                PlanForw);

                        ExtractedRegionsFT[irel].ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        ExtractedRegionsFT[irel].Bandpass(0, LowpassFraction, false, LowpassFraction * 0.1f);

                        GPU.IFFT(ExtractedRegionsFT[irel].GetDevice(Intent.Read),
                                 ExtractedRegions[irel].GetDevice(Intent.Write),
                                 new int3(SizeRegion).Slice(),
                                 (uint)PositionGridPhysical.Length,
                                 PlanBack,
                                 false);

                        ExtractedRegions[irel].Normalize();
                        ExtractedRegions[irel].MultiplySlices(RegionMask);
                    }

                    for (int irel = 0; irel < NRelevantTilts - 1; irel++)
                    {
                        ExtractedRegions[irel].Multiply(ExtractedRegions[irel + 1]);
                        Image Sums = ExtractedRegions[irel].AsSum2D();
                        float[] SumsData = Sums.GetHost(Intent.Read)[0];

                        for (int i = 0; i < SumsData.Length; i++)
                            Result[i] += SumsData[i] / (SizeRegion * SizeRegion);

                        Sums.Dispose();
                    }

                    return Result.Select(v => v / (NRelevantTilts - 1) * 1000).ToArray();
                };

                Func<double[], double> Eval = (input) =>
                {
                    var IndividualScores = EvalIndividual(input);
                    var Result = IndividualScores.Sum();

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
                    double Delta = 0.1;
                    double[] Result = new double[input.Length];

                    if (OptIterations++ > 40)
                        return Result;

                    double[] InputPlus = input.Select(v => v + Delta).ToArray();
                    double[] ScoresPlus = EvalIndividual(InputPlus);

                    double[] InputMinus = input.Select(v => v - Delta).ToArray();
                    double[] ScoresMinus = EvalIndividual(InputMinus);

                    for (int i = 0; i < input.Length; i++)
                        Result[i] = (ScoresPlus[i] - ScoresMinus[i]) / (Delta * 2);

                    SetPositions(input);

                    Console.WriteLine($"Lowpass = {LowpassFraction:F1}*Ny, Score = {Eval(input):F3}");

                    return Result;
                };

                double[] StartParams = new double[PositionGridPhysical.Length];
                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                Optimizer.Maximize(StartParams);

                SetPositions(BestInput);

                Image PositionsVis = new Image(PositionGridPhysical.Select(v => v.Z - VolumeDimensionsPhysical.Z / 2).ToArray(),
                                                new int3(DimsPositionGrid.X, DimsPositionGrid.Y, 1));
                PositionsVis.WriteMRC($"d_elevation_{LowpassFraction:F1}.mrc", true);
            }
        };

        #endregion

        #region Alignment optimization

        var OptimizeAlignment = () =>
        {
            if (GridMovementX == null || 
                GridMovementX.Values.Length != NTilts * options.GridSize * options.GridSize)
            {
                GridMovementX = new CubicGrid(new int3(options.GridSize, options.GridSize, NTilts));
                GridMovementY = new CubicGrid(new int3(options.GridSize, options.GridSize, NTilts));
            }

            foreach (float lowpassFraction in new float[] { 1.0f })
            {
                int CentralTilt = IndicesSortedAbsoluteAngle[0];

                foreach (var optimizedTilt in IndicesSortedAbsoluteAngle.Skip(1))
                {
                    float[] OriginalShiftX = GridMovementX.FlatValues.ToArray();
                    float[] OriginalShiftY = GridMovementY.FlatValues.ToArray();
                    int ElementsSlice = (int)GridMovementX.Dimensions.ElementsSlice();

                    float LowpassFraction = lowpassFraction;

                    double BestScore = double.NegativeInfinity;
                    double[] BestInput = null;

                    Action<double[], int[]> SetShifts = (input, tiltIds) =>
                    {
                        var UpdatedX = OriginalShiftX.ToArray();
                        foreach (var tiltId in tiltIds)
                            for (int i = 0; i < ElementsSlice; i++)
                                UpdatedX[tiltId * ElementsSlice + i] += (float)input[i * 2 + 0];
                        var UpdatedY = OriginalShiftY.ToArray();
                        foreach (var tiltId in tiltIds)
                            for (int i = 0; i < ElementsSlice; i++)
                                UpdatedY[tiltId * ElementsSlice + i] += (float)input[i * 2 + 1];

                        GridMovementX = new CubicGrid(new int3(options.GridSize, options.GridSize, NTilts), UpdatedX);
                        GridMovementY = new CubicGrid(new int3(options.GridSize, options.GridSize, NTilts), UpdatedY);
                    };

                    Func<double[], double> Eval = (input) =>
                    {
                        SetShifts(input, [optimizedTilt]);

                        double Result = 0;

                        int NeighborTilt = optimizedTilt < CentralTilt ? 
                                            optimizedTilt + 1 : 
                                            optimizedTilt - 1;
                        int[] TiltPair = [optimizedTilt, NeighborTilt];

                        foreach (var t in TiltPair)
                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t)
                                                        .Select(v => v / (float)options.BinnedPixelSizeMean)
                                                        .ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X),
                                                                                         -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2,
                                                                             v.Y - SizeRegion / 2,
                                                                             0)).ToArray();

                            GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                        ExtractedRegions[t].GetDevice(Intent.Write),
                                        TiltData[t].Dims,
                                        ExtractedRegions[t].Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(ExtractedRegions[t].GetDevice(Intent.Read),
                                    ExtractedRegionsFT[t].GetDevice(Intent.Write),
                                    new int3(SizeRegion).Slice(),
                                    (uint)PositionGridPhysical.Length,
                                    PlanForw);

                            ExtractedRegionsFT[t].ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            ExtractedRegionsFT[t].Bandpass(0, LowpassFraction, false, LowpassFraction * 0.1f);

                            GPU.IFFT(ExtractedRegionsFT[t].GetDevice(Intent.Read),
                                     ExtractedRegions[t].GetDevice(Intent.Write),
                                     new int3(SizeRegion).Slice(),
                                     (uint)PositionGridPhysical.Length,
                                     PlanBack,
                                     false);

                            ExtractedRegions[t].Normalize();
                            ExtractedRegions[t].MultiplySlices(RegionMask);
                        }

                        ExtractedRegions[TiltPair[0]].Multiply(ExtractedRegions[TiltPair[1]]);
                        Image Sum = ExtractedRegions[TiltPair[0]].AsSum3D();
                        Result = Sum.GetHost(Intent.Read)[0][0] / (PositionGridPhysical.Length * SizeRegion * SizeRegion);

                        Sum.Dispose();

                        Result = Result * 1000;

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
                        double Delta = 0.1;
                        double[] Result = new double[input.Length];

                        if (OptIterations++ > 9)
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

                        Console.WriteLine($"Tilt = {optimizedTilt}, Lowpass = {lowpassFraction:F1}*Ny, Score = {Eval(input):F3}, ShiftX = {input[0]:F2}, ShiftY = {input[1]:F2}");

                        return Result;
                    };

                    double[] StartParams = new double[ElementsSlice * 2];
                    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                    Optimizer.Maximize(StartParams);

                    //SetShifts(BestInput, [optimizedTilt]);

                    // Systematically shift all lower/higher tilts to propagate the change in alignment
                    if (optimizedTilt > CentralTilt && optimizedTilt < NTilts - 1)
                        SetShifts(BestInput, Enumerable.Range(optimizedTilt, NTilts - optimizedTilt).ToArray());
                    else if (optimizedTilt < CentralTilt && optimizedTilt > 0)
                        SetShifts(BestInput, Enumerable.Range(0, optimizedTilt + 1).ToArray());
                }
            }
        };

        #endregion

        #region Perform optimization

        OptimizeAngles();
        OptimizeElevation();

        OptimizeAngles();
        OptimizeElevation();

        OptimizeAlignment();

        #endregion

        #region Cleanup

        foreach (var im in TiltData)
            im.Dispose();
        foreach (var im in ExtractedRegions)
            im.Dispose();
        foreach (var im in ExtractedRegionsFT)
            im.Dispose();
        RegionMask.Dispose();
        GPU.DestroyFFTPlan(PlanForw);
        GPU.DestroyFFTPlan(PlanBack);

        #endregion

        Console.WriteLine($"Final leveling angles: X = {LevelAngleX}, Y = {LevelAngleY}");

        //SaveMeta();
    }
}

[Serializable]
public class ProcessingOptionsTomoAutoLevel : TomoProcessingOptionsBase
{
    [WarpSerializable] public int RegionSize { get; set; } = 500;

    [WarpSerializable] public int GridSize { get; set; } = 3;

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((ProcessingOptionsTomoFullReconstruction)obj);
    }

    protected bool Equals(ProcessingOptionsTomoAutoLevel other)
    {
        return base.Equals(other) &&
               RegionSize == other.RegionSize &&
               GridSize == other.GridSize;
    }

    public static bool operator ==(ProcessingOptionsTomoAutoLevel left, ProcessingOptionsTomoAutoLevel right)
    {
        return Equals(left, right);
    }

    public static bool operator !=(ProcessingOptionsTomoAutoLevel left, ProcessingOptionsTomoAutoLevel right)
    {
        return !Equals(left, right);
    }
}