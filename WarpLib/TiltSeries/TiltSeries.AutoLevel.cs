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

        Image[] ExtractedRegions = Helper.ArrayOfFunction(i => new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length)), NRelevantTilts);
        Image[] ExtractedRegionsFT = Helper.ArrayOfFunction(i => new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length), true, true), NRelevantTilts);
        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length);
        int PlanBack = GPU.CreateIFFTPlan(new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length);

        #endregion

        #region Optimize

        foreach (float lowpassFraction in new float[] { 0.5f, 1.0f })
        {
            float OriginalLevelAngleX = LevelAngleX;
            float OriginalLevelAngleY = LevelAngleY;

            float LowpassFraction = lowpassFraction;

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

                return Result / (NRelevantTilts - 1) * 1000;
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

                SetAngles(input);

                Console.WriteLine($"Lowpass = {lowpassFraction:F1}*Ny, Score = {Eval(input):F3}, LevelAngleX = {LevelAngleX:F3}, LevelAngleY = {LevelAngleY:F3}");

                return Result;
            };

            double[] StartParams = new double[2];
            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
            Optimizer.Maximize(StartParams);

            SetAngles(Optimizer.Solution);
        }

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

        SaveMeta();
    }
}

[Serializable]
public class ProcessingOptionsTomoAutoLevel : TomoProcessingOptionsBase
{
    [WarpSerializable] public int RegionSize { get; set; }

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
               RegionSize == other.RegionSize;
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