using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class Movie
{
    public void ProcessShift(Image originalStack, ProcessingOptionsMovieMovement options)
    {
        if (originalStack.Dims.Z == 1)
        {
            OptionsMovement = options;
            SaveMeta();

            return;
        }

        IsProcessing = true;

        // Deal with dimensions and grids.

        int NFrames = originalStack.Dims.Z;
        int2 DimsImage = new int2(originalStack.Dims);
        int2 DimsRegion = new int2(768, 768);

        float OverlapFraction = 0.5f;
        int2 DimsPositionGrid;
        int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, DimsRegion, OverlapFraction, out DimsPositionGrid);
        //PositionGrid = new[] { new int3(0, 0, 0) };
        //DimsPositionGrid = new int2(1, 1);
        int NPositions = PositionGrid.Length;

        // Auto grid dims
        if (options.GridDims.Elements() == 0)
        {
            float OverallDose = (float)(options.DosePerAngstromFrame < 0 ? -options.DosePerAngstromFrame : options.DosePerAngstromFrame * NFrames);

            int AutoZ = (int)MathF.Ceiling(Math.Max(1, OverallDose));
            int AutoX, AutoY;

            // For a FoV of 4000 Angstrom, use a grid side length of 5
            // Scale up linearly for larger FoV
            // Scale down linearly for lower dose than 30 e/A^2
            float ShortAngstrom = Math.Min(originalStack.Dims.X, originalStack.Dims.Y) * (float)options.BinnedPixelSizeMean;
            int ShortGrid = (int)Math.Max(1, MathF.Round(5f * Math.Min(1, OverallDose / 30f) * (ShortAngstrom / 4000f)));

            if (originalStack.Dims.X <= originalStack.Dims.Y)
            {
                AutoX = ShortGrid;
                AutoY = (int)MathF.Round(ShortGrid * (float)originalStack.Dims.Y / originalStack.Dims.X);
            }
            else
            {
                AutoY = ShortGrid;
                AutoX = (int)MathF.Round(ShortGrid * (float)originalStack.Dims.X / originalStack.Dims.Y);
            }

            options.GridDims = new int3(AutoX, AutoY, AutoZ);
        }

        int ShiftGridX = options.GridDims.X;
        int ShiftGridY = options.GridDims.Y;
        int ShiftGridZ = Math.Min(NFrames, options.GridDims.Z);
        GridMovementX = new CubicGrid(new int3(1, 1, ShiftGridZ));
        GridMovementY = new CubicGrid(new int3(1, 1, ShiftGridZ));

        int LocalGridX = Math.Min(DimsPositionGrid.X, options.GridDims.X);
        int LocalGridY = Math.Min(DimsPositionGrid.Y, options.GridDims.Y);
        int LocalGridZ = LocalGridX * LocalGridY <= 1 ? 1 : 4; //Math.Max(3, (int)Math.Ceiling(options.GridDims.Z / (float)(LocalGridX * LocalGridY)));
        GridLocalX = new CubicGrid(new int3(LocalGridX, LocalGridY, LocalGridZ));
        GridLocalY = new CubicGrid(new int3(LocalGridX, LocalGridY, LocalGridZ));

        PyramidShiftX = new List<CubicGrid>();
        PyramidShiftY = new List<CubicGrid>();

        int3 ShiftGrid = new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames);

        int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
        int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
        int NFreq = MaxFreqExclusive - MinFreqInclusive;

        int CentralFrame = NFrames / 2;

        int MaskExpansions = Math.Max(1, (int)Math.Ceiling(Math.Log(ShiftGridZ - 0.01, 3))); // Each expansion doubles the temporal resolution
        int[] MaskSizes = new int[MaskExpansions];

        // Allocate memory and create all prerequisites:
        int MaskLength;
        Image ShiftFactors;
        Image Patches;
        Image PatchesAverage;
        Image Shifts;
        {
            List<long> Positions = new List<long>();
            List<float2> Factors = new List<float2>();
            List<float2> Freq = new List<float2>();
            int Min2 = MinFreqInclusive * MinFreqInclusive;
            int Max2 = MaxFreqExclusive * MaxFreqExclusive;
            float PixelSize = (float)options.BinnedPixelSizeMean;

            for (int y = 0; y < DimsRegion.Y; y++)
            {
                int yy = y > DimsRegion.Y / 2 ? y - DimsRegion.Y : y;
                for (int x = 0; x < DimsRegion.X / 2 + 1; x++)
                {
                    int xx = x;
                    int r2 = xx * xx + yy * yy;
                    if (r2 >= Min2 && r2 < Max2)
                    {
                        Positions.Add(y * (DimsRegion.X / 2 + 1) + x);
                        Factors.Add(new float2((float)xx / DimsRegion.X * 2f * (float)Math.PI,
                            (float)yy / DimsRegion.Y * 2f * (float)Math.PI));

                        float Angle = (float)Math.Atan2(yy, xx);
                        float r = (float)Math.Sqrt(r2);
                        Freq.Add(new float2(r, Angle));
                    }
                }
            }

            // Sort everyone by ascending distance from center.
            List<KeyValuePair<float, int>> FreqIndices = Freq.Select((v, i) => new KeyValuePair<float, int>(v.X, i)).ToList();
            FreqIndices.Sort((a, b) => a.Key.CompareTo(b.Key));
            int[] SortedIndices = FreqIndices.Select(v => v.Value).ToArray();

            Helper.Reorder(Positions, SortedIndices);
            Helper.Reorder(Factors, SortedIndices);
            Helper.Reorder(Freq, SortedIndices);

            float[] CTF2D = Helper.ArrayOfConstant(1f, Freq.Count);
            if (OptionsCTF != null)
            {
                CTF CTFCopy = CTF.GetCopy();
                CTFCopy.PixelSize = (decimal)PixelSize;
                CTF2D = CTFCopy.Get2D(Freq.Select(v => new float2(v.X / DimsRegion.X, v.Y)).ToArray(), true);
            }

            long[] RelevantMask = Positions.ToArray();
            ShiftFactors = new Image(Helper.ToInterleaved(Factors.ToArray()));
            MaskLength = RelevantMask.Length;

            // Get mask sizes for different expansion steps.
            for (int i = 0; i < MaskExpansions; i++)
            {
                float CurrentMaxFreq = MinFreqInclusive + (MaxFreqExclusive - MinFreqInclusive) / (float)MaskExpansions * (i + 1);
                MaskSizes[i] = Freq.Count(v => v.X * v.X < CurrentMaxFreq * CurrentMaxFreq);
            }

            Patches = new Image(IntPtr.Zero, new int3(MaskLength, DimsPositionGrid.X * DimsPositionGrid.Y, NFrames), false, true, false);
            Image Sigma = new Image(IntPtr.Zero, new int3(DimsRegion), true);

            GPU.CreateShift(originalStack.GetDevice(Intent.Read),
                DimsImage,
                originalStack.Dims.Z,
                PositionGrid,
                PositionGrid.Length,
                DimsRegion,
                RelevantMask,
                (uint)MaskLength,
                Patches.GetDevice(Intent.Write),
                Sigma.GetDevice(Intent.Write));

            //Sigma.WriteMRC("d_sigma.mrc", true);
            float AmpsMean = MathHelper.Mean(Sigma.GetHostContinuousCopy());
            //float[] Sigma1D = Sigma.AsAmplitudes1D(false);
            Sigma.Dispose();
            //Sigma1D[0] = Sigma1D[1];
            //float Sigma1DMean = MathHelper.Mean(Sigma1D);
            //Sigma1D = Sigma1D.Select(v => v / Sigma1DMean).ToArray();
            //Sigma1D = Sigma1D.Select(v => v > 0 ? 1 / v : 0).ToArray();
            //Sigma1D = Sigma1D.Select(v => 1 / Sigma1DMean).ToArray();

            float Bfac = (float)options.Bfactor * 0.25f;
            float[] BfacWeightsData = Freq.Select((v, i) =>
            {
                float r2 = v.X / PixelSize / DimsRegion.X;
                r2 *= r2;
                return (float)Math.Exp(r2 * Bfac); // * CTF2D[i];// * Sigma1D[(int)Math.Round(v.X)];
            }).ToArray();
            Image BfacWeights = new Image(BfacWeightsData);

            Patches.MultiplyLines(BfacWeights);
            Patches.Multiply(1 / AmpsMean);
            BfacWeights.Dispose();

            originalStack.FreeDevice();
            PatchesAverage = new Image(IntPtr.Zero, new int3(MaskLength, NPositions, 1), false, true);
            Shifts = new Image(new int3(NPositions * NFrames * 2, 1, 1));
        }

        #region Fit movement

        var Timer0 = ShiftTimers[0].Start();
        {
            int MinXSteps = 1, MinYSteps = 1;
            int MinZSteps = Math.Min(NFrames, 3);
            int3 ExpansionGridSize = new int3(MinXSteps, MinYSteps, MinZSteps);
            int3 LocalGridSize = new int3(LocalGridX, LocalGridY, LocalGridZ);
            int LocalGridParams = (int)LocalGridSize.Elements() * 2;

            // Get wiggle weights for global and local shifts, will need latter in last iteration
            float[][] WiggleWeights = new CubicGrid(ExpansionGridSize).GetWiggleWeights(ShiftGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));
            float[][] WiggleWeightsLocal = new CubicGrid(LocalGridSize).GetWiggleWeights(ShiftGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

            double[] StartParams = new double[ExpansionGridSize.Elements() * 2 + LocalGridParams];

            for (int m = 0; m < MaskExpansions; m++)
            {
                bool LastIter = m == MaskExpansions - 1;
                double[] LastAverage = null;

                int ExpansionGridParams = (int)ExpansionGridSize.Elements() * 2;

                #region Helper methods

                Action<double[]> SetPositions = input =>
                {
                    // Construct CubicGrids and get interpolated shift values.
                    CubicGrid AlteredGridX = new CubicGrid(ExpansionGridSize, input.Take(ExpansionGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                    CubicGrid AlteredGridY = new CubicGrid(ExpansionGridSize, input.Take(ExpansionGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());

                    float[] AlteredX = AlteredGridX.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                        new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));
                    float[] AlteredY = AlteredGridY.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                        new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                    // In last iteration, also model local motion
                    //if (LastIter)
                    {
                        CubicGrid AlteredGridLocalX = new CubicGrid(LocalGridSize, input.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                        CubicGrid AlteredGridLocalY = new CubicGrid(LocalGridSize, input.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());

                        float[] AlteredLocalX = AlteredGridLocalX.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                            new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));
                        float[] AlteredLocalY = AlteredGridLocalY.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                            new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                        for (int i = 0; i < AlteredX.Length; i++)
                        {
                            AlteredX[i] += AlteredLocalX[i];
                            AlteredY[i] += AlteredLocalY[i];
                        }
                    }

                    // Finally, set the shift values in the device array.
                    float[] ShiftData = Shifts.GetHost(Intent.Write)[0];
                    Parallel.For(0, AlteredX.Length, i =>
                    {
                        ShiftData[i * 2] = AlteredX[i]; // - CenterFrameOffsets[i % NPositions].X;
                        ShiftData[i * 2 + 1] = AlteredY[i]; // - CenterFrameOffsets[i % NPositions].Y;
                    });
                };

                Action<double[]> DoAverage = input =>
                {
                    if (LastAverage == null || input.Where((t, i) => t != LastAverage[i]).Any())
                    {
                        SetPositions(input);
                        GPU.ShiftGetAverage(Patches.GetDevice(Intent.Read),
                            PatchesAverage.GetDevice(Intent.Write),
                            ShiftFactors.GetDevice(Intent.Read),
                            (uint)MaskLength,
                            (uint)MaskSizes[m],
                            Shifts.GetDevice(Intent.Read),
                            (uint)NPositions,
                            (uint)NFrames);

                        if (LastAverage == null)
                            LastAverage = new double[input.Length];
                        Array.Copy(input, LastAverage, input.Length);
                    }
                };

                #endregion

                #region Eval and gradient methods

                Func<double[], double> Eval = input =>
                {
                    DoAverage(input);
                    //SetPositions(input);

                    float[] Diff = new float[NPositions * NFrames];
                    GPU.ShiftGetDiff(Patches.GetDevice(Intent.Read),
                        PatchesAverage.GetDevice(Intent.Read),
                        ShiftFactors.GetDevice(Intent.Read),
                        (uint)MaskLength,
                        (uint)MaskSizes[m],
                        Shifts.GetDevice(Intent.Read),
                        Diff,
                        (uint)NPositions,
                        (uint)NFrames);

                    for (int i = 0; i < Diff.Length; i++)
                        Diff[i] = Diff[i];

                    //Debug.WriteLine(Diff.Sum());

                    return Diff.Sum();
                };

                Func<double[], double[]> Grad = input =>
                {
                    DoAverage(input);
                    //SetPositions(input);

                    float[] GradX = new float[NPositions * NFrames], GradY = new float[NPositions * NFrames];

                    float[] Diff = new float[NPositions * NFrames * 2];
                    GPU.ShiftGetGrad(Patches.GetDevice(Intent.Read),
                        PatchesAverage.GetDevice(Intent.Read),
                        ShiftFactors.GetDevice(Intent.Read),
                        (uint)MaskLength,
                        (uint)MaskSizes[m],
                        Shifts.GetDevice(Intent.Read),
                        Diff,
                        (uint)NPositions,
                        (uint)NFrames);

                    for (int i = 0; i < GradX.Length; i++)
                    {
                        GradX[i] = Diff[i * 2];
                        GradY[i] = Diff[i * 2 + 1];
                    }

                    double[] Result = new double[input.Length];

                    Parallel.For(0, ExpansionGridParams / 2, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, i =>
                        //for (int i = 0; i < ExpansionGridParams / 2; i++)
                    {
                        Result[i * 2] = MathHelper.ReduceWeighted(GradX, WiggleWeights[i]);
                        Result[i * 2 + 1] = MathHelper.ReduceWeighted(GradY, WiggleWeights[i]);
                    });

                    //if (LastIter)
                    Parallel.For(0, LocalGridParams / 2, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, i =>
                        //for (int i = 0; i < LocalGridParams / 2; i++)
                    {
                        Result[ExpansionGridParams + i * 2] = MathHelper.ReduceWeighted(GradX, WiggleWeightsLocal[i]);
                        Result[ExpansionGridParams + i * 2 + 1] = MathHelper.ReduceWeighted(GradY, WiggleWeightsLocal[i]);
                    });

                    return Result;
                };

                #endregion

                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                Optimizer.MaxIterations = 10;
                Optimizer.Minimize(StartParams);

                // Anything should be quite centered anyway

                //float MeanX = MathHelper.Mean(Optimizer.Solution.Where((v, i) => i % 2 == 0).Select(v => (float)v));
                //float MeanY = MathHelper.Mean(Optimizer.Solution.Where((v, i) => i % 2 == 1).Select(v => (float)v));
                //for (int i = 0; i < ExpansionGridSize.Elements(); i++)
                //{
                //    Optimizer.Solution[i * 2] -= MeanX;
                //    Optimizer.Solution[i * 2 + 1] -= MeanY;
                //}

                // Store coarse values in grids.
                GridMovementX = new CubicGrid(ExpansionGridSize, Optimizer.Solution.Take(ExpansionGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                GridMovementY = new CubicGrid(ExpansionGridSize, Optimizer.Solution.Take(ExpansionGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());

                //if (LastIter)
                {
                    GridLocalX = new CubicGrid(LocalGridSize, Optimizer.Solution.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                    GridLocalY = new CubicGrid(LocalGridSize, Optimizer.Solution.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());
                }

                if (!LastIter)
                {
                    // Refine sampling.
                    ExpansionGridSize = new int3(1, //(int)Math.Round((float)(ShiftGridX - MinXSteps) / (MaskExpansions - 1) * (m + 1) + MinXSteps),
                        1, //(int)Math.Round((float)(ShiftGridY - MinYSteps) / (MaskExpansions - 1) * (m + 1) + MinYSteps),
                        (int)Math.Round((float)Math.Min(ShiftGridZ, Math.Pow(3, m + 2))));
                    ExpansionGridParams = (int)ExpansionGridSize.Elements() * 2;

                    WiggleWeights = new CubicGrid(ExpansionGridSize).GetWiggleWeights(ShiftGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                    // Resize the grids to account for finer sampling.
                    GridMovementX = GridMovementX.Resize(ExpansionGridSize);
                    GridMovementY = GridMovementY.Resize(ExpansionGridSize);

                    // Construct start parameters for next optimization iteration.
                    StartParams = new double[ExpansionGridParams + LocalGridParams];
                    for (int i = 0; i < ExpansionGridParams / 2; i++)
                    {
                        StartParams[i * 2] = GridMovementX.FlatValues[i];
                        StartParams[i * 2 + 1] = GridMovementY.FlatValues[i];
                    }

                    for (int i = 0; i < LocalGridParams / 2; i++)
                    {
                        StartParams[ExpansionGridParams + i * 2] = GridLocalX.FlatValues[i];
                        StartParams[ExpansionGridParams + i * 2 + 1] = GridLocalY.FlatValues[i];
                    }
                    // Local shifts will be initialized with 0 for last iteration
                }
            }
        }
        ShiftTimers[0].Finish(Timer0);

        #endregion

        // Center the global shifts
        {
            float2 AverageShift = new float2(MathHelper.Mean(GridMovementX.FlatValues),
                MathHelper.Mean(GridMovementY.FlatValues));

            GridMovementX = new CubicGrid(GridMovementX.Dimensions, GridMovementX.FlatValues.Select(v => v - AverageShift.X).ToArray());
            GridMovementY = new CubicGrid(GridMovementY.Dimensions, GridMovementY.FlatValues.Select(v => v - AverageShift.Y).ToArray());
        }

        // Scale everything from (binned) pixels to Angstrom
        GridMovementX = new CubicGrid(GridMovementX.Dimensions, GridMovementX.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());
        GridMovementY = new CubicGrid(GridMovementY.Dimensions, GridMovementY.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());

        GridLocalX = new CubicGrid(GridLocalX.Dimensions, GridLocalX.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());
        GridLocalY = new CubicGrid(GridLocalY.Dimensions, GridLocalY.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());

        ShiftFactors.Dispose();
        Patches.Dispose();
        PatchesAverage.Dispose();
        Shifts.Dispose();

        OptionsMovement = options;

        // Calculate mean per-frame shift
        {
            float2[] Track = GetMotionTrack(new float2(0.5f, 0.5f), 1);
            float[] Diff = MathHelper.Diff(Track).Select(v => v.Length()).ToArray();
            MeanFrameMovement = (decimal)Diff.Take(Math.Max(1, Diff.Length / 3)).Average();
        }

        // Save XML metadata and export motion tracks json
        SaveMeta();
        SaveMotionTracks();

        //lock (ShiftTimers)
        //{
        //    if (ShiftTimers[0].NItems > 0)
        //        foreach (var timer in ShiftTimers)
        //            Console.WriteLine(timer.Name + ": " + timer.GetAverageMilliseconds(1).ToString("F0"));
        //}

        IsProcessing = false;
    }
}

[Serializable]
public class ProcessingOptionsMovieMovement : ProcessingOptionsBase
{
    [WarpSerializable] public decimal RangeMin { get; set; }
    [WarpSerializable] public decimal RangeMax { get; set; }
    [WarpSerializable] public decimal Bfactor { get; set; }
    [WarpSerializable] public int3 GridDims { get; set; }
    [WarpSerializable] public decimal DosePerAngstromFrame { get; set; }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((ProcessingOptionsMovieMovement)obj);
    }

    protected bool Equals(ProcessingOptionsMovieMovement other)
    {
        return base.Equals(other) &&
               RangeMin == other.RangeMin &&
               RangeMax == other.RangeMax &&
               Bfactor == other.Bfactor &&
               GridDims == other.GridDims &&
               DosePerAngstromFrame == other.DosePerAngstromFrame;
    }

    public static bool operator ==(ProcessingOptionsMovieMovement left, ProcessingOptionsMovieMovement right)
    {
        return Equals(left, right);
    }

    public static bool operator !=(ProcessingOptionsMovieMovement left, ProcessingOptionsMovieMovement right)
    {
        return !Equals(left, right);
    }
}