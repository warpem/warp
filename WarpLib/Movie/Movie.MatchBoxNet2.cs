using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class Movie
{
    public void MatchBoxNet2(BoxNetTorch[] networks, ProcessingOptionsBoxNet options, Func<int3, int, string, bool> progressCallback)
    {
        Stopwatch WatchPreflight = new Stopwatch();
        WatchPreflight.Start();

        IsProcessing = true;

        Directory.CreateDirectory(MatchingDir);

        Image Average = Image.FromFile(string.IsNullOrEmpty(options.OverrideImagePath) ? AveragePath : options.OverrideImagePath);
        float AveragePixelSize = Average.PixelSize;

        float PixelSizeBN = BoxNetTorch.PixelSize;
        int BatchSizeBN = networks[0].BatchSize;
        int2 DimsRegionBN = BoxNetTorch.DefaultDimensionsPredict;
        int2 DimsRegionValidBN = BoxNetTorch.DefaultDimensionsValidPredict;
        int BorderBN = (DimsRegionBN.X - DimsRegionValidBN.X) / 2;

        int2 DimsBN = (new int2(Average.Dims * AveragePixelSize / PixelSizeBN) + 1) / 2 * 2;

        //Image SoftMask = new Image(new int3(DimsBN.X * 2, DimsBN.Y * 2, 1));
        //SoftMask.TransformValues((x, y, z, v) =>
        //{
        //    float xx = (float)Math.Max(0, Math.Max(DimsBN.X / 2 - x, x - DimsBN.X * 3 / 2)) / (DimsBN.X / 2);
        //    float yy = (float)Math.Max(0, Math.Max(DimsBN.Y / 2 - y, y - DimsBN.Y * 3 / 2)) / (DimsBN.Y / 2);

        //    float r = Math.Min(1, (float)Math.Sqrt(xx * xx + yy * yy));
        //    float w = ((float)Math.Cos(r * Math.PI) + 1) * 0.5f;

        //    return w;
        //});

        Image AverageBN = Average.AsScaled(DimsBN).AndDisposeParent();
        //AverageBN.Normalize();

        AverageBN = AverageBN.AsPadded(DimsBN - 2).AndDisposeParent();
        Image AverageBNPadded = AverageBN.AsPaddedClamped(DimsBN * 2).AndDisposeParent();
        //AverageBNPadded.MultiplySlices(SoftMask);
        AverageBNPadded.MaskRectangularly(new int3(DimsBN), Math.Min(DimsBN.X, DimsBN.Y) / 2, false);
        AverageBNPadded.Bandpass(2f * PixelSizeBN / 500f, 1, false, 2f * PixelSizeBN / 500f);
        //SoftMask.Dispose();

        AverageBN = AverageBNPadded.AsPadded(DimsBN).AndDisposeParent();

        float2 MeanStd = new float2(0, 1);
        {
            Image AverageCenter = AverageBN.AsPadded(DimsBN / 2);
            MeanStd = MathHelper.MeanAndStd(AverageCenter.GetHost(Intent.Read)[0]);
            AverageCenter.Dispose();
        }

        MeanStd.Y = Math.Max(1e-16f, MeanStd.Y);
        //AverageBN.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
        AverageBN.Add(-MeanStd.X);
        AverageBN.Multiply(1f / MeanStd.Y * (options.PickingInvert ? -1 : 1));

        //GPU.Normalize(AverageBN.GetDevice(Intent.Read),
        //              AverageBN.GetDevice(Intent.Write),
        //              (uint)AverageBN.ElementsSliceReal,
        //              1);

        //if (options.PickingInvert)
        //    AverageBN.Multiply(-1f);

        int2 Margin = (BoxNetTorch.DefaultDimensionsPredict - BoxNetTorch.DefaultDimensionsValidPredict) / 2;
        DimsBN = DimsBN + Margin * 2;
        AverageBN = AverageBN.AsPadded(DimsBN).AndDisposeParent();

        float[] Predictions = new float[DimsBN.Elements()];
        float[] Mask = new float[DimsBN.Elements()];

        WatchPreflight.Stop();
        Debug.WriteLine("Preflight: " + WatchPreflight.ElapsedMilliseconds / 1000.0);

        {
            int2 DimsPositions = (DimsBN + DimsRegionValidBN - 1) / DimsRegionValidBN;
            float2 PositionStep = new float2(DimsBN - DimsRegionBN) / new float2(Math.Max(DimsPositions.X - 1, 1),
                Math.Max(DimsPositions.Y - 1, 1));

            int NPositions = (int)DimsPositions.Elements();

            int3[] Positions = new int3[NPositions];
            for (int p = 0; p < NPositions; p++)
            {
                int X = p % DimsPositions.X;
                int Y = p / DimsPositions.X;
                Positions[p] = new int3((int)(X * PositionStep.X + DimsRegionBN.X / 2),
                    (int)(Y * PositionStep.Y + DimsRegionBN.Y / 2),
                    0);
            }

            float[][] PredictionTiles = Helper.ArrayOfFunction(i => new float[DimsRegionBN.Elements()], NPositions);
            float[][] MaskTiles = Helper.ArrayOfFunction(i => new float[DimsRegionBN.Elements()], NPositions);

            int3[][] BatchedPositions = new int3[(NPositions + BatchSizeBN - 1) / BatchSizeBN][];
            for (int i = 0; i < BatchedPositions.Length; i++)
            {
                int3[] BatchPositions = new int3[BatchSizeBN];
                for (int j = 0; j < BatchSizeBN; j++)
                    BatchPositions[j] = Positions[Math.Min(NPositions - 1, i * BatchSizeBN + j)] - new int3(DimsRegionBN.X / 2, DimsRegionBN.Y / 2, 0);

                BatchedPositions[i] = BatchPositions;
            }

            Image Extracted = new Image(IntPtr.Zero, new int3(DimsRegionBN.X, DimsRegionBN.Y, BatchSizeBN));
            int DeviceID = GPU.GetDevice();

            int BatchesDone = 0;
            float Threshold = (float)options.MinimumScore;

            lock(networks[0])
            {
                Stopwatch WatchInference = new Stopwatch();
                WatchInference.Start();

                for (int b = 0; b < BatchedPositions.Length; b++)
                {
                    #region Extract and normalize windows

                    GPU.Extract(AverageBN.GetDevice(Intent.Read),
                        Extracted.GetDevice(Intent.Write),
                        AverageBN.Dims,
                        new int3(DimsRegionBN),
                        Helper.ToInterleaved(BatchedPositions[b]),
                        true,
                        (uint)BatchSizeBN);

                    #endregion

                    //Extracted[threadID].WriteMRC("d_extracted.mrc", true);

                    #region Predict

                    float[] BatchArgMax = null;
                    float[] BatchProbability = null;
                    networks[0].Predict(Extracted,
                        out BatchArgMax,
                        out BatchProbability);

                    //new Image(BatchArgMax.Select(v => (float)v).ToArray(), new int3(DimsRegionBN)).WriteMRC("d_labels.mrc", true);

                    int SliceElements = (int)DimsRegionBN.Elements();
                    for (int ib = 0; ib < BatchSizeBN; ib++)
                    {
                        if (b * BatchSizeBN + ib >= NPositions)
                            break;

                        for (int i = 0; i < SliceElements; i++)
                        {
                            int Label = (int)BatchArgMax[ib * SliceElements + i];
                            float Probability = BatchProbability[ib * SliceElements * 3 + Label * SliceElements + i];

                            PredictionTiles[b * BatchSizeBN + ib][i] = (Label == 1 && Probability >= Threshold ? Probability : 0);
                            MaskTiles[b * BatchSizeBN + ib][i] = Label == 2 && Probability >= 0.0f ? 1 : 0;
                        }
                    }

                    #endregion

                    progressCallback?.Invoke(new int3(NPositions, 1, 1), ++BatchesDone, "");
                }

                WatchInference.Stop();
                Debug.WriteLine("Inference: " + WatchInference.ElapsedMilliseconds / 1000.0);
            }

            AverageBN.FreeDevice();
            Extracted.Dispose();

            Parallel.For(0, DimsBN.Y, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, y =>
            {
                for (int x = 0; x < DimsBN.X; x++)
                {
                    int ClosestX = (int)Math.Max(0, Math.Min(DimsPositions.X - 1, (int)(((float)x - DimsRegionBN.X / 2) / PositionStep.X + 0.5f)));
                    int ClosestY = (int)Math.Max(0, Math.Min(DimsPositions.Y - 1, (int)(((float)y - DimsRegionBN.Y / 2) / PositionStep.Y + 0.5f)));
                    int ClosestID = ClosestY * DimsPositions.X + ClosestX;

                    int3 Position = Positions[ClosestID];
                    int LocalX = Math.Max(0, Math.Min(DimsRegionBN.X - 1, x - Position.X + DimsRegionBN.X / 2));
                    int LocalY = Math.Max(0, Math.Min(DimsRegionBN.Y - 1, y - Position.Y + DimsRegionBN.Y / 2));

                    Predictions[y * DimsBN.X + x] = PredictionTiles[ClosestID][LocalY * DimsRegionBN.X + LocalX];
                    Mask[y * DimsBN.X + x] = MaskTiles[ClosestID][LocalY * DimsRegionBN.X + LocalX];
                }
            });
        }


        AverageBN.Dispose();

        #region Rescale and save mask

        Image MaskImage = new Image(Mask, new int3(DimsBN));
        DimsBN -= Margin * 2;
        MaskImage = MaskImage.AsPadded(DimsBN).AndDisposeParent();

        // Get rid of all connected components in the mask that are too small
        {
            float[] MaskData = MaskImage.GetHost(Intent.ReadWrite)[0];

            List<List<int2>> Components = new List<List<int2>>();
            int[] PixelLabels = Helper.ArrayOfConstant(-1, MaskData.Length);

            for (int y = 0; y < DimsBN.Y; y++)
            {
                for (int x = 0; x < DimsBN.X; x++)
                {
                    int2 peak = new int2(x, y);

                    if (MaskData[DimsBN.ElementFromPosition(peak)] != 1 || PixelLabels[DimsBN.ElementFromPosition(peak)] >= 0)
                        continue;

                    List<int2> Component = new List<int2>() { peak };
                    int CN = Components.Count;

                    PixelLabels[DimsBN.ElementFromPosition(peak)] = CN;
                    Queue<int2> Expansion = new Queue<int2>(100);
                    Expansion.Enqueue(peak);

                    while(Expansion.Count > 0)
                    {
                        int2 pos = Expansion.Dequeue();
                        int PosElement = DimsBN.ElementFromPosition(pos);

                        if (pos.X > 0 && MaskData[PosElement - 1] == 1 && PixelLabels[PosElement - 1] < 0)
                        {
                            PixelLabels[PosElement - 1] = CN;
                            Component.Add(pos + new int2(-1, 0));
                            Expansion.Enqueue(pos + new int2(-1, 0));
                        }

                        if (pos.X < DimsBN.X - 1 && MaskData[PosElement + 1] == 1 && PixelLabels[PosElement + 1] < 0)
                        {
                            PixelLabels[PosElement + 1] = CN;
                            Component.Add(pos + new int2(1, 0));
                            Expansion.Enqueue(pos + new int2(1, 0));
                        }

                        if (pos.Y > 0 && MaskData[PosElement - DimsBN.X] == 1 && PixelLabels[PosElement - DimsBN.X] < 0)
                        {
                            PixelLabels[PosElement - DimsBN.X] = CN;
                            Component.Add(pos + new int2(0, -1));
                            Expansion.Enqueue(pos + new int2(0, -1));
                        }

                        if (pos.Y < DimsBN.Y - 1 && MaskData[PosElement + DimsBN.X] == 1 && PixelLabels[PosElement + DimsBN.X] < 0)
                        {
                            PixelLabels[PosElement + DimsBN.X] = CN;
                            Component.Add(pos + new int2(0, 1));
                            Expansion.Enqueue(pos + new int2(0, 1));
                        }
                    }

                    Components.Add(Component);
                }
            }

            foreach (var component in Components)
                if (component.Count < 20)
                    foreach (var pos in component)
                        MaskData[DimsBN.ElementFromPosition(pos)] = 0;

            MaskPercentage = (decimal)MaskData.Sum() / MaskData.Length * 100;
        }

        Image MaskImage8 = MaskImage.AsScaled(new int2(new float2(DimsBN) * BoxNetTorch.PixelSize / 8) / 2 * 2);
        MaskImage8.Binarize(0.5f);

        int MaxHitTestDistance = (int)((options.ExpectedDiameter / 2 + options.MinimumMaskDistance) / (decimal)BoxNetTorch.PixelSize) + 2;
        Image MaskDistance = MaskImage.AsDistanceMapExact(MaxHitTestDistance);
        MaskImage.Dispose();
        MaskDistance.Binarize(MaxHitTestDistance - 2);
        float[] MaskHitTest = MaskDistance.GetHostContinuousCopy();
        MaskDistance.Dispose();

        Directory.CreateDirectory(MaskDir);
        MaskImage8.WriteTIFF(MaskPath, 8, typeof(float));
        MaskImage8.Dispose();

        #endregion

        #region Find peaks

        Image PredictionsImage = new Image(Predictions, new int3(DimsBN + Margin * 2));
        PredictionsImage = PredictionsImage.AsPadded(DimsBN).AndDisposeParent();

        //Image PredictionsConvolved = PredictionsImage.AsConvolvedGaussian((float)options.ExpectedDiameter / PixelSizeBN / 6);
        //PredictionsConvolved.Multiply(PredictionsImage);
        //PredictionsImage.Dispose();

        //PredictionsImage.WriteMRC(MatchingDir + RootName + "_boxnet.mrc", PixelSizeBN, true);

        int3[] Peaks = PredictionsImage.GetLocalPeaks((int)((float)options.ExpectedDiameter / PixelSizeBN / 4 + 0.5f), 1e-6f);
        PredictionsImage.Dispose();

        int BorderDist = (int)((float)options.ExpectedDiameter / PixelSizeBN * 0.8f + 0.5f);
        Peaks = Peaks.Where(p => p.X > BorderDist && p.Y > BorderDist && p.X < DimsBN.X - BorderDist && p.Y < DimsBN.Y - BorderDist).ToArray();

        #endregion

        #region Label connected components and get centroids

        List<float2> Centroids;
        int[] Extents;
        {
            List<List<int2>> Components = new List<List<int2>>();
            int[] PixelLabels = Helper.ArrayOfConstant(-1, Predictions.Length);

            foreach (var peak in Peaks.Select(v => new int2(v)))
            {
                if (PixelLabels[DimsBN.ElementFromPosition(peak)] >= 0)
                    continue;

                List<int2> Component = new List<int2>() { peak };
                int CN = Components.Count;

                PixelLabels[DimsBN.ElementFromPosition(peak)] = CN;
                Queue<int2> Expansion = new Queue<int2>(100);
                Expansion.Enqueue(peak);

                while(Expansion.Count > 0)
                {
                    int2 pos = Expansion.Dequeue();
                    int PosElement = DimsBN.ElementFromPosition(pos);

                    if (pos.X > 0 && Predictions[PosElement - 1] > 0 && PixelLabels[PosElement - 1] < 0)
                    {
                        PixelLabels[PosElement - 1] = CN;
                        Component.Add(pos + new int2(-1, 0));
                        Expansion.Enqueue(pos + new int2(-1, 0));
                    }

                    if (pos.X < DimsBN.X - 1 && Predictions[PosElement + 1] > 0 && PixelLabels[PosElement + 1] < 0)
                    {
                        PixelLabels[PosElement + 1] = CN;
                        Component.Add(pos + new int2(1, 0));
                        Expansion.Enqueue(pos + new int2(1, 0));
                    }

                    if (pos.Y > 0 && Predictions[PosElement - DimsBN.X] > 0 && PixelLabels[PosElement - DimsBN.X] < 0)
                    {
                        PixelLabels[PosElement - DimsBN.X] = CN;
                        Component.Add(pos + new int2(0, -1));
                        Expansion.Enqueue(pos + new int2(0, -1));
                    }

                    if (pos.Y < DimsBN.Y - 1 && Predictions[PosElement + DimsBN.X] > 0 && PixelLabels[PosElement + DimsBN.X] < 0)
                    {
                        PixelLabels[PosElement + DimsBN.X] = CN;
                        Component.Add(pos + new int2(0, 1));
                        Expansion.Enqueue(pos + new int2(0, 1));
                    }
                }

                Components.Add(Component);
            }

            Centroids = Components.Select(c => MathHelper.Mean(c.Select(v => new float2(v)).ToArray())).ToList();
            Extents = Components.Select(c => c.Count).ToArray();
        }

        List<int> ToDelete = new List<int>();

        // Hit test with crap mask
        for (int c1 = 0; c1 < Centroids.Count; c1++)
        {
            float2 P1 = Centroids[c1];
            if (MaskHitTest[(int)P1.Y * DimsBN.X + (int)P1.X] == 0)
            {
                ToDelete.Add(c1);
                continue;
            }
        }

        for (int c1 = 0; c1 < Centroids.Count - 1; c1++)
        {
            float2 P1 = Centroids[c1];

            for (int c2 = c1 + 1; c2 < Centroids.Count; c2++)
            {
                if ((P1 - Centroids[c2]).Length() < (float)options.ExpectedDiameter / PixelSizeBN / 1.5f)
                {
                    int D = Extents[c1] < Extents[c2] ? c1 : c2;

                    if (!ToDelete.Contains(D))
                        ToDelete.Add(D);
                }
            }
        }

        ToDelete.Sort();
        for (int i = ToDelete.Count - 1; i >= 0; i--)
            Centroids.RemoveAt(ToDelete[i]);

        #endregion

        #region Write peak positions into table

        if (Centroids.Any())
        {
            Star TableOut = new Star(new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnMicrographName",
                "rlnAutopickFigureOfMerit"
            });

            {
                foreach (float2 peak in Centroids)
                {
                    float2 Position = peak * PixelSizeBN / AveragePixelSize;
                    float Score = Predictions[DimsBN.ElementFromPosition(new int2(peak))];

                    TableOut.AddRow(new string[]
                    {
                        Position.X.ToString(CultureInfo.InvariantCulture),
                        Position.Y.ToString(CultureInfo.InvariantCulture),
                        RootName + ".mrc",
                        Score.ToString(CultureInfo.InvariantCulture)
                    });
                }
            }

            string StarSuffix = string.IsNullOrEmpty(options.OverrideStarSuffix)
                ? Helper.PathToName(options.ModelName)
                : options.OverrideStarSuffix;

            TableOut.Save(System.IO.Path.Combine(MatchingDir, RootName + "_" + StarSuffix + ".star"));
            UpdateParticleCount("_" + StarSuffix);
        }

        #endregion

        OptionsBoxNet = options;
        SaveMeta();

        IsProcessing = false;
    }
}

[Serializable]
public class ProcessingOptionsBoxNet : ProcessingOptionsBase
{
    [WarpSerializable] public string ModelName { get; set; }
    [WarpSerializable] public bool OverwriteFiles { get; set; }
    [WarpSerializable] public bool PickingInvert { get; set; }
    [WarpSerializable] public decimal ExpectedDiameter { get; set; }
    [WarpSerializable] public decimal MinimumScore { get; set; }
    [WarpSerializable] public decimal MinimumMaskDistance { get; set; }
    [WarpSerializable] public bool ExportParticles { get; set; }
    [WarpSerializable] public int ExportBoxSize { get; set; }
    [WarpSerializable] public bool ExportInvert { get; set; }
    [WarpSerializable] public bool ExportNormalize { get; set; }
    [WarpSerializable] public string OverrideImagePath { get; set; }
    [WarpSerializable] public string OverrideStarSuffix { get; set; }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((ProcessingOptionsBoxNet)obj);
    }

    protected bool Equals(ProcessingOptionsBoxNet other)
    {
        return ModelName == other.ModelName &&
               OverwriteFiles == other.OverwriteFiles &&
               PickingInvert == other.PickingInvert &&
               ExpectedDiameter == other.ExpectedDiameter &&
               MinimumScore == other.MinimumScore &&
               ExportParticles == other.ExportParticles &&
               ExportBoxSize == other.ExportBoxSize &&
               ExportInvert == other.ExportInvert &&
               ExportNormalize == other.ExportNormalize &&
               OverrideImagePath == other.OverrideImagePath;
    }

    public static bool operator ==(ProcessingOptionsBoxNet left, ProcessingOptionsBoxNet right)
    {
        return Equals(left, right);
    }

    public static bool operator !=(ProcessingOptionsBoxNet left, ProcessingOptionsBoxNet right)
    {
        return !Equals(left, right);
    }
}