using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using Warp.Tools;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;
using static TorchSharp.NN.Losses;
using static TorchSharp.ScalarExtensionMethods;
using TorchSharp;
using ZLinq;

namespace Warp.NNModels
{
    public class CubeNetTorch
    {
        public float PixelSize = 10;

        public static readonly int3 DefaultDimensionsTrain = new int3(96);
        public static readonly int3 DefaultDimensionsValidTrain = new int3(64);
        public static readonly int3 DefaultDimensionsPredict = new int3(96);
        public static readonly int3 DefaultDimensionsValidPredict = new int3(64);

        public readonly int NClasses = 2;
        public readonly int3 BoxDimensions;
        public readonly int BatchSize = 4;
        public readonly int DeviceBatch = 4;
        public readonly int[] Devices;
        public readonly int NDevices;

        private UNet3D[] UNetModel;

        private TorchTensor[] TensorSource;
        private TorchTensor[] TensorTarget;
        private TorchTensor[] TensorClassWeights;

        private Loss[] Loss;
        private Optimizer Optimizer;

        private Image ResultPredictedLabel;
        private Image ResultPredictedScore;
        private float[] ResultLoss = new float[1];

        private bool IsDisposed = false;

        public CubeNetTorch(int nclasses, int3 boxDimensions, float[] classWeights, int[] devices, int batchSize = 4)
        {
            Devices = devices;
            NDevices = Devices.Length;

            NClasses = nclasses;
            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;

            UNetModel = new UNet3D[NDevices];
            TensorSource = new TorchTensor[NDevices];
            TensorTarget = new TorchTensor[NDevices];
            TensorClassWeights = new TorchTensor[NDevices];

            Loss = new Loss[NDevices];
            if (classWeights.Length != NClasses)
                throw new Exception();

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];

                UNetModel[i] = UNet3D(1, 1, 99999, 1, NClasses, true, true, true);
                UNetModel[i].ToCuda(DeviceID);

                TensorSource[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Z, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorTarget[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, NClasses, BoxDimensions.Z, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);

                TensorClassWeights[i] = Float32Tensor.Zeros(new long[] { NClasses }, DeviceType.CUDA, DeviceID);
                GPU.CopyHostToDevice(classWeights, TensorClassWeights[i].DataPtr(), NClasses);

                Loss[i] = CE(TensorClassWeights[i]);

            }, null);
            Optimizer = Optimizer.SGD(UNetModel[0].GetParameters(), 0.01, 0.9, false, 5e-4);

            ResultPredictedLabel = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BoxDimensions.Z * BatchSize));
            ResultPredictedScore = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BoxDimensions.Z * BatchSize));
        }

        private void ScatterData(Image src, TorchTensor[] dest)
        {
            src.GetDevice(Intent.Read);

            for (int i = 0; i < NDevices; i++)
                GPU.CopyDeviceToDevice(src.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       dest[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
        }

        private void SyncParams()
        {
            for (int i = 1; i < NDevices; i++)
                UNetModel[0].SynchronizeTo(UNetModel[i], Devices[i]);
        }

        private void GatherGrads()
        {
            for (int i = 1; i < NDevices; i++)
                UNetModel[0].GatherGrad(UNetModel[i]);
        }


        public void Predict(Image data, out Image predictionLabels, out Image predictionScores)
        {
            ScatterData(data, TensorSource);
            ResultPredictedLabel.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                UNetModel[i].Eval();

                using (TorchTensor Prediction = UNetModel[i].Forward(TensorSource[i]))
                using (TorchTensor PredictionArgMax = Prediction.Argmax(1))
                using (TorchTensor PredictionArgMaxFP = PredictionArgMax.ToType(ScalarType.Float32))
                using (TorchTensor PredictionSoftMax = Prediction.Softmax(1))
                {
                    GPU.CopyDeviceToDevice(PredictionArgMaxFP.DataPtr(),
                                           ResultPredictedLabel.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * BoxDimensions.Elements());

                    GPU.CopyDeviceToDevice(PredictionSoftMax.DataPtr(),
                                           ResultPredictedScore.GetDeviceSlice(i * DeviceBatch * NClasses, Intent.Write),
                                           DeviceBatch * BoxDimensions.Elements() * NClasses);
                }
            }, null);

            predictionLabels = ResultPredictedLabel;
            predictionScores = ResultPredictedScore;
        }

        public void Train(Image source,
                          Image target,
                          float learningRate,
                          bool needOutput,
                          out Image prediction,
                          out float[] loss)
        {
            GPU.CheckGPUExceptions();

            Optimizer.SetLearningRateSGD(learningRate);
            Optimizer.ZeroGrad();

            SyncParams();
            ResultPredictedLabel.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                UNetModel[i].Train();
                UNetModel[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTarget[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements() * NClasses);

                GPU.CheckGPUExceptions();

                using (TorchTensor TargetArgMax = TensorTarget[i].Argmax(1))
                using (TorchTensor Prediction = UNetModel[i].Forward(TensorSource[i]))
                using (TorchTensor PredictionLoss = Loss[i](Prediction, TargetArgMax))
                {
                    if (needOutput)
                    {
                        using (TorchTensor PredictionArgMax = Prediction.Argmax(1))
                        using (TorchTensor PredictionArgMaxFP = PredictionArgMax.ToType(ScalarType.Float32))
                        {
                            GPU.CopyDeviceToDevice(PredictionArgMaxFP.DataPtr(),
                                                   ResultPredictedLabel.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                                   DeviceBatch * (int)BoxDimensions.Elements());
                        }
                    }

                    if (i == 0)
                        GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLoss, 1);

                    PredictionLoss.Backward();
                }
            }, null);

            GatherGrads();

            if (NDevices > 1)
                UNetModel[0].ScaleGrad(1f / NDevices);

            Optimizer.Step();

            prediction = ResultPredictedLabel;
            loss = ResultLoss;
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Helper.PathToFolder(path));

            UNetModel[0].Save(path);
        }

        public void Load(string path)
        {
            for (int i = 0; i < NDevices; i++)
            {
                UNetModel[i].Load(path, DeviceType.CUDA, Devices[i]);
            }
        }

        ~CubeNetTorch()
        {
            Dispose();
        }

        public void Dispose()
        {
            lock (this)
            {
                if (!IsDisposed)
                {
                    IsDisposed = true;

                    ResultPredictedLabel.Dispose();
                    ResultPredictedScore.Dispose();

                    for (int i = 0; i < NDevices; i++)
                    {
                        TensorSource[i].Dispose();
                        TensorTarget[i].Dispose();
                        TensorClassWeights[i].Dispose();

                        UNetModel[i].Dispose();
                    }

                    Optimizer.Dispose();
                }
            }
        }

        public Image Segment(Image input, float threshold)
        {
            int3 Dims = input.Dims;
            int3 DimsWindow = BoxDimensions;

            int3 DimsValid = DimsWindow / 2;

            int3 DimsPositions = (Dims + DimsValid - 1) / DimsValid;
            float3 PositionStep = new float3(Dims - new int3(DimsWindow)) / new float3(Math.Max(DimsPositions.X - 1, 1),
                                                                                       Math.Max(DimsPositions.Y - 1, 1),
                                                                                       Math.Max(DimsPositions.Z - 1, 1));

            int NPositions = (int)DimsPositions.Elements();

            int3[] Positions = new int3[NPositions];
            for (int p = 0; p < NPositions; p++)
            {
                int X = p % DimsPositions.X;
                int Y = (p % (int)DimsPositions.ElementsSlice()) / DimsPositions.X;
                int Z = p / (int)DimsPositions.ElementsSlice();
                Positions[p] = new int3((int)(X * PositionStep.X + DimsWindow.X / 2),
                                        (int)(Y * PositionStep.Y + DimsWindow.Y / 2),
                                        (int)(Z * PositionStep.Z + DimsWindow.Z / 2));
            }

            float[][] PredictionTiles = new float[Positions.Length][];

            Image Extracted = new Image(new int3(DimsWindow.X, DimsWindow.Y, DimsWindow.Z * BatchSize));
            float[] TileScores = new float[DimsWindow.Elements() * NClasses];
            int TileElements = (int)DimsWindow.Elements();

            for (int ib = 0; ib < (Positions.Length + BatchSize - 1) / BatchSize; ib++)
            {
                int b = ib * BatchSize;
                int CurBatch = Math.Min(BatchSize, Positions.Length - b);

                int3[] CurPositions = Positions.Skip(b).Take(CurBatch).ToArray();
                GPU.Extract(input.GetDevice(Intent.Read),
                            Extracted.GetDevice(Intent.Write),
                            input.Dims,
                            new int3(DimsWindow),
                            Helper.ToInterleaved(CurPositions.Select(p => p - new int3(DimsWindow / 2)).ToArray()),
                            false,
                            (uint)CurBatch);

                Image PredictionLabels = null;
                Image PredictionScores = null;
                Predict(Extracted, out PredictionLabels, out PredictionScores);

                for (int bb = 0; bb < CurBatch; bb++)
                {
                    float[] TileData = new float[DimsWindow.Elements()];
                    GPU.CopyDeviceToHost(PredictionLabels.GetDeviceSlice(bb * DimsWindow.Z, Intent.Read),
                                         TileData,
                                         TileData.Length);

                    GPU.CopyDeviceToHost(PredictionScores.GetDeviceSlice(bb * DimsWindow.Z * NClasses, Intent.Read),
                                         TileScores,
                                         TileScores.Length);

                    for (int i = 0; i < TileData.Length; i++)
                    {
                        int Label = (int)(TileData[i] + 0.5f);
                        float Score = TileScores[Label * TileElements + i];
                        if (Score < threshold)
                            TileData[i] = 0;
                    }

                    PredictionTiles[b + bb] = TileData;
                }
            }

            Extracted.Dispose();

            input.FreeDevice();

            Image Segmented = new Image(input.Dims);
            float[][] SegmentedData = Segmented.GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int ClosestX = (int)Math.Max(0, Math.Min(DimsPositions.X - 1, (int)(((float)x - DimsWindow.X / 2) / PositionStep.X + 0.5f)));
                        int ClosestY = (int)Math.Max(0, Math.Min(DimsPositions.Y - 1, (int)(((float)y - DimsWindow.Y / 2) / PositionStep.Y + 0.5f)));
                        int ClosestZ = (int)Math.Max(0, Math.Min(DimsPositions.Z - 1, (int)(((float)z - DimsWindow.Z / 2) / PositionStep.Z + 0.5f)));
                        int ClosestID = (ClosestZ * DimsPositions.Y + ClosestY) * DimsPositions.X + ClosestX;

                        int3 Position = Positions[ClosestID];
                        int LocalX = Math.Max(0, Math.Min(DimsWindow.X - 1, x - Position.X + DimsWindow.X / 2));
                        int LocalY = Math.Max(0, Math.Min(DimsWindow.Y - 1, y - Position.Y + DimsWindow.Y / 2));
                        int LocalZ = Math.Max(0, Math.Min(DimsWindow.Z - 1, z - Position.Z + DimsWindow.Z / 2));

                        SegmentedData[z][y * Dims.X + x] = PredictionTiles[ClosestID][(LocalZ * DimsWindow.Y + LocalY) * DimsWindow.X + LocalX];
                    }
                }
            }

            return Segmented;
        }


        public float4[] Match(Image segmentation, float diameterPixels)
        {
            int3 DimsRegionBN = DefaultDimensionsPredict;
            int3 DimsRegionValidBN = DefaultDimensionsValidPredict;
            int BorderBN = (DimsRegionBN.X - DimsRegionValidBN.X) / 2;

            int3 DimsBN = segmentation.Dims;

            #region Apply Gaussian and find peaks


            //Image PredictionsConvolved = PredictionsImage.AsConvolvedGaussian((float)options.ExpectedDiameter / PixelSizeBN / 6);
            //PredictionsConvolved.Multiply(PredictionsImage);
            //PredictionsImage.Dispose();

            //PredictionsImage.WriteMRC(MatchingDir + RootName + "_boxnet.mrc", PixelSizeBN, true);

            int3[] Peaks = segmentation.GetLocalPeaks((int)(diameterPixels / 4 + 0.5f), 1e-6f);
            segmentation.FreeDevice();

            float[][] PredictionsData = segmentation.GetHost(Intent.Read);

            int BorderDist = (int)(diameterPixels * 0.8f + 0.5f);
            Peaks = Peaks.Where(p => p.X > BorderDist &&
                                     p.Y > BorderDist &&
                                     p.Z > BorderDist &&
                                     p.X < DimsBN.X - BorderDist &&
                                     p.Y < DimsBN.Y - BorderDist &&
                                     p.Z < DimsBN.Z - BorderDist).ToArray();

            #endregion

            #region Label connected components and get centroids

            List<float3> Centroids;
            List<int> Extents;
            {
                List<List<int3>> Components = new List<List<int3>>();
                int[][] PixelLabels = Helper.ArrayOfFunction(i => Helper.ArrayOfConstant(-1, PredictionsData[i].Length), PredictionsData.Length);

                foreach (var peak in Peaks)
                {
                    if (PixelLabels[peak.Z][peak.Y * DimsBN.X + peak.X] >= 0)
                        continue;

                    List<int3> Component = new List<int3>() { peak };
                    int CN = Components.Count;

                    PixelLabels[peak.Z][peak.Y * DimsBN.X + peak.X] = CN;
                    Queue<int3> Expansion = new Queue<int3>(100);
                    Expansion.Enqueue(peak);

                    while (Expansion.Count > 0)
                    {
                        int3 pos = Expansion.Dequeue();
                        int PosElement = pos.Y * DimsBN.X + pos.X;

                        if (pos.X > 0 && PredictionsData[pos.Z][PosElement - 1] > 0 && PixelLabels[pos.Z][PosElement - 1] < 0)
                        {
                            PixelLabels[pos.Z][PosElement - 1] = CN;
                            Component.Add(pos + new int3(-1, 0, 0));
                            Expansion.Enqueue(pos + new int3(-1, 0, 0));
                        }
                        if (pos.X < DimsBN.X - 1 && PredictionsData[pos.Z][PosElement + 1] > 0 && PixelLabels[pos.Z][PosElement + 1] < 0)
                        {
                            PixelLabels[pos.Z][PosElement + 1] = CN;
                            Component.Add(pos + new int3(1, 0, 0));
                            Expansion.Enqueue(pos + new int3(1, 0, 0));
                        }

                        if (pos.Y > 0 && PredictionsData[pos.Z][PosElement - DimsBN.X] > 0 && PixelLabels[pos.Z][PosElement - DimsBN.X] < 0)
                        {
                            PixelLabels[pos.Z][PosElement - DimsBN.X] = CN;
                            Component.Add(pos + new int3(0, -1, 0));
                            Expansion.Enqueue(pos + new int3(0, -1, 0));
                        }
                        if (pos.Y < DimsBN.Y - 1 && PredictionsData[pos.Z][PosElement + DimsBN.X] > 0 && PixelLabels[pos.Z][PosElement + DimsBN.X] < 0)
                        {
                            PixelLabels[pos.Z][PosElement + DimsBN.X] = CN;
                            Component.Add(pos + new int3(0, 1, 0));
                            Expansion.Enqueue(pos + new int3(0, 1, 0));
                        }

                        if (pos.Z > 0 && PredictionsData[pos.Z - 1][PosElement] > 0 && PixelLabels[pos.Z - 1][PosElement] < 0)
                        {
                            PixelLabels[pos.Z - 1][PosElement] = CN;
                            Component.Add(pos + new int3(0, 0, -1));
                            Expansion.Enqueue(pos + new int3(0, 0, -1));
                        }
                        if (pos.Z < DimsBN.Z - 1 && PredictionsData[pos.Z + 1][PosElement] > 0 && PixelLabels[pos.Z + 1][PosElement] < 0)
                        {
                            PixelLabels[pos.Z + 1][PosElement] = CN;
                            Component.Add(pos + new int3(0, 0, 1));
                            Expansion.Enqueue(pos + new int3(0, 0, 1));
                        }
                    }

                    Components.Add(Component);
                }

                Centroids = Components.Select(c => MathHelper.Mean(c.Select(v => new float3(v)).ToArray())).ToList();
                Extents = Components.Select(c => c.Count).ToList();
            }

            List<int> ToDelete = new List<int>();

            // Hit test with crap mask
            //for (int c1 = 0; c1 < Centroids.Count; c1++)
            //{
            //    float2 P1 = Centroids[c1];
            //    if (MaskHitTest[(int)P1.Y * DimsBN.X + (int)P1.X] == 0)
            //    {
            //        ToDelete.Add(c1);
            //        continue;
            //    }
            //}

            for (int c1 = 0; c1 < Centroids.Count - 1; c1++)
            {
                float3 P1 = Centroids[c1];

                for (int c2 = c1 + 1; c2 < Centroids.Count; c2++)
                {
                    if ((P1 - Centroids[c2]).Length() < diameterPixels / 1.5f)
                    {
                        int D = Extents[c1] < Extents[c2] ? c1 : c2;

                        if (!ToDelete.Contains(D))
                            ToDelete.Add(D);
                    }
                }
            }

            ToDelete.Sort();
            for (int i = ToDelete.Count - 1; i >= 0; i--)
            {
                Centroids.RemoveAt(ToDelete[i]);
                Extents.RemoveAt(ToDelete[i]);
            }

            #endregion

            //new Image(Predictions, DimsBN).WriteMRC("d_predictions.mrc", true);

            #region Write peak positions and angles into table

            return Centroids.Select((c, i) => new float4(c.X, c.Y, c.Z, Extents[i])).ToArray();

            #endregion
        }
    }
}
