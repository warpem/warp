using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

using Warp.Tools;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;
using static TorchSharp.NN.Losses;
using static TorchSharp.ScalarExtensionMethods;
using TorchSharp;
using ZLinq;

namespace Warp
{
    public class NoiseNet2DTorch : IDisposable
    {
        public static readonly float PixelSize = 5;

        public readonly int2 BoxDimensions;
        public readonly int BatchSize = 8;
        public readonly int DeviceBatch = 8;
        public readonly int[] Devices;
        public readonly int NDevices;

        private UNet2D[] UNetModel;

        private TorchTensor[] TensorSource;
        private TorchTensor[] TensorTarget;
        private TorchTensor[] TensorCTF;
        private TorchTensor[] TensorMask;

        private Loss Loss;
        private Optimizer Optimizer;

        private Image ResultPredicted;
        private Image ResultPredictedDeconv;
        private float[] ResultLoss = new float[1], ResultLossTwisted = new float[1];

        private bool IsDisposed = false;

        public NoiseNet2DTorch(int2 boxDimensions, int[] devices, int batchSize = 8)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;

            UNetModel = new UNet2D[NDevices];
            TensorSource = new TorchTensor[NDevices];
            TensorTarget = new TorchTensor[NDevices];
            TensorCTF = new TorchTensor[NDevices];
            TensorMask = new TorchTensor[NDevices];

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];

                UNetModel[i] = UNet2D(1, 1, 1, 1, 1, false, false);
                UNetModel[i].ToCuda(DeviceID);

                TensorSource[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorTarget[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorCTF[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y * 2, BoxDimensions.X * 2 / 2 + 1 }, DeviceType.CUDA, DeviceID);
                TensorMask[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y * 2, BoxDimensions.X * 2 }, DeviceType.CUDA, DeviceID);

                {
                    int2 BoxDim2 = BoxDimensions * 2;
                    int2 Margin = (BoxDim2 - BoxDimensions) / 2;
                    float[] MaskData = new float[BoxDim2.Elements()];
                    for (int y = 0; y < BoxDim2.Y; y++)
                    {
                        float yy = Math.Max(Margin.Y - y, y - (Margin.Y + BoxDimensions.Y)) / (float)Margin.Y;
                        yy = Math.Max(0, Math.Min(1, yy));

                        for (int x = 0; x < BoxDim2.X; x++)
                        {
                            float xx = Math.Max(Margin.X - x, x - (Margin.X + BoxDimensions.X)) / (float)Margin.X;
                            xx = Math.Max(0, Math.Min(1, xx));

                            float r = Math.Min(1, (float)Math.Sqrt(yy * yy + xx * xx));
                            float v = (float)Math.Cos(r * Math.PI) * 0.5f + 0.5f;

                            MaskData[y * BoxDim2.X + x] = v;
                        }
                    }

                    GPU.CopyHostToDevice(MaskData, TensorMask[i].DataPtr(), MaskData.Length);
                }
            }, null);

            Loss = MSE(Reduction.Mean);
            Optimizer = Optimizer.Adam(UNetModel[0].GetParameters(), 0.01, 1e-4);

            ResultPredicted = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
            ResultPredictedDeconv = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
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


        public void Predict(Image data, out Image prediction)
        {
            ScatterData(data, TensorSource);
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                using (var mode = new InferenceMode(true))
                {
                    UNetModel[i].Eval();

                    using (TorchTensor Prediction = UNetModel[i].Forward(TensorSource[i]))
                        GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                               ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * BoxDimensions.Elements());
                }
            }, null);

            prediction = ResultPredicted;
        }

        public void Train(Image source,
                          Image target,
                          float learningRate,
                          out Image prediction,
                          out float[] loss)
        {
            Optimizer.SetLearningRateAdam(learningRate);
            Optimizer.ZeroGrad();

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                UNetModel[i].Train();
                UNetModel[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTarget[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());

                using (TorchTensor Prediction = UNetModel[i].Forward(TensorSource[i]))
                using (TorchTensor PredictionLoss = Loss(Prediction, TensorTarget[i]))
                {

                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    if (i == 0)
                        GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLoss, 1);

                    PredictionLoss.Backward();
                }
            }, null);

            GatherGrads();

            if (NDevices > 1)
                UNetModel[0].ScaleGrad(1f / NDevices);

            Optimizer.Step();

            prediction = ResultPredicted;
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
                //UNetModel[i].ToCuda(Devices[i]);
            }
        }

        ~NoiseNet2DTorch()
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

                    ResultPredicted.Dispose();
                    ResultPredictedDeconv.Dispose();

                    for (int i = 0; i < NDevices; i++)
                    {
                        TensorSource[i].Dispose();
                        TensorTarget[i].Dispose();
                        TensorCTF[i].Dispose();

                        UNetModel[i].Dispose();
                    }

                    Optimizer.Dispose();
                }
            }
        }

        public static void Denoise(Image noisy, NoiseNet2DTorch[] networks)
        {
            int3 Dims = noisy.Dims;
            int Dim = networks[0].BoxDimensions.X;
            int BatchSize = networks[0].BatchSize;

            int2 DimsValid = new int2(Dim) / 2;

            int2 DimsPositions = (new int2(Dims) + DimsValid - 1) / DimsValid;
            float2 PositionStep = new float2(new int2(Dims) - new int2(Dim)) / new float2(Math.Max(DimsPositions.X - 1, 1),
                                                                                          Math.Max(DimsPositions.Y - 1, 1));

            int NPositions = (int)DimsPositions.Elements();

            int3[] Positions = new int3[NPositions];
            for (int p = 0; p < NPositions; p++)
            {
                int X = p % DimsPositions.X;
                int Y = p / DimsPositions.X;
                Positions[p] = new int3((int)(X * PositionStep.X + Dim / 2),
                                        (int)(Y * PositionStep.Y + Dim / 2),
                                        0);
            }

            float[][] PredictionTiles = new float[Positions.Length][];

            Image Extracted = new Image(new int3(Dim, Dim, BatchSize));

            for (int b = 0; b < Positions.Length; b += BatchSize)
            {
                int CurBatch = Math.Min(BatchSize, Positions.Length - b);

                int3[] CurPositions = Positions.Skip(b).Take(CurBatch).ToArray();
                GPU.Extract(noisy.GetDevice(Intent.Read),
                            Extracted.GetDevice(Intent.Write),
                            noisy.Dims,
                            new int3(Dim, Dim, 1),
                            Helper.ToInterleaved(CurPositions.Select(p => p - new int3(Dim / 2, Dim / 2, 0)).ToArray()),
                            false,
                            (uint)CurBatch);

                Image Prediction = null;
                networks[0].Predict(Extracted, out Prediction);

                for (int i = 0; i < CurBatch; i++)
                    PredictionTiles[b + i] = Prediction.GetHost(Intent.Read)[i].ToArray();
            }

            Extracted.Dispose();

            float[] Denoised = noisy.GetHost(Intent.Write)[0];
            for (int y = 0; y < Dims.Y; y++)
            {
                for (int x = 0; x < Dims.X; x++)
                {
                    int ClosestX = (int)Math.Max(0, Math.Min(DimsPositions.X - 1, (int)(((float)x - Dim / 2) / PositionStep.X + 0.5f)));
                    int ClosestY = (int)Math.Max(0, Math.Min(DimsPositions.Y - 1, (int)(((float)y - Dim / 2) / PositionStep.Y + 0.5f)));
                    int ClosestID = ClosestY * DimsPositions.X + ClosestX;

                    int3 Position = Positions[ClosestID];
                    int LocalX = Math.Max(0, Math.Min(Dim - 1, x - Position.X + Dim / 2));
                    int LocalY = Math.Max(0, Math.Min(Dim - 1, y - Position.Y + Dim / 2));

                    Denoised[y * Dims.X + x] = PredictionTiles[ClosestID][LocalY * Dim + LocalX];
                }
            }
        }
    }
}
