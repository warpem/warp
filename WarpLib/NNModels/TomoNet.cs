using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    public class TomoNet : IDisposable
    {
        public readonly int3 DimsSource;
        public readonly int3 DimsCTF;
        public readonly int3 DimsTarget;
        public readonly float PixelSize = 8;
        public readonly int BatchSize = 4;
        public readonly int DeviceBatch = 4;
        public readonly int[] Devices;
        public readonly int NDevices;

        private UNet3D[] UNetModel;

        private TorchTensor[] TensorSource;
        private TorchTensor[] TensorTarget;
        private TorchTensor[] TensorCTF;

        private TorchTensor[] TensorMask;

        private Loss Loss;
        private Optimizer Optimizer;

        private Image ResultPredicted;
        private float[] ResultLoss = new float[1];

        private bool IsDisposed = false;

        public TomoNet(int3 dimsSource, int3 dimsCTF, int3 dimsTarget, int[] devices, int batchSize = 4)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            DimsSource = dimsSource;
            DimsCTF = dimsCTF;
            DimsTarget = dimsTarget;

            UNetModel = new UNet3D[NDevices];

            TensorSource = new TorchTensor[NDevices];
            TensorTarget = new TorchTensor[NDevices];
            TensorCTF = new TorchTensor[NDevices];
            TensorMask = new TorchTensor[NDevices];

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];

                UNetModel[i] = UNet3D(2, 1, 99999, 1, 1, true, true, true);
                UNetModel[i].ToCuda(DeviceID);
                //UNetModel[i].ToType(ScalarType.BFloat16);

                TensorSource[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, DimsSource.Z, DimsSource.Y, DimsSource.X }, DeviceType.CUDA, DeviceID);
                TensorTarget[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, DimsTarget.Z, DimsTarget.Y, DimsTarget.X }, DeviceType.CUDA, DeviceID);
                TensorCTF[i] = Float32Tensor.Ones(new long[] { DeviceBatch, 1, DimsCTF.Z, DimsCTF.Y, DimsCTF.X / 2 + 1 }, DeviceType.CUDA, DeviceID);

                TensorMask[i] = Float32Tensor.Ones(new long[] { 1, 1, DimsCTF.Z, DimsCTF.Y, DimsCTF.X }, DeviceType.CUDA, DeviceID);
                {
                    int3 BoxDim2 = DimsCTF;
                    int3 Margin = (DimsCTF - DimsSource) / 2;
                    float[] MaskData = new float[BoxDim2.Elements()];
                    for (int z = 0; z < BoxDim2.Z; z++)
                    {
                        float zz = Math.Max(Margin.Z - z, z - (Margin.Z + DimsSource.Z - 1)) / (float)Margin.Z;
                        zz = Math.Max(0, Math.Min(1, zz));

                        for (int y = 0; y < BoxDim2.Y; y++)
                        {
                            float yy = Math.Max(Margin.Y - y, y - (Margin.Y + DimsSource.Y - 1)) / (float)Margin.Y;
                            yy = Math.Max(0, Math.Min(1, yy));

                            for (int x = 0; x < BoxDim2.X; x++)
                            {
                                float xx = Math.Max(Margin.X - x, x - (Margin.X + DimsSource.X - 1)) / (float)Margin.X;
                                xx = Math.Max(0, Math.Min(1, xx));

                                float r = Math.Min(1, (float)Math.Sqrt(zz * zz + yy * yy + xx * xx));
                                float v = (float)Math.Cos(r * Math.PI) * 0.5f + 0.5f;

                                MaskData[(z * BoxDim2.Y + y) * BoxDim2.X + x] = v;
                            }
                        }
                    }

                    GPU.CopyHostToDevice(MaskData,
                                         TensorMask[i].DataPtr(),
                                         MaskData.Length);

                    new Image(MaskData, BoxDim2).WriteMRC("d_mask.mrc", true);
                }

            }, null);

            Loss = MSE(Reduction.Mean);
            Optimizer = Optimizer.Adam(UNetModel[0].GetParameters(), 0.01, 1e-5);

            ResultPredicted = new Image(IntPtr.Zero, new int3(DimsSource.X, DimsSource.Y, DimsSource.Z * BatchSize));
        }

        private void ScatterData(Image src, TorchTensor[] dest)
        {
            src.GetDevice(Intent.Read);

            for (int i = 0; i < NDevices; i++)
                GPU.CopyDeviceToDevice(src.GetDeviceSlice(i * DeviceBatch * DimsSource.Z, Intent.Read),
                                       dest[i].DataPtr(),
                                       DeviceBatch * DimsSource.Elements());
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
                    using (TorchTensor PredictionPlusSource = Prediction.Add(TensorSource[i]))
                        GPU.CopyDeviceToDevice(PredictionPlusSource.DataPtr(),
                                               ResultPredicted.GetDeviceSlice(i * DeviceBatch * DimsSource.Z, Intent.Write),
                                               DeviceBatch * DimsSource.Elements());
                }
            }, null);

            prediction = ResultPredicted;
        }

        public void TrainDeconv(Image source,
                                Image target,
                                Image ctf,
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

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch * DimsSource.Z, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * (int)DimsSource.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch * DimsTarget.Z, Intent.Read),
                                       TensorTarget[i].DataPtr(),
                                       DeviceBatch * (int)DimsTarget.Elements());
                GPU.CopyDeviceToDevice(ctf.GetDeviceSlice(i * DeviceBatch * DimsCTF.Z, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * (int)DimsCTF.ElementsFFT());

                int3 MarginSource = (DimsCTF - DimsSource) / 2;
                int3 MarginTarget = (DimsCTF - DimsTarget) / 2;

                {
                    using (TorchTensor PredictionDeconv = UNetModel[i].Forward(TensorSource[i]))
                    using (TorchTensor PredictionDeconvPlusSource = PredictionDeconv.Add(TensorSource[i]))
                    using (TorchTensor PredictionDeconvPad = PredictionDeconvPlusSource.Pad(new long[] { MarginSource.X, MarginSource.X, MarginSource.Y, MarginSource.Y, MarginSource.Z, MarginSource.Z }))
                    using (TorchTensor PredictionDeconvPadMask = PredictionDeconvPad.Mul(TensorMask[i]))
                    using (TorchTensor PredictionFT = PredictionDeconvPadMask.rfftn(new long[] { 2, 3, 4 }))
                    using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                    using (TorchTensor PredictionPad = PredictionFTConv.irfftn(new long[] { 2, 3, 4 }))
                    using (TorchTensor Prediction1 = PredictionPad.Slice(4, MarginTarget.X, MarginTarget.X + DimsTarget.X, 1))
                    using (TorchTensor Prediction2 = Prediction1.Slice(3, MarginTarget.Y, MarginTarget.Y + DimsTarget.Y, 1))
                    using (TorchTensor Prediction3 = Prediction2.Slice(2, MarginTarget.Z, MarginTarget.Z + DimsTarget.Z, 1))
                    //using (TorchTensor Prediction3Dummy = Prediction3.Mul(1f))
                    using (TorchTensor PredictionLoss = Loss(Prediction3, TensorTarget[i]))
                    {
                        GPU.CopyDeviceToDevice(PredictionDeconv.DataPtr(),
                                               ResultPredicted.GetDeviceSlice(i * DeviceBatch * DimsSource.Z, Intent.Write),
                                               DeviceBatch * (int)DimsSource.Elements());

                        if (i == 0)
                            GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLoss, 1);

                        PredictionLoss.Backward();
                    }
                }

                
            }, null);

            GatherGrads();

            if (NDevices > 1)
            {
                UNetModel[0].ScaleGrad(1f / NDevices);
            }

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
                UNetModel[i].Load(path, DeviceType.CUDA, Devices[i]);
        }

        ~TomoNet()
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

                    for (int i = 0; i < NDevices; i++)
                    {
                        TensorSource[i].Dispose();
                        TensorTarget[i].Dispose();
                        TensorCTF[i].Dispose();
                        TensorMask[i].Dispose();

                        UNetModel[i].Dispose();
                    }

                    Optimizer.Dispose();
                }
            }
        }

        public static void Deconvolve(Image noisy, TomoNet[] networks)
        {
            int GPUID = GPU.GetDevice();
            int NThreads = 1;

            int3 Dims = noisy.Dims;
            int3 DimsWindow = networks[0].DimsSource;
            int BatchSize = networks[0].BatchSize;

            int3 DimsValid = networks[0].DimsTarget;

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

            Image[] Extracted = Helper.ArrayOfFunction(i => new Image(new int3(DimsWindow.X, DimsWindow.Y, DimsWindow.Z * BatchSize)), NThreads);

            Helper.ForCPU(0, (Positions.Length + BatchSize - 1) / BatchSize, NThreads,
                (threadID) => GPU.SetDevice(GPUID),
                (ib, threadID) =>
                //for (int b = 0; b < Positions.Length; b += BatchSize)
                {
                    int b = ib * BatchSize;
                    int CurBatch = Math.Min(BatchSize, Positions.Length - b);

                    int3[] CurPositions = Positions.Skip(b).Take(CurBatch).ToArray();
                    GPU.Extract(noisy.GetDevice(Intent.Read),
                                Extracted[threadID].GetDevice(Intent.Write),
                                noisy.Dims,
                                new int3(DimsWindow),
                                Helper.ToInterleaved(CurPositions.Select(p => p - new int3(DimsWindow / 2)).ToArray()),
                                false,
                                (uint)CurBatch);

                    Image PredictionData = null;
                    networks[0].Predict(Extracted[threadID], out PredictionData);

                    for (int i = 0; i < CurBatch; i++)
                    {
                        float[] TileData = new float[DimsWindow.X * DimsWindow.Y * DimsWindow.Z];
                        GPU.CopyDeviceToHost(PredictionData.GetDeviceSlice(i * DimsWindow.Z, Intent.Read),
                                             TileData,
                                             TileData.Length);

                        PredictionTiles[b + i] = TileData;
                        //PredictionTiles[b + i] = PredictionData.Skip(i * Dim * Dim * Dim).Take(Dim * Dim * Dim).ToArray();
                    }
                }, null);

            foreach (var item in Extracted)
                item.Dispose();

            noisy.FreeDevice();

            float[][] Denoised = noisy.GetHost(Intent.Write);
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

                        Denoised[z][y * Dims.X + x] = PredictionTiles[ClosestID][(LocalZ * DimsWindow.Y + LocalY) * DimsWindow.X + LocalX];
                    }
                }
            }
        }
    }
}
