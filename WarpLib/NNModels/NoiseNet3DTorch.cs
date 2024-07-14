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

namespace Warp
{
    public class NoiseNet3DTorch : IDisposable
    {
        public readonly int3 BoxDimensions;
        public readonly float PixelSize = 8;
        public readonly int BatchSize = 4;
        public readonly int Depth = 2;
        public readonly bool ProgressiveDepth = true;
        public readonly int MaxWidth = 99999;
        public readonly int DeviceBatch = 4;
        public readonly int[] Devices;
        public readonly int NDevices;

        private UNet3D[] UNetModel;

        private TorchTensor[] TensorSource;
        private TorchTensor[] TensorTarget;
        private TorchTensor[] TensorCTF;

        private TorchTensor[] TensorMask;

        private TorchTensor[] TensorOne;

        private Loss Loss;
        private Optimizer Optimizer;

        private Image ResultPredicted;
        private Image ResultPredictedDeconv;
        private Image ResultPredictedRotated;
        private float[] ResultLoss = new float[1], ResultLossTwisted = new float[1];

        private bool IsDisposed = false;

        public NoiseNet3DTorch(int3 boxDimensions, int[] devices, int batchSize = 4, int depth = 2, bool progressiveDepth = true, int maxWidth = 99999)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            Depth = depth;
            ProgressiveDepth = progressiveDepth;
            MaxWidth = maxWidth;

            BoxDimensions = boxDimensions;

            UNetModel = new UNet3D[NDevices];

            TensorSource = new TorchTensor[NDevices];
            TensorTarget = new TorchTensor[NDevices];
            TensorCTF = new TorchTensor[NDevices];
            TensorMask = new TorchTensor[NDevices];

            TensorOne = new TorchTensor[NDevices];

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];

                UNetModel[i] = UNet3D(Depth, 1, MaxWidth, 1, 1, true, false, ProgressiveDepth);
                UNetModel[i].ToCuda(DeviceID);

                //UNetPristineModel[i] = UNet3D(2, 1, 1, 1);
                //UNetPristineModel[i].ToCuda(DeviceID);

                //DiscriminatorModel[i] = Discriminator3D();
                //DiscriminatorModel[i].ToCuda(DeviceID);

                TensorSource[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Z, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorTarget[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Z, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorCTF[i] = Float32Tensor.Ones(new long[] { DeviceBatch, 1, BoxDimensions.Z * 2, BoxDimensions.Y * 2, BoxDimensions.X * 2 / 2 + 1 }, DeviceType.CUDA, DeviceID);

                TensorOne[i] = Float32Tensor.Ones(new long[] { 1, 1 }, DeviceType.CUDA, DeviceID);
                //TensorMinusOne[i] = Float32Tensor.Ones(new long[] { 1, 1 }, DeviceType.CUDA, DeviceID);
                //{
                //    GPU.CopyHostToDevice(new float[] { -1f }, TensorMinusOne[i].DataPtr(), 1);
                //}

                //TensorMatrices[i] = Float32Tensor.Ones(new long[] { DeviceBatch, 3, 4 }, DeviceType.CUDA, DeviceID);

                TensorMask[i] = Float32Tensor.Ones(new long[] { 1, 1, BoxDimensions.Z * 2, BoxDimensions.Y * 2, BoxDimensions.X * 2 }, DeviceType.CUDA, DeviceID);
                {
                    int3 BoxDim2 = BoxDimensions * 2;
                    int3 Margin = (BoxDim2 - BoxDimensions) / 2;
                    float[] MaskData = new float[BoxDim2.Elements()];
                    for (int z = 0; z < BoxDim2.Z; z++)
                    {
                        float zz = Math.Max(Margin.Z - z, z - (Margin.Z + BoxDimensions.Z)) / (float)Margin.Z;
                        zz = Math.Max(0, Math.Min(1, zz));

                        for (int y = 0; y < BoxDim2.Y; y++)
                        {
                            float yy = Math.Max(Margin.Y - y, y - (Margin.Y + BoxDimensions.Y)) / (float)Margin.Y;
                            yy = Math.Max(0, Math.Min(1, yy));

                            for (int x = 0; x < BoxDim2.X; x++)
                            {
                                float xx = Math.Max(Margin.X - x, x - (Margin.X + BoxDimensions.X)) / (float)Margin.X;
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

                    //new Image(MaskData, BoxDim2).WriteMRC("d_mask.mrc", true);
                }

            }, null);

            //for (int i = 0; i < NDevices; i++)
            //    UNetModel[0].SynchronizeTo(UNetPristineModel[i], Devices[i]);

            Loss = MSE(Reduction.Mean);
            Optimizer = Optimizer.Adam(UNetModel[0].GetParameters(), 0.01, 1e-4);
            //OptimizerPristine = Optimizer.Adam(UNetPristineModel[0].GetParameters(), 0.01, 1e-4);
            //OptimizerDisc = Optimizer.Adam(DiscriminatorModel[0].GetParameters(), 0.01, 1e-4);

            ResultPredicted = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BoxDimensions.Z * BatchSize));
            ResultPredictedDeconv = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BoxDimensions.Z * BatchSize));
            ResultPredictedRotated = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BoxDimensions.Z * BatchSize));
        }

        private void ScatterData(Image src, TorchTensor[] dest)
        {
            src.GetDevice(Intent.Read);

            for (int i = 0; i < NDevices; i++)
                GPU.CopyDeviceToDevice(src.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Read),
                                       dest[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
        }

        private void SyncParams()
        {
            for (int i = 1; i < NDevices; i++)
                UNetModel[0].SynchronizeTo(UNetModel[i], Devices[i]);
            //for (int i = 1; i < NDevices; i++)
            //    UNetPristineModel[0].SynchronizeTo(UNetPristineModel[i], Devices[i]);
        }

        private void GatherGrads()
        {
            for (int i = 1; i < NDevices; i++)
                UNetModel[0].GatherGrad(UNetModel[i]);
            //for (int i = 1; i < NDevices; i++)
            //    UNetPristineModel[0].GatherGrad(UNetPristineModel[i]);
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
                                               ResultPredicted.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Write),
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

            float[][] DeviceLosses = Helper.ArrayOfFunction(i => new float[1], NDevices);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                UNetModel[i].Train();
                UNetModel[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Read),
                                       TensorTarget[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());

                using (TorchTensor Prediction = UNetModel[i].Forward(TensorSource[i]))
                using (TorchTensor PredictionPlusSource = Prediction.Add(TensorSource[i]))
                using (TorchTensor PredictionLoss = Loss(PredictionPlusSource, TensorTarget[i]))
                {
                    //GPU.CopyDeviceToDevice(PredictionPlusSource.DataPtr(),
                    //                       ResultPredicted.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Write),
                    //                       DeviceBatch * (int)BoxDimensions.Elements());

                    GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), DeviceLosses[i], 1);

                    PredictionLoss.Backward();
                }
            }, null);

            GatherGrads();

            if (NDevices > 1)
                UNetModel[0].ScaleGrad(1f / NDevices);

            Optimizer.Step();

            ResultLoss[0] = MathHelper.Mean(DeviceLosses.Select(a => a[0]));

            prediction = ResultPredicted;
            loss = ResultLoss;
        }

        public void TrainDeconv(Image source,
                                Image target,
                                Image ctf,
                                float learningRate,
                                bool doAdversarial,
                                float[] adversarialAngles,
                                float2[] adversarialShifts,
                                out Image prediction,
                                out Image predictionDeconv,
                                out Image predictionRotated,
                                out float[] loss,
                                out float[] lossPristine)
        {
            Optimizer.SetLearningRateAdam(learningRate);
            Optimizer.ZeroGrad();
            //OptimizerPristine.SetLearningRateAdam(learningRate);
            //OptimizerPristine.ZeroGrad();

            SyncParams();
            //SyncParamsDiscriminator();

            ResultPredicted.GetDevice(Intent.Write);
            ResultPredictedDeconv.GetDevice(Intent.Write);
            ResultPredictedRotated.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                UNetModel[i].Train();
                UNetModel[i].ZeroGrad();
                //UNetPristineModel[i].Train();
                //UNetPristineModel[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Read),
                                       TensorTarget[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(ctf.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z * 2, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * (int)(BoxDimensions * 2).ElementsFFT());

                int3 Margin = BoxDimensions / 2;

                //using (TorchTensor PredictionDeconvTwist = UNetPristineModel[i].Forward(TensorSource[i]))
                //using (TorchTensor PredictionDeconvPad = PredictionDeconvTwist.Pad(new long[] { Margin.X, Margin.X, Margin.Y, Margin.Y, Margin.Z, Margin.Z }))
                //using (TorchTensor PredictionDeconvPadMask = PredictionDeconvPad.Mul(TensorMask[i]))
                //using (TorchTensor PredictionFT = PredictionDeconvPadMask.rfftn(new long[] { 2, 3, 4 }))
                //using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                //using (TorchTensor PredictionPad = PredictionFTConv.irfftn(new long[] { 2, 3, 4 }))
                //using (TorchTensor Prediction1 = PredictionPad.Slice(4, Margin.X, Margin.X + BoxDimensions.X, 1))
                //using (TorchTensor Prediction2 = Prediction1.Slice(3, Margin.Y, Margin.Y + BoxDimensions.Y, 1))
                //using (TorchTensor Prediction3 = Prediction2.Slice(2, Margin.Z, Margin.Z + BoxDimensions.Z, 1))
                //using (TorchTensor PredictionLoss = Loss(Prediction3, TensorTarget[i]))
                //{
                //    //GPU.CopyDeviceToDevice(PredictionDeconvTwist.DataPtr(),
                //    //                       ResultPredictedDeconv.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Write),
                //    //                       DeviceBatch * (int)BoxDimensions.Elements());

                //    //if (i == 0)
                //    //    GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLossTwisted, 1);

                //    PredictionLoss.Backward();
                //}

                //GPU.CopyDeviceToDevice(ResultPredictedDeconv.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Write),
                //                       TensorTarget[i].DataPtr(),
                //                       DeviceBatch * (int)BoxDimensions.Elements());

                //if (!doAdversarial)
                {
                    using (TorchTensor PredictionDeconv = UNetModel[i].Forward(TensorSource[i]))
                    using (TorchTensor PredictionDeconvPad = PredictionDeconv.Pad(new long[] { Margin.X, Margin.X, Margin.Y, Margin.Y, Margin.Z, Margin.Z }))
                    using (TorchTensor PredictionDeconvPadMask = PredictionDeconvPad.Mul(TensorMask[i]))
                    using (TorchTensor PredictionFT = PredictionDeconvPadMask.rfftn(new long[] { 2, 3, 4 }))
                    using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                    using (TorchTensor PredictionPad = PredictionFTConv.irfftn(new long[] { 2, 3, 4 }))
                    using (TorchTensor Prediction1 = PredictionPad.Slice(4, Margin.X, Margin.X + BoxDimensions.X, 1))
                    using (TorchTensor Prediction2 = Prediction1.Slice(3, Margin.Y, Margin.Y + BoxDimensions.Y, 1))
                    using (TorchTensor Prediction3 = Prediction2.Slice(2, Margin.Z, Margin.Z + BoxDimensions.Z, 1))
                    using (TorchTensor Prediction3Dummy = Prediction3.Mul(TensorOne[i]))
                    using (TorchTensor PredictionLoss = Loss(Prediction3, TensorTarget[i]))
                    //using (TorchTensor PredictionLoss = Loss(PredictionDeconv, TensorTarget[i]))
                    {
                        //GPU.CopyDeviceToDevice(Prediction3Dummy.DataPtr(),
                        //                       ResultPredicted.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Write),
                        //                       DeviceBatch * (int)BoxDimensions.Elements());
                        GPU.CopyDeviceToDevice(PredictionDeconv.DataPtr(),
                                               ResultPredictedDeconv.GetDeviceSlice(i * DeviceBatch * BoxDimensions.Z, Intent.Write),
                                               DeviceBatch * (int)BoxDimensions.Elements());

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
                //UNetPristineModel[0].ScaleGrad(1f / NDevices);
            }

            Optimizer.Step();
            //OptimizerPristine.Step();

            prediction = ResultPredicted;
            predictionDeconv = ResultPredictedDeconv;
            predictionRotated = ResultPredictedRotated;
            loss = ResultLoss;
            lossPristine = ResultLossTwisted;
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

        ~NoiseNet3DTorch()
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
                        //TensorMask[i].Dispose();

                        UNetModel[i].Dispose();
                    }

                    Optimizer.Dispose();
                }
            }
        }

        public static void Denoise(Image noisy, NoiseNet3DTorch[] networks)
        {
            int GPUID = GPU.GetDevice();
            int NThreads = 1;

            int3 Dims = noisy.Dims;
            int3 DimsWindow = networks[0].BoxDimensions;
            int BatchSize = networks[0].BatchSize;

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

        public static (Image[] Halves1, Image[] Halves2, float2[] Stats) TrainOnVolumes(NoiseNet3DTorch network,
                                                                                        Image[] halves1,
                                                                                        Image[] halves2,
                                                                                        Image[] masks,
                                                                                        float angpix,
                                                                                        float lowpass,
                                                                                        float upsample,
                                                                                        bool dontFlatten,
                                                                                        bool performTraining,
                                                                                        int niterations,
                                                                                        float startFrom,
                                                                                        int batchsize,
                                                                                        int gpuprocess,
                                                                                        Action<string> progressCallback)
        {
            GPU.SetDevice(gpuprocess);

            #region Mask

            Debug.Write("Preparing mask... ");
            progressCallback?.Invoke("Preparing mask... ");

            int3[] BoundingBox = Helper.ArrayOfFunction(i => new int3(-1), halves1.Length);
            if (masks != null)
            {
                for (int i = 0; i < masks.Length; i++)
                {
                    Image Mask = masks[i];

                    Mask.TransformValues((x, y, z, v) =>
                    {
                        if (v > 0.5f)
                        {
                            BoundingBox[i].X = Math.Max(BoundingBox[i].X, Math.Abs(x - Mask.Dims.X / 2) * 2);
                            BoundingBox[i].Y = Math.Max(BoundingBox[i].Y, Math.Abs(y - Mask.Dims.Y / 2) * 2);
                            BoundingBox[i].Z = Math.Max(BoundingBox[i].Z, Math.Abs(z - Mask.Dims.Z / 2) * 2);
                        }

                        return v;
                    });

                    if (BoundingBox[i].X < 2)
                        throw new Exception("Mask does not seem to contain any non-zero values.");

                    BoundingBox[i] += network.BoxDimensions * 2;

                    BoundingBox[i].X = Math.Min(BoundingBox[i].X, Mask.Dims.X);
                    BoundingBox[i].Y = Math.Min(BoundingBox[i].Y, Mask.Dims.Y);
                    BoundingBox[i].Z = Math.Min(BoundingBox[i].Z, Mask.Dims.Z);

                    BoundingBox[i] = Mask.Dims;
                }
            }

            #endregion

            #region Load and prepare data

            progressCallback?.Invoke("Preparing data:");

            List<Image> Maps1 = new List<Image>();
            List<Image> Maps2 = new List<Image>();

            List<Image> HalvesForDenoising1 = new List<Image>();
            List<Image> HalvesForDenoising2 = new List<Image>();
            List<float2> StatsForDenoising = new List<float2>();

            for (int imap = 0; imap < halves1.Length; imap++)
            {
                Debug.Write($"Preparing map {imap}... ");
                progressCallback?.Invoke($"Preparing map {imap}... ");

                Image Map1 = halves1[imap];
                Image Map2 = halves2[imap];

                float MapPixelSize = Map1.PixelSize / upsample;

                if (!dontFlatten)
                {
                    Image Average = Map1.GetCopy();
                    Average.Add(Map2);

                    if (masks != null)
                        Average.Multiply(masks[imap]);

                    float[] Spectrum = Average.AsAmplitudes1D(true, 1, (Average.Dims.X + Average.Dims.Y + Average.Dims.Z) / 6);
                    Average.Dispose();

                    int i10A = Math.Min((int)(angpix * 2 / 10 * Spectrum.Length), Spectrum.Length - 1);
                    float Amp10A = Spectrum[i10A];

                    for (int i = 0; i < Spectrum.Length; i++)
                        Spectrum[i] = i < i10A ? 1 : (Amp10A / Math.Max(1e-10f, Spectrum[i]));

                    Image Map1Flat = Map1.AsSpectrumMultiplied(true, Spectrum);
                    Map1.FreeDevice();
                    Map1 = Map1Flat;
                    Map1.FreeDevice();

                    Image Map2Flat = Map2.AsSpectrumMultiplied(true, Spectrum);
                    Map2.FreeDevice();
                    Map2 = Map2Flat;
                    Map2.FreeDevice();
                }

                if (lowpass > 0)
                {
                    Map1.Bandpass(0, angpix * 2 / lowpass, true, 0.05f);
                    Map2.Bandpass(0, angpix * 2 / lowpass, true, 0.05f);
                }

                if (upsample != 1f)
                {
                    Image Map1Scaled = Map1.AsScaled(Map1.Dims * upsample / 2 * 2);
                    Map1.FreeDevice();
                    Map1 = Map1Scaled;
                    Map1.FreeDevice();

                    Image Map2Scaled = Map2.AsScaled(Map2.Dims * upsample / 2 * 2);
                    Map2.FreeDevice();
                    Map2 = Map2Scaled;
                    Map2.FreeDevice();
                }

                Image ForDenoising1 = Map1.GetCopy();
                Image ForDenoising2 = Map2.GetCopy();

                if (BoundingBox[imap].X > 0)
                {
                    Image Map1Cropped = Map1.AsPadded(BoundingBox[imap]);
                    Map1.FreeDevice();
                    Map1 = Map1Cropped;
                    Map1.FreeDevice();

                    Image Map2Cropped = Map2.AsPadded(BoundingBox[imap]);
                    Map2.FreeDevice();
                    Map2 = Map2Cropped;
                    Map2.FreeDevice();
                }

                float2 MeanStd = MathHelper.MeanAndStd(Helper.Combine(Map1.GetHostContinuousCopy(), Map2.GetHostContinuousCopy()));

                Map1.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
                Map2.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);

                ForDenoising1.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
                ForDenoising2.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);

                HalvesForDenoising1.Add(ForDenoising1);
                HalvesForDenoising2.Add(ForDenoising2);
                StatsForDenoising.Add(MeanStd);

                GPU.PrefilterForCubic(Map1.GetDevice(Intent.ReadWrite), Map1.Dims);
                Map1.FreeDevice();
                Maps1.Add(Map1);

                GPU.PrefilterForCubic(Map2.GetDevice(Intent.ReadWrite), Map2.Dims);
                Map2.FreeDevice();
                Maps2.Add(Map2);

                Debug.WriteLine(" Done.");
            }

            if (masks != null)
                foreach (var mask in masks)
                    mask.FreeDevice();

            #endregion

            if (batchsize != 4 || Maps1.Count > 1)
            {
                if (batchsize < 1)
                    throw new Exception("Batch size must be at least 1.");

                niterations = niterations * 4 / batchsize / Maps1.Count;
                progressCallback?.Invoke($"Adjusting the number of iterations to {niterations} to match batch size and number of maps.");
            }

            int Dim = network.BoxDimensions.X;

            progressCallback?.Invoke($"0/{niterations}");

            if (performTraining)
            {
                GPU.SetDevice(gpuprocess);

                #region Training

                Random Rand = new Random(123);

                int NMaps = Maps1.Count;
                int NMapsPerBatch = Math.Min(128, NMaps);
                int MapSamples = batchsize;

                Image[] ExtractedSource = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, Dim * MapSamples)), NMapsPerBatch);
                Image[] ExtractedTarget = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, Dim * MapSamples)), NMapsPerBatch);

                ulong[][] MapTextures = Helper.ArrayOfFunction(i => new ulong[NMaps], 2);
                ulong[][] MapTextureArrays = Helper.ArrayOfFunction(i => new ulong[NMaps], 2);
                List<Image>[] AllMaps = { Maps1, Maps2 };
                for (int ihalf = 0; ihalf < 2; ihalf++)
                {
                    for (int imap = 0; imap < NMaps; imap++)
                    {
                        ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                        GPU.CreateTexture3D(AllMaps[ihalf][imap].GetDevice(Intent.Read), AllMaps[ihalf][imap].Dims, Texture, TextureArray, true);
                        MapTextures[ihalf][imap] = Texture[0];
                        MapTextureArrays[ihalf][imap] = TextureArray[0];
                    }
                }

                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                Queue<float> Losses = new Queue<float>();

                for (int iter = (int)(startFrom * niterations); iter < niterations; iter++)
                {
                    int[] ShuffledMapIDs = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMaps, 1), NMapsPerBatch, Rand.Next(9999));

                    for (int m = 0; m < NMapsPerBatch; m++)
                    {
                        int MapID = ShuffledMapIDs[m];

                        Image Map1 = Maps1[MapID];
                        Image Map2 = Maps2[MapID];
                        ulong Map1Texture = MapTextures[0][MapID];
                        ulong Map2Texture = MapTextures[1][MapID];

                        int3 DimsMap = Map1.Dims;

                        int3 Margin = new int3((int)(Dim / 2 * 1.5f));
                        //Margin.Z = 0;
                        float3[] Position = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * (DimsMap.X - Margin.X * 2) + Margin.X,
                                                                                   (float)Rand.NextDouble() * (DimsMap.Y - Margin.Y * 2) + Margin.Y,
                                                                                   (float)Rand.NextDouble() * (DimsMap.Z - Margin.Z * 2) + Margin.Z), MapSamples);

                        float3[] Angle = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * 360,
                                                                                (float)Rand.NextDouble() * 360,
                                                                                (float)Rand.NextDouble() * 360) * Helper.ToRad, MapSamples);

                        {
                            GPU.Rotate3DExtractAt(Map1Texture,
                                                  Map1.Dims,
                                                  ExtractedSource[m].GetDevice(Intent.Write),
                                                  new int3(Dim),
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedSource[MapID].WriteMRC("d_extractedsource.mrc", true);
                        }

                        {
                            GPU.Rotate3DExtractAt(Map2Texture,
                                                  Map2.Dims,
                                                  ExtractedTarget[m].GetDevice(Intent.Write),
                                                  new int3(Dim),
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedTarget.WriteMRC("d_extractedtarget.mrc", true);
                        }

                        //Map1.FreeDevice();
                        //Map2.FreeDevice();
                    }

                    Image Predicted = null;
                    float[] Loss = null;

                    {
                        float CurrentLearningRate = 0.0001f * (float)Math.Pow(10, -iter / (float)niterations * 2);

                        for (int m = 0; m < ShuffledMapIDs.Length; m++)
                        {
                            int MapID = m;

                            bool Twist = Rand.Next(2) == 0;

                            if (Twist)
                                network.Train(ExtractedSource[MapID],
                                              ExtractedTarget[MapID],
                                              CurrentLearningRate,
                                              out Predicted,
                                              out Loss);
                            else
                                network.Train(ExtractedTarget[MapID],
                                              ExtractedSource[MapID],
                                              CurrentLearningRate,
                                              out Predicted,
                                              out Loss);

                            Losses.Enqueue(Loss[0]);
                            if (Losses.Count > 100)
                                Losses.Dequeue();
                        }
                    }


                    TimeSpan TimeRemaining = Watch.Elapsed * (niterations - 1 - iter);
                    Watch.Restart();

                    string ProgressText = $"{iter + 1}/{niterations}, {TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, log(loss) = {Math.Log(MathHelper.Mean(Losses)).ToString("F4")}";

                    if (float.IsNaN(Loss[0]) || float.IsInfinity(Loss[0]))
                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                    Debug.WriteLine(ProgressText);
                    progressCallback?.Invoke(ProgressText);
                }

                Debug.WriteLine("\nDone training!\n");

                #endregion

                #region Cleanup

                for (int ihalf = 0; ihalf < 2; ihalf++)
                    for (int imap = 0; imap < NMaps; imap++)
                        GPU.DestroyTexture(MapTextures[ihalf][imap], MapTextureArrays[ihalf][imap]);

                foreach (var item in ExtractedSource)
                    item.Dispose();
                foreach (var item in ExtractedTarget)
                    item.Dispose();

                #endregion
            }

            return (HalvesForDenoising1.ToArray(), HalvesForDenoising2.ToArray(), StatsForDenoising.ToArray());
        }
    }
}
