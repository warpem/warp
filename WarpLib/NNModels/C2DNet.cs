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

namespace Warp
{
    public class C2DNet : IDisposable
    {
        public readonly int2 BoxDimensions;
        public readonly int CodeLength;
        public readonly int BatchSize = 8;
        public readonly int DeviceBatch = 8;
        public readonly int[] Devices;
        public readonly int NDevices;

        private C2DNetEncoder[] EncoderModel;
        private C2DNetDecoder[] DecoderModel;

        private TorchTensor[] TensorSource;
        private TorchTensor[] TensorTarget;
        private TorchTensor[] TensorCTF;
        private TorchTensor[] TensorMask;
        private TorchTensor[] TensorMaskNeg;
        private TorchTensor[] TensorRefDotProd;

        private TorchTensor[] TensorTargetAligned;

        private Loss MSELoss;
        private Optimizer OptimizerEncoder;
        private Optimizer OptimizerDecoder;

        private double KLDWeight = 1e-4;

        int NRotations = 8;

        private float[] ResultCode;
        private Image ResultPredicted;
        private Image ResultPredictedDeconv;
        private Image ResultAlignedTarget;
        private float[] ResultLoss = new float[1];
        private float[] ResultKLD = new float[1];

        private bool IsDisposed = false;

        public C2DNet(int boxSize, int codeLength, int[] devices, int batchSize = 8)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = new int2(boxSize);
            CodeLength = codeLength;

            EncoderModel = new C2DNetEncoder[NDevices];
            DecoderModel = new C2DNetDecoder[NDevices];

            TensorSource = new TorchTensor[NDevices];
            TensorTarget = new TorchTensor[NDevices];
            TensorCTF = new TorchTensor[NDevices];
            TensorMask = new TorchTensor[NDevices];
            TensorMaskNeg = new TorchTensor[NDevices];
            TensorRefDotProd = new TorchTensor[NDevices];

            TensorTargetAligned = new TorchTensor[NDevices];

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];

                EncoderModel[i] = C2DNetEncoder(BoxDimensions.X, CodeLength);
                EncoderModel[i].ToCuda(DeviceID);
                DecoderModel[i] = C2DNetDecoder(BoxDimensions.X, CodeLength);
                DecoderModel[i].ToCuda(DeviceID);

                TensorSource[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorTarget[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorCTF[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X / 2 + 1 }, DeviceType.CUDA, DeviceID);
                TensorMask[i] = Float32Tensor.Zeros(new long[] { 1, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorRefDotProd[i] = Float32Tensor.Zeros(new long[] { DeviceBatch / 2 }, DeviceType.CUDA, DeviceID);

                TensorTargetAligned[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);

                {
                    Image Mask = new Image(new int3(BoxDimensions.X, BoxDimensions.Y, 1));
                    Mask.Fill(1);
                    Mask.MaskSpherically(BoxDimensions.X / 2, 8, false);
                    Mask.TransformValues(v => 1 - v);

                    GPU.CopyDeviceToDevice(Mask.GetDevice(Intent.Read), TensorMask[i].DataPtr(), Mask.ElementsReal);

                    Mask.Dispose();
                }
            }, null);

            MSELoss = Losses.MSE();
            OptimizerEncoder = Optimizer.Adam(EncoderModel[0].GetParameters(), 0.01, 1e-4);
            OptimizerDecoder = Optimizer.Adam(DecoderModel[0].GetParameters(), 0.01, 1e-4);

            ResultCode = new float[BatchSize * CodeLength];
            ResultPredicted = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
            ResultPredictedDeconv = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
            ResultAlignedTarget = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
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
            {
                EncoderModel[0].SynchronizeTo(EncoderModel[i], Devices[i]);
                DecoderModel[0].SynchronizeTo(DecoderModel[i], Devices[i]);
            }
        }

        private void GatherGrads()
        {
            for (int i = 1; i < NDevices; i++)
            {
                EncoderModel[0].GatherGrad(EncoderModel[i]);
                DecoderModel[0].GatherGrad(DecoderModel[i]);
            }
        }


        public void Encode(Image data, out float[] code)
        {
            ScatterData(data, TensorSource);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                EncoderModel[i].Eval();

                using (TorchTensor PredictionCode = EncoderModel[i].Forward(TensorSource[i]))
                {
                    float[] TCode = new float[DeviceBatch * CodeLength];
                    GPU.CopyDeviceToHost(PredictionCode.DataPtr(), TCode, DeviceBatch * CodeLength);

                    Array.Copy(TCode, 0, ResultCode, i * DeviceBatch * CodeLength, DeviceBatch * CodeLength);
                }
            }, null);

            code = ResultCode;
        }

        double GenBoost = 5;

        public void Train(Image source,
                          Image target,
                          Image ctf,
                          float[] refDotProd,
                          float learningRate,
                          bool doAlignment,
                          float lowpass,
                          out Image prediction,
                          out Image predictionDeconv,
                          out Image alignedTarget,
                          out float[] loss,
                          out float[] kld)
        {
            OptimizerEncoder.SetLearningRateAdam(learningRate);
            OptimizerDecoder.SetLearningRateAdam(learningRate * GenBoost);

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);
            ResultPredictedDeconv.GetDevice(Intent.Write);

            //Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int i = 0;

                EncoderModel[i].Train();
                EncoderModel[i].ZeroGrad();
                DecoderModel[i].Train();
                DecoderModel[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTarget[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(ctf.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * (int)(BoxDimensions).ElementsFFT());
                GPU.CopyHostToDevice(refDotProd,
                                     TensorRefDotProd[i].DataPtr(),
                                     DeviceBatch / 2);

                using (TorchTensor InputPose = EncoderModel[i].ForwardPose(TensorSource[i]))
                using (TorchTensor PosedInput = EncoderModel[i].ApplyPose(TensorSource[i], InputPose))

                using (TorchTensor PredictionCode = EncoderModel[i].Forward(PosedInput))
                using (TorchTensor Code1 = PredictionCode.Slice(0, 0, DeviceBatch - 1, 2))
                using (TorchTensor Code2 = PredictionCode.Slice(0, 1, DeviceBatch - 0, 2))
                using (TorchTensor CodeLoss = MSELoss(Code1, Code2))

                using (TorchTensor PredictionDecoded = DecoderModel[i].Forward(PredictionCode, true))

                using (TorchTensor PredictionDecodedMasked = PredictionDecoded.Mul(TensorMask[i]))
                using (TorchTensor MaskSquared = PredictionDecodedMasked.Mul(PredictionDecodedMasked))
                using (TorchTensor MaskLoss = MaskSquared.Mean())

                using (TorchTensor PredictionDecodedFT = PredictionDecoded.rfftn(new long[] { 2, 3 }))
                using (TorchTensor PredictionDecodedFTConv = PredictionDecodedFT.Mul(TensorCTF[i]))
                using (TorchTensor Prediction = PredictionDecodedFTConv.irfftn(new long[] { 2, 3 }))
                {
                    float[] h_Pose = new float[DeviceBatch * 4];
                    GPU.CopyDeviceToHost(InputPose.DataPtr(), h_Pose, h_Pose.Length);

                    if (doAlignment)
                    {
                        using (Image PredictionFT = new Image(Prediction.DataPtr(), new int3(BoxDimensions.X, BoxDimensions.Y, DeviceBatch)).AsFFT().AndDisposeParent())
                        using (Image TargetFT = target.AsFFT())
                        {
                            PredictionFT.ShiftSlices(Helper.ArrayOfConstant(new float3(BoxDimensions.X / 2, BoxDimensions.Y / 2, 0), DeviceBatch));
                            TargetFT.ShiftSlices(Helper.ArrayOfConstant(new float3(BoxDimensions.X / 2, BoxDimensions.Y / 2, 0), DeviceBatch));

                            float minshell = 8;
                            float shiftstep = 32 / 8 * 3;
                            float anglestep = (float)Math.Asin(1f / minshell) * 3 * Helper.ToDeg;
                            int anglesteps = (int)Math.Ceiling(360 / anglestep);
                            anglestep = 360f / anglesteps;

                            List<float3> InitPoses = new List<float3>();
                            for (int b = 0; b < DeviceBatch; b++)
                                for (int angle = 0; angle < anglesteps; angle++)
                                    for (int y = -0; y <= 0; y++)
                                        for (int x = -0; x <= 0; x++)
                                            InitPoses.Add(new float3(x * shiftstep, y * shiftstep, angle * anglestep));
                            IntPtr d_InitPoses = GPU.MallocDeviceFromHost(Helper.ToInterleaved(InitPoses.ToArray()), InitPoses.Count * 3);

                            Image DataAligned = new Image(IntPtr.Zero, new int3(64, 64, DeviceBatch));

                            GPU.C2DNetAlign(PredictionFT.GetDevice(Intent.Read),
                                            PredictionFT.Dims.X,
                                            1,
                                            TargetFT.GetDevice(Intent.Read),
                                            target.GetDevice(Intent.Read),
                                            IntPtr.Zero,
                                            TargetFT.Dims.X,
                                            d_InitPoses,
                                            InitPoses.Count / DeviceBatch,
                                            8,
                                            (int)(32 * lowpass),
                                            1,
                                            anglestep / 3,
                                            shiftstep / 3,
                                            16,
                                            DeviceBatch,
                                            DataAligned.GetDevice(Intent.Write),
                                            IntPtr.Zero);

                            //DataAligned.WriteMRC("d_dataaligned.mrc", true);

                            GPU.FreeDevice(d_InitPoses);
                            GPU.CopyDeviceToDevice(DataAligned.GetDevice(Intent.Read), TensorTarget[0].DataPtr(), DataAligned.ElementsReal);
                            DataAligned.Dispose();
                        }
                    }

                    using (TorchTensor PredictionLoss = MSELoss(Prediction, /*TensorTarget[i]*/ PosedInput))
                    {
                        GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                               ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * (int)BoxDimensions.Elements());
                        GPU.CopyDeviceToDevice(PredictionDecoded.DataPtr(),
                                               ResultPredictedDeconv.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * (int)BoxDimensions.Elements());
                        GPU.CopyDeviceToDevice(PosedInput.DataPtr(),
                                               ResultAlignedTarget.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * (int)BoxDimensions.Elements());
                        if (i == 0)
                            GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLoss, 1);

                        //MaskLoss.Backward(true);
                        //PredictionLoss.Backward();
                        using (TorchTensor RecLoss = MaskLoss.Add(PredictionLoss))
                        using (TorchTensor KLD = DecoderModel[i].KLD(PredictionCode, KLDWeight))
                        using (TorchTensor PoseLoss = EncoderModel[i].PoseLoss(InputPose, TensorRefDotProd[i]))
                        //using (TorchTensor PoseLossScaled = PoseLoss.Mul(1.0))
                        using (TorchTensor OverallLoss = RecLoss.Add(KLD))
                        using (TorchTensor OverallLoss2 = OverallLoss.Add(PoseLoss))
                        using (TorchTensor OverallLoss3 = OverallLoss2.Add(CodeLoss))
                        {
                            if (i == 0)
                            {
                                GPU.CopyDeviceToHost(KLD.DataPtr(), ResultKLD, 1);
                                if (KLDWeight != 0)
                                    ResultKLD[0] /= (float)KLDWeight;
                            }

                            if (i == 0)
                                GPU.CopyDeviceToHost(PoseLoss.DataPtr(), ResultLoss, 1);

                            OverallLoss3.Backward();
                        }
                    }
                }
            }//, null);

            GatherGrads();

            OptimizerDecoder.Step();
            OptimizerEncoder.Step();

            prediction = ResultPredicted;
            predictionDeconv = ResultPredictedDeconv;
            alignedTarget = ResultAlignedTarget;
            loss = ResultLoss;
            kld = ResultKLD;
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Helper.PathToFolder(path));

            EncoderModel[0].Save(path + ".enc");
            DecoderModel[0].Save(path + ".dec");
        }

        public void Load(string path)
        {
            for (int i = 0; i < NDevices; i++)
            {
                EncoderModel[i].Load(path + ".enc", DeviceType.CUDA, Devices[i]);
                DecoderModel[i].Load(path + ".dec", DeviceType.CUDA, Devices[i]);
            }
        }

        ~C2DNet()
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

                        EncoderModel[i].Dispose();
                        DecoderModel[i].Dispose();
                    }

                    OptimizerEncoder.Dispose();
                    OptimizerDecoder.Dispose();
                }
            }
        }
    }
}
