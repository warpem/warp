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

namespace Warp.NNModels
{
    public class ParticleWGAN
    {
        public readonly int2 BoxDimensions;
        public readonly int CodeLength;
        public readonly int BatchSize = 8;
        public readonly int DeviceBatch = 8;
        public readonly int[] Devices;
        public readonly int NDevices;

        private ParticleWGANGenerator[] Generators;
        private ParticleWGANDiscriminator[] Discriminators;

        private TorchTensor[] TensorTrueImages;
        private TorchTensor[] TensorFakeImages;
        private TorchTensor[] TensorCTF;

        private TorchTensor[] TensorParticleCode;
        private TorchTensor[] TensorCrapCode;
        private TorchTensor[] TensorNoiseAdd;
        private TorchTensor[] TensorNoiseMul;

        private TorchTensor[] TensorOne;
        private TorchTensor[] TensorMinusOne;

        private TorchTensor[] TensorMask;

        private Optimizer OptimizerGen;
        private Optimizer OptimizerDisc;

        private Image ResultPredicted;
        private Image ResultPredictedNoisy;
        private float[] ResultLoss = new float[1];
        private float[] ResultLossDiscReal = new float[1];
        private float[] ResultLossDiscFake = new float[1];

        private bool IsDisposed = false;

        private double GenBoost = 10;

        public ParticleWGAN(int2 boxDimensions, int codeLength, int[] devices, int batchSize = 8)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;
            CodeLength = codeLength;

            Generators = new ParticleWGANGenerator[NDevices];
            Discriminators = new ParticleWGANDiscriminator[NDevices];

            TensorTrueImages = new TorchTensor[NDevices];
            TensorFakeImages = new TorchTensor[NDevices];
            TensorCTF = new TorchTensor[NDevices];

            TensorParticleCode = new TorchTensor[NDevices];
            TensorCrapCode = new TorchTensor[NDevices];
            TensorNoiseAdd = new TorchTensor[NDevices];
            TensorNoiseMul = new TorchTensor[NDevices];

            TensorOne = new TorchTensor[NDevices];
            TensorMinusOne = new TorchTensor[NDevices];

            TensorMask = new TorchTensor[NDevices];

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];

                Generators[i] = ParticleWGANGenerator(BoxDimensions.X, codeLength);
                Generators[i].ToCuda(DeviceID);

                Discriminators[i] = ParticleWGANDiscriminator();
                Discriminators[i].ToCuda(DeviceID);

                TensorTrueImages[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorFakeImages[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorCTF[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X / 2 + 1 }, DeviceType.CUDA, DeviceID);

                TensorParticleCode[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, CodeLength }, DeviceType.CUDA, DeviceID);
                TensorCrapCode[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, CodeLength }, DeviceType.CUDA, DeviceID);
                TensorNoiseAdd[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorNoiseMul[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);

                TensorOne[i] = Float32Tensor.Ones(new long[] { }, DeviceType.CUDA, DeviceID);
                TensorMinusOne[i] = Float32Tensor.Ones(new long[] { }, DeviceType.CUDA, DeviceID);
                {
                    GPU.CopyHostToDevice(new float[] { -1f }, TensorMinusOne[i].DataPtr(), 1);
                }

                TensorMask[i] = Float32Tensor.Zeros(new long[] { 1, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                {
                    Image Mask = new Image(new int3(BoxDimensions.X, BoxDimensions.Y, 1));
                    Mask.Fill(1);
                    Mask.MaskSpherically(BoxDimensions.X / 2, BoxDimensions.X / 8, false);

                    GPU.CopyDeviceToDevice(Mask.GetDevice(Intent.Read), TensorMask[i].DataPtr(), Mask.ElementsReal);

                    Mask.Dispose();
                }
            }, null);

            OptimizerGen = Optimizer.Adam(Generators[0].GetParameters(), 0.01, 1e-4);
            OptimizerDisc = Optimizer.Adam(Discriminators[0].GetParameters(), 0.01, 1e-4);

            ResultPredicted = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
            ResultPredictedNoisy = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
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
                Generators[0].SynchronizeTo(Generators[i], Devices[i]);
                Discriminators[0].SynchronizeTo(Discriminators[i], Devices[i]);
            }
        }

        private void GatherGrads()
        {
            for (int i = 1; i < NDevices; i++)
            {
                Generators[0].GatherGrad(Generators[i]);
                Discriminators[0].GatherGrad(Discriminators[i]);
            }
        }

        public void Predict(Image imagesFake,
                                   Image imagesCTF,
                                   out Image prediction)
        {
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Generators[i].Eval();

                GPU.CopyDeviceToDevice(imagesFake.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorFakeImages[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());

                TensorCrapCode[i].RandomNInPlace(TensorCrapCode[i].Shape);

                using (TorchTensor Prediction = Generators[i].ForwardNoise(TensorCrapCode[i], TensorFakeImages[i], TensorCTF[i]))
                {
                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                }
            }, null);

            prediction = ResultPredicted;
        }

        public void TrainGenerator(Image imagesFake,
                                   Image imagesCTF,
                                   float learningRate,
                                   out Image prediction,
                                   out float[] loss)
        {
            OptimizerGen.SetLearningRateAdam(learningRate);
            OptimizerGen.ZeroGrad();

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Generators[i].Train();
                Generators[i].ZeroGrad();

                GPU.CopyDeviceToDevice(imagesFake.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorFakeImages[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());

                TensorCrapCode[i].RandomNInPlace(TensorCrapCode[i].Shape);

                using (TorchTensor Prediction = Generators[i].ForwardNoise(TensorCrapCode[i], TensorFakeImages[i], TensorCTF[i]))
                using (TorchTensor IsItReal = Discriminators[i].Forward(Prediction))
                using (TorchTensor Loss = IsItReal.Mean())
                {
                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    if (i == 0)
                        GPU.CopyDeviceToHost(Loss.DataPtr(), ResultLoss, 1);

                    Loss.Backward(TensorOne[i]);
                }
            }, null);

            GatherGrads();

            OptimizerGen.Step();

            prediction = ResultPredicted;
            loss = ResultLoss;
        }

        public void TrainDiscriminator(Image imagesReal,
                                       Image imagesFake,
                                       Image imagesCTF,
                                       float learningRate,
                                       float penaltyLambda,
                                       out Image prediction,
                                       out float[] lossWasserstein,
                                       out float[] lossReal,
                                       out float[] lossFake)
        {
            OptimizerDisc.SetLearningRateAdam(learningRate);
            OptimizerDisc.ZeroGrad();

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Discriminators[i].Train();
                Discriminators[i].ZeroGrad();

                //Discriminators[i].ClipWeights(weightClip);

                GPU.CopyDeviceToDevice(imagesReal.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTrueImages[i].DataPtr(),
                                       BoxDimensions.Elements() * DeviceBatch);
                GPU.CopyDeviceToDevice(imagesFake.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorFakeImages[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());

                float LossWasserstein = 0;

                using (TorchTensor IsItReal = Discriminators[i].Forward(TensorTrueImages[i]))
                using (TorchTensor Loss = IsItReal.Mean())
                {
                    if (i == 0)
                    {
                        GPU.CopyDeviceToHost(Loss.DataPtr(), ResultLossDiscReal, 1);
                        LossWasserstein = ResultLossDiscReal[0];
                    }

                    Loss.Backward(TensorOne[i]);
                }

                TensorCrapCode[i].RandomNInPlace(TensorCrapCode[i].Shape);

                using (TorchTensor Prediction = Generators[i].ForwardNoise(TensorCrapCode[i], TensorFakeImages[i], TensorCTF[i]))
                using (TorchTensor PredictionDetached = Prediction.Detach())
                using (TorchTensor IsItReal = Discriminators[i].Forward(PredictionDetached))
                using (TorchTensor Loss = IsItReal.Mean())
                {
                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    if (i == 0)
                    {
                        GPU.CopyDeviceToHost(Loss.DataPtr(), ResultLossDiscFake, 1);
                        LossWasserstein = LossWasserstein - ResultLossDiscFake[0];
                        ResultLoss[0] = LossWasserstein;
                    }

                    Loss.Backward(TensorMinusOne[i]);

                    using (TorchTensor Penalty = Discriminators[i].PenalizeGradient(TensorTrueImages[i], Prediction, penaltyLambda))
                    {
                        Penalty.Backward();
                    }
                }
            }, null);

            GatherGrads();

            OptimizerDisc.Step();

            prediction = ResultPredicted;
            lossWasserstein = ResultLoss;
            lossReal = ResultLossDiscReal;
            lossFake = ResultLossDiscFake;
        }

        public void TrainGeneratorParticle(Image imagesCTF,
                                           float learningRate,
                                           out Image prediction,
                                           out Image predictionNoisy,
                                           out float[] loss)
        {
            OptimizerGen.SetLearningRateAdam(learningRate * GenBoost);
            OptimizerGen.ZeroGrad();

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Generators[i].Train();
                Generators[i].ZeroGrad();

                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());

                TensorParticleCode[i].RandomNInPlace(TensorParticleCode[i].Shape);
                TensorCrapCode[i].RandomNInPlace(TensorCrapCode[i].Shape);

                using (TorchTensor Prediction = Generators[i].ForwardParticle(TensorParticleCode[i], true, 0.5 * (2f / BoxDimensions.X)))
                using (TorchTensor PredictionFT = Prediction.rfftn(new long[] { 2, 3 }))
                using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                using (TorchTensor PredictionConv = PredictionFTConv.irfftn(new long[] { 2, 3 }))
                using (TorchTensor PredictionNoisy = Generators[i].ForwardNoise(TensorCrapCode[i], PredictionConv, TensorCTF[i]))
                using (TorchTensor PredictionMasked = PredictionNoisy.Mul(TensorMask[i]))
                using (TorchTensor IsItReal = Discriminators[i].Forward(PredictionMasked))
                using (TorchTensor Loss = IsItReal.Mean())
                {
                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    GPU.CopyDeviceToDevice(PredictionMasked.DataPtr(),
                                           ResultPredictedNoisy.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    if (i == 0)
                        GPU.CopyDeviceToHost(Loss.DataPtr(), ResultLoss, 1);

                    Loss.Backward(TensorOne[i]);
                }
            }, null);

            GatherGrads();

            OptimizerGen.Step();

            prediction = ResultPredicted;
            predictionNoisy = ResultPredictedNoisy;
            loss = ResultLoss;
        }

        public void TrainDiscriminatorParticle(Image imagesReal,
                                               Image imagesCTF,
                                               float learningRate,
                                               float penaltyLambda,
                                               out Image prediction,
                                               out float[] lossWasserstein,
                                               out float[] lossReal,
                                               out float[] lossFake)
        {
            OptimizerDisc.SetLearningRateAdam(learningRate);
            OptimizerDisc.ZeroGrad();

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Discriminators[i].Train();
                Discriminators[i].ZeroGrad();

                //Discriminators[i].ClipWeights(weightClip);

                GPU.CopyDeviceToDevice(imagesReal.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTrueImages[i].DataPtr(),
                                       BoxDimensions.Elements() * DeviceBatch);
                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());

                float LossWasserstein = 0;

                using (TorchTensor IsItReal = Discriminators[i].Forward(TensorTrueImages[i]))
                using (TorchTensor Loss = IsItReal.Mean())
                {
                    if (i == 0)
                    {
                        GPU.CopyDeviceToHost(Loss.DataPtr(), ResultLossDiscReal, 1);
                        LossWasserstein = ResultLossDiscReal[0];
                    }

                    Loss.Backward(TensorOne[i]);
                }

                TensorParticleCode[i].RandomNInPlace(TensorParticleCode[i].Shape);
                TensorCrapCode[i].RandomNInPlace(TensorCrapCode[i].Shape);

                using (TorchTensor Prediction = Generators[i].ForwardParticle(TensorParticleCode[i], true, 0.5 * (2f / BoxDimensions.X)))
                using (TorchTensor PredictionFT = Prediction.rfftn(new long[] { 2, 3 }))
                using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                using (TorchTensor PredictionConv = PredictionFTConv.irfftn(new long[] { 2, 3 }))
                using (TorchTensor PredictionNoisy = Generators[i].ForwardNoise(TensorCrapCode[i], PredictionConv, TensorCTF[i]))
                using (TorchTensor PredictionMasked = PredictionNoisy.Mul(TensorMask[i]))
                using (TorchTensor PredictionDetached = PredictionMasked.Detach())
                using (TorchTensor IsItReal = Discriminators[i].Forward(PredictionDetached))
                using (TorchTensor Loss = IsItReal.Mean())
                {
                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    if (i == 0)
                    {
                        GPU.CopyDeviceToHost(Loss.DataPtr(), ResultLossDiscFake, 1);
                        LossWasserstein = LossWasserstein - ResultLossDiscFake[0];
                        ResultLoss[0] = LossWasserstein;
                    }

                    Loss.Backward(TensorMinusOne[i]);

                    using (TorchTensor Penalty = Discriminators[i].PenalizeGradient(TensorTrueImages[i], PredictionMasked, penaltyLambda))
                    {
                        Penalty.Backward();
                    }
                }
            }, null);

            GatherGrads();

            OptimizerDisc.Step();

            prediction = ResultPredicted;
            lossWasserstein = ResultLoss;
            lossReal = ResultLossDiscReal;
            lossFake = ResultLossDiscFake;
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Helper.PathToFolder(path));

            Generators[0].Save(path + ".gen");
            Discriminators[0].Save(path + ".disc");
        }

        public void Load(string path)
        {
            for (int i = 0; i < NDevices; i++)
            {
                Generators[i].Load(path + ".gen", DeviceType.CUDA, Devices[i]);
                //Generators[i].ToCuda(Devices[i]);

                Discriminators[i].Load(path + ".disc", DeviceType.CUDA, Devices[i]);
                //Discriminators[i].ToCuda(Devices[i]);
            }
        }

        ~ParticleWGAN()
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
                        TensorCrapCode[i].Dispose();
                        TensorTrueImages[i].Dispose();
                        TensorCTF[i].Dispose();

                        TensorOne[i].Dispose();
                        TensorMinusOne[i].Dispose();

                        Generators[i].Dispose();
                        Discriminators[i].Dispose();
                    }

                    OptimizerGen.Dispose();
                    OptimizerDisc.Dispose();
                }
            }
        }
    }
}
