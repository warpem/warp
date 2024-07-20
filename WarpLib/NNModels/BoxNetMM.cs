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
using System.Diagnostics;

namespace Warp
{
    public class BoxNetMM
    {
        public const float PixelSize = 8;

        public static readonly int2 DefaultDimensionsTrain = new int2(512);
        public static readonly int2 DefaultDimensionsValidTrain = new int2(384);
        public static readonly int2 DefaultDimensionsPredict = new int2(512);
        public static readonly int2 DefaultDimensionsValidPredict = new int2(384);
        public static readonly int DefaultBatchTrain = 24;
        public static readonly int DefaultBatchPredict = 1;

        public readonly int2 BoxDimensions;
        public readonly int BatchSize = 24;
        public readonly int DeviceBatch = 24;
        public readonly int[] Devices;
        public readonly int NDevices;

        private TorchSharp.NN.BoxNetMM[] Model;

        private TorchTensor[] TensorSource;
        private TorchTensor[] TensorTargetPick;
        private TorchTensor[] TensorTargetDenoise;
        private TorchTensor[] TensorClassWeights;

        private Loss[] LossPick;
        private Loss[] LossDenoise;
        private Optimizer OptimizerEncoderPick;
        private Optimizer OptimizerEncoderFill;
        private Optimizer OptimizerDecoderPick;
        private Optimizer OptimizerDecoderFill;

        private Optimizer OptimizerDenoise;

        private BoxNetOptimizer OptimizerType;

        private Image ResultPredictedArgmax;
        private Image ResultPredictedSoftmax;
        private Image ResultPredictedDenoised;
        private float[] h_ResultPredictedArgmax;
        private float[] h_ResultPredictedSoftmax;
        private float[] h_ResultPredictedDenoised;

        private bool IsDisposed = false;

        public BoxNetMM(int2 boxDimensions, float[] classWeights, int[] devices, int batchSize = 8, BoxNetOptimizer optimizerType = BoxNetOptimizer.None)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;

            Model = new TorchSharp.NN.BoxNetMM[NDevices];
            TensorSource = new TorchTensor[NDevices];
            TensorTargetPick = new TorchTensor[NDevices];
            TensorTargetDenoise = new TorchTensor[NDevices];
            TensorClassWeights = new TorchTensor[NDevices];

            LossPick = new Loss[NDevices];
            LossDenoise = new Loss[NDevices];
            if (classWeights.Length != 3)
                throw new Exception();

            //Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            for (int i = 0; i < NDevices; i++)
            {
                int DeviceID = Devices[i];

                Model[i] = TorchSharp.NN.Modules.BoxNetMM(1, 1, 1);
                Model[i].ToCuda(DeviceID);

                TensorSource[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);

                if (optimizerType != BoxNetOptimizer.None)
                {
                    TensorTargetPick[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 3, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                    TensorTargetDenoise[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);

                    TensorClassWeights[i] = Float32Tensor.Zeros(new long[] { 3 }, DeviceType.CUDA, DeviceID);
                    GPU.CopyHostToDevice(classWeights, TensorClassWeights[i].DataPtr(), 3);

                    LossPick[i] = CE(TensorClassWeights[i]);
                    LossDenoise[i] = MSE(Reduction.Mean);
                }

            }//, null);

            OptimizerType = optimizerType;

            TorchTensor[] ParamsEncoder = Model[0].NamedParameters().Where(p => p.name.StartsWith("encoder")).Select(p => p.parameter).ToArray();
            TorchTensor[] ParamsDecoderPick = Model[0].NamedParameters().Where(p => p.name.StartsWith("decoder") || p.name.StartsWith("final_conv")).Select(p => p.parameter).ToArray();
            TorchTensor[] ParamsDecoderFill = Model[0].NamedParameters().Where(p => p.name.StartsWith("fill")).Select(p => p.parameter).ToArray();
            TorchTensor[] ParamsDenoise = Model[0].NamedParameters().Where(p => p.name.StartsWith("denoise")).Select(p => p.parameter).ToArray();

            if (OptimizerType == BoxNetOptimizer.SGD)
            {
                OptimizerEncoderPick = Optimizer.SGD(ParamsEncoder, 1e-4, 0.9, false, 1e-4);
                OptimizerEncoderFill = Optimizer.SGD(ParamsEncoder, 1e-4, 0.9, false, 1e-4);
                OptimizerDecoderPick = Optimizer.SGD(ParamsDecoderPick, 1e-4, 0.9, false, 1e-4);
                OptimizerDecoderFill = Optimizer.SGD(ParamsDecoderFill, 1e-4, 0.9, false, 1e-4);
            }
            else if (OptimizerType == BoxNetOptimizer.Adam)
            {
                OptimizerEncoderPick = Optimizer.Adam(ParamsEncoder, 1e-4, 1e-4);
                OptimizerEncoderFill = Optimizer.Adam(ParamsEncoder, 1e-4, 1e-4);
                OptimizerDecoderPick = Optimizer.Adam(ParamsDecoderPick, 1e-4, 1e-4);
                OptimizerDecoderFill = Optimizer.Adam(ParamsDecoderFill, 1e-4, 1e-4);
            }

            OptimizerDenoise = Optimizer.Adam(ParamsDenoise, 1e-4, 1e-4);

            ResultPredictedArgmax = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
            ResultPredictedSoftmax = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize * 3));
            ResultPredictedDenoised = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));

            h_ResultPredictedArgmax = new float[ResultPredictedArgmax.ElementsReal];
            h_ResultPredictedSoftmax = new float[ResultPredictedSoftmax.ElementsReal];
            h_ResultPredictedDenoised = new float[ResultPredictedDenoised.ElementsReal];
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
                Model[0].SynchronizeTo(Model[i], Devices[i]);
        }

        private void GatherGrads()
        {
            for (int i = 1; i < NDevices; i++)
                Model[0].GatherGrad(Model[i]);
        }

        public void PredictPick(Image data, out Image predictionArgmax, out Image predictionSoftmax)
        {
            ScatterData(data, TensorSource);
            ResultPredictedSoftmax.GetDevice(Intent.Write);
            ResultPredictedArgmax.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                using (var mode = new InferenceMode(true))
                {
                    Model[i].Eval();

                    using (TorchTensor t_Prediction = Model[i].PickForward(TensorSource[i]))
                    using (TorchTensor t_PredictionSoftMax = t_Prediction.Softmax(1))
                    using (TorchTensor t_PredictionArgMax = t_PredictionSoftMax.Argmax(1, false))
                    using (TorchTensor t_PredictionArgMaxFP = t_PredictionArgMax.ToType(ScalarType.Float32))
                    {
                        GPU.CopyDeviceToDevice(t_PredictionSoftMax.DataPtr(),
                                               ResultPredictedSoftmax.GetDeviceSlice(i * DeviceBatch * 3, Intent.Write),
                                               DeviceBatch * BoxDimensions.Elements() * 3);

                        GPU.CopyDeviceToDevice(t_PredictionArgMaxFP.DataPtr(),
                                               ResultPredictedArgmax.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * BoxDimensions.Elements());
                    }
                }
            }, null);

            predictionArgmax = ResultPredictedArgmax;
            predictionSoftmax = ResultPredictedSoftmax;
        }

        public void PredictPick(Image data, out float[] h_predictionArgmax, out float[] h_predictionSoftmax)
        {
            PredictPick(data, out ResultPredictedArgmax, out ResultPredictedSoftmax);

            GPU.CopyDeviceToHost(ResultPredictedSoftmax.GetDevice(Intent.Read),
                                 h_ResultPredictedSoftmax,
                                 BatchSize * BoxDimensions.Elements() * 3);

            GPU.CopyDeviceToHost(ResultPredictedArgmax.GetDevice(Intent.Read),
                                 h_ResultPredictedArgmax,
                                 BatchSize * BoxDimensions.Elements());

            h_predictionArgmax = h_ResultPredictedArgmax;
            h_predictionSoftmax = h_ResultPredictedSoftmax;
        }
        
        public void PredictDenoise(Image data, out Image predictionDenoised)
        {
            ScatterData(data, TensorSource);
            ResultPredictedDenoised.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                using (var mode = new InferenceMode(true))
                {
                    Model[i].Eval();

                    using (TorchTensor t_Prediction = Model[i].DenoiseForward(TensorSource[i]))
                    {
                        GPU.CopyDeviceToDevice(t_Prediction.DataPtr(),
                                               ResultPredictedDenoised.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * BoxDimensions.Elements());
                    }
                }
            }, null);

            predictionDenoised = ResultPredictedDenoised;
        }

        public void PredictDenoise(Image data, out float[] h_predictionDenoised)
        {
            PredictDenoise(data, out ResultPredictedDenoised);

            GPU.CopyDeviceToHost(ResultPredictedArgmax.GetDevice(Intent.Read),
                                 h_ResultPredictedDenoised,
                                 BatchSize * BoxDimensions.Elements());

            h_predictionDenoised = h_ResultPredictedDenoised;
        }

        public void PredictFill(Image data, out Image predictionFilled)
        {
            ScatterData(data, TensorSource);
            ResultPredictedDenoised.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                using (var mode = new InferenceMode(true))
                {
                    Model[i].Eval();

                    using (TorchTensor t_Prediction = Model[i].DenoiseForward(TensorSource[i]))
                    {
                        GPU.CopyDeviceToDevice(t_Prediction.DataPtr(),
                                               ResultPredictedDenoised.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * BoxDimensions.Elements());
                    }
                }
            }, null);

            predictionFilled = ResultPredictedDenoised;
        }

        public void TrainPick(Image source,
                              Image target,
                              float learningRate,
                              bool skipDecoder,
                              bool needOutput,
                              out Image prediction,
                              out float[] loss)
        {
            GPU.CheckGPUExceptions();

            var ResultLoss = new float[1];

            if (OptimizerType == BoxNetOptimizer.SGD)
            {
                OptimizerEncoderPick.SetLearningRateSGD(learningRate);
                OptimizerDecoderPick.SetLearningRateSGD(learningRate);
            }
            else if (OptimizerType == BoxNetOptimizer.Adam)
            {
                OptimizerEncoderPick.SetLearningRateAdam(learningRate);
                OptimizerDecoderPick.SetLearningRateAdam(learningRate);
            }
            else
                throw new Exception("Invalid optimizer type for training.");

            SyncParams();
            ResultPredictedArgmax.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Model[i].Train();
                Model[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch * 3, Intent.Read),
                                       TensorTargetPick[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements() * 3);

                GPU.CheckGPUExceptions();

                using (TorchTensor TargetArgMax = TensorTargetPick[i].Argmax(1))
                using (TorchTensor Prediction = Model[i].PickForward(TensorSource[i]))
                using (TorchTensor PredictionLoss = LossPick[i](Prediction, TargetArgMax))
                {
                    if (needOutput)
                    {
                        using (TorchTensor PredictionArgMax = Prediction.Argmax(1))
                        using (TorchTensor PredictionArgMaxFP = PredictionArgMax.ToType(ScalarType.Float32))
                        {
                            GPU.CopyDeviceToDevice(PredictionArgMaxFP.DataPtr(),
                                                   ResultPredictedArgmax.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                                   DeviceBatch * BoxDimensions.Elements());
                        }
                    }

                    if (i == 0)
                        GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLoss, 1);
                    //Debug.WriteLine(i + ": " + ResultLoss[0]);

                    PredictionLoss.Backward();
                }
            }, null);

            GatherGrads();

            OptimizerEncoderPick.Step();
            if (!skipDecoder)
                OptimizerDecoderPick.Step();

            prediction = ResultPredictedArgmax;
            loss = ResultLoss;
        }

        public void TrainDenoise(Image source,
                                 Image target,
                                 float consistencyWeight,
                                 float learningRate,
                                 bool needOutput,
                                 out Image prediction,
                                 out float[] loss)
        {
            GPU.CheckGPUExceptions();

            var ResultLoss = new float[1];

            if (OptimizerType == BoxNetOptimizer.SGD)
            {
                OptimizerEncoderFill.SetLearningRateSGD(learningRate);
                OptimizerDecoderFill.SetLearningRateSGD(learningRate);
            }
            else if (OptimizerType == BoxNetOptimizer.Adam)
            {
                OptimizerEncoderFill.SetLearningRateAdam(learningRate);
                OptimizerDecoderFill.SetLearningRateAdam(learningRate);
            }
            else
                throw new Exception("Invalid optimizer type for training.");

            OptimizerDenoise.SetLearningRateAdam(learningRate);

            SyncParams();
            ResultPredictedDenoised.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Model[i].Train();
                Model[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTargetDenoise[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());

                GPU.CheckGPUExceptions();

                using (TorchTensor PredictionOdd = Model[i].DenoiseForward(TensorSource[i]))
                using (TorchTensor PredictionOddLoss = LossDenoise[i](PredictionOdd, TensorTargetDenoise[i]))
                {
                    if (needOutput)
                    {
                        GPU.CopyDeviceToDevice(PredictionOdd.DataPtr(),
                                               ResultPredictedDenoised.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * BoxDimensions.Elements());
                    }

                    if (consistencyWeight > 0)
                    {
                        using (TorchTensor PredictionOddDetached = PredictionOdd.Detach())
                        using (TorchTensor PredictionEven = Model[i].DenoiseForward(TensorTargetDenoise[i]))
                        using (TorchTensor ConsistencyLoss = LossDenoise[i](PredictionEven, PredictionOddDetached))
                        using (TorchTensor ConsistencyWeighted = ConsistencyLoss.Mul(consistencyWeight))
                        using (TorchTensor OverallLoss = PredictionOddLoss.Add(ConsistencyWeighted))
                        {
                            if (i == 0)
                                GPU.CopyDeviceToHost(OverallLoss.DataPtr(), ResultLoss, 1);

                            OverallLoss.Backward();
                        }
                    }
                    else
                    {
                        if (i == 0)
                            GPU.CopyDeviceToHost(PredictionOddLoss.DataPtr(), ResultLoss, 1);

                        PredictionOddLoss.Backward();
                    }
                }
            }, null);

            GatherGrads();

            //OptimizerEncoderDenoise.Step();
            //if (!skipDecoder)
            //    OptimizerDecoderDenoise.Step();
            OptimizerDenoise.Step();

            prediction = ResultPredictedDenoised;
            loss = ResultLoss;
        }

        public void TrainFill(Image source,
                              Image target,
                              float learningRate,
                              bool skipDecoder,
                              bool needOutput,
                              out Image prediction,
                              out float[] loss)
        {
            GPU.CheckGPUExceptions();

            var ResultLoss = new float[1];

            if (OptimizerType == BoxNetOptimizer.SGD)
            {
                OptimizerEncoderFill.SetLearningRateSGD(learningRate);
                OptimizerDecoderFill.SetLearningRateSGD(learningRate);
            }
            else if (OptimizerType == BoxNetOptimizer.Adam)
            {
                OptimizerEncoderFill.SetLearningRateAdam(learningRate);
                OptimizerDecoderFill.SetLearningRateAdam(learningRate);
            }
            else
                throw new Exception("Invalid optimizer type for training.");

            SyncParams();
            ResultPredictedDenoised.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Model[i].Train();
                Model[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTargetDenoise[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());

                GPU.CheckGPUExceptions();

                using (TorchTensor Prediction = Model[i].DenoiseForward(TensorSource[i]))
                using (TorchTensor PredictioLoss = LossDenoise[i](Prediction, TensorTargetDenoise[i]))
                using (TorchTensor LossScaled = PredictioLoss.Mul(10))
                {
                    if (needOutput)
                        GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                               ResultPredictedDenoised.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * BoxDimensions.Elements());

                    if (i == 0)
                        GPU.CopyDeviceToHost(LossScaled.DataPtr(), ResultLoss, 1);

                    LossScaled.Backward();
                }
            }, null);

            GatherGrads();

            OptimizerEncoderFill.Step();
            if (!skipDecoder)
                OptimizerDecoderFill.Step();

            prediction = ResultPredictedDenoised;
            loss = ResultLoss;
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Helper.PathToFolder(path));

            Model[0].Save(path);
        }

        public void Load(string path)
        {
            for (int i = 0; i < NDevices; i++)
                Model[i].Load(path, DeviceType.CUDA, Devices[i]);
        }

        ~BoxNetMM()
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

                    ResultPredictedArgmax.Dispose();
                    ResultPredictedSoftmax.Dispose();
                    ResultPredictedDenoised.Dispose();

                    for (int i = 0; i < NDevices; i++)
                    {
                        TensorSource[i].Dispose();
                        TensorTargetPick[i]?.Dispose();
                        TensorTargetDenoise[i]?.Dispose();
                        TensorClassWeights[i]?.Dispose();

                        Model[i].Dispose();
                    }

                    OptimizerEncoderPick?.Dispose();
                    OptimizerDecoderPick?.Dispose();
                    OptimizerDecoderFill?.Dispose();
                }
            }
        }
    }

    public enum BoxNetOptimizer
    {
        None = 0,
        SGD = 1,
        Adam = 2
    }
}
