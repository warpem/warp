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
    public class BoxNetMulti
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

        private TorchSharp.NN.BoxNetMulti[] Model;

        private TorchTensor[] TensorSource;
        private TorchTensor[] TensorTargetPick;
        private TorchTensor[] TensorTargetDenoise;
        private TorchTensor[] TensorClassWeights;

        private Loss[] Loss;
        private Optimizer OptimizerEncoder;
        private Optimizer OptimizerDecoderPick;
        private Optimizer OptimizerDecoderDenoise;

        private Image ResultPredictedArgmax;
        private Image ResultPredictedSoftmax;
        private Image ResultPredictedDenoised;
        private float[] h_ResultPredictedArgmax;
        private float[] h_ResultPredictedSoftmax;
        private float[] h_ResultPredictedDenoised;

        private bool IsDisposed = false;

        public BoxNetMulti(int2 boxDimensions, float[] classWeights, int[] devices, int batchSize = 8)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;

            Model = new TorchSharp.NN.BoxNetMulti[NDevices];
            TensorSource = new TorchTensor[NDevices];
            TensorTargetPick = new TorchTensor[NDevices];
            TensorTargetDenoise = new TorchTensor[NDevices];
            TensorClassWeights = new TorchTensor[NDevices];

            Loss = new Loss[NDevices];
            if (classWeights.Length != 3)
                throw new Exception();

            //Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            for (int i = 0; i < NDevices; i++)
            {
                int DeviceID = Devices[i];

                Model[i] = BoxNetMulti(3, 1, 1);
                Model[i].ToCuda(DeviceID);

                TensorSource[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorTargetPick[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 3, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorTargetDenoise[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);

                TensorClassWeights[i] = Float32Tensor.Zeros(new long[] { 3 }, DeviceType.CUDA, DeviceID);
                GPU.CopyHostToDevice(classWeights, TensorClassWeights[i].DataPtr(), 3);

                Loss[i] = CE(TensorClassWeights[i]);

            }//, null);
            OptimizerEncoder = Optimizer.SGD(Model[0].NamedParameters().Where(p => p.name.StartsWith("encoder")).Select(p => p.parameter).ToArray(), 1e-4, 0.9, false, 1e-4);
            OptimizerDecoderPick = Optimizer.SGD(Model[0].NamedParameters().Where(p => p.name.StartsWith("decoder") || p.name.StartsWith("final_conv")).Select(p => p.parameter).ToArray(), 1e-4, 0.9, false, 1e-4);
            OptimizerDecoderDenoise = Optimizer.SGD(Model[0].NamedParameters().Where(p => p.name.StartsWith("denoise")).Select(p => p.parameter).ToArray(), 1e-4, 0.9, false, 1e-4);

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

            OptimizerEncoder.SetLearningRateSGD(learningRate);
            OptimizerDecoderPick.SetLearningRateSGD(learningRate);

            SyncParams();
            ResultPredictedArgmax.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Model[i].Train();
                Model[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch * 3, Intent.Read),
                                       TensorTargetPick[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements() * 3);

                GPU.CheckGPUExceptions();

                using (TorchTensor TargetArgMax = TensorTargetPick[i].Argmax(1))
                using (TorchTensor Prediction = Model[i].PickForward(TensorSource[i]))
                using (TorchTensor PredictionLoss = Loss[i](Prediction, TargetArgMax))
                {
                    if (needOutput)
                    {
                        using (TorchTensor PredictionArgMax = Prediction.Argmax(1))
                        using (TorchTensor PredictionArgMaxFP = PredictionArgMax.ToType(ScalarType.Float32))
                        {
                            GPU.CopyDeviceToDevice(PredictionArgMaxFP.DataPtr(),
                                                   ResultPredictedArgmax.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                                   DeviceBatch * (int)BoxDimensions.Elements());
                        }
                    }

                    if (i == 0)
                        GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLoss, 1);
                    //Debug.WriteLine(i + ": " + ResultLoss[0]);

                    PredictionLoss.Backward();
                }
            }, null);

            GatherGrads();

            OptimizerEncoder.Step();
            if (!skipDecoder)
                OptimizerDecoderPick.Step();

            prediction = ResultPredictedArgmax;
            loss = ResultLoss;
        }

        public void TrainDenoise(Image source,
                                 Image target,
                                 float learningRate,
                                 bool skipDecoder,
                                 bool needOutput,
                                 out Image prediction,
                                 out float[] loss)
        {
            GPU.CheckGPUExceptions();

            var ResultLoss = new float[1];

            OptimizerEncoder.SetLearningRateSGD(learningRate);
            OptimizerDecoderDenoise.SetLearningRateSGD(learningRate);

            SyncParams();
            ResultPredictedDenoised.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Model[i].Train();
                Model[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTargetDenoise[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());

                GPU.CheckGPUExceptions();

                using (TorchTensor TargetArgMax = TensorTargetDenoise[i].Argmax(1))
                using (TorchTensor Prediction = Model[i].DenoiseForward(TensorSource[i]))
                using (TorchTensor PredictionLoss = Loss[i](Prediction, TargetArgMax))
                {
                    if (needOutput)
                    {
                        GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                               ResultPredictedDenoised.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * (int)BoxDimensions.Elements());
                    }

                    if (i == 0)
                        GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLoss, 1);

                    PredictionLoss.Backward();
                }
            }, null);

            GatherGrads();

            OptimizerEncoder.Step();
            if (!skipDecoder)
                OptimizerDecoderDenoise.Step();

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

        ~BoxNetMulti()
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
                        TensorTargetPick[i].Dispose();
                        TensorTargetDenoise[i].Dispose();
                        TensorClassWeights[i].Dispose();

                        Model[i].Dispose();
                    }

                    OptimizerEncoder.Dispose();
                    OptimizerDecoderPick.Dispose();
                    OptimizerDecoderDenoise.Dispose();
                }
            }
        }
    }
}
