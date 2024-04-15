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
    public class DistanceNet
    {
        public readonly int2 BoxDimensions;
        public readonly int CodeLength;
        public readonly int BatchSize = 8;
        public readonly int DeviceBatch = 8;
        public readonly int[] Devices;
        public readonly int NDevices;

        private TorchSharp.NN.DistanceNet[] Models;

        private TorchTensor[] TensorReference;
        private TorchTensor[] TensorData;
        private TorchTensor[] TensorDiff;

        private Optimizer Optimizer;
        private Loss Loss;

        private float[] ResultPredicted;
        private Image DebugReference, DebugData;
        private float[] ResultLoss = new float[1];

        private bool IsDisposed = false;

        public DistanceNet(int2 boxDimensions, int[] devices, int batchSize = 8)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;

            Models = new TorchSharp.NN.DistanceNet[NDevices];

            TensorReference = new TorchTensor[NDevices];
            TensorData = new TorchTensor[NDevices];
            TensorDiff = new TorchTensor[NDevices];

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];

                Models[i] = Modules.DistanceNet();
                Models[i].ToCuda(DeviceID);

                TensorReference[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorData[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);

                TensorDiff[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1 }, DeviceType.CUDA, DeviceID);
            }, null);

            Optimizer = Optimizer.Adam(Models[0].GetParameters(), 0.01, 1e-4);
            Loss = Losses.MSE();

            ResultPredicted = new float[batchSize];
            DebugReference = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, 1));
            DebugData = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, 1));
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
                Models[0].SynchronizeTo(Models[i], Devices[i]);
            }
        }

        private void GatherGrads()
        {
            for (int i = 1; i < NDevices; i++)
            {
                Models[0].GatherGrad(Models[i]);
            }
        }

        public void Predict(Image reference,
                            Image data,
                            out float[] prediction,
                            out Image debugReference,
                            out Image debugData)
        {
            DebugReference.GetDevice(Intent.Write);
            DebugData.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Models[i].Eval();

                GPU.CopyDeviceToDevice(reference.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorReference[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(data.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorData[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());

                using (TorchTensor Prediction = Models[i].Forward(TensorReference[i], 
                                                                  TensorData[i], 
                                                                  DebugReference.GetDeviceSlice(i * DeviceBatch, Intent.Write), 
                                                                  DebugData.GetDeviceSlice(i * DeviceBatch, Intent.Write)))
                {
                    float[] ThreadPrediction = new float[DeviceBatch];
                    GPU.CopyDeviceToHost(Prediction.DataPtr(),
                                         ThreadPrediction,
                                         ThreadPrediction.Length);
                    Array.Copy(ThreadPrediction, 0, ResultPredicted, i * DeviceBatch, DeviceBatch);
                }
            }, null);

            prediction = ResultPredicted;
            debugReference = DebugReference;
            debugData = DebugData;
        }

        public void Train(Image reference,
                                   Image data,
                                   Image diff,
                                   float learningRate,
                                   out float[] prediction,
                                   out Image debugReference,
                                   out Image debugData,
                                   out float[] loss)
        {
            Optimizer.SetLearningRateAdam(learningRate);
            Optimizer.ZeroGrad();

            SyncParams();
            DebugReference.GetDevice(Intent.Write);
            DebugData.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                if (learningRate > 0)
                    Models[i].Train();
                else
                    Models[i].Eval();

                Models[i].ZeroGrad();

                GPU.CopyDeviceToDevice(reference.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorReference[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(data.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorData[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(diff.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorDiff[i].DataPtr(),
                                       DeviceBatch);

                using (TorchTensor Prediction = Models[i].Forward(TensorReference[i],
                                                                  TensorData[i],
                                                                  DebugReference.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                                                  DebugData.GetDeviceSlice(i * DeviceBatch, Intent.Write)))
                using (TorchTensor PredictionLoss = Loss(Prediction, TensorDiff[i]))
                {
                    float[] ThreadPrediction = new float[DeviceBatch];
                    GPU.CopyDeviceToHost(Prediction.DataPtr(),
                                         ThreadPrediction,
                                         ThreadPrediction.Length);
                    Array.Copy(ThreadPrediction, 0, ResultPredicted, i * DeviceBatch, DeviceBatch);

                    if (i == 0)
                        GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLoss, 1);

                    if (learningRate > 0)
                        PredictionLoss.Backward();
                }
            }, null);

            GatherGrads();

            if (learningRate > 0)
                Optimizer.Step();

            prediction = ResultPredicted;
            debugReference = DebugReference;
            debugData = DebugData;
            loss = ResultLoss;
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Helper.PathToFolder(path));

            Models[0].Save(path);
        }

        public void Load(string path)
        {
            for (int i = 0; i < NDevices; i++)
            {
                Models[i].Load(path, DeviceType.CUDA, Devices[i]);
            }
        }

        ~DistanceNet()
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

                    DebugReference.Dispose();
                    DebugData.Dispose();

                    for (int i = 0; i < NDevices; i++)
                    {
                        TensorReference[i].Dispose();
                        TensorData[i].Dispose();

                        TensorDiff[i].Dispose();
                    }

                    Optimizer.Dispose();
                }
            }
        }
    }
}
