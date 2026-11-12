using System;
using TorchSharp;
using TorchSharp.NN;
using TorchSharp.Tensor;
using static TorchSharp.NN.Losses;
using static TorchSharp.NN.Modules;

namespace Warp
{
    /// <summary>
    /// Noise2Mic's micrograph denoiser (R2-U-Net, 512-pixel patches, 32 base features, 4 blocks per stage, t = 3)
    /// on one GPU. Checkpoints are PyTorch state dicts: Noise2Mic's own checkpoints load here, and models saved
    /// here load in Noise2Mic.
    /// </summary>
    public sealed class Noise2MicTorch : IDisposable
    {
        public const int PatchSize = 512, Features = 32, Depth = 4, Recurrence = 3;
        /// <summary>Training and inference sampling, Å/pixel.</summary>
        public const float PixelSize = 6;
        public const float DeconvolutionHighpass = 600;

        readonly R2UNet2D Network;
        readonly int Device;
        Optimizer Optimizer;
        readonly Loss Criterion = MSE(Reduction.Mean);
        bool Disposed;

        /// <summary>Training autocast: bfloat16 where supported (no loss scaling needed), float32 otherwise.</summary>
        public Precision TrainingPrecision { get; set; }
        /// <summary>Inference autocast: float16 convolutions with float32 normalization.</summary>
        public Precision InferencePrecision { get; set; } = Precision.Float16;

        public Noise2MicTorch(int device)
        {
            Device = device;
            Network = R2UNet2D(PatchSize, Features, Depth, Recurrence);
            Network.ToCuda(device);
            TrainingPrecision = TorchSharp.NN.R2UNet2D.SupportsBFloat16(device) ? Precision.BFloat16 : Precision.Float32;
        }

        public void Load(string path) => Network.LoadStateDict(path);

        public void Save(string path, float deconvStrength, float deconvFalloff) =>
            Network.SaveStateDict(path, PatchSize, Features, Depth, Recurrence, deconvStrength, deconvFalloff, DeconvolutionHighpass);

        TorchTensor Upload(float[] data, int count)
        {
            var tensor = Float32Tensor.Zeros(new long[] { count, 1, PatchSize, PatchSize }, DeviceType.CUDA, Device);
            GPU.CopyHostToDevice(data, tensor.DataPtr(), (long)count * PatchSize * PatchSize);
            return tensor;
        }

        /// <summary>Samples per forward pass; halved when the GPU runs out of memory, and kept for later steps.</summary>
        public int MicroBatch { get; private set; }

        /// <summary>
        /// One Adam step on an MSE loss over count samples. The batch is accumulated in micro-batches: GroupNorm
        /// normalizes each sample alone, so this is the same step as one full batch, on GPUs too small for it.
        /// </summary>
        public float Train(float[] source, float[] target, int count, double learningRate, int microBatch)
        {
            Optimizer ??= Optimizer.Adam(Network.GetParameters(), learningRate, 0);
            Optimizer.SetLearningRateAdam(learningRate);
            Network.Train();
            if (MicroBatch == 0) MicroBatch = Math.Max(1, microBatch);
            while (true)
            {
                Optimizer.ZeroGrad();
                try
                {
                    float loss = Accumulate(source, target, count, MicroBatch);
                    Optimizer.Step();
                    return loss;
                }
                catch (System.Runtime.InteropServices.ExternalException e) when (MicroBatch > 1 && e.Message.Contains("out of memory", StringComparison.OrdinalIgnoreCase))
                {
                    MicroBatch /= 2;
                    Console.WriteLine($"Denoiser training ran out of GPU memory; continuing with {MicroBatch} samples per pass.");
                }
            }
        }

        float Accumulate(float[] source, float[] target, int count, int microBatch)
        {
            int size = PatchSize * PatchSize;
            double total = 0;
            for (int start = 0; start < count; start += microBatch)
            {
                int n = Math.Min(microBatch, count - start);
                using var input = Upload(source.AsSpan(start * size, n * size).ToArray(), n);
                using var expected = Upload(target.AsSpan(start * size, n * size).ToArray(), n);
                using var prediction = Network.Forward(input, TrainingPrecision);
                using var loss = Criterion(prediction, expected);
                // A mean over the full batch: each micro-batch contributes in proportion to its size.
                using var weighted = loss.Mul((float)n / count);
                weighted.Backward();
                total += loss.ToSingle() * n / count;
            }
            return (float)total;
        }

        /// <summary>Denoised patches for count PatchSize² inputs.</summary>
        public float[] Predict(float[] patches, int count)
        {
            using var mode = new InferenceMode(true);
            Network.Eval();
            using var input = Upload(patches, count);
            using var output = Network.Forward(input, InferencePrecision);
            var result = new float[(long)count * PatchSize * PatchSize];
            GPU.CopyDeviceToHost(output.DataPtr(), result, result.Length);
            return result;
        }

        public void Dispose()
        {
            if (Disposed) return;
            Disposed = true;
            Optimizer?.Dispose();
            Network.Dispose();
        }
    }
}
