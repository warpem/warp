using System;
using System.IO;
using TorchSharp;
using TorchSharp.Tensor;
using static TorchSharp.NN.Modules;

namespace Warp
{
    /// <summary>
    /// JunkNet, a junk segmentation network (carbon, gold, ice), on one GPU. Inference only, in float32, on whole
    /// micrographs at 6 Å/pixel: its attention is global, so tiling would change the result.
    /// </summary>
    public sealed class JunkNetTorch : IDisposable
    {
        /// <summary>Sampling the network was trained at, Å/pixel.</summary>
        public const float PixelSize = 6;
        public const string ModelFileName = "junknet.pt";
        public const string ModelVariable = "WARP_JUNKNET";
        /// <summary>Changes whenever the installed model or the way its input is prepared changes what it segments.</summary>
        public const int Revision = 1;

        /// <summary>Where this installation keeps the model: models/ beside the executables, or the file WARP_JUNKNET names.</summary>
        public static string InstalledPath =>
            Environment.GetEnvironmentVariable(ModelVariable) is { Length: > 0 } path ? path : Path.Combine(AppContext.BaseDirectory, "models", ModelFileName);

        /// <summary>The model file to load: <paramref name="path"/>, or the installed model when it is empty.</summary>
        public static string ResolvePath(string path)
        {
            if (!string.IsNullOrEmpty(path)) return path;
            string installed = InstalledPath;
            if (!File.Exists(installed))
                throw new FileNotFoundException($"The junk segmentation model is not installed: put {ModelFileName} at {installed}, or set {ModelVariable} to its path.", installed);
            return installed;
        }

        readonly TorchSharp.NN.JunkNet Network;
        readonly int Device;
        bool Disposed;

        public JunkNetTorch(int device)
        {
            Device = device;
            Network = JunkNet();
            Network.ToCuda(device);
        }

        public void Load(string path = null) => Network.LoadStateDict(ResolvePath(path));

        /// <summary>Junk logits (junk where positive) for one normalized width × height micrograph (<see cref="JunkNetInput"/>).</summary>
        public float[] Predict(float[] image, int width, int height)
        {
            if (image.Length != width * height) throw new ArgumentException("Image size does not match its dimensions.");
            using var mode = new InferenceMode(true);
            Network.Eval();
            using var input = Float32Tensor.Zeros(new long[] { 1, 1, height, width }, DeviceType.CUDA, Device);
            GPU.CopyHostToDevice(image, input.DataPtr(), image.Length);
            using var output = Network.Forward(input);
            var logits = new float[image.Length];
            GPU.CopyDeviceToHost(output.DataPtr(), logits, logits.Length);
            return logits;
        }

        public void Dispose()
        {
            if (Disposed) return;
            Disposed = true;
            Network.Dispose();
        }
    }
}
