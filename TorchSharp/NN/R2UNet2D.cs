using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>Noise2Mic's recurrent residual U-Net; parameter names match the PyTorch model.</summary>
    public class R2UNet2D : Module
    {
        internal R2UNet2D(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_R2UNet2D_forward(Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_R2UNet2D_forward_autocast(Module.HType module, IntPtr tensor, long precision);

        [DllImport("LibTorchSharp")]
        private static extern int THSTorch_cuda_supports_bf16(long device);

        [DllImport("LibTorchSharp")]
        private static extern void THSTorch_cuda_peak_memory(long device, out long allocated, out long reserved, bool reset);

        /// <summary>Peak allocated and reserved CUDA bytes since the last reset (optionally resetting).</summary>
        public static (long Allocated, long Reserved) PeakMemory(int device, bool reset)
        {
            THSTorch_cuda_peak_memory(device, out long allocated, out long reserved, reset);
            Torch.CheckForErrors();
            return (allocated, reserved);
        }

        /// <summary>Compute capability 8.0 or newer: bfloat16 autocast trains without loss scaling.</summary>
        public static bool SupportsBFloat16(int device) => THSTorch_cuda_supports_bf16(device) != 0;

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_Module_load_state_dict(Module.HType module, [MarshalAs(UnmanagedType.LPStr)] string location);

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_Module_save_state_dict(Module.HType module, [MarshalAs(UnmanagedType.LPStr)] string location,
            long patchSize, long features, long depth, long t, double deconvStrength, double deconvFalloff, double deconvHighpass);

        /// <summary>Forward pass under CUDA autocast; the output is float32 either way.</summary>
        public TorchTensor Forward(TorchTensor tensor, Precision precision)
        {
            var res = THSNN_R2UNet2D_forward_autocast(handle, tensor.Handle, (long)precision);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_R2UNet2D_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>Loads a PyTorch state dict (bare, or under "model_state_dict" as in Noise2Mic checkpoints).</summary>
        public void LoadStateDict(string path)
        {
            THSNN_Module_load_state_dict(handle, path);
            Torch.CheckForErrors();
        }

        /// <summary>Saves a checkpoint PyTorch's torch.load and Noise2Mic's inference read.</summary>
        public void SaveStateDict(string path, long patchSize, long features, long depth, long t, double deconvStrength, double deconvFalloff, double deconvHighpass)
        {
            THSNN_Module_save_state_dict(handle, path, patchSize, features, depth, t, deconvStrength, deconvFalloff, deconvHighpass);
            Torch.CheckForErrors();
        }
    }

    /// <summary>Autocast precision for convolutions; normalization and reductions stay in float32.</summary>
    public enum Precision { Float32 = 0, Float16 = 1, BFloat16 = 2 }

    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_R2UNet2D_ctor(long patchSize, long featuresBase, long blocks, long t, long minFeatureSize, out IntPtr pBoxedModule);

        public static R2UNet2D R2UNet2D(long patchSize, long featuresBase, long blocks, long t, long minFeatureSize = 8)
        {
            var res = THSNN_R2UNet2D_ctor(patchSize, featuresBase, blocks, t, minFeatureSize, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new R2UNet2D(res, boxedHandle);
        }
    }
}
