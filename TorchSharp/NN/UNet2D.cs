using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class UNet2D : Module
    {
        internal UNet2D(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_UNet2D_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_UNet2D_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_UNet2D_ctor(long depth_block, long width_block, long input_channels, long final_channels, long final_kernel, bool dochannelattn, bool dospatialattn, out IntPtr pBoxedModule);

        static public UNet2D UNet2D(long depth_block, long width_block, long input_channels, long final_channels, long final_kernel, bool dochannelattn, bool dospatialattn)
        {
            var res = THSNN_UNet2D_ctor(depth_block, width_block, input_channels, final_channels, final_kernel, dochannelattn, dospatialattn, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new UNet2D(res, boxedHandle);
        }
    }
}
