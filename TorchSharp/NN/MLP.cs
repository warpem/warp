using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class MLP : Module
    {
        internal MLP(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_MLP_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_MLP_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_MLP_ctor(long[] block_widths, int nblocks, bool residual, out IntPtr pBoxedModule);

        static public MLP MLP(long[] blockWidths, bool residual)
        {
            var res = THSNN_MLP_ctor(blockWidths, blockWidths.Length, residual, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new MLP(res, boxedHandle);
        }
    }
}
