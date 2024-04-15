using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class DistanceNet : Module
    {
        internal DistanceNet(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_DistanceNet_forward(Module.HType module, IntPtr reference, IntPtr data, IntPtr d_reference, IntPtr d_data);

        public TorchTensor Forward(TorchTensor reference, TorchTensor data, IntPtr d_reference, IntPtr d_data)
        {
            var res = THSNN_DistanceNet_forward(handle, reference.Handle, data.Handle, d_reference, d_data);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_DistanceNet_ctor(out IntPtr pBoxedModule);

        static public DistanceNet DistanceNet()
        {
            var res = THSNN_DistanceNet_ctor(out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new DistanceNet(res, boxedHandle);
        }
    }
}
