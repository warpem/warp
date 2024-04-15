using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class ResNet : Module
    {
        internal ResNet(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ResNet_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_ResNet_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ResNet_ctor(long size_input, long blocks1, long blocks2, long block3, long blocks4, long num_classes, out IntPtr pBoxedModule);

        static public ResNet ResNet(long size_input, long blocks1, long blocks2, long blocks3, long blocks4, long num_classes)
        {
            var res = THSNN_ResNet_ctor(size_input, blocks1, blocks2, blocks3, blocks4, num_classes, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ResNet(res, boxedHandle);
        }
    }
}
