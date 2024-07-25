using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class BoxNetMM : Module
    {
        internal BoxNetMM(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BoxNetMM_pick_forward(Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BoxNetMM_fill_forward(Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BoxNetMM_denoise_forward(Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BoxNetMM_deconv_forward(Module.HType module, IntPtr tensor);

        public TorchTensor PickForward(TorchTensor tensor)
        {
            var res = THSNN_BoxNetMM_pick_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor FillForward(TorchTensor tensor)
        {
            var res = THSNN_BoxNetMM_fill_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor DenoiseForward(TorchTensor tensor)
        {
            var res = THSNN_BoxNetMM_denoise_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor DeconvForward(TorchTensor tensor)
        {
            var res = THSNN_BoxNetMM_deconv_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }

    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BoxNetMM_ctor(long depth_block, long width_block, long input_channels, out IntPtr pBoxedModule);

        static public BoxNetMM BoxNetMM(long depth_block, long width_block, long input_channels)
        {
            var res = THSNN_BoxNetMM_ctor(depth_block, width_block, input_channels, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new BoxNetMM(res, boxedHandle);
        }
    }
}
