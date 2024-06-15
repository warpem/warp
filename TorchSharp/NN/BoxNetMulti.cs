using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class BoxNetMulti : Module
    {
        internal BoxNetMulti(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BoxNetMulti_pick_forward(Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BoxNetMulti_denoise_forward(Module.HType module, IntPtr tensor);

        public TorchTensor PickForward(TorchTensor tensor)
        {
            var res = THSNN_BoxNetMulti_pick_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor DenoiseForward(TorchTensor tensor)
        {
            var res = THSNN_BoxNetMulti_denoise_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }

    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BoxNetMulti_ctor(long depth_block, long width_block, long input_channels, out IntPtr pBoxedModule);

        static public BoxNetMulti BoxNetMulti(long depth_block, long width_block, long input_channels)
        {
            var res = THSNN_BoxNetMulti_ctor(depth_block, width_block, input_channels, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new BoxNetMulti(res, boxedHandle);
        }
    }
}
