using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class C2DNetEncoder : Module
    {
        internal C2DNetEncoder(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_C2DNetEncoder_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_C2DNetEncoder_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_C2DNetEncoder_forward_pose(Module.HType module, IntPtr tensor);

        public TorchTensor ForwardPose(TorchTensor tensor)
        {
            var res = THSNN_C2DNetEncoder_forward_pose(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_C2DNetEncoder_apply_pose(Module.HType module, IntPtr tensor, IntPtr pose);

        public TorchTensor ApplyPose(TorchTensor tensor, TorchTensor pose)
        {
            var res = THSNN_C2DNetEncoder_apply_pose(handle, tensor.Handle, pose.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_C2DNetEncoder_pose_loss(Module.HType module, IntPtr pose, IntPtr refdotprod);

        public TorchTensor PoseLoss(TorchTensor pose, TorchTensor refdotprod)
        {
            var res = THSNN_C2DNetEncoder_pose_loss(handle, pose.Handle, refdotprod.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_C2DNetEncoder_ctor(long boxsize, long codelength, out IntPtr pBoxedModule);

        static public C2DNetEncoder C2DNetEncoder(long boxSize, long codeLength)
        {
            var res = THSNN_C2DNetEncoder_ctor(boxSize, codeLength, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new C2DNetEncoder(res, boxedHandle);
        }
    }

    public class C2DNetDecoder : Module
    {
        internal C2DNetDecoder(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_C2DNetDecoder_forward(Module.HType module, IntPtr tensor, bool usekl);

        public TorchTensor Forward(TorchTensor tensor, bool usekl)
        {
            var res = THSNN_C2DNetDecoder_forward(handle, tensor.Handle, usekl);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_C2DNetDecoder_kld(Module.HType module, IntPtr tensor, double weight);

        public TorchTensor KLD(TorchTensor tensor, double weight)
        {
            var res = THSNN_C2DNetDecoder_kld(handle, tensor.Handle, weight);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_C2DNetDecoder_ctor(long boxsize, long codelength, out IntPtr pBoxedModule);

        static public C2DNetDecoder C2DNetDecoder(long boxSize, long codeLength)
        {
            var res = THSNN_C2DNetDecoder_ctor(boxSize, codeLength, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new C2DNetDecoder(res, boxedHandle);
        }
    }
}
