using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Discriminator3D : Module
    {
        internal Discriminator3D(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Discriminator3D_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_Discriminator3D_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Discriminator3D_penalizegradient(Module.HType module, IntPtr real, IntPtr fake, float lambda);

        public TorchTensor PenalizeGradient(TorchTensor real, TorchTensor fake, float lambda)
        {
            var res = THSNN_Discriminator3D_penalizegradient(handle, real.Handle, fake.Handle, lambda);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Discriminator3D_ctor(out IntPtr pBoxedModule);

        static public Discriminator3D Discriminator3D()
        {
            var res = THSNN_Discriminator3D_ctor(out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Discriminator3D(res, boxedHandle);
        }
    }
}
