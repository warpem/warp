using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class ParticleWGANGenerator : Module
    {
        internal ParticleWGANGenerator(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ParticleWGANGenerator_forward_noise(Module.HType module, IntPtr crapcode, IntPtr fakeimages, IntPtr ctf);

        public TorchTensor ForwardNoise(TorchTensor crapcode, TorchTensor fakeimages, TorchTensor ctf)
        {
            var res = THSNN_ParticleWGANGenerator_forward_noise(handle, crapcode.Handle, fakeimages.Handle, ctf.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ParticleWGANGenerator_forward_particle(Module.HType module, IntPtr code, bool transform, double sigmashift);

        public TorchTensor ForwardParticle(TorchTensor code, bool transform, double sigmashift)
        {
            var res = THSNN_ParticleWGANGenerator_forward_particle(handle, code.Handle, transform, sigmashift);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ParticleWGANGenerator_ctor(long boxsize, long codelength, out IntPtr pBoxedModule);

        static public ParticleWGANGenerator ParticleWGANGenerator(long boxsize, long codelength)
        {
            var res = THSNN_ParticleWGANGenerator_ctor(boxsize, codelength, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ParticleWGANGenerator(res, boxedHandle);
        }
    }

    public class ParticleWGANDiscriminator : Module
    {
        internal ParticleWGANDiscriminator(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ParticleWGANDiscriminator_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_ParticleWGANDiscriminator_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_ParticleWGANDiscriminator_clipweights(Module.HType module, double clip);

        public void ClipWeights(double clip)
        {
            THSNN_ParticleWGANDiscriminator_clipweights(handle, clip);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ParticleWGANDiscriminator_penalizegradient(Module.HType module, IntPtr real, IntPtr fake, float lambda);

        public TorchTensor PenalizeGradient(TorchTensor real, TorchTensor fake, float lambda)
        {
            var res = THSNN_ParticleWGANDiscriminator_penalizegradient(handle, real.Handle, fake.Handle, lambda);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ParticleWGANDiscriminator_ctor(out IntPtr pBoxedModule);

        static public ParticleWGANDiscriminator ParticleWGANDiscriminator()
        {
            var res = THSNN_ParticleWGANDiscriminator_ctor(out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ParticleWGANDiscriminator(res, boxedHandle);
        }
    }
}
