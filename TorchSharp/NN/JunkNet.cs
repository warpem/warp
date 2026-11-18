using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// The junk segmentation network (ConvNeXt-Tiny encoder, recurrent whole-image attention and decoder); parameter names
    /// match the PyTorch model. Inference only: the native forward pass runs in inference mode without autograd.
    /// </summary>
    public class JunkNet : Module
    {
        internal JunkNet(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_JunkNet_forward(Module.HType module, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_Module_load_state_dict(Module.HType module, [MarshalAs(UnmanagedType.LPStr)] string location);

        /// <summary>Junk logits, N×1×H×W, for N×1×H×W normalized 6 Å/pixel micrographs.</summary>
        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_JunkNet_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>Loads a PyTorch state dict, bare or under "model".</summary>
        public void LoadStateDict(string path)
        {
            THSNN_Module_load_state_dict(handle, path);
            Torch.CheckForErrors();
        }
    }

    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_JunkNet_ctor(out IntPtr pBoxedModule);

        public static JunkNet JunkNet()
        {
            var res = THSNN_JunkNet_ctor(out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new JunkNet(res, boxedHandle);
        }
    }
}
