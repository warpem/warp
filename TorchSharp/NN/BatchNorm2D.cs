// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class BatchNorm2D : Module
    {
        internal BatchNorm2D(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BatchNorm2d_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_BatchNorm2d_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_BatchNorm2d_ctor(long num_features, out IntPtr pBoxedModule);

        static public BatchNorm2D BatchNorm2D(long numFeatures)
        {
            var res = THSNN_BatchNorm2d_ctor(numFeatures, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new BatchNorm2D(res, boxedHandle);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor BatchNorm2D(TorchTensor x, long numFeatures)
        {
            using (var d = Modules.BatchNorm2D(numFeatures))
            {
                return d.Forward(x);
            }
        }
    }

}
