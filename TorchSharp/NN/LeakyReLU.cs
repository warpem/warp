// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a LeakyReLU module.
    /// </summary>
    public class LeakyReLU : Module
    {
        internal LeakyReLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_LeakyReLU_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_LeakyReLU_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public override string GetName()
        {
            return typeof(LeakyReLU).Name;
        }
    }

    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_LeakyReLU_ctor(double negative_slope, bool inplace, out IntPtr pBoxedModule);

        static public LeakyReLU LeakyReLU(double negative_slope = 0.1, bool inPlace = false)
        {
            var handle = THSNN_LeakyReLU_ctor(negative_slope, inPlace, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new LeakyReLU(handle, boxedHandle);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor LeakyReLU(TorchTensor x, double negative_slope = 0.1, bool inPlace = false)
        {
            using (var m = Modules.LeakyReLU(negative_slope, inPlace))
            {
                return m.Forward(x);
            }
        }
    }

}
