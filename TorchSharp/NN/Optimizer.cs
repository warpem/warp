// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public partial class Optimizer : IDisposable
    {
        /// <summary>
        ///    Class wrapping PyTorch's optimzer object reference.
        /// </summary>
        internal sealed class HType : SafeHandle
        {
            public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
            {
                SetHandle (preexistingHandle);
            }

            public override bool IsInvalid => handle == IntPtr.Zero;

            // This is just for marshalling
            internal HType () : base (IntPtr.Zero, true)
            {
            }

            [DllImport ("LibTorchSharp")]
            private static extern void THSNN_Optimizer_dispose (HType handle);

            protected override bool ReleaseHandle ()
            {
                THSNN_Optimizer_dispose (this);
                return true;
            }

            protected override void Dispose (bool disposing)
            {
                if (disposing) {
                    ReleaseHandle ();
                }
            }
        }

        internal HType handle;

        protected Optimizer (IntPtr handle)
        {
            this.handle = new HType (handle, true);
        }

        ~Optimizer ()
        {
            Dispose (false);
        }

        /// <summary>
        ///   Releases the storage.
        /// </summary>
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose (bool disposing)
        {
            if (disposing) {
                handle.Dispose ();
                handle.SetHandleAsInvalid ();
            }
        }
    }

    public partial class Optimizer
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Adam_ctor (IntPtr parameters, int len, double learningRate, double weightdecay, bool amsgrad, double beta1, double beta2);

        public static Optimizer Adam (IEnumerable<TorchTensor> parameters, double learningRate, double weightdecay, bool amsgrad = false, double beta1 = 0.9, double beta2 = 0.999)
        {
            var parray = new PinnedArray<IntPtr> ();
            IntPtr paramsRef = parray.CreateArray (parameters.Select (p => p.Handle).ToArray ());

            var res = THSNN_Adam_ctor (paramsRef, parray.Array.Length, learningRate, weightdecay, amsgrad, beta1, beta2);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Optimizer (res);
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_SGD_ctor (IntPtr parameters, int len, double learningRate, double momentum, bool nesterov, double weightdecay);

        public static Optimizer SGD (IEnumerable<TorchTensor> parameters, double learningRate, double momentum = 0, bool nesterov = false, double weightdecay = 0)
        {
            var parray = new PinnedArray<IntPtr> ();
            IntPtr paramsRef = parray.CreateArray (parameters.Select (p => p.Handle).ToArray ());

            var res = THSNN_SGD_ctor (paramsRef, parray.Array.Length, learningRate, momentum, nesterov, weightdecay);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Optimizer (res);
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_RMSprop_ctor(IntPtr parameters, int len, double learningRate, double weightdecay);

        public static Optimizer RMSprop(IEnumerable<TorchTensor> parameters, double learningRate, double weightdecay)
        {
            var parray = new PinnedArray<IntPtr>();
            IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

            var res = THSNN_RMSprop_ctor(paramsRef, parray.Array.Length, learningRate, weightdecay);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Optimizer(res);
        }

        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Optimizer_zeroGrad (HType module);

        public void ZeroGrad ()
        {
            THSNN_Optimizer_zeroGrad (handle);
            Torch.CheckForErrors ();
        }

        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Optimizer_step (HType module);

        public void Step ()
        {
            THSNN_Optimizer_step (handle);
            Torch.CheckForErrors ();
        }

        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Optimizer_getParameters (HType module, AllocatePinnedArray allocator);

        public IEnumerable<TorchTensor> GetParameters ()
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr> ()) {
                THSNN_Optimizer_getParameters (handle, pa.CreateArray);
                Torch.CheckForErrors ();
                ptrArray = pa.Array;
            }
            return ptrArray.Select (x => new TorchTensor (x));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_Adam_setlearningrate(HType module, double learningrate);

        public void SetLearningRateAdam(double learningrate)
        {
            THSNN_Adam_setlearningrate(handle, learningrate);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_SGD_setlearningrate(HType module, double learningrate);

        public void SetLearningRateSGD(double learningrate)
        {
            THSNN_SGD_setlearningrate(handle, learningrate);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_RMSprop_setlearningrate(HType module, double learningrate);

        public void SetLearningRateRMSprop(double learningrate)
        {
            THSNN_RMSprop_setlearningrate(handle, learningrate);
        }
    }
}
