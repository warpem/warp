using System;
using Warp;
using Warp.Tools;

namespace WarpWorker2
{
    /// <summary>
    /// Cheap GPU sanity check that exercises the CUDA/cuFFT path real work uses.
    /// Pass => hardware healthy. Throw/false => hardware fault (spec §9.3).
    /// Used at startup and as the arbiter after any task exception.
    /// </summary>
    static class GpuHealthProbe
    {
        public static bool Probe(int deviceId)
        {
            // Clear error from any previous kernels
            try
            {
                GPU.CheckGPUExceptions();
            }
            catch { }
            
            try
            {
                GPU.SetDevice(deviceId);
                // Small round trip: allocate, FFT, copy back. Any CUDA fault throws.
                Image test = new Image(new int3(64, 64, 1));
                test.Fill(1f);
                Image ft = test.AsFFT();
                ft.Dispose();
                test.Dispose();
                GPU.CheckGPUExceptions();
                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
