using System;
using System.Runtime.InteropServices;
using System.Security;
using System.Threading;
using Warp.Tools;

namespace Warp
{
    [SuppressUnmanagedCodeSecurity]
    public static class EERNative
    {
        [DllImport("NativeAcceleration", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "ReadEERCombinedFrame")]
        public static extern void ReadEER(string path, int firstFrameInclusive, int lastFrameExclusive, int eer_upsampling, float[] h_result);

        public static void ReadEERPatient(int attempts, int mswait, string path, int firstFrameInclusive, int lastFrameExclusive, int eer_upsampling, float[] h_result)
        {
            for (int a = 0; a < attempts; a++)
            {
                try
                {
                    ReadEER(path, firstFrameInclusive, lastFrameExclusive, eer_upsampling, h_result);
                    return;
                }
                catch
                {
                    Thread.Sleep(mswait);
                }
            }

            throw new Exception($"Could not successfully read {path} within the specified number of attempts.");
        }
    }
}
