using System;
using System.Runtime.InteropServices;
using System.Security;
using System.Threading;
using Warp.Tools;

namespace Warp
{
    [SuppressUnmanagedCodeSecurity]
    public static class TiffNative
    {
        [DllImport("NativeAcceleration", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "ReadTIFF")]
        public static extern void ReadTIFF(string path, int layer, bool flipy, float[] h_result);

        /// <summary>
        /// One layer of a TIFF held in memory, decoded but not converted or flipped: rows in file order, samples in the
        /// file's type. Returns <see cref="RawType"/> as int, or -1 when the layout is unsupported or the data is bad,
        /// in which case <see cref="ReadTIFF"/> should be used instead.
        /// </summary>
        [DllImport("NativeAcceleration", EntryPoint = "ReadTIFFRawFromMemory")]
        public static extern int ReadTIFFRawFromMemory(IntPtr data, long size, int layer, int width, int height, IntPtr h_result);

        public enum RawType { UChar = 0, SChar = 1, UShort = 2, Short = 3, Float = 4 }

        public static int RawBytesPerSample(int type) => type switch { 0 or 1 => 1, 2 or 3 => 2, 4 => 4, _ => throw new ArgumentOutOfRangeException(nameof(type)) };

        public static void ReadTIFFPatient(int attempts, int mswait, string path, int layer, bool flipy, float[] h_result)
        {
            Exception LastException = null;

            for (int a = 0; a < attempts; a++)
            {
                try
                {
                    ReadTIFF(path, layer, flipy, h_result);
                    return;
                }
                catch (Exception exc)
                {
                    LastException = exc;
                    Thread.Sleep(mswait);
                }
            }

            throw new Exception($"Could not successfully read {path} within the specified number of attempts:\n" + LastException);
        }
    }
}
