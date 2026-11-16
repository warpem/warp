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
            Exception exc = null;

            for (int a = 0; a < attempts; a++)
            {
                try
                {
                    ReadEER(path, firstFrameInclusive, lastFrameExclusive, eer_upsampling, h_result);
                    return;
                }
                catch (Exception e)
                {
                    exc = e;
                    Thread.Sleep(mswait);
                }
            }

            throw new Exception($"Could not successfully read {path} within the specified number of attempts: {exc.Message}");
        }

        /// <summary>
        /// Per frame, the offset and size of its compressed data, from one pass over the image directories of the file in
        /// memory (data, size) or, with data zero, opened from path. Returns the frame count (filling at most maxFrames
        /// entries), or -1 if this is not a supported EER movie: 4096x4096, compression 65000 or 65001, each frame's strips back to back.
        /// </summary>
        [DllImport("NativeAcceleration", CharSet = CharSet.Ansi, EntryPoint = "IndexEER")]
        public static extern int Index(string path, IntPtr data, long size, long[] offsets, long[] sizes, int maxFrames, out int compression);

        /// <summary>
        /// Decodes frames into electron positions on the 16K grid, (y &lt;&lt; 14) | x, frame after frame. Offsets are relative
        /// to data, readable for available bytes. Returns the electron count, -1 for a malformed frame, -2 if capacity is short.
        /// </summary>
        [DllImport("NativeAcceleration", EntryPoint = "DecodeEERFrames")]
        public static extern unsafe long DecodeFrames(IntPtr data, long available, long* offsets, long* sizes, int frames, int compression, IntPtr positions, long capacity);

        /// <summary>DecodeFrames for frames first..first+count of the offset and size arrays.</summary>
        public static unsafe long DecodeFrames(IntPtr data, long available, long[] offsets, long[] sizes, int first, int count, int compression, IntPtr positions, long capacity)
        {
            if (first < 0 || count < 1 || first + count > offsets.Length || first + count > sizes.Length)
                throw new ArgumentOutOfRangeException(nameof(count));
            fixed (long* o = &offsets[first])
            fixed (long* s = &sizes[first])
                return DecodeFrames(data, available, o, s, count, compression, positions, capacity);
        }

        /// <summary>The most electrons a frame of this compressed size can hold.</summary>
        [DllImport("NativeAcceleration", EntryPoint = "EERMaxElectrons")]
        public static extern long MaxElectrons(long size, int compression);

        /// <summary>
        /// A GPU renderer for outputSize² groups on a gridSize² grid (16K for kernel width 1), taking batches of up to
        /// slots frames of up to slotCapacity electrons and adding weights times scale in 32-bit fixed point; zero if the
        /// arguments are out of range or memory is short.
        /// </summary>
        [DllImport("NativeAcceleration", EntryPoint = "EERCreateRenderer")]
        public static extern IntPtr CreateRenderer(int outputSize, int gridSize, int kernelWidth, int slots, long slotCapacity, float scale);

        /// <summary>Copies a batch from pinned memory (slot s at positions + s * slotCapacity) on the renderer's copy stream.</summary>
        [DllImport("NativeAcceleration", EntryPoint = "EERRendererUpload")]
        public static extern void RendererUpload(IntPtr renderer, IntPtr h_positions, long[] h_counts, int slots);

        /// <summary>Adds slots of the uploaded batch to the current group, each electron times its gain pixel.</summary>
        [DllImport("NativeAcceleration", EntryPoint = "EERRendererSpread")]
        public static extern void RendererSpread(IntPtr renderer, int firstSlot, int slots, IntPtr d_gain, int gainSize);

        [DllImport("NativeAcceleration", EntryPoint = "EERRendererBatchDone")]
        public static extern void RendererBatchDone(IntPtr renderer);

        /// <summary>Returns once every upload so far has left host memory.</summary>
        [DllImport("NativeAcceleration", EntryPoint = "EERRendererWaitForUploads")]
        public static extern void RendererWaitForUploads(IntPtr renderer);

        /// <summary>Writes the current group into d_output (outputSize² floats on the device) and starts the next.</summary>
        [DllImport("NativeAcceleration", EntryPoint = "EERRendererFinish")]
        public static extern void RendererFinish(IntPtr renderer, IntPtr d_output);

        [DllImport("NativeAcceleration", EntryPoint = "EERDestroyRenderer")]
        public static extern void DestroyRenderer(IntPtr renderer);
    }
}
