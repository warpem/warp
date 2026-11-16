using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Warp.Tools
{
    /// <summary>
    /// A whole file read into reusable, pinned memory with a few large sequential reads. Decoding a TIFF or EER movie
    /// from here issues no further file I/O: on a network file system a cold TIFF opened in place pays one round trip
    /// per image directory (every frame's directory sits after its data, or all of them at the end in EER movies),
    /// which costs far more than streaming the file.
    /// </summary>
    public sealed class FileInMemory
    {
        /// <summary>Larger files are left on disk; the buffer is a single array.</summary>
        public const long MaxBytes = 1L << 30;
        const int ChunkBytes = 16 << 20, GrowthBytes = 64 << 20;

        byte[] Buffer = Array.Empty<byte>();

        public string Path { get; private set; } = "";
        public long Length { get; private set; }
        public IntPtr Pointer { get; private set; }

        public static bool Fits(string path) => new FileInfo(path).Length <= MaxBytes;

        /// <summary>The movie formats read into memory: TIFF and EER.</summary>
        public static bool IsMovie(string path)
        {
            string extension = path == null ? "" : System.IO.Path.GetExtension(path).ToLowerInvariant();
            return extension == ".tif" || extension == ".tiff" || extension == ".eer";
        }

        /// <summary>
        /// WARP_TIFF_IN_MEMORY=0 turns reading movies into memory (and prefetching them) off, TIFF and EER alike, back to
        /// decoding from the file. An older in-memory reader once made Slurm jobs die of OOM with memory no process
        /// showed; this one holds at most two movie-sized arrays per worker, in process memory, but the switch stays for
        /// safety.
        /// </summary>
        public static bool Enabled { get; } = Environment.GetEnvironmentVariable("WARP_TIFF_IN_MEMORY") != "0";

        /// <summary>Reads the file; the previous contents are gone even if this throws.</summary>
        public void Read(string path)
        {
            Path = ""; Length = 0;
            using var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            long length = RandomAccess.GetLength(handle);
            if (length > MaxBytes) throw new IOException($"{path} is too large to read into memory.");
            if (length > Buffer.Length)
            {
                // Grow in coarse steps, and collect the outgrown array now: it is 100 MB to 1 GB of pinned memory that
                // would otherwise wait for a full GC that a worker running tasks may not trigger for a long time.
                Buffer = Array.Empty<byte>(); Pointer = IntPtr.Zero;
                GC.Collect();
                long size = Math.Min(MaxBytes, (length + GrowthBytes - 1) / GrowthBytes * GrowthBytes);
                Buffer = GC.AllocateUninitializedArray<byte>((int)size, pinned: true);
                Pointer = Marshal.UnsafeAddrOfPinnedArrayElement(Buffer, 0);
            }
            for (long offset = 0; offset < length;)
            {
                int read = RandomAccess.Read(handle, Buffer.AsSpan((int)offset, (int)Math.Min(ChunkBytes, length - offset)), offset);
                if (read == 0) throw new EndOfStreamException($"{path} ended after {offset} of {length} bytes.");
                offset += read;
            }
            Path = path; Length = length;
        }

        public Stream AsStream() => new MemoryStream(Buffer, 0, (int)Length, writable: false);
    }
}
