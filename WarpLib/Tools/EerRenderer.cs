using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace Warp.Tools
{
    /// <summary>Where each frame's compressed data sits in an EER movie, from one pass over its image directories.</summary>
    public sealed class EerIndex
    {
        public int Compression { get; private init; }
        public long[] Offsets { get; private init; }
        public long[] Sizes { get; private init; }
        public int Frames => Offsets.Length;

        /// <summary>Indexes the movie in memory if given, else the file; expectedFrames sizes the first attempt.</summary>
        public static EerIndex Read(string path, FileInMemory memory, int expectedFrames)
        {
            for (int capacity = Math.Max(1, expectedFrames); ;)
            {
                var offsets = new long[capacity];
                var sizes = new long[capacity];
                int frames = memory != null ? EERNative.Index(null, memory.Pointer, memory.Length, offsets, sizes, capacity, out int compression)
                                            : EERNative.Index(path, IntPtr.Zero, 0, offsets, sizes, capacity, out compression);
                if (frames < 0)
                    throw new NotSupportedException($"{path} is not an EER movie this reader supports: 4096x4096 pixels, 7- or 8-bit compression, each frame's strips back to back.");
                if (frames > capacity)
                {
                    capacity = frames;
                    continue;
                }
                return new EerIndex { Compression = compression, Offsets = offsets[..frames], Sizes = sizes[..frames] };
            }
        }

        public long MaxElectrons(int frame) => EERNative.MaxElectrons(Sizes[frame], Compression);
    }

    /// <summary>
    /// Renders EER frame groups at any output size with the benefit of their 4x sub-pixel positions, on the GPU and from
    /// electron positions alone. The reference is rendering each group at 16K and Fourier cropping that to the output
    /// size, which costs a 1 GB image, its upload and a 16K FFT per group. Here the electrons are spread onto a grid
    /// about twice the output size with a compact kernel, and the grid's transform, with the kernel's own transform
    /// divided out, is cropped to the output (a type 1 non-uniform FFT). Kernel width 4 agrees with the reference to
    /// about 2e-3 of its variation; outputs of 8K or more use the 16K grid itself, where width 1 is the reference exactly.
    /// A renderer takes batches of frames: Upload a batch, Spread its frames group by group, Finish each group, BatchDone.
    /// </summary>
    public sealed class EerRenderer : IDisposable
    {
        public const int PhysicalSize = 4096, SuperSize = 16384;
        /// <summary>4, or WARP_EER_KERNEL_WIDTH (1 renders on the 16K grid itself: the reference, for comparisons).</summary>
        public static readonly int DefaultKernelWidth =
            int.TryParse(Environment.GetEnvironmentVariable("WARP_EER_KERNEL_WIDTH"), out int width) && (width == 1 || width >= 4 && width <= 8) ? width : 4;
        public const int MaxBatchFrames = 64;

        public int OutputSize { get; }
        public int GridSize { get; }
        public int KernelWidth { get; }
        public int BatchFrames { get; }
        public long SlotCapacity { get; }
        /// <summary>The fixed-point scale of the grid's sums.</summary>
        public float Scale { get; }

        IntPtr Handle;

        /// <summary>
        /// A renderer for groups of up to groupFrames frames, each electron weighted by at most maxWeight (the gain's
        /// largest value, or 1). Both bound the grid's sums: an EER frame has at most one electron per 4K pixel, and the
        /// kernel is at most 1, so a node sums at most groupFrames * maxWeight per 4K pixel its kernel reaches. The grid
        /// adds in 32-bit fixed point, for the same result in any order, scaled by the largest power of two that keeps
        /// twice that bound in range.
        /// </summary>
        public EerRenderer(int outputSize, int batchFrames, long slotCapacity, int groupFrames, float maxWeight, int kernelWidth = -1)
        {
            if (outputSize < 2 || outputSize % 2 != 0 || outputSize > SuperSize)
                throw new ArgumentOutOfRangeException(nameof(outputSize), $"EER output size must be even and at most {SuperSize}; {outputSize} was requested.");
            if (kernelWidth < 0)
                kernelWidth = DefaultKernelWidth;
            if (kernelWidth != 1 && (kernelWidth < 4 || kernelWidth > 8))
                throw new ArgumentOutOfRangeException(nameof(kernelWidth), "EER kernel width must be 1 or 4 to 8.");
            if (batchFrames < 1 || batchFrames > MaxBatchFrames)
                throw new ArgumentOutOfRangeException(nameof(batchFrames));

            OutputSize = outputSize;
            int grid = GoodFFTSize(outputSize * 2);
            GridSize = kernelWidth == 1 || grid >= SuperSize ? SuperSize : grid;
            KernelWidth = GridSize == SuperSize ? 1 : kernelWidth;
            BatchFrames = batchFrames;
            SlotCapacity = Math.Max(1, slotCapacity);
            Scale = FixedPointScale(GridSize, KernelWidth, groupFrames, maxWeight);

            Handle = EERNative.CreateRenderer(OutputSize, GridSize, KernelWidth, BatchFrames, SlotCapacity, Scale);
            if (Handle == IntPtr.Zero)
                throw new OutOfMemoryException($"Could not allocate an EER renderer for {OutputSize}² output on a {GridSize}² grid.");
        }

        public static float FixedPointScale(int gridSize, int kernelWidth, int groupFrames, float maxWeight)
        {
            if (!(maxWeight > 0) || float.IsInfinity(maxWeight))
                throw new ArgumentOutOfRangeException(nameof(maxWeight), $"EER weights must be finite; the largest is {maxWeight}.");
            double reach = (double)kernelWidth * PhysicalSize / gridSize + 2; // 4K pixels a kernel spans, either way
            double bound = 2.0 * Math.Max(1, groupFrames) * reach * reach * maxWeight;
            int exponent = (int)Math.Floor(Math.Log2(int.MaxValue / bound));
            if (exponent < 0)
                throw new ArgumentOutOfRangeException(nameof(groupFrames), $"Groups of {groupFrames} frames with weights up to {maxWeight} are beyond the fixed-point range.");
            return MathF.Pow(2, Math.Min(exponent, 30));
        }

        /// <summary>The smallest even size of at least size whose only prime factors are 2, 3, 5 and 7.</summary>
        public static int GoodFFTSize(int size)
        {
            for (int n = size + size % 2; ; n += 2)
            {
                int m = n;
                foreach (int p in new[] { 2, 3, 5, 7 })
                    while (m % p == 0)
                        m /= p;
                if (m == 1)
                    return n;
            }
        }

        /// <summary>
        /// Copies a batch's frames to the GPU from pinned memory, slot s at h_positions + s * SlotCapacity electrons,
        /// asynchronously: h_positions must stay as it is until WaitForUploads returns.
        /// </summary>
        public void Upload(IntPtr h_positions, long[] counts, int slots)
        {
            if (slots < 1 || slots > BatchFrames || counts.Length < slots)
                throw new ArgumentOutOfRangeException(nameof(slots));
            for (int s = 0; s < slots; s++)
                if (counts[s] < 0 || counts[s] > SlotCapacity)
                    throw new ArgumentOutOfRangeException(nameof(counts), $"{counts[s]} electrons exceed the renderer's slot capacity of {SlotCapacity}.");
            EERNative.RendererUpload(Handle, h_positions, counts, slots);
        }

        /// <summary>Adds slots first..first+count of the uploaded batch to the current group, times the gain if given.</summary>
        public void Spread(int first, int count, Image gain) =>
            EERNative.RendererSpread(Handle, first, count, gain?.GetDevice(Intent.Read) ?? IntPtr.Zero, gain?.Dims.X ?? 0);

        /// <summary>After the uploaded batch's last Spread: its device buffer can take the batch after next.</summary>
        public void BatchDone() => EERNative.RendererBatchDone(Handle);

        public void WaitForUploads() => EERNative.RendererWaitForUploads(Handle);

        /// <summary>Writes the current group into d_output (OutputSize² floats on the device) and starts the next.</summary>
        public void Finish(IntPtr d_output) => EERNative.RendererFinish(Handle, d_output);

        public void Dispose()
        {
            if (Handle != IntPtr.Zero)
                EERNative.DestroyRenderer(Handle);
            Handle = IntPtr.Zero;
        }

        static (Image Gain, long Version, float Largest) GainBound;

        /// <summary>The gain's largest magnitude (1 without one), remembered for the image and version it came from.</summary>
        static float LargestGain(Image gain)
        {
            if (gain == null)
                return 1;
            var known = GainBound;
            if (known.Gain == gain && known.Version == gain.Version)
                return known.Largest;
            float largest = 0;
            foreach (float[] slice in gain.GetHost(Intent.Read))
            {
                var (min, max) = Image.MinMax(slice);
                largest = MathF.Max(largest, MathF.Max(MathF.Abs(min), MathF.Abs(max)));
            }
            GainBound = (gain, gain.Version, largest);
            return largest;
        }

        /// <summary>A gain reference for EER multiplies each electron by its pixel's value at 4K, 8K or 16K.</summary>
        public static void CheckGain(Image gain)
        {
            if (gain != null && (gain.Dims.X != gain.Dims.Y || (gain.Dims.X != PhysicalSize && gain.Dims.X != PhysicalSize * 2 && gain.Dims.X != SuperSize)))
                throw new Exception($"An EER gain reference must be {PhysicalSize}, {PhysicalSize * 2} or {SuperSize} pixels square; this one is {gain.Dims.X}x{gain.Dims.Y}.");
        }

        /// <summary>Host buffers reused across movies: pinned electron positions and file reads, two sets for double buffering.</summary>
        sealed class Batch
        {
            public uint[] Positions = Array.Empty<uint>();
            public IntPtr PositionsPointer;
            public byte[] Staging = Array.Empty<byte>();
            public IntPtr StagingPointer;
            public long[] Counts = Array.Empty<long>();

            public void Reserve(int slots, long slotCapacity)
            {
                long needed = slots * slotCapacity;
                if (needed > Positions.LongLength)
                {
                    Positions = Array.Empty<uint>();
                    Positions = GC.AllocateUninitializedArray<uint>(checked((int)needed), pinned: true);
                    PositionsPointer = Marshal.UnsafeAddrOfPinnedArrayElement(Positions, 0);
                }
                if (Counts.Length < slots)
                    Counts = new long[slots];
            }

            public void ReserveStaging(long bytes)
            {
                if (bytes > Staging.LongLength)
                {
                    Staging = Array.Empty<byte>();
                    Staging = GC.AllocateUninitializedArray<byte>(checked((int)Math.Max(bytes, 16L << 20)), pinned: true);
                    StagingPointer = Marshal.UnsafeAddrOfPinnedArrayElement(Staging, 0);
                }
            }
        }

        static readonly Batch[] Batches = { new(), new() };
        static readonly object BatchesLock = new();

        /// <summary>
        /// Renders an EER movie's frame groups (groupFrames frames each, the first groups*groupFrames frames) into
        /// deviceStack (groups slices of outputSize² floats on the device) or hostStack, multiplying by gain if given.
        /// Batches of frames are decoded on threads CPU threads from the movie in memory, or else read from the file,
        /// while the GPU uploads and spreads the batch before. The calling thread's device renders. Returns the
        /// electron count.
        /// </summary>
        public static long RenderStack(string path, FileInMemory memory, EerIndex index, int groupFrames, int groups, Image gain,
                                       int outputSize, int threads, IntPtr deviceStack, float[][] hostStack, int kernelWidth = -1,
                                       Action<int, IntPtr> onGroup = null)
        {
            CheckGain(gain);
            int frames = groupFrames * groups;
            if (groupFrames < 1 || groups < 1 || frames > index.Frames)
                throw new ArgumentOutOfRangeException(nameof(groups), $"{groups} groups of {groupFrames} frames do not fit in {index.Frames} frames.");
            // onGroup takes each group on the GPU as it is rendered, instead of a stack.
            if ((deviceStack != IntPtr.Zero ? 1 : 0) + (hostStack != null ? 1 : 0) + (onGroup != null ? 1 : 0) != 1)
                throw new ArgumentException("Render into either a device or a host stack, or hand each group over.");

            threads = Math.Max(1, threads);
            int batchFrames = Math.Min(Math.Min(32, MaxBatchFrames), frames);
            long slotCapacity = Enumerable.Range(0, frames).Max(index.MaxElectrons);
            long sliceElements = (long)outputSize * outputSize;

            lock (BatchesLock)
            {
                foreach (var batch in Batches)
                    batch.Reserve(batchFrames, slotCapacity);

                using SafeFileHandle file = memory == null ? File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete) : null;
                using var renderer = new EerRenderer(outputSize, batchFrames, slotCapacity, groupFrames, LargestGain(gain), kernelWidth);
                IntPtr scratch = deviceStack == IntPtr.Zero ? GPU.MallocDevice(sliceElements) : IntPtr.Zero;

                Task Decode(int first, Batch batch) => Task.Run(() => DecodeBatch(path, memory, file, index, first, Math.Min(batchFrames, frames - first), slotCapacity, threads, batch));

                long electrons = 0;
                Task decoding = Decode(0, Batches[0]);
                try
                {
                    for (int first = 0, b = 0; first < frames; first += batchFrames, b++)
                    {
                        decoding.GetAwaiter().GetResult();
                        Batch current = Batches[b % 2];
                        int count = Math.Min(batchFrames, frames - first);

                        // The other set was uploaded with the previous batch; once that is done, it takes the next.
                        renderer.WaitForUploads();
                        if (first + batchFrames < frames)
                            decoding = Decode(first + batchFrames, Batches[(b + 1) % 2]);

                        renderer.Upload(current.PositionsPointer, current.Counts, count);
                        for (int j = 0; j < count; j++)
                            electrons += current.Counts[j];

                        // Frames of one group at a time; each group that ends here is finished.
                        for (int j = 0; j < count;)
                        {
                            int frame = first + j;
                            int end = Math.Min(count, j + groupFrames - frame % groupFrames);
                            renderer.Spread(j, end - j, gain);
                            j = end;
                            if ((first + end) % groupFrames != 0)
                                continue;
                            int z = (first + end) / groupFrames - 1;
                            renderer.Finish(scratch != IntPtr.Zero ? scratch : (IntPtr)((long)deviceStack + z * sliceElements * sizeof(float)));
                            if (hostStack != null)
                                GPU.CopyDeviceToHost(scratch, hostStack[z], sliceElements);
                            else
                                onGroup?.Invoke(z, scratch);
                        }
                        renderer.BatchDone();
                    }
                }
                finally
                {
                    try { decoding.Wait(); } catch (Exception) { } // a failure is reported by the loop; the buffers must be idle
                    renderer.WaitForUploads();
                    if (scratch != IntPtr.Zero)
                        GPU.FreeDevice(scratch);
                }
                return electrons;
            }
        }

        static IntPtr Slot(Batch batch, int slot, long slotCapacity) => (IntPtr)((long)batch.PositionsPointer + slot * slotCapacity * sizeof(uint));

        static void DecodeBatch(string path, FileInMemory memory, SafeFileHandle file, EerIndex index, int first, int count, long slotCapacity, int threads, Batch batch)
        {
            // The batch's frames, from 0: offsets into data, and sizes.
            IntPtr data;
            long available;
            long[] offsets = new long[count], sizes = index.Sizes[first..(first + count)];
            if (memory != null)
            {
                data = memory.Pointer;
                available = memory.Length;
                Array.Copy(index.Offsets, first, offsets, 0, count);
            }
            else
            {
                // Frames are usually stored back to back: one read for the batch, else one per frame.
                long start = index.Offsets[first], end = index.Offsets[first + count - 1] + index.Sizes[first + count - 1];
                bool contiguous = true;
                for (int f = first + 1; f < first + count; f++)
                    contiguous &= index.Offsets[f] == index.Offsets[f - 1] + index.Sizes[f - 1];
                if (contiguous)
                {
                    batch.ReserveStaging(end - start);
                    ReadExactly(file, batch.Staging.AsSpan(0, (int)(end - start)), start, path);
                    for (int j = 0; j < count; j++)
                        offsets[j] = index.Offsets[first + j] - start;
                    available = end - start;
                }
                else
                {
                    long total = 0;
                    for (int j = 0; j < count; j++)
                        total += index.Sizes[first + j];
                    batch.ReserveStaging(total);
                    long position = 0;
                    for (int j = 0; j < count; j++)
                    {
                        ReadExactly(file, batch.Staging.AsSpan((int)position, (int)index.Sizes[first + j]), index.Offsets[first + j], path);
                        offsets[j] = position;
                        position += index.Sizes[first + j];
                    }
                    available = position;
                }
                data = batch.StagingPointer;
            }

            try
            {
                Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = threads }, j =>
                {
                    long n = EERNative.DecodeFrames(data, available, offsets, sizes, j, 1, index.Compression, Slot(batch, j, slotCapacity), slotCapacity);
                    if (n < 0)
                        throw new InvalidDataException($"Frame {first + j} of {path} could not be decoded: {(n == -2 ? "it holds more electrons than its size allows" : "its data is malformed")}.");
                    batch.Counts[j] = n;
                });
            }
            catch (AggregateException e) when (e.InnerExceptions.Count > 0)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
            }
        }

        static void ReadExactly(SafeFileHandle file, Span<byte> buffer, long offset, string path)
        {
            for (int done = 0; done < buffer.Length;)
            {
                int read = RandomAccess.Read(file, buffer.Slice(done), offset + done);
                if (read == 0)
                    throw new EndOfStreamException($"{path} ended at byte {offset + done}, inside a frame.");
                done += read;
            }
        }
    }
}
