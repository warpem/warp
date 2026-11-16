using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Warp;
using Warp.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Tests;

/// <summary>
/// EER rendering from electron positions: decoding matches the legacy 16K renderer, and the GPU spread-and-crop matches
/// rendering at 16K with the gain applied per pixel and Fourier cropping that, as LoadStack used to.
/// </summary>
public class EerRenderCudaTests : IDisposable
{
    readonly ITestOutputHelper output;
    readonly string root = Path.Combine(Path.GetTempPath(), "warp-eer-" + Guid.NewGuid().ToString("N"));
    public EerRenderCudaTests(ITestOutputHelper output) { this.output = output; Directory.CreateDirectory(root); }
    public void Dispose() => Directory.Delete(root, true);

    const int Pixels = 4096 * 4096, Super = 16384;

    /// <summary>An electron: its pixel on the 4K grid and its sub-pixel (0..3 each way).</summary>
    public readonly record struct Electron(int Pixel, int SubX, int SubY)
    {
        public int X16 => (Pixel % 4096) * 4 + SubX;
        public int Y16 => (Pixel / 4096) * 4 + SubY;
        public uint Position => (uint)((Y16 << 14) | X16);
    }

    public static Electron[][] RandomFrames(int frames, int perFrame, int seed)
    {
        var random = new Random(seed);
        return Helper.ArrayOfFunction(f =>
        {
            var pixels = new HashSet<int>();
            while (pixels.Count < perFrame)
                pixels.Add(random.Next(Pixels));
            // The last pixel exercises the end of the stream.
            if (f == 0) { pixels.Remove(pixels.First()); pixels.Add(Pixels - 1); }
            return pixels.OrderBy(p => p).Select(p => new Electron(p, random.Next(4), random.Next(4))).ToArray();
        }, frames);
    }

    sealed class BitWriter
    {
        readonly List<byte> bytes = new();
        int bit;
        public void Write(int value, int bits)
        {
            for (int i = 0; i < bits; i++, bit++)
            {
                if (bit % 8 == 0) bytes.Add(0);
                if (((value >> i) & 1) != 0) bytes[^1] |= (byte)(1 << (bit % 8));
            }
        }
        public byte[] ToArray(int multiple)
        {
            while (bytes.Count % multiple != 0) bytes.Add(0);
            return bytes.ToArray();
        }
    }

    /// <summary>One frame's EER stream: run lengths with escapes, then a 4-bit symbol (stored as 0000YyXx ^ 0x0A).</summary>
    public static byte[] EncodeFrame(Electron[] electrons, bool sevenBit)
    {
        int runBits = sevenBit ? 7 : 8, escape = (1 << runBits) - 1;
        var writer = new BitWriter();
        void Run(int length, bool last)
        {
            while (length >= escape)
            {
                writer.Write(escape, runBits);
                if (!sevenBit) writer.Write(0, 4); // 8-bit codes always carry a symbol slot
                length -= escape;
                if (last && length == 0) return;
            }
            writer.Write(length, runBits);
        }
        int cursor = 0;
        foreach (var e in electrons)
        {
            Run(e.Pixel - cursor, false);
            writer.Write(((e.SubY << 2) | e.SubX) ^ 0x0A, 4);
            cursor = e.Pixel + 1;
        }
        Run(Pixels - cursor, true);
        return writer.ToArray(sevenBit ? 1 : 3);
    }

    /// <summary>
    /// An EER file: frames back to back, then one directory per frame, as BigTIFF or classic TIFF; with twoStrips,
    /// each frame's stream is split into two strips (BigTIFF only), whose offset and size arrays precede the directory.
    /// </summary>
    public static void WriteEer(string path, Electron[][] frames, bool sevenBit, bool bigTiff, bool twoStrips = false)
    {
        var data = frames.Select(f => EncodeFrame(f, sevenBit)).ToArray();
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write((byte)'I'); writer.Write((byte)'I');
        if (bigTiff) { writer.Write((ushort)43); writer.Write((ushort)8); writer.Write((ushort)0); writer.Write(0UL); }
        else { writer.Write((ushort)42); writer.Write(0U); }
        long headerEnd = writer.BaseStream.Position;

        var offsets = new long[data.Length];
        for (int f = 0; f < data.Length; f++) { offsets[f] = writer.BaseStream.Position; writer.Write(data[f]); }
        if (writer.BaseStream.Position % 2 == 1) writer.Write((byte)0);

        ushort compression = (ushort)(sevenBit ? 65001 : 65000);
        var ifdOffsets = new long[data.Length];
        for (int f = 0; f < data.Length; f++)
        {
            long stripOffsets = 0, stripSizes = 0;
            int half = data[f].Length / 2;
            if (twoStrips)
            {
                stripOffsets = writer.BaseStream.Position;
                writer.Write((ulong)offsets[f]); writer.Write((ulong)(offsets[f] + half));
                stripSizes = writer.BaseStream.Position;
                writer.Write((ulong)half); writer.Write((ulong)(data[f].Length - half));
            }
            ifdOffsets[f] = writer.BaseStream.Position;
            var tags = new (ushort Tag, ushort Type, long Count, long Value)[]
            {
                (256, 4, 1, 4096), (257, 4, 1, 4096), (259, 3, 1, compression), (262, 3, 1, 1),
                (273, (ushort)(bigTiff ? 16 : 4), twoStrips ? 2 : 1, twoStrips ? stripOffsets : offsets[f]), (277, 3, 1, 1),
                (278, 4, 1, twoStrips ? 2048 : 4096), (279, (ushort)(bigTiff ? 16 : 4), twoStrips ? 2 : 1, twoStrips ? stripSizes : data[f].Length),
            };
            if (bigTiff) writer.Write((ulong)tags.Length); else writer.Write((ushort)tags.Length);
            foreach (var (tag, type, count, value) in tags)
            {
                writer.Write(tag); writer.Write(type);
                if (bigTiff) { writer.Write((ulong)count); writer.Write((ulong)value); }
                else { writer.Write((uint)count); if (type == 3) { writer.Write((ushort)value); writer.Write((ushort)0); } else writer.Write((uint)value); }
            }
            long next = writer.BaseStream.Position;
            if (bigTiff) writer.Write(0UL); else writer.Write(0U);
            if (f > 0)
            {
                long here = writer.BaseStream.Position;
                writer.Seek((int)(ifdOffsets[f - 1] + (bigTiff ? 8 + tags.Length * 20 : 2 + tags.Length * 12)), SeekOrigin.Begin);
                if (bigTiff) writer.Write((ulong)ifdOffsets[f]); else writer.Write((uint)ifdOffsets[f]);
                writer.Seek((int)here, SeekOrigin.Begin);
            }
        }
        writer.Seek(bigTiff ? 8 : 4, SeekOrigin.Begin);
        if (bigTiff) writer.Write((ulong)ifdOffsets[0]); else writer.Write((uint)ifdOffsets[0]);
    }

    static uint[] Decode(FileInMemory memory, EerIndex index, int first, int count)
    {
        long capacity = Enumerable.Range(first, count).Sum(index.MaxElectrons);
        var positions = new uint[capacity];
        var handle = GCHandle.Alloc(positions, GCHandleType.Pinned);
        try
        {
            long n = EERNative.DecodeFrames(memory.Pointer, memory.Length, index.Offsets, index.Sizes, first, count, index.Compression, handle.AddrOfPinnedObject(), capacity);
            Assert.True(n >= 0, $"decoding failed with {n}");
            return positions[..(int)n];
        }
        finally { handle.Free(); }
    }

    [CtfCudaFact]
    public void DecodesPositionsAsTheLegacyRendererCountsThem()
    {
        var frames = RandomFrames(3, 20000, 7);
        var legacy = new float[(long)Super * Super];
        foreach (var (sevenBit, bigTiff, twoStrips) in new[] { (true, true, false), (true, false, false), (false, true, false), (false, false, false), (true, true, true), (false, true, true) })
        {
            string path = Path.Combine(root, $"movie-{(sevenBit ? 7 : 8)}-{(bigTiff ? "big" : "classic")}{(twoStrips ? "-strips" : "")}.eer");
            WriteEer(path, frames, sevenBit, bigTiff, twoStrips);

            var memory = new FileInMemory();
            memory.Read(path);
            var index = EerIndex.Read(path, memory, frames.Length);
            var fromFile = EerIndex.Read(path, null, 1); // a short first guess grows
            Assert.Equal(frames.Length, index.Frames);
            Assert.Equal(sevenBit ? 65001 : 65000, index.Compression);
            Assert.Equal(index.Offsets, fromFile.Offsets);
            Assert.Equal(index.Sizes, fromFile.Sizes);

            for (int f = 0; f < frames.Length; f++)
                Assert.Equal(frames[f].Select(e => e.Position), Decode(memory, index, f, 1));
            Assert.Equal(frames.SelectMany(f => f).Select(e => e.Position), Decode(memory, index, 0, frames.Length));

            // The legacy renderer reads the same file to the same electrons (two frames at 16K).
            EERNative.ReadEER(path, 0, 2, 3, legacy);
            var expected = new Dictionary<long, int>();
            foreach (var e in frames[0].Concat(frames[1]))
                expected[(long)e.Y16 * Super + e.X16] = expected.GetValueOrDefault((long)e.Y16 * Super + e.X16) + 1;
            Assert.Equal(expected.Values.Sum(), (int)legacy.Sum(v => (double)v));
            foreach (var (i, c) in expected)
                Assert.Equal(c, legacy[i]);
        }
    }

    [CtfCudaFact]
    public void MalformedFramesAreRejected()
    {
        var frames = RandomFrames(1, 1000, 3);
        string path = Path.Combine(root, "short.eer");
        WriteEer(path, frames, true, true);
        var memory = new FileInMemory();
        memory.Read(path);
        var index = EerIndex.Read(path, memory, 1);
        var truncated = new long[] { index.Sizes[0] / 2 };
        var positions = new uint[index.MaxElectrons(0)];
        var handle = GCHandle.Alloc(positions, GCHandleType.Pinned);
        try
        {
            Assert.Equal(-1, EERNative.DecodeFrames(memory.Pointer, memory.Length, index.Offsets, truncated, 0, 1, index.Compression, handle.AddrOfPinnedObject(), positions.Length));
            Assert.Equal(-2, EERNative.DecodeFrames(memory.Pointer, memory.Length, index.Offsets, index.Sizes, 0, 1, index.Compression, handle.AddrOfPinnedObject(), 10));
        }
        finally { handle.Free(); }

        File.WriteAllBytes(Path.Combine(root, "plain.tif"), File.ReadAllBytes(path)[..16]);
        Assert.Throws<NotSupportedException>(() => EerIndex.Read(Path.Combine(root, "plain.tif"), null, 1));
    }

    /// <summary>Rendering at 16K with the gain per pixel and Fourier cropping each group, the legacy way.</summary>
    static float[][] Reference(Electron[][] frames, int groupFrames, Image gain, int size)
    {
        int groups = frames.Length / groupFrames, shift = gain == null ? 0 : (int)Math.Log2(Super / gain.Dims.X);
        float[] gainData = gain?.GetHost(Intent.Read)[0];
        var result = new float[groups][];
        var image = new float[(long)Super * Super];
        using var input = new Image(IntPtr.Zero, new int3(Super, Super, 1));
        using var scaled = new Image(IntPtr.Zero, new int3(size, size, 1));
        for (int g = 0; g < groups; g++)
        {
            Array.Clear(image);
            for (int f = g * groupFrames; f < (g + 1) * groupFrames; f++)
                foreach (var e in frames[f])
                    image[(long)e.Y16 * Super + e.X16] += gainData == null ? 1 : gainData[(long)(e.Y16 >> shift) * gain.Dims.X + (e.X16 >> shift)];
            GPU.CopyHostToDevice(image, input.GetDevice(Intent.Write), image.LongLength);
            GPU.Scale(input.GetDevice(Intent.Read), scaled.GetDevice(Intent.Write), input.Dims, scaled.Dims, 1, 0, 0, IntPtr.Zero, IntPtr.Zero);
            result[g] = scaled.GetHost(Intent.Read)[0].ToArray();
        }
        return result;
    }

    /// <summary>The error relative to the reference's variation (its mean is the electron dose, which is easy to match).</summary>
    static (double Error, double MeanRatio) Compare(float[] actual, float[] reference)
    {
        double mean = reference.Average(v => (double)v), variation = 0, error = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            variation += (reference[i] - mean) * (reference[i] - mean);
            error += (actual[i] - reference[i]) * (double)(actual[i] - reference[i]);
        }
        return (Math.Sqrt(error / variation), actual.Average(v => (double)v) / mean);
    }

    [CtfCudaFact]
    public void RenderedGroupsMatchRenderingAt16KThenCropping()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var frames = RandomFrames(4, 150000, 11);
            string path = Path.Combine(root, "render.eer");
            WriteEer(path, frames, true, true);
            var memory = new FileInMemory();
            memory.Read(path);
            var index = EerIndex.Read(path, memory, frames.Length);

            var random = new Random(5);
            using var gain = new Image(new int3(8192, 8192, 1));
            gain.TransformValues(_ => 0.5f + (float)random.NextDouble());

            foreach (var (size, width, tolerance) in new[] { (512, 4, 3e-3), (1844, 4, 3e-3), (1844, 6, 3e-4), (1024, 1, 1e-5) })
            {
                var expected = Reference(frames, 2, gain, size);
                foreach (bool onDevice in new[] { false, true })
                {
                    using var stack = new Image(new int3(size, size, 2));
                    EerRenderer.RenderStack(path, onDevice ? memory : null, index, 2, 2, gain, size, 3,
                                            onDevice ? stack.GetDevice(Intent.Write) : IntPtr.Zero, onDevice ? null : stack.GetHost(Intent.Write), width);
                    var actual = stack.GetHost(Intent.Read);

                    // Fixed-point sums: the same floats every time, whatever order the atomics ran in.
                    using var again = new Image(new int3(size, size, 2));
                    EerRenderer.RenderStack(path, memory, index, 2, 2, gain, size, 3, IntPtr.Zero, again.GetHost(Intent.Write), width);
                    for (int g = 0; g < 2; g++)
                        Assert.Equal(actual[g], again.GetHost(Intent.Read)[g]);

                    for (int g = 0; g < 2; g++)
                    {
                        var (error, meanRatio) = Compare(actual[g], expected[g]);
                        output.WriteLine($"{size}px, width {width}, {(onDevice ? "memory, device stack" : "file, host stack")}, group {g}: error {error:E2}, mean ratio {meanRatio:F6}");
                        Assert.True(error < tolerance, $"error {error:E2} at {size}px, width {width}");
                        Assert.InRange(meanRatio, 1 - 1e-4, 1 + 1e-4);
                    }
                }
            }
        }
    }
}
