using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using Warp;
using Warp.Headers;
using Warp.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Tests;

/// <summary>
/// Real EER movies, not part of the regular suite: WARP_EER_BENCH names a folder of .eer files (WARP_EER_BENCH_COUNT
/// movies, 3 by default; WARP_EER_BENCH_SCALE the scale factor, 0.5 by default).
/// </summary>
public class EerRenderBenchmark
{
    readonly ITestOutputHelper output;
    public EerRenderBenchmark(ITestOutputHelper output) { this.output = output; }

    public sealed class EerBenchFactAttribute : FactAttribute
    {
        public EerBenchFactAttribute()
        {
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("WARP_EER_BENCH")))
                Skip = "Set WARP_EER_BENCH to a folder of EER movies on a CUDA host.";
        }
    }

    static string[] Movies(int fallbackCount)
    {
        string folder = Environment.GetEnvironmentVariable("WARP_EER_BENCH");
        int count = int.TryParse(Environment.GetEnvironmentVariable("WARP_EER_BENCH_COUNT"), out int c) ? c : fallbackCount;
        return Directory.GetFiles(folder, "*.eer").OrderBy(p => p).Take(count).ToArray();
    }

    // The tests do not reference the worker's assembly; it is loaded from the worker's own output folder.
    static readonly Type Worker = Assembly.LoadFrom(WorkerAssembly()).GetType("WarpWorker2.WorkerProcess", true);
    static string WorkerAssembly()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            foreach (string folder in new[] { "", "Release", "bin" })
                if (File.Exists(Path.Combine(dir.FullName, folder, "WarpWorker2.dll")))
                    return Path.Combine(dir.FullName, folder, "WarpWorker2.dll");
        throw new FileNotFoundException("WarpWorker2.dll was not found above the test folder.");
    }
    static T Call<T>(string name, params object[] args) =>
        (T)Worker.GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, args);

    /// <summary>The lowest free device memory seen while it runs, sampled every few milliseconds.</summary>
    sealed class FreeMemoryWatch : IDisposable
    {
        volatile bool stop;
        readonly Thread thread;
        public long LowestMB = long.MaxValue;
        public FreeMemoryWatch()
        {
            thread = new Thread(() =>
            {
                GPU.SetDevice(0);
                while (!stop) { LowestMB = Math.Min(LowestMB, GPU.GetFreeMemory(0)); Thread.Sleep(3); }
            });
            thread.Start();
        }
        public void Dispose() { stop = true; thread.Join(); }
    }

    [EerBenchFact]
    public void LoadStackTimings()
    {
        decimal scale = decimal.TryParse(Environment.GetEnvironmentVariable("WARP_EER_BENCH_SCALE"), out decimal s) ? s : 0.5M;
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            HeaderEER.GroupNFrames = -40;
            foreach (string movie in Movies(3))
            {
                long freeBefore = GPU.GetFreeMemory(0);
                var timer = Stopwatch.StartNew();
                Image stack;
                long lowest;
                using (var watch = new FreeMemoryWatch())
                {
                    stack = Call<Image>("LoadAndPrepareStack", movie, scale, false, 8);
                    lowest = watch.LowestMB;
                }
                double ms = timer.Elapsed.TotalMilliseconds;
                output.WriteLine($"{Path.GetFileName(movie)[..12]}…: {stack.Dims} in {ms:F0} ms, " +
                                 $"device peak {freeBefore - lowest} MB above the start, process {Process.GetCurrentProcess().WorkingSet64 >> 20} MB resident");
                Call<object>("ReleaseResidentStack");
            }
        }
    }

    /// <summary>Render time against decoding threads: flat means the GPU is the limit, falling means decoding is.</summary>
    [EerBenchFact]
    public void RenderScaling()
    {
        string movie = Movies(1)[0];
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var memory = new FileInMemory();
            memory.Read(movie);
            var index = EerIndex.Read(movie, memory, 1);
            int all = index.Frames / 15;
            using var stack = new Image(IntPtr.Zero, new int3(2048, 2048, all));
            foreach (int threads in new[] { 8, 1, 2, 4, 8, 12, 16 })
            {
                var timer = Stopwatch.StartNew();
                EerRenderer.RenderStack(movie, memory, index, 15, all, null, 2048, threads, stack.GetDevice(Intent.Write), null);
                output.WriteLine($"{threads} threads: {timer.Elapsed.TotalMilliseconds:F0} ms");
            }

            // Decoding alone, on all threads at once.
            long capacity = Enumerable.Range(0, index.Frames).Max(index.MaxElectrons);
            var buffers = Enumerable.Range(0, 16).Select(_ => GC.AllocateUninitializedArray<uint>((int)capacity, pinned: true)).ToArray();
            var decodeTimer = Stopwatch.StartNew();
            long total = 0;
            System.Threading.Tasks.Parallel.For(0, index.Frames, new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 1 }, f =>
            {
                var pointer = System.Runtime.InteropServices.Marshal.UnsafeAddrOfPinnedArrayElement(buffers[0], 0);
                System.Threading.Interlocked.Add(ref total, EERNative.DecodeFrames(memory.Pointer, memory.Length, index.Offsets, index.Sizes, f, 1, index.Compression, pointer, capacity));
            });
            output.WriteLine($"Decoding {index.Frames} frames on one thread: {decodeTimer.Elapsed.TotalMilliseconds:F0} ms, {total / decodeTimer.Elapsed.TotalSeconds / 1e6:F0}M electrons/s");
            foreach (int threads in new[] { 4, 8, 16 })
            {
                decodeTimer.Restart();
                System.Threading.Tasks.Parallel.For(0, index.Frames, new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = threads }, f =>
                {
                    var pointer = System.Runtime.InteropServices.Marshal.UnsafeAddrOfPinnedArrayElement(buffers[f % 16], 0);
                    EERNative.DecodeFrames(memory.Pointer, memory.Length, index.Offsets, index.Sizes, f, 1, index.Compression, pointer, capacity);
                });
                output.WriteLine($"Decoding {index.Frames} frames on {threads} threads: {decodeTimer.Elapsed.TotalMilliseconds:F0} ms");
            }

            // The GPU's parts: a batch of 32 frames uploaded and spread as one group, and a group's transforms.
            var counts = new long[32];
            for (int f = 0; f < 32; f++)
                counts[f] = EERNative.DecodeFrames(memory.Pointer, memory.Length, index.Offsets, index.Sizes, f, 1, index.Compression,
                    System.Runtime.InteropServices.Marshal.UnsafeAddrOfPinnedArrayElement(buffers[f % 16], 0), capacity);
            var batch = GC.AllocateUninitializedArray<uint>((int)(32 * capacity), pinned: true);
            for (int f = 0; f < 32; f++)
                Array.Copy(buffers[f % 16], 0, batch, f * capacity, counts[f]);
            IntPtr pin = System.Runtime.InteropServices.Marshal.UnsafeAddrOfPinnedArrayElement(batch, 0);
            foreach (int size in new[] { 2048, 4096 })
            {
                var setup = Stopwatch.StartNew();
                for (int r = 0; r < 5; r++)
                    new EerRenderer(size, 32, capacity, 15, 1).Dispose();
                output.WriteLine($"{size}px: creating and disposing a renderer {setup.Elapsed.TotalMilliseconds / 5:F1} ms");
            }
            foreach (int width in new[] { 4, 1 })
            foreach (int size in width == 1 ? new[] { 8192 } : new[] { 1844, 2048, 4096 })
            {
                using var renderer = new EerRenderer(size, 32, capacity, 32, 1, width);
                IntPtr slice = GPU.MallocDevice((long)size * size);
                var one = new float[1];
                void Sync() { renderer.WaitForUploads(); GPU.CopyDeviceToHost(slice, one, 1); } // waits for both streams
                const int repeats = 10;
                Sync();
                var t = Stopwatch.StartNew();
                for (int r = 0; r < repeats; r++) { renderer.Upload(pin, counts, 32); renderer.BatchDone(); }
                Sync();
                double upload = t.Elapsed.TotalMilliseconds / repeats / 32 * 1000;
                t.Restart();
                for (int r = 0; r < repeats; r++) { renderer.Upload(pin, counts, 32); renderer.Spread(0, 32, null); renderer.BatchDone(); }
                Sync();
                double both = t.Elapsed.TotalMilliseconds / repeats / 32 * 1000;
                t.Restart();
                for (int r = 0; r < repeats; r++) renderer.Finish(slice);
                Sync();
                double finish = t.Elapsed.TotalMilliseconds / repeats * 1000;
                output.WriteLine($"{size}px (grid {renderer.GridSize}, width {renderer.KernelWidth}), {counts.Sum() / 32000}k electrons a frame: upload {upload:F0} µs, upload and spread {both:F0} µs per frame; Finish {finish:F0} µs per group");
                GPU.FreeDevice(slice);
            }
        }
    }

    /// <summary>Power in frequency shells (fractions of the output's Nyquist), from the image's transform.</summary>
    static double[] ShellPower(float[] image, int size, double[] edges)
    {
        using var picture = new Image(image, new int3(size, size, 1));
        using var ft = picture.AsFFT();
        float[] data = ft.GetHost(Intent.Read)[0];
        var power = new double[edges.Length - 1];
        for (int y = 0; y < size; y++)
        {
            int ky = y <= size / 2 ? y : y - size;
            for (int x = 0; x <= size / 2; x++)
            {
                double r = Math.Sqrt(x * x + ky * ky) / (size / 2);
                int shell = Array.FindIndex(edges, 1, e => r < e) - 1;
                if (r == 0 || shell < 0)
                    continue;
                long i = ((long)y * (size / 2 + 1) + x) * 2;
                power[shell] += data[i] * data[i] + data[i + 1] * data[i + 1];
            }
        }
        return power;
    }

    static double Error(float[] actual, float[] reference)
    {
        double mean = reference.Average(v => (double)v), variation = 0, error = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            variation += (reference[i] - mean) * (reference[i] - mean);
            error += (actual[i] - reference[i]) * (double)(actual[i] - reference[i]);
        }
        return Math.Sqrt(error / variation);
    }

    /// <summary>
    /// Two groups of a real movie against rendering at 16K and Fourier cropping (the reference), and against rendering
    /// at 4K and cropping, which is what keeping no sub-pixel positions would give: the error, and each one's power
    /// relative to the reference's in frequency shells. Then whole-movie render times per kernel width.
    /// </summary>
    [EerBenchFact]
    public void AccuracyAgainstRenderingAt16K()
    {
        string movie = Movies(1)[0];
        const int groupFrames = 15, groups = 2;
        double[] edges = { 0, 0.25, 0.5, 0.75, 0.9, 1.0 };
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var memory = new FileInMemory();
            memory.Read(movie);
            int frames = new HeaderEER(movie, memory.AsStream()).DimensionsUngrouped.Z;
            var index = EerIndex.Read(movie, memory, frames);
            var super = new float[16384L * 16384];
            var physical = new float[4096 * 4096];

            foreach (int size in new[] { 2048, 4096 })
            {
                var reference = new float[groups][];
                var plain = new float[groups][];
                using (var input16 = new Image(IntPtr.Zero, new int3(16384, 16384, 1)))
                using (var input4 = new Image(IntPtr.Zero, new int3(4096, 4096, 1)))
                using (var scaled = new Image(IntPtr.Zero, new int3(size, size, 1)))
                    for (int g = 0; g < groups; g++)
                    {
                        EERNative.ReadEER(movie, g * groupFrames, (g + 1) * groupFrames, 3, super);
                        GPU.CopyHostToDevice(super, input16.GetDevice(Intent.Write), super.LongLength);
                        GPU.Scale(input16.GetDevice(Intent.Read), scaled.GetDevice(Intent.Write), input16.Dims, scaled.Dims, 1, 0, 0, IntPtr.Zero, IntPtr.Zero);
                        reference[g] = scaled.GetHost(Intent.Read)[0].ToArray();

                        EERNative.ReadEER(movie, g * groupFrames, (g + 1) * groupFrames, 1, physical);
                        if (size == 4096)
                            plain[g] = physical.Select(v => v / 16).ToArray();
                        else
                        {
                            GPU.CopyHostToDevice(physical, input4.GetDevice(Intent.Write), physical.Length);
                            GPU.Scale(input4.GetDevice(Intent.Read), scaled.GetDevice(Intent.Write), input4.Dims, scaled.Dims, 1, 0, 0, IntPtr.Zero, IntPtr.Zero);
                            plain[g] = scaled.GetHost(Intent.Read)[0].Select(v => v / 16).ToArray(); // per 16K pixel, as the reference
                        }
                    }

                void Report(string name, float[][] images)
                {
                    var ratios = new double[edges.Length - 1];
                    double error = 0;
                    for (int g = 0; g < groups; g++)
                    {
                        var p = ShellPower(images[g], size, edges);
                        var q = ShellPower(reference[g], size, edges);
                        for (int i = 0; i < ratios.Length; i++)
                            ratios[i] += p[i] / q[i] / groups;
                        error += Error(images[g], reference[g]) / groups;
                    }
                    output.WriteLine($"{size}px {name,-12}: error {error:E2}; power vs 16K by shell " +
                                     string.Join(", ", ratios.Select((r, i) => $"{edges[i]:F2}-{edges[i + 1]:F2}: {r:F4}")));
                }

                Report("4K render", plain);
                foreach (int width in new[] { 4, 5, 6 })
                {
                    using var stack = new Image(new int3(size, size, groups));
                    EerRenderer.RenderStack(movie, memory, index, groupFrames, groups, null, size, 8, IntPtr.Zero, stack.GetHost(Intent.Write), width);
                    Report($"width {width}", stack.GetHost(Intent.Read).Select(a => a.ToArray()).ToArray());
                }

                int all = index.Frames / groupFrames;
                foreach (int width in new[] { 4, 5, 6 })
                {
                    using var stack = new Image(new int3(size, size, all));
                    EerRenderer.RenderStack(movie, memory, index, groupFrames, all, null, size, 8, stack.GetDevice(Intent.Write), null, width); // warm up
                    var timer = Stopwatch.StartNew();
                    EerRenderer.RenderStack(movie, memory, index, groupFrames, all, null, size, 8, stack.GetDevice(Intent.Write), null, width);
                    double device = timer.Elapsed.TotalMilliseconds;
                    timer.Restart();
                    EerRenderer.RenderStack(movie, memory, index, groupFrames, all, null, size, 8, IntPtr.Zero, stack.GetHost(Intent.Write), width);
                    output.WriteLine($"{size}px width {width}: {all} groups in {device:F0} ms into a device stack, {timer.Elapsed.TotalMilliseconds:F0} ms into a host stack");
                }
            }
        }
    }
}
