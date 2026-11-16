using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Tests;

/// <summary>
/// MRC write and read times against plain file I/O of the same bytes, not part of the regular suite: WARP_MRC_BENCH
/// names folders to write into, separated by ':' (a network file system and local storage, say).
/// </summary>
public class MrcIoBenchmark
{
    readonly ITestOutputHelper output;
    public MrcIoBenchmark(ITestOutputHelper output) { this.output = output; }

    public sealed class MrcBenchFactAttribute : FactAttribute
    {
        public MrcBenchFactAttribute()
        {
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("WARP_MRC_BENCH")))
                Skip = "Set WARP_MRC_BENCH to folders to write into.";
        }
    }

    static double Time(Action action, int repeats = 3)
    {
        action();
        var timer = Stopwatch.StartNew();
        for (int r = 0; r < repeats; r++)
            action();
        return timer.Elapsed.TotalMilliseconds / repeats;
    }

    [MrcBenchFact]
    public void WriteAndReadTimes()
    {
        var random = new Random(1);
        foreach (string root in Environment.GetEnvironmentVariable("WARP_MRC_BENCH").Split(':'))
        {
            string folder = Path.Combine(root, "warp-mrc-bench-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(folder);
            try
            {
                foreach (var (name, dims) in new[] { ("micrograph 4096²", new int3(4096, 4096, 1)), ("micrograph 2048²", new int3(2048, 2048, 1)), ("particles 500×256²", new int3(256, 256, 500)) })
                {
                    using var image = new Image(dims);
                    foreach (var slice in image.GetHost(Intent.Write))
                        for (int i = 0; i < slice.Length; i++)
                            slice[i] = (float)random.NextDouble() * 100 - 50;

                    string half = Path.Combine(folder, "half.mrc"), single = Path.Combine(folder, "single.mrc"), raw = Path.Combine(folder, "raw.bin");
                    long halfBytes = dims.Elements() * 2, singleBytes = dims.Elements() * 4;
                    var bytes = new byte[singleBytes];

                    double writeHalf = Time(() => image.WriteMRC16b(half, 1, true));
                    double writeSingle = Time(() => image.WriteMRC(single, 1, true));
                    double rawHalf = Time(() => File.WriteAllBytes(raw, bytes.AsSpan(0, (int)halfBytes).ToArray()));
                    double rawSingle = Time(() => File.WriteAllBytes(raw, bytes));
                    double readHalf = Time(() => Image.FromFile(half).Dispose());
                    double readSingle = Time(() => Image.FromFile(single).Dispose());
                    double rawRead = Time(() => File.ReadAllBytes(single));
                    output.WriteLine($"{Path.GetFileName(root.TrimEnd('/'))}, {name}: write float16 {writeHalf:F1} ms (raw {rawHalf:F1}), float32 {writeSingle:F1} ms (raw {rawSingle:F1}); " +
                                     $"read float16 {readHalf:F1} ms, float32 {readSingle:F1} ms (raw {rawRead:F1})");
                }
            }
            finally { Directory.Delete(folder, true); }
        }
    }
}
