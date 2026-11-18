using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using Warp;
using Warp.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Tests;

public class JunkNetCudaTests
{
    readonly ITestOutputHelper Output;
    public JunkNetCudaTests(ITestOutputHelper output) { Output = output; }

    /// <summary>
    /// Against a reference bundle from the PyTorch model (WARP_JUNKNET_BUNDLE: its weights, and for each case the
    /// half-averages at 6 Å/pixel and every intermediate image as little-endian float32): input preparation, network
    /// output, and the average-only input for movies without halves.
    /// </summary>
    [TemplateMatchCudaFact]
    public void MatchesThePyTorchReference()
    {
        string bundle = Environment.GetEnvironmentVariable("WARP_JUNKNET_BUNDLE");
        if (string.IsNullOrEmpty(bundle)) return;
        using var handoff = JsonDocument.Parse(File.ReadAllText(Path.Combine(bundle, "handoff.json")));
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var network = new JunkNetTorch(0);
            network.Load(Environment.GetEnvironmentVariable(JunkNetTorch.ModelVariable) is { Length: > 0 } model ? model : Path.Combine(bundle, "inference-weights.pt"));
            foreach (var entry in handoff.RootElement.GetProperty("cases").EnumerateArray())
            {
                var files = entry.GetProperty("files");
                var shape = files.GetProperty("odd").GetProperty("shape");
                int height = shape[0].GetInt32(), width = shape[1].GetInt32();
                float[] Read(string name) => ReadFloats(Path.Combine(bundle, files.GetProperty(name).GetProperty("path").GetString()), width * height);
                float[] odd = Read("odd"), even = Read("even");

                float[] mean = JunkNetInput.MeanOfHalves(odd, even, width, height);
                Assert.Equal(0, Mismatches(mean, Read("pre_network_image"), 0));
                float[] input = JunkNetInput.NormalizeCentral(mean, width, height);
                float[] reference = Read("input");
                Assert.True(Mismatches(input, reference, 1e-6f) == 0, $"input differs by up to {MaxError(input, reference):G3}");

                network.Predict(reference, width, height);
                var timer = Stopwatch.StartNew();
                float[] logits = network.Predict(reference, width, height);
                double seconds = timer.Elapsed.TotalSeconds;
                float[] expected = Read("logits");
                double agreement = Agreement(logits, expected);
                Output.WriteLine($"{width}×{height}: logits max error {MaxError(logits, expected):G3}, mean {MeanError(logits, expected):G3}; " +
                                 $"mask agreement {agreement:P3}; junk {Junk(logits):P1} vs {Junk(expected):P1}; {seconds * 1000:F0} ms");
                Assert.True(agreement > 0.999, $"masks agree on only {agreement:P3}");

                // Movies without half-averages are segmented from the average alone.
                float[] average = odd.Zip(even, (a, b) => (a + b) / 2).ToArray();
                float[] fromAverage = network.Predict(JunkNetInput.FromAverage(average, width, height), width, height);
                double iou = IoU(fromAverage, logits);
                Output.WriteLine($"  average only: mask IoU {iou:P2} with the halves' mask, junk {Junk(fromAverage):P1}");
                Assert.True(iou > 0.99, $"average-only mask IoU {iou:P2}");
            }
        }
    }

    /// <summary>
    /// Outside the junk the result is the deconvolution of the erased image alone. Under it, the image before
    /// deconvolution comes back with its median and spread just inside the edge matched to the deconvolved ice just beyond
    /// the erased zone, so dark junk neither saturates nor leaves a seam, as it does in the ice's noise units.
    /// </summary>
    [TemplateMatchCudaFact]
    public void RestoredJunkMeetsTheIceAtItsEdge()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            // 600 pixels: the deconvolution canvas crops back to exactly the image, so canvas and image pixels coincide.
            const int w = 600, h = 600;
            var random = new Random(11);
            var original = new float[w * h]; var erased = new float[w * h];
            for (int i = 0; i < original.Length; i++) { float noise = (float)random.NextDouble() * 2 - 1; erased[i] = noise; original[i] = noise; }
            var junkMask = new bool[w * h];
            for (int y = 200; y < 400; y++) for (int x = 200; x < 400; x++) { junkMask[y * w + x] = true; original[y * w + x] -= 6; }
            // The masks preparation uses: grown and feathered for erasing, the junk itself for restoring (6 Å/pixel).
            float[] distance = MaskErasure.DistanceTo(junkMask, w, h);
            float[] objects = MaskErasure.Feather(Array.ConvertAll(distance, d => d <= 6), w, h, 10);
            float[] junk = MaskErasure.Feather(junkMask, w, h, 2);
            var ctf = new CTF { PixelSize = 6, Defocus = 1.5m, Voltage = 300, Amplitude = .07m };
            float[] alone = MovieDenoising.Deconvolve(erased, w, h, ctf, 6, .9f, .1f).Data;
            float[] restored = MovieDenoising.Deconvolve(erased, original, objects, junk, w, h, ctf, 6, .9f, .1f).Data;
            for (int i = 0; i < restored.Length; i += 7)
                if (junk[i] <= 0) Assert.InRange(Math.Abs(restored[i] - alone[i]), 0, 1e-4);

            float[] insideDistance = MaskErasure.DistanceTo(Array.ConvertAll(junkMask, v => !v), w, h);
            float[] Band(float[] image, Func<int, bool> where) => Enumerable.Range(0, w * h).Where(where).Select(i => image[i]).ToArray();
            var (innerMedian, innerSpread) = Noise2MicPreprocessing.MedianAndSpread(Band(restored, i => junkMask[i] && insideDistance[i] <= 4));
            var (outerMedian, outerSpread) = Noise2MicPreprocessing.MedianAndSpread(Band(restored, i => !junkMask[i] && objects[i] <= 0 && distance[i] <= 21));
            Output.WriteLine($"inner median {innerMedian:F3} spread {innerSpread:F3}; outer median {outerMedian:F3} spread {outerSpread:F3}");
            Assert.InRange(innerMedian - outerMedian, -0.15, 0.15);
            Assert.InRange(innerSpread / outerSpread, 0.8, 1.25);

            // In the ice's noise units (median and spread outside the erased zone), the junk sits 6 noise widths down.
            var (iceMedian, iceSpread) = Noise2MicPreprocessing.MedianAndSpread(Band(original, i => objects[i] <= 0));
            int clipped = Enumerable.Range(0, w * h).Count(i => junkMask[i] && Math.Abs(restored[i]) >= 3.999f);
            int clippedInIceUnits = Enumerable.Range(0, w * h).Count(i => junkMask[i] && Math.Abs((original[i] - iceMedian) / iceSpread) >= 3.999f);
            Output.WriteLine($"clipped junk pixels: {clipped} vs {clippedInIceUnits} in the ice's noise units");
            Assert.True(clipped < clippedInIceUnits / 4);
        }
    }

    static float[] ReadFloats(string path, int count)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != count * 4) throw new InvalidDataException($"{path}: expected {count} float32 values.");
        var values = new float[count];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    static int Mismatches(float[] a, float[] b, float tolerance) => a.Zip(b).Count(p => Math.Abs(p.First - p.Second) > tolerance);
    static double MaxError(float[] a, float[] b) => a.Zip(b).Max(p => Math.Abs((double)p.First - p.Second));
    static double MeanError(float[] a, float[] b) => a.Zip(b).Average(p => Math.Abs((double)p.First - p.Second));
    static double Agreement(float[] a, float[] b) => a.Zip(b).Count(p => p.First > 0 == p.Second > 0) / (double)a.Length;
    static double Junk(float[] logits) => logits.Count(v => v > 0) / (double)logits.Length;
    static double IoU(float[] a, float[] b)
    {
        int both = a.Zip(b).Count(p => p.First > 0 && p.Second > 0), either = a.Zip(b).Count(p => p.First > 0 || p.Second > 0);
        return either == 0 ? 1 : both / (double)either;
    }
}
