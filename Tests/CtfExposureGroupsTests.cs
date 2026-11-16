using System;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfExposureGroupsTests
{
    [Theory]
    [InlineData(40, 35.0, 12)]   // 2.92 e⁻/Å² per group beats 11 groups' 3.18
    [InlineData(40, 10.0, 3)]    // 3.33 beats 4 groups' 2.5
    [InlineData(40, 4.2, 2)]     // 2.1 beats one group's 4.2, though 4.2 / 3 rounds to 1
    [InlineData(40, 2.0, 1)]     // less than one group's worth: the whole movie
    [InlineData(5, 60.0, 5)]     // never more groups than frames
    public void GroupsAreTheIntegerCountClosestToThreeElectrons(int frames, double exposure, int groups)
    {
        var ranges = CtfExposureGroups.Partition(frames, exposure, (double)CtfExposureGroups.DefaultExposure);
        Assert.Equal(groups, ranges.Length);
        Assert.Equal(0, ranges[0].Start);
        Assert.Equal(frames, ranges[^1].End);
        for (int g = 1; g < ranges.Length; g++) Assert.Equal(ranges[g - 1].End, ranges[g].Start);
        int smallest = ranges.Min(r => r.End - r.Start), largest = ranges.Max(r => r.End - r.Start);
        Assert.True(largest - smallest <= 1, "frames are spread evenly");
    }

    [Theory]
    [InlineData(-34.85, 5)]   // Relay: total exposure
    [InlineData(0.85, 5)]     // WarpTools --exposure: per frame, times the movie's 41 frames
    public void AutomaticDefocusGridCountsMovieFramesNotExposureGroups(double dose, int side)
    {
        // 41 frames make 12 exposure groups; sizing the grid by the groups would see 10 e⁻/Å² and give 2 × 2.
        Assert.Equal(12, CtfExposureGroups.Partition(41, 34.85, 3).Length);
        Assert.Equal(new int2(side, side), Movie.AutoCtfGrid(new int2(4096, 4096), 41, (decimal)dose, 1m));
    }

    [Fact]
    public void UnknownExposureKeepsPerFrameSpectra()
    {
        var ranges = CtfExposureGroups.Partition(7, 0, 3);
        Assert.Equal(Enumerable.Range(0, 7).Select(z => (z, z + 1)), ranges);
    }

    [TemplateMatchCudaFact]
    public void GroupsAlignFramesLikeTheExportedAverage()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            // A smooth random scene, translated per frame; ExportMovie's warp brings each frame back to frame 0's
            // register, and a group average built with the same warps must too.
            int w = 128, h = 96;
            var random = new Random(3);
            float[] scene = MaskErasure.Gaussian(Enumerable.Range(0, w * h).Select(_ => (float)random.NextDouble()).ToArray(), w, h, 2);
            float2[] warps = [new(0, 0), new(2.5f, -1.25f), new(5, -2.5f), new(-3, 4)];
            var frames = new float[warps.Length][];
            using (var reference = new Image(scene, new int3(w, h, 1)))
                for (int z = 0; z < warps.Length; z++)
                {
                    // Frame z holds the scene displaced by some shift; which way round is for the warp to undo.
                    using var shifted = reference.GetCopyGPU();
                    shifted.ShiftSlicesMassive([new float3(-warps[z].X, -warps[z].Y, 0)]);
                    frames[z] = (float[])shifted.GetHost(Intent.Read)[0].Clone();
                }

            float[] exported = new float[w * h];
            IntPtr array = GPU.MallocArray(new int2(w, h));
            try
            {
                for (int z = 0; z < warps.Length; z++)
                {
                    using var frame = new Image(frames[z], new int3(w, h, 1));
                    using var aligned = new Image(new int3(w, h, 1));
                    GPU.PrefilterForCubic(frame.GetDevice(Intent.ReadWrite), frame.Dims);
                    var wx = Enumerable.Repeat(warps[z].X, 4).ToArray(); var wy = Enumerable.Repeat(warps[z].Y, 4).ToArray();
                    GPU.WarpImage(frame.GetDevice(Intent.Read), aligned.GetDevice(Intent.Write), new int2(w, h), wx, wy, new int2(2, 2), array);
                    float[] a = aligned.GetHost(Intent.Read)[0];
                    for (int i = 0; i < a.Length; i++) exported[i] += a[i] / warps.Length;
                }
            }
            finally { GPU.FreeArray(array); }

            using var stack = new Image(frames, new int3(w, h, warps.Length));
            using var grouped = CtfExposureGroups.Average(stack, warps, [(0, warps.Length)]);
            float[] mine = grouped.GetHost(Intent.Read)[0];

            // Compare away from the borders, where the warp clamps and the Fourier shift wraps.
            double error = 0, signal = 0;
            for (int y = 12; y < h - 12; y++)
                for (int x = 12; x < w - 12; x++)
                {
                    int i = y * w + x;
                    error += Math.Pow(mine[i] - exported[i], 2);
                    signal += Math.Pow(exported[i] - scene.Average(), 2);
                }
            // And the shifts must actually matter: unaligned frames would not match.
            double unaligned = 0;
            float[] first = frames[^1];
            for (int y = 12; y < h - 12; y++)
                for (int x = 12; x < w - 12; x++) unaligned += Math.Pow(first[y * w + x] - exported[y * w + x], 2);
            Assert.True(unaligned / signal > 0.1, $"test frames too similar ({unaligned / signal:E2})");
            Assert.True(error / signal < 0.01, $"relative error {error / signal:E2}");
        }
    }

    [TemplateMatchCudaFact]
    public void InPlaceAverageEqualsTheFrameByFrameAverageExactly()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            // Odd and even sizes (the Nyquist row and column), fractional shifts, uneven, overlapping and partial groups.
            foreach (var (w, h) in new[] { (128, 96), (75, 64), (64, 81) })
            {
                var random = new Random(w * h);
                int frames = 7;
                var data = Enumerable.Range(0, frames).Select(_ => Enumerable.Range(0, w * h).Select(_ => (float)(random.NextDouble() * 4 - 1)).ToArray()).ToArray();
                float2[] warps = Enumerable.Range(0, frames).Select(_ => new float2((float)(random.NextDouble() * 9 - 4.5), (float)(random.NextDouble() * 9 - 4.5))).ToArray();
                (int, int)[] groups = [(0, 3), (3, 5), (5, 7), (1, 6), (4, 5)];
                using var stack = new Image(data, new int3(w, h, frames));
                using var fused = CtfExposureGroups.Average(stack, warps, groups);
                using var reference = CtfExposureGroups.AverageFrameByFrame(stack, warps, groups);
                float[][] a = fused.GetHost(Intent.Read), b = reference.GetHost(Intent.Read);
                Assert.Equal(groups.Length, a.Length);
                for (int g = 0; g < groups.Length; g++)
                    for (int i = 0; i < w * h; i++)
                        Assert.True(BitConverter.SingleToInt32Bits(a[g][i]) == BitConverter.SingleToInt32Bits(b[g][i]),
                                    $"{w}x{h} group {g}, pixel {i}: {a[g][i]:G9} vs {b[g][i]:G9}");
                // The stack's device copy is read, not overwritten.
                using (var device = stack.GetCopyGPU())
                    for (int z = 0; z < frames; z++) Assert.Equal(data[z], device.GetHost(Intent.Read)[z]);
            }
        }
    }
}
