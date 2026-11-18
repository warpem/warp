using System;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Warp.Workers;
using Xunit;

namespace Tests;

public class MovieJunkTests
{
    [Fact]
    public void CentralNormalizationUsesTheCentralHalfMedianAndNumpysLinearPercentile()
    {
        // 8 × 8: the central half is rows and columns 2..5, values 100..115 there; everything else is far outside.
        const int n = 8;
        var image = new float[n * n];
        for (int i = 0; i < image.Length; i++) image[i] = -1000;
        for (int y = 2, v = 100; y < 6; y++)
            for (int x = 2; x < 6; x++)
                image[y * n + x] = v++;
        float[] normalized = JunkNetInput.NormalizeCentral(image, n, n);
        // numpy: median 107.5; percentile 84.13 at index 0.8413 * 15 = 12.6195 → 112 + 0.6195 = 112.6195.
        float scale = (float)(112.6195 - 107.5);
        Assert.InRange(Math.Abs((100 - 107.5f) / scale - normalized[2 * n + 2]), 0, 1e-5);
        Assert.InRange(Math.Abs((115 - 107.5f) / scale - normalized[5 * n + 5]), 0, 1e-5);
        Assert.Equal(-JunkNetInput.Clip, normalized[0]);
    }

    [Fact]
    public void HalvesAreNormalizedAloneAveragedAndStoredAsFloat16()
    {
        const int w = 520, h = 516;
        var random = new Random(3);
        float[] odd = Enumerable.Range(0, w * h).Select(_ => (float)random.NextDouble() * 10 + 50).ToArray();
        float[] even = odd.Select(v => v * 3 + 7).ToArray();
        float[] mean = JunkNetInput.MeanOfHalves(odd, even, w, h);
        // A half scaled and shifted normalizes to the same image, so the mean is that image, in float16 steps.
        float[] alone = Noise2MicPreprocessing.NormalizeRobustly(odd, w, h);
        for (int i = 0; i < mean.Length; i += 97)
        {
            Assert.Equal((float)(Half)mean[i], mean[i]);
            Assert.InRange(Math.Abs(alone[i] - mean[i]), 0, 0.005);
        }
        float[] input = JunkNetInput.FromHalves(odd, even, w, h);
        Assert.All(input, v => Assert.InRange(v, -JunkNetInput.Clip, JunkNetInput.Clip));
    }

    [Fact]
    public void SmallMicrographsAreNormalizedOnAPatchSizedReflection()
    {
        const int w = 300, h = 200;
        float[] image = Enumerable.Range(0, w * h).Select(i => (float)Math.Sin(i * 0.01)).ToArray();
        float[] input = JunkNetInput.FromAverage(image, w, h);
        Assert.Equal(w * h, input.Length);
        Assert.All(input, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void MasksAreSavedAt8AngstromsAndReadBackOntoAnyGrid()
    {
        string root = Path.Combine(Path.GetTempPath(), "junk-mask-" + Guid.NewGuid().ToString("N"));
        try
        {
            // 6 Å/pixel, 160 × 120, junk in a disc near the top left corner.
            const int w = 160, h = 120;
            var mask = new bool[w * h];
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) mask[y * w + x] = (x - 40) * (x - 40) + (y - 30) * (y - 30) <= 400;
            string path = Path.Combine(root, "mask", "movie.tif");
            MovieJunk.WriteMask(path, mask, w, h);

            using (var saved = Image.FromFile(path))
                Assert.Equal(new int3(120, 90, 1), saved.Dims); // 160 · 6/8, 120 · 6/8
            bool[] back = MovieJunk.ReadMask(path, w, h);
            int both = 0, either = 0;
            for (int i = 0; i < mask.Length; i++) { if (mask[i] && back[i]) both++; if (mask[i] || back[i]) either++; }
            Assert.True(both / (double)either > 0.9, $"IoU {both / (double)either:F3}");
            // Not flipped: the disc stays at the top left.
            Assert.True(back[30 * w + 40]); Assert.False(back[90 * w + 40]);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public void JunkIsErasedUnderAGrownAndFeatheredMask()
    {
        string root = Path.Combine(Path.GetTempPath(), "junk-erase-" + Guid.NewGuid().ToString("N"));
        try
        {
            const int w = 200, h = 200;
            var mask = new bool[w * h];
            for (int y = 0; y < h; y++) for (int x = 0; x < 60; x++) mask[y * w + x] = true;
            string path = Path.Combine(root, "movie.tif");
            MovieJunk.WriteMask(path, mask, w, h);
            float[] soft = MovieDenoising.JunkMask(path, w, h, 6);
            // Grown by 36 Å (6 pixels), then feathered over 60 Å (10 pixels).
            Assert.Equal(1, soft[100 * w + 30]);
            Assert.Equal(1, soft[100 * w + 64]);
            Assert.InRange(soft[100 * w + 70], 0.01f, 0.99f);
            Assert.Equal(0, soft[100 * w + 80]);

            // Restoring after deconvolution follows the junk itself, feathered over 12 Å (2 pixels).
            float[] restore = MovieDenoising.JunkMasks(path, w, h, 6).Restore;
            Assert.Equal(1, restore[100 * w + 30]);
            Assert.Equal(0, restore[100 * w + 64]);

            MovieJunk.WriteMask(path, new bool[w * h], w, h);
            Assert.Null(MovieDenoising.JunkMask(path, w, h, 6));
            Assert.Equal((null, null), MovieDenoising.JunkMasks(path, w, h, 6));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public void JunkIsLeftAsTheNetworkGotItAtTheDenoisedIcesLevel()
    {
        // 6 Å/pixel. Junk fills x < 60; it is erased 6 pixels further and feathered over 10 more, restored over 2.
        const int w = 200, h = 200;
        var random = new Random(7);
        var junkMask = new bool[w * h];
        for (int y = 0; y < h; y++) for (int x = 0; x < 60; x++) junkMask[y * w + x] = true;
        float[] distance = MaskErasure.DistanceTo(junkMask, w, h);
        float[] objects = MaskErasure.Feather(Array.ConvertAll(distance, d => d <= 6), w, h, 10);
        float[] junk = MaskErasure.Feather(junkMask, w, h, 2);
        // The network's input is noisy everywhere and offset under the junk; its output is smooth and offset elsewhere.
        float[] prepared = Enumerable.Range(0, w * h).Select(i => (float)(random.NextDouble() * 2 - 1) + (junkMask[i] ? -2 : 0)).ToArray();
        float[] denoised = Enumerable.Range(0, w * h).Select(i => 0.5f + 0.01f * (i % w)).ToArray();
        float[] shown = MovieDenoising.WithJunkAsPrepared(denoised, prepared, junk, objects, w, h, 6);

        for (int i = 0; i < shown.Length; i++)
            if (junk[i] <= 0) Assert.Equal(denoised[i], shown[i]);
        // Under the junk: the input's noise, shifted as one so the band inside the edge meets the denoised ice beyond the erased zone.
        float[] inside = MaskErasure.DistanceTo(Array.ConvertAll(junkMask, v => !v), w, h);
        int[] full = Enumerable.Range(0, w * h).Where(i => junk[i] >= 1).ToArray();
        float shift = shown[full[0]] - prepared[full[0]];
        Assert.All(full, i => Assert.InRange(shown[i] - prepared[i] - shift, -1e-4f, 1e-4f));
        var (inner, _) = Noise2MicPreprocessing.MedianAndSpread(full.Where(i => inside[i] <= 4).Select(i => shown[i]).ToArray());
        float reach = (MovieDenoising.EraseMarginAngstrom + MovieDenoising.EraseFeatherAngstrom + MovieDenoising.BandAngstrom) / 6 + 1;
        var (outer, _) = Noise2MicPreprocessing.MedianAndSpread(Enumerable.Range(0, w * h).Where(i => objects[i] <= 0 && distance[i] <= reach).Select(i => denoised[i]).ToArray());
        Assert.InRange(inner - outer, -0.05, 0.05);
    }

    [Fact]
    public void PicksWhoseEdgeComesTooCloseToJunkAreDropped()
    {
        string root = Path.Combine(Path.GetTempPath(), "junk-picks-" + Guid.NewGuid().ToString("N"));
        try
        {
            // Junk fills x < 80 of a 240 × 240 mask at 6 Å; the micrograph has the same field at 3 Å, 480 × 480.
            const int w = 240;
            var mask = new bool[w * w];
            for (int y = 0; y < w; y++) for (int x = 0; x < 80; x++) mask[y * w + x] = true;
            string path = Path.Combine(root, "movie.tif");
            MovieJunk.WriteMask(path, mask, w, w);
            // Junk edge at x = 160 micrograph pixels. Radius 20 pixels (60 Å) and 10 pixels more distance.
            var centres = new (float X, float Y)[] { (100, 240), (175, 240), (186, 240), (200, 240), (400, 240) };
            bool[] keep = MovieJunk.ClearOfJunk(path, 480, 480, centres, 20, 10);
            Assert.Equal(new[] { false, false, false, true, true }, keep);
            Assert.All(MovieJunk.ClearOfJunk(null, 480, 480, centres, 20, 10), Assert.True);
            Assert.Throws<FileNotFoundException>(() => MovieJunk.ClearOfJunk(Path.Combine(root, "missing.tif"), 480, 480, centres, 20, 10));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public void JunkCommandsCarryTheMaskTheDistanceAndTheModel()
    {
        Assert.Equal(new object[] { "", 0f }, WorkerCommands.MicrographPickBlobs("a.mrc", 10).Content[4..]);
        var culled = WorkerCommands.MicrographPickBlobs("a.mrc", 10, junkMaskPath: "mask/a.tif", junkDistance: 5);
        Assert.Equal(new object[] { "mask/a.tif", 5f }, culled.Content[4..]);
        Assert.Throws<ArgumentOutOfRangeException>(() => WorkerCommands.MicrographPickBlobs("a.mrc", 10, junkMaskPath: "m.tif", junkDistance: -1));
        Assert.Equal("", WorkerCommands.MovieSegmentJunk("a.eer", 1.2f).Content[2]);
        Assert.Equal("model.pt", WorkerCommands.MovieSegmentJunk("a.eer", 1.2f, "model.pt").Content[2]);
    }
}
