using System;
using System.Linq;
using Warp.Tools;
using Xunit;

namespace Tests;

public class MaskErasureTests
{
    const int N = 160;

    static float[] Plane(int width, int height, float gx, float gy, float c)
    {
        var data = new float[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                data[y * width + x] = gx * x + gy * y + c;
        return data;
    }

    static float[] Noise(int length, float sigma, int seed)
    {
        var random = new RandomNormal(seed);
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = random.NextSingle(0, sigma);
        return data;
    }

    static bool InDisk(int x, int y, float cx, float cy, float r) => (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r;

    [Fact]
    public void PushPullFillsAHoleAlongTheSurroundingGradient()
    {
        float[] plane = Plane(N, N, 0.3f, -0.2f, 5);
        var weights = Enumerable.Repeat(1f, N * N).ToArray();
        for (int y = 50; y < 110; y++)
            for (int x = 40; x < 120; x++) { weights[y * N + x] = 0; }
        float[] holed = plane.Select((v, i) => weights[i] > 0 ? v : 1000f).ToArray();

        float[] filled = MaskErasure.PushPull(holed, weights, N, N);

        // A constant fill (Warp's old per-component mean) would be off by up to 0.3·40 + 0.2·30 = 18 here.
        double worst = 0;
        for (int i = 0; i < plane.Length; i++) worst = Math.Max(worst, Math.Abs(filled[i] - plane[i]));
        Assert.True(worst < 2.5, $"worst deviation {worst}");
        for (int i = 0; i < plane.Length; i++)
            if (weights[i] == 1) Assert.Equal(plane[i], filled[i]);
    }

    [Fact]
    public void PushPullHandlesOddSizesAndHolesAtTheBorder()
    {
        int w = 77, h = 51;
        float[] plane = Plane(w, h, 0.1f, 0.05f, -2);
        var weights = new float[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) weights[y * w + x] = x < 30 ? 0 : 1;

        float[] filled = MaskErasure.PushPull(plane, weights, w, h);

        Assert.All(filled, v => Assert.True(float.IsFinite(v)));
        // Extrapolation beyond the known side is flat-ish; it must stay within the known range, not drop to zero.
        float lo = plane.Where((_, i) => weights[i] > 0).Min(), hi = plane.Where((_, i) => weights[i] > 0).Max();
        Assert.All(filled, v => Assert.InRange(v, lo - 0.5f, hi + 0.5f));
    }

    [Theory]
    [InlineData(1.5f)]
    [InlineData(12f)]
    public void GaussianPreservesMassAndHasTheRequestedWidth(float sigma)
    {
        int w = 201;
        var impulse = new float[w * w]; impulse[100 * w + 100] = 1;

        float[] blurred = MaskErasure.Gaussian(impulse, w, w, sigma);

        double mass = blurred.Sum(v => (double)v);
        double variance = 0;
        for (int y = 0; y < w; y++)
            for (int x = 0; x < w; x++) variance += blurred[y * w + x] * (x - 100.0) * (x - 100.0);
        Assert.InRange(mass, 0.999, 1.001);
        Assert.InRange(Math.Sqrt(variance / mass), sigma * 0.95, sigma * 1.05);
    }

    [Fact]
    public void DistanceTransformIsEuclidean()
    {
        int w = 40, h = 30;
        var features = new bool[w * h]; features[10 * w + 5] = true; features[25 * w + 35] = true;

        float[] distance = MaskErasure.DistanceTo(features, w, h);

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double expected = Math.Min(Math.Sqrt((x - 5) * (x - 5) + (y - 10) * (y - 10)), Math.Sqrt((x - 35) * (x - 35) + (y - 25) * (y - 25)));
                Assert.Equal(expected, distance[y * w + x], 4);
            }
        Assert.All(MaskErasure.DistanceTo(new bool[w * h], w, h), d => Assert.True(float.IsPositiveInfinity(d)));
    }

    // 640 pixels at 6 Å, the size of a K3 or Falcon micrograph at Noise2Mic's sampling; noise is 1 SD per pixel.
    const int Large = 640;

    static (float[] Image, bool[] Truth) Micrograph(float carbonStep, float carbonDepth, float bump, int seed)
    {
        var random = new RandomNormal(seed);
        var image = new float[Large * Large]; var truth = new bool[Large * Large];
        for (int y = 0; y < Large; y++)
            for (int x = 0; x < Large; x++)
            {
                int i = y * Large + x;
                float ice = 0.01f * x + 0.008f * y + bump * MathF.Exp(-((x - 450) * (x - 450) + (y - 450) * (y - 450)) / (2f * 120 * 120));
                image[i] = ice + random.NextSingle(0, 1);
                if (carbonStep != 0 && InDisk(x, y, 320, carbonDepth - 600, 600)) { image[i] += carbonStep; truth[i] = true; }
            }
        return (image, truth);
    }

    static (double Recall, double Extra) Score(bool[] found, bool[] truth)
    {
        int objects = 0, hits = 0, extra = 0;
        for (int i = 0; i < truth.Length; i++)
        {
            if (truth[i]) { objects++; if (found[i]) hits++; }
            else if (found[i]) extra++;
        }
        return (objects > 0 ? hits / (double)objects : 1, extra / (double)truth.Length);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(2f)]
    public void CleanIceWithGradientsAndSmoothThicknessBumpsHasNoObjects(float bump)
    {
        var (image, _) = Micrograph(0, 0, bump, 7);
        bool[] found = MaskErasure.Detect(image, Large, Large, 6);
        Assert.Equal(0, found.Count(v => v));
    }

    [Theory]
    [InlineData(2f, 60f)]
    [InlineData(2f, 250f)]
    [InlineData(6f, 150f)]
    public void FindsAllOfACarbonEdgeOnAGradient(float step, float depth)
    {
        // Depths 60 to 250 pixels cover 5 to 35% of the image; local trends alone only find the edge of the big ones.
        var (image, truth) = Micrograph(step, depth, 0, 7);
        var (recall, extra) = Score(MaskErasure.Detect(image, Large, Large, 6), truth);
        Assert.True(recall > 0.99, $"recall {recall:P1}");
        Assert.True(extra < 0.02, $"extra {extra:P2} of the image");
    }

    [Fact]
    public void FindsCarbonNextToAThicknessBump()
    {
        var (image, truth) = Micrograph(2, 150, 2, 9);
        var (recall, extra) = Score(MaskErasure.Detect(image, Large, Large, 6), truth);
        Assert.True(recall > 0.99, $"recall {recall:P1}");
        Assert.True(extra < 0.03, $"extra {extra:P2} of the image");
    }

    [Fact]
    public void FindsSmallDarkObjects()
    {
        // A 10 nm gold bead and a 25 nm ice crystal.
        var (image, _) = Micrograph(0, 0, 0, 3);
        var truth = new bool[image.Length];
        for (int y = 0; y < Large; y++)
            for (int x = 0; x < Large; x++)
                if (InDisk(x, y, 200, 300, 8) || InDisk(x, y, 500, 120, 21)) { image[y * Large + x] -= 4; truth[y * Large + x] = true; }
        var (recall, extra) = Score(MaskErasure.Detect(image, Large, Large, 6), truth);
        Assert.True(recall > 0.99, $"recall {recall:P1}");
        Assert.True(extra < 0.01, $"extra {extra:P2} of the image");
    }

    [Fact]
    public void ErasingContinuesTheIceThroughTheObject()
    {
        float[] ice = Plane(N, N, 0.04f, 0.03f, 0).Zip(Noise(N * N, 1, 11), (a, b) => a + b).ToArray();
        float[] carbon = (float[])ice.Clone();
        var truth = new bool[N * N];
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                if (InDisk(x, y, 80, 80, 30)) { carbon[y * N + x] += 6; truth[y * N + x] = true; }

        float[] soft = MaskErasure.Feather(truth, N, N, 6);
        float[] erased = MaskErasure.Erase(carbon, N, N, soft, MaskErasure.WhiteNoise(carbon, N, N, soft, 3));

        // The erased object continues the ice: its mean follows the gradient and its spread matches the noise.
        var objectPixels = Enumerable.Range(0, N * N).Where(i => truth[i]).ToArray();
        double meanError = objectPixels.Average(i => (double)(erased[i] - ice[i]));
        double erasedSpread = Math.Sqrt(objectPixels.Average(i => Math.Pow(erased[i] - (0.04 * (i % N) + 0.03 * (i / N)), 2)));
        Assert.InRange(meanError, -0.3, 0.3);
        Assert.InRange(erasedSpread, 0.8, 1.25);
        // Pixels outside the feather are untouched.
        float[] distance = MaskErasure.DistanceTo(truth, N, N);
        for (int i = 0; i < N * N; i++)
            if (distance[i] > 6) Assert.Equal(carbon[i], erased[i]);
    }

    [Fact]
    public void EraseDirtFollowsTheGradientUnderTheMask()
    {
        // Warp's tilt-series erasure used one mean per masked component, so a mask on a slope left a step.
        int w = 300, h = 200;
        float[] slope = Plane(w, h, 0.05f, 0.02f, 3);
        float[] data = slope.Zip(Noise(w * h, 1, 13), (a, b) => a + b).ToArray();
        var mask = new float[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (InDisk(x, y, 150, 100, 40) || (x > 260 && y < 30)) { mask[y * w + x] = 1; data[y * w + x] = -50; }
        var image = new Warp.Image(data, new int3(w, h, 1));
        var maskImage = new Warp.Image(mask, new int3(w, h, 1));

        Warp.TiltSeries.EraseDirt(image, maskImage, noiseScale: 1);

        float[] erased = image.GetHost(Warp.Intent.Read)[0];
        var masked = Enumerable.Range(0, w * h).Where(i => mask[i] > 0).ToArray();
        Assert.InRange(masked.Average(i => (double)(erased[i] - slope[i])), -0.3, 0.3);
        Assert.InRange(Math.Sqrt(masked.Average(i => Math.Pow(erased[i] - slope[i], 2))), 0.8, 1.3);
        // The disk's left and right edges sit 4 units apart on the slope; a constant fill would be off by 2 at each.
        int left = 100 * w + 112, right = 100 * w + 188;
        Assert.InRange(erased.Skip(left - 2).Take(5).Average() - slope[left], -1.2, 1.2);
        Assert.InRange(erased.Skip(right - 2).Take(5).Average() - slope[right], -1.2, 1.2);
    }

    [TemplateMatchCudaFact]
    public void MatchedNoiseHasTheTextureOfTheIce()
    {
        lock (Warp.GPU.Sync)
        {
            Warp.GPU.SetDevice(0);
            // Ice whose noise is correlated over neighbouring pixels (lag-1 correlation e^(−1/4σ²) ≈ 0.89 for σ = 1.5).
            int w = 256, h = 192;
            float[] texture = MaskErasure.Gaussian(Noise(w * h, 1, 21), w, h, 1.5f);
            float[] image = Plane(w, h, 0.01f, 0.02f, 0).Zip(texture, (a, b) => a + b).ToArray();
            var truth = new bool[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) truth[y * w + x] = InDisk(x, y, 128, 96, 50);
            float[] soft = MaskErasure.Feather(truth, w, h, 6);

            float[] noise = MaskErasure.MatchedNoise(image, w, h, soft, 4);

            (double Variance, double Lag1) Statistics(float[] data, Func<int, bool> select)
            {
                double sum = 0, sq = 0, cross = 0; int count = 0, pairs = 0;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x + 1 < w; x++)
                    {
                        int i = y * w + x;
                        if (!select(i) || !select(i + 1)) continue;
                        sum += data[i]; sq += data[i] * (double)data[i]; count++;
                        cross += data[i] * (double)data[i + 1]; pairs++;
                    }
                double mean = sum / count, variance = sq / count - mean * mean;
                return (variance, (cross / pairs - mean * mean) / variance);
            }
            var ice = Statistics(texture, i => !truth[i]);
            var fill = Statistics(noise, i => truth[i]);
            Assert.InRange(fill.Variance / ice.Variance, 0.85, 1.15);
            Assert.InRange(fill.Lag1, ice.Lag1 - 0.05, ice.Lag1 + 0.05);
            for (int i = 0; i < w * h; i++) if (soft[i] == 0) Assert.Equal(0, noise[i]);
        }
    }
}
