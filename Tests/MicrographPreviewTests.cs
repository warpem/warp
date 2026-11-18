using System;
using System.Linq;
using Warp.Tools;
using Xunit;

namespace Tests;

public class MicrographPreviewTests
{
    [Fact]
    public void LargeMicrographsAreBoxBinnedToFitTheSize()
    {
        var data = Enumerable.Range(0, 4096 * 3000).Select(i => (float)(i % 4096)).ToArray();
        var preview = MicrographPreview.Render(data, 4096, 3000, 1024);
        Assert.Equal(1024, preview.Width);
        Assert.Equal(750, preview.Height);
        Assert.Equal(1024 * 750, preview.Gray.Length);
    }

    [Fact]
    public void SmallMicrographsKeepTheirSize()
    {
        var preview = MicrographPreview.Render(new float[300 * 200], 300, 200, 1024);
        Assert.Equal((300, 200), (preview.Width, preview.Height));
    }

    [Fact]
    public void ContrastMatchesTheBrowserViewer()
    {
        // The viewer maps mean - 2.5 sd to black and mean + 2.5 sd to white.
        float[] data = [-1, 1, -1, 1, 0, float.NaN];
        var preview = MicrographPreview.Render(data, 3, 2, 1024);
        double mean = 0, sd = Math.Sqrt(4.0 / 5);
        Assert.Equal((0 - (mean - 2.5 * sd)) * 255 / (5 * sd), preview.Gray[4], 3);
        Assert.Equal((1 - (mean - 2.5 * sd)) * 255 / (5 * sd), preview.Gray[1], 3);
        Assert.Equal(0, preview.Gray[5]);
    }

    [Fact]
    public void ConstantMicrographsRenderBlackLikeTheViewer()
    {
        Assert.All(MicrographPreview.Render(Enumerable.Repeat(3f, 16).ToArray(), 4, 4, 1024).Gray, v => Assert.Equal(0, v));
    }

    [Fact]
    public void ContrastComesFromTheCentralHalfOnly()
    {
        // 8 × 8: the central 4 × 4 alternates ±1, the border is far brighter.
        const int n = 8;
        var data = Enumerable.Repeat(1000f, n * n).ToArray();
        for (int y = 2; y < 6; y++) for (int x = 2; x < 6; x++) data[y * n + x] = (x + y) % 2 == 0 ? 1 : -1;
        var preview = MicrographPreview.Render(data, n, n, 1024);
        Assert.Equal((1 + 2.5) * 255 / 5, preview.Gray[2 * n + 2], 3);
        Assert.Equal(255, preview.Gray[0]);
    }

    [Fact]
    public void JunkAndItsMarginAreLeftOutOfTheContrast()
    {
        // 40 × 40, mask 10 × 10 over the same field: junk in mask columns 0–3, i.e. image columns 0–15.
        const int n = 40, m = 10;
        var mask = new float[m * m];
        for (int y = 0; y < m; y++) for (int x = 0; x < 4; x++) mask[y * m + x] = 1;
        bool[] excluded = MicrographPreview.Excluded(mask, m, m, n, n);
        // Grown by two mask pixels: columns up to 5 in the mask, 23 in the image.
        Assert.True(excluded[20 * n + 23]); Assert.False(excluded[20 * n + 24]);

        var random = new Random(1);
        var data = new float[n * n];
        for (int i = 0; i < data.Length; i++) data[i] = (i % n) < 16 ? -50 : (float)(random.NextDouble() * 2 - 1);
        var (mean, sd) = MicrographPreview.ContrastStatistics(data, n, n, excluded);
        var central = Enumerable.Range(0, n * n).Where(i => i % n >= 24 && i % n < 30 && i / n >= 10 && i / n < 30).Select(i => (double)data[i]).ToArray();
        Assert.Equal(central.Average(), mean, 6);
        Assert.Equal(Math.Sqrt(central.Select(v => (v - central.Average()) * (v - central.Average())).Average()), sd, 6);

        // When junk covers the whole centre, the ice elsewhere counts.
        var everywhere = Enumerable.Range(0, n * n).Select(i => i % n < 34).ToArray();
        var (aside, _) = MicrographPreview.ContrastStatistics(data, n, n, everywhere);
        Assert.Equal(Enumerable.Range(0, n * n).Where(i => i % n >= 34).Average(i => (double)data[i]), aside, 6);
    }
}
