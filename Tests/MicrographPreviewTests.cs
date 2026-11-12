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
}
