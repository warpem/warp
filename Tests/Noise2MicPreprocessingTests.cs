using System;
using System.Linq;
using Warp;
using Xunit;

namespace Tests;

// CPU halves of Noise2Mic's preprocessing (noise2mic/src/utils/preprocessing.py, image_utils.py, cryo_utils.py).
public class Noise2MicPreprocessingTests
{
    [Fact]
    public void ReflectPaddingMirrorsWithoutRepeatingTheEdgeLikeTorch()
    {
        // torch.nn.functional.pad(x, (2, 1, 1, 0), mode='reflect') on [[1,2,3],[4,5,6]]
        float[] padded = Noise2MicPreprocessing.ReflectPad([1, 2, 3, 4, 5, 6], 3, 2, 2, 1, 1, 0);
        float[] expected = [6, 5, 4, 5, 6, 5,
                            3, 2, 1, 2, 3, 2,
                            6, 5, 4, 5, 6, 5];
        Assert.Equal(expected, padded);
    }

    [Theory]
    [InlineData(700, 600, 512, 1050, 900)]
    [InlineData(300, 200, 512, 512, 512)]
    public void DeconvolutionCanvasIsOneAndAHalfTimesAndAtLeastAPatch(int width, int height, int patch, int paddedWidth, int paddedHeight)
    {
        var canvas = Noise2MicPreprocessing.DeconvolutionCanvas(width, height, patch);
        Assert.Equal((paddedWidth, paddedHeight), (canvas.Width, canvas.Height));
        Assert.Equal(canvas.Width - width, canvas.Left + canvas.Right);
        Assert.Equal((canvas.Width - width) / 2, canvas.Left);
        Assert.Equal((canvas.Height - height) / 2, canvas.Top);
        Assert.Equal(Math.Min(Math.Max(width, patch), canvas.Width), canvas.CropWidth);
    }

    [Fact]
    public void RobustNormalizationUsesCentralThirdMedianAndSixtyEighthPercentile()
    {
        // 6x6 image; the central third is rows/columns 2..3.
        var image = new float[36];
        for (int i = 0; i < 36; i++) image[i] = i;
        image[0] = 1000; // clipped at median + 4 scale
        float[] central = [14, 15, 20, 21];
        float median = 17.5f;
        var deviations = central.Select(v => Math.Abs(v - median)).OrderBy(v => v).ToArray(); // 2.5 2.5 3.5 3.5
        double position = 0.68 * (deviations.Length - 1), fraction = position - Math.Floor(position);
        float scale = (float)(deviations[(int)position] + fraction * (deviations[(int)position + 1] - deviations[(int)position]));
        float[] normalized = Noise2MicPreprocessing.NormalizeRobustly(image, 6, 6);
        Assert.Equal((21 - median) / scale, normalized[21], 5);
        Assert.Equal(4, normalized[0], 5);
        Assert.Equal(-4, normalized[1], 5); // 1 is below median - 4 scale and clips
    }

    [Fact]
    public void WienerFilterFollowsTheNoise2MicFormula()
    {
        var ctf = new CTF { PixelSize = 6, Defocus = 2, Voltage = 300, Cs = 2.7M, Amplitude = 0.07M };
        int width = 64, height = 48;
        float[] filter = Noise2MicPreprocessing.WienerFilter(width, height, ctf, 6, 0.9f, 0.1f, 600);
        Assert.Equal((width / 2 + 1) * height, filter.Length);
        Assert.Equal(0, filter[0], 6); // the high-pass removes DC

        int x = 10, y = 40; // y >= height / 2 is a negative frequency in torch.fft.fftfreq
        double qx = x / (double)(width * 6), qy = (y - height) / (double)(height * 6), q2 = qx * qx + qy * qy, q = Math.Sqrt(q2);
        double lambda = 12.2643247 / Math.Sqrt(300e3 * (1 + 300e3 * 0.978466e-6));
        double gamma = Math.PI * lambda * (-2e4) * q2 + 0.5 * Math.PI * 2.7e7 * Math.Pow(lambda, 3) * q2 * q2;
        double value = 0.07 * Math.Cos(gamma) - Math.Sqrt(1 - 0.07 * 0.07) * Math.Sin(gamma);
        double highpass = (1 - Math.Cos(Math.Min(q * 600, 1) * Math.PI)) * 0.5;
        double snr = Math.Exp(-q * 6 * 2 * 0.1 * 100) * Math.Pow(10, 3 * 0.9) * highpass;
        Assert.Equal(value / (value * value + 1 / Math.Max(snr, 1e-15)), filter[y * (width / 2 + 1) + x], 4);
    }

    [Fact]
    public void TilesCoverTheImageWithTorchPositions()
    {
        // stride = 448; ceil(max(1, 1000 - 512) / 448) + 1 = 3 positions: 0, 448, 488 (clamped).
        Assert.Equal([0, 448, 488], Noise2MicPreprocessing.TilePositions(1000, 512, 64));
        Assert.Equal([0, 0], Noise2MicPreprocessing.TilePositions(512, 512, 64));
    }

    [Fact]
    public void HannWindowIsSymmetricWithZeroEnds()
    {
        float[] window = Noise2MicPreprocessing.HannWindow(5);
        Assert.Equal([0, .5f, 1, .5f, 0], window.Select(v => (float)Math.Round(v, 6)).ToArray());
    }
}
