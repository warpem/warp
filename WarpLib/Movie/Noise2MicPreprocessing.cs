using System;

namespace Warp;

/// <summary>
/// Noise2Mic's preprocessing, as used to train and run the denoiser behind the BoxNet picking corpus
/// (noise2mic/src/utils/preprocessing.py, image_utils.py and cryo_utils.py). Kept bit-for-bit where it can be,
/// including its quirks: a CTF from mean defocus only, and a symmetric Hann window whose zero ends zero the
/// outermost image pixels. Host-side pieces live here; <see cref="MovieDenoising"/> adds the FFTs.
/// </summary>
public static class Noise2MicPreprocessing
{
    /// <summary>The reflect-padded canvas the Wiener filter runs on, and the central crop kept afterwards.</summary>
    public readonly record struct Canvas(int Width, int Height, int Left, int Right, int Top, int Bottom, int CropWidth, int CropHeight);

    public static Canvas DeconvolutionCanvas(int width, int height, int patchSize)
    {
        int paddedWidth = Math.Max((int)Math.Ceiling(width * 1.5), patchSize), paddedHeight = Math.Max((int)Math.Ceiling(height * 1.5), patchSize);
        int padWidth = Math.Max(0, paddedWidth - width), padHeight = Math.Max(0, paddedHeight - height);
        int left = padWidth / 2, top = padHeight / 2;
        int width2 = width + padWidth, height2 = height + padHeight;
        return new Canvas(width2, height2, left, padWidth - left, top, padHeight - top,
                          Math.Min(Math.Max(width, patchSize), width2), Math.Min(Math.Max(height, patchSize), height2));
    }

    /// <summary>torch.nn.functional.pad(..., mode='reflect'): mirrored about the edge pixel, which is not repeated.</summary>
    public static float[] ReflectPad(float[] data, int width, int height, int left, int right, int top, int bottom)
    {
        if (left >= width || right >= width || top >= height || bottom >= height)
            throw new ArgumentException("Reflect padding must be smaller than the image.");
        int paddedWidth = width + left + right, paddedHeight = height + top + bottom;
        var padded = new float[paddedWidth * paddedHeight];
        for (int y = 0; y < paddedHeight; y++)
        {
            int sy = Reflect(y - top, height);
            for (int x = 0; x < paddedWidth; x++)
                padded[y * paddedWidth + x] = data[sy * width + Reflect(x - left, width)];
        }
        return padded;
    }

    static int Reflect(int i, int n) => i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i;

    public static float[] Crop(float[] data, int width, int height, int cropWidth, int cropHeight)
    {
        int left = (width - cropWidth) / 2, top = (height - cropHeight) / 2;
        var cropped = new float[cropWidth * cropHeight];
        for (int y = 0; y < cropHeight; y++)
            Array.Copy(data, (y + top) * width + left, cropped, y * cropWidth, cropWidth);
        return cropped;
    }

    /// <summary>
    /// The Wiener filter over an rfft2 grid ((width / 2 + 1) × height, x fastest), as deconvolve_ctf and simulate_ctf
    /// compute it: mean defocus, no astigmatism or phase shift.
    /// </summary>
    public static float[] WienerFilter(int width, int height, CTF ctf, float pixelSize, float strength, float falloff, float highpassAngstrom)
    {
        int halfWidth = width / 2 + 1;
        double voltage = (double)ctf.Voltage * 1000, lambda = 12.2643247 / Math.Sqrt(voltage * (1 + voltage * 0.978466e-6));
        double cs = (double)ctf.Cs * 1e7, defocus = -(double)ctf.Defocus * 1e4, amplitude = (double)ctf.Amplitude;
        double phaseContrast = Math.Sqrt(Math.Max(0, 1 - amplitude * amplitude));
        double cutoff = highpassAngstrom > 0 ? 1.0 / highpassAngstrom : 0, strengthTerm = Math.Pow(10, 3 * strength);
        var filter = new float[halfWidth * height];
        for (int y = 0; y < height; y++)
        {
            // torch.fft.fftfreq: the upper half are negative frequencies.
            double qy = (y < (height + 1) / 2 ? y : y - height) / (height * (double)pixelSize);
            for (int x = 0; x < halfWidth; x++)
            {
                double qx = x / (width * (double)pixelSize), q2 = qx * qx + qy * qy, q = Math.Sqrt(q2);
                double gamma = Math.PI * lambda * defocus * q2 + 0.5 * Math.PI * cs * lambda * lambda * lambda * q2 * q2;
                double value = amplitude * Math.Cos(gamma) - phaseContrast * Math.Sin(gamma);
                double highpass = cutoff > 0 ? (1 - Math.Cos(Math.Min(q / cutoff, 1) * Math.PI)) * 0.5 : 1;
                double snr = Math.Exp(-q * pixelSize * 2 * falloff * 100) * strengthTerm * highpass;
                filter[y * halfWidth + x] = (float)(value / (value * value + 1 / Math.Max(snr, 1e-15)));
            }
        }
        return filter;
    }

    /// <summary>
    /// normalize_image_robustly_central_stats: median and 68th-percentile absolute deviation of the central third,
    /// the whole image clipped to ±4 of those, then centered and scaled.
    /// </summary>
    public static float[] NormalizeRobustly(float[] data, int width, int height)
    {
        float[] central;
        if (width < 3 || height < 3) central = (float[])data.Clone();
        else
        {
            int x0 = width / 3, x1 = width - width / 3, y0 = height / 3, y1 = height - height / 3;
            central = new float[(x1 - x0) * (y1 - y0)];
            for (int y = y0, i = 0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                    central[i++] = data[y * width + x];
        }
        float median = Percentile(central, 50);
        var deviations = new float[central.Length];
        for (int i = 0; i < central.Length; i++) deviations[i] = Math.Abs(central[i] - median);
        float scale = Math.Max(Percentile(deviations, 68), 1e-6f);
        float low = median - 4 * scale, high = median + 4 * scale;
        var normalized = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
            normalized[i] = (Math.Clamp(data[i], low, high) - median) / scale;
        return normalized;
    }

    /// <summary>numpy's default ('linear') percentile.</summary>
    static float Percentile(float[] values, double percent)
    {
        var sorted = (float[])values.Clone();
        Array.Sort(sorted);
        double position = percent / 100 * (sorted.Length - 1);
        int lower = (int)Math.Floor(position), upper = Math.Min(sorted.Length - 1, lower + 1);
        return (float)(sorted[lower] + (position - lower) * (sorted[upper] - sorted[lower]));
    }

    /// <summary>Tile origins along one axis, as the inference loop places them.</summary>
    public static int[] TilePositions(int length, int patchSize, int overlap)
    {
        int stride = patchSize - overlap;
        int count = (int)Math.Ceiling(Math.Max(1, length - patchSize) / (double)stride) + 1;
        var positions = new int[count];
        for (int i = 0; i < count; i++) positions[i] = Math.Min(i * stride, length - patchSize);
        return positions;
    }

    /// <summary>torch.hann_window(n, periodic=False).</summary>
    public static float[] HannWindow(int n)
    {
        var window = new float[n];
        for (int i = 0; i < n; i++) window[i] = n == 1 ? 1f : (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (n - 1)));
        return window;
    }
}
