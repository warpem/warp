using System;
using System.IO;
using Warp.Tools;

namespace Warp;

/// <summary>
/// How JunkNet's input is made from a micrograph at 6 Å/pixel (no deconvolution, no denoising), as it was trained:
/// each half-average normalized robustly on its own (central third, as Noise2Mic does), their mean stored as float16,
/// then normalized by the central half's median and 84.13th percentile and clipped to ±5.
/// </summary>
public static class JunkNetInput
{
    /// <summary>Clip after the final normalization.</summary>
    public const float Clip = 5;

    /// <summary>The input from a micrograph's half-averages, exactly as the training corpus made it.</summary>
    public static float[] FromHalves(float[] odd, float[] even, int width, int height) =>
        NormalizeCentral(MeanOfHalves(odd, even, width, height), width, height);

    /// <summary>The mean of the robustly normalized halves, stored as float16 as the corpus stored it.</summary>
    public static float[] MeanOfHalves(float[] odd, float[] even, int width, int height)
    {
        float[] a = NormalizedHalf(odd, width, height), b = NormalizedHalf(even, width, height);
        var mean = new float[a.Length];
        for (int i = 0; i < mean.Length; i++) mean[i] = (float)(Half)((a[i] + b[i]) / 2);
        return mean;
    }

    /// <summary>
    /// The input from a micrograph's average, for movies without half-averages: the same steps on the average alone.
    /// On the reference micrographs its masks match those from the halves to 99.97% IoU or better.
    /// </summary>
    public static float[] FromAverage(float[] average, int width, int height)
    {
        float[] normalized = NormalizedHalf(average, width, height);
        for (int i = 0; i < normalized.Length; i++) normalized[i] = (float)(Half)normalized[i];
        return NormalizeCentral(normalized, width, height);
    }

    /// <summary>Noise2Mic's robust normalization, after reflect padding to at least one patch per side as it pads.</summary>
    static float[] NormalizedHalf(float[] data, int width, int height)
    {
        int patch = Noise2MicTorch.PatchSize;
        int paddedWidth = Math.Max(width, patch), paddedHeight = Math.Max(height, patch);
        if (paddedWidth == width && paddedHeight == height) return Noise2MicPreprocessing.NormalizeRobustly(data, width, height);
        int padWidth = paddedWidth - width, padHeight = paddedHeight - height;
        float[] padded = Noise2MicPreprocessing.ReflectPad(data, width, height, padWidth / 2, padWidth - padWidth / 2, padHeight / 2, padHeight - padHeight / 2);
        return Noise2MicPreprocessing.Crop(Noise2MicPreprocessing.NormalizeRobustly(padded, paddedWidth, paddedHeight), paddedWidth, paddedHeight, width, height);
    }

    /// <summary>(x − median) / (P84.13 − median) over the central half in each dimension, clipped to ±5, in float32 as numpy computes it.</summary>
    public static float[] NormalizeCentral(float[] image, int width, int height)
    {
        int x0 = width / 4, x1 = width - width / 4, y0 = height / 4, y1 = height - height / 4;
        var central = new float[(x1 - x0) * (y1 - y0)];
        for (int y = y0, i = 0; y < y1; y++)
            for (int x = x0; x < x1; x++)
                central[i++] = image[y * width + x];
        float median = Noise2MicPreprocessing.Percentile(central, 50);
        float scale = (float)Math.Max((double)Noise2MicPreprocessing.Percentile(central, 84.13) - median, 1e-6);
        var result = new float[image.Length];
        for (int i = 0; i < result.Length; i++) result[i] = Math.Clamp((image[i] - median) / scale, -Clip, Clip);
        return result;
    }
}

/// <summary>
/// Junk segmentation of a movie's micrograph with <see cref="JunkNetTorch"/>. The mask goes where Warp keeps micrograph
/// masks (<see cref="Movie.MaskPath"/>), at <see cref="MaskPixelSize"/> like BoxNet's, and the junk percentage into the
/// movie's metadata (<see cref="Movie.MaskPercentage"/>).
/// </summary>
public static class MovieJunk
{
    /// <summary>Sampling of saved masks, Å/pixel (Warp's convention).</summary>
    public const float MaskPixelSize = 8;

    public readonly record struct Segmentation(bool[] Mask, int Width, int Height, double Percentage, bool FromHalves);

    /// <summary>Segments junk in the movie's exported average (its half-averages where they exist) and saves the mask.</summary>
    public static Segmentation Segment(string moviePath, float pixelSize, JunkNetTorch network)
    {
        if (!float.IsFinite(pixelSize) || pixelSize <= 0) throw new ArgumentOutOfRangeException(nameof(pixelSize));
        var movie = new Movie(moviePath);
        if (!File.Exists(movie.AveragePath)) throw new FileNotFoundException("Junk segmentation needs the aligned average.", movie.AveragePath);
        var timer = System.Diagnostics.Stopwatch.StartNew();
        bool halves = File.Exists(movie.AverageOddPath) && File.Exists(movie.AverageEvenPath);
        float[] input; int width, height;
        if (halves)
        {
            var odd = MovieDenoising.Resample(movie.AverageOddPath, pixelSize, JunkNetTorch.PixelSize);
            var even = MovieDenoising.Resample(movie.AverageEvenPath, pixelSize, JunkNetTorch.PixelSize);
            if (odd.Width != even.Width || odd.Height != even.Height) throw new InvalidDataException("Half-averages differ in size.");
            (width, height) = (odd.Width, odd.Height);
            input = JunkNetInput.FromHalves(odd.Data, even.Data, width, height);
        }
        else
        {
            var average = MovieDenoising.Resample(movie.AveragePath, pixelSize, JunkNetTorch.PixelSize);
            (width, height) = (average.Width, average.Height);
            input = JunkNetInput.FromAverage(average.Data, width, height);
        }
        double prepareSeconds = timer.Elapsed.TotalSeconds; timer.Restart();
        float[] logits = network.Predict(input, width, height);
        double networkSeconds = timer.Elapsed.TotalSeconds; timer.Restart();

        var mask = new bool[logits.Length];
        int junk = 0;
        for (int i = 0; i < mask.Length; i++) if (mask[i] = logits[i] > 0) junk++;
        double percentage = 100.0 * junk / mask.Length;
        WriteMask(movie.MaskPath, mask, width, height);
        movie.MaskPercentage = (decimal)Math.Round(percentage, 2);
        movie.SaveMeta();
        Console.WriteLine($"Junk segmentation: {percentage:F1}% junk ({(halves ? "half-averages" : "average")}), prepare {prepareSeconds:F3}s, network {networkSeconds:F3}s, mask {timer.Elapsed.TotalSeconds:F3}s");
        return new Segmentation(mask, width, height, percentage, halves);
    }

    /// <summary>Dimensions of a mask at <see cref="MaskPixelSize"/> for a width × height image at 6 Å/pixel, rounded to even.</summary>
    public static (int Width, int Height) MaskDims(int width, int height) =>
        (Math.Max(2, (int)Math.Round(width * JunkNetTorch.PixelSize / MaskPixelSize / 2) * 2),
         Math.Max(2, (int)Math.Round(height * JunkNetTorch.PixelSize / MaskPixelSize / 2) * 2));

    /// <summary>Writes a 6 Å/pixel mask at <see cref="MaskPixelSize"/> (area-weighted, then thresholded at one half), as BoxNet writes its masks.</summary>
    public static void WriteMask(string path, bool[] mask, int width, int height)
    {
        var (maskWidth, maskHeight) = MaskDims(width, height);
        float[] scaled = Resize(Array.ConvertAll(mask, v => v ? 1f : 0f), width, height, maskWidth, maskHeight);
        for (int i = 0; i < scaled.Length; i++) scaled[i] = scaled[i] >= 0.5f ? 1 : 0;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var image = new Image(scaled, new int3(maskWidth, maskHeight, 1));
        image.WriteTIFF(path, MaskPixelSize, typeof(float));
    }

    /// <summary>A saved mask, resampled onto a width × height grid covering the same field, true where junk is.</summary>
    public static bool[] ReadMask(string path, int width, int height)
    {
        float[] data; int maskWidth, maskHeight;
        using (var image = Image.FromFile(path))
        {
            maskWidth = image.Dims.X; maskHeight = image.Dims.Y;
            data = (float[])image.GetHost(Intent.Read)[0].Clone();
        }
        float[] resized = Resize(data, maskWidth, maskHeight, width, height);
        var mask = new bool[resized.Length];
        for (int i = 0; i < mask.Length; i++) mask[i] = resized[i] >= 0.5f;
        return mask;
    }

    /// <summary>
    /// Bilinear resampling between two grids covering the same field (pixel centres aligned as align_corners=false does);
    /// shrinking averages over the source pixels each target pixel covers instead, so thin junk is not skipped.
    /// </summary>
    public static float[] Resize(float[] data, int width, int height, int newWidth, int newHeight)
    {
        if (width == newWidth && height == newHeight) return (float[])data.Clone();
        var result = new float[newWidth * newHeight];
        double sx = (double)width / newWidth, sy = (double)height / newHeight;
        if (sx >= 1 && sy >= 1)
        {
            for (int y = 0; y < newHeight; y++)
            {
                int ya = (int)Math.Floor(y * sy), yb = Math.Max(ya + 1, Math.Min(height, (int)Math.Ceiling((y + 1) * sy)));
                for (int x = 0; x < newWidth; x++)
                {
                    int xa = (int)Math.Floor(x * sx), xb = Math.Max(xa + 1, Math.Min(width, (int)Math.Ceiling((x + 1) * sx)));
                    double sum = 0;
                    for (int yy = ya; yy < yb; yy++)
                        for (int xx = xa; xx < xb; xx++)
                            sum += data[yy * width + xx];
                    result[y * newWidth + x] = (float)(sum / ((yb - ya) * (xb - xa)));
                }
            }
            return result;
        }
        for (int y = 0; y < newHeight; y++)
        {
            double fy = Math.Clamp((y + 0.5) * sy - 0.5, 0, height - 1);
            int y0 = (int)fy, y1 = Math.Min(height - 1, y0 + 1); double ty = fy - y0;
            for (int x = 0; x < newWidth; x++)
            {
                double fx = Math.Clamp((x + 0.5) * sx - 0.5, 0, width - 1);
                int x0 = (int)fx, x1 = Math.Min(width - 1, x0 + 1); double tx = fx - x0;
                double top = data[y0 * width + x0] * (1 - tx) + data[y0 * width + x1] * tx;
                double bottom = data[y1 * width + x0] * (1 - tx) + data[y1 * width + x1] * tx;
                result[y * newWidth + x] = (float)(top * (1 - ty) + bottom * ty);
            }
        }
        return result;
    }

    /// <summary>
    /// Which particles stay clear of junk: those whose centre is at least <paramref name="radius"/> +
    /// <paramref name="distance"/> pixels from it (BoxNet's rule), with coordinates in pixels of a width × height
    /// micrograph and the mask covering the same field. Without a mask path every particle stays.
    /// </summary>
    public static bool[] ClearOfJunk(string maskPath, int width, int height, (float X, float Y)[] centres, float radius, float distance)
    {
        var keep = new bool[centres.Length];
        if (string.IsNullOrEmpty(maskPath)) { Array.Fill(keep, true); return keep; }
        if (!File.Exists(maskPath)) throw new FileNotFoundException("Culling picks near junk needs the junk mask; segment junk first.", maskPath);
        // On the mask's own grid: an exact distance transform there, in micrograph pixels per mask pixel.
        bool[] mask; int maskWidth, maskHeight;
        using (var image = Image.FromFile(maskPath))
        {
            maskWidth = image.Dims.X; maskHeight = image.Dims.Y;
            float[] data = image.GetHost(Intent.Read)[0];
            mask = Array.ConvertAll(data, v => v >= 0.5f);
        }
        float[] distances = MaskErasure.DistanceTo(mask, maskWidth, maskHeight);
        double sx = (double)maskWidth / width, sy = (double)maskHeight / height, pixel = (1 / sx + 1 / sy) / 2;
        for (int i = 0; i < centres.Length; i++)
        {
            int mx = Math.Clamp((int)Math.Floor(centres[i].X * sx), 0, maskWidth - 1), my = Math.Clamp((int)Math.Floor(centres[i].Y * sy), 0, maskHeight - 1);
            keep[i] = distances[my * maskWidth + mx] * pixel >= radius + distance;
        }
        return keep;
    }
}
