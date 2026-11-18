using System;

namespace Warp.Tools;

/// <summary>
/// Display-ready gray levels for a micrograph preview PNG. Box binning to fit the size, then the browser viewer's
/// particle contrast (mean ± 2.5 sd mapped to 0–255, measured on the central half without junk), so a preview looks like
/// the plane it stands in for. Rows stay in MRC order; <see cref="Image.WritePNG"/> flips them as the viewer does.
/// </summary>
public static class MicrographPreview
{
    /// <summary>Mask pixels this far around junk are left out of the contrast too: restored junk blends into the ice over 12 Å, and masks are 8 Å/pixel.</summary>
    public const int MaskMargin = 2;

    /// <param name="mask">Optional junk mask on any grid covering the same field (rows in MRC order, junk ≥ 0.5).</param>
    public static (float[] Gray, int Width, int Height) Render(float[] data, int width, int height, int maxSize, float[] mask = null, int maskWidth = 0, int maskHeight = 0)
    {
        if (width < 1 || height < 1 || data.Length < (long)width * height) throw new ArgumentException("Data does not match the dimensions.");
        if (maxSize < 1) throw new ArgumentOutOfRangeException(nameof(maxSize));
        int factor = Math.Max(1, (int)Math.Ceiling(Math.Max(width, height) / (double)maxSize));
        int w = Math.Max(1, width / factor), h = Math.Max(1, height / factor);
        var binned = new float[w * h];
        // Rows bin independently; each output pixel sums its block in the same order as a serial pass.
        System.Threading.Tasks.Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                double sum = 0; int count = 0;
                for (int yy = y * factor; yy < Math.Min(height, (y + 1) * factor); yy++)
                    for (int xx = x * factor; xx < Math.Min(width, (x + 1) * factor); xx++)
                    {
                        float v = data[yy * width + xx];
                        if (float.IsFinite(v)) { sum += v; count++; }
                    }
                binned[y * w + x] = count > 0 ? (float)(sum / count) : float.NaN;
            }
        });
        var (mean, sd) = ContrastStatistics(binned, w, h, mask == null ? null : Excluded(mask, maskWidth, maskHeight, w, h));
        double low = mean - 2.5 * sd, scale = sd > 0 ? 255 / (5 * sd) : 0;
        var gray = new float[binned.Length];
        for (int i = 0; i < gray.Length; i++)
            gray[i] = float.IsFinite(binned[i]) ? (float)Math.Clamp((binned[i] - low) * scale, 0, 255) : 0;
        return (gray, w, h);
    }

    /// <summary>
    /// <see cref="Render(float[], int, int, int, float[], int, int)"/> for a micrograph file, leaving out the junk in
    /// <paramref name="junkMaskPath"/> (a TIFF or MRC on any grid) if given. A mask that does not exist is no mask: a preview
    /// only shows the micrograph, so it never fails for want of one.
    /// </summary>
    public static (float[] Gray, int Width, int Height) RenderFile(string micrographPath, int maxSize, string junkMaskPath = null)
    {
        float[] mask = null; int maskWidth = 0, maskHeight = 0;
        if (!string.IsNullOrEmpty(junkMaskPath) && System.IO.File.Exists(junkMaskPath))
            using (var junk = Image.FromFile(junkMaskPath))
            {
                (maskWidth, maskHeight) = (junk.Dims.X, junk.Dims.Y);
                mask = (float[])junk.GetHost(Intent.Read)[0].Clone();
            }
        using var micrograph = Image.FromFile(micrographPath);
        return Render(micrograph.GetHost(Intent.Read)[0], micrograph.Dims.X, micrograph.Dims.Y, maxSize, mask, maskWidth, maskHeight);
    }

    /// <summary>
    /// Mean and standard deviation the display stretches by, for particles in ice: over the central half of the image in
    /// each dimension, leaving out <paramref name="excluded"/> pixels (junk). Too few such pixels (under 1% of that region)
    /// fall back to all pixels not excluded, then to the central half, then to the whole image; non-finite values never count.
    /// </summary>
    public static (double Mean, double Sd) ContrastStatistics(float[] data, int width, int height, bool[] excluded = null)
    {
        if (excluded != null && excluded.Length != data.Length) throw new ArgumentException("The exclusion does not match the image.", nameof(excluded));
        int x0 = width / 4, x1 = width - x0, y0 = height / 4, y1 = height - y0;
        int minimum = Math.Max(16, (int)Math.Ceiling((x1 - x0) * (long)(y1 - y0) / 100.0));
        bool Central(int i) { int x = i % width, y = i / width; return x >= x0 && x < x1 && y >= y0 && y < y1; }
        bool Clear(int i) => excluded == null || !excluded[i];
        Func<int, bool>[] candidates = excluded != null ? [i => Central(i) && Clear(i), Clear, Central, _ => true] : [Central, _ => true];
        (double Mean, double Sd, long Count) stats = default;
        foreach (var include in candidates)
        {
            double mean = 0, m2 = 0; long n = 0;
            for (int i = 0; i < data.Length; i++)
            {
                float v = data[i];
                if (!float.IsFinite(v) || !include(i)) continue;
                n++; double delta = v - mean; mean += delta / n; m2 += delta * (v - mean);
            }
            stats = (mean, n > 0 ? Math.Sqrt(m2 / n) : 0, n);
            if (n >= minimum) break;
        }
        return (stats.Mean, stats.Sd);
    }

    /// <summary>
    /// The pixels of a <paramref name="width"/> × <paramref name="height"/> image that a junk mask on another grid over the
    /// same field covers, grown by <see cref="MaskMargin"/> mask pixels (a disc); nearest mask pixel per image pixel.
    /// </summary>
    public static bool[] Excluded(float[] mask, int maskWidth, int maskHeight, int width, int height)
    {
        if (maskWidth < 1 || maskHeight < 1 || mask == null || mask.Length < (long)maskWidth * maskHeight) throw new ArgumentException("Mask does not match its dimensions.");
        var grown = new bool[maskWidth * maskHeight];
        for (int y = 0; y < maskHeight; y++)
            for (int x = 0; x < maskWidth; x++)
            {
                if (!(mask[y * maskWidth + x] >= 0.5f)) continue;
                for (int dy = -MaskMargin; dy <= MaskMargin; dy++)
                    for (int dx = -MaskMargin; dx <= MaskMargin; dx++)
                    {
                        int xx = x + dx, yy = y + dy;
                        if (dx * dx + dy * dy <= MaskMargin * MaskMargin && xx >= 0 && yy >= 0 && xx < maskWidth && yy < maskHeight) grown[yy * maskWidth + xx] = true;
                    }
            }
        var excluded = new bool[width * height];
        for (int y = 0; y < height; y++)
        {
            int row = Math.Min(maskHeight - 1, (int)Math.Floor((y + 0.5) * maskHeight / height)) * maskWidth;
            for (int x = 0; x < width; x++)
                excluded[y * width + x] = grown[row + Math.Min(maskWidth - 1, (int)Math.Floor((x + 0.5) * maskWidth / width))];
        }
        return excluded;
    }
}
