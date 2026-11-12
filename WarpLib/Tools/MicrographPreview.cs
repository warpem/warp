using System;

namespace Warp.Tools;

/// <summary>
/// Display-ready gray levels for a micrograph preview PNG. Box binning to fit the size, then the browser
/// viewer's own contrast (mean ± 2.5 sd mapped to 0–255), so a preview looks like the plane it stands in for.
/// Rows stay in MRC order; <see cref="Image.WritePNG"/> flips them as the viewer does.
/// </summary>
public static class MicrographPreview
{
    public static (float[] Gray, int Width, int Height) Render(float[] data, int width, int height, int maxSize)
    {
        if (width < 1 || height < 1 || data.Length < (long)width * height) throw new ArgumentException("Data does not match the dimensions.");
        if (maxSize < 1) throw new ArgumentOutOfRangeException(nameof(maxSize));
        int factor = Math.Max(1, (int)Math.Ceiling(Math.Max(width, height) / (double)maxSize));
        int w = Math.Max(1, width / factor), h = Math.Max(1, height / factor);
        var binned = new float[w * h];
        for (int y = 0; y < h; y++)
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
        double mean = 0, m2 = 0; long n = 0;
        foreach (float v in binned)
        {
            if (!float.IsFinite(v)) continue;
            n++; double delta = v - mean; mean += delta / n; m2 += delta * (v - mean);
        }
        double sd = n > 0 ? Math.Sqrt(m2 / n) : 0, low = mean - 2.5 * sd, scale = sd > 0 ? 255 / (5 * sd) : 0;
        var gray = new float[binned.Length];
        for (int i = 0; i < gray.Length; i++)
            gray[i] = float.IsFinite(binned[i]) ? (float)Math.Clamp((binned[i] - low) * scale, 0, 255) : 0;
        return (gray, w, h);
    }
}
