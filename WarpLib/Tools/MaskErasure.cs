using System;
using System.Threading.Tasks;

namespace Warp.Tools;

/// <summary>
/// Erases masked regions of an image, e.g. high-contrast objects the fitted phase-contrast CTF does not describe (carbon,
/// thick ice, gold, as JunkNet segments them), so a Wiener deconvolution does not ring their sharp edges into halos
/// hundreds of ångströms wide.
/// <para>
/// Erasure: masked pixels continue the surrounding ice (<see cref="Continue"/>: a weighted plane plus push-pull of
/// the deviations from it), plus noise, blended in with a cosine edge.
/// </para>
/// Everything here is O(pixels) on the CPU except <see cref="MatchedNoise"/>, which needs two FFTs.
/// </summary>
public static class MaskErasure
{
    /// <summary>Gauss–Jordan elimination with partial pivoting on an augmented n × (n + 1) system.</summary>
    static bool Solve(double[,] system, double[] solution)
    {
        int n = solution.Length;
        for (int column = 0; column < n; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < n; row++) if (Math.Abs(system[row, column]) > Math.Abs(system[pivot, column])) pivot = row;
            if (Math.Abs(system[pivot, column]) < 1e-12) return false;
            for (int k = 0; k <= n; k++) (system[column, k], system[pivot, k]) = (system[pivot, k], system[column, k]);
            for (int row = 0; row < n; row++)
            {
                if (row == column) continue;
                double factor = system[row, column] / system[column, column];
                for (int k = column; k <= n; k++) system[row, k] -= factor * system[column, k];
            }
        }
        for (int i = 0; i < n; i++) solution[i] = system[i, n] / system[i, i];
        return true;
    }

    /// <summary>1 on the mask, falling to 0 over <paramref name="radius"/> pixels outside it along a cosine.</summary>
    public static float[] Feather(bool[] mask, int width, int height, float radius)
    {
        if (mask == null || mask.Length < width * height) throw new ArgumentException("Mask does not match the dimensions.");
        float[] soft = DistanceTo(mask, width, height);
        Parallel.For(0, height, y =>
        {
            for (int i = y * width; i < (y + 1) * width; i++)
            {
                float d = soft[i];
                soft[i] = d <= 0 ? 1 : d >= radius ? 0 : 0.5f + 0.5f * MathF.Cos(MathF.PI * d / radius);
            }
        });
        return soft;
    }

    /// <summary>
    /// <paramref name="image"/> with soft-masked pixels replaced by a push-pull continuation of the rest plus
    /// <paramref name="noise"/>; pixels with weight 0 are returned unchanged. A fully masked image is returned as is.
    /// </summary>
    public static float[] Erase(float[] image, int width, int height, float[] soft, float[] noise)
    {
        Check(image, width, height);
        int n = width * height;
        if (soft.Length < n || noise.Length < n) throw new ArgumentException("Mask or noise does not match the dimensions.");
        var known = new float[n];
        bool anyKnown = false, anyMasked = false;
        for (int i = 0; i < n; i++)
        {
            known[i] = soft[i] < 0.999f ? 1 - soft[i] : 0;
            anyKnown |= known[i] > 0; anyMasked |= soft[i] > 0;
        }
        var result = (float[])image.Clone();
        if (!anyKnown || !anyMasked) return result;
        float[] continuation = Continue(image, known, width, height);
        for (int i = 0; i < n; i++)
            if (soft[i] > 0) result[i] = (1 - soft[i]) * image[i] + soft[i] * (continuation[i] + noise[i]);
        return result;
    }

    /// <summary>
    /// Gaussian noise at the unmasked pixels' noise level times <paramref name="scale"/>, wherever
    /// <paramref name="soft"/> is non-zero. The level is the robust spread of each pixel's difference from the mean
    /// of its four neighbours (exact for white noise, blind to gradients), sampled from at most 100,000 pixels.
    /// </summary>
    public static float[] WhiteNoise(float[] image, int width, int height, float[] soft, int seed, float scale = 1)
    {
        Check(image, width, height);
        int n = width * height;
        int interior = Math.Max(0, width - 2) * Math.Max(0, height - 2);
        int stride = Math.Max(1, interior / 100000);
        var deviations = new float[interior / stride + 1];
        int count = 0, k = 0;
        for (int y = 1; y < height - 1; y++)
            for (int x = 1; x < width - 1; x++, k++)
            {
                if (k % stride != 0) continue;
                int i = y * width + x;
                if (soft[i] > 0 || soft[i - 1] > 0 || soft[i + 1] > 0 || soft[i - width] > 0 || soft[i + width] > 0) continue;
                deviations[count++] = image[i] - 0.25f * (image[i - 1] + image[i + 1] + image[i - width] + image[i + width]);
            }
        float sigma = 0;
        if (count > 0)
        {
            float median = Select(deviations, count, 0.5);
            for (int i = 0; i < count; i++) deviations[i] = Math.Abs(deviations[i] - median);
            // The difference from a 4-neighbour mean has 5/4 of the pixel variance.
            sigma = Select(deviations, count, 0.68) / MathF.Sqrt(1.25f) * scale;
        }
        var noise = new float[n];
        Parallel.For(0, height, y =>
        {
            RandomNormal random = null;   // one generator per row keeps the result independent of threading
            for (int i = y * width; i < (y + 1) * width; i++)
                if (soft[i] > 0) noise[i] = (random ??= new RandomNormal(unchecked(seed * 7919 + y))).NextSingle(0, sigma);
        });
        return noise;
    }

    /// <summary>
    /// Gaussian noise with the radially averaged power spectrum of the unmasked pixels (detrended, so the push-pull
    /// continuation carries the low frequencies), wherever <paramref name="soft"/> is non-zero. Unlike white noise,
    /// it deconvolves into the same texture as the ice around it. Needs the GPU for its FFTs.
    /// </summary>
    /// <param name="plans">Plans of the calling thread's own, if it runs beside other GPU work; else the shared cache's.</param>
    public static float[] MatchedNoise(float[] image, int width, int height, float[] soft, int seed, PrivateFftPlans plans = null)
    {
        Check(image, width, height);
        int n = width * height;
        var window = new float[n];
        double windowEnergy = 0;
        for (int i = 0; i < n; i++) { window[i] = 1 - Math.Clamp(soft[i], 0, 1); windowEnergy += window[i] * window[i]; }
        if (windowEnergy < 1) return new float[n];
        float[] local = NormalizedGaussian(image, window, width, height, 20);
        var windowed = new float[n];
        for (int i = 0; i < n; i++) windowed[i] = (image[i] - local[i]) * window[i];

        int halfWidth = width / 2 + 1, bins = (int)Math.Ceiling(Math.Sqrt(0.5) * Math.Max(width, height)) + 2;
        var radial = new double[bins]; var counts = new double[bins];
        int Bin(int x, int y)
        {
            double fx = (double)x / width, fy = (double)(y <= height / 2 ? y : y - height) / height;
            return Math.Min(bins - 1, (int)Math.Round(Math.Sqrt(fx * fx + fy * fy) * Math.Max(width, height)));
        }
        var dims = new int3(width, height, 1);
        using (var real = new Image(windowed, dims))
        using (var fft = real.AsFFT(false, plans?.Forward(dims) ?? 0))
        {
            float[] spectrum = fft.GetHost(Intent.Read)[0];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < halfWidth; x++)
                {
                    int i = y * halfWidth + x, b = Bin(x, y);
                    // Interior columns stand for their Hermitian twins too.
                    double multiplicity = x == 0 || (width % 2 == 0 && x == width / 2) ? 1 : 2;
                    radial[b] += multiplicity * (spectrum[i * 2] * (double)spectrum[i * 2] + spectrum[i * 2 + 1] * (double)spectrum[i * 2 + 1]);
                    counts[b] += multiplicity;
                }
        }
        for (int b = 0; b < bins; b++) radial[b] = counts[b] > 0 ? radial[b] / counts[b] / windowEnergy : 0;
        radial[0] = 0;

        var random = new RandomNormal(seed);
        var white = new float[n];
        for (int i = 0; i < n; i++) white[i] = random.NextSingle(0, 1);
        float[] shaped;
        using (var real = new Image(white, dims))
        using (var fft = real.AsFFT(false, plans?.Forward(dims) ?? 0))
        {
            float[] spectrum = fft.GetHost(Intent.ReadWrite)[0];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < halfWidth; x++)
                {
                    int i = y * halfWidth + x;
                    float amplitude = (float)Math.Sqrt(radial[Bin(x, y)]);
                    spectrum[i * 2] *= amplitude; spectrum[i * 2 + 1] *= amplitude;
                }
            using var back = fft.AsIFFT(false, plans?.Inverse(dims) ?? 0, true);
            shaped = (float[])back.GetHost(Intent.Read)[0].Clone();
        }
        for (int i = 0; i < n; i++) if (soft[i] <= 0) shaped[i] = 0;
        return shaped;
    }

    /// <summary>
    /// Push-pull inpainting: pixels with weight below 1 take a weighted pyramid average of their neighborhood at
    /// the coarsest scale that has data, so holes of any size fill smoothly and follow the gradients around them.
    /// </summary>
    public static float[] PushPull(float[] values, float[] weights, int width, int height)
    {
        Check(values, width, height);
        if (weights.Length < width * height) throw new ArgumentException("Weights do not match the dimensions.");
        var result = new float[width * height];
        PushPullLevel(values, weights, width, height, result);
        return result;
    }

    /// <summary>
    /// <see cref="PushPull"/> around the weighted plane of the known pixels. Push-pull alone fills a large hole with
    /// coarse averages of distant pixels, which across a gradient sit at the wrong level; the plane carries the
    /// gradient, and push-pull only the local deviations from it.
    /// </summary>
    public static float[] Continue(float[] values, float[] weights, int width, int height)
    {
        float[] plane = WeightedPlane(values, weights, width, height);
        var deviation = new float[width * height];
        for (int i = 0; i < deviation.Length; i++) deviation[i] = values[i] - plane[i];
        float[] result = PushPull(deviation, weights, width, height);
        for (int i = 0; i < result.Length; i++) result[i] = weights[i] >= 1 ? values[i] : result[i] + plane[i];
        return result;
    }

    /// <summary>Least-squares plane a + bx + cy through <paramref name="data"/> weighted by <paramref name="weights"/>.</summary>
    static float[] WeightedPlane(float[] data, float[] weights, int width, int height)
    {
        // A plane needs no more than a quarter million samples.
        int stride = Math.Max(1, (int)Math.Sqrt(width * (double)height / 250000));
        var normal = new double[3, 4];
        for (int y = stride / 2; y < height; y += stride)
        {
            double v = (y + 0.5) / height * 2 - 1;
            for (int x = stride / 2; x < width; x += stride)
            {
                double w = weights[y * width + x];
                if (w <= 0) continue;
                double u = (x + 0.5) / width * 2 - 1, z = data[y * width + x];
                normal[0, 0] += w; normal[0, 1] += w * u; normal[0, 2] += w * v; normal[0, 3] += w * z;
                normal[1, 1] += w * u * u; normal[1, 2] += w * u * v; normal[1, 3] += w * u * z;
                normal[2, 2] += w * v * v; normal[2, 3] += w * v * z;
            }
        }
        normal[1, 0] = normal[0, 1]; normal[2, 0] = normal[0, 2]; normal[2, 1] = normal[1, 2];
        var plane = new double[3];
        if (!Solve(normal, plane))
        {
            // Too few known pixels for a plane (or all on a line): their mean.
            plane[1] = plane[2] = 0;
            plane[0] = normal[0, 0] > 0 ? normal[0, 3] / normal[0, 0] : 0;
        }
        var surface = new float[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                surface[y * width + x] = (float)(plane[0] + plane[1] * ((x + 0.5) / width * 2 - 1) + plane[2] * ((y + 0.5) / height * 2 - 1));
        return surface;
    }

    static void PushPullLevel(float[] values, float[] weights, int width, int height, float[] output)
    {
        int n = width * height;
        bool complete = true;
        for (int i = 0; i < n && complete; i++) complete = weights[i] >= 0.999f;
        if (complete) { Array.Copy(values, output, n); return; }
        if (Math.Min(width, height) <= 2)
        {
            double sum = 0, total = 0;
            for (int i = 0; i < n; i++) { sum += values[i] * (double)weights[i]; total += weights[i]; }
            float mean = total > 0 ? (float)(sum / total) : 0;
            for (int i = 0; i < n; i++) output[i] = weights[i] * values[i] + (1 - weights[i]) * mean;
            return;
        }

        int cw = (width + 1) / 2, ch = (height + 1) / 2;
        var coarseValues = new float[cw * ch]; var coarseWeights = new float[cw * ch];
        Parallel.For(0, ch, cy =>
        {
            for (int cx = 0; cx < cw; cx++)
            {
                double sv = 0, sk = 0;
                for (int y = cy * 2; y < Math.Min(height, cy * 2 + 2); y++)
                    for (int x = cx * 2; x < Math.Min(width, cx * 2 + 2); x++)
                    {
                        float k = weights[y * width + x];
                        sv += values[y * width + x] * (double)k; sk += k;
                    }
                coarseValues[cy * cw + cx] = sk > 0 ? (float)(sv / sk) : 0;
                coarseWeights[cy * cw + cx] = (float)Math.Min(sk, 1);
            }
        });
        var coarse = new float[cw * ch];
        PushPullLevel(coarseValues, coarseWeights, cw, ch, coarse);

        Parallel.For(0, height, y =>
        {
            float sy = Math.Clamp((y + 0.5f) / 2 - 0.5f, 0, ch - 1);
            int y0 = (int)sy, y1 = Math.Min(ch - 1, y0 + 1); float fy = sy - y0;
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;
                float k = weights[i];
                if (k >= 1) { output[i] = values[i]; continue; }
                float sx = Math.Clamp((x + 0.5f) / 2 - 0.5f, 0, cw - 1);
                int x0 = (int)sx, x1 = Math.Min(cw - 1, x0 + 1); float fx = sx - x0;
                float up = (1 - fy) * ((1 - fx) * coarse[y0 * cw + x0] + fx * coarse[y0 * cw + x1]) +
                           fy * ((1 - fx) * coarse[y1 * cw + x0] + fx * coarse[y1 * cw + x1]);
                output[i] = k * values[i] + (1 - k) * up;
            }
        });
    }

    /// <summary>Gaussian blur with mirrored edges: exact up to σ = 3 pixels, three box passes above.</summary>
    public static float[] Gaussian(float[] data, int width, int height, float sigma)
    {
        Check(data, width, height);
        var result = (float[])data.Clone();
        if (!(sigma > 0)) return result;
        if (sigma <= 3)
        {
            int radius = (int)Math.Ceiling(4 * sigma);
            var kernel = new float[2 * radius + 1];
            double total = 0;
            for (int i = -radius; i <= radius; i++) total += kernel[i + radius] = MathF.Exp(-i * i / (2 * sigma * sigma));
            for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(kernel[i] / total);
            Separable(result, width, height, (line, length, buffer) => Convolve(line, length, kernel, radius, buffer));
        }
        else
        {
            foreach (int box in BoxWidths(sigma))
            {
                int radius = (box - 1) / 2;
                Separable(result, width, height, (line, length, buffer) => Box(line, length, radius, buffer));
            }
        }
        return result;
    }

    /// <summary>
    /// Gaussian average of <paramref name="data"/> over the pixels <paramref name="weights"/> selects, around their
    /// weighted plane, so a one-sided average next to the mask or the border does not lag behind a gradient.
    /// </summary>
    static float[] NormalizedGaussian(float[] data, float[] weights, int width, int height, float sigma)
    {
        int n = width * height;
        float[] plane = WeightedPlane(data, weights, width, height);
        var weighted = new float[n];
        for (int i = 0; i < n; i++) weighted[i] = (data[i] - plane[i]) * weights[i];
        float[] numerator = Gaussian(weighted, width, height, sigma), denominator = Gaussian(weights, width, height, sigma);
        for (int i = 0; i < n; i++) numerator[i] = numerator[i] / Math.Max(denominator[i], 1e-3f) + plane[i];
        return numerator;
    }

    static int[] BoxWidths(float sigma)
    {
        // Three boxes whose variances add up to σ² (Kovesi); widths are odd so each box stays centered.
        double ideal = Math.Sqrt(12 * sigma * sigma / 3 + 1);
        int lower = (int)Math.Floor(ideal); if (lower % 2 == 0) lower--;
        int upper = lower + 2;
        int m = (int)Math.Round((12 * sigma * sigma - 3 * lower * lower - 12 * lower - 9) / (-4.0 * lower - 4));
        return [m > 0 ? lower : upper, m > 1 ? lower : upper, m > 2 ? lower : upper];
    }

    delegate void LineFilter(float[] line, int length, float[] buffer);

    static void Separable(float[] data, int width, int height, LineFilter filter)
    {
        Parallel.For(0, height, () => (new float[width], new float[width]), (y, _, buffers) =>
        {
            Array.Copy(data, y * width, buffers.Item1, 0, width);
            filter(buffers.Item1, width, buffers.Item2);
            Array.Copy(buffers.Item2, 0, data, y * width, width);
            return buffers;
        }, _ => { });
        Parallel.For(0, width, () => (new float[height], new float[height]), (x, _, buffers) =>
        {
            for (int y = 0; y < height; y++) buffers.Item1[y] = data[y * width + x];
            filter(buffers.Item1, height, buffers.Item2);
            for (int y = 0; y < height; y++) data[y * width + x] = buffers.Item2[y];
            return buffers;
        }, _ => { });
    }

    static int Mirror(int i, int length)
    {
        // Inside the line (nearly every call from the filters below): itself, without the divisions.
        if ((uint)i < (uint)length) return i;
        // Half-sample symmetric (d c b a | a b c d | d c b a), any distance.
        int period = 2 * length;
        i %= period; if (i < 0) i += period;
        return i < length ? i : period - 1 - i;
    }

    static void Convolve(float[] line, int length, float[] kernel, int radius, float[] output)
    {
        for (int x = 0; x < length; x++)
        {
            float sum = 0;
            for (int k = -radius; k <= radius; k++) sum += kernel[k + radius] * line[Mirror(x + k, length)];
            output[x] = sum;
        }
    }

    static void Box(float[] line, int length, int radius, float[] output)
    {
        double sum = 0;
        for (int k = -radius; k <= radius; k++) sum += line[Mirror(k, length)];
        float scale = 1f / (2 * radius + 1);
        for (int x = 0; x < length; x++)
        {
            output[x] = (float)(sum * scale);
            sum += line[Mirror(x + radius + 1, length)] - line[Mirror(x - radius, length)];
        }
    }

    /// <summary>Euclidean distance from every pixel to the nearest true one (Felzenszwalb–Huttenlocher); ∞ if none.</summary>
    public static float[] DistanceTo(bool[] features, int width, int height)
    {
        if (features == null || features.Length < width * height) throw new ArgumentException("Mask does not match the dimensions.");
        const double far = 1e20;
        var squared = new double[width * height];
        for (int i = 0; i < squared.Length; i++) squared[i] = features[i] ? 0 : far;
        Parallel.For(0, width, () => new Workspace(height), (x, _, w) =>
        {
            for (int y = 0; y < height; y++) w.F[y] = squared[y * width + x];
            w.Transform(height);
            for (int y = 0; y < height; y++) squared[y * width + x] = w.D[y];
            return w;
        }, _ => { });
        Parallel.For(0, height, () => new Workspace(width), (y, _, w) =>
        {
            Array.Copy(squared, y * width, w.F, 0, width);
            w.Transform(width);
            Array.Copy(w.D, 0, squared, y * width, width);
            return w;
        }, _ => { });
        var distance = new float[squared.Length];
        for (int i = 0; i < distance.Length; i++) distance[i] = squared[i] >= far / 2 ? float.PositiveInfinity : (float)Math.Sqrt(squared[i]);
        return distance;
    }

    sealed class Workspace(int length)
    {
        public readonly double[] F = new double[length], D = new double[length];
        readonly int[] v = new int[length];
        readonly double[] z = new double[length + 1];

        public void Transform(int n)
        {
            int k = 0; v[0] = 0; z[0] = double.NegativeInfinity; z[1] = double.PositiveInfinity;
            for (int q = 1; q < n; q++)
            {
                double s = Intersection(q, v[k]);
                while (s <= z[k]) { k--; s = Intersection(q, v[k]); }   // z[0] = −∞ stops this at k = 0
                k++; v[k] = q; z[k] = s; z[k + 1] = double.PositiveInfinity;
            }
            k = 0;
            for (int q = 0; q < n; q++)
            {
                while (z[k + 1] < q) k++;
                double d = q - v[k];
                D[q] = d * d + F[v[k]];
            }
        }

        double Intersection(int q, int p) => ((F[q] + q * (double)q) - (F[p] + p * (double)p)) / (2.0 * q - 2.0 * p);
    }

    /// <summary>4-connected components of the true pixels: labels from 1 (0 elsewhere) and each label's size.</summary>
    public static int[] Label(bool[] mask, int width, int height, out int[] sizes)
    {
        int n = width * height;
        var labels = new int[n];
        var stack = new int[n];
        var counts = new System.Collections.Generic.List<int> { 0 };
        for (int start = 0; start < n; start++)
        {
            if (!mask[start] || labels[start] != 0) continue;
            int label = counts.Count, size = 0, top = 0;
            labels[start] = label; stack[top++] = start;
            while (top > 0)
            {
                int i = stack[--top]; size++;
                int x = i % width, y = i / width;
                if (x > 0 && mask[i - 1] && labels[i - 1] == 0) { labels[i - 1] = label; stack[top++] = i - 1; }
                if (x < width - 1 && mask[i + 1] && labels[i + 1] == 0) { labels[i + 1] = label; stack[top++] = i + 1; }
                if (y > 0 && mask[i - width] && labels[i - width] == 0) { labels[i - width] = label; stack[top++] = i - width; }
                if (y < height - 1 && mask[i + width] && labels[i + width] == 0) { labels[i + width] = label; stack[top++] = i + width; }
            }
            counts.Add(size);
        }
        sizes = counts.ToArray();
        return labels;
    }

    /// <summary>The <paramref name="quantile"/> of the first <paramref name="count"/> values (reordered in place).</summary>
    static float Select(float[] values, int count, double quantile)
    {
        int k = Math.Clamp((int)Math.Round(quantile * (count - 1)), 0, count - 1);
        int lo = 0, hi = count - 1;
        while (lo < hi)
        {
            float pivot = values[(lo + hi) >>> 1];
            int i = lo, j = hi;
            while (i <= j)
            {
                while (values[i] < pivot) i++;
                while (values[j] > pivot) j--;
                if (i <= j) { (values[i], values[j]) = (values[j], values[i]); i++; j--; }
            }
            if (k <= j) hi = j; else if (k >= i) lo = i; else break;
        }
        return values[k];
    }

    static void Check(float[] data, int width, int height)
    {
        if (width < 1 || height < 1 || data == null || data.Length < (long)width * height)
            throw new ArgumentException("Data does not match the dimensions.");
    }
}
