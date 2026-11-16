using System;
using System.Globalization;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Warp.Tools
{
    /// <summary>GPU blob picker for dark particles in a single micrograph.</summary>
    public static class BlobPicker
    {
        /// <summary>
        /// Changes whenever picks for the same image change, so processing keyed on it is redone.
        /// 2: difference-of-Gaussians template, mirrored borders, peak-to-peak suppression.
        /// </summary>
        public const int Revision = 2;

        public static float Validate(float radius, float innerRadius = 0, float? minimumSeparation = null)
        {
            if (!float.IsFinite(radius) || radius <= 0)
                throw new ArgumentOutOfRangeException(nameof(radius), "Outer radius must be finite and positive (pixels).");
            if (!float.IsFinite(innerRadius) || innerRadius < 0 || innerRadius >= radius)
                throw new ArgumentOutOfRangeException(nameof(innerRadius), "Inner radius must be in [0, outer radius).");
            float separation = minimumSeparation ?? 2 * radius;
            if (!float.IsFinite(separation) || separation <= 0)
                throw new ArgumentOutOfRangeException(nameof(minimumSeparation), "Separation must be finite and positive (pixels).");
            return separation;
        }

        /// <summary>
        /// Returns (x, y, Pearson correlation, local power), strongest first. Coordinates and radii are in input pixels,
        /// coordinates are zero-based. The template is a difference of Gaussians (centre σ = radius/2, surround
        /// σ = radius) over a support of radius 2*radius; an annulus when innerRadius &gt; 0. Power is the variance in
        /// that support, in input intensity squared.
        /// <para>
        /// The image is mirrored past its edges so centres down to one radius from the border are scored; peaks are true
        /// local maxima (within radius/2), and suppression compares peaks with peaks, so a weaker particle more than
        /// <paramref name="minimumSeparation"/> from a stronger one survives even when the stronger one's slope
        /// outscores it. No correlation threshold is imposed beyond zero: callers can filter the STAR.
        /// </para>
        /// </summary>
        public static float4[] Pick(Image micrograph, float radius, float innerRadius = 0, float? minimumSeparation = null)
        {
            float separation = Validate(radius, innerRadius, minimumSeparation);
            if (micrograph == null) throw new ArgumentNullException(nameof(micrograph));
            if (micrograph.IsFT || micrograph.IsComplex || micrograph.Dims.Z != 1)
                throw new ArgumentException("Expected a real 2D micrograph.", nameof(micrograph));
            int width = micrograph.Dims.X, height = micrograph.Dims.Y;
            // The whole particle disk must be inside: closer to the border, picks are mostly dark edge bands that the
            // mirrored padding turns into blobs, and the particle's extraction box would leave the micrograph anyway.
            float edge = radius;
            if (width <= 2 * edge + 1 || height <= 2 * edge + 1)
                throw new ArgumentException("Micrograph is too small for the particle radius.", nameof(micrograph));

            int pad = (int)Math.Ceiling(2 * radius) + 1;
            float[] padded = MirrorPad(micrograph.GetHost(Intent.Read)[0], width, height, pad);
            var dims = new int2(width + 2 * pad, height + 2 * pad);
            using Image source = new Image(padded, new int3(dims.X, dims.Y, 1));
            using Image correlation = new Image(new int3(dims.X, dims.Y, 1));
            using Image power = new Image(new int3(dims.X, dims.Y, 1));
            Check(GPU.BlobPickerScore(source.GetDevice(Intent.Read), dims, radius, innerRadius,
                correlation.GetDevice(Intent.Write), power.GetDevice(Intent.Write)));
            IntPtr pointer = IntPtr.Zero;
            var candidates = new List<float4>();
            try
            {
                Check(GPU.BlobPickerPeaks(correlation.GetDevice(Intent.Read), power.GetDevice(Intent.Read),
                    dims, Math.Max(1, radius / 2), out pointer, out int count));
                float[] packed = new float[checked(count * 4)];
                if (count > 0) Marshal.Copy(pointer, packed, 0, packed.Length);
                for (int i = 0; i < count; i++)
                {
                    float x = packed[i * 4] - pad, y = packed[i * 4 + 1] - pad;
                    if (x >= edge && y >= edge && x <= width - 1 - edge && y <= height - 1 - edge)
                        candidates.Add(new float4(x, y, packed[i * 4 + 2], packed[i * 4 + 3]));
                }
            }
            finally { if (pointer != IntPtr.Zero) CPU.HostFree(pointer); }
            return Suppress(candidates, separation);
        }

        /// <summary>Greedy non-maximum suppression: strongest first, dropping peaks closer than separation to a kept one.</summary>
        public static float4[] Suppress(IReadOnlyList<float4> peaks, float separation)
        {
            // Stable sort: equal scores keep the native order (lower pixel ID first).
            var order = new int[peaks.Count];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            Array.Sort(order, (a, b) => { int c = peaks[b].Z.CompareTo(peaks[a].Z); return c != 0 ? c : a.CompareTo(b); });
            float cell = Math.Max(separation, 1e-3f), separation2 = separation * separation;
            var grid = new Dictionary<(int, int), List<float4>>();
            var kept = new List<float4>();
            foreach (int i in order)
            {
                float4 p = peaks[i];
                int cx = (int)MathF.Floor(p.X / cell), cy = (int)MathF.Floor(p.Y / cell);
                bool clear = true;
                for (int dy = -1; dy <= 1 && clear; dy++)
                    for (int dx = -1; dx <= 1 && clear; dx++)
                        if (grid.TryGetValue((cx + dx, cy + dy), out var near))
                            foreach (float4 q in near)
                                if ((p.X - q.X) * (p.X - q.X) + (p.Y - q.Y) * (p.Y - q.Y) < separation2) { clear = false; break; }
                if (!clear) continue;
                kept.Add(p);
                if (!grid.TryGetValue((cx, cy), out var list)) grid[(cx, cy)] = list = new List<float4>();
                list.Add(p);
            }
            return kept.ToArray();
        }

        /// <summary>Half-sample symmetric padding (d c b a | a b c d | d c b a) by <paramref name="pad"/> on every side.</summary>
        static float[] MirrorPad(float[] data, int width, int height, int pad)
        {
            int w = width + 2 * pad, h = height + 2 * pad;
            static int Mirror(int i, int n) { int period = 2 * n; i %= period; if (i < 0) i += period; return i < n ? i : period - 1 - i; }
            var result = new float[w * h];
            for (int y = 0; y < h; y++)
            {
                int sy = Mirror(y - pad, height);
                for (int x = 0; x < w; x++) result[y * w + x] = data[sy * width + Mirror(x - pad, width)];
            }
            return result;
        }

        /// <summary>Load a micrograph and always write matching/&lt;name&gt;_blob.star, including empty results.</summary>
        public static string Match(string micrographPath, float radius, float innerRadius = 0, float? minimumSeparation = null)
        {
            Validate(radius, innerRadius, minimumSeparation);
            using Image micrograph = Image.FromFile(micrographPath);
            float4[] picks = Pick(micrograph, radius, innerRadius, minimumSeparation);
            Movie movie = new Movie(micrographPath);
            Directory.CreateDirectory(movie.MatchingDir);
            string output = Path.Combine(movie.MatchingDir, movie.RootName + "_blob.star");
            WriteStar(output, movie.Name, picks);
            return output;
        }

        public static void WriteStar(string outputPath, string micrographName, float4[] picks)
        {
            Star table = new Star(new[] { "rlnCoordinateX", "rlnCoordinateY", "rlnMicrographName",
                "rlnAutopickFigureOfMerit", "wrpAutopickPower" });
            foreach (float4 pick in picks)
                table.AddRow(new[] { pick.X.ToString("G9", CultureInfo.InvariantCulture),
                    pick.Y.ToString("G9", CultureInfo.InvariantCulture), micrographName,
                    pick.Z.ToString("G9", CultureInfo.InvariantCulture), pick.W.ToString("G9", CultureInfo.InvariantCulture) });
            table.Save(outputPath);
        }

        private static void Check(int status)
        {
            if (status != 0) throw new InvalidOperationException($"GPU blob picker failed (status {status}; -1 invalid input, -2 allocation, -3 FFT, positive CUDA error).");
        }
    }
}
