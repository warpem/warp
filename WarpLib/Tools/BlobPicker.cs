using System;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;

namespace Warp.Tools
{
    /// <summary>GPU disk/annulus picker for dark particles in a single micrograph.</summary>
    public static class BlobPicker
    {
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
        /// Returns (x, y, Pearson correlation, local power). Coordinates and radii are
        /// in input pixels, coordinates are zero-based. Power is variance in a circular
        /// support of radius 2*radius, in input intensity squared. Only positive local
        /// maxima survive Euclidean suppression; equal scores favor the earlier pixel.
        /// No correlation threshold is imposed beyond zero: callers can filter the STAR.
        /// </summary>
        public static float4[] Pick(Image micrograph, float radius, float innerRadius = 0, float? minimumSeparation = null)
        {
            float separation = Validate(radius, innerRadius, minimumSeparation);
            if (micrograph == null) throw new ArgumentNullException(nameof(micrograph));
            if (micrograph.IsFT || micrograph.IsComplex || micrograph.Dims.Z != 1 ||
                2 * Math.Ceiling(2 * radius) >= Math.Min(micrograph.Dims.X, micrograph.Dims.Y))
                throw new ArgumentException("Expected a real 2D micrograph with space for the full 2-radius background support.", nameof(micrograph));

            using Image correlation = new Image(micrograph.Dims);
            using Image power = new Image(micrograph.Dims);
            Check(GPU.BlobPickerScore(micrograph.GetDevice(Intent.Read), new int2(micrograph.Dims), radius, innerRadius,
                correlation.GetDevice(Intent.Write), power.GetDevice(Intent.Write)));
            IntPtr pointer = IntPtr.Zero;
            try
            {
                Check(GPU.BlobPickerPeaks(correlation.GetDevice(Intent.Read), power.GetDevice(Intent.Read),
                    new int2(micrograph.Dims), separation, out pointer, out int count));
                float[] packed = new float[checked(count * 4)];
                if (count > 0) Marshal.Copy(pointer, packed, 0, packed.Length);
                float4[] result = new float4[count];
                for (int i = 0; i < count; i++)
                    result[i] = new float4(packed[i * 4], packed[i * 4 + 1], packed[i * 4 + 2], packed[i * 4 + 3]);
                return result;
            }
            finally { if (pointer != IntPtr.Zero) CPU.HostFree(pointer); }
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
