using System;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class MovieExportParticlesCudaTests
{
    /// <summary>
    /// A resampled export writes the smaller box at the coarser pixel size, holding the full-size particles
    /// Fourier-cropped: the same particles, not a padded or rescaled copy of the original box.
    /// </summary>
    [TemplateMatchCudaFact]
    public void ResampledParticlesAreTheFullBoxFourierCropped()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            string root = Path.Combine(Path.GetTempPath(), "warp-export-" + Guid.NewGuid());
            Directory.CreateDirectory(root);
            try
            {
                const int size = 256, frames = 4, box = 64, resampled = 32;
                const float pixel = 2;
                float2[] centers = [new(64, 64), new(128, 160), new(190, 90)];
                var rng = new Random(7);
                float[][] stack = Helper.ArrayOfFunction(z =>
                {
                    var frame = new float[size * size];
                    for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
                    {
                        float signal = centers.Sum(c => MathF.Exp(-((x - c.X) * (x - c.X) + (y - c.Y) * (y - c.Y)) / 18f));
                        frame[y * size + x] = signal + .05f * ((float)rng.NextDouble() - .5f);
                    }
                    return frame;
                }, frames);
                var movie = new Movie(Path.Combine(root, "input.eer")) { CTF = new CTF { PixelSize = (decimal)pixel, Defocus = 1.5m, Voltage = 300, Amplitude = .07m } };
                using var original = new Image(stack, new int3(size, size, frames));
                float2[] positions = centers.Select(c => c * pixel).ToArray();

                ProcessingOptionsParticleExport Options(int resample, string suffix) => new()
                {
                    PixelSize = (decimal)pixel, BinTimes = 0, BoxSize = box, BoxSizeResample = resample, Diameter = 40,
                    Normalize = true, Invert = false, DoAverage = true, DosePerAngstromFrame = 0, Suffix = suffix,
                    Dimensions = new float3(size * pixel, size * pixel, frames)
                };
                movie.ExportParticles(original, positions, Options(-1, "_full"), releaseStack: false);
                movie.ExportParticles(original, positions, Options(resampled, "_half"), releaseStack: false);
                GlobalTasks.WaitAll();

                using var full = Image.FromFile(Path.Combine(movie.ParticlesDir, movie.RootName + "_full.mrcs"));
                using var half = Image.FromFile(Path.Combine(movie.ParticlesDir, movie.RootName + "_half.mrcs"));
                Assert.Equal(new int3(box, box, centers.Length), full.Dims);
                Assert.Equal(new int3(resampled, resampled, centers.Length), half.Dims);
                Assert.Equal(pixel * box / resampled, half.PixelSize, 3);

                using var fullFt = full.AsFFT();
                using var croppedFt = fullFt.AsPadded(new int2(resampled));
                using var cropped = croppedFt.AsIFFT(false, 0, true);
                float[][] expected = cropped.GetHost(Intent.Read), actual = half.GetHost(Intent.Read);
                for (int p = 0; p < centers.Length; p++)
                    Assert.True(Correlation(expected[p], actual[p]) > .98, $"Particle {p} differs from the cropped full-size particle.");
            }
            finally { Directory.Delete(root, true); }
        }
    }

    /// <summary>
    /// Particles from frames streamed in any order, from two threads with a lane each, are those from the loaded stack,
    /// whether that stack is on the host or the GPU, with more particles than the exporter extracts at once, skipped
    /// frames, exposure weighting and pre-flipped phases.
    /// </summary>
    [TemplateMatchCudaFact]
    public void StreamedFramesGiveTheParticlesOfTheLoadedStack()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            string root = Path.Combine(Path.GetTempPath(), "warp-export-" + Guid.NewGuid());
            Directory.CreateDirectory(root);
            try
            {
                const int size = 320, frames = 6, box = 24;
                const float pixel = 1.5f;
                var rng = new Random(3);
                float2[] centers = Helper.ArrayOfFunction(i => new float2(20 + (float)rng.NextDouble() * (size - 40), 20 + (float)rng.NextDouble() * (size - 40)), 300);
                float[][] stack = Helper.ArrayOfFunction(z => Helper.ArrayOfFunction(i => (float)rng.NextDouble(), size * size), frames);
                var movie = new Movie(Path.Combine(root, "input.tiff")) { CTF = new CTF { PixelSize = (decimal)pixel, Defocus = 1.5m, Voltage = 300, Amplitude = .07m } };
                movie.OptionsCTF = new ProcessingOptionsMovieCTF();
                float2[] positions = centers.Select(c => c * pixel).ToArray();
                int3 dims = new int3(size, size, frames);

                ProcessingOptionsParticleExport Options(string suffix) => new()
                {
                    PixelSize = (decimal)pixel, BinTimes = 0, BoxSize = box, BoxSizeResample = -1, Diameter = 20,
                    Normalize = true, Invert = true, DoAverage = true, DosePerAngstromFrame = 3, PreflipPhases = true,
                    SkipFirstN = 1, SkipLastN = 1, Suffix = suffix, Dimensions = new float3(size * pixel, size * pixel, frames)
                };

                using (var host = new Image(stack, dims))
                    movie.ExportParticles(host, positions, Options("_host"), releaseStack: false);

                using (var device = new Image(stack, dims))
                {
                    device.GetDevice(Intent.Read);
                    movie.ExportParticles(device, positions, Options("_device"), releaseStack: true);
                }

                using (var export = new MovieParticleExport(movie, Options("_streamed"), positions, dims, 2))
                {
                    var order = new[] { 4, 0, 2, 5, 1, 3 };
                    System.Threading.Tasks.Parallel.For(0, 2, lane =>
                    {
                        GPU.SetDevice(0);
                        IntPtr frame = GPU.MallocDevice(size * size);
                        for (int i = lane; i < order.Length; i += 2)
                        {
                            GPU.CopyHostToDevice(stack[order[i]], frame, size * size);
                            export.AddFrame(order[i], frame, lane);
                        }
                        GPU.CheckGPUExceptions();
                        GPU.FreeDevice(frame);
                    });
                    export.Finish();
                }
                GlobalTasks.WaitAll();

                Image Read(string suffix) => Image.FromFile(Path.Combine(movie.ParticlesDir, movie.RootName + suffix + ".mrcs"));
                using var fromHost = Read("_host");
                using var fromDevice = Read("_device");
                using var streamed = Read("_streamed");
                Assert.Equal(new int3(box, box, centers.Length), streamed.Dims);
                float[][] expected = fromHost.GetHost(Intent.Read);
                foreach (var actual in new[] { fromDevice.GetHost(Intent.Read), streamed.GetHost(Intent.Read) })
                    for (int p = 0; p < centers.Length; p++)
                        Assert.True(Correlation(expected[p], actual[p]) > .9999, $"Particle {p} differs.");
            }
            finally { Directory.Delete(root, true); }
        }
    }

    private static double Correlation(float[] a, float[] b)
    {
        double ma = a.Average(), mb = b.Average(), ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.Length; i++) { ab += (a[i] - ma) * (b[i] - mb); aa += (a[i] - ma) * (a[i] - ma); bb += (b[i] - mb) * (b[i] - mb); }
        return ab / Math.Sqrt(aa * bb);
    }
}
