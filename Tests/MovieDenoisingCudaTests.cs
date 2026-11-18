using System;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class MovieDenoisingCudaTests
{
    [TemplateMatchCudaFact]
    public void PreparedHalvesTrainAndDenoiseWithoutChangingSampling()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            string root = Path.Combine(Path.GetTempPath(), "warp-denoise-" + Guid.NewGuid());
            Directory.CreateDirectory(root);
            try
            {
                var movie = new Movie(Path.Combine(root, "input.eer"))
                {
                    CTF = new CTF { PixelSize = 5, Defocus = 1.5m, Voltage = 300, Amplitude = .07m },
                    OptionsCTF = new ProcessingOptionsMovieCTF { PixelSize = 5 }
                };
                movie.SaveMeta();
                var rng = new Random(42);
                var dimensions = new int3(320, 320, 1);
                var a = new float[320 * 320]; var b = new float[a.Length]; var average = new float[a.Length];
                for (int y = 0; y < 320; y++) for (int x = 0; x < 320; x++)
                {
                    int i = y * 320 + x;
                    float signal = -3 * MathF.Exp(-((x - 160) * (x - 160) + (y - 160) * (y - 160)) / 150f);
                    a[i] = signal + (float)rng.NextDouble() - .5f;
                    b[i] = signal + (float)rng.NextDouble() - .5f;
                    average[i] = (a[i] + b[i]) / 2;
                }
                Directory.CreateDirectory(movie.AverageOddDir); Directory.CreateDirectory(movie.AverageEvenDir);
                using (var image = new Image(a, dimensions)) image.WriteMRC(movie.AverageOddPath, 5);
                using (var image = new Image(b, dimensions)) image.WriteMRC(movie.AverageEvenPath, 5);
                using (var image = new Image(average, dimensions)) image.WriteMRC(movie.AveragePath, 5);
                // A junk disc at 6 Å/pixel (266 pixels), which training and denoising erase and restore.
                var junk = new bool[266 * 266];
                for (int y = 0; y < 266; y++) for (int x = 0; x < 266; x++) junk[y * 266 + x] = (x - 60) * (x - 60) + (y - 60) * (y - 60) < 400;
                MovieJunk.WriteMask(movie.MaskPath, junk, 266, 266);
                // Training and denoising deconvolve the halves and average on the fly; nothing deconvolved is written.
                string model = Path.Combine(root, "models", "first.pt");
                MovieDenoising.TrainMovies([movie.Path], model, "", 2, 0, 5, .9f, .1f);
                Assert.True(File.Exists(model));
                using (var network = new Noise2MicTorch(0))
                {
                    network.Load(model);
                    MovieDenoising.Denoise(movie.Path, 5, .9f, .1f, network);
                }
                Assert.False(File.Exists(movie.DeconvolvedPath));
                Assert.False(File.Exists(movie.DenoiseTrainingOddPath));
                using var enhanced = Image.FromFile(movie.AverageDenoisedPath);
                // Noise2Mic works at 6 Å/pixel: 320 pixels at 5 Å become 266 (rounded to even).
                Assert.Equal(new int3(266, 266, 1), enhanced.Dims);
                Assert.Equal(6f, enhanced.PixelSize);
                Assert.All(enhanced.GetHost(Intent.Read)[0], value => Assert.True(float.IsFinite(value)));
            }
            finally { Directory.Delete(root, true); }
        }
    }
}
