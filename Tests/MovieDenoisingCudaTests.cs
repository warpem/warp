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
            string root = Path.Combine(Path.GetTempPath(), "warp-live-denoise-" + Guid.NewGuid());
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
                MovieDenoising.Prepare(movie.Path, 5, .85f, .25f);
                Assert.True(File.Exists(movie.DenoiseTrainingOddPath));
                Assert.True(File.Exists(movie.DenoiseTrainingEvenPath));
                string model = Path.Combine(root, "models", "first.pt");
                MovieDenoising.Train([movie.Path], model, "", 2, 0);
                Assert.True(File.Exists(model));
                MovieDenoising.Apply(movie.Path, model, 0);
                using var enhanced = Image.FromFile(movie.AverageDenoisedPath);
                Assert.Equal(dimensions, enhanced.Dims);
                Assert.Equal(5f, enhanced.PixelSize);
                Assert.All(enhanced.GetHost(Intent.Read)[0], value => Assert.True(float.IsFinite(value)));
            }
            finally { Directory.Delete(root, true); }
        }
    }
}
