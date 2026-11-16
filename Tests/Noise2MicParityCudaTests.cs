using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Warp;
using Warp.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Tests;

/// <summary>Runs only where Tests/Noise2Mic/parity.py has written its references into WARP_NOISE2MIC_PARITY.</summary>
public sealed class Noise2MicParityFactAttribute : FactAttribute
{
    public Noise2MicParityFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("WARP_RUN_CUDA_TESTS") != "1" || !Directory.Exists(Environment.GetEnvironmentVariable("WARP_NOISE2MIC_PARITY") ?? ""))
            Skip = "Set WARP_RUN_CUDA_TESTS=1 and WARP_NOISE2MIC_PARITY to a folder of parity.py references.";
    }
}

/// <summary>Warp's denoiser against Noise2Mic, the PyTorch denoiser behind the BoxNet picking corpus.</summary>
public class Noise2MicParityCudaTests(ITestOutputHelper output)
{
    static string Root => Environment.GetEnvironmentVariable("WARP_NOISE2MIC_PARITY")!;
    static string Fixture(string name) => Path.Combine(Root, name);

    static float[] Load(string name, out int[] shape)
    {
        shape = JsonDocument.Parse(File.ReadAllText(Fixture(name) + ".json")).RootElement.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        return MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Fixture(name) + ".bin")).ToArray();
    }

    static void Save(string name, float[] data, params int[] shape)
    {
        File.WriteAllBytes(Fixture(name) + ".bin", MemoryMarshal.AsBytes(data.AsSpan()).ToArray());
        File.WriteAllText(Fixture(name) + ".json", JsonSerializer.Serialize(new { shape }));
    }

    static double Correlation(float[] a, float[] b)
    {
        double ma = a.Average(v => (double)v), mb = b.Average(v => (double)v), ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.Length; i++) { double x = a[i] - ma, y = b[i] - mb; ab += x * y; aa += x * x; bb += y * y; }
        return ab / Math.Sqrt(aa * bb);
    }

    static double RelativeRms(float[] actual, float[] expected)
    {
        double error = 0, scale = 0;
        for (int i = 0; i < actual.Length; i++) { error += Math.Pow(actual[i] - expected[i], 2); scale += Math.Pow(expected[i], 2); }
        return Math.Sqrt(error / scale);
    }

    static JsonElement Pair => JsonDocument.Parse(File.ReadAllText(Fixture("pair.json"))).RootElement;

    static (MovieDenoising.Prepared Odd, MovieDenoising.Prepared Even) PreparePair()
    {
        var pair = Pair;
        string odd = pair.GetProperty("odd").GetString()!, even = pair.GetProperty("even").GetString()!, xml = pair.GetProperty("xml").GetString()!;
        var ctf = new Movie(Path.ChangeExtension(xml, ".mrc")).CTF;
        float pixel = Warp.Headers.MapHeader.ReadFromFile(odd).PixelSize.X;
        return (MovieDenoising.Preprocess(odd, ctf, pixel, MovieDenoising.Sampling(pixel), .9f, .1f),
                MovieDenoising.Preprocess(even, ctf, pixel, MovieDenoising.Sampling(pixel), .9f, .1f));
    }

    [Noise2MicParityFact]
    public void NetworkMatchesPyTorchAndSavesWhatPyTorchLoads()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            float[] input = Load("network-input", out int[] shape), expected = Load("network-output", out _);
            using var network = new Noise2MicTorch(0);
            network.Load(Fixture("network-model.pt"));
            float[] actual = network.Predict(input, shape[0]);
            // The network predicts a residual on top of its input; compare that part.
            float[] residual = actual.Select((v, i) => v - input[i]).ToArray(), expectedResidual = expected.Select((v, i) => v - input[i]).ToArray();
            double error = RelativeRms(residual, expectedResidual);
            output.WriteLine($"Residual relative RMS error vs PyTorch: {error:E3}");
            Assert.True(error < 1e-2, $"Residual relative RMS error {error}");

            network.Save(Fixture("warp-model.pt"), .9f, .1f);
            Save("warp-output", actual, shape);
        }
    }

    [Noise2MicParityFact]
    public void PreprocessingMatchesNoise2Mic()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var (odd, even) = PreparePair();
            float[] expectedOdd = Load("preprocess-odd", out int[] shape), expectedEven = Load("preprocess-even", out _);
            Assert.Equal((shape[1], shape[0]), (odd.Width, odd.Height));
            double oddCorrelation = Correlation(odd.Data, expectedOdd), evenCorrelation = Correlation(even.Data, expectedEven);
            double maxError = odd.Data.Zip(expectedOdd, (a, b) => Math.Abs(a - b)).Max();
            output.WriteLine($"Odd r = {oddCorrelation:F7}, even r = {evenCorrelation:F7}, max |Δ| = {maxError:G4} (normalized units)");
            Assert.True(oddCorrelation > 0.9999 && evenCorrelation > 0.9999);
        }
    }

    [Noise2MicParityFact]
    public void PairedInferenceMatchesNoise2MicAndTheAverageStandsIn()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var pair = Pair;
            var (odd, even) = PreparePair();
            using var network = new Noise2MicTorch(0);
            network.Load(pair.GetProperty("model").GetString()!);
            float[] expected = Load("infer-output", out int[] shape);
            int width = shape[1], height = shape[0];

            float[] mean = odd.Data.Select((v, i) => (v + even.Data[i]) / 2).ToArray();
            float[] paired = null!;
            foreach (var precision in new[] { TorchSharp.NN.Precision.Float32, TorchSharp.NN.Precision.Float16 })
            {
                network.InferencePrecision = precision;
                var started = DateTime.UtcNow;
                paired = Noise2MicPreprocessing.Crop(MovieDenoising.Denoise(network, mean, odd.Width, odd.Height), odd.Width, odd.Height, width, height);
                double r = Correlation(paired, expected), error = RelativeRms(paired, expected);
                output.WriteLine($"Paired inference, {precision}: r = {r:F6}, relative RMS {error:E3}, {(DateTime.UtcNow - started).TotalMilliseconds:F0} ms");
                Assert.True(r > 0.999 && error < 0.02, $"{precision}: r = {r}, relative RMS {error}");
            }

            // Movies without halves: their average, deconvolved and normalized on its own, scaled as a mean of halves.
            string averagePath = Fixture("average.mrc");
            float[] oddRaw, evenRaw; int3 dims; float pixel;
            using (var image = Image.FromFile(pair.GetProperty("odd").GetString()!)) { oddRaw = (float[])image.GetHost(Intent.Read)[0].Clone(); dims = image.Dims; pixel = image.PixelSize; }
            using (var image = Image.FromFile(pair.GetProperty("even").GetString()!)) evenRaw = (float[])image.GetHost(Intent.Read)[0].Clone();
            using (var image = new Image(oddRaw.Select((v, i) => (v + evenRaw[i]) / 2).ToArray(), dims)) image.WriteMRC(averagePath, pixel, true);
            var ctf = new Movie(Path.ChangeExtension(pair.GetProperty("xml").GetString()!, ".mrc")).CTF;
            var average = MovieDenoising.Preprocess(averagePath, ctf, pixel, MovieDenoising.Sampling(pixel), .9f, .1f);
            float[] alone = Noise2MicPreprocessing.Crop(MovieDenoising.Denoise(network, average.Data, average.Width, average.Height), average.Width, average.Height, width, height);
            double averageR = Correlation(alone, expected), averageError = RelativeRms(alone, expected);
            output.WriteLine($"Average alone: r = {averageR:F6} against paired Noise2Mic, relative RMS {averageError:E3}");
            Assert.True(averageR > 0.999 && averageError < 0.1, $"Average alone r = {averageR}, relative RMS {averageError}");
        }
    }

    /// <summary>Peak GPU memory and time per fine-tuning step, by batch size and precision.</summary>
    [Noise2MicParityFact]
    public void FineTuningMemory()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            int size = Noise2MicTorch.PatchSize * Noise2MicTorch.PatchSize;
            var random = new Random(3);
            foreach (var precision in new[] { TorchSharp.NN.Precision.BFloat16, TorchSharp.NN.Precision.Float32 })
                foreach (int batch in new[] { 1, 2 })
                {
                    using var network = new Noise2MicTorch(0) { TrainingPrecision = precision };
                    network.Load(Fixture("network-model.pt"));
                    float[] source = Enumerable.Range(0, batch * size).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
                    float[] target = Enumerable.Range(0, batch * size).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
                    network.Train(source, target, batch, 1e-5, batch);
                    TorchSharp.NN.R2UNet2D.PeakMemory(0, true);
                    var started = DateTime.UtcNow;
                    for (int i = 0; i < 5; i++) network.Train(source, target, batch, 1e-5, batch);
                    double seconds = (DateTime.UtcNow - started).TotalSeconds / 5;
                    var (allocated, reserved) = TorchSharp.NN.R2UNet2D.PeakMemory(0, true);
                    output.WriteLine($"{precision}, batch {batch}: peak {allocated / 1e9:F1} GB allocated, {reserved / 1e9:F1} GB reserved, {seconds:F2} s/step");
                }
        }
    }

    /// <summary>
    /// Batch-1 fine-tuning of the corpus model at each rate in WARP_NOISE2MIC_FINETUNE_RATES, scored by Noise2Noise
    /// loss on held-out pairs (one half denoised, compared with the other): lower is a better denoiser.
    /// </summary>
    [Noise2MicParityFact]
    public void FineTuningTheCorpusModel()
    {
        string listing = Fixture("train.txt"), heldOut = Fixture("heldout.txt"), rates = Environment.GetEnvironmentVariable("WARP_NOISE2MIC_FINETUNE_RATES") ?? "";
        if (!File.Exists(listing) || !File.Exists(heldOut) || rates.Length == 0) return;
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            string corpus = Pair.GetProperty("model").GetString()!;
            var movies = PrepareTrainingPairs(listing);
            var validation = File.ReadAllLines(heldOut).Where(l => l.Trim().Length > 0).Select(line =>
            {
                string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                var ctf = new Movie(Path.ChangeExtension(parts[2], ".mrc")).CTF;
                float pixel = Warp.Headers.MapHeader.ReadFromFile(parts[0]).PixelSize.X;
                return (Odd: MovieDenoising.Preprocess(parts[0], ctf, pixel, MovieDenoising.Sampling(pixel), .9f, .1f),
                        Even: MovieDenoising.Preprocess(parts[1], ctf, pixel, MovieDenoising.Sampling(pixel), .9f, .1f));
            }).ToArray();
            double Score(string model)
            {
                using var network = new Noise2MicTorch(0);
                network.Load(model);
                double total = 0;
                foreach (var (odd, even) in validation)
                    foreach (var (input, target) in new[] { (odd, even), (even, odd) })
                    {
                        float[] denoised = Noise2MicPreprocessing.Crop(MovieDenoising.Denoise(network, input.Data, input.Width, input.Height), input.Width, input.Height, input.OriginalWidth, input.OriginalHeight);
                        float[] truth = Noise2MicPreprocessing.Crop(target.Data, target.Width, target.Height, target.OriginalWidth, target.OriginalHeight);
                        total += denoised.Zip(truth, (a, b) => (double)(a - b) * (a - b)).Average();
                    }
                return total / (validation.Length * 2);
            }
            output.WriteLine($"Corpus model: held-out Noise2Noise loss {Score(corpus):F5}");
            foreach (string text in rates.Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                double rate = double.Parse(text, System.Globalization.CultureInfo.InvariantCulture);
                string model = Fixture($"finetuned-{text}.pt");
                var started = DateTime.UtcNow;
                MovieDenoising.Train(movies, model, corpus, 8, 0, .9f, .1f, fineTuneLearningRate: rate);
                output.WriteLine($"Fine-tuned from {rate:G2}: held-out Noise2Noise loss {Score(model):F5} ({(DateTime.UtcNow - started).TotalMinutes:F1} min training)");
            }
        }
    }

    static (MovieDenoising.Prepared Odd, MovieDenoising.Prepared Even)[] PrepareTrainingPairs(string listing) =>
        File.ReadAllLines(listing).Where(l => l.Trim().Length > 0).Select(line =>
        {
            string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var ctf = new Movie(Path.ChangeExtension(parts[2], ".mrc")).CTF;
            float pixel = Warp.Headers.MapHeader.ReadFromFile(parts[0]).PixelSize.X;
            return (MovieDenoising.Preprocess(parts[0], ctf, pixel, MovieDenoising.Sampling(pixel), .9f, .1f),
                    MovieDenoising.Preprocess(parts[1], ctf, pixel, MovieDenoising.Sampling(pixel), .9f, .1f));
        }).ToArray();

    /// <summary>Warp's training recipe on the listed corpus pairs, for parity.py compare against PyTorch-trained models.</summary>
    [Noise2MicParityFact]
    public void TrainsOnCorpusPairs()
    {
        string listing = Fixture("train.txt");
        if (!File.Exists(listing)) return;
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var movies = PrepareTrainingPairs(listing);
            var started = DateTime.UtcNow;
            MovieDenoising.Train(movies, Fixture("warp-trained.pt"), "", 8, 0, .9f, .1f);
            output.WriteLine($"Trained on {movies.Length} pairs in {(DateTime.UtcNow - started).TotalMinutes:F1} min");
            Assert.True(File.Exists(Fixture("warp-trained.pt")));
        }
    }
}
