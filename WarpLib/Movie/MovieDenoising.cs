using System;
using System.IO;
using System.Linq;
using Warp.Tools;
using ZLinq;

namespace Warp;

/// <summary>Shared frame-series enhancement used by disk workers. No image or network lives in Relay.</summary>
public static class MovieDenoising
{
    public static void Prepare(string moviePath, float pixelSize, float strength, float falloff)
    {
        if (!float.IsFinite(pixelSize) || pixelSize <= 0 || !float.IsFinite(strength) || !float.IsFinite(falloff))
            throw new ArgumentOutOfRangeException(nameof(pixelSize));
        var movie = new Movie(moviePath);
        if (movie.OptionsCTF == null) throw new InvalidOperationException("CTF must be processed before enhancement.");
        float sampling = Math.Max(NoiseNet2DTorch.PixelSize, pixelSize);
        PrepareImage(movie.AveragePath, movie.DeconvolvedPath, movie.CTF, pixelSize, sampling, strength, falloff);
        if (File.Exists(movie.AverageOddPath) && File.Exists(movie.AverageEvenPath))
        {
            PrepareImage(movie.AverageOddPath, movie.DenoiseTrainingOddPath, movie.CTF, pixelSize, sampling, strength, falloff);
            PrepareImage(movie.AverageEvenPath, movie.DenoiseTrainingEvenPath, movie.CTF, pixelSize, sampling, strength, falloff);
        }
        else
        {
            File.Delete(movie.DenoiseTrainingOddPath);
            File.Delete(movie.DenoiseTrainingEvenPath);
        }
    }

    static void PrepareImage(string source, string destination, CTF ctf, float inputPixel, float sampling, float strength, float falloff)
    {
        using var image = Image.FromFile(source);
        if (image.Dims.Z != 1) throw new InvalidDataException("Expected one micrograph slice.");
        var dims = new int2(Math.Max(4, (int)Math.Round(image.Dims.X * inputPixel / sampling / 2) * 2),
                            Math.Max(4, (int)Math.Round(image.Dims.Y * inputPixel / sampling / 2) * 2));
        using var fft = image.AsFFT();
        using var scaledFft = fft.AsPadded(dims);
        using var scaled = scaledFft.AsIFFT(false, 0, true);
        // Keep amplitudes independent of the Fourier crop size, including power scores from the picker.
        scaled.Multiply((float)dims.Elements() / (image.Dims.X * image.Dims.Y));
        using var padded = scaled.AsPaddedClamped(dims * 2);
        using var paddedFft = padded.AsFFT();
        var localCtf = ctf.GetCopy();
        localCtf.PixelSize = (decimal)sampling;
        GPU.DeconvolveCTF(paddedFft.GetDevice(Intent.Read), paddedFft.GetDevice(Intent.Write), paddedFft.Dims,
            localCtf.ToStruct(), strength, falloff, sampling * 2 / 500f);
        using var deconvolved = paddedFft.AsIFFT(false, 0, true);
        using var cropped = deconvolved.AsPadded(dims);
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        cropped.WriteMRC16b(destination, sampling, true);
    }

    /// <summary>Noise2Noise training with reusable crop buffers and at most one image pair resident.</summary>
    public static void Train(string[] moviePaths, string outputModel, string previousModel, int iterations, int device)
    {
        if (moviePaths == null || moviePaths.Length == 0) throw new ArgumentException("No training micrographs.");
        if (iterations < 1) throw new ArgumentOutOfRangeException(nameof(iterations));
        const int box = 256, batch = 8, stepsPerPair = 8;
        using var network = new NoiseNet2DTorch(new int2(box), new[] { device }, batch);
        if (!string.IsNullOrWhiteSpace(previousModel)) network.Load(previousModel);
        using var source = new Image(new int3(box, box, batch));
        using var target = new Image(new int3(box, box, batch));
        var random = new Random(123);
        int[] order = Enumerable.Range(0, moviePaths.Length).OrderBy(_ => random.Next()).ToArray();
        Image odd = null, even = null;
        try
        {
            for (int step = 0; step < iterations; step++)
            {
                if (step % stepsPerPair == 0)
                {
                    odd?.Dispose(); even?.Dispose(); odd = even = null;
                    var movie = new Movie(moviePaths[order[(step / stepsPerPair) % order.Length]]);
                    odd = Image.FromFile(movie.DenoiseTrainingOddPath);
                    even = Image.FromFile(movie.DenoiseTrainingEvenPath);
                    if (odd.Dims != even.Dims || odd.Dims.Z != 1 || Math.Min(odd.Dims.X, odd.Dims.Y) < box)
                        throw new InvalidDataException($"Training pair has incompatible dimensions: {movie.RootName}");
                    // The same scale on both halves avoids introducing a target-dependent intensity bias.
                    var stats = Moments(odd.GetHost(Intent.Read)[0]);
                    Normalize(odd.GetHost(Intent.ReadWrite)[0], stats.mean, stats.std);
                    Normalize(even.GetHost(Intent.ReadWrite)[0], stats.mean, stats.std);
                }
                var oddData = odd.GetHost(Intent.Read)[0];
                var evenData = even.GetHost(Intent.Read)[0];
                var sources = source.GetHost(Intent.Write);
                var targets = target.GetHost(Intent.Write);
                for (int b = 0; b < batch; b++)
                {
                    int x0 = random.Next(odd.Dims.X - box + 1), y0 = random.Next(odd.Dims.Y - box + 1);
                    int transform = random.Next(8);
                    bool swap = random.Next(2) == 0;
                    FillCrop(sources[b], swap ? evenData : oddData, odd.Dims.X, x0, y0, box, transform);
                    FillCrop(targets[b], swap ? oddData : evenData, odd.Dims.X, x0, y0, box, transform);
                }
                float rate = 0.0001f * (1 - .9f * step / iterations);
                network.Train(source, target, rate, out _, out var loss);
                if (!float.IsFinite(loss[0])) throw new InvalidOperationException("Denoiser loss became non-finite.");
                if (step % 25 == 0 || step == iterations - 1)
                    Console.WriteLine($"Denoiser training {step + 1}/{iterations}, loss {loss[0]:G5}");
            }
            network.Save(outputModel);
        }
        finally { odd?.Dispose(); even?.Dispose(); }
    }

    static void FillCrop(float[] output, float[] input, int width, int x0, int y0, int box, int transform)
    {
        for (int y = 0; y < box; y++)
            for (int x = 0; x < box; x++)
            {
                int xx = (transform & 1) != 0 ? box - 1 - x : x;
                int yy = (transform & 2) != 0 ? box - 1 - y : y;
                if ((transform & 4) != 0) (xx, yy) = (yy, xx);
                output[y * box + x] = input[(y0 + yy) * width + x0 + xx];
            }
    }

    public static void Apply(string moviePath, string modelPath, int device)
    {
        using var network = new NoiseNet2DTorch(new int2(256), new[] { device }, 4);
        network.Load(modelPath);
        Apply(moviePath, network);
    }

    public static void Apply(string moviePath, NoiseNet2DTorch network)
    {
        var movie = new Movie(moviePath);
        using var image = Image.FromFile(movie.DeconvolvedPath);
        if (image.Dims.Z != 1 || Math.Min(image.Dims.X, image.Dims.Y) < 256)
            throw new InvalidDataException("Denoising requires a micrograph at least 256 pixels across.");
        var stats = Moments(image.GetHost(Intent.Read)[0]);
        Normalize(image.GetHost(Intent.ReadWrite)[0], stats.mean, stats.std);
        NoiseNet2DTorch.Denoise(image, new[] { network });
        image.TransformValues(v => v * stats.std + stats.mean);
        Directory.CreateDirectory(movie.AverageDenoisedDir);
        image.WriteMRC16b(movie.AverageDenoisedPath, image.PixelSize, true);
    }

    static (float mean, float std) Moments(float[] values)
    {
        double sum = 0, square = 0;
        foreach (float value in values)
        {
            if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite values in denoising input.");
            sum += value; square += (double)value * value;
        }
        double mean = sum / values.Length;
        double std = Math.Sqrt(Math.Max(0, square / values.Length - mean * mean));
        if (std <= 1e-20) throw new InvalidDataException("Denoising input has no intensity variance.");
        return ((float)mean, (float)std);
    }

    static void Normalize(float[] values, float mean, float std)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i])) throw new InvalidDataException("Non-finite values in denoising input.");
            values[i] = Math.Clamp((values[i] - mean) / std, -5, 5);
        }
    }
}
