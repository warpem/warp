using System;
using System.Collections.Generic;
using System.IO;
using Warp.Headers;
using Warp.Tools;

namespace Warp;

/// <summary>
/// Frame-series enhancement for disk workers, following Noise2Mic (the denoiser behind the BoxNet picking corpus):
/// 6 Å/pixel, its Wiener deconvolution and robust normalization, its R2-U-Net and training recipe, and paired
/// inference on the mean of the normalized halves. No image or network lives in Relay.
/// </summary>
public static class MovieDenoising
{
    const int TileOverlap = 64, InferenceBatch = 16;
    const int TrainingBatch = 12, PatchesPerImage = 4, WarmupSteps = 50;
    const double LearningRateStart = 1e-4, LearningRateFinish = 1e-6;
    /// <summary>
    /// Fine-tuning an existing model takes one 512-pixel patch per step: GroupNorm needs no batch statistics, and a
    /// patch carries plenty of gradient at this rate. With bfloat16 it peaks at 9.5 GB (batch 2 needs 18.3 GB).
    /// Chosen on corpus data (100 pairs, 8 epochs, 9 held-out pairs): Noise2Noise loss fell steadily from 3e-6 to 1e-4,
    /// while outputs still matched the corpus model's at r ≥ 0.996, so they stay what BoxNet was trained on.
    /// </summary>
    public const double FineTuneLearningRate = 1e-4;
    public const int FineTuneBatch = 1;
    const float BlankThreshold = 1e-4f;

    public static float Sampling(float pixelSize) => Math.Max(Noise2MicTorch.PixelSize, pixelSize);

    /// <summary>A normalized, deconvolved image on the canvas Noise2Mic crops to (at least one patch per side).</summary>
    public readonly record struct Prepared(float[] Data, int Width, int Height, int OriginalWidth, int OriginalHeight);

    /// <summary>
    /// The deconvolved average (shown, and picked when denoising is off) at the micrograph's own size, and the
    /// deconvolved halves Noise2Mic trains on and denoises from, when the export wrote halves.
    /// </summary>
    public static void Prepare(string moviePath, float pixelSize, float strength, float falloff)
    {
        if (!float.IsFinite(pixelSize) || pixelSize <= 0 || !float.IsFinite(strength) || !float.IsFinite(falloff))
            throw new ArgumentOutOfRangeException(nameof(pixelSize));
        var movie = new Movie(moviePath);
        if (movie.OptionsCTF == null) throw new InvalidOperationException("CTF must be processed before enhancement.");
        float sampling = Sampling(pixelSize);

        var average = Preprocess(movie.AveragePath, movie.CTF, pixelSize, sampling, strength, falloff);
        Write(movie.DeconvolvedPath, Noise2MicPreprocessing.Crop(average.Data, average.Width, average.Height, average.OriginalWidth, average.OriginalHeight),
              average.OriginalWidth, average.OriginalHeight, sampling);
        if (File.Exists(movie.AverageOddPath) && File.Exists(movie.AverageEvenPath))
        {
            var odd = Preprocess(movie.AverageOddPath, movie.CTF, pixelSize, sampling, strength, falloff);
            var even = Preprocess(movie.AverageEvenPath, movie.CTF, pixelSize, sampling, strength, falloff);
            Write(movie.DenoiseTrainingOddPath, odd.Data, odd.Width, odd.Height, sampling);
            Write(movie.DenoiseTrainingEvenPath, even.Data, even.Width, even.Height, sampling);
        }
        else
        {
            File.Delete(movie.DenoiseTrainingOddPath);
            File.Delete(movie.DenoiseTrainingEvenPath);
        }
    }

    /// <summary>MicrographPreprocessor for one image: resampled to <paramref name="sampling"/>, deconvolved, normalized.</summary>
    public static Prepared Preprocess(string source, CTF ctf, float inputPixel, float sampling, float strength, float falloff)
    {
        float[] data; int width, height;
        using (var image = Image.FromFile(source))
        {
            if (image.Dims.Z != 1) throw new InvalidDataException("Expected one micrograph slice.");
            var dims = new int2(Math.Max(4, (int)Math.Round(image.Dims.X * inputPixel / sampling / 2) * 2),
                                Math.Max(4, (int)Math.Round(image.Dims.Y * inputPixel / sampling / 2) * 2));
            // Fourier cropping, as Warp's export bins halves; normalization below makes the scale irrelevant.
            using var fft = image.AsFFT();
            using var scaledFft = fft.AsPadded(dims);
            using var scaled = scaledFft.AsIFFT(false, 0, true);
            data = (float[])scaled.GetHost(Intent.Read)[0].Clone();
            width = dims.X; height = dims.Y;
        }
        foreach (float v in data) if (!float.IsFinite(v)) throw new InvalidDataException("Non-finite values in denoising input.");

        var canvas = Noise2MicPreprocessing.DeconvolutionCanvas(width, height, Noise2MicTorch.PatchSize);
        float[] padded = Noise2MicPreprocessing.ReflectPad(data, width, height, canvas.Left, canvas.Right, canvas.Top, canvas.Bottom);
        float[] filter = Noise2MicPreprocessing.WienerFilter(canvas.Width, canvas.Height, ctf, sampling, strength, falloff, Noise2MicTorch.DeconvolutionHighpass);
        float[] deconvolved;
        using (var image = new Image(padded, new int3(canvas.Width, canvas.Height, 1)))
        using (var fft = image.AsFFT())
        {
            float[] spectrum = fft.GetHost(Intent.ReadWrite)[0];
            for (int i = 0; i < filter.Length; i++) { spectrum[i * 2] *= filter[i]; spectrum[i * 2 + 1] *= filter[i]; }
            using var result = fft.AsIFFT(false, 0, true);
            deconvolved = (float[])result.GetHost(Intent.Read)[0].Clone();
        }
        float[] cropped = Noise2MicPreprocessing.Crop(deconvolved, canvas.Width, canvas.Height, canvas.CropWidth, canvas.CropHeight);
        return new Prepared(Noise2MicPreprocessing.NormalizeRobustly(cropped, canvas.CropWidth, canvas.CropHeight),
                            canvas.CropWidth, canvas.CropHeight, width, height);
    }

    static void Write(string path, float[] data, int width, int height, float sampling)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var image = new Image(data, new int3(width, height, 1));
        image.WriteMRC16b(path, sampling, true);
    }

    static (float[] Data, int Width, int Height) Read(string path)
    {
        using var image = Image.FromFile(path);
        return ((float[])image.GetHost(Intent.Read)[0].Clone(), image.Dims.X, image.Dims.Y);
    }

    /// <summary>
    /// Noise2Mic training: random 512-pixel patches of the prepared halves (4 per image per epoch), quarter-turn
    /// and mirror augmentation, either half as input, batches of 12, MSE, Adam from 1e-4 with 50 warm-up steps,
    /// then exponential decay to 1e-6 by the last epoch. A previous model (the corpus model, or the session's last
    /// one) is fine-tuned instead: one patch per step from <see cref="FineTuneLearningRate"/>, same warm-up and decay.
    /// </summary>
    public static void Train(string[] moviePaths, string outputModel, string previousModel, int epochs, int device,
                             float strength, float falloff, int microBatch = 4, int seed = 123, double? fineTuneLearningRate = null, int? fineTuneBatch = null)
    {
        if (moviePaths == null || moviePaths.Length == 0) throw new ArgumentException("No training micrographs.");
        if (epochs < 1) throw new ArgumentOutOfRangeException(nameof(epochs));
        using var network = new Noise2MicTorch(device);
        bool fineTune = !string.IsNullOrWhiteSpace(previousModel);
        if (fineTune) network.Load(previousModel);
        var random = new Random(seed);
        int patch = Noise2MicTorch.PatchSize, size = patch * patch, batch = fineTune ? fineTuneBatch ?? FineTuneBatch : TrainingBatch;
        double start = fineTune ? fineTuneLearningRate ?? FineTuneLearningRate : LearningRateStart;
        int samples = moviePaths.Length * PatchesPerImage, steps = (samples + batch - 1) / batch;
        double gamma = epochs > 1 ? Math.Pow(Math.Min(1, LearningRateFinish / start), 1.0 / (epochs - 1)) : 1;
        var source = new float[batch * size];
        var target = new float[batch * size];
        double learningRate = start;
        for (int epoch = 0, step = 0; epoch < epochs; epoch++)
        {
            int[] order = new int[samples];
            for (int i = 0; i < samples; i++) order[i] = i;
            random.Shuffle(order);
            for (int batchStart = 0; batchStart < samples; batchStart += batch, step++)
            {
                int count = Math.Min(batch, samples - batchStart);
                bool swap = random.Next(2) == 0;
                for (int b = 0; b < count; b++)
                {
                    var (odd, even) = SamplePatch(moviePaths, order[batchStart + b] / PatchesPerImage, random);
                    Array.Copy(swap ? even : odd, 0, source, b * size, size);
                    Array.Copy(swap ? odd : even, 0, target, b * size, size);
                }
                if (epoch == 0 && step < WarmupSteps) learningRate = start * (step + 1) / WarmupSteps;
                float loss = network.Train(source, target, count, learningRate, Math.Max(1, microBatch));
                if (!float.IsFinite(loss)) throw new InvalidOperationException("Denoiser loss became non-finite.");
                if (step % 25 == 0 || step == epochs * steps - 1)
                    Console.WriteLine($"Denoiser training step {step + 1}/{epochs * steps}, epoch {epoch + 1}/{epochs}, loss {loss:G5}, learning rate {learningRate:G3}");
            }
            // ExponentialLR scales whatever rate the epoch ended on, so a warm-up longer than the first epoch carries over.
            learningRate *= gamma;
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputModel))!);
        network.Save(outputModel, strength, falloff);
    }

    /// <summary>A random patch of one movie's prepared halves; blank patches are redrawn as Noise2Mic does.</summary>
    static (float[] Odd, float[] Even) SamplePatch(string[] moviePaths, int index, Random random)
    {
        int patch = Noise2MicTorch.PatchSize;
        for (int attempt = 0; ; attempt++)
        {
            var movie = new Movie(moviePaths[index]);
            var (oddData, width, height) = Read(movie.DenoiseTrainingOddPath);
            var (evenData, evenWidth, evenHeight) = Read(movie.DenoiseTrainingEvenPath);
            if (width != evenWidth || height != evenHeight || width < patch || height < patch)
                throw new InvalidDataException($"Training pair has incompatible dimensions: {movie.RootName}");
            int x0 = random.Next(width - patch + 1), y0 = random.Next(height - patch + 1);
            int turns = random.Next(4);
            bool flipX = random.Next(2) == 0, flipY = random.Next(2) == 0;
            var odd = Augment(oddData, width, x0, y0, patch, turns, flipX, flipY);
            var even = Augment(evenData, width, x0, y0, patch, turns, flipX, flipY);
            if (StandardDeviation(odd) >= BlankThreshold && StandardDeviation(even) >= BlankThreshold || attempt >= 10)
                return (odd, even);
            index = random.Next(moviePaths.Length);
        }
    }

    static float[] Augment(float[] data, int width, int x0, int y0, int patch, int turns, bool flipX, bool flipY)
    {
        var result = new float[patch * patch];
        for (int y = 0; y < patch; y++)
            for (int x = 0; x < patch; x++)
            {
                int sx = flipX ? patch - 1 - x : x, sy = flipY ? patch - 1 - y : y;
                for (int t = 0; t < turns; t++) (sx, sy) = (sy, patch - 1 - sx);
                result[y * patch + x] = data[(y0 + sy) * width + x0 + sx];
            }
        return result;
    }

    static float StandardDeviation(float[] values)
    {
        double sum = 0, square = 0;
        foreach (float v in values) { sum += v; square += (double)v * v; }
        double mean = sum / values.Length;
        return (float)Math.Sqrt(Math.Max(0, square / values.Length - mean * mean));
    }

    public static void Apply(string moviePath, string modelPath, int device)
    {
        using var network = new Noise2MicTorch(device);
        network.Load(modelPath);
        Apply(moviePath, network);
    }

    /// <summary>
    /// Paired inference on the mean of the normalized halves when the movie has them. Otherwise the deconvolved,
    /// normalized average stands in as it is: on corpus data it matches Noise2Mic's paired output to a 3% relative
    /// RMS (r = 0.9998), because the normalization scale comes from low-frequency signal, which halves and average
    /// share. Scaling it as a mean of two unit-noise halves (1/√2) was tested and is 27% off.
    /// </summary>
    public static void Apply(string moviePath, Noise2MicTorch network)
    {
        var movie = new Movie(moviePath);
        var header = MapHeader.ReadFromFile(movie.DeconvolvedPath);
        int originalWidth = header.Dimensions.X, originalHeight = header.Dimensions.Y;
        float sampling = header.PixelSize.X;
        float[] input; int width, height;
        if (File.Exists(movie.DenoiseTrainingOddPath) && File.Exists(movie.DenoiseTrainingEvenPath))
        {
            var (odd, oddWidth, oddHeight) = Read(movie.DenoiseTrainingOddPath);
            var (even, evenWidth, evenHeight) = Read(movie.DenoiseTrainingEvenPath);
            if (oddWidth != evenWidth || oddHeight != evenHeight) throw new InvalidDataException("Training halves differ in size.");
            input = new float[odd.Length];
            for (int i = 0; i < input.Length; i++) input[i] = (odd[i] + even[i]) / 2;
            (width, height) = (oddWidth, oddHeight);
        }
        else
        {
            var (average, averageWidth, averageHeight) = Read(movie.DeconvolvedPath);
            int patch = Noise2MicTorch.PatchSize;
            int padWidth = Math.Max(0, patch - averageWidth), padHeight = Math.Max(0, patch - averageHeight);
            input = Noise2MicPreprocessing.ReflectPad(average, averageWidth, averageHeight, padWidth / 2, padWidth - padWidth / 2, padHeight / 2, padHeight - padHeight / 2);
            (width, height) = (averageWidth + padWidth, averageHeight + padHeight);
        }
        foreach (float v in input) if (!float.IsFinite(v)) throw new InvalidDataException("Non-finite values in denoising input.");

        float[] denoised = Denoise(network, input, width, height);
        float[] output = Noise2MicPreprocessing.Crop(denoised, width, height, Math.Min(width, originalWidth), Math.Min(height, originalHeight));
        Directory.CreateDirectory(movie.AverageDenoisedDir);
        Write(movie.AverageDenoisedPath, output, Math.Min(width, originalWidth), Math.Min(height, originalHeight), sampling);
    }

    /// <summary>Tiled inference with Hann-weighted blending, as Noise2Mic's inference loop does it.</summary>
    public static float[] Denoise(Noise2MicTorch network, float[] image, int width, int height)
    {
        int patch = Noise2MicTorch.PatchSize, size = patch * patch;
        if (width < patch || height < patch) throw new ArgumentException("Denoising input must be at least one patch across.");
        int[] ys = Noise2MicPreprocessing.TilePositions(height, patch, TileOverlap), xs = Noise2MicPreprocessing.TilePositions(width, patch, TileOverlap);
        float[] window1D = Noise2MicPreprocessing.HannWindow(patch);
        var tiles = new List<(int X, int Y)>();
        foreach (int y in ys) foreach (int x in xs) tiles.Add((x, y));
        var sum = new float[width * height];
        var weight = new float[width * height];
        var batch = new float[InferenceBatch * size];
        for (int start = 0; start < tiles.Count; start += InferenceBatch)
        {
            int count = Math.Min(InferenceBatch, tiles.Count - start);
            for (int b = 0; b < count; b++)
            {
                var (x0, y0) = tiles[start + b];
                for (int y = 0; y < patch; y++)
                    Array.Copy(image, (y0 + y) * width + x0, batch, b * size + y * patch, patch);
            }
            float[] result = network.Predict(batch, count);
            for (int b = 0; b < count; b++)
            {
                var (x0, y0) = tiles[start + b];
                for (int y = 0; y < patch; y++)
                    for (int x = 0; x < patch; x++)
                    {
                        float w = window1D[y] * window1D[x];
                        int i = (y0 + y) * width + x0 + x;
                        sum[i] += result[b * size + y * patch + x] * w;
                        weight[i] += w;
                    }
            }
        }
        for (int i = 0; i < sum.Length; i++) sum[i] /= Math.Max(weight[i], 1e-6f);
        return sum;
    }
}
