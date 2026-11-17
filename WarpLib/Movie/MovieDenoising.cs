using System;
using System.Collections.Generic;
using System.IO;
using Warp.Headers;
using Warp.Tools;

namespace Warp;

/// <summary>
/// Frame-series enhancement for disk workers, following Noise2Mic (the denoiser behind the BoxNet picking corpus):
/// 6 Å/pixel, its Wiener deconvolution and robust normalization, its R2-U-Net and training recipe, and paired
/// inference on the mean of the normalized halves. As in Noise2Mic, deconvolution happens on the fly from the
/// exported averages and the movie's current CTF, for training and inference alike, so nothing deconvolved is
/// written and a new CTF only needs denoising again. No image or network lives in Relay.
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
    /// The part of preparing a movie that needs only the exported averages, not the CTF: the average and halves
    /// resampled, high-contrast objects found on the average and erased from all of them. A worker computes it while
    /// the CTF is fitted (<see cref="Erase"/>) and hands it to <see cref="Denoise(string, float, float, float, Noise2MicTorch, Erasure)"/>,
    /// which checks that it was made from the same files, unchanged since, or computes it again.
    /// </summary>
    public sealed class Erasure
    {
        internal string MoviePath;
        internal float PixelSize;
        internal (string Path, long Length, DateTime Written)[] Sources;
        internal float[] Average, Odd, Even, Objects;
        internal int Width, Height;
        /// <summary>Whether objects were looked for at all (<see cref="Objects"/> is also null when none were found).</summary>
        internal bool ObjectsErased;
        internal double ResampleSeconds, MaskSeconds, EraseSeconds, HalvesSeconds;
        internal bool Halves => Odd != null;
    }

    static (string Path, long Length, DateTime Written) Stamp(string path)
    {
        var info = new FileInfo(path);
        return (path, info.Length, info.LastWriteTimeUtc);
    }

    static bool IsCurrent(Erasure erasure, Movie movie, string moviePath, float pixelSize, bool eraseObjects)
    {
        if (erasure == null || erasure.MoviePath != moviePath || erasure.PixelSize != pixelSize || erasure.ObjectsErased != eraseObjects) return false;
        bool halves = File.Exists(movie.AverageOddPath) && File.Exists(movie.AverageEvenPath);
        if (halves != erasure.Halves) return false;
        foreach (var source in erasure.Sources)
            if (!File.Exists(source.Path) || Stamp(source.Path) != source) return false;
        return true;
    }

    /// <summary>
    /// High-contrast objects (carbon, ice crystals, gold) are found once, on the average, and erased from the
    /// average and both halves with independent noise, so Noise2Noise learns only their smooth continuation.
    /// <paramref name="plans"/>: the calling thread's own FFT plans, when it runs beside the worker's main thread.
    /// <paramref name="eraseObjects"/>: false only resamples, leaving every object in place.
    /// </summary>
    public static Erasure Erase(string moviePath, float pixelSize, PrivateFftPlans plans = null, bool eraseObjects = true)
    {
        if (!float.IsFinite(pixelSize) || pixelSize <= 0) throw new ArgumentOutOfRangeException(nameof(pixelSize));
        var movie = new Movie(moviePath);
        float sampling = Sampling(pixelSize);
        var result = new Erasure { MoviePath = moviePath, PixelSize = pixelSize, ObjectsErased = eraseObjects };
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var sources = new List<(string, long, DateTime)> { Stamp(movie.AveragePath) };
        var (averageData, width, height) = Resample(movie.AveragePath, pixelSize, sampling, plans);
        result.Width = width; result.Height = height;
        result.ResampleSeconds = timer.Elapsed.TotalSeconds; timer.Restart();
        result.Objects = eraseObjects ? ObjectMask(averageData, width, height, sampling) : null;
        result.MaskSeconds = timer.Elapsed.TotalSeconds; timer.Restart();
        result.Average = Erased(averageData, width, height, result.Objects, 1, plans);
        result.EraseSeconds = timer.Elapsed.TotalSeconds; timer.Restart();
        if (File.Exists(movie.AverageOddPath) && File.Exists(movie.AverageEvenPath))
        {
            sources.Add(Stamp(movie.AverageOddPath)); sources.Add(Stamp(movie.AverageEvenPath));
            result.Odd = ErasedHalf(movie.AverageOddPath, pixelSize, sampling, result.Objects, width, height, 2, plans);
            result.Even = ErasedHalf(movie.AverageEvenPath, pixelSize, sampling, result.Objects, width, height, 3, plans);
            result.HalvesSeconds = timer.Elapsed.TotalSeconds;
        }
        result.Sources = sources.ToArray();
        return result;
    }

    /// <summary>
    /// Denoises a movie's average, preparing it on the fly as Noise2Mic does: resampled to 6 Å, high-contrast objects
    /// erased, Wiener-deconvolved with the movie's current CTF and normalized. A movie with half-averages is denoised
    /// from the mean of its two prepared halves, as Noise2Mic infers; others from the prepared average. Writes only the
    /// denoised micrograph. <paramref name="erased"/>: the objects already erased from the same averages
    /// (<see cref="Erase"/>), if available. <paramref name="eraseObjects"/>: false denoises the averages as they are.
    /// </summary>
    /// <remarks>
    /// Without halves, the prepared average stands in as it is: on corpus data it matches Noise2Mic's paired output to a
    /// 3% relative RMS (r = 0.9998), because the normalization scale comes from low-frequency signal, which halves and
    /// average share. Scaling it as a mean of two unit-noise halves (1/√2) was tested and is 27% off.
    /// </remarks>
    public static void Denoise(string moviePath, float pixelSize, float strength, float falloff, Noise2MicTorch network, Erasure erased = null, bool eraseObjects = true)
    {
        if (!float.IsFinite(pixelSize) || pixelSize <= 0 || !float.IsFinite(strength) || !float.IsFinite(falloff))
            throw new ArgumentOutOfRangeException(nameof(pixelSize));
        var movie = new Movie(moviePath);
        if (movie.OptionsCTF == null) throw new InvalidOperationException("CTF must be processed before denoising.");
        float sampling = Sampling(pixelSize);
        bool reused = IsCurrent(erased, movie, moviePath, pixelSize, eraseObjects);
        if (!reused) erased = Erase(moviePath, pixelSize, eraseObjects: eraseObjects);
        var timer = System.Diagnostics.Stopwatch.StartNew();
        int originalWidth = erased.Width, originalHeight = erased.Height;
        float[] input; int width, height;
        if (erased.Halves)
        {
            var odd = Deconvolve(erased.Odd, erased.Width, erased.Height, movie.CTF, sampling, strength, falloff);
            var even = Deconvolve(erased.Even, erased.Width, erased.Height, movie.CTF, sampling, strength, falloff);
            input = new float[odd.Data.Length];
            for (int i = 0; i < input.Length; i++) input[i] = (odd.Data[i] + even.Data[i]) / 2;
            (width, height) = (odd.Width, odd.Height);
        }
        else
        {
            var average = Deconvolve(erased.Average, erased.Width, erased.Height, movie.CTF, sampling, strength, falloff);
            float[] cropped = Noise2MicPreprocessing.Crop(average.Data, average.Width, average.Height, originalWidth, originalHeight);
            int patch = Noise2MicTorch.PatchSize;
            int padWidth = Math.Max(0, patch - originalWidth), padHeight = Math.Max(0, patch - originalHeight);
            input = Noise2MicPreprocessing.ReflectPad(cropped, originalWidth, originalHeight, padWidth / 2, padWidth - padWidth / 2, padHeight / 2, padHeight - padHeight / 2);
            (width, height) = (originalWidth + padWidth, originalHeight + padHeight);
        }
        foreach (float v in input) if (!float.IsFinite(v)) throw new InvalidDataException("Non-finite values in denoising input.");
        double deconvolveSeconds = timer.Elapsed.TotalSeconds; timer.Restart();
        float[] denoised = Denoise(network, input, width, height);
        double denoiseSeconds = timer.Elapsed.TotalSeconds; timer.Restart();
        int outputWidth = Math.Min(width, originalWidth), outputHeight = Math.Min(height, originalHeight);
        Directory.CreateDirectory(movie.AverageDenoisedDir);
        Write(movie.AverageDenoisedPath, Noise2MicPreprocessing.Crop(denoised, width, height, outputWidth, outputHeight), outputWidth, outputHeight, sampling);
        Console.WriteLine($"Denoising{(reused ? " (objects erased while the CTF was fitted)" : "")}: resample {erased.ResampleSeconds:F3}s, " +
                          $"objects {erased.MaskSeconds:F3}s ({(!erased.ObjectsErased ? "not erased" : erased.Objects == null ? "none" : "erased")}), erase {erased.EraseSeconds + erased.HalvesSeconds:F3}s, " +
                          $"deconvolution {deconvolveSeconds:F3}s ({(erased.Halves ? "halves" : "average")}), network {denoiseSeconds:F3}s, output {timer.Elapsed.TotalSeconds:F3}s");
    }

    /// <summary>Feathered mask of the objects to erase before deconvolution at <paramref name="sampling"/>, or null.</summary>
    public static float[] ObjectMask(float[] data, int width, int height, float sampling)
    {
        bool[] objects = MaskErasure.Detect(data, width, height, sampling);
        if (!Array.Exists(objects, v => v)) return null;
        return MaskErasure.Feather(objects, width, height, EraseFeatherAngstrom / sampling);
    }

    /// <summary>
    /// Changes whenever preparing a movie for the denoiser (resampling, erasure, deconvolution) gives different input for
    /// the same files, so denoising keyed on it (Relay's denoising fingerprint) is redone.
    /// 2: high-contrast objects erased before deconvolution.
    /// </summary>
    public const int PreparationRevision = 2;

    /// <summary>Over this many ångströms erased objects blend into the ice around them.</summary>
    public const float EraseFeatherAngstrom = 60;

    static float[] Erased(float[] data, int width, int height, float[] objects, int seed, PrivateFftPlans plans = null) =>
        objects == null ? data : MaskErasure.Erase(data, width, height, objects, MaskErasure.MatchedNoise(data, width, height, objects, seed, plans));

    static float[] ErasedHalf(string source, float pixelSize, float sampling, float[] objects, int width, int height, int seed, PrivateFftPlans plans)
    {
        var (data, w, h) = Resample(source, pixelSize, sampling, plans);
        if (w != width || h != height) throw new InvalidDataException("Half-averages differ in size from the average.");
        return Erased(data, w, h, objects, seed, plans);
    }

    /// <summary>MicrographPreprocessor for one image: resampled to <paramref name="sampling"/>, deconvolved, normalized.</summary>
    public static Prepared Preprocess(string source, CTF ctf, float inputPixel, float sampling, float strength, float falloff)
    {
        var (data, width, height) = Resample(source, inputPixel, sampling);
        return Deconvolve(data, width, height, ctf, sampling, strength, falloff);
    }

    /// <summary>One micrograph slice Fourier-cropped to <paramref name="sampling"/>, as Warp's export bins halves.</summary>
    public static (float[] Data, int Width, int Height) Resample(string source, float inputPixel, float sampling, PrivateFftPlans plans = null)
    {
        float[] data; int width, height;
        using (var image = Image.FromFile(source))
        {
            if (image.Dims.Z != 1) throw new InvalidDataException("Expected one micrograph slice.");
            var dims = new int2(Math.Max(4, (int)Math.Round(image.Dims.X * inputPixel / sampling / 2) * 2),
                                Math.Max(4, (int)Math.Round(image.Dims.Y * inputPixel / sampling / 2) * 2));
            // Normalization after deconvolution makes the scale irrelevant.
            using var fft = image.AsFFT(false, plans?.Forward(image.Dims.Slice()) ?? 0);
            using var scaledFft = fft.AsPadded(dims);
            using var scaled = scaledFft.AsIFFT(false, plans?.Inverse(new int3(dims)) ?? 0, true);
            data = (float[])scaled.GetHost(Intent.Read)[0].Clone();
            width = dims.X; height = dims.Y;
        }
        foreach (float v in data) if (!float.IsFinite(v)) throw new InvalidDataException("Non-finite values in denoising input.");
        return (data, width, height);
    }

    /// <summary>The Wiener deconvolution and robust normalization of a resampled micrograph.</summary>
    /// <param name="plans">The calling thread's own FFT plans, when it runs beside other threads using the GPU.</param>
    public static Prepared Deconvolve(float[] data, int width, int height, CTF ctf, float sampling, float strength, float falloff, PrivateFftPlans plans = null)
    {
        var canvas = Noise2MicPreprocessing.DeconvolutionCanvas(width, height, Noise2MicTorch.PatchSize);
        float[] padded = Noise2MicPreprocessing.ReflectPad(data, width, height, canvas.Left, canvas.Right, canvas.Top, canvas.Bottom);
        float[] filter = Noise2MicPreprocessing.WienerFilter(canvas.Width, canvas.Height, ctf, sampling, strength, falloff, Noise2MicTorch.DeconvolutionHighpass);
        float[] deconvolved;
        var dims = new int3(canvas.Width, canvas.Height, 1);
        using (var image = new Image(padded, dims))
        using (var fft = image.AsFFT(false, plans?.Forward(dims) ?? 0))
        {
            float[] spectrum = fft.GetHost(Intent.ReadWrite)[0];
            for (int i = 0; i < filter.Length; i++) { spectrum[i * 2] *= filter[i]; spectrum[i * 2 + 1] *= filter[i]; }
            using var result = fft.AsIFFT(false, plans?.Inverse(dims) ?? 0, true);
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

    /// <summary>
    /// Trains on movies' half-averages, prepared on the fly (<see cref="PrepareHalves"/>) as Noise2Mic trains on them.
    /// The deconvolution settings are saved with the model.
    /// </summary>
    public static void TrainMovies(string[] moviePaths, string outputModel, string previousModel, int epochs, int device,
                                   float pixelSize, float strength, float falloff, int microBatch = 4, int seed = 123, bool eraseObjects = true)
    {
        if (moviePaths == null || moviePaths.Length == 0) throw new ArgumentException("No training micrographs.");
        // A missing pretrained model fails here, before the halves are prepared.
        string start = string.IsNullOrWhiteSpace(previousModel) ? "from scratch" : previousModel == Noise2MicTorch.PretrainedModel ? $"from the pretrained model ({Noise2MicTorch.ResolvePath(previousModel)})" : $"from {previousModel}";
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var pairs = PrepareHalves(moviePaths, pixelSize, strength, falloff, device, eraseObjects);
        Console.WriteLine($"Denoiser training: prepared {pairs.Length} half-average pairs in {timer.Elapsed.TotalSeconds:F1}s{(eraseObjects ? "" : ", objects not erased")}; training {start}");
        Train(pairs, outputModel, previousModel, epochs, device, strength, falloff, microBatch, seed);
    }

    /// <summary>
    /// Each movie's half-averages prepared for training: resampled, the objects found on the average erased with
    /// independent noise (unless <paramref name="eraseObjects"/> is false), deconvolved with the movie's current CTF and
    /// normalized. A few movies at a time, each thread with its own FFT plans.
    /// </summary>
    public static (Prepared Odd, Prepared Even)[] PrepareHalves(string[] moviePaths, float pixelSize, float strength, float falloff, int device, bool eraseObjects = true)
    {
        if (!float.IsFinite(pixelSize) || pixelSize <= 0 || !float.IsFinite(strength) || !float.IsFinite(falloff))
            throw new ArgumentOutOfRangeException(nameof(pixelSize));
        float sampling = Sampling(pixelSize);
        var pairs = new (Prepared Odd, Prepared Even)[moviePaths.Length];
        System.Threading.Tasks.Parallel.For(0, moviePaths.Length, new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 4 },
            () => { GPU.SetDevice(device); return new PrivateFftPlans(); },
            (i, _, plans) =>
            {
                var movie = new Movie(moviePaths[i]);
                if (movie.OptionsCTF == null) throw new InvalidOperationException($"Training movie has no CTF: {movie.RootName}");
                var erased = Erase(moviePaths[i], pixelSize, plans, eraseObjects);
                if (!erased.Halves) throw new InvalidDataException($"Training movie has no half-averages: {movie.RootName}");
                pairs[i] = (Deconvolve(erased.Odd, erased.Width, erased.Height, movie.CTF, sampling, strength, falloff, plans),
                            Deconvolve(erased.Even, erased.Width, erased.Height, movie.CTF, sampling, strength, falloff, plans));
                return plans;
            },
            plans => plans.Dispose());
        return pairs;
    }

    /// <summary>
    /// Noise2Mic training: random 512-pixel patches of the prepared halves (4 per image per epoch), quarter-turn
    /// and mirror augmentation, either half as input, batches of 12, MSE, Adam from 1e-4 with 50 warm-up steps,
    /// then exponential decay to 1e-6 by the last epoch. A previous model (the corpus model, or the session's last
    /// one) is fine-tuned instead: one patch per step from <see cref="FineTuneLearningRate"/>, same warm-up and decay.
    /// </summary>
    public static void Train((Prepared Odd, Prepared Even)[] pairs, string outputModel, string previousModel, int epochs, int device,
                             float strength, float falloff, int microBatch = 4, int seed = 123, double? fineTuneLearningRate = null, int? fineTuneBatch = null)
    {
        if (pairs == null || pairs.Length == 0) throw new ArgumentException("No training micrographs.");
        if (epochs < 1) throw new ArgumentOutOfRangeException(nameof(epochs));
        using var network = new Noise2MicTorch(device);
        bool fineTune = !string.IsNullOrWhiteSpace(previousModel);
        if (fineTune) network.Load(previousModel);
        var random = new Random(seed);
        int patch = Noise2MicTorch.PatchSize, size = patch * patch, batch = fineTune ? fineTuneBatch ?? FineTuneBatch : TrainingBatch;
        double start = fineTune ? fineTuneLearningRate ?? FineTuneLearningRate : LearningRateStart;
        int samples = pairs.Length * PatchesPerImage, steps = (samples + batch - 1) / batch;
        double gamma = epochs > 1 ? Math.Pow(Math.Min(1, LearningRateFinish / start), 1.0 / (epochs - 1)) : 1;
        var source = new float[batch * size];
        var target = new float[batch * size];
        double learningRate = start;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputModel))!);
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
                    var (odd, even) = SamplePatch(pairs, order[batchStart + b] / PatchesPerImage, random);
                    Array.Copy(swap ? even : odd, 0, source, b * size, size);
                    Array.Copy(swap ? odd : even, 0, target, b * size, size);
                }
                if (epoch == 0 && step < WarmupSteps) learningRate = start * (step + 1) / WarmupSteps;
                float loss = network.Train(source, target, count, learningRate, Math.Max(1, microBatch));
                if (!float.IsFinite(loss)) throw new InvalidOperationException("Denoiser loss became non-finite.");
                if (step % 25 == 0 || step == epochs * steps - 1)
                {
                    Console.WriteLine($"Denoiser training step {step + 1}/{epochs * steps}, epoch {epoch + 1}/{epochs}, loss {loss:G5}, learning rate {learningRate:G3}");
                    try { new TrainingProgress(step + 1, epochs * steps, epoch + 1, epochs, loss).Write(TrainingProgress.PathFor(outputModel)); }
                    catch (IOException) { } // Progress is advisory; training goes on without it.
                }
            }
            // ExponentialLR scales whatever rate the epoch ended on, so a warm-up longer than the first epoch carries over.
            learningRate *= gamma;
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputModel))!);
        network.Save(outputModel, strength, falloff);
        File.Delete(TrainingProgress.PathFor(outputModel));
    }

    /// <summary>A random patch of one movie's prepared halves; blank patches are redrawn as Noise2Mic does.</summary>
    static (float[] Odd, float[] Even) SamplePatch((Prepared Odd, Prepared Even)[] pairs, int index, Random random)
    {
        int patch = Noise2MicTorch.PatchSize;
        for (int attempt = 0; ; attempt++)
        {
            var (oddImage, evenImage) = pairs[index];
            float[] oddData = oddImage.Data, evenData = evenImage.Data;
            int width = oddImage.Width, height = oddImage.Height;
            if (evenImage.Width != width || evenImage.Height != height || width < patch || height < patch)
                throw new InvalidDataException($"Training pair {index} has incompatible dimensions.");
            int x0 = random.Next(width - patch + 1), y0 = random.Next(height - patch + 1);
            int turns = random.Next(4);
            bool flipX = random.Next(2) == 0, flipY = random.Next(2) == 0;
            var odd = Augment(oddData, width, x0, y0, patch, turns, flipX, flipY);
            var even = Augment(evenData, width, x0, y0, patch, turns, flipX, flipY);
            if (StandardDeviation(odd) >= BlankThreshold && StandardDeviation(even) >= BlankThreshold || attempt >= 10)
                return (odd, even);
            index = random.Next(pairs.Length);
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
