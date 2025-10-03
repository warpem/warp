using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Item in the denoising pipeline
    /// </summary>
    public class DenoisingItem : IDisposable
    {
        public Image Map { get; set; }
        public Image Map2 { get; set; } // For separate denoising
        public string MapName { get; set; }
        public float2 MeanStd { get; set; }
        public float PixelSize { get; set; }
        public int Index { get; set; }

        public void Dispose()
        {
            Map?.Dispose();
            Map2?.Dispose();
        }
    }

    /// <summary>
    /// Streaming pipeline for memory-efficient denoising
    /// Loads → Denoises → Saves maps one at a time with concurrent execution
    /// </summary>
    public class DenoisingPipeline
    {
        private readonly List<MapFileInfo> mapInfoList;
        private readonly Options options;
        private readonly ProcessingContext context;
        private readonly NoiseNet3DTorch model;

        private readonly BlockingCollection<DenoisingItem> loadedQueue;
        private readonly BlockingCollection<DenoisingItem> denoisedQueue;

        private const int QueueCapacity = 2; // Small capacity to limit memory usage

        // Progress tracking
        private readonly Stopwatch progressWatch = new Stopwatch();
        private int completedMaps = 0;
        private double smoothedMapTime = 0;

        public DenoisingPipeline(List<MapFileInfo> mapInfoList, Options options, ProcessingContext context, NoiseNet3DTorch model)
        {
            this.mapInfoList = mapInfoList;
            this.options = options;
            this.context = context;
            this.model = model;

            this.loadedQueue = new BlockingCollection<DenoisingItem>(QueueCapacity);
            this.denoisedQueue = new BlockingCollection<DenoisingItem>(QueueCapacity);
        }

        /// <summary>
        /// Runs the full pipeline: loader thread → denoiser (main thread) → saver thread
        /// </summary>
        public void ProcessAll()
        {
            Directory.CreateDirectory(Path.Combine(context.WorkingDirectory, "denoised"));

            Console.WriteLine($"Denoising {mapInfoList.Count} maps:\n");
            progressWatch.Start();

            var cancellationSource = new CancellationTokenSource();

            // Start loader thread
            var loaderTask = Task.Run(() => LoaderThread(cancellationSource.Token));

            // Start saver thread
            var saverTask = Task.Run(() => SaverThread(cancellationSource.Token));

            try
            {
                // Denoiser runs on main thread (GPU operations)
                DenoiserThread();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Denoising error: {ex.Message}");
                cancellationSource.Cancel();
                throw;
            }
            finally
            {
                // Wait for threads to complete
                try
                {
                    Task.WaitAll(loaderTask, saverTask);
                }
                catch (AggregateException)
                {
                    // Ignore cancellation exceptions
                }

                loadedQueue?.Dispose();
                denoisedQueue?.Dispose();
            }

            Console.WriteLine("\n\nAll done!");
        }

        private void LoaderThread(CancellationToken cancellationToken)
        {
            try
            {
                GPU.SetDevice(options.GPUPreprocess);

                for (int i = 0; i < mapInfoList.Count; i++)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    MapFileInfo info = mapInfoList[i];

                    // Load maps
                    Image map1 = Image.FromFile(info.Map1Path);
                    Image map2 = Image.FromFile(info.Map2Path);
                    Image mapCombined = info.MapCombinedPath == null ? null : Image.FromFile(info.MapCombinedPath);

                    // Calculate preprocessing params on first load
                    if (info.SpectrumMultipliers == null && !options.DontFlatten)
                    {

                        Image average = map1.GetCopy();
                        average.Add(map2);

                        if (context.Mask != null)
                            average.Multiply(context.Mask);

                        float[] spectrum = average.AsAmplitudes1D(true, 1, (average.Dims.X + average.Dims.Y + average.Dims.Z) / 6);
                        average.Dispose();

                        int j10A = (int)(options.PixelSize * 2 / 10 * spectrum.Length);
                        float amp10A = spectrum[j10A];

                        for (int j = 0; j < spectrum.Length; j++)
                            spectrum[j] = j < j10A ? 1 : (amp10A / spectrum[j] * options.Overflatten);

                        info.SpectrumMultipliers = spectrum;
                    }

                    // Apply spectral flattening
                    if (info.SpectrumMultipliers != null)
                    {
                        Image map1Flat = map1.AsSpectrumMultiplied(true, info.SpectrumMultipliers);
                        map1.Dispose();
                        map1 = map1Flat;
                        map1.FreeDevice();

                        Image map2Flat = map2.AsSpectrumMultiplied(true, info.SpectrumMultipliers);
                        map2.Dispose();
                        map2 = map2Flat;
                        map2.FreeDevice();

                        if (mapCombined != null)
                        {
                            Image mapCombinedFlat = mapCombined.AsSpectrumMultiplied(true, info.SpectrumMultipliers);
                            mapCombined.Dispose();
                            mapCombined = mapCombinedFlat;
                            mapCombined.FreeDevice();
                        }
                    }

                    // Apply lowpass
                    if (options.Lowpass > 0)
                    {
                        map1.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
                        map2.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
                        mapCombined?.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
                    }

                    // Crop if needed
                    if (info.CropBox.X > 0)
                    {
                        Image map1Cropped = map1.AsPadded(info.CropBox);
                        map1.Dispose();
                        map1 = map1Cropped;
                        map1.FreeDevice();

                        Image map2Cropped = map2.AsPadded(info.CropBox);
                        map2.Dispose();
                        map2 = map2Cropped;
                        map2.FreeDevice();

                        if (mapCombined != null)
                        {
                            Image mapCombinedCropped = mapCombined.AsPadded(info.CropBox);
                            mapCombined.Dispose();
                            mapCombined = mapCombinedCropped;
                            mapCombined.FreeDevice();
                        }
                    }

                    // Calculate mean/std on first load (after flattening)
                    if (info.MeanStd.X == 0 && info.MeanStd.Y == 1)
                    {
                        Image map1Center = map1.AsPadded(map1.Dims / 2);
                        Image map2Center = map2.AsPadded(map2.Dims / 2);
                        float2 meanStd = MathHelper.MeanAndStd(Helper.Combine(map1Center.GetHostContinuousCopy(), map2Center.GetHostContinuousCopy()));
                        map1Center.Dispose();
                        map2Center.Dispose();

                        info.MeanStd = meanStd;
                    }

                    // Normalize
                    map1.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));
                    map2.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));
                    mapCombined?.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));

                    // Prepare for denoising
                    Image forDenoising = (mapCombined == null || options.DenoiseSeparately) ? map1.GetCopy() : mapCombined;
                    Image forDenoising2 = options.DenoiseSeparately ? map2.GetCopy() : null;

                    if (!options.DenoiseSeparately)
                    {
                        forDenoising.Add(map2);
                        forDenoising.Multiply(0.5f);
                    }

                    forDenoising.FreeDevice();
                    forDenoising2?.FreeDevice();

                    // Dispose temporary maps
                    map1.Dispose();
                    map2.Dispose();
                    mapCombined?.Dispose();

                    var item = new DenoisingItem
                    {
                        Map = forDenoising,
                        Map2 = forDenoising2,
                        MapName = info.MapName,
                        MeanStd = info.MeanStd,
                        PixelSize = info.PixelSize,
                        Index = i
                    };

                    loadedQueue.Add(item, cancellationToken);
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during cancellation
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Loader error: {ex.Message}");
                throw;
            }
            finally
            {
                loadedQueue.CompleteAdding();
            }
        }

        private void DenoiserThread()
        {
            GPU.SetDevice(options.GPUPreprocess);

            foreach (var item in loadedQueue.GetConsumingEnumerable())
            {
                try
                {
                    // Denoise first map
                    NoiseNet3DTorch.Denoise(item.Map, new NoiseNet3DTorch[] { model });
                    item.Map.TransformValues(v => v * item.MeanStd.Y + item.MeanStd.X);
                    item.Map.PixelSize = item.PixelSize;
                    ApplyMask(item.Map);

                    // Denoise second map if needed
                    if (item.Map2 != null)
                    {
                        NoiseNet3DTorch.Denoise(item.Map2, new NoiseNet3DTorch[] { model });
                        item.Map2.TransformValues(v => v * item.MeanStd.Y + item.MeanStd.X);
                        item.Map2.PixelSize = item.PixelSize;
                        ApplyMask(item.Map2);
                    }

                    denoisedQueue.Add(item);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\nError denoising {item.MapName}: {ex.Message}");
                    item.Dispose();
                    throw;
                }
            }

            denoisedQueue.CompleteAdding();
        }

        private void SaverThread(CancellationToken cancellationToken)
        {
            try
            {
                foreach (var item in denoisedQueue.GetConsumingEnumerable())
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    try
                    {
                        string savePath = Path.Combine(context.WorkingDirectory, "denoised",
                                                      item.MapName + (options.DenoiseSeparately ? "_1" : "") + ".mrc");
                        item.Map.WriteMRC16b(savePath, true);

                        if (item.Map2 != null)
                        {
                            string savePath2 = Path.Combine(context.WorkingDirectory, "denoised", item.MapName + "_2.mrc");
                            item.Map2.WriteMRC16b(savePath2, true);
                        }

                        // Update progress
                        completedMaps++;
                        double mapTime = progressWatch.Elapsed.TotalSeconds;

                        // Update smoothed time
                        if (completedMaps == 1)
                        {
                            smoothedMapTime = mapTime;
                        }
                        else if (completedMaps <= 5)
                        {
                            smoothedMapTime = (smoothedMapTime * (completedMaps - 1) + mapTime) / completedMaps;
                        }
                        else
                        {
                            double alpha = 0.1;
                            smoothedMapTime = alpha * mapTime + (1 - alpha) * smoothedMapTime;
                        }

                        PrintProgress(item.MapName);
                        progressWatch.Restart();
                    }
                    finally
                    {
                        item.Dispose();
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during cancellation
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nSaver error: {ex.Message}");
                throw;
            }
        }

        private void PrintProgress(string currentMapName)
        {
            int remainingMaps = mapInfoList.Count - completedMaps;
            double totalSecondsRemaining = smoothedMapTime * remainingMaps;
            TimeSpan timeRemaining = TimeSpan.FromSeconds(totalSecondsRemaining);

            string toWrite = $"{completedMaps}/{mapInfoList.Count}, " +
                            (timeRemaining.Days > 0 ? (timeRemaining.Days + " days ") : "") +
                            $"{timeRemaining.Hours}:{timeRemaining.Minutes:D2}:{timeRemaining.Seconds:D2} remaining, " +
                            $"saved {currentMapName}";

            try
            {
                VirtualConsole.ClearLastLine();
                Console.Write(toWrite);
            }
            catch
            {
                // When outputting to a text file on HPC cluster
                Console.WriteLine(toWrite);
            }
        }

        private void ApplyMask(Image map)
        {
            if (options.MaskOutput)
                map.Multiply(context.Mask);
            else if (context.Mask != null && !options.DontKeepDimensions)
                map.MaskSpherically(map.Dims.X - 32, 16, true);
        }
    }
}
