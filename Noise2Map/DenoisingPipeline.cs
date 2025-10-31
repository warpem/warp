using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using Warp.Tools.Async;

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

        private const int QueueCapacity = 2; // Small capacity to limit memory usage

        // Progress tracking
        private ProgressTracker progressTracker;

        public DenoisingPipeline(List<MapFileInfo> mapInfoList, Options options, ProcessingContext context, NoiseNet3DTorch model)
        {
            this.mapInfoList = mapInfoList;
            this.options = options;
            this.context = context;
            this.model = model;
        }

        /// <summary>
        /// Runs the full pipeline: loader thread → denoiser (main thread) → saver thread
        /// </summary>
        public void ProcessAll(CancellationToken cancellationToken = default)
        {
            Directory.CreateDirectory(Path.Combine(context.WorkingDirectory, "denoised"));

            Console.WriteLine($"Denoising {mapInfoList.Count} maps:\n");
            progressTracker = new ProgressTracker(mapInfoList.Count, warmupItems: 1, emaAlpha: 0.1);

            // Build streaming pipeline: Load → Denoise → Save
            // Note: Saver runs on main thread to ensure pipeline doesn't exit prematurely
            var pipeline = new StreamingPipeline<MapFileInfo>.Builder()
                .AddStage<MapFileInfo, DenoisingItem>("Loader", LoadMap, QueueCapacity, runInBackground: true, gpuDevice: options.GPUPreprocess)
                .AddStage<DenoisingItem, DenoisingItem>("Denoiser", DenoiseMap, QueueCapacity, runInBackground: true, gpuDevice: options.GPUPreprocess)
                .AddStage<DenoisingItem, bool>("Saver", SaveMap, QueueCapacity, runInBackground: false, gpuDevice: -1)
                .Build();

            try
            {
                pipeline.ProcessAll(mapInfoList, cancellationToken);
            }
            finally
            {
                pipeline.Dispose();
            }

            progressTracker?.Complete();
            Console.WriteLine("\nAll done!");
        }

        /// <summary>
        /// Pipeline stage: Loads and preprocesses a map
        /// </summary>
        private DenoisingItem LoadMap(MapFileInfo info, CancellationToken cancellationToken)
        {
            GPU.SetDevice(options.GPUPreprocess);

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
                map1 = map1.AsSpectrumMultiplied(true, info.SpectrumMultipliers).AndDisposeParent();
                map2 = map2.AsSpectrumMultiplied(true, info.SpectrumMultipliers).AndDisposeParent();

                if (mapCombined != null)
                {
                    mapCombined = mapCombined.AsSpectrumMultiplied(true, info.SpectrumMultipliers).AndDisposeParent();
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
                map1 = map1.AsPadded(info.CropBox).AndDisposeParent();
                map2 = map2.AsPadded(info.CropBox).AndDisposeParent();

                if (mapCombined != null)
                {
                    mapCombined = mapCombined.AsPadded(info.CropBox).AndDisposeParent();
                }
            }

            // Calculate mean/std on first load (after flattening)
            if (info.MeanStd.X == 0 && info.MeanStd.Y == 1)
            {
                using Image map1Center = map1.AsPadded(map1.Dims / 2);
                using Image map2Center = map2.AsPadded(map2.Dims / 2);
                float2 meanStd = MathHelper.MeanAndStd(Helper.Combine(map1Center.GetHost(Intent.Read), 
                                                                      map2Center.GetHost(Intent.Read)));

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

            return new DenoisingItem
            {
                Map = forDenoising,
                Map2 = forDenoising2,
                MapName = info.MapName,
                MeanStd = info.MeanStd,
                PixelSize = info.PixelSize,
                Index = mapInfoList.IndexOf(info)
            };
        }

        /// <summary>
        /// Pipeline stage: Denoises a map
        /// </summary>
        private DenoisingItem DenoiseMap(DenoisingItem item, CancellationToken cancellationToken)
        {
            GPU.SetDevice(options.GPUPreprocess);

            // Denoise first map
            NoiseNet3DTorch.Denoise(item.Map, [model]);
            item.Map.TransformValues(v => v * item.MeanStd.Y + item.MeanStd.X);
            item.Map.PixelSize = item.PixelSize;
            ApplyMask(item.Map);

            // Denoise second map if needed
            if (item.Map2 != null)
            {
                NoiseNet3DTorch.Denoise(item.Map2, [model]);
                item.Map2.TransformValues(v => v * item.MeanStd.Y + item.MeanStd.X);
                item.Map2.PixelSize = item.PixelSize;
                ApplyMask(item.Map2);
            }

            return item;
        }

        /// <summary>
        /// Pipeline stage: Saves a denoised map
        /// </summary>
        private bool SaveMap(DenoisingItem item, CancellationToken cancellationToken)
        {
            try
            {
                string savePath = Path.Combine(context.WorkingDirectory, "denoised",
                                              item.MapName + (options.DenoiseSeparately ? "_1" : "") + ".mrc");
                item.Map.WriteMRC16b(savePath, true);
                item.Map.Dispose();

                if (item.Map2 != null)
                {
                    string savePath2 = Path.Combine(context.WorkingDirectory, "denoised", item.MapName + "_2.mrc");
                    item.Map2.WriteMRC16b(savePath2, true);
                    item.Map2.Dispose();
                }

                // Update progress
                progressTracker.Update($"saved {item.MapName}");

                return true;
            }
            finally
            {
                item.Dispose();
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
