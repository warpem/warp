using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Manages a rotating pool of loaded maps for memory-efficient training on large datasets.
    /// Uses FIFO strategy to prevent overfitting on any particular map pair.
    /// Background thread pre-loads maps for instant rotation without blocking training.
    /// </summary>
    public class RotatingMapPool : IDisposable
    {
        private readonly List<MapFileInfo> allMapInfo;
        private readonly int maxLoadedMaps;
        private readonly Options options;
        private readonly ProcessingContext context;
        private readonly Random rand;
        private readonly object poolLock = new object();

        // Currently loaded maps
        private readonly List<LoadedMapEntry> loadedMaps;
        private readonly Queue<int> loadOrder; // Tracks indices in loadedMaps for FIFO
        private readonly HashSet<int> loadedMapInfoIndices; // Tracks which MapFileInfo entries are loaded

        // Preloaded maps (ready for instant swap)
        private readonly BlockingCollection<PreloadedMapEntry> preloadedMaps;
        private readonly CancellationTokenSource preloaderCancellation;
        private readonly Task preloaderTask;

        public int TotalMapCount => allMapInfo.Count;
        public int CurrentPoolSize => loadedMaps.Count;

        private class LoadedMapEntry
        {
            public int MapInfoIndex { get; set; }
            public Image Map1 { get; set; }
            public Image Map2 { get; set; }
            public Image MapCTF { get; set; }
        }

        private class PreloadedMapEntry
        {
            public int MapInfoIndex { get; set; }
            public Image Map1 { get; set; }
            public Image Map2 { get; set; }
            public Image MapCTF { get; set; }
        }

        public RotatingMapPool(List<MapFileInfo> mapInfo, int maxLoadedMaps, Options options, ProcessingContext context, int seed = 123)
        {
            this.allMapInfo = mapInfo;
            this.maxLoadedMaps = maxLoadedMaps;
            this.options = options;
            this.context = context;
            this.rand = new Random(seed);

            this.loadedMaps = new List<LoadedMapEntry>();
            this.loadOrder = new Queue<int>();
            this.loadedMapInfoIndices = new HashSet<int>();

            // Initialize preload queue (capacity 2 for instant swaps)
            this.preloadedMaps = new BlockingCollection<PreloadedMapEntry>(2);
            this.preloaderCancellation = new CancellationTokenSource();

            // Load initial pool
            LoadInitialPool();

            // Start background preloader thread (only if we have more maps than pool size)
            if (allMapInfo.Count > maxLoadedMaps)
            {
                preloaderTask = Task.Run(() => PreloaderThread(), preloaderCancellation.Token);
            }
        }

        private void LoadInitialPool()
        {
            int mapsToLoad = Math.Min(maxLoadedMaps, allMapInfo.Count);

            // Randomly select initial maps
            int[] initialIndices = Helper.RandomSubset(
                Helper.ArrayOfSequence(0, allMapInfo.Count, 1),
                mapsToLoad,
                rand.Next(9999999));

            foreach (int mapInfoIndex in initialIndices)
            {
                LoadMapAtIndex(mapInfoIndex);
            }

            Console.WriteLine($"Loaded initial pool of {loadedMaps.Count} maps out of {allMapInfo.Count} total.");
        }

        private void LoadMapAtIndex(int mapInfoIndex)
        {
            MapFileInfo info = allMapInfo[mapInfoIndex];

            GPU.SetDevice(options.GPUPreprocess);

            // Load images
            Image map1 = Image.FromFile(info.Map1Path);
            Image map2 = Image.FromFile(info.Map2Path);

            // Calculate preprocessing params on first load
            if (info.SpectrumMultipliers == null && !options.DontFlatten)
            {
                Console.Write($"Calculating spectrum for {info.MapName}... ");

                // Calculate spectrum multipliers from both maps
                Image average = map1.GetCopy();
                average.Add(map2);

                if (context.Mask != null)
                    average.Multiply(context.Mask);

                float[] spectrum = average.AsAmplitudes1D(true, 1, (average.Dims.X + average.Dims.Y + average.Dims.Z) / 6);
                average.Dispose();

                int i10A = (int)(options.PixelSize * 2 / 10 * spectrum.Length);
                float amp10A = spectrum[i10A];

                for (int i = 0; i < spectrum.Length; i++)
                    spectrum[i] = i < i10A ? 1 : (amp10A / spectrum[i] * options.Overflatten);

                info.SpectrumMultipliers = spectrum;
                Console.WriteLine("Done.");
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

            // Apply lowpass if specified
            if (options.Lowpass > 0)
            {
                map1.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
                map2.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
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
            }

            // Normalize
            map1.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));
            map2.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));

            // Prefilter for cubic interpolation
            GPU.PrefilterForCubic(map1.GetDevice(Intent.ReadWrite), map1.Dims);
            GPU.PrefilterForCubic(map2.GetDevice(Intent.ReadWrite), map2.Dims);

            map1.FreeDevice();
            map2.FreeDevice();

            // Load CTF
            Image mapCTF = LoadCTFForMap(info);

            // Add to pool
            var entry = new LoadedMapEntry
            {
                MapInfoIndex = mapInfoIndex,
                Map1 = map1,
                Map2 = map2,
                MapCTF = mapCTF
            };

            int entryIndex = loadedMaps.Count;
            loadedMaps.Add(entry);
            loadOrder.Enqueue(entryIndex);
            loadedMapInfoIndices.Add(mapInfoIndex);
        }

        private Image LoadCTFForMap(MapFileInfo info)
        {
            if (!string.IsNullOrEmpty(info.CTFPath) && System.IO.File.Exists(info.CTFPath))
            {
                Image mapCTF = Image.FromFile(info.CTFPath);
                ProcessCTF(ref mapCTF);
                mapCTF.GetDevice(Intent.Read); // Ensure on GPU
                return mapCTF;
            }
            else
            {
                Image mapCTF = new Image(new int3(128), true);
                mapCTF.TransformValues(v => 1f);
                mapCTF.GetDevice(Intent.Read);
                return mapCTF;
            }
        }

        private void ProcessCTF(ref Image mapCTF)
        {
            int dimCTF = mapCTF.Dims.Y;
            mapCTF.Dims = new int3(dimCTF);
            mapCTF.IsFT = true;
            Image ctfComplex = new Image(mapCTF.Dims, true, true);
            ctfComplex.Fill(new float2(1, 0));
            ctfComplex.Multiply(mapCTF);
            mapCTF.Dispose();
            Image ctfReal = ctfComplex.AsIFFT(true).AndDisposeParent();
            Image ctfPadded = ctfReal.AsPadded(context.TrainingDims * 2, true).AndDisposeParent();
            ctfComplex = ctfPadded.AsFFT(true).AndDisposeParent();
            mapCTF = ctfComplex.AsReal().AndDisposeParent();
            mapCTF.Multiply(1f / (dimCTF * dimCTF * dimCTF));

            float[][] ctfData = mapCTF.GetHost(Intent.ReadWrite);
            int ctfDimsX = mapCTF.Dims.X;
            int3 trainingDims = context.TrainingDims;

            Helper.ForEachElementFT(trainingDims * 2, (x, y, z, xx, yy, zz, r) =>
            {
                float xxx = xx / (float)trainingDims.X;
                float yyy = yy / (float)trainingDims.Y;
                float zzz = zz / (float)trainingDims.Z;

                r = (float)Math.Sqrt(xxx * xxx + yyy * yyy + zzz * zzz);

                float b = Math.Min(Math.Max(0, r - 0.98f) / 0.02f, 1);

                r = Math.Min(1, r / 0.05f);
                r = (float)Math.Cos(r * Math.PI) * 0.5f + 0.5f;

                float a = 90;
                if (zzz != 0)
                    a = (float)Math.Atan(Math.Abs(xxx / zzz)) * Helper.ToDeg;
                a = Math.Max(0, Math.Min(1, (a - 20) / 5));
                a = 1;

                int i = y * (ctfDimsX / 2 + 1) + x;
                ctfData[z][i] = MathHelper.Lerp(MathHelper.Lerp(MathHelper.Lerp(ctfData[z][i], 1, r), 1, b), 1, 1 - a);
            });
        }

        /// <summary>
        /// Rotates out the oldest map with a preloaded map (instant swap, non-blocking).
        /// If no preloaded map is ready, skips rotation to avoid blocking training.
        /// </summary>
        public void RotateOldest()
        {
            // Don't rotate if we have fewer maps than max
            if (allMapInfo.Count <= maxLoadedMaps)
                return;

            // Try to get a preloaded map (non-blocking)
            if (!preloadedMaps.TryTake(out PreloadedMapEntry newEntry))
            {
                // No preloaded map ready, skip rotation to keep training going
                return;
            }

            lock (poolLock)
            {
                // Check if this map is already loaded (edge case - unlikely but possible)
                if (loadedMapInfoIndices.Contains(newEntry.MapInfoIndex))
                {
                    // Already loaded somehow, dispose the preloaded one
                    newEntry.Map1.Dispose();
                    newEntry.Map2.Dispose();
                    newEntry.MapCTF.Dispose();
                    return;
                }

                // Unload oldest
                int oldestSlotIndex = loadOrder.Dequeue();
                LoadedMapEntry oldEntry = loadedMaps[oldestSlotIndex];

                loadedMapInfoIndices.Remove(oldEntry.MapInfoIndex);
                oldEntry.Map1.Dispose();
                oldEntry.Map2.Dispose();
                oldEntry.MapCTF.Dispose();

                // Install new entry in the slot
                loadedMaps[oldestSlotIndex] = new LoadedMapEntry
                {
                    MapInfoIndex = newEntry.MapInfoIndex,
                    Map1 = newEntry.Map1,
                    Map2 = newEntry.Map2,
                    MapCTF = newEntry.MapCTF
                };

                loadOrder.Enqueue(oldestSlotIndex);
                loadedMapInfoIndices.Add(newEntry.MapInfoIndex);
            }
        }

        /// <summary>
        /// Background thread that continuously preloads maps for instant rotation
        /// </summary>
        private void PreloaderThread()
        {
            try
            {
                while (!preloaderCancellation.Token.IsCancellationRequested)
                {
                    // Find an unloaded map to preload
                    int mapIndexToLoad = -1;
                    lock (poolLock)
                    {
                        List<int> unloadedIndices = new List<int>();
                        for (int i = 0; i < allMapInfo.Count; i++)
                        {
                            if (!loadedMapInfoIndices.Contains(i))
                                unloadedIndices.Add(i);
                        }

                        if (unloadedIndices.Count == 0)
                        {
                            // All maps are loaded, wait a bit
                            Thread.Sleep(100);
                            continue;
                        }

                        mapIndexToLoad = unloadedIndices[rand.Next(unloadedIndices.Count)];
                    }

                    // Load the map (this takes time, done outside lock)
                    PreloadedMapEntry entry = LoadMapForPreload(mapIndexToLoad);

                    // Add to preload queue (blocks if full, which is fine - limits memory)
                    preloadedMaps.Add(entry, preloaderCancellation.Token);
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during shutdown
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Preloader thread error: {ex.Message}");
            }
        }

        /// <summary>
        /// Loads and preprocesses a map for the preload queue
        /// </summary>
        private PreloadedMapEntry LoadMapForPreload(int mapInfoIndex)
        {
            MapFileInfo info = allMapInfo[mapInfoIndex];

            GPU.SetDevice(options.GPUPreprocess);

            Image map1 = Image.FromFile(info.Map1Path);
            Image map2 = Image.FromFile(info.Map2Path);

            // Calculate preprocessing params on first load
            if (info.SpectrumMultipliers == null && !options.DontFlatten)
            {
                Image average = map1.GetCopy();
                average.Add(map2);

                if (context.Mask != null)
                    average.Multiply(context.Mask);

                float[] spectrum = average.AsAmplitudes1D(true, 1, (average.Dims.X + average.Dims.Y + average.Dims.Z) / 6);
                average.Dispose();

                int i10A = (int)(options.PixelSize * 2 / 10 * spectrum.Length);
                float amp10A = spectrum[i10A];

                for (int i = 0; i < spectrum.Length; i++)
                    spectrum[i] = i < i10A ? 1 : (amp10A / spectrum[i] * options.Overflatten);

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

            // Apply lowpass
            if (options.Lowpass > 0)
            {
                map1.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
                map2.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
            }

            // Crop
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
            }

            // Normalize
            map1.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));
            map2.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));

            // Prefilter for cubic interpolation
            GPU.PrefilterForCubic(map1.GetDevice(Intent.ReadWrite), map1.Dims);
            GPU.PrefilterForCubic(map2.GetDevice(Intent.ReadWrite), map2.Dims);

            map1.FreeDevice();
            map2.FreeDevice();

            // Load CTF
            Image mapCTF = LoadCTFForMap(info);

            return new PreloadedMapEntry
            {
                MapInfoIndex = mapInfoIndex,
                Map1 = map1,
                Map2 = map2,
                MapCTF = mapCTF
            };
        }

        private void LoadMapAtIndexIntoSlot(int mapInfoIndex, int slotIndex)
        {
            MapFileInfo info = allMapInfo[mapInfoIndex];

            GPU.SetDevice(options.GPUPreprocess);

            Image map1 = Image.FromFile(info.Map1Path);
            Image map2 = Image.FromFile(info.Map2Path);

            // Calculate preprocessing params on first load
            if (info.SpectrumMultipliers == null && !options.DontFlatten)
            {
                Console.Write($"Calculating spectrum for {info.MapName}... ");

                Image average = map1.GetCopy();
                average.Add(map2);

                if (context.Mask != null)
                    average.Multiply(context.Mask);

                float[] spectrum = average.AsAmplitudes1D(true, 1, (average.Dims.X + average.Dims.Y + average.Dims.Z) / 6);
                average.Dispose();

                int i10A = (int)(options.PixelSize * 2 / 10 * spectrum.Length);
                float amp10A = spectrum[i10A];

                for (int i = 0; i < spectrum.Length; i++)
                    spectrum[i] = i < i10A ? 1 : (amp10A / spectrum[i] * options.Overflatten);

                info.SpectrumMultipliers = spectrum;
                Console.WriteLine("Done.");
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

            if (options.Lowpass > 0)
            {
                map1.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
                map2.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
            }

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
            }

            map1.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));
            map2.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));

            GPU.PrefilterForCubic(map1.GetDevice(Intent.ReadWrite), map1.Dims);
            GPU.PrefilterForCubic(map2.GetDevice(Intent.ReadWrite), map2.Dims);

            map1.FreeDevice();
            map2.FreeDevice();

            Image mapCTF = LoadCTFForMap(info);

            var entry = new LoadedMapEntry
            {
                MapInfoIndex = mapInfoIndex,
                Map1 = map1,
                Map2 = map2,
                MapCTF = mapCTF
            };

            loadedMaps[slotIndex] = entry;
            loadOrder.Enqueue(slotIndex);
            loadedMapInfoIndices.Add(mapInfoIndex);
        }

        /// <summary>
        /// Gets a map from the current pool. Thread-safe.
        /// </summary>
        public void GetMap(int poolIndex, out Image map1, out Image map2, out Image mapCTF)
        {
            lock (poolLock)
            {
                if (poolIndex < 0 || poolIndex >= loadedMaps.Count)
                    throw new ArgumentOutOfRangeException(nameof(poolIndex));

                LoadedMapEntry entry = loadedMaps[poolIndex];
                if (entry == null)
                    throw new InvalidOperationException($"Map at pool index {poolIndex} is not loaded");

                map1 = entry.Map1;
                map2 = entry.Map2;
                mapCTF = entry.MapCTF;
            }
        }

        public void Dispose()
        {
            // Stop preloader thread
            preloaderCancellation?.Cancel();
            try
            {
                preloaderTask?.Wait(1000); // Wait up to 1 second
            }
            catch (AggregateException)
            {
                // Ignore cancellation exceptions
            }

            // Dispose preloaded maps
            while (preloadedMaps.TryTake(out PreloadedMapEntry entry))
            {
                entry.Map1?.Dispose();
                entry.Map2?.Dispose();
                entry.MapCTF?.Dispose();
            }
            preloadedMaps?.Dispose();

            // Dispose loaded maps
            foreach (var entry in loadedMaps)
            {
                if (entry != null)
                {
                    entry.Map1?.Dispose();
                    entry.Map2?.Dispose();
                    entry.MapCTF?.Dispose();
                }
            }

            loadedMaps.Clear();
            loadOrder.Clear();
            loadedMapInfoIndices.Clear();

            preloaderCancellation?.Dispose();
        }
    }
}
