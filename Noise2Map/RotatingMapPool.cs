using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using Warp.Tools.Async;

namespace Noise2Map
{
    /// <summary>
    /// Container for a loaded map triple (Map1, Map2, CTF)
    /// </summary>
    public class LoadedMapData : IDisposable
    {
        public Image Map1 { get; set; }
        public Image Map2 { get; set; }
        public Image MapCTF { get; set; }

        public void Dispose()
        {
            Map1?.Dispose();
            Map2?.Dispose();
            MapCTF?.Dispose();
        }
    }

    /// <summary>
    /// Manages a rotating pool of loaded maps for memory-efficient training on large datasets.
    /// Uses FIFO strategy to prevent overfitting on any particular map pair.
    /// Background thread pre-loads maps for instant rotation without blocking training.
    /// </summary>
    public class RotatingMapPool : IDisposable
    {
        private readonly List<MapFileInfo> allMapInfo;
        private readonly Options options;
        private readonly ProcessingContext context;
        private readonly RotatingPool<LoadedMapData, MapFileInfo> pool;
        private readonly Queue<MapFileInfo> pendingMaps = new Queue<MapFileInfo>();
        private readonly object pendingLock = new object();

        public int TotalMapCount => pool.TotalCount;
        public int CurrentPoolSize => pool.LoadedCount;

        public RotatingMapPool(List<MapFileInfo> mapInfo, int maxLoadedMaps, Options options, ProcessingContext context, int seed = 123)
        {
            this.allMapInfo = mapInfo;
            this.options = options;
            this.context = context;

            // Create WarpLib RotatingPool with map loading function
            this.pool = new RotatingPool<LoadedMapData, MapFileInfo>(
                allMetadata: mapInfo,
                maxLoaded: maxLoadedMaps,
                loadFunction: LoadMapData,
                preloadCapacity: 2,
                gpuDevice: options.GPUPreprocess);
        }

        /// <summary>
        /// Loads and preprocesses a map from metadata
        /// </summary>
        private LoadedMapData LoadMapData(MapFileInfo info, CancellationToken cancellationToken)
        {
            Console.WriteLine($"[DEBUG] RotatingMapPool: Loading map {info.MapName}, CurrentPoolSize will become {CurrentPoolSize + 1}");
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
                map1 = map1.AsSpectrumMultiplied(true, info.SpectrumMultipliers).AndDisposeParent();
                map2 = map2.AsSpectrumMultiplied(true, info.SpectrumMultipliers).AndDisposeParent();
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

            // Apply lowpass if specified
            if (options.Lowpass > 0)
            {
                map1.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
                map2.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
            }

            // Crop if needed
            if (info.CropBox.X > 0)
            {
                map1 = map1.AsPadded(info.CropBox).AndDisposeParent();
                map2 = map2.AsPadded(info.CropBox).AndDisposeParent();
            }

            // Normalize
            map1.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));
            map2.TransformValues(v => Math.Max(-30, Math.Min(30, (v - info.MeanStd.X) / info.MeanStd.Y)));

            // Prefilter for cubic interpolation
            GPU.PrefilterForCubic(map1.GetDevice(Intent.ReadWrite), map1.Dims);
            GPU.PrefilterForCubic(map2.GetDevice(Intent.ReadWrite), map2.Dims);

            // Load CTF
            Image mapCTF = LoadCTFForMap(info);

            return new LoadedMapData
            {
                Map1 = map1,
                Map2 = map2,
                MapCTF = mapCTF
            };
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
            pool.RotateOldest();
        }


        /// <summary>
        /// Gets a map from the current pool. Thread-safe.
        /// </summary>
        public void GetMap(int poolIndex, out Image map1, out Image map2, out Image mapCTF)
        {
            pool.GetItem(poolIndex, out LoadedMapData data);
            map1 = data.Map1;
            map2 = data.Map2;
            mapCTF = data.MapCTF;
        }

        /// <summary>
        /// Adds a new map to the pending queue for online mode. Thread-safe.
        /// </summary>
        public void AddNewMap(MapFileInfo mapInfo)
        {
            lock (pendingLock)
            {
                pendingMaps.Enqueue(mapInfo);
            }

            ProcessPendingMaps();
        }

        /// <summary>
        /// Processes pending maps and adds them to the pool's metadata list
        /// </summary>
        private void ProcessPendingMaps()
        {
            lock (pendingLock)
            {
                while (pendingMaps.Count > 0)
                {
                    MapFileInfo newMap = pendingMaps.Dequeue();

                    // Check for duplicates (by map name)
                    if (allMapInfo.Any(m => m.MapName == newMap.MapName))
                    {
                        Console.WriteLine($"Skipping duplicate map: {newMap.MapName}");
                        continue;
                    }

                    // Add to pool metadata (allMapInfo and pool.allMetadata share the same list reference)
                    pool.AddMetadata(newMap);
                    Console.WriteLine($"New map added: {newMap.MapName} (Total: {TotalMapCount})");
                }
            }
        }

        public void Dispose()
        {
            pool?.Dispose();
        }
    }
}
