using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Monitors data directories for new map pairs and notifies RotatingMapPool
    /// </summary>
    public class MapFileWatcher : IDisposable
    {
        private readonly ProcessingContext context;
        private readonly Options options;
        private readonly RotatingMapPool mapPool;
        private readonly FileSystemWatcher watcher1;
        private readonly FileSystemWatcher watcher2;
        private readonly HashSet<string> processedMaps;
        private readonly object lockObj = new object();

        public MapFileWatcher(ProcessingContext context, Options options, RotatingMapPool mapPool)
        {
            this.context = context;
            this.options = options;
            this.mapPool = mapPool;
            this.processedMaps = new HashSet<string>();

            // Set up file system watchers for both observation directories
            if (!string.IsNullOrEmpty(options.Observation1Path))
            {
                string path1 = Path.Combine(context.WorkingDirectory, options.Observation1Path);
                string path2 = Path.Combine(context.WorkingDirectory, options.Observation2Path);

                watcher1 = CreateWatcher(path1);
                watcher2 = CreateWatcher(path2);

                watcher1.Created += OnFileCreated;
                watcher2.Created += OnFileCreated;

                watcher1.EnableRaisingEvents = true;
                watcher2.EnableRaisingEvents = true;

                Console.WriteLine($"Monitoring directories for new map pairs:");
                Console.WriteLine($"  {path1}");
                Console.WriteLine($"  {path2}\n");
            }
        }

        private FileSystemWatcher CreateWatcher(string path)
        {
            return new FileSystemWatcher
            {
                Path = path,
                Filter = "*.mrc",
                NotifyFilter = NotifyFilters.FileName  // Only watch for new files, not writes
            };
        }

        private void OnFileCreated(object sender, FileSystemEventArgs e)
        {
            // Wait a bit to ensure file is fully written
            Thread.Sleep(1000);

            string mapName = Helper.PathToName(e.FullPath);

            lock (lockObj)
            {
                // Skip if already processed
                if (processedMaps.Contains(mapName))
                    return;

                // Check if matching pair exists
                if (TryCreateMapInfo(mapName, out MapFileInfo mapInfo))
                {
                    processedMaps.Add(mapName);
                    Console.WriteLine($"\nNew map pair detected: {mapName}");

                    // Add to rotating pool
                    mapPool.AddNewMap(mapInfo);
                }
            }
        }

        private bool TryCreateMapInfo(string mapName, out MapFileInfo mapInfo)
        {
            mapInfo = null;

            try
            {
                // Get paths for both observations
                string map1Path = Path.Combine(context.WorkingDirectory, options.Observation1Path, mapName + ".mrc");
                string map2Path = Path.Combine(context.WorkingDirectory, options.Observation2Path, mapName + ".mrc");

                // Both files must exist
                if (!File.Exists(map1Path) || !File.Exists(map2Path))
                    return false;

                // Check for combined observation if specified
                string mapCombinedPath = null;
                if (!string.IsNullOrEmpty(options.ObservationCombinedPath))
                {
                    mapCombinedPath = Path.Combine(context.WorkingDirectory, options.ObservationCombinedPath, mapName + ".mrc");
                    if (!File.Exists(mapCombinedPath))
                        return false;
                }

                // Read header for metadata
                MapHeader header = MapHeader.ReadFromFile(map1Path);

                // Determine CTF path
                string ctfPath = null;
                if (!string.IsNullOrEmpty(options.CTFPath))
                {
                    string potentialPath = Path.Combine(context.WorkingDirectory, options.CTFPath, mapName + ".mrc");
                    if (File.Exists(potentialPath))
                        ctfPath = potentialPath;
                }

                // Calculate crop box
                int3 cropBox = new int3(-1);
                if (options.DontKeepDimensions && context.CropBox.X > 0)
                {
                    cropBox = context.CropBox;
                }

                // Create MapFileInfo
                mapInfo = new MapFileInfo
                {
                    Map1Path = map1Path,
                    Map2Path = map2Path,
                    MapCombinedPath = mapCombinedPath,
                    MapName = mapName,
                    PixelSize = header.PixelSize.X,
                    MeanStd = new float2(0, 1), // Will be calculated on first load
                    CropBox = cropBox,
                    SpectrumMultipliers = null, // Will be calculated on first load
                    CTFPath = ctfPath
                };

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing new map {mapName}: {ex.Message}");
                return false;
            }
        }

        public void Dispose()
        {
            watcher1?.Dispose();
            watcher2?.Dispose();
        }
    }
}
