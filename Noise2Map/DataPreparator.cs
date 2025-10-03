using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Handles data loading and preprocessing for training and denoising
    /// </summary>
    public static class DataPreparator
    {
        /// <summary>
        /// Enumerates all map files without loading them - preprocessing params calculated lazily on first load
        /// </summary>
        public static List<MapFileInfo> PrepareMapMetadata(ProcessingContext context, Options options)
        {
            Console.WriteLine("Enumerating map files:");

            List<MapFileInfo> mapInfoList = new List<MapFileInfo>();
            string[] oddMapPaths = GetOddMapPaths(context, options);

            foreach (var file in oddMapPaths)
            {
                string mapName = Helper.PathToName(file);
                string[] map2Paths = GetMap2Paths(context, options, mapName);

                if (map2Paths == null || map2Paths.Length == 0)
                    continue;

                string mapCombinedPath = GetMapCombinedPath(context, options, mapName);
                if (!string.IsNullOrEmpty(options.ObservationCombinedPath) && mapCombinedPath == null)
                    continue;

                UpdatePixelSizeFromHeader(context, options, file);
                CheckIfFlattening(options, file);

                // Just read header for dimensions and pixel size
                MapHeader header = MapHeader.ReadFromFile(file);

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
                if (!options.DontKeepDimensions)
                    context.BoundsMax = int3.Min(context.BoundsMax, header.Dimensions);

                if (options.DontKeepDimensions && context.CropBox.X > 0)
                {
                    cropBox = context.CropBox;
                }

                // Create MapFileInfo with paths only - spectrum/mean/std calculated on first load
                var info = new MapFileInfo
                {
                    Map1Path = file,
                    Map2Path = map2Paths.First(),
                    MapCombinedPath = mapCombinedPath,
                    MapName = mapName,
                    PixelSize = header.PixelSize.X,
                    MeanStd = new float2(0, 1), // Will be calculated on first load
                    CropBox = cropBox,
                    SpectrumMultipliers = null, // Will be calculated on first load
                    CTFPath = ctfPath
                };

                mapInfoList.Add(info);
            }

            context.Mask?.FreeDevice();

            if (mapInfoList.Count == 0)
                throw new Exception("No maps were found. Please make sure the paths are correct and the names are consistent between the two observations.");

            Console.WriteLine($"Found {mapInfoList.Count} map pairs.\n");
            return mapInfoList;
        }


        private static string[] GetOddMapPaths(ProcessingContext context, Options options)
        {
            if (!string.IsNullOrEmpty(options.Observation1Path))
                return Directory.EnumerateFiles(Path.Combine(context.WorkingDirectory, options.Observation1Path), "*.mrc").ToArray();
            else if (!string.IsNullOrEmpty(options.HalfMap1Path))
                return new string[] { Path.Combine(context.WorkingDirectory, options.HalfMap1Path) };
            else
                throw new Exception("Shouldn't be here!");
        }

        private static string[] GetMap2Paths(ProcessingContext context, Options options, string mapName)
        {
            if (!string.IsNullOrEmpty(options.Observation2Path))
                return Directory.EnumerateFiles(Path.Combine(context.WorkingDirectory, options.Observation2Path), mapName + ".mrc").ToArray();
            else if (!string.IsNullOrEmpty(options.HalfMap1Path))
                return new string[] { Path.Combine(context.WorkingDirectory, options.HalfMap2Path) };
            else
                throw new Exception("Shouldn't be here!");
        }

        private static string GetMapCombinedPath(ProcessingContext context, Options options, string mapName)
        {
            if (string.IsNullOrEmpty(options.ObservationCombinedPath))
                return null;

            string[] mapCombinedPaths = Directory.EnumerateFiles(Path.Combine(context.WorkingDirectory, options.ObservationCombinedPath), mapName + ".mrc").ToArray();
            if (mapCombinedPaths == null || mapCombinedPaths.Length == 0)
                return null;

            return mapCombinedPaths.First();
        }

        private static void UpdatePixelSizeFromHeader(ProcessingContext context, Options options, string file)
        {
            if (options.PixelSize < 0)
            {
                MapHeader header = MapHeader.ReadFromFile(file);
                options.PixelSize = header.PixelSize.X;
                Console.WriteLine($"Set pixel size to {options.PixelSize} based on map header.");
            }
        }

        private static void CheckIfFlattening(Options options, string file)
        {
            if (!options.DontFlatten && options.PixelSize > 0)
            {
                MapHeader header = MapHeader.ReadFromFile(file);
                if (!header.Dimensions.IsCubic)
                {
                    Console.WriteLine("Map is not cubic and thus likely a tomogram. Enabling --dont_flatten_spectrum because flattening only works on cubic volumes");
                    options.DontFlatten = true;
                }
            }
        }

    }
}