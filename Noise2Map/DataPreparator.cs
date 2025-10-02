using System;
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
        /// Loads and prepares all maps for training and denoising
        /// </summary>
        public static void LoadAndPrepareData(ProcessingContext context, Options options)
        {
            Console.WriteLine("Preparing data:");

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

                Console.Write($"Preparing {mapName}... ");

                ProcessMapPair(context, options, file, map2Paths.First(), mapCombinedPath, mapName);

                Console.WriteLine($" Done.");
                GPU.CheckGPUExceptions();
            }

            context.Mask?.FreeDevice();

            if (context.Maps1.Count == 0)
                throw new Exception("No maps were found. Please make sure the paths are correct and the names are consistent between the two observations.");

            Console.WriteLine("");
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
            if (options.PixelSize < 0 && context.Maps1.Count == 0)
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

        private static void ProcessMapPair(ProcessingContext context, Options options, string map1Path, string map2Path, string mapCombinedPath, string mapName)
        {
            Image map1 = Image.FromFile(map1Path);
            Image map2 = Image.FromFile(map2Path);
            Image mapCombined = mapCombinedPath == null ? null : Image.FromFile(mapCombinedPath);

            float mapPixelSize = map1.PixelSize;

            // Spectral flattening
            if (!options.DontFlatten)
            {
                ApplySpectralFlattening(context, options, ref map1, ref map2, ref mapCombined);
            }

            // Lowpass filtering
            if (options.Lowpass > 0)
            {
                map1.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
                map2.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
                mapCombined?.Bandpass(0, options.PixelSize * 2 / options.Lowpass, true, 0.01f);
            }

            context.OriginalBoxForDenoising.Add(map1.Dims);

            if (!options.DontKeepDimensions)
                context.BoundsMax = int3.Min(context.BoundsMax, map1.Dims);

            // Cropping
            if (options.DontKeepDimensions && context.CropBox.X > 0)
            {
                CropMaps(ref map1, ref map2, ref mapCombined, context.CropBox);
            }

            // Normalization
            float2 meanStd = CalculateMeanStd(map1, map2);
            context.MeanStdForDenoising.Add(meanStd);

            NormalizeMaps(ref map1, ref map2, ref mapCombined, meanStd);

            // Prepare for denoising
            PrepareForDenoising(context, options, map1, map2, mapCombined, mapName, mapPixelSize);

            // Prefilter for cubic interpolation
            GPU.PrefilterForCubic(map1.GetDevice(Intent.ReadWrite), map1.Dims);
            GPU.PrefilterForCubic(map2.GetDevice(Intent.ReadWrite), map2.Dims);

            map1.FreeDevice();
            context.Maps1.Add(map1);
            map2.FreeDevice();
            context.Maps2.Add(map2);

            // Load CTF
            LoadCTF(context, options, mapName);
        }

        private static void ApplySpectralFlattening(ProcessingContext context, Options options, ref Image map1, ref Image map2, ref Image mapCombined)
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

            Image map1Flat = map1.AsSpectrumMultiplied(true, spectrum);
            map1.Dispose();
            map1 = map1Flat;
            map1.FreeDevice();

            Image map2Flat = map2.AsSpectrumMultiplied(true, spectrum);
            map2.Dispose();
            map2 = map2Flat;
            map2.FreeDevice();

            if (mapCombined != null)
            {
                Image mapCombinedFlat = mapCombined.AsSpectrumMultiplied(true, spectrum);
                mapCombined.Dispose();
                mapCombined = mapCombinedFlat;
                mapCombined.FreeDevice();
            }
        }

        private static void CropMaps(ref Image map1, ref Image map2, ref Image mapCombined, int3 cropBox)
        {
            Image map1Cropped = map1.AsPadded(cropBox);
            map1.Dispose();
            map1 = map1Cropped;
            map1.FreeDevice();

            Image map2Cropped = map2.AsPadded(cropBox);
            map2.Dispose();
            map2 = map2Cropped;
            map2.FreeDevice();

            if (mapCombined != null)
            {
                Image mapCombinedCropped = mapCombined.AsPadded(cropBox);
                mapCombined.Dispose();
                mapCombined = mapCombinedCropped;
                mapCombined.FreeDevice();
            }
        }

        private static float2 CalculateMeanStd(Image map1, Image map2)
        {
            Image map1Center = map1.AsPadded(map1.Dims / 2);
            Image map2Center = map2.AsPadded(map2.Dims / 2);
            float2 meanStd = MathHelper.MeanAndStd(Helper.Combine(map1Center.GetHostContinuousCopy(), map2Center.GetHostContinuousCopy()));

            map1Center.Dispose();
            map2Center.Dispose();

            return meanStd;
        }

        private static void NormalizeMaps(ref Image map1, ref Image map2, ref Image mapCombined, float2 meanStd)
        {
            float maxStd = 30;
            map1.TransformValues(v => Math.Max(-maxStd, Math.Min(maxStd, (v - meanStd.X) / meanStd.Y)));
            map2.TransformValues(v => Math.Max(-maxStd, Math.Min(maxStd, (v - meanStd.X) / meanStd.Y)));
            mapCombined?.TransformValues(v => Math.Max(-maxStd, Math.Min(maxStd, (v - meanStd.X) / meanStd.Y)));
        }

        private static void PrepareForDenoising(ProcessingContext context, Options options, Image map1, Image map2, Image mapCombined, string mapName, float mapPixelSize)
        {
            Image forDenoising = (mapCombined == null || options.DenoiseSeparately) ? map1.GetCopy() : mapCombined;
            Image forDenoising2 = options.DenoiseSeparately ? map2.GetCopy() : null;

            if (!options.DenoiseSeparately)
            {
                forDenoising.Add(map2);
                forDenoising.Multiply(0.5f);
            }

            forDenoising.FreeDevice();
            context.MapsForDenoising.Add(forDenoising);
            context.NamesForDenoising.Add(mapName);
            context.PixelSizeForDenoising.Add(mapPixelSize);

            if (options.DenoiseSeparately)
            {
                forDenoising2.FreeDevice();
                context.MapsForDenoising2.Add(forDenoising2);
            }
        }

        private static void LoadCTF(ProcessingContext context, Options options, string mapName)
        {
            if (!string.IsNullOrEmpty(options.CTFPath) &&
                File.Exists(Path.Combine(context.WorkingDirectory, options.CTFPath, mapName + ".mrc")))
            {
                Image mapCTF = Image.FromFile(Path.Combine(context.WorkingDirectory, options.CTFPath, mapName + ".mrc"));
                ProcessCTF(context, options, ref mapCTF, mapName);
                context.MapCTFs.Add(mapCTF);
                Console.Write("Found CTF");
            }
            else
            {
                Image mapCTF = new Image(new int3(128), true);
                mapCTF.TransformValues(v => 1f);
                context.MapCTFs.Add(mapCTF);
            }
        }

        private static void ProcessCTF(ProcessingContext context, Options options, ref Image mapCTF, string mapName)
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
            int ctfDimsX = mapCTF.Dims.X;  // Capture for use in lambda
            int3 trainingDims = context.TrainingDims;  // Capture for use in lambda

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

            mapCTF.WriteMRC(Path.Combine(context.WorkingDirectory, options.CTFPath, mapName + "_scaled.mrc"), true);
        }
    }
}