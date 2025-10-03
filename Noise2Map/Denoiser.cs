using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Handles map denoising using a trained model
    /// </summary>
    public class Denoiser
    {
        private readonly ProcessingContext context;
        private readonly Options options;
        private readonly List<MapFileInfo> mapInfoList;
        private NoiseNet3DTorch model;

        public Denoiser(ProcessingContext context, Options options, List<MapFileInfo> mapInfoList = null)
        {
            this.context = context;
            this.options = options;
            this.mapInfoList = mapInfoList;
        }

        /// <summary>
        /// Loads the trained model
        /// </summary>
        public void LoadModel(string trainedModelName)
        {
            // Adjust batch size for denoising
            options.BatchSize = options.GPUNetwork.Count();

            Console.WriteLine("Loading trained model, " + GPU.GetFreeMemory(options.GPUNetwork.First()) + " MB free.");
            model = new NoiseNet3DTorch(context.TrainingDims,
                                        options.GPUNetwork.ToArray(),
                                        options.BatchSize,
                                        depth: options.MiniModel ? 1 : 2,
                                        progressiveDepth: !options.MiniModel,
                                        maxWidth: options.MiniModel ? 64 : 99999);

            string modelPath = Path.Combine(context.WorkingDirectory, trainedModelName);
            if (!File.Exists(modelPath))
                throw new Exception("Old model could not be found.");

            model.Load(modelPath);
            Console.WriteLine("Loaded trained model, " + GPU.GetFreeMemory(options.GPUNetwork.First()) + " MB remaining.\n");
        }

        /// <summary>
        /// Denoises all prepared maps and saves the results
        /// </summary>
        public void DenoiseAll()
        {
            // Use streaming pipeline if mapInfoList is available (memory-efficient)
            if (mapInfoList != null && mapInfoList.Count > 0)
            {
                var pipeline = new DenoisingPipeline(mapInfoList, options, context, model);
                pipeline.ProcessAll();
                return;
            }

            // Legacy path: denoise from pre-loaded maps
            Directory.CreateDirectory(Path.Combine(context.WorkingDirectory, "denoised"));

            GPU.SetDevice(options.GPUPreprocess);

            for (int imap = 0; imap < context.MapsForDenoising.Count; imap++)
            {
                DenoiseMap(imap);

                if (options.DenoiseSeparately)
                {
                    DenoiseMap2(imap);
                }
            }

            Console.WriteLine("\nAll done!");
        }

        private void DenoiseMap(int imap)
        {
            Console.Write($"Denoising {context.NamesForDenoising[imap]}... ");

            Image map1 = context.MapsForDenoising[imap];
            NoiseNet3DTorch.Denoise(map1, new NoiseNet3DTorch[] { model });

            float2 meanStd = context.MeanStdForDenoising[imap];
            map1.TransformValues(v => v * meanStd.Y + meanStd.X);

            map1.PixelSize = context.PixelSizeForDenoising[imap];

            ApplyMask(map1);

            string savePath = Path.Combine(context.WorkingDirectory, "denoised",
                                          context.NamesForDenoising[imap] + (options.DenoiseSeparately ? "_1" : "") + ".mrc");
            map1.WriteMRC16b(savePath, true);
            map1.Dispose();

            Console.WriteLine("Done. Saved to " + savePath);
        }

        private void DenoiseMap2(int imap)
        {
            Console.Write($"Denoising {context.NamesForDenoising[imap]} (2nd observation)... ");

            Image map2 = context.MapsForDenoising2[imap];
            NoiseNet3DTorch.Denoise(map2, new NoiseNet3DTorch[] { model });

            float2 meanStd = context.MeanStdForDenoising[imap];
            map2.TransformValues(v => v * meanStd.Y + meanStd.X);

            map2.PixelSize = context.PixelSizeForDenoising[imap];

            ApplyMask(map2);

            string savePath = Path.Combine(context.WorkingDirectory, "denoised",
                                          context.NamesForDenoising[imap] + "_2.mrc");
            map2.WriteMRC16b(savePath, true);
            map2.Dispose();

            Console.WriteLine("Done. Saved to " + savePath);
        }

        private void ApplyMask(Image map)
        {
            if (options.MaskOutput)
                map.Multiply(context.Mask);
            else if (context.Mask != null && !options.DontKeepDimensions)
                map.MaskSpherically(map.Dims.X - 32, 16, true);
        }

        public void Dispose()
        {
            model?.Dispose();
        }
    }
}