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
        /// Denoises all maps using streaming pipeline
        /// </summary>
        public void DenoiseAll()
        {
            if (mapInfoList == null || mapInfoList.Count == 0)
                throw new Exception("No map information available for denoising. MapFileInfo list is required.");

            var pipeline = new DenoisingPipeline(mapInfoList, options, context, model);
            pipeline.ProcessAll();
        }

        public void Dispose()
        {
            model?.Dispose();
        }
    }
}