using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace Noise2Map
{
    class Noise2Map
    {
        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
            VirtualConsole.AttachToConsole();

            // Parse and validate configuration
            if (!ConfigurationManager.ParseAndValidate(args, out Options options))
                return;

            // Set up directories
            string programFolder = System.AppContext.BaseDirectory;
            string workingDirectory = Environment.CurrentDirectory;

            // Initialize processing context
            var context = new ProcessingContext
            {
                WorkingDirectory = workingDirectory,
                ProgramFolder = programFolder,
                TrainingDims = new int3(options.WindowSize),
                IsTomo = !string.IsNullOrEmpty(options.CTFPath)
            };

            try
            {
                // Initialize GPU
                GpuManager.Initialize(options);

                // Load mask
                MaskProcessor.LoadMask(context, options);

                // Prepare map metadata and create rotating pool
                List<MapFileInfo> mapInfo = DataPreparator.PrepareMapMetadata(context, options);
                context.MapPool = new RotatingMapPool(mapInfo, options.MaxLoadedMaps, options, context);

                // Train model (or load existing)
                string trainedModelName;
                if (!string.IsNullOrEmpty(options.OldModelName))
                {
                    trainedModelName = options.OldModelName;
                }
                else
                {
                    var coordinator = new TrainingCoordinator(context, options);
                    trainedModelName = coordinator.RunConcurrentTraining(numPreparationThreads: 3, queueCapacity: 6);
                }

                // Denoise maps (using streaming pipeline for memory efficiency)
                var denoiser = new Denoiser(context, options, mapInfo);
                denoiser.LoadModel(trainedModelName);
                denoiser.DenoiseAll();
                denoiser.Dispose();
            }
            finally
            {
                context.Dispose();
            }
        }
    }
}
