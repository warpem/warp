using System;
using System.Collections.Generic;
using System.Globalization;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    class Noise2Map
    {
        private static System.Threading.CancellationTokenSource shutdownTokenSource = new System.Threading.CancellationTokenSource();
        private static bool isOnlineMode = false;

        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
            VirtualConsole.AttachToConsole();

            // Set up graceful shutdown handler
            Console.CancelKeyPress += (sender, e) =>
            {
                e.Cancel = true;  // Prevent immediate termination
                if (isOnlineMode)
                {
                    Console.WriteLine("\n\nShutdown signal received. Finishing current batch and saving model...");
                }
                shutdownTokenSource.Cancel();
            };

            // Parse and validate configuration
            if (!ConfigurationManager.ParseAndValidate(args, out Options options))
                return;

            isOnlineMode = options.OnlineMode;

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

                // In online mode, wait for at least 1 map pair before starting
                if (options.OnlineMode && mapInfo.Count == 0)
                {
                    Console.WriteLine("Online mode: Waiting for at least 1 map pair to appear...");
                    while (mapInfo.Count == 0)
                    {
                        System.Threading.Thread.Sleep(5000);
                        mapInfo = DataPreparator.PrepareMapMetadata(context, options);
                    }
                    Console.WriteLine($"Found {mapInfo.Count} map pair(s). Starting training...\n");
                }

                context.MapPool = new RotatingMapPool(mapInfo, options.MaxLoadedMaps, options, context);

                // Set up file watcher for online mode
                MapFileWatcher watcher = null;
                if (options.OnlineMode)
                {
                    watcher = new MapFileWatcher(context, options, context.MapPool);
                }

                try
                {
                    // Train model (or load existing)
                    string trainedModelName;
                    if (!string.IsNullOrEmpty(options.OldModelName))
                    {
                        trainedModelName = options.OldModelName;
                    }
                    else
                    {
                        var coordinator = new TrainingCoordinator(context, options);
                        trainedModelName = coordinator.RunConcurrentTraining(
                            numPreparationThreads: 3,
                            queueCapacity: 6,
                            externalCancellationToken: shutdownTokenSource.Token);
                    }

                    // Denoise maps (skip in online mode)
                    if (!options.OnlineMode)
                    {
                        var denoiser = new Denoiser(context, options, mapInfo);
                        denoiser.LoadModel(trainedModelName);
                        denoiser.DenoiseAll();
                        denoiser.Dispose();
                    }
                }
                finally
                {
                    watcher?.Dispose();
                }
            }
            finally
            {
                context.Dispose();
            }
        }
    }
}
