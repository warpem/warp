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
            programFolder = programFolder.Substring(0, Math.Max(programFolder.LastIndexOf('\\'), programFolder.LastIndexOf('/')) + 1);
            string workingDirectory = Environment.CurrentDirectory + "/";

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

                // Load and prepare data
                DataPreparator.LoadAndPrepareData(context, options);

                // Train model (or load existing)
                var trainer = new ModelTrainer(context, options);
                trainer.Train();
                string trainedModelName = trainer.TrainedModelName;
                trainer.Dispose();

                // Denoise maps
                var denoiser = new Denoiser(context, options);
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
