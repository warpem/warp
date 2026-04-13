using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_apply_denoiser", HelpText = "Denoise reconstructed tomograms using a trained Noise2Map model")]
    [CommandRunner(typeof(ApplyDenoiserTiltseries))]
    class ApplyDenoiserTiltseriesOptions : DistributedOptions
    {
        [Option("model", Required = true, HelpText = "Path to the trained Noise2Map model file")]
        public string ModelPath { get; set; }

        [Option("window", Default = 64, HelpText = "Window size used during model training (e.g. 64)")]
        public int WindowSize { get; set; }

        [Option("batch", Default = 4, HelpText = "Batch size for denoising (adjust based on available GPU memory)")]
        public int BatchSize { get; set; }

        [Option("angpix", Required = true, HelpText = "Pixel size of the reconstruction to denoise (must match an existing reconstruction)")]
        public double AngPix { get; set; }
    }

    class ApplyDenoiserTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ApplyDenoiserTiltseriesOptions CLI = options as ApplyDenoiserTiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (!File.Exists(CLI.ModelPath))
                throw new Exception($"Model file not found at {CLI.ModelPath}");

            if (CLI.WindowSize <= 0 || CLI.WindowSize % 2 != 0)
                throw new Exception("Window size must be a positive even number");

            if (CLI.BatchSize <= 0)
                throw new Exception("Batch size must be positive");

            if (CLI.AngPix <= 0)
                throw new Exception("Pixel size must be positive");

            #endregion

            #region Create processing options

            var OptionsDenoise = new ProcessingOptionsTomoDenoise
            {
                PixelSize = (decimal)CLI.AngPix
            };

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            // Load denoiser model on all workers
            Console.Write("Loading denoiser model on all workers... ");
            Parallel.ForEach(Workers, worker =>
            {
                worker.LoadTomoDenoiser(CLI.ModelPath, new int3(CLI.WindowSize), CLI.BatchSize);
            });
            Console.WriteLine("Done");

            try
            {
                IterateOverItems<TiltSeries>(Workers, CLI, (worker, m) =>
                {
                    worker.TomoDenoise(m.Path, OptionsDenoise);
                });
            }
            finally
            {
                // No need to drop model explicitly since we're shutting down all workers anyway

                Console.Write("Saying goodbye to all workers...");
                foreach (var worker in Workers)
                    worker.Dispose();
                Console.WriteLine(" Done");
            }
        }
    }
}