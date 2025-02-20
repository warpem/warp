using CommandLine;
using System;
using System.IO;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;


namespace WarpTools.Commands
{
    [VerbGroup("Frame series")]
    [Verb("fs_trace_membranes", HelpText = "Model membranes in images")]
    [CommandRunner(typeof(TraceMembranesCommand))]
    class TraceMembranesOptions : DistributedOptions
    {
        [Option("high_res", Default = 300, HelpText = "High resolution limit in Angstrom")]
        public double HighResolutionLimit { get; set; }

        [Option("low_res", Default = 20, HelpText = "Low resolution limit in Angstrom")]
        public double LowResolutionLimit { get; set; }

        [Option("rolloff", Default = 600, HelpText = "Rolloff width in Angstrom")]
        public double RolloffWidth { get; set; }

        [Option("membrane_width", Default = 60, HelpText = "Membrane half-width in Angstrom")]
        public double MembraneHalfWidth { get; set; }

        [Option("edge_softness", Default = 30, HelpText = "Soft edge width in Angstrom")]
        public double MembraneEdgeSoftness { get; set; }

        [Option("spline_spacing", Default = 15, HelpText = "Spline point spacing in Angstrom")]
        public double SplinePointSpacing { get; set; }

        [Option("refinement_iters", Default = 2, HelpText = "Number of refinement iterations")]
        public int RefinementIterations { get; set; }

        [Option("min_pixels", Default = 20, HelpText = "Minimum component size in pixels")]
        public int MinimumComponentPixels { get; set; }
    }

    class TraceMembranesCommand : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            TraceMembranesOptions CLI = options as TraceMembranesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Set processing options
            
            var processingOptions = new ProcessingOptionsTraceMembranes()
            {
                HighResolutionLimit = (decimal)CLI.HighResolutionLimit,
                LowResolutionLimit = (decimal)CLI.LowResolutionLimit,
                RolloffWidth = (decimal)CLI.RolloffWidth,
                MembraneHalfWidth = (decimal)CLI.MembraneHalfWidth,
                MembraneEdgeSoftness = (decimal)CLI.MembraneEdgeSoftness,
                SplinePointSpacing = (decimal)CLI.SplinePointSpacing,
                RefinementIterations = CLI.RefinementIterations,
                MinimumComponentPixels = CLI.MinimumComponentPixels
            };

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            IterateOverItems<Movie>(Workers, CLI, (worker, m) =>
            {
                // Verify required files exist
                if (!File.Exists(m.AveragePath) || !File.Exists(m.MembraneSegmentationPath))
                {
                    Console.WriteLine($"Warning: Skipping {m.Path} - Required input files not found");
                    return;
                }

                try
                {
                    worker.MovieTraceMembranes(m.Path, processingOptions);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing {m.Path}: {ex.Message}");
                }
            });

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            Console.Write("Saving settings...");
            Options.Save(Path.Combine(CLI.OutputProcessing, "trace_membranes.settings"));
            Console.WriteLine(" Done");
        }
    }
}
