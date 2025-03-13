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
                }
            );

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