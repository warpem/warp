using CommandLine;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;


namespace WarpTools.Commands
{
    [VerbGroup("Frame series")]
    [Verb("fs_subtract_membranes", HelpText = "Subtract membrane models from images")]
    [CommandRunner(typeof(SubtractMembranesCommand))]
    class SubtractMembranesOptions : DistributedOptions
    {
        [Option("membrane_subtraction_factor", Default = 0.75, HelpText = "subtract this fraction of the modelled membrane signal from the images")]
        public decimal MembraneSubtractionFactor { get; set; }
    }

    class SubtractMembranesCommand : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            SubtractMembranesOptions CLI = options as SubtractMembranesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Set processing options

            var processingOptions = new ProcessingOptionsSubtractMembranes()
            {
                MembraneSubtractionFactor = CLI.MembraneSubtractionFactor,
            };

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            IterateOverItems<Movie>(Workers, CLI, (worker, m) =>
                {
                    // Verify required files exist
                    string[] membraneModelFiles = Directory.GetFiles(m.MembraneModelsDir)
                        .Where(f => Path.GetFileName(f).StartsWith($"{m.RootName}_membrane"))
                        .ToArray();
                    if (!File.Exists(m.AveragePath) || membraneModelFiles.Length == 0)
                    {
                        Console.WriteLine($"Warning: Skipping {m.Path} - Required input files not found");
                        return;
                    }
                    try
                    {
                        worker.MovieSubtractMembranes(m.Path, processingOptions);
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