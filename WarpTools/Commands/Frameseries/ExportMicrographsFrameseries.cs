using CommandLine;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Frame series")]
    [Verb("fs_export_micrographs", HelpText = "Create aligned averages or half-averages from frame series with previously estimated motion")]
    [CommandRunner(typeof(ExportMicrographsFrameseries))]
    class ExportFrameseriesOptions : DistributedOptions
    {
        [Option("averages", HelpText = "Export aligned averages")]
        public bool Averages { get; set; }

        [Option("average_halves", HelpText = "Export aligned averages of odd and even frames separately, e.g. for denoiser training")]
        public bool AverageHalves { get; set; }

        [Option("skip_first", Default = 0, HelpText = "Skip first N frames when exporting averages")]
        public int SkipFirst { get; set; }

        [Option("skip_last", Default = 0, HelpText = "Skip last N frames when exporting averages")]
        public int SkipLast { get; set; }
    }

    class ExportMicrographsFrameseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ExportFrameseriesOptions CLI = options as ExportFrameseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Set options

            Options.Export.DoAverage = CLI.Averages;
            Options.Export.DoDenoise = CLI.AverageHalves;
            Options.Export.SkipFirstN = CLI.SkipFirst;
            Options.Export.SkipLastN = CLI.SkipLast;

            if (!CLI.Averages && !CLI.AverageHalves)
                throw new Exception("No output types requested");

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            ProcessingOptionsMovieExport OptionsMovieExport = Options.GetProcessingMovieExport();

            IterateOverItems(Workers, CLI, (worker, m) =>
            {
                decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                worker.LoadStack(m.DataPath, ScaleFactor, Options.Import.EERGroupFrames);
                worker.MovieExportMovie(m.Path, OptionsMovieExport);
            });

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            Console.Write("Saving settings...");
            Options.Save(Path.Combine(CLI.OutputProcessing, "export_movies.settings"));
            Console.WriteLine(" Done");
        }
    }
}
