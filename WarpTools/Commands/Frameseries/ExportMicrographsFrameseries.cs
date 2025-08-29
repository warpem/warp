using CommandLine;
using System;
using System.IO;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using WorkerWrapper = Warp.Workers.WorkerWrapper;


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

        [Option("thumbnails", HelpText = "Export thumbnails, scaled so that the long edge has this length in pixels")]
        public int? Thumbnails { get; set; }

        [Option("skip_first", Default = 0, HelpText = "Skip first N frames when exporting averages")]
        public int SkipFirst { get; set; }

        [Option("skip_last", Default = 0, HelpText = "Skip last N frames when exporting averages")]
        public int SkipLast { get; set; }

        [Option("bin_angpix", HelpText = "Downsample the output to have this pixel size; leave empty to use value specified in settings")]
        public double? BinAngpix { get; set; }

        [Option("highpass", HelpText = "Optional high-pass filter to be applied to averages, in Angstroms")]
        public double? Highpass { get; set; }
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
            
            if (CLI.BinAngpix.HasValue && CLI.BinAngpix.Value > 0)
                Options.Import.BinTimes = (decimal)Math.Log2((double)CLI.BinAngpix / (double)Options.Import.PixelSize);

            if (!CLI.Averages && !CLI.AverageHalves)
                throw new Exception("No output types requested");

            if (CLI.Thumbnails.HasValue)
            {
                if (!CLI.Averages)
                    throw new Exception("Can't export thumbnails without exporting averages");
                else if (CLI.Thumbnails.Value <= 0)
                    throw new Exception("Thumbnail size must be a positive integer");
                else if (CLI.Thumbnails.Value % 2 != 0)
                    throw new Exception("Thumbnail size must be an even number");
            }

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            ProcessingOptionsMovieExport OptionsMovieExport = Options.GetProcessingMovieExport();
            
            if (CLI.Highpass.HasValue && CLI.Highpass.Value > 0)
                OptionsMovieExport.HighpassAngstrom = (decimal)CLI.Highpass.Value;

            IterateOverItems<Movie>(Workers, 
                                    CLI, 
                                    (worker, m) =>
            {
                decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                worker.LoadStack(m.DataPath, ScaleFactor, Options.Import.EERGroupFrames);
                worker.MovieExportMovie(m.Path, OptionsMovieExport);

                if (CLI.Thumbnails.HasValue)
                    worker.MovieCreateThumbnail(m.Path, CLI.Thumbnails.Value, 3);

                worker.GcCollect();
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
