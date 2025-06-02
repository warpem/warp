using CommandLine;
using System;
using System.IO;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;


namespace WarpTools.Commands
{
    [VerbGroup("Frame series")]
    [Verb("fs_motion", HelpText = "Estimate motion in frame series, produce aligned averages")]
    [CommandRunner(typeof(MotionFrameseries))]
    class AlignFrameseriesOptions : DistributedOptions
    {
        [Option("range_min", Default = 500, HelpText = "Minimum resolution in Angstrom to consider in fit")]
        public double RangeMin { get; set; }

        [Option("range_max", Default = 10, HelpText = "Maximum resolution in Angstrom to consider in fit")]
        public double RangeMax { get; set; }

        [Option("bfac", Default = -500, HelpText = "Downweight higher spatial frequencies using a B-factor, in Angstrom^2")]
        public double Bfactor { get; set; }

        [Option("grid", HelpText = "Resolution of the motion model grid in X, Y, and temporal dimensions, separated by 'x': e.g. 5x5x40; empty = auto")]
        public string GridDims { get; set; }

        [Option("averages", HelpText = "Export aligned averages")]
        public bool Averages { get; set; }

        [Option("average_halves", HelpText = "Export aligned averages of odd and even frames separately, e.g. for denoiser training")]
        public bool AverageHalves { get; set; }

        [Option("skip_first", Default = 0, HelpText = "Skip first N frames when exporting averages")]
        public int SkipFirst { get; set; }

        [Option("skip_last", Default = 0, HelpText = "Skip last N frames when exporting averages")]
        public int SkipLast { get; set; }
    }

    class MotionFrameseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            AlignFrameseriesOptions CLI = options as AlignFrameseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Set options

            Options.Movement.RangeMin = (decimal)CLI.RangeMin;
            Options.Movement.RangeMax = (decimal)CLI.RangeMax;
            Options.Movement.Bfactor = (decimal)CLI.Bfactor;

            if (!string.IsNullOrEmpty(CLI.GridDims))
            {
                try
                {
                    var Dims = CLI.GridDims.Split('x');

                    Options.Grids.MovementX = int.Parse(Dims[0]);
                    Options.Grids.MovementY = int.Parse(Dims[1]);
                    Options.Grids.MovementZ = int.Parse(Dims[2]);
                }
                catch
                {
                    throw new Exception("Grid dimensions must be specified as XxYxZ, e.g. 5x5x40, or left empty for auto");
                }
            }
            else
            {
                Options.Grids.MovementX = 0;
                Options.Grids.MovementY = 0;
                Options.Grids.MovementZ = 0;
            }

            Options.Export.DoAverage = CLI.Averages;
            Options.Export.DoDenoise = CLI.AverageHalves;
            Options.Export.SkipFirstN = CLI.SkipFirst;
            Options.Export.SkipLastN = CLI.SkipLast;

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
            ProcessingOptionsMovieExport OptionsMovieExport = Options.GetProcessingMovieExport();

            IterateOverItems<Movie>(Workers, CLI, (worker, m) =>
            {
                decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                worker.LoadStack(m.DataPath, ScaleFactor, Options.Import.EERGroupFrames);
                worker.MovieProcessMovement(m.Path, OptionsMovement);
                if (CLI.Averages || CLI.AverageHalves)
                    worker.MovieExportMovie(m.Path, OptionsMovieExport);

                worker.GcCollect();
            });

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            Console.Write("Saving settings...");
            Options.Save(Path.Combine(CLI.OutputProcessing, "align_frameseries.settings"));
            Console.WriteLine(" Done");
        }
    }
}
