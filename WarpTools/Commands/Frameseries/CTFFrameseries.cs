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

using ProcessingOptionsMovieCTF = Warp.Movie.ProcessingOptionsMovieCTF;


namespace WarpTools.Commands
{
    [VerbGroup("Frame series")]
    [Verb("fs_ctf", HelpText = "Estimate CTF parameters in frame series")]
    [CommandRunner(typeof(CTFFrameseries))]
    class CTFFrameseriesOptions : DistributedOptions
    {
        [Option("window", Default = 512, HelpText = "Patch size for CTF estimation in binned pixels")]
        public int Window { get; set; }

        [Option("range_min", Default = 30, HelpText = "Minimum resolution in Angstrom to consider in fit")]
        public double RangeMin { get; set; }

        [Option("range_max", Default = 4, HelpText = "Maximum resolution in Angstrom to consider in fit")]
        public double RangeMax { get; set; }

        [Option("defocus_min", Default = 0.5, HelpText = "Minimum defocus value in um to explore during fitting")]
        public double ZMin { get; set; }

        [Option("defocus_max", Default = 5.0, HelpText = "Maximum defocus value in um to explore during fitting")]
        public double ZMax { get; set; }


        [Option("voltage", Default = 300, HelpText = "Acceleration voltage of the microscope in kV")]
        public int Voltage { get; set; }

        [Option("cs", Default = 2.7, HelpText = "Spherical aberration of the microscope in mm")]
        public double? Cs { get; set; }

        [Option("amplitude", Default = 0.07, HelpText = "Amplitude contrast of the sample, usually 0.07-0.10 for cryo")]
        public double? Amplitude { get; set; }

        [Option("fit_phase", HelpText = "Fit the phase shift of a phase plate")]
        public bool PhaseEnable { get; set; }


        [Option("use_sum", HelpText = "Use the movie average spectrum instead of the average of individual frames' spectra. " +
                                      "Can help in the absence of an energy filter, or when signal is low.")]
        public bool MovieSumEnable { get; set; }


        [Option("grid", HelpText = "Resolution of the defocus model grid in X, Y, and temporal dimensions, separated by 'x': e.g. 5x5x40; empty = auto; Z > 1 is purely experimental")]
        public string GridDims { get; set; }
    }

    class CTFFrameseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            CTFFrameseriesOptions CLI = options as CTFFrameseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Set options

            Options.CTF.Window = CLI.Window;
            Options.CTF.RangeMin = (decimal)CLI.RangeMin;
            Options.CTF.RangeMax = (decimal)CLI.RangeMax;
            Options.CTF.ZMin = (decimal)CLI.ZMin;
            Options.CTF.ZMax = (decimal)CLI.ZMax;

            Options.CTF.Voltage = (int)CLI.Voltage;
            Options.CTF.Cs = (decimal)CLI.Cs;
            Options.CTF.Amplitude = (decimal)CLI.Amplitude;

            Options.CTF.DoPhase = CLI.PhaseEnable;
            Options.CTF.UseMovieSum = CLI.MovieSumEnable;

            if (!string.IsNullOrEmpty(CLI.GridDims))
            {
                try
                {
                    var Dims = CLI.GridDims.Split('x');

                    Options.Grids.CTFX = int.Parse(Dims[0]);
                    Options.Grids.CTFY = int.Parse(Dims[1]);
                    Options.Grids.CTFZ = int.Parse(Dims[2]);
                }
                catch
                {
                    throw new Exception("Grid dimensions must be specified as XxYxZ, e.g. 5x5x40, or left empty for auto");
                }
            }
            else
            {
                Options.Grids.CTFX = 0;
                Options.Grids.CTFY = 0;
                Options.Grids.CTFZ = 0;
            }

            if (Options.Grids.CTFZ > 1 && Options.CTF.UseMovieSum)
                throw new Exception("Grid can't be larger than 1 in Z dimension when using movie sums because they have only 1 frame");

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();

            IterateOverItems<Movie>(
                Workers,
                CLI,
                (worker, m) =>
                {
                    decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                    if (Options.CTF.UseMovieSum && File.Exists(m.AveragePath))
                        worker.LoadStack(m.AveragePath, 1, Options.Import.EERGroupFrames);
                    else
                        worker.LoadStack(m.DataPath, ScaleFactor, Options.Import.EERGroupFrames);
                    
                    worker.MovieProcessCTF(m.Path, OptionsCTF);
                }
            );

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            Console.Write("Saving settings...");
            Options.Save(Path.Combine(CLI.OutputProcessing, "ctf_movies.settings"));
            Console.WriteLine(" Done");
        }
    }
}
