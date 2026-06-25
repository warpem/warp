using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;


namespace WarpTools.Commands
{
    [VerbGroup("Frame series")]
    [Verb("fs_motion_and_ctf", HelpText = "Estimate motion in frame series, produce aligned averages, estimate CTF – all in one go!")]
    [CommandRunner(typeof(MotionCTFFrameseries))]
    class AlignCTFFrameseriesOptions : DistributedOptions
    {
        #region Motion

        [Option("m_range_min", Default = 500, HelpText = "Minimum resolution in Angstrom to consider in fit")]
        public double MotionRangeMin { get; set; }

        [Option("m_range_max", Default = 10, HelpText = "Maximum resolution in Angstrom to consider in fit")]
        public double MotionRangeMax { get; set; }

        [Option("m_bfac", Default = -500, HelpText = "Downweight higher spatial frequencies using a B-factor, in Angstrom^2")]
        public double MotionBfactor { get; set; }

        [Option("m_grid", HelpText = "Resolution of the motion model grid in X, Y, and temporal dimensions, separated by 'x': e.g. 5x5x40; empty = auto")]
        public string MotionGridDims { get; set; }

        #endregion

        #region CTF

        [Option("c_window", Default = 512, HelpText = "Patch size for CTF estimation in binned pixels")]
        public int CTFWindow { get; set; }

        [Option("c_range_min", Default = 30, HelpText = "Minimum resolution in Angstrom to consider in fit")]
        public double CTFRangeMin { get; set; }

        [Option("c_range_max", Default = 4, HelpText = "Maximum resolution in Angstrom to consider in fit")]
        public double CTFRangeMax { get; set; }

        [Option("c_defocus_min", Default = 0.5, HelpText = "Minimum defocus value in um to explore during fitting")]
        public double CTFZMin { get; set; }

        [Option("c_defocus_max", Default = 5.0, HelpText = "Maximum defocus value in um to explore during fitting")]
        public double CTFZMax { get; set; }


        [Option("c_voltage", Default = 300, HelpText = "Acceleration voltage of the microscope in kV")]
        public int CTFVoltage { get; set; }

        [Option("c_cs", Default = 2.7, HelpText = "Spherical aberration of the microscope in mm")]
        public double? CTFCs { get; set; }

        [Option("c_amplitude", Default = 0.07, HelpText = "Amplitude contrast of the sample, usually 0.07-0.10 for cryo")]
        public double? CTFAmplitude { get; set; }

        [Option("c_fit_phase", HelpText = "Fit the phase shift of a phase plate")]
        public bool CTFPhaseEnable { get; set; }


        [Option("c_use_sum", HelpText = "Use the movie average spectrum instead of the average of individual frames' spectra. " +
                                      "Can help in the absence of an energy filter, or when signal is low.")]
        public bool CTFMovieSumEnable { get; set; }


        [Option("c_grid", HelpText = "Resolution of the defocus model grid in X, Y, and temporal dimensions, separated by 'x': e.g. 5x5x40; empty = auto; Z > 1 is purely experimental")]
        public string CTFGridDims { get; set; }

        #endregion

        #region Output

        [Option("out_averages", HelpText = "Export aligned averages")]
        public bool OutAverages { get; set; }

        [Option("out_average_halves", HelpText = "Export aligned averages of odd and even frames separately, e.g. for denoiser training")]
        public bool OutAverageHalves { get; set; }

        [Option("out_thumbnails", HelpText = "Export thumbnails, scaled so that the long edge has this length in pixels")]
        public int? OutThumbnails { get; set; }

        [Option("out_skip_first", Default = 0, HelpText = "Skip first N frames when exporting averages")]
        public int OutSkipFirst { get; set; }

        [Option("out_skip_last", Default = 0, HelpText = "Skip last N frames when exporting averages")]
        public int OutSkipLast { get; set; }

        #endregion
    }

    class MotionCTFFrameseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            AlignCTFFrameseriesOptions CLI = options as AlignCTFFrameseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Set options

            #region Motion

            Options.Movement.RangeMin = (decimal)CLI.MotionRangeMin;
            Options.Movement.RangeMax = (decimal)CLI.MotionRangeMax;
            Options.Movement.Bfactor = (decimal)CLI.MotionBfactor;

            if (!string.IsNullOrEmpty(CLI.MotionGridDims))
            {
                try
                {
                    var Dims = CLI.MotionGridDims.Split('x');

                    Options.Grids.MovementX = int.Parse(Dims[0]);
                    Options.Grids.MovementY = int.Parse(Dims[1]);
                    Options.Grids.MovementZ = int.Parse(Dims[2]);
                }
                catch
                {
                    throw new Exception("Motion grid dimensions must be specified as XxYxZ, e.g. 5x5x40, or left empty for auto");
                }
            }
            else
            {
                Options.Grids.MovementX = 0;
                Options.Grids.MovementY = 0;
                Options.Grids.MovementZ = 0;
            }

            #endregion

            #region CTF

            Options.CTF.Window = CLI.CTFWindow;
            Options.CTF.RangeMin = (decimal)CLI.CTFRangeMin;
            Options.CTF.RangeMax = (decimal)CLI.CTFRangeMax;
            Options.CTF.ZMin = (decimal)CLI.CTFZMin;
            Options.CTF.ZMax = (decimal)CLI.CTFZMax;

            Options.CTF.Voltage = (int)CLI.CTFVoltage;
            Options.CTF.Cs = (decimal)CLI.CTFCs;
            Options.CTF.Amplitude = (decimal)CLI.CTFAmplitude;

            Options.CTF.DoPhase = CLI.CTFPhaseEnable;
            Options.CTF.UseMovieSum = CLI.CTFMovieSumEnable;

            if (!string.IsNullOrEmpty(CLI.CTFGridDims))
            {
                try
                {
                    var Dims = CLI.CTFGridDims.Split('x');

                    Options.Grids.CTFX = int.Parse(Dims[0]);
                    Options.Grids.CTFY = int.Parse(Dims[1]);
                    Options.Grids.CTFZ = int.Parse(Dims[2]);
                }
                catch
                {
                    throw new Exception("CTF grid dimensions must be specified as XxYxZ, e.g. 5x5x40, or left empty for auto");
                }
            }
            else
            {
                Options.Grids.CTFX = 0;
                Options.Grids.CTFY = 0;
                Options.Grids.CTFZ = 0;
            }

            if (Options.Grids.CTFZ > 1 && Options.CTF.UseMovieSum)
                throw new Exception("CTF grid can't be larger than 1 in Z dimension when using movie sums because they have only 1 frame");

            #endregion

            #region Output

            Options.Export.DoAverage = CLI.OutAverages;
            Options.Export.DoDenoise = CLI.OutAverageHalves;
            Options.Export.SkipFirstN = CLI.OutSkipFirst;
            Options.Export.SkipLastN = CLI.OutSkipLast;

            if (CLI.OutThumbnails.HasValue)
            {
                if (!CLI.OutAverages)
                    throw new Exception("Can't export thumbnails without exporting averages");
                else if (CLI.OutThumbnails.Value <= 0)
                    throw new Exception("Thumbnail size must be a positive integer");
                else if (CLI.OutThumbnails.Value % 2 != 0)
                    throw new Exception("Thumbnail size must be an even number");
            }

            #endregion

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
            ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
            ProcessingOptionsMovieExport OptionsMovieExport = Options.GetProcessingMovieExport();

            // Filter to only process files that haven't been processed or where options changed
            Movie[] OriginalInputSeries = CLI.InputSeries;
            List<Movie> ItemsToProcess = new List<Movie>();
            List<Movie> ItemsSkipped = new List<Movie>();

            foreach (Movie m in OriginalInputSeries)
            {
                m.LoadMeta();

                bool needsMotionProcessing = m.OptionsMovement == null || m.OptionsMovement != OptionsMovement;
                bool needsCTFProcessing = m.OptionsCTF == null || m.OptionsCTF != OptionsCTF;

                if (needsMotionProcessing || needsCTFProcessing)
                {
                    ItemsToProcess.Add(m);
                }
                else
                {
                    ItemsSkipped.Add(m);
                }
            }

            CLI.InputSeries = ItemsToProcess.ToArray();

            if (ItemsSkipped.Count > 0)
            {
                Console.WriteLine($"Skipping {ItemsSkipped.Count} files that were already processed with the same options.");
            }

            if (ItemsToProcess.Count == 0)
            {
                Console.WriteLine("All files have already been processed with the current options. Nothing to do.");

                Console.Write("Saying goodbye to all workers...");
                foreach (var worker in Workers)
                    worker.Dispose();
                Console.WriteLine(" Done");

                return;
            }

            Console.WriteLine($"Processing {ItemsToProcess.Count} files...");

            IterateOverItems<Movie>(Workers, CLI, (worker, m) =>
            {
                decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                worker.LoadStack(m.DataPath, ScaleFactor, Options.Import.EERGroupFrames);
                worker.MovieProcessMovement(m.Path, OptionsMovement);
                if (CLI.OutAverages || CLI.OutAverageHalves)
                    worker.MovieExportMovie(m.Path, OptionsMovieExport);

                if (CLI.OutThumbnails.HasValue)
                    worker.MovieCreateThumbnail(m.Path, CLI.OutThumbnails.Value, 3);

                if (Options.CTF.UseMovieSum)
                    worker.WaitAsyncTasks();

                if (Options.CTF.UseMovieSum && File.Exists(m.AveragePath))
                    worker.LoadStack(m.AveragePath, 1, Options.Import.EERGroupFrames, false);
                worker.MovieProcessCTF(m.Path, OptionsCTF);

                worker.GcCollect();
            });

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            Console.Write("Saving settings...");
            Options.Save(Path.Combine(CLI.OutputProcessing, "align_and_ctf_frameseries.settings"));
            Console.WriteLine(" Done");
        }
    }
}
