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
    [Verb("ts_ctf", HelpText = "Estimate CTF parameters in frame series")]
    [CommandRunner(typeof(CTFTiltseries))]
    class CTFTiltseriesOptions : DistributedOptions
    {
        [Option("window", Default = 512, HelpText = "Patch size for CTF estimation in binned pixels")]
        public int Window { get; set; }

        [Option("range_low", Default = 30, HelpText = "Lowest (worst) resolution in Angstrom to consider in fit")]
        public double RangeMin { get; set; }

        [Option("range_high", Default = 4, HelpText = "Highest (best) resolution in Angstrom to consider in fit")]
        public double RangeMax { get; set; }

        [Option("defocus_min", Default = 0.5, HelpText = "Minimum defocus value in um to explore during fitting (positive = underfocus)")]
        public double ZMin { get; set; }

        [Option("defocus_max", Default = 5.0, HelpText = "Maximum defocus value in um to explore during fitting (positive = underfocus)")]
        public double ZMax { get; set; }


        [Option("voltage", Default = 300, HelpText = "Acceleration voltage of the microscope in kV")]
        public int Voltage { get; set; }

        [Option("cs", Default = 2.7, HelpText = "Spherical aberration of the microscope in mm")]
        public double Cs { get; set; }

        [Option("amplitude", Default = 0.07, HelpText = "Amplitude contrast of the sample, usually 0.07-0.10 for cryo")]
        public double Amplitude { get; set; }

        [Option("fit_phase", HelpText = "Fit the phase shift of a phase plate")]
        public bool PhaseEnable { get; set; }


        [Option("auto_hand", HelpText = "Run defocus handedness estimation based on this many tilt series (e.g. 10), then estimate CTF with the correct handedness")]
        public int AutoHand { get; set; }
    }

    class CTFTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            CTFTiltseriesOptions CLI = options as CTFTiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;
            WorkerWrapper[] Workers = CLI.GetWorkers();

            #region Validate options

            if (CLI.Window < 128)
                throw new Exception("--window must be at least 128");

            if (CLI.RangeMin > 999)
                throw new Exception("--range_low can't be lower than 999 A");

            if (CLI.RangeMax < 1)
                throw new Exception("--range_high can't be higher than 1 A");

            if (CLI.RangeMax <= (double)Options.Import.BinnedPixelSize * 2)
                throw new Exception("--range_high can't be higher than the binned data's Nyquist resolution");

            if (CLI.RangeMax >= CLI.RangeMin)
                throw new Exception("--range_high can't be higher than --range_low");

            if (CLI.ZMin > CLI.ZMax)
                throw new Exception("--defocus_min can't be higher than --defocus_max");

            if (CLI.Voltage <= 0)
                throw new Exception("--voltage must be a positive number");

            if (CLI.Cs <= 0)
                throw new Exception("--cs must be a positive number");

            if (CLI.Amplitude < 0)
                throw new Exception("--amplitude can't be negative");

            if (CLI.Amplitude > 1)
                throw new Exception("--amplitude can't be higher than 1");

            if (CLI.AutoHand < 0)
                throw new Exception("--auto_hand can't be negative");

            #endregion

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

            #endregion

            ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();

            if (CLI.AutoHand == 0)
            {
                IterateOverItems(Workers, CLI, (worker, m) =>
                {
                    worker.TomoProcessCTF(m.Path, OptionsCTF);
                });
            }
            else
            {
                Movie[] AllSeries = CLI.InputSeries;
                Movie[] SeriesToUse = AllSeries.Take(CLI.AutoHand).ToArray();

                CLI.InputSeries = SeriesToUse;

                Console.WriteLine($"Estimating the CTF for {CLI.AutoHand} tilt series to check handedness...");
                IterateOverItems(Workers, CLI, (worker, m) =>
                {
                    worker.TomoProcessCTF(m.Path, OptionsCTF);
                });

                #region Calculate correlation

                // Check if all tilt series have the same handedness
                bool AllSame = CLI.InputSeries.Select(m => ((TiltSeries)m).AreAnglesInverted).Distinct().Count() == 1;
                if (!AllSame)
                    throw new Exception("--auto_hand requested, but not all tilt series have the same handedness going into this procedure.\n" +
                                        "Please use the ts_defocus_hand tool to set all tilt series' handedness to a common value.");

                Console.WriteLine("Checking defocus handedness for all tilt series...");
                Console.Write($"0/{CLI.InputSeries.Length}");

                var Correlations = new List<float>();

                foreach (var series in CLI.InputSeries.Select(m => (TiltSeries)m))
                {
                    // Because metadata was modified by workers, reload it
                    series.LoadMeta();
                    Movie[] TiltMovies = series.TiltMoviePaths.Select(s => new Movie(Path.Combine(series.DataOrProcessingDirectoryName, s))).ToArray();

                    if (TiltMovies.Any(m => m.GridCTFDefocus.Values.Length < 2))
                        throw new Exception("One or more tilt movies don't have local defocus information. " +
                                            "Please run ctf_frameseries on all individual tilt movies using a 2x2x1 grid.");

                    series.VolumeDimensionsPhysical = new float3((float)Options.Tomo.DimensionsX,
                                                                 (float)Options.Tomo.DimensionsY,
                                                                 (float)Options.Tomo.DimensionsZ) * (float)Options.Import.PixelSize;
                    series.ImageDimensionsPhysical = new float2(series.VolumeDimensionsPhysical.X, series.VolumeDimensionsPhysical.Y);

                    float[] GradientsEstimated = new float[series.NTilts];
                    float[] GradientsAssumed = new float[series.NTilts];

                    float3[] Points =
                    {
                        new float3(0, series.VolumeDimensionsPhysical.Y / 2, series.VolumeDimensionsPhysical.Z / 2),
                        new float3(series.VolumeDimensionsPhysical.X, series.VolumeDimensionsPhysical.Y / 2, series.VolumeDimensionsPhysical.Z / 2)
                    };

                    float3[] Projected0 = series.GetPositionInAllTilts(Points[0]).Select(v => v / new float3(series.ImageDimensionsPhysical.X, series.ImageDimensionsPhysical.Y, 1)).ToArray();
                    float3[] Projected1 = series.GetPositionInAllTilts(Points[1]).Select(v => v / new float3(series.ImageDimensionsPhysical.X, series.ImageDimensionsPhysical.Y, 1)).ToArray();

                    for (int t = 0; t < series.NTilts; t++)
                    {
                        float Interp0 = TiltMovies[t].GridCTFDefocus.GetInterpolated(new float3(Projected0[t].X, Projected0[0].Y, 0.5f));
                        float Interp1 = TiltMovies[t].GridCTFDefocus.GetInterpolated(new float3(Projected1[t].X, Projected1[0].Y, 0.5f));
                        GradientsEstimated[t] = Interp1 - Interp0;

                        GradientsAssumed[t] = Projected1[t].Z - Projected0[t].Z;
                    }

                    if (GradientsEstimated.Length > 1)
                    {
                        GradientsEstimated = MathHelper.Normalize(GradientsEstimated);
                        GradientsAssumed = MathHelper.Normalize(GradientsAssumed);
                    }
                    else
                    {
                        GradientsEstimated[0] = Math.Sign(GradientsEstimated[0]);
                        GradientsAssumed[0] = Math.Sign(GradientsAssumed[0]);
                    }

                    float Correlation = MathHelper.DotProduct(GradientsEstimated, GradientsAssumed) / GradientsEstimated.Length;
                    Correlations.Add(Correlation);

                    VirtualConsole.ClearLastLine();
                    Console.Write($"{Correlations.Count}/{CLI.InputSeries.Length}, {Correlations.Average():F3}");
                }
                Console.WriteLine("");

                float CorrelationAll = Correlations.Average();

                Console.WriteLine($"Average correlation: {CorrelationAll:F3}");
                if (CorrelationAll < 0)
                    Console.WriteLine("The average correlation is negative, which means that the defocus handedness needs to be flipped compared to its current state");
                else if (CorrelationAll > 0)
                    Console.WriteLine("The average correlation is positive, which means that the defocus handedness doesn't need flipping compared to its current state");
                else
                    throw new Exception("The average correlation is 0, which shouldn't happen");

                #endregion

                // Go back to the full set of tilt series
                CLI.InputSeries = AllSeries;

                #region Set defocus handedness if necessary

                if (CorrelationAll < 0)
                {
                    Console.WriteLine($"Flipping the defocus handedness for all tilt series...");
                    Console.Write($"0/{CLI.InputSeries.Length}");

                    int NDone = 0;
                    foreach (var series in CLI.InputSeries.Select(m => (TiltSeries)m))
                    {
                        series.AreAnglesInverted = !series.AreAnglesInverted;
                        series.SaveMeta();

                        VirtualConsole.ClearLastLine();
                        Console.Write($"{++NDone}/{CLI.InputSeries.Length}");
                    }

                    Console.WriteLine("");
                }

                #endregion

                Console.WriteLine("Now running CTF estimation for all tilt series with correct defocus handedness...");
                IterateOverItems(Workers, CLI, (worker, m) =>
                {
                    worker.TomoProcessCTF(m.Path, OptionsCTF);
                });
            }

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            Console.Write("Saving settings...");
            Options.Save(Path.Combine(CLI.OutputProcessing, "ctf_tiltseries.settings"));
            Console.WriteLine(" Done");
        }
    }
}
