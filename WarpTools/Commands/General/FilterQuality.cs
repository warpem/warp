using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;
using Warp;
using System.IO;
using MathNet.Numerics.Statistics;

namespace WarpTools.Commands
{
    [VerbGroup("General")]
    [Verb("filter_quality", HelpText = "Filter frame/tilt series by various quality metrics, or just print out histograms")]
    [CommandRunner(typeof(FilterQuality))]
    class FilterQualityOptions : BaseOptions
    {
        [OptionGroup("Output mode (mutually exclusive)")]
        [Option("histograms", HelpText = "Print a histogram for each quality metric and exit")]
        public bool Histograms { get; set; }

        [Option('o', "output", HelpText = "Path to a .txt file that will contain a list of series that pass the filter criteria")]
        public string OutPath { get; set; }

        [OptionGroup("CTF metrics")]
        [Option("defocus", HelpText = "Defocus in µm: 1 value = min; 2 values = min & max")]
        public IEnumerable<double> Defocus { get; set; }

        [Option("astigmatism", HelpText = "Astigmatism deviation from the dataset's mean, expressed in standard deviations: 1 value = min; 2 values = min & max")]
        public IEnumerable<double> Astigmatism { get; set; }

        [Option("phase", HelpText = "Phase shift as a fraction of π: 1 value = min; 2 values = min & max")]
        public IEnumerable<double> PhaseShift { get; set; }

        [Option("resolution", HelpText = "Resolution estimate based on CTF fit: 1 value = min; 2 values = min & max")]
        public IEnumerable<double> Resolution { get; set; }

        [OptionGroup("Motion metrics")]
        [Option("motion", HelpText = "Average motion in first 1/3 of a frame series in Å: 1 value = min; 2 values = min & max")]
        public IEnumerable<double> Motion { get; set; }

        [OptionGroup("Image content metrics")]
        [Option("crap", HelpText = "Percentage of masked area in an image (frame series only): 1 value = min; 2 values = min & max")]
        public IEnumerable<double> Crap { get; set; }

        [Option("particles", HelpText = "Number of particles: 1 value = min; 2 values = min & max; requires --particles_star to be set")]
        public IEnumerable<int> Particles { get; set; }

        [Option("particles_star", HelpText = "Path to STAR file(s) with particle information; may contain a wildcard that matches multiple files")]
        public IEnumerable<string> ParticlesStar { get; set; }
        
        [OptionGroup("Tilt series metrics")]
        [Option("ntilts", HelpText = "Minimum number of tilts in a tilt series")]
        public int NTilts { get; set; }
    }

    class FilterQuality : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            FilterQualityOptions CLI = options as FilterQualityOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            bool UsingFilters = false;

            Star ParticleTableIn = null;
            List<string> AllParticlePaths = new();

            #region Validate options

            if (string.IsNullOrEmpty(CLI.OutPath) && !CLI.Histograms)
                throw new Exception("No output mode selected");

            if (!string.IsNullOrEmpty(CLI.OutPath) && CLI.Histograms)
                throw new Exception("--histograms and --output are mutually exclusive");

            if (CLI.Defocus.Any())
            {
                if (CLI.Defocus.Count() != 1 && CLI.Defocus.Count() != 2)
                    throw new Exception("--defocus requires 1 or 2 values");

                if (CLI.Defocus.Count() == 1)
                    CLI.Defocus = new[] { CLI.Defocus.First(), double.MaxValue };

                if (CLI.Defocus.First() > CLI.Defocus.Last())
                    throw new Exception($"--defocus requires the first value ({CLI.Defocus.First()}) to be smaller than the second ({CLI.Defocus.Last()})");

                UsingFilters = true;
            }

            if (CLI.Astigmatism.Any())
            {
                if (CLI.Astigmatism.Count() != 1 && CLI.Astigmatism.Count() != 2)
                    throw new Exception("--astigmatism requires 1 or 2 values");

                if (CLI.Astigmatism.Count() == 1)
                    CLI.Astigmatism = new[] { CLI.Astigmatism.First(), double.MaxValue };

                if (CLI.Astigmatism.Any(v => v < 0))
                    throw new Exception("--astigmatism values cannot be negative");

                if (CLI.Astigmatism.First() > CLI.Astigmatism.Last())
                    throw new Exception($"--astigmatism requires the first value ({CLI.Astigmatism.First()}) to be smaller than the second ({CLI.Astigmatism.Last()})");

                UsingFilters = true;
            }

            if (CLI.PhaseShift.Any())
            {
                if (CLI.PhaseShift.Count() != 1 && CLI.PhaseShift.Count() != 2)
                    throw new Exception("--phase requires 1 or 2 values");

                if (CLI.PhaseShift.Count() == 1)
                    CLI.PhaseShift = new[] { CLI.PhaseShift.First(), double.MaxValue };

                if (CLI.PhaseShift.First() > CLI.PhaseShift.Last())
                    throw new Exception($"--phase requires the first value ({CLI.PhaseShift.First()}) to be smaller than the second ({CLI.PhaseShift.Last()})");

                UsingFilters = true;
            }

            if (CLI.Resolution.Any())
            {
                if (CLI.Resolution.Count() != 1 && CLI.Resolution.Count() != 2)
                    throw new Exception("--resolution requires 1 or 2 values");

                if (CLI.Resolution.Count() == 1)
                    CLI.Resolution = new[] { CLI.Resolution.First(), double.MaxValue };

                if (CLI.Resolution.Any(v => v <= 0))
                    throw new Exception("--resolution values must be positive");

                if (CLI.Resolution.First() > CLI.Resolution.Last())
                    throw new Exception($"--resolution requires the first value ({CLI.Resolution.First()}) to be smaller than the second ({CLI.Resolution.Last()})");

                UsingFilters = true;
            }

            if (CLI.Motion.Any())
            {
                if (CLI.SeriesType != SeriesType.Frame)
                    throw new Exception("--motion can only be used with frame series");

                if (CLI.Motion.Count() != 1 && CLI.Motion.Count() != 2)
                    throw new Exception("--motion requires 1 or 2 values");

                if (CLI.Motion.Count() == 1)
                    CLI.Motion = new[] { CLI.Motion.First(), double.MaxValue };

                if (CLI.Motion.Any(v => v < 0))
                    throw new Exception("--motion values cannot be negative");

                if (CLI.Motion.First() > CLI.Motion.Last())
                    throw new Exception($"--motion requires the first value ({CLI.Motion.First()}) to be smaller than the second ({CLI.Motion.Last()})");

                UsingFilters = true;
            }

            if (CLI.Crap.Any())
            {
                if (CLI.SeriesType != SeriesType.Frame)
                    throw new Exception("--crap can only be used with frame series");

                if (CLI.Crap.Count() != 1 && CLI.Crap.Count() != 2)
                    throw new Exception("--crap requires 1 or 2 values");

                if (CLI.Crap.Count() == 1)
                    CLI.Crap = new[] { CLI.Crap.First(), 100 };

                if (CLI.Crap.Any(v => v < 0 || v > 100))
                    throw new Exception("--crap values must be between 0 and 100");

                if (CLI.Crap.First() > CLI.Crap.Last())
                    throw new Exception($"--crap requires the first value ({CLI.Crap.First()}) to be smaller than the second ({CLI.Crap.Last()})");

                UsingFilters = true;
            }

            if (CLI.Particles.Any())
            {
                if (CLI.Particles.Count() != 1 && CLI.Particles.Count() != 2)
                    throw new Exception("--particles requires 1 or 2 values");

                if (CLI.Particles.Count() == 1)
                    CLI.Particles = new[] { CLI.Particles.First(), int.MaxValue };

                if (CLI.Particles.Any(v => v < 0))
                    throw new Exception("--particles values cannot be negative");

                if (CLI.Particles.First() > CLI.Particles.Last())
                    throw new Exception($"--particles requires the first value ({CLI.Particles.First()}) to be smaller than the second ({CLI.Particles.Last()})");

                UsingFilters = true;
            }

            #region Discover STAR files

            if (CLI.ParticlesStar.Any() && (CLI.Histograms || CLI.Particles.Any()))
                foreach (var Path in CLI.ParticlesStar)
                {
                    if (Path.Contains('*'))
                    {
                        string Directory = System.IO.Path.GetDirectoryName(Path);
                        string Pattern = System.IO.Path.GetFileName(Path);

                        foreach (var File in System.IO.Directory.GetFiles(Directory, Pattern))
                            AllParticlePaths.Add(File);
                    }
                    else
                    {
                        AllParticlePaths.Add(Path);
                    }
                }

            if (CLI.Particles.Any())
            {
                if (!CLI.ParticlesStar.Any())
                    throw new Exception("--particles requires --particles_star to be set");

                if (!AllParticlePaths.Any())
                    throw new Exception("Particle statistics requested, but no STAR files found");
            }

            #endregion

            #endregion

            #region Figure out and load STAR files with particles

            if (AllParticlePaths.Any() && (CLI.Histograms || (CLI.Particles != null && CLI.Particles.Any())))
            {
                Console.WriteLine("Particle statistics requested, so will attempt to read particle STAR files...");

                List<Star> AllTablesIn = new List<Star>();
                List<Star> AllOpticsIn = new List<Star>();

                bool Relion3 = false;
                bool PerItemStar = false;

                if (AllParticlePaths.Count == 0)
                    throw new Exception("No STAR files found");

                Console.WriteLine($"Found {AllParticlePaths.Count} STAR files");

                // Figure out if we're dealing with a RELION 3-style STAR file, and if files are per-item
                {
                    Relion3 = Star.ContainsTable(AllParticlePaths.First(), "particles");

                    Star FirstTable = Relion3 ? new Star(AllParticlePaths.First(), "particles") : new Star(AllParticlePaths.First());

                    if (!FirstTable.HasColumn("rlnMicrographName"))
                    {
                        Console.Write("No rlnMicrographName column found, trying to match file names to movies... ");

                        if (CLI.InputSeries.Any(s => Path.GetFileNameWithoutExtension(AllParticlePaths.First()).Contains(s.RootName)))
                        {
                            Console.WriteLine("Succeeded");
                            PerItemStar = true;
                        }
                        else
                        {
                            Console.WriteLine("Failed");
                            throw new Exception("Please make sure the STAR files contain a column named rlnMicrographName, " +
                                                "or that the STAR file names are of the format {movie name}{suffix}.star");
                        }

                        AllParticlePaths = AllParticlePaths.Where(p => CLI.InputSeries.Any(s => Path.GetFileNameWithoutExtension(p).Contains(s.RootName))).ToList();
                        Console.WriteLine($"Found {AllParticlePaths.Count} matching STAR files");
                    }
                }

                Console.Write($"Reading STAR files... 0/{AllParticlePaths.Count}");
                Helper.ForCPU(0, AllParticlePaths.Count, 8, null, (istar, threadID) =>
                {
                    Star ThisTable = Relion3 ? new Star(AllParticlePaths[istar], "particles") : new Star(AllParticlePaths[istar]);
                    Star ThisOptics = Relion3 ? new Star(AllParticlePaths[istar], "optics") : null;

                    if (PerItemStar)
                    {
                        string StarName = Path.GetFileNameWithoutExtension(AllParticlePaths[istar]);
                        string MicrographName = Path.GetFileName(CLI.InputSeries.First(s => StarName.Contains(s.RootName)).DataPath);

                        ThisTable.AddColumn("rlnMicrographName", MicrographName);
                    }

                    lock (AllTablesIn)
                    {
                        AllTablesIn.Add(ThisTable);
                        if (ThisOptics != null)
                            AllOpticsIn.Add(ThisOptics);

                        VirtualConsole.ClearLastLine();
                        Console.Write($"Reading STAR files... {AllTablesIn.Count}/{AllParticlePaths.Count}");
                    }
                }, null);

                if (AllTablesIn.Any())
                    ParticleTableIn = new Star(AllTablesIn.ToArray());
            }

            #endregion

            if (CLI.Histograms && UsingFilters)
                Console.WriteLine("\nHistogram mode requested, but also some filter ranges specified,\n" +
                                  "histograms will be calculated only for items passing the filters\n");

            List<Movie> FilteredSeries = CLI.InputSeries.ToList();

            #region Perform filtering

            if (CLI.Defocus.Any())
            {
                string Min = CLI.Defocus.First().ToString("F2");
                string Max = CLI.Defocus.Last() == double.MaxValue ? "Inf" : CLI.Defocus.Last().ToString("F2");
                Console.Write($"Filtering by defocus, {Min} - {Max} µm: ");

                int Before = FilteredSeries.Count;
                FilteredSeries = FilteredSeries.Where(s => s.CTF != null && 
                                                           (double)s.CTF.Defocus >= CLI.Defocus.First() && 
                                                           (double)s.CTF.Defocus <= CLI.Defocus.Last()).ToList();

                Console.WriteLine($"{Before - FilteredSeries.Count} removed, {FilteredSeries.Count} remain");
            }

            float2 AstigmatismMean = new float2();
            float AstigmatismStd = 0;
            Dictionary<Movie, float> AstigmatismDeviations = new();
            {
                float2[] AstigmatismVectors = FilteredSeries.Select(s =>
                    new float2(MathF.Cos((float)s.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)s.CTF.DefocusDelta,
                               MathF.Sin((float)s.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)s.CTF.DefocusDelta)).ToArray();

                AstigmatismMean = MathHelper.Mean(AstigmatismVectors);
                AstigmatismStd = (float)Math.Sqrt(AstigmatismVectors.Select(v => (v - AstigmatismMean).LengthSq()).Mean());

                for (int i = 0; i < FilteredSeries.Count; i++)
                    AstigmatismDeviations.Add(FilteredSeries[i], 
                                              (AstigmatismVectors[i] - AstigmatismMean).Length() / MathF.Max(1e-16f, AstigmatismStd));
            }

            if (CLI.Astigmatism.Any())
            {
                string Min = CLI.Astigmatism.First().ToString("F2");
                string Max = CLI.Astigmatism.Last() == double.MaxValue ? "Inf" : CLI.Astigmatism.Last().ToString("F2");
                Console.Write($"Filtering by astigmatism deviation from mean, {Min} - {Max} σ: ");

                int Before = FilteredSeries.Count;
                float AstigmatismStdBefore = AstigmatismStd;
                FilteredSeries = FilteredSeries.Where(s => AstigmatismDeviations[s] >= CLI.Astigmatism.First() &&
                                                                          AstigmatismDeviations[s] <= CLI.Astigmatism.Last()).ToList();

                Console.WriteLine($"{Before - FilteredSeries.Count} removed, {FilteredSeries.Count} remain");

                float2[] AstigmatismVectors = FilteredSeries.Select(s =>
                    new float2(MathF.Cos((float)s.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)s.CTF.DefocusDelta,
                               MathF.Sin((float)s.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)s.CTF.DefocusDelta)).ToArray();

                AstigmatismMean = MathHelper.Mean(AstigmatismVectors);
                AstigmatismStd = (float)Math.Sqrt(AstigmatismVectors.Select(v => (v - AstigmatismMean).LengthSq()).Mean());

                AstigmatismDeviations.Clear();
                for (int i = 0; i < FilteredSeries.Count; i++)
                    AstigmatismDeviations.Add(FilteredSeries[i],
                                              (AstigmatismVectors[i] - AstigmatismMean).Length() / MathF.Max(1e-16f, AstigmatismStd));

                Console.WriteLine($"Astigmatism σ changed from {AstigmatismStdBefore} to {AstigmatismStd} µm");
            }

            if (CLI.PhaseShift.Any())
            {
                string Min = CLI.PhaseShift.First().ToString("F2");
                string Max = CLI.PhaseShift.Last() == double.MaxValue ? "Inf" : CLI.PhaseShift.Last().ToString("F2");
                Console.Write($"Filtering by phase shift, {Min} - {Max} π: ");

                int Before = FilteredSeries.Count;
                FilteredSeries = FilteredSeries.Where(s => s.CTF != null && 
                                                           (double)s.CTF.PhaseShift >= CLI.PhaseShift.First() && 
                                                           (double)s.CTF.PhaseShift <= CLI.PhaseShift.Last()).ToList();

                Console.WriteLine($"{Before - FilteredSeries.Count} removed, {FilteredSeries.Count} remain");
            }

            if (CLI.Resolution.Any())
            {
                string Min = CLI.Resolution.First().ToString("F1");
                string Max = CLI.Resolution.Last() == double.MaxValue ? "Inf" : CLI.Resolution.Last().ToString("F1");
                Console.Write($"Filtering by CTF resolution, {Min} - {Max} Å: ");

                int Before = FilteredSeries.Count;
                FilteredSeries = FilteredSeries.Where(s => s.CTF != null && 
                                                           (double)s.CTFResolutionEstimate >= CLI.Resolution.First() && 
                                                           (double)s.CTFResolutionEstimate <= CLI.Resolution.Last()).ToList();

                Console.WriteLine($"{Before - FilteredSeries.Count} removed, {FilteredSeries.Count} remain");
            }

            if (CLI.Motion.Any())
            {
                string Min = CLI.Motion.First().ToString("F1");
                string Max = CLI.Motion.Last() == double.MaxValue ? "Inf" : CLI.Motion.Last().ToString("F1");
                Console.Write($"Filtering by motion in first 1/3 frames, {Min} - {Max} Å: ");

                int Before = FilteredSeries.Count;
                FilteredSeries = FilteredSeries.Where(s => (double)s.MeanFrameMovement >= CLI.Motion.First() && 
                                                           (double)s.MeanFrameMovement <= CLI.Motion.Last()).ToList();

                Console.WriteLine($"{Before - FilteredSeries.Count} removed, {FilteredSeries.Count} remain");
            }

            if (CLI.Crap.Any())
            {
                string Min = CLI.Crap.First().ToString("F0");
                string Max = CLI.Crap.Last().ToString("F0");
                Console.Write($"Filtering by masked area, {Min} - {Max} %: ");

                int Before = FilteredSeries.Count;
                FilteredSeries = FilteredSeries.Where(s => (double)Math.Max(0, s.MaskPercentage) >= CLI.Crap.First() && 
                                                           (double)Math.Max(0, s.MaskPercentage) <= CLI.Crap.Last()).ToList();

                Console.WriteLine($"{Before - FilteredSeries.Count} removed, {FilteredSeries.Count} remain");
            }

            Dictionary<Movie, int> ParticleCounts = new();
            if (ParticleTableIn != null)
                foreach (var s in FilteredSeries)
                    ParticleCounts.Add(s, ParticleTableIn.CountRows("rlnMicrographName", v => v.Contains(s.RootName)));

            if (CLI.Particles.Any() && ParticleTableIn != null)
            {
                string Min = CLI.Particles.First().ToString();
                string Max = CLI.Particles.Last() == int.MaxValue ? "Inf" : CLI.Particles.Last().ToString();
                Console.Write($"Filtering by number of particles, {Min} - {Max}: ");

                int Before = FilteredSeries.Count;
                FilteredSeries = FilteredSeries.Where(s =>
                {
                    int NRows = ParticleCounts[s];
                    return NRows >= CLI.Particles.First() &&
                           NRows <= CLI.Particles.Last();
                }).ToList();

                Console.WriteLine($"{Before - FilteredSeries.Count} removed, {FilteredSeries.Count} remain");
            }

            if (CLI.NTilts != null)
            {
                Console.Write($"Filtering out tilt series with less than {CLI.NTilts.ToString()} tilts");
                FilteredSeries = FilteredSeries.Where(s => (s as TiltSeries).NTilts >= CLI.NTilts).ToList();
            }

            #endregion

            #region Write out file list if necessary

            if (!CLI.Histograms)
            {
                if (Path.GetExtension(CLI.OutPath).ToLower() != ".star")
                    File.WriteAllLines(CLI.OutPath, FilteredSeries.Select(s => s.Path));
                else
                    new Star(new[] { FilteredSeries.Select(s => s.Path).ToArray() }, "rlnMicrographPath").Save(CLI.OutPath);

                Console.WriteLine($"{FilteredSeries.Count} items written to {CLI.OutPath}, {CLI.InputSeries.Length - FilteredSeries.Count} items removed");

                return;
            }

            #endregion

            #region Calculate histograms

            Console.Write("\nCalculating histograms...");
            Dictionary<string, object> Histograms = new();

            Histograms.Add("Defocus (µm)", CalculateHistogram1D(FilteredSeries.Select(s => (float)s.CTF.Defocus).ToArray(), 17));

            {
                float2[] AstigmatismVectors = FilteredSeries.Select(s =>
                    new float2(MathF.Cos((float)s.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)s.CTF.DefocusDelta,
                               MathF.Sin((float)s.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)s.CTF.DefocusDelta)).ToArray();

                Histograms.Add("Astigmatism (µm)", CalculateSymmetricHistogram2D(AstigmatismVectors, 17));
            }

            Histograms.Add("Phase shift (π)", CalculateHistogram1D(FilteredSeries.Select(s => (float)s.CTF.PhaseShift).ToArray(), 17));

            Histograms.Add("CTF resolution (Å)", CalculateHistogram1D(FilteredSeries.Select(s => (float)s.CTFResolutionEstimate).ToArray(), 17));

            if (CLI.SeriesType == SeriesType.Frame)
                Histograms.Add("Motion in first 1/3 (Å)", CalculateHistogram1D(FilteredSeries.Select(s => (float)s.MeanFrameMovement).ToArray(), 17));

            if (CLI.SeriesType == SeriesType.Frame)
                Histograms.Add("Masked area (%)", CalculateHistogram1D(FilteredSeries.Select(s => (float)Math.Max(0, s.MaskPercentage)).ToArray(), 17));

            if (ParticleCounts.Any())
                Histograms.Add("Particles", CalculateHistogram1D(FilteredSeries.Select(s => (float)ParticleCounts[s]).ToArray(), 17));

            Console.WriteLine(" Done");

            #endregion

            #region Print histograms

            foreach (var h in Histograms)
            {
                if (h.Value is HistogramBin1D[] h1d)
                {
                    Console.WriteLine($"\n{h.Key}:");
                    PrintHistogram1D(h1d, 80);
                }
                else if (h.Value is HistogramBin2D[][] h2d)
                {
                    Console.WriteLine($"\n{h.Key}: max = {h2d.Select(r => r.Select(c => c.Count).Max()).Max()}");
                    PrintHistogram2D(h2d);
                }
            }

            #endregion
        }

        HistogramBin1D[] CalculateHistogram1D(float[] values, int nbins)
        {
            float Min = values.Min();
            float Max = values.Max();

            if (Min == Max)
                return new HistogramBin1D[] { new HistogramBin1D() { Min = Min, Max = Max, Count = values.Length } };

            float StepSize = (Max - Min) / nbins;

            HistogramBin1D[] bins = new HistogramBin1D[nbins];

            for (int i = 0; i < nbins; i++)
            {
                bins[i].Min = Min + i * StepSize;
                bins[i].Max = Min + (i + 1) * StepSize;
            }

            foreach (var v in values)
            {
                int bin = Math.Max(0, Math.Min(nbins - 1, (int)((v - Min) / StepSize)));
                bins[bin].Count++;
            }

            return bins;
        }

        HistogramBin2D[][] CalculateSymmetricHistogram2D(float2[] values, int nbins)
        {
            float Max = values.Max(v => MathF.Abs(MathF.Max(v.X, v.Y)));
            float Min = -Max;

            if (Min == Max)
                return new HistogramBin2D[][] { new HistogramBin2D[] { new HistogramBin2D() { Min = new float2(Min, Min), Max = new float2(Max, Max), Count = values.Length } } };

            float StepSize = (Max - Min) / nbins;

            HistogramBin2D[][] bins = Helper.ArrayOfFunction(i => new HistogramBin2D[nbins], nbins);

            for (int j = 0; j < nbins; j++)
                for (int i = 0; i < nbins; i++)
                {
                    bins[j][i].Min = new float2(Min + i * StepSize, Min + j * StepSize);
                    bins[j][i].Max = new float2(Min + (i + 1) * StepSize, Min + (j + 1) * StepSize);
                }

            foreach (var v in values)
            {
                int i = Math.Max(0, Math.Min(nbins - 1, (int)((v.X - Min) / StepSize)));
                int j = Math.Max(0, Math.Min(nbins - 1, (int)((v.Y - Min) / StepSize)));
                bins[j][i].Count++;
            }

            return bins;
        }

        void PrintHistogram1D(HistogramBin1D[] bin, int maxLength)
        {
            int MaxCount = bin.Max(b => b.Count);
            int MaxCountDigits = MaxCount.ToString().Length;

            string SignificantDigits = "F" + CalculateSignificantDigits(bin[0].Max - bin[0].Min);

            int MaxBinDigits = bin.Max(b => b.Max.ToString(SignificantDigits).Length);

            foreach (var b in bin)
            {
                Console.Write($"{b.Min.ToString(SignificantDigits).PadLeft(MaxBinDigits)} - {b.Max.ToString(SignificantDigits).PadRight(MaxBinDigits)}: ");
                int Length = (int)Math.Round((double)b.Count / MaxCount * maxLength);
                Console.WriteLine($"{new string('█', Length)} {b.Count}");
            }
        }

        void PrintHistogram2D(HistogramBin2D[][] bins)
        {
            int MaxCount = bins.Max(row => row.Max(bin => bin.Count));
            char[] brightnessLevels = new char[] { ' ', '░', '▒', '▓', '█' }; // Lightest to darkest

            string SignificantDigits = "F" + CalculateSignificantDigits(bins[0][0].Max.X - bins[0][0].Min.X);
            int MaxBinDigits = bins.Max(row => row.Max(b => b.Max.X.ToString(SignificantDigits).Length));

            for (int j = bins.Length - 1; j >= 0; j--)
            {
                Console.Write($"{bins[j][0].Min.Y.ToString(SignificantDigits).PadLeft(MaxBinDigits)} - {bins[j][0].Max.Y.ToString(SignificantDigits).PadRight(MaxBinDigits)}: |");

                for (int i = 0; i < bins[j].Length; i++)
                {
                    float ExactLevel = (float)bins[j][i].Count / MaxCount * (brightnessLevels.Length - 1);
                    if (ExactLevel > 0.5f)
                    {
                        int level = (int)Math.Round((double)bins[j][i].Count / MaxCount * (brightnessLevels.Length - 1));
                        Console.Write(brightnessLevels[level]);
                        Console.Write(brightnessLevels[level]);
                    }
                    else if (ExactLevel > 0.25)
                    {
                        Console.Write("··");
                    }
                    else if (ExactLevel > 0.0)
                    {
                        Console.Write("· ");
                    }
                    else
                    {
                        Console.Write("  ");
                    }
                }

                Console.WriteLine('|');
            }
        }

        int CalculateSignificantDigits(float stepSize)
        {
            if (stepSize == 0)
                return 1;

            return Math.Max(0, (int)MathF.Ceiling(-MathF.Log10(stepSize)));
        }
    }
    
    struct HistogramBin1D
    {
        public float Min;
        public float Max;
        public int Count;
    }

    struct HistogramBin2D
    {
        public float2 Min;
        public float2 Max;
        public int Count;
    }
}
