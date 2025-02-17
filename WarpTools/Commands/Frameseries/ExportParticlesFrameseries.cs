using CommandLine;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Warp;
using Warp.Headers;
using Warp.Tools;
using ProcessingOptionsParticleExport = Warp.Movie.ProcessingOptionsParticleExport;


namespace WarpTools.Commands.Frameseries
{
    [VerbGroup("Frame series")]
    [Verb("fs_export_particles", HelpText = "Extract particles from tilt series")]
    [CommandRunner(typeof(ExportParticlesFrameseries))]
    class ExportParticlesFrameseriesOptions : DistributedOptions
    {
        [OptionGroup("STAR files with particle coordinates")]
        [Option('i', "input", Required = true, HelpText = "Path to folder containing the STAR files; or path to a single STAR file")]
        public string StarFolder { get; set; }

        [Option("patterns", HelpText = "Space-separated list of file name search patterns or STAR file names when --star is a folder")]
        public IEnumerable<string> StarPattern { get; set; }

        [OptionGroup("Output")]
        [Option('o', "output", Required = true, HelpText = "Where to write the STAR file containing information about the exported particles")]
        public string StarOut { get; set; }

        [Option("suffix_out", Default = "", HelpText = "Suffix to add at the end of each stack's name; the full name will be [movie name][--suffix_out].mrcs")]
        public string SuffixOut { get; set; }

        [Option("angpix_out", HelpText = "Pixel size the extracted particles will be scaled to; leave out to use binned pixel size from input settings")]
        public double? AngpixOut { get; set; }

        [Option("box", Required = true, HelpText = "Particle box size in pixels")]
        public int BoxSize { get; set; }

        [Option("diameter", Required = true, HelpText = "Particle diameter in Angstrom")]
        public int Diameter { get; set; }

        [Option("relative_output_paths", HelpText = "Make paths in output STAR file relative to the location of the STAR file. They will be relative to the working directory otherwise.")]
        public bool OutputPathsRelativeToStarFile { get; set; }

        [OptionGroup("Export type (REQUIRED, mutually exclusive)")]
        [Option("averages", HelpText = "Export particle averages; mutually exclusive with other export types")]
        public bool ExportAverages { get; set; }

        [Option("halves", HelpText = "Export particle half-averages e.g. for denoising; mutually exclusive with other export types")]
        public bool ExportAverageHalves { get; set; }

        [Option("only_star", HelpText = "Don't export, only write out STAR table; mutually exclusive with other export types")]
        public bool ExportStar { get; set; }

        [OptionGroup("Coordinate scaling")]
        [Option("angpix_coords", Required = true, HelpText = "Pixel size for the input coordinates")]
        public double AngpixCoords { get; set; }

        [Option("angpix_shifts", HelpText = "Pixel size for refined shifts if not given in Angstrom (when using rlnOriginX instead of rlnOriginXAngst)")]
        public double? AngpixShifts { get; set; }

        //[Option("from_averages", HelpText = "Extract from averages rather than movies to save a lot of time; requires averages; affects CTF accuracy if local motion was corrected")]
        //public bool FromAverages { get; set; }

        [OptionGroup("Expert options")]
        [Option("dont_invert", HelpText = "Don't invert contrast, e.g. for negative stain data")]
        public bool DontInvert { get; set; }

        [Option("dont_normalize", HelpText = "Don't normalize background (RELION will complain!)")]
        public bool DontNormalize { get; set; }

        [Option("dont_center", HelpText = "Don't re-center particles based on refined shifts")]
        public bool DontCenter { get; set; }

        [Option("flip_phases", HelpText = "Pre-flip phases in bigger box to avoid signal loss due to delocalization")]
        public bool PreflipPhases { get; set; }

        [Option("keep_ctf", HelpText = "Keep CTF information from STAR inputs")]
        public bool KeepCTF { get; set; }

        [Option("skip_first_frames", Default = 0, HelpText = "Skip first N frames")]
        public int SkipFirst { get; set; }

        [Option("skip_last_frames", Default = 0, HelpText = "Skip last N frames")]
        public int SkipLast { get; set; }
    }

    class ExportParticlesFrameseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ExportParticlesFrameseriesOptions CLI = options as ExportParticlesFrameseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (CLI.StarPattern == null || !CLI.StarPattern.Any())
                CLI.StarPattern = new string[] { "*.star" };

            if (string.IsNullOrEmpty(Path.GetFileName(CLI.StarOut)))
                throw new Exception("Please specify a file name for --output");
            CLI.StarOut = Helper.PathCombine(Environment.CurrentDirectory, CLI.StarOut);

            if (CLI.AngpixCoords <= 0)
                throw new Exception("--angpix_coords must be positive");

            if (CLI.AngpixOut != null && (CLI.AngpixOut <= 0 || CLI.AngpixOut < (double)Options.Import.BinnedPixelSize))
                throw new Exception("--angpix_out must be positive and not smaller than the binned import pixel size specified in .settings");

            if (CLI.BoxSize <= 2 || CLI.BoxSize % 2 != 0)
                throw new Exception("--box must be an even number greater than 2");

            if (CLI.Diameter <= 0)
                throw new Exception("--diameter must be positive");

            { 
                var Exports = new bool[] { CLI.ExportAverages, CLI.ExportAverageHalves, CLI.ExportStar };

                if (!Exports.Any())
                    throw new Exception("No output types requested (--averages, --halves, --only_star)");

                if (Exports.Count(e => e) > 1)
                    throw new Exception("Only one output type can be requested at a time (--averages, --halves, --only_star)");

                //if (CLI.FromAverages && !CLI.ExportAverages)
                //    throw new Exception("--from_averages requires --averages as export type");
            }

            if (CLI.ExportAverages || CLI.ExportAverageHalves)
            {
                if (CLI.SkipFirst < 0)
                    throw new Exception("--skip_first_frames cannot be negative");

                if (CLI.SkipLast < 0)
                    throw new Exception("--skip_last_frames cannot be negative");
            }

            #endregion

            #region Set options

            Options.Export.SkipFirstN = CLI.SkipFirst;
            Options.Export.SkipLastN = CLI.SkipLast;

            Options.Tasks.Export2DBoxSize = CLI.BoxSize;
            Options.Tasks.Export2DParticleDiameter = CLI.Diameter;

            Options.Tasks.Export2DPreflip = CLI.PreflipPhases;
            Options.Tasks.Export2DPixel = CLI.AngpixOut != null ? (decimal)CLI.AngpixOut.Value : Options.Import.BinnedPixelSize;

            ProcessingOptionsParticleExport ExportOptions = Options.GetProcessingParticleExportTask();
            ExportOptions.DoAverage = CLI.ExportAverages;
            ExportOptions.DoDenoisingPairs = CLI.ExportAverageHalves;
            ExportOptions.Invert = !CLI.DontInvert;
            ExportOptions.Normalize = !CLI.DontNormalize;
            ExportOptions.PreflipPhases = CLI.PreflipPhases;
            ExportOptions.BoxSize = CLI.BoxSize;
            ExportOptions.Diameter = CLI.Diameter;
            ExportOptions.Suffix = CLI.SuffixOut;
            
            {
                MapHeader Header = MapHeader.ReadFromFile(CLI.InputSeries.First().DataPath);
                ExportOptions.Dimensions = Header.Dimensions.MultXY((float)Options.Import.PixelSize);
            }

            #endregion

            #region Figure out and load STAR files

            List<Star> AllTablesIn = new List<Star>();
            List<Star> AllOpticsIn = new List<Star>();

            bool Relion3 = false;
            bool PerItemStar = false;

            if (File.Exists(CLI.StarFolder))
            {
                if (CLI.StarPattern != null && CLI.StarPattern.Any())
                    throw new Exception("Can't specify --files when --star already points to a single file");

                CLI.StarPattern = new string[] { Path.GetFileName(CLI.StarFolder) };
                CLI.StarFolder = Path.GetDirectoryName(CLI.StarFolder);
            }

            List<string> AllStarPaths = new List<string>();
            foreach (var pattern in CLI.StarPattern)
                if (File.Exists(Path.Combine(CLI.StarFolder, pattern)))
                {
                    AllStarPaths.Add(Path.Combine(CLI.StarFolder, pattern));
                }
                else
                {
                    string[] StarPaths = Directory.GetFiles(CLI.StarFolder, pattern);
                    AllStarPaths.AddRange(StarPaths);
                }

            if (AllStarPaths.Count == 0)
                throw new Exception("No STAR files found");

            Console.WriteLine($"Found {AllStarPaths.Count} STAR files");

            // Figure out if we're dealing with a RELION 3-style STAR file, and if files are per-item
            {
                Relion3 = Star.ContainsTable(AllStarPaths.First(), "particles");

                Star FirstTable = Relion3 ? new Star(AllStarPaths.First(), "particles") : new Star(AllStarPaths.First());

                if (!FirstTable.HasColumn("rlnMicrographName"))
                {
                    Console.Write("No rlnMicrographName column found, trying to match file names to movies... ");

                    if (CLI.InputSeries.Any(s => Path.GetFileNameWithoutExtension(AllStarPaths.First()).Contains(s.RootName)))
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

                    AllStarPaths = AllStarPaths.Where(p => CLI.InputSeries.Any(s => Path.GetFileNameWithoutExtension(p).Contains(s.RootName))).ToList();
                    Console.WriteLine($"Found {AllStarPaths.Count} matching STAR files");
                }

                if (!FirstTable.HasColumn("rlnCoordinateX") || !FirstTable.HasColumn("rlnCoordinateY"))
                    throw new Exception("Please make sure the STAR files contain columns named rlnCoordinateX and rlnCoordinateY");
            }

            Console.Write($"Reading STAR files... 0/{AllStarPaths.Count}");
            Helper.ForCPU(0, AllStarPaths.Count, 8, null, (istar, threadID) =>
            {
                Star ThisTable = Relion3 ? new Star(AllStarPaths[istar], "particles") : new Star(AllStarPaths[istar]);
                Star ThisOptics = Relion3 ? new Star(AllStarPaths[istar], "optics") : null;

                if (PerItemStar)
                {
                    string StarName = Path.GetFileNameWithoutExtension(AllStarPaths[istar]);
                    string MicrographName = Path.GetFileName(CLI.InputSeries.First(s => StarName.Contains(s.RootName)).DataPath);

                    ThisTable.AddColumn("rlnMicrographName", MicrographName);
                }

                lock (AllTablesIn)
                {
                    AllTablesIn.Add(ThisTable);
                    if (ThisOptics != null)
                        AllOpticsIn.Add(ThisOptics);

                    VirtualConsole.ClearLastLine();
                    Console.Write($"Reading STAR files... {AllTablesIn.Count}/{AllStarPaths.Count}");
                }
            }, null);

            Star TableIn = new Star(AllTablesIn.ToArray());
            Star OpticsIn = AllOpticsIn.Count > 0 ? new Star(AllOpticsIn.ToArray()) : new Star();

            if (!TableIn.HasColumn("rlnImageName"))
                TableIn.AddColumn("rlnImageName");

            #region Optics table

            if (!CLI.KeepCTF && CLI.InputSeries.First().CTF != null)
            {
                if (!TableIn.HasColumn("rlnDefocusU"))
                    TableIn.AddColumn("rlnDefocusU", "0.0");
                if (!TableIn.HasColumn("rlnDefocusV"))
                    TableIn.AddColumn("rlnDefocusV", "0.0");
                if (!TableIn.HasColumn("rlnDefocusAngle"))
                    TableIn.AddColumn("rlnDefocusAngle", "0.0");
                if (!TableIn.HasColumn("rlnPhaseShift"))
                    TableIn.AddColumn("rlnPhaseShift", "0.0");
                if (!TableIn.HasColumn("rlnCtfMaxResolution"))
                    TableIn.AddColumn("rlnCtfMaxResolution", "999");

                if (!TableIn.HasColumn("rlnOpticsGroup"))
                    TableIn.AddColumn("rlnOpticsGroup", "1");
                else
                    TableIn.SetColumn("rlnOpticsGroup", Helper.ArrayOfConstant("1", TableIn.RowCount));

                OpticsIn = new Star(new string[] 
                { 
                    "rlnVoltage", 
                    "rlnImagePixelSize", 
                    "rlnSphericalAberration", 
                    "rlnAmplitudeContrast",
                    "rlnOpticsGroup",
                    "rlnImageSize",
                    "rlnImageDimensionality",
                    "rlnOpticsGroupName"
                });
                OpticsIn.AddRow(new string[]
                {
                    CLI.InputSeries.First().CTF.Voltage.ToString("F6", CultureInfo.InvariantCulture),
                    Options.Tasks.Export2DPixel.ToString("F6", CultureInfo.InvariantCulture),
                    CLI.InputSeries.First().CTF.Cs.ToString("F6", CultureInfo.InvariantCulture),
                    CLI.InputSeries.First().CTF.Amplitude.ToString("F6", CultureInfo.InvariantCulture),
                    "1",
                    Options.Tasks.Export2DBoxSize.ToString(),
                    "2",
                    "opticsGroup1"
                });
            }

            #endregion

            #endregion

            #region Group rows per item, and find unused rows to remove once finished

            List<string> MovieNames = CLI.InputSeries.Select(s => s.RootName).ToList();

            Dictionary<string, List<int>> RowGroups = new Dictionary<string, List<int>>();
            {
                string[] ColumnMicNames = TableIn.GetColumn("rlnMicrographName").Select(s => Helper.PathToName(s)).ToArray();
                for (int r = 0; r < ColumnMicNames.Length; r++)
                {
                    if (!RowGroups.ContainsKey(ColumnMicNames[r]))
                        RowGroups.Add(ColumnMicNames[r], new List<int>());
                    RowGroups[ColumnMicNames[r]].Add(r);
                }

                RowGroups = RowGroups.Where(group => MovieNames.Contains(group.Key)).ToDictionary(group => group.Key, group => group.Value);
            }

            if (RowGroups.Count == 0)
                throw new Exception("No rows found matching any of the input movies' names in the rlnMicrographName column");

            List<int> RowsNotIncluded = new List<int>();
            {
                bool[] RowsIncluded = new bool[TableIn.RowCount];
                foreach (var group in RowGroups)
                    foreach (var r in group.Value)
                        RowsIncluded[r] = true;
                for (int r = 0; r < RowsIncluded.Length; r++)
                    if (!RowsIncluded[r])
                        RowsNotIncluded.Add(r);
            }

            #endregion

            #region Get coordinates and shifts

            float[] PosX = TableIn.GetColumn("rlnCoordinateX").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * (float)CLI.AngpixCoords).ToArray();
            float[] PosY = TableIn.GetColumn("rlnCoordinateY").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * (float)CLI.AngpixCoords).ToArray();

            float[] ShiftX = null;
            float[] ShiftY = null;
            bool HasShifts = TableIn.HasColumn("rlnOriginX") || TableIn.HasColumn("rlnOriginXAngst");

            if (HasShifts)
            {
                bool ShiftsInAngstrom = TableIn.HasColumn("rlnOriginXAngst") && TableIn.HasColumn("rlnOriginYAngst");

                if (!CLI.DontCenter && !ShiftsInAngstrom && CLI.AngpixShifts == null)
                    throw new Exception("Please specify --angpix_shifts if the input STAR files specify refined shifts in pixels (i.e. rlnOriginX instead of rlnOriginXAngst)");

                string LabelShiftX = ShiftsInAngstrom ? "rlnOriginXAngst" : "rlnOriginX";
                string LabelShiftY = ShiftsInAngstrom ? "rlnOriginYAngst" : "rlnOriginY";
                if (ShiftsInAngstrom)
                    CLI.AngpixShifts = 1;

                ShiftX = TableIn.HasColumn(LabelShiftX) ? 
                         TableIn.GetColumn(LabelShiftX).Select(v => float.Parse(v, CultureInfo.InvariantCulture) * (float)CLI.AngpixShifts).ToArray() : 
                         new float[TableIn.RowCount];
                ShiftY = TableIn.HasColumn(LabelShiftY) ? 
                         TableIn.GetColumn(LabelShiftY).Select(v => float.Parse(v, CultureInfo.InvariantCulture) * (float)CLI.AngpixShifts).ToArray() : 
                         new float[TableIn.RowCount];

                if (!CLI.DontCenter)
                {
                    for (int r = 0; r < TableIn.RowCount; r++)
                    {
                        PosX[r] -= ShiftX[r];
                        PosY[r] -= ShiftY[r];
                    }

                    if (TableIn.HasColumn(LabelShiftX))
                        TableIn.RemoveColumn(LabelShiftX);
                    if (TableIn.HasColumn(LabelShiftY))
                        TableIn.RemoveColumn(LabelShiftY);
                }
            }

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            IterateOverItems<Movie>(Workers, CLI, (worker, m) =>
            {
                if (!RowGroups.ContainsKey(m.RootName))
                    return;

                #region Stack and micrograph paths

                string PathStack = Path.Combine(m.ParticlesDir, m.RootName + ExportOptions.Suffix + ".mrcs");
                string PathMicrograph = m.Path;
                if (CLI.OutputPathsRelativeToStarFile)
                {
                    PathMicrograph = Helper.MakePathRelativeTo(PathMicrograph, CLI.StarOut);
                    PathStack = Helper.MakePathRelativeTo(PathStack, CLI.StarOut);
                }
                else
                {
                    PathMicrograph = Helper.MakePathRelativeTo(PathMicrograph, Path.Combine(Environment.CurrentDirectory, "my.star"));
                    PathStack = Helper.MakePathRelativeTo(PathStack, Path.Combine(Environment.CurrentDirectory, "my.star"));
                }

                #endregion

                #region Update row values

                List<float2> Positions = new List<float2>();
                float Astigmatism = (float)m.CTF.DefocusDelta / 2;
                float PhaseShift = (m.OptionsCTF.DoPhase && m.GridCTFPhase != null) ? 
                                   m.GridCTFPhase.GetInterpolated(new float3(0.5f)) * 180 : 
                                   0;

                int iparticle = 0;
                foreach (var r in RowGroups[m.RootName])
                {
                    float3 Position = new float3(PosX[r] / ExportOptions.Dimensions.X,
                                                 PosY[r] / ExportOptions.Dimensions.Y,
                                                 0.5f);
                    float LocalDefocus = m.GridCTFDefocus.GetInterpolated(Position);

                    if (!CLI.KeepCTF)
                    {
                        TableIn.SetRowValue(r, "rlnDefocusU", ((LocalDefocus + Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture));
                        TableIn.SetRowValue(r, "rlnDefocusV", ((LocalDefocus - Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture));
                        TableIn.SetRowValue(r, "rlnDefocusAngle", m.CTF.DefocusAngle.ToString("F1", CultureInfo.InvariantCulture));

                        TableIn.SetRowValue(r, "rlnPhaseShift", PhaseShift.ToString("F1", CultureInfo.InvariantCulture));
                        TableIn.SetRowValue(r, "rlnCtfMaxResolution", m.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture));
                    }

                    TableIn.SetRowValue(r, "rlnCoordinateX", (PosX[r] / (float)CLI.AngpixCoords).ToString("F3", CultureInfo.InvariantCulture));
                    TableIn.SetRowValue(r, "rlnCoordinateY", (PosY[r] / (float)CLI.AngpixCoords).ToString("F3", CultureInfo.InvariantCulture));

                    if (CLI.DontCenter && HasShifts)
                    {
                        if (TableIn.HasColumn("rlnOriginX"))
                        {
                            TableIn.SetRowValue(r, "rlnOriginX", (ShiftX[r] / (float)ExportOptions.BinnedPixelSizeMean).ToString("F3", CultureInfo.InvariantCulture));
                            TableIn.SetRowValue(r, "rlnOriginY", (ShiftY[r] / (float)ExportOptions.BinnedPixelSizeMean).ToString("F3", CultureInfo.InvariantCulture));
                        }
                    }

                    TableIn.SetRowValue(r, "rlnImageName", $"{++iparticle:D7}@{PathStack}");

                    TableIn.SetRowValue(r, "rlnMicrographName", PathMicrograph);

                    Positions.Add(new float2(PosX[r], PosY[r]));
                }

                #endregion

                #region Export actual particle (half-)averages if needed

                if (CLI.ExportAverages || CLI.ExportAverageHalves)
                {
                    decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)ExportOptions.BinTimes);
                    worker.LoadStack(m.DataPath, ScaleFactor, Options.Import.EERGroupFrames);

                    worker.MovieExportParticles(m.Path, ExportOptions, Positions.ToArray());
                }

                #endregion
            });

            Console.Write("Saving STAR file...");      
            
            TableIn.RemoveRows(RowsNotIncluded.ToArray());
            Star.SaveMultitable(CLI.StarOut, new()
            {
                { "optics", OpticsIn },
                { "particles", TableIn }
            });

            Console.WriteLine(" Done");

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");
        }
    }
}
