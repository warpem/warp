using CommandLine;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_import", HelpText = "Create .tomostar files based on a combination of MDOC files, aligned frame series, and optional tilt series alignments from IMOD or AreTomo")]
    [CommandRunner(typeof(ImportTiltseries))]
    class ImportTiltseriesOptions
    {
        [Option("mdocs", Required = true, HelpText = "Path to the folder containing MDOC files")]
        public string MdocFolder { get; set; }

        [Option("pattern", Default = "*.mdoc", HelpText = "File name pattern to search for in the MDOC folder")]
        public string MdocPattern { get; set; }

        [Option("frameseries", Required = true, HelpText = "Path to a folder containing frame series processing results and their aligned averages")]
        public string FrameseriesPath { get; set; }

        [Option("tilt_exposure", Required = true, HelpText = "Per-tilt exposure in e-/A^2")]
        public double TiltExposure { get; set; }

        [Option("dont_invert", HelpText = "Don't invert tilt angles compared to IMOD's convention (inversion is usually needed to match IMOD's geometric handedness). This will flip the geometric handedness")]
        public bool DontInvert { get; set; }

        [Option("override_axis", HelpText = "Override the tilt axis angle with this value")]
        public double? OverrideAxis { get; set; }

        [Option("max_tilt", Default = 90, HelpText = "Exclude all tilts above this (absolute) tilt angle")]
        public int MaxTilt { get; set; }

        [Option("min_intensity", Default = 0.0, HelpText = "Exclude tilts if their average intensity is below MinIntensity * cos(angle) * 0-tilt intensity; set to 0 to not exclude anything")]
        public double MinIntensity { get; set; }

        [Option("max_mask", Default = 1.0, HelpText = "Exclude tilts if more than this fraction of their pixels is masked; needs frame series with BoxNet masking results")]
        public double MaxMask { get; set; }

        [Option('o', "output", Required = true, HelpText = "Path to a folder where the created .tomostar files will be saved")]
        public string OutputPath { get; set; }
    }

    class ImportTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ImportTiltseriesOptions CLI = options as ImportTiltseriesOptions;

            #region Validate options

            CLI.MdocFolder = Helper.PathCombine(Environment.CurrentDirectory, CLI.MdocFolder);
            CLI.FrameseriesPath = Helper.PathCombine(Environment.CurrentDirectory, CLI.FrameseriesPath);
            CLI.OutputPath = Helper.PathCombine(Environment.CurrentDirectory, CLI.OutputPath);

            if (!Directory.Exists(CLI.MdocFolder))
                throw new Exception($"MDOC folder does not exist: {CLI.MdocFolder}");
            if (!Directory.Exists(CLI.FrameseriesPath))
                throw new Exception($"Frame series folder does not exist: {CLI.FrameseriesPath}");

            if (CLI.TiltExposure <= 0)
                throw new Exception($"--tilt_exposure must be higher than 0");
            if (CLI.MaxTilt <= 0)
                throw new Exception($"--max_tilt must be higher than 0");
            if (CLI.MinIntensity < 0)
                throw new Exception($"--min_intensity must be positive or 0");
            if (CLI.MaxMask < 0 || CLI.MaxMask > 1)
                throw new Exception($"--max_mask must be between 0 and 1");

            #endregion

            #region Find MDOCs

            List<string> MdocPaths;
            {
                string PatternDir = Path.GetFullPath(CLI.MdocFolder);
                string PatternFile = CLI.MdocPattern;
                MdocPaths = Directory.EnumerateFiles(PatternDir, PatternFile).ToList();

                Console.WriteLine($"Found {MdocPaths.Count} MDOC files, searching for {PatternFile} in {PatternDir}");
            }

            #endregion

            #region Find frame series

            Dictionary<string, Movie> Movies;
            {
                Console.WriteLine("Looking for frame series...");

                string[] MoviePaths = Directory.EnumerateFiles(CLI.FrameseriesPath, "*.xml").ToArray();

                if (!MoviePaths.Any())
                    throw new Exception($"No frame series metadata found at {CLI.FrameseriesPath}");

                Movie[] ParsedMovies = new Movie[MoviePaths.Length];

                Console.Write($"0/{ParsedMovies.Length}");
                int NDone = 0;
                Helper.ForCPU(0, ParsedMovies.Length, 8, null, (i, threadID) =>
                {
                    ParsedMovies[i] = new Movie(Path.Combine(Path.GetDirectoryName(MoviePaths[i]), Path.GetFileNameWithoutExtension(MoviePaths[i]) + ".mrc"));

                    lock (ParsedMovies)
                    {
                        NDone++;
                        VirtualConsole.ClearLastLine();
                        Console.Write($"{NDone}/{ParsedMovies.Length}");
                    }
                }, null);
                Console.WriteLine("");

                Movies = ParsedMovies.ToDictionary(m => m.RootName);
            }

            #endregion

            #region Parse MDOCs

            Directory.CreateDirectory(CLI.OutputPath);

            Console.WriteLine("Parsing MDOCs and creating .tomostar files...");

            {
                int NDone = 0;
                int NFailed = 0;
                Parallel.ForEach(MdocPaths, mdocPath =>
                {
                    try
                    {
                        float AxisAngle = 0;
                        List<MdocEntry> Entries = new List<MdocEntry>();
                        bool FoundTime = false;

                        using (TextReader Reader = File.OpenText(mdocPath))
                        {
                            string Line;
                            while ((Line = Reader.ReadLine()) != null)
                            {
                                if (Line.Contains("Tilt axis angle = "))
                                {
                                    string Suffix = Line.Substring(Line.IndexOf("Tilt axis angle = ") + "Tilt axis angle = ".Length);
                                    Suffix = Suffix.Substring(0, Suffix.IndexOf(","));

                                    AxisAngle = float.Parse(Suffix, CultureInfo.InvariantCulture);
                                    continue;
                                }
                                else if (Line.Contains("TiltAxisAngle = "))
                                {
                                    string Suffix = Line.Substring(Line.IndexOf("TiltAxisAngle = ") + "TiltAxisAngle = ".Length);
                                    Suffix = Suffix.Substring(0, Suffix.IndexOf(" "));

                                    AxisAngle = float.Parse(Suffix, CultureInfo.InvariantCulture);
                                    continue;
                                }

                                if (Line.Length < 7 || Line.Substring(1, 6) != "ZValue")
                                    continue;

                                MdocEntry NewEntry = new MdocEntry();

                                {
                                    string[] Parts = Line.Split(new[] { " = " }, StringSplitOptions.RemoveEmptyEntries);
                                    if (Parts[0] == "[ZValue")
                                        NewEntry.ZValue = int.Parse(Parts[1].Replace("]", ""));
                                }

                                while ((Line = Reader.ReadLine()) != null)
                                {
                                    string[] Parts = Line.Split(new[] { " = " }, StringSplitOptions.RemoveEmptyEntries);
                                    if (Parts.Length < 2)
                                        break;

                                    if (Parts[0] == "TiltAngle")
                                        NewEntry.TiltAngle = (float)Math.Round(float.Parse(Parts[1], CultureInfo.InvariantCulture), 2);
                                    else if (Parts[0] == "SubFramePath")
                                        // Can't use built-in Path.GetFileName because it won't expect backward slashes when running on Unix
                                        // but file path most likely comes from a Windows system
                                        NewEntry.Name = Parts[1].Substring(Math.Max(Parts[1].LastIndexOf('/'), Parts[1].LastIndexOf('\\')) + 1);
                                    else if (Parts[0] == "DateTime")
                                    {
                                        try
                                        {
                                            try
                                            {
                                                NewEntry.Time = DateTime.ParseExact(Parts[1], "dd-MMM-yy  HH:mm:ss", CultureInfo.InvariantCulture);
                                            }
                                            catch
                                            {
                                                NewEntry.Time = DateTime.ParseExact(Parts[1], "dd-MMM-yyyy  HH:mm:ss", CultureInfo.InvariantCulture);
                                            }

                                            FoundTime = true;
                                        }
                                        catch
                                        {
                                            FoundTime = false;
                                        }
                                    }
                                }

                                if (!FoundTime)
                                    throw new Exception($"No time stamp found: {NewEntry.Name}");

                                if (Entries.Any(v => v.ZValue == NewEntry.ZValue))
                                    throw new Exception($"Duplicate ZValue in MDOC file: {NewEntry.ZValue}");

                                if (!Movies.ContainsKey(Path.GetFileNameWithoutExtension(NewEntry.Name)))
                                    throw new Exception($"At least one of the referenced frame series could not be found: {NewEntry.Name}");

                                if (!File.Exists(Movies[Path.GetFileNameWithoutExtension(NewEntry.Name)].AveragePath))
                                    throw new Exception($"At least one of the referenced frame series does not have an aligned average result: {NewEntry.Name}");

                                if (CLI.OverrideAxis != null)
                                    NewEntry.AxisAngle = (float)CLI.OverrideAxis;
                                else
                                    NewEntry.AxisAngle = AxisAngle;

                                decimal MaskedPercentage = Movies[Path.GetFileNameWithoutExtension(NewEntry.Name)].MaskPercentage;
                                if (MaskedPercentage > 0)
                                    NewEntry.MaskedFraction = (float)MaskedPercentage / 100f;

                                Entries.Add(NewEntry);
                            }
                        }

                        List<MdocEntry> SortedTime = new List<MdocEntry>(Entries);
                        SortedTime.Sort((a, b) => a.Time.CompareTo(b.Time));


                        // Replace exposure value with accumulated per-tilt exposure based on --tilt_exposure
                        for (int i = 0; i < SortedTime.Count; i++)
                            SortedTime[i].Dose = i * (float)CLI.TiltExposure;


                        // Sort entires by angle and time (accumulated dose)
                        List<MdocEntry> SortedAngle = new List<MdocEntry>(Entries);
                        SortedAngle.Sort((a, b) => a.TiltAngle.CompareTo(b.TiltAngle));

                        SortedAngle.RemoveAll(v =>
                        {
                            Movie M = Movies[Path.GetFileNameWithoutExtension(v.Name)];
                            if (M.UnselectManual != null && (bool)M.UnselectManual)
                                return true;
                            return false;
                        });

                        SortedAngle.RemoveAll(v => Math.Abs(v.TiltAngle) > CLI.MaxTilt + 0.05);

                        SortedAngle.RemoveAll(v => v.MaskedFraction > CLI.MaxMask);

                        if (SortedAngle.Count == 0)
                        {
                            Console.WriteLine($"No images left for {Path.GetFileName(mdocPath)} after removing tilts based on selection status, max tilt angle and masked fraction criteria");
                            return;
                        }

                        #region Determine average intensity in each tilt
                        {
                            var SortedAbsoluteAngle = new List<MdocEntry>(SortedAngle);
                            SortedAbsoluteAngle.Sort((a, b) => Math.Abs(a.TiltAngle).CompareTo(Math.Abs(b.TiltAngle)));

                            MapHeader Header = MapHeader.ReadFromFile(Movies[Path.GetFileNameWithoutExtension(SortedAbsoluteAngle[0].Name)].AveragePath);
                            var AverageReadBuffer = new float[Header.Dimensions.ElementsSlice()];
                            var AverageSparse = new float[AverageReadBuffer.Length / 10];

                            foreach (var entry in SortedAngle)
                            {
                                IOHelper.ReadMapFloat(Movies[Path.GetFileNameWithoutExtension(entry.Name)].AveragePath, new[] { 0 }, null, new float[][] { AverageReadBuffer });
                                                                
                                for (int i = 0; i < AverageReadBuffer.Length / 10; i++)
                                    AverageSparse[i] = AverageReadBuffer[i * 10];

                                entry.AverageIntensity = MathHelper.Median(AverageSparse);
                            }

                            float MaxAverage = Helper.ArrayOfFunction(i => SortedAbsoluteAngle[i].AverageIntensity, Math.Min(10, SortedAbsoluteAngle.Count)).Max();
                            MdocEntry ZeroAngleEntry = SortedAbsoluteAngle.Where(e => e.AverageIntensity == MaxAverage).First();
                            int ZeroAngleId = SortedAngle.IndexOf(ZeroAngleEntry);
                            float ActualZeroAngle = ZeroAngleEntry.TiltAngle;

                            foreach (var entry in SortedAngle)
                                entry.TiltAngle -= ActualZeroAngle;

                            int LowestAngleId = ZeroAngleId;
                            int HighestAngleId = ZeroAngleId;

                            bool[] Passed = SortedAngle.Select(e => e.AverageIntensity >= CLI.MinIntensity * MathF.Cos(e.TiltAngle * Helper.ToRad) * MaxAverage * 0.999f).ToArray();
                            for (int i = ZeroAngleId - 1; i >= 0; i--)
                            {
                                if (!Passed[i])
                                    break;
                                LowestAngleId = i;
                            }
                            for (int i = ZeroAngleId + 1; i < SortedAngle.Count; i++)
                            {
                                if (!Passed[i])
                                    break;
                                HighestAngleId = i;
                            }

                            SortedAngle = SortedAngle.GetRange(LowestAngleId, HighestAngleId - LowestAngleId + 1);
                        }
                        #endregion

                        if (SortedAngle.Count == 0)
                            throw new Exception("0 tilts remain after parsing and culling");

                        Star Table = new Star(new[]
                        {
                            "wrpMovieName",
                            "wrpAngleTilt",
                            "wrpAxisAngle",
                            "wrpDose",
                            "wrpAverageIntensity",
                            "wrpMaskedFraction"
                        });

                        string OutputPath = Path.Combine(CLI.OutputPath, Path.GetFileNameWithoutExtension(mdocPath.Replace(".mrc.mdoc", ".mdoc")) + ".tomostar");

                        for (int i = 0; i < SortedAngle.Count; i++)
                        {
                            //if (CreateStacks)
                            //    StackData[i] = SortedAngle[i].Micrograph.GetHost(Intent.Read)[0];

                            Movie M = Movies[Path.GetFileNameWithoutExtension(SortedAngle[i].Name)];
                            string PathToMovie = Path.Combine(Path.GetDirectoryName(M.Path), SortedAngle[i].Name);

                            string MovieRelativePath = Helper.MakePathRelativeTo(PathToMovie, OutputPath);

                            Table.AddRow(new string[]
                            {
                                MovieRelativePath,
                                (SortedAngle[i].TiltAngle * (CLI.DontInvert ? 1 : -1)).ToString("F2", CultureInfo.InvariantCulture),
                                SortedAngle[i].AxisAngle.ToString("F3", CultureInfo.InvariantCulture),
                                SortedAngle[i].Dose.ToString(CultureInfo.InvariantCulture),
                                SortedAngle[i].AverageIntensity.ToString("F3", CultureInfo.InvariantCulture),
                                SortedAngle[i].MaskedFraction.ToString("F3", CultureInfo.InvariantCulture)
                            });
                        }

                        Table.Save(OutputPath);

                        lock (MdocPaths)
                        {
                            NDone++;
                            Console.WriteLine($"Successfully parsed {Path.GetFileName(mdocPath)}, {SortedAngle.Count} tilts");
                        }
                    }
                    catch (Exception exc)
                    {
                        lock (MdocPaths)
                        {
                            NFailed++;
                            Console.WriteLine($"Failed to parse {Path.GetFileName(mdocPath)}: {exc.Message}");
                        }
                    }
                });

                Console.WriteLine($"Successfully parsed {NDone} MDOC files, {NFailed} failed");
            }

            #endregion
        }
    }
}
