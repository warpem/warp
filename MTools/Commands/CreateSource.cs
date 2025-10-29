using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Warp;
using Warp.Headers;

namespace MTools.Commands
{
    [Verb("create_source", HelpText = "Create a new data source")]
    [CommandRunner(typeof(CreateSource))]
    class CreateSourceOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file to which to add the new data source.")]
        public string Population { get; set; }

        [Option('s', "processing_settings", Required = true, HelpText = "Path to a .settings file used to pre-process the frame or tilt series this source should include; desktop Warp will usually generate a previous.settings file")]
        public string ProcessingSettings { get; set; }

        [Option('n', "name", Required = true, HelpText = "Name of the new data source.")]
        public string Name { get; set; }

        [Option("nframes", HelpText = "Maximum number of tilts or frames to use in refinements. Leave empty or set to 0 to use the maximum number available.")]
        public int? NFrames { get; set; }

        [Option("files", HelpText = "Optional STAR file with a list of files to intersect with the full list of frame or tilt series referenced by the settings.")]
        public string Files { get; set; }

        [Option('o', "output", HelpText = "Optionally, override the default path where the .source file will be saved.")]
        public string OutputPath { get; set; }
        
        [Option("dont_version", HelpText = "If set, the source will not be versioned.")]
        public bool DontVersion { get; set; } = false;
    }

    class CreateSource : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            CreateSourceOptions OptionsCLI = options as CreateSourceOptions;

            Population Population = new Population(OptionsCLI.Population);


            #region Load preprocessing options

            OptionsWarp OptionsWarp = new OptionsWarp();
            try
            {
                OptionsWarp.Load(OptionsCLI.ProcessingSettings);
            }
            catch (Exception exc)
            {
                Console.Error.WriteLine($"An error was encountered when reading processing settings:\n{exc.Message}");
                return;
            }

            #endregion

            #region Create source metadata and check if one with the same path already exists

            DataSource NewSource = new DataSource
            {
                PixelSize = OptionsWarp.Import.PixelSize,

                DimensionsX = OptionsWarp.Tomo.DimensionsX,
                DimensionsY = OptionsWarp.Tomo.DimensionsY,
                DimensionsZ = OptionsWarp.Tomo.DimensionsZ,
                FrameLimit = 0,

                GainPath = !string.IsNullOrEmpty(OptionsWarp.Import.GainPath) && OptionsWarp.Import.CorrectGain ?
                               Helper.PathCombine(Environment.CurrentDirectory,
                                                  Path.GetDirectoryName(OptionsCLI.ProcessingSettings),
                                                  OptionsWarp.Import.GainPath) :
                               "",
                GainFlipX = OptionsWarp.Import.GainFlipX,
                GainFlipY = OptionsWarp.Import.GainFlipY,
                GainTranspose = OptionsWarp.Import.GainTranspose,

                DefectsPath = !string.IsNullOrEmpty(OptionsWarp.Import.DefectsPath) && OptionsWarp.Import.CorrectDefects ?
                                  Helper.PathCombine(Environment.CurrentDirectory,
                                                     Path.GetDirectoryName(OptionsCLI.ProcessingSettings),
                                                     OptionsWarp.Import.DefectsPath) :
                                  "",

                DosePerAngstromFrame = OptionsWarp.Import.DosePerAngstromFrame,
                EERGroupFrames = OptionsWarp.Import.EERGroupFrames,

                Name = OptionsCLI.Name,
                Path = string.IsNullOrWhiteSpace(OptionsCLI.OutputPath) ?
                           Helper.PathCombine(Environment.CurrentDirectory,
                                              Path.GetDirectoryName(OptionsCLI.ProcessingSettings),
                                              OptionsWarp.Import.ProcessingOrDataFolder,
                                              Helper.RemoveInvalidChars(OptionsCLI.Name) + ".source") :
                           OptionsCLI.OutputPath,
                
                DontVersion = OptionsCLI.DontVersion
            };

            if (Population.Sources.Any(s => Path.GetFullPath(s.Path) == Path.GetFullPath(NewSource.Path)))
            {
                Console.Error.WriteLine($"{Helper.PathToNameWithExtension(NewSource.Path)} already exists in this population.");
                return;
            }

            #endregion

            #region Load items with metadata

            Console.Write("Looking for data... ");

            string FileExtension = OptionsWarp.Import.Extension;
            var AvailableFiles = Directory.EnumerateFiles(string.IsNullOrEmpty(OptionsWarp.Import.DataFolder) ? OptionsWarp.Import.ProcessingFolder : OptionsWarp.Import.DataFolder, 
                                                          OptionsWarp.Import.Extension, 
                                                          OptionsWarp.Import.DoRecursiveSearch ? SearchOption.AllDirectories : 
                                                                                                 SearchOption.TopDirectoryOnly).ToArray();

            Movie[] Items = new Movie[AvailableFiles.Length];

            Console.WriteLine($"found {AvailableFiles.Length} files");


            {
                Console.Write("Loading metadata... ");

                int Done = 0;
                Parallel.For(0, AvailableFiles.Length, new ParallelOptions() { MaxDegreeOfParallelism = 8}, i =>
                {
                    string file = Path.Combine(OptionsWarp.Import.ProcessingOrDataFolder, Helper.PathToNameWithExtension(AvailableFiles[i]));
                    string XmlPath = file.Substring(0, file.LastIndexOf(".")) + ".xml";
                    if (File.Exists(XmlPath))
                        Items[i] = FileExtension == "*.tomostar" ? new TiltSeries(file) : new Movie(file);

                    lock (Items)
                    {
                        Done++;
                        if (Done % 10 == 0)
                        {
                            VirtualConsole.ClearLastLine();
                            Console.Write($"Loading metadata... {Done}/{AvailableFiles.Length}");
                        }
                    }
                });

                VirtualConsole.ClearLastLine();
                Console.WriteLine($"Loading metadata... {Done}/{AvailableFiles.Length}");
            }
            Console.WriteLine();

            if (Items.Length == 0)
            {
                Console.Error.WriteLine($"No movies or tilt series found to match these settings.");
                return;
            }

            #endregion

            List<Movie> AllItems = new List<Movie>();

            #region Figure out filtering status

            if (!string.IsNullOrEmpty(OptionsCLI.Files))
            {
                Console.WriteLine("Using file list to limit items");

                HashSet<string> TakeFiles = null;

                /*if (Path.GetExtension(OptionsCLI.Files).ToLower() == ".star")
                {
                    try
                    {
                        Console.Write("Interpreting as tilt series particle list... ");

                        Star TableIn = new Star(OptionsCLI.Files, "particles", new[] { "rlnTomoName" });
                    }
                    catch
                    {

                    }
                }
                else*/ if (Path.GetExtension(OptionsCLI.Files).ToLower() == ".txt")
                {
                    string[] Lines = File.ReadAllLines(OptionsCLI.Files).Select(p => Helper.PathToNameWithExtension(p)).ToArray();
                    TakeFiles = Helper.GetUniqueElements(Lines);
                }
                else
                {
                    Console.Error.WriteLine("Only plain text files with one file name per line are supported.");
                    return;
                }

                if (TakeFiles == null || TakeFiles.Count == 0)
                {
                    Console.Error.WriteLine("List contains 0 files.");
                    return;
                }

                Console.WriteLine($"File list contains {TakeFiles.Count} items");

                AllItems = Items.Where(m => TakeFiles.Contains(Helper.PathToNameWithExtension(m.Path))).ToList();

                Console.WriteLine($"{AllItems.Count} files match the list");
            }
            else
            {
                AllItems = Items.ToList();
            }

            #endregion

            #region Figure out how many frames/tilts there are to use

            int UsableFrames = 1;
            // With movies, checking one is enough since they all have identical dimensions
            if (AllItems[0].GetType() == typeof(Movie))
                UsableFrames = MapHeader.ReadFromFile(AllItems[0].DataPath).Dimensions.Z;
            else
            // With tilt series, the number of tilts is variable, so need to check all
                foreach (var item in AllItems)
                    UsableFrames = Math.Max(UsableFrames, ((TiltSeries)item).NTilts);

            if (OptionsCLI.NFrames != null && (int)OptionsCLI.NFrames > 0)
                UsableFrames = Math.Min(UsableFrames, (int)OptionsCLI.NFrames);

            NewSource.FrameLimit = UsableFrames;

            #endregion            

            #region Add all items and their data hashes

            Console.WriteLine($"Adding {AllItems.Count} items.");

            if (AllItems.Count == 0)
            {
                Console.Error.WriteLine($"No micrographs or tilt series found to match these settings.");
                return;
            }

            {
                Console.Write("Calculating data hashes... ");

                int Done = 0;
                foreach (var item in AllItems)
                {
                    NewSource.Files.Add(item.GetDataHash(), item.Name);
                    Done++;

                    VirtualConsole.ClearLastLine();
                    Console.Write($"Calculating data hashes... {Done}/{AllItems.Count}");
                }

                VirtualConsole.ClearLastLine();
                Console.WriteLine($"Calculating data hashes... {Done}/{AllItems.Count}");
            }

            #endregion

            #region Check for overlapping hashes

            if (Population.Sources.Count > 0)
            {
                Console.Write("Checking for hash overlaps with existing data sources... ");

                string[] Overlapping = Population.Sources.SelectMany(s => s.Files.Where(f => NewSource.Files.ContainsKey(f.Key)).Select(f => f.Value).ToArray()).ToArray();
                if (Overlapping.Length > 0)
                {
                    string Offenders = "";
                    for (int o = 0; o < Math.Min(5, Overlapping.Length); o++)
                        Offenders += "\n" + Overlapping[o];
                    if (Overlapping.Length > 5)
                        Offenders += $"\n... and {Overlapping.Length - 5} more.";

                    Console.Error.WriteLine("\nThe new source contains files that are already used in this population:" + Offenders);
                    return;
                }

                Console.WriteLine("Done");
            }

            #endregion

            #region Commit

            {
                Console.Write("Committing initial version... ");

                NewSource.Commit();

                Console.WriteLine("Done");
            }

            Population.Sources.Add(NewSource);
            Population.Save();

            #endregion


            Console.WriteLine($"Data source created: {NewSource.Path}");
        }
    }
}
