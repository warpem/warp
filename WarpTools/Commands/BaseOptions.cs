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
    class BaseOptions
    {
        [OptionGroup("Data import settings", -100)]
        [Option("settings", Required = true, HelpText = "Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.")]
        public string SettingsPath { get; set; }

        [OptionGroup("Advanced data import & flow options", 101)]
        [Option("input_data", HelpText = "Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.")]
        public IEnumerable<string> InputData { get; set; }

        [Option("input_data_recursive", HelpText = "Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.")]
        public bool InputDataRecursive { get; set; }

        [Option("input_processing", HelpText = "Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.")]
        public string InputProcessing { get; set; }

        [Option("output_processing", HelpText = "Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.")]
        public string OutputProcessing { get; set; }

        public OptionsWarp Options { get; set; }

        public Movie[] InputSeries { get; set; }

        public SeriesType SeriesType { get; set; } = SeriesType.None;

        public virtual void Evaluate()
        {
            // Behavior summary:
            // 1. If --input_data is specified, use that instead of Options.Import.DataFolder
            //    and assume it to be relative to the current directory.
            //    Options.Import.ProcessingOrDataFolder might still be used to find processing results if no --input_processing is specified.
            // 2. If --input_processing is specified, it will override Options.Import.ProcessingOrDataFolder.
            // 3. If --output_processing is specified, it will override Options.Import.ProcessingOrDataFolder for output.
            // 4. If --input_data is not specified, files will be searched for relative to SettingsDataDirectory.
            // 5. If --input_processing or --output_processing is not specified, the path will be relative to the directory of the .settings file.
            // 6. If --input_data is a wildcard pattern, files will be searched for relative to the current directory or the absolute path specified in the pattern.
            // 7. If --input_data is a .txt file, files will be searched for relative to the current directory unless absolute paths are specified.

            Options = new OptionsWarp();
            Options.Load(SettingsPath);

            string SettingsDataDirectory = Helper.PathCombine(Environment.CurrentDirectory, Path.GetDirectoryName(SettingsPath), Options.Import.DataFolder);
            string WorkingDataDirectory = Environment.CurrentDirectory;

            // If OutputProcessing is not specified, fall back to InputProcessing value (if available)
            if (string.IsNullOrEmpty(OutputProcessing) && !string.IsNullOrEmpty(InputProcessing))
                OutputProcessing = InputProcessing;

            // Flags to keep track of whether InputProcessing and OutputProcessing have been explicitly set by the user
            bool InputProcessingOverride = !string.IsNullOrEmpty(InputProcessing);
            bool OutputProcessingOverride = !string.IsNullOrEmpty(OutputProcessing);

            // Set default directories based on Options.Import if not explicitly set
            if (string.IsNullOrEmpty(InputProcessing))
                InputProcessing = Options.Import.ProcessingOrDataFolder;

            if (string.IsNullOrEmpty(OutputProcessing) && !string.IsNullOrEmpty(InputProcessing))
                OutputProcessing = InputProcessing;

            // Check if separate processing directories are not specified
            bool NoSeparateOutputProcessing = string.IsNullOrEmpty(OutputProcessing) || OutputProcessing == Options.Import.DataFolder;
            bool NoSeparateInputProcessing = string.IsNullOrEmpty(InputProcessing) || InputProcessing == Options.Import.DataFolder;

            // Turn off recursive search if no separate processing directory specified
            if (NoSeparateOutputProcessing)
            {
                Console.WriteLine("No separate processing directory specified, turning off recursive search");
                InputDataRecursive = false;
                Options.Import.DoRecursiveSearch = false;
            }

            // Use input parameters from .settings file if --input_data is not specified
            if (InputData == null || InputData.Count() == 0)
            {
                Console.WriteLine($"No alternative input specified, will use input parameters from {Helper.PathToNameWithExtension(SettingsPath)}");
                Console.WriteLine($"File search will be relative to {SettingsDataDirectory}");

                string[] InputFiles = Directory.EnumerateFiles(SettingsDataDirectory,
                                                               Options.Import.Extension,
                                                               Options.Import.DoRecursiveSearch ? SearchOption.AllDirectories :
                                                                                                  SearchOption.TopDirectoryOnly).ToArray();

                InputData = InputFiles;
            }
            else
            {
                // Using the files specified in --input_data for processing
                Console.WriteLine("Using alternative input specified by --input_data");
                Console.WriteLine($"File search will be relative to {WorkingDataDirectory}");
                List<string> AllInputFiles = new List<string>();

                foreach (var pattern in InputData)
                {
                    // If pattern is a .txt file, read each line as a file path
                    if (Path.GetExtension(pattern).ToLower() == ".txt")
                    {
                        AllInputFiles.AddRange(File.ReadAllLines(pattern));
                    }
                    // If pattern contains wildcards '*', '?'
                    else if (pattern.Contains("*") || pattern.Contains("?"))
                    {
                        string PatternDir = Path.GetDirectoryName(pattern);
                        string PatternFile = Path.GetFileName(pattern);

                        // Find all matching files based on the pattern
                        AllInputFiles.AddRange(Directory.EnumerateFiles(Helper.PathCombine(WorkingDataDirectory, PatternDir), PatternFile, InputDataRecursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly));
                    }
                    else
                    {
                        AllInputFiles.Add(Helper.PathCombine(WorkingDataDirectory, pattern));
                    }
                }

                InputData = AllInputFiles.ToArray();
            }

            // Check for duplicate file names in the input list
            var InputDataUnique = InputData.Select(p => Helper.PathToNameWithExtension(p)).Distinct();
            if (InputData.Count() != InputDataUnique.Count())
                throw new Exception("Duplicate file names found in input data list! When using recursive search, please make sure that all file names are unique because the per-file processing results will all go in the same folder.");

            Console.WriteLine($"{InputData.Count()} files found");

            #region Processing results

            // Resolve InputProcessing and OutputProcessing directories based on whether they were overridden
            if (!InputProcessingOverride)
                InputProcessing = Helper.PathCombine(Environment.CurrentDirectory, Path.GetDirectoryName(SettingsPath), InputProcessing);
            else
                InputProcessing = Helper.PathCombine(WorkingDataDirectory, InputProcessing);

            if (!OutputProcessingOverride)
                OutputProcessing = Helper.PathCombine(Environment.CurrentDirectory, Path.GetDirectoryName(SettingsPath), OutputProcessing);
            else
                OutputProcessing = Helper.PathCombine(WorkingDataDirectory, OutputProcessing);

            InputSeries = new Movie[InputData.Count()];

            Console.WriteLine("Parsing previous results for each item, if available...");
            Console.Write($"0/{InputSeries.Length}");

            int NDone = 0;
            int NResults = 0;
            Helper.ForCPU(0, InputSeries.Length, 8, null, (i, threadID) =>
            {
                string FilePath = InputData.ElementAt(i);

                if (!File.Exists(FilePath))
                    throw new Exception($"File not found: {FilePath}");

                bool IsTomo = Path.GetExtension(FilePath).ToLower() == ".tomostar";
                string DataDir = null;
                
                if (Path.GetFullPath(Path.GetDirectoryName(FilePath)) != Path.GetFullPath(InputProcessing))
                    DataDir = Path.GetDirectoryName(FilePath);

                if (IsTomo)
                    InputSeries[i] = new TiltSeries(Helper.PathCombine(InputProcessing, Path.GetFileName(FilePath)), DataDir);
                else
                    InputSeries[i] = new Movie(Helper.PathCombine(InputProcessing, Path.GetFileName(FilePath)), DataDir, Array.Empty<string>());

                bool HasResults = File.Exists(Helper.PathCombine(InputProcessing, Path.GetFileNameWithoutExtension(FilePath) + ".xml"));

                if (OutputProcessingOverride)
                    InputSeries[i].Path = Helper.PathCombine(OutputProcessing, Path.GetFileName(FilePath));

                lock (InputSeries)
                {
                    NDone++;
                    if (HasResults)
                        NResults++;

                    if (NDone % 10 == 0 || NDone == InputSeries.Length)
                    {
                        VirtualConsole.ClearLastLine();
                        Console.Write($"{NDone}/{InputSeries.Length}, previous metadata found for {NResults}");
                    }
                }
            }, null);

            Console.WriteLine("");

            #endregion

            if (InputSeries.Any())
            {
                if (InputSeries.First() is TiltSeries)
                    SeriesType = SeriesType.Tilt;
                else if (InputSeries.First() is Movie)
                    SeriesType = SeriesType.Frame;
                else
                    throw new Exception($"Unknown series type: {InputSeries.First().GetType()}");
            }
        }
    }

    public enum SeriesType
    {
        None = 0,
        Frame = 1,
        Tilt = 2
    }
}
