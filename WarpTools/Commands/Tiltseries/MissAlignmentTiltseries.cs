using CommandLine;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using VYaml.Annotations;
using VYaml.Serialization;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_miss", HelpText = "Run MissAlignment to perform iterative self-supervised tilt series alignment")]
    [CommandRunner(typeof(MissAlignmentTiltseries))]
    class MissAlignmentTiltseriesOptions : DistributedOptions
    {
        [Option("angpix", HelpText = "Rescale tilt images to this pixel size; leave out to keep the original pixel size")]
        public double? AngPix { get; set; }

        [Option("mask", HelpText = "Apply mask to each image if available; masked areas will be filled with Gaussian noise")]
        public bool ApplyMask { get; set; }

        [Option("iterations", Required = true, HelpText = "Iteration settings as a semicolon-separated list of 'downsample:alignment' pairs. " +
            "Alignment modes: 'global', 'anchoring', or 'NxM' for local grid (e.g. '3x3'). " +
            "Example: '2:anchoring;2:anchoring;1:global;1:3x3'")]
        public string IterationSettings { get; set; }

        [Option("model_checkpoint", HelpText = "Path to model checkpoint file for weight initialization")]
        public string ModelCheckpoint { get; set; }

        [Option("max_epochs", Default = 30, HelpText = "Maximum epochs per iteration")]
        public int MaxEpochsPerIteration { get; set; }

        [Option("learning_rate", Default = 1e-3, HelpText = "Learning rate for model training")]
        public double LearningRate { get; set; }

        [Option("batch_size", Default = 32, HelpText = "Batch size for training")]
        public int BatchSize { get; set; }

        [Option("patch_size", Default = 96, HelpText = "Patch size for training and alignment")]
        public int PatchSize { get; set; }

        [Option("steps_per_epoch", Default = 1000, HelpText = "Number of steps per training epoch")]
        public int StepsPerEpoch { get; set; }

        [Option("patch_overlap", Default = 0.1, HelpText = "Tolerated overlap between patches used for optimizing the alignment")]
        public double PatchOverlap { get; set; }

        [Option("apply_ctf", Default = true, HelpText = "Apply CTF correction if CTF estimates are available")]
        public bool ApplyCTF { get; set; }

        [Option("seed", Default = 45132, HelpText = "Random seed for reproducibility")]
        public int Seed { get; set; }

        [Option("exe", Default = "miss", HelpText = "Name of the MissAlignment executable; must be in $PATH")]
        public string Executable { get; set; }

        [Option("delete_intermediate", HelpText = "Delete intermediate files generated during alignment")]
        public bool DeleteIntermediate { get; set; }
    }

    class MissAlignmentTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            MissAlignmentTiltseriesOptions CLI = options as MissAlignmentTiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (CLI.AngPix.HasValue && CLI.AngPix.Value < (double)Options.Import.BinnedPixelSize)
                throw new Exception("--angpix can't be smaller than the binned pixel size of the original data");
            else if (!CLI.AngPix.HasValue)
                CLI.AngPix = (double)Options.Import.BinnedPixelSize;

            if (CLI.MaxEpochsPerIteration < 1)
                throw new Exception("--max_epochs must be at least 1");

            if (CLI.BatchSize < 8)
                throw new Exception("--batch_size must be at least 8");

            if (CLI.PatchSize < 32)
                throw new Exception("--patch_size must be at least 32");

            if (CLI.StepsPerEpoch < 100)
                throw new Exception("--steps_per_epoch must be at least 100");

            if (CLI.PatchOverlap < 0 || CLI.PatchOverlap >= 1)
                throw new Exception("--patch_overlap must be between 0 and 1 (exclusive)");

            if (CLI.LearningRate <= 0)
                throw new Exception("--learning_rate must be positive");

            if (!string.IsNullOrEmpty(CLI.ModelCheckpoint) && !File.Exists(CLI.ModelCheckpoint))
                throw new Exception($"Model checkpoint file not found: {CLI.ModelCheckpoint}");

            if (!Helper.ExeutableIsOnPath(CLI.Executable))
                throw new Exception($"Executable '{CLI.Executable}' not found on PATH");

            // Parse iteration settings
            List<MissIterationSetting> iterationSettings = ParseIterationSettings(CLI.IterationSettings);
            if (iterationSettings.Count == 0)
                throw new Exception("At least one iteration setting must be specified");

            #endregion

            #region Create processing options

            var OptionsStack = (ProcessingOptionsTomoStack)Options.FillTomoProcessingBase(new ProcessingOptionsTomoStack());
            OptionsStack.ApplyMask = CLI.ApplyMask;
            OptionsStack.BinTimes = (decimal)Math.Log(CLI.AngPix.Value / (double)Options.Import.PixelSize, 2.0);

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            // Step 1: Create tilt series stacks for all series
            Console.WriteLine("Creating tilt series stacks...");
            IterateOverItems<TiltSeries>(Workers, CLI, (worker, t) =>
            {
                worker.TomoStack(t.Path, OptionsStack);
            });

            Console.Write("Saying goodbye to all workers because MissAlignment doesn't need them...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            // Get the data directory (where tilt series stacks are located)
            string trainingDir = CLI.OutputProcessing;

            string configPath = Path.Combine(trainingDir, "config.yaml");
            GenerateConfig(configPath, trainingDir, iterationSettings, CLI);

            Console.WriteLine($"Generated MissAlignment config at: {configPath}");

            // Step 3: Run MissAlignment (single process for all tilt series)
            Console.WriteLine($"Running MissAlignment with {iterationSettings.Count} iterations on {CLI.InputSeries.Length} tilt series...");
            RunMissAlignment(CLI.Executable, configPath, trainingDir);

            if (CLI.DeleteIntermediate)
            {
                Console.Write("Deleting intermediate stacks... ");

                foreach (var t in CLI.InputSeries)
                {
                    foreach (var dir in Directory.GetDirectories((t as TiltSeries).TiltStackDir))
                        if (!dir.EndsWith("thumbnails"))
                            Directory.Delete(dir, true);
                    
                    foreach (var file in Directory.GetFiles((t as TiltSeries).TiltStackDir))
                        File.Delete(file);
                }

                Console.WriteLine("Done");
            }
        }

        private void RunMissAlignment(string executable, string configPath, string workingDir)
        {
            string arguments = $"--config-file {configPath} --n-workers 5";

            Console.WriteLine($"Executing: {executable} {arguments}");

            Process miss = new Process
            {
                StartInfo =
                {
                    FileName = executable,
                    Arguments = arguments,
                    WorkingDirectory = workingDir,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };

            miss.OutputDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                    Console.WriteLine(e.Data);
            };

            miss.ErrorDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                    Console.Error.WriteLine(e.Data);
            };

            miss.Start();
            miss.BeginOutputReadLine();
            miss.BeginErrorReadLine();
            miss.WaitForExit();

            if (miss.ExitCode != 0)
                throw new Exception($"MissAlignment exited with code {miss.ExitCode}");
        }

        private List<MissIterationSetting> ParseIterationSettings(string input)
        {
            var result = new List<MissIterationSetting>();
            var pairs = input.Split(';', StringSplitOptions.RemoveEmptyEntries);

            foreach (var pair in pairs)
            {
                var parts = pair.Trim().Split(':');
                if (parts.Length != 2)
                    throw new Exception($"Invalid iteration setting format: '{pair}'. Expected 'downsample:alignment'");

                if (!int.TryParse(parts[0].Trim(), out int downsample) || downsample < 1)
                    throw new Exception($"Invalid downsample value: '{parts[0]}'. Must be a positive integer");

                string alignmentStr = parts[1].Trim().ToLowerInvariant();
                var setting = new MissIterationSetting { Downsample = downsample };

                if (alignmentStr == "global" || alignmentStr == "anchoring")
                {
                    setting.Alignment = alignmentStr;
                }
                else if (alignmentStr.Contains('x'))
                {
                    // Parse grid dimensions like "3x3"
                    var gridParts = alignmentStr.Split('x');
                    if (gridParts.Length != 2 ||
                        !int.TryParse(gridParts[0], out int gridX) ||
                        !int.TryParse(gridParts[1], out int gridY) ||
                        gridX < 1 || gridY < 1)
                    {
                        throw new Exception($"Invalid grid alignment format: '{alignmentStr}'. Expected 'NxM' (e.g. '3x3')");
                    }
                    setting.Alignment = new int[] { gridX, gridY };
                }
                else
                {
                    throw new Exception($"Unknown alignment mode: '{alignmentStr}'. " +
                        "Valid modes: 'global', 'anchoring', or 'NxM' for local deformation grid");
                }

                result.Add(setting);
            }

            return result;
        }

        private void GenerateConfig(string outputPath, 
                                    string trainingDir,
                                    List<MissIterationSetting> iterations, 
                                    MissAlignmentTiltseriesOptions cli)
        {
            var config = new MissAlignmentConfig
            {
                General = new MissGeneralConfig
                {
                    TrainingDirectory = trainingDir,
                    ApplyCtf = cli.ApplyCTF,
                    IterationSettings = iterations,
                    Seed = cli.Seed
                },
                ModelTraining = new MissModelTrainingConfig
                {
                    ModelArchitecture = "default",
                    ModelCheckpoint = cli.ModelCheckpoint,
                    LossMargin = 0.5,
                    LearningRate = cli.LearningRate,
                    WeightDecay = 1e-4,
                    MaxEpochsPerIteration = cli.MaxEpochsPerIteration,
                    WarmupSteps = 500,
                    MultistepLrScheduler = new MissLrSchedulerConfig
                    {
                        Milestones = [ 5, 15 ],
                        Gamma = 0.5
                    }
                },
                DataLoading = new MissDataLoadingConfig
                {
                    BatchSize = cli.BatchSize,
                    PatchSize = cli.PatchSize,
                    StepsPerEpoch = cli.StepsPerEpoch
                },
                ShiftGeneration = new MissShiftGenerationConfig
                {
                    TrajectoryProbability = 0.5,
                    TrajectoryMaxShift = 10.0,
                    JitterProbability = 0.5,
                    JitterMaxStd = 2.0,
                    OutlierProbability = 0.5,
                    OutlierMaxShift = 20.0,
                    FractureProbability = 0.5,
                    FractureMaxShift = 30.0
                },
                TiltSeriesAlignment = new MissTiltSeriesAlignmentConfig
                {
                    PatchSize = cli.PatchSize,
                    PatchOverlap = cli.PatchOverlap,
                    BatchSize = 16
                }
            };

            var yaml = YamlSerializer.SerializeToString(config);
            File.WriteAllText(outputPath, yaml);
        }
    }

    #region YAML Config Model Classes

    [YamlObject(NamingConvention.SnakeCase)]
    public partial class MissAlignmentConfig
    {
        public MissGeneralConfig General { get; set; }
        public MissModelTrainingConfig ModelTraining { get; set; }
        public MissDataLoadingConfig DataLoading { get; set; }
        public MissShiftGenerationConfig ShiftGeneration { get; set; }
        public MissTiltSeriesAlignmentConfig TiltSeriesAlignment { get; set; }
    }

    [YamlObject(NamingConvention.SnakeCase)]
    public partial class MissGeneralConfig
    {
        public string TrainingDirectory { get; set; }
        public bool ApplyCtf { get; set; }
        public List<MissIterationSetting> IterationSettings { get; set; }
        public int Seed { get; set; }
    }

    [YamlObject(NamingConvention.SnakeCase)]
    public partial class MissIterationSetting
    {
        public int Downsample { get; set; }
        public object Alignment { get; set; }
    }

    [YamlObject(NamingConvention.SnakeCase)]
    public partial class MissModelTrainingConfig
    {
        public string ModelArchitecture { get; set; }
        public string ModelCheckpoint { get; set; }
        public double LossMargin { get; set; }
        public double LearningRate { get; set; }
        public double WeightDecay { get; set; }
        public int MaxEpochsPerIteration { get; set; }
        public int WarmupSteps { get; set; }
        public MissLrSchedulerConfig MultistepLrScheduler { get; set; }
    }

    [YamlObject(NamingConvention.SnakeCase)]
    public partial class MissLrSchedulerConfig
    {
        public int[] Milestones { get; set; }
        public double Gamma { get; set; }
    }

    [YamlObject(NamingConvention.SnakeCase)]
    public partial class MissDataLoadingConfig
    {
        public int BatchSize { get; set; }
        public int PatchSize { get; set; }
        public int StepsPerEpoch { get; set; }
    }

    [YamlObject(NamingConvention.SnakeCase)]
    public partial class MissShiftGenerationConfig
    {
        public double TrajectoryProbability { get; set; }
        public double TrajectoryMaxShift { get; set; }
        public double JitterProbability { get; set; }
        public double JitterMaxStd { get; set; }
        public double OutlierProbability { get; set; }
        public double OutlierMaxShift { get; set; }
        public double FractureProbability { get; set; }
        public double FractureMaxShift { get; set; }
    }

    [YamlObject(NamingConvention.SnakeCase)]
    public partial class MissTiltSeriesAlignmentConfig
    {
        public int PatchSize { get; set; }
        public double PatchOverlap { get; set; }
        public int BatchSize { get; set; }
    }

    #endregion
}