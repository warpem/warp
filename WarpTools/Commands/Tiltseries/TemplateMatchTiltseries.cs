using CommandLine;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Headers;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_template_match", HelpText = "Match previously reconstructed tomograms against a 3D template, producing a list of the highest-scoring matches")]
    [CommandRunner(typeof(TemplateMatchTiltseries))]
    class TemplateMatchTiltseriesOptions : DistributedOptions
    {
        [Option("tomo_angpix", Required = true, HelpText = "Pixel size of the reconstructed tomograms in Angstrom")]
        public double TomoAngPix { get; set; }

        [Option("template_path", HelpText = "Path to the template file")]
        public string TemplatePath { get; set; }
        public string FlippedTemplatePath { get; set; } = null;

        [Option("template_emdb", HelpText = "Instead of providing a local map, download the EMDB entry with this ID and use its main map")]
        public int? TemplateEMDB { get; set; }

        [Option("template_angpix", HelpText = "Pixel size of the template; leave empty to use value from map header")]
        public double? TemplateAngPix { get; set; }

        [Option("template_diameter", Required = true, HelpText = "Template diameter in Angstrom")]
        public int TemplateDiameter { get; set; }

        [Option("template_flip", HelpText = "Mirror the template along the X axis to flip the handedness; '_flipx' will be added to the template's name")]
        public bool TemplateFlip { get; set; }

        [Option("symmetry", Default = "C1", HelpText = "Symmetry of the template, e.g. C1, D7, O")]
        public string TemplateSymmetry { get; set; }

        [Option("subdivisions", Default = 3, HelpText = "Number of subdivisions defining the angular search step: 2 = 15° step, 3 = 7.5°, 4 = 3.75° and so on")]
        public int HealpixOrder { get; set; }

        [Option("optimize_poses", HelpText = "Additionally optimize poses for each detected position using a local GPU gradient-based search")]
        public bool OptimizePoses { get; set; }

        [Option("match_topk", Default = 8, HelpText = "With --optimize_poses, retain this many orientation scores per voxel; leaderboard GPU memory is 8*K bytes per padded voxel")]
        public int MatchTopK { get; set; }

        [Option("refine_starts", Default = 32, HelpText = "Maximum GPU pose hypotheses per peak, pooled from that voxel and its six neighbors")]
        public int RefineStarts { get; set; }

        [Option("refine_optimizer", Default = "bfgs", HelpText = "GPU pose optimizer: bfgs (FP32) or gauss-newton (original trust-region implementation)")]
        public string RefineOptimizer { get; set; } = "bfgs";

        [Option("refine_iterations", Default = 90, HelpText = "Maximum accepted GPU optimization steps per hypothesis and resolution stage")]
        public int RefineIterations { get; set; }

        [Option("refine_merge_fraction", Default = 0.005, HelpText = "Merge thresholds as a fraction of the current band pixel, for translation and rotation displacement at the template edge; 0 disables merging")]
        public double RefineMergeFraction { get; set; }

        [Option("refine_max_shift", Default = 0.0, HelpText = "Maximum displacement per coordinate from the proposal center, in Angstrom; 0 uses three tomogram pixels")]
        public double RefineMaxShift { get; set; }

        [Option("refine_noise_patches", Default = 32, HelpText = "Unselected patches per tilt for the fixed radial background power estimate")]
        public int RefineNoisePatches { get; set; }

        [Option("refine_fit_bfactor", HelpText = "Jointly fit amplitude and B at each final pose; write envelope diagnostics without changing particle scores or selection")]
        public bool RefineFitBfactor { get; set; }

        [Option("refine_fit_highpass", Default = 30.0, HelpText = "Amplitude/B fitting only: use frequencies above 1/value in Angstrom; 0 uses the full refinement band") ]
        public double RefineFitHighpass { get; set; } = 30;

        [Option("refine_export_tilt_spectra", HelpText = "Export per-tilt sufficient statistics for experimental shared tilt-scale calibration; requires --refine_fit_bfactor")]
        public bool RefineExportTiltSpectra { get; set; }

        [Option("decoy_templates", Separator = ',', HelpText = "Optional comma-separated decoy map paths. Each runs the complete search and writes empirical false-count diagnostics. Maps must match the target dimensions and pixel size; --template_angpix overrides all map headers")]
        public IEnumerable<string> DecoyTemplates { get; set; } = Array.Empty<string>();

        [Option("optimize_poses_angpix", HelpText = "Minimum pixel size to use for pose optimization. Leave empty to set it to --tomo_angpix")]
        public double? OptimizePosesAngPix { get; set; }

        [Option("optimize_poses_steps", Default = 1, HelpText = "Number of steps in which to decrease the pixel size from --tomo_angpix to --optimize_poses_angpix")]
        public int OptimizePosesSteps { get; set; }

        [Option("tilt_range", HelpText = "Limit the range of angles between the reference's Z axis and the tomogram's XY plane to plus/minus this value, in °; " +
                                         "useful for matching filaments lying flat in the XY plane")]
        public double? TiltRange { get; set; }

        [Option("batch_angles", Default = 8, HelpText = "How many orientations to evaluate at once; memory consumption scales linearly with this; " +
                                                         "higher than 32 probably won't lead to speed-ups")]
        public int BatchAngles { get; set; }

        [Option("peak_distance", HelpText = "Minimum distance (in Angstrom) between peaks; leave empty to use template diameter")]
        public int? PeakDistance { get; set; }

        [Option("npeaks", Default = 2000, HelpText = "Maximum number of peak positions to save")]
        public int PeakNumber { get; set; }

        [Option("tophat", HelpText = "Filter peaks by applying tophat transform with this connectivity level. Valid values: 1, 2, 3")]
        public int? Tophat { get; set; }

        [Option("dont_normalize", HelpText = "Don't set score distribution to median = 0, stddev = 1")]
        public bool DontNormalizeScores { get; set; }

        [Option("whiten", HelpText = "Perform spectral whitening to give higher-resolution information more weight; " +
                                     "this can help when the alignments are already good and you need more selective matching")]
        public bool Whiten { get; set; }

        [Option("lowpass", Default = 1.0, HelpText = "Gaussian low-pass filter to be applied to template and tomogram, in fractions of Nyquist; " +
                                                     "1.0 = no low-pass, <1.0 = low-pass")]
        public double Lowpass { get; set; }

        [Option("lowpass_sigma", Default = 0.1, HelpText = "Sigma (i.e. fall-off) of the Gaussian low-pass filter, in fractions of Nyquist; " +
                                                           "larger value = slower fall-off")]
        public double LowpassSigma { get; set; }

        [Option("max_missing_tilts", Default = 2, HelpText = "Dismiss positions not covered by at least this many tilts; " +
                                                             "set to -1 to disable position culling")]
        public int MaxMissingTilts { get; set; }

        [Option("reuse_results", HelpText = "Reuse correlation volumes from a previous run if available, only extract peak positions")]
        public bool ReuseResults { get; set; }

        [Option("check_hand", Default = 0, HelpText = "Also try a flipped version of the template on this many tomograms to see what geometric hand they have")]
        public int CheckHandN { get; set; }

        [Option("subvolume_size", Default = 192, HelpText = "Matching is performed locally using sub-volumes of this size in pixel")]
        public int SubVolumeSize { get; set; }

        [Option("override_suffix", HelpText = "Override the default STAR file suffix derived from the template name; " +
                                              "must include the leading underscore if you want to have it")]
        public string OverrideSuffix { get; set; } = "";

        [Option("dont_save_corr", HelpText = "Don't save volume with correlation scores. Makes --reuse_results impossible later.")]
        public bool DontSaveCorr { get; set; }

        [Option("dont_save_angles", HelpText = "Don't save volume with angle information. Makes --reuse_results impossible later.")]
        public bool DontSaveAngles { get; set; }
    }

    class TemplateMatchTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            TemplateMatchTiltseriesOptions CLI = options as TemplateMatchTiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (CLI.TomoAngPix <= 0)
                throw new Exception("--tomo_angpix can't be 0 or negative");

            if (string.IsNullOrEmpty(CLI.TemplatePath) && !CLI.TemplateEMDB.HasValue)
                throw new Exception("Either --template_path or --template_emdb must be specified");

            if (!string.IsNullOrEmpty(CLI.TemplatePath) && CLI.TemplateEMDB.HasValue)
                throw new Exception("Only one of --template_path and --template_emdb can be specified");

            if (!string.IsNullOrEmpty(CLI.TemplatePath) && !File.Exists(CLI.TemplatePath))
                throw new Exception("Template file doesn't exist");

            if (CLI.TemplateEMDB.HasValue && CLI.TemplateEMDB.Value <= 0)
                throw new Exception("--template_emdb can't be 0 or negative");

            if (CLI.TemplateAngPix.HasValue && CLI.TemplateAngPix.Value <= 0)
                throw new Exception("--template_angpix can't be 0 or negative");

            if (CLI.TemplateDiameter <= 0)
                throw new Exception("--template_diameter can't be 0 or negative");

            try
            {
                Symmetry TemplateSymmetry = new Symmetry(CLI.TemplateSymmetry);
            }
            catch (Exception e)
            {
                throw new Exception("Invalid --symmetry specified: " + e.Message);
            }

            if (CLI.HealpixOrder < 0)
                throw new Exception("--subdivisions can't be negative");

            if (CLI.TiltRange != null && (CLI.TiltRange.Value > 90 || CLI.TiltRange.Value < 0))
                throw new Exception("--tilt_range must be between 0 and 90");

            if (CLI.BatchAngles < 1)
                throw new Exception("--batch_angles must be positive");

            CLI.RefineOptimizer = CLI.RefineOptimizer?.ToLowerInvariant();
            if (CLI.RefineOptimizer != "bfgs" && CLI.RefineOptimizer != "gauss-newton")
                throw new Exception("--refine_optimizer must be bfgs or gauss-newton");
            if (!double.IsFinite(CLI.RefineMergeFraction) || CLI.RefineMergeFraction < 0 || CLI.RefineMergeFraction > 0.5)
                throw new Exception("--refine_merge_fraction must lie between 0 (disabled) and 0.5");
            if (CLI.MatchTopK < 1 || CLI.RefineStarts < 1 || CLI.RefineIterations < 1 || CLI.RefineNoisePatches < 2)
                throw new Exception("--match_topk, --refine_starts and --refine_iterations must be positive; --refine_noise_patches must be at least 2");
            if (!double.IsFinite(CLI.RefineMaxShift) || CLI.RefineMaxShift < 0)
                throw new Exception("--refine_max_shift must be finite and nonnegative");
            if (CLI.OptimizePosesSteps < 1 || (CLI.OptimizePosesAngPix.HasValue && (!double.IsFinite(CLI.OptimizePosesAngPix.Value) || CLI.OptimizePosesAngPix.Value <= 0 || CLI.OptimizePosesAngPix.Value > CLI.TomoAngPix)))
                throw new Exception("Pose optimization requires positive steps and a pixel size no larger than --tomo_angpix");
            if (CLI.OptimizePoses && CLI.ReuseResults)
                throw new Exception("--optimize_poses cannot reuse legacy single-orientation volumes; omit --reuse_results to build top-K lists");

            if (CLI.RefineFitBfactor && !CLI.OptimizePoses)
                throw new Exception("--refine_fit_bfactor requires --optimize_poses");
            if (CLI.RefineExportTiltSpectra && !CLI.RefineFitBfactor)
                throw new Exception("--refine_export_tilt_spectra requires --refine_fit_bfactor");

            if (!double.IsFinite(CLI.RefineFitHighpass) || CLI.RefineFitHighpass < 0)
                throw new Exception("--refine_fit_highpass must be finite and nonnegative");
            if (CLI.RefineFitBfactor && CLI.RefineFitHighpass > 0 &&
                CLI.RefineFitHighpass <= 2 * (CLI.OptimizePosesAngPix ?? CLI.TomoAngPix) / Math.Min(1, CLI.Lowpass))
                throw new Exception("--refine_fit_highpass must be coarser than the final refinement resolution; otherwise the fitting band is empty");

            string[] DecoyPaths = (CLI.DecoyTemplates ?? Array.Empty<string>()).Select(Path.GetFullPath).ToArray();
            if (DecoyPaths.Length > 0 && !CLI.OptimizePoses)
                throw new Exception("--decoy_templates requires --optimize_poses");
            foreach (string decoy in DecoyPaths)
                if (!File.Exists(decoy))
                    throw new FileNotFoundException("Decoy template not found", decoy);
            if (DecoyPaths.Select(Path.GetFileNameWithoutExtension).Distinct(StringComparer.Ordinal).Count() != DecoyPaths.Length)
                throw new Exception("Decoy templates must have distinct filenames");
            bool TemplatePixelOverride = CLI.TemplateAngPix.HasValue;

            if (CLI.PeakDistance.HasValue && CLI.PeakDistance.Value <= 0)
                throw new Exception("--peak_distance can't be 0 or negative");

            if (CLI.PeakNumber <= 0)
                throw new Exception("--npeaks can't be 0 or negative");

            if (CLI.CheckHandN < 0)
                throw new Exception("--check_hand can't be negative");

            if (CLI.TemplateFlip && CLI.CheckHandN > 0)
                throw new Exception("--template_flip and --check_hand can't be used together");

            if (CLI.SubVolumeSize < 64)
                throw new Exception("--subvolume_size can't be lower than 64");

            if (!double.IsFinite(CLI.Lowpass) || CLI.Lowpass < 0 || CLI.Lowpass > 1 || CLI.OptimizePoses && CLI.Lowpass == 0)
                throw new Exception("--lowpass must be between 0 and 1");

            if (CLI.LowpassSigma < 0)
                throw new Exception("--lowpass_sigma can't be negative");


            #endregion

            #region Create processing options

            Options.Tasks.TomoFullReconstructPixel = (decimal)CLI.TomoAngPix;

            if (CLI.TemplateAngPix.HasValue)
                Options.Tasks.TomoMatchTemplatePixel = (decimal)CLI.TemplateAngPix;
            Options.Tasks.TomoMatchTemplateDiameter = CLI.TemplateDiameter;
            Options.Tasks.TomoMatchPeakDistance = CLI.PeakDistance.HasValue ? CLI.PeakDistance.Value : CLI.TemplateDiameter;
            Options.Tasks.TomoMatchTemplateFraction = 1;

            Options.Tasks.TomoMatchHealpixOrder = CLI.HealpixOrder;
            Options.Tasks.TomoMatchBatchAngles = CLI.BatchAngles;
            Options.Tasks.TomoMatchSymmetry = CLI.TemplateSymmetry;
            Options.Tasks.TomoMatchNResults = CLI.PeakNumber;

            Options.Tasks.ReuseCorrVolumes = CLI.ReuseResults;
            Options.Tasks.TomoMatchWhitenSpectrum = CLI.Whiten;

            var OptionsMatch = Options.GetProcessingTomoFullMatch();

            OptionsMatch.UseTophat = CLI.Tophat ?? 0;
            OptionsMatch.OptimizePoses = CLI.OptimizePoses;
            OptionsMatch.MatchTopK = CLI.MatchTopK;
            OptionsMatch.RefineStarts = CLI.RefineStarts;
            OptionsMatch.RefineOptimizer = CLI.RefineOptimizer;
            OptionsMatch.RefineIterations = CLI.RefineIterations;
            OptionsMatch.RefineMergeFraction = (decimal)CLI.RefineMergeFraction;
            OptionsMatch.RefineMaxShift = (decimal)CLI.RefineMaxShift;
            OptionsMatch.RefineNoisePatches = CLI.RefineNoisePatches;
            OptionsMatch.RefineFitBfactor = CLI.RefineFitBfactor;
            OptionsMatch.RefineFitHighpass = (decimal)CLI.RefineFitHighpass;
            OptionsMatch.RefineExportTiltSpectra = CLI.RefineExportTiltSpectra;
            OptionsMatch.OptimizePosesAngPix = (decimal?)CLI.OptimizePosesAngPix;
            OptionsMatch.OptimizePosesSteps = CLI.OptimizePosesSteps;
            OptionsMatch.TiltRange = CLI.TiltRange != null ? (decimal)CLI.TiltRange.Value : -1;
            OptionsMatch.SubVolumeSize = CLI.SubVolumeSize;
            OptionsMatch.Supersample = 1;
            OptionsMatch.MaxMissingTilts = CLI.MaxMissingTilts;
            OptionsMatch.NormalizeScores = !CLI.DontNormalizeScores;
            OptionsMatch.Lowpass = (decimal)CLI.Lowpass;
            OptionsMatch.LowpassSigma = (decimal)CLI.LowpassSigma;
            
            OptionsMatch.OverrideSuffix = CLI.OverrideSuffix ?? "";

            OptionsMatch.DontSaveCorrVolume = CLI.DontSaveCorr;
            OptionsMatch.DontSaveAngleIDVolume = CLI.DontSaveAngles;

            #endregion

            string TemplateDir = Path.Combine(CLI.OutputProcessing, "template");
            Directory.CreateDirectory(TemplateDir);

            #region Download EMDB if necessary

            if (CLI.TemplateEMDB.HasValue)
            {
                // EMD IDs below 10000 are padded with zeros
                string ID = CLI.TemplateEMDB.Value.ToString("D4");
                string Url = $"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{ID}/map/emd_{ID}.map.gz";
                string OutputPath = Path.Combine(TemplateDir, $"emd_{ID}.mrc");

                if (!File.Exists(OutputPath))
                {
                    byte[] DownloadedData = await DownloadFileAsync(Url);

                    Console.Write("Extracting downloaded map...");
                    ExtractGzArchive(DownloadedData, OutputPath);
                    Console.WriteLine(" Done");
                }
                else
                {
                    Console.WriteLine($"EMD-{ID} already exists in {OutputPath}, skipping download");
                }

                CLI.TemplatePath = OutputPath;
            }
            else
            {
                CLI.TemplatePath = Helper.PathCombine(Environment.CurrentDirectory, CLI.TemplatePath);
            }

            #endregion

            #region Prepare template

            using Image TemplateOri = Image.FromFile(CLI.TemplatePath);
            if (CLI.TemplateAngPix == null)
            {
                if (TemplateOri.PixelSize <= 0)
                    throw new Exception("Couldn't determine pixel size from template, please specify --template_angpix");

                CLI.TemplateAngPix = TemplateOri.PixelSize;
                OptionsMatch.TemplatePixel = (decimal)TemplateOri.PixelSize;
                Console.WriteLine($"Setting --template_angpix to {TemplateOri.PixelSize} based on template map");
            }

            foreach (string decoy in DecoyPaths)
            {
                if (string.Equals(decoy, Path.GetFullPath(CLI.TemplatePath), StringComparison.Ordinal))
                    throw new Exception("A decoy template cannot be the target template itself");
                MapHeader header = MapHeader.ReadFromFile(decoy);
                if (header.Dimensions != TemplateOri.Dims)
                    throw new Exception($"Decoy template {decoy} must have the same dimensions as the target ({TemplateOri.Dims})");
                if (!TemplatePixelOverride)
                {
                    double expected = CLI.TemplateAngPix.Value;
                    double tolerance = Math.Max(1e-6, expected * 1e-4);
                    foreach (float pixelSize in new[] { header.PixelSize.X, header.PixelSize.Y, header.PixelSize.Z })
                        if (!float.IsFinite(pixelSize) || Math.Abs(pixelSize - expected) > tolerance)
                            throw new Exception($"Decoy template {decoy} pixel size does not match the target ({expected} Å). Use --template_angpix only if all maps share that actual sampling.");
                }
            }

            Image TemplateFlipped = null;
            if (CLI.TemplateFlip || CLI.CheckHandN > 0)
            {
                Console.Write("Preparing flipped template... ");

                TemplateFlipped = TemplateOri.AsFlippedX();

                string FlippedPath = Path.Combine(TemplateDir, Path.GetFileNameWithoutExtension(CLI.TemplatePath) + "_flipx.mrc");
                CLI.FlippedTemplatePath = FlippedPath;
                TemplateFlipped.WriteMRC(FlippedPath, (float)CLI.TemplateAngPix, true);

                if (CLI.TemplateFlip)
                    CLI.TemplatePath = FlippedPath;

                Console.WriteLine("Done");
                TemplateFlipped.Dispose();
            }

            #endregion

            {
                var HealpixAngles = Helper.GetHealpixAngles(OptionsMatch.HealpixOrder, OptionsMatch.Symmetry);
                if (CLI.TiltRange >= 0)
                {
                    float Limit = MathF.Sin((float)CLI.TiltRange * Helper.ToRad);
                    HealpixAngles = HealpixAngles.Where(a => MathF.Abs(Matrix3.Euler(a).C3.Z) <= Limit).ToArray();
                }
                Console.WriteLine($"Using {HealpixAngles.Length} orientations for matching");
                if (HealpixAngles.Length == 0)
                    throw new Exception("Can't match with 0 orientations, please increase --tilt_range");
            }

            // Local helper: one template-matching task per current InputSeries item,
            // distributed via the filesystem work queue, against the given template.
            // The optional hook runs orchestrator-side after each item completes — used
            // by --check_hand to read back the per-series peak STAR the worker wrote.
            string PeakTablePath(TiltSeries series) => Path.Combine(series.MatchingDir,
                TiltSeries.ToTomogramWithPixelSize(series.Path, OptionsMatch.BinnedPixelSizeMean) +
                (string.IsNullOrWhiteSpace(OptionsMatch.OverrideSuffix) ? "_" + OptionsMatch.TemplateName : OptionsMatch.OverrideSuffix) + ".star");
            string CalibrationPath(string peakTablePath) => Path.Combine(Path.GetDirectoryName(peakTablePath),
                Path.GetFileNameWithoutExtension(peakTablePath) + "_decoy_calibration.tsv");

            int MatchRun = 0;
            void RunMatch(string templatePath, Action<TiltSeries> onSuccess = null, bool invalidateCalibration = false)
            {
                foreach (var item in CLI.InputSeries)
                    item.ProcessingStatus = ProcessingStatus.Unprocessed;

                int run = ++MatchRun;
                string templateId = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
                    Path.GetFullPath(templatePath) + "|" + OptionsMatch.TemplateName))).Substring(0, 12).ToLowerInvariant();
                var previousOutputs = new Dictionary<TiltSeries, (string Path, bool Exists, long Length, DateTime Modified)>();
                CLI.DistributeItems<TiltSeries>(
                    buildTask: (t, i) =>
                    {
                        if (onSuccess != null)
                        {
                            string outputPath = PeakTablePath(t);
                            FileInfo previous = new FileInfo(outputPath);
                            previousOutputs[t] = (outputPath, previous.Exists,
                                previous.Exists ? previous.Length : 0, previous.Exists ? previous.LastWriteTimeUtc : DateTime.MinValue);
                            if (invalidateCalibration)
                            {
                                // A first-ever decoy run has no matching directory yet.
                                Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
                                File.Delete(CalibrationPath(outputPath));
                            }
                        }
                        var task = new TaskItem
                        {
                            TaskId = $"{i:D7}-match-{run:D3}-{templateId}-{t.RootName}",
                            Stage = "preprocess",
                            RequiresGpu = true,
                            Init = Array.Empty<NamedSerializableObject>(),
                            Main = new[]
                            {
                                WorkerCommands.TomoMatch(t.Path, OptionsMatch, templatePath),
                                WorkerCommands.GcCollect(),
                            },
                        };
                        task.ComputeInitFingerprint();
                        return task;
                    },
                    onSuccess: onSuccess == null ? null : t =>
                    {
                        var previous = previousOutputs[t];
                        FileInfo current = new FileInfo(previous.Path);
                        if (!current.Exists || (previous.Exists && current.Length == previous.Length && current.LastWriteTimeUtc == previous.Modified))
                            throw new IOException($"Template matching did not produce a fresh peak table: {previous.Path}");
                        onSuccess(t);
                    });
            }

            if (CLI.CheckHandN > 0)
            {
                var AllItems = CLI.InputSeries;
                List<float> ScoresOriginal = new();
                List<float> ScoresFlipped = new();

                // onSuccess runs single-threaded on the result-polling thread, so the
                // score lists need no locking (unlike the old concurrent callbacks).
                Action<TiltSeries> CollectTopPeaks(List<float> scores) => t =>
                {
                    List<float> PeakValues = Star.LoadFloat(PeakTablePath(t), "rlnAutopickFigureOfMerit").ToList();
                    PeakValues.Sort();
                    PeakValues = PeakValues.TakeLast(20).ToList();
                    scores.AddRange(PeakValues);
                };

                CLI.InputSeries = AllItems.Take(CLI.CheckHandN).ToArray();

                // Unflipped
                {
                    Console.WriteLine("Testing matching with original template:");
                    OptionsMatch.TemplateName = Path.GetFileNameWithoutExtension(CLI.TemplatePath);
                    RunMatch(CLI.TemplatePath, CollectTopPeaks(ScoresOriginal));
                    Console.WriteLine($"Average top peak value with original template: {ScoresOriginal.Average():F3}");
                }

                // Flipped
                {
                    Console.WriteLine("Testing matching with flipped template:");
                    OptionsMatch.TemplateName = Path.GetFileNameWithoutExtension(CLI.FlippedTemplatePath);
                    RunMatch(CLI.FlippedTemplatePath, CollectTopPeaks(ScoresFlipped));
                    Console.WriteLine($"Average top peak value with flipped template: {ScoresFlipped.Average():F3}");
                }

                if (ScoresFlipped.Average() > ScoresOriginal.Average())
                {
                    Console.WriteLine("Flipped template has higher peak values, using it for further processing");
                    CLI.TemplatePath = CLI.FlippedTemplatePath;
                }
                else
                {
                    Console.WriteLine("Original template has higher peak values, using it for further processing");
                }

                CLI.InputSeries = AllItems;
            }

            OptionsMatch.TemplateName = Path.GetFileNameWithoutExtension(CLI.TemplatePath);

            if (DecoyPaths.Length == 0)
            {
                RunMatch(CLI.TemplatePath);
                return;
            }

            string TargetName = OptionsMatch.TemplateName;
            string TargetSuffix = OptionsMatch.OverrideSuffix;
            var TargetScores = new Dictionary<TiltSeries, (string Path, float[] Scores)>();
            var DecoyScores = DecoyPaths.Select(_ => new Dictionary<TiltSeries, float[]>()).ToArray();
            float[] ReadFinalScores(TiltSeries series)
            {
                var table = new Star(PeakTablePath(series));
                if (!table.HasColumn("wrpTemplateMatchProjectionZ"))
                    throw new IOException("Decoy diagnostics require newly refined projection-space scores.");
                float[] scores = table.GetFloat("rlnAutopickFigureOfMerit");
                if (scores.Any(s => !float.IsFinite(s)))
                    throw new IOException("Peak table contains nonfinite final detection scores.");
                return scores;
            }

            RunMatch(CLI.TemplatePath, t => TargetScores[t] = (PeakTablePath(t), ReadFinalScores(t)), invalidateCalibration: true);
            try
            {
                for (int decoy = 0; decoy < DecoyPaths.Length; decoy++)
                {
                    int index = decoy;
                    string tag = $"__decoy_{decoy + 1:D3}_{Path.GetFileNameWithoutExtension(DecoyPaths[decoy])}";
                    OptionsMatch.TemplateName = TargetName + tag;
                    OptionsMatch.OverrideSuffix = string.IsNullOrWhiteSpace(TargetSuffix) ? "" : TargetSuffix + tag;
                    Console.WriteLine($"Running complete decoy search {decoy + 1}/{DecoyPaths.Length}: {DecoyPaths[decoy]}");
                    RunMatch(DecoyPaths[decoy], t => DecoyScores[index][t] = ReadFinalScores(t));
                }
            }
            finally
            {
                OptionsMatch.TemplateName = TargetName;
                OptionsMatch.OverrideSuffix = TargetSuffix;
            }

            int Calibrated = 0;
            foreach (TiltSeries series in CLI.InputSeries)
            {
                if (!TargetScores.TryGetValue(series, out var target) || DecoyScores.Any(search => !search.ContainsKey(series)))
                {
                    Console.WriteLine($"Skipping decoy diagnostics for {series.RootName}: target and every decoy must complete successfully.");
                    continue;
                }
                float[][] searches = DecoyScores.Select(search => search[series]).ToArray();
                string output = CalibrationPath(target.Path);
                string temporary = output + ".tmp";
                using (var writer = new StreamWriter(temporary, false, new UTF8Encoding(false)))
                {
                    writer.WriteLine("# Empirical false-output counts for this series and this complete search procedure.");
                    writer.WriteLine("# Validity depends on decoys reproducing false matches to the target; these are not posterior probabilities or calibrated FDR values.");
                    writer.WriteLine("# Zero observed exceedances means unresolved tail support, not proof of zero false outputs.");
                    writer.WriteLine($"# Target STAR: {target.Path}");
                    for (int decoy = 0; decoy < DecoyPaths.Length; decoy++)
                        writer.WriteLine($"# Decoy {decoy + 1}: {DecoyPaths[decoy]}");
                    writer.WriteLine("target_row\ttarget_score\tdecoy_exceedances\tdecoy_searches\testimated_false_count\tone_count_resolution\tcalibration_support\tcounts_by_decoy");
                    for (int row = 0; row < target.Scores.Length; row++)
                    {
                        var count = TemplateMatchDecoyCalibration.Count(target.Scores[row], searches);
                        writer.WriteLine(string.Join("\t", new[]
                        {
                            (row + 1).ToString(CultureInfo.InvariantCulture),
                            target.Scores[row].ToString("R", CultureInfo.InvariantCulture),
                            count.TotalExceedances.ToString(CultureInfo.InvariantCulture),
                            count.SearchCount.ToString(CultureInfo.InvariantCulture),
                            count.MeanCount.ToString("R", CultureInfo.InvariantCulture),
                            count.OneCountResolution.ToString("R", CultureInfo.InvariantCulture),
                            count.TailUnresolved ? "unresolved_zero_exceedances" : "observed_exceedances",
                            string.Join(",", count.CountsBySearch)
                        }));
                    }
                }
                File.Move(temporary, output, true);
                Calibrated++;
            }
            Console.WriteLine($"Wrote decoy diagnostics for {Calibrated} tilt series. Validate decoy false-match behavior before interpreting these counts as significance.");
        }

        async Task<byte[]> DownloadFileAsync(string url)
        {
            HttpClient HttpClient = new HttpClient();
            HttpResponseMessage Response = await HttpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);

            if (!Response.IsSuccessStatusCode)
                throw new HttpRequestException($"Failed to download {url}: HTTP {(int)Response.StatusCode} ({Response.StatusCode}).");

            Console.Write($"Downloading map from EMDB: 0%");
            using (MemoryStream memoryStream = new MemoryStream())
            {
                // Get content stream and length
                using (Stream contentStream = await Response.Content.ReadAsStreamAsync())
                {
                    long TotalBytes = Response.Content.Headers.ContentLength.GetValueOrDefault(0L);
                    long TotalReadBytes = 0L;

                    byte[] Buffer = new byte[8192];
                    int BytesRead;

                    while ((BytesRead = await contentStream.ReadAsync(Buffer, 0, Buffer.Length)) != 0)
                    {
                        await memoryStream.WriteAsync(Buffer, 0, BytesRead);
                        TotalReadBytes += BytesRead;

                        double Progress = (double)TotalReadBytes / TotalBytes * 100;

                        VirtualConsole.ClearLastLine();
                        Console.Write($"Downloading map from EMDB: {Progress:F2}%");
                    }
                    Console.WriteLine("");
                }

                return memoryStream.ToArray();
            }
        }

        void ExtractGzArchive(byte[] data, string outputPath)
        {
            using (MemoryStream memoryStr = new MemoryStream(data))
            {
                using (GZipStream decompressStream = new GZipStream(memoryStr, CompressionMode.Decompress))
                {
                    string OutputFile = outputPath;
                    using (FileStream OutputStr = new FileStream(OutputFile, FileMode.Create))
                    {
                        decompressStream.CopyTo(OutputStr);
                    }
                }
            }
        }
    }
}
