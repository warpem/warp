using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warp;
using Warp.Sociology;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;
using Warp.Workers.Scheduling;

namespace MTools.Commands
{
    [Verb("reconstruct", HelpText = "Reconstruct an average from particle poses you already have, " +
                                   "using M's full reconstruction machinery — previously determined CTF, " +
                                   "astigmatism, higher-order aberrations, image and volume deformation, " +
                                   "dose weighting and magnification — without refining anything. " +
                                   "Needs no reference and no pre-existing population.")]
    [CommandRunner(typeof(Reconstruct))]
    class ReconstructOptions
    {
        [Option("source", HelpText = "Path(s) to one or more .source files, space-separated. " +
                                     "Frame-series and tilt-series sources can be mixed in one run. " +
                                     "Mutually exclusive with --population.")]
        public IEnumerable<string> Sources { get; set; }

        [Option('p', "population", HelpText = "Instead of --source, use every data source of this .population. " +
                                              "Also makes the population's folder the default output root.")]
        public string Population { get; set; }

        [Option("particles", Required = true, HelpText = "Path to a particle STAR file, either RELION's _data.star or M's " +
                                                        "_particles.star. Which one it is is worked out from the columns.")]
        public string Particles { get; set; }

        [Option("angpix_coords", HelpText = "Override the pixel size of the particle coordinates in a RELION file. " +
                                            "Also applies to the shifts when those are in pixels rather than Angstrom.")]
        public float? AngPixCoords { get; set; }

        [Option("ignore_unmatched", HelpText = "Don't fail if there are particles that don't match any data sources.")]
        public bool IgnoreUnmatched { get; set; }

        [Option("angpix", Required = true, HelpText = "Pixel size of the reconstruction in Angstrom.")]
        public float AngPix { get; set; }

        [Option('d', "diameter", Required = true, HelpText = "Molecule diameter in Angstrom. Also sets the box size to " +
                                                            "2 x round(diameter / angpix), as elsewhere in M.")]
        public int Diameter { get; set; }

        [Option('s', "sym", Default = "C1", HelpText = "Point symmetry, e.g. C1, D7, O.")]
        public string Symmetry { get; set; }

        [Option("helical_units", Default = 1, HelpText = "Number of helical asymmetric units (only relevant for helical symmetry).")]
        public int HelicalUnits { get; set; }

        [Option("helical_twist", HelpText = "Helical twist in degrees, positive = right-handed (only relevant for helical symmetry).")]
        public double HelicalTwist { get; set; }

        [Option("helical_rise", HelpText = "Helical rise in Angstrom (only relevant for helical symmetry).")]
        public double HelicalRise { get; set; }

        [Option("helical_height", HelpText = "Height of the helical segment along the Z axis in Angstrom (only relevant for helical symmetry).")]
        public int HelicalHeight { get; set; }

        [Option("max_resolution", HelpText = "Highest resolution in Angstrom the reconstruction should be able to reach. " +
                                             "Defaults to Nyquist (2 x angpix), which is the most accurate and the most " +
                                             "expensive: it is the only input to the CTF super-resolution box size, so " +
                                             "loosening it makes extraction cheaper and less memory-hungry.")]
        public float? MaxResolution { get; set; }

        [Option('m', "mask", HelpText = "Optional tight binary mask for the postprocess. If omitted, a sphere of the " +
                                        "molecule diameter is used, which is unbiased but gives a slightly pessimistic " +
                                        "FSC. Sharpening itself never uses a mask.")]
        public string Mask { get; set; }

        [Option("no_postprocess", HelpText = "Write only the raw half-maps, skipping FSC estimation, sharpening and local resolution.")]
        public bool NoPostprocess { get; set; }

        [Option("denoise", HelpText = "Train and apply a denoiser during the postprocess. Off by default; " +
                                      "it is the slowest part and is only useful for visualisation.")]
        public bool Denoise { get; set; }

        [Option("batchsize", Default = 16, HelpText = "Particles per extraction batch. The main lever if you run out of GPU " +
                                                     "memory, since the extraction box is enlarged by the CTF super-resolution factor.")]
        public int BatchSize { get; set; }

        [Option("min_particles", Default = 1, HelpText = "Only use series with at least N particles in the field of view.")]
        public int MinParticles { get; set; }

        [Option("cpu_memory", HelpText = "Use CPU memory to store particle images (GPU by default).")]
        public bool UseHostMemory { get; set; }

        [Option('o', "output", HelpText = "Root directory for reconstructions. Each run creates <root>/<name> inside it. " +
                                          "Defaults to <population folder>/reconstructions with --population, else ./reconstructions.")]
        public string Output { get; set; }

        [Option('n', "name", HelpText = "Name of this reconstruction, used for its output files and, with a unique suffix " +
                                        "appended, its directory. Defaults to the particle file's name without its extension.")]
        public string Name { get; set; }

        [Option("keep_temp", HelpText = "Keep the temporary directory (queue, logs, per-worker back-projection partials) " +
                                        "on success. It is always kept on failure.")]
        public bool KeepTemp { get; set; }

        [Option("device_list", HelpText = "Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system.")]
        public IEnumerable<int> DeviceList { get; set; }

        [Option("perdevice", Default = 1, HelpText = "Number of worker processes per GPU.")]
        public int ProcessesPerDevice { get; set; }

        [Option("task_dir", HelpText = "Directory for the filesystem work queue used by this run. Defaults to a 'tasks' " +
                                       "subdirectory inside the reconstruction's temp folder. Set this to fast local " +
                                       "scratch when the output directory is on a slow network filesystem.")]
        public string TaskDir { get; set; }

        [Option("external_provisioner", HelpText = "Don't spawn local worker processes. An external system (e.g. Relay) " +
                                                   "provisions workers that claim tasks from the queue directory.")]
        public bool UseExternalProvisioner { get; set; }

        [Option("cluster_script", HelpText = "Path to a batch-scheduler submission-script template. Presence of this " +
                                             "option selects cluster mode.")]
        public string ClusterScript { get; set; }

        [Option("cluster_config", HelpText = "Path to the cluster queue-definition JSON. Required with --cluster_script.")]
        public string ClusterConfig { get; set; }

        [Option("pool_size", HelpText = "Cluster mode: number of worker jobs to submit to the scheduler.")]
        public int PoolSize { get; set; }

        [Option("cluster_var", HelpText = "Cluster mode: a key=value pair substituted into the submission template (repeatable).")]
        public IEnumerable<string> ClusterVars { get; set; }
    }

    /// <summary>
    /// Reconstruction-only averaging.
    ///
    /// M's back-projection is already a self-contained "reconstruct using the parameters we
    /// have" engine — the refinement is a wrapper around it that
    /// PerformMultiParticleRefinement skips entirely when NIterations == 0. This command
    /// therefore builds a fresh, minimal population and species, runs that same
    /// back-projection over every item, and reduces the per-worker partials into half-maps.
    ///
    /// Compared with MCore, the per-species pre-flight phase (denoise and SSNR-filter the
    /// references into staging) disappears completely: there are no references. Item
    /// metadata is never rewritten either, because nothing was refined.
    /// </summary>
    class Reconstruct : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            ReconstructOptions Options = options as ReconstructOptions;

            #region Argument validation

            if (!File.Exists(Options.Particles))
            {
                Console.Error.WriteLine($"Particle file not found: {Options.Particles}");
                return;
            }

            bool HasSources = Options.Sources != null && Options.Sources.Any();
            if (HasSources == !string.IsNullOrEmpty(Options.Population))
            {
                Console.Error.WriteLine("Must provide either --source (one or more .source files) or --population, but not both.");
                return;
            }

            if (Options.AngPix <= 0)
            {
                Console.Error.WriteLine("--angpix must be positive.");
                return;
            }

            if (Options.Diameter <= 0)
            {
                Console.Error.WriteLine("--diameter must be positive.");
                return;
            }

            try
            {
                Symmetry S = new Symmetry(Options.Symmetry);
            }
            catch
            {
                Console.Error.WriteLine($"Unknown point symmetry: {Options.Symmetry}");
                return;
            }

            if (new[] { Options.HelicalUnits > 1,
                        Options.HelicalRise != 0,
                        Options.HelicalTwist != 0,
                        Options.HelicalHeight != 0 }.Any(v => v))
            {
                if (Options.HelicalUnits <= 1)
                {
                    Console.Error.WriteLine($"Helical symmetry requires at least 2 asymmetric units, specified {Options.HelicalUnits}.");
                    return;
                }

                if (Options.HelicalRise <= 0)
                {
                    Console.Error.WriteLine($"Helical rise must be positive, specified {Options.HelicalRise}.");
                    return;
                }

                if (Options.HelicalTwist == 0)
                {
                    Console.Error.WriteLine($"Helical twist must be non-zero, specified {Options.HelicalTwist}.");
                    return;
                }

                if (Options.HelicalHeight <= 0)
                {
                    Console.Error.WriteLine($"Helical height must be positive, specified {Options.HelicalHeight}.");
                    return;
                }

                double MinHeight = Options.HelicalRise * (Options.HelicalUnits - 1);
                if (Options.HelicalHeight < MinHeight)
                {
                    Console.Error.WriteLine($"Helical height must be at least {MinHeight:F2} A (but probably a bit more), specified {Options.HelicalHeight} A.");
                    return;
                }

                Console.WriteLine($"Will use helical symmetry: {Options.HelicalUnits} units, {Options.HelicalRise:F4} A rise, {Options.HelicalTwist:F4} deg twist, {Options.HelicalHeight} A height.");
            }

            if (Options.BatchSize < 1)
            {
                Console.Error.WriteLine("--batchsize can't be lower than 1.");
                return;
            }

            float MaxResolution = Options.MaxResolution ?? Options.AngPix * 2;
            if (MaxResolution < Options.AngPix * 2)
            {
                Console.Error.WriteLine($"--max_resolution can't be beyond Nyquist ({Options.AngPix * 2:F2} A at {Options.AngPix:F3} A/px).");
                return;
            }

            // Box size follows the molecule diameter, exactly as it does for a species
            // created with create_species. Printed because it is derived rather than given.
            int BoxSize = (int)MathF.Round(Options.Diameter / Options.AngPix) * 2;
            if (BoxSize < 64)
            {
                Console.Error.WriteLine($"The resulting box size would be {BoxSize} px, which is too small to reconstruct " +
                                        $"(the box edge is soft-masked over 32 px). Use a finer --angpix or a larger --diameter.");
                return;
            }
            Console.WriteLine($"Box size will be {BoxSize} px at {Options.AngPix:F3} A/px ({Options.Diameter} A diameter).");

            if (!string.IsNullOrEmpty(Options.Mask) && !File.Exists(Options.Mask))
            {
                Console.Error.WriteLine($"Mask not found: {Options.Mask}");
                return;
            }

            #endregion

            #region Resolve data sources

            Population SourcePopulation = null;
            List<DataSource> Sources = new List<DataSource>();

            if (!string.IsNullOrEmpty(Options.Population))
            {
                if (!File.Exists(Options.Population))
                {
                    Console.Error.WriteLine($"Population not found: {Options.Population}");
                    return;
                }

                SourcePopulation = new Population(Options.Population);
                Sources.AddRange(SourcePopulation.Sources);

                if (Sources.Count == 0)
                {
                    Console.Error.WriteLine($"{Options.Population} contains no data sources.");
                    return;
                }
            }
            else
            {
                foreach (string Path in Options.Sources)
                {
                    if (!File.Exists(Path))
                    {
                        Console.Error.WriteLine($"Data source not found: {Path}");
                        return;
                    }

                    Sources.Add(DataSource.FromFile(Path));
                }

                var DuplicateGUIDs = Sources.GroupBy(s => s.GUID).Where(g => g.Count() > 1).ToArray();
                if (DuplicateGUIDs.Any())
                {
                    Console.Error.WriteLine($"The same data source was specified more than once ({DuplicateGUIDs.First().First().Name}).");
                    return;
                }
            }

            Console.WriteLine($"Using {Sources.Count} data source(s): {string.Join(", ", Sources.Select(s => s.Name))}");

            #endregion

            #region Resolve output paths

            string Name = string.IsNullOrEmpty(Options.Name)
                ? Helper.PathToName(Options.Particles)
                : Options.Name;
            string NameSafe = Helper.RemoveInvalidChars(Name);
            if (string.IsNullOrWhiteSpace(NameSafe))
            {
                Console.Error.WriteLine($"'{Name}' does not yield a usable directory name; please specify --name.");
                return;
            }

            string Root = !string.IsNullOrEmpty(Options.Output)
                ? Options.Output
                : (SourcePopulation != null
                    ? Path.Combine(SourcePopulation.FolderPath, "reconstructions")
                    : Path.Combine(Environment.CurrentDirectory, "reconstructions"));

            // Every run gets its own directory, named the way species directories are:
            // <name>_<first 8 of a fresh GUID>. Two runs from the same particle file
            // therefore never collide, and nothing existing can be overwritten. The same
            // GUID is given to the species inside, so the directory identifies its contents.
            // Files within keep the bare name, again matching species directories
            // (species/class_5_d6f2824b/class_5.species).
            Guid ReconstructionGUID = Guid.NewGuid();
            string DirName = NameSafe + "_" + ReconstructionGUID.ToString().Substring(0, 8);
            string OutputDir = Path.Combine(Root, DirName);

            if (Directory.Exists(OutputDir))
            {
                Console.Error.WriteLine($"{OutputDir} already exists, which should be impossible for a fresh GUID. Try again.");
                return;
            }

            Directory.CreateDirectory(OutputDir);

            string TempDir = Path.Combine(OutputDir, "temp");
            string LogDir = Path.Combine(TempDir, "logs");
            string TaskDir = !string.IsNullOrEmpty(Options.TaskDir) ? Options.TaskDir : Path.Combine(TempDir, "tasks");
            Directory.CreateDirectory(TempDir);
            Directory.CreateDirectory(LogDir);

            #endregion

            bool Succeeded = false;
            // Distinguishes "failed after workers did real work" (keep the partials, they
            // are expensive) from "failed while still validating inputs" (nothing was
            // produced, so don't litter or claim there are intermediate results).
            bool WorkStarted = false;

            try
            {
                #region Build a fresh population holding just these sources

                // A real, self-contained population and species, created here rather than
                // borrowed from an existing one. Nothing belonging to the user's project is
                // mutated: the sources are referenced read-only and never committed, so no
                // versions/ trees appear and no item .xml is rewritten.
                Population NewPopulation = new Population(Path.Combine(OutputDir, NameSafe + ".population"))
                {
                    Name = Name
                };

                foreach (var Source in Sources)
                    NewPopulation.Sources.Add(Source);

                #endregion

                #region Particles

                Console.Write("Parsing particle table... ");

                // temporalSamples 0 = keep whatever the table has, so an M table carrying
                // per-tilt pose trajectories from a previous refinement is used as-is
                // rather than being collapsed to a single pose.
                ParticleImportResult Imported = ParticleStarParsing.FromAnyStar(Options.Particles,
                                                                                Sources,
                                                                                temporalSamples: 0,
                                                                                Options.AngPixCoords);
                if (Imported == null)
                    return;

                Console.WriteLine("Done");

                if (!Options.IgnoreUnmatched && Imported.Unmatched > 0)
                {
                    Console.Error.WriteLine($"{Imported.Unmatched} particles couldn't be matched to a data source. " +
                                            $"Please run again with --ignore_unmatched to proceed anyway.");
                    return;
                }

                // Nothing downstream can succeed without particles, and the usual cause is a
                // naming mismatch between the star file and the data sources. Fail here
                // rather than after provisioning workers and loading every tilt series.
                if (Imported.Particles.Length == 0)
                {
                    Console.Error.WriteLine("No particles could be matched to any of the data sources. " +
                                            "Check that the micrograph/tomogram names in the particle file correspond " +
                                            "to the items in the source(s) — for tilt series this is usually " +
                                            "rlnTomoName rather than rlnMicrographName.");
                    return;
                }

                int NHalf1 = Imported.Particles.Count(p => p.RandomSubset == 0);
                Console.WriteLine($"Matched {Imported.Particles.Length} particles across " +
                                  $"{Imported.Particles.Select(p => p.SourceHash).Distinct().Count()} series; " +
                                  $"half-sets {NHalf1}/{Imported.Particles.Length - NHalf1} from {Imported.SubsetSource}.");

                #endregion

                #region Build the species

                Species NewSpecies = new Species(null, null, null)
                {
                    // Shared with the output directory name, as create_species does.
                    GUID = ReconstructionGUID,
                    Name = Name,
                    PixelSize = (decimal)Options.AngPix,
                    DiameterAngstrom = Options.Diameter,
                    Symmetry = Options.Symmetry,
                    HelicalUnits = Options.HelicalUnits,
                    HelicalTwist = (decimal)Options.HelicalTwist,
                    HelicalRise = (decimal)Options.HelicalRise,
                    HelicalHeight = Options.HelicalHeight,
                    // Taken from the imported table so the species matches the pose
                    // trajectories the particles actually carry.
                    TemporalResolutionMovement = Imported.TemporalResolutionMovement,
                    TemporalResolutionRotation = Imported.TemporalResolutionRotation,

                    // The only consumer of this is the CTF super-resolution factor picked at
                    // back-projection time; there is no reference whose resolution it could
                    // describe. Nyquist by default = most accurate, most expensive.
                    GlobalResolution = (decimal)MaxResolution,

                    ApplyDenoising = Options.Denoise,
                    UseForAlignment = false,

                    // Nothing here is worth versioning: the species is an output of this run,
                    // not a refinement checkpoint.
                    DontVersion = true
                };
                NewSpecies.Path = Path.Combine(OutputDir, NameSafe + ".species");

                NewSpecies.AddParticles(Imported.Particles);

                // The postprocess needs a mask; sharpening itself does not (it fits a global
                // B-factor from the FSC curve). A sphere of the molecule diameter is the
                // unbiased default, and is the same construction M falls back to internally
                // when it has to estimate a resolution without one. It costs a slightly
                // pessimistic FSC compared with a tight mask, and no bias.
                if (!Options.NoPostprocess)
                {
                    if (!string.IsNullOrEmpty(Options.Mask))
                    {
                        Image Mask = Image.FromFile(Options.Mask);
                        if (Mask.Dims.X != BoxSize || !Mask.Dims.IsCubic)
                        {
                            Console.Write($"Adjusting mask from {Mask.Dims} to {BoxSize}^3... ");
                            Mask = Mask.AsPadded(new int3(BoxSize)).AndDisposeParent();
                            Mask.Binarize(0.25f);
                            Console.WriteLine("Done");
                        }
                        Mask.PixelSize = Options.AngPix;
                        NewSpecies.Mask = Mask;
                    }
                    else
                    {
                        Console.WriteLine($"No mask given; using a sphere of {Options.Diameter} A for FSC estimation.");
                        Image Mask = CreateSphereMask(BoxSize, Options.Diameter / Options.AngPix);
                        Mask.PixelSize = Options.AngPix;
                        NewSpecies.Mask = Mask;
                    }
                }

                NewSpecies.Save();
                NewSpecies.DropMaps();

                // Registering the species autosaves the population, which stores it by
                // relative path — so the worker resolves it next to the .population file.
                NewPopulation.Species.Add(NewSpecies);
                NewPopulation.Save();

                #endregion

                #region Refinement options with every refinement switched off

                // NIterations = 0 makes PerformMultiParticleRefinement skip its entire
                // optimizer — spectral weights, tilt-movie alignment, BFGS, defocus search,
                // FSC against references — and run only extraction plus back-projection.
                // The individual toggles are unreachable at zero iterations; they are set
                // anyway so the echoed options are honest, and so that raising NIterations
                // later cannot silently start refining something.
                ProcessingOptionsMPARefine RefineOptions = new ProcessingOptionsMPARefine
                {
                    NIterations = 0,
                    BatchSize = Options.BatchSize,
                    MinParticlesPerItem = Options.MinParticles,
                    UseHostMemory = Options.UseHostMemory,

                    DoImageWarp = false,
                    DoVolumeWarp = false,
                    DoAxisAngles = false,
                    DoParticlePoses = false,
                    DoMagnification = false,
                    DoDoming = false,
                    DoTiltMovies = false,

                    DoDefocus = false,
                    DoAstigmatismDelta = false,
                    DoAstigmatismAngle = false,
                    DoDefocusGridSearch = false,
                    DoPhaseShift = false,
                    DoCs = false,
                    DoZernike13 = false,
                    DoZernike2 = false,
                    DoZernike4 = false,
                    DoZernike5 = false
                };

                WorkerPoolSettings PoolSettings = new WorkerPoolSettings
                {
                    DeviceList = Options.DeviceList?.ToArray(),
                    ProcessesPerDevice = Options.ProcessesPerDevice,
                    UseExternalProvisioner = Options.UseExternalProvisioner,
                    ClusterScript = Options.ClusterScript,
                    ClusterConfig = Options.ClusterConfig,
                    PoolSize = Options.PoolSize,
                    ClusterVars = Options.ClusterVars
                };

                #endregion

                #region Back-project (per source, per item)

                Console.WriteLine("Extracting and back-projecting particles");

                WorkStarted = true;

                // Which series actually carry particles. Queueing an item with none of them
                // just pays for loading and preprocessing a tilt series to then do nothing.
                HashSet<string> HashesWithParticles = new HashSet<string>(Imported.Particles.Select(p => p.SourceHash));

                List<string> AllProgressFolders = new List<string>();

                foreach (var Source in Sources)
                {
                    string SourceTempDir = Path.Combine(TempDir, Helper.RemoveInvalidChars(Source.Name));
                    Directory.CreateDirectory(SourceTempDir);

                    // Amortized init, runs once per worker: identical across this source's
                    // tasks, so the init fingerprint matches and it is skipped afterwards.
                    // makeRefs: false because the species has no half-maps to build
                    // references from — only the reconstruction accumulators are allocated.
                    var InitHeaderless = WorkerCommands.SetHeaderlessParams(new int2(0), 0, "float");
                    var InitGain = WorkerCommands.LoadGainRef(Source.GainPath,
                                                              Source.GainFlipX,
                                                              Source.GainFlipY,
                                                              Source.GainTranspose,
                                                              Source.DefectsPath);
                    var InitPopulation = WorkerCommands.MPAPreparePopulation(NewPopulation.Path, null, makeRefs: false);

                    var Tasks = new List<TaskItem>();
                    var Items = Source.Files.Where(f => HashesWithParticles.Contains(f.Key)).ToArray();

                    for (int i = 0; i < Items.Length; i++)
                    {
                        string ItemPath = Path.Combine(Source.FolderPath, Items[i].Value);
                        var Task = new TaskItem
                        {
                            TaskId = $"{i:D6}-reconstruct-{Helper.PathToName(Items[i].Value)}",
                            Stage = "preprocess",
                            RequiresGpu = true,
                            Init = new[] { InitHeaderless, InitGain, InitPopulation },
                            Main = new[]
                            {
                                // saveItemMeta: false — nothing was refined, so the user's
                                // item .xml must not be rewritten.
                                WorkerCommands.MPARefineAndSave(ItemPath, RefineOptions, Source, SourceTempDir,
                                                                saveItemMeta: false),
                                WorkerCommands.GcCollect(),
                            },
                        };
                        Task.ComputeInitFingerprint();
                        Tasks.Add(Task);
                    }

                    int Skipped = Source.Files.Count - Items.Length;
                    Console.WriteLine($"Back-projecting {Tasks.Count} series in data source {Source.Name}" +
                                      (Skipped > 0 ? $" ({Skipped} without particles skipped)" : "") + "...");

                    TaskRunner.Run(TaskDir, LogDir, Tasks, PoolSettings);

                    if (Directory.Exists(SourceTempDir))
                        AllProgressFolders.AddRange(Directory.GetDirectories(SourceTempDir, "worker_*"));
                }

                if (AllProgressFolders.Count == 0)
                    throw new Exception("No worker produced any back-projection partials; nothing to reconstruct. " +
                                        $"Check the logs in {LogDir}.");

                #endregion

                #region Reduce: gather partials, reconstruct, postprocess

                Console.WriteLine("Gathering partials and reconstructing...");

                {
                    var Task = new TaskItem
                    {
                        TaskId = $"0000-reconstruct-{NameSafe}",
                        Stage = "preprocess",
                        RequiresGpu = true,
                        Main = new[]
                        {
                            WorkerCommands.MPAReconstructAverage(NewSpecies.Path,
                                                                 AllProgressFolders.ToArray(),
                                                                 !Options.NoPostprocess),
                            WorkerCommands.GcCollect(),
                        },
                    };
                    Task.ComputeInitFingerprint();

                    TaskRunner.Run(TaskDir, LogDir, new[] { Task }, PoolSettings);
                }

                #endregion

                Succeeded = true;

                Console.WriteLine();

                if (!Options.NoPostprocess)
                {
                    // Reloaded because the postprocess ran in a worker process; these are the
                    // values it saved.
                    Species Finished = Species.FromFile(NewSpecies.Path);

                    Console.WriteLine($"Resolution {Finished.GlobalResolution:F2} A, global B-factor {Finished.GlobalBFactor}.");
                    if (string.IsNullOrEmpty(Options.Mask))
                        Console.WriteLine("The FSC used a spherical mask, so the resolution is a conservative estimate; " +
                                          "pass --mask with a tight mask for a tighter number.");
                    Console.WriteLine();
                }

                Console.WriteLine($"Reconstruction '{Name}' written to {OutputDir}");
            }
            finally
            {
                // The partials are the expensive artifact, so keep them when a run that had
                // already started fails. On success they are redundant with the
                // reconstruction, and when we never got as far as running anything there is
                // nothing worth keeping — clean up rather than leaving an empty directory
                // and a misleading message.
                if (Succeeded && !Options.KeepTemp)
                {
                    try { Directory.Delete(TempDir, true); } catch { }
                }
                else if (!Succeeded && WorkStarted)
                {
                    Console.Error.WriteLine($"Intermediate results kept in {TempDir}; worker logs are in {LogDir}.");
                }
                else if (!Succeeded)
                {
                    try { Directory.Delete(OutputDir, true); } catch { }
                }
            }
        }

        /// <summary>
        /// A hard binary sphere of <paramref name="diameterPixels"/> centred in a cubic box,
        /// matching what create_species expects from --mask ("a tight binary mask; M will
        /// automatically expand and smooth it"). Built on the CPU so the orchestrator needs
        /// no GPU, which matters when the workers are remote or on a cluster.
        /// </summary>
        internal static Image CreateSphereMask(int size, float diameterPixels)
        {
            float[] Data = new float[size * size * size];

            float Radius = diameterPixels / 2;
            float RadiusSq = Radius * Radius;
            int Center = size / 2;

            for (int z = 0, i = 0; z < size; z++)
            {
                float dz = z - Center;
                for (int y = 0; y < size; y++)
                {
                    float dy = y - Center;
                    for (int x = 0; x < size; x++, i++)
                    {
                        float dx = x - Center;
                        Data[i] = (dx * dx + dy * dy + dz * dz) <= RadiusSq ? 1f : 0f;
                    }
                }
            }

            return new Image(Data, new int3(size));
        }
    }
}
