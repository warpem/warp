using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;
using Warp.Workers.Scheduling;


namespace WarpTools.Commands
{
    [VerbGroup("Frame series")]
    [Verb("fs_ctf", HelpText = "Estimate CTF parameters in frame series")]
    [CommandRunner(typeof(CTFFrameseries))]
    class CTFFrameseriesOptions : DistributedOptions
    {
        [Option("window", Default = 512, HelpText = "Patch size for CTF estimation in binned pixels")]
        public int Window { get; set; }

        [Option("range_min", Default = 30, HelpText = "Minimum resolution in Angstrom to consider in fit")]
        public double RangeMin { get; set; }

        [Option("range_max", Default = 4, HelpText = "Maximum resolution in Angstrom to consider in fit")]
        public double RangeMax { get; set; }

        [Option("defocus_min", Default = 0.5, HelpText = "Minimum defocus value in um to explore during fitting")]
        public double ZMin { get; set; }

        [Option("defocus_max", Default = 5.0, HelpText = "Maximum defocus value in um to explore during fitting")]
        public double ZMax { get; set; }


        [Option("voltage", Default = 300, HelpText = "Acceleration voltage of the microscope in kV")]
        public int Voltage { get; set; }

        [Option("cs", Default = 2.7, HelpText = "Spherical aberration of the microscope in mm")]
        public double? Cs { get; set; }

        [Option("amplitude", Default = 0.07, HelpText = "Amplitude contrast of the sample, usually 0.07-0.10 for cryo")]
        public double? Amplitude { get; set; }

        [Option("fit_phase", HelpText = "Fit the phase shift of a phase plate")]
        public bool PhaseEnable { get; set; }


        [Option("use_sum", HelpText = "Use the movie average spectrum instead of the average of individual frames' spectra. " +
                                      "Can help in the absence of an energy filter, or when signal is low.")]
        public bool MovieSumEnable { get; set; }


        [Option("grid", HelpText = "Resolution of the defocus model grid in X, Y, and temporal dimensions, separated by 'x': e.g. 5x5x40; empty = auto; Z > 1 is purely experimental")]
        public string GridDims { get; set; }
    }

    class CTFFrameseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            CTFFrameseriesOptions CLI = options as CTFFrameseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

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
            Options.CTF.UseMovieSum = CLI.MovieSumEnable;

            if (!string.IsNullOrEmpty(CLI.GridDims))
            {
                try
                {
                    var Dims = CLI.GridDims.Split('x');

                    Options.Grids.CTFX = int.Parse(Dims[0]);
                    Options.Grids.CTFY = int.Parse(Dims[1]);
                    Options.Grids.CTFZ = int.Parse(Dims[2]);
                }
                catch
                {
                    throw new Exception("Grid dimensions must be specified as XxYxZ, e.g. 5x5x40, or left empty for auto");
                }
            }
            else
            {
                Options.Grids.CTFX = 0;
                Options.Grids.CTFY = 0;
                Options.Grids.CTFZ = 0;
            }

            if (Options.Grids.CTFZ > 1 && Options.CTF.UseMovieSum)
                throw new Exception("Grid can't be larger than 1 in Z dimension when using movie sums because they have only 1 frame");

            #endregion

            // Remote workers (the old REST hostname:port distribution) are not part of
            // the filesystem work-distribution path. Cluster execution is handled by
            // Relay provisioning workers against the shared queue, not by this CLI.
            if (CLI.Workers != null && CLI.Workers.Any())
                throw new Exception("The --workers (remote hostname:port) option is not supported by the filesystem-based " +
                                    "fs_ctf path. Use local GPU distribution (--device_list / --perdevice), or run under " +
                                    "Relay for cluster execution.");

            ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
            decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

            #region Build one task per movie

            // The init sequence loads the gain reference + defect map. It is IDENTICAL
            // for every movie, so all tasks share an init fingerprint and a worker runs
            // it exactly once, then reuses the loaded gain across all its tasks (the same
            // amortization the old GetWorkers() gave by loading the gain into each worker
            // up front). The per-movie work (LoadStack + MovieProcessCTF) is the main
            // sequence. Empty gain/defect paths make LoadGainRef a cheap no-op.
            var loadGainRef = new NamedSerializableObject(
                nameof(WorkerWrapper.LoadGainRef),
                Options.Import.GainPath ?? "",
                Options.Import.GainFlipX,
                Options.Import.GainFlipY,
                Options.Import.GainTranspose,
                Options.Import.DefectsPath ?? "");

            string logDirectory = Path.Combine(CLI.OutputProcessing, "logs");
            Directory.CreateDirectory(logDirectory);

            foreach (var item in CLI.InputSeries)
                item.ProcessingStatus = ProcessingStatus.Unprocessed;

            var layout = new QueueLayout(Path.Combine(CLI.OutputProcessing, "work_ctf"));
            layout.EnsureDirectories();
            var queue = new TaskQueue(layout);
            var pool = new WorkPool(layout, queue);

            var tasks = new List<TaskItem>();
            var taskIdToMovie = new Dictionary<string, Movie>();

            for (int i = 0; i < CLI.InputSeries.Length; i++)
            {
                Movie m = (Movie)CLI.InputSeries[i];

                // Path correction (identical to the old IterateOverItems): if the output
                // directory differs from where the raw data lives, relocate the item's
                // meta path into OutputProcessing while remembering the data directory.
                // Must happen BEFORE building the task so the worker writes meta to the
                // right place (MovieProcessCTF uses m.Path) and reads raw data from
                // m.DataPath.
                if (Path.GetFullPath(CLI.OutputProcessing) !=
                    Path.GetFullPath(Path.GetDirectoryName(m.DataPath)))
                {
                    if (string.IsNullOrEmpty(m.DataDirectoryName))
                        m.DataDirectoryName = Path.GetDirectoryName(m.Path);

                    m.Path = Path.Combine(CLI.OutputProcessing, Path.GetFileName(m.Path));
                    m.SaveMeta();
                }

                bool useSum = Options.CTF.UseMovieSum && File.Exists(m.AveragePath);
                var loadStack = useSum
                    ? new NamedSerializableObject(nameof(WorkerWrapper.LoadStack),
                        m.AveragePath, 1M, Options.Import.EERGroupFrames, true)
                    : new NamedSerializableObject(nameof(WorkerWrapper.LoadStack),
                        m.DataPath, ScaleFactor, Options.Import.EERGroupFrames, true);

                var task = new TaskItem
                {
                    TaskId = $"{i:D7}-ctf-{m.RootName}",
                    Stage = "preprocess",
                    RequiresGpu = true,
                    Init = new[] { loadGainRef },
                    Main = new[]
                    {
                        loadStack,
                        new NamedSerializableObject(nameof(WorkerWrapper.MovieProcessCTF), m.Path, OptionsCTF),
                        new NamedSerializableObject(nameof(WorkerWrapper.GcCollect)),
                    },
                };
                task.ComputeInitFingerprint();
                tasks.Add(task);
                taskIdToMovie[task.TaskId] = m;
            }

            #endregion

            #region Distribute via the local worker pool

            List<int> devices = (CLI.DeviceList == null || !CLI.DeviceList.Any())
                ? Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList()
                : CLI.DeviceList.ToList();
            if (devices.Count <= 0)
                throw new Exception("No devices found or specified");

            // No point spawning more workers than there are movies.
            int target = Math.Min(CLI.InputSeries.Length, devices.Count * CLI.ProcessesPerDevice);

            var provisioner = new LocalProvisioner(layout.Root, devices.ToArray(), CLI.ProcessesPerDevice);
            var scheduler = new Scheduler(layout, queue, provisioner, target);

            Console.WriteLine($"Distributing {tasks.Count} CTF tasks across up to {target} local worker(s)...");

            var schedThread = new Thread(() => scheduler.RunToDrain()) { IsBackground = true };
            schedThread.Start();

            Dictionary<string, WorkResult> results;
            try
            {
                results = pool.Distribute(tasks);
            }
            finally
            {
                // Distribute returns once every task is terminal; the scheduler also
                // tears workers down on drain, but make sure nothing lingers.
                provisioner.Shutdown();
            }

            #endregion

            #region Collect results, update meta, write manifests

            List<Movie> processedItems = new();
            List<Movie> failedItems = new();

            foreach (var kv in results)
            {
                Movie m = taskIdToMovie[kv.Key];
                if (kv.Value.Outcome == WorkOutcome.Done)
                {
                    // Worker already wrote the CTF results to meta; refresh our copy and
                    // mark processed.
                    m.LoadMeta();
                    m.ProcessingStatus = ProcessingStatus.Processed;
                    m.SaveMeta();
                    processedItems.Add(m);
                }
                else
                {
                    // Poisoned (exhausted retries / failed on too many hosts). Mirror the
                    // old failure handling: deselect and mark to leave out.
                    m.LoadMeta();
                    m.UnselectManual = true;
                    m.ProcessingStatus = ProcessingStatus.LeaveOut;
                    m.SaveMeta();
                    failedItems.Add(m);
                    Console.Error.WriteLine($"Failed to process {m.Path}, marked as unselected. " +
                                            $"Use the change_selection WarpTool to reactivate it if required." +
                                            (string.IsNullOrEmpty(kv.Value.Error) ? "" : "\n" + kv.Value.Error));
                }
            }

            WriteMiniJson(Path.Combine(CLI.OutputProcessing, "processed_items.json"), processedItems);
            if (failedItems.Any())
                WriteMiniJson(Path.Combine(CLI.OutputProcessing, "failed_items.json"), failedItems);

            Console.WriteLine($"Finished: {processedItems.Count} processed, {failedItems.Count} failed");

            if (failedItems.Count == CLI.InputSeries.Length && CLI.InputSeries.Length > 0)
                throw new Exception("All items failed to process. Check logs for more info.");

            #endregion

            Console.Write("Saving settings...");
            Options.Save(Path.Combine(CLI.OutputProcessing, "ctf_movies.settings"));
            Console.WriteLine(" Done");
        }
    }
}
