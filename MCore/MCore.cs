using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using System.Diagnostics;
using System.IO;
using System.Globalization;
using Warp.Headers;
using System.Threading;
using Microsoft.AspNetCore.Hosting;
using CommandLine;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.OpenApi.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Components;
using System.Reflection;
using Warp.Sociology;
using Microsoft.Extensions.Options;
using Warp.Workers;
using Warp.Workers.Queue;
using Warp.Workers.Scheduling;


namespace MCore
{
    class MCore
    {
        public static OptionsCLI OptionsCLI = new OptionsCLI();
        public static ProcessingOptionsMPARefine Options = new ProcessingOptionsMPARefine();
        public static string WorkingDirectory;

        public static IHost RESTHost;

        public static bool AppRunning = true;
        public static string ProcessingMessage = "";

        public static Task ProcessingTask;

        static async Task Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            WorkingDirectory = Environment.CurrentDirectory;

            string ProgramFolder = AppContext.BaseDirectory;

            var Result = Parser.Default.ParseArguments<OptionsCLI>(args).WithParsed<OptionsCLI>(opts => OptionsCLI = opts);

            if (Result.Tag == ParserResultType.NotParsed ||
                Result.Errors.Any(e => e.Tag == ErrorType.HelpVerbRequestedError ||
                                       e.Tag == ErrorType.HelpRequestedError))
                return;

            WorkingDirectory = Environment.CurrentDirectory;

            VirtualConsole.AttachToConsole();

            #region Figure out settings

            if (OptionsCLI.ProcessesPerDevice < 1)
                throw new Exception("Number of processes per device should be at least 1");

            SetOptionsFromCLI(OptionsCLI);

            #endregion

            #region Start listening to Web API calls if desired

            if (OptionsCLI.Port > 0)
            {
                try
                {
                    RESTHost = Host.CreateDefaultBuilder().ConfigureWebHostDefaults(webBuilder =>
                    {
                        webBuilder.UseContentRoot(Directory.GetCurrentDirectory())
                                  .UseKestrel(options => options.ListenAnyIP(OptionsCLI.Port))
                                  .UseStartup<RESTStartup>()
                                  .ConfigureLogging(logging => logging.SetMinimumLevel(LogLevel.Warning));
                    }).Build();
                    await RESTHost.StartAsync();
                }
                catch (Exception exc)
                {
                    throw new Exception("There was a problem starting the REST API:\n" + exc.Message);
                }
            }

            #endregion

            await DoProcessing();

            RESTHost?.StopAsync().Wait();
        }

        static void SetOptionsFromCLI(OptionsCLI cli)
        {
            Options.NIterations = cli.NIterations;
            Options.InitialResolutionPercent = (int)(cli.FirstIterationFraction * 100);
            Options.MinParticlesPerItem = cli.NParticles;

            Options.UseHostMemory = cli.UseHostMemory;
            Options.BFactorWeightingThreshold = (decimal)cli.WeightThreshold;

            #region Geometry

            if (!string.IsNullOrEmpty(cli.RefineImageWarp))
            {
                try
                {
                    string[] Parts = cli.RefineImageWarp.Split(new[] { 'x' }, StringSplitOptions.RemoveEmptyEntries);
                    Options.ImageWarpWidth = int.Parse(Parts[0]);
                    Options.ImageWarpHeight = int.Parse(Parts[1]);
                    Options.DoImageWarp = true;
                }
                catch
                {
                    Console.WriteLine("Couldn't parse --refine_imagewarp, ignoring");
                }
            }
            else
            {
                Options.DoImageWarp = false;
            }

            if (!string.IsNullOrEmpty(cli.RefineVolumeWarp))
            {
                try
                {
                    string[] Parts = cli.RefineVolumeWarp.Split(new[] { 'x' }, StringSplitOptions.RemoveEmptyEntries);
                    Options.VolumeWarpWidth = int.Parse(Parts[0]);
                    Options.VolumeWarpHeight = int.Parse(Parts[1]);
                    Options.VolumeWarpDepth = int.Parse(Parts[2]);
                    Options.VolumeWarpLength = int.Parse(Parts[3]);
                    Options.DoVolumeWarp = true;
                }
                catch
                {
                    Console.WriteLine("Couldn't parse --refine_volumewarp, ignoring");
                }
            }
            else
            {
                Options.DoVolumeWarp = false;
            }

            Options.DoParticlePoses = cli.RefinePoses;
            Options.DoMagnification = cli.RefineMag;
            Options.DoDoming = cli.RefineDoming;
            Options.DoAxisAngles = cli.RefineStageAngles;
            Options.DoTiltMovies = cli.RefineTiltMovies;

            #endregion

            #region CTF

            Options.BatchSize = cli.CTFBatch;
            Options.MinimumCTFRefinementResolution = (decimal)cli.CTFMinResolution;
            Options.DoDefocus = cli.CTFDefocus;
            Options.DoAstigmatismDelta = Options.DoDefocus;
            Options.DoAstigmatismAngle = Options.DoDefocus;
            Options.DoDefocusGridSearch = cli.CTFDefocusExhaustive;
            Options.DoPhaseShift = cli.CTFPhase;
            Options.DoCs = cli.CTFCs;
            Options.DoZernike13 = cli.CTFZernike3;
            Options.DoZernike2 = cli.CTFZernike2;
            Options.DoZernike4 = cli.CTFZernike4;
            Options.DoZernike5 = cli.CTFZernike5;

            #endregion
        }

        #region Work distribution

        /// <summary>
        /// Run an explicit list of tasks through the filesystem work queue (scheduler +
        /// ephemeral worker pool) and block until all reach a terminal state. Mirrors
        /// WarpTools' DistributedOptions.DistributeTasks but is parameterized by MCore's
        /// OptionsCLI. Each call sets up a fresh queue, spawns workers, and tears them
        /// down on completion — so refinement's three phases (and each data source) run on
        /// their own short-lived pool, matching the ts_reconstruct_average pattern and
        /// keeping only one pool's worth of CUDA-initialized processes alive at a time.
        /// <paramref name="onItemDone"/> runs single-threaded on the polling thread after
        /// each task finishes (the seam for per-item orchestrator output).
        /// </summary>
        static void RunTasks(string queueDir, string logDir, IReadOnlyList<TaskItem> tasks,
                             Action<TaskItem, bool> onItemDone = null, int pollMs = 500)
        {
            if (tasks == null || tasks.Count == 0)
                return;

            var layout = new QueueLayout(queueDir);
            layout.EnsureDirectories();
            var queue = new TaskQueue(layout);
            queue.Clear();
            var pool = new WorkPool(layout, queue);

            Directory.CreateDirectory(logDir);

            IWorkerProvisioner provisioner;
            int target;
            if (OptionsCLI.UseExternalProvisioner)
            {
                provisioner = new ExternalProvisioner();
                target = 0;
                Console.WriteLine($"Distributing {tasks.Count} task(s); workers provisioned externally...");
            }
            else
            {
                List<int> devices = (OptionsCLI.DeviceList == null || !OptionsCLI.DeviceList.Any())
                    ? Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList()
                    : OptionsCLI.DeviceList.ToList();
                if (devices.Count <= 0)
                    throw new Exception("No devices found or specified");
                target = Math.Min(tasks.Count, devices.Count * OptionsCLI.ProcessesPerDevice);
                provisioner = new LocalProvisioner(layout.Root, devices.ToArray(), OptionsCLI.ProcessesPerDevice, logDir: logDir);
                Console.WriteLine($"Distributing {tasks.Count} task(s) across up to {target} local worker(s)...");
            }

            var scheduler = new Scheduler(layout, queue, provisioner, target);

            var taskList = tasks.ToList();
            var taskById = taskList.ToDictionary(t => t.TaskId, t => t);
            pool.Enqueue(taskList);

            int total = taskList.Count;
            int nDone = 0, nFailed = 0;
            var progressSync = new object();
            Console.Write($"0/{total}");

            var schedCts = new CancellationTokenSource();
            var schedThread = new Thread(() => scheduler.RunToDrain(cancel: schedCts.Token)) { IsBackground = true };
            schedThread.Start();

            try
            {
                pool.Distribute(taskList,
                    onResult: result =>
                    {
                        bool succeeded = result.Outcome == WorkOutcome.Done;
                        taskById.TryGetValue(result.TaskId, out var task);

                        lock (progressSync)
                        {
                            nDone++;
                            if (!succeeded)
                            {
                                nFailed++;
                                VirtualConsole.ClearLastLine();
                                Console.Error.WriteLine($"Task {result.TaskId} failed.");
                                Console.Error.WriteLine($"Check logs in {logDir} for more info.");
                                if (!string.IsNullOrEmpty(result.Error))
                                    Console.Error.WriteLine("Exception details:\n" + result.Error);
                            }

                            try { onItemDone?.Invoke(task, succeeded); } catch { }

                            VirtualConsole.ClearLastLine();
                            string failedString = nFailed > 0 ? $", {nFailed} failed" : "";
                            Console.Write($"{nDone}/{total}{failedString}");
                        }
                    },
                    pollMs: pollMs);
            }
            finally
            {
                schedCts.Cancel();
                schedThread.Join();
                provisioner.Shutdown();
            }

            Console.WriteLine();

            if (nFailed == total && total > 0)
                throw new Exception("All tasks failed to process. Check logs for more info.");
        }

        #endregion

        static async Task DoProcessing()
        {
            Console.Write("Loading population... ");
            if (!File.Exists(Path.Combine(WorkingDirectory, OptionsCLI.Population)))
                throw new Exception("Population file not found");
            var ActivePopulation = new Population(Path.Combine(WorkingDirectory, OptionsCLI.Population));
            Console.WriteLine("Done");

            Console.Write("Creating directories... ");
            string StagingDirectory = Path.Combine(ActivePopulation.FolderPath, "refinement_temp", "staging");
            if (Directory.Exists(StagingDirectory))
                Directory.Delete(StagingDirectory, true);
            Directory.CreateDirectory(StagingDirectory);

            string LogDirectory = Path.Combine(ActivePopulation.FolderPath, "refinement_temp", "logs");
            if (Directory.Exists(LogDirectory))
                Directory.Delete(LogDirectory, true);
            Directory.CreateDirectory(LogDirectory);

            // Filesystem work queue. Lives under refinement_temp by default; --task_dir
            // can point it at fast local scratch when the population is on a slow share.
            string TaskDirectory = !string.IsNullOrEmpty(OptionsCLI.TaskDir)
                ? OptionsCLI.TaskDir
                : Path.Combine(ActivePopulation.FolderPath, "refinement_temp", "tasks");
            Console.WriteLine("Done");

            try
            {
                #region Pre-flight (per species): denoise/filter references into staging

                {
                    Console.WriteLine("Preparing refinement requisites (this takes a few minutes per species)...");

                    var tasks = new List<TaskItem>();
                    for (int i = 0; i < ActivePopulation.Species.Count; i++)
                    {
                        Species S = ActivePopulation.Species[i];
                        var task = new TaskItem
                        {
                            TaskId = $"{i:D4}-preprocess-{S.NameSafe}",
                            Stage = "preprocess",
                            RequiresGpu = true,
                            Main = new[]
                            {
                                WorkerCommands.MPAPrepareSpecies(S.Path, StagingDirectory),
                                WorkerCommands.GcCollect(),
                            },
                        };
                        task.ComputeInitFingerprint();
                        tasks.Add(task);
                    }

                    RunTasks(TaskDirectory, LogDirectory, tasks);
                }

                #endregion

                #region Refine (per source, per item): accumulate per-worker progress

                Console.WriteLine("Performing refinement");

                List<string> AllProcessingFolders = new List<string>();

                foreach (var source in ActivePopulation.Sources)
                {
                    // Per-source temp dir; each worker safe-saves its running progress into
                    // its own worker_<id> subfolder here after every item it refines.
                    string SourceTempDir = Path.Combine(ActivePopulation.FolderPath, "refinement_temp", source.Name);
                    if (Directory.Exists(SourceTempDir))
                        Directory.Delete(SourceTempDir, true);
                    Directory.CreateDirectory(SourceTempDir);

                    // Amortized init (runs once per worker): load the resident population
                    // and this source's gain/defects. Identical across the source's tasks,
                    // so the init fingerprint matches and it is skipped after the first.
                    var initHeaderless = WorkerCommands.SetHeaderlessParams(new int2(0), 0, "float");
                    var initGain = WorkerCommands.LoadGainRef(source.GainPath,
                                                              source.GainFlipX,
                                                              source.GainFlipY,
                                                              source.GainTranspose,
                                                              source.DefectsPath);
                    var initPopulation = WorkerCommands.MPAPreparePopulation(ActivePopulation.Path, StagingDirectory);

                    string[] AllPaths = source.Files.Values.ToArray();

                    var tasks = new List<TaskItem>();
                    for (int i = 0; i < AllPaths.Length; i++)
                    {
                        string ItemPath = Path.Combine(source.FolderPath, AllPaths[i]);
                        var task = new TaskItem
                        {
                            TaskId = $"{i:D6}-refine-{Helper.PathToName(AllPaths[i])}",
                            Stage = "preprocess",
                            RequiresGpu = true,
                            Init = new[] { initHeaderless, initGain, initPopulation },
                            Main = new[]
                            {
                                WorkerCommands.MPARefineAndSave(ItemPath, Options, source, SourceTempDir),
                                WorkerCommands.GcCollect(),
                            },
                        };
                        task.ComputeInitFingerprint();
                        tasks.Add(task);
                    }

                    Console.WriteLine($"Refining {tasks.Count} series in data source {source.Name}...");
                    RunTasks(TaskDirectory, LogDirectory, tasks);

                    Console.Write($"Committing changes in {source.Name}... ");
                    source.Commit();
                    Console.WriteLine("Done");

                    // Gather this source's per-worker progress folders for post-flight.
                    if (Directory.Exists(SourceTempDir))
                        AllProcessingFolders.AddRange(Directory.GetDirectories(SourceTempDir, "worker_*"));
                }

                #endregion

                #region Post-flight (per species): gather, reconstruct, filter, commit

                Console.WriteLine("Finishing refinement");
                Console.WriteLine("Gathering intermediate results, then reconstructing and filtering...");

                {
                    string[] ProgressFolders = AllProcessingFolders.ToArray();

                    var taskToSpecies = new Dictionary<string, Species>();
                    var tasks = new List<TaskItem>();
                    for (int i = 0; i < ActivePopulation.Species.Count; i++)
                    {
                        Species S = ActivePopulation.Species[i];
                        var task = new TaskItem
                        {
                            TaskId = $"{i:D4}-postprocess-{S.NameSafe}",
                            Stage = "preprocess",
                            RequiresGpu = true,
                            Main = new[]
                            {
                                WorkerCommands.MPAFinishSpecies(S.Path, StagingDirectory, ProgressFolders),
                                WorkerCommands.GcCollect(),
                            },
                        };
                        task.ComputeInitFingerprint();
                        tasks.Add(task);
                        taskToSpecies[task.TaskId] = S;
                    }

                    RunTasks(TaskDirectory, LogDirectory, tasks, onItemDone: (task, ok) =>
                    {
                        if (ok && task != null && taskToSpecies.TryGetValue(task.TaskId, out var S))
                        {
                            try
                            {
                                VirtualConsole.ClearLastLine();
                                Console.WriteLine($"{S.Name}: {Species.FromFile(S.Path).GlobalResolution:F2} Å");
                            }
                            catch { }
                        }
                    });
                }

                #endregion

                #region Clean up

                foreach (var folder in AllProcessingFolders)
                    if (Directory.Exists(folder))
                        try { Directory.Delete(folder, true); } catch { }

                ActivePopulation.Save();

                #endregion
            }
            catch (Exception exc)
            {
                throw new Exception("Something went wrong during refinement. Sorry! Here are the details:\n\n" +
                                    exc.ToString());
            }

            AppRunning = false;
        }
    }

    public class RESTStartup
    {
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddControllers().AddNewtonsoftJson();
            services.AddApiVersioning(opt => opt.ReportApiVersions = true);

            services.AddSwaggerGen(c => c.SwaggerDoc("v1", new OpenApiInfo
            {
                Title = "MCore",
                Description = "",
                Version = "v1"
            }));
        }

        public void Configure(IApplicationBuilder app)
        {
            app.UseSwagger();
            app.UseSwaggerUI(opt => opt.SwaggerEndpoint("/swagger/v1/swagger.json", "MCore v1"));

            app.UseRouting();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
            });
        }
    }
}
