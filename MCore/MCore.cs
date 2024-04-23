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

        public static readonly List<WorkerWrapper> WorkersPostprocess = new List<WorkerWrapper>();
        public static readonly List<WorkerWrapper> WorkersRefine = new List<WorkerWrapper>();
        public static readonly List<WorkerWrapper> WorkersPreprocess = new List<WorkerWrapper>();

        public static Task ProcessingTask;

        static void Main(string[] args)
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

            if (OptionsCLI.ProcessesPerDeviceRefine < 1)
                throw new Exception("Number of processes per device should be at least 1");

            if (OptionsCLI.ProcessesPerDevicePreprocess == null || (int)OptionsCLI.ProcessesPerDevicePreprocess <= 0)
                OptionsCLI.ProcessesPerDevicePreprocess = OptionsCLI.ProcessesPerDeviceRefine;
            if (OptionsCLI.ProcessesPerDevicePostprocess == null || (int)OptionsCLI.ProcessesPerDevicePostprocess <= 0)
                OptionsCLI.ProcessesPerDevicePostprocess = OptionsCLI.ProcessesPerDeviceRefine;

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
                    RESTHost.RunAsync();
                }
                catch (Exception exc)
                {
                    throw new Exception("There was a problem starting the REST API:\n" + exc.Message);
                }
            }

            #endregion

            ProcessingTask = new Task(DoProcessing);
            ProcessingTask.Start();

            while (AppRunning) Thread.Sleep(1);

            RESTHost.StopAsync().Wait();
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

        #region Worker management

        static void PrepareWorkers()
        {
            PopulateWorkerCollection(WorkersPreprocess, (int)OptionsCLI.ProcessesPerDevicePreprocess, OptionsCLI.WorkersPreprocess);
            PopulateWorkerCollection(WorkersRefine, OptionsCLI.ProcessesPerDeviceRefine, OptionsCLI.WorkersRefine);
            PopulateWorkerCollection(WorkersPostprocess, (int)OptionsCLI.ProcessesPerDevicePostprocess, OptionsCLI.WorkersPostprocess);
        }

        static void PopulateWorkerCollection(List<WorkerWrapper> collection, int perDevice, IEnumerable<string> remoteArgs, bool attachDebugger = false)
        {
            if (remoteArgs == null || remoteArgs.Count() == 0)
            {
                List<int> UsedDevices = (OptionsCLI.DeviceList == null || !OptionsCLI.DeviceList.Any()) ? 
                                        Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList() : 
                                        OptionsCLI.DeviceList.ToList();
                int NDevices = UsedDevices.Count;
                //List<int> UsedDeviceProcesses = Helper.Combine(Helper.ArrayOfFunction(i => UsedDevices.Select(d => d + i * NDevices).ToArray(), perDevice)).ToList();
                if (NDevices <= 0)
                    throw new Exception("No devices found or specified");

                foreach (var id in UsedDevices)
                {
                    for (int i = 0; i < perDevice; i++)
                    {
                        WorkerWrapper NewWorker = new WorkerWrapper(id, true, attachDebugger: attachDebugger);
                        NewWorker.WorkerDied += WorkerDied;

                        collection.Add(NewWorker);
                    }
                }
            }
            else
            {
                foreach (var workerspec in remoteArgs)
                    AttachWorker(workerspec, collection);
            }
        }

        static void AttachWorker(string spec, List<WorkerWrapper> collection)
        {
            string[] Parts = spec.Split(new[] { ':' }, StringSplitOptions.RemoveEmptyEntries);
            string Host = Parts[0];
            int Port = int.Parse(Parts[1]);

            if (collection.Any(w => w.Host == Host && w.Port == Port))
                throw new Exception($"Worker {spec} already attached");

            var ConnectedWorker = new WorkerWrapper(Host, Port);
            ConnectedWorker.WorkerDied += WorkerDied;

            collection.Add(ConnectedWorker);
        }

        static void DropWorkers()
        {
            if (OptionsCLI.WorkersPreprocess == null || OptionsCLI.WorkersPreprocess.Count() == 0)
            {
                foreach (var worker in WorkersPreprocess)
                {
                    worker.WorkerDied -= WorkerDied;
                    worker.Dispose();
                }
            }

            if (OptionsCLI.WorkersRefine == null || OptionsCLI.WorkersRefine.Count() == 0)
            {
                foreach (var worker in WorkersRefine)
                {
                    worker.WorkerDied -= WorkerDied;
                    worker.Dispose();
                }
            }

            if (OptionsCLI.WorkersPostprocess == null || OptionsCLI.WorkersPostprocess.Count() == 0)
            {
                foreach (var worker in WorkersPostprocess)
                {
                    worker.WorkerDied -= WorkerDied;
                    worker.Dispose();
                }
            }
        }

        #endregion

        static void DoProcessing()
        {
            Console.Write("Loading population... ");
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
            Console.WriteLine("Done");

            Console.Write("Spawning workers... ");
            PrepareWorkers();
            Console.WriteLine("Done");

            string[] WorkerFolders = new string[WorkersRefine.Count];

            Console.WriteLine("Preparing for refinement – this will take a few minutes per species");

            int ItemsCompleted = 0;
            int ItemsToDo = ActivePopulation.Sources.Select(s => s.Files.Count).Sum();

            string[] CurrentlyRefinedItems = new string[WorkersRefine.Count];

            System.Timers.Timer StatusUpdater = null;

            {
                StatusUpdater = new System.Timers.Timer(501);
                StatusUpdater.Elapsed += (s, e) =>
                {
                    lock (CurrentlyRefinedItems)
                    {
                        StringBuilder StatusMessage = new StringBuilder();

                        for (int iworker = 0; iworker < WorkersRefine.Count; iworker++)
                        { 
                            try
                            {
                                var Lines = WorkersRefine[iworker].Console.GetLastNLines(1);
                                if (Lines.Count > 0 && !string.IsNullOrWhiteSpace(Lines[0].Message))
                                    StatusMessage.Append(CurrentlyRefinedItems[iworker] + ": " + Lines[0].Message + "\n");
                            }
                            catch { }
                        }

                        string FinalMessage = StatusMessage.ToString();

                        if (!string.IsNullOrWhiteSpace(FinalMessage))
                            Console.Write(FinalMessage);
                    }
                };
            }

            try
            {
                #region Preprocessing
                {
                    Console.WriteLine($"Preparing refinement requisites...");
                    Console.Write($"0/{ActivePopulation.Species.Count}");

                    Queue<WorkerWrapper> Preparers = new Queue<WorkerWrapper>(WorkersPreprocess);
                    int NDone = 0;

                    Helper.ForCPU(0, ActivePopulation.Species.Count, WorkersPreprocess.Count, null, (ispecies, threadID) =>
                    {
                        WorkerWrapper Preparer = null;
                        while (true)
                            lock (Preparers)
                            {
                                if (Preparers.Count == 0)
                                {
                                    Thread.Sleep(10);
                                    continue;
                                }

                                Preparer = Preparers.Dequeue();
                                break;
                            }

                        Species S = ActivePopulation.Species[ispecies];

                        Preparer.Console.Clear();
                        Preparer.Console.SetFileOutput(Path.Combine(LogDirectory, $"preprocess_{S.NameSafe}.log"));

                        Preparer.MPAPrepareSpecies(S.Path, StagingDirectory);

                        Preparer.Console.SetFileOutput("");

                        lock (Preparers)
                        {
                            Preparers.Enqueue(Preparer);

                            VirtualConsole.ClearLastLine();
                            NDone++;
                            Console.Write($"{NDone}/{ActivePopulation.Species.Count}");
                        }
                    }, null);

                    foreach (var item in WorkersPreprocess)
                    {
                        item.WorkerDied -= WorkerDied;
                        item.Dispose();
                    }

                    Console.WriteLine("");
                }
                #endregion

                #region Refining

                Console.WriteLine("Performing refinement");

                List<string> AllProcessingFolders = new List<string>();

                int SourcesDone = 0;

                foreach (var source in ActivePopulation.Sources)
                {
                    for (int iworker = 0; iworker < WorkersRefine.Count; iworker++)
                    {
                        WorkerFolders[iworker] = Path.Combine(ActivePopulation.FolderPath, "refinement_temp", source.Name, $"worker{iworker}");
                        Directory.CreateDirectory(WorkerFolders[iworker]);

                        AllProcessingFolders.Add(WorkerFolders[iworker]);

                        WorkersRefine[iworker].SetHeaderlessParams(new int2(0), 0, "float");
                    }

                    #region Data source preflight

                    Console.Write($"Preparing population for data source {source.Name}...");

                    Helper.ForCPU(0, WorkersRefine.Count, WorkersRefine.Count, null, (iworker, threadID) =>
                    {
                        WorkersRefine[iworker].MPAPreparePopulation(ActivePopulation.Path, StagingDirectory);
                    }, null);
                    Console.WriteLine("Done");

                    Console.Write($"Loading gain reference for {source.Name}... ");

                    try
                    {
                        Helper.ForCPU(0, WorkersRefine.Count, WorkersRefine.Count, null, (iworker, threadID) =>
                        {
                            WorkersRefine[iworker].LoadGainRef(source.GainPath,
                                                                source.GainFlipX,
                                                                source.GainFlipY,
                                                                source.GainTranspose,
                                                                source.DefectsPath);
                        }, null);
                    }
                    catch
                    {
                        throw new Exception($"Could not load gain reference for {source.Name}.");
                    }
                    Console.WriteLine("Done");

                    #endregion

                    //StatusUpdater.Start();

                    Queue<WorkerWrapper> Refiners = new Queue<WorkerWrapper>(WorkersRefine);
                    string[] AllPaths = source.Files.Values.ToArray();

                    bool IsCanceled = false;

                    Console.WriteLine($"Refining all series in data source...");
                    Console.Write($"0/{source.Files.Count}");

                    int NDone = 0;
                    Helper.ForCPU(0, source.Files.Count, WorkersRefine.Count, null, (ifile, threadID) =>
                    {
                        if (IsCanceled)
                            return;

                        WorkerWrapper Refiner = null;
                        while (true)
                            lock (Refiners)
                            {
                                if (Refiners.Count == 0)
                                {
                                    Thread.Sleep(10);
                                    continue;
                                }

                                Refiner = Refiners.Dequeue();
                                break;
                            }

                        int iworker = WorkersRefine.IndexOf(Refiner);

                        {
                            lock (CurrentlyRefinedItems)
                                CurrentlyRefinedItems[iworker] = Helper.PathToName(AllPaths[ifile]);

                            Refiner.Console.Clear();
                            Refiner.Console.SetFileOutput(Path.Combine(LogDirectory, $"refine_{Helper.PathToName(AllPaths[ifile])}.log"));

                            Refiner.MPARefine(Path.Combine(source.FolderPath, AllPaths[ifile]),
                                              WorkerFolders[iworker],
                                              Options,
                                              source);

                            Refiner.Console.SetFileOutput("");

                            lock (CurrentlyRefinedItems)
                                CurrentlyRefinedItems[iworker] = null;
                        }

                        lock (Refiners)
                        {
                            Refiners.Enqueue(Refiner);

                            VirtualConsole.ClearLastLine();
                            NDone++;
                            Console.Write($"{NDone}/{source.Files.Count}");
                        }
                    }, null);

                    Console.WriteLine("");
                    //StatusUpdater.Stop();

                    Console.Write($"Commiting changes in {source.Name}...");

                    source.Commit();
                    SourcesDone++;

                    Console.WriteLine("Done");

                    Console.Write($"Saving intermediate refinement results for {source.Name}...");

                    Helper.ForCPU(0, WorkersRefine.Count, WorkersRefine.Count, null, (iworker, threadID) =>
                    {
                        WorkersRefine[iworker].MPASaveProgress(WorkerFolders[iworker]);
                    }, null);
                    Console.WriteLine("Done");
                }

                foreach (var refiner in WorkersRefine)
                {
                    refiner.WorkerDied -= WorkerDied;
                    refiner.Dispose();
                }

                #endregion

                #region Finish

                Console.WriteLine("Finishing refinement");

                {
                    Console.WriteLine("Gathering intermediate results, then reconstructing and filtering...");
                    Console.Write($"0/{ActivePopulation.Species.Count}");
                    int NDone = 0;

                    Queue<WorkerWrapper> Finishers = new Queue<WorkerWrapper>(WorkersPostprocess);

                    Helper.ForCPU(0, ActivePopulation.Species.Count, WorkersPreprocess.Count, null, (ispecies, threadID) =>
                    {
                        WorkerWrapper Finisher = null;
                        while (true)
                            lock (Finishers)
                            {
                                if (Finishers.Count == 0)
                                {
                                    Thread.Sleep(10);
                                    continue;
                                }

                                Finisher = Finishers.Dequeue();
                                break;
                            }

                        Species S = ActivePopulation.Species[ispecies];

                        Finisher.Console.Clear();
                        Finisher.Console.SetFileOutput(Path.Combine(LogDirectory, $"postprocess_{S.NameSafe}.log"));

                        Finisher.MPAFinishSpecies(S.Path, StagingDirectory, AllProcessingFolders.ToArray());

                        Finisher.Console.SetFileOutput("");

                        lock (Finishers)
                        {
                            Finishers.Enqueue(Finisher);

                            VirtualConsole.ClearLastLine();
                                                        
                            Console.WriteLine($"{S.Name}: {Species.FromFile(S.Path).GlobalResolution:F2} Å");

                            NDone++;
                            Console.Write($"{NDone}/{ActivePopulation.Species.Count}");
                        }
                    }, null);

                    foreach (var finisher in WorkersPostprocess)
                    {
                        finisher.WorkerDied -= WorkerDied;
                        finisher.Dispose();
                    }

                    foreach (var folder in AllProcessingFolders)
                        Directory.Delete(folder, true);

                    Console.WriteLine("");
                }

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

        private static void WorkerDied(object sender, EventArgs e)
        {
            throw new NotImplementedException();
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
