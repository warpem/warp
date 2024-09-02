using Microsoft.Extensions.Hosting;
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using Warp;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;
using CommandLine;
using System.IO.Pipes;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Hosting.Server;
using System.Runtime.InteropServices;

namespace WarpWorker
{
    static class WarpWorker
    {
        static bool DebugMode = false;
        static bool IsSilent = false;

        static int DeviceID = 0;
        static int Port = 0;

        static Thread Heartbeat;
        static Stopwatch PulseWatch;
        static bool Terminating = false;

        static Image GainRef = null;
        static DefectModel DefectMap = null;
        static int2 HeaderlessDims = new int2(2);
        static long HeaderlessOffset = 0;
        static string HeaderlessType = "float32";

        static float[][] RawLayers = null;

        static string OriginalStackOwner = "";
        static Image OriginalStack = null;

        static BoxNetTorch BoxNetModel = null;

        static Population MPAPopulation = null;

        static async Task Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            OptionsCLI OptionsCLI = null;
            Parser.Default.ParseArguments<OptionsCLI>(args).WithParsed(opts => OptionsCLI = opts);

            if (OptionsCLI.DebugAttach && !Debugger.IsAttached)
                Debugger.Launch();

            VirtualConsole.AttachToConsole();

            DeviceID = OptionsCLI.Device % GPU.GetDeviceCount();
            Port = OptionsCLI.Port;
            IsSilent = OptionsCLI.Silent;
            DebugMode = OptionsCLI.Debug;

            VirtualConsole.IsSilent = IsSilent;

            GPU.SetDevice(DeviceID);

            var Host = Microsoft.Extensions.Hosting.Host.CreateDefaultBuilder().ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseKestrel(options => options.ListenAnyIP(Port))
                          .UseStartup<RESTStartup>()
                          .ConfigureLogging(logging => logging.SetMinimumLevel(LogLevel.Warning));
            }).Build();
            Host.Start();

            // Retrieve the actual port used by Kestrel
            var Server = Host.Services.GetService(typeof(IServer)) as IServer;
            var ServerFeatures = Server.Features.Get<IServerAddressesFeature>();

            if (ServerFeatures == null || !ServerFeatures.Addresses.Any())
                throw new InvalidOperationException("Unable to determine the server's address.");

            // Ensure that we have a valid address to work with
            var Address = ServerFeatures.Addresses.First();
            if (string.IsNullOrEmpty(Address))
                throw new InvalidOperationException("The server address is not valid.");

            Port = new Uri(Address).Port;

            if (!string.IsNullOrEmpty(OptionsCLI.Pipe))
                await SendPort(OptionsCLI.Pipe, Port, 10_000);

            Console.WriteLine($"Running on GPU #{DeviceID} ({GPU.GetFreeMemory(DeviceID)} MB free), port {Port}\n");
            if (DebugMode)
                Console.WriteLine("Debug mode");

            #region Heartbeat

            PulseWatch = new Stopwatch();
            PulseWatch.Start();
            Heartbeat = new Thread(new ThreadStart(() =>
            {
                while (true)
                {
                    long WaitedMS = PulseWatch.ElapsedMilliseconds;

                    if (!DebugMode && WaitedMS > 10000)
                    {
                        Console.WriteLine($"{WaitedMS / 1000} seconds without heartbeat, exiting");

                        Process.GetCurrentProcess().Kill();
                    }
                    else if (Terminating)
                    {
                        Console.WriteLine("Exiting");

                        //Thread.Sleep(200);
                        Process.GetCurrentProcess().Kill();
                    }

                    Thread.Sleep(100);
                }
            }));
            Heartbeat.Start();

            #endregion

            while (true) Thread.Sleep(1);
        }

        static async Task SendPort(string pipeName, int port, int timeoutMilliseconds)
        {
            using (var client = new NamedPipeClientStream(pipeName))
            using (var writer = new StreamWriter(client))
            {
                // Task to connect to the server
                Task connectTask = client.ConnectAsync(timeoutMilliseconds);

                // Wait for the connection or timeout
                if (await Task.WhenAny(connectTask, Task.Delay(timeoutMilliseconds)) == connectTask)
                {
                    // Connection established within timeout
                    await connectTask;  // Ensure any exceptions are propagated
                    writer.WriteLine(port);
                    await writer.FlushAsync();
                }
                else
                {
                    Console.WriteLine("Failed to connect to the master process within the timeout period.");
                }
            }
        }

        #region Service

        public static void SendPulse()
        {
            PulseWatch?.Restart();
        }

        public static void Exit()
        {
            Process.GetCurrentProcess().Kill();
            //Terminating = true;
        }

        public static void EvaluateCommand(NamedSerializableObject Command)
        {
            GPU.SetDevice(DeviceID);
            Console.WriteLine($"Received \"{Command.Name}\", with {Command.Content.Length} arguments, for GPU #{GPU.GetDevice()}, {GPU.GetFreeMemory(DeviceID)} MB free:");
            if (DebugMode)
                foreach (var item in Command.Content)
                    Console.WriteLine($"{item.GetType().Name}: {item}");

            try
            {
                Stopwatch Watch = new Stopwatch();
                Watch.Start();


                if (Command.Name == nameof(WorkerWrapper.WaitAsyncTasks))
                {
                    Console.Write("Waiting for all async tasks to finish...");

                    GlobalTasks.WaitAll();

                    Console.WriteLine($" Done");
                }
                else if (Command.Name == "SetHeaderlessParams")
                {
                    HeaderlessDims = (int2)Command.Content[0];
                    HeaderlessOffset = (long)Command.Content[1];
                    HeaderlessType = (string)Command.Content[2];

                    Console.WriteLine($"Set headerless parameters to {HeaderlessDims}, {HeaderlessOffset}, {HeaderlessType}");
                }
                else if (Command.Name == "LoadGainRef")
                {
                    GainRef?.Dispose();
                    DefectMap?.Dispose();

                    string GainPath = (string)Command.Content[0];
                    bool FlipX = (bool)Command.Content[1];
                    bool FlipY = (bool)Command.Content[2];
                    bool Transpose = (bool)Command.Content[3];
                    string DefectsPath = (string)Command.Content[4];

                    if (!string.IsNullOrEmpty(GainPath))
                    {
                        GainRef = LoadAndPrepareGainReference(GainPath, FlipX, FlipY, Transpose);
                    }
                    if (!string.IsNullOrEmpty(DefectsPath))
                    {
                        DefectMap = LoadAndPrepareDefectMap(DefectsPath, FlipX, FlipY, Transpose);
                    }

                    Console.WriteLine($"Loaded gain reference and defect map: {GainRef}, {FlipX}, {FlipY}, {Transpose}, {DefectsPath}");
                }
                else if (Command.Name == "LoadStack")
                {
                    OriginalStack?.Dispose();

                    string Path = (string)Command.Content[0];
                    decimal ScaleFactor = (decimal)Command.Content[1];
                    int EERGroupFrames = (int)Command.Content[2];
                    bool CorrectGain = (bool)Command.Content[3];

                    HeaderEER.GroupNFrames = EERGroupFrames;

                    OriginalStack = LoadAndPrepareStack(Path, ScaleFactor, CorrectGain);
                    OriginalStackOwner = Helper.PathToNameWithExtension(Path);

                    Console.WriteLine($"Loaded stack: {OriginalStack}, {ScaleFactor}");
                }
                else if (Command.Name == "LoadBoxNet")
                {
                    BoxNetModel?.Dispose();

                    string Path = (string)Command.Content[0];
                    int BoxSize = (int)Command.Content[1];
                    int BatchSize = (int)Command.Content[2];

                    BoxNetModel = new BoxNetTorch(new int2(BoxSize), new float[3], new[] { DeviceID }, BatchSize);
                    BoxNetModel.Load(Path);

                    Console.WriteLine($"Model with box size = {BoxSize}, batch size = {BatchSize} loaded from {Path}");
                }
                else if (Command.Name == "DropBoxNet")
                {
                    BoxNetModel?.Dispose();

                    Console.WriteLine("Model dropped");
                }
                else if (Command.Name == "MovieProcessCTF")
                {
                    string Path = (string)Command.Content[0];
                    ProcessingOptionsMovieCTF Options = (ProcessingOptionsMovieCTF)Command.Content[1];
                    Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);

                    Movie M = new Movie(Path);
                    M.ProcessCTF(OriginalStack, Options);
                    M.SaveMeta();

                    Console.WriteLine($"Processed CTF for {Path}");
                }
                else if (Command.Name == "MovieProcessMovement")
                {
                    string Path = (string)Command.Content[0];
                    ProcessingOptionsMovieMovement Options = (ProcessingOptionsMovieMovement)Command.Content[1];
                    Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);

                    Movie M = new Movie(Path);
                    M.ProcessShift(OriginalStack, Options);
                    M.SaveMeta();

                    Console.WriteLine($"Processed movement for {Path}");
                }
                else if (Command.Name == "MoviePickBoxNet")
                {
                    if (BoxNetModel == null)
                        throw new Exception("No BoxNet model loaded");

                    string Path = (string)Command.Content[0];
                    ProcessingOptionsBoxNet Options = (ProcessingOptionsBoxNet)Command.Content[1];

                    Movie M = new Movie(Path);
                    M.MatchBoxNet2(new[] { BoxNetModel }, Options, null);
                    M.SaveMeta();

                    Console.WriteLine($"Picked particles for {Path}");
                }
                else if (Command.Name == "MovieExportMovie")
                {
                    string Path = (string)Command.Content[0];
                    ProcessingOptionsMovieExport Options = (ProcessingOptionsMovieExport)Command.Content[1];

                    Movie M = new Movie(Path);
                    M.ExportMovie(OriginalStack, Options);
                    M.SaveMeta();

                    Console.WriteLine($"Exported movie for {Path}");
                }
                else if (Command.Name == "MovieCreateThumbnail")
                {
                    string Path = (string)Command.Content[0];
                    int Size = (int)Command.Content[1];
                    float Range = (float)Command.Content[2];

                    Movie M = new Movie(Path);
                    M.CreateThumbnail(Size, Range);

                    Console.WriteLine($"Exported movie for {Path}");
                }
                else if (Command.Name == "MovieExportParticles")
                {
                    string Path = (string)Command.Content[0];
                    ProcessingOptionsParticleExport Options = (ProcessingOptionsParticleExport)Command.Content[1];
                    float2[] Coordinates = (float2[])Command.Content[2];

                    Movie M = new Movie(Path);
                    M.ExportParticles(OriginalStack, Coordinates, Options);
                    M.SaveMeta();

                    Console.WriteLine($"Exported {Coordinates.Length} particles for {Path}");
                }
                else if (Command.Name == "TomoStack")
                {
                    string Path = (string)Command.Content[0];
                    ProcessingOptionsTomoStack Options = (ProcessingOptionsTomoStack)Command.Content[1];

                    TiltSeries T = new TiltSeries(Path);
                    T.StackTilts(Options);

                    Console.WriteLine($"Created tilt stack for {Path}");
                }
                else if (Command.Name == "TomoAretomo")
                {
                    string SeriesPath = (string)Command.Content[0];
                    var Options = (ProcessingOptionsTomoAretomo)Command.Content[1];

                    TiltSeries T = new TiltSeries(SeriesPath);

                    string StackDir = T.TiltStackDir;
                    string StackPath = Path.GetFileName(T.TiltStackPath);
                    string AnglePath = Path.GetFileNameWithoutExtension(StackPath) + ".rawtlt";
                    string OutStack = T.RootName + "_aligned.mrc";
                    string Axis = Options.AxisAngle.ToString() + (Options.DoAxisSearch ? " 0" : " -1");
                    string AlignZ = Options.AlignZ.ToString();
                    string NPatchesX = Options.NPatchesXY[0].ToString();
                    string NPatchesY = Options.NPatchesXY[1].ToString();

                    string Arguments = $"-InMrc {StackPath} -AngFile {AnglePath} -VolZ 0 -OutBin 0 -TiltAxis {Axis} -AlignZ {AlignZ} -TiltCor 1 -OutImod 1 -DarkTol 0 -OutMrc {OutStack} -Gpu {DeviceID} -Patch {NPatchesX} {NPatchesY}";

                    Console.WriteLine($"Executing {Options.Executable} in {StackDir} with arguments: {Arguments}");

                    Process AreTomo = new Process
                    {
                        StartInfo =
                        {
                            FileName = Options.Executable,
                            CreateNoWindow = false,
                            WindowStyle = ProcessWindowStyle.Minimized,
                            WorkingDirectory = StackDir,
                            Arguments = Arguments,
                            RedirectStandardOutput = true,
                            RedirectStandardError = true
                        }
                    };
                    DataReceivedEventHandler Handler = (sender, args) => { if (args.Data != null) Console.WriteLine(args.Data); };
                    AreTomo.OutputDataReceived += Handler;
                    AreTomo.ErrorDataReceived += Handler;

                    AreTomo.Start();

                    AreTomo.BeginOutputReadLine();
                    AreTomo.BeginErrorReadLine();

                    AreTomo.WaitForExit();

                    Console.WriteLine($"Executed AreTomo for {SeriesPath}");
                }
                else if (Command.Name == "TomoEtomoPatchTrack")
                {
                    string TiltSeriesPath = (string)Command.Content[0];
                    var Options = (ProcessingOptionsTomoEtomoPatch)Command.Content[1];

                    TiltSeries T = new TiltSeries(TiltSeriesPath);
                    
                    // First generate a directive file to run Etomo automatically through batchruntomo
                    int PatchSize = (int)(Options.PatchSizeAngstroms / Options.TiltStackAngPix);
                    int RotOption = Options.DoAxisAngleSearch ? -1 : 0; // Fit single value (-1) or leave fixed (0)
                    var DirectiveFile = Path.Combine(Path.GetTempPath(), Path.GetTempFileName());
                    DirectiveFile = Path.ChangeExtension(DirectiveFile, ".adoc");

                    var BRTConfig = $"setupset.copyarg.userawtlt = 1\n" +
                                    $"setupset.copyarg.stackext = st\n" +
                                    $"setupset.copyarg.rotation = {Options.AxisAngle}\n" +
                                    $"setupset.copyarg.pixel = {Options.TiltStackAngPix / 10}\n" +
                                    $"setupset.copyarg.dual = 0\n" +
                                    $"runtime.Fiducials.any.trackingMethod = 1\n" +
                                    $"comparam.xcorr.tiltxcorr.FilterRadius2 = 0.5\n" +
                                    $"comparam.xcorr.tiltxcorr.ExcludeCentralPeak = 1\n" +
                                    $"comparam.xcorr_pt.tiltxcorr.IterateCorrelations = 4\n" +
                                    $"comparam.xcorr_pt.imodchopconts.LengthOfPieces = -1\n" +
                                    $"comparam.xcorr_pt.tiltxcorr.SizeOfPatchesXandY = {PatchSize},{PatchSize}\n" +
                                    $"comparam.xcorr_pt.tiltxcorr.OverlapOfPatchesXandY = 0.8,0.8\n" +
                                    $"comparam.align.tiltalign.MagOption = 0\n" +
                                    $"comparam.align.tiltalign.TiltOption = 0\n" +
                                    $"comparam.align.tiltalign.RotOption = {RotOption}\n" +
                                    $"comparam.align.tiltalign.RobustFitting = 1\n" +
                                    $"comparam.align.tiltalign.WeightWholeTracks = 1\n";
                    File.WriteAllText(path: DirectiveFile, contents: BRTConfig);
                    
                    // Then run batchruntomo
                    // Only do setup and alignment calculation if on second pass, otherwise do fiducial model generation too
                    int EndingStep = Options.DoPatchTracking ? 5 : 0;
                    string Arguments = $"-DirectiveFile {DirectiveFile} -CurrentLocation {T.TiltStackDir} " +
                                       $"-RootName {T.RootName} -EndingStep {EndingStep}";

                    if (Options.DoPatchTracking)
                        Console.WriteLine($"Performing patch tracking in {T.TiltStackDir} with arguments: {Arguments}");

                    // we execute batchruntomo even if not doing patch tracking to create com file for alignment

                    bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
                    string BatchRunTomoExecutable = IsWindows ? "batchruntomo.cmd" : "batchruntomo";
                    Console.WriteLine($"Running '{BatchRunTomoExecutable} {Arguments}'");
                    Process BatchRunTomo = new Process 
                    {
                        StartInfo =
                        {
                            FileName = BatchRunTomoExecutable,
                            CreateNoWindow = false,
                            WindowStyle = ProcessWindowStyle.Minimized,
                            Arguments = Arguments,
                            WorkingDirectory = T.TiltStackDir,
                            RedirectStandardOutput = true,
                            RedirectStandardError = true
                        }
                    };
                    DataReceivedEventHandler Handler = (sender, args) => { if (args.Data != null) Console.WriteLine(args.Data); };
                    BatchRunTomo.OutputDataReceived += Handler;
                    BatchRunTomo.ErrorDataReceived += Handler;

                    BatchRunTomo.Start();

                    BatchRunTomo.BeginOutputReadLine();
                    BatchRunTomo.BeginErrorReadLine();

                    BatchRunTomo.WaitForExit();
                    
                    // Run alignment separately from batchruntomo to avoid expensive cross-validation calculations
                    if (Options.DoTiltAlign)
                    {
                        Console.WriteLine($"Calculating projection parameters from patch tracking results in {T.TiltStackDir}");
                        string SubMfgExecutable = IsWindows ? "submfg.cmd" : "submfg";
                        Console.WriteLine($"Running '{SubMfgExecutable} align.com'");
                        Process TiltAlign = new Process
                        {
                            StartInfo =
                            {
                                FileName = SubMfgExecutable,
                                CreateNoWindow = false,
                                WindowStyle = ProcessWindowStyle.Minimized,
                                Arguments = "align.com",
                                WorkingDirectory = T.TiltStackDir,
                                RedirectStandardOutput = true,
                                RedirectStandardError = true
                            }
                        };
                        TiltAlign.OutputDataReceived += Handler;
                        TiltAlign.ErrorDataReceived += Handler;

                        TiltAlign.Start();

                        TiltAlign.BeginOutputReadLine();
                        TiltAlign.BeginErrorReadLine();

                        TiltAlign.WaitForExit();
                    }

                    if (Options.DoPatchTracking)
                        Console.WriteLine($"Finished for {TiltSeriesPath}");
                }
                else if (Command.Name == "TomoEtomoFiducials")
                {
                    string TiltSeriesPath = (string)Command.Content[0];
                    var Options = (ProcessingOptionsTomoEtomoFiducials)Command.Content[1];

                    TiltSeries T = new TiltSeries(TiltSeriesPath);
                    
                    // first generate a directive file to run Etomo automatically through batchruntomo
                    int RotOption = Options.DoAxisAngleSearch ? -1 : 0; // fit single value (-1) or leave fixed (0)
                    var BRTConfig = $"setupset.copyarg.userawtlt = 1\n" +
                                    $"setupset.copyarg.stackext = st\n" +
                                    $"setupset.copyarg.rotation = {Options.AxisAngle}\n" +
                                    $"setupset.copyarg.pixel = {Options.TiltStackAngPix / 10}\n" +
                                    $"setupset.copyarg.gold = {Options.FiducialSizeNanometers}\n" +
                                    $"setupset.copyarg.dual = 0\n" +
                                    $"comparam.xcorr.tiltxcorr.FilterRadius2 = 0.5\n" +
                                    $"comparam.xcorr.tiltxcorr.ExcludeCentralPeak = 1\n" +
                                    $"runtime.Fiducials.any.trackingMethod = 0\n" +
                                    $"runtime.Fiducials.any.seedingMethod = 1\n" +
                                    $"comparam.autofidseed.autofidseed.TargetNumberOfBeads = {Options.TargetNBeads}\n" +
                                    $"comparam.align.tiltalign.MagOption = 0\n" +
                                    $"comparam.align.tiltalign.TiltOption = 0\n" +
                                    $"comparam.align.tiltalign.RotOption = {RotOption}\n" +
                                    $"comparam.align.tiltalign.RobustFitting = 1\n" +
                                    $"comparam.align.tiltalign.WeightWholeTracks = 1\n";

                    var DirectiveFile = Path.Combine(Path.GetTempPath(), Path.GetTempFileName());
                    DirectiveFile = Path.ChangeExtension(DirectiveFile, ".adoc");
                    File.WriteAllText(path: DirectiveFile, contents: BRTConfig);
                    
                    // then run batchruntomo
                    // only do setup and alignment calculation if on second pass, otherwise do fiducial model generation too
                    int EndingStep = Options.DoFiducialTracking ? 5 : 0;
                    string Arguments = $"-DirectiveFile {DirectiveFile} -CurrentLocation {T.TiltStackDir} -RootName {T.RootName} -EndingStep {EndingStep}";

                    if (Options.DoFiducialTracking)
                        Console.WriteLine($"Performing fiducial tracking in {T.TiltStackDir} with arguments: {Arguments}");

                    // we execute batchruntomo even if not doing fiducial tracking to create com file for alignment
                    bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
                    string BatchRunTomoExecutable = IsWindows ? "batchruntomo.cmd" : "batchruntomo";
                    Console.WriteLine($"Running '{BatchRunTomoExecutable} {Arguments}'");
                    Process BatchRunTomo = new Process 
                    {
                        StartInfo =
                        {
                            FileName = BatchRunTomoExecutable,
                            CreateNoWindow = false,
                            WindowStyle = ProcessWindowStyle.Minimized,
                            Arguments = Arguments,
                            WorkingDirectory = T.TiltStackDir,
                            RedirectStandardOutput = true,
                            RedirectStandardError = true
                        }
                    };
                    DataReceivedEventHandler Handler = (sender, args) => { if (args.Data != null) Console.WriteLine(args.Data); };
                    BatchRunTomo.OutputDataReceived += Handler;
                    BatchRunTomo.ErrorDataReceived += Handler;

                    BatchRunTomo.Start();

                    BatchRunTomo.BeginOutputReadLine();
                    BatchRunTomo.BeginErrorReadLine();

                    BatchRunTomo.WaitForExit();

                    // run alignment separately from batchruntomo to avoid expensive cross-validation calculations
                    if (Options.DoTiltAlign)
                    {
                        Console.WriteLine($"Calculating projection parameters from fiducial tracking results in {T.TiltStackDir}");
                        string SubMfgExecutable = IsWindows ? "submfg.cmd" : "submfg";
                        Console.WriteLine($"Running '{SubMfgExecutable} align.com'");
                        Process TiltAlign = new Process
                        {
                            StartInfo =
                            {
                                FileName = SubMfgExecutable,
                                CreateNoWindow = false,
                                WindowStyle = ProcessWindowStyle.Minimized,
                                Arguments = "align.com",
                                WorkingDirectory = T.TiltStackDir,
                                RedirectStandardOutput = true,
                                RedirectStandardError = true
                            }
                        };
                        TiltAlign.OutputDataReceived += Handler;
                        TiltAlign.ErrorDataReceived += Handler;

                        TiltAlign.Start();

                        TiltAlign.BeginOutputReadLine();
                        TiltAlign.BeginErrorReadLine();

                        TiltAlign.WaitForExit();
                    }

                    if (Options.DoFiducialTracking)
                        Console.WriteLine($"Finished for {TiltSeriesPath}");
                }
                else if (Command.Name == "TomoProcessCTF")
                {
                    string Path = (string)Command.Content[0];
                    ProcessingOptionsMovieCTF Options = (ProcessingOptionsMovieCTF)Command.Content[1];

                    TiltSeries T = new TiltSeries(Path);
                    T.ProcessCTFSimultaneous(Options);
                    T.SaveMeta();

                    Console.WriteLine($"Processed CTF for {Path}");
                }
                else if (Command.Name == "TomoAlignLocallyWithoutReferences")
                {
                    string Path = (string)Command.Content[0];
                    ProcessingOptionsTomoFullReconstruction Options = (ProcessingOptionsTomoFullReconstruction)Command.Content[1];

                    TiltSeries T = new TiltSeries(Path);
                    T.AlignLocallyWithoutReferences(Options);
                    T.SaveMeta();

                    Console.WriteLine($"Aligned tilts for {Path}");
                }
                else if (Command.Name == "TomoReconstruct")
                {
                    string Path = (string)Command.Content[0];
                    var Options = (ProcessingOptionsTomoFullReconstruction)Command.Content[1];

                    string LastMessage = "";
                    TiltSeries T = new TiltSeries(Path);
                    T.ReconstructFull(Options, (grid, gridElements, message) =>
                    {
                        if (message != LastMessage)
                        {
                            LastMessage = message;
                            Console.WriteLine(message);
                        }
                        return false;
                    });

                    Console.WriteLine($"Reconstructed full tomogram for {Path}");
                }
                else if (Command.Name == "TomoMatch")
                {
                    string Path = (string)Command.Content[0];
                    var Options = (ProcessingOptionsTomoFullMatch)Command.Content[1];
                    var TemplatePath = (string)Command.Content[2];

                    Image Template = Image.FromFile(TemplatePath);

                    string LastMessage = "";
                    TiltSeries T = new TiltSeries(Path);
                    T.MatchFull(Options, Template, (grid, gridElements, message) =>
                    {
                        if (message != LastMessage)
                        {
                            LastMessage = message;
                            Console.WriteLine(message);
                        }
                        Console.WriteLine($"{(float)gridElements / grid.Elements() * 100}%");
                        return false;
                    });

                    Template.Dispose();

                    Console.WriteLine($"Template-matched {Path}");
                }
                else if (Command.Name == "TomoExportParticleSubtomos")
                {
                    string Path = (string)Command.Content[0];
                    ProcessingOptionsTomoSubReconstruction Options = (ProcessingOptionsTomoSubReconstruction)Command.Content[1];
                    float3[] Coordinates = (float3[])Command.Content[2];
                    float3[] Angles = Command.Content[3] != null ? (float3[])Command.Content[3] : null;

                    TiltSeries T = new TiltSeries(Path);
                    T.ReconstructSubtomos(Options, Coordinates, Angles);
                    T.SaveMeta();

                    Console.WriteLine($"Exported {Coordinates.Length / T.NTilts} particles for {Path}");
                }
                else if (Command.Name == "TomoExportParticleSeries")
                {
                    string Path = (string)Command.Content[0];
                    ProcessingOptionsTomoSubReconstruction Options = (ProcessingOptionsTomoSubReconstruction)Command.Content[1];
                    float3[] Coordinates = (float3[])Command.Content[2];
                    float3[] Angles = Command.Content[3] != null ? (float3[])Command.Content[3] : null;
                    string PathsRelativeTo = (string)Command.Content[4];
                    string PathTableOut = (string)Command.Content[5];

                    Star TableOut;

                    TiltSeries T = new TiltSeries(Path);
                    T.ReconstructParticleSeries(Options, Coordinates, Angles, PathsRelativeTo, out TableOut);
                    T.SaveMeta();

                    if (!string.IsNullOrEmpty(PathTableOut))
                        TableOut.Save(PathTableOut);

                    Console.WriteLine($"Exported {Coordinates.Length / T.NTilts} particles for {Path}");
                }
                else if (Command.Name == "MPAPrepareSpecies")
                {
                    string Path = (string)Command.Content[0];
                    string StagingSave = (string)Command.Content[1];

                    Species S = Species.FromFile(Path);
                    Console.Write($"Preparing {S.Name} for refinement... ");

                    S.PrepareRefinementRequisites(true, DeviceID, StagingSave, null);

                    Console.WriteLine("Done.");
                }
                else if (Command.Name == "MPAPreparePopulation")
                {
                    string Path = (string)Command.Content[0];
                    string StagingLoad = (string)Command.Content[1];

                    MPAPopulation = new Population(Path);

                    foreach (var species in MPAPopulation.Species)
                    {
                        Console.Write($"Preparing {species.Name} for refinement... ");

                        species.PrepareRefinementRequisites(true, DeviceID, null, StagingLoad);

                        Console.WriteLine("Done.");
                    }
                }
                else if (Command.Name == "MPARefine")
                {
                    string Path = (string)Command.Content[0];
                    string WorkingDirectory = (string)Command.Content[1];
                    ProcessingOptionsMPARefine Options = (ProcessingOptionsMPARefine)Command.Content[2];
                    DataSource Source = (DataSource)Command.Content[3];

                    Movie Item = null;

                    if (Helper.PathToExtension(Path).ToLower() == ".tomostar")
                        Item = new TiltSeries(Path);
                    else
                        Item = new Movie(Path);

                    GPU.SetDevice(DeviceID);

                    Item.PerformMultiParticleRefinement(WorkingDirectory, Options, MPAPopulation.Species.ToArray(), Source, GainRef, DefectMap, Console.WriteLine);

                    Item.SaveMeta();

                    GPU.CheckGPUExceptions();

                    Console.WriteLine($"Finished refining {Item.Name}");
                }
                else if (Command.Name == "MPASaveProgress")
                {
                    string Path = (string)Command.Content[0];

                    MPAPopulation.SaveRefinementProgress(Path);
                }
                else if (Command.Name == "MPAFinishSpecies")
                {
                    string Path = (string)Command.Content[0];
                    string StagingDirectory = (string)Command.Content[1];
                    string[] ProgressFolders = (string[])Command.Content[2];

                    Species S = Species.FromFile(Path);
                    S.PrepareRefinementRequisites(false, 0, null, StagingDirectory);
                    S.GatherRefinementProgress(ProgressFolders, DeviceID);
                    S.FinishRefinement(DeviceID);
                    S.Commit();
                    S.Save();
                }

                Watch.Stop();
                Console.WriteLine($"Execution took {(Watch.ElapsedMilliseconds / 1000f):F3} seconds");

                Console.WriteLine("");
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());

                File.WriteAllText($"worker_{Port}_crash.txt", e.ToString());

                throw;
            }
        }

        #endregion

        #region Data loading

        static Image LoadAndPrepareGainReference(string path, bool flipX, bool flipY, bool transpose)
        {
            Image Gain = Image.FromFilePatient(10, 500,
                                               path,
                                               HeaderlessDims,
                                               (int)HeaderlessOffset,
                                               ImageFormatsHelper.StringToType(HeaderlessType));

            float Mean = MathHelper.Mean(Gain.GetHost(Intent.Read)[0]);
            Gain.TransformValues(v => v == 0 ? 1 : v / Mean);

            if (flipX)
                Gain = Gain.AsFlippedX();
            if (flipY)
                Gain = Gain.AsFlippedY();
            if (transpose)
                Gain = Gain.AsTransposed();

            return Gain;
        }

        static DefectModel LoadAndPrepareDefectMap(string path, bool flipX, bool flipY, bool transpose)
        {
            Image Defects = Image.FromFilePatient(10, 500,
                                                  path,
                                                  HeaderlessDims,
                                                  (int)HeaderlessOffset,
                                                  ImageFormatsHelper.StringToType(HeaderlessType));

            if (flipX)
                Defects = Defects.AsFlippedX();
            if (flipY)
                Defects = Defects.AsFlippedY();
            if (transpose)
                Defects = Defects.AsTransposed();

            DefectModel Model = new DefectModel(Defects, 4);
            Defects.Dispose();

            return Model;
        }

        static Image LoadAndPrepareStack(string path, decimal scaleFactor, bool correctGain, int maxThreads = 8)
        {
            Image stack = null;

            MapHeader header = MapHeader.ReadFromFilePatient(10, 500,
                                                             path,
                                                             HeaderlessDims,
                                                             (int)HeaderlessOffset,
                                                             ImageFormatsHelper.StringToType(HeaderlessType));

            string Extension = Helper.PathToExtension(path).ToLower();
            bool IsTiff = header.GetType() == typeof(HeaderTiff);
            bool IsEER = header.GetType() == typeof(HeaderEER);

            if (GainRef != null && correctGain)
                if (!IsEER)
                    if (header.Dimensions.X != GainRef.Dims.X || header.Dimensions.Y != GainRef.Dims.Y)
                        throw new Exception($"Gain reference dimensions ({GainRef.Dims.X}x{GainRef.Dims.Y}) do not match image ({header.Dimensions.X}x{header.Dimensions.Y}).");

            int EERSupersample = 1;
            if (GainRef != null && correctGain && IsEER)
            {
                if (header.Dimensions.X == GainRef.Dims.X)
                    EERSupersample = 1;
                else if (header.Dimensions.X * 2 == GainRef.Dims.X)
                    EERSupersample = 2;
                else if (header.Dimensions.X * 4 == GainRef.Dims.X)
                    EERSupersample = 3;
                else
                    throw new Exception("Invalid supersampling factor requested for EER based on gain reference dimensions");
            }
            int EERGroupFrames = 1;
            if (IsEER)
            {
                if (HeaderEER.GroupNFrames > 0)
                    EERGroupFrames = HeaderEER.GroupNFrames;
                else if (HeaderEER.GroupNFrames < 0)
                {
                    int NFrames = -HeaderEER.GroupNFrames;
                    EERGroupFrames = header.Dimensions.Z / NFrames;
                }

                header.Dimensions.Z /= EERGroupFrames;
            }

            HeaderEER.SuperResolution = EERSupersample;

            if (IsEER && GainRef != null && correctGain)
            {
                header.Dimensions.X = GainRef.Dims.X;
                header.Dimensions.Y = GainRef.Dims.Y;
            }

            int NThreads = (IsTiff || IsEER) ? 1 : 1;
            int GPUThreads = 1;

            int CurrentDevice = GPU.GetDevice();

            if (RawLayers == null || RawLayers.Length != NThreads || RawLayers[0].Length < header.Dimensions.ElementsSlice())
                RawLayers = Helper.ArrayOfFunction(i => new float[header.Dimensions.ElementsSlice()], NThreads);

            Image[] GPULayers = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, header.Dimensions.Slice()), GPUThreads);
            Image[] GPULayers2 = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, header.Dimensions.Slice()), GPUThreads);

            if (scaleFactor == 1M)
            {
                if (OriginalStack == null || OriginalStack.Dims != header.Dimensions)
                {
                    OriginalStack?.Dispose();
                    OriginalStack = new Image(header.Dimensions);
                }

                stack = OriginalStack;
                float[][] OriginalStackData = stack.GetHost(Intent.Write);

                object[] Locks = Helper.ArrayOfFunction(i => new object(), GPUThreads);

                Helper.ForCPU(0, header.Dimensions.Z, NThreads, threadID => GPU.SetDevice(DeviceID), (z, threadID) =>
                {
                    if (IsTiff)
                        TiffNative.ReadTIFFPatient(10, 500, path, z, true, RawLayers[threadID]);
                    else if (IsEER)
                        EERNative.ReadEERPatient(10, 500, path, z * EERGroupFrames, Math.Min(((HeaderEER)header).DimensionsUngrouped.Z, (z + 1) * EERGroupFrames), EERSupersample, RawLayers[threadID]);
                    else
                        IOHelper.ReadMapFloatPatient(10, 500,
                                                     path,
                                                     HeaderlessDims,
                                                     (int)HeaderlessOffset,
                                                     ImageFormatsHelper.StringToType(HeaderlessType),
                                                     new[] { z },
                                                     null,
                                                     new[] { RawLayers[threadID] });

                    int GPUThreadID = threadID % GPUThreads;

                    lock (Locks[GPUThreadID])
                    {
                        GPU.CopyHostToDevice(RawLayers[threadID], GPULayers[GPUThreadID].GetDevice(Intent.Write), header.Dimensions.ElementsSlice());

                        if (GainRef != null && correctGain)
                        {
                            //if (IsEER)
                            //    GPULayers[GPUThreadID].DivideSlices(GainRef);
                            //else
                                GPULayers[GPUThreadID].MultiplySlices(GainRef); // EER .gain is now multiplicative??
                        }

                        if (DefectMap != null)
                        {
                            GPU.CopyDeviceToDevice(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                                   GPULayers2[GPUThreadID].GetDevice(Intent.Write),
                                                   header.Dimensions.Elements());
                            DefectMap.Correct(GPULayers2[GPUThreadID], GPULayers[GPUThreadID]);
                        }

                        //GPU.Xray(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                        //         GPULayers2[GPUThreadID].GetDevice(Intent.Write),
                        //         20f,
                        //         new int2(header.Dimensions),
                        //         1);

                        GPU.CopyDeviceToHost(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                             OriginalStackData[z],
                                             header.Dimensions.ElementsSlice());
                    }

                }, null);
            }
            else
            {
                int3 ScaledDims = new int3((int)Math.Round(header.Dimensions.X * scaleFactor) / 2 * 2,
                                            (int)Math.Round(header.Dimensions.Y * scaleFactor) / 2 * 2,
                                            header.Dimensions.Z);

                if (OriginalStack == null || OriginalStack.Dims != ScaledDims)
                {
                    OriginalStack?.Dispose();
                    OriginalStack = new Image(ScaledDims);
                }

                stack = OriginalStack;
                float[][] OriginalStackData = stack.GetHost(Intent.Write);

                int[] PlanForw = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(header.Dimensions.Slice(), 1), GPUThreads);
                int[] PlanBack = Helper.ArrayOfFunction(i => GPU.CreateIFFTPlan(ScaledDims.Slice(), 1), GPUThreads);

                Image[] GPULayersInputFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, header.Dimensions.Slice(), true, true), GPUThreads);
                Image[] GPULayersOutputFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, ScaledDims.Slice(), true, true), GPUThreads);

                Image[] GPULayersScaled = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, ScaledDims.Slice()), GPUThreads);

                object[] Locks = Helper.ArrayOfFunction(i => new object(), GPUThreads);

                Helper.ForCPU(0, ScaledDims.Z, NThreads, threadID => GPU.SetDevice(DeviceID), (z, threadID) =>
                {
                    if (IsTiff)
                        TiffNative.ReadTIFFPatient(10, 500, path, z, true, RawLayers[threadID]);
                    else if (IsEER)
                        EERNative.ReadEERPatient(10, 500, path, z * EERGroupFrames, Math.Min(((HeaderEER)header).DimensionsUngrouped.Z, (z + 1) * EERGroupFrames), EERSupersample, RawLayers[threadID]);
                    else
                        IOHelper.ReadMapFloatPatient(10, 500,
                                                     path,
                                                     HeaderlessDims,
                                                     (int)HeaderlessOffset,
                                                     ImageFormatsHelper.StringToType(HeaderlessType),
                                                     new[] { z },
                                                     null,
                                                     new[] { RawLayers[threadID] });

                    int GPUThreadID = threadID % GPUThreads;

                    lock (Locks[GPUThreadID])
                    {
                        GPU.CopyHostToDevice(RawLayers[threadID], GPULayers[GPUThreadID].GetDevice(Intent.Write), header.Dimensions.ElementsSlice());

                        if (GainRef != null && correctGain)
                        {
                            //if (IsEER)
                            //    GPULayers[GPUThreadID].DivideSlices(GainRef);
                            //else
                                GPULayers[GPUThreadID].MultiplySlices(GainRef); // EER .gain is now multiplicative??
                        }

                        if (DefectMap != null)
                        {
                            GPU.CopyDeviceToDevice(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                                   GPULayers2[GPUThreadID].GetDevice(Intent.Write),
                                                   header.Dimensions.Elements());
                            DefectMap.Correct(GPULayers2[GPUThreadID], GPULayers[GPUThreadID]);
                        }

                        //GPU.Xray(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                        //         GPULayers2[GPUThreadID].GetDevice(Intent.Write),
                        //         20f,
                        //         new int2(header.Dimensions),
                        //         1);

                        GPU.Scale(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                  GPULayersScaled[GPUThreadID].GetDevice(Intent.Write),
                                  header.Dimensions.Slice(),
                                  ScaledDims.Slice(),
                                  1,
                                  PlanForw[GPUThreadID],
                                  PlanBack[GPUThreadID],
                                  GPULayersInputFT[GPUThreadID].GetDevice(Intent.Write),
                                  GPULayersOutputFT[GPUThreadID].GetDevice(Intent.Write));

                        GPU.CopyDeviceToHost(GPULayersScaled[GPUThreadID].GetDevice(Intent.Read),
                                             OriginalStackData[z],
                                             ScaledDims.ElementsSlice());
                    }

                }, null);

                for (int i = 0; i < GPUThreads; i++)
                {
                    GPU.DestroyFFTPlan(PlanForw[i]);
                    GPU.DestroyFFTPlan(PlanBack[i]);
                    GPULayersInputFT[i].Dispose();
                    GPULayersOutputFT[i].Dispose();
                    GPULayersScaled[i].Dispose();
                }
            }

            foreach (var layer in GPULayers)
                layer.Dispose();
            foreach (var layer in GPULayers2)
                layer.Dispose();

            return stack;
        }

        #endregion
    }
}
