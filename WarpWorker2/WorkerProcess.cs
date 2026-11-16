using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using CommandLine;
using Warp;
using Warp.Tools;

namespace WarpWorker2
{
    static partial class WorkerProcess
    {
        static readonly Dictionary<string, MethodInfo> CommandMethods = new();
        static readonly Dictionary<string, MethodInfo> MockCommandMethods = new();

        static int DeviceID = 0;
        // This worker's id, used by accumulating commands to name their per-worker
        // output (e.g. reconstruction partials). Set once in Main.
        static string WorkerId = "";
        static bool MockMode = false;
        static int MockDelayMs = -1;
        static string MockFailCommand = "";
        static string MockFailPath = "";
        static bool DebugMode = false;
        static bool IsSilent = false;

        // Set by the SIGTERM handler; checked between tasks so the current
        // MarkDone/MarkFailed always completes before we honour the signal.
        static volatile bool _sigTermReceived = false;
        static volatile string _heartbeatError = null;

        // The task currently being executed, or null between tasks. Read by the SIGABRT
        // handler so it can call MarkFailed before the process dies.
        static volatile Warp.Workers.Queue.TaskItem _currentTask = null;

        // Worker resource state (loaded by init commands; survives across tasks).
        static Image GainRef = null;
        static DefectModel DefectMap = null;
        static int2 HeaderlessDims = new int2(2);
        static long HeaderlessOffset = 0;
        static string HeaderlessType = "float32";
        static float[][] RawLayers = null;
        static string OriginalStackOwner = "";
        static Image OriginalStack = null;
        // LoadStack left OriginalStack on the GPU; movie commands then keep it there rather than freeing it.
        static bool StackResident = false;
        // The queue and stages this worker claims from, for LoadStack to prefetch the next movie (see MoviePrefetch).
        static Warp.Workers.Queue.TaskQueue PrefetchQueue = null;
        static string[] PrefetchStages = null;
        static BoxNetTorch BoxNetModel = null;
        static NoiseNet3DTorch DenoiserModel = null;
        // Fourier-space accumulators for averaged reconstruction. Allocated by the
        // InitReconstructions init command (amortized once per worker) and added to by
        // each TomoAddToReconstructionAndSave task — so a worker keeps accumulating
        // across the tilt series it claims, and safe-saves its running partial each time.
        static Projector[] Reconstructions = null;
        // Resident population for multi-particle refinement. Loaded once per data source
        // by the MPAPreparePopulation init command and accumulated into by each
        // MPARefineAndSave task (half-map reconstructions + updated poses), which then
        // safe-saves the worker's running progress. Reloaded (and freed) per source.
        static Warp.Sociology.Population MPAPopulation = null;

        static void RegisterCommands()
        {
            var methods = typeof(WorkerProcess).GetMethods(
                BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
            foreach (var method in methods)
            {
                var cmd = method.GetCustomAttribute<CommandAttribute>();
                if (cmd != null) CommandMethods[cmd.Name] = method;
                var mock = method.GetCustomAttribute<MockCommandAttribute>();
                if (mock != null) MockCommandMethods[mock.Name] = method;
            }
        }

        /// <summary>
        /// Execute one command. In mock mode, only MockCommand handlers run — real
        /// GPU commands are skipped entirely so mock mode needs no GPU. Throws on an
        /// unknown (non-mock) command or on handler failure.
        /// </summary>
        static void EvaluateCommand(NamedSerializableObject command)
        {
            if (!MockMode) GPU.SetDevice(DeviceID);
            if (string.IsNullOrWhiteSpace(command?.Name))
                throw new ArgumentException("Command name cannot be null or empty");

            if (MockMode)
            {
                bool injectFailure = (!string.IsNullOrEmpty(MockFailCommand) || !string.IsNullOrEmpty(MockFailPath)) &&
                    (string.IsNullOrEmpty(MockFailCommand) || command.Name == MockFailCommand) &&
                    (string.IsNullOrEmpty(MockFailPath) || command.Content.OfType<string>().Any(path => path.Contains(MockFailPath, StringComparison.Ordinal)) ||
                     command.Content.OfType<string[]>().SelectMany(paths => paths).Any(path => path.Contains(MockFailPath, StringComparison.Ordinal)));
                if (injectFailure) throw new InvalidOperationException($"[MOCK] Injected failure for {command.Name}");
                if (MockCommandMethods.TryGetValue(command.Name, out var mockMethod))
                    mockMethod.Invoke(null, new object[] { command });
                else
                    throw new NotSupportedException($"No CPU mock handler for command '{command.Name}'.");
                return;
            }

            if (CommandMethods.TryGetValue(command.Name, out var method))
                method.Invoke(null, new object[] { command });
            else
                throw new ArgumentException($"Unknown command: '{command.Name}'");
        }

        /// <summary>
        /// Runs a command and logs its wall time, so per-item logs show where a task spends its time, and how much of
        /// it the garbage collector paused the process for.
        /// </summary>
        static void EvaluateTimed(NamedSerializableObject command, string prefix)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            TimeSpan paused = GC.GetTotalPauseDuration();
            int fullCollections = GC.CollectionCount(2);
            EvaluateCommand(command);
            double pausedMs = (GC.GetTotalPauseDuration() - paused).TotalMilliseconds;
            Console.WriteLine($"[time] {prefix}{command.Name} {System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds:F0} ms" +
                              (pausedMs >= 1 ? $", GC paused {pausedMs:F0} ms ({GC.CollectionCount(2) - fullCollections} full)" : ""));
        }

        static readonly System.Diagnostics.Stopwatch UpSince = System.Diagnostics.Stopwatch.StartNew();

        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            OptionsCLI opts = null;
            var parseResult = CommandLine.Parser.Default.ParseArguments<OptionsCLI>(args).WithParsed(o => opts = o);
            if (opts == null)
            {
                CommandLineParserHelper.SetExitCode(parseResult);
                return;
            }

            if (opts.DebugAttach && !System.Diagnostics.Debugger.IsAttached)
                System.Diagnostics.Debugger.Launch();

            if (opts.MockDelayMs < -1)
            {
                Console.Error.WriteLine("--mock-delay-ms must be -1 or nonnegative.");
                Environment.ExitCode = 2;
                return;
            }
            MockMode = opts.Mock;
            MockDelayMs = opts.MockDelayMs;
            MockFailCommand = opts.MockFailCommand ?? "";
            MockFailPath = opts.MockFailPath ?? "";
            DebugMode = opts.Debug;
            IsSilent = opts.Silent;

            // In mock mode we never touch the GPU, so don't call GetDeviceCount or
            // SetDevice — the native CUDA library may not be present on the test machine.
            if (MockMode)
                DeviceID = opts.Device;
            else
            {
                DeviceID = opts.Device % GPU.GetDeviceCount();
                GPU.SetDevice(DeviceID);
            }

            RegisterCommands();

            string workerId = string.IsNullOrEmpty(opts.WorkerId)
                ? $"local-{Environment.ProcessId}-gpu{DeviceID}"
                : opts.WorkerId;
            WorkerId = workerId;

            var layout = new Warp.Workers.Queue.QueueLayout(opts.QueueDir);
            var queue = new Warp.Workers.Queue.TaskQueue(layout);
            PrefetchStages = opts.Stages != null && opts.Stages.Any() ? opts.Stages.ToArray() : null;
            PrefetchQueue = MockMode ? null : queue;
            string wdir = layout.RunningFor(workerId);
            System.IO.Directory.CreateDirectory(wdir);

            // Per-item processing logs. Capture command-body Console output into a
            // VirtualConsole buffer that is flushed to <logDir>/<task_id>.log after
            // every write (mirrors the old WorkerWrapper per-item log files). Run
            // silent so this child process does not write through to the manager's
            // inherited stdout; everything of interest lives in the per-item log.
            string logDir = string.IsNullOrEmpty(opts.LogDir) ? layout.Logs : opts.LogDir;
            System.IO.Directory.CreateDirectory(logDir);
            VirtualConsole.IsSilent = true;
            VirtualConsole.AttachToConsole();

            // Record hostname for the manager's failure matrix (spec §12.3).
            System.IO.File.WriteAllText(System.IO.Path.Combine(wdir, "hostname"), Environment.MachineName);

            // Startup health probe (spec §9.3). Fail => sick + exit.
            if (!MockMode && !GpuHealthProbe.Probe(DeviceID))
            {
                MarkSick(layout, wdir, workerId, "startup health probe failed");
                Environment.Exit(3);
            }

            // SIGTERM handler (spec §A3): set a flag checked between tasks so the
            // current MarkDone/MarkFailed always completes before we honour the signal.
            // Preempted tasks are left in running/ for the sweep — not marked failed.
            // The registration is kept alive for the process lifetime via `using`.
            using var _ = PosixSignalRegistration.Create(PosixSignal.SIGTERM, ctx =>
            {
                ctx.Cancel = true;   // suppress default termination; we exit cleanly below
                _sigTermReceived = true;
            });

            // Heartbeats (spec §8). Worker writes its own; monitors the manager's.
            // Timeout is generous (120 s, not 30 s) because the manager's single
            // scheduler thread refreshes the tick from the same shared filesystem the
            // workers hammer: with many workers listing/moving files in pending/ and
            // running/ on a contended NFS/GPFS mount, an individual metadata syscall on
            // the manager can block for tens of seconds. A 30 s window made every such
            // hiccup look like a dead manager, so all live workers exited at once and the
            // external pool churned (re-provisioned wave after wave). 120 s tolerates the
            // stalls while still catching a genuinely dead manager within ~2 min.
            var myHeartbeat = new Warp.Workers.Queue.HeartbeatWriter(wdir, "hb-");
            var managerMonitor = new Warp.Workers.Queue.HeartbeatMonitor(
                layout.Heartbeat, "tick-", timeoutMs: 120_000, startupGraceMs: 120_000);

            string lastFingerprint = null;
            int consecutiveEmpty = 0;

            // Heartbeat on a dedicated background thread, NOT once per claim. A single
            // task can run for minutes (e.g. a tomogram reconstruction); if the only
            // tick happened at the top of the claim loop, the heartbeat would go stale
            // mid-task, the manager's stall sweep would declare the (still-alive) worker
            // dead, re-pend its task and delete running/<wid>/ — after which the worker
            // crashes writing into the vanished dir and the task is redone concurrently.
            // The background ticker keeps a busy worker visibly alive. It is the sole
            // caller of WriteTick (HeartbeatWriter is not safe for concurrent writers),
            // and stops quietly if the dir is removed by a genuine sweep.
            const int HeartbeatIntervalMs = 5000;
            var hbCts = new System.Threading.CancellationTokenSource();
            var hbThread = new System.Threading.Thread(() =>
            {
                while (!hbCts.IsCancellationRequested)
                {
                    try { myHeartbeat.WriteTick(); }
                    catch (Exception ex)
                    {
                        _heartbeatError = ex.ToString();
                        break;
                    }
                    for (int slept = 0; slept < HeartbeatIntervalMs && !hbCts.IsCancellationRequested; slept += 100)
                        System.Threading.Thread.Sleep(100);
                }
            }) { IsBackground = true, Name = "worker-heartbeat" };
            hbThread.Start();

            try
            {
            while (true)
            {
                if (_heartbeatError != null)
                {
                    WriteExit(layout, workerId, "worker heartbeat failed: " + _heartbeatError);
                    CommandLineParserHelper.SetErrorExitCode();
                    return;
                }

                // Check SIGTERM first — after any prior task completed, before claiming.
                if (_sigTermReceived)
                {
                    WriteExit(layout, workerId, "SIGTERM received");
                    return;
                }

                managerMonitor.Observe();
                if (!DebugMode && managerMonitor.IsStalled())
                {
                    WriteExit(layout, workerId, "manager heartbeat stalled");
                    return;
                }

                // Self-exclusion: if this host is blacklisted, stop claiming GPU
                // work and exit (spec §12.3). CPU-only fallback is Phase 2.
                bool hostBlacklisted = System.IO.File.Exists(
                    System.IO.Path.Combine(layout.Blacklist, Environment.MachineName));
                if (hostBlacklisted && !MockMode)
                {
                    WriteExit(layout, workerId, "host blacklisted");
                    return;
                }

                // A worker on this GPU is running an exclusive task (training): leave the GPU to it.
                if (Warp.Workers.Queue.DeviceLock.IsHeldByOther(layout, Environment.MachineName, DeviceID, Environment.ProcessId))
                {
                    System.Threading.Thread.Sleep(500);
                    continue;
                }

                var task = queue.ClaimOne(workerId, PrefetchStages, MoviePrefetch.PreferredTaskId);
                if (task == null)
                {
                    // Persistent workers keep polling instead of exiting on an empty
                    // queue. This avoids end-of-run churn under an externally managed
                    // pool (Relay relaunches a worker that quits before the manager has
                    // exited) and is required for online processing, where the queue is
                    // transiently empty between bursts. The SIGTERM and manager-heartbeat
                    // checks at the loop top still let a persistent worker exit cleanly
                    // when it is torn down or the manager goes away.
                    if (!opts.Persistent)
                    {
                        consecutiveEmpty++;
                        if (consecutiveEmpty >= 2) { WriteExit(layout, workerId, "queue empty"); return; }
                    }
                    System.Threading.Thread.Sleep(500);
                    continue;
                }
                consecutiveEmpty = 0;

                // Start a fresh per-item log: detach the previous file, reset the
                // buffer, then point at this task's log so init+main output for this
                // item lands in its own <task_id>.log.
                VirtualConsole.FileOutputPath = null;
                VirtualConsole.ClearAll();
                VirtualConsole.FileOutputPath = System.IO.Path.Combine(logDir, task.TaskId + ".log");

                _currentTask = task;

                // ---- init sequence (skip if fingerprint matches) ----
                if (task.InitFingerprint != lastFingerprint)
                {
                    try
                    {
                        foreach (var cmd in task.Init) EvaluateTimed(cmd, "init:");
                        lastFingerprint = task.InitFingerprint;
                    }
                    catch (Exception ex)
                    {
                        if (!MockMode && !GpuHealthProbe.Probe(DeviceID))
                        {
                            _currentTask = null;
                            MarkSick(layout, wdir, workerId, "GPU fault during init: " + Flatten(ex));
                            CommandLineParserHelper.SetErrorExitCode(3);
                            return; // leave task in running/ for the sweep
                        }
                        // Healthy hardware, but init half-ran: reset state, clear fp, fail task, continue.
                        ResetResourceState();
                        lastFingerprint = null;
                        _currentTask = null;
                        queue.MarkFailed(workerId, task, "init failed: " + Flatten(ex), Environment.MachineName);
                        continue;
                    }
                }

                // ---- main sequence ----
                task.CompletedCommands = 0;
                using var exclusive = task.Exclusive ? Warp.Workers.Queue.DeviceLock.Acquire(layout, Environment.MachineName, DeviceID, Environment.ProcessId) : null;
                try
                {
                    for (int i = 0; i < task.Main.Length; i++)
                    {
                        StartErasureAhead(task.Main, i);
                        EvaluateTimed(task.Main[i], ""); task.CompletedCommands++;
                    }
                    _currentTask = null;
                    queue.MarkDone(workerId, task, null);
                }
                catch (Exception ex)
                {
                    if (!MockMode && !GpuHealthProbe.Probe(DeviceID))
                    {
                        _currentTask = null;
                        MarkSick(layout, wdir, workerId, "GPU fault during main: " + Flatten(ex));
                        CommandLineParserHelper.SetErrorExitCode(3);
                        return; // leave task in running/ for the sweep
                    }
                    // Healthy hardware: init state intact, do NOT reset. Fail task, continue.
                    _currentTask = null;
                    queue.MarkFailed(workerId, task, Flatten(ex), Environment.MachineName);
                }
                // Outside the try: the task's outcome is recorded, and a failing release must not turn a finished
                // task into a failed one. Idle workers, and those waiting on another's GPU lock, hold no stack.
                FinishErasure();
                if (!MockMode)
                    try { ReleaseResidentStack(); }
                    catch (Exception ex) { Console.WriteLine("Releasing the resident stack failed: " + Flatten(ex)); }
            }
            }
            finally
            {
                hbCts.Cancel();
                try { hbThread.Join(1000); } catch { }
            }
        }

        static string Flatten(Exception ex) =>
            ex is TargetInvocationException tie && tie.InnerException != null
                ? tie.InnerException.ToString() : ex.ToString();

        static void ResetResourceState()
        {
            GainRef?.Dispose(); GainRef = null;
            MicrographDenoiserModel?.Dispose(); MicrographDenoiserModel = null; MicrographDenoiserPath = "";
            OriginalStack?.Dispose(); OriginalStack = null;
            DefectMap?.Dispose(); DefectMap = null;
            if (Reconstructions != null)
            {
                foreach (var rec in Reconstructions) rec?.Dispose();
                Reconstructions = null;
            }
            if (MPAPopulation != null)
            {
                MPAPopulation.FreeRefinementResources();
                MPAPopulation = null;
            }
            if (!MockMode) GPU.SetDevice(DeviceID);
        }

        static void MarkSick(Warp.Workers.Queue.QueueLayout layout, string wdir, string workerId, string reason)
        {
            try { System.IO.File.WriteAllText(System.IO.Path.Combine(wdir, "sick"), Environment.MachineName + "\n" + reason); } catch { }
            try { System.IO.File.WriteAllText(System.IO.Path.Combine(layout.Sick, workerId), Environment.MachineName + "\n" + reason); } catch { }
            WriteExit(layout, workerId, "sick: " + reason);
        }

        static void WriteExit(Warp.Workers.Queue.QueueLayout layout, string workerId, string reason)
        {
            try { System.IO.File.WriteAllText(System.IO.Path.Combine(layout.Logs, workerId + ".exit"), reason); } catch { }
        }
    }
}
