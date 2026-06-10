using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using CommandLine;
using Warp;
using Warp.Tools;
using WorkerWrapper = Warp.WorkerWrapper;

namespace WarpWorker2
{
    static partial class WorkerProcess
    {
        static readonly Dictionary<string, MethodInfo> CommandMethods = new();
        static readonly Dictionary<string, MethodInfo> MockCommandMethods = new();

        static int DeviceID = 0;
        static bool MockMode = false;
        static bool DebugMode = false;
        static bool IsSilent = false;

        // Worker resource state (loaded by init commands; survives across tasks).
        static Image GainRef = null;
        static DefectModel DefectMap = null;
        static int2 HeaderlessDims = new int2(2);
        static long HeaderlessOffset = 0;
        static string HeaderlessType = "float32";
        static float[][] RawLayers = null;
        static string OriginalStackOwner = "";
        static Image OriginalStack = null;

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
            GPU.SetDevice(DeviceID);
            if (string.IsNullOrWhiteSpace(command?.Name))
                throw new ArgumentException("Command name cannot be null or empty");

            if (MockMode)
            {
                if (MockCommandMethods.TryGetValue(command.Name, out var mockMethod))
                    mockMethod.Invoke(null, new object[] { command });
                // No mock handler => no-op in mock mode (e.g. init commands like
                // LoadStack do no real GPU work). Intentionally does NOT fall through
                // to the real command.
                return;
            }

            if (CommandMethods.TryGetValue(command.Name, out var method))
                method.Invoke(null, new object[] { command });
            else
                throw new ArgumentException($"Unknown command: '{command.Name}'");
        }

        static readonly System.Diagnostics.Stopwatch UpSince = System.Diagnostics.Stopwatch.StartNew();

        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            OptionsCLI opts = null;
            CommandLine.Parser.Default.ParseArguments<OptionsCLI>(args).WithParsed(o => opts = o);
            if (opts == null) Environment.Exit(2);

            if (opts.DebugAttach && !System.Diagnostics.Debugger.IsAttached)
                System.Diagnostics.Debugger.Launch();

            DeviceID = opts.Device % GPU.GetDeviceCount();
            MockMode = opts.Mock;
            DebugMode = opts.Debug;
            IsSilent = opts.Silent;
            GPU.SetDevice(DeviceID);

            RegisterCommands();

            string workerId = string.IsNullOrEmpty(opts.WorkerId)
                ? $"local-{Environment.ProcessId}-gpu{DeviceID}"
                : opts.WorkerId;

            var layout = new Warp.Workers.Queue.QueueLayout(opts.QueueDir);
            var queue = new Warp.Workers.Queue.TaskQueue(layout);
            string wdir = layout.RunningFor(workerId);
            System.IO.Directory.CreateDirectory(wdir);

            // Record hostname for the manager's failure matrix (spec §12.3).
            System.IO.File.WriteAllText(System.IO.Path.Combine(wdir, "hostname"), Environment.MachineName);

            // Startup health probe (spec §9.3). Fail => sick + exit.
            if (!MockMode && !GpuHealthProbe.Probe(DeviceID))
            {
                MarkSick(layout, wdir, workerId, "startup health probe failed");
                Environment.Exit(3);
            }

            // Heartbeats (spec §8). Worker writes its own; monitors the manager's.
            var myHeartbeat = new Warp.Workers.Queue.HeartbeatWriter(wdir, "hb-");
            var managerMonitor = new Warp.Workers.Queue.HeartbeatMonitor(
                layout.Heartbeat, "tick-", timeoutMs: 30_000, startupGraceMs: 60_000);

            string lastFingerprint = null;
            int consecutiveEmpty = 0;

            while (true)
            {
                myHeartbeat.WriteTick();
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

                var task = queue.ClaimOne(workerId, opts.Stages != null && opts.Stages.Any() ? opts.Stages : null);
                if (task == null)
                {
                    consecutiveEmpty++;
                    if (consecutiveEmpty >= 2) { WriteExit(layout, workerId, "queue empty"); return; }
                    System.Threading.Thread.Sleep(500);
                    continue;
                }
                consecutiveEmpty = 0;

                // ---- init sequence (skip if fingerprint matches) ----
                if (task.InitFingerprint != lastFingerprint)
                {
                    try
                    {
                        foreach (var cmd in task.Init) EvaluateCommand(cmd);
                        lastFingerprint = task.InitFingerprint;
                    }
                    catch (Exception ex)
                    {
                        if (!MockMode && !GpuHealthProbe.Probe(DeviceID))
                        {
                            MarkSick(layout, wdir, workerId, "GPU fault during init: " + Flatten(ex));
                            return; // leave task in running/ for the sweep
                        }
                        // Healthy hardware, but init half-ran: reset state, clear fp, fail task, continue.
                        ResetResourceState();
                        lastFingerprint = null;
                        queue.MarkFailed(workerId, task, "init failed: " + Flatten(ex), Environment.MachineName);
                        continue;
                    }
                }

                // ---- main sequence ----
                try
                {
                    foreach (var cmd in task.Main) EvaluateCommand(cmd);
                    queue.MarkDone(workerId, task, null);
                }
                catch (Exception ex)
                {
                    if (!MockMode && !GpuHealthProbe.Probe(DeviceID))
                    {
                        MarkSick(layout, wdir, workerId, "GPU fault during main: " + Flatten(ex));
                        return; // leave task in running/ for the sweep
                    }
                    // Healthy hardware: init state intact, do NOT reset. Fail task, continue.
                    queue.MarkFailed(workerId, task, Flatten(ex), Environment.MachineName);
                }
            }
        }

        static string Flatten(Exception ex) =>
            ex is TargetInvocationException tie && tie.InnerException != null
                ? tie.InnerException.ToString() : ex.ToString();

        static void ResetResourceState()
        {
            GainRef?.Dispose(); GainRef = null;
            OriginalStack?.Dispose(); OriginalStack = null;
            DefectMap?.Dispose(); DefectMap = null;
            GPU.SetDevice(DeviceID);
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
