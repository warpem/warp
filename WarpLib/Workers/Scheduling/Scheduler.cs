using System;
using System.IO;
using System.Linq;
using Warp.Workers.Queue;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// One scheduler for both local and cluster mode (spec §7). Per tick:
    /// write manager heartbeat, sweep stalled workers (via worker heartbeats,
    /// never the cluster scheduler), process failures (matrix + blacklist +
    /// re-pend/poison), top up via the provisioner. Exits when the queue drains.
    /// </summary>
    public class Scheduler : IDisposable
    {
        private readonly QueueLayout _layout;
        private readonly TaskQueue _queue;
        private readonly IWorkerProvisioner _provisioner;
        private readonly int _target;
        private readonly long _workerStallTimeoutMs;
        private readonly long _workerStartupGraceMs;
        private readonly HeartbeatWriter _managerHeartbeat;
        private readonly FailureMatrix _failures;
        private readonly string _logDir;   // external per-item log dir; null → fall back to _layout.Logs
        private FileStream _lockHandle;

        // Per-worker monitors, created lazily as worker dirs appear.
        private readonly System.Collections.Generic.Dictionary<string, HeartbeatMonitor> _monitors = new();

        // Stall/grace default to 120 s to mirror the worker's view of the manager
        // (WorkerProcess.cs). The manager and workers share one contended NFS/GPFS mount,
        // so the same metadata hiccups that can delay the manager's tick can delay a live
        // worker's hb- write (or the manager's read of it). At 30 s the manager would
        // wrongly sweep a still-alive worker, re-pend its in-flight task and delete
        // running/<wid>/ out from under it (duplicate work + the MarkSick crash described
        // in WorkerProcess.cs). Keeping both sides at 120 s avoids that asymmetry; tests
        // inject smaller values to exercise sweeping quickly.
        public Scheduler(QueueLayout layout, TaskQueue queue, IWorkerProvisioner provisioner,
                         int target, long workerStallTimeoutMs = 120_000, long workerStartupGraceMs = 120_000,
                         FailureMatrix failureMatrix = null, string logDir = null)
        {
            _layout = layout;
            _queue = queue;
            _provisioner = provisioner;
            _target = target;
            _workerStallTimeoutMs = workerStallTimeoutMs;
            _workerStartupGraceMs = workerStartupGraceMs;
            _logDir = logDir;
            _managerHeartbeat = new HeartbeatWriter(layout.Heartbeat, "tick-");

            // Load persisted failure matrix if one exists (spec §A2), then apply
            // the caller-supplied matrix on top (for tests that inject specific thresholds).
            var persisted = FailureMatrix.LoadFromFile(layout.ManagerState);
            _failures = persisted != null
                ? persisted.WithThresholds(failureMatrix ?? new FailureMatrix())
                : failureMatrix ?? new FailureMatrix();

            // Acquire an exclusive lock on the queue dir. Fails fast if another
            // Scheduler is already running against the same dir (spec §A1).
            Directory.CreateDirectory(Path.GetDirectoryName(layout.Lock) ?? layout.Root);
            try
            {
                _lockHandle = File.Open(layout.Lock, FileMode.OpenOrCreate,
                    FileAccess.Write, FileShare.None);
            }
            catch (IOException)
            {
                throw new IOException(
                    $"Another manager is already using queue dir '{layout.Root}'. " +
                    "Use --task_dir to choose a different location, or wait for the previous run to finish.");
            }
        }

        public bool IsDrained()
        {
            var s = _queue.Summary();
            // failed/ is transient (drained by ProcessFailures into pending/ or
            // poisoned/), so it does not count toward "drained". A task sitting in
            // failed/ at the drain check would be processed on the next tick.
            return s.Pending == 0 && s.Running == 0 && s.Failed == 0;
        }

        public void Tick()
        {
            _managerHeartbeat.WriteTick();
            SweepStalledWorkers();
            ProcessFailures();
            _failures.SaveToFile(_layout.ManagerState);
            _provisioner.EnsureWorkers(_target);
        }

        /// <summary>
        /// Drain failed/: record each failure into the matrix, blacklist bad hosts,
        /// then either re-pend (below retry cap) or poison the task (spec §10, §12.3).
        /// Writes a summary including the tail of the item's processing log to stderr
        /// so the user can see what went wrong without digging through worker logs.
        /// </summary>
        private void ProcessFailures()
        {
            if (!Directory.Exists(_layout.Failed)) return;

            foreach (string f in Directory.GetFiles(_layout.Failed, "*.json"))
            {
                TaskItem t;
                try { t = TaskItem.FromJson(File.ReadAllText(f)); }
                catch (FileNotFoundException) { continue; }

                if (!string.IsNullOrEmpty(t.FailedOnHost))
                {
                    _failures.RecordFailure(t.FailedOnHost, t.TaskId);
                    foreach (string host in _failures.BlacklistedHosts())
                    {
                        string marker = Path.Combine(_layout.Blacklist, host);
                        if (!File.Exists(marker))
                            try { File.WriteAllText(marker, "blacklisted"); } catch { }
                    }
                }

                bool willPoison = _failures.ShouldPoison(t.TaskId) || _failures.ShouldPoisonByRetry(t.RetryCount + 1);
                string disposition = willPoison
                    ? $"poisoned after {t.RetryCount + 1} attempt(s)"
                    : $"will retry (attempt {t.RetryCount + 1})";
                string failedHost = string.IsNullOrEmpty(t.FailedOnHost) ? "unknown host" : t.FailedOnHost;
                Console.Error.WriteLine($"[pool] task {t.TaskId} failed on {failedHost} — {disposition}");
                if (!string.IsNullOrEmpty(t.Error))
                    Console.Error.WriteLine($"[pool]   error: {t.Error}");
                WriteLogTailToStderr(t.TaskId, tailLines: 8);

                if (willPoison)
                {
                    File.Move(f, Path.Combine(_layout.Poisoned, t.TaskId + ".json"), overwrite: true);
                }
                else
                {
                    // Re-pend for another attempt on a (hopefully) different worker.
                    t.RetryCount++;
                    t.Error = null;
                    t.FailedOnHost = null;
                    File.Delete(f);
                    _queue.Enqueue(t);
                }
            }
        }

        /// <summary>
        /// Write the last <paramref name="tailLines"/> lines of the per-item processing
        /// log to stderr, prefixed so they're easy to find in the manager's SLURM output.
        /// Silently skips if the log doesn't exist (e.g. crash before any output was written).
        /// </summary>
        private void WriteLogTailToStderr(string taskId, int tailLines = 8)
        {
            string logPath = Path.Combine(_logDir ?? _layout.Logs, taskId + ".log");
            if (!File.Exists(logPath)) return;
            try
            {
                string[] lines = File.ReadAllLines(logPath);
                int start = Math.Max(0, lines.Length - tailLines);
                Console.Error.WriteLine($"[pool]   last {lines.Length - start} line(s) from {taskId}.log:");
                for (int i = start; i < lines.Length; i++)
                    Console.Error.WriteLine($"[pool]     {lines[i]}");
            }
            catch { }
        }

        private void SweepStalledWorkers()
        {
            if (!Directory.Exists(_layout.Running)) return;

            foreach (string wdir in Directory.GetDirectories(_layout.Running))
            {
                string workerId = Path.GetFileName(wdir);
                bool hasTasks = Directory.GetFiles(wdir, "*.json").Length > 0;

                if (!_monitors.TryGetValue(workerId, out var mon))
                    _monitors[workerId] = mon = new HeartbeatMonitor(
                        wdir, "hb-", _workerStallTimeoutMs, _workerStartupGraceMs);
                mon.Observe();

                if (mon.IsStalled())
                {
                    if (hasTasks)
                    {
                        // Read the hostname written by the worker so we can attribute
                        // the crash to the right node in the failure matrix.
                        string crashedHost = null;
                        try { crashedHost = File.ReadAllText(Path.Combine(wdir, "hostname")).Trim(); } catch { }

                        var orphans = _queue.RecoverOrphans(workerId);
                        foreach (var t in orphans)
                        {
                            if (crashedHost != null)
                                _failures.RecordFailure(crashedHost, t.TaskId);

                            bool willPoison = _failures.ShouldPoison(t.TaskId) || _failures.ShouldPoisonByRetry(t.RetryCount);
                            string disposition = willPoison
                                ? $"poisoned after {t.RetryCount} attempt(s)"
                                : $"will retry (attempt {t.RetryCount})";
                            string host = crashedHost ?? "unknown host";
                            Console.Error.WriteLine($"[pool] task {t.TaskId} crashed on {host} (worker stalled) — {disposition}");
                            WriteLogTailToStderr(t.TaskId, tailLines: 8);

                            if (willPoison)
                            {
                                string dst = Path.Combine(_layout.Poisoned, t.TaskId + ".json");
                                try { File.WriteAllText(dst, t.ToJson()); } catch { }
                            }
                            else
                            {
                                _queue.Enqueue(t);
                            }
                        }

                        // Blacklist host if it hit the threshold.
                        if (crashedHost != null)
                            foreach (string h in _failures.BlacklistedHosts())
                            {
                                string marker = Path.Combine(_layout.Blacklist, h);
                                if (!File.Exists(marker))
                                    try { File.WriteAllText(marker, "blacklisted"); } catch { }
                            }

                        TryRemoveEmptyDir(wdir);
                    }
                    else
                        TryRemoveEmptyDir(wdir);           // clean exit / sick worker, nothing to recover
                    _monitors.Remove(workerId);
                }
            }
        }

        private static void TryRemoveEmptyDir(string dir)
        {
            try
            {
                // Remove only heartbeat / hostname / sick sidecars, then the dir.
                foreach (string f in Directory.GetFiles(dir))
                    File.Delete(f);
                Directory.Delete(dir);
            }
            catch { /* not empty or racing; leave it */ }
        }

        /// <summary>Release the pool.lock without draining. Use in tests or error paths.</summary>
        public void Dispose()
        {
            _lockHandle?.Dispose();
            _lockHandle = null;
            try { File.Delete(_layout.Lock); } catch { }
        }

        /// <summary>
        /// Run until drained or <paramref name="cancel"/> is requested.
        /// The caller should cancel the token after <c>WorkPool.Distribute</c>
        /// returns so the background scheduler thread exits promptly rather than
        /// spinning until its next poll interval.
        /// </summary>
        public void RunToDrain(int pollMs = 2000,
            System.Threading.CancellationToken cancel = default)
        {
            try
            {
                while (!cancel.IsCancellationRequested)
                {
                    Tick();
                    if (IsDrained()) { _provisioner.Shutdown(); return; }
                    // Sleep in short increments so cancellation is noticed quickly.
                    int slept = 0;
                    while (slept < pollMs && !cancel.IsCancellationRequested)
                    {
                        System.Threading.Thread.Sleep(Math.Min(50, pollMs - slept));
                        slept += 50;
                    }
                }
            }
            finally
            {
                Dispose();
            }
        }
    }
}
