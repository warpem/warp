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
    public class Scheduler
    {
        private readonly QueueLayout _layout;
        private readonly TaskQueue _queue;
        private readonly IWorkerProvisioner _provisioner;
        private readonly int _target;
        private readonly long _workerStallTimeoutMs;
        private readonly long _workerStartupGraceMs;
        private readonly HeartbeatWriter _managerHeartbeat;
        private readonly FailureMatrix _failures;
        private FileStream _lockHandle;

        // Per-worker monitors, created lazily as worker dirs appear.
        private readonly System.Collections.Generic.Dictionary<string, HeartbeatMonitor> _monitors = new();

        public Scheduler(QueueLayout layout, TaskQueue queue, IWorkerProvisioner provisioner,
                         int target, long workerStallTimeoutMs = 30_000, long workerStartupGraceMs = 60_000,
                         FailureMatrix failureMatrix = null)
        {
            _layout = layout;
            _queue = queue;
            _provisioner = provisioner;
            _target = target;
            _workerStallTimeoutMs = workerStallTimeoutMs;
            _workerStartupGraceMs = workerStartupGraceMs;
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

                if (_failures.ShouldPoison(t.TaskId) || _failures.ShouldPoisonByRetry(t.RetryCount + 1))
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
                        _queue.RecoverOrphans(workerId);   // re-pend; sweep handles dead worker
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

        /// <summary>Run until drained. poll = ms between ticks.</summary>
        public void RunToDrain(int pollMs = 2000)
        {
            try
            {
                while (true)
                {
                    Tick();
                    if (IsDrained()) { _provisioner.Shutdown(); return; }
                    System.Threading.Thread.Sleep(pollMs);
                }
            }
            finally
            {
                _lockHandle?.Dispose();
                _lockHandle = null;
                try { File.Delete(_layout.Lock); } catch { }
            }
        }
    }
}
