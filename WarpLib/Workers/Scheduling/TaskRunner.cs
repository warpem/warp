using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Warp.Tools;
using Warp.Workers.Queue;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Runs an explicit list of tasks through the filesystem work queue (scheduler +
    /// ephemeral worker pool) and blocks until every task reaches a terminal state.
    ///
    /// This is the "whole-run step" form of distribution: unlike WarpTools'
    /// DistributedOptions.DistributeItems, there is no per-item metadata handling
    /// (LoadMeta / ProcessingStatus / SaveMeta / live snapshots) — the caller owns whatever
    /// the tasks read and write. Each call sets up a fresh queue, provisions workers, and
    /// tears them down on completion, so a multi-phase pipeline runs each phase on its own
    /// short-lived pool and only one pool's worth of CUDA-initialized processes is ever alive.
    ///
    /// Exists so MCore's refinement phases, MTools' reconstruction phases, and WarpTools'
    /// reduce steps can share one implementation instead of three near-identical copies.
    /// </summary>
    public static class TaskRunner
    {
        /// <summary>
        /// Enqueue <paramref name="tasks"/> and block until all are terminal.
        /// <paramref name="onItemDone"/> runs single-threaded on the polling thread after
        /// each task finishes — the seam for per-task orchestrator output.
        /// Throws if every task failed.
        /// </summary>
        public static void Run(string queueDir,
                               string logDir,
                               IReadOnlyList<TaskItem> tasks,
                               WorkerPoolSettings settings,
                               Action<TaskItem, WorkResult> onItemDone = null,
                               int pollMs = 500)
        {
            if (tasks == null || tasks.Count == 0)
                return;
            if (settings == null)
                throw new ArgumentNullException(nameof(settings));

            var layout = new QueueLayout(queueDir);
            layout.EnsureDirectories();
            var queue = new TaskQueue(layout);
            queue.Clear();
            var pool = new WorkPool(layout, queue);

            Directory.CreateDirectory(logDir);

            IWorkerProvisioner provisioner =
                settings.CreateProvisioner(layout, logDir, tasks.Count, out int target);

            var scheduler = new Scheduler(layout, queue, provisioner, target, logDir: logDir);

            var taskList = tasks.ToList();
            var taskById = taskList.ToDictionary(t => t.TaskId, t => t);

            // Enqueue ALL tasks before starting the scheduler thread, so workers find work
            // in pending/ on their very first claim attempt. If the scheduler starts first
            // it spawns workers against an empty queue; they poll twice, see nothing and
            // exit, and the run stalls. Distribute's Enqueue is idempotent, so calling it
            // here as well is free.
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
                                if (!settings.StrictFormatting) VirtualConsole.ClearLastLine();
                                Console.Error.WriteLine($"Task {result.TaskId} failed.");
                                Console.Error.WriteLine($"Check logs in {logDir} for more info.");
                                if (!string.IsNullOrEmpty(result.Error))
                                    Console.Error.WriteLine("Exception details:\n" + result.Error);
                            }

                            // Orchestrator hooks must never take down the run; a hook that
                            // throws would otherwise abort Distribute mid-poll and leave
                            // workers orphaned until the finally block.
                            try { onItemDone?.Invoke(task, result); } catch { }

                            VirtualConsole.ClearLastLine();
                            string failedString = nFailed > 0 ? $", {nFailed} failed" : "";
                            Console.Write($"{nDone}/{total}{failedString}");
                        }
                    },
                    pollMs: pollMs);
            }
            finally
            {
                // Cancel the scheduler thread so it exits promptly instead of spinning
                // until its next poll interval. Shut workers down only after it has
                // exited, so no new workers are spawned post-cancel.
                schedCts.Cancel();
                schedThread.Join();
                provisioner.Shutdown();
            }

            Console.WriteLine();

            if (nFailed == total && total > 0)
                throw new Exception("All tasks failed to process. Check logs for more info.");
        }
    }
}
