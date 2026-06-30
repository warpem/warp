using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warp.Workers.Queue;

namespace Warp.Workers
{
    // Failed is reserved but never returned by Distribute: a failed task is
    // transient (Scheduler re-pends or poisons it), so terminal results are only
    // Done or Poisoned.
    public enum WorkOutcome { Done, Failed, Poisoned }

    public class WorkResult
    {
        public string TaskId { get; init; }
        public WorkOutcome Outcome { get; init; }
        public string Error { get; init; }
    }

    /// <summary>
    /// Enqueues a batch of tasks and blocks until each reaches a TERMINAL state
    /// (done/ or poisoned/), returning results keyed by task_id (spec §6). Does
    /// not touch failed/ — the Scheduler owns retry/poison/blacklist. Assumes a
    /// Scheduler is ticking concurrently.
    /// </summary>
    public class WorkPool
    {
        private readonly QueueLayout _layout;
        private readonly TaskQueue _queue;

        public WorkPool(QueueLayout layout, TaskQueue queue)
        {
            _layout = layout;
            _queue = queue;
        }

        /// <summary>
        /// Enqueue tasks and block until every task reaches a terminal state.
        /// <paramref name="onResult"/> is called synchronously on the polling thread
        /// as each task resolves — use it for per-item status updates and live JSON
        /// writes. The return dictionary contains all results (same data, convenient
        /// for callers that want a batch view after the fact).
        /// </summary>
        /// <summary>
        /// Enqueue a batch of tasks into pending/ and return their ids. Tasks whose
        /// file already exists anywhere in the queue are skipped (idempotent). Call
        /// this before starting the Scheduler so workers always find work on startup.
        /// </summary>
        public IReadOnlyCollection<string> Enqueue(IEnumerable<TaskItem> tasks)
        {
            var ids = new HashSet<string>();
            foreach (var t in tasks)
            {
                if (string.IsNullOrEmpty(t.InitFingerprint)) t.ComputeInitFingerprint();
                // Skip tasks already present anywhere in the queue (idempotent).
                if (!File.Exists(Path.Combine(_layout.Pending,   t.TaskId + ".json")) &&
                    !File.Exists(Path.Combine(_layout.Done,      t.TaskId + ".json")) &&
                    !File.Exists(Path.Combine(_layout.Poisoned,  t.TaskId + ".json")))
                    _queue.Enqueue(t);
                ids.Add(t.TaskId);
            }
            return ids;
        }

        public Dictionary<string, WorkResult> Distribute(
            IEnumerable<TaskItem> tasks,
            Action<WorkResult> onResult = null,
            int pollMs = 1000)
        {
            var ids = Enqueue(tasks);

            var results = new Dictionary<string, WorkResult>();
            while (results.Count < ids.Count)
            {
                foreach (string id in ids)
                {
                    if (results.ContainsKey(id)) continue;

                    WorkResult resolved = null;
                    if (File.Exists(Path.Combine(_layout.Done, id + ".json")))
                        resolved = new WorkResult { TaskId = id, Outcome = WorkOutcome.Done };
                    else if (File.Exists(Path.Combine(_layout.Poisoned, id + ".json")))
                    {
                        string err = null;
                        try { err = TaskItem.FromJson(
                            File.ReadAllText(Path.Combine(_layout.Poisoned, id + ".json"))).Error; }
                        catch { }
                        resolved = new WorkResult { TaskId = id, Outcome = WorkOutcome.Poisoned, Error = err };
                    }
                    // failed/ is transient and owned by the Scheduler; keep waiting.

                    if (resolved != null)
                    {
                        results[id] = resolved;
                        onResult?.Invoke(resolved);
                    }
                }
                if (results.Count < ids.Count)
                    System.Threading.Thread.Sleep(pollMs);
            }
            return results;
        }
    }
}
