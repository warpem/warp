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

        public Dictionary<string, WorkResult> Distribute(IEnumerable<TaskItem> tasks, int pollMs = 1000)
        {
            var ids = new HashSet<string>();
            foreach (var t in tasks)
            {
                if (string.IsNullOrEmpty(t.InitFingerprint)) t.ComputeInitFingerprint();
                _queue.Enqueue(t);
                ids.Add(t.TaskId);
            }

            var results = new Dictionary<string, WorkResult>();
            while (results.Count < ids.Count)
            {
                foreach (string id in ids)
                {
                    if (results.ContainsKey(id)) continue;

                    if (File.Exists(Path.Combine(_layout.Done, id + ".json")))
                        results[id] = new WorkResult { TaskId = id, Outcome = WorkOutcome.Done };
                    else if (File.Exists(Path.Combine(_layout.Poisoned, id + ".json")))
                    {
                        string err = null;
                        try { err = TaskItem.FromJson(
                            File.ReadAllText(Path.Combine(_layout.Poisoned, id + ".json"))).Error; }
                        catch { }
                        results[id] = new WorkResult { TaskId = id, Outcome = WorkOutcome.Poisoned, Error = err };
                    }
                    // failed/ is transient and owned by the Scheduler; keep waiting.
                }
                if (results.Count < ids.Count)
                    System.Threading.Thread.Sleep(pollMs);
            }
            return results;
        }
    }
}
