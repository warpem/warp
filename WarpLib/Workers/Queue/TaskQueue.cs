using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Warp.Workers.Queue
{
    public readonly struct QueueSummary
    {
        public int Pending { get; init; }
        public int Running { get; init; }
        public int Done { get; init; }
        public int Failed { get; init; }
        public int Poisoned { get; init; }
    }

    /// <summary>
    /// Filesystem work queue. State transitions are atomic os.rename / File.Move
    /// within one filesystem (spec §11.1). Safe on local disk and well-configured
    /// Lustre/GPFS/NFSv4.
    /// </summary>
    public class TaskQueue
    {
        private readonly QueueLayout _layout;

        public TaskQueue(QueueLayout layout) { _layout = layout; }

        public void Enqueue(TaskItem task)
        {
            if (string.IsNullOrEmpty(task.CreatedAt))
                task.CreatedAt = DateTime.UtcNow.ToString("o");
            AtomicWrite(Path.Combine(_layout.Pending, task.TaskId + ".json"), task.ToJson());
        }

        /// <summary>
        /// Atomically claim one pending task. Honors allowedStages: tasks whose
        /// stage is not in the set are skipped (left in pending). Returns null if
        /// nothing claimable. Lost races (FileNotFoundException on Move) are
        /// retried on the next candidate.
        /// </summary>
        public TaskItem ClaimOne(string workerId, IEnumerable<string> allowedStages = null)
        {
            string wdir = _layout.RunningFor(workerId);
            Directory.CreateDirectory(wdir);

            HashSet<string> allowed = allowedStages == null ? null : new HashSet<string>(allowedStages);

            if (!Directory.Exists(_layout.Pending)) return null;
            string[] candidates = Directory.GetFiles(_layout.Pending, "*.json");

            // Claim in random order, NOT sorted. Workers have no knowledge of each
            // other, so a deterministic order (e.g. Ordinal) makes every worker target
            // the same lexicographically-first file: all but one lose the File.Move race
            // and fall through, producing a thundering herd of failed metadata ops on the
            // same inodes — worst exactly at startup when pending/ is largest. Sampling a
            // random permutation spreads concurrent workers across the pending set with no
            // coordination, approximating load balancing. On a lost race we simply advance
            // to the next random candidate. Processing order is irrelevant here; the
            // processed/failed snapshots are re-sorted canonically before they are written.
            for (int i = candidates.Length - 1; i > 0; i--)
            {
                int j = System.Random.Shared.Next(i + 1);
                (candidates[i], candidates[j]) = (candidates[j], candidates[i]);
            }

            foreach (string src in candidates)
            {
                if (allowed != null)
                {
                    string stage;
                    try { stage = TaskItem.FromJson(File.ReadAllText(src)).Stage ?? ""; }
                    catch (FileNotFoundException) { continue; } // raced
                    if (!allowed.Contains(stage))
                        continue;
                }

                string dst = Path.Combine(wdir, Path.GetFileName(src));
                try { File.Move(src, dst); }
                catch (FileNotFoundException) { continue; } // lost the race; next candidate

                return TaskItem.FromJson(File.ReadAllText(dst));
            }
            return null;
        }

        public void MarkDone(string workerId, TaskItem task, object result)
        {
            task.Result = result;
            string src = Path.Combine(_layout.RunningFor(workerId), task.TaskId + ".json");
            string dst = Path.Combine(_layout.Done, task.TaskId + ".json");
            // Publish final content atomically into done/ BEFORE deleting the
            // running/ copy, so a crash mid-operation leaves an extra copy in
            // running/ (harmlessly re-pended by sweep) rather than losing the task.
            AtomicWrite(dst, task.ToJson());
            if (File.Exists(src)) File.Delete(src);
        }

        public void MarkFailed(string workerId, TaskItem task, string error, string hostname = null)
        {
            task.Error = error;
            task.FailedOnHost = hostname;
            string src = Path.Combine(_layout.RunningFor(workerId), task.TaskId + ".json");
            string dst = Path.Combine(_layout.Failed, task.TaskId + ".json");
            // Same publish-then-delete ordering as MarkDone: terminal record is
            // durable before the running/ copy is removed.
            AtomicWrite(dst, task.ToJson());
            if (File.Exists(src)) File.Delete(src);
        }

        /// <summary>
        /// Remove every task file from running/&lt;workerId&gt;/ and return the task items
        /// with RetryCount already incremented. Does NOT re-enqueue or poison — the caller
        /// decides. Used by the Scheduler sweep (which applies the failure matrix before
        /// re-pending or poisoning) and by Clear() (which discards the return value).
        /// </summary>
        public List<TaskItem> RecoverOrphans(string workerId)
        {
            string wdir = _layout.RunningFor(workerId);
            var recovered = new List<TaskItem>();
            if (!Directory.Exists(wdir)) return recovered;

            foreach (string f in Directory.GetFiles(wdir, "*.json"))
            {
                try
                {
                    TaskItem t = TaskItem.FromJson(File.ReadAllText(f));
                    t.RetryCount++;
                    File.Delete(f);
                    recovered.Add(t);
                }
                catch (FileNotFoundException) { continue; } // raced away; skip
                catch (IOException) { continue; }           // concurrent claim/sweep; skip
            }
            return recovered;
        }

        /// <summary>
        /// Delete all task files from every queue directory. Call before a new
        /// run to prevent stale done/ or poisoned/ files from matching this run's
        /// task_ids and causing WorkPool.Distribute to return false terminal results
        /// without the workers processing anything.
        /// </summary>
        public void Clear()
        {
            foreach (string dir in new[] {
                _layout.Pending, _layout.Done, _layout.Failed, _layout.Poisoned })
            {
                if (!Directory.Exists(dir)) continue;
                foreach (string f in Directory.GetFiles(dir, "*.json"))
                    try { File.Delete(f); } catch { }
            }
            // running/ subdirs are only created by live workers; any that exist here
            // are orphans from a previous crash. Recover them so they don't block the run.
            if (Directory.Exists(_layout.Running))
                foreach (string wdir in Directory.GetDirectories(_layout.Running))
                    RecoverOrphans(Path.GetFileName(wdir));
        }

        public QueueSummary Summary()
        {
            int Count(string dir) => Directory.Exists(dir) ? Directory.GetFiles(dir, "*.json").Length : 0;
            int running = 0;
            if (Directory.Exists(_layout.Running))
                foreach (string wdir in Directory.GetDirectories(_layout.Running))
                    running += Directory.GetFiles(wdir, "*.json").Length;

            return new QueueSummary
            {
                Pending = Count(_layout.Pending),
                Running = running,
                Done = Count(_layout.Done),
                Failed = Count(_layout.Failed),
                Poisoned = Count(_layout.Poisoned),
            };
        }

        private static void AtomicWrite(string path, string content)
        {
            // Include a GUID so two threads in the same process writing the same
            // target path never collide on the same tmp file name.
            string tmp = path + ".tmp." + Environment.ProcessId + "." + Guid.NewGuid().ToString("N");
            File.WriteAllText(tmp, content);
            File.Move(tmp, path, overwrite: true);
        }
    }
}
