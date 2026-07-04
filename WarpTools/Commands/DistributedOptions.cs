using CommandLine;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Warp;
using Warp.Workers;
using Warp.Workers.Queue;
using Warp.Workers.Scheduling;
using Warp.Tools;

namespace WarpTools.Commands
{
    abstract class DistributedOptions : BaseOptions
    {
        [OptionGroup("Work distribution", 100)]
        [Option("device_list", HelpText = "Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system")]
        public IEnumerable<int> DeviceList { get; set; }

        [Option("perdevice", Default = 1, HelpText = "Number of processes per GPU")]
        public int ProcessesPerDevice { get; set; }

        [Option("task_dir", HelpText = "Directory for the filesystem work queue used by this run. " +
                                       "Defaults to a 'tasks' subdirectory inside the output directory. " +
                                       "Set this to a fast local scratch path when the output directory is on a slow network filesystem.")]
        public string TaskDir { get; set; }

        [OptionGroup("Advanced remote work distribution", 102)]
        [Option("external_provisioner", HelpText = "Don't spawn local worker processes. An external system (e.g. Relay) provisions " +
                                                   "workers that claim tasks from the queue directory. Used for cluster execution.")]
        public bool UseExternalProvisioner { get; set; }

        [Option("cluster_script", HelpText = "Path to a batch-scheduler submission-script template. " +
                                             "Use {{command}} where the WarpWorker2 invocation should go, plus any " +
                                             "{{custom}} placeholders filled by --cluster_var. Presence of this option " +
                                             "selects cluster mode. Assumes the queue dir is on a shared filesystem, " +
                                             "the WarpTools install is at the same path on compute nodes, and the script " +
                                             "shell expands $(hostname)/$$.")]
        public string ClusterScript { get; set; }

        [Option("cluster_config", HelpText = "Path to the cluster queue-definition JSON. Required with --cluster_script. " +
                                             "Fields: submit, submit_job_id_regex (first capture group = job id), cancel.")]
        public string ClusterConfig { get; set; }

        [Option("pool_size", HelpText = "Cluster mode: number of worker jobs to submit to the scheduler.")]
        public int PoolSize { get; set; }

        [Option("cluster_var", HelpText = "Cluster mode: a key=value pair substituted into the submission template " +
                                          "(repeatable). Whitespace around '=' is tolerated. Quote values containing " +
                                          "spaces, e.g. --cluster_var \"account=my project\".")]
        public IEnumerable<string> ClusterVars { get; set; }

        public override void Evaluate()
        {
            base.Evaluate();
        }

        /// <summary>
        /// Filesystem work-distribution path: build one task per input item, schedule
        /// them via LocalProvisioner + Scheduler, and block until all are terminal.
        ///
        /// All generic per-item handling is owned here and runs identically for every
        /// ported command: LoadMeta → ProcessingStatus update → SaveMeta → atomic live
        /// snapshot write → progress line update → error reporting on failure.
        ///
        /// <paramref name="onSuccess"/> and <paramref name="onFailure"/> are optional
        /// hooks for domain-specific extras that run after the standard handling.
        /// Most commands leave both null.
        /// </summary>
        /// <summary>
        /// Select and construct the worker provisioner for this run: cluster mode
        /// (--cluster_script), external mode (--external_provisioner), or local mode
        /// (default). Sets <paramref name="target"/> to the desired live worker count.
        /// </summary>
        private IWorkerProvisioner CreateProvisioner(
            QueueLayout layout, string logDir, int itemCount, out int target)
        {
            if (!string.IsNullOrEmpty(ClusterScript))
            {
                if (DeviceList != null && DeviceList.Any())
                    Console.Error.WriteLine("Warning: --device_list is ignored in cluster mode " +
                                            "(each cluster job is allocated one GPU by the scheduler).");

                // --pool_size counts cluster jobs (one GPU each); --perdevice worker
                // processes run per job, so the pool holds up to pool_size * perdevice workers.
                target = Math.Min(itemCount, PoolSize);
                string workerExe = Path.Combine(AppContext.BaseDirectory, "WarpWorker2");
                var prov = ClusterProvisioner.Create(
                    clusterScriptPath: ClusterScript,
                    clusterConfigPath: ClusterConfig,
                    externalProvisioner: UseExternalProvisioner,
                    poolSize: PoolSize,
                    perDevice: ProcessesPerDevice,
                    clusterVars: ClusterVars,
                    workerExePath: workerExe,
                    queueDir: layout.Root,
                    logDir: logDir);
                Console.WriteLine($"Distributing {itemCount} item(s) across a cluster pool of up to " +
                                  $"{target} job(s) x {ProcessesPerDevice} worker(s)...");
                return prov;
            }

            if (UseExternalProvisioner)
            {
                target = 0;
                Console.WriteLine($"Distributing {itemCount} item(s); workers provisioned externally...");
                return new ExternalProvisioner();
            }

            List<int> devices = (DeviceList == null || !DeviceList.Any())
                ? Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList()
                : DeviceList.ToList();
            if (devices.Count <= 0)
                throw new Exception("No devices found or specified");
            target = Math.Min(itemCount, devices.Count * ProcessesPerDevice);
            Console.WriteLine($"Distributing {itemCount} item(s) across up to {target} local worker(s)...");
            return new LocalProvisioner(layout.Root, devices.ToArray(), ProcessesPerDevice, logDir: logDir);
        }

        internal (List<T> Processed, List<T> Failed) DistributeItems<T>(
            Func<T, int, TaskItem> buildTask,
            Action<T> onSuccess = null,
            Action<T, WorkResult> onFailure = null,
            int pollMs = 500) where T : Movie
        {
            string queuePath = !string.IsNullOrEmpty(TaskDir)
                ? TaskDir
                : Path.Combine(OutputProcessing, "tasks");
            var layout = new QueueLayout(queuePath);
            layout.EnsureDirectories();
            var queue = new TaskQueue(layout);
            queue.Clear();
            var pool = new WorkPool(layout, queue);

            // Build tasks; apply path correction before handing off to buildTask so
            // every command gets the right m.Path and m.DataPath automatically.
            var tasks = new List<TaskItem>();
            var taskIdToItem = new Dictionary<string, T>();

            for (int i = 0; i < InputSeries.Length; i++)
            {
                T m = (T)InputSeries[i];

                // Path correction: if output dir differs from data dir, relocate the
                // item's meta path into OutputProcessing while preserving the data path.
                if (Path.GetFullPath(OutputProcessing) !=
                    Path.GetFullPath(Path.GetDirectoryName(m.DataPath)))
                {
                    if (string.IsNullOrEmpty(m.DataDirectoryName))
                        m.DataDirectoryName = Path.GetDirectoryName(m.Path);
                    m.Path = Path.Combine(OutputProcessing, Path.GetFileName(m.Path));
                    m.SaveMeta();
                }

                TaskItem task = buildTask(m, i);
                if (task == null) continue;   // buildTask returns null to skip this item
                tasks.Add(task);
                taskIdToItem[task.TaskId] = m;
            }

            // Per-item processing logs go to <output>/logs/<task_id>.log, matching the
            // location the legacy IterateOverItems path used. The queue dir (which may be
            // on fast scratch via --task_dir) keeps only worker lifecycle files.
            string logDir = Path.Combine(OutputProcessing, "logs");
            Directory.CreateDirectory(logDir);

            IWorkerProvisioner provisioner = CreateProvisioner(layout, logDir, InputSeries.Length, out int target);

            var scheduler = new Scheduler(layout, queue, provisioner, target);

            // Enqueue ALL tasks before starting the scheduler thread so workers always
            // find work in pending/ on their first claim attempt. If tasks are enqueued
            // after the scheduler starts, workers may poll an empty queue and exit before
            // any tasks land.
            pool.Enqueue(tasks);

            var processed = new List<T>();
            var failed = new List<T>();

            string jsonSuccessPath = Path.Combine(OutputProcessing, "processed_items.json");
            string jsonFailPath    = Path.Combine(OutputProcessing, "failed_items.json");
            var snapshotTasks = new List<System.Threading.Tasks.Task>();

            // Live progress indicator updated after every item, matching the legacy
            // IterateOverItems format ("{done}/{total}, {failed} failed, {eta} remaining").
            // ETA is wall-time/done * remaining, which already accounts for the worker
            // count because elapsed wall time reflects whatever concurrency is running.
            int total = tasks.Count;
            int nDone = 0, nFailed = 0;
            var progressSync = new object();
            var progressTimer = Stopwatch.StartNew();
            Console.Write($"0/{total}");

            var schedCts = new System.Threading.CancellationTokenSource();
            var schedThread = new Thread(() => scheduler.RunToDrain(cancel: schedCts.Token)) { IsBackground = true };
            schedThread.Start();

            try
            {
                pool.Distribute(tasks,
                    onResult: result =>
                    {
                        T item = taskIdToItem[result.TaskId];
                        bool workerSucceeded = result.Outcome == WorkOutcome.Done;
                        // Tracks the final per-item outcome after the onSuccess hook runs;
                        // a hook exception downgrades a successful worker result to failed.
                        bool itemSucceeded = workerSucceeded;

                        // Standard per-item handling, identical for every ported command.
                        item.LoadMeta();
                        if (workerSucceeded)
                        {
                            item.ProcessingStatus = ProcessingStatus.Processed;
                            item.SaveMeta();

                            if (onSuccess != null)
                            {
                                try
                                {
                                    onSuccess(item);
                                }
                                catch (Exception hookEx)
                                {
                                    // The worker completed successfully but the orchestrator-side
                                    // hook (e.g. ImportAlignments) failed.  Treat this the same as
                                    // a worker failure: deselect the item and log the error.
                                    itemSucceeded = false;
                                    item.UnselectManual = true;
                                    item.ProcessingStatus = ProcessingStatus.LeaveOut;
                                    item.SaveMeta();
                                    failed.Add(item);
                                    result = new WorkResult
                                    {
                                        TaskId = result.TaskId,
                                        Outcome = WorkOutcome.Poisoned,
                                        Error = hookEx.ToString()
                                    };
                                }
                            }

                            if (itemSucceeded)
                                processed.Add(item);
                        }
                        else
                        {
                            item.UnselectManual = true;
                            item.ProcessingStatus = ProcessingStatus.LeaveOut;
                            item.SaveMeta();
                            failed.Add(item);
                            onFailure?.Invoke(item, result);
                        }

                        // Atomic live snapshot: Relay reads these files for progress updates
                        // without parsing per-item XML. Fire on a background task so the
                        // polling thread is not blocked by I/O; take immutable list copies.
                        var snapProcessed = processed.ToList();
                        var snapFailed    = failed.ToList();
                        snapshotTasks.Add(System.Threading.Tasks.Task.Run(() =>
                            WriteItemSnapshots(jsonSuccessPath, jsonFailPath, snapProcessed, snapFailed)));

                        lock (progressSync)
                        {
                            nDone++;
                            if (!itemSucceeded) nFailed++;

                            // On failure: clear the current progress line so the error block
                            // does not append to it, then redraw the updated progress below.
                            if (!itemSucceeded)
                            {
                                if (!StrictFormatting) VirtualConsole.ClearLastLine();
                                Console.Error.WriteLine($"Failed to process {item.Path}, marked as unselected.");
                                Console.Error.WriteLine($"Check logs in {logDir} for more info.");
                                Console.Error.WriteLine("Use the change_selection WarpTool to reactivate this item if required.");
                                if (!string.IsNullOrEmpty(result.Error))
                                    Console.Error.WriteLine("Exception details:\n" + result.Error);
                            }

                            long avgMs = (long)(progressTimer.ElapsedMilliseconds / (double)nDone);
                            TimeSpan remaining = TimeSpan.FromMilliseconds((total - nDone) * avgMs);
                            string failedString = nFailed > 0 ? $", {nFailed} failed" : "";
                            string timeString = remaining.ToString((int)remaining.TotalDays > 0
                                ? @"dd\.hh\:mm\:ss"
                                : ((int)remaining.TotalHours > 0 ? @"hh\:mm\:ss" : @"mm\:ss"));

                            // Clear unconditionally (matching legacy IterateOverItems):
                            // the progress line is refreshed in place by emptying the last
                            // log entry and re-writing it, so consecutive updates consolidate
                            // into one line. Gating this on StrictFormatting (as Relay sets)
                            // makes every Write append instead, producing one ever-growing
                            // line Relay cannot parse.
                            VirtualConsole.ClearLastLine();
                            Console.Write($"{nDone}/{total}{failedString}, {timeString} remaining");
                        }
                    },
                    pollMs: pollMs);
            }
            finally
            {
                // Cancel the scheduler thread so it exits promptly instead of
                // spinning until its next poll interval. Shutdown workers after
                // the thread exits so no new workers are spawned post-cancel.
                schedCts.Cancel();
                schedThread.Join();
                provisioner.Shutdown();
            }

            System.Threading.Tasks.Task.WaitAll(snapshotTasks.ToArray());

            Console.WriteLine();
            Console.WriteLine($"Finished processing in {TimeSpan.FromMilliseconds(progressTimer.ElapsedMilliseconds):hh\\:mm\\:ss}");
            Console.WriteLine($"Finished: {processed.Count} processed, {failed.Count} failed");

            if (failed.Count == total && total > 0)
                throw new Exception("All items failed to process. Check logs for more info.");

            return (processed, failed);
        }

        /// <summary>
        /// Distribute an explicit list of tasks that are NOT derived per-item from
        /// InputSeries, through the same scheduler + worker pool as DistributeItems,
        /// blocking until all reach a terminal state. Used for whole-run steps — e.g.
        /// the single reduce task that finalizes an averaged reconstruction by summing
        /// the per-worker partials. There is no per-item metadata handling here; the
        /// caller owns whatever the tasks read/write.
        /// </summary>
        internal void DistributeTasks(IReadOnlyList<TaskItem> tasks, int pollMs = 500)
        {
            if (tasks == null || tasks.Count == 0)
                return;

            string queuePath = !string.IsNullOrEmpty(TaskDir)
                ? TaskDir
                : Path.Combine(OutputProcessing, "tasks");
            var layout = new QueueLayout(queuePath);
            layout.EnsureDirectories();
            var queue = new TaskQueue(layout);
            queue.Clear();
            var pool = new WorkPool(layout, queue);

            string logDir = Path.Combine(OutputProcessing, "logs");
            Directory.CreateDirectory(logDir);

            IWorkerProvisioner provisioner = CreateProvisioner(layout, logDir, tasks.Count, out int target);

            var scheduler = new Scheduler(layout, queue, provisioner, target);

            var taskList = tasks.ToList();
            pool.Enqueue(taskList);

            int total = taskList.Count;
            int nDone = 0, nFailed = 0;
            var progressTimer = Stopwatch.StartNew();
            Console.Write($"0/{total}");

            var schedCts = new System.Threading.CancellationTokenSource();
            var schedThread = new Thread(() => scheduler.RunToDrain(cancel: schedCts.Token)) { IsBackground = true };
            schedThread.Start();

            try
            {
                pool.Distribute(taskList,
                    onResult: result =>
                    {
                        bool succeeded = result.Outcome == WorkOutcome.Done;
                        nDone++;
                        if (!succeeded)
                        {
                            nFailed++;
                            if (!StrictFormatting) VirtualConsole.ClearLastLine();
                            Console.Error.WriteLine($"Task {result.TaskId} failed.");
                            Console.Error.WriteLine($"Check logs in {logDir} for more info.");
                            if (!string.IsNullOrEmpty(result.Error))
                                Console.Error.WriteLine("Exception details:\n" + result.Error);
                        }

                        VirtualConsole.ClearLastLine();
                        string failedString = nFailed > 0 ? $", {nFailed} failed" : "";
                        Console.Write($"{nDone}/{total}{failedString}");
                    },
                    pollMs: pollMs);
            }
            finally
            {
                schedCts.Cancel();
                schedThread.Join();
                provisioner.Shutdown();
            }

            Console.WriteLine();

            if (nFailed == total && total > 0)
                throw new Exception("All tasks failed to process. Check logs for more info.");
        }

        // Writes two atomic JSON snapshots (processed + failed item lists) for Relay
        // live-progress consumption. Uses a temp-file + rename to prevent partial reads.
        private static void WriteItemSnapshots<T>(
            string successPath, string failPath,
            List<T> processed, List<T> failed) where T : Movie
        {
            AtomicWriteMiniJson(successPath, processed);
            if (failed.Any())
                AtomicWriteMiniJson(failPath, failed);
        }

        private static void AtomicWriteMiniJson<T>(string path, IEnumerable<T> items) where T : Movie
        {
            string tmp = path + ".tmp." + Environment.ProcessId + "." + Guid.NewGuid().ToString("N");
            // Sort canonically by Path so the file is stable regardless of the order items
            // happened to finish (workers now claim tasks in random order). This keeps
            // processed_items.json / failed_items.json — and Relay's GUI, which reads them —
            // in a consistent order across runs and across re-runs.
            var json = new System.Text.Json.Nodes.JsonArray(
                items.OrderBy(m => m.Path, StringComparer.Ordinal)
                     .Select(m => m.ToMiniJson("particles")).ToArray());
            File.WriteAllText(tmp, json.ToJsonString(
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            File.Move(tmp, path, overwrite: true);
        }
    }
}
