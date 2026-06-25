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
        [Option("workers", HelpText = "List of remote workers to be used instead of locally spawned processes. Formatted as hostname:port, separated by spaces")]
        public IEnumerable<string> Workers { get; set; }

        [Option("external_provisioner", HelpText = "Don't spawn local worker processes. An external system (e.g. Relay) provisions " +
                                                   "workers that claim tasks from the queue directory. Used for cluster execution.")]
        public bool UseExternalProvisioner { get; set; }

        private WorkerWrapper[] ConnectedWorkers = null;

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
        internal (List<T> Processed, List<T> Failed) DistributeItems<T>(
            Func<T, int, TaskItem> buildTask,
            Action<T> onSuccess = null,
            Action<T, WorkResult> onFailure = null,
            int pollMs = 500) where T : Movie
        {
            if (Workers != null && Workers.Any())
                throw new Exception(
                    $"The --workers (remote hostname:port) option is not supported by the " +
                    $"filesystem-based distribution path. Use --device_list / --perdevice for " +
                    $"local GPU distribution, or run under Relay for cluster execution.");

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

            IWorkerProvisioner provisioner;
            int target;
            if (UseExternalProvisioner)
            {
                // Cluster mode: an external system (Relay) spawns workers that claim
                // from the queue dir. The manager still runs the Scheduler (heartbeat,
                // orphan sweep, failure processing) but never spawns or counts local
                // processes — so we skip device resolution entirely, which also avoids
                // touching the GPU on a manager node that may have none.
                provisioner = new ExternalProvisioner();
                target = 0;
                Console.WriteLine($"Distributing {tasks.Count} tasks; workers provisioned externally...");
            }
            else
            {
                // Local mode: resolve devices, cap workers to actual item count.
                List<int> devices = (DeviceList == null || !DeviceList.Any())
                    ? Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList()
                    : DeviceList.ToList();
                if (devices.Count <= 0)
                    throw new Exception("No devices found or specified");
                target = Math.Min(InputSeries.Length, devices.Count * ProcessesPerDevice);
                provisioner = new LocalProvisioner(layout.Root, devices.ToArray(), ProcessesPerDevice, logDir: logDir);
                Console.WriteLine($"Distributing {tasks.Count} tasks across up to {target} local worker(s)...");
            }

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
                        bool succeeded = result.Outcome == WorkOutcome.Done;

                        // Standard per-item handling, identical for every ported command.
                        item.LoadMeta();
                        if (succeeded)
                        {
                            item.ProcessingStatus = ProcessingStatus.Processed;
                            item.SaveMeta();
                            processed.Add(item);
                            onSuccess?.Invoke(item);
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
                            if (!succeeded) nFailed++;

                            // On failure: clear the current progress line so the error block
                            // does not append to it, then redraw the updated progress below.
                            if (!succeeded)
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
            var json = new System.Text.Json.Nodes.JsonArray(
                items.Select(m => m.ToMiniJson("particles")).ToArray());
            File.WriteAllText(tmp, json.ToJsonString(
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            File.Move(tmp, path, overwrite: true);
        }

        public virtual WorkerWrapper[] GetWorkers(bool attachDebugger = false)
        {
            if (ConnectedWorkers != null)
                return ConnectedWorkers;

            Console.WriteLine("Connecting to workers...");

            if (Workers == null || !Workers.Any())
            {
                List<int> UsedDevices = (DeviceList == null || !DeviceList.Any()) ?
                                        Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList() :
                                        DeviceList.ToList();
                int NDevices = UsedDevices.Count;

                if (NDevices <= 0)
                    throw new Exception("No devices found or specified");

                int MaxWorkers = Math.Min(InputSeries.Length, UsedDevices.Count * ProcessesPerDevice);
                List<WorkerWrapper> NewWorkers = new List<WorkerWrapper>();
                foreach (var id in UsedDevices)
                {
                    for (int i = 0; i < ProcessesPerDevice; i++)
                    {
                        if (NewWorkers.Count < MaxWorkers)
                        {
                            WorkerWrapper NewWorker = new WorkerWrapper(id,
                                                                        !Helper.IsDebug, 
                                                                        attachDebugger: attachDebugger);
                            NewWorkers.Add(NewWorker);
                        }
                    }
                }

                ConnectedWorkers = NewWorkers.ToArray();
            }
            else
            {
                ConnectedWorkers = new WorkerWrapper[Workers.Count()];

                for (int i = 0; i < Workers.Count(); i++)
                {
                    string[] Parts = Workers.ElementAt(i).Split(new[] { ':' }, StringSplitOptions.RemoveEmptyEntries);
                    string Host = Parts[0];
                    int Port = int.Parse(Parts[1]);

                    ConnectedWorkers[i] = new WorkerWrapper(Host, Port);
                }
            }

            if (Options != null)
            {
                Helper.ForCPU(0, ConnectedWorkers.Length, ConnectedWorkers.Length, null, (i, threadID) =>
                {
                    ConnectedWorkers[i].LoadGainRef(Options.Import.GainPath, 
                                                    Options.Import.GainFlipX, 
                                                    Options.Import.GainFlipY, 
                                                    Options.Import.GainTranspose, 
                                                    Options.Import.DefectsPath);
                }, null);
            }

            Console.WriteLine($"Connected to {ConnectedWorkers.Length} workers");

            return ConnectedWorkers;
        }
    }
}
