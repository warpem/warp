using CommandLine;
using System;
using System.Collections.Generic;
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

        private WorkerWrapper[] ConnectedWorkers = null;

        public override void Evaluate()
        {
            base.Evaluate();
        }

        /// <summary>
        /// Filesystem work-distribution path: build one task per input item, schedule
        /// them via LocalProvisioner + Scheduler, and block until all are terminal.
        ///
        /// The queue directory is taken from --task_dir if provided, otherwise defaults
        /// to a 'tasks' subdirectory inside OutputProcessing. <paramref name="buildTask"/>
        /// receives the path-corrected movie and its index and returns the TaskItem to
        /// enqueue. <paramref name="onItemResult"/> is called on the polling thread as
        /// each task resolves — update ProcessingStatus, call SaveMeta, and fire
        /// ItemSnapshotWriter.Record here.
        /// </summary>
        internal (List<T> Processed, List<T> Failed) DistributeItems<T>(
            Func<T, int, TaskItem> buildTask,
            Action<T, WorkResult> onItemResult,
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
                tasks.Add(task);
                taskIdToItem[task.TaskId] = m;
            }

            // Resolve devices, cap workers to actual item count.
            List<int> devices = (DeviceList == null || !DeviceList.Any())
                ? Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList()
                : DeviceList.ToList();
            if (devices.Count <= 0)
                throw new Exception("No devices found or specified");
            int target = Math.Min(InputSeries.Length, devices.Count * ProcessesPerDevice);

            var provisioner = new LocalProvisioner(layout.Root, devices.ToArray(), ProcessesPerDevice);
            var scheduler = new Scheduler(layout, queue, provisioner, target);

            Console.WriteLine($"Distributing {tasks.Count} tasks across up to {target} local worker(s)...");

            // Enqueue ALL tasks before starting the scheduler thread so workers always
            // find work in pending/ on their first claim attempt. If tasks are enqueued
            // after the scheduler starts, workers may poll an empty queue and exit before
            // any tasks land.
            pool.Enqueue(tasks);

            var processed = new List<T>();
            var failed = new List<T>();

            var schedThread = new Thread(() => scheduler.RunToDrain()) { IsBackground = true };
            schedThread.Start();

            try
            {
                pool.Distribute(tasks,
                    onResult: result =>
                    {
                        T item = taskIdToItem[result.TaskId];
                        onItemResult(item, result);

                        // Accumulate for return value (onItemResult may also record these,
                        // but we track here so callers don't have to manage two lists).
                        if (result.Outcome == WorkOutcome.Done)
                            lock (processed) processed.Add(item);
                        else
                            lock (failed) failed.Add(item);
                    },
                    pollMs: pollMs);
            }
            finally
            {
                provisioner.Shutdown();
            }

            return (processed, failed);
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
