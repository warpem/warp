using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warp.Tools;
using Warp.Workers.Queue;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Transport-agnostic description of how a run should provision its workers, plus the
    /// provisioner-selection logic itself. Lifted out of WarpTools' DistributedOptions so
    /// callers that are not WarpTools commands — MTools verbs, MCore — can reuse the same
    /// three provisioning modes (cluster, external, local) without duplicating them or
    /// depending on CommandLineParser option classes.
    ///
    /// Callers map their own CLI options onto these fields; nothing here parses arguments.
    /// </summary>
    public class WorkerPoolSettings
    {
        /// <summary>GPU IDs to use. Null or empty means every device in the system.</summary>
        public int[] DeviceList { get; set; } = null;

        /// <summary>Worker processes per GPU. Raise to improve utilization on big GPUs.</summary>
        public int ProcessesPerDevice { get; set; } = 1;

        /// <summary>
        /// Don't spawn workers; an external system (e.g. Relay) provisions them and they
        /// claim tasks from the queue directory. Mutually exclusive with cluster mode.
        /// </summary>
        public bool UseExternalProvisioner { get; set; } = false;

        /// <summary>Path to a batch-scheduler submission-script template. Its presence selects cluster mode.</summary>
        public string ClusterScript { get; set; } = null;

        /// <summary>Path to the cluster queue-definition JSON. Required with <see cref="ClusterScript"/>.</summary>
        public string ClusterConfig { get; set; } = null;

        /// <summary>Cluster mode: number of worker jobs to submit.</summary>
        public int PoolSize { get; set; } = 0;

        /// <summary>Cluster mode: key=value pairs substituted into the submission template.</summary>
        public IEnumerable<string> ClusterVars { get; set; } = null;

        /// <summary>Worker executable name, resolved against AppContext.BaseDirectory.</summary>
        public string WorkerExeName { get; set; } = "WarpWorker2";

        /// <summary>Run workers in mock mode (no GPU, commands log only). Local mode only.</summary>
        public bool Mock { get; set; } = false;

        /// <summary>
        /// Suppress in-place progress-line rewrites. Set when the consumer is a log parser
        /// (Relay) rather than a terminal.
        /// </summary>
        public bool StrictFormatting { get; set; } = false;

        /// <summary>
        /// Select and construct the worker provisioner for this run: cluster mode
        /// (<see cref="ClusterScript"/>), external mode (<see cref="UseExternalProvisioner"/>),
        /// or local mode (default). Sets <paramref name="target"/> to the desired live
        /// worker count.
        /// </summary>
        public IWorkerProvisioner CreateProvisioner(
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
                string workerExe = Path.Combine(AppContext.BaseDirectory, WorkerExeName);
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
            return new LocalProvisioner(layout.Root, devices.ToArray(), ProcessesPerDevice,
                                        mock: Mock, workerExeName: WorkerExeName, logDir: logDir);
        }
    }
}
