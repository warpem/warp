using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using Warp.Tools;
using Warp.Workers.Distribution;
using ZLinq;

namespace Warp.Workers.WorkerController
{
    public class WorkerControllerService : IDisposable
    {
        private readonly ConcurrentDictionary<string, WorkerInfo> _workers;
        private readonly ConcurrentDictionary<string, List<LogEntry>> _workerConsoleLines;
        private readonly Timer _heartbeatMonitor;
        private volatile bool _disposed;
        private readonly int _maxConsoleLines = 10000;

        // Work distribution
        private readonly WorkDistributor _workDistributor;
        
        public event EventHandler<WorkerInfo> WorkerRegistered;
        public event EventHandler<WorkerInfo> WorkerDisconnected;

        /// <summary>
        /// Gets the work distributor for submitting work packages directly.
        /// </summary>
        public WorkDistributor WorkDistributor => _workDistributor;

        public WorkerControllerService(WorkDistributor workDistributor)
        {
            _workDistributor = workDistributor ?? throw new ArgumentNullException(nameof(workDistributor));
            
            _workers = new ConcurrentDictionary<string, WorkerInfo>();
            _workerConsoleLines = new ConcurrentDictionary<string, List<LogEntry>>();

            // Monitor worker heartbeats every 1 second for immediate failure detection
            _heartbeatMonitor = new Timer(MonitorWorkerHeartbeats, null, TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(1));
        }

        #region Worker Management

        public WorkerRegistrationResponse RegisterWorker(WorkerRegistrationRequest request)
        {
            var worker = new WorkerInfo
            {
                Host = request.Host,
                DeviceId = request.DeviceId,
                Status = WorkerStatus.Idle,
                LastHeartbeat = DateTime.UtcNow
            };

            _workers[worker.WorkerId] = worker;
            
            Console.WriteLine($"Worker {worker.WorkerId} registered from {worker.Host}, GPU #{worker.DeviceId}");
            WorkerRegistered?.Invoke(this, worker);

            return new WorkerRegistrationResponse
            {
                WorkerId = worker.WorkerId,
                Token = worker.Token
            };
        }

        public bool UpdateHeartbeat(string workerId, HeartbeatRequest heartbeat)
        {
            if (!_workers.TryGetValue(workerId, out var worker))
                return false;

            worker.LastHeartbeat = DateTime.UtcNow;
            worker.Status = heartbeat.Status;
            worker.CurrentWorkPackageId = heartbeat.CurrentWorkPackageId;

            return true;
        }

        public IEnumerable<WorkerInfo> GetActiveWorkers()
        {
            return _workers.Values.Where(w => w.Status != WorkerStatus.Offline).ToArray();
        }

        private void MonitorWorkerHeartbeats(object state)
        {
            if (_disposed) return;

            var cutoff = DateTime.UtcNow.AddSeconds(-1);
            var disconnectedWorkers = _workers.Values
                .Where(w => w.LastHeartbeat < cutoff && w.Status != WorkerStatus.Offline)
                .ToList();

            foreach (var worker in disconnectedWorkers)
            {
                Console.WriteLine($"[HEARTBEAT] Worker {worker.WorkerId} missed heartbeat (last: {worker.LastHeartbeat}, cutoff: {cutoff}), marking as offline");
                worker.Status = WorkerStatus.Offline;
                
                // Report worker death to work distributor for work package handling
                _workDistributor.ReportWorkerDeath(worker.WorkerId);

                WorkerDisconnected?.Invoke(this, worker);
                
                // Clean up console lines for offline worker
                _workerConsoleLines.TryRemove(worker.WorkerId, out _);
            }
        }

        #endregion

        #region Work Package Management

        public PollResponse PollForTask(string workerId, PollRequest pollRequest = null)
        {
            if (!_workers.TryGetValue(workerId, out var worker))
                return new PollResponse(); // Worker not registered

            worker.LastHeartbeat = DateTime.UtcNow;

            // Store console lines if provided
            if (pollRequest?.ConsoleLines != null && pollRequest.ConsoleLines.Count > 0)
            {
                var workerConsole = _workerConsoleLines.GetOrAdd(workerId, _ => new List<LogEntry>());
                
                lock (workerConsole)
                {
                    workerConsole.AddRange(pollRequest.ConsoleLines);
                    
                    // Limit stored lines to prevent unbounded growth
                    if (workerConsole.Count > _maxConsoleLines)
                    {
                        int excess = workerConsole.Count - _maxConsoleLines;
                        workerConsole.RemoveRange(0, excess);
                    }
                }
            }

            // Get work package from distributor
            var workPackage = _workDistributor.GetNextWork(workerId);
            if (workPackage != null)
            {
                worker.Status = WorkerStatus.Working;
                worker.CurrentWorkPackageId = workPackage.Id;
                Console.WriteLine($"Work package {workPackage.Id} assigned to worker {workerId}");
                
                return new PollResponse 
                { 
                    WorkPackage = workPackage,
                    NextPollDelayMs = 100 // Poll quickly after receiving work
                };
            }

            // No work available, set worker to idle
            if (worker.Status == WorkerStatus.Working && string.IsNullOrEmpty(worker.CurrentWorkPackageId))
            {
                worker.Status = WorkerStatus.Idle;
            }

            return new PollResponse 
            { 
                NextPollDelayMs = 500 // Standard polling interval
            };
        }

        public bool UpdateWorkPackageStatus(string workerId, string workPackageId, WorkPackageUpdateRequest update)
        {
            if (!_workers.TryGetValue(workerId, out var worker))
                return false;

            worker.LastHeartbeat = DateTime.UtcNow;
            
            // Update work package status through distributor
            _workDistributor.UpdateProgress(workPackageId, update);
            
            // Update worker state based on package status
            switch (update.Status)
            {
                case WorkPackageStatus.Completed:
                case WorkPackageStatus.Failed:
                    worker.Status = WorkerStatus.Idle;
                    worker.CurrentWorkPackageId = null;
                    Console.WriteLine($"Work package {workPackageId} {update.Status.ToString().ToLower()} on worker {workerId}");
                    break;
                    
                case WorkPackageStatus.Executing:
                    worker.Status = WorkerStatus.Working;
                    break;
            }

            return true;
        }

        #endregion

        #region Console Management

        public List<LogEntry> GetWorkerConsoleLines(string workerId)
        {
            if (!_workerConsoleLines.TryGetValue(workerId, out var consoleLines))
                return new List<LogEntry>();
                
            lock (consoleLines)
            {
                return new List<LogEntry>(consoleLines);
            }
        }
        
        public List<LogEntry> GetWorkerConsoleLines(string workerId, int count)
        {
            var allLines = GetWorkerConsoleLines(workerId);
            return allLines.Skip(Math.Max(0, allLines.Count - count)).ToList();
        }
        
        public List<LogEntry> GetWorkerConsoleLines(string workerId, int start, int count)
        {
            var allLines = GetWorkerConsoleLines(workerId);
            return allLines.Skip(start).Take(count).ToList();
        }

        #endregion

        #region Status and Statistics

        public object GetStatus()
        {
            var workers = _workers.Values.ToList();
            var distributorStats = _workDistributor.GetStatistics();

            return new
            {
                Workers = new
                {
                    Total = workers.Count,
                    Online = workers.Count(w => w.Status != WorkerStatus.Offline),
                    Idle = workers.Count(w => w.Status == WorkerStatus.Idle),
                    Working = workers.Count(w => w.Status == WorkerStatus.Working),
                    Failed = workers.Count(w => w.Status == WorkerStatus.Failed)
                },
                WorkPackages = new
                {
                    Queued = distributorStats.QueuedPackages,
                    Active = distributorStats.ActivePackages,
                    TargetQueueSize = distributorStats.TargetQueueSize
                },
                Timestamp = DateTime.UtcNow
            };
        }

        #endregion

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _heartbeatMonitor?.Dispose();

            // Notify all workers to stop
            foreach (var worker in _workers.Values)
            {
                worker.Status = WorkerStatus.Offline;
            }
        }
    }
}