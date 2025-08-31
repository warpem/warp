using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Warp.Workers;
using Warp.Workers.WorkerController;

namespace WarpCore.Core
{
    public class WorkerPool : IDisposable
    {
        private readonly ILogger<WorkerPool> _logger;
        private readonly StartupOptions _startupOptions;
        
        private WorkerControllerHost _controllerHost;
        private readonly Dictionary<string, WorkerInfo> _workers = new Dictionary<string, WorkerInfo>();
        private readonly Dictionary<string, WorkerWrapper> _workerWrappers = new Dictionary<string, WorkerWrapper>();
        private readonly object _workersLock = new object();

        public event EventHandler<WorkerEventArgs> WorkerConnected;
        public event EventHandler<WorkerEventArgs> WorkerDisconnected;

        public WorkerPool(ILogger<WorkerPool> logger, StartupOptions startupOptions)
        {
            _logger = logger;
            _startupOptions = startupOptions;
            
            InitializeController();
        }

        private void InitializeController()
        {
            try
            {
                // Use WorkerWrapper's shared controller instead of creating our own
                WorkerWrapper.StartControllerOnPort(_startupOptions.ControllerPort);
                
                _logger.LogInformation($"Pool: Worker controller running on port {WorkerWrapper.GetControllerPort()}");

                // Subscribe to WorkerWrapper's static events
                WorkerWrapper.WorkerRegistered += OnWorkerRegistered;
                WorkerWrapper.WorkerDisconnected += OnWorkerDisconnected;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize worker controller");
                throw;
            }
        }

        public IReadOnlyList<WorkerInfo> GetWorkers()
        {
            lock (_workersLock)
            {
                return _workers.Values.ToList();
            }
        }

        public WorkerInfo GetWorker(string workerId)
        {
            lock (_workersLock)
            {
                return _workers.TryGetValue(workerId, out var worker) ? worker : null;
            }
        }

        public async Task<List<string>> GetWorkerLogsAsync(string workerId)
        {
            // Get logs directly from WorkerWrapper's shared controller
            var allWorkers = WorkerWrapper.GetAllWorkers();
            var worker = allWorkers.FirstOrDefault(w => w.WorkerId == workerId);
            
            if (worker != null)
            {
                // For now, return empty logs as console integration needs more work
                return new List<string> { $"Worker {workerId} logs would be available here" };
            }
            
            return new List<string>();
        }

        private void OnWorkerRegistered(object sender, Warp.Workers.WorkerController.WorkerInfo controllerWorkerInfo)
        {
            lock (_workersLock)
            {
                var workerInfo = new WorkerInfo
                {
                    Id = controllerWorkerInfo.WorkerId,
                    DeviceId = controllerWorkerInfo.DeviceId,
                    Status = "Idle",
                    ConnectedAt = DateTime.UtcNow,
                    LastHeartbeat = DateTime.UtcNow,
                    ProcessedItems = 0,
                    CurrentTask = null
                };

                _workers[controllerWorkerInfo.WorkerId] = workerInfo;
                    
                _logger.LogInformation($"Pool: Worker {controllerWorkerInfo.WorkerId} registered with device ID {controllerWorkerInfo.DeviceId}");
            }

            WorkerConnected?.Invoke(this, new WorkerEventArgs(GetWorker(controllerWorkerInfo.WorkerId)));
        }

        private void OnWorkerDisconnected(object sender, Warp.Workers.WorkerController.WorkerInfo controllerWorkerInfo)
        {
            WorkerInfo disconnectedWorker = null;
            
            lock (_workersLock)
            {
                if (_workers.TryGetValue(controllerWorkerInfo.WorkerId, out disconnectedWorker))
                {
                    _workers.Remove(controllerWorkerInfo.WorkerId);
                    
                    // Clean up WorkerWrapper
                    if (_workerWrappers.TryGetValue(controllerWorkerInfo.WorkerId, out var wrapper))
                    {
                        wrapper.Dispose();
                        _workerWrappers.Remove(controllerWorkerInfo.WorkerId);
                    }
                    
                    _logger.LogWarning($"Worker {controllerWorkerInfo.WorkerId} disconnected");
                }
            }

            if (disconnectedWorker != null)
            {
                WorkerDisconnected?.Invoke(this, new WorkerEventArgs(disconnectedWorker));
            }
        }

        public void Dispose()
        {
            lock (_workersLock)
            {
                // Dispose all WorkerWrappers
                foreach (var wrapper in _workerWrappers.Values)
                {
                    try
                    {
                        wrapper.Dispose();
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error disposing WorkerWrapper");
                    }
                }
                _workerWrappers.Clear();
                _workers.Clear();
            }

            // Dispose controller
            try
            {
                _controllerHost?.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error disposing WorkerControllerHost");
            }
        }
    }
}