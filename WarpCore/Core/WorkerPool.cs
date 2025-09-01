using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Warp;
using Warp.Workers;
using Warp.Workers.WorkerController;
using WarpCore.Core.Processing;
using TaskStatus = Warp.Workers.WorkerController.TaskStatus;

namespace WarpCore.Core
{
    public class WorkerPool : IDisposable
    {
        private readonly ILogger<WorkerPool> _logger;
        private readonly StartupOptions _startupOptions;
        
        private WorkerControllerHost _controllerHost;
        private readonly Dictionary<string, WorkerWrapper> _workers = new();
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

        public IReadOnlyList<WorkerWrapper> GetWorkers()
        {
            lock (_workersLock)
            {
                return _workers.Values.ToList();
            }
        }

        public WorkerWrapper GetWorker(string workerId)
        {
            lock (_workersLock)
            {
                return _workers.TryGetValue(workerId, out var worker) ? worker : null;
            }
        }

        /// <summary>
        /// Thread-safe method to reserve an available worker for a task
        /// </summary>
        public ProcessingTask ReserveWorker(Movie movie, string taskId)
        {
            lock (_workersLock)
            {
                // Find first available worker
                var availableWorker = _workers.Values.FirstOrDefault(w => w.Status == WorkerStatus.Idle);
                if (availableWorker == null)
                    return null;

                // Reserve the worker atomically
                availableWorker.Status = WorkerStatus.Working;
                availableWorker.CurrentTask = taskId;

                // Create and return the task with worker reference
                return new ProcessingTask
                {
                    TaskId = taskId,
                    Movie = movie,
                    WorkerId = availableWorker.WorkerId,
                    Worker = availableWorker,
                    StartedAt = DateTime.UtcNow,
                    Status = TaskStatus.Assigned
                };
            }
        }

        /// <summary>
        /// Thread-safe method to return a worker to the available pool
        /// </summary>
        public void ReturnWorker(string workerId)
        {
            lock (_workersLock)
            {
                if (_workers.TryGetValue(workerId, out var worker))
                {
                    worker.Status = WorkerStatus.Idle;
                    worker.CurrentTask = null;
                    _logger.LogDebug($"Worker {workerId} returned to pool");
                }
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

        /// <summary>
        /// Execute a processing task on a worker
        /// </summary>
        public async Task<bool> ExecuteProcessingTaskAsync(ProcessingTask task, OptionsWarp settings)
        {
            try
            {
                _logger.LogInformation($"Executing task {task.TaskId} for movie {task.Movie.Name} on worker {task.WorkerId}");

                // Get current processing options from settings
                var ctfOptions = settings.GetProcessingMovieCTF();
                var movementOptions = settings.GetProcessingMovieMovement();
                var exportOptions = settings.GetProcessingMovieExport();
                var pickingOptions = settings.GetProcessingBoxNet();

                // Save current state before processing
                task.Movie.SaveMeta();

                // Process steps based on what needs to be done
                bool success = await ExecuteProcessingStepsAsync(task, 
                                                                 settings, 
                                                                 ctfOptions, 
                                                                 movementOptions, 
                                                                 exportOptions, 
                                                                 pickingOptions);

                if (success)
                {
                    // Reload metadata after processing (worker has updated it)
                    task.Movie.LoadMeta();
                    task.Movie.ProcessingStatus = ProcessingStatus.Processed;
                }

                return success;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error executing processing task {task.TaskId} for movie {task.Movie.Name}");
                return false;
            }
        }

        private async Task<bool> ExecuteProcessingStepsAsync(
            ProcessingTask task,
            OptionsWarp settings,
            ProcessingOptionsMovieCTF ctfOptions,
            ProcessingOptionsMovieMovement movementOptions,
            ProcessingOptionsMovieExport exportOptions,
            ProcessingOptionsBoxNet pickingOptions)
        {
            try
            {
                var movie = task.Movie;
                var worker = task.Worker;

                // Determine what processing steps are needed
                bool needsMovement = settings.ProcessMovement && 
                    (movie.OptionsMovement == null || !movie.OptionsMovement.Equals(movementOptions));
                
                bool needsCTF = settings.ProcessCTF && 
                    (movie.OptionsCTF == null || !movie.OptionsCTF.Equals(ctfOptions));
                
                bool needsPicking = settings.ProcessPicking && 
                    (movie.OptionsBoxNet == null || !movie.OptionsBoxNet.Equals(pickingOptions));

                bool needsExport = (exportOptions.DoAverage || exportOptions.DoStack || exportOptions.DoDeconv) &&
                    (movie.OptionsMovieExport == null || !movie.OptionsMovieExport.Equals(exportOptions) || needsMovement);

                // Determine if we need to load the stack (matches original logic)
                bool needStack = needsCTF || needsMovement || needsExport || (needsPicking && pickingOptions.ExportParticles);

                // Load stack data if any processing step needs it
                if (needStack)
                {
                    _logger.LogDebug($"Loading stack data for {movie.Name}");
                    decimal scaleFactor = 1M / (decimal)Math.Pow(2, (double)settings.Import.BinTimes);
                    worker.LoadStack(movie.DataPath, scaleFactor, exportOptions.EERGroupFrames);
                }

                // Execute processing steps in order (matching original desktop implementation)
                if (needsMovement)
                {
                    _logger.LogDebug($"Processing motion correction for {movie.Name}");
                    worker.MovieProcessMovement(movie.Path, movementOptions);
                    movie.LoadMeta(); // Reload metadata after processing
                }

                if (needsCTF)
                {
                    _logger.LogDebug($"Processing CTF correction for {movie.Name}");
                    worker.MovieProcessCTF(movie.Path, ctfOptions);
                    movie.LoadMeta(); // Reload metadata after processing
                }

                if (needsExport)
                {
                    _logger.LogDebug($"Processing export for {movie.Name}");
                    worker.MovieExportMovie(movie.Path, exportOptions);
                    movie.LoadMeta(); // Reload metadata after processing
                }

                if (needsPicking)
                {
                    _logger.LogDebug($"Processing particle picking for {movie.Name}");
                    worker.MoviePickBoxNet(movie.Path, pickingOptions);
                    movie.LoadMeta(); // Reload metadata after processing
                }

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error in processing steps for movie {task.Movie.Name}");
                return false;
            }
        }

        private void OnWorkerRegistered(object sender, WorkerWrapper worker)
        {
            lock (_workersLock)
            {
                _workers[worker.WorkerId] = worker;
                _logger.LogInformation($"Pool: Worker {worker.WorkerId} registered with device ID {worker.DeviceID}");
            }

            WorkerConnected?.Invoke(this, new WorkerEventArgs(worker));
        }

        private void OnWorkerDisconnected(object sender, WorkerWrapper worker)
        {
            WorkerWrapper disconnectedWorker = null;
            
            lock (_workersLock)
            {
                if (_workers.TryGetValue(worker.WorkerId, out disconnectedWorker))
                {
                    _workers.Remove(worker.WorkerId);
                    _logger.LogWarning($"Worker {worker.WorkerId} disconnected");
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
                foreach (var worker in _workers.Values)
                {
                    try
                    {
                        worker.Dispose();
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error disposing WorkerWrapper");
                    }
                }
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