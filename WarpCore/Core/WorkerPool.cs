using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Warp;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.WorkerController;
using WarpCore.Core.Processing;
using TaskStatus = Warp.Workers.WorkerController.TaskStatus;

namespace WarpCore.Core
{
    /// <summary>
    /// Manages a pool of distributed workers for processing tasks. Handles worker registration,
    /// task assignment with thread-safe reservations, processing step execution, and lifecycle management.
    /// Integrates with WorkerWrapper's static controller for communication with remote workers.
    /// </summary>
    public class WorkerPool : IDisposable
    {
        private readonly ILogger<WorkerPool> _logger;
        private readonly StartupOptions _startupOptions;
        
        private WorkerControllerHost _controllerHost;
        private readonly Dictionary<string, WorkerWrapper> _workers = new();
        private readonly object _workersLock = new object();

        /// <summary>
        /// Event raised when a new worker connects to the pool.
        /// </summary>
        public event EventHandler<WorkerEventArgs> WorkerConnected;
        
        /// <summary>
        /// Event raised when a worker disconnects from the pool.
        /// </summary>
        public event EventHandler<WorkerEventArgs> WorkerDisconnected;

        /// <summary>
        /// Initializes the worker pool and sets up the worker controller for communication.
        /// </summary>
        /// <param name="logger">Logger for recording worker pool operations</param>
        /// <param name="startupOptions">Configuration options including controller port settings</param>
        public WorkerPool(ILogger<WorkerPool> logger, StartupOptions startupOptions)
        {
            _logger = logger;
            _startupOptions = startupOptions;
            
            InitializeController();
        }

        /// <summary>
        /// Initializes the worker controller and subscribes to worker lifecycle events.
        /// Uses WorkerWrapper's shared controller infrastructure for communication.
        /// </summary>
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

        /// <summary>
        /// Gets a read-only list of all registered workers.
        /// Thread-safe operation that returns a snapshot of the current worker state.
        /// </summary>
        /// <returns>List of all registered workers</returns>
        public IReadOnlyList<WorkerWrapper> GetWorkers()
        {
            lock (_workersLock)
            {
                return _workers.Values.ToList();
            }
        }

        /// <summary>
        /// Gets a specific worker by its unique identifier.
        /// Thread-safe operation.
        /// </summary>
        /// <param name="workerId">Unique identifier of the worker to retrieve</param>
        /// <returns>Worker wrapper if found, null otherwise</returns>
        public WorkerWrapper GetWorker(string workerId)
        {
            lock (_workersLock)
            {
                return _workers.TryGetValue(workerId, out var worker) ? worker : null;
            }
        }

        /// <summary>
        /// Atomically reserves an available worker for a processing task.
        /// Thread-safe method that finds an idle worker, marks it as working, and creates a processing task.
        /// </summary>
        /// <param name="movie">Movie to be processed</param>
        /// <param name="taskId">Unique identifier for the processing task</param>
        /// <returns>Processing task with assigned worker, or null if no workers are available</returns>
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
        /// Returns a worker to the available pool after task completion.
        /// Thread-safe method that marks the worker as idle and clears its current task assignment.
        /// </summary>
        /// <param name="workerId">Unique identifier of the worker to return to the pool</param>
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

        /// <summary>
        /// Retrieves execution logs from a specific worker for monitoring and debugging.
        /// Currently returns placeholder logs as full console integration is pending.
        /// </summary>
        /// <param name="workerId">Unique identifier of the worker to get logs from</param>
        /// <returns>List of log messages from the worker</returns>
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
        /// Executes a complete processing task on an assigned worker.
        /// Coordinates all processing steps (motion correction, CTF estimation, picking, export)
        /// based on current settings and updates movie metadata accordingly.
        /// </summary>
        /// <param name="task">Processing task containing movie and worker assignment</param>
        /// <param name="settings">Current processing settings configuration</param>
        /// <returns>True if processing completed successfully, false otherwise</returns>
        public async Task<bool> ExecuteProcessingTaskAsync(ProcessingTask task, OptionsWarp settings)
        {
            try
            {
                bool success = false;

                await Task.Run(async () =>
                {
                    _logger.LogInformation($"Executing task {task.TaskId} for movie {task.Movie.Name} on worker {task.WorkerId}");

                    // Get current processing options from settings
                    var ctfOptions = settings.GetProcessingMovieCTF();
                    var movementOptions = settings.GetProcessingMovieMovement();
                    var exportOptions = settings.GetProcessingMovieExport();
                    var pickingOptions = settings.GetProcessingBoxNet();
                    var particleExportOptions = settings.GetProcessingParticleExport();

                    // Save current state before processing
                    task.Movie.SaveMeta();

                    // Process steps based on what needs to be done
                    success = await ExecuteProcessingStepsAsync(task,
                                                                settings,
                                                                ctfOptions,
                                                                movementOptions,
                                                                exportOptions,
                                                                pickingOptions,
                                                                particleExportOptions);

                    if (success)
                    {
                        // Reload metadata after processing (worker has updated it)
                        task.Movie.LoadMeta();
                        task.Movie.ProcessingStatus = ProcessingStatus.Processed;
                    }
                });
                
                return success;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error executing processing task {task.TaskId} for movie {task.Movie.Name}");
                return false;
            }
        }

        /// <summary>
        /// Executes the individual processing steps for a movie based on current settings.
        /// Determines which steps are needed, loads movie data if required, and executes
        /// motion correction, CTF estimation, export, and particle picking in proper order.
        /// </summary>
        /// <param name="task">Processing task containing movie and worker information</param>
        /// <param name="settings">Overall processing settings</param>
        /// <param name="ctfOptions">CTF processing options</param>
        /// <param name="movementOptions">Motion correction options</param>
        /// <param name="exportOptions">Export processing options</param>
        /// <param name="pickingOptions">Particle picking options</param>
        /// <returns>True if all required steps completed successfully, false otherwise</returns>
        private async Task<bool> ExecuteProcessingStepsAsync(
            ProcessingTask task,
            OptionsWarp settings,
            ProcessingOptionsMovieCTF ctfOptions,
            ProcessingOptionsMovieMovement movementOptions,
            ProcessingOptionsMovieExport exportOptions,
            ProcessingOptionsBoxNet pickingOptions,
            ProcessingOptionsParticleExport particleExportOptions)
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

                bool needsParticleExport = settings.Picking.DoExport;

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
                
                if (needsParticleExport)
                {
                    _logger.LogDebug($"Exporting particles for {movie.Name}");
                    worker.MovieExportParticles(movie.Path, 
                                                particleExportOptions, 
                                                [new float2(0, 0), new float2(1, 2)]);
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

        /// <summary>
        /// Event handler for when a new worker registers with the system.
        /// Adds the worker to the pool and raises the WorkerConnected event.
        /// </summary>
        /// <param name="sender">Event sender (WorkerWrapper static events)</param>
        /// <param name="worker">Worker that registered</param>
        private void OnWorkerRegistered(object sender, WorkerWrapper worker)
        {
            lock (_workersLock)
            {
                _workers[worker.WorkerId] = worker;
                _logger.LogInformation($"Pool: Worker {worker.WorkerId} registered with device ID {worker.DeviceID}");
            }

            WorkerConnected?.Invoke(this, new WorkerEventArgs(worker));
        }

        /// <summary>
        /// Event handler for when a worker disconnects from the system.
        /// Removes the worker from the pool and raises the WorkerDisconnected event.
        /// </summary>
        /// <param name="sender">Event sender (WorkerWrapper static events)</param>
        /// <param name="worker">Worker that disconnected</param>
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

        /// <summary>
        /// Disposes the worker pool, cleaning up all resources and unsubscribing from events.
        /// Safely disposes all registered workers and the controller host to prevent resource leaks.
        /// </summary>
        public void Dispose()
        {
            // Unsubscribe from static WorkerWrapper events to prevent duplicate handlers
            WorkerWrapper.WorkerRegistered -= OnWorkerRegistered;
            WorkerWrapper.WorkerDisconnected -= OnWorkerDisconnected;

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