using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Warp;
using Warp.Workers;
using Warp.Workers.WorkerController;
using WarpCore.Core.Processing;

namespace WarpCore.Core
{
    /// <summary>
    /// Central orchestrator that coordinates all aspects of distributed image processing.
    /// Manages the main processing loop, coordinates between file discovery, worker pools,
    /// and task distribution, handles settings changes, and provides processing statistics.
    /// This is the core component that ties the entire system together.
    /// </summary>
    public class ProcessingOrchestrator : IDisposable
    {
        private readonly ILogger<ProcessingOrchestrator> _logger;
        private readonly WorkerPool _workerPool;
        private readonly FileDiscoverer _fileDiscoverer;
        private readonly ChangeTracker _changeTracker;
        private readonly StartupOptions _startupOptions;

        private OptionsWarp _currentSettings;
        private CancellationTokenSource _processingCancellation;
        private Task _processingTask;
        private readonly object _processingLock = new object();

        // Processing components
        private readonly ProcessingQueue _processingQueue;
        private readonly ProcessingTaskDistributor _taskDistributor;
        private readonly SettingsChangeHandler _settingsChangeHandler;

        /// <summary>
        /// Gets a value indicating whether the processing system is currently active.
        /// </summary>
        public bool IsProcessing { get; private set; }

        /// <summary>
        /// Initializes the processing orchestrator with all required dependencies.
        /// Sets up event subscriptions for worker management and file discovery.
        /// </summary>
        /// <param name="logger">Logger for recording orchestration operations</param>
        /// <param name="workerPool">Pool of workers for distributed processing</param>
        /// <param name="fileDiscoverer">Service for discovering new files to process</param>
        /// <param name="changeTracker">Service for tracking processing state changes</param>
        /// <param name="startupOptions">Application startup configuration</param>
        /// <param name="processingQueue">Queue manager for discovered movies</param>
        /// <param name="taskDistributor">Task assignment and distribution logic</param>
        /// <param name="settingsChangeHandler">Handler for processing settings changes</param>
        public ProcessingOrchestrator(
            ILogger<ProcessingOrchestrator> logger,
            WorkerPool workerPool, 
            FileDiscoverer fileDiscoverer,
            ChangeTracker changeTracker,
            StartupOptions startupOptions,
            ProcessingQueue processingQueue,
            ProcessingTaskDistributor taskDistributor,
            SettingsChangeHandler settingsChangeHandler)
        {
            _logger = logger;
            _workerPool = workerPool;
            _fileDiscoverer = fileDiscoverer;
            _changeTracker = changeTracker;
            _startupOptions = startupOptions;
            
            // Use DI-provided processing components
            _processingQueue = processingQueue;
            _taskDistributor = taskDistributor;
            _settingsChangeHandler = settingsChangeHandler;

            _currentSettings = new OptionsWarp();
            
            // Subscribe to worker events for redistribution
            _workerPool.WorkerConnected += OnWorkerConnected;
            _workerPool.WorkerDisconnected += OnWorkerDisconnected;
            
            // Subscribe to file discovery events
            _fileDiscoverer.FileDiscovered += OnFileDiscovered;
        }

        /// <summary>
        /// Starts the processing orchestrator. Initializes file discovery for the data directory
        /// and begins the main processing loop that continuously processes discovered files.
        /// </summary>
        /// <param name="cancellationToken">Token for cancelling the startup operation</param>
        /// <returns>Task representing the asynchronous startup operation</returns>
        public async Task StartProcessingAsync(CancellationToken cancellationToken = default)
        {
            lock (_processingLock)
            {
                if (IsProcessing)
                    return;

                IsProcessing = true;
                _processingCancellation = new CancellationTokenSource();
            }

            _logger.LogInformation("Starting processing orchestrator");

            // Initialize file discovery
            await _fileDiscoverer.InitializeAsync(_startupOptions.DataDirectory, "*.tiff", true);

            // Start the main processing loop
            _processingTask = ProcessingLoopAsync(_processingCancellation.Token);
        }

        /// <summary>
        /// Pauses the processing orchestrator. Stops the main processing loop and waits
        /// for any currently running tasks to complete gracefully.
        /// </summary>
        /// <returns>Task representing the asynchronous pause operation</returns>
        public async Task PauseProcessingAsync()
        {
            lock (_processingLock)
            {
                if (!IsProcessing)
                    return;

                IsProcessing = false;
                _processingCancellation?.Cancel();
            }

            if (_processingTask != null)
            {
                try
                {
                    await _processingTask;
                }
                catch (OperationCanceledException)
                {
                    // Expected when cancelling
                }
            }

            _logger.LogInformation("Processing orchestrator paused");
        }

        /// <summary>
        /// Updates the processing settings and triggers re-evaluation of all movies.
        /// Analyzes the impact of settings changes and may trigger immediate work redistribution
        /// if processing steps were enabled or disabled.
        /// </summary>
        /// <param name="newSettings">New processing settings to apply</param>
        public void UpdateSettings(OptionsWarp newSettings)
        {
            var oldSettings = _currentSettings;
            _currentSettings = newSettings;
            
            _logger.LogInformation("Settings updated, analyzing impact");
            
            // Analyze and apply settings changes
            var impact = _settingsChangeHandler.AnalyzeSettingsChange(oldSettings, newSettings);
            _settingsChangeHandler.ApplySettingsChange(_processingQueue, oldSettings, newSettings, impact);
            
            // Trigger immediate redistribution if needed
            if (_settingsChangeHandler.ShouldTriggerImmediateRedistribution(impact))
            {
                _ = Task.Run(() => RedistributeWorkAsync(oldSettings, newSettings));
            }
        }

        /// <summary>
        /// Gets the current processing settings being used by the orchestrator.
        /// </summary>
        /// <returns>Current processing settings configuration</returns>
        public OptionsWarp GetCurrentSettings()
        {
            return _currentSettings;
        }

        /// <summary>
        /// Calculates and returns comprehensive processing statistics including item counts,
        /// worker status, and processing rates for monitoring and display purposes.
        /// </summary>
        /// <returns>Statistics object containing processing metrics and status information</returns>
        public ProcessingStatistics GetStatistics()
        {
            var workers = _workerPool.GetWorkers();
            var allMovies = _processingQueue.GetAllMovies();
            
            var processedCount = allMovies.Count(m => _currentSettings.GetMovieProcessingStatus(m, false) == Warp.ProcessingStatus.Processed);
            var failedCount = allMovies.Count(m => _currentSettings.GetMovieProcessingStatus(m, false) == Warp.ProcessingStatus.LeaveOut);
            var queuedCount = allMovies.Count(m => 
            {
                var status = _currentSettings.GetMovieProcessingStatus(m, false);
                return status == Warp.ProcessingStatus.Unprocessed || status == Warp.ProcessingStatus.Outdated;
            });

            return new ProcessingStatistics
            {
                TotalItems = allMovies.Count,
                ProcessedItems = processedCount,
                FailedItems = failedCount,
                QueuedItems = queuedCount,
                ActiveWorkers = workers.Count(w => w.Status == WorkerStatus.Idle), // Available workers
                ProcessingRate = CalculateProcessingRate(),
                Status = IsProcessing ? "Running" : "Paused"
            };
        }

        /// <summary>
        /// Main processing loop that continuously monitors for work and coordinates task execution.
        /// Handles stale task cleanup, assigns work to available workers, and manages error recovery.
        /// Runs until cancellation is requested.
        /// </summary>
        /// <param name="cancellationToken">Token for cancelling the processing loop</param>
        /// <returns>Task representing the processing loop execution</returns>
        private async Task ProcessingLoopAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting main processing loop");

            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    // Clean up any stale tasks (timeout after 30 minutes)
                    var staleMovies = _taskDistributor.CleanupStaleTasks(TimeSpan.FromMinutes(30));
                    if (staleMovies.Any())
                    {
                        _logger.LogWarning($"Found {staleMovies.Count} stale tasks, movies will be reprocessed");
                    }

                    // Process items that need processing
                    await ProcessPendingItemsAsync(cancellationToken);

                    // Wait before next iteration
                    await Task.Delay(2000, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    _logger.LogInformation("Processing loop cancelled");
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in processing loop");
                    await Task.Delay(5000, cancellationToken);
                }
            }
        }

        /// <summary>
        /// Processes all pending items that are ready for processing.
        /// Checks for available workers, gets the next batch of movies from the queue,
        /// assigns tasks to workers, and executes them in parallel.
        /// </summary>
        /// <param name="cancellationToken">Token for cancelling the processing operation</param>
        /// <returns>Task representing the pending items processing operation</returns>
        private async Task ProcessPendingItemsAsync(CancellationToken cancellationToken)
        {
            // Count available workers
            var availableWorkerCount = _workerPool.GetWorkers()
                .Count(w => w.Status == WorkerStatus.Idle);

            if (availableWorkerCount == 0)
                return;

            // Get movies that need processing from the queue
            var moviesToProcess = _processingQueue.GetNextBatch(availableWorkerCount, _currentSettings);

            if (!moviesToProcess.Any())
                return;

            _logger.LogInformation($"Processing {moviesToProcess.Count} items with {availableWorkerCount} available workers");

            // Assign tasks to workers (WorkerPool handles reservation)
            var assignedTasks = await _taskDistributor.AssignTasksAsync(moviesToProcess, _currentSettings);

            // Execute tasks in parallel
            var processingTasks = assignedTasks.Select(task => 
                ProcessTaskAsync(task, cancellationToken));

            await Task.WhenAll(processingTasks);
            
            // Update change tracking
            await _changeTracker.RecordChangeAsync();
            await _changeTracker.UpdateProcessedItemsAsync();
        }

        /// <summary>
        /// Processes a single task by executing it on the assigned worker.
        /// Handles task completion and failure scenarios, updating task status accordingly.
        /// </summary>
        /// <param name="task">Processing task to execute</param>
        /// <param name="cancellationToken">Token for cancelling the task execution</param>
        /// <returns>Task representing the processing operation</returns>
        private async Task ProcessTaskAsync(ProcessingTask task, CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogDebug($"Processing task {task.TaskId} for movie {task.Movie.Name}");

                // Execute the processing task via WorkerPool
                bool success = await _workerPool.ExecuteProcessingTaskAsync(task, _currentSettings);

                if (success)
                {
                    _taskDistributor.CompleteTask(task.TaskId);
                    _logger.LogInformation($"Successfully completed processing {task.Movie.Name}");
                }
                else
                {
                    _taskDistributor.FailTask(task.TaskId, "Processing failed");
                    _logger.LogWarning($"Failed to process {task.Movie.Name}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error processing task {task.TaskId} for movie {task.Movie.Name}");
                _taskDistributor.FailTask(task.TaskId, ex.Message);
            }
        }


        /// <summary>
        /// Event handler for when a new worker connects to the system.
        /// Triggers work redistribution to take advantage of the new capacity.
        /// </summary>
        /// <param name="sender">Event sender (WorkerPool)</param>
        /// <param name="e">Event arguments containing worker information</param>
        private void OnWorkerConnected(object sender, WorkerEventArgs e)
        {
            _logger.LogInformation($"Worker {e.Worker.WorkerId} connected, redistributing work");
            _ = Task.Run(() => RedistributeWorkAsync());
        }

        /// <summary>
        /// Event handler for when a worker disconnects from the system.
        /// Reassigns any tasks that were running on the disconnected worker back to the queue.
        /// </summary>
        /// <param name="sender">Event sender (WorkerPool)</param>
        /// <param name="e">Event arguments containing worker information</param>
        private void OnWorkerDisconnected(object sender, WorkerEventArgs e)
        {
            _logger.LogWarning($"Worker {e.Worker.WorkerId} disconnected, reassigning tasks");
            
            // Reassign any tasks that were running on the disconnected worker
            var reassignedMovies = _taskDistributor.ReassignTasksFromWorker(e.Worker.WorkerId);
            if (reassignedMovies.Any())
            {
                _logger.LogInformation($"Reassigned {reassignedMovies.Count} movies from disconnected worker");
            }
        }

        /// <summary>
        /// Event handler for when the file discoverer finds a new movie file.
        /// Creates a Movie object, loads any existing metadata, and adds it to the processing queue.
        /// </summary>
        /// <param name="sender">Event sender (FileDiscoverer)</param>
        /// <param name="e">Event arguments containing file discovery information</param>
        private void OnFileDiscovered(object sender, FileDiscoveredEventArgs e)
        {
            try
            {
                var fileName = Path.GetFileNameWithoutExtension(e.FilePath);
                var relativePath = Path.GetRelativePath(_startupOptions.DataDirectory, e.FilePath);
                var baseName = Path.GetFileNameWithoutExtension(relativePath);
                
                // The first argument (path) should point to where the raw data file would be in the processing folder
                var processingRawPath = Path.Combine(_startupOptions.ProcessingDirectory, Path.GetFileName(e.FilePath));
                
                // Create Movie with proper constructor arguments
                Movie movie = new Movie(processingRawPath,
                                        string.IsNullOrWhiteSpace(_currentSettings.Import.DataFolder) ?
                                            null :
                                            Path.GetDirectoryName(e.FilePath));
                
                // Load existing metadata if it exists
                var processingMetaPath = Path.Combine(_startupOptions.ProcessingDirectory, baseName + ".xml");
                if (File.Exists(processingMetaPath))
                {
                    movie.LoadMeta();
                }
                else
                {
                    movie.ProcessingStatus = Warp.ProcessingStatus.Unprocessed;
                }

                // Add the movie to the processing queue
                _processingQueue.AddMovie(movie);
                _logger.LogDebug($"New file discovered and added to queue: {fileName}");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, $"Error processing discovered file {e.FilePath}");
            }
        }

        /// <summary>
        /// Redistributes work among available workers. Currently, the queue-based approach
        /// handles redistribution automatically, so this method serves as a placeholder
        /// for future more sophisticated redistribution logic.
        /// </summary>
        /// <param name="oldSettings">Previous settings (optional)</param>
        /// <param name="newSettings">New settings (optional)</param>
        /// <returns>Task representing the redistribution operation</returns>
        private async Task RedistributeWorkAsync(OptionsWarp oldSettings = null, OptionsWarp newSettings = null)
        {
            if (!IsProcessing)
                return;

            _logger.LogDebug("Redistributing work among available workers");
            
            // The main processing loop will automatically pick up any work that needs to be done
            // No explicit redistribution is needed since the queue-based approach handles this
            await Task.CompletedTask;
        }

        /// <summary>
        /// Calculates the current processing rate (items per time unit).
        /// Currently returns 0.0 as a placeholder - implementation is pending.
        /// </summary>
        /// <returns>Processing rate in items per unit time</returns>
        private double CalculateProcessingRate()
        {
            // TODO: Implement processing rate calculation based on recent processing history
            return 0.0;
        }

        /// <summary>
        /// Disposes the orchestrator, cleaning up event subscriptions and cancelling
        /// any ongoing processing operations to prevent resource leaks.
        /// </summary>
        public void Dispose()
        {
            // Unsubscribe from events to prevent memory leaks and duplicate handlers
            _workerPool.WorkerConnected -= OnWorkerConnected;
            _workerPool.WorkerDisconnected -= OnWorkerDisconnected;
            _fileDiscoverer.FileDiscovered -= OnFileDiscovered;
            
            // Cancel any ongoing processing
            _processingCancellation?.Cancel();
            _processingCancellation?.Dispose();
        }
    }
}