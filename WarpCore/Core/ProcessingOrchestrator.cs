using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Warp;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.WorkerController;
using Warp.Workers.Distribution;
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
        private readonly WorkerControllerService _workerControllerService;
        private readonly FileDiscoverer _fileDiscoverer;
        private readonly ChangeTracker _changeTracker;
        private readonly StartupOptions _startupOptions;

        private OptionsWarp _currentSettings;
        private CancellationTokenSource _processingCancellation;
        private Task _processingTask;
        private readonly object _processingLock = new object();

        // Processing components - new system
        private readonly ProcessingQueue _processingQueue;
        private readonly WorkDistributor _workDistributor;
        private readonly SettingsChangeHandler _settingsChangeHandler;
        
        
        // Debounced change tracking
        private readonly Timer _changeTrackingTimer;
        private volatile bool _hasChangesToWrite;
        private readonly object _changeTrackingLock = new object();

        /// <summary>
        /// Gets a value indicating whether the processing system is currently active.
        /// </summary>
        public bool IsProcessing { get; private set; }

        /// <summary>
        /// Initializes the processing orchestrator with all required dependencies.
        /// Sets up event subscriptions for worker management, file discovery, and work distribution.
        /// </summary>
        /// <param name="logger">Logger for recording orchestration operations</param>
        /// <param name="workerControllerService">Worker controller service for distributed processing</param>
        /// <param name="fileDiscoverer">Service for discovering new files to process</param>
        /// <param name="changeTracker">Service for tracking processing state changes</param>
        /// <param name="startupOptions">Application startup configuration</param>
        /// <param name="processingQueue">Queue manager for discovered movies</param>
        /// <param name="settingsChangeHandler">Handler for processing settings changes</param>
        public ProcessingOrchestrator(
            ILogger<ProcessingOrchestrator> logger,
            WorkerControllerService workerControllerService, 
            FileDiscoverer fileDiscoverer,
            ChangeTracker changeTracker,
            StartupOptions startupOptions,
            ProcessingQueue processingQueue,
            SettingsChangeHandler settingsChangeHandler)
        {
            _logger = logger;
            _workerControllerService = workerControllerService;
            _fileDiscoverer = fileDiscoverer;
            _changeTracker = changeTracker;
            _startupOptions = startupOptions;
            
            // Use DI-provided processing components
            _processingQueue = processingQueue;
            _workDistributor = WorkDistribution.Instance;
            _settingsChangeHandler = settingsChangeHandler;

            _currentSettings = new OptionsWarp();
            
            // Initialize debounced change tracking (10 second debounce)
            _changeTrackingTimer = new Timer(WriteChangesCallback, null, Timeout.Infinite, Timeout.Infinite);
            
            // Subscribe to work distributor events
            _workDistributor.QueueRunningLow += OnQueueRunningLow;
            
            // Subscribe to worker events for redistribution
            _workerControllerService.WorkerRegistered += OnWorkerRegistered;
            _workerControllerService.WorkerDisconnected += OnWorkerDisconnected;
            
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
            await _fileDiscoverer.InitializeAsync(_startupOptions.DataDirectory, _currentSettings.Import.Extension, true);

            // Set work distributor target based on available workers
            var workerCount = _workerControllerService.GetActiveWorkers().Count();
            _workDistributor.SetQueueTarget(Math.Max(4 * workerCount, 10)); // At least 10 packages

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
            
            // Check if file pattern changed and reconfigure file discovery
            if (oldSettings.Import.Extension != newSettings.Import.Extension)
            {
                _logger.LogInformation($"File extension changed from {oldSettings.Import.Extension} to {newSettings.Import.Extension}, reconfiguring file discovery");
                _fileDiscoverer.ChangeConfiguration(_startupOptions.DataDirectory, newSettings.Import.Extension, true);
            }
            
            // Analyze and apply settings changes
            var impact = _settingsChangeHandler.AnalyzeSettingsChange(oldSettings, newSettings);
            _settingsChangeHandler.ApplySettingsChange(_processingQueue, oldSettings, newSettings, impact);
            
            // Purge work distributor queue if processing steps changed
            if (_settingsChangeHandler.ShouldTriggerImmediateRedistribution(impact))
            {
                _logger.LogInformation("Settings change detected, purging work distributor queue");
                _workDistributor.PurgeQueue(); // Clear all pending work packages
                
                // Queue will be refilled automatically via QueueRunningLow event
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
            var workers = _workerControllerService.GetActiveWorkers();
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
                IdleWorkers = workers.Count(w => w.Status == WorkerStatus.Idle), // Available workers
                ProcessingRate = CalculateProcessingRate(),
                Status = IsProcessing ? "Running" : "Paused"
            };
        }

        /// <summary>
        /// Main processing loop that continuously monitors the processing system.
        /// Refreshes processing queue statuses to maintain accurate state information.
        /// Work distribution is handled automatically by the WorkDistributor.
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

                    // Monitor processing status (work distribution is automatic via WorkDistributor)
                    await ProcessPendingItemsAsync(cancellationToken);

                    // Refresh processing queue statuses after completing tasks
                    // This ensures movies that just finished processing are no longer considered for reprocessing
                    _processingQueue.RefreshAllStatuses(_currentSettings);

                    // Wait before next iteration
                    await Task.Delay(50, cancellationToken);
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
        /// Work distribution is now handled automatically by the WorkDistributor
        /// via QueueRunningLow events and WorkPackage submission.
        /// </summary>
        /// <param name="cancellationToken">Token for cancelling the processing operation</param>
        /// <returns>Task representing the pending items processing operation</returns>
        private async Task ProcessPendingItemsAsync(CancellationToken cancellationToken)
        {
            // Work distribution is now handled automatically by the WorkDistributor via QueueRunningLow events
            // This method is kept for any future direct processing needs
            await Task.CompletedTask;
        }


        /// <summary>
        /// Event handler for when the work distributor queue is running low.
        /// Submits new work packages from the processing queue to maintain optimal throughput.
        /// </summary>
        /// <param name="sender">Event sender (WorkDistributor)</param>
        /// <param name="e">Event arguments containing queue size information</param>
        private void OnQueueRunningLow(object sender, QueueLowEventArgs e)
        {
            try
            {
                var moviesToProcess = _processingQueue.GetNextBatch(e.PackagesNeeded, _currentSettings);
                
                foreach (var movie in moviesToProcess)
                {
                    var workPackage = CreateMovieWorkPackage(movie, _currentSettings);
                    _workDistributor.SubmitWorkPackage(workPackage);
                }
                
                if (moviesToProcess.Any())
                {
                    _logger.LogDebug($"Submitted {moviesToProcess.Count} work packages to distributor");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling queue running low event");
            }
        }

        /// <summary>
        /// Creates a work package for processing a single movie with all required steps.
        /// </summary>
        /// <param name="movie">Movie to create work package for</param>
        /// <param name="settings">Current processing settings</param>
        /// <returns>Work package containing all required processing commands</returns>
        private WorkPackage CreateMovieWorkPackage(Movie movie, OptionsWarp settings)
        {
            var commands = new List<NamedSerializableObject>();
            
            // Determine what processing steps are needed and build command sequence
            bool needsMovement = settings.ProcessMovement && 
                (movie.OptionsMovement == null || !movie.OptionsMovement.Equals(settings.GetProcessingMovieMovement()));
            
            bool needsCTF = settings.ProcessCTF && 
                (movie.OptionsCTF == null || !movie.OptionsCTF.Equals(settings.GetProcessingMovieCTF()));
            
            bool needsPicking = settings.ProcessPicking && 
                (movie.OptionsBoxNet == null || !movie.OptionsBoxNet.Equals(settings.GetProcessingBoxNet()));

            bool needsExport = (settings.GetProcessingMovieExport().DoAverage || 
                               settings.GetProcessingMovieExport().DoStack || 
                               settings.GetProcessingMovieExport().DoDeconv) &&
                (movie.OptionsMovieExport == null || 
                 !movie.OptionsMovieExport.Equals(settings.GetProcessingMovieExport()) || 
                 needsMovement);

            // Determine if we need to load the stack
            bool needStack = needsCTF || needsMovement || needsExport || 
                            (needsPicking && settings.GetProcessingBoxNet().ExportParticles);

            // Build command sequence
            if (needStack)
            {
                decimal scaleFactor = 1M / (decimal)Math.Pow(2, (double)settings.Import.BinTimes);
                commands.Add(new NamedSerializableObject("LoadStack", 
                    movie.DataPath, 
                    scaleFactor, 
                    settings.GetProcessingMovieExport().EERGroupFrames, 
                    true // CorrectGain
                ));
            }

            if (needsMovement)
            {
                commands.Add(new NamedSerializableObject("MovieProcessMovement", 
                    movie.Path, 
                    settings.GetProcessingMovieMovement()
                ));
            }

            if (needsCTF)
            {
                commands.Add(new NamedSerializableObject("MovieProcessCTF", 
                    movie.Path, 
                    settings.GetProcessingMovieCTF()
                ));
            }

            if (needsExport)
            {
                commands.Add(new NamedSerializableObject("MovieExportMovie", 
                    movie.Path, 
                    settings.GetProcessingMovieExport()
                ));
            }

            if (needsPicking)
            {
                commands.Add(new NamedSerializableObject("MoviePickBoxNet", 
                    movie.Path, 
                    settings.GetProcessingBoxNet()
                ));
            }

            if (settings.Picking.DoExport)
            {
                commands.Add(new NamedSerializableObject("MovieExportParticles", 
                    movie.Path, 
                    settings.GetProcessingParticleExport(),
                    new[] { new float2(0, 0), new float2(1, 2) }
                ));
            }

            var workPackage = new WorkPackage
            {
                Commands = commands,
                MaxRetries = 2,
                OnStart = (pkg, workerId) => 
                {
                    _logger.LogInformation($"Movie {movie.Name} processing started on worker {workerId}");
                },
                OnSuccess = (pkg) => 
                {
                    // Reload metadata to pick up processing options set by the worker
                    movie.LoadMeta();
                    ScheduleDebouncedChangeTracking();
                    _logger.LogInformation($"Movie {movie.Name} processing completed successfully");
                },
                OnFailure = (pkg, error) => 
                {
                    movie.ProcessingStatus = Warp.ProcessingStatus.LeaveOut;
                    movie.UnselectManual = true;
                    movie.SaveMeta();
                    _logger.LogWarning($"Movie {movie.Name} processing failed: {error}");
                }
            };

            return workPackage;
        }

        /// <summary>
        /// Event handler for when a new worker connects to the system.
        /// Triggers work redistribution to take advantage of the new capacity.
        /// </summary>
        /// <param name="sender">Event sender (WorkerControllerService)</param>
        /// <param name="e">Event arguments containing worker information</param>
        private void OnWorkerRegistered(object sender, WorkerInfo e)
        {
            _logger.LogInformation($"Worker {e.WorkerId} connected, adjusting queue target");
            
            // Adjust queue target based on new worker count
            var workerCount = _workerControllerService.GetActiveWorkers().Count();
            _workDistributor.SetQueueTarget(Math.Max(4 * workerCount, 10));
        }

        /// <summary>
        /// Event handler for when a worker disconnects from the system.
        /// Reassigns any tasks that were running on the disconnected worker back to the queue.
        /// </summary>
        /// <param name="sender">Event sender (WorkerControllerService)</param>
        /// <param name="e">Event arguments containing worker information</param>
        private void OnWorkerDisconnected(object sender, WorkerInfo e)
        {
            _logger.LogWarning($"Worker {e.WorkerId} disconnected, adjusting queue target");
            
            // Work distributor automatically handles work package reassignment
            // Just need to adjust queue target for remaining workers
            var workerCount = _workerControllerService.GetActiveWorkers().Count();
            _workDistributor.SetQueueTarget(Math.Max(4 * workerCount, 10));
            
            // Work distributor automatically handles work package reassignment for disconnected workers
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
        /// Schedules change tracking to be written with a 10-second debounce.
        /// This prevents excessive I/O when many items complete processing in quick succession.
        /// </summary>
        private void ScheduleDebouncedChangeTracking()
        {
            lock (_changeTrackingLock)
            {
                _hasChangesToWrite = true;
                // Reset the timer - this implements the debounce behavior
                // Timer will fire 10 seconds after the LAST change
                _changeTrackingTimer.Change(TimeSpan.FromSeconds(10), Timeout.InfiniteTimeSpan);
            }
        }

        /// <summary>
        /// Timer callback that performs the actual change tracking write operations.
        /// Only executes if there are pending changes to write.
        /// </summary>
        private void WriteChangesCallback(object state)
        {
            lock (_changeTrackingLock)
            {
                if (!_hasChangesToWrite)
                    return;

                _hasChangesToWrite = false;
            }

            // Perform the actual I/O outside the lock to avoid blocking
            try
            {
                _changeTracker.RecordChangeAsync().Wait();
                _changeTracker.UpdateProcessedItemsAsync().Wait();
                _logger.LogDebug("Debounced change tracking completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error writing debounced change tracking");
            }
        }

        /// <summary>
        /// Disposes the orchestrator, cleaning up event subscriptions and cancelling
        /// any ongoing processing operations to prevent resource leaks.
        /// </summary>
        public void Dispose()
        {
            // Unsubscribe from events to prevent memory leaks and duplicate handlers
            _workDistributor.QueueRunningLow -= OnQueueRunningLow;
            _workerControllerService.WorkerRegistered -= OnWorkerRegistered;
            _workerControllerService.WorkerDisconnected -= OnWorkerDisconnected;
            _fileDiscoverer.FileDiscovered -= OnFileDiscovered;
            
            // Cancel any ongoing processing
            _processingCancellation?.Cancel();
            _processingCancellation?.Dispose();
            
            // Flush any pending change tracking before disposal
            if (_hasChangesToWrite)
            {
                try
                {
                    _changeTracker.RecordChangeAsync().Wait(TimeSpan.FromSeconds(5));
                    _changeTracker.UpdateProcessedItemsAsync().Wait(TimeSpan.FromSeconds(5));
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error flushing final change tracking during disposal");
                }
            }
            
            // Dispose the debounced change tracking timer
            _changeTrackingTimer?.Dispose();
        }
    }
}