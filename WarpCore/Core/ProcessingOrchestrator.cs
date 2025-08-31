using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Warp;
using Warp.Workers;

namespace WarpCore.Core
{
    public class ProcessingOrchestrator
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

        private List<Movie> _discoveredMovies = new List<Movie>();
        private readonly object _moviesLock = new object();

        public bool IsProcessing { get; private set; }

        public ProcessingOrchestrator(
            ILogger<ProcessingOrchestrator> logger,
            WorkerPool workerPool, 
            FileDiscoverer fileDiscoverer,
            ChangeTracker changeTracker,
            StartupOptions startupOptions)
        {
            _logger = logger;
            _workerPool = workerPool;
            _fileDiscoverer = fileDiscoverer;
            _changeTracker = changeTracker;
            _startupOptions = startupOptions;

            _currentSettings = new OptionsWarp();
            
            // Subscribe to worker events for redistribution
            _workerPool.WorkerConnected += OnWorkerConnected;
            _workerPool.WorkerDisconnected += OnWorkerDisconnected;
            
            // Subscribe to file discovery events
            _fileDiscoverer.FileDiscovered += OnFileDiscovered;
        }

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

        public void UpdateSettings(OptionsWarp newSettings)
        {
            var oldSettings = _currentSettings;
            _currentSettings = newSettings;
            
            _logger.LogInformation("Settings updated, triggering redistribution");
            
            // Settings change triggers redistribution of work
            _ = Task.Run(() => RedistributeWorkAsync(oldSettings, newSettings));
        }

        public OptionsWarp GetCurrentSettings()
        {
            return _currentSettings;
        }

        public ProcessingStatistics GetStatistics()
        {
            var workers = _workerPool.GetWorkers();
            
            lock (_moviesLock)
            {
                var processedCount = _discoveredMovies.Count(m => m.ProcessingStatus == ProcessingStatus.Processed);
                var failedCount = _discoveredMovies.Count(m => m.ProcessingStatus == ProcessingStatus.LeaveOut);
                var queuedCount = _discoveredMovies.Count(m => 
                    m.ProcessingStatus == ProcessingStatus.Unprocessed || 
                    m.ProcessingStatus == ProcessingStatus.Outdated);

                return new ProcessingStatistics
                {
                    TotalItems = _discoveredMovies.Count,
                    ProcessedItems = processedCount,
                    FailedItems = failedCount,
                    QueuedItems = queuedCount,
                    ActiveWorkers = workers.Count(w => w.Status == "Active"),
                    ProcessingRate = CalculateProcessingRate(),
                    Status = IsProcessing ? "Running" : "Paused"
                };
            }
        }

        private async Task ProcessingLoopAsync(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    // File discovery happens via events, no need to poll

                    // Process items that need processing
                    await ProcessPendingItemsAsync(cancellationToken);

                    // Wait before next iteration
                    await Task.Delay(5000, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in processing loop");
                    await Task.Delay(10000, cancellationToken);
                }
            }
        }

        private async Task ProcessPendingItemsAsync(CancellationToken cancellationToken)
        {
            var availableWorkers = _workerPool.GetWorkers()
                .Where(w => w.Status == "Idle")
                .ToList();

            if (!availableWorkers.Any())
                return;

            List<Movie> itemsToProcess;
            lock (_moviesLock)
            {
                itemsToProcess = _discoveredMovies
                    .Where(NeedsProcessing)
                    .Take(availableWorkers.Count)
                    .ToList();
            }

            if (!itemsToProcess.Any())
                return;

            _logger.LogInformation($"Processing {itemsToProcess.Count} items with {availableWorkers.Count} workers");

            // Process items in parallel across available workers
            var processingTasks = itemsToProcess.Zip(availableWorkers, (movie, worker) =>
                ProcessMovieAsync(movie, worker.Id, cancellationToken));

            await Task.WhenAll(processingTasks);
            
            // Update change tracking
            await _changeTracker.RecordChangeAsync();
            await _changeTracker.UpdateProcessedItemsAsync();
        }

        private async Task ProcessMovieAsync(Movie movie, string workerId, CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogDebug($"Processing {movie.Name} on worker {workerId}");

                // Get processing options
                var ctfOptions = _currentSettings.GetProcessingMovieCTF();
                var movementOptions = _currentSettings.GetProcessingMovieMovement();
                var exportOptions = _currentSettings.GetProcessingMovieExport();
                var pickingOptions = _currentSettings.GetProcessingBoxNet();

                // Determine what processing steps are needed
                bool needsCTF = _currentSettings.ProcessCTF && (movie.OptionsCTF == null || movie.OptionsCTF != ctfOptions);
                bool needsMovement = _currentSettings.ProcessMovement && (movie.OptionsMovement == null || movie.OptionsMovement != movementOptions);
                bool needsPicking = _currentSettings.ProcessPicking && (movie.OptionsBoxNet == null || movie.OptionsBoxNet != pickingOptions);

                // TODO: Execute processing steps using WorkerWrapper
                // This would call the appropriate worker methods:
                // - Workers[workerId].MovieProcessMovement()
                // - Workers[workerId].MovieProcessCTF()  
                // - Workers[workerId].MovieExportMovie()
                // - Workers[workerId].MovieExportParticles()

                movie.ProcessingStatus = ProcessingStatus.Processed;
                movie.SaveMeta();

                _logger.LogDebug($"Completed processing {movie.Name}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Failed to process {movie.Name}");
                movie.ProcessingStatus = ProcessingStatus.LeaveOut;
                movie.UnselectManual = true;
                movie.SaveMeta();
            }
        }

        private bool NeedsProcessing(Movie movie)
        {
            if (movie.UnselectManual == true)
                return false;

            var status = _currentSettings.GetMovieProcessingStatus(movie, false);
            return status == ProcessingStatus.Unprocessed || status == ProcessingStatus.Outdated;
        }

        private void OnWorkerConnected(object sender, WorkerEventArgs e)
        {
            _logger.LogInformation($"Worker {e.Worker.Id} connected, redistributing work");
            _ = Task.Run(() => RedistributeWorkAsync());
        }

        private void OnWorkerDisconnected(object sender, WorkerEventArgs e)
        {
            _logger.LogWarning($"Worker {e.Worker.Id} disconnected, redistributing work immediately");
            _ = Task.Run(() => RedistributeWorkAsync());
        }

        private void OnFileDiscovered(object sender, FileDiscoveredEventArgs e)
        {
            try
            {
                var fileName = Path.GetFileNameWithoutExtension(e.FilePath);
                var relativePath = Path.GetRelativePath(_startupOptions.DataDirectory, e.FilePath);
                var baseName = Path.GetFileNameWithoutExtension(relativePath);
                var processingPath = Path.Combine(_startupOptions.ProcessingDirectory, baseName + ".xml");

                var movie = new Movie(processingPath);
                
                if (File.Exists(processingPath))
                {
                    movie.LoadMeta();
                }
                else
                {
                    movie.ProcessingStatus = ProcessingStatus.Unprocessed;
                }

                lock (_moviesLock)
                {
                    if (!_discoveredMovies.Any(m => m.Path == movie.Path))
                    {
                        _discoveredMovies.Add(movie);
                        _logger.LogDebug($"New file discovered: {fileName}");
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, $"Error processing discovered file {e.FilePath}");
            }
        }

        private async Task RedistributeWorkAsync(OptionsWarp oldSettings = null, OptionsWarp newSettings = null)
        {
            // Immediate redistribution logic - reassign work that was in progress on disconnected workers
            // or needs reprocessing due to settings changes
            
            if (!IsProcessing)
                return;

            _logger.LogDebug("Redistributing work among available workers");
            
            // This would implement the logic to:
            // 1. Find items that were being processed by disconnected workers
            // 2. Find items that need reprocessing due to settings changes  
            // 3. Reassign them to available workers
        }

        private double CalculateProcessingRate()
        {
            // TODO: Implement processing rate calculation based on recent processing history
            return 0.0;
        }
    }
}