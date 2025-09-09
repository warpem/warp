using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Warp;
using WarpCore.Core.Processing;

namespace WarpCore.Core
{
    /// <summary>
    /// Tracks changes to processing state and maintains JSON export files for processed and failed items.
    /// Provides timestamps for change detection and summary information for monitoring.
    /// Used by clients to determine when processing results have been updated.
    /// </summary>
    public class ChangeTracker
    {
        private readonly ILogger<ChangeTracker> _logger;
        private readonly StartupOptions _startupOptions;
        private readonly FileDiscoverer _fileDiscoverer;
        private readonly ProcessingQueue _processingQueue;
        
        private DateTime _lastModified = DateTime.UtcNow;
        private readonly object _timestampLock = new object();

        /// <summary>
        /// Gets the timestamp of the last recorded change to processing state.
        /// Thread-safe property used by clients for change detection.
        /// </summary>
        public DateTime LastModified
        {
            get
            {
                lock (_timestampLock)
                {
                    return _lastModified;
                }
            }
        }

        /// <summary>
        /// Initializes the change tracker with required dependencies.
        /// </summary>
        /// <param name="logger">Logger for recording change tracking operations</param>
        /// <param name="startupOptions">Application startup configuration containing directory paths</param>
        /// <param name="fileDiscoverer">File discoverer service (currently unused but available for future features)</param>
        /// <param name="processingQueue">Processing queue containing movie data</param>
        public ChangeTracker(ILogger<ChangeTracker> logger, StartupOptions startupOptions, FileDiscoverer fileDiscoverer, ProcessingQueue processingQueue)
        {
            _logger = logger;
            _startupOptions = startupOptions;
            _fileDiscoverer = fileDiscoverer;
            _processingQueue = processingQueue;
        }

        /// <summary>
        /// Records that a change has occurred in the processing system by updating the last modified timestamp.
        /// This is used by clients to detect when processing results have been updated.
        /// </summary>
        /// <returns>Task representing the change recording operation</returns>
        public Task RecordChangeAsync()
        {
            lock (_timestampLock)
            {
                _lastModified = DateTime.UtcNow;
            }
            
            _logger.LogDebug($"Change recorded at {_lastModified}");
            return Task.CompletedTask;
        }

        /// <summary>
        /// Updates JSON files containing processed and failed items for external consumption.
        /// Creates processed_items.json and failed_items.json files in the processing directory.
        /// Currently has placeholder implementation - movie data source needs to be connected.
        /// </summary>
        /// <returns>Task representing the JSON file update operation</returns>
        public async Task UpdateProcessedItemsAsync()
        {
            try
            {
                var movies = _processingQueue.GetAllMovies();
                
                var processedItems = movies.Where(m => m.ProcessingStatus == ProcessingStatus.Processed).ToList();
                var failedItems = movies.Where(m => m.ProcessingStatus == ProcessingStatus.LeaveOut || m.UnselectManual == true).ToList();

                // Write processed_items.json
                var processedItemsPath = Path.Combine(_startupOptions.ProcessingDirectory, "processed_items.json");
                await WriteItemsJsonAsync(processedItemsPath, processedItems);

                // Write failed_items.json if there are any failures
                if (failedItems.Any())
                {
                    var failedItemsPath = Path.Combine(_startupOptions.ProcessingDirectory, "failed_items.json");
                    await WriteItemsJsonAsync(failedItemsPath, failedItems);
                }

                await RecordChangeAsync();
                
                _logger.LogDebug($"Updated processed items: {processedItems.Count} processed, {failedItems.Count} failed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating processed items JSON files");
            }
        }

        /// <summary>
        /// Gets a summary of processing results including counts of processed, failed, and queued movies.
        /// Currently has placeholder implementation - movie data source needs to be connected.
        /// </summary>
        /// <returns>Processing summary with item counts and last modified timestamp</returns>
        public async Task<ProcessingSummary> GetSummaryAsync()
        {
            try
            {
                var movieList = _processingQueue.GetAllMovies();
                
                var processedCount = movieList.Count(m => m.ProcessingStatus == ProcessingStatus.Processed);
                var failedCount = movieList.Count(m => m.ProcessingStatus == ProcessingStatus.LeaveOut || m.UnselectManual == true);
                var queuedCount = movieList.Count(m => 
                    m.ProcessingStatus == ProcessingStatus.Unprocessed || 
                    m.ProcessingStatus == ProcessingStatus.Outdated);

                return new ProcessingSummary
                {
                    TotalMovies = movieList.Count,
                    ProcessedMovies = processedCount,
                    FailedMovies = failedCount,
                    QueuedMovies = queuedCount,
                    LastModified = LastModified
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting processing summary");
                return new ProcessingSummary
                {
                    LastModified = LastModified
                };
            }
        }

        /// <summary>
        /// Writes a collection of movie items to a JSON file using atomic write operations.
        /// Uses a temporary file and move operation to prevent partial writes.
        /// </summary>
        /// <typeparam name="T">Type of movie items to write</typeparam>
        /// <param name="filePath">Destination path for the JSON file</param>
        /// <param name="items">Collection of movie items to serialize</param>
        /// <returns>Task representing the JSON write operation</returns>
        private async Task WriteItemsJsonAsync<T>(string filePath, IEnumerable<T> items) where T : Movie
        {
            try
            {
                var itemsJson = new JsonArray(items.Select(item => item.ToMiniJson("particles")).ToArray());
                
                var options = new JsonSerializerOptions 
                { 
                    WriteIndented = true 
                };
                
                var jsonString = itemsJson.ToJsonString(options);
                
                // Ensure directory exists before writing
                var directory = Path.GetDirectoryName(filePath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }
                
                // Write to temp file first, then move to avoid partial writes
                var tempPath = filePath + ".tmp";
                await File.WriteAllTextAsync(tempPath, jsonString);
                File.Move(tempPath, filePath, true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error writing JSON file {filePath}");
            }
        }
    }
}