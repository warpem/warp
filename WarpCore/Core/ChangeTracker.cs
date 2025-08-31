using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Warp;

namespace WarpCore.Core
{
    public class ChangeTracker
    {
        private readonly ILogger<ChangeTracker> _logger;
        private readonly StartupOptions _startupOptions;
        private readonly FileDiscoverer _fileDiscoverer;
        
        private DateTime _lastModified = DateTime.UtcNow;
        private readonly object _timestampLock = new object();

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

        public ChangeTracker(ILogger<ChangeTracker> logger, StartupOptions startupOptions, FileDiscoverer fileDiscoverer)
        {
            _logger = logger;
            _startupOptions = startupOptions;
            _fileDiscoverer = fileDiscoverer;
        }

        public Task RecordChangeAsync()
        {
            lock (_timestampLock)
            {
                _lastModified = DateTime.UtcNow;
            }
            
            _logger.LogDebug($"Change recorded at {_lastModified}");
            return Task.CompletedTask;
        }

        public async Task UpdateProcessedItemsAsync()
        {
            try
            {
                // TODO: Need to get movies from somewhere else since FileDiscoverer is now generic
                var movies = new List<Movie>();
                
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

        public async Task<ProcessingSummary> GetSummaryAsync()
        {
            try
            {
                // TODO: Need to get movies from somewhere else since FileDiscoverer is now generic
                var movies = new List<Movie>();
                var movieList = movies.ToList();
                
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