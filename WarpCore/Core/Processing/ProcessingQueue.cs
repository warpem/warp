using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;
using Warp;

namespace WarpCore.Core.Processing
{
    /// <summary>
    /// Manages a queue of discovered movies for processing. Handles prioritization,
    /// status tracking, and batch retrieval for task assignment. Thread-safe operations
    /// ensure consistent state across concurrent access from multiple components.
    /// </summary>
    public class ProcessingQueue
    {
        private readonly ILogger<ProcessingQueue> _logger;
        private readonly List<Movie> _discoveredMovies = new List<Movie>();
        private readonly object _moviesLock = new object();

        /// <summary>
        /// Initializes a new processing queue.
        /// </summary>
        /// <param name="logger">Logger for recording queue operations</param>
        public ProcessingQueue(ILogger<ProcessingQueue> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Adds a newly discovered movie to the processing queue if it's not already present.
        /// Thread-safe operation that prevents duplicate entries based on movie path.
        /// </summary>
        /// <param name="movie">Movie to add to the processing queue</param>
        public void AddMovie(Movie movie)
        {
            lock (_moviesLock)
            {
                if (!_discoveredMovies.Any(m => m.Path == movie.Path))
                {
                    _discoveredMovies.Add(movie);
                    _logger.LogDebug($"Added movie to queue: {movie.Name}");
                }
            }
        }

        /// <summary>
        /// Gets the next batch of movies that need processing, ordered by priority.
        /// Prioritizes outdated items over unprocessed items, maintaining discovery order
        /// within each priority level. Thread-safe operation.
        /// </summary>
        /// <param name="maxCount">Maximum number of movies to return in the batch</param>
        /// <param name="currentSettings">Current processing settings to determine what needs processing</param>
        /// <param name="excludeAssignments">Set of movie paths currently being processed to exclude from selection</param>
        /// <returns>List of movies ready for processing, up to maxCount items</returns>
        public List<Movie> GetNextBatch(int maxCount, OptionsWarp currentSettings, ISet<string> excludeAssignments = null)
        {
            lock (_moviesLock)
            {
                var needProcessing = _discoveredMovies
                    .Where(movie => NeedsProcessing(movie, currentSettings))
                    .Where(movie => excludeAssignments == null || !excludeAssignments.Contains(movie.Path))
                    .ToList();

                // Sort by priority: outdated items first, then by discovery order
                var prioritized = needProcessing
                    .OrderBy(movie => GetProcessingPriority(movie, currentSettings))
                    .ThenBy(movie => _discoveredMovies.IndexOf(movie))
                    .Take(maxCount)
                    .ToList();

                return prioritized;
            }
        }

        /// <summary>
        /// Gets all movies currently in the queue regardless of processing status.
        /// Thread-safe operation that returns a snapshot of the entire queue.
        /// </summary>
        /// <returns>List containing all movies in the queue</returns>
        public List<Movie> GetAllMovies()
        {
            lock (_moviesLock)
            {
                return _discoveredMovies.ToList();
            }
        }

        /// <summary>
        /// Re-evaluates the processing status of all movies when settings change.
        /// Updates each movie's status based on the new settings and logs any changes.
        /// Thread-safe operation that ensures consistent state after settings updates.
        /// </summary>
        /// <param name="currentSettings">New settings to use for status evaluation</param>
        public void RefreshAllStatuses(OptionsWarp currentSettings)
        {
            lock (_moviesLock)
            {
                foreach (var movie in _discoveredMovies)
                {
                    var oldStatus = movie.ProcessingStatus;
                    var newStatus = currentSettings.GetMovieProcessingStatus(movie, false);
                    
                    if (oldStatus != newStatus)
                    {
                        movie.ProcessingStatus = newStatus;
                        _logger.LogDebug($"Movie {movie.Name} status changed from {oldStatus} to {newStatus}");
                    }
                }
            }
        }

        /// <summary>
        /// Gets processing statistics for the current queue including counts of movies
        /// in different processing states. Thread-safe operation that evaluates each
        /// movie's status against current settings.
        /// </summary>
        /// <param name="currentSettings">Current processing settings for status evaluation</param>
        /// <returns>Processing summary with counts and timestamp</returns>
        public ProcessingSummary GetSummary(OptionsWarp currentSettings)
        {
            lock (_moviesLock)
            {
                var summary = new ProcessingSummary
                {
                    TotalMovies = _discoveredMovies.Count,
                    LastModified = DateTime.UtcNow
                };

                foreach (var movie in _discoveredMovies)
                {
                    var status = currentSettings.GetMovieProcessingStatus(movie, false);
                    switch (status)
                    {
                        case Warp.ProcessingStatus.Processed:
                        case Warp.ProcessingStatus.FilteredOut:
                            summary.ProcessedMovies++;
                            break;
                        case Warp.ProcessingStatus.LeaveOut:
                            summary.FailedMovies++;
                            break;
                        case Warp.ProcessingStatus.Unprocessed:
                        case Warp.ProcessingStatus.Outdated:
                            summary.QueuedMovies++;
                            break;
                    }
                }

                return summary;
            }
        }

        /// <summary>
        /// Removes a movie from the queue by path. Used for cleanup operations
        /// when movies are no longer needed or have been moved/deleted.
        /// Thread-safe operation.
        /// </summary>
        /// <param name="moviePath">Full path of the movie to remove</param>
        /// <returns>True if the movie was found and removed, false otherwise</returns>
        public bool RemoveMovie(string moviePath)
        {
            lock (_moviesLock)
            {
                var movie = _discoveredMovies.FirstOrDefault(m => m.Path == moviePath);
                if (movie != null)
                {
                    _discoveredMovies.Remove(movie);
                    _logger.LogDebug($"Removed movie from queue: {movie.Name}");
                    return true;
                }
                return false;
            }
        }

        /// <summary>
        /// Determines if a movie needs processing based on its current status and manual selection state.
        /// Movies marked as manually unselected are excluded from processing.
        /// </summary>
        /// <param name="movie">Movie to evaluate</param>
        /// <param name="currentSettings">Current processing settings</param>
        /// <returns>True if the movie needs processing, false otherwise</returns>
        private bool NeedsProcessing(Movie movie, OptionsWarp currentSettings)
        {
            if (movie.UnselectManual == true)
                return false;

            // If the movie is already marked as processed in memory, don't reprocess it
            // This prevents infinite loops when options comparison fails due to object equality issues
            if (movie.ProcessingStatus == Warp.ProcessingStatus.Processed)
                return false;

            var status = currentSettings.GetMovieProcessingStatus(movie, false);
            return status == Warp.ProcessingStatus.Unprocessed || status == Warp.ProcessingStatus.Outdated;
        }

        /// <summary>
        /// Gets the processing priority for a movie based on its status.
        /// Outdated items have higher priority than unprocessed items.
        /// </summary>
        /// <param name="movie">Movie to get priority for</param>
        /// <param name="currentSettings">Current processing settings</param>
        /// <returns>Priority value where lower numbers indicate higher priority</returns>
        private int GetProcessingPriority(Movie movie, OptionsWarp currentSettings)
        {
            var status = currentSettings.GetMovieProcessingStatus(movie, false);
            
            // Outdated items have higher priority (lower number) than unprocessed
            return status switch
            {
                Warp.ProcessingStatus.Outdated => 1,
                Warp.ProcessingStatus.Unprocessed => 2,
                _ => 3
            };
        }
    }
}