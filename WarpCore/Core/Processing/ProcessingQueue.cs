using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;
using Warp;

namespace WarpCore.Core.Processing
{
    public class ProcessingQueue
    {
        private readonly ILogger<ProcessingQueue> _logger;
        private readonly List<Movie> _discoveredMovies = new List<Movie>();
        private readonly object _moviesLock = new object();

        public ProcessingQueue(ILogger<ProcessingQueue> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Add a newly discovered movie to the queue
        /// </summary>
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
        /// Get the next batch of movies that need processing, in priority order
        /// Priority: discovery order, but outdated items before unprocessed items
        /// </summary>
        public List<Movie> GetNextBatch(int maxCount, OptionsWarp currentSettings)
        {
            lock (_moviesLock)
            {
                var needProcessing = _discoveredMovies
                    .Where(movie => NeedsProcessing(movie, currentSettings))
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
        /// Get all movies in the queue
        /// </summary>
        public List<Movie> GetAllMovies()
        {
            lock (_moviesLock)
            {
                return _discoveredMovies.ToList();
            }
        }

        /// <summary>
        /// Re-evaluate all movies when settings change
        /// </summary>
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
        /// Get processing statistics for the current queue
        /// </summary>
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
        /// Remove a movie from the queue (for cleanup)
        /// </summary>
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

        private bool NeedsProcessing(Movie movie, OptionsWarp currentSettings)
        {
            if (movie.UnselectManual == true)
                return false;

            var status = currentSettings.GetMovieProcessingStatus(movie, false);
            return status == Warp.ProcessingStatus.Unprocessed || status == Warp.ProcessingStatus.Outdated;
        }

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