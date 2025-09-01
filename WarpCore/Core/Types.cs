using System;
using System.Diagnostics;
using Warp;
using Warp.Workers;

namespace WarpCore.Core
{
    /// <summary>
    /// Event arguments for worker-related events such as connection and disconnection.
    /// Contains information about the worker that triggered the event.
    /// </summary>
    public class WorkerEventArgs : EventArgs
    {
        /// <summary>
        /// Gets the worker wrapper that triggered the event.
        /// </summary>
        public WorkerWrapper Worker { get; }

        /// <summary>
        /// Initializes event arguments with the worker that triggered the event.
        /// </summary>
        /// <param name="worker">Worker that triggered the event</param>
        public WorkerEventArgs(WorkerWrapper worker)
        {
            Worker = worker;
        }
    }

    /// <summary>
    /// Event arguments for file discovery events. Contains information about
    /// newly discovered files that are ready for processing.
    /// </summary>
    public class FileDiscoveredEventArgs : EventArgs
    {
        /// <summary>
        /// Gets the full path to the discovered file.
        /// </summary>
        public string FilePath { get; }
        
        /// <summary>
        /// Gets the filename without path information.
        /// </summary>
        public string FileName { get; }
        
        /// <summary>
        /// Gets the size of the discovered file in bytes.
        /// </summary>
        public long FileSize { get; }
        
        /// <summary>
        /// Gets the timestamp when the file was discovered and deemed stable.
        /// </summary>
        public DateTime DiscoveredAt { get; }

        /// <summary>
        /// Initializes event arguments with file discovery information.
        /// </summary>
        /// <param name="filePath">Full path to the discovered file</param>
        /// <param name="fileName">Name of the discovered file</param>
        /// <param name="fileSize">Size of the file in bytes</param>
        /// <param name="discoveredAt">Timestamp when file was discovered</param>
        public FileDiscoveredEventArgs(string filePath, string fileName, long fileSize, DateTime discoveredAt)
        {
            FilePath = filePath;
            FileName = fileName;
            FileSize = fileSize;
            DiscoveredAt = discoveredAt;
        }
    }

    /// <summary>
    /// Tracks information about a file during the incubation period to ensure
    /// it is stable (not being written to) before processing begins.
    /// </summary>
    public class FileIncubationInfo
    {
        /// <summary>
        /// Gets the full path to the file being incubated.
        /// </summary>
        public string FilePath { get; }
        
        /// <summary>
        /// Gets the timer used to track how long the file has been incubating.
        /// </summary>
        public Stopwatch Timer { get; }
        
        /// <summary>
        /// Gets or sets the current file size, used to detect if the file is still being written to.
        /// </summary>
        public long FileSize { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp when the file was last checked for stability.
        /// </summary>
        public DateTime LastChecked { get; set; }

        /// <summary>
        /// Initializes file incubation tracking for a newly discovered file.
        /// </summary>
        /// <param name="filePath">Full path to the file to incubate</param>
        /// <param name="fileSize">Initial size of the file</param>
        public FileIncubationInfo(string filePath, long fileSize)
        {
            FilePath = filePath;
            FileSize = fileSize;
            Timer = Stopwatch.StartNew();
            LastChecked = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Contains comprehensive processing statistics for monitoring system performance
    /// and providing status information to users and external systems.
    /// </summary>
    public class ProcessingStatistics
    {
        /// <summary>
        /// Gets or sets the total number of items (movies) discovered by the system.
        /// </summary>
        public int TotalItems { get; set; }
        
        /// <summary>
        /// Gets or sets the number of items that have been successfully processed.
        /// </summary>
        public int ProcessedItems { get; set; }
        
        /// <summary>
        /// Gets or sets the number of items that failed processing or were manually excluded.
        /// </summary>
        public int FailedItems { get; set; }
        
        /// <summary>
        /// Gets or sets the number of items waiting to be processed or reprocessed.
        /// </summary>
        public int QueuedItems { get; set; }
        
        /// <summary>
        /// Gets or sets the number of workers currently available for processing tasks.
        /// </summary>
        public int ActiveWorkers { get; set; }
        
        /// <summary>
        /// Gets or sets the current processing rate (items per time unit).
        /// </summary>
        public double ProcessingRate { get; set; }
        
        /// <summary>
        /// Gets or sets the overall status of the processing system (e.g., "Running", "Paused").
        /// </summary>
        public string Status { get; set; }
    }

    /// <summary>
    /// Contains summary information about processing results for reporting and monitoring.
    /// Provides a simplified view of processing state for clients.
    /// </summary>
    public class ProcessingSummary
    {
        /// <summary>
        /// Gets or sets the total number of movies discovered.
        /// </summary>
        public int TotalMovies { get; set; }
        
        /// <summary>
        /// Gets or sets the number of movies that have been successfully processed.
        /// </summary>
        public int ProcessedMovies { get; set; }
        
        /// <summary>
        /// Gets or sets the number of movies that failed processing or were excluded.
        /// </summary>
        public int FailedMovies { get; set; }
        
        /// <summary>
        /// Gets or sets the number of movies waiting in the processing queue.
        /// </summary>
        public int QueuedMovies { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp when processing state was last modified.
        /// </summary>
        public DateTime LastModified { get; set; }
    }
}