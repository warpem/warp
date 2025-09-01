using System;
using System.Diagnostics;
using Warp;
using Warp.Workers;

namespace WarpCore.Core
{
    public class WorkerEventArgs : EventArgs
    {
        public WorkerWrapper Worker { get; }

        public WorkerEventArgs(WorkerWrapper worker)
        {
            Worker = worker;
        }
    }

    public class FileDiscoveredEventArgs : EventArgs
    {
        public string FilePath { get; }
        public string FileName { get; }
        public long FileSize { get; }
        public DateTime DiscoveredAt { get; }

        public FileDiscoveredEventArgs(string filePath, string fileName, long fileSize, DateTime discoveredAt)
        {
            FilePath = filePath;
            FileName = fileName;
            FileSize = fileSize;
            DiscoveredAt = discoveredAt;
        }
    }

    public class FileIncubationInfo
    {
        public string FilePath { get; }
        public Stopwatch Timer { get; }
        public long FileSize { get; set; }
        public DateTime LastChecked { get; set; }

        public FileIncubationInfo(string filePath, long fileSize)
        {
            FilePath = filePath;
            FileSize = fileSize;
            Timer = Stopwatch.StartNew();
            LastChecked = DateTime.UtcNow;
        }
    }

    public class ProcessingStatistics
    {
        public int TotalItems { get; set; }
        public int ProcessedItems { get; set; }
        public int FailedItems { get; set; }
        public int QueuedItems { get; set; }
        public int ActiveWorkers { get; set; }
        public double ProcessingRate { get; set; }
        public string Status { get; set; }
    }

    public class ProcessingSummary
    {
        public int TotalMovies { get; set; }
        public int ProcessedMovies { get; set; }
        public int FailedMovies { get; set; }
        public int QueuedMovies { get; set; }
        public DateTime LastModified { get; set; }
    }
}