using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using Warp.Tools;

namespace Warp.Workers.Distribution
{
    /// <summary>
    /// Represents the current status of a work package in the distribution system.
    /// </summary>
    public enum WorkPackageStatus
    {
        /// <summary>
        /// Work package is queued and waiting for assignment to a worker.
        /// </summary>
        Queued,
        
        /// <summary>
        /// Work package has been assigned to a worker and is being executed.
        /// </summary>
        Executing,
        
        /// <summary>
        /// Work package completed successfully.
        /// </summary>
        Completed,
        
        /// <summary>
        /// Work package failed and exceeded maximum retry attempts.
        /// </summary>
        Failed
    }

    /// <summary>
    /// A self-contained collection of commands to be executed sequentially on the same worker
    /// without interruption. Supports retry logic, worker affinity, and lifecycle callbacks.
    /// </summary>
    public class WorkPackage
    {
        /// <summary>
        /// Gets or sets the unique identifier for this work package.
        /// </summary>
        public string Id { get; set; }
        
        /// <summary>
        /// Gets or sets the list of commands to execute sequentially on the same worker.
        /// </summary>
        public List<NamedSerializableObject> Commands { get; set; }
        
        /// <summary>
        /// Gets or sets the priority of this work package. Higher values = higher priority.
        /// </summary>
        public int Priority { get; set; } = 0;
        
        /// <summary>
        /// Gets or sets the preferred worker ID. If null, any available worker can take this package.
        /// </summary>
        public string PreferredWorkerId { get; set; }
        
        /// <summary>
        /// Gets or sets the maximum number of retry attempts if the package fails.
        /// </summary>
        public int MaxRetries { get; set; } = 2;
        
        /// <summary>
        /// Gets or sets the callback invoked when a worker starts executing this package.
        /// Parameters: (package, workerId)
        /// </summary>
        [JsonIgnore]
        public Action<WorkPackage, string> OnStart { get; set; }
        
        /// <summary>
        /// Gets or sets the callback invoked when the package completes successfully.
        /// Parameters: (package)
        /// </summary>
        [JsonIgnore]
        public Action<WorkPackage> OnSuccess { get; set; }
        
        /// <summary>
        /// Gets or sets the callback invoked when the package fails permanently.
        /// Parameters: (package, errorMessage)
        /// </summary>
        [JsonIgnore]
        public Action<WorkPackage, string> OnFailure { get; set; }
        
        // Internal tracking properties
        
        /// <summary>
        /// Gets or sets the current attempt count for this package.
        /// </summary>
        public int AttemptCount { get; set; } = 0;
        
        /// <summary>
        /// Gets or sets the index of the currently executing command.
        /// </summary>
        public int CurrentCommandIndex { get; set; } = 0;
        
        /// <summary>
        /// Gets or sets the timestamp when this package was created.
        /// </summary>
        public DateTime CreatedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the current status of this work package.
        /// </summary>
        public WorkPackageStatus Status { get; set; }
        
        /// <summary>
        /// Gets or sets the ID of the worker currently assigned to this package.
        /// </summary>
        public string AssignedWorkerId { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp when this package was last assigned to a worker.
        /// </summary>
        public DateTime? AssignedAt { get; set; }

        /// <summary>
        /// Initializes a new work package with a unique ID and creation timestamp.
        /// </summary>
        public WorkPackage()
        {
            Id = Guid.NewGuid().ToString();
            Commands = new List<NamedSerializableObject>();
            CreatedAt = DateTime.UtcNow;
            Status = WorkPackageStatus.Queued;
        }
    }

    /// <summary>
    /// Event arguments for queue running low notifications.
    /// </summary>
    public class QueueLowEventArgs : EventArgs
    {
        /// <summary>
        /// Gets the current number of packages in the queue.
        /// </summary>
        public int CurrentQueueSize { get; }
        
        /// <summary>
        /// Gets the target queue size that should be maintained.
        /// </summary>
        public int TargetQueueSize { get; }
        
        /// <summary>
        /// Gets the number of additional packages needed to reach the target.
        /// </summary>
        public int PackagesNeeded { get; }

        /// <summary>
        /// Initializes new queue low event arguments.
        /// </summary>
        /// <param name="currentSize">Current queue size</param>
        /// <param name="targetSize">Target queue size</param>
        public QueueLowEventArgs(int currentSize, int targetSize)
        {
            CurrentQueueSize = currentSize;
            TargetQueueSize = targetSize;
            PackagesNeeded = Math.Max(0, targetSize - currentSize);
        }
    }

    /// <summary>
    /// Request for updating work package progress and status.
    /// </summary>
    public class WorkPackageUpdateRequest
    {
        /// <summary>
        /// Gets or sets the current status of the work package.
        /// </summary>
        public WorkPackageStatus Status { get; set; }
        
        /// <summary>
        /// Gets or sets the index of the currently executing command (0-based).
        /// </summary>
        public int CurrentCommandIndex { get; set; }
        
        /// <summary>
        /// Gets or sets the error message if the package failed.
        /// </summary>
        public string ErrorMessage { get; set; }
        
        /// <summary>
        /// Gets or sets optional progress message for monitoring.
        /// </summary>
        public string ProgressMessage { get; set; }
        
        /// <summary>
        /// Gets or sets optional progress percentage (0.0 to 1.0).
        /// </summary>
        public double? ProgressPercentage { get; set; }
    }
}