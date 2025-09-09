using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;
using ZLinq;

namespace Warp.Workers.Distribution;

/// <summary>
/// Manages distribution of work packages to available workers with support for priorities,
/// worker affinity, retry logic, and queue size management. Thread-safe operations ensure
/// consistent state across concurrent access from multiple components.
/// </summary>
public class WorkDistributor : IDisposable
{
    private readonly ILogger<WorkDistributor> _logger;
    private readonly object _queueLock = new object();
    
    // Priority queue: higher priority packages are retrieved first
    private readonly SortedDictionary<int, Queue<WorkPackage>> _priorityQueue = new SortedDictionary<int, Queue<WorkPackage>>();
    
    // Active packages currently being executed by workers
    private readonly ConcurrentDictionary<string, WorkPackage> _activePackages = new ConcurrentDictionary<string, WorkPackage>();
    
    // Track which packages are assigned to which workers for cleanup
    private readonly ConcurrentDictionary<string, HashSet<string>> _workerAssignments = new ConcurrentDictionary<string, HashSet<string>>();
    
    // Queue management
    private int _targetQueueSize = 0;
    private volatile bool _disposed = false;
    
    /// <summary>
    /// Event raised when the queue size drops below the target threshold.
    /// Listeners should submit more work packages to maintain optimal throughput.
    /// </summary>
    public event EventHandler<QueueLowEventArgs> QueueRunningLow;

    /// <summary>
    /// Initializes a new work distributor.
    /// </summary>
    /// <param name="logger">Logger for recording distribution operations</param>
    public WorkDistributor(ILogger<WorkDistributor> logger = null)
    {
        _logger = logger;
    }

    /// <summary>
    /// Submits a work package to the distribution queue. The package will be assigned
    /// to the next available worker based on priority and worker affinity preferences.
    /// Thread-safe operation.
    /// </summary>
    /// <param name="package">Work package to submit for execution</param>
    /// <returns>The unique ID of the submitted work package</returns>
    /// <exception cref="ArgumentNullException">Thrown when package is null</exception>
    /// <exception cref="ObjectDisposedException">Thrown when distributor is disposed</exception>
    public string SubmitWorkPackage(WorkPackage package)
    {
        if (package == null)
            throw new ArgumentNullException(nameof(package));
            
        if (_disposed)
            throw new ObjectDisposedException(nameof(WorkDistributor));

        lock (_queueLock)
        {
            // Reset package state for submission
            package.Status = WorkPackageStatus.Queued;
            package.AssignedWorkerId = null;
            package.AssignedAt = null;
            package.CurrentCommandIndex = 0;

            // Add to priority queue
            if (!_priorityQueue.ContainsKey(package.Priority))
                _priorityQueue[package.Priority] = new Queue<WorkPackage>();

            _priorityQueue[package.Priority].Enqueue(package);
            
            _logger?.LogDebug($"Work package {package.Id} submitted with priority {package.Priority}");
        }

        return package.Id;
    }

    /// <summary>
    /// Gets the next available work package for a specific worker. Considers worker affinity,
    /// priorities, and queue order. Returns null if no suitable work is available.
    /// Thread-safe operation.
    /// </summary>
    /// <param name="workerId">Unique identifier of the requesting worker</param>
    /// <returns>Next work package for the worker, or null if none available</returns>
    /// <exception cref="ArgumentNullException">Thrown when workerId is null or empty</exception>
    /// <exception cref="ObjectDisposedException">Thrown when distributor is disposed</exception>
    public WorkPackage GetNextWork(string workerId)
    {
        if (string.IsNullOrEmpty(workerId))
            throw new ArgumentNullException(nameof(workerId));
            
        if (_disposed)
            throw new ObjectDisposedException(nameof(WorkDistributor));

        WorkPackage assignedPackage = null;

        lock (_queueLock)
        {
            // Find suitable work package (highest priority first)
            foreach (var priority in _priorityQueue.Keys.OrderByDescending(p => p).ToArray())
            {
                var queue = _priorityQueue[priority];
                if (queue.Count == 0)
                    continue;

                // Look for packages with worker affinity first
                var affinityPackage = queue.FirstOrDefault(p => p.PreferredWorkerId == workerId);
                if (affinityPackage != null)
                {
                    // Remove from queue and assign
                    var tempQueue = new Queue<WorkPackage>();
                    while (queue.Count > 0)
                    {
                        var pkg = queue.Dequeue();
                        if (pkg.Id != affinityPackage.Id)
                            tempQueue.Enqueue(pkg);
                    }
                    while (tempQueue.Count > 0)
                        queue.Enqueue(tempQueue.Dequeue());

                    assignedPackage = affinityPackage;
                    break;
                }

                // Look for packages without worker preference
                var generalPackage = queue.FirstOrDefault(p => p.PreferredWorkerId == null);
                if (generalPackage != null)
                {
                    // Remove from queue and assign
                    var tempQueue = new Queue<WorkPackage>();
                    while (queue.Count > 0)
                    {
                        var pkg = queue.Dequeue();
                        if (pkg.Id != generalPackage.Id)
                            tempQueue.Enqueue(pkg);
                    }
                    while (tempQueue.Count > 0)
                        queue.Enqueue(tempQueue.Dequeue());

                    assignedPackage = generalPackage;
                    break;
                }
            }

            // Clean up empty priority levels
            var emptyPriorities = _priorityQueue.Where(kvp => kvp.Value.Count == 0).Select(kvp => kvp.Key).ToList();
            foreach (var emptyPriority in emptyPriorities)
                _priorityQueue.Remove(emptyPriority);

            // Assign the package if found
            if (assignedPackage != null)
            {
                assignedPackage.Status = WorkPackageStatus.Executing;
                assignedPackage.AssignedWorkerId = workerId;
                assignedPackage.AssignedAt = DateTime.UtcNow;
                assignedPackage.AttemptCount++;

                _activePackages[assignedPackage.Id] = assignedPackage;
                
                // Track worker assignment
                _workerAssignments.AddOrUpdate(workerId, 
                    new HashSet<string> { assignedPackage.Id },
                    (key, existing) => { existing.Add(assignedPackage.Id); return existing; });

                _logger?.LogDebug($"Work package {assignedPackage.Id} assigned to worker {workerId} (attempt {assignedPackage.AttemptCount})");
                
                // Invoke OnStart callback
                try
                {
                    assignedPackage.OnStart?.Invoke(assignedPackage, workerId);
                }
                catch (Exception ex)
                {
                    _logger?.LogWarning(ex, $"OnStart callback failed for package {assignedPackage.Id}");
                }
            }

            // Check if we need more work packages
            CheckQueueLevel();
        }

        return assignedPackage;
    }

    /// <summary>
    /// Updates the progress and status of a work package. Handles completion, failure,
    /// and retry logic based on the package configuration. Thread-safe operation.
    /// </summary>
    /// <param name="packageId">Unique identifier of the work package</param>
    /// <param name="update">Progress update information</param>
    /// <exception cref="ArgumentNullException">Thrown when packageId or update is null</exception>
    public void UpdateProgress(string packageId, WorkPackageUpdateRequest update)
    {
        if (string.IsNullOrEmpty(packageId))
            throw new ArgumentNullException(nameof(packageId));
        if (update == null)
            throw new ArgumentNullException(nameof(update));

        if (!_activePackages.TryGetValue(packageId, out var package))
        {
            _logger?.LogWarning($"Received update for unknown package {packageId}");
            return;
        }

        // Update package state
        package.CurrentCommandIndex = update.CurrentCommandIndex;
        
        if (!string.IsNullOrEmpty(update.ProgressMessage))
            _logger?.LogDebug($"Package {packageId} progress: {update.ProgressMessage}");

        switch (update.Status)
        {
            case WorkPackageStatus.Executing:
                // Just a progress update
                break;

            case WorkPackageStatus.Completed:
                CompletePackage(package);
                break;

            case WorkPackageStatus.Failed:
                HandlePackageFailure(package, update.ErrorMessage);
                break;
        }
    }

    /// <summary>
    /// Reports that a worker has died or disconnected. All active packages assigned to
    /// the worker will be requeued for execution by other workers. Thread-safe operation.
    /// </summary>
    /// <param name="workerId">Unique identifier of the dead worker</param>
    public void ReportWorkerDeath(string workerId)
    {
        if (string.IsNullOrEmpty(workerId))
            return;

        if (!_workerAssignments.TryRemove(workerId, out var assignedPackageIds))
            return;

        _logger?.LogWarning($"Worker {workerId} died, requeuing {assignedPackageIds.Count} packages");

        foreach (var packageId in assignedPackageIds)
        {
            if (_activePackages.TryRemove(packageId, out var package))
            {
                // Reset package for requeuing
                package.Status = WorkPackageStatus.Queued;
                package.AssignedWorkerId = null;
                package.AssignedAt = null;
                package.CurrentCommandIndex = 0;

                // Requeue the package
                lock (_queueLock)
                {
                    if (!_priorityQueue.ContainsKey(package.Priority))
                        _priorityQueue[package.Priority] = new Queue<WorkPackage>();

                    _priorityQueue[package.Priority].Enqueue(package);
                    _logger?.LogDebug($"Requeued package {packageId} due to worker death");
                }
            }
        }
    }

    /// <summary>
    /// Removes all packages from the queue that match the specified condition.
    /// Used for purging work when settings change or system state changes.
    /// Thread-safe operation.
    /// </summary>
    /// <param name="condition">Predicate to determine which packages to remove. If null, all packages are removed.</param>
    /// <returns>Number of packages that were purged</returns>
    public int PurgeQueue(Predicate<WorkPackage> condition = null)
    {
        int purgedCount = 0;

        lock (_queueLock)
        {
            if (condition == null)
            {
                // Purge all packages
                foreach (var queue in _priorityQueue.Values)
                    purgedCount += queue.Count;
                
                _priorityQueue.Clear();
            }
            else
            {
                // Purge packages matching condition
                var priorities = _priorityQueue.Keys.ToList();
                foreach (var priority in priorities)
                {
                    var queue = _priorityQueue[priority];
                    var newQueue = new Queue<WorkPackage>();
                    
                    while (queue.Count > 0)
                    {
                        var package = queue.Dequeue();
                        if (condition(package))
                        {
                            purgedCount++;
                        }
                        else
                        {
                            newQueue.Enqueue(package);
                        }
                    }
                    
                    if (newQueue.Count > 0)
                        _priorityQueue[priority] = newQueue;
                    else
                        _priorityQueue.Remove(priority);
                }
            }
        }

        if (purgedCount > 0)
            _logger?.LogInformation($"Purged {purgedCount} packages from queue");

        return purgedCount;
    }

    /// <summary>
    /// Sets the target queue size for optimal throughput. When the queue drops below
    /// this threshold, the QueueRunningLow event will be raised.
    /// </summary>
    /// <param name="targetSize">Target number of packages to maintain in queue</param>
    public void SetQueueTarget(int targetSize)
    {
        _targetQueueSize = Math.Max(0, targetSize);
        _logger?.LogDebug($"Queue target set to {_targetQueueSize} packages");
        
        // Check current level immediately
        lock (_queueLock)
        {
            CheckQueueLevel();
        }
    }

    /// <summary>
    /// Gets the current number of packages in the queue.
    /// </summary>
    /// <returns>Total number of queued packages across all priority levels</returns>
    public int GetQueueSize()
    {
        lock (_queueLock)
        {
            return _priorityQueue.Values.Sum(q => q.Count);
        }
    }

    /// <summary>
    /// Gets the current number of packages being actively executed.
    /// </summary>
    /// <returns>Number of packages currently assigned to workers</returns>
    public int GetActivePackageCount()
    {
        return _activePackages.Count;
    }

    /// <summary>
    /// Gets distribution statistics for monitoring and debugging.
    /// </summary>
    /// <returns>Statistics object containing current queue and execution metrics</returns>
    public WorkDistributorStats GetStatistics()
    {
        lock (_queueLock)
        {
            return new WorkDistributorStats
            {
                QueuedPackages = GetQueueSize(),
                ActivePackages = GetActivePackageCount(),
                TargetQueueSize = _targetQueueSize,
                PriorityLevels = _priorityQueue.Keys.Count,
                WorkersWithAssignments = _workerAssignments.Count
            };
        }
    }

    private void CompletePackage(WorkPackage package)
    {
        // Remove from active tracking
        _activePackages.TryRemove(package.Id, out _);
        
        // Remove from worker assignments
        if (!string.IsNullOrEmpty(package.AssignedWorkerId) && 
            _workerAssignments.TryGetValue(package.AssignedWorkerId, out var assignments))
        {
            assignments.Remove(package.Id);
            if (assignments.Count == 0)
                _workerAssignments.TryRemove(package.AssignedWorkerId, out _);
        }

        package.Status = WorkPackageStatus.Completed;
        _logger?.LogInformation($"Work package {package.Id} completed successfully");

        // Invoke OnSuccess callback
        try
        {
            package.OnSuccess?.Invoke(package);
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, $"OnSuccess callback failed for package {package.Id}");
        }
    }

    private void HandlePackageFailure(WorkPackage package, string errorMessage)
    {
        _logger?.LogWarning($"Work package {package.Id} failed: {errorMessage}");

        if (package.AttemptCount < package.MaxRetries)
        {
            // Requeue for retry
            _activePackages.TryRemove(package.Id, out _);
            
            // Remove from worker assignments
            if (!string.IsNullOrEmpty(package.AssignedWorkerId) && 
                _workerAssignments.TryGetValue(package.AssignedWorkerId, out var assignments))
            {
                assignments.Remove(package.Id);
                if (assignments.Count == 0)
                    _workerAssignments.TryRemove(package.AssignedWorkerId, out _);
            }

            // Reset for retry
            package.Status = WorkPackageStatus.Queued;
            package.AssignedWorkerId = null;
            package.AssignedAt = null;
            package.CurrentCommandIndex = 0;

            lock (_queueLock)
            {
                if (!_priorityQueue.ContainsKey(package.Priority))
                    _priorityQueue[package.Priority] = new Queue<WorkPackage>();

                _priorityQueue[package.Priority].Enqueue(package);
                _logger?.LogInformation($"Requeued package {package.Id} for retry (attempt {package.AttemptCount}/{package.MaxRetries})");
            }
        }
        else
        {
            // Permanent failure
            CompletePackageAsFailure(package, errorMessage);
        }
    }

    private void CompletePackageAsFailure(WorkPackage package, string errorMessage)
    {
        // Remove from active tracking
        _activePackages.TryRemove(package.Id, out _);
        
        // Remove from worker assignments
        if (!string.IsNullOrEmpty(package.AssignedWorkerId) && 
            _workerAssignments.TryGetValue(package.AssignedWorkerId, out var assignments))
        {
            assignments.Remove(package.Id);
            if (assignments.Count == 0)
                _workerAssignments.TryRemove(package.AssignedWorkerId, out _);
        }

        package.Status = WorkPackageStatus.Failed;
        _logger?.LogError($"Work package {package.Id} permanently failed after {package.AttemptCount} attempts: {errorMessage}");

        // Invoke OnFailure callback
        try
        {
            package.OnFailure?.Invoke(package, errorMessage);
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, $"OnFailure callback failed for package {package.Id}");
        }
    }

    private void CheckQueueLevel()
    {
        if (_targetQueueSize <= 0)
            return;

        int currentSize = _priorityQueue.Values.Sum(q => q.Count);
        if (currentSize < _targetQueueSize)
        {
            var eventArgs = new QueueLowEventArgs(currentSize, _targetQueueSize);
            
            try
            {
                QueueRunningLow?.Invoke(this, eventArgs);
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "QueueRunningLow event handler failed");
            }
        }
    }

    /// <summary>
    /// Disposes the work distributor, cleaning up all resources and cancelling any pending work.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
            
        _disposed = true;

        // Clear all queues and active packages
        lock (_queueLock)
        {
            _priorityQueue.Clear();
        }
        
        _activePackages.Clear();
        _workerAssignments.Clear();
        
        _logger?.LogInformation("Work distributor disposed");
    }
}

/// <summary>
/// Statistics about the current state of work distribution.
/// </summary>
public class WorkDistributorStats
{
    /// <summary>
    /// Gets or sets the number of packages currently in the queue awaiting assignment.
    /// </summary>
    public int QueuedPackages { get; set; }

    /// <summary>
    /// Gets or sets the number of packages currently being executed by workers.
    /// </summary>
    public int ActivePackages { get; set; }

    /// <summary>
    /// Gets or sets the target queue size for optimal throughput.
    /// </summary>
    public int TargetQueueSize { get; set; }

    /// <summary>
    /// Gets or sets the number of distinct priority levels currently in use.
    /// </summary>
    public int PriorityLevels { get; set; }

    /// <summary>
    /// Gets or sets the number of workers that currently have package assignments.
    /// </summary>
    public int WorkersWithAssignments { get; set; }
}