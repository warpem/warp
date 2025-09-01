using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using Warp.Tools;
using ZLinq;

namespace Warp.Workers.WorkerController
{
    public class WorkerControllerService : IDisposable
    {
        private readonly ConcurrentDictionary<string, WorkerInfo> _workers;
        private readonly ConcurrentQueue<TaskInfo> _taskQueue;
        private readonly ConcurrentDictionary<string, TaskInfo> _activeTasks;
        private readonly ConcurrentDictionary<string, List<LogEntry>> _workerConsoleLines;
        private readonly Timer _heartbeatMonitor;
        private readonly Timer _taskMonitor;
        private volatile bool _disposed;
        private readonly int _maxConsoleLines = 10000;

        private readonly object _assignmentLock = new object();
        
        public event EventHandler<WorkerInfo> WorkerRegistered;
        public event EventHandler<WorkerInfo> WorkerDisconnected;
        public event EventHandler<TaskInfo> TaskCompleted;
        public event EventHandler<TaskInfo> TaskFailed;

        public WorkerControllerService()
        {
            _workers = new ConcurrentDictionary<string, WorkerInfo>();
            _taskQueue = new ConcurrentQueue<TaskInfo>();
            _activeTasks = new ConcurrentDictionary<string, TaskInfo>();
            _workerConsoleLines = new ConcurrentDictionary<string, List<LogEntry>>();

            // Monitor worker heartbeats every 30 seconds
            _heartbeatMonitor = new Timer(MonitorWorkerHeartbeats, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
            
            // Monitor task timeouts every 60 seconds
            _taskMonitor = new Timer(MonitorTaskTimeouts, null, TimeSpan.FromSeconds(60), TimeSpan.FromSeconds(60));
        }

        #region Worker Management

        public WorkerRegistrationResponse RegisterWorker(WorkerRegistrationRequest request)
        {
            var worker = new WorkerInfo
            {
                Host = request.Host,
                DeviceId = request.DeviceId,
                FreeMemoryMB = request.FreeMemoryMB,
                Status = WorkerStatus.Idle
            };

            _workers[worker.WorkerId] = worker;
            
            Console.WriteLine($"Worker {worker.WorkerId} registered from {worker.Host}, GPU #{worker.DeviceId}");
            Console.WriteLine($"WorkerControllerService: FIRING EVENT, subscribers: {WorkerRegistered?.GetInvocationList().Length}");
            WorkerRegistered?.Invoke(this, worker);

            return new WorkerRegistrationResponse
            {
                WorkerId = worker.WorkerId,
                Token = worker.Token
            };
        }

        public bool UpdateHeartbeat(string workerId, HeartbeatRequest heartbeat)
        {
            if (!_workers.TryGetValue(workerId, out var worker))
                return false;

            worker.LastHeartbeat = DateTime.UtcNow;
            worker.Status = heartbeat.Status;
            worker.FreeMemoryMB = heartbeat.FreeMemoryMB;
            worker.CurrentTaskId = heartbeat.CurrentTaskId;

            return true;
        }

        public IEnumerable<WorkerInfo> GetActiveWorkers()
        {
            return _workers.Values.Where(w => w.Status != WorkerStatus.Offline).ToArray();
        }

        private void MonitorWorkerHeartbeats(object state)
        {
            if (_disposed) return;

            var cutoff = DateTime.UtcNow.AddMinutes(-2);
            var disconnectedWorkers = _workers.Values
                .Where(w => w.LastHeartbeat < cutoff && w.Status != WorkerStatus.Offline)
                .ToList();

            foreach (var worker in disconnectedWorkers)
            {
                Console.WriteLine($"Worker {worker.WorkerId} missed heartbeat, marking as offline");
                worker.Status = WorkerStatus.Offline;
                
                // Reassign any active tasks
                if (!string.IsNullOrEmpty(worker.CurrentTaskId) && 
                    _activeTasks.TryGetValue(worker.CurrentTaskId, out var task))
                {
                    task.Status = TaskStatus.Pending;
                    task.WorkerId = null;
                    task.AssignedAt = null;
                    _taskQueue.Enqueue(task);
                    Console.WriteLine($"Reassigning task {task.TaskId} due to worker disconnection");
                }

                WorkerDisconnected?.Invoke(this, worker);
                
                // Clean up console lines for offline worker
                _workerConsoleLines.TryRemove(worker.WorkerId, out _);
            }
        }

        #endregion

        #region Task Management

        public string SubmitTask(NamedSerializableObject command)
        {
            var task = new TaskInfo(command);
            _taskQueue.Enqueue(task);
            
            Console.WriteLine($"Task {task.TaskId} queued: {command.Name}");
            return task.TaskId;
        }

        public PollResponse PollForTask(string workerId, PollRequest pollRequest = null)
        {
            if (!_workers.TryGetValue(workerId, out var worker))
                return new PollResponse(); // Worker not registered

            worker.LastHeartbeat = DateTime.UtcNow;

            // Store console lines if provided
            if (pollRequest?.ConsoleLines != null && pollRequest.ConsoleLines.Count > 0)
            {
                var workerConsole = _workerConsoleLines.GetOrAdd(workerId, _ => new List<LogEntry>());
                
                lock (workerConsole)
                {
                    workerConsole.AddRange(pollRequest.ConsoleLines);
                    
                    // Limit stored lines to prevent unbounded growth
                    if (workerConsole.Count > _maxConsoleLines)
                    {
                        int excess = workerConsole.Count - _maxConsoleLines;
                        workerConsole.RemoveRange(0, excess);
                    }
                }
            }

            // Try to assign a task
            lock (_assignmentLock)
            {
                if (_taskQueue.TryDequeue(out var task))
                {
                    // Assign task to worker
                    task.WorkerId = workerId;
                    task.Status = TaskStatus.Assigned;
                    task.AssignedAt = DateTime.UtcNow;
                    
                    _activeTasks[task.TaskId] = task;
                    worker.Status = WorkerStatus.Working;
                    worker.CurrentTaskId = task.TaskId;

                    Console.WriteLine($"Task {task.TaskId} assigned to worker {workerId}");
                    
                    return new PollResponse 
                    { 
                        Task = task,
                        NextPollDelayMs = 1000 // Poll quickly after receiving a task
                    };
                }
            }

            // No tasks available, set worker to idle
            if (worker.Status == WorkerStatus.Working && string.IsNullOrEmpty(worker.CurrentTaskId))
                worker.Status = WorkerStatus.Idle;

            return new PollResponse 
            { 
                NextPollDelayMs = 5000 // Standard polling interval
            };
        }

        public bool UpdateTaskStatus(string workerId, string taskId, TaskUpdateRequest update)
        {
            if (!_workers.TryGetValue(workerId, out var worker))
                return false;

            if (!_activeTasks.TryGetValue(taskId, out var task))
                return false;

            if (task.WorkerId != workerId)
                return false; // Task not assigned to this worker

            worker.LastHeartbeat = DateTime.UtcNow;

            var oldStatus = task.Status;
            task.Status = update.Status;

            switch (update.Status)
            {
                case TaskStatus.Running:
                    if (oldStatus == TaskStatus.Assigned)
                        task.StartedAt = DateTime.UtcNow;
                    break;

                case TaskStatus.Completed:
                    task.CompletedAt = DateTime.UtcNow;
                    task.Result = update.Result;
                    _activeTasks.TryRemove(taskId, out _);
                    worker.Status = WorkerStatus.Idle;
                    worker.CurrentTaskId = null;
                    Console.WriteLine($"Task {taskId} completed by worker {workerId}");
                    TaskCompleted?.Invoke(this, task);
                    break;

                case TaskStatus.Failed:
                    task.CompletedAt = DateTime.UtcNow;
                    task.ErrorMessage = update.ErrorMessage;
                    _activeTasks.TryRemove(taskId, out _);
                    worker.Status = WorkerStatus.Idle;
                    worker.CurrentTaskId = null;
                    Console.WriteLine($"Task {taskId} failed on worker {workerId}: {update.ErrorMessage}");
                    TaskFailed?.Invoke(this, task);
                    break;

                case TaskStatus.Cancelled:
                    task.CompletedAt = DateTime.UtcNow;
                    _activeTasks.TryRemove(taskId, out _);
                    worker.Status = WorkerStatus.Idle;
                    worker.CurrentTaskId = null;
                    Console.WriteLine($"Task {taskId} cancelled on worker {workerId}");
                    break;
            }

            if (!string.IsNullOrEmpty(update.ProgressMessage))
                Console.WriteLine($"Task {taskId} progress: {update.ProgressMessage}");

            return true;
        }

        public TaskInfo GetTask(string taskId)
        {
            _activeTasks.TryGetValue(taskId, out var task);
            return task;
        }

        public IEnumerable<TaskInfo> GetActiveTasks()
        {
            return _activeTasks.Values.ToList();
        }

        public int GetQueueLength()
        {
            return _taskQueue.Count;
        }

        public List<LogEntry> GetWorkerConsoleLines(string workerId)
        {
            if (!_workerConsoleLines.TryGetValue(workerId, out var consoleLines))
                return new List<LogEntry>();
                
            lock (consoleLines)
            {
                return new List<LogEntry>(consoleLines);
            }
        }
        
        public List<LogEntry> GetWorkerConsoleLines(string workerId, int count)
        {
            var allLines = GetWorkerConsoleLines(workerId);
            return allLines.Skip(Math.Max(0, allLines.Count - count)).ToList();
        }
        
        public List<LogEntry> GetWorkerConsoleLines(string workerId, int start, int count)
        {
            var allLines = GetWorkerConsoleLines(workerId);
            return allLines.Skip(start).Take(count).ToList();
        }

        private void MonitorTaskTimeouts(object state)
        {
            if (_disposed) return;

            // Check for tasks that have been running too long (configurable, default 30 minutes)
            var cutoff = DateTime.UtcNow.AddMinutes(-30);
            var timedOutTasks = _activeTasks.Values
                .Where(t => t.Status == TaskStatus.Running && t.StartedAt.HasValue && t.StartedAt < cutoff)
                .ToList();

            foreach (var task in timedOutTasks)
            {
                Console.WriteLine($"Task {task.TaskId} timed out, reassigning");
                
                // Mark worker as failed if it has a timed-out task
                if (_workers.TryGetValue(task.WorkerId, out var worker))
                {
                    worker.Status = WorkerStatus.Failed;
                    worker.CurrentTaskId = null;
                }

                // Reassign task
                task.Status = TaskStatus.Pending;
                task.WorkerId = null;
                task.AssignedAt = null;
                task.StartedAt = null;
                _activeTasks.TryRemove(task.TaskId, out _);
                _taskQueue.Enqueue(task);
            }
        }

        #endregion

        #region Status and Statistics

        public object GetStatus()
        {
            var workers = _workers.Values.ToList();
            var tasks = _activeTasks.Values.ToList();

            return new
            {
                Workers = new
                {
                    Total = workers.Count,
                    Online = workers.Count(w => w.Status != WorkerStatus.Offline),
                    Idle = workers.Count(w => w.Status == WorkerStatus.Idle),
                    Working = workers.Count(w => w.Status == WorkerStatus.Working),
                    Failed = workers.Count(w => w.Status == WorkerStatus.Failed)
                },
                Tasks = new
                {
                    Queued = _taskQueue.Count,
                    Active = tasks.Count,
                    Running = tasks.Count(t => t.Status == TaskStatus.Running),
                    Assigned = tasks.Count(t => t.Status == TaskStatus.Assigned)
                },
                Timestamp = DateTime.UtcNow
            };
        }

        #endregion

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _heartbeatMonitor?.Dispose();
            _taskMonitor?.Dispose();

            // Notify all workers to stop
            foreach (var worker in _workers.Values)
            {
                worker.Status = WorkerStatus.Offline;
            }
        }
    }
}