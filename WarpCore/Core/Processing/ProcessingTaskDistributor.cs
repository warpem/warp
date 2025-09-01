using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Warp;
using Warp.Workers;
using TaskStatus = Warp.Workers.WorkerController.TaskStatus;

namespace WarpCore.Core.Processing
{
    public class ProcessingTask
    {
        public string TaskId { get; set; }
        public Movie Movie { get; set; }
        public string WorkerId { get; set; }
        public WorkerWrapper Worker { get; set; }
        public DateTime StartedAt { get; set; }
        public ProcessingStep CurrentStep { get; set; }
        public TaskStatus Status { get; set; }
        public string ErrorMessage { get; set; }
    }

    public enum ProcessingStep
    {
        Motion,
        CTF,
        Picking,
        Export
    }

    public class ProcessingTaskDistributor
    {
        private readonly ILogger<ProcessingTaskDistributor> _logger;
        private readonly Dictionary<string, ProcessingTask> _activeTasks = new Dictionary<string, ProcessingTask>();
        private readonly object _tasksLock = new object();
        private readonly WorkerPool _workerPool;

        public ProcessingTaskDistributor(ILogger<ProcessingTaskDistributor> logger, WorkerPool workerPool)
        {
            _logger = logger;
            _workerPool = workerPool;
        }

        /// <summary>
        /// Assign movies to available workers (thread-safe via WorkerPool)
        /// </summary>
        public async Task<List<ProcessingTask>> AssignTasksAsync(
            List<Movie> movies, 
            OptionsWarp currentSettings)
        {
            var assignedTasks = new List<ProcessingTask>();

            foreach (var movie in movies)
            {
                var taskId = Guid.NewGuid().ToString();
                
                // Use WorkerPool's thread-safe reservation method
                var task = _workerPool.ReserveWorker(movie, taskId);
                if (task == null)
                {
                    // No available workers - break out of the loop
                    break;
                }

                // Complete task initialization
                task.CurrentStep = DetermineFirstStep(movie, currentSettings);

                lock (_tasksLock)
                {
                    _activeTasks[task.TaskId] = task;
                }

                assignedTasks.Add(task);
                _logger.LogInformation($"Assigned movie {movie.Name} to worker {task.WorkerId} (task {task.TaskId})");
            }

            return assignedTasks;
        }

        /// <summary>
        /// Mark a task as completed successfully
        /// </summary>
        public void CompleteTask(string taskId)
        {
            lock (_tasksLock)
            {
                if (_activeTasks.TryGetValue(taskId, out var task))
                {
                    task.Status = TaskStatus.Completed;
                    task.Movie.ProcessingStatus = Warp.ProcessingStatus.Processed;
                    
                    // Save metadata with updated processing options
                    task.Movie.SaveMeta();
                    
                    _logger.LogInformation($"Task {taskId} completed successfully for movie {task.Movie.Name}");
                    _activeTasks.Remove(taskId);
                    
                    // Return worker to pool (thread-safe)
                    _workerPool.ReturnWorker(task.WorkerId);
                }
            }
        }

        /// <summary>
        /// Mark a task as failed
        /// </summary>
        public void FailTask(string taskId, string errorMessage)
        {
            lock (_tasksLock)
            {
                if (_activeTasks.TryGetValue(taskId, out var task))
                {
                    task.Status = TaskStatus.Failed;
                    task.ErrorMessage = errorMessage;
                    
                    // Mark movie as deselected due to processing failure
                    task.Movie.UnselectManual = true;
                    task.Movie.ProcessingStatus = Warp.ProcessingStatus.LeaveOut;
                    task.Movie.SaveMeta();
                    
                    _logger.LogWarning($"Task {taskId} failed for movie {task.Movie.Name}: {errorMessage}");
                    _activeTasks.Remove(taskId);
                    
                    // Return worker to pool (thread-safe)
                    _workerPool.ReturnWorker(task.WorkerId);
                }
            }
        }

        /// <summary>
        /// Update task progress
        /// </summary>
        public void UpdateTaskProgress(string taskId, ProcessingStep currentStep)
        {
            lock (_tasksLock)
            {
                if (_activeTasks.TryGetValue(taskId, out var task))
                {
                    task.CurrentStep = currentStep;
                    task.Status = TaskStatus.Running;
                    _logger.LogDebug($"Task {taskId} progress: {currentStep}");
                }
            }
        }

        /// <summary>
        /// Get all active tasks
        /// </summary>
        public List<ProcessingTask> GetActiveTasks()
        {
            lock (_tasksLock)
            {
                return _activeTasks.Values.ToList();
            }
        }

        /// <summary>
        /// Get tasks assigned to a specific worker
        /// </summary>
        public List<ProcessingTask> GetTasksForWorker(string workerId)
        {
            lock (_tasksLock)
            {
                return _activeTasks.Values
                    .Where(t => t.WorkerId == workerId)
                    .ToList();
            }
        }

        /// <summary>
        /// Reassign tasks from a disconnected worker
        /// </summary>
        public List<Movie> ReassignTasksFromWorker(string disconnectedWorkerId)
        {
            var moviesToReprocess = new List<Movie>();

            lock (_tasksLock)
            {
                var workerTasks = _activeTasks.Values
                    .Where(t => t.WorkerId == disconnectedWorkerId)
                    .ToList();

                foreach (var task in workerTasks)
                {
                    _logger.LogWarning($"Reassigning task {task.TaskId} from disconnected worker {disconnectedWorkerId}");
                    
                    // Reset movie processing status so it can be reprocessed
                    task.Movie.ProcessingStatus = Warp.ProcessingStatus.Outdated;
                    moviesToReprocess.Add(task.Movie);
                    
                    _activeTasks.Remove(task.TaskId);
                }
            }

            return moviesToReprocess;
        }

        /// <summary>
        /// Clean up stale tasks (timeout handling)
        /// </summary>
        public List<Movie> CleanupStaleTasks(TimeSpan timeout)
        {
            var staleMovies = new List<Movie>();
            var cutoffTime = DateTime.UtcNow - timeout;

            lock (_tasksLock)
            {
                var staleTasks = _activeTasks.Values
                    .Where(t => t.StartedAt < cutoffTime)
                    .ToList();

                foreach (var task in staleTasks)
                {
                    _logger.LogWarning($"Task {task.TaskId} timed out after {timeout.TotalMinutes} minutes");
                    
                    // Mark movie for reprocessing
                    task.Movie.ProcessingStatus = Warp.ProcessingStatus.Outdated;
                    staleMovies.Add(task.Movie);
                    
                    _activeTasks.Remove(task.TaskId);
                }
            }

            return staleMovies;
        }

        private ProcessingStep DetermineFirstStep(Movie movie, OptionsWarp currentSettings)
        {
            // Determine which processing step to start with based on current settings
            // and what's already been done
            
            if (currentSettings.ProcessMovement && 
                (movie.OptionsMovement == null || movie.OptionsMovement != currentSettings.GetProcessingMovieMovement()))
            {
                return ProcessingStep.Motion;
            }
            
            if (currentSettings.ProcessCTF && 
                (movie.OptionsCTF == null || movie.OptionsCTF != currentSettings.GetProcessingMovieCTF()))
            {
                return ProcessingStep.CTF;
            }
            
            if (currentSettings.ProcessPicking && 
                (movie.OptionsBoxNet == null || movie.OptionsBoxNet != currentSettings.GetProcessingBoxNet()))
            {
                return ProcessingStep.Picking;
            }
            
            // Default to export if other steps are up to date
            return ProcessingStep.Export;
        }
    }
}