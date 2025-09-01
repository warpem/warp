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
    /// <summary>
    /// Represents a processing task assigned to a worker. Contains all information
    /// needed to track and manage the execution of movie processing operations.
    /// </summary>
    public class ProcessingTask
    {
        /// <summary>
        /// Gets or sets the unique identifier for this processing task.
        /// </summary>
        public string TaskId { get; set; }
        
        /// <summary>
        /// Gets or sets the movie being processed in this task.
        /// </summary>
        public Movie Movie { get; set; }
        
        /// <summary>
        /// Gets or sets the unique identifier of the worker assigned to this task.
        /// </summary>
        public string WorkerId { get; set; }
        
        /// <summary>
        /// Gets or sets the worker wrapper instance handling this task.
        /// </summary>
        public WorkerWrapper Worker { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp when task execution began.
        /// </summary>
        public DateTime StartedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the current processing step being executed.
        /// </summary>
        public ProcessingStep CurrentStep { get; set; }
        
        /// <summary>
        /// Gets or sets the current status of the task execution.
        /// </summary>
        public TaskStatus Status { get; set; }
        
        /// <summary>
        /// Gets or sets the error message if the task failed.
        /// </summary>
        public string ErrorMessage { get; set; }
    }

    /// <summary>
    /// Enumeration of processing steps that can be performed on movies.
    /// Represents the different phases of electron microscopy data processing.
    /// </summary>
    public enum ProcessingStep
    {
        /// <summary>
        /// Motion correction processing step.
        /// </summary>
        Motion,
        
        /// <summary>
        /// Contrast Transfer Function (CTF) estimation step.
        /// </summary>
        CTF,
        
        /// <summary>
        /// Particle picking step.
        /// </summary>
        Picking,
        
        /// <summary>
        /// Movie export step.
        /// </summary>
        Export
    }

    /// <summary>
    /// Manages the distribution and lifecycle of processing tasks. Handles task assignment
    /// to workers, progress tracking, completion/failure handling, and task reassignment
    /// when workers disconnect or tasks timeout.
    /// </summary>
    public class ProcessingTaskDistributor
    {
        private readonly ILogger<ProcessingTaskDistributor> _logger;
        private readonly Dictionary<string, ProcessingTask> _activeTasks = new Dictionary<string, ProcessingTask>();
        private readonly object _tasksLock = new object();
        private readonly WorkerPool _workerPool;

        /// <summary>
        /// Initializes a new processing task distributor.
        /// </summary>
        /// <param name="logger">Logger for recording task distribution operations</param>
        /// <param name="workerPool">Worker pool for task assignment and management</param>
        public ProcessingTaskDistributor(ILogger<ProcessingTaskDistributor> logger, WorkerPool workerPool)
        {
            _logger = logger;
            _workerPool = workerPool;
        }

        /// <summary>
        /// Assigns movies to available workers using thread-safe worker pool operations.
        /// Creates processing tasks for each movie and tracks them until completion.
        /// Stops assignment if no more workers are available.
        /// </summary>
        /// <param name="movies">List of movies to assign to workers</param>
        /// <param name="currentSettings">Current processing settings to determine first processing step</param>
        /// <returns>List of processing tasks that were successfully assigned to workers</returns>
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
        /// Marks a task as completed successfully. Updates the movie's processing status,
        /// saves metadata, removes the task from active tracking, and returns the worker to the pool.
        /// </summary>
        /// <param name="taskId">Unique identifier of the task to complete</param>
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
        /// Marks a task as failed with an error message. Updates the movie to be excluded
        /// from processing, saves metadata, removes the task from active tracking,
        /// and returns the worker to the pool.
        /// </summary>
        /// <param name="taskId">Unique identifier of the task that failed</param>
        /// <param name="errorMessage">Description of the error that caused the failure</param>
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
        /// Updates the progress of a running task by recording the current processing step.
        /// Used for monitoring and debugging task execution.
        /// </summary>
        /// <param name="taskId">Unique identifier of the task to update</param>
        /// <param name="currentStep">Current processing step being executed</param>
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
        /// Gets a snapshot of all currently active tasks.
        /// Thread-safe operation that returns a copy of the active tasks list.
        /// </summary>
        /// <returns>List of all active processing tasks</returns>
        public List<ProcessingTask> GetActiveTasks()
        {
            lock (_tasksLock)
            {
                return _activeTasks.Values.ToList();
            }
        }

        /// <summary>
        /// Gets all active tasks assigned to a specific worker.
        /// Used for monitoring worker load and handling worker disconnections.
        /// </summary>
        /// <param name="workerId">Unique identifier of the worker</param>
        /// <returns>List of tasks assigned to the specified worker</returns>
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
        /// Reassigns all tasks from a disconnected worker back to the processing queue.
        /// Marks affected movies as outdated so they can be picked up by other workers.
        /// </summary>
        /// <param name="disconnectedWorkerId">Unique identifier of the disconnected worker</param>
        /// <returns>List of movies that need to be reprocessed</returns>
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
        /// Cleans up tasks that have been running longer than the specified timeout.
        /// Marks affected movies as outdated for reprocessing and removes stale tasks
        /// from active tracking. Used to handle hung or extremely slow processing.
        /// </summary>
        /// <param name="timeout">Maximum allowed task execution time</param>
        /// <returns>List of movies from stale tasks that need reprocessing</returns>
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

        /// <summary>
        /// Determines the first processing step that needs to be executed for a movie
        /// based on current settings and what processing has already been completed.
        /// Follows the processing pipeline order: Motion -> CTF -> Picking -> Export.
        /// </summary>
        /// <param name="movie">Movie to determine processing step for</param>
        /// <param name="currentSettings">Current processing settings</param>
        /// <returns>The first processing step that needs to be executed</returns>
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