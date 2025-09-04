using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Warp.Tools;
using Warp.Workers.WorkerController;
using ZLinq;
using TaskStatus = Warp.Workers.WorkerController.TaskStatus;


namespace Warp.Workers
{
    public partial class WorkerWrapper : IDisposable
    {
        private static int NWorkers = 0;
        
        // Static controller management
        private static WorkerControllerHost _sharedController = null;
        private static int _controllerPort = 0;
        private static readonly object _controllerLock = new object();
        private static readonly Dictionary<string, WorkerWrapper> _activeWorkers = new Dictionary<string, WorkerWrapper>();

        bool IsAlive = true;

        public int DeviceID = 0;
        public int Port { get; private set; }
        public string Host { get; private set; }
        
        private string _workerId;
        
        // WorkerInfo fields integrated into WorkerWrapper
        public string WorkerId => _workerId;
        public WorkerStatus Status { get; set; } = WorkerStatus.Idle;
        public DateTime ConnectedAt { get; set; }
        public DateTime LastHeartbeat { get; set; }
        public string CurrentTask { get; set; }
        private List<LogEntry> _consoleLines = new List<LogEntry>();

        public readonly WorkerConsole WorkerConsole;

        Thread Heartbeat;

        public event EventHandler<EventArgs> WorkerDied;
        
        // Static events for pool integration
        public static event EventHandler<WorkerWrapper> WorkerRegistered;
        public static event EventHandler<WorkerWrapper> WorkerDisconnected;

        /// <summary>
        /// Create WorkerWrapper for a worker that has registered with the controller
        /// This is the new unified constructor for both local and remote workers
        /// </summary>
        public WorkerWrapper(WorkerInfo controllerWorkerInfo)
        {
            // Copy properties from controller info
            _workerId = controllerWorkerInfo.WorkerId;
            DeviceID = controllerWorkerInfo.DeviceId;
            Host = "localhost"; // All workers connect through our controller
            Port = _controllerPort;
            
            // Initialize status tracking
            Status = WorkerStatus.Idle;
            ConnectedAt = DateTime.UtcNow;
            LastHeartbeat = DateTime.UtcNow;
            CurrentTask = null;
            
            // Initialize console and monitoring
            WorkerConsole = new WorkerConsole(this);
            StartHeartbeat();
            
            Console.WriteLine($"WorkerWrapper created for worker {_workerId} on device {DeviceID}");
        }

        #region Static Controller Management
        
        private static void EnsureControllerStarted(int port = 0)
        {
            lock (_controllerLock)
            {
                if (_sharedController == null)
                {
                    _sharedController = new WorkerControllerHost();
                    _controllerPort = _sharedController.StartAsync(port).GetAwaiter().GetResult();
                    Console.WriteLine($"Started shared controller on port {_controllerPort}");
                    
                    // Subscribe to controller events for all WorkerWrapper instances
                    var controllerService = _sharedController.GetService();
                    controllerService.WorkerRegistered += OnSharedControllerWorkerRegistered;
                    controllerService.WorkerDisconnected += OnSharedControllerWorkerDisconnected;
                }
            }
        }
        
        // Start controller on specific port for remote worker scenarios
        public static void StartControllerOnPort(int port)
        {
            EnsureControllerStarted(port);
        }
        
        // Static methods for worker management
        
        /// <summary>
        /// Spawn a local worker process that will connect to the shared controller
        /// </summary>
        public static async Task<Process> SpawnLocalWorkerAsync(
            int deviceId, 
            bool silent = false, 
            bool attachDebugger = false, 
            bool mockMode = false,
            TimeSpan? startupTimeout = null)
        {
            // Ensure controller is running
            EnsureControllerStarted();
            
            bool isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            string controllerEndpoint = $"localhost:{_controllerPort}";
            var timeout = startupTimeout ?? TimeSpan.FromSeconds(attachDebugger ? 200 : 30);
            
            ProcessStartInfo startInfo;
            if (isWindows)
            {
                startInfo = new ProcessStartInfo()
                {
                    FileName = Path.Combine(AppContext.BaseDirectory, "WarpWorker"),
                    CreateNoWindow = false,
                    WindowStyle = ProcessWindowStyle.Minimized,
                    Arguments = $"-d {deviceId} --controller {controllerEndpoint} {(silent ? "-s" : "")} " +
                                $"{(Debugger.IsAttached ? "--debug" : "")} " +
                                $"{(attachDebugger ? "--debug_attach" : "")} " +
                                $"{(mockMode ? "--mock" : "")}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false
                };
            }
            else
            {
                if (Helper.IsDebug)
                    startInfo = new ProcessStartInfo()
                    {
                        FileName = "bash",
                        Arguments = $"-c \"{Path.Combine(AppContext.BaseDirectory, "WarpWorker")} " +
                                    $"-d {deviceId} --controller {controllerEndpoint} " +
                                    $"{(silent ? "-s" : "")} {(Debugger.IsAttached ? "--debug" : "")} " +
                                    $"{(attachDebugger ? "--debug_attach" : "")} " +
                                    $"{(mockMode ? "--mock" : "")} " +
                                    $"> worker_dev{deviceId}.out 2> worker_dev{deviceId}.err\"",
                        UseShellExecute = false
                    };
                else
                    startInfo = new ProcessStartInfo()
                    {
                        FileName = Path.Combine(AppContext.BaseDirectory, "WarpWorker"),
                        Arguments = $"-d {deviceId} --controller {controllerEndpoint} {(silent ? "-s" : "")} " +
                                    $"{(Debugger.IsAttached ? "--debug" : "")} " +
                                    $"{(attachDebugger ? "--debug_attach" : "")} " +
                                    $"{(mockMode ? "--mock" : "")}",
                        UseShellExecute = false
                    };
            }

            try
            {
                Console.WriteLine($"Spawning local worker for device {deviceId}: {startInfo.FileName} {startInfo.Arguments}");
                
                var process = new Process { StartInfo = startInfo };
                process.Start();
                
                // Give the process a moment to start
                await Task.Delay(500);
                
                // Check if process is still running (didn't crash immediately)
                if (process.HasExited)
                {
                    Console.WriteLine($"Worker process for device {deviceId} exited immediately with code {process.ExitCode}");
                    return null;
                }
                
                Console.WriteLine($"Worker process for device {deviceId} started successfully (PID: {process.Id})");
                
                // Don't wait for registration - that's handled by controller events
                // Worker will register itself with the controller when ready
                return process;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to spawn worker for device {deviceId}: {ex.Message}");
                return null;
            }
        }
        
        // Static methods for pool integration
        public static List<WorkerWrapper> GetAllWorkers()
        {
            lock (_activeWorkers)
            {
                return _activeWorkers.Values.ToList();
            }
        }
        
        public static int GetControllerPort()
        {
            return _controllerPort;
        }
        
        private static void OnSharedControllerWorkerRegistered(object sender, WorkerInfo workerInfo)
        {
            WorkerWrapper workerWrapper = null;
            
            lock (_activeWorkers)
            {
                // Create new WorkerWrapper for this registered worker
                workerWrapper = new WorkerWrapper(workerInfo);
                _activeWorkers[workerInfo.WorkerId] = workerWrapper;
                
                Console.WriteLine($"Created WorkerWrapper for registered worker {workerInfo.WorkerId} on device {workerInfo.DeviceId}");
            }
            
            // Fire static event for pool integration
            //Console.WriteLine($"Firing WorkerRegistered event, subscribers: {WorkerRegistered?.GetInvocationList()?.Length ?? 0}");
            WorkerRegistered?.Invoke(null, workerWrapper);
        }
        
        private static void OnSharedControllerWorkerDisconnected(object sender, WorkerInfo worker)
        {
            WorkerWrapper disconnectedWorker = null;
            
            lock (_activeWorkers)
            {
                if (_activeWorkers.TryGetValue(worker.WorkerId, out disconnectedWorker))
                {
                    disconnectedWorker.Status = WorkerStatus.Offline;
                    disconnectedWorker.ReportDeath();
                    _activeWorkers.Remove(worker.WorkerId);
                    
                    Console.WriteLine($"Removed WorkerWrapper for disconnected worker {worker.WorkerId}");
                }
            }
            
            // Fire static event for pool integration only if we had a WorkerWrapper
            if (disconnectedWorker != null)
            {
                WorkerDisconnected?.Invoke(null, disconnectedWorker);
            }
            else
            {
                Console.WriteLine($"Worker {worker.WorkerId} disconnected but no WorkerWrapper found");
            }
        }
        
        
        #endregion

        #region Private

        void StartHeartbeat()
        {
            Heartbeat = new Thread(new ThreadStart(() =>
            {
                Thread.Sleep(2000);

                while (IsAlive)
                {
                    try
                    {
                        // For local workers with workerId, check if worker is still registered with controller
                        // For remote worker managers (workerId is null), just monitor controller health
                        if (_workerId != null)
                        {
                            var controllerService = _sharedController.GetService();
                            var activeWorkers = controllerService.GetActiveWorkers();
                            bool workerStillActive = activeWorkers.Any(w => w.WorkerId == _workerId);
                            
                            if (!workerStillActive)
                            {
                                // Worker is no longer registered with controller - mark any active tasks as failed
                                var activeTasks = controllerService.GetActiveTasks().Where(t => t.WorkerId == _workerId).ToList();
                                foreach (var task in activeTasks)
                                {
                                    try
                                    {
                                        controllerService.UpdateTaskStatus(_workerId, task.TaskId, new TaskUpdateRequest
                                        {
                                            Status = TaskStatus.Failed,
                                            ErrorMessage = "Worker disconnected during task execution"
                                        });
                                    }
                                    catch { } // Task might already be completed/failed
                                }
                                
                                ReportDeath();
                                IsAlive = false;
                                break;
                            }
                            
                            // Update console lines from controller
                            _consoleLines = controllerService.GetWorkerConsoleLines(_workerId);
                        }
                        
                        Thread.Sleep(2000);
                    }
                    catch
                    {
                        if (!Debugger.IsAttached)
                        {
                            ReportDeath();
                            IsAlive = false;
                        }
                    }
                }
            }));
            Heartbeat.Start();
        }

        void SendCommand(NamedSerializableObject command)
        {
            // All workers now use controller task submission to specific worker
            var controllerService = _sharedController.GetService();
            
            string taskId;
            try
            {
                taskId = controllerService.SubmitTaskToWorker(_workerId, command);
            }
            catch (ArgumentException ex) when (ex.Message.Contains("offline"))
            {
                throw new Exception($"Worker {_workerId} is offline and cannot accept new tasks");
            }

            // Wait for task completion with timeout
            var timeout = DateTime.UtcNow.AddMinutes(30);
            
            while (DateTime.UtcNow < timeout)
            {
                var task = controllerService.GetTask(taskId);
                if (task == null) // Task completed and removed
                    return;

                if (task.Status == TaskStatus.Completed)
                    return;

                if (task.Status == TaskStatus.Failed)
                    throw new Exception($"Task failed: {task.ErrorMessage}");

                // Only check worker health if task is actively running
                // Pending/Assigned tasks will be handled by disposal logic
                if (task.Status == TaskStatus.Running && !IsAlive)
                    throw new Exception($"Worker {_workerId} died during task execution, status was {task.Status}");

                Thread.Sleep(50);
            }
            
            // If we get here, we timed out
            throw new Exception($"Task execution timed out after 30 minutes");
        }

        void SendExit()
        {
            // All workers now use controller for exit commands
            try
            {
                var controllerService = _sharedController.GetService();
                controllerService.SubmitTaskToWorker(_workerId, new NamedSerializableObject("Exit"));
            }
            catch { }
        }

        
        void ReportDeath()
        {
            WorkerDied?.Invoke(this, null);
        }

        public void Dispose()
        {
            if (IsAlive)
            {
                Debug.WriteLine($"Disposing worker {_workerId}");

                // First, wait for active tasks to complete (with timeout)
                if (_workerId != null)
                {
                    try
                    {
                        var controllerService = _sharedController.GetService();
                        var deadline = DateTime.UtcNow.AddSeconds(1);
                        
                        while (DateTime.UtcNow < deadline)
                        {
                            var activeTasks = controllerService.GetActiveTasks().Where(t => t.WorkerId == _workerId).ToList();
                            if (!activeTasks.Any())
                                break; // No active tasks, safe to dispose
                                
                            Thread.Sleep(50); // Check every 50ms
                        }
                        
                        // After timeout, mark any remaining tasks as failed
                        var remainingTasks = controllerService.GetActiveTasks().Where(t => t.WorkerId == _workerId).ToList();
                        foreach (var task in remainingTasks)
                        {
                            try
                            {
                                controllerService.UpdateTaskStatus(_workerId, task.TaskId, new TaskUpdateRequest
                                {
                                    Status = TaskStatus.Failed,
                                    ErrorMessage = "Worker disposed during task execution"
                                });
                            }
                            catch { } // Task might already be completed/failed
                        }
                    }
                    catch { } // Controller might already be disposed
                }

                IsAlive = false;

                WaitAsyncTasks();
                SendExit();
                
                // Remove from active workers tracking
                if (_workerId != null)
                    lock (_activeWorkers)
                    {
                        _activeWorkers.Remove(_workerId);
                    }
            }
        }

        ~WorkerWrapper()
        {
            Dispose();
        }

        #endregion
        
        #region Console
        
        public List<LogEntry> GetConsoleLines()
        {
            return [.._consoleLines];
        }
        
        public List<LogEntry> GetLastNConsoleLines(int count)
        {
            return _consoleLines.Skip(Math.Max(0, _consoleLines.Count - count)).ToList();
        }
        
        #endregion
    }

    public class WorkerConsole
    {
        private readonly WorkerWrapper _workerWrapper;

        public WorkerConsole(WorkerWrapper workerWrapper)
        {
            _workerWrapper = workerWrapper;
        }

        public int GetLineCount()
        {
            return _workerWrapper.GetConsoleLines().Count;
        }

        public List<LogEntry> GetAllLines()
        {
            return _workerWrapper.GetConsoleLines();
        }

        public List<LogEntry> GetLastNLines(int count)
        {
            return _workerWrapper.GetLastNConsoleLines(count);
        }

        public List<LogEntry> GetFirstNLines(int count)
        {
            var allLines = _workerWrapper.GetConsoleLines();
            return allLines.Take(count).ToList();
        }

        public List<LogEntry> GetLinesRange(int start, int end)
        {
            var allLines = _workerWrapper.GetConsoleLines();
            return allLines.Skip(Math.Max(0, start)).Take(Math.Max(0, end - start)).ToList();
        }

        public void Clear()
        {
            // Cannot clear remote worker console - this is a no-op
            // The console is managed by the worker process itself
        }

        public void SetFileOutput(string path)
        {
            // Cannot set file output on remote worker - this is a no-op
            // File output is managed by the worker process itself
        }

        public void WriteToFile(string path)
        {
            // Write current console lines to local file
            var lines = _workerWrapper.GetConsoleLines();
            using (var writer = File.CreateText(path))
            {
                foreach (var line in lines)
                {
                    writer.WriteLine(line.Timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff") + " " + line.Message);
                }
            }
        }
    }
}