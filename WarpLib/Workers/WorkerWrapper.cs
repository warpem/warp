using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
        private readonly List<string> _workerOutput = new List<string>();
        private readonly List<string> _workerErrors = new List<string>();
        private readonly object _outputLock = new object();
        private List<LogEntry> _consoleLines = new List<LogEntry>();

        public readonly WorkerConsole WorkerConsole;

        Thread Heartbeat;
        Process Worker;

        public event EventHandler<EventArgs> WorkerDied;
        
        // Static events for pool integration
        public static event EventHandler<WorkerWrapper> WorkerRegistered;
        public static event EventHandler<WorkerWrapper> WorkerDisconnected;

        // Create new worker using controller architecture
        public WorkerWrapper(int deviceID, bool silent = false, bool attachDebugger = false, bool mockMode = false)
        {
            DeviceID = deviceID;
            Host = "localhost";
            
            EnsureControllerStarted();
            Port = _controllerPort;
            
            WorkerConsole = new WorkerConsole(this);
            
            _workerId = SpawnWorkerProcess(deviceID, silent, attachDebugger, mockMode).GetAwaiter().GetResult();
            
            lock (_activeWorkers)
            {
                _activeWorkers[_workerId] = this;
                Console.WriteLine($"WorkerWrapper tracking worker {_workerId} for device {deviceID}");
            }

            StartHeartbeat();
        }

        // Start controller for remote workers to connect to (for HPC scenarios)
        public WorkerWrapper(int controllerPort, string hostName = null)
        {
            Host = hostName ?? "localhost";
            
            EnsureControllerStarted(controllerPort);
            Port = _controllerPort;
            
            // No worker process to spawn - remote workers will connect to our controller
            _workerId = null;
            
            StartHeartbeat();
            WorkerConsole = new WorkerConsole(this);
            
            if (!string.IsNullOrEmpty(hostName))
            {
                Console.WriteLine($"Controller started on {hostName}:{Port} for remote worker connections");
            }
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
        
        private static void OnSharedControllerWorkerRegistered(object sender, WorkerInfo worker)
        {
            // For now, only fire events for workers that have WorkerWrapper instances
            // Remote workers connecting directly to controller won't have WorkerWrapper instances
            lock (_activeWorkers)
            {
                if (_activeWorkers.TryGetValue(worker.WorkerId, out var workerWrapper))
                {
                    // Update the status from controller
                    workerWrapper.Status = WorkerStatus.Idle;
                    workerWrapper.ConnectedAt = DateTime.UtcNow;
                    workerWrapper.LastHeartbeat = DateTime.UtcNow;
                    
                    Console.WriteLine($"FIRING EVENT, subscribers: {WorkerRegistered?.GetInvocationList().Length}");
                    
                    // Fire static event for pool integration
                    WorkerRegistered?.Invoke(null, workerWrapper);
                }
                else
                    Console.WriteLine($"WorkerWrapper ERROR: Worker {worker.WorkerId} registered but no WorkerWrapper instance found");
            }
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
                }
            }
            
            // Fire static event for pool integration only if we have a WorkerWrapper
            if (disconnectedWorker != null)
            {
                WorkerDisconnected?.Invoke(null, disconnectedWorker);
            }
        }
        
        private async Task<string> SpawnWorkerProcess(int deviceID, bool silent, bool attachDebugger, bool mockMode = false)
        {
            bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            
            string controllerEndpoint = $"{Host}:{Port}";
            
            ProcessStartInfo startInfo;
            if (IsWindows)
            {
                startInfo = new ProcessStartInfo()
                {
                    FileName = Path.Combine(AppContext.BaseDirectory, "WarpWorker"),
                    CreateNoWindow = false,
                    WindowStyle = ProcessWindowStyle.Minimized,
                    Arguments = $"-d {deviceID} --controller {controllerEndpoint} {(silent ? "-s" : "")} " +
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
                                    $"-d {deviceID} --controller {controllerEndpoint} " +
                                    $"{(silent ? "-s" : "")} {(Debugger.IsAttached ? "--debug" : "")} " +
                                    $"{(attachDebugger ? "--debug_attach" : "")} " +
                                    $"{(mockMode ? "--mock" : "")} " +
                                    $"> worker_dev{deviceID}.out 2> worker_dev{deviceID}.err\"",
                        UseShellExecute = false
                    };
                else
                    startInfo = new ProcessStartInfo()
                    {
                        FileName = Path.Combine(AppContext.BaseDirectory, "WarpWorker"),
                        Arguments = $"-d {deviceID} --controller {controllerEndpoint} {(silent ? "-s" : "")} " +
                                    $"{(Debugger.IsAttached ? "--debug" : "")} " +
                                    $"{(attachDebugger ? "--debug_attach" : "")} " +
                                    $"{(mockMode ? "--mock" : "")}",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false
                    };
            }

            Console.Error.WriteLine(startInfo.Arguments);
            Worker = new Process { StartInfo = startInfo };
            Worker.OutputDataReceived += OnWorkerOutputReceived;
            Worker.ErrorDataReceived += OnWorkerErrorReceived;
            Worker.Start();
            Worker.BeginOutputReadLine();
            Worker.BeginErrorReadLine();

            // Wait for worker registration (with timeout)
            var registrationTimeout = DateTime.UtcNow.AddSeconds(attachDebugger ? 200 : 30);
            string workerId = null;

            while (DateTime.UtcNow < registrationTimeout && workerId == null)
            {
                await Task.Delay(500);
                
                // Check if a worker with this device ID has registered
                var controllerService = _sharedController.GetService();
                var workers = controllerService.GetActiveWorkers();
                // Find a worker that hasn't been claimed by another WorkerWrapper
                var deviceWorker = workers.FirstOrDefault(w => !_activeWorkers.ContainsKey(w.WorkerId));
                
                if (deviceWorker != null)
                {
                    workerId = deviceWorker.WorkerId;
                    Console.WriteLine($"Worker {workerId} registered for device {deviceID}");
                    break;
                }
            }

            if (workerId == null)
                throw new TimeoutException($"Worker for device {deviceID} failed to register within timeout period");

            return workerId;
        }
        
        private void OnWorkerOutputReceived(object sender, DataReceivedEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                lock (_outputLock)
                {
                    _workerOutput.Add(e.Data);
                }
                
                //if (Debugger.IsAttached)
                {
                    Console.WriteLine($"Worker {DeviceID}: {e.Data}");
                }
            }
        }
        
        private void OnWorkerErrorReceived(object sender, DataReceivedEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                lock (_outputLock)
                {
                    _workerErrors.Add(e.Data);
                }
                
                // Always display errors
                Console.WriteLine($"Worker {DeviceID} ERROR: {e.Data}");
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
            // All workers now use controller task submission
            var controllerService = _sharedController.GetService();
            var taskId = controllerService.SubmitTask(command);

            // Wait for task completion with timeout
            var timeout = DateTime.UtcNow.AddMinutes(30);
            while (DateTime.UtcNow < timeout)
            {
                var task = controllerService.GetTask(taskId);
                if (task == null) // Task completed and removed
                    break;

                if (task.Status == TaskStatus.Completed)
                    break;

                if (task.Status == TaskStatus.Failed)
                    throw new Exception($"Task failed: {task.ErrorMessage}");

                Thread.Sleep(100);
            }
        }

        void SendExit()
        {
            // All workers now use controller for exit commands
            try
            {
                var controllerService = _sharedController.GetService();
                controllerService.SubmitTask(new NamedSerializableObject("Exit")); // Note: Exit is not a WorkerWrapper method
            }
            catch { }
        }

        private void DisplayWorkerOutput()
        {
            lock (_outputLock)
            {
                bool hasErrors = _workerErrors.Count > 0;
                bool shouldDisplay = Debugger.IsAttached || hasErrors;
                
                if (shouldDisplay && (_workerOutput.Count > 0 || _workerErrors.Count > 0))
                {
                    Console.WriteLine($"\n=== Worker {DeviceID} Output History ===");
                    
                    if (_workerOutput.Count > 0)
                    {
                        Console.WriteLine("--- Standard Output ---");
                        foreach (var line in _workerOutput)
                        {
                            Console.WriteLine(line);
                        }
                    }
                    
                    if (_workerErrors.Count > 0)
                    {
                        Console.WriteLine("--- Error Output ---");
                        foreach (var line in _workerErrors)
                        {
                            Console.WriteLine($"ERROR: {line}");
                        }
                    }
                    
                    Console.WriteLine($"=== End Worker {DeviceID} Output ===");
                }
            }
        }
        
        void ReportDeath()
        {
            DisplayWorkerOutput();
            WorkerDied?.Invoke(this, null);
        }

        public void Dispose()
        {
            if (IsAlive)
            {
                Debug.WriteLine($"Told worker {(_workerId ?? $"{Host}:{Port}")} to exit");
                IsAlive = false;

                WaitAsyncTasks();
                SendExit();
                
                // Display output if there were any issues (for local workers with processes)
                if (Worker != null)
                    DisplayWorkerOutput();
                
                // Clean up the worker process
                if (Worker != null)
                {
                    try
                    {
                        if (!Worker.HasExited)
                            Worker.Kill();
                        
                        Worker.OutputDataReceived -= OnWorkerOutputReceived;
                        Worker.ErrorDataReceived -= OnWorkerErrorReceived;
                        Worker.Dispose();
                    }
                    catch { }
                }
                
                // Remove from active workers tracking (for local workers)
                if (_workerId != null)
                {
                    lock (_activeWorkers)
                    {
                        _activeWorkers.Remove(_workerId);
                    }
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