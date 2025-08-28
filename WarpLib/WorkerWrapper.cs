using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http.Headers;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using Warp.Sociology;
using Warp.Tools;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.IO.Pipes;
using System.Threading.Tasks;
using Warp.WorkerController;
using System.Linq;
using TaskStatus = Warp.WorkerController.TaskStatus;


namespace Warp
{
    public class WorkerWrapper : IDisposable
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
        private bool _isRemoteWorker = false;
        private readonly List<string> _workerOutput = new List<string>();
        private readonly List<string> _workerErrors = new List<string>();
        private readonly object _outputLock = new object();

        public readonly WorkerConsole WorkerConsole;

        Thread Heartbeat;
        Process Worker;

        public event EventHandler<EventArgs> WorkerDied;

        // Create new worker using controller architecture
        public WorkerWrapper(int deviceID, bool silent = false, bool attachDebugger = false)
        {
            DeviceID = deviceID;
            Host = "localhost";
            
            EnsureControllerStarted();
            Port = _controllerPort;
            
            _workerId = SpawnWorkerProcess(deviceID, silent, attachDebugger).GetAwaiter().GetResult();
            
            lock (_activeWorkers)
            {
                _activeWorkers[_workerId] = this;
            }

            StartHeartbeat();
            WorkerConsole = new WorkerConsole(Host, Port);
        }

        // Connect to remote controller (for HPC scenarios)  
        public WorkerWrapper(string host, int port)
        {
            Host = host;
            Port = port;
            _isRemoteWorker = true;

            StartHeartbeat();
            WorkerConsole = new WorkerConsole(Host, Port);
        }

        #region Static Controller Management
        
        private static void EnsureControllerStarted()
        {
            lock (_controllerLock)
            {
                if (_sharedController == null)
                {
                    _sharedController = new WorkerControllerHost();
                    _controllerPort = _sharedController.StartAsync(0).GetAwaiter().GetResult();
                    Console.WriteLine($"Started shared controller on port {_controllerPort}");
                    
                    // Subscribe to controller events for all WorkerWrapper instances
                    var controllerService = _sharedController.GetService();
                    controllerService.WorkerDisconnected += OnSharedControllerWorkerDisconnected;
                }
            }
        }
        
        private static void OnSharedControllerWorkerDisconnected(object sender, WorkerInfo worker)
        {
            lock (_activeWorkers)
            {
                if (_activeWorkers.TryGetValue(worker.WorkerId, out var workerWrapper))
                {
                    workerWrapper.ReportDeath();
                    _activeWorkers.Remove(worker.WorkerId);
                }
            }
        }
        
        private async Task<string> SpawnWorkerProcess(int deviceID, bool silent, bool attachDebugger)
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
                                $"{(attachDebugger ? "--debug_attach" : "")}",
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
                                    $"> worker_dev{deviceID}.out 2> worker_dev{deviceID}.err\"",
                        UseShellExecute = false
                    };
                else
                    startInfo = new ProcessStartInfo()
                    {
                        FileName = Path.Combine(AppContext.BaseDirectory, "WarpWorker"),
                        CreateNoWindow = false,
                        WindowStyle = ProcessWindowStyle.Minimized,
                        Arguments = $"-d {deviceID} --controller {controllerEndpoint} {(silent ? "-s" : "")} " +
                                    $"{(Debugger.IsAttached ? "--debug" : "")} " +
                                    $"{(attachDebugger ? "--debug_attach" : "")}",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false
                    };
            }

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
                var deviceWorker = workers.FirstOrDefault(w => w.DeviceId == deviceID && 
                    !_activeWorkers.ContainsValue(this));
                
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
                
                if (Debugger.IsAttached)
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

        void StartHeartbeat()
        {
            Heartbeat = new Thread(new ThreadStart(() =>
            {
                Thread.Sleep(2000);

                while (IsAlive)
                {
                    try
                    {
                        if (!_isRemoteWorker)
                        {
                            // For local workers, check if worker is still registered with controller
                            var controllerService = _sharedController.GetService();
                            var activeWorkers = controllerService.GetActiveWorkers();
                            bool workerStillActive = activeWorkers.Any(w => w.WorkerId == _workerId);
                            
                            if (!workerStillActive)
                            {
                                ReportDeath();
                                IsAlive = false;
                                break;
                            }
                        }
                        else
                        {
                            // For remote workers, send a pulse to check connectivity
                            SendPulse();
                        }
                        
                        Thread.Sleep(5000);
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

        #region Private

        void SendPulse()
        {
            if (_isRemoteWorker)
            {
                // For remote workers, use direct HTTP connection
                using (var httpClient = new HttpClient())
                {
                    httpClient.BaseAddress = new Uri($"http://{Host}:{Port}");
                    httpClient.DefaultRequestHeaders.Accept.Clear();
                    httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
                    httpClient.Timeout = TimeSpan.FromSeconds(5);

                    HttpResponseMessage Response = httpClient.PostAsync("v1/Service/SendPulse", null).GetAwaiter().GetResult();

                    if (!Response.IsSuccessStatusCode)
                        throw new Exception();
                }
            }
            else
            {
                // For local workers using controller, check if worker is active
                var controllerService = _sharedController.GetService();
                var activeWorkers = controllerService.GetActiveWorkers();
                bool workerActive = activeWorkers.Any(w => w.WorkerId == _workerId);
                
                if (!workerActive)
                    throw new Exception("Worker not found in controller");
            }
        }

        void SendCommand(NamedSerializableObject command)
        {
            if (_isRemoteWorker)
            {
                // For remote workers, use direct HTTP connection
                using (var httpClient = new HttpClient())
                {
                    httpClient.BaseAddress = new Uri($"http://{Host}:{Port}");
                    httpClient.DefaultRequestHeaders.Accept.Clear();
                    httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
                    httpClient.Timeout = TimeSpan.FromSeconds(999999);

                    string Json = JsonSerializer.Serialize(command);
                    var Content = new StringContent(Json, Encoding.UTF8, "application/json");

                    HttpResponseMessage Response = httpClient.PostAsync("v1/Service/EvaluateCommand", Content).GetAwaiter().GetResult();

                    if (!Response.IsSuccessStatusCode)
                    {
                        var responseContent = Response.Content.ReadAsStringAsync().Result;
                        var jsonDocument = JsonDocument.Parse(responseContent);
                        
                        if (jsonDocument.RootElement.TryGetProperty("details", out var details))
                            throw new ExternalException(details.GetRawText());

                        throw new ExternalException(responseContent);
                    }
                }
            }
            else
            {
                // For local workers, use controller task submission
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
        }

        void SendExit()
        {
            if (_isRemoteWorker)
            {
                // For remote workers, use direct HTTP connection
                using (var httpClient = new HttpClient())
                {
                    httpClient.BaseAddress = new Uri($"http://{Host}:{Port}");
                    httpClient.DefaultRequestHeaders.Accept.Clear();
                    httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
                    httpClient.Timeout = TimeSpan.FromSeconds(5);

                    try
                    {
                        HttpResponseMessage Response = httpClient.PostAsync("v1/Service/Exit", null).GetAwaiter().GetResult();
                    }
                    catch { }
                }
            }
            else
            {
                // For local workers, send exit command through controller
                try
                {
                    var controllerService = _sharedController.GetService();
                    controllerService.SubmitTask(new NamedSerializableObject("Exit"));
                }
                catch { }
            }
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
                
                // Display output if there were any issues
                if (!_isRemoteWorker)
                {
                    DisplayWorkerOutput();
                }
                
                // Clean up the worker process
                if (Worker != null)
                {
                    try
                    {
                        if (!Worker.HasExited)
                        {
                            Worker.Kill();
                        }
                        Worker.OutputDataReceived -= OnWorkerOutputReceived;
                        Worker.ErrorDataReceived -= OnWorkerErrorReceived;
                        Worker.Dispose();
                    }
                    catch { }
                }
                
                // Remove from active workers tracking
                if (!_isRemoteWorker && _workerId != null)
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

        public void WaitAsyncTasks()
        {
            SendCommand(new NamedSerializableObject(nameof(WaitAsyncTasks)));
        }

        public void GcCollect()
        {
            SendCommand(new NamedSerializableObject("GcCollect"));
        }

        public void SetHeaderlessParams(int2 dims, long offset, string type)
        {
            SendCommand(new NamedSerializableObject("SetHeaderlessParams",
                                                    dims,
                                                    offset,
                                                    type));
        }

        public void LoadGainRef(string path, bool flipX, bool flipY, bool transpose, string defectsPath)
        {
            SendCommand(new NamedSerializableObject("LoadGainRef",
                                                    path,
                                                    flipX,
                                                    flipY,
                                                    transpose,
                                                    defectsPath));
        }

        public void LoadStack(string path, decimal scaleFactor, int eerGroupFrames, bool correctGain = true)
        {
            SendCommand(new NamedSerializableObject("LoadStack", 
                                                    path, 
                                                    scaleFactor, 
                                                    eerGroupFrames,
                                                    correctGain));
        }

        public void LoadBoxNet(string path, int boxSize, int batchSize)
        {
            SendCommand(new NamedSerializableObject("LoadBoxNet",
                                                    path,
                                                    boxSize,
                                                    batchSize));
        }

        public void DropBoxNet()
        {
            SendCommand(new NamedSerializableObject("DropBoxNet"));
        }

        public void MovieProcessCTF(string path, ProcessingOptionsMovieCTF options)
        {
            SendCommand(new NamedSerializableObject("MovieProcessCTF",
                                                    path,
                                                    options));
        }

        public void MovieProcessMovement(string path, ProcessingOptionsMovieMovement options)
        {
            SendCommand(new NamedSerializableObject("MovieProcessMovement",
                                                    path,
                                                    options));
        }

        public void MoviePickBoxNet(string path, ProcessingOptionsBoxNet options)
        {
            SendCommand(new NamedSerializableObject("MoviePickBoxNet",
                                                    path,
                                                    options));
        }

        public void MovieExportMovie(string path, ProcessingOptionsMovieExport options)
        {
            SendCommand(new NamedSerializableObject("MovieExportMovie",
                                                    path,
                                                    options));
        }

        public void MovieCreateThumbnail(string path, int size, float range)
        {
            SendCommand(new NamedSerializableObject("MovieCreateThumbnail",
                                                    path,
                                                    size,
                                                    range));
        }

        public void TardisSegmentMembranes2D(string[] paths, ProcessingOptionsTardisSegmentMembranes2D options)
        {
            SendCommand(new NamedSerializableObject("MoviesTardisSegmentMembranes2D",
                string.Join(';', paths),
                options));
        }

        public void MovieExportParticles(string path, ProcessingOptionsParticleExport options, float2[] coordinates)
        {
            SendCommand(new NamedSerializableObject("MovieExportParticles",
                                                    path,
                                                    options,
                                                    coordinates));
        }

        public void TomoStack(string path, ProcessingOptionsTomoStack options)
        {
            SendCommand(new NamedSerializableObject("TomoStack",
                                                    path,
                                                    options));
        }

        public void TomoAretomo(string path, ProcessingOptionsTomoAretomo options)
        {
            SendCommand(new NamedSerializableObject("TomoAretomo",
                                                    path,
                                                    options));
        }
        
        public void TomoEtomoPatchTrack(string path, ProcessingOptionsTomoEtomoPatch options)
        {
            SendCommand(new NamedSerializableObject("TomoEtomoPatchTrack",
                                                    path,
                                                    options));
        }
        
        public void TomoEtomoFiducials(string path, ProcessingOptionsTomoEtomoFiducials options)
        {
            SendCommand(new NamedSerializableObject("TomoEtomoFiducials",
                path,
                options));
        }

        public void TomoProcessCTF(string path, ProcessingOptionsMovieCTF options)
        {
            SendCommand(new NamedSerializableObject("TomoProcessCTF",
                                                    path,
                                                    options));
        }

        public void TomoAlignLocallyWithoutReferences(string path, ProcessingOptionsTomoFullReconstruction options)
        {
            SendCommand(new NamedSerializableObject("TomoAlignLocallyWithoutReferences",
                                                    path,
                                                    options));
        }

        public void TomoReconstruct(string path, ProcessingOptionsTomoFullReconstruction options)
        {
            SendCommand(new NamedSerializableObject("TomoReconstruct",
                                                    path,
                                                    options));
        }

        public void TomoMatch(string path, ProcessingOptionsTomoFullMatch options, string templatePath)
        {
            SendCommand(new NamedSerializableObject("TomoMatch",
                                                    path,
                                                    options,
                                                    templatePath));
        }

        public void TomoExportParticleSubtomos(string path, ProcessingOptionsTomoSubReconstruction options, float3[] coordinates, float3[] angles)
        {
            SendCommand(new NamedSerializableObject("TomoExportParticleSubtomos",
                                                    path,
                                                    options,
                                                    coordinates,
                                                    angles));
        }

        public void TomoExportParticleSeries(string path, ProcessingOptionsTomoSubReconstruction options, float3[] coordinates, float3[] angles, string pathsRelativeTo, string pathTableOut)
        {
            SendCommand(new NamedSerializableObject("TomoExportParticleSeries",
                                                    path,
                                                    options,
                                                    coordinates,
                                                    angles,
                                                    pathsRelativeTo,
                                                    pathTableOut));
        }

        public void MPAPrepareSpecies(string path, string stagingSave)
        {
            SendCommand(new NamedSerializableObject("MPAPrepareSpecies",
                                                    path,
                                                    stagingSave));
        }

        public void MPAPreparePopulation(string path, string stagingLoad)
        {
            SendCommand(new NamedSerializableObject("MPAPreparePopulation",
                                                    path,
                                                    stagingLoad));
        }

        public void MPARefine(string path, string workingDirectory, ProcessingOptionsMPARefine options, DataSource source)
        {
            SendCommand(new NamedSerializableObject("MPARefine",
                                                    path,
                                                    workingDirectory,
                                                    options,
                                                    source));
        }

        public void MPASaveProgress(string path)
        {
            SendCommand(new NamedSerializableObject("MPASaveProgress",
                                                    path));
        }

        public void MPAFinishSpecies(string path, string stagingDirectory, string[] progressFolders)
        {
            SendCommand(new NamedSerializableObject("MPAFinishSpecies",
                                                    path,
                                                    stagingDirectory,
                                                    progressFolders));
        }
    }

    public class WorkerConsole
    {
        string Host;
        int Port;
        HttpClient client;

        public WorkerConsole(string host, int port)
        {
            Host = host;
            Port = port;
            client = new HttpClient();
        }

        private string GetBaseUrl => $"http://{Host}:{Port}/v1/Console";

        public int GetLineCount()
        {
            HttpResponseMessage response = client.GetAsync($"{GetBaseUrl}/linecount").Result;
            response.EnsureSuccessStatusCode();

            string content = response.Content.ReadAsStringAsync().Result;
            return JsonSerializer.Deserialize<int>(content);
        }

        public List<LogEntry> GetAllLines()
        {
            HttpResponseMessage response = client.GetAsync($"{GetBaseUrl}/lines").Result;
            response.EnsureSuccessStatusCode();

            string content = response.Content.ReadAsStringAsync().Result;
            return JsonSerializer.Deserialize<List<LogEntry>>(content);
        }

        public List<LogEntry> GetLastNLines(int count)
        {
            HttpResponseMessage response = client.GetAsync($"{GetBaseUrl}/lines/last{count}").Result;
            response.EnsureSuccessStatusCode();

            string content = response.Content.ReadAsStringAsync().Result;
            return JsonSerializer.Deserialize<List<LogEntry>>(content);
        }

        public List<LogEntry> GetFirstNLines(int count)
        {
            HttpResponseMessage response = client.GetAsync($"{GetBaseUrl}/lines/first{count}").Result;
            response.EnsureSuccessStatusCode();

            string content = response.Content.ReadAsStringAsync().Result;
            return JsonSerializer.Deserialize<List<LogEntry>>(content);
        }

        public List<LogEntry> GetLinesRange(int start, int end)
        {
            HttpResponseMessage response = client.GetAsync($"{GetBaseUrl}/lines/range{start}_{end}").Result;
            response.EnsureSuccessStatusCode();

            string content = response.Content.ReadAsStringAsync().Result;
            return JsonSerializer.Deserialize<List<LogEntry>>(content);
        }

        public void Clear()
        {
            HttpResponseMessage response = client.PostAsync($"{GetBaseUrl}/clear", null).Result;
            response.EnsureSuccessStatusCode();
        }

        public void SetFileOutput(string path)
        {
            using (var content = new StringContent(Newtonsoft.Json.JsonConvert.SerializeObject(path), Encoding.UTF8, "application/json"))
            {
                HttpResponseMessage response = client.PostAsync($"{GetBaseUrl}/setfileoutput", content).Result;
                response.EnsureSuccessStatusCode();
            }
        }

        public void WriteToFile(string path)
        {
            using (var content = new StringContent(Newtonsoft.Json.JsonConvert.SerializeObject(path), Encoding.UTF8, "application/json"))
            {
                HttpResponseMessage response = client.PostAsync($"{GetBaseUrl}/writetofile", content).Result;
                response.EnsureSuccessStatusCode();
            }
        }
    }
}