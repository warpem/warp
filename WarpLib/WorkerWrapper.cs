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

namespace Warp
{
    public class WorkerWrapper : IDisposable
    {
        private static int NWorkers = 0;

        bool IsAlive = true;

        public int DeviceID = 0;
        public int Port;
        public string Host;

        public readonly WorkerConsole Console;

        Thread Heartbeat;

        Process Worker;

        public event EventHandler<EventArgs> WorkerDied;

        // Create new worker and automatically assign port
        public WorkerWrapper(int deviceID, bool silent = false, bool attachDebugger = false)
        {
            DeviceID = deviceID;

            int WorkerID = Interlocked.Increment(ref NWorkers);
            string PipeName = $"WarpWorkerPipe_{WorkerID}_{Process.GetCurrentProcess().Id}_{new Random().Next()}";

            Port = 0;

            Host = "localhost";

            bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            ProcessStartInfo StartInfo = null;
            if (IsWindows)
            {
                StartInfo = new ProcessStartInfo()
                {
                    FileName = Path.Combine(AppContext.BaseDirectory, "WarpWorker"),
                    CreateNoWindow = false,
                    WindowStyle = ProcessWindowStyle.Minimized,
                    Arguments = $"-d {DeviceID} -p {Port} --pipe {PipeName} {(silent ? "-s" : "")} " +
                                $"{(Debugger.IsAttached ? "--debug" : "")} " +
                                $"{(attachDebugger ? "--debug_attach" : "")}"
                };
            }
            else
            {
                // named pipes on linux don't work well on remote filesystems
                // https://github.com/warpem/warp/issues/28#issuecomment-2197168677
                string PipeDirectory = Environment.GetEnvironmentVariable("TMPDIR");
                if (string.IsNullOrEmpty(PipeDirectory))
                {
                    PipeDirectory = "/tmp";
                }
                PipeName = Path.Combine(PipeDirectory, PipeName);
                
                if (Helper.IsDebug)
                    StartInfo = new ProcessStartInfo()
                    {
                        FileName = "bash",
                        Arguments = $"-c \"{Path.Combine(AppContext.BaseDirectory, "WarpWorker")} " +
                                        $"-d {DeviceID} -p {Port} --pipe {PipeName} " +
                                        $"{(silent ? "-s" : "")} {(Debugger.IsAttached ? "--debug" : "")} " +
                                        $"{(attachDebugger ? "--debug_attach" : "")} " +
                                        $"> worker{Port}.out 2> worker{Port}.err\"",
                        UseShellExecute = false
                    };
                else
                    StartInfo = new ProcessStartInfo()
                    {
                        FileName = Path.Combine(AppContext.BaseDirectory, "WarpWorker"),
                        CreateNoWindow = false,
                        WindowStyle = ProcessWindowStyle.Minimized,
                        Arguments = $"-d {DeviceID} -p {Port} --pipe {PipeName} {(silent ? "-s" : "")} " +
                                    $"{(Debugger.IsAttached ? "--debug" : "")} " +
                                    $"{(attachDebugger ? "--debug_attach" : "")}"
                    };
            }

            Worker = new Process { StartInfo = StartInfo };
            Worker.Start();
            
            Port = ListenForPort(PipeName, 100_000);

            Stopwatch Timeout = new Stopwatch();
            Timeout.Start();
            while (true)
            {
                try
                {
                    SendPulse();
                    break;
                }
                catch { }

                if (Timeout.Elapsed.TotalSeconds > (attachDebugger ? 200 : 100))
                    throw new Exception($"Couldn't connect to newly created worker at {Host}:{Port} for 20 seconds, something must be wrong");
            }

            StartHeartbeat();

            Console = new WorkerConsole(Host, Port);
        }

        public WorkerWrapper(string host, int port)
        {
            Host = host;
            Port = port;

            Stopwatch Timeout = new Stopwatch();
            Timeout.Start();
            while (true)
            {
                try
                {
                    SendPulse();
                    break;
                }
                catch { }

                if (Timeout.Elapsed.TotalSeconds > 20)
                    throw new Exception($"Couldn't connect to worker at {Host}:{Port} for 20 seconds, something must be wrong");
            }

            StartHeartbeat();

            Console = new WorkerConsole(Host, Port);
        }

        int ListenForPort(string pipeName, int timeoutMilliseconds)
        {
            using (var server = new NamedPipeServerStream(pipeName))
            {
                var waitForConnectionTask = server.WaitForConnectionAsync();

                if (Task.WaitAny(waitForConnectionTask, Task.Delay(timeoutMilliseconds)) == 0)
                {
                    // Connection was made before timeout.
                    using (var reader = new StreamReader(server))
                    {
                        string portString = reader.ReadLine();
                        if (int.TryParse(portString, out int port))
                            return port;
                        else
                            throw new InvalidOperationException("Received invalid port number.");
                    }
                }
                else
                {
                    // Timeout occurred
                    throw new TimeoutException("Worker process did not connect within the allotted time.");
                }
            }
        }

        void StartHeartbeat()
        {
            Heartbeat = new Thread(new ThreadStart(() =>
            {
                Thread.Sleep(2000);

                while (IsAlive)
                {
                    try
                    {
                        SendPulse();
                        Thread.Sleep(1000);
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

        void SendCommand(NamedSerializableObject command)
        {
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
                    throw new Exception(Response.StatusCode + ": " + Response.Content);
            }
        }

        void SendExit()
        {
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

        void ReportDeath()
        {
            WorkerDied?.Invoke(this, null);
        }

        public void Dispose()
        {
            if (IsAlive)
            {
                Debug.WriteLine($"Told {Port} to exit");
                IsAlive = false;

                WaitAsyncTasks();
                SendExit();
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
