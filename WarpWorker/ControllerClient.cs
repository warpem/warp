using System;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpWorker
{
    public class ControllerClient : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly string _controllerEndpoint;
        private readonly int _deviceId;
        private readonly long _freeMemoryMB;
        
        public string WorkerId { get; private set; }
        public string Token { get; private set; }
        
        private volatile bool _disposed = false;
        private Thread _pollingThread;
        private Thread _heartbeatThread;
        
        // Connection state tracking
        private volatile bool _isConnected = false;
        private int _reconnectAttempts = 0;
        private readonly int _maxReconnectAttempts = 10;
        private volatile bool _isReconnecting = false;
        
        // Failure counting for auto-shutdown
        private int _consecutiveFailures = 0;
        private readonly int _maxConsecutiveFailures = 10;
        private readonly object _failureLock = new object();
        
        // Console output tracking
        private int _lastSentConsoleLineIndex = 0;
        
        // Persistent connection mode - for external workers that should retry indefinitely until first connection
        private readonly bool _persistentConnection;
        private volatile bool _hasEverConnected = false;
        
        // Events for task execution
        public event Action<NamedSerializableObject> TaskReceived;
        public event Action<string> ErrorOccurred;
        public event Action Connected;
        public event Action Disconnected;
        public event Action<int> ReconnectAttempt;
        
        public ControllerClient(string controllerEndpoint, int deviceId, long freeMemoryMB, bool persistentConnection = false)
        {
            _controllerEndpoint = controllerEndpoint.StartsWith("http://") ? controllerEndpoint : $"http://{controllerEndpoint}";
            _deviceId = deviceId;
            _freeMemoryMB = freeMemoryMB;
            _persistentConnection = persistentConnection;
            
            // Generate worker's own token for identification
            Token = Guid.NewGuid().ToString();
            
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Accept.Clear();
            _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
            _httpClient.Timeout = TimeSpan.FromSeconds(30);
        }
        
        public async Task<bool> RegisterAsync()
        {
            int attempts = 0;
            const int maxAttempts = 5;
            
            while ((_persistentConnection || attempts < maxAttempts) && !_disposed)
            {
                try
                {
                    var request = new
                    {
                        Host = Environment.MachineName,
                        DeviceId = _deviceId,
                        FreeMemoryMB = _freeMemoryMB,
                        Token = Token // Send our self-generated token
                    };
                    
                    var json = JsonSerializer.Serialize(request);
                    var content = new StringContent(json, Encoding.UTF8, "application/json");
                    
                    var response = await _httpClient.PostAsync($"{_controllerEndpoint}/api/workers/register", content);
                    response.EnsureSuccessStatusCode();
                    
                    var responseJson = await response.Content.ReadAsStringAsync();
                    var registrationResponse = JsonSerializer.Deserialize<JsonElement>(responseJson);
                    
                    WorkerId = registrationResponse.GetProperty("workerId").GetString();
                    // Keep our self-generated token
                    
                    Console.WriteLine($"Successfully registered with controller as worker {WorkerId} (token: {Token})");
                    
                    _isConnected = true;
                    _hasEverConnected = true;
                    _reconnectAttempts = 0;
                    ResetFailureCount(); // Reset on successful registration
                    Connected?.Invoke();
                    
                    // Start polling and heartbeat threads
                    StartPolling();
                    StartHeartbeat();
                    
                    return true;
                }
                catch (Exception ex)
                {
                    attempts++;
                    
                    if (_persistentConnection)
                    {
                        Console.WriteLine($"Failed to register with controller (attempt {attempts}, will retry indefinitely): {ex.Message}");
                    }
                    else
                    {
                        Console.WriteLine($"Failed to register with controller (attempt {attempts}/{maxAttempts}): {ex.Message}");
                        
                        if (attempts >= maxAttempts)
                        {
                            ErrorOccurred?.Invoke($"Registration failed after {maxAttempts} attempts: {ex.Message}");
                            return false;
                        }
                    }
                    
                    // Exponential backoff with cap
                    int delay = Math.Min(1000 * (int)Math.Pow(2, Math.Min(attempts - 1, 6)), 30000);
                    await Task.Delay(delay);
                }
            }
            
            return false;
        }
        
        private void StartPolling()
        {
            _pollingThread = new Thread(() =>
            {
                while (!_disposed)
                {
                    try
                    {
                        if (!_isConnected)
                        {
                            if (!_isReconnecting)
                            {
                                Thread reconnectThread = new Thread(() => TryReconnect()) { IsBackground = true };
                                reconnectThread.Start();
                            }
                            Thread.Sleep(5000);
                            continue;
                        }

                        var pollResponse = PollForTask();
                        
                        if (pollResponse != null)
                        {
                            ResetFailureCount(); // Reset on successful communication
                        }
                        
                        if (pollResponse?.Task != null)
                        {
                            Console.WriteLine($"Received task: {pollResponse.Task.Command.Name}");
                            
                            // Update task status to Running
                            UpdateTaskStatus(pollResponse.Task.TaskId, "Running", null, null);
                            
                            // Execute the task
                            try
                            {
                                TaskReceived?.Invoke(pollResponse.Task.Command);
                                
                                // Update task status to Completed
                                UpdateTaskStatus(pollResponse.Task.TaskId, "Completed", null, null);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Task execution failed: {ex.Message}");
                                UpdateTaskStatus(pollResponse.Task.TaskId, "Failed", ex.Message, null);
                            }
                            
                            // Poll immediately for next task
                            continue;
                        }
                        
                        // No task available, wait before polling again
                        Thread.Sleep(pollResponse?.NextPollDelayMs ?? 5000);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Polling error: {ex.Message}");
                        HandleConnectionError(ex);
                        CheckForShutdown(ex, "polling");
                        Thread.Sleep(5000);
                    }
                }
            })
            {
                IsBackground = true,
                Name = "ControllerPolling"
            };
            
            _pollingThread.Start();
        }
        
        private void StartHeartbeat()
        {
            _heartbeatThread = new Thread(() =>
            {
                while (!_disposed)
                {
                    try
                    {
                        if (_isConnected)
                        {
                            SendHeartbeat();
                            ResetFailureCount(); // Reset on successful heartbeat
                            Thread.Sleep(30000); // Heartbeat every 30 seconds
                        }
                        else
                        {
                            Thread.Sleep(5000); // Check connection state more frequently when disconnected
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Heartbeat error: {ex.Message}");
                        HandleConnectionError(ex);
                        CheckForShutdown(ex, "heartbeat");
                        Thread.Sleep(10000); // Retry in 10 seconds on error
                    }
                }
            })
            {
                IsBackground = true,
                Name = "ControllerHeartbeat"
            };
            
            _heartbeatThread.Start();
        }
        
        private PollResponseData PollForTask()
        {
            try
            {
                // Get new console lines since last poll
                var allConsoleLines = VirtualConsole.GetAllLines();
                var newConsoleLines = allConsoleLines.Skip(_lastSentConsoleLineIndex).ToList();
                
                var pollRequest = new
                {
                    ConsoleLines = newConsoleLines
                };
                
                var json = JsonSerializer.Serialize(pollRequest);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = _httpClient.PostAsync($"{_controllerEndpoint}/api/workers/{WorkerId}/poll", content).Result;
                
                if (!response.IsSuccessStatusCode)
                {
                    if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
                    {
                        // Worker not found - may need to re-register
                        throw new Exception("Worker not found - may need to re-register");
                    }
                    return null;
                }
                
                // Update index after successful poll
                _lastSentConsoleLineIndex = allConsoleLines.Count;
                
                var responseJson = response.Content.ReadAsStringAsync().Result;
                var pollResponse = JsonSerializer.Deserialize<JsonElement>(responseJson);
                
                PollResponseData result = new PollResponseData
                {
                    NextPollDelayMs = pollResponse.TryGetProperty("nextPollDelayMs", out var delay) ? delay.GetInt32() : 5000
                };
                
                if (pollResponse.TryGetProperty("task", out var taskElement) && taskElement.ValueKind != JsonValueKind.Null)
                {
                    result.Task = new TaskData
                    {
                        TaskId = taskElement.GetProperty("taskId").GetString(),
                        Command = JsonSerializer.Deserialize<NamedSerializableObject>(taskElement.GetProperty("command").GetRawText())
                    };
                }
                
                return result;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Poll request failed: {ex.Message}");
                HandleConnectionError(ex);
                CheckForShutdown(ex, "poll request");
                return null;
            }
        }
        
        private void SendHeartbeat()
        {
            try
            {
                var heartbeat = new
                {
                    Status = "Idle", // Could be dynamic based on current state
                    FreeMemoryMB = GPU.GetFreeMemory(_deviceId),
                    CurrentTaskId = (string)null
                };
                
                var json = JsonSerializer.Serialize(heartbeat);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = _httpClient.PostAsync($"{_controllerEndpoint}/api/workers/{WorkerId}/heartbeat", content).Result;
                // Don't throw on heartbeat errors, just log them
                if (!response.IsSuccessStatusCode)
                {
                    Console.WriteLine($"Heartbeat failed with status: {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Heartbeat request failed: {ex.Message}");
            }
        }
        
        private void UpdateTaskStatus(string taskId, string status, string errorMessage, object result)
        {
            try
            {
                var update = new
                {
                    Status = status,
                    ErrorMessage = errorMessage,
                    Result = result
                };
                
                var json = JsonSerializer.Serialize(update);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = _httpClient.PostAsync($"{_controllerEndpoint}/api/workers/{WorkerId}/tasks/{taskId}/status", content).Result;
                
                if (!response.IsSuccessStatusCode)
                {
                    Console.WriteLine($"Failed to update task status: {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Task status update failed: {ex.Message}");
            }
        }
        
        private void HandleConnectionError(Exception ex)
        {
            if (_isConnected)
            {
                _isConnected = false;
                Console.WriteLine($"Connection to controller lost: {ex.Message}");
                Disconnected?.Invoke();
            }
        }
        
        private void CheckForShutdown(Exception ex, string operation)
        {
            lock (_failureLock)
            {
                _consecutiveFailures++;
                Console.WriteLine($"Consecutive failures: {_consecutiveFailures}/{_maxConsecutiveFailures} (from {operation})");
                
                // Only auto-shutdown if we've successfully connected before
                // Persistent workers should keep trying until first connection
                if (_consecutiveFailures >= _maxConsecutiveFailures && _hasEverConnected)
                {
                    Console.WriteLine($"Worker shutting down after {_maxConsecutiveFailures} consecutive communication failures");
                    Console.WriteLine($"Last error: {ex.Message}");
                    
                    try
                    {
                        ErrorOccurred?.Invoke($"Worker auto-shutdown after {_maxConsecutiveFailures} failures");
                    }
                    catch { }
                    
                    Environment.Exit(1);
                }
                else if (!_hasEverConnected)
                {
                    Console.WriteLine($"Worker has never connected, will continue trying (failures: {_consecutiveFailures})");
                }
            }
        }
        
        private void ResetFailureCount()
        {
            lock (_failureLock)
            {
                if (_consecutiveFailures > 0)
                {
                    Console.WriteLine($"Communication restored, resetting failure count (was {_consecutiveFailures})");
                    _consecutiveFailures = 0;
                }
            }
        }
        
        private void TryReconnect()
        {
            if (_isReconnecting || _disposed || _isConnected)
                return;
                
            _isReconnecting = true;
            
            try
            {
                while (!_isConnected && !_disposed && _reconnectAttempts < _maxReconnectAttempts)
                {
                    _reconnectAttempts++;
                    Console.WriteLine($"Attempting to reconnect to controller (attempt {_reconnectAttempts}/{_maxReconnectAttempts})...");
                    ReconnectAttempt?.Invoke(_reconnectAttempts);
                    
                    try
                    {
                        // Test connection with a simple heartbeat
                        SendHeartbeat();
                        
                        // If heartbeat succeeds, we're reconnected
                        _isConnected = true;
                        _reconnectAttempts = 0;
                        ResetFailureCount(); // Reset on successful reconnection
                        Console.WriteLine("Successfully reconnected to controller");
                        Connected?.Invoke();
                        break;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Reconnection attempt {_reconnectAttempts} failed: {ex.Message}");
                        
                        // Exponential backoff with max delay
                        int delay = Math.Min(1000 * (int)Math.Pow(2, _reconnectAttempts - 1), 60000);
                        Thread.Sleep(delay);
                    }
                }
                
                if (!_isConnected && _reconnectAttempts >= _maxReconnectAttempts)
                {
                    Console.WriteLine($"Failed to reconnect after {_maxReconnectAttempts} attempts. Giving up.");
                    ErrorOccurred?.Invoke($"Could not reconnect to controller after {_maxReconnectAttempts} attempts");
                }
            }
            finally
            {
                _isReconnecting = false;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            
            _httpClient?.Dispose();
            
            Console.WriteLine("Controller client disposed");
        }
    }
    
    internal class PollResponseData
    {
        public TaskData Task { get; set; }
        public int NextPollDelayMs { get; set; }
    }
    
    internal class TaskData
    {
        public string TaskId { get; set; }
        public NamedSerializableObject Command { get; set; }
    }
}