using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;
using Warp;
using Warp.Workers;
using Warp.Workers.WorkerController;
using WarpCore.Core;

namespace WarpCore.Tests
{
    public class TestWebApplicationFactory : WebApplicationFactory<Startup>
    {
        private readonly StartupOptions _startupOptions;

        public TestWebApplicationFactory(StartupOptions startupOptions)
        {
            _startupOptions = startupOptions;
        }

        protected override IHostBuilder CreateHostBuilder()
        {
            return Host.CreateDefaultBuilder()
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                    webBuilder.UseTestServer();
                })
                .ConfigureServices(services =>
                {
                    services.AddSingleton(_startupOptions);
                });
        }
    }

    public class WarpCoreApiTests : IAsyncLifetime
    {
        private readonly ITestOutputHelper _testOutputHelper;
        private WebApplicationFactory<Startup> _factory;
        private HttpClient _client;
        private string _testDataDir;
        private string _testProcessingDir;
        private readonly List<WorkerWrapper> _testWorkers = new List<WorkerWrapper>();
        private StartupOptions _testStartupOptions;

        public WarpCoreApiTests(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        public async Task InitializeAsync()
        {
            // Create temporary directories
            _testDataDir = Path.Combine(Path.GetTempPath(), $"WarpCoreApiTest_Data_{Guid.NewGuid()}");
            _testProcessingDir = Path.Combine(Path.GetTempPath(), $"WarpCoreApiTest_Processing_{Guid.NewGuid()}");
            Directory.CreateDirectory(_testDataDir);
            Directory.CreateDirectory(_testProcessingDir);

            // Create startup options for testing
            _testStartupOptions = new StartupOptions
            {
                DataDirectory = _testDataDir,
                ProcessingDirectory = _testProcessingDir,
                Port = 5001,
                ControllerPort = 0
            };

            // Create test web application factory with custom host builder
            _factory = new TestWebApplicationFactory(_testStartupOptions);

            _client = _factory.CreateClient();
        }

        public async Task DisposeAsync()
        {
            // Dispose all test workers
            foreach (var worker in _testWorkers)
            {
                try
                {
                    worker.Dispose();
                }
                catch (Exception ex)
                {
                    _testOutputHelper.WriteLine($"Error disposing worker: {ex.Message}");
                }
            }
            _testWorkers.Clear();

            // Clean up directories
            if (Directory.Exists(_testDataDir))
            {
                try
                {
                    Directory.Delete(_testDataDir, true);
                }
                catch { }
            }

            if (Directory.Exists(_testProcessingDir))
            {
                try
                {
                    Directory.Delete(_testProcessingDir, true);
                }
                catch { }
            }

            _client?.Dispose();
            _factory?.Dispose();
        }

        #region Helper Methods

        private async Task CreateTestMovieFile(string fileName)
        {
            var filePath = Path.Combine(_testDataDir, fileName);
            
            // Create a minimal valid file that Movie can handle
            await File.WriteAllTextAsync(filePath, "test movie data");
        }

        private async Task<WorkerWrapper> SpawnMockWorkerAsync(int deviceId = 0)
        {
            try
            {
                // Set up event handler to capture registered worker
                WorkerWrapper registeredWorker = null;
                var workerRegistered = new TaskCompletionSource<WorkerWrapper>();
                
                void OnWorkerRegistered(object sender, WorkerWrapper worker)
                {
                    if (worker.DeviceID == deviceId)
                    {
                        registeredWorker = worker;
                        workerRegistered.TrySetResult(worker);
                    }
                }
                
                WorkerWrapper.WorkerRegistered += OnWorkerRegistered;
                
                try
                {
                    // Spawn worker process
                    bool spawned = await WorkerWrapper.SpawnLocalWorkerAsync(deviceId, silent: true, attachDebugger: false, mockMode: true);
                    if (!spawned)
                    {
                        throw new Exception($"Failed to spawn worker for device {deviceId}");
                    }
                    
                    // Wait for worker to register with controller (with timeout)
                    var worker = await workerRegistered.Task.ConfigureAwait(false);
                    _testWorkers.Add(worker);
                    
                    return worker;
                }
                finally
                {
                    WorkerWrapper.WorkerRegistered -= OnWorkerRegistered;
                }
            }
            catch (Exception ex)
            {
                _testOutputHelper.WriteLine($"Failed to spawn mock worker: {ex.Message}");
                throw;
            }
        }

        private async Task WaitForFileDiscoveryAsync(int expectedFileCount, int timeoutMs = 5000)
        {
            var startTime = DateTime.UtcNow;
            while (DateTime.UtcNow - startTime < TimeSpan.FromMilliseconds(timeoutMs))
            {
                var response = await _client.GetAsync("/api/items/summary");
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var summary = JsonConvert.DeserializeObject<ProcessingSummary>(content);
                    
                    if (summary.TotalMovies >= expectedFileCount)
                        return;
                }
                
                await Task.Delay(200);
            }
            
            throw new TimeoutException($"File discovery did not find {expectedFileCount} files within {timeoutMs}ms");
        }

        #endregion

        #region Settings Tests

        [Fact]
        public async Task Settings_GetSettings_ReturnsDefaultSettings()
        {
            var response = await _client.GetAsync("/api/settings");
            
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            var settings = JsonConvert.DeserializeObject<OptionsWarp>(content);
            
            Assert.NotNull(settings);
        }

        [Fact]
        public async Task Settings_UpdateSettings_PersistsChanges()
        {
            // Get current settings
            var getResponse = await _client.GetAsync("/api/settings");
            getResponse.EnsureSuccessStatusCode();
            var currentSettings = JsonConvert.DeserializeObject<OptionsWarp>(
                await getResponse.Content.ReadAsStringAsync());

            // Modify settings
            currentSettings.ProcessCTF = !currentSettings.ProcessCTF;
            
            // Update settings
            var json = JsonConvert.SerializeObject(currentSettings);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var putResponse = await _client.PutAsync("/api/settings", content);
            
            putResponse.EnsureSuccessStatusCode();

            // Verify settings were updated
            var verifyResponse = await _client.GetAsync("/api/settings");
            verifyResponse.EnsureSuccessStatusCode();
            var updatedSettings = JsonConvert.DeserializeObject<OptionsWarp>(
                await verifyResponse.Content.ReadAsStringAsync());
            
            Assert.Equal(currentSettings.ProcessCTF, updatedSettings.ProcessCTF);
        }

        #endregion

        #region File Management Tests


        [Fact]
        public async Task Timestamp_Get_ReturnsCurrentTimestamp()
        {
            var response = await _client.GetAsync("/api/items/timestamp");
            
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            var result = JsonConvert.DeserializeObject<dynamic>(content);
            
            Assert.NotNull(result.timestamp);
        }

        #endregion

        #region Processing Control Tests

        [Fact]
        public async Task Processing_StartAndPause_UpdatesStatus()
        {
            // Start processing
            var startResponse = await _client.PostAsync("/api/processing/start", null);
            startResponse.EnsureSuccessStatusCode();

            // Check status shows processing
            var statusResponse = await _client.GetAsync("/api/processing/status");
            statusResponse.EnsureSuccessStatusCode();
            var statusContent = await statusResponse.Content.ReadAsStringAsync();
            Assert.Contains("\"isProcessing\":true", statusContent);

            // Pause processing
            var pauseResponse = await _client.PostAsync("/api/processing/pause", null);
            pauseResponse.EnsureSuccessStatusCode();

            // Check status shows paused
            statusResponse = await _client.GetAsync("/api/processing/status");
            statusResponse.EnsureSuccessStatusCode();
            statusContent = await statusResponse.Content.ReadAsStringAsync();
            Assert.Contains("\"isProcessing\":false", statusContent);
        }

        [Fact]
        public async Task ProcessingStatus_Get_ReturnsStatistics()
        {
            var response = await _client.GetAsync("/api/processing/status");
            
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            var status = JsonConvert.DeserializeObject<dynamic>(content);
            
            Assert.NotNull(status.statistics);
            Assert.NotNull(status.lastModified);
            Assert.NotNull(status.isProcessing);
        }

        #endregion

        #region Worker Management Tests

        [Fact]
        public async Task Workers_WithMockWorkers_ReturnsWorkerList()
        {
            // Spawn a mock worker
            await SpawnMockWorkerAsync(0);
            
            // Give time for worker registration
            await Task.Delay(2000);

            var response = await _client.GetAsync("/api/workers");
            
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            _testOutputHelper.WriteLine(content);
            
            var workers = JsonConvert.DeserializeObject<List<WorkerInfo>>(content);
            
            Assert.NotNull(workers);
            Assert.True(workers.Count > 0, "At least one worker should be registered");
        }

        [Fact]
        public async Task Workers_GetLogs_ReturnsWorkerLogs()
        {
            // Spawn a mock worker
            var worker = await SpawnMockWorkerAsync(0);
            await Task.Delay(2000);

            // Get worker list to find the worker ID
            var workersResponse = await _client.GetAsync("/api/workers");
            workersResponse.EnsureSuccessStatusCode();
            var workers = JsonConvert.DeserializeObject<List<WorkerInfo>>(
                await workersResponse.Content.ReadAsStringAsync());
            
            if (workers.Count > 0)
            {
                var workerId = workers[0].WorkerId;
                var logsResponse = await _client.GetAsync($"/api/workers/{workerId}/logs");
                
                logsResponse.EnsureSuccessStatusCode();
                var logsContent = await logsResponse.Content.ReadAsStringAsync();
                var logsResult = JsonConvert.DeserializeObject<dynamic>(logsContent);
                
                Assert.NotNull(logsResult.workerId);
                Assert.NotNull(logsResult.logs);
            }
        }

        #endregion

        #region Change Tracking Tests

        [Fact]
        public async Task Summary_Get_ReturnsSummaryData()
        {
            var response = await _client.GetAsync("/api/items/summary");
            
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            var summary = JsonConvert.DeserializeObject<ProcessingSummary>(content);
            
            Assert.NotNull(summary);
            Assert.True(summary.LastModified > DateTime.MinValue);
        }


        #endregion

        #region Integration Tests

        [Fact]
        public async Task EndToEnd_FileDiscoveryAndProcessing_Works()
        {
            // Spawn mock workers
            await SpawnMockWorkerAsync(0);
            await SpawnMockWorkerAsync(0);
            await Task.Delay(2000);

            // Create test files
            await CreateTestMovieFile("integration1.tiff");
            await CreateTestMovieFile("integration2.tiff");

            // Wait a bit for automatic file discovery
            await Task.Delay(2000);

            // Start processing
            await _client.PostAsync("/api/processing/start", null);
            await Task.Delay(1000);

            // Check that workers are available
            var workersResponse = await _client.GetAsync("/api/workers");
            workersResponse.EnsureSuccessStatusCode();

            // Check processing status
            var statusResponse = await _client.GetAsync("/api/processing/status");
            statusResponse.EnsureSuccessStatusCode();
            var statusContent = await statusResponse.Content.ReadAsStringAsync();
            
            Assert.Contains("\"isProcessing\":true", statusContent);

            // Pause processing
            await _client.PostAsync("/api/processing/pause", null);
            await Task.Delay(500);

            // Verify processing stopped
            statusResponse = await _client.GetAsync("/api/processing/status");
            statusResponse.EnsureSuccessStatusCode();
            statusContent = await statusResponse.Content.ReadAsStringAsync();
            Assert.Contains("\"isProcessing\":false", statusContent);
        }

        #endregion
    }
}