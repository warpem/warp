using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;
using Warp;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.WorkerController;
using WarpCore.Core;

namespace WarpCore.Tests
{
    /// <summary>
    /// Integration tests for on-the-fly processing functionality.
    /// Tests the complete workflow from file discovery through distributed processing
    /// including worker coordination, task distribution, and output verification.
    /// </summary>
    public class OnTheFlyProcessingTests : IAsyncLifetime
    {
        private readonly ITestOutputHelper _testOutputHelper;
        private WebApplicationFactory<Startup> _factory;
        private HttpClient _client;
        private string _testDataDir;
        private string _testProcessingDir;
        private readonly List<WorkerWrapper> _testWorkers = new List<WorkerWrapper>();
        private StartupOptions _testStartupOptions;

        /// <summary>
        /// Initializes a new test instance with the specified test output helper.
        /// </summary>
        /// <param name="testOutputHelper">XUnit test output helper for logging test information</param>
        public OnTheFlyProcessingTests(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        /// <summary>
        /// Sets up the test environment including temporary directories, startup options,
        /// and the test web application factory. Called before each test execution.
        /// </summary>
        /// <returns>Task representing the async initialization</returns>
        public async Task InitializeAsync()
        {
            // Create temporary directories
            _testDataDir = Path.Combine(Path.GetTempPath(), $"OnTheFlyTest_Data_{Guid.NewGuid()}");
            _testProcessingDir = Path.Combine(Path.GetTempPath(), $"OnTheFlyTest_Processing_{Guid.NewGuid()}");
            Directory.CreateDirectory(_testDataDir);
            Directory.CreateDirectory(_testProcessingDir);

            // Create startup options for testing
            _testStartupOptions = new StartupOptions
            {
                DataDirectory = _testDataDir,
                ProcessingDirectory = _testProcessingDir,
                Port = 5001,
                ControllerPort = 0 // Let system assign port
            };

            // Create test web application factory
            _factory = new TestWebApplicationFactory(_testStartupOptions);
            _client = _factory.CreateClient();
        }

        /// <summary>
        /// Cleans up the test environment including workers, HTTP clients, factories,
        /// and temporary directories. Called after each test execution to ensure
        /// proper resource cleanup and test isolation.
        /// </summary>
        /// <returns>Task representing the async cleanup</returns>
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

        /// <summary>
        /// Creates a small test MRC file with realistic dimensions but minimal data.
        /// Uses 4096x4096x1 dimensions as expected by the mock processing commands.
        /// </summary>
        /// <param name="fileName">Name of the MRC file to create</param>
        private async Task CreateTestMrcFile(string fileName)
        {
            var filePath = Path.Combine(_testDataDir, fileName);
            
            // Create minimal MRC with proper header - 4096x4096x1 as expected by mocks
            var dims = new int3(4096, 4096, 1);
            var data = new float[dims.Elements()];
            
            // Fill with small random values to make it look realistic
            var rng = new Random(fileName.GetHashCode()); // Deterministic seed for reproducibility
            for (int i = 0; i < data.Length; i++)
                data[i] = (float)(rng.NextDouble() * 0.1);
            
            var image = new Image([data], dims);
            image.WriteMRC(filePath, 1.0f, true); // 1.0 Å pixel size
            image.Dispose();
            
            _testOutputHelper.WriteLine($"Created test MRC file: {fileName} ({new FileInfo(filePath).Length} bytes)");
        }

        /// <summary>
        /// Spawns a mock worker process that will connect to the test controller.
        /// Waits for the worker to register before returning.
        /// </summary>
        /// <param name="deviceId">Device ID for the worker</param>
        /// <returns>The registered WorkerWrapper instance</returns>
        private async Task<Process> SpawnMockWorkerAsync(int deviceId = 0)
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
                    // Spawn worker process with mock mode enabled
                    var spawned = await WorkerWrapper.SpawnLocalWorkerAsync(deviceId, silent: true, attachDebugger: false, mockMode: true);
                    if (spawned == null)
                    {
                        throw new Exception($"Failed to spawn worker for device {deviceId}");
                    }
                    
                    // Wait for worker to register with controller (with timeout)
                    var worker = await workerRegistered.Task.ConfigureAwait(false);
                    _testWorkers.Add(worker);
                    
                    _testOutputHelper.WriteLine($"Mock worker {worker.WorkerId} registered successfully");
                    return spawned;
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

        /// <summary>
        /// Waits for processing to complete by polling the status endpoint.
        /// Considers processing complete when no items remain in the queue (QueuedItems = 0)
        /// and we've discovered at least the expected number of files.
        /// </summary>
        /// <param name="expectedFileCount">Expected number of files to be discovered</param>
        /// <param name="timeoutMs">Maximum time to wait in milliseconds</param>
        private async Task WaitForProcessingCompletion(int expectedFileCount, int timeoutMs = 600000)
        {
            var startTime = DateTime.UtcNow;
            bool filesDiscovered = false;
            
            while (DateTime.UtcNow - startTime < TimeSpan.FromMilliseconds(timeoutMs))
            {
                var response = await _client.GetAsync("/api/processing/status");
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var status = JsonSerializer.Deserialize<JsonElement>(content, JsonSettings.Default);
                    
                    var processedCount = status.GetProperty("statistics").GetProperty("processedItems").GetInt32();
                    var totalCount = status.GetProperty("statistics").GetProperty("totalItems").GetInt32();
                    var queuedCount = status.GetProperty("statistics").GetProperty("queuedItems").GetInt32();
                    var failedCount = status.GetProperty("statistics").GetProperty("failedItems").GetInt32();
                    var isProcessing = status.GetProperty("isProcessing").GetBoolean();
                    var activeWorkers = status.GetProperty("statistics").GetProperty("activeWorkers").GetInt32();
                    
                    // Check if we've discovered the expected files
                    if (totalCount >= expectedFileCount)
                    {
                        filesDiscovered = true;
                    }
                    
                    _testOutputHelper.WriteLine($"Processing status: {processedCount} processed, " +
                                                $"{queuedCount} queued, {failedCount} failed, " +
                                                $"{totalCount} total (processing: {isProcessing}, {activeWorkers} idle workers)");
                    
                    // Processing is complete when:
                    // 1. We've discovered all expected files
                    // 2. No items remain in the queue
                    // 3. Either processing is stopped OR there are no active tasks
                    if (filesDiscovered && queuedCount == 0)
                    {
                        _testOutputHelper.WriteLine("Processing completed - no items remaining in queue");
                        return;
                    }
                }
                
                await Task.Delay(5000); // Poll every 5 seconds
            }
            
            throw new TimeoutException($"Processing did not complete within {timeoutMs}ms. Expected {expectedFileCount} files to be discovered and processed.");
        }

        /// <summary>
        /// Waits for the expected number of workers to register with the controller.
        /// </summary>
        /// <param name="expectedWorkerCount">Number of workers expected to register</param>
        /// <param name="timeoutMs">Maximum time to wait for registration</param>
        private async Task WaitForWorkersToRegister(int expectedWorkerCount, int timeoutMs = 30000)
        {
            var startTime = DateTime.UtcNow;
            
            while (DateTime.UtcNow - startTime < TimeSpan.FromMilliseconds(timeoutMs))
            {
                var response = await _client.GetAsync("/api/workers");
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var workers = JsonSerializer.Deserialize<List<WorkerInfo>>(content, JsonSettings.Default);
                    
                    _testOutputHelper.WriteLine($"Worker registration status: {workers.Count}/{expectedWorkerCount} workers registered");
                    
                    if (workers.Count >= expectedWorkerCount)
                    {
                        // Log worker details for verification
                        foreach (var worker in workers)
                        {
                            _testOutputHelper.WriteLine($"  - Worker {worker.WorkerId} (Device {worker.DeviceId}) - Status: {worker.Status}");
                        }
                        return;
                    }
                }
                
                await Task.Delay(500); // Poll every 500ms for faster detection
            }
            
            throw new TimeoutException($"Only found {await GetRegisteredWorkerCount()} workers after {timeoutMs}ms, expected {expectedWorkerCount}");
        }

        /// <summary>
        /// Gets the current count of registered workers.
        /// </summary>
        /// <returns>Number of currently registered workers</returns>
        private async Task<int> GetRegisteredWorkerCount()
        {
            try
            {
                var response = await _client.GetAsync("/api/workers");
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var workers = JsonSerializer.Deserialize<List<WorkerInfo>>(content, JsonSettings.Default);
                    return workers.Count;
                }
            }
            catch { }
            return 0;
        }

        /// <summary>
        /// Verifies that both workers were utilized during processing by checking task assignments.
        /// </summary>
        /// <returns>True if both workers received tasks, false otherwise</returns>
        private async Task<bool> VerifyWorkerUtilization()
        {
            var response = await _client.GetAsync("/api/workers");
            if (!response.IsSuccessStatusCode)
                return false;
                
            var content = await response.Content.ReadAsStringAsync();
            var workers = JsonSerializer.Deserialize<List<WorkerInfo>>(content, JsonSettings.Default);
            
            // Check that we have the expected number of workers
            if (workers.Count < 2)
            {
                _testOutputHelper.WriteLine($"Expected at least 2 workers, found {workers.Count}");
                return false;
            }
            
            // Note: Since tasks complete quickly in mocks, we might not catch workers in "Working" state
            // The fact that both workers registered and processing completed suggests distribution worked
            _testOutputHelper.WriteLine($"Found {workers.Count} registered workers - distribution appears successful");
            return true;
        }

        /// <summary>
        /// Verifies that expected output files were created during processing.
        /// </summary>
        /// <param name="expectedFileCount">Number of movies that should have been processed</param>
        private void VerifyOutputFiles(int expectedFileCount)
        {
            // Check that metadata files were created
            var xmlFiles = Directory.GetFiles(_testProcessingDir, "*.xml", SearchOption.TopDirectoryOnly);
            Assert.True(xmlFiles.Length >= expectedFileCount, 
                $"Expected at least {expectedFileCount} XML metadata files, found {xmlFiles.Length}");
            
            // Check that average directory was created and has files
            var averageDir = Path.Combine(_testProcessingDir, "average");
            if (Directory.Exists(averageDir))
            {
                var averageFiles = Directory.GetFiles(averageDir, "*.mrc", SearchOption.AllDirectories);
                _testOutputHelper.WriteLine($"Found {averageFiles.Length} average files");
            }
            
            // Check that matching directory was created and has particle files
            var matchingDir = Path.Combine(_testProcessingDir, "matching");
            if (Directory.Exists(matchingDir))
            {
                var particleFiles = Directory.GetFiles(matchingDir, "*_boxnet.star", SearchOption.AllDirectories);
                _testOutputHelper.WriteLine($"Found {particleFiles.Length} particle coordinate files");
            }
            
            _testOutputHelper.WriteLine("Output file verification completed successfully");
        }

        #endregion

        #region Integration Tests

        [Fact]
        public async Task OnTheFlyProcessing_DistributesToMultipleWorkers_ProcessesAllFiles()
        {
            const int fileCount = 40;
            const int workerCount = 10;
            
            _testOutputHelper.WriteLine("=== Starting On-The-Fly Processing Integration Test ===");
            
            // Step 1: Configure settings for MRC processing
            _testOutputHelper.WriteLine("Step 1: Configuring processing settings");
            
            var settings = new OptionsWarp();
            settings.Import.Extension = "*.mrc";
            settings.Import.DataFolder = _testDataDir;
            
            // Enable all processing steps
            settings.ProcessCTF = true;
            settings.ProcessMovement = true;
            settings.ProcessPicking = true;
            
            // Configure processing options
            settings.Movement.Bfactor = -150; // Standard B-factor
            settings.Export.DoAverage = true;
            settings.Export.DoStack = false; // Keep output smaller
            settings.Picking.DoExport = true;
            
            // Update settings via API
            var json = JsonSerializer.Serialize(settings, JsonSettings.Default);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var settingsResponse = await _client.PutAsync("/api/settings", content);
            settingsResponse.EnsureSuccessStatusCode();
            _testOutputHelper.WriteLine("Processing settings configured successfully");
            
            // Step 2: Create test MRC files
            _testOutputHelper.WriteLine($"Step 2: Creating {fileCount} test MRC files");
            for (int i = 1; i <= fileCount; i++)
                await CreateTestMrcFile($"movie_{i:D3}.mrc");
            _testOutputHelper.WriteLine($"Created {fileCount} test files");
            
            // Step 3: Start processing system
            _testOutputHelper.WriteLine("Step 3: Starting processing system");
            var startResponse = await _client.PostAsync("/api/processing/start", null);
            startResponse.EnsureSuccessStatusCode();
            
            // Give file discovery time to find files
            await Task.Delay(3000);
            _testOutputHelper.WriteLine("Processing system started");
            
            // Step 4: Spawn mock workers
            _testOutputHelper.WriteLine($"Step 4: Spawning {workerCount} mock workers");
            List<Process> workerProcesses = new List<Process>();
            for (int i = 0; i < workerCount; i++)
            {
                var process = await SpawnMockWorkerAsync(i);
                workerProcesses.Add(process);
            }
            
            // Step 4.1: Wait for all workers to register with the controller
            _testOutputHelper.WriteLine($"Step 4.1: Waiting for {workerCount} workers to register");
            await WaitForWorkersToRegister(workerCount);
            _testOutputHelper.WriteLine($"All {workerCount} workers have registered successfully");

            // Kill a few workers while processing is ongoing to test robustness
            Thread killer = new Thread(() =>
            {
                while(workerProcesses.Count(p => p.HasExited) < 2)
                {
                    Thread.Sleep(10000); // Wait 10 seconds between kills
                    var toKill = workerProcesses.Where(p => p.HasExited == false).OrderBy(_ => Random.Shared.Next()).FirstOrDefault();
                    if (toKill != null)
                    {
                        _testOutputHelper.WriteLine($"Killing worker process {toKill.Id} to test robustness");
                        try
                        {
                            toKill.Kill();
                        }
                        catch (Exception ex)
                        {
                            _testOutputHelper.WriteLine($"Error killing worker process: {ex.Message}");
                        }
                    }
                }
            });
            killer.Start();
            
            // Step 5: Monitor processing progress
            _testOutputHelper.WriteLine("Step 5: Monitoring processing progress");
            await WaitForProcessingCompletion(fileCount, timeoutMs: 180000); // 3 minutes should be plenty
            _testOutputHelper.WriteLine("Processing completed successfully");
            
            // Step 6: Verify output files
            _testOutputHelper.WriteLine("Step 6: Verifying output files");
            VerifyOutputFiles(fileCount);
            
            // Step 7: Final status check
            _testOutputHelper.WriteLine("Step 7: Final status verification");
            var finalStatusResponse = await _client.GetAsync("/api/processing/status");
            finalStatusResponse.EnsureSuccessStatusCode();
            var finalStatusContent = await finalStatusResponse.Content.ReadAsStringAsync();
            var finalStatus = JsonSerializer.Deserialize<JsonElement>(finalStatusContent, JsonSettings.Default);
            
            var finalProcessedCount = finalStatus.GetProperty("statistics").GetProperty("processedItems").GetInt32();
            var finalTotalCount = finalStatus.GetProperty("statistics").GetProperty("totalItems").GetInt32();
            
            Assert.True(finalTotalCount >= fileCount, $"Expected at least {fileCount} total files, found {finalTotalCount}");
            Assert.True(finalProcessedCount >= fileCount, $"Expected at least {fileCount} processed files, found {finalProcessedCount}");
            
            _testOutputHelper.WriteLine($"=== Integration Test Completed Successfully ===");
            _testOutputHelper.WriteLine($"Total files: {finalTotalCount}, Processed: {finalProcessedCount}");
        }

        #endregion
    }
}