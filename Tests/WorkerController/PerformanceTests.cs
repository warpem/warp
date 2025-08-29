using System.Diagnostics;
using Warp;
using Xunit;
using Xunit.Abstractions;
using WorkerWrapper = Warp.Workers.WorkerWrapper;

namespace Tests.WorkerController;

public class PerformanceTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly List<WorkerWrapper> _workersToDispose = new();

    public PerformanceTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task StressTest_ShouldHandle50TasksQuickly()
    {
        // Arrange
        var worker = new WorkerWrapper(0, silent: true);
        _workersToDispose.Add(worker);
        
        await Task.Delay(2000); // Let worker initialize

        // Act
        var stopwatch = Stopwatch.StartNew();
        
        for (int i = 0; i < 50; i++)
        {
            worker.GcCollect();
        }
        
        stopwatch.Stop();

        // Assert
        Assert.True(stopwatch.ElapsedMilliseconds < 30000, // 30 seconds max
            $"50 tasks took {stopwatch.ElapsedMilliseconds}ms, which is too slow");
        
        _output.WriteLine($"Completed 50 tasks in {stopwatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"Average: {stopwatch.ElapsedMilliseconds / 50.0:F2}ms per task");
    }

    [Fact]
    public async Task MultiWorkerTest_ShouldHandleConcurrentWorkers()
    {
        // Arrange
        var workers = new List<WorkerWrapper>();
        int workerCount = Math.Min(3, GPU.GetDeviceCount());
        
        if (workerCount == 0)
        {
            _output.WriteLine("No GPU devices available, skipping multi-worker test");
            return;
        }

        // Create multiple workers
        for (int i = 0; i < workerCount; i++)
        {
            var worker = new WorkerWrapper(0, silent: true); // Use device 0 for all for testing
            workers.Add(worker);
            _workersToDispose.Add(worker);
        }

        await Task.Delay(3000); // Let all workers initialize

        // Act
        var stopwatch = Stopwatch.StartNew();
        
        // Submit tasks across all workers
        var tasks = new List<Task>();
        for (int i = 0; i < 15; i++) // 5 tasks per worker
        {
            var worker = workers[i % workers.Count];
            tasks.Add(Task.Run(() => worker.GcCollect()));
        }

        await Task.WhenAll(tasks);
        stopwatch.Stop();

        // Assert
        Assert.True(stopwatch.ElapsedMilliseconds < 20000, // 20 seconds max
            $"Multi-worker test took {stopwatch.ElapsedMilliseconds}ms, which is too slow");
        
        _output.WriteLine($"Multi-worker test with {workers.Count} workers completed in {stopwatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"Average: {stopwatch.ElapsedMilliseconds / 15.0:F2}ms per task");
    }

    [Theory]
    [InlineData(10)]
    [InlineData(25)]
    public async Task TaskBurst_ShouldHandleVariousLoadSizes(int taskCount)
    {
        // Arrange
        var worker = new WorkerWrapper(0, silent: true);
        _workersToDispose.Add(worker);
        
        await Task.Delay(2000);

        // Act
        var stopwatch = Stopwatch.StartNew();
        
        for (int i = 0; i < taskCount; i++)
        {
            worker.GcCollect();
        }
        
        stopwatch.Stop();

        // Assert
        var maxTimeMs = taskCount * 1000; // 1 second per task max
        Assert.True(stopwatch.ElapsedMilliseconds < maxTimeMs,
            $"{taskCount} tasks took {stopwatch.ElapsedMilliseconds}ms, expected < {maxTimeMs}ms");
        
        _output.WriteLine($"Task burst test: {taskCount} tasks in {stopwatch.ElapsedMilliseconds}ms");
    }

    [Fact]
    public async Task ControllerStartup_ShouldBeReasonablyFast()
    {
        // Act
        var stopwatch = Stopwatch.StartNew();
        
        var worker = new WorkerWrapper(0, silent: true);
        _workersToDispose.Add(worker);
        
        stopwatch.Stop();

        // Assert
        Assert.True(stopwatch.ElapsedMilliseconds < 5000, // 5 seconds max for startup
            $"Controller startup took {stopwatch.ElapsedMilliseconds}ms, which is too slow");
        
        _output.WriteLine($"Controller startup took {stopwatch.ElapsedMilliseconds}ms");
    }

    public void Dispose()
    {
        foreach (var worker in _workersToDispose)
        {
            try
            {
                worker.Dispose();
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Error disposing worker: {ex.Message}");
            }
        }
        _workersToDispose.Clear();
    }
}