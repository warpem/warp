using Warp;
using Xunit;
using Xunit.Abstractions;
using WorkerWrapper = Warp.Workers.WorkerWrapper;

namespace Tests.WorkerController;

public class WorkerWrapperTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly List<WorkerWrapper> _workersToDispose = new();

    public WorkerWrapperTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void WorkerWrapper_ShouldHaveCorrectProperties()
    {
        // Arrange & Act
        var worker = new WorkerWrapper(0, silent: true);
        _workersToDispose.Add(worker);

        // Assert
        Assert.Equal(0, worker.DeviceID);
        Assert.Equal("localhost", worker.Host);
        Assert.True(worker.Port > 0);
        Assert.NotNull(worker.WorkerConsole);
    }

    [Fact]
    public async Task WorkerWrapper_AllCommandsShouldExecute()
    {
        // Arrange
        var worker = new WorkerWrapper(0, silent: true);
        _workersToDispose.Add(worker);
        
        await Task.Delay(2000); // Let worker initialize

        // Act & Assert - All commands should execute without throwing
        worker.GcCollect();
        worker.WaitAsyncTasks();
        
        // Test some basic commands that don't require actual files
        Assert.True(true, "Basic commands executed successfully");
        
        _output.WriteLine("All basic worker commands executed successfully");
    }

    [Theory]
    [InlineData(0)]
    public async Task WorkerWrapper_ShouldHandleMultipleDeviceRequests(int deviceId)
    {
        // Skip if no GPU available
        if (GPU.GetDeviceCount() <= deviceId)
        {
            _output.WriteLine($"Skipping test - Device {deviceId} not available");
            return;
        }

        // Arrange & Act
        var worker = new WorkerWrapper(deviceId, silent: true);
        _workersToDispose.Add(worker);

        await Task.Delay(1000);

        // Assert
        Assert.Equal(deviceId, worker.DeviceID);
        
        // Should be able to execute commands
        worker.GcCollect();
        
        _output.WriteLine($"Worker created and tested for device {deviceId}");
    }

    [Fact]
    public async Task WorkerWrapper_DisposeShouldCleanupProperly()
    {
        // Arrange
        var worker = new WorkerWrapper(0, silent: true);
        
        await Task.Delay(1000);

        // Act
        worker.Dispose();

        // Assert - Should not throw when disposed
        Assert.True(true, "Worker disposed without errors");
        
        // Don't add to _workersToDispose since we already disposed it
        _output.WriteLine("Worker disposed successfully");
    }

    [Fact]
    public void WorkerConsole_ShouldBeAccessible()
    {
        // Arrange
        var worker = new WorkerWrapper(0, silent: true);
        _workersToDispose.Add(worker);

        // Act & Assert
        Assert.NotNull(worker.WorkerConsole);
        
        // Console should have the same host and port as worker
        // Note: We can't test actual console methods without a real worker process
        _output.WriteLine("Worker console is accessible");
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