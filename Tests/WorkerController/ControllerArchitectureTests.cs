using Warp;
using Warp.WorkerController;
using Xunit;
using Xunit.Abstractions;

namespace Tests.WorkerController;

public class ControllerArchitectureTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly List<WorkerWrapper> _workersToDispose = new();

    public ControllerArchitectureTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task BasicWorkerCreation_ShouldStartControllerAutomatically()
    {
        // Arrange & Act
        var worker = new WorkerWrapper(0, silent: true);
        _workersToDispose.Add(worker);

        // Assert
        Assert.True(worker.Port > 0, "Controller should be running on a valid port");
        Assert.Equal(0, worker.DeviceID);
        Assert.Equal("localhost", worker.Host);
        
        _output.WriteLine($"Controller started on port {worker.Port}");
    }

    [Fact]
    public async Task WorkerCommands_ShouldExecuteSuccessfully()
    {
        // Arrange
        var worker = new WorkerWrapper(0, silent: true);
        _workersToDispose.Add(worker);
        
        // Wait for worker to initialize
        await Task.Delay(2000);

        // Act & Assert - these should not throw
        worker.GcCollect();
        worker.WaitAsyncTasks();
        
        _output.WriteLine("Worker commands executed successfully");
    }

    [Fact]
    public async Task MultipleWorkers_ShouldShareController()
    {
        // Arrange & Act
        var worker1 = new WorkerWrapper(0, silent: true);
        var worker2 = new WorkerWrapper(0, silent: true); // Same device for testing
        
        _workersToDispose.AddRange(new[] { worker1, worker2 });

        // Assert
        Assert.Equal(worker1.Port, worker2.Port); // Should share same controller port
        Assert.Equal(worker1.Host, worker2.Host);
        
        _output.WriteLine($"Both workers share controller on port {worker1.Port}");
    }

    [Fact]
    public async Task RemoteWorkerConnection_ShouldConnect()
    {
        // Arrange - Create local worker to start controller
        var localWorker = new WorkerWrapper(0, silent: true);
        _workersToDispose.Add(localWorker);
        
        await Task.Delay(1000); // Let controller start

        // Act - Connect as remote worker
        var remoteWorker = new WorkerWrapper("localhost", localWorker.Port);
        _workersToDispose.Add(remoteWorker);

        // Assert
        Assert.Equal("localhost", remoteWorker.Host);
        Assert.Equal(localWorker.Port, remoteWorker.Port);
        
        _output.WriteLine($"Remote worker connected to controller on port {localWorker.Port}");
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