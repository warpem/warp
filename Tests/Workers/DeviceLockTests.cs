using System;
using System.Diagnostics;
using System.IO;
using Warp.Workers.Queue;
using Xunit;

namespace Tests.Workers;

public class DeviceLockTests : IDisposable
{
    readonly string _root = Path.Combine(Path.GetTempPath(), "devicelock-" + Guid.NewGuid().ToString("N"));
    readonly QueueLayout _layout;
    public DeviceLockTests() { _layout = new QueueLayout(_root); _layout.EnsureDirectories(); }
    public void Dispose() => Directory.Delete(_root, true);

    [Fact]
    public void OtherWorkersOnTheSameGpuWaitWhileATaskHoldsIt()
    {
        int me = Environment.ProcessId;
        using (DeviceLock.Acquire(_layout, "host", 0, me))
        {
            Assert.True(DeviceLock.IsHeldByOther(_layout, "host", 0, me + 1));
            Assert.False(DeviceLock.IsHeldByOther(_layout, "host", 0, me));
            Assert.False(DeviceLock.IsHeldByOther(_layout, "host", 1, me + 1));
            Assert.False(DeviceLock.IsHeldByOther(_layout, "other-host", 0, me + 1));
        }
        Assert.False(DeviceLock.IsHeldByOther(_layout, "host", 0, me + 1));
    }

    [Fact]
    public void ALockFromAProcessThatDiedIsCleared()
    {
        var finished = Process.Start(new ProcessStartInfo("true") { UseShellExecute = false })!;
        finished.WaitForExit();
        DeviceLock.Acquire(_layout, Environment.MachineName, 0, finished.Id); // never released: its process is gone
        Assert.False(DeviceLock.IsHeldByOther(_layout, Environment.MachineName, 0, Environment.ProcessId));
        Assert.Empty(Directory.GetFiles(_layout.Exclusive));
    }
}
