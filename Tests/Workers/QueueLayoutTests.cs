using System;
using System.IO;
using Xunit;
using Warp.Workers.Queue;

namespace Tests.Workers;

public class QueueLayoutTests
{
    [Fact]
    public void CreatesAllSubdirectories()
    {
        string root = Path.Combine(Path.GetTempPath(), "qltest-" + Guid.NewGuid().ToString("N"));
        var layout = new QueueLayout(root);
        layout.EnsureDirectories();

        Assert.True(Directory.Exists(layout.Pending));
        Assert.True(Directory.Exists(layout.Running));
        Assert.True(Directory.Exists(layout.Done));
        Assert.True(Directory.Exists(layout.Failed));
        Assert.True(Directory.Exists(layout.Poisoned));
        Assert.True(Directory.Exists(layout.Heartbeat));
        Assert.True(Directory.Exists(layout.Sick));
        Assert.True(Directory.Exists(layout.Blacklist));
        Assert.True(Directory.Exists(layout.Logs));

        Directory.Delete(root, true);
    }

    [Fact]
    public void RunningDirForWorkerIsUnderRunning()
    {
        var layout = new QueueLayout("/tmp/q");
        string wdir = layout.RunningFor("local-123-gpu0");
        Assert.Equal(Path.Combine("/tmp/q", "running", "local-123-gpu0"), wdir);
    }
}
