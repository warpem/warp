using System;
using Xunit;
using Warp.Tools;
using Warp.Workers.Queue;

namespace Tests.Workers;

public class TaskItemTests
{
    [Fact]
    public void RoundTripsThroughJson()
    {
        var task = new TaskItem
        {
            TaskId = "0000001-ctf-stack001",
            Stage = "preprocess",
            RequiresGpu = true,
            Init = new[] { new NamedSerializableObject("LoadStack", "movie.mrc", 1.0m, 1) },
            Main = new[] { new NamedSerializableObject("MovieProcessCTF", "movie.mrc") },
            MaxRuntimeSeconds = 3600,
            RetryCount = 0,
        };
        task.ComputeInitFingerprint();

        string json = task.ToJson();
        TaskItem back = TaskItem.FromJson(json);

        Assert.Equal(task.TaskId, back.TaskId);
        Assert.Equal(task.Stage, back.Stage);
        Assert.Equal(task.RequiresGpu, back.RequiresGpu);
        Assert.Equal(task.InitFingerprint, back.InitFingerprint);
        Assert.Equal("LoadStack", back.Init[0].Name);
        Assert.Equal("MovieProcessCTF", back.Main[0].Name);
    }

    [Fact]
    public void FingerprintIsStableAndContentSensitive()
    {
        var a = new TaskItem { Init = new[] { new NamedSerializableObject("LoadStack", "m.mrc", 1.0m, 1) } };
        var b = new TaskItem { Init = new[] { new NamedSerializableObject("LoadStack", "m.mrc", 1.0m, 1) } };
        var c = new TaskItem { Init = new[] { new NamedSerializableObject("LoadStack", "other.mrc", 1.0m, 1) } };
        a.ComputeInitFingerprint(); b.ComputeInitFingerprint(); c.ComputeInitFingerprint();

        Assert.Equal(a.InitFingerprint, b.InitFingerprint);   // same content -> same fp
        Assert.NotEqual(a.InitFingerprint, c.InitFingerprint); // different content -> different fp
    }

    [Fact]
    public void EmptyInitHasStableFingerprint()
    {
        var a = new TaskItem { Init = new NamedSerializableObject[0] };
        a.ComputeInitFingerprint();
        Assert.False(string.IsNullOrEmpty(a.InitFingerprint));
    }
}
