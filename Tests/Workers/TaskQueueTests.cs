using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Warp.Tools;
using Warp.Workers.Queue;

namespace Tests.Workers;

public class TaskQueueTests : IDisposable
{
    private readonly string _root;
    private readonly QueueLayout _layout;
    private readonly TaskQueue _queue;

    public TaskQueueTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "tqtest-" + Guid.NewGuid().ToString("N"));
        _layout = new QueueLayout(_root);
        _layout.EnsureDirectories();
        _queue = new TaskQueue(_layout);
    }

    public void Dispose() => Directory.Delete(_root, true);

    private TaskItem MakeTask(string id, string stage = "")
    {
        var t = new TaskItem
        {
            TaskId = id,
            Stage = stage,
            Main = new[] { new NamedSerializableObject("MovieProcessCTF", id) },
        };
        t.ComputeInitFingerprint();
        return t;
    }

    [Fact]
    public void EnqueueWritesToPending()
    {
        _queue.Enqueue(MakeTask("0000001-a"));
        Assert.Single(Directory.GetFiles(_layout.Pending, "*.json"));
    }

    [Fact]
    public void ClaimMovesToRunningSubdir()
    {
        _queue.Enqueue(MakeTask("0000001-a"));
        TaskItem claimed = _queue.ClaimOne("local-1-gpu0");

        Assert.NotNull(claimed);
        Assert.Equal("0000001-a", claimed.TaskId);
        Assert.Empty(Directory.GetFiles(_layout.Pending, "*.json"));
        Assert.Single(Directory.GetFiles(_layout.RunningFor("local-1-gpu0"), "*.json"));
    }

    [Fact]
    public void ClaimReturnsNullWhenEmpty()
    {
        Assert.Null(_queue.ClaimOne("local-1-gpu0"));
    }

    [Fact]
    public void ClaimRespectsStageFilter()
    {
        _queue.Enqueue(MakeTask("0000001-refine", stage: "refine"));
        TaskItem claimed = _queue.ClaimOne("local-1-gpu0", allowedStages: new[] { "preprocess" });
        Assert.Null(claimed);
        Assert.Single(Directory.GetFiles(_layout.Pending, "*.json")); // left untouched
    }

    [Fact]
    public void ClaimsInSortedOrder()
    {
        _queue.Enqueue(MakeTask("0000003-c"));
        _queue.Enqueue(MakeTask("0000001-a"));
        _queue.Enqueue(MakeTask("0000002-b"));
        Assert.Equal("0000001-a", _queue.ClaimOne("w").TaskId);
        Assert.Equal("0000002-b", _queue.ClaimOne("w").TaskId);
        Assert.Equal("0000003-c", _queue.ClaimOne("w").TaskId);
    }

    [Fact]
    public void MarkDoneMovesToDoneWithResult()
    {
        _queue.Enqueue(MakeTask("0000001-a"));
        TaskItem claimed = _queue.ClaimOne("w");
        _queue.MarkDone("w", claimed, new { defocus = 1.23 });

        Assert.Single(Directory.GetFiles(_layout.Done, "*.json"));
        Assert.Empty(Directory.GetFiles(_layout.RunningFor("w"), "*.json"));
    }

    [Fact]
    public void MarkFailedMovesToFailedWithErrorAndHost()
    {
        _queue.Enqueue(MakeTask("0000001-a"));
        TaskItem claimed = _queue.ClaimOne("w");
        _queue.MarkFailed("w", claimed, "boom", hostname: "nodeA");

        string[] failed = Directory.GetFiles(_layout.Failed, "*.json");
        Assert.Single(failed);
        TaskItem back = TaskItem.FromJson(File.ReadAllText(failed[0]));
        Assert.Equal("boom", back.Error);
        Assert.Equal("nodeA", back.FailedOnHost);
    }

    [Fact]
    public void RecoverOrphansReturnsTasksToPendingAndIncrementsRetry()
    {
        _queue.Enqueue(MakeTask("0000001-a"));
        _queue.ClaimOne("w-dead");
        int recovered = _queue.RecoverOrphans("w-dead");

        Assert.Equal(1, recovered);
        string[] pend = Directory.GetFiles(_layout.Pending, "*.json");
        Assert.Single(pend);
        TaskItem back = TaskItem.FromJson(File.ReadAllText(pend[0]));
        Assert.Equal(1, back.RetryCount);
    }

    [Fact]
    public void SummaryCountsEachState()
    {
        _queue.Enqueue(MakeTask("0000001-a"));
        _queue.Enqueue(MakeTask("0000002-b"));
        _queue.MarkDone("w", _queue.ClaimOne("w"), null);

        var s = _queue.Summary();
        Assert.Equal(1, s.Pending);
        Assert.Equal(1, s.Done);
        Assert.Equal(0, s.Running);
    }

    [Fact]
    public void ConcurrentClaimsYieldNoDuplicatesAndNoLosses()
    {
        const int TaskCount = 50;
        const int ThreadCount = 8;

        for (int i = 0; i < TaskCount; i++)
            _queue.Enqueue(MakeTask($"{i:D7}-task"));

        var claimed = new ConcurrentBag<string>();
        var threads = new List<Thread>();

        for (int t = 0; t < ThreadCount; t++)
        {
            int idx = t;
            var thread = new Thread(() =>
            {
                string workerId = $"worker-{idx}";
                while (true)
                {
                    TaskItem item = _queue.ClaimOne(workerId);
                    if (item == null) break;
                    claimed.Add(item.TaskId);
                }
            });
            threads.Add(thread);
        }

        foreach (var thread in threads) thread.Start();
        foreach (var thread in threads) thread.Join();

        Assert.Equal(TaskCount, claimed.Count);
        Assert.Equal(TaskCount, new HashSet<string>(claimed).Count); // no duplicates
    }
}
