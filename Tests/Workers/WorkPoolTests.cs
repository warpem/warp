using System;
using System.IO;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;
using Xunit;

namespace Tests.Workers;

public class WorkPoolTests : IDisposable
{
    private readonly string _root;
    private readonly QueueLayout _layout;
    private readonly TaskQueue _queue;

    public WorkPoolTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wptest-" + Guid.NewGuid().ToString("N"));
        _layout = new QueueLayout(_root);
        _layout.EnsureDirectories();
        _queue = new TaskQueue(_layout);
    }
    public void Dispose() => Directory.Delete(_root, true);

    [Fact]
    public void DistributeBlocksUntilAllTerminalAndReturnsResults()
    {
        var pool = new WorkPool(_layout, _queue);

        var tasks = new[]
        {
            MakeTask("0000001-a"),
            MakeTask("0000002-b"),
        };

        // Simulate a worker draining the queue on a background thread.
        var worker = new System.Threading.Thread(() =>
        {
            int done = 0;
            while (done < 2)
            {
                var t = _queue.ClaimOne("w");
                if (t == null) { System.Threading.Thread.Sleep(20); continue; }
                _queue.MarkDone("w", t, new { ok = true });
                done++;
            }
        });
        worker.Start();

        var results = pool.Distribute(tasks, pollMs: 20);
        worker.Join();

        Assert.Equal(2, results.Count);
        Assert.True(results.ContainsKey("0000001-a"));
        Assert.Equal(WorkOutcome.Done, results["0000001-a"].Outcome);
    }

    [Fact]
    public void DistributeReportsPoisonedAsTerminal()
    {
        var pool = new WorkPool(_layout, _queue);
        var task = MakeTask("0000001-a");

        // Simulate the Scheduler having poisoned the task.
        var poison = new System.Threading.Thread(() =>
        {
            System.Threading.Thread.Sleep(30);
            task.Error = "dead";
            File.WriteAllText(Path.Combine(_layout.Poisoned, "0000001-a.json"), task.ToJson());
        });
        poison.Start();

        var results = pool.Distribute(new[] { task }, pollMs: 20);
        poison.Join();

        Assert.Single(results);
        Assert.Equal(WorkOutcome.Poisoned, results["0000001-a"].Outcome);
    }

    private static TaskItem MakeTask(string id)
    {
        var t = new TaskItem { TaskId = id, Main = new[] { new NamedSerializableObject("MovieProcessCTF", id) } };
        t.ComputeInitFingerprint();
        return t;
    }
}
