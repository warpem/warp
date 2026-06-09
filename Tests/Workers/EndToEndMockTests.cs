using System;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

public class EndToEndMockTests : IDisposable
{
    private readonly string _root;
    public EndToEndMockTests() { _root = Path.Combine(Path.GetTempPath(), "e2e-" + Guid.NewGuid().ToString("N")); }
    public void Dispose() { try { Directory.Delete(_root, true); } catch { } }

    [Fact(Skip = "Requires WarpWorker2 binary on disk; run manually after `dotnet build`")]
    public void MockPipelineDrainsQueue()
    {
        var layout = new QueueLayout(_root);
        layout.EnsureDirectories();
        var queue = new TaskQueue(layout);
        var pool = new WorkPool(layout, queue);

        // A small batch of mock CTF tasks. The mock handler does not touch the GPU.
        var tasks = Enumerable.Range(1, 4).Select(i =>
        {
            var t = new TaskItem
            {
                TaskId = $"{i:D7}-mockctf",
                Main = new[] { new NamedSerializableObject(
                    nameof(WorkerWrapper.MovieProcessCTF), $"movie{i}.mrc",
                    new ProcessingOptionsMovieCTF()) },
            };
            t.ComputeInitFingerprint();
            return t;
        }).ToArray();

        var provisioner = new LocalProvisioner(_root, new[] { 0 }, perDevice: 2, mock: true);
        var scheduler = new Scheduler(layout, queue, provisioner, target: 2,
            workerStallTimeoutMs: 30_000, workerStartupGraceMs: 60_000);

        // Run the scheduler on a background thread; Distribute blocks until done.
        var schedThread = new System.Threading.Thread(() => scheduler.RunToDrain(pollMs: 500)) { IsBackground = true };
        schedThread.Start();

        var results = pool.Distribute(tasks, pollMs: 200);

        Assert.Equal(4, results.Count);
        Assert.All(results.Values, r => Assert.Equal(WorkOutcome.Done, r.Outcome));
    }
}
