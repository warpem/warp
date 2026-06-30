using System;
using System.Diagnostics;
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

    [Fact]
    public void MockPipelineDrainsQueue()
    {
        var layout = new QueueLayout(_root);
        layout.EnsureDirectories();
        var queue = new TaskQueue(layout);
        var pool = new WorkPool(layout, queue);

        // A small batch of mock CTF tasks, structured exactly like the real fs_ctf
        // tasks: a LoadStack step (mock allocates a tiny placeholder stack) followed by
        // MovieProcessCTF (mock fabricates a CTF). Neither touches the GPU.
        var tasks = Enumerable.Range(1, 4).Select(i =>
        {
            var t = new TaskItem
            {
                TaskId = $"{i:D7}-mockctf",
                Main = new[]
                {
                    new NamedSerializableObject(nameof(WorkerWrapper.LoadStack),
                        $"movie{i}.mrc", 1M, 1, true),
                    new NamedSerializableObject(nameof(WorkerWrapper.MovieProcessCTF),
                        $"movie{i}.mrc", new ProcessingOptionsMovieCTF()),
                },
            };
            t.ComputeInitFingerprint();
            return t;
        }).ToArray();

        var provisioner = new LocalProvisioner(_root, new[] { 0 }, perDevice: 2, mock: true);
        var scheduler = new Scheduler(layout, queue, provisioner, target: 2,
            workerStallTimeoutMs: 30_000, workerStartupGraceMs: 60_000);

        // Enqueue tasks BEFORE starting the scheduler so workers always find work on
        // their first claim attempt. Distribute's idempotent Enqueue skips them when
        // called below.
        pool.Enqueue(tasks);

        // Run the scheduler on a background thread; Distribute polls until all terminal.
        // Cancel the scheduler as soon as Distribute returns so the thread exits
        // promptly and doesn't spin against the (soon-to-be-deleted) temp directory.
        var schedCts = new System.Threading.CancellationTokenSource();
        var schedThread = new System.Threading.Thread(
            () => scheduler.RunToDrain(pollMs: 500, cancel: schedCts.Token)) { IsBackground = true };
        schedThread.Start();

        var results = pool.Distribute(tasks, pollMs: 200);
        schedCts.Cancel();
        schedThread.Join();

        Assert.Equal(4, results.Count);
        Assert.All(results.Values, r => Assert.Equal(WorkOutcome.Done, r.Outcome));
    }

    // Regression test for the heartbeat-during-long-task bug: the worker used to write
    // its heartbeat only at the top of the claim loop (once per task), so any task that
    // ran longer than the manager's stall timeout looked dead. The sweep would re-pend
    // the task and delete running/<wid>/ under the still-alive worker, which then crashed
    // and the task got redone. Tilt-series reconstruction (minutes per task) hit this on
    // essentially every task. With the background heartbeat ticker, a busy worker stays
    // alive: the task completes once and is never re-pended (RetryCount stays 0).
    [Fact]
    public void LongTaskIsNotSweptWhileWorkerIsAlive()
    {
        var layout = new QueueLayout(_root);
        layout.EnsureDirectories();
        var queue = new TaskQueue(layout);
        var pool = new WorkPool(layout, queue);

        // One task whose Main runs ~10 s (three mock CTF steps, each sleeps 3-4 s) —
        // comfortably longer than the 8 s stall timeout below, but the 5 s heartbeat
        // ticker keeps the worker visibly alive throughout.
        var task = new TaskItem
        {
            TaskId = "0000001-longmock",
            Main = new[]
            {
                new NamedSerializableObject(nameof(WorkerWrapper.LoadStack), "movie.mrc", 1M, 1, true),
                new NamedSerializableObject(nameof(WorkerWrapper.MovieProcessCTF), "movie.mrc", new ProcessingOptionsMovieCTF()),
                new NamedSerializableObject(nameof(WorkerWrapper.MovieProcessCTF), "movie.mrc", new ProcessingOptionsMovieCTF()),
                new NamedSerializableObject(nameof(WorkerWrapper.MovieProcessCTF), "movie.mrc", new ProcessingOptionsMovieCTF()),
            },
        };
        task.ComputeInitFingerprint();
        var tasks = new[] { task };

        var provisioner = new LocalProvisioner(_root, new[] { 0 }, perDevice: 1, mock: true);
        var scheduler = new Scheduler(layout, queue, provisioner, target: 1,
            workerStallTimeoutMs: 8_000, workerStartupGraceMs: 60_000);

        pool.Enqueue(tasks);

        var schedCts = new System.Threading.CancellationTokenSource();
        var schedThread = new System.Threading.Thread(
            () => scheduler.RunToDrain(pollMs: 500, cancel: schedCts.Token)) { IsBackground = true };
        schedThread.Start();

        var results = pool.Distribute(tasks, pollMs: 200);
        schedCts.Cancel();
        schedThread.Join();

        Assert.Single(results);
        Assert.Equal(WorkOutcome.Done, results[task.TaskId].Outcome);

        // The decisive check: a falsely-swept task would have been re-pended with
        // RetryCount incremented before being redone. A never-swept task stays at 0.
        var donePath = Path.Combine(layout.Done, task.TaskId + ".json");
        Assert.True(File.Exists(donePath));
        var doneTask = TaskItem.FromJson(File.ReadAllText(donePath));
        Assert.Equal(0, doneTask.RetryCount);
    }

    // A --persistent worker must NOT exit when the queue is empty: it keeps polling
    // for work that arrives later. Without this, an externally managed pool (Relay)
    // relaunches workers that quit before the manager has exited — causing end-of-run
    // job churn — and online processing (queue empty between bursts) is impossible.
    //
    // Spawns a real WarpWorker2 process, so it requires an arch-compatible worker
    // binary: run in Debug locally (AnyCPU). The Release build is x64-only for the
    // CUDA/cluster artifact and will not load on an arm64 dev machine.
    [Fact]
    public void PersistentWorkerSurvivesEmptyQueueAndDrainsLateTask()
    {
        var layout = new QueueLayout(_root);
        layout.EnsureDirectories();
        var queue = new TaskQueue(layout);
        var pool = new WorkPool(layout, queue);

        // Located the same way the LocalProvisioner finds it, against an empty queue.
        string exe = Path.Combine(AppContext.BaseDirectory, "WarpWorker2");
        var psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = $"-d 0 -q \"{_root}\" --mock --persistent --worker-id testpersist",
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        using var proc = Process.Start(psi);
        try
        {
            // A non-persistent worker exits "queue empty" after ~1 s (2 empty polls).
            // Give it well past that, then assert it is still alive.
            System.Threading.Thread.Sleep(4000);
            Assert.False(proc.HasExited, "persistent worker exited on empty queue");

            // Work that arrives later must still be claimed and completed.
            var t = new TaskItem
            {
                TaskId = "0000001-latemock",
                Main = new[]
                {
                    new NamedSerializableObject(nameof(WorkerWrapper.LoadStack), "m.mrc", 1M, 1, true),
                    new NamedSerializableObject(nameof(WorkerWrapper.MovieProcessMovement), "m.mrc", new ProcessingOptionsMovieMovement()),
                },
            };
            t.ComputeInitFingerprint();
            pool.Enqueue(new[] { t });

            string donePath = Path.Combine(layout.Done, t.TaskId + ".json");
            var sw = Stopwatch.StartNew();
            while (!File.Exists(donePath) && sw.ElapsedMilliseconds < 15_000)
                System.Threading.Thread.Sleep(200);

            Assert.True(File.Exists(donePath), "persistent worker did not process the late-arriving task");
            Assert.False(proc.HasExited, "persistent worker exited after draining the queue");
        }
        finally
        {
            try { if (!proc.HasExited) proc.Kill(true); } catch { }
        }
    }
}
