using System;
using System.IO;
using Warp.Workers.Queue;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

public class SchedulerTests : IDisposable
{
    private readonly string _root;
    private readonly QueueLayout _layout;
    private readonly TaskQueue _queue;

    public SchedulerTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "schedtest-" + Guid.NewGuid().ToString("N"));
        _layout = new QueueLayout(_root);
        _layout.EnsureDirectories();
        _queue = new TaskQueue(_layout);
    }
    public void Dispose() => Directory.Delete(_root, true);

    private class FakeProvisioner : IWorkerProvisioner
    {
        public int Target;
        public void EnsureWorkers(int target) => Target = target;
        public int LiveWorkerCount() => 0;
        public void Shutdown() { }
    }

    [Fact]
    public void TickSweepsStalledWorkerTasksBackToPending()
    {
        var t = new TaskItem { TaskId = "0000001-a", Main = new Warp.Tools.NamedSerializableObject[0] };
        t.ComputeInitFingerprint();
        _queue.Enqueue(t);
        _queue.ClaimOne("local-dead-gpu0");

        var sched = new Scheduler(_layout, _queue, new FakeProvisioner(),
            target: 1, workerStallTimeoutMs: 0, workerStartupGraceMs: 0);

        sched.Tick();   // should detect the heartbeat-less, past-grace worker and re-pend

        Assert.Equal(1, _queue.Summary().Pending);
        Assert.Equal(0, _queue.Summary().Running);
    }

    [Fact]
    public void IsDrainedWhenNoPendingNoRunning()
    {
        var sched = new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1);
        Assert.True(sched.IsDrained());
    }

    [Fact]
    public void NotDrainedWhilePendingExists()
    {
        var t = new TaskItem { TaskId = "0000001-a" };
        t.ComputeInitFingerprint();
        _queue.Enqueue(t);
        var sched = new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1);
        Assert.False(sched.IsDrained());
    }

    private TaskItem EnqueueClaimFail(string id, string worker, string host, string err = "boom")
    {
        var t = new TaskItem { TaskId = id, Main = new Warp.Tools.NamedSerializableObject[0] };
        t.ComputeInitFingerprint();
        _queue.Enqueue(t);
        var claimed = _queue.ClaimOne(worker);
        _queue.MarkFailed(worker, claimed, err, hostname: host);
        return t;
    }

    [Fact]
    public void ProcessFailuresRepenssBelowRetryCap()
    {
        EnqueueClaimFail("0000001-a", "w", "nodeA");
        var sched = new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1,
            failureMatrix: new FailureMatrix(hostBlacklistThreshold: 99, taskPoisonThreshold: 99, retryCap: 4));

        sched.Tick();   // ProcessFailures should re-pend (retry now 1, below cap)

        Assert.Equal(1, _queue.Summary().Pending);
        Assert.Equal(0, _queue.Summary().Failed);
        Assert.Equal(0, _queue.Summary().Poisoned);
    }

    [Fact]
    public void ProcessFailuresPoisonsAtRetryCap()
    {
        // retryCap=1: first failure already meets cap -> poison, not re-pend.
        EnqueueClaimFail("0000001-a", "w", "nodeA");
        var sched = new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1,
            failureMatrix: new FailureMatrix(hostBlacklistThreshold: 99, taskPoisonThreshold: 99, retryCap: 1));

        sched.Tick();

        Assert.Equal(1, _queue.Summary().Poisoned);
        Assert.Equal(0, _queue.Summary().Pending);
    }

    [Fact]
    public void ProcessFailuresBlacklistsBadHost()
    {
        var sched = new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1,
            failureMatrix: new FailureMatrix(hostBlacklistThreshold: 2, taskPoisonThreshold: 99, retryCap: 99));

        // Two DISTINCT tasks fail on nodeA. Stage both into failed/ before ticking:
        // the first EnqueueClaimFail leaves task1 in failed/, so the second claim
        // necessarily picks task2 (pending then holds only task2). This avoids any
        // dependency on re-pend/claim ordering across ticks.
        EnqueueClaimFail("0000001-a", "w1", "nodeA");
        EnqueueClaimFail("0000002-b", "w2", "nodeA");
        Assert.False(File.Exists(Path.Combine(_layout.Blacklist, "nodeA"))); // no marker before processing

        sched.Tick(); // processes both failures: 2 distinct tasks on nodeA -> blacklisted
        Assert.True(File.Exists(Path.Combine(_layout.Blacklist, "nodeA")));
    }

    // ── A1: pool.lock ───────────────────────────────────────────────────────

    [Fact]
    public void SecondSchedulerOnSameQueueDirThrows()
    {
        // First Scheduler acquires the lock.
        var sched1 = new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1);

        // Second Scheduler on the same dir must fail fast with a clear error.
        var ex = Assert.Throws<IOException>(() =>
            new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1));
        Assert.Contains("Another manager", ex.Message);

        // Disposing sched1 releases the lock.
        sched1.Dispose();

        // A third Scheduler can now acquire the (released) lock.
        var sched3 = new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1);
        sched3.Dispose();  // must not throw
    }

    // ── A2: failure matrix persistence ──────────────────────────────────────

    [Fact]
    public void FailureMatrixSurvivesSchedulerRestart()
    {
        var matrix = new FailureMatrix(hostBlacklistThreshold: 2, taskPoisonThreshold: 99, retryCap: 99);
        var sched1 = new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1,
            failureMatrix: matrix);

        // Record one failure on nodeA and tick so it's persisted.
        EnqueueClaimFail("0000001-a", "w1", "nodeA");
        sched1.Tick();
        sched1.Dispose();  // persist state and release the lock
        Assert.True(File.Exists(_layout.ManagerState));

        // Tick() re-pended 0000001-a back to pending/. Clear it so the next
        // EnqueueClaimFail claims 0000002-b (a distinct task), not 0000001-a again.
        foreach (var f in Directory.GetFiles(_layout.Pending, "*.json")) File.Delete(f);

        // Restart: new Scheduler loads persisted matrix.  nodeA has 1 failure so
        // far (threshold=2), so not yet blacklisted — but the data is in memory.
        var matrix2 = new FailureMatrix(hostBlacklistThreshold: 2, taskPoisonThreshold: 99, retryCap: 99);
        using var sched2 = new Scheduler(_layout, _queue, new FakeProvisioner(), target: 1,
            failureMatrix: matrix2);

        // A second failure on nodeA from a DIFFERENT task should trigger blacklist.
        EnqueueClaimFail("0000002-b", "w2", "nodeA");
        sched2.Tick();

        Assert.True(File.Exists(Path.Combine(_layout.Blacklist, "nodeA")),
            "nodeA should be blacklisted after 2 distinct failures, even across a restart");
    }
}
