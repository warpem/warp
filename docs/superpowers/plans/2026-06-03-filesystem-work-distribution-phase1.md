# Filesystem Work Distribution — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the filesystem work-queue core, a new queue-consuming worker binary, the scheduler + local provisioner, and the distribution helper — then port the single `fs_ctf` (frame-series CTF) task type onto it end-to-end as the first proof.

**Architecture:** A pure-filesystem queue library in WarpLib (atomic-rename claim, sweep, heartbeats, retry/poison) is the testable core. A new worker binary lifts the `WarpCore` branch's `[Command]`/reflection dispatch and command bodies but drops all REST/controller networking — it reads commands from claimed task files instead. A scheduler maintains the worker pool via a pluggable provisioner (`Local` spawns processes, `External` is a no-op for cluster/Relay). A distribution helper enqueues task batches and blocks on results. Full design: `docs/superpowers/specs/2026-06-03-filesystem-work-distribution.md`.

**Tech Stack:** C# / .NET 10, existing `NamedSerializableObject` JSON serialization, xUnit for tests (new test project), `System.Text.Json`.

**Naming note:** The new worker project is named `WarpWorker2` throughout this plan as a **placeholder** — confirm the final name before Task 6. The queue library lives in namespace `Warp.Workers` (matching the `WarpCore` branch convention).

---

## File Structure

**New: queue library (pure filesystem, no GPU) — `WarpLib/Workers/Queue/`**
- `QueueLayout.cs` — resolves queue-dir subpaths (`pending/`, `running/<wid>/`, `done/`, `failed/`, `poisoned/`, `heartbeat/`, `sick/`, `blacklisted_nodes/`, `logs/`). One responsibility: path construction + dir creation.
- `TaskItem.cs` — the task model (`task_id`, `stage`, `requires_gpu`, `init`, `main`, `init_fingerprint`, `max_runtime_s`, `retry_count`, `created_at`) + JSON read/write + fingerprint computation.
- `TaskQueue.cs` — queue operations: `Enqueue`, `ClaimOne` (atomic rename + stage filter), `MarkDone`, `MarkFailed`, `RecoverOrphans`, `Summary`. The atomic-claim core.
- `Heartbeat.cs` — sequence-number tick writer/reader (manager `heartbeat/tick-N`, worker `running/<wid>/hb-N`), local-clock stall detection.

**New: scheduler — `WarpLib/Workers/Scheduling/`**
- `IWorkerProvisioner.cs` — provisioning strategy interface (`EnsureWorkers(target)`, `LiveWorkerCount()`).
- `LocalProvisioner.cs` — spawns/maintains local worker processes (reuses today's spawn path).
- `ExternalProvisioner.cs` — no-op (cluster: Relay provisions).
- `FailureMatrix.cs` — host×task failure sets, blacklist decision, retry-cap/poison decision.
- `Scheduler.cs` — the per-tick loop: heartbeat, sweep, failure tracking, top-up, exit-on-drain, stats/stdout.

**New: distribution helper — `WarpLib/Workers/WorkPool.cs`**
- `WorkPool.cs` — `Distribute(tasks)`: enqueue batch, block until all terminal, return results keyed by `task_id`. Builders refactored from `WorkerWrapper` command construction.

**New: worker binary — `WarpWorker2/`** (placeholder name)
- `WarpWorker2.csproj` — exe, references WarpLib + TorchSharp; **no** ASP.NET/Swashbuckle refs.
- `OptionsCLI.cs` — `--device`, `--queue-dir`, `--stages`, `--silent`, `--mock`, `--debug`, `--debug_attach`.
- `WorkerProcess.cs` — `Main`, GPU bind, health probe, claim loop, exception taxonomy, exit conditions, command dispatch (reflection registration lifted from `WarpCore`).
- `Commands/CommandAttribute.cs` — `[Command]` / `[MockCommand]` (lifted from `WarpCore`).
- `Commands/Movie.cs` — `LoadStack`, `MovieProcessCTF`, etc. command bodies (lifted/trimmed from `WarpCore`; only what `fs_ctf` needs for Phase 1).
- `Commands/Service.cs` — `GcCollect`, `SetHeaderlessParams`, `LoadGainRef`, `WaitAsyncTasks` (lifted).
- `MockCommands/MovieMock.cs` — mock CTF for GPU-free CI (lifted from `WarpCore`).
- `GpuHealthProbe.cs` — small-tensor FFT/matmul probe.

**New: test project — `Tests/`**
- `Tests.csproj` — xUnit, references WarpLib.
- `Workers/TaskItemTests.cs`, `Workers/TaskQueueTests.cs`, `Workers/HeartbeatTests.cs`, `Workers/FailureMatrixTests.cs`, `Workers/SchedulerTests.cs`, `Workers/WorkPoolTests.cs`.

**Modified (Task 6, CTF port):**
- `WarpTools/Commands/Frameseries/CTFFrameseries.cs` — switch from `WorkerWrapper`/`IterateOverItems` to `WorkPool.Distribute`. **Delete the legacy `WorkerWrapper` distribution code path for this command.**
- `Warp.sln` — add `WarpWorker2`, `Tests` projects.

---

## Task 0: Test project scaffold

**Files:**
- Create: `Tests/Tests.csproj`
- Create: `Tests/Workers/SmokeTests.cs`

- [ ] **Step 1: Create the test project file**

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>disable</Nullable>
    <IsPackable>false</IsPackable>
    <AppendTargetFrameworkToOutputPath>false</AppendTargetFrameworkToOutputPath>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.11.1" />
    <PackageReference Include="xunit" Version="2.9.2" />
    <PackageReference Include="xunit.runner.visualstudio" Version="2.8.2" />
  </ItemGroup>
  <ItemGroup>
    <ProjectReference Include="..\WarpLib\WarpLib.csproj" />
  </ItemGroup>
</Project>
```

- [ ] **Step 2: Write a smoke test**

```csharp
namespace Tests.Workers;

public class SmokeTests
{
    [Fact]
    public void TestProjectRuns()
    {
        Assert.True(true);
    }
}
```

- [ ] **Step 3: Run it to confirm the harness works**

Run: `dotnet test Tests/Tests.csproj`
Expected: PASS, 1 test passed. (First build may be slow as WarpLib compiles.)

- [ ] **Step 4: Commit**

```bash
git add Tests/Tests.csproj Tests/Workers/SmokeTests.cs
git commit -m "test: add xUnit test project for work-distribution"
```

---

## Task 1: QueueLayout — path resolution

**Files:**
- Create: `WarpLib/Workers/Queue/QueueLayout.cs`
- Test: `Tests/Workers/QueueLayoutTests.cs`

- [ ] **Step 1: Write the failing test**

```csharp
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test Tests/Tests.csproj --filter QueueLayoutTests`
Expected: FAIL — `QueueLayout` does not exist (compile error).

- [ ] **Step 3: Write minimal implementation**

```csharp
using System.IO;

namespace Warp.Workers.Queue
{
    /// <summary>
    /// Resolves the subpaths of a queue directory. Pure path construction;
    /// no I/O except EnsureDirectories(). See the work-distribution spec §4.
    /// </summary>
    public class QueueLayout
    {
        public string Root { get; }

        public QueueLayout(string root) { Root = root; }

        public string Pending   => Path.Combine(Root, "pending");
        public string Running   => Path.Combine(Root, "running");
        public string Done      => Path.Combine(Root, "done");
        public string Failed    => Path.Combine(Root, "failed");
        public string Poisoned  => Path.Combine(Root, "poisoned");
        public string Heartbeat => Path.Combine(Root, "heartbeat");
        public string Sick      => Path.Combine(Root, "sick");
        public string Blacklist => Path.Combine(Root, "blacklisted_nodes");
        public string Logs      => Path.Combine(Root, "logs");

        public string RunningFor(string workerId) => Path.Combine(Running, workerId);

        public void EnsureDirectories()
        {
            foreach (string d in new[] { Pending, Running, Done, Failed, Poisoned, Heartbeat, Sick, Blacklist, Logs })
                Directory.CreateDirectory(d);
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test Tests/Tests.csproj --filter QueueLayoutTests`
Expected: PASS, 2 tests. (`Blacklist` asserted in the first test.)

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/Queue/QueueLayout.cs Tests/Workers/QueueLayoutTests.cs
git commit -m "feat: add QueueLayout for work-distribution queue dir"
```

---

## Task 2: TaskItem — model, JSON round-trip, fingerprint

**Files:**
- Create: `WarpLib/Workers/Queue/TaskItem.cs`
- Test: `Tests/Workers/TaskItemTests.cs`

**Note:** `NamedSerializableObject` (in `Warp.Tools`) already serializes to JSON via its custom converter and is the SAME type used for REST today. We embed it directly in the task file so worker command bodies are reused unchanged (spec §5, §13).

- [ ] **Step 1: Write the failing test**

```csharp
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test Tests/Tests.csproj --filter TaskItemTests`
Expected: FAIL — `TaskItem` does not exist.

- [ ] **Step 3: Write minimal implementation**

```csharp
using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Warp.Tools;

namespace Warp.Workers.Queue
{
    /// <summary>
    /// One unit of work. Carries an init command sequence (resource loading,
    /// fingerprint-skipped by the worker) and a main command sequence. Commands
    /// are NamedSerializableObject, identical to the REST transport. Spec §5.
    /// </summary>
    public class TaskItem
    {
        public string TaskId { get; set; }
        public string Stage { get; set; } = "";
        public bool RequiresGpu { get; set; } = true;
        public NamedSerializableObject[] Init { get; set; } = Array.Empty<NamedSerializableObject>();
        public NamedSerializableObject[] Main { get; set; } = Array.Empty<NamedSerializableObject>();
        public string InitFingerprint { get; set; } = "";
        public int MaxRuntimeSeconds { get; set; } = 0;   // 0 = no self-imposed limit
        public int RetryCount { get; set; } = 0;
        public string CreatedAt { get; set; } = "";

        // Result payload (set by the worker on completion); inline scalars only.
        public object Result { get; set; }
        public string Error { get; set; }

        // Hostname the task most recently failed on. Stamped by the worker on
        // MarkFailed so the Scheduler can attribute failures to hosts for the
        // bad-node blacklist (spec §12.3).
        public string FailedOnHost { get; set; }

        private static readonly JsonSerializerOptions Opts = new() { WriteIndented = true };

        public string ToJson() => JsonSerializer.Serialize(this, Opts);

        public static TaskItem FromJson(string json) =>
            JsonSerializer.Deserialize<TaskItem>(json, Opts);

        /// <summary>SHA-256 over the serialized init array. Computed by the enqueuer.</summary>
        public void ComputeInitFingerprint()
        {
            string initJson = JsonSerializer.Serialize(Init, Opts);
            byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(initJson));
            InitFingerprint = Convert.ToHexString(hash);
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test Tests/Tests.csproj --filter TaskItemTests`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/Queue/TaskItem.cs Tests/Workers/TaskItemTests.cs
git commit -m "feat: add TaskItem model with init-fingerprint"
```

---

## Task 3: TaskQueue — enqueue, atomic claim, done/failed, sweep, summary

**Files:**
- Create: `WarpLib/Workers/Queue/TaskQueue.cs`
- Test: `Tests/Workers/TaskQueueTests.cs`

This is the core. The claim mechanism relies on POSIX `rename` atomicity (spec §11.1): the winner of a race gets the file, losers get `FileNotFoundException` and try the next candidate.

- [ ] **Step 1: Write the failing tests**

```csharp
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
        // Worker only allowed to do "preprocess" -> should not claim the refine task.
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
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test Tests/Tests.csproj --filter TaskQueueTests`
Expected: FAIL — `TaskQueue` does not exist.

- [ ] **Step 3: Write the implementation**

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Warp.Workers.Queue
{
    public readonly struct QueueSummary
    {
        public int Pending { get; init; }
        public int Running { get; init; }
        public int Done { get; init; }
        public int Failed { get; init; }
        public int Poisoned { get; init; }
    }

    /// <summary>
    /// Filesystem work queue. State transitions are atomic os.rename / File.Move
    /// within one filesystem (spec §11.1). Safe on local disk and well-configured
    /// Lustre/GPFS/NFSv4.
    /// </summary>
    public class TaskQueue
    {
        private readonly QueueLayout _layout;

        public TaskQueue(QueueLayout layout) { _layout = layout; }

        public void Enqueue(TaskItem task)
        {
            if (string.IsNullOrEmpty(task.CreatedAt))
                task.CreatedAt = DateTime.UtcNow.ToString("o");
            AtomicWrite(Path.Combine(_layout.Pending, task.TaskId + ".json"), task.ToJson());
        }

        /// <summary>
        /// Atomically claim one pending task. Honors allowedStages: tasks whose
        /// stage is not in the set are skipped (left in pending). Returns null if
        /// nothing claimable. Lost races (FileNotFoundException on Move) are
        /// retried on the next candidate.
        /// </summary>
        public TaskItem ClaimOne(string workerId, IEnumerable<string> allowedStages = null)
        {
            string wdir = _layout.RunningFor(workerId);
            Directory.CreateDirectory(wdir);

            HashSet<string> allowed = allowedStages == null ? null : new HashSet<string>(allowedStages);

            string[] candidates = Directory.GetFiles(_layout.Pending, "*.json");
            Array.Sort(candidates, StringComparer.Ordinal);

            foreach (string src in candidates)
            {
                if (allowed != null)
                {
                    string stage;
                    try { stage = TaskItem.FromJson(File.ReadAllText(src)).Stage ?? ""; }
                    catch (FileNotFoundException) { continue; } // raced
                    if (!allowed.Contains(stage))
                        continue;
                }

                string dst = Path.Combine(wdir, Path.GetFileName(src));
                try { File.Move(src, dst); }
                catch (FileNotFoundException) { continue; } // lost the race; next candidate

                return TaskItem.FromJson(File.ReadAllText(dst));
            }
            return null;
        }

        public void MarkDone(string workerId, TaskItem task, object result)
        {
            string src = Path.Combine(_layout.RunningFor(workerId), task.TaskId + ".json");
            task.Result = result;
            AtomicWrite(src, task.ToJson());
            File.Move(src, Path.Combine(_layout.Done, task.TaskId + ".json"));
        }

        public void MarkFailed(string workerId, TaskItem task, string error, string hostname = null)
        {
            string src = Path.Combine(_layout.RunningFor(workerId), task.TaskId + ".json");
            task.Error = error;
            task.FailedOnHost = hostname;
            AtomicWrite(src, task.ToJson());
            File.Move(src, Path.Combine(_layout.Failed, task.TaskId + ".json"));
        }

        /// <summary>
        /// Return every task file in running/&lt;workerId&gt;/ to pending/,
        /// incrementing RetryCount. Used by sweep and by a worker recovering its
        /// own dir at startup. Returns count recovered.
        /// </summary>
        public int RecoverOrphans(string workerId)
        {
            string wdir = _layout.RunningFor(workerId);
            if (!Directory.Exists(wdir)) return 0;

            int n = 0;
            foreach (string f in Directory.GetFiles(wdir, "*.json"))
            {
                TaskItem t = TaskItem.FromJson(File.ReadAllText(f));
                t.RetryCount++;
                string pendPath = Path.Combine(_layout.Pending, t.TaskId + ".json");
                AtomicWrite(f, t.ToJson());          // persist incremented retry first
                File.Move(f, pendPath);
                n++;
            }
            return n;
        }

        public QueueSummary Summary()
        {
            int Count(string dir) => Directory.Exists(dir) ? Directory.GetFiles(dir, "*.json").Length : 0;
            int running = 0;
            if (Directory.Exists(_layout.Running))
                foreach (string wdir in Directory.GetDirectories(_layout.Running))
                    running += Directory.GetFiles(wdir, "*.json").Length;

            return new QueueSummary
            {
                Pending = Count(_layout.Pending),
                Running = running,
                Done = Count(_layout.Done),
                Failed = Count(_layout.Failed),
                Poisoned = Count(_layout.Poisoned),
            };
        }

        private static void AtomicWrite(string path, string content)
        {
            string tmp = path + ".tmp." + Environment.ProcessId;
            File.WriteAllText(tmp, content);
            File.Move(tmp, path, overwrite: true);
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test Tests/Tests.csproj --filter TaskQueueTests`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/Queue/TaskQueue.cs Tests/Workers/TaskQueueTests.cs
git commit -m "feat: add filesystem TaskQueue with atomic claim and sweep"
```

---

## Task 4: Heartbeat — sequence-number liveness

**Files:**
- Create: `WarpLib/Workers/Queue/Heartbeat.cs`
- Test: `Tests/Workers/HeartbeatTests.cs`

No cross-node clocks: liveness is a monotonically increasing sequence number in the filename; elapsed time is measured locally by the observer (spec §8). The writer keeps only the latest tick file.

- [ ] **Step 1: Write the failing tests**

```csharp
using Warp.Workers.Queue;

namespace Tests.Workers;

public class HeartbeatTests : IDisposable
{
    private readonly string _dir;

    public HeartbeatTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "hbtest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose() => Directory.Delete(_dir, true);

    [Fact]
    public void WriteTickIncrementsAndKeepsOnlyLatest()
    {
        var hb = new HeartbeatWriter(_dir, "tick-");
        hb.WriteTick();
        hb.WriteTick();
        hb.WriteTick();

        string[] files = Directory.GetFiles(_dir, "tick-*");
        Assert.Single(files);                       // only latest kept
        Assert.Equal(3, HeartbeatReader.MaxSequence(_dir, "tick-"));
    }

    [Fact]
    public void MaxSequenceReturnsMinusOneWhenNoTicks()
    {
        Assert.Equal(-1, HeartbeatReader.MaxSequence(_dir, "tick-"));
    }

    [Fact]
    public void MonitorReportsStallWhenSequenceDoesNotAdvance()
    {
        var hb = new HeartbeatWriter(_dir, "tick-");
        hb.WriteTick();

        // timeout window of 0 ms means: any elapsed time without advance = stalled
        var mon = new HeartbeatMonitor(_dir, "tick-", timeoutMs: 0);
        mon.Observe();                  // sees seq=1 now
        System.Threading.Thread.Sleep(5);
        Assert.True(mon.IsStalled());   // seq did not advance, window exceeded
    }

    [Fact]
    public void MonitorNotStalledWhenSequenceAdvances()
    {
        var hb = new HeartbeatWriter(_dir, "tick-");
        var mon = new HeartbeatMonitor(_dir, "tick-", timeoutMs: 10_000);

        hb.WriteTick();
        mon.Observe();
        hb.WriteTick();
        mon.Observe();
        Assert.False(mon.IsStalled());
    }

    [Fact]
    public void MonitorHonorsStartupGraceWhenNoTicksYet()
    {
        // No ticks written. With a long grace, not yet stalled.
        var mon = new HeartbeatMonitor(_dir, "tick-", timeoutMs: 0, startupGraceMs: 10_000);
        mon.Observe();
        Assert.False(mon.IsStalled());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test Tests/Tests.csproj --filter HeartbeatTests`
Expected: FAIL — types do not exist.

- [ ] **Step 3: Write the implementation**

```csharp
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Warp.Workers.Queue
{
    /// <summary>
    /// Writes monotonically increasing tick files (prefix + sequence) into a
    /// directory, keeping only the latest. Used for manager and worker
    /// heartbeats. Spec §8.
    /// </summary>
    public class HeartbeatWriter
    {
        private readonly string _dir;
        private readonly string _prefix;
        private long _seq;

        public HeartbeatWriter(string dir, string prefix)
        {
            _dir = dir;
            _prefix = prefix;
            Directory.CreateDirectory(dir);
            _seq = HeartbeatReader.MaxSequence(dir, prefix); // resume after restart
            if (_seq < 0) _seq = 0;
        }

        public void WriteTick()
        {
            _seq++;
            string newPath = Path.Combine(_dir, _prefix + _seq.ToString("D12"));
            File.WriteAllText(newPath, "");
            foreach (string old in Directory.GetFiles(_dir, _prefix + "*"))
                if (!string.Equals(old, newPath, StringComparison.Ordinal))
                    try { File.Delete(old); } catch { }
        }
    }

    public static class HeartbeatReader
    {
        /// <summary>Highest sequence number present, or -1 if none.</summary>
        public static long MaxSequence(string dir, string prefix)
        {
            if (!Directory.Exists(dir)) return -1;
            long max = -1;
            foreach (string f in Directory.GetFiles(dir, prefix + "*"))
            {
                string name = Path.GetFileName(f);
                string num = name.Substring(prefix.Length);
                if (long.TryParse(num, out long v) && v > max) max = v;
            }
            return max;
        }
    }

    /// <summary>
    /// Observes a heartbeat directory and decides, using the OBSERVER's own
    /// clock, whether the writer has stalled. No cross-node clock comparison.
    /// </summary>
    public class HeartbeatMonitor
    {
        private readonly string _dir;
        private readonly string _prefix;
        private readonly long _timeoutMs;
        private readonly long _startupGraceMs;
        private readonly Stopwatch _sinceStart = Stopwatch.StartNew();

        private long _lastSeq = -1;
        private long _lastAdvanceMs;

        public HeartbeatMonitor(string dir, string prefix, long timeoutMs, long startupGraceMs = 0)
        {
            _dir = dir;
            _prefix = prefix;
            _timeoutMs = timeoutMs;
            _startupGraceMs = startupGraceMs;
            _lastAdvanceMs = 0;
        }

        public void Observe()
        {
            long seq = HeartbeatReader.MaxSequence(_dir, _prefix);
            if (seq > _lastSeq)
            {
                _lastSeq = seq;
                _lastAdvanceMs = _sinceStart.ElapsedMilliseconds;
            }
        }

        public bool IsStalled()
        {
            // Still within startup grace and never saw a tick -> alive.
            if (_lastSeq < 0)
                return _sinceStart.ElapsedMilliseconds > _startupGraceMs;

            return _sinceStart.ElapsedMilliseconds - _lastAdvanceMs > _timeoutMs;
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test Tests/Tests.csproj --filter HeartbeatTests`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/Queue/Heartbeat.cs Tests/Workers/HeartbeatTests.cs
git commit -m "feat: add sequence-number heartbeat writer/monitor"
```

---

## Task 5: FailureMatrix — bad-node blacklist & poison decisions

**Files:**
- Create: `WarpLib/Workers/Scheduling/FailureMatrix.cs`
- Test: `Tests/Workers/FailureMatrixTests.cs`

Two orthogonal signals via set-cardinality (spec §10, §12.3): a bad **host** fails many distinct tasks; a bad **task** fails on many distinct hosts. No timing heuristic.

- [ ] **Step 1: Write the failing tests**

```csharp
using Warp.Workers.Scheduling;

namespace Tests.Workers;

public class FailureMatrixTests
{
    [Fact]
    public void BlacklistsHostAfterDistinctTaskThreshold()
    {
        var m = new FailureMatrix(hostBlacklistThreshold: 3, taskPoisonThreshold: 99);
        m.RecordFailure("nodeA", "task1");
        m.RecordFailure("nodeA", "task2");
        Assert.False(m.IsHostBlacklisted("nodeA"));
        m.RecordFailure("nodeA", "task3");
        Assert.True(m.IsHostBlacklisted("nodeA"));
    }

    [Fact]
    public void DuplicateTaskOnSameHostDoesNotInflate()
    {
        var m = new FailureMatrix(hostBlacklistThreshold: 3, taskPoisonThreshold: 99);
        m.RecordFailure("nodeA", "task1");
        m.RecordFailure("nodeA", "task1");
        m.RecordFailure("nodeA", "task1");
        Assert.False(m.IsHostBlacklisted("nodeA")); // only 1 distinct task
    }

    [Fact]
    public void PoisonsTaskAfterDistinctHostThreshold()
    {
        var m = new FailureMatrix(hostBlacklistThreshold: 99, taskPoisonThreshold: 3);
        m.RecordFailure("nodeA", "task1");
        m.RecordFailure("nodeB", "task1");
        Assert.False(m.ShouldPoison("task1"));
        m.RecordFailure("nodeC", "task1");
        Assert.True(m.ShouldPoison("task1"));
    }

    [Fact]
    public void RetryCapAlsoPoisons()
    {
        var m = new FailureMatrix(hostBlacklistThreshold: 99, taskPoisonThreshold: 99, retryCap: 3);
        Assert.False(m.ShouldPoisonByRetry(2));
        Assert.True(m.ShouldPoisonByRetry(3));
        Assert.True(m.ShouldPoisonByRetry(4));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test Tests/Tests.csproj --filter FailureMatrixTests`
Expected: FAIL — `FailureMatrix` does not exist.

- [ ] **Step 3: Write the implementation**

```csharp
using System.Collections.Generic;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Tracks distinct cross-failures to separate bad hardware from bad tasks
    /// (spec §12.3) and applies the per-task retry cap (spec §10). In-memory;
    /// the Scheduler persists/reloads it via manager.state.json.
    /// </summary>
    public class FailureMatrix
    {
        private readonly Dictionary<string, HashSet<string>> _hostFailures = new(); // host -> distinct task ids
        private readonly Dictionary<string, HashSet<string>> _taskFailures = new(); // task -> distinct hostnames

        private readonly int _hostBlacklistThreshold;
        private readonly int _taskPoisonThreshold;
        private readonly int _retryCap;

        public FailureMatrix(int hostBlacklistThreshold = 4, int taskPoisonThreshold = 4, int retryCap = 4)
        {
            _hostBlacklistThreshold = hostBlacklistThreshold;
            _taskPoisonThreshold = taskPoisonThreshold;
            _retryCap = retryCap;
        }

        public void RecordFailure(string hostname, string taskId)
        {
            if (!_hostFailures.TryGetValue(hostname, out var ht)) _hostFailures[hostname] = ht = new();
            ht.Add(taskId);
            if (!_taskFailures.TryGetValue(taskId, out var th)) _taskFailures[taskId] = th = new();
            th.Add(hostname);
        }

        public bool IsHostBlacklisted(string hostname) =>
            _hostFailures.TryGetValue(hostname, out var ht) && ht.Count >= _hostBlacklistThreshold;

        public bool ShouldPoison(string taskId) =>
            _taskFailures.TryGetValue(taskId, out var th) && th.Count >= _taskPoisonThreshold;

        public bool ShouldPoisonByRetry(int retryCount) => retryCount >= _retryCap;

        public IEnumerable<string> BlacklistedHosts()
        {
            foreach (var kv in _hostFailures)
                if (kv.Value.Count >= _hostBlacklistThreshold)
                    yield return kv.Key;
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test Tests/Tests.csproj --filter FailureMatrixTests`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/Scheduling/FailureMatrix.cs Tests/Workers/FailureMatrixTests.cs
git commit -m "feat: add FailureMatrix for host blacklist and task poison"
```

---

## Task 6: New worker project — scaffold, dispatch, mock mode

**Files:**
- Create: `WarpWorker2/WarpWorker2.csproj`
- Create: `WarpWorker2/OptionsCLI.cs`
- Create: `WarpWorker2/Commands/CommandAttribute.cs`
- Create: `WarpWorker2/Commands/Service.cs`
- Create: `WarpWorker2/Commands/Movie.cs`
- Create: `WarpWorker2/MockCommands/MovieMock.cs`
- Create: `WarpWorker2/WorkerProcess.cs` (dispatch only in this task; loop added in Task 7)
- Modify: `Warp.sln`

**Confirm the project name with the user before starting** (placeholder `WarpWorker2`).

This task lifts from the `origin/WarpCore` branch. Reference commands to view the originals:
- `git show origin/WarpCore:WarpWorker/Commands/CommandAttribute.cs`
- `git show origin/WarpCore:WarpWorker/Commands/Movie.cs`
- `git show origin/WarpCore:WarpWorker/Commands/Service.cs`
- `git show origin/WarpCore:WarpWorker/MockCommands/MovieMock.cs`
- `git show origin/WarpCore:WarpWorker/WarpWorker.cs` (RegisterCommands + EvaluateCommand)

Key difference from `WarpCore`: **no ASP.NET / Swashbuckle / ControllerClient**. The dispatch table and command bodies are identical; only the feeder changes (Task 7).

- [ ] **Step 1: Create the csproj (no web refs)**

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <OutputType>Exe</OutputType>
    <PublishSingleFile>true</PublishSingleFile>
    <ServerGarbageCollection>true</ServerGarbageCollection>
    <InvariantGlobalization>true</InvariantGlobalization>
    <Version>2.0.0</Version>
    <AssemblyVersion>2.0.0</AssemblyVersion>
    <AppendTargetFrameworkToOutputPath>false</AppendTargetFrameworkToOutputPath>
  </PropertyGroup>
  <PropertyGroup Condition=" '$(Configuration)|$(Platform)' == 'Debug|AnyCPU' ">
    <OutputPath>..\bin\</OutputPath>
    <PlatformTarget>x64</PlatformTarget>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  </PropertyGroup>
  <PropertyGroup Condition=" '$(Configuration)|$(Platform)' == 'Release|AnyCPU' ">
    <OutputPath>..\Release\</OutputPath>
    <PlatformTarget>x64</PlatformTarget>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="CommandLineParser" Version="2.9.1" />
  </ItemGroup>
  <ItemGroup>
    <ProjectReference Include="..\TorchSharp\TorchSharp.csproj" />
    <ProjectReference Include="..\WarpLib\WarpLib.csproj" />
  </ItemGroup>
</Project>
```

- [ ] **Step 2: Create OptionsCLI.cs**

```csharp
using CommandLine;

namespace WarpWorker2
{
    class OptionsCLI
    {
        [Option('d', "device", Required = true, HelpText = "GPU ID used for processing")]
        public int Device { get; set; }

        [Option('q', "queue-dir", Required = true, HelpText = "Path to the shared queue directory")]
        public string QueueDir { get; set; }

        [Option("stages", HelpText = "Space-separated stages this worker may claim; empty = any")]
        public IEnumerable<string> Stages { get; set; }

        [Option("worker-id", HelpText = "Explicit worker id; if empty, derived from PID and device")]
        public string WorkerId { get; set; }

        [Option('s', "silent", HelpText = "Suppress stdout")]
        public bool Silent { get; set; }

        [Option("mock", HelpText = "Mock mode: run MockCommand handlers instead of real GPU work")]
        public bool Mock { get; set; }

        [Option("debug", HelpText = "Debug output; do not exit on heartbeat stall")]
        public bool Debug { get; set; }

        [Option("debug_attach", HelpText = "Attach a debugger to this worker")]
        public bool DebugAttach { get; set; }
    }
}
```

- [ ] **Step 3: Lift CommandAttribute.cs, Service.cs, Movie.cs, MovieMock.cs**

Copy the four files from `origin/WarpCore` into `WarpWorker2/Commands/` and
`WarpWorker2/MockCommands/`, changing the namespace to `WarpWorker2` and the
containing class to `static partial class WorkerProcess`. For Phase 1, **trim
`Movie.cs` to only the commands `fs_ctf` needs**: `LoadStack`, `MovieProcessCTF`,
plus `GcCollect` / `SetHeaderlessParams` / `LoadGainRef` / `WaitAsyncTasks` in
`Service.cs`. Leave Tomo/MPA commands out (ported later). `MovieMock.cs` keeps
`MockMovieProcessCTF`.

The command-name keys MUST use `nameof(WorkerWrapper.MovieProcessCTF)` etc. so the
WarpLib `WorkerWrapper` method names remain the single source of truth (spec §13).

- [ ] **Step 4: Create WorkerProcess.cs with dispatch registration (no loop yet)**

```csharp
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using Warp;
using Warp.Tools;

namespace WarpWorker2
{
    static partial class WorkerProcess
    {
        static readonly Dictionary<string, MethodInfo> CommandMethods = new();
        static readonly Dictionary<string, MethodInfo> MockCommandMethods = new();

        static int DeviceID = 0;
        static bool MockMode = false;
        static bool DebugMode = false;
        static bool IsSilent = false;

        // Worker resource state (loaded by init commands; survives across tasks).
        static Image GainRef = null;
        static DefectModel DefectMap = null;
        static int2 HeaderlessDims = new int2(2);
        static long HeaderlessOffset = 0;
        static string HeaderlessType = "float32";
        static Image OriginalStack = null;

        static void RegisterCommands()
        {
            var methods = typeof(WorkerProcess).GetMethods(
                BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
            foreach (var method in methods)
            {
                var cmd = method.GetCustomAttribute<CommandAttribute>();
                if (cmd != null) CommandMethods[cmd.Name] = method;
                var mock = method.GetCustomAttribute<MockCommandAttribute>();
                if (mock != null) MockCommandMethods[mock.Name] = method;
            }
        }

        /// <summary>Execute one command. Throws on unknown command or on handler failure.</summary>
        static void EvaluateCommand(NamedSerializableObject command)
        {
            GPU.SetDevice(DeviceID);
            if (string.IsNullOrWhiteSpace(command?.Name))
                throw new ArgumentException("Command name cannot be null or empty");

            if (MockMode && MockCommandMethods.TryGetValue(command.Name, out var mockMethod))
            {
                mockMethod.Invoke(null, new object[] { command });
                return;
            }

            if (CommandMethods.TryGetValue(command.Name, out var method))
                method.Invoke(null, new object[] { command });
            else
                throw new ArgumentException($"Unknown command: '{command.Name}'");
        }
    }
}
```

- [ ] **Step 5: Add a temporary Main so the project builds**

Add to `WorkerProcess.cs`:

```csharp
        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            RegisterCommands();
            Console.WriteLine($"WarpWorker2 registered {CommandMethods.Count} commands, " +
                              $"{MockCommandMethods.Count} mock commands");
        }
```

- [ ] **Step 6: Add both projects to the solution**

Run:
```bash
dotnet sln Warp.sln add WarpWorker2/WarpWorker2.csproj
dotnet sln Warp.sln add Tests/Tests.csproj
```

- [ ] **Step 7: Build to verify it compiles**

Run: `dotnet build WarpWorker2/WarpWorker2.csproj`
Expected: Build succeeded. Running the binary prints a nonzero registered-command count.

- [ ] **Step 8: Commit**

```bash
git add WarpWorker2/ Warp.sln
git commit -m "feat: scaffold WarpWorker2 with reflection command dispatch (no networking)"
```

---

## Task 7: Worker claim loop, health probe, exception taxonomy, heartbeats

**Files:**
- Create: `WarpWorker2/GpuHealthProbe.cs`
- Modify: `WarpWorker2/WorkerProcess.cs` (replace temporary Main with the real loop)

Implements spec §9 (exception taxonomy, health-probe-decides rule, init/main
asymmetry), §8.1 (worker exits on manager heartbeat stall), §8.2 (worker writes
its own heartbeat), §5.2 (fingerprint skip).

- [ ] **Step 1: Create GpuHealthProbe.cs**

```csharp
using System;
using Warp;
using Warp.Tools;

namespace WarpWorker2
{
    /// <summary>
    /// Cheap GPU sanity check that exercises the CUDA/cuFFT path real work uses.
    /// Pass => hardware healthy. Throw/false => hardware fault (spec §9.3).
    /// Used at startup and as the arbiter after any task exception.
    /// </summary>
    static class GpuHealthProbe
    {
        public static bool Probe(int deviceId)
        {
            try
            {
                GPU.SetDevice(deviceId);
                // Small round trip: allocate, FFT, copy back. Any CUDA fault throws.
                Image test = new Image(new int3(64, 64, 1));
                test.Fill(1f);
                Image ft = test.AsFFT();
                ft.Dispose();
                test.Dispose();
                GPU.CheckGPUExceptions();
                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
```

(If `Image.AsFFT()` / `GPU.CheckGPUExceptions()` differ in this codebase, use the
nearest available small-allocation + FFT + exception-check primitives — grep
`WarpLib/Image*.cs` and `WarpLib/GPU.cs`.)

- [ ] **Step 2: Replace the temporary Main with the real loop**

```csharp
        static readonly System.Diagnostics.Stopwatch UpSince = System.Diagnostics.Stopwatch.StartNew();

        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            OptionsCLI opts = null;
            CommandLine.Parser.Default.ParseArguments<OptionsCLI>(args).WithParsed(o => opts = o);
            if (opts == null) Environment.Exit(2);

            if (opts.DebugAttach && !System.Diagnostics.Debugger.IsAttached)
                System.Diagnostics.Debugger.Launch();

            DeviceID = opts.Device % GPU.GetDeviceCount();
            MockMode = opts.Mock;
            DebugMode = opts.Debug;
            IsSilent = opts.Silent;
            GPU.SetDevice(DeviceID);

            RegisterCommands();

            string workerId = string.IsNullOrEmpty(opts.WorkerId)
                ? $"local-{Environment.ProcessId}-gpu{DeviceID}"
                : opts.WorkerId;

            var layout = new Warp.Workers.Queue.QueueLayout(opts.QueueDir);
            var queue = new Warp.Workers.Queue.TaskQueue(layout);
            string wdir = layout.RunningFor(workerId);
            System.IO.Directory.CreateDirectory(wdir);

            // Record hostname for the manager's failure matrix (spec §12.3).
            System.IO.File.WriteAllText(System.IO.Path.Combine(wdir, "hostname"), Environment.MachineName);

            // Startup health probe (spec §9.3). Fail => sick + exit.
            if (!MockMode && !GpuHealthProbe.Probe(DeviceID))
            {
                MarkSick(layout, wdir, workerId, "startup health probe failed");
                Environment.Exit(3);
            }

            // Heartbeats (spec §8). Worker writes its own; monitors the manager's.
            var myHeartbeat = new Warp.Workers.Queue.HeartbeatWriter(wdir, "hb-");
            var managerMonitor = new Warp.Workers.Queue.HeartbeatMonitor(
                layout.Heartbeat, "tick-", timeoutMs: 30_000, startupGraceMs: 60_000);

            string lastFingerprint = null;
            int consecutiveEmpty = 0;

            while (true)
            {
                myHeartbeat.WriteTick();
                managerMonitor.Observe();
                if (!DebugMode && managerMonitor.IsStalled())
                {
                    WriteExit(layout, workerId, "manager heartbeat stalled");
                    return;
                }

                // Self-exclusion: if this host is blacklisted, stop claiming GPU
                // work and exit (spec §12.3). CPU-only fallback is Phase 2.
                bool hostBlacklisted = System.IO.File.Exists(
                    System.IO.Path.Combine(layout.Blacklist, Environment.MachineName));
                if (hostBlacklisted && !MockMode)
                {
                    WriteExit(layout, workerId, "host blacklisted");
                    return;
                }

                var task = queue.ClaimOne(workerId, opts.Stages != null && opts.Stages.Any() ? opts.Stages : null);
                if (task == null)
                {
                    consecutiveEmpty++;
                    if (consecutiveEmpty >= 2) { WriteExit(layout, workerId, "queue empty"); return; }
                    System.Threading.Thread.Sleep(500);
                    continue;
                }
                consecutiveEmpty = 0;

                // ---- init sequence (skip if fingerprint matches) ----
                if (task.InitFingerprint != lastFingerprint)
                {
                    try
                    {
                        foreach (var cmd in task.Init) EvaluateCommand(cmd);
                        lastFingerprint = task.InitFingerprint;
                    }
                    catch (Exception ex)
                    {
                        if (!MockMode && !GpuHealthProbe.Probe(DeviceID))
                        {
                            MarkSick(layout, wdir, workerId, "GPU fault during init: " + Flatten(ex));
                            return; // leave task in running/ for the sweep
                        }
                        // Healthy hardware, but init half-ran: reset state, clear fp, fail task, continue.
                        ResetResourceState();
                        lastFingerprint = null;
                        queue.MarkFailed(workerId, task, "init failed: " + Flatten(ex), Environment.MachineName);
                        continue;
                    }
                }

                // ---- main sequence ----
                try
                {
                    foreach (var cmd in task.Main) EvaluateCommand(cmd);
                    queue.MarkDone(workerId, task, null);
                }
                catch (Exception ex)
                {
                    if (!MockMode && !GpuHealthProbe.Probe(DeviceID))
                    {
                        MarkSick(layout, wdir, workerId, "GPU fault during main: " + Flatten(ex));
                        return; // leave task in running/ for the sweep
                    }
                    // Healthy hardware: init state intact, do NOT reset. Fail task, continue.
                    queue.MarkFailed(workerId, task, Flatten(ex), Environment.MachineName);
                }
            }
        }

        static string Flatten(Exception ex) =>
            ex is TargetInvocationException tie && tie.InnerException != null
                ? tie.InnerException.ToString() : ex.ToString();

        static void ResetResourceState()
        {
            GainRef?.Dispose(); GainRef = null;
            OriginalStack?.Dispose(); OriginalStack = null;
            DefectMap = null;
            GPU.SetDevice(DeviceID);
        }

        static void MarkSick(Warp.Workers.Queue.QueueLayout layout, string wdir, string workerId, string reason)
        {
            try { System.IO.File.WriteAllText(System.IO.Path.Combine(wdir, "sick"), Environment.MachineName + "\n" + reason); } catch { }
            try { System.IO.File.WriteAllText(System.IO.Path.Combine(layout.Sick, workerId), Environment.MachineName + "\n" + reason); } catch { }
            WriteExit(layout, workerId, "sick: " + reason);
        }

        static void WriteExit(Warp.Workers.Queue.QueueLayout layout, string workerId, string reason)
        {
            try { System.IO.File.WriteAllText(System.IO.Path.Combine(layout.Logs, workerId + ".exit"), reason); } catch { }
        }
```

- [ ] **Step 3: Build**

Run: `dotnet build WarpWorker2/WarpWorker2.csproj`
Expected: Build succeeded.

- [ ] **Step 4: Manual mock smoke test (no GPU needed)**

Run:
```bash
Q=$(mktemp -d)
mkdir -p "$Q"/{pending,running,done,failed,poisoned,heartbeat,sick,logs}
# write one mock CTF task by hand or via a tiny enqueue helper, then:
dotnet run --project WarpWorker2 -- --device 0 --queue-dir "$Q" --mock
ls "$Q/done"
```
Expected: with the manager heartbeat absent but `--mock` and startup-grace active,
the worker claims the task, runs the mock handler, moves it to `done/`, then exits
when the queue is empty (and eventually on heartbeat stall). For an automated
version of this, see Task 9 (end-to-end).

- [ ] **Step 5: Commit**

```bash
git add WarpWorker2/GpuHealthProbe.cs WarpWorker2/WorkerProcess.cs
git commit -m "feat: WarpWorker2 claim loop with health probe and exception taxonomy"
```

---

## Task 8: Scheduler + provisioners

**Files:**
- Create: `WarpLib/Workers/Scheduling/IWorkerProvisioner.cs`
- Create: `WarpLib/Workers/Scheduling/ExternalProvisioner.cs`
- Create: `WarpLib/Workers/Scheduling/LocalProvisioner.cs`
- Create: `WarpLib/Workers/Scheduling/Scheduler.cs`
- Test: `Tests/Workers/SchedulerTests.cs`

The scheduler is one code path; local vs cluster differ only by the provisioner
(spec §7). The sweep uses worker heartbeats — NEVER the cluster scheduler (spec §8).

- [ ] **Step 1: Write the provisioner interface and ExternalProvisioner**

`IWorkerProvisioner.cs`:
```csharp
namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Strategy for keeping the worker pool populated. Local mode spawns
    /// processes; cluster mode is a no-op because Relay provisions workers (spec §7).
    /// </summary>
    public interface IWorkerProvisioner
    {
        /// <summary>Ensure ~target workers exist. Implementation decides how.</summary>
        void EnsureWorkers(int target);

        /// <summary>How many workers this provisioner currently believes are alive.</summary>
        int LiveWorkerCount();

        /// <summary>Tear down any workers this provisioner owns.</summary>
        void Shutdown();
    }
}
```

`ExternalProvisioner.cs`:
```csharp
namespace Warp.Workers.Scheduling
{
    /// <summary>Cluster mode: Relay provisions workers, so this does nothing (spec §7).</summary>
    public class ExternalProvisioner : IWorkerProvisioner
    {
        public void EnsureWorkers(int target) { }
        public int LiveWorkerCount() => 0;
        public void Shutdown() { }
    }
}
```

- [ ] **Step 2: Write the LocalProvisioner**

`LocalProvisioner.cs` — spawns the new worker binary, mirroring today's spawn
path in `WarpLib/WorkerWrapper.cs:38-118` (resolve the executable next to
`AppContext.BaseDirectory`, one process per (device × perDevice), pass
`--device N --queue-dir Q`). Track `Process` handles; respawn any that have exited.

```csharp
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Local mode: maintains worker child processes pinned to the configured
    /// devices, pointing them at the queue dir. Respawns dead ones up to target.
    /// Mirrors the spawn path in WorkerWrapper (spec §7).
    /// </summary>
    public class LocalProvisioner : IWorkerProvisioner
    {
        private readonly string _queueDir;
        private readonly int[] _devices;
        private readonly int _perDevice;
        private readonly bool _mock;
        private readonly string _workerExeName;
        private readonly List<Process> _procs = new();
        private readonly object _sync = new();

        public LocalProvisioner(string queueDir, int[] devices, int perDevice,
                                bool mock = false, string workerExeName = "WarpWorker2")
        {
            _queueDir = queueDir;
            _devices = devices;
            _perDevice = perDevice;
            _mock = mock;
            _workerExeName = workerExeName;
        }

        public void EnsureWorkers(int target)
        {
            lock (_sync)
            {
                _procs.RemoveAll(p => p.HasExited);
                var slots = new List<int>();
                foreach (int dev in _devices)
                    for (int i = 0; i < _perDevice; i++)
                        slots.Add(dev);

                while (_procs.Count < Math.Min(target, slots.Count))
                {
                    int dev = slots[_procs.Count];
                    _procs.Add(Spawn(dev));
                }
            }
        }

        private Process Spawn(int device)
        {
            string exe = Path.Combine(AppContext.BaseDirectory, _workerExeName);
            string args = $"-d {device} -q \"{_queueDir}\"{(_mock ? " --mock" : "")}";
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                Arguments = args,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            var p = new Process { StartInfo = psi };
            p.Start();
            return p;
        }

        public int LiveWorkerCount()
        {
            lock (_sync) { _procs.RemoveAll(p => p.HasExited); return _procs.Count; }
        }

        public void Shutdown()
        {
            lock (_sync)
                foreach (var p in _procs)
                    try { if (!p.HasExited) p.Kill(true); } catch { }
        }
    }
}
```

- [ ] **Step 3: Write the failing Scheduler test (uses a fake provisioner, no processes)**

```csharp
using Warp.Workers.Queue;
using Warp.Workers.Scheduling;

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
        // A worker claimed a task but never wrote a heartbeat; past the grace it's stalled.
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

        EnqueueClaimFail("0000001-a", "w", "nodeA");
        sched.Tick();
        Assert.False(File.Exists(Path.Combine(_layout.Blacklist, "nodeA"))); // 1 distinct task

        EnqueueClaimFail("0000002-b", "w", "nodeA");
        sched.Tick();
        Assert.True(File.Exists(Path.Combine(_layout.Blacklist, "nodeA")));  // 2 distinct tasks -> blacklisted
    }
}
```

- [ ] **Step 4: Run to verify failure**

Run: `dotnet test Tests/Tests.csproj --filter SchedulerTests`
Expected: FAIL — `Scheduler` does not exist.

- [ ] **Step 5: Write the Scheduler**

The Scheduler is the SOLE owner of failure processing (spec §10, §12.3): each tick
it drains `failed/`, records every failure into the `FailureMatrix`, writes
blacklist markers for bad hosts, and either re-pends (below retry cap) or poisons
the task. WorkPool only watches terminal states — this keeps a single writer on
`failed/` and ensures the blacklist is live in Phase 1.

```csharp
using System;
using System.IO;
using System.Linq;
using Warp.Workers.Queue;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// One scheduler for both local and cluster mode (spec §7). Per tick:
    /// write manager heartbeat, sweep stalled workers (via worker heartbeats,
    /// never the cluster scheduler), process failures (matrix + blacklist +
    /// re-pend/poison), top up via the provisioner. Exits when the queue drains.
    /// </summary>
    public class Scheduler
    {
        private readonly QueueLayout _layout;
        private readonly TaskQueue _queue;
        private readonly IWorkerProvisioner _provisioner;
        private readonly int _target;
        private readonly long _workerStallTimeoutMs;
        private readonly long _workerStartupGraceMs;
        private readonly HeartbeatWriter _managerHeartbeat;
        private readonly FailureMatrix _failures;

        // Per-worker monitors, created lazily as worker dirs appear.
        private readonly System.Collections.Generic.Dictionary<string, HeartbeatMonitor> _monitors = new();

        public Scheduler(QueueLayout layout, TaskQueue queue, IWorkerProvisioner provisioner,
                         int target, long workerStallTimeoutMs = 30_000, long workerStartupGraceMs = 60_000,
                         FailureMatrix failureMatrix = null)
        {
            _layout = layout;
            _queue = queue;
            _provisioner = provisioner;
            _target = target;
            _workerStallTimeoutMs = workerStallTimeoutMs;
            _workerStartupGraceMs = workerStartupGraceMs;
            _failures = failureMatrix ?? new FailureMatrix();
            _managerHeartbeat = new HeartbeatWriter(layout.Heartbeat, "tick-");
        }

        public bool IsDrained()
        {
            var s = _queue.Summary();
            // failed/ is transient (drained by ProcessFailures into pending/ or
            // poisoned/), so it does not count toward "drained". A task sitting in
            // failed/ at the drain check would be processed on the next tick.
            return s.Pending == 0 && s.Running == 0 && s.Failed == 0;
        }

        public void Tick()
        {
            _managerHeartbeat.WriteTick();
            SweepStalledWorkers();
            ProcessFailures();
            _provisioner.EnsureWorkers(_target);
        }

        /// <summary>
        /// Drain failed/: record each failure into the matrix, blacklist bad hosts,
        /// then either re-pend (below retry cap) or poison the task (spec §10, §12.3).
        /// </summary>
        private void ProcessFailures()
        {
            if (!Directory.Exists(_layout.Failed)) return;

            foreach (string f in Directory.GetFiles(_layout.Failed, "*.json"))
            {
                TaskItem t;
                try { t = TaskItem.FromJson(File.ReadAllText(f)); }
                catch (FileNotFoundException) { continue; }

                if (!string.IsNullOrEmpty(t.FailedOnHost))
                {
                    _failures.RecordFailure(t.FailedOnHost, t.TaskId);
                    foreach (string host in _failures.BlacklistedHosts())
                    {
                        string marker = Path.Combine(_layout.Blacklist, host);
                        if (!File.Exists(marker))
                            try { File.WriteAllText(marker, "blacklisted"); } catch { }
                    }
                }

                if (_failures.ShouldPoison(t.TaskId) || _failures.ShouldPoisonByRetry(t.RetryCount + 1))
                {
                    File.Move(f, Path.Combine(_layout.Poisoned, t.TaskId + ".json"), overwrite: true);
                }
                else
                {
                    // Re-pend for another attempt on a (hopefully) different worker.
                    t.RetryCount++;
                    t.Error = null;
                    t.FailedOnHost = null;
                    File.Delete(f);
                    _queue.Enqueue(t);
                }
            }
        }

        private void SweepStalledWorkers()
        {
            if (!Directory.Exists(_layout.Running)) return;

            foreach (string wdir in Directory.GetDirectories(_layout.Running))
            {
                string workerId = Path.GetFileName(wdir);
                bool hasTasks = Directory.GetFiles(wdir, "*.json").Length > 0;

                if (!_monitors.TryGetValue(workerId, out var mon))
                    _monitors[workerId] = mon = new HeartbeatMonitor(
                        wdir, "hb-", _workerStallTimeoutMs, _workerStartupGraceMs);
                mon.Observe();

                if (mon.IsStalled())
                {
                    if (hasTasks)
                        _queue.RecoverOrphans(workerId);   // re-pend; sweep handles dead worker
                    else
                        TryRemoveEmptyDir(wdir);           // clean exit / sick worker, nothing to recover
                    _monitors.Remove(workerId);
                }
            }
        }

        private static void TryRemoveEmptyDir(string dir)
        {
            try
            {
                // Remove only heartbeat / hostname / sick sidecars, then the dir.
                foreach (string f in Directory.GetFiles(dir))
                    File.Delete(f);
                Directory.Delete(dir);
            }
            catch { /* not empty or racing; leave it */ }
        }

        /// <summary>Run until drained. poll = ms between ticks.</summary>
        public void RunToDrain(int pollMs = 2000)
        {
            while (true)
            {
                Tick();
                if (IsDrained()) { _provisioner.Shutdown(); return; }
                System.Threading.Thread.Sleep(pollMs);
            }
        }
    }
}
```

- [ ] **Step 6: Run to verify pass**

Run: `dotnet test Tests/Tests.csproj --filter SchedulerTests`
Expected: PASS, 6 tests (sweep, drained, not-drained, re-pend, poison-at-cap, blacklist-bad-host).

- [ ] **Step 7: Commit**

```bash
git add WarpLib/Workers/Scheduling/ Tests/Workers/SchedulerTests.cs
git commit -m "feat: add Scheduler owning failure processing, blacklist, and re-pend/poison"
```

---

## Task 9: WorkPool distribution helper

**Files:**
- Create: `WarpLib/Workers/WorkPool.cs`
- Test: `Tests/Workers/WorkPoolTests.cs`

The helper enqueues a batch and blocks until every task reaches a **terminal**
state — `done/` or `poisoned/` — returning results keyed by `task_id` (spec §6).
It assumes a Scheduler is maintaining the pool. **WorkPool does not touch
`failed/`**: a failed task is transient (the Scheduler's `ProcessFailures` either
re-pends it or moves it to `poisoned/`), so WorkPool simply keeps waiting until it
resurfaces in `done/` or `poisoned/`. This keeps a single writer on `failed/` (the
Scheduler) and avoids a race. Retry-cap, poison, and host-blacklist all live in
the Scheduler/FailureMatrix.

**Ordering requirement:** because the Scheduler owns `failed/` processing, a
`WorkPool.Distribute` call MUST run with a Scheduler ticking concurrently (as in
Task 11). The unit test below drives `done/` and `poisoned/` directly to stay
self-contained.

- [ ] **Step 1: Write the failing test (drives terminal states by hand, no worker)**

```csharp
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;

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
```

- [ ] **Step 2: Run to verify failure**

Run: `dotnet test Tests/Tests.csproj --filter WorkPoolTests`
Expected: FAIL — `WorkPool` / `WorkOutcome` do not exist.

- [ ] **Step 3: Write the implementation**

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warp.Workers.Queue;

namespace Warp.Workers
{
    // Failed is reserved but never returned by Distribute: a failed task is
    // transient (Scheduler re-pends or poisons it), so terminal results are only
    // Done or Poisoned.
    public enum WorkOutcome { Done, Failed, Poisoned }

    public class WorkResult
    {
        public string TaskId { get; init; }
        public WorkOutcome Outcome { get; init; }
        public string Error { get; init; }
    }

    /// <summary>
    /// Enqueues a batch of tasks and blocks until each reaches a TERMINAL state
    /// (done/ or poisoned/), returning results keyed by task_id (spec §6). Does
    /// not touch failed/ — the Scheduler owns retry/poison/blacklist. Assumes a
    /// Scheduler is ticking concurrently.
    /// </summary>
    public class WorkPool
    {
        private readonly QueueLayout _layout;
        private readonly TaskQueue _queue;

        public WorkPool(QueueLayout layout, TaskQueue queue)
        {
            _layout = layout;
            _queue = queue;
        }

        public Dictionary<string, WorkResult> Distribute(IEnumerable<TaskItem> tasks, int pollMs = 1000)
        {
            var ids = new HashSet<string>();
            foreach (var t in tasks)
            {
                if (string.IsNullOrEmpty(t.InitFingerprint)) t.ComputeInitFingerprint();
                _queue.Enqueue(t);
                ids.Add(t.TaskId);
            }

            var results = new Dictionary<string, WorkResult>();
            while (results.Count < ids.Count)
            {
                foreach (string id in ids)
                {
                    if (results.ContainsKey(id)) continue;

                    if (File.Exists(Path.Combine(_layout.Done, id + ".json")))
                        results[id] = new WorkResult { TaskId = id, Outcome = WorkOutcome.Done };
                    else if (File.Exists(Path.Combine(_layout.Poisoned, id + ".json")))
                    {
                        string err = null;
                        try { err = TaskItem.FromJson(
                            File.ReadAllText(Path.Combine(_layout.Poisoned, id + ".json"))).Error; }
                        catch { }
                        results[id] = new WorkResult { TaskId = id, Outcome = WorkOutcome.Poisoned, Error = err };
                    }
                    // failed/ is transient and owned by the Scheduler; keep waiting.
                }
                if (results.Count < ids.Count)
                    System.Threading.Thread.Sleep(pollMs);
            }
            return results;
        }
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `dotnet test Tests/Tests.csproj --filter WorkPoolTests`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/WorkPool.cs Tests/Workers/WorkPoolTests.cs
git commit -m "feat: add WorkPool helper that blocks on terminal (done/poisoned) states"
```

---

## Task 10: End-to-end mock integration test (no GPU)

**Files:**
- Create: `Tests/Workers/EndToEndMockTests.cs`

Proves the whole pipeline — LocalProvisioner spawns the real `WarpWorker2 --mock`
binary, the Scheduler maintains the pool + heartbeat, and WorkPool collects
results — without a GPU. This is the CI-runnable confidence check before the CTF
port.

- [ ] **Step 1: Write the test**

```csharp
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;
using Warp.Workers.Scheduling;

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
```

**Note on the Skip attribute:** the test launches the compiled `WarpWorker2`
binary via the LocalProvisioner, so it needs `dotnet build WarpWorker2` first and
the binary discoverable from the test's `AppContext.BaseDirectory`. Keep it
`Skip`-able in CI until the build wiring guarantees the binary is present; run it
manually during development. If the test project can take a build-order dependency
on `WarpWorker2` and copy the binary next to the test output, remove the Skip.

- [ ] **Step 2: Build everything, then run the test manually**

Run:
```bash
dotnet build Warp.sln
dotnet test Tests/Tests.csproj --filter EndToEndMockTests
```
Expected (manual, un-skipped): PASS — 4 mock tasks land in `done/`.

- [ ] **Step 3: Commit**

```bash
git add Tests/Workers/EndToEndMockTests.cs
git commit -m "test: add end-to-end mock pipeline test"
```

---

## Task 11: Port `fs_ctf` to the new architecture (delete legacy path)

**Files:**
- Modify: `WarpTools/Commands/Frameseries/CTFFrameseries.cs`
- Reference (do not change yet): `WarpTools/Commands/DistributedOptions.cs`, `WarpTools/Commands/BaseCommand.cs`

Today `CTFFrameseries.Run` (see `WarpTools/Commands/Frameseries/CTFFrameseries.cs:55-150`)
calls `CLI.GetWorkers()` then `IterateOverItems(...)` with a body that does
`worker.LoadStack(...)` + `worker.MovieProcessCTF(...)` + `worker.GcCollect()`.
We replace that distribution path with task-file enqueue + `WorkPool.Distribute`.
**Per the migration rule (spec §2), delete the `WorkerWrapper`-based path for this
command — do not leave both.**

- [ ] **Step 1: Add a GPU-gated end-to-end test fixture (manual)**

Create `Tests/Workers/CtfPortManualTest.md` documenting the manual acceptance run
(a tiny real dataset, `fs_ctf` via the new path, compare `*.xml` CTF estimates
against a golden run from the old path). This is a manual GPU test, not CI.

```markdown
# fs_ctf new-architecture acceptance (manual, needs GPU)

1. Build: `dotnet build Warp.sln`
2. Run fs_ctf on the standard small test set with the new path.
3. Compare resulting per-movie defocus/CTF values against a golden run
   produced by the pre-port binary (commit before Task 11).
4. Tolerance: identical within numerical noise (same algorithm, same options).
```

- [ ] **Step 2: Build the init+main task per movie and distribute**

Replace the distribution block in `CTFFrameseries.Run` (the
`WorkerWrapper[] Workers = CLI.GetWorkers();` line through the
`foreach (var worker in Workers) worker.Dispose();` block) with:

```csharp
            ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();

            // Build one task per movie: init = LoadStack (amortizable), main = MovieProcessCTF.
            var layout = new Warp.Workers.Queue.QueueLayout(
                Path.Combine(CLI.OutputProcessing, "work_ctf"));
            layout.EnsureDirectories();
            var queue = new Warp.Workers.Queue.TaskQueue(layout);
            var pool = new Warp.Workers.WorkPool(layout, queue);

            var tasks = new List<Warp.Workers.Queue.TaskItem>();
            for (int i = 0; i < CLI.InputSeries.Length; i++)
            {
                Movie m = (Movie)CLI.InputSeries[i];
                decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                bool useSum = Options.CTF.UseMovieSum && File.Exists(m.AveragePath);
                var loadStack = useSum
                    ? new NamedSerializableObject(nameof(WorkerWrapper.LoadStack),
                        m.AveragePath, 1M, Options.Import.EERGroupFrames, true)
                    : new NamedSerializableObject(nameof(WorkerWrapper.LoadStack),
                        m.DataPath, ScaleFactor, Options.Import.EERGroupFrames, true);

                var task = new Warp.Workers.Queue.TaskItem
                {
                    TaskId = $"{i:D7}-ctf-{m.RootName}",
                    Stage = "preprocess",
                    RequiresGpu = true,
                    Init = new[] { loadStack },
                    Main = new[] { new NamedSerializableObject(
                        nameof(WorkerWrapper.MovieProcessCTF), m.Path, OptionsCTF) },
                };
                task.ComputeInitFingerprint();
                tasks.Add(task);
            }

            // Maintain the local worker pool while we distribute.
            List<int> devices = (CLI.DeviceList == null || !CLI.DeviceList.Any())
                ? Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList()
                : CLI.DeviceList.ToList();
            var provisioner = new Warp.Workers.Scheduling.LocalProvisioner(
                layout.Root, devices.ToArray(), CLI.ProcessesPerDevice);
            var scheduler = new Warp.Workers.Scheduling.Scheduler(
                layout, queue, provisioner, target: devices.Count * CLI.ProcessesPerDevice);

            var schedThread = new System.Threading.Thread(() => scheduler.RunToDrain()) { IsBackground = true };
            schedThread.Start();

            var results = pool.Distribute(tasks);

            int nFailed = results.Values.Count(r => r.Outcome != Warp.Workers.WorkOutcome.Done);
            Console.WriteLine($"CTF done: {results.Count - nFailed} ok, {nFailed} failed/poisoned");
```

(Note: `LoadStack`'s 4th argument `correctGain` defaults to true in
`WorkerWrapper.LoadStack`; pass it explicitly here to match.)

- [ ] **Step 3: Delete the legacy distribution path for this command**

Remove from `CTFFrameseries.cs`: the `CLI.GetWorkers()` call, the
`IterateOverItems<Movie>(...)` block, and the worker `Dispose()` loop. Keep the
options-setup region and the final `Options.Save(...)`. Do not modify
`GetWorkers` / `IterateOverItems` themselves yet — other commands still use them
and will be ported in later phases.

- [ ] **Step 4: Build**

Run: `dotnet build WarpTools/WarpTools.csproj`
Expected: Build succeeded. (If `WorkerWrapper` is now unused in this file, its
`using` can stay — it's still in WarpLib.)

- [ ] **Step 5: Manual GPU acceptance run**

Follow `Tests/Workers/CtfPortManualTest.md`. Confirm CTF estimates match the
golden run within numerical noise.

- [ ] **Step 6: Commit**

```bash
git add WarpTools/Commands/Frameseries/CTFFrameseries.cs Tests/Workers/CtfPortManualTest.md
git commit -m "feat: port fs_ctf to filesystem work distribution; remove legacy path"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- §4 queue layout → Task 1 (QueueLayout).
- §5 task format + fingerprint → Task 2 (TaskItem).
- §6 distribution helper + two-tier results → Task 9 (WorkPool); result inline via `TaskItem.Result`.
- §7 scheduler + pluggable provisioner → Task 8.
- §8 heartbeats (both directions, sequence numbers, grace) → Task 4 (primitives), Task 7 (worker side), Task 8 (manager side + sweep).
- §9 exception taxonomy + health probe + init/main asymmetry → Task 7.
- §10 retry cap / poison → Task 5 (decision logic), Task 8 (enforcement: Scheduler's `ProcessFailures` re-pends below cap, poisons at cap).
- §11 priority via sortable id + stage filter + atomic claim → Task 3.
- §12.3 bad-node blacklist (host×task matrix, blacklist markers, worker self-exclusion) → **pulled into Phase 1**: decision in Task 5 (`FailureMatrix`), marker-writing in Task 8 (`Scheduler.ProcessFailures`), self-exclusion check in Task 7 (worker loop), `Blacklist` dir in Task 1. The rest of §12 (Relay layer) remains deferred by design.
- §13 WarpCore reuse (dispatch, command bodies, mock) → Task 6.
- Migration: one task type first + delete legacy → Task 11.

**Failure-processing ownership:** the **Scheduler** is the single writer on
`failed/` (records into `FailureMatrix`, writes blacklist markers, re-pends or
poisons). **WorkPool** only watches terminal `done/` and `poisoned/`. This avoids
a two-writer race on `failed/` and makes the blacklist live in Phase 1. Therefore
any `WorkPool.Distribute` call must have a Scheduler ticking concurrently — Task 11
does exactly this; the WorkPool unit test drives terminal states directly.

**Deferred beyond Phase 1 (not gaps — explicitly out of scope):** `pool.lock`
(§7), `manager.state.json` persistence of the failure matrix across manager
restart (§7, §12.3 — in Phase 1 the matrix is in-memory, so a manager restart
resets blacklist progress), CPU-only fallback for blacklisted workers (§12.3 — in
Phase 1 a blacklisted worker exits rather than falling back to CPU tasks),
sick-worker stdout count parsing (§12.3), `max_runtime_s` self-timeout enforcement
in the worker (§9.4), preemption SIGTERM handler (§9.4), per-worker `logs/<wid>.log`
virtual-console wiring (§4). Each should become its own follow-up task once the
single-task-type port is proven. Add a Phase 2 plan covering them before porting
additional task types that depend on them (notably the SIGTERM handler and
manager-state persistence, which matter most in a real preemptable cluster queue).

**Type consistency:** `QueueLayout` members (`Pending`/`Running`/`Failed`/
`Poisoned`/`Sick`/`Blacklist`/`RunningFor`/…), `TaskItem`
(`TaskId`/`Init`/`Main`/`InitFingerprint`/`RetryCount`/`Result`/`Error`/`FailedOnHost`),
`TaskQueue` (`Enqueue`/`ClaimOne`/`MarkDone`/`MarkFailed(workerId,task,error,hostname)`/
`RecoverOrphans`/`Summary`), heartbeat (`HeartbeatWriter.WriteTick`/
`HeartbeatReader.MaxSequence`/`HeartbeatMonitor.Observe`/`IsStalled`),
`FailureMatrix` (`RecordFailure`/`IsHostBlacklisted`/`ShouldPoison`/
`ShouldPoisonByRetry`/`BlacklistedHosts`), `IWorkerProvisioner`
(`EnsureWorkers`/`LiveWorkerCount`/`Shutdown`), `Scheduler`
(`Tick`/`ProcessFailures`/`IsDrained`/`RunToDrain`, ctor takes optional
`FailureMatrix`), `WorkPool` (`Distribute` → `WorkResult`/`WorkOutcome`) are used
consistently across tasks. Note `IsDrained` now also requires `Failed == 0`
(failed/ is transient, drained by `ProcessFailures`).

**Open confirmations before execution:**
1. Final name for the `WarpWorker2` project.
2. Whether `Image.AsFFT()` / `GPU.CheckGPUExceptions()` are the right probe
   primitives in this codebase (Task 7 Step 1) — verify against `WarpLib/Image*.cs`.
