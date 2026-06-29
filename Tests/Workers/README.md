# Tests/Workers — Work-Distribution Test Suite

xUnit tests for the filesystem work-distribution stack. All tests run without a
GPU; the end-to-end test spawns the real `WarpWorker2 --mock` binary.

---

## Test files

| File | What it tests |
|---|---|
| `QueueLayoutTests.cs` | `QueueLayout` path construction + `EnsureDirectories()` |
| `TaskItemTests.cs` | `TaskItem` JSON round-trip + fingerprint stability + content-sensitivity |
| `TaskQueueTests.cs` | `TaskQueue`: enqueue, atomic claim, stage filter, full random-order claim coverage, done/failed, orphan recovery, summary |
| `HeartbeatTests.cs` | `HeartbeatWriter`/`HeartbeatReader`/`HeartbeatMonitor`: tick increment, max sequence, stall detection, startup grace |
| `FailureMatrixTests.cs` | `FailureMatrix`: host blacklist threshold, dedup, task poison threshold, retry cap |
| `SchedulerTests.cs` | `Scheduler`: stall sweep, drain detection, re-pend below cap, poison at cap, host blacklist marker |
| `WorkPoolTests.cs` | `WorkPool`: distribute blocks until terminal, poisoned outcome |
| `EndToEndMockTests.cs` | Full pipeline (spawns WarpWorker2 --mock): drain 4 tasks; long-task-not-falsely-swept regression; `--persistent` worker survives an empty queue |
| `SmokeTests.cs` | Sanity check (test harness works) |

---

## Test isolation

Every test class is `IDisposable` and creates its queue directory under
`Path.GetTempPath()` with a fresh GUID per test run. `Dispose()` deletes the
directory. Tests never share filesystem state.

---

## End-to-end test

`EndToEndMockTests` launches real `WarpWorker2 --mock` processes (the other
classes never spawn workers). `MockPipelineDrainsQueue` exercises the full stack:

```
Test process
  │
  ├─ pool.Enqueue(4 mock tasks)           ← before scheduler starts (race prevention)
  │
  ├─ schedThread → Scheduler.RunToDrain()
  │       │
  │       ├─ tick: write manager heartbeat/tick-N
  │       ├─ tick: LocalProvisioner.EnsureWorkers(2)
  │       │           └─ spawn 2 × WarpWorker2 --mock --device 0 --queue-dir <tmp>
  │       └─ tick: poll until IsDrained()
  │
  ├─ pool.Distribute(tasks, pollMs:200)
  │       └─ polls done/ until all 4 task IDs appear
  │
  └─ Assert: 4 results, all WorkOutcome.Done
```

Each `WarpWorker2 --mock` worker:
1. Claims a task from `pending/` (atomic rename).
2. Runs `MockLoadStack` + `MockMovieProcessCTF` (CPU-only; no GPU).
3. Moves task to `done/`.
4. Exits when the queue is empty (consecutiveEmpty ≥ 2).

**Binary availability:** `Tests.csproj` has a `CopyWorkerBinary` MSBuild target
that copies all `bin/WarpWorker2*` files to the test output directory after build.
The test runs automatically in `dotnet test` without manual setup.

---

## Running the tests

```bash
# All tests (including e2e) — Debug, the default:
dotnet test Tests/Tests.csproj

# Unit tests only (fast, no subprocess):
dotnet test Tests/Tests.csproj --filter "FullyQualifiedName!~EndToEndMockTests"

# Specific class:
dotnet test Tests/Tests.csproj --filter SchedulerTests
```

**Run the worker-spawning e2e tests in Debug, not Release.** WarpWorker2's Release
config sets `PlatformTarget=x64` (the deployed worker must be an x64 process so the
x64-only CUDA/native libs load; the cluster artifact is pinned x64 by the
`--runtime linux-x64` publish anyway). On an arm64 machine that x64 worker DLL
fails to load (`FileLoadException: architecture is not compatible`), so a spawned
worker dies immediately and `WorkPool.Distribute` — which has no timeout — polls
forever and the test hangs. The Debug build is AnyCPU, so it runs natively on both
arm64 and x64; mock mode never touches the native libs. `CopyWorkerBinary` copies
from `..\bin\` (the Debug output) for this reason.

---

## Test flow diagram (SchedulerTests)

The scheduler tests use a `FakeProvisioner` (no-op) so they test scheduling logic
without spawning any workers. Failures are staged by hand with `EnqueueClaimFail`.

```
SchedulerTests
  │
  ├─ FakeProvisioner ──► no processes spawned
  ├─ TaskQueue ──► real filesystem ops in temp dir
  └─ Scheduler.Tick() ──► reads/writes queue state directly

  ProcessFailuresBlacklistsBadHost:
    Step 1: EnqueueClaimFail("0000001-a", host="nodeA") → failed/
            Tick()  → RecordFailure("nodeA","0000001-a")
                      |hostFailures["nodeA"]|=1 < threshold=2
                      → re-pend (retry 1); no blacklist marker

    Step 2: EnqueueClaimFail("0000002-b", host="nodeA") → failed/
            Tick()  → RecordFailure("nodeA","0000002-b")
                      |hostFailures["nodeA"]|=2 >= threshold=2
                      → write blacklisted_nodes/nodeA
                      → re-pend (retry 1)
```
