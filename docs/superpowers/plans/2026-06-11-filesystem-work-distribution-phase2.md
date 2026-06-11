# Filesystem Work Distribution — Phase 2 Implementation Plan

**Status:** Ready to implement (Phase 1 complete 2026-06-09)
**Scope:** Hardening + additional task-type ports. Relay integration is explicitly
deferred until several more WarpTools are ported. Full architecture:
`docs/superpowers/specs/2026-06-03-filesystem-work-distribution.md`.

---

## Summary

Phase 1 proved the end-to-end pipeline with `fs_ctf`. Phase 2 has two tracks that
are independent and can be done in any order:

**Track A — Infrastructure hardening** (three items; none block the port track)
- A1: `pool.lock` — guard against two managers on one queue dir
- A2: Failure matrix persistence — survive manager restart
- A3: SIGTERM handler — clean cluster preemption exit
- A4: `LocalProvisioner` slot bug — wrong device index after differential exits

**Track B — Port additional WarpTools commands**
- B1: `fs_motion` (frame-series motion)
- B2: `fs_motion_and_ctf` (combined motion + CTF — requires both worker commands)
- B3: Additional commands as needed (tilt-series ports will follow a separate plan
  once the frame-series set is done, since they use different item types)

Track A items are ordered by impact: A1 is cheap and prevents a real foot-gun; A2
and A3 only matter in production cluster use; A4 only bites with `--perdevice > 1`.

---

## Track A — Infrastructure hardening

### A1: `pool.lock` — prevent two managers sharing one queue dir

**Problem:** running `fs_ctf` twice on the same output directory (or two different
commands with the same `--task_dir`) starts two Schedulers, each spinning up their
own worker pool against the same queue. Workers from both pools race for the same
tasks; results land in `done/` from whichever worker got there first; the second
manager thinks the tasks were done by its workers. Corrupts both runs silently.

**Fix:** write an exclusive lock file at `<queue_dir>/pool.lock` on Scheduler
startup and release it on shutdown. Use `FileShare.None` open + `DeleteOnClose`
(or an explicit delete in a `finally` block). A second Scheduler attempting to
acquire the same lock should fail fast with a clear error: `"Another manager is
already using queue dir <path>. Use --task_dir to choose a different location."`

**Files to change:**
- `WarpLib/Workers/Scheduling/Scheduler.cs` — acquire lock in ctor, release in `RunToDrain`'s finally
- `WarpLib/Workers/Queue/QueueLayout.cs` — add `Lock` path property
- `Tests/Workers/SchedulerTests.cs` — add test: second Scheduler on same dir throws

**Implementation sketch:**

```csharp
// Scheduler ctor:
_lockHandle = File.Open(layout.Lock, FileMode.OpenOrCreate,
    FileAccess.Write, FileShare.None);   // throws IOException if already held

// RunToDrain finally:
_lockHandle?.Dispose();
try { File.Delete(layout.Lock); } catch { }
```

On Linux/macOS `FileShare.None` is advisory via `flock`; on Windows it is
mandatory. Both are sufficient for the local-mode use case.

---

### A2: Failure matrix persistence — `manager.state.json`

**Problem:** the `FailureMatrix` is in-memory only. A manager restart (crash,
requeue on cluster) loses all failure history, so a blacklisted host gets a clean
slate and will fail the same tasks again. Retry counts are on the task files (so
they survive), but the host×task matrix does not.

**Fix:** serialize the `FailureMatrix` to `manager.state.json` after every call to
`ProcessFailures`. Load it on Scheduler startup if the file exists.

**Format:**
```json
{
  "host_failures": { "nodeA": ["task1", "task2"] },
  "task_failures": { "task1": ["nodeA"] }
}
```

**Files to change:**
- `WarpLib/Workers/Scheduling/FailureMatrix.cs` — add `ToJson()` / `FromJson()`
- `WarpLib/Workers/Scheduling/Scheduler.cs` — load on ctor; save after `ProcessFailures`
- `WarpLib/Workers/Queue/QueueLayout.cs` — add `ManagerState` path property
- `Tests/Workers/SchedulerTests.cs` — add test: restart Scheduler, prior blacklist survives

---

### A3: SIGTERM handler — clean cluster preemption exit

**Problem:** when a cluster job hits walltime and gets SIGKILL, the worker dies
mid-task. Because all queue state transitions are atomic renames the task JSON is
never corrupted, and the sweep will re-pend it — but the worker has no chance to
write an exit reason or flush any in-progress I/O. More importantly, if the worker
is in the middle of `MarkDone`'s two-step (write temp file, then rename to `done/`),
an abrupt SIGKILL between those two steps leaves a `.tmp.*` orphan in `running/`.

**Fix:** register a `PosixSignalRegistration` for `SIGTERM` that sets a volatile
flag `_sigTermReceived`. The claim loop checks the flag after each completed task
and before claiming the next. When set: write an exit reason file, let the current
task's `MarkDone`/`MarkFailed` complete normally (it's already in flight), then
return without claiming another task.

This is safe because:
- If SIGTERM arrives while no task is claimed: exit immediately, nothing to sweep.
- If SIGTERM arrives mid-init or mid-main: the exception taxonomy handles it — the
  worker does not check the flag during execution, only between tasks. The cluster's
  `--signal B:SIGTERM@60` gives 60 s of grace, which is enough for one task.
- SIGKILL after the grace period leaves the task in `running/` for the sweep, same
  as today. The SIGTERM handler just handles the clean-exit case.

**Files to change:**
- `WarpWorker2/WorkerProcess.cs` — add `PosixSignalRegistration`, flag check between tasks
- `Tests/Workers/` — this is hard to unit-test; document as manual test

**Cluster config note:** set `--signal B:SIGTERM@60` (SLURM) or equivalent so the
worker receives SIGTERM 60 s before SIGKILL. This is a deployment concern, not
code.

---

### A4: `LocalProvisioner` slot-assignment bug

**Problem:** `EnsureWorkers` assigns device slots using `slots[_procs.Count]`
after removing exited processes. If device-0 has 2 workers and device-1 has 1
worker, and one device-0 worker exits while device-1's worker is still alive, the
slot list is `[0, 0, 1]` but `_procs.Count` after removal is 2, so the replacement
worker gets device-1 (`slots[2] = 1`) instead of device-0 (`slots[0] or slots[1]`).

**Fix:** track which device slot each process was assigned at spawn time. On
`EnsureWorkers`, build the set of occupied slots from still-alive processes and
fill unoccupied ones.

```csharp
// Replace List<Process> _procs with:
private readonly List<(Process proc, int device)> _procs = new();

// EnsureWorkers: build all slots, find unoccupied ones, fill them.
var occupiedDevices = _procs
    .Where(e => !e.proc.HasExited)
    .Select(e => e.device)
    .ToList();

var slots = _devices.SelectMany(d => Enumerable.Repeat(d, _perDevice)).ToList();
foreach (int dev in slots)
{
    if (occupiedDevices.Remove(dev)) continue;   // slot is filled
    if (_procs.Count(e => !e.proc.HasExited) >= target) break;
    _procs.Add((Spawn(dev), dev));
}
```

**Files to change:**
- `WarpLib/Workers/Scheduling/LocalProvisioner.cs`
- `Tests/Workers/SchedulerTests.cs` — add test with 2 devices, kill one worker, verify replacement gets correct device

---

## Track B — Port additional WarpTools commands

The pattern is established by `fs_ctf` and mechanically identical for every frame-
series command. For each:

1. Identify `Init` commands (gain ref load — same `WorkerCommands.LoadGainRef`
   call for all frame-series commands, fingerprint-amortized).
2. Identify `Main` commands from the old `IterateOverItems` body.
3. Add any missing `WorkerCommands` factory methods.
4. Add any missing worker command handlers (`WarpWorker2/Commands/Movie.cs`).
5. Replace the `GetWorkers() + IterateOverItems` block with `CLI.DistributeItems(buildTask: ...)`.
6. Delete the now-unused legacy path for that command.
7. Add mock handlers to `WarpWorker2/MockCommands/MovieMock.cs` for any new commands.
8. Update the e2e mock test if the new command exercises new mock paths.

### B1: `fs_motion` — frame-series motion correction

**Old worker calls:** `LoadGainRef`, `LoadStack`, `MovieProcessMovement`,
`MovieExportMovie` (async), `WaitAsyncTasks`, `GcCollect`.

`MovieExportMovie` is async — the worker starts writing the output movie on a
background thread. The old path explicitly called `WaitAsyncTasks` before the next
item. In the new path, `WaitAsyncTasks` should be the last command in `Main` (same
effect: the task is not marked done until the async export finishes).

**New worker commands needed:** `MovieExportMovie` and `MovieProcessMovement` are
already in `WorkerCommands.cs`. Verify `WarpWorker2/Commands/Movie.cs` handlers
exist for both and add mock stubs if not present.

**Task structure:**
```
Init: [LoadGainRef(...)]
Main: [LoadStack(...), MovieProcessMovement(m.Path, optionsMotion),
       MovieExportMovie(m.Path, optionsExport), WaitAsyncTasks(), GcCollect()]
```

### B2: `fs_motion_and_ctf` — combined motion + CTF

**Old worker calls:** same as `fs_motion` plus `MovieProcessCTF`. If
`--use_sum` is active the worker loads the average stack after motion, then runs
CTF on it. This requires `WaitAsyncTasks` before `LoadStack` (average) in the old
path. In the new task-file model, sequence is preserved naturally — just order the
commands correctly in `Main`.

Note: `LoadStack` for the average uses `correctGain=false` here (the average is
already gain-corrected by motion correction). This differs from `fs_ctf`'s
`correctGain=true`. Use `WorkerCommands.LoadStack(..., correctGain: false)`.

**Task structure (with `--use_sum`):**
```
Init: [LoadGainRef(...)]
Main: [LoadStack(dataPath, scale, eer),          // raw movie
       MovieProcessMovement(m.Path, optionsMotion),
       MovieExportMovie(m.Path, optionsExport),
       WaitAsyncTasks(),
       LoadStack(m.AveragePath, 1M, eer, correctGain: false),
       MovieProcessCTF(m.Path, optionsCTF),
       GcCollect()]
```

**Task structure (without `--use_sum`):**
```
Init: [LoadGainRef(...)]
Main: [LoadStack(dataPath, scale, eer),
       MovieProcessMovement(m.Path, optionsMotion),
       MovieProcessCTF(m.Path, optionsCTF),
       MovieExportMovie(m.Path, optionsExport),
       WaitAsyncTasks(), GcCollect()]
```

---

## Dependency order

```
A1 (pool.lock)         — standalone, do first; prevents foot-gun in all ports
A4 (provisioner bug)   — standalone, do with A1
B1 (fs_motion)         — standalone; no infrastructure needed
B2 (fs_motion_and_ctf) — after B1 (reuses same worker commands)
A2 (matrix persist)    — after B1/B2 proven on real data; only matters for cluster
A3 (SIGTERM handler)   — last; only matters for cluster preemption
```

---

## What is explicitly NOT in Phase 2

- **Relay integration** — deferred until more WarpTools are ported and the local
  path is well-exercised. Relay work requires: `ExternalProvisioner` wiring,
  CLI flag to suppress local worker spawning, Relay job-type pool interface,
  submission cap, re-adoption on Relay restart. These become Phase 3.
- **Tilt-series command ports** — different item type (`TiltSeries` vs `Movie`);
  `DistributeItems<T>` is constrained to `T : Movie` today. Either generalize the
  constraint or add a parallel `DistributeTiltSeriesItems` helper. Separate plan.
- **ETA rolling window** — the global wall-clock average in `DistributeItems` is
  correct in expectation; the 20-item rolling window is a cosmetic improvement.
- **`failed_items.json` stale across runs** — pre-existing behavior; `Clear()` on
  the queue cleans task state but not the output JSON files. Low priority.
