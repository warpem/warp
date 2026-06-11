# WarpLib/Workers — Filesystem Work-Distribution Library

Pure-filesystem work queue and scheduler. No GPU, no network — every component is
testable in isolation.  Full architecture description: `docs/superpowers/specs/2026-06-03-filesystem-work-distribution.md`.

---

## Directory layout

```
WarpLib/Workers/
  Queue/
    QueueLayout.cs        — resolves subpaths; EnsureDirectories()
    TaskItem.cs           — task model + JSON round-trip + SHA-256 init fingerprint
    TaskQueue.cs          — atomic claim (POSIX rename), done/failed, sweep, Clear()
    Heartbeat.cs          — sequence-number tick writer/reader/monitor
  Scheduling/
    IWorkerProvisioner.cs — strategy interface (EnsureWorkers, LiveWorkerCount, Shutdown)
    LocalProvisioner.cs   — spawns WarpWorker2 child processes
    ExternalProvisioner.cs — no-op (Relay provisions workers in cluster mode)
    FailureMatrix.cs      — host×task failure sets → blacklist / poison decisions
    Scheduler.cs          — per-tick loop: heartbeat, sweep, failures, top-up, drain
  WorkPool.cs             — Enqueue (idempotent) + Distribute (poll until terminal)
  WorkerCommands.cs       — factory for NamedSerializableObject command construction
```

---

## On-disk queue structure

```
queue_dir/
  pending/       <task_id>.json        ← awaiting claim
  running/
    <wid>/       <task_id>.json        ← claimed by this worker
               hb-NNNNNN              ← worker heartbeat tick (latest only)
               hostname               ← worker's machine name
               sick                   ← present if worker self-excluded (hardware fault)
  done/          <task_id>.json        ← completed
  failed/        <task_id>.json        ← failed (transient — Scheduler re-pends or poisons)
  poisoned/      <task_id>.json        ← exceeded retry cap (terminal)
  heartbeat/     tick-NNNNNN          ← manager liveness tick (latest only)
  sick/          <wid>                ← hardware-excluded workers
  blacklisted_nodes/  <hostname>      ← hosts blacklisted by FailureMatrix
  logs/          <wid>.exit           ← per-worker exit reason
```

**Key invariant:** `rename(2)` atomicity within one filesystem is the claim
mechanism. NFSv3 with attribute caching must not be used as the queue directory.

---

## State machine for one task file

```
                  Enqueue()
                     │
                     ▼
              ┌─────────────┐
              │  pending/   │◄──────────────────────────────────────┐
              └─────────────┘                                        │
                     │ ClaimOne() — atomic File.Move                 │
                     ▼                                               │
              ┌─────────────────┐                                    │
              │ running/<wid>/  │                                    │
              └─────────────────┘                                    │
                  │         │                                        │
          success │         │ failure                   re-pend (retry below cap)
                  ▼         ▼                                        │
             ┌────────┐  ┌────────┐   ProcessFailures()   ┌─────────┴──────┐
             │ done/  │  │failed/ │──────────────────────►│  Scheduler     │
             └────────┘  └────────┘                        └────────┬───────┘
              (terminal)                                            │ poison (retry cap exceeded
                                                                    │  or bad-task signal)
                                                                    ▼
                                                             ┌──────────────┐
                                                             │  poisoned/   │
                                                             └──────────────┘
                                                              (terminal)
```

Worker crash / stall → task stays in `running/<wid>/` → `SweepStalledWorkers`
calls `RecoverOrphans`, which moves it back to `pending/` (incrementing `retry_count`).

---

## Heartbeats — no cross-node clocks

Both directions use the same mechanism: monotonically increasing sequence numbers
in filenames. The **observer** measures elapsed time on its own clock; no
cross-node clock comparison ever occurs.

```
Manager                              Worker
───────                              ──────
heartbeat/tick-000001                running/<wid>/hb-000001
heartbeat/tick-000002   (deletes     running/<wid>/hb-000002   (deletes
heartbeat/tick-000003    prior)      running/<wid>/hb-000003    prior)
        │                                       │
        └── Worker reads MaxSequence,           └── Scheduler reads MaxSequence,
            measures elapsed on own clock           measures elapsed on own clock
            → exits if stalled > timeout            → sweeps if stalled > timeout
```

---

## Failure classification

```
Exception during task execution
           │
           ▼
    GPU health probe
           │
    ┌──────┴──────┐
    │ probe FAILS │  → write sick/ marker, EXIT (leave task for sweep)
    └─────────────┘    worker hardware is dead; task was not at fault
           │
    ┌──────┴──────┐
    │ probe PASSES│
    └─────────────┘
           │
    ┌──────┴───────────────────┐
    │ exception in INIT?        │  → reset in-memory state, clear fingerprint,
    │                           │    MarkFailed (counted), CONTINUE (no exit)
    └───────────────────────────┘
    │ exception in MAIN?        │  → init state intact; MarkFailed (counted),
    │                           │    CONTINUE; no reset (amortization preserved)
    └───────────────────────────┘
```

---

## `WorkerCommands` — command signature registry

`WorkerCommands.cs` is the **single source of truth** for the argument order of
every worker command. It replaces the previous pattern of constructing
`NamedSerializableObject` inline at call sites. All callers use the typed factory:

```csharp
// Old (fragile — argument order duplicated everywhere):
new NamedSerializableObject(nameof(WorkerWrapper.LoadStack), path, scale, eer, true)

// New (refactor-safe):
WorkerCommands.LoadStack(path, scale, eerGroupFrames)
```

`nameof(WorkerWrapper.X)` is the command-name anchor until `WorkerWrapper` is
retired. After the full port, the anchor moves here.

---

## Ownership rules (critical for correctness)

| Component | Owns | Never touches |
|---|---|---|
| `TaskQueue` | `pending/`, `running/<wid>/`, `done/`, `failed/` writes | `poisoned/` |
| `Scheduler.ProcessFailures` | drains `failed/` → re-pend or `poisoned/` | |
| `WorkPool.Distribute` | polls `done/` and `poisoned/` | `failed/` |
| Worker | writes to `running/<wid>/` → `done/` or `failed/` | `poisoned/` |

The Scheduler is the **sole writer on `failed/`** (reading + deleting) and the
**sole writer on `poisoned/`**. This prevents any two-writer race.

---

## Phase 2 items (not yet implemented)

- `pool.lock` (prevent two managers sharing one queue dir)
- `manager.state.json` persistence (failure matrix survives manager restart)
- Preemption SIGTERM handler (clean exit leaving task for sweep, not crash)
- `LocalProvisioner` slot-assignment fix for multi-device respawn after differential exits
- Relay integration: `ExternalProvisioner` wiring, CLI flag to suppress local worker spawning
