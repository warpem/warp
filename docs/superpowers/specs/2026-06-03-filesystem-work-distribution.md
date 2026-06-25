# Filesystem-Based Work Distribution — Architecture Spec

**Status:** Phase 1 complete — 2026-06-09. One task type ported (fs_ctf). Relay integration, Phase 2 hardening deferred.
**Date:** 2026-06-03 (updated 2026-06-09)
**Scope:** Warp (WarpTools / MTools / MCore) + a documented-but-deferred Relay integration

---

## 1. Motivation

Today, work is distributed to `WarpWorker` processes over a persistent HTTP/REST
connection: the manager (MCore / WarpTools loops) pushes commands to each worker
and maintains a 1 Hz heartbeat; a worker self-terminates if it loses the
heartbeat. This model has two problems for the direction we want to go:

1. **Firewalls.** Workers on cluster compute nodes must accept inbound
   connections from the manager. Compute-node firewall rules frequently block
   this, and the manager must hold one open port per worker.
2. **Ephemeral cluster workers.** We want to spread jobs across many short-lived
   workers launched as individual cluster jobs (potentially in a preemptable
   queue, which always has resources but kills workers without warning).
   Persistent connections do not survive preemption gracefully.

A previous attempt (the `WarpCore` branch) reversed the communication direction
(worker → manager pull model) so only one port on the manager system needed to
be open. That work got too convoluted once it had to integrate with Relay, which
runs on a port-limited system and would have to host a manager per processing job
and forward worker messages to them. **That branch is not the path forward, but
parts of it are reused — see §13.**

Instead we adopt the simpler model proven in `fab-optimizer`: a **filesystem work
queue**. Workers pick task items off disk, move them to a running directory while
processing, then to done/failed. No persistent connections, no inbound ports — a
worker needs only filesystem access to a shared directory.

---

## 2. Guiding principles

- **Warp and Relay are separate products.** Relay depends on WarpLib to make some
  things easier, but **Warp must never depend on Relay's Refund library**. The
  task/queue system is defined entirely inside Warp. Relay is an *optional* outer
  layer that provisions cluster workers; everything works without it (local mode).
- **The task system does not care where workers run.** The only thing that differs
  between local and cluster mode is *who maintains the worker lifecycle*.
- **One worker binary, one scheduler code path.** Local vs. cluster differ only by
  a pluggable provisioning strategy (§7), not by separate implementations.
- **No cross-node clock dependency.** Compute nodes may have unsynchronized clocks.
  All liveness signaling uses monotonic sequence numbers; all elapsed-time
  measurements are local to the measuring process (§8).
- **No scheduler knowledge inside Warp.** Warp must work with SLURM, LSF, PBS, SGE,
  or custom schedulers. The Warp manager **never runs `squeue`** or any
  scheduler command. Worker liveness is determined by worker→manager heartbeats
  (§8), not by querying the cluster.
- **Gradual migration.** A new worker project is created alongside the existing
  one. We port one task type first, prove it, then port the rest one by one.
  **When a task is ported, its legacy code is deleted** — we do not keep two paths
  for the same task.
- **Keep the worker alive.** Restarting a worker is expensive (cluster requeue +
  resource/model reload). Only genuine hardware failure, preemption, an empty
  queue, or a self-imposed runtime backstop causes a worker to exit. Bad-data
  failures never cause an exit (§9).

---

## 3. Roles & layering

```
Relay  (OPTIONAL outer layer — DEFERRED, documented in §12, not implemented now)
  ├── Submits the Manager as a single cluster job; monitors it via the scheduler.
  ├── Maintains a "dumb" worker pool: ensures N worker cluster jobs are
  │   submitted/running; replaces ended ones; caps total submissions.
  └── Knows nothing about the task queue contents. Reads only:
        - the Manager job's scheduler status (alive / ended / failed)
        - the Manager's stdout (progress), same as today

Manager  (WarpTools / MTools / MCore binary, runs on a cluster node in cluster mode,
          or on the local machine in local mode)
  ├── Populates the task queue (it alone knows what work needs doing).
  ├── Runs the scheduler loop (§7): heartbeat, sweep, failure tracking, stats.
  ├── In LOCAL mode: also spawns/maintains worker processes (LocalProvisioner).
  ├── In CLUSTER mode: does NOT spawn workers (ExternalProvisioner = no-op);
  │   Relay provisions them. Manager still does heartbeat/sweep/stats.
  ├── Distribution helpers (§6) enqueue task batches and block on results.
  └── Exits when pending==0 && running==0 (after sweeping orphans).

Workers  (new worker binary, ephemeral, one per GPU)
  ├── On startup: bind GPU, run GPU health probe (§9.3). Probe fail → sick+exit.
  ├── Monitor the manager heartbeat; exit if it stalls (manager dead).
  ├── Write own heartbeat ticks so the manager can detect worker death.
  ├── Claim tasks (atomic rename), run init+main command sequences (§5).
  └── Mark done / failed; classify exceptions (§9).
```

Each layer communicates **only with the layer directly below it, via the shared
filesystem** — except Relay↔Manager, which communicate only through the cluster
scheduler's job status and the Manager's stdout. There is **no** Relay↔Manager
heartbeat and **no** shared code between them.

---

## 4. On-disk contract — the queue directory

The queue directory MUST live on a filesystem visible to the manager host and all
worker nodes (local disk in local mode; a shared parallel FS — Lustre/GPFS/NFSv4 —
in cluster mode). POSIX `rename(2)` atomicity within a single filesystem is the
foundation of the claim mechanism; this holds on local disks and on properly
configured Lustre/GPFS/NFSv4. **Document that the queue dir must not be on an
NFSv3 mount with attribute caching that breaks rename semantics.**

**Queue directory control:** WarpTools commands expose a `--task_dir` CLI
argument (on `DistributedOptions`) for the queue path. When not provided it
defaults to a `tasks/` subdirectory inside the output processing directory. Setting
it to a fast local scratch path matters when the output directory is on a slow
network filesystem.

**Stale-run cleanup:** `TaskQueue.Clear()` is called at the start of each
`DistributeItems` invocation. It deletes all `*.json` files from `pending/`,
`done/`, `failed/`, and `poisoned/`, and recovers any orphaned `running/` subdirs.
This prevents files from a previous (possibly failed) run from appearing as
false-terminal results in a subsequent run.

```
queue_dir/
  pending/                 <task_id>.json            # awaiting claim
  running/<wid>/           <task_id>.json            # claimed by worker <wid>
                           hostname                  # worker's hostname (written at startup)
                           hb-NNNNNN                 # worker heartbeat tick (latest only kept)
                           sick                      # present iff worker hardware-excluded itself
  done/                    <task_id>.json            # completed, with "result"
  failed/                  <task_id>.json            # failed, with "error" + failure history
  poisoned/                <task_id>.json            # exceeded per-task retry cap (§10)
  heartbeat/               tick-NNNNNN               # manager liveness (latest only kept)
  sick/                    <wid>                     # hardware-excluded workers (contains hostname)
  logs/                    <wid>.exit                # per-worker exit reason (one line)
  manager.state.json       # manager bookkeeping: failure matrices, counters, (local) child PIDs
  pool.config.json         # target count, allowed stages, worker command(s), heartbeat params
  pool.lock                # guards against two managers sharing one queue dir
```

Worker IDs (`<wid>`):
- **Cluster:** `<schedulerjobid>-<procid>-gpu<N>` — encodes the cluster job, the
  process index, and the bound GPU so logs and sweeps stay distinct, and so
  multiple workers per node remain individually identifiable.
- **Local:** `local-<pid>-gpu<N>`.

The `<wid>` format is opaque to the manager's sweep logic (§8) — the manager does
not parse a scheduler job id out of it (that would be scheduler-specific
knowledge). Liveness is determined purely by the worker's own heartbeat.

---

## 5. Task file format

A task is a JSON file. Commands are serialized as `NamedSerializableObject`
(the **same** type used today for REST communication — it already round-trips
through JSON, so it can be written to and reloaded from disk without change).

```jsonc
{
  "task_id": "0000042-ctf-stack017",   // sortable numeric prefix = priority (§11)
  "stage": "preprocess",                // for stage filtering (§11)
  "requires_gpu": true,                 // workers excluded to CPU-only skip GPU tasks (§12)
  "init":  [ <NamedSerializableObject>, ... ],   // resource-loading prerequisites
  "main":  [ <NamedSerializableObject>, ... ],   // the actual work
  "init_fingerprint": "<sha256 over the serialized init array>",
  "retry_count": 0,                     // incremented each time it lands in failed/ then re-pended
  "created_at": "<iso8601>"
}
```

### 5.1 Why init + main

Commands are **stateful and chained**. Example movie-processing chain: `LoadStack`
(expensive — loads the movie into GPU memory) then `MovieProcessCTF` and
`MovieProcessMovement` reuse that loaded stack without reloading. We bundle a
sequence of commands into one task file precisely to amortize expensive
operations like loading a movie or a PyTorch model across the commands that need
them.

- **`init`** = commands that load shared resources (gain reference, defect map,
  BoxNet model, denoiser model, population). These are idempotent loads.
- **`main`** = the commands that do the per-item work.

### 5.2 Fingerprint skip

`init_fingerprint` is computed **by the enqueuer** (the distribution helper),
not the worker, as a SHA-256 over the serialized `init` array. The worker keeps,
in memory, the fingerprint of the init sequence it most recently ran
successfully. On claiming a task:

- If the task's `init_fingerprint == worker's last successful fingerprint`, the
  worker **skips** the init sequence (resources already loaded).
- Otherwise it runs the full init sequence and records the new fingerprint.

A freshly spawned worker has no fingerprint, so its first task always runs init.
The fingerprint is **never persisted** — no stale-fingerprint risk across worker
restarts.

### 5.3 Worker-startup actions vs. per-task init — distinct concepts

- **Worker-startup actions** run **once**, before the worker claims anything:
  bind the GPU device, run the GPU health probe (§9.3). These are not
  data-dependent and are not part of any task file.
- **Per-task init** is the `init` sequence in each task file, fingerprint-skipped
  as above. This subsumes the "load a gain reference before crunching an
  arbitrary number of movies" use case — the first movie task loads the gain
  ref; subsequent tasks with a matching fingerprint skip it.

---

## 6. Distribution helpers

These are the drop-in replacement for today's `WorkerWrapper` + `Helper.ForCPU`
distribution loops in WarpTools/MCore. Command objects are `NamedSerializableObject`
— identical to the REST transport; only the delivery mechanism changes (write to a
task file vs. POST over REST).

**As implemented, the distribution layer is split into three tiers:**

### 6.0 `WorkerCommands` (WarpLib — `WarpLib/Workers/WorkerCommands.cs`)

A static factory class with one typed method per worker command. Each method
encapsulates the `NamedSerializableObject` construction and its argument order,
using `nameof(WorkerWrapper.X)` as the command-name anchor. This is the **single
source of truth for command signatures** — callers never construct
`NamedSerializableObject` directly. When `WorkerWrapper` is eventually retired
after the full port, the `nameof` anchor moves here.

### 6.1 `WorkPool` (WarpLib — `WarpLib/Workers/WorkPool.cs`)

Low-level poll-and-block helper. Two public methods:

- **`Enqueue(tasks)`** — idempotent: writes tasks to `pending/` skipping any
  already present in `pending/`, `done/`, or `poisoned/`. Must be called **before
  the Scheduler thread starts** so workers always find work on their first claim
  attempt. Calling it again after Distribute starts is safe (idempotent).
- **`Distribute(tasks, onResult, pollMs)`** — calls `Enqueue` internally
  (idempotent), polls `done/` and `poisoned/`, fires `onResult` synchronously on
  the polling thread as each task resolves, returns results keyed by `task_id`.
  Does not touch `failed/` — the Scheduler owns that (§7).

### 6.2 `DistributeItems<T>` (WarpTools — `WarpTools/Commands/DistributedOptions.cs`)

Orchestration helper on the `DistributedOptions` base class. Builds and owns the
full local-distribution lifecycle for one WarpTools command invocation:

1. Reads `--task_dir` (or defaults to `<output>/tasks`). Calls `Clear()` on the
   queue to discard any stale files from a prior run.
2. Applies path correction per input item before calling the `buildTask` lambda.
3. Calls `pool.Enqueue(tasks)` **before** starting the Scheduler thread.
4. Starts `Scheduler.RunToDrain()` on a background thread.
5. Calls `pool.Distribute(tasks, onResult)` — blocks until all tasks are terminal.
6. Shuts down the provisioner in a `finally` block.
7. Returns `(List<T> Processed, List<T> Failed)`.

The `onResult` callback (called per task on the polling thread) is where callers
update `ProcessingStatus`, call `SaveMeta()`, and invoke `ItemSnapshotWriter.Record`.

`DistributeItems<T>` is constrained to `T : Movie`; the pattern will generalize to
other item types as additional task types are ported.

### 6.3 `ItemSnapshotWriter<T>` (WarpTools — `WarpTools/Commands/BaseCommand.cs`)

Nested class on `BaseCommand` for live JSON snapshot writes used by Relay to track
progress on large datasets (30k+ items) without parsing individual item XML files.

- **`Record(item, succeeded)`** — adds the item to the processed or failed list
  under a lock, then fires a background `Task` that atomically writes two JSON
  files (temp + rename): one for succeeded items, one for failed items. Concurrent
  records coalesce safely.
- **`WaitAll()`** — waits for all background writes to complete before the command
  exits.

Any WarpTool command (distributed or not) can instantiate an `ItemSnapshotWriter`
to provide Relay live updates. Two paths therefore exist:
- Commands using `DistributeItems<T>`: call `Record` from the `onResult` callback.
- Commands using legacy `IterateOverItems`: call `Record` inside the item loop body.

### 6.4 Results channel — two tiers

- **Small structured results** (CTF values, picking counts, scores) go inline in
  the done file's `"result"` object.
- **Large outputs** (images, coordinate files, reconstructions) are written to
  disk at paths the **task specified up front** in its params — the worker does
  not invent output locations. The done file's `"result"` carries scalars plus
  the paths to any bulk outputs.

This mirrors today's REST behavior: small values are returned in the response,
big data is already on disk.

---

## 7. The scheduler & pluggable provisioning

A single scheduler implementation runs in both local and cluster mode. The only
difference is a **pluggable provisioner**:

- **`LocalProvisioner`** — spawns and maintains local worker processes (the local
  worker scheduler). Maintains `target` worker processes; respawns dead ones up to
  the cap (§12). Uses the existing process-launching path (today's `WorkerWrapper`
  spawns local worker processes; we reuse that, configured to point the new worker
  at the queue dir + a `--device N` GPU index).
- **`ExternalProvisioner`** — a no-op. In cluster mode, **Relay** provisions
  workers; the manager only does heartbeat + sweep + failure-tracking + stats.
  Selected by the `--external_provisioner` CLI flag (`DistributedOptions`): when
  set, `DistributeItems` skips device resolution (the manager node may have no GPU)
  and builds an `ExternalProvisioner` instead of a `LocalProvisioner`. Relay invokes
  the WarpTool with this flag and spawns workers pointed at the same queue dir.

Per scheduler tick the manager:

1. Writes the next manager heartbeat tick (`heartbeat/tick-NNNNNN`), deleting the
   prior one.
2. Scans `running/<wid>/` subdirs and sweeps orphans (§8): any worker whose
   heartbeat has stalled has its task file(s) renamed back to `pending/`
   (incrementing `retry_count`), and its `running/<wid>/` dir cleaned up.
3. Updates the host×task failure matrices and the blacklist (§12).
4. (LocalProvisioner only) Tops up worker processes to `target` if any died.
5. Checks the exit condition: if `pending==0 && running==0`, the queue is drained
   — exit. (Orphan sweep in step 2 guarantees in-flight work has settled into
   done/failed/poisoned or been re-pended before this check.)
6. Emits progress to stdout.

`pool.lock` prevents two managers from sharing one queue dir (which would double
the worker count). The scheduler acquires it on startup and fails fast if held.

---

## 8. Heartbeats — symmetric, sequence-number based

Two independent heartbeats, both using monotonically increasing sequence numbers
in filenames so **no cross-node clock comparison ever occurs**:

### 8.1 Manager → Worker (manager liveness)

- Manager writes `heartbeat/tick-NNNNNN` every **5 s** (cluster) / **2 s** (local),
  incrementing an in-memory counter and deleting the prior tick file.
- A worker lists `heartbeat/`, takes the **max** N it sees, and measures (on its
  **own** clock) how long since that max last advanced. If it hasn't advanced
  within **30 s** (cluster) / **10 s** (local), the worker concludes the manager
  is dead and **exits**.

This recreates today's "worker dies if it loses the heartbeat" semantics. When the
manager crashes, the sequence stops advancing → all workers exit cleanly instead
of waiting indefinitely on commands.

### 8.2 Worker → Manager (worker liveness)

- Each worker writes `running/<wid>/hb-NNNNNN` on the same interval, same scheme.
- The manager, per tick, reads each worker's max hb number and measures (on its
  **own** clock) elapsed time since it last advanced. Stall beyond the timeout
  window → the manager sweeps that worker's tasks back to `pending/`.

The manager uses worker heartbeats — **not the cluster scheduler** — to detect
dead workers. This keeps Warp free of scheduler-specific knowledge.

### 8.3 Startup grace

A `running/<wid>/` dir can exist before the worker has written its first `hb-*`
(it was just created at claim time). The manager treats a heartbeat-less dir as
alive for up to **60 s** (cluster) / **15 s** (local) from the dir's mtime; after
that, absence of any heartbeat counts as a stall.

### 8.4 Distinguishing orphaned tasks from hardware exits

When the manager sweeps `running/<wid>/`:
- A dir containing **task JSON files** → orphan: re-pend the tasks (a worker died
  mid-task; preemption, crash, walltime).
- A dir containing only a **`sick`** marker and no task files → the worker
  hardware-excluded itself and exited cleanly *after* releasing its task; count it
  toward the sick total (§12.3) but there is nothing to re-pend.

---

## 9. Worker exception taxonomy & exit conditions

### 9.1 The three buckets

- **(a) Task-level failure** — the exception comes from processing logic on
  healthy hardware: bad input file, numerical failure, malformed params, missing
  input. → `mark_failed` (counted toward retry cap), **continue** to the next
  task.
- **(b) Hardware fault** — CUDA device error, illegal memory access, driver
  reset/hang, ECC error, or the GPU health probe failing. → write the `sick`
  marker (with hostname) to `running/<wid>/` and to `sick/<wid>`, **exit**. Do
  **not** mark the task failed (it wasn't the task's fault); leave the task in
  `running/` so the sweep re-pends it onto a different worker.
- **(c) Unrecoverable worker state** — init half-completed, CUDA context corrupt,
  OOM (in-memory state no longer trustworthy). → **exit** without marking; the
  task is swept back to pending. Distinct from (b) in that it does **not**
  blacklist the host — the next worker (possibly the same node) should retry.

### 9.2 The health-probe-decides rule

The GPU health probe (§9.3) is the arbiter that separates "hardware is dead" from
"the task was just bad." The per-claimed-task control flow:

```
1. If task.init_fingerprint != last_successful_fingerprint:
     run init sequence.
     On exception:
       run GPU health probe.
       - probe FAILS  -> bucket (b): write sick marker, exit, leave task in running/.
       - probe PASSES -> init may have half-loaded resources, so RESET all
                         in-memory resources (null GainRef/Stack/models) and
                         CLEAR the fingerprint, then mark_failed (counted),
                         continue.   [bucket (c)-ish but stays in-process]
     On success: record new fingerprint.

2. Run main sequence.
   On exception:
     run GPU health probe.
     - probe FAILS  -> bucket (b): write sick marker, exit, leave task in running/.
     - probe PASSES -> init state is INTACT (main does not touch loaded resources),
                       so mark_failed (counted), continue — NO reset, NO reload.

3. Success -> mark_done, continue.
```

**Crucial asymmetry:** an exception in **main** does not reset in-memory state, so
the loaded gain ref / stack / model survives for the next task — the amortization
we bundled the task for is preserved. Only an exception in **init** forces a reset
(init may have left things half-loaded), and even then the worker stays
in-process rather than respawning.

OOM folds in naturally: catch OOM → probe (the reset frees memory) → passes →
reset state + mark_failed → continue. A task too large for this GPU keeps failing,
hits the per-task retry cap, and is poisoned (§10). No respawn spiral.

**When uncertain whether an exception is (a) or (b)/(c):** the classification is
driven by the probe result, not by guessing from the exception. The probe is
cheap and authoritative. Exiting (b/c) is always safe (task gets retried);
wrongly marking a task failed (a) burns a retry. But because main-sequence
failures keep the worker alive and only cost a counted retry, the common bad-data
path neither exits nor reloads.

### 9.3 GPU health probe

Runs on every worker startup (before the claim loop) and again after any
init/main exception. Allocate a small tensor on the bound device, run a trivial op
that exercises the same CUDA/cuFFT path real work uses (e.g. a small FFT or
matmul), copy back, verify the result. Pass → healthy. Fail/throw → hardware
fault. Cheap (<1 s).

**Mock mode is fully GPU-free.** When `--mock` is passed to `WarpWorker2`:
- `GPU.GetDeviceCount()` is never called (not even at startup).
- `GPU.SetDevice()` is skipped in `EvaluateCommand`, `ResetResourceState`, and
  startup.
- The startup health probe is skipped entirely.
- `MockCommand` handlers are invoked instead of real `Command` handlers; they
  allocate CPU-only placeholder data (e.g. `new Image(new[] { new float[64*64] },
  new int3(64,64,1))`) — no CUDA P/Invoke.

This makes mock mode safe on machines without CUDA (arm64 dev machines, CI
runners). The `--device` argument is still accepted but used only as an identifier,
not to call into the CUDA runtime.

### 9.4 Exit conditions (complete list)

A worker exits on any of:
- **Empty queue** polled twice with nothing to claim (natural completion).
- **Manager heartbeat stall** (§8.1) — manager dead.
- **Hardware fault / unrecoverable state** (§9.1 b/c).
- **Preemption / walltime SIGTERM** — handled via `PosixSignalRegistration`:
  finish any filesystem rename already in flight, then exit, leaving the current
  task in `running/` for the sweep. Do **not** mark it failed (preemption is not
  the task's fault). This is the C# analogue of fab-optimizer's `TransientFailure`.

`max_units` (exit after N tasks) is intentionally **not** included — no clear need
and it complicates the common case.

`max_runtime_s` per-task self-timeout is also **not** included. The bidirectional
heartbeat already handles both manager death (worker exits) and worker death
(manager sweeps). A genuinely hung task will either be caught by the heartbeat
sweep when the worker process itself hangs, or by the cluster walltime. A
per-task runtime backstop adds complexity for a case that is already covered.

Set the cluster `--signal` option (e.g. `B:SIGTERM@60`) so workers get a grace
window before walltime SIGKILL.

---

## 10. Per-task retry cap & poison

Each task file carries `retry_count`, incremented every time the sweep returns it
from `running/` to `pending/`, or it lands in `failed/` and is re-pended. When
`retry_count` exceeds the configured cap (default suggestion: 3–5), the manager
moves the task to `poisoned/` instead of re-pending it, and surfaces it as a
job-level error on exit. This prevents a single poison-pill task (corrupt input,
or work that exhausts walltime every time) from causing infinite churn.

The retry cap lives in the **manager** (it is the component that reads the queue
and owns the re-pend decision), not in Relay and not in the worker.

---

## 11. Task ordering, priority, and stage filtering

- **Priority = sortable numeric prefix** in `task_id` (e.g. `0000042-...`). Workers
  claim in sorted filename order; lower number = higher priority. No separate
  priority field; fold it into the id prefix. Sufficient for now; richer
  scheduling can be added later without changing the on-disk contract.
- **Stage filtering** (from fab-optimizer): a worker reads its `allowed_stages`
  from `pool.config.json`. When claiming, it reads a candidate task's `stage`
  *before* renaming and skips tasks whose stage it is not allowed to service
  (leaving them untouched in `pending/`). This lets different worker roles run
  against the same queue without claiming work they can't do (e.g. a preprocess
  worker not claiming a refine task).

### 11.1 Atomic claim

Identical to fab-optimizer: list `pending/*.json` sorted; for each candidate
(respecting stage filter), attempt `rename(pending/<id>.json,
running/<wid>/<id>.json)`. The winner of a race gets the file; losers get
`FileNotFoundError` and move to the next candidate. A `rename` that fails because
the filesystem is full leaves the task in `pending/` (nothing moved) — safe; the
worker logs and retries after a poll interval. Contrast with a failed
`mark_done`/`mark_failed`, which would leave a task stuck in `running/` (recovered
later by the sweep).

---

## 12. Relay integration — DEFERRED (documented to keep everything in sync)

**Not implemented in this phase.** Documented here so the Warp side exposes
exactly what Relay will later need, and nothing drifts. Relay code is **not**
touched now.

### 12.1 Job-type pool interface

A Relay job type that works with pools declares:
- that it uses a pool;
- the worker submission command(s) — **one per (GPU × workers-per-device)** so a
  node with 8 GPUs, or GPU oversubscription for better utilization, is expressed
  as multiple commands, each pinned to a `--device N`;
- a default pool size.

Relay fills in the device index per command; the queue is oblivious to how many
workers share a node or a GPU — that is purely a provisioning detail.

### 12.2 The "dumb" pool

Relay's responsibility reduces to: **ensure N worker cluster jobs are submitted or
running**, replacing ended ones, using its existing `ClusterQueue` machinery
(`submit` / `status` / `cancel` templates, already supporting SLURM/LSF/PBS/SGE/
custom). Relay also:
- submits the **Manager** as a single cluster job and monitors its scheduler
  status; Manager job ended → Relay cancels the whole worker pool and marks the
  Relay job done/failed per the Manager's exit;
- caps total worker submissions at e.g. `2 × target` (or `target + max`) so the
  bad-node sick-worker scenario (§12.3) cannot spiral;
- on Relay restart, reads its own private state (Manager cluster job id, queue dir,
  target) and re-adopts the Manager job via scheduler status, resuming pool
  maintenance. This Relay state lives in Relay's own store — **WarpTools never
  touches it.**

Relay **never reads the task queue**. Any richer per-task status it surfaces comes
from the Manager's stdout (§12.3), parsed the way Relay already parses WarpTools
progress output.

### 12.3 Bad-node detection, blacklist, sick workers

Hardware faults on the cluster (a node with a crashed GPU driver) chew through
many tasks quickly and are almost guaranteed to be hit when saturating a free
queue (the bad node is usually the free one). Handling:

- The **GPU health probe** (§9.3) on worker startup catches a dead node *before*
  the worker wastes tasks. A failing probe → the worker writes a `sick` marker and
  **exits** (it does not idle — an idle process that keeps heartbeating is a
  confusing half-alive state).
- The Manager maintains, in `manager.state.json`, two **sets** (cardinality, not
  raw counts) from observed failures, keyed by **hostname** (a host may have up to
  8 GPUs that all fail together when the driver crashes — grouping by host, not by
  wid, is correct):
  - `host_failures[hostname]` = distinct task ids that failed on this host;
  - `task_failures[task_id]`  = distinct hostnames this task failed on.
- A bad **node** shows up as large `|host_failures[h]|` (many distinct tasks fail
  on one host) → blacklist the hostname (write `blacklisted_nodes/<hostname>` —
  represented in the layout under `sick/` / a blacklist file; exact filename TBD
  at implementation). A bad **task** shows up as large `|task_failures[t]|` (one
  task fails on many distinct hosts) → poison it (§10). The two signals are
  orthogonal, so no timing heuristic is needed — and timing would be unreliable
  anyway because the amount of CPU work before a task hits the GPU is
  unpredictable. Thresholds (e.g. 3–5 distinct) are configurable.
- **Worker self-exclusion:** a worker checks the blacklist for its own hostname
  before each claim. If blacklisted, it exits. This is correct because in
  practice a GPU hardware fault on one GPU usually means the whole node's driver
  is unhealthy; CPU-only work on that node is not worth the complexity of a
  fallback path.
- **The all-on-one-bad-node deadlock:** if every requested worker lands on the same
  bad host and all go sick, Relay must not spin forever submitting replacements
  that also land there. Two guards: (1) the Manager reports its known **sick-worker
  count** on stdout, and Relay parses it (same channel as completed-item counts
  today); (2) Relay's submission cap (§12.2). Beyond the cap the job runs with
  fewer effective workers — slower but correct, never spiraling.

The **dummy GPU init task** also extends to mid-task hardware faults: a CUDA
device error / driver reset caught during main execution routes through the same
probe (§9.2) and, on probe failure, becomes a `sick` exit rather than a task
failure — keeping the hardware/software separation clean throughout the worker's
life, not just at startup.

### 12.4 Why no Relay↔Manager heartbeat

The Manager is self-sufficient: it drains the queue and exits on its own. Relay
learns of Manager death through the cluster scheduler's job status (the same way
it monitors every other job). "Relay crashed but Manager + workers are fine" is a
non-event — workers heartbeat to the Manager, the Manager is alive, work
continues; Relay re-adopts on restart. "Manager crashed" cascades cleanly: workers
lose the Manager heartbeat and exit; Relay sees the Manager job end and tears down.
A Relay↔Manager heartbeat would also wrongly kill a healthy Manager that is merely
waiting hours for a congested queue to schedule its first worker. So: none.

---

## 13. Reuse from the `WarpCore` branch

The `WarpCore` branch's *reversed-networking* design is abandoned, but it already
built the command-dispatch refactor we want, and that is reused wholesale:

- **`CommandAttribute`** + reflection registration. Command methods are tagged
  `[Command(nameof(WorkerWrapper.MovieProcessCTF))]` and registered into a
  `Dictionary<string, MethodInfo>` at static init by scanning
  `typeof(WarpWorkerProcess).GetMethods(...)` for the attribute. Dispatch is a
  dictionary lookup + `method.Invoke(null, new object[]{ Command })`. This
  replaces the old brittle hardcoded-string + giant-if/else block, and
  `nameof(WorkerWrapper.X)` makes the wrapper method the single source of truth
  for each command name (refactor-safe, no string duplication between wrapper and
  worker).
- **Per-domain command files** (`Commands/Movie.cs`, `Commands/Tomo.cs`,
  `Commands/MPA.cs`, `Commands/Service.cs`) and the worker's static resource state
  (`GainRef`, `OriginalStack`, `BoxNetModel`, `MPAPopulation`, …) port unchanged.
- **Mock mode** (`[MockCommand]` / `MockCommandMethods`, simulated delays) ports
  too — directly useful for testing the new architecture without a GPU.

**What we drop from `WarpCore`:** the entire `ControllerClient` /
`WorkerController*` / `WorkDistributor` networking layer. What feeds
`EvaluateCommand` changes from "REST POST / controller poll" to "read the next
command from the claimed task file's init/main sequence." The command method
bodies and the dispatch mechanism are unchanged.

---

## 14. Components to build (this phase)

1. **New worker project** (e.g. `WarpWorker2` / final name TBD) — queue-consuming
   worker binary. Lifts the WarpCore command dispatch + command bodies + mock
   mode; drops all controller networking. Adds: claim loop, init/main execution
   with fingerprint skip, exception taxonomy + health probe, both heartbeats,
   self-exclusion check, exit conditions, per-worker exit files.
2. **Distribution helpers** in WarpLib — refactored from the existing
   `WorkerWrapper` command-builders to emit task files (init+main sequences,
   computed `init_fingerprint`) and block on results keyed by `task_id`. Shares
   `NamedSerializableObject` construction with the worker.
3. **Scheduler** in WarpLib — heartbeat writer, orphan sweep, host×task failure
   matrices + blacklist, retry-cap/poison logic, stats/stdout reporting,
   `pool.lock`, exit-on-drain. Pluggable provisioner: `LocalProvisioner`
   (spawns/maintains processes via the existing worker-launch path, pointed at the
   queue dir + `--device N`) and `ExternalProvisioner` (no-op).
4. **Local worker scheduler** — the `LocalProvisioner` + scheduler wired into
   WarpTools/MTools/MCore local mode, replacing today's local distribution loop
   for the first ported task type.

**Migration order:** port **one** task type first (suggested: frameseries CTF —
self-contained, GPU, exercises the load-stack-then-process amortization), prove
the whole pipeline end-to-end (including mock mode for CI), then port remaining
task types one by one, **deleting each task's legacy distribution code as it is
ported.** Relay is not touched this phase.

---

## 15. Failure-mode summary (reference)

| Failure | Handling |
|---|---|
| Worker crash mid-task | Task in `running/<wid>/`; manager heartbeat sweep re-pends it (§8). |
| Worker preempted (SIGTERM) | Finish in-flight rename, exit, leave task in running/ for sweep; not marked failed (§9.4). |
| Worker walltime SIGKILL | Same as crash — task swept (§8). `--signal` gives grace. |
| GPU/driver dead at startup | Health probe fails → `sick` + exit (§9.3, §12.3). |
| GPU fault mid-task | Probe fails → `sick` + exit, task left for sweep (§9.2). |
| OOM | Probe passes → reset state + mark_failed → continue; repeats → poison (§9.2, §10). |
| Bad input / numerical failure | mark_failed, counted; worker keeps loaded resources, continues (§9.1a). |
| Poison-pill task | Retry cap exceeded → `poisoned/` + job-level error (§10). |
| Bad node chewing tasks | host×task matrices → blacklist host; workers self-exclude (§12.3). |
| All workers on one bad node | Relay submission cap + sick-count on stdout (§12.3). |
| Manager crash | Heartbeat stops → workers exit; Relay sees job end → teardown (§8.1, §12.4). |
| Relay crash | Non-event; Manager+workers continue; Relay re-adopts on restart (§12.2, §12.4). |
| Two managers, one queue dir | `pool.lock` — second fails fast (§7). |
| Filesystem full on claim | `rename` fails, task stays in `pending/`; worker retries (§11.1). |
| Stuck-but-alive worker (algorithmic) | Out of scope — user notices and kills the job (matches old arch). |
| Scheduler controller unavailable | Relay's concern (existing `ClusterQueue` retry); Manager never calls the scheduler. |
```
