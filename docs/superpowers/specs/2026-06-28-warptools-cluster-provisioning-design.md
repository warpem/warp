# WarpTools minimal cluster pool provisioning — design

## Background and motivation

WarpTools now distributes work through the filesystem work-queue system
(`pending/` → `running/<wid>/` → `done/`/`failed/`/`poisoned/`). Two provisioning
modes exist today:

- **Local** (default): `LocalProvisioner` spawns `WarpWorker2` processes on the
  local machine, one per GPU slot (`device_list` × `perdevice`).
- **External**: `ExternalProvisioner` is a no-op; an external orchestrator (Relay)
  provisions workers that claim from the shared queue directory.

Relay is the intended way to drive large multi-node cluster runs, but it is months
from release. In the meantime the only way for a user to run WarpTools across an HPC
cluster is to roll their own worker provisioning by hand. This design adds a
**minimal, self-contained cluster provisioning mode to WarpTools itself** so users
can submit a worker pool to a batch scheduler (SLURM/LSF/PBS/SGE/anything) without
Relay and without writing their own submission glue.

The design deliberately mirrors Relay's templating approach (`{{ }}` placeholders, a
queue-definition describing scheduler commands) so that knowledge transfers and a
future migration to Relay is natural. It is intentionally a **subset** of Relay: no
status polling, no deficit reconciliation, no pool-state persistence.

## Goals

- Submit a pool of `WarpWorker2` workers to a batch scheduler that claim tasks from
  the shared filesystem queue.
- Keep the scheduler-specific knowledge (submit/cancel commands, job-id parsing) in a
  user-supplied JSON config, not hard-coded.
- Keep the submission-script shape in a user-supplied template with `{{ }}`
  placeholders.
- Let users pass arbitrary template values on the command line without those values
  being hard-coded as `DistributedOptions` properties.
- Never leave cluster jobs lingering: cancel the pool on clean drain, on Ctrl-C /
  SIGTERM, and rely on the existing heartbeat net for hard crashes.
- Fail fast and clearly on any misconfiguration.

## Non-goals (explicitly out of scope for this minimal version)

- Status polling / deficit reconciliation (resubmitting dead workers). If a worker's
  node dies, the existing `Scheduler` stall-sweep re-pends its in-flight task and a
  surviving worker picks it up; we just lose that worker's throughput for the rest of
  the run. This is the accepted trade for "minimal".
- Pool-state persistence / restart recovery (Relay's `pool_state.json`).
- Per-scheduler built-in parsers. A single user-supplied regex covers job-id
  extraction for any scheduler.
- Batch cancel (`{{job_ids}}`). Cancel is invoked once per job id.
- SSH-proxy shell override. Users who must submit from a login node bake `ssh` into
  the `submit` command itself (the script path is on the shared filesystem and thus
  visible on the login node).

## Provisioning lifecycle chosen: submit-once + cancel-on-shutdown

At startup the provisioner submits `pool_size` worker jobs, records their scheduler
job ids, and cancels all of them on shutdown. There is no status polling and no
resubmission. Correctness rests on the existing queue machinery:

- All tasks are enqueued before workers start, so `pending/` only ever shrinks.
- Workers run `--persistent`, so they do **not** exit on a transient empty queue.
  This matters: if a node dies and the `Scheduler` re-pends its task, persistent
  workers are still alive to claim it, instead of having exited and hanging the
  drain.
- Persistent workers self-terminate when the manager's heartbeat goes stale
  (`WorkerProcess.cs` "manager heartbeat stalled"), which ends their cluster job.

This leaves a clean seam to grow into Relay-style deficit reconciliation later
(see "Future work").

## CLI options (added to `DistributedOptions`, "Advanced remote work distribution" group)

- `--cluster_script <path>` — path to the submission-script template file.
- `--cluster_config <path>` — path to the queue-definition JSON.
- `--pool_size <N>` — number of worker jobs to submit (`int`).
- `--cluster_var key=value` — repeatable (`IEnumerable<string>`); arbitrary template
  values. Lenient around `=` (see "cluster_var parsing").

### Mode selection and validation

Three mutually exclusive provisioning modes: **local** (default), **external**
(`--external_provisioner`), **cluster** (new).

Cluster mode is selected when `--cluster_script` is present. Then:

- `--cluster_config` is **required** (error if missing).
- `--pool_size` must be **> 0** (error otherwise).
- `--external_provisioner` must **not** also be set (error on conflict).
- `--device_list` is **ignored** (the scheduler allocates each job its GPU). Emit a
  warning if it is explicitly set.
- `--pool_size` counts cluster **jobs** (one GPU each); `--perdevice` worker processes
  run per job, so the pool holds up to `pool_size × perdevice` workers.

All validation throws early with an actionable message before any task is enqueued.

## New components (all in `WarpLib/Workers/Scheduling/`)

### `ClusterQueueDefinition`

Model + JSON loader for the queue-definition file. Exactly three fields:

```json
{
  "submit": "sbatch {{script_path}}",
  "submit_job_id_regex": "Submitted batch job (\\d+)",
  "cancel": "scancel {{job_id}}"
}
```

- `submit` — command to submit the rendered script. `{{script_path}}` is substituted
  with the path to the script we wrote to disk. Run it, capture stdout.
- `submit_job_id_regex` — the **first capture group** extracts the job id from that
  stdout. One configurable regex covers all schedulers; we ship no per-scheduler
  logic.
- `cancel` — invoked once per tracked job id with `{{job_id}}` substituted.

Loader throws listing any missing or empty field.

### `ClusterVarParser` (static)

Reassembles the flat `--cluster_var` token stream into a
`Dictionary<string,string>`. Because the shell splits on spaces before WarpTools sees
the args, a single logical pair can arrive split across up to three tokens. The
parser accepts all four spellings:

- `partition=gpu` (one token)
- `partition= gpu` (`key=` + value)
- `partition =gpu` (key + `=value`)
- `partition = gpu` (key + `=` + value)

Keys and values are trimmed. A value that itself contains a space must be quoted into
a single shell token (`--cluster_var "account=my project"`). A `--cluster_var` with
no `=` anywhere throws with a clear message.

### `TemplateRenderer` (static)

Regex `{{\s*name\s*}}` substitution, whitespace-tolerant, **literal** replacement so
a `$` in a value is never treated as a regex backreference. After substitution it
scans for any leftover `{{...}}` and **throws listing every unfilled placeholder
name**. This is the opposite of Relay's silent-empty behaviour and is a hard
requirement: a malformed directive should surface as a clear error, not a silently
broken job script.

### `ClusterProvisioner : IWorkerProvisioner`

The orchestrator. Implements `EnsureWorkers(int target)`, `LiveWorkerCount()`,
`Shutdown()`.

**Constructor**

- Builds the built-in `{{command}}` value. For `--perdevice N` it launches N worker
  processes in the background on the job's one GPU and ends with `wait` (so the job
  holds its allocation until the workers exit):
  ```
  "<abs WarpWorker2>" --device 0 --queue-dir "<queueDir>" --log-dir "<logDir>" --persistent --worker-id "$(hostname)-$$-0" &
  ... (one line per process, index 0..N-1) ...
  wait
  ```
  - Executable is the absolute path `Path.Combine(AppContext.BaseDirectory,
    "WarpWorker2")`, same as `LocalProvisioner`. Assumes the WarpTools install is
    reachable at the same path on compute nodes (typical shared-filesystem HPC).
  - Device is always `0` — each cluster job owns one GPU and sees it as device 0
    (scheduler isolates via `CUDA_VISIBLE_DEVICES`); `--perdevice` processes share it.
  - `--persistent` so workers survive transient empty queues (see lifecycle).
  - `--worker-id "$(hostname)-$$-<i>"` — the compute node's shell expands `$(hostname)`
    and `$$` at runtime; the per-process index `<i>` is required because `$$` (the
    script's pid) is shared by all background children of one job. (Our default worker
    id is `local-{pid}-gpu{dev}` with **no** hostname, which would collide across nodes,
    so an explicit id is mandatory here.)
- Renders the template **once** (built-in `command` + `cluster_var` values) to
  `<queue>/cluster/worker.sh`. Every job is byte-identical, so one script is submitted
  `pool_size` times (each submission running `--perdevice` workers).
- Registers `Console.CancelKeyPress` and `PosixSignalRegistration` for SIGINT/SIGTERM,
  each calling `Shutdown()`, so a Ctrl-C or `kill` on the manager cancels the pool
  before the process dies.

**`EnsureWorkers(int target)`**

- Submits `target - submittedCount` jobs (on the first tick this submits the whole
  pool; subsequent ticks no-op). This matches the shape the `Scheduler` already
  drives every tick and leaves the seam for future deficit reconciliation.
- For each submit: run the `submit` command (with `{{script_path}}`) through the
  shell, capture stdout, apply `submit_job_id_regex`. **Throw if no match** — an
  untrackable job cannot be cancelled and would linger.
- Store each parsed job id.

**`LiveWorkerCount()`** — returns the count of submitted job ids (no scheduler query
in this minimal version).

**`Shutdown()`** — idempotent (cancel-once guard, since the `finally` block and the
signal handlers may all call it). Runs the `cancel` command (with `{{job_id}}`) once
per stored job id. Disposes the signal registrations.

## Integration into `DistributedOptions`

The provisioner-selection block is currently duplicated in `DistributeItems`
(lines ~101-125) and `DistributeTasks` (~273-291). Extract it into one helper:

```csharp
private IWorkerProvisioner CreateProvisioner(
    QueueLayout layout, string logDir, int itemCount, out int target)
```

with three branches:

- **cluster**: `target = Math.Min(itemCount, PoolSize)`; construct `ClusterProvisioner`.
- **external**: `target = 0`; construct `ExternalProvisioner`.
- **local**: resolve devices; `target = Math.Min(itemCount, devices * perdevice)`;
  construct `LocalProvisioner`.

Both `DistributeItems` and `DistributeTasks` call the helper, so cluster mode works
for every ported command and for whole-run reduce tasks for free.

## Data / control flow

1. Enqueue all tasks (before the scheduler thread starts).
2. `Scheduler` tick → `EnsureWorkers(pool_size)` → render-once → `submit` × N →
   parse and store job ids.
3. Cluster workers land on compute nodes, self-name `$(hostname)-$$-<i>`, claim tasks
   from the shared queue, run `--persistent`.
4. Manager polls `done/`/`poisoned/` until drained.
5. On drain (or Ctrl-C / SIGTERM) → `Shutdown()` → `cancel` every stored job id.
6. Hard crash without cleanup: running workers self-exit on stale manager heartbeat;
   only still-pending scheduler jobs would linger, which is exactly what the explicit
   cancel in step 5 covers.

## Failure / validation summary (all throw with actionable messages)

- `--cluster_script` set but `--cluster_config` missing.
- `--pool_size` <= 0 in cluster mode.
- `--cluster_script` and `--external_provisioner` both set.
- Malformed `--cluster_var` (no `=`).
- Queue-definition JSON missing/empty field.
- Unfilled `{{placeholder}}` after rendering (message lists every unfilled name).
- `submit` stdout does not match `submit_job_id_regex`.

## Testing

- **Unit — `ClusterVarParser`**: the four `=` spacings, a quoted value containing a
  space, and the no-`=` error.
- **Unit — `TemplateRenderer`**: substitution (whitespace-tolerant, `$` literal) and
  the unfilled-placeholder crash listing names.
- **Unit — `ClusterQueueDefinition`**: load success and missing-field errors.
- **Integration (Unix) — `ClusterProvisioner`**: a fake scheduler where `submit` is
  `echo "Submitted batch job 12345"` (varying the id) and `cancel` touches a marker
  file. Assert `EnsureWorkers` submits `pool_size` times, parses the ids, and
  `Shutdown` runs `cancel` for each. No GPU or real cluster needed.

## Stated assumptions (surface in `--help` / docs)

- The queue directory is on a **shared filesystem** visible to all compute nodes.
- The WarpTools install is reachable at the **same path** on compute nodes.
- The submission script runs in a shell that expands `$(hostname)` and `$$`.

## Future work (the seam left for Relay-grade behaviour)

- Add optional `status`/`list` commands to the queue definition and have
  `EnsureWorkers` resubmit to maintain the pool (deficit reconciliation) with a
  submission cap.
- Pool-state persistence for manager restart recovery.
- Batch cancel via `{{job_ids}}`.
