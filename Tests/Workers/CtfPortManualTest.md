# fs_ctf new-architecture acceptance (manual, needs GPU)

The `fs_ctf` (frame-series CTF) command was ported from the legacy
`WorkerWrapper` / `IterateOverItems` REST distribution to the filesystem
work-distribution path (`WorkPool` + `Scheduler` + `LocalProvisioner` +
`WarpWorker2`). This is the manual GPU acceptance run — it is NOT part of CI
(the dev/CI machines have no CUDA GPU, and the worker binary is x64-only).

## Procedure

1. Build everything on a Linux x64 machine with CUDA GPUs:
   `dotnet build Warp.sln -c Release`
2. Produce a GOLDEN run with the pre-port binary (the commit immediately
   before the fs_ctf port) on a small standard test set: run `fs_ctf` and keep
   the per-movie `*.xml` meta (defocus, defocus delta/angle, phase shift, fit
   resolution).
3. Run `fs_ctf` with the new binary on the same test set and the same options,
   including a run WITH a gain reference configured (see the gain note below).
4. Compare per-movie CTF estimates (defocus, delta, angle, phase, resolution)
   against the golden run.

## Acceptance

- Per-movie CTF values match the golden run within numerical noise (same
  algorithm, same options, same inputs — differences should be at the level of
  nondeterministic GPU reduction order, not algorithmic).
- `processed_items.json` lists every successfully processed movie;
  `failed_items.json` (if present) lists only genuinely failing movies, which
  are also marked `UnselectManual` / `LeaveOut` in their meta.
- Per-movie logs appear under `<output>/logs/` (worker stdout per task — wiring
  of per-task log files into `logs/<wid>.log` is a Phase-2 item; at minimum the
  worker `.exit` markers and any worker stdout should be inspectable).

## Gain-reference check (important)

The old path loaded the gain reference into every worker up front
(`DistributedOptions.GetWorkers`). The port preserves this by placing
`LoadGainRef` in the task **init** sequence: it is identical for every movie, so
all tasks share an init fingerprint and each worker runs it exactly once, then
reuses the loaded gain across all its tasks. **Explicitly verify** a dataset
WITH a configured gain reference (and ideally a defect map) produces CTF
estimates matching the golden run — a regression here would silently CTF-fit
uncorrected movies.

## Notes

- The queue lives under `<output>/work_ctf/`. After a successful run it should
  contain only `done/` entries (no stragglers in `pending/`, `running/`,
  `failed/`); `poisoned/` entries correspond to `failed_items.json`.
- Remote `--workers hostname:port` distribution is intentionally NOT supported
  by this path (cluster execution is Relay's job via the shared queue); the
  command throws early if `--workers` is supplied.
