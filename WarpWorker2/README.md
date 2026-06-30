# WarpWorker2 — Filesystem Queue Worker

Queue-consuming worker binary. Lifts the `WarpCore` branch's reflection-based
command dispatch and command bodies, drops all REST/controller networking. Reads
commands from claimed task files instead.

Full architecture: `docs/superpowers/specs/2026-06-03-filesystem-work-distribution.md`.

---

## File layout

```
WarpWorker2/
  WarpWorker2.csproj       — exe; Debug=AnyCPU (no CUDA), Release=x64
  OptionsCLI.cs            — --device, --queue-dir, --stages, --mock, --debug, …
  WorkerProcess.cs         — Main, GPU bind, claim loop, exception taxonomy,
                             heartbeats, blacklist check, RegisterCommands()
  GpuHealthProbe.cs        — small FFT round-trip to detect dead GPU hardware
  Commands/
    CommandAttribute.cs    — [Command] / [MockCommand] attributes
    Movie.cs               — LoadStack, MovieProcessCTF, MovieProcessMovement,
                             MovieExportMovie, MovieCreateThumbnail, MoviePickBoxNet,
                             MovieExportParticles (real GPU handlers)
    Service.cs             — LoadGainRef, SetHeaderlessParams, GcCollect,
                             WaitAsyncTasks (shared service commands)
  MockCommands/
    MovieMock.cs           — MockLoadStack, MockMovieProcessCTF — CPU-only
                             stubs used with --mock for GPU-free testing
  DataLoading.cs           — shared data-loading helpers
```

---

## Worker lifecycle

```
Main()
  │
  ├─ Parse --mock before any GPU calls
  │
  ├─ if !mock: GPU.SetDevice(device)
  │
  ├─ RegisterCommands()  ← scans [Command] / [MockCommand] attributes
  │
  ├─ Create running/<wid>/ dir, write hostname file
  │
  ├─ if !mock: GpuHealthProbe.Probe()
  │     └─ fail → write sick/ marker, exit(3)
  │
  └─ Claim loop ──────────────────────────────────────────────────────┐
       │                                                              │
       ├─ HeartbeatWriter.WriteTick()   (running/<wid>/hb-N)         │
       ├─ HeartbeatMonitor.Observe()    (heartbeat/tick-N)           │
       ├─ if stalled && !debug → exit ("manager heartbeat stalled")  │
       ├─ if host blacklisted → exit ("host blacklisted")            │
       │                                                              │
       ├─ queue.ClaimOne() ─── null ──► consecutiveEmpty++           │
       │                                  ≥ 2 → exit ("queue empty") │
       │                                  else sleep(500ms)  ─────── ┘
       │
       ├─ if task.InitFingerprint != lastFingerprint:
       │     run Init sequence
       │     exception → probe
       │       probe FAIL  → sick + exit (leave task for sweep)
       │       probe PASS  → ResetResourceState + clear fp + MarkFailed + continue
       │     success       → lastFingerprint = task.InitFingerprint
       │
       └─ run Main sequence
             exception → probe
               probe FAIL  → sick + exit (leave task for sweep)
               probe PASS  → MarkFailed + continue  (NO reset — init state preserved)
             success       → MarkDone + continue ──────────────────────────────────┘
```

---

## Command dispatch

Commands are registered at startup by scanning `WorkerProcess` methods for
`[Command(name)]` / `[MockCommand(name)]` attributes, building two dictionaries.

```csharp
// Registration (once, at startup):
//   CommandMethods["LoadStack"]        → MethodInfo for LoadStack(NamedSerializableObject)
//   MockCommandMethods["LoadStack"]    → MethodInfo for MockLoadStack(NamedSerializableObject)

// Dispatch (per command in Init/Main):
if (MockMode && MockCommandMethods.TryGetValue(cmd.Name, out var mock))
    mock.Invoke(null, new object[] { cmd });
else
    CommandMethods[cmd.Name].Invoke(null, new object[] { cmd });
```

Command names are `nameof(WorkerWrapper.X)` so a rename in `WorkerWrapper` is
caught at compile time. `WorkerCommands` (WarpLib) uses the same names — a task
file built by `WorkerCommands.LoadStack(...)` is dispatched to the worker's
`LoadStack` handler without any string duplication.

---

## Mock mode — fully GPU-free

`--mock` skips every CUDA call:

| Call | Real mode | Mock mode |
|---|---|---|
| `GPU.GetDeviceCount()` | startup | **skipped** |
| `GPU.SetDevice()` | startup + EvaluateCommand + ResetResourceState | **skipped** |
| `GpuHealthProbe.Probe()` | startup + after exceptions | **skipped** |
| Command handlers | real GPU work | `MockCommand` stubs (CPU-only allocations) |

This lets mock mode run on arm64 dev machines and CI runners without
`NativeAcceleration.dll`. The `--device` argument is accepted but used only as an
identifier in the worker ID (`local-<pid>-gpu<N>`).

---

## Build configuration

| Config | Platform | Output | Notes |
|---|---|---|---|
| Debug \| AnyCPU | AnyCPU | `../bin/` | No CUDA needed; runs on arm64 |
| Release \| AnyCPU | x64 | `../Release/` | CUDA required; production |

The `Tests` project copies all `bin/WarpWorker2*` files to its output directory
via a `CopyWorkerBinary` MSBuild target, so the end-to-end mock test runs without
manual setup.

---

## Adding a new command

1. Add the handler method to the appropriate `Commands/*.cs` or `MockCommands/*.cs`
   file (as a `static partial class WorkerProcess` method).
2. Tag it `[Command(nameof(WorkerWrapper.YourMethod))]` (or `[MockCommand(...)]`).
3. Add the factory method to `WarpLib/Workers/WorkerCommands.cs`.
4. Add the task builder call in the relevant WarpTools command using
   `WorkerCommands.YourMethod(...)`.

No other registration or wiring is needed — the reflection scan at startup picks
up the new attribute automatically.
