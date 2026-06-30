[![Latest Release](https://img.shields.io/conda/vn/warpem/warp.svg)](https://anaconda.org/warpem/warp)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13982246-blue)](https://doi.org/10.5281/zenodo.13982246)

# Warp?

Warp is a set of tools for cryo-EM and cryo-ET data processing including, among other tools: [Warp](https://doi.org/10.1038/s41592-019-0580-y), [M](https://doi.org/10.1038/s41592-020-01054-7), WarpTools, MTools, MCore, and Noise2Map.

# Citing Warp

https://doi.org/10.5281/zenodo.13982246

This DOI represents all versions and will resolve to the latest version.

# Install Warp

## Windows

If you want to use Warp on Windows, tutorials and binaries (currently only for v1) can be found at http://www.warpem.com.

## Linux

If you're installing from scratch and don't have an environment yet, here is the easiest way to get everything inside a new environment called `warp`:
```
conda create -n warp warp -c warpem -c nvidia/label/cuda-12.9.0 -c conda-forge --channel-priority flexible
conda activate warp  # Activate the environment whenever you want to use Warp
```

If you're installing on a machine without an NVIDIA GPU (e.g. a cluster login node), set `CONDA_OVERRIDE_CUDA` so that conda can resolve the CUDA dependencies:
```
CONDA_OVERRIDE_CUDA=12.9 conda create -n warp warp -c warpem -c nvidia/label/cuda-12.9.0 -c conda-forge --channel-priority flexible
```

If you want to install in an already existing environment:
```
conda install warp -c warpem -c nvidia/label/cuda-12.9.0 -c conda-forge --channel-priority flexible
```

If you want to update to the latest version and already have all channels set up in your environment:
```
conda update warp -c warpem
```

### Upgrading from v2.0.0dev37 or earlier

Versions up to v2.0.0dev37 used CUDA 11.8 and .NET 8. Starting with v2.0.0dev38, Warp requires CUDA 12.9 and .NET 10. These changes are too large for `conda update` to handle, so a fresh environment is needed:
```
conda env remove -n warp
conda create -n warp warp -c warpem -c nvidia/label/cuda-12.9.0 -c conda-forge --channel-priority flexible
```

You will also need an NVIDIA driver that supports CUDA 12.x (version >= 525.60.13). Check with `nvidia-smi`.

# Use Warp

For information on how to use Warp, M and friends please check out the user guide section
of [warpem.github.io/warp](https://warpem.github.io/warp/).

# Distributing WarpTools across a cluster (SLURM example)

Every WarpTools command that processes many items in parallel (e.g. `fs_motion_and_ctf`,
`fs_ctf`, `ts_ctf`, `ts_reconstruct`, …) can spread the work across a pool of worker
processes. By default it spawns those workers locally, one per GPU. It can instead
submit a pool of workers to a batch scheduler such as SLURM, where each worker is a job
that claims tasks from a shared work queue. You point WarpTools at two files — a
submission-script template and a small cluster-config JSON — and it submits `--pool_size`
identical worker jobs, waits for them to chew through the queue, then cancels them when
the run is done (also on Ctrl-C).

> This is a minimal, built-in alternative to a full workflow manager. It assumes the
> output/queue directory is on a **shared filesystem** visible to every compute node, and
> that the Warp install is reachable at the **same path** on the compute nodes (e.g. via
> the same module). For larger deployments, Relay will provide richer scheduling.

## 1. The cluster-config JSON

Describes how to talk to the scheduler. Exactly three fields:

```json
{
  "submit": "sbatch {{script_path}}",
  "submit_job_id_regex": "Submitted batch job (\\d+)",
  "cancel": "scancel {{job_id}}"
}
```

- `submit` — command that submits the rendered script. `{{script_path}}` is filled in
  with the path of the script WarpTools writes. WarpTools reads the command's stdout to
  learn the job id.
- `submit_job_id_regex` — the **first capture group** extracts the job id from that
  stdout. The example matches the default `sbatch` output. (If you prefer, use
  `"submit": "sbatch --parsable {{script_path}}"` together with `"submit_job_id_regex": "(\\d+)"`.)
- `cancel` — runs once per job id, with `{{job_id}}` substituted, when the run finishes
  or is interrupted.

Note the doubled backslash (`\\d`) — it is a JSON string, so the backslash must be escaped.

## 2. The submission-script template

A normal SLURM batch script with `{{ }}` placeholders. The only required placeholder is
`{{command}}`, which WarpTools replaces with the full worker invocation (it already
contains the queue directory, the log directory, the GPU device, and a unique worker id).
Everything else is up to you; any `{{custom}}` placeholders you add are filled from the
command line (next step).

```bash
#!/bin/bash
#SBATCH --job-name=warp-worker
#SBATCH --partition={{partition}}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={{cpus}}
#SBATCH --mem={{mem}}
#SBATCH --time={{time}}
#SBATCH --output=warp-worker-%j.out

module load warp

{{command}}
```

A few notes on `{{command}}`:

- Each worker job requests **one GPU** (`--gres=gpu:1`); the worker(s) use device `0`,
  i.e. whatever SLURM exposes to that job via `CUDA_VISIBLE_DEVICES`.
- With `--perdevice N` (see below), `{{command}}` launches **N worker processes** in the
  background on that GPU and ends with `wait`, so the job holds its allocation until they
  finish. Each process names itself `"$(hostname)-$$-<i>"` — your shell expands
  `$(hostname)` and `$$` on the compute node, and the index keeps ids unique across the
  whole pool. This is why the script must run in a shell.
- Workers are started with `--persistent` so a momentarily empty queue does not make them
  quit early; WarpTools cancels them (via `cancel`) once everything is done.

## 3. Run a command in cluster mode

Add four options to any distributed WarpTools command:

```bash
WarpTools fs_motion_and_ctf \
    --settings warp_frameseries.settings \
    --cluster_script worker.slurm \
    --cluster_config slurm.json \
    --pool_size 16 \
    --perdevice 2 \
    --cluster_var partition=gpu \
    --cluster_var time=04:00:00 \
    --cluster_var cpus=8 \
    --cluster_var mem=32G
```

- `--cluster_script` — path to the submission-script template. Its presence switches the
  command into cluster mode.
- `--cluster_config` — path to the cluster-config JSON.
- `--pool_size` — how many worker **jobs** to submit (one GPU each).
- `--perdevice` — worker processes per job (per GPU); default `1`. The pool holds up to
  `pool_size × perdevice` workers — the example above is 16 GPUs × 2 = 32 workers.
- `--cluster_var key=value` — repeatable; fills one `{{key}}` placeholder in the template.
  Whitespace around `=` is tolerated (`--cluster_var partition = gpu` works too); quote
  values that contain spaces, e.g. `--cluster_var "account=my project"`.

In cluster mode `--device_list` is ignored (the scheduler allocates each job its GPU). If
the template still contains any placeholder you didn't provide, the command stops
immediately and tells you which ones are missing, rather than submitting a broken script.

# Build Warp on Linux

After cloning this repository, run these commands:
```
conda env create -f warp_build.yml --channel-priority flexible
conda activate warp_build
./scripts/build-native-unix.sh
./scripts/publish-unix.sh
```

If you're building on a machine without an NVIDIA GPU, set `CONDA_OVERRIDE_CUDA=12.9` before creating the environment so that conda installs the CUDA variant of PyTorch:
```
CONDA_OVERRIDE_CUDA=12.9 conda env create -f warp_build.yml --channel-priority flexible
```
All binaries will be in `Release/linux-x64/publish`.

Here is some inspiration for an lmod module file:
```
local root = "/path/to/warp/Release/linux-x64/publish"

conflict("warp")

prepend_path("PATH", root)
prepend_path("LD_LIBRARY_PATH", "/path/to/conda_envs/warp_build/lib")
setenv("RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE", pathJoin(root, "Noise2Half"))
```

# Other programs you'll want to install (on Linux)

- [IMOD](https://bio3d.colorado.edu/imod/download.html#Development) `>=4.12.50`
- [AreTomo](https://msg.ucsf.edu/software) `==1.3.4`
- [RELION](https://github.com/3dem/relion) `==5`

AreTomo2 seems broken (c.f. [#159](https://github.com/warpem/warp/issues/159) and [AreTomo2/#21](https://github.com/czimaginginstitute/AreTomo2/issues/21)) and we have not added compatibility for AreTomo3 ([#279](https://github.com/warpem/warp/issues/279)).

We only ensure compatibility with RELION 5.

## Editing Documentation
Install `mkdocs-material` into your conda environment then run

```sh
mkdocs serve
```

To preview the site. This includes hot reloading so you can preview any changes you make.

The documentation is built and deployed by calling `mkdocs build` on GitHub actions.

## Authorship

Warp was originally developed by [Dimitry Tegunov](mailto:tegunov@gmail.com) in Patrick Cramer's lab at the Max Planck Institute for Biophysical Chemistry in Göttingen, Germany. This code is available [in its original repository](https://github.com/cramerlab/warp).

Warp is now being developed by Dimitry Tegunov at Genentech, Inc. in South San Francisco, USA. For a list of changes that occurred between the last release under the Max Planck Society and the first release at Genentech, please see CHANGELOG.
