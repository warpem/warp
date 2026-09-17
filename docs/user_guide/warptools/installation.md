# Installation

## Linux

We distribute *WarpTools* as part of a conda package for *Warp* on x86-64 (`linux-64`) and ARM64 (`linux-aarch64`, including NVIDIA GH200). Both packages use CUDA 12.9; Conda selects the native architecture automatically. ARM64 packages are available starting with v2.0.0dev41.

### Installing Conda

If you're new to the *conda* package manager we recommend installing [`mambaforge`](https://conda-forge.org/miniforge/).

### Creating a conda environment and installing Warp into it

The following command will create a new environment called `warp` and install `warp` and all
dependencies into it.

```sh
conda create -n warp warp -c warpem -c nvidia/label/cuda-12.9.0 -c conda-forge --channel-priority flexible
```

The environment can then be activated whenever you want to use *WarpTools*

```sh
conda activate warp
```

### Updating

To update your installation, run the following command

```sh
conda update warp -c warpem -c nvidia/label/cuda-12.9.0 -c conda-forge --channel-priority flexible
```

## Checking what version you are running

To check which version you are running use

```sh
conda list warp
```

## Windows

We don't currently provide pre-built binaries for *WarpTools* on Windows.
