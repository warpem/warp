# Installation

## Linux

We distribute *WarpTools* as part of a conda package for *Warp*.

### Installing Conda

If you're new to the *conda* package manager we recommend installing [`mambaforge`](https://conda-forge.org/miniforge/). 

### Creating a conda environment and installing Warp into it

The following command will create a new environment called `warp` and install `warp`and all 
dependencies into it.

```sh
conda create -n warp warp=2.0.0 -c warpem -c nvidia/label/cuda-11.8.0 -c pytorch -c conda-forge
```

The environment can then be activated whenever you want to use *WarpTools*

```sh
conda activate warp 
```

### Updating

To update your installation, run the following command

```sh
conda update warp -c warpem -c nvidia/label/cuda-11.8.0 -c pytorch -c conda-forge
```

## Windows

We don't currently provide pre-built binaries for *WarpTools* on Windows.