# Warp?

Warp is a set of tools for cryo-EM and cryo-ET data processing including, among other tools: [Warp](https://doi.org/10.1038/s41592-019-0580-y), [M](https://doi.org/10.1038/s41592-020-01054-7), WarpTools, MTools, MCore, and Noise2Map.

# Install Warp

## Windows

If you want to use Warp on Windows, tutorials and binaries (currently only for v1) can be found at http://www.warpem.com.

## Linux

If you're installing from scratch and don't have an environment yet, here is the easiest way to get everything inside a new environment called `warp`:
```
conda create -n warp warp -c warpem -c nvidia/label/cuda-11.7.0 -c pytorch -c conda-forge
conda activate warp  # Activate the environment whenever you want to use Warp
```

If you want to install in an already existing environment:
```
conda install warp -c warpem -c nvidia/label/cuda-11.7.0 -c pytorch -c conda-forge
```

If you want to update to the latest version and already have all channels set up in your environment:
```
conda update warp
```

# Use Warp

For information on how to use Warp, M and friends please check out the user guide section
of [warpem.github.io/warp](https://warpem.github.io/warp/).

# Build Warp on Linux

After cloning this repository, run these commands:
```
conda env create -f warp_build.yml
conda activate warp_build
./scripts/build-native-unix.sh
./scripts/publish-unix.sh
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

- [IMOD](https://bio3d.colorado.edu/imod/)
- [AreTomo](https://github.com/czimaginginstitute/AreTomo2)
- [RELION](https://github.com/3dem/relion)

## Editing Documentation
Install `mkdocs-material` into your conda environment then run

```sh
mkdocs serve
```

To preview the site. This includes hot reloading so you can preview any changes you make.

The documentation is built and deployed by calling `mkdocs build` on GitHub actions.

## Authorship

Warp was originally developed by [Dimitry Tegunov](mailto:tegunov@gmail.com) in Patrick Cramer's lab at the Max Planck Institute for Biophysical Chemistry in Göttingen, Germany. This code is available [in its original repository](https://github.com/cramerlab/warp).

Warp is now being developed by Dimitry Tegunov and Alister Burt at Genentech, Inc. in South San Francisco, USA. For a list of changes that occurred between the last release under the Max Planck Society and the first release at Genentech, please see CHANGELOG.
