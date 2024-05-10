# Warp?

Warp is a set of tools for cryo-EM and cryo-ET data processing including, among other tools: [Warp](https://doi.org/10.1038/s41592-019-0580-y), [M](https://doi.org/10.1038/s41592-020-01054-7), WarpTools, MTools, MCore, and Noise2Map.

# Using Warp

If you want to use Warp on Windows, tutorials and binaries (currently only for v1) can be found at http://www.warpem.com.

~~Obtain the Linux CLI version by installing this Conda package: `conda install warp -c conda-forge`~~ The conda-forge package isn't ready yet! Check back in a few days.

While there are currently no dedicated tutorials for the Linux CLI tools, you'll get a very good idea of how to use WarpTools, MTools & MCore by looking at the EMPIAR-10491_5TS_e2e.sh script. This is an end-to-end test that starts by downloading 5 tilt series from EMPIAR-10491, pre-processing them in Warp and IMOD, refining the particles in RELION, and then finishing with multi-particle refinement in MCore to obtain a ca. 2.9 Å apoferritin map.

# Building Warp on Linux

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
