# Warp?

Warp is a set of tools for cryo-EM and cryo-ET data processing including, among other tools: [Warp](https://doi.org/10.1038/s41592-019-0580-y), [M](https://doi.org/10.1038/s41592-020-01054-7), WarpTools, MTools, MCore, and Noise2Map.

# Using Warp

If you want to use Warp on Windows, tutorials and binaries can be found at http://www.warpem.com.
Obtain the Linux CLI version by installing this Conda package: `conda install warp -c conda-forge`

While there are currently no dedicated tutorials for the Linux CLI tools, you'll get a very good idea of how to use WarpTools, MTools & MCore by looking at the EMPIAR-10491_5TS_e2e.sh script. This is an end-to-end test that starts by downloading 5 tilt series from EMPIAR-10491, pre-processing them in Warp and IMOD, refining the particles in RELION, and then finishing with multi-particle refinement in MCore to obtain a ca. 2.9 Å apoferritin map.

# Building Warp on Linux

After cloning this repository, run these commands on a system with a running Nvidia GPU driver (conda-forge pytorch weirdness ¯\\\_(ツ)\_/¯ ) and a discoverable CUDA 11.7:
```
conda env create -f warp_build.yml
conda activate warp_build
./build-native-unix.sh
./publish-unix.sh
```
All binaries will be in `Release/linux-x64/publish`.

Here is some inspiration for an lmod module file:
```
local root = "/path/to/warp/Release/linux-x64/publish"

conflict("warp")

if not isloaded("CUDA/11.7.0") then
    load("CUDA/11.7.0")
end

prepend_path("PATH", root)
prepend_path("LD_LIBRARY_PATH", "/path/to/conda_envs/warp_build/lib")
setenv("RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE", pathJoin(root, "Noise2Half"))
```

# Other programs you'll want to install (on Linux)

- [IMOD](https://bio3d.colorado.edu/imod/)
- [AreTomo](https://github.com/czimaginginstitute/AreTomo2)
- [RELION](https://github.com/3dem/relion)

## Authorship

Warp was originally developed by [Dimitry Tegunov](mailto:tegunov@gmail.com) in Patrick Cramer's lab at the Max Planck Institute for Biophysical Chemistry in Göttingen, Germany. This code is available [in its original repository](https://github.com/cramerlab/warp).

Warp is now being developed by Dimitry Tegunov and Alister Burt at Genentech, Inc. in South San Francisco, USA. For a list of changes that occurred between the last release under the Max Planck Society and the first release at Genentech, please see CHANGELOG.
