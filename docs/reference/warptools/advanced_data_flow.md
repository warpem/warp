# Advanced Data Flow

By default *WarpTools* directories for raw data and processed data are specified
in a `.settings` text file created with the `create_settings` tool.

This page is for advanced users who want to customise data flow in their *WarpTools* 
processing.

## Concepts

Customising data flow using *WarpTools* requires understanding a few key concepts.

1. raw data directory
2. processing directory

### Raw data directory

The raw data directory contains what *WarpTools* considers the raw data.
For frame series, raw data are movie files, usually `.mrc` or `.tif`. 
For tilt series, raw data are `.tomostar` files.

### Processing directory

Outputs from all *WarpTools* programs will be written into the processing directory.

Per movie or per tilt series metadata will be written in `.xml` files in the root of
the processing directory whilst images and other process specific data are written into 
subdirectories of the processing directory with names like `average`, `matching` or
`reconstruction`.

## Redirecting the Flow

Now that we know the basics, we're ready to redirect the processing flow.

Here are the relevant options available in all *WarpTools* commands.

### Options
#### `--input_data`

The `--input_data` option overrides the list of input files specified in the .settings file. 
It accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.

#### `--input_processing`

`--input_processing` specifies an alternative directory containing pre-processed results.
This overrides the processing directory specified in the `.settings` file and affects 
both file **input** and **output**.

#### `--output_processing`

`--output_processing` specifies an alternative directory to save processing results. 
This also overrides the processing directory in the .settings file, but only for file
**output**.

### Examples

Let's go through some examples...

### Generating Particle Stacks for Multiple Species

When *WarpTools* generates 2D or 3D particles from tilt series data it writes image files
into `<processing_directory>/particleseries` and `<processing_directory>/subtomo` respectively
with file names like `TS_1_4.00A_000001.mrcs`.


If you're working on many different objects in your data and want to extract particles 
at the same pixel size you risk overwriting previous particle sets. By specifying
`--output_processing`, we redirect the output to a new directory which we will call
`relion_15854` here.

```sh
WarpTools ts_export_particles \
--settings warp_tiltseries.settings \
--input_directory warp_tiltseries/matching \
--input_pattern "*15854_clean.star" \
--output_processing relion_15854 \
--output_angpix 4 \
--output_star relion_15854/matching_4apx.star \
--relative_output_paths \
--normalized_coords \
--box 96 \
--diameter 130 \
--2d 
```

This produces a new directory `relion_15854` the following structure

```txt
relion_15854
├── logs
├── particleseries
│   ├── TS_1
│   ├── TS_11
│   ├── TS_17
│   ├── TS_23
│   └── TS_32
├── dummy_tiltseries.mrc
├── matching_4apx_optimisation_set.star
├── matching_4apx.star
└── matching_4apx_tomograms.star
```

### Running different tilt-series alignment programs

We can use the `--output_processing` option as we did in the particle export example
to test different tilt series alignment methods. 
We will need to add `--input_processing` to our `ts_reconstruct` call 
if we want to reconstruct using these alignments.

Let's run both *Etomo*'s patch tracking and *AreTomo* on some data

```sh
WarpTools ts_etomo_patches \
--settings warp_tiltseries.settings \
--angpix 10 \
--patch_size 1000 \
--do_axis_search \
--output_processing etomo_patches_1000
```

```sh
WarpTools ts_aretomo \
--settings warp_tiltseries.settings \
--angpix 10 \
--alignz 800 \
--axis_iter 5 \
--min_fov 0 \
--output_processing aretomo_alignz_800
```

To reconstruct each of these datasets, we will need to specify the `--output_processing`
directory as the `--input_processing` directory for `ts_reconstruct`.

```sh
WarpTools ts_reconstruct \
--settings warp_tiltseries.settings \
--input_processing aretomo_alignz_800 \
--angpix 10 
```

!!! tip
    
    If `--output_processing` is not specified then output will be written 
    into the same directory. Output can be further redirected by specifying `--output_processing`.

## Some caveats

This mechanism is imperfect, if you try to run a process which depends on earlier results
you might run into trouble. We're working on this 🙂