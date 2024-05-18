# Working with subsets of data in WarpTools

This page explains the mechanisms available for working with subsets of data in
*WarpTools*.

Each program in *WarpTools* operates over potentially large sets of frame series or tilt
series data. Sometimes, we want to work with subsets.

## Using `--input_data`

Adding `--input_data` when calling any *WarpTool* allows you to run it on one or
multiple frame series, overriding the list of input files specified in a `.settings` file.

Input data can be provided as

- a space-separated list of files
- a wildcard pattern
- a `.txt` file with one file name per line

### Examples

#### One Frame Series

`--input_data frames/file.tif`

### Multiple Frame Series

`--input_data frames/file1.tif frames/file2.tif`

### One Tilt Series

`--input_data tomostar/TS_1.tomostar`

### Multiple Tilt Series

`--input_data tomostar/TS_1.tomostar tomostar/TS_2.tomostar`

## Exclusion from Further Processing with `WarpTools change_selection`

It is possible to exclude frame series or tilt series from further processing using
the `change_selection` *WarpTool* to deselect them.

### Deselecting a frame series

```shell
WarpTools change_selection \
--settings warp_frameseries.settings \
--deselect \
--input_data frames/2Dvs3D_53-1_00001_-0.0_Jul31_10.36.03.tif
```

```text
Running command change_selection with:
select = False
deselect = True
null = False
invert = False
settings = warp_frameseries.settings
input_data = { frames/2Dvs3D_53-1_00001_-0.0_Jul31_10.36.03.tif }
input_data_recursive = False
input_processing = null
output_processing = null

Using alternative input specified by --input_data
File search will be relative to /projects/site/gred/cryoem/burta2/cryo_tomo_test_data/EMPIAR-10491-5TS
1 files found
Parsing previous results for each item, if available...
1/1, previous metadata found for 1                                                                                                                                          
1/1 processed                                                                                                                                                               
Before change: 0 deselected, 0 selected, 1 null
After change: 1 deselected, 0 selected, 0 null
```

### Deselecting a tilt series

```shell
WarpTools change_selection \
--settings warp_tiltseries.settings \
--deselect \
--input_data tomostar/TS_1.tomostar
```

```text
Running command change_selection with:
select = False
deselect = True
null = False
invert = False
settings = warp_tiltseries.settings
input_data = { tomostar/TS_1.tomostar }
input_data_recursive = False
input_processing = null
output_processing = null

Using alternative input specified by --input_data
File search will be relative to /projects/site/gred/cryoem/burta2/cryo_tomo_test_data/EMPIAR-10491-5TS
1 files found
Parsing previous results for each item, if available...
1/1, previous metadata found for 1                                                                                                                                          
1/1 processed                                                                                                                                                               
Before change: 0 deselected, 1 selected, 0 null
After change: 1 deselected, 0 selected, 0 null
```

### Under the hood

Under the hood, this selection mechanism modifies the `UnselectManual` entry in the
`.xml` file associated with each frame series or tilt series.

Below is an example of the first two lines of a deselected tilt series xml file.

```xml
<?xml version="1.0" encoding="utf-8"?>
<TiltSeries DataDirectory="/projects/EMPIAR-10491-5TS/tomostar" AreAnglesInverted="True"
            PlaneNormal="-0.13401411, 0.0035971305, 0.9909729" Bfactor="-7.300099"
            Weight="1" MagnificationCorrection="1, 0, 0, 1" UnselectFilter="False"
            UnselectManual="True" CTFResolutionEstimate="4.3">
```
