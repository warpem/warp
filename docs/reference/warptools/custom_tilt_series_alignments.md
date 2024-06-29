# Custom Tilt Series Alignment Workflows

*WarpTools* provides wrappers around some commonly used programs for
tilt series alignment *IMOD* and *AreTomo2*.

- `ts_aretomo`
- `ts_etomo_fiducials`
- `ts_etomo_patches`

These wrappers provide fully automated solutions and are well integrated into the
*WarpTools* processing flow.

What if you want to do something **different**?

## Tilt Series Alignment in *WarpTools*

Tilt series alignment in *WarpTools* can be broken down into three steps

1. generating tilt series stacks
2. running a tilt series alignment program
3. importing alignments

## Easy as 1, 2, 3...

*WarpTools* provides everything you need to do each of these steps separately yourself
if you want to work outside the wrappers.

### Generate tilt series stacks

*WarpTools* has a subcommand called `ts_stack` which can be used to generate tilt series
at a specific pixel size.
Masked regions from per-tilt mask files in `<processing_directory>/mask` will be
replaced with gaussian noise matching local image statistics which may help subsequent
alignments.

```shell
WarpTools ts_stack \
--settings <settings_file> \
--angpix 10
```

Tilt series in `.mrc` format and tilt angles in the IMOD format
`.rawtlt` file will be placed in subdirectories of `<processing_directory>/tiltstack`.

### Run a tilt series alignment program

Any tilt series alignment program can be run inside these tilt series directories.

### Import alignments

*WarpTools* can import alignment metadata from the *IMOD* format metadata files

- `.xf` text files containing 2D image transformations
- `.tlt` text files containing tilt angles

The IMOD `xf` format is specified
[in their documentation](https://bio3d.colorado.edu/imod/doc/man/xfmodel.html#transforming_models).

!!! quote "IMOD `xf` spec"

    Each linear transformation in a transform file is specified by a line
    with six numbers:
    A11 A12 A21 A22 DX DY where the coordinate (X, Y) is transformed to
    (X', Y') by:
    X' = A11 * X + A12 * Y + DX
    Y' = A21 * X + A22 * Y + DY

This is done with the program [`ts_import_alignments`](./api/tilt_series.md#ts_import_alignments).

