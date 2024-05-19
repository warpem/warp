# Frame Series Training

*BoxNet* can be trained on frame series data using *WarpTools* `fs_boxnet_train`.

## Training Data Structure

The `--examples` option to `WarpTools fs_boxnet_train` should be set to a folder
containing TIFF files with examples.

Each file should have three z-slices

- slice 0: the image data
- slice 1: ternary mask, 0 is bg, 1 is particle, 2 is mask
- slice 2: importance mask (not supported but need empty layer in file)

All files should be at the same pixel size.
Different files can contain images with different dimensions.