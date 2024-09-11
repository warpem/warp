# Denoising Tomograms in WarpTools

*WarpTools* comes with everything you need to denoise tomograms using the 
[Noise2Noise](https://arxiv.org/abs/1803.04189) denoising scheme.

## Outline

Tomogram denoising is performed in two steps
1. generate tomograms from half sets of data
2. train and apply a denoiser using *Noise2Map*

*Noise2Noise* denoising requires two independent noisy observations of a target signal.
In cryo-EM we can generate independent noisy copies of the same image by averaging the 
even and odd frames in a frame series.

We perform reconstructions from these even and odd images to 
produce independently noisy tomograms for denoising.

## Generating tomograms from half sets of data

### Generating even and odd frame series averages
We can generate even and odd movie averages by adding the `--out_average_halves` flag 
when running
[`WarpTools fs_motion_and_ctf`](./api/frame_series.md#fs_motion_and_ctf). If you didn't 
do this don't worry, you can run 
[`WarpTools fs_export_micrographs`](./api/frame_series.md#fs_export_micrographs) 
with the `--average_halves` flag.

This will produce `even` and `odd` subfolders filled with images in the `average` folder of your frame series
processing directory.

### Generating even and odd tomograms

Now that we have even and odd images, producing even and odd tomograms is as simple as 
running [`WarpTools ts_reconstruct`](./api/tilt_series.md#ts_reconstruct) with the 
`--halfmap_frames` flag.

!!! question "What if I don't have frame series?"

    Don't worry! You can generate tomograms from even and odd tilts instead. 
    Just add `--halfmap_tilts` to your `ts_reconstruct` command.

This will generate `even` and `odd` subfolders filled with images in the `reconstruction` 
folder of your tilt series processing directory.

## Training and applying the denoiser

Now we can use *Noise2Map* to train a denoiser and apply it to our tomograms. 
We typically need to train for >10,000 iterations to get good results.

An example *Noise2Map* command for tomogram denoising is provided below.

```sh
Noise2Map
--observation1 warp_tiltseries/reconstruction/even \
--observation2 warp_tiltseries/reconstruction/odd \
--observation_combined warp_tiltseries/reconstruction \
--dont_flatten_spectrum \
--dont_augment
```

