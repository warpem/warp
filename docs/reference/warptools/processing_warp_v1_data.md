# Moving Tilt Series Processing to v2.X

Some people want to continue projects from Warp 1.X in Warp 2.0 to enable processing their 
data in HPC environments using *WarpTools*. 

Rasmus Jensen wrote this useful guide to doing just that, thanks Rasmus!

## Steps
- set up WarpTools settings files (c.f. [Quick Start: Tilt Series](../../user_guide/warptools/quick_start_warptools_tilt_series.md#create-warp-settings-files))
- symlink motion corrected averages into `<frame_series_processing_dir>/average`
- run frame series CTF estimation ([`WarpTools fs_ctf`](./api/frame_series.md#fs_ctf))
- copy masks from Warp1.0.X BoxNet (if relevant)
- import previous .xf files ([`WarpTools ts_import_alignments`](./api/tilt_series.md#ts_import_alignments))
- run tilt series CTF estimation ([`WarpTools ts_ctf`](./api/tilt_series.md#ts_ctf))
- reconstruct tomograms ([`WarpTools ts_reconstruct`](./api/tilt_series.md#ts_reconstruct))

## Notes
- it is important to ensure you have the same handedness and tomogram dimensions after tomogram reconstruction
- you can export particles using [`WarpTools ts_export_particles`](./api/tilt_series.md#ts_export_particles) after this and get nearly identical maps without re-refining angles



