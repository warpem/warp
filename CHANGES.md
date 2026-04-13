# Changelog

## v2.0.0dev37

### New Commands

- **`ts_peak_align`**: Tilt series alignment using correlation peaks averaged over multiple particles. Requires a template volume and particle coordinates from a STAR file. Supports optional per-particle pose optimization.

- **`ts_reconstruct_average`**: Subtomogram averaging performed in-memory across multiple workers. Distributes back-projection, then merges half-maps with optional symmetry. Outputs `.mrc` files with automatic half-map splitting.

- **`ts_apply_denoiser`**: Applies a trained Noise2Map model to reconstructed tomograms. Outputs denoised volumes to a new `denoised/` subdirectory under the tilt series processing folder.

### New Features

- **Whole-tomogram template matching**: New `MatchLargeVolume` algorithm that pads the entire tomogram to an FFT-friendly size and correlates globally, instead of iterating over a grid of sub-volumes.

- **Top-hat transform for score post-processing**: New CUDA kernels for greyscale erosion, dilation, and top-hat transform (`d_GreyscaleErode3D`, `d_GreyscaleDilate3D`, `d_TophatTransform`). Configurable connectivity exposed via `--tophat_connectivity` in `ts_template_match`.

- **Fault tolerance in batch processing**: `IterateOverItems` now catches and counts failed items instead of aborting the entire batch. Items that fail throw exceptions that are tracked and reported at the end.

- **EER rendering thread limit**: New environment variable to cap the number of threads used for EER frame rendering, preventing resource contention on shared systems.

- **Async pipeline infrastructure in WarpLib**: Reusable building blocks for producer-consumer workflows: `BoundedQueue<T>`, `RotatingPool<T>`, `ProgressTracker`, and `StreamingPipeline`. Used internally by Noise2Map and the new commands.

### Improvements

- **Noise2Map refactoring**: The monolithic `Noise2Map.cs` has been decomposed into modular components (`TrainingCoordinator`, `ModelTrainer`, `BatchPreparationWorker`, `DenoisingPipeline`, `RotatingMapPool`, `DataPreparator`, etc.). Training batch preparation is now parallelized across multiple threads. CLI interface is unchanged.

- **`SizeRoundingFactors` coordinate correction**: Particle extraction coordinates now account for the rounding of downsampled image dimensions to multiples of 2, improving sub-pixel accuracy.

- **Memory management**: Fixed several memory leaks, improved GPU object disposal, and optimized array allocations throughout the template matching and reconstruction pipelines.

- **AreTomo2 compatibility**: Import no longer fails on AreTomo2's occasionally buggy `.tlt` output.

### Breaking Changes

- **`ts_template_match` uses a different algorithm**: The CLI now dispatches to `MatchLargeVolume` instead of `MatchFull`. This produces different correlation scores and peak positions compared to v2.0.0dev36. Memory requirements also differ: the entire tomogram must fit in GPU memory. The Warp GUI still uses the original `MatchFull` algorithm.

- **`ts_autolevel --patch_size` is now in Angstroms**: Previously interpreted as pixels, the `--patch_size` parameter is now in Angstroms and converted to pixels internally using the binned pixel size. Existing scripts passing a pixel value will produce different patch sizes unless the binned pixel size happens to be ~1 A/px.

- **Noise2Map training is no longer deterministic**: Batch preparation is now parallelized across 3 threads with independent random seeds. Identical inputs will produce slightly different training trajectories compared to v2.0.0dev36.

## v2.0.0dev36 (2025-09-12)

### New Commands
- **`ts_autolevel`**: Estimate sample inclination around the X and Y axes to level it out. Optimizes leveling angles and per-patch elevation by maximizing cross-correlation between neighboring low-tilt projections. Options include `--angpix` for rescaling and `--patch_size` (default 500 A) for patch dimensions.

### New Features
- **Sample leveling angles in geometry model**: Added `LevelAngleX` and `LevelAngleY` parameters to `TiltSeries`, which apply additional rotations in the tilt geometry to compensate for tilted samples. These are incorporated into all coordinate projection and reconstruction routines and persisted in the XML metadata.

### Bug Fixes
- **EER gain reference handling**: Fixed EER dimensions bug where the gain reference was not being properly scaled to match 4x super-resolution data. EER super-resolution is now fixed to 4x, and the gain reference is rescaled to match rather than relying on dimension-matching heuristics.
- **Reconstruction memory management**: Fixed a post-dev35 reconstruction bug by allocating output volumes (main, deconvolved, odd/even halves) upfront as `Image` objects and copying data into them, instead of reassigning host arrays. This avoids dangling references when the reconstruction data is replaced during deconvolution.

### Improvements
- **Reference-free local alignment preprocessing**: Re-enabled 2x padding with clamped edges and rectangular masking before bandpass filtering in `AlignLocallyWithoutReferences`, improving patch extraction quality.

### Internal
- **Removed legacy `GetPositionInAllTiltsOld`**: Cleaned up ~90 lines of unused old coordinate projection code from `TiltSeries.cs`.

## v2.0.0dev35 (2025-08-28)

### New Commands
- **`ts_aretomo3`**: Wrapper for AreTomo3 tilt series alignment, with support for iterative tilt axis refinement, local patch alignment, automatic or manual axis angle, and optional deletion of intermediate stacks.

### New Features
- **`--exclude_pattern` for MDOC import**: Skip MDOC files matching a pattern during `ts_import`, e.g. to exclude unsorted files.
- **`--dont_version` for data sources and species**: Disable automatic versioning when creating data sources (`create_source`) or during M refinement.
- **`--output` for `create_source`**: Override the default path where `.source` files are saved.
- **Row pre-filtering in STAR file reading**: New `rowFilter`/`rowFilterColumn` parameters allow filtering rows during parsing without loading the entire file first.
- **`Star.LoadSplitByValue`**: New utility to load a STAR file and split it into a dictionary of sub-tables keyed by a column value.
- **Sub-table name selection**: STAR file constructor now accepts a `tableName` parameter for multi-table files.
- **Manual GC in WarpWorkers**: Workers can now trigger explicit garbage collection between processing steps to reduce memory pressure.

### Improvements
- **Memory management overhaul**: Introduced `ArrayPool<T>` and `GpuArrayPool` for host and GPU memory, replacing direct allocations across `Image`, CTF fitting, movie export, TIFF reading, I/O helpers, and more. This significantly reduces GC pressure during large processing runs.
- **ZLinq adoption**: Switched from standard LINQ to ZLinq across the codebase for better allocation performance in hot paths.
- **Removed fp16 (`IsHalf`) support from `Image`**: All `Image` constructors and GPU operations now use fp32 exclusively, simplifying the code and removing unused half-precision paths.
- **Avoid second GPU buffer when no defect map**: `LoadStack` no longer allocates a redundant GPU buffer when no defect map is present.
- **Reuse raw layer buffers**: Frame loading reuses pooled raw layer arrays instead of reallocating per stack.
- **Lower parallelism for species hash computation**: Hash computation now caps parallelism at 20 threads to avoid excessive resource contention.
- **Particle series path standardization**: Particle series file names and directory paths are now generated through centralized helper methods (`ToParticleSeriesFilePath`, `ToParticleSeriesAveragePath`).
- **Save exported particle count in `processed_items.json`**: `ts_export_particles` now records the number of exported particles per tilt series.
- **Hidden file filtering**: STAR file directory searches now skip files whose names start with `.`.
- **Tilt stacking saves metadata**: `TomoStack` worker command now calls `SaveMeta()` after creating the stack.

### Bug Fixes
- **Euler angle detection in `ts_export_particles`**: Detection of input Euler angles now checks for column existence rather than checking if all values are non-zero, fixing incorrect behavior when particles legitimately have zero-valued angles.
- **`--relative_output_paths` respected in particle export**: Output STAR file paths are now correctly resolved relative to the STAR file location or working directory as specified.
- **ZLinq compatibility fixes**: Resolved multiple issues caused by ZLinq's different behavior with `string.Join` and LINQ materialisation by adding explicit `.ToArray()` calls where needed.
- **Relative path handling in `DataSource`**: Path construction now uses `Helper.PathCombine` instead of string concatenation, fixing issues on some platforms.
- **EER 4K rendering bounds check**: Added index validation during EER 4K rasterization to catch out-of-bounds writes.

### Breaking Changes
- **`Image` constructor signature change**: The `ishalf` parameter has been removed from all `Image` constructors. Any code passing `ishalf: true` will fail to compile.
- **JSON field names in `processed_items.json` renamed**: Abbreviated keys (`Stat`, `Def`, `Phs`, `Rsn`, `AsX`, `AsY`, `Mtn`, `Jnk`, `Ptc`, `Tlts`) are now spelled out (`ProcessingStatus`, `Defocus`, `Phase`, `Resolution`, `AstigX`, `AstigY`, `Motion`, `Junk`, `Particles`, `Tilts`). Downstream consumers of this JSON must be updated.

### Internal
- **`WarpWorker` logic moved to `WarpLib`**: The core worker dispatch logic now lives in `WarpLib/WarpWorker.cs`, with the executable (`WarpWorker/WarpWorker.cs`) renamed to `WarpWorkerProcess`.
- **Removed obsolete EER super-resolution experiment**: Cleaned up references to `LanczosEER.hpp` and related experimental code.
- **Skia version bump**.
- **IO buffer sizes reduced**: `IOHelper` file stream buffers reduced from 4 MB to 4 KB, relying on OS-level buffering instead.

## v2.0.0dev34 (2025-06-26)

### New Features
- **Tilt stack thumbnails**: New `--thumbnails` option for `ts_stack`, `ts_aretomo`, `ts_etomo_fiducials`, and `ts_etomo_patches` to generate per-tilt PNG thumbnails alongside stacks.
- **High-pass filter for micrograph export**: New `--highpass` option in `fs_export_micrographs` to apply a high-pass filter (in Angstroms) to exported averages.
- **Custom pixel size for micrograph export**: New `--bin_angpix` option in `fs_export_micrographs` to specify output pixel size directly.
- **Custom STAR suffix for template matching**: New `--override_suffix` option in `ts_template_match` to override the default STAR file name derived from the template name.
- **Strict mode for tilt series import**: New `--strict` flag for `ts_import`.
- **Relocate population data sources and species**: New static methods `Population.MoveSpecies` and `Population.MoveSources` allow updating species and data source paths in a population XML without loading everything into memory.
- **Per-tilt FOV fraction stored in metadata**: FOV fraction is now calculated for all tilt series (not only when `MinFOV` is set) and persisted in the XML metadata as `FOVFraction`.

### Improvements
- **AVX2-vectorized EER reading**: The EER electron rendering code (4K, 8K, 16K modes) is now vectorized using AVX2 intrinsics, processing 8 electrons at a time for significantly faster EER frame decoding.
- **Richer tilt series mini-JSON output**: Mini-JSON now includes per-tilt movie paths, tilt angle range, axis angle statistics, shift statistics, per-tilt defocus min/mean/max, astigmatism, phase shift range, CTF resolution, and sample inclination angle.
- **Average motion in mini-JSON**: Movie mini-JSON now includes mean frame movement.
- **Standardized directory and file naming**: All output directory names and file paths for Movies and TiltSeries are now exposed as static methods (e.g., `ToTomogramWithPixelSize`, `ToReconstructionTomogramPath`, `ToPowerSpectrumPath`), enabling path resolution without instantiating full objects.
- **Sorted input items**: Input files are now sorted alphabetically for deterministic processing order.
- **Hidden files excluded**: Files starting with `.` are now filtered out when enumerating input data.
- **Tilt axis angle warning on import**: `ts_import` now prints a warning when a tilt axis angle is read from an mdoc file, as Tomo5 mdoc values are known to be unreliable.
- **Better WarpWorker error reporting**: Worker errors now parse the JSON response to extract the `details` field and throw `ExternalException` with a cleaner message.
- **Star.ReadAllTables**: New method to discover and load all tables from a multi-table STAR file in one call.

### Bug Fixes
- **Particle series export with single tilt**: Fixed a crash when exporting particle series with only a single tilt requested (previously failed with index-out-of-range when computing per-tilt dose).
- **Failed items mini-JSON in ts_import**: Fixed `failed_items.json` to correctly list actually failed items instead of all items.
- **DirtErasure static buffer initialization**: Dirt erasure buffers are no longer initialized at class load time (which called `GPU.GetDeviceCount()` too early); they are now lazily allocated, avoiding GPU access in static contexts.
- **Tilt stack thumbnail path**: Fixed incorrect path used for tilt stack thumbnails.
- **Error message typo**: Corrected "wrpTilt" to "wrpAngleTilt" in the STAR file validation error message.
- **Alignment intermediate cleanup preserves thumbnails**: `--delete_intermediate` in AreTomo/ETomo commands now preserves the `thumbnails` subdirectory when cleaning up tilt stacks.
- **Binned pixel size for alternative exports**: Properly set the alternative binned pixel size when the user specifies a custom value.

### Internal
- **CI build updates**: Migrated to `setup-micromamba@v2`, explicit shell activation in CI build/upload steps, and pinned cmake < 4.
- **Skia version bump**.
- **PadClampedSoft GPU kernel**: New padding mode that clamps edge values and applies a soft falloff, used by the high-pass filter implementation.

## v2.0.0dev33 (2025-03-05)

### New Commands
- **`fs_tardis_segment_membranes`**: Added 2D membrane segmentation for frame series using TARDIS. Images are downsampled to 15 A/px, processed through `tardis_mem2d`, and results are stored in the membrane segmentation directory.

### Improvements
- **Unified item iteration across all commands**: Replaced ad-hoc `foreach`/`ForCPUGreedy` loops in many commands with a generic `IterateOverItems<T>` method that provides consistent progress reporting, error handling, JSON output of processed/failed items, and time-remaining estimates.
- **Batch processing support in `IterateOverItems`**: The unified iteration method now supports batch mode via a `getBatch` delegate, enabling commands like TARDIS segmentation to submit multiple items to a single worker process at once.
- **Failed items tracking**: Processing now writes a separate `failed_items.json` alongside `processed_items.json`, making it easier to identify and reprocess failures.
- **`ts_import_tiltseries` progress reporting**: Import now shows a standard progress line (`N/M, N failed`) instead of per-item log messages, and writes `processed_items.json`/`failed_items.json` incrementally during import.
- **`--strict` formatting option**: Added a `--strict` global flag that prevents VirtualConsole line-clearing, ensuring stable progress output for programmatic consumers.
- **Exception details in error output**: When item processing fails, the full exception is now printed to stderr for easier debugging.

### Bug Fixes
- **Metadata loss during processing**: Fixed a bug where item metadata saved by a worker (e.g., CTF results) could be overwritten because the parent process did not reload metadata before setting the processing status. `LoadMeta()` is now called before updating status after both success and failure.
- **TIFF reading with reuse buffer**: Fixed a bug where TIFF data was not copied into a caller-provided `reuseBuffer`, causing the buffer to remain empty.
- **`threshold_picks` directory linking crash**: Removed symbolic link creation logic from `threshold_picks` that could fail or overwrite existing directories.
- **`ts_template_match` directory linking crash**: Removed the same symbolic link creation logic and the now-unnecessary `--tomo_directory` option.

### Breaking Changes
- **Removed `--tomo_directory` from `ts_template_match`**: This option is no longer available. Reconstruction directories are expected to be managed externally (e.g., by Relay).
- **Removed directory linking from `move_data`**: Symbolic link creation between input and output processing directories has been removed from several commands; this responsibility is deferred to Relay.

### Internal
- **`Movie` and `TiltSeries` split into partial classes**: The monolithic `Movie.cs` and `TiltSeries.cs` files were split into multiple partial-class files organized by processing method (e.g., `Movie.ProcessCTF.cs`, `TiltSeries.MPARefinement.cs`), with no functional changes.
- **`IterateOverItems` is now generic**: The method signature changed from `IterateOverItems(workers, cli, Action<WorkerWrapper, Movie>)` to `IterateOverItems<T>(workers, cli, Action<WorkerWrapper, T>)` where `T : Movie`, eliminating manual casts in tilt series commands.

## v2.0.0dev32 (2025-01-22)

### Improvements
- **Automatic directory linking for split input/output workflows**: `ts_template_match` and `threshold_picks` now automatically create symbolic links from input processing directories (reconstruction, matching) to the output processing directory when both `--input_processing` and `--output_processing` are set.
- **Better error handling in item processing**: Failed items are now tracked with a `ProcessingStatus` and still written to `processed_items.json`, preventing loss of progress tracking. Items are marked as `LeaveOut` on failure instead of silently skipping the JSON update.
- **Increased WarpWorker request size limit**: Maximum request body size raised from the default to 100 MB to support larger payloads.
- **ARM64 compatibility**: TorchSharp and WarpLib debug builds changed from x64 to AnyCPU platform target, enabling use as libraries on ARM64 systems.

### Bug Fixes
- **CUDA build compatibility**: Added `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` preprocessor define to fix NativeAcceleration builds after Visual Studio minor version upgrades.

### Documentation
- **Tilt series quick start guide**: Added tips for EER file processing options and a signpost to the API reference pages.

## v2.0.0dev31 (2024-11-25)

### Bug Fixes
- **Motion tracks directory creation**: Create the output directory before writing the motion tracks JSON file, preventing errors when the directory does not yet exist.
- **Path namespace conflict**: Use `IOPath` alias instead of `Path` to avoid ambiguity with `System.IO.Path` in `Movie.cs`.

## v2.0.0dev30 (2024-11-21)

### New Features
- **Motion track JSON export**: Motion tracks are now exported as JSON files alongside averages during motion estimation, enabling external analysis of per-patch motion trajectories.

### Improvements
- **SIMD-vectorized math helpers**: `MeanAndStd`, `Min`, and `MinMax` for `float[]` now use `System.Numerics.Vector<float>` for significantly faster computation.
- **Bin2 downsampling method**: Added a fast 2x bilinear downsampling utility in `MathHelper`.
- **Quantized ColorScale**: `ColorScale` now pre-computes a lookup table of integer RGBA values, replacing per-pixel floating-point interpolation with a simple table lookup for faster rendering.
- **Graceful fallback when Einspline is unavailable**: `CubicGrid` now detects at startup whether the native Einspline library is present and silently falls back, avoiding crashes in environments without it.
- **Rename motion track oversampling parameter**: `GetMotionTrack`'s `samples` parameter renamed to `oversampleFactorAlongZ` for clarity; default sampling no longer oversamples when saving tracks.

### Bug Fixes
- **Skip motion estimation for single-frame stacks**: Stacks with only 1 frame now save metadata and return immediately instead of attempting motion correction.
- **Parse Relion-style Inf/NaN in STAR files**: `Star.GetRowValueFloat` now converts Relion's lowercase `inf`, `nan`, and `-nan` to .NET-compatible representations before parsing.
- **Throw on missing STAR table**: Requesting a non-existent table name in a STAR file now throws an informative exception instead of silently producing incorrect results.
- **Check directories exist before deleting in MCore post-refinement cleanup**: Prevents crashes when cleanup targets have already been removed.
- **Fix Reconstruct call signature in PCA3D**: Updated to match a new parameter added to `Reconstruct`.

### Internal
- **Remove Bridge project**: The entire Blazor-based Bridge web UI has been removed from the solution.

## v2.0.0dev29 (2024-10-09)

### Bug Fixes
- **Fix crash when hashing large maps in M**: Use a safe, truncating byte conversion (`ToBytesSafe`) when computing data hashes for species version strings, preventing out-of-memory errors with large half-maps and masks.
- **Fix file move race condition for processed_items.json**: Retry the `File.Move` for up to 10 seconds when writing `processed_items.json`, avoiding transient filesystem lock failures during parallel processing.

### Improvements
- **Better console output when piped or redirected**: Detect piped stdout and fall back to `\r`-based line clearing instead of cursor manipulation, preventing garbled output when WarpTools is piped into a file.
- **Write error messages to stderr**: Processing failure messages (e.g. "Failed to process...") are now written to `Console.Error` instead of `Console.Out`, so they don't pollute stdout when capturing output.

## v2.0.0dev28 (2024-09-24)

### New Features
- **Thumbnail export for frame series**: Added `--thumbnails` option to `fs_export_micrographs` and `--out_thumbnails` option to `fs_motion_and_ctf`, allowing export of scaled-down PNG thumbnails alongside averages.
- **Processed items JSON output**: WarpTools commands now write a `processed_items.json` file to the output directory during processing, updated asynchronously and thread-safely after each item completes.
- **Bridge web UI (preview)**: Added a new Blazor-based web application project ("Bridge") with form-based interfaces for several WarpTools commands (`fs_motion`, `fs_ctf`, `fs_motion_and_ctf`, `fs_export_micrographs`, `filter_quality`, `threshold_picks`, `move_data`).

### Improvements
- **Thumbnail contrast normalization**: Improved thumbnail generation to use median-based statistics computed from the central crop of the image, producing more robust contrast scaling.
- **Re-normalize rotated templates in subtomogram correlation**: Added Fourier-space normalization of the projected reference after rotation and CTF multiplication, correcting intensity biases during template matching.
- **Reduce GPU memory allocation in subtomogram correlation**: Buffer sizes for projected templates now scale with batch size only rather than `max(nvolumes, batchsize)`, lowering peak GPU memory usage.
- **Image file reading avoids extra copy**: `Image.FromFile` now reads directly into the pre-allocated host buffer instead of allocating a separate array and copying.

### Bug Fixes
- **Fix crash on narrow terminals**: `VirtualConsole` no longer throws when `Console.WindowWidth` is very small by clamping the blank-padding count to zero.
- **Fix use-after-dispose in thumbnail creation**: Thumbnail code was calling `AsPadded` on the already-disposed original image instead of the scaled copy; fixed to operate on the correct object.

### Internal
- **CUDA compiler compatibility**: Added `--allow-unsupported-compiler` flag to the NativeAcceleration CUDA build to support newer MSVC toolchain versions.

## v2.0.0dev27 (2024-09-18)

### New Features
- **Allow keeping positions with incomplete tilt coverage in template matching**: The `ts_template_match` command now has a `--max_missing_tilts` option (default: 2) that controls how many tilts a position can be missing from before being discarded. Set to -1 to disable culling entirely. This replaces the previous all-or-nothing `KeepOnlyFullVoxels` behavior.
- **Filter particles by tilt coverage in particle series export**: The `ts_export_particles` command now has a `--max_missing_tilts` option (default: 5) that excludes particles not visible in more than the specified number of tilts when exporting 2D particle series.

### Bug Fixes
- **Fix helical symmetrization**: Helical twist is now correctly converted to radians before symmetrization, and the result is multiplied by the number of helical units. Previously the twist was passed in degrees, producing incorrect half-maps.
- **Fix top series selection in `threshold_picks`**: The condition checked `NTopPicks` instead of `NTopSeries`, causing the top-N series filtering to never activate.
- **Fix help text in `ts_ctf`**: Corrected the description from "frame series" to "tilt series".
- **Fix MDOC timestamp parsing for PACE data**: Normalize whitespace in MDOC timestamp fields before parsing, accommodating the slightly different format produced by PACE. Timestamp parse failures now throw an informative error instead of silently continuing.

### Improvements
- **Make `--fiducial_size` a required option in `ts_etomo_fiducials`**: The parameter is now enforced by the CLI parser rather than validated manually, giving clearer error messages.

### Documentation
- Add tomogram denoising guide.
- Add note about spectrum whitening in template matching to the tilt series user guide.
- Add MTools/MCore API reference documentation.
- Add missing WarpTools API docs and fix docs generation script.

## v2.0.0dev26 (2024-09-03)

### New Features
- **Helical symmetry expansion**: `expand_symmetry` in M now supports helical symmetry via `--helical_units`, `--helical_twist`, and `--helical_rise` parameters, in addition to the existing point group expansion.
- **Helical symmetrization in real space**: Half-map reconstruction now applies helical symmetry averaging using a new GPU-accelerated real-space kernel (`HelicalSymmetrize`), replacing the previous rectangular masking and Fourier-space approach from the RELION backprojector.

### Improvements
- **Configurable bead count for IMOD fiducial tracking**: Added `--n_beads_target` option to `ts_etomo_fiducials` (default: 50), allowing users to set the target number of beads for `autofidseed`.
- **Full-amplitude noise for dirt erasure during reconstruction**: `EraseDirt` now uses 100% noise scale during tomogram reconstruction (previously 10%), better masking contaminated regions.
- **Particle count output in `threshold_picks`**: Reports the number of particles before and after thresholding.
- **fp16 overflow protection when saving MRC files**: Automatically switches to float32 if data values exceed the half-precision range, with a warning.
- **Pixel size mismatch warning in species creation**: M warns if the pixel size specified via `--angpix` does not match the pixel size in the half-maps.
- **IMOD wrapper logging**: External process commands (`batchruntomo`, `submfg`) are now logged to the console for easier debugging.
- **Template matching angle ID persistence**: Angle IDs from template matching are now saved in a separate TIFF volume, enabling correct angle recovery when reusing pre-calculated correlation volumes.
- **Template matching score normalization**: Scores are now normalized using the 68th percentile of absolute values in the central crop rather than standard deviation, providing more robust scaling.

### Bug Fixes
- **Fix double rad-to-deg conversion**: Particle angles during subtomogram series export were converted from radians to degrees twice, producing incorrect orientations.
- **Fix error message for missing `.tlt` file**: Corrected a misleading error that reported a missing `.xf` file when the `.tlt` file was not found.
- **Require `--fiducial_size` in etomo fiducials CLI**: Now throws an explicit error if the required parameter is omitted, instead of failing silently.

### Internal
- **Remove obsolete GTOM reconstruction code**: Deleted unused Fourier reconstruction, SIRT, and WBP source files from the GTOM library.

## v2.0.0dev25 (2024-08-14)

### Bug Fixes
- **Fix helical symmetry detection in species creation**: The check for whether any helical symmetry parameters were specified always evaluated to `true` because `.Any()` was called without a predicate on a non-empty array, instead of `.Any(v => v)` to test if any value is `true`.

## v2.0.0dev24 (2024-08-14)

### New Features
- **Helical symmetry in M**: `mtools create_species` now accepts `--helical_units`, `--helical_twist`, `--helical_rise`, and `--helical_height` parameters. Helical symmetry is applied during reconstruction and maps are masked along the Z axis accordingly.
- **`--input_norawdata` option**: New global option allows processing when raw data are unavailable; metadata XML files are read from the processing directory instead.
- **Tilt range filtering for template matching**: New `--tilt_range` option in `ts_template_match` limits the angular search to orientations within a given range of the XY plane, useful for matching filaments lying flat.
- **Gaussian low-pass filter for template matching**: New `--lowpass` and `--lowpass_sigma` options provide explicit Gaussian low-pass filtering of template and tomogram during matching.
- **Cube application**: Integrated the Cube GUI tool for interactive 3D coordinate visualization and picking into the solution.

### Improvements
- **Spectral whitening default changed in template matching**: Whitening is now off by default (previously on); the old `--dont_whiten` flag is replaced by `--whiten` to opt in. This better reflects typical use where alignments may not yet be well refined.
- **Noise2Map outputs saved as fp16**: Denoised maps are now written in half-precision (16-bit float) MRC format, halving file sizes.
- **PATH checks for external alignment tools**: `ts_aretomo`, `ts_etomo_fiducials`, and `ts_etomo_patches` now verify that the required executables (AreTomo, batchruntomo) are on PATH before starting, giving a clear error message upfront.
- **Template matching CUDA cleanup**: Removed dead code and unused correlation normalization paths in the sub-tomogram correlation kernel; reduced unnecessary GPU memory allocations.

### Bug Fixes
- **Fixed crash when `--tilt_range` not specified**: Template matching crashed accessing a null `TiltRange` value when the option was omitted; now defaults to -1 (no filtering).
- **Fixed hardcoded image dimensions in template matching**: `ImageDimensionsPhysical` was hardcoded to 4096 x 4096 at 1.18 A/px instead of being read from the actual tilt image headers.
- **Restored sub-volume size control**: The sub-volume size search loop in template matching now correctly respects the user-specified `--subvolume_size` upper bound instead of being capped at 256.
- **Fixed memory leak in spectrum calculation**: `AsSpectrumMultiplied1D` now properly disposes intermediate FT amplitude images.
- **Fixed help text for `--dont_mask`**: Clarified that when masking is enabled (the default), masked areas are filled with Gaussian noise.

### Internal
- **Centralized debug flag**: Replaced scattered `Environment.GetEnvironmentVariable("WARP_DEBUG")` checks with `Helper.IsDebug` throughout the codebase.
- **New Image methods**: Added `AsScaledCTF`, `BandpassGauss`, `Normalize`, and `NormalizeWithinMask` to the Image class.

## v2.0.0dev23 (2024-07-14)

### Bug Fixes
- **Multi-species refinement crash with empty particle sets**: Fixed a bug in tilt series multi-species refinement where a species having 0 particles in a tilt series would cause a crash. All species iteration loops now correctly skip species with empty particle arrays.

### Improvements
- **Clarified `--ctf_defocusexhaustive` help text**: The CLI help now states that `--ctf_defocusexhaustive` only works in combination with `--ctf_defocus`.

## v2.0.0dev22 (2024-07-13)

### Bug Fixes
- **Species creation on multi-GPU systems**: Fixed GPU device selection during species resolution calculation and denoiser training. Previously, when no specific GPU was requested, the device was not set correctly, leading to errors on multi-GPU systems. The method now consistently selects the GPU with the most free memory and sets it as the active device.

### Improvements
- **Denoiser training progress reporting**: Replaced direct console output in the 3D noise network training with proper progress callbacks, ensuring status messages are routed through the UI rather than printed to stdout.

## v2.0.0dev21 (2024-07-11)

### Improvements
- **CUDA 11.8 upgrade**: Bumped CUDA dependency from 11.7 to 11.8 across all build environments, conda recipe, and installation instructions. PyTorch build now uses `cuda11.8_cudnn8.7.0_0` instead of `cuda11.7_cudnn8.5.0_0`.
- **Updated conda update instructions**: The `conda update` command in the installation docs now includes all required channels, preventing potential dependency resolution failures.

### Bug Fixes
- **Fixed conda release workflow**: Reordered CI steps so the GitHub source release happens after the conda package build and upload, preventing the workflow from failing when the conda build step couldn't find the package.

## v2.0.0dev20 (2024-07-10)

### New Features
- **RELION 5 particle STAR file support in `ts_export_particles`**: Particle coordinates can now be imported from RELION 5 style multi-table STAR files. Optics group data is automatically merged with particle data during parsing.
- **`rlnTomoName` support for tilt series identification**: Tilt series can now be identified by either `rlnMicrographName` or `rlnTomoName` columns in the input STAR file, improving compatibility with different RELION workflows.
- **`rlnImagePixelSize` column support**: Pixel sizes can now be read from `rlnImagePixelSize` in addition to `rlnPixelSize`, matching RELION 5 conventions.

### Improvements
- **Refactored coordinate and shift parsing**: Shift parsing (pixel-based and angstrom-based) is now handled by a unified `ParseShifts` helper method, reducing code duplication across X/Y/Z axes.
- **Improved euler angle detection**: Euler angle presence is now determined by checking whether angle columns exist in the STAR file and whether values are non-zero, rather than relying on exception handling.
- **Relaxed required columns**: `rlnMicrographName` is no longer unconditionally required; the validator now accepts either `rlnMicrographName` or `rlnTomoName`.

## v2.0.0dev19 (2024-07-08)

### Bug Fixes
- **Fix tomogram dimension warning in `ts_import`**: The warning for small unbinned tomogram dimensions now triggers when *any* dimension exceeds 3500, rather than requiring *all* dimensions to exceed 3500. This prevents the warning from being incorrectly suppressed when only one or two dimensions are large. The warning message was also clarified.

## v2.0.0dev18 (2024-07-03)

### Bug Fixes
- **Fix pipe path construction on Linux**: Handle null/empty `TMPDIR` environment variable by falling back to `/tmp`, and use `Path.Combine` instead of string concatenation to avoid missing path separators when constructing named pipe paths.

### Documentation
- **Add WarpTools CLI API reference**: Add comprehensive API documentation for all WarpTools commands covering frame series, tilt series, and general utilities, along with a script to auto-generate the docs.
- **Add cross-reference to `ts_import_alignments`**: Link the custom tilt series alignments guide to the corresponding CLI command documentation.

## v2.0.0dev17 (2024-06-28)

### Bug Fixes
- **Named pipes on Linux with remote filesystems**: Worker processes now use an absolute path under `TMPDIR` for named pipes on Linux, fixing failures when the working directory is on a remote filesystem (e.g. NFS, GPFS).

## v2.0.0dev16 (2024-06-28)

### New Features
- **Pre-tilt compensation in `ts_import`**: Added `--auto_zero` flag to automatically adjust tilt angles so the tilt with the highest average intensity becomes the 0-tilt, and `--tilt_offset` to subtract a fixed value from all tilt angles. The two options are mutually exclusive.

### Improvements
- **Reinstated average intensity calculation during tilt series import**: The per-tilt average intensity calculation, previously commented out, has been restored. It is now used for `--auto_zero` determination and for the `--min_intensity`-based tilt exclusion logic.

### Bug Fixes
- **Fixed multi-GPU denoiser training bug**: Corrected the GPU device selection for denoiser training in multi-species refinement, which was previously hardcoded to GPU 0 instead of using the assigned device.
- **Fixed all-zeros .tlt file handling**: Parsing a `.tlt` file where every tilt angle is zero now throws a clear error instead of silently proceeding with incorrect data.
- **Fixed GPU memory leak during half-map reconstruction**: Added missing `FreeDevice()` calls after half-map bandpass/masking to release GPU memory before subsequent operations.
- **Improved error reporting for settings file parsing**: Options parsing now catches `FileNotFoundException` separately and exits with a clear message, rather than silently swallowing all exceptions.

## v2.0.0dev15 (2024-06-22)

### Bug Fixes
- **Fix threshold_picks with ambiguous suffix matches**: When multiple files match a pick suffix pattern, the command now attempts an exact match before failing, and provides a clearer error message if no unique match is found.
- **Fail loudly on invalid CTF frequency range**: CTF fitting now throws an explicit error if the max frequency exceeds the Nyquist frequency, instead of silently producing wrong results.

### Improvements
- **Warn on small tomogram dimensions**: `create_settings` now prints a warning if unbinned tomogram dimensions are all below 3500, since tomograms should typically encompass the whole field of view.
- **Better guidance on processing failures**: When an item fails during processing and is marked as unselected, the log now points the user to the log directory and mentions the `change_selection` command for reactivation.
- **Reduce file I/O retry patience**: WarpWorker retry attempts for missing gain references, defect maps, and movie files reduced from 50 to 10 retries, so failures surface faster instead of hanging.

## v2.0.0dev14 (2024-06-17)

### New Features
- **Prototype tilt series alignment to maximize L2 norm**: Added `AlignToMaxL2` method for tilt series, implementing a new alignment strategy that optimizes per-tilt image shifts by maximizing the L2 norm of the reconstruction. Uses BFGS optimization with progressive tilt-by-tilt annealing outward from the zero-tilt image.
- **Multi-volume back-projector**: The `Projector` class now supports reconstructing multiple volumes simultaneously via a new `nvolumes` parameter, propagated through the native GPU reconstruction pipeline.

### Improvements
- **More robust .tlt file search during tilt series import**: Expanded search to cover many more directory and filename conventions (including `_st.tlt` and `_aligned_Imod` variants), with optional `WARP_DEBUG` environment variable for logging searched paths. Import now throws a clear error if no .tlt file is found instead of silently continuing.
- **Less aggressive low-pass filtering in Noise2Mic**: Changed the high-pass cutoff from a fixed `1/32` Nyquist fraction to a pixel-size-aware `2 / (500 / pixelsize)` calculation, and removed unnecessary border cropping before bandpass filtering.
- **Noise2Mic denoised output saved as float16**: Denoised micrographs are now written as 16-bit MRC files to reduce disk usage.
- **BoxNet training improvements**: Reduced weight decay from 5e-4 to 1e-4. Less aggressive border cropping (2 px instead of 8 px). Bandpass cutoff made pixel-size-aware. Training checkpoints now write example predictions as MRC files and report average loss since the previous checkpoint. Also added a configurable patch size and an all-zeros sanity check during data loading.
- **BoxNet inference bandpass made pixel-size-aware**: The bandpass filter applied to micrographs before BoxNet picking now scales with pixel size instead of using a hardcoded cutoff.

### Bug Fixes
- **Invariant culture parsing throughout the codebase**: Fixed numerous `decimal.Parse`, `float.Parse`, and `int.Parse` calls that lacked `CultureInfo.InvariantCulture`, which caused failures on systems with non-US locale settings (e.g., comma as decimal separator). Affected: `CreateSettings`, `ValueSlider`, M2 species dialogs, and tilt angle import.
- **Fix argument handling bug in Noise2Map**: Added a check for empty arguments so the tool shows help text instead of crashing when invoked without arguments.
- **Projector weight validation**: `BackProject` now correctly checks that `projweights` is not complex, catching a previously undetected input error.

### Internal
- **Conda environment for Windows builds**: Added `warp_build_windows.yml` with dependencies (CUDA 11.7, PyTorch, .NET 8, FFTW, libtiff).

## v2.0.0dev13 (2024-06-13)

### Bug Fixes
- **Fix tilt angle parsing on non-English locales**: Added invariant culture specifier to float parsing when reading tilt angles from .rawtlt files, preventing failures on systems where the decimal separator is a comma instead of a period.

## v2.0.0dev12 (2024-06-11)

### New Features
- **FP16 TIFF support**: TIFF files stored in half-precision (16-bit float) format can now be read correctly.

### Bug Fixes
- **Locale-independent parsing of tilt angles and dose**: `wrpAngleTilt` and `wrpDose` values in STAR files are now parsed using invariant culture, fixing failures on systems with non-English locale settings (e.g. comma as decimal separator).
- **Blob precomputation skipped when unused**: Fourier reconstruction no longer allocates or frees blob lookup tables when `iterations` is 0, avoiding uninitialized memory access.

### Improvements
- **Disable unreliable intensity-based tilt filtering during `ts_import`**: The average-intensity-based tilt exclusion and tilt axis angle estimation steps have been commented out, as they were unreliable in some cases.
- **Least-squares scaling factor fit**: Added `MathHelper.FitScaleLeastSq` for fitting a multiplicative scaling factor between two arrays via parabolic interpolation.

### Documentation
- **Tilt series stack processing guide**: Added documentation explaining how to process tilt series stacks with WarpTools.

## v2.0.0dev11 (2024-06-05)

### Bug Fixes
- **`create_settings` no longer errors for missing data directories**: When creating settings for files (e.g. tomostar) whose data directory doesn't exist yet, the command now emits a warning instead of crashing with an unhandled `DirectoryNotFoundException`.
- **Fix `update_mask` help text**: The help string incorrectly described the command as shifting particles to reflect a map shift; it now correctly reads "Create a new mask for a species".

### Improvements
- **Tilt series quick start guide**: Added a link to pre-calculated results (Zenodo), fixed `wget` invocations to remove `--show-progress` and `--quiet` flags that require newer wget versions, and corrected formatting in admonition blocks and the `+Weights` code fence.
- **Updated M result FSC figure**: Replaced the FSC curve screenshot with one showing correct x-axis labels.

## v2.0.0dev10 (2024-05-29)

### Bug Fixes
- **Fix exception handling in M processing loop**: Converted `Main` and `DoProcessing` to proper async methods, replacing fire-and-forget task pattern with `await`. The REST host is now started with `StartAsync` instead of unawaited `RunAsync`, and null-checked before stopping, preventing crashes from unhandled exceptions.

### Improvements
- **Remove debug file outputs from M**: Commented out numerous `WriteMRC` calls that wrote intermediate debug files (half-maps, average amplitudes, weights, CTF weights) during species refinement and tilt series processing, reducing disk clutter during production runs.

### Documentation
- **Add installation update instructions**: Added a section to the WarpTools installation guide explaining how to update an existing conda installation with `conda update`.

## v2.0.0dev9 (2024-05-29)

### New Features
- **Tilt axis angle validation in `ts_import`**: Automatically checks whether the tilt axis angle from the MDOC file is consistent with spatially resolved defocus estimates from individual tilt movies, warning the user if the correlation is poor.
- **`--min_ntilts` option for `ts_import`**: Allows excluding tilt series that have fewer than a specified number of tilts after all filters have been applied.
- **`--axis_batch` option for `ts_aretomo`**: Allows restricting tilt axis angle search iterations to a subset of tilt series, then running the final iteration on all series. This can speed up axis refinement on large datasets.

### Improvements
- **Improved tilt intensity filtering in `ts_import`**: Intensity-based tilt culling now uses median (instead of mean) for per-tilt intensity, considers the 10 lowest-angle tilts (up from 3) for the reference maximum, applies cos-weighted thresholds, and removes tilts contiguously from the zero-tilt outward rather than removing arbitrary tilts from the middle of the range.
- **Auto-create processing directory on metadata save**: `SaveMeta` in both `Movie` and `TiltSeries` now creates the processing directory if it does not exist, preventing errors when writing metadata to a new location.
- **Reduced verbosity in custom tilt series alignment import**: The "Importing 2D transforms from..." message during `.xf` file import is now only printed when `WARP_DEBUG` is set.
- **Additional search path for `.xf` files**: Added `{RootName}_aligned_Imod` as a search directory when importing IMOD alignment transforms, improving compatibility with some IMOD output conventions.

### Bug Fixes
- **Fixed `--override_axis` in `ts_import`**: The axis angle override is now applied before parsing MDOC lines, so specifying `--override_axis` actually takes effect instead of being overwritten by the MDOC value.

### Documentation
- Added WarpTools installation instructions.
- Added reference docs for working with subsets of data, custom tilt series alignment workflows, advanced data flow (input/output redirection), and BoxNet frame series training.
- Updated quick start tutorial with patch size notes, corrected defocus handedness walkthrough to account for `--dont_invert` in `ts_import`, and fixed `ntilts` to `nframes` terminology.

### Internal
- **Build script `-j` option**: `build-native-unix.sh` now accepts a `-j` flag to configure the number of parallel make jobs (default 8).

## v2.0.0dev8 (2024-05-17)

### New Features
- **`--batch_angles` for tilt series template matching**: New parameter to control how many orientations are evaluated at once, allowing users to trade off memory consumption for speed. Defaults to 32; memory scales linearly with this value.

### Improvements
- **Default `--patch_size` for `ts_etomo_patches`**: The `--patch_size` parameter now defaults to 500 Angstroms instead of requiring an explicit value.
- **Windows build sources libraries from conda environment**: Native library build (NativeAcceleration, LibTorchSharp) now resolves libtiff, zlib, FFTW, and PyTorch paths from conda environment variables (`CONDA_INCLUDE`, `CONDA_LIB`, `CONDA_BIN`, `PYTORCH_DIR`) instead of standalone installs.

### Bug Fixes
- **Fix CPU FFT/IFFT for 3D volumes**: Corrected buffer offset calculations in `FFT_CPU` and `IFFT_CPU` that ignored the slice index, causing all slices to read from and write to the same memory offset. Multi-slice transforms now produce correct results.
- **Fix crash when reusing correlation volumes in template matching**: When `--reuse_corr_volumes` was active, the reconstruction was not loaded before peak extraction, causing a crash. The reconstruction is now loaded in both the fresh-computation and reuse paths.
- **Fix missing reconstruction treated as silent skip**: Template matching previously returned silently when no reconstruction was found at the desired resolution. It now throws a `FileNotFoundException` immediately, making the error visible.
- **Fix IMOD/Etomo runners on Windows**: `batchruntomo` and `submfg` are now invoked with the `.cmd` extension on Windows, matching how IMOD ships its executables on that platform.

## v2.0.0dev7 (2024-05-17)

### New Features
- **AreTomo2 local alignments**: Added `--patches` option to `ts_aretomo` for patch-based local alignments (e.g. `--patches 6x4`), passed to AreTomo2 as `-Patch X Y`.
- **Pixel size from file in `create_settings`**: The `--angpix` option now accepts a path to an image or MDOC file (including wildcards) in addition to a numeric value, reading `PixelSpacing` from MDOC or pixel size from image headers automatically.

### Bug Fixes
- **Template matching memory**: Reduced concurrent template matching processes to 1 per GPU (from 2) to avoid excessive memory usage.

## v2.0.0dev6 (2024-05-15)

### Bug Fixes
- **Fix binning in fs_motion_and_ctf**: When using movie sum for CTF estimation, gain correction is now correctly skipped for already-corrected average images, preventing dimension mismatches and double-correction.
- **Fix exception swallowing in MCore**: Processing exceptions are no longer silently lost; `DoProcessing` is now properly awaited so errors propagate to the caller.
- **Check if population exists**: MCore now checks for the population file before attempting to load it, providing a clear error message instead of an unhandled exception.

### Improvements
- **Increase worker handshake timeout**: Worker connection timeout increased from 20s/10s to 100s/100s, improving reliability when workers are slow to start (e.g. on heavily loaded systems).
- **Improved gain dimension mismatch error message**: The error now reports both the gain reference and image dimensions for easier debugging.

## v2.0.0dev5 (2024-05-10)

### Bug Fixes
- **Fix conda package versioning**: Strip leading `v` from git tags when setting the conda package version, avoiding malformed version strings.
- **Fix update instructions in README**: Change `conda install warp` to `conda update warp` for upgrading an existing installation.

## v2.0.0dev4 (2024-05-10)

### Improvements
- **Fixed CI/CD pipeline**: Restored proper tag-based deployment triggers in the GitHub Actions workflow, replacing the debug/testing configuration from dev3 with correct branch and tag filters.
- **Dynamic conda package versioning**: Conda recipe now derives the package version and build number from git tags (`GIT_DESCRIBE_TAG`, `GIT_DESCRIBE_NUMBER`) instead of a hardcoded `2.0.0dev0`, enabling automated version-correct builds from CI.

## v2.0.0dev3 (2024-05-10)

### Internal
- **CI: free up disk space before checkout**: Reordered the GitHub Actions release workflow to maximize build space before checking out the repository, preventing potential out-of-disk-space failures during the build.

## v2.0.0dev2 (2024-05-10)

### Internal
- **CI pipeline debugging**: Temporarily disabled tag and branch filters on the CI workflow and added debug output to the conda build step, allowing the deployment pipeline to run without a tagged commit for testing purposes.

## v2.0.0dev1 (2024-05-10)

### New Commands
- **`ts_eval_model`**: New tool to evaluate a tilt series' projection and deformation model at a set of 3D positions, mapping them to per-tilt 2D image coordinates, angles, and defocus values. Accepts positions from a STAR file or generates a regular grid.

### New Features
- **32-bit MRC output**: Set the `WARP_FORCE_MRC_FLOAT32` environment variable to write MRC files in full 32-bit float precision instead of the default 16-bit half.
- **`--eer_groupexposure` option in `create_settings`**: Fractionate EER frames by specifying a target per-group exposure (e-/A^2) instead of a fixed number of groups; warns when frames would be discarded due to indivisible group sizes.
- **Multi-GPU training for Noise2Mic**: The `--gpuid_network` option now accepts a comma-separated list of GPU IDs for data-parallel denoiser training.

### Improvements
- **AreTomo2 compatibility**: `ts_aretomo` now defaults to the `AreTomo2` executable. Alignment import (`ts_import_alignments`) searches additional `_st.xf` filename variants and more directory structures produced by AreTomo2.
- **Phase shift propagation**: Per-tilt phase shift values are now correctly included in all CTF parameter construction paths during particle export, fixing incorrect CTF modeling for phase-plate data.
- **Template matching orientation reporting**: `ts_template_match` now prints the number of orientations used for matching before processing begins.
- **Improved training time estimates**: Noise2Map and Noise2Mic use per-iteration timing instead of cumulative averages, giving more accurate remaining-time estimates.
- **Upgraded to PyTorch 2.0.1**: Build environment updated from PyTorch 1.11 to 2.0.1 with CUDA 11.7, replacing the legacy `cudatoolkit` packages.

### Bug Fixes
- **Euler angle extraction fix**: Corrected `EulerXYZExtrinsicFromMatrix` which had transposed matrix element access (row/column swap), producing wrong Euler angles. The redundant and incorrectly named `EulerXYZExtrinsicFromMatrixRELION` method was removed.
- **Console deadlock in AreTomo worker**: Removed synchronous `ReadToEnd()` calls on AreTomo stdout/stderr after `WaitForExit()`, which could cause deadlocks when output buffers filled.
- **LibTorchSharp autograd state**: Fixed `THSAutograd_setInference` to pass the required fourth parameter to `c10::AutogradState`, matching the PyTorch 2 API.

### Internal
- **Documentation site**: Added an mkdocs-based documentation site with quick-start guides for Warp, WarpTools, and M, plus reference pages for the interface, tilt series processing, environment variables, and Noise2Map.
- **Conda build recipe**: Added conda-recipe and GitHub Actions workflow for building a conda package.