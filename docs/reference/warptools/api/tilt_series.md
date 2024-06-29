## ts_import

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_import:

--------------------------------------------------------------------------------

--mdocs            REQUIRED Path to the folder containing MDOC files

--pattern          Default: *.mdoc. File name pattern to search for in the MDOC 
                   folder

--frameseries      REQUIRED Path to a folder containing frame series processing 
                   results and their aligned averages

--tilt_exposure    REQUIRED Per-tilt exposure in e-/A^2

--dont_invert      Don't invert tilt angles compared to IMOD's convention (inver
                   sion is usually needed to match IMOD's geometric handedness).
                    This will flip the geometric handedness

--override_axis    Override the tilt axis angle with this value

--auto_zero        Adjust tilt angles so that the tilt with the highest average 
                   intensity becomes the 0-tilt

--tilt_offset      Subtract this value from all tilt angle values to compensate 
                   pre-tilt

--max_tilt         Default: 90. Exclude all tilts above this (absolute) tilt ang
                   le

--min_intensity    Default: 0. Exclude tilts if their average intensity is below
                    MinIntensity * cos(angle) * 0-tilt intensity; set to 0 to no
                   t exclude anything

--max_mask         Default: 1. Exclude tilts if more than this fraction of their
                    pixels is masked; needs frame series with BoxNet masking res
                   ults

--min_ntilts       Default: 1. Only import tilt series that have at least this m
                   any tilts after all the other filters have been applied

-o, --output       REQUIRED Path to a folder where the created .tomostar files w
                   ill be saved


```


## ts_stack

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_stack:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--angpix                  Rescale tilt images to this pixel size; leave out to k
                          eep the original pixel size

--mask                    Apply mask to each image if available; masked areas wi
                          ll be filled with Gaussian noise


-------------------------------Work distribution--------------------------------

--device_list             Space-separated list of GPU IDs to use for processing.
                           Default: all GPUs in the system

--perdevice               Default: 1. Number of processes per GPU


----------------------Advanced data import & flow options-----------------------

--input_data              Overrides the list of input files specified in the .se
                          ttings file. Accepts a space-separated list of files, 
                          wildcard patterns, or .txt files with one file name pe
                          r line.

--input_data_recursive    Enables recursive search for files matching the wildca
                          rd pattern specified in --input_data. Only applicable 
                          when processing and directories are separate. All file
                           names must be unique.

--input_processing        Specifies an alternative directory containing pre-proc
                          essed results. Overrides the processing directory in t
                          he .settings file.

--output_processing       Specifies an alternative directory to save processing 
                          results. Overrides the processing directory in the .se
                          ttings file.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## ts_aretomo

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_aretomo:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--angpix                  Rescale tilt images to this pixel size; normally 10–15
                           for cryo data; leave out to keep the original pixel s
                          ize

--mask                    Apply mask to each image if available; masked areas wi
                          ll be filled with Gaussian noise

--alignz                  REQUIRED Value for AreTomo's AlignZ parameter in Angst
                          rom (will be auto-converted to binned pixels), i.e. th
                          e thickness of the reconstructed tomogram used for ali
                          gnments

--axis_iter               Default: 0. Number of tilt axis angle refinement itera
                          tions; each iteration will be started with median valu
                          e from previous iteration, final iteration will use fi
                          xed angle

--axis_batch              Default: 0. Use only this many tilt series for the til
                          t axis angle search; only relevant if --axis_iter isn'
                          t 0

--min_fov                 Default: 0. Disable tilts that contain less than this 
                          fraction of the tomogram's field of view due to excess
                          ive shifts

--axis                    Override tilt axis angle with this value

--patches                 Number of patches for local alignments in X, Y, separa
                          ted by 'x': e.g. 6x4. Increases processing time.

--delete_intermediate     Delete tilt series stacks generated for AreTomo

--exe                     Default: AreTomo2. Name of the AreTomo2 executable; mu
                          st be in $PATH


-------------------------------Work distribution--------------------------------

--device_list             Space-separated list of GPU IDs to use for processing.
                           Default: all GPUs in the system

--perdevice               Default: 1. Number of processes per GPU


----------------------Advanced data import & flow options-----------------------

--input_data              Overrides the list of input files specified in the .se
                          ttings file. Accepts a space-separated list of files, 
                          wildcard patterns, or .txt files with one file name pe
                          r line.

--input_data_recursive    Enables recursive search for files matching the wildca
                          rd pattern specified in --input_data. Only applicable 
                          when processing and directories are separate. All file
                           names must be unique.

--input_processing        Specifies an alternative directory containing pre-proc
                          essed results. Overrides the processing directory in t
                          he .settings file.

--output_processing       Specifies an alternative directory to save processing 
                          results. Overrides the processing directory in the .se
                          ttings file.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## ts_etomo_fiducials

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_etomo_fiducials:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--angpix                  Rescale tilt images to this pixel size; normally 10–15
                           for cryo data; leave out to keep the original pixel s
                          ize

--mask                    Apply mask to each image if available; masked areas wi
                          ll be filled with Gaussian noise

--min_fov                 Default: 0. Disable tilts that contain less than this 
                          fraction of the tomogram's field of view due to excess
                          ive shifts

--initial_axis            Override initial tilt axis angle with this value

--do_axis_search          Fit a new tilt axis angle for the whole dataset

--fiducial_size           size of gold fiducials in nanometers

--delete_intermediate     Delete tilt series stacks generated for ETomo


-------------------------------Work distribution--------------------------------

--device_list             Space-separated list of GPU IDs to use for processing.
                           Default: all GPUs in the system

--perdevice               Default: 1. Number of processes per GPU


----------------------Advanced data import & flow options-----------------------

--input_data              Overrides the list of input files specified in the .se
                          ttings file. Accepts a space-separated list of files, 
                          wildcard patterns, or .txt files with one file name pe
                          r line.

--input_data_recursive    Enables recursive search for files matching the wildca
                          rd pattern specified in --input_data. Only applicable 
                          when processing and directories are separate. All file
                           names must be unique.

--input_processing        Specifies an alternative directory containing pre-proc
                          essed results. Overrides the processing directory in t
                          he .settings file.

--output_processing       Specifies an alternative directory to save processing 
                          results. Overrides the processing directory in the .se
                          ttings file.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## ts_etomo_patches

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_etomo_patches:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--angpix                  Rescale tilt images to this pixel size; normally 10–15
                           for cryo data; leave out to keep the original pixel s
                          ize

--mask                    Apply mask to each image if available; masked areas wi
                          ll be filled with Gaussian noise

--min_fov                 Default: 0. Disable tilts that contain less than this 
                          fraction of the tomogram's field of view due to excess
                          ive shifts

--initial_axis            Override initial tilt axis angle with this value

--do_axis_search          Fit a new tilt axis angle for the whole dataset

--patch_size              Default: 500. patch size for patch tracking in Angstro
                          ms

--delete_intermediate     Delete tilt series stacks generated for Etomo


-------------------------------Work distribution--------------------------------

--device_list             Space-separated list of GPU IDs to use for processing.
                           Default: all GPUs in the system

--perdevice               Default: 1. Number of processes per GPU


----------------------Advanced data import & flow options-----------------------

--input_data              Overrides the list of input files specified in the .se
                          ttings file. Accepts a space-separated list of files, 
                          wildcard patterns, or .txt files with one file name pe
                          r line.

--input_data_recursive    Enables recursive search for files matching the wildca
                          rd pattern specified in --input_data. Only applicable 
                          when processing and directories are separate. All file
                           names must be unique.

--input_processing        Specifies an alternative directory containing pre-proc
                          essed results. Overrides the processing directory in t
                          he .settings file.

--output_processing       Specifies an alternative directory to save processing 
                          results. Overrides the processing directory in the .se
                          ttings file.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## ts_ctf

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_ctf:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--window                  Default: 512. Patch size for CTF estimation in binned 
                          pixels

--range_low               Default: 30. Lowest (worst) resolution in Angstrom to 
                          consider in fit

--range_high              Default: 4. Highest (best) resolution in Angstrom to c
                          onsider in fit

--defocus_min             Default: 0.5. Minimum defocus value in um to explore d
                          uring fitting (positive = underfocus)

--defocus_max             Default: 5. Maximum defocus value in um to explore dur
                          ing fitting (positive = underfocus)

--voltage                 Default: 300. Acceleration voltage of the microscope i
                          n kV

--cs                      Default: 2.7. Spherical aberration of the microscope i
                          n mm

--amplitude               Default: 0.07. Amplitude contrast of the sample, usual
                          ly 0.07-0.10 for cryo

--fit_phase               Fit the phase shift of a phase plate

--auto_hand               Run defocus handedness estimation based on this many t
                          ilt series (e.g. 10), then estimate CTF with the corre
                          ct handedness


-------------------------------Work distribution--------------------------------

--device_list             Space-separated list of GPU IDs to use for processing.
                           Default: all GPUs in the system

--perdevice               Default: 1. Number of processes per GPU


----------------------Advanced data import & flow options-----------------------

--input_data              Overrides the list of input files specified in the .se
                          ttings file. Accepts a space-separated list of files, 
                          wildcard patterns, or .txt files with one file name pe
                          r line.

--input_data_recursive    Enables recursive search for files matching the wildca
                          rd pattern specified in --input_data. Only applicable 
                          when processing and directories are separate. All file
                           names must be unique.

--input_processing        Specifies an alternative directory containing pre-proc
                          essed results. Overrides the processing directory in t
                          he .settings file.

--output_processing       Specifies an alternative directory to save processing 
                          results. Overrides the processing directory in the .se
                          ttings file.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## ts_reconstruct

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_reconstruct:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--angpix                  REQUIRED Pixel size of the reconstructed tomograms in 
                          Angstrom

--halfmap_frames          Also produce two half-tomograms, each reconstructed fr
                          om half of the frames (requires running align_frameser
                          ies with --average_halves previously)

--halfmap_tilts           Also produce two half-tomograms, each reconstructed fr
                          om half of the tilts (doesn't work quite as well as --
                          halfmap_frames)

--deconv                  Also produce a deconvolved version; all half-tomograms
                          , if requested, will also be deconvolved

--deconv_strength         Default: 1. Strength of the deconvolution filter, if r
                          equested

--deconv_falloff          Default: 1. Fall-off of the deconvolution filter, if r
                          equested

--deconv_highpass         Default: 300. High-pass value (in Angstrom) of the dec
                          onvolution filter, if requested

--keep_full_voxels        Mask out voxels that aren't contained in some of the t
                          ilt images (due to excessive sample shifts); don't use
                           if you intend to run template matching

--dont_invert             Don't invert the contrast; contrast inversion is neede
                          d for template matching on cryo data, i.e. when the de
                          nsity is dark in original images

--dont_normalize          Don't normalize the tilt images

--dont_mask               Don't apply a mask to each tilt image if available; ma
                          sked areas will be filled with Gaussian noise

--dont_overwrite          Don't overwrite existing tomograms in output directory

--subvolume_size          Default: 64. Reconstruction is performed locally using
                           sub-volumes of this size in pixel

--subvolume_padding       Default: 3. Padding factor for the reconstruction sub-
                          volumes (helps with aliasing effects at sub-volume bor
                          ders)


-------------------------------Work distribution--------------------------------

--device_list             Space-separated list of GPU IDs to use for processing.
                           Default: all GPUs in the system

--perdevice               Default: 1. Number of processes per GPU


----------------------Advanced data import & flow options-----------------------

--input_data              Overrides the list of input files specified in the .se
                          ttings file. Accepts a space-separated list of files, 
                          wildcard patterns, or .txt files with one file name pe
                          r line.

--input_data_recursive    Enables recursive search for files matching the wildca
                          rd pattern specified in --input_data. Only applicable 
                          when processing and directories are separate. All file
                           names must be unique.

--input_processing        Specifies an alternative directory containing pre-proc
                          essed results. Overrides the processing directory in t
                          he .settings file.

--output_processing       Specifies an alternative directory to save processing 
                          results. Overrides the processing directory in the .se
                          ttings file.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## ts_template_match

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_template_match:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--tomo_angpix             REQUIRED Pixel size of the reconstructed tomograms in 
                          Angstrom

--template_path           Path to the template file

--template_emdb           Instead of providing a local map, download the EMDB en
                          try with this ID and use its main map

--template_angpix         Pixel size of the template; leave empty to use value f
                          rom map header

--template_diameter       REQUIRED Template diameter in Angstrom

--template_flip           Mirror the template along the X axis to flip the hande
                          dness; '_flipx' will be added to the template's name

--symmetry                Default: C1. Symmetry of the template, e.g. C1, D7, O

--subdivisions            Default: 3. Number of subdivisions defining the angula
                          r search step: 2 = 15° step, 3 = 7.5°, 4 = 3.75° and s
                          o on

--batch_angles            Default: 32. How many orientations to evaluate at once
                          ; memory consumption scales linearly with this; higher
                           than 32 probably won't lead to speed-ups

--peak_distance           Minimum distance (in Angstrom) between peaks; leave em
                          pty to use template diameter

--npeaks                  Default: 2000. Maximum number of peak positions to sav
                          e

--dont_normalize          Don't set score distribution to median = 0, stddev = 1

--dont_whiten             Don't perform spectral whitening; this can help when t
                          he alignments are poor by up-weighting lower frequenci
                          es

--reuse_results           Reuse correlation volumes from a previous run if avail
                          able, only extract peak positions

--check_hand              Default: 0. Also try a flipped version of the template
                           on this many tomograms to see what geometric hand the
                          y have

--subvolume_size          Default: 192. Matching is performed locally using sub-
                          volumes of this size in pixel


-------------------------------Work distribution--------------------------------

--device_list             Space-separated list of GPU IDs to use for processing.
                           Default: all GPUs in the system

--perdevice               Default: 1. Number of processes per GPU


----------------------Advanced data import & flow options-----------------------

--input_data              Overrides the list of input files specified in the .se
                          ttings file. Accepts a space-separated list of files, 
                          wildcard patterns, or .txt files with one file name pe
                          r line.

--input_data_recursive    Enables recursive search for files matching the wildca
                          rd pattern specified in --input_data. Only applicable 
                          when processing and directories are separate. All file
                           names must be unique.

--input_processing        Specifies an alternative directory containing pre-proc
                          essed results. Overrides the processing directory in t
                          he .settings file.

--output_processing       Specifies an alternative directory to save processing 
                          results. Overrides the processing directory in the .se
                          ttings file.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## ts_import_alignments

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_import_alignments:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--alignments              REQUIRED Path to a folder containing one sub-folder pe
                          r tilt series with alignment results from IMOD or AreT
                          omo

--alignment_angpix        REQUIRED Pixel size (in Angstrom) of the images used t
                          o create the alignments (used to convert the alignment
                           shifts from pixels to Angstrom)

--min_fov                 Default: 0. Disable tilts that contain less than this 
                          fraction of the tomogram's field of view due to excess
                          ive shifts


----------------------Advanced data import & flow options-----------------------

--input_data              Overrides the list of input files specified in the .se
                          ttings file. Accepts a space-separated list of files, 
                          wildcard patterns, or .txt files with one file name pe
                          r line.

--input_data_recursive    Enables recursive search for files matching the wildca
                          rd pattern specified in --input_data. Only applicable 
                          when processing and directories are separate. All file
                           names must be unique.

--input_processing        Specifies an alternative directory containing pre-proc
                          essed results. Overrides the processing directory in t
                          he .settings file.

--output_processing       Specifies an alternative directory to save processing 
                          results. Overrides the processing directory in the .se
                          ttings file.


```


## ts_eval_model

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command ts_eval_model:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--input_star              Path to a STAR file containing custom positions specif
                          ied using tomoCenteredCoordinate(X/Y/Z)Angst labels. L
                          eave empty to use a regular grid of positions instead.

--grid_extent             Instead of custom positions in --input_star, calculate
                           an evenly spaced grid with this extent in Angstrom, s
                          pecified as 'XxYxZ', e.g. 6000x4000x1000.

--grid_dims               When calculating an evenly spaced grid, it will have t
                          his many points in each dimension, specified as 'XxYxZ
                          ', e.g. 30x20x5. The grid spacing will be grid_extent 
                          / (grid_dims - 1).

--output                  Output location for the per-tilt series STAR files


----------------------Advanced data import & flow options-----------------------

--input_data              Overrides the list of input files specified in the .se
                          ttings file. Accepts a space-separated list of files, 
                          wildcard patterns, or .txt files with one file name pe
                          r line.

--input_data_recursive    Enables recursive search for files matching the wildca
                          rd pattern specified in --input_data. Only applicable 
                          when processing and directories are separate. All file
                           names must be unique.

--input_processing        Specifies an alternative directory containing pre-proc
                          essed results. Overrides the processing directory in t
                          he .settings file.

--output_processing       Specifies an alternative directory to save processing 
                          results. Overrides the processing directory in the .se
                          ttings file.


```


