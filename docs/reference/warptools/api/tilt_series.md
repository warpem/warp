# tilt_series

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


## ts_template_matchts_import_alignments

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Unknown command, showing all available commands:

------------------------------------General-------------------------------------

change_selection         Change the manual selection status (selected | deselect
                         ed | null) for every item

create_settings          Create data import settings

filter_quality           Filter frame/tilt series by various quality metrics, or
                          just print out histograms

move_data                Changes the location of raw data in XML metadata; use i
                         t when the raw data moves, or you switch between Window
                         s and Linux

threshold_picks          Apply a score threshold to particles picked through tem
                         plate-matching from tilt or frame series


----------------------------------Frame series----------------------------------

fs_motion_and_ctf        Estimate motion in frame series, produce aligned averag
                         es, estimate CTF – all in one go!

fs_motion                Estimate motion in frame series, produce aligned averag
                         es

fs_boxnet_infer          Run a trained BoxNet model on frameseries averages, pro
                         ducing particle positions and masks

fs_boxnet_train          (Re)train a BoxNet model on image/label pairs, producin
                         g a new model

fs_ctf                   Estimate CTF parameters in frame series

fs_export_micrographs    Create aligned averages or half-averages from frame ser
                         ies with previously estimated motion

fs_export_particles      Extract particles from tilt series


--------------------------------------Help--------------------------------------

helpgpt                  Get help from ChatGPT; requires an OpenAI API key store
                         d in ~/openai.key


----------------------------------Tilt series-----------------------------------

ts_aretomo               Create tilt series stacks and run AreTomo2 to obtain ti
                         lt series alignments

ts_ctf                   Estimate CTF parameters in frame series

ts_defocus_hand          Check and/or set defocus handedness for all tilt series

ts_etomo_fiducials       Create tilt series stacks and run Etomo fiducial tracki
                         ng to obtain tilt series alignments

ts_etomo_patches         Create tilt series stacks and run Etomo patch tracking 
                         to obtain tilt series alignments

ts_eval_model            Map 3D positions to sets of 2D image coordinates consid
                         ering a tilt series' deformation model

ts_export_particles      Export particles as 3D volumes or 2D image series.

ts_import_alignments     Import tilt series alignments from IMOD or AreTomo

ts_import                Create .tomostar files based on a combination of MDOC f
                         iles, aligned frame series, and optional tilt series al
                         ignments from IMOD or AreTomo

ts_reconstruct           Reconstruct tomograms for various tasks and, optionally
                         , half-tomograms for denoiser training

ts_stack                 Create tilt series stacks, i.e. put all of a series' ti
                         lt images in one .st file, to be used with IMOD, AreTom
                         o etc.

ts_template_match        Match previously reconstructed tomograms against a 3D t
                         emplate, producing a list of the highest-scoring matche
                         s


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


