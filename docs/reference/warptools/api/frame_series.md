## fs_motion_and_ctf

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command fs_motion_and_ctf:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--m_range_min             Default: 500. Minimum resolution in Angstrom to consid
                          er in fit

--m_range_max             Default: 10. Maximum resolution in Angstrom to conside
                          r in fit

--m_bfac                  Default: -500. Downweight higher spatial frequencies u
                          sing a B-factor, in Angstrom^2

--m_grid                  Resolution of the motion model grid in X, Y, and tempo
                          ral dimensions, separated by 'x': e.g. 5x5x40; empty =
                           auto

--c_window                Default: 512. Patch size for CTF estimation in binned 
                          pixels

--c_range_min             Default: 30. Minimum resolution in Angstrom to conside
                          r in fit

--c_range_max             Default: 4. Maximum resolution in Angstrom to consider
                           in fit

--c_defocus_min           Default: 0.5. Minimum defocus value in um to explore d
                          uring fitting

--c_defocus_max           Default: 5. Maximum defocus value in um to explore dur
                          ing fitting

--c_voltage               Default: 300. Acceleration voltage of the microscope i
                          n kV

--c_cs                    Default: 2.7. Spherical aberration of the microscope i
                          n mm

--c_amplitude             Default: 0.07. Amplitude contrast of the sample, usual
                          ly 0.07-0.10 for cryo

--c_fit_phase             Fit the phase shift of a phase plate

--c_use_sum               Use the movie average spectrum instead of the average 
                          of individual frames' spectra. Can help in the absence
                           of an energy filter, or when signal is low.

--c_grid                  Resolution of the defocus model grid in X, Y, and temp
                          oral dimensions, separated by 'x': e.g. 5x5x40; empty 
                          = auto; Z > 1 is purely experimental

--out_averages            Export aligned averages

--out_average_halves      Export aligned averages of odd and even frames separat
                          ely, e.g. for denoiser training

--out_skip_first          Default: 0. Skip first N frames when exporting average
                          s

--out_skip_last           Default: 0. Skip last N frames when exporting averages


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

--input_norawdata         Ignore the existence of raw data and look for XML meta
                          data in the processing directory instead.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## fs_motion

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command fs_motion:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--range_min               Default: 500. Minimum resolution in Angstrom to consid
                          er in fit

--range_max               Default: 10. Maximum resolution in Angstrom to conside
                          r in fit

--bfac                    Default: -500. Downweight higher spatial frequencies u
                          sing a B-factor, in Angstrom^2

--grid                    Resolution of the motion model grid in X, Y, and tempo
                          ral dimensions, separated by 'x': e.g. 5x5x40; empty =
                           auto

--averages                Export aligned averages

--average_halves          Export aligned averages of odd and even frames separat
                          ely, e.g. for denoiser training

--skip_first              Default: 0. Skip first N frames when exporting average
                          s

--skip_last               Default: 0. Skip last N frames when exporting averages


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

--input_norawdata         Ignore the existence of raw data and look for XML meta
                          data in the processing directory instead.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## fs_ctf

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command fs_ctf:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--window                  Default: 512. Patch size for CTF estimation in binned 
                          pixels

--range_min               Default: 30. Minimum resolution in Angstrom to conside
                          r in fit

--range_max               Default: 4. Maximum resolution in Angstrom to consider
                           in fit

--defocus_min             Default: 0.5. Minimum defocus value in um to explore d
                          uring fitting

--defocus_max             Default: 5. Maximum defocus value in um to explore dur
                          ing fitting

--voltage                 Default: 300. Acceleration voltage of the microscope i
                          n kV

--cs                      Default: 2.7. Spherical aberration of the microscope i
                          n mm

--amplitude               Default: 0.07. Amplitude contrast of the sample, usual
                          ly 0.07-0.10 for cryo

--fit_phase               Fit the phase shift of a phase plate

--use_sum                 Use the movie average spectrum instead of the average 
                          of individual frames' spectra. Can help in the absence
                           of an energy filter, or when signal is low.

--grid                    Resolution of the defocus model grid in X, Y, and temp
                          oral dimensions, separated by 'x': e.g. 5x5x40; empty 
                          = auto; Z > 1 is purely experimental


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

--input_norawdata         Ignore the existence of raw data and look for XML meta
                          data in the processing directory instead.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## fs_boxnet_infer

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command fs_boxnet_infer:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--model                   REQUIRED Path to the .pt file containing the model wei
                          ghts

--perprocess              Default: 4. Number of threads per process; the model i
                          s loaded only once per process and oversubscription ca
                          n save memory

--diameter                Default: 100. Approximate particle diameter in Angstro
                          m

--threshold               Default: 0.5. Picking score threshold, between 0.0 and
                           1.0

--distance                Default: 0. Minimum distance in Angstrom to maintain b
                          etween picked positions and masked pixels

--negative                Expect negative stain-like contrast (mass = bright)

--suffix_star             Override the suffix added to the particle STAR file; l
                          eave empty to use model name

--patchsize               Default: 512. Size of the BoxNet input window that the
                           model was trained with, a multiple of 256; the defaul
                          t for models shipped with Warp is 512


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

--input_norawdata         Ignore the existence of raw data and look for XML meta
                          data in the processing directory instead.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## fs_boxnet_train

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command fs_boxnet_train:

--------------------------------------------------------------------------------

--model_in            Path to the .pt file containing the old model weights; mod
                      el will be initialized from scratch if this is left empty

--model_out           REQUIRED Path to the .pt file where the new model weights 
                      will be saved

--examples            REQUIRED Path to a folder containing TIFF files with examp
                      les prepared with boxnet_examples_frameseries

--examples_general    Path to a folder containing TIFF files with examples used 
                      to train a more general model, which will be mixed 1:1 wit
                      h new examples to reduce overfitting

--no_mask             Don't consider mask labels in training; they will be conve
                      rted to background labels

--patchsize           Default: 512. Size of the BoxNet input window, a multiple 
                      of 256; remember to use the same window with boxnet_infer_
                      frameseries

--batchsize           Default: 8. Size of the minibatches used in training; larg
                      er batches require more GPU memory; must be divisible by n
                      umber of devices

--lr_start            Default: 5E-05. Learning rate at training start

--lr_end              Default: 1E-05. Learning rate at training end, with linear
                       interpolation in-between

--epochs              Default: 100. Number of training epochs

--checkpoints         Default: 0. Save checkpoints every N minutes; set to 0 to 
                      disable

--devices             Space-separated list of GPU IDs to be used for training


```


## fs_export_micrographs

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command fs_export_micrographs:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--averages                Export aligned averages

--average_halves          Export aligned averages of odd and even frames separat
                          ely, e.g. for denoiser training

--skip_first              Default: 0. Skip first N frames when exporting average
                          s

--skip_last               Default: 0. Skip last N frames when exporting averages


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

--input_norawdata         Ignore the existence of raw data and look for XML meta
                          data in the processing directory instead.


-----------------------Advanced remote work distribution------------------------

--workers                 List of remote workers to be used instead of locally s
                          pawned processes. Formatted as hostname:port, separate
                          d by spaces


```


## fs_export_particles

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command fs_export_particles:

------------------------------Data import settings------------------------------

--settings                 REQUIRED Path to Warp's .settings file, typically loc
                           ated in the processing folder. Default file name is '
                           previous.settings'.


----------------------STAR files with particle coordinates----------------------

-i, --input                REQUIRED Path to folder containing the STAR files; or
                            path to a single STAR file

--patterns                 Space-separated list of file name search patterns or 
                           STAR file names when --star is a folder


-------------------------------------Output-------------------------------------

-o, --output               REQUIRED Where to write the STAR file containing info
                           rmation about the exported particles

--suffix_out               Default: . Suffix to add at the end of each stack's n
                           ame; the full name will be [movie name][--suffix_out]
                           .mrcs

--angpix_out               Pixel size the extracted particles will be scaled to;
                            leave out to use binned pixel size from input settin
                           gs

--box                      REQUIRED Particle box size in pixels

--diameter                 REQUIRED Particle diameter in Angstrom

--relative_output_paths    Make paths in output STAR file relative to the locati
                           on of the STAR file. They will be relative to the wor
                           king directory otherwise.


-------------------Export type (REQUIRED, mutually exclusive)-------------------

--averages                 Export particle averages; mutually exclusive with oth
                           er export types

--halves                   Export particle half-averages e.g. for denoising; mut
                           ually exclusive with other export types

--only_star                Don't export, only write out STAR table; mutually exc
                           lusive with other export types


-------------------------------Coordinate scaling-------------------------------

--angpix_coords            REQUIRED Pixel size for the input coordinates

--angpix_shifts            Pixel size for refined shifts if not given in Angstro
                           m (when using rlnOriginX instead of rlnOriginXAngst)


---------------------------------Expert options---------------------------------

--dont_invert              Don't invert contrast, e.g. for negative stain data

--dont_normalize           Don't normalize background (RELION will complain!)

--dont_center              Don't re-center particles based on refined shifts

--flip_phases              Pre-flip phases in bigger box to avoid signal loss du
                           e to delocalization

--keep_ctf                 Keep CTF information from STAR inputs

--skip_first_frames        Default: 0. Skip first N frames

--skip_last_frames         Default: 0. Skip last N frames


-------------------------------Work distribution--------------------------------

--device_list              Space-separated list of GPU IDs to use for processing
                           . Default: all GPUs in the system

--perdevice                Default: 1. Number of processes per GPU


----------------------Advanced data import & flow options-----------------------

--input_data               Overrides the list of input files specified in the .s
                           ettings file. Accepts a space-separated list of files
                           , wildcard patterns, or .txt files with one file name
                            per line.

--input_data_recursive     Enables recursive search for files matching the wildc
                           ard pattern specified in --input_data. Only applicabl
                           e when processing and directories are separate. All f
                           ile names must be unique.

--input_processing         Specifies an alternative directory containing pre-pro
                           cessed results. Overrides the processing directory in
                            the .settings file.

--output_processing        Specifies an alternative directory to save processing
                            results. Overrides the processing directory in the .
                           settings file.

--input_norawdata          Ignore the existence of raw data and look for XML met
                           adata in the processing directory instead.


-----------------------Advanced remote work distribution------------------------

--workers                  List of remote workers to be used instead of locally 
                           spawned processes. Formatted as hostname:port, separa
                           ted by spaces


```


