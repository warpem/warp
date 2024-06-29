## create_settings

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command create_settings:

--------------------------------------------------------------------------------

-o, --output           REQUIRED Path to the new settings file

--folder_processing    Processing folder location

--folder_data          REQUIRED Raw data folder location

--recursive            Recursively search for files in sub-folders (only when pr
                       ocessing and raw data locations are different)

--extension            Import file search term: Use e.g. *.mrc to process all MR
                       C files, or something more specific like FoilHole1_*.mrc

--angpix               REQUIRED Unbinned pixel size in Angstrom. Alternatively s
                       pecify the path to an image or MDOC file to read the valu
                       e from. If a wildcard pattern is specified, the first fil
                       e will be used

--bin                  2^x pre-binning factor, applied in Fourier space when loa
                       ding raw data. 0 = no binning, 1 = 2x2 binning, 2 = 4x4 b
                       inning, supports non-integer values

--bin_angpix           Choose the binning exponent automatically to match this t
                       arget pixel size in Angstrom

--gain_path            Path to gain file, relative to import folder

--defects_path         Path to defects file, relative to import folder

--gain_flip_x          Flip X axis of the gain image

--gain_flip_y          Flip Y axis of the gain image

--gain_transpose       Transpose gain image (i.e. swap X and Y axes)

--exposure             Default: 1. Overall exposure per Angstrom^2; use negative
                        value to specify exposure/frame instead

--eer_ngroups          Default: 40. Number of groups to combine raw EER frames i
                       nto, i.e. number of 'virtual' frames in resulting stack; 
                       use negative value to specify the number of frames per vi
                       rtual frame instead

--eer_groupexposure    As an alternative to --eer_ngroups, fractionate the frame
                       s so that a group will have this exposure in e-/A^2; this
                        overrides --eer_ngroups

--tomo_dimensions      X, Y, and Z dimensions of the full tomogram in unbinned p
                       ixels, separated by 'x', e.g. 4096x4096x1000


```


## move_data

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command move_data:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--to                      REQUIRED New directory containing raw data; if raw dat
                          a are located in sub-folders and require recursive sea
                          rch, specify the top directory here

--new_settings            REQUIRED Where to save an updated .settings file conta
                          ining the new raw data location


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


## filter_quality

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command filter_quality:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


------------------------Output mode (mutually exclusive)------------------------

--histograms              Print a histogram for each quality metric and exit

-o, --output              Path to a .txt file that will contain a list of series
                           that pass the filter criteria


----------------------------------CTF metrics-----------------------------------

--defocus                 Defocus in µm: 1 value = min; 2 values = min & max

--astigmatism             Astigmatism deviation from the dataset's mean, express
                          ed in standard deviations: 1 value = min; 2 values = m
                          in & max

--phase                   Phase shift as a fraction of π: 1 value = min; 2 value
                          s = min & max

--resolution              Resolution estimate based on CTF fit: 1 value = min; 2
                           values = min & max


---------------------------------Motion metrics---------------------------------

--motion                  Average motion in first 1/3 of a frame series in Å: 1 
                          value = min; 2 values = min & max


-----------------------------Image content metrics------------------------------

--crap                    Percentage of masked area in an image (frame series on
                          ly): 1 value = min; 2 values = min & max

--particles               Number of particles: 1 value = min; 2 values = min & m
                          ax; requires --particles_star to be set

--particles_star          Path to STAR file(s) with particle information; may co
                          ntain a wildcard that matches multiple files


------------------------------Tilt series metrics-------------------------------

--ntilts                  Number of tilts in a tilt series: 1 value = min; 2 val
                          ues = min & max


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


## change_selection

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command change_selection:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--select                  Change status to selected

--deselect                Change status to deselected

--null                    Change status to null, which is the default status

--invert                  Invert status if it isn't null


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


## threshold_picks

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command threshold_picks:

------------------------------Data import settings------------------------------

--settings                REQUIRED Path to Warp's .settings file, typically loca
                          ted in the processing folder. Default file name is 'pr
                          evious.settings'.


--------------------------------------------------------------------------------

--in_suffix               REQUIRED Suffix for the names of the input STAR files 
                          (file names will be assumed to match {item name}_{--in
                          _suffix}.star pattern)

--out_suffix              REQUIRED Suffix for the names of the output STAR files
                           (file names will be {item name}_{--in_suffix}_{--outs
                          uffix}.star)

--out_combined            Path to a single STAR file into which all results will
                           be combined; internal paths will be made relative to 
                          this location

--minimum                 Remove all particles below this threshold

--maximum                 Remove all particles above this threshold

--top_series              Keep this many top-scoring series

--top_picks               Keep this many top-scoring particles for each series


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


## helpgpt

```
WarpTools - a collection of tools for EM data pre-processing
Version 2.0.0

Showing all available options for command helpgpt:

--------------------------------------------------------------------------------

--key    OpenAI key to be saved to ~/openai.key


```


