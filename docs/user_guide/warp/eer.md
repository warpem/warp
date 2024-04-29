# EER handling

Rudimentary support for Thermo Fisher Scientific's Electron Event Registration (EER) format is implemented in Warp & M `>=1.0.9`. 
EER supports very fine exposure fractionation, which can be subjected to *in silico* exposure grouping later when the data are processed. 
EER's compression efficiency is similar to TIFF.

EER is available as input format in Warp. 
When selected, a new setting will appear in the UI: **EER frame group**. 
This determines how many consecutive frames will be summed up without taking into account any alignment when an EER frame series is read. 
Set a smaller value if you expect fast motion at the beginning of an exposure, e.g. because you're collecting on a tilted sample. 
If the overall number of frames isn't divisible by the group size, the last incomplete group will be dropped.

The **Exposure** setting in Warp refers to the cumulative exposure of 1 group of frames, i.e. you will likely need to adjust it if you change the group size. 
Please note that the group size currently can't be changed for M's processing later. 
M also doesn't implement any optimizations for finely fractionated data yet, so if you set the group size too small, you might run out of memory.

EER files are currently presumed to originate from a Falcon 4 detector, which has a physical size of `4096`*x*`4096` px. 
The acquisition software calculates non-integer centroid coordinates for every electron event, which the format preserves in up to 4x super-resolution. 
In Warp, the size of the **gain reference** determines whether the super-resolution information is used. If you provide a 8192x8192 px gain reference, the EER data will be rendered in 2x super-resolution; 16384x16384 px will result in 4x super-resolution; a 4096x4096 px gain reference turns off super-resolution rendering.
