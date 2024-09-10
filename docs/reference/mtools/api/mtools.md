## create_population

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -d, --directory    Required. Path to the directory where the new population
                     will be located. All future species will also go there, so
                     make sure there is enough space.

  -n, --name         Required. Name of the new population.

  --help             Display this help screen.

  --version          Display version information.

```


## create_source

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population             Required. Path to the .population file to which
                               to add the new data source.

  -s, --processing_settings    Required. Path to a .settings file used to
                               pre-process the frame or tilt series this source
                               should include; desktop Warp will usually
                               generate a previous.settings file

  -n, --name                   Required. Name of the new data source.

  --nframes                    Maximum number of tilts or frames to use in
                               refinements. Leave empty or set to 0 to use the
                               maximum number available.

  --files                      Optional STAR file with a list of files to
                               intersect with the full list of frame or tilt
                               series referenced by the settings.

  --help                       Display this help screen.

  --version                    Display version information.

```


## create_species

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population          Required. Path to the .population file to which to
                            add the new data source.

  -n, --name                Required. Name of the new species.

  -d, --diameter            Required. Molecule diameter in Angstrom.

  -s, --sym                 (Default: C1) Point symmetry, e.g. C1, D7, O.

  --helical_units           (Default: 1) Number of helical asymmetric units
                            (only relevant for helical symmetry).

  --helical_twist           Helical twist in degrees, positive = right-handed
                            (only relevant for helical symmetry).

  --helical_rise            Helical rise in Angstrom (only relevant for helical
                            symmetry).

  --helical_height          Height of the helical segment along the Z axis in
                            Angstrom (only relevant for helical symmetry).

  -t, --temporal_samples    (Default: 1) Number of temporal samples in each
                            particle pose's trajectory.

  --half1                   Required. Path to first half-map file.

  --half2                   Required. Path to second half-map file.

  -m, --mask                Required. Path to a tight binary mask file. M will
                            automatically expand and smooth it based on current
                            resolution

  --angpix                  Override pixel size value found in half-maps.

  --angpix_resample         Resample half-maps and masks to this pixel size.

  --lowpass                 Optional low-pass filter (in Angstrom), applied to
                            both half-maps.

  --particles_relion        Path to _data.star-like particle metadata from
                            RELION.

  --particles_m             Path to particle metadata from M.

  --angpix_coords           Override pixel size for RELION particle coordinates.

  --angpix_shifts           Override pixel size for RELION particle shifts.

  --ignore_unmatched        Don't fail if there are particles that don't match
                            any data sources.

  --help                    Display this help screen.

  --version                 Display version information.

```


## rotate_species

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  -s, --species       Required. Path to the .species file, or its GUID.

  --angle_rot         Required. First Euler angle (Rot in RELION) in degrees.

  --angle_tilt        Required. Second Euler angle (Tilt in RELION) in degrees.

  --angle_psi         Required. Third Euler angle (Psi in RELION) in degrees.

  --help              Display this help screen.

  --version           Display version information.

```


## shift_species

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  -s, --species       Required. Path to the .species file, or its GUID.

  -x                  Required. Shift along the X axis in Angstrom. New map
                      center will be at current center + this value.

  -y                  Required. Shift along the X axis in Angstrom. New map
                      center will be at current center + this value.

  -z                  Required. Shift along Z axis in Angstrom. New map center
                      will be at current center + this value.

  --help              Display this help screen.

  --version           Display version information.

```


## expand_symmetry

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  -s, --species       Required. Path to the .species file, or its GUID.

  --expand_from       Symmetry to use for the expansion if it is different from
                      the one specified in the species (e.g. expand only one
                      sub-symmetry of a higher symmetry).

  --expand_to         Remaining symmetry that will be set as the species'
                      symmetry, e.g. C1 (when using --expand_from to expand only
                      part of the symmetry).

  --helical_units     (Default: 1) Number of asymmetric subunits in the helical
                      symmetry to expand

  --helical_twist     Twist of the helical symmetry to expand, in degrees

  --helical_rise      Rise of the helical symmetry to expand, in Angstrom

  --help              Display this help screen.

  --version           Display version information.

```


## resample_trajectories

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  -s, --species       Required. Path to the .species file, or its GUID.

  --samples           Required. The new number of samples, usually between 1
                      (small particles) and 3 (very large particles).

  --help              Display this help screen.

  --version           Display version information.

```


## update_mask

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  -s, --species       Required. Path to the .species file, or its GUID.

  -m, --map           Required. Path to the MRC map to be used to create the new
                      mask.

  -t, --threshold     Required. Binarization threshold to convert the input map
                      to a mask.

  -d, --dilate        (Default: 0) Dilate the binary mask by this many voxels.

  -c, --center        Center the species around the new mask's center of mass.

  --help              Display this help screen.

  --version           Display version information.

```


## list_species

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  --help              Display this help screen.

  --version           Display version information.

```


## list_sources

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  --help              Display this help screen.

  --version           Display version information.

```


## add_source

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  -s, --source        Required. Path to the .source file.

  --help              Display this help screen.

  --version           Display version information.

```


## remove_species

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  -s, --species       Required. Path to the .species file, or its GUID.

  --help              Display this help screen.

  --version           Display version information.

```


## remove_source

```
MTools 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MTools

  -p, --population    Required. Path to the .population file.

  -s, --source        Required. Path to the .source file, or its GUID.

  --help              Display this help screen.

  --version           Display version information.

```


