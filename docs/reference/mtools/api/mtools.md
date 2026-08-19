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


## reconstruct

```
MTools 2.0.0+0f6d296f141ac2eed153f18907c98bdb89a89f88
Copyright (C) 2026 MTools

  --source                  Path(s) to one or more .source files,
                            space-separated. Frame-series and tilt-series
                            sources can be mixed in one run. Mutually exclusive
                            with --population.

  -p, --population          Instead of --source, use every data source of this
                            .population. Also makes the population's folder the
                            default output root.

  --particles               Required. Path to a particle STAR file, either
                            RELION's _data.star or M's _particles.star. Which
                            one it is is worked out from the columns.

  --angpix_coords           Override the pixel size of the particle coordinates
                            in a RELION file. Also applies to the shifts when
                            those are in pixels rather than Angstrom.

  --ignore_unmatched        Don't fail if there are particles that don't match
                            any data sources.

  --angpix                  Required. Pixel size of the reconstruction in
                            Angstrom.

  -d, --diameter            Required. Molecule diameter in Angstrom. Also sets
                            the box size to 2 x round(diameter / angpix), as
                            elsewhere in M.

  -s, --sym                 (Default: C1) Point symmetry, e.g. C1, D7, O.

  --helical_units           (Default: 1) Number of helical asymmetric units
                            (only relevant for helical symmetry).

  --helical_twist           Helical twist in degrees, positive = right-handed
                            (only relevant for helical symmetry).

  --helical_rise            Helical rise in Angstrom (only relevant for helical
                            symmetry).

  --helical_height          Height of the helical segment along the Z axis in
                            Angstrom (only relevant for helical symmetry).

  --max_resolution          Highest resolution in Angstrom the reconstruction
                            should be able to reach. Defaults to Nyquist (2 x
                            angpix), which is the most accurate and the most
                            expensive: it is the only input to the CTF
                            super-resolution box size, so loosening it makes
                            extraction cheaper and less memory-hungry.

  -m, --mask                Optional tight binary mask for the postprocess. If
                            omitted, a sphere of the molecule diameter is used,
                            which is unbiased but gives a slightly pessimistic
                            FSC. Sharpening itself never uses a mask.

  --no_postprocess          Write only the raw half-maps, skipping FSC
                            estimation, sharpening and local resolution.

  --denoise                 Train and apply a denoiser during the postprocess.
                            Off by default; it is the slowest part and is only
                            useful for visualisation.

  --batchsize               (Default: 16) Particles per extraction batch. The
                            main lever if you run out of GPU memory, since the
                            extraction box is enlarged by the CTF
                            super-resolution factor.

  --min_particles           (Default: 1) Only use series with at least N
                            particles in the field of view.

  --cpu_memory              Use CPU memory to store particle images (GPU by
                            default).

  -o, --output              Root directory for reconstructions. Each run creates
                            <root>/<name> inside it. Defaults to <population
                            folder>/reconstructions with --population, else
                            ./reconstructions.

  -n, --name                Name of this reconstruction, used for its output
                            files and, with a unique suffix appended, its
                            directory. Defaults to the particle file's name
                            without its extension.

  --keep_temp               Keep the temporary directory (queue, logs,
                            per-worker back-projection partials) on success. It
                            is always kept on failure.

  --device_list             Space-separated list of GPU IDs to use for
                            processing. Default: all GPUs in the system.

  --perdevice               (Default: 1) Number of worker processes per GPU.

  --task_dir                Directory for the filesystem work queue used by this
                            run. Defaults to a 'tasks' subdirectory inside the
                            reconstruction's temp folder. Set this to fast local
                            scratch when the output directory is on a slow
                            network filesystem.

  --external_provisioner    Don't spawn local worker processes. An external
                            system (e.g. Relay) provisions workers that claim
                            tasks from the queue directory.

  --cluster_script          Path to a batch-scheduler submission-script
                            template. Presence of this option selects cluster
                            mode.

  --cluster_config          Path to the cluster queue-definition JSON. Required
                            with --cluster_script.

  --pool_size               Cluster mode: number of worker jobs to submit to the
                            scheduler.

  --cluster_var             Cluster mode: a key=value pair substituted into the
                            submission template (repeatable).

  --help                    Display this help screen.

  --version                 Display version information.

```


## postprocess

```
MTools 2.0.0+0f6d296f141ac2eed153f18907c98bdb89a89f88
Copyright (C) 2026 MTools

  -s, --species    Required. Path to a .species file, or to a directory
                   containing one (e.g. a reconstruction directory).

  -m, --mask       Path to a tight binary mask. M will expand and smooth it
                   based on the resolution. Omit to keep whatever mask the
                   species already has.

  --sphere         Replace the mask with a sphere of the molecule diameter —
                   unbiased, but a more conservative FSC. This is what a
                   reconstruction made without --mask already uses.

  --lowpass        Skip FSC estimation and filter to this fixed resolution in
                   Angstrom instead.

  --denoise        Train and apply a denoiser. Off by default; it is by far the
                   slowest part.

  --device         (Default: -1) GPU ID to use. Default: let the runtime pick.

  --help           Display this help screen.

  --version        Display version information.

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


