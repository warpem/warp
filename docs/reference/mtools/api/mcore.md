## MCore

```
MCore 2.0.0+db859c58158e0ac5179769d57c317a6c3b73b03d
Copyright (C) 2024 MCore

  --port                        (Default: 14300) Port to use for REST API calls,
                                set to -1 to disable

  --devicelist                  Space-separated list of GPU IDs to use for
                                processing. Default: all GPUs in the system

  --perdevice_preprocess        Number of processes per GPU used for map
                                pre-processing; leave blank = default to
                                --perdevice_refine value

  --perdevice_refine            (Default: 1) Number of processes per GPU used
                                for refinement; set to >1 to improve utilization
                                if your GPUs have enough memory

  --perdevice_postprocess       Number of processes per GPU used for map
                                pre-processing; leave blank = default to
                                --perdevice_refine value

  --workers_preprocess          List of remote workers to be used instead of
                                locally spawned processes for map
                                pre-processing. Formatted as hostname:port,
                                separated by spaces

  --workers_refine              List of remote workers to be used instead of
                                locally spawned processes for refinement.
                                Formatted as hostname:port, separated by spaces

  --workers_postprocess         List of remote workers to be used instead of
                                locally spawned processes for map
                                post-processing. Formatted as hostname:port,
                                separated by spaces

  --population                  Required. Path to the .population file
                                containing descriptions of data sources and
                                species

  --iter                        (Default: 3) Number of refinement sub-iterations

  --first_iteration_fraction    (Default: 1) Use this fraction of available
                                resolution for alignment in first sub-iteration,
                                increase linearly to 1.0 towards last
                                sub-iterations

  --min_particles               (Default: 1) Only use series with at least N
                                particles in the field of view

  --cpu_memory                  Use CPU memory to store particle images during
                                refinement (GPU by default)

  --weight_threshold            (Default: 0.05) Refine each tilt/frame up to the
                                resolution at which the exposure weighting
                                function (B-factor) reaches this value

  --refine_imagewarp            Refine image warp with a grid of XxY dimensions.
                                Examples: leave blank = don't refine, '1x1',
                                '6x4'

  --refine_particles            Refine particle poses

  --refine_mag                  Refine anisotropic magnification

  --refine_doming               Refine doming (frame series only)

  --refine_stageangles          Refine stage angles (tilt series only)

  --refine_volumewarp           Refine volume warp with a grid of XxYxZxT
                                dimensions (tilt series only). Examples: leave
                                blank = don't refine, '1x1x1x20', '4x6x1x41'

  --refine_tiltmovies           Refine tilt movie alignments (tilt series only)

  --ctf_batch                   (Default: 32) Batch size for CTF refinements.
                                Lower = less memory, higher = faster

  --ctf_minresolution           (Default: 8) Use only species with at least this
                                resolution (in Angstrom) for CTF refinement

  --ctf_defocus                 Refine defocus using a local search

  --ctf_defocusexhaustive       Refine defocus using a more exhaustive grid
                                search in the first sub-iteration; only works in
                                combination with ctf_defocus

  --ctf_phase                   Refine phase shift (phase plate data only)

  --ctf_cs                      Refine spherical aberration, which is also a
                                proxy for pixel size

  --ctf_zernike3                Refine Zernike polynomials of 3rd order (beam
                                tilt, trefoil – fast)

  --ctf_zernike5                Refine Zernike polynomials of 5th order (fast)

  --ctf_zernike2                Refine Zernike polynomials of 2nd order (slow)

  --ctf_zernike4                Refine Zernike polynomials of 4th order (slow)

  --help                        Display this help screen.

  --version                     Display version information.

```


