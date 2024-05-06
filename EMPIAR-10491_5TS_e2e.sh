# process 5 tilt-series from EMPIAR-10491 end to end in WarpTools
ROOT_DIR=${PWD}

FRAME_SERIES_DATA_DIR=${ROOT_DIR}/frames
MDOC_DIR=${ROOT_DIR}/mdoc

# re-download the data if not already present
wget --show-progress -N -q -nd -P ./ ftp://ftp.ebi.ac.uk/empiar/world_availability/10491/data/gain_ref.mrc;

for i in 1 11 17 23 32;
do
    echo "======================================================"
    echo "================= Downloading TS_${i} ================"
    wget --show-progress --timestamping --quiet --no-directories --directory-prefix ./mdoc ftp://ftp.ebi.ac.uk/empiar/world_availability/10491/data/tiltseries/mdoc/TS_${i}.mrc.mdoc;
    wget --show-progress --timestamping --quiet --no-directories --directory-prefix ./frames ftp://ftp.ebi.ac.uk/empiar/world_availability/10491/data/tiltseries/data/*-${i}_*.tif;
done

# load necessary software
ml gRED; ml spaces/cryoem; ml IMOD; ml relion/devel; ml aretomo;

# Create settings for frame series
WarpTools create_settings \
--folder_data ${FRAME_SERIES_DATA_DIR} \
--output warp_frameseries.settings \
--folder_processing warp_frameseries \
--extension "*.tif" \
--angpix 0.7894 \
--gain_path gain_ref.mrc \
--gain_flip_y \
--exposure 2.64

# Perform motion estimation, ctf parameter estimation 
# then produce aligned averages and averaged half set images for each frame series
WarpTools fs_motion_and_ctf \
--settings warp_frameseries.settings \
--m_grid 1x1x3 \
--c_grid 2x2x1 \
--c_range_max 7 \
--c_defocus_max 8 \
--c_use_sum \
--out_averages \
--out_average_halves \
--perdevice 4 

# Check results
WarpTools filter_quality --settings warp_frameseries.settings --histograms


# prepare tilt-series metadata for Warp
# this creates one tomostar file per tilt series in the tomostar directory
WarpTools ts_import \
--mdocs mdoc \
--frameseries warp_frameseries \
--tilt_exposure 2.64 \
--min_intensity 0.3 \
--output tomostar

# Create settings for tilt-series processing
# this creates a settings file for running programs on tilt-series data
WarpTools create_settings \
--output warp_tiltseries.settings \
--folder_processing warp_tiltseries \
--folder_data tomostar \
--extension "*.tomostar" \
--angpix 0.7894 \
--gain_path gain_ref.mrc \
--gain_flip_y \
--exposure 2.64 \
--tomo_dimensions 4400x6000x1000


# Estimate projection model parameters with patch tracking in eTomo
WarpTools ts_etomo_patches \
--settings warp_tiltseries.settings \
--angpix 10 \
--patch_size 500 \
--initial_axis -85.6 \
--perdevice 5

# # or aretomo (performs badly on this dataset)
# WarpTool ts_aretomo \
# --settings warp_tiltseries.settings \
# --angpix 10 \
# --alignz 800 \
# --axis_iter 5 \
# --min_fov 0 

# Check defocus handedness
WarpTools ts_defocus_hand \
--settings warp_tiltseries.settings \
--check

# > The average correlation is negative, which means that the defocus handedness should be set to 'flip'
# apply defocus handedness correction based on check
WarpTools ts_defocus_hand \
--settings warp_tiltseries.settings \
--set_flip

# Estimate per tilt defocus with constraints from tilt-series geometry
WarpTools ts_ctf \
--settings warp_tiltseries.settings \
--range_high 7 \
--defocus_max 8 \
--perdevice 4 

# Reconstruct tomograms
WarpTools ts_reconstruct \
--settings warp_tiltseries.settings \
--angpix 10 \
--perdevice 4 

## if a TS was misaligned you could realign in etomo in warp_tiltseries/tiltstack/TS_1 then:
# WarpTools ts_import_alignments \
# --settings warp_tiltseries.settings \
# --alignments warp_tiltseries/tiltstack/TS_1 \
# --alignment_angpix 10 

# then reconstruct with new alignments
# WarpTools ts_reconstruct \
# --settings warp_tiltseries.settings \
# --angpix 10 \
# --perdevice 4 

# Match template
WarpTools ts_template_match \
--settings warp_tiltseries.settings \
--tomo_angpix 10 \
--template_emdb 15854 \
--template_diameter 130 \
--symmetry O \
--perdevice 2

# threshold picks - scores are normalised so 6 means 6 * SD away from mean
WarpTools threshold_picks \
--settings warp_tiltseries.settings \
--in_suffix 15854 \
--out_suffix clean \
--minimum 6

# Write out 2D 'particle series' images for 3D refinement
WarpTools ts_export_particles \
--settings warp_tiltseries.settings \
--input_directory warp_tiltseries/matching \
--input_pattern "*15854_clean.star" \
--normalized_coords \
--output_star relion/matching.star \
--output_angpix 4 \
--box 64 \
--diameter 130 \
--relative_output_paths \
--2d 

# Run initial model generation in RELION
cd relion
mkdir -p InitialModel/job001

`which relion_refine` \
--o InitialModel/job001/run \
--iter 50 \
--grad \
--denovo_3dref  \
--ios matching_optimisation_set.star \
--ctf \
--K 1 \
--sym O \
--flatten_solvent  \
--zero_mask  \
--dont_combine_weights_via_disc \
--pool 30 \
--pad 1  \
--particle_diameter 130 \
--oversampling 1  \
--healpix_order 1  \
--offset_range 6  \
--offset_step 2 \
--auto_sampling  \
--tau2_fudge 4 \
--j 8 \
--gpu ""  \
--pipeline_control InitialModel/job001/

# Do an unmasked refinement in RELION
mkdir -p Refine3D/job002

mpirun --oversubscribe  -n 3 `which relion_refine_mpi` \
--o Refine3D/job002/run \
--auto_refine \
--split_random_halves \
--ios matching_optimisation_set.star \
--ref InitialModel/job001/run_it050_class001.mrc \
--trust_ref_size \
--ini_high 40 \
--dont_combine_weights_via_disc \
--pool 10 \
--pad 2  \
--ctf \
--particle_diameter 130 \
--flatten_solvent \
--zero_mask \
--oversampling 1 \
--healpix_order 2 \
--auto_local_healpix_order 4\
 --offset_range 5 \
 --offset_step 2 \
 --sym O \
 --low_resol_join_halves 40 \
 --norm \
 --scale  \
 --j 2 \
 --gpu ""  \
 --pipeline_control Refine3D/job002/


# 8A (Nyquist)

# back to root dir
cd ${ROOT_DIR}

# Create population
MTools create_population \
--directory m \
--name 10491

# Create data source
MTools create_source \
--name 10491 \
--population m/10491.population \
--processing_settings warp_tiltseries.settings

# Create species from RELION's results and resample maps to a smaller pixel size
relion_mask_create \
--i relion/Refine3D/job002/run_class001.mrc \
--o m/mask_4apx.mrc \
--ini_threshold 0.04

MTools create_species \
--population m/10491.population \
--name apoferritin \
--diameter 130 \
--sym O \
--temporal_samples 1 \
--half1 relion/Refine3D/job002/run_half1_class001_unfil.mrc \
--half2 relion/Refine3D/job002/run_half2_class001_unfil.mrc \
--mask m/mask_4apx.mrc \
--particles_relion relion/Refine3D/job002/run_data.star \
--angpix_resample 0.7894 \
--lowpass 10 

# Run an iteration of M without any refinements to check that everything imported correctly
MCore \
--population m/10491.population \
--perdevice_refine 4 \
--iter 0

# 6.4 A

# Run an iteration of M with image warp, particle pose and CTF refinement
# (running exhaustive defocus search because this is the first refinement)
MCore \
--population m/10491.population \
--refine_imagewarp 6x4 \
--refine_particles \
--ctf_defocus \
--ctf_defocusexhaustive \
--perdevice_refine 4 

# 3.6 A

# Run another iteration now that the reference has better resolution
MCore \
--population m/10491.population \
--refine_imagewarp 6x4 \
--refine_particles \
--ctf_defocus \
--perdevice_refine 4 

# 3.1 A

# Now also refine stage angles
MCore \
--population m/10491.population \
--refine_imagewarp 6x4 \
--refine_particles \
--refine_stageangles \
--perdevice_refine 4

# 3.0 A

# Now throw (almost everything) at it
MCore \
--population m/10491.population \
--refine_imagewarp 6x4 \
--refine_particles \
--refine_mag \
--ctf_cs \
--ctf_defocus \
--ctf_zernike3 \
--perdevice_refine 4 

# 3.0 A

# Fit weights per-tilt-series
EstimateWeights \
--population m/10491.population \
--source 10491 \
--resolve_items

MCore \
--population m/10491.population \
--perdevice_refine 4

# 3.0 A

# Fit per-tilt weights, averaged over all tilt sries
EstimateWeights \
--population m/10491.population \
--source 10491 \
--resolve_frames

MCore \
--population m/10491.population \
--perdevice_refine 4 \
--refine_particles 

# 3.0 A

# Resample particle pose trajectories to 2 temporal samples
# NOTE: Please adjust species folder name to account for GUID bit
MTools resample_trajectories \
--population m/10491.population \
--species m/species/apoferritin_797f75c2/apoferritin.species \
--samples 2

MCore \
--population m/10491.population \
--perdevice_refine 4 \
--refine_imagewarp 6x4 \
--refine_particles \
--refine_stageangles \
--refine_mag \
--ctf_cs \
--ctf_defocus \
--ctf_zernike3

# 2.9 A
