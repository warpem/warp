# Tilt-series template matching

`ts_template_match` builds its own matched-filter volume from the original tilt images, proposes poses in 3D, and refines them against the tilt images. An existing reconstructed tomogram is not required. Alignment and CTF metadata and the processed movie averages must remain accessible. This is the WarpTools workflow.

```bash
WarpTools ts_template_match \
  --settings processing.settings \
  --tomo_angpix 4.5 \
  --template_path template.mrc --template_angpix 1.5 \
  --template_diameter 130 --symmetry C1 \
  --subdivisions 3 --peak_distance 60 --npeaks 8000
```

The example describes a small target; choose diameter, symmetry, minimum separation, and sampling for the actual specimen. A missing proposal cannot be recovered by refinement. Score threshold selection remains with the user.

## Algorithm

1. Estimate each tilt's independent white-noise level from unselected, unmasked patches at the original sampling, using the median high-frequency periodogram divided by `ln(2)`. Estimation precedes binning so the annulus is not moved into the protein signal band. This is an image-derived approximation pending propagated movie noise spectra.
2. Reconstruct `V = sum BP[H* data / N]`, where `H` contains CTF and the existing dose/tilt envelope. Filter images at defocus intervals of at most 200 Å and interpolate between them during backprojection. The spatial mapping uses Warp's geometry, sampled every 16 coarse voxels and interpolated; it includes image shifts, specimen deformation, and defocus handedness. Original tilt images and metadata are not overwritten.
3. Estimate the anisotropic 3D power spectrum of `V` by Gaussian tapering its autocorrelation with a 130 Å standard deviation. Whiten the data by `sqrt(P)`. Filter the rotated template by the center-of-volume slice transfer `sum |H|²/N` and the same whitening spectrum. This center transfer is a stationary approximation; local defocus is retained in the data reconstruction and particle refinement. Normalize templates and apply local volume-standard-deviation normalization to the correlations.
4. Maintain the top eight orientation scores and IDs per voxel while streaming orientations. Detect immediate-neighbor maxima and greedily apply spherical spatial suppression. Select up to `--npeaks` spatial peaks per tomogram, then pool each peak's full 3×3×3 neighborhood and retain up to `--refine_starts` pose hypotheses per peak. Neighboring voxels and alternative orientations do not consume the spatial-peak limit; only sparse lists leave the GPU. Defaults: 8,000 spatial peaks, up to 32 initial pose hypotheses per peak. A memory budget caps the concurrent orientation batch.
5. Refine poses using GPU-resident FP32 BFGS with analytic derivatives of Fourier interpolation, translation, and defocus. Particle rotations are exact rotations; local deformation and viewing geometry are frozen at each proposal. Physically pad the template and particle patches for its support, CTF delocalization, and allowed motion. Start at the coarse-search resolution limit and double spatial frequency at each transition (halve the resolution in Å), capped at the requested fine limit: for example, 20 → 10 → 6 Å. Identical coarse and fine limits give one band and zero resolution transitions. There is no stage-count parameter. Merge only nearly identical hypotheses between bands.
6. Measure an anisotropic per-tilt patch PSD and use the hybrid metric
   `W = 1 / [N + c * max(P - N, 0)]`.
   `c` counts geometrically overlapping components among that candidate's usable tilts. The overlap length is `max(400 Å, 3 × target diameter)`. The quadratic metric receives this weight once; independent detector noise is not multiplied by multiplicity. The particle core and its allowed motion must remain in view. Padding outside image boundaries is filled with zero after subtracting the observed patch mean; patches touching contamination masks are excluded. At least three usable tilts are required.
7. Keep the existing imaging parameters fixed throughout discovery. Read dose B-factors and tilt weights from tilt-series metadata through the existing CTF path. When those grids are absent, use Warp's standard dose/tilt defaults. Candidate matches do not change the observation model by default.
8. Select each anchor's best final pose by signed projection score, then apply final spherical suppression using that same score. Fit one free overall nonnegative particle amplitude for reporting; do not learn an amplitude interval from the detected population.

Two independent experimental switches retain the previous behavior for comparisons. `--refine_calibrate` uses up to 300 spatially distinct strong candidates before the final band to fit a shared dose-damage slope and relative tilt amplitudes. It requires eight usable candidates and repeats the band after fitting for a single-band request. `--refine_bounded_score` learns amplitude bounds from the 2nd and 98th percentiles of up to 300 distinct strong candidates, then selects final poses and ranks peaks by bounded-amplitude gain. With fewer than eight candidates it falls back to nonnegative free-amplitude gain; paired decoy scoring instead requires eight. Use both switches to request the previous combination. For an imaging-calibration comparison, change only `--refine_calibrate` and keep projection ranking in both arms.

Saved `GridDoseWeights` and `GridDoseBfacs` are consumed through the normal Warp CTF path; template matching does not overwrite them. That path retains its existing B-factor limit, `B <= -3 * dose`, and uses `B = -4 * dose` and `scale = cos(tilt)` when the respective grids are absent. Thus “fixed” means fixed under these existing CTF conventions. The experimental candidate calibration is an in-memory override, not a source of validated metadata for future runs.

M's existing weight estimator derives these grids from correlation curves. Those values are not established least-squares estimates of the transfer amplitude between a projection and the data. Reusing this metadata preserves existing behavior; it does not validate it as a physical forward model. A future calibration from identified particles needs an explicit residual objective and separate provenance before its parameters can be treated as transferable fitted amplitudes.

Boundary padding follows the research implementation; truncation of weak delocalized template tails at detector edges is an approximation, not an explicit masked-pixel likelihood. The weighting is a diagonal approximation to correlated structural background, not a full covariance likelihood. The multiplicity length remains an approximation whose transfer to heterogeneous experimental particles needs evaluation. Projection ranking removes the candidate-derived amplitude interval; it does not guarantee comparable null distributions across tilt series. The highest-scoring starts are alternative explanations, not independent evidence.

## Scores and outputs

For weighted inner product `C` and model power `P`:

```text
projection score = C / sqrt(P)
unconstrained nonnegative amplitude = max(C, 0) / P
bounded amplitude = clamp(C / P, lower, upper)
default final score = projection score
experimental bounded score = bounded_amplitude * C - bounded_amplitude² * P / 2
```

`rlnAutopickFigureOfMerit` contains the final score. `wrpTemplateMatchProjectionZ` retains the projection score and `wrpTemplateMatchAmplitude` the unconstrained nonnegative amplitude. These are ranking statistics, not calibrated Gaussian Z scores or false-discovery probabilities. Experimental bounded scores can be negative. In the default target-only workflow, nonpositive projection matches are omitted.

The final STAR always includes `wrpTemplateMatchAmplitude`, including when amplitude-based ranking is disabled. It is the fitted multiplier of the transferred template at the selected final pose, `max(C, 0) / P`, before any amplitude bounds are applied. Users can plot projection score against amplitude and apply their own amplitude filters after matching. Amplitude depends on the reference-map scale, image scale, and imaging parameters; it is not a normalized image-power measurement or automatically comparable across differently scaled jobs. In the default workflow it does not affect ranking or spatial suppression.

The matching directory also contains:

- `_starting_poses.tsv`: proposal centers, neighboring starts, angle IDs, and coarse scores.
- `_refinement.tsv`: per-band poses, scores, convergence, merging, and optimizer diagnostics.
- `_tilt_scores.tsv`: final per-tilt cross terms and model powers.
- `_matching_policy.json`: component scoring mode (before optional decoy subtraction), requested candidate calibration, applied shared corrections, stored dose/tilt grid values, and resolution bands. This is authoritative for the current run if optional diagnostics from a previous run with the same name remain.
- `_calibration.tsv`: slope objective values and selected tilt-scale corrections, when candidate calibration succeeds.
- `_amplitude_bounds.tsv`: final interval and calibration-set size, when bounded scoring learns an interval.

Optional amplitude/B diagnostics remain available through `--refine_fit_bfactor` and `--refine_export_tilt_spectra`. These diagnostics do not alter ranking or shared imaging parameters. With `--decoy_templates`, each decoy searches its own orientations at target-selected anchors using shared imaging parameters and noise weights. The final ranking becomes target score minus mean decoy score, using projection scores by default or bounded scores with `--refine_bounded_score`. Missing decoy fits are excluded. This contrast is not a calibrated false-discovery rate; decoys are absent from the default workflow.

## Controls and resource use

| Option | Default | Meaning |
| --- | --- | --- |
| `--tomo_angpix` | required | Coarse reconstruction and search sampling in Å. |
| `--subdivisions` | 3 | HEALPix angular sampling, approximately 7.5°. |
| `--peak_distance` | half the diameter | Spherical minimum particle separation in Å. |
| `--npeaks` | 8000 | Maximum spatial peaks per tomogram, before refinement; excludes neighboring pose hypotheses. |
| `--match_topk` | 8 | Orientations retained per voxel. |
| `--refine_starts` | 32 | Maximum initial pose hypotheses per peak, pooled from its 3×3×3 neighborhood. |
| `--refine_iterations` | 90 | Maximum accepted BFGS steps per hypothesis per band. |
| `--refine_calibrate` | off | Experimental candidate-derived shared dose/tilt fitting. |
| `--refine_bounded_score` | off | Experimental candidate-derived amplitude bounds and bounded-gain ranking. |
| `--refine_noise_patches` | 256 | Unselected patches per tilt for the directional background PSD. |
| `--optimize_poses_angpix` | `--tomo_angpix` | Finest refinement sampling in Å/px; defaults to the coarse-search pixel size. |
| `--max_missing_tilts` | -1 | Optional coarse coverage culling; disabled by default. |
| `--refine_max_shift` | 3 coarse pixels | Maximum displacement per coordinate from the proposal anchor. |
| `--refine_merge_fraction` | 0.25 | Merge thresholds relative to the current band pixel. |
| `--batch_angles` | 8 | Upper bound on simultaneous orientations; reduced to fit GPU memory. |

`--npeaks 1000 --refine_starts 32` selects at most 1,000 spatial peaks and starts at most 32 pose optimizations for each: up to 32,000 initial hypotheses. Each peak contributes at most one final pick, and final spatial suppression can reduce the pick count further. Set the peak limit with room for false positives; it is a search limit, not a requested number of accepted particles.

Leaderboards require `8*K` bytes per padded voxel. Large volumes may require more than 48 GB even with a single orientation batch. Background-patch FFTs and refinement particles are processed in bounded batches.

Whitening and the multiresolution schedule are automatic. Top-K search followed by BFGS refinement is mandatory; there is no coarse-only mode or `--optimize_poses` switch. Saved coarse correlation/angle volumes are diagnostics, not restart files. The obsolete `--reuse_results` and `--dont_normalize` options have been removed; final scores are not standardized against a background correlation distribution. The CTF estimator itself is unchanged by this template-matching update.

Pose refinement uses only FP32 BFGS. The experimental Gauss–Newton optimizer and `--refine_optimizer` selector have been removed.
