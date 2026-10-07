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
4. Maintain the top eight orientation scores and IDs per voxel while streaming orientations. Detect immediate-neighbor maxima and greedily apply spherical spatial suppression. Gather the peak's and six adjacent voxels' pose lists; only sparse lists leave the GPU. Default: 8,000 proposals, up to 32 starts per proposal. A memory budget caps the concurrent orientation batch.
5. Refine poses using GPU-resident FP32 BFGS with analytic derivatives of Fourier interpolation, translation, and defocus. Particle rotations are exact rotations; local deformation and viewing geometry are frozen at each proposal. Physically pad the template and particle patches for its support, CTF delocalization, and allowed motion. Start at the coarse-search resolution limit and double spatial frequency at each transition (halve the resolution in Å), capped at the requested fine limit: for example, 20 → 10 → 6 Å. Identical coarse and fine limits give one band and zero resolution transitions. There is no stage-count parameter. Merge only nearly identical hypotheses between bands.
6. Measure an anisotropic per-tilt patch PSD and use the hybrid metric
   `W = 1 / [N + c * max(P - N, 0)]`.
   `c` counts geometrically overlapping components among that candidate's usable tilts. The overlap length is `max(400 Å, 3 × target diameter)`. The quadratic metric receives this weight once; independent detector noise is not multiplied by multiplicity. The particle core and its allowed motion must remain in view. Padding outside image boundaries is filled with zero after subtracting the observed patch mean; patches touching contamination masks are excluded. At least three usable tilts are required.
7. Before the final band, use up to 300 spatially distinct strong candidates to fit one shared dose-damage slope and pooled relative tilt amplitudes. The slope grid is 0, 1, 2, 3, 4, 5, 6, 8, 10 Å²/(e/Å²); the reference envelope is `exp(-slope * dose * q² / 4)`. Score comparisons use fixed-pose per-tilt sufficient statistics. Tilt amplitudes have their overall scale removed. A single-band request repeats that band after calibration. Calibration requires at least eight usable candidates and does not use labels or a requested particle count.
8. At final poses, estimate amplitude bounds from the 2nd and 98th percentiles of up to 300 distinct strong candidates. Rank by bounded-amplitude weighted log likelihood, then apply final spherical suppression. With fewer than eight calibration candidates, use the nonnegative free-amplitude profile score.

Boundary padding follows the research implementation; truncation of weak delocalized template tails at detector edges is an approximation, not an explicit masked-pixel likelihood. The weighting is a diagonal approximation to correlated structural background, not a full covariance likelihood. The multiplicity length and empirical amplitude interval are defaults supported by the synthetic experiments; their transfer to heterogeneous experimental particles needs evaluation. The highest-scoring starts are alternative explanations, not independent evidence.

## Scores and outputs

For weighted inner product `C` and model power `P`:

```text
projection score = C / sqrt(P)
unconstrained nonnegative amplitude = max(C, 0) / P
bounded amplitude = clamp(C / P, lower, upper)
final score = bounded_amplitude * C - bounded_amplitude² * P / 2
```

`rlnAutopickFigureOfMerit` contains the final score. `wrpTemplateMatchProjectionZ` retains the projection score and `wrpTemplateMatchAmplitude` the unconstrained nonnegative amplitude. These are ranking statistics, not calibrated Gaussian Z scores or false-discovery probabilities. Final bounded scores can be negative.

The matching directory also contains:

- `_starting_poses.tsv`: proposal centers, neighboring starts, angle IDs, and coarse scores.
- `_refinement.tsv`: per-band poses, scores, convergence, merging, and optimizer diagnostics.
- `_tilt_scores.tsv`: final per-tilt cross terms and model powers.
- `_calibration.tsv`: slope objective values and selected tilt-scale corrections.
- `_amplitude_bounds.tsv`: final interval and calibration-set size.

Optional amplitude/B diagnostics remain available through `--refine_fit_bfactor` and `--refine_export_tilt_spectra`. They do not select particles. Optional independent complete decoy searches remain diagnostics; decoys are not part of the default ranking.

## Controls and resource use

| Option | Default | Meaning |
| --- | --- | --- |
| `--tomo_angpix` | required | Coarse reconstruction and search sampling in Å. |
| `--subdivisions` | 3 | HEALPix angular sampling, approximately 7.5°. |
| `--peak_distance` | half the diameter | Spherical minimum particle separation in Å. |
| `--npeaks` | 8000 | Maximum coarse candidate positions. |
| `--match_topk` | 8 | Orientations retained per voxel. |
| `--refine_starts` | 32 | Maximum hypotheses pooled from each peak and its neighbors. |
| `--refine_iterations` | 90 | Maximum accepted BFGS steps per hypothesis per band. |
| `--refine_noise_patches` | 256 | Unselected patches per tilt for the directional background PSD. |
| `--optimize_poses_angpix` | `--tomo_angpix` | Finest refinement sampling in Å/px; defaults to the coarse-search pixel size. |
| `--max_missing_tilts` | -1 | Optional coarse coverage culling; disabled by default. |
| `--refine_max_shift` | 3 coarse pixels | Maximum displacement per coordinate from the proposal anchor. |
| `--refine_merge_fraction` | 0.005 | Merge thresholds relative to the current band pixel. |
| `--batch_angles` | 8 | Upper bound on simultaneous orientations; reduced to fit GPU memory. |

Leaderboards require `8*K` bytes per padded voxel. Large volumes may require more than 48 GB even with a single orientation batch. Background-patch FFTs and refinement particles are processed in bounded batches.

Whitening and the multiresolution schedule are automatic. Top-K search followed by BFGS refinement is mandatory; there is no coarse-only mode or `--optimize_poses` switch. Saved coarse correlation/angle volumes are diagnostics, not restart files. The obsolete `--reuse_results` and `--dont_normalize` options have been removed; final scores are not standardized against a background correlation distribution. The CTF estimator itself is unchanged by this template-matching update.

Pose refinement uses only FP32 BFGS. The experimental Gauss–Newton optimizer and `--refine_optimizer` selector have been removed.
