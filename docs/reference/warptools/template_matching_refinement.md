# Template matching with multistart pose refinement

`WarpTools ts_template_match --optimize_poses` proposes positions in a reconstructed tomogram, then refines each candidate against the original tilt images. Rebuild both WarpTools and NativeAcceleration together; this path uses new native entry points.

## First run

Use one tilt series and a modest proposal count for the first GPU smoke test:

```bash
WarpTools ts_template_match \
  --settings warp_tiltseries.settings \
  --input_data path/to/one.tomostar \
  --tomo_angpix 10 \
  --template_path target.mrc \
  --template_diameter 200 \
  --symmetry C1 \
  --subdivisions 3 \
  --batch_angles 1 \
  --npeaks 200 \
  --optimize_poses \
  --match_topk 8 \
  --refine_starts 32 \
  --refine_iterations 90
```

The tomogram at `--tomo_angpix` must already exist. The original tilt images and alignment/CTF metadata must remain accessible. Template dimensions and diameter, symmetry, and sampling should describe the actual target; the example values are placeholders.

## Search and refinement

1. For each orientation batch, the native correlation loop applies the effective 3D CTF to the rotated template before normalizing and correlating it with the tomogram. Each voxel maintains exactly its best K finite scores and angle IDs. Scores are sorted descending, with ascending angle IDs breaking ties.
2. Spatial peaks are selected from rank zero using the existing proposal settings. At each selected peak, its list and the six face-sharing neighbors' lists are gathered from the GPU. A neighbor's orientation retains that neighbor's 3D starting position. Only these small candidate lists are transferred; the K complete volumes are not copied to host memory.
3. Up to `--refine_starts` independent GPU optimizations refine the shared 3D position and orientation against fixed tilt patches. A 128-thread block cooperatively evaluates all tilts for one hypothesis and keeps optimization state on the GPU. Analytic derivatives propagate through Fourier interpolation, image shifts and CTF phase. Each particle has a precomputed local affine position-to-image/defocus mapping; its local tilt rotations are frozen. Particle rotations update exactly on SO(3). There are no CPU optimizer iterations or production central-difference pose gradients.
4. At each resolution stage, a separate GPU kernel greedily merges hypotheses close in both 3D position and symmetry-aware rotation. The higher-scoring representative survives; ties prefer the original seed index. Distinct survivors continue at finer resolution without averaging poses or replenishing discarded hypotheses.
5. The best solution supplies the candidate's final projection-space score. Duplicate final spatial detections are suppressed. Multiple starts reaching one answer do not add evidence.

Exact streaming top-K does not guarantee different angular basins. Closely spaced orientations may occupy every retained slot. The implementation keeps distinct translations when pooling neighboring lists; it does not replace a neighbor's position with the central peak.

The shared native local-peak finder excludes nonfinite values and breaks equal-score ties within the search radius by retaining the lower linear voxel index. This also prevents a flat plateau from producing a candidate at every voxel.

| Option | Default | Meaning |
| --- | --- | --- |
| `--match_topk` | 8 | Orientations retained per voxel with pose optimization enabled. |
| `--refine_starts` | 32 | Maximum starts per spatial peak, selected from at most 7K entries. |
| `--refine_iterations` | 90 | Maximum accepted optimization steps per start and resolution stage. |
| `--refine_optimizer` | `bfgs` | FP32 full BFGS; `gauss-newton` selects the retained mixed-precision trust-region implementation. |
| `--refine_merge_fraction` | 0.005 | Translation and template-edge rotation merge thresholds as a fraction of the current band pixel. Zero disables merging; maximum 0.5. |
| `--refine_max_shift` | 0 | Maximum displacement **per coordinate** from the proposal center, in Å. Zero selects three tomogram pixels. |
| `--refine_noise_patches` | 32 | Unselected patches per tilt used for the fixed radial background-power estimate; minimum 2. |
| `--optimize_poses_angpix` | tomogram sampling | Finest refinement sampling. Must be positive and no coarser than `--tomo_angpix`. |
| `--optimize_poses_steps` | 1 | Number of refinement resolution stages. |
| `--npeaks` | 2000 | Maximum spatial proposals; discarded proposals cannot be recovered by refinement. |
| `--batch_angles` | 8 | Concurrent global-search orientations. Lower this first if FFT scratch memory is too large. |
| `--decoy_templates` | none | Comma-separated maps for optional independent complete decoy searches. |

Leaderboard storage costs **8K bytes per padded voxel**: one float score and one exactly represented float angle ID. For example, K=4 on a 512³ padded volume costs 4 GiB for the leaderboards alone. FFT buffers, the tomogram, templates, and refinement observations need additional memory. The last partial angular batch uses correctly sized FFT plans.

`--reuse_results` is rejected with `--optimize_poses`: old saved correlation/angle volumes retain only one orientation per voxel. Full top-K volumes are not persisted in this version. Runs without `--optimize_poses` retain the existing coarse picking mode.

## Meaning of the refined score

With the background spectra and relative tilt weights fixed, accumulate across the tilt stack:

```text
C = weighted data–model inner product
P = weighted model power
Z = C / sqrt(P)
amplitude = max(C, 0) / P
score = 0.5 * max(Z, 0)^2
```

The optimizer maximizes signed `Z`; the reported score profiles one nonnegative particle amplitude shared by the entire tilt stack. Model power normalizes the score, without dividing by the candidate patch's observed power. Multiplying the complete predicted tilt stack by a positive constant leaves the score invariant.

For refined output, the STAR columns are:

| Column | Meaning |
| --- | --- |
| `rlnAutopickFigureOfMerit` | Profiled projection-space detection score. |
| `wrpTemplateMatchProjectionZ` | Signed normalized cross term Z; the name does not imply calibrated Gaussian significance. |
| `wrpTemplateMatchAmplitude` | Fitted shared nonnegative particle amplitude, in the current template's scale. |

These scores are not absolute Bayes factors or posterior probabilities. The background model is diagonal in Fourier space with radial spectra per tilt; correlated biological clutter across tilts and spatial variation within a tilt remain approximations to validate on data. Local patches, interpolation, support, template mismatch, and the proposal cap can affect outcomes.

Patches stay fixed throughout each candidate's optimization, with all starts sharing the same usable tilts. Patches that extend outside an image or touch a contamination mask are excluded; at least three usable tilts are required. Image position and defocus follow the full local affine Jacobian at the original coarse anchor, including deformation, movement, image-size rounding and defocus hand. Local tilt rotation, spatially varying envelopes and astigmatism stay fixed at that anchor. This assumes local deformation varies negligibly over the allowed particle displacement; spatial angle-grid variation and higher-order geometry changes are deliberately omitted. Refinement uses a hard Fourier cutoff at `--lowpass` times Nyquist; the coarse search keeps its existing Gaussian filter.

The template mask preserves the specified particle radius (`--template_diameter / 2`) and tapers outward over `max(20 Å, 5 coarse pixels)` in global search or `max(20 Å, 3 current pixels)` in refinement. On 2026-09-19 an extra division by two was removed from both callers of `Image.MaskSpherically`, whose argument is a diameter. Earlier benchmark results below used the smaller masks. Changing the mask changes the objective and coarse proposals, so new scores must not be treated as another comparison on the earlier objective.

The coarse score-normalization option affects proposal selection only. Final projection scores are not converted to z-scores using random locations. Starts outside the stage merge thresholds survive, so a lower-scoring distinct coarse-stage start can win at finer resolution. With effective band pixel size `b = box * pixel / (2 * cutoffRadius)`, with `f = --refine_merge_fraction`, merging requires position separation below `f*b` and symmetry-aware angular separation below `asin(min(1, 2*f*b/diameter))`. The conservative default targets nearly identical poses; Beam’s half-band-pixel rule (`f=0.5`) was too permissive in the apoferritin comparison. Only already-retained representatives suppress later candidates.

The default optimizer is **full BFGS in FP32**, with one 6×6 inverse Hessian in shared memory per hypothesis. It computes the model and derivatives, 14 weighted sufficient statistics, compensated per-thread sums, tree reductions, normalized gradient, inverse-Hessian update and line search in single precision. Fourier-sample inclusion uses integer squared radii with a cutoff threshold computed once on the host, preserving the cutoff without FP64 arithmetic per sample. Diagnostic arrays retain the existing double-precision ABI; conversion at output does not increase the calculation's precision. CPU noise-spectrum estimation and output bookkeeping still use double precision. CUDA's standard trigonometric functions retain an internal FP64 range-reduction fallback for arguments above approximately 105,615 radians; this is not expected for ordinary bounded refinement poses. The compiled score, gradient, BFGS update and line-search arithmetic contains no FP64 operations.

BFGS minimizes `-Z` in scaled coordinates `(position/pixel, omega*diameter/(2*pixel))`. Orientations use a fixed local chart `R = Rbase * exp([omega]x)` with the SO(3) right-Jacobian chain rule. If the chart exceeds 0.5 radians, the current orientation becomes the new base and the inverse Hessian resets. Gradients from different charts are never subtracted to form a curvature update. Each stage also starts with a fresh inverse Hessian.

The initial inverse Hessian scales a gradient step to approximately one pixel in the combined position/rotation metric. Trial directions are uniformly capped to at most two current pixels of translation and 5° rotation, and translations are clipped to the original coarse-anchor bounds. Projected Armijo backtracking tries at most 16 step lengths, followed by one fresh projected-gradient direction with up to 16 trials if needed. Every accepted step must improve Z by more than `8*FLT_EPSILON*max(1,abs(Z))`. A small-step stop requires a full, uncapped accepted direction below 0.005 current pixels and 0.01°; backtracking or bound clipping alone cannot imply convergence. Unreliable curvature updates are skipped, and a candidate inverse Hessian must pass an FP32 positive-definiteness check before replacing the previous one. Exhausting both searches reports a line-search stall.

The FP32 merge kernel compares squared rotation-matrix differences against `8*sin(angle/2)^2`, avoiding loss of precision from a cosine near one at the very small merge thresholds. The Gauss–Newton path retains its original double-precision merge calculation.

The retained `--refine_optimizer gauss-newton` uses six-dimensional Gauss–Newton curvature for signed Z, a trust-region eigensolve, and a rotation metric of half the particle diameter. With model Jacobian J, `c = Re(J* W d)`, `v = Re(J* W m)`, and `G = Re(J* W J)`, the ascent gradient and positive-semidefinite curvature are:

```text
g = (c - C/P * v) / sqrt(P)
s = max(abs(Z), 0.001)
H = s/P * (G - v vᵀ/P)
```

Here `s` is held fixed for each trust step: the normalized-model residual `(d - s*m/sqrt(P))/sqrt(s)` has squared norm equal to a constant minus `2Z`. This retains an ascent direction for negative-scoring hypotheses. Steps are capped at 5° rotation and two current pixels of 3D translation, while the original coarse-anchor translation bounds remain in force. The trust radius contracts or expands according to actual versus predicted signed-Z improvement. An accepted step below 0.01° and 0.005 current pixels stops that hypothesis only when the proposed step was smaller than 80% of the pre-trial trust radius; a contracted radius alone must not imply convergence; five consecutive rejections stop it without claiming convergence. Each resolution stage initializes a fresh trust radius and curvature at its surviving poses.

Observations are shared by all hypotheses of a particle. Particle batches use at most 64 particles and a conservative transient-memory budget of the smaller of 512 MiB and one quarter of free GPU memory. Separate score-only trial evaluations avoid computing Jacobians for rejected poses. Final per-tilt statistics and symmetry-aware merging use separate kernels.

`*_starting_poses.tsv` preserves the original coarse anchor, starting position/orientation, angle ID, rank and proposal score for each seed.

`*_refinement.tsv` records every hypothesis before merging: stage, original peak and seed, position, Euler angles, signed Z, gain, amplitude, accepted-step count, convergence flag, usable tilt count, termination reason, evaluation count, initial Z, final trust radius and the retained seed that absorbed it (`-1` means retained; `-2` means invalid or insufficient tilt coverage). `StepTolerance` means a small accepted step; `IterationLimit` means the accepted-step budget was exhausted; `TrustRegionStalled` means five consecutive rejected or unusable Gauss–Newton steps; `LineSearchStalled` means BFGS exhausted its line searches. The `trust_radius_A` column is zero for BFGS, which has no trust radius. `InsufficientTilts` and `InvalidPose` record discarded starts explicitly. A rounding-level score change never counts as an accepted step. `*_tilt_scores.tsv` records the winning solution's cross term and model power by tilt. The final STAR contains particles after spatial duplicate suppression.

## Joint amplitude/B diagnostics

Add `--refine_fit_bfactor` with `--optimize_poses` to fit a shared nonnegative amplitude and additional isotropic B at each final winning pose. This is diagnostic: poses, noise/CTF parameters, STAR scores, ranking and final spatial suppression retain their existing behavior. No candidate is removed based on the fitted parameters.

The amplitude/B fit now defaults to **frequencies strictly above 1/30 Å⁻¹** (`--refine_fit_highpass 30`). Its upper frequency remains the final pose-refinement cutoff, giving a 30–10 Å band in the TS_1 test. `--refine_fit_highpass 0` restores full-band fitting. This option affects only the final amplitude/B fit; pose refinement retains its original band. The lower boundary is applied to each physical Fourier sample before histogram deposition, including pixel anisotropy and magnification corrections. Samples at the boundary are excluded. The noise estimate remains fixed and unchanged.

```text
model(a, B, q) = a * exp[-B * (q² - qref²) / 4] * model_at_B0(q)
a(B) = max(C(B), 0) / P(B)
gain(B) = 0.5 * max(C(B), 0)² / P(B)
```

`q` is physical frequency in Å⁻¹, including the same magnification and pixel anisotropy corrections as the CTF. **Positive B attenuates high frequencies; negative B sharpens relative to the existing transfer model.** The model already includes dose and local B terms: a negative additional particle B can compensate for excess attenuation in those terms, without implying a sharper particle than the unfiltered reference. Amplitude is solved analytically at every B evaluation, making this a joint fit with only one numerical search parameter. The B range is −5,000 to +20,000 Å². The search scans the profile, refines local maxima and both edge intervals, and always considers B=0 and the exact bounds.

One fixed-pose GPU pass uses the same projection, depth-dependent CTF and independent Fourier samples as refinement. It accumulates weighted cross terms and model power into 8,192 uniform q² bins using linear deposition and FP32 arithmetic. The compact spectra are then fitted on the CPU in double precision, without repeated projections or patch extraction. Binning approximates the continuous-frequency envelope; CUDA tests compare nonzero-B results against independent per-sample evaluation without binning.

Amplitude is reported at **one common frequency per tomogram**: the mean of the candidates' fitted, normalized model-power mean q² values, with equal weight for each retained candidate. This places the reporting frequency near the signal surviving attenuation. It is an exact change of parameterization, changing amplitude and its covariance with B while preserving B, prediction and likelihood. The amplitude has the template's arbitrary scale and differs from the existing B=0 STAR amplitude.

`*_envelope.tsv` contains the 1-based final STAR row, original zero-based coarse peak ID, physical position, original Z/amplitude, common qref², fitted B/amplitude/log-amplitude, fitted Z/gain, gain improvement, uncertainties, log-amplitude/B correlation and fit status. `fit_highpass_A` records the fitting cutoff; `fit_z_at_b0` and `fit_amplitude_at_b0` record the baseline within that same restricted band. The gain improvement compares B=0 and fitted B using identical frequencies. The original `raw_z` and `raw_amplitude` retain the full pose-refinement-band values and should not be substituted for the restricted-band baseline. Bounds produce `lower_bound` or `upper_bound`; degeneracies produce `unidentified`, `nonpositive` or `zero_power`. The 1σ values come from two-parameter Fisher information, conditional on the selected pose and fixed noise model. They are not calibrated outlier probabilities.

`*_envelope_spectra.bin` preserves the sufficient statistics. All numbers are little endian: eight ASCII bytes `WRPENV01`; int32 version (2), row count and bin count; float64 common qref² and float64 minimum q². Each row contains int32 original peak ID, float32 maximum q², float32 cross[bin count], then float32 power[bin count]. Bin centers are `i * maximum_q² / (bin_count - 1)`; rows follow the TSV and final STAR order. The analysis reader also accepts version 1, which has no minimum-q² field and represents full-band spectra. Because the cutoff precedes linear deposition, a bin immediately below the boundary may contain a contribution from an eligible sample just above it; do not apply a second cutoff to bin centers.

Run `scripts/template-matching-decoys/analyze_envelope.py PREFIX --output NEW_DIRECTORY` to verify the spectra against the B=0 scores in the fitting band, independently refit with SciPy and plot the joint distribution and frequency fits. `--previous PREVIOUS.star` adds a comparison to an earlier run. Additional B is relative to the reference and existing transfer model; inspect it against the within-tomogram distribution and its covariance with amplitude, rather than against zero alone.

### First joint-fit test on TS_1, 2026-09-19

The final opt-in run completed in **49.28 s** on the L40S at `sc1ns051is14`, retaining all 200 candidates with the established 80 Å spacing, 32 starts and 20→10 Å refinement settings. This is a whole-command timing, not an isolated overhead benchmark. The additional spectra kernel uses 47 registers per thread with no register spills. All **64 template-matching tests passed**, including 27 CUDA cases, and Compute Sanitizer reported zero memory errors for the three new CUDA cases.

The common reporting frequency corresponds to **45.31 Å**. Amplitude has median **113.89** and interquartile range **107.60–119.30**, in the current template's arbitrary scale. B has median **4,647 Å²**, interquartile range **2,625–5,538 Å²**, and range **−136 to 12,403 Å²**. No fit hit a bound. The median improvement in projection Z is **0.769**; the existing STAR continues to report the B=0 score. Median conditional uncertainties are 724 Å² for B and 0.050 for log-amplitude.

Row **200** (coarse peak 107) is a joint-parameter outlier worth inspecting: amplitude **54.34**, B **12,403 Å²**, original Z **7.075**, and 13 usable tilts. Its conditional B uncertainty is 3,096 Å², so these parameters do not establish a false detection. In contrast, row **199** has Z 8.655 with only seven usable tilts but an amplitude of 107.48 and B −136 Å². The fit helps distinguish unusual parameter combinations from a low score caused partly by limited observations. No removal threshold was applied.

The B profiles also expose a limitation of a single optimum plus local uncertainty: **51/200** candidates have another interior maximum; **18** have an alternative within two profile-gain units, including six within one unit. The analysis saves these alternatives in `summary.json`. Large positive B is common in this dataset, and a blanket B threshold or a local Fisher error alone would misrepresent the result. Future outlier scoring should account for the full joint profile and candidate coverage.

Independent SciPy fits agree with the reported B values within 0.0022 Å² and reproduce gains within 3×10⁻¹²; the saved B=0 spectra reproduce the original scores. Against the stored corrected-mask target run, all 200 candidates match uniquely, with median coordinate difference 0.0091 Å, maximum 0.8742 Å and maximum absolute Z difference 0.00914. This rerun is close but not bit-identical to that earlier run.

Final outputs are under `real-data/apoferritin-ts1/joint_envelope_20260919_v2` in `/home/tegunovd/projects/warp-template-match-20260916-01a0a787`; final analysis is in `analysis_final`. Build/test/sanitizer logs, the command and source/binary hashes are in `cluster-validation/joint-envelope-20260919`. The original experimental job was untouched.

### Audit of the large B values

The user subsequently confirmed that all 200 TS_1 picks are particles. The unusual fitted parameters therefore cannot be used as evidence of false detections in this test. The previous outlier description refers only to fitted parameters, not particle identity.

The objective and units were audited. The candidate B modifies only the reference prediction: data, inverse-noise weights and the selected Fourier samples stay fixed. With `D = sum(w*|data|²)`, `C(B) = sum(w*Re(data*conj(model_B)))` and `P(B) = sum(w*|model_B|²)`, the loss is `D - 2*a*C(B) + a²*P(B)`. Profiling the nonnegative amplitude gives `D - max(C(B),0)²/P(B)`, so the implemented gain maximization is weighted-L2 minimization. At a fixed pose and fixed data norm, it also has the same positive-correlation optimum as weighted NCC; that equivalence alone does not distinguish the methods. Attenuating the reference does not remove high-frequency data residuals.

The coordinates are cycles/Å, with an amplitude envelope `exp(-B*q²/4)`. For B=4,000 Å², half amplitude relative to zero frequency occurs at **37.98 Å**. The factors at 40, 30, 20 and 10 Å are 0.5353, 0.3292, 0.08208 and 0.00004540. With the displayed amplitude normalized at the common 45.31 Å pivot instead, half of that reference amplitude occurs at 29.11 Å. The reporting pivot changes neither the fitted prediction nor B.

Independent band-restricted fits to the saved spectra, holding every pose fixed, expose strong dependence on coarse-frequency model mismatch:

| Included band | Median B (Å²) | Median for original B<1,000 group | Median for original B>3,000 group |
| --- | ---: | ---: | ---: |
| Full existing band through 10 Å | 4,647 | 330 | 5,132 |
| 60–10 Å | 305 | 154 | 376 |
| 40–10 Å | 3 | −54 | 5 |
| 30–10 Å | −144 | −145 | −138 |

With 40–10 Å fitting, 192/200 have B below 1,000 Å²; eight remain above 3,000, including five at the 20,000 Å² bound. The variation is not simply monotonic with a high-pass cutoff: selecting 50–10 Å produces many very large/boundary fits, emphasizing the non-exponential shape of the spectral mismatch. These are diagnostics, not changes to production defaults or a validated recommendation to apply a 40 Å high-pass.

The two groups are consistent with switching between competing low-B and broad-envelope solutions. Sixteen members of the original low-B group have a second high-B maximum; 34 members of the original high-B group have a second low-B maximum. For example, row 155 has a second solution near B=3,685 only 0.136 gain units below its selected low-B solution. The audit supports treating large full-band B values as a model diagnostic rather than a physical blur estimate. It does not yet identify whether the coarse-frequency mismatch comes from surrounding-particle/background signal, reference shape/scale, transfer modelling, or another forward-model issue.

Audit scripts, per-particle band fits and a comparison figure are under `cluster-validation/joint-envelope-audit-20260919` in the same isolated cluster project. The existing native/managed optimization code and all particle selections were left unchanged during this audit.

### Exact 30–10 Å fitting on TS_1, 2026-09-19

The native spectrum pass now excludes physical samples with `q² <= 1/900 Å⁻²` before binning. This differs slightly from the preceding audit, which selected already-binned statistics. The rerun used `--refine_fit_highpass 30` and the same 200 candidates, 80 Å spacing, 32 starts and 20→10 Å pose-refinement settings. It completed in **53.30 s** on the L40S. The final STAR is **byte-for-byte identical** to the full-band joint-fit run, including poses and original scores.

| Diagnostic | Restricted 30–10 Å fit |
| --- | ---: |
| Median B | −147.57 Å² |
| B interquartile range | −322.17 to +25.26 Å² |
| B range | −1,528.28 to +1,031.96 Å² |
| Fits at a B bound | 0 / 200 |
| Median amplitude | 59.93 at a common 13.95 Å reporting frequency |
| Amplitude interquartile range | 50.70–75.00 |
| Median conditional B uncertainty | 289.50 Å² |
| Median conditional log-amplitude uncertainty | 0.210 |
| Median improvement in Z within the fitting band | 0.06406 |

The large positive-B population disappears. Only **2/200** profiles have a second interior maximum, compared with 51 for the full-band fit; neither alternative is within two profile-gain units of the selected solution. This removes the earlier two-population appearance in this test, without establishing the cause of the low-frequency mismatch. Amplitudes use the same arbitrary template scale but a different reporting frequency, so their numerical values should not be compared directly with the full-band amplitudes reported at 45.31 Å.

All **65 template-matching tests passed**, including 28 CUDA cases with none skipped. Tests check the strict cutoff against independent per-sample scoring, poison excluded data with NaNs, and verify the managed physical-frequency mapping and unchanged pose scores. Compute Sanitizer reported zero memory errors for the cutoff and managed-pipeline cases. Independent SciPy fits reproduce B within 0.00037 Å² and profile gains within 1.4×10⁻¹³.

Outputs and final analysis are in `real-data/apoferritin-ts1/joint_envelope_highpass30_20260919` under the same isolated cluster project; build/test logs, commands and source/binary hashes are in `cluster-validation/envelope-highpass30-20260919`. The original experimental job was untouched.

## Experimental shared tilt-image signal scales

Add `--refine_export_tilt_spectra` with `--refine_fit_bfactor` to preserve noise-weighted C/P statistics separately for each final particle and tilt image. The existing aggregate fit is still written. This export is opt-in because its host storage and output scale as `particles × tilts × 2 × 8192` floats. Device batches account for the additional spectrum buffer. The original aggregate native ABI is retained; `TemplateMatchEnvelopeSpectraByTilt` uses the same physical samples and model with output `[particle, tilt, cross-or-power, bin]`.

Run `python scripts/template-matching-decoys/fit_tilt_scales.py PREFIX --output NEW_DIRECTORY` to fit the diagnostic calibration. At fixed poses, the prediction is `a[p] * s[t] * exp(-B[p]*(q²-qref²)/4) * original_model[p,t,q]`. There is **one nonnegative scalar s[t] per entire tilt image**, shared by every particle. Each particle has its own nonnegative amplitude and B; no particle/tilt pair gets a free signal scale. The current 30–10 Å band and per-tilt noise weights remain fixed. A factor of 1.2 means 20% more predicted signal than the original transfer model in that tilt.

The original model uses `cos(tilt angle)` unless a dose-weight grid is available, multiplied by local spatial weights. Its B includes dose-dependent and local terms. These are signal-model terms; the loader does not independently standardize each refinement patch. The reported new factors multiply the entire original prediction, including any existing fitted weights, rather than replacing the noise spectrum or dividing both data and model by the factor.

Particle amplitudes are profiled analytically: with per-tilt cross and power after applying B denoted C[p,t] and P[p,t], `a[p] = max(sum_t s[t]*C[p,t], 0) / sum_t s[t]²*P[p,t]`. The optimizer maximizes half the sum of these profiled squared Z values, equivalent to minimizing weighted L2 with a fixed data term. SciPy L-BFGS-B optimizes nonnegative tilt scales and particle B with analytic derivatives. The arithmetic mean of active tilt scales is fixed to 1; this normalization remains defined when a weak tilt attains zero fitted signal and resolves their common multiplicative ambiguity with particle amplitudes. Disconnected tilt-coverage graphs are rejected because a single normalization cannot identify their relative scales.

The script also fits a B=0 control. The joint fit starts from both that control and the original particle B values, then audits each B using an independent grid/bracket profile search. It fits scales independently on two disjoint, seeded particle halves and applies each scale curve to the other half, refitting only that half's particle amplitudes and B. This is a stability diagnostic, not strictly independent noise validation: all patches come from the same tilt images, some overlap, and the poses were selected using these data.

Outputs are `tilt_scales.tsv`, `particle_fits.tsv`, `fit_arrays.npz`, `summary.json` and `tilt_calibration.png`. The scale table includes tilt identity, angle, dose, particle coverage, original model scale/B at the tomogram center, joint-fit and B=0-control multipliers, and the two half-set estimates. Particle amplitudes before and after calibration use one common reporting frequency. No STAR scores, poses, image pixels, XML weights or original experimental files are changed by the offline fit. This first experiment measures calibration; it does not yet apply the scales during pose optimization or global search.

The binary export is little endian: eight ASCII bytes `WRPTENV1`, int32 version (1), particle count, tilt count and bin count, then float64 minimum q². Each particle row contains int32 original peak ID and float32 maximum q², followed by `tilt count` pairs of float32 cross[bin count], power[bin count]. Rows match the final STAR. `*_tilt_model.tsv` supplies tilt identity and baseline transfer-model metadata. Summing over tilts reproduces the aggregate histograms within FP32 accumulation error; the analysis verifies this before fitting.

### TS_1 shared-scale experiment, 2026-09-19

The 200-particle, 41-tilt export completed in **58.02 s** on the L40S at `sc1ns051is14`, retaining the established 80 Å spacing and 30–10 Å fitting band. The final STAR is byte-for-byte identical to the preceding high-pass run. All **66 template-matching tests passed**, including 29 CUDA cases; Compute Sanitizer reported zero memory errors. Six independent Python tests check the profiled objective against direct weighted L2, analytic derivatives, recovery of shared scales and particle parameters with missing views, the amplitude gauge, zero-signal tilt boundaries and disconnected coverage. All passed.

The inferred multipliers span **0.2608–2.0699** with arithmetic mean 1. They are highest around the low-angle tilts and much smaller at high absolute angles, after accounting for the existing cosine scale and dose B. For example, the +2.01° tilt has factor 2.0699 and +36.01° has factor 0.2608. The +40.01° and −39.98° endpoints have factors 0.2777 and 0.5412. These are relative corrections to the signal model, not calibrated absolute image gains. Angle, accumulated dose and other acquisition effects are coupled in this series, so this test does not isolate the physical cause.

| Particle B diagnostic | Existing tilt model | Shared tilt scales |
| --- | ---: | ---: |
| Median | −147.57 Å² | +8.70 Å² |
| Interquartile range | −322.17 to +25.26 Å² | −148.45 to +182.90 Å² |
| Negative estimates | 146 / 200 | 94 / 200 |
| Fits at a B bound | 0 | 0 |

Fixing all particle B values to zero gives almost the same scale curve: the RMS factor difference from the joint fit is **0.0174**, with maximum 0.0424. Two distinct joint-fit starts agree in total profile gain within 3×10⁻¹⁰, and independent conditional B searches improve it by less than 7×10⁻¹⁰. The joint total profile gain increases from 2,953.47 to 3,550.59. This is a 20.2% increase in profile gain, not a percentage reduction in the full data residual, whose constant term is omitted.

The two disjoint 100-particle halves produce scale curves with correlation **0.9298** and RMS difference 0.2303. Applied to the opposite half with only particle a/B refitted, their total gains improve by **263.72** and **271.67**; 80/100 and 76/100 particles improve individually. The weak +40.01° tilt hits zero scale in one half, compared with 0.5715 in the other, indicating uncertainty at that endpoint. All 41 scales are positive in the full-set fit. This is evidence of a reproducible mismatch in the existing relative tilt signal model, with the patch-overlap and pose-selection qualifications above.

The analysis, including both joint-fit starts, profile audits, two half-set calibrations, controls and plots, takes about **14 s** on the cluster CPU after spectrum export. The final arrays, per-tilt and per-particle tables, plot and summary are in `real-data/apoferritin-ts1/shared_tilt_scales_20260919/calibration_final` within the isolated validation project. Commands, tests, build/sanitizer logs and source/binary hashes are in `cluster-validation/shared-tilt-scales-20260919`. Conditional particle B errors in the output hold learned tilt scales, poses and the noise model fixed. The original experimental directory was untouched.

## Optional whole-pipeline decoys

Add, for example:

```bash
--decoy_templates decoy_01.mrc,decoy_02.mrc,decoy_03.mrc
```

Every decoy runs its own global orientation search, peak selection, neighbor pooling, multistart refinement, and final suppression. It uses the target's diameter, symmetry, proposal limits, and refinement settings. Decoy maps must have the target's dimensions and sampling. Without `--template_angpix`, the headers are checked for matching isotropic pixel sizes; an explicit override applies to every supplied map.

Decoy output names contain `__decoy_NNN_` and the map name. A custom `--override_suffix` also gets that unique decoy suffix, protecting the target STAR. Work-queue task IDs distinguish passes and templates. Only fresh outputs from successful target and decoy runs are collected.

For a series where the target and **every** decoy succeed, a companion `*_decoy_calibration.tsv` contains the target STAR row, target score, total and per-decoy exceedance counts, number of complete decoy searches, and the mean number of decoy outputs at least that strong. Successful searches with no detections contribute zero counts and one search of exposure. Failed or missing searches never count as empty searches.

The count increment is `1 / number_of_decoy_searches`. Zero observed exceedances is explicitly marked `unresolved_zero_exceedances`, not treated as zero false-positive risk. These empirical counts describe the chosen, potentially capped search procedure. They become meaningful false-match predictions only if the decoys reproduce false matching to the target; matching power spectra and running identical code do not establish that assumption. No per-particle posterior or FDR is produced.

### First apoferritin decoy pilot, 2026-09-19

The target and eight independently generated O-symmetric maps were searched on TS_1 using the L40S at `sc1ns051is14.eth.rsiec.sc1.science.roche.com`. Every pass used 12,096 global orientations, 20 Å search resolution, 200 coarse proposals, 80 Å peak spacing, K=8, 32 refinement starts, 90 accepted steps per stage, GPU BFGS and 20→10 Å refinement. All nine searches succeeded in **427.77 s** total. The original experimental directory was untouched; inputs and outputs remained under the isolated validation project.

The experimental generator is in [`scripts/template-matching-decoys`](../../../scripts/template-matching-decoys/README.md). It starts from seeded random fields, enforces the 24 proper O rotations and alternates radial Fourier-power matching with real-space constraints for 80 iterations. Four `support` maps use a 65 Å core radius and a 20 Å outward cosine mask. Four `envelope` maps additionally preserve the target's radial mean and variance, retaining the hollow shell while changing texture. Generation uses a 128³ grid at approximately 2.04 Å/voxel, then restores the source's exact 330³ dimensions and 0.7894 Å sampling. Seeds and hashes are recorded; no map was selected using its tilt-series score.

The masks were corrected before this pilot. The former final-stage mask retained approximately **9.63%** of the template's squared density relative to the corrected mask in FFT-resampling QA. Target mean Z changed from 9.7315 to 20.5093; these are different objectives and proposal sets. There are 167 mutually nearest old/new picks within 40 Å, with median displacement 6.84 Å. The mask correction passed all **50 template-matching tests, including 24 CUDA cases with none skipped**. The first full decoy invocation also exposed a missing-output-directory error when invalidating old calibration; the command now creates that directory before removing stale calibration.

| Search | Final picks | Median Z | Maximum Z | Picks within 20 Å of a target pick |
| --- | ---: | ---: | ---: | ---: |
| Target | 200 | 20.919 | 25.135 | — |
| Support 1 | 194 | 11.768 | 17.046 | 1 |
| Support 2 | 199 | 15.910 | 20.641 | 125 |
| Support 3 | 197 | 10.355 | 14.486 | 0 |
| Support 4 | 200 | 17.857 | 21.707 | 157 |
| Envelope 1 | 200 | 18.977 | 23.118 | 175 |
| Envelope 2 | 200 | 19.461 | 23.167 | 173 |
| Envelope 3 | 200 | 19.302 | 23.615 | 167 |
| Envelope 4 | 200 | 19.503 | 24.204 | 171 |

The support family's tails vary widely, and two maps largely rediscover target locations. The shell controls consistently rediscover 167–175 of the target locations; their median nearest-target distance is 3.07–3.41 Å. At locations paired within 20 Å, the target wins 171/175, 171/173, 163/167 and 164/171 comparisons, respectively, with median Z advantages 1.585, 1.382, 1.495 and 1.322. These comparisons support a structural-specificity diagnostic but provide no particle truth labels.

| Target rank | Target Z | Mean support exceedances/search | Mean envelope exceedances/search |
| --- | ---: | ---: | ---: |
| 10 | 23.661 | 0 | 0.50 |
| 25 | 22.817 | 0 | 3.75 |
| 50 | 22.113 | 0 | 11.00 |
| 100 | 20.929 | 2.00 | 36.75 |
| 150 | 19.403 | 10.50 | 97.00 |

There are 62 target scores above all support decoys and five above all shell controls. **These are observed tail counts, not validated detection counts.** With four searches per family, the count increment is 0.25 outputs/search. Pooling the two families would hide their different behavior. An independent Python recount matched every count in WarpTools' pooled and per-decoy calibration TSV.

Construction also remains approximate: post-resampling/masking radial-power log-RMS errors were 0.227–0.326 at coarse sampling and 0.070–0.146 at fine sampling. The QA approximates the input masks, not the complete CTF/projection scorer. Density histograms and nonnegativity were unconstrained; a support decoy can have an inverted radial density profile. Shell controls deliberately retain true-template signal. Neither family has been validated as an exchangeable false-match model, and the pilot does not add per-tilt signal-scale fitting, pose marginalization or cross-defocus calibration.

Cluster artifacts are under `/home/tegunovd/projects/warp-template-match-20260916-01a0a787`:

- `real-data/apoferritin-ts1/decoys_20260919`: maps, construction manifest and template QA.
- `real-data/apoferritin-ts1/decoy_pilot_20260919_v2`: all nine searches and `analysis/summary.json`, `counts_by_target_rank.tsv`, `score_tails.png`.
- `cluster-validation/decoy-pilot-20260919`: commands, build/test logs and source/binary/output hashes.

## Validation

Host tests for the actual leaderboard insertion, spline derivatives, and scoring math can run without CUDA:

```bash
c++ -std=c++14 -O2 NativeAcceleration/tests/TopKCorrelationTests.cpp -o /tmp/warp-topk-test
/tmp/warp-topk-test

c++ -std=c++14 -O2 NativeAcceleration/tests/EinsplineGradientTests.cpp \
  NativeAcceleration/src/einspline/bspline_create.cpp -o /tmp/warp-spline-test
/tmp/warp-spline-test

c++ -std=c++14 -O2 -I NativeAcceleration/include \
  NativeAcceleration/tests/TemplateMatchRefineMathTests.cpp -o /tmp/warp-score-test
/tmp/warp-score-test

c++ -std=c++14 -O2 -I NativeAcceleration/include \
  NativeAcceleration/tests/TemplateMatchRefineBatchTests.cpp -o /tmp/warp-batch-test
/tmp/warp-batch-test
c++ -std=c++14 -O2 -I NativeAcceleration/include \
  NativeAcceleration/tests/TemplateMatchBatchMathTests.cpp -o /tmp/warp-batch-math-test
/tmp/warp-batch-math-test
c++ -std=c++14 -O2 -I NativeAcceleration/include \
  NativeAcceleration/tests/TemplateMatchBfgsTests.cpp -o /tmp/warp-bfgs-math-test
/tmp/warp-bfgs-math-test

dotnet test Tests/Tests.csproj --filter 'FullyQualifiedName~TemplateMatch'
```

On the CUDA host, rebuild NativeAcceleration, put it on the library search path, then enable the integration tests:

```bash
# Activate an environment matching warp_build.yml: CUDA 12.9, .NET 10, CMake <4.
# Architecture 89 targets an L40S; omit this setting for the release architecture list.
cmake -S NativeAcceleration -B NativeAcceleration/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DFFTW_ROOT="$CONDA_PREFIX" \
  -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build NativeAcceleration/build --parallel 8

export LD_LIBRARY_PATH="$PWD/NativeAcceleration/build/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
WARP_RUN_CUDA_TESTS=1 dotnet test Tests/Tests.csproj \
  --filter 'FullyQualifiedName~TemplateMatch'
```

Ensure there is no older NativeAcceleration library next to the test assembly, where it could take precedence over the rebuilt library. This build uses the shared CUDA runtime. Building LibTorchSharp is not required for these template-matching tests.

On 2026-09-16, the first cluster validation passed all 36 template-matching tests, including eight CUDA integration tests with no skipped tests, on an NVIDIA L40S using CUDA 12.9 and .NET 10. The three standalone C++ host suites also passed. WarpTools and its worker assemblies built, and `ts_template_match --help` ran successfully. CUDA coverage includes streaming top-K and sparse gathering, projector and independent CPU score comparisons, analytic derivatives, alignment geometry, synthetic noisy pose recovery, and independence between starting hypotheses.

An initial real-data run on one 41-tilt apoferritin series (130 Å diameter, O symmetry, 200 proposals, 10 Å search and 10/5 Å refinement) completed for K=1/R=1 and K=4/R=8. Start 0 reproduced the single-start result exactly, and additional starts improved projection Z by more than 0.01 for 90/200 candidates. This run exposed equal-score steps consuming the optimizer budget at float precision; the strict-improvement fix and a regression test subsequently passed the expanded 37-test suite, including all eight CUDA tests. A 90-iteration single-start follow-up also completed. These initial timings and outcomes do not establish detector accuracy or tuned iteration defaults.

The GPU-resident version was validated on 2026-09-18 UTC on the same L40S. All **39 then-current tests passed**, including 13 CUDA tests with none skipped; both new portable optimizer-math suites passed. The six retired managed-BFGS tests and their unused optimizer were removed. The scalar CUDA scorer remains an independent numerical oracle. The fused 128-thread optimizer uses **168 registers per thread, 2,544 bytes shared memory, and zero register spills** on SM89; the score-only trial path, final tilt-statistics kernel and separate merge kernel are retained.

The final apoferritin comparison uses the same 200 coarse anchors, 41 tilt images, 130 Å template, O symmetry and 10→5 Å/pixel stages (20→10 Å resolution cutoffs with `--lowpass 1`):

| Run | Starts per peak | Full command elapsed | Refinement preparation | Refinement batch processing | Final picks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous CPU-driven BFGS | 8 | 598.94 s | 8.722 s | 556.243 s | 179 |
| Final GPU trust-region defaults | 32 | 58.29 s | 10.133 s | 13.064 s | 183 |

Batch-processing intervals include extraction, FFT/CTF preparation, transfers, optimization and diagnostics; they are not pure kernel timings. These are single runs with different filesystem cache states. The algorithm and stopping/merge policies also changed, so the 10.28× total speedup is an end-to-end observation, not a scheduling-only benchmark. A separate synthetic 128²×41-tilt benchmark with 32 particles ×32 starts takes 0.352 s including native allocation/readback; those hypotheses accept only two steps, so this is not a 90-step timing.

Validation drove two deliberate differences from Beam's initial defaults. First, half-band-pixel merging discarded starts that later won at finer resolution. The default now merges only almost identical poses (`--refine_merge_fraction 0.005`); a controlled pre-guard comparison against retaining every start lost at most 0.001869 Z, with mean loss 0.000023. Second, the accepted-step budget is 90 and a small accepted step can terminate only when the proposed step was interior to the pre-trial trust region. This prevents a contracted radius by itself from being reported as convergence.

The final run has no invalid scores or within-stage score decreases, and stage survivors propagate exactly. Best final Z versus the earlier BFGS result differs by −0.000050 at the median, −0.026218 on average, and −1.063171 in the worst case; 26/200 candidates lose more than 0.1 Z. Median positional difference is 0.159 Å. The stop guard changes labels honestly: final-stage hypotheses report 5,232 `TrustRegionStalled` and 21 `IterationLimit`, rather than small-step convergence caused by a contracted radius. Remaining differences show local-optimum sensitivity, including a case where about 0.006 Å of coarse-stage motion changes the fine-stage basin. Score-equivalent replacement and detection accuracy are therefore not established by the speed result.

All runs used isolated copies. The original job's 569-entry metadata inventory remains unchanged. Code, logs, source/binary hashes, control runs and the reproducible `analyze_gpu_batch.py` comparison live under `/home/tegunovd/projects/warp-template-match-20260916-01a0a787/cluster-validation/gpu-batch` on the cluster; final data outputs are in `real-data/apoferritin-ts1/gpu_k8r32_final` beneath the same project.

A follow-up on 2026-09-18 changed only `--peak_distance` from 130 to **80 Å**, as appropriate for the closely packed TS_1 particles. This setting affects both global peak selection and final spatial suppression. The 200-proposal cap, 32 starts, 90 accepted-step limit, merge fraction and resolution stages were unchanged; serialized matching options were checked to differ only in `PeakDistance`.

| Minimum peak distance | Full command elapsed | Hypotheses through stages | Final picks |
| --- | ---: | --- | ---: |
| 130 Å | 58.29 s | 6,400 → 5,253 → 4,059 | 183 |
| 80 Å | 60.44 s | 6,400 → 5,242 → 3,951 | 200 |

The 80 Å run selected 176 of the same coarse anchors and 24 new anchors, displacing 24 others under the unchanged proposal cap. Its final picks contain 31 pairs separated by 80–130 Å; the closest pair is 122.25 Å. All scores were finite, no within-stage score decreases were found, and survivor propagation between stages was exact. Median accepted steps remained 18 and 19. For shared coarse anchors, median final position change was 0.0233 Å and median Z change was +0.0000012, although individual trajectories still diverged (maximum position change 11.17 Å). The changed candidate set prevents interpreting all 200 outputs as paired comparisons with the earlier run. These counts do not establish the number of true particles, and the 200-proposal cap can still limit coverage.

This run is preserved in `real-data/apoferritin-ts1/gpu_k8r32_distance80` under the same cluster project. Its command, binary/input hashes and coordinate-matched analysis are in `cluster-validation/gpu-batch/run-apoferritin-distance80.sh`, `source-distance80.json`, `analyze_peak_distance80.py` and `comparison-distance80.json`. The original experimental job was not used as an output directory.

The BFGS comparison was then repeated at **80 Å spacing, 32 starts, and 90 accepted steps per stage**. The original refinement driver was recovered from the session record and matched the initial archived SHA-256 exactly before adding validation-only seed replay and diagnostics. The recovered optimizer retains its historical strict-improvement fix. It was built separately in `/home/tegunovd/projects/warp-template-bfgs-validation-20260918`; the GPU implementation was not replaced. All 39 current template-matching tests passed with this build, including the CUDA tests and synthetic recovery through BFGS.

Both runs used exactly the same 200 coarse anchors and 6,400 starting hypotheses. The BFGS replay reconstructs the original source voxels and angle IDs, verifies their recorded positions and Euler angles exactly, and keeps every hypothesis between stages. At the starting poses, the two models' Z values differ by a median 0.0000080 and a maximum 0.000167 in absolute value.

| Optimizer at 80 Å spacing | Full command elapsed | Refinement processing | Final picks | Median accepted steps, coarse/fine |
| --- | ---: | ---: | ---: | --- |
| Restored BFGS | 3,246.72 s (54 min 6.72 s) | 3,197.569 s | 200 | 32 / 30 |
| GPU Gauss–Newton | 60.44 s | 13.209 s | 200 | 18 / 19 |

The observed full-command speed ratio is 53.72×. BFGS retains 6,400 hypotheses at both stages; Gauss–Newton reduces 6,400 → 5,242 → 3,951. BFGS recorded no invalid scores or within-stage decreases. Its termination counts were 6,368 line-search stalls and 32 iteration limits at the coarse stage, then 6,386 stalls and 14 limits at the finer stage; these are not gradient-convergence claims.

For the best final pose at each common anchor, **GPU Z minus BFGS Z** has median −0.000216, mean −0.108838, minimum −1.636022 and maximum +0.643225. BFGS is higher by more than 0.1 Z for 45/200 proposals; the GPU method is higher by that amount for 5/200. Another 121/200 differ by less than 0.01 Z in absolute value. Median pose differences are 0.1126 Å and 0.2124° modulo O symmetry, but their 95th percentiles are 18.568 Å and 34.469°, and maxima are 37.105 Å and 56.898°. BFGS has a noticeable reported-score advantage in the divergent cases. Neither the scores nor the common 200-pick count establish correct particle identities or poses.

This is a comparison of the complete refinement methods: BFGS recomputes full local geometry at every trial, uses bounded XYZ rotation increments of ±60° relative to each stage's starting rotation, and does not merge hypotheses. The GPU method freezes local geometry, uses accumulated SO(3) updates and conservative merging. Both use the same ±30 Å translation bounds around the original coarse anchors. The result therefore does not isolate GPU scheduling or the optimizer from those other differences.

Outputs are in `real-data/apoferritin-ts1/bfgs_k8r32_distance80_i90` under the original validation project; `cluster-validation/gpu-batch/comparison-bfgs-distance80.json` contains the paired results. The separate BFGS build preserves its recovered sources, instrumentation, command, test logs, binary hashes and `validation/analyze_bfgs_distance80.py` script. All experimental inputs remain isolated copies.

### FP32 GPU BFGS comparison

The FP32 BFGS implementation was validated on 2026-09-18 on the same L40S. **All 49 tests passed, including 23 CUDA cases with none skipped.** Portable BFGS tests compare rotation-chart derivatives against independent quaternion/finite-difference references and check positive definiteness, secant updates, unreliable-curvature rejection and quadratic minimization. CUDA tests exercise both optimizers, independent batch lanes, negative-Z ascent, total bounds, angular recovery, final-pose scalar scoring, zero model power and the production-scale angular merge threshold.

Compute Sanitizer found a shared-memory hazard between a trial-loop exit reading the line-search scale and the fallback attempt resetting it. A block barrier now orders those accesses. The final build passes **memcheck with zero errors and racecheck with zero hazards or warnings** on a 41-tilt optimization fixture. The results below use the subsequent `gpu_bfgs_fp32_distance80_checked` real-data run; earlier BFGS timing and trajectory results are superseded.

The checked run and Gauss–Newton control use the same 200 anchors and exact multiset of 6,400 physical starting poses, 80 Å minimum peak spacing, 130 Å diameter, O symmetry, 20→10 Å resolution cutoffs, 90 accepted steps per stage and merge fraction 0.005. Two angle entries at one peak swap initial rank between methods; comparisons match physical poses or coarse anchors, not raw start indices.

| GPU optimizer | Full command | Refinement batch processing | Hypotheses through stages | Final picks |
| --- | ---: | ---: | --- | ---: |
| Gauss–Newton control | 58.02 s | 13.204 s | 6,400 → 5,242 → 4,004 | 200 |
| FP32 BFGS, checked build | 51.93 s | 9.257 s | 6,400 → 4,426 → 2,851 | 200 |

BFGS reduced the measured batch-processing interval by **29.9%** and whole-command time by **10.5%**. These are single runs, not isolated CUDA-kernel timings: batch processing includes host work, transfers and diagnostics, while filesystem cache state and stage preparation also affect the full command. Separate refinement preparation took 9.815 s for Gauss–Newton and 7.680 s for BFGS. BFGS also retains fewer starts at the finer stage, so the runtime comparison includes that change in workload.

Best final **BFGS Z minus Gauss–Newton Z** has median +0.000029 and mean +0.02538. Seventeen proposals improve by more than 0.1 Z and six worsen by that amount; the range is −0.89495 to +1.07920. Median positional and O-symmetry angular differences are 0.0939 Å and 0.0656°, with 95th percentiles 10.53 Å and 16.43°. Neither optimizer universally wins. All scores are finite, accepted trajectories have no within-stage score decreases, and final picks obey the spatial separation constraint. The counts and scores do not establish detection accuracy.

A separate 128²×41-tilt accumulation comparison at 128 identical poses measured maximum relative C and P errors below 9.6×10⁻⁸ and maximum relative Z error **1.33×10⁻⁷** against the retained FP64-reduction scorer. Maximum absolute Z error was 0.0000139 at Z approximately 106. Across the real-data starting poses, the initial BFGS and Gauss–Newton Z values also agree to a few millionths. These comparisons check accumulation precision separately from changes in optimizer trajectories.

The checked BFGS optimizer uses **128 registers per thread, 964 bytes of shared memory and zero register spills**, versus 168 registers and 2,544 shared bytes for Gauss–Newton. Compiled PTX/SASS was audited: score, gradient, inverse-Hessian, line-search and frequency-mask calculations use FP32 or integers; remaining FP64 arithmetic is confined to CUDA's standard trigonometric large-argument fallback, with output conversions preserving the existing double ABI.

Source and binary hashes, exact commands, numerical comparisons, portable/CUDA test logs, sanitizer logs and precision disassembly evidence live under `cluster-validation/gpu-bfgs-fp32` in the same cluster validation project. `analyze_comparison.py` and `comparison.json` reproduce the matched analysis; the final run is `real-data/apoferritin-ts1/gpu_bfgs_fp32_distance80_checked`, and the retained GN control is `gpu_gn_control_distance80`. Both GPU optimizers remain selectable.

Scientific validation remains incomplete. Extend these comparisons with injections containing template mismatch in real backgrounds, proposal-recall measurements, credible target-absent controls, and per-tilt score covariance checks. In particular, inspect usable-tilt counts: coarse position visibility does not guarantee a full CTF-padded refinement patch fits in the same tilts. The first real-data comparison retained several edge candidates with limited patch coverage.

## Legacy removal after validation

The maintained product is WarpTools; desktop Warp is no longer developed. The intended endpoint is one maintained tilt-series matching implementation used by WarpTools and its workers. Once the new native path passes GPU tests and representative data validation, remove the obsolete implementation and update the supported call paths as part of the same change. Desktop callers do not create compatibility or GUI migration requirements.

- The standalone workers already call `MatchLargeVolume`. Audit the old `TomoMatch` dispatch in `WarpLib/WarpWorker.cs` for reachability from maintained WarpTools workflows; remove unused dispatch code or adapt it only if a supported route needs it. Keep the shared library build consistent when deleting `TiltSeries.MatchFull`. The obsolete desktop dialog is outside this migration's scope.
- Move the still-used `ProcessingOptionsTomoFullMatch` and `ParticlePeak` declarations out of `TiltSeries.MatchFull.cs`, then delete that file's old subvolume search, central-difference refinement and random-position score normalization. Preserve or explicitly migrate serialized settings and worker payloads when changing option types.
- Make the validated pipeline the normal tilt-series workflow. Consolidate K=1 and K>1 searches through the same top-K implementation. Any useful proposal-only diagnostic should use this implementation too. Retire obsolete result-reuse branches and flags; a future cache should store the small candidate/start lists and the metadata needed to validate them.
- Remove unused tilt-series options and their help text, validation and settings assignments. In particular, `--whiten` and `--subvolume_size` currently have no effect in `MatchLargeVolume`; the old fixed whitening option is distinct from the new refinement's estimated background spectra. Audit `TemplateFraction`, `Supersample` and `OverwriteFiles` on the tilt-series option type as well. Check shared movie-matching settings against maintained WarpTools workflows.
- Remove native exports, managed bindings and kernels after tracing reachability from supported workflows. `CorrelateSubTomos` and `d_PickSubTomograms` also have a caller in `Movie.MatchFull`; determine whether that route is maintained before retaining or deleting them. A reference solely from obsolete desktop code is not a retention requirement. The filename `SubTomograms.cu` also contains the new top-K implementation. Shared projection, CTF, FFT and peak-finding primitives used by the new method remain dependencies.
- The native `CorrelateLargeVolume`/`d_PickLargeVolume` functions are already thin K=1 aliases of the top-K implementation. Their eventual removal is API cleanup, not deletion of a separate search algorithm. The old finite-difference refiner itself lives in C# and has no exclusive native gradient kernel to remove.
- Remove the orphan `gtom/src/Correlation/TomoPicker.cu` implementation, its class declaration and build entries after a final caller check. Its initialization already throws and no repository callers were found; it is separate from the old managed `TiltSeries.MatchFull` path.
- Delete obsolete reference-only tests when their legacy entry points disappear, retaining independent CPU mathematical oracles and tests of the new search/refinement. Build WarpTools, its supported workers and shared libraries, check settings round trips, and rerun the CUDA tests and representative series after removal. Desktop builds are outside the validation scope.

The retired per-start managed BFGS implementation is removed. Both GPU BFGS and GPU Gauss–Newton remain available for comparison; Gauss–Newton is deliberately retained. The GPU-resident optimizer passes CUDA integration and real-data execution checks; broad removal of the remaining legacy subvolume implementation is pending resolution of the local-optimum differences and coverage review described above. The audit above does not claim the legacy callers have already been migrated.
