# Tilt Series CTF Estimation

### Contrast transfer function estimation in tilt series

From [*Real-time cryo-electron microscopy data preprocessing with Warp*](https://doi.org/10.1038/s41592-019-0580-y)
> The single micrograph CTF estimation procedure with planar sample geometry described in
the previous section can be used for tilted 2D data collection. However, full tilt
series pose additional challenges for CTF fitting. Mechanical stage instabilities and
imperfect eucentric height setting necessitate additional exposures for tracking and
focusing to correct the stage position between individual tilt images. Thus, the defocus
cannot be assumed to stay constant, or change smoothly over the course of a tilt series.
Each tilt image requires its own defocus value, which can be challenging due to the
small amount of signal available. Even at 120 e- per Å² for an entire series of 60
images, each tilt only has 2 e- per Å² to perform the same estimation as for a 40 e- per
Å² 2D image, while striving to achieve comparable accuracy.
>
>CTF estimation in tilt series has traditionally received less attention than its
equivalent in 2D data, with the most recent publication predating the introduction
of direct electron detectors and phase plates. As the resolution obtainable through
sub-tomogram averaging has come close to parity with 2D data since then,
simplifying assumptions such as the neglect of astigmatism or the assumed flatness
of the sample can limit the resolution. Combined with the lack of integration of
dedicated tilt series CTF estimation tools into common sub-tomogram averaging
pipelines, this has created a situation in which state-of-the-art studies
employ tools designed for 2D data, such as CTFFIND.
>
>To improve the fit accuracy, the individual tilt image fits must be subjected to a
common set of constraints. As the imaged sample content does not change significantly
throughout the tilt series, 1D background and envelope can be derived from the average
1D spectrum of all tilt images. The relative tilt angles and the tilt axis orientation
are known to a higher precision than could be derived from fitting a planar geometry de
novo, and are kept constant throughout the optimization, as suggested previously.
However, the absolute inclination of the sample plane is unknown. This is especially
critical in some of the typical applications of tomography, like the imaging of lamellae
prepared through focused ion beam (FIB) milling because a lamella can be tilted by over
20° relative to the grid. This additional inclination remains constant throughout the
tilt series, and is made a single optimizable parameter for all tilt images. Astigmatism
and, optionally, phase shift can be kept constant throughout 2D image exposures, but can
benefit from a temporally resolved model in a tilt series where the overall exposure is
fractionated over a much longer time, for example 20–30 min. Warp uses these three
control points in the temporal dimension to model these parameters.
>
>With these considerations, the full estimation process is as follows. 2D patches are
extracted from all aligned tilt movie averages, as described in the micrograph CTF
fitting procedure, and treated in parallel in all subsequent calculations. To provide a
better initialization for the per-tilt defocus searches, an estimate for the average
defocus of the entire series is obtained by prepping 1D spectra from all patches, and
comparing them to simulated CTF curves for the defocus values at the respective
positions and tilts, taking into account the fixed relative tilt information and the
currently tested average defocus (and phase shift, optionally). This search is performed
exhaustively over a range of values specified by the user. The result is used as the
starting point of a more complex optimization. Defocus values for all individual tilts,
three astigmatism magnitude–angle pairs, three optional phase shift values, and the two
global inclination angles (that is the plane normal) are optimized using the L-BFGS
algorithm with the derivatives obtained numerically as described in the micrograph CTF
fitting section. Upon convergence, the 1D spectra of all patches are rescaled to a
common defocus value. This is especially useful for validation in tilt series since the
individual images will have very noisy spectra. If the useful resolution range does not
extend sufficiently beyond the fitting range, the latter is automatically decreased and
the optimization repeated.
>
>In our experience, the direction of the tilt axis is often miscalibrated. Correct
handedness in structures obtained from sub-tomogram averaging does not guarantee the
tilt angle sign is not flipped. In Warp, a positive rotation around the positive Y image
axis is defined to result in an increased underfocus at positions to the positive X side
of the tilt axis, that is those parts of the sample comically closer to the electron
beam source. The CTF fitting procedure in Warp can detect such mistakes by optionally
repeating the optimization with the tilt angles flipped, and notifying the user if the '
wrong' set of angles provides a better fit. Such a test can be used to re-calibrate the
acquisition software for future data collection.
