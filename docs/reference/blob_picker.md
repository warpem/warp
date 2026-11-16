# Micrograph blob picker (WarpWorker2 only)

Queue a worker command with:

```csharp
WorkerCommands.MicrographPickBlobs(
    path: "/data/micrographs/mic.mrc",
    radius: 30f,
    innerRadius: 15f,           // optional; 0 for a disk
    minimumSeparation: 60f);    // optional; defaults to 2 * radius
```

The wire command is `MicrographPickBlobs`, with content `(string path,
float radius, float innerRadius, float minimumSeparation)`. The handler also
accepts omitted trailing arguments. It loads the micrograph directly, so no
`LoadStack` init command is needed. There is no WarpTools verb for this command.

All distances are in **input micrograph pixels**. The input must be a real,
finite, single-plane micrograph. Radius must be positive, inner radius must be
in `[0, radius)`, and separation must be positive. A ring must contain at least
one sampled pixel. Intensities should have dark particles on a lighter background.

Output: `/data/micrographs/matching/mic_blob.star`, with columns:

| Column | Meaning |
| --- | --- |
| `rlnCoordinateX`, `rlnCoordinateY` | Zero-based input pixel coordinates, following Warp's existing picking convention |
| `rlnMicrographName` | Input micrograph filename |
| `rlnAutopickFigureOfMerit` | Pearson correlation with the dark blob template, in `(0, 1]` |
| `wrpAutopickPower` | Local variance (mean-subtracted mean-square intensity), in input intensity squared |

The template lives in a circular support of radius `2 * radius`. For a disk
(`innerRadius` 0) it is a dark difference of Gaussians, with a centre sigma of
`radius / 2` against a surround sigma of `radius`: a flat disk would score a
particle equally wherever it fits inside, so its peak would wander, while this
one peaks on the particle's dark mass. For a ring it is -1 within the annulus
and 0 elsewhere, so pixels in the hole participate in the background. Either
way, its mean over the support is subtracted. Correlation normalizes by the
local variance and template energy; power is measured over the same full
support. Positive intensity scaling preserves correlation and scales power by
the square of the intensity scale. Constant offsets affect neither statistic.

The micrograph is mirrored past its edges, so centres down to one radius from
the border are scored and FFT wraparound cannot contribute; picks closer than
one radius to an edge are dropped, since the particle disk must lie inside and
its extraction box would leave the micrograph anyway. CUDA builds the template,
computes correlations and local moments with cuFFT, and finds positive local
maxima within `radius / 2` (ties resolved by the lower linear pixel index); only
those are copied to the host. There is no further threshold, and the STAR
exposes both statistics for downstream filtering.

Suppression then compares picks with picks: strongest first, a pick is dropped
if a stronger kept pick lies closer than the separation. A weaker particle more
than the separation from a stronger one therefore survives even where the
stronger one's slope outscores it. Picks are written strongest first.
`BlobPicker.Revision` changes whenever the picks for the same image change.

An empty result writes an empty STAR with all columns, replacing any previous
picks. GPU failures or invalid input fail the queue item rather than writing
an apparently successful empty result. Rebuild NativeAcceleration and the
managed worker together before using the command.

## Validation

Managed validation, queue serialization and STAR round-trip tests:

```sh
dotnet test Tests/Tests.csproj --filter 'FullyQualifiedName~BlobPickerTests'
```

The managed CUDA smoke test is enabled with `WARP_RUN_CUDA_TESTS=1`.
The standalone CUDA suite tests the scoring kernel without the managed
mirroring: it compares every valid pixel against a direct spatial reference on
disk and ring images, and checks correlation/power scaling, border exclusion,
flat/nonfinite input and fractional Euclidean separation of local maxima:

```sh
cd NativeAcceleration
nvcc -std=c++17 -arch=sm_89 -I. src/BlobPicker.cu tests/BlobPickerTests.cu -lcufft -o /tmp/blob_picker_tests
/tmp/blob_picker_tests
```

## Relation to RELION LoG

[RELION's LoG picker](https://github.com/3dem/relion/blob/master/src/autopicker.cpp)
searches multiple blob scales, including scales outside the requested diameter
range for rejecting objects of the wrong size. That is useful when a single
radius does not describe the particles. This command instead supplies one
difference of Gaussians or an explicit annulus at one radius, with a locally normalized correlation and
separate intensity power. A ring template can represent a prescribed hole that
a standard LoG filter cannot parameterize independently. These are algorithmic
differences, not evidence that either picker is better on experimental data;
that requires comparison on representative micrographs.
