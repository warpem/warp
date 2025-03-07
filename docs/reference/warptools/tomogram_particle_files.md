# Tomogram Particle File Reference

*WarpTools* uses RELION's particle STAR metadata format.
This format has changed over time, making users unsure exactly what needs to be in files 
they provide for particle extraction when using external picking tools.

This page provides examples of metadata that can be used as input to 
[`WarpTools ts_export_particles`](./api/tilt_series.md#ts_export_particles).

- the list is currently non-exhaustive
- you can use https://teamtomo.org/starfile/ to read/write these files in Python

## RELION 3.0 - single STAR file

```txt
data_

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnCoordinateZ #3
_rlnAngleRot #4
_rlnAngleTilt #5
_rlnAnglePsi #6
_rlnMicrographName #7
443.701994	214.586768	203.739618	54.077932	-29.726951	92.706063	TS_01.tomostar
216.467133	171.667023	157.704667	-168.434232	40.321521	156.800994	TS_01.tomostar
348.701000	91.300066	102.117338	118.892984	-1.043545	127.385870	TS_01.tomostar
465.590176	324.308040	70.420590	105.319824	-37.039451	34.633991	TS_01.tomostar
258.605650	172.224427	97.064928	-74.303821	-3.118226	150.759924	TS_01.tomostar
374.679231	423.019164	205.292795	-130.204570	27.222522	29.593324	TS_02.tomostar
221.453592	425.672254	45.511323	41.711155	48.543276	-82.870213	TS_02.tomostar
348.248079	60.036645	137.823081	123.710065	-86.671079	-34.849912	TS_02.tomostar
229.928165	480.004330	113.754747	-100.103965	165.027964	-76.947619	TS_02.tomostar
385.706010	30.954625	200.434299	26.881529	41.836412	-108.699923	TS_02.tomostar
```

- coordinates (required) are interpreted as pixel coordinates in a tomogram, starting from zero at the center of the first voxel
- rlnMicrographName (optional) should match the tomostar file name for your Warp tilt series processing
- euler angles (optional) are interpreted as [RELION format (ZYZ intrinsic) rotations](https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html#orientations)

