# Spatiotemporal Models

From [Real-time cryo-electron microscopy data preprocessing with Warp](https://doi.org/10.1038/s41592-019-0580-y)
> Many methods in Warp are based on a continuous parametrization of 1- to 3-dimensional
> spaces.
> This parameterization is achieved by spline interpolation between points on a coarse,
> uniform grid, which is computationally efficient. A grid extends over the entirety
> of each dimension that needs to be modeled. The grid resolution is defined by the
> number of control points in each dimension and is scaled according to physical
> constraints (for example, the number of frames or pixels) and available signal.
> The latter provides regularization to prevent overfitting of sparse data with too
> many parameters. When a parameter described by the grid is retrieved for a point
> in space (and time), for example for a particle (frame), B-spline interpolation
> is performed at that point on the grid. To fit a grid’s parameters, in general, a
> cost function associated with the interpolants at specifc positions on the grid is
> optimized. In the following, we distinguish between 2–3 spatial grid dimensions
(X and Y axes in micrographs; X, Y and Z in tomographic volumes), and a
> temporal dimension as a function of the accumulated electron dose. We note
> that B-splines are only used to interpolate parameters, not image data. For the
> latter higher-order schemes are used.
> 
> 