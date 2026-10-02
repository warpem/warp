#ifndef WARP_EINSPLINE_GRADIENT_H
#define WARP_EINSPLINE_GRADIENT_H

#include "../src/einspline/bspline_base.h"
#include "../src/einspline/bspline_structs.h"
#include <cmath>
#include <cstddef>

namespace warp_einspline
{
    // Uses the same cubic basis and end-cell extrapolation as
    // einspline/bspline_eval_std_s.h, differentiating the polynomial directly.
    inline void Basis(const Ugrid& grid, float position, int& index,
                      float weights[4], float derivatives[4])
    {
        static const float basis[16] =
        {
            -1.0f / 6,  3.0f / 6, -3.0f / 6, 1.0f / 6,
             3.0f / 6, -6.0f / 6,  0.0f / 6, 4.0f / 6,
            -3.0f / 6,  3.0f / 6,  3.0f / 6, 1.0f / 6,
             1.0f / 6,  0.0f / 6,  0.0f / 6, 0.0f / 6
        };
        const float u = (float)(((double)position - grid.start) * grid.delta_inv);
        float t;
        if (u >= 0 && u < grid.num - 2)
        {
            float integer;
            t = std::modf(u, &integer);
            index = (int)integer;
        }
        else if (u < 0)
        {
            t = u;
            index = 0;
        }
        else
        {
            t = u - grid.num + 2;
            index = grid.num - 2;
        }
        for (int i = 0; i < 4; ++i)
        {
            const float* row = basis + i * 4;
            weights[i] = row[0] * t * t * t + row[1] * t * t + row[2] * t + row[3];
            derivatives[i] = (float)((3 * row[0] * t * t + 2 * row[1] * t + row[2]) * grid.delta_inv);
        }
    }

    // dimensionSet matches Warp.DimensionSets (not a count or axis bitmask).
    // Positions and gradients are always in Warp XYZ order; native einspline
    // storage reverses the active axes to match X-fastest managed arrays.
    inline void Evaluate(void* spline, int dimensionSet, const float position[3],
                         float& value, float gradient[3])
    {
        gradient[0] = gradient[1] = gradient[2] = 0;
        value = 0;
        int axes[3] = {0, 0, 0};
        int ndims = 0;
        Ugrid grids[3];
        ptrdiff_t strides[3] = {0, 0, 0};
        const float* coefs = NULL;
        switch (dimensionSet)
        {
            case 64: // XYZ: native X=Warp Z, Y=Y, Z=X.
            {
                const UBspline_3d_s* s = (UBspline_3d_s*)spline;
                ndims = 3;
                axes[0] = 2; axes[1] = 1; axes[2] = 0;
                grids[0] = s->x_grid; grids[1] = s->y_grid; grids[2] = s->z_grid;
                strides[0] = s->x_stride; strides[1] = s->y_stride; strides[2] = 1;
                coefs = s->coefs;
                break;
            }
            case 8: case 16: case 32: // XY, XZ, YZ.
            {
                const UBspline_2d_s* s = (UBspline_2d_s*)spline;
                ndims = 2;
                axes[0] = dimensionSet == 8 ? 1 : 2;
                axes[1] = dimensionSet == 32 ? 1 : 0;
                grids[0] = s->x_grid; grids[1] = s->y_grid;
                strides[0] = s->x_stride; strides[1] = 1;
                coefs = s->coefs;
                break;
            }
            case 1: case 2: case 4: // X, Y, Z.
            {
                const UBspline_1d_s* s = (UBspline_1d_s*)spline;
                ndims = 1;
                axes[0] = dimensionSet == 1 ? 0 : dimensionSet == 2 ? 1 : 2;
                grids[0] = s->x_grid;
                strides[0] = 1;
                coefs = s->coefs;
                break;
            }
            default:
                return;
        }

        float weights[3][4] = {{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}};
        float derivatives[3][4] = {};
        ptrdiff_t offset = 0;
        for (int a = 0; a < ndims; ++a)
        {
            int index;
            Basis(grids[a], position[axes[a]], index, weights[a], derivatives[a]);
            offset += index * strides[a];
        }

        double total = 0, nativeGradient[3] = {};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < (ndims >= 2 ? 4 : 1); ++j)
                for (int k = 0; k < (ndims >= 3 ? 4 : 1); ++k)
                {
                    const double c = coefs[offset + i * strides[0] + j * strides[1] + k * strides[2]];
                    total += c * weights[0][i] * weights[1][j] * weights[2][k];
                    nativeGradient[0] += c * derivatives[0][i] * weights[1][j] * weights[2][k];
                    nativeGradient[1] += c * weights[0][i] * derivatives[1][j] * weights[2][k];
                    nativeGradient[2] += c * weights[0][i] * weights[1][j] * derivatives[2][k];
                }
        value = (float)total;
        for (int a = 0; a < ndims; ++a)
            gradient[axes[a]] = (float)nativeGradient[a];
    }
}

#endif
