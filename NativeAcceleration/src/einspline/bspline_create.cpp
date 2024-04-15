/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Copyright (C) 2007 Kenneth P. Esler, Jr.                               //
//                                                                         //
//  This program is free software; you can redistribute it and/or modify   //
//  it under the terms of the GNU General Public License as published by   //
//  the Free Software Foundation; either version 2 of the License, or      //
//  (at your option) any later version.                                    //
//                                                                         //
//  This program is distributed in the hope that it will be useful,        //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//  GNU General Public License for more details.                           //
//                                                                         //
//  You should have received a copy of the GNU General Public License      //
//  along with this program; if not, write to the Free Software            //
//  Foundation, Inc., 51 Franklin Street, Fifth Floor,                     //
//  Boston, MA  02110-1301  USA                                            //
/////////////////////////////////////////////////////////////////////////////

#include "bspline_create.h"
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif
#ifndef __USE_XOPEN2K
#define __USE_XOPEN2K
#endif
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

//#include "aligned_alloc.h"

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////       Helper functions for spline creation         ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void init_sse_data();

void
solve_deriv_interp_1d_s(float bands[], float coefs[],
    int M, int cstride)
{
    // Solve interpolating equations
    // First and last rows are different
    bands[4 * (0) + 1] /= bands[4 * (0) + 0];
    bands[4 * (0) + 2] /= bands[4 * (0) + 0];
    bands[4 * (0) + 3] /= bands[4 * (0) + 0];
    bands[4 * (0) + 0] = 1.0;
    bands[4 * (1) + 1] -= bands[4 * (1) + 0] * bands[4 * (0) + 1];
    bands[4 * (1) + 2] -= bands[4 * (1) + 0] * bands[4 * (0) + 2];
    bands[4 * (1) + 3] -= bands[4 * (1) + 0] * bands[4 * (0) + 3];
    bands[4 * (0) + 0] = 0.0;
    bands[4 * (1) + 2] /= bands[4 * (1) + 1];
    bands[4 * (1) + 3] /= bands[4 * (1) + 1];
    bands[4 * (1) + 1] = 1.0;

    // Now do rows 2 through M+1
    for (int row = 2; row < (M + 1); row++) {
        bands[4 * (row)+1] -= bands[4 * (row)+0] * bands[4 * (row - 1) + 2];
        bands[4 * (row)+3] -= bands[4 * (row)+0] * bands[4 * (row - 1) + 3];
        bands[4 * (row)+2] /= bands[4 * (row)+1];
        bands[4 * (row)+3] /= bands[4 * (row)+1];
        bands[4 * (row)+0] = 0.0;
        bands[4 * (row)+1] = 1.0;
    }

    // Do last row
    bands[4 * (M + 1) + 1] -= bands[4 * (M + 1) + 0] * bands[4 * (M - 1) + 2];
    bands[4 * (M + 1) + 3] -= bands[4 * (M + 1) + 0] * bands[4 * (M - 1) + 3];
    bands[4 * (M + 1) + 2] -= bands[4 * (M + 1) + 1] * bands[4 * (M)+2];
    bands[4 * (M + 1) + 3] -= bands[4 * (M + 1) + 1] * bands[4 * (M)+3];
    bands[4 * (M + 1) + 3] /= bands[4 * (M + 1) + 2];
    bands[4 * (M + 1) + 2] = 1.0;

    coefs[(M + 1) * cstride] = bands[4 * (M + 1) + 3];
    // Now back substitute up
    for (int row = M; row > 0; row--)
        coefs[row * cstride] = bands[4 * (row)+3] - bands[4 * (row)+2] * coefs[cstride * (row + 1)];

    // Finish with first row
    coefs[0] = bands[4 * (0) + 3] - bands[4 * (0) + 1] * coefs[1 * cstride] - bands[4 * (0) + 2] * coefs[2 * cstride];
}



void find_coefs_1d_s(Ugrid grid, BCtype_s bc,
    float* data, intptr_t dstride,
    float* coefs, intptr_t cstride)
{
    int M = grid.num;
    float basis[4] = { 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0 };
    {
        // Setup boundary conditions
        float abcd_left[4], abcd_right[4];
        // Left boundary
        if (bc.lCode == FLAT || bc.lCode == NATURAL)
            bc.lVal = 0.0;
        if (bc.lCode == FLAT || bc.lCode == DERIV1) {
            abcd_left[0] = -0.5 * grid.delta_inv;
            abcd_left[1] = 0.0 * grid.delta_inv;
            abcd_left[2] = 0.5 * grid.delta_inv;
            abcd_left[3] = bc.lVal;
        }
        if (bc.lCode == NATURAL || bc.lCode == DERIV2) {
            abcd_left[0] = 1.0 * grid.delta_inv * grid.delta_inv;
            abcd_left[1] = -2.0 * grid.delta_inv * grid.delta_inv;
            abcd_left[2] = 1.0 * grid.delta_inv * grid.delta_inv;
            abcd_left[3] = bc.lVal;
        }

        // Right boundary
        if (bc.rCode == FLAT || bc.rCode == NATURAL)
            bc.rVal = 0.0;
        if (bc.rCode == FLAT || bc.rCode == DERIV1) {
            abcd_right[0] = -0.5 * grid.delta_inv;
            abcd_right[1] = 0.0 * grid.delta_inv;
            abcd_right[2] = 0.5 * grid.delta_inv;
            abcd_right[3] = bc.rVal;
        }
        if (bc.rCode == NATURAL || bc.rCode == DERIV2) {
            abcd_right[0] = 1.0 * grid.delta_inv * grid.delta_inv;
            abcd_right[1] = -2.0 * grid.delta_inv * grid.delta_inv;
            abcd_right[2] = 1.0 * grid.delta_inv * grid.delta_inv;
            abcd_right[3] = bc.rVal;
        }
#ifdef HAVE_C_VARARRAYS
        float bands[4 * (M + 2)];
#else
        float* bands = (float*)malloc((M + 2) * 4 * sizeof(float));
#endif    
        for (int i = 0; i < 4; i++) {
            bands[4 * (0) + i] = abcd_left[i];
            bands[4 * (M + 1) + i] = abcd_right[i];
        }

        //#pragma omp parallel for
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < 3; j++)
                bands[4 * (i + 1) + j] = basis[j];
            bands[4 * (i + 1) + 3] = data[i * dstride];
        }
        // Now, solve for coefficients
        solve_deriv_interp_1d_s(bands, coefs, M, cstride);
#ifndef HAVE_C_VARARRAYS
        free(bands);
#endif
    }
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////     Single-Precision, Real Creation Routines       ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
UBspline_1d_s*
create_UBspline_1d_s(Ugrid x_grid, BCtype_s xBC, float* data)
{
    // Create new spline
    UBspline_1d_s* restrict spline = (UBspline_1d_s*)malloc(sizeof(UBspline_1d_s));
    spline->spcode = U1D;
    spline->tcode = SINGLE_REAL;
    spline->xBC = xBC; spline->x_grid = x_grid;

    // Setup internal variables
    int M = x_grid.num;
    int N;

    if (xBC.lCode == PERIODIC || xBC.lCode == ANTIPERIODIC) {
        x_grid.delta = (x_grid.end - x_grid.start) / (double)(x_grid.num);
        N = M + 3;
    }
    else {
        x_grid.delta = (x_grid.end - x_grid.start) / (double)(x_grid.num - 1);
        N = M + 2;
    }

    x_grid.delta_inv = 1.0 / x_grid.delta;
    spline->x_grid = x_grid;
#ifndef HAVE_SSE2
    spline->coefs = (float*)malloc(sizeof(float) * N);
#else
    posix_memalign((void**)&spline->coefs, 16, (sizeof(float) * N));
#endif
    find_coefs_1d_s(spline->x_grid, xBC, data, 1, spline->coefs, 1);

    //init_sse_data();
    return spline;
}

void
recompute_UBspline_1d_s(UBspline_1d_s* spline, float* data)
{
    find_coefs_1d_s(spline->x_grid, spline->xBC, data, 1, spline->coefs, 1);
}


UBspline_2d_s*
create_UBspline_2d_s(Ugrid x_grid, Ugrid y_grid,
    BCtype_s xBC, BCtype_s yBC, float* data)
{
    // Create new spline
    UBspline_2d_s* restrict spline = (UBspline_2d_s*)malloc(sizeof(UBspline_2d_s));
    spline->spcode = U2D;
    spline->tcode = SINGLE_REAL;
    spline->xBC = xBC;
    spline->yBC = yBC;
    // Setup internal variables
    int Mx = x_grid.num;
    int My = y_grid.num;
    int Nx, Ny;

    if (xBC.lCode == PERIODIC || xBC.lCode == ANTIPERIODIC)     Nx = Mx + 3;
    else                           Nx = Mx + 2;
    x_grid.delta = (x_grid.end - x_grid.start) / (double)(Nx - 3);
    x_grid.delta_inv = 1.0 / x_grid.delta;
    spline->x_grid = x_grid;

    if (yBC.lCode == PERIODIC || yBC.lCode == ANTIPERIODIC)     Ny = My + 3;
    else                           Ny = My + 2;
    y_grid.delta = (y_grid.end - y_grid.start) / (double)(Ny - 3);
    y_grid.delta_inv = 1.0 / y_grid.delta;
    spline->y_grid = y_grid;
    spline->x_stride = Ny;
#ifndef HAVE_SSE2
    spline->coefs = (float*)malloc(sizeof(float) * Nx * Ny);
#else
    posix_memalign((void**)&spline->coefs, 16, sizeof(float) * Nx * Ny);
#endif

    // First, solve in the X-direction 
    for (int iy = 0; iy < My; iy++) {
        intptr_t doffset = iy;
        intptr_t coffset = iy;
        find_coefs_1d_s(spline->x_grid, spline->xBC, data + doffset, My,
            spline->coefs + coffset, Ny);
    }

    // Now, solve in the Y-direction
    for (int ix = 0; ix < Nx; ix++) {
        intptr_t doffset = ix * Ny;
        intptr_t coffset = ix * Ny;
        find_coefs_1d_s(spline->y_grid, spline->yBC, spline->coefs + doffset, 1,
            spline->coefs + coffset, 1);
    }

    //init_sse_data();
    return spline;
}

void
recompute_UBspline_2d_s(UBspline_2d_s* spline, float* data)
{
    int Mx = spline->x_grid.num;
    int My = spline->y_grid.num;
    int Nx, Ny;

    if (spline->xBC.lCode == PERIODIC || spline->xBC.lCode == ANTIPERIODIC)     Nx = Mx + 3;
    else                           Nx = Mx + 2;
    if (spline->yBC.lCode == PERIODIC || spline->yBC.lCode == ANTIPERIODIC)     Ny = My + 3;
    else                           Ny = My + 2;

    // First, solve in the X-direction 
  ///#pragma omp parallel for
    for (int iy = 0; iy < My; iy++) {
        intptr_t doffset = iy;
        intptr_t coffset = iy;
        find_coefs_1d_s(spline->x_grid, spline->xBC, data + doffset, My,
            spline->coefs + coffset, Ny);
    }

    // Now, solve in the Y-direction
  ///#pragma omp parallel for
    for (int ix = 0; ix < Nx; ix++) {
        intptr_t doffset = ix * Ny;
        intptr_t coffset = ix * Ny;
        find_coefs_1d_s(spline->y_grid, spline->yBC, spline->coefs + doffset, 1,
            spline->coefs + coffset, 1);
    }
}


UBspline_3d_s*
create_UBspline_3d_s(Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
    BCtype_s xBC, BCtype_s yBC, BCtype_s zBC,
    float* data)
{
    // Create new spline
    UBspline_3d_s* restrict spline = (UBspline_3d_s*)malloc(sizeof(UBspline_3d_s));
    spline->spcode = U3D;
    spline->tcode = SINGLE_REAL;
    spline->xBC = xBC;
    spline->yBC = yBC;
    spline->zBC = zBC;
    // Setup internal variables
    int Mx = x_grid.num;  int My = y_grid.num; int Mz = z_grid.num;
    int Nx, Ny, Nz;

    if (xBC.lCode == PERIODIC || xBC.lCode == ANTIPERIODIC)     Nx = Mx + 3;
    else                           Nx = Mx + 2;
    x_grid.delta = (x_grid.end - x_grid.start) / (double)(Nx - 3);
    x_grid.delta_inv = 1.0 / x_grid.delta;
    spline->x_grid = x_grid;

    if (yBC.lCode == PERIODIC || yBC.lCode == ANTIPERIODIC)     Ny = My + 3;
    else                           Ny = My + 2;
    y_grid.delta = (y_grid.end - y_grid.start) / (double)(Ny - 3);
    y_grid.delta_inv = 1.0 / y_grid.delta;
    spline->y_grid = y_grid;

    if (zBC.lCode == PERIODIC || zBC.lCode == ANTIPERIODIC)     Nz = Mz + 3;
    else                           Nz = Mz + 2;
    z_grid.delta = (z_grid.end - z_grid.start) / (double)(Nz - 3);
    z_grid.delta_inv = 1.0 / z_grid.delta;
    spline->z_grid = z_grid;

    spline->x_stride = Ny * Nz;
    spline->y_stride = Nz;

#ifndef HAVE_SSE2
    spline->coefs = (float*)malloc(sizeof(float) * Nx * Ny * Nz);
#else
    posix_memalign((void**)&spline->coefs, 16, (sizeof(float) * Nx * Ny * Nz));
#endif

    // First, solve in the X-direction 
  ///#pragma omp parallel for
    for (int iy = 0; iy < My; iy++)
        for (int iz = 0; iz < Mz; iz++) {
            intptr_t doffset = iy * Mz + iz;
            intptr_t coffset = iy * Nz + iz;
            find_coefs_1d_s(spline->x_grid, xBC, data + doffset, My * Mz,
                spline->coefs + coffset, Ny * Nz);
        }

    // Now, solve in the Y-direction
  ///#pragma omp parallel for
    for (int ix = 0; ix < Nx; ix++)
        for (int iz = 0; iz < Nz; iz++) {
            intptr_t doffset = ix * Ny * Nz + iz;
            intptr_t coffset = ix * Ny * Nz + iz;
            find_coefs_1d_s(spline->y_grid, yBC, spline->coefs + doffset, Nz,
                spline->coefs + coffset, Nz);
        }

    // Now, solve in the Z-direction
  ///#pragma omp parallel for
    for (int ix = 0; ix < Nx; ix++)
        for (int iy = 0; iy < Ny; iy++) {
            intptr_t doffset = (ix * Ny + iy) * Nz;
            intptr_t coffset = (ix * Ny + iy) * Nz;
            find_coefs_1d_s(spline->z_grid, zBC, spline->coefs + doffset, 1,
                spline->coefs + coffset, 1);
        }

    //init_sse_data();
    return spline;
}

void
recompute_UBspline_3d_s(UBspline_3d_s* spline, float* data)
{
    int Mx = spline->x_grid.num;
    int My = spline->y_grid.num;
    int Mz = spline->z_grid.num;
    int Nx, Ny, Nz;

    if (spline->xBC.lCode == PERIODIC || spline->xBC.lCode == ANTIPERIODIC)     Nx = Mx + 3;
    else                           Nx = Mx + 2;
    if (spline->yBC.lCode == PERIODIC || spline->yBC.lCode == ANTIPERIODIC)     Ny = My + 3;
    else                           Ny = My + 2;
    if (spline->zBC.lCode == PERIODIC || spline->zBC.lCode == ANTIPERIODIC)     Nz = Mz + 3;
    else                           Nz = Mz + 2;

    // First, solve in the X-direction 
    for (int iy = 0; iy < My; iy++)
        for (int iz = 0; iz < Mz; iz++) {
            intptr_t doffset = iy * Mz + iz;
            intptr_t coffset = iy * Nz + iz;
            find_coefs_1d_s(spline->x_grid, spline->xBC, data + doffset, My * Mz,
                spline->coefs + coffset, Ny * Nz);
        }

    // Now, solve in the Y-direction
    for (int ix = 0; ix < Nx; ix++)
        for (int iz = 0; iz < Nz; iz++) {
            intptr_t doffset = ix * Ny * Nz + iz;
            intptr_t coffset = ix * Ny * Nz + iz;
            find_coefs_1d_s(spline->y_grid, spline->yBC, spline->coefs + doffset, Nz,
                spline->coefs + coffset, Nz);
        }

    // Now, solve in the Z-direction
    for (int ix = 0; ix < Nx; ix++)
        for (int iy = 0; iy < Ny; iy++) {
            intptr_t doffset = (ix * Ny + iy) * Nz;
            intptr_t coffset = (ix * Ny + iy) * Nz;
            find_coefs_1d_s(spline->z_grid, spline->zBC, spline->coefs + doffset, 1,
                spline->coefs + coffset, 1);
        }
}


void
destroy_UBspline(Bspline* spline)
{
    free(spline->coefs);
    free(spline);
}

void
destroy_Bspline(void* spline)
{
    Bspline* sp = (Bspline*)spline;
    if (sp->sp_code <= U3D)
        destroy_UBspline(sp);
    else
        fprintf(stderr, "Error in destroy_Bspline:  invalid spline code %d.\n",
            sp->sp_code);
}
