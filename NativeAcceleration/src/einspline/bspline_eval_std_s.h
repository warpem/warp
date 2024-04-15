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

#ifndef BSPLINE_EVAL_STD_S_H
#define BSPLINE_EVAL_STD_S_H

#include <math.h>
#include <stdio.h>


//////////////////////
// Single precision //
//////////////////////
const float A44f[16] =
{ -1.0 / 6.0,  3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0,
3.0 / 6.0, -6.0 / 6.0,  0.0 / 6.0, 4.0 / 6.0,
-3.0 / 6.0,  3.0 / 6.0,  3.0 / 6.0, 1.0 / 6.0,
1.0 / 6.0,  0.0 / 6.0,  0.0 / 6.0, 0.0 / 6.0 };
const float* restrict Af = A44f;

inline void split_fraction(float u, int num, float& t, int& i)
{
	if (u >= 0 && u < num - 2)
	{
		float ipart;
		t = modff(u, &ipart);
		i = (int)ipart;
	}
	else if (u < 0)
	{
		t = u;
		i = 0;
	}
	else
	{
		t = u - num + 2;
		i = num - 2;
	}
}

/************************************************************/
/* 1D single-precision, real evaulation functions           */
/************************************************************/

/* Value only */
inline void
eval_UBspline_1d_s(UBspline_1d_s* restrict spline,
	double x, float* restrict val)
{
	x -= spline->x_grid.start;
	float u = x * spline->x_grid.delta_inv;
	float t;
	int i;
	split_fraction(u, spline->x_grid.num, t, i);

	float tp[4];
	tp[0] = t * t * t;  tp[1] = t * t;  tp[2] = t;  tp[3] = 1.0;
	float* restrict coefs = spline->coefs;

	*val =
		(coefs[i + 0] * (Af[0] * tp[0] + Af[1] * tp[1] + Af[2] * tp[2] + Af[3] * tp[3]) +
			coefs[i + 1] * (Af[4] * tp[0] + Af[5] * tp[1] + Af[6] * tp[2] + Af[7] * tp[3]) +
			coefs[i + 2] * (Af[8] * tp[0] + Af[9] * tp[1] + Af[10] * tp[2] + Af[11] * tp[3]) +
			coefs[i + 3] * (Af[12] * tp[0] + Af[13] * tp[1] + Af[14] * tp[2] + Af[15] * tp[3]));
}

/************************************************************/
/* 2D single-precision, real evaulation functions           */
/************************************************************/

/* Value only */
inline void
eval_UBspline_2d_s(UBspline_2d_s* restrict spline,
	double x, double y, float* restrict val)
{
	x -= spline->x_grid.start;
	y -= spline->y_grid.start;
	float ux = x * spline->x_grid.delta_inv;
	float uy = y * spline->y_grid.delta_inv;
	float tx, ty;
	int ix, iy;
	split_fraction(ux, spline->x_grid.num, tx, ix);
	split_fraction(uy, spline->y_grid.num, ty, iy);

	float tpx[4], tpy[4], a[4], b[4];
	tpx[0] = tx * tx * tx;  tpx[1] = tx * tx;  tpx[2] = tx;  tpx[3] = 1.0;
	tpy[0] = ty * ty * ty;  tpy[1] = ty * ty;  tpy[2] = ty;  tpy[3] = 1.0;
	float* restrict coefs = spline->coefs;

	a[0] = (Af[0] * tpx[0] + Af[1] * tpx[1] + Af[2] * tpx[2] + Af[3] * tpx[3]);
	a[1] = (Af[4] * tpx[0] + Af[5] * tpx[1] + Af[6] * tpx[2] + Af[7] * tpx[3]);
	a[2] = (Af[8] * tpx[0] + Af[9] * tpx[1] + Af[10] * tpx[2] + Af[11] * tpx[3]);
	a[3] = (Af[12] * tpx[0] + Af[13] * tpx[1] + Af[14] * tpx[2] + Af[15] * tpx[3]);

	b[0] = (Af[0] * tpy[0] + Af[1] * tpy[1] + Af[2] * tpy[2] + Af[3] * tpy[3]);
	b[1] = (Af[4] * tpy[0] + Af[5] * tpy[1] + Af[6] * tpy[2] + Af[7] * tpy[3]);
	b[2] = (Af[8] * tpy[0] + Af[9] * tpy[1] + Af[10] * tpy[2] + Af[11] * tpy[3]);
	b[3] = (Af[12] * tpy[0] + Af[13] * tpy[1] + Af[14] * tpy[2] + Af[15] * tpy[3]);

	int xs = spline->x_stride;
#define C(i,j) coefs[(ix+(i))*xs+iy+(j)]
	* val = (a[0] * (C(0, 0) * b[0] + C(0, 1) * b[1] + C(0, 2) * b[2] + C(0, 3) * b[3]) +
		a[1] * (C(1, 0) * b[0] + C(1, 1) * b[1] + C(1, 2) * b[2] + C(1, 3) * b[3]) +
		a[2] * (C(2, 0) * b[0] + C(2, 1) * b[1] + C(2, 2) * b[2] + C(2, 3) * b[3]) +
		a[3] * (C(3, 0) * b[0] + C(3, 1) * b[1] + C(3, 2) * b[2] + C(3, 3) * b[3]));
#undef C

}


/************************************************************/
/* 3D single-precision, real evaulation functions           */
/************************************************************/

/* Value only */
inline void
eval_UBspline_3d_s(UBspline_3d_s* restrict spline,
	double x, double y, double z,
	float* restrict val)
{
	x -= spline->x_grid.start;
	y -= spline->y_grid.start;
	z -= spline->z_grid.start;
	float ux = x * spline->x_grid.delta_inv;
	float uy = y * spline->y_grid.delta_inv;
	float uz = z * spline->z_grid.delta_inv;
	float tx, ty, tz;
	int ix, iy, iz;
	split_fraction(ux, spline->x_grid.num, tx, ix);
	split_fraction(uy, spline->y_grid.num, ty, iy);
	split_fraction(uz, spline->z_grid.num, tz, iz);


	float tpx[4], tpy[4], tpz[4], a[4], b[4], c[4];
	tpx[0] = tx * tx * tx;  tpx[1] = tx * tx;  tpx[2] = tx;  tpx[3] = 1.0;
	tpy[0] = ty * ty * ty;  tpy[1] = ty * ty;  tpy[2] = ty;  tpy[3] = 1.0;
	tpz[0] = tz * tz * tz;  tpz[1] = tz * tz;  tpz[2] = tz;  tpz[3] = 1.0;
	float* restrict coefs = spline->coefs;

	a[0] = (Af[0] * tpx[0] + Af[1] * tpx[1] + Af[2] * tpx[2] + Af[3] * tpx[3]);
	a[1] = (Af[4] * tpx[0] + Af[5] * tpx[1] + Af[6] * tpx[2] + Af[7] * tpx[3]);
	a[2] = (Af[8] * tpx[0] + Af[9] * tpx[1] + Af[10] * tpx[2] + Af[11] * tpx[3]);
	a[3] = (Af[12] * tpx[0] + Af[13] * tpx[1] + Af[14] * tpx[2] + Af[15] * tpx[3]);

	b[0] = (Af[0] * tpy[0] + Af[1] * tpy[1] + Af[2] * tpy[2] + Af[3] * tpy[3]);
	b[1] = (Af[4] * tpy[0] + Af[5] * tpy[1] + Af[6] * tpy[2] + Af[7] * tpy[3]);
	b[2] = (Af[8] * tpy[0] + Af[9] * tpy[1] + Af[10] * tpy[2] + Af[11] * tpy[3]);
	b[3] = (Af[12] * tpy[0] + Af[13] * tpy[1] + Af[14] * tpy[2] + Af[15] * tpy[3]);

	c[0] = (Af[0] * tpz[0] + Af[1] * tpz[1] + Af[2] * tpz[2] + Af[3] * tpz[3]);
	c[1] = (Af[4] * tpz[0] + Af[5] * tpz[1] + Af[6] * tpz[2] + Af[7] * tpz[3]);
	c[2] = (Af[8] * tpz[0] + Af[9] * tpz[1] + Af[10] * tpz[2] + Af[11] * tpz[3]);
	c[3] = (Af[12] * tpz[0] + Af[13] * tpz[1] + Af[14] * tpz[2] + Af[15] * tpz[3]);

	int xs = spline->x_stride;
	int ys = spline->y_stride;
#define P(i,j,k) coefs[(ix+(i))*xs+(iy+(j))*ys+(iz+(k))]
	* val = (a[0] * (b[0] * (P(0, 0, 0) * c[0] + P(0, 0, 1) * c[1] + P(0, 0, 2) * c[2] + P(0, 0, 3) * c[3]) +
		b[1] * (P(0, 1, 0) * c[0] + P(0, 1, 1) * c[1] + P(0, 1, 2) * c[2] + P(0, 1, 3) * c[3]) +
		b[2] * (P(0, 2, 0) * c[0] + P(0, 2, 1) * c[1] + P(0, 2, 2) * c[2] + P(0, 2, 3) * c[3]) +
		b[3] * (P(0, 3, 0) * c[0] + P(0, 3, 1) * c[1] + P(0, 3, 2) * c[2] + P(0, 3, 3) * c[3])) +
		a[1] * (b[0] * (P(1, 0, 0) * c[0] + P(1, 0, 1) * c[1] + P(1, 0, 2) * c[2] + P(1, 0, 3) * c[3]) +
			b[1] * (P(1, 1, 0) * c[0] + P(1, 1, 1) * c[1] + P(1, 1, 2) * c[2] + P(1, 1, 3) * c[3]) +
			b[2] * (P(1, 2, 0) * c[0] + P(1, 2, 1) * c[1] + P(1, 2, 2) * c[2] + P(1, 2, 3) * c[3]) +
			b[3] * (P(1, 3, 0) * c[0] + P(1, 3, 1) * c[1] + P(1, 3, 2) * c[2] + P(1, 3, 3) * c[3])) +
		a[2] * (b[0] * (P(2, 0, 0) * c[0] + P(2, 0, 1) * c[1] + P(2, 0, 2) * c[2] + P(2, 0, 3) * c[3]) +
			b[1] * (P(2, 1, 0) * c[0] + P(2, 1, 1) * c[1] + P(2, 1, 2) * c[2] + P(2, 1, 3) * c[3]) +
			b[2] * (P(2, 2, 0) * c[0] + P(2, 2, 1) * c[1] + P(2, 2, 2) * c[2] + P(2, 2, 3) * c[3]) +
			b[3] * (P(2, 3, 0) * c[0] + P(2, 3, 1) * c[1] + P(2, 3, 2) * c[2] + P(2, 3, 3) * c[3])) +
		a[3] * (b[0] * (P(3, 0, 0) * c[0] + P(3, 0, 1) * c[1] + P(3, 0, 2) * c[2] + P(3, 0, 3) * c[3]) +
			b[1] * (P(3, 1, 0) * c[0] + P(3, 1, 1) * c[1] + P(3, 1, 2) * c[2] + P(3, 1, 3) * c[3]) +
			b[2] * (P(3, 2, 0) * c[0] + P(3, 2, 1) * c[1] + P(3, 2, 2) * c[2] + P(3, 2, 3) * c[3]) +
			b[3] * (P(3, 3, 0) * c[0] + P(3, 3, 1) * c[1] + P(3, 3, 2) * c[2] + P(3, 3, 3) * c[3])));
#undef P

}

#endif
