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

#ifndef BSPLINE_EVAL_SSE_S_H
#define BSPLINE_EVAL_SSE_S_H

#include <xmmintrin.h>
#include <emmintrin.h>
#ifdef HAVE_SSE3
#include <pmmintrin.h>
#endif
#include <stdio.h>
#include <math.h>


// extern __m128   A0,   A1,   A2,   A3;
// extern __m128  dA0,  dA1,  dA2,  dA3;
// extern __m128 d2A0, d2A1, d2A2, d2A3;
extern __m128* restrict A_s;

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

/// SSE3 add "horizontal add" instructions, which makes things
/// simpler and faster
#ifdef HAVE_SSE9
#define _MM_MATVEC4_PS(M0, M1, M2, M3, v, r)                        \
do {                                                                \
  __m128 r0 = _mm_hadd_ps (_mm_mul_ps (M0, v), _mm_mul_ps (M1, v)); \
  __m128 r1 = _mm_hadd_ps (_mm_mul_ps (M2, v), _mm_mul_ps (M3, v)); \
  r = _mm_hadd_ps (r0, r1);                                         \
 } while (0);
#define _MM_DOT4_PS(A, B, _p)                                       \
do {                                                                \
  __m128 t  = _mm_mul_ps (A, B);                                    \
  __m128 t1 = _mm_hadd_ps (t,t);                                    \
  __m128 r  = _mm_hadd_ps (t1, t1);                                 \
  _mm_store_ss (&(_p), r);                                          \
} while(0);
#else
// Use plain-old SSE instructions
#define _MM_MATVEC4_PS(M0, M1, M2, M3, v, r)                        \
do {                                                                \
  __m128 _r0 = _mm_mul_ps (M0, v);                                  \
  __m128 _r1 = _mm_mul_ps (M1, v);				    \
  __m128 _r2 = _mm_mul_ps (M2, v);                                  \
  __m128 _r3 = _mm_mul_ps (M3, v);				    \
  _MM_TRANSPOSE4_PS (_r0, _r1, _r2, _r3);                           \
  r = _mm_add_ps (_mm_add_ps (_r0, _r1), _mm_add_ps (_r2, _r3));    \
 } while (0);
#define _MM_DOT4_PS(A, B, p)                                        \
do {                                                                \
  __m128 t    = _mm_mul_ps (A, B);                                  \
  __m128 alo  = _mm_shuffle_ps (t, t, _MM_SHUFFLE(0,1,0,1));	    \
  __m128 ahi  = _mm_shuffle_ps (t, t, _MM_SHUFFLE(2,3,2,3));	    \
  __m128 _a    = _mm_add_ps (alo, ahi);                              \
  __m128 rlo  = _mm_shuffle_ps (_a, _a, _MM_SHUFFLE(0,0,0,0));	    \
  __m128 rhi  = _mm_shuffle_ps (_a, _a, _MM_SHUFFLE(1,1,1,1));	    \
  __m128 _r   = _mm_add_ps (rlo, rhi);                              \
  _mm_store_ss (&(p), _r);                                          \
} while(0);
#endif


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
eval_UBspline_2d_s (UBspline_2d_s * restrict spline, 
		    double x, double y, float* restrict val)
{
  _mm_prefetch ((const char*)  &A_s[0],_MM_HINT_T0);  _mm_prefetch ((const char*)  &A_s[1],_MM_HINT_T0);  
  _mm_prefetch ((const char*)  &A_s[2],_MM_HINT_T0);  _mm_prefetch ((const char*)  &A_s[3],_MM_HINT_T0);
  /// SSE mesh point determination
  __m128 xy        = _mm_set_ps (x, y, 0.0, 0.0);
  __m128 x0y0      = _mm_set_ps (spline->x_grid.start,  spline->y_grid.start, 0.0, 0.0);
  __m128 delta_inv = _mm_set_ps (spline->x_grid.delta_inv,spline->y_grid.delta_inv, 0.0, 0.0);
  xy = _mm_sub_ps (xy, x0y0);
  // ux = (x - x0)/delta_x and same for y
  __m128 uxuy    = _mm_mul_ps (xy, delta_inv);
  // intpart = trunc (ux, uy)
  __m128i intpart  = _mm_cvttps_epi32(uxuy);
  __m128i ixiy;
  _mm_storeu_si128 (&ixiy, intpart);
  // Store to memory for use in C expressions
  // xmm registers are stored to memory in reverse order
  int ix = ((int *)&ixiy)[3];
  int iy = ((int *)&ixiy)[2];

  int xs = spline->x_stride;
  // This macro is used to give the pointer to coefficient data.
  // i and j should be in the range [0,3].  Coefficients are read four
  // at a time, so no j value is needed.
#define P(i) (spline->coefs+(ix+(i))*xs+iy)
  // Prefetch the data from main memory into cache so it's available
  // when we need to use it.
  _mm_prefetch ((const char*)P(0), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(1), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(3), _MM_HINT_T0);

  // Now compute the vectors:
  // tpx = [t_x^3 t_x^2 t_x 1]
  // tpy = [t_y^3 t_y^2 t_y 1]
  // tpz = [t_z^3 t_z^2 t_z 1]
  __m128 ipart  = _mm_cvtepi32_ps (intpart);
  __m128 txty   = _mm_sub_ps (uxuy, ipart);
  __m128 one    = _mm_set_ps (1.0, 1.0, 1.0, 1.0);
  __m128 t2     = _mm_mul_ps (txty, txty);
  __m128 t3     = _mm_mul_ps (t2, txty);
  __m128 tpx    = t3;
  __m128 tpy    = t2;
  __m128 tpz    = txty;
  __m128 zero   = one;
  _MM_TRANSPOSE4_PS(zero, tpz, tpy, tpx);

  // a  =  A * tpx,   b =  A * tpy,   c =  A * tpz
  // da = dA * tpx,  db = dA * tpy,  dc = dA * tpz, etc.
  // A is 4x4 matrix given by the rows A_s[0], A_s[1], A_s[ 2], A_s[ 3]
  __m128 a, b, bP, tmp0, tmp1, tmp2, tmp3;
  // x-dependent vectors
  _MM_MATVEC4_PS (A_s[0], A_s[1], A_s[2], A_s[3], tpx,   a);
  // y-dependent vectors
  _MM_MATVEC4_PS (A_s[0], A_s[1], A_s[2], A_s[3], tpy,   b);
  // Compute cP, dcP, and d2cP products 1/4 at a time to maximize
  // register reuse and avoid rerereading from memory or cache.
  // 1st quarter
  tmp0 = _mm_loadu_ps (P(0));
  tmp1 = _mm_loadu_ps (P(1));
  tmp2 = _mm_loadu_ps (P(2));
  tmp3 = _mm_loadu_ps (P(3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   b,   bP);
  // Compute value
  _MM_DOT4_PS (a, bP, *val);
#undef P
}

/************************************************************/
/* 3D single-precision, real evaulation functions           */
/************************************************************/

/* Value only */
inline void
eval_UBspline_3d_s (UBspline_3d_s * restrict spline, 
		    double x, double y, double z,
		    float* restrict val)
{
  _mm_prefetch ((const char*)  &A_s[ 0],_MM_HINT_T0);  _mm_prefetch ((const char*)  &A_s[ 1],_MM_HINT_T0);  
  _mm_prefetch ((const char*)  &A_s[ 2],_MM_HINT_T0);  _mm_prefetch ((const char*)  &A_s[ 3],_MM_HINT_T0);

  /// SSE mesh point determination
  __m128 xyz       = _mm_set_ps (x, y, z, 0.0);
  __m128 x0y0z0    = _mm_set_ps (spline->x_grid.start,  spline->y_grid.start, 
				 spline->z_grid.start, 0.0);
  __m128 delta_inv = _mm_set_ps (spline->x_grid.delta_inv,spline->y_grid.delta_inv, 
				 spline->z_grid.delta_inv, 0.0);
  xyz = _mm_sub_ps (xyz, x0y0z0);
  // ux = (x - x0)/delta_x and same for y and z
  __m128 uxuyuz    = _mm_mul_ps (xyz, delta_inv);
  // intpart = trunc (ux, uy, uz)
  __m128i intpart  = _mm_cvttps_epi32(uxuyuz);
  __m128i ixiyiz;
  _mm_storeu_si128 (&ixiyiz, intpart);
  // Store to memory for use in C expressions
  // xmm registers are stored to memory in reverse order
  int ix = ((int *)&ixiyiz)[3];
  int iy = ((int *)&ixiyiz)[2];
  int iz = ((int *)&ixiyiz)[1];

  int xs = spline->x_stride;
  int ys = spline->y_stride;

  // This macro is used to give the pointer to coefficient data.
  // i and j should be in the range [0,3].  Coefficients are read four
  // at a time, so no k value is needed.
#define P(i,j) (spline->coefs+(ix+(i))*xs+(iy+(j))*ys+(iz))
  // Prefetch the data from main memory into cache so it's available
  // when we need to use it.
  _mm_prefetch ((const char*)P(0,0), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(0,1), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(0,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(0,3), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(1,0), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(1,1), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(1,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(1,3), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(2,0), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(2,1), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(2,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(2,3), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(3,0), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(3,1), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(3,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(3,3), _MM_HINT_T0);

  // Now compute the vectors:
  // tpx = [t_x^3 t_x^2 t_x 1]
  // tpy = [t_y^3 t_y^2 t_y 1]
  // tpz = [t_z^3 t_z^2 t_z 1]
  __m128 ipart  = _mm_cvtepi32_ps (intpart);
  __m128 txtytz = _mm_sub_ps (uxuyuz, ipart);
  __m128 one    = _mm_set_ps (1.0, 1.0, 1.0, 1.0);
  __m128 t2     = _mm_mul_ps (txtytz, txtytz);
  __m128 t3     = _mm_mul_ps (t2, txtytz);
  __m128 tpx    = t3;
  __m128 tpy    = t2;
  __m128 tpz    = txtytz;
  __m128 zero   = one;
  _MM_TRANSPOSE4_PS(zero, tpz, tpy, tpx);

  // a  =  A * tpx,   b =  A * tpy,   c =  A * tpz
  // da = dA * tpx,  db = dA * tpy,  dc = dA * tpz, etc.
  // A is 4x4 matrix given by the rows A_s[0], A_s[1], A_s[ 2], A_s[ 3]
  __m128 a, b, c, cP[4],bcP,
    tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

  // x-dependent vectors
  _MM_MATVEC4_PS (A_s[0], A_s[1], A_s[2], A_s[3], tpx,   a);
  // y-dependent vectors
  _MM_MATVEC4_PS (A_s[0], A_s[1], A_s[2], A_s[3], tpy,   b);
  // z-dependent vectors
  _MM_MATVEC4_PS (A_s[0], A_s[1], A_s[2], A_s[3], tpz,   c);

  // Compute cP, dcP, and d2cP products 1/4 at a time to maximize
  // register reuse and avoid rerereading from memory or cache.
  // 1st quarter
  tmp0 = _mm_loadu_ps (P(0,0));  tmp1 = _mm_loadu_ps (P(0,1));
  tmp2 = _mm_loadu_ps (P(0,2));  tmp3 = _mm_loadu_ps (P(0,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[0]);
  // 2nd quarter
  tmp0 = _mm_loadu_ps (P(1,0));  tmp1 = _mm_loadu_ps (P(1,1));
  tmp2 = _mm_loadu_ps (P(1,2));  tmp3 = _mm_loadu_ps (P(1,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[1]);
  // 3rd quarter
  tmp0 = _mm_loadu_ps (P(2,0));  tmp1 = _mm_loadu_ps (P(2,1));
  tmp2 = _mm_loadu_ps (P(2,2));  tmp3 = _mm_loadu_ps (P(2,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[2]);
  // 4th quarter
  tmp0 = _mm_loadu_ps (P(3,0));  tmp1 = _mm_loadu_ps (P(3,1));
  tmp2 = _mm_loadu_ps (P(3,2));  tmp3 = _mm_loadu_ps (P(3,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[3]);
  
  // Now compute bcP, dbcP, bdcP, d2bcP, bd2cP, and dbdc products
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3],   b,   bcP);

  // Compute value
  _MM_DOT4_PS (a, bcP, *val);

#undef P
}

#undef _MM_MATVEC4_PS
#undef _MM_DOT4_PS

#endif
