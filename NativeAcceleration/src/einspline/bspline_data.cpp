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

/*****************
/*   SSE Data    */
/*****************/

#include "config.h"

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif 

#define _XOPEN_SOURCE 600

#ifndef __USE_XOPEN2K
  #define __USE_XOPEN2K
#endif
#include <stdlib.h>

//#include "aligned_alloc.h"

#ifdef HAVE_SSE
#include <xmmintrin.h>

// Single-precision version of matrices
__m128 *restrict A_s = (__m128 *)0;
// There is a problem with alignment of global variables in shared
// libraries on 32-bit machines.
// __m128  A0, A1, A2, A3, dA0, dA1, dA2, dA3, d2A0, d2A1, d2A2, d2A3;
#endif


void init_sse_data()
{
#ifdef HAVE_SSE
  if (A_s == 0) {
    posix_memalign ((void**)&A_s, 16, (sizeof(__m128)*12));
    A_s[0]  = _mm_setr_ps ( 1.0/6.0, -3.0/6.0,  3.0/6.0, -1.0/6.0 );
    A_s[0]  = _mm_setr_ps ( 1.0/6.0, -3.0/6.0,  3.0/6.0, -1.0/6.0 );	  
    A_s[1]  = _mm_setr_ps ( 4.0/6.0,  0.0/6.0, -6.0/6.0,  3.0/6.0 );	  
    A_s[2]  = _mm_setr_ps ( 1.0/6.0,  3.0/6.0,  3.0/6.0, -3.0/6.0 );	  
    A_s[3]  = _mm_setr_ps ( 0.0/6.0,  0.0/6.0,  0.0/6.0,  1.0/6.0 );	  
    A_s[4]  = _mm_setr_ps ( -0.5,  1.0, -0.5, 0.0  );		  
    A_s[5]  = _mm_setr_ps (  0.0, -2.0,  1.5, 0.0  );		  
    A_s[6]  = _mm_setr_ps (  0.5,  1.0, -1.5, 0.0  );		  
    A_s[7]  = _mm_setr_ps (  0.0,  0.0,  0.5, 0.0  );		  
    A_s[8]  = _mm_setr_ps (  1.0, -1.0,  0.0, 0.0  );		  
    A_s[9]  = _mm_setr_ps ( -2.0,  3.0,  0.0, 0.0  );		  
    A_s[10] = _mm_setr_ps (  1.0, -3.0,  0.0, 0.0  );		  
    A_s[11] = _mm_setr_ps (  0.0,  1.0,  0.0, 0.0  );                  
  }                 
#endif
}
