/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2010, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.

When using this code in a scientific project, please cite one or all of the
following papers:
*  Daniel Ruijters and Philippe Thévenaz,
   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
   The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
*  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
   Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
\*--------------------------------------------------------------------------*/

#ifndef _2D_CUBIC_BSPLINE_PREFILTER_H_
#define _2D_CUBIC_BSPLINE_PREFILTER_H_

#include <stdio.h>
#include "internal/cubicPrefilter_kernel.cu"
#include "gtom/include/Prerequisites.cuh"

namespace gtom
{
	// ***************************************************************************
	// *	Global GPU procedures
	// ***************************************************************************
	template<class T> __global__ void SamplesToCoefficients2DX(T* d_image, uint2 dims)
	{
		d_image += Elements2(dims) * blockIdx.y;

		// process lines in x-direction
		const uint y = blockIdx.x * blockDim.x + threadIdx.x;
		if (y >= dims.y)
			return;

		ConvertToInterpolationCoefficients(d_image + y * dims.x, dims.x, 1);
	}

	template<class T> __global__ void SamplesToCoefficients2DY(T* d_image, uint2 dims)
	{
		d_image += Elements2(dims) * blockIdx.y;

		// process lines in x-direction
		const uint x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= dims.x)
			return;

		ConvertToInterpolationCoefficients(d_image + x, dims.y, dims.x);
	}

	// ***************************************************************************
	// *	Exported functions
	// ***************************************************************************

	//! Convert the pixel values into cubic b-spline coefficients
	//! @param image  pointer to the image bitmap in GPU (device) memory
	//! @param pitch   width in bytes (including padding bytes)
	//! @param width   image width in number of pixels
	//! @param height  image height in number of pixels
	template<class T> void d_CubicBSplinePrefilter2D(T* image, int2 dims, int batch)
	{
		int TpB = (min(dims.y, 192));
		dim3 grid((dims.y + TpB - 1) / TpB, batch);
		SamplesToCoefficients2DX<T> << <grid, TpB >> >(image, make_uint2(dims.x, dims.y));

		TpB = (min(dims.x, 192));
		grid = dim3((dims.x + TpB - 1) / TpB, batch);
		SamplesToCoefficients2DY<T> << <grid, TpB >> >(image, make_uint2(dims.x, dims.y));
	}
	template void d_CubicBSplinePrefilter2D<tfloat>(tfloat* image, int2 dims, int batch);
}
#endif  //_2D_CUBIC_BSPLINE_PREFILTER_H_