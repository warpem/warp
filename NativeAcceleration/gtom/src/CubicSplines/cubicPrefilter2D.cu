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

	// The same recursions, each line still computed by one thread with the same operations in the same order, but
	// with memory access that does not serialize them. Rows: a warp takes 32 rows and walks them in 32-column tiles
	// staged through shared memory, so loads and stores are coalesced (one thread per row read its own row, 16 KB
	// apart from its neighbours'). Columns: each thread loads a batch of samples ahead of the recursion instead of
	// waiting for one load per step.
	template<class T> __global__ void SamplesToCoefficients2DXTiled(T* d_image, uint2 dims)
	{
		__shared__ T tile[32][33];
		d_image += Elements2(dims) * blockIdx.y;
		const uint lane = threadIdx.x, row0 = blockIdx.x * 32, width = dims.x;
		const uint rows = min(32u, dims.y - row0), chunks = (width + 31) / 32;
		const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);
		T previous_c = 0;

		for (uint chunk = 0; chunk < chunks; chunk++)
		{
			uint col0 = chunk * 32, cols = min(32u, width - col0);
			if (lane < cols)
				for (uint r = 0; r < rows; r++)
					tile[r][lane] = d_image[(size_t)(row0 + r) * width + col0 + lane];
			__syncwarp();
			if (lane < rows)
			{
				uint k = 0;
				if (chunk == 0)
				{
					// InitialCausalCoefficient: its horizon of at most 12 samples lies in the first tile.
					const int Horizon = min(12, (int)width);
					float zn = Pole;
					T Sum = tile[lane][0];
					for (uint n = 0; n < Horizon; n++)
					{
						Sum += zn * tile[lane][n];
						zn *= Pole;
					}
					tile[lane][0] = previous_c = Lambda * Sum;
					k = 1;
				}
				for (; k < cols; k++)
					tile[lane][k] = previous_c = Lambda * tile[lane][k] + Pole * previous_c;
			}
			__syncwarp();
			// The last tile stays in shared memory for the anticausal pass.
			if (chunk + 1 < chunks && lane < cols)
				for (uint r = 0; r < rows; r++)
					d_image[(size_t)(row0 + r) * width + col0 + lane] = tile[r][lane];
			__syncwarp();
		}

		for (int chunk = (int)chunks - 1; chunk >= 0; chunk--)
		{
			uint col0 = chunk * 32, cols = min(32u, width - col0);
			if (chunk + 1 < (int)chunks)
			{
				if (lane < cols)
					for (uint r = 0; r < rows; r++)
						tile[r][lane] = d_image[(size_t)(row0 + r) * width + col0 + lane];
				__syncwarp();
			}
			if (lane < rows)
			{
				int k = (int)cols - 1;
				if (chunk + 1 == (int)chunks)
				{
					tile[lane][k] = previous_c = InitialAntiCausalCoefficient(&tile[lane][k]);
					k--;
				}
				for (; k >= 0; k--)
					tile[lane][k] = previous_c = Pole * (previous_c - tile[lane][k]);
			}
			__syncwarp();
			if (lane < cols)
				for (uint r = 0; r < rows; r++)
					d_image[(size_t)(row0 + r) * width + col0 + lane] = tile[r][lane];
			__syncwarp();
		}
	}

	template<class T> __global__ void SamplesToCoefficients2DYBatched(T* d_image, uint2 dims)
	{
		d_image += Elements2(dims) * blockIdx.y;
		const uint x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= dims.x)
			return;

		const int Batch = 16;
		const size_t stride = dims.x;
		const int length = dims.y;
		const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);
		T* c = d_image + x;
		T previous_c;
		c[0] = previous_c = Lambda * InitialCausalCoefficient(c, length, dims.x);
		int n = 1;
		for (; n + Batch <= length; n += Batch)
		{
			T v[Batch];
#pragma unroll
			for (int k = 0; k < Batch; k++)
				v[k] = c[(n + k) * stride];
#pragma unroll
			for (int k = 0; k < Batch; k++)
				c[(n + k) * stride] = previous_c = Lambda * v[k] + Pole * previous_c;
		}
		for (; n < length; n++)
			c[n * stride] = previous_c = Lambda * c[n * stride] + Pole * previous_c;

		c[(length - 1) * stride] = previous_c = InitialAntiCausalCoefficient(c + (length - 1) * stride);
		int m = length - 2;
		for (; m - Batch + 1 >= 0; m -= Batch)
		{
			T v[Batch];
#pragma unroll
			for (int k = 0; k < Batch; k++)
				v[k] = c[(m - k) * stride];
#pragma unroll
			for (int k = 0; k < Batch; k++)
				c[(m - k) * stride] = previous_c = Pole * (previous_c - v[k]);
		}
		for (; m >= 0; m--)
			c[m * stride] = previous_c = Pole * (previous_c - c[m * stride]);
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
		dim3 grid((dims.y + 31) / 32, batch);
		SamplesToCoefficients2DXTiled<T> << <grid, 32 >> >(image, make_uint2(dims.x, dims.y));

		int TpB = (min(dims.x, 128));
		grid = dim3((dims.x + TpB - 1) / TpB, batch);
		SamplesToCoefficients2DYBatched<T> << <grid, TpB >> >(image, make_uint2(dims.x, dims.y));
	}

	// The original one-thread-per-line kernels, for comparison in tests.
	template<class T> void d_CubicBSplinePrefilter2DReference(T* image, int2 dims, int batch)
	{
		int TpB = (min(dims.y, 192));
		dim3 grid((dims.y + TpB - 1) / TpB, batch);
		SamplesToCoefficients2DX<T> << <grid, TpB >> >(image, make_uint2(dims.x, dims.y));

		TpB = (min(dims.x, 192));
		grid = dim3((dims.x + TpB - 1) / TpB, batch);
		SamplesToCoefficients2DY<T> << <grid, TpB >> >(image, make_uint2(dims.x, dims.y));
	}
	template void d_CubicBSplinePrefilter2DReference<tfloat>(tfloat* image, int2 dims, int batch);
	template void d_CubicBSplinePrefilter2D<tfloat>(tfloat* image, int2 dims, int batch);
}
#endif  //_2D_CUBIC_BSPLINE_PREFILTER_H_