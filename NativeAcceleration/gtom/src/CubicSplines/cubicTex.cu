/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2013, Danny Ruijters. All rights reserved.
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

#ifndef CUBIC_TEX_CU
#define CUBIC_TEX_CU

#include "internal/bspline_kernel.cu"

namespace gtom
{
	//! Cubic interpolated texture lookup, using unnormalized coordinates.
	//! Straight forward implementation, using 4 nearest neighbour lookups.
	//! @param tex  1D texture
	//! @param x  unnormalized x texture coordinate
	template <class T> __device__ tfloat cubicTex1DSimple(cudaTex tex, tfloat x)
	{
		// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
		const tfloat coord_grid = x - (tfloat)0.5;
		tfloat index = floor(coord_grid);
		const tfloat fraction = coord_grid - index;
		index += (tfloat)0.5;  //move from [-0.5, extent-0.5] to [0, extent]

		T result = (T)0;
		for (tfloat x = -1; x < (tfloat)2.5; x++)
		{
			tfloat bsplineX = bspline(x - fraction);
			tfloat u = index + x;
			result += bsplineX * tex1D<T>(tex, u);
		}
		return result;
	}

	//! Bicubic interpolated texture lookup, using unnormalized coordinates.
	//! Straight forward implementation, using 16 nearest neighbour lookups.
	//! @param tex  2D texture
	//! @param x  unnormalized x texture coordinate
	//! @param y  unnormalized y texture coordinate
	template <class T> __device__ float cubicTex2DSimple(cudaTex tex, float x, float y)
	{
		// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
		const float2 coord_grid = make_float2(x - 0.5f, y - 0.5f);
		float2 index = floorf(coord_grid);
		const float2 fraction = coord_grid - index;
		index.x += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]
		index.y += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

		T result = (T)0;
		for (float y = -1; y < 2.5f; y++)
		{
			float bsplineY = bspline(y - fraction.y);
			float v = index.y + y;
			for (float x = -1; x < 2.5f; x++)
			{
				float bsplineXY = bspline(x - fraction.x) * bsplineY;
				float u = index.x + x;
				result += bsplineXY * tex2D<T>(tex, u, v);
			}
		}
		return result;
	}

	//! Tricubic interpolated texture lookup, using unnormalized coordinates.
	//! Straight forward implementation, using 64 nearest neighbour lookups.
	//! @param tex  3D texture
	//! @param coord  unnormalized 3D texture coordinate
	template <class T> __device__ float cubicTex3DSimple(cudaTex tex, float x, float y, float z)
	{
		// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
		const float3 coord_grid = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
		float3 index = floorf(coord_grid);
		const float3 fraction = coord_grid - index;
		index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

		T result = (T)0;
		for (float z = -1; z < 2.5f; z++)  //range [-1, 2]
		{
			float bsplineZ = bspline(z - fraction.z);
			float w = index.z + z;
			for (float y = -1; y < 2.5f; y++)
			{
				float bsplineYZ = bspline(y - fraction.y) * bsplineZ;
				float v = index.y + y;
				for (float x = -1; x < 2.5f; x++)
				{
					float bsplineXYZ = bspline(x - fraction.x) * bsplineYZ;
					float u = index.x + x;
					result += bsplineXYZ * tex3D<T>(tex, u, v, w);
				}
			}
		}
		return result;
	}
}
#endif // _CUBIC1D_KERNEL_H_