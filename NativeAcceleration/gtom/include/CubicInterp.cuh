#include "Prerequisites.cuh"

#ifndef CUBIC_INTERP
#define CUBIC_INTERP

namespace gtom
{
	///////////////////////
	//Cubic interpolation//
	///////////////////////

	/**
	* \brief Prefilters a 2D map to accelerate cubic interpolation over it; the operation is performed in-place, i. e. original data will be overwritten
	* \param[in] d_image	Array with input image, prefiltered data will be written here
	* \param[in] pitch	Length of one line in bytes (mind 4 byte alignment for CUDA textures)
	* \param[in] dims	Image dimensions
	*/
	template<class T> void d_CubicBSplinePrefilter2D(T* d_image, int2 dims, int batch = 1);

	/**
	* \brief Prefilters a 3D volume to accelerate cubic interpolation over it; the operation is performed in-place, i. e. original data will be overwritten
	* \param[in] d_image	Array with input volume, prefiltered data will be written here
	* \param[in] pitch	Length of one line in bytes (mind 4 byte alignment for CUDA textures)
	* \param[in] dims	Volume dimensions
	*/
	template<class T> void d_CubicBSplinePrefilter3D(T* d_volume, int3 dims, int batch = 1);


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


	// Inline calculation of the bspline convolution weights, without conditional statements
	inline __device__ void bspline_weights(float fraction, float& w0, float& w1, float& w2, float& w3)
	{
		const float one_frac = 1.0f - fraction;
		const float squared = fraction * fraction;
		const float one_sqd = one_frac * one_frac;

		w0 = 1.0f / 6.0f * one_sqd * one_frac;
		w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
		w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
		w3 = 1.0f / 6.0f * squared * fraction;
	}

	inline __device__ void bspline_weights(float2 fraction, float2& w0, float2& w1, float2& w2, float2& w3)
	{
		const float2 one_frac = 1.0f - fraction;
		const float2 squared = fraction * fraction;
		const float2 one_sqd = one_frac * one_frac;

		w0 = 1.0f / 6.0f * one_sqd * one_frac;
		w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
		w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
		w3 = 1.0f / 6.0f * squared * fraction;
	}

	inline __device__ void bspline_weights(float3 fraction, float3& w0, float3& w1, float3& w2, float3& w3)
	{
		const float3 one_frac = 1.0f - fraction;
		const float3 squared = fraction * fraction;
		const float3 one_sqd = one_frac * one_frac;

		w0 = 1.0f / 6.0f * one_sqd * one_frac;
		w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
		w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
		w3 = 1.0f / 6.0f * squared * fraction;
	}

	inline __host__ __device__ float bspline(float t)
	{
		t = fabs(t);
		const float a = 2.0f - t;

		if (t < 1.0f)
			return 2.0f / 3.0f - 0.5f*t*t*a;
		else if (t < 2.0f)
			return a*a*a / 6.0f;
		else
			return 0.0f;
	}


	//! Bicubic interpolated texture lookup, using unnormalized coordinates.
	//! Fast implementation, using 2 linear lookups.
	//! @param tex  1D texture
	//! @param x  unnormalized x texture coordinate
	//! @param y  unnormalized y texture coordinate
	inline __device__ tfloat cubicTex1D(cudaTex tex, float x)
	{
		// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
		const float coord_grid = x - 0.5f;
		const float index = floor(coord_grid);
		const float fraction = coord_grid - index;
		float w0, w1, w2, w3;
		bspline_weights(fraction, w0, w1, w2, w3);

		const float g0 = w0 + w1;
		const float g1 = w2 + w3;
		const float h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
		const float h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

		// fetch the two linear interpolations
		float tex0 = tex1D<tfloat>(tex, h0);
		float tex1 = tex1D<tfloat>(tex, h1);

		// weigh along the x-direction
		return (g0 * tex0 + g1 * tex1);
	}


	//! Bicubic interpolated texture lookup, using unnormalized coordinates.
	//! Fast implementation, using 4 bilinear lookups.
	//! @param tex  2D texture object
	//! @param x  unnormalized x texture coordinate
	//! @param y  unnormalized y texture coordinate
	inline __device__ float cubicTex2D(cudaTex tex, float x, float y)
	{
		// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
		const float2 coord_grid = make_float2(x - 0.5f, y - 0.5f);
		const float2 index = floorf(coord_grid);
		const float2 fraction = coord_grid - index;
		float2 w0, w1, w2, w3;
		bspline_weights(fraction, w0, w1, w2, w3);

		const float2 g0 = w0 + w1;
		const float2 g1 = w2 + w3;
		const float2 h0 = (w1 / g0) - make_float2(0.5f) + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
		const float2 h1 = (w3 / g1) + make_float2(1.5f) + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

		// fetch the four linear interpolations
		float tex00 = tex2D<tfloat>(tex, h0.x, h0.y);
		float tex10 = tex2D<tfloat>(tex, h1.x, h0.y);
		float tex01 = tex2D<tfloat>(tex, h0.x, h1.y);
		float tex11 = tex2D<tfloat>(tex, h1.x, h1.y);

		// weigh along the y-direction
		tex00 = g0.y * tex00 + g1.y * tex01;
		tex10 = g0.y * tex10 + g1.y * tex11;

		// weigh along the x-direction
		return (g0.x * tex00 + g1.x * tex10);
	}

	//! Tricubic interpolated texture lookup, using unnormalized coordinates.
	//! Fast implementation, using 8 trilinear lookups.
	//! @param tex  3D texture
	//! @param coord  unnormalized 3D texture coordinate
	inline __device__ float cubicTex3D(cudaTex tex, float x, float y, float z)
	{
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
		const float3 coord_grid = make_float3(x, y, z) - 0.5f;
		const float3 index = floorf(coord_grid);
		const float3 fraction = coord_grid - index;
		float3 w0, w1, w2, w3;
		bspline_weights(fraction, w0, w1, w2, w3);

		const float3 g0 = w0 + w1;
		const float3 g1 = w2 + w3;
		const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
		const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

		// fetch the eight linear interpolations
		// weighting and fetching is interleaved for performance and stability reasons
		float tex000 = tex3D<tfloat>(tex, h0.x, h0.y, h0.z);
		float tex100 = tex3D<tfloat>(tex, h1.x, h0.y, h0.z);
		tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
		float tex010 = tex3D<tfloat>(tex, h0.x, h1.y, h0.z);
		float tex110 = tex3D<tfloat>(tex, h1.x, h1.y, h0.z);
		tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
		tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
		float tex001 = tex3D<tfloat>(tex, h0.x, h0.y, h1.z);
		float tex101 = tex3D<tfloat>(tex, h1.x, h0.y, h1.z);
		tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
		float tex011 = tex3D<tfloat>(tex, h0.x, h1.y, h1.z);
		float tex111 = tex3D<tfloat>(tex, h1.x, h1.y, h1.z);
		tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
		tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

		return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
	}

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
			result += bsplineX * tex3D<T>(tex, u, 0.5f, 0.5f);
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

	class Cubic1D
	{
	public:
		std::vector<tfloat2> Data;
		std::vector<tfloat> Breaks;
		std::vector<tfloat4> Coefficients;

		Cubic1D(std::vector<tfloat2> data);
		std::vector<tfloat> Interp(std::vector<tfloat> x);

	private:
		std::vector<tfloat> GetPCHIPSlopes(std::vector<tfloat2> data, std::vector<tfloat> del);
		std::vector<tfloat> Diff(std::vector<tfloat> series);
	};
}
#endif