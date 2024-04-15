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

#ifndef _3D_CUBIC_BSPLINE_PREFILTER_H_
#define _3D_CUBIC_BSPLINE_PREFILTER_H_

#include <stdio.h>
#include "internal/cubicPrefilter_kernel.cu"
#include "gtom/include/Prerequisites.cuh"

namespace gtom
{
	//--------------------------------------------------------------------------
	// Global CUDA procedures
	//--------------------------------------------------------------------------
	template<class T> __global__ void SamplesToCoefficients3DX(T* d_volume, uint3 dims)
	{
		d_volume += Elements(dims) * blockIdx.z;

		// process lines in x-direction
		const uint y = blockIdx.x * blockDim.x + threadIdx.x;
		if (y >= dims.y)
			return;
		const uint z = blockIdx.y * blockDim.y + threadIdx.y;
		if (z >= dims.z)
			return;
		const uint startIdx = (z * dims.y + y) * dims.x;

		ConvertToInterpolationCoefficients(d_volume + startIdx, dims.x, 1);
	}

	template<class T> __global__ void SamplesToCoefficients3DY(T* d_volume, uint3 dims)
	{
		d_volume += Elements(dims) * blockIdx.z;

		// process lines in y-direction
		const uint x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= dims.x)
			return;
		const uint z = blockIdx.y * blockDim.y + threadIdx.y;
		if (z >= dims.z)
			return;
		const uint startIdx = z * dims.y * dims.x + x;

		ConvertToInterpolationCoefficients(d_volume + startIdx, dims.y, dims.x);
	}

	template<class T> __global__ void SamplesToCoefficients3DZ(T* d_volume, uint3 dims)
	{
		d_volume += Elements(dims) * blockIdx.z;

		// process lines in z-direction
		const uint x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= dims.x)
			return;
		const uint y = blockIdx.y * blockDim.y + threadIdx.y;
		if (y >= dims.y)
			return;
		const uint startIdx = y * dims.x + x;

		ConvertToInterpolationCoefficients(d_volume + startIdx, dims.z, dims.x * dims.y);
	}

	//--------------------------------------------------------------------------
	// Exported functions
	//--------------------------------------------------------------------------

	//! Convert the voxel values into cubic b-spline coefficients
	//! @param volume  pointer to the voxel volume in GPU (device) memory
	//! @param pitch   width in bytes (including padding bytes)
	//! @param width   volume width in number of voxels
	//! @param height  volume height in number of voxels
	//! @param depth   volume depth in number of voxels
	template<class T> void d_CubicBSplinePrefilter3D(T* volume, int3 dims, int batch)
	{
		dim3 TpB(16, 16);

		// Replace the voxel values by the b-spline coefficients
		dim3 dimGridX((dims.y + 15) / 16, (dims.z + 15) / 16, batch);
		SamplesToCoefficients3DX<T> << <dimGridX, TpB >> >(volume, make_uint3(dims.x, dims.y, dims.z));

		dim3 dimGridY((dims.x + 15) / 16, (dims.z + 15) / 16, batch);
		SamplesToCoefficients3DY<T> << <dimGridY, TpB >> >(volume, make_uint3(dims.x, dims.y, dims.z));

		dim3 dimGridZ((dims.x + 15) / 16, (dims.y + 15) / 16, batch);
		SamplesToCoefficients3DZ<T> << <dimGridZ, TpB >> >(volume, make_uint3(dims.x, dims.y, dims.z));
	}
	template void d_CubicBSplinePrefilter3D<float>(float* volume, int3 dims, int batch);
}
#endif  //_3D_CUBIC_BSPLINE_PREFILTER_H_