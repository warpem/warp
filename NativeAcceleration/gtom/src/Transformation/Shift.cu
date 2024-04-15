#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template<int ndims, bool iszerocentered> __global__ void ShiftFourierKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* d_delta);
	template<int ndims, bool iszerocentered> __global__ void MotionBlurKernel(tfloat* d_output, int3 dims, float3* d_shifts, ushort nshifts);


	////////////////////////////////////////
	//Equivalent of TOM's tom_shift method//
	////////////////////////////////////////

	void d_Shift(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* h_delta, cufftHandle* planforw, cufftHandle* planback, tcomplex* d_sharedintermediate, int batch)
	{
		tcomplex* d_intermediate = NULL;
		if (d_sharedintermediate == NULL)
			cudaMalloc((void**)&d_intermediate, batch * ElementsFFT(dims) * sizeof(tcomplex));
		else
			d_intermediate = d_sharedintermediate;

		if (planforw == NULL)
			d_FFTR2C(d_input, d_intermediate, DimensionCount(dims), dims, batch);
		else
			d_FFTR2C(d_input, d_intermediate, planforw);

		d_Shift(d_intermediate, d_intermediate, dims, h_delta, false, batch);

		if (planback == NULL)
			d_IFFTC2R(d_intermediate, d_output, DimensionCount(dims), dims, batch);
		else
			d_IFFTC2R(d_intermediate, d_output, planback, dims);

		if (d_sharedintermediate == NULL)
			cudaFree(d_intermediate);
	}

	void d_Shift(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* h_delta, bool iszerocentered, int batch)
	{
		tfloat3* h_deltanorm = (tfloat3*)malloc(batch * sizeof(tfloat3));
		for (int b = 0; b < batch; b++)
			h_deltanorm[b] = tfloat3(h_delta[b].x / (tfloat)dims.x, h_delta[b].y / (tfloat)dims.y, h_delta[b].z / (tfloat)dims.z);
		tfloat3* d_delta = (tfloat3*)CudaMallocFromHostArray(h_deltanorm, batch * sizeof(tfloat3));
		free(h_deltanorm);

		int TpB = tmin(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		if (!iszerocentered)
		{
			if (DimensionCount(dims) == 3)
				ShiftFourierKernel <3, false> << <grid, TpB >> > (d_input, d_output, dims, d_delta);
			else
				ShiftFourierKernel <2, false> << <grid, TpB >> > (d_input, d_output, dims, d_delta);
		}
		else
		{
			if (DimensionCount(dims) == 3)
				ShiftFourierKernel <3, true> << <grid, TpB >> > (d_input, d_output, dims, d_delta);
			else
				ShiftFourierKernel <2, true> << <grid, TpB >> > (d_input, d_output, dims, d_delta);
		}

		cudaFree(d_delta);
	}

	void d_MotionBlur(tfloat* d_output, int3 dims, float3* h_shifts, uint nshifts, bool iszerocentered, uint batch)
	{
		float3* h_deltanorm = (float3*)malloc(nshifts * batch * sizeof(float3));
		for (int b = 0; b < nshifts * batch; b++)
			h_deltanorm[b] = make_float3(h_shifts[b].x / (float)dims.x, h_shifts[b].y / (float)dims.y, h_shifts[b].z / (float)dims.z);
		float3* d_delta = (float3*)CudaMallocFromHostArray(h_deltanorm, nshifts * batch * sizeof(float3));
		free(h_deltanorm);

		int TpB = tmin(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		if (!iszerocentered)
		{
			if (DimensionCount(dims) == 3)
				MotionBlurKernel <3, false> << <grid, TpB >> > (d_output, dims, d_delta, nshifts);
			else
				MotionBlurKernel <2, false> << <grid, TpB >> > (d_output, dims, d_delta, nshifts);
		}
		else
		{
			if (DimensionCount(dims) == 3)
				MotionBlurKernel <3, true> << <grid, TpB >> > (d_output, dims, d_delta, nshifts);
			else
				MotionBlurKernel <2, true> << <grid, TpB >> > (d_output, dims, d_delta, nshifts);
		}

		cudaFree(d_delta);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<int ndims, bool iszerocentered> __global__ void ShiftFourierKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* d_delta)
	{
		int idy = blockIdx.x;
		int idz = blockIdx.y;

		int x, y, z;
		if (!iszerocentered)
		{
			y = idy > dims.y / 2 ? idy - dims.y : idy;
			z = idz > dims.z / 2 ? idz - dims.z : idz;
		}
		else
		{
			y = dims.y / 2 - idy;
			z = dims.z / 2 - idz;
		}

		d_input += ((blockIdx.z * dims.z + idz) * dims.y + idy) * (dims.x / 2 + 1);
		d_output += ((blockIdx.z * dims.z + idz) * dims.y + idy) * (dims.x / 2 + 1);
		tfloat3 delta = d_delta[blockIdx.z];

		for (int idx = threadIdx.x; idx <= dims.x / 2; idx += blockDim.x)
		{
			if (!iszerocentered)
				x = idx;
			else
				x = dims.x / 2 - idx;

			tfloat factor = -(delta.x * (tfloat)x + delta.y * (tfloat)y + (ndims > 2 ? delta.z * (tfloat)z : (tfloat)0)) * (tfloat)PI2;
			tcomplex multiplicator = make_cuComplex(cos(factor), sin(factor));

			d_output[idx] = cmul(d_input[idx], multiplicator);
		}
	}

	template<int ndims, bool iszerocentered> __global__ void MotionBlurKernel(tfloat* d_output, int3 dims, float3* d_shifts, ushort nshifts)
	{
		int idy = blockIdx.x;
		int idz = blockIdx.y;

		int x, y, z;
		if (!iszerocentered)
		{
			y = FFTShift(idy, dims.y) - dims.y / 2;
			z = FFTShift(idz, dims.z) - dims.z / 2;
		}
		else
		{
			y = dims.y / 2 - idy;
			z = dims.z / 2 - idz;
		}

		d_output += ((blockIdx.z * dims.z + idz) * dims.y + idy) * (dims.x / 2 + 1);
		d_shifts += blockIdx.z * nshifts;

		for (int idx = threadIdx.x; idx <= dims.x / 2; idx += blockDim.x)
		{
			if (!iszerocentered)
				x = FFTShift(idx, dims.x) - dims.x / 2;
			else
				x = dims.x / 2 - idx;

			float2 shift = make_float2(0, 0);
			for (ushort s = 0; s < nshifts; s++)
			{
				float3 delta = d_shifts[s];
				float factor = -(delta.x * (float)x + delta.y * (float)y + (ndims > 2 ? delta.z * (float)z : (float)0)) * (float)PI2;
				shift += make_cuComplex(cos(factor), sin(factor));
			}

			d_output[idx] = length(shift) / (float)nshifts;
		}
	}
}