#include "gtom/include/Prerequisites.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T, int ndims> __global__ void RectangleMaskKernel(T* d_input, T* d_output, int3 dims, int3 dimsbox, int3 center);


	////////////////
	//Host methods//
	////////////////

	template <class T> void d_RectangleMask(T* d_input,
		T* d_output,
		int3 dimsmask,
		int3 dimsbox,
		int3* center,
		int batch)
	{
		int3 _center = center != NULL ? *center : toInt3(dimsmask.x / 2, dimsmask.y / 2, dimsmask.z / 2);

		int TpB = 256;
		dim3 grid = dim3(dimsmask.y, dimsmask.z, batch);
		if (DimensionCount(dimsmask) == 3)
			RectangleMaskKernel<T, 3> << <grid, TpB >> > (d_input, d_output, dimsmask, dimsbox, _center);
		else if (DimensionCount(dimsmask) == 2)
			RectangleMaskKernel<T, 2> << <grid, TpB >> > (d_input, d_output, dimsmask, dimsbox, _center);
		else if (DimensionCount(dimsmask) == 1)
			RectangleMaskKernel<T, 1> << <grid, TpB >> > (d_input, d_output, dimsmask, dimsbox, _center);
	}
	template void d_RectangleMask<tfloat>(tfloat* d_input, tfloat* d_output, int3 dimsmask, int3 dimsbox, int3* center, int batch);
	template void d_RectangleMask<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dimsmask, int3 dimsbox, int3* center, int batch);

	////////////////
	//CUDA kernels//
	////////////////

	template <class T, int ndims> __global__ void RectangleMaskKernel(T* d_input, T* d_output, int3 dims, int3 dimsbox, int3 center)
	{
		d_input += Elements(dims) * blockIdx.z;
		d_output += Elements(dims) * blockIdx.z;

		int mask = 1;

		if (ndims > 2)
		{
			int idz = blockIdx.y;
			int offsetz = idz - center.z;
			if (offsetz < -dimsbox.z / 2 || offsetz >(dimsbox.z - 1) / 2)
				mask = 0;
		}
		if (ndims > 1)
		{
			if (mask == 1)
			{
				int idy = blockIdx.x;
				int offsety = idy - center.y;
				if (offsety < -dimsbox.y / 2 || offsety >(dimsbox.y - 1) / 2)
					mask = 0;
			}
		}

		if (ndims == 3)
		{
			d_input += (blockIdx.y * dims.y + blockIdx.x) * dims.x;
			d_output += (blockIdx.y * dims.y + blockIdx.x) * dims.x;
		}
		else if (ndims == 2)
		{
			d_input += blockIdx.x * dims.x;
			d_output += blockIdx.x * dims.x;
		}

		if (mask == 1)
		{
			for (int idx = threadIdx.x; idx < dims.x; idx += blockDim.x)
			{
				int offsetx = idx - center.x;
				if (offsetx < -dimsbox.x / 2 || offsetx >(dimsbox.x - 1) / 2)
					d_output[idx] = d_input[idx] * 0;
				else
					d_output[idx] = d_input[idx];
			}
		}
		else
		{
			for (int idx = threadIdx.x; idx < dims.x; idx += blockDim.x)
				d_output[idx] = d_input[idx] * 0;
		}
	}
}