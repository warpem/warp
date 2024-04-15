#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Binary.cuh"

namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void Dilate2DKernel(T* d_input, T* d_output, int3 dims);
	template <class T> __global__ void Dilate3DKernel(T* d_input, T* d_output, int3 dims);


	//////////
	//Dilate//
	//////////

	template <class T> void d_Dilate(T* d_input, T* d_output, int3 dims, int batch)
	{
		dim3 TpB = dim3(32, 8);
		dim3 grid = dim3((dims.x + TpB.x - 1) / TpB.x, (dims.y + TpB.y - 1) / TpB.y, dims.z);
		for (int b = 0; b < batch; b++)
			if (DimensionCount(dims) <= 2)
				Dilate2DKernel << <grid, TpB >> > (d_input + Elements(dims) * b, d_output + Elements(dims) * b, dims);
			else
				Dilate3DKernel << <grid, TpB >> > (d_input + Elements(dims) * b, d_output + Elements(dims) * b, dims);
	}
	template void d_Dilate<char>(char* d_input, char* d_output, int3 dims, int batch);
	template void d_Dilate<int>(int* d_input, int* d_output, int3 dims, int batch);
	template void d_Dilate<float>(float* d_input, float* d_output, int3 dims, int batch);
	template void d_Dilate<double>(double* d_input, double* d_output, int3 dims, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void Dilate2DKernel(T* d_input, T* d_output, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y)
			return;

		d_output += idy * dims.x + idx;

		if (idx > 0 && d_input[idy * dims.x + idx - 1] > (T)0)
			*d_output = (T)1;
		else if (idx < dims.x - 1 && d_input[idy * dims.x + idx + 1] >(T)0)
			*d_output = (T)1;
		else if (blockIdx.y > 0 && d_input[(idy - 1) * dims.x + idx] > (T)0)
			*d_output = (T)1;
		else if (blockIdx.y < dims.y - 1 && d_input[(idy + 1) * dims.x + idx] >(T)0)
			*d_output = (T)1;
		else
			*d_output = d_input[idy * dims.x + idx];
	}

	template <class T> __global__ void Dilate3DKernel(T* d_input, T* d_output, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y)
			return;
		int idz = blockIdx.z;

		d_output += (idz * dims.y + idy) * dims.x + idx;

		if (idx > 0 && d_input[(idz * dims.y + idy) * dims.x + idx - 1] > (T)0)
			*d_output = (T)1;
		else if (idx < dims.x - 1 && d_input[(idz * dims.y + idy) * dims.x + idx + 1] >(T)0)
			*d_output = (T)1;
		else if (blockIdx.y > 0 && d_input[(idz * dims.y + idy - 1) * dims.x + idx] > (T)0)
			*d_output = (T)1;
		else if (blockIdx.y < dims.y - 1 && d_input[(idz * dims.y + idy + 1) * dims.x + idx] >(T)0)
			*d_output = (T)1;
		else if (blockIdx.z > 0 && d_input[((idz - 1) * dims.y + idy) * dims.x + idx] > (T)0)
			*d_output = (T)1;
		else if (blockIdx.y < dims.y - 1 && d_input[((idz + 1) * dims.y + idy) * dims.x + idx] >(T)0)
			*d_output = (T)1;
		else
			*d_output = d_input[idy * dims.x + idx];
	}
}