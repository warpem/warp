#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"

namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims);
	template <class T> __global__ void RemapFullFFT2HalfFFTKernel(T* d_input, T* d_output, int3 dims);
	template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, uint3 dims, uint elements);
	template <class T> __global__ void RemapFull2FullFFTKernel(T* d_input, T* d_output, uint3 dims, uint elements);
	template <class T> __global__ void RemapHalfFFT2FullFFTKernel(T* d_input, T* d_output, int3 dims);
	template <class T> __global__ void RemapHalfFFT2HalfKernel(T* d_input, T* d_output, int3 dims);
	template <class T> __global__ void RemapHalf2HalfFFTKernel(T* d_input, T* d_output, int3 dims);


	////////////////
	//Host methods//
	////////////////

	template <class T> void d_RemapFull2HalfFFT(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, ElementsFFT(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapFull2HalfFFTKernel << <grid, TpB >> > (d_input, d_intermediate, dims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, ElementsFFT(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapFull2HalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapFull2HalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapFull2HalfFFT<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapFullFFT2HalfFFT(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, ElementsFFT(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapFullFFT2HalfFFTKernel << <grid, TpB >> > (d_input, d_intermediate, dims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, ElementsFFT(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapFullFFT2HalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapFullFFT2HalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapFullFFT2HalfFFT<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapFullFFT2Full(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, Elements(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapFullFFT2FullKernel << <grid, TpB >> > (d_input, d_intermediate, make_uint3(dims.x, dims.y, dims.z), Elements(dims));

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, Elements(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapFullFFT2Full<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapFullFFT2Full<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapFullFFT2Full<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapFull2FullFFT(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, Elements(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapFull2FullFFTKernel << <grid, TpB >> > (d_input, d_intermediate, make_uint3(dims.x, dims.y, dims.z), Elements(dims));

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, Elements(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapFull2FullFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapFull2FullFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapFull2FullFFT<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapHalfFFT2FullFFT(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, Elements(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapHalfFFT2FullFFTKernel << <grid, TpB >> > (d_input, d_intermediate, dims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, Elements(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapHalfFFT2FullFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapHalfFFT2FullFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapHalfFFT2FullFFT<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapHalfFFT2Half(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, ElementsFFT(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapHalfFFT2HalfKernel << <grid, TpB >> > (d_input, d_intermediate, dims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, ElementsFFT(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapHalfFFT2Half<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapHalfFFT2Half<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapHalfFFT2Half<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapHalf2HalfFFT(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, ElementsFFT(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapHalf2HalfFFTKernel << <grid, TpB >> > (d_input, d_intermediate, dims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, ElementsFFT(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapHalf2HalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapHalf2HalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapHalf2HalfFFT<int>(int* d_input, int* d_output, int3 dims, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims)
	{
		d_input += Elements(dims) * blockIdx.z;
		d_output += Elements(dims) * blockIdx.z;

		for (uint x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			uint rx = (x + (dims.x / 2)) % dims.x;
			uint ry = ((blockIdx.x + ((dims.y + 1) / 2)) % dims.y);
			uint rz = ((blockIdx.y + ((dims.z + 1) / 2)) % dims.z);

			d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(blockIdx.y * dims.y + blockIdx.x) * dims.x + rx];
		}
	}

	template <class T> __global__ void RemapFullFFT2HalfFFTKernel(T* d_input, T* d_output, int3 dims)
	{
		d_input += Elements(dims) * blockIdx.z;
		d_output += ElementsFFT(dims) * blockIdx.z;

		for (uint x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			uint rx = x;
			uint ry = blockIdx.x;
			uint rz = blockIdx.y;

			d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(rz * dims.y + ry) * dims.x + rx];
		}
	}

	template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, uint3 dims, uint elements)
	{
		uint ry = FFTShift(blockIdx.x, dims.x);
		uint rz = FFTShift(blockIdx.y, dims.z);

		d_output += elements * blockIdx.z + (rz * dims.y + ry) * dims.x;
		d_input += elements * blockIdx.z + (blockIdx.y * dims.y + blockIdx.x) * dims.x;

		for (uint x = threadIdx.x; x < dims.x; x += blockDim.x)
		{
			uint rx = FFTShift(x, dims.x);
			d_output[rx] = d_input[x];
		}
	}

	template <class T> __global__ void RemapFull2FullFFTKernel(T* d_input, T* d_output, uint3 dims, uint elements)
	{
		uint ry = IFFTShift(blockIdx.x, dims.y);
		uint rz = IFFTShift(blockIdx.y, dims.z);

		d_output += elements * blockIdx.z + (rz * dims.y + ry) * dims.x;
		d_input += elements * blockIdx.z + (blockIdx.y * dims.y + blockIdx.x) * dims.x;

		for (uint x = threadIdx.x; x < dims.x; x += blockDim.x)
		{
			uint rx = IFFTShift(x, dims.x);
			d_output[rx] = d_input[x];
		}
	}

	template <class T> __global__ void RemapHalfFFT2FullFFTKernel(T* d_input, T* d_output, int3 dims)
	{
		d_input += ElementsFFT(dims) * blockIdx.z;
		d_output += Elements(dims) * blockIdx.z;

		for (int x = threadIdx.x; x < dims.x; x += blockDim.x)
		{
			int rx = x;
			int ry = blockIdx.x;
			int rz = blockIdx.y;

			rx = rx < dims.x / 2 + 1 ? rx : rx - dims.x;
			ry = ry < dims.y / 2 + 1 ? ry : ry - dims.y;
			rz = rz < dims.z / 2 + 1 ? rz : rz - dims.z;

			if (rx < 0)
			{
				rx = -rx;
				ry = -ry;
				rz = -rz;
			}

			ry = ry > 0 ? ry : ry + dims.y;
			rz = rz > 0 ? rz : rz + dims.z;

			rx = tmax(0, tmin(rx, dims.x / 2));
			ry = tmax(0, tmin(ry, dims.y - 1));
			rz = tmax(0, tmin(rz, dims.z - 1));

			d_output[(blockIdx.y * dims.y + blockIdx.x) * dims.x + x] = d_input[(rz * dims.y + ry) * (dims.x / 2 + 1) + rx];
		}
	}

	template <class T> __global__ void RemapHalfFFT2HalfKernel(T* d_input, T* d_output, int3 dims)
	{
		d_input += ElementsFFT(dims) * blockIdx.z;
		d_output += ElementsFFT(dims) * blockIdx.z;

		int y = blockIdx.x;
		int z = blockIdx.y;

		int rz = z < dims.z / 2 + 1 ? z : z - dims.x;
		rz += dims.z / 2;
		int ry = y < dims.y / 2 + 1 ? y : y - dims.x;
		ry += dims.y / 2;

		for (uint x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			int rx = x;

			d_output[(blockIdx.y * dims.y + blockIdx.x) * (dims.x / 2 + 1) + x] = d_input[(rz * dims.y + ry) * (dims.x / 2 + 1) + rx];
		}
	}

	template <class T> __global__ void RemapHalf2HalfFFTKernel(T* d_input, T* d_output, int3 dims)
	{
		d_input += ElementsFFT(dims) * blockIdx.z;
		d_output += ElementsFFT(dims) * blockIdx.z;

		int y = blockIdx.x;
		int z = blockIdx.y;

		int rz = z < dims.z / 2 + 1 ? z : z - dims.x;
		rz += dims.z / 2;
		int ry = y < dims.y / 2 + 1 ? y : y - dims.x;
		ry += dims.y / 2;

		for (uint x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			int rx = x;

			d_output[(blockIdx.y * dims.y + blockIdx.x) * (dims.x / 2 + 1) + x] = d_input[(rz * dims.y + ry) * (dims.x / 2 + 1) + rx];
		}
	}
}