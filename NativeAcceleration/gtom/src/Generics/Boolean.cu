#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Generics.cuh"


namespace gtom
{
#define MonoTpB 192

	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void OrMonolithicKernel(T* d_input, T* d_output, uint n);
	template <class T> __global__ void AndMonolithicKernel(T* d_input, T* d_output, uint n);


	//////
	//Or//
	//////

	template <class T> void d_Or(T* d_input, T* d_output, uint n, uint batch)
	{
		for (int b = 0; b < batch; b += 32768)
		{
			dim3 grid = dim3(min(batch - b, 32768));
			OrMonolithicKernel << <grid, MonoTpB >> > (d_input + n * b, d_output + b, n);
		}
	}
	template void d_Or<char>(char* d_input, char* d_output, uint n, uint batch);
	template void d_Or<uchar>(uchar* d_input, uchar* d_output, uint n, uint batch);
	template void d_Or<short>(short* d_input, short* d_output, uint n, uint batch);
	template void d_Or<ushort>(ushort* d_input, ushort* d_output, uint n, uint batch);
	template void d_Or<int>(int* d_input, int* d_output, uint n, uint batch);
	template void d_Or<uint>(uint* d_input, uint* d_output, uint n, uint batch);
	template void d_Or<bool>(bool* d_input, bool* d_output, uint n, uint batch);


	///////
	//And//
	///////

	template <class T> void d_And(T* d_input, T* d_output, uint n, uint batch)
	{
		for (int b = 0; b < batch; b += 32768)
		{
			dim3 grid = dim3(min(batch - b, 32768));
			AndMonolithicKernel << <grid, MonoTpB >> > (d_input + n * b, d_output + b, n);
		}
	}
	template void d_And<char>(char* d_input, char* d_output, uint n, uint batch);
	template void d_And<uchar>(uchar* d_input, uchar* d_output, uint n, uint batch);
	template void d_And<short>(short* d_input, short* d_output, uint n, uint batch);
	template void d_And<ushort>(ushort* d_input, ushort* d_output, uint n, uint batch);
	template void d_And<int>(int* d_input, int* d_output, uint n, uint batch);
	template void d_And<uint>(uint* d_input, uint* d_output, uint n, uint batch);
	template void d_And<bool>(bool* d_input, bool* d_output, uint n, uint batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void OrMonolithicKernel(T* d_input, T* d_output, uint n)
	{
		__shared__ T sums[MonoTpB];

		d_input += n * blockIdx.x;

		T result = 0;

		for (int id = threadIdx.x; id < n; id += blockDim.x)
			result |= d_input[id];

		sums[threadIdx.x] = result;

		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < blockDim.x; i++)
				result |= sums[i];

			d_output[blockIdx.x] = result;
		}
	}

	template <class T> __global__ void AndMonolithicKernel(T* d_input, T* d_output, uint n)
	{
		__shared__ T sums[MonoTpB];

		d_input += n * blockIdx.x;

		T result = 0;
		if (threadIdx.x < n)
			result = d_input[threadIdx.x];

		for (int id = threadIdx.x + blockDim.x; id < n; id += blockDim.x)
			result &= d_input[id];

		sums[threadIdx.x] = result;

		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < tmin(n, blockDim.x); i++)
				result &= sums[i];

			d_output[blockIdx.x] = result;
		}
	}
}