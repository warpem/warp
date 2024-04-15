#include "gtom/include/Prerequisites.cuh"

namespace gtom
{
#define MonoTpB 256

	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void MinMonoKernel(T* d_input, tuple2<T, size_t>* d_output, int n, int batch);
	template <class T> __global__ void MinMonoKernel(T* d_input, T* d_output, int n, int batch);
	template <class T> __global__ void MaxMonoKernel(T* d_input, tuple2<T, size_t>* d_output, int n, int batch);
	template <class T> __global__ void MaxMonoKernel(T* d_input, T* d_output, int n, int batch);


	///////
	//Min//
	///////

	template <class T> void d_MinMonolithic(T* d_input, tuple2<T, size_t>* d_output, int n, int batch)
	{
		size_t TpB = MonoTpB;
		size_t totalblocks = min(batch, 32768);
		dim3 grid = dim3((uint)totalblocks);

		MinMonoKernel << <grid, TpB >> > (d_input, d_output, n, batch);
	}
	template void d_MinMonolithic<int>(int* d_input, tuple2<int, size_t>* d_output, int n, int batch);
	template void d_MinMonolithic<float>(float* d_input, tuple2<float, size_t>* d_output, int n, int batch);
	template void d_MinMonolithic<double>(double* d_input, tuple2<double, size_t>* d_output, int n, int batch);

	template <class T> void d_MinMonolithic(T* d_input, T* d_output, int n, int batch)
	{
		size_t TpB = MonoTpB;
		size_t totalblocks = min(batch, 32768);
		dim3 grid = dim3((uint)totalblocks);

		MinMonoKernel << <grid, TpB >> > (d_input, d_output, n, batch);
	}
	template void d_MinMonolithic<int>(int* d_input, int* d_output, int n, int batch);
	template void d_MinMonolithic<float>(float* d_input, float* d_output, int n, int batch);
	template void d_MinMonolithic<double>(double* d_input, double* d_output, int n, int batch);


	///////
	//Max//
	///////

	template <class T> void d_MaxMonolithic(T* d_input, tuple2<T, size_t>* d_output, int n, int batch)
	{
		size_t TpB = MonoTpB;
		size_t totalblocks = min(batch, 32768);
		dim3 grid = dim3((uint)totalblocks);

		MaxMonoKernel << <grid, TpB >> > (d_input, d_output, n, batch);
	}
	template void d_MaxMonolithic<int>(int* d_input, tuple2<int, size_t>* d_output, int n, int batch);
	template void d_MaxMonolithic<float>(float* d_input, tuple2<float, size_t>* d_output, int n, int batch);
	template void d_MaxMonolithic<double>(double* d_input, tuple2<double, size_t>* d_output, int n, int batch);

	template <class T> void d_MaxMonolithic(T* d_input, T* d_output, int n, int batch)
	{
		size_t TpB = MonoTpB;
		size_t totalblocks = min(batch, 32768);
		dim3 grid = dim3((uint)totalblocks);

		MaxMonoKernel << <grid, TpB >> > (d_input, d_output, n, batch);
	}
	template void d_MaxMonolithic<int>(int* d_input, int* d_output, int n, int batch);
	template void d_MaxMonolithic<float>(float* d_input, float* d_output, int n, int batch);
	template void d_MaxMonolithic<double>(double* d_input, double* d_output, int n, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void MinMonoKernel(T* d_input, tuple2<T, size_t>* d_output, int n, int batch)
	{
		__shared__ T values[MonoTpB];
		__shared__ int locations[MonoTpB];

		for (int b = blockIdx.x; b < batch; b += gridDim.x)
		{
			T* offsetinput = d_input + (size_t)n * (size_t)b;

			values[threadIdx.x] = (T)999999999;
			T value;
			for (int i = threadIdx.x; i < n; i += MonoTpB)
			{
				value = offsetinput[i];
				if (value < values[threadIdx.x])
				{
					values[threadIdx.x] = value;
					locations[threadIdx.x] = i;
				}
			}
			__syncthreads();

			if (threadIdx.x == 0)
			{
				for (int i = 1; i < MonoTpB; i++)
					if (values[i] < values[0])
					{
						values[0] = values[i];
						locations[0] = locations[i];
					}

				d_output[b].t1 = values[0];
				d_output[b].t2 = locations[0];
			}
			__syncthreads();
		}
	}

	template <class T> __global__ void MinMonoKernel(T* d_input, T* d_output, int n, int batch)
	{
		__shared__ T values[MonoTpB];

		for (int b = blockIdx.x; b < batch; b += gridDim.x)
		{
			T* offsetinput = d_input + (size_t)n * (size_t)b;

			values[threadIdx.x] = (T)999999999;
			for (int i = threadIdx.x; i < n; i += MonoTpB)
				values[threadIdx.x] = min(values[threadIdx.x], offsetinput[i]);
			__syncthreads();

			if (threadIdx.x == 0)
			{
				for (int i = 1; i < MonoTpB; i++)
					values[0] = min(values[0], values[i]);

				d_output[b] = values[0];
			}
			__syncthreads();
		}
	}

	template <class T> __global__ void MaxMonoKernel(T* d_input, tuple2<T, size_t>* d_output, int n, int batch)
	{
		__shared__ T values[MonoTpB];
		__shared__ int locations[MonoTpB];

		for (int b = blockIdx.x; b < batch; b += gridDim.x)
		{
			T* offsetinput = d_input + (size_t)n * (size_t)b;

			values[threadIdx.x] = (T)-999999999;
			T value;
			for (int i = threadIdx.x; i < n; i += MonoTpB)
			{
				value = offsetinput[i];
				if (value > values[threadIdx.x])
				{
					values[threadIdx.x] = value;
					locations[threadIdx.x] = i;
				}
			}
			__syncthreads();

			if (threadIdx.x == 0)
			{
				for (int i = 1; i < MonoTpB; i++)
					if (values[i] > values[0])
					{
						values[0] = values[i];
						locations[0] = locations[i];
					}

				d_output[b].t1 = values[0];
				d_output[b].t2 = locations[0];
			}
			__syncthreads();
		}
	}

	template <class T> __global__ void MaxMonoKernel(T* d_input, T* d_output, int n, int batch)
	{
		__shared__ T values[MonoTpB];

		for (int b = blockIdx.x; b < batch; b += gridDim.x)
		{
			T* offsetinput = d_input + (size_t)n * (size_t)b;

			values[threadIdx.x] = (T)-999999999;
			for (int i = threadIdx.x; i < n; i += MonoTpB)
				values[threadIdx.x] = max(values[threadIdx.x], offsetinput[i]);
			__syncthreads();

			if (threadIdx.x == 0)
			{
				for (int i = 1; i < MonoTpB; i++)
					values[0] = max(values[0], values[i]);

				d_output[b] = values[0];
			}
			__syncthreads();
		}
	}
}