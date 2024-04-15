#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Generics.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T, uint blockSize, bool nIsPow2> __global__ void MinKernel(T* d_input, tuple2<T, size_t>* d_output, size_t n);
	template <class T, uint blockSize, bool nIsPow2> __global__ void MinKernel(T* d_input, T* d_output, size_t n);
	template <class T, uint blockSize, bool nIsPow2> __global__ void MaxKernel(T* d_input, tuple2<T, size_t>* d_output, size_t n);
	template <class T, uint blockSize, bool nIsPow2> __global__ void MaxKernel(T* d_input, T* d_output, size_t n);


	//////////
	//Common//
	//////////

	void MinMaxGetNumBlocksAndThreads(size_t n, int &blocks, int &threads, int maxblocks)
	{
		//get device capability, to avoid block/grid size excceed the upbound
		cudaDeviceProp prop;
		int device;
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&prop, device);

		size_t maxthreads = 512;
		threads = (int)((n < maxthreads * 2) ? NextPow2((n + 1) / 2) : maxthreads);
		long long totalblocks = (n + (threads * 2 - 1)) / (threads * 2);
		totalblocks = min((long long)maxblocks, totalblocks);
		blocks = (int)totalblocks;
	}

	///////
	//Min//
	///////

	template <class T> void MinReduce(T* d_input, tuple2<T, size_t>* d_output, size_t n, int blocks, int threads)
	{
		dim3 dimBlock = dim3(threads);
		dim3 dimGrid = dim3(blocks);

		// when there is only one warp per block, we need to allocate two warps
		// worth of shared memory so that we don't index shared memory out of bounds
		int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

		// choose which of the optimized versions of reduction to launch

		if (IsPow2(n))
			switch (threads)
		{
			case 512:
				MinKernel<T, 512, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 256:
				MinKernel<T, 256, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 128:
				MinKernel<T, 128, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 64:
				MinKernel<T, 64, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 32:
				MinKernel<T, 32, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 16:
				MinKernel<T, 16, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  8:
				MinKernel<T, 8, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  4:
				MinKernel<T, 4, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  2:
				MinKernel<T, 2, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  1:
				MinKernel<T, 1, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
		}
		else
			switch (threads)
		{
			case 512:
				MinKernel<T, 512, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 256:
				MinKernel<T, 256, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 128:
				MinKernel<T, 128, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 64:
				MinKernel<T, 64, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 32:
				MinKernel<T, 32, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 16:
				MinKernel<T, 16, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  8:
				MinKernel<T, 8, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  4:
				MinKernel<T, 4, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  2:
				MinKernel<T, 2, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  1:
				MinKernel<T, 1, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
		}
		cudaStreamQuery(0);
	}

	template <class T> void MinReduce(T* d_input, T* d_output, size_t n, int blocks, int threads)
	{
		dim3 dimBlock = dim3(threads);
		dim3 dimGrid = dim3(blocks);

		// when there is only one warp per block, we need to allocate two warps
		// worth of shared memory so that we don't index shared memory out of bounds
		int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

		// choose which of the optimized versions of reduction to launch

		if (IsPow2(n))
			switch (threads)
		{
			case 512:
				MinKernel<T, 512, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 256:
				MinKernel<T, 256, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 128:
				MinKernel<T, 128, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 64:
				MinKernel<T, 64, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 32:
				MinKernel<T, 32, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 16:
				MinKernel<T, 16, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  8:
				MinKernel<T, 8, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  4:
				MinKernel<T, 4, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  2:
				MinKernel<T, 2, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  1:
				MinKernel<T, 1, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
		}
		else
			switch (threads)
		{
			case 512:
				MinKernel<T, 512, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 256:
				MinKernel<T, 256, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 128:
				MinKernel<T, 128, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 64:
				MinKernel<T, 64, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 32:
				MinKernel<T, 32, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 16:
				MinKernel<T, 16, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  8:
				MinKernel<T, 8, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  4:
				MinKernel<T, 4, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  2:
				MinKernel<T, 2, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  1:
				MinKernel<T, 1, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
		}
		cudaStreamQuery(0);
	}

	template <class T> void d_Min(T* d_input, tuple2<T, size_t>* d_output, size_t n, int batch)
	{
		int maxblocks = 512;

		for (int b = 0; b < batch; b++)
		{
			int numblocks = 0;
			int numthreads = 0;
			MinMaxGetNumBlocksAndThreads(n, numblocks, numthreads, maxblocks);

			tuple2<T, size_t>* d_intermediate;
			cudaMalloc((void**)&d_intermediate, numblocks * sizeof(tuple2<T, size_t>));
			MinReduce<T>(d_input + (n * (size_t)b), d_intermediate, n, numblocks, numthreads);

			int s = numblocks;

			/*while (s > 1)
			{
			MinMaxGetNumBlocksAndThreads(s, numblocks, numthreads, maxblocks);
			MinReduce<T>(d_intermediate, d_intermediate, s, numblocks, numthreads);
			s = (s + (numthreads * 2 - 1)) / (numthreads * 2);
			}*/

			tuple2<T, size_t> result = tuple2<T, size_t>((T)INT_MAX, 0);
			s = max(s, 1);

			// copy result from device to host
			tuple2<T, size_t>* h_intermediate = (tuple2<T, size_t>*)malloc(s * sizeof(tuple2<T, size_t>));
			cudaMemcpy(h_intermediate, d_intermediate, s * sizeof(tuple2<T, size_t>), cudaMemcpyDeviceToHost);

			for (int i = 0; i < s; i++)
				if (h_intermediate[i].t1 < result.t1)
					result = h_intermediate[i];

			free(h_intermediate);
			cudaFree(d_intermediate);

			cudaMemcpy(d_output + b, &result, sizeof(tuple2<T, size_t>), cudaMemcpyHostToDevice);
		}
	}
	template void d_Min<float>(float* d_input, tuple2<float, size_t>* d_output, size_t n, int batch);
	template void d_Min<double>(double* d_input, tuple2<double, size_t>* d_output, size_t n, int batch);
	template void d_Min<int>(int* d_input, tuple2<int, size_t>* d_output, size_t n, int batch);

	template <class T> void d_Min(T* d_input, T* d_output, size_t n, int batch)
	{
		int maxblocks = 512;

		for (int b = 0; b < batch; b++)
		{
			int numblocks = 0;
			int numthreads = 0;
			MinMaxGetNumBlocksAndThreads(n, numblocks, numthreads, maxblocks);

			T* d_intermediate;
			cudaMalloc((void**)&d_intermediate, numblocks * sizeof(T));
			MinReduce<T>(d_input + (n * (size_t)b), d_intermediate, n, numblocks, numthreads);

			int s = numblocks;

			/*while (s > 1)
			{
			MinMaxGetNumBlocksAndThreads(s, numblocks, numthreads, maxblocks);
			MinReduce<T>(d_intermediate, d_intermediate, s, numblocks, numthreads);
			s = (s + (numthreads * 2 - 1)) / (numthreads * 2);
			}*/

			T result = (T)INT_MAX;
			s = max(s, 1);

			// copy result from device to host
			T* h_intermediate = (T*)malloc(s * sizeof(T));
			cudaMemcpy(h_intermediate, d_intermediate, s * sizeof(T), cudaMemcpyDeviceToHost);

			for (int i = 0; i < s; i++)
				if (h_intermediate[i] < result)
					result = h_intermediate[i];

			free(h_intermediate);
			cudaFree(d_intermediate);

			cudaMemcpy(d_output + b, &result, sizeof(T), cudaMemcpyHostToDevice);
		}
	}
	template void d_Min<float>(float* d_input, float* d_output, size_t n, int batch);
	template void d_Min<double>(double* d_input, double* d_output, size_t n, int batch);
	template void d_Min<int>(int* d_input, int* d_output, size_t n, int batch);
	template void d_Min<char>(char* d_input, char* d_output, size_t n, int batch);


	///////
	//Max//
	///////

	template <class T> void MaxReduce(T* d_input, tuple2<T, size_t>* d_output, size_t n, int blocks, int threads)
	{
		dim3 dimBlock = dim3(threads);
		dim3 dimGrid = dim3(blocks);

		// when there is only one warp per block, we need to allocate two warps
		// worth of shared memory so that we don't index shared memory out of bounds
		int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

		// choose which of the optimized versions of reduction to launch

		if (IsPow2(n))
			switch (threads)
		{
			case 512:
				MaxKernel<T, 512, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 256:
				MaxKernel<T, 256, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 128:
				MaxKernel<T, 128, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 64:
				MaxKernel<T, 64, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 32:
				MaxKernel<T, 32, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 16:
				MaxKernel<T, 16, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  8:
				MaxKernel<T, 8, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  4:
				MaxKernel<T, 4, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  2:
				MaxKernel<T, 2, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  1:
				MaxKernel<T, 1, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
		}
		else
			switch (threads)
		{
			case 512:
				MaxKernel<T, 512, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 256:
				MaxKernel<T, 256, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 128:
				MaxKernel<T, 128, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 64:
				MaxKernel<T, 64, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 32:
				MaxKernel<T, 32, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 16:
				MaxKernel<T, 16, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  8:
				MaxKernel<T, 8, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  4:
				MaxKernel<T, 4, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  2:
				MaxKernel<T, 2, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  1:
				MaxKernel<T, 1, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
		}
		cudaStreamQuery(0);
	}

	template <class T> void MaxReduce(T* d_input, T* d_output, size_t n, int blocks, int threads)
	{
		dim3 dimBlock = dim3(threads);
		dim3 dimGrid = dim3(blocks);

		// when there is only one warp per block, we need to allocate two warps
		// worth of shared memory so that we don't index shared memory out of bounds
		int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

		// choose which of the optimized versions of reduction to launch

		if (IsPow2(n))
			switch (threads)
		{
			case 512:
				MaxKernel<T, 512, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 256:
				MaxKernel<T, 256, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 128:
				MaxKernel<T, 128, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 64:
				MaxKernel<T, 64, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 32:
				MaxKernel<T, 32, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 16:
				MaxKernel<T, 16, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  8:
				MaxKernel<T, 8, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  4:
				MaxKernel<T, 4, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  2:
				MaxKernel<T, 2, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  1:
				MaxKernel<T, 1, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
		}
		else
			switch (threads)
		{
			case 512:
				MaxKernel<T, 512, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 256:
				MaxKernel<T, 256, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 128:
				MaxKernel<T, 128, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 64:
				MaxKernel<T, 64, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 32:
				MaxKernel<T, 32, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case 16:
				MaxKernel<T, 16, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  8:
				MaxKernel<T, 8, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  4:
				MaxKernel<T, 4, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  2:
				MaxKernel<T, 2, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
			case  1:
				MaxKernel<T, 1, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_output, n); break;
		}
		cudaStreamQuery(0);
	}

	template <class T> void d_Max(T* d_input, tuple2<T, size_t>* d_output, size_t n, int batch)
	{
		int maxblocks = 512;

		for (int b = 0; b < batch; b++)
		{
			int numblocks = 0;
			int numthreads = 0;
			MinMaxGetNumBlocksAndThreads(n, numblocks, numthreads, maxblocks);

			tuple2<T, size_t>* d_intermediate;
			cudaMalloc((void**)&d_intermediate, numblocks * sizeof(tuple2<T, size_t>));
			MaxReduce<T>(d_input + (n * (size_t)b), d_intermediate, n, numblocks, numthreads);

			int s = numblocks;

			/*while (s > 1)
			{
			MinMaxGetNumBlocksAndThreads(s, numblocks, numthreads, maxblocks);
			MaxReduce<T>(d_intermediate, d_intermediate, s, numblocks, numthreads);
			s = (s + (numthreads * 2 - 1)) / (numthreads * 2);
			}*/

			tuple2<T, size_t> result = tuple2<T, size_t>((T)INT_MIN, 0);
			s = max(s, 1);

			// copy result from device to host
			tuple2<T, size_t>* h_intermediate = (tuple2<T, size_t>*)malloc(s * sizeof(tuple2<T, size_t>));
			cudaMemcpy(h_intermediate, d_intermediate, s * sizeof(tuple2<T, size_t>), cudaMemcpyDeviceToHost);

			for (int i = 0; i < s; i++)
				if (h_intermediate[i].t1 > result.t1)
					result = h_intermediate[i];

			free(h_intermediate);
			cudaFree(d_intermediate);

			cudaMemcpy(d_output + b, &result, sizeof(tuple2<T, size_t>), cudaMemcpyHostToDevice);
		}
	}
	template void d_Max<float>(float* d_input, tuple2<float, size_t>* d_output, size_t n, int batch);
	template void d_Max<double>(double* d_input, tuple2<double, size_t>* d_output, size_t n, int batch);
	template void d_Max<int>(int* d_input, tuple2<int, size_t>* d_output, size_t n, int batch);
	template void d_Max<char>(char* d_input, tuple2<char, size_t>* d_output, size_t n, int batch);

	template <class T> void d_Max(T* d_input, T* d_output, size_t n, int batch)
	{
		int maxblocks = 512;

		for (int b = 0; b < batch; b++)
		{
			int numblocks = 0;
			int numthreads = 0;
			MinMaxGetNumBlocksAndThreads(n, numblocks, numthreads, maxblocks);

			T* d_intermediate;
			cudaMalloc((void**)&d_intermediate, numblocks * sizeof(T));
			MaxReduce<T>(d_input + (n * (size_t)b), d_intermediate, n, numblocks, numthreads);

			int s = numblocks;

			/*while (s > 1)
			{
			MinMaxGetNumBlocksAndThreads(s, numblocks, numthreads, maxblocks);
			MaxReduce<T>(d_intermediate, d_intermediate, s, numblocks, numthreads);
			s = (s + (numthreads * 2 - 1)) / (numthreads * 2);
			}*/

			T result = (T)INT_MIN;
			s = max(s, 1);

			// copy result from device to host
			T* h_intermediate = (T*)malloc(s * sizeof(T));
			cudaMemcpy(h_intermediate, d_intermediate, s * sizeof(T), cudaMemcpyDeviceToHost);

			for (int i = 0; i < s; i++)
				if (h_intermediate[i] > result)
					result = h_intermediate[i];

			free(h_intermediate);
			cudaFree(d_intermediate);

			cudaMemcpy(d_output + b, &result, sizeof(T), cudaMemcpyHostToDevice);
		}
	}
	template void d_Max<float>(float* d_input, float* d_output, size_t n, int batch);
	template void d_Max<double>(double* d_input, double* d_output, size_t n, int batch);
	template void d_Max<int>(int* d_input, int* d_output, size_t n, int batch);
	template void d_Max<char>(char* d_input, char* d_output, size_t n, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	//Slightly modified version of the reduce kernel from CUDA SDK 5.5
	template <class T, uint blockSize, bool nIsPow2> __global__ void MinKernel(T* d_input, tuple2<T, size_t>* d_output, size_t n)
	{
		__shared__ tuple2<T, size_t> sdata[blockSize];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		uint tid = threadIdx.x;
		size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
		uint gridSize = blockSize * 2 * gridDim.x;

		tuple2<T, size_t> myMin = tuple2<T, size_t>(INT_MAX, i);
		T val;

		// we reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i < n)
		{
			if (d_input[i] < myMin.t1)
				myMin = tuple2<T, size_t>(d_input[i], i);

			// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize < n)
			{
				val = d_input[i + blockSize];
				if (val < myMin.t1)
					myMin = tuple2<T, size_t>(val, i + blockSize);
			}

			i += gridSize;
		}

		// each thread puts its local Min into shared memory
		sdata[tid] = myMin;
		__syncthreads();


		// do reduction in shared mem
		if (blockSize >= 512)
		{
			if (tid < 256)
			{
				if (sdata[tid + 256].t1 < myMin.t1)
					sdata[tid] = myMin = sdata[tid + 256];
			}

			__syncthreads();
		}

		if (blockSize >= 256)
		{
			if (tid < 128)
			{
				if (sdata[tid + 128].t1 < myMin.t1)
					sdata[tid] = myMin = sdata[tid + 128];
			}

			__syncthreads();
		}

		if (blockSize >= 128)
		{
			if (tid < 64)
			{
				if (sdata[tid + 64].t1 < myMin.t1)
					sdata[tid] = myMin = sdata[tid + 64];
			}

			__syncthreads();
		}

		if (tid < 32)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			tuple2<T, size_t>* smem = sdata;

			if (blockSize >= 64)
			{
				if (smem[tid + 32].t1 < myMin.t1)
					smem[tid] = myMin = smem[tid + 32];
				__syncthreads();
			}

			if (blockSize >= 32)
			{
				if (smem[tid + 16].t1 < myMin.t1)
					smem[tid] = myMin = smem[tid + 16];
				__syncthreads();
			}

			if (blockSize >= 16)
			{
				if (smem[tid + 8].t1 < myMin.t1)
					smem[tid] = myMin = smem[tid + 8];
				__syncthreads();
			}

			if (blockSize >= 8)
			{
				if (smem[tid + 4].t1 < myMin.t1)
					smem[tid] = myMin = smem[tid + 4];
				__syncthreads();
			}

			if (blockSize >= 4)
			{
				if (smem[tid + 2].t1 < myMin.t1)
					smem[tid] = myMin = smem[tid + 2];
				__syncthreads();
			}

			if (blockSize >= 2)
			{
				if (smem[tid + 1].t1 < myMin.t1)
					smem[tid] = myMin = smem[tid + 1];
				__syncthreads();
			}
		}

		// write result for this block to global mem
		if (tid == 0)
			d_output[blockIdx.x] = sdata[0];
	}

	//Slightly modified version of the reduce kernel from CUDA SDK 5.5
	template <class T, uint blockSize, bool nIsPow2> __global__ void MinKernel(T* d_input, T* d_output, size_t n)
	{
		__shared__ T sdata[blockSize];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		uint tid = threadIdx.x;
		size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
		uint gridSize = blockSize * 2 * gridDim.x;

		T myMin = (T)INT_MAX;
		T val;

		// we reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i < n)
		{
			if (d_input[i] < myMin)
				myMin = d_input[i];

			// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize < n)
			{
				val = d_input[i + blockSize];
				if (val < myMin)
					myMin = val;
			}

			i += gridSize;
		}

		// each thread puts its local Min into shared memory
		sdata[tid] = myMin;
		__syncthreads();


		// do reduction in shared mem
		if (blockSize >= 512)
		{
			if (tid < 256)
			{
				if (sdata[tid + 256] < myMin)
					sdata[tid] = myMin = sdata[tid + 256];
			}

			__syncthreads();
		}

		if (blockSize >= 256)
		{
			if (tid < 128)
			{
				if (sdata[tid + 128] < myMin)
					sdata[tid] = myMin = sdata[tid + 128];
			}

			__syncthreads();
		}

		if (blockSize >= 128)
		{
			if (tid < 64)
			{
				if (sdata[tid + 64] < myMin)
					sdata[tid] = myMin = sdata[tid + 64];
			}

			__syncthreads();
		}

		if (tid < 32)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			T* smem = sdata;

			if (blockSize >= 64)
			{
				if (smem[tid + 32] < myMin)
					smem[tid] = myMin = smem[tid + 32];
				__syncthreads();
			}

			if (blockSize >= 32)
			{
				if (smem[tid + 16] < myMin)
					smem[tid] = myMin = smem[tid + 16];
				__syncthreads();
			}

			if (blockSize >= 16)
			{
				if (smem[tid + 8] < myMin)
					smem[tid] = myMin = smem[tid + 8];
				__syncthreads();
			}

			if (blockSize >= 8)
			{
				if (smem[tid + 4] < myMin)
					smem[tid] = myMin = smem[tid + 4];
				__syncthreads();
			}

			if (blockSize >= 4)
			{
				if (smem[tid + 2] < myMin)
					smem[tid] = myMin = smem[tid + 2];
				__syncthreads();
			}

			if (blockSize >= 2)
			{
				if (smem[tid + 1] < myMin)
					smem[tid] = myMin = smem[tid + 1];
				__syncthreads();
			}
		}

		// write result for this block to global mem
		if (tid == 0)
			d_output[blockIdx.x] = sdata[0];
	}

	//Slightly modified version of the reduce kernel from CUDA SDK 5.5
	template <class T, uint blockSize, bool nIsPow2> __global__ void MaxKernel(T* d_input, tuple2<T, size_t>* d_output, size_t n)
	{
		__shared__ tuple2<T, size_t> sdata[blockSize];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		uint tid = threadIdx.x;
		size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
		uint gridSize = blockSize * 2 * gridDim.x;

		tuple2<T, size_t> myMax = tuple2<T, size_t>(INT_MIN, i);
		T val;

		// we reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i < n)
		{
			if (d_input[i] > myMax.t1)
				myMax = tuple2<T, size_t>(d_input[i], i);

			// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize < n)
			{
				val = d_input[i + blockSize];
				if (val > myMax.t1)
					myMax = tuple2<T, size_t>(val, i + blockSize);
			}

			i += gridSize;
		}

		// each thread puts its local Max into shared memory
		sdata[tid] = myMax;
		__syncthreads();


		// do reduction in shared mem
		if (blockSize >= 512)
		{
			if (tid < 256)
			{
				if (sdata[tid + 256].t1 > myMax.t1)
					sdata[tid] = myMax = sdata[tid + 256];
			}

			__syncthreads();
		}

		if (blockSize >= 256)
		{
			if (tid < 128)
			{
				if (sdata[tid + 128].t1 > myMax.t1)
					sdata[tid] = myMax = sdata[tid + 128];
			}

			__syncthreads();
		}

		if (blockSize >= 128)
		{
			if (tid <  64)
			{
				if (sdata[tid + 64].t1 > myMax.t1)
					sdata[tid] = myMax = sdata[tid + 64];
			}

			__syncthreads();
		}

		if (tid < 32)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			tuple2<T, size_t>* smem = sdata;

			if (blockSize >= 64)
			{
				if (smem[tid + 32].t1 > myMax.t1)
					smem[tid] = myMax = smem[tid + 32];
				__syncthreads();
			}

			if (blockSize >= 32)
			{
				if (smem[tid + 16].t1 > myMax.t1)
					smem[tid] = myMax = smem[tid + 16];
				__syncthreads();
			}

			if (blockSize >= 16)
			{
				if (smem[tid + 8].t1 > myMax.t1)
					smem[tid] = myMax = smem[tid + 8];
				__syncthreads();
			}

			if (blockSize >= 8)
			{
				if (smem[tid + 4].t1 > myMax.t1)
					smem[tid] = myMax = smem[tid + 4];
				__syncthreads();
			}

			if (blockSize >= 4)
			{
				if (smem[tid + 2].t1 > myMax.t1)
					smem[tid] = myMax = smem[tid + 2];
				__syncthreads();
			}

			if (blockSize >= 2)
			{
				if (smem[tid + 1].t1 > myMax.t1)
					smem[tid] = myMax = smem[tid + 1];
				__syncthreads();
			}
		}

		// write result for this block to global mem
		if (tid == 0)
			d_output[blockIdx.x] = sdata[0];
	}

	//Slightly modified version of the reduce kernel from CUDA SDK 5.5
	template <class T, uint blockSize, bool nIsPow2> __global__ void MaxKernel(T* d_input, T* d_output, size_t n)
	{
		__shared__ T sdata[blockSize];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		uint tid = threadIdx.x;
		size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
		uint gridSize = blockSize * 2 * gridDim.x;

		T myMax = (T)INT_MIN;
		T val;

		// we reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i < n)
		{
			if (d_input[i] > myMax)
				myMax = d_input[i];

			// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize < n)
			{
				val = d_input[i + blockSize];
				if (val > myMax)
					myMax = val;
			}

			i += gridSize;
		}

		// each thread puts its local Max into shared memory
		sdata[tid] = myMax;
		__syncthreads();


		// do reduction in shared mem
		if (blockSize >= 512)
		{
			if (tid < 256)
			{
				if (sdata[tid + 256] > myMax)
					sdata[tid] = myMax = sdata[tid + 256];
			}

			__syncthreads();
		}

		if (blockSize >= 256)
		{
			if (tid < 128)
			{
				if (sdata[tid + 128] > myMax)
					sdata[tid] = myMax = sdata[tid + 128];
			}

			__syncthreads();
		}

		if (blockSize >= 128)
		{
			if (tid <  64)
			{
				if (sdata[tid + 64] > myMax)
					sdata[tid] = myMax = sdata[tid + 64];
			}

			__syncthreads();
		}

		if (tid < 32)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			T* smem = sdata;

			if (blockSize >= 64)
			{
				if (smem[tid + 32] > myMax)
					smem[tid] = myMax = smem[tid + 32];
				__syncthreads();
			}

			if (blockSize >= 32)
			{
				if (smem[tid + 16] > myMax)
					smem[tid] = myMax = smem[tid + 16];
				__syncthreads();
			}

			if (blockSize >= 16)
			{
				if (smem[tid + 8] > myMax)
					smem[tid] = myMax = smem[tid + 8];
				__syncthreads();
			}

			if (blockSize >= 8)
			{
				if (smem[tid + 4] > myMax)
					smem[tid] = myMax = smem[tid + 4];
				__syncthreads();
			}

			if (blockSize >= 4)
			{
				if (smem[tid + 2] > myMax)
					smem[tid] = myMax = smem[tid + 2];
				__syncthreads();
			}

			if (blockSize >= 2)
			{
				if (smem[tid + 1] > myMax)
					smem[tid] = myMax = smem[tid + 1];
				__syncthreads();
			}
		}

		// write result for this block to global mem
		if (tid == 0)
			d_output[blockIdx.x] = sdata[0];
	}
}