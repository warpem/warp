#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Generics.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T, uint blockSize, bool nIsPow2> __global__ void SumMinMaxKernel(T* d_input, T* d_sum, T* d_min, T* d_max, size_t n);


	//////////
	//Common//
	//////////

	void SumMinMaxGetNumBlocksAndThreads(size_t n, int &blocks, int &threads, int maxblocks)
	{
		//get device capability, to avoid block/grid size excceed the upbound
		cudaDeviceProp prop;
		int device;
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&prop, device);

		size_t maxthreads = 512;
		threads = (int)((n < maxthreads * 2) ? NextPow2((n + 1) / 2) : maxthreads);
		size_t totalblocks = (n + (threads * 2 - 1)) / (threads * 2);
		totalblocks = tmin((size_t)maxblocks, totalblocks);
		blocks = (int)totalblocks;
	}

	/////////////
	//SumMinMax//
	/////////////

	template <class T> void SumMinMaxReduce(T* d_input, T* d_sum, T* d_min, T* d_max, size_t n, int blocks, int threads)
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
				SumMinMaxKernel<T, 512, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 256:
				SumMinMaxKernel<T, 256, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 128:
				SumMinMaxKernel<T, 128, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 64:
				SumMinMaxKernel<T, 64, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 32:
				SumMinMaxKernel<T, 32, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 16:
				SumMinMaxKernel<T, 16, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case  8:
				SumMinMaxKernel<T, 8, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case  4:
				SumMinMaxKernel<T, 4, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case  2:
				SumMinMaxKernel<T, 2, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case  1:
				SumMinMaxKernel<T, 1, true> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
		}
		else
			switch (threads)
		{
			case 512:
				SumMinMaxKernel<T, 512, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 256:
				SumMinMaxKernel<T, 256, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 128:
				SumMinMaxKernel<T, 128, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 64:
				SumMinMaxKernel<T, 64, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 32:
				SumMinMaxKernel<T, 32, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case 16:
				SumMinMaxKernel<T, 16, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case  8:
				SumMinMaxKernel<T, 8, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case  4:
				SumMinMaxKernel<T, 4, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case  2:
				SumMinMaxKernel<T, 2, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
			case  1:
				SumMinMaxKernel<T, 1, false> << <dimGrid, dimBlock, smemSize >> > (d_input, d_sum, d_min, d_max, n); break;
		}
		cudaStreamQuery(0);
	}

	template <class T> void d_SumMinMax(T* d_input, T* d_sum, T* d_min, T* d_max, size_t n, int batch)
	{
		int maxblocks = 512;
		int numblocks = 0;
		int numthreads = 0;
		SumMinMaxGetNumBlocksAndThreads(n, numblocks, numthreads, maxblocks);

		T *d_intermediateSum, *d_intermediateMin, *d_intermediateMax;
		cudaMalloc((void**)&d_intermediateSum, numblocks * sizeof(T));
		cudaMalloc((void**)&d_intermediateMin, numblocks * sizeof(T));
		cudaMalloc((void**)&d_intermediateMax, numblocks * sizeof(T));

		T* h_intermediateSum = (T*)malloc(numblocks * sizeof(T));
		T* h_intermediateMin = (T*)malloc(numblocks * sizeof(T));
		T* h_intermediateMax = (T*)malloc(numblocks * sizeof(T));

		for (int b = 0; b < batch; b++)
		{
			SumMinMaxReduce<T>(d_input + (n * (size_t)b), d_intermediateSum, d_intermediateMin, d_intermediateMax, n, numblocks, numthreads);

			cudaMemcpy(h_intermediateSum, d_intermediateSum, numblocks * sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_intermediateMin, d_intermediateMin, numblocks * sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_intermediateMax, d_intermediateMax, numblocks * sizeof(T), cudaMemcpyDeviceToHost);

			T resultSum = (T)0;
			T resultMin = (T)INT_MAX;
			T resultMax = (T)INT_MIN;

			T c = 0, y, t;
			for (int i = 0; i < numblocks; i++)
			{
				if (h_intermediateMin[i] < resultMin)
					resultMin = h_intermediateMin[i];
				if (h_intermediateMax[i] > resultMax)
					resultMax = h_intermediateMax[i];

				y = h_intermediateSum[i] - c;
				t = resultSum + y;
				c = (t - resultSum) - y;
				resultSum = t;
			}

			cudaMemcpy(d_sum + b, &resultSum, sizeof(T), cudaMemcpyHostToDevice);
			cudaMemcpy(d_min + b, &resultMin, sizeof(T), cudaMemcpyHostToDevice);
			cudaMemcpy(d_max + b, &resultMax, sizeof(T), cudaMemcpyHostToDevice);
		}

		free(h_intermediateSum);
		free(h_intermediateMin);
		free(h_intermediateMax);
		cudaFree(d_intermediateSum);
		cudaFree(d_intermediateMin);
		cudaFree(d_intermediateMax);
	}
	template void d_SumMinMax<float>(float* d_input, float* d_sum, float* d_min, float* d_max, size_t n, int batch);
	template void d_SumMinMax<double>(double* d_input, double* d_sum, double* d_min, double* d_max, size_t n, int batch);
	template void d_SumMinMax<int>(int* d_input, int* d_sum, int* d_min, int* d_max, size_t n, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	//Modified version of the reduce kernel from CUDA SDK 5.5
	template <class T, uint blockSize, bool nIsPow2> __global__ void SumMinMaxKernel(T* d_input, T* d_sum, T* d_min, T* d_max, size_t n)
	{
		__shared__ T sdata[blockSize];
		__shared__ T mindata[blockSize];
		__shared__ T maxdata[blockSize];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		uint tid = threadIdx.x;
		size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
		uint gridSize = blockSize * 2 * gridDim.x;

		T myMin = (T)INT_MAX;
		T myMax = (T)INT_MIN;
		T mySum = (T)0;
		T c = 0, y, t, val;

		// we reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i < n)
		{
			val = d_input[i];
			if (val < myMin)
				myMin = val;
			if (val > myMax)
				myMax = val;

			y = val - c;
			t = mySum + y;
			c = (t - mySum) - y;
			mySum = t;

			// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize < n)
			{
				val = d_input[i + blockSize];
				if (val < myMin)
					myMin = val;
				if (val > myMax)
					myMax = val;

				y = val - c;
				t = mySum + y;
				c = (t - mySum) - y;
				mySum = t;
			}

			i += gridSize;
		}

		// each thread puts its local Min into shared memory
		sdata[tid] = mySum;
		mindata[tid] = myMin;
		maxdata[tid] = myMax;
		__syncthreads();

		//Sum
		{
			if (blockSize >= 512)
			{
				if (tid < 256)
				{
					sdata[tid] = mySum = mySum + sdata[tid + 256];
				}

				__syncthreads();
			}

			if (blockSize >= 256)
			{
				if (tid < 128)
				{
					sdata[tid] = mySum = mySum + sdata[tid + 128];
				}

				__syncthreads();
			}

			if (blockSize >= 128)
			{
				if (tid < 64)
				{
					sdata[tid] = mySum = mySum + sdata[tid + 64];
				}

				__syncthreads();
			}

			if (tid < 32)
			{
				T* smem = sdata;

				if (blockSize >= 64)
				{
					smem[tid] = mySum = mySum + smem[tid + 32];
					__syncthreads();
				}

				if (blockSize >= 32)
				{
					smem[tid] = mySum = mySum + smem[tid + 16];
					__syncthreads();
				}

				if (blockSize >= 16)
				{
					smem[tid] = mySum = mySum + smem[tid + 8];
					__syncthreads();
				}

				if (blockSize >= 8)
				{
					smem[tid] = mySum = mySum + smem[tid + 4];
					__syncthreads();
				}

				if (blockSize >= 4)
				{
					smem[tid] = mySum = mySum + smem[tid + 2];
					__syncthreads();
				}

				if (blockSize >= 2)
				{
					smem[tid] = mySum = mySum + smem[tid + 1];
					__syncthreads();
				}
			}
		}

		if (tid == 0)
			d_sum[blockIdx.x] = sdata[0];
		__syncthreads();

		//Min
		{
			if (blockSize >= 512)
			{
				if (tid < 256)
				{
					if (mindata[tid + 256] < myMin)
						mindata[tid] = myMin = mindata[tid + 256];
				}

				__syncthreads();
			}

			if (blockSize >= 256)
			{
				if (tid < 128)
				{
					if (mindata[tid + 128] < myMin)
						mindata[tid] = myMin = mindata[tid + 128];
				}

				__syncthreads();
			}

			if (blockSize >= 128)
			{
				if (tid < 64)
				{
					if (mindata[tid + 64] < myMin)
						mindata[tid] = myMin = mindata[tid + 64];
				}

				__syncthreads();
			}

			if (tid < 32)
			{

				if (blockSize >= 64)
				{
					if (mindata[tid + 32] < myMin)
						mindata[tid] = myMin = mindata[tid + 32];
					__syncthreads();
				}

				if (blockSize >= 32)
				{
					if (mindata[tid + 16] < myMin)
						mindata[tid] = myMin = mindata[tid + 16];
					__syncthreads();
				}

				if (blockSize >= 16)
				{
					if (mindata[tid + 8] < myMin)
						mindata[tid] = myMin = mindata[tid + 8];
					__syncthreads();
				}

				if (blockSize >= 8)
				{
					if (mindata[tid + 4] < myMin)
						mindata[tid] = myMin = mindata[tid + 4];
					__syncthreads();
				}

				if (blockSize >= 4)
				{
					if (mindata[tid + 2] < myMin)
						mindata[tid] = myMin = mindata[tid + 2];
					__syncthreads();
				}

				if (blockSize >= 2)
				{
					if (mindata[tid + 1] < myMin)
						mindata[tid] = myMin = mindata[tid + 1];
					__syncthreads();
				}
			}
		}

		if (tid == 0)
			d_min[blockIdx.x] = mindata[0];
		__syncthreads();

		//Max
		{
			if (blockSize >= 512)
			{
				if (tid < 256)
				{
					if (maxdata[tid + 256] > myMax)
						maxdata[tid] = myMax = maxdata[tid + 256];
				}

				__syncthreads();
			}

			if (blockSize >= 256)
			{
				if (tid < 128)
				{
					if (maxdata[tid + 128] > myMax)
						maxdata[tid] = myMax = maxdata[tid + 128];
				}

				__syncthreads();
			}

			if (blockSize >= 128)
			{
				if (tid <  64)
				{
					if (maxdata[tid + 64] > myMax)
						maxdata[tid] = myMax = maxdata[tid + 64];
				}

				__syncthreads();
			}

			if (tid < 32)
			{

				if (blockSize >= 64)
				{
					if (maxdata[tid + 32] > myMax)
						maxdata[tid] = myMax = maxdata[tid + 32];
					__syncthreads();
				}

				if (blockSize >= 32)
				{
					if (maxdata[tid + 16] > myMax)
						maxdata[tid] = myMax = maxdata[tid + 16];
					__syncthreads();
				}

				if (blockSize >= 16)
				{
					if (maxdata[tid + 8] > myMax)
						maxdata[tid] = myMax = maxdata[tid + 8];
					__syncthreads();
				}

				if (blockSize >= 8)
				{
					if (maxdata[tid + 4] > myMax)
						maxdata[tid] = myMax = maxdata[tid + 4];
					__syncthreads();
				}

				if (blockSize >= 4)
				{
					if (maxdata[tid + 2] > myMax)
						maxdata[tid] = myMax = maxdata[tid + 2];
					__syncthreads();
				}

				if (blockSize >= 2)
				{
					if (maxdata[tid + 1] > myMax)
						maxdata[tid] = myMax = maxdata[tid + 1];
					__syncthreads();
				}
			}
		}

		if (tid == 0)
			d_max[blockIdx.x] = maxdata[0];
	}
}