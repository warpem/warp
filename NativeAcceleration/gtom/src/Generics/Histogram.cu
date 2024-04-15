#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"


namespace gtom
{
	///////////////////////////
	//CUDA kernel declaration//
	///////////////////////////

	template<class T, int maxbins, int subdivs> __global__ void HistogramKernel(T* d_input, uint* d_histogram, size_t elements, int nbins, T minval, tfloat binsize);

	/////////////
	//Histogram//
	/////////////

	template<class T> void d_Histogram(T* d_input, uint* d_histogram, size_t elements, int nbins, T minval, T maxval, int batch)
	{
		int TpB = tmin((size_t)160, NextMultipleOf(elements, 32));
		int totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)2048);
		dim3 grid = dim3(min(totalblocks, 2048), 1, batch);

		uint* d_temp = CudaMallocValueFilled(totalblocks * nbins * batch, (uint)0);

		if (nbins <= 16)
			HistogramKernel<T, 16, 96> << <grid, TpB >> > (d_input, d_temp, elements, nbins, minval, ((tfloat)maxval - (tfloat)minval) / (tfloat)nbins);
		else if (nbins <= 32)
			HistogramKernel<T, 32, 48> << <grid, TpB >> > (d_input, d_temp, elements, nbins, minval, ((tfloat)maxval - (tfloat)minval) / (tfloat)nbins);
		else if (nbins <= 64)
			HistogramKernel<T, 64, 24> << <grid, TpB >> > (d_input, d_temp, elements, nbins, minval, ((tfloat)maxval - (tfloat)minval) / (tfloat)nbins);
		else if (nbins <= 128)
			HistogramKernel<T, 128, 12> << <grid, TpB >> > (d_input, d_temp, elements, nbins, minval, ((tfloat)maxval - (tfloat)minval) / (tfloat)nbins);
		else if (nbins <= 256)
			HistogramKernel<T, 256, 6> << <grid, TpB >> > (d_input, d_temp, elements, nbins, minval, ((tfloat)maxval - (tfloat)minval) / (tfloat)nbins);
		else if (nbins <= 512)
			HistogramKernel<T, 512, 3> << <grid, TpB >> > (d_input, d_temp, elements, nbins, minval, ((tfloat)maxval - (tfloat)minval) / (tfloat)nbins);
		else if (nbins <= 1024)
			HistogramKernel<T, 1024, 1> << <grid, TpB >> > (d_input, d_temp, elements, nbins, minval, ((tfloat)maxval - (tfloat)minval) / (tfloat)nbins);
		else if (nbins <= 2048)
			HistogramKernel<T, 2048, 1> << <grid, TpB >> > (d_input, d_temp, elements, nbins, minval, ((tfloat)maxval - (tfloat)minval) / (tfloat)nbins);
		else if (nbins <= 4096)
			HistogramKernel<T, 4096, 1> << <grid, TpB >> > (d_input, d_temp, elements, nbins, minval, ((tfloat)maxval - (tfloat)minval) / (tfloat)nbins);

		d_ReduceAdd(d_temp, d_histogram, nbins, totalblocks, batch);
		cudaFree(d_temp);
	}
	template void d_Histogram<float>(float* d_input, uint* d_histogram, size_t elements, int nbins, float minval, float maxval, int batch);
	template void d_Histogram<double>(double* d_input, uint* d_histogram, size_t elements, int nbins, double minval, double maxval, int batch);
	template void d_Histogram<int>(int* d_input, uint* d_histogram, size_t elements, int nbins, int minval, int maxval, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template<class T, int maxbins, int subdivs> __global__ void HistogramKernel(T* d_input, uint* d_histogram, size_t elements, int nbins, T minval, tfloat binsize)
	{
		__shared__ uint localhist[maxbins * subdivs];

		int threadgroup = threadIdx.x % subdivs;

		for (int i = threadIdx.x; i < nbins * subdivs; i += blockDim.x)
			localhist[i] = 0;

		d_input += elements * blockIdx.z;
		d_histogram += nbins * (blockIdx.z * gridDim.x + blockIdx.x);

		__syncthreads();

		int bin;

		for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			bin = (int)((tfloat)(d_input[i] - minval) / binsize + (tfloat)0.5);
			if (bin >= 0 && bin < nbins)
				atomicAdd(localhist + nbins * threadgroup + bin, 1);
		}

		__syncthreads();

		for (int i = threadIdx.x; i < nbins; i += blockDim.x)
			for (int b = 1; b < subdivs; b++)
				localhist[i] += localhist[nbins * b + i];

		__syncthreads();

		for (int i = threadIdx.x; i < nbins; i += blockDim.x)
			d_histogram[i] = localhist[i];
	}
}