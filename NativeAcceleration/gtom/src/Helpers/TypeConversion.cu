#include "gtom/include/Prerequisites.cuh"

namespace gtom
{
#ifndef BlockSize
#define BlockSize 1024
#endif


	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class Tfrom, class Tto> __global__ void ConvertToKernel(Tfrom* d_original, Tto* d_copy, size_t n);
	template <class T> __global__ void ConvertSplitComplexToTComplexKernel(T* d_originalr, T* d_originali, tcomplex* d_copy, size_t n);
	template <class T> __global__ void ConvertTComplexToSplitComplexKernel(tcomplex* d_original, T* d_copyr, T* d_copyi, size_t n);
	template <class T> __global__ void ReKernel(tcomplex* d_input, T* d_output, size_t n);
	template <class T> __global__ void ImKernel(tcomplex* d_input, T* d_output, size_t n);


	////////////////////
	//Host conversions//
	////////////////////

	template <class T> void ConvertToTFloat(T* original, tfloat* copy, size_t n)
	{
		#pragma omp for schedule(dynamic, BlockSize)
		for (int i = 0; i < n; i++)
			copy[i] = (tfloat)original[i];
	}
	template void ConvertToTFloat<double>(double* original, tfloat* copy, size_t n);
	template void ConvertToTFloat<float>(float* original, tfloat* copy, size_t n);
	template void ConvertToTFloat<int>(int* original, tfloat* copy, size_t n);
	template void ConvertToTFloat<uint>(uint* original, tfloat* copy, size_t n);
	template void ConvertToTFloat<short>(short* original, tfloat* copy, size_t n);
	template void ConvertToTFloat<ushort>(ushort* original, tfloat* copy, size_t n);

	template <class T> tfloat* ConvertToTFloat(T* original, size_t n)
	{
		tfloat* converted = (tfloat*)malloc(n * sizeof(tfloat));
		ConvertToTFloat<T>(original, converted, n);

		return converted;
	}
	template tfloat* ConvertToTFloat<double>(double* original, size_t n);
	template tfloat* ConvertToTFloat<float>(float* original, size_t n);
	template tfloat* ConvertToTFloat<int>(int* original, size_t n);
	template tfloat* ConvertToTFloat<uint>(uint* original, size_t n);
	template tfloat* ConvertToTFloat<short>(short* original, size_t n);
	template tfloat* ConvertToTFloat<ushort>(ushort* original, size_t n);

	template <class T> void ConvertTFloatTo(tfloat* original, T* copy, size_t n)
	{
		#pragma omp for schedule(dynamic, BlockSize)
		for (int i = 0; i < n; i++)
			copy[i] = (T)original[i];
	}
	template void ConvertTFloatTo<double>(tfloat* original, double* copy, size_t n);
	template void ConvertTFloatTo<float>(tfloat* original, float* copy, size_t n);

	template <class T> T* ConvertTFloatTo(tfloat* original, size_t n)
	{
		T* converted = (T*)malloc(n * sizeof(T));
		ConvertTFloatTo<T>(original, converted, n);

		return converted;
	}
	template double* ConvertTFloatTo<double>(tfloat* original, size_t n);
	template float* ConvertTFloatTo<float>(tfloat* original, size_t n);

	template <class T> void ConvertSplitComplexToTComplex(T* originalr, T* originali, tcomplex* copy, size_t n)
	{
		#pragma omp for schedule(dynamic, BlockSize)
		for (int i = 0; i < n; i++)
		{
			copy[i].x = (tfloat)originalr[i];
			copy[i].y = (tfloat)originali[i];
		}
	}
	template void ConvertSplitComplexToTComplex<double>(double* originalr, double* originali, tcomplex* copy, size_t n);
	template void ConvertSplitComplexToTComplex<float>(float* originalr, float* originali, tcomplex* copy, size_t n);

	template <class T> tcomplex* ConvertSplitComplexToTComplex(T* originalr, T* originali, size_t n)
	{
		tcomplex* converted = (tcomplex*)malloc(n * sizeof(tcomplex));
		ConvertSplitComplexToTComplex(originalr, originali, converted, n);

		return converted;
	}
	template tcomplex* ConvertSplitComplexToTComplex<double>(double* originalr, double* originali, size_t n);
	template tcomplex* ConvertSplitComplexToTComplex<float>(float* originalr, float* originali, size_t n);

	template <class T> void ConvertTComplexToSplitComplex(tcomplex* original, T* copyr, T* copyi, size_t n)
	{
		#pragma omp for schedule(dynamic, BlockSize)
		for (int i = 0; i < n; i++)
		{
			copyr[i] = (T)original[i].x;
			copyi[i] = (T)original[i].y;
		}
	}
	template void ConvertTComplexToSplitComplex<double>(tcomplex* original, double* copyr, double* copyi, size_t n);
	template void ConvertTComplexToSplitComplex<float>(tcomplex* original, float* copyr, float* copyi, size_t n);

	template <class T> T* ConvertTComplexToSplitComplex(tcomplex* original, size_t n)
	{
		T* converted = (T*)malloc(n * 2 * sizeof(T));
		ConvertTComplexToSplitComplex<T>(original, converted, converted + n, n);

		return converted;
	}
	template double* ConvertTComplexToSplitComplex<double>(tcomplex* original, size_t n);
	template float* ConvertTComplexToSplitComplex<float>(tcomplex* original, size_t n);


	//////////////////////
	//Device conversions//
	//////////////////////

	template <class T> void d_ConvertToTFloat(T* d_original, tfloat* d_copy, size_t n)
	{
		size_t TpB = tmin((size_t)256, NextMultipleOf(n, 32));
		size_t totalblocks = tmin((n + TpB - 1) / TpB, (size_t)128);
		dim3 grid = dim3((uint)totalblocks);
		ConvertToKernel<T, tfloat> << <grid, (uint)TpB >> > (d_original, d_copy, n);
	}
	template void d_ConvertToTFloat<double>(double* d_original, tfloat* d_copy, size_t n);
	template void d_ConvertToTFloat<float>(float* d_original, tfloat* d_copy, size_t n);
	template void d_ConvertToTFloat<half>(half* d_original, tfloat* d_copy, size_t n);
	template void d_ConvertToTFloat<int>(int* d_original, tfloat* d_copy, size_t n);
	template void d_ConvertToTFloat<uint>(uint* d_original, tfloat* d_copy, size_t n);
	template void d_ConvertToTFloat<short>(short* d_original, tfloat* d_copy, size_t n);
	template void d_ConvertToTFloat<ushort>(ushort* d_original, tfloat* d_copy, size_t n);

	template <class T> void d_ConvertTFloatTo(tfloat* d_original, T* d_copy, size_t n)
	{
		size_t TpB = tmin((size_t)256, NextMultipleOf(n, 32));
		size_t totalblocks = tmin((n + TpB - 1) / TpB, (size_t)128);
		dim3 grid = dim3((uint)totalblocks);
		ConvertToKernel<tfloat, T> << <grid, (uint)TpB >> > (d_original, d_copy, n);
	}
	template void d_ConvertTFloatTo<double>(tfloat* d_original, double* d_copy, size_t n);
	template void d_ConvertTFloatTo<float>(tfloat* d_original, float* d_copy, size_t n);
	template void d_ConvertTFloatTo<half>(tfloat* d_original, half* d_copy, size_t n);

	template <class T> void d_ConvertSplitComplexToTComplex(T* d_originalr, T* d_originali, tcomplex* d_copy, size_t n)
	{
		size_t TpB = tmin((size_t)256, NextMultipleOf(n, 32));
		size_t totalblocks = tmin((n + TpB - 1) / TpB, (size_t)128);
		dim3 grid = dim3((uint)totalblocks);
		ConvertSplitComplexToTComplexKernel<T> << <grid, (uint)TpB >> > (d_originalr, d_originali, d_copy, n);
	}
	template void d_ConvertSplitComplexToTComplex<double>(double* d_originalr, double* d_originali, tcomplex* d_copy, size_t n);
	template void d_ConvertSplitComplexToTComplex<float>(float* d_originalr, float* d_originali, tcomplex* d_copy, size_t n);

	template <class T> void d_ConvertTComplexToSplitComplex(tcomplex* d_original, T* d_copyr, T* d_copyi, size_t n)
	{
		size_t TpB = tmin((size_t)256, NextMultipleOf(n, 32));
		size_t totalblocks = tmin((n + TpB - 1) / TpB, (size_t)128);
		dim3 grid = dim3((uint)totalblocks);
		ConvertTComplexToSplitComplexKernel<T> << <grid, (uint)TpB >> > (d_original, d_copyr, d_copyi, n);
	}
	template void d_ConvertTComplexToSplitComplex<double>(tcomplex* d_original, double* d_copyr, double* d_copyi, size_t n);
	template void d_ConvertTComplexToSplitComplex<float>(tcomplex* d_original, float* d_copyr, float* d_copyi, size_t n);

	template <class T> void d_Re(tcomplex* d_input, T* d_output, size_t n)
	{
		size_t TpB = tmin((size_t)256, NextMultipleOf(n, 32));
		size_t totalblocks = tmin((n + TpB - 1) / TpB, (size_t)128);
		dim3 grid = dim3((uint)totalblocks);
		ReKernel<T> << <grid, (uint)TpB >> > (d_input, d_output, n);
		cudaStreamQuery(0);
	}
	template void d_Re<tfloat>(tcomplex* d_input, tfloat* d_output, size_t n);

	template <class T> void d_Im(tcomplex* d_input, T* d_output, size_t n)
	{
		size_t TpB = tmin((size_t)256, NextMultipleOf(n, 32));
		size_t totalblocks = tmin((n + TpB - 1) / TpB, (size_t)128);
		dim3 grid = dim3((uint)totalblocks);
		ImKernel<T> << <grid, (uint)TpB >> > (d_input, d_output, n);
	}
	template void d_Im<tfloat>(tcomplex* d_input, tfloat* d_output, size_t n);


	///////////////////////////////////////
	//CUDA kernels for device conversions//
	///////////////////////////////////////

	template <class Tfrom, class Tto> __global__ void ConvertToKernel(Tfrom* d_original, Tto* d_copy, size_t n)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < n;
			id += blockDim.x * gridDim.x)
			d_copy[id] = (Tto)d_original[id];
	}

	template<> __global__ void ConvertToKernel<float, half>(float* d_original, half* d_copy, size_t n)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < n;
			id += blockDim.x * gridDim.x)
			d_copy[id] = __float2half(d_original[id]);
	}

	template<> __global__ void ConvertToKernel<half, float>(half* d_original, float* d_copy, size_t n)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < n;
			id += blockDim.x * gridDim.x)
			d_copy[id] = __half2float(d_original[id]);
	}

	template<> __global__ void ConvertToKernel<double, half>(double* d_original, half* d_copy, size_t n)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < n;
			id += blockDim.x * gridDim.x)
			d_copy[id] = __float2half((float)d_original[id]);
	}

	template<> __global__ void ConvertToKernel<half, double>(half* d_original, double* d_copy, size_t n)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < n;
			id += blockDim.x * gridDim.x)
			d_copy[id] = (double)__half2float(d_original[id]);
	}

	template <class T> __global__ void ConvertSplitComplexToTComplexKernel(T* d_originalr, T* d_originali, tcomplex* d_copy, size_t n)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < n;
			id += blockDim.x * gridDim.x)
		{
			d_copy[id].x = (tfloat)d_originalr[id];
			d_copy[id].y = (tfloat)d_originali[id];
		}
	}

	template <class T> __global__ void ConvertTComplexToSplitComplexKernel(tcomplex* d_original, T* d_copyr, T* d_copyi, size_t n)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < n;
			id += blockDim.x * gridDim.x)
		{
			d_copyr[id] = (T)d_original[id].x;
			d_copyi[id] = (T)d_original[id].y;
		}
	}

	template <class T> __global__ void ReKernel(tcomplex* d_input, T* d_output, size_t n)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < n;
			id += blockDim.x * gridDim.x)
			d_output[id] = (T)d_input[id].x;
	}

	template <class T> __global__ void ImKernel(tcomplex* d_input, T* d_output, size_t n)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < n;
			id += blockDim.x * gridDim.x)
			d_output[id] = (T)d_input[id].y;
	}
}
