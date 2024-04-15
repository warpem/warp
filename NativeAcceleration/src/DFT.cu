#include "include/Functions.h"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/GTOM.cuh"

using namespace gtom;

#define DFT_THREADS 128
#define DFT_BATCH 16

void MakeSinCos(float2* d_output, int elements);

__global__ void SinCosKernel(float2* d_output,
							uint elements,
							float elementsinv);

__global__ void DFTR2CKernel(volatile float* d_input,
							float2* d_sincos,
							float2* d_output,
							uint elements,
							uint nbatches);

void DFTR2C1D(float* d_input, float2* d_sincos, float2* d_output, int elements, int batch);

__declspec(dllexport) void BenchmarkCUFFT(float* d_input, float2* d_output, cufftHandle planforw, int elements, int batch, int repeats)
{
	std::cout << (int)cudaDeviceSynchronize() << "\n";

	for (size_t r = 0; r < repeats; r++)
	{
		d_FFTR2C(d_input, d_output, &planforw);
		//cudaDeviceSynchronize();
	}

	std::cout << (int)cudaDeviceSynchronize() << "\n";
}

__declspec(dllexport) void BenchmarkDFT(float* d_input, float2* d_sincos, float2* d_output, int elements, int batch, int repeats)
{
	MakeSinCos(d_sincos, elements);

	std::cout << (int)cudaDeviceSynchronize() << "\n";

	for (size_t r = 0; r < repeats; r++)
	{
		DFTR2C1D(d_input, d_sincos, d_output, elements, batch);
		//cudaDeviceSynchronize();
	}

	std::cout << (int)cudaDeviceSynchronize() << "\n";
}

__declspec(dllexport) void BenchmarkBLASFT(void* cublas, float2* d_input, float2* d_sincos, float2* d_output, int elements, int batch, int repeats)
{
	MakeSinCos(d_sincos, elements);

	std::cout << (int)cudaDeviceSynchronize() << "\n";

	for (size_t r = 0; r < repeats; r++)
	{
		/*cublasStatus_t result = cublasCgemm((cublasHandle_t)cublas,
											CUBLAS_OP_N, 
											CUBLAS_OP_N, 
											elements / 2 + 0, 
											batch, 
											elements,
											&(make_cuComplex(1.0f, 0.0f)),
											(cuComplex*)d_sincos, 
											elements / 2 + 0, 
											(cuComplex*)d_input, 
											elements, 
											&(make_cuComplex(0.0f, 0.0f)),
											(cuComplex*)d_output, 
											elements / 2 + 0);*/
		float one = 1.0f;
		float zero = 0.0f;
		cublasStatus_t result = cublasSgemm((cublasHandle_t)cublas,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			elements / 2 + 0,
			batch,
			elements,
			&one,
			(float*)d_sincos,
			elements / 2 + 0,
			(float*)d_input,
			elements,
			&zero,
			(float*)d_output,
			elements / 2 + 0);
	}

	std::cout << (int)cudaDeviceSynchronize() << "\n";
}

void MakeSinCos(float2* d_output, int elements)
{
	dim3 TpB = dim3(DFT_THREADS, 1, 1);
	dim3 Grid = dim3(elements / 2 + 1, 1, 1);

	SinCosKernel << <Grid, TpB >> > (d_output, elements, PI2 / elements);
}

void DFTR2C1D(float* d_input, float2* d_sincos, float2* d_output, int elements, int batch)
{
	dim3 TpB = dim3(DFT_THREADS, 1, 1);
	dim3 Grid = dim3(batch / 4, 1, 1);

	DFTR2CKernel << <Grid, TpB >> > (d_input, d_sincos, d_output, elements, batch);
}

__device__ float warpReduceSum(float val) 
{
	for (int offset = 32 / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

//__global__ void DFTR2CKernel(float* d_input,
//							float2* d_output,
//							uint elements,
//							uint nbatches)
//{
//	uint elementsft = elements / 2 + 1;
//	float elementsinv = 1.0f / elements;
//
//	d_input += elements * blockIdx.x;
//	d_output += elementsft * blockIdx.x;
//
//	__shared__ float s_sums_real[DFT_THREADS / 32];
//	__shared__ float s_sums_imag[DFT_THREADS / 32];
//
//	__shared__ float s_input[256];
//	for (uint i = threadIdx.x; i < elements; i += blockDim.x)
//		s_input[i] = d_input[i];
//	__syncthreads();
//
//	for (uint k = 0; k < elementsft; k++)
//	{
//		float2 sum = make_float2(0, 0);
//
//		for (uint i = threadIdx.x; i < elements; i += blockDim.x)
//		{
//			float angle = PI2 * k * i * elementsinv;
//			float2 factor;
//			__sincosf(angle, &(factor.x), &(factor.y));
//
//			float2 sample = make_float2(s_input[i], 0);
//
//			float2 product = cuCmulf(sample, factor);
//			sum += product;
//		}
//
//		sum.x = warpReduceSum(sum.x);
//		sum.y = warpReduceSum(sum.y);
//
//		if (threadIdx.x % 32 == 0)
//		{
//			s_sums_real[threadIdx.x / 32] = sum.x;
//			s_sums_imag[threadIdx.x / 32] = sum.y;
//		}
//		__syncthreads();
//
//		if (threadIdx.x < 2)
//		{
//			s_sums_real[threadIdx.x] += s_sums_real[threadIdx.x + 2];
//			s_sums_imag[threadIdx.x] += s_sums_imag[threadIdx.x + 2];
//		}
//		__syncthreads();
//
//		if (threadIdx.x == 0)
//		{
//			d_output[k] = make_float2(s_sums_real[0] + s_sums_real[1], s_sums_imag[0] + s_sums_imag[1]);
//		}
//		__syncthreads();
//	}
//}

__global__ void SinCosKernel(float2* d_output,
							uint elements,
							float elementsinv)
{
	uint elementsft = elements / 2 + 1;

	uint k = blockIdx.x;
	float angle = k * elementsinv;

	for (uint i = threadIdx.x; i < elements; i += blockDim.x)
	{
		float2 factor;
		__sincosf(angle * i, &(factor.x), &(factor.y));

		d_output[i * elementsft + blockIdx.x] = factor;
	}
}

__global__ void DFTR2CKernel(volatile float* d_input,
							float2* d_sincos,
							float2* d_output,
							uint elements,
							uint nbatches)
{
	uint elementsft = elements / 2 + 1; 

	d_input += elements * blockIdx.x * 4;
	d_output += elementsft * blockIdx.x * 4;

	__shared__ float s_input1[256];
	for (uint i = threadIdx.x; i < elements; i += blockDim.x)
		s_input1[i] = d_input[i];

	__shared__ float s_input2[256];
	for (uint i = threadIdx.x; i < elements; i += blockDim.x)
		s_input2[i] = d_input[elements + i];

	__shared__ float s_input3[256];
	for (uint i = threadIdx.x; i < elements; i += blockDim.x)
		s_input3[i] = d_input[elements * 2 + i];

	__shared__ float s_input4[256];
	for (uint i = threadIdx.x; i < elements; i += blockDim.x)
		s_input4[i] = d_input[elements * 3 + i];
	__syncthreads();

	for (uint k = threadIdx.x; k < elementsft; k += blockDim.x)
	{
		float2 sum1 = make_float2(0, 0);
		float2 sum2 = make_float2(0, 0);
		float2 sum3 = make_float2(0, 0);
		float2 sum4 = make_float2(0, 0);

		for (uint i = 0; i < elements; i++)
		{
			float2 factor = d_sincos[i * elementsft + k];

			sum1 += s_input1[i] * factor;
			sum2 += s_input2[i] * factor;
			sum3 += s_input3[i] * factor;
			sum4 += s_input4[i] * factor;
		}
			
		d_output[k] = sum1;
		d_output[elementsft + k] = sum2;
		d_output[elementsft * 2 + k] = sum3;
		d_output[elementsft * 3 + k] = sum4;
	}
}