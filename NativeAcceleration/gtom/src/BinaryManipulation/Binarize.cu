#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Binary.cuh"

namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void BinarizeKernel(tfloat* d_input, T* d_output, size_t elements, tfloat threshold);


	//////////
	//Dilate//
	//////////

	template <class T> void d_Binarize(tfloat* d_input, T* d_output, size_t elements, tfloat threshold, int batch)
	{
		size_t TpB = tmin((size_t)192, NextMultipleOf(elements, 32));
		dim3 grid = dim3(tmin((elements + TpB - 1) / TpB, (size_t)32768));
		BinarizeKernel << <grid, TpB >> > (d_input, d_output, elements, threshold);
	}
	template void d_Binarize<char>(tfloat* d_input, char* d_output, size_t elements, tfloat threshold, int batch);
	template void d_Binarize<int>(tfloat* d_input, int* d_output, size_t elements, tfloat threshold, int batch);
	template void d_Binarize<float>(tfloat* d_input, float* d_output, size_t elements, tfloat threshold, int batch);
	template void d_Binarize<double>(tfloat* d_input, double* d_output, size_t elements, tfloat threshold, int batch);

	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void BinarizeKernel(tfloat* d_input, T* d_output, size_t elements, tfloat threshold)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
			d_output[id] = d_input[id] >= threshold ? (T)1 : (T)0;
	}
}