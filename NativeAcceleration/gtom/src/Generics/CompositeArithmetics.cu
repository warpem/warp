#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Generics.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void SquaredDistanceFromVectorKernel(T* d_input, T* d_vector, T* d_output, size_t elements, int batch);
	template <class T> __global__ void SquaredDistanceFromScalarKernel(T* d_input, T* d_output, size_t elements, T scalar);
	template <class T> __global__ void SquaredDistanceFromScalarKernel(T* d_input, T* d_scalars, T* d_output, size_t elements);


	////////////////////
	//Squared Distance//
	////////////////////

	template <class T> void d_SquaredDistanceFromVector(T* d_input, T* d_vector, T* d_output, size_t elements, int batch)
	{
		size_t TpB = 256;
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)32768);
		dim3 grid = dim3((uint)totalblocks);
		SquaredDistanceFromVectorKernel<T> << <grid, (uint)TpB >> > (d_input, d_vector, d_output, elements, batch);
	}
	template void d_SquaredDistanceFromVector<tfloat>(tfloat* d_input, tfloat* d_vector, tfloat* d_output, size_t elements, int batch);
	template void d_SquaredDistanceFromVector<int>(int* d_input, int* d_vector, int* d_output, size_t elements, int batch);

	template <class T> void d_SquaredDistanceFromScalar(T* d_input, T* d_output, size_t elements, T scalar)
	{
		size_t TpB = 256;
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)32768);
		dim3 grid = dim3((uint)totalblocks);
		SquaredDistanceFromScalarKernel<T> << <grid, (uint)TpB >> > (d_input, d_output, elements, scalar);
	}
	template void d_SquaredDistanceFromScalar<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat scalar);
	template void d_SquaredDistanceFromScalar<int>(int* d_input, int* d_output, size_t elements, int scalar);

	template <class T> void d_SquaredDistanceFromScalar(T* d_input, T* d_scalars, T* d_output, size_t elements, int batch)
	{
		size_t TpB = 256;
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)32768);
		dim3 grid = dim3((uint)totalblocks, batch);
		SquaredDistanceFromScalarKernel<T> << <grid, (uint)TpB >> > (d_input, d_scalars, d_output, elements);
	}
	template void d_SquaredDistanceFromScalar<tfloat>(tfloat* d_input, tfloat* d_scalars, tfloat* d_output, size_t elements, int batch);
	template void d_SquaredDistanceFromScalar<int>(int* d_input, int* d_scalars, int* d_output, size_t elements, int batch);

	template <class T> __global__ void SquaredDistanceFromVectorKernel(T* d_input, T* d_vector, T* d_output, size_t elements, int batch)
	{
		T val, temp;
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
		{
			val = d_vector[id];
			for (size_t n = 0; n < batch; n++)
			{
				temp = d_input[id + elements * n] - val;
				d_output[id + elements * n] = temp * temp;
			}
		}
	}

	template <class T> __global__ void SquaredDistanceFromScalarKernel(T* d_input, T* d_output, size_t elements, T scalar)
	{
		T temp;
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
		{
			temp = d_input[id] - scalar;
			d_output[id] = temp * temp;
		}
	}

	template <class T> __global__ void SquaredDistanceFromScalarKernel(T* d_input, T* d_multiplicators, T* d_output, size_t elements)
	{
		__shared__ T scalar;
		if (threadIdx.x == 0)
			scalar = d_multiplicators[blockIdx.y];
		__syncthreads();

		T temp;
		size_t offset = elements * blockIdx.y;
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
		{
			temp = d_input[id + offset] - scalar;
			d_output[id + offset] = temp * temp;
		}
	}
}