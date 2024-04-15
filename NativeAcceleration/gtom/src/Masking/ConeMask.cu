#include "gtom/include/Prerequisites.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void ConeMaskFTKernel(T* d_input, T* d_output, uint sidelength, uint sidelengthft, float3 direction, float coneangle);


	////////////////
	//Host methods//
	////////////////

	template <class T> void d_ConeMaskFT(T* d_input, T* d_output, int3 dims, float3 direction, float coneangle, int batch)
	{
		int TpB = 256;
		dim3 grid = dim3((ElementsFFT(dims) + TpB - 1) / TpB, batch);
		ConeMaskFTKernel<T> << <grid, TpB >> > (d_input, d_output, dims.x, ElementsFFT1(dims.x), direction, coneangle);
	}
	template void d_ConeMaskFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 size, float3 direction, float coneangle, int batch);
	template void d_ConeMaskFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 size, float3 direction, float coneangle, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void ConeMaskFTKernel(T* d_input, T* d_output, uint sidelength, uint sidelengthft, float3 direction, float coneangle)
	{
		uint elementsslice = sidelengthft * sidelength;
		uint elementscube = elementsslice * sidelength;

		d_input += elementscube * blockIdx.y;
		d_output += elementscube * blockIdx.y;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elementscube; id += gridDim.x * blockDim.x)
		{
			uint idz = id / elementsslice;
			uint idy = (id % elementsslice) / sidelengthft;
			uint idx = id % sidelengthft;

			int rx = FFTShift((int)idx, sidelength) - sidelength / 2;
			int ry = FFTShift((int)idy, sidelength) - sidelength / 2;
			int rz = FFTShift((int)idz, sidelength) - sidelength / 2;
			int radius2 = rx * rx + ry * ry + rz * rz;
			if (radius2 == 0)
			{
				d_output[id] = d_input[id];
				continue;
			}

			float dotprod = dotp(make_float3(rx, ry, rz), direction);
			float3 posondir = make_float3(direction.x * dotprod, direction.y * dotprod, direction.z * dotprod);
			float conewidth = tan(coneangle) * abs(dotprod);
			float3 postocone = make_float3(posondir.x - rx, posondir.y - ry, posondir.z - rz);
			float distfromcone = sqrt(dotp(postocone, postocone)) - conewidth;
			float weight = max(0.0f, min(1.0f, 1.0f - distfromcone));

			d_output[id] = d_input[id] * weight;
		}
	}
}