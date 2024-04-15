#include "gtom/include/Prerequisites.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void Polynomial1DKernel(tfloat* d_x, tfloat* d_output, int npoints, tfloat* d_factors, int degree);


	////////////////////////////////////////
	//Calculate polynomial at given points//
	////////////////////////////////////////

	void d_Polynomial1D(tfloat* d_x, tfloat* d_output, int npoints, tfloat* d_factors, int degree, int batch)
	{
		dim3 TpB = dim3(min(256, NextMultipleOf(npoints, 32)));
		dim3 grid = dim3((npoints + TpB.x - 1) / TpB.x, batch);
		Polynomial1DKernel << <grid, TpB >> > (d_x, d_output, npoints, d_factors, degree);
	}


	__global__ void Polynomial1DKernel(tfloat* d_x, tfloat* d_output, int npoints, tfloat* d_factors, int degree)
	{
		d_x += npoints * blockIdx.y;
		d_output += npoints * blockIdx.y;
		d_factors += degree * blockIdx.y;

		__shared__ tfloat s_factors[1024];
		for (int i = threadIdx.x; i < degree; i += blockDim.x)
			s_factors[i] = d_factors[i];
		__syncthreads();

		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < npoints; idx++)
		{
			tfloat val = s_factors[0];
			tfloat x = d_x[idx];
			tfloat lastx = x;
			for (int f = 1; f < degree; f++)
			{
				val += lastx * s_factors[f];
				lastx *= x;
			}

			d_output[idx] = val;
		}
	}
}