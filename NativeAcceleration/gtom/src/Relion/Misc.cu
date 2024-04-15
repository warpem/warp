#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Helper.cuh"
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

namespace gtom
{
	__global__ void CreateMinvsigma2sKernel(tfloat* d_output, int* d_mresol, uint elements, tfloat* d_sigma2noise, tfloat sigma2fudge);

	void d_rlnCreateMinvsigma2s(tfloat* d_output, int* d_mresol, uint elements, tfloat* d_sigma2noise, tfloat sigma2fudge)
	{
		int TpB = 128;
		dim3 grid = dim3((elements + TpB - 1) / TpB, 1, 1);
		CreateMinvsigma2sKernel <<<grid, TpB>>> (d_output, d_mresol, elements, d_sigma2noise, sigma2fudge);
	}

	__global__ void CreateMinvsigma2sKernel(tfloat* d_output, int* d_mresol, uint elements, tfloat* d_sigma2noise, tfloat sigma2fudge)
	{
		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += blockDim.x * gridDim.x)
		{
			int ires = d_mresol[id];
			if (ires > 0)
				d_output[id] = (tfloat)1 / (sigma2fudge * d_sigma2noise[ires]);
			else
				d_output[id] = 0;
		}
	}
}
