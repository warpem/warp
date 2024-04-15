#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Resolution.cuh"


namespace gtom
{
	///////////////////////////
	//CUDA kernel declaration//
	///////////////////////////

	__global__ void FSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, uint sidelength, uint sidelengthft, int maxradius, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2);


	/////////////////////////////
	//Fourier Shell Correlation//
	/////////////////////////////

	void d_FSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_curve, int maxradius, cufftHandle* plan, int batch)
	{
		cufftHandle localplanforw;
		if (plan == NULL)
			localplanforw = d_FFTR2CGetPlan(DimensionCount(dimsvolume), dimsvolume, batch);
		else
			localplanforw = *plan;

		tcomplex* d_volumeft1;
		cudaMalloc((void**)&d_volumeft1, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex));
		d_FFTR2C(d_volume1, d_volumeft1, &localplanforw);

		tcomplex* d_volumeft2;
		cudaMalloc((void**)&d_volumeft2, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex));
		d_FFTR2C(d_volume2, d_volumeft2, &localplanforw);

		d_FSC(d_volumeft1, d_volumeft2, dimsvolume, d_curve, maxradius, NULL, NULL, NULL, batch);

		cudaFree(d_volumeft1);
		cudaFree(d_volumeft2);
		if (plan == NULL)
			cufftDestroy(localplanforw);
	}

	void d_FSC(tcomplex* d_volumeft1, tcomplex* d_volumeft2, int3 dimsvolume, tfloat* d_curve, int maxradius, tfloat* d_outnumerators, tfloat* d_outdenominators1, tfloat* d_outdenominators2, int batch)
	{
		uint TpB = min(256, ElementsFFT(dimsvolume));
		dim3 grid = dim3(min((ElementsFFT(dimsvolume) + TpB - 1) / TpB, 128), batch);

		tfloat *d_nums, *d_denoms1, *d_denoms2;
		cudaMalloc((void**)&d_nums, maxradius * grid.x * batch * sizeof(tfloat));
		cudaMalloc((void**)&d_denoms1, maxradius * grid.x * batch * sizeof(tfloat));
		cudaMalloc((void**)&d_denoms2, maxradius * grid.x * batch * sizeof(tfloat));

		FSCKernel << <grid, TpB >> > (d_volumeft1, d_volumeft2, dimsvolume.x, dimsvolume.x / 2 + 1, maxradius, d_nums, d_denoms1, d_denoms2);

		tfloat *d_rednums, *d_reddenoms1, *d_reddenoms2;
		cudaMalloc((void**)&d_rednums, maxradius * batch * sizeof(tfloat));
		cudaMalloc((void**)&d_reddenoms1, maxradius * batch * sizeof(tfloat));
		cudaMalloc((void**)&d_reddenoms2, maxradius * batch * sizeof(tfloat));

		d_ReduceAdd(d_nums, d_rednums, maxradius, grid.x, batch);
		d_ReduceAdd(d_denoms1, d_reddenoms1, maxradius, grid.x, batch);
		d_ReduceAdd(d_denoms2, d_reddenoms2, maxradius, grid.x, batch);

		cudaFree(d_denoms2);
		cudaFree(d_denoms1);
		cudaFree(d_nums);

		if (d_outnumerators != NULL)
			cudaMemcpy(d_outnumerators, d_rednums, maxradius * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		if (d_outdenominators1 != NULL)
			cudaMemcpy(d_outdenominators1, d_reddenoms1, maxradius * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		if (d_outdenominators2 != NULL)
			cudaMemcpy(d_outdenominators2, d_reddenoms2, maxradius * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);

		d_MultiplyByVector(d_reddenoms1, d_reddenoms2, d_reddenoms1, maxradius * batch);
		d_Sqrt(d_reddenoms1, d_reddenoms1, maxradius * batch);
		d_DivideSafeByVector(d_rednums, d_reddenoms1, d_curve, maxradius * batch);

		cudaFree(d_reddenoms2);
		cudaFree(d_reddenoms1);
		cudaFree(d_rednums);
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void FSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, uint sidelength, uint sidelengthft, int maxradius, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2)
	{
		__shared__ tfloat nums[512];
		__shared__ tfloat denoms1[512];
		__shared__ tfloat denoms2[512];

		uint elementsslice = sidelengthft * sidelength;
		uint elementscube = elementsslice * sidelength;

		d_volume1 += elementscube * blockIdx.y;
		d_volume2 += elementscube * blockIdx.y;

		for (int i = threadIdx.x; i < maxradius; i += blockDim.x)
		{
			if (i > 0)
			{
				nums[i] = 0;
				denoms1[i] = 0;
				denoms2[i] = 0;
			}
			else
			{
				nums[i] = 1;
				denoms1[i] = 1;
				denoms2[i] = 1;
			}
		}
		__syncthreads();

		int maxradius2 = maxradius * maxradius;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elementscube; id += gridDim.x * blockDim.x)
		{
			uint idz = id / elementsslice;
			uint idy = (id - idz * elementsslice) / sidelengthft;
			uint idx = id % sidelengthft;

			int rx = FFTShift((int)idx, sidelength) - sidelength / 2;
			int ry = FFTShift((int)idy, sidelength) - sidelength / 2;
			int rz = FFTShift((int)idz, sidelength) - sidelength / 2;
			int radius2 = rx * rx + ry * ry + rz * rz;
			if (radius2 >= maxradius2 || radius2 == 0)
				continue;

			tfloat radius = sqrt((tfloat)radius2);

			int radiuslow = (int)radius;
			int radiushigh = min(maxradius - 1, radiuslow + 1);
			tfloat frachigh = radius - (tfloat)radiuslow;
			tfloat fraclow = (tfloat)1 - frachigh;

			tcomplex val1 = d_volume1[id];
			tfloat denomsval = val1.x * val1.x + val1.y * val1.y;
			atomicAdd(denoms1 + radiuslow, denomsval * fraclow);
			atomicAdd(denoms1 + radiushigh, denomsval * frachigh);

			tcomplex val2 = d_volume2[id];
			denomsval = val2.x * val2.x + val2.y * val2.y;
			atomicAdd(denoms2 + radiuslow, denomsval * fraclow);
			atomicAdd(denoms2 + radiushigh, denomsval * frachigh);

			denomsval = val1.x * val2.x + val1.y * val2.y;
			atomicAdd(nums + radiuslow, denomsval * fraclow);
			atomicAdd(nums + radiushigh, denomsval * frachigh);
		}
		__syncthreads();

		d_nums += maxradius * (blockIdx.y * gridDim.x + blockIdx.x);
		d_denoms1 += maxradius * (blockIdx.y * gridDim.x + blockIdx.x);
		d_denoms2 += maxradius * (blockIdx.y * gridDim.x + blockIdx.x);

		for (int i = threadIdx.x; i < maxradius; i += blockDim.x)
		{
			d_nums[i] = nums[i];
			d_denoms1[i] = denoms1[i];
			d_denoms2[i] = denoms2[i];
		}
	}
}