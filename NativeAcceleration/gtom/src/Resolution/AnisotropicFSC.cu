#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Masking.cuh"
#include "gtom/include/Resolution.cuh"


namespace gtom
{
	///////////////////////////
	//CUDA kernel declaration//
	///////////////////////////

	__global__ void AnisotropicFSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, uint sidelength, uint sidelengthft, int maxradius, float3 direction, float coneangle, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2);


	/////////////////////////////
	//Fourier Shell Correlation//
	/////////////////////////////

	void d_AnisotropicFSC(tcomplex* d_volumeft1, tcomplex* d_volumeft2, int3 dimsvolume, tfloat* d_curve, int maxradius, float3 direction, float coneangle, tfloat* d_outnumerators, tfloat* d_outdenominators1, tfloat* d_outdenominators2, int batch)
	{
		if (dimsvolume.x != dimsvolume.y || (dimsvolume.z > 1 && dimsvolume.x != dimsvolume.z))
			throw;

		uint TpB = min(256, ElementsFFT(dimsvolume));
		dim3 grid = dim3(min((ElementsFFT(dimsvolume) + TpB - 1) / TpB, 128), batch);

		tfloat *d_nums, *d_denoms1, *d_denoms2;
		cudaMalloc((void**)&d_nums, maxradius * grid.x * batch * sizeof(tfloat));
		cudaMalloc((void**)&d_denoms1, maxradius * grid.x * batch * sizeof(tfloat));
		cudaMalloc((void**)&d_denoms2, maxradius * grid.x * batch * sizeof(tfloat));

		AnisotropicFSCKernel << <grid, TpB >> > (d_volumeft1, d_volumeft2, dimsvolume.x, dimsvolume.x / 2 + 1, maxradius, direction, coneangle, d_nums, d_denoms1, d_denoms2);

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

		tfloat* h_curve = (tfloat*)MallocFromDeviceArray(d_curve, maxradius * sizeof(tfloat));
		free(h_curve);

		cudaFree(d_reddenoms2);
		cudaFree(d_reddenoms1);
		cudaFree(d_rednums);
	}

	void d_AnisotropicFSCMap(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_map, int2 anglesteps, int maxradius, T_FSC_MODE fscmode, tfloat threshold, cufftHandle* plan, int batch)
	{
		cufftHandle localplanforw;
		if (plan == NULL)
			localplanforw = d_FFTR2CGetPlan(DimensionCount(dimsvolume), dimsvolume, batch);

		tcomplex* d_volumeft1;
		cudaMalloc((void**)&d_volumeft1, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex));
		d_FFTR2C(d_volume1, d_volumeft1, &localplanforw);

		tcomplex* d_volumeft2;
		cudaMalloc((void**)&d_volumeft2, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex));
		d_FFTR2C(d_volume2, d_volumeft2, &localplanforw);

		float phistep = PI2 / (float)max(anglesteps.x - 1, 1);
		float thetastep = PIHALF / (float)max(anglesteps.y - 1, 1);

		tfloat* d_curve = CudaMallocValueFilled(maxradius * batch, (tfloat)0);
		tfloat* d_maptemp = CudaMallocValueFilled(batch * sizeof(tfloat), (tfloat)0);

		for (int idtheta = 0; idtheta < anglesteps.y; idtheta++)
		{
			float theta = (float)idtheta * thetastep;
			float z = cos(theta);

			for (int idphi = 0; idphi < anglesteps.x; idphi++)
			{
				float phi = (dimsvolume.z == 1 ? ToRad(-90.0f) : 0.0f) + (float)idphi * phistep;
				float x = dimsvolume.z == 1 ? 0.0f : -cos(phi) * sin(theta);
				float y = sin(phi) * (dimsvolume.z == 1 ? 1.0f : sin(theta));
				if (dimsvolume.z == 1)
					x = -cos(phi);

				d_AnisotropicFSC(d_volumeft1, d_volumeft2, dimsvolume, d_curve, maxradius, make_float3(x, y, z), tmin(ToRad(85.0f), tmin(phistep * 2.0f, thetastep * 2.0f)), NULL, NULL, NULL, batch);

				/*tfloat* d_masktemp = CudaMallocValueFilled(ElementsFFT(dimsvolume), (tfloat)1);
				d_ConeMaskFT(d_masktemp, d_masktemp, dimsvolume, make_float3(x, y, z), min(phistep / 2.0f, thetastep / 2.0f));
				d_WriteMRC(d_masktemp, toInt3(dimsvolume.x / 2 + 1, dimsvolume.y, dimsvolume.z), "d_mask.mrc");*/

				d_FirstIndexOf(d_curve, d_maptemp, maxradius, threshold, T_INTERP_LINEAR, batch);

				CudaMemcpyStrided(d_map + idtheta * anglesteps.x + idphi, d_maptemp, batch, anglesteps.x * anglesteps.y, 1);
			}
		}

		cudaFree(d_maptemp);
		cudaFree(d_curve);
		cudaFree(d_volumeft1);
		cudaFree(d_volumeft2);
		if (plan == NULL)
			cufftDestroy(localplanforw);
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void AnisotropicFSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, uint sidelength, uint sidelengthft, int maxradius, float3 direction, float coneangle, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2)
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
			if (i == 0)
			{
				nums[i] = 1;
				denoms1[i] = 1;
				denoms2[i] = 1;
			}
			else
			{
				nums[i] = 0;
				denoms1[i] = 0;
				denoms2[i] = 0;
			}
		}
		__syncthreads();

		int maxradius2 = maxradius * maxradius;
		uint halfminusone = sidelength / 2 - 1;

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

			float dotprod = dotp(make_float3(rx, ry, rz), direction);
			float3 posondir = make_float3(direction.x * dotprod, direction.y * dotprod, direction.z * dotprod);
			float conewidth = tan(coneangle) * abs(dotprod);
			float3 postocone = make_float3(posondir.x - rx, posondir.y - ry, posondir.z - rz);
			float fractionofcone = sqrt(dotp(postocone, postocone)) / max(10.0f, conewidth) / 1.2f;
			float weight = max(0.0f, min(1.0f, fractionofcone));
			weight = (cos(weight * PI) + 1.0f) / 2.0f;

			tfloat radius = sqrt((tfloat)radius2);

			int radiuslow = (int)radius;
			int radiushigh = min(maxradius - 1, radiuslow + 1);
			tfloat frachigh = radius - (tfloat)radiuslow;
			tfloat fraclow = (tfloat)1 - frachigh;

			tcomplex val1 = d_volume1[id] * weight;
			tfloat denomsval = val1.x * val1.x + val1.y * val1.y;
			atomicAdd(denoms1 + radiuslow, denomsval * fraclow);
			atomicAdd(denoms1 + radiushigh, denomsval * frachigh);

			tcomplex val2 = d_volume2[id] * weight;
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