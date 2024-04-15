#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Masking.cuh"


namespace gtom
{
	__global__ void DoseFilterKernel(tfloat* d_freq, tfloat* d_output, float2* d_doserange, tfloat3 nikoconst, float voltagescaling, uint length);

	/////////////////////////////////////////////
	//Multiplies input by dose-dependent weight//
	/////////////////////////////////////////////

	void d_DoseFilter(tfloat* d_freq, tfloat* d_output, uint length, float2* h_doserange, tfloat3 nikoconst, float voltagescaling, uint batch)
	{
		float2* d_doserange = (float2*)CudaMallocFromHostArray(h_doserange, batch * sizeof(float2));

		int TpB = tmin(128, NextMultipleOf(length, 32));
		dim3 grid = dim3((length + TpB - 1) / TpB, batch, 1);
		DoseFilterKernel << <grid, TpB >> > (d_freq, d_output, d_doserange, nikoconst, voltagescaling, length);

		cudaFree(d_doserange);
	}

	__global__ void DoseFilterKernel(tfloat* d_freq, tfloat* d_output, float2* d_doserange, tfloat3 nikoconst, float voltagescaling, uint length)
	{
		d_output += blockIdx.y * length;
		float2 doserange = d_doserange[blockIdx.y];

		for (uint i = blockIdx.x * blockDim.x + threadIdx.x;
			i < length;
			i += gridDim.x * blockDim.x)
		{
			float criticaldose = (nikoconst.x * pow(d_freq[i], nikoconst.y) + nikoconst.z) * voltagescaling;
			float optimaldose = 2.51284f * criticaldose;

			if (abs(doserange.y - optimaldose) < abs(doserange.x - optimaldose))
			{
				d_output[i] = exp(-0.5f * doserange.y / criticaldose);
			}
			else
			{
				d_output[i] = 1e-10f;
			}
		}
	}
}