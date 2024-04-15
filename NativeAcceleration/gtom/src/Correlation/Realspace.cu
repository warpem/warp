#include "gtom/include/Prerequisites.cuh"

namespace gtom
{
	//////////////////////////////
	// CUDA kernel declarations //
	//////////////////////////////

	__global__ void CorrelateRealspaceKernel(tfloat* d_image1, tfloat* d_image2, int elements, tfloat* d_mask, tfloat* d_corr);


	////////////////////////////////////////////////////////////////
	// Performs masked, locally normalized real-space correlation //
	////////////////////////////////////////////////////////////////

	void d_CorrelateRealspace(tfloat* d_image1, tfloat* d_image2, int3 dims, tfloat* d_mask, tfloat* d_corr, uint batch)
	{
		int TpB = 128;
		dim3 grid = dim3(batch, 1, 1);
		CorrelateRealspaceKernel <<<grid, TpB>>> (d_image1, d_image2, Elements(dims), d_mask, d_corr);
	}


	//////////////////
	// CUDA kernels //
	//////////////////

	__global__ void CorrelateRealspaceKernel(tfloat* d_image1, tfloat* d_image2, int elements, tfloat* d_mask, tfloat* d_corr)
	{
		__shared__ tfloat s_i1s1[128];
		__shared__ tfloat s_i1s2[128];
		__shared__ tfloat s_i2s1[128];
		__shared__ tfloat s_i2s2[128];
		__shared__ tfloat s_samples[128];

		d_image1 += elements * blockIdx.x;
		d_image2 += elements * blockIdx.x;
		d_mask += elements * blockIdx.x;

		double i1s1 = 0;
		double i1s2 = 0;
		double i2s1 = 0;
		double i2s2 = 0;
		double samples = 0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
		{
			tfloat mask = d_mask[i];
			
			tfloat image1 = d_image1[i];
			i1s1 += image1 * mask;
			i1s2 += image1 * image1 * mask;

			tfloat image2 = d_image2[i];
			i2s1 += image2 * mask;
			i2s2 += image2 * image2 * mask;

			samples += mask;
		}

		s_i1s1[threadIdx.x] = (tfloat)i1s1;
		s_i1s2[threadIdx.x] = (tfloat)i1s2;
		s_i2s1[threadIdx.x] = (tfloat)i2s1;
		s_i2s2[threadIdx.x] = (tfloat)i2s2;
		s_samples[threadIdx.x] = (tfloat)samples;
		__syncthreads();
		
		if (threadIdx.x == 0)
		{
			for (int i = 1; i < 128; i++)
			{
				i1s1 += s_i1s1[i];
				i1s2 += s_i1s2[i];
				i2s1 += s_i2s1[i];
				i2s2 += s_i2s2[i];
				samples += s_samples[i];
			}

			s_i1s1[0] = i1s1 / samples;	// mean1
			s_i1s1[1] = i2s1 / samples; // mean2
			s_i1s1[2] = sqrt((samples * i1s2 - (i1s1 * i1s1))) / samples;	// std1
			s_i1s1[3] = sqrt((samples * i2s2 - (i2s1 * i2s1))) / samples;	// std2
		}
		__syncthreads();

		tfloat mean1 = s_i1s1[0];
		tfloat mean2 = s_i1s1[1];
		tfloat std1 = 1 / s_i1s1[2];
		tfloat std2 = 1 / s_i1s1[3];

		__syncthreads();

		double corrsum = 0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
		{
			tfloat mask = d_mask[i];

			tfloat image1 = (d_image1[i] - mean1) * std1;
			tfloat image2 = (d_image2[i] - mean2) * std2;

			corrsum += image1 * image2 * mask;
		}

		s_i1s1[threadIdx.x] = corrsum;

		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < 128; i++)
				corrsum += s_i1s1[i];

			d_corr[blockIdx.x] = corrsum / samples;
		}
	}
}