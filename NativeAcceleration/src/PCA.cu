#include "include/Functions.h"
#include "gtom/include/DeviceFunctions.cuh"
using namespace gtom;

#define TOMO_THREADS 128

__global__ void PCALeastSqKernel(float* d_result,
								float2* d_experimental,
								float* d_ctf,
								float* d_spectral,
								int dimdata,
								uint dimdataft,
								int elementsdata,
								float rmax2,
								float2* d_shifts,
								float3* d_angles,
								glm::mat2 magnification,
								cudaTex t_reference1Re,
								cudaTex t_reference1Im,
								int dimprojector,
								int ntilts);


__declspec(dllexport) void PCALeastSq(float* h_result,
									  float2* d_experimental,
									  float* d_ctf,
									  float* d_spectral,
									  int dimdata,
									  float rmax,
									  float2* h_shifts,
									  float3* h_angles,
									  float4 magnification,
									  cudaTex t_volumeRe,
									  cudaTex t_volumeIm,
									  int dimprojector,
									  int nparticles,
									  int ntilts)
{
	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, nparticles * ntilts * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, nparticles * ntilts * sizeof(float3));
	float* d_result;
	cudaMalloc((void**)&d_result, nparticles * sizeof(float));

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	glm::mat2 m_magnification;
	m_magnification[0][0] = magnification.x;
	m_magnification[0][1] = magnification.y;
	m_magnification[1][0] = magnification.z;
	m_magnification[1][1] = magnification.w;

	PCALeastSqKernel << <Grid, TpB >> > (d_result,
										d_experimental,
										d_ctf,
										d_spectral,
										dimdata,
										ElementsFFT1(dimdata),
										elementsdata,
										rmax * rmax,
										d_shifts,
										d_angles,
										m_magnification,
										t_volumeRe,
										t_volumeIm,
										dimprojector,
										ntilts);

	cudaMemcpy(h_result, d_result, nparticles * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(d_result);
	cudaFree(d_angles);
	cudaFree(d_shifts);
}

__global__ void PCALeastSqKernel(float* d_result,
								float2* d_experimental,
								float* d_ctf,
								float* d_spectral,
								int dimdata,
								uint dimdataft,
								int elementsdata,
								float rmax2,
								float2* d_shifts,
								float3* d_angles,
								glm::mat2 magnification,
								cudaTex t_reference1Re,
								cudaTex t_reference1Im,
								int dimprojector,
								int ntilts)
{
	__shared__ float s_diff_09[128];
	__shared__ float s_diff_10[128];
	__shared__ float s_diff_11[128];
	//__shared__ float s_weight[128];

	d_experimental += blockIdx.x * elementsdata * ntilts;
	d_ctf += blockIdx.x * elementsdata * ntilts;
	//d_spectral += blockIdx.x * elementsdata * ntilts;

	cudaTex referenceRe = t_reference1Re;
	cudaTex referenceIm = t_reference1Im;

	float diff_09 = 0, diff_10 = 0, diff_11 = 0;

	for (int t = 0; t < ntilts; t++)
	{
		float2 shift = d_shifts[blockIdx.x * ntilts + t];
		glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x * ntilts + t]);

		for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
		{
			uint idx = id % dimdataft;
			uint idy = id / dimdataft;
			int x = idx;
			int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

			if (x * x + y * y >= rmax2 || id == 0)
			{
				//d_experimental[id] = make_float2(0, 0);
				continue;
			}

			glm::vec2 posmag = magnification * glm::vec2(x, y);

			float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, 0), angles, shift, dimdata);
			float ctf = d_ctf[id];
			float spectral = d_spectral == NULL ? 1.0f : d_spectral[id];

			//d_experimental[id] = val;
			//continue;
			val *= ctf;
			val *= spectral;

			float2 part = d_experimental[id];
			//part *= ctf;
			part *= spectral;


			float2 diff = (val * 0.9f) - part;
			diff_09 += diff.x * diff.x + diff.y * diff.y;

			diff = val - part;
			diff_10 += diff.x * diff.x + diff.y * diff.y;

			diff = (val * 1.1f) - part;
			diff_11 += diff.x * diff.x + diff.y * diff.y;
		}

		d_experimental += elementsdata;
		d_ctf += elementsdata;
		//d_spectral += elementsdata;
	}

	s_diff_09[threadIdx.x] = diff_09;
	s_diff_10[threadIdx.x] = diff_10;
	s_diff_11[threadIdx.x] = diff_11;

	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < 128; i++)
		{
			diff_09 += s_diff_09[i];
			diff_10 += s_diff_10[i];
			diff_11 += s_diff_11[i];
			//weight += s_weight[i];
		}

		const float x1 = 0.9f, x2 = 1.0f, x3 = 1.1f;
		const float y1 = diff_09 / (elementsdata * ntilts), 
					y2 = diff_10 / (elementsdata * ntilts), 
					y3 = diff_11 / (elementsdata * ntilts);

		float denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
		float A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
		float B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom;
		// C value no longer used since yv calculation is commented out
		// float C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

		float xv = -B / (2 * A);
		//float yv = C - B * B / (4 * A);

		d_result[blockIdx.x] = xv;
		//s_ref2[0] = ref2;
	}
}