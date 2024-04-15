#include "include/Functions.h"
#include "gtom/include/DeviceFunctions.cuh"
using namespace gtom;

#define TOMO_THREADS 128

__global__ void MultiParticleDiffKernel(float3* d_result,
										float2* hp_experimental,
										int dimdata,
										int dimdatafull,
										uint dimdataft,
										int elementsdata,
										float2* d_shifts,
										float3* d_angles,
										glm::mat2 magnification,
										float* d_weights,
										float2* d_phasecorrection,
										float ewalddiameterinv,
										int maxshell2,
										cudaTex d_reference1Re,
										cudaTex d_reference1Im,
										cudaTex d_reference2Re,
										cudaTex d_reference2Im,
										float supersample,
										int dimprojector,
										int* d_subsets,
										int ntilts,
										int itilt);
__global__ void MultiParticleCorr2DKernel(float* d_result2d,
											float* d_result1dparticles,
											float* d_resultphaseresiduals,
											int dimresult,
											float2* hp_experimental,
											float* d_weights,
											int dimdata,
											int dimdatafull,
											uint dimdataft,
											int elementsdata,
											float scalingfactor,
											float2* d_shifts,
											float3* d_angles,
											glm::mat2 magnification,
											float ewalddiameterinv,
											cudaTex d_reference1Re,
											cudaTex d_reference1Im,
											cudaTex d_reference2Re,
											cudaTex d_reference2Im,
											float supersample,
											int dimprojector,
											int* d_subsets,
											int ntilts,
											int itilt);

void PeekLastError()
{
	cudaDeviceSynchronize();
	cudaError_t result = cudaPeekAtLastError();
	if (result != 0)
		throw;
}

__declspec(dllexport) void MultiParticleDiff(float3* h_result,
											float2** hp_experimental,
											int dimdata,
											int* h_relevantdims,
											float2* h_shifts,
											float3* h_angles,
											float4 magnification,
											float* d_weights,
											float2* d_phasecorrection,
											float ewaldradius,
											int maxshell,
											cudaTex* h_volumeRe,
											cudaTex* h_volumeIm,
											float supersample,
											int dimprojector,
											int* d_subsets,
											int nparticles,
											int ntilts)
{
	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, ntilts * nparticles * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, ntilts * nparticles * sizeof(float3));
	float3* d_result;
	cudaMalloc((void**)&d_result, nparticles * ntilts * sizeof(float3));

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	glm::mat2 m_magnification;
	m_magnification[0][0] = magnification.x;
	m_magnification[0][1] = magnification.y;
	m_magnification[1][0] = magnification.z;
	m_magnification[1][1] = magnification.w;

	ewaldradius *= supersample; 
	float ewalddiameterinv = ewaldradius == 0 ? 0 : (1.0f / (2.0f * ewaldradius));

	//PeekLastError();

	for (int i = 0; i < ntilts; i++)
	{
		int relevantdim = dimdata;
		if (h_relevantdims != NULL)
			relevantdim = h_relevantdims[i];

		MultiParticleDiffKernel << <Grid, TpB >> > (d_result,
													hp_experimental[i],
													relevantdim,
													dimdata,
													ElementsFFT1(relevantdim),
													ElementsFFT1(relevantdim) * relevantdim,
													d_shifts,
													d_angles,
													m_magnification,
													(d_weights != NULL) ? (d_weights + i * elementsdata) : (float*)NULL,
													(h_relevantdims == NULL) ? d_phasecorrection : (d_phasecorrection + i * elementsdata),
													ewalddiameterinv,
													maxshell * maxshell,
													h_volumeRe[0],
													h_volumeIm[0],
													h_volumeRe[1],
													h_volumeIm[1],
													supersample,
													dimprojector,
													d_subsets,
													ntilts,
													i);

		//PeekLastError();
	}

	cudaMemcpy(h_result, d_result, nparticles * ntilts * sizeof(float3), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(d_result);

	cudaFree(d_angles);
	cudaFree(d_shifts);
}


__declspec(dllexport) void MultiParticleCorr2D(float* d_result2d,
												float* d_result1dparticles,
												float* d_resultphaseresiduals,
												int dimresult,
												float2** hp_experimental,
												float* d_weights,
												int dimdata,
												float scalingfactor,
												int* h_relevantdims,
												float2* h_shifts,
												float3* h_angles,
												float4 magnification,
												float ewaldradius,
												cudaTex* h_volumeRe,
												cudaTex* h_volumeIm,
												float supersample,
												int dimprojector,
												int* d_subsets,
												int nparticles,
												int ntilts)
{
	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, ntilts * nparticles * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, ntilts * nparticles * sizeof(float3));

	int elementsresults = ElementsFFT1(dimresult) * dimresult;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	glm::mat2 m_magnification;
	m_magnification[0][0] = magnification.x;
	m_magnification[0][1] = magnification.y;
	m_magnification[1][0] = magnification.z;
	m_magnification[1][1] = magnification.w;

	ewaldradius *= supersample;
	float ewalddiameterinv = ewaldradius == 0 ? 0 : 1.0f / (2.0f * ewaldradius);

	for (int i = 0; i < ntilts; i++)
	{
		int relevantdim = dimdata;
		if (h_relevantdims != NULL)
			relevantdim = h_relevantdims[i];

		MultiParticleCorr2DKernel << <Grid, TpB >> > (d_result2d + elementsresults * 3 * i,
														d_result1dparticles,
														d_resultphaseresiduals,
														dimresult,
														hp_experimental[i],
														d_weights,
														relevantdim,
														dimdata,
														ElementsFFT1(relevantdim),
														ElementsFFT1(relevantdim) * relevantdim,
														scalingfactor,
														d_shifts,
														d_angles,
														m_magnification,
														ewalddiameterinv,
														h_volumeRe[0],
														h_volumeIm[0],
														h_volumeRe[1],
														h_volumeIm[1],
														supersample,
														dimprojector,
														d_subsets,
														ntilts,
														i);
	}

	cudaDeviceSynchronize();

	cudaFree(d_angles);
	cudaFree(d_shifts);
}

__global__ void MultiParticleDiffKernel(float3* d_result,
										float2* hp_experimental,
										int dimdata,
										int dimdatafull,
										uint dimdataft,
										int elementsdata,
										float2* d_shifts,
										float3* d_angles,
										glm::mat2 magnification,
										float* d_weights,
										float2* d_phasecorrection,
										float ewalddiameterinv,
										int maxshell2,
										cudaTex d_reference1Re,
										cudaTex d_reference1Im,
										cudaTex d_reference2Re,
										cudaTex d_reference2Im,
										float supersample,
										int dimprojector,
										int* d_subsets,
										int ntilts,
										int itilt)
{
	__shared__ float s_diff2[128];
	__shared__ float s_ref2[128];
	__shared__ float s_part2[128];
	//__shared__ float s_weight[128];

	hp_experimental += blockIdx.x * elementsdata;
	float2 shift = d_shifts[blockIdx.x * ntilts + itilt];
	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x * ntilts + itilt]);

	int subset = d_subsets == NULL ? 0 : d_subsets[blockIdx.x];

	cudaTex referenceRe = subset == 0 ? d_reference1Re : d_reference2Re;
	cudaTex referenceIm = subset == 0 ? d_reference1Im : d_reference2Im;

	float diff2 = 0, ref2 = 0, part2 = 0, weight = 0;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= tmin(dimdata * dimdata / 4, maxshell2))
		{
			//hp_experimental[id] = make_float2(0, 0);
			continue;
		}

		glm::vec2 posmag = magnification * glm::vec2(x, y);
		float ewaldz = 0;
		if (ewalddiameterinv != 0)
			ewaldz = ewalddiameterinv * (x * x + y * y);

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, ewaldz) * supersample, angles, shift, dimdatafull);
				
		float weight = d_weights == NULL ? 1 : d_weights[id];

		val *= weight;

		float2 part = hp_experimental[id];

		if (d_phasecorrection != NULL)
			part = cmul(part, d_phasecorrection[id]);

		float cc = dotp2(part, val);

		diff2 += cc;
		ref2 += dotp2(val, val);
		part2 += dotp2(part, part);
	}
	//return;

	s_diff2[threadIdx.x] = diff2;
	s_ref2[threadIdx.x] = ref2;
	s_part2[threadIdx.x] = part2;

	__syncthreads();

	/*for (uint lim = 64; lim > 1; lim >>= 1)
	{
		if (threadIdx.x < lim)
		{
			diff2 += s_diff2[threadIdx.x + lim];
			s_diff2[threadIdx.x] = diff2;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		diff2 += s_diff2[1];

		hp_result[blockIdx.x * ntilts + itilt] = blockIdx.x;
	}*/

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < 128; i++)
		{
			diff2 += s_diff2[i];
			ref2 += s_ref2[i];
			part2 += s_part2[i];
		}
		
		d_result[blockIdx.x * ntilts + itilt] = make_float3(diff2, ref2, part2);
	}
}

__global__ void MultiParticleCorr2DKernel(float* d_result2d,
											float* d_result1dparticles,
											float* d_resultphaseresiduals,
											int dimresult,
											float2* hp_experimental,
											float* d_weights,
											int dimdata,
											int dimdatafull,
											uint dimdataft,
											int elementsdata,
											float scalingfactor,
											float2* d_shifts,
											float3* d_angles,
											glm::mat2 magnification,
											float ewalddiameterinv,
											cudaTex d_reference1Re,
											cudaTex d_reference1Im,
											cudaTex d_reference2Re,
											cudaTex d_reference2Im,
											float supersample,
											int dimprojector,
											int* d_subsets,
											int ntilts,
											int itilt)
{
	hp_experimental += blockIdx.x * elementsdata;
	d_weights += blockIdx.x * elementsdata;
	d_result1dparticles += blockIdx.x * (dimresult / 2) * 3;

	float2 shift = d_shifts[blockIdx.x * ntilts + itilt];
	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x * ntilts + itilt]);

	cudaTex referenceRe = d_subsets[blockIdx.x] == 0 ? d_reference2Re : d_reference1Re;
	cudaTex referenceIm = d_subsets[blockIdx.x] == 0 ? d_reference2Im : d_reference1Im;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
			continue;

		glm::vec2 posmag = magnification * glm::vec2(x, y);
		float ewaldz = 0;
		if (ewalddiameterinv != 0)
			ewaldz = ewalddiameterinv * (x * x + y * y);

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, ewaldz) * supersample, angles, shift, dimdatafull);
		//float2 valb = d_GetProjectionSlice(d_reference2Re, d_reference2Im, dimprojector, glm::vec3(posmag.x, posmag.y, ewaldz) * supersample, angles, shift, dimdatafull);
		//float2 val = (vala + valb) * 0.5f;

		float2 part = hp_experimental[id];

		float cc = dotp2(part, val);
		float val2 = dotp2(val, val);
		float part2 = dotp2(part, part);

		float weight = d_weights[id];
		float2 phaseresidual = cmul(part, cconj(val * weight));

		float2 posresult = make_float2(x, y) * scalingfactor;
		float r = sqrt(dotp2(posresult, posresult));
		if (posresult.y < 0)
			posresult.y += dimresult;

		posresult.x = tmin(posresult.x, dimresult / 2);
		posresult.y = tmin(posresult.y, dimresult - 1);

		int2 posresult0 = make_int2((int)posresult.x, 
									(int)posresult.y);
		int2 posresult1 = make_int2(tmin(posresult0.x + 1, dimresult / 2), 
									tmin(posresult0.y + 1, dimresult - 1));

		// These are interpolation weights now
		posresult.x -= posresult0.x;
		posresult.y -= posresult0.y;

		int elementsresult = ElementsFFT1(dimresult) * dimresult;

		float w0 = 1.0f - posresult.y;
		float w1 = posresult.y;

		float w00 = (1.0f - posresult.x) * w0;
		float w10 = posresult.x * w0;
		float w01 = (1.0f - posresult.x) * w1;
		float w11 = posresult.x * w1;
			   
		atomicAdd((float*)(d_result2d + (posresult0.y * (dimresult / 2 + 1) + posresult0.x)), cc * w00);
		atomicAdd((float*)(d_result2d + (posresult0.y * (dimresult / 2 + 1) + posresult1.x)), cc * w01);
		atomicAdd((float*)(d_result2d + (posresult1.y * (dimresult / 2 + 1) + posresult0.x)), cc * w10);
		atomicAdd((float*)(d_result2d + (posresult1.y * (dimresult / 2 + 1) + posresult1.x)), cc * w11);

		atomicAdd((float*)(d_result2d + elementsresult + (posresult0.y * (dimresult / 2 + 1) + posresult0.x)), val2 * w00);
		atomicAdd((float*)(d_result2d + elementsresult + (posresult0.y * (dimresult / 2 + 1) + posresult1.x)), val2 * w01);
		atomicAdd((float*)(d_result2d + elementsresult + (posresult1.y * (dimresult / 2 + 1) + posresult0.x)), val2 * w10);
		atomicAdd((float*)(d_result2d + elementsresult + (posresult1.y * (dimresult / 2 + 1) + posresult1.x)), val2 * w11);

		atomicAdd((float*)(d_result2d + elementsresult * 2 + (posresult0.y * (dimresult / 2 + 1) + posresult0.x)), part2 * w00);
		atomicAdd((float*)(d_result2d + elementsresult * 2 + (posresult0.y * (dimresult / 2 + 1) + posresult1.x)), part2 * w01);
		atomicAdd((float*)(d_result2d + elementsresult * 2 + (posresult1.y * (dimresult / 2 + 1) + posresult0.x)), part2 * w10);
		atomicAdd((float*)(d_result2d + elementsresult * 2 + (posresult1.y * (dimresult / 2 + 1) + posresult1.x)), part2 * w11);


		atomicAdd((float*)(d_resultphaseresiduals + (posresult0.y * (dimresult / 2 + 1) + posresult0.x)), phaseresidual.x * w00);
		atomicAdd((float*)(d_resultphaseresiduals + (posresult0.y * (dimresult / 2 + 1) + posresult1.x)), phaseresidual.x * w01);
		atomicAdd((float*)(d_resultphaseresiduals + (posresult1.y * (dimresult / 2 + 1) + posresult0.x)), phaseresidual.x * w10);
		atomicAdd((float*)(d_resultphaseresiduals + (posresult1.y * (dimresult / 2 + 1) + posresult1.x)), phaseresidual.x * w11);

		atomicAdd((float*)(d_resultphaseresiduals + ((dimresult + posresult0.y) * (dimresult / 2 + 1) + posresult0.x)), phaseresidual.y * w00);
		atomicAdd((float*)(d_resultphaseresiduals + ((dimresult + posresult0.y) * (dimresult / 2 + 1) + posresult1.x)), phaseresidual.y * w01);
		atomicAdd((float*)(d_resultphaseresiduals + ((dimresult + posresult1.y) * (dimresult / 2 + 1) + posresult0.x)), phaseresidual.y * w10);
		atomicAdd((float*)(d_resultphaseresiduals + ((dimresult + posresult1.y) * (dimresult / 2 + 1) + posresult1.x)), phaseresidual.y * w11);


		int r0 = tmin((int)r, dimresult / 2 - 1);
		int r1 = tmin(r0 + 1, dimresult / 2 - 1);
		r -= r0;
		w0 = 1.0f - r;
		w1 = r;

		atomicAdd((float*)(d_result1dparticles + dimresult / 2 * 0 + r0), cc * w0);
		atomicAdd((float*)(d_result1dparticles + dimresult / 2 * 0 + r1), cc * w1);

		atomicAdd((float*)(d_result1dparticles + dimresult / 2 * 1 + r0), val2 * w0);
		atomicAdd((float*)(d_result1dparticles + dimresult / 2 * 1 + r1), val2 * w1);

		atomicAdd((float*)(d_result1dparticles + dimresult / 2 * 2 + r0), part2 * w0);
		atomicAdd((float*)(d_result1dparticles + dimresult / 2 * 2 + r1), part2 * w1);
	}
}

__global__ void MultiParticleSumAmplitudesKernel(float* d_result,
												int dimdata,
												uint dimdataft,
												int elementsdata,
												float3* d_angles,
												cudaTex t_referenceRe,
												cudaTex t_referenceIm,
												int dimprojector)
{
	__shared__ float s_sum2[128];
	__shared__ float s_samples[128];

	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x]);

	cudaTex referenceRe = t_referenceRe;
	cudaTex referenceIm = t_referenceIm;

	float sum2 = 0;
	int samples = 0;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
			continue;

		glm::vec2 posmag = glm::vec2(x, y);

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, 0), angles, make_float2(0, 0), dimdata);
		
		sum2 += dotp2(val, val);
		samples++;
	}

	s_sum2[threadIdx.x] = sum2;
	s_samples[threadIdx.x] = (float)samples;

	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < 128; i++)
		{
			sum2 += s_sum2[i];
			samples += s_samples[i];
		}

		d_result[blockIdx.x] = sqrt(sum2 / tmax(1e-5f, samples));
	}
}

__global__ void MultiParticleSimulateKernel(float2* d_result,
	int dimdata,
	uint dimdataft,
	int elementsdata,
	float2* d_shifts,
	float3* d_angles,
	float* d_defoci,
	CTFParamsLean ctfparams,
	float* d_weights,
	cudaTex d_reference1Re,
	cudaTex d_reference1Im,
	cudaTex d_reference2Re,
	cudaTex d_reference2Im,
	int dimprojector,
	int* d_subsets,
	int ntilts,
	int itilt)
{
	d_result += blockIdx.x * elementsdata;
	float2 shift = d_shifts[blockIdx.x * ntilts + itilt];
	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x * ntilts + itilt]);
	float defocus = d_defoci[blockIdx.x * ntilts + itilt];

	cudaTex referenceRe = d_subsets[blockIdx.x] == 0 ? d_reference1Re : d_reference2Re;
	cudaTex referenceIm = d_subsets[blockIdx.x] == 0 ? d_reference1Im : d_reference2Im;

	ctfparams.defocus = d_defoci[blockIdx.x * ntilts + itilt] * (-1e4f);

	float diff2 = 0, ref2 = 0;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
		{
			d_result[id] = make_float2(0, 0);
			continue;
		}

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(x, y, 0), angles, shift, dimdata);

		float angle = atan2f((float)y, (float)x);
		float r = sqrt((float)x * x + y * y) / dimdata;
		float pixelsize = ctfparams.pixelsize + ctfparams.pixeldelta * cosf(2.0f * (angle - ctfparams.pixelangle));
		r /= pixelsize;
		float ctf = d_GetCTF<false, false>(r, angle, 0, ctfparams);

		val *= ctf;// *d_weights[id];

		d_result[id] = val;
	}
}