#include "include/Functions.h"
using namespace gtom;

#define MULTIBODY_THREADS 128

__global__ void ParticleNMAGetDiffKernel(float2* d_experimental, float2* d_reference, float* d_ctf, float* d_invsigma2, uint length, float* d_diff, float* d_debugdiff);
__global__ void ParticleNMAMeanDisplacementKernel(float3* d_normalmodes, uint natoms, float* d_modefactors, uint nmodes, float* d_displacement);
__global__ void ParticleNMARigidTransformKernel(float3* d_positions, float3* d_normalmodes, uint natoms, float* d_modefactors, uint nmodes, float3* d_temptrans, float3* d_centertrans, glm::mat3* d_rotmat);


__declspec(dllexport) void ParticleNMAGetDiff(float2* d_experimental, 
												float2* d_reference, 
												float* d_ctf, 
												float* d_invsigma2,
												int2 dims, 
												float* h_diff,
												uint nparticles)
{
	int TpB = MULTIBODY_THREADS;
	dim3 grid = dim3(nparticles);

	float* d_diff;
	cudaMalloc((void**)&d_diff, nparticles * sizeof(float));
	
	float* d_debugdiff = NULL;
	//cudaMalloc((void**)&d_debugdiff, npositions * nframes * ElementsFFT2(dims) * sizeof(float));

	ParticleNMAGetDiffKernel <<<grid, TpB>>> (d_experimental, d_reference, d_ctf, d_invsigma2, ElementsFFT2(dims), d_diff, d_debugdiff);
	
	cudaMemcpy(h_diff, d_diff, nparticles * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_diff);
}

__global__ void ParticleNMAGetDiffKernel(float2* d_experimental, float2* d_reference, float* d_ctf, float* d_invsigma2, uint length, float* d_diff, float* d_debugdiff)
{
	__shared__ float s_num[MULTIBODY_THREADS];
	s_num[threadIdx.x] = 0.0f;

	uint specid = blockIdx.x;
	d_experimental += specid * length;
	d_reference += specid * length;
	d_ctf += specid * length;
	d_invsigma2 += specid * length;
	d_debugdiff += specid * length;

	float numsum = 0.0f;

	for (uint id = threadIdx.x; 
		 id < length; 
		 id += MULTIBODY_THREADS)
	{
		float2 experimental = d_experimental[id];
		float2 reference = d_reference[id];

		reference = reference * d_ctf[id];

		float2 diff = experimental - reference;
		numsum += dotp2(diff, diff) * d_invsigma2[id];
	}
	
	s_num[threadIdx.x] = numsum;
	__syncthreads();

	for (uint lim = 64; lim > 1; lim >>= 1)
	{
		if (threadIdx.x < lim)
		{
			numsum += s_num[threadIdx.x + lim];
			s_num[threadIdx.x] = numsum;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		numsum += s_num[1];
		d_diff[specid] = numsum * 0.5f;
	}
}

__declspec(dllexport) void ParticleNMAGetMeanDisplacement(float3* d_normalmodes, uint natoms, float* h_modefactors, uint nmodes, uint nmodels, float* h_displacement)
{
    float* d_modefactors = (float*)CudaMallocFromHostArray(h_modefactors, nmodes * nmodels * sizeof(float));
	float* d_displacement;
	cudaMalloc((void**)&d_displacement, nmodels * sizeof(float));

	dim3 grid = dim3(nmodels, 1, 1);
	ParticleNMAMeanDisplacementKernel <<<grid, 128>>> (d_normalmodes, natoms, d_modefactors, nmodes, d_displacement);

	cudaMemcpy(h_displacement, d_displacement, nmodels * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_displacement);
	cudaFree(d_modefactors);
}

__global__ void ParticleNMAMeanDisplacementKernel(float3* d_normalmodes, uint natoms, float* d_modefactors, uint nmodes, float* d_displacement)
{
    __shared__ float s_modefactors[128];
	for (int i = threadIdx.x; i < nmodes; i += blockDim.x)
		s_modefactors[i] = d_modefactors[blockIdx.x * nmodes + i];
	__syncthreads();

	__shared__ float3 s_mean[128];

	float3 mean = make_float3(0, 0, 0);
	for (int a = threadIdx.x; a < natoms; a += blockDim.x)
	{
		for (int m = 0; m < nmodes; m++)
			mean += d_normalmodes[m * natoms + a] * s_modefactors[m];
	}

	s_mean[threadIdx.x] = mean;
	__syncthreads();

	if (threadIdx.x == 0)
	{
	    for (int i = 1; i < blockDim.x; i++)
			mean += s_mean[i];
		s_mean[0] = mean / natoms;
	}
	__syncthreads();

	mean = s_mean[0] * -1;

	float disp = 0;
	for (int a = threadIdx.x; a < natoms; a += blockDim.x)
	{
		float3 atomdisp = mean;
		for (int m = 0; m < nmodes; m++)
			atomdisp += d_normalmodes[m * natoms + a] * s_modefactors[m];

		disp += dotp(atomdisp, atomdisp);
	}
	__syncthreads();

	s_modefactors[threadIdx.x] = disp;
	__syncthreads();

	if (threadIdx.x == 0)
	{
	    for (int i = 1; i < blockDim.x; i++)
			disp += s_modefactors[i];

		d_displacement[blockIdx.x] = sqrt(disp / natoms);
	}
}

__declspec(dllexport) void ParticleNMAGetRigidTransform(float3* d_positions, float3* d_normalmodes, uint natoms, float* h_modefactors, uint nmodes, uint nmodels, float3* h_centertrans, float* h_rotmat)
{
    float* d_modefactors = (float*)CudaMallocFromHostArray(h_modefactors, nmodes * nmodels * sizeof(float));
	float3* d_centertrans;
	cudaMalloc((void**)&d_centertrans, nmodels * sizeof(float3));
	glm::mat3* d_rotmat;
	cudaMalloc((void**)&d_rotmat, nmodels * sizeof(glm::mat3));
	float3* d_temptrans;
	cudaMalloc((void**)&d_temptrans, natoms * nmodels * sizeof(float3));
	glm::mat3* d_temprot;
	cudaMalloc((void**)&d_temprot, 128 * nmodels * sizeof(glm::mat3));

	dim3 grid = dim3(nmodels, 1, 1);
	ParticleNMARigidTransformKernel <<<grid, 128>>> (d_positions, d_normalmodes, natoms, d_modefactors, nmodes, d_temptrans, d_centertrans, d_temprot);

	d_ReduceAdd((float*)d_temprot, (float*)d_rotmat, 9, 128, nmodels);
	
	cudaMemcpy(h_centertrans, d_centertrans, nmodels * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rotmat, d_rotmat, nmodels * sizeof(glm::mat3), cudaMemcpyDeviceToHost);

	cudaFree(d_temprot);
	cudaFree(d_temptrans);
	cudaFree(d_rotmat);
	cudaFree(d_centertrans);
	cudaFree(d_modefactors);
}

__global__ void ParticleNMARigidTransformKernel(float3* d_positions, float3* d_normalmodes, uint natoms, float* d_modefactors, uint nmodes, float3* d_temptrans, float3* d_centertrans, glm::mat3* d_rotmat)
{
    __shared__ float s_modefactors[128];
	for (int i = threadIdx.x; i < nmodes; i += blockDim.x)
		s_modefactors[i] = d_modefactors[blockIdx.x * nmodes + i];
	__syncthreads();

	__shared__ float3 s_shared[128];
	__shared__ float3 s_centertrans[1];

	float3 centertrans = make_float3(0, 0, 0);
	for (int a = threadIdx.x; a < natoms; a += blockDim.x)
	{
		float3 pos = d_positions[a];
		for (int m = 0; m < nmodes; m++)
			pos += d_normalmodes[m * natoms + a] * s_modefactors[m];

		d_temptrans[blockIdx.x * natoms + a] = pos;
		centertrans += pos;
	}

	s_shared[threadIdx.x] = centertrans;
	__syncthreads();

	if (threadIdx.x == 0)
	{
	    for (int i = 1; i < blockDim.x; i++)
			centertrans += s_shared[i];

		s_centertrans[0] = centertrans / natoms;
		d_centertrans[blockIdx.x] = centertrans / natoms;
	}
	__syncthreads();

	centertrans = s_centertrans[0];

	glm::mat3 rotmat;
	
	for (int a = threadIdx.x; a < natoms; a += blockDim.x)
	{
		float3 ori = d_positions[a];
		float3 trans = d_temptrans[blockIdx.x * natoms + a] - centertrans;
		
		rotmat += glm::mat3(ori.x * trans.x, ori.y * trans.x, ori.z * trans.x,
							ori.x * trans.y, ori.y * trans.y, ori.z * trans.y,
							ori.x * trans.z, ori.y * trans.z, ori.z * trans.z);
	}

	d_rotmat[blockIdx.x * blockDim.x + threadIdx.x] = rotmat;

	/*__syncthreads();

	if (threadIdx.x == 0)
	{
	    for (int i = 1; i < blockDim.x; i++)
			rotmat += s_shared[i];

		d_rotmat[blockIdx.x] = rotmat;
	}*/
}