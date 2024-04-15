#include "include/Functions.h"
using namespace gtom;

#define MULTIBODY_THREADS 128

__global__ void ParticleMultibodyGetDiffKernel(float2* d_experimental, float2* d_reference, float2* d_shiftfactors, float* d_ctf, float* d_invsigma2, uint length, float2* d_shifts, uint nparticles, uint nbodies, float* d_diff, float* d_debugdiff);
__global__ void ProjectMultibodyKernel(cudaTex* d_textureRe,
										cudaTex* d_textureIm,
										int dimvolume, 
										tcomplex* d_proj, 
										int dimproj, 
										int elementsproj, 
										glm::mat2x3* d_rotations, 
										float2* d_shifts,
										float* d_globalweights,
										int nbodies,
										int rmax2);


__declspec(dllexport) void ParticleMultibodyGetDiff(float2* d_experimental, 
												float2* d_reference, 
												float2* d_shiftfactors, 
												float* d_ctf,
												float* d_invsigma2,
												int2 dims, 
												float2* h_shifts,
												float* h_diff,
												uint nparticles,
												uint nbodies)
{
	int TpB = MULTIBODY_THREADS;
	dim3 grid = dim3(nparticles);

	float* d_diff;
	cudaMalloc((void**)&d_diff, nparticles * sizeof(float));

	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, nparticles * nbodies * sizeof(float2));
	
	float* d_debugdiff = NULL;
	//cudaMalloc((void**)&d_debugdiff, npositions * nframes * ElementsFFT2(dims) * sizeof(float));

	ParticleMultibodyGetDiffKernel <<<grid, TpB>>> (d_experimental, d_reference, d_shiftfactors, d_ctf, d_invsigma2, ElementsFFT2(dims), d_shifts, nparticles, nbodies, d_diff, d_debugdiff);
	
	cudaMemcpy(h_diff, d_diff, nparticles * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_shifts);
	cudaFree(d_diff);
}

__global__ void ParticleMultibodyGetDiffKernel(float2* d_experimental, float2* d_reference, float2* d_shiftfactors, float* d_ctf, float* d_invsigma2, uint length, float2* d_shifts, uint nparticles, uint nbodies, float* d_diff, float* d_debugdiff)
{
	__shared__ float s_num[MULTIBODY_THREADS];
	s_num[threadIdx.x] = 0.0f;
	//__shared__ float s_denom1[MULTIBODY_THREADS];
	//s_denom1[threadIdx.x] = 0.0f;
	//__shared__ float s_denom2[MULTIBODY_THREADS];
	//s_denom2[threadIdx.x] = 0.0f;

	uint specid = blockIdx.x;
	d_experimental += specid * length;
	d_reference += specid * length;
	d_ctf += specid * length;
	d_invsigma2 += specid * length;
	d_shifts += specid;
	d_debugdiff += specid * length;

	float numsum = 0.0f;//, denomsum1 = 0.0f, denomsum2 = 0.0f;

	for (uint id = threadIdx.x; 
		 id < length; 
		 id += MULTIBODY_THREADS)
	{
		float2 experimental = d_experimental[id];
		float2 referencesum = make_float2(0, 0);
		//float bodyfiltersum = 0;
			
		float2 shiftfactors = d_shiftfactors[id];

		for (uint b = 0; b < nbodies; b++)
		{
			float2 reference = d_reference[b * nparticles * length + id];
			float2 shift = d_shifts[b * nparticles];

			float phase = shiftfactors.x * (-shift.x) + shiftfactors.y * (-shift.y);
			float2 change = make_float2(__cosf(phase), __sinf(phase));
			reference = cuCmulf(reference, change);

			//float bodyfilter = d_bodyfilter[b * length + id];
			referencesum += reference;// * bodyfilter;
			//bodyfiltersum += bodyfilter;
		}

		referencesum *= d_ctf[id];

		float2 diff = experimental - referencesum;
		numsum += dotp2(diff, diff) * d_invsigma2[id] * 0.5f;

		//numsum += experimental.x * referencesum.x + experimental.y * referencesum.y;
		//denomsum1 += dotp2(experimental, experimental);
		//denomsum2 += dotp2(referencesum, referencesum);
	}
	
	s_num[threadIdx.x] = numsum;
	//s_denom1[threadIdx.x] = denomsum1;
	//s_denom2[threadIdx.x] = denomsum2;
	__syncthreads();

	for (uint lim = 64; lim > 1; lim >>= 1)
	{
		if (threadIdx.x < lim)
		{
			numsum += s_num[threadIdx.x + lim];
			s_num[threadIdx.x] = numsum;
			
			//denomsum1 += s_denom1[threadIdx.x + lim];
			//s_denom1[threadIdx.x] = denomsum1;
			
			//denomsum2 += s_denom2[threadIdx.x + lim];
			//s_denom2[threadIdx.x] = denomsum2;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		numsum += s_num[1];
		//denomsum1 += s_denom1[1];
		//denomsum2 += s_denom2[1];

		d_diff[specid] = numsum;// / tmax(1e-6f, sqrt(denomsum1 * denomsum2));
	}
}

__declspec(dllexport) void ParticleMultibodyProject(uint64_t* h_textureRe, uint64_t* h_textureIm, int3 dimsvolume, tcomplex* d_proj, int2 dimsproj, float3* h_angles, float2* h_shifts, float* h_globalweights, float supersample, uint nbodies, uint batch)
{
    glm::mat2x3* h_matrices = (glm::mat2x3*)malloc(sizeof(glm::mat2x3) * nbodies * batch);
	float2* h_shiftsscaled = (float2*)malloc(nbodies * batch * sizeof(float2));
	
	#pragma omp parallel for
	for (int i = 0; i < nbodies * batch; i++)
	{
		h_matrices[i] = glm::mat2x3(glm::transpose(Matrix3Euler(h_angles[i])) * Matrix3Scale(supersample));
		h_shiftsscaled[i] = make_float2(h_shifts[i].x * PI2 / dimsproj.x,
										h_shifts[i].y * PI2 / dimsproj.x);
	}

	glm::mat2x3* d_matrices = (glm::mat2x3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat2x3) * nbodies * batch);
	free(h_matrices);

	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shiftsscaled, nbodies * batch * sizeof(float2));
	free(h_shiftsscaled);

	float* d_globalweights = (float*)CudaMallocFromHostArray(h_globalweights, nbodies * batch * sizeof(float));
	
	cudaTex* d_textureRe = (cudaTex*)CudaMallocFromHostArray(h_textureRe, nbodies * sizeof(uint64_t));
	cudaTex* d_textureIm = (cudaTex*)CudaMallocFromHostArray(h_textureIm, nbodies * sizeof(uint64_t));
    
	{
		uint rmax = dimsproj.x / 2;

		uint elements = ElementsFFT2(dimsproj);
		dim3 grid = dim3((elements + 127) / 128, batch, 1);

		ProjectMultibodyKernel << <grid, 128 >> > (d_textureRe, d_textureIm, dimsvolume.x, d_proj, dimsproj.x, elements, d_matrices, d_shifts, d_globalweights, nbodies, rmax * rmax);
    }

	cudaFree(d_textureIm);
	cudaFree(d_textureRe);
	cudaFree(d_globalweights);
	cudaFree(d_shifts);
	cudaFree(d_matrices);
}

__global__ void ProjectMultibodyKernel(cudaTex* d_textureRe,
										cudaTex* d_textureIm,
										int dimvolume, 
										tcomplex* d_proj, 
										int dimproj, 
										int elementsproj, 
										glm::mat2x3* d_rotations, 
										float2* d_shifts,
										float* d_globalweights,
										int nbodies,
										int rmax2)
{
	__shared__ cudaTex s_textureRe[128];
	__shared__ cudaTex s_textureIm[128];
	//__shared__ float s_globalweights[128];
	for (int id = threadIdx.x; id < nbodies; id += blockDim.x)
	{
	    s_textureRe[id] = d_textureRe[id];
	    s_textureIm[id] = d_textureIm[id];
		//s_globalweights[id] = d_globalweights[blockIdx.x * nbodies + id];
	}
	__syncthreads();

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= elementsproj)
		return;

	d_proj += elementsproj * blockIdx.y;

	int dimft = ElementsFFT1(dimproj);

	int idy = id / dimft;
	int idx = id - dimft * idy;

	int x = idx;
	int y = idy <= dimproj / 2 ? idy : idy - dimproj;
	int r2 = y * y + x * x;
	if (r2 > rmax2)
	{
		d_proj[id] = make_cuComplex(0, 0);
		return;
	}

	glm::vec2 oripos = glm::vec2(x, y);
	tcomplex valsum = make_float2(0, 0);

	for (int b = 0; b < nbodies; b++)
	{
		glm::vec3 pos = d_rotations[blockIdx.y * nbodies + b] * oripos;

		// Only asymmetric half is stored
		float is_neg_x = 1.0f;
		if (pos.x < -1e-5f)
		{
			// Get complex conjugated hermitian symmetry pair
			pos.x = abs(pos.x);
			pos.y = -pos.y;
			pos.z = -pos.z;
			is_neg_x = -1.0f;
		}

		float x0, x1, y0, y1, z0, z1;
		tcomplex d000, d010, d100, d110, d001, d011, d101, d111, dx00, dx10, dxy0, dx01, dx11, dxy1;

		// Trilinear interpolation (with physical coords)
		x0 = floor(pos.x + 1e-5f);
		pos.x -= x0;
		x0 += 0.5f;
		x1 = x0 + 1.0f;

		y0 = floor(pos.y);
		pos.y -= y0;
		y1 = y0 + 1;
		if (y0 < 0)
			y0 += dimvolume;
		y0 += 0.5f;
		if (y1 < 0)
			y1 += dimvolume;
		y1 += 0.5f;

		z0 = floor(pos.z);
		pos.z -= z0;
		z1 = z0 + 1;
		if (z0 < 0)
			z0 += dimvolume;
		z0 += 0.5f;
		if (z1 < 0)
			z1 += dimvolume;
		z1 += 0.5f;

		d000 = make_cuComplex(tex3D<tfloat>(s_textureRe[b], x0, y0, z0), tex3D<tfloat>(s_textureIm[b], x0, y0, z0));
		d001 = make_cuComplex(tex3D<tfloat>(s_textureRe[b], x1, y0, z0), tex3D<tfloat>(s_textureIm[b], x1, y0, z0));
		d010 = make_cuComplex(tex3D<tfloat>(s_textureRe[b], x0, y1, z0), tex3D<tfloat>(s_textureIm[b], x0, y1, z0));
		d011 = make_cuComplex(tex3D<tfloat>(s_textureRe[b], x1, y1, z0), tex3D<tfloat>(s_textureIm[b], x1, y1, z0));
		d100 = make_cuComplex(tex3D<tfloat>(s_textureRe[b], x0, y0, z1), tex3D<tfloat>(s_textureIm[b], x0, y0, z1));
		d101 = make_cuComplex(tex3D<tfloat>(s_textureRe[b], x1, y0, z1), tex3D<tfloat>(s_textureIm[b], x1, y0, z1));
		d110 = make_cuComplex(tex3D<tfloat>(s_textureRe[b], x0, y1, z1), tex3D<tfloat>(s_textureIm[b], x0, y1, z1));
		d111 = make_cuComplex(tex3D<tfloat>(s_textureRe[b], x1, y1, z1), tex3D<tfloat>(s_textureIm[b], x1, y1, z1));

		dx00 = lerp(d000, d001, pos.x);
		dx01 = lerp(d010, d011, pos.x);
		dx10 = lerp(d100, d101, pos.x);
		dx11 = lerp(d110, d111, pos.x);

		dxy0 = lerp(dx00, dx01, pos.y);
		dxy1 = lerp(dx10, dx11, pos.y);

		tcomplex val;
		val = lerp(dxy0, dxy1, pos.z);

		val.y *= is_neg_x;

		float2 shift = d_shifts[blockIdx.y * nbodies + b];
		float phase = -(x * shift.x + y * shift.y);
		val = cmul(val, make_float2(__cosf(phase), __sinf(phase)));

		val *= d_globalweights[blockIdx.y * nbodies + b];

		valsum += val;
	}

	d_proj[id] = valsum;
}