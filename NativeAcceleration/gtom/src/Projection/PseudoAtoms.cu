#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
	__constant__ float c_blobvalues[512];


	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void ProjNMAPseudoAtomsKernel(float3* d_positions,
											float* d_intensities,
											uint natoms,
											glm::vec3 volcenter,
											float sigmaprecomp,
											int kernelextent,
											float blobsampling,
											int nblobvalues,
											float3* d_normalmodes,
											float* d_normalmodefactors,
											uint nmodes,
											glm::mat3* d_rotations,
											float2* d_offsets,
											float scale,
											float* d_proj,
											int2 dimsproj,
											uint batch);

	__global__ void ProjSoftPseudoAtomsKernel(float3* d_positions,
												float* d_intensities,
												uint natoms,
												glm::vec3 volcenter,
												float sigmaprecomp,
												int kernelextent,
												float3* d_coarsedeltas,
												float* d_coarseweights,
												int* d_coarseneighbors,
												uint ncoarseneighbors,
												uint ncoarse,
												glm::mat3* d_rotations,
												float2* d_offsets,
												float scale,
												float* d_proj,
												int2 dimsproj,
												uint batch);


	////////////////////////////////////////////////////
	//Project a 3D cloud of pseudo-atoms onto 2D plane//
	////////////////////////////////////////////////////

	void d_ProjNMAPseudoAtoms(float3* d_positions, 
								float* d_intensities, 
								uint natoms,
								int3 dimsvol,
								float sigma, 
								uint kernelextent, 
								float* h_blobvalues,
								float blobsampling,
								uint nblobvalues,
								float3* d_normalmodes, 
								float* d_normalmodefactors, 
								uint nmodes, 
								float3* h_angles, 
								float2* h_offsets, 
								float scale,
								float* d_proj,
								int2 dimsproj,
								uint batch)
	{
		if (nblobvalues > 512)
			throw;

		glm::mat3* h_rotations = (glm::mat3*)malloc(batch * sizeof(glm::mat3));
		for (int i = 0; i < batch; i++)
			h_rotations[i] = Matrix3Euler(h_angles[i]);
		glm::mat3* d_rotations = (glm::mat3*)CudaMallocFromHostArray(h_rotations, batch * sizeof(glm::mat3));
		free(h_rotations);

		float2* d_offsets = (float2*)CudaMallocFromHostArray(h_offsets, batch * sizeof(float2));

		cudaMemcpyToSymbol(c_blobvalues, h_blobvalues, nblobvalues * sizeof(float), 0, cudaMemcpyHostToDevice);

		sigma = -1 / (2 * sigma * sigma);
		dim3 grid = dim3(tmin(32, (natoms + 127) / 128), batch, 1);

		ProjNMAPseudoAtomsKernel <<<grid, 128>>> (d_positions,
													d_intensities,
													natoms,
													glm::vec3(dimsvol.x / 2, dimsvol.y / 2, dimsvol.z / 2),
													sigma,
													kernelextent,
													blobsampling,
													nblobvalues,
													d_normalmodes,
													d_normalmodefactors,
													nmodes,
													d_rotations,
													d_offsets,
													scale,
													d_proj,
													dimsproj,
													batch);

		cudaFree(d_offsets);
		cudaFree(d_rotations);
	}

	void d_ProjSoftPseudoAtoms(float3* d_positions,
								float* d_intensities,
								uint natoms,
								int3 dimsvol,
								float sigma,
								uint kernelextent,
								float3* d_coarsedeltas,
								float* d_coarseweights,
								int* d_coarseneighbors,
								uint ncoarseneighbors,
								uint ncoarse,
								float3* h_angles,
								float2* h_offsets,
								float scale,
								float* d_proj,
								int2 dimsproj,
								uint batch)
	{
		glm::mat3* h_rotations = (glm::mat3*)malloc(batch * sizeof(glm::mat3));
		for (int i = 0; i < batch; i++)
			h_rotations[i] = Matrix3Euler(h_angles[i]);
		glm::mat3* d_rotations = (glm::mat3*)CudaMallocFromHostArray(h_rotations, batch * sizeof(glm::mat3));
		free(h_rotations);

		float2* d_offsets = (float2*)CudaMallocFromHostArray(h_offsets, batch * sizeof(float2));

		sigma = -1 / (2 * sigma * sigma);
		dim3 grid = dim3(tmin(32, (natoms + 127) / 128), batch, 1);

		ProjSoftPseudoAtomsKernel << <grid, 128 >> > (d_positions,
														d_intensities,
														natoms,
														glm::vec3(dimsvol.x / 2, dimsvol.y / 2, dimsvol.z / 2),
														sigma,
														kernelextent,
														d_coarsedeltas,
														d_coarseweights,
														d_coarseneighbors,
														ncoarseneighbors,
														ncoarse,
														d_rotations,
														d_offsets,
														scale,
														d_proj,
														dimsproj,
														batch);

		cudaFree(d_offsets);
		cudaFree(d_rotations);
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void ProjNMAPseudoAtomsKernel(float3* d_positions,
											float* d_intensities,
											uint natoms,
											glm::vec3 volcenter,
											float sigmaprecomp,
											int kernelextent,
											float blobsampling,
											int nblobvalues,
											float3* d_normalmodes,
											float* d_normalmodefactors,
											uint nmodes,
											glm::mat3* d_rotations,
											float2* d_offsets,
											float scale,
											float* d_proj,
											int2 dimsproj,
											uint batch)
	{
		d_normalmodefactors += blockIdx.y * nmodes;	// Offset to the current projection
		d_proj += Elements2(dimsproj) * blockIdx.y;

		__shared__ glm::mat3 rotation;
		if (threadIdx.x == 0)
			rotation = d_rotations[blockIdx.y];
		__syncthreads();

		for (uint aid = blockIdx.x * blockDim.x + threadIdx.x; aid < natoms; aid += blockDim.x * gridDim.x)	// Atom ID
		{
			// Combine normal modes, and rotate atom position
			float3 pos = d_positions[aid];
			for (uint i = 0; i < nmodes; i++)
				pos += d_normalmodes[i * natoms + aid] * d_normalmodefactors[i];

			glm::vec3 transformed = rotation * (glm::vec3(pos.x, pos.y, pos.z) - volcenter) * scale +
									glm::vec3(dimsproj.x / 2 + d_offsets[blockIdx.y].x * scale, dimsproj.y / 2 + d_offsets[blockIdx.y].y * scale, 0);

			int2 kernelcenter = make_int2((int)transformed.x, (int)transformed.y);

			float intensity = d_intensities[aid];
			for (int dy = -kernelextent; dy <= kernelextent; dy++)
			{
				int y = kernelcenter.y + dy;
				if (y < 0 || y >= dimsproj.y)
					continue;

				for (int dx = -kernelextent; dx <= kernelextent; dx++)
				{
					int x = kernelcenter.x + dx;
					if (x < 0 || x >= dimsproj.x)
						continue;

					float2 diff = make_float2(transformed.x - x, transformed.y - y);
					int iblob = (int)(sqrt(dotp2(diff, diff)) / blobsampling);
					if (iblob >= nblobvalues)
						continue;

					float weight = c_blobvalues[iblob];

					atomicAdd(d_proj + y * dimsproj.x + x, intensity * weight);
				}
			}
		}
	}

	__global__ void ProjSoftPseudoAtomsKernel(float3* d_positions,
												float* d_intensities,
												uint natoms,
												glm::vec3 volcenter,
												float sigmaprecomp,
												int kernelextent,
												float3* d_coarsedeltas,
												float* d_coarseweights,
												int* d_coarseneighbors,
												uint ncoarseneighbors,
												uint ncoarse,
												glm::mat3* d_rotations,
												float2* d_offsets,
												float scale,
												float* d_proj,
												int2 dimsproj,
												uint batch)
	{
		d_coarsedeltas += blockIdx.y * ncoarse;	// Offset to the current projection
		d_proj += blockIdx.y * Elements2(dimsproj);

		__shared__ glm::mat3 rotation;
		if (threadIdx.x == 0)
			rotation = d_rotations[blockIdx.y];
		__syncthreads();

		for (uint aid = blockIdx.x * blockDim.x + threadIdx.x; aid < natoms; aid += blockDim.x * gridDim.x)	// Atom ID
		{
			// Combine normal modes, and rotate atom position
			float3 pos = d_positions[aid];
			for (uint i = 0; i < ncoarseneighbors; i++)
			{
				int coarseneighbor = d_coarseneighbors[aid * ncoarseneighbors + i];
				pos += d_coarsedeltas[coarseneighbor] * d_coarseweights[aid * ncoarseneighbors + i];
			}

			glm::vec3 transformed = rotation * (glm::vec3(pos.x, pos.y, pos.z) - volcenter) * scale +
									glm::vec3(dimsproj.x / 2 + d_offsets[blockIdx.y].x * scale, dimsproj.y / 2 + d_offsets[blockIdx.y].y * scale, 0);

			int2 kernelcenter = make_int2((int)transformed.x, (int)transformed.y);

			float intensity = d_intensities[aid];
			for (int dy = -kernelextent; dy <= kernelextent; dy++)
			{
				int y = kernelcenter.y + dy;
				if (y < 0 || y >= dimsproj.y)
					continue;

				for (int dx = -kernelextent; dx <= kernelextent; dx++)
				{
					int x = kernelcenter.x + dx;
					if (x < 0 || x >= dimsproj.x)
						continue;

					float2 dist = make_float2(transformed.x - x, transformed.y - y);
					float weight = exp(dotp2(dist, dist) * sigmaprecomp);

					atomicAdd(d_proj + y * dimsproj.x + x, intensity * weight);
				}
			}
		}
	}
}