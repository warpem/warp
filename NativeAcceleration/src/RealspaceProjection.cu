#include "include/Functions.h"
using namespace gtom;

__global__ void RealspaceProjectForwardKernel(float* d_volume, int3 dimsvolume, float* d_projections, int2 dimsproj, glm::mat3* d_rotations);
__global__ void RealspaceProjectBackwardKernel(float* d_volume, int3 dimsvolume, float* d_projections, int2 dimsproj, glm::mat3* d_rotations, bool normalizesamples, int batch);

__declspec(dllexport) void RealspaceProjectForward(float* d_volume,
													int3 dimsvolume,
													float* d_projections,
													int2 dimsproj,
													float supersample,
													float3* h_angles,
													int batch)
{
	d_ValueFill(d_projections, Elements2(dimsproj) * batch, 0.0f);

	glm::mat3* d_matrices;
	
	glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
	for (int i = 0; i < batch; i++)
		h_matrices[i] = Matrix3Euler(h_angles[i]) * Matrix3Scale(supersample);
	d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
	free(h_matrices);

	dim3 grid = dim3(tmin(128, (Elements(dimsvolume) + 127) / 128), batch, 1);
	uint elements = 128;

	RealspaceProjectForwardKernel <<<grid, elements>>> (d_volume, dimsvolume, d_projections, dimsproj, d_matrices);

	cudaFree(d_matrices);
}

__declspec(dllexport) void RealspaceProjectBackward(float* d_volume,
													int3 dimsvolume,
													float* d_projections,
													int2 dimsproj,
													float supersample,
													float3* h_angles,
													bool normalizesamples,
													int batch)
{
	glm::mat3* d_matrices;

	glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
	for (int i = 0; i < batch; i++)
		h_matrices[i] = Matrix3Euler(h_angles[i]) * Matrix3Scale(supersample);
	d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
	free(h_matrices);

	dim3 grid = dim3(tmin(1024, (Elements(dimsvolume) + 127) / 128), 1, 1);
	uint elements = 128;

	RealspaceProjectBackwardKernel << <grid, elements >> > (d_volume, dimsvolume, d_projections, dimsproj, d_matrices, normalizesamples, batch);

	cudaFree(d_matrices);
}

__global__ void RealspaceProjectForwardKernel(float* d_volume, int3 dimsvolume, float* d_projections, int2 dimsproj, glm::mat3* d_rotations)
{
	d_projections += Elements2(dimsproj) * blockIdx.y;

	uint dimx = dimsvolume.x;
	uint slice = Elements2(dimsvolume);

	glm::mat3 rotation = d_rotations[blockIdx.y];
	glm::vec3 volumecenter = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
	glm::vec3 projectioncenter = glm::vec3(dimsproj.x / 2, dimsproj.y / 2, 0);

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < Elements(dimsvolume); id += gridDim.x * blockDim.x)
	{
		uint idx = id % dimx;
		uint idy = (id % slice) / dimx;
		uint idz = id / slice;

		glm::vec3 pos = glm::vec3(idx, idy, idz);
		pos -= volumecenter;
		pos = rotation * pos;
		pos += projectioncenter;

		// Bilinear interpolation
		int x0 = floor(pos.x);
		pos.x -= x0;
		int x1 = x0 + 1;

		int y0 = floor(pos.y);
		pos.y -= y0;
		int y1 = y0 + 1;

		float c0 = 1.0f - pos.y;
		float c1 = pos.y;

		float c00 = (1.0f - pos.x) * c0;
		float c01 = pos.x * c0;
		float c10 = (1.0f - pos.x) * c1;
		float c11 = pos.x * c1;

		float val = d_volume[id];

		if (x0 >= 0 && y0 >= 0 && x0 < dimsproj.x && y0 < dimsproj.y)
			atomicAdd((tfloat*)(d_projections + y0 * dimsproj.x + x0), c00 * val);

		if (x1 >= 0 && y0 >= 0 && x1 < dimsproj.x && y0 < dimsproj.y)
			atomicAdd((tfloat*)(d_projections + y0 * dimsproj.x + x1), c01 * val);

		if (x0 >= 0 && y1 >= 0 && x0 < dimsproj.x && y1 < dimsproj.y)
			atomicAdd((tfloat*)(d_projections + y1 * dimsproj.x + x0), c10 * val);

		if (x1 >= 0 && y1 >= 0 && x1 < dimsproj.x && y1 < dimsproj.y)
			atomicAdd((tfloat*)(d_projections + y1 * dimsproj.x + x1), c11 * val);
	}
}

__global__ void RealspaceProjectBackwardKernel(float* d_volume, int3 dimsvolume, float* d_projections, int2 dimsproj, glm::mat3* d_rotations, bool normalizesamples, int batch)
{
	uint dimx = dimsvolume.x;
	uint slice = Elements2(dimsvolume);

	glm::vec3 volumecenter = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
	glm::vec3 projectioncenter = glm::vec3(dimsproj.x / 2, dimsproj.y / 2, 0);

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < Elements(dimsvolume); id += gridDim.x * blockDim.x)
	{
		uint idx = id % dimx;
		uint idy = (id % slice) / dimx;
		uint idz = id / slice;

		float sum = 0;
		float samples = 0;

		for (int b = 0; b < batch; b++)
		{
			glm::vec3 pos = glm::vec3(idx, idy, idz);
			pos -= volumecenter;
			pos = d_rotations[b] * pos;
			pos += projectioncenter;

			// Bilinear interpolation
			int x0 = floor(pos.x);
			pos.x -= x0;
			int x1 = x0 + 1;

			int y0 = floor(pos.y);
			pos.y -= y0;
			int y1 = y0 + 1;

			float c0 = 1.0f - pos.y;
			float c1 = pos.y;

			float c00 = (1.0f - pos.x) * c0;
			float c01 = pos.x * c0;
			float c10 = (1.0f - pos.x) * c1;
			float c11 = pos.x * c1;

			if (x0 >= 0 && y0 >= 0 && x0 < dimsproj.x && y0 < dimsproj.y)
			{
				sum += d_projections[b * Elements2(dimsproj) + y0 * dimsproj.x + x0] * c00;
				samples += c00;
			}

			if (x1 >= 0 && y0 >= 0 && x1 < dimsproj.x && y0 < dimsproj.y)
			{
				sum += d_projections[b * Elements2(dimsproj) + y0 * dimsproj.x + x1] * c01;
				samples += c01;
			}

			if (x0 >= 0 && y1 >= 0 && x0 < dimsproj.x && y1 < dimsproj.y)
			{
				sum += d_projections[b * Elements2(dimsproj) + y1 * dimsproj.x + x0] * c10;
				samples += c10;
			}

			if (x1 >= 0 && y1 >= 0 && x1 < dimsproj.x && y1 < dimsproj.y)
			{
				sum += d_projections[b * Elements2(dimsproj) + y1 * dimsproj.x + x1] * c11;
				samples += c11;
			}
		}

		if (normalizesamples)
			d_volume[id] = sum / tmax(1, samples);
		else
			d_volume[id] = sum;
	}
}