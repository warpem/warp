#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/Helper.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <bool iscentered, class T> __global__ void Exact2DWeightingKernel(T* d_weights, int2 dims, int index, glm::vec3* d_normals, glm::mat3x2 globalB2localB, tfloat* d_imageweights, int nimages, uint currentimage, tfloat maxfreq);
	template <bool iscentered, class T> __global__ void Exact3DWeightingKernel(T* d_weights, int3 dims, glm::vec3* d_normals, int nimages, tfloat maxfreq);


	/////////////////////////////////////////////////////////////////////////////////////
	//2D weighting of frequency components for WBP reconstruction, using sinc(distance)//
	/////////////////////////////////////////////////////////////////////////////////////

	template <class T> void d_Exact2DWeighting(T* d_weights, int2 dimsimage, int* h_indices, tfloat3* h_angles, tfloat* d_imageweights, int nimages, tfloat maxfreq, bool iszerocentered, int batch)
	{
		glm::vec3* h_normals = (glm::vec3*)malloc(nimages * sizeof(glm::vec3));
		glm::mat3x2* h_globalB2localB = (glm::mat3x2*)malloc(nimages * sizeof(glm::mat3x2));

		for (int i = 0; i < nimages; i++)
		{
			glm::mat3 tB = Matrix3Euler(h_angles[i]);
			h_normals[i] = glm::vec3(tB[2][0], tB[2][1], tB[2][2]);
			h_globalB2localB[i] = glm::mat3x2(tB[0][0], tB[1][0], tB[0][1], tB[1][1], tB[0][2], tB[1][2]);	//Column-major layout in constructor
		}

		glm::vec3* d_normals = (glm::vec3*)CudaMallocFromHostArray(h_normals, nimages * sizeof(glm::vec3));

		tfloat* d_tempweights;
		if (d_imageweights == NULL)
			d_tempweights = CudaMallocValueFilled(nimages, (tfloat)1);
		else
			d_tempweights = d_imageweights;

		uint TpB = tmin(NextMultipleOf(dimsimage.x / 2 + 1, 32), 128);
		dim3 grid = dim3(dimsimage.y);
		for (int b = 0; b < batch; b++)
			if (iszerocentered)
				Exact2DWeightingKernel<true, T> << <grid, TpB >> > (d_weights + (dimsimage.x / 2 + 1) * dimsimage.y * b, dimsimage, h_indices[b], d_normals, h_globalB2localB[b], d_tempweights, nimages, b, maxfreq);
			else
				Exact2DWeightingKernel<false, T> << <grid, TpB >> > (d_weights + (dimsimage.x / 2 + 1) * dimsimage.y * b, dimsimage, h_indices[b], d_normals, h_globalB2localB[b], d_tempweights, nimages, b, maxfreq);

		free(h_globalB2localB);
		free(h_normals);
		cudaFree(d_normals);
		if (d_imageweights == NULL)
			cudaFree(d_tempweights);
	}
	template void d_Exact2DWeighting<tfloat>(tfloat* d_weights, int2 dimsimage, int* h_indices, tfloat3* h_angles, tfloat* d_imageweights, int nimages, tfloat maxfreq, bool iszerocentered, int batch);
	template void d_Exact2DWeighting<tcomplex>(tcomplex* d_weights, int2 dimsimage, int* h_indices, tfloat3* h_angles, tfloat* d_imageweights, int nimages, tfloat maxfreq, bool iszerocentered, int batch);

	template <class T> void d_Exact3DWeighting(T* d_weights, int3 dimsvolume, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered)
	{
		glm::vec3* h_normals = (glm::vec3*)malloc(nimages * sizeof(glm::vec3));

		for (int i = 0; i < nimages; i++)
		{
			glm::mat3 tB = Matrix3Euler(tfloat3(h_angles[i].x, h_angles[i].y, 0.0f));
			h_normals[i] = glm::vec3(tB[2][0], tB[2][1], tB[2][2]);
		}

		glm::vec3* d_normals = (glm::vec3*)CudaMallocFromHostArray(h_normals, nimages * sizeof(glm::vec3));

		uint TpB = min(NextMultipleOf(nimages, 32), 128);
		dim3 grid = dim3(dimsvolume.x / 2 + 1, dimsvolume.y, dimsvolume.z);
		if (iszerocentered)
			Exact3DWeightingKernel<true, T> << <grid, TpB >> > (d_weights, dimsvolume, d_normals, nimages, maxfreq);
		else
			Exact3DWeightingKernel<false, T> << <grid, TpB >> > (d_weights, dimsvolume, d_normals, nimages, maxfreq);

		free(h_normals);
		cudaFree(d_normals);
	}
	template void d_Exact3DWeighting<tfloat>(tfloat* d_weights, int3 dimsvolume, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered);
	template void d_Exact3DWeighting<tcomplex>(tcomplex* d_weights, int3 dimsvolume, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered);


	////////////////
	//CUDA kernels//
	////////////////

	template <bool iscentered, class T> __global__ void Exact2DWeightingKernel(T* d_weights, int2 dims, int index, glm::vec3* d_normals, glm::mat3x2 globalB2localB, tfloat* d_imageweights, int nimages, uint currentimage, tfloat maxfreq)
	{
		int idy = blockIdx.x;

		int x, y;
		if (!iscentered)
			y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);
		else
			y = idy;
		d_weights += y * (dims.x / 2 + 1);

		glm::vec2 center = glm::vec2((float)(dims.x / 2), (float)(dims.y / 2));
		glm::vec3 normalA = d_normals[index];

		for (int idx = threadIdx.x; idx < dims.x / 2 + 1; idx += blockDim.x)
		{
			if (!iscentered)
				x = dims.x / 2 - idx;
			else
				x = idx;

			glm::vec2 localA = glm::vec2((float)idx, (float)idy) - center;
			if (glm::length(localA) <= maxfreq)
			{
				glm::vec3 globalA = localA * globalB2localB;
				float weightsum = 0.0f;

				for (int b = 0; b < nimages; b++)
				{
					glm::vec3 normalB = d_normals[b];
					float distance = dotp(globalA, normalB);
					weightsum += abs(sinc(distance)) * d_imageweights[b]; //tmax(0, 1 - abs(distance));
				}

				d_weights[x] *= d_imageweights[currentimage] * d_imageweights[currentimage] / weightsum;
			}
			else
			{
				d_weights[x] *= 0.0f;
			}
		}
	}

	template <bool iscentered, class T> __global__ void Exact3DWeightingKernel(T* d_weights, int3 dims, glm::vec3* d_normals, int nimages, tfloat maxfreq)
	{
		__shared__ tfloat s_sums[128];
		s_sums[threadIdx.x] = 0.0f;

		int idx = blockIdx.x;
		int idy = blockIdx.y;
		int idz = blockIdx.z;

		int x, y, z;
		if (!iscentered)
		{
			x = dims.x / 2 - idx;
			y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);
			z = dims.z - 1 - ((idz + dims.z / 2 - 1) % dims.z);
		}
		else
		{
			x = idx;
			y = idy;
			z = idz;
		}

		glm::vec3 center = glm::vec3(dims.x / 2, dims.y / 2, dims.z / 2);
		glm::vec3 globalA = glm::vec3(idx, idy, idz) - center;
		float sum = 0.0f;

		if (glm::length(globalA) <= maxfreq)
		{
			for (int b = threadIdx.x; b < nimages; b += blockDim.x)
			{
				glm::vec3 normalB = d_normals[b];
				float distance = dotp(globalA, normalB);
				sum += sinc(distance);
			}
		}
		else
		{
			sum = 0.0f;
		}
		s_sums[threadIdx.x] = sum;
		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < blockDim.x; i++)
				sum += s_sums[i];
			d_weights[(z * dims.y + y) * (dims.x / 2 + 1) + x] *= 1.0f / max(sum, 1.0f);
		}
	}
}