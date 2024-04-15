#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Relion.cuh"

namespace gtom
{
	__global__ void BackprojectTomoKernel(tcomplex* d_volumeft, int dimvolume, tcomplex* d_projft, tfloat* d_projweights, uint dimproj, size_t elementsproj, glm::mat3* d_rotations, int rmax, int batch, int zoffset);

	void d_BackprojectTomo(tcomplex* d_volumeft, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, tfloat3* h_angles, uint batch)
	{
		glm::mat3* d_matrices;

		{
			glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
			for (int i = 0; i < batch; i++)
				h_matrices[i] = glm::transpose(Matrix3Euler(h_angles[i]));
			d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
			free(h_matrices);
		}

		d_BackprojectTomo(d_volumeft, dimsvolume, d_projft, d_projweights, dimsproj, rmax, d_matrices, batch);

		{
			cudaFree(d_matrices);
		}
	}

	void d_BackprojectTomo(tcomplex* d_volumeft, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, glm::mat3* d_matrices, uint batch)
	{
		uint ndimsvolume = DimensionCount(dimsvolume);
		uint ndimsproj = DimensionCount(dimsproj);
		if (ndimsvolume < ndimsproj)
			throw;

		rmax = tmin(rmax, dimsvolume.x / 2);

		for (int z = 0; z < dimsvolume.z; z += 1)
		{
			dim3 grid = dim3(dimsvolume.x / 2 + 1, dimsvolume.y, tmin(1, dimsvolume.z - z));
			uint elements = ElementsFFT2(dimsproj);

			BackprojectTomoKernel << <grid, 128 >> > (d_volumeft, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, rmax, batch, z);
		}
	}

	__global__ void BackprojectTomoKernel(tcomplex* d_volumeft, int dimvolume, tcomplex* d_projft, tfloat* d_projweights, uint dimproj, size_t elementsproj, glm::mat3* d_rotations, int rmax, int batch, int zoffset)
	{
		glm::vec3 volpos;
		volpos.x = blockIdx.x;
		volpos.y = blockIdx.y < dimvolume / 2 ? (int)(blockIdx.y) : (int)(blockIdx.y) - dimvolume;
		volpos.z = (blockIdx.z + zoffset) < dimvolume / 2 ? (int)(blockIdx.z + zoffset) : (int)(blockIdx.z + zoffset) - dimvolume;

		if (glm::length(volpos) >= rmax)
			return;
		/*else
		{
			if (threadIdx.x == 0)
				d_volumeft[(blockIdx.z * dimvolume + blockIdx.y) * (dimvolume / 2 + 1) + blockIdx.x] = make_cuComplex(volpos.y, 0);
			return;
		}*/

		//__shared__ glm::mat3 s_rotation[1];
		__shared__ tcomplex s_valuesum[128];
		__shared__ tfloat s_weightsum[128];
		__shared__ tfloat s_samplesum[128];
		//__shared__ tfloat s_samplemax[128];

		uint dimprojft = ElementsFFT1(dimproj);

		tcomplex valuesum = make_cuComplex(0, 0);
		tfloat weightsum = 0;
		tfloat samplesum = 0;
		//tfloat samplemax = -1;

		for (int b = 0; b < batch; b++)
		{
			//if (threadIdx.x == 0)
			glm::mat3 s_rotation = d_rotations[b];

			if (abs((glm::transpose(s_rotation) * volpos).z) > 5)
				continue;

			//__syncthreads();

			for (uint id = threadIdx.x; id < elementsproj; id += blockDim.x)
			{
				uint idx = id % dimprojft;
				uint idy = id / dimprojft;

				int x = idx;
				int y = idy <= dimproj / 2 ? idy : (int)idy - (int)dimproj;

				glm::vec3 pos = glm::vec3(x, y, 0);
				pos = s_rotation * pos;

				// Only asymmetric half is stored
				float is_neg_x = 1.0f;
				if (pos.x < 0)
				{
					// Get complex conjugated hermitian symmetry pair
					pos.x = -pos.x;
					pos.y = -pos.y;
					pos.z = -pos.z;
					is_neg_x = -1.0f;
				}

				glm::vec3 dist = volpos - pos;

				if (abs(dist.x) > 5 || abs(dist.y) > 5 || abs(dist.z) > 5)
					continue;

				float sincx = sinc(dist.x) * sinc(dist.x / 5);
				float sincy = sinc(dist.y) * sinc(dist.y / 5);
				float sincz = sinc(dist.z) * sinc(dist.z / 5);

				float sample = sincx * sincy * sincz;
				
				//samplemax = tmax(samplemax, sample);
				samplesum += sample;
				weightsum += sample * d_projweights[elementsproj * b + id];

				tcomplex value = d_projft[elementsproj * b + id];
				value.y *= is_neg_x;
				valuesum += sample * value;
			}
		}

		s_valuesum[threadIdx.x] = valuesum;
		s_weightsum[threadIdx.x] = weightsum;
		s_samplesum[threadIdx.x] = samplesum;
		//s_samplemax[threadIdx.x] = samplemax;

		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < 128; i++)
			{
				valuesum += s_valuesum[i];
				weightsum += s_weightsum[i];
				samplesum += s_samplesum[i];
				//samplemax = tmax(samplemax, s_samplemax[i]);
			}

			if (abs(samplesum) > 1e-3f)
			{
				valuesum /= weightsum;
				valuesum *= tmin(1, tmax(-1, samplesum));

				d_volumeft[((blockIdx.z + zoffset) * dimvolume + blockIdx.y) * (dimvolume / 2 + 1) + blockIdx.x] = valuesum;// *samplemax;
			}
		}
	}
}
