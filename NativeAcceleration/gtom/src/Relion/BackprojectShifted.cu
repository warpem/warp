#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Relion.cuh"

namespace gtom
{
	template<uint ndims, uint TpB> __global__ void BackprojectShifted3DtoNDKernel(tcomplex* d_volumeft, tfloat* d_volumeweights, uint dimvolume, tcomplex* d_projft, tfloat* d_projweights, uint dimproj, size_t elementsproj, glm::mat3* d_rotations, tfloat3* d_shifts, float* d_globalweights, uint rmax, int rmax2);
	
	void d_rlnBackprojectShifted(tcomplex* d_volumeft, tfloat* d_volumeweights, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, tfloat3* h_angles, tfloat3* h_shifts, float* h_globalweights, float supersample, uint batch)
	{
		glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
		for (int i = 0; i < batch; i++)
			h_matrices[i] = glm::transpose(Matrix3Euler(h_angles[i])) * Matrix3Scale(supersample);
		glm::mat3* d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
		free(h_matrices);
		
		tfloat3* h_shiftsscaled = (tfloat3*)malloc(batch * sizeof(tfloat3));
		for (int i = 0; i < batch; i++)
			h_shiftsscaled[i] = tfloat3(h_shifts[i].x * PI2 / dimsproj.x, 
										h_shifts[i].y * PI2 / dimsproj.x, 
										h_shifts[i].z * PI2 / dimsproj.x);
		tfloat3* d_shifts = (tfloat3*)CudaMallocFromHostArray(h_shiftsscaled, batch * sizeof(tfloat3));
		free(h_shiftsscaled);

		float* d_globalweights = (float*)CudaMallocFromHostArray(h_globalweights, batch * sizeof(float));

		d_rlnBackprojectShifted(d_volumeft, d_volumeweights, dimsvolume, d_projft, d_projweights, dimsproj, rmax, d_matrices, d_shifts, d_globalweights, batch);

		cudaFree(d_globalweights);
		cudaFree(d_shifts);
		cudaFree(d_matrices);
	}

	void d_rlnBackprojectShifted(tcomplex* d_volumeft, tfloat* d_volumeweights, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, glm::mat3* d_matrices, tfloat3* d_shifts, float* d_globalweights, uint batch)
	{
		uint ndimsvolume = DimensionCount(dimsvolume);
		uint ndimsproj = DimensionCount(dimsproj);
		if (ndimsvolume < ndimsproj)
			throw;

		rmax = tmin(rmax, dimsproj.x / 2);

		if (ndimsvolume == 3)
		{
			dim3 grid = dim3(1, batch, 1);
			uint elements = ElementsFFT(dimsproj);

			if (ndimsproj == 2)
				BackprojectShifted3DtoNDKernel<2, 128> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_shifts, d_globalweights, rmax, rmax * rmax);
			else if (ndimsproj == 3)
				BackprojectShifted3DtoNDKernel<3, 128> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_shifts, d_globalweights, rmax, rmax * rmax);
		}
		else
		{
			throw;
		}
	}

	template<uint ndims, uint TpB> __global__ void BackprojectShifted3DtoNDKernel(tcomplex* d_volumeft, 
																					tfloat* d_volumeweights, 
																					uint dimvolume, 
																					tcomplex* d_projft, 
																					tfloat* d_projweights, 
																					uint dimproj, 
																					size_t elementsproj, 
																					glm::mat3* d_rotations, 
																					tfloat3* d_shifts, 
																					float* d_globalweights,
																					uint rmax, 
																					int rmax2)
	{
		d_projft += elementsproj * blockIdx.y;
		d_projweights += elementsproj * blockIdx.y;

		uint slice = ndims == 3 ? ElementsFFT1(dimproj) * dimproj : 1;
		uint dimft = ElementsFFT1(dimproj);
		uint dimvolumeft = ElementsFFT1(dimvolume);

		glm::mat3 rotation = d_rotations[blockIdx.y];
		tfloat3 shift = d_shifts[blockIdx.y];
		float globalweight = d_globalweights[blockIdx.y];

		for (uint id = threadIdx.x; id < elementsproj; id += TpB)
		{
			uint idx = id % dimft;
			uint idy = (ndims == 3 ? id % slice : id) / dimft;
			uint idz = ndims == 3 ? id / slice : 0;

			int x = idx;
			int y = idy <= rmax ? idy : (int)idy - (int)dimproj;
			int z = ndims == 3 ? (idz <= rmax ? idz : (int)idz - (int)dimproj) : 0;
			int r2 = ndims == 3 ? z * z + y * y + x * x : y * y + x * x;
			if (r2 > rmax2)
				continue;

			glm::vec3 pos = glm::vec3(x, y, z);
			pos = rotation * pos;

			// Only asymmetric half is stored
			float is_neg_x = 1.0f;
			if (pos.x + 1e-5f < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				pos.x = abs(pos.x);
				pos.y = -pos.y;
				pos.z = -pos.z;
				is_neg_x = -1.0f;
			}

			// Trilinear interpolation
			int x0 = floor(pos.x + 1e-5f);
			pos.x -= x0;
			int x1 = x0 + 1;

			int y0 = floor(pos.y);
			pos.y -= y0;
			y0 += dimvolume / 2;
			int y1 = y0 + 1;

			int z0 = floor(pos.z);
			pos.z -= z0;
			z0 += dimvolume / 2;
			int z1 = z0 + 1;

			float c0 = 1.0f - pos.z;
			float c1 = pos.z;

			float c00 = (1.0f - pos.y) * c0;
			float c10 = pos.y * c0;
			float c01 = (1.0f - pos.y) * c1;
			float c11 = pos.y * c1;

			float c000 = (1.0f - pos.x) * c00;
			float c100 = pos.x * c00;
			float c010 = (1.0f - pos.x) * c10;
			float c110 = pos.x * c10;
			float c001 = (1.0f - pos.x) * c01;
			float c101 = pos.x * c01;
			float c011 = (1.0f - pos.x) * c11;
			float c111 = pos.x * c11;

			tcomplex val = d_projft[id];

			float phase = ndims == 3 ? -(x * shift.x + y * shift.y + z * shift.z) : -(x * shift.x + y * shift.y);
			val = cmul(val, make_float2(__cosf(phase), __sinf(phase)));

			val.y *= is_neg_x;

			val *= globalweight;


			tfloat weight = d_projweights[id] * globalweight;

			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y0) * dimvolumeft + x0), c000 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y0) * dimvolumeft + x0) + 1, c000 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z0 * dimvolume + y0) * dimvolumeft + x0), c000 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y0) * dimvolumeft + x1), c100 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y0) * dimvolumeft + x1) + 1, c100 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z0 * dimvolume + y0) * dimvolumeft + x1), c100 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y1) * dimvolumeft + x0), c010 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y1) * dimvolumeft + x0) + 1, c010 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z0 * dimvolume + y1) * dimvolumeft + x0), c010 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y1) * dimvolumeft + x1), c110 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y1) * dimvolumeft + x1) + 1, c110 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z0 * dimvolume + y1) * dimvolumeft + x1), c110 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y0) * dimvolumeft + x0), c001 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y0) * dimvolumeft + x0) + 1, c001 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z1 * dimvolume + y0) * dimvolumeft + x0), c001 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y0) * dimvolumeft + x1), c101 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y0) * dimvolumeft + x1) + 1, c101 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z1 * dimvolume + y0) * dimvolumeft + x1), c101 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y1) * dimvolumeft + x0), c011 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y1) * dimvolumeft + x0) + 1, c011 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z1 * dimvolume + y1) * dimvolumeft + x0), c011 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y1) * dimvolumeft + x1), c111 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y1) * dimvolumeft + x1) + 1, c111 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z1 * dimvolume + y1) * dimvolumeft + x1), c111 * weight);
		}
	}
}
