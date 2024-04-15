#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "Helper.cuh"
#include "Relion.cuh"

namespace gtom
{
	//template<uint TpB> __global__ void Project2Dto2DKernel(cudaTex t_volumeRe, cudaTex t_volumeIm, uint dimvolume, tcomplex* d_proj, uint dimproj, uint rmax, uint rmax2);
	template<uint ndims, bool decentered, bool squareinterpweights> __global__ void Backproject3DtoNDKernel(tcomplex* d_volumeft,
		tfloat* d_volumeweights,
		uint dimvolume,
		tcomplex* d_projft,
		tfloat* d_projweights,
		uint dimproj,
		size_t elementsproj,
		glm::mat3* d_rotations,
		int* d_ivolume,
		glm::mat2 magnification,
		float ewalddiameterinv,
		uint rmax,
		int rmax2);

	void d_rlnBackproject(tcomplex* d_volumeft, tfloat* d_volumeweights, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, tfloat3* h_angles, int* h_ivolume, float3 magnification, float ewaldradius, float supersample, bool outputdecentered, bool squareinterpweights, uint batch)
	{
		glm::mat3* d_matrices;
		int* d_ivolume = NULL;

		{
			glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
			for (int i = 0; i < batch; i++)
				h_matrices[i] = glm::transpose(Matrix3Euler(h_angles[i])) * Matrix3Scale(supersample);
			d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
			free(h_matrices);

			if (h_ivolume != NULL)
				d_ivolume = (int*)CudaMallocFromHostArray(h_ivolume, sizeof(int) * batch);
		}

		d_rlnBackproject(d_volumeft, d_volumeweights, dimsvolume, d_projft, d_projweights, dimsproj, rmax, d_matrices, d_ivolume, magnification, ewaldradius * supersample, outputdecentered, squareinterpweights, batch);

		{
			cudaFree(d_matrices);
			if (d_ivolume != NULL)
				cudaFree(d_ivolume);
		}
	}

	void d_rlnBackproject(tcomplex* d_volumeft, tfloat* d_volumeweights, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, glm::mat3* d_matrices, int* d_ivolume, float3 magnification, float ewaldradiussuper, bool outputdecentered, bool squareinterpweights, uint batch)
	{
		uint ndimsvolume = DimensionCount(dimsvolume);
		uint ndimsproj = DimensionCount(dimsproj);
		if (ndimsvolume < ndimsproj)
			throw;

		rmax = tmin(rmax, dimsproj.x / 2);

		glm::mat2 m_magnification = Matrix2Rotation(-magnification.z) * Matrix2Scale(tfloat2(magnification.x, magnification.y)) * Matrix2Rotation(magnification.z);

		float ewalddiameterinv = ewaldradiussuper == 0 ? 0 : 1.0f / (2.0f * ewaldradiussuper);

		if (ndimsvolume == 3)
		{
			dim3 grid = dim3(1, batch, 1);
			uint elements = ElementsFFT(dimsproj);

			if (squareinterpweights)
			{
				if (ndimsproj == 2)
				{
					if (outputdecentered)
						Backproject3DtoNDKernel<2, true, true> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_ivolume, m_magnification, ewalddiameterinv, rmax, rmax * rmax);
					else
						Backproject3DtoNDKernel<2, false, true> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_ivolume, m_magnification, ewalddiameterinv, rmax, rmax * rmax);
				}
				else if (ndimsproj == 3)
				{
					if (outputdecentered)
						Backproject3DtoNDKernel<3, true, true> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_ivolume, m_magnification, ewalddiameterinv, rmax, rmax * rmax);
					else
						Backproject3DtoNDKernel<3, false, true> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_ivolume, m_magnification, ewalddiameterinv, rmax, rmax * rmax);
				}
			}
			else
			{
				if (ndimsproj == 2)
				{
					if (outputdecentered)
						Backproject3DtoNDKernel<2, true, false> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_ivolume, m_magnification, ewalddiameterinv, rmax, rmax * rmax);
					else
						Backproject3DtoNDKernel<2, false, false> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_ivolume, m_magnification, ewalddiameterinv, rmax, rmax * rmax);
				}
				else if (ndimsproj == 3)
				{
					if (outputdecentered)
						Backproject3DtoNDKernel<3, true, false> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_ivolume, m_magnification, ewalddiameterinv, rmax, rmax * rmax);
					else
						Backproject3DtoNDKernel<3, false, false> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft, d_projweights, dimsproj.x, elements, d_matrices, d_ivolume, m_magnification, ewalddiameterinv, rmax, rmax * rmax);
				}
			}
		}
		else
		{
			/*cudaMemcpyToSymbol(c_backmatrices, d_matrices, batch * sizeof(glm::mat3), 0, cudaMemcpyDeviceToDevice);

			dim3 grid = dim3(1, batch, 1);
			uint elements = ElementsFFT(dimsproj);
			uint TpB = 1 << tmin(7, tmax(7, (uint)(log(elements / 4.0) / log(2.0))));

			if (TpB == 32)
			Project2Dto2DKernel<32> << <grid, 32 >> > (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
			else if (TpB == 64)
			Project2Dto2DKernel<64> << <grid, 64 >> > (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
			else if (TpB == 128)
			Project2Dto2DKernel<128> << <grid, 128 >> > (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
			else if (TpB == 256)
			Project2Dto2DKernel<256> << <grid, 256 >> > (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
			else
			throw;*/
		}
	}

	template<uint ndims, bool decentered, bool squareinterpweights> __global__ void Backproject3DtoNDKernel(tcomplex* d_volumeft,
																											tfloat* d_volumeweights,
																											uint dimvolume,
																											tcomplex* d_projft,
																											tfloat* d_projweights,
																											uint dimproj,
																											size_t elementsproj,
																											glm::mat3* d_rotations,
																											int* d_ivolume,
																											glm::mat2 magnification,
																											float ewalddiameterinv,
																											uint rmax,
																											int rmax2)
	{
		if (d_projft != NULL)
			d_projft += elementsproj * blockIdx.y;
		if (d_projweights != NULL)
			d_projweights += elementsproj * blockIdx.y;

		if (d_ivolume != NULL)
		{
			int ivolume = d_ivolume[blockIdx.y];
			if (d_volumeft != NULL)
				d_volumeft += ElementsFFT1(dimvolume) * dimvolume * dimvolume * ivolume;
			if (d_volumeweights != NULL)
				d_volumeweights += ElementsFFT1(dimvolume) * dimvolume * dimvolume * ivolume;
		}

		uint slice = ndims == 3 ? ElementsFFT1(dimproj) * dimproj : 1;
		uint dimft = ElementsFFT1(dimproj);
		uint dimvolumeft = ElementsFFT1(dimvolume);

		glm::mat3 rotation = d_rotations[blockIdx.y];

		for (uint id = threadIdx.x; id < elementsproj; id += blockDim.x)
		{
			uint idx = id % dimft;
			uint idy = (ndims == 3 ? id % slice : id) / dimft;
			uint idz = ndims == 3 ? id / slice : 0;

			int x = idx;
			int y = idy <= dimproj / 2 ? idy : (int)idy - (int)dimproj;
			int z = ndims == 3 ? (idz <= dimproj / 2 ? idz : (int)idz - (int)dimproj) : 0;

			if (ndims == 3)
			{
				if (x == 0 && y < 0 && z < 0)
					continue;
			}
			else
			{
				if (x == 0 && y < 0)
					continue;
			}

			glm::vec2 posmag = glm::vec2(x, y);
			if (ndims == 2)
				posmag = magnification * posmag;

			glm::vec3 pos = glm::vec3(posmag.x, posmag.y, z);
			if (ndims == 2)
				pos.z = ewalddiameterinv * (x * x + y * y);

			int r2 = ndims == 3 ? (z * z + y * y + x * x) : (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

			if (r2 >= rmax2)
				continue;

			pos = rotation * pos;

			// Trilinear interpolation
			short x0 = (short)floor(pos.x + 1e-5f);
			pos.x -= x0;
			short x1 = x0 + 1;

			short y0 = (short)floor(pos.y);
			pos.y -= y0;
			short y1 = y0 + 1;

			short z0 = (short)floor(pos.z);
			pos.z -= z0;
			short z1 = z0 + 1;

			short3 positions[8];

			positions[0] = make_short3(x0, y0, z0);
			positions[1] = make_short3(x1, y0, z0);
			positions[2] = make_short3(x0, y1, z0);
			positions[3] = make_short3(x1, y1, z0);

			positions[4] = make_short3(x0, y0, z1);
			positions[5] = make_short3(x1, y0, z1);
			positions[6] = make_short3(x0, y1, z1);
			positions[7] = make_short3(x1, y1, z1);

			float c0 = 1.0f - pos.z;
			float c1 = pos.z;

			half c00 = __float2half((1.0f - pos.y) * c0);
			half c10 = __float2half(pos.y * c0);
			half c01 = __float2half((1.0f - pos.y) * c1);
			half c11 = __float2half(pos.y * c1);

			half interpw[8];
			interpw[0] = __float2half((1.0f - pos.x) * __half2float(c00));
			interpw[1] = __float2half(pos.x * __half2float(c00));
			interpw[2] = __float2half((1.0f - pos.x) * __half2float(c10));
			interpw[3] = __float2half(pos.x * __half2float(c10));

			interpw[4] = __float2half((1.0f - pos.x) * __half2float(c01));
			interpw[5] = __float2half(pos.x * __half2float(c01));
			interpw[6] = __float2half((1.0f - pos.x) * __half2float(c11));
			interpw[7] = __float2half(pos.x * __half2float(c11));

			tcomplex val = make_float2(1, 0);
			if (d_projft != NULL)
				val = d_projft[id];

			tfloat weight = 1;
			if (d_projweights != NULL)
				weight = d_projweights[id];

			for (uint i = 0; i < 8; i++)
			{
				tcomplex valsym = val;
				short3 position = positions[i];

				if (positions[i].x < 0)
				{
					position.x *= -1;
					position.y *= -1;
					position.z *= -1;
					valsym.y *= -1;
				}

				if (decentered)
				{
					position.y = position.y < 0 ? position.y + dimvolume : position.y;
					position.z = position.z < 0 ? position.z + dimvolume : position.z;
				}
				else
				{
					position.y += dimvolume / 2;
					position.z += dimvolume / 2;
				}

				position.x = tmin(dimvolume / 2, position.x);
				position.y = tmax(0, tmin(dimvolume - 1, position.y));
				position.z = tmax(0, tmin(dimvolume - 1, position.z));

				float interpweight = __half2float(interpw[i]);
				if (squareinterpweights)
					interpweight *= interpweight;

				if (d_volumeft != NULL)
				{
					atomicAdd((tfloat*)(d_volumeft + (position.z * dimvolume + position.y) * dimvolumeft + position.x), interpweight * valsym.x);
					atomicAdd((tfloat*)(d_volumeft + (position.z * dimvolume + position.y) * dimvolumeft + position.x) + 1, interpweight * valsym.y);
				}

				if (d_volumeweights != NULL)
					atomicAdd((tfloat*)(d_volumeweights + (position.z * dimvolume + position.y) * dimvolumeft + position.x), interpweight * weight);

				if (positions[i].x == 0 && (positions[i].y != 0 || positions[i].z != 0))
				{
					position = positions[i];
					position.x *= -1;
					position.y *= -1;
					position.z *= -1;

					if (decentered)
					{
						position.y = position.y < 0 ? position.y + dimvolume : position.y;
						position.z = position.z < 0 ? position.z + dimvolume : position.z;
					}
					else
					{
						position.y += dimvolume / 2;
						position.z += dimvolume / 2;
					}

					position.x = tmin(dimvolume / 2, position.x);
					position.y = tmax(0, tmin(dimvolume - 1, position.y));
					position.z = tmax(0, tmin(dimvolume - 1, position.z));

					if (d_volumeft != NULL)
					{
						atomicAdd((tfloat*)(d_volumeft + (position.z * dimvolume + position.y) * dimvolumeft + position.x), interpweight * valsym.x);
						atomicAdd((tfloat*)(d_volumeft + (position.z * dimvolume + position.y) * dimvolumeft + position.x) + 1, interpweight * (-valsym.y));
					}

					if (d_volumeweights != NULL)
						atomicAdd((tfloat*)(d_volumeweights + (position.z * dimvolume + position.y) * dimvolumeft + position.x), interpweight * weight);
				}
			}
		}
	}
}
