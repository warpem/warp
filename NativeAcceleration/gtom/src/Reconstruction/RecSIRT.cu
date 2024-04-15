#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/Projection.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void PathlengthKernel(tfloat* d_distmin, tfloat* d_distmax, int2 dims, tfloat3 boxmin, tfloat3 boxmax, glm::vec3 invdirection, char3 signs, glm::mat4 transform);
	template <bool cubicinterp> __global__ void RaytraceVolumeKernel(cudaTex t_volume, uint3 dimsvolume, tfloat* d_image, tfloat* d_projection, uint2 dimsimage, float* d_distmin, float* d_distmax, glm::vec3* d_directions, glm::mat4* d_transforms, tfloat2* d_intensities);
	template <bool cubicinterp> __global__ void CorrectionKernel(tfloat* d_volume, uint3 dimsvolume, cudaTex* t_image, uint2 dimsimage, uint nimages, glm::mat4* d_transform);
	template <bool cubicinterp> __global__ void ResidualKernel(tfloat* d_residual, uint3 dimsvolume, cudaTex* t_image, uint2 dimsimage, uint nimages, glm::mat4* d_transform);


	/////////////////////////////////////////
	//Performs 3D reconstruction using SIRT//
	/////////////////////////////////////////

	void d_RecSIRT(tfloat* d_volume, tfloat* d_residual, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, int nimages, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, tfloat2* h_intensities, T_INTERP_MODE mode, int supersample, int iterations, bool outputzerocentered)
	{
		dimsimage = toInt2(dimsimage.x * supersample, dimsimage.y * supersample);
		dimsvolume = toInt3(dimsvolume.x * supersample, dimsvolume.y * supersample, dimsvolume.z * supersample);

		tfloat* d_superimage, *d_supervolume, *d_superresidual;
		if (supersample > 1)
		{
			cudaMalloc((void**)&d_superimage, Elements2(dimsimage) * nimages * sizeof(tfloat));
			cudaMalloc((void**)&d_supervolume, Elements(dimsvolume) * sizeof(tfloat));
			d_Scale(d_image, d_superimage, toInt3(dimsimage.x / supersample, dimsimage.y / supersample, 1), toInt3(dimsimage), T_INTERP_FOURIER, NULL, NULL, nimages);
			if (d_residual != NULL)
				cudaMalloc((void**)&d_superresidual, Elements(dimsvolume) * sizeof(tfloat));
		}
		else
		{
			d_superimage = d_image;
			d_supervolume = d_volume;
			d_superresidual = d_residual;
		}
		//CudaWriteToBinaryFile("d_superimage.bin", d_superimage, Elements2(dimsimage) * nimages * sizeof(tfloat));

		d_ValueFill(d_supervolume, Elements(dimsvolume), 0.0f);
		tfloat* d_prefilteredvolume;
		if (mode == T_INTERP_CUBIC)
			cudaMalloc((void**)&d_prefilteredvolume, Elements(dimsvolume) * sizeof(tfloat));

		float* d_distmin, *d_distmax;
		cudaMalloc((void**)&d_distmin, Elements2(dimsimage) * nimages * sizeof(float));
		cudaMalloc((void**)&d_distmax, Elements2(dimsimage) * nimages * sizeof(float));

		tfloat* d_reprojections;
		cudaMalloc((void**)&d_reprojections, Elements2(dimsimage) * nimages * sizeof(tfloat));

		glm::mat4* h_raytransforms = (glm::mat4*)malloc(nimages * sizeof(glm::mat4));
		for (int n = 0; n < nimages; n++)
			h_raytransforms[n] = Matrix4Translation(tfloat3(dimsvolume.x / 2 + 0.5f, dimsvolume.y / 2 + 0.5f, dimsvolume.z / 2 + 0.5f)) *
			Matrix4Euler(h_angles[n]) *
			Matrix4Translation(tfloat3(h_offsets[n].x * supersample, h_offsets[n].y * supersample, 0.0f)) *
			Matrix4Scale(tfloat3(h_scales[n].x, h_scales[n].y, 1.0f)) *
			Matrix4Translation(tfloat3(-dimsimage.x / 2, -dimsimage.y / 2, 0));
		glm::mat4* d_raytransforms = (glm::mat4*)CudaMallocFromHostArray(h_raytransforms, nimages * sizeof(glm::mat4));

		glm::mat4* h_transforms = (glm::mat4*)malloc(nimages * sizeof(glm::mat4));
		for (int n = 0; n < nimages; n++)
		{
			h_transforms[n] = Matrix4Translation(tfloat3(dimsimage.x / 2 + 0.5f, dimsimage.y / 2 + 0.5f, 0.0f)) *
				Matrix4Scale(tfloat3(1.0f / h_scales[n].x, 1.0f / h_scales[n].y, 1.0f)) *
				Matrix4Translation(tfloat3(-h_offsets[n].x * supersample, -h_offsets[n].y * supersample, 0.0f)) *
				glm::transpose(Matrix4Euler(h_angles[n])) *
				Matrix4Translation(offsetfromcenter) *
				Matrix4Translation(tfloat3(-dimsvolume.x / 2, -dimsvolume.y / 2, -dimsvolume.z / 2));
		}
		glm::mat4* d_transforms = (glm::mat4*)CudaMallocFromHostArray(h_transforms, nimages * sizeof(glm::mat4));

		tfloat2* d_intensities = (tfloat2*)CudaMallocFromHostArray(h_intensities, nimages * sizeof(tfloat2));

		glm::vec3* h_directions = (glm::vec3*)malloc(nimages * sizeof(glm::vec3));
		for (int n = 0; n < nimages; n++)
			h_directions[n] = Matrix3Euler(h_angles[n]) * glm::vec3(0.0f, 0.0f, -1.0f);
		glm::vec3* d_directions = (glm::vec3*)CudaMallocFromHostArray(h_directions, nimages * sizeof(glm::vec3));

		tfloat3 boxmin = offsetfromcenter;
		tfloat3 boxmax = tfloat3(dimsvolume.x + offsetfromcenter.x,
			dimsvolume.y + offsetfromcenter.y,
			dimsvolume.z + offsetfromcenter.z);
		for (int n = 0; n < nimages; n++)
		{
			int TpB = tmin(NextMultipleOf(dimsimage.x, 32), 256);
			dim3 grid = dim3((dimsimage.x + TpB - 1) / TpB, dimsimage.y);
			glm::vec3 direction = Matrix3Euler(h_angles[n]) * glm::vec3(0.0f, 0.0f, -1.0f);
			glm::vec3 invdirection = glm::vec3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
			char3 signs = make_char3(invdirection.x < 0.0f ? 1 : 0, invdirection.y < 0.0f ? 1 : 0, invdirection.z < 0.0f ? 1 : 0);

			PathlengthKernel << <grid, TpB >> > (d_distmin + Elements2(dimsimage) * n, d_distmax + Elements2(dimsimage) * n, dimsimage, boxmin, boxmax, invdirection, signs, h_raytransforms[n]);
		}
		//CudaWriteToBinaryFile("d_distmin.bin", d_distmin, Elements2(dimsimage) * nimages * sizeof(tfloat));
		//CudaWriteToBinaryFile("d_distmax.bin", d_distmax, Elements2(dimsimage) * nimages * sizeof(tfloat));

		for (int i = 0; i < iterations; i++)
		{
			cudaArray_t a_volume;
			cudaTex t_volume;

			if (mode == T_INTERP_CUBIC)
			{
				cudaMemcpy(d_prefilteredvolume, d_supervolume, Elements(dimsvolume) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
				d_CubicBSplinePrefilter3D(d_prefilteredvolume, dimsvolume);
				d_BindTextureTo3DArray(d_prefilteredvolume, a_volume, t_volume, dimsvolume, cudaFilterModeLinear, false);
			}
			else
			{
				d_BindTextureTo3DArray(d_supervolume, a_volume, t_volume, dimsvolume, cudaFilterModeLinear, false);
			}

			dim3 TpB = dim3(16, 16);
			dim3 grid = dim3((dimsimage.x + 15) / 16, (dimsimage.y + 15) / 16, nimages);

			if (mode == T_INTERP_CUBIC)
				RaytraceVolumeKernel<true> << <grid, TpB >> > (t_volume,
																toUint3(dimsvolume),
																d_superimage,
																d_reprojections,
																toUint2(dimsimage),
																d_distmin,
																d_distmax,
																d_directions,
																d_raytransforms,
																d_intensities);
			else
				RaytraceVolumeKernel<false> << <grid, TpB >> > (t_volume,
																toUint3(dimsvolume),
																d_superimage,
																d_reprojections,
																toUint2(dimsimage),
																d_distmin,
																d_distmax,
																d_directions,
																d_raytransforms,
																d_intensities);
			cudaDestroyTextureObject(t_volume);
			cudaFreeArray(a_volume);

			//CudaWriteToBinaryFile("d_reprojections.bin", d_reprojections, Elements2(dimsimage) * nimages * sizeof(tfloat));

			//Backproject and correct
			{
				cudaArray_t* ha_image = (cudaArray_t*)malloc(nimages * sizeof(cudaArray_t));
				cudaTex* ht_image = (cudaTex*)malloc(nimages * sizeof(cudaTex));
				tfloat* d_temp;
				cudaMalloc((void**)&d_temp, Elements2(dimsimage) * nimages * sizeof(tfloat));

				if (mode == T_INTERP_CUBIC)
				{
					cudaMemcpy(d_temp, d_reprojections, Elements2(dimsimage) * nimages * sizeof(tfloat), cudaMemcpyDeviceToDevice);
					d_CubicBSplinePrefilter2D(d_temp, dimsimage, nimages);
					d_BindTextureToArray(d_temp, ha_image, ht_image, dimsimage, cudaFilterModeLinear, false, nimages);
				}
				else
					d_BindTextureToArray(d_reprojections, ha_image, ht_image, dimsimage, cudaFilterModeLinear, false, nimages);

				cudaTex* dt_image = (cudaTex*)CudaMallocFromHostArray(ht_image, nimages * sizeof(cudaTex));

				dim3 TpB = dim3(8, 8, 8);
				dim3 grid = dim3((dimsvolume.x + 7) / 8, (dimsvolume.y + 7) / 8, (dimsvolume.z + 7) / 8);

				if (mode == T_INTERP_LINEAR)
				{
					CorrectionKernel<false> << <grid, TpB >> > (d_supervolume, toUint3(dimsvolume), dt_image, toUint2(dimsimage), nimages, d_transforms);
					if (i == iterations - 1 && d_superresidual != NULL)
						ResidualKernel<false> << <grid, TpB >> > (d_residual, toUint3(dimsvolume), dt_image, toUint2(dimsimage), nimages, d_transforms);
				}
				else
				{
					CorrectionKernel<true> << <grid, TpB >> > (d_supervolume, toUint3(dimsvolume), dt_image, toUint2(dimsimage), nimages, d_transforms);
					if (i == iterations - 1 && d_superresidual != NULL)
						ResidualKernel<true> << <grid, TpB >> > (d_residual, toUint3(dimsvolume), dt_image, toUint2(dimsimage), nimages, d_transforms);
				}

				for (int n = 0; n < nimages; n++)
				{
					cudaDestroyTextureObject(ht_image[n]);
					cudaFreeArray(ha_image[n]);
				}

				free(ht_image);
				free(ha_image);
				cudaFree(dt_image);
				cudaFree(d_temp);
			}
		}

		if (supersample > 1)
		{
			d_Scale(d_supervolume, d_volume, dimsvolume, toInt3(dimsvolume.x / supersample, dimsvolume.y / supersample, dimsvolume.z / supersample), T_INTERP_FOURIER);
			if (d_residual != NULL)
				d_Scale(d_superresidual, d_residual, dimsvolume, toInt3(dimsvolume.x / supersample, dimsvolume.y / supersample, dimsvolume.z / supersample), T_INTERP_FOURIER);
		}

		free(h_directions);
		free(h_transforms);
		free(h_raytransforms);
		cudaFree(d_intensities);
		cudaFree(d_directions);
		cudaFree(d_transforms);
		cudaFree(d_raytransforms);
		cudaFree(d_reprojections);
		cudaFree(d_distmax);
		cudaFree(d_distmin);
		if (mode == T_INTERP_CUBIC)
			cudaFree(d_prefilteredvolume);
		if (supersample > 1)
		{
			cudaFree(d_supervolume);
			cudaFree(d_superimage);
			if (d_residual != NULL)
				cudaFree(d_superresidual);
		}
	}

	__global__ void PathlengthKernel(float* d_distmin, float* d_distmax, int2 dims, tfloat3 boxmin, tfloat3 boxmax, glm::vec3 invdirection, char3 signs, glm::mat4 transform)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;
		int idy = blockIdx.y;
		d_distmin += idy * dims.x + idx;
		d_distmax += idy * dims.x + idx;

		glm::vec4 origin = transform * glm::vec4((float)idx, (float)idy, 9999.0f, 1.0f);

		float tmin, tmax, tymin, tymax, tzmin, tzmax;

		tmin = ((signs.x ? boxmax.x : boxmin.x) - origin.x) * invdirection.x;
		tmax = ((signs.x ? boxmin.x : boxmax.x) - origin.x) * invdirection.x;
		tymin = ((signs.y ? boxmax.y : boxmin.y) - origin.y) * invdirection.y;
		tymax = ((signs.y ? boxmin.y : boxmax.y) - origin.y) * invdirection.y;
		if ((tmin > tymax) || (tymin > tmax))
		{
			*d_distmin = 0.0f;
			*d_distmax = 0.0f;
			return;
		}
		if (tymin > tmin)
			tmin = tymin;
		if (tymax < tmax)
			tmax = tymax;
		tzmin = ((signs.z ? boxmax.z : boxmin.z) - origin.z) * invdirection.z;
		tzmax = ((signs.z ? boxmin.z : boxmax.z) - origin.z) * invdirection.z;
		if ((tmin > tzmax) || (tzmin > tmax))
		{
			*d_distmin = 0.0f;
			*d_distmax = 0.0f;
			return;
		}
		if (tzmin > tmin)
			tmin = tzmin;
		if (tzmax < tmax)
			tmax = tzmax;

		if (!isnan(tmin) && !isnan(tmax))
		{
			*d_distmin = tmin;
			*d_distmax = tmax;
		}
		else
		{
			*d_distmin = 0.0f;
			*d_distmax = 0.0f;
		}
	}

	template <bool cubicinterp> __global__ void RaytraceVolumeKernel(cudaTex t_volume, uint3 dimsvolume, tfloat* d_image, tfloat* d_projection, uint2 dimsimage, float* d_distmin, float* d_distmax, glm::vec3* d_directions, glm::mat4* d_transforms, tfloat2* d_intensities)
	{
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dimsimage.x)
			return;
		uint idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dimsimage.y)
			return;

		uint offset = dimsimage.x * dimsimage.y * blockIdx.z;
		d_image += offset;
		d_projection += offset;
		d_distmin += offset;
		d_distmax += offset;
		d_directions += blockIdx.z;
		d_transforms += blockIdx.z;

		float distmin = d_distmin[idy * dimsimage.x + idx];
		float distmax = d_distmax[idy * dimsimage.x + idx];
		d_projection += idy * dimsimage.x + idx;
		d_image += idy * dimsimage.x + idx;

		float pathlength = distmax - distmin;
		ushort steps = ceil(pathlength * 5.0f);
		float sum = 0.0;
		if (pathlength > 0.1f)
		{
			float steplength = pathlength / (float)steps;
			glm::vec4 origin4 = *d_transforms * glm::vec4((float)idx, (float)idy, 9999.0f, 1.0f);
			glm::vec3 origin = glm::vec3(origin4.x, origin4.y, origin4.z);
			glm::vec3 direction = *d_directions;
			distmin += steplength / 2.0f;

			for (ushort i = 0; i < steps; i++)
			{
				glm::vec3 point = (distmin + (float)i * steplength) * direction + origin;
				if (cubicinterp)
					sum += cubicTex3D(t_volume, point.x, point.y, point.z) * steplength;
				else
					sum += tex3D<tfloat>(t_volume, point.x, point.y, point.z) * steplength;
			}
		}

		tfloat2 intensity = d_intensities[blockIdx.z];

		if (pathlength > 0.1f)
			sum = ((*d_image + intensity.x) * intensity.y - sum) / pathlength;

		*d_projection = sum;
	}

	template <bool cubicinterp> __global__ void CorrectionKernel(tfloat* d_volume, uint3 dimsvolume, cudaTex* t_image, uint2 dimsimage, uint nimages, glm::mat4* d_transform)
	{
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dimsvolume.x)
			return;
		uint idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dimsvolume.y)
			return;
		uint idz = blockIdx.z * blockDim.z + threadIdx.z;
		if (idz >= dimsvolume.z)
			return;

		tfloat sum = 0.0f;
		uint samples = 0;

		float2 dimsfloat = make_float2(dimsimage.x, dimsimage.y);

		for (uint n = 0; n < nimages; n++)
		{
			glm::vec4 position = glm::vec4(idx, idy, idz, 1);
			position = d_transform[n] * position;
			if (position.x < 0.0f || position.x > dimsfloat.x || position.y < 0.0f || position.y > dimsfloat.y)
				continue;

			if (cubicinterp)
				sum += cubicTex2D(t_image[n], position.x, position.y);
			else
				sum += tex2D<tfloat>(t_image[n], position.x, position.y);
			samples++;
		}

		d_volume[(idz * dimsvolume.y + idy) * dimsvolume.x + idx] += samples > 0 ? sum / (tfloat)samples : 0.0f;
	}

	template <bool cubicinterp> __global__ void ResidualKernel(tfloat* d_residual, uint3 dimsvolume, cudaTex* t_image, uint2 dimsimage, uint nimages, glm::mat4* d_transform)
	{
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dimsvolume.x)
			return;
		uint idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dimsvolume.y)
			return;
		uint idz = blockIdx.z * blockDim.z + threadIdx.z;
		if (idz >= dimsvolume.z)
			return;

		tfloat sum = 0.0f;
		int samples = 0;

		float2 dimsfloat = make_float2(dimsimage.x, dimsimage.y);

		for (uint n = 0; n < nimages; n++)
		{
			glm::vec4 position = glm::vec4(idx, idy, idz, 1);
			position = d_transform[n] * position;
			if (position.x < 0.0f || position.x > dimsfloat.x || position.y < 0.0f || position.y > dimsfloat.y)
				continue;

			tfloat val;
			if (cubicinterp)
				val = cubicTex2D(t_image[n], position.x, position.y);
			else
				val = tex2D<tfloat>(t_image[n], position.x, position.y);
			sum += val * val;
			samples++;
		}

		if (samples > 0)
			sum = sqrt(sum / (tfloat)samples);

		d_residual[(idz * dimsvolume.y + idy) * dimsvolume.x + idx] = sum;
	}
}