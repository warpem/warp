#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
#define SincWindow 16

	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template<bool cubicinterp, bool zerocentered> __global__ void GetFFTPlaneKernel(cudaTex t_volumeRe, cudaTex t_volumeIm, cudaTex t_volumepsf, int dim, uint dimft, uint n, tcomplex* d_imageft, tfloat* d_imagepsf, glm::mat2x3* d_rotations, tfloat2* d_shifts);
	template<bool zerocentered> __global__ void GetFFTPlaneSincKernel(tcomplex* d_volumeft, tfloat* d_volumepsf, int dim, uint dimft, uint elementsvolume, tcomplex* d_imageft, tfloat* d_imagepsf, glm::mat2x3* d_rotations, tfloat2* d_shifts);
	__global__ void IntersectionKernel(float* d_distmin, float* d_distmax, int2 dims, tfloat3 boxmin, tfloat3 boxmax, glm::vec3 invdirection, char3 signs, glm::mat4 transform);
	template <bool cubicinterp> __global__ void RaytraceVolumeKernel(cudaTex t_volume, int3 dimsvolume, tfloat* d_projection, int2 dimsimage, float* d_distmin, float* d_distmax, glm::vec3 direction, glm::mat4 transform);


	/////////////////////////////////////////
	//Equivalent of TOM's tom_proj3d method//
	/////////////////////////////////////////

	void d_ProjForward(tcomplex* d_volumeft, tfloat* d_volumepsf, int3 dimsvolume, tcomplex* d_projectionsft, tfloat* d_projectionspsf, tfloat3* h_angles, tfloat2* h_shifts, T_INTERP_MODE mode, bool outputzerocentered, int batch)
	{
		int2 dimsimage = toInt2(dimsvolume.x, dimsvolume.x);
		int3 dimsvolumeft = toInt3(dimsvolume.x / 2 + 1, dimsvolume.y, dimsvolume.z);

		// Calculate rotation matrices and shifts
		glm::mat2x3* h_matrices = (glm::mat2x3*)malloc(batch * sizeof(glm::mat2x3));
		tfloat2* h_normshifts = (tfloat2*)malloc(batch * sizeof(tfloat2));
		for (int i = 0; i < batch; i++)
		{
			glm::mat3 r = Matrix3Euler(h_angles[i]);
			h_matrices[i] = glm::mat2x3(r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2]);
			h_normshifts[i] = tfloat2(-h_shifts[i].x / (tfloat)dimsimage.x, -h_shifts[i].y / (tfloat)dimsimage.y);
		}
		glm::mat2x3* d_matrices = (glm::mat2x3*)CudaMallocFromHostArray(h_matrices, batch * sizeof(glm::mat2x3));
		tfloat2* d_normshifts = (tfloat2*)CudaMallocFromHostArray(h_normshifts, batch * sizeof(tfloat2));
		free(h_normshifts);
		free(h_matrices);

		if (mode == T_INTERP_LINEAR || mode == T_INTERP_CUBIC)
		{
			// Prefilter and bind 3D textures
			tfloat* d_tempRe, *d_tempIm;
			cudaMalloc((void**)&d_tempRe, ElementsFFT(dimsvolume) * sizeof(tfloat));
			cudaMalloc((void**)&d_tempIm, ElementsFFT(dimsvolume) * sizeof(tfloat));

			d_ConvertTComplexToSplitComplex(d_volumeft, d_tempRe, d_tempIm, ElementsFFT(dimsvolume));
			if (mode == T_INTERP_CUBIC)
			{
				d_CubicBSplinePrefilter3D(d_tempRe, dimsvolumeft);
				d_CubicBSplinePrefilter3D(d_tempIm, dimsvolumeft);
			}
			cudaArray_t a_volumeRe = 0, a_volumeIm = 0;
			cudaTex t_volumeRe = 0, t_volumeIm = 0;
			d_BindTextureTo3DArray(d_tempRe, a_volumeRe, t_volumeRe, dimsvolumeft, cudaFilterModeLinear, false);
			d_BindTextureTo3DArray(d_tempIm, a_volumeIm, t_volumeIm, dimsvolumeft, cudaFilterModeLinear, false);

			cudaMemcpy(d_tempRe, d_volumepsf, ElementsFFT(dimsvolume) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			if (mode == T_INTERP_CUBIC)
				d_CubicBSplinePrefilter3D(d_tempRe, dimsvolumeft);
			cudaArray_t a_volumepsf = 0;
			cudaTex t_volumepsf = 0;
			d_BindTextureTo3DArray(d_tempRe, a_volumepsf, t_volumepsf, dimsvolumeft, cudaFilterModeLinear, false);

			// Sample the planes
			uint TpB = tmin(128, NextMultipleOf(ElementsFFT2(dimsimage), 32));
			dim3 grid = dim3((ElementsFFT2(dimsimage) + TpB - 1) / TpB, batch);
			if (mode == T_INTERP_CUBIC)
				if (outputzerocentered)
					GetFFTPlaneKernel<true, true> << <grid, TpB >> > (t_volumeRe, t_volumeIm, t_volumepsf, dimsimage.x, dimsimage.x / 2 + 1, ElementsFFT2(dimsimage), d_projectionsft, d_projectionspsf, d_matrices, d_normshifts);
				else
					GetFFTPlaneKernel<true, false> << <grid, TpB >> > (t_volumeRe, t_volumeIm, t_volumepsf, dimsimage.x, dimsimage.x / 2 + 1, ElementsFFT2(dimsimage), d_projectionsft, d_projectionspsf, d_matrices, d_normshifts);
			else
				if (outputzerocentered)
					GetFFTPlaneKernel<false, true> << <grid, TpB >> > (t_volumeRe, t_volumeIm, t_volumepsf, dimsimage.x, dimsimage.x / 2 + 1, ElementsFFT2(dimsimage), d_projectionsft, d_projectionspsf, d_matrices, d_normshifts);
				else
					GetFFTPlaneKernel<false, false> << <grid, TpB >> > (t_volumeRe, t_volumeIm, t_volumepsf, dimsimage.x, dimsimage.x / 2 + 1, ElementsFFT2(dimsimage), d_projectionsft, d_projectionspsf, d_matrices, d_normshifts);

			// Tear down
			cudaDestroyTextureObject(t_volumeIm);
			cudaFreeArray(a_volumeIm);
			cudaDestroyTextureObject(t_volumeRe);
			cudaFreeArray(a_volumeRe);
			cudaDestroyTextureObject(t_volumepsf);
			cudaFreeArray(a_volumepsf);
			cudaFree(d_tempRe);
			cudaFree(d_tempIm);
		}
		else if (mode == T_INTERP_SINC)
		{
			uint TpB = 192;
			dim3 grid = dim3((ElementsFFT2(dimsimage) + TpB - 1) / TpB, batch);
			if (outputzerocentered)
				GetFFTPlaneSincKernel<true> << <grid, TpB >> > (d_volumeft, d_volumepsf, dimsvolume.x, dimsvolume.x / 2 + 1, ElementsFFT(dimsvolume), d_projectionsft, d_projectionspsf, d_matrices, d_normshifts);
			else
				GetFFTPlaneSincKernel<false> << <grid, TpB >> > (d_volumeft, d_volumepsf, dimsvolume.x, dimsvolume.x / 2 + 1, ElementsFFT(dimsvolume), d_projectionsft, d_projectionspsf, d_matrices, d_normshifts);
		}

		cudaFree(d_normshifts);
		cudaFree(d_matrices);
	}

	void d_ProjForward(tfloat* d_volume, tfloat* d_volumepsf, int3 dimsvolume, tfloat* d_projections, tfloat* d_projectionspsf, tfloat3* h_angles, tfloat2* h_shifts, T_INTERP_MODE mode, int batch)
	{
		int3 dimsimage = toInt3(dimsvolume.x, dimsvolume.y, 1);

		// Alloc buffers for FFTed volume and projections
		tcomplex* d_volumeft;
		cudaMalloc((void**)&d_volumeft, ElementsFFT(dimsvolume) * sizeof(tcomplex));
		tcomplex* d_projft;
		cudaMalloc((void**)&d_projft, ElementsFFT2(dimsimage) * batch * sizeof(tcomplex));

		d_FFTR2C(d_volume, d_volumeft, 3, dimsvolume);
		d_RemapHalfFFT2Half(d_volumeft, d_volumeft, dimsvolume);

		// Sample planes and transform back into real space
		d_ProjForward(d_volumeft, d_volumepsf, dimsvolume, d_projft, d_projectionspsf, h_angles, h_shifts, mode, false, batch);
		d_IFFTC2R(d_projft, d_projections, 2, dimsimage, batch);

		// Tear down
		cudaFree(d_projft);
		cudaFree(d_volumeft);
	}

	void d_ProjForwardRaytrace(tfloat* d_volume, int3 dimsvolume, tfloat3 volumeoffset, tfloat* d_projections, int2 dimsproj, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int supersample, int batch)
	{
		dimsproj = toInt2(dimsproj.x * supersample, dimsproj.y * supersample);
		dimsvolume = toInt3(dimsvolume.x * supersample, dimsvolume.y * supersample, dimsvolume.z * supersample);

		tfloat* d_superproj, *d_supervolume;
		if (supersample > 1)
		{
			cudaMalloc((void**)&d_superproj, Elements2(dimsproj) * batch * sizeof(tfloat));
			cudaMalloc((void**)&d_supervolume, Elements(dimsvolume) * sizeof(tfloat));
			d_Scale(d_volume, d_supervolume, toInt3(dimsvolume.x / supersample, dimsvolume.y / supersample, dimsvolume.z / supersample), dimsvolume, T_INTERP_FOURIER);
		}
		else
		{
			d_superproj = d_projections;
			d_supervolume = d_volume;
		}

		tfloat* d_prefilteredvolume;
		if (mode == T_INTERP_CUBIC)
			cudaMalloc((void**)&d_prefilteredvolume, Elements(dimsvolume) * sizeof(tfloat));

		float* d_distmin, *d_distmax;
		cudaMalloc((void**)&d_distmin, Elements2(dimsproj) * batch * sizeof(float));
		cudaMalloc((void**)&d_distmax, Elements2(dimsproj) * batch * sizeof(float));

		glm::mat4* h_raytransforms = (glm::mat4*)malloc(batch * sizeof(glm::mat4));
		for (int n = 0; n < batch; n++)
			h_raytransforms[n] = Matrix4Translation(tfloat3(dimsvolume.x / 2 + 0.5f, dimsvolume.y / 2 + 0.5f, dimsvolume.z / 2 + 0.5f)) *
								Matrix4Translation(tfloat3(-volumeoffset.x, -volumeoffset.y, -volumeoffset.z)) *
								Matrix4Euler(h_angles[n]) *
								Matrix4Translation(tfloat3(h_offsets[n].x * supersample, h_offsets[n].y * supersample, 0.0f)) *
								Matrix4Scale(tfloat3(h_scales[n].x, h_scales[n].y, 1.0f)) *
								Matrix4Translation(tfloat3(-dimsproj.x / 2, -dimsproj.y / 2, 0));

		tfloat3 boxmin = tfloat3(0, 0, 0);
		tfloat3 boxmax = tfloat3(dimsvolume.x,
			dimsvolume.y,
			dimsvolume.z);
		for (int n = 0; n < batch; n++)
		{
			int TpB = min(NextMultipleOf(dimsproj.x, 32), 256);
			dim3 grid = dim3((dimsproj.x + TpB - 1) / TpB, dimsproj.y);
			glm::vec3 direction = Matrix3Euler(h_angles[n]) * glm::vec3(0.0f, 0.0f, -1.0f);
			glm::vec3 invdirection = glm::vec3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
			char3 signs = make_char3(invdirection.x < 0.0f ? 1 : 0, invdirection.y < 0.0f ? 1 : 0, invdirection.z < 0.0f ? 1 : 0);

			IntersectionKernel << <grid, TpB >> > (d_distmin + Elements2(dimsproj) * n, d_distmax + Elements2(dimsproj) * n, dimsproj, boxmin, boxmax, invdirection, signs, h_raytransforms[n]);
		}

		cudaArray* a_volume;
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
		dim3 grid = dim3((dimsproj.x + 15) / 16, (dimsproj.y + 15) / 16);
		for (int n = 0; n < batch; n++)
		{
			glm::vec3 direction = Matrix3Euler(h_angles[n]) * glm::vec3(0.0f, 0.0f, -1.0f);
			if (mode == T_INTERP_CUBIC)
				RaytraceVolumeKernel<true> << <grid, TpB >> > (t_volume,
				dimsvolume,
				d_superproj + Elements2(dimsproj) * n,
				dimsproj,
				d_distmin + Elements2(dimsproj) * n,
				d_distmax + Elements2(dimsproj) * n,
				direction,
				h_raytransforms[n]);
			else
				RaytraceVolumeKernel<false> << <grid, TpB >> > (t_volume,
				dimsvolume,
				d_superproj + Elements2(dimsproj) * n,
				dimsproj,
				d_distmin + Elements2(dimsproj) * n,
				d_distmax + Elements2(dimsproj) * n,
				direction,
				h_raytransforms[n]);
		}

		cudaDestroyTextureObject(t_volume);
		cudaFreeArray(a_volume);

		if (supersample > 1)
		{
			d_Scale(d_superproj, d_projections, toInt3(dimsproj), toInt3(dimsproj.x / supersample, dimsproj.y / supersample, 1), T_INTERP_FOURIER);
		}

		free(h_raytransforms);
		cudaFree(d_distmax);
		cudaFree(d_distmin);
		if (mode == T_INTERP_CUBIC)
			cudaFree(d_prefilteredvolume);
		if (supersample > 1)
		{
			cudaFree(d_supervolume);
			cudaFree(d_superproj);
		}
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<bool cubicinterp, bool zerocentered> __global__ void GetFFTPlaneKernel(cudaTex t_volumeRe, cudaTex t_volumeIm, cudaTex t_volumepsf, int dim, uint dimft, uint n, tcomplex* d_imageft, tfloat* d_imagepsf, glm::mat2x3* d_rotations, tfloat2* d_shifts)
	{
		uint id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n)
			return;

		int idx = id % dimft;
		int idy = id / dimft;

		d_imageft += blockIdx.y * n;
		d_imagepsf += blockIdx.y * n;

		int x, y;
		if (zerocentered)
		{
			x = idx;
			y = idy;
		}
		else
		{
			x = dim / 2 - idx;
			y = dim - 1 - ((idy + dim / 2 - 1) % dim);
		}

		glm::vec2 poslocal = glm::vec2(x, y);
		poslocal -= (float)(dim / 2);
		if (poslocal.x * poslocal.x + poslocal.y * poslocal.y >= (float)(dim * dim / 4))
		{
			d_imageft[id] = make_cuComplex(0, 0);
			d_imagepsf[id] = 0;
			return;
		}

		glm::vec3 posglobal = d_rotations[blockIdx.y] * poslocal;
		bool flip = false;
		if (posglobal.x > 0)
		{
			posglobal = -posglobal;
			flip = true;
		}
		posglobal += (float)(dimft - 1) + 0.5f;

		tfloat valRe = 0, valIm = 0, valpsf = 0;
		if (cubicinterp)
		{
			valRe = cubicTex3D(t_volumeRe, posglobal.x, posglobal.y, posglobal.z);
			valIm = cubicTex3D(t_volumeIm, posglobal.x, posglobal.y, posglobal.z);
			valpsf = cubicTex3D(t_volumepsf, posglobal.x, posglobal.y, posglobal.z);
		}
		else
		{
			valRe = tex3D<tfloat>(t_volumeRe, posglobal.x, posglobal.y, posglobal.z);
			valIm = tex3D<tfloat>(t_volumeIm, posglobal.x, posglobal.y, posglobal.z);
			valpsf = tex3D<tfloat>(t_volumepsf, posglobal.x, posglobal.y, posglobal.z);
		}

		if (flip)
			valIm = -valIm;

		tfloat2 delta = d_shifts[blockIdx.y];
		tfloat factor = (delta.x * poslocal.x + delta.y * poslocal.y) * PI2;

		d_imageft[id] = cmul(make_cuComplex(valRe, valIm), make_cuComplex(cos(factor), sin(factor)));
		d_imagepsf[id] = valpsf;
	}

	template<bool zerocentered> __global__ void GetFFTPlaneSincKernel(tcomplex* d_volumeft, tfloat* d_volumepsf, int dim, uint dimft, uint elementsvolume, tcomplex* d_imageft, tfloat* d_imagepsf, glm::mat2x3* d_rotations, tfloat2* d_shifts)
	{
		__shared__ tfloat s_volumeftRe[192], s_volumeftIm[192];
		__shared__ tfloat s_volumeweights[192];

		uint id = blockIdx.x * blockDim.x + threadIdx.x;

		d_imageft += blockIdx.y * dimft * dim;
		d_imagepsf += blockIdx.y * dimft * dim;

		uint x, y;
		if (zerocentered)
		{
			x = id % dimft;
			y = id / dimft;
		}
		else
		{
			x = dim / 2 - id % dimft;
			y = dim - 1 - ((id / dimft + dim / 2 - 1) % dim);
		}

		tcomplex sumft = make_cuComplex(0, 0);
		tfloat sumweight = 0;

		float center = dim / 2;
		glm::vec2 pos = glm::vec2(x, y);
		pos -= center;
		glm::vec3 posglobal = d_rotations[blockIdx.y] * pos;

		for (uint poffset = 0; poffset < elementsvolume; poffset += 192)
		{
			{
				uint pglobal = poffset + threadIdx.x;
				if (pglobal < elementsvolume)
				{
					s_volumeftRe[threadIdx.x] = d_volumeft[pglobal].x;
					s_volumeftIm[threadIdx.x] = d_volumeft[pglobal].y;
					s_volumeweights[threadIdx.x] = d_volumepsf[pglobal];
				}
			}

			__syncthreads();

			uint plimit = min(192, elementsvolume - poffset);
			for (uint p = 0; p < plimit; p++)
			{
				uint pglobal = poffset + p;
				uint px = pglobal % dimft;
				uint py = (pglobal / dimft) % dim;
				uint pz = pglobal / (dimft * dim);

				glm::vec3 pospixel = glm::vec3(px, py, pz) - center;

				if (pospixel.x * pospixel.x + pospixel.y * pospixel.y + pospixel.z * pospixel.z < dim * dim * 4)
				{
					float s = sinc(pospixel.x - posglobal.x) * sinc(pospixel.y - posglobal.y) * sinc(pospixel.z - posglobal.z);

					tcomplex val = make_cuComplex(s_volumeftRe[p], s_volumeftIm[p]);
					tfloat valweight = s_volumeweights[p];

					sumft += val * s;
					sumweight += valweight * s;

					if (px == dim / 2)
						continue;

					s = sinc(-pospixel.x - posglobal.x) * sinc(-pospixel.y - posglobal.y) * sinc(-pospixel.z - posglobal.z);

					val.y = -val.y;

					sumft += val * s;
					sumweight += valweight * s;
				}
			}

			__syncthreads();
		}

		if (id >= dimft * dim)
			return;

		float falloff = pos.x * pos.x + pos.y * pos.y >= (float)(dim * dim / 4) ? 0.0f : 1.0f;
		sumft *= falloff;
		sumweight *= falloff;

		tfloat2 delta = d_shifts[blockIdx.y];
		tfloat factor = (delta.x * pos.x + delta.y * pos.y) * PI2;

		d_imageft[id] = cmul(sumft, make_cuComplex(cos(factor), sin(factor)));
		d_imagepsf[id] = sumweight;
	}

	__global__ void IntersectionKernel(float* d_distmin, float* d_distmax, int2 dims, tfloat3 boxmin, tfloat3 boxmax, glm::vec3 invdirection, char3 signs, glm::mat4 transform)
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

	template <bool cubicinterp> __global__ void RaytraceVolumeKernel(cudaTex t_volume, int3 dimsvolume, tfloat* d_projection, int2 dimsimage, float* d_distmin, float* d_distmax, glm::vec3 direction, glm::mat4 transform)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dimsimage.x)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dimsimage.y)
			return;

		float distmin = d_distmin[idy * dimsimage.x + idx];
		float distmax = d_distmax[idy * dimsimage.x + idx];
		d_projection += idy * dimsimage.x + idx;

		float pathlength = distmax - distmin;
		ushort steps = ceil(pathlength * 5.0f);
		double sum = 0.0;
		if (steps > 0)
		{
			float steplength = pathlength / (float)steps;
			glm::vec4 origin4 = transform * glm::vec4((float)idx, (float)idy, 9999.0f, 1.0f);
			glm::vec3 origin = glm::vec3(origin4.x, origin4.y, origin4.z);
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

		*d_projection = sum;
	}
}