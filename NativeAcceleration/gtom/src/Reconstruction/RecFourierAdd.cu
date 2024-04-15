#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Reconstruction.cuh"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_INLINE
#define GLM_FORCE_CUDA
#include "gtom/include/glm/glm.hpp"
#include "gtom/include/glm/gtc/matrix_transform.hpp"
#include "gtom/include/glm/gtx/quaternion.hpp"
#include "gtom/include/glm/gtx/euler_angles.hpp"
#include "gtom/include/glm/gtc/type_ptr.hpp"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void ReconstructFourierKernel(tcomplex* d_imageft, tfloat* d_imagepsf, tcomplex* d_volumeft, tfloat* d_volumepsf, uint dim, uint dimft, uint n, glm::mat2x3* d_rotations, tfloat2* d_shifts);
	template<bool zerocentered, bool finalize> __global__ void ReconstructFourierPreciseKernel(tcomplex* d_volumeft, tfloat* d_samples, int dim, uint dimft, cudaTex* t_imageftRe, cudaTex* t_imageftIm, cudaTex* t_imageweights, glm::mat3x2* d_global2local, glm::vec3* d_normals, tfloat2* d_shifts, uint nimages);
	template<bool zerocentered, bool finalize> __global__ void ReconstructFourierPreciseSincKernel(tcomplex* d_volumeft, tfloat* d_samples, uint dim, uint dimft, uint elementsimage, tcomplex* d_imagesft, tfloat* d_imageweights, glm::mat3* d_rotations, tfloat2* d_shifts, uint nimages);

	//////////////////////////////////////////////////////
	//Performs 3D reconstruction using Fourier inversion//
	//////////////////////////////////////////////////////

	void d_ReconstructFourierAdd(tcomplex* d_volumeft, tfloat* d_volumepsf, int3 dims, tcomplex* d_imagesft, tfloat* d_imagespsf, tfloat3* h_angles, tfloat2* h_shifts, int nimages)
	{
		int3 dimsimage = toInt3(dims.x, dims.y, 1);

		glm::mat2x3* h_rotations = (glm::mat2x3*)malloc(nimages * sizeof(glm::mat2x3));
		tfloat2* h_normshifts = (tfloat2*)malloc(nimages * sizeof(tfloat2));

		for (int n = 0; n < nimages; n++)
		{
			glm::mat3 m = Matrix3Euler(h_angles[n]);
			h_rotations[n] = glm::mat2x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2]);
			h_normshifts[n] = tfloat2(h_shifts[n].x / (tfloat)dimsimage.x, h_shifts[n].y / (tfloat)dimsimage.y);
		}

		glm::mat2x3* d_rotations = (glm::mat2x3*)CudaMallocFromHostArray(h_rotations, nimages * sizeof(glm::mat2x3));
		tfloat2* d_normshifts = (tfloat2*)CudaMallocFromHostArray(h_normshifts, nimages * sizeof(tfloat2));
		free(h_rotations);
		free(h_normshifts);

		int TpB = min(NextMultipleOf(ElementsFFT2(dimsimage), 32), 192);
		dim3 grid = dim3((ElementsFFT2(dimsimage) + TpB - 1) / TpB, nimages);
		ReconstructFourierKernel << <grid, TpB >> > (d_imagesft, d_imagespsf, d_volumeft, d_volumepsf, dims.x, dims.x / 2 + 1, ElementsFFT2(dimsimage), d_rotations, d_normshifts);

		cudaFree(d_rotations);
		cudaFree(d_normshifts);
	}

	void d_ReconstructFourierPreciseAdd(tcomplex* d_volumeft, tfloat* d_samples, int3 dims, tfloat* d_images, tfloat* d_imagespsf, tfloat3* h_angles, tfloat2* h_shifts, int nimages, T_INTERP_MODE mode, bool outputzerocentered, bool finalize)
	{
		// FFT and remap projections
		int3 dimsimage = toInt3(dims.x, dims.y, 1);
		tcomplex* d_imagesft;
		cudaMalloc((void**)&d_imagesft, ElementsFFT(dimsimage) * nimages * sizeof(tcomplex));
		d_FFTR2C(d_images, d_imagesft, 2, dimsimage, nimages);
		d_RemapHalfFFT2Half(d_imagesft, d_imagesft, dimsimage, nimages);

		if (mode == T_INTERP_LINEAR)
		{
			// Prefilter FFTed projections and bind to textures
			int2 dimsft = toInt2(ElementsFFT1(dimsimage.x), dimsimage.y);
			tfloat* d_tempRe, *d_tempIm;
			cudaMalloc((void**)&d_tempRe, ElementsFFT(dimsimage) * nimages * sizeof(tfloat));
			cudaMalloc((void**)&d_tempIm, ElementsFFT(dimsimage) * nimages * sizeof(tfloat));
			d_ConvertTComplexToSplitComplex(d_imagesft, d_tempRe, d_tempIm, ElementsFFT(dimsimage) * nimages);

			cudaArray_t* h_a_projftRe = (cudaArray_t*)malloc(nimages * sizeof(cudaArray_t)), *h_a_projftIm = (cudaArray_t*)malloc(nimages * sizeof(cudaArray_t));
			cudaTex* h_t_projftRe = (cudaTex*)malloc(nimages * sizeof(cudaTex)), *h_t_projftIm = (cudaTex*)malloc(nimages * sizeof(cudaTex));
			d_BindTextureToArray(d_tempRe, h_a_projftRe, h_t_projftRe, dimsft, cudaFilterModeLinear, false, nimages);
			cudaFree(d_tempRe);
			d_BindTextureToArray(d_tempIm, h_a_projftIm, h_t_projftIm, dimsft, cudaFilterModeLinear, false, nimages);
			cudaTex* d_t_projftRe = (cudaTex*)CudaMallocFromHostArray(h_t_projftRe, nimages * sizeof(cudaTex));
			cudaTex* d_t_projftIm = (cudaTex*)CudaMallocFromHostArray(h_t_projftIm, nimages * sizeof(cudaTex));

			// Prefilter PSF and bind to textures
			cudaArray_t* h_a_projpsf = (cudaArray_t*)malloc(nimages * sizeof(cudaArray_t));
			cudaTex* h_t_projpsf = (cudaTex*)malloc(nimages * sizeof(cudaTex));
			d_BindTextureToArray(d_imagespsf, h_a_projpsf, h_t_projpsf, dimsft, cudaFilterModeLinear, false, nimages);
			cudaFree(d_tempIm);
			cudaTex* d_t_projpsf = (cudaTex*)CudaMallocFromHostArray(h_t_projpsf, nimages * sizeof(cudaTex));

			// Add to volume
			d_ReconstructFourierPreciseAdd(d_volumeft, d_samples, dims, d_t_projftRe, d_t_projftIm, d_t_projpsf, h_angles, h_shifts, nimages, outputzerocentered, finalize);

			// Tear down
			cudaFree(d_t_projftIm);
			cudaFree(d_t_projftRe);
			cudaFree(d_t_projpsf);
			for (int n = 0; n < nimages; n++)
			{
				cudaDestroyTextureObject(h_t_projftRe[n]);
				cudaDestroyTextureObject(h_t_projftIm[n]);
				cudaDestroyTextureObject(h_t_projpsf[n]);
				cudaFreeArray(h_a_projftRe[n]);
				cudaFreeArray(h_a_projftIm[n]);
				cudaFreeArray(h_a_projpsf[n]);
			}
			free(h_t_projftIm);
			free(h_t_projftRe);
			free(h_t_projpsf);
			free(h_a_projftIm);
			free(h_a_projftRe);
			free(h_a_projpsf);
		}
		else if (mode == T_INTERP_SINC)
		{
			d_ReconstructFourierSincAdd(d_volumeft, d_samples, dims, d_imagesft, d_imagespsf, h_angles, h_shifts, nimages, outputzerocentered, finalize);
		}

		cudaFree(d_imagesft);
	}

	void d_ReconstructFourierSincAdd(tcomplex* d_volumeft, tfloat* d_samples, int3 dims, tcomplex* d_imagesft, tfloat* d_imagespsf, tfloat3* h_angles, tfloat2* h_shifts, int nimages, bool outputzerocentered, bool finalize)
	{
		int3 dimsimage = toInt3(dims.x, dims.y, 1);
		tfloat2* h_normshifts = (tfloat2*)malloc(nimages * sizeof(tfloat2));
		glm::mat3* h_rotations = (glm::mat3*)malloc(nimages * sizeof(glm::mat3));
		for (int i = 0; i < nimages; i++)
		{
			h_rotations[i] = glm::transpose(Matrix3Euler(h_angles[i]));
			h_normshifts[i] = tfloat2(h_shifts[i].x / (tfloat)dims.x, h_shifts[i].y / (tfloat)dims.y);
		}
		glm::mat3* d_rotations = (glm::mat3*)CudaMallocFromHostArray(h_rotations, nimages * sizeof(glm::mat3));
		tfloat2* d_shifts = (tfloat2*)CudaMallocFromHostArray(h_normshifts, nimages * sizeof(tfloat2));
		free(h_rotations);
		free(h_normshifts);

		int TpB = 192;
		dim3 grid = dim3((ElementsFFT(dims) + TpB - 1) / TpB);
		if (outputzerocentered)
			if (finalize)
				ReconstructFourierPreciseSincKernel<true, true> << <grid, TpB >> > (d_volumeft, d_samples, dims.x, dims.x / 2 + 1, ElementsFFT2(dimsimage), d_imagesft, d_imagespsf, d_rotations, d_shifts, nimages);
			else
				ReconstructFourierPreciseSincKernel<true, false> << <grid, TpB >> > (d_volumeft, d_samples, dims.x, dims.x / 2 + 1, ElementsFFT2(dimsimage), d_imagesft, d_imagespsf, d_rotations, d_shifts, nimages);
		else
			if (finalize)
				ReconstructFourierPreciseSincKernel<false, true> << <grid, TpB >> > (d_volumeft, d_samples, dims.x, dims.x / 2 + 1, ElementsFFT2(dimsimage), d_imagesft, d_imagespsf, d_rotations, d_shifts, nimages);
			else
				ReconstructFourierPreciseSincKernel<false, false> << <grid, TpB >> > (d_volumeft, d_samples, dims.x, dims.x / 2 + 1, ElementsFFT2(dimsimage), d_imagesft, d_imagespsf, d_rotations, d_shifts, nimages);

		cudaFree(d_shifts);
		cudaFree(d_rotations);
	}

	void d_ReconstructFourierPreciseAdd(tcomplex* d_volumeft, tfloat* d_samples, int3 dims, cudaTex* t_imageftRe, cudaTex* t_imageftIm, cudaTex* t_imageweights, tfloat3* h_angles, tfloat2* h_shifts, int nimages, bool outputzerocentered, bool finalize)
	{
		tfloat2* h_normshifts = (tfloat2*)malloc(nimages * sizeof(tfloat2));
		glm::vec3* h_normals = (glm::vec3*)malloc(nimages * sizeof(glm::vec3));
		glm::mat3x2* h_global2local = (glm::mat3x2*)malloc(nimages * sizeof(glm::mat3x2));

		for (int i = 0; i < nimages; i++)
		{
			glm::mat3 tB = Matrix3Euler(h_angles[i]);
			h_normals[i] = glm::vec3(tB[2][0], tB[2][1], tB[2][2]);
			h_global2local[i] = glm::mat3x2(tB[0][0], tB[1][0], tB[0][1], tB[1][1], tB[0][2], tB[1][2]);	//Column-major layout in constructor
			h_normshifts[i] = tfloat2(h_shifts[i].x / (tfloat)dims.x, h_shifts[i].y / (tfloat)dims.y);
		}

		glm::vec3* d_normals = (glm::vec3*)CudaMallocFromHostArray(h_normals, nimages * sizeof(glm::vec3));
		glm::mat3x2* d_global2local = (glm::mat3x2*)CudaMallocFromHostArray(h_global2local, nimages * sizeof(glm::mat3x2));
		tfloat2* d_shifts = (tfloat2*)CudaMallocFromHostArray(h_normshifts, nimages * sizeof(tfloat2));

		int TpB = min(192, NextMultipleOf(ElementsFFT(dims), 32));
		dim3 grid = dim3((ElementsFFT(dims) + TpB - 1) / TpB);

		if (outputzerocentered)
			if (finalize)
				ReconstructFourierPreciseKernel<true, true> << <grid, TpB >> > (d_volumeft, d_samples, dims.x, dims.x / 2 + 1, t_imageftRe, t_imageftIm, t_imageweights, d_global2local, d_normals, d_shifts, nimages);
			else
				ReconstructFourierPreciseKernel<true, false> << <grid, TpB >> > (d_volumeft, d_samples, dims.x, dims.x / 2 + 1, t_imageftRe, t_imageftIm, t_imageweights, d_global2local, d_normals, d_shifts, nimages);
		else
			if (finalize)
				ReconstructFourierPreciseKernel<false, true> << <grid, TpB >> > (d_volumeft, d_samples, dims.x, dims.x / 2 + 1, t_imageftRe, t_imageftIm, t_imageweights, d_global2local, d_normals, d_shifts, nimages);
			else
				ReconstructFourierPreciseKernel<false, false> << <grid, TpB >> > (d_volumeft, d_samples, dims.x, dims.x / 2 + 1, t_imageftRe, t_imageftIm, t_imageweights, d_global2local, d_normals, d_shifts, nimages);
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void ReconstructFourierKernel(tcomplex* d_imageft, tfloat* d_imagepsf, tcomplex* d_volumeft, tfloat* d_volumepsf, uint dim, uint dimft, uint n, glm::mat2x3* d_rotations, tfloat2* d_shifts)
	{
		uint id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n)
			return;

		d_imageft += n * blockIdx.y;
		d_imagepsf += n * blockIdx.y;

		int x = id % dimft;
		x -= dim / 2;
		int y = id / dimft;
		y -= dim / 2;

		glm::vec3 pos = d_rotations[blockIdx.y] * glm::vec2(x, y);
		if (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z >= dim * dim / 4)
			return;

		tfloat2 delta = d_shifts[blockIdx.y];
		tfloat factor = (delta.x * pos.x + pos.y * pos.y) * PI2;

		bool flip = false;
		if (pos.x > 1e-6f)
		{
			pos = -pos;
			pos.x += 1.0f;
			flip = true;
		}

		pos += (float)(dim / 2);
		uint x0 = max(pos.x, 0.0f);
		uint y0 = max(pos.y, 0.0f);
		uint z0 = max(pos.z, 0.0f);

		if (x0 >= dim || y0 >= dim || z0 >= dim)
			return;

		uint x1 = x0 + 1;
		uint y1 = y0 + 1;
		uint z1 = z0 + 1;

		float xd = pos.x - floor(pos.x);
		float yd = pos.y - floor(pos.y);
		float zd = pos.z - floor(pos.z);

		float c0 = 1.0f - zd;
		float c1 = zd;

		float c00 = (1.0f - yd) * c0;
		float c10 = yd * c0;
		float c01 = (1.0f - yd) * c1;
		float c11 = yd * c1;

		float c000 = (1.0f - xd) * c00;
		float c100 = xd * c00;
		float c010 = (1.0f - xd) * c10;
		float c110 = xd * c10;
		float c001 = (1.0f - xd) * c01;
		float c101 = xd * c01;
		float c011 = (1.0f - xd) * c11;
		float c111 = xd * c11;

		tcomplex val = d_imageft[id];
		tfloat psf = d_imagepsf[id];
		if (flip)
			val.y = -val.y;
		val = cmul(val, make_cuComplex(cos(factor), sin(factor)));


		atomicAdd((tfloat*)(d_volumeft + (z0 * dim + y0) * dimft + x0), c000 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dim + y0) * dimft + x0)) + 1, c000 * val.y);
		atomicAdd((tfloat*)(d_volumepsf + (z0 * dim + y0) * dimft + x0), c000 * psf);

		atomicAdd((tfloat*)(d_volumeft + (z0 * dim + y1) * dimft + x0), c010 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dim + y1) * dimft + x0)) + 1, c010 * val.y);
		atomicAdd((tfloat*)(d_volumepsf + (z0 * dim + y1) * dimft + x0), c010 * psf);

		atomicAdd((tfloat*)(d_volumeft + (z1 * dim + y0) * dimft + x0), c001 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dim + y0) * dimft + x0)) + 1, c001 * val.y);
		atomicAdd((tfloat*)(d_volumepsf + (z1 * dim + y0) * dimft + x0), c001 * psf);

		atomicAdd((tfloat*)(d_volumeft + (z1 * dim + y1) * dimft + x0), c011 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dim + y1) * dimft + x0)) + 1, c011 * val.y);
		atomicAdd((tfloat*)(d_volumepsf + (z1 * dim + y1) * dimft + x0), c011 * psf);

		if (x1 > dim)
			return;

		if (x1 > dim / 2)
		{
			y0 = dim - y0;
			z0 = dim - z0;
			x1 = dim - x1;
			y1 = dim - y1;
			z1 = dim - z1;
			val.y = -val.y;
		}

		atomicAdd((tfloat*)(d_volumeft + (z0 * dim + y0) * dimft + x1), c100 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dim + y0) * dimft + x1)) + 1, c100 * val.y);
		atomicAdd((tfloat*)(d_volumepsf + (z0 * dim + y0) * dimft + x1), c100 * psf);

		atomicAdd((tfloat*)(d_volumeft + (z0 * dim + y1) * dimft + x1), c110 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dim + y1) * dimft + x1)) + 1, c110 * val.y);
		atomicAdd((tfloat*)(d_volumepsf + (z0 * dim + y1) * dimft + x1), c110 * psf);


		atomicAdd((tfloat*)(d_volumeft + (z1 * dim + y0) * dimft + x1), c101 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dim + y0) * dimft + x1)) + 1, c101 * val.y);
		atomicAdd((tfloat*)(d_volumepsf + (z1 * dim + y0) * dimft + x1), c101 * psf);

		atomicAdd((tfloat*)(d_volumeft + (z1 * dim + y1) * dimft + x1), c111 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dim + y1) * dimft + x1)) + 1, c111 * val.y);
		atomicAdd((tfloat*)(d_volumepsf + (z1 * dim + y1) * dimft + x1), c111 * psf);
	}

	template<bool zerocentered, bool finalize> __global__ void ReconstructFourierPreciseKernel(tcomplex* d_volumeft, tfloat* d_samples, int dim, uint dimft, cudaTex* t_imageftRe, cudaTex* t_imageftIm, cudaTex* t_imageweights, glm::mat3x2* d_global2local, glm::vec3* d_normals, tfloat2* d_shifts, uint nimages)
	{
		uint id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= dimft * dim * dim)
			return;
		int idx = id % dimft;
		int idy = (id / dimft) % dim;
		int idz = id / (dimft * dim);

		int x, y, z;
		if (zerocentered)
		{
			x = idx;
			y = idy;
			z = idz;
		}
		else
		{
			x = dim / 2 - idx;
			y = dim - 1 - ((idy + dim / 2 - 1) % dim);
			z = dim - 1 - ((idz + dim / 2 - 1) % dim);
		}

		tcomplex sumft = make_cuComplex(0, 0);
		tfloat sumweight = 0;

		glm::vec3 posglobal = glm::vec3(x, y, z);
		posglobal -= (float)(dim / 2);
		//float falloff = min(max(0.0f, sqrt(posglobal.x * posglobal.x + posglobal.y * posglobal.y + posglobal.z * posglobal.z) - (float)(dim / 2 - 4)), 4.0f);
		//falloff = cos(falloff * 0.25f * PI) * 0.5f + 0.5f;
		float falloff = posglobal.x * posglobal.x + posglobal.y * posglobal.y + posglobal.z * posglobal.z >= (float)(dim * dim / 4) ? 0.0f : 1.0f;
		if (falloff < 1e-6f)
			return;

		for (uint n = 0; n < nimages; n++)
		{
			float distweight = max(0.0f, 1.0f - abs(dotp(d_normals[n], posglobal))) * falloff;
			if (distweight == 0.0f)
				continue;

			glm::vec2 poslocal = d_global2local[n] * posglobal;
			tfloat2 delta = d_shifts[n];
			tfloat factor = (delta.x * poslocal.x + delta.y * poslocal.y) * PI2;

			bool flip = false;
			if (poslocal.x > 0)
			{
				poslocal = -poslocal;
				flip = true;
			}

			poslocal += (float)(dimft - 1) + 0.5f;
			tfloat valRe = 0, valIm = 0, valweight = 0;
			valRe = tex2D<tfloat>(t_imageftRe[n], poslocal.x, poslocal.y);
			valIm = tex2D<tfloat>(t_imageftIm[n], poslocal.x, poslocal.y);
			valweight = tex2D<tfloat>(t_imageweights[n], poslocal.x, poslocal.y);

			if (flip)
				valIm = -valIm;

			sumweight += valweight * distweight;
			sumft += cmul(make_cuComplex(valRe, valIm), make_cuComplex(cos(factor), sin(factor))) * distweight;
		}

		sumft += d_volumeft[id];
		if (!finalize)
			d_volumeft[id] = sumft;
		sumweight += d_samples[id];
		if (!finalize)
			d_samples[id] = sumweight;
		else
			d_samples[id] = min((tfloat)1, sumweight);

		if (finalize)
		{
			sumweight = max(abs(sumweight), 1.0);
			sumft /= sumweight;
			d_volumeft[id] = sumft;
		}
	}

	template<bool zerocentered, bool finalize> __global__ void ReconstructFourierPreciseSincKernel(tcomplex* d_volumeft, tfloat* d_samples, uint dim, uint dimft, uint elementsimage, tcomplex* d_imagesft, tfloat* d_imageweights, glm::mat3* d_rotations, tfloat2* d_shifts, uint nimages)
	{
		__shared__ tfloat s_imagesftRe[192], s_imagesftIm[192];
		__shared__ tfloat s_imageweights[192];

		uint id = blockIdx.x * blockDim.x + threadIdx.x;

		uint x, y, z;
		if (zerocentered)
		{
			x = id % dimft;
			y = (id / dimft) % dim;
			z = id / (dimft * dim);
		}
		else
		{
			x = dim / 2 - id % dimft;
			y = dim - 1 - (((id / dimft) % dim + dim / 2 - 1) % dim);
			z = dim - 1 - ((id / (dimft * dim) + dim / 2 - 1) % dim);
		}

		tcomplex sumft = make_cuComplex(0, 0);
		tfloat sumweight = 0;

		float center = dim / 2;

		glm::vec3 pos = glm::vec3(x, y, z);
		pos -= center;

		for (uint n = 0; n < nimages; n++)
		{
			tfloat2 delta = d_shifts[n];
			glm::vec3 posglobal = d_rotations[n] * pos;
			float sincz = sinc(posglobal.z);

			for (uint poffset = 0; poffset < elementsimage; poffset += 192)
			{
				{
					uint pglobal = poffset + threadIdx.x;
					if (pglobal < elementsimage)
					{
						uint px = pglobal % dimft;
						uint py = pglobal / dimft;
						tfloat factor = (delta.x * ((float)px - center) + delta.y * ((float)py - center)) * PI2;
						tcomplex val = cmul(d_imagesft[pglobal], make_cuComplex(cos(factor), sin(factor)));

						s_imagesftRe[threadIdx.x] = val.x;
						s_imagesftIm[threadIdx.x] = val.y;
						s_imageweights[threadIdx.x] = d_imageweights[pglobal];
					}
				}

				__syncthreads();

				uint plimit = min(192, elementsimage - poffset);
				for (uint p = 0; p < plimit; p++)
				{
					uint pglobal = poffset + p;
					uint px = pglobal % dimft;
					uint py = pglobal / dimft;

					glm::vec2 pospixel = glm::vec2(px, py) - center;

					if (pospixel.x * pospixel.x + pospixel.y * pospixel.y < dim * dim * 4)
					{
						float s = sinc(pospixel.x - posglobal.x) * sinc(pospixel.y - posglobal.y) * sincz;

						tcomplex val = make_cuComplex(s_imagesftRe[p], s_imagesftIm[p]);
						tfloat valweight = s_imageweights[p];

						sumft += val * s;
						sumweight += valweight * s;

						if (px == dim / 2)
							continue;

						s = sinc(-pospixel.x - posglobal.x) * sinc(-pospixel.y - posglobal.y) * sincz;

						val.y = -val.y;

						sumft += val * s;
						sumweight += valweight * s;
					}
				}

				__syncthreads();
			}

			d_imagesft += elementsimage;
			d_imageweights += elementsimage;
		}

		if (id >= dimft * dim * dim)
			return;

		float falloff = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z >= (float)(dim * dim / 4) ? 0.0f : 1.0f;
		sumft *= falloff;
		sumweight *= falloff;

		sumft += d_volumeft[id];
		if (!finalize)
			d_volumeft[id] = sumft;
		sumweight += d_samples[id];
		if (!finalize)
			d_samples[id] = sumweight;
		else
			d_samples[id] = min((tfloat)1, sumweight);

		if (finalize)
		{
			sumweight = max(abs(sumweight), 1.0);
			sumft /= sumweight;
			d_volumeft[id] = sumft;
		}
	}
}