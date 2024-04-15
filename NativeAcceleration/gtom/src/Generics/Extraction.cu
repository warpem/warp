#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/Helper.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void ExtractKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3 regionorigin);
	template <class T> __global__ void ExtractKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3* d_regionorigins);
	template <class T> __global__ void ExtractManyKernel(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, size_t regionelements, int3* d_regionorigins, bool zeropad);
	template <class T> __global__ void ExtractManyMultisourceKernel(T** d_inputs, T* d_output, int3 sourcedims, int3 regiondims, size_t regionelements, int3* d_regionorigins, uint nsources);
	template <bool cubicinterp> __global__ void Extract2DTransformedKernel(cudaTex t_input, tfloat* d_output, int2 sourcedims, int2 regiondims, glm::mat3* d_transforms);


	/////////////////////////////////////////////////////////////////////
	//Extract a portion of 1/2/3-dimensional data with cyclic boudaries//
	/////////////////////////////////////////////////////////////////////

	template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch)
	{
		int3 regionorigin;
		regionorigin.x = (regioncenter.x - (regiondims.x / 2) + sourcedims.x) % sourcedims.x;
		regionorigin.y = (regioncenter.y - (regiondims.y / 2) + sourcedims.y) % sourcedims.y;
		regionorigin.z = (regioncenter.z - (regiondims.z / 2) + sourcedims.z) % sourcedims.z;

		size_t TpB = min(256, NextMultipleOf(regiondims.x, 32));
		dim3 grid = dim3(regiondims.y, regiondims.z, batch);
		ExtractKernel << <grid, (int)TpB >> > (d_input, d_output, sourcedims, Elements(sourcedims), regiondims, Elements(regiondims), regionorigin);
	}
	template void d_Extract<half>(half* d_input, half* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);
	template void d_Extract<float>(float* d_input, float* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);
	template void d_Extract<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);
	template void d_Extract<double>(double* d_input, double* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);
	template void d_Extract<int>(int* d_input, int* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);
	template void d_Extract<char>(char* d_input, char* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);

	template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch)
	{
		size_t TpB = min(256, NextMultipleOf(regiondims.x, 32));
		dim3 grid = dim3(regiondims.y, regiondims.z, batch);
		ExtractKernel << <grid, (int)TpB >> > (d_input, d_output, sourcedims, Elements(sourcedims), regiondims, Elements(regiondims), d_regionorigins);
	}
	template void d_Extract<half>(half* d_input, half* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);
	template void d_Extract<float>(float* d_input, float* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);
	template void d_Extract<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);
	template void d_Extract<double>(double* d_input, double* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);
	template void d_Extract<int>(int* d_input, int* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);
	template void d_Extract<char>(char* d_input, char* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);

	template <class T> void d_ExtractMany(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, bool zeropad, int batch)
	{
		size_t TpB = min(256, NextMultipleOf(regiondims.x, 32));
		dim3 grid = dim3(regiondims.y, regiondims.z, batch);
		ExtractManyKernel << <grid, (int)TpB >> > (d_input, d_output, sourcedims, regiondims, Elements(regiondims), d_regionorigins, zeropad);
	}
	template void d_ExtractMany<half>(half* d_input, half* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, bool zeropad, int batch);
	template void d_ExtractMany<float>(float* d_input, float* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, bool zeropad, int batch);
	template void d_ExtractMany<double>(double* d_input, double* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, bool zeropad, int batch);
	template void d_ExtractMany<int>(int* d_input, int* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, bool zeropad, int batch);
	template void d_ExtractMany<char>(char* d_input, char* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, bool zeropad, int batch);

	template <class T> void d_ExtractManyMultisource(T** d_inputs, T* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int nsources, int batch)
	{
		size_t TpB = min(256, NextMultipleOf(regiondims.x, 32));
		dim3 grid = dim3(regiondims.y, regiondims.z, batch);
		ExtractManyMultisourceKernel << <grid, (int)TpB >> > (d_inputs, d_output, sourcedims, regiondims, Elements(regiondims), d_regionorigins, (uint)nsources);
	}
	template void d_ExtractManyMultisource<half>(half** d_inputs, half* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int nsources, int batch);
	template void d_ExtractManyMultisource<float>(float** d_inputs, float* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int nsources, int batch);
	template void d_ExtractManyMultisource<double>(double** d_inputs, double* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int nsources, int batch);
	template void d_ExtractManyMultisource<int>(int** d_inputs, int* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int nsources, int batch);
	template void d_ExtractManyMultisource<char>(char** d_inputs, char* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int nsources, int batch);


	/////////////////////////////////////////////////////////////////////////////////
	//Extract a portion of 2-dimensional data with translation and rotation applied//
	/////////////////////////////////////////////////////////////////////////////////

	void d_Extract2DTransformed(tfloat* d_input, tfloat* d_output, int2 dimsinput, int2 dimsregion, tfloat2* h_scale, tfloat* h_rotation, tfloat2* h_translation, T_INTERP_MODE mode, int batch)
	{
		cudaArray* a_input;
		cudaTex t_input;
		if (mode == T_INTERP_LINEAR)
			d_BindTextureToArray(d_input, a_input, t_input, dimsinput, cudaFilterModeLinear, false);
		else
		{
			tfloat* d_temp;
			cudaMalloc((void**)&d_temp, Elements2(dimsinput) * sizeof(tfloat));
			cudaMemcpy(d_temp, d_input, Elements2(dimsinput) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			d_CubicBSplinePrefilter2D(d_temp, dimsinput);
			d_BindTextureToArray(d_temp, a_input, t_input, dimsinput, cudaFilterModeLinear, false);
			cudaFree(d_temp);
		}

		glm::mat3* h_transforms = (glm::mat3*)malloc(batch * sizeof(glm::mat3));
		for (int b = 0; b < batch; b++)
			h_transforms[b] = Matrix3Translation(tfloat2(h_translation[b].x + dimsregion.x / 2 + 0.5f, h_translation[b].y + dimsregion.y / 2 + 0.5f))*
			Matrix3Scale(tfloat3(h_scale[b].x, h_scale[b].y, 1.0f)) *
			glm::transpose(Matrix3RotationZ(h_rotation[b])) *
			Matrix3Translation(tfloat2(-dimsregion.x / 2, -dimsregion.y / 2));
		glm::mat3* d_transforms = (glm::mat3*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat3));

		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dimsregion.x + 15) / 16, (dimsregion.y + 15) / 16, batch);

		if (mode == T_INTERP_LINEAR)
			Extract2DTransformedKernel<false> << <grid, TpB >> > (t_input, d_output, dimsinput, dimsregion, d_transforms);
		else if (mode == T_INTERP_CUBIC)
			Extract2DTransformedKernel<true> << <grid, TpB >> > (t_input, d_output, dimsinput, dimsregion, d_transforms);

		cudaFree(d_transforms);
		cudaDestroyTextureObject(t_input);
		cudaFreeArray(a_input);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void ExtractKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3 regionorigin)
	{
		int oy = (blockIdx.x + regionorigin.y) % sourcedims.y;
		int oz = (blockIdx.y + regionorigin.z) % sourcedims.z;

		T* offsetoutput = d_output + blockIdx.z * regionelements + (blockIdx.y * regiondims.y + blockIdx.x) * regiondims.x;
		T* offsetinput = d_input + blockIdx.z * sourceelements + (oz * sourcedims.y + oy) * sourcedims.x;

		for (int idx = threadIdx.x; idx < regiondims.x; idx += blockDim.x)
			offsetoutput[idx] = offsetinput[(idx + regionorigin.x) % sourcedims.x];
	}

	template <class T> __global__ void ExtractKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3* d_regionorigins)
	{
		int3 regionorigin = d_regionorigins[blockIdx.z];
		int idy = blockIdx.x;
		int idz = blockIdx.y;

		int oy = (idy + regionorigin.y + sourcedims.y * 999) % sourcedims.y;
		int oz = (idz + regionorigin.z + sourcedims.z * 999) % sourcedims.z;

		d_output += blockIdx.z * regionelements + (idz * regiondims.y + idy) * regiondims.x;
		d_input += blockIdx.z * sourceelements + (oz * sourcedims.y + oy) * sourcedims.x;

		for (int idx = threadIdx.x; idx < regiondims.x; idx += blockDim.x)
			d_output[idx] = d_input[(idx + regionorigin.x + sourcedims.x * 999) % sourcedims.x];
	}

	template <class T> __global__ void ExtractManyKernel(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, size_t regionelements, int3* d_regionorigins, bool zeropad)
	{
		int3 regionorigin = d_regionorigins[blockIdx.z];
		int idy = blockIdx.x;
		int idz = blockIdx.y;

		int oy = idy + regionorigin.y;
		int oz = idz + regionorigin.z;
		if (!zeropad)
		{
			oy = (oy + sourcedims.y * 999) % sourcedims.y;
			oz = (oz + sourcedims.z * 999) % sourcedims.z;
		}

		d_output += blockIdx.z * regionelements + (idz * regiondims.y + idy) * regiondims.x;

		if (zeropad)
		{
			if (oy < 0 || oy >= sourcedims.y || oz < 0 || oz >= sourcedims.z)
			{
				for (int idx = threadIdx.x; idx < regiondims.x; idx += blockDim.x)
					d_output[idx] = (T)0;

				return;
			}
		}

		d_input += (oz * sourcedims.y + oy) * sourcedims.x;

		if (!zeropad)
		{
			for (int idx = threadIdx.x; idx < regiondims.x; idx += blockDim.x)
				d_output[idx] = d_input[(idx + regionorigin.x + sourcedims.x * 999) % sourcedims.x];
		}
		else
		{
			for (int idx = threadIdx.x; idx < regiondims.x; idx += blockDim.x)
			{
				int ox = idx + regionorigin.x;
				if (ox < 0 || ox >= sourcedims.x)
					d_output[idx] = (T)0;
				else
					d_output[idx] = d_input[ox];
			}
		}
	}

	template <class T> __global__ void ExtractManyMultisourceKernel(T** d_inputs, T* d_output, int3 sourcedims, int3 regiondims, size_t regionelements, int3* d_regionorigins, uint nsources)
	{
		T* d_input = d_inputs[blockIdx.z % nsources];

		int3 regionorigin = d_regionorigins[blockIdx.z];
		int idy = blockIdx.x;
		int idz = blockIdx.y;

		int oy = tmax(0, tmin(idy + regionorigin.y, sourcedims.y - 1));
		int oz = tmax(0, tmin(idz + regionorigin.z, sourcedims.z - 1));

		d_output += blockIdx.z * regionelements + (idz * regiondims.y + idy) * regiondims.x;
		d_input += (oz * sourcedims.y + oy) * sourcedims.x;

		for (int idx = threadIdx.x; idx < regiondims.x; idx += blockDim.x)
			d_output[idx] = d_input[tmax(0, tmin(idx + regionorigin.x, sourcedims.x - 1))];
	}

	template <bool cubicinterp> __global__ void Extract2DTransformedKernel(cudaTex t_input, tfloat* d_output, int2 sourcedims, int2 regiondims, glm::mat3* d_transforms)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= regiondims.x)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= regiondims.y)
			return;
		int idz = blockIdx.z;

		d_output += (idz * regiondims.y + idy) * regiondims.x + idx;

		glm::vec3 pos = d_transforms[idz] * glm::vec3(idx, idy, 1.0f);

		if (pos.x < 0.0f || pos.x >(float)sourcedims.x || pos.y < 0.0f || pos.y >(float)sourcedims.y)
		{
			*d_output = (tfloat)0;
			return;
		}
		else
		{
			if (cubicinterp)
				*d_output = cubicTex2D(t_input, pos.x, pos.y);
			else
				*d_output = tex2D<tfloat>(t_input, pos.x, pos.y);
		}
	}
}