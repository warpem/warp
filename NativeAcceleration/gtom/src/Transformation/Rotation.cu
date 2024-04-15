#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate3DKernel(cudaTex t_input, tfloat* d_output, int3 dims, glm::mat4* d_transforms, uint nangles);
    template<bool cubicinterp> __global__ void Rotate3DExtractAt(cudaTex t_input, int3 dimvolume, tfloat* d_proj, int3 dimproj, size_t elementsproj, glm::mat3* d_rotations, tfloat3* d_positions);
	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate2DKernel(cudaTex* t_input, tfloat* d_output, int2 dims, glm::mat2* d_transforms);
	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate2DFTKernel(cudaTex t_Re, cudaTex t_Im, tcomplex* d_output, int3 dims, glm::mat2 transform, tfloat maxfreq);
	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate3DFTKernel(cudaTex t_Re, cudaTex t_Im, tcomplex* d_output, int3 dims, glm::mat4* d_transform, float maxfreq2);
	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate3DFTKernel(cudaTex t_Re, tfloat* d_output, int3 dims, glm::mat4* d_transform, float maxfreq2);


	////////////////////
	//Rotate 3D volume//
	////////////////////

	void d_Rotate3D(tfloat* d_volume, tfloat* d_output, int3 dims, tfloat3* h_angles, uint nangles, T_INTERP_MODE mode, bool iszerocentered)
	{
		tfloat* d_temp;
		if (mode == T_INTERP_CUBIC)
			cudaMalloc((void**)&d_temp, Elements(dims) * sizeof(tfloat));

		cudaArray* a_input;
		cudaTex t_input;
		if (mode == T_INTERP_LINEAR)
			d_BindTextureTo3DArray(d_volume, a_input, t_input, dims, cudaFilterModeLinear, false);
		else
		{
			cudaMemcpy(d_temp, d_volume, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			d_CubicBSplinePrefilter3D(d_temp, dims);
			d_BindTextureTo3DArray(d_temp, a_input, t_input, dims, cudaFilterModeLinear, false);
		}

		d_Rotate3D(t_input, d_output, dims, h_angles, nangles, mode, iszerocentered);

		cudaDestroyTextureObject(t_input);
		cudaFreeArray(a_input);

		if (mode == T_INTERP_CUBIC)
			cudaFree(d_temp);
	}

	void d_Rotate3D(cudaTex t_volume, tfloat* d_output, int3 dims, tfloat3* h_angles, uint nangles, T_INTERP_MODE mode, bool iszerocentered)
	{
		glm::mat4* h_transforms = (glm::mat4*)malloc(nangles * sizeof(glm::mat4));
		for (uint n = 0; n < nangles; n++)
		h_transforms[n] = Matrix4Translation(tfloat3(dims.x / 2 + 0.5f, dims.y / 2 + 0.5f, dims.z / 2 + 0.5f)) *
							glm::transpose(Matrix4Euler(h_angles[n])) *
							Matrix4Translation(tfloat3(-dims.x / 2, -dims.y / 2, -dims.z / 2));

		glm::mat4* d_transforms = (glm::mat4*)CudaMallocFromHostArray(h_transforms, nangles * sizeof(glm::mat4));
		free(h_transforms);

		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dims.x + 15) / 16, (dims.y + 15) / 16, dims.z);

		if (iszerocentered)
		{
			if (mode == T_INTERP_LINEAR)
				Rotate3DKernel<false, true> << <grid, TpB >> > (t_volume, d_output, dims, d_transforms, nangles);
			else if (mode == T_INTERP_CUBIC)
				Rotate3DKernel<true, true> << <grid, TpB >> > (t_volume, d_output, dims, d_transforms, nangles);
		}
		else
		{
			if (mode == T_INTERP_LINEAR)
				Rotate3DKernel<false, false> << <grid, TpB >> > (t_volume, d_output, dims, d_transforms, nangles);
			else if (mode == T_INTERP_CUBIC)
				Rotate3DKernel<true, false> << <grid, TpB >> > (t_volume, d_output, dims, d_transforms, nangles);
		}

		cudaFree(d_transforms);
	}


    ////////////////////////////
    //Rotate 3D and extract at//
    ////////////////////////////

    void d_Rotate3DExtractAt(cudaTex t_volume, int3 dimsvolume, tfloat* d_proj, int3 dimsproj, tfloat3* h_angles, tfloat3* h_positions, T_INTERP_MODE mode, uint batch)
    {
        glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
        for (int i = 0; i < batch; i++)
            h_matrices[i] = glm::transpose(Matrix3Euler(h_angles[i]));
        glm::mat3* d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
        free(h_matrices);

        tfloat3* d_positions = (tfloat3*)CudaMallocFromHostArray(h_positions, batch * sizeof(tfloat3));

        d_Rotate3DExtractAt(t_volume, dimsvolume, d_proj, dimsproj, d_matrices, d_positions, mode, batch);

        cudaFree(d_matrices);
        cudaFree(d_positions);
    }

    void d_Rotate3DExtractAt(cudaTex t_volume, int3 dimsvolume, tfloat* d_proj, int3 dimsproj, glm::mat3* d_matrices, tfloat3* d_positions, T_INTERP_MODE mode, uint batch)
    {
        uint ndimsvolume = DimensionCount(dimsvolume);
        uint ndimsproj = DimensionCount(dimsproj);
        if (ndimsvolume < ndimsproj)
            throw;

        uint elements = Elements(dimsproj);
        dim3 grid = dim3(tmin(128, (elements + 127) / 128), batch, 1);

        if (ndimsproj >= 2)
        {
            if (mode == T_INTERP_CUBIC)
                Rotate3DExtractAt<true> << <grid, 128 >> > (t_volume, dimsvolume, d_proj, dimsproj, elements, d_matrices, d_positions);
            else
                Rotate3DExtractAt<false> << <grid, 128 >> > (t_volume, dimsvolume, d_proj, dimsproj, elements, d_matrices, d_positions);
        }
        else
            throw;
    }


	/////////////
	//Rotate 2D//
	/////////////
	
	void d_Rotate2D(tfloat* d_input, tfloat* d_output, int2 dims, tfloat* h_angles, T_INTERP_MODE mode, bool isoutputzerocentered, uint batch)
	{
		tfloat* d_temp;
		if (mode == T_INTERP_CUBIC)
			cudaMalloc((void**)&d_temp, Elements2(dims) * batch * sizeof(tfloat));

		cudaArray_t* a_input = (cudaArray_t*)malloc(batch * sizeof(cudaArray_t));
		cudaTex* t_input = (cudaTex*)malloc(batch * sizeof(cudaTex));
		if (mode == T_INTERP_LINEAR)
			d_BindTextureToArray(d_input, a_input, t_input, dims, cudaFilterModeLinear, false, batch);
		else
		{
			cudaMemcpy(d_temp, d_input, Elements2(dims) * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			d_CubicBSplinePrefilter2D(d_temp, dims, batch);
			d_BindTextureToArray(d_temp, a_input, t_input, dims, cudaFilterModeLinear, false, batch);
		}
		cudaTex* dt_input = (cudaTex*)CudaMallocFromHostArray(t_input, batch * sizeof(cudaTex));

		d_Rotate2D(dt_input, d_output, dims, h_angles, mode, isoutputzerocentered, batch);

		cudaFree(dt_input);
		for (uint b = 0; b < batch; b++)
		{
			cudaDestroyTextureObject(t_input[b]);
			cudaFreeArray(a_input[b]);
		}
		free(t_input);
		free(a_input);
		if (mode == T_INTERP_CUBIC)
			cudaFree(d_temp);
	}

	void d_Rotate2D(cudaTex* t_input, tfloat* d_output, int2 dims, tfloat* h_angles, T_INTERP_MODE mode, bool isoutputzerocentered, uint batch)
	{
		glm::mat2* h_transforms = (glm::mat2*)malloc(batch * sizeof(glm::mat2));
		for (uint b = 0; b < batch; b++)
			h_transforms[b] = Matrix2Rotation(-h_angles[b]);
		glm::mat2* d_transforms = (glm::mat2*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat2));
		free(h_transforms);

		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dims.x + 15) / 16, (dims.y + 15) / 16, batch);

		if (isoutputzerocentered)
		{
			if (mode == T_INTERP_LINEAR)
				Rotate2DKernel<false, true> << <grid, TpB >> > (t_input, d_output, dims, d_transforms);
			else if (mode == T_INTERP_CUBIC)
				Rotate2DKernel<true, true> << <grid, TpB >> > (t_input, d_output, dims, d_transforms);
		}
		else
		{
			if (mode == T_INTERP_LINEAR)
				Rotate2DKernel<false, false> << <grid, TpB >> > (t_input, d_output, dims, d_transforms);
			else if (mode == T_INTERP_CUBIC)
				Rotate2DKernel<true, false> << <grid, TpB >> > (t_input, d_output, dims, d_transforms);
		}

		cudaFree(d_transforms);
	}


	//////////////////////////////
	//Rotate 2D in Fourier space//
	//////////////////////////////

	void d_Rotate2DFT(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat* angles, tfloat maxfreq, T_INTERP_MODE mode, bool isoutputzerocentered, int batch)
	{
		tfloat* d_real;
		cudaMalloc((void**)&d_real, ElementsFFT(dims) * sizeof(tfloat));
		tfloat* d_imag;
		cudaMalloc((void**)&d_imag, ElementsFFT(dims) * sizeof(tfloat));

		for (int b = 0; b < batch; b++)
		{
			d_ConvertTComplexToSplitComplex(d_input + ElementsFFT(dims) * b, d_real, d_imag, ElementsFFT(dims));

			if (mode == T_INTERP_CUBIC)
			{
				d_CubicBSplinePrefilter2D(d_real, toInt2(dims.x / 2 + 1, dims.y));
				d_CubicBSplinePrefilter2D(d_imag, toInt2(dims.x / 2 + 1, dims.y));
			}

			cudaArray* a_Re;
			cudaArray* a_Im;
			cudaTex t_Re, t_Im;
			d_BindTextureToArray(d_real, a_Re, t_Re, toInt2(dims.x / 2 + 1, dims.y), cudaFilterModeLinear, false);
			d_BindTextureToArray(d_imag, a_Im, t_Im, toInt2(dims.x / 2 + 1, dims.y), cudaFilterModeLinear, false);

			d_Rotate2DFT(t_Re, t_Im, d_output + ElementsFFT(dims) * b, dims, angles[b], maxfreq, mode, isoutputzerocentered);

			cudaDestroyTextureObject(t_Re);
			cudaDestroyTextureObject(t_Im);
			cudaFreeArray(a_Re);
			cudaFreeArray(a_Im);
		}

		cudaFree(d_imag);
		cudaFree(d_real);
	}

	void d_Rotate2DFT(cudaTex t_inputRe, cudaTex t_inputIm, tcomplex* d_output, int3 dims, tfloat angle, tfloat maxfreq, T_INTERP_MODE mode, bool isoutputzerocentered)
	{
		glm::mat2 rotation = Matrix2Rotation(-angle);

		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dims.x / 2 + 1 + 15) / 16, (dims.y + 15) / 16);

		if (isoutputzerocentered)
		{
			if (mode == T_INTERP_LINEAR)
				Rotate2DFTKernel<false, true> << <grid, TpB >> > (t_inputRe, t_inputIm, d_output, dims, rotation, maxfreq);
			else if (mode == T_INTERP_CUBIC)
				Rotate2DFTKernel<true, true> << <grid, TpB >> > (t_inputRe, t_inputIm, d_output, dims, rotation, maxfreq);
		}
		else
		{
			if (mode == T_INTERP_LINEAR)
				Rotate2DFTKernel<false, false> << <grid, TpB >> > (t_inputRe, t_inputIm, d_output, dims, rotation, maxfreq);
			else if (mode == T_INTERP_CUBIC)
				Rotate2DFTKernel<true, false> << <grid, TpB >> > (t_inputRe, t_inputIm, d_output, dims, rotation, maxfreq);
		}
	}
	

	//////////////////////////////
	//Rotate 3D in Fourier space//
	//////////////////////////////

	void d_Rotate3DFT(tcomplex* d_volume, tcomplex* d_output, int3 dims, tfloat3* h_angles, int nangles, T_INTERP_MODE mode, bool outputzerocentered)
	{
		int3 dimsfft = toInt3(dims.x / 2 + 1, dims.y, dims.z);
		tfloat* d_tempRe;
		cudaMalloc((void**)&d_tempRe, ElementsFFT(dims) * sizeof(tfloat));
		tfloat* d_tempIm;
		cudaMalloc((void**)&d_tempIm, ElementsFFT(dims) * sizeof(tfloat));

		cudaArray* a_Re, *a_Im;
		cudaTex t_Re, t_Im;

		d_ConvertTComplexToSplitComplex(d_volume, d_tempRe, d_tempIm, ElementsFFT(dims));
		if (mode == T_INTERP_CUBIC)
		{
			d_CubicBSplinePrefilter3D(d_tempRe, dimsfft);
			d_CubicBSplinePrefilter3D(d_tempIm, dimsfft);
		}
		d_BindTextureTo3DArray(d_tempRe, a_Re, t_Re, dimsfft, cudaFilterModeLinear, false);
		d_BindTextureTo3DArray(d_tempIm, a_Im, t_Im, dimsfft, cudaFilterModeLinear, false);
		cudaFree(d_tempRe);
		cudaFree(d_tempIm);

		d_Rotate3DFT(t_Re, t_Im, d_output, dims, h_angles, nangles, mode, outputzerocentered);

		cudaDestroyTextureObject(t_Re);
		cudaDestroyTextureObject(t_Im);
		cudaFreeArray(a_Re);
		cudaFreeArray(a_Im);
	}

	void d_Rotate3DFT(cudaTex t_Re, cudaTex t_Im, tcomplex* d_output, int3 dims, tfloat3* h_angles, int nangles, T_INTERP_MODE mode, bool outputzerocentered)
	{
		glm::mat4* h_transform = (glm::mat4*)malloc(nangles * sizeof(glm::mat4));
		for (int b = 0; b < nangles; b++)
			h_transform[b] = glm::transpose(Matrix4Euler(h_angles[b])) *
							 Matrix4Translation(tfloat3(-dims.x / 2, -dims.y / 2, -dims.z / 2));
		glm::mat4* d_transform = (glm::mat4*)CudaMallocFromHostArray(h_transform, nangles * sizeof(glm::mat4));

		float maxfreq2 = (float)(dims.x * dims.x / 4);

		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dims.x / 2 + 1 + 15) / 16, (dims.y + 15) / 16, dims.z * nangles);
		if (outputzerocentered)
		{
			if (mode == T_INTERP_LINEAR)
				Rotate3DFTKernel<false, true> << <grid, TpB >> > (t_Re, t_Im, d_output, dims, d_transform, maxfreq2);
			if (mode == T_INTERP_CUBIC)
				Rotate3DFTKernel<true, true> << <grid, TpB >> > (t_Re, t_Im, d_output, dims, d_transform, maxfreq2);
		}
		else
		{
			if (mode == T_INTERP_LINEAR)
				Rotate3DFTKernel<false, false> << <grid, TpB >> > (t_Re, t_Im, d_output, dims, d_transform, maxfreq2);
			if (mode == T_INTERP_CUBIC)
				Rotate3DFTKernel<true, false> << <grid, TpB >> > (t_Re, t_Im, d_output, dims, d_transform, maxfreq2);
		}

		cudaFree(d_transform);
		free(h_transform);
	}

	void d_Rotate3DFT(tfloat* d_volume, tfloat* d_output, int3 dims, tfloat3* h_angles, int nangles, T_INTERP_MODE mode, bool outputzerocentered)
	{
		int3 dimsfft = toInt3(dims.x / 2 + 1, dims.y, dims.z);
		tfloat* d_tempRe;
		cudaMalloc((void**)&d_tempRe, ElementsFFT(dims) * sizeof(tfloat));

		cudaArray* a_Re;
		cudaTex t_Re;

		cudaMemcpy(d_tempRe, d_volume, ElementsFFT(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		if (mode == T_INTERP_CUBIC)
			d_CubicBSplinePrefilter3D(d_tempRe, dimsfft);
		d_BindTextureTo3DArray(d_tempRe, a_Re, t_Re, dimsfft, cudaFilterModeLinear, false);
		cudaFree(d_tempRe);

		d_Rotate3DFT(t_Re, d_output, dims, h_angles, nangles, mode, outputzerocentered);

		cudaDestroyTextureObject(t_Re);
		cudaFreeArray(a_Re);
	}

	void d_Rotate3DFT(cudaTex t_volume, tfloat* d_output, int3 dims, tfloat3* h_angles, int nangles, T_INTERP_MODE mode, bool outputzerocentered)
	{
		glm::mat4* h_transform = (glm::mat4*)malloc(nangles * sizeof(glm::mat4));
		for (int b = 0; b < nangles; b++)
			h_transform[b] = glm::transpose(Matrix4Euler(h_angles[b])) *
							 Matrix4Translation(tfloat3(-dims.x / 2, -dims.y / 2, -dims.z / 2));
		glm::mat4* d_transform = (glm::mat4*)CudaMallocFromHostArray(h_transform, nangles * sizeof(glm::mat4));

		float maxfreq2 = (float)(dims.x * dims.x / 4);

		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dims.x / 2 + 1 + 15) / 16, (dims.y + 15) / 16, dims.z * nangles);
		if (outputzerocentered)
		{
			if (mode == T_INTERP_LINEAR)
				Rotate3DFTKernel<false, true> << <grid, TpB >> > (t_volume, d_output, dims, d_transform, maxfreq2);
			if (mode == T_INTERP_CUBIC)
				Rotate3DFTKernel<true, true> << <grid, TpB >> > (t_volume, d_output, dims, d_transform, maxfreq2);
		}
		else
		{
			if (mode == T_INTERP_LINEAR)
				Rotate3DFTKernel<false, false> << <grid, TpB >> > (t_volume, d_output, dims, d_transform, maxfreq2);
			if (mode == T_INTERP_CUBIC)
				Rotate3DFTKernel<true, false> << <grid, TpB >> > (t_volume, d_output, dims, d_transform, maxfreq2);
		}

		cudaFree(d_transform);
		free(h_transform);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate3DKernel(cudaTex t_input, tfloat* d_output, int3 dims, glm::mat4* d_transforms, uint nangles)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y)
			return;
		int idz = blockIdx.z;

		int x, y, z;
		if (outputzerocentered)
		{
			x = idx;
			y = idy;
			z = idz;
		}
		else
		{
			x = FFTShift(idx, dims.x);
			y = FFTShift(idy, dims.y);
			z = FFTShift(idz, dims.z);
		}

		for (uint b = 0; b < nangles; b++)
		{
			glm::vec4 pos = d_transforms[b] * glm::vec4(x, y, z, 1);	// No need to center pos, done by transform
			tfloat value;

			if (cubicinterp)
				value = cubicTex3DSimple<tfloat>(t_input, pos.x, pos.y, pos.z);
			else
				value = tex3D<tfloat>(t_input, pos.x, pos.y, pos.z);

			d_output[(b * dims.z + (idz * dims.y + idy)) * dims.x + idx] = value;
		}
	}

    template<bool cubicinterp> __global__ void Rotate3DExtractAt(cudaTex t_input, int3 dimvolume, tfloat* d_proj, int3 dimproj, size_t elementsproj, glm::mat3* d_rotations, tfloat3* d_positions)
    {
        d_proj += elementsproj * blockIdx.y;

        uint line = dimproj.x;
        uint slice = Elements2(dimproj);

        glm::mat3 rotation = d_rotations[blockIdx.y];
        glm::vec3 position = glm::vec3(d_positions[blockIdx.y].x, d_positions[blockIdx.y].y, d_positions[blockIdx.y].z);
        int3 centervolume = dimproj / 2;

        for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elementsproj; id += gridDim.x * blockDim.x)
        {
            uint idx = id % line;
            uint idy = (id % slice) / line;
            uint idz = id / slice;

            int x = idx;
            int y = idy;
            int z = idz;

            glm::vec3 pos = glm::vec3(x - centervolume.x, y - centervolume.y, z - centervolume.z);

            pos = rotation * pos;
            pos += position;

			if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0 &&
				pos.x <= dimvolume.x - 1 && pos.y <= dimvolume.y - 1 && pos.z <= dimvolume.z - 1)
			{
				pos += 0.5f;

				if (cubicinterp)
					d_proj[id] = cubicTex3DSimple<tfloat>(t_input, pos.x, pos.y, pos.z);
				else
					d_proj[id] = tex3D<tfloat>(t_input, pos.x, pos.y, pos.z);
			}
			else
			{
				glm::vec3 posclamped = glm::vec3(tmax(tmin(dimvolume.x - 1, pos.x), 0),
												 tmax(tmin(dimvolume.y - 1, pos.y), 0),
												 tmax(tmin(dimvolume.z - 1, pos.z), 0));
				float borderdist = glm::distance(pos, posclamped);
				
				if (borderdist > 16)
				{
					d_proj[id] = 0;
					continue;
				}

				int samples = 0;
				tfloat blursum = 0;

				for (int z = -3; z <= 3; z++)
				{
					for (int y = -3; y <= 3; y++)
					{
						for (int x = -3; x <= 3; x++)
						{
							glm::vec3 blurpos = posclamped + glm::vec3(x, y, z) / 3.0f * tmin(borderdist * 0.5f, 8);

							if (blurpos.x < -0.1f || blurpos.y < -0.1f || blurpos.z < -0.1f ||
								blurpos.x >= dimvolume.x - 0.9f || blurpos.y >= dimvolume.y - 0.9f || blurpos.z >= dimvolume.z - 0.9f)
								continue;

							samples++;
							blurpos += 0.5f;

							blursum += tex3D<tfloat>(t_input, blurpos.x, blurpos.y, blurpos.z);
						}
					}
				}

				float weight = cosf(borderdist / 16.0f * PI) * 0.5f + 0.5f;
				d_proj[id] = blursum / tmax(1, samples) * weight;
			}
        }
    }

	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate2DKernel(cudaTex* t_input, tfloat* d_output, int2 dims, glm::mat2* d_transforms)
	{
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;
		uint idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y)
			return;

		int x, y;
		if (outputzerocentered)
		{
			x = idx;
			y = idy;
		}
		else
		{
			x = FFTShift(idx, dims.x);
			y = FFTShift(idy, dims.y);
		}

		glm::vec2 pos = d_transforms[blockIdx.z] * glm::vec2(x - dims.x / 2, y - dims.y / 2) + glm::vec2(dims.x / 2 + 0.5f, dims.y / 2 + 0.5f);
		tfloat val;

		if (!cubicinterp)
			val = tex2D<tfloat>(t_input[blockIdx.z], pos.x, pos.y);
		else
			val = cubicTex2D(t_input[blockIdx.z], pos.x, pos.y);

		d_output[(blockIdx.z * dims.y + idy) * dims.x + idx] = val;
	}

	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate2DFTKernel(cudaTex t_Re, cudaTex t_Im, tcomplex* d_output, int3 dims, glm::mat2 transform, tfloat maxfreq)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx > dims.x / 2)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y)
			return;

		int x, y;
		if (outputzerocentered)
		{
			x = idx;
			y = idy;
		}
		else
		{
			x = dims.x / 2 - idx;
			y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);
		}

		glm::vec2 pos = transform * glm::vec2(idx - dims.x / 2, idy - dims.y / 2);

		if (glm::length(pos) > maxfreq)
		{
			d_output[y * (dims.x / 2 + 1) + x] = make_cuComplex(0.0f, 0.0f);
			return;
		}

		bool isnegative = false;
		if (pos.x > 0.00001f)
		{
			pos = -pos;
			isnegative = true;
		}

		pos += glm::vec2((float)(dims.x / 2) + 0.5f, (float)(dims.y / 2) + 0.5f);

		tfloat valre, valim;
		if (!cubicinterp)
		{
			valre = tex2D<tfloat>(t_Re, pos.x, pos.y);
			valim = tex2D<tfloat>(t_Im, pos.x, pos.y);
		}
		else
		{
			valre = cubicTex2D(t_Re, pos.x, pos.y);
			valim = cubicTex2D(t_Im, pos.x, pos.y);
		}

		if (isnegative)
			valim = -valim;

		d_output[y * (dims.x / 2 + 1) + x] = make_cuComplex(valre, valim);
	}

	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate3DFTKernel(cudaTex t_Re, cudaTex t_Im, tcomplex* d_output, int3 dims, glm::mat4* d_transform, float maxfreq2)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx > dims.x / 2)
			return;
		uint idglobal = blockIdx.z / dims.z;
		d_output += ElementsFFT(dims) * idglobal;
		d_transform += idglobal;

		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y)
			return;
		int idz = blockIdx.z % dims.z;

		int x, y, z;
		if (outputzerocentered)
		{
			x = idx;
			y = idy;
			z = idz;
		}
		else
		{
			x = dims.x / 2 - idx;
			y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);
			z = dims.z - 1 - ((idz + dims.z / 2 - 1) % dims.z);
		}

		glm::vec4 pos = *d_transform * glm::vec4(x, y, z, 1);

		float radiussq = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
		if (radiussq >= maxfreq2)
		{
			d_output[(idz * dims.y + idy) * (dims.x / 2 + 1) + idx] = make_cuComplex(0, 0);
			return;
		}

		bool isnegative = false;
		if (pos.x > 1e-6f)
		{
			pos = -pos;
			isnegative = true;
		}

		pos += (float)(dims.x / 2) + 0.5f;

		tfloat valre, valim;
		if (!cubicinterp)
		{
			valre = tex3D<tfloat>(t_Re, pos.x, pos.y, pos.z);
			valim = tex3D<tfloat>(t_Im, pos.x, pos.y, pos.z);
		}
		else
		{
			valre = cubicTex3D(t_Re, pos.x, pos.y, pos.z);
			valim = cubicTex3D(t_Im, pos.x, pos.y, pos.z);
		}

		if (isnegative)
			valim = -valim;

		d_output[(idz * dims.y + idy) * (dims.x / 2 + 1) + idx] = make_cuComplex(valre, valim);
	}

	template<bool cubicinterp, bool outputzerocentered> __global__ void Rotate3DFTKernel(cudaTex t_Re, tfloat* d_output, int3 dims, glm::mat4* d_transform, float maxfreq2)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx > dims.x / 2)
			return;
		uint idglobal = blockIdx.z / dims.z;
		d_output += ElementsFFT(dims) * idglobal;
		d_transform += idglobal;

		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y)
			return;
		int idz = blockIdx.z % dims.z;

		int x, y, z;
		if (outputzerocentered)
		{
			x = idx;
			y = idy;
			z = idz;
		}
		else
		{
			x = dims.x / 2 - idx;
			y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);
			z = dims.z - 1 - ((idz + dims.z / 2 - 1) % dims.z);
		}

		glm::vec4 pos = *d_transform * glm::vec4(x, y, z, 1);

		float radiussq = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
		if (radiussq >= maxfreq2)
		{
			d_output[(idz * dims.y + idy) * (dims.x / 2 + 1) + idx] = 0;
			return;
		}

		if (pos.x > 1e-6f)
			pos = -pos;

		pos += (float)(dims.x / 2) + 0.5f;

		tfloat valre;
		if (!cubicinterp)
			valre = tex3D<tfloat>(t_Re, pos.x, pos.y, pos.z);
		else
			valre = cubicTex3D(t_Re, pos.x, pos.y, pos.z);

		d_output[(idz * dims.y + idy) * (dims.x / 2 + 1) + idx] = valre;
	}
}