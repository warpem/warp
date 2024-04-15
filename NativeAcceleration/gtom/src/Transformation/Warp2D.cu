#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void Warp2DKernel(cudaTex* t_image, int2 dimsimage, cudaTex* t_gridx, cudaTex* t_gridy, int2 dimsgrid, tfloat* d_output);
	__global__ void Warp2DKernel(cudaTex t_image, int2 dimsimage, cudaTex t_gridx, cudaTex t_gridy, int2 dimsgrid, tfloat* d_output);


	////////////////////////////////////////
	//Equivalent of TOM's tom_shift method//
	////////////////////////////////////////

	void d_Warp2D(tfloat* d_image, int2 dimsimage, tfloat2* d_grid, int2 dimsgrid, tfloat* d_output, uint batch)
	{
		tfloat* d_gridx, *d_gridy;
		cudaMalloc((void**)&d_gridx, Elements2(dimsgrid) * batch * sizeof(tfloat));
		cudaMalloc((void**)&d_gridy, Elements2(dimsgrid) * batch * sizeof(tfloat));
		d_ConvertTComplexToSplitComplex((tcomplex*)d_grid, d_gridx, d_gridy, Elements2(dimsgrid) * batch);

		d_CubicBSplinePrefilter2D(d_gridx, dimsgrid, batch);
		d_CubicBSplinePrefilter2D(d_gridy, dimsgrid, batch);
		tfloat* d_temp;
		cudaMalloc((void**)&d_temp, Elements2(dimsimage) * batch * sizeof(tfloat));
		cudaMemcpy(d_temp, d_image, Elements2(dimsimage) * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		d_CubicBSplinePrefilter2D(d_temp, dimsimage, batch);

		cudaTex* t_image = (cudaTex*)malloc(batch * sizeof(cudaTex));
		cudaTex* t_gridx = (cudaTex*)malloc(batch * sizeof(cudaTex));
		cudaTex* t_gridy = (cudaTex*)malloc(batch * sizeof(cudaTex));
		cudaArray_t* a_image = (cudaArray_t*)malloc(batch * sizeof(cudaArray_t));
		cudaArray_t* a_gridx = (cudaArray_t*)malloc(batch * sizeof(cudaArray_t));
		cudaArray_t* a_gridy = (cudaArray_t*)malloc(batch * sizeof(cudaArray_t));

		d_BindTextureToArray(d_temp, a_image, t_image, dimsimage, cudaFilterModeLinear, false, batch);
		d_BindTextureToArray(d_gridx, a_gridx, t_gridx, dimsgrid, cudaFilterModeLinear, false, batch);
		d_BindTextureToArray(d_gridy, a_gridy, t_gridy, dimsgrid, cudaFilterModeLinear, false, batch);

		cudaTex* dt_image = (cudaTex*)CudaMallocFromHostArray(t_image, batch * sizeof(cudaTex));
		cudaTex* dt_gridx = (cudaTex*)CudaMallocFromHostArray(t_gridx, batch * sizeof(cudaTex));
		cudaTex* dt_gridy = (cudaTex*)CudaMallocFromHostArray(t_gridy, batch * sizeof(cudaTex));

		d_Warp2D(dt_image, dimsimage, dt_gridx, dt_gridy, dimsgrid, d_output, batch);

		cudaFree(dt_gridy);
		cudaFree(dt_gridx);
		cudaFree(dt_image);
		cudaFree(d_temp);

		for (uint b = 0; b < batch; b++)
		{
			cudaDestroyTextureObject(t_image[b]);
			cudaDestroyTextureObject(t_gridx[b]);
			cudaDestroyTextureObject(t_gridy[b]);

			cudaFreeArray(a_image[b]);
			cudaFreeArray(a_gridx[b]);
			cudaFreeArray(a_gridy[b]);
		}

		free(t_image);
		free(t_gridx);
		free(t_gridy);
		free(a_image);
		free(a_gridx);
		free(a_gridy);
	}

	void d_Warp2D(cudaTex* dt_image, int2 dimsimage, cudaTex* dt_gridx, cudaTex* dt_gridy, int2 dimsgrid, tfloat* d_output, uint batch)
	{
		dim3 TpB = dim3(16, 16, 1);
		dim3 grid = dim3((dimsimage.x + TpB.x - 1) / TpB.x, (dimsimage.y + TpB.y - 1) / TpB.y, batch);
		Warp2DKernel << <grid, TpB >> > (dt_image, dimsimage, dt_gridx, dt_gridy, dimsgrid, d_output);
	}

	void d_Warp2D(cudaTex dt_image, int2 dimsimage, cudaTex dt_gridx, cudaTex dt_gridy, int2 dimsgrid, tfloat* d_output)
	{
		dim3 TpB = dim3(16, 16, 1);
		dim3 grid = dim3((dimsimage.x + TpB.x - 1) / TpB.x, (dimsimage.y + TpB.y - 1) / TpB.y);
		Warp2DKernel << <grid, TpB >> > (dt_image, dimsimage, dt_gridx, dt_gridy, dimsgrid, d_output);
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void Warp2DKernel(cudaTex* t_image, int2 dimsimage, cudaTex* t_gridx, cudaTex* t_gridy, int2 dimsgrid, tfloat* d_output)
	{
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dimsimage.x)
			return;
		uint idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dimsimage.y)
			return;
		d_output += Elements2(dimsimage) * blockIdx.z;

		float posx = (float)idx / (float)(dimsimage.x - 1) * (float)(dimsgrid.x - 1) + 0.5f;
		float posy = (float)idy / (float)(dimsimage.y - 1) * (float)(dimsgrid.y - 1) + 0.5f;

		float gridx = cubicTex2D(t_gridx[blockIdx.z], posx, posy) + 0.5f;
		float gridy = cubicTex2D(t_gridy[blockIdx.z], posx, posy) + 0.5f;

		tfloat val = cubicTex2D(t_image[blockIdx.z], (float)idx + gridx, (float)idy + gridy);

		d_output[idy * dimsimage.x + idx] = val;
	}

	__global__ void Warp2DKernel(cudaTex t_image, int2 dimsimage, cudaTex t_gridx, cudaTex t_gridy, int2 dimsgrid, tfloat* d_output)
	{
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dimsimage.x)
			return;
		uint idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dimsimage.y)
			return;

		float posx = (float)idx / (float)(dimsimage.x - 1) * (float)(dimsgrid.x - 1) + 0.5f;
		float posy = (float)idy / (float)(dimsimage.y - 1) * (float)(dimsgrid.y - 1) + 0.5f;

		float gridx = cubicTex2D(t_gridx, posx, posy) + 0.5f;
		float gridy = cubicTex2D(t_gridy, posx, posy) + 0.5f;

		tfloat val = cubicTex2D(t_image, (float)idx + gridx, (float)idy + gridy);

		d_output[idy * dimsimage.x + idx] = val;
	}
}