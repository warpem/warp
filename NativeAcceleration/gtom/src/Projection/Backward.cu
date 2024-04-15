#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/Helper.cuh"

namespace gtom
{
#define SincWindow 16

	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <bool iscentered, bool cubicinterp> __global__ void ProjBackwardKernel(tfloat* d_volume, uint3 dimsvolume, cudaTex* dt_image, uint2 dimsimage, glm::mat4* d_transforms, uint nimages);
	template <bool iscentered> __global__ void ProjBackwardSincKernel(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int2 dimsimage, glm::mat4 transform);


	/////////////////////////////////////////////
	//Equivalent of TOM's tom_backproj3d method//
	/////////////////////////////////////////////

	void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, bool outputzerocentered, int batch)
	{
		glm::mat4* h_transforms = (glm::mat4*)malloc(batch * sizeof(glm::mat4));
		for (int b = 0; b < batch; b++)
		{
			h_transforms[b] = Matrix4Translation(tfloat3((tfloat)dimsimage.x / 2.0f + 0.5f, (tfloat)dimsimage.y / 2.0f + 0.5f, 0.0f)) *
							  Matrix4Scale(tfloat3(1.0f / h_scales[b].x, 1.0f / h_scales[b].y, 1.0f)) *
							  Matrix4Translation(tfloat3(-h_offsets[b].x, -h_offsets[b].y, 0.0f)) *
							  (Matrix4Euler(h_angles[b])) *
							  Matrix4Translation(offsetfromcenter) *
							  Matrix4Translation(tfloat3(-dimsvolume.x / 2, -dimsvolume.y / 2, -dimsvolume.z / 2));
		}

		if (mode == T_INTERP_LINEAR || mode == T_INTERP_CUBIC)
		{
			cudaArray_t* ha_image = (cudaArray_t*)malloc(batch * sizeof(cudaArray_t));
			cudaTex* ht_image = (cudaTex*)malloc(batch * sizeof(cudaTex));

			if (mode == T_INTERP_CUBIC)
			{
				tfloat* d_temp;
				cudaMalloc((void**)&d_temp, Elements2(dimsimage) * batch * sizeof(tfloat));
				cudaMemcpy(d_temp, d_image, Elements2(dimsimage) * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);
				d_CubicBSplinePrefilter2D(d_temp, dimsimage, batch);
				d_BindTextureToArray(d_temp, ha_image, ht_image, dimsimage, cudaFilterModeLinear, false, batch);
				cudaFree(d_temp);
			}
			else
				d_BindTextureToArray(d_image, ha_image, ht_image, dimsimage, cudaFilterModeLinear, false, batch);

			cudaTex* dt_image = (cudaTex*)CudaMallocFromHostArray(ht_image, batch * sizeof(cudaTex));
			glm::mat4* d_transforms = (glm::mat4*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat4));

			dim3 TpB = dim3(8, 8, 8);
			dim3 grid = dim3((dimsvolume.x + 7) / 8, (dimsvolume.y + 7) / 8, (dimsvolume.z + 7) / 8);

			if (outputzerocentered)
			{
				if (mode == T_INTERP_LINEAR)
					ProjBackwardKernel<true, false> << <grid, TpB >> > (d_volume, toUint3(dimsvolume), dt_image, toUint2(dimsimage), d_transforms, batch);
				else
					ProjBackwardKernel<true, true> << <grid, TpB >> > (d_volume, toUint3(dimsvolume), dt_image, toUint2(dimsimage), d_transforms, batch);
			}
			else
			{
				if (mode == T_INTERP_LINEAR)
					ProjBackwardKernel<false, false> << <grid, TpB >> > (d_volume, toUint3(dimsvolume), dt_image, toUint2(dimsimage), d_transforms, batch);
				else
					ProjBackwardKernel<false, true> << <grid, TpB >> > (d_volume, toUint3(dimsvolume), dt_image, toUint2(dimsimage), d_transforms, batch);
			}

			for (int n = 0; n < batch; n++)
			{
				cudaDestroyTextureObject(ht_image[n]);
				cudaFreeArray(ha_image[n]);
			}

			free(ht_image);
			free(ha_image);
			cudaFree(d_transforms);
			cudaFree(dt_image);
		}
		else if (mode == T_INTERP_SINC)
		{
			dim3 TpB = dim3(SincWindow, SincWindow);
			dim3 grid = dim3(dimsvolume.x, dimsvolume.y, dimsvolume.z);

			for (int b = 0; b < batch; b++)
				if (outputzerocentered)
					ProjBackwardSincKernel<true> << <grid, TpB >> > (d_volume, dimsvolume, d_image + Elements2(dimsimage) * b, dimsimage, h_transforms[b]);
				else
					ProjBackwardSincKernel<false> << <grid, TpB >> > (d_volume, dimsvolume, d_image + Elements2(dimsimage) * b, dimsimage, h_transforms[b]);
		}

		free(h_transforms);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template <bool iscentered, bool cubicinterp> __global__ void ProjBackwardKernel(tfloat* d_volume, uint3 dimsvolume, cudaTex* dt_image, uint2 dimsimage, glm::mat4* d_transforms, uint nimages)
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

		uint outx, outy, outz;
		if (!iscentered)
		{
			outx = (idx + (dimsvolume.x + 1) / 2) % dimsvolume.x;
			outy = (idy + (dimsvolume.y + 1) / 2) % dimsvolume.y;
			outz = (idz + (dimsvolume.z + 1) / 2) % dimsvolume.z;
		}
		else
		{
			outx = idx;
			outy = idy;
			outz = idz;
		}

		glm::vec4 position = glm::vec4(idx, idy, idz, 1);
		tfloat sum = 0.0f;
		uint samples = 0;

		for (uint n = 0; n < nimages; n++)
		{
			glm::vec4 positiontrans = d_transforms[n] * position;

			if (positiontrans.x >= -1e-6f && positiontrans.x <= dimsimage.x + 1e-6f && positiontrans.y >= -1e-6f && positiontrans.y <= dimsimage.y + 1e-6f)
			{
				if (cubicinterp)
					sum += cubicTex2D(dt_image[n], positiontrans.x, positiontrans.y);
				else
					sum += tex2D<tfloat>(dt_image[n], positiontrans.x, positiontrans.y);
				samples++;
			}
		}

		if (samples > 0)
			d_volume[(outz * dimsvolume.y + outy) * dimsvolume.x + outx] += sum / (tfloat)samples;
	}

	template <bool iscentered> __global__ void ProjBackwardSincKernel(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int2 dimsimage, glm::mat4 transform)
	{
		__shared__ float s_sums[SincWindow][SincWindow];
		s_sums[threadIdx.y][threadIdx.x] = 0.0f;

		int outx, outy, outz;
		if (!iscentered)
		{
			outx = (blockIdx.x + (dimsvolume.x + 1) / 2) % dimsvolume.x;
			outy = (blockIdx.y + (dimsvolume.y + 1) / 2) % dimsvolume.y;
			outz = (blockIdx.z + (dimsvolume.z + 1) / 2) % dimsvolume.z;
		}
		else
		{
			outx = blockIdx.x;
			outy = blockIdx.y;
			outz = blockIdx.z;
		}

		glm::vec4 position = glm::vec4((int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, 1);
		position = transform * position;
		if (position.x < 0 || position.x > dimsimage.x - 1 || position.y < 0 || position.y > dimsimage.y - 1)
			return;

		int startx = (int)position.x - SincWindow / 2;
		int starty = (int)position.y - SincWindow / 2;
		float sum = 0.0;

		int y = (int)threadIdx.y + starty;
		int addressy = (y + dimsimage.y) % dimsimage.y;

		int x = (int)threadIdx.x + startx;
		float weight = sinc(position.x - (float)x) * sinc(position.y - (float)y);
		int addressx = (x + dimsimage.x) % dimsimage.x;

		sum += d_image[addressy * dimsimage.x + addressx] * weight;

		s_sums[threadIdx.y][threadIdx.x] = sum;
		__syncthreads();

		if (threadIdx.x == 0)
		{
#pragma unroll
			for (char i = 1; i < SincWindow; i++)
				sum += s_sums[threadIdx.y][i];
			s_sums[threadIdx.y][0] = sum;
		}
		__syncthreads();

		if (threadIdx.y == 0 && threadIdx.x == 0)
		{
#pragma unroll
			for (char i = 1; i < SincWindow; i++)
				sum += s_sums[i][0];
			d_volume[(outz * gridDim.y + outy) * gridDim.x + outx] += sum;
		}
	}
}