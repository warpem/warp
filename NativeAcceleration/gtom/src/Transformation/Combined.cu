#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
#define SincWindow 16

	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template<bool iscentered> __global__ void ScaleRotateShift2DSincKernel(tfloat* d_input, tfloat* d_output, int2 dims, glm::mat3 transform);
	template<bool iscentered> __global__ void ScaleRotateShift2DCubicKernel(cudaTex t_input, tfloat* d_output, int2 dims, glm::mat3 transform);


	//////////////////////////////////////////
	//Scale, rotate and shift 2D in one step//
	//////////////////////////////////////////

	void d_ScaleRotateShift2D(tfloat* d_input, tfloat* d_output, int2 dims, tfloat2* h_scales, tfloat* h_angles, tfloat2* h_shifts, T_INTERP_MODE mode, bool outputzerocentered, int batch)
	{
		glm::mat3* h_transforms = (glm::mat3*)malloc(batch * sizeof(glm::mat3));
		for (int b = 0; b < batch; b++)
			h_transforms[b] = Matrix3Translation(tfloat2(dims.x / 2, dims.y / 2)) *
			Matrix3Scale(tfloat3(1.0f / h_scales[b].x, 1.0f / h_scales[b].y, 1.0f)) *
			Matrix3RotationZ(-h_angles[b]) *
			Matrix3Translation(tfloat2(-dims.x / 2 - h_shifts[b].x, -dims.y / 2 - h_shifts[b].y));

		if (mode == T_INTERP_SINC)
		{
			dim3 TpB = dim3(SincWindow, SincWindow);
			dim3 grid = dim3(dims.x, dims.y);
			for (int b = 0; b < batch; b++)
				if (outputzerocentered)
					ScaleRotateShift2DSincKernel<true> << <grid, TpB >> > (d_input + dims.x * dims.y * b, d_output + dims.x * dims.y * b, dims, h_transforms[b]);
				else
					ScaleRotateShift2DSincKernel<false> << <grid, TpB >> > (d_input + dims.x * dims.y * b, d_output + dims.x * dims.y * b, dims, h_transforms[b]);
		}
		else if (mode == T_INTERP_CUBIC)
		{
			cudaArray* a_input;
			cudaTex t_input;
			tfloat* d_prefilter;
			cudaMalloc((void**)&d_prefilter, dims.x * dims.y * sizeof(tfloat));

			for (int b = 0; b < batch; b++)
			{
				cudaMemcpy(d_prefilter, d_input + dims.x * dims.y * b, dims.x * dims.y * sizeof(tfloat), cudaMemcpyDeviceToDevice);
				d_CubicBSplinePrefilter2D(d_prefilter, dims);
				d_BindTextureToArray(d_prefilter, a_input, t_input, dims, cudaFilterModeLinear, false);

				d_ScaleRotateShiftCubic2D(t_input, d_output + dims.x * dims.y * b, dims, h_scales[b], h_angles[b], h_shifts[b], outputzerocentered);

				cudaDestroyTextureObject(t_input);
				cudaFree(a_input);
			}
		}

		free(h_transforms);
	}

	void d_ScaleRotateShiftCubic2D(cudaTex t_input, tfloat* d_output, int2 dims, tfloat2 scale, tfloat angle, tfloat2 shift, bool outputzerocentered)
	{
		glm::mat3 transform = Matrix3Translation(tfloat2(dims.x / 2, dims.y / 2)) *
			Matrix3Scale(tfloat3(1.0f / scale.x, 1.0f / scale.y, 1.0f)) *
			Matrix3RotationZ(-angle) *
			Matrix3Translation(tfloat2(-dims.x / 2 - shift.x, -dims.y / 2 - shift.y));

		dim3 grid = dim3((dims.x + 15) / 16, (dims.y + 15) / 16);
		dim3 TpB = dim3(16, 16);

		if (outputzerocentered)
			ScaleRotateShift2DCubicKernel<true> << <grid, TpB >> > (t_input, d_output, dims, transform);
		else
			ScaleRotateShift2DCubicKernel<false> << <grid, TpB >> > (t_input, d_output, dims, transform);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<bool iscentered> __global__ void ScaleRotateShift2DSincKernel(tfloat* d_input, tfloat* d_output, int2 dims, glm::mat3 transform)
	{
		__shared__ tfloat s_sums[SincWindow][SincWindow];
		s_sums[threadIdx.y][threadIdx.x] = 0.0f;

		int outx, outy;
		if (!iscentered)
		{
			outx = (blockIdx.x + (dims.x + 1) / 2) % dims.x;
			outy = (blockIdx.y + (dims.y + 1) / 2) % dims.y;
		}
		else
		{
			outx = blockIdx.x;
			outy = blockIdx.y;
		}

		glm::vec3 position = glm::vec3(blockIdx.x, blockIdx.y, 1.0f);
		position = transform * position;
		if (position.x < 0 || position.x > dims.x - 1 || position.y < 0 || position.y > dims.y - 1)
		{
			if (threadIdx.y == 0 && threadIdx.x == 0)
				d_output[outy * dims.x + outx] = 0.0f;
			return;
		}

		float sum = 0.0f;

		int xx = (int)threadIdx.x + (int)position.x - SincWindow / 2;
		int yy = (int)threadIdx.y + (int)position.y - SincWindow / 2;
		float weight = sinc(position.x - (float)xx) * sinc(position.y - (float)yy);

		int addressx = (xx + dims.x) % dims.x;
		int addressy = (yy + dims.y) % dims.y;

		sum = d_input[addressy * dims.x + addressx] * weight;
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
			d_output[outy * dims.x + outx] = sum;
		}
	}

	template<bool iscentered> __global__ void ScaleRotateShift2DCubicKernel(cudaTex t_input, tfloat* d_output, int2 dims, glm::mat3 transform)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idx >= dims.x || idy >= dims.y)
			return;

		int outx, outy;
		if (!iscentered)
		{
			outx = (idx + (dims.x + 1) / 2) % dims.x;
			outy = (idy + (dims.y + 1) / 2) % dims.y;
		}
		else
		{
			outx = idx;
			outy = idy;
		}

		glm::vec3 position = glm::vec3(idx, idy, 1);
		position = transform * position + 0.5f;

		d_output[dims.x * outy + outx] = cubicTex2D(t_input, position.x, position.y);
	}
}