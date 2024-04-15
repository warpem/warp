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

	template <bool cubicinterp> __global__ void RaySumKernel(cudaTex t_volume, glm::vec3* d_start, glm::vec3* d_finish, tfloat* d_sums, uint supersample, uint batch);


	///////////////////////////////////////////
	//Computes integrals along the given rays//
	///////////////////////////////////////////

	void d_RaySum(cudaTex t_volume, glm::vec3* d_start, glm::vec3* d_finish, tfloat* d_sums, T_INTERP_MODE mode, uint supersample, uint batch)
	{
		uint TpB = tmin(128, NextMultipleOf(batch, 32));
		dim3 grid = dim3(tmin((batch + TpB - 1) / TpB, 32768), 1, 1);
		if (mode == T_INTERP_CUBIC)
			RaySumKernel<true> << <grid, TpB >> > (t_volume, d_start, d_finish, d_sums, supersample, batch);
		else
			RaySumKernel<false> << <grid, TpB >> > (t_volume, d_start, d_finish, d_sums, supersample, batch);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template <bool cubicinterp> __global__ void RaySumKernel(cudaTex t_volume, glm::vec3* d_start, glm::vec3* d_finish, tfloat* d_sums, uint supersample, uint batch)
	{
		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < batch; id += gridDim.x * blockDim.x)
		{
			glm::vec3 start = d_start[id];
			glm::vec3 direction = d_finish[id] - start;
			uint steps = ceil(glm::length(direction)) * supersample;
			direction /= (tfloat)(steps - 1);

			tfloat sum = 0;
			for (uint s = 0; s < steps; s++)
			{
				glm::vec3 pos = start + (float)s * direction;
				if (cubicinterp)
					sum += cubicTex3D(t_volume, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);
				else
					sum += tex3D<tfloat>(t_volume, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);
			}

			d_sums[id] = sum;
		}
	}
}