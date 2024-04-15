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

	__global__ void InsertAdditiveKernel(tfloat* d_input, tfloat* d_output, int3 sourcedims, int3 regiondims, size_t regionelements, int3* d_regionorigins);


	///////////////////////////////////////////////////////////////////////////////////////
	//Add windows of data to a common target at specified positions with cyclic boudaries//
	///////////////////////////////////////////////////////////////////////////////////////
	
	void d_InsertAdditive(tfloat* d_input, tfloat* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch)
	{
		size_t TpB = min(256, NextMultipleOf(regiondims.x, 32));
		dim3 grid = dim3(regiondims.y, regiondims.z, batch);
		InsertAdditiveKernel << <grid, (int)TpB >> > (d_input, d_output, sourcedims, regiondims, Elements(regiondims), d_regionorigins);
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void InsertAdditiveKernel(tfloat* d_input, tfloat* d_output, int3 sourcedims, int3 regiondims, size_t regionelements, int3* d_regionorigins)
	{
		int3 regionorigin = d_regionorigins[blockIdx.z];
		int idy = blockIdx.x;
		int idz = blockIdx.y;

		int oy = (idy + regionorigin.y + sourcedims.y * 999) % sourcedims.y;
		int oz = (idz + regionorigin.z + sourcedims.z * 999) % sourcedims.z;

		d_input += blockIdx.z * regionelements + (idz * regiondims.y + idy) * regiondims.x;
		d_output += (oz * sourcedims.y + oy) * sourcedims.x;

		for (int idx = threadIdx.x; idx < regiondims.x; idx += blockDim.x)
			atomicAdd((tfloat*)(d_output + ((idx + regionorigin.x + sourcedims.x * 999) % sourcedims.x)), d_input[idx]);
	}
}