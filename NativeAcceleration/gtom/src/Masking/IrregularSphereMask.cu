#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Helper.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T, int ndims> __global__ void IrregularSphereMaskKernel(T* d_input, T* d_output, int3 dims, tfloat sigma, tfloat3 center, cudaTex radiusTex);


	///////////
	//Globals//
	///////////

	// Will replace this global texture reference with texture objects


	////////////////
	//Host methods//
	////////////////

	template <class T> void d_IrregularSphereMask(T* d_input,
		T* d_output,
		int3 dims,
		tfloat* d_radiusmap,
		int2 anglesteps,
		tfloat sigma,
		tfloat3* center,
		int batch)
	{
		// Create pitched memory with proper alignment
		tfloat* d_pitched = NULL;
		int pitchedwidth = anglesteps.x * sizeof(tfloat);
		d_pitched = (tfloat*)CudaMallocAligned2D(anglesteps.x * sizeof(tfloat), anglesteps.y, &pitchedwidth);
		for (int y = 0; y < anglesteps.y; y++)
			cudaMemcpy((char*)d_pitched + y * pitchedwidth,
			d_radiusmap + y * anglesteps.x,
			anglesteps.x * sizeof(tfloat),
			cudaMemcpyDeviceToDevice);

		// Create a CUDA array and texture object
		cudaArray_t radiusArray;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
		cudaMallocArray(&radiusArray, &desc, anglesteps.x, anglesteps.y);
		d_CopyToArray(d_pitched, radiusArray, anglesteps.x, anglesteps.y, 1, sizeof(tfloat), cudaMemcpyDeviceToDevice);
		
		// Create texture object with mirror address mode
		cudaTex radiusTex = d_CreateTextureObject(radiusArray, cudaFilterModeLinear, cudaReadModeElementType, true, cudaAddressModeMirror);

		tfloat3 _center = center != NULL ? *center : tfloat3(dims.x / 2, dims.y / 2, dims.z / 2);

		int TpB = min(NextMultipleOf(dims.x, 32), 256);
		dim3 grid = dim3(dims.y, dims.z, batch);
		if (DimensionCount(dims) <= 2)
			IrregularSphereMaskKernel<T, 2> << <grid, TpB >> > (d_input, d_output, dims, sigma, _center, radiusTex);
		else
			IrregularSphereMaskKernel<T, 3> << <grid, TpB >> > (d_input, d_output, dims, sigma, _center, radiusTex);

		// Clean up
		cudaDestroyTextureObject(radiusTex);
		cudaFreeArray(radiusArray);
		cudaFree(d_pitched);
	}
	template void d_IrregularSphereMask<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* d_radiusmap, int2 anglesteps, tfloat sigma, tfloat3* center, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T, int ndims> __global__ void IrregularSphereMaskKernel(T* d_input, T* d_output, int3 dims, tfloat sigma, tfloat3 center, cudaTex radiusTex)
	{
		if (threadIdx.x >= dims.x)
			return;

		//For batch mode
		int offset = blockIdx.z * Elements(dims) + blockIdx.y * dims.x * dims.y + blockIdx.x * dims.x;

		int x, y, z;
		float length;
		T maskvalue;

		//Squared y and z distance from center
		y = blockIdx.x - center.y;
		if (ndims > 2)
			z = blockIdx.y - center.z;
		else
			z = 0;

		for (int idx = threadIdx.x; idx < dims.x; idx += blockDim.x)
		{
			x = idx - center.x;

			length = sqrt((float)(x * x + y * y + z * z));

			glm::vec3 direction((float)x / length, (float)y / length, (float)z / length);
			float theta = acos((float)(-direction.x));
			float phi = atan2((float)direction.y / sin(theta), (float)direction.z / sin(theta));

			theta /= PI * 0.5f;
			phi /= PI2;

			// Use modern texture object API
			tfloat radius = tex2D<tfloat>(radiusTex, phi, theta);

			if (length < radius)
				maskvalue = 1;
			else
			{
				//Smooth border
				if (sigma > (tfloat)0)
				{
					maskvalue = exp(-((length - radius) * (length - radius) / (sigma * sigma)));
					if (maskvalue < (tfloat)0.1353)
						maskvalue = 0;
				}
				//Hard border
				else
					maskvalue = max((T)1 - (length - radius), (T)0);
			}

			//Write masked input to output
			d_output[offset + idx] = maskvalue * d_input[offset + idx];
			//d_output[offset + idx] = radius;
		}
	}
}