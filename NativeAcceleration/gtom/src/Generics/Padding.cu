#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Generics.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void PadNothingKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, uint batch);
	template <class T> __global__ void PadValueKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, T value, uint batch);
	template <class T> __global__ void PadMirrorKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, uint batch);
	template <class T> __global__ void PadTileKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, uint batch);
	template <class T> __global__ void PadClampKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, uint batch);


	/////////////////////////////////////////////////////////////////////
	//Extract a portion of 1/2/3-dimensional data with cyclic boudaries//
	/////////////////////////////////////////////////////////////////////

	template <class T> void d_Pad(T* d_input, T* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, T value, int batch)
	{
		int3 inputcenter = toInt3(inputdims.x / 2, inputdims.y / 2, inputdims.z / 2);
		int3 outputcenter = toInt3(outputdims.x / 2, outputdims.y / 2, outputdims.z / 2);

		size_t TpB = min(256, NextMultipleOf(outputdims.x, 32));
		dim3 grid = dim3((outputdims.x + TpB - 1) / TpB, outputdims.y, outputdims.z);

		int3 offset = toInt3(inputcenter.x - outputcenter.x, inputcenter.y - outputcenter.y, inputcenter.z - outputcenter.z);

		if (mode == T_PAD_VALUE)
			PadValueKernel << <grid, (int)TpB >> > (d_input, d_output, inputdims, outputdims, offset, value, batch);
		else if (mode == T_PAD_MIRROR)
			PadMirrorKernel << <grid, (int)TpB >> > (d_input, d_output, inputdims, outputdims, offset, batch);
		else if (mode == T_PAD_TILE)
			PadTileKernel << <grid, (int)TpB >> > (d_input, d_output, inputdims, outputdims, offset, batch);
		else if (mode == T_PAD_CLAMP)
			PadClampKernel << <grid, (int)TpB >> > (d_input, d_output, inputdims, outputdims, offset, batch);
		else if (mode == T_PAD_NOTHING)
			PadNothingKernel << <grid, (int)TpB >> > (d_input, d_output, inputdims, outputdims, offset, batch);
	}
	template void d_Pad<int>(int* d_input, int* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, int value, int batch);
	template void d_Pad<half>(half* d_input, half* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, half value, int batch);
	template void d_Pad<float>(float* d_input, float* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, float value, int batch);
	template void d_Pad<double>(double* d_input, double* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, double value, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void PadNothingKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, uint batch)
	{
		int idy = blockIdx.y;
		int idz = blockIdx.z;

		bool outofbounds = false;

		int oy, oz, ox;

		oy = offset.y + idy;
		if (oy < 0 || oy >= inputdims.y)
		{
			outofbounds = true;
		}
		else
		{
			oz = offset.z + idz;
			if (oz < 0 || oz >= inputdims.z)
				outofbounds = true;
		}

		d_output += (idz * outputdims.y + idy) * outputdims.x;
		d_input += (oz * inputdims.y + oy) * inputdims.x;
		size_t elementsinput = Elements(inputdims);
		size_t elementsoutput = Elements(outputdims);

		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outputdims.x; idx += blockDim.x * gridDim.x)
		{
			for (uint b = 0; b < batch; b++)
			{
				if (outofbounds)
					continue;
				else
				{
					ox = offset.x + idx;
					if (ox < 0 || ox >= inputdims.x)
						continue;
					else
						d_output[b * elementsoutput + idx] = d_input[b * elementsinput + ox];
				}
			}
		}
	}

	template <class T> __global__ void PadValueKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, T value, uint batch)
	{
		int idy = blockIdx.y;
		int idz = blockIdx.z;

		bool outofbounds = false;

		int oy, oz, ox;

		oy = offset.y + idy;
		if (oy < 0 || oy >= inputdims.y)
		{
			outofbounds = true;
		}
		else
		{
			oz = offset.z + idz;
			if (oz < 0 || oz >= inputdims.z)
				outofbounds = true;
		}

		d_output += (idz * outputdims.y + idy) * outputdims.x;
		d_input += (oz * inputdims.y + oy) * inputdims.x;
		size_t elementsinput = Elements(inputdims);
		size_t elementsoutput = Elements(outputdims);

		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outputdims.x; idx += blockDim.x * gridDim.x)
		{
			for (uint b = 0; b < batch; b++)
			{
				if (outofbounds)
					d_output[b * elementsoutput + idx] = value;
				else
				{
					ox = offset.x + idx;
					if (ox < 0 || ox >= inputdims.x)
						d_output[b * elementsoutput + idx] = value;
					else
						d_output[b * elementsoutput + idx] = d_input[b * elementsinput + ox];
				}
			}
		}
	}

	template <class T> __global__ void PadMirrorKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, uint batch)
	{
		int idy = blockIdx.y;
		int idz = blockIdx.z;

		uint ox = 0, oy = 0, oz = 0;
		if (inputdims.y > 1)
		{
			oy = (uint)(offset.y + idy + inputdims.y * 999) % (uint)(inputdims.y * 2);
			if (oy >= inputdims.y)
				oy = inputdims.y * 2 - 1 - oy;
		}
		if (inputdims.z > 1)
		{
			oz = (uint)(offset.z + idz + inputdims.z * 999) % (uint)(inputdims.z * 2);
			if (oz >= inputdims.z)
				oz = inputdims.z * 2 - 1 - oz;
		}

		d_output += (idz * outputdims.y + idy) * outputdims.x;
		d_input += (oz * inputdims.y + oy) * inputdims.x;
		size_t elementsinput = Elements(inputdims);
		size_t elementsoutput = Elements(outputdims);

		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outputdims.x; idx += blockDim.x * gridDim.x)
		{
			ox = (uint)(offset.x + idx + inputdims.x * 99998) % (uint)(inputdims.x * 2);
			if (ox >= inputdims.x)
				ox = inputdims.x * 2 - 1 - ox;
			for (uint b = 0; b < batch; b++)
				d_output[b * elementsoutput + idx] = d_input[b * elementsinput + ox];
		}
	}

	template <class T> __global__ void PadTileKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, uint batch)
	{
		int idy = blockIdx.y;
		int idz = blockIdx.z;

		int oy = (offset.y + idy + inputdims.y * 999) % inputdims.y;
		int oz = (offset.z + idz + inputdims.z * 999) % inputdims.z;

		d_output += (idz * outputdims.y + idy) * outputdims.x;
		d_input += (oz * inputdims.y + oy) * inputdims.x;
		size_t elementsinput = Elements(inputdims);
		size_t elementsoutput = Elements(outputdims);

		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outputdims.x; idx += blockDim.x * gridDim.x)
			for (uint b = 0; b < batch; b++)
				d_output[b * elementsoutput + idx] = d_input[b * elementsinput + (offset.x + idx + inputdims.x * 99999) % inputdims.x];
	}

	template <class T> __global__ void PadClampKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, uint batch)
	{
		int idy = blockIdx.y;
		int idz = blockIdx.z;

		int oy, oz, ox;

		oy = offset.y + idy;
		oy = tmax(0, tmin(inputdims.y - 1, oy));

		oz = offset.z + idz;
		oz = tmax(0, tmin(inputdims.z - 1, oz));

		d_output += (idz * outputdims.y + idy) * outputdims.x;
		d_input += (oz * inputdims.y + oy) * inputdims.x;
		size_t elementsinput = Elements(inputdims);
		size_t elementsoutput = Elements(outputdims);

		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outputdims.x; idx += blockDim.x * gridDim.x)
		{
			for (uint b = 0; b < batch; b++)
			{
				ox = offset.x + idx;
				ox = tmax(0, tmin(inputdims.x - 1, ox));

				d_output[b * elementsoutput + idx] = d_input[b * elementsinput + ox];
			}
		}
	}
}
