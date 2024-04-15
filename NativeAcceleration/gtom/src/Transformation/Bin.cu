#include "gtom/include/Prerequisites.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void Bin1DKernel(tfloat* d_input, tfloat* d_output, size_t elements, int binsize);
	__global__ void Bin2DKernel(tfloat* d_output, int width);
	__global__ void Bin3DKernel(tfloat* d_input, tfloat* d_output, int width, int height, int binnedwidth, int binnedheight, int binsize);


	///////////
	//Globals//
	///////////

	texture<tfloat, 2> texInput2d;

	/////////////////////////////////////
	//Binning with linear interpolation//
	/////////////////////////////////////

	void d_Bin(tfloat* d_input, tfloat* d_output, int3 dims, int bincount, int batch)
	{
		for (int b = 0; b < batch; b++)
		{
			tfloat* d_intermediate = NULL;

			if (dims.z <= 1 && dims.y <= 1)	//1D
			{
				int TpB = min(192, dims.x / (1 << bincount));
				int totalblocks = min((dims.x / (1 << bincount) + TpB - 1) / TpB, 32768);
				dim3 grid = dim3((uint)totalblocks);

				size_t elements = dims.x;
				size_t binnedelements = dims.x / (1 << bincount);

				Bin1DKernel << <grid, (uint)TpB >> > (d_input + elements * b, d_output + binnedelements * b, dims.x / (1 << bincount), 1 << bincount);
			}
			else if (dims.z <= 1)			//2D
			{
				if (bincount > 1)
					cudaMalloc((void**)&d_intermediate, dims.x * dims.y / 4 * sizeof(tfloat));

				texInput2d.normalized = false;
				texInput2d.filterMode = cudaFilterModeLinear;

				size_t elements = dims.x * dims.y;
				size_t binnedelements = dims.x * dims.y / (1 << (bincount * 2));

				for (int i = 0; i < bincount; i++)
				{
					cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
					cudaBindTexture2D(NULL,
						texInput2d,
						i == 0 ? d_input + elements * b : d_intermediate,
						desc,
						dims.x / (1 << i),
						dims.y / (1 << i),
						dims.x / (1 << i) * sizeof(tfloat));

					int TpB = min(256, dims.x / (2 << i));
					int totalblocks = min((dims.x / (2 << i) + TpB - 1) / TpB, 32768);
					dim3 grid = dim3((uint)totalblocks, dims.y / (2 << i));

					tfloat* d_result;
					if (i < bincount - 1)
						cudaMalloc((void**)&d_result, dims.x / (2 << i) * dims.y / (2 << i) * sizeof(tfloat));
					else
						d_result = d_output + binnedelements * b;

					Bin2DKernel << <grid, (uint)TpB >> > (d_result, dims.x / (2 << i));

					cudaUnbindTexture(texInput2d);
					if (d_result != d_output + binnedelements * b)
					{
						if (d_intermediate != NULL)
							cudaFree(d_intermediate);
						d_intermediate = d_result;
					}
				}
			}
			else							//3D
			{
				int TpB = min(192, dims.x / (1 << bincount));
				int totalblocks = min((dims.x / (1 << bincount) + TpB - 1) / TpB, 32768);
				dim3 grid = dim3((uint)totalblocks, dims.y / (1 << bincount), dims.z / (1 << bincount));

				size_t elements = dims.x * dims.y * dims.z;
				size_t binnedelements = dims.x * dims.y * dims.z / (1 << (bincount * 3));

				Bin3DKernel << <grid, (uint)TpB >> > (d_input + elements * b,
					d_output + binnedelements * b,
					dims.x,
					dims.y,
					dims.x / (1 << bincount),
					dims.y / (1 << bincount),
					1 << bincount);
			}

			if (d_intermediate != NULL)
				cudaFree(d_intermediate);
		}
	}

	////////////////
	//CUDA kernels//
	////////////////

	__global__ void Bin1DKernel(tfloat* d_input, tfloat* d_output, size_t elements, int binsize)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
		{
			tfloat binsum = (tfloat)0;
			for (int i = 0; i < binsize; i++)
				binsum += d_input[id * binsize + i];
			d_output[id] = binsum / (tfloat)binsize;
		}
	}

	__global__ void Bin2DKernel(tfloat* d_output, int width)
	{
		for (int x = blockIdx.x * blockDim.x + threadIdx.x;
			x < width;
			x += blockDim.x * gridDim.x)
			d_output[blockIdx.y * width + x] = tex2D(texInput2d, (float)(x * 2 + 1), (float)(blockIdx.y * 2 + 1));
	}

	__global__ void Bin3DKernel(tfloat* d_input, tfloat* d_output, int width, int height, int binnedwidth, int binnedheight, int binsize)
	{
		int binvolume = binsize * binsize * binsize;
		for (int x = blockIdx.x * blockDim.x + threadIdx.x;
			x < binnedwidth;
			x += blockDim.x * gridDim.x)
		{
			tfloat binsum = (tfloat)0;
			for (int bz = 0; bz < binsize; bz++)
			{
				int offsetz = blockIdx.z * binsize + bz;
				for (int by = 0; by < binsize; by++)
				{
					int offsety = blockIdx.y * binsize + by;
					for (int bx = 0; bx < binsize; bx++)
						binsum += d_input[(offsetz * height + offsety) * width + x * binsize + bx];
				}
			}

			d_output[(blockIdx.z * binnedheight + blockIdx.y) * binnedwidth + x] = binsum / (tfloat)binvolume;
		}
	}
}