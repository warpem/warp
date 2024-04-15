#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Correlation.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/IO.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Helper.cuh"
#include <vector>


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void LocalPeaksKernel(tfloat* d_input, float* d_output, int3 dims, int localextent, tfloat threshold, int idz);
	template<int ndims, int rad> __global__ void SubpixelMaxKernel(tfloat* d_input, tfloat* d_output, int3 dims, float3 s);


	////////////////////////////////////////////////
	// Find local peaks above specified threshold //
	////////////////////////////////////////////////

	void d_LocalPeaks(tfloat* d_input, int3** h_peaks, int* h_peaksnum, int3 dims, int localextent, tfloat threshold, int batch)
	{
		int TpB = tmin(128, NextMultipleOf(dims.x, 32));
		dim3 grid = dim3(tmin((dims.x + TpB - 1) / TpB, 32768), dims.y, 1);

		float* h_output = (float*)malloc(Elements(dims) * sizeof(float));

		std::vector<int3> peaks;

		for (int b = 0; b < batch; b++)
		{
			peaks.clear();

			float* d_output = CudaMallocValueFilled(Elements(dims), 0.0f);

			for (int idz = 0; idz < dims.z; idz++)
				LocalPeaksKernel << <grid, (uint)TpB >> > (d_input + Elements(dims) * b, d_output, dims, localextent, threshold, idz);

			//d_WriteMRC(d_output, dims, "d_localpeaks.mrc");

			cudaMemcpy(h_output, d_output, Elements(dims) * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_output);

			for (int z = 0; z < dims.z; z++)
				for (int y = 0; y < dims.y; y++)
					for (int x = 0; x < dims.x; x++)
						if (h_output[(z * dims.y + y) * dims.x + x] > 0)
							peaks.push_back(toInt3(x, y, z));

			if (peaks.size() > 0)
			{
				h_peaks[b] = (int3*)malloc(peaks.size() * sizeof(int3));
				memcpy(h_peaks[b], &peaks[0], peaks.size() * sizeof(int3));
			}
			h_peaksnum[b] = peaks.size();
		}

		free(h_output);
	}

	void d_SubpixelMax(tfloat* d_input, tfloat* d_output, int3 dims, int subpixsteps)
	{
		int ndims = DimensionCount(dims);
		float steplength = 1.0f / subpixsteps;

		int TpB = 128;
		dim3 grid = dim3((dims.x - 15 + TpB - 1) / TpB, dims.y - 15, ndims > 2 ? dims.z - 15 : 1);

		for (int sz = 0; sz < (ndims == 3 ? subpixsteps : 1); sz++)
			for (int sy = 0; sy < subpixsteps; sy++)
				for (int sx = 0; sx < subpixsteps; sx++)
				{
					float3 s = make_float3(sx * steplength - 0.5f + steplength / 2,
										   sy * steplength - 0.5f + steplength / 2,
										   sz * steplength - 0.5f + steplength / 2);
					if (ndims < 3)
						s.z = 0;

					if (ndims == 2)
						SubpixelMaxKernel<2, 6> << <grid, TpB >> > (d_input, d_output, dims, s);
					else if (ndims == 3)
						SubpixelMaxKernel<3, 6> << <grid, TpB >> > (d_input, d_output, dims, s);
					else
						throw;

					cudaDeviceSynchronize();
				}
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void LocalPeaksKernel(tfloat* d_input, float* d_output, int3 dims, int localextent, tfloat threshold, int idz)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;

		int idy = blockIdx.y;

		tfloat value = d_input[(idz * dims.y + idy) * dims.x + idx];
		if (value < threshold)
			return;

		int limx = tmin(dims.x - 1, idx + localextent);
		int limy = tmin(dims.y - 1, idy + localextent);
		int limz = tmin(dims.z - 1, idz + localextent);

		int sqlocalextent = localextent * localextent;
		int sqy, sqz;
		int sqdist;

		for (int z = tmax(0, idz - localextent); z <= limz; z++)
		{
			sqz = idz - z;
			sqz *= sqz;
			for (int y = tmax(0, idy - localextent); y <= limy; y++)
			{
				sqy = idy - y;
				sqy *= sqy;
				sqy += sqz;
				for (int x = tmax(0, idx - localextent); x <= limx; x++)
				{
					sqdist = idx - x;
					sqdist *= sqdist;
					sqdist += sqy;

					if (sqdist > sqlocalextent + 1e-5f || sqdist == 0)
						continue;

					if (value < d_input[(z * dims.y + y) * dims.x + x])
						return;
				}
			}
		}

		d_output[(idz * dims.y + idy) * dims.x + idx] = 1.0f;
	}

	template<int ndims, int rad> __global__ void SubpixelMaxKernel(tfloat* d_input, tfloat* d_output, int3 dims, float3 s)
	{
		int cz = ndims == 3 ? blockIdx.z + rad : 0;
		int cy = blockIdx.y + rad;
		int cx = blockIdx.x * blockDim.x + threadIdx.x + rad;

		if (cx >= dims.x - rad - 2 || cy >= dims.y - rad - 2)
			return;

		if (ndims == 3 && cz >= dims.z - rad - 2)
			return;
	
		tfloat sum = 0;

		for (int z = (ndims == 3 ? -rad : 0); z <= (ndims == 3 ? rad : 0); z++)
		{
			float sincz = ndims == 3 ? sinc(s.z - z) : 1;

			for (int y = -rad; y <= rad; y++)
			{
				float sincy = sinc(s.y - y);

				for (int x = -rad; x <= rad; x++)
				{
					float sincx = sinc(s.x - x);

					sum += d_input[((cz + z) * dims.y + cy + y) * dims.x + cx + x] * sincx * sincy * sincz;
				}
			}
		}

		d_output[(cz * dims.y + cy) * dims.x + cx] = tmax(d_output[(cz * dims.y + cy) * dims.x + cx], sum);
	}
}