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

	__global__ void DistanceMapKernel(tfloat* d_olddistance, tfloat* d_newdistance, int* d_upstreamneighbor, int3 dims);
	__global__ void DistanceMapFinalizeKernel(tfloat* d_olddistance, tfloat* d_newdistance, int* d_upstreamneighbor, int3 dims);
	__global__ void DistanceMapExact3DKernel(tfloat* d_input, tfloat* d_output, int3 dims, int maxdistance, int idz);
	__global__ void DistanceMapExact2DKernel(tfloat* d_input, tfloat* d_output, int3 dims, int maxdistance);


	///////////////////////////////////////////
	//Compute distance map for a binary input//
	///////////////////////////////////////////

	void d_DistanceMap(tfloat* d_input, tfloat* d_output, int3 dims, int maxiterations)
	{
		tfloat* d_distance;
		cudaMalloc((void**)&d_distance, Elements(dims) * sizeof(tfloat));
		tfloat* d_newdistance;
		cudaMalloc((void**)&d_newdistance, Elements(dims) * sizeof(tfloat));
		int* d_upstreamneighbor = CudaMallocValueFilled(Elements(dims), -1);

		d_OneMinus(d_input, d_distance, Elements(dims));
		d_MultiplyByScalar(d_distance, d_distance, Elements(dims), (tfloat)(maxiterations * sqrt(3.0f))); // Squared to save the extra sqrt() later
		cudaMemcpy(d_newdistance, d_distance, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);

		dim3 grid = dim3((dims.x - 2 + 31) / 32, (dims.y - 2 + 3) / 4, dims.z - 2);
		dim3 TpB = dim3(32, 4, 1);

		for (int i = 0; i < maxiterations; i++)
		{
			DistanceMapKernel <<<grid, TpB>>> (d_distance, d_newdistance, d_upstreamneighbor, dims);

			tfloat* d_temp = d_newdistance;
			d_newdistance = d_distance;
			d_distance = d_temp;
		}

		//DistanceMapFinalizeKernel <<<grid, TpB>>> (d_distance, d_newdistance, d_upstreamneighbor, dims);
		//d_Sqrt(d_distance, d_distance, Elements(dims));

		cudaMemcpy(d_output, d_distance, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);

		cudaFree(d_upstreamneighbor);
		cudaFree(d_newdistance);
		cudaFree(d_distance);
	}

	void d_DistanceMapExact(tfloat* d_input, tfloat* d_output, int3 dims, int maxdistance)
	{
		dim3 grid = dim3((dims.x + 31) / 32, (dims.y + 3) / 4, 1);
		dim3 TpB = dim3(32, 4, 1);

		if (dims.z > 1)
			for (int idz = 0; idz < dims.z; idz++)
				DistanceMapExact3DKernel << <grid, TpB >> > (d_input, d_output, dims, maxdistance, idz);
		else
			DistanceMapExact2DKernel << <grid, TpB >> > (d_input, d_output, dims, maxdistance);
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void DistanceMapKernel(tfloat* d_olddistance, tfloat* d_newdistance, int* d_upstreamneighbor, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (idx > dims.x - 2)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y + 1;
		if (idy > dims.y - 2)
			return;
		int idz = blockIdx.z + 1;

		tfloat bestdist = d_olddistance[(idz * dims.y + idy) * dims.x + idx];
		int closestneighbor = d_upstreamneighbor[(idz * dims.y + idy) * dims.x + idx];

		for (int z = -1; z <= 1; z++)
		{
			int zz = idz + z;
			for (int y = -1; y <= 1; y++)
			{
				int yy = idy + y;
				for (int x = -1; x <= 1; x++)
				{
					int xx = idx + x;
					int neighbor = (zz * dims.y + yy) * dims.x + xx;

					tfloat dist = d_olddistance[neighbor] + sqrtf(abs(x) + abs(y) + abs(z));	// Everything is in squared distances
					if (dist < bestdist)
					{
						bestdist = dist;
						closestneighbor = neighbor;
					}
				}
			}
		}

		d_newdistance[(idz * dims.y + idy) * dims.x + idx] = bestdist;
		d_upstreamneighbor[(idz * dims.y + idy) * dims.x + idx] = closestneighbor;
	}

	__global__ void DistanceMapFinalizeKernel(tfloat* d_olddistance, tfloat* d_newdistance, int* d_upstreamneighbor, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (idx > dims.x - 2)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y + 1;
		if (idy > dims.y - 2)
			return;
		int idz = blockIdx.z + 1;
		
		int furthestneighbor = d_upstreamneighbor[(idz * dims.y + idy) * dims.x + idx];

		if (furthestneighbor >= 0)
		{
			while (true)
			{
				int testneighbor = d_upstreamneighbor[furthestneighbor];
				if (testneighbor >= 0)
					furthestneighbor = testneighbor;
				else
					break;
			}

			int nz = furthestneighbor / (dims.y * dims.x);
			int ny = furthestneighbor % (dims.y * dims.x) / dims.x;
			int nx = furthestneighbor % dims.x;

			int3 diff = make_int3(nx - idx, ny - idy, nz - idz);
			tfloat distance = sqrt((tfloat)dotp(diff, diff)) + sqrt(d_olddistance[furthestneighbor]);

			d_newdistance[(idz * dims.y + idy) * dims.x + idx] = distance;
		}
		else
		{
			d_newdistance[(idz * dims.y + idy) * dims.x + idx] = sqrt(d_olddistance[(idz * dims.y + idy) * dims.x + idx]);
		}
	}

	__global__ void DistanceMapExact3DKernel(tfloat* d_input, tfloat* d_output, int3 dims, int maxdistance, int idz)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y)
			return;
		//int idz = blockIdx.z;

		int startx = tmax(0, idx - maxdistance);
		int endx = tmin(idx + maxdistance, dims.x - 1);
		int starty = tmax(0, idy - maxdistance);
		int endy = tmin(idy + maxdistance, dims.y - 1);
		int startz = tmax(0, idz - maxdistance);
		int endz = tmin(idz + maxdistance, dims.z - 1);

		int mindistance2 = maxdistance * maxdistance;
		if (d_input[(idz * dims.y + idy) * dims.x + idx] == 1)
		{
			mindistance2 = 0;
		}
		else
		{
			for (int z = startz; z <= endz; z++)
			{
				int zz = z - idz;
				zz *= zz;

				for (int y = starty; y <= endy; y++)
				{
					int yy = y - idy;
					yy *= yy;

					for (int x = startx; x <= endx; x++)
					{
						int xx = idx - x;
						xx *= xx;

						float curdistance2 = xx + yy + zz;
						if (curdistance2 > mindistance2)
							continue;

						if (d_input[(z * dims.y + y) * dims.x + x] == 1)
						{
							mindistance2 = tmin(curdistance2, mindistance2);
							if (mindistance2 <= 1)
								break;
						}
					}
					if (mindistance2 <= 1)
						break;
				}
				if (mindistance2 <= 1)
					break;
			}
		}

		d_output[(idz * dims.y + idy) * dims.x + idx] = sqrt((float)mindistance2);
	}

	__global__ void DistanceMapExact2DKernel(tfloat* d_input, tfloat* d_output, int3 dims, int maxdistance)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y)
			return;

		int startx = tmax(0, idx - maxdistance);
		int endx = tmin(idx + maxdistance, dims.x - 1);
		int starty = tmax(0, idy - maxdistance);
		int endy = tmin(idy + maxdistance, dims.y - 1);

		int mindistance2 = maxdistance * maxdistance;
		if (d_input[idy * dims.x + idx] == 1)
		{
			mindistance2 = 0;
		}
		else
		{
			for (int y = starty; y <= endy; y++)
			{
				int yy = y - idy;
				yy *= yy;

				for (int x = startx; x <= endx; x++)
				{
					int xx = idx - x;
					xx *= xx;

					float curdistance2 = xx + yy;
					if (curdistance2 > mindistance2)
						continue;

					if (d_input[y * dims.x + x] == 1)
					{
						mindistance2 = tmin(curdistance2, mindistance2);
						if (mindistance2 <= 1)
							break;
					}
				}
				if (mindistance2 <= 1)
					break;
			}
		}

		d_output[idy * dims.x + idx] = sqrt((float)mindistance2);
	}
}