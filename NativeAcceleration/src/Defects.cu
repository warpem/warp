#include "include/Functions.h"
using namespace gtom;

__global__ void CorrectDefectsKernel(float* d_input, float* d_output, int3* d_locations, int* d_neighbors, int ndefects);

__declspec(dllexport) void CorrectDefects(float* d_input, float* d_output, int3* d_locations, int* d_neighbors, int ndefects)
{
	dim3 grid = dim3(tmin(512, ndefects), 1, 1);
	CorrectDefectsKernel <<<(uint)tmin(512, ndefects), (uint)1>>> (d_input, d_output, d_locations, d_neighbors, ndefects);
}

__global__ void CorrectDefectsKernel(float* d_input, float* d_output, int3* d_locations, int* d_neighbors, int ndefects)
{
	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < ndefects; id += blockDim.x * gridDim.x)
	{
		int3 location = d_locations[id];
		float mean = 0;

		for (int i = location.y; i < location.z; i++)
			mean += d_input[d_neighbors[i]];

		d_output[location.x] = mean / (location.z - location.y);
	}
}