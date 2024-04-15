#include "include/Functions.h"
using namespace gtom;
#include "gtom/include/CubicInterp.cuh"

__global__ void CubicGPUInterpIrregularKernel(cudaTex t_input, int3 dimsgrid, float3* d_positions, int npositions, float* d_output);

__declspec(dllexport) void __stdcall CubicGPUInterpIrregular(cudaTex t_input, int3 dimsgrid, float3* h_positions, int npositions, float* h_output)
{
	float3* d_positions = (float3*)CudaMallocFromHostArray(h_positions, npositions * sizeof(float3));
	float* d_output;
	cudaMalloc((void**)&d_output, npositions * sizeof(float3));

	int TpB = 128;
	dim3 grid = dim3(tmin((npositions + TpB - 1) / TpB, 65535), 1, 1);
	CubicGPUInterpIrregularKernel <<<grid, TpB>>> (t_input, dimsgrid, d_positions, npositions, d_output);

	cudaMemcpy(h_output, d_output, npositions * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_positions);
}

__global__ void CubicGPUInterpIrregularKernel(cudaTex t_input, int3 dimsgrid, float3* d_positions, int npositions, float* d_output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < npositions; i += gridDim.x * blockDim.x)
	{
        d_output[i] = cubicTex1DSimple<float>(t_input, d_positions[i].x * dimsgrid.x);
	}
}