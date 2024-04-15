#include "include/Functions.h"
using namespace gtom;

__global__ void DeconvolveCTFKernel(float2* d_inputft, float2* d_outputft, int3 dims, CTFParamsLean params, float strength, float falloff, float highpassnyquist);

__declspec(dllexport) void DeconvolveCTF(float2* d_inputft, float2* d_outputft, int3 dims, CTFParams ctfparams, float strength, float falloff, float highpassnyquist)
{
	CTFParamsLean lean = CTFParamsLean(ctfparams, toInt3(1, 1, 1));

	dim3 TpB = dim3(8, 16, 1);
	dim3 grid = dim3((ElementsFFT1(dims.x) + 7) / 8, (dims.y + 15) / 16, dims.z);
	DeconvolveCTFKernel <<<grid, TpB>>> (d_inputft, d_outputft, dims, lean, strength, falloff, highpassnyquist);
}

__global__ void DeconvolveCTFKernel(float2* d_inputft, float2* d_outputft, int3 dims, CTFParamsLean params, float strength, float falloff, float highpassnyquist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z;

	if (idx >= ElementsFFT1(dims.x) || idy >= dims.y)
		return;

	int x = idx;
	int y = idy > dims.y / 2 ? idy - dims.y : idy;
	int z = idz > dims.z / 2 ? idz - dims.z : idz;

	float xx = x / (float)dims.x;
	float yy = y / (float)dims.y;
	float zz = z / (float)dims.z;

	float r = sqrt(xx * xx + yy * yy + zz * zz);

	float ctfval = d_GetCTF<false, false>(r / params.pixelsize, atan2(yy, xx), 0, params);

	float highpass = tmin(1, r * 1 / highpassnyquist) * PI;
	highpass = 1 - (cos(highpass) * 0.5f + 0.5f);

	float snr = exp(-r * 2 * falloff * 100 / params.pixelsize) * pow(10.0f, 3 * strength) * highpass;

	float wiener = ctfval / (ctfval * ctfval + 1 / tmax(1e-15f, snr));

	d_outputft[(idz * dims.y + idy) * ElementsFFT1(dims.x) + idx] = d_inputft[(idz * dims.y + idy) * ElementsFFT1(dims.x) + idx] * wiener;
}