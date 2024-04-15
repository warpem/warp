#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	template<bool amplitudesquared, bool ignorefirstpeak> __global__ void CTFSimulateKernel(tfloat* d_output, float2* d_addresses, float* d_gammacorrection, int n, CTFParamsLean* d_p);
	template<bool amplitudesquared, bool ignorefirstpeak> __global__ void CTFSimulateKernel(half* d_output, half2* d_addresses, int n, CTFParamsLean* d_p);
	__global__ void CTFSimulateComplexKernel(float2* d_output, float2* d_addresses, float* d_gammacorrection, int n, CTFParamsLean* d_p, bool reverse);
	__global__ void CTFSimulateEwaldWeightsKernel(tfloat* d_output, float2* d_addresses, float* d_gammacorrection, float particlediameter, int n, CTFParamsLean* d_p);


	/////////////////////////////////////////////
	//Simulate the CTF function at given points//
	/////////////////////////////////////////////

	void d_CTFSimulate(CTFParams* h_params, float2* d_addresses, float* d_gammacorrection, tfloat* d_output, uint n, bool amplitudesquared, bool ignorefirstpeak, int batch)
	{
		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));

		//#pragma omp parallel for
		for (int i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i], toInt3(1, 1, 1));	// Sidelength is already included in d_addresses

		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
		free(h_lean);

		int TpB = tmin(128, NextMultipleOf(n, 32));
		dim3 grid = dim3(tmin(batch > 1 ? 16 : 128, (n + TpB - 1) / TpB), batch);
		if (amplitudesquared)
			if (ignorefirstpeak)
				CTFSimulateKernel<true, true> << <grid, TpB >> > (d_output, d_addresses, d_gammacorrection, n, d_lean);
			else
				CTFSimulateKernel<true, false> << <grid, TpB >> > (d_output, d_addresses, d_gammacorrection, n, d_lean);
		else
			if (ignorefirstpeak)
				CTFSimulateKernel<false, true> << <grid, TpB >> > (d_output, d_addresses, d_gammacorrection, n, d_lean);
			else
				CTFSimulateKernel<false, false> << <grid, TpB >> > (d_output, d_addresses, d_gammacorrection, n, d_lean);

		cudaFree(d_lean);
	}

	void d_CTFSimulate(CTFParams* h_params, half2* d_addresses, half* d_output, uint n, bool amplitudesquared, bool ignorefirstpeak, int batch)
	{
		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));

		//#pragma omp parallel for
		for (int i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i], toInt3(1, 1, 1));	// Sidelength and pixelsize are already included in d_addresses

		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
		free(h_lean);

		int TpB = tmin(128, NextMultipleOf(n, 32));
		dim3 grid = dim3(tmin(batch > 1 ? 16 : 128, (n + TpB - 1) / TpB), batch);
		if (amplitudesquared)
			if (ignorefirstpeak)
				CTFSimulateKernel<true, true> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
			else
				CTFSimulateKernel<true, false> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
		else
			if (ignorefirstpeak)
				CTFSimulateKernel<false, true> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
			else
				CTFSimulateKernel<false, false> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);

		cudaFree(d_lean);
	}


	void d_CTFSimulateComplex(CTFParams* h_params, float2* d_addresses, float* d_gammacorrection, float2* d_output, uint n, bool reverse, int batch)
	{
		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));

		//#pragma omp parallel for
		for (int i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i], toInt3(1, 1, 1));	// Sidelength is already included in d_addresses

		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
		free(h_lean);

		int TpB = tmin(128, NextMultipleOf(n, 32));
		dim3 grid = dim3(tmin(batch > 1 ? 16 : 128, (n + TpB - 1) / TpB), batch);
		CTFSimulateComplexKernel << <grid, TpB >> > (d_output, d_addresses, d_gammacorrection, n, d_lean, reverse);

		cudaFree(d_lean);
	}

	void d_CTFSimulateEwaldWeights(CTFParams* h_params, float2* d_addresses, float* d_gammacorrection, float particlediameter, tfloat* d_output, uint n, int batch)
	{
		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));

		//#pragma omp parallel for
		for (int i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i], toInt3(1, 1, 1));	// Sidelength is already included in d_addresses

		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
		free(h_lean);

		int TpB = tmin(128, NextMultipleOf(n, 32));
		dim3 grid = dim3(tmin(batch > 1 ? 16 : 128, (n + TpB - 1) / TpB), batch);
		CTFSimulateEwaldWeightsKernel << <grid, TpB >> > (d_output, d_addresses, d_gammacorrection, particlediameter, n, d_lean);

		cudaFree(d_lean);
	}

	////////////////
	//CUDA kernels//
	////////////////

	template<bool amplitudesquared, bool ignorefirstpeak> __global__ void CTFSimulateKernel(tfloat* d_output, float2* d_addresses, float* d_gammacorrection, int n, CTFParamsLean* d_p)
	{
		CTFParamsLean p = d_p[blockIdx.y];
		d_output += blockIdx.y * n;

		for (uint idx = blockIdx.x * blockDim.x + threadIdx.x;
			 idx < n;
			 idx += gridDim.x * blockDim.x)
		{
			float2 address = d_addresses[idx];
			float angle = address.y;
			float k = address.x;

			float pixelsize = p.pixelsize + p.pixeldelta * __cosf(2.0f * (angle - p.pixelangle));
			k /= pixelsize;

			float gammacorrection = 0;
			if (d_gammacorrection != NULL)
				gammacorrection = d_gammacorrection[idx];

			d_output[idx] = d_GetCTF<amplitudesquared, ignorefirstpeak>(k, angle, gammacorrection, p);
		}
	}

	__global__ void CTFSimulateComplexKernel(float2* d_output, float2* d_addresses, float* d_gammacorrection, int n, CTFParamsLean* d_p, bool reverse)
	{
		CTFParamsLean p = d_p[blockIdx.y];
		d_output += blockIdx.y * n;

		for (uint idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < n;
			idx += gridDim.x * blockDim.x)
		{
			float2 address = d_addresses[idx];
			float angle = address.y;
			float k = address.x;

			float pixelsize = p.pixelsize + p.pixeldelta * __cosf(2.0f * (angle - p.pixelangle));
			k /= pixelsize;

			float gammacorrection = 0;
			if (d_gammacorrection != NULL)
				gammacorrection = d_gammacorrection[idx];

			d_output[idx] = d_GetCTFComplex<true>(k, angle, gammacorrection, p, reverse);
		}
	}

	template<bool amplitudesquared, bool ignorefirstpeak> __global__ void CTFSimulateKernel(half* d_output, half2* d_addresses, int n, CTFParamsLean* d_p)
	{
		CTFParamsLean p = d_p[blockIdx.y];
		d_output += blockIdx.y * n;

		for (uint idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < n;
			idx += gridDim.x * blockDim.x)
		{
			float2 address = __half22float2(d_addresses[idx]);
			float angle = address.y;
			float k = address.x;

			float pixelsize = p.pixelsize + p.pixeldelta * __cosf(2.0f * (angle - p.pixelangle));
			k /= pixelsize;

			d_output[idx] = __float2half(d_GetCTF<amplitudesquared, ignorefirstpeak>(k, angle, 0, p));
		}
	}

	__global__ void CTFSimulateEwaldWeightsKernel(tfloat* d_output, float2* d_addresses, float* d_gammacorrection, float particlediameter, int n, CTFParamsLean* d_p)
	{
		CTFParamsLean p = d_p[blockIdx.y];
		d_output += blockIdx.y * n;

		for (uint idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < n;
			idx += gridDim.x * blockDim.x)
		{
			float2 address = d_addresses[idx];
			float angle = address.y;
			float k = address.x;

			float pixelsize = p.pixelsize + p.pixeldelta * __cosf(2.0f * (angle - p.pixelangle));
			k /= pixelsize;

			float gammacorrection = 0;
			if (d_gammacorrection != NULL)
				gammacorrection = d_gammacorrection[idx];

			float k2 = k * k;
			float k4 = k2 * k2;

			float deltaf = p.defocus + p.defocusdelta * __cosf(2.0f * (angle - p.astigmatismangle));
			float argument = p.K1 * deltaf * k2 + p.K2 * k4 - p.phaseshift - p.K3 + gammacorrection;

			float aux = 2 * abs(deltaf) * p.lambda * k / particlediameter;
			float A = (aux > 1) ? 0 : (2 / PI) * (acos(aux) - aux * __sinf(acos(aux)));

			d_output[idx] = (1 + A * (2 * abs(-__sinf(argument)) - 1)) * 0.5f;
		}
	}
}