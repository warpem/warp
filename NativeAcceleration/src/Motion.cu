#include "include/Functions.h"
#include <cuda_runtime.h>
using namespace gtom;

#define SHIFT_THREADS 128

__global__ void ShiftGetAverageKernel(float2* d_individual, float2* d_average, float2* d_shiftfactors, float2* d_shifts, uint length, uint probelength, uint nspectra, uint nframes);
__global__ void ShiftGetDiffKernel(float2* d_individual, float2* d_average, float2* d_shiftfactors, uint length, uint probelength, float2* d_shifts, float* d_diff);
__global__ void ShiftGetGradKernel(float2* d_individual, float2* d_average, float2* d_shiftfactors, uint length, uint probelength, float2* d_shifts, float2* d_grad);

/*

Supplied with a stack of frames, extraction positions for sub-regions, and a mask of relevant pixels in Fspace, 
this method extracts portions of each frame, computes the FT, and returns the requested FT components.

*/

__declspec(dllexport) void CreateShift(float* d_frame,
										int2 dimsframe,
										int nframes,
										int3* h_origins,
										int norigins,
										int2 dimsregion,
										size_t* h_mask,
										uint masklength,
										float2* d_outputall,
										float* d_sigma)
{
	int2 dimsunpadded = toInt2(dimsregion.x / 1, dimsregion.y / 1);

	int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, norigins * sizeof(int3));
	size_t* d_mask = (size_t*)CudaMallocFromHostArray(h_mask, masklength * sizeof(size_t));
	tfloat* d_framenorm;
	cudaMalloc((void**)&d_framenorm, Elements2(dimsframe) * sizeof(float));
	tfloat* d_temp;
	cudaMalloc((void**)&d_temp, norigins * ElementsFFT2(dimsregion) * sizeof(tcomplex));
	tcomplex* d_tempft;
	cudaMalloc((void**)&d_tempft, norigins * ElementsFFT2(dimsregion) * sizeof(tcomplex));
	tcomplex* d_dense;
	cudaMalloc((void**)&d_dense, norigins * masklength * sizeof(tcomplex));
	
	tfloat* d_reduced = CudaMallocValueFilled(norigins * Elements2(dimsregion), 0.0f);
	tfloat* d_amps;
	cudaMalloc((void**)&d_amps, norigins * ElementsFFT2(dimsregion) * sizeof(float));
	tfloat* d_ampsreduced = CudaMallocValueFilled(norigins * ElementsFFT2(dimsregion), 0.0f);

	cufftHandle planforw = d_FFTR2CGetPlan(2, toInt3(dimsregion), norigins);

	for (size_t z = 0; z < nframes; z++)
	{
		d_Norm(d_frame + Elements2(dimsframe) * z, d_framenorm, Elements2(dimsframe), (float*)NULL, gtom::T_NORM_MEAN01STD, 0.0f);

		d_ExtractMany(d_framenorm, d_temp, toInt3(dimsframe), toInt3(dimsregion), d_origins, false, norigins);
		//d_NormMonolithic(d_temp, d_temp, Elements2(dimsregion), T_NORM_MEAN01STD, norigins);
		float radius = dimsregion.x / 4.0f;
		d_SphereMask(d_temp, d_temp, toInt3(dimsregion), &radius, dimsregion.x / 4.0f, NULL, norigins);
		//d_WriteMRC(d_temp, toInt3(dimsregion.x, dimsregion.y, norigins), "d_shifttemp.mrc");

		//d_AddVector(d_reduced, d_temp, d_reduced, norigins * Elements2(dimsregion));

		d_FFTR2C(d_temp, d_tempft, &planforw);
		d_ComplexMultiplyByScalar(d_tempft, d_tempft, norigins * ElementsFFT2(dimsregion), 1.0f / Elements2(dimsregion));

		d_Abs(d_tempft, d_amps, norigins * ElementsFFT2(dimsregion));
		//d_Square(d_amps, d_amps, norigins * ElementsFFT2(dimsregion));
		d_AddVector(d_ampsreduced, d_amps, d_ampsreduced, norigins * ElementsFFT2(dimsregion));

		d_Remap(d_tempft, d_mask, d_outputall + masklength * norigins * z, masklength, ElementsFFT2(dimsregion), make_cuComplex(0.0f, 0.0f), norigins);
		//d_ComplexNormalize(d_dense, d_dense, masklength * norigins);
	}

	//d_WriteMRC(d_reduced, toInt3(dimsregion.x, dimsregion.y, norigins), "d_reduced.mrc");
	//d_WriteMRC(d_ampsreduced, toInt3(dimsregion.x / 2 + 1, dimsregion.y, norigins), "d_ampsreduced.mrc");

	d_MultiplyByScalar(d_ampsreduced, d_ampsreduced, norigins * ElementsFFT2(dimsregion), 1.0f / nframes);
	//d_MultiplyByScalar(d_reduced, d_reduced, norigins * Elements2(dimsregion), 1.0f / nframes);

	//d_ReduceMean(d_reduced, d_temp, Elements2(dimsregion), norigins);
	//d_FFTR2C(d_reduced, d_tempft, &planforw);
	//d_ComplexMultiplyByScalar(d_tempft, d_tempft, norigins * ElementsFFT2(dimsregion), 1.0f / Elements2(dimsregion));
	//d_Abs(d_tempft, d_reduced, norigins * ElementsFFT2(dimsregion));
	//d_Square(d_reduced, d_reduced, norigins * ElementsFFT2(dimsregion));
	//d_WriteMRC(d_reduced, toInt3(dimsregion.x / 2 + 1, dimsregion.y, 1), "d_reducedamps.mrc");

	//d_SubtractVector(d_ampsreduced, d_reduced, d_ampsreduced, norigins * ElementsFFT2(dimsregion), 1);
	d_ReduceMean(d_ampsreduced, d_sigma, ElementsFFT2(dimsregion), norigins);

	cufftDestroy(planforw);

	cudaFree(d_ampsreduced);
	cudaFree(d_amps);
	cudaFree(d_reduced);
	cudaFree(d_dense);
	cudaFree(d_tempft);
	cudaFree(d_temp);
	cudaFree(d_framenorm);
	cudaFree(d_mask);
	cudaFree(d_origins);
}

__declspec(dllexport) void ShiftGetAverage(float2* d_individual, 
											float2* d_average, 
											float2* d_shiftfactors,
											uint length,  
											uint probelength,
											float2* d_shifts, 
											uint npositions, 
											uint nframes)
{
	float2* d_shiftshalf;
	cudaMalloc((void**)&d_shiftshalf, npositions * nframes * sizeof(float2));
	d_ConvertTFloatTo((float*)d_shifts, (float*)d_shiftshalf, npositions * nframes * 2);
	
	int TpB = tmin(SHIFT_THREADS, NextMultipleOf(length, 32));
	dim3 grid = dim3((length + TpB - 1) / TpB, npositions, 1);
	ShiftGetAverageKernel <<<grid, TpB>>> (d_individual, d_average, d_shiftfactors, d_shiftshalf, length, probelength, npositions, nframes);

	/*float2* d_averagef;
	cudaMalloc((void**)&d_averagef, length * npositions * sizeof(float2));
	d_ConvertToTFloat((half*)d_average, (float*)d_averagef, npositions * length * 2);
	float2* h_averagef = (float2*)MallocFromDeviceArray(d_averagef, length * npositions * sizeof(float2));
	cudaFree(d_averagef);
	
	float2* d_phasef;
	cudaMalloc((void**)&d_phasef, length * npositions * nframes * sizeof(float2));
	d_ConvertToTFloat((half*)d_phase, (float*)d_phasef, npositions * nframes * length * 2);
	float2* h_phasef = (float2*)MallocFromDeviceArray(d_phasef, length * npositions * nframes * sizeof(float2));
	cudaFree(d_phasef);

	free(h_averagef);
	free(h_phasef);*/

	cudaFree(d_shiftshalf);
}

__global__ void ShiftGetAverageKernel(float2* d_individual, float2* d_average, float2* d_shiftfactors, float2* d_shifts, uint length, uint probelength, uint npositions, uint nframes)
{
	d_individual += blockIdx.y * length;
	d_average += blockIdx.y * probelength;
	d_shifts += blockIdx.y;

	__shared__ float2 s_shifts[256];	// 256 frames should be enough for everyone
	for (uint i = threadIdx.x; i < nframes; i += blockDim.x)
		s_shifts[i] = d_shifts[npositions * i];
	__syncthreads();


	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; 
		 id < probelength; 
		 id += gridDim.x * blockDim.x)
	{
		float2 shiftfactors = d_shiftfactors[id];
		float2 sum = make_float2(0.0f, 0.0f);

		for (uint frame = 0; frame < nframes; frame++)
		{
			float2 shift = s_shifts[frame];
			float phase = shiftfactors.x * shift.x + shiftfactors.y * shift.y;
			float2 change = make_float2(__cosf(-phase), __sinf(-phase));

			float2 value = d_individual[length * npositions * frame + id];
			value = cuCmulf(value, change);

			sum += value;
		}
		
		sum /= nframes;

		d_average[id] = sum;
	}
}

__declspec(dllexport) void ShiftGetDiff(float2* d_individual, 
											float2* d_average, 
											float2* d_shiftfactors, 
											uint length, 
											uint probelength,
											float2* d_shifts,
											float* h_diff, 
											uint npositions, 
											uint nframes)
{
	int TpB = tmin(SHIFT_THREADS, NextMultipleOf(probelength, 32));
	dim3 grid = dim3(npositions, nframes, 1);

	float* d_diff;
	cudaMalloc((void**)&d_diff, npositions * nframes * grid.x * sizeof(float));
	float* d_diffreduced;
	cudaMalloc((void**)&d_diffreduced, npositions * nframes * sizeof(float));

	ShiftGetDiffKernel <<<grid, TpB>>> (d_individual, d_average, d_shiftfactors, length, probelength, d_shifts, d_diff);

	//d_SumMonolithic(d_diff, d_diffreduced, grid.x, npositions * nframes);
	cudaMemcpy(h_diff, d_diff, npositions * nframes * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_diffreduced);
	cudaFree(d_diff);
}

__global__ void ShiftGetDiffKernel(float2* d_individual, float2* d_average, float2* d_shiftfactors, uint length, uint probelength, float2* d_shifts, float* d_diff)
{
	__shared__ float s_diff[SHIFT_THREADS];
	s_diff[threadIdx.x] = 0.0f;

	const uint iorigin = blockIdx.x;
	const uint norigins = gridDim.x;
	const uint iframe = blockIdx.y;
	const uint nframes = gridDim.y;

	__shared__ float2 s_shifts[256];
	for (uint f = threadIdx.x; f < nframes; f += blockDim.x)
		s_shifts[f] = d_shifts[f * norigins + iorigin];
	__syncthreads();

	uint specid = blockIdx.y * gridDim.x + blockIdx.x;
	d_individual += specid * length;
	d_average += blockIdx.x * probelength;

	float2 shift = s_shifts[iframe];
	float diffsum = 0.0f;

	for (uint id = threadIdx.x; id < probelength; id += blockDim.x)
	{
		float2 value = d_individual[id];
		float2 average = d_average[id];

		float2 shiftfactors = d_shiftfactors[id];
		/*float2 average = make_float2(0);
		for (uint f = 0; f < nframes; f++)
		{
			if (f == iframe)
				continue;

			float2 frameshift = s_shifts[f];
			float framephase = shiftfactors.x * frameshift.x + shiftfactors.y * frameshift.y;
			float2 framechange = make_float2(__cosf(-framephase), __sinf(-framephase));
			float2 framevalue = d_individual[(f * norigins + iorigin) * length + id];
			framevalue = cuCmulf(framevalue, framechange);
			average += framevalue;
		}
		average /= (float)(nframes - 1);*/


		float phase = shiftfactors.x * shift.x + shiftfactors.y * shift.y;
		float2 change = make_float2(__cosf(-phase), __sinf(-phase));

		value = cuCmulf(value, change);

		float2 diff = value - average;
		diffsum += dotp2(diff, diff);
	}

	s_diff[threadIdx.x] = diffsum;
	__syncthreads();

	for (uint lim = 64; lim > 1; lim >>= 1)
	{
		if (threadIdx.x < lim)
		{
			diffsum += s_diff[threadIdx.x + lim];
			s_diff[threadIdx.x] = diffsum;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		diffsum += s_diff[1];

		d_diff[specid] = diffsum;
	}
}

__declspec(dllexport) void ShiftGetGrad(float2* d_individual, 
										float2* d_average, 
										float2* d_shiftfactors, 
										uint length, 
										uint probelength,
										float2* d_shifts,
										float2* h_grad, 
										uint npositions, 
										uint nframes)
{
	int TpB = tmin(SHIFT_THREADS, NextMultipleOf(probelength, 32));
	dim3 grid = dim3(npositions, nframes, 1);

	float2* d_grad;
	cudaMalloc((void**)&d_grad, npositions * nframes * grid.x * sizeof(float2));
	float2* d_gradreduced;
	cudaMalloc((void**)&d_gradreduced, npositions * nframes * sizeof(float2));

	ShiftGetGradKernel <<<grid, TpB>>> (d_individual, d_average, d_shiftfactors, length, probelength, d_shifts, d_grad);

	float2* h_grad2 = (float2*)MallocFromDeviceArray(d_grad, npositions * nframes * grid.x * sizeof(float2));
	free(h_grad2);

	//d_SumMonolithic(d_grad, d_gradreduced, grid.x, npositions * nframes);
	cudaMemcpy(h_grad, d_grad, npositions * nframes * sizeof(float2), cudaMemcpyDeviceToHost);
	
	cudaFree(d_gradreduced);
	cudaFree(d_grad);
}

__global__ void ShiftGetGradKernel(float2* d_individual, 
									float2* d_average, 
									float2* d_shiftfactors, 
									uint length, 
									uint probelength, 
									float2* d_shifts, 
									float2* d_grad)
{
	__shared__ float2 s_grad[SHIFT_THREADS];

	const uint iorigin = blockIdx.x;
	const uint norigins = gridDim.x;
	const uint iframe = blockIdx.y;
	const uint nframes = gridDim.y;

	__shared__ float2 s_shifts[256];
	for (uint f = threadIdx.x; f < nframes; f += blockDim.x)
		s_shifts[f] = d_shifts[f * norigins + iorigin];
	__syncthreads();

	uint specid = blockIdx.y * gridDim.x + blockIdx.x;
	d_individual += specid * length;
	d_average += blockIdx.x * probelength;

	float2 shift = s_shifts[iframe];
	float2 gradsum = make_float2(0.0f, 0.0f);

	for (uint id = threadIdx.x; id < probelength; id += blockDim.x)
	{
		float2 value = d_individual[id];
		float2 average = d_average[id];

		float2 shiftfactors = d_shiftfactors[id];
		/*float2 average = make_float2(0);
		for (uint f = 0; f < nframes; f++)
		{
			if (f == iframe)
				continue;

			float2 frameshift = s_shifts[f];
			float framephase = shiftfactors.x * frameshift.x + shiftfactors.y * frameshift.y;
			float2 framechange = make_float2(__cosf(-framephase), __sinf(-framephase));
			float2 framevalue = d_individual[(f * norigins + iorigin) * length + id];
			framevalue = cuCmulf(framevalue, framechange);
			average += framevalue;
		}
		average /= (float)(nframes - 1);*/

		{
			float phase = shiftfactors.x * (shift.x + 1e-4f) + shiftfactors.y * shift.y;
			float phasesin = 0, phasecos = 0;
			__sincosf(-phase, &phasesin, &phasecos);
			float2 change = make_float2(phasecos, phasesin);
			float2 shiftedPluxX = cmul(value, change);		
			float2 diffPlusX = shiftedPluxX - average;

			phase = shiftfactors.x * (shift.x - 1e-4f) + shiftfactors.y * shift.y;
			__sincosf(-phase, &phasesin, &phasecos);
			change = make_float2(phasecos, phasesin);
			float2 shiftedMinusX = cmul(value, change);		
			float2 diffMinusX = shiftedMinusX - average;

			gradsum.x += dotp2(diffPlusX, diffPlusX) - dotp2(diffMinusX, diffMinusX);
		}

		{
			float phase = shiftfactors.x * shift.x + shiftfactors.y * (shift.y + 1e-4f);
			float phasesin = 0, phasecos = 0;
			__sincosf(-phase, &phasesin, &phasecos);
			float2 change = make_float2(phasecos, phasesin);
			float2 shiftedPluxY = cmul(value, change);		
			float2 diffPlusY = shiftedPluxY - average;

			phase = shiftfactors.x * shift.x + shiftfactors.y * (shift.y - 1e-4f);
			__sincosf(-phase, &phasesin, &phasecos);
			change = make_float2(phasecos, phasesin);
			float2 shiftedMinusY = cmul(value, change);		
			float2 diffMinusY = shiftedMinusY - average;

			gradsum.y += dotp2(diffPlusY, diffPlusY) - dotp2(diffMinusY, diffMinusY);
		}
	}

	s_grad[threadIdx.x] = gradsum;
	__syncthreads();

	for (uint lim = 64; lim > 1; lim >>= 1)
	{
		if (threadIdx.x < lim)
		{
			gradsum += s_grad[threadIdx.x + lim];
			s_grad[threadIdx.x] = gradsum;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		gradsum += s_grad[1];

		d_grad[specid] = gradsum;
	}
}

__declspec(dllexport) void CreateMotionBlur(float* d_output, int3 dims, float* h_shifts, uint nshifts, uint batch)
{
    d_MotionBlur(d_output, dims, (float3*)h_shifts, nshifts, false, batch);
}