#include "include/Functions.h"
using namespace gtom;

__declspec(dllexport) void GetMotionFilter(float* d_output, int3 dims, float3* h_shifts, uint nshifts, uint batch)
{
	tcomplex* d_phases = CudaMallocValueFilled(ElementsFFT2(dims) * nshifts * batch, make_cuComplex(1.0f, 0.0f));
	tcomplex* d_meanphases;
	cudaMalloc((void**)&d_meanphases, ElementsFFT2(dims) * batch * sizeof(tcomplex));
	
	d_Shift(d_phases, d_phases, dims, (tfloat3*)h_shifts, false, nshifts * batch);
	d_ReduceMean(d_phases, d_meanphases, ElementsFFT2(dims), nshifts, batch);
	d_Abs(d_meanphases, d_output, ElementsFFT2(dims) * batch);

	cudaFree(d_meanphases);
	cudaFree(d_phases);
}

__declspec(dllexport) void CorrectMagAnisotropy(float* d_image, int2 dimsimage, float* d_scaled, int2 dimsscaled, float majorpixel, float minorpixel, float majorangle, uint supersample, uint batch)
{
	d_MagAnisotropyCorrect(d_image, dimsimage, d_scaled, dimsscaled, majorpixel, minorpixel, majorangle, supersample, batch);
}

__declspec(dllexport) void WeightedFrameSum(float* d_frames, 
											float* d_ctf, 
											float* d_dose, 
											float* d_outputframes, 
											float* d_outputspectrum, 
											int2 dims, 
											uint nframes, 
											uint batch)
{
	float* d_framespectra;
	cudaMalloc((void**)&d_framespectra, ElementsFFT2(dims) * nframes * batch * sizeof(float));
	d_MultiplyByVector(d_ctf, d_dose, d_framespectra, ElementsFFT2(dims) * nframes * batch);
	float* d_sumspectra;
	cudaMalloc((void**)&d_sumspectra, ElementsFFT2(dims) * batch * sizeof(float));
	d_ReduceAdd(d_framespectra, d_sumspectra, ElementsFFT2(dims), nframes, batch);

	tcomplex* d_framesft;
	cudaMalloc((void**)&d_framesft, ElementsFFT2(dims) * nframes * batch * sizeof(tcomplex));
	tcomplex* d_sumsft;
	cudaMalloc((void**)&d_sumsft, ElementsFFT2(dims) * batch * sizeof(tcomplex));

	long batchsize = tmax(1, (1 << 28) / (ElementsFFT2(dims) * sizeof(tcomplex)));
	for (int b = 0; b < batch; b += batchsize)
	{
		uint curbatch = tmin(batch - b, batchsize);
		d_FFTR2C(d_frames + Elements2(dims) * b, d_framesft + ElementsFFT2(dims) * b, 2, toInt3(dims), curbatch);
	}

	d_ComplexMultiplyByVector(d_framesft, d_framespectra, d_framesft, ElementsFFT2(dims) * nframes * batch);
	d_ReduceAdd(d_framesft, d_sumsft, ElementsFFT2(dims), nframes, batch);

	d_ComplexDivideByVector(d_sumsft, d_sumspectra, d_sumsft, ElementsFFT2(dims) * batch);

	cudaFree(d_framesft);
	cudaFree(d_framesft);
}

__declspec(dllexport) void DoseWeighting(float* d_freq, 
										float* d_output, 
										uint length, 
										float2* h_doserange, 
										float3 nikoconst, 
										float voltagescaling,
										uint batch)
{
    d_DoseFilter(d_freq, d_output, length, h_doserange, nikoconst, voltagescaling, batch);
}

__declspec(dllexport) void NormParticles(float* d_input, float* d_output, int3 dims, uint particleradius, bool flipsign, uint batch)
{
    d_NormBackground(d_input, d_output, dims, particleradius, flipsign, batch);
}