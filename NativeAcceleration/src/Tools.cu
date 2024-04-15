#include "include/Functions.h"
#include "gtom/include/CubicInterp.cuh"
#include <cublas_v2.h>
using namespace gtom;

__declspec(dllexport) void FFT(float* d_input, float2* d_output, int3 dims, uint batch, cufftHandle plan)
{
	if (plan <= 0)
		d_FFTR2C(d_input, d_output, DimensionCount(dims), dims, batch);
	else
		d_FFTR2C(d_input, d_output, &plan);
}

__declspec(dllexport) void IFFT(float2* d_input, float* d_output, int3 dims, uint batch, cufftHandle plan, bool normalize)
{
	if (plan <= 0)
		d_IFFTC2R(d_input, d_output, DimensionCount(dims), dims, batch, normalize);
	else if (!normalize)
		d_IFFTC2R(d_input, d_output, &plan);
	else
		d_IFFTC2R(d_input, d_output, &plan, dims, batch);
}

__declspec(dllexport) void Pad(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch)
{
    d_Pad(d_input, d_output, olddims, newdims, T_PAD_VALUE, 0.0f, batch);
}

__declspec(dllexport) void PadClamped(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch)
{
	d_Pad(d_input, d_output, olddims, newdims, T_PAD_CLAMP, 0.0f, batch);
}

__declspec(dllexport) void PadFT(float2* d_input, float2* d_output, int3 olddims, int3 newdims, uint batch)
{
    d_FFTPad(d_input, d_output, olddims, newdims, batch);
}

__declspec(dllexport) void PadFTRealValued(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch)
{
	d_FFTPad(d_input, d_output, olddims, newdims, batch);
}

__declspec(dllexport) void PadFTFull(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch)
{
	d_FFTFullPad(d_input, d_output, olddims, newdims, batch);
}

__declspec(dllexport) void CropFT(float2* d_input, float2* d_output, int3 olddims, int3 newdims, uint batch)
{
    d_FFTCrop(d_input, d_output, olddims, newdims, batch);
}

__declspec(dllexport) void CropFTRealValued(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch)
{
	d_FFTCrop(d_input, d_output, olddims, newdims, batch);
}

__declspec(dllexport) void CropFTFull(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch)
{
	d_FFTFullCrop(d_input, d_output, olddims, newdims, batch);
}

__declspec(dllexport) void RemapToFTComplex(float2* d_input, float2* d_output, int3 dims, uint batch)
{
    d_RemapHalfFFT2Half(d_input, d_output, dims, batch);
}

__declspec(dllexport) void RemapToFTFloat(float* d_input, float* d_output, int3 dims, uint batch)
{
    d_RemapHalfFFT2Half(d_input, d_output, dims, batch);
}

__declspec(dllexport) void RemapFromFTComplex(float2* d_input, float2* d_output, int3 dims, uint batch)
{
    d_RemapHalf2HalfFFT(d_input, d_output, dims, batch);
}

__declspec(dllexport) void RemapFromFTFloat(float* d_input, float* d_output, int3 dims, uint batch)
{
    d_RemapHalf2HalfFFT(d_input, d_output, dims, batch);
}

__declspec(dllexport) void RemapFullToFTFloat(float* d_input, float* d_output, int3 dims, uint batch)
{
    d_RemapFullFFT2Full(d_input, d_output, dims, batch);
}

__declspec(dllexport) void RemapFullFromFTFloat(float* d_input, float* d_output, int3 dims, uint batch)
{
    d_RemapFull2FullFFT(d_input, d_output, dims, batch);
}

__declspec(dllexport) void Extract(float* d_input, float* d_output, int3 dims, int3 dimsregion, int3* h_origins, bool zeropad, uint batch)
{
	int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, batch * sizeof(int3));

	d_ExtractMany(d_input, d_output, dims, dimsregion, d_origins, zeropad, batch);

	cudaFree(d_origins);
}

__declspec(dllexport) void ExtractMultisource(void** h_inputs, float* d_output, int3 dims, int3 dimsregion, int3* h_origins, int nsources, uint batch)
{
	int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, batch * sizeof(int3));
	float** d_inputs = (float**)CudaMallocFromHostArray(h_inputs, nsources * sizeof(float*));

	d_ExtractManyMultisource(d_inputs, d_output, dims, dimsregion, d_origins, nsources, batch);

	cudaFree(d_inputs);
	cudaFree(d_origins);
}

__declspec(dllexport) void ExtractHalf(half* d_input, half* d_output, int3 dims, int3 dimsregion, int3* h_origins, uint batch)
{
	int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, batch * sizeof(int3));

	d_ExtractMany(d_input, d_output, dims, dimsregion, d_origins, false, batch);

	cudaFree(d_origins);
}

__declspec(dllexport) void ReduceMean(float* d_input, float* d_output, uint vectorlength, uint nvectors, uint batch)
{
	d_ReduceMean(d_input, d_output, vectorlength, nvectors, batch);
}

__declspec(dllexport) void ReduceMeanHalf(half* d_input, half* d_output, uint vectorlength, uint nvectors, uint batch)
{
	d_ReduceMean(d_input, d_output, vectorlength, nvectors, batch);
}

__declspec(dllexport) void ReduceAdd(float* d_input, float* d_output, uint vectorlength, uint nvectors, uint batch)
{
	d_ReduceAdd(d_input, d_output, vectorlength, nvectors, batch);
}

__declspec(dllexport) void ReduceMax(float* d_input, float* d_output, uint vectorlength, uint nvectors, uint batch)
{
	d_ReduceMax(d_input, d_output, vectorlength, nvectors, batch);
}

__declspec(dllexport) void Normalize(float* d_ps, float* d_output, uint length, uint batch)
{
	d_NormMonolithic(d_ps, d_output, length, T_NORM_MEAN01STD, batch);
}

__declspec(dllexport) void NormalizeMasked(float* d_ps, float* d_output, float* d_mask, uint length, uint batch)
{
	d_NormMonolithic(d_ps, d_output, length, d_mask, T_NORM_MEAN01STD, batch);
}

__declspec(dllexport) void SphereMask(float* d_input, float* d_output, int3 dims, float radius, float sigma, bool decentered, uint batch)
{
	d_SphereMask(d_input, d_output, dims, &radius, sigma, NULL, decentered, batch);
}

__declspec(dllexport) void CreateCTF(float* d_output, float2* d_coords, float* d_gammacorrection, uint length, CTFParams* h_params, bool amplitudesquared, uint batch)
{
	d_CTFSimulate(h_params, d_coords, d_gammacorrection, d_output, length, amplitudesquared, false, batch);
}

__declspec(dllexport) void CreateCTFComplex(float* d_output, float2* d_coords, float* d_gammacorrection, uint length, CTFParams* h_params, bool reverse, uint batch)
{
	d_CTFSimulateComplex(h_params, d_coords, d_gammacorrection, (float2*)d_output, length, reverse, batch);
}

__declspec(dllexport) void CreateCTFEwaldWeights(float* d_output, float2* d_coords, float* d_gammacorrection, float particlediameter, uint length, CTFParams* h_params, uint batch)
{
	d_CTFSimulateEwaldWeights(h_params, d_coords, d_gammacorrection, particlediameter, d_output, length, batch);
}

__declspec(dllexport) void Resize(float* d_input, int3 dimsinput, float* d_output, int3 dimsoutput, uint batch)
{
	d_Scale(d_input, d_output, dimsinput, dimsoutput, T_INTERP_FOURIER, NULL, NULL, batch);
}

__declspec(dllexport) void ShiftStack(float* d_input, float* d_output, int3 dims, float* h_shifts, uint batch)
{
	d_Shift(d_input, d_output, dims, (tfloat3*)h_shifts, NULL, NULL, NULL, batch);
}

__declspec(dllexport) void ShiftStackFT(float* d_input, float* d_output, int3 dims, float* h_shifts, uint batch)
{
	d_Shift((tcomplex*)d_input, (tcomplex*)d_output, dims, (tfloat3*)h_shifts, false, batch);
}

__declspec(dllexport) void ShiftStackMassive(float* d_input, float* d_output, int3 dims, float* h_shifts, uint batch)
{
	cufftHandle planforw = d_FFTR2CGetPlan(DimensionCount(dims), dims);
	cufftHandle planback = d_IFFTC2RGetPlan(DimensionCount(dims), dims);
	float2* d_intermediate;
	cudaMalloc((void**)&d_intermediate, ElementsFFT(dims) * sizeof(float2));

	for (int b = 0; b < batch; b++)
		d_Shift(d_input + Elements(dims) * b, d_output + Elements(dims) * b, dims, (tfloat3*)h_shifts + b, &planforw, &planback, d_intermediate);

	cufftDestroy(planforw);
	cufftDestroy(planback);
	cudaFree(d_intermediate);
}

__declspec(dllexport) void Cart2Polar(float* d_input, float* d_output, int2 dims, uint innerradius, uint exclusiveouterradius, uint batch)
{
	d_Cart2Polar(d_input, d_output, dims, T_INTERP_LINEAR, innerradius, exclusiveouterradius, batch);
}

__declspec(dllexport) void Cart2PolarFFT(float* d_input, float* d_output, int2 dims, uint innerradius, uint exclusiveouterradius, uint batch)
{
	d_Cart2PolarFFT(d_input, d_output, dims, T_INTERP_LINEAR, innerradius, exclusiveouterradius, batch);
}

__declspec(dllexport) void Xray(float* d_input, float* d_output, float ndevs, int2 dims, uint batch)
{
    d_Xray(d_input, d_output, toInt3(dims), ndevs, 5, batch);
}

// Arithmetics:

__declspec(dllexport) void Sum(float* d_input, float* d_output, uint length, uint batch)
{
    d_SumMonolithic(d_input, d_output, length, batch);
}

__declspec(dllexport) void Abs(float* d_input, float* d_output, size_t length)
{
    d_Abs(d_input, d_output, length);
}

__declspec(dllexport) void Amplitudes(float2* d_input, float* d_output, size_t length)
{
    d_Abs(d_input, d_output, length);
}

__declspec(dllexport) void Sign(float* d_input, float* d_output, size_t length)
{
    d_Sign(d_input, d_output, length);
}

__declspec(dllexport) void Sqrt(float* d_input, float* d_output, size_t length)
{
	d_Sqrt(d_input, d_output, length);
}

__declspec(dllexport) void Cos(float* d_input, float* d_output, size_t length)
{
	d_Cos(d_input, d_output, length);
}

__declspec(dllexport) void Sin(float* d_input, float* d_output, size_t length)
{
	d_Sin(d_input, d_output, length);
}

__declspec(dllexport) void AddToSlices(float* d_input, float* d_summands, float* d_output, size_t sliceelements, uint slices)
{
	d_AddVector(d_input, d_summands, d_output, sliceelements, slices);
}

__declspec(dllexport) void AddScalar(float* d_input, float summand, float* d_output, size_t elements)
{
	d_AddScalar(d_input, d_output, elements, summand);
}

__declspec(dllexport) void SubtractFromSlices(float* d_input, float* d_subtrahends, float* d_output, size_t sliceelements, uint slices)
{
	d_SubtractVector(d_input, d_subtrahends, d_output, sliceelements, slices);
}

__declspec(dllexport) void MultiplySlices(float* d_input, float* d_multiplicators, float* d_output, size_t sliceelements, uint slices)
{
	d_MultiplyByVector(d_input, d_multiplicators, d_output, sliceelements, slices);
}

__declspec(dllexport) void DivideSlices(float* d_input, float* d_divisors, float* d_output, size_t sliceelements, uint slices)
{
	d_DivideSafeByVector(d_input, d_divisors, d_output, sliceelements, slices);
}

__declspec(dllexport) void AddToSlicesHalf(half* d_input, half* d_summands, half* d_output, size_t sliceelements, uint slices)
{
	d_AddVector(d_input, d_summands, d_output, sliceelements, slices);
}

__declspec(dllexport) void SubtractFromSlicesHalf(half* d_input, half* d_subtrahends, half* d_output, size_t sliceelements, uint slices)
{
	d_SubtractVector(d_input, d_subtrahends, d_output, sliceelements, slices);
}

__declspec(dllexport) void MultiplySlicesHalf(half* d_input, half* d_multiplicators, half* d_output, size_t sliceelements, uint slices)
{
	d_MultiplyByVector(d_input, d_multiplicators, d_output, sliceelements, slices);
}

__declspec(dllexport) void MultiplyComplexSlicesByScalar(float2* d_input, float* d_multiplicators, float2* d_output, size_t sliceelements, uint slices)
{
	d_ComplexMultiplyByVector(d_input, d_multiplicators, d_output, sliceelements, slices);
}

__declspec(dllexport) void MultiplyComplexSlicesByComplex(float2* d_input, float2* d_multiplicators, float2* d_output, size_t sliceelements, uint slices)
{
	d_ComplexMultiplyByVector(d_input, d_multiplicators, d_output, sliceelements, slices);
}

__declspec(dllexport) void MultiplyComplexSlicesByComplexConj(float2* d_input, float2* d_multiplicators, float2* d_output, size_t sliceelements, uint slices)
{
	d_ComplexMultiplyByConjVector(d_input, d_multiplicators, d_output, sliceelements, slices);
}

__declspec(dllexport) void DivideComplexSlicesByScalar(float2* d_input, float* d_multiplicators, float2* d_output, size_t sliceelements, uint slices)
{
	d_ComplexDivideSafeByVector(d_input, d_multiplicators, d_output, sliceelements, slices);
}

__declspec(dllexport) void MultiplyByScalar(float* d_input, float* d_output, float multiplicator, size_t elements)
{
    d_MultiplyByScalar(d_input, d_output, elements, multiplicator);
}

__declspec(dllexport) void MultiplyByScalars(float* d_input, float* d_output, float* d_multiplicators, size_t elements, uint batch)
{
    d_MultiplyByScalar(d_input, d_multiplicators, d_output, elements, batch);
}

__declspec(dllexport) void Scale(float* d_input, float* d_output, int3 dimsinput, int3 dimsoutput, uint batch, int planforw, int planback, float2* d_inputfft, float2* d_outputfft)
{
	d_Scale(d_input, d_output, dimsinput, dimsoutput, T_INTERP_FOURIER, planforw != 0 ? &planforw : NULL, planback != 0 ? &planback : NULL, batch, d_inputfft, d_outputfft);
}

__declspec(dllexport) void ProjectForward(float2* d_inputft, float2* d_outputft, int3 dimsinput, int2 dimsoutput, float3* h_angles, float supersample, uint batch)
{
    d_rlnProject(d_inputft, dimsinput, d_outputft, toInt3(dimsoutput), (tfloat3*)h_angles, supersample, batch);
}

__declspec(dllexport) void ProjectForwardShifted(float2* d_inputft, float2* d_outputft, int3 dimsinput, int2 dimsoutput, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch)
{
    d_rlnProjectShifted(d_inputft, dimsinput, d_outputft, toInt3(dimsoutput), (tfloat3*)h_angles, (tfloat3*)h_shifts, h_globalweights, supersample, batch);
}

__declspec(dllexport) void ProjectForward3D(float2* d_inputft, float2* d_outputft, int3 dimsinput, int3 dimsoutput, float3* h_angles, float supersample, uint batch)
{
    d_rlnProject(d_inputft, dimsinput, d_outputft, dimsoutput, (tfloat3*)h_angles, supersample, batch);
}

__declspec(dllexport) void ProjectForward3DShifted(float2* d_inputft, float2* d_outputft, int3 dimsinput, int3 dimsoutput, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch)
{
    d_rlnProjectShifted(d_inputft, dimsinput, d_outputft, dimsoutput, (tfloat3*)h_angles, (tfloat3*)h_shifts, h_globalweights, supersample, batch);
}

__declspec(dllexport) void ProjectForwardTex(unsigned long long t_inputRe, unsigned long long t_inputIm, float2* d_outputft, int3 dimsinput, int2 dimsoutput, float3* h_angles, float supersample, uint batch)
{
    d_rlnProject(t_inputRe, t_inputIm, dimsinput, d_outputft, toInt3(dimsoutput), (tfloat3*)h_angles, supersample, batch);
}

__declspec(dllexport) void ProjectForwardShiftedTex(unsigned long long t_inputRe, unsigned long long t_inputIm, float2* d_outputft, int3 dimsinput, int2 dimsoutput, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch)
{
    d_rlnProjectShifted(t_inputRe, t_inputIm, dimsinput, d_outputft, toInt3(dimsoutput), (tfloat3*)h_angles, (tfloat3*)h_shifts, h_globalweights, supersample, batch);
}

__declspec(dllexport) void ProjectForward3DTex(unsigned long long t_inputRe, unsigned long long t_inputIm, float2* d_outputft, int3 dimsinput, int3 dimsoutput, float3* h_angles, float supersample, uint batch)
{
    d_rlnProject(t_inputRe, t_inputIm, dimsinput, d_outputft, dimsoutput, (tfloat3*)h_angles, supersample, batch);
}

__declspec(dllexport) void ProjectForward3DShiftedTex(unsigned long long t_inputRe, unsigned long long t_inputIm, float2* d_outputft, int3 dimsinput, int3 dimsoutput, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch)
{
    d_rlnProjectShifted(t_inputRe, t_inputIm, dimsinput, d_outputft, dimsoutput, (tfloat3*)h_angles, (tfloat3*)h_shifts, h_globalweights, supersample, batch);
}

__declspec(dllexport) void ProjectBackward(float2* d_volumeft, float* d_volumeweights, int3 dimsvolume, float2* d_projft, float* d_projweights, int2 dimsproj, int rmax, float3* h_angles, int* h_ivolume, float4 magnification, float ewaldradius, float supersample, bool outputdecentered, bool squareinterpweights, uint batch)
{
    d_rlnBackproject(d_volumeft, d_volumeweights, dimsvolume, d_projft, d_projweights, toInt3(dimsproj), rmax, (tfloat3*)h_angles, h_ivolume, magnification, ewaldradius, supersample, outputdecentered, squareinterpweights, batch);
}

__declspec(dllexport) void ProjectBackwardShifted(float2* d_volumeft, float* d_volumeweights, int3 dimsvolume, float2* d_projft, float* d_projweights, int2 dimsproj, int rmax, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch)
{
    d_rlnBackprojectShifted(d_volumeft, d_volumeweights, dimsvolume, d_projft, d_projweights, toInt3(dimsproj), rmax, (tfloat3*)h_angles, (tfloat3*)h_shifts, h_globalweights, supersample, batch);
}

__declspec(dllexport) void Bandpass(float* d_input, float* d_output, int3 dims, float nyquistlow, float nyquisthigh, float nyquistsoftedge, uint batch)
{
	d_BandpassNonCubic(d_input, d_output, dims, nyquistlow, nyquisthigh, nyquistsoftedge, batch);
}

__declspec(dllexport) void FourierBandpass(float2* d_inputft, int3 dims, float nyquistlow, float nyquisthigh, float nyquistsoftedge, uint batch)
{
	d_FourierBandpassNonCubic(d_inputft, dims, nyquistlow, nyquisthigh, nyquistsoftedge, batch);
}

__declspec(dllexport) void BandpassButter(float* d_input, float* d_output, int3 dims, float nyquistlow, float nyquisthigh, int order, uint batch)
{
	d_BandpassNonCubicButter(d_input, d_output, dims, nyquistlow, nyquisthigh, order, batch);
}

__declspec(dllexport) void FourierBandpassButter(float2* d_inputft, int3 dims, float nyquistlow, float nyquisthigh, int order, uint batch)
{
	d_FourierBandpassNonCubicButter(d_inputft, dims, nyquistlow, nyquisthigh, order, batch);
}

__declspec(dllexport) void LocalStd(float* d_map, int3 dimsmap, float localradius, float* d_std, float* d_mean, cufftHandle planforw, cufftHandle planback)
{
	d_LocalStd(d_map, dimsmap, NULL, localradius, d_std, d_mean, planforw, planback);
}

__declspec(dllexport) void Rotate2D(float* d_input, float* d_output, int2 dims, float* h_angles, int oversample, uint batch)
{
	if (oversample <= 1)
	{
		d_Rotate2D(d_input, d_output, dims, h_angles, T_INTERP_CUBIC, true, batch);
	}
	else
	{
		int2 dimspadded = dims * oversample;
	    float* d_temp;
		cudaMalloc((void**)&d_temp, Elements2(dimspadded) * sizeof(float));

		for (int b = 0; b < batch; b++)
		{
		    d_Scale(d_input + Elements2(dims) * b, d_temp, toInt3(dims), toInt3(dimspadded), T_INTERP_FOURIER);
			d_Rotate2D(d_temp, d_temp, dimspadded, h_angles + b, T_INTERP_CUBIC, true, 1);
			d_Scale(d_temp, d_output + Elements2(dims) * b, toInt3(dimspadded), toInt3(dims), T_INTERP_FOURIER);
		}

		cudaFree(d_temp);
	}
}

__global__ void ShiftAndRotate2DKernel(float* d_input, float* d_output, int2 dims, int2 dimsori, glm::mat3* d_transforms)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dims.x)
		return;
	uint idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= dims.y)
		return;

	d_input += Elements2(dimsori) * blockIdx.z;

	int x, y;
	x = idx;
	y = idy;

	glm::vec3 pos = d_transforms[blockIdx.z] * glm::vec3(x - dims.x / 2, y - dims.y / 2, 1.0f) + glm::vec3(dimsori.x / 2, dimsori.y / 2, 0.0f);
	
	float val = 0;
	if (pos.x >= 0 && pos.x < dims.x && pos.y >= 0 && pos.y < dims.y)
	{
	    int x0 = floor(pos.x);
		int x1 = tmin(x0 + 1, dims.x - 1);
		pos.x -= x0;

		int y0 = floor(pos.y);
		int y1 = tmin(y0 + 1, dims.y - 1);
		pos.y -= y0;

		float d000 = d_input[y0 * dimsori.x + x0];
		float d001 = d_input[y0 * dimsori.x + x1];
		float d010 = d_input[y1 * dimsori.x + x0];
		float d011 = d_input[y1 * dimsori.x + x1];

		float dx00 = lerp(d000, d001, pos.x);
		float dx01 = lerp(d010, d011, pos.x);

		val = lerp(dx00, dx01, pos.y);
	}

	d_output[(blockIdx.z * dims.y + idy) * dims.x + idx] = val;
}

__declspec(dllexport) void ShiftAndRotate2D(float* d_input, float* d_output, int2 dims, float2* h_shifts, float* h_angles, uint batch)
{
	glm::mat3* h_transforms = (glm::mat3*)malloc(batch * sizeof(glm::mat3));
	for (uint b = 0; b < batch; b++)
		h_transforms[b] = Matrix3Translation(tfloat2(-h_shifts[b].x, -h_shifts[b].y)) * Matrix3RotationZ(h_angles[b]);
	glm::mat3* d_transforms = (glm::mat3*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat3));
	free(h_transforms);

	dim3 TpB = dim3(16, 16);
	dim3 grid = dim3((dims.x + 15) / 16, (dims.y + 15) / 16, batch);

	ShiftAndRotate2DKernel << <grid, TpB >> > (d_input, d_output, dims, dims * 1, d_transforms);

	cudaFree(d_transforms);
}

__declspec(dllexport) void MinScalar(float* d_input, float* d_output, float value, uint elements)
{
    d_MinOp(d_input, value, d_output, elements);
}

__declspec(dllexport) void Min(float* d_input1, float* d_input2, float* d_output, uint elements)
{
	d_MinOp(d_input1, d_input2, d_output, elements);
}

__declspec(dllexport) void MaxScalar(float* d_input, float* d_output, float value, uint elements)
{
    d_MaxOp(d_input, value, d_output, elements);
}

__declspec(dllexport) void Max(float* d_input1, float* d_input2, float* d_output, uint elements)
{
	d_MaxOp(d_input1, d_input2, d_output, elements);
}

__declspec(dllexport) int CreateFFTPlan(int3 dims, uint batch)
{
    return d_FFTR2CGetPlan(DimensionCount(dims), dims, batch);
}

__declspec(dllexport) int CreateIFFTPlan(int3 dims, uint batch)
{
    return d_IFFTC2RGetPlan(DimensionCount(dims), dims, batch);
}

__declspec(dllexport) void DestroyFFTPlan(cufftHandle plan)
{
    cufftDestroy(plan);
}

__declspec(dllexport) void* CreateBLAS()
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	return handle;
}

__declspec(dllexport) void DestroyBLAS(void* handle)
{
	cublasDestroy((cublasHandle_t)handle);
}

__declspec(dllexport) void LocalRes(float* d_volume1, 
									float* d_volume2, 
									int3 dims, 
									float pixelsize, 
									float* d_filtered, 
									float* d_filteredsharp, 
									float* d_localres, 
									float* d_localbfac,
									int windowsize,
									float fscthreshold, 
									bool dolocalbfactor, 
									float minresbfactor,
									float globalbfactor, 
									float mtfslope,
									bool doanisotropy,
									bool dofilterhalfmaps)
{
	d_LocalFSCBfac(d_volume1, 
					d_volume2, 
					dims, 
					d_localres, 
					d_localbfac, 
					d_filteredsharp, 
					d_filtered, 
					windowsize, 
					fscthreshold, 
					dolocalbfactor, 
					globalbfactor, 
					minresbfactor, 
					pixelsize, 
					0, 
					0, 
					mtfslope, 
					doanisotropy,
					dofilterhalfmaps);
}

__declspec(dllexport) void LocalFSC(float* d_volume1, 
									float* d_volume2, 
									float* d_volumemask,
									int3 dims, 
									int spacing,
									float pixelsize, 
									float* d_localres, 
									int windowsize,
									float fscthreshold,
									float* d_avgfsc,
									float* d_avgamps,
									float* d_avgsamples,
									int avgoversample,
									float* h_globalfsc)
{
	d_LocalFSC(d_volume1, 
				d_volume2,
				d_volumemask,
				dims, 
				d_localres, 
				windowsize,
				spacing, 
				fscthreshold, 
				pixelsize,
				d_avgfsc,
				d_avgamps,
				d_avgsamples,
				avgoversample,
				h_globalfsc);
}

__declspec(dllexport) void LocalFilter(float* d_input,
										float* d_filtered,
										int3 dims,
										float* d_localres,
										int windowsize,
										float angpix,
										float* d_filterramps,
										int rampsoversample)
{
	d_LocalFilter(d_input,
					d_filtered,
					dims,
					d_localres,
					windowsize,
					angpix,
					d_filterramps,
					rampsoversample);
}

__declspec(dllexport) void ProjectNMAPseudoAtoms(float3* d_positions,
												float* d_intensities,
												uint natoms,
												int3 dimsvol,
												float sigma,
												uint kernelextent,
												float* h_blobvalues,
												float blobsampling,
												uint nblobvalues,
												float3* d_normalmodes,
												float* d_normalmodefactors,
												uint nmodes,
												float3* h_angles,
												float2* h_offsets,
												float scale,
												float* d_proj,
												int2 dimsproj,
												uint batch)
{
	d_ValueFill(d_proj, Elements2(dimsproj) * batch, 0.0f);

    d_ProjNMAPseudoAtoms(d_positions,
						d_intensities,
						natoms,
						dimsvol,
						sigma,
						kernelextent,
						h_blobvalues,
						blobsampling,
						nblobvalues,
						d_normalmodes,
						d_normalmodefactors,
						nmodes,
						h_angles,
						h_offsets,
						scale,
						d_proj,
						dimsproj,
						batch);
}

__declspec(dllexport) void ProjectSoftPseudoAtoms(float3* d_positions,
													float* d_intensities,
													uint natoms,
													int3 dimsvol,
													float sigma,
													uint kernelextent,
													float3* d_coarsedeltas,
													float* d_coarseweights,
													int* d_coarseneighbors,
													uint ncoarseneighbors,
													uint ncoarse,
													float3* h_angles,
													float2* h_offsets,
													float scale,
													float* d_proj,
													int2 dimsproj,
													uint batch)
{
	d_ValueFill(d_proj, Elements2(dimsproj) * batch, 0.0f);

    d_ProjSoftPseudoAtoms(d_positions,
						d_intensities,
						natoms,
						dimsvol,
						sigma,
						kernelextent,
						d_coarsedeltas,
						d_coarseweights,
						d_coarseneighbors,
						ncoarseneighbors,
						ncoarse,
						h_angles,
						h_offsets,
						scale,
						d_proj,
						dimsproj,
						batch);
}

__declspec(dllexport) void DistanceMap(float* d_input, float* d_output, int3 dims, int maxiterations)
{
    d_DistanceMap(d_input, d_output, dims, maxiterations);
}

__declspec(dllexport) void DistanceMapExact(float* d_input, float* d_output, int3 dims, int maxdistance)
{
	d_DistanceMapExact(d_input, d_output, dims, maxdistance);
}

__declspec(dllexport) void PrefilterForCubic(float* d_data, int3 dims)
{
	if (dims.z > 1)
		d_CubicBSplinePrefilter3D(d_data, dims);
	else
		d_CubicBSplinePrefilter2D(d_data, toInt2(dims));
}

__declspec(dllexport) void CreateTexture3D(float* d_data, int3 dims, unsigned long long* h_textureid, unsigned long long* h_arrayid, bool linearfiltering)
{
	d_BindTextureTo3DArray(d_data, ((cudaArray_t*)h_arrayid)[0], ((cudaTex*)h_textureid)[0], dims, linearfiltering ? cudaFilterModeLinear : cudaFilterModePoint, false);
}

__declspec(dllexport) void CreateTexture3DComplex(float2* d_data, int3 dims, unsigned long long* h_textureid, unsigned long long* h_arrayid, bool linearfiltering)
{
    float* d_dataRe;
	cudaMalloc((void**)&d_dataRe, Elements(dims) * sizeof(float));
	d_Re(d_data, d_dataRe, Elements(dims));
    float* d_dataIm;
	cudaMalloc((void**)&d_dataIm, Elements(dims) * sizeof(float));
	d_Im(d_data, d_dataIm, Elements(dims));

	d_BindTextureTo3DArray(d_dataRe, ((cudaArray_t*)h_arrayid)[0], ((cudaTex*)h_textureid)[0], dims, linearfiltering ? cudaFilterModeLinear : cudaFilterModePoint, false);
	d_BindTextureTo3DArray(d_dataIm, ((cudaArray_t*)h_arrayid)[1], ((cudaTex*)h_textureid)[1], dims, linearfiltering ? cudaFilterModeLinear : cudaFilterModePoint, false);
	
	cudaFree(d_dataRe);
	cudaFree(d_dataIm);
}

__declspec(dllexport) void DestroyTexture(unsigned long long textureid, unsigned long long arrayid)
{
	cudaDestroyTextureObject(*(cudaTex*)&textureid);
    cudaFreeArray(*(cudaArray_t*)&arrayid);
}

__declspec(dllexport) void ValueFill(float* d_input, size_t elements, float value)
{
	d_ValueFill(d_input, elements, value);
}

__declspec(dllexport) void ValueFillComplex(float2* d_input, size_t elements, float2 value)
{
	d_ValueFill(d_input, elements, value);
}

__declspec(dllexport) void Real(float2* d_input, float* d_output, size_t elements)
{
	d_Re(d_input, d_output, elements);
}

__declspec(dllexport) void Imag(float2* d_input, float* d_output, size_t elements)
{
	d_Im(d_input, d_output, elements);
}

__declspec(dllexport) int PeekLastCUDAError()
{
    cudaDeviceSynchronize();
	cudaError_t result = cudaPeekAtLastError();
	return (int)result;
}

__declspec(dllexport) void DistortImages(float* d_input, int2 dimsinput, float* d_output, int2 dimsoutput, float2* h_offsets, float* h_rotations, float3* h_scales, float noisestddev, int seed, uint batch)
{
    d_DistortImages(d_input, dimsinput, d_output, dimsoutput, h_offsets, h_rotations, h_scales, batch);

    d_NormMonolithic(d_output, d_output, Elements2(dimsoutput), T_NORM_MEAN01STD, batch);

    float* d_noise = CudaMallocRandomFilled(Elements2(dimsoutput) * batch, 0, noisestddev, seed);
    d_AddVector(d_output, d_noise, d_output, Elements2(dimsoutput) * batch);

    cudaFree(d_noise);

    d_NormMonolithic(d_output, d_output, Elements2(dimsoutput), T_NORM_MEAN01STD, batch);
}

__declspec(dllexport) void DistortImagesAffine(float* d_input, int2 dimsinput, float* d_output, int2 dimsoutput, float2* h_offsets, float4* h_distortions, uint batch)
{
	d_DistortImages(d_input, dimsinput, d_output, dimsoutput, h_offsets, h_distortions, batch);
}

__declspec(dllexport) void WarpImage(float* d_input, float* d_output, int2 dims, float* h_warpx, float* h_warpy, int2 dimswarp, cudaArray_t a_input)
{
    d_WarpImage(d_input, d_output, dims, h_warpx, h_warpy, dimswarp, a_input);
}

__declspec(dllexport) void Rotate3DExtractAt(unsigned long long t_volume, int3 dimsvolume, float* d_proj, int3 dimsproj, float3* h_angles, float3* h_positions, uint batch)
{
    d_Rotate3DExtractAt((cudaTex)t_volume, dimsvolume, (tfloat*)d_proj, dimsproj, (tfloat3*)h_angles, (tfloat3*)h_positions, T_INTERP_CUBIC, batch);
}

__declspec(dllexport) void BackProjectTomo(float2* d_volumeft, int3 dimsvolume, float2* d_projft, float* d_projweights, int3 dimsproj, uint rmax, float3* h_angles, uint batch)
{
	d_BackprojectTomo(d_volumeft, dimsvolume, d_projft, d_projweights, dimsproj, rmax, (tfloat3*)h_angles, batch);
}

__global__ void RepeatKernel(float* d_input, float* d_output, uint elements, uint ncopies)
{
	d_input += elements * blockIdx.x;
	d_output += elements * ncopies * blockIdx.x;

	for (uint id = threadIdx.x; id < elements; id += blockDim.x)
	{
		float val = d_input[id];
		for (uint c = 0; c < ncopies; c++)
			d_output[c * elements + id] = val;
	}
}

__declspec(dllexport) void Repeat(float* d_input, float* d_output, uint elements, uint ncopies, int batch)
{
	dim3 TpB = tmin(128, elements);
	dim3 grid = batch;

	RepeatKernel << <grid, TpB >> > (d_input, d_output, elements, ncopies);
}