#include "Prerequisites.cuh"
#include "Angles.cuh"

#ifndef CTF_CUH
#define CTF_CUH

namespace gtom
{
	// All lengths in meters
	struct CTFParams
	{
		tfloat pixelsize;
        tfloat pixeldelta;
        tfloat pixelangle;
		tfloat Cs;
		tfloat voltage;
		tfloat defocus;
		tfloat astigmatismangle;
		tfloat defocusdelta;
		tfloat amplitude;
		tfloat Bfactor;
		tfloat Bfactordelta;
		tfloat Bfactorangle;
		tfloat scale;
		tfloat phaseshift;

		CTFParams() :
			pixelsize(1e-10),
			pixeldelta(0),
			pixelangle(0),
			Cs(2e-3),
			voltage(300e3),
			defocus(-3e-6),
			astigmatismangle(0),
			defocusdelta(0),
			amplitude(0.07),
			Bfactor(0),
			Bfactordelta(0),
			Bfactorangle(0),
			scale(1.0),
			phaseshift(0) {}
	};

	struct CTFFitParams
	{
		tfloat3 pixelsize;
		tfloat3 pixeldelta;
		tfloat3 pixelangle;
		tfloat3 Cs;
		tfloat3 voltage;
		tfloat3 defocus;
		tfloat3 astigmatismangle;
		tfloat3 defocusdelta;
		tfloat3 amplitude;
		tfloat3 Bfactor;
		tfloat3 scale;
		tfloat3 phaseshift;

		int2 dimsperiodogram;
		int maskinnerradius;
		int maskouterradius;

		CTFFitParams() :
			pixelsize(0),
			pixeldelta(0),
			pixelangle(0),
			Cs(0),
			voltage(0),
			defocus(0),
			astigmatismangle(0),
			defocusdelta(0),
			amplitude(0),
			Bfactor(0),
			scale(0),
			phaseshift(0),
			dimsperiodogram(toInt2(512, 512)),
			maskinnerradius(1),
			maskouterradius(128) {}

		CTFFitParams(CTFParams p) :
			pixelsize(p.pixelsize),
			pixeldelta(p.pixeldelta),
			pixelangle(p.pixelangle),
			Cs(p.Cs),
			voltage(p.voltage),
			defocus(p.defocus),
			astigmatismangle(p.astigmatismangle),
			defocusdelta(p.defocusdelta),
			amplitude(p.amplitude),
			Bfactor(p.Bfactor),
			scale(p.scale),
			phaseshift(p.phaseshift),
			dimsperiodogram(toInt2(512, 512)),
			maskinnerradius(1),
			maskouterradius(128) {}
	};

	// All lengths in Angstrom
	struct CTFParamsLean
	{
		tfloat ny;
		tfloat pixelsize;
		tfloat pixeldelta;
		tfloat pixelangle;
		tfloat lambda;
		tfloat defocus;
		tfloat astigmatismangle;
		tfloat defocusdelta;
		tfloat Cs;
		tfloat scale;
		tfloat phaseshift;
		tfloat K1, K2, K3;
		tfloat Bfactor, Bfactordelta, Bfactorangle;

		CTFParamsLean(CTFParams p, int3 dims) :
			ny(1.0f / (dims.z > 1 ? (tfloat)dims.x * (p.pixelsize * 1e10) : (tfloat)dims.x)),
			pixelsize(p.pixelsize * 1e10),
			pixeldelta(p.pixeldelta * 0.5e10),
			pixelangle(p.pixelangle),
			lambda(12.2643247 / sqrt(p.voltage * (1.0 + p.voltage * 0.978466e-6))),
			defocus(p.defocus * 1e10),
			astigmatismangle(p.astigmatismangle),
			defocusdelta(p.defocusdelta * 0.5e10),
			Cs(p.Cs * 1e10),
			scale(p.scale),
			phaseshift(p.phaseshift),
			K1(PI * lambda),
			K2(PIHALF * (p.Cs * 1e10) * lambda * lambda * lambda),
			K3(atan(p.amplitude / sqrt(1 - p.amplitude * p.amplitude))),
			Bfactor(p.Bfactor * 0.25e20),
			Bfactordelta(p.Bfactordelta * 0.25e20),
			Bfactorangle(p.Bfactorangle) {}
	};

	template<bool ampsquared, bool ignorefirstpeak> __device__ tfloat d_GetCTF(tfloat r, tfloat angle, tfloat gammacorrection, CTFParamsLean p)
	{
		tfloat r2 = r * r;
		tfloat r4 = r2 * r2;
				
		tfloat deltaf = p.defocus + p.defocusdelta * __cosf((tfloat)2 * (angle - p.astigmatismangle));
		tfloat gamma = p.K1 * deltaf * r2 + p.K2 * r4 - p.phaseshift - p.K3 + gammacorrection;
		tfloat retval;
		if (ignorefirstpeak && abs(gamma) < PI / 2)
			retval = 1;
		else
			retval = -__sinf(gamma);

		if (p.Bfactor != 0 || p.Bfactordelta != 0)
		{
			tfloat Bfacaniso = p.Bfactor;
			if (p.Bfactordelta != 0)
				Bfacaniso += p.Bfactordelta * __cosf((tfloat)2 * (angle - p.Bfactorangle));

			retval *= __expf(Bfacaniso * r2);
		}

		if (ampsquared)
			retval = abs(retval);

		retval *= p.scale;

		return retval;
	}

	template<bool dummy> __device__ float2 d_GetCTFComplex(tfloat r, tfloat angle, tfloat gammacorrection, CTFParamsLean p, bool reverse)
	{
		tfloat r2 = r * r;
		tfloat r4 = r2 * r2;

		tfloat deltaf = p.defocus + p.defocusdelta * cos((tfloat)2 * (angle - p.astigmatismangle));
		tfloat gamma = p.K1 * deltaf * r2 + p.K2 * r4 - p.phaseshift - p.K3 + PI / 2 + gammacorrection;
		float2 retval = make_float2(cos(gamma), sin(gamma));
		if (reverse)
			retval.y *= -1;

		if (p.Bfactor != 0 || p.Bfactordelta != 0)
		{
			tfloat Bfacaniso = p.Bfactor;
			if (p.Bfactordelta != 0)
				Bfacaniso += p.Bfactordelta * cos((tfloat)2 * (angle - p.Bfactorangle));

			retval *= exp(Bfacaniso * r2);
		}

		retval *= p.scale;

		return retval;
	}

	//AliasingCutoff.cu:
	uint CTFGetAliasingCutoff(CTFParams params, uint sidelength);

	//CommonPSF.cu:
	void d_ForceCommonPSF(tcomplex* d_inft1, tcomplex* d_inft2, tcomplex* d_outft1, tcomplex* d_outft2, tfloat* d_psf1, tfloat* d_psf2, tfloat* d_commonpsf, uint n, bool same2, int batch);

	//Correct.cu:
	void d_CTFCorrect(tcomplex* d_input, int3 dimsinput, CTFParams params, tcomplex* d_output);

	//Decay.cu:
	void d_CTFDecay(tfloat* d_input, tfloat* d_output, int2 dims, int degree, int stripwidth);

	//InterpolateIrregular.cu:
	void Interpolate1DOntoGrid(std::vector<tfloat2> sortedpoints, tfloat* h_output, uint gridstart, uint gridend);

	//Periodogram.cu:
	void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, float overlapfraction, int2 dimsregion, int2 dimspadded, tfloat* d_output2d, bool dopost = true);
	void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, int2 dimsregion, int2 dimspadded, tfloat* d_output2d, bool dopost = true, cufftHandle planforw = 0, tfloat* d_extracted = NULL, tcomplex* d_extractedft = NULL);

	//RotationalAverage.cu:
	void d_CTFRotationalAverage(tfloat* d_re, 
								int2 dimsinput, 
								CTFParams* h_params, 
								tfloat* d_average, 
								ushort freqlow, 
								ushort freqhigh, 
								int batch = 1);
	void d_CTFRotationalAverage(tfloat* d_input, 
								float2* d_inputcoords, 
								uint inputlength, 
								uint sidelength, 
								CTFParams* h_params, 
								tfloat* d_average, 
								ushort freqlow, 
								ushort freqhigh, 
								int batch = 1);
	template<class T> void d_CTFRotationalAverageToTarget(T* d_input, 
														float2* d_inputcoords, 
														uint inputlength, 
														uint sidelength, 
														CTFParams* h_params, 
														CTFParams targetparams, 
														tfloat* d_average, 
														ushort freqlow, 
														ushort freqhigh, 
														int batch = 1);
	void d_CTFRotationalAverageToTargetDeterministic(tfloat* d_input,
													float2* d_inputcoords,
													uint inputlength,
													uint sidelength,
													CTFParams* h_params,
													CTFParams targetparams,
													tfloat* d_average,
													ushort freqlow,
													ushort freqhigh,
													int batch);

	//Simulate.cu:
	void d_CTFSimulate(CTFParams* h_params, float2* d_addresses, float* d_gammacorrection, tfloat* d_output, uint n, bool amplitudesquared = false, bool ignorefirstpeak = false, int batch = 1);
	void d_CTFSimulate(CTFParams* h_params, half2* d_addresses, half* d_output, uint n, bool amplitudesquared = false, bool ignorefirstpeak = false, int batch = 1);
	void d_CTFSimulateComplex(CTFParams* h_params, float2* d_addresses, float* d_gammacorrection, float2* d_output, uint n, bool reverse, int batch = 1);
	void d_CTFSimulateEwaldWeights(CTFParams* h_params, float2* d_addresses, float* d_gammacorrection, float particlediameter, tfloat* d_output, uint n, int batch = 1);

	//Wiener.cu:
	void d_CTFWiener(tcomplex* d_input, int3 dimsinput, tfloat* d_fsc, CTFParams* h_params, tcomplex* d_output, tfloat* d_outputweights, uint batch = 1);
	void d_CTFWiener(tcomplex* d_input, int3 dimsinput, tfloat snr, CTFParams* h_params, tcomplex* d_output, tfloat* d_outputweights, uint batch = 1);
}
#endif