#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Correlation.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Masking.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
	__global__ void ProbSolventStatisticsKernel(tfloat* d_mean, tfloat* d_std, tfloat normfactor, size_t elements);
	__global__ void ProbPickingUpdateKernel(tfloat* d_corr,
		tfloat normfactor,
		tfloat* d_solventmean,
		tfloat* d_solventstd,
		tfloat summaskedref,
		tfloat summaskedref2,
		double expectedpratio,
		size_t elements,
		tfloat3 angle,
		int refid,
		tfloat* d_bestccf,
		tfloat3* d_bestangle,
		int* d_bestref);

	ProbabilisticPicker::ProbabilisticPicker()
	{

	}

	void ProbabilisticPicker::Initialize(tfloat* _d_refs, int3 _dimsref, uint _nrefs, tfloat* _d_refmasks, bool _ismaskcircular, bool _doctf, int3 _dimsimage, uint _lowpassfreq)
	{
		dimsref = _dimsref;
		nrefs = _nrefs;
		ndims = DimensionCount(dimsref);
		dimsimage = _dimsimage;
		dimslowpass = ndims == 3 ? toInt3(_lowpassfreq * 2, _lowpassfreq * 2, _lowpassfreq * 2) : toInt3(_lowpassfreq * 2, _lowpassfreq * 2, 1);
		ismaskcircular = _ismaskcircular;
		doctf = _doctf;

		cudaMallocHost((void**)&h_refsRe, ElementsFFT(dimslowpass) * nrefs * sizeof(tfloat));
		cudaMallocHost((void**)&h_refsIm, ElementsFFT(dimslowpass) * nrefs * sizeof(tfloat));
		cudaMallocHost((void**)&h_masks, Elements(dimsimage) * nrefs * sizeof(tfloat));

		// (I)FFT plans
		planforw = d_FFTR2CGetPlan(ndims, dimsimage);
		planback = d_IFFTC2RGetPlan(ndims, dimsimage);

		// Allocate device memory used for each reference's correlation
		cudaMalloc((void**)&d_maskpadded, ElementsFFT(dimsimage) * sizeof(tcomplex));
		cudaMalloc((void**)&d_invmaskft, ElementsFFT(dimsimage) * sizeof(tcomplex));
		cudaMalloc((void**)&d_invmask, ElementsFFT(dimsimage) * sizeof(tcomplex));

		cudaMalloc((void**)&d_solventmean, Elements(dimsimage) * sizeof(tfloat));
		cudaMalloc((void**)&d_solventstd, Elements(dimsimage) * sizeof(tfloat));

		cudaMalloc((void**)&d_imageft, ElementsFFT(dimsimage) * sizeof(tcomplex));
		cudaMalloc((void**)&d_image2ft, ElementsFFT(dimsimage) * sizeof(tcomplex));
		cudaMalloc((void**)&d_ctf, ElementsFFT(dimslowpass) * sizeof(tfloat));

		cudaMalloc((void**)&d_reflowft, ElementsFFT(dimslowpass) * sizeof(tcomplex));
		cudaMalloc((void**)&d_refft, ElementsFFT(dimsimage) * sizeof(tcomplex));
		
		cudaMalloc((void**)&d_refcropped, Elements(dimsref) * sizeof(tfloat));
		cudaMalloc((void**)&d_maskcropped, Elements(dimsref) * sizeof(tfloat));

		cudaMalloc((void**)&d_buffer1, ElementsFFT(dimsimage) * sizeof(tcomplex));
		cudaMalloc((void**)&d_buffer2, ElementsFFT(dimsimage) * sizeof(tcomplex));

		// Prepare normal and inverted masks
		{
			// Count samples within the mask
			h_masksum = (tfloat*)malloc(nrefs * sizeof(tfloat));
			tfloat* d_masksum = CudaMallocValueFilled(nrefs, (tfloat)0);
			d_SumMonolithic(_d_refmasks, d_masksum, Elements(dimsref), nrefs);
			cudaMemcpy(h_masksum, d_masksum, nrefs * sizeof(tfloat), cudaMemcpyDeviceToHost);

			// Count samples within the inverse mask
			h_invmasksum = (tfloat*)malloc(nrefs * sizeof(tfloat));
			for (uint n = 0; n < nrefs; n++)
				h_invmasksum[n] = (tfloat)Elements(dimsref) - h_masksum[n];

			cudaMallocHost((void**)&h_masks, Elements(dimsimage) * nrefs * sizeof(tfloat));
			cudaMallocHost((void**)&h_invmasksft, ElementsFFT(dimsimage) * nrefs * sizeof(tcomplex));

			tfloat* d_invmask;
			cudaMalloc((void**)&d_invmask, ElementsFFT(dimsref) * sizeof(tcomplex));

			for (uint n = 0; n < nrefs; n++)
			{
				// Pad inverse mask to image/volume size and FFT
				d_OneMinus(_d_refmasks + Elements(dimsref) * n, d_invmask, Elements(dimsref));
				d_Pad(d_invmask, (tfloat*)d_invmaskft, dimsref, dimsimage, T_PAD_VALUE, (tfloat)0);
				d_RemapFull2FullFFT((tfloat*)d_invmaskft, d_maskpadded, dimsimage);
				//d_WriteMRC(d_maskpadded, dimsimage, "d_invmaskpadded.mrc");
				d_FFTR2C(d_maskpadded, d_invmaskft, &planforw);
				cudaMemcpy(h_invmasksft + ElementsFFT(dimsimage) * n, d_invmaskft, ElementsFFT(dimsimage) * sizeof(tcomplex), cudaMemcpyDeviceToHost);

				// Pad mask to image/volume size and FFT
				d_Pad(_d_refmasks + Elements(dimsref) * n, d_maskpadded, dimsref, dimsimage, T_PAD_VALUE, (tfloat)0);
				if (ismaskcircular)
					d_RemapFull2FullFFT(d_maskpadded, d_maskpadded, dimsimage);
				else
				{
					// Prefilter for cubic interpolation, since it needs to be rotated
					if (ndims == 2)
						d_CubicBSplinePrefilter2D(d_maskpadded, toInt2(dimsimage));
					else
						d_CubicBSplinePrefilter3D(d_maskpadded, dimsimage);
				}
				//d_WriteMRC(d_maskpadded, dimsimage, "d_maskpadded.mrc");
				cudaMemcpy(h_masks + Elements(dimsimage) * n, d_maskpadded, Elements(dimsimage) * sizeof(tfloat), cudaMemcpyDeviceToHost);
			}

			cudaFree(d_invmask);
			cudaFree(d_masksum);
		}

		// Prepare references
		{
			tfloat* d_refpadded;
			cudaMalloc((void**)&d_refpadded, ElementsFFT(dimsimage) * sizeof(tcomplex));
			tcomplex* d_refpaddedft;
			cudaMalloc((void**)&d_refpaddedft, ElementsFFT(dimsimage) * sizeof(tcomplex));

			for (uint n = 0; n < nrefs; n++)
			{
				// Pad to image/volume size, resize to lowpass
				d_Pad(_d_refs + Elements(dimsref) * n, (tfloat*)d_refpaddedft, dimsref, dimsimage, T_PAD_VALUE, (tfloat)0);
				d_RemapFull2FullFFT((tfloat*)d_refpaddedft, d_refpadded, dimsimage);
				//d_WriteMRC(d_refpadded, dimsimage, "d_refspadded.mrc");
				d_FFTR2C(d_refpadded, d_refpaddedft, &planforw);
				d_FFTCrop(d_refpaddedft, (tcomplex*)d_refpadded, dimsimage, dimslowpass);
				d_Bandpass((tcomplex*)d_refpadded, (tcomplex*)d_refpadded, dimslowpass, 0, dimslowpass.x / 2, 0, NULL);
				d_RemapHalfFFT2Half((tcomplex*)d_refpadded, d_refpaddedft, dimslowpass);

				// Store in texture for subsequent rotation
				d_ConvertTComplexToSplitComplex(d_refpaddedft, d_refpadded, d_refpadded + ElementsFFT(dimslowpass), ElementsFFT(dimslowpass));
				cudaMemcpy(h_refsRe + ElementsFFT(dimslowpass) * n, d_refpadded, ElementsFFT(dimslowpass) * sizeof(tfloat), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_refsIm + ElementsFFT(dimslowpass) * n, d_refpadded + ElementsFFT(dimslowpass), ElementsFFT(dimslowpass) * sizeof(tfloat), cudaMemcpyDeviceToHost);
			}

			cudaFree(d_refpaddedft);
			cudaFree(d_refpadded);
		}
	}

	ProbabilisticPicker::~ProbabilisticPicker()
	{
		cufftDestroy(planforw);
		cufftDestroy(planback);

		cudaFree(d_buffer2);
		cudaFree(d_buffer1);
		cudaFree(d_maskcropped);
		cudaFree(d_refcropped);
		cudaFree(d_invmask);
		cudaFree(d_refft);
		cudaFree(d_reflowft);
		cudaFree(d_ctf);
		cudaFree(d_image2ft);
		cudaFree(d_imageft);
		cudaFree(d_solventstd);
		cudaFree(d_solventmean);
		cudaFree(d_maskpadded);
		cudaFree(d_invmaskft);

		free(h_masksum);
		free(h_invmasksum);

		cudaFreeHost(h_refsRe);
		cudaFreeHost(h_refsIm);
		cudaFreeHost(h_invmasksft);
		cudaFreeHost(h_masks);
	}

	void ProbabilisticPicker::SetImage(tfloat* _d_image, tfloat* _d_ctf)
	{
		// Normalize image/volume
		d_Norm(_d_image, d_buffer1, Elements(dimsimage), (tfloat*)NULL, T_NORM_MEAN01STD, 0);
		//d_WriteMRC(d_buffer1, dimsimage, "d_image.mrc");

		// Calc FFT of image, and image^2
		d_FFTR2C(d_buffer1, d_imageft, &planforw);
		d_Bandpass(d_imageft, d_imageft, dimsimage, 0, dimslowpass.x / 2, 0);

		d_Square(d_buffer1, d_buffer1, Elements(dimsimage));
		//d_WriteMRC(d_buffer1, dimsimage, "d_image2.mrc");
		d_FFTR2C(d_buffer1, d_image2ft, &planforw);
		d_Bandpass(d_image2ft, d_image2ft, dimsimage, 0, dimslowpass.x / 2, 0);
		
		// Crop CTF to account for low-pass
		d_FFTCrop(_d_ctf, d_ctf, dimsimage, dimslowpass);
		//d_WriteMRC(d_ctf, toInt3(dimsimage.x / 2 + 1, dimsimage.y, dimsimage.z), "d_ctf.mrc");
	}

	void ProbabilisticPicker::PerformCorrelation(uint n, tfloat anglestep, tfloat* d_bestccf, tfloat3* d_bestangle, int* d_bestref)
	{
		// Initialize rand for Gaussian noise needed later
		srand(123);

		// Texture and array references
		cudaTex t_refRe, t_refIm, t_mask;
		cudaTex* dt_mask;	// For d_Rotate2D
		cudaArray_t a_refRe, a_refIm, a_mask;

		// Get all relevant angles
		std::vector<float3> angles;
		if (ndims == 3)
			angles = GetEqualAngularSpacing(make_float2(0, PI2), make_float2(0, PI), make_float2(0, PI2), anglestep);
		else
		{
			int anglesteps = ceil(PI2 / anglestep);
			anglestep = PI2 / (tfloat)anglesteps;
			for (uint a = 0; a < anglesteps; a++)
				angles.push_back(make_float3(0, 0, (float)a * anglestep));
		}

		// Copy data from host and assign to textures
		{
			cudaMemcpy(d_buffer1, h_refsRe + ElementsFFT(dimslowpass) * n, ElementsFFT(dimslowpass) * sizeof(tfloat), cudaMemcpyHostToDevice);
			if (ndims == 2)
				d_BindTextureToArray(d_buffer1, a_refRe, t_refRe, toInt2FFT(dimslowpass), cudaFilterModeLinear, false);
			else
				d_BindTextureTo3DArray(d_buffer1, a_refRe, t_refRe, toInt3FFT(dimslowpass), cudaFilterModeLinear, false);

			cudaMemcpy(d_buffer1, h_refsIm + ElementsFFT(dimslowpass) * n, ElementsFFT(dimslowpass) * sizeof(tfloat), cudaMemcpyHostToDevice);
			if (ndims == 2)
				d_BindTextureToArray(d_buffer1, a_refIm, t_refIm, toInt2FFT(dimslowpass), cudaFilterModeLinear, false);
			else
				d_BindTextureTo3DArray(d_buffer1, a_refIm, t_refIm, toInt3FFT(dimslowpass), cudaFilterModeLinear, false);

			cudaMemcpy(d_maskpadded, h_masks + Elements(dimsimage) * n, Elements(dimsimage) * sizeof(tfloat), cudaMemcpyHostToDevice);
			d_WriteMRC(d_maskpadded, dimsimage, "d_maskpadded.mrc");
			if (!ismaskcircular)	// Need texture for rotation
			{
				if (ndims == 2)
					d_BindTextureToArray(d_maskpadded, a_mask, t_mask, toInt2(dimsimage), cudaFilterModeLinear, false);
				else
					d_BindTextureTo3DArray(d_maskpadded, a_mask, t_mask, dimsimage, cudaFilterModeLinear, false);
				dt_mask = (cudaTex*)CudaMallocFromHostArray(&t_mask, sizeof(cudaTex));
			}

			// If not circular, this will be created later based on the rotated mask
			if (ismaskcircular)
				cudaMemcpy(d_invmaskft, h_invmasksft + ElementsFFT(dimsimage) * n, ElementsFFT(dimsimage) * sizeof(tcomplex), cudaMemcpyHostToDevice);
		}

		double summaskedref = 0, summaskedref2 = 0;
		double expectedpratio = 0;

		// For each angle, rotate everything and correlate
		for (uint a = 0; a < angles.size(); a++)
		{
			tfloat3 angle = tfloat3(angles[a]);

			if (ndims == 3)
				d_Rotate3DFT(t_refRe, t_refIm, d_reflowft, dimslowpass, &angle, 1, T_INTERP_LINEAR, false);
			else
				d_Rotate2DFT(t_refRe, t_refIm, d_reflowft, dimslowpass, angle.z, dimslowpass.x / 2, T_INTERP_LINEAR, false);
					
			// Apply image/volume CTF to reference, pad to original resolution, and IFFT it for masking and statistics
			if (doctf)
				d_ComplexMultiplyByVector(d_reflowft, d_ctf, d_reflowft, ElementsFFT(dimslowpass));
			d_FFTPad(d_reflowft, d_refft, dimslowpass, dimsimage);
			d_IFFTC2R(d_refft, d_buffer2, &planback, dimsimage);
			d_WriteMRC((tfloat*)d_buffer2, dimsimage, "d_refft.mrc");

			// Get mask (for current angle, if not circular)
			if (!ismaskcircular)
			{
				if (ndims == 3)
					d_Rotate3D(t_mask, d_maskpadded, dimsimage, &angle, 1, T_INTERP_CUBIC, false);
				else
					d_Rotate2D(dt_mask, d_maskpadded, toInt2(dimsimage), &angle.z, T_INTERP_CUBIC, false);
				//d_WriteMRC(d_maskpadded, dimsimage, "d_maskpadded.mrc");
			}

			// Apply rotated mask to corrected reference
			d_MultiplyByVector(d_buffer2, d_maskpadded, d_buffer2, Elements(dimsimage));
			//d_WriteMRC((tfloat*)d_refft, dimsimage, "d_refftmasked.mrc");

			// Correlate image/volume with corrected reference
			{
				d_FFTR2C(d_buffer2, (tcomplex*)d_buffer1, &planforw);
				d_ComplexMultiplyByConjVector(d_imageft, (tcomplex*)d_buffer1, (tcomplex*)d_buffer1, ElementsFFT(dimsimage));
				d_IFFTC2R((tcomplex*)d_buffer1, d_buffer1, &planback);
				//d_WriteMRC(d_buffer1, dimsimage, "d_buffer1.mrc");
			}

			// Do the statistics
			{
				// fftshift reference and crop to dimsref
				d_FFTFullCrop(d_buffer2, d_refcropped, dimsimage, dimsref);
				//d_WriteMRC(d_refcropped, dimsref, "d_refcropped.mrc");
				// fftshift mask and crop to dimsref
				d_FFTFullCrop(d_maskpadded, d_maskcropped, dimsimage, dimsref);
				//d_WriteMRC(d_maskcropped, dimsref, "d_maskcropped.mrc");

				// Calc expected ratio of probabilities
				{
					tfloat* h_refcropped = (tfloat*)MallocFromDeviceArray(d_refcropped, Elements(dimsref) * sizeof(tfloat));
					tfloat* h_maskcropped = (tfloat*)MallocFromDeviceArray(d_maskcropped, Elements(dimsref) * sizeof(tfloat));
					summaskedref = 0;
					summaskedref2 = 0;
					expectedpratio = 0;
					double expectedpratios[5];
					for (uint r = 0; r < 5; r++)
						expectedpratios[r] = 0;
					double samples = 0;

					for (uint i = 0; i < Elements(dimsref); i++)
					{
						double masksample = h_maskcropped[i];
						samples += masksample;

						double refsample = h_refcropped[i];
						double refsample2 = refsample * refsample;

						for (uint r = 0; r < 5; r++)
							expectedpratios[r] += refsample2 + 2.0 * refsample * gaussrand();
						summaskedref += refsample;
						summaskedref2 += refsample2;
					}

					summaskedref /= samples;
					summaskedref2 /= samples;
					for (uint r = 0; r < 5; r++)
						expectedpratio += expectedpratios[r];
					expectedpratio /= 5.0;
					expectedpratio = exp(expectedpratio / (2.0 * samples));

					h_masksum[n] = samples;
					h_invmasksum[n] = (double)Elements(dimsref) - samples;

					free(h_maskcropped);
					free(h_refcropped);
				}

				// If mask was rotated, need to update the inverted version
				if (!ismaskcircular)
				{
					d_OneMinus(d_maskcropped, d_maskcropped, Elements(dimsref));
					d_FFTFullPad(d_maskcropped, d_invmask, dimsref, dimsimage);
					//d_WriteMRC(d_invmask, dimsimage, "d_invmask.mrc");
					d_FFTR2C(d_invmask, d_invmaskft, &planforw);
				}
				
				if (a == 0 || !ismaskcircular)
				{
					CalcSolventStatistics(d_imageft, d_image2ft, d_invmaskft, h_invmasksum[n], d_solventmean, d_solventstd);
					//d_WriteMRC(d_solventstd, dimsimage, "d_solventstd.mrc");
				}
			}

			int TpB = 256;
			dim3 grid = dim3(tmin((Elements(dimsimage) + TpB - 1) / TpB, 16384), 1, 1);
			ProbPickingUpdateKernel <<<grid, TpB>>> (d_buffer1, 
													(tfloat)1 / ((tfloat)Elements(dimsimage) * h_masksum[n]), 
													d_solventmean, 
													d_solventstd, 
													summaskedref, 
													summaskedref2, 
													expectedpratio, 
													Elements(dimsimage), 
													angle, 
													n, 
													d_bestccf, 
													d_bestangle, 
													d_bestref);

			//d_WriteMRC(d_bestccf, dimsimage, "d_bestccf" + (std::string)(ismaskcircular ? "circ" : "") + ".mrc");
		}

		// Clean up textures and arrays
		cudaDestroyTextureObject(t_refRe);
		cudaDestroyTextureObject(t_refIm);
		cudaFreeArray(a_refRe);
		cudaFreeArray(a_refIm);
		if (!ismaskcircular)
		{
			cudaDestroyTextureObject(t_mask);
			cudaFreeArray(a_mask);
			cudaFree(dt_mask);
		}
	}

	__global__ void ProbPickingUpdateKernel(tfloat* d_corr,
										tfloat normfactor,
										tfloat* d_solventmean,
										tfloat* d_solventstd,
										tfloat summaskedref,
										tfloat summaskedref2,
										double expectedpratio,
										size_t elements,
										tfloat3 angle,
										int refid,
										tfloat* d_bestccf, 
										tfloat3* d_bestangle, 
										int* d_bestref)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
		{
			double diff2 = -2.0 * normfactor * d_corr[id];
			diff2 += 2.0 * d_solventmean[id] * summaskedref;
			diff2 /= tmax(1e-7, d_solventstd[id]);
			diff2 += summaskedref2;
			diff2 = exp(-diff2 / 2.0);
			diff2 = (diff2 - 1.0) / (expectedpratio - 1.0);

			if (diff2 > d_bestccf[id])
			{
				d_bestccf[id] = diff2;
				d_bestangle[id] = angle;
				d_bestref[id] = refid;
			}
		}
	}

	void ProbabilisticPicker::CalcSolventStatistics(tcomplex* d_imageft, tcomplex* d_image2ft, tcomplex* d_solventmaskft, tfloat solventsamples, tfloat* d_solventmean, tfloat* d_solventstd)
	{
		tcomplex* d_buffer;
		cudaMalloc((void**)&d_buffer, ElementsFFT(dimsimage) * sizeof(tcomplex));

		// Calculate mean
		d_ComplexMultiplyByConjVector(d_imageft, d_solventmaskft, d_buffer, ElementsFFT(dimsimage));
		d_IFFTC2R(d_buffer, d_solventmean, &planback);

		// Calculate stddev
		d_ComplexMultiplyByConjVector(d_image2ft, d_solventmaskft, d_buffer, ElementsFFT(dimsimage));
		d_IFFTC2R(d_buffer, d_solventstd, &planback);

		int TpB = 256;
		dim3 grid = dim3(tmin((Elements(dimsimage) + TpB - 1) / TpB, 16384), 1, 1);
		ProbSolventStatisticsKernel << <grid, TpB >> > (d_solventmean, d_solventstd, (tfloat)1 / ((tfloat)Elements(dimsimage) * solventsamples), Elements(dimsimage));
		//d_WriteMRC(d_solventmean, dimsimage, "d_solventmean.mrc");
		//d_WriteMRC(d_solventstd, dimsimage, "d_solventstd.mrc");

		cudaFree(d_buffer);
	}

	__global__ void ProbSolventStatisticsKernel(tfloat* d_mean, tfloat* d_std, tfloat normfactor, size_t elements)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
		{
			tfloat mean = d_mean[id] * normfactor;
			d_mean[id] = mean;

			tfloat std = d_std[id] * normfactor - mean * mean;
			d_std[id] = sqrt(tmax(0, std));
		}
	}
}
