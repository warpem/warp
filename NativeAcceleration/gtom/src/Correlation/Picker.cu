#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Binary.cuh"
#include "gtom/include/Correlation.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Masking.cuh"
#include "gtom/include/Relion.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
	__global__ void PickingUpdateKernel(tfloat* d_corr,
										tfloat normfactor,
										tfloat* d_imagesum1,
										tfloat* d_imagesum2,
										size_t elements,
										tfloat3 angle,
										tfloat* d_bestccf,
										tfloat3* d_bestangle);

	Picker::Picker()
	{
		throw; // Needs to be corrected to use arrays instead of textures
	}

	void Picker::Initialize(tfloat* _d_ref, int3 _dimsref, tfloat* _d_refmask, int3 _dimsimage)
	{
		throw; // Needs to be corrected to use arrays instead of textures

		dimsref = _dimsref;
		ndims = DimensionCount(dimsref);
		dimsimage = _dimsimage;
		
		// (I)FFT plans
		planforw = d_FFTR2CGetPlan(ndims, dimsimage);
		planback = d_IFFTC2RGetPlan(ndims, dimsimage);
		planrefback = d_IFFTC2RGetPlan(ndims, dimsref);

		// Allocate device memory used for each reference's correlation
		cudaMalloc((void**)&d_imagesum1, ElementsFFT(dimsimage) * sizeof(tcomplex));
		cudaMalloc((void**)&d_imagesum2, ElementsFFT(dimsimage) * sizeof(tcomplex));

		cudaMalloc((void**)&d_imageft, ElementsFFT(dimsimage) * sizeof(tcomplex));
		cudaMalloc((void**)&d_image2ft, ElementsFFT(dimsimage) * sizeof(tcomplex));

		cudaMalloc((void**)&d_refft, ElementsFFT(dimsref) * sizeof(tcomplex));
		cudaMalloc((void**)&d_maskft, ElementsFFT(dimsref) * sizeof(tcomplex));

		cudaMalloc((void**)&d_refpadded, ElementsFFT(dimsimage) * sizeof(tcomplex));

		// Prepare normal and inverted masks
		{
			tfloat* d_mask;
			cudaMalloc((void**)&d_mask, ElementsFFT(dimsref) * sizeof(tcomplex));

			// Pad inverse mask to image/volume size and FFT
			d_RemapFull2FullFFT(_d_refmask, d_mask, dimsref);
			d_WriteMRC(d_mask, dimsref, "d_mask.mrc");
			d_FFTR2C(d_mask, (tcomplex*)d_mask, ndims, dimsref);

			tfloat* d_maskRe, *d_maskIm;
			cudaMalloc((void**)&d_maskRe, ElementsFFT(dimsref) * sizeof(tfloat));
			cudaMalloc((void**)&d_maskIm, ElementsFFT(dimsref) * sizeof(tfloat));
			d_ConvertTComplexToSplitComplex((tcomplex*)d_mask, d_maskRe, d_maskIm, ElementsFFT(dimsref));
			
			if (ndims == 3)
			{
				d_BindTextureTo3DArray(d_maskRe, a_maskRe, t_maskRe, toInt3FFT(dimsref), cudaFilterModeLinear, false);
				d_BindTextureTo3DArray(d_maskIm, a_maskIm, t_maskIm, toInt3FFT(dimsref), cudaFilterModeLinear, false);
			}
			else
			{
				d_BindTextureToArray(d_maskRe, a_maskRe, t_maskRe, toInt2FFT(dimsref), cudaFilterModeLinear, false);
				d_BindTextureToArray(d_maskIm, a_maskIm, t_maskIm, toInt2FFT(dimsref), cudaFilterModeLinear, false);
			}

			cudaFree(d_maskIm);
			cudaFree(d_maskRe);
			cudaFree(d_mask);
		}

		// Prepare references
		{
			tfloat* d_ref;
			cudaMalloc((void**)&d_ref, ElementsFFT(dimsref) * sizeof(tcomplex));

			// Pad inverse mask to image/volume size and FFT
			d_RemapFull2FullFFT(_d_ref, d_ref, dimsref);
			//d_WriteMRC(d_maskpadded, dimsimage, "d_invmaskpadded.mrc");
			d_FFTR2C(d_ref, (tcomplex*)d_ref, ndims, dimsref);

			tfloat* d_refRe, *d_refIm;
			cudaMalloc((void**)&d_refRe, ElementsFFT(dimsref) * sizeof(tfloat));
			cudaMalloc((void**)&d_refIm, ElementsFFT(dimsref) * sizeof(tfloat));
			d_ConvertTComplexToSplitComplex((tcomplex*)d_ref, d_refRe, d_refIm, ElementsFFT(dimsref));

			if (ndims == 3)
			{
				d_BindTextureTo3DArray(d_refRe, a_refRe, t_refRe, toInt3FFT(dimsref), cudaFilterModeLinear, false);
				d_BindTextureTo3DArray(d_refIm, a_refIm, t_refIm, toInt3FFT(dimsref), cudaFilterModeLinear, false);
			}
			else
			{
				d_BindTextureToArray(d_refRe, a_refRe, t_refRe, toInt2FFT(dimsref), cudaFilterModeLinear, false);
				d_BindTextureToArray(d_refIm, a_refIm, t_refIm, toInt2FFT(dimsref), cudaFilterModeLinear, false);
			}

			cudaFree(d_refIm);
			cudaFree(d_refRe);
			cudaFree(d_ref);
		}
	}

	Picker::~Picker()
	{
		cufftDestroy(planforw);
		cufftDestroy(planback);
		cufftDestroy(planrefback);

		cudaDestroyTextureObject(t_refRe);
		cudaDestroyTextureObject(t_refIm);
		cudaDestroyTextureObject(t_maskRe);
		cudaDestroyTextureObject(t_maskIm);

		cudaFreeArray(a_refRe);
		cudaFreeArray(a_refIm);
		cudaFreeArray(a_maskRe);
		cudaFreeArray(a_maskIm);

		cudaFree(d_refft);
		cudaFree(d_maskft);
		cudaFree(d_refpadded);
		cudaFree(d_imageft);
		cudaFree(d_image2ft);
		cudaFree(d_imagesum1);
		cudaFree(d_imagesum2);
	}

	void Picker::SetImage(tfloat* _d_image, tfloat* _d_ctf)
	{
		d_ctf = _d_ctf;

		// Normalize image/volume
		d_Norm(_d_image, _d_image, Elements(dimsimage), (tfloat*)NULL, T_NORM_MEAN01STD, 0);
		//d_WriteMRC(d_buffer1, dimsimage, "d_image.mrc");

		// Calc FFT of image, and image^2
		d_FFTR2C(_d_image, d_imageft, &planforw);

		d_Square(_d_image, _d_image, Elements(dimsimage));
		//d_WriteMRC(d_buffer1, dimsimage, "d_image2.mrc");
		d_FFTR2C(_d_image, d_image2ft, &planforw);
	}

	void Picker::PerformCorrelation(tfloat anglestep, tfloat* d_bestccf, tfloat3* d_bestangle)
	{
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

		// For each angle, rotate everything and correlate
		for (uint a = 0; a < angles.size(); a++)
		{
			tfloat3 angle = tfloat3(angles[a]);

			glm::mat3 anglematrix = glm::transpose(Matrix3Euler(angle));
			glm::mat3* d_anglematrix = (glm::mat3*)CudaMallocFromHostArray(&anglematrix, sizeof(glm::mat3));

			// Rotate reference, apply CTF, transform back into real space for padding
			
			// BROKEN!!!!!!
			//
			//
			//d_rlnProject(t_refRe, t_refIm, dimsref, d_refft, dimsref, dimsref.x / 2, d_anglematrix, 1);
			//
			//
			
			d_ComplexMultiplyByVector(d_refft, d_ctf, d_refft, ElementsFFT(dimsref));
			d_IFFTC2R(d_refft, (tfloat*)d_refft, &planrefback, dimsref);
			//d_WriteMRC((tfloat*)d_refft, dimsref, "d_refft.mrc");

			// Rotate mask, transform back into real space for padding
			//d_rlnProject(t_maskRe, t_maskIm, dimsref, d_maskft, dimsref, dimsref.x / 2, d_anglematrix, 1);
			//d_ComplexMultiplyByVector(d_maskft, d_ctf, d_maskft, ElementsFFT(dimsref));
			//ad_IFFTC2R(d_maskft, (tfloat*)d_maskft, &planrefback, dimsref);
			//d_WriteMRC((tfloat*)d_maskft, dimsref, "d_maskft.mrc");

			cudaFree(d_anglematrix);

			// Normalize CTF-corrected reference
			//d_NormMonolithic((tfloat*)d_refft, (tfloat*)d_refft, Elements(dimsref), T_NORM_MEAN01STD, 1);

			// Calculate sum of mask values
			//tfloat* d_masksum = CudaMallocValueFilled(1, (tfloat)0);
			//d_SumMonolithic((tfloat*)d_maskft, d_masksum, Elements(dimsref), 1);
			//tfloat masksum = 0;
			//cudaMemcpy(&masksum, d_masksum, sizeof(tfloat), cudaMemcpyDeviceToHost);
			//cudaFree(d_masksum);

			// Pad mask to image size and go to FT for correlation
			//d_FFTFullPad((tfloat*)d_maskft, d_refpadded, dimsref, dimsimage);
			//d_FFTR2C(d_refpadded, (tcomplex*)d_refpadded, &planforw);

			// Correlate mask with image
			//d_ComplexMultiplyByConjVector(d_imageft, (tcomplex*)d_refpadded, (tcomplex*)d_imagesum1, ElementsFFT(dimsimage));
			//d_IFFTC2R((tcomplex*)d_imagesum1, d_imagesum1, &planback, dimsimage);
			//d_WriteMRC(d_imagesum1, dimsimage, "d_imagesum1.mrc");

			// Correlate mask with image^2
			//d_ComplexMultiplyByConjVector(d_image2ft, (tcomplex*)d_refpadded, (tcomplex*)d_imagesum2, ElementsFFT(dimsimage));
			//d_IFFTC2R((tcomplex*)d_imagesum2, d_imagesum2, &planback, dimsimage);
			//d_WriteMRC(d_imagesum2, dimsimage, "d_imagesum2.mrc");

			d_FFTFullPad((tfloat*)d_refft, d_refpadded, dimsref, dimsimage);
			d_Norm(d_refpadded, d_refpadded, Elements(dimsimage), (tfloat*)NULL, T_NORM_MEAN01STD, 0);
			d_FFTR2C(d_refpadded, (tcomplex*)d_refpadded, &planforw);
			d_ComplexMultiplyByConjVector(d_imageft, (tcomplex*)d_refpadded, (tcomplex*)d_refpadded, ElementsFFT(dimsimage));
			d_IFFTC2R((tcomplex*)d_refpadded, d_refpadded, &planback, dimsimage);
			//d_WriteMRC(d_refpadded, dimsimage, "d_refpadded.mrc");


			int TpB = 128;
			dim3 grid = dim3(tmin((Elements(dimsimage) + TpB - 1) / TpB, 16384), 1, 1);
			PickingUpdateKernel << <grid, TpB >> > (d_refpadded,
													1,
													d_imagesum1,
													d_imagesum2,
													Elements(dimsimage),
													angle,
													d_bestccf,
													d_bestangle);

			//d_WriteMRC(d_bestccf, dimsimage, "d_bestccf.mrc");
		}
	}

	__global__ void PickingUpdateKernel(tfloat* d_corr,
										tfloat normfactor,
										tfloat* d_imagesum1,
										tfloat* d_imagesum2,
										size_t elements,
										tfloat3 angle,
										tfloat* d_bestccf,
										tfloat3* d_bestangle)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
		{
			/*tfloat sum1 = d_imagesum1[id] / normfactor;
			tfloat sum2 = d_imagesum2[id] / normfactor;
			tfloat stddev = sqrt(tmax(0, sum2 - sum1 * sum1));
			if (stddev < 1e-4f)
				stddev = 0;
			else
				stddev = 1.0f / stddev;*/

			tfloat ccf = d_corr[id];// / normfactor * stddev;

			if (ccf > d_bestccf[id])
			{
				d_bestccf[id] = ccf;
				d_bestangle[id] = angle;
			}
		}
	}
}
