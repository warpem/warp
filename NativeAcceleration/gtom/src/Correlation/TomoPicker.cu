#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Binary.cuh"
#include "gtom/include/Correlation.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Projection.cuh"
#include "gtom/include/Reconstruction.cuh"
#include "gtom/include/Relion.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
	__global__ void SolventStatisticsKernel(tfloat* d_mean, tfloat* d_std, tfloat normfactor, size_t elements);
	__global__ void UpdateTomoCorrKernel(tfloat* d_corr, tfloat* d_corrsum1, tfloat* d_corrsum2, tfloat3* d_corrangles, uint3 dimscorr, cudaTex t_image, uint2 dimsimage, tfloat* d_imageweights, uint nimages, tfloat3 angle);

	__constant__ float c_tiltmatrices[128 * 6];

	TomoPicker::TomoPicker()
	{
		throw; // Needs to be corrected to use arrays instead of textures
	}

	void TomoPicker::Initialize(tfloat* _d_ref, int3 _dimsref, tfloat* _d_refmask, bool _ismaskcircular, int2 _dimsimage, uint _nimages)
	{
		throw; // Needs to be corrected to use arrays instead of textures

		dimsref = _dimsref;
		dimsrefpadded = _dimsref * 2;
		dimsimage = _dimsimage;
		ismaskcircular = _ismaskcircular;
		nimages = _nimages;

		// Reference and mask 3D textures
		cudaMalloc((void**)&d_refRe, ElementsFFT(dimsrefpadded) * sizeof(tfloat));
		cudaMalloc((void**)&d_refIm, ElementsFFT(dimsrefpadded) * sizeof(tfloat));
		cudaMalloc((void**)&d_maskRe, ElementsFFT(dimsrefpadded) * sizeof(tfloat));
		cudaMalloc((void**)&d_maskIm, ElementsFFT(dimsrefpadded) * sizeof(tfloat));

		cudaMalloc((void**)&d_refrotated, ElementsFFT(dimsrefpadded) * sizeof(tfloat));
		cudaMalloc((void**)&d_maskrotated, ElementsFFT(dimsrefpadded) * sizeof(tfloat));

		// Reference and mask 2D projections
		cudaMalloc((void**)&d_ref2dft, ElementsFFT2(dimsrefpadded) * nimages * sizeof(tcomplex));
		cudaMalloc((void**)&d_mask2dft, ElementsFFT2(dimsrefpadded) * nimages * sizeof(tcomplex));
		cudaMalloc((void**)&d_ref2d, Elements2(dimsrefpadded) * nimages * sizeof(tfloat));
		cudaMalloc((void**)&d_mask2d, Elements2(dimsrefpadded) * nimages * sizeof(tfloat));
		cudaMalloc((void**)&d_ref2dcropped, Elements2(dimsref) * nimages * sizeof(tfloat));
		cudaMalloc((void**)&d_mask2dcropped, Elements2(dimsref) * nimages * sizeof(tfloat));

		cudaMalloc((void**)&d_masksum, nimages * sizeof(tfloat));

		// (I)FFT plans
		planimageforw = d_FFTR2CGetPlan(2, toInt3(dimsimage), nimages);
		planimageback = d_IFFTC2RGetPlan(2, toInt3(dimsimage), nimages);
		planrefback = d_IFFTC2RGetPlan(2, toInt3(toInt2(dimsrefpadded)), nimages);

		// Local image statistics
		cudaMalloc((void**)&d_imageft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));
		cudaMalloc((void**)&d_image2ft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));
		cudaMalloc((void**)&d_imagesum1, Elements2(dimsimage) * nimages * sizeof(tfloat));
		cudaMalloc((void**)&d_imagesum2, Elements2(dimsimage) * nimages * sizeof(tfloat));
		cudaMalloc((void**)&d_imagecorr, Elements2(dimsimage) * nimages * sizeof(tfloat));

		// Padded 2D mask projections for local image statistics
		cudaMalloc((void**)&d_maskpaddedft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));
		cudaMalloc((void**)&d_maskpadded, Elements2(dimsimage) * nimages * sizeof(tfloat));
		cudaMalloc((void**)&d_maskpaddedcorr, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));

		cudaMalloc((void**)&d_imageweights, nimages * sizeof(tfloat));

		// Pad ref and mask, store their FT
		{
			tfloat* d_padded;
			cudaMalloc((void**)&d_padded, ElementsFFT(dimsrefpadded) * sizeof(tcomplex));

			d_Pad(_d_ref, d_padded, dimsref, dimsrefpadded, T_PAD_VALUE, (tfloat)0);
			d_RemapFull2FullFFT(d_padded, d_padded, dimsrefpadded);
			d_WriteMRC(d_padded, dimsrefpadded, "d_padded.mrc");
			d_FFTR2C(d_padded, (tcomplex*)d_padded, 3, dimsrefpadded);
			d_ConvertTComplexToSplitComplex((tcomplex*)d_padded, d_refRe, d_refIm, ElementsFFT(dimsrefpadded));
			d_BindTextureTo3DArray(d_refRe, a_refRe, t_refRe, toInt3FFT(dimsrefpadded), cudaFilterModeLinear, false);
			d_BindTextureTo3DArray(d_refIm, a_refIm, t_refIm, toInt3FFT(dimsrefpadded), cudaFilterModeLinear, false);
			d_WriteMRC(d_refRe, toInt3FFT(dimsrefpadded), "d_refRe.mrc");
			d_WriteMRC(d_refIm, toInt3FFT(dimsrefpadded), "d_refIm.mrc");

			d_Pad(_d_refmask, d_padded, dimsref, dimsrefpadded, T_PAD_VALUE, (tfloat)0);
			d_RemapFull2FullFFT(d_padded, d_padded, dimsrefpadded);
			d_WriteMRC(d_padded, dimsrefpadded, "d_padded.mrc");
			d_FFTR2C(d_padded, (tcomplex*)d_padded, 3, dimsrefpadded);
			d_ConvertTComplexToSplitComplex((tcomplex*)d_padded, d_maskRe, d_maskIm, ElementsFFT(dimsrefpadded));
			d_BindTextureTo3DArray(d_maskRe, a_maskRe, t_maskRe, toInt3FFT(dimsrefpadded), cudaFilterModeLinear, false);
			d_BindTextureTo3DArray(d_maskIm, a_maskIm, t_maskIm, toInt3FFT(dimsrefpadded), cudaFilterModeLinear, false);
			d_WriteMRC(d_maskRe, toInt3FFT(dimsrefpadded), "d_maskRe.mrc");
			d_WriteMRC(d_maskIm, toInt3FFT(dimsrefpadded), "d_maskIm.mrc");

			cudaFree(d_padded);
		}
	}

	TomoPicker::~TomoPicker()
	{
		cufftDestroy(planimageforw);
		cufftDestroy(planimageback);
		cufftDestroy(planrefback);

		cudaDestroyTextureObject(t_refRe);
		cudaDestroyTextureObject(t_refIm);
		cudaDestroyTextureObject(t_maskRe);
		cudaDestroyTextureObject(t_maskIm);

		cudaFreeArray(a_refRe);
		cudaFreeArray(a_refIm);
		cudaFreeArray(a_maskRe);
		cudaFreeArray(a_maskIm);

		cudaFree(d_refRe);
		cudaFree(d_refIm);
		cudaFree(d_maskRe);
		cudaFree(d_maskIm);

		cudaFree(d_refrotated);
		cudaFree(d_maskrotated);

		cudaFree(d_ref2dft);
		cudaFree(d_mask2dft);
		cudaFree(d_ref2d);
		cudaFree(d_mask2d);
		cudaFree(d_ref2dcropped);
		cudaFree(d_mask2dcropped);

		cudaFree(d_masksum);

		cudaFree(d_imageft);
		cudaFree(d_image2ft);
		cudaFree(d_imagesum1);
		cudaFree(d_imagesum2);
		cudaFree(d_imagecorr);

		cudaFree(d_maskpaddedft);
		cudaFree(d_maskpadded);

		cudaFree(d_imageweights);
	}

	void TomoPicker::SetImage(tfloat* _d_image, tfloat3* _h_imageangles, tfloat* _h_imageweights)
	{
		d_image = _d_image;
		h_imageangles = _h_imageangles;
		d_WriteMRC(d_image, toInt3(dimsimage.x, dimsimage.y, nimages), "d_image.mrc");

		cudaMemcpy(d_imageweights, _h_imageweights, nimages * sizeof(tfloat), cudaMemcpyHostToDevice);

		d_FFTR2C(d_image, d_imageft, &planimageforw);
		d_WriteMRC((tfloat*)d_imageft, toInt3((dimsimage.x / 2 + 1) * 2, dimsimage.y, nimages), "d_imageft.mrc");

		tfloat* d_filter = CudaMallocValueFilled(ElementsFFT2(dimsimage) * nimages, (tfloat)1);

		int* h_indices = MallocValueFilled(nimages, 0);
		for (int i = 0; i < nimages; i++)
			h_indices[i] = i;

		d_Exact2DWeighting(d_filter, dimsimage, h_indices, h_imageangles, d_imageweights, nimages, dimsimage.x, false, nimages);
		d_WriteMRC(d_filter, toInt3(dimsimage.x / 2 + 1, dimsimage.y, nimages), "d_filter.mrc");

		d_FFTR2C(d_image, d_imageft, &planimageforw);
		d_ComplexMultiplyByVector(d_imageft, d_filter, d_imageft, ElementsFFT2(dimsimage) * nimages);
		d_IFFTC2R(d_imageft, d_image, &planimageback, toInt3(dimsimage), nimages);
		d_WriteMRC(d_image, toInt3(dimsimage.x, dimsimage.y, nimages), "d_image.mrc");

		free(h_indices);
		cudaFree(d_filter);

		d_Square(d_image, d_imagesum2, Elements2(dimsimage) * nimages);
		d_FFTR2C(d_imagesum2, d_image2ft, &planimageforw);
		d_WriteMRC((tfloat*)d_image2ft, toInt3((dimsimage.x / 2 + 1) * 2, dimsimage.y, nimages), "d_image2ft.mrc");
	}

	void TomoPicker::PerformCorrelation(tfloat* d_corr, tfloat3* d_corrangle, int3 dimscorr, tfloat anglestep)
	{
		// Get all relevant angles
		std::vector<float3> angles;
		angles = GetEqualAngularSpacing(make_float2(0, PI2), make_float2(0, PI), make_float2(0, PI2), anglestep);

		glm::mat3x2* d_transforms;
		{
			glm::mat3x2* h_transforms = (glm::mat3x2*)malloc(nimages * sizeof(glm::mat3x2));
			for (int b = 0; b < nimages; b++)
			{
				glm::mat3 temp = Matrix3Euler(h_imageangles[b]);
				h_transforms[b] = glm::mat3x2(glm::vec2(temp[0].x, temp[0].y), glm::vec2(temp[1].x, temp[1].y), glm::vec2(temp[2].x, temp[2].y));
			}

			d_transforms = (glm::mat3x2*)CudaMallocFromHostArray(h_transforms, nimages * sizeof(glm::mat3x2));
			free(h_transforms);

			cudaMemcpyToSymbol(c_tiltmatrices, d_transforms, nimages * sizeof(glm::mat3x2), 0, cudaMemcpyDeviceToDevice);
		}

		tfloat* d_corrsum1 = CudaMallocValueFilled(Elements(dimscorr), (tfloat)0);
		tfloat* d_corrsum2 = CudaMallocValueFilled(Elements(dimscorr), (tfloat)0);

		// For each angle, rotate everything and correlate
		for (uint a = 0; a < angles.size(); a++)
		{
			tfloat3 angle = angles[a];
			glm::mat3* d_matrices;
			{
				glm::mat3* h_matrices = (glm::mat3*)malloc(nimages * sizeof(glm::mat3));
				glm::mat3 anglematrix = glm::transpose(Matrix3Euler(angle));
				for (int i = 0; i < nimages; i++)
					h_matrices[i] = anglematrix * glm::transpose(Matrix3Euler(h_imageangles[i]));
				d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, nimages * sizeof(glm::mat3));
				free(h_matrices);
			}

			if (a == 0 || !ismaskcircular)
			{
				//d_rlnProject(t_maskRe, t_maskIm, dimsrefpadded, d_maskrotated, dimsrefpadded, &angle, 1);
				
				// BROKEN!!!!!!!
				//
				//
				//d_rlnProject(t_maskRe, t_maskIm, dimsrefpadded, d_mask2dft, toInt3(toInt2(dimsrefpadded)), dimsrefpadded.x / 2, d_matrices, nimages);
				//
				//
				//

				d_IFFTC2R(d_mask2dft, d_mask2d, &planrefback, toInt3(toInt2(dimsrefpadded)), nimages);
				d_FFTFullCrop(d_mask2d, d_mask2d, toInt3(toInt2(dimsrefpadded)), toInt3(toInt2(dimsref)), nimages);
				//d_WriteMRC(d_mask2d, toInt3(dimsref.x, dimsref.x, nimages), "d_mask2d.mrc");

				// Re-binarize and count number of samples
				d_Binarize(d_mask2d, d_mask2d, Elements2(dimsref) * nimages, (tfloat)0.99);
				//d_WriteMRC(d_mask2d, toInt3(dimsref.x, dimsref.x, nimages), "d_mask2d.mrc");
				d_SumMonolithic(d_mask2d, d_masksum, Elements2(dimsref), nimages);
				tfloat* h_masksum = (tfloat*)MallocFromDeviceArray(d_masksum, nimages * sizeof(tfloat));
				free(h_masksum);

				// Pad and FFT for correlation with images
				d_FFTFullPad(d_mask2d, d_maskpadded, toInt3(toInt2(dimsref)), toInt3(dimsimage), nimages);
				//d_WriteMRC(d_maskpadded, toInt3(dimsimage.x, dimsimage.x, nimages), "d_maskpadded.mrc");
				d_FFTR2C(d_maskpadded, d_maskpaddedft, &planimageforw);

				// Remap back to zero-centered
				d_RemapFullFFT2Full(d_mask2d, d_mask2d, toInt3(toInt2(dimsref)), nimages);
				//d_WriteMRC(d_mask2d, toInt3(dimsref.x, dimsref.x, nimages), "d_mask2d.mrc");

				// Local image statistics under the mask
				d_ComplexMultiplyByConjVector(d_imageft, d_maskpaddedft, d_maskpaddedcorr, ElementsFFT2(dimsimage) * nimages);
				d_IFFTC2R(d_maskpaddedcorr, d_imagesum1, &planimageback, toInt3(dimsimage), nimages);
				//d_WriteMRC(d_imagesum1, toInt3(dimsimage.x, dimsimage.y, nimages), "d_imagesum1.mrc");

				d_ComplexMultiplyByConjVector(d_image2ft, d_maskpaddedft, d_maskpaddedcorr, ElementsFFT2(dimsimage) * nimages);
				d_IFFTC2R(d_maskpaddedcorr, d_imagesum2, &planimageback, toInt3(dimsimage), nimages);
				//d_WriteMRC(d_imagesum2, toInt3(dimsimage.x, dimsimage.y, nimages), "d_imagesum2.mrc");
			}

			// Project reference at tilt angles
			//d_rlnProject(t_refRe, t_refIm, dimsrefpadded, d_refrotated, dimsrefpadded, &angle, 1);

			// BROKEN!!!!!
			//
			//
			//d_rlnProject(t_refRe, t_refIm, dimsrefpadded, d_ref2dft, toInt3(toInt2(dimsrefpadded)), dimsrefpadded.x / 2, d_matrices, nimages);
			//
			//

			cudaFree(d_matrices);
			d_IFFTC2R(d_ref2dft, d_ref2d, &planrefback);
			d_FFTFullCrop(d_ref2d, d_ref2d, toInt3(toInt2(dimsrefpadded)), toInt3(toInt2(dimsref)), nimages);
			d_RemapFullFFT2Full(d_ref2d, d_ref2d, toInt3(toInt2(dimsref)), nimages);

			// Normalize reference projections within mask
			d_NormMonolithic(d_ref2d, d_ref2d, Elements2(dimsref), d_mask2d, T_NORM_MEAN01STD, nimages);
			//d_WriteMRC(d_ref2d, toInt3(dimsref.x, dimsref.y, nimages), "d_ref2d.mrc");

			// Correlate reference projections with tomographic tilt series
			//d_CorrelateRealspace(d_image, d_imagesum1, d_imagesum2, dimsimage, d_ref2d, d_mask2d, toInt2(dimsref), d_masksum, d_imagecorr, nimages); BROKEN!!!!!
			//d_WriteMRC(d_imagecorr, toInt3(dimsimage.x, dimsimage.y, nimages), "d_imagecorr.mrc");
			
			// Back-project correlation values into tomographic volume
			{
				cudaArray_t a_imagecorr;
				cudaTex t_imagecorr;

				d_BindTextureTo3DArray(d_imagecorr, a_imagecorr, t_imagecorr, toInt3(dimsimage.x, dimsimage.y, nimages), cudaFilterModeLinear, false);

				dim3 TpB = dim3(128, 1, 1);
				dim3 grid = dim3((dimscorr.x + TpB.x - 1) / TpB.x, dimscorr.y, dimscorr.z);

				UpdateTomoCorrKernel <<<grid, TpB>>> (d_corr, d_corrsum1, d_corrsum2, d_corrangle, toUint3(dimscorr), t_imagecorr, toUint2(dimsimage), d_imageweights, nimages, angle);

				//d_WriteMRC(d_corr, dimscorr, "d_corr.mrc");

				cudaDestroyTextureObject(t_imagecorr);
				cudaFreeArray(a_imagecorr);
			}

			std::cout << (float)a / (float)angles.size() << std::endl;
		}

		d_WriteMRC(d_corrsum1, dimscorr, "d_corrsum1.mrc");
		d_WriteMRC(d_corrsum2, dimscorr, "d_corrsum2.mrc");

		cudaFree(d_transforms);
	}

	__global__ void UpdateTomoCorrKernel(tfloat* d_corr, tfloat* d_corrsum1, tfloat* d_corrsum2, tfloat3* d_corrangles, uint3 dimscorr, cudaTex t_image, uint2 dimsimage, tfloat* d_imageweights, uint nimages, tfloat3 angle)
	{
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dimscorr.x)
			return;
		uint idy = blockIdx.y * blockDim.y;
		uint idz = blockIdx.z * blockDim.z;

		uint outx, outy, outz;
		outx = idx;
		outy = idy;
		outz = idz;

		glm::vec3 position = glm::vec3((float)idx - dimscorr.x / 2, (float)idy - dimscorr.y / 2, (float)idz - dimscorr.z / 2);
		glm::vec2 imagecenter = glm::vec2(dimsimage.x / 2 + 0.5f, dimsimage.y / 2 + 0.5f);
		tfloat sum = 0.0f;
		tfloat samples = 0;

		for (uint n = 0; n < nimages; n++)
		{
			glm::mat3x2 transform = glm::mat3x2(glm::vec2(c_tiltmatrices[n * 6 + 0], c_tiltmatrices[n * 6 + 1]),
												glm::vec2(c_tiltmatrices[n * 6 + 2], c_tiltmatrices[n * 6 + 3]),
												glm::vec2(c_tiltmatrices[n * 6 + 4], c_tiltmatrices[n * 6 + 5]));
			glm::vec2 positiontrans = transform * position + imagecenter;

			if (positiontrans.x >= 0 && positiontrans.x <= dimsimage.x && positiontrans.y >= 0 && positiontrans.y <= dimsimage.y)
			{
				tfloat weight = d_imageweights[n];
				sum += tex3D<tfloat>(t_image, positiontrans.x, positiontrans.y, (float)n + 0.5f) * weight;
				samples += weight;
			}
		}

		if (samples > 0)
		{
			sum /= (tfloat)samples;
			size_t offset = (outz * dimscorr.y + outy) * dimscorr.x + outx;
			tfloat oldval = d_corr[offset];
			if (oldval < sum)
			{
				d_corr[offset] = sum;
				d_corrangles[offset] = angle;
			}

			d_corrsum1[offset] += sum;
			d_corrsum2[offset] += sum * sum;
		}
	}
}
