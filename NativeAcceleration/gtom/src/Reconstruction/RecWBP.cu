#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Projection.cuh"
#include "gtom/include/Reconstruction.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	////////////////////////////////////////////////////////////
	//Performs 3D reconstruction using Weighted Backprojection//
	////////////////////////////////////////////////////////////

	void d_RecWBP(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, int nimages, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, tfloat* h_weights, T_INTERP_MODE mode, bool outputzerocentered)
	{
		int3 dimspadded = toInt3(dimsimage.x * 1, dimsimage.y * 1, 1);

		/*tfloat* d_paddedimage;
		cudaMalloc((void**)&d_paddedimage, Elements2(dimspadded) * nimages * sizeof(tfloat));
		d_Pad(d_image, d_paddedimage, toInt3(dimsimage), dimspadded, T_PAD_VALUE, 0.0f, nimages);*/

		int* h_indices = (int*)malloc(nimages * sizeof(int));
		for (int n = 0; n < nimages; n++)
			h_indices[n] = n;

		tfloat* d_weights = h_weights != NULL ? (tfloat*)CudaMallocFromHostArray(h_weights, nimages * sizeof(tfloat)) : CudaMallocValueFilled(nimages, (tfloat)1);

		tcomplex* d_imageft;
		cudaMalloc((void**)&d_imageft, ElementsFFT(dimspadded) * nimages * sizeof(tcomplex));

		tfloat* d_weighted;
		cudaMalloc((void**)&d_weighted, Elements2(dimsimage) * nimages * sizeof(tfloat));

		size_t memlimit = 2048 << 20;
		int ftbatch = memlimit / (Elements2(dimsimage) * sizeof(tfloat) * 6);

		for (int b = 0; b < nimages; b += ftbatch)
			d_FFTR2C(d_image + Elements2(dimsimage) * b, d_imageft + ElementsFFT2(dimsimage) * b, 2, dimspadded, tmin(nimages - b, ftbatch));

		d_Exact2DWeighting(d_imageft, toInt2(dimspadded), h_indices, h_angles, d_weights, nimages, dimspadded.x, false, nimages);

		for (int b = 0; b < nimages; b += ftbatch)
			d_IFFTC2R(d_imageft + ElementsFFT2(dimsimage) * b, d_weighted + Elements2(dimsimage) * b, 2, dimspadded, tmin(nimages - b, ftbatch));
		free(h_indices);
		cudaFree(d_imageft);

		//d_Pad(d_paddedimage, (tfloat*)d_imageft, dimspadded, toInt3(dimsimage), T_PAD_VALUE, 0.0f, nimages);

		d_ValueFill(d_volume, Elements(dimsvolume), 0.0f);
		d_ProjBackward(d_volume, dimsvolume, offsetfromcenter, d_weighted, dimsimage, h_angles, h_offsets, h_scales, mode, outputzerocentered, nimages);

		/*tfloat* d_points = CudaMallocValueFilled(nimages, (tfloat)1);
		tfloat* d_pointsstack;
		cudaMalloc((void**)&d_pointsstack, Elements2(dimsimage) * nimages * sizeof(tfloat));
		d_Pad(d_points, d_pointsstack, toInt3(1, 1, 1), toInt3(dimsimage), T_PAD_VALUE, (tfloat)0, nimages);
		cudaFree(d_points);

		d_MultiplyByScalar(d_pointsstack, d_weights, d_pointsstack, Elements2(dimsimage), nimages);

		tfloat* d_psf = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		d_ProjBackward(d_psf, dimsvolume, offsetfromcenter, d_pointsstack, dimsimage, h_angles, h_offsets, h_scales, mode, outputzerocentered, nimages);

		tcomplex* d_psfft;
		cudaMalloc((void**)&d_psfft, ElementsFFT(dimsvolume) * sizeof(tcomplex));
		d_FFTR2C(d_psf, d_psfft, 3, dimsvolume);
		d_Abs(d_psfft, d_psf, ElementsFFT(dimsvolume));
		//d_DivideByScalar(d_psf, d_psf, ElementsFFT(dimsvolume), (tfloat)4);
		d_MaxOp(d_psf, (tfloat)1, d_psf, ElementsFFT(dimsvolume));

		d_FFTR2C(d_volume, d_psfft, 3, dimsvolume);
		d_ComplexDivideByVector(d_psfft, d_psf, d_psfft, ElementsFFT(dimsvolume));
		d_IFFTC2R(d_psfft, d_volume, 3, dimsvolume);

		cudaFree(d_psfft);
		cudaFree(d_psf);

		cudaFree(d_pointsstack);*/

		cudaFree(d_weighted);
		cudaFree(d_weights);
	}
}