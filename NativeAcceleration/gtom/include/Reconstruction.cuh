#include "cufft.h"
#include "Prerequisites.cuh"

#ifndef RECONSTRUCTION_CUH
#define RECONSTRUCTION_CUH

namespace gtom
{
	//////////////////
	//Reconstruction//
	//////////////////

	//RecFourier.cu:
	void d_ReconstructFourier(tcomplex* d_imagesft, tfloat* d_imagespsf, tcomplex* d_volumeft, tfloat* d_volumepsf, int3 dims, tfloat3* h_angles, tfloat2* h_shifts, int nimages, bool performgridding, bool everythingcentered);
	void d_ReconstructFourierPrecise(tfloat* d_images, tfloat* d_imagespsf, tfloat* d_volume, tfloat* d_volumepsf, int3 dims, tfloat3* h_angles, tfloat2* h_shifts, int nimages, bool dogridding);
	void d_ReconstructGridding(tcomplex* d_dataft, tfloat* d_weight, tfloat* d_reconstructed, int3 dimsori, int3 dimspadded, int paddingfactor = 2, cufftHandle pre_planforw = NULL, cufftHandle pre_planback = NULL, int iterations = 10, double blobradius = 1.9, int bloborder = 0, double blobalpha = 15);
	void d_ReconstructFourierAdd(tcomplex* d_volumeft, tfloat* d_volumepsf, int3 dims, tcomplex* d_imagesft, tfloat* d_imagespsf, tfloat3* h_angles, tfloat2* h_shifts, int nimages);
	void d_ReconstructFourierPreciseAdd(tcomplex* d_volumeft, tfloat* d_samples, int3 dims, tfloat* d_images, tfloat* d_imagespsf, tfloat3* h_angles, tfloat2* h_shifts, int nimages, T_INTERP_MODE mode, bool outputzerocentered = true, bool finalize = false);
	void d_ReconstructFourierSincAdd(tcomplex* d_volumeft, tfloat* d_samples, int3 dims, tcomplex* d_imagesft, tfloat* d_imagespsf, tfloat3* h_angles, tfloat2* h_shifts, int nimages, bool outputzerocentered, bool finalize);
	void d_ReconstructFourierPreciseAdd(tcomplex* d_volumeft, tfloat* d_samples, int3 dims, cudaTex* t_imageftRe, cudaTex* t_imageftIm, cudaTex* t_imageweights, tfloat3* h_angles, tfloat2* h_shifts, int nimages, bool outputzerocentered = true, bool finalize = false);

	//RecSIRT.cu:
	void d_RecSIRT(tfloat* d_volume, tfloat* d_residual, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, int nimages, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, tfloat2* h_intensities, T_INTERP_MODE mode, int supersample, int iterations, bool outputzerocentered);

	//RecWBP.cu:
	void d_RecWBP(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, int nimages, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, tfloat* h_weights, T_INTERP_MODE mode, bool outputzerocentered);

	//Weighting.cu:
	template <class T> void d_Exact2DWeighting(T* d_weights, int2 dimsimage, int* h_indices, tfloat3* h_angles, tfloat* d_imageweights, int nimages, tfloat maxfreq, bool iszerocentered, int batch = 1);
	template <class T> void d_Exact3DWeighting(T* d_weights, int3 dimsvolume, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered);
}
#endif