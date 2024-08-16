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
	void d_ReconstructGridding(tcomplex* d_dataft, tfloat* d_weight, tfloat* d_reconstructed, int3 dimsori, int3 dimspadded, int paddingfactor = 2, cufftHandle pre_planforw = NULL, cufftHandle pre_planback = NULL, int iterations = 10, double blobradius = 1.9, int bloborder = 0, double blobalpha = 15);
	
	//Weighting.cu:
	template <class T> void d_Exact2DWeighting(T* d_weights, int2 dimsimage, int* h_indices, tfloat3* h_angles, tfloat* d_imageweights, int nimages, tfloat maxfreq, bool iszerocentered, int batch = 1);
	template <class T> void d_Exact3DWeighting(T* d_weights, int3 dimsvolume, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered);
}
#endif