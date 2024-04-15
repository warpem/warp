#include "cufft.h"
#include "Prerequisites.cuh"

#ifndef MASKING_CUH
#define MASKING_CUH

namespace gtom
{
	///////////
	//Masking//
	///////////

	//ConeMask.cu:
	template <class T> void d_ConeMaskFT(T* d_input, T* d_output, int3 dims, float3 direction, float coneangle, int batch = 1);

	//IrregularSphereMask.cu:
	template <class T> void d_IrregularSphereMask(T* d_input, T* d_output, int3 dims, tfloat* radiusmap, int2 anglesteps, tfloat sigma, tfloat3* center, int batch = 1);

	//SphereMask.cu:
	template <class T> void d_SphereMask(T* d_input, T* d_output, int3 size, tfloat* radius, tfloat sigma, tfloat3* center, bool decentered, int batch = 1);
	void d_SphereMaskFT(tfloat* d_input, tfloat* d_output, int3 dims, int radius, uint batch = 1);

	//RectangleMask.cu:
	template <class T> void d_RectangleMask(T* d_input, T* d_output, int3 dimsmask, int3 dimsbox, int3* center, int batch = 1);

	//Remap.cu:
	template <class T> void d_Remap(T* d_input, size_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
	template <class T> void d_RemapReverse(T* d_input, size_t* d_map, T* d_output, size_t elementsmapped, size_t elementsdestination, T defvalue, int batch = 1);
	template <class T> void h_Remap(T* h_input, size_t* h_map, T* h_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
	template <class T> void d_MaskSparseToDense(T* d_input, size_t** d_mapforward, size_t* d_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	template <class T> void h_MaskSparseToDense(T* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	void d_RemapInterpolated2D(tfloat* d_input, int2 dimsinput, tfloat* d_output, float2* d_addresses, int n, T_INTERP_MODE mode);

	//Windows.cu:
	void d_HannMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* radius, tfloat3* center, int batch = 1);
	void d_HammingMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* radius, tfloat3* center, int batch = 1);
	void d_GaussianMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* sigma, tfloat3* center, int batch = 1);
	void d_HannMaskBorderDistance(tfloat* d_input, tfloat* d_output, int3 dims, int falloff, int batch = 1);
	void d_HammingMaskBorderDistance(tfloat* d_input, tfloat* d_output, int3 dims, int falloff, int batch = 1);
}
#endif