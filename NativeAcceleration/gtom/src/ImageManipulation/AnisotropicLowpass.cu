#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Masking.cuh"


namespace gtom
{
	///////////////////////////////////////////
	//Equivalent of TOM's tom_bandpass method//
	///////////////////////////////////////////

	void d_AnisotropicLowpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* d_radiusmap, int2 anglesteps, tfloat smooth, cufftHandle* planforw, cufftHandle* planback, int batch)
	{
		int dimensions = DimensionCount(dims);

		//Prepare mask:

		tfloat* d_mask = (tfloat*)CudaMallocValueFilled(Elements(dims), (tfloat)1);

		d_IrregularSphereMask(d_mask, d_mask, dims, d_radiusmap, anglesteps, smooth, (tfloat3*)NULL, 1);

		tfloat* d_maskFFT;
		cudaMalloc((void**)&d_maskFFT, ElementsFFT(dims) * sizeof(tfloat));
		d_RemapFull2HalfFFT(d_mask, d_maskFFT, dims);
		cudaFree(d_mask);

		//Forward FFT:

		tcomplex* d_inputFFT;
		cudaMalloc((void**)&d_inputFFT, ElementsFFT(dims) * sizeof(tcomplex));

		if (planforw == NULL)
			d_FFTR2C(d_input, d_inputFFT, dimensions, dims, batch);
		else
			d_FFTR2C(d_input, d_inputFFT, planforw);

		//Mask FFT:

		d_ComplexMultiplyByVector(d_inputFFT, d_mask, d_inputFFT, ElementsFFT(dims), batch);

		//Inverse FFT:

		if (planforw == NULL)
			d_IFFTC2R(d_inputFFT, d_output, dimensions, dims, batch);
		else
			d_IFFTC2R(d_inputFFT, d_output, planback);

		cudaFree(d_inputFFT);
		cudaFree(d_maskFFT);
	}
}