#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Correlation.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/ImageManipulation.cuh"


namespace gtom
{
	//////////////////////////////////////
	//Equivalent of TOM's tom_ccf method//
	//////////////////////////////////////

	template<class T> void d_CCF(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, T* d_mask, int batch)
	{
		d_CCFUnshifted(d_input1, d_input2, d_output, dims, normalized, d_mask, batch);
		d_RemapFullFFT2Full(d_output, d_output, dims, batch);
	}
	template void d_CCF<tfloat>(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, tfloat* d_mask, int batch);
	template void d_CCF<int>(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, int* d_mask, int batch);
	template void d_CCF<char>(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, char* d_mask, int batch);

	////////////////////////////////////////////////////////////////////////////
	//Equivalent of TOM's tom_ccf method, but without the fftshift at the end//
	////////////////////////////////////////////////////////////////////////////

	template<class T> void d_CCFUnshifted(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, T* d_mask, int batch)
	{
		size_t elements = dims.x * dims.y * dims.z;
		size_t elementsFFT = (dims.x / 2 + 1) * dims.y * dims.z;

		tfloat* d_intermediate1;
		cudaMalloc((void**)&d_intermediate1, elementsFFT * sizeof(tcomplex));
		tfloat* d_intermediate2;
		cudaMalloc((void**)&d_intermediate2, elementsFFT * sizeof(tcomplex));

		for (int b = 0; b < batch; b++)
		{
			if (normalized)
			{
				d_Norm(d_input1, d_intermediate1, elements, d_mask, T_NORM_MEAN01STD, 0);
				d_Norm(d_input2, d_intermediate2, elements, d_mask, T_NORM_MEAN01STD, 0);

				d_FFTR2C(d_intermediate1, (tcomplex*)d_intermediate1, DimensionCount(dims), dims);
				d_FFTR2C(d_intermediate2, (tcomplex*)d_intermediate2, DimensionCount(dims), dims);

			}
			else
			{
				d_FFTR2C(d_input1, (tcomplex*)d_intermediate1, DimensionCount(dims), dims);
				d_FFTR2C(d_input2, (tcomplex*)d_intermediate2, DimensionCount(dims), dims);
			}

			d_ComplexMultiplyByConjVector((tcomplex*)d_intermediate2, (tcomplex*)d_intermediate1, (tcomplex*)d_intermediate1, elementsFFT);

			d_IFFTC2R((tcomplex*)d_intermediate1, d_output, DimensionCount(dims), dims);
		}

		cudaFree(d_intermediate1);
		cudaFree(d_intermediate2);

		d_MultiplyByScalar(d_output, d_output, elements * batch, (tfloat)1 / (tfloat)elements);
	}
	template void d_CCFUnshifted<tfloat>(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, tfloat* d_mask, int batch);
	template void d_CCFUnshifted<int>(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, int* d_mask, int batch);
	template void d_CCFUnshifted<char>(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, char* d_mask, int batch);
}