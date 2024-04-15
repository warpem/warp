#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Masking.cuh"


namespace gtom
{
	///////////////////////////
	//CUDA kernel declaration//
	///////////////////////////


	/////////////////
	//Local Lowpass//
	/////////////////

	void d_LocalLowpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* d_resolution, tfloat maxprecision)
	{
		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * sizeof(tcomplex));
		d_ValueFill(d_output, Elements(dims), (tfloat)0);
		tcomplex* d_maskedft;
		cudaMalloc((void**)&d_maskedft, ElementsFFT(dims) * sizeof(tcomplex));
		tfloat* d_cleanresolution;
		cudaMalloc((void**)&d_cleanresolution, Elements(dims) * sizeof(tfloat));
		tfloat* d_mask;
		cudaMalloc((void**)&d_mask, Elements(dims) * sizeof(tfloat));
		tfloat* d_maskhalf;
		cudaMalloc((void**)&d_maskhalf, ElementsFFT(dims) * sizeof(tfloat));

		d_Xray(d_resolution, d_cleanresolution, dims);
		d_FFTR2C(d_input, d_inputft, DimensionCount(dims), dims);

		imgstats5* d_resstats = (imgstats5*)CudaMallocValueFilled(5, 0);
		d_Dev(d_cleanresolution, d_resstats, Elements(dims), (char*)NULL);
		imgstats5* h_resstats = (imgstats5*)MallocFromDeviceArray(d_resstats, 5 * sizeof(tfloat));

		tfloat minval = h_resstats[0].min;
		tfloat maxval = h_resstats[0].max;

		tfloat* d_tempsum = CudaMallocValueFilled(1, (tfloat)0);
		d_Sum(d_input, d_tempsum, Elements(dims));
		tfloat* h_tempsum = (tfloat*)MallocFromDeviceArray(d_tempsum, sizeof(tfloat));

		tfloat originalmean = *h_tempsum / (tfloat)Elements(dims);

		int bins = (int)min((maxval - minval) / maxprecision, (tfloat)4096);
		tfloat binsize = (maxval - minval) / (tfloat)bins;
		uint* d_histogram = CudaMallocValueFilled(bins, (uint)0);
		d_Histogram(d_cleanresolution, d_histogram, Elements(dims), bins, minval, maxval);
		uint* h_histogram = (uint*)MallocFromDeviceArray(d_histogram, bins * sizeof(uint));

		cufftHandle planback = d_IFFTC2RGetPlan(DimensionCount(dims), dims);

		for (int b = 0; b < bins; b++)
		{
			if (h_histogram[b] == 0)
				continue;

			tfloat res = (tfloat)b * binsize + minval;
			tfloat freq = (tfloat)dims.x / res;

			d_ValueFill(d_mask, Elements(dims), (tfloat)1);
			d_SphereMask(d_mask, d_mask, dims, &freq, 0, NULL, false);
			d_RemapFull2HalfFFT(d_mask, d_maskhalf, dims);

			d_ComplexMultiplyByVector(d_inputft, d_maskhalf, d_maskedft, ElementsFFT(dims));
			d_IFFTC2R(d_maskedft, d_mask, &planback, dims);

			d_IsBetween(d_resolution, (tfloat*)d_maskedft, Elements(dims), (tfloat)b * binsize + minval, (tfloat)(b + 1) * binsize + minval);
			d_MultiplyByVector(d_mask, (tfloat*)d_maskedft, d_mask, Elements(dims));
			d_AddVector(d_output, d_mask, d_output, Elements(dims));
		}


		free(h_histogram);
		cudaFree(d_histogram);
		free(h_tempsum);
		cudaFree(d_tempsum);
		free(h_resstats);
		cudaFree(d_resstats);
		cudaFree(d_maskhalf);
		cudaFree(d_mask);
		cudaFree(d_cleanresolution);
		cudaFree(d_maskedft);
		cudaFree(d_inputft);
	}


	////////////////
	//CUDA kernels//
	////////////////

}