#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Masking.cuh"


namespace gtom
{
	void d_LocalStd(tfloat* d_map, int3 dimsmap, tfloat* d_fouriermask, tfloat localradius, tfloat* d_std, tfloat* d_mean, cufftHandle planforw, cufftHandle planback)
	{
		cufftHandle localplanforw = planforw;
		if (planforw == 0)
			localplanforw = d_FFTR2CGetPlan(DimensionCount(dimsmap), dimsmap, 1);
		cufftHandle localplanback = planback;
		if (planback == 0)
			localplanback = d_IFFTC2RGetPlan(DimensionCount(dimsmap), dimsmap, 1);

		tcomplex* d_maskft = CudaMallocValueFilled(ElementsFFT(dimsmap), make_cuComplex(1, 1));
		tfloat masksum = 0;

		// Create spherical mask, calculate its sum, and pre-FFT it for convolution
		{
			tfloat* d_mask = CudaMallocValueFilled(Elements(dimsmap), (tfloat)1);
			d_SphereMask(d_mask, d_mask, dimsmap, &localradius, 1, NULL, false);
			d_RemapFull2FullFFT(d_mask, d_mask, dimsmap);

			if (d_fouriermask == 0)
			{
				tfloat* d_sum = CudaMallocValueFilled(1, (tfloat)0);
				d_Sum(d_mask, d_sum, Elements(dimsmap));
				cudaMemcpy(&masksum, d_sum, sizeof(tfloat), cudaMemcpyDeviceToHost);
				cudaFree(d_sum);
			}

			d_FFTR2C(d_mask, d_maskft, &localplanforw);
			cudaFree(d_mask);

			if (d_fouriermask != NULL)
			{
				d_ComplexMultiplyByVector(d_maskft, d_fouriermask, d_maskft, ElementsFFT(dimsmap));

				tfloat* d_maskconv;
				cudaMalloc((void**)&d_maskconv, Elements(dimsmap) * sizeof(tfloat));

				d_IFFTC2R(d_maskft, d_maskconv, DimensionCount(dimsmap), dimsmap, 1, true);
				
				tfloat* d_sum = CudaMallocValueFilled(1, (tfloat)0);
				d_Sum(d_maskconv, d_sum, Elements(dimsmap));
				cudaMemcpy(&masksum, d_sum, sizeof(tfloat), cudaMemcpyDeviceToHost);
				cudaFree(d_sum);

				cudaFree(d_maskconv);
			}
		}

		tcomplex* d_mapft;
		cudaMalloc((void**)&d_mapft, ElementsFFT(dimsmap) * sizeof(tcomplex));
		tcomplex* d_map2ft;
		cudaMalloc((void**)&d_map2ft, ElementsFFT(dimsmap) * sizeof(tcomplex));
		
		// Create FTs of map and map^2
		{
			d_FFTR2C(d_map, d_mapft, &localplanforw);

			tfloat* d_map2;
			cudaMalloc((void**)&d_map2, Elements(dimsmap) * sizeof(tfloat));
			d_Square(d_map, d_map2, Elements(dimsmap));
			d_FFTR2C(d_map2, d_map2ft, &localplanforw);

			cudaFree(d_map2);
		}

		tfloat* d_mapconv;
		tfloat* d_map2conv;

		// Convolve
		{
			d_ComplexMultiplyByConjVector(d_mapft, d_maskft, d_mapft, ElementsFFT(dimsmap));
			d_ComplexMultiplyByConjVector(d_map2ft, d_maskft, d_map2ft, ElementsFFT(dimsmap));
			cudaFree(d_maskft);

			cudaMalloc((void**)&d_mapconv, Elements(dimsmap) * sizeof(tfloat));
			d_IFFTC2R(d_mapft, d_mapconv, &localplanback, dimsmap);
			cudaFree(d_mapft);

			cudaMalloc((void**)&d_map2conv, Elements(dimsmap) * sizeof(tfloat));
			d_IFFTC2R(d_map2ft, d_map2conv, &localplanback, dimsmap);
			cudaFree(d_map2ft);
		}

		// Optionally, also output local mean
		if (d_mean != NULL)
		{
			d_DivideByScalar(d_mapconv, d_mean, Elements(dimsmap), masksum);
		}

		// std = sqrt(max(0, masksum * conv2 - conv1^2)) / masksum
		{
			d_MultiplyByScalar(d_map2conv, d_map2conv, Elements(dimsmap), masksum);
			d_Square(d_mapconv, d_mapconv, Elements(dimsmap));

			d_SubtractVector(d_map2conv, d_mapconv, d_map2conv, Elements(dimsmap));
			d_MaxOp(d_map2conv, (tfloat)0, d_map2conv, Elements(dimsmap));

			d_Sqrt(d_map2conv, d_map2conv, Elements(dimsmap));

			d_DivideByScalar(d_map2conv, d_std, Elements(dimsmap), masksum);
		}

		cudaFree(d_mapconv);
		cudaFree(d_map2conv);

		if (planforw == 0)
			cufftDestroy(localplanforw);
		if (planback == 0)
			cufftDestroy(localplanback);
	}
}