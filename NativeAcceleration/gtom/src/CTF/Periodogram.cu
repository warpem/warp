#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Masking.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	//////////////////////////////////////////////////////
	//Calculate power spectrum based on multiple regions//
	//////////////////////////////////////////////////////

	void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, float overlapfraction, int2 dimsregion, int2 dimspadded, tfloat* d_output2d, bool dopost)
	{
		// Create uniform grid over the image
		int2 regions;
		int3* h_origins = GetEqualGridSpacing(dimsimage, dimsregion, overlapfraction, regions);
		int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, Elements2(regions) * sizeof(int3));
		free(h_origins);

		int norigins = Elements2(regions);

		tfloat* d_temp2d;
		cudaMalloc((void**)&d_temp2d, ElementsFFT2(dimspadded) * norigins * sizeof(tfloat));

		// Call the custom-grid version to extract 2D spectra
		d_CTFPeriodogram(d_image, dimsimage, d_origins, norigins, dimsregion, dimspadded, d_temp2d, dopost);

		d_ReduceMean(d_temp2d, d_output2d, ElementsFFT2(dimspadded), norigins);

		cudaFree(d_temp2d);
		cudaFree(d_origins);
	}

	void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, int2 dimsregion, int2 dimspadded, tfloat* d_output2d, bool dopost, cufftHandle planforw, tfloat* d_extracted, tcomplex* d_extractedft)
	{
		cufftHandle ownplanforw = planforw;
		if (planforw == 0)
			ownplanforw = d_FFTR2CGetPlan(2, toInt3(dimspadded), norigins);

		tfloat* d_ownextracted;
        if (d_extracted == 0)
            cudaMalloc((void**)&d_ownextracted, norigins * Elements2(dimspadded) * sizeof(tfloat));
        else
            d_ownextracted = d_extracted;

		tcomplex* d_ownextractedft;
        if (d_extractedft == 0)
            cudaMalloc((void**)&d_ownextractedft, norigins * ElementsFFT2(dimspadded) * sizeof(tcomplex));
        else
            d_ownextractedft = d_extractedft;
				
		d_ExtractMany(d_image, d_ownextracted, toInt3(dimsimage), toInt3(dimsregion), d_origins, true, norigins);
		//d_WriteMRC(d_ownextracted, toInt3(dimsregion.x, dimsregion.y, norigins), "d_ownextracted.mrc");

		d_NormMonolithic(d_ownextracted, d_ownextracted, Elements2(dimsregion), T_NORM_MEAN01STD, norigins);
		tfloat radius = dimsregion.x * 3 / 4.0f / 2;
		d_SphereMask(d_ownextracted, d_ownextracted, toInt3(dimsregion), &radius, dimsregion.x * 1 / 4.0f / 2, NULL, norigins);
		//d_HammingMask(d_extracted, d_extracted, toInt3(dimsregion), &radius, NULL, norigins);
		//d_HammingMaskBorderDistance(d_extracted, d_extracted, toInt3(dimsregion), dimsregion.x / 4, curbatch);
		if (dimsregion.x != dimspadded.x || dimsregion.y != dimspadded.y)
		{
			d_Pad(d_ownextracted, (tfloat*)d_ownextractedft, toInt3(dimsregion), toInt3(dimspadded), T_PAD_VALUE, (tfloat)0, norigins);
			//d_NormMonolithic((tfloat*)d_ownextractedft, d_ownextracted, Elements2(dimspadded), T_NORM_MEAN01STD, curbatch);
		}
		else
		{
			//d_NormMonolithic(d_ownextracted, d_ownextracted, Elements2(dimspadded), T_NORM_MEAN01STD, curbatch);
		}
		//d_WriteMRC(d_ownextracted, toInt3(dimspadded.x, dimspadded.y, norigins), "d_ownextracted.mrc");
		d_FFTR2C(d_ownextracted, d_ownextractedft, &ownplanforw);
		//d_WriteMRC(d_ownextracted, toInt3(dimspadded.x / 2 + 1, dimspadded.y, norigins), "d_ownextractedft.mrc");

		if (dopost)
		{
			d_Abs(d_ownextractedft, d_ownextracted, norigins * ElementsFFT2(dimspadded));
			d_AddScalar(d_ownextracted, d_ownextracted, norigins * ElementsFFT2(dimspadded), (tfloat)1e-6);
			d_Log(d_ownextracted, d_ownextracted, norigins * ElementsFFT2(dimspadded));
			d_MultiplyByVector(d_ownextracted, d_ownextracted, d_ownextracted, ElementsFFT2(dimspadded) * norigins);
		}
		else
		{
			d_Abs(d_ownextractedft, d_output2d, norigins * ElementsFFT2(dimspadded));
		}

		//d_RemapHalfFFT2Half(d_ownextracted, d_output2d, toInt3(dimspadded), norigins);
		//d_WriteMRC(d_output2d, toInt3(dimspadded.x / 2 + 1, dimspadded.y, norigins), "d_extractedoutput.mrc");

        if (d_extractedft == 0)            
		    cudaFree(d_extractedft);
        if (d_extracted == 0)
		    cudaFree(d_ownextracted);

		if (planforw == 0)
			cufftDestroy(ownplanforw);
	}
}