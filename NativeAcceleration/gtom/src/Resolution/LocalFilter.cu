#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Masking.cuh"
#include "gtom/include/Resolution.cuh"


namespace gtom
{
	///////////////////////////
	//CUDA kernel declaration//
	///////////////////////////

	__global__ void LocalFilterKernel(tcomplex* d_input,
										tcomplex* d_output,
										uint sidelength,
										uint sidelengthft,
										tfloat angpix,
										tfloat* d_resolution,
										tfloat* d_filterramps,
										int rampsoversample,
										tfloat* d_debugfsc);

	///////////////////////////////////
	//Local Fourier Shell Correlation//
	///////////////////////////////////

	void d_LocalFilter(tfloat* d_input,
						tfloat* d_filtered,
						int3 dimsvolume,
						tfloat* d_resolution,
						int windowsize,
						tfloat angpix,
						tfloat* d_filterramps,
						int rampsoversample)
	{
		// dimsvolume sans the region where window around position of interest would exceed the volume
		int3 dimsaccessiblevolume = toInt3(dimsvolume.x - windowsize, dimsvolume.y - windowsize, dimsvolume.z - windowsize);
		int3 dimswindow = toInt3(windowsize, windowsize, windowsize);

		uint batchmemory = 512 << 20;
		uint windowmemory = Elements(dimswindow) * sizeof(tfloat);
		uint batchsize = batchmemory / windowmemory;

		tfloat* d_accessibleresolution = CudaMallocValueFilled(Elements(dimsaccessiblevolume), (tfloat)0);
		tfloat* d_accessiblecorrected = CudaMallocValueFilled(Elements(dimsaccessiblevolume), (tfloat)0);

		d_Pad(d_resolution, d_accessibleresolution, dimsvolume, dimsaccessiblevolume, T_PAD_VALUE, (tfloat)0);

		// Allocate buffers for batch window extraction
		tfloat *d_extracts1;
		cudaMalloc((void**)&d_extracts1, Elements(dimswindow) * batchsize * sizeof(tfloat));

		// ... and their FT
		tcomplex* d_extractsft1;
		cudaMalloc((void**)&d_extractsft1, ElementsFFT(dimswindow) * batchsize * sizeof(tcomplex));

		// Hann mask for extracted portions
		tfloat* d_mask = CudaMallocValueFilled(Elements(dimswindow), (tfloat)1);
		d_HannMask(d_mask, d_mask, dimswindow, NULL, NULL);
		//d_WriteMRC(d_mask, dimswindow, "d_mask.mrc");

		// Positions at which the windows will be extracted
		int3* h_extractorigins;
		cudaMallocHost((void**)&h_extractorigins, batchsize * sizeof(int3));
		int3* d_extractorigins;
		cudaMalloc((void**)&d_extractorigins, batchsize * sizeof(int3));

		// Batch FFT for extracted windows
		cufftHandle planforw = d_FFTR2CGetPlan(3, dimswindow, batchsize);
		cufftHandle planback = d_IFFTC2RGetPlan(3, dimswindow, batchsize);

		int elementsvol = Elements(dimsaccessiblevolume);
		int elementsslice = dimsaccessiblevolume.x * dimsaccessiblevolume.y;
		int elementswindow = Elements(dimswindow);

		d_Inv(d_accessibleresolution, d_accessibleresolution, Elements(dimsaccessiblevolume));
		d_MultiplyByScalar(d_accessibleresolution, d_accessibleresolution, Elements(dimsaccessiblevolume), windowsize * angpix);

		for (int i = 0; i < elementsvol; i += batchsize)
		{
			uint curbatch = tmin(batchsize, elementsvol - i);

			for (int b = 0; b < curbatch; b++)
			{
				// Set origins for window extraction
				int z = (i + b) / elementsslice;
				int y = ((i + b) % elementsslice) / dimsaccessiblevolume.x;
				int x = (i + b) % dimsaccessiblevolume.x;
				h_extractorigins[b] = toInt3(x, y, z);
			}
			cudaMemcpy(d_extractorigins, h_extractorigins, curbatch * sizeof(int3), cudaMemcpyHostToDevice);

			// Extract windows
			d_ExtractMany(d_input, d_extracts1, dimsvolume, dimswindow, d_extractorigins, false, curbatch);

			// Multiply by Hann mask
			d_MultiplyByVector(d_extracts1, d_mask, d_extracts1, elementswindow, curbatch);

			//d_WriteMRC(d_extracts1, dimswindow, "d_extracts1.mrc");
			//d_WriteMRC(d_extracts2, dimswindow, "d_extracts2.mrc");

			// FFT
			d_FFTR2C(d_extracts1, d_extractsft1, &planforw);

			//tfloat* d_debugfsc = CudaMallocValueFilled(windowsize / 2, (tfloat)0);

			int TpB = 128;
			dim3 grid = dim3(curbatch, 1, 1);
			LocalFilterKernel << <grid, TpB >> > (d_extractsft1,
												d_extractsft1,
												windowsize,
												windowsize / 2 + 1,
												angpix,
												d_accessibleresolution + i,
												d_filterramps,
												rampsoversample,
												NULL);

			// Low-pass and sharpened
			d_IFFTC2R(d_extractsft1, d_extracts1, &planback, dimswindow, curbatch);
			CudaMemcpyStrided(d_accessiblecorrected + i, d_extracts1 + (dimswindow.z / 2 * dimswindow.y + dimswindow.y / 2) * dimswindow.x + dimswindow.x / 2, curbatch, 1, Elements(dimswindow));

			/*tfloat* h_debugfsc = (tfloat*)MallocFromDeviceArray(d_debugfsc, windowsize / 2 * sizeof(tfloat));
			cudaFree(d_debugfsc);
			free(h_debugfsc);*/

			//tfloat* h_resolution = (tfloat*)MallocFromDeviceArray(d_accessibleresolution + i, curbatch * sizeof(tfloat));
			//tfloat* h_bfac = (tfloat*)MallocFromDeviceArray(d_accessiblebfactors + i, curbatch * sizeof(tfloat));
			//free(h_resolution);
			//free(h_bfac);

			//progressbar(i, elementsvol, 100);
		}

		d_Pad(d_accessiblecorrected, d_filtered, dimsaccessiblevolume, dimsvolume, T_PAD_VALUE, (tfloat)0);

		cufftDestroy(planback);
		cufftDestroy(planforw);


		cudaFree(d_accessiblecorrected);
		cudaFree(d_accessibleresolution);
		cudaFree(d_extractorigins);
		cudaFree(d_mask);
		cudaFree(d_extractsft1);
		cudaFree(d_extracts1);

		cudaFreeHost(h_extractorigins);
	}

__global__ void LocalFilterKernel(tcomplex* d_input,
									tcomplex* d_output,
									uint sidelength,
									uint sidelengthft,
									tfloat angpix,
									tfloat* d_resolution,
									tfloat* d_filterramps,
									int rampsoversample,
									tfloat* d_debugfsc)
	{
		float cutoffshell;
		cutoffshell = d_resolution[blockIdx.x];

		uint elementsslice = sidelengthft * sidelength;
		uint elementscube = elementsslice * sidelength;

		d_input += elementscube * blockIdx.x;
		d_output += elementscube * blockIdx.x;
		
		// Filter and sharpen the input
		{
			uint sidelengthhalf = sidelength / 2;

			for (uint id = threadIdx.x; id < elementscube; id += 128)
			{
				int idz = (int)(id / elementsslice);
				int idy = (int)((id % elementsslice) / sidelengthft);
				int idx = (int)(id % sidelengthft);

				tfloat rx = idx;
				tfloat ry = idy <= sidelengthhalf ? idy : idy - (int)sidelength;
				tfloat rz = idz <= sidelengthhalf ? idz : idz - (int)sidelength;
				tfloat radius = sqrt(rx * rx + ry * ry + rz * rz);

				int sidehalf = sidelength / 2;

				if (radius >= sidehalf)
				{
					d_output[id] = make_cuComplex(0, 0);
					continue;
				}

				tfloat ramp00 = d_filterramps[tmin((int)(cutoffshell * rampsoversample), sidehalf * rampsoversample - 1) * sidehalf + (int)radius];
				tfloat ramp01 = d_filterramps[tmin((int)(cutoffshell * rampsoversample), sidehalf * rampsoversample - 1) * sidehalf + tmin((int)radius + 1, sidehalf - 1)];

				tfloat ramp10 = d_filterramps[tmin((int)(cutoffshell * rampsoversample) + 1, sidehalf * rampsoversample - 1) * sidehalf + (int)radius];
				tfloat ramp11 = d_filterramps[tmin((int)(cutoffshell * rampsoversample) + 1, sidehalf * rampsoversample - 1) * sidehalf + tmin((int)radius + 1, sidelength / 2 - 1)];

				tfloat ramp0 = lerp(ramp00, ramp01, radius - floor(radius));
				tfloat ramp1 = lerp(ramp10, ramp11, radius - floor(radius));

				tfloat ramp = lerp(ramp0, ramp1, cutoffshell * rampsoversample - floor(cutoffshell * rampsoversample));

				tcomplex val = d_input[id];
				val *= ramp;

				if (isnan(val.x))
					val.x = 0;
				if (isnan(val.y))
					val.y = 0;
				d_output[id] = val;	// sharpened
			}
		}
	}
}