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

	__global__ void LocalFSCVariableKernel(tcomplex* d_volume1,
											tcomplex* d_volume2,
											uint sidelength,
											uint sidelengthft,
											float4 fscthreshold,
											tfloat angpix,
											tfloat* d_sparsemasksums,
											int3* d_sparseorigins,
											int3 dimsvolume,
											tfloat* d_resolution,
											tfloat* d_avgfsc,
											tfloat* d_avgamps,
											tfloat* d_avgsamples,
											int avgoversample,
											tfloat* d_globalfscnums,
											tfloat* d_globalfscdenoms1,
											tfloat* d_globalfscdenoms2,
											tfloat* d_debugfsc);

	///////////////////////////////////
	//Local Fourier Shell Correlation//
	///////////////////////////////////

	void d_LocalFSC(tfloat* d_volume1,
					tfloat* d_volume2,
					tfloat* d_volumemask,
					int3 dimsvolume,
					tfloat* d_resolution,
					int windowsize,
					int spacing,
					tfloat fscthreshold,
					tfloat angpix,
					tfloat* d_avgfsc,
					tfloat* d_avgamps,
					tfloat* d_avgsamples,
					int avgoversample,
					tfloat* h_globalfsc)
	{
		// dimsvolume sans the region where window around position of interest would exceed the volume
		int3 dimsaccessiblevolume = toInt3(dimsvolume.x - windowsize, dimsvolume.y - windowsize, dimsvolume.z - windowsize);
		int3 dimsaccessiblespaced = dimsaccessiblevolume / spacing;
		int3 dimswindow = toInt3(windowsize, windowsize, windowsize);

		int elementsvol = Elements(dimsaccessiblespaced);
		int elementsslice = dimsaccessiblespaced.x * dimsaccessiblespaced.y;
		int elementswindow = Elements(dimswindow);

		uint batchmemory = 256 << 20;
		uint windowmemory = Elements(dimswindow) * sizeof(tfloat);
		uint batchsize = batchmemory / windowmemory;

		// Hann mask for extracted portions
		tfloat* h_masksums;
		tfloat* d_mask;
		{
			d_mask = CudaMallocValueFilled(Elements(dimswindow), (tfloat)1);
			tfloat maskradius = 0;
			d_SphereMask(d_mask, d_mask, dimswindow, &maskradius, dimswindow.x / 2, NULL, false);
			//d_WriteMRC(d_mask, dimswindow, "d_mask.mrc");

			// Calculate all sub-mask sums
			tfloat* d_maskpadded = CudaMallocValueFilled(Elements(dimsvolume) * 2, (tfloat)0);
			d_Pad(d_mask, d_maskpadded, dimswindow, dimsvolume, T_PAD_VALUE, (tfloat)0);
			d_RemapFull2FullFFT(d_maskpadded, d_maskpadded, dimsvolume);
			tcomplex* d_maskpaddedft = CudaMallocValueFilled(ElementsFFT(dimsvolume), make_cuComplex(0, 0));
			d_FFTR2C(d_maskpadded, d_maskpaddedft, 3, dimsvolume);
			cudaFree(d_maskpadded);

			tcomplex* d_masksumsft = CudaMallocValueFilled(ElementsFFT(dimsvolume), make_cuComplex(0, 0));
			d_FFTR2C(d_volumemask, d_masksumsft, 3, dimsvolume);
			d_ComplexMultiplyByConjVector(d_masksumsft, d_maskpaddedft, d_masksumsft, ElementsFFT(dimsvolume));
			cudaFree(d_maskpaddedft);

			tfloat* d_masksums = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
			d_IFFTC2R(d_masksumsft, d_masksums, 3, dimsvolume, 1, false);
			cudaFree(d_masksumsft);
			d_DivideByScalar(d_masksums, d_masksums, Elements(dimsvolume), (tfloat)Elements(dimsvolume) * Elements(dimswindow));

			h_masksums = (tfloat*)MallocFromDeviceArray(d_masksums, Elements(dimsvolume) * sizeof(tfloat));
			cudaFree(d_masksums);

			//WriteMRC(h_masksums, dimsvolume, "d_masksums.mrc");
		}

		tfloat* d_accessibleresolution = CudaMallocValueFilled(Elements(dimsaccessiblevolume), windowsize * angpix);

		// Allocate buffers for batch window extraction
		tfloat *d_extracts1, *d_extracts2;
		cudaMalloc((void**)&d_extracts1, Elements(dimswindow) * batchsize * sizeof(tfloat));
		cudaMalloc((void**)&d_extracts2, Elements(dimswindow) * batchsize * sizeof(tfloat));

		// ... and their FT
		tcomplex* d_extractsft1, *d_extractsft2;
		cudaMalloc((void**)&d_extractsft1, ElementsFFT(dimswindow) * batchsize * sizeof(tcomplex));
		cudaMalloc((void**)&d_extractsft2, ElementsFFT(dimswindow) * batchsize * sizeof(tcomplex));

		// Positions at which the windows will be extracted in one batch
		int3* h_extractorigins;
		cudaMallocHost((void**)&h_extractorigins, batchsize * sizeof(int3));
		int3* d_extractorigins;
		cudaMalloc((void**)&d_extractorigins, batchsize * sizeof(int3));

		// All potential positions for mask sum checks
		int3* h_allorigins;
		cudaMallocHost((void**)&h_allorigins, elementsvol * sizeof(int3));
		for (int i = 0; i < elementsvol; i++)
		{
			// Set origins for window extraction
			int z = i / elementsslice * spacing;
			int y = (i % elementsslice) / dimsaccessiblevolume.x * spacing;
			int x = i % dimsaccessiblevolume.x * spacing;
			h_allorigins[i] = toInt3(x, y, z);
		}

		// Make origins and mask sums sparse, i. e. remove all positions where mask sum is <= 0
		int elementssparse = 0;
		int3* h_sparseorigins;
		cudaMallocHost((void**)&h_sparseorigins, elementsvol * sizeof(int3));
		tfloat* h_sparsemasksums;
		cudaMallocHost((void**)&h_sparsemasksums, elementsvol * sizeof(tfloat));
		for (int i = 0; i < elementsvol; i++)
		{
			int3 ori = h_allorigins[i] + dimswindow / 2;
			tfloat masksum = h_masksums[(ori.z * dimsvolume.y + ori.y) * dimsvolume.x + ori.x];
			if (masksum > 1e-2f)
			{
				h_sparseorigins[elementssparse] = h_allorigins[i];
				h_sparsemasksums[elementssparse] = masksum;
				elementssparse++;
			}
		}
		int3* d_sparseorigins = (int3*)CudaMallocFromHostArray(h_sparseorigins, elementssparse * sizeof(int3));
		tfloat* d_sparsemasksums = (tfloat*)CudaMallocFromHostArray(h_sparsemasksums, elementssparse * sizeof(tfloat));
		cudaFreeHost(h_sparseorigins);
		cudaFreeHost(h_sparsemasksums);
		cudaFreeHost(h_allorigins);
		free(h_masksums);

		// Batch FFT for extracted windows
		cufftHandle planforw = d_FFTR2CGetPlan(3, dimswindow, batchsize);
		cufftHandle planback = d_IFFTC2RGetPlan(3, dimswindow, batchsize);

		float s = -fscthreshold / (fscthreshold - 1);
		float4 thresholdparts = make_float4(s,
											2 * sqrt(s) + 1,
											s + 1,
											2 * sqrt(s));

		tfloat* d_globalfscnums = CudaMallocValueFilled(windowsize / 2, (tfloat)0);
		tfloat* d_globalfscdenoms1 = CudaMallocValueFilled(windowsize / 2, (tfloat)0);
		tfloat* d_globalfscdenoms2 = CudaMallocValueFilled(windowsize / 2, (tfloat)0);

		for (int i = 0; i < elementssparse; i += batchsize)
		{
			uint curbatch = tmin(batchsize, elementssparse - i);
			
			// Extract windows
			d_ExtractMany(d_volume1, d_extracts1, dimsvolume, dimswindow, d_sparseorigins + i, false, curbatch);
			d_ExtractMany(d_volume2, d_extracts2, dimsvolume, dimswindow, d_sparseorigins + i, false, curbatch);

			// Multiply by Hann mask
			d_MultiplyByVector(d_extracts1, d_mask, d_extracts1, elementswindow, curbatch);
			d_MultiplyByVector(d_extracts2, d_mask, d_extracts2, elementswindow, curbatch);

			//d_WriteMRC(d_extracts1, dimswindow, "d_extracts1.mrc");
			//d_WriteMRC(d_extracts2, dimswindow, "d_extracts2.mrc");

			// FFT
			d_FFTR2C(d_extracts1, d_extractsft1, &planforw);
			d_FFTR2C(d_extracts2, d_extractsft2, &planforw);

			//tfloat* d_debugfsc = CudaMallocValueFilled(windowsize / 2, (tfloat)0);

			int TpB = 128;
			dim3 grid = dim3(curbatch, 1, 1);
			LocalFSCVariableKernel << <grid, TpB >> > (d_extractsft1,
														d_extractsft2,
														windowsize,
														windowsize / 2 + 1,
														thresholdparts,
														angpix,
														d_sparsemasksums + i,
														d_sparseorigins + i,
														dimsaccessiblespaced,
														d_accessibleresolution,
														d_avgfsc,
														d_avgamps,
														d_avgsamples,
														avgoversample,
														d_globalfscnums,
														d_globalfscdenoms1,
														d_globalfscdenoms2,
														NULL);

			/*tfloat* h_debugfsc = (tfloat*)MallocFromDeviceArray(d_debugfsc, windowsize / 2 * sizeof(tfloat));
			cudaFree(d_debugfsc);
			free(h_debugfsc);*/

			//tfloat* h_resolution = (tfloat*)MallocFromDeviceArray(d_accessibleresolution + i, curbatch * sizeof(tfloat));
			//tfloat* h_bfac = (tfloat*)MallocFromDeviceArray(d_accessiblebfactors + i, curbatch * sizeof(tfloat));
			//free(h_resolution);
			//free(h_bfac);

			//progressbar(i, elementsvol, 100);
		}

		//d_Inv(d_accessibleresolution, d_accessibleresolution, Elements(dimsaccessiblevolume));
		//d_MultiplyByScalar(d_accessibleresolution, d_accessibleresolution, Elements(dimsaccessiblevolume), windowsize * angpix);

		//d_WriteMRC(d_accessibleresolution, dimsaccessiblevolume, "d_accessibleresolution.mrc");
		d_Pad(d_accessibleresolution, d_resolution, dimsaccessiblevolume, dimsvolume, T_PAD_VALUE, (tfloat)windowsize * angpix);

		cufftDestroy(planback);
		cufftDestroy(planforw);

		tfloat* h_globalfscnums = (tfloat*)MallocFromDeviceArray(d_globalfscnums, windowsize / 2 * sizeof(tfloat));
		cudaFree(d_globalfscnums);
		tfloat* h_globalfscdenoms1 = (tfloat*)MallocFromDeviceArray(d_globalfscdenoms1, windowsize / 2 * sizeof(tfloat));
		cudaFree(d_globalfscdenoms1);
		tfloat* h_globalfscdenoms2 = (tfloat*)MallocFromDeviceArray(d_globalfscdenoms2, windowsize / 2 * sizeof(tfloat));
		cudaFree(d_globalfscdenoms2);

		for (int i = 0; i < windowsize / 2; i++)
			h_globalfsc[i] = h_globalfscnums[i] / tmax(1e-20f, sqrt(h_globalfscdenoms1[i] * h_globalfscdenoms2[i]));

		free(h_globalfscnums);
		free(h_globalfscdenoms1);
		free(h_globalfscdenoms2);

		cudaFree(d_sparsemasksums);
		cudaFree(d_sparseorigins);
		cudaFree(d_accessibleresolution);
		cudaFree(d_extractorigins);
		cudaFree(d_mask);
		cudaFree(d_extractsft2);
		cudaFree(d_extractsft1);
		cudaFree(d_extracts2);
		cudaFree(d_extracts1);

		cudaFreeHost(h_extractorigins);
	}

	__global__ void LocalFSCVariableKernel(tcomplex* d_volume1,
											tcomplex* d_volume2,
											uint sidelength,
											uint sidelengthft,
											float4 fscthreshold,
											tfloat angpix,
											tfloat* d_sparsemasksums,
											int3* d_sparseorigins,
											int3 dimsvolume,
											tfloat* d_resolution,
											tfloat* d_avgfsc,
											tfloat* d_avgamps,
											tfloat* d_avgsamples,
											int avgoversample,
											tfloat* d_globalfscnums,
											tfloat* d_globalfscdenoms1,
											tfloat* d_globalfscdenoms2,
											tfloat* d_debugfsc)
	{
		__shared__ tfloat s_nums[64];
		__shared__ tfloat s_denoms1[64];
		__shared__ tfloat s_denoms2[64];

		float finalres;

		__shared__ tfloat s_amps[64];
		__shared__ tfloat s_samples[64];

		uint elementsslice = sidelengthft * sidelength;
		uint elementscube = elementsslice * sidelength;

		d_volume1 += elementscube * blockIdx.x;
		d_volume2 += elementscube * blockIdx.x;

		if (threadIdx.x < 64)
		{
			s_nums[threadIdx.x] = 0;
			s_denoms1[threadIdx.x] = 0;
			s_denoms2[threadIdx.x] = 0;

			s_amps[threadIdx.x] = 0;
			s_samples[threadIdx.x] = 0;
		}
		__syncthreads();

		uint sidelengthhalf = sidelength / 2;

		// Gather samples for FSC
		for (uint id = threadIdx.x; id < elementscube; id += 128)
		{
			int idz = (int)(id / elementsslice);
			int idy = (int)((id % elementsslice) / sidelengthft);
			int idx = (int)(id % sidelengthft);

			tfloat rx = idx;
			tfloat ry = idy <= sidelengthhalf ? idy : idy - (int)sidelength;
			tfloat rz = idz <= sidelengthhalf ? idz : idz - (int)sidelength;
			tfloat radius = sqrt(rx * rx + ry * ry + rz * rz);
			uint ri = (uint)(radius + 0.5f);
			if (ri >= sidelengthhalf)
				continue;

			tcomplex val1 = d_volume1[id];
			atomicAdd(s_denoms1 + ri, dotp2(val1, val1));

			tcomplex val2 = d_volume2[id];
			atomicAdd(s_denoms2 + ri, dotp2(val2, val2));

			atomicAdd(s_nums + ri, dotp2(val1, val2));

			val1 += val2;
			atomicAdd(s_amps + ri, sqrt(dotp2(val1, val1)));
			atomicAdd(s_samples + ri, 2);
		}
		__syncthreads();

		// Calculate FSC
		if (threadIdx.x < sidelength / 2)
		{
			atomicAdd(d_globalfscnums + threadIdx.x, s_nums[threadIdx.x]);
			atomicAdd(d_globalfscdenoms1 + threadIdx.x, s_denoms1[threadIdx.x]);
			atomicAdd(d_globalfscdenoms2 + threadIdx.x, s_denoms2[threadIdx.x]);

			s_nums[threadIdx.x] /= tmax(1e-20f, sqrt(s_denoms1[threadIdx.x] * s_denoms2[threadIdx.x]));
			//if (blockIdx.x == 0)
			//	d_debugfsc[threadIdx.x] = nums[threadIdx.x];

			s_amps[threadIdx.x] /= tmax(1e-20f, s_samples[threadIdx.x]);
		}
		__syncthreads();


		//if (threadIdx.x == 0)
		{
			// Find where FSC crosses threshold
			float masksum = d_sparsemasksums[blockIdx.x];
			int i;
			float currentthreshold;
			for (i = 1; i < sidelengthhalf; i++)
			{
				float n = 1.0f / (2 * PI * (i * i) * masksum);
				currentthreshold = tmin(0.95f, (fscthreshold.x + fscthreshold.y * n) / (fscthreshold.z + fscthreshold.w * n));
				if (s_nums[i] < currentthreshold)
					break;
			}
			i = tmax(0, i - 1);

			tfloat current = s_nums[i];
			tfloat next = s_nums[tmin(i + 1, sidelengthhalf - 1)];

			finalres = tmax(1, (tfloat)i + tmax(tmin((current - currentthreshold) / (current - next + 1e-5f), 1.0f), 0.0f));

			if (threadIdx.x == 0)
			{
				int3 origin = d_sparseorigins[blockIdx.x];
				d_resolution[(origin.z * dimsvolume.y + origin.y) * dimsvolume.x + origin.x] = angpix * sidelength / finalres;
			}
		}

		__syncthreads();

		int iavg = tmin(sidelength / 2 * avgoversample - 1, (int)(finalres * avgoversample + 0.5f));
		//int iavg = 0;
		
		if (threadIdx.x < sidelength / 2)
		{
			float weight = d_sparsemasksums[blockIdx.x];

			atomicAdd(d_avgfsc + iavg * (sidelength / 2) + threadIdx.x, s_nums[threadIdx.x] * weight);
			atomicAdd(d_avgamps + iavg * (sidelength / 2) + threadIdx.x, s_amps[threadIdx.x]);

			if (threadIdx.x == 0)
				atomicAdd(d_avgsamples + iavg, weight);
		}
	}
}