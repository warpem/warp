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

	template<bool anisotropic, bool dolocalbfac> __global__ void FSCBFacKernel(tcomplex* d_volume1,
																				tcomplex* d_volume2,
																				uint sidelength,
																				uint sidelengthft,
																				tfloat fscthreshold,
																				uint minbfacshell,
																				tfloat angpix,
																				tfloat globalbfac,
																				tfloat minbfac,
																				tfloat bfacbias,
																				tfloat mtfslope,
																				tfloat* d_resolution,
																				tfloat* d_bfactors,
																				bool dofilterhalfmaps,
																				tfloat* d_debugfsc);

	///////////////////////////////////
	//Local Fourier Shell Correlation//
	///////////////////////////////////

	void d_LocalFSCBfac(tfloat* d_volume1, 
						tfloat* d_volume2, 
						int3 dimsvolume, 
						tfloat* d_resolution, 
						tfloat* d_bfactors, 
						tfloat* d_corrected, 
						tfloat* d_unsharpened, 
						int windowsize, 
						tfloat fscthreshold, 
						bool dolocalbfac,
						tfloat globalbfac,
						tfloat minresbfac, 
						tfloat angpix, 
						tfloat minbfac, 
						tfloat bfacbias, 
						tfloat mtfslope, 
						bool doanisotropy,
						bool dofilterhalfmaps)
	{
		// dimsvolume sans the region where window around position of interest would exceed the volume
		int3 dimsaccessiblevolume = toInt3(dimsvolume.x - windowsize, dimsvolume.y - windowsize, dimsvolume.z - windowsize);
		int3 dimswindow = toInt3(windowsize, windowsize, windowsize);

		uint batchmemory = 512 << 20;
		uint windowmemory = Elements(dimswindow) * sizeof(tfloat);
		uint batchsize = batchmemory / windowmemory;

		tfloat* d_accessibleresolution = CudaMallocValueFilled(Elements(dimsaccessiblevolume), (tfloat)0);
		tfloat* d_accessiblebfactors = CudaMallocValueFilled(Elements(dimsaccessiblevolume), (tfloat)0);
		tfloat* d_accessiblecorrected = CudaMallocValueFilled(Elements(dimsaccessiblevolume), (tfloat)0);
		tfloat* d_accessibleunsharpened = CudaMallocValueFilled(Elements(dimsaccessiblevolume), (tfloat)0);
		tfloat* d_accessiblefilteredhalf1 = dofilterhalfmaps ? CudaMallocValueFilled(Elements(dimsaccessiblevolume), (tfloat)0) : NULL;
		tfloat* d_accessiblefilteredhalf2 = dofilterhalfmaps ? CudaMallocValueFilled(Elements(dimsaccessiblevolume), (tfloat)0) : NULL;

		// Allocate buffers for batch window extraction
		tfloat *d_extracts1, *d_extracts2;
		cudaMalloc((void**)&d_extracts1, Elements(dimswindow) * batchsize * sizeof(tfloat));
		cudaMalloc((void**)&d_extracts2, Elements(dimswindow) * batchsize * sizeof(tfloat));

		// ... and their FT
		tcomplex* d_extractsft1, *d_extractsft2;
		cudaMalloc((void**)&d_extractsft1, ElementsFFT(dimswindow) * batchsize * sizeof(tcomplex));
		cudaMalloc((void**)&d_extractsft2, ElementsFFT(dimswindow) * batchsize * sizeof(tcomplex));

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
		uint minbfacshell = windowsize / (minresbfac / angpix) + 1;

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
			d_ExtractMany(d_volume1, d_extracts1, dimsvolume, dimswindow, d_extractorigins, false, curbatch);
			d_ExtractMany(d_volume2, d_extracts2, dimsvolume, dimswindow, d_extractorigins, false, curbatch);

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
			if (doanisotropy)
				if (dolocalbfac)
					FSCBFacKernel<true, true> << <grid, TpB >> > (d_extractsft1, 
																	d_extractsft2, 
																	windowsize, 
																	windowsize / 2 + 1, 
																	fscthreshold, 
																	minbfacshell, 
																	angpix, 
																	globalbfac,
																	minbfac, 
																	bfacbias,
																	mtfslope,
																	d_accessibleresolution + i,
																	d_accessiblebfactors + i, 
																	dofilterhalfmaps,
																	NULL);
				else
					FSCBFacKernel<true, false> << <grid, TpB >> > (d_extractsft1,
																	d_extractsft2,
																	windowsize,
																	windowsize / 2 + 1,
																	fscthreshold,
																	minbfacshell,
																	angpix,
																	globalbfac,
																	minbfac,
																	bfacbias,
																	mtfslope,
																	d_accessibleresolution + i,
																	d_accessiblebfactors + i,
																	dofilterhalfmaps,
																	NULL);
			else
				if (dolocalbfac)
					FSCBFacKernel<false, true> << <grid, TpB >> > (d_extractsft1,
																	d_extractsft2,
																	windowsize,
																	windowsize / 2 + 1,
																	fscthreshold,
																	minbfacshell,
																	angpix,
																	globalbfac,
																	minbfac,
																	bfacbias,
																	mtfslope,
																	d_accessibleresolution + i,
																	d_accessiblebfactors + i,
																	dofilterhalfmaps,
																	NULL);
				else
					FSCBFacKernel<false, false> << <grid, TpB >> > (d_extractsft1,
																	d_extractsft2,
																	windowsize,
																	windowsize / 2 + 1,
																	fscthreshold,
																	minbfacshell,
																	angpix,
																	globalbfac,
																	minbfac,
																	bfacbias,
																	mtfslope,
																	d_accessibleresolution + i,
																	d_accessiblebfactors + i,
																	dofilterhalfmaps,
																	NULL);

			if (!dofilterhalfmaps)
			{
				// Low-pass and sharpened
				if (d_corrected != NULL)
				{
					d_IFFTC2R(d_extractsft1, d_extracts1, &planback, dimswindow, curbatch);
					CudaMemcpyStrided(d_accessiblecorrected + i, d_extracts1 + (dimswindow.z / 2 * dimswindow.y + dimswindow.y / 2) * dimswindow.x + dimswindow.x / 2, curbatch, 1, Elements(dimswindow));
				}

				// Low-pass, but not sharpened
				if (d_unsharpened != NULL)
				{
					d_IFFTC2R(d_extractsft2, d_extracts2, &planback, dimswindow, curbatch);
					CudaMemcpyStrided(d_accessibleunsharpened + i, d_extracts2 + (dimswindow.z / 2 * dimswindow.y + dimswindow.y / 2) * dimswindow.x + dimswindow.x / 2, curbatch, 1, Elements(dimswindow));
				}
			}
			else
			{
				d_IFFTC2R(d_extractsft1, d_extracts1, &planback, dimswindow, curbatch);
				CudaMemcpyStrided(d_accessiblefilteredhalf1 + i, d_extracts1 + (dimswindow.z / 2 * dimswindow.y + dimswindow.y / 2) * dimswindow.x + dimswindow.x / 2, curbatch, 1, Elements(dimswindow));

				d_IFFTC2R(d_extractsft2, d_extracts2, &planback, dimswindow, curbatch);
				CudaMemcpyStrided(d_accessiblefilteredhalf2 + i, d_extracts2 + (dimswindow.z / 2 * dimswindow.y + dimswindow.y / 2) * dimswindow.x + dimswindow.x / 2, curbatch, 1, Elements(dimswindow));
			}

			/*tfloat* h_debugfsc = (tfloat*)MallocFromDeviceArray(d_debugfsc, windowsize / 2 * sizeof(tfloat));
			cudaFree(d_debugfsc);
			free(h_debugfsc);*/

			//tfloat* h_resolution = (tfloat*)MallocFromDeviceArray(d_accessibleresolution + i, curbatch * sizeof(tfloat));
			//tfloat* h_bfac = (tfloat*)MallocFromDeviceArray(d_accessiblebfactors + i, curbatch * sizeof(tfloat));
			//free(h_resolution);
			//free(h_bfac);

			//progressbar(i, elementsvol, 100);
		}

		d_Inv(d_accessibleresolution, d_accessibleresolution, Elements(dimsaccessiblevolume));
		d_MultiplyByScalar(d_accessibleresolution, d_accessibleresolution, Elements(dimsaccessiblevolume), windowsize * angpix);

		//d_WriteMRC(d_accessibleresolution, dimsaccessiblevolume, "d_accessibleresolution.mrc");
		if (d_resolution != NULL)
			d_Pad(d_accessibleresolution, d_resolution, dimsaccessiblevolume, dimsvolume, T_PAD_VALUE, (tfloat)windowsize);
		if (d_bfactors != NULL)
			d_Pad(d_accessiblebfactors, d_bfactors, dimsaccessiblevolume, dimsvolume, T_PAD_VALUE, (tfloat)0);
		if (!dofilterhalfmaps)
		{
			if (d_corrected != NULL)
				d_Pad(d_accessiblecorrected, d_corrected, dimsaccessiblevolume, dimsvolume, T_PAD_VALUE, (tfloat)0);
			if (d_unsharpened != NULL)
				d_Pad(d_accessibleunsharpened, d_unsharpened, dimsaccessiblevolume, dimsvolume, T_PAD_VALUE, (tfloat)0);
		}
		else
		{
			d_Pad(d_accessiblefilteredhalf1, d_volume1, dimsaccessiblevolume, dimsvolume, T_PAD_VALUE, (tfloat)0);
			d_Pad(d_accessiblefilteredhalf2, d_volume2, dimsaccessiblevolume, dimsvolume, T_PAD_VALUE, (tfloat)0);
		}

		cufftDestroy(planback);
		cufftDestroy(planforw);

		if (dofilterhalfmaps)
		{
			cudaFree(d_accessiblefilteredhalf2);
			cudaFree(d_accessiblefilteredhalf1);
		}
		cudaFree(d_accessibleunsharpened);
		cudaFree(d_accessiblecorrected);
		cudaFree(d_accessibleresolution);
		cudaFree(d_accessiblebfactors);
		cudaFree(d_extractorigins);
		cudaFree(d_mask);
		cudaFree(d_extractsft2);
		cudaFree(d_extractsft1);
		cudaFree(d_extracts2);
		cudaFree(d_extracts1);

		cudaFreeHost(h_extractorigins);
	}

	#define unitlength 0.5773502692f

	template<bool anisotropic, bool dolocalbfac> __global__ void FSCBFacKernel(tcomplex* d_volume1, 
																				tcomplex* d_volume2, 
																				uint sidelength, 
																				uint sidelengthft, 
																				tfloat fscthreshold, 
																				uint minbfacshell, 
																				tfloat angpix, 
																				tfloat globalbfac, 
																				tfloat minbfac, 
																				tfloat bfacbias, 
																				tfloat mtfslope, 
																				tfloat* d_resolution, 
																				tfloat* d_bfactors, 
																				bool dofilterhalfmaps,
																				tfloat* d_debugfsc)
	{
		__shared__ tfloat nums[64];
		__shared__ tfloat denoms1[64];
		__shared__ tfloat denoms2[64];
		__shared__ tfloat amps[64];
		__shared__ tfloat samples[64];
		__shared__ int maxresbfac;

		__shared__ tfloat fscweights[7][64];

		__shared__ tfloat finalres[7];
		__shared__ tfloat finalbfac[7];

		uint elementsslice = sidelengthft * sidelength;
		uint elementscube = elementsslice * sidelength;

		d_volume1 += elementscube * blockIdx.x;
		d_volume2 += elementscube * blockIdx.x;

		uint naxes = anisotropic ? 7 : 1;

		for (uint axisid = 0; axisid < naxes; axisid++)
		{
			if (threadIdx.x < 64)
			{
				nums[threadIdx.x] = 0;
				denoms1[threadIdx.x] = 0;
				denoms2[threadIdx.x] = 0;
				amps[threadIdx.x] = 0;
				samples[threadIdx.x] = 0;
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
				tfloat radius = tmax(1e-6f, sqrt(rx * rx + ry * ry + rz * rz));
				uint ri = (uint)(radius + 0.5f);
				if (ri >= sidelengthhalf)
					continue;

				tfloat weight = 1;
				if (anisotropic)
				{
					if (axisid == 0)
						weight = abs(rx / radius);
					else if (axisid == 1)
						weight = abs(ry / radius);
					else if (axisid == 2)
						weight = abs(rz / radius);
					else if (axisid == 3)
						weight = abs(rx * unitlength + ry * unitlength + rz * unitlength) / radius;
					else if (axisid == 4)
						weight = abs(rx * unitlength - ry * unitlength + rz * unitlength) / radius;
					else if (axisid == 5)
						weight = abs(rx * unitlength + ry * unitlength - rz * unitlength) / radius;
					else if (axisid == 6)
						weight = abs(rx * unitlength - ry * unitlength - rz * unitlength) / radius;

					weight = tmax((weight - 0.707106781f) / 0.292893219f, 0);
				}

				tcomplex val1 = d_volume1[id] * weight;
				atomicAdd(denoms1 + ri, dotp2(val1, val1));

				tcomplex val2 = d_volume2[id] * weight;
				atomicAdd(denoms2 + ri, dotp2(val2, val2));

				atomicAdd(nums + ri, dotp2(val1, val2));

				val1 += val2;
				atomicAdd(amps + ri, sqrt(dotp2(val1, val1)));

				atomicAdd(samples + ri, weight);
			}
			__syncthreads();

			// Calculate FSC
			if (threadIdx.x < 64)
			{
				nums[threadIdx.x] /= tmax(1e-20f, sqrt(denoms1[threadIdx.x] * denoms2[threadIdx.x]));					// Calculate FSC
				nums[threadIdx.x] = tmax(1e-20f, nums[threadIdx.x]);
				fscweights[axisid][threadIdx.x] = tmin(sqrt(2 * nums[threadIdx.x] / (nums[threadIdx.x] + 1)), 1);
				amps[threadIdx.x] *= fscweights[axisid][threadIdx.x];													// Weight amplitudes by sqrt(2 * FSC / (FSC + 1))
				amps[threadIdx.x] /= 1.0f + (float)threadIdx.x / sidelength * mtfslope;									// Divide by MTF
			}

			__syncthreads();

			//if (threadIdx.x < sidelengthhalf)
			//d_debugfsc[threadIdx.x] = nums[threadIdx.x];

			if (threadIdx.x == 0)
			{
				// Find where FSC crosses threshold
				int i;
				for (i = 1; i < sidelengthhalf; i++)
					if (nums[i] < fscthreshold)
						break;
				i = tmax(0, i - 1);

				tfloat current = nums[i];
				tfloat next = nums[tmin(i + 1, sidelengthhalf - 1)];

				finalres[axisid] = tmax(1, (tfloat)i + tmax(tmin((fscthreshold - current) / (next - current + (tfloat)0.00001), 1.0f), 0.0f));

				// Find where FSC drops to 0 for the first time
				int i0;
				for (i0 = i; i0 < sidelengthhalf; i0++)
					if (nums[i0] < 0.01f)
						break;
				maxresbfac = i0;
			}

			__syncthreads();

			if (threadIdx.x >= minbfacshell && threadIdx.x < maxresbfac)
			{
				tfloat res = tmax(1e-6f, (sidelength * angpix) / threadIdx.x);
				nums[threadIdx.x] = tmin(999, 1 / (res * res));
				denoms1[threadIdx.x] = log(tmax(amps[threadIdx.x] / tmax(1, samples[threadIdx.x]), 1e-20f));
			}

			__syncthreads();

			if (threadIdx.x == 0)
			{
				// Fit line through Guinier-plotted, FSC-weighted PS
				tfloat ss_xy = 0;
				tfloat ss_xx = 0;
				tfloat ss_yy = 0;
				tfloat ave_x = 0;
				tfloat ave_y = 0;
				uint sum_w;
				for (uint i = minbfacshell; i < maxresbfac; i++)
				{
					ave_x += nums[i];
					ave_y += denoms1[i];
					ss_xx += nums[i] * nums[i];
					ss_yy += denoms1[i] * denoms1[i];
					ss_xy += nums[i] * denoms1[i];
				}
				sum_w = tmax(1, maxresbfac - minbfacshell);
				ave_x /= sum_w;
				ave_y /= sum_w;
				ss_xx -= sum_w * ave_x * ave_x;
				ss_xy -= sum_w * ave_x * ave_y;

				if (ss_xx > 0)
					finalbfac[axisid] = 4 * ss_xy / ss_xx;
				else
					finalbfac[axisid] = 0;

				if (isnan(finalbfac[axisid]))
					finalbfac[axisid] = 0;

				finalbfac[axisid] = tmin(tmax(minbfac, finalbfac[axisid] + bfacbias), 0);
			}

			__syncthreads();
		}

		// Filter and sharpen the sum of the two half-maps (or the individual half-maps if dofilterhalfmaps is true)
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
				uint ri = tmin((uint)(radius + 0.5f), sidelengthhalf - 1);

				tcomplex val1 = d_volume1[id];
				tcomplex val2 = d_volume2[id];
				tcomplex valavg = val1 + val2;
				if (ri == 0)
				{
					d_volume2[id] = valavg;
					d_volume1[id] = valavg;
					continue;
				}

				tfloat finalresaniso, fscweightaniso, bfacaniso;

				if (anisotropic)
				{
					rx /= radius;
					ry /= radius;
					rz /= radius;
					tfloat weights[] =
					{
						abs(rx),
						abs(ry),
						abs(rz),
						abs(rx * unitlength + ry * unitlength + rz * unitlength),
						abs(rx * unitlength - ry * unitlength + rz * unitlength),
						abs(rx * unitlength + ry * unitlength - rz * unitlength),
						abs(rx * unitlength - ry * unitlength - rz * unitlength)
					};
					tfloat weightsum = weights[0] + weights[1] + weights[2] + weights[3] + weights[4] + weights[5] + weights[6];
					for (uint i = 0; i < 7; i++)
						weights[i] /= weightsum;

					finalresaniso = finalres[0] * weights[0] + finalres[1] * weights[1] + finalres[2] * weights[2] + finalres[3] * weights[3] + finalres[4] * weights[4] + finalres[5] * weights[5] + finalres[6] * weights[6];
					fscweightaniso = fscweights[0][ri] * weights[0] + fscweights[1][ri] * weights[1] + fscweights[2][ri] * weights[2] + fscweights[3][ri] * weights[3] + fscweights[4][ri] * weights[4] + fscweights[5][ri] * weights[5] + fscweights[6][ri] * weights[6];

					bfacaniso = finalbfac[0] * weights[0] + finalbfac[1] * weights[1] + finalbfac[2] * weights[2] + finalbfac[3] * weights[3] + finalbfac[4] * weights[4] + finalbfac[5] * weights[5] + finalbfac[6] * weights[6];
				}
				else
				{
					finalresaniso = finalres[0];
					fscweightaniso = fscweights[0][ri];

					if (dolocalbfac)
						bfacaniso = finalbfac[0];
					else
						bfacaniso = globalbfac;
				}

				//finalresaniso = tmin(sidelength / 3.5f * angpix, tmax(finalresaniso, sidelength / 100.0f * angpix));

				if (!dofilterhalfmaps)
				{
					tfloat fscweight = 1 - tmax(0, tmin(1, radius - finalresaniso));
					valavg *= fscweight;
					valavg *= fscweightaniso;

					if (isnan(valavg.x))
						valavg.x = 0;
					if (isnan(valavg.y))
						valavg.y = 0;
					d_volume2[id] = valavg;	// unsharpened

					tfloat res = radius / (sidelength * angpix);
					tfloat bfaccorr = exp(-bfacaniso * 0.25f * res * res);

					valavg *= bfaccorr;

					if (isnan(valavg.x))
						valavg.x = 0;
					if (isnan(valavg.y))
						valavg.y = 0;
					d_volume1[id] = valavg;	// sharpened
				}
				else
				{
					tfloat fscweight = 1 - tmax(0, tmin(1, radius - finalresaniso));
					val1 *= fscweight;
					val1 *= fscweightaniso;
					val2 *= fscweight;
					val2 *= fscweightaniso;

					if (isnan(val1.x))
						val1.x = 0;
					if (isnan(val1.y))
						val1.y = 0;
					if (isnan(val2.x))
						val2.x = 0;
					if (isnan(val2.y))
						val2.y = 0;

					d_volume1[id] = val1;	// only unsharpened output
					d_volume2[id] = val2;	// only unsharpened output
				}
			}
		}

		// Write out local resolution and bfactor values
		if (threadIdx.x == 0)
		{
			if (anisotropic)
			{
				d_resolution[blockIdx.x] = (finalres[0] + finalres[1] + finalres[2] + finalres[3] + finalres[4] + finalres[5] + finalres[6]) / 7;
				d_bfactors[blockIdx.x] = (finalbfac[0] + finalbfac[1] + finalbfac[2] + finalbfac[3] + finalbfac[4] + finalbfac[5] + finalbfac[6]) / 7;
			}
			else
			{
				d_resolution[blockIdx.x] = finalres[0];
				if (dolocalbfac)
					d_bfactors[blockIdx.x] = finalbfac[0];
				else
					d_bfactors[blockIdx.x] = globalbfac;
			}
		}
	}
}