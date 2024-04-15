#include "include/Functions.h"
using namespace gtom;

__global__ void ScaleNormCorrSumKernel(float2* d_simcoords, float* d_sim, float* d_scale, float* d_target, CTFParamsLean* d_params, float* d_scores, uint length);

/*

Supplied with a stack of frames, and extraction positions for sub-regions, this method 
extracts portions of each frame, computes the FT, and averages the results as follows:

-3D full fitting: d_output contains all individual spectra from each frame
-2D spatial fitting: d_output contains averages for all positions over all frames
-1D temporal fitting: d_output contains averages for all frames over all positions
-0D no fitting: d_outputall equals d_outputmean

*/

__declspec(dllexport) void CreateSpectra(float* d_frame, 
										int2 dimsframe, 
										int nframes, 
										int3* h_origins, 
										int norigins, 
										int2 dimsregion, 
										int3 ctfgrid, 
										int2 dimsregionscaled,
										float* d_outputall,
										float* d_outputmean,
										cufftHandle planforw,
										cufftHandle planback)
{

	int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, norigins * sizeof(int3));
	tfloat* d_tempspectra;
	cudaMalloc((void**)&d_tempspectra, tmax(norigins, nframes) * Elements2(dimsregion) * sizeof(tfloat));
	tfloat* d_tempspectrascaled;
	cudaMalloc((void**)&d_tempspectrascaled, tmax(norigins, nframes) * Elements2(dimsregionscaled) * sizeof(tfloat));
	tfloat* d_tempaverages;
	cudaMalloc((void**)&d_tempaverages, nframes * ElementsFFT2(dimsregionscaled) * sizeof(tfloat));

    tfloat* d_extracted;
    cudaMalloc((void**)&d_extracted, norigins * Elements2(dimsregion) * sizeof(tfloat));
    tcomplex* d_extractedft;
    cudaMalloc((void**)&d_extractedft, norigins * ElementsFFT2(dimsregion) * sizeof(tcomplex));

	bool ctfspace = ctfgrid.x * ctfgrid.y > 1;
	bool ctftime = ctfgrid.z > 1;
	int nspectra = (ctfspace || ctftime) ? (ctfspace ? norigins : 1) * (ctftime ? ctfgrid.z : 1) : 1;

	int pertimegroup = nframes / ctfgrid.z;

	// Temp spectra will be summed up to be averaged later
	d_ValueFill(d_outputall, ElementsFFT2(dimsregionscaled) * nspectra, 0.0f);

	cufftHandle ownplanforw = planforw > 0 ? planforw : d_FFTR2CGetPlan(2, toInt3(dimsregion), norigins);
	cufftHandle ownplanback = planback > 0 ? planback : d_IFFTC2RGetPlan(2, toInt3(dimsregionscaled), norigins);

    if (ctftime || !ctfspace)
    {
        for (size_t z = 0; z < nframes; z++)
        {
            size_t framegroup = z / pertimegroup;
            if (framegroup >= ctfgrid.z)
                break;

            // Write spectra to temp and reduce them to a temporary average spectrum
            d_CTFPeriodogram(d_frame + Elements2(dimsframe) * z, dimsframe, d_origins, norigins, dimsregion, dimsregion, d_tempspectra, false, ownplanforw, d_extracted, d_extractedft);
            //d_WriteMRC(d_tempspectra, toInt3(dimsregion.x / 2 + 1, dimsregion.y, norigins), "d_tempscpectra.mrc");

            d_RemapHalfFFT2FullFFT(d_tempspectra, d_tempspectra, toInt3(dimsregion), norigins);
            //d_WriteMRC(d_tempspectra, toInt3(dimsregion.x, dimsregion.y, norigins), "d_tempscpectra.mrc");

            d_Scale(d_tempspectra, d_tempspectrascaled, toInt3(dimsregion), toInt3(dimsregionscaled), T_INTERP_FOURIER, &ownplanforw, &ownplanback, norigins);

            d_RemapFullFFT2HalfFFT(d_tempspectrascaled, d_tempspectrascaled, toInt3(dimsregionscaled), norigins);
            //d_WriteMRC(d_tempspectrascaled, toInt3(dimsregionscaled.x / 2 + 1, dimsregionscaled.y, norigins), "d_tempscpectrascaled.mrc");

            d_ReduceMean(d_tempspectrascaled, d_tempaverages + ElementsFFT2(dimsregionscaled) * z, ElementsFFT2(dimsregionscaled), norigins);

            // Spatially resolved, add to output which has norigins spectra
            if (ctfspace)
            {
                d_AddVector(d_outputall + ElementsFFT2(dimsregionscaled) * norigins * framegroup, d_tempspectrascaled, d_outputall + ElementsFFT2(dimsregionscaled) * norigins * framegroup, ElementsFFT2(dimsregionscaled) * norigins);
            }
            else if (ctftime)
            {
                d_AddVector(d_outputall + ElementsFFT2(dimsregionscaled) * framegroup, d_tempaverages + ElementsFFT2(dimsregionscaled) * z, d_outputall + ElementsFFT2(dimsregionscaled) * framegroup, ElementsFFT2(dimsregionscaled));
            }
        }

        if (pertimegroup > 1)
            d_DivideByScalar(d_outputall, d_outputall, ElementsFFT2(dimsregionscaled) * nspectra, (tfloat)pertimegroup);

        // Just average over all individual spectra in d_outputall
        d_ReduceMean(d_tempaverages, d_outputmean, ElementsFFT2(dimsregionscaled), nframes);
    }
    else if (ctfspace)
    {
        tfloat* d_tempspectraaccumulated;
        cudaMalloc((void**)&d_tempspectraaccumulated, norigins * ElementsFFT2(dimsregion) * sizeof(tfloat));
        d_ValueFill(d_tempspectraaccumulated, ElementsFFT2(dimsregion) * norigins, (tfloat)0);

        for (size_t z = 0; z < nframes; z++)
        {
            // Write spectra to temp and reduce them to a temporary average spectrum
            d_CTFPeriodogram(d_frame + Elements2(dimsframe) * z, dimsframe, d_origins, norigins, dimsregion, dimsregion, d_tempspectra, false, ownplanforw, d_extracted, d_extractedft);
            //d_WriteMRC(d_tempspectra, toInt3(dimsregion.x / 2 + 1, dimsregion.y, norigins), "d_tempscpectra.mrc");

            // Spatially resolved, add to output which has norigins spectra
            d_AddVector(d_tempspectraaccumulated, d_tempspectra, d_tempspectraaccumulated, ElementsFFT2(dimsregion) * norigins);
        }

        d_RemapHalfFFT2FullFFT(d_tempspectraaccumulated, d_tempspectra, toInt3(dimsregion), norigins);
        //d_WriteMRC(d_tempspectra, toInt3(dimsregion.x, dimsregion.y, norigins), "d_tempscpectra.mrc");

        d_Scale(d_tempspectra, d_tempspectrascaled, toInt3(dimsregion), toInt3(dimsregionscaled), T_INTERP_FOURIER, &ownplanforw, &ownplanback, norigins);

        d_RemapFullFFT2HalfFFT(d_tempspectrascaled, d_outputall, toInt3(dimsregionscaled), norigins);
        //d_WriteMRC(d_tempspectrascaled, toInt3(dimsregionscaled.x / 2 + 1, dimsregionscaled.y, norigins), "d_tempscpectrascaled.mrc");

        d_ReduceMean(d_outputall, d_outputmean, ElementsFFT2(dimsregionscaled), norigins);

        cudaFree(d_tempspectraaccumulated);
    }
    else
        throw;

	if (planback <= 0)
		cufftDestroy(ownplanback);
	if (planforw <= 0)
		cufftDestroy(ownplanforw);
	
	// 0D case, only one average spectrum in outputall
	if (nspectra == 1)
		cudaMemcpy(d_outputall, d_outputmean, ElementsFFT2(dimsregionscaled) * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(d_extractedft);
    cudaFree(d_extracted);
	cudaFree(d_origins);
	cudaFree(d_tempspectrascaled);
	cudaFree(d_tempspectra);
	cudaFree(d_tempaverages);
}

__declspec(dllexport) void CTFMakeAverage(float* d_ps, float2* d_pscoords, uint length, uint sidelength, CTFParams* h_sourceparams, CTFParams targetparams, uint minbin, uint maxbin, uint batch, float* d_output)
{
	uint nbins = maxbin - minbin;
	d_CTFRotationalAverageToTarget((tfloat*)d_ps, d_pscoords, length, sidelength, h_sourceparams, targetparams, d_output, minbin, maxbin, batch);
}

__declspec(dllexport) void CTFCompareToSim(float* d_ps, float2* d_pscoords, float* d_scale, uint length, CTFParams* h_sourceparams, float* h_scores, uint batch)
{
	float* d_sim;
	cudaMalloc((void**)&d_sim, length * batch * sizeof(float));
	float* d_scores;
	cudaMalloc((void**)&d_scores, batch * sizeof(float));

	CTFParamsLean* h_lean;
	cudaMallocHost((void**)&h_lean, batch * sizeof(CTFParamsLean));
	#pragma omp parallel for
	for (int i = 0; i < batch; i++)
		h_lean[i] = CTFParamsLean(h_sourceparams[i], toInt3(1, 1, 1));	// Sidelength and pixelsize are already included in d_addresses
	CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
	cudaFreeHost(h_lean);

	//d_CTFSimulate(h_sourceparams, d_pscoords, d_sim, length, true, batch);

	int TpB = 128;
	dim3 grid = dim3(batch, 1, 1);
	ScaleNormCorrSumKernel <<<grid, TpB>>> (d_pscoords, d_sim, d_scale, d_ps, d_lean, d_scores, length);

	//d_MultiplyByVector(d_sim, d_scale, d_sim, length, batch);
	//d_NormMonolithic(d_sim, d_sim, length, T_NORM_MEAN01STD, batch);
	//d_WriteMRC(d_sim, toInt3(207, 512, 1), "d_sim.mrc");
	//d_WriteMRC(d_ps, toInt3(207, 512, 1), "d_ps.mrc");

	//d_MultiplyByVector(d_ps, d_sim, d_sim, length * batch);

	//d_SumMonolithic(d_sim, d_scores, length, batch);

	cudaMemcpy(h_scores, d_scores, batch * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_lean);
	cudaFree(d_sim);
	cudaFree(d_scores);

	//for (uint i = 0; i < batch; i++)
		//h_scores[i] /= (float)length;
}

__global__ void ScaleNormCorrSumKernel(float2* d_simcoords, float* d_sim, float* d_scale, float* d_target, CTFParamsLean* d_params, float* d_scores, uint length)
{
	__shared__ float s_sums1[128];
	__shared__ float s_sums2[128];
	__shared__ float s_mean, s_stddev;

	d_sim += blockIdx.x * length;
	d_target += blockIdx.x * length;

	CTFParamsLean params = d_params[blockIdx.x];
	float proteindefocus = params.defocus;

	float sum1 = 0.0, sum2 = 0.0;
	for (uint i = threadIdx.x; i < length; i += blockDim.x)
	{
		float2 simcoords = d_simcoords[i];
		float pixelsize = params.pixelsize + params.pixeldelta * __cosf(2.0f * (simcoords.y - params.pixelangle));
		simcoords.x /= pixelsize;

		params.defocus = proteindefocus;
		float ctfprotein = d_GetCTF<true, false>(simcoords.x, simcoords.y, 0, params);

		float val = abs(ctfprotein);

		d_sim[i] = val;
		sum1 += val;
		sum2 += val * val;
	}
	s_sums1[threadIdx.x] = sum1;
	s_sums2[threadIdx.x] = sum2;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < 128; i++)
		{
			sum1 += s_sums1[i];
			sum2 += s_sums2[i];
		}

		s_mean = sum1 / (float)length;
		s_stddev = sqrt(((float)length * sum2 - (sum1 * sum1))) / (float)length;
	}
	__syncthreads();

	float mean = s_mean;
	float stddev = s_stddev > 0.0f ? 1.0f / s_stddev : 0.0f;

	sum1 = 0.0f;
	for (uint i = threadIdx.x; i < length; i += blockDim.x)
		sum1 += (d_sim[i] - mean) * stddev * d_target[i];
	s_sums1[threadIdx.x] = sum1;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < 128; i++)
			sum1 += s_sums1[i];

		d_scores[blockIdx.x] = sum1 / (float)length;
	}
}