#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/DeviceFunctions.cuh"


namespace gtom
{
	template<int maxbins> __global__ void CTFRotationalAverageKernel(tfloat* d_input, float2* d_inputcoords, tfloat* d_average, tfloat* d_averageweights, uint inputlength, uint sidelength, ushort numbins, ushort freqlow, ushort freqhigh, CTFParamsLean* d_params);
	template<class T> __global__ void CTFRotationalAverageToTargetKernel(T* d_input,
																		float2* d_inputcoords,
																		tfloat* d_average,
																		tfloat* d_averageweights,
																		uint inputlength,
																		uint sidelength,
																		ushort numbins,
																		ushort freqlow,
																		ushort freqhigh,
																		CTFParamsLean* d_params,
																		tfloat2 ny,
																		tfloat2 cs,
																		tfloat2 targetpx,
																		tfloat2 targetz,
																		tfloat2 lambda);
	template<int maxbins> __global__ void CTFRotationalAverageToTargetDeterministicKernel(tfloat* d_input,
																							float2* d_inputcoords,
																							tfloat* d_average,
																							tfloat* d_averageweights,
																							uint inputlength,
																							ushort numbins,
																							ushort freqlow,
																							CTFParamsLean* d_params,
																							tfloat2 ny,
																							tfloat2 cs,
																							tfloat2 targetpx,
																							tfloat2 targetz,
																							tfloat2 lambda);


	////////////////////////////////////////////////////////////
	//Correct the CTF function to make all amplitudes positive//
	////////////////////////////////////////////////////////////

	void d_CTFRotationalAverage(tfloat* d_re, int2 dimsinput, CTFParams* h_params, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch)
	{
		float2* h_targetcoords = (float2*)malloc(ElementsFFT2(dimsinput) * sizeof(float2));
		float invhalfsize = 1.0f / (float)dimsinput.x;
		float center = dimsinput.x / 2;
		for (int y = 0; y < dimsinput.y; y++)
		{
			for (int x = 0; x < ElementsFFT1(dimsinput.x); x++)
			{
				float2 point = make_float2(x - center, y - center);
				float angle = atan2(point.y, point.x);
				h_targetcoords[y * ElementsFFT1(dimsinput.x) + x] = make_float2(sqrt(point.x * point.x + point.y * point.y) * invhalfsize, angle);
			}
		}
		float2* d_targetcoords = (float2*)CudaMallocFromHostArray(h_targetcoords, ElementsFFT2(dimsinput) * sizeof(float2));
		free(h_targetcoords);

		d_CTFRotationalAverage(d_re, d_targetcoords, ElementsFFT2(dimsinput), dimsinput.x, h_params, d_average, freqlow, freqhigh, batch);
		cudaFree(d_targetcoords);
	}

	void d_CTFRotationalAverage(tfloat* d_input, float2* d_inputcoords, uint inputlength, uint sidelength, CTFParams* h_params, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch)
	{
		uint numbins = freqhigh - freqlow;

		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));
		for (uint i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i], toInt3(sidelength, sidelength, 1));
		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));

		dim3 TpB = dim3(192);
		dim3 grid = dim3(tmin(32, (inputlength + TpB.x - 1) / TpB.x), batch);

		tfloat* d_tempbins, *d_tempweights;
		cudaMalloc((void**)&d_tempbins, numbins * grid.x * grid.y * sizeof(tfloat));
		cudaMalloc((void**)&d_tempweights, numbins * grid.x * grid.y * sizeof(tfloat));

		if (numbins <= 513)
			CTFRotationalAverageKernel<513> << <grid, TpB >> > (d_input, d_inputcoords, d_tempbins, d_tempweights, inputlength, sidelength, numbins, freqlow, freqhigh, d_lean);
		else if (numbins <= 1025)
			CTFRotationalAverageKernel<1025> << <grid, TpB >> > (d_input, d_inputcoords, d_tempbins, d_tempweights, inputlength, sidelength, numbins, freqlow, freqhigh, d_lean);
		else if (numbins <= 2049)
			CTFRotationalAverageKernel<2049> << <grid, TpB >> > (d_input, d_inputcoords, d_tempbins, d_tempweights, inputlength, sidelength, numbins, freqlow, freqhigh, d_lean);
		else if (numbins <= 4097)
			CTFRotationalAverageKernel<4097> << <grid, TpB >> > (d_input, d_inputcoords, d_tempbins, d_tempweights, inputlength, sidelength, numbins, freqlow, freqhigh, d_lean);
		else
			throw;

		d_ReduceMeanWeighted(d_tempbins, d_tempweights, d_average, numbins, grid.x, batch);
		//cudaMemcpy(d_average, d_tempbins, numbins * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);

		cudaFree(d_tempweights);
		cudaFree(d_tempbins);
		cudaFree(d_lean);
		free(h_lean);
	}
	
	template<class T> void d_CTFRotationalAverageToTarget(T* d_input, float2* d_inputcoords, uint inputlength, uint sidelength, CTFParams* h_params, CTFParams targetparams, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch)
	{
		uint numbins = freqhigh - freqlow;

		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));
		for (uint i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i], toInt3(sidelength, sidelength, 1));
		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));

		CTFParamsLean targetparamslean = CTFParamsLean(targetparams, toInt3(sidelength, sidelength, 1));

		dim3 TpB = dim3(192);
		dim3 grid = dim3(tmin(32, (inputlength + TpB.x - 1) / TpB.x), batch);

		tfloat* d_tempbins, *d_tempweights;
		cudaMalloc((void**)&d_tempbins, numbins * grid.x * grid.y * sizeof(tfloat));
		cudaMalloc((void**)&d_tempweights, numbins * grid.x * grid.y * sizeof(tfloat));

		tfloat2 ny = tfloat2(1.0 / (sidelength * sidelength), 1.0 / pow(sidelength, 4.0));
		tfloat2 cs = tfloat2(h_lean[0].Cs, h_lean[0].Cs * h_lean[0].Cs);
		tfloat2 targetpx = tfloat2(pow(targetparamslean.pixelsize, 2.0), pow(targetparamslean.pixelsize, 4.0));
		tfloat2 targetz = tfloat2(targetparamslean.defocus, targetparamslean.defocus * targetparamslean.defocus);
		tfloat2 lambda = tfloat2(pow(targetparamslean.lambda, 2.0), pow(targetparamslean.lambda, 4.0));

		if (numbins <= 1024)
			CTFRotationalAverageToTargetKernel << <grid, TpB >> > (d_input, d_inputcoords, d_tempbins, d_tempweights, inputlength, sidelength, numbins, freqlow, freqhigh, d_lean, ny, cs, targetpx, targetz, lambda);
		else
			throw;

		d_ReduceMeanWeighted(d_tempbins, d_tempweights, d_average, numbins, grid.x * batch, 1);

		cudaFree(d_tempweights);
		cudaFree(d_tempbins);
		cudaFree(d_lean);
		free(h_lean);
	}
	template void d_CTFRotationalAverageToTarget<tfloat>(tfloat* d_input, float2* d_inputcoords, uint inputlength, uint sidelength, CTFParams* h_params, CTFParams targetparams, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch);
	template void d_CTFRotationalAverageToTarget<tcomplex>(tcomplex* d_input, float2* d_inputcoords, uint inputlength, uint sidelength, CTFParams* h_params, CTFParams targetparams, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch);

	void d_CTFRotationalAverageToTargetDeterministic(tfloat* d_input, float2* d_inputcoords, uint inputlength, uint sidelength, CTFParams* h_params, CTFParams targetparams, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch)
	{
		uint numbins = freqhigh - freqlow;

		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));
		for (uint i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i], toInt3(sidelength, sidelength, 1));
		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));

		CTFParamsLean targetparamslean = CTFParamsLean(targetparams, toInt3(sidelength, sidelength, 1));

		dim3 TpB = dim3(128);
		dim3 grid = dim3(tmin(64, (inputlength + TpB.x - 1) / TpB.x), 1);

		tfloat* d_tempbins, *d_tempweights;
		cudaMalloc((void**)&d_tempbins, numbins * grid.x * batch * sizeof(tfloat));
		cudaMalloc((void**)&d_tempweights, numbins * grid.x * batch * sizeof(tfloat));

		tfloat2 ny = tfloat2(1.0 / (sidelength * sidelength), 1.0 / pow(sidelength, 4.0));
		tfloat2 cs = tfloat2(h_lean[0].Cs, h_lean[0].Cs * h_lean[0].Cs);
		tfloat2 targetpx = tfloat2(pow(targetparamslean.pixelsize, 2.0), pow(targetparamslean.pixelsize, 4.0));
		tfloat2 targetz = tfloat2(targetparamslean.defocus, targetparamslean.defocus * targetparamslean.defocus);
		tfloat2 lambda = tfloat2(pow(targetparamslean.lambda, 2.0), pow(targetparamslean.lambda, 4.0));
		
		for (int b = 0; b < batch; b++)
			if (numbins <= 512)
				CTFRotationalAverageToTargetDeterministicKernel<512> << <grid, TpB >> > (d_input + inputlength * b, 
																						d_inputcoords, 
																						d_tempbins + numbins * grid.x * b, 
																						d_tempweights + numbins * grid.x * b, 
																						inputlength, 
																						numbins, 
																						freqlow, 
																						d_lean + b, 
																						ny, 
																						cs, 
																						targetpx, 
																						targetz, 
																						lambda);
			else if (numbins <= 1024)
				CTFRotationalAverageToTargetDeterministicKernel<1024> << <grid, TpB >> > (d_input + inputlength * b, 
																						d_inputcoords,
																						d_tempbins + numbins * grid.x * b,
																						d_tempweights + numbins * grid.x * b,
																						inputlength,
																						numbins,
																						freqlow,
																						d_lean + b,
																						ny,
																						cs,
																						targetpx,
																						targetz,
																						lambda);
			else if (numbins <= 2048)
				CTFRotationalAverageToTargetDeterministicKernel<2048> << <grid, TpB >> > (d_input + inputlength * b, 
																						d_inputcoords,
																						d_tempbins + numbins * grid.x * b,
																						d_tempweights + numbins * grid.x * b,
																						inputlength,
																						numbins,
																						freqlow,
																						d_lean + b,
																						ny,
																						cs,
																						targetpx,
																						targetz,
																						lambda);
			else if (numbins <= 4096)
				CTFRotationalAverageToTargetDeterministicKernel<4096> << <grid, TpB >> > (d_input + inputlength * b, 
																						d_inputcoords,
																						d_tempbins + numbins * grid.x * b,
																						d_tempweights + numbins * grid.x * b,
																						inputlength,
																						numbins,
																						freqlow,
																						d_lean + b,
																						ny,
																						cs,
																						targetpx,
																						targetz,
																						lambda);
			else
				throw;

		/*if (h_consider != NULL)
		{
			std::vector<int> positions;
			for (int i = 0; i < batch; i++)
				if (h_consider[i] > 0)
					positions.push_back(i);

			tfloat* d_densebins;
			cudaMalloc((void**)&d_densebins, numbins * grid.x * positions.size() * sizeof(tfloat));
			tfloat* d_denseweights;
			cudaMalloc((void**)&d_denseweights, numbins * grid.x * positions.size() * sizeof(tfloat));

			for (int i = 0; i < positions.size(); i++)
			{
				cudaMemcpy(d_densebins + numbins * grid.x * i, d_tempbins + numbins * grid.x * positions[i], numbins * grid.x * sizeof(tfloat), cudaMemcpyDeviceToDevice);
				cudaMemcpy(d_denseweights + numbins * grid.x * i, d_tempweights + numbins * grid.x * positions[i], numbins * grid.x * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			}

			d_ReduceMeanWeighted(d_densebins, d_denseweights, d_average, numbins, grid.x * positions.size(), 1);

			cudaFree(d_densebins);
			cudaFree(d_denseweights);
		}
		else*/
		{
			d_ReduceMeanWeighted(d_tempbins, d_tempweights, d_average, numbins, grid.x * batch, 1);
			//cudaMemcpy(d_average, d_tempbins, numbins * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		}

		cudaFree(d_tempweights);
		cudaFree(d_tempbins);
		cudaFree(d_lean);
		free(h_lean);
	}


	////////////////
	//CUDA kernels//
	////////////////

	__device__ tfloat d_CTFRescale(tfloat srcx,
									tfloat ny2, tfloat ny4,
									tfloat cs, tfloat cs2,
									tfloat srcpx,
									tfloat trgtpx2, tfloat trgtpx4,
									tfloat srcz,
									tfloat targetz, tfloat targetz2,
									tfloat lambda2, tfloat lambda4)
	{
		tfloat srcx2 = srcx * srcx;
		tfloat srcx4 = srcx2 * srcx2;
		tfloat srcpx2 = srcpx * srcpx;
		tfloat srcpx4 = srcpx2 * srcpx2;

		double summand1 = (double)cs2 * lambda4 * ny4 * srcx4;
		double summand2 = 2.0 * cs * lambda2 * ny2 * srcpx2 * srcx2 * srcz;
		double summand3 = (double)srcpx4 * targetz2;

		double firstroot = -sqrt(trgtpx4 * srcpx4 * (summand1 + summand2 + summand3));
		double numerator = firstroot + trgtpx2 * srcpx4 * abs(targetz);
		double denominator = (double)cs * lambda2 * ny2 * srcpx4;

		tfloat x = (tfloat)sqrt(abs(numerator / denominator));

		return x;
	}

	template<int maxbins> __global__ void CTFRotationalAverageKernel(tfloat* d_input, float2* d_inputcoords, tfloat* d_average, tfloat* d_averageweights, uint inputlength, uint sidelength, ushort numbins, ushort freqlow, ushort freqhigh, CTFParamsLean* d_params)
	{
		__shared__ tfloat s_bins[maxbins], s_weights[maxbins];
		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			s_bins[i] = 0;
			s_weights[i] = 0;
		}
		__syncthreads();

		CTFParamsLean p = d_params[blockIdx.y];
		d_input += blockIdx.y * inputlength;
		d_average += blockIdx.y * gridDim.x * numbins;
		d_averageweights += blockIdx.y * gridDim.x * numbins;

		double cs2 = p.Cs * p.Cs;
		double defocus2 = p.defocus * p.defocus;
		double lambda2 = p.lambda * p.lambda;
		double lambda4 = lambda2 * lambda2;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < inputlength; id += gridDim.x * blockDim.x)
		{
			float radius = d_inputcoords[id].x;
			float angle = d_inputcoords[id].y;

			radius *= p.ny;
			double radius2 = radius * radius;
			double radius4 = radius2 * radius2;

			double astdefocus = p.defocus - p.defocusdelta * cos(2.0f * (angle + (float)p.astigmatismangle));
			double originalradius = sqrt(abs(abs(p.defocus) - sqrt(cs2 * radius4 * lambda4 + 2.0 * p.Cs * astdefocus * radius2 * lambda2 + defocus2)) / (p.Cs * lambda2));
			originalradius /= p.ny * 2.0 / (double)sidelength;

			tfloat val = d_input[id];
			short lowbin = floor(originalradius), highbin = lowbin + 1;
			tfloat lowweight = (tfloat)(1 + lowbin) - originalradius, highweight = (tfloat)1 - lowweight;
			if (lowbin >= freqlow && lowbin < freqhigh)
			{
				lowbin -= freqlow;
				atomicAdd(s_bins + lowbin, val * lowweight);
				atomicAdd(s_weights + lowbin, lowweight);
			}
			if (highbin >= freqlow && highbin < freqhigh)
			{
				highbin -= freqlow;
				atomicAdd(s_bins + highbin, val * highweight);
				atomicAdd(s_weights + highbin, highweight);
			}
		}
		__syncthreads();

		d_average += blockIdx.x * numbins;
		d_averageweights += blockIdx.x * numbins;
		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			d_average[i] = s_weights[i] != 0 ? s_bins[i] / s_weights[i] : 0;
			d_averageweights[i] = s_weights[i];
		}
	}

	template<class T> __global__ void CTFRotationalAverageToTargetKernel(T* d_input, 
																		float2* d_inputcoords, 
																		tfloat* d_average, 
																		tfloat* d_averageweights, 
																		uint inputlength, 
																		uint sidelength, 
																		ushort numbins, 
																		ushort freqlow, 
																		ushort freqhigh, 
																		CTFParamsLean* d_params,
																		tfloat2 ny,
																		tfloat2 cs,
																		tfloat2 targetpx,
																		tfloat2 targetz,
																		tfloat2 lambda)
	{
		__shared__ tfloat s_bins[1024], s_weights[1024];
		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			s_bins[i] = 0;
			s_weights[i] = 0;
		}
		__syncthreads();

		CTFParamsLean p = d_params[blockIdx.y];
		d_input += blockIdx.y * inputlength;
		d_average += blockIdx.y * gridDim.x * numbins;
		d_averageweights += blockIdx.y * gridDim.x * numbins;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < inputlength; id += gridDim.x * blockDim.x)
		{
			float sourcex = d_inputcoords[id].x;
			float angle = d_inputcoords[id].y;
			
			tfloat sourcepx = p.pixelsize + p.pixeldelta * cos(2.0f * (angle - (float)p.pixelangle));
			tfloat sourcez = p.defocus + p.defocusdelta * cos(2.0f * (angle - (float)p.astigmatismangle));

			tfloat targetx = d_CTFRescale(sourcex,
										  ny.x, ny.y,
										  cs.x, cs.y,
										  sourcepx,
										  targetpx.x, targetpx.y,
										  sourcez,
										  targetz.x, targetz.y,
										  lambda.x, lambda.y);

			tfloat val = d_input[id];
			short lowbin = floor(targetx), highbin = lowbin + 1;
			tfloat lowweight = (tfloat)(1 + lowbin) - targetx, highweight = (tfloat)1 - lowweight;
			lowweight *= p.scale;
			highweight *= p.scale;

			if (lowbin >= freqlow && lowbin < freqhigh)
			{
				lowbin -= freqlow;
				atomicAdd(s_bins + lowbin, val * lowweight);
				atomicAdd(s_weights + lowbin, lowweight);
			}
			if (highbin >= freqlow && highbin < freqhigh)
			{
				highbin -= freqlow;
				atomicAdd(s_bins + highbin, val * highweight);
				atomicAdd(s_weights + highbin, highweight);
			}
		}
		__syncthreads();

		d_average += blockIdx.x * numbins;
		d_averageweights += blockIdx.x * numbins;
		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			d_average[i] = s_weights[i] != 0 ? s_bins[i] / s_weights[i] : 0;
			d_averageweights[i] = s_weights[i];
		}
	}

	template<> __global__ void CTFRotationalAverageToTargetKernel<tcomplex>(tcomplex* d_input,
																			float2* d_inputcoords,
																			tfloat* d_average,
																			tfloat* d_averageweights,
																			uint inputlength,
																			uint sidelength,
																			ushort numbins,
																			ushort freqlow,
																			ushort freqhigh,
																			CTFParamsLean* d_params,
																			tfloat2 ny,
																			tfloat2 cs,
																			tfloat2 targetpx,
																			tfloat2 targetz,
																			tfloat2 lambda)
	{
		__shared__ tfloat s_bins[1024], s_weights[1024];
		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			s_bins[i] = 0;
			s_weights[i] = 0;
		}
		__syncthreads();

		CTFParamsLean p = d_params[blockIdx.y];
		d_input += blockIdx.y * inputlength;
		d_average += blockIdx.y * gridDim.x * numbins;
		d_averageweights += blockIdx.y * gridDim.x * numbins;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < inputlength; id += gridDim.x * blockDim.x)
		{
			float sourcex = d_inputcoords[id].x;
			float angle = d_inputcoords[id].y;

			tfloat sourcepx = p.pixelsize + p.pixeldelta * cos(2.0f * (angle - (float)p.pixelangle));
			tfloat sourcez = p.defocus + p.defocusdelta * cos(2.0f * (angle - (float)p.astigmatismangle));

			tfloat targetx = d_CTFRescale(sourcex,
				ny.x, ny.y,
				cs.x, cs.y,
				sourcepx,
				targetpx.x, targetpx.y,
				sourcez,
				targetz.x, targetz.y,
				lambda.x, lambda.y);

			tcomplex valc = d_input[id];
			tfloat val = sqrt(valc.x * valc.x + valc.y * valc.y);
			short lowbin = floor(targetx), highbin = lowbin + 1;
			tfloat lowweight = (tfloat)(1 + lowbin) - targetx, highweight = (tfloat)1 - lowweight;

			if (lowbin >= freqlow && lowbin < freqhigh)
			{
				lowbin -= freqlow;
				atomicAdd(s_bins + lowbin, val * lowweight);
				atomicAdd(s_weights + lowbin, lowweight);
			}
			if (highbin >= freqlow && highbin < freqhigh)
			{
				highbin -= freqlow;
				atomicAdd(s_bins + highbin, val * highweight);
				atomicAdd(s_weights + highbin, highweight);
			}
		}
		__syncthreads();

		d_average += blockIdx.x * numbins;
		d_averageweights += blockIdx.x * numbins;
		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			d_average[i] = s_weights[i] != 0 ? s_bins[i] / s_weights[i] : 0;
			d_averageweights[i] = s_weights[i];
		}
	}

	template<int maxbins> __global__ void CTFRotationalAverageToTargetDeterministicKernel(tfloat* d_input,
																						float2* d_inputcoords,
																						tfloat* d_average,
																						tfloat* d_averageweights,
																						uint inputlength,
																						ushort numbins,
																						ushort freqlow,
																						CTFParamsLean* d_params,
																						tfloat2 ny,
																						tfloat2 cs,
																						tfloat2 targetpx,
																						tfloat2 targetz,
																						tfloat2 lambda)
	{
		__shared__ tfloat s_bins[maxbins], s_weights[maxbins];
		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			s_bins[i] = 0;
			s_weights[i] = 0;
		}
		__syncthreads();

		CTFParamsLean p = d_params[blockIdx.y];
		d_input += blockIdx.y * inputlength;
		d_average += (blockIdx.y * gridDim.x + blockIdx.x) * numbins;
		d_averageweights += (blockIdx.y * gridDim.x + blockIdx.x) * numbins;

		for (uint id = blockIdx.x; id < inputlength; id += gridDim.x)
		{
			float sourcex = d_inputcoords[id].x;
			float angle = d_inputcoords[id].y;

			tfloat sourcepx = p.pixelsize + p.pixeldelta * cos(2.0f * (angle - (float)p.pixelangle));
			tfloat sourcez = p.defocus + p.defocusdelta * cos(2.0f * (angle - (float)p.astigmatismangle));

			tfloat targetx = d_CTFRescale(sourcex,
											ny.x, ny.y,
											cs.x, cs.y,
											sourcepx,
											targetpx.x, targetpx.y,
											sourcez,
											targetz.x, targetz.y,
											lambda.x, lambda.y);

			tfloat val = d_input[id];

			for (ushort bin = threadIdx.x; bin < numbins; bin += blockDim.x)
			{
				float dist = abs(targetx - (float)(freqlow + bin));
				if (dist < 4.0f)
				{
					float weight = sinc(dist);
					s_bins[bin] += val * weight;
					s_weights[bin] += weight;
				}
			}
		}
		__syncthreads();

		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			d_average[i] = s_weights[i] != 0 ? s_bins[i] / s_weights[i] : 0;
			d_averageweights[i] = s_weights[i];
		}
	}
}