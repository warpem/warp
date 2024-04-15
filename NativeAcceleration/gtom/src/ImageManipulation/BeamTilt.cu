#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Masking.cuh"


namespace gtom
{
	__global__ void BeamTiltKernel(tcomplex* d_input, tcomplex* d_output, int2 dims, uint elements, const tfloat2* __restrict__ d_beamtilt, const tfloat* __restrict__ d_factors);

	//////////////////////////
	//Corrects for beam tilt//
	//////////////////////////

	void d_BeamTilt(tcomplex* d_input, tcomplex* d_output, int2 dims, tfloat2* d_beamtilt, CTFParams* h_params, uint batch)
	{
		tfloat* h_factors;
		cudaMallocHost((void**)&h_factors, batch * sizeof(tfloat));
		for (uint b = 0; b < batch; b++)
		{
			CTFParamsLean lean = CTFParamsLean(h_params[b], toInt3(dims));
			tfloat boxsize = (tfloat)dims.x * (h_params[b].pixelsize * 1e10);
			tfloat factor = 1e-3f * PI2 * lean.Cs * lean.lambda * lean.lambda / (boxsize * boxsize * boxsize);
			h_factors[b] = factor;
		}
		tfloat* d_factors = (tfloat*)CudaMallocFromHostArray(h_factors, batch * sizeof(tfloat));
		cudaFreeHost(h_factors);

		int TpB = tmin(NextMultipleOf(ElementsFFT2(dims), 32), 128);
		dim3 grid = dim3((ElementsFFT2(dims) + TpB - 1) / TpB, batch);
		BeamTiltKernel <<<grid, TpB>>> (d_input, d_output, dims, ElementsFFT2(dims), d_beamtilt, d_factors);
	}	

	__global__ void BeamTiltKernel(tcomplex* d_input, tcomplex* d_output, int2 dims, uint elements, const tfloat2* __restrict__ d_beamtilt, const tfloat* __restrict__ d_factors)
	{
		d_input += elements * blockIdx.y;
		d_output += elements * blockIdx.y;
		tfloat2 beamtilt = d_beamtilt[blockIdx.y];
		tfloat factor = d_factors[blockIdx.y];

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
		{
			int y = id / ElementsFFT1(dims.x);
			uint x = id - y * ElementsFFT1(dims.x);

			tfloat xx = x;
			tfloat yy = y <= dims.y / 2 ? y : y - dims.y;

			tfloat phase = factor * (xx * xx + yy * yy) * (xx * beamtilt.x + yy * beamtilt.y);
			tcomplex shift = make_cuComplex(cos(phase), sin(phase));

			d_output[id] = cmul(d_input[id], shift);
		}
	}
}