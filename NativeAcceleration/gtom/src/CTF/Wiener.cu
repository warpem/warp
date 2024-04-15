#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	template<int ndims, bool fscweighted> __global__ void WienerPerFreqKernel(tcomplex* d_input, tfloat* d_fsc, tfloat nsr, tcomplex* d_output, tfloat* d_outputweights, int3 dims, CTFParamsLean* d_p);


	/////////////////////////////////////////////
	//Rectify the CTF envelope depending on SNR//
	/////////////////////////////////////////////

	void d_CTFWiener(tcomplex* d_input, int3 dimsinput, tfloat* d_fsc, CTFParams* h_params, tcomplex* d_output, tfloat* d_outputweights, uint batch)
	{
		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));
		for (uint b = 0; b < batch; b++)
			h_lean[b] = CTFParamsLean(h_params[b], dimsinput);
		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
		free(h_lean);

		dim3 TpB = dim3(min(128, NextMultipleOf(ElementsFFT2(dimsinput), 32)));
		dim3 grid = dim3((ElementsFFT2(dimsinput) + TpB.x - 1) / TpB.x, dimsinput.z, batch);

		if (DimensionCount(dimsinput) == 1)
			WienerPerFreqKernel<1, true> << <grid, TpB >> > (d_input, d_fsc, 1.0f, d_output, d_outputweights, dimsinput, d_lean);
		if (DimensionCount(dimsinput) == 2)
			WienerPerFreqKernel<2, true> << <grid, TpB >> > (d_input, d_fsc, 1.0f, d_output, d_outputweights, dimsinput, d_lean);
		else if (DimensionCount(dimsinput) == 3)
			WienerPerFreqKernel<3, true> << <grid, TpB >> > (d_input, d_fsc, 1.0f, d_output, d_outputweights, dimsinput, d_lean);

		cudaFree(d_lean);
	}

	void d_CTFWiener(tcomplex* d_input, int3 dimsinput, tfloat snr, CTFParams* h_params, tcomplex* d_output, tfloat* d_outputweights, uint batch)
	{
		if (snr <= 0)
			throw;

		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));
		for (uint b = 0; b < batch; b++)
			h_lean[b] = CTFParamsLean(h_params[b], dimsinput);
		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
		free(h_lean);

		dim3 TpB = dim3(min(128, NextMultipleOf(ElementsFFT2(dimsinput), 32)));
		dim3 grid = dim3((ElementsFFT2(dimsinput) + TpB.x - 1) / TpB.x, dimsinput.z, batch);

		if (DimensionCount(dimsinput) == 1)
			WienerPerFreqKernel<1, false> << <grid, TpB >> > (d_input, NULL, 1.0f / snr, d_output, d_outputweights, dimsinput, d_lean);
		if (DimensionCount(dimsinput) == 2)
			WienerPerFreqKernel<2, false> << <grid, TpB >> > (d_input, NULL, 1.0f / snr, d_output, d_outputweights, dimsinput, d_lean);
		else if (DimensionCount(dimsinput) == 3)
			WienerPerFreqKernel<3, false> << <grid, TpB >> > (d_input, NULL, 1.0f / snr, d_output, d_outputweights, dimsinput, d_lean);

		cudaFree(d_lean);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<int ndims, bool fscweighted> __global__ void WienerPerFreqKernel(tcomplex* d_input, tfloat* d_fsc, tfloat nsr, tcomplex* d_output, tfloat* d_outputweights, int3 dims, CTFParamsLean* d_p)
	{
		uint idxy = blockIdx.x * blockDim.x + threadIdx.x;
		if (idxy >= ElementsFFT2(dims))
			return;
		int idx = idxy % ElementsFFT1(dims.x);
		uint idy = idxy / ElementsFFT1(dims.x);
		uint idz = blockIdx.y;

		CTFParamsLean p = d_p[blockIdx.z];

		tfloat k, angle, radius;
		if (ndims == 1)
		{
			angle = 0.0;
			radius = idx;

			k = radius * p.ny;
		}
		else if (ndims == 2)
		{
			int y = dims.y - 1 - FFTShift(idy, dims.y) - dims.y / 2;

			float2 position = make_float2(-idx, y);
			angle = atan2(position.y, position.x);
			radius = sqrt(position.x * position.x + position.y * position.y);
			float pixelsize = p.pixelsize + p.pixeldelta * cos(2.0f * (angle - p.pixelangle));

			k = radius * p.ny / pixelsize;
		}
		else if (ndims == 3)
		{
			// No dims.x -... because angle is irrelevant
			int y = FFTShift(idy, dims.y) - dims.y / 2;
			int z = FFTShift(idz, dims.z) - dims.z / 2;

			float3 position = make_float3(idx, y, z);
			angle = 0.0;
			radius = sqrt(position.x * position.x + position.y * position.y + position.z * position.z);

			k = radius * p.ny;
		}


		{
			size_t offset;
			if (ndims == 1)
				offset = ElementsFFT1(dims.x) * blockIdx.z + idx;
			else if (ndims == 2)
				offset = ElementsFFT2(dims) * blockIdx.z + getOffset(idx, idy, dims.x / 2 + 1);
			else if (ndims == 3)
				offset = ElementsFFT(dims) * blockIdx.z + getOffset3(idx, idy, idz, dims.x / 2 + 1, dims.y);
			d_input += offset;
			d_output += offset;
			if (d_outputweights != NULL)
				d_outputweights += offset;
			if (fscweighted)
				d_fsc += dims.x / 2 * blockIdx.z;
		}

		double amplitude = 1;
		tfloat weight = 1;
		tcomplex input = *d_input;

		if (radius > 0)
		{
			amplitude = d_GetCTF<false, false>(k, angle, 0, p);

			if (fscweighted)
			{
				// Linear interpolation over the FSC curve
				tfloat fsc = abs(lerp(d_fsc[min(dims.x / 2 - 1, (int)radius)], d_fsc[min(dims.x / 2 - 1, (int)radius + 1)], radius - floor(radius)));

				// FSC too small, avoid numerical error in division later
				if (fsc < 1e-6f)
				{
					*d_output = make_cuComplex(0, 0);
					if (d_outputweights != NULL)
						*d_outputweights = 0;
					return;
				}
				// FSC significant enough, SNR = FSC / (1 - FSC), but Wiener needs 1/SNR
				else
					weight = amplitude / (amplitude * amplitude + (1.0f - fsc) / fsc);
			}
			else
			{
				weight = amplitude / (amplitude * amplitude + nsr);
			}
			//weight = amplitude < 0.0f ? 1.0f : 1.0f;
		}

		*d_output = make_cuComplex(input.x * weight, input.y * weight);
		//*d_output = make_cuComplex(amplitude, 0.0f);
		if (d_outputweights != NULL)
			*d_outputweights = amplitude * weight;
	}
}