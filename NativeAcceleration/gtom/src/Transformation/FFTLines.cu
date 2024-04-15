#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
	template<bool cubicinterp> __global__ void FFTLinesKernel(cudaTex t_Re, cudaTex t_Im, tcomplex* d_output, int2 dimsft, float anglestep, int linewidth);


	void d_FFTLines(tcomplex* d_input, tcomplex* d_output, int2 dims, T_INTERP_MODE mode, int anglesteps, int linewidth, int batch)
	{
		int2 dimslines = toInt2(dims.x / 2 + 1, anglesteps * linewidth);
		tfloat* d_temp;
		cudaMalloc((void**)&d_temp, ElementsFFT2(dims) * 2 * sizeof(tfloat));
		float anglestep = PI / (float)anglesteps;

		for (int b = 0; b < batch; b++)
		{
			cudaArray* a_Re, *a_Im;
			cudaTex t_Re, t_Im;

			d_ConvertTComplexToSplitComplex(d_input + ElementsFFT2(dims) * b, d_temp, d_temp + ElementsFFT2(dims), ElementsFFT2(dims));
			d_RemapHalfFFT2Half(d_temp, d_temp, toInt3(dims), 2);

			if (mode == T_INTERP_CUBIC)
			{
				d_CubicBSplinePrefilter2D(d_temp, toInt2(dims.x / 2 + 1, dims.y));
				d_CubicBSplinePrefilter2D(d_temp + ElementsFFT2(dims), toInt2(dims.x / 2 + 1, dims.y));
			}

			d_BindTextureToArray(d_temp, a_Re, t_Re, toInt2(dims.x / 2 + 1, dims.y), cudaFilterModeLinear, false);
			d_BindTextureToArray(d_temp + ElementsFFT2(dims), a_Im, t_Im, toInt2(dims.x / 2 + 1, dims.y), cudaFilterModeLinear, false);

			dim3 TpB = dim3(min(256, NextMultipleOf(dimslines.x, 32)), linewidth);
			dim3 grid = dim3(anglesteps);

			if (mode == T_INTERP_LINEAR)
				FFTLinesKernel<false> << <grid, TpB >> > (t_Re, t_Im, d_output + Elements2(dimslines) * b, toInt2(dims.x / 2 + 1, dims.y), anglestep, linewidth);
			else if (mode == T_INTERP_CUBIC)
				FFTLinesKernel<true> << <grid, TpB >> > (t_Re, t_Im, d_output + Elements2(dimslines) * b, toInt2(dims.x / 2 + 1, dims.y), anglestep, linewidth);

			cudaDestroyTextureObject(t_Re);
			cudaDestroyTextureObject(t_Im);
			cudaFreeArray(a_Re);
			cudaFreeArray(a_Im);
		}

		cudaFree(d_temp);
	}

	template<bool cubicinterp> __global__ void FFTLinesKernel(cudaTex t_Re, cudaTex t_Im, tcomplex* d_output, int2 dimsft, float anglestep, int linewidth)
	{
		int line = (int)threadIdx.y - linewidth / 2;
		float angle = (float)blockIdx.x * anglestep + PIHALF;
		float cosangle = cos(angle);
		float sinangle = sin(angle);
		float center = (float)(dimsft.x - 1) + 0.5f;

		int outy = ((int)threadIdx.y + (linewidth + 1) / 2) % linewidth;
		d_output += (blockIdx.x * linewidth + threadIdx.y) * dimsft.x;

		for (uint id = threadIdx.x; id < dimsft.x; id += blockDim.x)
		{
			bool inverse = false;
			glm::vec2 pos = glm::vec2(id, line);
			pos = glm::vec2(pos.x * cosangle - pos.y * sinangle, pos.x * sinangle + pos.y * cosangle);
			if (pos.x > 1e-8f)
			{
				pos = -pos;
				inverse = true;
			}
			pos += center;

			tcomplex val;
			if (cubicinterp)
				val = make_cuComplex(cubicTex2D(t_Re, pos.x, pos.y), cubicTex2D(t_Im, pos.x, pos.y));
			else
				val = make_cuComplex(tex2D<tfloat>(t_Re, pos.x, pos.y), tex2D<tfloat>(t_Im, pos.x, pos.y));
			if (inverse)
				val = cconj(val);
			if (pos.y >= (float)dimsft.y)
				val = make_cuComplex(0, 0);

			d_output[id] = val;
		}
	}
}