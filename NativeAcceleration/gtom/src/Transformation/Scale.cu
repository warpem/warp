#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <int ndims, bool cubicinterp> __global__ void InterpolateKernel(tfloat* d_output, int3 dimsnew, cudaTex t_input, int3 dimsold, tfloat3 factor, tfloat3 offset);


	////////////////////////////////////////////////////////////////////////////////
	//Combines the functionality of TOM's tom_rescale and MATLAB's interp* methods// 
	////////////////////////////////////////////////////////////////////////////////

	void d_Scale(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, T_INTERP_MODE mode, cufftHandle* planforw, cufftHandle* planback, int batch, tcomplex* d_inputfft, tcomplex* d_outputfft)
	{
		//Both sizes should have an equal number of dimensions
		int ndims = DimensionCount(olddims);
		if (ndims != DimensionCount(newdims))
			throw;

		//All new dimensions must be either bigger than the old or smaller, not mixed
		int biggerdims = 0;
        int equaldims = 0;
		for (int i = 0; i < ndims; i++)
			if (((int*)&newdims)[i] > ((int*)&olddims)[i])
				biggerdims++;
            else if (((int*)&newdims)[i] == ((int*)&olddims)[i])
                equaldims++;

		if (biggerdims != 0 && biggerdims != (ndims - equaldims))
			throw;

		if (mode == T_INTERP_LINEAR || mode == T_INTERP_CUBIC)
		{
			cudaArray* a_image;
			cudaTex t_image;
			tfloat* d_temp;
			cudaMalloc((void**)&d_temp, Elements(olddims) * sizeof(tfloat));

			for (int b = 0; b < batch; b++)
			{
				cudaMemcpy(d_temp, d_input + Elements(olddims) * b, Elements(olddims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
				if (mode == T_INTERP_CUBIC)
				{
					if (ndims == 2)
						d_CubicBSplinePrefilter2D(d_temp, toInt2(olddims));
					else if (ndims == 3)
						d_CubicBSplinePrefilter3D(d_temp, olddims);
				}
				if (ndims == 3)
					d_BindTextureTo3DArray(d_temp, a_image, t_image, olddims, cudaFilterModeLinear, false);
				else
					d_BindTextureToArray(d_temp, a_image, t_image, toInt2(olddims), cudaFilterModeLinear, false);

				dim3 TpB, grid;
				if (ndims > 1)
				{
					TpB = dim3(16, 16);
					grid = dim3((newdims.x + 15) / 16, (newdims.y + 15) / 16, newdims.z);
				}
				else
				{
					TpB = dim3(256);
					grid = dim3((newdims.x + 255) / 256);
				}

				tfloat3 factor = tfloat3((tfloat)olddims.x / (tfloat)newdims.x, (tfloat)olddims.y / (tfloat)newdims.y, (tfloat)olddims.z / (tfloat)newdims.z);
				tfloat3 offset = tfloat3(0.5f * factor.x, 0.5f * factor.y, 0.5f * factor.z);

				if (mode == T_INTERP_CUBIC)
				{
					if (ndims == 1)
						InterpolateKernel<1, true> << <grid, TpB >> > (d_output + Elements(newdims) * b, newdims, t_image, olddims, factor, offset);
					else if (ndims == 2)
						InterpolateKernel<2, true> << <grid, TpB >> > (d_output + Elements(newdims) * b, newdims, t_image, olddims, factor, offset);
					else if (ndims == 3)
						InterpolateKernel<3, true> << <grid, TpB >> > (d_output + Elements(newdims) * b, newdims, t_image, olddims, factor, offset);
				}
				else
				{
					if (ndims == 1)
						InterpolateKernel<1, false> << <grid, TpB >> > (d_output + Elements(newdims) * b, newdims, t_image, olddims, factor, offset);
					else if (ndims == 2)
						InterpolateKernel<2, false> << <grid, TpB >> > (d_output + Elements(newdims) * b, newdims, t_image, olddims, factor, offset);
					else if (ndims == 3)
						InterpolateKernel<3, false> << <grid, TpB >> > (d_output + Elements(newdims) * b, newdims, t_image, olddims, factor, offset);
				}

				cudaDestroyTextureObject(t_image);
				cudaFreeArray(a_image);
			}

			cudaFree(d_temp);
		}
		else if (mode == T_INTERP_FOURIER)
		{
			tcomplex* d_owninputFFT;
			if (d_inputfft == NULL)
				cudaMalloc((void**)& d_owninputFFT, ElementsFFT(olddims) * batch * sizeof(tcomplex));
			else
				d_owninputFFT = d_inputfft;

			tcomplex* d_ownoutputFFT;
			if (d_outputfft == NULL)
				cudaMalloc((void**)& d_ownoutputFFT, ElementsFFT(newdims) * batch * sizeof(tcomplex));
			else
				d_ownoutputFFT = d_outputfft;

			tfloat normfactor = (tfloat)1 / (tfloat)Elements(olddims);

			if (planforw == NULL)
				d_FFTR2C(d_input, d_owninputFFT, ndims, olddims, batch);
			else
				d_FFTR2C(d_input, d_owninputFFT, planforw);

			if (biggerdims > 0)
				d_FFTPad(d_owninputFFT, d_ownoutputFFT, olddims, newdims, batch);
			else
				d_FFTCrop(d_owninputFFT, d_ownoutputFFT, olddims, newdims, batch);

			if (planback == NULL)
				d_IFFTC2R(d_ownoutputFFT, d_output, ndims, newdims, batch, false);
			else
				d_IFFTC2R(d_ownoutputFFT, d_output, planback);

			d_MultiplyByScalar(d_output, d_output, Elements(newdims) * batch, normfactor);

			if (d_inputfft == NULL)
				cudaFree(d_owninputFFT);
			if (d_outputfft == NULL)
				cudaFree(d_ownoutputFFT);
		}
	}


	////////////////
	//CUDA kernels//
	////////////////

	template <int ndims, bool cubicinterp> __global__ void InterpolateKernel(tfloat* d_output, int3 dimsnew, cudaTex t_input, int3 dimsold, tfloat3 factor, tfloat3 offset)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dimsnew.x)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dimsnew.y)
			return;
		int idz = blockIdx.z;

		tfloat3 position = tfloat3(idx - dimsnew.x / 2, idy - dimsnew.y / 2, idz - dimsnew.z / 2);
		position = tfloat3(position.x * factor.x, position.y * factor.y, position.z * factor.z);
		position = tfloat3(position.x + dimsold.x / 2 + offset.x, position.y + dimsold.y / 2 + offset.y, position.z + dimsold.z / 2 + offset.z);

		if (cubicinterp)
		{
			if (ndims == 1)
				d_output[idx] = cubicTex1D(t_input, position.x);
			else if (ndims == 2)
				d_output[idy * dimsnew.x + idx] = cubicTex2D(t_input, position.x, position.y);
			else if (ndims == 3)
				d_output[(idz * dimsnew.y + idy) * dimsnew.x + idx] = cubicTex3D(t_input, position.x, position.y, position.z);
		}
		else
		{
			if (ndims == 1)
				d_output[idx] = tex1D<tfloat>(t_input, position.x);
			else if (ndims == 2)
				d_output[idy * dimsnew.x + idx] = tex2D<tfloat>(t_input, position.x, position.y);
			else if (ndims == 3)
				d_output[(idz * dimsnew.y + idy) * dimsnew.x + idx] = tex3D<tfloat>(t_input, position.x, position.y, position.z);
		}
	}
}