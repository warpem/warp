#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Masking.cuh"


namespace gtom
{
	__global__ void BandpassNonCubicKernel(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, tfloat nyquistsoftedge);
	__global__ void BandpassNonCubicGaussKernel(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, tfloat twosigma2);
	__global__ void BandpassNonCubicButterKernel(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, int order);

	///////////////////////////////////////////
	//Equivalent of TOM's tom_bandpass method//
	///////////////////////////////////////////

	void d_Bandpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, tfloat* d_mask, cufftHandle* planforw, cufftHandle* planback, int batch)
	{
		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * batch * sizeof(tcomplex));

		if (planforw == NULL)
			d_FFTR2C(d_input, d_inputft, DimensionCount(dims), dims, batch);
		else
			d_FFTR2C(d_input, d_inputft, planforw);

		d_Bandpass(d_inputft, d_inputft, dims, low, high, smooth, d_mask, batch);

		if (planback == NULL)
			d_IFFTC2R(d_inputft, d_output, DimensionCount(dims), dims, batch);
		else
			d_IFFTC2R(d_inputft, d_output, planback);

		cudaFree(d_inputft);
	}

	void d_Bandpass(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat low, tfloat high, tfloat smooth, tfloat* d_mask, int batch)
	{
		//Prepare mask:

		tfloat* d_localmask;

		if (d_mask == NULL)
		{
			tfloat* d_maskhigh = (tfloat*)CudaMallocValueFilled(Elements(dims), (tfloat)1);

			d_SphereMask(d_maskhigh, d_maskhigh, dims, &high, smooth, (tfloat3*)NULL, 1);

			tfloat* d_maskhighFFT;
			cudaMalloc((void**)&d_maskhighFFT, ElementsFFT(dims) * sizeof(tfloat));
			d_RemapFull2HalfFFT(d_maskhigh, d_maskhighFFT, dims);

			d_localmask = d_maskhighFFT;

			tfloat* d_masklowFFT;
			if (low > 0)
			{
				tfloat* d_masklow = (tfloat*)CudaMallocValueFilled(Elements(dims), (tfloat)1);
				d_SphereMask(d_masklow, d_masklow, dims, &low, smooth, (tfloat3*)NULL, 1);
				cudaMalloc((void**)&d_masklowFFT, ElementsFFT(dims) * sizeof(tfloat));
				d_RemapFull2HalfFFT(d_masklow, d_masklowFFT, dims);
				d_SubtractVector(d_localmask, d_masklowFFT, d_localmask, ElementsFFT(dims), 1);

				cudaFree(d_masklow);
				cudaFree(d_masklowFFT);
			}

			cudaFree(d_maskhigh);
		}
		else
			d_localmask = d_mask;

		//Mask FFT:

		d_ComplexMultiplyByVector(d_inputft, d_localmask, d_outputft, ElementsFFT(dims), batch);

		if (d_mask == NULL)
			cudaFree(d_localmask);
	}

	void d_BandpassNonCubic(tfloat* d_input, tfloat* d_output, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, tfloat nyquistsoftedge, uint batch)
	{
		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * batch * sizeof(tcomplex));
		d_FFTR2C(d_input, d_inputft, DimensionCount(dims), dims, batch);

		d_FourierBandpassNonCubic(d_inputft, dims, nyquistlow, nyquisthigh, nyquistsoftedge, batch);

		d_IFFTC2R(d_inputft, d_output, DimensionCount(dims), dims, batch);
		cudaFree(d_inputft);
	}

	void d_FourierBandpassNonCubic(tcomplex* d_inputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, tfloat nyquistsoftedge, uint batch)
	{
		dim3 grid = dim3(dims.y, dims.z, batch);
		BandpassNonCubicKernel << <grid, 128 >> > (d_inputft, d_inputft, dims, nyquistlow, nyquisthigh, nyquistsoftedge);
	}

	__global__ void BandpassNonCubicKernel(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, tfloat nyquistsoftedge)
	{
		int y = blockIdx.x;
		int z = blockIdx.y;

		d_inputft += ElementsFFT(dims) * blockIdx.z + (z * dims.y + y) * (dims.x / 2 + 1);
		d_outputft += ElementsFFT(dims) * blockIdx.z + (z * dims.y + y) * (dims.x / 2 + 1);

		float yy = y >= dims.y / 2 + 1 ? y - dims.y : y;
		yy /= dims.y / 2.0f;
		yy *= yy;
		float zz = z >= dims.z / 2 + 1 ? z - dims.z : z;
		zz /= dims.z / 2.0f;
		zz *= zz;

		for (int x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			float xx = x;
			xx /= dims.x / 2.0f;
			xx *= xx;

			float r = sqrt(xx + yy + zz);

            float filter = 1;
            if (nyquistsoftedge > 0)
            {
                float edgelow = cos(tmin(1, tmax(0, nyquistlow - r) / nyquistsoftedge) * PI) * 0.5f + 0.5f;
                float edgehigh = cos(tmin(1, tmax(0, (r - nyquisthigh) / nyquistsoftedge)) * PI) * 0.5f + 0.5f;
                filter = edgelow * edgehigh;
            }
            else
            {
                filter = (r >= nyquistlow && r <= nyquisthigh) ? 1 : 0;
            }

			d_outputft[x] = d_inputft[x] * filter;
		}
	}

	void d_BandpassNonCubicGauss(tfloat* d_input, tfloat* d_output, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, tfloat sigma, uint batch)
	{
		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * batch * sizeof(tcomplex));
		d_FFTR2C(d_input, d_inputft, DimensionCount(dims), dims, batch);

		d_FourierBandpassNonCubicGauss(d_inputft, dims, nyquistlow, nyquisthigh, sigma, batch);

		d_IFFTC2R(d_inputft, d_output, DimensionCount(dims), dims, batch);
		cudaFree(d_inputft);
	}

	void d_FourierBandpassNonCubicGauss(tcomplex* d_inputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, tfloat sigma, uint batch)
	{
		dim3 grid = dim3(dims.y, dims.z, batch);
		BandpassNonCubicGaussKernel << <grid, 128 >> > (d_inputft, d_inputft, dims, nyquistlow, nyquisthigh, 2 * sigma * sigma);
	}

	__global__ void BandpassNonCubicGaussKernel(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, tfloat twosigma2)
	{
		int y = blockIdx.x;
		int z = blockIdx.y;

		d_inputft += ElementsFFT(dims) * blockIdx.z + (z * dims.y + y) * (dims.x / 2 + 1);
		d_outputft += ElementsFFT(dims) * blockIdx.z + (z * dims.y + y) * (dims.x / 2 + 1);

		float yy = y >= dims.y / 2 + 1 ? y - dims.y : y;
		yy /= dims.y / 2.0f;
		yy *= yy;
		float zz = z >= dims.z / 2 + 1 ? z - dims.z : z;
		zz /= dims.z / 2.0f;
		zz *= zz;

		for (int x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			float xx = x;
			xx /= dims.x / 2.0f;
			xx *= xx;

			float r = sqrt(xx + yy + zz);

			float filter = 1;
			if (twosigma2 > 0)
			{
				float xlow = tmax(0, nyquistlow - r);
				float xhigh = tmax(0, r - nyquisthigh);
				float edgelow = xlow > 0 ? expf(-(xlow * xlow / twosigma2)) : 1;
				float edgehigh = xhigh > 0 ? expf(-(xhigh * xhigh / twosigma2)) : 1;
				filter = edgelow * edgehigh;
			}
			else
			{
				filter = (r >= nyquistlow && r <= nyquisthigh) ? 1 : 0;
			}

			d_outputft[x] = d_inputft[x] * filter;
		}
	}

	void d_BandpassNonCubicButter(tfloat* d_input, tfloat* d_output, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, int order, uint batch)
	{
		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * batch * sizeof(tcomplex));
		d_FFTR2C(d_input, d_inputft, DimensionCount(dims), dims, batch);

		d_FourierBandpassNonCubicButter(d_inputft, dims, nyquistlow, nyquisthigh, order, batch);

		d_IFFTC2R(d_inputft, d_output, DimensionCount(dims), dims, batch);
		cudaFree(d_inputft);
	}

	void d_FourierBandpassNonCubicButter(tcomplex* d_inputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, int order, uint batch)
	{
		dim3 grid = dim3(dims.y, dims.z, batch);
		BandpassNonCubicButterKernel << <grid, 128 >> > (d_inputft, d_inputft, dims, nyquistlow, nyquisthigh, order);
	}

	__global__ void BandpassNonCubicButterKernel(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, int order)
	{
		int y = blockIdx.x;
		int z = blockIdx.y;

		d_inputft += ElementsFFT(dims) * blockIdx.z + (z * dims.y + y) * (dims.x / 2 + 1);
		d_outputft += ElementsFFT(dims) * blockIdx.z + (z * dims.y + y) * (dims.x / 2 + 1);

		float yy = y >= dims.y / 2 + 1 ? y - dims.y : y;
		yy /= dims.y / 2.0f;
		yy *= yy;
		float zz = z >= dims.z / 2 + 1 ? z - dims.z : z;
		zz /= dims.z / 2.0f;
		zz *= zz;

		for (int x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			float xx = x;
			xx /= dims.x / 2.0f;
			xx *= xx;

			float r = sqrt(xx + yy + zz);

			float filter = 1;
			if (nyquistlow > 0 && nyquisthigh < 1)
			{
				float w = nyquisthigh - nyquistlow;
				float d0 = (nyquistlow + nyquisthigh) * 0.5f;

				if (r != d0)
				{
					float base = r * w / (r * r - d0 * d0);
					float result = 1;
					for (int i = 0; i < 2 * order; i++)
						result *= base;

					filter = 1 - 1 / (1 + result);
				}
				else
					filter = 1;
			}
			else if (nyquistlow > 0)
			{
				if (r > 0)
					filter = 1 / (1 + powf(nyquistlow / r, 2 * order));
				else
					filter = 0;
			}
			else if (nyquisthigh <= 1)
			{
				if (nyquisthigh != 0)
					filter = 1 / (1 + powf(r / nyquisthigh, 2 * order));
				else
					filter = 0;
			}
			else
			{
				filter = r > 1 ? 0 : 1;
			}

			d_outputft[x] = d_inputft[x] * filter;
		}
	}
}