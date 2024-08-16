#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Masking.cuh"
#include "gtom/include/Reconstruction.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void PrecomputeBlobKernel(tfloat* d_precompblob, int paddim, int oripadded, tfloat* d_funcvals, tfloat funcsampling, int funcelements);
	template<class T> __global__ void DecenterKernel(T* d_input, T* d_output, int3 newdims, int3 olddims);
	__global__ void Iteration1Kernel(tcomplex* d_convft, tfloat* d_Fweight, tfloat* d_Fnewweight, uint elements);
	__global__ void Iteration2Kernel(tcomplex* d_convft, tfloat* d_Fnewweight, uint elements);
	__global__ void CorrectGriddingKernel(tfloat* d_volume, int dim, int oripadded);


	void d_ReconstructGridding(tcomplex* d_dataft, tfloat* d_weight, tfloat* d_reconstructed, int3 dimsori, int3 dimspadded, int paddingfactor, cufftHandle pre_planforw, cufftHandle pre_planback, int iterations, double blobradius, int bloborder, double blobalpha)
	{
		int3 dimsoripad = dimsori * paddingfactor;

		// Precalc blob values
		tfloat* d_precompblob;
		if (iterations > 0)
		{
			double radius = blobradius * paddingfactor;
			double alpha = blobalpha;
			int order = bloborder;
			int elements = 10000;
			double sampling = 0.5 / elements;
			tfloat* h_blobvalues = (tfloat*)malloc(elements * sizeof(tfloat));
			for (int i = 0; i < elements; i++)
				h_blobvalues[i] = kaiser_Fourier_value(i * sampling, radius, alpha, order);
			float blobnorm = h_blobvalues[0];
			for (int i = 0; i < elements; i++)
				h_blobvalues[i] /= blobnorm;
			tfloat* d_blobvalues = (tfloat*)CudaMallocFromHostArray(h_blobvalues, elements * sizeof(tfloat));

			cudaMalloc((void**)&d_precompblob, Elements(dimsoripad) * sizeof(tfloat));
			int TpB = tmin(128, NextMultipleOf(dimsoripad.x, 32));
			dim3 grid = dim3(dimsoripad.y, dimsoripad.z);
			PrecomputeBlobKernel << <grid, TpB >> > (d_precompblob, dimsoripad.x, dimsoripad.x, d_blobvalues, (tfloat)sampling, elements);

			//d_WriteMRC(d_blobvalues, toInt3(elements, 1, 1), "d_blobvalues.mrc");
			//d_WriteMRC(d_precompblob, dimsoripad, "d_precompblob.mrc");

			cudaFree(d_blobvalues);
			free(h_blobvalues);
		}

		int r_max = dimsori.x / 2;
		int max_r2 = r_max * r_max * paddingfactor * paddingfactor;

		tcomplex* d_convft;
		cudaMalloc((void**)&d_convft, ElementsFFT(dimsoripad) * sizeof(tcomplex));
		tfloat* d_conv;
		cudaMalloc((void**)&d_conv, Elements(dimsoripad) * sizeof(tfloat));

		tfloat* d_Fweight;
		cudaMalloc((void**)&d_Fweight, ElementsFFT(dimsoripad) * sizeof(tfloat));
		{
			int TpB = tmin(128, NextMultipleOf(dimsoripad.x, 32));
			dim3 grid = dim3(dimsoripad.y, dimsoripad.z);
			DecenterKernel<tfloat> <<<grid, TpB>>> (d_weight, d_Fweight, dimsori * paddingfactor, dimspadded);
		}
		//d_WriteMRC(d_Fweight, toInt3FFT(dimsoripad), "d_Fweight.mrc");

		// Fnewweight is initialized to 1 within r_max
		tfloat* d_Fnewweight = CudaMallocValueFilled(ElementsFFT(dimsoripad), (tfloat)1);
		d_SphereMaskFT(d_Fnewweight, d_Fnewweight, dimsoripad, r_max * paddingfactor);
		//d_WriteMRC(d_Fnewweight, toInt3FFT(dimsoripad), "d_Fnewweight.mrc");

		cufftHandle planforw = pre_planforw, planback = pre_planback;
		if (pre_planforw <= NULL)
			planforw = d_FFTR2CGetPlan(3, dimsoripad);
		if (pre_planback <= NULL)
			planback = d_IFFTC2RGetPlan(3, dimsoripad);

		for (int i = 0; i < iterations; i++)
		{
			int TpB = 128;
			dim3 grid = dim3((ElementsFFT(dimsoripad) + TpB - 1) / TpB, 1, 1);
			Iteration1Kernel << <grid, TpB >> > (d_convft, d_Fweight, d_Fnewweight, ElementsFFT(dimsoripad));

			// Convolute with blob in real space
			d_IFFTC2R(d_convft, d_conv, &planback, dimsoripad);
			d_MultiplyByVector(d_conv, d_precompblob, d_conv, Elements(dimsoripad));
			d_FFTR2C(d_conv, d_convft, &planforw);

			Iteration2Kernel << <grid, TpB >> > (d_convft, d_Fnewweight, ElementsFFT(dimsoripad));

		}
		//d_MinOp(d_Fnewweight, 1e20f, d_Fnewweight, ElementsFFT(dimsoripad));
		//d_WriteMRC(d_Fnewweight, toInt3FFT(dimsoripad), "d_Fnewweight.mrc");

		{
			int TpB = tmin(128, NextMultipleOf(dimsoripad.x, 32));
			dim3 grid = dim3(dimsoripad.y, dimsoripad.z);
			DecenterKernel<tcomplex> << <grid, TpB >> > (d_dataft, d_convft, dimsori * paddingfactor, dimspadded);
		}

		if (iterations == 0)
		{
			//tcomplex* h_convft = (tcomplex*)MallocFromDeviceArray(d_convft, ElementsFFT(dimsoripad) * sizeof(tcomplex));
			//tfloat* h_Fweight = (tfloat*)MallocFromDeviceArray(d_Fweight, ElementsFFT(dimsoripad) * sizeof(tfloat));

			//for (size_t i = 0; i < ElementsFFT(dimsoripad); i++)
			//{
			//	if (abs(h_Fweight[i]) > 1e-4f)
			//	{
			//		h_convft[i] *= 1 / h_Fweight[i];
			//		//h_Fweight[i] *= 1 / h_Fweight[i];
			//	}
			//}

			//cudaMemcpy(d_convft, h_convft, ElementsFFT(dimsoripad) * sizeof(tcomplex), cudaMemcpyHostToDevice);
			////cudaMemcpy(d_Fweight, h_Fweight, ElementsFFT(dimsoripad) * sizeof(tfloat), cudaMemcpyHostToDevice);
			//free(h_convft);
			//free(h_Fweight);

			d_Abs(d_Fweight, d_Fweight, ElementsFFT(dimsoripad));
			d_MaxOp(d_Fweight, 1e-4f, d_Fweight, ElementsFFT(dimsoripad));
			d_ComplexDivideByVector(d_convft, d_Fweight, d_convft, ElementsFFT(dimsoripad), 1);
		}
		else
		{
			d_ComplexMultiplyByVector(d_convft, d_Fnewweight, d_convft, ElementsFFT(dimsoripad));
		}

		cudaFree(d_Fweight);
		cudaFree(d_Fnewweight);
		if (iterations > 0)
			cudaFree(d_precompblob);

		tfloat3 decenter_shift[] = { tfloat3(dimsoripad.x / 2) };
		d_Shift(d_convft, d_convft, dimsoripad, decenter_shift);

		d_IFFTC2R(d_convft, d_conv, &planback);
		//d_WriteMRC(d_conv, dimsori * paddingfactor, "d_reconstructed.mrc");
		d_Pad(d_conv, d_reconstructed, dimsoripad, dimsori, T_PAD_MODE::T_PAD_VALUE, (tfloat)0);
		//d_RemapFullFFT2Full(d_reconstructed, d_reconstructed, dimsori);

		if (pre_planforw <= NULL)
			cufftDestroy(planforw);
		if (pre_planback <= NULL)
			cufftDestroy(planback);

		tfloat rf = r_max - 1;
		//d_SphereMask(d_reconstructed, d_reconstructed, dimsori, &rf, 3.0f, NULL);

		{
			int TpB = tmin(128, NextMultipleOf(dimsori.x, 32));
			dim3 grid = dim3(dimsori.y, dimsori.z);
			CorrectGriddingKernel <<<grid, TpB>>> (d_reconstructed, dimsori.x, dimsori.x * paddingfactor);
		}
		//d_DivideByScalar(d_reconstructed, d_reconstructed, Elements(dimsori), (tfloat)paddingfactor * paddingfactor * paddingfactor);
		//d_WriteMRC(d_reconstructed, dimsori, "d_reconstructed.mrc");

		d_MultiplyByScalar(d_reconstructed, d_reconstructed, Elements(dimsori), 1.0f / (paddingfactor * paddingfactor * paddingfactor * dimsori.x));

		cudaFree(d_conv);
		cudaFree(d_convft);
	}

	__global__ void PrecomputeBlobKernel(tfloat* d_precompblob, int paddim, int oripadded, tfloat* d_funcvals, tfloat funcsampling, int funcelements)
	{
		int z = blockIdx.y;
		int y = blockIdx.x;

		d_precompblob += (z * paddim + y) * paddim;

		int zp = z < paddim / 2 ? z : z - paddim;
		zp *= zp;
		int yp = y < paddim / 2 ? y : y - paddim;
		yp *= yp;

		for (int x = threadIdx.x; x < paddim; x += blockDim.x)
		{
			float xp = x < paddim / 2 ? x : x - paddim;
			float r = sqrt(xp * xp + yp + zp) / oripadded / funcsampling;

			d_precompblob[x] = d_funcvals[tmin(funcelements - 1, (int)r)];
		}
	}

	template<class T> __global__ void DecenterKernel(T* d_input, T* d_output, int3 newdims, int3 olddims)
	{
		int z = blockIdx.y;
		int y = blockIdx.x;
		
		float r = 0;
		int zp = z < newdims.z / 2 + 1 ? z : z - newdims.x;
		r += zp * zp;
		zp += olddims.z / 2;
		int yp = y < newdims.y / 2 + 1 ? y : y - newdims.x;
		r += yp * yp;
		yp += olddims.y / 2;

		for (int x = threadIdx.x; x < newdims.x / 2 + 1; x += blockDim.x)
		{
			int xp = x;
			float rr = r + xp * xp;
			float mask = rr < newdims.x * newdims.x / 4 ? 1 : 0;

			d_output[(z * newdims.y + y) * (newdims.x / 2 + 1) + x] = d_input[(zp * olddims.y + yp) * (olddims.x / 2 + 1) + xp] * mask;
		}
	}
	
	__global__ void Iteration1Kernel(tcomplex* d_convft, tfloat* d_Fweight, tfloat* d_Fnewweight, uint elements)
	{
		for (uint i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			d_convft[i] = make_cuComplex(d_Fweight[i] * d_Fnewweight[i], 0);
	}

	__global__ void Iteration2Kernel(tcomplex* d_convft, tfloat* d_Fnewweight, uint elements)
	{
		for (uint i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			//if (d_convft[i].x >= 0)
			{
				tfloat w = tmax(1e-4f, d_convft[i].x);
				d_Fnewweight[i] = tmin(1e20f, d_Fnewweight[i] / w);
			}
			/*else
			{
				tfloat w = tmin(-1e-6f, d_convft[i].x);
				d_Fnewweight[i] = tmin(1e20f, d_Fnewweight[i] / w);
			}*/
		}
	}

	__global__ void CorrectGriddingKernel(tfloat* d_volume, int dim, int oripadded)
	{
		int z = blockIdx.y;
		int y = blockIdx.x;

		d_volume += (z * dim + y) * dim;
		
		y -= dim / 2;
		y *= y;
		z -= dim / 2;
		z *= z;

		for (int x = threadIdx.x; x < dim; x += blockDim.x)
		{
			float xx = x - dim / 2;

			float r = sqrt(xx * xx + y + z);
			r /= oripadded;

			if (r > 0)
				d_volume[x] /= sinc(r) * sinc(r);
		}
	}
}