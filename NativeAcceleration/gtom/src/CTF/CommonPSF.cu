#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	template<bool same2> __global__ void CommonPSFKernel(tcomplex* d_inft1, tcomplex* d_inft2, tcomplex* d_outft1, tcomplex* d_outft2, tfloat* d_psf1, tfloat* d_psf2, tfloat* d_commonpsf, size_t n);


	/////////////////////////////////////////////
	//Equalize two PSFs to their common minimum//
	/////////////////////////////////////////////

	void d_ForceCommonPSF(tcomplex* d_inft1, tcomplex* d_inft2, tcomplex* d_outft1, tcomplex* d_outft2, tfloat* d_psf1, tfloat* d_psf2, tfloat* d_commonpsf, uint n, bool same2, int batch)
	{
		uint TpB = min(192, NextMultipleOf(n, 32));
		dim3 grid = dim3((n + TpB - 1) / TpB, batch);
		if (same2)
			CommonPSFKernel<true> << <grid, TpB >> > (d_inft1, d_inft2, d_outft1, d_outft2, d_psf1, d_psf2, d_commonpsf, n);
		else
			CommonPSFKernel<false> << <grid, TpB >> > (d_inft1, d_inft2, d_outft1, d_outft2, d_psf1, d_psf2, d_commonpsf, n);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<bool same2> __global__ void CommonPSFKernel(tcomplex* d_inft1, tcomplex* d_inft2, tcomplex* d_outft1, tcomplex* d_outft2, tfloat* d_psf1, tfloat* d_psf2, tfloat* d_commonpsf, size_t n)
	{
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= n)
			return;

		size_t offset = n * blockIdx.y;
		d_psf1 += offset;
		if (!same2)
			d_psf2 += offset;
		d_commonpsf += offset;
		if (!same2)
			d_inft2 += offset;
		d_outft2 += offset;
		d_inft1 += offset;
		d_outft1 += offset;

		tfloat psf1 = abs(d_psf1[idx]);
		tfloat psf2 = abs(d_psf2[idx]);
		tfloat minpsf = min(psf1, psf2);
		tfloat conv1 = 0, conv2 = 0;
		if (psf1 > 0)
			conv1 = minpsf / psf1;
		if (psf2 > 0)
			conv2 = minpsf / psf2;

		tcomplex ft1 = d_inft1[idx];
		d_outft1[idx] = make_cuComplex(ft1.x * conv1, ft1.y * conv1);
		tcomplex ft2 = d_inft2[idx];
		d_outft2[idx] = make_cuComplex(ft2.x * conv2, ft2.y * conv2);
		d_commonpsf[idx] = minpsf;
	}
}