#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"

namespace gtom
{
	void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch, bool renormalize)
	{
		cufftHandle plan = d_IFFTC2RGetPlan(ndimensions, dimensions, batch);
		if (renormalize)
			d_IFFTC2R(d_input, d_output, &plan, dimensions, batch);
		else
			d_IFFTC2R(d_input, d_output, &plan);
		cufftDestroy(plan);
	}

	cufftHandle d_IFFTC2RGetPlan(int const ndimensions, int3 const dimensions, int batch)
	{
		cufftHandle plan;
		cufftType direction = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
		int n[3] = { dimensions.z, dimensions.y, dimensions.x };

		CHECK_CUFFT_ERRORS(cufftPlanMany(&plan, ndimensions, n + (3 - ndimensions),
										 NULL, 1, 0,
										 NULL, 1, 0,
										 direction, batch));

		//cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);

		cufftSetStream(plan, cudaStreamDefault);

		return plan;
	}

	void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, cufftHandle* plan, int3 dimensions, int batch)
	{
#ifdef GTOM_DOUBLE
		cufftExecZ2D(*plan, d_input, d_output);
#else
		CHECK_CUFFT_ERRORS(cufftExecC2R(*plan, d_input, d_output));
#endif
		cudaStreamSynchronize(cudaStreamDefault);

		d_MultiplyByScalar(d_output, d_output, Elements(dimensions) * batch, 1.0f / (float)Elements(dimensions));
	}

	void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, cufftHandle* plan)
	{
#ifdef GTOM_DOUBLE
		cufftExecZ2D(*plan, d_input, d_output);
#else
		cufftExecC2R(*plan, d_input, d_output);
#endif
		cudaStreamSynchronize(cudaStreamDefault);
	}

	void d_IFFTZ2D(cufftDoubleComplex* const d_input, double* const d_output, int const ndimensions, int3 const dimensions, int batch)
	{
		cufftHandle plan;
		cufftType direction = CUFFT_Z2D;
		int n[3] = { dimensions.z, dimensions.y, dimensions.x };

		cufftPlanMany(&plan, ndimensions, n + (3 - ndimensions),
			NULL, 1, 0,
			NULL, 1, 0,
			direction, batch);

		//cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);
#ifdef GTOM_DOUBLE
		cufftExecZ2D(plan, d_input, d_output);
#else
		cufftExecZ2D(plan, d_input, d_output);
#endif

		cufftDestroy(plan);

		size_t elements = dimensions.x * dimensions.y * dimensions.z;
		d_MultiplyByScalar(d_output, d_output, elements, 1.0 / (double)elements);
	}

	void d_IFFTC2RFull(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch)
	{
		tcomplex* d_complexoutput;
		cudaMalloc((void**)&d_complexoutput, Elements(dimensions) * sizeof(tcomplex));

		d_IFFTC2C(d_input, d_complexoutput, ndimensions, dimensions, batch);
		d_Re(d_complexoutput, d_output, Elements(dimensions));

		cudaFree(d_complexoutput);
	}

	void d_IFFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch)
	{
		cufftHandle plan = d_IFFTC2CGetPlan(ndimensions, dimensions, batch);
		d_IFFTC2C(d_input, d_output, &plan, dimensions);
		cufftDestroy(plan);
	}

	cufftHandle d_IFFTC2CGetPlan(int const ndimensions, int3 const dimensions, int batch)
	{
		cufftHandle plan;
		cufftType direction = IS_TFLOAT_DOUBLE ? CUFFT_Z2Z : CUFFT_C2C;
		int n[3] = { dimensions.z, dimensions.y, dimensions.x };

		cufftPlanMany(&plan, ndimensions, n + (3 - ndimensions),
			NULL, 1, 0,
			NULL, 1, 0,
			direction, batch);

		//cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);

		return plan;
	}

	void d_IFFTC2C(tcomplex* const d_input, tcomplex* const d_output, cufftHandle* plan, int3 const dimensions)
	{
#ifdef GTOM_DOUBLE
		cufftExecZ2Z(*plan, d_input, d_output);
#else
		cufftExecC2C(*plan, d_input, d_output, CUFFT_INVERSE);
#endif
		cudaStreamQuery(0);

		size_t elements = dimensions.x * dimensions.y * dimensions.z;
		d_MultiplyByScalar((tfloat*)d_output, (tfloat*)d_output, elements * 2, 1.0f / (float)elements);
	}
}