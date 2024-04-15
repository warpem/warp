#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Correlation.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Masking.cuh"


namespace gtom
{
	__global__ void RotationSeriesKernel(cudaTex t_input, tfloat* d_output, uint sidelength, float anglestep);


	void d_SimilarityMatrixRow(tfloat* d_images, tcomplex* d_imagesft, int2 dimsimage, int nimages, int anglesteps, int target, tfloat* d_similarity)
	{
		int startvalid = target + 1;
		int nvalid = nimages - startvalid;

		tfloat* d_rotations;
		cudaMalloc((void**)&d_rotations, Elements2(dimsimage) * nvalid * sizeof(tfloat));
		tcomplex* d_rotationsft;
		cudaMalloc((void**)&d_rotationsft, ElementsFFT2(dimsimage) * anglesteps * sizeof(tcomplex));
		tcomplex* d_corr;
		cudaMalloc((void**)&d_corr, ElementsFFT2(dimsimage) * nvalid * sizeof(tcomplex));

		d_RotationSeries(d_images + Elements2(dimsimage) * target, d_rotations, dimsimage, anglesteps);
		d_HannMask(d_rotations, d_rotations, toInt3(dimsimage), NULL, NULL, anglesteps);
		d_NormMonolithic(d_rotations, d_rotations, Elements2(dimsimage), T_NORM_MEAN01STD, anglesteps);

		CudaWriteToBinaryFile("d_rotations.bin", d_rotations, Elements2(dimsimage) * anglesteps * sizeof(tfloat));

		d_FFTR2C(d_rotations, d_rotationsft, 2, toInt3(dimsimage), anglesteps);
		cufftHandle planback;
		planback = d_IFFTC2RGetPlan(2, toInt3(dimsimage), nvalid);

		d_imagesft += ElementsFFT2(dimsimage) * startvalid;
		d_similarity += startvalid;
		d_ValueFill(d_similarity, nvalid, (tfloat)-1);

		for (int i = 0; i < anglesteps; i++)
		{
			d_ComplexMultiplyByConjVector(d_imagesft, d_rotationsft + ElementsFFT2(dimsimage) * i, d_corr, ElementsFFT2(dimsimage), nvalid);
			d_IFFTC2R(d_corr, d_rotations, &planback);
			d_MaxMonolithic(d_rotations, (tfloat*)d_corr, Elements2(dimsimage), nvalid);
			d_MaxOp((tfloat*)d_corr, d_similarity, d_similarity, nvalid);
		}

		cufftDestroy(planback);
		cudaFree(d_corr);
		cudaFree(d_rotationsft);
		cudaFree(d_rotations);
	}

	void d_LineSimilarityMatrixRow(tcomplex* d_linesft, int2 dimsimage, int nimages, int linewidth, int anglesteps, int target, tfloat* d_similarity)
	{
		// Computing only the non-redundant half of the matrix
		int startvalid = target + 1;
		int nvalid = nimages - startvalid;

		int2 dimsline = toInt2(dimsimage.x, linewidth);
		int2 dimslines = toInt2(dimsline.x, dimsline.y * anglesteps);

		tcomplex* d_corrft;
		cudaMalloc((void**)&d_corrft, ElementsFFT2(dimslines) * nvalid * sizeof(tcomplex));
		tfloat* d_corr;
		cudaMalloc((void**)&d_corr, Elements2(dimslines) * nvalid * sizeof(tfloat));

		cufftHandle planback = d_IFFTC2RGetPlan(DimensionCount(toInt3(dimsline)), toInt3(dimsline), anglesteps * nvalid);

		tcomplex* d_target = d_linesft + ElementsFFT2(dimslines) * target;
		d_linesft += ElementsFFT2(dimslines) * startvalid;
		d_similarity += startvalid;
		d_ValueFill(d_similarity, nvalid, (tfloat)-1);

		for (int i = 0; i < anglesteps; i++)
		{
			d_ComplexMultiplyByConjVector(d_linesft, d_target + ElementsFFT2(dimsline) * i, d_corrft, ElementsFFT2(dimsline), anglesteps * nvalid);
			d_IFFTC2R(d_corrft, d_corr, &planback);
			d_MaxMonolithic(d_corr, (tfloat*)d_corrft, Elements2(dimslines), nvalid);
			d_MaxOp((tfloat*)d_corrft, d_similarity, d_similarity, nvalid);

			// d_FFTLines outputs only the non-redundant part, but comparing 
			// with conjugated (= mirrored) version is just as valid.

			d_ComplexMultiplyByVector(d_linesft, d_target + ElementsFFT2(dimsline) * i, d_corrft, ElementsFFT2(dimsline), anglesteps * nvalid);
			d_IFFTC2R(d_corrft, d_corr, &planback);
			d_MaxMonolithic(d_corr, (tfloat*)d_corrft, Elements2(dimslines), nvalid);
			d_MaxOp((tfloat*)d_corrft, d_similarity, d_similarity, nvalid);
		}

		cufftDestroy(planback);
		cudaFree(d_corr);
		cudaFree(d_corrft);
	}

	void d_RotationSeries(tfloat* d_image, tfloat* d_series, int2 dimsimage, int anglesteps)
	{
		dim3 TpB = min(192, NextMultipleOf(Elements2(dimsimage), 32));
		dim3 grid = anglesteps;
		float anglestep = PI2 / (float)anglesteps;

		tfloat* d_prefiltered;
		cudaMalloc((void**)&d_prefiltered, Elements2(dimsimage) * sizeof(tfloat));
		cudaMemcpy(d_prefiltered, d_image, Elements2(dimsimage) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		d_CubicBSplinePrefilter2D(d_prefiltered, dimsimage);

		cudaArray* a_prefiltered;
		cudaTex t_prefiltered;
		d_BindTextureToArray(d_prefiltered, a_prefiltered, t_prefiltered, dimsimage, cudaFilterModeLinear, false);

		RotationSeriesKernel << <grid, TpB >> > (t_prefiltered, d_series, dimsimage.x, anglestep);

		cudaDestroyTextureObject(t_prefiltered);
		cudaFreeArray(a_prefiltered);
		cudaFree(d_prefiltered);
	}


	__global__ void RotationSeriesKernel(cudaTex t_input, tfloat* d_output, uint sidelength, float anglestep)
	{
		uint elements = sidelength * sidelength;
		d_output += elements * blockIdx.x;

		float angle = -(float)blockIdx.x * anglestep;
		float cosangle = cos(angle);
		float sinangle = sin(angle);
		float center = sidelength / 2;

		for (uint id = threadIdx.x; id < elements; id += blockDim.x)
		{
			uint idx = id % sidelength;
			uint idy = id / sidelength;

			glm::vec2 pos = glm::vec2((float)idx - center, (float)idy - center);
			pos = glm::vec2(pos.x * cosangle - pos.y * sinangle, pos.x * sinangle + pos.y * cosangle);
			pos += center + 0.5f;

			tfloat val = cubicTex2D(t_input, pos.x, pos.y);
			d_output[id] = val;
		}
	}
}