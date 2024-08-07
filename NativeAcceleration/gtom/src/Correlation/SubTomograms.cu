#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Binary.cuh"
#include "gtom/include/Correlation.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Projection.cuh"
#include "gtom/include/Reconstruction.cuh"
#include "gtom/include/Relion.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
	__global__ void BatchComplexConjMultiplyKernel(tcomplex* d_input1, tcomplex* d_input2, tcomplex* d_output, uint vectorlength, uint batch);
	__global__ void UpdateCorrelationKernel(tfloat* d_correlation, uint vectorlength, uint batch, int batchoffset, tfloat* d_bestcorrelation, float* d_bestangle);

	void d_PickSubTomograms(cudaTex t_projectordataRe,
							cudaTex t_projectordataIm,
							tfloat projectoroversample,
							int3 dimsprojector,
							tcomplex* d_experimentalft,
							tfloat* d_ctf,
							int3 dimsvolume,
							uint nvolumes,
							tfloat3* h_angles,
							uint nangles,
							uint batchangles,
							tfloat maskradius,
							tfloat* d_bestcorrelation,
							float* d_bestangle,
							float* h_progressfraction)
	{
		int ndims = DimensionCount(dimsvolume);
		uint batchsize = batchangles;
		if (ndims == 2)
			batchsize = 240;
		/*if (nvolumes > batchsize)
			throw;*/

		d_ValueFill(d_bestcorrelation, Elements(dimsvolume) * nvolumes, (tfloat)-1e30);
		d_ValueFill(d_bestangle, Elements(dimsvolume) * nvolumes, (float)0);

		tcomplex* d_projectedftctf;
		cudaMalloc((void**)&d_projectedftctf, ElementsFFT(dimsvolume) * tmax(nvolumes, batchsize) * sizeof(tcomplex));
		tcomplex* d_projectedftctfcorr;
		cudaMalloc((void**)&d_projectedftctfcorr, ElementsFFT(dimsvolume) * tmax(nvolumes, batchsize) * sizeof(tcomplex));
		tfloat* d_projected;
		cudaMalloc((void**)&d_projected, Elements(dimsvolume) * tmax(nvolumes, batchsize) * sizeof(tfloat));

		cufftHandle planback = d_IFFTC2RGetPlan(ndims, dimsvolume, batchsize);

		bool debug = false;

		for (uint b = 0; b < nangles; b += batchsize)
		{
			uint curbatch = tmin(batchsize, nangles - b);

			// d_projectedftctf will contain rotated reference volume multiplied by CTF
			d_rlnProjectCTFMult(t_projectordataRe, t_projectordataIm, d_ctf, dimsprojector, d_projectedftctf, dimsvolume, h_angles + b, projectoroversample, curbatch);

			for (uint v = 0; v < nvolumes; v++)
			{
				// Multiply current experimental volume by conjugate references
				{
					int TpB = 128;
					dim3 grid = dim3(tmin((ElementsFFT(dimsvolume) + TpB - 1) / TpB, 2048), 1, 1);
					BatchComplexConjMultiplyKernel << <grid, TpB >> > (d_experimentalft + ElementsFFT(dimsvolume) * v, d_projectedftctf, d_projectedftctfcorr, ElementsFFT(dimsvolume), curbatch);
				}

				d_IFFTC2R(d_projectedftctfcorr, d_projected, &planback);

				if (debug && b == 0 && v == 0)
					d_WriteMRC(d_projected, toInt3(dimsvolume.x, dimsvolume.y, dimsvolume.z * curbatch), "d_projected_corrected.mrc");

				// Update correlation and angles with best values
				{
					int TpB = 128;
					dim3 grid = dim3((Elements(dimsvolume) + TpB - 1) / TpB, 1, 1);
					UpdateCorrelationKernel << <grid, TpB >> > (d_projected,
						Elements(dimsvolume),
						curbatch,
						b,
						d_bestcorrelation + Elements(dimsvolume) * v,
						d_bestangle + Elements(dimsvolume) * v);
				}

				//d_WriteMRC(d_bestcorrelation + Elements(dimsvolume) * v, dimsvolume, "d_correlation_best.mrc");
			}

			if (h_progressfraction)
				*h_progressfraction = (float)(b + curbatch) / nangles;
		}


		cufftDestroy(planback);

		cudaFree(d_projected);
		cudaFree(d_projectedftctfcorr);
		cudaFree(d_projectedftctf);
	}

	__global__ void BatchComplexConjMultiplyKernel(tcomplex* d_input1, tcomplex* d_input2, tcomplex* d_output, uint vectorlength, uint batch)
	{
		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
		{
			tcomplex input1 = d_input1[id];

			for (uint b = 0; b < batch; b++)
				d_output[b * vectorlength + id] = cmul(input1, cconj(d_input2[b * vectorlength + id]));
		}
	}

	__global__ void UpdateCorrelationKernel(tfloat* d_correlation, uint vectorlength, uint batch, int batchoffset, tfloat* d_bestcorrelation, float* d_bestangle)
	{
		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
		{
			tfloat bestcorrelation = d_bestcorrelation[id];
			float bestangle = d_bestangle[id];

			for (uint b = 0; b < batch; b++)
			{
				tfloat newcorrelation = d_correlation[b * vectorlength + id];
				if (newcorrelation > bestcorrelation)
				{
					bestcorrelation = newcorrelation;
					bestangle = b + batchoffset;
				}
			}

			d_bestcorrelation[id] = bestcorrelation;
			d_bestangle[id] = bestangle;
		}
	}
}