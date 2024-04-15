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
							tfloat maskradius,
							tfloat* d_bestcorrelation,
							float* d_bestangle,
							float* h_progressfraction)
	{
		int ndims = DimensionCount(dimsvolume);
		uint batchsize = 128;
		if (ndims == 2)
			batchsize = 240;
		/*if (nvolumes > batchsize)
			throw;*/


		d_ValueFill(d_bestcorrelation, Elements(dimsvolume) * nvolumes, (tfloat)-1e30);
		d_ValueFill(d_bestangle, Elements(dimsvolume) * nvolumes, (float)0);

		//tfloat3* d_angles = (tfloat3*)CudaMallocFromHostArray(h_angles, nangles * sizeof(tfloat3));

		//tcomplex* d_projectedft;
		//cudaMalloc((void**)&d_projectedft, ElementsFFT(dimsvolume) * tmax(nvolumes, batchsize) * sizeof(tcomplex));
		tcomplex* d_projectedftctf;
		cudaMalloc((void**)&d_projectedftctf, ElementsFFT(dimsvolume) * tmax(nvolumes, batchsize) * sizeof(tcomplex));
		tcomplex* d_projectedftctfcorr;
		cudaMalloc((void**)&d_projectedftctfcorr, ElementsFFT(dimsvolume) * tmax(nvolumes, batchsize) * sizeof(tcomplex));
		tfloat* d_projected;
		cudaMalloc((void**)&d_projected, Elements(dimsvolume) * tmax(nvolumes, batchsize) * sizeof(tfloat));

		//tfloat* d_corrmag;
		//cudaMalloc((void**)&d_corrmag, Elements(dimsvolume) * tmax(nvolumes, batchsize) * sizeof(tfloat));

		//cufftHandle planforw = d_FFTR2CGetPlan(ndims, dimsvolume, batchsize);
		cufftHandle planback = d_IFFTC2RGetPlan(ndims, dimsvolume, batchsize);
		
		for (uint b = 0; b < nangles; b += batchsize)
		{
			uint curbatch = tmin(batchsize, nangles - b);

			d_rlnProjectCTFMult(t_projectordataRe, t_projectordataIm, d_ctf, dimsprojector, d_projectedftctf, dimsvolume, h_angles + b, projectoroversample, curbatch);

			// Multiply by experimental CTF, norm in realspace, go back into Fourier space for convolution
			//d_ComplexMultiplyByVector(d_projectedft, d_ctf, d_projectedftctf, ElementsFFT(dimsvolume), curbatch);
			//d_IFFTC2R(d_projectedftctf, d_projected, &planback);
			//d_NormMonolithic(d_projected, d_projected, Elements(dimsvolume), T_NORM_MEAN01STD, curbatch);
			//d_WriteMRC(d_projected, toInt3(dimsvolume.x, dimsvolume.y, dimsvolume.z * curbatch), "d_projected.mrc");
			//d_FFTR2C(d_projected, d_projectedftctf, &planforw);
			//d_NormFTMonolithic(d_projectedftctf, d_projectedftctf, ElementsFFT(dimsvolume), curbatch);

			for (uint v = 0; v < nvolumes; v++)
			{
				// Multiply current experimental volume by conjugate references
				{
					int TpB = 128;
					dim3 grid = dim3((ElementsFFT(dimsvolume) + TpB - 1) / TpB, 1, 1);
					BatchComplexConjMultiplyKernel << <grid, TpB >> > (d_experimentalft + ElementsFFT(dimsvolume) * v, d_projectedftctf, d_projectedftctfcorr, ElementsFFT(dimsvolume), curbatch);
				}

				//d_Abs(d_projectedftctfcorr, d_corrmag, ElementsFFT(dimsvolume) * curbatch);
				//d_MaxOp(d_corrmag, 1e-20f, d_corrmag, ElementsFFT(dimsvolume) * curbatch);
				//d_Sqrt(d_corrmag, d_corrmag, ElementsFFT(dimsvolume) * curbatch);
				//d_ComplexDivideByVector(d_projectedftctfcorr, d_corrmag, d_projectedftctfcorr, ElementsFFT(dimsvolume) * curbatch);
				//d_ComplexMultiplyByVector(d_projectedftctfcorr, d_ctf, d_projectedftctfcorr, ElementsFFT(dimsvolume), curbatch);

				d_IFFTC2R(d_projectedftctfcorr, d_projected, &planback);
				//d_WriteMRC(d_projected, toInt3(dimsvolume.x, dimsvolume.y, dimsvolume.z * curbatch), "d_correlation_individual.mrc");

				// Update correlation and angles with best values
				{
					int TpB = 128;
					dim3 grid = dim3((Elements(dimsvolume) + TpB - 1) / TpB, 1, 1);
					UpdateCorrelationKernel <<<grid, TpB>>> (d_projected, 
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

		//cufftDestroy(planforw);
		cufftDestroy(planback);

		// Normalize correlation by local standard deviation
		{
			/*d_IFFTC2R(d_experimentalft, d_projected, ndims, dimsvolume, nvolumes, false);
			cufftHandle planforwstd = d_FFTR2CGetPlan(ndims, dimsvolume);
			cufftHandle planbackstd = d_IFFTC2RGetPlan(ndims, dimsvolume);

			for (uint v = 0; v < nvolumes; v++)
				d_LocalStd(d_projected + Elements(dimsvolume) * v, dimsvolume, NULL, maskradius, d_projected + Elements(dimsvolume) * v, NULL, planforwstd, planbackstd);

			cufftDestroy(planbackstd);
			cufftDestroy(planforwstd);*/

			//d_WriteMRC(d_projected, toInt3(dimsvolume.x, dimsvolume.y, dimsvolume.z * nvolumes), "d_localstd.mrc");

			//d_DivideSafeByVector(d_bestcorrelation, d_projected, d_bestcorrelation, Elements(dimsvolume) * nvolumes);
		}

		//cudaFree(d_corrmag);
		cudaFree(d_projected);
		cudaFree(d_projectedftctfcorr);
		cudaFree(d_projectedftctf);
		//cudaFree(d_projectedft);
		//cudaFree(d_angles);
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

	#define AngleBatch 2
	#define Diff2TpB 128

	__global__ void PickSubTomogramsDiff2Kernel(tcomplex* d_projectedft, tfloat* d_ctf, tcomplex* d_experimentalft, int3 dimsvolume, uint elementsvolume, int3 dimsrelevant, tfloat* d_bestdiff2, float* d_bestangle, int firstangle);

	void d_PickSubTomogramsDiff2(cudaTex t_projectordataRe,
								 cudaTex t_projectordataIm,
								 tfloat projectoroversample,
								 int3 dimsprojector,
								 tcomplex* d_experimentalft,
								 tfloat* d_ctf,
								 int3 dimsvolume,
								 uint nvolumes,
								 int3 dimsrelevant,
								 tfloat3* h_angles,
								 uint nangles,
								 tfloat* d_bestdiff2,
								 float* d_bestangle)
	{
		uint batchsize = AngleBatch;

		d_ValueFill(d_bestdiff2, Elements(dimsrelevant) * nvolumes, (tfloat)1e30f);
		d_ValueFill(d_bestangle, Elements(dimsrelevant) * nvolumes, (float)0);

		tfloat3* d_angles = (tfloat3*)CudaMallocFromHostArray(h_angles, nangles * sizeof(tfloat3));

		tcomplex* d_projectedft;
		cudaMalloc((void**)&d_projectedft, ElementsFFT(dimsvolume) * batchsize * nvolumes * sizeof(tcomplex));

		for (uint b = 0; b < nangles; b += batchsize)
		{
			uint curbatch = tmin(batchsize, nangles - b);

			d_rlnProject(t_projectordataRe, t_projectordataIm, dimsprojector, d_projectedft, dimsvolume, h_angles + b, projectoroversample, curbatch);

			for (uint v = 0; v < nvolumes; v++)
			{
				dim3 grid = dim3((Elements(dimsrelevant) + Diff2TpB - 1) / Diff2TpB, 1, 1);
				PickSubTomogramsDiff2Kernel <<<grid, Diff2TpB>>> (d_projectedft, 
																	d_ctf + v * ElementsFFT(dimsvolume), 
																	d_experimentalft + v * ElementsFFT(dimsvolume), 
																	dimsvolume, 
																	ElementsFFT(dimsvolume), 
																	dimsrelevant, 
																	d_bestdiff2 + v * Elements(dimsrelevant), 
																	d_bestangle + v * Elements(dimsrelevant), 
																	b);
			}
		}

		cudaFree(d_projectedft);
		cudaFree(d_angles);
	}

	__global__ void PickSubTomogramsDiff2Kernel(tcomplex* d_projectedft, tfloat* d_ctf, tcomplex* d_experimentalft, int3 dimsvolume, uint elementsvolume, int3 dimsrelevant, tfloat* d_bestdiff2, float* d_bestangle, int firstangle)
	{
		float3 shift;
		{
			uint id = blockIdx.x * blockDim.x + threadIdx.x;

			uint idz = id / (dimsrelevant.x * dimsrelevant.y);
			uint idy = (id - idz * dimsrelevant.x * dimsrelevant.y) / dimsrelevant.x;
			uint idx = id - (idz * dimsrelevant.y + idy) * dimsrelevant.x;

			shift = make_float3(-((float)idx - dimsrelevant.x / 2) / dimsvolume.x,
								-((float)idy - dimsrelevant.y / 2) / dimsvolume.y,
								-((float)idz - dimsrelevant.z / 2) / dimsvolume.z) * PI2;
		}

		__shared__ tcomplex s_projectedft[Diff2TpB];
		__shared__ tcomplex s_experimentalft[Diff2TpB];
		__shared__ tfloat s_ctf[Diff2TpB];
		__shared__ float s_deltax[Diff2TpB];
		__shared__ float s_deltay[Diff2TpB];
		__shared__ float s_deltaz[Diff2TpB];

		float scores[AngleBatch];
		for (int a = 0; a < AngleBatch; a++)
			scores[a] = 0;

		for (uint i = 0; i < elementsvolume; i += Diff2TpB)
		{
			uint id = i + threadIdx.x;
			if (id < elementsvolume)
			{
				s_experimentalft[threadIdx.x] = d_experimentalft[id];
				s_ctf[threadIdx.x] = d_ctf[id];

				uint idz = id / ElementsFFT2(dimsvolume);
				uint idy = (id - idz * ElementsFFT2(dimsvolume)) / ElementsFFT1(dimsvolume.x);
				uint idx = id - (idz * dimsvolume.y + idy) * ElementsFFT1(dimsvolume.x);

				s_deltax[threadIdx.x] = (float)idx;
				s_deltay[threadIdx.x] = idy > dimsvolume.y / 2 ? (float)idy - dimsvolume.y : (float)idy;
				s_deltaz[threadIdx.x] = idz > dimsvolume.z / 2 ? (float)idz - dimsvolume.z : (float)idz;
			}
			else
			{
				s_experimentalft[threadIdx.x] = make_cuComplex(0, 0);
				s_ctf[threadIdx.x] = 0;
			}

			__syncthreads();

			for (int a = 0; a < AngleBatch; a++)
			{
				if (id < elementsvolume)
					s_projectedft[threadIdx.x] = d_projectedft[a * elementsvolume + id] * s_ctf[threadIdx.x];
				else
					s_projectedft[threadIdx.x] = make_cuComplex(0, 0);

				__syncthreads();

				float tempscore = 0;

				for (int ii = 0; ii < Diff2TpB; ii++)
				{
					float factor = s_deltax[ii] * shift.x + s_deltay[ii] * shift.y + s_deltaz[ii] * shift.z;
					float fs, fc;
					__sincosf(factor, &fs, &fc);
					tcomplex delta = make_cuComplex(fc, fs);

					tcomplex diff = cmul(s_projectedft[ii], delta) - s_experimentalft[ii];
					tempscore += dotp2(diff, diff);
				}

				scores[a] += tempscore;

				__syncthreads();
			}
		}

		int bestangle = 0;
		float bestscore = 1e30f;
		for (int a = 0; a < AngleBatch; a++)
			if (scores[a] < bestscore)
			{
				bestscore = scores[a];
				bestangle = a;
			}

		{
			uint id = blockIdx.x * blockDim.x + threadIdx.x;

			if (id < Elements(dimsrelevant))
			{
				if (bestscore < d_bestdiff2[id])
				{
					d_bestdiff2[id] = bestscore;
					d_bestangle[id] = bestangle + firstangle;
				}
			}
		}
	}
}