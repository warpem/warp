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

	// Forward declarations for morphological kernels
	template<int connectivity> __global__ void GreyscaleErode3DKernel(tfloat* d_input, tfloat* d_output, int3 dims);
	template<int connectivity> __global__ void GreyscaleDilate3DKernel(tfloat* d_input, tfloat* d_output, int3 dims);

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

		d_ValueFill<tfloat>(d_bestcorrelation, Elements(dimsvolume) * nvolumes, (tfloat)-1e30);
		d_ValueFill<float>(d_bestangle, Elements(dimsvolume) * nvolumes, (float)0);

		tcomplex* d_projectedftctf;
		cudaMalloc((void**)&d_projectedftctf, ElementsFFT(dimsvolume) * batchsize * sizeof(tcomplex));
		tcomplex* d_projectedftctfcorr;
		cudaMalloc((void**)&d_projectedftctfcorr, ElementsFFT(dimsvolume) * batchsize * sizeof(tcomplex));
		tfloat* d_projected;
		cudaMalloc((void**)&d_projected, Elements(dimsvolume) * batchsize * sizeof(tfloat));

		cufftHandle planback = d_IFFTC2RGetPlan(ndims, dimsvolume, batchsize);

		bool debug = false;

		for (uint b = 0; b < nangles; b += batchsize)
		{
			uint curbatch = tmin(batchsize, nangles - b);

			// d_projectedftctf will contain rotated reference volume multiplied by CTF
			d_rlnProjectCTFMult(t_projectordataRe, t_projectordataIm, d_ctf, dimsprojector, d_projectedftctf, dimsvolume, h_angles + b, projectoroversample, curbatch);

			d_NormFTMonolithic(d_projectedftctf, d_projectedftctf, ElementsFFT(dimsvolume), curbatch);

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

	void d_PickLargeVolume(
		cudaTex t_projectordataRe,
		cudaTex t_projectordataIm,
		tfloat projectoroversample,
		int3 dimsprojector,
		tcomplex* d_experimentalft,
		tfloat* d_ctf,
		int3 dimsvolume,
		tfloat3* h_angles,
		uint nangles,
		uint batchangles,
		tfloat maskradius,
		tfloat* d_bestcorrelation,
		float* d_bestangle,
		float* h_progressfraction)
	{
		uint batchsize = batchangles;
		int3 dimsvolumecube = make_int3(dimsvolume.z, dimsvolume.z, dimsvolume.z);

		d_ValueFill(d_bestcorrelation, Elements(dimsvolume), (tfloat)-1e30);
		d_ValueFill(d_bestangle, Elements(dimsvolume), (float)0);

		tcomplex* d_projectedftconv;
		cudaMalloc((void**)&d_projectedftconv, ElementsFFT(dimsvolumecube) * batchsize * sizeof(tcomplex));
		tfloat* d_projected;
		cudaMalloc((void**)&d_projected, Elements(dimsvolumecube) * batchsize * sizeof(tfloat));
		tfloat* d_projectedpadded;
		cudaMalloc((void**)&d_projectedpadded, Elements(dimsvolume) * batchsize * sizeof(tfloat));

		tcomplex* d_projectedftctfcorr;
		cudaMalloc((void**)&d_projectedftctfcorr, ElementsFFT(dimsvolume) * batchsize * sizeof(tcomplex));

		cufftHandle planbackcube = d_IFFTC2RGetPlan(3, dimsvolumecube, batchsize);

		cufftHandle planforw = d_FFTR2CGetPlan(3, dimsvolume, batchsize);
		cufftHandle planback = d_IFFTC2RGetPlan(3, dimsvolume, batchsize);

		bool debug = false;

		for (uint b = 0; b < nangles; b += batchsize)
		{
			uint curbatch = tmin(batchsize, nangles - b);

			// d_projectedftconv will contain rotated reference volume multiplied by CTF
			d_rlnProjectCTFMult(t_projectordataRe, t_projectordataIm, d_ctf, dimsprojector, d_projectedftconv, dimsvolumecube, h_angles + b, projectoroversample, curbatch);
			d_NormFTMonolithic(d_projectedftconv, d_projectedftconv, ElementsFFT(dimsvolumecube), curbatch);

			// IFFT and pad to dimsvolume
			d_IFFTC2R(d_projectedftconv, d_projected, &planbackcube);
			d_MultiplyByScalar(d_projected, d_projected, Elements(dimsvolumecube) * curbatch, 1.0f / (tfloat)Elements(dimsvolumecube));
			if (debug && b == 0)
				d_WriteMRC(d_projected, toInt3(dimsvolumecube.x, dimsvolumecube.y, dimsvolumecube.z * curbatch), "d_projected.mrc");

			d_FFTFullPad(d_projected, d_projectedpadded, dimsvolumecube, dimsvolume, curbatch);
			if (debug && b == 0)
				d_WriteMRC(d_projectedpadded, toInt3(dimsvolume.x, dimsvolume.y, dimsvolume.z * curbatch), "d_projectedpadded.mrc");

			// FFT back for cross-correlation
			d_FFTR2C(d_projectedpadded, d_projectedftctfcorr, &planforw);

			{
				// Multiply current experimental volume by conjugate references
				{
					int TpB = 128;
					dim3 grid = dim3(tmin((ElementsFFT(dimsvolume) + TpB - 1) / TpB, 2048), 1, 1);
					BatchComplexConjMultiplyKernel << <grid, TpB >> > (d_experimentalft, d_projectedftctfcorr, d_projectedftctfcorr, ElementsFFT(dimsvolume), curbatch);
				}

				d_IFFTC2R(d_projectedftctfcorr, d_projectedpadded, &planback);

				if (debug && b == 0)
					d_WriteMRC(d_projectedpadded, toInt3(dimsvolume.x, dimsvolume.y, dimsvolume.z * curbatch), "d_corr.mrc");

				// Update correlation and angles with best values
				{
					int TpB = 128;
					dim3 grid = dim3(tmin((Elements(dimsvolume) + TpB - 1) / TpB, 2048), 1, 1);
					UpdateCorrelationKernel << <grid, TpB >> > (d_projectedpadded,
						Elements(dimsvolume),
						curbatch,
						b,
						d_bestcorrelation,
						d_bestangle);
				}

				//d_WriteMRC(d_bestcorrelation + Elements(dimsvolume) * v, dimsvolume, "d_correlation_best.mrc");
			}

			if (h_progressfraction)
				*h_progressfraction = (float)(b + curbatch) / nangles;
		}


		cufftDestroy(planbackcube);
		cufftDestroy(planforw);
		cufftDestroy(planback);

		cudaFree(d_projected);
		cudaFree(d_projectedpadded);
		cudaFree(d_projectedftctfcorr);
		cudaFree(d_projectedftconv);
	}

	////////////////////
	// Top-Hat Transform
	////////////////////

	// Erosion kernel - connectivity 1 (6 face neighbors)
	template<>
	__global__ void GreyscaleErode3DKernel<1>(tfloat* d_input, tfloat* d_output, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x) return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y) return;
		int idz = blockIdx.z;

		size_t stride_y = dims.x;
		size_t stride_z = (size_t)dims.x * dims.y;
		size_t centerIdx = idz * stride_z + idy * stride_y + idx;

		tfloat minVal = d_input[centerIdx];

		// 6 face neighbors
		if (idx > 0)           minVal = fminf(minVal, d_input[centerIdx - 1]);
		if (idx < dims.x - 1)  minVal = fminf(minVal, d_input[centerIdx + 1]);
		if (idy > 0)           minVal = fminf(minVal, d_input[centerIdx - stride_y]);
		if (idy < dims.y - 1)  minVal = fminf(minVal, d_input[centerIdx + stride_y]);
		if (idz > 0)           minVal = fminf(minVal, d_input[centerIdx - stride_z]);
		if (idz < dims.z - 1)  minVal = fminf(minVal, d_input[centerIdx + stride_z]);

		d_output[centerIdx] = minVal;
	}

	// Erosion kernel - connectivity 2 (18 neighbors: faces + edges)
	template<>
	__global__ void GreyscaleErode3DKernel<2>(tfloat* d_input, tfloat* d_output, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x) return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y) return;
		int idz = blockIdx.z;

		size_t stride_y = dims.x;
		size_t stride_z = (size_t)dims.x * dims.y;
		size_t centerIdx = idz * stride_z + idy * stride_y + idx;

		tfloat minVal = d_input[centerIdx];

		// 6 face neighbors
		if (idx > 0)           minVal = fminf(minVal, d_input[centerIdx - 1]);
		if (idx < dims.x - 1)  minVal = fminf(minVal, d_input[centerIdx + 1]);
		if (idy > 0)           minVal = fminf(minVal, d_input[centerIdx - stride_y]);
		if (idy < dims.y - 1)  minVal = fminf(minVal, d_input[centerIdx + stride_y]);
		if (idz > 0)           minVal = fminf(minVal, d_input[centerIdx - stride_z]);
		if (idz < dims.z - 1)  minVal = fminf(minVal, d_input[centerIdx + stride_z]);

		// 12 edge neighbors
		if (idx > 0 && idy > 0)                     minVal = fminf(minVal, d_input[centerIdx - 1 - stride_y]);
		if (idx < dims.x - 1 && idy > 0)            minVal = fminf(minVal, d_input[centerIdx + 1 - stride_y]);
		if (idx > 0 && idy < dims.y - 1)            minVal = fminf(minVal, d_input[centerIdx - 1 + stride_y]);
		if (idx < dims.x - 1 && idy < dims.y - 1)   minVal = fminf(minVal, d_input[centerIdx + 1 + stride_y]);
		if (idx > 0 && idz > 0)                     minVal = fminf(minVal, d_input[centerIdx - 1 - stride_z]);
		if (idx < dims.x - 1 && idz > 0)            minVal = fminf(minVal, d_input[centerIdx + 1 - stride_z]);
		if (idx > 0 && idz < dims.z - 1)            minVal = fminf(minVal, d_input[centerIdx - 1 + stride_z]);
		if (idx < dims.x - 1 && idz < dims.z - 1)   minVal = fminf(minVal, d_input[centerIdx + 1 + stride_z]);
		if (idy > 0 && idz > 0)                     minVal = fminf(minVal, d_input[centerIdx - stride_y - stride_z]);
		if (idy < dims.y - 1 && idz > 0)            minVal = fminf(minVal, d_input[centerIdx + stride_y - stride_z]);
		if (idy > 0 && idz < dims.z - 1)            minVal = fminf(minVal, d_input[centerIdx - stride_y + stride_z]);
		if (idy < dims.y - 1 && idz < dims.z - 1)   minVal = fminf(minVal, d_input[centerIdx + stride_y + stride_z]);

		d_output[centerIdx] = minVal;
	}

	// Erosion kernel - connectivity 3 (26 neighbors: full 3x3x3 cube)
	template<>
	__global__ void GreyscaleErode3DKernel<3>(tfloat* d_input, tfloat* d_output, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x) return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y) return;
		int idz = blockIdx.z;

		size_t stride_y = dims.x;
		size_t stride_z = (size_t)dims.x * dims.y;
		size_t centerIdx = idz * stride_z + idy * stride_y + idx;

		tfloat minVal = d_input[centerIdx];

		// Iterate over 3x3x3 neighborhood
		for (int dz = -1; dz <= 1; dz++)
		{
			int nz = idz + dz;
			if (nz < 0 || nz >= dims.z) continue;

			for (int dy = -1; dy <= 1; dy++)
			{
				int ny = idy + dy;
				if (ny < 0 || ny >= dims.y) continue;

				for (int dx = -1; dx <= 1; dx++)
				{
					int nx = idx + dx;
					if (nx < 0 || nx >= dims.x) continue;

					size_t neighborIdx = nz * stride_z + ny * stride_y + nx;
					minVal = fminf(minVal, d_input[neighborIdx]);
				}
			}
		}

		d_output[centerIdx] = minVal;
	}

	// Dilation kernel - connectivity 1 (6 face neighbors)
	template<>
	__global__ void GreyscaleDilate3DKernel<1>(tfloat* d_input, tfloat* d_output, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x) return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y) return;
		int idz = blockIdx.z;

		size_t stride_y = dims.x;
		size_t stride_z = (size_t)dims.x * dims.y;
		size_t centerIdx = idz * stride_z + idy * stride_y + idx;

		tfloat maxVal = d_input[centerIdx];

		// 6 face neighbors
		if (idx > 0)           maxVal = fmaxf(maxVal, d_input[centerIdx - 1]);
		if (idx < dims.x - 1)  maxVal = fmaxf(maxVal, d_input[centerIdx + 1]);
		if (idy > 0)           maxVal = fmaxf(maxVal, d_input[centerIdx - stride_y]);
		if (idy < dims.y - 1)  maxVal = fmaxf(maxVal, d_input[centerIdx + stride_y]);
		if (idz > 0)           maxVal = fmaxf(maxVal, d_input[centerIdx - stride_z]);
		if (idz < dims.z - 1)  maxVal = fmaxf(maxVal, d_input[centerIdx + stride_z]);

		d_output[centerIdx] = maxVal;
	}

	// Dilation kernel - connectivity 2 (18 neighbors: faces + edges)
	template<>
	__global__ void GreyscaleDilate3DKernel<2>(tfloat* d_input, tfloat* d_output, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x) return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y) return;
		int idz = blockIdx.z;

		size_t stride_y = dims.x;
		size_t stride_z = (size_t)dims.x * dims.y;
		size_t centerIdx = idz * stride_z + idy * stride_y + idx;

		tfloat maxVal = d_input[centerIdx];

		// 6 face neighbors
		if (idx > 0)           maxVal = fmaxf(maxVal, d_input[centerIdx - 1]);
		if (idx < dims.x - 1)  maxVal = fmaxf(maxVal, d_input[centerIdx + 1]);
		if (idy > 0)           maxVal = fmaxf(maxVal, d_input[centerIdx - stride_y]);
		if (idy < dims.y - 1)  maxVal = fmaxf(maxVal, d_input[centerIdx + stride_y]);
		if (idz > 0)           maxVal = fmaxf(maxVal, d_input[centerIdx - stride_z]);
		if (idz < dims.z - 1)  maxVal = fmaxf(maxVal, d_input[centerIdx + stride_z]);

		// 12 edge neighbors
		if (idx > 0 && idy > 0)                     maxVal = fmaxf(maxVal, d_input[centerIdx - 1 - stride_y]);
		if (idx < dims.x - 1 && idy > 0)            maxVal = fmaxf(maxVal, d_input[centerIdx + 1 - stride_y]);
		if (idx > 0 && idy < dims.y - 1)            maxVal = fmaxf(maxVal, d_input[centerIdx - 1 + stride_y]);
		if (idx < dims.x - 1 && idy < dims.y - 1)   maxVal = fmaxf(maxVal, d_input[centerIdx + 1 + stride_y]);
		if (idx > 0 && idz > 0)                     maxVal = fmaxf(maxVal, d_input[centerIdx - 1 - stride_z]);
		if (idx < dims.x - 1 && idz > 0)            maxVal = fmaxf(maxVal, d_input[centerIdx + 1 - stride_z]);
		if (idx > 0 && idz < dims.z - 1)            maxVal = fmaxf(maxVal, d_input[centerIdx - 1 + stride_z]);
		if (idx < dims.x - 1 && idz < dims.z - 1)   maxVal = fmaxf(maxVal, d_input[centerIdx + 1 + stride_z]);
		if (idy > 0 && idz > 0)                     maxVal = fmaxf(maxVal, d_input[centerIdx - stride_y - stride_z]);
		if (idy < dims.y - 1 && idz > 0)            maxVal = fmaxf(maxVal, d_input[centerIdx + stride_y - stride_z]);
		if (idy > 0 && idz < dims.z - 1)            maxVal = fmaxf(maxVal, d_input[centerIdx - stride_y + stride_z]);
		if (idy < dims.y - 1 && idz < dims.z - 1)   maxVal = fmaxf(maxVal, d_input[centerIdx + stride_y + stride_z]);

		d_output[centerIdx] = maxVal;
	}

	// Dilation kernel - connectivity 3 (26 neighbors: full 3x3x3 cube)
	template<>
	__global__ void GreyscaleDilate3DKernel<3>(tfloat* d_input, tfloat* d_output, int3 dims)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x) return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dims.y) return;
		int idz = blockIdx.z;

		size_t stride_y = dims.x;
		size_t stride_z = (size_t)dims.x * dims.y;
		size_t centerIdx = idz * stride_z + idy * stride_y + idx;

		tfloat maxVal = d_input[centerIdx];

		// Iterate over 3x3x3 neighborhood
		for (int dz = -1; dz <= 1; dz++)
		{
			int nz = idz + dz;
			if (nz < 0 || nz >= dims.z) continue;

			for (int dy = -1; dy <= 1; dy++)
			{
				int ny = idy + dy;
				if (ny < 0 || ny >= dims.y) continue;

				for (int dx = -1; dx <= 1; dx++)
				{
					int nx = idx + dx;
					if (nx < 0 || nx >= dims.x) continue;

					size_t neighborIdx = nz * stride_z + ny * stride_y + nx;
					maxVal = fmaxf(maxVal, d_input[neighborIdx]);
				}
			}
		}

		d_output[centerIdx] = maxVal;
	}

	void d_GreyscaleErode3D(tfloat* d_input, tfloat* d_output, int3 dims, int connectivity)
	{
		dim3 TpB(32, 8);
		dim3 grid((dims.x + TpB.x - 1) / TpB.x, (dims.y + TpB.y - 1) / TpB.y, dims.z);

		switch (connectivity)
		{
		case 1:
			GreyscaleErode3DKernel<1><<<grid, TpB>>>(d_input, d_output, dims);
			break;
		case 2:
			GreyscaleErode3DKernel<2><<<grid, TpB>>>(d_input, d_output, dims);
			break;
		case 3:
			GreyscaleErode3DKernel<3><<<grid, TpB>>>(d_input, d_output, dims);
			break;
		default:
			throw std::invalid_argument("connectivity must be 1, 2, or 3");
		}
	}

	void d_GreyscaleDilate3D(tfloat* d_input, tfloat* d_output, int3 dims, int connectivity)
	{
		dim3 TpB(32, 8);
		dim3 grid((dims.x + TpB.x - 1) / TpB.x, (dims.y + TpB.y - 1) / TpB.y, dims.z);

		switch (connectivity)
		{
		case 1:
			GreyscaleDilate3DKernel<1><<<grid, TpB>>>(d_input, d_output, dims);
			break;
		case 2:
			GreyscaleDilate3DKernel<2><<<grid, TpB>>>(d_input, d_output, dims);
			break;
		case 3:
			GreyscaleDilate3DKernel<3><<<grid, TpB>>>(d_input, d_output, dims);
			break;
		default:
			throw std::invalid_argument("connectivity must be 1, 2, or 3");
		}
	}

	void d_TopHatTransform(tfloat* d_input, tfloat* d_output, int3 dims, int connectivity)
	{
		// Allocate temporary buffer for erosion result
		tfloat* d_eroded;
		cudaMalloc((void**)&d_eroded, Elements(dims) * sizeof(tfloat));

		// Step 1: Erosion - input -> d_eroded
		d_GreyscaleErode3D(d_input, d_eroded, dims, connectivity);

		// Step 2: Dilation - d_eroded -> d_output (this is the opening)
		d_GreyscaleDilate3D(d_eroded, d_output, dims, connectivity);

		// Step 3: Subtraction - input - opening -> d_output
		d_SubtractVector(d_input, d_output, d_output, Elements(dims), 1);

		cudaFree(d_eroded);
	}
}