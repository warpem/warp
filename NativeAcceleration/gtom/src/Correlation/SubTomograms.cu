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
#include "gtom/include/TopKCorrelation.h"

namespace gtom
{
	__global__ void BatchComplexConjMultiplyKernel(tcomplex* d_input1, tcomplex* d_input2, tcomplex* d_output, uint vectorlength, uint batch);
	__global__ void UpdateCorrelationKernel(tfloat* d_correlation, uint vectorlength, uint batch, int batchoffset, tfloat* d_bestcorrelation, float* d_bestangle);
	__global__ void UpdateCorrelationTopKKernel(const tfloat* d_correlation, size_t vectorlength, uint batch, uint batchoffset, uint topk, tfloat* d_topcorrelations, float* d_topangles);

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
		d_PickLargeVolumeTopK(t_projectordataRe, t_projectordataIm, projectoroversample,
			dimsprojector, d_experimentalft, d_ctf, dimsvolume, h_angles, nangles,
			batchangles, maskradius, 1, d_bestcorrelation, d_bestangle, h_progressfraction);
	}

	void d_PickLargeVolumeTopK(
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
		uint topk,
		tfloat* d_topcorrelations,
		float* d_topangles,
		float* h_progressfraction)
	{
		// Native callers must validate these as well as their output allocation.
		// Float angle IDs represent all integers in this range exactly.
		if (topk == 0 || batchangles == 0 || nangles > 16777216U)
			return;

		const size_t elements = Elements(dimsvolume);
		d_ValueFill(d_topcorrelations, elements * topk, (tfloat)-INFINITY);
		d_ValueFill(d_topangles, elements * topk, (float)-1);
		if (h_progressfraction)
			*h_progressfraction = nangles == 0 ? 1.0f : 0.0f;
		if (nangles == 0)
			return;

		uint batchsize = tmin(batchangles, nangles);
		int3 dimsvolumecube = make_int3(dimsvolume.z, dimsvolume.z, dimsvolume.z);

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
			if (curbatch != batchsize)
			{
				// Resize the plans for the final partial batch. This avoids reading
				// uninitialized scratch entries without retaining extra FFT workspaces.
				cufftDestroy(planbackcube);
				cufftDestroy(planforw);
				cufftDestroy(planback);
				planbackcube = d_IFFTC2RGetPlan(3, dimsvolumecube, curbatch);
				planforw = d_FFTR2CGetPlan(3, dimsvolume, curbatch);
				planback = d_IFFTC2RGetPlan(3, dimsvolume, curbatch);
			}

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

				// Update each voxel's sorted orientation leaderboard.
				{
					int TpB = 128;
					dim3 grid = dim3(tmin((Elements(dimsvolume) + TpB - 1) / TpB, 2048), 1, 1);
					UpdateCorrelationTopKKernel << <grid, TpB >> > (d_projectedpadded,
						elements,
						curbatch,
						b,
						topk,
						d_topcorrelations,
						d_topangles);
				}
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

	__global__ void UpdateCorrelationTopKKernel(const tfloat* d_correlation,
		size_t vectorlength, uint batch, uint batchoffset, uint topk,
		tfloat* d_topcorrelations, float* d_topangles)
	{
		for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
			id < vectorlength; id += (size_t)gridDim.x * blockDim.x)
		{
			for (uint b = 0; b < batch; ++b)
				InsertCorrelationTopK(d_correlation[(size_t)b * vectorlength + id],
					(float)(batchoffset + b), d_topcorrelations, d_topangles,
					id, vectorlength, topk);
		}
	}

    __global__ void GatherTemplateMatchTopKKernel(const tfloat* d_scores,
        const float* d_angles, int3 dims, const int3* d_positions,
        size_t nentries, int topk, float2* d_output)
    {
        const size_t elements = (size_t)dims.x * dims.y * dims.z;
        for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
             id < nentries; id += (size_t)gridDim.x * blockDim.x)
        {
            const int3 p = d_positions[id / topk];
            float2 value = make_float2(-INFINITY, -1);
            if (p.x >= 0 && p.x < dims.x && p.y >= 0 && p.y < dims.y && p.z >= 0 && p.z < dims.z)
            {
                const size_t voxel = ((size_t)p.z * dims.y + p.y) * dims.x + p.x;
                const size_t offset = (id % topk) * elements + voxel;
                value = make_float2((float)d_scores[offset], d_angles[offset]);
            }
            d_output[id] = value;
        }
    }

    cudaError_t d_GatherTemplateMatchTopK(const tfloat* d_topcorrelations,
        const float* d_topangles, int3 dims, const int3* h_positions,
        int npositions, int topk, float* h_scores, float* h_angles)
    {
        if (npositions < 0 || topk < 1 || dims.x < 1 || dims.y < 1 || dims.z < 1)
            return cudaErrorInvalidValue;
        if (npositions == 0)
            return cudaSuccess;
        if (!d_topcorrelations || !d_topangles || !h_positions || !h_scores || !h_angles)
            return cudaErrorInvalidValue;

        const size_t nentries = (size_t)npositions * topk;
        if (nentries > SIZE_MAX / sizeof(float2))
            return cudaErrorInvalidValue;
        int3* d_positions = NULL;
        float2* d_output = NULL;
        std::vector<float2> output(nentries);
        cudaError_t error = cudaMalloc((void**)&d_positions, (size_t)npositions * sizeof(int3));
        if (error == cudaSuccess)
            error = cudaMalloc((void**)&d_output, nentries * sizeof(float2));
        if (error == cudaSuccess)
            error = cudaMemcpy(d_positions, h_positions, (size_t)npositions * sizeof(int3), cudaMemcpyHostToDevice);
        if (error == cudaSuccess)
        {
            const int threads = 128;
            dim3 grid(tmin((nentries + threads - 1) / threads, 2048), 1, 1);
            GatherTemplateMatchTopKKernel<<<grid, threads>>>(d_topcorrelations,
                d_topangles, dims, d_positions, nentries, topk, d_output);
            error = cudaGetLastError();
        }
        if (error == cudaSuccess)
            error = cudaMemcpy(output.data(), d_output, nentries * sizeof(float2), cudaMemcpyDeviceToHost);
        if (d_output)
            cudaFree(d_output);
        if (d_positions)
            cudaFree(d_positions);
        if (error != cudaSuccess)
            return error;
        for (size_t id = 0; id < nentries; ++id)
        {
            h_scores[id] = output[id].x;
            h_angles[id] = output[id].y;
        }
        return cudaSuccess;
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
