#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Helper.cuh"
#include <thrust/sort.h>

namespace gtom
{
	template<bool doscale> __global__ void StoreWeightsAddKernel(tcomplex* d_input, tfloat* d_ctf, tcomplex* d_ref, tfloat* d_minvsigma2, int* d_mresol, uint elements, tfloat* d_weights, tfloat sigweight, tfloat sumweight, tcomplex* d_output, tfloat* d_outputweights, tfloat* d_sigma2noise, tfloat* d_normcorrection, tfloat* d_priorclass, tfloat* d_correctionxa, tfloat* d_correctionaa);

	void d_rlnStoreWeightsAdd(tcomplex* d_input, tfloat* d_ctf, tcomplex* d_ref, tfloat* d_minvsigma2, int* d_mresol, uint elements, uint ntrans, tfloat* d_weights, tfloat sigweight, tfloat sumweight, tcomplex* d_output, tfloat* d_outputweights, tfloat* d_sigma2noise, tfloat* d_normcorrection, tfloat* d_priorclass, tfloat* d_correctionxa, tfloat* d_correctionaa, bool doscale)
	{
		int TpB = 128;
		dim3 grid = dim3((elements + TpB - 1) / TpB, ntrans, 1);
		if (doscale)
			StoreWeightsAddKernel<true> << <grid, TpB >> > (d_input, d_ctf, d_ref, d_minvsigma2, d_mresol, elements, d_weights, sigweight, 1.0 / sumweight, d_output, d_outputweights, d_sigma2noise, d_normcorrection, d_priorclass, d_correctionxa, d_correctionaa);
		else
			StoreWeightsAddKernel<false> << <grid, TpB >> > (d_input, d_ctf, d_ref, d_minvsigma2, d_mresol, elements, d_weights, sigweight, 1.0 / sumweight, d_output, d_outputweights, d_sigma2noise, d_normcorrection, d_priorclass, d_correctionxa, d_correctionaa);
	}

	template<bool doscale> __global__ void StoreWeightsAddKernel(tcomplex* d_input, tfloat* d_ctf, tcomplex* d_ref, tfloat* d_minvsigma2, int* d_mresol, uint elements, tfloat* d_weights, tfloat sigweight, tfloat sumweight, tcomplex* d_output, tfloat* d_outputweights, tfloat* d_sigma2noise, tfloat* d_normcorrection, tfloat* d_priorclass, tfloat* d_correctionxa, tfloat* d_correctionaa)
	{
		tfloat weight = d_weights[blockIdx.y];
		d_input += elements * blockIdx.y;
		d_output += elements * blockIdx.y;
		d_outputweights += elements * blockIdx.y;
		
		if (weight >= sigweight)
		{
			weight *= sumweight;
			for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
			{
				tfloat ctf = d_ctf[id];
				tfloat weightxinvsigma2 = weight * ctf * d_minvsigma2[id];

				tcomplex shifted = d_input[id];
				d_output[id] += shifted * weightxinvsigma2;
				d_outputweights[id] += weightxinvsigma2 * ctf;

				int ires = d_mresol[id];
				if (ires > -1)
				{
					tcomplex ref = d_ref[id];
					tcomplex diff = make_cuComplex(ref.x - shifted.x, ref.y - shifted.y);
					tfloat wdiff2 = weight * (diff.x * diff.x + diff.y * diff.y);

					atomicAdd(d_sigma2noise + ires, wdiff2);
					atomicAdd(d_normcorrection, wdiff2);

					if (doscale)
						if (d_priorclass[ires] > (tfloat)3)
						{
							atomicAdd(d_correctionxa + ires, weight * dotp2(ref, shifted));
							atomicAdd(d_correctionaa + ires, weight * dotp2(ref, ref));
						}
				}
			}
		}
		else
			for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
			{
				d_output[id] = make_cuComplex(0, 0);
				d_outputweights[id] = 0;
			}
	}
}

