#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Helper.cuh"
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

namespace gtom
{
	__global__ void ConvertWeightsDenseKernel(tfloat* d_weights, uint elements, uint elementsperpart, uint elementsperclass, uint nclasses, uint nrot, uint ntrans, tfloat* d_pdfrot, const tfloat* __restrict__ d_pdftrans, const tfloat* __restrict__ d_mindiff2);
	__global__ void ConvertWeightsSparseKernel(tfloat* d_weightsdense, tfloat* d_weightssparse, uint4* d_combinations, uint nsparse, uint nrot, uint ntrans, tfloat* d_pdfrot, const tfloat* __restrict__ d_pdftrans, const tfloat* __restrict__ d_mindiff2);

	void d_rlnConvertWeightsDense(tfloat* d_weights, uint nparticles, uint nclasses, uint nrot, uint ntrans, tfloat* d_pdfrot, tfloat* d_pdftrans, tfloat* d_mindiff2)
	{
		uint elementsperclass = nrot * ntrans;
		uint elementsperpart = nclasses * elementsperclass;
		uint elements = nparticles * nclasses * elementsperclass;
		int TpB = 128;
		dim3 grid = dim3((elements + TpB - 1) / TpB);
		ConvertWeightsDenseKernel << <grid, TpB >> > (d_weights, elements, elementsperpart, elementsperclass, nclasses, nrot, ntrans, d_pdfrot, d_pdftrans, d_mindiff2);
	}

	__global__ void ConvertWeightsDenseKernel(tfloat* d_weights, uint elements, uint elementsperpart, uint elementsperclass, uint nclasses, uint nrot, uint ntrans, tfloat* d_pdfrot, const tfloat* __restrict__ d_pdftrans, const tfloat* __restrict__ d_mindiff2)
	{
		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
		{
			uint ipart = id / elementsperpart;
			uint iclass = (id % elementsperpart) / elementsperclass;
			uint irot = (id % elementsperclass) / ntrans;
			uint itrans = id % ntrans;

			tfloat pdfrot = d_pdfrot[iclass * nrot + irot];		
			tfloat pdftrans = d_pdftrans[(ipart * nclasses + iclass) * ntrans + itrans];

			tfloat diff2 = d_weights[id];
			diff2 -= d_mindiff2[ipart];

			tfloat weight = pdfrot * pdftrans;
			weight *= exp(-diff2);

			d_weights[id] = weight;
		}
	}

	void d_rlnConvertWeightsSparse(tfloat* d_weightsdense, tfloat* d_weightssparse, uint4* d_combinations, uint nsparse, uint nrot, uint ntrans, tfloat* d_pdfrot, tfloat* d_pdftrans, tfloat* d_mindiff2)
	{
		int TpB = 128;
		dim3 grid = dim3((nsparse + TpB - 1) / TpB);
		ConvertWeightsSparseKernel << <grid, TpB >> > (d_weightsdense, d_weightssparse, d_combinations, nsparse, nrot, ntrans, d_pdfrot, d_pdftrans, d_mindiff2);
	}

	__global__ void ConvertWeightsSparseKernel(tfloat* d_weightsdense, tfloat* d_weightssparse, uint4* d_combinations, uint nsparse, uint nrot, uint ntrans, tfloat* d_pdfrot, const tfloat* __restrict__ d_pdftrans, const tfloat* __restrict__ d_mindiff2)
	{
		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < nsparse; id += gridDim.x * blockDim.x)
		{
			uint iclass = d_combinations[id].x;

			tfloat pdfrot = d_pdfrot[iclass * nrot + d_combinations[id].y];
			tfloat pdftrans = d_pdftrans[iclass * ntrans + d_combinations[id].z];

			uint ihidden = d_combinations[id].w;
			tfloat diff2 = d_weightsdense[ihidden];
			diff2 -= *d_mindiff2;

			tfloat weight = pdfrot * pdftrans;
			weight *= exp(-diff2);

			d_weightsdense[ihidden] = weight;
			d_weightssparse[id] = weight;
		}
	}

	void d_rlnConvertWeightsSort(tfloat* d_input, uint n)
	{
		thrust::sort(thrust::device, d_input, d_input + n);
	}

	struct is_positive
	{
		__host__ __device__ bool operator()(const tfloat x)
		{
			return x > 0.0f;
		}
	};

	void d_rlnConvertWeightsCompact(tfloat* d_input, tfloat* d_output, uint &n)
	{
		tfloat* d_result = thrust::copy_if(thrust::device, d_input, d_input + n, d_output, is_positive());
		n = d_result - d_output;
	}
}
