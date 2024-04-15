#include "gtom/include/Prerequisites.cuh"

namespace gtom
{
	template<bool docc> __global__ void DiffRemapDenseKernel(tfloat* d_input, tfloat* d_output, uint3* d_orientationindices, uint elements, uint iclass, uint nparticles, uint nclasses, uint nrot, uint ntrans, uint ntranspadded, const tfloat* __restrict__ d_xi2imgs, const tfloat* __restrict__ d_sqrtxi2);
	template<bool docc> __global__ void DiffRemapSparseKernel(tfloat* d_input, tfloat* d_output, tfloat* d_mindiff2, const uint3* __restrict__ d_combinations, uint* d_hiddenover, uint elements, uint tileelements, uint weightsperpart, uint nparticles, const tfloat* __restrict__ d_xi2imgs, const tfloat* __restrict__ d_sqrtxi2);

	void d_rlnDiffRemapDense(tfloat* d_input, tfloat* d_output, uint3* d_orientationindices, uint norientations, uint iclass, uint nparticles, uint nclasses, uint nrot, uint ntrans, uint ntranspadded, tfloat* d_xi2imgs, tfloat* d_sqrtxi2, bool docc)
	{
		uint elements = norientations * nparticles * ntrans;
		uint TpB = 128;
		dim3 grid = dim3((elements + TpB - 1) / TpB, 1, 1);
		if (docc)
			DiffRemapDenseKernel<true> <<<grid, TpB>>> (d_input, d_output, d_orientationindices, elements, iclass, nparticles, nclasses, nrot, ntrans, ntranspadded, d_xi2imgs, d_sqrtxi2);
		else
			DiffRemapDenseKernel<false> <<<grid, TpB>>> (d_input, d_output, d_orientationindices, elements, iclass, nparticles, nclasses, nrot, ntrans, ntranspadded, d_xi2imgs, d_sqrtxi2);
	}

	void d_rlnDiffRemapSparse(tfloat* d_input, tfloat* d_output, tfloat* d_mindiff2, uint3* d_combinations, uint* d_hiddenover, uint elements, uint tileelements, uint weightsperpart, uint nparticles, tfloat* d_xi2imgs, tfloat* d_sqrtxi2, bool docc)
	{
		uint TpB = 128;
		dim3 grid = dim3((elements + TpB - 1) / TpB, 1, 1);
		if (docc)
			DiffRemapSparseKernel<true> << <grid, TpB >> > (d_input, d_output, d_mindiff2, d_combinations, d_hiddenover, elements, tileelements, weightsperpart, nparticles, d_xi2imgs, d_sqrtxi2);
		else
			DiffRemapSparseKernel<false> << <grid, TpB >> > (d_input, d_output, d_mindiff2, d_combinations, d_hiddenover, elements, tileelements, weightsperpart, nparticles, d_xi2imgs, d_sqrtxi2);
	}

	template<bool docc> __global__ void DiffRemapDenseKernel(tfloat* d_input, tfloat* d_output, uint3* d_orientationindices, uint elements, uint iclass, uint nparticles, uint nclasses, uint nrot, uint ntrans, uint ntranspadded, const tfloat* __restrict__ d_xi2imgs, const tfloat* __restrict__ d_sqrtxi2)
	{
		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
		{
			uint irot = id / (nparticles * ntrans);
			uint ipart = (id - irot * nparticles * ntrans) / ntrans;
			uint itrans = id % ntrans;
			uint iorient = d_orientationindices[irot].y;

			size_t idinput = (irot * nparticles + ipart) * ntranspadded + itrans;
			size_t idoutput = ((ipart * nclasses + iclass) * nrot + iorient) * ntrans + itrans;

			tfloat val = d_input[idinput];
			if (docc)
				val *= d_sqrtxi2[ipart];	// Already 1/x
			else
				val += d_xi2imgs[ipart];	// Already x/2

			d_output[idoutput] = val;
		}
	}

	__device__ inline void fatomicMin(tfloat* addr, tfloat val)
	{
		int* address_as_i = (int*)addr;
		int old = *address_as_i, assumed;
		do 
		{
			assumed = old;
			old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
		} 
		while (assumed != old);
	}

	template<bool docc> __global__ void DiffRemapSparseKernel(tfloat* d_input, tfloat* d_output, tfloat* d_mindiff2, const uint3* __restrict__ d_combinations, uint* d_hiddenover, uint elements, uint tileelements, uint weightsperpart, uint nparticles, const tfloat* __restrict__ d_xi2imgs, const tfloat* __restrict__ d_sqrtxi2)
	{
		__shared__ tfloat s_mindiff2[1024];
		for (uint i = threadIdx.x; i < nparticles; i += blockDim.x)
			s_mindiff2[i] = 1e30;
		__syncthreads();

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
		{
			uint ipart = d_combinations[id / tileelements].y;

			tfloat val = d_input[id];
			if (docc)
				val *= d_sqrtxi2[ipart];	// Already 1/x
			else
				val += d_xi2imgs[ipart];	// Already x/2

			d_output[weightsperpart * ipart + d_hiddenover[id]] = val;
			
			fatomicMin(&s_mindiff2[ipart], val);
		}
		__syncthreads();

		for (uint i = threadIdx.x; i < nparticles; i += blockDim.x)
			fatomicMin(d_mindiff2 + i, s_mindiff2[i]);
	}
}
