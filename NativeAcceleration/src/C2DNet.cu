#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>

#include "include/Functions.h"
#include "gtom/include/DeviceFunctions.cuh"
using namespace gtom;

#define C2DNET_THREADS 128
#define C2DNET_FINEPOSES 27

__global__ void C2DNetAlignKernel(float2* d_data,
								  int dimdata,
								  float3* d_poses,
								  int nposesperdata,
								  float anglestep,
								  float shiftstep,
								  int maxshell2,
								  cudaTex referenceRe,
								  cudaTex referenceIm,
								  float supersample,
								  int dimprojector,
								  float* d_scores,
								  float3* d_scoredposes/*,
								  float2* d_debugout*/);

__global__ void TakeFirstNPosesKernel(float3* d_poses,
									  float3* d_posestaken,
									  int ntake,
									  int ntotal,
									  bool skipequal);

__global__ void TransformData2DKernel(float* d_input, float* d_output, int2 dims, float3* d_poses);



__declspec(dllexport) void C2DNetAlign(float2* d_refs,
									   int dimprojector,
									   float supersample,
									   float2* d_dataft,
									   float* d_data,
									   float* d_ctf,
									   int dimdata,
									   float3* d_initposes,
									   int nposesperdata,
									   int initshell,
									   int maxshell,
									   int niters,
									   float anglestep,
									   float shiftstep,
									   int ntop,
									   int batch,
									   float* d_aligneddata,
									   float* d_alignedctf)
{
	cudaTex t_referenceRe, t_referenceIm;
	cudaArray_t a_referenceRe, a_referenceIm;
	{
		float* d_component;
		cudaMalloc((void**)&d_component, ElementsFFT1(dimprojector) * dimprojector * batch * sizeof(float));

		d_Re(d_refs, d_component, ElementsFFT1(dimprojector) * dimprojector * batch);
		d_BindTextureToArray(d_component, a_referenceRe, t_referenceRe, make_int2(ElementsFFT1(dimprojector), dimprojector * batch), cudaFilterModePoint, false);

		d_Im(d_refs, d_component, ElementsFFT1(dimprojector) * dimprojector * batch);
		d_BindTextureToArray(d_component, a_referenceIm, t_referenceIm, make_int2(ElementsFFT1(dimprojector), dimprojector * batch), cudaFilterModePoint, false);

		cudaFree(d_component);
	}

	for (int iter = 0; iter < niters; iter++)
	{
		int curmaxshell = initshell + (int)((float)(maxshell - initshell) / tmax(niters - 1, 1) * iter + 0.5f);

		dim3 TpB = C2DNET_THREADS;
		dim3 grid = dim3(nposesperdata, batch, 1);

		thrust::device_vector<float> d_scores(nposesperdata * C2DNET_FINEPOSES * batch);
		thrust::device_vector<float3> d_scoredposes(nposesperdata * C2DNET_FINEPOSES * batch);

		//float2* d_debugout = CudaMallocValueFilled(ElementsFFT1(dimdata) * dimdata * nposesperdata * C2DNET_FINEPOSES * batch, make_float2(0, 0));
		//cudaMalloc((void**)&d_debugout, ElementsFFT1(dimdata) * dimdata * nposesperdata * batch * sizeof(float2));

		C2DNetAlignKernel << <grid, TpB >> > (d_dataft,
												dimdata,
												d_initposes,
												nposesperdata,
												anglestep,
												shiftstep,
												curmaxshell * curmaxshell,
												t_referenceRe,
												t_referenceIm,
												supersample,
												dimprojector,
												d_scores.data().get(),
												d_scoredposes.data().get()/*,
												d_debugout*/);

		/*float* d_debugoutift;
		cudaMalloc((void**)&d_debugoutift, dimdata * dimdata * nposesperdata * C2DNET_FINEPOSES * batch * sizeof(float));
		d_IFFTC2R(d_debugout, d_debugoutift, 2, make_int3(dimdata, dimdata, 1), nposesperdata * C2DNET_FINEPOSES * batch);
		d_WriteMRC(d_debugoutift, make_int3(dimdata, dimdata, nposesperdata * C2DNET_FINEPOSES * batch), "d_debugout.mrc");*/

		//d_WriteMRC((float*)d_scores.data().get(), make_int3(1, nposesperdata, batch), "d_scores.mrc");

		thrust::host_vector<int> h_dataid(nposesperdata * C2DNET_FINEPOSES * batch);
		for (int i = 0; i < h_dataid.size(); i++)
			h_dataid[i] = i / (nposesperdata * C2DNET_FINEPOSES);
		thrust::device_vector<int> d_dataid(h_dataid);

		thrust::stable_sort_by_key(d_scores.begin(), d_scores.end(), thrust::make_zip_iterator(thrust::make_tuple(d_dataid.begin(), d_scoredposes.begin())));
		thrust::stable_sort_by_key(d_dataid.begin(), d_dataid.end(), thrust::make_zip_iterator(thrust::make_tuple(d_scores.begin(), d_scoredposes.begin())));

		thrust::host_vector<float3> h_sortedposes(d_scoredposes);

		TpB = 32;
		grid = dim3(batch);

		TakeFirstNPosesKernel << < grid, TpB >> > (d_scoredposes.data().get(),
												   d_initposes,
												   iter == niters - 1 ? 1 : ntop,
												   nposesperdata * C2DNET_FINEPOSES,
												   true);

		nposesperdata = ntop;
		anglestep /= 2;
		shiftstep /= 2;
	}

	{
		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dimdata + 15) / 16, (dimdata + 15) / 16, batch);

		TransformData2DKernel << <grid, TpB >> > (d_data, d_aligneddata, make_int2(dimdata, dimdata), d_initposes);
	}

	// Clean up
	{
		cudaDestroyTextureObject(t_referenceRe);
		cudaFreeArray(a_referenceRe);
		cudaDestroyTextureObject(t_referenceIm);
		cudaFreeArray(a_referenceIm);
	}
}


__global__ void C2DNetAlignKernel(float2* d_data,
								  int dimdata,
								  float3* d_poses,
								  int nposesperdata,
								  float anglestep,
								  float shiftstep,
								  int maxshell2,
								  cudaTex referenceRe,
								  cudaTex referenceIm,
								  float supersample,
								  int dimprojector,
								  float* d_scores,
								  float3* d_scoredposes/*,
								  float2* d_debugout*/)
{
	__shared__ float s_ab[C2DNET_THREADS];
	__shared__ float s_a2[C2DNET_THREADS];
	__shared__ float s_b2[C2DNET_THREADS];

	const uint dimdataft = ElementsFFT1(dimdata);
	d_data += dimdataft * dimdata * blockIdx.y;
	d_poses += nposesperdata * blockIdx.y + blockIdx.x;
	//d_debugout += (dimdataft * dimdata * C2DNET_FINEPOSES) * (blockIdx.y * nposesperdata + blockIdx.x);

	int ifine = 0;

	for (int fineangle = -1; fineangle <= 1; fineangle++)
	{
		for (int fineshifty = -1; fineshifty <= 1; fineshifty++)
		{
			for (int fineshiftx = -1; fineshiftx <= 1; fineshiftx++)
			{
				/*if ((fineshifty == 0 || fineshiftx == 0) && fineshifty != fineshiftx)
					continue;*/

				float ab = 0, a2 = 0, b2 = 0;

				float2 shift = make_float2(d_poses[0].x + fineshiftx * shiftstep, d_poses[0].y + fineshifty * shiftstep) / dimdata * PI2;
				float angle = (d_poses[0].z + fineangle * anglestep) * (PI / 180);

				for (uint id = threadIdx.x; id < dimdataft * dimdata; id += blockDim.x)
				{
					uint idx = id % dimdataft;
					uint idy = id / dimdataft;
					int x = idx;
					int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

					if (x * x + y * y >= tmin(dimdata * dimdata / 4, maxshell2))
						continue;

					glm::vec2 pos = glm::vec2(x, y);

					float2 refval = d_GetProjectionSliceFrom2D(referenceRe, referenceIm, dimprojector, pos * supersample, d_Matrix2Rotation(angle), blockIdx.y);

					float2 dataval = d_data[id];
					float shiftfactor = -(shift.x * pos.x + shift.y * pos.y);
					float2 shiftmultiplicator = make_cuComplex(cos(shiftfactor), sin(shiftfactor));
					dataval = cmul(dataval, shiftmultiplicator);
					//d_debugout[id] = refval;

					float cc = dotp2(refval, dataval);

					ab += cc;
					a2 += dotp2(refval, refval);
					b2 += dotp2(dataval, dataval);
				}
				//return;

				s_ab[threadIdx.x] = ab;
				s_a2[threadIdx.x] = a2;
				s_b2[threadIdx.x] = b2;

				__syncthreads();


				if (threadIdx.x == 0)
				{
					for (int i = 1; i < 128; i++)
					{
						ab += s_ab[i];
						a2 += s_a2[i];
						b2 += s_b2[i];
					}

					d_scores[(nposesperdata * blockIdx.y + blockIdx.x) * C2DNET_FINEPOSES + ifine] = -ab / tmax(sqrt(a2 * b2), 1e-20);
					d_scoredposes[(nposesperdata * blockIdx.y + blockIdx.x) * C2DNET_FINEPOSES + ifine] = make_float3(d_poses[0].x + fineshiftx * shiftstep,
																													  d_poses[0].y + fineshifty * shiftstep, 
																													  d_poses[0].z + fineangle * anglestep);
				}

				__syncthreads();

				ifine++;
				//d_debugout += dimdataft * dimdata;
			}
		}
	}
}

__global__ void TakeFirstNPosesKernel(float3* d_poses,
									  float3* d_posestaken,
									  int ntake,
									  int ntotal,
									  bool skipequal)
{
	d_poses += ntotal * blockIdx.x;
	d_posestaken += ntake * blockIdx.x;

	if (threadIdx.x == 0)
	{
		int taken = 0;
		float3 lastval = make_float3(1e10f, 1e10f, 1e10f);

		for (int i = 0; i < ntotal; i++)
		{
			float3 val = d_poses[i];

			if (skipequal)
			{
				float3 diff3 = val - lastval;
				float diff = abs(diff3.x) + abs(diff3.y) + abs(diff3.z);

				// Assuming equal poses will be adjacent after sorting
				if (diff > 1e-6f)
				{
					lastval = val;
					d_posestaken[taken] = val;
					taken++;
				}
			}
			else
			{
				d_posestaken[taken] = val;
				taken++;
			}

			if (taken >= ntake)
				break;
		}

		// If there were not enough unique poses, fill the rest with last value
		for (int i = taken; i < ntake; i++)
			d_posestaken[i] = d_poses[ntotal - 1];
	}
}

__global__ void TransformData2DKernel(float* d_input, float* d_output, int2 dims, float3* d_poses)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dims.x)
		return;
	uint idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= dims.y)
		return;

	d_input += Elements2(dims) * blockIdx.z;
	d_output += Elements2(dims) * blockIdx.z;

	int x, y;
	x = idx;
	y = idy;

	float angle = d_poses[blockIdx.z].z * (PI / 180);
	glm::vec2 shift = glm::vec2(d_poses[blockIdx.z].x, d_poses[blockIdx.z].y);

	glm::vec2 pos = d_Matrix2Rotation(angle) * glm::vec2(x - dims.x / 2, y - dims.y / 2) + glm::vec2(dims.x / 2, dims.y / 2) - shift;

	float val = 0, weights = 0;

	for (int y = -8; y <= 8; y++)
	{
		float yy = floor(pos.y) + y;
		float sincy = sinc(pos.y - yy) * sinc((pos.y - yy) / 8);
		float yy2 = pos.y - yy;
		yy2 *= yy2;

		for (int x = -8; x <= 8; x++)
		{
			float xx = floor(pos.x) + x;
			float sincx = sinc(pos.x - xx) * sinc((pos.x - xx) / 8);
			float xx2 = pos.x - xx;
			xx2 *= xx2;

			tfloat weight = sincy * sincx;
			val += d_input[tmax(0, tmin((int)yy, dims.y - 1)) * dims.x + tmax(0, tmin((int)xx, dims.x - 1))] * weight;
			weights += weight;
		}
	}

	d_output[idy * dims.x + idx] = val / weights;
}