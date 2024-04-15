#include "include/Functions.h"
using namespace gtom;

template<bool useactuators> __global__ void ParticleSoftBodyDeformKernel (float3* d_initialpositions,
																		float3* d_oldpositions,
																		float3* d_newpositions,
																		ushort npositions,
																		int* d_neighborids,
																		int* d_neighboroffsets,
																		float* d_edgelengths,
																		int* d_connectedactuator,
																		float3* d_actuatordeltas,
																		ushort nactuators,
																		ushort batch,
																		float3* d_debug);


__declspec(dllexport) void ParticleSoftBodyDeform (float3* d_initialpositions,
													float3* d_finalpositions,
													uint npositions,
													int* d_neighborids,
													int* d_neighboroffsets,
													float* d_edgelengths,
													int* d_connectedactuator,
													float3* d_actuatordeltas,
													uint nactuators,
													uint niterations,
													uint nrelaxations,
													uint batch)
{
	float3* d_oldpositions;
	cudaMalloc((void**)&d_oldpositions, npositions * batch * sizeof(float3));
	CudaMemcpyMulti((float*)d_oldpositions, (float*)d_initialpositions, npositions * 3, batch);

	float3* d_newpositions;
	cudaMalloc((void**)&d_newpositions, npositions * batch * sizeof(float3));

	//float3* d_debug = (float3*)CudaMallocValueFilled(npositions * 3 * batch, 0.0f);

	for (int i = 0; i < niterations; i++)
	{
	    int TpB = 128;
		dim3 grid = dim3((npositions + TpB - 1) / TpB, batch, 1);

		ParticleSoftBodyDeformKernel<true> <<<grid, TpB>>> (d_initialpositions,
															d_oldpositions,
															d_newpositions,
															npositions,
															d_neighborids,
															d_neighboroffsets,
															d_edgelengths,
															d_connectedactuator,
															d_actuatordeltas,
															nactuators,
															batch,
															NULL);

		//float3* h_debug = (float3*)MallocFromDeviceArray(d_debug, npositions * batch * sizeof(float3));
		//free(h_debug);
		//float3* h_newpositions = (float3*)MallocFromDeviceArray(d_newpositions, npositions * batch * sizeof(float3));
		//free(h_newpositions);
		//cudaDeviceSynchronize();
		float3* d_temp = d_newpositions;
		d_newpositions = d_oldpositions;
		d_oldpositions = d_temp;
	}

	for (int i = 0; i < nrelaxations; i++)
	{
	    int TpB = 128;
		dim3 grid = dim3((npositions + TpB - 1) / TpB, batch, 1);

		ParticleSoftBodyDeformKernel<false> <<<grid, TpB>>> (d_initialpositions,
															d_oldpositions,
															d_newpositions,
															npositions,
															d_neighborids,
															d_neighboroffsets,
															d_edgelengths,
															d_connectedactuator,
															d_actuatordeltas,
															nactuators,
															batch,
															NULL);

		float3* d_temp = d_newpositions;
		d_newpositions = d_oldpositions;
		d_oldpositions = d_temp;
	}

	d_SubtractVector((float*)d_oldpositions, (float*)d_initialpositions, (float*)d_oldpositions, npositions * 3, batch);

	cudaMemcpy(d_finalpositions, d_oldpositions, npositions * batch * sizeof(float3), cudaMemcpyDeviceToDevice);

	cudaFree(d_newpositions);
	cudaFree(d_oldpositions);
}

template<bool useactuators> __global__ void ParticleSoftBodyDeformKernel (float3* d_initialpositions,
																		float3* d_oldpositions,
																		float3* d_newpositions,
																		ushort npositions,
																		int* d_neighborids,
																		int* d_neighboroffsets,
																		float* d_edgelengths,
																		int* d_connectedactuator,
																		float3* d_actuatordeltas,
																		ushort nactuators,
																		ushort batch,
																		float3* d_debug)
{
    ushort n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= npositions)
		return;

	d_oldpositions += npositions * blockIdx.y;
	d_newpositions += npositions * blockIdx.y;
	d_actuatordeltas += nactuators * blockIdx.y;
	//d_debug += npositions * blockIdx.y;
		
	int neighborsstart = d_neighboroffsets[n];
	d_neighborids += neighborsstart;
	d_edgelengths += neighborsstart;
	ushort nneighbors = (ushort)(d_neighboroffsets[n + 1] - neighborsstart);

	short connectedactuator = d_connectedactuator[n];

	float3 pos1 = d_oldpositions[n];
	float3 force = make_float3(0);

	for (ushort i = 0; i < nneighbors; i++)
	{
		int n2 = d_neighborids[i];
		float3 pos2 = d_oldpositions[n2];

		float orilength = d_edgelengths[i];
        float3 diff = pos1 - pos2;
        float curlength = sqrt(dotp(diff, diff));

        float delta = (orilength - curlength) * 0.1f;
        float3 diffnorm;
        if (curlength > 0)
            diffnorm = diff / curlength;
        else
            diffnorm = make_float3(1e-3f, 0, 0);
            
		force += diffnorm * delta;
	}

	if (useactuators && connectedactuator >= 0)
	{
		float3 actuatorpos = d_initialpositions[n] + d_actuatordeltas[connectedactuator];
        float3 diff = actuatorpos - pos1;
        float curlength = sqrt(dotp(diff, diff));

        float delta = curlength * 0.2f;
        float3 diffnorm = make_float3(0);
        if (curlength > 0)
            diffnorm = diff / curlength;

        force += diffnorm * delta;
	}

	d_newpositions[n] = pos1 + force;
	//d_debug[n] = force;
	//d_debug[n].x = connectedactuator;
}