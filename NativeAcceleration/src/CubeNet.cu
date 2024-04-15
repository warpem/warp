#include "include/Functions.h"
#include "gtom/include/DeviceFunctions.cuh"
using namespace gtom;

__global__ void CubeNetAugmentKernel(cudaTex t_inputvol,
									float* d_inputlabel,
									int3 dimsinput,
									float* d_outputmic,
									float* d_outputlabel,
									int nclasses,
									int3 dimsoutput,
									float3* d_offsets,
									glm::mat3* d_transforms);

__declspec(dllexport) void CubeNetAugment(float* d_inputvol,
										float* d_inputlabel,
										int3 dimsinput,
										float* d_outputmic,
										float* d_outputlabel,
										int nclasses,
										int3 dimsoutput,
										float3* h_offsets,
										float3* h_rotations,
										float3* h_scales,
										float noisestddev,
										int seed,
										uint batch)
{
	cudaArray_t a_inputvol;
	cudaTex t_inputvol;

	{
		d_BindTextureTo3DArray(d_inputvol, a_inputvol, t_inputvol, dimsinput, cudaFilterModePoint, false);
	}

	glm::mat3* h_transforms = (glm::mat3*)malloc(batch * sizeof(glm::mat3));
	for (int i = 0; i < batch; i++)
		h_transforms[i] = Matrix3Euler(h_rotations[i]);

	glm::mat3* d_transforms = (glm::mat3*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat3));
	free(h_transforms);

	float3* d_offsets = (float3*)CudaMallocFromHostArray(h_offsets, batch * sizeof(float3));

	dim3 grid = dim3(tmin(32768, (Elements(dimsoutput) + 127) / 128), batch, 1);
	CubeNetAugmentKernel << <grid, 128 >> > (t_inputvol, d_inputlabel, dimsinput, d_outputmic, d_outputlabel, nclasses, dimsoutput, d_offsets, d_transforms);

	cudaFree(d_offsets);
	cudaFree(d_transforms);

	{
		cudaDestroyTextureObject(t_inputvol);
		cudaFreeArray(a_inputvol);
	}

	//float* d_noise = CudaMallocRandomFilled(Elements(dimsoutput) * batch, 0, noisestddev, seed);
	//d_AddVector(d_outputmic, d_noise, d_outputmic, Elements(dimsoutput) * batch);

	//cudaFree(d_noise);

	//d_NormMonolithic(d_outputmic, d_outputmic, Elements(dimsoutput), T_NORM_MEAN01STD, batch);
}

__global__ void CubeNetAugmentKernel(cudaTex t_inputvol,
									float* d_inputlabel,
									int3 dimsinput,
									float* d_outputmic,
									float* d_outputlabel,
									int nclasses,
									int3 dimsoutput,
									float3* d_offsets,
									glm::mat3* d_transforms)
{
	d_outputmic += Elements(dimsoutput) * blockIdx.y;
	d_outputlabel += Elements(dimsoutput) * nclasses * blockIdx.y;

	int3 outputcenter = dimsoutput / 2;

	glm::mat3 transform = d_transforms[blockIdx.y];
	float3 offset = d_offsets[blockIdx.y];

	uint elements = Elements(dimsoutput);
	uint elementsslice = Elements2(dimsoutput);

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
	{
		uint idz = id / elementsslice;
		uint idy = (id - idz * elementsslice) / (uint)dimsoutput.x;
		uint idx = id % (uint)dimsoutput.x;

		float val = 0;
		int label = 0;

		int posx = (int)idx - outputcenter.x;
		int posy = (int)idy - outputcenter.y;
		int posz = (int)idz - outputcenter.z;

		glm::vec3 pos = transform * glm::vec3(posx, posy, posz);

		pos.x += offset.x;
		pos.y += offset.y;
		pos.z += offset.z;


		if (pos.x > 0 && pos.y > 0 && pos.z > 0 && pos.x < dimsinput.x - 1 && pos.y < dimsinput.y - 1 && pos.z < dimsinput.z - 1)
		{
			for (int z = -4; z <= 4; z++)
			{
				float zz = floor(pos.z) + z;
				float sincz = sinc(pos.z - zz);
				float zz2 = pos.z - zz;
				zz2 *= zz2;
				zz += 0.5f;

				for (int y = -4; y <= 4; y++)
				{
					float yy = floor(pos.y) + y;
					float sincy = sinc(pos.y - yy);
					float yy2 = pos.y - yy;
					yy2 *= yy2;
					yy += 0.5f;

					for (int x = -4; x <= 4; x++)
					{
						float xx = floor(pos.x) + x;
						float sincx = sinc(pos.x - xx);
						float xx2 = pos.x - xx;
						xx2 *= xx2;
						float r2 = xx2 + yy2 + zz2;

						if (r2 > 16)
							continue;

						val += tex3D<float>(t_inputvol, xx + 0.5f, yy, zz) * sincz * sincy * sincx;
					}
				}
			}

			label = (int)d_inputlabel[((int)(pos.z + 0.5f) * dimsinput.y + (int)(pos.y + 0.5f)) * dimsinput.x + (int)(pos.x + 0.5f)];
		}

		d_outputmic[id] = val;
		d_outputlabel[label * Elements(dimsoutput) + id] = 1;
	}
}