#include "include/Functions.h"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/DeviceFunctions.cuh"
using namespace gtom;

__global__ void HelicalSymmetrizeKernel(cudaTex tcpf_volume,
										float* d_output,
										int3 dims,
										float twist,
										float rise,
										float maxz,
										float maxr);

__declspec(dllexport) void HelicalSymmetrize(unsigned long long tcpf_volume,
											float* d_output,
											int3 dims,
											float twist,
											float rise,
											float maxz,
											float maxr)
{

	dim3 grid = dim3((dims.x + 127) / 128, dims.y, dims.z);
	HelicalSymmetrizeKernel << <grid, 128 >> > (tcpf_volume,
												d_output,
												dims,
												twist,
												rise,
												maxz,
												maxr);
}

__global__ void HelicalSymmetrizeKernel(cudaTex tcpf_volume,
										float* d_output,
										int3 dims,
										float twist,
										float rise,
										float maxz,
										float maxr)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= dims.x)
		return;

	float sum = 0;
	int samples = 0;

	glm::vec3 dimshalf = glm::vec3(dims.x, dims.y, dims.z) * 0.5f;
	glm::vec3 pos0 = glm::vec3(x, blockIdx.y, blockIdx.z) - dimshalf;

	if (glm::length(glm::vec2(pos0.x, pos0.y)) > maxr)
	{
		d_output[(blockIdx.z * dims.y + blockIdx.y) * dims.x + x] = 0;
		return;
	}

	int steps = 0;
	while (true && steps < 9999)
	{
		glm::mat2 rotation = d_Matrix2Rotation(twist * steps);
		glm::vec2 posxy = rotation * glm::vec2(pos0.x, pos0.y);
		glm::vec3 pos = glm::vec3(posxy.x, posxy.y, pos0.z + rise * steps) + dimshalf;

		if (pos.z < 0 || pos.z >= dims.z)
			break;

		if (abs(pos.z - dimshalf.z) <= maxz)
		{
			float val = cubicTex3D(tcpf_volume, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);
			sum += val;
			samples++;
		}

		steps++;
	}
	
	steps = 1;
	while (true && steps < 9999)
	{
		glm::mat2 rotation = d_Matrix2Rotation(-twist * steps);
		glm::vec2 posxy = rotation * glm::vec2(pos0.x, pos0.y);
		glm::vec3 pos = glm::vec3(posxy.x, posxy.y, pos0.z - rise * steps) + dimshalf;

		if (pos.z < 0 || pos.z >= dims.z)
			break;

		if (abs(pos.z - dimshalf.z) <= maxz)
		{
			float val = cubicTex3D(tcpf_volume, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);
			sum += val;
			samples++;
		}

		steps++;
	}

	if (samples > 0)
		sum /= samples;

	d_output[(blockIdx.z * dims.y + blockIdx.y) * dims.x + x] = sum;
}