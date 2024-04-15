#include "include/Functions.h"
#include "liblion/liblion.h"
using namespace gtom;

__declspec(dllexport) void __stdcall ConicalFSC(float2* volume1ft, float2* volume2ft, int3 dims, float3* directions, int ndirections, float anglestep, int minshell, float threshold, float particlefraction, float* result)
{
	int nshells = dims.x / 2;

	float conesigma2 = anglestep / 2 / 180 * PI;
	conesigma2 = 2.0f * conesigma2 * conesigma2;

	float dotcutoff = cos(anglestep * 1.0f / 180.0f * PI);

	float t = -threshold / (threshold - 1);
	float4 thresholdparts = make_float4(t,
										2 * sqrt(t) + 1,
										t + 1,
										2 * sqrt(t));
	
	float* h_nums = MallocValueFilled(ndirections * nshells, (float)0);
	float* h_denoms1 = MallocValueFilled(ndirections * nshells, (float)0);
	float* h_denoms2 = MallocValueFilled(ndirections * nshells, (float)0);
	float* h_weightsums = MallocValueFilled(ndirections * nshells, (float)0);
	
	for (int z = 0; z < dims.z; z++)
	{
		int zz = z < dims.z / 2 + 1 ? z : z - dims.z;
		float zz2 = zz * zz;

		for (int y = 0; y < dims.y; y++)
		{
			int yy = y < dims.y / 2 + 1 ? y : y - dims.y;
			float yy2 = yy * yy;

			for (int x = 0; x < dims.x / 2 + 1; x++)
			{
				int xx = x;
				float xx2 = xx * xx;

				float r = sqrt(zz2 + yy2 + xx2);
				int ri = (int)(r + 0.5f);
				if (ri >= nshells)
					continue;

				float3 dir = make_float3(xx / r, yy / r, zz / r);

				tcomplex val1 = volume1ft[(z * dims.y + y) * (dims.x / 2 + 1) + x];
				tcomplex val2 = volume2ft[(z * dims.y + y) * (dims.x / 2 + 1) + x];

				float num = dotp2(val1, val2);
				float denom1 = dotp2(val1, val1);
				float denom2 = dotp2(val2, val2);

				#pragma omp parallel for
				for (int a = 0; a < ndirections; a++)
				{
					float dot = abs(dotp(dir, directions[a]));
					if (dot < dotcutoff)
						continue;

					float anglediff = acos(tmin(1, dot));

					float weight = 1;// exp(-(anglediff * anglediff) / conesigma2);

					h_nums[a * nshells + ri] += num * weight;
					h_denoms1[a * nshells + ri] += denom1 * weight;
					h_denoms2[a * nshells + ri] += denom2 * weight;
					h_weightsums[a * nshells + ri] += weight;
				}
			}
		}
	}

	for (int a = 0; a < ndirections; a++)
	{
		bool foundlimit = false;
		for (int s = 0; s < nshells; s++)
		{
			int i = a * nshells + s;

			float weightsum = sqrt(h_weightsums[i] * particlefraction);
			if (s < minshell || weightsum < 6.0f)
				h_nums[i] = 1;
			else
				h_nums[i] = h_nums[i] / sqrt(h_denoms1[i] * h_denoms2[i]);

			float currentthreshold = tmin(0.95f, (thresholdparts.x + thresholdparts.y / weightsum) / (thresholdparts.z + thresholdparts.w / weightsum));
			//currentthreshold = threshold;

			if (!foundlimit && h_nums[i] < currentthreshold)
			{
				foundlimit = true;
				float current = h_nums[i - 1];
				float next = h_nums[i];

				result[a] = tmax(1, (float)(s - 1) + tmax(tmin((currentthreshold - current) / (next - current + 1e-5f), 1.0f), 0.0f));
			}
		}

		if (!foundlimit)
			result[a] = nshells - 1;
	}

	free(h_nums);
	free(h_denoms1);
	free(h_denoms2);
	free(h_weightsums);
}