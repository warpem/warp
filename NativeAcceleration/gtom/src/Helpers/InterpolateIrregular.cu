#include "gtom/include/Prerequisites.cuh"


namespace gtom
{
	void Interpolate1DOntoGrid(std::vector<tfloat2> sortedpoints, tfloat* h_output, uint gridstart, uint gridend)
	{
		// Nearest neighbor extrapolation
		if (sortedpoints[0].x > gridstart)
		{
			tfloat val = sortedpoints[0].y;
			for (uint i = gridstart; i < (uint)sortedpoints[0].x; i++)
				h_output[i - gridstart] = val;
		}
		if (sortedpoints[sortedpoints.size() - 1].x < gridend)
		{
			tfloat val = sortedpoints[sortedpoints.size() - 1].y;
			for (uint i = (uint)sortedpoints[sortedpoints.size() - 1].x; i <= gridend; i++)
				h_output[i - gridstart] = val;
			gridend = sortedpoints[sortedpoints.size() - 1].x;
		}

		// Interpolated interval not within any two points
		if (gridstart >= gridend)
			return;

		// Cubic interpolation
		tfloat2 samples[4];
		int p1 = 0;
		samples[0] = sortedpoints[max(0, p1 - 1)];
		samples[1] = sortedpoints[p1];
		samples[2] = sortedpoints[min((int)sortedpoints.size() - 1, p1 + 1)];
		samples[3] = sortedpoints[min((int)sortedpoints.size() - 1, p1 + 2)];
		for (int i = max(gridstart, (int)sortedpoints[0].x); i <= gridend; i++)
		{
			while (i > sortedpoints[min((int)sortedpoints.size() - 1, p1 + 1)].x && p1 < sortedpoints.size() - 1)
			{
				p1++;
				samples[0] = sortedpoints[max(0, p1 - 1)];
				samples[1] = sortedpoints[p1];
				samples[2] = sortedpoints[min((int)sortedpoints.size() - 1, p1 + 1)];
				samples[3] = sortedpoints[min((int)sortedpoints.size() - 1, p1 + 2)];
			}

			tfloat interp = ((tfloat)i - samples[1].x) / max(1.0, samples[2].x - samples[1].x);
			//h_output[i - gridstart] = ((factors[0] * interp + factors[1]) * interp + factors[2]) * interp + factors[3];
			h_output[i - gridstart] = samples[1].y + (samples[2].y - samples[1].y) * interp;
		}
	}
}