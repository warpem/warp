#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"

namespace gtom
{
	int3* GetEqualGridSpacing(int2 dimsimage, int2 dimsregion, float overlapfraction, int2 &dimsgrid)
	{
		int2 dimsoverlap = toInt2((float)dimsregion.x * (1.0f - overlapfraction), (float)dimsregion.y * (1.0f - overlapfraction));
		dimsgrid = toInt2(NextMultipleOf(dimsimage.x - (dimsregion.x - dimsoverlap.x), dimsoverlap.x) / dimsoverlap.x, NextMultipleOf(dimsimage.y - (dimsregion.y - dimsoverlap.y), dimsoverlap.y) / dimsoverlap.y);

		int2 shift;
		shift.x = dimsgrid.x > 1 ? (tfloat)(dimsimage.x - dimsregion.x) / (tfloat)(dimsgrid.x - 1) : (dimsimage.x - dimsregion.x) / 2;
		shift.y = dimsgrid.y > 1 ? (tfloat)(dimsimage.y - dimsregion.y) / (tfloat)(dimsgrid.y - 1) : (dimsimage.y - dimsregion.y) / 2;
		int2 offset = toInt2((dimsimage.x - shift.x * (dimsgrid.x - 1) - dimsregion.x) / 2,
			(dimsimage.y - shift.y * (dimsgrid.y - 1) - dimsregion.y) / 2);

		int3* h_origins = (int3*)malloc(Elements2(dimsgrid) * sizeof(int3));

		for (int y = 0; y < dimsgrid.y; y++)
			for (int x = 0; x < dimsgrid.x; x++)
				h_origins[y * dimsgrid.x + x] = toInt3(x * shift.x + offset.x, y * shift.y + offset.y, 0);

		return h_origins;
	}
}