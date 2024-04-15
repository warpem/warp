#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"

namespace gtom
{
	////////////////////////////////////////////////////////////////////////
	//Packs the input stack into a texture atlas with quadratic dimensions//
	////////////////////////////////////////////////////////////////////////

	template <class T> T* d_MakeAtlas(T* d_input, int3 inputdims, int3 &outputdims, int2 &primitivesperdim, int2* h_primitivecoords)
	{
		int sidelength = NextPow2((size_t)ceil(sqrt((tfloat)inputdims.z)) * (size_t)inputdims.x);
		outputdims.x = max(16, sidelength);
		outputdims.y = sidelength;
		outputdims.z = 1;
		int atlascolumns = outputdims.x / inputdims.x;
		int atlasrows = (inputdims.z + atlascolumns - 1) / atlascolumns;
		primitivesperdim.x = atlascolumns;
		primitivesperdim.y = atlasrows;

		T* d_atlas = CudaMallocValueFilled(Elements(outputdims), (T)0);
		int3 primitivedims = toInt3(inputdims.x, inputdims.y, 1);

		for (int b = 0; b < inputdims.z; b++)
		{
			int offsetx = (b % atlascolumns) * primitivedims.x;
			int offsety = (b / atlascolumns) * primitivedims.y;
			h_primitivecoords[b] = toInt2(offsetx, offsety);
			for (int y = 0; y < primitivedims.y; y++)
				cudaMemcpy(d_atlas + (offsety + y) * outputdims.x + offsetx, d_input + b * Elements(primitivedims) + y * primitivedims.x, primitivedims.x * sizeof(T), cudaMemcpyDeviceToDevice);
		}

		return d_atlas;
	}
	template bool* d_MakeAtlas<bool>(bool* d_input, int3 inputdims, int3 &outputdims, int2 &primitivesperdim, int2* h_primitivecoords);
	template int* d_MakeAtlas<int>(int* d_input, int3 inputdims, int3 &outputdims, int2 &primitivesperdim, int2* h_primitivecoords);
	template float* d_MakeAtlas<float>(float* d_input, int3 inputdims, int3 &outputdims, int2 &primitivesperdim, int2* h_primitivecoords);
	template double* d_MakeAtlas<double>(double* d_input, int3 inputdims, int3& outputdims, int2 &primitivesperdim, int2* h_primitivecoords);
}