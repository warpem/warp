#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/IO.cuh"


namespace gtom
{
	void ReadRAW(std::string path, void** data, EM_DATATYPE datatype, int3 dims, size_t headerbytes, int nframe)
	{
		FILE* inputfile = fopen(path.c_str(), "rb");
#if _MSC_VER > 1
		_fseeki64(inputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(inputfile, 0L, SEEK_SET);
#endif

		size_t bytesperfield = EM_DATATYPE_SIZE[(int)datatype];

		size_t datasize = Elements(dims) * bytesperfield;
		cudaMallocHost(data, datasize);

		if (nframe >= 0)
#if _MSC_VER > 1
			_fseeki64(inputfile, headerbytes + datasize * (size_t)nframe, SEEK_CUR);
#elif __GNUC__ > 3
			fseeko64(inputfile, headerbytes + datasize * (size_t)nframe, SEEK_CUR);
#endif

		fread(*data, sizeof(char), datasize, inputfile);

		fclose(inputfile);
	}
}