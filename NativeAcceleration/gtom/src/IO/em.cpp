#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/IO.cuh"

namespace gtom
{
	void ReadEM(std::string path, void** data, int nframe)
	{
		FILE* inputfile = fopen(path.c_str(), "rb");
#if _MSC_VER > 1
		_fseeki64(inputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(inputfile, 0L, SEEK_SET);
#endif

		HeaderEM header = ReadEMHeader(inputfile);

		size_t datasize;
		if (nframe >= 0)
			datasize = Elements2(header.dimensions) * EM_DATATYPE_SIZE[(int)header.mode];
		else
			datasize = Elements(header.dimensions) * EM_DATATYPE_SIZE[(int)header.mode];

		cudaMallocHost(data, datasize);

		if (nframe >= 0)
#if _MSC_VER > 1
			_fseeki64(inputfile, datasize * (size_t)nframe, SEEK_CUR);
#elif __GNUC__ > 3
			fseeko64(inputfile, datasize * (size_t)nframe, SEEK_CUR);
#endif

		fread(*data, sizeof(char), datasize, inputfile);

		fclose(inputfile);
	}

	HeaderEM ReadEMHeader(std::string path)
	{
		FILE* inputfile = fopen(path.c_str(), "rb");
#if _MSC_VER > 1
		_fseeki64(inputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(inputfile, 0L, SEEK_SET);
#endif

		HeaderEM header = ReadEMHeader(inputfile);
		fclose(inputfile);

		return header;
	}

	HeaderEM ReadEMHeader(FILE* inputfile)
	{
		HeaderEM header;
		char* headerp = (char*)&header;

		fread(headerp, sizeof(char), sizeof(HeaderEM), inputfile);

		return header;
	}
}