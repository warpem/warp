#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/IO.cuh"

namespace gtom
{
	void ReadMRC(std::string path, void** data, int nframe)
	{
		FILE* inputfile = fopen(path.c_str(), "rb");
#if _MSC_VER > 1
		_fseeki64(inputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(inputfile, 0L, SEEK_SET);
#endif

		HeaderMRC header = ReadMRCHeader(inputfile);

		size_t datasize;
		if (nframe >= 0)
			datasize = Elements2(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];
		else
			datasize = Elements(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];

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

	HeaderMRC ReadMRCHeader(std::string path)
	{
		FILE* inputfile = fopen(path.c_str(), "rb");
#if _MSC_VER > 1
		_fseeki64(inputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(inputfile, 0L, SEEK_SET);
#endif

		HeaderMRC header = ReadMRCHeader(inputfile);
		fclose(inputfile);

		return header;
	}

	HeaderMRC ReadMRCHeader(FILE* inputfile)
	{
		HeaderMRC header;
		char* headerp = (char*)&header;

		fread(headerp, sizeof(char), sizeof(HeaderMRC), inputfile);
#if _MSC_VER > 1
		_fseeki64(inputfile, (long)header.extendedbytes, SEEK_CUR);
#elif __GNUC__ > 3
		fseeko64(inputfile, (long)header.extendedbytes, SEEK_CUR);
#endif

		return header;
	}

	void WriteMRC(void* data, HeaderMRC header, std::string path)
	{
		FILE* outputfile = fopen(path.c_str(), "wb");
#if _MSC_VER > 1
		_fseeki64(outputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(outputfile, 0L, SEEK_SET);
#endif

		fwrite(&header, sizeof(HeaderMRC), 1, outputfile);

		size_t elementsize = MRC_DATATYPE_SIZE[(int)header.mode];
		fwrite(data, elementsize, Elements(header.dimensions), outputfile);

		fclose(outputfile);
	}

	void WriteMRC(tfloat* data, int3 dims, std::string path)
	{
		HeaderMRC header;
		header.dimensions = dims;
#ifdef GTOM_DOUBLE
		throw;	// MRC can't do double!
#else
		header.mode = MRC_FLOAT;
#endif

		tfloat minval = 1e30f, maxval = -1e30f;
		size_t elements = Elements(dims);
		for (size_t i = 0; i < elements; i++)
		{
			minval = tmin(minval, data[i]);
			maxval = tmax(maxval, data[i]);
		}
		header.maxvalue = maxval;
		header.minvalue = minval;

		WriteMRC(data, header, path);
	}

	void d_WriteMRC(tfloat* d_data, int3 dims, std::string path)
	{
		tfloat* h_data = (tfloat*)MallocFromDeviceArray(d_data, Elements(dims) * sizeof(tfloat));

		WriteMRC(h_data, dims, path);

		free(h_data);
	}

	void d_WriteMRC(half* d_data, int3 dims, std::string path)
	{
		tfloat* d_data32;
		cudaMalloc((void**)&d_data32, Elements(dims) * sizeof(tfloat));
		d_ConvertToTFloat(d_data, d_data32, Elements(dims));

		tfloat* h_data = (tfloat*)MallocFromDeviceArray(d_data32, Elements(dims) * sizeof(tfloat));
		cudaFree(d_data32);

		WriteMRC(h_data, dims, path);

		free(h_data);
	}
}