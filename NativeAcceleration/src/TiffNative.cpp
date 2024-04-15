#include "include/Functions.h"
#include "liblion/liblion.h"
#include "tiffio.h"

using namespace gtom;

// Adapted from RELION's https://github.com/3dem/relion/blob/master/src/rwTIFF.h

__declspec(dllexport) void ReadTIFF(const char* path, int layer, bool flipy, float* h_result)
{
	TIFF* ftiff = TIFFOpen(path, "rm");
	
	// libtiff's types
	uint32 width, length;

	if (TIFFGetField(ftiff, TIFFTAG_IMAGEWIDTH, &width) != 1 ||
		TIFFGetField(ftiff, TIFFTAG_IMAGELENGTH, &length) != 1)
	{
		throw std::runtime_error("The input TIFF file does not have the width or height field.");
	}

	int3 dims = make_int3(width, length, TIFFNumberOfDirectories(ftiff));

	if (layer >= dims.z)
		throw std::runtime_error("Requested layer exceeds the stack size.");

	if (layer >= 0)
		dims.z = 1;

	uint16 sampleFormat, bitsPerSample;
	TIFFGetFieldDefaulted(ftiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
	TIFFGetFieldDefaulted(ftiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat);

	TIFFSetDirectory(ftiff, 0);
	
	relion::DataType datatype;

	if (bitsPerSample == 8 && sampleFormat == 1) {
		datatype = relion::UChar;
	}
	else if (bitsPerSample == 8 && sampleFormat == 2) {
		datatype = relion::SChar;
	}
	else if (bitsPerSample == 16 && sampleFormat == 1) {
		datatype = relion::UShort;
	}
	else if (bitsPerSample == 16 && sampleFormat == 2) {
		datatype = relion::Short;
	}
	else if (bitsPerSample == 32 && sampleFormat == 3) {
		datatype = relion::Float;
	}
	else {
		throw std::runtime_error("Unsupported TIFF format");
	}

	float* h_tempstrip = (float*)malloc(dims.x * sizeof(float));

	//dims.z = 20;

	for (int z = 0; z < dims.z; z++) 
	{
		TIFFSetDirectory(ftiff, layer < 0 ? z : layer);

		float* h_resultlayer = h_result + Elements2(dims) * z;
		
		tsize_t stripSize = TIFFStripSize(ftiff);
		tstrip_t numberOfStrips = TIFFNumberOfStrips(ftiff);
		tdata_t buf = _TIFFmalloc(stripSize);
	
		size_t haveread_n = 0;

		for (tstrip_t strip = 0; strip < numberOfStrips; strip++) 
		{
			tsize_t actually_read = TIFFReadEncodedStrip(ftiff, strip, buf, stripSize);
			tsize_t actually_read_n = actually_read * 8 / bitsPerSample;

			tsize_t actually_read_n_8 = actually_read_n / 8 * 8;

			if (datatype == relion::UChar)
			{
				for (tsize_t i = 0; i < actually_read_n_8; i += 8)
				{
					h_resultlayer[haveread_n + i] = (float)((uchar*)buf)[i];
					h_resultlayer[haveread_n + i + 1] = (float)((uchar*)buf)[i + 1];
					h_resultlayer[haveread_n + i + 2] = (float)((uchar*)buf)[i + 2];
					h_resultlayer[haveread_n + i + 3] = (float)((uchar*)buf)[i + 3];
					h_resultlayer[haveread_n + i + 4] = (float)((uchar*)buf)[i + 4];
					h_resultlayer[haveread_n + i + 5] = (float)((uchar*)buf)[i + 5];
					h_resultlayer[haveread_n + i + 6] = (float)((uchar*)buf)[i + 6];
					h_resultlayer[haveread_n + i + 7] = (float)((uchar*)buf)[i + 7];
				}
				for (tsize_t i = actually_read_n_8; i < actually_read_n; i++)
					h_resultlayer[haveread_n + i] = (float)((uchar*)buf)[i];
			}
			else if (datatype == relion::SChar)
			{
				for (tsize_t i = 0; i < actually_read_n_8; i += 8)
				{
					h_resultlayer[haveread_n + i] = (float)((char*)buf)[i];
					h_resultlayer[haveread_n + i + 1] = (float)((char*)buf)[i + 1];
					h_resultlayer[haveread_n + i + 2] = (float)((char*)buf)[i + 2];
					h_resultlayer[haveread_n + i + 3] = (float)((char*)buf)[i + 3];
					h_resultlayer[haveread_n + i + 4] = (float)((char*)buf)[i + 4];
					h_resultlayer[haveread_n + i + 5] = (float)((char*)buf)[i + 5];
					h_resultlayer[haveread_n + i + 6] = (float)((char*)buf)[i + 6];
					h_resultlayer[haveread_n + i + 7] = (float)((char*)buf)[i + 7];
				}
				for (tsize_t i = actually_read_n_8; i < actually_read_n; i++)
					h_resultlayer[haveread_n + i] = (float)((char*)buf)[i];
			}
			else if (datatype == relion::UShort)
				for (tsize_t i = 0; i < actually_read_n; i++)
					h_resultlayer[haveread_n + i] = (float)((ushort*)buf)[i];
			else if (datatype == relion::Short)
				for (tsize_t i = 0; i < actually_read_n; i++)
					h_resultlayer[haveread_n + i] = (float)((short*)buf)[i];
			else if (datatype == relion::Float)
				memcpy(h_resultlayer + haveread_n, buf, actually_read);
			
			haveread_n += actually_read_n;
		}

		_TIFFfree(buf);

		// Flip the Y axis if requested

		if (flipy)
		{
			for (tsize_t y1 = 0; y1 < dims.y / 2; y1++)
			{
				tsize_t y2 = dims.y - 1 - y1;

				memcpy(h_tempstrip, h_resultlayer + y1 * dims.x, dims.x * sizeof(float));

				memcpy(h_resultlayer + y1 * dims.x, h_resultlayer + y2 * dims.x, dims.x * sizeof(float));

				memcpy(h_resultlayer + y2 * dims.x, h_tempstrip, dims.x * sizeof(float));
			}
		}
	}

	free(h_tempstrip);

	TIFFClose(ftiff);
}