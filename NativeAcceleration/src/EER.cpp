#include "include/Functions.h"
#include "tiffio.h"

using namespace gtom;

// Adapted from RELION's https://github.com/3dem/relion/blob/devel-eer/src/renderEER.*


const char EER_FOOTER_OK[] = "ThermoFisherECComprOK000";
const char EER_FOOTER_ERR[] = "ThermoFisherECComprERR00";
const int EER_IMAGE_WIDTH = 4096;
const int EER_IMAGE_HEIGHT = 4096;
const int EER_IMAGE_PIXELS = EER_IMAGE_WIDTH * EER_IMAGE_HEIGHT;
const unsigned int EER_LEN_FOOTER = 24;
const uint16_t TIFF_COMPRESSION_EER8bit = 65000;
const uint16_t TIFF_COMPRESSION_EER7bit = 65001;


void render16K(float* image, std::vector<unsigned int>& positions, std::vector<unsigned char>& symbols, int n_electrons)
{
	for (int i = 0; i < n_electrons; i++)
	{
		int x = ((positions[i] & 4095) << 2) | (symbols[i] & 3); // 4095 = 111111111111b, 3 = 00000011b
		int y = ((positions[i] >> 12) << 2) | ((symbols[i] & 12) >> 2); //  4096 = 2^12, 12 = 00001100b
		image[(y << 14) + x]++;
	}
}
void render8K(float* image, std::vector<unsigned int>& positions, std::vector<unsigned char>& symbols, int n_electrons)
{
	for (int i = 0; i < n_electrons; i++)
	{
		int x = ((positions[i] & 4095) << 1) | ((symbols[i] & 2) >> 1); // 4095 = 111111111111b, 2 = 00000010b
		int y = ((positions[i] >> 12) << 1) | ((symbols[i] & 8) >> 3); //  4096 = 2^12, 8 = 00001000b
		image[(y << 13) + x]++;
	}
}
void render4K(float* image, std::vector<unsigned int>& positions, std::vector<unsigned char>& symbols, int n_electrons)
{
	for (int i = 0; i < n_electrons; i++)
	{
		int x = positions[i] & 4095; // 4095 = 111111111111b
		int y = positions[i] >> 12; //  4096 = 2^12
		image[(y << 12) + x]++;
	}
}


// image is cleared.
// This function is thread-safe (except for timing).
long long renderFrames(int frame_start, int frame_end, float* image);	

__declspec(dllexport) void ReadEERCombinedFrame(const char* path, int firstFrameInclusive, int lastFrameExclusive, int eer_upsampling, float* h_result)
{
	TIFFSetWarningHandler(0);

	if (eer_upsampling < 1 || eer_upsampling > 3)
		throw("EERRenderer::read: eer_upsampling must be 1, 2 or 3.");
	
	// First of all, check the file size
	FILE* fh = fopen(path, "r");
	if (fh == NULL)
		throw std::runtime_error("Failed to open file");

	fseek(fh, 0, SEEK_END);
	long long file_size = ftell(fh);
	fseek(fh, 0, SEEK_SET);
	fclose(fh);

	int nframes = 0;
	bool is_7bit = false;
	
	// Try reading as TIFF; this handle is kept open
	TIFF* ftiff = TIFFOpen(path, "rm");

	if (ftiff == NULL)
	{
		throw std::runtime_error("Legacy mode not implemented");
	}
	else
	{
		// Check width & size
		int width, height;
		uint16_t compression = 0;
		TIFFGetField(ftiff, TIFFTAG_IMAGEWIDTH, &width);
		TIFFGetField(ftiff, TIFFTAG_IMAGELENGTH, &height);
		TIFFGetField(ftiff, TIFFTAG_COMPRESSION, &compression);

		// TIA can write an EER file whose first page is a sum and compressoin == 1.
		// This is not supported (yet). EPU never writes such movies.
		if (compression == TIFF_COMPRESSION_EER8bit)
			is_7bit = false;
		else if (compression == TIFF_COMPRESSION_EER7bit)
			is_7bit = true;
		else
			throw std::runtime_error("Unknown compression scheme for EER");

		if (width != EER_IMAGE_WIDTH || height != EER_IMAGE_HEIGHT)
			throw std::runtime_error("Currently we support only 4096x4096 pixel EER movies.");

		// Find the number of frames
		nframes = TIFFNumberOfDirectories(ftiff);
	}
	
	std::vector<long long> frame_starts, frame_sizes;
	unsigned char* buf;

	{
		frame_starts.resize(nframes, 0);
		frame_sizes.resize(nframes, 0);
		buf = (unsigned char*)malloc(file_size); // This is big enough
		if (buf == NULL)
			throw std::runtime_error("Failed to allocate buffer");
		long long pos = 0;

		// Read everything
		for (int frame = firstFrameInclusive; frame < lastFrameExclusive; frame++)
		{
			TIFFSetDirectory(ftiff, frame);
			const int nstrips = TIFFNumberOfStrips(ftiff);
			frame_starts[frame] = pos;

			for (int strip = 0; strip < nstrips; strip++)
			{
				const int strip_size = TIFFRawStripSize(ftiff, strip);
				if (pos + strip_size >= file_size)
					throw std::runtime_error("EER: buffer overflow when reading raw strips.");

				TIFFReadRawStrip(ftiff, strip, buf + pos, strip_size);
				pos += strip_size;
				frame_sizes[frame] += strip_size;
			}
		}

		TIFFClose(ftiff);
	}

	{
		long long total_n_electron = 0;

		long long supersize = 4096 << (eer_upsampling - 1);

		std::vector<unsigned int> positions;
		std::vector<unsigned char> symbols;
		memset(h_result, 0, supersize * supersize * sizeof(float));

		for (int iframe = firstFrameInclusive; iframe < lastFrameExclusive; iframe++)
		{
			long long pos = frame_starts[iframe];
			unsigned int n_pix = 0, n_electron = 0;
			const int max_electrons = frame_sizes[iframe] * 2; // at 4 bits per electron (very permissive bound!)
			if (positions.size() < max_electrons)
			{
				positions.resize(max_electrons);
				symbols.resize(max_electrons);
			}

			if (is_7bit)
			{
				unsigned int bit_pos = 0; // 4 K * 4 K * 11 bit << 2 ** 32
				unsigned char p, s;

				while (true)
				{
					// Fetch 32 bits and unpack up to 2 chunks of 7 + 4 bits.
					// This is faster than unpack 7 and 4 bits sequentially.
					// Since the size of buf is larger than the actual size by the TIFF header size,
					// it is always safe to read ahead.

					long long first_byte = pos + (bit_pos >> 3);
					const unsigned int bit_offset_in_first_byte = bit_pos & 7; // 7 = 00000111 (same as % 8)
					const unsigned int chunk = (*(unsigned int*)(buf + first_byte)) >> bit_offset_in_first_byte;

					p = (unsigned char)(chunk & 127); // 127 = 01111111
					bit_pos += 7; // TODO: we can remove this for further speed.
					n_pix += p;
					if (n_pix == EER_IMAGE_PIXELS) break;
					if (p == 127) continue; // this should be rare.

					s = (unsigned char)((chunk >> 7) & 15) ^ 0x0A; // 15 = 00001111; See below for 0x0A
					bit_pos += 4;
					positions[n_electron] = n_pix;
					symbols[n_electron] = s;
					n_electron++;
					n_pix++;

					p = (unsigned char)((chunk >> 11) & 127); // 127 = 01111111
					bit_pos += 7;
					n_pix += p;
					if (n_pix == EER_IMAGE_PIXELS) break;
					if (p == 127) continue;

					s = (unsigned char)((chunk >> 18) & 15) ^ 0x0A; // 15 = 00001111; See below for 0x0A
					bit_pos += 4;
					positions[n_electron] = n_pix;
					symbols[n_electron] = s;
					n_electron++;
					n_pix++;

					if (n_electron >= max_electrons - 1)
					{
						n_electron = 0;
						n_pix = EER_IMAGE_PIXELS;
						break;
					}
				}
			}
			else
			{
				// unpack every two symbols = 12 bit * 2 = 24 bit = 3 byte
				// high <- |bbbbBBBB|BBBBaaaa|AAAAAAAA| -> low
				// With SIMD intrinsics at the SSSE3 level, we can unpack 10 symbols (120 bits) simultaneously.
				unsigned char p1, p2, s1, s2;

				const long long pos_limit = frame_starts[iframe] + frame_sizes[iframe];
				// Because there is a footer, it is safe to go beyond the limit by two bytes.
				while (pos < pos_limit)
				{
					// symbol is bit tricky. 0000YyXx; Y and X must be flipped.
					p1 = buf[pos];
					s1 = (buf[pos + 1] & 0x0F) ^ 0x0A; // 0x0F = 00001111, 0x0A = 00001010

					p2 = (buf[pos + 1] >> 4) | (buf[pos + 2] << 4);
					s2 = (buf[pos + 2] >> 4) ^ 0x0A;

					// Note the order. Add p before checking the size and placing a new electron.
					n_pix += p1;
					if (n_pix == EER_IMAGE_PIXELS) break;
					if (p1 < 255)
					{
						positions[n_electron] = n_pix;
						symbols[n_electron] = s1;
						n_electron++;
						n_pix++;
					}

					n_pix += p2;
					if (n_pix == EER_IMAGE_PIXELS) break;
					if (p2 < 255)
					{
						positions[n_electron] = n_pix;
						symbols[n_electron] = s2;
						n_electron++;
						n_pix++;
					}
					
					pos += 3;
				}
			}

			if (n_pix != EER_IMAGE_PIXELS)
				throw std::runtime_error("Number of pixels is not right.");

			if (eer_upsampling == 3)
				render16K(h_result, positions, symbols, n_electron);
			else if (eer_upsampling == 2)
				render8K(h_result, positions, symbols, n_electron);
			else if (eer_upsampling == 1)
				render4K(h_result, positions, symbols, n_electron);
			else
				throw std::runtime_error("Invalid EER upsamle");

			total_n_electron += n_electron;
		}

		free(buf);
	}
}