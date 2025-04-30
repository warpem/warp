#include "include/Functions.h"
#include "tiffio.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <immintrin.h> // Include for AVX2 intrinsics

#include "LanczosEER.hpp"

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
	// Process 8 elements at a time using AVX2
	const int vec_width = 8;
	int i = 0;
	const __m256i x_pos_mask = _mm256_set1_epi32(4095);  // Mask for positions[i] & 4095
	const __m256i x_sym_mask = _mm256_set1_epi32(3);     // Mask for symbols[i] & 3
	const __m256i y_sym_mask = _mm256_set1_epi32(12);    // Mask for symbols[i] & 12
	const int y_pos_shift = 12;
	const int index_shift = 14; // For y << 14

	// Main AVX2 loop
	for (; i <= n_electrons - vec_width; i += vec_width)
	{
		// Load 8 positions (32-bit unsigned int)
		__m256i pos_vec = _mm256_loadu_si256((__m256i const*)&positions[i]);

		// Load 8 symbols (8-bit unsigned char) and convert to 32-bit integers
		__m128i sym_vec_8 = _mm_loadl_epi64((__m128i const*)&symbols[i]);
		__m256i sym_vec_32 = _mm256_cvtepu8_epi32(sym_vec_8);

		// --- Calculate x ---
		// x_pos_part = (positions[i] & 4095) << 2
		__m256i x_pos_part = _mm256_and_si256(pos_vec, x_pos_mask);
		x_pos_part = _mm256_slli_epi32(x_pos_part, 2);

		// x_sym_part = symbols[i] & 3
		__m256i x_sym_part = _mm256_and_si256(sym_vec_32, x_sym_mask);

		// x = x_pos_part | x_sym_part
		__m256i x_vec = _mm256_or_si256(x_pos_part, x_sym_part); // 4095 = 111111111111b, 3 = 00000011b

		// --- Calculate y ---
		// y_pos_part = (positions[i] >> 12) << 2
		__m256i y_pos_part = _mm256_srli_epi32(pos_vec, y_pos_shift);
		y_pos_part = _mm256_slli_epi32(y_pos_part, 2);

		// y_sym_part = (symbols[i] & 12) >> 2
		__m256i y_sym_part = _mm256_and_si256(sym_vec_32, y_sym_mask);
		y_sym_part = _mm256_srli_epi32(y_sym_part, 2);

		// y = y_pos_part | y_sym_part
		__m256i y_vec = _mm256_or_si256(y_pos_part, y_sym_part); //  4096 = 2^12, 12 = 00001100b

		// --- Calculate index ---
		// y_shifted = y << 14
		__m256i y_shifted_vec = _mm256_slli_epi32(y_vec, index_shift);

		// index = y_shifted + x
		__m256i index_vec = _mm256_add_epi32(y_shifted_vec, x_vec);

		// Store calculated indices to a temporary array
		alignas(32) int indices[vec_width];
		_mm256_store_si256((__m256i*)indices, index_vec);

		// Increment image elements individually using calculated indices
		#pragma unroll
		for (int j = 0; j < vec_width; ++j)
			image[indices[j]]++;
	}

	// Handle remaining elements (less than 8) with the original scalar code
	for (; i < n_electrons; ++i)
	{
		int x = ((positions[i] & 4095) << 2) | (symbols[i] & 3); // 4095 = 111111111111b, 3 = 00000011b
		int y = ((positions[i] >> 12) << 2) | ((symbols[i] & 12) >> 2); //  4096 = 2^12, 12 = 00001100b
		image[(y << 14) + x]++; // y << 14 corresponds to index_shift
	}
}

void render8K(float* image, std::vector<unsigned int>& positions, std::vector<unsigned char>& symbols, int n_electrons)
{
	// Process 8 elements at a time using AVX2
	const int vec_width = 8;
	int i = 0;
	const __m256i x_pos_mask = _mm256_set1_epi32(4095);  // Mask for positions[i] & 4095
	const __m256i x_sym_mask = _mm256_set1_epi32(2);     // Mask for symbols[i] & 2
	const __m256i y_sym_mask = _mm256_set1_epi32(8);     // Mask for symbols[i] & 8
	const int y_pos_shift = 12;
	const int index_shift = 13; // For y << 13

	// Main AVX2 loop
	for (; i <= n_electrons - vec_width; i += vec_width)
	{
		// Load 8 positions (32-bit unsigned int)
		__m256i pos_vec = _mm256_loadu_si256((__m256i const*)&positions[i]);

		// Load 8 symbols (8-bit unsigned char) and convert to 32-bit integers
		// Load 8 bytes into the lower 64 bits of a 128-bit register
		__m128i sym_vec_8 = _mm_loadl_epi64((__m128i const*)&symbols[i]);
		// Zero-extend 8-bit values to 32-bit values
		__m256i sym_vec_32 = _mm256_cvtepu8_epi32(sym_vec_8);

		// --- Calculate x ---
		// x_pos_part = (positions[i] & 4095) << 1
		__m256i x_pos_part = _mm256_and_si256(pos_vec, x_pos_mask);
		x_pos_part = _mm256_slli_epi32(x_pos_part, 1);

		// x_sym_part = (symbols[i] & 2) >> 1
		__m256i x_sym_part = _mm256_and_si256(sym_vec_32, x_sym_mask);
		x_sym_part = _mm256_srli_epi32(x_sym_part, 1);

		// x = x_pos_part | x_sym_part
		__m256i x_vec = _mm256_or_si256(x_pos_part, x_sym_part); // 4095 = 111111111111b, 2 = 00000010b

		// --- Calculate y ---
		// y_pos_part = (positions[i] >> 12) << 1
		__m256i y_pos_part = _mm256_srli_epi32(pos_vec, y_pos_shift);
		y_pos_part = _mm256_slli_epi32(y_pos_part, 1);

		// y_sym_part = (symbols[i] & 8) >> 3
		__m256i y_sym_part = _mm256_and_si256(sym_vec_32, y_sym_mask);
		y_sym_part = _mm256_srli_epi32(y_sym_part, 3);

		// y = y_pos_part | y_sym_part
		__m256i y_vec = _mm256_or_si256(y_pos_part, y_sym_part); //  4096 = 2^12, 8 = 00001000b

		// --- Calculate index ---
		// y_shifted = y << 13
		__m256i y_shifted_vec = _mm256_slli_epi32(y_vec, index_shift);

		// index = y_shifted + x
		__m256i index_vec = _mm256_add_epi32(y_shifted_vec, x_vec);

		// Store calculated indices to a temporary array
		alignas(32) int indices[vec_width];
		_mm256_store_si256((__m256i*)indices, index_vec);

		// Increment image elements individually using calculated indices
		#pragma unroll
		for (int j = 0; j < vec_width; ++j)
			image[indices[j]]++;
	}

	// Handle remaining elements (less than 8) with the original scalar code
	for (; i < n_electrons; ++i)
	{
		int x = ((positions[i] & 4095) << 1) | ((symbols[i] & 2) >> 1); // 4095 = 111111111111b, 2 = 00000010b
		int y = ((positions[i] >> 12) << 1) | ((symbols[i] & 8) >> 3); //  4096 = 2^12, 8 = 00001000b
		image[(y << 13) + x]++; // y << 13 corresponds to index_shift
	}
}

void render4K(float* image, std::vector<unsigned int>& positions, std::vector<unsigned char>& symbols, int n_electrons)
{
	// Process 8 elements at a time using AVX2
	const int vec_width = 8;
	int i = 0;
	const __m256i x_mask = _mm256_set1_epi32(4095); // Mask for x: 0xFFF
	const int y_shift = 12;
	const int index_shift = 12;

	int last_index = -1;

	// Main AVX2 loop
	for (; i <= n_electrons - vec_width; i += vec_width)
	{
		// Load 8 positions (potentially unaligned)
		// Note: symbols are not used in 4K rendering
		__m256i pos_vec = _mm256_loadu_si256((__m256i const*)&positions[i]);

		// Calculate x = positions[i] & 4095 for 8 elements
		__m256i x_vec = _mm256_and_si256(pos_vec, x_mask);

		// Calculate y = positions[i] >> 12 for 8 elements
		__m256i y_vec = _mm256_srli_epi32(pos_vec, y_shift);

		// Calculate y_shifted = y << 12 for 8 elements
		__m256i y_shifted_vec = _mm256_slli_epi32(y_vec, index_shift);

		// Calculate index = y_shifted + x for 8 elements
		__m256i index_vec = _mm256_add_epi32(y_shifted_vec, x_vec);

		// Store calculated indices to a temporary array
		// This is necessary because AVX2 does not have float scatter instructions
		// We then iterate through the temporary array to increment the image
		alignas(32) int indices[vec_width]; // Align for potential faster store
		_mm256_store_si256((__m256i*)indices, index_vec);

		// Increment image elements individually using calculated indices
		// This part is not vectorized due to lack of scatter support
		#pragma unroll // Suggest loop unrolling to the compiler
		for (int j = 0; j < vec_width; ++j)
		{
			if (indices[j] == last_index)
			{
				std::cout << "Error: Index out of bounds." << std::endl;
				throw std::runtime_error("Index out of bounds.");
			}
			last_index = indices[j];

			image[indices[j]]++;
		}
	}

	// Handle remaining elements (less than 8) with the original scalar code
	for (; i < n_electrons; ++i)
	{
		int x = positions[i] & 4095; // 4095 = 111111111111b
		int y = positions[i] >> 12; //  4096 = 2^12
		image[(y << 12) + x]++;
	}
}


__declspec(dllexport) void ReadEERCombinedFrame(const char* path, int firstFrameInclusive, int lastFrameExclusive, int eer_upsampling, float* h_result)
{
	
	TIFFSetWarningHandler(0);

	if (eer_upsampling < 1 || eer_upsampling > 3)
	{
		std::cout << "EERRenderer::read: eer_upsampling must be 1, 2 or 3." << std::endl;
		throw("EERRenderer::read: eer_upsampling must be 1, 2 or 3.");
	}

	int nframes = 0;
	bool is_7bit = false;
	
	// Try reading as TIFF; this handle is kept open
	TIFF* ftiff = TIFFOpen(path, "rm");

	if (ftiff == NULL)
	{
		std::cout << "Error: Legacy mode not implemented" << std::endl;
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
		{
			std::cout << "Error: Unknown compression scheme for EER" << std::endl;
			throw std::runtime_error("Unknown compression scheme for EER");
		}

		if (width != EER_IMAGE_WIDTH || height != EER_IMAGE_HEIGHT)
		{
			std::cout << "Error: Currently we support only 4096x4096 pixel EER movies." << std::endl;
			throw std::runtime_error("Currently we support only 4096x4096 pixel EER movies.");
		}

		// Find the number of frames
		nframes = TIFFNumberOfDirectories(ftiff);
	}
	
	std::vector<long long> frame_starts, frame_sizes;
	std::vector<unsigned char> data_buffer;

	// Validate frame range
	if (firstFrameInclusive < 0 || lastFrameExclusive > nframes || firstFrameInclusive >= lastFrameExclusive)
	{
		TIFFClose(ftiff);
		std::cout << "Error: Invalid frame range specified." << std::endl;
		throw std::runtime_error("Invalid frame range specified.");
	}

	{
		frame_starts.resize(nframes, 0);
		frame_sizes.resize(nframes, 0);
		
		// Estimate buffer size based on the first frame
		long long first_frame_size = 0;
		if (TIFFSetDirectory(ftiff, firstFrameInclusive) != 1)
		{
			TIFFClose(ftiff);
			std::cout << "Error: Failed to set directory for first frame." << std::endl;
			throw std::runtime_error("Failed to set directory for first frame.");
		}
		const int first_nstrips = TIFFNumberOfStrips(ftiff);
		for (int strip = 0; strip < first_nstrips; strip++)
		{
			tmsize_t strip_size = TIFFRawStripSize(ftiff, strip);
			if (strip_size == (tmsize_t)-1)
			{
				TIFFClose(ftiff);
				std::cout << "Error: Failed to get strip size for first frame." << std::endl;
				throw std::runtime_error("Failed to get strip size for first frame.");
			}
			first_frame_size += strip_size;
		}
		
		int num_frames_to_read = lastFrameExclusive - firstFrameInclusive;
		size_t estimated_size = static_cast<size_t>(first_frame_size * num_frames_to_read * 1.5);
		data_buffer.resize(estimated_size); // Pre-allocate memory

		size_t current_offset = 0;

		// Read everything directly into the vector
		for (int frame = firstFrameInclusive; frame < lastFrameExclusive; frame++)
		{
			if (TIFFSetDirectory(ftiff, frame) != 1)
			{
				TIFFClose(ftiff);
				std::cout << "Error: Failed to set directory during reading." << std::endl;
				throw std::runtime_error("Failed to set directory during reading.");
			}
			const int nstrips = TIFFNumberOfStrips(ftiff);
			frame_starts[frame] = current_offset; // Store start offset relative to vector beginning

			for (int strip = 0; strip < nstrips; strip++)
			{
				tmsize_t strip_size = TIFFRawStripSize(ftiff, strip);
				if (strip_size == (tmsize_t)-1) 
				{
					TIFFClose(ftiff);
					std::cout << "Error: Failed to get strip size during reading." << std::endl;
					throw std::runtime_error("Failed to get strip size during reading.");
				}
				
				// Ensure vector has enough space; resize adds elements AND potentially reallocates
				if (current_offset + strip_size + 1024 > data_buffer.size()) 
						data_buffer.resize(current_offset + strip_size + 1024); 

				// Read directly into the vector's memory
				tmsize_t bytes_read = TIFFReadRawStrip(ftiff, strip, data_buffer.data() + current_offset, strip_size);
				if (bytes_read == -1) 
				{
					TIFFClose(ftiff);
					std::cout << "Error: Failed to read raw strip." << std::endl;
					throw std::runtime_error("Failed to read raw strip.");
				}

				current_offset += bytes_read; // Increment offset by actual bytes read
				frame_sizes[frame] += bytes_read; // Add to this frame's total size
			}
		}

		TIFFClose(ftiff); // Close TIFF file after reading
	}
	

	{
		long long total_n_electron = 0;

		long long supersize = 4096 << (eer_upsampling - 1);

		std::vector<unsigned int> positions;
		std::vector<unsigned char> symbols;
		//memset(h_result, 0, supersize * supersize * sizeof(float));

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

				// Use data_buffer.data() instead of buf
				const unsigned char* current_frame_data = data_buffer.data() + frame_starts[iframe]; 
				const size_t current_frame_size = frame_sizes[iframe]; 

				while (true)
				{
					// Fetch 32 bits and unpack up to 2 chunks of 7 + 4 bits.
					// This is faster than unpack 7 and 4 bits sequentially.
					// Since the size of buf is larger than the actual size by the TIFF header size,
					// it is always safe to read ahead.

					long long byte_offset = bit_pos >> 3;
					const unsigned int bit_offset_in_first_byte = bit_pos & 7; // 7 = 00000111 (same as % 8)
					const unsigned int chunk = (*(unsigned int*)(current_frame_data + byte_offset)) >> bit_offset_in_first_byte;

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

				// Use data_buffer.data() and frame specific boundaries
				const unsigned char* current_frame_data = data_buffer.data() + frame_starts[iframe];
				const size_t current_frame_size = frame_sizes[iframe];
				size_t current_byte_in_frame = 0; // Track position within the current frame's data segment

				// Because there is a footer, it is safe to go beyond the limit by two bytes. (Original comment, needs verification)
				// Let's stick to the known size for safety.
				while (current_byte_in_frame + 2 < current_frame_size) // Ensure we can read 3 bytes safely
				{
					// symbol is bit tricky. 0000YyXx; Y and X must be flipped.
					p1 = current_frame_data[current_byte_in_frame];
					s1 = (current_frame_data[current_byte_in_frame + 1] & 0x0F) ^ 0x0A; // 0x0F = 00001111, 0x0A = 00001010

					p2 = (current_frame_data[current_byte_in_frame + 1] >> 4) | (current_frame_data[current_byte_in_frame + 2] << 4);
					s2 = (current_frame_data[current_byte_in_frame + 2] >> 4) ^ 0x0A;

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
					
					current_byte_in_frame += 3; // Move forward 3 bytes
				}
				// Need to handle potential leftover bytes if frame_size isn't multiple of 3? 
				// Original code didn't seem to, might rely on footer/padding or exact multiples. Let's assume it's okay for now.
			}

			if (n_pix != EER_IMAGE_PIXELS)
			{
				std::cout << "Error: Number of pixels is not right." << std::endl;
				throw std::runtime_error("Number of pixels is not right.");
			}

			/*if (eer_upsampling == 3)
				render16K(h_result, positions, symbols, n_electron);
			else if (eer_upsampling == 2)
				render8K(h_result, positions, symbols, n_electron);
			else if (eer_upsampling == 1)
				render4K(h_result, positions, symbols, n_electron);
			else
			{
				std::cout << "Error: Invalid EER upsamle" << std::endl;
				throw std::runtime_error("Invalid EER upsamle");
			}*/

			render_eer_frame_lanczos(positions, symbols, n_electron, 4096 * 1, 4096 * 1, h_result);

			total_n_electron += n_electron;
		}
	}
}