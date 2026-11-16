#ifndef MEMORY_TIFF_H
#define MEMORY_TIFF_H

#include "tiffio.h"
#include <string.h>

// A TIFF file already read into memory. libtiff reads it through these procedures, and the map procedure lets it
// decode strips straight from the buffer, so decoding issues no file system calls at all.
struct MemoryTiff
{
	const unsigned char* data;
	toff_t size, position;
};

static inline tsize_t MemoryTiffRead(thandle_t handle, tdata_t buffer, tsize_t count)
{
	MemoryTiff* file = (MemoryTiff*)handle;
	toff_t available = file->position < file->size ? file->size - file->position : 0;
	if ((toff_t)count > available)
		count = (tsize_t)available;
	memcpy(buffer, file->data + file->position, count);
	file->position += count;
	return count;
}

static inline tsize_t MemoryTiffWrite(thandle_t, tdata_t, tsize_t) { return -1; }

static inline toff_t MemoryTiffSeek(thandle_t handle, toff_t offset, int whence)
{
	MemoryTiff* file = (MemoryTiff*)handle;
	if (whence == SEEK_SET)
		file->position = offset;
	else if (whence == SEEK_CUR)
		file->position += offset;
	else
		file->position = file->size + offset;
	return file->position;
}

static inline int MemoryTiffClose(thandle_t) { return 0; }

static inline toff_t MemoryTiffSize(thandle_t handle) { return ((MemoryTiff*)handle)->size; }

static inline int MemoryTiffMap(thandle_t handle, tdata_t* base, toff_t* size)
{
	MemoryTiff* file = (MemoryTiff*)handle;
	*base = (tdata_t)file->data;
	*size = file->size;
	return 1;
}

static inline void MemoryTiffUnmap(thandle_t, tdata_t, toff_t) {}

static inline TIFF* OpenMemoryTiff(MemoryTiff* file)
{
	return TIFFClientOpen("memory", "r", (thandle_t)file, MemoryTiffRead, MemoryTiffWrite, MemoryTiffSeek, MemoryTiffClose,
		MemoryTiffSize, MemoryTiffMap, MemoryTiffUnmap);
}

#endif
