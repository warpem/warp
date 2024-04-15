#ifndef ALIGNED_ALLOC_H
#define ALIGNED_ALLOC_H

#include <stdlib.h>
#include "config.h"

#ifdef _WIN32
#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)
#define posix_memalign_free _aligned_free
#else
#define posix_memalign_free free
#endif

#ifdef HAVE_POSIX_MEMALIGN
inline void*
aligned_alloc(size_t size, size_t alignment)
{
	void* ptr;
	posix_memalign(&ptr, alignment, size);
	return ptr;
}

inline void
aligned_free(void* ptr)
{
	posix_memalign_free(ptr);
}

#else

inline void*
aligned_alloc(size_t size, size_t alignment)
{
	size += (alignment - 1) + sizeof(void*);
	void* ptr = malloc(size);
	if (ptr == NULL)
		return NULL;
	else {
		void* shifted = ptr + sizeof(void*);
		size_t offset = alignment - (size_t)shifted % (size_t)alignment;
		void* aligned = shifted + offset;
		*((void**)aligned - 1) = ptr;
		return aligned;
	}
}

inline void
aligned_free(void* aligned)
{
	void* ptr = *((void**)aligned - 1);
	free(ptr);
}
#endif


#endif
