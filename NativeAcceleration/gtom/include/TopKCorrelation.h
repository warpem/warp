#ifndef TOP_K_CORRELATION_H
#define TOP_K_CORRELATION_H

#include <stddef.h>
#include <math.h>

#ifdef __CUDACC__
#define GTOM_TOPK_INLINE __host__ __device__ inline
#else
#define GTOM_TOPK_INLINE inline
#endif

namespace gtom
{
    // The same insertion routine is used by the CUDA kernel and its host test.
    // Each angle must be inserted once. The caller initializes scores to -inf
    // and angle IDs to -1. Lists are ordered by score descending, then ID ascending.
    template <typename T>
    GTOM_TOPK_INLINE void InsertCorrelationTopK(T score, float angle,
                                               T* scores, float* angles,
                                               size_t voxel, size_t rankstride,
                                               unsigned int topk)
    {
        if (topk == 0 || !(score > -INFINITY && score < INFINITY))
            return;

        unsigned int rank = topk - 1;
        size_t offset = rank * rankstride + voxel;
        if (!(score > scores[offset] || (score == scores[offset] && angle < angles[offset])))
            return;

        // Move only the entries displaced by this new score. Rank-major storage
        // keeps accesses coalesced across the GPU threads handling adjacent voxels.
        while (rank > 0)
        {
            size_t previous = offset - rankstride;
            if (!(score > scores[previous] || (score == scores[previous] && angle < angles[previous])))
                break;

            scores[offset] = scores[previous];
            angles[offset] = angles[previous];
            offset = previous;
            --rank;
        }

        scores[offset] = score;
        angles[offset] = angle;
    }
}

#undef GTOM_TOPK_INLINE

#endif
