#pragma once
#include <cuda_runtime.h>
#ifdef _WIN32
#define WARP_BLOB_EXPORT extern "C" __declspec(dllexport)
#else
#define WARP_BLOB_EXPORT extern "C"
#endif

// Pixel units. Dark blob against a circular background of radius 2*radius: a difference of Gaussians
// (centre sigma radius/2, surround sigma radius) when innerRadius is 0, else a flat annulus.
// Power = local mean-subtracted mean-square intensity in that support.
// Correlation = Pearson coefficient with the zero-mean template. Invalid/border
// pixels have correlation -infinity and power zero. Status: 0 success, -1 invalid
// arguments/nonfinite input, -2 allocation failure, -3 FFT failure, else CUDA code.
WARP_BLOB_EXPORT int BlobPickerScore(const float* input, int2 dims, float radius,
                                    float innerRadius, float* correlation, float* power);
// Euclidean local maxima within separation; positive correlations only, ties favor lower pixel ID.
// Callers wanting peak-to-peak suppression pass a small separation and suppress the list themselves.
// Returned float4 = (x,y,correlation,power), sorted by pixel ID, zero-based pixels.
// Caller releases malloc-owned host memory with HostFree, including on exceptions.
WARP_BLOB_EXPORT int BlobPickerPeaks(const float* correlation, const float* power,
                                    int2 dims, float separation, float4** picks, int* count);
