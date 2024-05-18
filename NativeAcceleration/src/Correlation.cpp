#include "include/Functions.h"
using namespace gtom;

__declspec(dllexport) void __stdcall CorrelateSubTomos(unsigned long long t_projectordataRe,
                                                        unsigned long long t_projectordataIm,
                                                        float projectoroversample, 
                                                        int3 dimsprojector,
                                                        float2* d_experimentalft,
                                                        float* d_ctf,
                                                        int3 dimsvolume,
                                                        uint nvolumes,
                                                        float3* h_angles,
                                                        uint nangles,
                                                        uint batchangles,
                                                        float maskradius,
                                                        float* d_bestcorrelation,
                                                        int* d_bestangle,
                                                        float* h_progressfraction)
{
    d_PickSubTomograms(t_projectordataRe,
                        t_projectordataIm,
                        projectoroversample,
                        dimsprojector,
                        d_experimentalft,
                        d_ctf,
                        dimsvolume,
                        nvolumes,
                        (tfloat3*)h_angles,
                        nangles,
                        batchangles,
                        maskradius,
                        d_bestcorrelation,
                        (float*)d_bestangle,
                        h_progressfraction);
}

__declspec(dllexport) void __stdcall CorrelateSubTomosDiff2(unsigned long long t_projectordataRe,
                                                            unsigned long long t_projectordataIm,
                                                            float projectoroversample,
                                                            int3 dimsprojector,
                                                            float2* d_experimentalft,
                                                            float* d_ctf,
                                                            int3 dimsvolume,
                                                            uint nvolumes,
                                                            int3 dimsrelevant,
                                                            float3* h_angles,
                                                            uint nangles,
                                                            float* d_bestcorrelation,
                                                            int* d_bestangle)
{
    d_PickSubTomogramsDiff2(t_projectordataRe,
                            t_projectordataIm,
                            projectoroversample,
                            dimsprojector,
                            d_experimentalft,
                            d_ctf,
                            dimsvolume,
                            nvolumes,
                            dimsrelevant,
                            (tfloat3*)h_angles,
                            nangles,
                            d_bestcorrelation,
                            (float*)d_bestangle);
}

__declspec(dllexport) int* __stdcall LocalPeaks(float* d_input, int* h_peaksnum, int3 dims, int localextent, float threshold)
{
    int* h_peaks = 0;

    d_LocalPeaks(d_input, (int3**)&h_peaks, h_peaksnum, dims, localextent, threshold);

    return h_peaks;
}

__declspec(dllexport) void SubpixelMax(float* d_input, float* d_output, int3 dims, int subpixsteps)
{
    d_SubpixelMax(d_input, d_output, dims, subpixsteps);
}

__declspec(dllexport) void PeakOne2D(float* d_input, float3* d_positions, float* d_values, int2 dims, int2 dimsregion, bool subtractcenter, int batch)
{
    d_PeakOne2D(d_input, d_positions, d_values, dims, dimsregion, subtractcenter, batch);
}

__declspec(dllexport) void CorrelateRealspace(float* d_image1, float* d_image2, int3 dims, float* d_mask, float* d_corr, uint batch)
{
    d_CorrelateRealspace(d_image1, d_image2, dims, d_mask, d_corr, batch);
}