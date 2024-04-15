#include "include/Functions.h"
#include "liblion/liblion.h"
using namespace gtom;

__declspec(dllexport) void __stdcall OptimizeWeights(int nrecs, float* h_recft, float* h_recweights, float* h_r2, int elements, int* h_subsets, float* h_bfacs, float* h_weightfactors, float* h_recsum1, float* h_recsum2, float* h_weightsum1, float* h_weightsum2)
{
    for (int n = 0; n < nrecs; n++)
    {
        float Weight = h_weightfactors[n];
        float Bfac = h_bfacs[n];

        float2* recsum;
        float* weightsum;

        if (h_subsets[n] % 2 == 0)
        {
            recsum = (float2*)h_recsum1;
            weightsum = h_weightsum1;
        }
        else
        {
            recsum = (float2*)h_recsum2;
            weightsum = h_weightsum2;
        }

        float2* rec = (float2*)h_recft + elements * n;
        float* weights = h_recweights + elements * n;

#pragma omp parallel for
        for (int i = 0; i < elements; i++)
        {
            float OverallWeight = Weight * (float)exp(h_r2[i] * Bfac) * weights[i];
            recsum[i] += rec[i] * OverallWeight;
            weightsum[i] += OverallWeight;
        }
    }
}