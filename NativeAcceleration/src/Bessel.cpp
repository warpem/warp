#include "include/Functions.h"
#include "liblion/liblion.h"

__declspec(dllexport) float __stdcall KaiserBessel(float r, float a, float alpha, int m)
{
    return (float)relion::kaiser_value(r, a, alpha, m);
}

__declspec(dllexport) float __stdcall KaiserBesselFT(float w, float a, float alpha, int m)
{
    return (float)relion::kaiser_Fourier_value(w, a, alpha, m);
}

__declspec(dllexport) float __stdcall KaiserBesselProj(float r, float a, float alpha, int m)
{
    return (float)relion::kaiser_proj(r, a, alpha, m);
}