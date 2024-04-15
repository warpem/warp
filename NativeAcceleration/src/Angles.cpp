#include "include/Functions.h"
#include "liblion/liblion.h"

__declspec(dllexport) int __stdcall GetAnglesCount(int healpixorder, char* c_symmetry, float limittilt)
{
    relion::FileName fn_symmetry(c_symmetry);

    relion::HealpixSampling sampling;
    sampling.setTranslations(1, 0);
    sampling.setOrientations(healpixorder);
    sampling.psi_step = -1;
    sampling.limit_tilt = limittilt;
    sampling.fn_sym = fn_symmetry;
    sampling.initialise(NOPRIOR, 3);

    return sampling.directions_ipix.size() * sampling.psi_angles.size();
}

__declspec(dllexport) void __stdcall GetAngles(float3* h_angles, int healpixorder, char* c_symmetry, float limittilt)
{
    relion::FileName fn_symmetry(c_symmetry);

    relion::HealpixSampling sampling;
    sampling.setTranslations(1, 0);
    sampling.setOrientations(healpixorder);
    sampling.psi_step = -1;
    sampling.limit_tilt = limittilt;
    sampling.fn_sym = fn_symmetry;
    sampling.initialise(NOPRIOR, 3);

    for (int rt = 0; rt < sampling.directions_ipix.size(); rt++)
        for (int p = 0; p < sampling.psi_angles.size(); p++)
            h_angles[rt * sampling.psi_angles.size() + p] = make_float3(sampling.rot_angles[rt], sampling.tilt_angles[rt], sampling.psi_angles[p]);
}