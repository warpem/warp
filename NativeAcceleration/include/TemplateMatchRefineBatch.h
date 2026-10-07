#ifndef WARP_TEMPLATE_MATCH_REFINE_BATCH_H
#define WARP_TEMPLATE_MATCH_REFINE_BATCH_H

#include <cmath>
#include <cfloat>

#ifdef __CUDACC__
#define TMB_HD __host__ __device__
#else
#define TMB_HD
#endif

namespace warp_template_match_batch
{
constexpr double Pi = 3.1415926535897932384626433832795;

TMB_HD inline bool ProperRotation(const float* r)
{
    for (int i = 0; i < 9; ++i) if (!::isfinite(r[i])) return false;
    const double determinant = r[0] * (double(r[4]) * r[8] - double(r[7]) * r[5]) -
        r[3] * (double(r[1]) * r[8] - double(r[7]) * r[2]) + r[6] * (double(r[1]) * r[5] - double(r[4]) * r[2]);
    if (::fabs(determinant - 1.0) > 2.0e-3) return false;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            double dot = 0;
            for (int row = 0; row < 3; ++row) dot += r[row + a * 3] * double(r[row + b * 3]);
            if (::fabs(dot - (a == b ? 1.0 : 0.0)) > 2.0e-3) return false;
        }
    return true;
}

}

#undef TMB_HD

struct float2;
// All d_* arrays are borrowed contiguous device [particles,views,box*(box/2+1)].
// h_geometry [P,T,18]: column-major G[9], shift Jacobian [dx/dX,dy/dX,...][6]
// in pixels/Angstrom, CTF phase Jacobian [3] in radian*Angstrom^2/Angstrom.
// Poses [P,H,12]: anchor-relative XYZ Angstrom then
// column-major rotation. Bounds [P,6]: lowerXYZ,upperXYZ. Symmetry [S,9] acts R*S.
// seedIds is in/out: -1 inactive; merged and invalid slots become -1. No compaction.
// summary [P,H,4]: C,P,initialZ,reservedZero. diagnostics [P,H,4]: accepted steps,
// evaluations, termination (0 budget,1 small step,2 rejected,3 inactive,4 invalid),
// mergedIntoSlot (-1 retained,-2 invalid/inactive,>=0 original slot within particle).
// tiltStats [P,H,T,2]: final C,P, also retained for merged slots. Inactive/invalid
// statistics are zero. Return 0 success, -1 invalid host inputs, or positive CUDA
// error (including allocation failure). Numerical failures are per-hypothesis diagnostics.
// maxIterations=0 evaluates seeds and merges without optimization. Either zero
// merge threshold disables merging. Each call initializes a fresh BFGS inverse Hessian.
extern "C" int TemplateMatchRefineBatchBfgs(
    unsigned long long textureRe, unsigned long long textureIm,
    int dim, int box, int views, int particles, int hypotheses,
    const float2* d_data, const float* d_ctf, const float* d_quad,
    const float* d_inverseNoise, const float* d_phaseRadii,
    const float* h_geometry, const float* h_bounds, const float* h_symmetry,
    int symmetryCount, float* h_poses, int* h_seedIds,
    float pixel, float cutoff, float diameter, int maxIterations,
    float mergeDistance, float mergeAngle,
    double* h_summary, int* h_diagnostics, double* h_tiltStats);

#endif
