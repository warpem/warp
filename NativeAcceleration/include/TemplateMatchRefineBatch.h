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
constexpr int Parameters = 6;
constexpr int UpperHessian = 21;
constexpr double Pi = 3.1415926535897932384626433832795;
constexpr double MaximumRotation = 5.0 * Pi / 180.0;
constexpr double RotationTolerance = 0.01 * Pi / 180.0;
constexpr double TranslationTolerancePixels = 0.005;
constexpr double CurvatureScaleFloor = 1.0e-3;

// Parameter order is position XYZ (Angstrom), right-local rotation XYZ (radian).
TMB_HD inline int UpperIndex(int row, int column)
{
    if (column < row) { const int t = row; row = column; column = t; }
    return row * Parameters - row * (row - 1) / 2 + column - row;
}

TMB_HD inline double Maximum(double a, double b) { return a > b ? a : b; }
TMB_HD inline double Minimum(double a, double b) { return a < b ? a : b; }

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

// All matrix storage is column-major. Keep this shared with portable tests so
// changing the CUDA update cannot silently change the rotation convention.
TMB_HD inline void RotateRight(const float* rotation, const double* omega, float* output)
{
    const double theta2 = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    double a, b;
    if (theta2 > 1.0e-6)
    {
        const double theta = ::sqrt(theta2);
        a = ::sin(theta) / theta;
        b = (1.0 - ::cos(theta)) / theta2;
    }
    else
    {
        a = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0;
        b = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0;
    }
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
        {
            double value = 0;
            for (int k = 0; k < 3; ++k)
            {
                double skew = 0;
                if (k == 0 && column == 1) skew = -omega[2];
                if (k == 0 && column == 2) skew = omega[1];
                if (k == 1 && column == 0) skew = omega[2];
                if (k == 1 && column == 2) skew = -omega[0];
                if (k == 2 && column == 0) skew = -omega[1];
                if (k == 2 && column == 1) skew = omega[0];
                const double exponential = (k == column ? 1.0 : 0.0) + a * skew +
                    b * (omega[k] * omega[column] - (k == column ? theta2 : 0.0));
                value += rotation[row + 3 * k] * exponential;
            }
            output[row + 3 * column] = float(value);
        }
}

// Signed Z=C/sqrt(P), including C<=0. Input sufficient statistics are summed
// across every tilt before materialization: [C,P,c[6],b[6],G_upper[21]].
// The positive scale is held fixed for each trust step. This PSD Gauss-Newton
// curvature belongs to (d-s*m/sqrt(P))/sqrt(s), whose objective is constant-Z.
TMB_HD inline bool SignedNormal(const double* stats, double& z, double* gradient, double* hessian)
{
    if (!(stats[1] > 0) || !::isfinite(stats[0]) || !::isfinite(stats[1])) return false;
    const double root = ::sqrt(stats[1]);
    z = stats[0] / root;
    if (!::isfinite(z)) return false;
    const double scale = Maximum(::fabs(z), CurvatureScaleFloor);
    for (int j = 0; j < Parameters; ++j)
    {
        gradient[j] = -(stats[2 + j] - stats[0] / stats[1] * stats[8 + j]) / root;
        if (!::isfinite(gradient[j])) return false;
        for (int k = j; k < Parameters; ++k)
        {
            const int index = UpperIndex(j, k);
            hessian[index] = scale * (stats[14 + index] / stats[1] -
                (stats[8 + j] / stats[1]) * (stats[8 + k] / stats[1]));
            if (!::isfinite(hessian[index])) return false;
        }
    }
    return true;
}

// Six-dimensional symmetric Jacobi eigensolve. CUDA calls it in thread zero
// with shared-memory arrays; no per-thread dynamically indexed matrix arrays.
TMB_HD inline bool Eigen6(double* matrix, double* vectors, double* values)
{
    for (int i = 0; i < 36; ++i) vectors[i] = i / 6 == i % 6 ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 40; ++sweep)
    {
        double largest = 0, scale = 0;
        for (int row = 0; row < 6; ++row)
        {
            scale = Maximum(scale, ::fabs(matrix[row * 6 + row]));
            for (int col = row + 1; col < 6; ++col)
                largest = Maximum(largest, ::fabs(matrix[row * 6 + col]));
        }
        if (largest <= 16.0 * DBL_EPSILON * Maximum(scale, largest))
        {
            for (int i = 0; i < 6; ++i) values[i] = matrix[i * 6 + i];
            return true;
        }
        for (int p = 0; p < 5; ++p)
            for (int q = p + 1; q < 6; ++q)
            {
                const double off = matrix[p * 6 + q];
                if (off == 0) continue;
                const double tau = (matrix[q * 6 + q] - matrix[p * 6 + p]) / (2.0 * off);
                const double tangent = ::copysign(1.0 / (::fabs(tau) + ::hypot(1.0, tau)), tau);
                const double cosine = 1.0 / ::hypot(1.0, tangent), sine = tangent * cosine;
                matrix[p * 6 + p] -= tangent * off;
                matrix[q * 6 + q] += tangent * off;
                matrix[p * 6 + q] = matrix[q * 6 + p] = 0;
                for (int k = 0; k < 6; ++k)
                {
                    if (k != p && k != q)
                    {
                        const double kp = matrix[k * 6 + p], kq = matrix[k * 6 + q];
                        matrix[k * 6 + p] = matrix[p * 6 + k] = cosine * kp - sine * kq;
                        matrix[k * 6 + q] = matrix[q * 6 + k] = sine * kp + cosine * kq;
                    }
                    const double vp = vectors[k * 6 + p], vq = vectors[k * 6 + q];
                    vectors[k * 6 + p] = cosine * vp - sine * vq;
                    vectors[k * 6 + q] = sine * vp + cosine * vq;
                }
            }
    }
    return false;
}

TMB_HD inline double EigenStep(const double* values, const double* vectors,
                              const double* rotatedGradient, double floor, double lambda,
                              double* step)
{
    double norm2 = 0;
    for (int row = 0; row < 6; ++row)
    {
        double value = 0;
        for (int k = 0; k < 6; ++k)
            value -= vectors[row * 6 + k] * rotatedGradient[k] / Maximum(floor, values[k] + lambda);
        step[row] = value;
        norm2 += value * value;
    }
    return ::sqrt(norm2);
}

TMB_HD inline bool TrustStep(const double* gradient, const double* hessian,
                            double rotationRadius, double trustRadius,
                            double* matrix, double* vectors, double* values,
                            double* rotatedGradient, double* step)
{
    for (int row = 0; row < 6; ++row)
        for (int col = 0; col < 6; ++col)
            matrix[row * 6 + col] = hessian[UpperIndex(row, col)] /
                ((row < 3 ? 1.0 : rotationRadius) * (col < 3 ? 1.0 : rotationRadius));
    if (!Eigen6(matrix, vectors, values)) return false;
    double largest = 0, smallest = values[0], gradientNorm2 = 0;
    for (int k = 0; k < 6; ++k)
    {
        largest = Maximum(largest, ::fabs(values[k]));
        smallest = Minimum(smallest, values[k]);
        rotatedGradient[k] = 0;
        for (int row = 0; row < 6; ++row)
            rotatedGradient[k] += vectors[row * 6 + k] * gradient[row] / (row < 3 ? 1.0 : rotationRadius);
        gradientNorm2 += rotatedGradient[k] * rotatedGradient[k];
    }
    const double floor = Maximum(DBL_MIN, largest * 8.0 * FLT_EPSILON);
    double lower = Maximum(0.0, -smallest + floor);
    if (EigenStep(values, vectors, rotatedGradient, floor, lower, step) > trustRadius * (1.0 + 8.0 * FLT_EPSILON))
    {
        // Keep the bracket in curvature units. An absolute lower bound of one
        // loses all useful lambda precision when the objective is rescaled.
        double upper = Maximum(DBL_MIN, lower + largest + ::sqrt(gradientNorm2) / trustRadius);
        for (int i = 0; i < 32 && EigenStep(values, vectors, rotatedGradient, floor, upper, step) > trustRadius; ++i)
            upper *= 2;
        if (!::isfinite(upper) || EigenStep(values, vectors, rotatedGradient, floor, upper, step) > trustRadius) return false;
        for (int i = 0; i < 32; ++i)
        {
            const double middle = 0.5 * (lower + upper);
            if (EigenStep(values, vectors, rotatedGradient, floor, middle, step) > trustRadius) lower = middle;
            else upper = middle;
        }
        EigenStep(values, vectors, rotatedGradient, floor, upper, step);
    }
    for (int i = 0; i < 6; ++i)
    {
        step[i] /= i < 3 ? 1.0 : rotationRadius;
        if (!::isfinite(step[i])) return false;
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
// summary [P,H,4]: C,P,initialZ,finalTrust. diagnostics [P,H,4]: accepted steps,
// evaluations, termination (0 budget,1 small step,2 rejected,3 inactive,4 invalid),
// mergedIntoSlot (-1 retained,-2 invalid/inactive,>=0 original slot within particle).
// tiltStats [P,H,T,2]: final C,P, also retained for merged slots. Inactive/invalid
// statistics are zero. Return 0 success, -1 invalid host inputs, or positive CUDA
// error (including allocation failure). Numerical failures are per-hypothesis diagnostics.
// maxIterations=0 evaluates seeds and merges without optimization. Either zero
// merge threshold disables merging. Each call initializes a fresh trust radius.
extern "C" int TemplateMatchRefineBatch(
    unsigned long long textureRe, unsigned long long textureIm,
    int dim, int box, int views, int particles, int hypotheses,
    const float2* d_data, const float* d_ctf, const float* d_quad,
    const float* d_inverseNoise, const float* d_phaseRadii,
    const float* h_geometry, const float* h_bounds, const float* h_symmetry,
    int symmetryCount, float* h_poses, int* h_seedIds,
    float pixel, float cutoff, float diameter, int maxIterations,
    float mergeDistance, float mergeAngle,
    double* h_summary, int* h_diagnostics, double* h_tiltStats);

// BFGS uses the same inputs and output layouts, but computes in FP32 and promotes
// the output statistics to double for ABI compatibility. summary[...,3] is zero
// (no trust radius); termination 2 means exhausted line search. Each call starts
// with a fresh inverse Hessian. The Gauss-Newton entry point above is unchanged.
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
