#ifndef WARP_TEMPLATE_MATCH_BFGS_H
#define WARP_TEMPLATE_MATCH_BFGS_H

#include <cmath>
#include <cfloat>

#ifdef __CUDACC__
#define TMBFGS_HD __host__ __device__
#else
#define TMBFGS_HD
#endif

namespace warp_template_match_bfgs
{
constexpr int Parameters = 6;
constexpr int UpdateWorkspace = 42;

TMBFGS_HD inline bool Finite(float value)
{
#ifdef __CUDA_ARCH__
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

// Frequency coordinates are integers, so comparing x*x+y*y against the floor
// of the original squared cutoff preserves its exact disk membership. The host
// computes cutoffSquaredLimit=floor(double(cutoff)*double(cutoff)) once; no
// double arithmetic or conversion is needed for each GPU Fourier sample.
TMBFGS_HD inline bool IndependentSample(int x, int y, int box, unsigned long long cutoffSquaredLimit)
{
    if (x < 0 || (x == 0 && y <= 0)) return false;
    if (x >= box / 2 || y <= -box / 2 || y >= box / 2) return false;
    const long long dx = x, dy = y;
    const unsigned long long radiusSquared = static_cast<unsigned long long>(dx * dx) +
        static_cast<unsigned long long>(dy * dy);
    return radiusSquared <= cutoffSquaredLimit;
}

TMBFGS_HD inline float Dot6(const float* a, const float* b)
{
    float value = 0.0f;
    for (int i = 0; i < Parameters; ++i) value = ::fmaf(a[i], b[i], value);
    return value;
}

// Coefficients of Exp(omega) and its right Jacobian. Series avoid subtracting
// nearly equal floats in (1-cos(theta)) and (theta-sin(theta)).
TMBFGS_HD inline void RotationCoefficients(float thetaSquared, float& a, float& b, float& c)
{
    if (thetaSquared < 0.01f)
    {
        a = ::fmaf(thetaSquared, ::fmaf(thetaSquared, ::fmaf(thetaSquared,
            -1.0f / 5040.0f, 1.0f / 120.0f), -1.0f / 6.0f), 1.0f);
        b = ::fmaf(thetaSquared, ::fmaf(thetaSquared, ::fmaf(thetaSquared,
            -1.0f / 40320.0f, 1.0f / 720.0f), -1.0f / 24.0f), 0.5f);
        c = ::fmaf(thetaSquared, ::fmaf(thetaSquared, ::fmaf(thetaSquared,
            -1.0f / 362880.0f, 1.0f / 5040.0f), -1.0f / 120.0f), 1.0f / 6.0f);
    }
    else
    {
        const float theta = ::sqrtf(thetaSquared);
        a = ::sinf(theta) / theta;
        b = (1.0f - ::cosf(theta)) / thetaSquared;
        c = (1.0f - a) / thetaSquared;
    }
}

// Rotation matrices are column-major. Omega is a FIXED chart coordinate:
// R(omega)=Rbase Exp(omega), never an accumulated sequence of tangent steps.
// Output must not alias base. Recenter by adopting R as base and resetting H.
TMBFGS_HD inline void RotateRight(const float* base, const float* omega, float* output)
{
    const float thetaSquared = ::fmaf(omega[0], omega[0],
        ::fmaf(omega[1], omega[1], omega[2] * omega[2]));
    float a, b, c;
    RotationCoefficients(thetaSquared, a, b, c);
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
        {
            float value = 0.0f;
            for (int k = 0; k < 3; ++k)
            {
                float skew = 0.0f;
                if (k == 0 && column == 1) skew = -omega[2];
                if (k == 0 && column == 2) skew = omega[1];
                if (k == 1 && column == 0) skew = omega[2];
                if (k == 1 && column == 2) skew = -omega[0];
                if (k == 2 && column == 0) skew = -omega[1];
                if (k == 2 && column == 1) skew = omega[0];
                const float identity = k == column ? 1.0f : 0.0f;
                const float square = ::fmaf(omega[k], omega[column], -identity * thetaSquared);
                const float exponential = ::fmaf(b, square, ::fmaf(a, skew, identity));
                value = ::fmaf(base[row + 3 * k], exponential, value);
            }
            output[row + 3 * column] = value;
        }
}

// The model evaluator differentiates R Exp(delta), giving right-tangent g.
// Exp(omega+domega)=Exp(omega) Exp(Jr(omega)domega)+O(domega^2),
// so the fixed-chart gradient is Jr(omega)^T g, with Jr^T=I+b[omega]+c[omega]^2.
// Aliasing rightGradient and chartGradient is supported.
TMBFGS_HD inline void ChartGradient(const float* omega, const float* rightGradient, float* chartGradient)
{
    const float thetaSquared = ::fmaf(omega[0], omega[0],
        ::fmaf(omega[1], omega[1], omega[2] * omega[2]));
    float a, b, c;
    RotationCoefficients(thetaSquared, a, b, c);
    const float dot = ::fmaf(omega[0], rightGradient[0],
        ::fmaf(omega[1], rightGradient[1], omega[2] * rightGradient[2]));
    const float cross[3] = {
        ::fmaf(omega[1], rightGradient[2], -omega[2] * rightGradient[1]),
        ::fmaf(omega[2], rightGradient[0], -omega[0] * rightGradient[2]),
        ::fmaf(omega[0], rightGradient[1], -omega[1] * rightGradient[0])
    };
    for (int i = 0; i < 3; ++i)
        chartGradient[i] = ::fmaf(c, ::fmaf(omega[i], dot, -thetaSquared * rightGradient[i]),
            ::fmaf(b, cross[i], rightGradient[i]));
}

// Coordinates q=(position/pixel, omega*radius/pixel). Gradients passed in are
// in Angstrom and right-tangent radians; sign is unchanged (use -Z to minimize).
TMBFGS_HD inline void ScaledGradient(const float* physicalGradient, const float* omega,
                                   float pixel, float radius, float* output)
{
    ChartGradient(omega, physicalGradient + 3, output + 3);
    for (int i = 0; i < 3; ++i)
    {
        output[i] = physicalGradient[i] * pixel;
        output[i + 3] *= pixel / radius;
    }
}

TMBFGS_HD inline bool ProperRotation(const float* r)
{
    for (int i = 0; i < 9; ++i) if (!Finite(r[i])) return false;
    const float determinant = r[0] * ::fmaf(r[4], r[8], -r[7] * r[5]) -
        r[3] * ::fmaf(r[1], r[8], -r[7] * r[2]) + r[6] * ::fmaf(r[1], r[5], -r[4] * r[2]);
    if (::fabsf(determinant - 1.0f) > 2.0e-3f) return false;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            float dot = 0.0f;
            for (int row = 0; row < 3; ++row) dot = ::fmaf(r[row + a * 3], r[row + b * 3], dot);
            if (::fabsf(dot - (a == b ? 1.0f : 0.0f)) > 2.0e-3f) return false;
        }
    return true;
}

// Modified Gram-Schmidt first two columns, then their cross product. Call only
// on a previously valid rotation; reject degenerate/nonfinite input unchanged.
TMBFGS_HD inline bool OrthonormalizeRotation(float* r)
{
    for (int i = 0; i < 9; ++i) if (!Finite(r[i])) return false;
    float first[3], second[3];
    float norm = 0.0f;
    for (int i = 0; i < 3; ++i) norm = ::fmaf(r[i], r[i], norm);
    if (!(norm > FLT_MIN) || !Finite(norm)) return false;
    const float inverseNorm = 1.0f / ::sqrtf(norm);
    float projection = 0.0f;
    for (int i = 0; i < 3; ++i)
    {
        first[i] = r[i] * inverseNorm;
        projection = ::fmaf(first[i], r[i + 3], projection);
    }
    norm = 0.0f;
    for (int i = 0; i < 3; ++i)
    {
        second[i] = ::fmaf(-projection, first[i], r[i + 3]);
        norm = ::fmaf(second[i], second[i], norm);
    }
    if (!(norm > FLT_MIN) || !Finite(norm)) return false;
    const float inverseSecondNorm = 1.0f / ::sqrtf(norm);
    for (int i = 0; i < 3; ++i)
    {
        r[i] = first[i];
        r[i + 3] = second[i] * inverseSecondNorm;
    }
    r[6] = ::fmaf(r[1], r[5], -r[2] * r[4]);
    r[7] = ::fmaf(r[2], r[3], -r[0] * r[5]);
    r[8] = ::fmaf(r[0], r[4], -r[1] * r[3]);
    return true;
}

// Inverse Hessians use row-major storage; updates retain exact symmetry.
TMBFGS_HD inline void ResetInverse(float* h, float scale = 1.0f)
{
    if (!(scale > 0.0f) || !Finite(scale)) scale = 1.0f;
    for (int i = 0; i < 36; ++i) h[i] = i / 6 == i % 6 ? scale : 0.0f;
}

TMBFGS_HD inline bool Direction(const float* h, const float* gradient, float* direction)
{
    for (int row = 0; row < Parameters; ++row)
    {
        float value = 0.0f;
        for (int col = 0; col < Parameters; ++col) value = ::fmaf(h[row * 6 + col], gradient[col], value);
        direction[row] = -value;
        if (!Finite(direction[row])) return false;
    }
    const float directionalDerivative = Dot6(gradient, direction);
    return Finite(directionalDerivative) && directionalDerivative < 0.0f;
}

TMBFGS_HD inline float CandidateEntry(float original, float si, float sj, float hi, float hj,
                                    float inverseCurvature, float correction)
{
    // h_i=(H*y)_i, correction=1+(y^T*H*y)/(y^T*s). Keep the rank-two
    // correction grouped to reduce cancellation relative to three large sums.
    return ::fmaf(inverseCurvature,
        ::fmaf(si, ::fmaf(correction, sj, -hj), -hi * sj), original);
}

// s is the actual accepted displacement in fixed, scaled chart coordinates;
// y is the difference of minimization gradients in those SAME coordinates.
// Reject unreliable curvature and any update losing positive definiteness in
// FP32, leaving H untouched. Work holds H*y[6] and a Cholesky factor[36].
TMBFGS_HD inline bool UpdateInverse(float* h, const float* s, const float* y, float* work,
                                  float relativeThreshold = 1.0e-3f)
{
    const float ys = Dot6(y, s), ss = Dot6(s, s), yy = Dot6(y, y);
    if (!Finite(ys) || !Finite(ss) || !Finite(yy) ||
        !(ss > FLT_MIN) || !(yy > FLT_MIN) || !(ys > FLT_MIN)) return false;
    // Two roots avoid overflowing ss*yy for finite vectors.
    const float relativeFloor = relativeThreshold * ::sqrtf(ss) * ::sqrtf(yy);
    if (!(ys > relativeFloor)) return false;
    for (int row = 0; row < Parameters; ++row)
    {
        float value = 0.0f;
        for (int col = 0; col < Parameters; ++col) value = ::fmaf(h[row * 6 + col], y[col], value);
        work[row] = value;
        if (!Finite(value)) return false;
    }
    const float yhy = Dot6(y, work), inverseCurvature = 1.0f / ys;
    const float correction = ::fmaf(yhy, inverseCurvature, 1.0f);
    if (!(yhy > 0.0f) || !Finite(correction) || !Finite(inverseCurvature)) return false;
    float* factor = work + Parameters;
    for (int row = 0; row < Parameters; ++row)
        for (int col = 0; col <= row; ++col)
        {
            float value = CandidateEntry(h[row * 6 + col], s[row], s[col], work[row], work[col],
                inverseCurvature, correction);
            if (!Finite(value)) return false;
            for (int k = 0; k < col; ++k)
                value = ::fmaf(-factor[row * 6 + k], factor[col * 6 + k], value);
            if (row == col)
            {
                if (!(value > FLT_MIN) || !Finite(value)) return false;
                factor[row * 6 + col] = ::sqrtf(value);
            }
            else
            {
                factor[row * 6 + col] = value / factor[col * 6 + col];
                if (!Finite(factor[row * 6 + col])) return false;
            }
        }
    for (int row = 0; row < Parameters; ++row)
        for (int col = 0; col <= row; ++col)
        {
            const float value = CandidateEntry(h[row * 6 + col], s[row], s[col], work[row], work[col],
                inverseCurvature, correction);
            h[row * 6 + col] = h[col * 6 + row] = value;
        }
    return true;
}
}

#undef TMBFGS_HD
#endif
