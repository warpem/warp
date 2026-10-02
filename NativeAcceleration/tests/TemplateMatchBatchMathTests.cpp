// Complementary independent checks of score scaling and six-dimensional trust geometry.
// c++ -std=c++14 -O2 -I NativeAcceleration/include \
//   NativeAcceleration/tests/TemplateMatchBatchMathTests.cpp -o /tmp/tm_batch_math_test
#include "TemplateMatchRefineBatch.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace warp_template_match_batch;

static void Check(bool condition, const char* label)
{
    if (!condition) { std::cerr << label << '\n'; std::exit(1); }
}

static void Near(double actual, double expected, double tolerance, const char* label)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * (1 + std::abs(expected)))
    {
        std::cerr.precision(17);
        std::cerr << label << ": " << actual << " != " << expected << '\n';
        std::exit(1);
    }
}

using Statistics = std::array<double, 35>;
struct Normal { double z, gradient[6], hessian[21]; };

static Statistics Accumulate(double modelScale, double dataScale, double weightScale)
{
    Statistics result{};
    for (int sample = 0; sample < 29; ++sample)
    {
        const double m0 = 1 + .3 * std::cos(.31 * sample);
        const double m = modelScale * m0;
        const double data = dataScale * (.8 * m0 + .19 * std::sin(.73 * sample + .2));
        const double weight = weightScale * (.4 + .03 * sample);
        double jacobian[6];
        for (int j = 0; j < 6; ++j)
            jacobian[j] = modelScale * (j == 5 ? m0 : std::sin((sample + .7) * (j + .3)));
        result[0] += weight * data * m;
        result[1] += weight * m * m;
        for (int row = 0; row < 6; ++row)
        {
            result[2 + row] += weight * data * jacobian[row];
            result[8 + row] += weight * m * jacobian[row];
            for (int col = row; col < 6; ++col)
                result[14 + UpperIndex(row, col)] += weight * jacobian[row] * jacobian[col];
        }
    }
    return result;
}

static Normal Materialize(const Statistics& stats)
{
    Normal n{};
    Check(SignedNormal(stats.data(), n.z, n.gradient, n.hessian), "valid normal must materialize");
    return n;
}

static void CompareNormal(const Normal& actual, const Normal& original, double zFactor, double hFactor)
{
    Near(actual.z, original.z * zFactor, 2e-12, "score scaling");
    for (int j = 0; j < 6; ++j) Near(actual.gradient[j], original.gradient[j] * zFactor, 3e-12, "gradient scaling");
    for (int j = 0; j < 21; ++j) Near(actual.hessian[j], original.hessian[j] * hFactor, 3e-12, "curvature scaling");
}

static void ScalingIdentities()
{
    for (double sign : {1., -1.})
    {
        const Statistics base = Accumulate(1, sign, 1);
        const Normal reference = Materialize(base);
        for (double scale : {.013, .25, 4., 130.})
        {
            const Statistics model = Accumulate(scale, sign, 1);
            CompareNormal(Materialize(model), reference, 1, 1);
            Near(model[0] / model[1], base[0] / base[1] / scale, 2e-12, "model scaling of amplitude");
            const Statistics observation = Accumulate(1, sign * scale, 1);
            CompareNormal(Materialize(observation), reference, scale, scale);
            Near(observation[0] / observation[1], base[0] / base[1] * scale, 2e-12, "data scaling of amplitude");
            const Statistics noise = Accumulate(1, sign, scale);
            CompareNormal(Materialize(noise), reference, std::sqrt(scale), std::sqrt(scale));
            Near(noise[0] / noise[1], base[0] / base[1], 2e-12, "inverse variance scaling of amplitude");
        }
        Statistics repeated = base;
        for (double& value : repeated) value *= 7;
        CompareNormal(Materialize(repeated), reference, std::sqrt(7.), std::sqrt(7.));
        Near(repeated[0] / repeated[1], base[0] / base[1], 2e-12, "replicated tilt amplitude");
        // Parameter 5 changes only overall template amplitude. Profiling must remove it.
        Near(reference.gradient[5], 0, 2e-12, "amplitude nuisance gradient");
        for (int j = 0; j < 6; ++j) Near(reference.hessian[UpperIndex(j, 5)], 0, 2e-12, "amplitude nuisance curvature");
    }
}

static void ZeroScoreAndInvalidStatistics()
{
    Statistics base = Accumulate(1, 1, 1);
    // Subtract the exact fitted component from the data: c changes together with C.
    const double amplitude = base[0] / base[1];
    for (int j = 0; j < 6; ++j) base[2 + j] -= amplitude * base[8 + j];
    base[0] = 0;
    const Normal zero = Materialize(base);
    double gradientPower = 0;
    for (double value : zero.gradient) gradientPower += value * value;
    Check(gradientPower > 1e-4, "zero correlation must not erase a nonzero gradient");
    for (double z : {-1e-8, 0., 1e-8})
    {
        Statistics shifted = base;
        const double a = z / std::sqrt(base[1]);
        shifted[0] = z * std::sqrt(base[1]);
        for (int j = 0; j < 6; ++j) shifted[2 + j] += a * base[8 + j];
        const Normal nearZero = Materialize(shifted);
        Near(nearZero.z, z, 1e-14, "near-zero signed Z");
        for (int j = 0; j < 6; ++j) Near(nearZero.gradient[j], zero.gradient[j], 2e-12, "near-zero gradient continuity");
        for (int j = 0; j < 21; ++j) Near(nearZero.hessian[j], zero.hessian[j], 2e-12, "explicit curvature floor");
    }
    for (int index : {0, 1, 2, 8, 14})
    {
        Statistics invalid = base;
        invalid[index] = std::numeric_limits<double>::quiet_NaN();
        Normal output{};
        Check(!SignedNormal(invalid.data(), output.z, output.gradient, output.hessian), "nonfinite sufficient statistic must fail");
    }
    for (double power : {0., -1.})
    {
        Statistics invalid = base; invalid[1] = power;
        Normal output{};
        Check(!SignedNormal(invalid.data(), output.z, output.gradient, output.hessian), "nonpositive model power must fail");
    }
}

static std::array<double, 36> Basis()
{
    std::array<double, 36> q{};
    for (int j = 0; j < 6; ++j) q[j * 6 + j] = 1;
    // A deterministic product of plane rotations mixes every position and rotation axis.
    for (int a = 0; a < 5; ++a)
        for (int b = a + 1; b < 6; ++b)
        {
            const double angle = .13 + .09 * a - .04 * b;
            const double cosine = std::cos(angle), sine = std::sin(angle);
            for (int row = 0; row < 6; ++row)
            {
                const double x = q[row * 6 + a], y = q[row * 6 + b];
                q[row * 6 + a] = cosine * x - sine * y;
                q[row * 6 + b] = sine * x + cosine * y;
            }
        }
    return q;
}

static std::array<double, 6> Solve(const double* eigenvalues, const double* coefficients,
                                   double rotationRadius, double radius, double objectiveScale)
{
    const auto q = Basis();
    double gradient[6]{}, hessian[21]{};
    for (int row = 0; row < 6; ++row)
    {
        const double rowScale = row < 3 ? 1 : rotationRadius;
        for (int k = 0; k < 6; ++k) gradient[row] += objectiveScale * rowScale * q[row * 6 + k] * coefficients[k];
        for (int col = row; col < 6; ++col)
            for (int k = 0; k < 6; ++k)
                hessian[UpperIndex(row, col)] += objectiveScale * rowScale * (col < 3 ? 1 : rotationRadius) *
                    q[row * 6 + k] * eigenvalues[k] * q[col * 6 + k];
    }
    double matrix[36], vectors[36], values[6], rotated[6];
    std::array<double, 6> step{};
    Check(TrustStep(gradient, hessian, rotationRadius, radius, matrix, vectors, values, rotated, step.data()), "trust solver must succeed");
    return step;
}

static void TrustGeometryAndScale()
{
    const auto q = Basis();
    const double coefficients[6] = {.13, -.24, .03, -.16, .21, .12};
    for (bool indefinite : {false, true})
    {
        const double eigenvalues[6] = {indefinite ? -.35 : .08, .3, .7, 1.7, 3.8, 6.};
        const double radius = indefinite ? .7 : .24, rotationRadius = 7.;
        const auto step = Solve(eigenvalues, coefficients, rotationRadius, radius, 1);
        double eigenStep[6]{}, norm2 = 0, model = 0;
        for (int row = 0; row < 6; ++row)
        {
            const double metricStep = step[row] * (row < 3 ? 1 : rotationRadius);
            norm2 += metricStep * metricStep;
            for (int k = 0; k < 6; ++k) eigenStep[k] += q[row * 6 + k] * metricStep;
        }
        Near(std::sqrt(norm2), radius, 2e-8, "trust metric boundary");
        const double lambda = -coefficients[0] / eigenStep[0] - eigenvalues[0];
        Check(lambda >= std::max(0., -eigenvalues[0]), "KKT Hessian must be positive semidefinite");
        for (int k = 0; k < 6; ++k)
        {
            Near((eigenvalues[k] + lambda) * eigenStep[k], -coefficients[k], 2e-8, "full six-dimensional trust KKT");
            model += coefficients[k] * eigenStep[k] + .5 * eigenvalues[k] * eigenStep[k] * eigenStep[k];
        }
        Check(model < 0, "trust model must decrease");
        // Multiplying an objective by a positive constant must not alter its trust step.
        // This detects absolute brackets/tolerances that fail for weak signed-Z signals.
        for (double scale : {1e-12, 1e-6, 1e6, 1e12})
        {
            const auto scaled = Solve(eigenvalues, coefficients, rotationRadius, radius, scale);
            for (int j = 0; j < 6; ++j) Near(scaled[j], step[j], 3e-8, "trust objective-scale invariance");
        }
        // Re-express angular coordinates with a different radius while preserving the
        // same physical metric problem. Translational displacement must be unchanged.
        const auto rescaledRadius = Solve(eigenvalues, coefficients, 2 * rotationRadius, radius, 1);
        for (int j = 0; j < 6; ++j) Near(rescaledRadius[j] * (j < 3 ? 1 : 2), step[j], 2e-8, "rotation metric conversion");
    }
    const double positive[6] = {.08, .3, .7, 1.7, 3.8, 6.};
    const auto interior = Solve(positive, coefficients, 7., 20., 1);
    for (int row = 0; row < 6; ++row)
    {
        double expected = 0;
        for (int k = 0; k < 6; ++k) expected -= q[row * 6 + k] * coefficients[k] / positive[k];
        Near(interior[row], expected / (row < 3 ? 1 : 7), 3e-11, "unconstrained dense SPD solution");
    }
}

int main()
{
    ScalingIdentities();
    ZeroScoreAndInvalidStatistics();
    TrustGeometryAndScale();
    std::cout << "Batch score/amplitude/noise scaling, zero-score handling, and full trust geometry tests passed.\n";
}
