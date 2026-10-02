// c++ -std=c++14 -O2 -I NativeAcceleration/include \
//   NativeAcceleration/tests/TemplateMatchBfgsTests.cpp -o /tmp/tm_bfgs_test
// The independent reference calculations deliberately use double precision.
#include "TemplateMatchBfgs.h"
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace warp_template_match_bfgs;

static void Check(bool condition, const char* label)
{
    if (!condition) { std::cerr << label << '\n'; std::exit(1); }
}

static void Near(double actual, double expected, double tolerance, const char* label)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * (1.0 + std::abs(expected)))
    {
        std::cerr.precision(17);
        std::cerr << label << ": " << actual << " != " << expected << '\n';
        std::exit(1);
    }
}

static void IntegerFrequencyMask()
{
    for (int box : {8, 10, 16, 17, 31, 64, 80, 255})
    {
        std::vector<float> cutoffs = {0.0f, 0.01f, 0.5f, 1.0f, 1.1f, 2.25f, 3.7f, box * .49f, box * .5f};
        for (int squared : {2, 5, 8, 13, 25, 41, 100, 257, 1000, 10001})
        {
            const float radius = std::sqrt(float(squared));
            cutoffs.push_back(std::nextafter(radius, 0.0f));
            cutoffs.push_back(radius);
            cutoffs.push_back(std::nextafter(radius, std::numeric_limits<float>::infinity()));
        }
        for (float cutoff : cutoffs)
        {
            const double squaredCutoff = double(cutoff) * double(cutoff);
            const auto limit = static_cast<unsigned long long>(std::floor(squaredCutoff));
            for (int y = -box / 2 - 1; y <= box / 2 + 1; ++y)
                for (int x = 0; x <= box / 2 + 1; ++x)
                {
                    const bool oldSelection = !(x == 0 && y <= 0) && x < box / 2 &&
                        y > -box / 2 && y < box / 2 &&
                        double(x) * x + double(y) * y <= squaredCutoff;
                    Check(IndependentSample(x, y, box, limit) == oldSelection,
                        "integer frequency mask must exactly match original double cutoff");
                }
        }
    }
    // Large integer coordinates must not overflow a 32-bit multiplication.
    const int box = 2000000000, x = 800000000, y = -700000000;
    const unsigned long long exact = 1130000000000000000ULL;
    Check(IndependentSample(x, y, box, exact), "64-bit integer mask includes exact large boundary");
    Check(!IndependentSample(x, y, box, exact - 1), "64-bit integer mask excludes outside large boundary");
}

static void QuaternionRotation(const double* omega, double* r)
{
    const double theta = std::hypot(std::hypot(omega[0], omega[1]), omega[2]);
    const double scale = theta == 0 ? 0.5 : std::sin(0.5 * theta) / theta;
    const double w = std::cos(0.5 * theta), x = scale * omega[0], y = scale * omega[1], z = scale * omega[2];
    const double values[9] = {
        1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
        2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x),
        2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
    };
    std::copy(values, values + 9, r);
}

static void ReferenceRotate(const float* base, const double* omega, double* r)
{
    double delta[9];
    QuaternionRotation(omega, delta);
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
        {
            r[row + 3 * col] = 0;
            for (int k = 0; k < 3; ++k) r[row + 3 * col] += base[row + 3 * k] * delta[k + 3 * col];
        }
}

static void RotationAccuracy()
{
    const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float baseOmega[3] = {0.4f, -0.3f, 0.7f};
    float base[9];
    RotateRight(identity, baseOmega, base);
    for (double size : {0.0, 1e-7, 1e-4, 0.09, 0.1, 0.3, 0.5, 1.0, 2.7})
    {
        const float omega[3] = {float(size * 0.8), float(size * -0.48), float(size * 0.36)};
        const double omega64[3] = {omega[0], omega[1], omega[2]};
        float actual[9]; double expected[9];
        RotateRight(base, omega, actual);
        ReferenceRotate(base, omega64, expected);
        for (int i = 0; i < 9; ++i) Near(actual[i], expected[i], 3e-7, "FP32 rotation versus quaternion reference");
        Check(ProperRotation(actual), "exponential must preserve proper rotation");
    }
    for (int iteration = 0; iteration < 2000; ++iteration)
    {
        const float omega[3] = {0.031f, -0.063f, 0.043f};
        float next[9];
        RotateRight(base, omega, next);
        Check(OrthonormalizeRotation(next), "recenter orthonormalization");
        Check(ProperRotation(next), "repeated recentering must preserve rotation");
        std::copy(next, next + 9, base);
    }
    float bad[9] = {1, 0, 0, 0, 1, 0, 0, 0, -1};
    Check(!ProperRotation(bad), "reflection is not a proper rotation");
    bad[8] = std::numeric_limits<float>::quiet_NaN();
    Check(!ProperRotation(bad), "nonfinite rotation must fail validation");
    const std::array<float, 9> preserved = {{bad[0], bad[1], bad[2], bad[3], bad[4], bad[5], bad[6], bad[7], bad[8]}};
    Check(!OrthonormalizeRotation(bad), "nonfinite rotation cannot be recentered");
    Check(std::memcmp(bad, preserved.data(), sizeof(bad)) == 0, "failed recenter must preserve input");
}

static const double Points[4][3] = {{1.0, -.3, .9}, {.2, 1.4, -.4}, {-.7, .1, 1.2}, {.6, -.8, -.5}};
static const double Targets[4][3] = {{.4, .1, 1.3}, {-.3, .9, .2}, {-.2, -.2, 1.6}, {.1, -1.2, -.3}};

static double PoseObjective(const float* base, const double* position, const double* omega, double* tangentGradient = nullptr)
{
    double r[9];
    ReferenceRotate(base, omega, r);
    if (tangentGradient) std::fill(tangentGradient, tangentGradient + 6, 0.0);
    double value = 0;
    for (int point = 0; point < 4; ++point)
    {
        double error[3], localError[3] = {};
        for (int row = 0; row < 3; ++row)
        {
            error[row] = position[row] - Targets[point][row];
            for (int col = 0; col < 3; ++col) error[row] += r[row + 3 * col] * Points[point][col];
            value += 0.5 * error[row] * error[row];
            if (tangentGradient) tangentGradient[row] += error[row];
        }
        if (tangentGradient)
        {
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) localError[row] += r[col + 3 * row] * error[col];
            for (int axis = 0; axis < 3; ++axis)
            {
                const int j = (axis + 1) % 3, k = (axis + 2) % 3;
                tangentGradient[3 + axis] += Points[point][j] * localError[k] - Points[point][k] * localError[j];
            }
        }
    }
    return value;
}

static void RotationChartGradient()
{
    const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float baseOmega[3] = {-.43f, .27f, .18f};
    float base[9];
    RotateRight(identity, baseOmega, base);
    const double position[3] = {.23, -.39, .14};
    for (double size : {0.0, 1e-5, .099, .101, .5, 1.0, 2.5})
    {
        const float omega[3] = {float(size * .8), float(size * -.48), float(size * .36)};
        double omega64[3] = {omega[0], omega[1], omega[2]}, tangent64[6];
        PoseObjective(base, position, omega64, tangent64);
        float tangent[6], gradient[6];
        for (int i = 0; i < 6; ++i) tangent[i] = float(tangent64[i]);
        const float pixel = 5.0f, radius = 65.0f;
        ScaledGradient(tangent, omega, pixel, radius, gradient);
        // Independent central differences perturb the scaled chart coordinates,
        // evaluating rotations through a double quaternion implementation.
        for (int axis = 0; axis < 6; ++axis)
        {
            double plusPosition[3], minusPosition[3], plusOmega[3], minusOmega[3];
            std::copy(position, position + 3, plusPosition); std::copy(position, position + 3, minusPosition);
            std::copy(omega64, omega64 + 3, plusOmega); std::copy(omega64, omega64 + 3, minusOmega);
            const double epsilon = 1e-5;
            if (axis < 3) { plusPosition[axis] += epsilon * pixel; minusPosition[axis] -= epsilon * pixel; }
            else { plusOmega[axis - 3] += epsilon * pixel / radius; minusOmega[axis - 3] -= epsilon * pixel / radius; }
            const double difference = (PoseObjective(base, plusPosition, plusOmega) -
                PoseObjective(base, minusPosition, minusOmega)) / (2 * epsilon);
            Near(gradient[axis], difference, 5e-7, "nonzero fixed-chart scaled gradient");
        }
        float aliased[3] = {tangent[3], tangent[4], tangent[5]}, separate[3];
        ChartGradient(omega, aliased, separate);
        ChartGradient(omega, aliased, aliased);
        for (int i = 0; i < 3; ++i) Near(aliased[i], separate[i], 0, "chart gradient permits in-place use");
    }
}

static void CheckSpd(const float* h)
{
    double factor[36] = {};
    for (int row = 0; row < 6; ++row)
        for (int col = 0; col <= row; ++col)
        {
            Near(h[row * 6 + col], h[col * 6 + row], 0, "inverse Hessian remains exactly symmetric");
            double value = h[row * 6 + col];
            for (int k = 0; k < col; ++k) value -= factor[row * 6 + k] * factor[col * 6 + k];
            if (row == col)
            {
                Check(std::isfinite(value) && value > 0, "independent inverse-Hessian Cholesky must remain positive");
                factor[row * 6 + col] = std::sqrt(value);
            }
            else factor[row * 6 + col] = value / factor[col * 6 + col];
        }
}

static std::array<float, 36> Quadratic()
{
    // Dense SPD matrix with known eigenvalues, built independently as Q D Q^T.
    const double v[6] = {1, 2, -.5, .3, -.8, 1.7}, eigenvalues[6] = {.25, .5, 1, 3, 9, 20};
    double norm = 0;
    for (double x : v) norm += x * x;
    double q[36];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) q[i * 6 + j] = (i == j ? 1 : 0) - 2 * v[i] * v[j] / norm;
    std::array<float, 36> a{};
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
        {
            double value = 0;
            for (int k = 0; k < 6; ++k) value += q[i * 6 + k] * eigenvalues[k] * q[j * 6 + k];
            a[i * 6 + j] = float(value);
        }
    return a;
}

static void Multiply(const float* a, const float* x, float* output)
{
    for (int i = 0; i < 6; ++i)
    {
        double value = 0;
        for (int j = 0; j < 6; ++j) value += a[i * 6 + j] * double(x[j]);
        output[i] = float(value);
    }
}

static void SecantAndCurvature()
{
    const auto a = Quadratic();
    float h[36], work[UpdateWorkspace];
    ResetInverse(h);
    std::mt19937 generator(941);
    std::uniform_real_distribution<float> random(-1, 1);
    for (int update = 0; update < 200; ++update)
    {
        float s[6], y[6], hy[6];
        for (float& value : s) value = random(generator);
        // Deliberately clip one component: the secant condition must use the
        // actual accepted displacement, even if the search direction differed.
        s[update % 6] = std::max(-0.03f, std::min(0.03f, s[update % 6]));
        Multiply(a.data(), s, y);
        Check(UpdateInverse(h, s, y, work), "well-conditioned positive curvature must be accepted");
        CheckSpd(h);
        Multiply(h, y, hy);
        for (int i = 0; i < 6; ++i) Near(hy[i], s[i], 6e-6, "BFGS secant condition with clipped displacement");
    }
    const std::array<float, 36> saved = [&]() { std::array<float, 36> out; std::copy(h, h + 36, out.begin()); return out; }();
    const float s[6] = {1, 0, 0, 0, 0, 0};
    for (int test = 0; test < 7; ++test)
    {
        float y[6] = {1, 0, 0, 0, 0, 0};
        if (test == 0) y[0] = -1;
        if (test == 1) y[0] = 0;
        if (test == 2) { y[0] = 1e-5f; y[1] = 1; }
        if (test == 3) y[1] = std::numeric_limits<float>::quiet_NaN();
        if (test == 4) y[0] = std::numeric_limits<float>::infinity();
        if (test == 5) y[0] = 1e-30f;
        if (test == 6) y[0] = std::numeric_limits<float>::max();
        Check(!UpdateInverse(h, s, y, work), "nonfinite or unreliable curvature must be skipped");
        Check(std::memcmp(h, saved.data(), sizeof(h)) == 0, "rejected update must leave inverse Hessian unchanged");
    }
    float gradient[6] = {1, -1, .5, .2, -.7, .3}, direction[6];
    Check(Direction(h, gradient, direction), "SPD Hessian gives a descent direction");
    float zero[6] = {};
    Check(!Direction(h, zero, direction), "zero direction is not a strict descent direction");
    // Positive curvature alone does not guarantee that a badly scaled update
    // remains SPD after FP32 cancellation. The transactional check must catch it.
    ResetInverse(h, 1e8f);
    std::array<float, 36> largeSaved;
    std::copy(h, h + 36, largeSaved.begin());
    Check(!UpdateInverse(h, s, s, work), "roundoff-induced non-SPD candidate must be rejected");
    Check(std::memcmp(h, largeSaved.data(), sizeof(h)) == 0, "failed candidate factorization must preserve H");
    ResetInverse(h, -1);
    for (int i = 0; i < 36; ++i) Near(h[i], i / 6 == i % 6 ? 1 : 0, 0, "invalid initial scale uses identity");
}

static void QuadraticMinimization()
{
    const auto a = Quadratic();
    float h[36], work[UpdateWorkspace], x[6] = {2, -.7f, 1.1f, -1.8f, .4f, .9f};
    ResetInverse(h);
    for (int iteration = 0; iteration < 12; ++iteration)
    {
        float gradient[6], direction[6], ad[6], step[6], y[6];
        Multiply(a.data(), x, gradient);
        if (Dot6(x, x) < 1e-12f) break;
        Check(Direction(h, gradient, direction), "quadratic direction must descend");
        Multiply(a.data(), direction, ad);
        const float alpha = -Dot6(gradient, direction) / Dot6(direction, ad);
        const double oldObjective = .5 * Dot6(x, gradient);
        for (int i = 0; i < 6; ++i)
        {
            const float next = std::fma(alpha, direction[i], x[i]);
            step[i] = next - x[i];
            x[i] = next;
        }
        Multiply(a.data(), step, y);
        Check(UpdateInverse(h, step, y, work), "quadratic BFGS update");
        CheckSpd(h);
        Multiply(a.data(), x, gradient);
        Check(.5 * Dot6(x, gradient) < oldObjective, "exact line search lowers quadratic objective");
    }
    Check(Dot6(x, x) < 1e-10f, "FP32 BFGS recovers dense six-dimensional quadratic minimizer");
}

int main()
{
    IntegerFrequencyMask();
    RotationAccuracy();
    RotationChartGradient();
    SecantAndCurvature();
    QuadraticMinimization();
    std::cout << "FP32 BFGS math tests passed\n";
}
