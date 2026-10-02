// CUDA-free validation of the exact math compiled into TemplateMatchRefine.cu:
// c++ -std=c++14 -O2 -I NativeAcceleration/include \
//     NativeAcceleration/tests/TemplateMatchRefineMathTests.cpp -o /tmp/template_match_math_test
// /tmp/template_match_math_test
#include "TemplateMatchRefineMath.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace warp_template_match;

struct Grid
{
    int dim;
    std::vector<Complex<double>> values;
    explicit Grid(int size) : dim(size), values(size_t(size) * size * (size / 2 + 1))
    {
        for (int z = 0; z < size; ++z)
            for (int y = 0; y < size; ++y)
                for (int x = 0; x <= size / 2; ++x)
                    values[(size_t(z) * size + y) * (size / 2 + 1) + x] =
                        Complex<double>(std::sin(x * .37 + y * .13 + z * .19),
                                        std::cos(x * .23 - y * .41 + z * .11));
    }
    Complex<double> operator()(int x, int y, int z) const
    { return values[(size_t(z) * dim + y) * (dim / 2 + 1) + x]; }
};

static void Near(double actual, double expected, double tolerance, const char* label)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * (1 + std::abs(expected)))
    {
        std::cerr << label << ": " << actual << " != " << expected << '\n';
        std::exit(1);
    }
}

static void CheckInterpolation(const Grid& grid, double x, double y, double z)
{
    const double point[] = {x, y, z};
    const Sample<double> sampled = Interpolate<double>(grid, grid.dim, x, y, z);
    const double step = 1e-6;
    for (int axis = 0; axis < 3; ++axis)
    {
        double plus[] = {point[0], point[1], point[2]};
        double minus[] = {point[0], point[1], point[2]};
        plus[axis] += step;
        minus[axis] -= step;
        const Complex<double> a = Interpolate<double>(grid, grid.dim, plus[0], plus[1], plus[2]).value;
        const Complex<double> b = Interpolate<double>(grid, grid.dim, minus[0], minus[1], minus[2]).value;
        Near(sampled.gradient[axis].re, (a.re - b.re) / (2 * step), 2e-8, "interpolation real gradient");
        Near(sampled.gradient[axis].im, (a.im - b.im) / (2 * step), 2e-8, "interpolation imaginary gradient");
    }
    const Sample<double> opposite = Interpolate<double>(grid, grid.dim, -x, -y, -z);
    Near(sampled.value.re, opposite.value.re, 1e-12, "Hermitian real");
    Near(sampled.value.im, -opposite.value.im, 1e-12, "Hermitian imaginary");
}

struct Pose
{
    double matrix[9] = {.71, -.29, .33, .23, .82, -.41, 0, 0, 1};
    double matrixDerivatives[54] = {};
    double shift[2] = {.37, -.48};
    double shiftDerivatives[12] = {};
    double beta = .017;
    double betaDerivatives[6] = {};
    Pose()
    {
        // Mixed derivatives exercise every chain-rule contribution together.
        for (int p = 0; p < 6; ++p)
        {
            for (int j = 0; j < 9; ++j)
                matrixDerivatives[p * 9 + j] = .023 * std::sin(1 + p * 9 + j);
            shiftDerivatives[p * 2] = .13 * std::cos(p + 1);
            shiftDerivatives[p * 2 + 1] = -.17 * std::sin(p + 1);
            betaDerivatives[p] = .0013 * (p + 1);
        }
    }
    Pose Perturbed(int p, double delta) const
    {
        Pose other = *this;
        for (int j = 0; j < 9; ++j) other.matrix[j] += delta * matrixDerivatives[p * 9 + j];
        for (int j = 0; j < 2; ++j) other.shift[j] += delta * shiftDerivatives[p * 2 + j];
        other.beta += delta * betaDerivatives[p];
        return other;
    }
};

static void Model(const Grid& grid, const Pose& pose, int x, int y,
                  Complex<double>& model, Complex<double>* derivatives)
{
    ModelAndDerivatives<double>(grid, grid.dim, 16, x, y,
        pose.matrix, pose.matrixDerivatives, pose.shift, pose.shiftDerivatives,
        pose.beta, pose.betaDerivatives, .63, -.27, model, derivatives, .73);
}

static void CheckModel(const Grid& grid, Pose pose, int x, int y)
{
    Complex<double> model, derivatives[6], scratch[6];
    Model(grid, pose, x, y, model, derivatives);
    const Complex<double> observed(.39, -.83);
    const double weight = 2.7, step = 1e-6;
    for (int p = 0; p < 6; ++p)
    {
        Complex<double> plus, minus;
        Model(grid, pose.Perturbed(p, step), x, y, plus, scratch);
        Model(grid, pose.Perturbed(p, -step), x, y, minus, scratch);
        Near(derivatives[p].re, (plus.re - minus.re) / (2 * step), 3e-8, "model real gradient");
        Near(derivatives[p].im, (plus.im - minus.im) / (2 * step), 3e-8, "model imaginary gradient");
        const double analyticCross = weight * Inner(observed, derivatives[p]);
        const double analyticPower = 2 * weight * Inner(model, derivatives[p]);
        const double numericalCross = weight * (Inner(observed, plus) - Inner(observed, minus)) / (2 * step);
        const double numericalPower = weight * (Inner(plus, plus) - Inner(minus, minus)) / (2 * step);
        Near(analyticCross, numericalCross, 3e-8, "cross gradient");
        Near(analyticPower, numericalPower, 3e-8, "power gradient");
        Near(analyticCross - .5 * analyticPower, numericalCross - .5 * numericalPower, 5e-8, "log LR gradient");
    }
}

struct FloatGrid
{
    const Grid* grid;
    Complex<float> operator()(int x, int y, int z) const
    {
        const auto value = (*grid)(x, y, z);
        return Complex<float>(float(value.re), float(value.im));
    }
};

static void CheckFloatModel(const Grid& grid)
{
    const Pose pose;
    float matrix[9], dm[54], shift[2], ds[12], db[6];
    for (int j = 0; j < 9; ++j) matrix[j] = float(pose.matrix[j]);
    for (int j = 0; j < 54; ++j) dm[j] = float(pose.matrixDerivatives[j]);
    for (int j = 0; j < 2; ++j) shift[j] = float(pose.shift[j]);
    for (int j = 0; j < 12; ++j) ds[j] = float(pose.shiftDerivatives[j]);
    for (int j = 0; j < 6; ++j) db[j] = float(pose.betaDerivatives[j]);
    const FloatGrid fetch = {&grid};
    Complex<float> model, derivatives[6];
    ModelAndDerivatives<float>(fetch, grid.dim, 16, 3, 2, matrix, dm, shift, ds,
        float(pose.beta), db, .63f, -.27f, model, derivatives, .73f);
    Complex<double> reference, referenceDerivatives[6];
    Model(grid, pose, 3, 2, reference, referenceDerivatives);
    Near(model.re, reference.re, 2e-6, "FP32 model real");
    Near(model.im, reference.im, 2e-6, "FP32 model imaginary");
    for (int p = 0; p < 6; ++p)
    {
        Near(derivatives[p].re, referenceDerivatives[p].re, 2e-6, "FP32 real gradient");
        Near(derivatives[p].im, referenceDerivatives[p].im, 2e-6, "FP32 imaginary gradient");
    }
}

int main()
{
    const Grid grid(35); // Odd oversampled dimension matches Warp's projector.
    CheckInterpolation(grid, 3.27, -2.41, 4.63);
    CheckInterpolation(grid, -3.27, 2.41, -4.63);
    CheckInterpolation(grid, .27, -.41, -.63); // Wrapped y/z neighbours.
    CheckInterpolation(grid, -.27, -.41, -.63);
    const auto outside = Interpolate<double>(grid, grid.dim, 19., 2., 3.);
    Near(Inner(outside.value, outside.value), 0, 0, "outside projector support");

    Pose pose;
    CheckModel(grid, pose, 3, 2);
    for (double& value : pose.matrix) value = -value;
    CheckModel(grid, pose, 3, 2); // Reflected model and gradients.
    CheckModel(grid, Pose(), 2, -3);
    CheckFloatModel(grid);
    for (int y = -8; y <= 8; ++y)
        for (int x = 0; x <= 8; ++x)
        {
            const bool keep = IndependentSample(x, y, 16, 8);
            if ((x == 0 && y <= 0) || x == 8 || std::abs(y) == 8)
                Near(keep ? 1 : 0, 0, 0, "independent RFFT exclusions");
        }
    // For complex-noise power P, w=2/P. Analytic null score variance is Q;
    // a deterministic quadrature ensemble checks the real/imag convention.
    const Complex<double> target(.8, -.6);
    const double componentVariance = 2., w = 1. / componentVariance;
    const double amplitude = std::sqrt(2 * componentVariance);
    const Complex<double> noise[] = {{amplitude, 0}, {-amplitude, 0}, {0, amplitude}, {0, -amplitude}};
    double mean = 0, variance = 0;
    for (const auto& n : noise) { const double cross = w * Inner(n, target); mean += cross / 4; variance += cross * cross / 4; }
    Near(mean, 0, 1e-12, "null cross mean");
    Near(variance, w * Inner(target, target), 1e-12, "null cross variance Q");
    std::cout << "Template matching interpolation, Hermitian, phase/CTF, six-parameter gradients, and noise-convention tests passed.\n";
}
