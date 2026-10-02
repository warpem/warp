// CUDA-free checks of the exact numerical routines used by the device optimizer.
// clang++ -std=c++14 -O2 -I NativeAcceleration/include \
//   NativeAcceleration/tests/TemplateMatchRefineBatchTests.cpp -o /tmp/tm_batch_test
#include "TemplateMatchRefineBatch.h"
#include <cassert>
#include <cstdio>

using namespace warp_template_match_batch;

void Near(double actual, double expected, double tolerance)
{
    if (!(std::fabs(actual - expected) <= tolerance))
    {
        std::fprintf(stderr, "%.17g != %.17g, tolerance %.3g\n", actual, expected, tolerance);
        assert(false);
    }
}

void TestRotation()
{
    const float initial[9] = {0, 1, 0, -1, 0, 0, 0, 0, 1};
    const double delta[3] = {0.03, -0.04, 0.02};
    float updated[9];
    RotateRight(initial, delta, updated);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            double dot = 0;
            for (int row = 0; row < 3; ++row) dot += updated[row + 3 * a] * double(updated[row + 3 * b]);
            Near(dot, a == b ? 1 : 0, 1e-7);
        }
    const double step = 1e-4;
    const double xStep[3] = {step, 0, 0};
    RotateRight(initial, xStep, updated);
    // Right rotation around local X changes column Y toward initial column Z.
    Near((updated[7] - initial[7]) / step, -initial[4], 1e-7);
    Near((updated[5] - initial[5]) / step, initial[8], 1e-7);
}

void TestEigenAndTrust()
{
    double matrix[36], original[36], vectors[36], values[6];
    // A dense positive-definite Gram matrix exercises all 36 entries, including
    // those that would be missed by a naive five-to-six CUDA lane conversion.
    for (int row = 0; row < 6; ++row)
        for (int col = 0; col < 6; ++col)
        {
            double value = row == col ? 0.2 : 0;
            for (int k = 0; k < 8; ++k)
                value += std::sin((row + 1) * (k + 2.3)) * std::sin((col + 1) * (k + 2.3));
            matrix[row * 6 + col] = original[row * 6 + col] = value;
        }
    assert(Eigen6(matrix, vectors, values));
    for (int row = 0; row < 6; ++row)
        for (int column = 0; column < 6; ++column)
        {
            double reconstructed = 0;
            for (int k = 0; k < 6; ++k) reconstructed += vectors[row * 6 + k] * values[k] * vectors[column * 6 + k];
            Near(reconstructed, original[row * 6 + column], 1e-12);
        }
    double gradient[6] = {0.8, -0.3, 0.2, -1.4, 0.6, 0.7}, hessian[21], rotated[6], step[6];
    for (int row = 0; row < 6; ++row)
        for (int col = row; col < 6; ++col) hessian[UpperIndex(row, col)] = original[row * 6 + col];
    assert(TrustStep(gradient, hessian, 5.0, 0.04, matrix, vectors, values, rotated, step));
    double norm2 = 0, prediction = 0;
    for (int row = 0; row < 6; ++row)
    {
        norm2 += step[row] * step[row] * (row < 3 ? 1.0 : 25.0);
        prediction -= gradient[row] * step[row];
        for (int col = 0; col < 6; ++col) prediction -= 0.5 * step[row] * hessian[UpperIndex(row, col)] * step[col];
    }
    Near(std::sqrt(norm2), 0.04, 1e-8);
    assert(prediction > 0);
    // Zero curvature still produces a bounded downhill gradient step.
    for (double& value : hessian) value = 0;
    assert(TrustStep(gradient, hessian, 5.0, 0.04, matrix, vectors, values, rotated, step));
    norm2 = 0;
    for (int row = 0; row < 6; ++row) norm2 += step[row] * step[row] * (row < 3 ? 1 : 25);
    Near(std::sqrt(norm2), 0.04, 1e-8);
}

constexpr int Samples = 17;
double ModelValue(int sample, const double* position)
{
    double result = std::cos(0.7 * sample) + 0.1;
    for (int parameter = 0; parameter < 6; ++parameter)
        result += position[parameter] * std::sin((sample + 1.2) * (parameter + 0.4));
    return result;
}

double Objective(const double* data, const double* position)
{
    double c = 0, p = 0;
    for (int sample = 0; sample < Samples; ++sample)
    {
        const double model = ModelValue(sample, position), weight = 0.1 + sample * 0.09;
        c += weight * data[sample] * model;
        p += weight * model * model;
    }
    return -c / std::sqrt(p);
}

void TestSignedNormal(double sign)
{
    double stats[35] = {}, data[Samples], position[6] = {}, gradient[6], hessian[21], z;
    for (int sample = 0; sample < Samples; ++sample)
    {
        const double model = ModelValue(sample, position), weight = 0.1 + sample * 0.09;
        data[sample] = sign * model + 0.13 * std::sin(0.3 + sample * 0.9);
        stats[0] += weight * data[sample] * model;
        stats[1] += weight * model * model;
        for (int row = 0; row < 6; ++row)
        {
            const double jacobian = std::sin((sample + 1.2) * (row + 0.4));
            stats[2 + row] += weight * data[sample] * jacobian;
            stats[8 + row] += weight * model * jacobian;
            for (int col = row; col < 6; ++col)
                stats[14 + UpperIndex(row, col)] += weight * jacobian * std::sin((sample + 1.2) * (col + 0.4));
        }
    }
    assert(SignedNormal(stats, z, gradient, hessian));
    Near(z, -Objective(data, position), 1e-13);
    const double epsilon = 1e-5;
    for (int parameter = 0; parameter < 6; ++parameter)
    {
        position[parameter] = epsilon;
        const double positive = Objective(data, position);
        position[parameter] = -epsilon;
        const double negative = Objective(data, position);
        position[parameter] = 0;
        Near(gradient[parameter], (positive - negative) / (2 * epsilon), 1e-9);
    }
    // Independently finite-difference the normalized model to verify the GN
    // curvature, including cross-parameter terms, with the scale held fixed.
    double residualJacobian[Samples][6];
    const double scale = std::fmax(std::fabs(z), CurvatureScaleFloor);
    for (int parameter = 0; parameter < 6; ++parameter)
    {
        double plus[Samples], minus[Samples], pp = 0, pm = 0;
        position[parameter] = epsilon;
        for (int sample = 0; sample < Samples; ++sample) { plus[sample] = ModelValue(sample, position); pp += (0.1 + sample * 0.09) * plus[sample] * plus[sample]; }
        position[parameter] = -epsilon;
        for (int sample = 0; sample < Samples; ++sample) { minus[sample] = ModelValue(sample, position); pm += (0.1 + sample * 0.09) * minus[sample] * minus[sample]; }
        position[parameter] = 0;
        for (int sample = 0; sample < Samples; ++sample)
            residualJacobian[sample][parameter] = -std::sqrt(scale) * (plus[sample] / std::sqrt(pp) - minus[sample] / std::sqrt(pm)) / (2 * epsilon);
    }
    for (int row = 0; row < 6; ++row)
        for (int col = row; col < 6; ++col)
        {
            double gram = 0;
            for (int sample = 0; sample < Samples; ++sample) gram += (0.1 + sample * 0.09) * residualJacobian[sample][row] * residualJacobian[sample][col];
            Near(hessian[UpperIndex(row, col)], gram, 1e-8);
        }
    if (sign < 0)
    {
        assert(z < 0);
        double norm = 0;
        for (double value : gradient) norm += value * value;
        assert(norm > 0.001); // Negative starts retain a useful gradient.
    }
}

int main()
{
    TestRotation();
    TestEigenAndTrust();
    TestSignedNormal(1.0);
    TestSignedNormal(-1.0);
    TestSignedNormal(0.0);
    std::puts("TemplateMatchRefineBatch portable math tests passed");
}
