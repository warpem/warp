// GPU-free test using Warp's actual spline coefficient construction and value evaluator.
// clang++ -std=c++14 -O2 NativeAcceleration/tests/EinsplineGradientTests.cpp NativeAcceleration/src/einspline/bspline_create.cpp -o /tmp/warp-spline-test && /tmp/warp-spline-test
#include "../src/einspline/bspline.h"
#include "../include/EinsplineGradient.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

float ExistingValue(void* spline, int dimensions, const float p[3])
{
    float result = 0;
    switch (dimensions)
    {
        case 64: eval_UBspline_3d_s((UBspline_3d_s*)spline, p[2], p[1], p[0], &result); break;
        case 8: eval_UBspline_2d_s((UBspline_2d_s*)spline, p[1], p[0], &result); break;
        case 16: eval_UBspline_2d_s((UBspline_2d_s*)spline, p[2], p[0], &result); break;
        case 32: eval_UBspline_2d_s((UBspline_2d_s*)spline, p[2], p[1], &result); break;
        case 1: eval_UBspline_1d_s((UBspline_1d_s*)spline, p[0], &result); break;
        case 2: eval_UBspline_1d_s((UBspline_1d_s*)spline, p[1], &result); break;
        case 4: eval_UBspline_1d_s((UBspline_1d_s*)spline, p[2], &result); break;
    }
    return result;
}

int main()
{
    const int axisMasks[] = {1, 2, 4, 3, 5, 6, 7};
    const int dimensionSets[] = {1, 2, 4, 8, 16, 32, 64};
    const float slopes[] = {1.25f, -2.0f, 0.7f};
    std::mt19937 random(391);
    std::uniform_real_distribution<float> distribution(-0.05f, 1.05f);
    for (int n = 0; n < 7; ++n)
        for (bool nonlinear : {false, true})
            for (bool margins : {false, true})
            {
                const int mask = axisMasks[n], dimensions = dimensionSets[n];
                const int counts[3] = {mask & 1 ? 5 : 1, mask & 2 ? 4 : 1, mask & 4 ? 6 : 1};
                Ugrid grids[3] = {};
                for (int axis = 0; axis < 3; ++axis)
                {
                    grids[axis].start = margins ? 0.08 + 0.03 * axis : 0;
                    grids[axis].end = 1 - grids[axis].start;
                    grids[axis].num = counts[axis];
                }
                std::vector<float> data(counts[0] * counts[1] * counts[2]);
                for (int z = 0; z < counts[2]; ++z)
                    for (int y = 0; y < counts[1]; ++y)
                        for (int x = 0; x < counts[0]; ++x)
                        {
                            const int indices[3] = {x, y, z};
                            float value = 0.3f;
                            for (int axis = 0; axis < 3; ++axis)
                            {
                                if (!(mask & (1 << axis))) continue;
                                const double p = grids[axis].start +
                                    indices[axis] * (grids[axis].end - grids[axis].start) / (counts[axis] - 1);
                                value += nonlinear ? (float)std::sin(p * (2 + axis)) : slopes[axis] * p;
                            }
                            if (nonlinear) value += 0.02f * x * y * z;
                            data[(z * counts[1] + y) * counts[0] + x] = value;
                        }
                BCtype_s natural = {NATURAL, NATURAL, 0, 0};
                void* spline = NULL;
                switch (dimensions)
                {
                    case 64: spline = create_UBspline_3d_s(grids[2], grids[1], grids[0], natural, natural, natural, data.data()); break;
                    case 8: spline = create_UBspline_2d_s(grids[1], grids[0], natural, natural, data.data()); break;
                    case 16: spline = create_UBspline_2d_s(grids[2], grids[0], natural, natural, data.data()); break;
                    case 32: spline = create_UBspline_2d_s(grids[2], grids[1], natural, natural, data.data()); break;
                    case 1: spline = create_UBspline_1d_s(grids[0], natural, data.data()); break;
                    case 2: spline = create_UBspline_1d_s(grids[1], natural, data.data()); break;
                    case 4: spline = create_UBspline_1d_s(grids[2], natural, data.data()); break;
                }

                for (int sample = 0; sample < 100; ++sample)
                {
                    float p[3] = {distribution(random), distribution(random), distribution(random)};
                    float value, gradient[3];
                    warp_einspline::Evaluate(spline, dimensions, p, value, gradient);
                    assert(std::abs(value - ExistingValue(spline, dimensions, p)) < 1e-5f);
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        if (!(mask & (1 << axis)))
                        {
                            assert(gradient[axis] == 0);
                            continue;
                        }
                        if (!nonlinear)
                            assert(std::abs(gradient[axis] - slopes[axis]) < 2e-5f);
                        const float original = p[axis], step = 0.001f;
                        p[axis] = original + step;
                        const float plus = ExistingValue(spline, dimensions, p);
                        p[axis] = original - step;
                        const float minus = ExistingValue(spline, dimensions, p);
                        p[axis] = original;
                        const float numerical = (plus - minus) / (2 * step);
                        assert(std::abs(gradient[axis] - numerical) < 0.002f * std::max(1.0f, std::abs(numerical)));
                    }
                }
                destroy_Bspline(spline);
            }
    std::cout << "Spline values and analytic gradients match all seven axis layouts, margins, and extrapolation.\n";
}
