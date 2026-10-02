using System;
using System.Collections.Generic;
using Warp;
using Warp.Tools;
using Xunit;
using Complex = System.Numerics.Complex;

namespace Tests;

/// <summary>Run with WARP_RUN_CUDA_TESTS=1 and the rebuilt native library on the search path.</summary>
public sealed class TemplateMatchCudaFactAttribute : FactAttribute
{
    public TemplateMatchCudaFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("WARP_RUN_CUDA_TESTS") != "1")
            Skip = "Set WARP_RUN_CUDA_TESTS=1 on a CUDA host with rebuilt NativeAcceleration.";
    }
}

public class TemplateMatchRefineCudaTests
{
    [TemplateMatchCudaFact]
    public void PersistentScorerMatchesWarpProjectorAndAnalyticGradients()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            const int box = 16, dim = 35, views = 2;
            const float oversampling = 2, cutoff = 6.7f;
            int elements = box * (box / 2 + 1);
            var allocations = new List<IntPtr>();
            var textures = new ulong[2];
            var arrays = new ulong[2];
            IntPtr context = IntPtr.Zero;
            IntPtr Upload(float[] values)
            {
                IntPtr pointer = GPU.MallocDeviceFromHost(values, values.Length);
                allocations.Add(pointer);
                return pointer;
            }

            try
            {
                float[] volume = new float[(dim / 2 + 1) * dim * dim * 2];
                for (int i = 0; i < volume.Length / 2; i++)
                {
                    volume[i * 2] = MathF.Sin(i * .0173f);
                    volume[i * 2 + 1] = MathF.Cos(i * .0137f);
                }
                // Match Projector.PutTexturesOnDevice: legacy projection kernels fetch
                // integer coordinates through point sampling, then interpolate manually.
                // Linear textures would blend those fetches before the manual lerps.
                GPU.CreateTexture3DComplex(Upload(volume), new int3(dim / 2 + 1, dim, dim), textures, arrays, false);

                float[] observed = new float[elements * views * 2];
                float[] ctfBase = new float[elements * views];
                float[] quadrature = new float[elements * views];
                float[] inverseNoise = new float[elements * views];
                float[] phaseRadiusSquared = new float[elements * views];
                for (int i = 0; i < ctfBase.Length; i++)
                {
                    observed[i * 2] = MathF.Sin(i * .17f);
                    observed[i * 2 + 1] = MathF.Cos(i * .11f);
                    ctfBase[i] = .7f * MathF.Sin(i * .07f);
                    quadrature[i] = -.7f * MathF.Cos(i * .07f);
                    inverseNoise[i] = .8f + .2f * MathF.Cos(i * .03f);
                    int fx = (i % elements) % (box / 2 + 1), fy = (i % elements) / (box / 2 + 1);
                    if (fy > box / 2) fy -= box;
                    // A view-dependent anisotropic radius differs from raw k^2,
                    // detecting accidental recomputation inside the native scorer.
                    phaseRadiusSquared[i] = (1 + .1f * (i / elements)) *
                        (1.13f * fx * fx + .83f * fy * fy + .1f * fx * fy);
                }
                Assert.Equal(0, GPU.TemplateMatchRefineCreate(textures[0], textures[1], dim, box, views,
                    Upload(observed), Upload(ctfBase), Upload(quadrature), Upload(inverseNoise), Upload(phaseRadiusSquared), out context));

                float[] matrices = new float[views * 9], matrixDerivatives = new float[views * 54];
                float[] shifts = { .37f, -.48f, -.29f, .61f }, shiftDerivatives = new float[views * 12];
                float[] beta = { .017f, -.013f }, betaDerivatives = new float[views * 6];
                float[] eulers = new float[views * 3], nativeShifts = new float[views * 3];
                for (int view = 0; view < views; view++)
                {
                    // Test the same tilt*particle composition and transpose supplied by
                    // the managed refiner against Warp's existing Euler projector API.
                    Matrix3 rotation = Matrix3.Euler(.13f + view * .2f, .67f, -.31f) *
                                       Matrix3.Euler(-.23f, .41f + view * .17f, .19f);
                    float3 angles = Matrix3.EulerFromMatrix(rotation);
                    eulers[view * 3] = angles.X;
                    eulers[view * 3 + 1] = angles.Y;
                    eulers[view * 3 + 2] = angles.Z;
                    Matrix3 transposed = rotation.Transposed();
                    float[] entries = { transposed.M11, transposed.M21, transposed.M31,
                                        transposed.M12, transposed.M22, transposed.M32,
                                        transposed.M13, transposed.M23, transposed.M33 };
                    for (int j = 0; j < 9; j++) matrices[view * 9 + j] = entries[j] * oversampling;
                    nativeShifts[view * 3] = shifts[view * 2];
                    nativeShifts[view * 3 + 1] = shifts[view * 2 + 1];
                    for (int p = 0; p < 6; p++)
                    {
                        for (int j = 0; j < 9; j++)
                            matrixDerivatives[(view * 6 + p) * 9 + j] = .013f * MathF.Sin(1 + view * 54 + p * 9 + j);
                        shiftDerivatives[(view * 6 + p) * 2] = .13f * MathF.Cos(p + view + 1);
                        shiftDerivatives[(view * 6 + p) * 2 + 1] = -.17f * MathF.Sin(p + view + 1);
                        betaDerivatives[view * 6 + p] = .0013f * (p + view + 1);
                    }
                }

                double[] Evaluate(float[] m, float[] s, float[] b)
                {
                    double[] output = new double[views * 14];
                    Assert.Equal(0, GPU.TemplateMatchRefineEvaluate(context, m, matrixDerivatives,
                        s, shiftDerivatives, b, betaDerivatives, cutoff, output));
                    return output;
                }
                double[] actual = Evaluate(matrices, shifts, beta);
                IntPtr nativeProjection = Upload(new float[elements * views * 2]);
                GPU.ProjectForwardShiftedTex(textures[0], textures[1], nativeProjection,
                    new int3(dim), new int2(box), eulers, nativeShifts, new[] { 1f, 1f }, oversampling, views);
                float[] projected = new float[elements * views * 2];
                GPU.CopyDeviceToHost(nativeProjection, projected, projected.Length);
                for (int view = 0; view < views; view++)
                {
                    double exactCross = 0, exactPower = 0, legacyCross = 0, legacyPower = 0;
                    for (int row = 0; row < box; row++)
                        for (int x = 0; x <= box / 2; x++)
                        {
                            int y = row <= box / 2 ? row : row - box;
                            if ((x == 0 && y <= 0) || x == box / 2 || Math.Abs(y) == box / 2 || x * x + y * y > cutoff * cutoff) continue;
                            int id = view * elements + row * (box / 2 + 1) + x;
                            double phase = beta[view] * phaseRadiusSquared[id];
                            double transfer = ctfBase[id] * Math.Cos(phase) + quadrature[id] * Math.Sin(phase);
                            Complex exact = CpuProject(volume, dim, box, matrices, shifts, view, x, y) * transfer;
                            exactCross += inverseNoise[id] * (observed[id * 2] * exact.Real + observed[id * 2 + 1] * exact.Imaginary);
                            exactPower += inverseNoise[id] * (exact.Real * exact.Real + exact.Imaginary * exact.Imaginary);
                            double real = projected[id * 2] * transfer, imaginary = projected[id * 2 + 1] * transfer;
                            legacyCross += inverseNoise[id] * (observed[id * 2] * real + observed[id * 2 + 1] * imaginary);
                            legacyPower += inverseNoise[id] * (real * real + imaginary * imaginary);
                        }
                    // Independent double-precision CPU quadrature is the score oracle.
                    Near(actual[view * 14], exactCross, 2e-5, "exact CPU cross");
                    Near(actual[view * 14 + 1], exactPower, 2e-5, "exact CPU power");
                    // ProjectShifted3DtoNDKernel also fetches eight point-sampled voxels
                    // and performs manual lerps: texture-fraction quantization is absent.
                    // Its slightly wider tolerance allows the Euler roundtrip and
                    // different FP32 interpolation/phase evaluation order.
                    Near(actual[view * 14], legacyCross, 5e-5, "Warp projector cross");
                    Near(actual[view * 14 + 1], legacyPower, 5e-5, "Warp projector power");
                }

                // Differences are confined to this test; production derivatives are analytic.
                const float step = .001f;
                for (int parameter = 0; parameter < 6; parameter++)
                {
                    double[][] displaced = new double[2][];
                    for (int side = 0; side < 2; side++)
                    {
                        float delta = side == 0 ? -step : step;
                        float[] m = (float[])matrices.Clone(), s = (float[])shifts.Clone(), b = (float[])beta.Clone();
                        for (int view = 0; view < views; view++)
                        {
                            for (int j = 0; j < 9; j++) m[view * 9 + j] += delta * matrixDerivatives[(view * 6 + parameter) * 9 + j];
                            for (int j = 0; j < 2; j++) s[view * 2 + j] += delta * shiftDerivatives[(view * 6 + parameter) * 2 + j];
                            b[view] += delta * betaDerivatives[view * 6 + parameter];
                        }
                        displaced[side] = Evaluate(m, s, b);
                    }
                    for (int view = 0; view < views; view++)
                    {
                        Near(actual[view * 14 + 2 + parameter], (displaced[1][view * 14] - displaced[0][view * 14]) / (2 * step), 6e-3, "CUDA cross derivative");
                        Near(actual[view * 14 + 8 + parameter], (displaced[1][view * 14 + 1] - displaced[0][view * 14 + 1]) / (2 * step), 6e-3, "CUDA power derivative");
                    }
                }
                Assert.Equal(-1, GPU.TemplateMatchRefineEvaluate(context, matrices, matrixDerivatives,
                    shifts, shiftDerivatives, beta, betaDerivatives, -1, new double[views * 14]));
                double[] repeated = Evaluate(matrices, shifts, beta);
                for (int i = 0; i < actual.Length; i++) Near(repeated[i], actual[i], 1e-12, "persistent context reuse");
            }
            finally
            {
                if (context != IntPtr.Zero) GPU.TemplateMatchRefineDestroy(context);
                for (int i = 0; i < textures.Length; i++)
                    if (textures[i] != 0) GPU.DestroyTexture(textures[i], arrays[i]);
                foreach (IntPtr pointer in allocations) GPU.FreeDevice(pointer);
            }
        }
    }

    // Deliberately independent of the production shared C++ math: evaluate the
    // eight-node tensor-product interpolant in double precision from host data.
    private static Complex CpuProject(float[] volume, int dim, int box, float[] matrices,
        float[] shifts, int view, int kx, int ky)
    {
        int offset = view * 9;
        double x = (double)matrices[offset] * kx + (double)matrices[offset + 3] * ky;
        double y = (double)matrices[offset + 1] * kx + (double)matrices[offset + 4] * ky;
        double z = (double)matrices[offset + 2] * kx + (double)matrices[offset + 5] * ky;
        bool conjugate = x < 0;
        if (conjugate) { x = -x; y = -y; z = -z; }
        if (x >= dim / 2) return Complex.Zero;
        int ix = (int)Math.Floor(x), iy = (int)Math.Floor(y), iz = (int)Math.Floor(z);
        double fx = x - ix, fy = y - iy, fz = z - iz;
        Complex value = Complex.Zero;
        for (int dz = 0; dz <= 1; dz++)
            for (int dy = 0; dy <= 1; dy++)
                for (int dx = 0; dx <= 1; dx++)
                {
                    int wrappedY = ((iy + dy) % dim + dim) % dim;
                    int wrappedZ = ((iz + dz) % dim + dim) % dim;
                    int index = ((wrappedZ * dim + wrappedY) * (dim / 2 + 1) + ix + dx) * 2;
                    double weight = (dx == 0 ? 1 - fx : fx) * (dy == 0 ? 1 - fy : fy) * (dz == 0 ? 1 - fz : fz);
                    value += new Complex(volume[index], volume[index + 1]) * weight;
                }
        if (conjugate) value = Complex.Conjugate(value);
        double phase = -2 * Math.PI * ((double)kx * shifts[view * 2] + (double)ky * shifts[view * 2 + 1]) / box;
        return value * Complex.FromPolarCoordinates(1, phase);
    }

    private static void Near(double actual, double expected, double tolerance, string label)
    {
        Assert.True(double.IsFinite(actual) && Math.Abs(actual - expected) <= tolerance * (1 + Math.Abs(expected)),
            $"{label}: {actual:R} != {expected:R}");
    }
}
