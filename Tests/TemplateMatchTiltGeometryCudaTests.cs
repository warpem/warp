using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

/// <summary>
/// Opt-in native integration coverage of the actual refinement geometry, including native cubic-spline
/// gradients. Run with WARP_RUN_CUDA_TESTS=1 and rebuilt NativeAcceleration, like the scorer tests.
/// </summary>
public class TemplateMatchTiltGeometryCudaTests
{
    [TemplateMatchCudaFact]
    public void GeometryValuesMatchExistingOneTiltApisWithNonconstantAlignmentGrids()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new GeometryFixture();
            TiltSeries series = fixture.Series;
            foreach (bool inverted in new[] { false, true })
            {
                series.AreAnglesInverted = inverted;
                foreach (float3 position in fixture.Positions)
                    for (int tilt = 0; tilt < series.NTilts; tilt++)
                    {
                        Geometry actual = Evaluate(series, position, tilt);
                        float3 expectedPosition = series.GetPositionsInOneTilt(new[] { position }, tilt)[0];
                        // Deliberately use the one-tilt API: the legacy all-tilt API normalizes Y by X.
                        float3 expectedAngles = series.GetAnglesInOneTilt(new[] { position }, new[] { new float3(0) }, tilt)[0];
                        Matrix3 expectedRotation = Matrix3.Euler(expectedAngles);
                        Near(actual.Position.X, expectedPosition.X, 3e-4, 3e-7, "image X");
                        Near(actual.Position.Y, expectedPosition.Y, 3e-4, 3e-7, "image Y");
                        Near(actual.Position.Z, expectedPosition.Z, 1e-6, 3e-7, "defocus");
                        CompareMatrices(actual.Rotation, expectedRotation, 2e-6, 2e-6, "rotation");
                    }
            }
        }
    }

    [TemplateMatchCudaFact]
    public void GeometryPositionAndRotationJacobiansMatchPhysicalCoordinateDifferences()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new GeometryFixture();
            TiltSeries series = fixture.Series;
            const float step = 1.5f; // Angstrom; sufficiently large to resolve single-precision defocus differences.
            foreach (bool inverted in new[] { false, true })
            {
                series.AreAnglesInverted = inverted;
                foreach (float3 position in fixture.Positions)
                    for (int tilt = 0; tilt < series.NTilts; tilt++)
                    {
                        Geometry actual = Evaluate(series, position, tilt);
                        for (int axis = 0; axis < 3; axis++)
                        {
                            float3 delta = axis == 0 ? new float3(step, 0, 0) : axis == 1 ? new float3(0, step, 0) : new float3(0, 0, step);
                            Geometry minus = Evaluate(series, position - delta, tilt);
                            Geometry plus = Evaluate(series, position + delta, tilt);
                            float3 numericalPosition = (plus.Position - minus.Position) / (2 * step);
                            float3 analyticPosition = actual.PositionDerivatives[axis];
                            string context = $"tilt {tilt}, inverted {inverted}, axis {axis}";
                            Near(analyticPosition.X, numericalPosition.X, 1.5e-4, 4e-4, context + " image X derivative");
                            Near(analyticPosition.Y, numericalPosition.Y, 1.5e-4, 4e-4, context + " image Y derivative");
                            Near(analyticPosition.Z, numericalPosition.Z, 2e-7, 2e-3, context + " defocus derivative");
                            Matrix3 numericalRotation = (plus.Rotation - minus.Rotation) / (2 * step);
                            CompareMatrices(actual.RotationDerivatives[axis], numericalRotation,
                                1e-7, 3e-3, context + " rotation derivative");
                        }
                        // Avoid a vacuous derivative test if the fixture's angle grids become constant.
                        Assert.True(MaxAbs(actual.RotationDerivatives[0]) > 1e-6);
                        Assert.True(MaxAbs(actual.RotationDerivatives[1]) > 1e-6);
                    }
            }
        }
    }

    private sealed class Geometry
    {
        public float3 Position;
        public float3[] PositionDerivatives;
        public Matrix3 Rotation;
        public Matrix3[] RotationDerivatives;
    }

    private static Geometry Evaluate(TiltSeries series, float3 position, int tilt)
    {
        MethodInfo method = typeof(TiltSeries).GetMethod("GetTemplateMatchGeometry", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        object result = method.Invoke(series, new object[] { position, tilt });
        Assert.NotNull(result);
        T Read<T>(string name)
        {
            FieldInfo field = result.GetType().GetField(name, BindingFlags.Instance | BindingFlags.Public);
            Assert.NotNull(field);
            return (T)field.GetValue(result);
        }
        return new Geometry
        {
            Position = Read<float3>("ImagePosition"),
            PositionDerivatives = Read<float3[]>("PositionDerivatives"),
            Rotation = Read<Matrix3>("Rotation"),
            RotationDerivatives = Read<Matrix3[]>("RotationDerivatives")
        };
    }

    private sealed class GeometryFixture : IDisposable
    {
        private readonly string directory = Path.Combine(Path.GetTempPath(), "warp_match_geometry_" + Guid.NewGuid().ToString("N"));
        private readonly List<CubicGrid> grids = new();
        public TiltSeries Series { get; }
        public float3[] Positions { get; } = { new float3(279, 546, 392), new float3(567, 793, 196) };

        public GeometryFixture()
        {
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, "geometry.tomostar");
            File.WriteAllText(path, "data_\n\nloop_\n_wrpMovieName #1\n_wrpAngleTilt #2\n_wrpDose #3\n" +
                "tilt0.mrc -42 10\ntilt1.mrc -14 1\ntilt2.mrc 19 22\ntilt3.mrc 47 6\n");
            Series = new TiltSeries(path)
            {
                VolumeDimensionsPhysical = new float3(900, 1300, 700), // Non-square to catch X/Y normalization errors.
                ImageDimensionsPhysical = new float2(2400, 2100),
                SizeRoundingFactors = new float3(0.998f, 1.003f, 1),
                LevelAngleX = 3.5f,
                LevelAngleY = -4.25f,
                TiltAxisAngles = new[] { 7f, 14f, -11f, 22f },
                TiltAxisOffsetX = new[] { 3f, -5f, 7f, -2f },
                TiltAxisOffsetY = new[] { -4f, 6f, 2f, -3f },
                GridVolumeWarpX = WarpGrid((x, y, z, t) => 22 * x + 13 * y * z + 9 * x * t - 3),
                GridVolumeWarpY = WarpGrid((x, y, z, t) => -17 * y + 19 * x * z + 7 * z * t + 2),
                GridVolumeWarpZ = WarpGrid((x, y, z, t) => 15 * z - 12 * x * y + 11 * y * t - 4)
            };
            Series.GridMovementX = Cubic(new int3(4, 3, 5), (x, y, t) => 18 * x - 11 * y + 7 * x * y + 4 * t * t);
            Series.GridMovementY = Cubic(new int3(3, 1, 5), (x, y, t) => -14 * x + 9 * x * t + 5 * t * t);
            Series.GridCTFDefocus = Cubic(new int3(4, 3, 5), (x, y, t) => 2.1 + 0.3 * x - 0.2 * y + 0.15 * x * y + 0.12 * t);
            Series.GridAngleX = Cubic(new int3(4, 3, 5), (x, y, t) => 3 + 8 * x + 5 * y * y + 2 * x * y + 3 * t);
            Series.GridAngleY = Cubic(new int3(1, 4, 5), (x, y, t) => -2 + 9 * y + 4 * y * t);
            Series.GridAngleZ = Cubic(new int3(4, 3, 1), (x, y, t) => 1 - 7 * x + 6 * y + 3 * x * y, centered: true);
        }

        private CubicGrid Cubic(int3 dims, Func<double, double, double, double> field, bool centered = false)
        {
            var values = new float[dims.Elements()];
            for (int z = 0, index = 0; z < dims.Z; z++)
                for (int y = 0; y < dims.Y; y++)
                    for (int x = 0; x < dims.X; x++)
                    {
                        double Coordinate(int i, int count) => count == 1 ? 0 : centered ? (i + 0.5) / count : i / (double)(count - 1);
                        values[index++] = (float)field(Coordinate(x, dims.X), Coordinate(y, dims.Y), Coordinate(z, dims.Z));
                    }
            var grid = new CubicGrid(dims, values, centered);
            grids.Add(grid);
            return grid;
        }

        private static LinearGrid4D WarpGrid(Func<double, double, double, double, double> field)
        {
            var dims = new int4(3, 3, 3, 4);
            var values = new float[dims.Elements()];
            for (int t = 0, index = 0; t < dims.W; t++)
                for (int z = 0; z < dims.Z; z++)
                    for (int y = 0; y < dims.Y; y++)
                        for (int x = 0; x < dims.X; x++)
                            values[index++] = (float)field(x / 2.0, y / 2.0, z / 2.0, t / 3.0);
            return new LinearGrid4D(dims, values);
        }

        public void Dispose()
        {
            foreach (CubicGrid grid in grids) grid.Dispose();
            Directory.Delete(directory, true);
        }
    }

    // Avoid Matrix3.ToArray: its historical third-column typo would hide rotation errors.
    private static double[] Entries(Matrix3 m) => new double[]
        { m.M11, m.M21, m.M31, m.M12, m.M22, m.M32, m.M13, m.M23, m.M33 };
    private static double MaxAbs(Matrix3 m)
    {
        double maximum = 0;
        foreach (double value in Entries(m)) maximum = Math.Max(maximum, Math.Abs(value));
        return maximum;
    }
    private static void CompareMatrices(Matrix3 actual, Matrix3 expected, double absolute, double relative, string label)
    {
        double[] a = Entries(actual), e = Entries(expected);
        for (int i = 0; i < 9; i++) Near(a[i], e[i], absolute, relative, label + $" entry {i}");
    }
    private static void Near(double actual, double expected, double absolute, double relative, string label) =>
        Assert.True(double.IsFinite(actual) && Math.Abs(actual - expected) <= absolute + relative * Math.Abs(expected),
            $"{label}: analytic {actual:R}, expected {expected:R}");
}
