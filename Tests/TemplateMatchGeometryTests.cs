using System;
using System.IO;
using System.Reflection;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

/// <summary>Tests use the managed interpolants only; no NativeAcceleration dependency.</summary>
public class TemplateMatchGeometryTests
{
    [Theory]
    [InlineData(0.21f, 0.34f, 0.67f, 0.41f)]
    [InlineData(-0.22f, 0.27f, 0.31f, -0.1f)]
    [InlineData(1.22f, 0.48f, -0.17f, 1.1f)]
    [InlineData(0f, 0f, 0f, 0f)]
    [InlineData(1f, 1f, 1f, 1f)]
    public void MultilinearFieldHasExpectedValueAndNormalizedSpatialGradient(float x, float y, float z, float t)
    {
        var grid = SampleField(new int4(4, 3, 5, 2));
        var point = new float4(x, y, z, t);
        float value = grid.GetInterpolatedWithGradient(point, out float3 gradient);

        // Existing interpolation extrapolates below zero, but holds the last sample above one.
        double cx = Math.Min(x, 1), cy = Math.Min(y, 1), cz = Math.Min(z, 1), ct = Math.Min(t, 1);
        Close(Field(cx, cy, cz, ct), value, 3e-6);
        Close(grid.GetInterpolatedOld(point), value, 3e-6);
        Close(x >= 1 ? 0 : 2 + 1.5 * cy + 0.8 * cy * cz + 1.2 * ct + 2 * cy * cz * ct, gradient.X, 3e-6);
        Close(y >= 1 ? 0 : -3 + 1.5 * cx + 0.8 * cx * cz + 2 * cx * cz * ct, gradient.Y, 3e-6);
        Close(z >= 1 ? 0 : 4 - 0.7 * ct + 0.8 * cx * cy + 2 * cx * cy * ct, gradient.Z, 3e-6);
    }

    [Fact]
    public void RandomFieldGradientsAgreeWithFiniteDifferencesInsideCellsAndOutsideTheGrid()
    {
        var random = new Random(7149);
        var dims = new int4(4, 3, 5, 2);
        var values = new float[dims.Elements()];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 4 - 2);
        var grid = new LinearGrid4D(dims, values);
        int[] dimensions = { dims.X, dims.Y, dims.Z, dims.W };

        for (int sample = 0; sample < 60; sample++)
        {
            var coordinates = new float[4];
            for (int axis = 0; axis < 4; axis++)
            {
                // Keep interior samples well away from interpolation knots.
                int cell = random.Next(dimensions[axis] - 1);
                coordinates[axis] = (float)((cell + 0.2 + 0.6 * random.NextDouble()) / (dimensions[axis] - 1));
                if (sample % 3 == 1 && axis == sample % 4) coordinates[axis] = -0.25f;
                if (sample % 3 == 2 && axis == sample % 4) coordinates[axis] = 1.25f;
            }
            float4 point = Point(coordinates);
            float actual = grid.GetInterpolatedWithGradient(point, out float3 gradient);
            Close(grid.GetInterpolatedOld(point), actual, 3e-6);
            double[] analytic = { gradient.X, gradient.Y, gradient.Z };
            for (int axis = 0; axis < 3; axis++)
            {
                const float step = 0.0005f;
                float[] minus = (float[])coordinates.Clone(), plus = (float[])coordinates.Clone();
                minus[axis] -= step;
                plus[axis] += step;
                double finiteDifference = (grid.GetInterpolatedOld(Point(plus)) - grid.GetInterpolatedOld(Point(minus))) / (plus[axis] - minus[axis]);
                Close(finiteDifference, analytic[axis], 0.002);
            }
        }
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(0.5f)]
    [InlineData(1f)]
    public void GradientAtAKnotUsesTheRightHandCellIncludingClampedUpperBoundary(float x)
    {
        // Slopes are 4 and -6 in normalized coordinates, then zero above the upper boundary.
        var grid = new LinearGrid4D(new int4(3, 1, 1, 1), new[] { 1f, 3f, 0f });
        var point = new float4(x, 0.3f, -0.2f, 0.5f);
        float value = grid.GetInterpolatedWithGradient(point, out float3 gradient);
        double expected = x == 0 ? 4 : x == 0.5f ? -6 : 0;
        Close(expected, gradient.X, 1e-6);
        Assert.Equal(0, gradient.Y);
        Assert.Equal(0, gradient.Z);
        const float step = 0.001f;
        double rightDerivative = (grid.GetInterpolatedOld(new float4(x + step, point.Y, point.Z, point.W)) - grid.GetInterpolatedOld(point)) / ((x + step) - x);
        Close(rightDerivative, gradient.X, 0.001);
        Close(grid.GetInterpolatedOld(point), value, 1e-6);
    }

    [Fact]
    public void SingletonSpatialAxesHaveZeroDerivativesAtAnyCoordinate()
    {
        var grid = SampleField(new int4(1, 4, 1, 2));
        var point = new float4(-3, 0.43f, 7, 0.24f);
        float value = grid.GetInterpolatedWithGradient(point, out float3 gradient);

        Close(Field(0, point.Y, 0, point.W), value, 3e-6);
        Close(grid.GetInterpolatedOld(point), value, 3e-6);
        Assert.Equal(0, gradient.X);
        Close(-3, gradient.Y, 3e-6);
        Assert.Equal(0, gradient.Z);
    }

    [Fact]
    public void SingletonTimeAxisIgnoresTimeWithoutChangingSpatialDerivatives()
    {
        var grid = SampleField(new int4(3, 3, 3, 1));
        var point = new float4(0.23f, 0.31f, 0.62f, -8);
        float value = grid.GetInterpolatedWithGradient(point, out float3 gradient);

        Close(Field(point.X, point.Y, point.Z, 0), value, 3e-6);
        Close(2 + 1.5 * point.Y + 0.8 * point.Y * point.Z, gradient.X, 3e-6);
        Close(-3 + 1.5 * point.X + 0.8 * point.X * point.Z, gradient.Y, 3e-6);
        Close(4 + 0.8 * point.X * point.Y, gradient.Z, 3e-6);
    }

    [Fact]
    public void SingleValueGridRemainsConstantEverywhere()
    {
        var grid = new LinearGrid4D(new int4(1, 1, 1, 1), new[] { 7.5f });
        float value = grid.GetInterpolatedWithGradient(new float4(-2, 4, 0.2f, 6), out float3 gradient);

        Assert.Equal(7.5f, value);
        Assert.Equal(0, gradient.X);
        Assert.Equal(0, gradient.Y);
        Assert.Equal(0, gradient.Z);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FrozenAffineGeometryPreservesProjectionDefocusHandAndMatrixConvention(bool inverted)
    {
        using var fixture = new FrozenGeometryFixture(nonlinear: false);
        TiltSeries series = fixture.Series;
        series.AreAnglesInverted = inverted;
        float3 anchor = new(279, 546, 392);
        float3[] offsets = { new(30, -20, 15), new(-25, 30, -30) };
        Matrix3 particleRotation = Matrix3.Euler(0.31f, 0.77f, -0.28f);
        Matrix3 magnification = new(1.02f, -0.013f, 0, -0.013f, 0.985f, 0, 0, 0, 1);
        float3 frequency = new(7, -11, 0);
        MethodInfo pack = typeof(TiltSeries).GetMethod("PackMatchMatrix", BindingFlags.Static | BindingFlags.NonPublic);
        Assert.NotNull(pack);
        for (int tilt = 0; tilt < series.NTilts; tilt++)
        {
            var local = ReadGeometry(series, anchor, tilt);
            foreach (float3 delta in offsets)
            {
                var full = ReadGeometry(series, anchor + delta, tilt);
                float3 frozen = local.Position + local.Jacobian[0] * delta.X
                    + local.Jacobian[1] * delta.Y + local.Jacobian[2] * delta.Z;
                Assert.InRange(Math.Abs(full.Position.X - frozen.X), 0, 4e-4);
                Assert.InRange(Math.Abs(full.Position.Y - frozen.Y), 0, 4e-4);
                Assert.InRange(Math.Abs(full.Position.Z - frozen.Z), 0, 5e-7);

                // Follow the batch ABI's actual column-major readout independently of Matrix3.ToArray.
                var packed = new float[9];
                pack.Invoke(null, new object[] { local.Rotation.Transposed() * magnification, packed, 0, 2f });
                float3 g = new(packed[0] * frequency.X + packed[3] * frequency.Y,
                    packed[1] * frequency.X + packed[4] * frequency.Y,
                    packed[2] * frequency.X + packed[5] * frequency.Y);
                float3 batch = particleRotation.Transposed() * g;
                float3 scalar = (full.Rotation * particleRotation).Transposed() * magnification * frequency * 2;
                Assert.InRange((batch - scalar).Length(), 0, 1e-5);

                const double phaseScale = -Math.PI * 0.019687 * 10000; // 300 kV, micrometers to Angstrom.
                const double physicalRadiusSquared = 0.01;
                double batchPhase = phaseScale * (local.Jacobian[0].Z * delta.X
                    + local.Jacobian[1].Z * delta.Y + local.Jacobian[2].Z * delta.Z) * physicalRadiusSquared;
                double scalarPhase = phaseScale * (full.Position.Z - local.Position.Z) * physicalRadiusSquared;
                Assert.InRange(Math.Abs(batchPhase - scalarPhase), 0, 4e-6);
            }
        }
    }

    [Fact]
    public void FrozenGeometryErrorIsSecondOrderForTheExistingNonlinearWarpFixture()
    {
        using var fixture = new FrozenGeometryFixture(nonlinear: true);
        float3 anchor = new(279, 546, 392);
        for (int tilt = 0; tilt < fixture.Series.NTilts; tilt++)
        {
            var local = ReadGeometry(fixture.Series, anchor, tilt);
            double Error(float distance)
            {
                float3 delta = new(distance, -distance, distance);
                var full = ReadGeometry(fixture.Series, anchor + delta, tilt);
                float3 frozen = local.Position + local.Jacobian[0] * delta.X
                    + local.Jacobian[1] * delta.Y + local.Jacobian[2] * delta.Z;
                float3 residual = full.Position - frozen;
                residual.Z *= 10000; // Compare defocus residual in Angstrom too.
                return residual.Length();
            }
            double at15 = Error(15), at30 = Error(30);
            // These are the same 13*y*z, 19*x*z and -12*x*y warp terms as the native
            // geometry fixture. This characterizes that fixture, not a bound for arbitrary data.
            Assert.InRange(at15, 0.005, 0.011);
            Assert.InRange(at30, 0.025, 0.040);
            Assert.InRange(at30 / at15, 3.4, 4.6);
        }
    }

    private static (float3 Position, float3[] Jacobian, Matrix3 Rotation) ReadGeometry(TiltSeries series, float3 position, int tilt)
    {
        MethodInfo method = typeof(TiltSeries).GetMethod("GetTemplateMatchGeometry", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        object geometry = method.Invoke(series, new object[] { position, tilt });
        T Read<T>(string field) => (T)geometry.GetType().GetField(field).GetValue(geometry);
        return (Read<float3>("ImagePosition"), Read<float3[]>("PositionDerivatives"), Read<Matrix3>("Rotation"));
    }

    private sealed class FrozenGeometryFixture : IDisposable
    {
        private readonly string directory = Path.Combine(Path.GetTempPath(), "warp_match_frozen_geometry_" + Guid.NewGuid().ToString("N"));
        public TiltSeries Series { get; }

        public FrozenGeometryFixture(bool nonlinear)
        {
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, "geometry.tomostar");
            File.WriteAllText(path, "data_\n\nloop_\n_wrpMovieName #1\n_wrpAngleTilt #2\n_wrpDose #3\n" +
                "tilt0.mrc -42 10\ntilt1.mrc -14 1\ntilt2.mrc 19 22\ntilt3.mrc 47 6\n");
            Series = new TiltSeries(path)
            {
                VolumeDimensionsPhysical = new float3(900, 1300, 700),
                ImageDimensionsPhysical = new float2(2400, 2100),
                SizeRoundingFactors = new float3(0.998f, 1.003f, 1),
                LevelAngleX = 3.5f, LevelAngleY = -4.25f,
                TiltAxisAngles = new[] { 7f, 14f, -11f, 22f },
                TiltAxisOffsetX = new[] { 3f, -5f, 7f, -2f },
                TiltAxisOffsetY = new[] { -4f, 6f, 2f, -3f },
                GridVolumeWarpX = WarpGrid((x, y, z, t) => 22 * x + 13 * y * (nonlinear ? z : 1) + 9 * x * t - 3),
                GridVolumeWarpY = WarpGrid((x, y, z, t) => -17 * y + 19 * x * (nonlinear ? z : 1) + 7 * z * t + 2),
                GridVolumeWarpZ = WarpGrid((x, y, z, t) => 15 * z - 12 * x * (nonlinear ? y : 1) + 11 * y * t - 4)
            };
            // Singleton cubic grids exercise the real geometry without requiring native einspline.
            Series.GridMovementX.Values[0] = 7;
            Series.GridMovementY.Values[0] = -4;
            Series.GridCTFDefocus.Values[0] = 2.1f;
            Series.GridAngleX.Values[0] = 3;
            Series.GridAngleY.Values[0] = -2;
            Series.GridAngleZ.Values[0] = 1;
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

        public void Dispose() => Directory.Delete(directory, true);
    }

    private static LinearGrid4D SampleField(int4 dims)
    {
        var values = new float[dims.Elements()];
        int index = 0;
        for (int t = 0; t < dims.W; t++)
            for (int z = 0; z < dims.Z; z++)
                for (int y = 0; y < dims.Y; y++)
                    for (int x = 0; x < dims.X; x++)
                        values[index++] = (float)Field(x / (double)Math.Max(1, dims.X - 1), y / (double)Math.Max(1, dims.Y - 1),
                                                       z / (double)Math.Max(1, dims.Z - 1), t / (double)Math.Max(1, dims.W - 1));
        return new LinearGrid4D(dims, values);
    }

    private static double Field(double x, double y, double z, double t) =>
        1 + 2 * x - 3 * y + 4 * z + 0.5 * t + 1.5 * x * y - 0.7 * z * t + 0.8 * x * y * z + 1.2 * x * t + 2 * x * y * z * t;
    private static float4 Point(float[] p) => new float4(p[0], p[1], p[2], p[3]);
    private static void Close(double expected, double actual, double tolerance) =>
        Assert.True(Math.Abs(expected - actual) <= tolerance * Math.Max(1, Math.Abs(expected)), $"Expected {expected:G12}, got {actual:G12}.");
}
