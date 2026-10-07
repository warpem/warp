using System;
using System.Linq;
using ZLinq;

namespace Warp.Tools;

/// <summary>Exact chain rule from grid nodes and specimen-plane slopes to each local CTF.
/// Parameters: defocus nodes (µm), astigmatic cos/sin components (µm; half the principal defocus difference),
/// phase nodes (radians), then optional specimen-plane X/Y slopes. Positions X/Y are in µm.</summary>
public sealed class CtfFitGeometry
{
    public readonly double[] DefocusWeights, PhaseWeights;
    public readonly double X, Y;
    public readonly Matrix3? Rotation;
    public CtfFitGeometry(double[] defocusWeights, double[] phaseWeights, double x = 0, double y = 0, Matrix3? rotation = null)
    { DefocusWeights = defocusWeights; PhaseWeights = phaseWeights; X = x; Y = y; Rotation = rotation; }
    public (double Defocus, double Phase, double SlopeX, double SlopeY) Evaluate(double[] p)
    {
        int nd = DefocusWeights.Length, np = PhaseWeights.Length;
        double df = 0, phase = 0, dx = 0, dy = 0;
        for (int j = 0; j < nd; j++) df += p[j] * DefocusWeights[j];
        for (int j = 0; j < np; j++) phase += p[nd + 2 + j] * PhaseWeights[j];
        if (Rotation.HasValue)
        {
            Matrix3 r = Rotation.Value;
            double sx = p[nd + 2 + np], sy = p[nd + 3 + np];
            double nx = r.M11 * sx + r.M12 * sy + r.M13, ny = r.M21 * sx + r.M22 * sy + r.M23, nz = r.M31 * sx + r.M32 * sy + r.M33;
            if (Math.Abs(nz) < .08) return (double.NaN, phase, 0, 0);
            double numerator = nx * X + ny * Y;
            df -= numerator / nz;
            dx = -((r.M11 * X + r.M21 * Y) * nz - numerator * r.M31) / (nz * nz);
            dy = -((r.M12 * X + r.M22 * Y) * nz - numerator * r.M32) / (nz * nz);
        }
        return (df, phase, dx, dy);
    }
    public void Accumulate(double[] gradient, double[] local, double dx, double dy)
    {
        int nd = DefocusWeights.Length, np = PhaseWeights.Length;
        for (int j = 0; j < nd; j++) gradient[j] += local[0] * DefocusWeights[j];
        gradient[nd] += local[1]; gradient[nd + 1] += local[2];
        for (int j = 0; j < np; j++) gradient[nd + 2 + j] += local[3] * PhaseWeights[j];
        if (Rotation.HasValue) { gradient[nd + 2 + np] += local[0] * dx; gradient[nd + 3 + np] += local[0] * dy; }
    }
    public static double[][] GridWeights(int3 dimensions, float3[] positions)
    {
        int n = (int)dimensions.Elements();
        double[][] weights = positions.Select(_ => new double[n]).ToArray();
        for (int j = 0; j < n; j++)
        {
            float[] values = new float[n]; values[j] = 1;
            using var grid = new CubicGrid(dimensions, values);
            float[] sampled = grid.GetInterpolated(positions);
            for (int i = 0; i < positions.Length; i++) weights[i][j] = sampled[i];
        }
        return weights;
    }
}
