using System;
using System.Linq;
using ZLinq;

namespace Warp.Tools;

/// <summary>Exact chain rule from grid nodes and specimen-plane slopes to each local CTF.
/// Parameters: defocus nodes (µm), astigmatic cos/sin components (µm; half the principal defocus difference),
/// phase nodes (radians), optional specimen-plane X/Y slopes, then squared normal thickness (µm²).
/// Positions X/Y and patch width are in µm.</summary>
public sealed class CtfFitGeometry
{
    public readonly double[] DefocusWeights, PhaseWeights;
    public readonly double X, Y;
    public readonly Matrix3? Rotation;
    public readonly double PatchWidth;
    public int ThicknessIndex => DefocusWeights.Length + 2 + PhaseWeights.Length + (Rotation.HasValue ? 2 : 0);
    public CtfFitGeometry(double[] defocusWeights, double[] phaseWeights, double x = 0, double y = 0, Matrix3? rotation = null, double patchWidth = 0)
    { DefocusWeights = defocusWeights; PhaseWeights = phaseWeights; X = x; Y = y; Rotation = rotation; PatchWidth = patchWidth; }
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
    // The final parameter is the squared physical thickness normal to the specimen plane.
    // At fixed image coordinates the beam path is T/|n_z|, not T*cos(tilt).
    public void WritePose(double[] p, double[] poses, int offset)
    {
        var local = Evaluate(p); int nd = DefocusWeights.Length;
        poses[offset] = local.Defocus; poses[offset+1] = p[nd]; poses[offset+2] = p[nd+1]; poses[offset+3] = local.Phase;
        var slab = SlabGeometry(p);
        poses[offset+4] = p[ThicknessIndex]*slab.Factor;
        poses[offset+5] = slab.WidthX; poses[offset+6] = slab.WidthY;
    }
    public (double Factor, double WidthX, double WidthY, double FactorX, double FactorY,
        double WidthXX, double WidthXY, double WidthYX, double WidthYY) SlabGeometry(double[] p)
    {
        if (!Rotation.HasValue) return (1,0,0,0,0,0,0,0,0);
        var r = Rotation.Value; int j = ThicknessIndex-2;
        double sx=p[j], sy=p[j+1], norm=1+sx*sx+sy*sy;
        double nx=r.M11*sx+r.M12*sy+r.M13, ny=r.M21*sx+r.M22*sy+r.M23, nz=r.M31*sx+r.M32*sy+r.M33;
        double nz2=nz*nz, factor=norm/nz2;
        return (factor,-PatchWidth*nx/nz,-PatchWidth*ny/nz,
            2*sx/nz2-2*factor*r.M31/nz,2*sy/nz2-2*factor*r.M32/nz,
            -PatchWidth*(r.M11*nz-nx*r.M31)/nz2,-PatchWidth*(r.M12*nz-nx*r.M32)/nz2,
            -PatchWidth*(r.M21*nz-ny*r.M31)/nz2,-PatchWidth*(r.M22*nz-ny*r.M32)/nz2);
    }
    public void AccumulateVolume(double[] gradient, double[] local, double[] p)
    {
        var center = Evaluate(p);
        Accumulate(gradient,local,center.SlopeX,center.SlopeY);
        var slab = SlabGeometry(p); int t = ThicknessIndex;
        gradient[t] += local[4]*slab.Factor;
        if (Rotation.HasValue)
        {
            gradient[t-2] += local[4]*p[t]*slab.FactorX+local[5]*slab.WidthXX+local[6]*slab.WidthYX;
            gradient[t-1] += local[4]*p[t]*slab.FactorY+local[5]*slab.WidthXY+local[6]*slab.WidthYY;
        }
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
