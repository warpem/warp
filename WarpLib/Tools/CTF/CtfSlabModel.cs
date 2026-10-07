using System;

namespace Warp.Tools;

/// <summary>Power transfer of independent scatterers uniformly distributed through a slab.
/// Thickness squared and defocus widths are in µm² and µm. A patch uses the squared Hann aperture.</summary>
public static class CtfSlabModel
{
    public static (double Value, double Derivative) Sinc(double x)
    {
        double x2 = x*x;
        if (Math.Abs(x) < .05)
            return (1-x2/6+x2*x2/120-x2*x2*x2/5040, -x/3+x*x2/30-x*x2*x2/840);
        return (Math.Sin(x)/x, (x*Math.Cos(x)-Math.Sin(x))/x2);
    }
    public static (double Value, double Derivative) HannPower(double x)
    {
        var a = Sinc(x); var b = Sinc(x-Math.PI); var c = Sinc(x+Math.PI);
        var d = Sinc(x-2*Math.PI); var e = Sinc(x+2*Math.PI);
        return (a.Value+2.0/3*(b.Value+c.Value)+(d.Value+e.Value)/6,
            a.Derivative+2.0/3*(b.Derivative+c.Derivative)+(d.Derivative+e.Derivative)/6);
    }
    public static (double Value, double ThicknessSquared, double WidthX, double WidthY) Modulation(double kq2, double thicknessSquared, double widthX, double widthY)
    {
        double z = kq2*kq2*Math.Max(0,thicknessSquared), x = Math.Sqrt(z);
        var slab = Sinc(x);
        // Optimize thickness squared so the derivative is finite and nonzero at a zero-thickness start.
        double dt = kq2*kq2*(z < .0025 ? -1.0/6+z/60-z*z/1680 : slab.Derivative/(2*x));
        var hx = HannPower(kq2*widthX); var hy = HannPower(kq2*widthY);
        return (slab.Value*hx.Value*hy.Value,dt*hx.Value*hy.Value,
            slab.Value*hx.Derivative*kq2*hy.Value,slab.Value*hx.Value*hy.Derivative*kq2);
    }
    public static double Power(double gamma, double kq2, double thicknessSquared, double widthX = 0, double widthY = 0)
        => .5-.5*Math.Cos(2*gamma)*Modulation(kq2,thicknessSquared,widthX,widthY).Value;
}
