using System;
using System.Runtime.InteropServices;
using Warp.Tools;

namespace Warp;

public static partial class GPU
{
    [DllImport("NativeAcceleration")]
    public static extern void MatchTransfer(IntPtr ctf, IntPtr output, int size, int tilts, float[] rotations, float[] inverseNoise);
    [DllImport("NativeAcceleration")]
    public static extern void MatchHybridWeights(IntPtr weights, int box, int particles, int tilts,
        float[] rotations, float[] noise, float pixel, float diameter);

    [DllImport("NativeAcceleration")]
    public static extern void MatchAutocorrelationWindow(IntPtr values, int3 dims, float pixel, float length);

    [DllImport("NativeAcceleration")]
    public static extern void MatchSpectrumResample(IntPtr input, int3 dims, IntPtr output, int size);

    [DllImport("NativeAcceleration")]
    public static extern void MatchBackproject(IntPtr image, int2 imageDims, IntPtr volume, int3 dims,
        IntPtr geometry, int3 grid, float spacing, float defocus, float step);
}
