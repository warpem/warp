using System;
using System.Runtime.InteropServices;

namespace Warp.Tools;

internal static class CtfNative
{
    public static void Check(int error, string operation)
    {
        if (error != 0) throw new InvalidOperationException($"CUDA CTF {operation} failed (error {error}).");
    }
    [DllImport("NativeAcceleration", EntryPoint = "CtfPowerCreate")]
    public static extern int PowerCreate(int width, int height, int window, int fft, int batch, int patches, int bins,
        int3[] origins, float[] hann, int[] starts, int[] indices, int[] displayIndices, out IntPtr context);
    [DllImport("NativeAcceleration", EntryPoint = "CtfPowerBegin")]
    public static extern int PowerBegin(IntPtr context);
    [DllImport("NativeAcceleration", EntryPoint = "CtfPowerAdd")]
    public static extern int PowerAdd(IntPtr context, float[] frame);
    [DllImport("NativeAcceleration", EntryPoint = "CtfPowerRead")]
    public static extern int PowerRead(IntPtr context, [Out] double[] power, [Out] float[] display);
    [DllImport("NativeAcceleration", EntryPoint = "CtfPowerDestroy")]
    public static extern void PowerDestroy(IntPtr context);

    [DllImport("NativeAcceleration", EntryPoint = "CtfFitCreate")]
    public static extern int FitCreate(int records, int samples, int knots, double[] moments, double[] basis,
        double[] data, double[] counts, double[] currentWeights, out IntPtr context);
    [DllImport("NativeAcceleration", EntryPoint = "CtfFitSetEnvelopeLayout")]
    public static extern int FitSetEnvelopeLayout(IntPtr context, int groups, int anchors, int[] ids, double[] blends);
    [DllImport("NativeAcceleration", EntryPoint = "CtfFitSearch")]
    public static extern int FitSearch(IntPtr context, double[] trials, double[] offsets, int trialCount, [Out] double[] scores);
    [DllImport("NativeAcceleration", EntryPoint = "CtfFitEvaluate")]
    public static extern int FitEvaluate(IntPtr context, double[] poses, int reweight, [Out] double[] output);
    [DllImport("NativeAcceleration", EntryPoint = "CtfFitReadWeights")]
    public static extern int FitReadWeights(IntPtr context, [Out] double[] weights);
    [DllImport("NativeAcceleration", EntryPoint = "CtfFitReadCoefficients")]
    public static extern int FitReadCoefficients(IntPtr context, [Out] float[] coefficients);
    [DllImport("NativeAcceleration", EntryPoint = "CtfFitDestroy")]
    public static extern void FitDestroy(IntPtr context);
}
