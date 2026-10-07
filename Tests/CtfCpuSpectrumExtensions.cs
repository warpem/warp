using System.Runtime.CompilerServices;
using System.Reflection;
using Warp.Tools;

namespace Tests;

// Test-only oracle, with its own double-precision spline construction and solves.
internal static class CtfCpuSpectrumExtensions
{
    sealed class Entry
    {
        public CtfCpuSpectrum Cpu;
        public int Version = -1;
    }
    static readonly ConditionalWeakTable<CtfSpectrumFit, Entry> Cache = new();
    // Avoid InternalsVisibleTo: WarpLib's generated LINQ extensions would leak into all tests.
    static double Field(CtfSpectrumFit s, string name) => (double)typeof(CtfSpectrumFit).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic).GetValue(s);
    static int Version(CtfSpectrumFit s) => (int)typeof(CtfSpectrumFit).GetProperty("WeightVersion", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(s);
    static double[] Weights(CtfSpectrumFit s) => (double[])typeof(CtfSpectrumFit).GetProperty("CurrentWeights", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(s);
    static CtfCpuSpectrum Reference(CtfSpectrumFit spectrum)
    {
        var entry = Cache.GetValue(spectrum, s => new Entry { Cpu = new(s.Samples, Field(s, "VoltageKV"), Field(s, "CsMM"), Field(s, "Amplitude")) });
        if (entry.Version != Version(spectrum))
        {
            if (Weights(spectrum) is { } weights) entry.Cpu.SetGpuWeights(weights, 0);
            entry.Version = Version(spectrum);
        }
        return entry.Cpu;
    }
    public static CtfCpuSpectrum.Evaluation Evaluate(this CtfSpectrumFit s, double df, double ax, double ay, double phase, bool details = false) => Reference(s).Evaluate(df, ax, ay, phase, details);
    public static double QuickScore(this CtfSpectrumFit s, double df, double phase = 0) => Reference(s).QuickScore(df, phase);
    public static double Reweight(this CtfSpectrumFit s, double df, double ax, double ay, double phase) => Reference(s).Reweight(df, ax, ay, phase);
}
