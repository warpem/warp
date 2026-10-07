using System;
using System.Collections.Generic;
using System.Linq;
using ZLinq;

namespace Warp.Tools;

/// <summary>Spectrum data and spline layout for GPU fitting. Frequencies are in Å⁻¹, defocus/astigmatism in µm,
/// phase in radians. The data and model use the same Fourier-bin moments.</summary>
public sealed class CtfSpectrumFit
{
    public readonly record struct Sample(double Q2, double Q4, double AstigX, double AstigY, double Power, double Count);
    public readonly Sample[] Samples;
    public readonly double PowerScale;
    readonly double[,] basis;
    readonly int[] firstBasis, lastBasis;
    readonly double[] data;
    internal double[] CurrentWeights { get; private set; }
    internal int WeightVersion { get; private set; }
    internal readonly double VoltageKV, CsMM, Amplitude;
    readonly int knots;
    readonly double kDefocus, kCs, amplitudePhase;
    public static double Wavelength(double voltageKV) => 12.2643247 / Math.Sqrt(voltageKV * 1000 * (1 + voltageKV * 1000 * .978466e-6));

    public CtfSpectrumFit(Sample[] samples, double voltageKV, double csMM, double amplitude, CtfSpectrumFit basisSource = null)
    {
        if (samples == null || samples.Length < 16 || !double.IsFinite(voltageKV) || !(voltageKV > 0) || !double.IsFinite(csMM) || csMM < 0 || !double.IsFinite(amplitude) || amplitude < 0 || amplitude >= 1)
            throw new ArgumentException("Insufficient spectrum samples or invalid microscope parameters.");
        if (samples.Any(s => !double.IsFinite(s.Power) || s.Power < 0 || !(s.Count > 0) || !double.IsFinite(s.Count) || !double.IsFinite(s.Q2) || !(s.Q2 > 0) || !double.IsFinite(s.Q4) || !double.IsFinite(s.AstigX) || !double.IsFinite(s.AstigY)))
            throw new ArgumentException("Invalid power-spectrum samples.");
        Samples = samples; VoltageKV = voltageKV; CsMM = csMM; Amplitude = amplitude;
        double lambda = Wavelength(voltageKV);
        kDefocus = Math.PI * lambda * 1e4;
        kCs = -.5 * Math.PI * csMM * 1e7 * lambda * lambda * lambda;
        amplitudePhase = Math.Asin(amplitude);
        double lo = Math.Sqrt(samples.Min(s => s.Q2)), hi = Math.Sqrt(samples.Max(s => s.Q2));
        if (!(hi > lo) || !samples.Any(s => s.Power > 0)) throw new ArgumentException("The spectrum has no frequency range or no power.");
        // A smooth nuisance basis, shared across angular sectors, describes broadband power.
        int intervals = Math.Max(1, (int)Math.Round((hi - lo) / .025));
        knots = intervals + 3;
        bool reuse = basisSource != null && basisSource.knots == knots && basisSource.Samples.Length == samples.Length;
        if (reuse) for (int i = 0; i < samples.Length; i++) reuse &= samples[i].Q2 == basisSource.Samples[i].Q2;
        if (reuse)
        {
            basis = basisSource.basis; firstBasis = basisSource.firstBasis; lastBasis = basisSource.lastBasis;
        }
        else
        {
            basis = new double[samples.Length, knots];
            double[] edges = new double[knots + 4];
            for (int i = 0; i < edges.Length; i++) edges[i] = lo + (hi - lo) * Math.Clamp(i - 3, 0, intervals) / intervals;
            for (int i = 0; i < samples.Length; i++)
            {
                double q = Math.Sqrt(samples[i].Q2);
                if (q >= hi) { basis[i, knots - 1] = 1; continue; }
                double[] b = new double[knots + 3];
                for (int j = 0; j < b.Length; j++) b[j] = q >= edges[j] && q < edges[j + 1] ? 1 : 0;
                for (int degree = 1; degree <= 3; degree++)
                    for (int j = 0; j < b.Length - degree; j++)
                        b[j] = (edges[j + degree] > edges[j] ? (q - edges[j]) / (edges[j + degree] - edges[j]) * b[j] : 0)
                             + (edges[j + degree + 1] > edges[j + 1] ? (edges[j + degree + 1] - q) / (edges[j + degree + 1] - edges[j + 1]) * b[j + 1] : 0);
                for (int j = 0; j < knots; j++) basis[i, j] = b[j];
            }
            firstBasis = new int[samples.Length]; lastBasis = new int[samples.Length];
            for (int i = 0; i < samples.Length; i++)
            {
                int first = 0; while (first < knots - 1 && basis[i, first] == 0) first++;
                firstBasis[i] = first; lastBasis[i] = Math.Min(knots, first + 4);
            }
        }
        double scale = PowerScale = Math.Max(1e-30, samples.Sum(s => s.Power * s.Count) / samples.Sum(s => s.Count));
        data = samples.Select(s => s.Power / scale).ToArray();
    }

    internal int KnotCount => knots;
    // Evaluate the accepted GPU spline without rebuilding normal equations or refitting.
    internal (float Background, float Envelope) EvaluateNuisance(int sample, float[] coefficients, int offset)
    {
        float background = 0, envelope = 0;
        for (int j = firstBasis[sample]; j < lastBasis[sample]; j++)
        {
            float b = (float)basis[sample, j];
            background += b * coefficients[offset+j];
            envelope += b * coefficients[offset+j+knots];
        }
        return (background, envelope);
    }
    internal bool HasSameGpuLayout(CtfSpectrumFit other)
    {
        if (knots != other.knots || Samples.Length != other.Samples.Length || kDefocus != other.kDefocus || kCs != other.kCs || amplitudePhase != other.amplitudePhase) return false;
        for (int i = 0; i < Samples.Length; i++)
        {
            var a = Samples[i]; var b = other.Samples[i];
            // Radial averaging with different frame counts can round identical moments
            // differently in double precision. These differences are far below GPU precision.
            static bool Same(double x, double y) => Math.Abs(x-y) <= 1e-12 * Math.Max(1e-30, Math.Max(Math.Abs(x), Math.Abs(y)));
            if (!Same(a.Q2,b.Q2) || !Same(a.Q4,b.Q4) || !Same(a.AstigX,b.AstigX) || !Same(a.AstigY,b.AstigY)) return false;
        }
        return true;
    }
    internal double[] GpuMoments()
    {
        var result = new double[Samples.Length * 4];
        for (int i = 0; i < Samples.Length; i++)
        {
            var s = Samples[i]; result[4*i] = kDefocus*s.Q2; result[4*i+1] = kDefocus*s.AstigX;
            result[4*i+2] = kDefocus*s.AstigY; result[4*i+3] = kCs*s.Q4+amplitudePhase;
        }
        return result;
    }
    internal double[] GpuBasis()
    {
        var result = new double[basis.Length];
        System.Buffer.BlockCopy(basis, 0, result, 0, result.Length * sizeof(double));
        return result;
    }
    internal void CopyGpuData(double[] targetData, double[] counts, double[] currentWeights, int offset)
    {
        Array.Copy(data, 0, targetData, offset, data.Length);
        for (int i = 0; i < Samples.Length; i++)
        {
            counts[offset+i] = Samples[i].Count;
            currentWeights[offset+i] = CurrentWeights == null ? -1 : CurrentWeights[i];
        }
    }
    internal void SetGpuWeights(double[] source, int offset)
    {
        CurrentWeights = new double[Samples.Length];
        Array.Copy(source, offset, CurrentWeights, 0, Samples.Length);
        WeightVersion++;
    }
}
