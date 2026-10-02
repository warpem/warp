using System;
using System.Collections.Generic;
using System.Linq;

namespace Warp.Tools;

/// <summary>Noise-weighted C and P on a uniform grid of physical frequency squared (1/A²).</summary>
public sealed class TemplateMatchEnvelopeSpectrum
{
    public float MaximumFrequencySquared { get; }
    // Applied to individual physical Fourier samples before histogram deposition.
    public double MinimumFrequencySquared { get; }
    public float[] Cross { get; }
    public float[] Power { get; }
    public TemplateMatchEnvelopeSpectrum(float maximumFrequencySquared, float[] cross, float[] power, double minimumFrequencySquared = 0)
    {
        if (!(maximumFrequencySquared > 0) || !float.IsFinite(maximumFrequencySquared) ||
            cross == null || power == null || cross.Length < 2 || cross.Length != power.Length ||
            !double.IsFinite(minimumFrequencySquared) || minimumFrequencySquared < 0 || minimumFrequencySquared >= maximumFrequencySquared)
            throw new ArgumentException("Invalid envelope spectrum dimensions or frequency range.");
        if (cross.Any(v => !float.IsFinite(v)) || power.Any(v => !float.IsFinite(v) || v < 0))
            throw new ArgumentException("Envelope spectra must be finite with nonnegative model power.");
        MaximumFrequencySquared = maximumFrequencySquared;
        MinimumFrequencySquared = minimumFrequencySquared;
        Cross = cross;
        Power = power;
    }

    public double MeanFrequencySquared()
    {
        double sum = 0, moment = 0;
        for (int i = 0; i < Power.Length; i++) { sum += Power[i]; moment += Power[i] * (double)i; }
        return sum > 0 ? moment / sum * MaximumFrequencySquared / (Power.Length - 1) : double.NaN;
    }
}

public sealed class TemplateMatchEnvelopeFit
{
    public double B, Amplitude, LogAmplitude, Z, Gain, GainAtZeroB, ZAtZeroB, AmplitudeAtZeroB;
    public double SigmaB, SigmaLogAmplitude, CorrelationLogAmplitudeB;
    public double ReferenceFrequencySquared;
    // Moments of the fitted, normalized noise-weighted model power. These also
    // let a whole tomogram use one reporting pivot near its surviving signal.
    public double MeanFrequencySquared, VarianceFrequencySquared;
    public string Status;

    public void SetReferenceFrequencySquared(double reference)
    {
        if (!double.IsFinite(reference) || reference < 0)
            throw new ArgumentException("The amplitude reference frequency squared must be finite and nonnegative.");
        LogAmplitude += B * (ReferenceFrequencySquared - reference) / 4;
        Amplitude = Math.Exp(LogAmplitude);
        ReferenceFrequencySquared = reference;
        if (double.IsFinite(SigmaB))
        {
            double offset = MeanFrequencySquared - reference;
            SigmaLogAmplitude = Math.Sqrt(1 + offset * offset / VarianceFrequencySquared) / Z;
            CorrelationLogAmplitudeB = offset / Math.Sqrt(VarianceFrequencySquared + offset * offset);
        }
    }
}

/// <summary>
/// Joint amplitude/B fit at a fixed pose. Every B evaluation analytically profiles
/// a>=0 in m=a*exp[-B*(q²-qref²)/4]*m0. GPU spectra are accumulated in FP32; this
/// small scalar fit uses double precision. No priors or particle filtering.
/// </summary>
public static class TemplateMatchEnvelope
{
    public const int SpectrumBins = 8192;
    public const double DefaultMinimumB = -5000, DefaultMaximumB = 20000;

    public static TemplateMatchEnvelopeFit Fit(TemplateMatchEnvelopeSpectrum spectrum,
        double referenceFrequencySquared, double minimumB = DefaultMinimumB, double maximumB = DefaultMaximumB)
    {
        if (spectrum == null || !double.IsFinite(referenceFrequencySquared) || referenceFrequencySquared < 0 ||
            !double.IsFinite(minimumB) || !double.IsFinite(maximumB) || minimumB >= maximumB || minimumB > 0 || maximumB < 0)
            throw new ArgumentException("A finite B interval containing zero and a nonnegative reference frequency are required.");
        var frequencies = new List<double>();
        var crosses = new List<double>();
        var powers = new List<double>();
        for (int i = 0; i < spectrum.Power.Length; i++)
            if (spectrum.Power[i] > 0)
            {
                frequencies.Add((double)i * spectrum.MaximumFrequencySquared / (spectrum.Power.Length - 1));
                crosses.Add(spectrum.Cross[i]);
                powers.Add(spectrum.Power[i]);
            }
        double[] q = frequencies.ToArray(), c = crosses.ToArray(), p = powers.ToArray();
        if (q.Length == 0) return Invalid("zero_power", referenceFrequencySquared);

        TemplateMatchEnvelopeFit Evaluate(double b, bool uncertainty = false)
        {
            // Shift all exponents by their maximum. It cancels from Z, and is
            // restored in log amplitude, avoiding overflow for either sign of B.
            double origin = b >= 0 ? q[0] : q[^1];
            double cross = 0, power = 0, first = 0, second = 0;
            for (int i = 0; i < q.Length; i++)
            {
                double envelope = Math.Exp(-0.25 * b * (q[i] - origin));
                double weightedPower = p[i] * envelope * envelope;
                cross += c[i] * envelope;
                power += weightedPower;
                if (uncertainty)
                {
                    double offset = q[i] - referenceFrequencySquared;
                    first += weightedPower * offset;
                    second += weightedPower * offset * offset;
                }
            }
            double z = cross / Math.Sqrt(power), positiveZ = Math.Max(0, z);
            double logAmplitude = cross > 0 ? Math.Log(cross) - Math.Log(power)
                + 0.25 * b * (origin - referenceFrequencySquared) : double.NegativeInfinity;
            var result = new TemplateMatchEnvelopeFit
            {
                B = b, Z = z, Gain = 0.5 * positiveZ * positiveZ,
                LogAmplitude = logAmplitude, Amplitude = Math.Exp(logAmplitude),
                ReferenceFrequencySquared = referenceFrequencySquared, Status = cross > 0 ? "ok" : "nonpositive",
                SigmaB = double.PositiveInfinity, SigmaLogAmplitude = double.PositiveInfinity,
                CorrelationLogAmplitudeB = double.NaN
            };
            if (uncertainty)
            {
                double mean = first / power, moment = second / power;
                double variance = Math.Max(0, moment - mean * mean);
                result.MeanFrequencySquared = mean + referenceFrequencySquared;
                result.VarianceFrequencySquared = variance;
                if (positiveZ > 0 && variance > spectrum.MaximumFrequencySquared * (double)spectrum.MaximumFrequencySquared * 1e-14)
                {
                    result.SigmaB = 4 / (positiveZ * Math.Sqrt(variance));
                    result.SigmaLogAmplitude = Math.Sqrt(moment / variance) / positiveZ;
                    result.CorrelationLogAmplitudeB = mean / Math.Sqrt(moment);
                }
                else if (positiveZ > 0) result.Status = "unidentified";
            }
            return result;
        }

        TemplateMatchEnvelopeFit zero = Evaluate(0), best = zero;
        // Scan first: signed cross-spectra need not give a unimodal B profile.
        const int intervals = 256;
        double step = (maximumB - minimumB) / intervals;
        var grid = new TemplateMatchEnvelopeFit[intervals + 1];
        for (int i = 0; i <= intervals; i++)
        {
            grid[i] = Evaluate(minimumB + i * step);
            if (grid[i].Gain > best.Gain) best = grid[i];
        }
        for (int i = 0; i <= intervals; i++)
        {
            if (i > 0 && i < intervals &&
                !(grid[i].Gain > grid[i - 1].Gain && grid[i].Gain >= grid[i + 1].Gain)) continue;
            // Also search the two edge intervals: an interior optimum very near
            // a bound can otherwise look like a boundary hit on the coarse grid.
            double lower = grid[Math.Max(0, i - 1)].B, upper = grid[Math.Min(intervals, i + 1)].B;
            const double golden = 0.6180339887498948482;
            var left = Evaluate(upper - golden * (upper - lower));
            var right = Evaluate(lower + golden * (upper - lower));
            for (int iteration = 0; iteration < 64 && upper - lower > Math.Min(0.01, (maximumB - minimumB) * 1e-7); iteration++)
            {
                if (left.Gain > right.Gain)
                {
                    upper = right.B; right = left;
                    left = Evaluate(upper - golden * (upper - lower));
                }
                else
                {
                    lower = left.B; left = right;
                    right = Evaluate(lower + golden * (upper - lower));
                }
            }
            if (left.Gain > best.Gain) best = left;
            if (right.Gain > best.Gain) best = right;
        }
        best = Evaluate(best.B, true);
        best.GainAtZeroB = zero.Gain;
        best.ZAtZeroB = zero.Z;
        best.AmplitudeAtZeroB = zero.Amplitude;
        if (best.Status == "ok")
        {
            if (best.B == minimumB) best.Status = "lower_bound";
            if (best.B == maximumB) best.Status = "upper_bound";
        }
        return best;
    }

    private static TemplateMatchEnvelopeFit Invalid(string status, double reference) => new()
    {
        B = double.NaN, Amplitude = double.NaN, LogAmplitude = double.NaN,
        Z = double.NaN, Gain = double.NaN, GainAtZeroB = double.NaN,
        ZAtZeroB = double.NaN, AmplitudeAtZeroB = double.NaN,
        MeanFrequencySquared = double.NaN, VarianceFrequencySquared = double.NaN,
        SigmaB = double.PositiveInfinity, SigmaLogAmplitude = double.PositiveInfinity,
        CorrelationLogAmplitudeB = double.NaN, ReferenceFrequencySquared = reference, Status = status
    };
}
