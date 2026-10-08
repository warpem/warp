using System;
using System.Collections.Generic;
using System.Linq;
using ZLinq;

namespace Warp.Tools;

/// <summary>Shared statistical conventions for native tilt-series template matching.</summary>
public static class TemplateMatchStatistics
{
    public static double HybridWeight(double totalPower, double detectorPower, double multiplicity)
    {
        if (!(detectorPower > 0) || !double.IsFinite(detectorPower) || totalPower < 0 ||
            !double.IsFinite(totalPower) || !double.IsFinite(multiplicity) || multiplicity < 1)
            throw new ArgumentOutOfRangeException(nameof(totalPower));
        return 1 / (detectorPower + multiplicity * Math.Max(0, totalPower - detectorPower));
    }

    public static double BoundedGain(double cross, double power, double lower, double upper)
    {
        if (!(power > 0) || !double.IsFinite(cross) || !double.IsFinite(power)) return double.NegativeInfinity;
        if (lower < 0 || upper < lower || !double.IsFinite(lower) || !double.IsFinite(upper))
            throw new ArgumentOutOfRangeException(nameof(lower));
        double amplitude = Math.Clamp(cross / power, lower, upper);
        return amplitude * cross - 0.5 * amplitude * amplitude * power;
    }

    /// <summary>Starting band followed by frequency doublings, capped at the requested final band.</summary>
    public static float[] ResolutionSchedule(float coarseResolution, float finalResolution)
    {
        if (!(coarseResolution > 0) || !float.IsFinite(coarseResolution))
            throw new ArgumentOutOfRangeException(nameof(coarseResolution));
        if (!(finalResolution > 0) || !float.IsFinite(finalResolution) || finalResolution > coarseResolution)
            throw new ArgumentOutOfRangeException(nameof(finalResolution));
        var bands = new List<float> { coarseResolution };
        while (bands[^1] > finalResolution)
            bands.Add(Math.Max(finalResolution, bands[^1] / 2));
        return bands.ToArray();
    }

    /// <summary>Carry a broad coarse search into finer bands without multiplying its
    /// full hypothesis count by the growing Fourier area. Retain at least eight alternatives;
    /// equal-band calibration passes keep the original budget.</summary>
    public static int ContinuationHypotheses(int initial, float coarseResolution, float nextResolution)
    {
        if (initial < 1 || !(coarseResolution > 0) || !float.IsFinite(coarseResolution) ||
            !(nextResolution > 0) || !float.IsFinite(nextResolution) || nextResolution > coarseResolution)
            throw new ArgumentOutOfRangeException(nameof(initial));
        double ratio = nextResolution / coarseResolution;
        return Math.Min(initial, Math.Max(8, (int)Math.Ceiling(initial * ratio * ratio)));
    }

    public static double Quantile(IEnumerable<double> values, double fraction)
    {
        double[] sorted = values.Where(double.IsFinite).OrderBy(v => v).ToArray();
        if (sorted.Length == 0) throw new ArgumentException("No finite calibration values.");
        double at = Math.Clamp(fraction, 0, 1) * (sorted.Length - 1);
        int lo = (int)at;
        return sorted[lo] + (sorted[Math.Min(lo + 1, sorted.Length - 1)] - sorted[lo]) * (at - lo);
    }

    // Reweight fixed-pose sufficient statistics; the reference alone receives exp(deltaB*q²/4).
    // Warp's B convention is negative for attenuation.
    public static (double Cross, double Power) Reweight(TemplateMatchEnvelopeSpectrum s, double deltaB)
        => Reweight(s, EnvelopeFactors(s.MaximumFrequencySquared, s.Cross.Length, deltaB));

    public static double[] EnvelopeFactors(float maximumQ2, int bins, double deltaB)
    {
        if (bins < 2) throw new ArgumentOutOfRangeException(nameof(bins));
        var factors = new double[bins];
        for (int i = 0; i < bins; i++)
            factors[i] = Math.Exp(Math.Clamp(deltaB * (i * (double)maximumQ2 / (bins - 1)) / 4, -80, 80));
        return factors;
    }

    public static (double Cross, double Power) Reweight(TemplateMatchEnvelopeSpectrum s, ReadOnlySpan<double> factors)
    {
        if (factors.Length != s.Cross.Length) throw new ArgumentException("Envelope grid does not match the spectrum.", nameof(factors));
        double cross = 0, power = 0;
        for (int i = 0; i < s.Cross.Length; i++)
        {
            double a = factors[i];
            cross += s.Cross[i] * a;
            power += s.Power[i] * a * a;
        }
        return (cross, power);
    }
}
