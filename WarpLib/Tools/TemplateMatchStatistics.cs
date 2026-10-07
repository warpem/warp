using System;
using System.Collections.Generic;
using System.Linq;
using ZLinq;

namespace Warp.Tools;

/// <summary>Shared statistical conventions for native tilt-series template matching.</summary>
public static class TemplateMatchStatistics
{
    /// <summary>Remove the observed background before zero-padding an image-boundary patch.</summary>
    public static void CopyCenteredPatch(float[] source, int width, int height, int x, int y, int box, float[] destination)
    {
        Array.Clear(destination);
        int left = Math.Max(0, x), top = Math.Max(0, y);
        int right = Math.Min(width, x + box), bottom = Math.Min(height, y + box);
        if (left >= right || top >= bottom) return;
        double sum = 0;
        for (int row = top; row < bottom; row++)
            for (int col = left; col < right; col++) sum += source[row * width + col];
        float mean = (float)(sum / ((right - left) * (bottom - top)));
        for (int row = top; row < bottom; row++)
            for (int col = left; col < right; col++)
                destination[(row-y)*box+col-x] = source[row*width+col] - mean;
    }
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
    {
        double cross = 0, power = 0;
        for (int i = 0; i < s.Cross.Length; i++)
        {
            double q2 = i * (double)s.MaximumFrequencySquared / (s.Cross.Length - 1);
            double a = Math.Exp(Math.Clamp(deltaB * q2 / 4, -80, 80));
            cross += s.Cross[i] * a;
            power += s.Power[i] * a * a;
        }
        return (cross, power);
    }
}
