using System;
using System.Collections.Generic;

namespace Warp.Tools;

/// <summary>Paired target-minus-mean-decoy contrast, in the shared input score units.
/// Inputs are one independently refined maximum per template at the same target anchor.
/// This contrast is not a calibrated significance or false-discovery estimate.</summary>
public static class TemplateMatchSpecificity
{
    public static double Score(float target, IReadOnlyList<float> decoyMaxima)
    {
        if (!float.IsFinite(target)) throw new ArgumentOutOfRangeException(nameof(target));
        if (decoyMaxima == null || decoyMaxima.Count == 0)
            throw new ArgumentException("At least one paired decoy maximum is required.", nameof(decoyMaxima));
        double sum = 0;
        foreach (float value in decoyMaxima)
        {
            if (!float.IsFinite(value))
                throw new ArgumentException("A missing or invalid decoy fit cannot be replaced by zero.", nameof(decoyMaxima));
            sum += value;
        }
        return target - sum / decoyMaxima.Count;
    }
}
