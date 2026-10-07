using System;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfDiagnosticTests
{
    [Fact]
    public void ClosedFormAlignmentPreservesPhaseOnBothBranchesAndWithoutCs()
    {
        foreach (float cs in new[] { 0f, -300f })
            foreach (float df in new[] { 0f, 30f, 1400f })
                foreach (float q2 in new[] { 1e-6f, .001f, .02f, .15f })
                {
                    if (cs == 0 && df == 0) continue;
                    float target = df * q2 + cs * q2 * q2;
                    float result = CtfFitDiagnostics.AlignFrequencySquared(target, df, cs, q2);
                    Assert.InRange(MathF.Abs(result - q2), 0, Math.Max(1e-8f, q2 * 2e-5f));
                }
        Assert.True(float.IsNaN(CtfFitDiagnostics.AlignFrequencySquared(10, 1, -1, .1f)));
        Assert.True(float.IsNaN(CtfFitDiagnostics.AlignFrequencySquared(1, 0, 0, .1f)));
    }
}
