using System;
using System.Linq;
using Warp.Tools;
using Xunit;

namespace Tests;

public class TemplateMatchEnvelopeTests
{
    private const float MaximumQ2 = .01f;
    private const double Pivot = .002;

    private static TemplateMatchEnvelopeSpectrum Synthetic(double amplitude, double b)
    {
        const int bins = 301;
        var power = new float[bins];
        var cross = new float[bins];
        for (int i = 1; i < bins; i++)
        {
            double q2 = (double)i * MaximumQ2 / (bins - 1);
            power[i] = (float)(Math.Exp(-q2 * 300) * (1.1 + Math.Sin(i * .17)));
            cross[i] = (float)(power[i] * amplitude * Math.Exp(-b * (q2 - Pivot) / 4));
        }
        return new(MaximumQ2, cross, power);
    }

    [Theory]
    [InlineData(.4, -2200)]
    [InlineData(2.7, 1300)]
    [InlineData(14, 9000)]
    [InlineData(1.1, 0)]
    [InlineData(.9, -4999)]
    [InlineData(1.8, 19999)]
    public void JointFitRecoversAmplitudeAndBothSignsOfB(double amplitude, double b)
    {
        var spectrum = Synthetic(amplitude, b);
        var fit = TemplateMatchEnvelope.Fit(spectrum, Pivot);
        Assert.Equal("ok", fit.Status);
        Assert.InRange(Math.Abs(fit.B - b), 0, .03);
        Assert.InRange(Math.Abs(Math.Log(fit.Amplitude / amplitude)), 0, 1e-5);
        Assert.True(fit.Gain >= fit.GainAtZeroB);
        Assert.True(double.IsFinite(fit.SigmaB) && fit.SigmaB > 0);

        // Multiplying the data must rescale a, never be absorbed by B.
        var scaled = new TemplateMatchEnvelopeSpectrum(MaximumQ2,
            spectrum.Cross.Select(v => v * 3).ToArray(), spectrum.Power);
        var scaledFit = TemplateMatchEnvelope.Fit(scaled, Pivot);
        Assert.InRange(Math.Abs(scaledFit.B - b), 0, .03);
        Assert.InRange(Math.Abs(Math.Log(scaledFit.Amplitude / (3 * amplitude))), 0, 1e-5);
    }

    [Fact]
    public void ChangingReferenceFrequencyPreservesModelAndProfileLikelihood()
    {
        var spectrum = Synthetic(2.7, 1600);
        var first = TemplateMatchEnvelope.Fit(spectrum, Pivot);
        const double otherPivot = .004;
        var second = TemplateMatchEnvelope.Fit(spectrum, otherPivot);
        Assert.Equal(first.B, second.B);
        Assert.Equal(first.Gain, second.Gain);
        Assert.InRange(Math.Abs(second.LogAmplitude - first.LogAmplitude + first.B * (otherPivot - Pivot) / 4), 0, 1e-12);
        first.SetReferenceFrequencySquared(otherPivot);
        Assert.InRange(Math.Abs(first.LogAmplitude - second.LogAmplitude), 0, 1e-12);
        Assert.InRange(Math.Abs(first.SigmaLogAmplitude - second.SigmaLogAmplitude), 0, 1e-12);
        Assert.InRange(Math.Abs(first.CorrelationLogAmplitudeB - second.CorrelationLogAmplitudeB), 0, 1e-12);
    }

    [Fact]
    public void FisherUncertaintyMatchesIndependentTwoByTwoInformationMatrix()
    {
        var spectrum = Synthetic(1.7, 1100);
        var fit = TemplateMatchEnvelope.Fit(spectrum, Pivot);
        double aa = 0, ab = 0, bb = 0;
        for (int i = 0; i < spectrum.Power.Length; i++)
        {
            double r = (double)i * MaximumQ2 / (spectrum.Power.Length - 1) - Pivot;
            double m2 = spectrum.Power[i] * fit.Amplitude * fit.Amplitude * Math.Exp(-fit.B * r / 2);
            aa += m2; ab += -r / 4 * m2; bb += r * r / 16 * m2;
        }
        double determinant = aa * bb - ab * ab;
        Assert.InRange(Math.Abs(fit.SigmaB / Math.Sqrt(aa / determinant) - 1), 0, 1e-10);
        Assert.InRange(Math.Abs(fit.SigmaLogAmplitude / Math.Sqrt(bb / determinant) - 1), 0, 1e-10);
        Assert.InRange(Math.Abs(fit.CorrelationLogAmplitudeB + ab / Math.Sqrt(aa * bb)), 0, 1e-10);
    }

    [Theory]
    [InlineData(-8000, "lower_bound", -5000)]
    [InlineData(30000, "upper_bound", 20000)]
    public void OutOfRangeOptimaAreFlagged(double truth, string status, double boundary)
    {
        var fit = TemplateMatchEnvelope.Fit(Synthetic(1, truth), Pivot);
        Assert.Equal(status, fit.Status);
        Assert.Equal(boundary, fit.B);
    }

    [Fact]
    public void UnidentifiableAndNonpositiveCasesDoNotLookLikeGoodFits()
    {
        Assert.Equal("zero_power", TemplateMatchEnvelope.Fit(new(MaximumQ2, new float[3], new float[3]), Pivot).Status);
        var narrow = TemplateMatchEnvelope.Fit(new(MaximumQ2, new float[] { 0, 3, 0 }, new float[] { 0, 1, 0 }), Pivot);
        Assert.Equal("unidentified", narrow.Status);
        Assert.True(double.IsPositiveInfinity(narrow.SigmaB));
        var negative = TemplateMatchEnvelope.Fit(Synthetic(-2, 1300), Pivot);
        Assert.Equal("nonpositive", negative.Status);
        Assert.Equal(0, negative.Amplitude);
        Assert.Equal(0, negative.Gain);
    }
}
