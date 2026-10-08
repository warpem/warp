using System;
using System.Linq;
using Warp.Tools;
using Xunit;

namespace Tests;

public class TemplateMatchStatisticsTests
{
    [Theory]
    [InlineData(56,20,20,56)]
    [InlineData(56,20,10,14)]
    [InlineData(56,20,6,8)]
    [InlineData(32,20,10,8)]
    [InlineData(8,12,6,8)]
    [InlineData(4,12,6,4)]
    public void ContinuationBudgetKeepsAlternativesAndPreservesEqualBandPasses(int initial,float coarse,float next,int expected)
        => Assert.Equal(expected,TemplateMatchStatistics.ContinuationHypotheses(initial,coarse,next));

    [Fact]
    public void BoundaryPaddingDoesNotTurnAConstantBackgroundIntoAnEdge()
    {
        float[] destination = new float[16];
        TemplateMatchPreparationReference.CopyCenteredPatch(Enumerable.Repeat(7f, 25).ToArray(), 5, 5, -2, -1, 4, destination);
        Assert.All(destination, v => Assert.Equal(0, v));
        float[] source = Enumerable.Range(0, 25).Select(i => (float)i).ToArray();
        TemplateMatchPreparationReference.CopyCenteredPatch(source, 5, 5, -2, -1, 4, destination);
        Assert.Equal(new[]{0f,0,0,0,0,0,-5.5f,-4.5f,0,0,-.5f,.5f,0,0,4.5f,5.5f}, destination);
    }
    [Fact]
    public void IndependentNoiseAndSharedClutterHaveDifferentMultiplicityScaling()
    {
        Assert.Equal(.25, TemplateMatchStatistics.HybridWeight(4, 4, 20));
        Assert.Equal(1.0 / 44, TemplateMatchStatistics.HybridWeight(6, 4, 20), 12);
        Assert.Equal(.25, TemplateMatchStatistics.HybridWeight(3, 4, 20));
        Assert.Equal(1.0 / 6, TemplateMatchStatistics.HybridWeight(6, 4, 1), 12);
    }
    [Fact]
    public void BoundedAmplitudeIsTheMaximumOfTheConstrainedLikelihood()
    {
        foreach (double c in new[] { -2.0, 0, 1, 10, 100 })
        {
            double score = TemplateMatchStatistics.BoundedGain(c, 3, .5, 2);
            for (int i = 0; i <= 1000; i++)
            {
                double a = .5 + 1.5 * i / 1000;
                Assert.True(score >= a * c - .5 * a * a * 3 - 1e-12);
            }
        }
    }
    [Fact]
    public void ScheduleDoublesFrequencyFromCoarseLimitAndCapsAtRequestedLimit()
    {
        Assert.Equal(new[] { 20f }, TemplateMatchStatistics.ResolutionSchedule(20, 20));
        Assert.Equal(new[] { 20f, 10f, 6f }, TemplateMatchStatistics.ResolutionSchedule(20, 6));
        Assert.Equal(new[] { 24f, 12f, 6f, 3f }, TemplateMatchStatistics.ResolutionSchedule(24, 3));
        Assert.Equal(new[] { 12f, 10f }, TemplateMatchStatistics.ResolutionSchedule(12, 10));
        Assert.Equal(new[] { 4f, 2f }, TemplateMatchStatistics.ResolutionSchedule(4, 2));
    }
    [Fact]
    public void ScheduleRejectsInvalidOrCoarserFinalLimits()
    {
        foreach (float invalid in new[] { 0f, -1f, float.NaN, float.PositiveInfinity })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => TemplateMatchStatistics.ResolutionSchedule(invalid, 6));
            Assert.Throws<ArgumentOutOfRangeException>(() => TemplateMatchStatistics.ResolutionSchedule(20, invalid));
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => TemplateMatchStatistics.ResolutionSchedule(6, 12));
    }
    [Fact]
    public void ReweightedSufficientStatisticsMatchExplicitReferenceAttenuation()
    {
        float[] cross = { 2, 3, 5 }, power = { 1, 4, 9 };
        var spectrum = new TemplateMatchEnvelopeSpectrum(.04f, cross, power);
        var terms = TemplateMatchStatistics.Reweight(spectrum, -120);
        double c = 0, p = 0;
        for (int i = 0; i < 3; i++)
        {
            double a = Math.Exp(-120 * i * (double)spectrum.MaximumFrequencySquared / 8);
            c += cross[i] * a; p += power[i] * a * a;
        }
        Assert.Equal(c, terms.Cross, 12); Assert.Equal(p, terms.Power, 12);
    }
}
