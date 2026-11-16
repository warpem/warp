using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class TemplateMatchSpecificityTests
{
    [Fact]
    public void ContrastUsesMeanOfIndependentMaximaAndPreservesLikelihoodUnits()
    {
        // One shape-only match versus a structure-specific match. A single lucky
        // decoy must not replace the mean and reverse their ordering.
        double shape = TemplateMatchSpecificity.Score(50, new[] { 48f, 44f, 46f });
        double specific = TemplateMatchSpecificity.Score(40, new[] { 35f, 20f, 5f });
        Assert.Equal(4, shape);
        Assert.Equal(20, specific);
        Assert.True(specific > shape);
        Assert.Equal(60, TemplateMatchSpecificity.Score(120, new[] { 105f, 60f, 15f }));
        Assert.Equal(0, TemplateMatchSpecificity.Score(17, new[] { 17f, 17f }));
        Assert.Equal(4, TemplateMatchSpecificity.Score(-6, new[] { -10f }));
    }

    [Fact]
    public void MissingOrInvalidDecoysCannotGiveSpuriousSpecificity()
    {
        Assert.Throws<ArgumentException>(() => TemplateMatchSpecificity.Score(1, null));
        Assert.Throws<ArgumentException>(() => TemplateMatchSpecificity.Score(1, Array.Empty<float>()));
        foreach (float invalid in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            Assert.Throws<ArgumentException>(() => TemplateMatchSpecificity.Score(1, new[] { invalid }));
            Assert.Throws<ArgumentOutOfRangeException>(() => TemplateMatchSpecificity.Score(invalid, new[] { 1f }));
        }
    }

    [Fact]
    public void WorkerSerializationRetainsDecoyMapsAndHandedness()
    {
        var command = new NamedSerializableObject("test", new ProcessingOptionsTomoFullMatch
        {
            DecoyTemplatePaths = new[] { "/existing/decoy 1.mrc", "/existing/decoy2.mrc" }, DecoyFlipX = true,
            RefineCalibrate = true, RefineBoundedScore = true
        });
        var roundtrip = JsonSerializer.Deserialize<NamedSerializableObject>(JsonSerializer.Serialize(command));
        var options = Assert.IsType<ProcessingOptionsTomoFullMatch>(Assert.Single(roundtrip.Content));
        Assert.Equal(new[] { "/existing/decoy 1.mrc", "/existing/decoy2.mrc" }, options.DecoyTemplatePaths);
        Assert.True(options.DecoyFlipX);
        Assert.True(options.RefineCalibrate);
        Assert.True(options.RefineBoundedScore);
    }

    [Fact]
    public void FinalHypothesisSelectionUsesProjectionUnlessBoundedScoringIsRequested()
    {
        // Data (5,0), model A=(1,0), B=(2,1). Unconstrained amplitude favors A,
        // but the common amplitude interval [1,2] gives residual² 9 for A, 5 for B.
        Type solutionType = typeof(TiltSeries).GetNestedType("MatchSolution", BindingFlags.NonPublic);
        var solutions = (IList)Activator.CreateInstance(typeof(System.Collections.Generic.List<>).MakeGenericType(solutionType));
        foreach (var item in new[] { (Cross: 5.0, Power: 1.0), (Cross: 10.0, Power: 5.0) })
        {
            object solution = Activator.CreateInstance(solutionType, nonPublic: true);
            solutionType.GetField("Z").SetValue(solution, item.Cross / Math.Sqrt(item.Power));
            double[] statistics = new double[14];
            statistics[0] = item.Cross; statistics[1] = item.Power;
            solutionType.GetField("TiltStatistics").SetValue(solution, statistics);
            solutions.Add(solution);
        }
        MethodInfo select = typeof(TiltSeries).GetMethod("SelectMatchSolution", BindingFlags.Static | BindingFlags.NonPublic);
        Assert.Same(solutions[0], select.Invoke(null, new object[] { solutions, (1.0, 2.0), false }));
        Assert.Same(solutions[0], select.Invoke(null, new object[] { solutions, null, false }));
        object best = select.Invoke(null, new object[] { solutions, (1.0, 2.0), true });
        Assert.Same(solutions[1], best);
        Assert.True((double)solutionType.GetField("Z").GetValue(solutions[0]) >
            (double)solutionType.GetField("Z").GetValue(best));
    }

    [Fact]
    public void SpatialSuppressionUsesCorrectedScoreAndKeepsAnchorIdentity()
    {
        Type peakType = typeof(TiltSeries).Assembly.GetType("Warp.ParticlePeak");
        Array peaks = Array.CreateInstance(peakType, 3);
        float[] targets = { 50, 40, 30 }, decoys = { 46, 20, 5 };
        for (int i = 0; i < 3; i++)
        {
            object peak = Activator.CreateInstance(peakType);
            peakType.GetField("PositionF").SetValue(peak, new float3(i == 2 ? 100 : i, 0, 0));
            peakType.GetField("Score").SetValue(peak, (float)TemplateMatchSpecificity.Score(targets[i], new[] { decoys[i] }));
            peakType.GetField("AnchorIndex").SetValue(peak, i);
            peaks.SetValue(peak, i);
        }
        MethodInfo suppress = typeof(TiltSeries).GetMethod("SuppressMatchPeaks", BindingFlags.Static | BindingFlags.NonPublic);
        var result = ((IEnumerable)suppress.Invoke(null, new object[] { peaks, 10f })).Cast<object>().ToArray();
        Assert.Equal(new[] { 2, 1 }, result.Select(p => (int)peakType.GetField("AnchorIndex").GetValue(p)));
    }
}
