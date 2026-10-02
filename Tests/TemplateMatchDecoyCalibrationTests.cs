using System;
using Warp.Tools;
using Xunit;

namespace Tests;

public class TemplateMatchDecoyCalibrationTests
{
    [Fact]
    public void CountsEqualScoresAndUsesCompletedSearchExposure()
    {
        var result = TemplateMatchDecoyCalibration.Count(5,
            new[] { new[] { 4f, 5f, 8f }, Array.Empty<float>(), new[] { 7f } });

        Assert.Equal(new[] { 2, 0, 1 }, result.CountsBySearch);
        Assert.Equal(3, result.SearchCount);
        Assert.Equal(3L, result.TotalExceedances);
        Assert.Equal(1, result.MeanCount);
        Assert.Equal(1.0 / 3, result.OneCountResolution);
        Assert.False(result.TailUnresolved);
    }

    [Fact]
    public void ZeroExceedancesRetainsUnresolvedTailAndFiniteResolution()
    {
        var result = TemplateMatchDecoyCalibration.Count(100,
            new[] { new[] { 1f, 2f }, Array.Empty<float>() });

        Assert.Equal(0, result.MeanCount);
        Assert.Equal(0.5, result.OneCountResolution);
        Assert.True(result.TailUnresolved);
    }

    [Fact]
    public void MissingSearchesAreNotSilentlyCountedAsZero()
    {
        Assert.Throws<ArgumentException>(() => TemplateMatchDecoyCalibration.Count(1, null));
        Assert.Throws<ArgumentException>(() => TemplateMatchDecoyCalibration.Count(1, Array.Empty<float[]>()));
        Assert.Throws<ArgumentException>(() => TemplateMatchDecoyCalibration.Count(1, new float[][] { null }));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void NonfiniteScoresInvalidateCalibration(float invalid)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TemplateMatchDecoyCalibration.Count(invalid, new[] { Array.Empty<float>() }));
        Assert.Throws<ArgumentException>(() => TemplateMatchDecoyCalibration.Count(1, new[] { new[] { invalid } }));
    }
}
