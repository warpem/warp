using System;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class RelionParticleSeriesExportTests
{
    [Fact]
    public void VisibleTiltIndicesFollowSetBitOrder()
    {
        int[] result = RelionParticleSeriesExport.GetVisibleTiltIndices(
            new[] { 0, 2, 4, 7 },
            new[] { true, false, true, false });

        Assert.Equal(new[] { 0, 4 }, result);
    }

    [Fact]
    public void VisibleTiltIndicesRejectMismatchedMetadata()
    {
        Assert.Throws<ArgumentException>(() =>
            RelionParticleSeriesExport.GetVisibleTiltIndices(
                new[] { 0, 1 },
                new[] { true }));
    }

    [Fact]
    public void NoVisibleTiltsProduceEmptyStackSelection()
    {
        int[] result = RelionParticleSeriesExport.GetVisibleTiltIndices(
            new[] { 0, 2, 4 },
            new[] { false, false, false });

        Assert.Empty(result);
    }

    [Fact]
    public void VirtualTomogramUsesParticleSampling()
    {
        int3 result = RelionParticleSeriesExport.GetVirtualTomogramDimensions(
            new float3(6144, 3072, 768),
            3f);

        Assert.Equal(new int3(2048, 1024, 256), result);
    }

    [Theory]
    [InlineData(false, -1f)]
    [InlineData(true, 1f)]
    public void RelionHandTracksDefocusInversion(bool inverted, float expected)
    {
        Assert.Equal(expected, RelionParticleSeriesExport.GetRelionHand(inverted));
    }

    [Fact]
    public void PhaseShiftConvertsPiUnitsToDegrees()
    {
        Assert.Equal(45M, RelionParticleSeriesExport.GetPhaseShiftDegrees(0.25M));
    }
}
