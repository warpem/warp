using System.Collections.Generic;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

public class ClusterVarParserTests
{
    [Fact]
    public void OneToken_KeyEqualsValue()
    {
        var r = ClusterVarParser.Parse(new[] { "partition=gpu" });
        Assert.Equal("gpu", r["partition"]);
    }

    [Fact]
    public void TwoTokens_KeyEquals_ThenValue()
    {
        var r = ClusterVarParser.Parse(new[] { "partition=", "gpu" });
        Assert.Equal("gpu", r["partition"]);
    }

    [Fact]
    public void TwoTokens_Key_ThenEqualsValue()
    {
        var r = ClusterVarParser.Parse(new[] { "partition", "=gpu" });
        Assert.Equal("gpu", r["partition"]);
    }

    [Fact]
    public void ThreeTokens_Key_Equals_Value()
    {
        var r = ClusterVarParser.Parse(new[] { "partition", "=", "gpu" });
        Assert.Equal("gpu", r["partition"]);
    }

    [Fact]
    public void MultiplePairs_MixedSpacing()
    {
        var r = ClusterVarParser.Parse(new[] { "partition=gpu", "walltime", "=", "04:00:00", "mem=", "16G" });
        Assert.Equal("gpu", r["partition"]);
        Assert.Equal("04:00:00", r["walltime"]);
        Assert.Equal("16G", r["mem"]);
    }

    [Fact]
    public void QuotedValueWithSpace_IsOneToken()
    {
        var r = ClusterVarParser.Parse(new[] { "account=my project" });
        Assert.Equal("my project", r["account"]);
    }

    [Fact]
    public void TrimsWhitespaceAroundEquals_WithinSingleToken()
    {
        var r = ClusterVarParser.Parse(new[] { "a = 1" });
        Assert.Equal("1", r["a"]);
    }

    [Fact]
    public void NullInput_ReturnsEmpty()
    {
        var r = ClusterVarParser.Parse(null);
        Assert.Empty(r);
    }

    [Fact]
    public void KeyWithNoEquals_Throws()
    {
        Assert.Throws<System.ArgumentException>(() => ClusterVarParser.Parse(new[] { "partition" }));
    }

    [Fact]
    public void LeadingEquals_Throws()
    {
        Assert.Throws<System.ArgumentException>(() => ClusterVarParser.Parse(new[] { "=gpu" }));
    }
}
