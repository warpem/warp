using System.Collections.Generic;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

public class TemplateRendererTests
{
    [Fact]
    public void SubstitutesSimplePlaceholder()
    {
        var r = TemplateRenderer.Render("a {{x}} b", new Dictionary<string, string> { ["x"] = "1" });
        Assert.Equal("a 1 b", r);
    }

    [Fact]
    public void ToleratesWhitespaceInsidePlaceholder()
    {
        var r = TemplateRenderer.Render("[{{  x  }}]", new Dictionary<string, string> { ["x"] = "1" });
        Assert.Equal("[1]", r);
    }

    [Fact]
    public void ReplacesAllOccurrences()
    {
        var r = TemplateRenderer.Render("{{x}}-{{x}}", new Dictionary<string, string> { ["x"] = "z" });
        Assert.Equal("z-z", r);
    }

    [Fact]
    public void ValueWithDollarIsLiteral_NotRegexBackref()
    {
        var r = TemplateRenderer.Render("{{x}}", new Dictionary<string, string> { ["x"] = "$1foo$$" });
        Assert.Equal("$1foo$$", r);
    }

    [Fact]
    public void LeavesNonPlaceholderShellSyntaxAlone()
    {
        var r = TemplateRenderer.Render("id=$(hostname)-$$ {{x}}", new Dictionary<string, string> { ["x"] = "1" });
        Assert.Equal("id=$(hostname)-$$ 1", r);
    }

    [Fact]
    public void UnfilledPlaceholder_ThrowsListingName()
    {
        var ex = Assert.Throws<System.ArgumentException>(() =>
            TemplateRenderer.Render("{{x}} {{y}}", new Dictionary<string, string> { ["x"] = "1" }));
        Assert.Contains("y", ex.Message);
    }

    [Fact]
    public void MultipleUnfilledPlaceholders_AllListed()
    {
        var ex = Assert.Throws<System.ArgumentException>(() =>
            TemplateRenderer.Render("{{a}} {{b}}", new Dictionary<string, string>()));
        Assert.Contains("a", ex.Message);
        Assert.Contains("b", ex.Message);
    }
}
