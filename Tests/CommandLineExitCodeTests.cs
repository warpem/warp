using CommandLine;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CommandLineExitCodeTests
{
    private sealed class Options
    {
        [Option("input", Required = true)]
        public string Input { get; set; }
    }

    [Fact]
    public void ValidArgumentsReturnSuccess()
    {
        var result = Parser.Default.ParseArguments<Options>(new[] { "--input", "file.mrc" });

        Assert.Equal(0, CommandLineParserHelper.GetExitCode(result));
    }

    [Fact]
    public void InvalidArgumentsReturnUsageError()
    {
        var result = Parser.Default.ParseArguments<Options>(new[] { "--unknown" });

        Assert.Equal(CommandLineParserHelper.InvalidArgumentsExitCode,
                     CommandLineParserHelper.GetExitCode(result));
    }

    [Theory]
    [InlineData("--help")]
    [InlineData("--version")]
    public void InformationalRequestsReturnSuccess(string argument)
    {
        var result = Parser.Default.ParseArguments<Options>(new[] { argument });

        Assert.Equal(0, CommandLineParserHelper.GetExitCode(result));
    }
}
