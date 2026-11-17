using System;
using System.IO;
using Warp;
using Xunit;

namespace Tests;

public class Noise2MicPretrainedTests
{
    [Fact]
    public void PretrainedTokenResolvesToTheInstalledModelAndOtherPathsStayAsTheyAre()
    {
        string file = Path.Combine(Path.GetTempPath(), "noise2mic-" + Guid.NewGuid().ToString("N") + ".pt");
        string previous = Environment.GetEnvironmentVariable(Noise2MicTorch.PretrainedVariable);
        try
        {
            Environment.SetEnvironmentVariable(Noise2MicTorch.PretrainedVariable, file);
            var missing = Assert.Throws<FileNotFoundException>(() => Noise2MicTorch.ResolvePath(Noise2MicTorch.PretrainedModel));
            Assert.Contains(file, missing.Message);
            File.WriteAllText(file, "weights");
            Assert.Equal(file, Noise2MicTorch.ResolvePath(Noise2MicTorch.PretrainedModel));
            Assert.Equal("models/session.pt", Noise2MicTorch.ResolvePath("models/session.pt"));

            // Without the variable, an installation keeps it in models/ beside the executables.
            Environment.SetEnvironmentVariable(Noise2MicTorch.PretrainedVariable, null);
            Assert.Equal(Path.Combine(AppContext.BaseDirectory, "models", Noise2MicTorch.PretrainedFileName), Noise2MicTorch.PretrainedPath);
        }
        finally
        {
            Environment.SetEnvironmentVariable(Noise2MicTorch.PretrainedVariable, previous);
            File.Delete(file);
        }
    }
}
