using System;
using System.IO;
using Warp;
using Xunit;

namespace Tests;

public class InstalledModelsTests
{
    [Fact]
    public void ModelsAreFoundInTheFolderTheVariableNamesOrBesideTheExecutables()
    {
        string folder = Path.Combine(Path.GetTempPath(), "warp-models-" + Guid.NewGuid().ToString("N"));
        string previous = Environment.GetEnvironmentVariable(InstalledModels.Variable);
        try
        {
            Directory.CreateDirectory(folder);
            Environment.SetEnvironmentVariable(InstalledModels.Variable, folder);
            var missing = Assert.Throws<FileNotFoundException>(() => Noise2MicTorch.ResolvePath(Noise2MicTorch.PretrainedModel));
            Assert.Contains(folder, missing.Message);
            Assert.Throws<FileNotFoundException>(() => JunkNetTorch.ResolvePath(null));

            string pretrained = Path.Combine(folder, Noise2MicTorch.PretrainedFileName), junk = Path.Combine(folder, JunkNetTorch.ModelFileName);
            File.WriteAllText(pretrained, "weights"); File.WriteAllText(junk, "weights");
            Assert.Equal(pretrained, Noise2MicTorch.ResolvePath(Noise2MicTorch.PretrainedModel));
            Assert.Equal(junk, JunkNetTorch.ResolvePath(""));
            // Other paths stay as they are.
            Assert.Equal("models/session.pt", Noise2MicTorch.ResolvePath("models/session.pt"));
            Assert.Equal("other.pt", JunkNetTorch.ResolvePath("other.pt"));

            Environment.SetEnvironmentVariable(InstalledModels.Variable, null);
            Assert.Equal(Path.Combine(AppContext.BaseDirectory, "models"), InstalledModels.Folder);
        }
        finally
        {
            Environment.SetEnvironmentVariable(InstalledModels.Variable, previous);
            if (Directory.Exists(folder)) Directory.Delete(folder, true);
        }
    }
}
