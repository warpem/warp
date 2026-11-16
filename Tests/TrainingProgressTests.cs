using System;
using System.IO;
using Warp.Tools;
using Xunit;

namespace Tests;

public class TrainingProgressTests
{
    [Fact]
    public void ProgressRoundTripsAndReportsAPercentage()
    {
        string path = Path.Combine(Path.GetTempPath(), "progress-" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            new TrainingProgress(1280, 3200, 4, 8, 0.415f).Write(path);
            var read = TrainingProgress.TryRead(path);
            Assert.Equal(new TrainingProgress(1280, 3200, 4, 8, 0.415f), read);
            Assert.Equal(40, read!.Percent);
            Assert.Null(TrainingProgress.TryRead(path + ".missing"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ProgressLivesNextToTheModelBeingTrained() =>
        Assert.Equal("/models/model-1.pt.progress.json", TrainingProgress.PathFor("/models/model-1.pt"));
}
