using System;
using System.Globalization;
using System.IO;
using Warp;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;
using Xunit;

namespace Tests;

public class BlobPickerTests
{
    [Fact]
    public void CommandDefaultsSurviveQueueSerialization()
    {
        var task = new TaskItem { Main = new[] { WorkerCommands.MicrographPickBlobs("mic.mrc", 12.5f) } };
        var command = TaskItem.FromJson(task.ToJson()).Main[0];
        Assert.Equal(WorkerCommandNames.MicrographPickBlobs, command.Name);
        Assert.Equal("mic.mrc", command.Content[0]);
        Assert.Equal(12.5f, command.Content[1]);
        Assert.Equal(0f, command.Content[2]);
        Assert.Equal(25f, command.Content[3]);
        command = WorkerCommands.MicrographPickBlobs("mic.mrc", 12.5f, 8, 20.25f);
        Assert.Equal(8f, command.Content[2]);
        Assert.Equal(20.25f, command.Content[3]);
    }

    [Theory]
    [InlineData(0, 0, 1)]
    [InlineData(-1, 0, 1)]
    [InlineData(float.NaN, 0, 1)]
    [InlineData(float.PositiveInfinity, 0, 1)]
    [InlineData(5, -1, 1)]
    [InlineData(5, 5, 1)]
    [InlineData(5, float.NaN, 1)]
    [InlineData(5, 1, 0)]
    [InlineData(5, 1, float.PositiveInfinity)]
    public void InvalidOptionsFailBeforeTouchingGpu(float radius, float inner, float separation)
        => Assert.Throws<ArgumentOutOfRangeException>(() => WorkerCommands.MicrographPickBlobs("mic.mrc", radius, inner, separation));

    [Fact]
    public void StarContainsScoresAndPowerAndEmptyResultsReplaceOldPicks()
    {
        string path = Path.Combine(Path.GetTempPath(), "warp-blob-" + Guid.NewGuid() + ".star");
        CultureInfo original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("de-DE");
            BlobPicker.WriteStar(path, "mic.mrc", new[] { new float4(22, 34, .75f, 2.5f) });
            Star table = new Star(path);
            Assert.Equal(1, table.RowCount);
            Assert.Equal("22", table.GetColumn("rlnCoordinateX")[0]);
            Assert.Equal("34", table.GetColumn("rlnCoordinateY")[0]);
            Assert.Equal("mic.mrc", table.GetColumn("rlnMicrographName")[0]);
            Assert.Equal("0.75", table.GetColumn("rlnAutopickFigureOfMerit")[0]);
            Assert.Equal("2.5", table.GetColumn("wrpAutopickPower")[0]);
            BlobPicker.WriteStar(path, "mic.mrc", Array.Empty<float4>());
            table = new Star(path);
            Assert.Equal(0, table.RowCount);
            Assert.True(table.HasColumn("wrpAutopickPower"));
        }
        finally { CultureInfo.CurrentCulture = original; File.Delete(path); }
    }

    [TemplateMatchCudaFact]
    public void CudaPickerUsesManagedAbiAndRejectsStacks()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var dims = new int3(65, 61, 1);
            float[] values = new float[dims.Elements()];
            for (int y = 0; y < dims.Y; y++)
                for (int x = 0; x < dims.X; x++)
                    values[y * dims.X + x] = (x - 32) * (x - 32) + (y - 30) * (y - 30) <= 25 ? -3 : 0;
            using Image image = new Image(values, dims);
            float4[] picks = BlobPicker.Pick(image, 5);
            Assert.Contains(picks, p => p.X == 32 && p.Y == 30 && p.Z > .99f && p.W > 0);
            string directory = Path.Combine(Path.GetTempPath(), "warp-blob-cuda-" + Guid.NewGuid());
            Directory.CreateDirectory(directory);
            try
            {
                string path = Path.Combine(directory, "mic.mrc");
                image.WriteMRC(path, 1f);
                string output = BlobPicker.Match(path, 5);
                Assert.Equal(Path.Combine(directory, "matching", "mic_blob.star"), output);
                Star table = new Star(output);
                Assert.Equal(picks.Length, table.RowCount);
                Assert.Contains("mic.mrc", table.GetColumn("rlnMicrographName"));
            }
            finally { Directory.Delete(directory, true); }
            using Image stack = new Image(new int3(16, 16, 2));
            Assert.Throws<ArgumentException>(() => BlobPicker.Pick(stack, 2));
        }
    }
}
