using System;
using System.Globalization;
using System.IO;
using System.Linq;
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
            Assert.Equal(new float2(32, 30), new float2(picks[0].X, picks[0].Y));
            Assert.True(picks[0].Z > .7f && picks[0].W > 0, $"best pick scores {picks[0].Z}");
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

    [Fact]
    public void SuppressionComparesPeaksWithPeaks()
    {
        // A weaker particle 1.3 separations from a strong one survives; one 0.8 separations away does not.
        var peaks = new[] { new float4(100, 100, .3f, 1), new float4(10, 10, .9f, 1), new float4(23, 10, .4f, 1), new float4(18, 10, .5f, 1) };
        float4[] kept = BlobPicker.Suppress(peaks, 10);
        Assert.Equal(new[] { .9f, .4f, .3f }, kept.Select(p => p.Z));
    }

    static float[] Particles(int width, int height, (float X, float Y, float Radius, float Contrast)[] particles, int seed)
    {
        var random = new Random(seed);
        var data = new float[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                float v = (float)(random.NextDouble() - .5) * .5f;
                foreach (var p in particles)
                    if ((x - p.X) * (x - p.X) + (y - p.Y) * (y - p.Y) <= p.Radius * p.Radius) v -= p.Contrast;
                data[y * width + x] = v;
            }
        return data;
    }

    [TemplateMatchCudaFact]
    public void PicksParticlesAtTheEdgeAndNextToStrongerOnes()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            // Radius 10, separation 20. The old picker scored nothing within 2 radii of the border (now: 1), and dropped any
            // peak with a higher pixel within one separation, such as the slope of a stronger neighbour.
            var particles = new (float X, float Y, float Radius, float Contrast)[]
            {
                (11, 60, 7, 2),         // 4 pixels from the left edge
                (60, 60, 7, 3),         // strong
                (86, 60, 7, 1),         // weaker, 1.3 separations from the strong one
                (60, 112, 7, 2),        // near the bottom edge
            };
            using var image = new Image(Particles(128, 124, particles, 1), new int3(128, 124, 1));
            float4[] picks = BlobPicker.Pick(image, 10, 0, 20);
            foreach (var p in particles)
                Assert.Contains(picks, q => (q.X - p.X) * (q.X - p.X) + (q.Y - p.Y) * (q.Y - p.Y) <= 2 * 2);
        }
    }
}
