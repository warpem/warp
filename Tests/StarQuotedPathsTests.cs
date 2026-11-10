using System;
using System.IO;
using Warp;
using Xunit;

namespace Tests;

public sealed class StarQuotedPathsTests
{
    [Theory]
    [InlineData("/data/normal.mrc")]
    [InlineData("/data/microscope 1/movie 01.mrc")]
    [InlineData("/data/O'Brien's movies/movie.mrc")]
    [InlineData("/data/' microscope/\"camera\"/movie.mrc")]
    [InlineData("#movie.mrc")]
    [InlineData("")]
    public void RoundTripPreservesQuotedPathsAndColumnSubset(string path)
    {
        string file = Path.Combine(Path.GetTempPath(), "warp-star-quotes-" + Guid.NewGuid() + ".star");
        try
        {
            var star = new Star(new[] { "rlnImageName", "rlnCoordinateX" });
            star.AddRow(new[] { path, "12.5" });
            star.Save(file);
            Assert.Equal(path, new Star(file).GetColumn("rlnImageName")[0]);
            Assert.Equal(path, new Star(file, onlyColumns: new[] { "rlnImageName" }).GetColumn(0)[0]);
            Assert.Equal("12.5", new Star(file, onlyColumns: new[] { "rlnCoordinateX" }).GetColumn(0)[0]);
        }
        finally { File.Delete(file); }
    }
}
