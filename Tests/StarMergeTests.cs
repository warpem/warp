using System;
using System.IO;
using Warp;
using Xunit;

namespace Tests;

/// <summary>
/// The Star[] merge constructor is fed by --input_directory/--input_pattern in several
/// WarpTools commands. When the pattern matches nothing it used to receive an empty array
/// and die on tables[0] with a bare IndexOutOfRangeException, which says nothing about the
/// actual problem (a shell that ate the glob, a wrong directory).
/// </summary>
public class StarMergeTests : IDisposable
{
    private readonly string Folder;

    public StarMergeTests()
    {
        Folder = Path.Combine(Path.GetTempPath(), "warp_star_merge_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Folder);
    }

    public void Dispose()
    {
        try { Directory.Delete(Folder, true); } catch { }
    }

    private string WriteTable(string fileName, params string[] rows)
    {
        string Path0 = Path.Combine(Folder, fileName);
        using (StreamWriter Writer = new StreamWriter(Path0))
        {
            Writer.WriteLine();
            Writer.WriteLine("data_");
            Writer.WriteLine();
            Writer.WriteLine("loop_");
            Writer.WriteLine("_rlnCoordinateX #1");
            Writer.WriteLine("_rlnCoordinateY #2");
            Writer.WriteLine("_rlnCoordinateZ #3");

            foreach (string Row in rows)
                Writer.WriteLine(Row);
        }

        return Path0;
    }

    [Fact]
    public void MergingNoTablesFailsWithAnExplanation()
    {
        var Exception0 = Assert.Throws<ArgumentException>(() => new Star(new Star[0]));
        Assert.Contains("no tables", Exception0.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void MergingNullFailsWithAnExplanation()
    {
        Assert.Throws<ArgumentException>(() => new Star((Star[])null));
    }

    [Fact]
    public void MergingTablesStillConcatenatesRows()
    {
        Star A = new Star(WriteTable("a.star", "0.1 0.2 0.3", "0.4 0.5 0.6"));
        Star B = new Star(WriteTable("b.star", "0.7 0.8 0.9"));

        Star Merged = new Star(new[] { A, B });

        Assert.Equal(3, Merged.ColumnCount);
        Assert.Equal(3, Merged.RowCount);
        Assert.Equal("0.1", Merged.GetRowValue(0, "rlnCoordinateX"));
        Assert.Equal("0.7", Merged.GetRowValue(2, "rlnCoordinateX"));
    }
}
