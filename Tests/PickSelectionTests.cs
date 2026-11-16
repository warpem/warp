using System.Globalization;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public sealed class PickSelectionTests
{
    static Star Picks(params (double Correlation, double Power)[] scores) => new(
        [scores.Select((_, i) => i.ToString(CultureInfo.InvariantCulture)).ToArray(), scores.Select(s => "5").ToArray(),
         scores.Select(s => s.Correlation.ToString("R", CultureInfo.InvariantCulture)).ToArray(), scores.Select(s => s.Power.ToString("R", CultureInfo.InvariantCulture)).ToArray()],
        "rlnCoordinateX", "rlnCoordinateY", PickSelection.CorrelationColumn, PickSelection.PowerColumn);

    [Fact]
    public void KeepsPicksWithinInclusiveBoundsInOrderWithAllColumns()
    {
        var selected = PickSelection.Select(Picks((.2, 1), (.5, 2), (.7, 9), (.5, .5), (.9, 3)), .5, .9, 1, 3);
        Assert.Equal(["1", "4"], selected.GetColumn("rlnCoordinateX"));
        Assert.Equal(4, selected.ColumnCount);
        Assert.Single(PickSelection.Select(Picks((.5, 1e6)), 0, 1, 0, double.MaxValue).GetColumn("rlnCoordinateX"));
    }

    [Fact]
    public void AnEmptySelectionIsWrittenAsAReadableEmptyTable()
    {
        string path = Path.Combine(Path.GetTempPath(), $"pick-selection-{System.Guid.NewGuid():N}.star");
        try
        {
            PickSelection.Write(PickSelection.Select(Picks((.2, 1)), .5, 1, 0, double.MaxValue), path);
            var read = new Star(path);
            Assert.Equal(0, read.RowCount);
            Assert.Empty(read.GetColumn(PickSelection.PowerColumn));
            Assert.False(File.Exists(path + ".writing"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ScoresAreRequiredOnlyWhenThereArePicks()
    {
        var unscored = new Star([["1"], ["2"]], "rlnCoordinateX", "rlnCoordinateY");
        Assert.Throws<InvalidDataException>(() => PickSelection.Select(unscored, 0, 1, 0, double.MaxValue));
        Assert.Equal(0, PickSelection.Select(new Star(["rlnCoordinateX", "rlnCoordinateY"]), 0, 1, 0, double.MaxValue).RowCount);
    }
}
