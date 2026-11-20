using Warp;
using Xunit;

namespace Tests;

public class StarRemoveRowsTests
{
    [Fact]
    public void RemovesTheGivenRowsAndKeepsTheOrderOfTheRest()
    {
        var table = new Star(new[] { "rlnImageName", "rlnClassNumber" });
        for (int r = 0; r < 10; r++)
            table.AddRow(new[] { $"{r + 1}@stack.mrcs", (r % 3).ToString() });

        // Unsorted and repeated indices, as callers collect them.
        table.RemoveRows(new[] { 7, 0, 3, 7, 9 });

        Assert.Equal(new[] { "2@stack.mrcs", "3@stack.mrcs", "5@stack.mrcs", "6@stack.mrcs", "7@stack.mrcs", "9@stack.mrcs" },
                     table.GetColumn("rlnImageName"));

        table.RemoveRows(System.Array.Empty<int>());
        Assert.Equal(6, table.RowCount);
    }
}
