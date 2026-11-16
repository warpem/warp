using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace Warp.Tools;

/// <summary>
/// Chooses which blob picks get extracted by their scores, and records the chosen rows as the extracted particles'
/// STAR, whose row order is the particle stack's.
/// </summary>
public static class PickSelection
{
    /// <summary>Changes when the selection, or what it records, changes; callers key extraction results on it.</summary>
    public const int Revision = 1;
    public const string CorrelationColumn = "rlnAutopickFigureOfMerit", PowerColumn = "wrpAutopickPower";

    /// <summary>
    /// The picks whose Pearson correlation and local power lie within the bounds (inclusive), with all their columns.
    /// <paramref name="maxPower"/> = <see cref="double.MaxValue"/> leaves power unbounded.
    /// </summary>
    public static Star Select(Star picks, double minCorrelation, double maxCorrelation, double minPower, double maxPower)
    {
        var keep = new List<int>();
        if (picks.RowCount > 0)
        {
            if (!picks.HasColumn(CorrelationColumn) || !picks.HasColumn(PowerColumn))
                throw new InvalidDataException("Selecting picks by score requires correlation and local power columns.");
            string[] correlation = picks.GetColumn(CorrelationColumn), power = picks.GetColumn(PowerColumn);
            for (int i = 0; i < picks.RowCount; i++)
            {
                double c = double.Parse(correlation[i], CultureInfo.InvariantCulture), p = double.Parse(power[i], CultureInfo.InvariantCulture);
                if (c >= minCorrelation && c <= maxCorrelation && p >= minPower && p <= maxPower) keep.Add(i);
            }
        }
        return picks.CreateSubset(keep);
    }

    /// <summary>Writes the table whole or not at all, so a reader never sees a partial list.</summary>
    public static void Write(Star table, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        string temporary = path + ".writing";
        table.Save(temporary, "particles");
        File.Move(temporary, path, true);
    }
}
