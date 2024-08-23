using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;
using Warp;
using System.IO;
using System.Globalization;

namespace WarpTools.Commands
{
    [VerbGroup("General")]
    [Verb("threshold_picks", HelpText = "Apply a score threshold to particles picked through template-matching from tilt or frame series")]
    [CommandRunner(typeof(ThresholdPicks))]
    class ThresholdPicksOptions : BaseOptions
    {
        [Option("in_suffix", Required = true, HelpText = "Suffix for the names of the input STAR files (file names will be assumed to match {item name}_{--in_suffix}.star pattern)")]
        public string InSuffix { get; set; }

        [Option("out_suffix", Required = true, HelpText = "Suffix for the names of the output STAR files (file names will be {item name}_{--in_suffix}_{--outsuffix}.star)")]
        public string OutSuffix { get; set; }

        [Option("out_combined", HelpText = "Path to a single STAR file into which all results will be combined; internal paths will be made relative to this location")]
        public string OutCombined { get; set; }

        [Option("minimum", HelpText = "Remove all particles below this threshold")]
        public double? Minimum { get; set; }

        [Option("maximum", HelpText = "Remove all particles above this threshold")]
        public double? Maximum { get; set; }

        [Option("top_series", HelpText = "Keep this many top-scoring series")]
        public int? NTopSeries { get; set; }

        [Option("top_picks", HelpText = "Keep this many top-scoring particles for each series")]
        public int? NTopPicks { get; set; }
    }

    class ThresholdPicks : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ThresholdPicksOptions CLI = options as ThresholdPicksOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (string.IsNullOrWhiteSpace(CLI.OutCombined) && !CLI.Minimum.HasValue && !CLI.Maximum.HasValue &&
                                                         !CLI.NTopSeries.HasValue && !CLI.NTopPicks.HasValue)
                throw new Exception("Specify at least one of the thresholding or top options, or --out_combined");

            if (CLI.Minimum.HasValue && CLI.Maximum.HasValue && CLI.Minimum.Value >= CLI.Maximum.Value)
                throw new Exception("Minimum must be less than maximum");

            if (string.IsNullOrWhiteSpace(CLI.OutCombined) && string.IsNullOrWhiteSpace(CLI.OutSuffix))
                throw new Exception("Specify at least one of --out_suffix or --out_combined");

            if (CLI.NTopPicks.HasValue && CLI.NTopPicks.Value <= 0)
                throw new Exception("--top_picks must be positive");

            if (CLI.NTopSeries.HasValue && CLI.NTopSeries.Value <= 0)
                throw new Exception("--top_series must be positive");

            #endregion

            var TablesIn = new Dictionary<Movie, Star>();
            int ParticlesIn = 0;
            int ParticlesOut = 0;
            var AverageScores = new Dictionary<Movie, float>();

            int NDone = 0;
            string FullSuffix = "";  // store the full suffix for use when writing output files
            Console.Write($"0/{CLI.InputSeries.Length}");
            foreach (var item in CLI.InputSeries)
            {
                var MatchingFiles = Directory.EnumerateFiles(
                    path: item.MatchingDir, searchPattern: $"{item.RootName}_*{CLI.InSuffix}.star"
                );
                if (MatchingFiles.Count() > 1)
                {
                    Console.WriteLine($"found multiple files matching {item.RootName}_*{CLI.InSuffix}.star");
                    Console.WriteLine($"trying exact match: {item.RootName}_{CLI.InSuffix}.star");
                    MatchingFiles = MatchingFiles.Where(file => Path.GetFileName(file) == $"{item.RootName}_{CLI.InSuffix}.star");
                    if (MatchingFiles.Count() > 1 || MatchingFiles.Count() == 0)
                        throw new Exception($"Please provide a suffix with an exact match for {item.RootName}_{{in_suffix}}.star");
                }
                else if (MatchingFiles.Count() == 0)
                    throw new Exception($"No files found matching {item.RootName}_*{CLI.InSuffix}.star");
                string PathTable = MatchingFiles.First();
                FullSuffix = Path.GetFileNameWithoutExtension(PathTable).Substring(startIndex: item.RootName.Length + 1);

                Star TableIn = new Star(PathTable);

                if (!TableIn.HasColumn("rlnMicrographName"))
                    throw new Exception($"At least one STAR file does not have a rlnMicrographName column: {PathTable}");
                if (!TableIn.HasColumn("rlnAutopickFigureOfMerit"))
                    throw new Exception($"At least one STAR file does not have a rlnAutopickFigureOfMerit column: {PathTable}");

                ParticlesIn += TableIn.RowCount;

                float[] Scores = TableIn.GetColumn("rlnAutopickFigureOfMerit").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                var Rows = Enumerable.Range(0, Scores.Length);

                if (CLI.Minimum.HasValue)
                    Rows = Rows.Where(i => Scores[i] >= CLI.Minimum.Value);
                if (CLI.Maximum.HasValue)
                    Rows = Rows.Where(i => Scores[i] <= CLI.Maximum.Value);

                if (CLI.NTopPicks.HasValue)
                {
                    Rows = Rows.OrderBy(i => -Scores[i]);
                    Rows = Rows.Take(CLI.NTopPicks.Value);
                }

                ParticlesOut += Rows.Count();

                TableIn = TableIn.CreateSubset(Rows);
                TablesIn.Add(item, TableIn);

                AverageScores.Add(item, Rows.Count() > 0 ? Helper.IndexedSubset(Scores, Rows.ToArray()).Average() : 0);
                if (Rows.Count() == 0)
                    Console.WriteLine($"\nWarning: {item.RootName} has no particles left after thresholding");

                VirtualConsole.ClearLastLine();
                Console.Write($"{++NDone}/{CLI.InputSeries.Length} parsed");
            }
            Console.WriteLine("");
            Console.WriteLine($"{ParticlesIn} particles found");
            Console.WriteLine($"{ParticlesOut} particles left after thresholding");

            if (CLI.NTopPicks.HasValue)
            {
                var SelectedSeries = AverageScores.OrderByDescending(v => v.Value).Take(CLI.NTopSeries.Value).Select(v => v.Key);
                TablesIn = TablesIn.Where(v => SelectedSeries.Contains(v.Key)).ToDictionary(v => v.Key, v => v.Value);
            }

            if (!string.IsNullOrWhiteSpace(CLI.OutCombined))
            {
                string OutCombinedAbsolute = Helper.PathCombine(Environment.CurrentDirectory, CLI.OutCombined);

                foreach (var pair in TablesIn)
                {
                    string SeriesPathAbsolute = pair.Key.Path;
                    string SeriesPathRelative = Helper.MakePathRelativeTo(SeriesPathAbsolute, OutCombinedAbsolute);

                    pair.Value.ModifyAllValuesInColumn("rlnMicrographName", v => SeriesPathRelative);
                }

                var TableCombined = new Star(TablesIn.Values.ToArray());
                TableCombined.Save(CLI.OutCombined);
            }
            else
            {
                NDone = 0;
                Console.Write($"0/{TablesIn.Count} saved");
                foreach (var item in TablesIn)
                {
                    var Table = item.Value;
                    Table.Save(Path.Combine(item.Key.MatchingDir, $"{item.Key.RootName}_{FullSuffix}_{CLI.OutSuffix}.star"));

                    VirtualConsole.ClearLastLine();
                    Console.Write($"{++NDone}/{TablesIn.Count} saved");
                }
                Console.WriteLine("");
            }
        }
    }
}