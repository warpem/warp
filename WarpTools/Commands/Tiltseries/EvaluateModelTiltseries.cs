using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;
using Warp;
using System.Xml.Linq;
using Warp.Headers;
using SkiaSharp;
using System.Globalization;

namespace WarpTools.Commands.Tiltseries
{
    [VerbGroup("Tilt series")]
    [Verb("ts_eval_model", HelpText = "Map 3D positions to sets of 2D image coordinates considering a tilt series' deformation model")]
    [CommandRunner(typeof(EvalModelTiltseries))]
    class EvalModelTiltseriesOptions : BaseOptions
    {
        [Option("input_star", HelpText = "Path to a STAR file containing custom positions specified using tomoCenteredCoordinate(X/Y/Z)Angst labels. Leave empty to use a regular grid of positions instead.")]
        public string InputStar { get; set; }

        [Option("grid_extent", HelpText = "Instead of custom positions in --input_star, calculate an evenly spaced grid with this extent in Angstrom, specified as 'XxYxZ', e.g. 6000x4000x1000.")]
        public string GridExtent { get; set; }

        [Option("grid_dims", HelpText = "When calculating an evenly spaced grid, it will have this many points in each dimension, specified as 'XxYxZ', e.g. 30x20x5. The grid spacing will be grid_extent / (grid_dims - 1).")]
        public string GridDims { get; set; }

        [Option("output", HelpText = "Output location for the per-tilt series STAR files")]
        public string OutputFolder { get; set; }
    }

    class EvalModelTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            EvalModelTiltseriesOptions CLI = options as EvalModelTiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            int3 GridExtent = new int3();
            int3 GridDims = new int3();

            #region Validate options

            if (CLI.SeriesType != SeriesType.Tilt)
                throw new ArgumentException("This tool only works with tilt series");

            if (new[] { CLI.InputStar, CLI.GridExtent, CLI.GridDims}.All(string.IsNullOrEmpty))
                throw new ArgumentException("Must specify either --input_star or --grid_extent/--grid_dims");

            if (!string.IsNullOrEmpty(CLI.InputStar) && (!string.IsNullOrEmpty(CLI.GridExtent) || !string.IsNullOrEmpty(CLI.GridDims)))
                throw new ArgumentException("Cannot specify both --input_star and --grid_extent/--grid_dims at the same time");

            if (string.IsNullOrEmpty(CLI.GridExtent) != string.IsNullOrEmpty(CLI.GridDims))
                throw new ArgumentException("Must specify both --grid_extent and --grid_dims together");

            if (!string.IsNullOrEmpty(CLI.InputStar) && !File.Exists(CLI.InputStar))
                throw new ArgumentException($"Input STAR file '{CLI.InputStar}' does not exist");

            if (string.IsNullOrEmpty(CLI.InputStar))
            {
                try
                {
                    var Extent = CLI.GridExtent.Split('x');
                    if (Extent.Length != 3)
                        throw new Exception($"Expected 3 dimensions, got {Extent.Length}");
                    GridExtent = new int3(int.Parse(Extent[0]), int.Parse(Extent[1]), int.Parse(Extent[2]));
                }
                catch (Exception exc)
                {
                    throw new Exception("Grid extent must be specified as XxYxZ, e.g. 6000x4000x1000\n" +
                                        $"Error: {exc.Message}");
                }

                try
                {
                    var Dims = CLI.GridDims.Split('x');
                    if (Dims.Length != 3)
                        throw new Exception($"Expected 3 dimensions, got {Dims.Length}");
                    GridDims = new int3(int.Parse(Dims[0]), int.Parse(Dims[1]), int.Parse(Dims[2]));
                }
                catch (Exception exc)
                {
                    throw new Exception("Grid dimensions must be specified as XxYxZ, e.g. 30x20x5\n" +
                                        $"Error: {exc.Message}");
                }
            }

            Directory.CreateDirectory(CLI.OutputFolder);

            #endregion

            #region Create processing options

            float3[] InputPositions = null;

            if (!string.IsNullOrEmpty(CLI.InputStar))
            {
                Star TableIn = new Star(CLI.InputStar);

                if (!TableIn.HasColumn("tomoCenteredCoordinateXAngst") || 
                    !TableIn.HasColumn("tomoCenteredCoordinateYAngst") || 
                    !TableIn.HasColumn("tomoCenteredCoordinateZAngst"))
                    throw new ArgumentException("Input STAR file must contain columns tomoCenteredCoordinateXAngst, " +
                                                "tomoCenteredCoordinateYAngst, and tomoCenteredCoordinateZAngst");

                InputPositions = TableIn.GetFloat3("tomoCenteredCoordinateXAngst", 
                                                   "tomoCenteredCoordinateYAngst", 
                                                   "tomoCenteredCoordinateZAngst");
            }
            else
            {
                InputPositions = new float3[GridDims.Elements()];
                float3 Spacing = new float3(GridExtent.X / Math.Max(1, GridDims.X - 1),
                                            GridExtent.Y / Math.Max(1, GridDims.Y - 1),
                                            GridExtent.Z / Math.Max(1, GridDims.Z - 1));
                float3 Center = new float3(GridExtent) * 0.5f;

                int i = 0;
                for (int z = 0; z < GridDims.Z; z++)
                    for (int y = 0; y < GridDims.Y; y++)
                        for (int x = 0; x < GridDims.X; x++)
                            InputPositions[i++] = new float3(x, y, z) * Spacing - Center;
            }

            #endregion

            float3 TomogramDims = new float3((float)Options.Tomo.DimensionsX, 
                                             (float)Options.Tomo.DimensionsY, 
                                             (float)Options.Tomo.DimensionsZ) * (float)Options.Import.PixelSize;
            float3 TomogramCenter = TomogramDims * 0.5f;
            float3[] InputPositionsDecentered = InputPositions.Select(p => p + TomogramCenter).ToArray();

            Star TableOutInputs = new Star(InputPositions,
                                           "tomoCenteredCoordinateXAngst",
                                           "tomoCenteredCoordinateYAngst",
                                           "tomoCenteredCoordinateZAngst");

            Helper.ForCPUGreedy(0, CLI.InputSeries.Length, 8, null, (iseries, threadID) =>
            {
                TiltSeries S = CLI.InputSeries[iseries] as TiltSeries;
                S.VolumeDimensionsPhysical = TomogramDims;
                var ImageHeader = MapHeader.ReadFromFile(new Movie(Path.Combine(S.DataOrProcessingDirectoryName, S.TiltMoviePaths.First())).DataPath);
                S.ImageDimensionsPhysical = new float2(new int2(ImageHeader.Dimensions)) * (float)Options.Import.PixelSize;
                float2 ImageCenter = S.ImageDimensionsPhysical * 0.5f;

                List<float2> AllPositions = new();
                List<float3> AllAngles = new();
                List<float3> AllDefocus = new();

                int NTiltsValid = S.UseTilt.Count(b => b);

                foreach (var pos in InputPositionsDecentered)
                {
                    var Positions = S.GetPositionInAllTilts(pos);
                    var Angles = S.GetAngleInAllTilts(pos);

                    for (int t = 0; t < S.NTilts; t++)
                    {
                        if (!S.UseTilt[t])
                            continue;

                        AllPositions.Add(new float2(Positions[t].X, Positions[t].Y) - ImageCenter);
                        AllAngles.Add(Angles[t] * Helper.ToDeg);

                        CTF CTF = S.GetCTFParamsForOneTilt(1, [Positions[t].Z], [pos], t, false).First();
                        AllDefocus.Add(new float3((float)(CTF.Defocus + CTF.DefocusDelta / 2) * 1e4f,
                                                  (float)(CTF.Defocus - CTF.DefocusDelta / 2) * 1e4f,
                                                  (float)CTF.DefocusAngle));
                    }
                }

                Star TableOutMappings = new Star(["pointID",
                                                  "tiltID",
                                                  "centeredCoordinateXAngst",
                                                  "centeredCoordinateYAngst",
                                                  "angleRot",
                                                  "angleTilt",
                                                  "anglePsi",
                                                  "defocusU",
                                                  "defocusV",
                                                  "defocusAngle"]);

                for (int ipoint = 0; ipoint < InputPositionsDecentered.Length; ipoint++)
                    for (int t = 0; t < NTiltsValid; t++)
                        TableOutMappings.AddRow([(ipoint + 1).ToString(),
                                                (t + 1).ToString(),

                                                AllPositions[ipoint * NTiltsValid + t].X.ToString("F4", CultureInfo.InvariantCulture),
                                                AllPositions[ipoint * NTiltsValid + t].Y.ToString("F4", CultureInfo.InvariantCulture),

                                                AllAngles[ipoint * NTiltsValid + t].X.ToString("F4", CultureInfo.InvariantCulture),
                                                AllAngles[ipoint * NTiltsValid + t].Y.ToString("F4", CultureInfo.InvariantCulture),
                                                AllAngles[ipoint * NTiltsValid + t].Z.ToString("F4", CultureInfo.InvariantCulture),

                                                AllDefocus[ipoint * NTiltsValid + t].X.ToString("F4", CultureInfo.InvariantCulture),
                                                AllDefocus[ipoint * NTiltsValid + t].Y.ToString("F4", CultureInfo.InvariantCulture),
                                                AllDefocus[ipoint * NTiltsValid + t].Z.ToString("F4", CultureInfo.InvariantCulture)]);

                Star TableOutTilts = new Star(["tiltID", "imageName", "voltage", "cs", "phaseShift"]);
                for (int t = 0, tValid = 0; t < S.NTilts; t++)
                {
                    if (!S.UseTilt[t])
                        continue;

                    TableOutTilts.AddRow([(++tValid + 1).ToString(),
                                          S.TiltMoviePaths[t],
                                          S.CTF.Voltage.ToString("F2", CultureInfo.InvariantCulture),
                                          S.CTF.Cs.ToString("F4", CultureInfo.InvariantCulture),
                                          (S.GetTiltPhase(t) * 180).ToString("F2", CultureInfo.InvariantCulture)]);
                }

                Star.SaveMultitable(Path.Combine(CLI.OutputFolder, $"{S.RootName}.star"),
                                    new Dictionary<string, Star>() { { "points", TableOutInputs },
                                                                     { "tilts", TableOutTilts},
                                                                     { "mappings", TableOutMappings } });
            }, null);
        }
    }
}
