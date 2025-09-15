using CommandLine;
using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Warp;
using Warp.Sociology;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_peak_align",
          HelpText = "Aligns per-tilt correlation peaks averaged over multiple particles.")]
    [CommandRunner(typeof(PeakAlignTiltseries))]
    class PeakAlignTiltseriesOptions : DistributedOptions
    {
        [OptionGroup("STAR files with particle coordinates")]
        [Option("input_star",
                HelpText = "Single STAR file containing particle poses to be exported")]
        public string InputStarFile { get; set; }

        [Option("input_directory",
                HelpText =
                    "Directory containing multiple STAR files each with particle poses to be exported")]
        public string InputDirectory { get; set; }

        [Option("input_pattern",
                HelpText = "Wildcard pattern to search for from the input directory",
                Default = "*.star")]
        public string InputPattern { get; set; }

        [OptionGroup("Coordinate scaling")]
        [Option("coords_angpix",
                HelpText = "Pixel size for particles coordinates in input star file(s)",
                Default = null)]
        public double? InputPixelSize { get; set; }

        [Option("normalized_coords",
                HelpText =
                    "Are coordinates normalised to the range [0, 1] (e.g. from Warp's template matching)")]
        public bool InputCoordinatesAreNormalized { get; set; }

        [OptionGroup("Correlation options")]
        [Option("corr_angpix", Default = 10.0, HelpText = "Pixel size at which to calculate the correlation")]
        public double CorrAngPix { get; set; }

        [OptionGroup("Template options")]
        [Option("template_path", Required = true, HelpText = "Path to the template file")]
        public string TemplatePath { get; set; }
        public string FlippedTemplatePath { get; set; } = null;

        [Option("template_angpix", HelpText = "Pixel size of the template; leave empty to use value from map header")]
        public double? TemplateAngPix { get; set; }

        [Option("template_diameter", Required = true, HelpText = "Template diameter in Angstrom")]
        public int TemplateDiameter { get; set; }
    }

    class PeakAlignTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            PeakAlignTiltseriesOptions cli = options as PeakAlignTiltseriesOptions;
            cli.Evaluate();

            #region Validate options

            if (cli.InputPixelSize == null &&
                cli.InputCoordinatesAreNormalized == false)
                throw new Exception(
                                    "Invalid combination of arguments, either input pixel size or coordinates are normalized must be set.");
            else if (cli.InputPixelSize != null &&
                     cli.InputCoordinatesAreNormalized == true)
                throw new Exception(
                                    "Invalid combination of arguments, only one of input pixel size and coordinates are normalized can be set.");

            if (cli.InputPixelSize != null && cli.InputPixelSize <= 0)
                throw new Exception("Input pixel size must be a positive number.");

            if (!string.IsNullOrEmpty(cli.TemplatePath) && !File.Exists(cli.TemplatePath))
                throw new Exception("Template file doesn't exist");

            if (cli.TemplateAngPix.HasValue && cli.TemplateAngPix.Value <= 0)
                throw new Exception("--template_angpix can't be 0 or negative");

            if (cli.TemplateDiameter <= 0)
                throw new Exception("--template_diameter can't be 0 or negative");

            string TemplateDir = Path.Combine(cli.OutputProcessing, "template");
            Directory.CreateDirectory(TemplateDir);

            #endregion

            ProcessingOptionsTomoPeakAlign optionsAlign = (ProcessingOptionsTomoPeakAlign)cli.Options.FillTomoProcessingBase(new ProcessingOptionsTomoPeakAlign());
            optionsAlign.BinTimes = (decimal)Math.Log(cli.CorrAngPix / (double)cli.Options.Import.PixelSize, 2.0);
            optionsAlign.Normalize = true;
            optionsAlign.Invert = true;

            #region Parse input

            bool handleSingleFile =
                !string.IsNullOrEmpty(cli.InputStarFile) &&
                string.IsNullOrEmpty(cli.InputDirectory);
            bool handleMultipleFiles =
                !string.IsNullOrEmpty(cli.InputDirectory) &&
                !string.IsNullOrEmpty(cli.InputPattern);

            Star inputStar;

            if (handleSingleFile)
                inputStar = ParseRelionParticleStar(cli.InputStarFile);
            else if (handleMultipleFiles)
            {
                string[] inputStarFiles = Directory.EnumerateFiles(path: cli.InputDirectory, searchPattern: cli.InputPattern)
                                                   .Where(p => !Helper.PathToName(p).StartsWith('.')).ToArray();

                Console.WriteLine($"Found {inputStarFiles.Length} files in {cli.InputDirectory} matching {cli.InputPattern};");
                inputStar = new Star(inputStarFiles.Select(file => new Star(file)).ToArray());
            }
            else
            {
                throw new Exception("Either a single input file or a directory and wildcard pattern must be provided.");
            }

            ValidateInputStar(inputStar);
            string[] tiltSeriesIDs = inputStar.HasColumn("rlnMicrographName") ? inputStar.GetColumn("rlnMicrographName") : inputStar.GetColumn("rlnTomoName");
            Dictionary<string, List<int>> tiltSeriesIdToParticleIndices = GroupParticles(tiltSeriesIDs);

            if (Helper.IsDebug)
                foreach (var kvp in tiltSeriesIdToParticleIndices)
                    Console.WriteLine($"TS: {kvp.Key}   Particles: {kvp.Value.Count}");

            float3[] xyz = GetCoordinates(inputStar);

            // Check for the existence of Euler angle columns directly instead of checking for non-zero values.
            bool inputHasEulerAngles = inputStar.HasColumn("rlnAngleRot") &&
                                       inputStar.HasColumn("rlnAngleTilt") &&
                                       inputStar.HasColumn("rlnAnglePsi");

            if (!inputHasEulerAngles)
                throw new Exception("Input STAR file must contain Euler angles (rlnAngleRot, rlnAngleTilt, rlnAnglePsi).");

            // GetEulerAngles is already designed to return zeroed vectors if columns are missing.
            float3[] rotTiltPsi = GetEulerAngles(inputStar); // degrees

            if (Helper.IsDebug)
                Console.WriteLine($"input has euler angles?: {inputHasEulerAngles}");

            Console.WriteLine($"Found {xyz.Count()} particles in {tiltSeriesIdToParticleIndices.Count()} tilt series");

            #endregion

            #region Prepare template

            Image TemplateOri = Image.FromFile(cli.TemplatePath);
            if (cli.TemplateAngPix == null)
            {
                if (TemplateOri.PixelSize <= 0)
                    throw new Exception("Couldn't determine pixel size from template, please specify --template_angpix");

                cli.TemplateAngPix = TemplateOri.PixelSize;
                optionsAlign.TemplatePixel = (decimal)TemplateOri.PixelSize;
                Console.WriteLine($"Setting --template_angpix to {TemplateOri.PixelSize} based on template map");
            }

            #endregion

            #region Do processing

            WorkerWrapper[] Workers = cli.GetWorkers();

            IterateOverItems<TiltSeries>(Workers, cli, (worker, t) =>
            {
                // Validate presence of CTF info and particles for this TS, early exit if not found
                if (t.OptionsCTF == null)
                {
                    Console.WriteLine($"No CTF metadata found for {t.Name}, skipping...");
                    return;
                }

                if (!tiltSeriesIdToParticleIndices.ContainsKey(t.Name))
                {
                    Console.WriteLine($"no particles found in {t.Name}, skipping...");
                    return;
                }

                // Get positions and orientations for this tilt-series, rescale to Angstroms
                List<int> tsParticleIdx = tiltSeriesIdToParticleIndices[t.Name];
                float3[] tsPositionAngst = new float3[tsParticleIdx.Count];
                float3[] tsAngles = new float3[tsParticleIdx.Count];

                if (Helper.IsDebug)
                    Console.WriteLine($"{tsParticleIdx.Count} particles for {t.Name}");

                for (int i = 0; i < tsParticleIdx.Count; i++)
                {
                    // get positions
                    if (cli.InputCoordinatesAreNormalized)
                    {
                        xyz[tsParticleIdx[i]].X *= (float)(cli.Options.Tomo.DimensionsX - 1);
                        xyz[tsParticleIdx[i]].Y *= (float)(cli.Options.Tomo.DimensionsY - 1);
                        xyz[tsParticleIdx[i]].Z *= (float)(cli.Options.Tomo.DimensionsZ - 1);
                        tsPositionAngst[i] = xyz[tsParticleIdx[i]] * (float)cli.Options.Import.PixelSize;
                    }
                    else
                        tsPositionAngst[i] = xyz[tsParticleIdx[i]] * (float)cli.InputPixelSize;

                    // get euler angles
                    tsAngles[i] = rotTiltPsi[tsParticleIdx[i]];
                }

                // Replicate positions and angles NTilts times because the
                // WarpWorker method is parametrised for particle trajectories
                float3[] tsPositionsRepl = tsPositionAngst.SelectMany(p => Helper.ArrayOfConstant(p, t.NTilts)).ToArray();
                float3[] tsAnglesRepl = tsAngles.SelectMany(p => Helper.ArrayOfConstant(p, t.NTilts)).ToArray();

                worker.TomoPeakAlign(t.Path, optionsAlign, cli.TemplatePath, tsPositionsRepl, tsAnglesRepl);

                worker.GcCollect();
            });

            #endregion

            #region Dispose of workers

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            #endregion
        }

        private Star ParseRelionParticleStar(string path)
        {
            // Modified from Star.LoadRelion3Particles
            // that method renames columns so prefer to simply parse data here

            // does it look like a 3.0+ style file? if not, just grab the table
            if (!Star.IsMultiTable(path) || !Star.ContainsTable(path, "optics"))
            {
                if (Helper.IsDebug)
                    Console.WriteLine(
                                      $"file is not a multitable or does not contain an optics table, parsing as single table...");
                return new Star(path);
            }


            // okay, join the optics data with the particle data
            Star opticsTable = new Star(path, "optics");
            int[] groupIDs = opticsTable.GetColumn("rlnOpticsGroup")
                                        .Select(s => int.Parse(s)).ToArray();
            string[][] opticsGroupData = new string[groupIDs.Max() + 1][];
            string[] opticsColumnNames = opticsTable.GetColumnNames()
                                                    .Where(n => n != "rlnOpticsGroup" && n != "rlnOpticsGroupName")
                                                    .ToArray();

            // first parse data out of the optics table
            for (int r = 0; r < opticsTable.RowCount; r++)
            {
                List<string> opticsRowData = new List<string>();
                foreach (var columnName in opticsColumnNames)
                    opticsRowData.Add(opticsTable.GetRowValue(r, columnName));
                opticsGroupData[groupIDs[r]] = opticsRowData.ToArray();
            }

            // load the particle table and remove and columns present in optics group data
            Star particlesTable = new Star(path, "particles");
            foreach (var columnName in opticsColumnNames)
                if (particlesTable.HasColumn(columnName))
                    particlesTable.RemoveColumn(columnName);

            // construct a new column with the data from the correct optics group
            // and add it to the particle table
            int[] columnOpticsGroupId = particlesTable.GetColumn("rlnOpticsGroup")
                                                      .Select(s => int.Parse(s)).ToArray();

            for (int iField = 0; iField < opticsColumnNames.Length; iField++)
            {
                string[] NewColumn = new string[particlesTable.RowCount];

                for (int r = 0; r < columnOpticsGroupId.Length; r++)
                {
                    int GroupID = columnOpticsGroupId[r];
                    NewColumn[r] = opticsGroupData[GroupID][iField];
                }

                particlesTable.AddColumn(opticsColumnNames[iField], NewColumn);
            }

            return particlesTable;
        }

        private void ValidateInputStar(Star star)
        {
            if (Helper.IsDebug)
            {
                string[] columnNames = star.GetColumnNames();
                Console.WriteLine("columns in table...");
                foreach (var colName in columnNames)
                    Console.WriteLine($"{colName}");
            }

            string[] requiredColumns = new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnCoordinateZ",
            };

            foreach (var column in requiredColumns)
            {
                if (!star.HasColumn(column))
                {
                    throw new Exception($"Couldn't find {column} column in input STAR file.");
                }
            }

            string[] tsIDColumns = new string[]
            {
                "rlnMicrographName",
                "rlnTomoName",
            };
            if (!tsIDColumns.Any(columnName => star.HasColumn(columnName)))
            {
                throw new Exception($"Input STAR must have one of rlnMicrographName or rlnTomoName to identify tilt series.");
            }
        }

        /// <summary>
        /// Groups particles by tilt series ID.
        /// </summary>
        /// <returns>
        /// Returns a dictionary where each key is a tilt series ID and the value is a list of indices 
        /// corresponding to particles belonging to that tilt series
        /// .</returns>
        private static Dictionary<string, List<int>> GroupParticles(string[] tiltSeriesIDs)
        {
            var groups = new Dictionary<string, List<int>>();

            for (int idx = 0; idx < tiltSeriesIDs.Length; idx++)
            {
                string tiltSeriesID = tiltSeriesIDs[idx];
                if (!groups.ContainsKey(tiltSeriesID))
                    groups.Add(tiltSeriesID, new List<int>());
                groups[tiltSeriesID].Add(idx);
            }

            return groups;
        }

        private float3[] GetCoordinates(Star InputStar)
        {
            // parse positions
            float[] posX = InputStar.GetFloat("rlnCoordinateX");
            float[] posY = InputStar.GetFloat("rlnCoordinateY");
            float[] posZ = InputStar.GetFloat("rlnCoordinateZ");

            // parse shifts
            bool inputHasPixelSize = InputStar.HasColumn("rlnPixelSize") ||
                                     InputStar.HasColumn("rlnImagePixelSize");
            if (Helper.IsDebug)
                Console.WriteLine($"input has pixel size?: {inputHasPixelSize}");

            float[] pixelSizes = inputHasPixelSize ? ParsePixelSizes(InputStar) : null;
            if (Helper.IsDebug)
            {
                if (pixelSizes == null)
                    Console.WriteLine($"pixel sizes: {pixelSizes}");
                else
                    Console.WriteLine(
                                      $"pixel sizes: [{pixelSizes[0]}, {pixelSizes[1]}, ...]");
            }


            float[] shiftsX = ParseShifts(
                                          InputStar,
                                          shiftColumn: "rlnOriginX",
                                          angstromShiftColumn: "rlnOriginXAngst",
                                          pixelSizes: pixelSizes
                                         );
            float[] shiftsY = ParseShifts(
                                          InputStar,
                                          shiftColumn: "rlnOriginY",
                                          angstromShiftColumn: "rlnOriginYAngst",
                                          pixelSizes: pixelSizes
                                         );
            float[] shiftsZ = ParseShifts(
                                          InputStar,
                                          shiftColumn: "rlnOriginZ",
                                          angstromShiftColumn: "rlnOriginZAngst",
                                          pixelSizes: pixelSizes
                                         );

            // combine extraction positions and shifts into absolute positions
            float3[] XYZ = new float3[InputStar.RowCount];
            for (int r = 0; r < InputStar.RowCount; r++)
            {
                XYZ[r] = new float3(
                                    x: posX[r] - shiftsX[r],
                                    y: posY[r] - shiftsY[r],
                                    z: posZ[r] - shiftsZ[r]
                                   );
            }

            return XYZ;
        }

        private float[] ParsePixelSizes(Star star)
        {
            float[] pixelSizes = star.HasColumn("rlnPixelSize") ? star.GetFloat("rlnPixelSize") : star.GetFloat("rlnImagePixelSize");

            return pixelSizes;
        }

        private float[] ParseShifts(Star star,
                                    string shiftColumn,
                                    string angstromShiftColumn,
                                    float[] pixelSizes)
        {
            if (star.HasColumn(shiftColumn))
            {
                if (Helper.IsDebug)
                    Console.WriteLine($"got shifts from {shiftColumn}");
                return star.GetFloat(shiftColumn);
            }

            if (star.HasColumn(angstromShiftColumn))
            {
                if (pixelSizes == null)
                    throw new Exception(
                                        "shifts in angstroms found without pixel sizes...");
                if (Helper.IsDebug)
                    Console.WriteLine($"got shifts from {angstromShiftColumn}");
                return star.GetFloat(angstromShiftColumn)
                           .Zip(pixelSizes, (shift, pixelSize) => shift / pixelSize)
                           .ToArray();
            }

            if (Helper.IsDebug)
                Console.WriteLine($"no shifts found in table");
            return new float[star.RowCount]; // all zeros
        }


        private float3[] GetEulerAngles(Star star)
        {
            string[] relionEulerColumns =
                ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"];

            // set to all zeros if any column not found
            foreach (var colName in relionEulerColumns)
            {
                if (!star.HasColumn(colName))
                {
                    if (Helper.IsDebug)
                        Console.WriteLine($"no euler angles found for {star.RowCount}");
                    return new float3[star.RowCount];
                }
            }

            // otherwise get from table
            float[] rot = star.GetFloat("rlnAngleRot");
            float[] tilt = star.GetFloat("rlnAngleTilt");
            float[] psi = star.GetFloat("rlnAnglePsi");
            float3[] rotTiltPsi = new float3[star.RowCount];
            for (int r = 0; r < star.RowCount; r++)
                rotTiltPsi[r] = new float3(x: rot[r], y: tilt[r], z: psi[r]);
            return rotTiltPsi;
        }
    }
}
