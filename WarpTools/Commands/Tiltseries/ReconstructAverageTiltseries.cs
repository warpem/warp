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
    [Verb("ts_reconstruct_average",
          HelpText = "Reconstructs one or multiple volumes using particles from tilt series.")]
    [CommandRunner(typeof(ReconstructAverageTiltseries))]
    class ReconstructAverageTiltseriesOptions : DistributedOptions
    {
        [OptionGroup("STAR files with particle coordinates")]
        [Option("input_star",
                HelpText = "Single STAR file containing particle poses to be exported")]
        public string InputStarFile { get; set; }

        [Option("input_directory",
                HelpText = "Directory containing multiple STAR files each with particle poses to be exported")]
        public string InputDirectory { get; set; }

        [Option("input_pattern",
                HelpText = "Wildcard pattern to search for from the input directory",
                Default = "*.star")]
        public string InputPattern { get; set; }

        [OptionGroup("Output options")]
        [Option("output", Default = "reconstruction",
                HelpText = "Name of the output file without the extension; for half-maps, a '_halfX' suffix will be added")]
        public string OutputName { get; set; }

        [OptionGroup("Coordinate scaling")]
        [Option("coords_angpix",
                HelpText = "Pixel size for particles coordinates in input star file(s)",
                Default = null)]
        public double? InputPixelSize { get; set; }

        [Option("normalized_coords",
                HelpText = "Are coordinates normalised to the range [0, 1] (e.g. from Warp's template matching)")]
        public bool InputCoordinatesAreNormalized { get; set; }

        [OptionGroup("Reconstruction options")]
        [Option("rec_angpix", Required = true, HelpText = "Pixel size at which to extract particles and reconstruct volumes")]
        public double RecAngPix { get; set; }

        [Option("boxsize", Required = true, HelpText = "Reconstruction box size in pixels")]
        public int BoxSize { get; set; }

        [Option("symmetry", Default = "C1", HelpText = "Point-group symmetry to apply to the reconstructions")]
        public string Symmetry { get; set; }

        [Option("ignore_split", HelpText = "Ignore the half-set split specified in the metadata and reconstruct a single volume using all particles")]
        public bool IgnoreSplit { get; set; }

        [Option("force_split", HelpText = "In the absence of a rlnRandomSet column in the metadata, divide the particles randomly to reconstruct 2 half-maps")]
        public bool ForceSplit { get; set; }

        [Option("first_ntilts", HelpText = "Only use data from the first N tilts (sorted by ascending exposure)")]
        public int? FirstNTilts { get; set; }

        [Option("oversample", Default = 2, HelpText = "Oversample by this factor to improve interpolation accuracy at the expense of a higher memory footprint")]
        public int Oversample { get; set; }

        [Option("batchsize", Default = 128, HelpText = "Particles are processed in batches of this size. Reducing it will decrease the memory footprint")]
        public int BatchSize { get; set; }
    }

    class ReconstructAverageTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ReconstructAverageTiltseriesOptions cli = options as ReconstructAverageTiltseriesOptions;
            cli.Evaluate();

            #region Validate options

            if (string.IsNullOrWhiteSpace(cli.OutputName))
                throw new Exception("Output name must be specified.");

            if (cli.InputPixelSize == null &&
                cli.InputCoordinatesAreNormalized == false)
                throw new Exception("Invalid combination of arguments, either input pixel size or coordinates are normalized must be set.");
            else if (cli.InputPixelSize != null &&
                     cli.InputCoordinatesAreNormalized == true)
                throw new Exception("Invalid combination of arguments, only one of input pixel size and coordinates are normalized can be set.");

            if (cli.InputPixelSize != null && cli.InputPixelSize <= 0)
                throw new Exception("Input pixel size must be a positive number.");

            if (cli.RecAngPix <= 0)
                throw new Exception("Reconstruction pixel size must be a positive number.");

            if (cli.BoxSize <= 0 || cli.BoxSize % 2 != 0)
                throw new Exception("Box size must be a positive even integer.");

            if (cli.BatchSize <= 0)
                throw new Exception("Batch size must be a positive integer.");

            Symmetry sym;
            try
            { 
                sym = new Symmetry(cli.Symmetry);
            }
            catch (Exception exc)
            {
                throw new Exception($"Couldn't parse symmetry \"{cli.Symmetry}\": {exc.Message}");
            }

            if (cli.IgnoreSplit && cli.ForceSplit)
                throw new Exception("Cannot both ignore and force half-set split.");

            if (cli.Oversample < 1)
                throw new Exception("Oversample factor must be >= 1.");

            string TemplateDir = Path.Combine(cli.OutputProcessing, "template");
            Directory.CreateDirectory(TemplateDir);

            #endregion

            var optionsBackproject = (ProcessingOptionsTomoAddToReconstruction)cli.Options.FillTomoProcessingBase(new ProcessingOptionsTomoAddToReconstruction());
            optionsBackproject.BinTimes = (decimal)Math.Log(cli.RecAngPix / (double)cli.Options.Import.PixelSize, 2.0);
            optionsBackproject.Normalize = true;
            optionsBackproject.Invert = true;
            optionsBackproject.BoxSize = cli.BoxSize;
            optionsBackproject.LimitFirstNTilts = cli.FirstNTilts ?? 0;
            optionsBackproject.BatchSize = cli.BatchSize;

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

            if (!inputHasEulerAngles)
                throw new Exception("Input STAR file must contain Euler angles (rlnAngleRot, rlnAngleTilt, rlnAnglePsi).");

            if (Helper.IsDebug)
                Console.WriteLine($"input has euler angles?: {inputHasEulerAngles}");

            Console.WriteLine($"Found {xyz.Count()} particles in {tiltSeriesIdToParticleIndices.Count()} tilt series");

            bool doingHalves = (inputStar.HasColumn("rlnRandomSubset") && !cli.IgnoreSplit) || 
                               cli.ForceSplit;

            if (doingHalves)
                if (!inputStar.HasColumn("rlnRandomSubset"))
                {
                    Console.WriteLine("No rlnRandomSubset column found but split requested, randomly splitting particles into two halves");

                    int[] randomSets = new int[inputStar.RowCount];
                    Random rand = new Random(42);
                    for (int i = 0; i < inputStar.RowCount; i++)
                        randomSets[i] = rand.Next(1, 3); // 1 or 2
                    inputStar.AddColumn("rlnRandomSubset", randomSets.Select(i => i.ToString()).ToArray());
                }

            int nreconstructions = doingHalves ? 2 : 1;

            #endregion

            #region Do processing

            WorkerWrapper[] Workers = cli.GetWorkers();

            #region Initialise reconstructions

            Console.Write("Initialising reconstructions...");

            Task.WaitAll(Workers.Select(w => Task.Run(() =>
            {
                w.InitReconstructions(nreconstructions, cli.BoxSize, cli.Oversample);
            })).ToArray());

            Console.WriteLine(" Done");

            #endregion

            #region Perform back-projection from individual tilt series

            Console.WriteLine("Extracting particles from tilt series and back-projecting...");

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

                float3[][] subsetPositions = doingHalves ?
                                               new float3[2][] :
                                               new float3[1][];
                float3[][] subsetAngles = doingHalves ?
                                            new float3[2][] :
                                            new float3[1][];

                if (doingHalves)
                {
                    int subsetColumn = inputStar.GetColumnID("rlnRandomSubset");
                    int[] indices1 = tsParticleIdx.Select((id, i) => (id, i))
                                                  .Where(pair => inputStar.GetRowValue(pair.id, subsetColumn) == "1")
                                                  .Select(pair => pair.i)
                                                  .ToArray();
                    int[] indices2 = tsParticleIdx.Select((id, i) => (id, i))
                                                  .Where(pair => inputStar.GetRowValue(pair.id, subsetColumn) == "2")
                                                  .Select(pair => pair.i)
                                                  .ToArray();

                    subsetPositions[0] = indices1.Select(i => tsPositionAngst[i]).ToArray();
                    subsetPositions[1] = indices2.Select(i => tsPositionAngst[i]).ToArray();
                    subsetAngles[0] = indices1.Select(i => tsAngles[i]).ToArray();
                    subsetAngles[1] = indices2.Select(i => tsAngles[i]).ToArray();
                }
                else
                {
                    subsetPositions[0] = tsPositionAngst;
                    subsetAngles[0] = tsAngles;
                }

                // Replicate positions and angles NTilts times because the
                // WarpWorker method is parametrised for particle trajectories
                for (int i = 0; i < subsetPositions.Length; i++)
                {
                    subsetPositions[i] = subsetPositions[i].SelectMany(p => Helper.ArrayOfConstant(p, t.NTilts)).ToArray();
                    subsetAngles[i] = subsetAngles[i].SelectMany(p => Helper.ArrayOfConstant(p, t.NTilts)).ToArray();
                }

                worker.TomoAddToReconstructions(t.Path, optionsBackproject, subsetPositions, subsetAngles);

                worker.GcCollect();
            });

            #endregion

            #region Save intermediate results

            string tempFolder = null;
            string[][] intermediatePaths = [[]];

            // We only need to do this if we combine intermediate results from multiple workers
            if (Workers.Length > 1)
            {
                tempFolder = Path.Combine(cli.Options.Import.ProcessingFolder,
                                         $"temp-{Guid.NewGuid().ToString().Substring(0, 8)}");
                Directory.CreateDirectory(tempFolder);

                intermediatePaths = new string[nreconstructions][];
                for (int i = 0; i < nreconstructions; i++)
                { 
                    intermediatePaths[i] = new string[Workers.Length - 1];
                    for (int j = 0; j < Workers.Length - 1; j++)
                        intermediatePaths[i][j] = Path.Combine(tempFolder, $"{cli.OutputName}_half{i + 1}_part{j + 1}.mrc");
                }

                Task.WaitAll(Workers.Skip(1)
                                    .Select((worker, i) => Task.Run(() =>
                {
                    string[] workerPaths = intermediatePaths.Select(paths => paths[i]).ToArray();
                    worker.SaveIntermediateReconstructions(workerPaths);
                })).ToArray());
            }

            #endregion

            #region Dispose all except first worker

            Console.Write("Saying goodbye to all but first worker...");
            foreach (var worker in Workers.Skip(1))
                worker.Dispose();
            Console.WriteLine(" Done");

            #endregion

            #region Finish reconstructions on first worker

            string[] symmetries = Enumerable.Repeat(cli.Symmetry, nreconstructions).ToArray();
            string[] outputPaths = nreconstructions > 1 ?
                                     Enumerable.Range(1, nreconstructions)
                                               .Select(half => Path.Combine(cli.Options.Import.ProcessingFolder, $"{cli.OutputName}_half{half}.mrc"))
                                               .ToArray() :
                                     [Path.Combine(cli.Options.Import.ProcessingFolder, $"{cli.OutputName}.mrc")];

            Console.Write("Finalizing reconstructions...");

            Workers.First().FinishReconstructions(intermediatePaths, symmetries, outputPaths, (float)optionsBackproject.BinnedPixelSizeMean);

            if (!string.IsNullOrEmpty(tempFolder))
                Directory.Delete(tempFolder, recursive: true);

            Console.WriteLine(" Done");

            #endregion
            #endregion

            #region Dispose of workers

            Console.Write("Saying goodbye to remaining workers...");
            Workers.First().Dispose();
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
