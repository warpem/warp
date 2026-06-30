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
using Warp.Workers;
using Warp.Workers.Queue;

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

            // Per-worker reconstruction partials accumulate here. Because workers are
            // ephemeral, we don't reduce per-item: instead each worker keeps a resident
            // accumulator (InitReconstructions runs once via the amortized init), adds
            // each tilt series it claims, and atomically rewrites its own partial after
            // every item — so a crash loses at most the current item. A single reduce
            // task then sums all per-worker partials into the final map(s).
            string tempFolder = Path.Combine(cli.Options.Import.ProcessingFolder,
                                             $"temp-{Guid.NewGuid().ToString().Substring(0, 8)}");
            Directory.CreateDirectory(tempFolder);

            var initReconstructions = WorkerCommands.InitReconstructions(
                nreconstructions, cli.BoxSize, cli.Oversample);

            #region Map: back-project each tilt series into its worker's partial

            Console.WriteLine("Extracting particles from tilt series and back-projecting...");

            foreach (var item in cli.InputSeries)
                item.ProcessingStatus = ProcessingStatus.Unprocessed;

            cli.DistributeItems<TiltSeries>(buildTask: (t, i) =>
            {
                if (!tiltSeriesIdToParticleIndices.ContainsKey(t.Name))
                {
                    Console.WriteLine($"no particles found in {t.Name}, skipping...");
                    return null;
                }

                // Get positions and orientations for this tilt-series, rescale to Angstroms
                List<int> tsParticleIdx = tiltSeriesIdToParticleIndices[t.Name];
                float3[] tsPositionAngst = new float3[tsParticleIdx.Count];
                float3[] tsAngles = new float3[tsParticleIdx.Count];

                for (int p = 0; p < tsParticleIdx.Count; p++)
                {
                    // get positions
                    if (cli.InputCoordinatesAreNormalized)
                    {
                        xyz[tsParticleIdx[p]].X *= (float)(cli.Options.Tomo.DimensionsX - 1);
                        xyz[tsParticleIdx[p]].Y *= (float)(cli.Options.Tomo.DimensionsY - 1);
                        xyz[tsParticleIdx[p]].Z *= (float)(cli.Options.Tomo.DimensionsZ - 1);
                        tsPositionAngst[p] = xyz[tsParticleIdx[p]] * (float)cli.Options.Import.PixelSize;
                    }
                    else
                        tsPositionAngst[p] = xyz[tsParticleIdx[p]] * (float)cli.InputPixelSize;

                    // get euler angles
                    tsAngles[p] = rotTiltPsi[tsParticleIdx[p]];
                }

                float3[][] subsetPositions = doingHalves ? new float3[2][] : new float3[1][];
                float3[][] subsetAngles = doingHalves ? new float3[2][] : new float3[1][];

                if (doingHalves)
                {
                    int subsetColumn = inputStar.GetColumnID("rlnRandomSubset");
                    int[] indices1 = tsParticleIdx.Select((id, idx) => (id, idx))
                                                  .Where(pair => inputStar.GetRowValue(pair.id, subsetColumn) == "1")
                                                  .Select(pair => pair.idx)
                                                  .ToArray();
                    int[] indices2 = tsParticleIdx.Select((id, idx) => (id, idx))
                                                  .Where(pair => inputStar.GetRowValue(pair.id, subsetColumn) == "2")
                                                  .Select(pair => pair.idx)
                                                  .ToArray();

                    subsetPositions[0] = indices1.Select(idx => tsPositionAngst[idx]).ToArray();
                    subsetPositions[1] = indices2.Select(idx => tsPositionAngst[idx]).ToArray();
                    subsetAngles[0] = indices1.Select(idx => tsAngles[idx]).ToArray();
                    subsetAngles[1] = indices2.Select(idx => tsAngles[idx]).ToArray();
                }
                else
                {
                    subsetPositions[0] = tsPositionAngst;
                    subsetAngles[0] = tsAngles;
                }

                // Replicate positions and angles NTilts times because the worker
                // method is parametrised for particle trajectories
                for (int s = 0; s < subsetPositions.Length; s++)
                {
                    subsetPositions[s] = subsetPositions[s].SelectMany(p => Helper.ArrayOfConstant(p, t.NTilts)).ToArray();
                    subsetAngles[s] = subsetAngles[s].SelectMany(p => Helper.ArrayOfConstant(p, t.NTilts)).ToArray();
                }

                var task = new TaskItem
                {
                    TaskId = $"{i:D7}-recadd-{t.RootName}",
                    Stage = "preprocess",
                    RequiresGpu = true,
                    Init = new[] { initReconstructions },
                    Main = new[]
                    {
                        WorkerCommands.TomoAddToReconstructionAndSave(t.Path, optionsBackproject, subsetPositions, subsetAngles, tempFolder),
                        WorkerCommands.GcCollect(),
                    },
                };
                task.ComputeInitFingerprint();
                return task;
            });

            #endregion

            #region Reduce: sum per-worker partials into the final map(s)

            string[] symmetries = Enumerable.Repeat(cli.Symmetry, nreconstructions).ToArray();
            string[] outputPaths = nreconstructions > 1 ?
                                     Enumerable.Range(1, nreconstructions)
                                               .Select(half => Path.Combine(cli.Options.Import.ProcessingFolder, $"{cli.OutputName}_half{half}.mrc"))
                                               .ToArray() :
                                     [Path.Combine(cli.Options.Import.ProcessingFolder, $"{cli.OutputName}.mrc")];

            // Each worker that did any work left one partial per reconstruction; gather
            // them per reconstruction index. Workers that died before saving leave none
            // (their items were re-pended and redone elsewhere).
            string[][] partialPaths = new string[nreconstructions][];
            for (int irec = 0; irec < nreconstructions; irec++)
                partialPaths[irec] = Directory.GetFiles(tempFolder, $"partial_*_rec{irec}.mrc");

            int totalPartials = partialPaths.Sum(p => p.Length);
            if (totalPartials == 0)
                throw new Exception("No reconstruction partials were produced. Check that rlnMicrographName/rlnTomoName " +
                                    "entries match the tilt series and that particles were found.");

            Console.WriteLine($"Reducing {totalPartials} per-worker partial(s) into {nreconstructions} reconstruction(s)...");

            var reduceTask = new TaskItem
            {
                TaskId = "reduce-reconstruction",
                Stage = "preprocess",
                RequiresGpu = true,
                Init = Array.Empty<NamedSerializableObject>(),
                Main = new[]
                {
                    WorkerCommands.TomoFinishReconstruction(partialPaths, symmetries, outputPaths,
                        (float)optionsBackproject.BinnedPixelSizeMean, cli.BoxSize, cli.Oversample),
                },
            };
            reduceTask.ComputeInitFingerprint();

            cli.DistributeTasks(new[] { reduceTask });

            Directory.Delete(tempFolder, recursive: true);

            Console.WriteLine($"Wrote {nreconstructions} reconstruction(s)");

            #endregion

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
