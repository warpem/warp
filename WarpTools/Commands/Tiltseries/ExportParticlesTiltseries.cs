using CommandLine;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Accord;
using Warp;
using Warp.Sociology;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_export_particles", HelpText = "Export particles as 3D volumes or 2D image series.")]
    [CommandRunner(typeof(ExportParticlesTiltseries))]
    class ExportParticlesTiltseriesOptions : DistributedOptions
    {
        [OptionGroup("STAR files with particle coordinates")]
        [Option("input_star", HelpText = "Single STAR file containing particle poses to be exported")]
        public string InputStarFile { get; set; }

        [Option("input_directory",
            HelpText = "Directory containing multiple STAR files each with particle poses to be exported")]
        public string InputDirectory { get; set; }

        [Option("input_pattern", HelpText = "Wildcard pattern to search for from the input directory",
            Default = "*.star")]
        public string InputPattern { get; set; }

        [OptionGroup("Coordinate scaling")]
        [Option("coords_angpix", HelpText = "Pixel size for particles coordinates in input star file(s)", Default = null)]
        public double? InputPixelSize { get; set; }

        [Option("normalized_coords", HelpText = "Are coordinates normalised to the range [0, 1] (e.g. from Warp's template matching)")]
        public bool InputCoordinatesAreNormalized { get; set; }

        [OptionGroup("Output")]
        [Option("output_star", HelpText = "STAR file for exported particles", Required = true)]
        public string OutputStarFile { get; set; }

        [Option("output_angpix", HelpText = "Pixel size at which to export particles", Required = true)]
        public float OutputPixelSize { get; set; }

        [Option("box", HelpText = "Output has this many pixels/voxels on each side", Required = true)]
        public int OutputBoxSize { get; set; }

        [Option("diameter", HelpText = "Particle diameter in angstroms", Required = true)]
        public int ParticleDiameter { get; set; }

        [Option("relative_output_paths", HelpText = "Make paths in output STAR file relative to the location of the STAR file. They will be relative to the working directory otherwise.")]
        public bool OutputPathsRelativeToStarFile { get; set; }

        [OptionGroup("Export type (REQUIRED, mutually exclusive)")]
        [Option("2d", HelpText = "Output particles as 2d image series centered on the particle (particle series)")]
        public bool Output2DParticles { get; set; }

        [Option("3d", HelpText = "Output particles as 3d images (subtomograms)")]
        public bool Output3DParticles { get; set; }

        [OptionGroup("Expert options")]
        [Option("dont_normalize_input", HelpText = "Don't normalize the entire field of view in input 2D images after high-pass filtering")]
        public bool DontNormalizeInputImages { get; set; }

        [Option("dont_normalize_3d", HelpText = "Don't normalize output particle volumes (only works with --3d)")]
        public bool DontNormalizeSubtomos { get; set; }

        [Option("n_tilts",
            HelpText = "Number of tilt images to include in the output, images with the lowest overall exposure will be included first",
            Default = null)]
        public int? OutputNTilts { get; set; }
    }

    class ExportParticlesTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ExportParticlesTiltseriesOptions cli = options as ExportParticlesTiltseriesOptions;
            cli.Evaluate();

            #region Validate options

            if (cli.InputPixelSize == null && cli.InputCoordinatesAreNormalized == false)
                throw new Exception(
                    "Invalid combination of arguments, either input pixel size or coordinates are normalized must be set.");
            else if (cli.InputPixelSize != null && cli.InputCoordinatesAreNormalized == true)
                throw new Exception(
                    "Invalid combination of arguments, only one of input pixel size and coordinates are normalized can be set.");

            if (cli.InputPixelSize != null && cli.InputPixelSize <= 0)
                throw new Exception("Input pixel size must be a positive number.");

            if (cli.OutputBoxSize % 2 != 0 || cli.OutputBoxSize < 2)
                throw new Exception("Output box size must be an even, positive number.");

            if (cli.ParticleDiameter < 1)
                throw new Exception("Particle diameter must be a positive number.");

            if (cli.OutputNTilts != null && cli.OutputNTilts < 1)
                throw new Exception("Output number of tilts must be a positive integer.");

            if (new bool[] { cli.Output2DParticles, cli.Output3DParticles }.Count(v => v) != 1)
                throw new Exception("Choose either --2d or --3d as the export type");

            #endregion

            #region Parse input

            bool handleSingleFile =
                !string.IsNullOrEmpty(cli.InputStarFile) && string.IsNullOrEmpty(cli.InputDirectory);
            bool handleMultipleFiles =
                !string.IsNullOrEmpty(cli.InputDirectory) && !string.IsNullOrEmpty(cli.InputPattern);

            Star inputStar;

            if (handleSingleFile)
                inputStar = ParseRelionParticleStar(cli.InputStarFile);
            else if (handleMultipleFiles)
            {
                string[] inputStarFiles = Directory.GetFiles(path: cli.InputDirectory, searchPattern: cli.InputPattern);
                Console.WriteLine(
                    $"Found {inputStarFiles.Length} files in {cli.InputDirectory} matching {cli.InputPattern};");
                inputStar = new Star(inputStarFiles.Select(file => new Star(file)).ToArray());
            }
            else
            {
                throw new Exception(
                    "Either a single input file or a directory and wildcard pattern must be provided."
                );
            }

            ValidateInputStar(inputStar);
            string[] tiltSeriesIDs = inputStar.HasColumn("rlnMicrographName")
                ? inputStar.GetColumn("rlnMicrographName")
                : inputStar.GetColumn("rlnTomoName");
            Dictionary<string, List<int>> tiltSeriesIdToParticleIndices = GroupParticles(tiltSeriesIDs);
            
            if (Helper.IsDebug)
                foreach (var kvp in tiltSeriesIdToParticleIndices)
                    Console.WriteLine($"TS: {kvp.Key}   Particles: {kvp.Value.Count}");
            
            float3[] xyz = GetCoordinates(inputStar);
            float3[] rotTiltPsi = GetEulerAngles(inputStar); // degrees
            bool inputHasEulerAngles = rotTiltPsi.All(v => !v.EqualsZero());
            
            if (Helper.IsDebug)
                Console.WriteLine($"input has euler angles?: {inputHasEulerAngles}");
            Console.WriteLine($"Found {xyz.Count()} particles in {tiltSeriesIdToParticleIndices.Count()} tilt series");

            #endregion
            

            #region Prepare WarpWorker options and output-related variables

            ProcessingOptionsTomoSubReconstruction ExportOptions = PrepareWarpWorkerExportOptions(cli, cli.Options);
            int OutputImageDimensionality = DetermineOutputImageDimensionality(cli);

            var OutputStarTables = new Dictionary<string, Star>();
            string OutputStarPath = Path.GetFullPath(cli.OutputStarFile);
            int currentOpticsGroup = 1;
            var opticsGroupLock = new object();

            #endregion

            #region Process tilt-series and accumulate metadata

            WorkerWrapper[] Workers = cli.GetWorkers(attachDebugger: false);

            IterateOverItems(Workers, cli, (worker, tiltSeries) =>
            {
                TiltSeries TiltSeries = (TiltSeries)tiltSeries;
                if (Helper.IsDebug)
                    Console.WriteLine($"Processing {tiltSeries.Name}");

                // Validate presence of CTF info and particles for this TS, early exit if not found
                if (tiltSeries.OptionsCTF == null)
                {
                    Console.WriteLine($"No CTF metadata found for {TiltSeries.Name}, skipping...");
                    return;
                }

                if (!tiltSeriesIdToParticleIndices.ContainsKey(tiltSeries.Name))
                {
                    Console.WriteLine($"no particles found in {tiltSeries.Name}, skipping...");
                    return;
                }
                
                // Get positions and orientations for this tilt-series, rescale to Angstroms
                List<int> tsParticleIdx = tiltSeriesIdToParticleIndices[tiltSeries.Name];
                float3[] tsParticleXyzAngstroms = new float3[tsParticleIdx.Count];
                float3[] tsParticleRotTiltPsi = new float3[tsParticleIdx.Count];
                
                if (Helper.IsDebug)
                    Console.WriteLine($"{tsParticleIdx.Count} particles for {TiltSeries.Name}");

                for (int i = 0; i < tsParticleIdx.Count; i++)
                {
                    // get positions
                    if (cli.InputCoordinatesAreNormalized)
                    {
                        xyz[tsParticleIdx[i]].X *= (float)(cli.Options.Tomo.DimensionsX - 1);
                        xyz[tsParticleIdx[i]].Y *= (float)(cli.Options.Tomo.DimensionsY - 1);
                        xyz[tsParticleIdx[i]].Z *= (float)(cli.Options.Tomo.DimensionsZ - 1);
                        tsParticleXyzAngstroms[i] = xyz[tsParticleIdx[i]] * (float)cli.Options.Import.PixelSize;
                    }
                    else
                        tsParticleXyzAngstroms[i] = xyz[tsParticleIdx[i]] * (float)cli.InputPixelSize;
                    
                    // get euler angles
                    tsParticleRotTiltPsi[i] = rotTiltPsi[tsParticleIdx[i]];
                }


                // Replicate positions and angles NTilts times because the
                // WarpWorker method is parametrised for particle trajectories
                float3[] tsParticleXYZAngstromsReplicated = Helper.Combine(
                    tsParticleXyzAngstroms.Select(
                        p => Helper.ArrayOfConstant(p, ((TiltSeries)tiltSeries).NTilts)
                    ).ToArray()
                );
                float3[] TSParticleRotTiltPsiReplicated = Helper.Combine(
                    tsParticleRotTiltPsi.Select(
                        p => Helper.ArrayOfConstant(p, ((TiltSeries)tiltSeries).NTilts)
                    ).ToArray());


                Star TiltSeriesTable = null;
                if (OutputImageDimensionality == 3)
                {
                    if (Helper.IsDebug)
                        Console.WriteLine($"Sending export options to worker for {TiltSeries.Name}");
                    worker.TomoExportParticleSubtomos(
                        path: tiltSeries.Path,
                        options: ExportOptions,
                        coordinates: tsParticleXYZAngstromsReplicated,
                        angles: TSParticleRotTiltPsiReplicated
                    );
                    if (Helper.IsDebug)
                        Console.WriteLine($"Constructing output table for {TiltSeries.Name}");
                    TiltSeriesTable = ConstructSubvolumeOutputTable(
                        tiltSeries: TiltSeries,
                        xyz: tsParticleXyzAngstroms,
                        eulerAngles: tsParticleRotTiltPsi,
                        inputHasEulerAngles: inputHasEulerAngles,
                        outputPixelSize: cli.OutputPixelSize,
                        relativeToParticleStarFile: cli.OutputPathsRelativeToStarFile,
                        particleStarFile: cli.OutputStarFile
                    );
                    lock (OutputStarTables)
                        OutputStarTables.Add(tiltSeries.Name, TiltSeriesTable);
                }
                else if (OutputImageDimensionality == 2)
                {
                    lock (OutputStarTables)
                    {
                        lock (opticsGroupLock)
                        {
                            // do export, save particle metadata to a temporary location
                            string TempTiltSeriesParticleStarPath = Path.Combine(
                                TiltSeries.ParticleSeriesDir, TiltSeries.RootName + "_temp.star"
                            );
                            if (Helper.IsDebug)
                                Console.WriteLine($"Sending export options to worker for {TiltSeries.Name}");
                            worker.TomoExportParticleSeries(
                                path: tiltSeries.Path,
                                options: ExportOptions,
                                coordinates: tsParticleXYZAngstromsReplicated,
                                angles: TSParticleRotTiltPsiReplicated,
                                pathTableOut: TempTiltSeriesParticleStarPath,
                                pathsRelativeTo: OutputStarPath
                            );

                            // generate necessary metadata for particles.star
                            if (Helper.IsDebug)
                                Console.WriteLine($"\nConstructing output table for {TiltSeries.Name}");
                            Star ParticleTable = Construct2DParticleTable(
                                tempParticleStarPath: TempTiltSeriesParticleStarPath, opticsGroup: currentOpticsGroup
                            );
                            Star ParticleOpticsTable = Construct2DOpticsTable(
                                tiltSeries: TiltSeries,
                                tiltSeriesPixelSize: (float)cli.Options.Import.PixelSize,
                                downsamplingFactor: (float)ExportOptions.DownsampleFactor,
                                boxSize: ExportOptions.BoxSize,
                                opticsGroup: currentOpticsGroup
                            );

                            // generate necessary metadata for tomograms.star 
                            Star TomogramsGeneralTable = Construct2DTomogramStarGeneralTable(
                                tiltSeries: TiltSeries, exportOptions: ExportOptions, opticsGroup: currentOpticsGroup
                            );
                            Star TomogramsTiltSeriesTable = Construct2DTomogramStarTiltSeriesTable(
                                tiltSeries: TiltSeries, exportOptions: ExportOptions, opticsGroup: currentOpticsGroup
                            );


                            // store per tilt-series metadata in dictionary and update optics group
                            OutputStarTables.Add(tiltSeries.RootName + "_particles", ParticleTable);
                            OutputStarTables.Add(tiltSeries.RootName + "_optics", ParticleOpticsTable);
                            OutputStarTables.Add(tiltSeries.RootName + "_tomograms_global", TomogramsGeneralTable);
                            OutputStarTables.Add(tiltSeries.RootName + "_tomograms_tiltseries",
                                TomogramsTiltSeriesTable);
                            currentOpticsGroup += 1;
                        }
                    }
                }
            });

            #endregion

            #region Write output metadata

            WriteOutputStarFile(
                perTiltSeriesTables: OutputStarTables,
                particleStarPath: cli.OutputStarFile,
                outputDimensionality: OutputImageDimensionality
            );

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
                    Console.WriteLine($"file is not a multitable or does not contain an optics table, parsing as single table...");
                return new Star(path);
            }

            
            // okay, join the optics data with the particle data
            Star opticsTable = new Star(path, "optics");
            int[] groupIDs = opticsTable.GetColumn("rlnOpticsGroup").Select(s => int.Parse(s)).ToArray();
            string[][] opticsGroupData = new string[groupIDs.Max() + 1][];
            string[] opticsColumnNames = opticsTable.GetColumnNames().Where(n => n != "rlnOpticsGroup" && n != "rlnOpticsGroupName").ToArray();

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
            int[] columnOpticsGroupId = particlesTable.GetColumn("rlnOpticsGroup").Select(s => int.Parse(s)).ToArray();

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

        private ProcessingOptionsTomoSubReconstruction PrepareWarpWorkerExportOptions(
            ExportParticlesTiltseriesOptions cli, OptionsWarp options
        )
        {
            options.Tasks.TomoSubReconstructPixel = (decimal)cli.OutputPixelSize;
            options.Tasks.TomoSubReconstructDiameter = cli.ParticleDiameter;
            options.Tasks.TomoSubReconstructBox = cli.OutputBoxSize;
            options.Tasks.Export2DBoxSize = cli.OutputBoxSize;
            options.Tasks.Export2DParticleDiameter = cli.OutputBoxSize;
            options.Tasks.InputNormalize = !cli.DontNormalizeInputImages;

            options.Tasks.OutputNormalize = cli.Output3DParticles && !cli.DontNormalizeSubtomos;
            if (cli.OutputNTilts != null)
            {
                options.Tasks.TomoSubReconstructDoLimitDose = true;
                options.Tasks.TomoSubReconstructNTilts = (int)cli.OutputNTilts;
            }

            ProcessingOptionsTomoSubReconstruction ExportOptions = options.GetProcessingTomoSubReconstruction();
            return ExportOptions;
        }


        private int DetermineOutputImageDimensionality(ExportParticlesTiltseriesOptions cli)
        {
            if ((cli.Output2DParticles && cli.Output3DParticles) || (!cli.Output2DParticles && !cli.Output3DParticles))
                throw new Exception("one of --2d or --3d must be set.");
            else if (cli.Output2DParticles)
                return 2;
            else // if (CLI.Output3DParticles)
                return 3;
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
            float[] posX = ParseFloatColumn(InputStar, "rlnCoordinateX");
            float[] posY = ParseFloatColumn(InputStar, "rlnCoordinateY");
            float[] posZ = ParseFloatColumn(InputStar, "rlnCoordinateZ");
            
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
                    Console.WriteLine($"pixel sizes: [{pixelSizes[0]}, {pixelSizes[1]}, ...]");
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
        
        private float[] ParseFloatColumn(Star star, string columnName)
        {
            return star.GetColumn(columnName)
                .Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
        }
        
        private float[] ParsePixelSizes(Star star)
        {
            float[] pixelSizes = star.HasColumn("rlnPixelSize") 
                ? ParseFloatColumn(star, "rlnPixelSize") 
                : ParseFloatColumn(star, "rlnImagePixelSize");

            return pixelSizes;
        }
        
        private float[] ParseShifts(Star star, string shiftColumn, string angstromShiftColumn, float[] pixelSizes)
        {
            if (star.HasColumn(shiftColumn))
            {
                if (Helper.IsDebug)
                    Console.WriteLine($"got shifts from {shiftColumn}");
                return ParseFloatColumn(star, shiftColumn);
            }

            if (star.HasColumn(angstromShiftColumn))
            {
                if (pixelSizes == null)
                    throw new Exception("shifts in angstroms found without pixel sizes...");
                if (Helper.IsDebug)
                    Console.WriteLine($"got shifts from {angstromShiftColumn}");
                return ParseFloatColumn(star, angstromShiftColumn)
                    .Zip(pixelSizes, (shift, pixelSize) => shift / pixelSize)
                    .ToArray();
            }
            if (Helper.IsDebug)
                Console.WriteLine($"no shifts found in table");
            return new float[star.RowCount];  // all zeros
        }
        


        private float3[] GetEulerAngles(Star star)
        {
            string[] relionEulerColumns = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"];
            
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
            float[] rot = ParseFloatColumn(star, "rlnAngleRot");
            float[] tilt = ParseFloatColumn(star, "rlnAngleTilt");
            float[] psi = ParseFloatColumn(star, "rlnAnglePsi");
            float3[] rotTiltPsi = new float3[star.RowCount];
            for (int r = 0; r < star.RowCount; r++)
                rotTiltPsi[r] = new float3(x: rot[r], y: tilt[r], z: psi[r]);
            return rotTiltPsi;
        }

        private string GetOutputImagePath(
            TiltSeries tiltSeries,
            int particleIndex,
            float pixelSize,
            string suffix,
            bool relativeToParticleStarFile, // default is relative to working directory 
            string? particleStarFilePath
        )
        {
            string path = Path.Combine(
                tiltSeries.SubtomoDir,
                $"{tiltSeries.RootName}{suffix}_{particleIndex:D7}_{pixelSize:F2}A.mrc"
            );

            if (relativeToParticleStarFile)
            {
                path = Path.GetRelativePath(relativeTo: Path.GetDirectoryName(particleStarFilePath), path);
            }
            else // relative to current working directory
            {
                path = Path.GetRelativePath(relativeTo: Directory.GetCurrentDirectory(), path);
            }
            path = path.Replace(oldValue: "\\", newValue: "/");

            return path;
        }

        private string GetOutputCTFPath(
            TiltSeries tiltSeries,
            int particleIndex,
            float pixelSize,
            string suffix,
            bool relativeToParticleStarFile, // default is relative to working directory 
            string? particleStarFilePath
        )
        {
            string path = Path.Combine(
                tiltSeries.SubtomoDir,
                $"{tiltSeries.RootName}{suffix}_{particleIndex:D7}_ctf_{pixelSize:F2}A.mrc"
            );

            if (relativeToParticleStarFile)
            {
                path = Path.GetRelativePath(relativeTo: Path.GetDirectoryName(particleStarFilePath), path);
            }
            else // relative to current working directory
            {
                path = Path.GetRelativePath(relativeTo: Directory.GetCurrentDirectory(), path);
            }
            path = path.Replace(oldValue: "\\", newValue: "/");

            return path;
        }

        private Star ConstructSubvolumeOutputTable(
            TiltSeries tiltSeries,
            float3[] xyz,
            float3[] eulerAngles,
            bool inputHasEulerAngles,
            float outputPixelSize,
            bool relativeToParticleStarFile, // default is relative to working directory 
            string? particleStarFile
        )
        {
            int nParticles = xyz.Length;
            string[] particleCoordinateX = new string[nParticles];
            string[] particleCoordinateY = new string[nParticles];
            string[] particleCoordinateZ = new string[nParticles];
            string[] particleAngleRot = new string[nParticles];
            string[] particleAngleTilt = new string[nParticles];
            string[] particleAnglePsi = new string[nParticles];
            string[] tsIdentifier = new string[nParticles];
            string[] particleMagnification = new string[nParticles];
            string[] particleDetectorPixelSize = new string[nParticles];
            string[] particleResolutionEstimate = new string[nParticles];
            string[] particleImageFilePaths = new string[nParticles];
            string[] particleCtfFilePaths = new string[nParticles];
            string[] particlePixelSize = new string[nParticles];
            string[] particleCtfVoltage = new string[nParticles];
            string[] particleCtfSphericalAberration = new string[nParticles];

            for (int i = 0; i < nParticles; i++)
            {
                particleCoordinateX[i] = FormattableString.Invariant($"{xyz[i].X / outputPixelSize:F3}");
                particleCoordinateY[i] = FormattableString.Invariant($"{xyz[i].Y / outputPixelSize:F3}");
                particleCoordinateZ[i] = FormattableString.Invariant($"{xyz[i].Z / outputPixelSize:F3}");
                particleAngleRot[i] = FormattableString.Invariant($"{eulerAngles[i].X}");
                particleAngleTilt[i] = FormattableString.Invariant($"{eulerAngles[i].Y}");
                particleAnglePsi[i] = FormattableString.Invariant($"{eulerAngles[i].Z}");
                tsIdentifier[i] = tiltSeries.Name;
                particleMagnification[i] = "10000.0";
                particleDetectorPixelSize[i] = FormattableString.Invariant($"{outputPixelSize:F5}");
                particleResolutionEstimate[i] = FormattableString.Invariant($"{tiltSeries.CTFResolutionEstimate}");
                particleImageFilePaths[i] = GetOutputImagePath(
                    tiltSeries, 
                    particleIndex: i, 
                    pixelSize: outputPixelSize, 
                    suffix: "",
                    relativeToParticleStarFile: relativeToParticleStarFile, 
                    particleStarFilePath: particleStarFile
                    );
                particleCtfFilePaths[i] = GetOutputCTFPath(
                    tiltSeries, 
                    particleIndex: i, 
                    pixelSize: outputPixelSize, 
                    suffix: "",
                    relativeToParticleStarFile: relativeToParticleStarFile, 
                    particleStarFilePath: particleStarFile
                    );
                particlePixelSize[i] = FormattableString.Invariant($"{outputPixelSize:F5}");
                particleCtfVoltage[i] = FormattableString.Invariant($"{tiltSeries.CTF.Voltage:F3}");
                particleCtfSphericalAberration[i] = FormattableString.Invariant($"{tiltSeries.CTF.Cs:F3}");
            }

            string[] columnNames = new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnCoordinateZ",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnMicrographName",
                "rlnMagnification",
                "rlnDetectorPixelSize",
                "rlnCtfMaxResolution",
                "rlnImageName",
                "rlnCtfImage",
                "rlnPixelSize",
                "rlnVoltage",
                "rlnSphericalAberration",
            };

            string[][] columns = new string[][]
            {
                particleCoordinateX,
                particleCoordinateY,
                particleCoordinateZ,
                particleAngleRot,
                particleAngleTilt,
                particleAnglePsi,
                tsIdentifier,
                particleMagnification,
                particleDetectorPixelSize,
                particleResolutionEstimate,
                particleImageFilePaths,
                particleCtfFilePaths,
                particlePixelSize,
                particleCtfVoltage,
                particleCtfSphericalAberration,
            };

            Star table = new Star(columns, columnNames);
            if (inputHasEulerAngles == false)
            {
                table.RemoveColumn("rlnAngleRot");
                table.RemoveColumn("rlnAngleTilt");
                table.RemoveColumn("rlnAnglePsi");
            }

            return table;
        }

        private Star Construct2DOpticsTable(
            TiltSeries tiltSeries, float tiltSeriesPixelSize, float downsamplingFactor, int boxSize, int opticsGroup
        )
        {
            string[] columnNames = new string[]
            {
                "rlnOpticsGroup",
                "rlnOpticsGroupName",
                "rlnSphericalAberration",
                "rlnVoltage",
                "rlnTomoTiltSeriesPixelSize",
                "rlnCtfDataAreCtfPremultiplied",
                "rlnImageDimensionality",
                "rlnTomoSubtomogramBinning",
                "rlnImagePixelSize",
                "rlnImageSize",
                "rlnAmplitudeContrast"
            };
            string[][] columns = new string[][]
            {
                new string[] { $"{opticsGroup}" },
                new string[] { $"opticsGroup{opticsGroup}" },
                new string[] { FormattableString.Invariant($"{tiltSeries.CTF.Cs:F3}") },
                new string[] { FormattableString.Invariant($"{tiltSeries.CTF.Voltage:F3}") },
                new string[] { FormattableString.Invariant($"{tiltSeriesPixelSize:F5}") },
                new string[] { "1" },
                new string[] { "2" },
                new string[] { FormattableString.Invariant($"{downsamplingFactor:F5}") },
                new string[] { FormattableString.Invariant($"{tiltSeriesPixelSize * downsamplingFactor:F5}") },
                new string[] { FormattableString.Invariant($"{boxSize}") },
                new string[] { FormattableString.Invariant($"{tiltSeries.CTF.Amplitude:F3}") },
            };
            return new Star(columns, columnNames);
        }

        private Star Construct2DParticleTable(string tempParticleStarPath, int opticsGroup)
        {
            Star ParticleTable = new Star(tempParticleStarPath);
            ParticleTable.ModifyAllValuesInColumn("rlnOpticsGroup", v => $"{opticsGroup}");
            return ParticleTable;
        }

        private Star Construct2DTomogramStarGeneralTable(
            TiltSeries tiltSeries, ProcessingOptionsTomoSubReconstruction exportOptions, int opticsGroup
        )
        {
            Star GeneralTable = new Star(new string[]
            {
                "rlnTomoName",
                "rlnTomoTiltSeriesName",
                "rlnTomoFrameCount",
                "rlnTomoSizeX",
                "rlnTomoSizeY",
                "rlnTomoSizeZ",
                "rlnTomoHand",
                "rlnOpticsGroupName",
                "rlnTomoTiltSeriesPixelSize",
                "rlnVoltage",
                "rlnSphericalAberration",
                "rlnAmplitudeContrast",
                "rlnTomoImportFractionalDose"
            });

            List<int> UsedTilts = exportOptions.DoLimitDose
                ? tiltSeries.IndicesSortedDose.Take(exportOptions.NTilts).ToList()
                : tiltSeries.IndicesSortedDose.ToList();
            float TiltDose = tiltSeries.Dose[UsedTilts[1]] - tiltSeries.Dose[UsedTilts[0]];
            UsedTilts.Sort();

            tiltSeries.VolumeDimensionsPhysical = exportOptions.DimensionsPhysical;

            GeneralTable.AddRow(new string[]
            {
                tiltSeries.RootName + ".tomostar",
                "dummy.mrc", //series.RootName + ".mrc",
                UsedTilts.Count.ToString(),
                exportOptions.Dimensions.X.ToString(CultureInfo.InvariantCulture),
                exportOptions.Dimensions.Y.ToString(CultureInfo.InvariantCulture),
                exportOptions.Dimensions.Z.ToString(CultureInfo.InvariantCulture),
                "-1.0",
                $"opticsGroup{opticsGroup}",
                exportOptions.PixelSize.ToString("F5", CultureInfo.InvariantCulture),
                tiltSeries.CTF.Voltage.ToString("F3", CultureInfo.InvariantCulture),
                tiltSeries.CTF.Cs.ToString("F3", CultureInfo.InvariantCulture),
                tiltSeries.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture),
                TiltDose.ToString("F3", CultureInfo.InvariantCulture)
            });
            return GeneralTable;
        }

        private Star Construct2DTomogramStarTiltSeriesTable(
            TiltSeries tiltSeries, ProcessingOptionsTomoSubReconstruction exportOptions, int opticsGroup
        )
        {
            Star TiltSeriesTable = new Star(new string[]
            {
                "rlnTomoProjX",
                "rlnTomoProjY",
                "rlnTomoProjZ",
                "rlnTomoProjW",
                "rlnDefocusU",
                "rlnDefocusV",
                "rlnDefocusAngle",
                "rlnCtfScalefactor",
                "rlnMicrographPreExposure"
            });
            List<int> UsedTilts = exportOptions.DoLimitDose
                ? tiltSeries.IndicesSortedDose.Take(exportOptions.NTilts).ToList()
                : tiltSeries.IndicesSortedDose.ToList();
            UsedTilts.Sort();
            float3[] TiltAngles = tiltSeries.GetAngleInAllTilts(tiltSeries.VolumeDimensionsPhysical * 0.5f);
            foreach (var i in UsedTilts)
            {
                Matrix3 M = Matrix3.Euler(TiltAngles[i]);
                float3 ImageCoords = tiltSeries.GetPositionsInOneTilt(
                    coords: new[] { tiltSeries.VolumeDimensionsPhysical * 0.5f }, tiltID: i
                    ).First();
                CTF TiltCTF = tiltSeries.GetCTFParamsForOneTilt(
                    pixelSize: (float)exportOptions.PixelSize,
                    defoci: new[] { ImageCoords.Z },
                    coords: new[] { ImageCoords },
                    tiltID: i,
                    weighted: true
                    ).First();

                TiltSeriesTable.AddRow(new string[]
                {
                    $"[{M.M11},{M.M12},{M.M13},0]",
                    $"[{M.M21},{M.M22},{M.M23},0]",
                    $"[{M.M31},{M.M32},{M.M33},0]",
                    "[0,0,0,1]",
                    ((TiltCTF.Defocus + TiltCTF.DefocusDelta / 2) * 1e4M).ToString("F1", CultureInfo.InvariantCulture),
                    ((TiltCTF.Defocus - TiltCTF.DefocusDelta / 2) * 1e4M).ToString("F1", CultureInfo.InvariantCulture),
                    TiltCTF.DefocusAngle.ToString("F3", CultureInfo.InvariantCulture),
                    TiltCTF.Scale.ToString("F3", CultureInfo.InvariantCulture),
                    tiltSeries.Dose[i].ToString("F3", CultureInfo.InvariantCulture)
                });
            }

            return TiltSeriesTable;
        }

        private void WriteDummyTiltSeries(string path)
        {
            int3 dims = new int3(2, 2, 2);
            Image dummyImage = new Image(dims, isft: false, iscomplex: false, ishalf: false);
            dummyImage.WriteMRC16b(path);
        }

        private void WriteOutputStarFile(
            Dictionary<string, Star> perTiltSeriesTables, string particleStarPath, int outputDimensionality
        )
        {
            string particleStarDirectory = Path.GetDirectoryName(Path.GetFullPath(particleStarPath));
            Directory.CreateDirectory(particleStarDirectory);

            if (perTiltSeriesTables.Count == 0)
            {
                throw new InvalidOperationException("No particles to write out. Check that rlnMicrographName entries match tomostar file for each tilt series.");
            }

            if (outputDimensionality == 2)
            {
                #region combine info and write out particles.star

                Star table2DMode = new StarParameters(new[] { "rlnTomoSubTomosAre2DStacks" }, new[] { "1" });
                Star tableOpticsCombined = new Star(perTiltSeriesTables.Where(
                        kvp => kvp.Key.EndsWith("_optics")
                    ).ToDictionary(
                        kvp => kvp.Key, kvp => kvp.Value
                    ).Values.ToArray()
                );
                Star tableParticles = new Star(perTiltSeriesTables.Where(
                        kvp => kvp.Key.EndsWith("_particles")
                    ).ToDictionary(
                        kvp => kvp.Key, kvp => kvp.Value
                    ).Values.ToArray()
                );
                Star.SaveMultitable(
                    particleStarPath, new Dictionary<string, Star>()
                    {
                        { "general", table2DMode },
                        { "optics", tableOpticsCombined },
                        { "particles", tableParticles }
                    }
                );

                #endregion

                #region combine info and write out tomograms.star
                
                // construct global table
                Star tomogramsTableGlobalCombined = new Star(perTiltSeriesTables.Where(
                        kvp => kvp.Key.EndsWith("_tomograms_global")
                    ).ToDictionary(
                        kvp => kvp.Key, kvp => kvp.Value
                    ).Values.ToArray()
                );
                string dummyTiltSeriesPath = Helper.PathCombine(particleStarDirectory, "dummy_tiltseries.mrc");
                WriteDummyTiltSeries(path: dummyTiltSeriesPath);
                tomogramsTableGlobalCombined.ModifyAllValuesInColumn(
                    columnName: "rlnTomoTiltSeriesName",
                    f: v => Path.GetRelativePath(relativeTo: particleStarDirectory, dummyTiltSeriesPath)
                );
                
                // get per tilt-series tables
                string tiltSeriesTableSuffix = "_tomograms_tiltseries";
                var tomogramsTiltSeriesTables = perTiltSeriesTables.Where(
                    kvp => kvp.Key.EndsWith(tiltSeriesTableSuffix)
                ).ToDictionary(
                    kvp => kvp.Key, kvp => kvp.Value
                );
                
                // combine all and write out tomograms.star
                Dictionary<string, Star> tomogramStarDict = new Dictionary<string, Star>()
                {
                    { "global", tomogramsTableGlobalCombined },
                };
                foreach (var kvp in tomogramsTiltSeriesTables)
                {
                    string rootName = kvp.Key.Substring(0, kvp.Key.Length - tiltSeriesTableSuffix.Length);
                    tomogramStarDict.Add(rootName + ".tomostar", kvp.Value);
                }

                string tomogramsStarPath = Path.Combine(
                    Helper.PathToFolder(particleStarPath),
                    Helper.PathToName(particleStarPath) + "_tomograms.star"
                );
                Star.SaveMultitable(
                    tomogramsStarPath, tomogramStarDict
                );

                #endregion

                #region write optimisation set

                string particleFile = Helper.PathToNameWithExtension(particleStarPath);
                string tomogramsFile = Helper.PathToNameWithExtension(tomogramsStarPath);
                string optimisationSetPath= Path.Combine(
                    particleStarDirectory,
                    Helper.PathToName(particleStarPath) + "_optimisation_set.star"
                );
                string contents = $@"
data_

_rlnTomoParticlesFile   {particleFile}
_rlnTomoTomogramsFile   {tomogramsFile}
";
                File.WriteAllText(path: optimisationSetPath, contents: contents);

                #endregion
            }
            else if (outputDimensionality == 3)
            {
                Star combinedTable = new Star(perTiltSeriesTables.Values.ToArray());
                combinedTable.Save(particleStarPath);
            }
        }
    }
}