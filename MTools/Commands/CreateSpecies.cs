using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Warp;
using System.Globalization;

namespace MTools.Commands
{
    [Verb("create_species", HelpText = "Create a new species")]
    [CommandRunner(typeof(CreateSpecies))]
    class CreateSpeciesOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file to which to add the new data source.")]
        public string Population { get; set; }

        [Option('n', "name", Required = true, HelpText = "Name of the new species.")]
        public string Name { get; set; }

        [Option('d', "diameter", Required = true, HelpText = "Molecule diameter in Angstrom.")]
        public int Diameter { get; set; }

        [Option('s', "sym", Default = "C1", HelpText = "Point symmetry, e.g. C1, D7, O.")]
        public string Symmetry { get; set; }

        [Option("helical_units", Default = 1, HelpText = "Number of helical asymmetric units (only relevant for helical symmetry).")]
        public int HelicalUnits { get; set; }

        [Option("helical_twist", HelpText = "Helical twist in degrees, positive = right-handed (only relevant for helical symmetry).")]
        public double HelicalTwist { get; set; }

        [Option("helical_rise", HelpText = "Helical rise in Angstrom (only relevant for helical symmetry).")]
        public double HelicalRise { get; set; }

        [Option("helical_height", HelpText = "Height of the helical segment along the Z axis in Angstrom (only relevant for helical symmetry).")]
        public int HelicalHeight { get; set; }

        [Option('t', "temporal_samples", Default = 1, HelpText = "Number of temporal samples in each particle pose's trajectory.")]
        public int TemporalSamples { get; set; }

        [Option("half1", Required = true, HelpText = "Path to first half-map file.")]
        public string Half1 { get; set; }

        [Option("half2", Required = true, HelpText = "Path to second half-map file.")]
        public string Half2 { get; set; }

        [Option('m', "mask", Required = true, HelpText = "Path to a tight binary mask file. M will automatically expand and smooth it based on current resolution")]
        public string Mask { get; set; }

        [Option("angpix", HelpText = "Override pixel size value found in half-maps.")]
        public float? AngPix { get; set; }

        [Option("angpix_resample", HelpText = "Resample half-maps and masks to this pixel size.")]
        public float? AngPixResample { get; set; }

        [Option("lowpass", HelpText = "Optional low-pass filter (in Angstrom), applied to both half-maps.")]
        public float? Lowpass { get; set; }

        [Option("particles_relion", HelpText = "Path to _data.star-like particle metadata from RELION.")]
        public string ParticlesRelion { get; set; }

        [Option("particles_m", HelpText = "Path to particle metadata from M.")]
        public string ParticlesM { get; set; }

        [Option("angpix_coords", HelpText = "Override pixel size for RELION particle coordinates.")]
        public float? AngPixRelionPos { get; set; }

        [Option("angpix_shifts", HelpText = "Override pixel size for RELION particle shifts.")]
        public float? AngPixRelionShifts { get; set; }

        [Option("ignore_unmatched", HelpText = "Don't fail if there are particles that don't match any data sources.")]
        public bool IgnoreUnmatched { get; set; }
    }

    class CreateSpecies : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            CreateSpeciesOptions Options = options as CreateSpeciesOptions;

            Population Population = new Population(Options.Population);

            #region Argument validation

            if (string.IsNullOrEmpty(Options.ParticlesRelion) == string.IsNullOrEmpty(Options.ParticlesM))
            {
                Console.Error.WriteLine("Must provide particle file from either RELION or M.");
                return;
            }

            if (Options.AngPix != null && Options.AngPix <= 0)
            {
                Console.Error.WriteLine("--angpix must be positive.");
                return;
            }

            if (Options.AngPixResample != null && Options.AngPixResample <= 0)
            {
                Console.Error.WriteLine("--angpix_resample must be positive.");
                return;
            }

            try
            {
                Symmetry S = new Symmetry(Options.Symmetry);
            }
            catch
            {
                Console.Error.WriteLine($"Unknown point symmetry: {Options.Symmetry}");
                return;
            }

            if (new[] { Options.HelicalUnits > 1,
                        Options.HelicalRise != 0,
                        Options.HelicalTwist != 0,
                        Options.HelicalHeight != 0 }.Any(v => v))
            {
                if (Options.HelicalUnits <= 1)
                {
                    Console.Error.WriteLine($"Helical symmetry requires at least 2 asymmetric units, specified {Options.HelicalUnits}.");
                    return;
                }

                if (Options.HelicalRise <= 0)
                {
                    Console.Error.WriteLine($"Helical rise must be positive, specified {Options.HelicalRise}.");
                    return;
                }

                if (Options.HelicalTwist == 0)
                {
                    Console.Error.WriteLine($"Helical twist must be non-zero, specified {Options.HelicalTwist}.");
                    return;
                }

                if (Options.HelicalHeight <= 0)
                {
                    Console.Error.WriteLine($"Helical height must be positive, specified {Options.HelicalHeight}.");
                    return;
                }

                double MinHeight = Options.HelicalRise * (Options.HelicalUnits - 1);
                if (Options.HelicalHeight < MinHeight)
                {
                    Console.Error.WriteLine($"Helical height must be at least {MinHeight:F2} A (but probably a bit more), specified {Options.HelicalHeight} A.");
                    return;
                }

                Console.WriteLine($"Will use helical symmetry: {Options.HelicalUnits} units, {Options.HelicalRise:F4} A rise, {Options.HelicalTwist:F4} deg twist, {Options.HelicalHeight} A height.");
            }

            if (Options.TemporalSamples < 1)
            {
                Console.Error.WriteLine("--temporal_samples can't be lower than 1.");
                return;
            }

            #endregion

            #region Half-maps and mask

            #region Read maps and and deal with pixel sizes

            Console.Write("Reading maps... ");

            Image Half1 = Image.FromFile(Options.Half1);
            Image Half2 = Image.FromFile(Options.Half2);
            Image Mask = Image.FromFile(Options.Mask);

            if (Half1.PixelSize != Mask.PixelSize)
            {
                Console.Error.WriteLine($"Half-map and mask pixel sizes don't match ({Half1.PixelSize} vs. {Mask.PixelSize}).");
                return;
            }

            if (!Half1.Dims.IsCubic || !Half2.Dims.IsCubic)
            {
                Console.Error.WriteLine($"Half-maps must be cubic.");
                return;
            }

            if (!Mask.Dims.IsCubic)
            {
                Console.Error.WriteLine($"Mask must be cubic.");
                return;
            }

            if (Half1.Dims != Half2.Dims)
            {
                Console.Error.WriteLine($"Half-map dimensions don't match ({Half1.Dims} vs. {Half2.Dims}).");
                return;
            }

            Console.WriteLine("Done");

            if (Options.AngPix == null)
            {
                Options.AngPix = Half1.PixelSize;
                Console.WriteLine($"--angpix not specified, using {Options.AngPix:F4} A/px from half-map.");
            }
            if (Options.AngPixResample == null)
                Options.AngPixResample = Options.AngPix;

            #endregion

            #region Rescale and pad maps if needed

            float AngPix = Options.AngPix.Value;

            if (AngPix != Options.AngPixResample)
            {
                Console.Write($"Resampling maps to {(float)Options.AngPixResample:F4} A/px... ");

                int DimMapResampled = (int)MathF.Round(Half1.Dims.X * (float)Options.AngPix / (float)Options.AngPixResample / 2) * 2;
                Half1 = Half1.AsScaled(new int3(DimMapResampled)).AndDisposeParent();
                Half2 = Half2.AsScaled(new int3(DimMapResampled)).AndDisposeParent();

                int DimMaskResampled = (int)MathF.Round(Mask.Dims.X * (float)Options.AngPix / (float)Options.AngPixResample / 2) * 2;
                Mask = Mask.AsScaled(new int3(DimMaskResampled)).AndDisposeParent().AsPadded(Half1.Dims).AndDisposeParent();
                Mask.Binarize(0.25f);

                AngPix = (float)Options.AngPixResample;

                Console.WriteLine("Done");
            }

            Half1.PixelSize = AngPix;
            Half2.PixelSize = AngPix;
            Mask.PixelSize = AngPix;

            #endregion

            #region Pad maps to 2x diameter

            int DimPadded = (int)MathF.Round(Options.Diameter / AngPix) * 2;
            if (DimPadded != Half1.Dims.X)
            {
                Console.Write("Padding or cropping half-maps to 2x molecule diameter... ");

                Half1 = Half1.AsPadded(new int3(DimPadded)).AndDisposeParent();
                Half2 = Half2.AsPadded(new int3(DimPadded)).AndDisposeParent();

                Console.WriteLine("Done");
            }

            if (DimPadded != Mask.Dims.X)
            {
                Console.Write("Padding or cropping mask to 2x molecule diameter... ");

                Mask = Mask.AsPadded(new int3(DimPadded)).AndDisposeParent();

                Console.WriteLine("Done");
            }

            #endregion

            #region Low-pass, add a little noise to half-maps to avoid instability in FSC later, mask spherically

            Console.Write("Processing half-maps... ");

            if (Options.Lowpass != null)
            {
                if ((float)Options.Lowpass < AngPix * 2)
                {
                    Console.Error.WriteLine($"Low-pass can't be beyond Nyquist ({(AngPix * 2):F4} A)");
                    return;
                }

                Half1.Bandpass(0, (float)Options.Lowpass / AngPix / 2, true, 0.05f);
                Half2.Bandpass(0, (float)Options.Lowpass / AngPix / 2, true, 0.05f);
            }

            RandomNormal RandN = new RandomNormal(123);
            Half1.TransformValues(v => v + RandN.NextSingle(0, 1e-10f));
            Half2.TransformValues(v => v + RandN.NextSingle(0, 1e-10f));

            Half1.MaskSpherically(Half1.Dims.X - 32, 16, true);
            Half2.MaskSpherically(Half2.Dims.X - 32, 16, true);
            Mask.MaskSpherically(Mask.Dims.X - 32, 16, true);

            Console.WriteLine("Done");

            #endregion

            #endregion

            Species NewSpecies = new Species(Half1, Half2, Mask)
            {
                Name = Options.Name,
                PixelSize = (decimal)Options.AngPixResample,
                Symmetry = Options.Symmetry,
                HelicalUnits = Options.HelicalUnits,
                HelicalTwist = (decimal)Options.HelicalTwist,
                HelicalRise = (decimal)Options.HelicalRise,
                HelicalHeight = Options.HelicalHeight,
                DiameterAngstrom = Options.Diameter,
                TemporalResolutionMovement = Options.TemporalSamples,
                TemporalResolutionRotation = Options.TemporalSamples
            };

            NewSpecies.Path = Path.Combine(Population.SpeciesDir,
                                           NewSpecies.NameSafe + "_" + NewSpecies.GUID.ToString().Substring(0, 8),
                                           NewSpecies.NameSafe + ".species");
            if (File.Exists(NewSpecies.Path))
            {
                Console.Error.WriteLine($"{NewSpecies.Path} already exists. Please use a different name, or delete the old species.");
                return;
            }
            Directory.CreateDirectory(NewSpecies.FolderPath);

            #region Particles

            Console.Write("Parsing particle table... ");

            Particle[] ParticlesFinal = null;
            int ParticlesUnmatched = 0;
            int ParticlesMatched = 0;

            float AngPixCoords = -1;
            float AngPixShifts = -1;

            if (!string.IsNullOrEmpty(Options.ParticlesM))
            {
                #region Parse

                Star TableWarp = new Star(Options.ParticlesM);

                if (!TableWarp.HasColumn("wrpCoordinateX1") ||
                    !TableWarp.HasColumn("wrpCoordinateY1") ||
                    !TableWarp.HasColumn("wrpAngleRot1") ||
                    !TableWarp.HasColumn("wrpAngleTilt1") ||
                    !TableWarp.HasColumn("wrpAnglePsi1") ||
                    !TableWarp.HasColumn("wrpSourceHash"))
                {
                    Console.Error.WriteLine("M particle table must contain at least these columns:\n" +
                                            "wrpCoordinateX1\n" +
                                            "wrpCoordinateY1\n" +
                                            "wrpAngleRot1\n" +
                                            "wrpAngleTilt1\n" +
                                            "wrpAnglePsi1\n" +
                                            "wrpSourceHash");
                    return;
                }

                #endregion

                #region Figure out missing sources

                Dictionary<string, int> ParticleHashes = new Dictionary<string, int>();
                foreach (var hash in TableWarp.GetColumn("wrpSourceHash"))
                {
                    if (!ParticleHashes.ContainsKey(hash))
                        ParticleHashes.Add(hash, 0);
                    ParticleHashes[hash]++;
                }

                HashSet<string> AvailableHashes = new HashSet<string>(Helper.Combine(Population.Sources.Select(s => s.Files.Keys.ToArray())));
                List<string> HashesNotFound = ParticleHashes.Keys.Where(hash => !AvailableHashes.Contains(hash)).ToList();

                ParticlesUnmatched = HashesNotFound.Sum(h => ParticleHashes[h]);
                ParticlesMatched = TableWarp.RowCount - ParticlesUnmatched;

                #endregion

                #region Create particles

                int TableResMov = 1, TableResRot = 1;
                string[] PrefixesMov = { "wrpCoordinateX", "wrpCoordinateY", "wrpCoordinateZ" };
                string[] PrefixesRot = { "wrpAngleRot", "wrpAngleTilt", "wrpAnglePsi" };

                while (true)
                {
                    if (PrefixesMov.Any(p => !TableWarp.HasColumn(p + (TableResMov + 1).ToString())))
                        break;
                    TableResMov++;
                }
                while (true)
                {
                    if (PrefixesRot.Any(p => !TableWarp.HasColumn(p + (TableResRot + 1).ToString())))
                        break;
                    TableResRot++;
                }

                string[] NamesCoordX = Helper.ArrayOfFunction(i => $"wrpCoordinateX{i + 1}", TableResMov);
                string[] NamesCoordY = Helper.ArrayOfFunction(i => $"wrpCoordinateY{i + 1}", TableResMov);
                string[] NamesCoordZ = Helper.ArrayOfFunction(i => $"wrpCoordinateZ{i + 1}", TableResMov);

                string[] NamesAngleRot = Helper.ArrayOfFunction(i => $"wrpAngleRot{i + 1}", TableResRot);
                string[] NamesAngleTilt = Helper.ArrayOfFunction(i => $"wrpAngleTilt{i + 1}", TableResRot);
                string[] NamesAnglePsi = Helper.ArrayOfFunction(i => $"wrpAnglePsi{i + 1}", TableResRot);

                float[][] ColumnsCoordX = NamesCoordX.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
                float[][] ColumnsCoordY = NamesCoordY.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
                float[][] ColumnsCoordZ = NamesCoordZ.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();

                float[][] ColumnsAngleRot = NamesAngleRot.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
                float[][] ColumnsAngleTilt = NamesAngleTilt.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
                float[][] ColumnsAnglePsi = NamesAnglePsi.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();

                int[] ColumnSubset = TableWarp.GetColumn("wrpRandomSubset").Select(v => int.Parse(v) - 1).ToArray();

                string[] ColumnSourceName = TableWarp.GetColumn("wrpSourceName");
                string[] ColumnSourceHash = TableWarp.GetColumn("wrpSourceHash");

                ParticlesFinal = new Particle[TableWarp.RowCount];

                for (int p = 0; p < ParticlesFinal.Length; p++)
                {
                    float3[] Coordinates = Helper.ArrayOfFunction(i => new float3(ColumnsCoordX[i][p],
                                                                                  ColumnsCoordY[i][p],
                                                                                  ColumnsCoordZ[i][p]), TableResMov);
                    float3[] Angles = Helper.ArrayOfFunction(i => new float3(ColumnsAngleRot[i][p],
                                                                             ColumnsAngleTilt[i][p],
                                                                             ColumnsAnglePsi[i][p]), TableResRot);

                    ParticlesFinal[p] = new Particle(Coordinates, Angles, ColumnSubset[p], ColumnSourceName[p], ColumnSourceHash[p]);
                    ParticlesFinal[p].ResampleCoordinates(Options.TemporalSamples);
                    ParticlesFinal[p].ResampleAngles(Options.TemporalSamples);
                }

                #endregion
            }
            else if (!string.IsNullOrEmpty(Options.ParticlesRelion))
            {
                #region Parse

                bool Is3;
                (Star TableRelion, Is3) = Star.LoadRelion3Particles(Options.ParticlesRelion);
                Star TableOptics = null;
                try { TableOptics = new Star(Options.ParticlesRelion, "optics"); } catch { }

                string MicrographColName = TableRelion.HasColumn("rlnMicrographName") ? "rlnMicrographName" : "rlnTomoName";

                if (!TableRelion.HasColumn("rlnCoordinateX") ||
                    !TableRelion.HasColumn("rlnCoordinateY") ||
                    !TableRelion.HasColumn("rlnAngleRot") ||
                    !TableRelion.HasColumn("rlnAngleTilt") ||
                    !TableRelion.HasColumn("rlnAnglePsi") ||
                    !TableRelion.HasColumn(MicrographColName))
                {
                    TableRelion = null;
                    Console.Error.WriteLine("RELION particle table must contain at least these columns:\n" +
                                            "rlnCoordinateX\n" +
                                            "rlnCoordinateY\n" +
                                            "rlnAngleRot\n" +
                                            "rlnAngleTilt\n" +
                                            "rlnAnglePsi\n" +
                                            $"{MicrographColName}");
                    return;
                }

                // We care only about file names for matching
                int NameIndex = TableRelion.GetColumnID(MicrographColName);
                for (int r = 0; r < TableRelion.RowCount; r++)
                    TableRelion.SetRowValue(r, NameIndex, Helper.PathToNameWithExtension(TableRelion.GetRowValue(r, NameIndex)));

                #endregion

                #region Many different ways to determine the pixel size

                // 3.0+ doesn't have these columns anymore, but Star.LoadRelion3Particles brings them back
                if (TableRelion.HasColumn("rlnDetectorPixelSize") && TableRelion.HasColumn("rlnMagnification"))
                {
                    try
                    {
                        float DetectorPixel = float.Parse(TableRelion.GetRowValue(0, "rlnDetectorPixelSize")) * 1e4f;
                        float Mag = float.Parse(TableRelion.GetRowValue(0, "rlnMagnification"));

                        AngPixCoords = DetectorPixel / Mag;
                        if (!Is3)
                            AngPixShifts = DetectorPixel / Mag;
                        else
                            AngPixShifts = 1;   // Already in Angstrom in 3.0+
                    }
                    catch { }
                }
                // None of these should be needed with Star.LoadRelion3Particles
                else if (TableRelion.HasColumn("rlnImagePixelSize"))
                {
                    AngPixCoords = float.Parse(TableRelion.GetRowValue(0, "rlnImagePixelSize"));
                }
                else if (TableOptics != null && TableOptics.HasColumn("rlnImagePixelSize"))
                {
                    AngPixCoords = float.Parse(TableOptics.GetRowValue(0, "rlnImagePixelSize"));
                }

                // Just to be extra sure
                if (Is3 || TableRelion.HasColumn("rlnOriginXAngst"))
                    AngPixShifts = 1;

                if (Options.AngPixRelionPos != null)
                    AngPixCoords = (float)Options.AngPixRelionPos;
                if (Options.AngPixRelionShifts != null)
                    AngPixShifts = (float)Options.AngPixRelionShifts;

                // All hope is lost
                if (AngPixCoords <= 0)
                {
                    Console.Error.WriteLine("Couldn't determine pixel size for particle coordinates, please specify it manually as --angpix_coords.");
                    return;
                }
                if (AngPixShifts <= 0)
                {
                    Console.Error.WriteLine("Couldn't determine pixel size for particle shifts, please specify it manually as --angpix_shifts.");
                    return;
                }

                #endregion

                #region Figure out missing and ambiguous sources

                Dictionary<string, int> ParticleImageNames = new Dictionary<string, int>();
                foreach (var imageName in TableRelion.GetColumn(MicrographColName))
                {
                    if (!ParticleImageNames.ContainsKey(imageName))
                        ParticleImageNames.Add(imageName, 0);
                    ParticleImageNames[imageName]++;
                }

                List<string> NamesNotFound = new List<string>();
                List<string> NamesAmbiguous = new List<string>();
                HashSet<string> NamesGood = new HashSet<string>();
                foreach (var imageName in ParticleImageNames.Keys)
                {
                    int Possibilities = Population.Sources.Count(source => source.Files.Values.Any(n => n == imageName || Helper.PathToName(n) == imageName));

                    if (Possibilities == 0)
                        NamesNotFound.Add(imageName);
                    else if (Possibilities > 1)
                        NamesAmbiguous.Add(imageName);
                    else
                        NamesGood.Add(imageName);
                }

                if (NamesAmbiguous.Count > 0)
                {
                    Console.Error.WriteLine($"{NamesAmbiguous.Count} image names are ambiguous between selected data sources.");
                    return;
                }

                ParticlesUnmatched = NamesNotFound.Sum(h => ParticleImageNames[h]);
                ParticlesMatched = TableRelion.RowCount - ParticlesUnmatched;

                #endregion

                #region Create particles

                Dictionary<string, string> ReverseMapping = new Dictionary<string, string>();
                foreach (var source in Population.Sources)
                    foreach (var pair in source.Files)
                        if (NamesGood.Contains(pair.Value))
                            ReverseMapping.Add(pair.Value, pair.Key);
                        else if (NamesGood.Contains(Helper.PathToName(pair.Value)))
                            ReverseMapping.Add(Helper.PathToName(pair.Value), pair.Key);

                List<int> ValidRows = new List<int>(TableRelion.RowCount);
                string[] ColumnMicNames = TableRelion.GetColumn(MicrographColName);
                for (int r = 0; r < ColumnMicNames.Length; r++)
                    if (ReverseMapping.ContainsKey(ColumnMicNames[r]))
                        ValidRows.Add(r);
                Star CleanRelion = TableRelion.CreateSubset(ValidRows);

                int NParticles = CleanRelion.RowCount;
                bool IsTomogram = CleanRelion.HasColumn("rlnCoordinateZ");

                float[] CoordinatesX = CleanRelion.GetColumn("rlnCoordinateX").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixCoords).ToArray();
                float[] CoordinatesY = CleanRelion.GetColumn("rlnCoordinateY").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixCoords).ToArray();
                float[] CoordinatesZ = IsTomogram ? CleanRelion.GetColumn("rlnCoordinateZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixCoords).ToArray() : new float[NParticles];

                float[] OffsetsX = new float[NParticles];
                float[] OffsetsY = new float[NParticles];
                float[] OffsetsZ = new float[NParticles];

                if (CleanRelion.HasColumn("rlnOriginX"))
                {
                    OffsetsX = CleanRelion.HasColumn("rlnOriginX") ? CleanRelion.GetColumn("rlnOriginX").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixShifts).ToArray() : OffsetsX;
                    OffsetsY = CleanRelion.HasColumn("rlnOriginY") ? CleanRelion.GetColumn("rlnOriginY").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixShifts).ToArray() : OffsetsY;
                    OffsetsZ = CleanRelion.HasColumn("rlnOriginZ") ? CleanRelion.GetColumn("rlnOriginZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixShifts).ToArray() : OffsetsZ;
                }
                else if (CleanRelion.HasColumn("rlnOriginXAngst"))
                {
                    OffsetsX = CleanRelion.HasColumn("rlnOriginXAngst") ? CleanRelion.GetColumn("rlnOriginXAngst").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : OffsetsX;
                    OffsetsY = CleanRelion.HasColumn("rlnOriginYAngst") ? CleanRelion.GetColumn("rlnOriginYAngst").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : OffsetsY;
                    OffsetsZ = CleanRelion.HasColumn("rlnOriginZAngst") ? CleanRelion.GetColumn("rlnOriginZAngst").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : OffsetsZ;
                }

                float3[] Coordinates = Helper.ArrayOfFunction(p => new float3(CoordinatesX[p] - OffsetsX[p], CoordinatesY[p] - OffsetsY[p], CoordinatesZ[p] - OffsetsZ[p]), NParticles);

                float[] AnglesRot = CleanRelion.GetColumn("rlnAngleRot").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                float[] AnglesTilt = CleanRelion.GetColumn("rlnAngleTilt").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                float[] AnglesPsi = CleanRelion.GetColumn("rlnAnglePsi").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

                float3[] Angles = Helper.ArrayOfFunction(p => new float3(AnglesRot[p], AnglesTilt[p], AnglesPsi[p]), NParticles);

                int[] Subsets = CleanRelion.HasColumn("rlnRandomSubset") ? CleanRelion.GetColumn("rlnRandomSubset").Select(v => int.Parse(v, CultureInfo.InvariantCulture) - 1).ToArray() : Helper.ArrayOfFunction(i => i % 2, NParticles);

                string[] MicrographNames = CleanRelion.GetColumn(MicrographColName).ToArray();
                string[] MicrographHashes = MicrographNames.Select(v => ReverseMapping[v]).ToArray();

                ParticlesFinal = Helper.ArrayOfFunction(p => new Particle(new[] { Coordinates[p] }, new[] { Angles[p] }, Subsets[p], MicrographNames[p], MicrographHashes[p]), NParticles);
                foreach (var particle in ParticlesFinal)
                {
                    particle.ResampleCoordinates(Options.TemporalSamples);
                    particle.ResampleAngles(Options.TemporalSamples);
                }

                #endregion
            }
            else
                throw new Exception("Shouldn't be here");

            if (!Options.IgnoreUnmatched && ParticlesUnmatched > 0)
            {
                Console.Error.WriteLine($"{ParticlesUnmatched} particles couldn't be matched to data source. Please run again with --ignore_unmatched to proceed anyway.");
                return;
            }

            NewSpecies.AddParticles(ParticlesFinal);

            Console.WriteLine("Done");

            #endregion

            #region Calculate resolution

            Console.WriteLine("Calculating resolution and training denoiser model...");

            NewSpecies.Path = Path.Combine(Population.SpeciesDir, 
                                           NewSpecies.NameSafe + "_" + NewSpecies.GUID.ToString().Substring(0, 8), 
                                           NewSpecies.NameSafe + ".species");
            Directory.CreateDirectory(NewSpecies.FolderPath);

            NewSpecies.CalculateResolutionAndFilter(Options.Lowpass ?? -1, (message) => { VirtualConsole.ClearLastLine(); Console.Write(message); });

            Console.Write("\nCalculating particle statistics... ");

            NewSpecies.CalculateParticleStats();

            Console.WriteLine("Done");
            Console.Write("Committing results... ");

            NewSpecies.Commit();
            NewSpecies.Save();

            Console.WriteLine("Done");

            #endregion

            Population.Species.Add(NewSpecies);
            Population.Save();

            Console.WriteLine($"Species created: '{NewSpecies.Name}' ({NewSpecies.GUID}), {NewSpecies.Path}");
            Console.WriteLine("To check if everything went alright, it's best to run M once with all refinements turned off and see if the new maps resemble your input.");
        }
    }
}
