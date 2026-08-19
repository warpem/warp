using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Warp;
using Warp.Sociology;
using Warp.Tools;

namespace MTools.Commands
{
    /// <summary>
    /// Outcome of importing a particle table. <see cref="Particles"/> is null if the table
    /// could not be used; the reason has already been written to stderr in that case, so
    /// callers only need to bail out.
    /// </summary>
    class ParticleImportResult
    {
        public Particle[] Particles;

        /// <summary>Rows whose micrograph/tomogram matched an item in one of the data sources.</summary>
        public int Matched;

        /// <summary>Rows that matched nothing. Non-zero is usually a naming mismatch.</summary>
        public int Unmatched;

        /// <summary>Pixel size used to scale RELION coordinates, or -1 for M tables.</summary>
        public float AngPixCoords = -1;

        /// <summary>Pixel size used to scale RELION shifts, or -1 for M tables.</summary>
        public float AngPixShifts = -1;

        /// <summary>Temporal samples per particle position after import.</summary>
        public int TemporalResolutionMovement = 1;

        /// <summary>Temporal samples per particle orientation after import.</summary>
        public int TemporalResolutionRotation = 1;

        /// <summary>Where the half-set split came from, for reporting.</summary>
        public string SubsetSource = "wrpRandomSubset";
    }

    /// <summary>Which flavour of particle metadata a STAR file holds.</summary>
    enum ParticleTableFormat
    {
        Unknown,
        M,
        Relion
    }

    /// <summary>
    /// Turns particle metadata — either M's own wrp* table or a RELION _data.star — into
    /// <see cref="Particle"/> objects bound to data-source items by data hash.
    ///
    /// Shared by create_species and reconstruct so both accept exactly the same inputs and
    /// report the same diagnostics. Takes the data sources rather than a Population, since a
    /// reconstruction-only run has sources without necessarily having a population.
    /// </summary>
    static class ParticleStarParsing
    {
        /// <summary>
        /// Import from whichever of the two paths is set. Exactly one must be non-empty;
        /// callers are expected to have validated that already.
        /// </summary>
        public static ParticleImportResult FromStar(string pathM,
                                                    string pathRelion,
                                                    IReadOnlyList<DataSource> sources,
                                                    int temporalSamples,
                                                    float? angPixCoordsOverride = null,
                                                    float? angPixShiftsOverride = null)
        {
            if (!string.IsNullOrEmpty(pathM))
                return FromMStar(pathM, sources, temporalSamples);

            if (!string.IsNullOrEmpty(pathRelion))
                return FromRelionStar(pathRelion, sources, temporalSamples,
                                      angPixCoordsOverride, angPixShiftsOverride);

            throw new Exception("No particle table specified");
        }

        /// <summary>
        /// Import a particle table, working out for itself whether it is M's or RELION's.
        /// The two are unambiguous at the column level, so callers don't have to declare it.
        /// </summary>
        /// <param name="temporalSamples">
        /// Resample every pose trajectory to this many samples, or 0 to keep whatever the
        /// table already has (1 for RELION, possibly more for an M table written by a
        /// previous refinement).
        /// </param>
        /// <param name="angPixCoordsOverride">
        /// Pixel size for RELION coordinates. Also applied to the shifts when those are in
        /// pixels rather than Angstrom — see <see cref="FromRelionStar"/>.
        /// </param>
        public static ParticleImportResult FromAnyStar(string path,
                                                       IReadOnlyList<DataSource> sources,
                                                       int temporalSamples,
                                                       float? angPixCoordsOverride = null)
        {
            switch (DetectFormat(path))
            {
                case ParticleTableFormat.M:
                    return FromMStar(path, sources, temporalSamples);

                case ParticleTableFormat.Relion:
                    return FromRelionStar(path, sources, temporalSamples,
                                          angPixCoordsOverride, null,
                                          coordsOverrideAppliesToShifts: true);

                default:
                    Console.Error.WriteLine($"Couldn't tell what kind of particle table {path} is.\n" +
                                            "Expected either a RELION table with an _rlnCoordinateX column, " +
                                            "or an M table with a _wrpCoordinateX1 column.");
                    return null;
            }
        }

        /// <summary>
        /// Decide whether a STAR file holds M or RELION particle metadata by looking for the
        /// column label each format always has. Reads the labels directly rather than
        /// through <see cref="Star"/> so it doesn't have to guess which of several tables in
        /// a RELION 3.1+ file to open.
        /// </summary>
        public static ParticleTableFormat DetectFormat(string path)
        {
            bool HasM = false, HasRelion = false;

            foreach (string Line in File.ReadLines(path))
            {
                string Trimmed = Line.TrimStart();
                if (Trimmed.Length == 0 || Trimmed[0] != '_')
                    continue;

                // Labels look like "_rlnCoordinateX #3"; compare the label itself so
                // neighbours such as _rlnCoordinateY can't match.
                string Label = Trimmed.Split((char[])null, StringSplitOptions.RemoveEmptyEntries)[0];

                if (Label == "_wrpCoordinateX1")
                    HasM = true;
                else if (Label == "_rlnCoordinateX")
                    HasRelion = true;
            }

            if (HasM == HasRelion)      // neither, or (pathologically) both
                return ParticleTableFormat.Unknown;

            return HasM ? ParticleTableFormat.M : ParticleTableFormat.Relion;
        }

        /// <summary>
        /// Import M's own particle table. Coordinates are already in Angstrom and poses may
        /// carry a temporal trajectory, whose length is detected from the column names.
        /// </summary>
        public static ParticleImportResult FromMStar(string path,
                                                     IReadOnlyList<DataSource> sources,
                                                     int temporalSamples)
        {
            var Result = new ParticleImportResult();

            #region Parse

            Star TableWarp = new Star(path);

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
                return null;
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

            HashSet<string> AvailableHashes = new HashSet<string>(sources.SelectMany(s => s.Files.Keys.ToArray()).ToArray());
            List<string> HashesNotFound = ParticleHashes.Keys.Where(hash => !AvailableHashes.Contains(hash)).ToList();

            Result.Unmatched = HashesNotFound.Sum(h => ParticleHashes[h]);
            Result.Matched = TableWarp.RowCount - Result.Unmatched;

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

            Particle[] ParticlesFinal = new Particle[TableWarp.RowCount];

            for (int p = 0; p < ParticlesFinal.Length; p++)
            {
                float3[] Coordinates = Helper.ArrayOfFunction(i => new float3(ColumnsCoordX[i][p],
                                                                              ColumnsCoordY[i][p],
                                                                              ColumnsCoordZ[i][p]), TableResMov);
                float3[] Angles = Helper.ArrayOfFunction(i => new float3(ColumnsAngleRot[i][p],
                                                                         ColumnsAngleTilt[i][p],
                                                                         ColumnsAnglePsi[i][p]), TableResRot);

                ParticlesFinal[p] = new Particle(Coordinates, Angles, ColumnSubset[p], ColumnSourceName[p], ColumnSourceHash[p]);

                // temporalSamples <= 0 means "keep whatever the table has". An M table
                // written by a previous refinement can carry a real pose trajectory per
                // particle; resampling it to 1 would silently throw that away.
                if (temporalSamples > 0)
                {
                    ParticlesFinal[p].ResampleCoordinates(temporalSamples);
                    ParticlesFinal[p].ResampleAngles(temporalSamples);
                }
            }

            Result.TemporalResolutionMovement = temporalSamples > 0 ? temporalSamples : TableResMov;
            Result.TemporalResolutionRotation = temporalSamples > 0 ? temporalSamples : TableResRot;

            Species.AttachExtraColumns(ParticlesFinal, TableWarp, Species.IsReservedParticleColumn);

            #endregion

            Result.Particles = ParticlesFinal;
            return Result;
        }

        /// <summary>
        /// Import a RELION _data.star. Coordinates and shifts are scaled to Angstrom using
        /// whichever pixel-size convention the file uses, and micrograph/tomogram names are
        /// matched to data-source items to recover each particle's data hash.
        /// </summary>
        /// <param name="coordsOverrideAppliesToShifts">
        /// When set, <paramref name="angPixCoordsOverride"/> also overrides the shift pixel
        /// size, but only for files whose shifts are in pixels. Pre-3.0 RELION stores
        /// rlnOriginX/Y/Z in the same pixel units as the coordinates — the auto-detection
        /// below derives both from rlnDetectorPixelSize/rlnMagnification — so an explicit
        /// coordinate pixel size has to apply to the shifts too or they are left scaled by a
        /// value the caller just said was wrong. 3.0+ files already hold shifts in Angstrom
        /// and are never touched. This lets a caller expose one pixel-size option instead of
        /// two without losing any case; create_species keeps its separate options and leaves
        /// this off.
        /// </param>
        public static ParticleImportResult FromRelionStar(string path,
                                                          IReadOnlyList<DataSource> sources,
                                                          int temporalSamples,
                                                          float? angPixCoordsOverride,
                                                          float? angPixShiftsOverride,
                                                          bool coordsOverrideAppliesToShifts = false)
        {
            var Result = new ParticleImportResult();

            float AngPixCoords = -1;
            float AngPixShifts = -1;

            #region Parse

            bool Is3;
            (Star TableRelion, Is3) = Star.LoadRelion3Particles(path);
            Star TableOptics = null;
            try { TableOptics = new Star(path, "optics"); } catch { }

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
                return null;
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

            if (angPixCoordsOverride != null)
            {
                // Shifts already in Angstrom (RELION 3.0+ / rlnOriginXAngst) are left alone;
                // pixel-based shifts share the coordinates' pixel size, so they track it.
                bool ShiftsAreInPixels = AngPixShifts != 1;

                AngPixCoords = (float)angPixCoordsOverride;

                if (coordsOverrideAppliesToShifts && ShiftsAreInPixels)
                    AngPixShifts = (float)angPixCoordsOverride;
            }
            if (angPixShiftsOverride != null)
                AngPixShifts = (float)angPixShiftsOverride;

            // All hope is lost
            if (AngPixCoords <= 0)
            {
                Console.Error.WriteLine("Couldn't determine pixel size for particle coordinates, please specify it manually as --angpix_coords.");
                return null;
            }
            if (AngPixShifts <= 0)
            {
                Console.Error.WriteLine("Couldn't determine pixel size for particle shifts, please specify it manually as --angpix_shifts.");
                return null;
            }

            Result.AngPixCoords = AngPixCoords;
            Result.AngPixShifts = AngPixShifts;

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
                int Possibilities = sources.Count(source => source.Files.Values.Any(n => n == imageName || Helper.PathToName(n) == imageName));

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
                return null;
            }

            Result.Unmatched = NamesNotFound.Sum(h => ParticleImageNames[h]);
            Result.Matched = TableRelion.RowCount - Result.Unmatched;

            #endregion

            #region Create particles

            Dictionary<string, string> ReverseMapping = new Dictionary<string, string>();
            foreach (var source in sources)
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

            // Half-set assignment, in order of trust. A file exported by WarpTools from an M
            // population carries wrpRandomSubset but no rlnRandomSubset; honouring it keeps
            // the original split, which matters because these poses were refined against a
            // common reference under that split. Reassigning halves arbitrarily would make
            // the two half-maps non-independent and inflate the FSC.
            string SubsetColumn = CleanRelion.HasColumn("rlnRandomSubset") ? "rlnRandomSubset"
                                : CleanRelion.HasColumn("wrpRandomSubset") ? "wrpRandomSubset"
                                : null;

            int[] Subsets = SubsetColumn != null
                ? CleanRelion.GetColumn(SubsetColumn).Select(v => int.Parse(v, CultureInfo.InvariantCulture) - 1).ToArray()
                : Helper.ArrayOfFunction(i => i % 2, NParticles);

            Result.SubsetSource = SubsetColumn ?? "alternating (no half-set column found)";

            string[] MicrographNames = CleanRelion.GetColumn(MicrographColName).ToArray();
            string[] MicrographHashes = MicrographNames.Select(v => ReverseMapping[v]).ToArray();

            Particle[] ParticlesFinal = Helper.ArrayOfFunction(p => new Particle(new[] { Coordinates[p] }, new[] { Angles[p] }, Subsets[p], MicrographNames[p], MicrographHashes[p]), NParticles);

            // A RELION table only ever has one pose per particle, so temporalSamples <= 0
            // ("keep what the table has") simply leaves them at 1.
            if (temporalSamples > 0)
                foreach (var particle in ParticlesFinal)
                {
                    particle.ResampleCoordinates(temporalSamples);
                    particle.ResampleAngles(temporalSamples);
                }

            Result.TemporalResolutionMovement = Math.Max(1, temporalSamples);
            Result.TemporalResolutionRotation = Math.Max(1, temporalSamples);

            // Keep only non-RELION (custom) columns, e.g. aisFilamentID; drop all rln* bookkeeping
            Species.AttachExtraColumns(ParticlesFinal, CleanRelion, n => n.StartsWith("rln", StringComparison.Ordinal));

            #endregion

            Result.Particles = ParticlesFinal;
            return Result;
        }
    }
}
