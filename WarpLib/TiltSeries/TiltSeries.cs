using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Xml;
using System.Xml.XPath;
using Accord;
using Accord.Math.Optimization;
using MathNet.Numerics;
using MathNet.Numerics.Statistics;
using SkiaSharp;
using TorchSharp.NN;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;
using ZLinq;
using IOPath = System.IO.Path;

namespace Warp
{
    public partial class TiltSeries : Movie
    {
        #region Directories

        public static readonly string TiltStackDirName = "tiltstack";
        public string TiltStackDir => IOPath.Combine(ProcessingDirectoryName, TiltStackDirName, RootName);

        public static string ToTiltStackPath (string name) => IOPath.Combine(TiltStackDirName, Helper.PathToName(name) + ".st");
        
        public static readonly string TiltStackThumbnailDirName = "thumbnails";
        public string TiltStackThumbnailDir => IOPath.Combine(TiltStackDir, TiltStackThumbnailDirName);
        public string TiltStackThumbnailPath (string tiltName) => IOPath.Combine(TiltStackThumbnailDir,
                                                                                 Helper.PathToName(tiltName) + ".png");
        public static string ToTiltStackThumbnailPath (string seriesName, string tiltName) => IOPath.Combine(TiltStackDirName, 
                                                                                                             Helper.PathToName(seriesName), 
                                                                                                             TiltStackThumbnailDirName,
                                                                                                             Helper.PathToName(tiltName) + ".png");
        public string TiltStackPath => IOPath.Combine(TiltStackDir, RootName + ".st");

        public static string ToAngleFilePath (string name) => IOPath.Combine(TiltStackDirName, Helper.PathToName(name) + ".rawtlt");
        public string AngleFilePath => IOPath.Combine(TiltStackDir, RootName + ".rawtlt");

        public static string ToTomogramWithPixelSize(string name, decimal pixelSize) => $"{Helper.PathToName(name)}_{pixelSize:F2}Apx";
        
        public static readonly string ReconstructionDirName = "reconstruction";
        public string ReconstructionDir => IOPath.Combine(ProcessingDirectoryName, ReconstructionDirName);
        public static string ToReconstructionTomogramPath(string name, decimal pixelSize) => IOPath.Combine(ReconstructionDirName, ToTomogramWithPixelSize(name, pixelSize) + ".mrc");
        public static string ToReconstructionThumbnailPath(string name, decimal pixelSize) => IOPath.Combine(ReconstructionDirName, ToTomogramWithPixelSize(name, pixelSize) + ".png");

        public static readonly string ReconstructionDeconvDirName = IOPath.Combine(ReconstructionDirName, "deconv");
        public string ReconstructionDeconvDir => IOPath.Combine(ProcessingDirectoryName, ReconstructionDeconvDirName);
        public static string ToReconstructionDeconvPath(string name, decimal pixelSize) => IOPath.Combine(ReconstructionDeconvDirName, ToTomogramWithPixelSize(name, pixelSize) + ".mrc");

        public static readonly string ReconstructionOddDirName = IOPath.Combine(ReconstructionDirName, "odd");
        public string ReconstructionOddDir => IOPath.Combine(ProcessingDirectoryName, ReconstructionOddDirName);
        public static string ToReconstructionOddPath(string name, decimal pixelSize) => IOPath.Combine(ReconstructionOddDirName, ToTomogramWithPixelSize(name, pixelSize) + ".mrc");

        public static readonly string ReconstructionEvenDirName = IOPath.Combine(ReconstructionDirName, "even");
        public string ReconstructionEvenDir => IOPath.Combine(ProcessingDirectoryName, ReconstructionEvenDirName);
        public static string ToReconstructionEvenPath(string name, decimal pixelSize) => IOPath.Combine(ReconstructionEvenDirName, ToTomogramWithPixelSize(name, pixelSize) + ".mrc");

        public static readonly string ReconstructionCTFDirName = IOPath.Combine(ReconstructionDirName, "ctf");
        public string ReconstructionCTFDir => IOPath.Combine(ProcessingDirectoryName, ReconstructionCTFDirName);
        public static string ToReconstructionCTFPath(string name, decimal pixelSize) => IOPath.Combine(ReconstructionCTFDirName, ToTomogramWithPixelSize(name, pixelSize) + ".mrc");

        public static readonly string SubtomoDirName = "subtomo"; 
        public static string ToSubtomoDirPath(string name) => IOPath.Combine(SubtomoDirName, Helper.PathToName(name));
        public string SubtomoDir => IOPath.Combine(ProcessingDirectoryName, SubtomoDirName, RootName);

        public static readonly string ParticleSeriesDirName = "particleseries";
        public static string ToParticleSeriesDirPath(string name) => IOPath.Combine(ParticleSeriesDirName, Helper.PathToName(name));
        public string ParticleSeriesDir => IOPath.Combine(ProcessingDirectoryName, ParticleSeriesDirName, RootName);

        #endregion

        #region Runtime dimensions

        /// <summary>
        /// These must be populated before most operations, otherwise exceptions will be thrown.
        /// Not an elegant solution, but it avoids passing them to a lot of methods.
        /// Given in Angstrom.
        /// </summary>
        public float3 VolumeDimensionsPhysical;

        /// <summary>
        /// Used to account for rounding the size of downsampled raw images to multiples of 2
        /// </summary>
        public float3 SizeRoundingFactors = new float3(1, 1, 1);

        #endregion

        private bool _AreAnglesInverted = false;
        public bool AreAnglesInverted
        {
            get { return _AreAnglesInverted; }
            set { if (value != _AreAnglesInverted) { _AreAnglesInverted = value; OnPropertyChanged(); } }
        }

        public float3 PlaneNormal;

        #region Grids

        private LinearGrid4D _GridVolumeWarpX = new LinearGrid4D(new int4(1, 1, 1, 1));
        public LinearGrid4D GridVolumeWarpX
        {
            get { return _GridVolumeWarpX; }
            set { if (value != _GridVolumeWarpX) { _GridVolumeWarpX = value; OnPropertyChanged(); } }
        }

        private LinearGrid4D _GridVolumeWarpY = new LinearGrid4D(new int4(1, 1, 1, 1));
        public LinearGrid4D GridVolumeWarpY
        {
            get { return _GridVolumeWarpY; }
            set { if (value != _GridVolumeWarpY) { _GridVolumeWarpY = value; OnPropertyChanged(); } }
        }

        private LinearGrid4D _GridVolumeWarpZ = new LinearGrid4D(new int4(1, 1, 1, 1));
        public LinearGrid4D GridVolumeWarpZ
        {
            get { return _GridVolumeWarpZ; }
            set { if (value != _GridVolumeWarpZ) { _GridVolumeWarpZ = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Per-tilt CTF data

        private ObservableCollection<float2[]> _TiltPS1D = new ObservableCollection<float2[]>();
        public ObservableCollection<float2[]> TiltPS1D
        {
            get { return _TiltPS1D; }
            set { if (value != _TiltPS1D) { _TiltPS1D = value; OnPropertyChanged(); } }
        }

        private ObservableCollection<Cubic1D> _TiltSimulatedBackground = new ObservableCollection<Cubic1D>();
        public ObservableCollection<Cubic1D> TiltSimulatedBackground
        {
            get { return _TiltSimulatedBackground; }
            set { if (value != _TiltSimulatedBackground) { _TiltSimulatedBackground = value; OnPropertyChanged(); } }
        }

        private ObservableCollection<Cubic1D> _TiltSimulatedScale = new ObservableCollection<Cubic1D>();
        public ObservableCollection<Cubic1D> TiltSimulatedScale
        {
            get { return _TiltSimulatedScale; }
            set { if (value != _TiltSimulatedScale) { _TiltSimulatedScale = value; OnPropertyChanged(); } }
        }

        public float GetTiltDefocus(int tiltID)
        {
            if (GridCTFDefocus != null && GridCTFDefocus.FlatValues.Length > tiltID)
                return GridCTFDefocus.FlatValues[tiltID];
            return 0;
        }

        public float GetTiltDefocusDelta(int tiltID)
        {
            if (GridCTFDefocusDelta != null && GridCTFDefocusDelta.FlatValues.Length > tiltID)
                return GridCTFDefocusDelta.FlatValues[tiltID];
            return 0;
        }

        public float GetTiltDefocusAngle(int tiltID)
        {
            if (GridCTFDefocusAngle != null && GridCTFDefocusAngle.FlatValues.Length > tiltID)
                return GridCTFDefocusAngle.FlatValues[tiltID];
            return 0;
        }

        public float GetTiltPhase(int tiltID)
        {
            if (GridCTFPhase != null && GridCTFPhase.FlatValues.Length > tiltID)
                return GridCTFPhase.FlatValues[tiltID];
            return 0;
        }

        public CTF GetTiltCTF(int tiltID)
        {
            CTF Result = CTF.GetCopy();
            Result.Defocus = (decimal)GetTiltDefocus(tiltID);
            Result.DefocusDelta = (decimal)GetTiltDefocusDelta(tiltID);
            Result.DefocusAngle = (decimal)GetTiltDefocusAngle(tiltID);
            Result.PhaseShift = (decimal)GetTiltPhase(tiltID);

            return Result;
        }

        public float2[] GetTiltSimulated1D(int tiltID)
        {
            if (TiltPS1D.Count <= tiltID ||
                TiltPS1D[tiltID] == null ||
                TiltSimulatedScale.Count <= tiltID ||
                TiltSimulatedScale[tiltID] == null)
                return null;

            CTF TiltCTF = GetTiltCTF(tiltID);

            float[] SimulatedCTF = TiltCTF.Get1D(TiltPS1D[tiltID].Length, true);

            float2[] Result = new float2[SimulatedCTF.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = new float2(TiltPS1D[tiltID][i].X, SimulatedCTF[i] *
                                                              TiltSimulatedScale[tiltID].Interp(TiltPS1D[tiltID][i].X));

            return Result;
        }

        public event Action TiltCTFProcessed;

        #endregion

        #region Per-tilt parameters

        public float[] Angles = { 0 };
        public float[] Dose = { 0 };
        public bool[] UseTilt = { true };

        public float[] TiltAxisAngles = { 0 };
        public float[] TiltAxisOffsetX = { 0 };
        public float[] TiltAxisOffsetY = { 0 };
        public string[] TiltMoviePaths = { "" };
        
        public float[] FOVFraction = { 1 };

        public int[] IndicesSortedAngle
        {
            get
            {
                if (Angles == null)
                    return null;

                List<int> Sorted = new List<int>(Angles.Length);
                for (int i = 0; i < Angles.Length; i++)
                    Sorted.Add(i);

                Sorted.Sort((a, b) => Angles[a].CompareTo(Angles[b]));

                return Sorted.ToArray();
            }
        }

        public int[] IndicesSortedAbsoluteAngle
        {
            get
            {
                if (Angles == null)
                    return null;

                List<int> Sorted = new List<int>(Helper.ArrayOfSequence(0, Angles.Length, 1));

                Sorted.Sort((a, b) => Math.Abs(Angles[a]).CompareTo(Math.Abs(Angles[b])));

                return Sorted.ToArray();
            }
        }

        private int[] _IndicesSortedDose;
        public int[] IndicesSortedDose
        {
            get
            {
                if (Dose == null)
                    return null;

                if (_IndicesSortedDose == null)
                {
                    List<int> Sorted = new List<int>(Dose.Length);
                    for (int i = 0; i < Dose.Length; i++)
                        Sorted.Add(i);

                    Sorted.Sort((a, b) => Dose[a].CompareTo(Dose[b]));

                    _IndicesSortedDose = Sorted.ToArray();
                }

                return _IndicesSortedDose;
            }
        }

        public int NUniqueTilts
        {
            get
            {
                HashSet<float> UniqueAngles = new HashSet<float>();
                foreach (var angle in Angles)
                    if (!UniqueAngles.Contains(angle))
                        UniqueAngles.Add(angle);

                return UniqueAngles.Count;
            }
        }

        public int NTilts => Angles.Length;

        public float MinTilt => MathHelper.Min(Angles);
        public float MaxTilt => MathHelper.Max(Angles);

        public float MinDose => MathHelper.Min(Dose);
        public float MaxDose => MathHelper.Max(Dose);

        #endregion

        public TiltSeries(string path, string dataDirectoryName = null) : base(path, dataDirectoryName)
        {
            // XML loading is done in base constructor

            if (Angles.Length <= 1)   // In case angles and dose haven't been read and stored in .xml yet.
            {
                InitializeFromTomoStar(new Star(DataPath));
            }
        }

        public void InitializeFromTomoStar(Star table)
        {
            if (!table.HasColumn("wrpDose") || !table.HasColumn("wrpAngleTilt"))
                throw new Exception("STAR file has no wrpDose or wrpAngleTilt column.");

            List<float> TempAngles = new List<float>();
            List<float> TempDose = new List<float>();
            List<float> TempAxisAngles = new List<float>();
            List<float> TempOffsetX = new List<float>();
            List<float> TempOffsetY = new List<float>();
            List<string> TempMoviePaths = new List<string>();

            for (int i = 0; i < table.RowCount; i++)
            {
                TempAngles.Add(float.Parse(table.GetRowValue(i, "wrpAngleTilt"), CultureInfo.InvariantCulture));
                TempDose.Add(float.Parse(table.GetRowValue(i, "wrpDose"), CultureInfo.InvariantCulture));

                if (table.HasColumn("wrpAxisAngle"))
                    TempAxisAngles.Add(float.Parse(table.GetRowValue(i, "wrpAxisAngle"), CultureInfo.InvariantCulture));
                else
                    TempAxisAngles.Add(0);

                if (table.HasColumn("wrpAxisOffsetX") && table.HasColumn("wrpAxisOffsetY"))
                {
                    TempOffsetX.Add(float.Parse(table.GetRowValue(i, "wrpAxisOffsetX"), CultureInfo.InvariantCulture));
                    TempOffsetY.Add(float.Parse(table.GetRowValue(i, "wrpAxisOffsetY"), CultureInfo.InvariantCulture));
                }
                else
                {
                    TempOffsetX.Add(0);
                    TempOffsetY.Add(0);
                }

                if (table.HasColumn("wrpMovieName"))
                    TempMoviePaths.Add(table.GetRowValue(i, "wrpMovieName"));
            }

            if (TempAngles.Count == 0 || TempMoviePaths.Count == 0)
                throw new Exception("Metadata must contain at least 3 values per tilt: movie paths, tilt angles, and accumulated dose.");

            Angles = TempAngles.ToArray();
            Dose = TempDose.ToArray();
            TiltAxisAngles = TempAxisAngles.Count > 0 ? TempAxisAngles.ToArray() : Helper.ArrayOfConstant(0f, NTilts);
            TiltAxisOffsetX = TempOffsetX.Count > 0 ? TempOffsetX.ToArray() : Helper.ArrayOfConstant(0f, NTilts);
            TiltAxisOffsetY = TempOffsetY.Count > 0 ? TempOffsetY.ToArray() : Helper.ArrayOfConstant(0f, NTilts);

            TiltMoviePaths = TempMoviePaths.ToArray();

            UseTilt = Helper.ArrayOfConstant(true, NTilts);
        }

        #region Helper methods

        #region GetPosition methods

        public float3[] GetPositionInAllTilts(float3 coords)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
                PerTiltCoords[i] = coords;

            return GetPositionInAllTilts(PerTiltCoords);
        }

        public float3[] GetPositionInAllTiltsOld(float3[] coords)
        {
            float3[] Result = new float3[coords.Length];

            float3 VolumeCenter = VolumeDimensionsPhysical / 2;
            float2 ImageCenter = ImageDimensionsPhysical / 2;

            float GridStep = 1f / (NTilts - 1);
            float DoseStep = 1f / (MaxDose - MinDose);
            float _MinDose = MinDose;

            float3[] GridCoords = new float3[coords.Length];
            float4[] TemporalGridCoords4 = new float4[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                GridCoords[i] = new float3(coords[i].X / VolumeDimensionsPhysical.X, coords[i].Y / VolumeDimensionsPhysical.Y, t * GridStep);
                TemporalGridCoords4[i] = new float4(GridCoords[i].X, GridCoords[i].Y, coords[i].Z / VolumeDimensionsPhysical.Z, (Dose[t] - _MinDose) * DoseStep);
            }

            float[] GridVolumeWarpXInterp = GridVolumeWarpX.GetInterpolated(TemporalGridCoords4);
            float[] GridVolumeWarpYInterp = GridVolumeWarpY.GetInterpolated(TemporalGridCoords4);
            float[] GridVolumeWarpZInterp = GridVolumeWarpZ.GetInterpolated(TemporalGridCoords4);

            float[] GridDefocusInterp = GridCTFDefocus.GetInterpolatedNative(GridCoords.Take(NTilts).ToArray());

            Matrix3[] TiltMatrices = Helper.ArrayOfFunction(t => Matrix3.Euler(0, Angles[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);
            Matrix3[] TiltMatricesFlipped = null;
            if (AreAnglesInverted)
                TiltMatricesFlipped = Helper.ArrayOfFunction(t => Matrix3.Euler(0, -Angles[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);

            float3[] TransformedCoords = new float3[coords.Length];

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;
                float3 Centered = coords[i] - VolumeCenter;

                Matrix3 Rotation = TiltMatrices[t];

                float3 SampleWarping = new float3(GridVolumeWarpXInterp[i],
                                                  GridVolumeWarpYInterp[i],
                                                  GridVolumeWarpZInterp[i]);
                Centered += SampleWarping;

                float3 Transformed = (Rotation * Centered);

                Transformed.X += TiltAxisOffsetX[t];   // Tilt axis offset is in image space
                Transformed.Y += TiltAxisOffsetY[t];

                Transformed.X += ImageCenter.X;
                Transformed.Y += ImageCenter.Y;

                TransformedCoords[i] = new float3(Transformed.X / ImageDimensionsPhysical.X, Transformed.Y / ImageDimensionsPhysical.Y, t * GridStep);

                Result[i] = Transformed;

                // Do the same, but now with Z coordinate and tilt angle flipped
                if (AreAnglesInverted)
                {
                    Rotation = TiltMatricesFlipped[t];
                    Centered.Z *= -1;

                    Transformed = (Rotation * Centered);

                    Result[i].Z = Transformed.Z;
                }
            }

            float[] GridMovementXInterp = GridMovementX.GetInterpolatedNative(TransformedCoords);
            float[] GridMovementYInterp = GridMovementY.GetInterpolatedNative(TransformedCoords);

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                // Additional stage shift determined for this tilt
                Result[i].X -= GridMovementXInterp[i];
                Result[i].Y -= GridMovementYInterp[i];

                // Coordinates are in Angstrom, can be converted directly in um
                Result[i].Z = GridDefocusInterp[t] + 1e-4f * Result[i].Z;

                Result[i] *= SizeRoundingFactors;
            }

            return Result;
        }

        Spandex<float3> BuffersCoords3 = new Spandex<float3>();
        Spandex<float4> BuffersCoords4 = new Spandex<float4>();
        Spandex<float> BuffersValues = new Spandex<float>();
        public float3[] GetPositionInAllTilts(float3[] coords, float3[] result = null)
        {
            if (result == null)
                result = new float3[coords.Length];

            float3 VolumeCenter = VolumeDimensionsPhysical / 2;
            float2 ImageCenter = ImageDimensionsPhysical / 2;

            float GridStep = 1f / (NTilts - 1);
            float DoseStep = 1f / (MaxDose - MinDose);
            float _MinDose = MinDose;

            Span<float3> GridCoords = BuffersCoords3.Rent(coords.Length);
            Span<float4> TemporalGridCoords4 = BuffersCoords4.Rent(coords.Length);
            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                GridCoords[i] = new float3(coords[i].X / VolumeDimensionsPhysical.X, coords[i].Y / VolumeDimensionsPhysical.Y, t * GridStep);
                TemporalGridCoords4[i] = new float4(GridCoords[i].X, GridCoords[i].Y, coords[i].Z / VolumeDimensionsPhysical.Z, (Dose[t] - _MinDose) * DoseStep);
            }

            Span<float> GridVolumeWarpXInterp = GridVolumeWarpX.GetInterpolated(TemporalGridCoords4, BuffersValues.Rent(coords.Length));
            Span<float> GridVolumeWarpYInterp = GridVolumeWarpY.GetInterpolated(TemporalGridCoords4, BuffersValues.Rent(coords.Length));
            Span<float> GridVolumeWarpZInterp = GridVolumeWarpZ.GetInterpolated(TemporalGridCoords4, BuffersValues.Rent(coords.Length));
            BuffersCoords4.Return(TemporalGridCoords4);

            Span<float> GridDefocusInterp = GridCTFDefocus.GetInterpolated(GridCoords.Slice(0, NTilts), BuffersValues.Rent(NTilts));
            BuffersCoords3.Return(GridCoords);

            Matrix3[] TiltMatrices = Helper.ArrayOfFunction(t => Matrix3.Euler(0, Angles[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);
            Matrix3[] TiltMatricesFlipped = null;
            if (AreAnglesInverted)
                TiltMatricesFlipped = Helper.ArrayOfFunction(t => Matrix3.Euler(0, -Angles[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);

            Span<float3> TransformedCoords = BuffersCoords3.Rent(coords.Length);

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;
                float3 Centered = coords[i] - VolumeCenter;

                Matrix3 Rotation = TiltMatrices[t];

                float3 SampleWarping = new float3(GridVolumeWarpXInterp[i],
                                                  GridVolumeWarpYInterp[i],
                                                  GridVolumeWarpZInterp[i]);
                Centered += SampleWarping;

                float3 Transformed = (Rotation * Centered);

                Transformed.X += TiltAxisOffsetX[t];   // Tilt axis offset is in image space
                Transformed.Y += TiltAxisOffsetY[t];

                Transformed.X += ImageCenter.X;
                Transformed.Y += ImageCenter.Y;

                TransformedCoords[i] = new float3(Transformed.X / ImageDimensionsPhysical.X, Transformed.Y / ImageDimensionsPhysical.Y, t * GridStep);

                result[i] = Transformed;

                // Do the same, but now with Z coordinate and tilt angle flipped
                if (AreAnglesInverted)
                {
                    Rotation = TiltMatricesFlipped[t];
                    Centered.Z *= -1;

                    Transformed = (Rotation * Centered);

                    result[i].Z = Transformed.Z;
                }
            }

            BuffersValues.Return(GridVolumeWarpXInterp);
            BuffersValues.Return(GridVolumeWarpYInterp);
            BuffersValues.Return(GridVolumeWarpZInterp);

            Span<float> GridMovementXInterp = GridMovementX.GetInterpolated(TransformedCoords, BuffersValues.Rent(coords.Length));
            Span<float> GridMovementYInterp = GridMovementY.GetInterpolated(TransformedCoords, BuffersValues.Rent(coords.Length));
            BuffersCoords3.Return(TransformedCoords);

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                // Additional stage shift determined for this tilt
                result[i].X -= GridMovementXInterp[i];
                result[i].Y -= GridMovementYInterp[i];

                // Coordinates are in Angstrom, can be converted directly in um
                result[i].Z = GridDefocusInterp[t] + 1e-4f * result[i].Z;

                result[i] *= SizeRoundingFactors;
            }

            BuffersValues.Return(GridDefocusInterp);
            BuffersValues.Return(GridMovementXInterp);
            BuffersValues.Return(GridMovementYInterp);

            //Console.WriteLine(BuffersCoords4.HasUnreturned());
            //Console.WriteLine(BuffersCoords3.HasUnreturned());
            //Console.WriteLine(BuffersValues.HasUnreturned());

            return result;
        }

        // No support for AreTiltAnglesInverted because this method is only used to trim partially covered voxels
        public float3[] GetPositionInAllTiltsNoLocalWarp(float3[] coords, float3[] result = null)
        {
            if (result == null)
                result = new float3[coords.Length * NTilts];

            float3 VolumeCenter = VolumeDimensionsPhysical / 2;
            float2 ImageCenter = ImageDimensionsPhysical / 2;

            float GridStep = 1f / (NTilts - 1);
            float DoseStep = 1f / (MaxDose - MinDose);
            float _MinDose = MinDose;

            Span<float3> GridCoords = BuffersCoords3.Rent(NTilts);
            Span<float4> TemporalGridCoords4 = BuffersCoords4.Rent(NTilts);
            for (int t = 0; t < NTilts; t++)
            {
                GridCoords[t] = new float3(0.5f, 0.5f, t * GridStep);
                TemporalGridCoords4[t] = new float4(0.5f, 0.5f, 0.5f, (Dose[t] - _MinDose) * DoseStep);
            }

            Span<float> GridVolumeWarpXInterp = GridVolumeWarpX.GetInterpolated(TemporalGridCoords4, BuffersValues.Rent(NTilts));
            Span<float> GridVolumeWarpYInterp = GridVolumeWarpY.GetInterpolated(TemporalGridCoords4, BuffersValues.Rent(NTilts));
            Span<float> GridVolumeWarpZInterp = GridVolumeWarpZ.GetInterpolated(TemporalGridCoords4, BuffersValues.Rent(NTilts));
            BuffersCoords4.Return(TemporalGridCoords4);

            Span<float3> SampleWarpings = BuffersCoords3.Rent(NTilts);
            for (int t = 0; t < NTilts; t++)
                SampleWarpings[t] = new float3(GridVolumeWarpXInterp[t],
                                               GridVolumeWarpYInterp[t],
                                               GridVolumeWarpZInterp[t]);
            BuffersValues.Return(GridVolumeWarpXInterp);
            BuffersValues.Return(GridVolumeWarpYInterp);
            BuffersValues.Return(GridVolumeWarpZInterp);

            Span<float> GridMovementXInterp = GridMovementX.GetInterpolated(GridCoords, BuffersValues.Rent(NTilts));
            Span<float> GridMovementYInterp = GridMovementY.GetInterpolated(GridCoords, BuffersValues.Rent(NTilts));
            BuffersCoords3.Return(GridCoords);


            Matrix3[] OverallRotations = Helper.ArrayOfFunction(t => Matrix3.Euler(0, Angles[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);
            Span<float3> OverallOffsets = BuffersCoords3.Rent(NTilts);
            for (int t = 0; t < NTilts; t++)
                OverallOffsets[t] = new float3(TiltAxisOffsetX[t] + ImageCenter.X - GridMovementXInterp[t],
                                               TiltAxisOffsetY[t] + ImageCenter.Y - GridMovementYInterp[t],
                                               0);
            BuffersValues.Return(GridMovementXInterp);
            BuffersValues.Return(GridMovementYInterp);


            for (int i = 0; i < coords.Length; i++)
            {
                float3 Centered = coords[i] - VolumeCenter;

                for (int t = 0; t < NTilts; t++)
                {
                    float3 Transformed = OverallRotations[t] * (Centered + SampleWarpings[t]) + OverallOffsets[t];

                    result[i * NTilts + t] = Transformed * SizeRoundingFactors;
                }
            }

            BuffersCoords3.Return(OverallOffsets);
            BuffersCoords3.Return(SampleWarpings);

            return result;
        }

        public float3[] GetPositionsInOneTilt(float3[] coords, int tiltID)
        {
            float3[] Result = new float3[coords.Length];

            float3 VolumeCenter = VolumeDimensionsPhysical / 2;
            float2 ImageCenter = ImageDimensionsPhysical / 2;

            float GridStep = 1f / (NTilts - 1);
            float DoseStep = 1f / (MaxDose - MinDose);
            float _MinDose = MinDose;

            Matrix3 TiltMatrix = Matrix3.Euler(0, Angles[tiltID] * Helper.ToRad, -TiltAxisAngles[tiltID] * Helper.ToRad);
            Matrix3 TiltMatrixFlipped = AreAnglesInverted ? Matrix3.Euler(0, -Angles[tiltID] * Helper.ToRad, -TiltAxisAngles[tiltID] * Helper.ToRad) : Matrix3.Zero();

            for (int p = 0; p < coords.Length; p++)
            {
                float3 GridCoords = new float3(coords[p].X / VolumeDimensionsPhysical.X, coords[p].Y / VolumeDimensionsPhysical.Y, tiltID * GridStep);
                float3 Centered = coords[p] - VolumeCenter;

                Matrix3 Rotation = TiltMatrix;

                float4 TemporalGridCoords4 = new float4(GridCoords.X, GridCoords.Y, coords[p].Z / VolumeDimensionsPhysical.Z, (Dose[tiltID] - _MinDose) * DoseStep);
                float3 SampleWarping = new float3(GridVolumeWarpX.GetInterpolated(TemporalGridCoords4),
                                                  GridVolumeWarpY.GetInterpolated(TemporalGridCoords4),
                                                  GridVolumeWarpZ.GetInterpolated(TemporalGridCoords4));
                Centered += SampleWarping;

                float3 Transformed = (Rotation * Centered);

                Transformed.X += TiltAxisOffsetX[tiltID];   // Tilt axis offset is in image space
                Transformed.Y += TiltAxisOffsetY[tiltID];

                Transformed.X += ImageCenter.X;
                Transformed.Y += ImageCenter.Y;

                float3 TransformedCoords = new float3(Transformed.X / ImageDimensionsPhysical.X, Transformed.Y / ImageDimensionsPhysical.Y, tiltID * GridStep);

                // Additional stage shift determined for this tilt
                Transformed.X -= GridMovementX.GetInterpolated(TransformedCoords);
                Transformed.Y -= GridMovementY.GetInterpolated(TransformedCoords);

                // Coordinates are in Angstrom, can be converted directly in um
                Transformed.Z = GridCTFDefocus.GetInterpolated(GridCoords) + 1e-4f * Transformed.Z;

                Result[p] = Transformed;

                // Do the same, but now with Z coordinate and tilt angle flipped
                if (AreAnglesInverted)
                {
                    Rotation = TiltMatrixFlipped;

                    Centered.Z *= -1;

                    Transformed = (Rotation * Centered);

                    // Coordinates are in Angstrom, can be converted directly in um
                    Result[p].Z = GridCTFDefocus.GetInterpolated(GridCoords) + 1e-4f * Transformed.Z;
                }

                Result[p] *= SizeRoundingFactors;
            }

            return Result;
        }

        #endregion

        #region GetAngle methods

        public float3[] GetAngleInAllTilts(float3 coords)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
                PerTiltCoords[i] = coords;

            return GetAngleInAllTilts(PerTiltCoords);
        }

        public float3[] GetAngleInAllTilts(float3[] coords)
        {
            float3[] Result = new float3[coords.Length];

            float GridStep = 1f / (NTilts - 1);

            float3[] GridCoords = new float3[coords.Length];
            float3[] TemporalGridCoords = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;
                GridCoords[i] = new float3(coords[i].X / VolumeDimensionsPhysical.X, coords[i].Y / VolumeDimensionsPhysical.X, t * GridStep);
            }

            float[] GridAngleXInterp = GridAngleX.GetInterpolatedNative(GridCoords);
            float[] GridAngleYInterp = GridAngleY.GetInterpolatedNative(GridCoords);
            float[] GridAngleZInterp = GridAngleZ.GetInterpolatedNative(GridCoords);

            Matrix3[] TiltMatrices = Helper.ArrayOfFunction(t => Matrix3.Euler(0, Angles[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleYInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleXInterp[i] * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * TiltMatrices[t];

                Result[i] = Matrix3.EulerFromMatrix(Rotation);
            }

            return Result;
        }

        public Matrix3[] GetParticleRotationMatrixInAllTilts(float3[] coords, float3[] angle)
        {
            Matrix3[] Result = new Matrix3[coords.Length];

            float GridStep = 1f / (NTilts - 1);

            float3[] GridCoords = new float3[coords.Length];
            float3[] TemporalGridCoords = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;
                GridCoords[i] = new float3(coords[i].X / VolumeDimensionsPhysical.X, coords[i].Y / VolumeDimensionsPhysical.X, t * GridStep);
            }

            float[] GridAngleXInterp = GridAngleX.GetInterpolatedNative(GridCoords);
            float[] GridAngleYInterp = GridAngleY.GetInterpolatedNative(GridCoords);
            float[] GridAngleZInterp = GridAngleZ.GetInterpolatedNative(GridCoords);

            Matrix3[] TiltMatrices = Helper.ArrayOfFunction(t => Matrix3.Euler(0, Angles[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                Matrix3 ParticleMatrix = Matrix3.Euler(angle[i].X * Helper.ToRad,
                                                       angle[i].Y * Helper.ToRad,
                                                       angle[i].Z * Helper.ToRad);


                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleYInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleXInterp[i] * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * TiltMatrices[t] * ParticleMatrix;

                Result[i] = Rotation;
            }

            return Result;
        }

        public float3[] GetParticleAngleInAllTilts(float3 coords, float3 angle)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            float3[] PerTiltAngles = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
            {
                PerTiltCoords[i] = coords;
                PerTiltAngles[i] = angle;
            }

            return GetParticleAngleInAllTilts(PerTiltCoords, PerTiltAngles);
        }

        public float3[] GetParticleAngleInAllTilts(float3[] coords, float3[] angle)
        {
            float3[] Result = new float3[coords.Length];

            Matrix3[] Matrices = GetParticleRotationMatrixInAllTilts(coords, angle);

            for (int i = 0; i < Result.Length; i++)
                Result[i] = Matrix3.EulerFromMatrix(Matrices[i]);

            return Result;
        }

        public float3[] GetAnglesInOneTilt(float3[] coords, float3[] particleAngles, int tiltID)
        {
            int NParticles = coords.Length;
            float3[] Result = new float3[NParticles];

            float GridStep = 1f / (NTilts - 1);

            for (int p = 0; p < NParticles; p++)
            {
                float3 GridCoords = new float3(coords[p].X / VolumeDimensionsPhysical.X, coords[p].Y / VolumeDimensionsPhysical.Y, tiltID * GridStep);

                Matrix3 ParticleMatrix = Matrix3.Euler(particleAngles[p].X * Helper.ToRad,
                                                       particleAngles[p].Y * Helper.ToRad,
                                                       particleAngles[p].Z * Helper.ToRad);

                Matrix3 TiltMatrix = Matrix3.Euler(0, Angles[tiltID] * Helper.ToRad, -TiltAxisAngles[tiltID] * Helper.ToRad);

                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZ.GetInterpolated(GridCoords) * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleY.GetInterpolated(GridCoords) * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleX.GetInterpolated(GridCoords) * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * TiltMatrix * ParticleMatrix;

                Result[p] = Matrix3.EulerFromMatrix(Rotation);
            }

            return Result;
        }

        #endregion

        #region GetImages methods

        public override Image GetImagesForOneParticle(ProcessingOptionsBase options, Image[] tiltData, int size, float3 coords, int planForw = 0, int maskDiameter = -1, int maskEdge = 8, Image result = null, Image resultFT = null)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
                PerTiltCoords[i] = coords;

            return GetImagesForOneParticle(options, tiltData, size, PerTiltCoords, planForw, maskDiameter, maskEdge, true, result, resultFT);
        }

        public override Image GetImagesForOneParticle(ProcessingOptionsBase options, Image[] tiltData, int size, float3[] coordsMoving, int planForw = 0, int maskDiameter = -1, int maskEdge = 8, bool doDecenter = true, Image result = null, Image resultFT = null)
        {
            float3[] ImagePositions = GetPositionInAllTilts(coordsMoving);
            for (int t = 0; t < NTilts; t++)
                ImagePositions[t] /= (float)options.BinnedPixelSizeMean;

            Image Result = result == null ? new Image(new int3(size, size, NTilts)) : result;
            //float[][] ResultData = Result.GetHost(Intent.Write);
            float3[] Shifts = new float3[NTilts];

            int Decenter = doDecenter ? size / 2 : 0;

            IntPtr[] TiltSources = new IntPtr[NTilts];
            int3[] h_Origins = new int3[NTilts];

            for (int t = 0; t < NTilts; t++)
            {
                int3 DimsMovie = tiltData[t].Dims;

                ImagePositions[t] -= size / 2;

                int2 IntPosition = new int2((int)ImagePositions[t].X, (int)ImagePositions[t].Y);
                float2 Residual = new float2(-(ImagePositions[t].X - IntPosition.X), -(ImagePositions[t].Y - IntPosition.Y));
                //IntPosition.X = (IntPosition.X + DimsMovie.X * 99) % DimsMovie.X;                                               // In case it is negative, for the periodic boundaries modulo later
                //IntPosition.Y = (IntPosition.Y + DimsMovie.Y * 99) % DimsMovie.Y;
                Shifts[t] = new float3(Residual.X + Decenter, Residual.Y + Decenter, 0);                                        // Include an fftshift() for Fourier-space rotations later

                TiltSources[t] = tiltData[t].GetDevice(Intent.Read);
                h_Origins[t] = new int3(IntPosition.X, IntPosition.Y, 0);
            }

            GPU.ExtractMultisource(TiltSources,
                                   Result.GetDevice(Intent.Write),
                                   tiltData[0].Dims,
                                   new int3(size).Slice(),
                                   Helper.ToInterleaved(h_Origins),
                                   NTilts,
                                   (uint)NTilts);

            //GPU.NormParticles(Result.GetDevice(Intent.Read),
            //                  Result.GetDevice(Intent.Write),
            //                  Result.Dims.Slice(),
            //                  (uint)Result.Dims.X / 3,
            //                  false,
            //                  (uint)Result.Dims.Z);

            if (maskDiameter > 0)
                GPU.SphereMask(Result.GetDevice(Intent.Read),
                                Result.GetDevice(Intent.Write),
                                Result.Dims.Slice(),
                                maskDiameter / 2f,
                                maskEdge,
                                false,
                                (uint)Result.Dims.Z);

            Image ResultFT = resultFT == null ? new Image(IntPtr.Zero, new int3(size, size, NTilts), true, true) : resultFT;
            GPU.FFT(Result.GetDevice(Intent.Read),
                    ResultFT.GetDevice(Intent.Write),
                    Result.Dims.Slice(),
                    (uint)Result.Dims.Z,
                    planForw);
            ResultFT.Multiply(1f / (size * size));
            ResultFT.ShiftSlices(Shifts);

            if (result == null)
                Result.Dispose();

            return ResultFT;
        }

        #endregion

        #region GetCTFs methods

        public override Image GetCTFsForOneParticle(ProcessingOptionsBase options, float3 coords, Image ctfCoords, Image gammaCorrection, bool weighted = true, bool weightsonly = false, bool useglobalweights = false, Image result = null)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
                PerTiltCoords[i] = coords;

            return GetCTFsForOneParticle(options, PerTiltCoords, ctfCoords, gammaCorrection, weighted, weightsonly, useglobalweights, result);
        }

        public override Image GetCTFsForOneParticle(ProcessingOptionsBase options, float3[] coordsMoving, Image ctfCoords, Image gammaCorrection, bool weighted = true, bool weightsonly = false, bool useglobalweights = false, Image result = null)
        {
            float3[] ImagePositions = GetPositionInAllTilts(coordsMoving);

            float GridStep = 1f / (NTilts - 1);
            CTFStruct[] Params = new CTFStruct[NTilts];
            for (int t = 0; t < NTilts; t++)
            {
                decimal Defocus = (decimal)ImagePositions[t].Z;
                decimal DefocusDelta = (decimal)GetTiltDefocusDelta(t);
                decimal DefocusAngle = (decimal)GetTiltDefocusAngle(t);
                decimal PhaseShift = (decimal)GetTiltPhase(t);

                CTF CurrCTF = CTF.GetCopy();
                CurrCTF.PixelSize = options.BinnedPixelSizeMean;
                if (!weightsonly)
                {
                    CurrCTF.Defocus = Defocus;
                    CurrCTF.DefocusDelta = DefocusDelta;
                    CurrCTF.DefocusAngle = DefocusAngle;
                    CurrCTF.PhaseShift = PhaseShift;
                }
                else
                {
                    CurrCTF.Defocus = 0;
                    CurrCTF.DefocusDelta = 0;
                    CurrCTF.Cs = 0;
                    CurrCTF.Amplitude = 1;
                }

                if (weighted)
                {
                    float3 InterpAt = new float3(coordsMoving[t].X / VolumeDimensionsPhysical.X,
                                                 coordsMoving[t].Y / VolumeDimensionsPhysical.Y,
                                                 t * GridStep);

                    if (GridDoseWeights.Dimensions.Elements() <= 1)
                        CurrCTF.Scale = (decimal)Math.Cos(Angles[t] * Helper.ToRad);
                    else
                        CurrCTF.Scale = (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep)) *
                                        (decimal)GridLocationWeights.GetInterpolated(InterpAt);

                    CurrCTF.Scale *= UseTilt[t] ? 1 : 0.0001M;

                    if (GridDoseBfacs.Dimensions.Elements() <= 1)
                        CurrCTF.Bfactor = (decimal)-Dose[t] * 4;
                    else
                        CurrCTF.Bfactor = (decimal)Math.Min(GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep)), -Dose[t] * 3) +
                                          (decimal)GridLocationBfacs.GetInterpolated(InterpAt);

                    CurrCTF.BfactorDelta = (decimal)GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep));
                    CurrCTF.BfactorAngle = (decimal)GridDoseBfacsAngle.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep));

                    if (useglobalweights)
                    {
                        CurrCTF.Bfactor += (decimal)GlobalBfactor;
                        CurrCTF.Scale *= (decimal)GlobalWeight;
                    }
                }

                Params[t] = CurrCTF.ToStruct();
            }

            Image Result = result == null ? new Image(IntPtr.Zero, new int3(ctfCoords.Dims.X, ctfCoords.Dims.Y, NTilts), true) : result;
            GPU.CreateCTF(Result.GetDevice(Intent.Write),
                                           ctfCoords.GetDevice(Intent.Read),
                                           gammaCorrection == null ? IntPtr.Zero : gammaCorrection.GetDevice(Intent.Read),
                                           (uint)ctfCoords.ElementsSliceComplex,
                                           Params,
                                           false,
                                           (uint)NTilts);

            return Result;
        }

        public void GetCTFsForOneTilt(float pixelSize, float[] defoci, float3[] coords, Image ctfCoords, Image gammaCorrection, int tiltID, Image outSimulated, bool weighted = true, bool weightsonly = false, bool useglobalweights = false)
        {
            int NParticles = defoci.Length;
            CTFStruct[] Params = new CTFStruct[NParticles];

            float GridStep = 1f / (NTilts - 1);

            decimal DefocusDelta = (decimal)GetTiltDefocusDelta(tiltID);
            decimal DefocusAngle = (decimal)GetTiltDefocusAngle(tiltID);
            decimal PhaseShift = (decimal)GetTiltPhase(tiltID);

            CTF ProtoCTF = CTF.GetCopy();
            ProtoCTF.PixelSize = (decimal)pixelSize;
            if (!weightsonly)
            {
                ProtoCTF.DefocusDelta = DefocusDelta;
                ProtoCTF.DefocusAngle = DefocusAngle;
                ProtoCTF.PhaseShift = PhaseShift;
            }
            else
            {
                ProtoCTF.Defocus = 0;
                ProtoCTF.DefocusDelta = 0;
                ProtoCTF.Cs = 0;
                ProtoCTF.Amplitude = 1;
            }

            decimal Bfac = 0;
            decimal BfacDelta = 0;
            decimal BfacAngle = 0;
            decimal Weight = 1;

            if (weighted)
            {
                if (GridDoseBfacs.Dimensions.Elements() <= 1)
                    Bfac = (decimal)-Dose[tiltID] * 4;
                else
                    Bfac = (decimal)Math.Min(GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep)), -Dose[tiltID] * 3);

                if (GridDoseWeights.Dimensions.Elements() <= 1)
                    Weight = (decimal)Math.Cos(Angles[tiltID] * Helper.ToRad);
                else
                    Weight = (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));

                Weight *= UseTilt[tiltID] ? 1 : 0.0001M;

                if (useglobalweights)
                {
                    Bfac += (decimal)GlobalBfactor;
                    Weight *= (decimal)GlobalWeight;
                }

                BfacDelta = (decimal)GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));
                BfacAngle = (decimal)GridDoseBfacsAngle.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));
            }

            for (int p = 0; p < NParticles; p++)
            {
                if (weighted)
                {
                    ProtoCTF.Bfactor = Bfac;
                    ProtoCTF.BfactorDelta = BfacDelta;
                    ProtoCTF.BfactorAngle = BfacAngle;
                    ProtoCTF.Scale = Weight;

                    float3 InterpAt = new float3(coords[p].X / VolumeDimensionsPhysical.X,
                                                 coords[p].Y / VolumeDimensionsPhysical.Y,
                                                 0.5f);
                    ProtoCTF.Bfactor += (decimal)GridLocationBfacs.GetInterpolated(InterpAt);
                    ProtoCTF.Scale *= (decimal)GridLocationWeights.GetInterpolated(InterpAt);
                }

                if (!weightsonly)
                    ProtoCTF.Defocus = (decimal)defoci[p];

                Params[p] = ProtoCTF.ToStruct();
            }

            GPU.CreateCTF(outSimulated.GetDevice(Intent.Write),
                                                 ctfCoords.GetDevice(Intent.Read),
                                                 gammaCorrection == null ? IntPtr.Zero : gammaCorrection.GetDevice(Intent.Read),
                                                 (uint)ctfCoords.ElementsSliceComplex,
                                                 Params,
                                                 false,
                                                 (uint)NParticles);
        }

        public void GetComplexCTFsForOneTilt(float pixelSize, float[] defoci, float3[] coords, Image ctfCoords, Image gammaCorrection, int tiltID, bool reverse, Image outSimulated, bool weighted = true, bool weightsonly = false, bool useglobalweights = false)
        {
            int NParticles = defoci.Length;
            CTFStruct[] Params = new CTFStruct[NParticles];

            float GridStep = 1f / (NTilts - 1);

            decimal DefocusDelta = (decimal)GetTiltDefocusDelta(tiltID);
            decimal DefocusAngle = (decimal)GetTiltDefocusAngle(tiltID);
            decimal PhaseShift = (decimal)GetTiltPhase(tiltID);

            CTF ProtoCTF = CTF.GetCopy();
            ProtoCTF.PixelSize = (decimal)pixelSize;
            if (!weightsonly)
            {
                ProtoCTF.DefocusDelta = DefocusDelta;
                ProtoCTF.DefocusAngle = DefocusAngle;
                ProtoCTF.PhaseShift = PhaseShift;
            }
            else
            {
                ProtoCTF.Defocus = 0;
                ProtoCTF.DefocusDelta = 0;
                ProtoCTF.Cs = 0;
                ProtoCTF.Amplitude = 1;
            }

            decimal Bfac = 0;
            decimal BfacDelta = 0;
            decimal BfacAngle = 0;
            decimal Weight = 1;

            if (weighted)
            {
                if (GridDoseBfacs.Dimensions.Elements() <= 1)
                    Bfac = (decimal)-Dose[tiltID] * 4;
                else
                    Bfac = (decimal)Math.Min(GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep)), -Dose[tiltID] * 3);

                if (GridDoseWeights.Dimensions.Elements() <= 1)
                    Weight = (decimal)Math.Cos(Angles[tiltID] * Helper.ToRad);
                else
                    Weight = (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));

                Weight *= UseTilt[tiltID] ? 1 : 0.0001M;

                if (useglobalweights)
                {
                    Bfac += (decimal)GlobalBfactor;
                    Weight *= (decimal)GlobalWeight;
                }

                BfacDelta = (decimal)GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));
                BfacAngle = (decimal)GridDoseBfacsAngle.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));
            }

            for (int p = 0; p < NParticles; p++)
            {
                if (weighted)
                {
                    ProtoCTF.Bfactor = Bfac;
                    ProtoCTF.BfactorDelta = BfacDelta;
                    ProtoCTF.BfactorAngle = BfacAngle;
                    ProtoCTF.Scale = Weight;

                    float3 InterpAt = new float3(coords[p].X / VolumeDimensionsPhysical.X,
                                                 coords[p].Y / VolumeDimensionsPhysical.Y,
                                                 0.5f);
                    ProtoCTF.Bfactor += (decimal)GridLocationBfacs.GetInterpolated(InterpAt);
                    ProtoCTF.Scale *= (decimal)GridLocationWeights.GetInterpolated(InterpAt);
                }

                if (!weightsonly)
                    ProtoCTF.Defocus = (decimal)defoci[p];

                Params[p] = ProtoCTF.ToStruct();
            }

            GPU.CreateCTFComplex(outSimulated.GetDevice(Intent.Write),
                                                 ctfCoords.GetDevice(Intent.Read),
                                                 gammaCorrection == null ? IntPtr.Zero : gammaCorrection.GetDevice(Intent.Read),
                                                 (uint)ctfCoords.ElementsSliceComplex,
                                                 Params,
                                                 reverse,
                                                 (uint)NParticles);
        }

        public CTF[] GetCTFParamsForOneTilt(float pixelSize, float[] defoci, float3[] coords, int tiltID, bool weighted = true, bool weightsonly = false, bool useglobalweights = false)
        {
            int NParticles = defoci.Length;
            CTF[] Params = new CTF[NParticles];

            float GridStep = 1f / (NTilts - 1);

            decimal DefocusDelta = (decimal)GetTiltDefocusDelta(tiltID);
            decimal DefocusAngle = (decimal)GetTiltDefocusAngle(tiltID);
            decimal PhaseShift = (decimal)GetTiltPhase(tiltID);

            CTF ProtoCTF = CTF.GetCopy();
            ProtoCTF.PixelSize = (decimal)pixelSize;
            if (!weightsonly)
            {
                ProtoCTF.DefocusDelta = DefocusDelta;
                ProtoCTF.DefocusAngle = DefocusAngle;
                ProtoCTF.PhaseShift = PhaseShift;
            }
            else
            {
                ProtoCTF.Defocus = 0;
                ProtoCTF.DefocusDelta = 0;
                ProtoCTF.Cs = 0;
                ProtoCTF.Amplitude = 1;
            }

            decimal Bfac = 0;
            decimal BfacDelta = 0;
            decimal BfacAngle = 0;
            decimal Weight = 1;

            if (weighted)
            {
                if (GridDoseBfacs.Dimensions.Elements() <= 1)
                    Bfac = (decimal)-Dose[tiltID] * 4;
                else
                    Bfac = (decimal)Math.Min(GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep)), -Dose[tiltID] * 3);

                if (GridDoseWeights.Dimensions.Elements() <= 1)
                    Weight = (decimal)Math.Cos(Angles[tiltID] * Helper.ToRad);
                else
                    Weight = (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));

                Weight *= UseTilt[tiltID] ? 1 : 0.0001M;

                if (useglobalweights)
                {
                    Bfac += (decimal)GlobalBfactor;
                    Weight *= (decimal)GlobalWeight;
                }

                BfacDelta = (decimal)GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));
                BfacAngle = (decimal)GridDoseBfacsAngle.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));
            }

            for (int p = 0; p < NParticles; p++)
            {
                if (weighted)
                {
                    ProtoCTF.Bfactor = Bfac;
                    ProtoCTF.BfactorDelta = BfacDelta;
                    ProtoCTF.BfactorAngle = BfacAngle;
                    ProtoCTF.Scale = Weight;

                    float3 InterpAt = new float3(coords[p].X / VolumeDimensionsPhysical.X,
                                                 coords[p].Y / VolumeDimensionsPhysical.Y,
                                                 0.5f);
                    ProtoCTF.Bfactor += (decimal)GridLocationBfacs.GetInterpolated(InterpAt);
                    ProtoCTF.Scale *= (decimal)GridLocationWeights.GetInterpolated(InterpAt);
                }

                if (!weightsonly)
                    ProtoCTF.Defocus = (decimal)defoci[p];

                Params[p] = ProtoCTF.GetCopy();
            }

            return Params;
        }

        #endregion

        #region Many-particles GetImages and GetCTFs

        public Image GetParticleImagesFromOneTilt(Image tiltStack, int size, float3[] particleOrigins, int angleID, bool normalize)
        {
            int NParticles = particleOrigins.Length;

            float3[] ImagePositions = GetPositionsInOneTilt(particleOrigins, angleID);

            Image Result = new Image(new int3(size, size, NParticles));
            float[][] ResultData = Result.GetHost(Intent.Write);
            float3[] Shifts = new float3[NParticles];

            int3 DimsStack = tiltStack.Dims;

            Parallel.For(0, NParticles, p =>
            {
                ImagePositions[p] -= new float3(size / 2, size / 2, 0);
                int2 IntPosition = new int2((int)ImagePositions[p].X, (int)ImagePositions[p].Y);
                float2 Residual = new float2(-(ImagePositions[p].X - IntPosition.X), -(ImagePositions[p].Y - IntPosition.Y));
                Residual -= size / 2;
                Shifts[p] = new float3(Residual);

                float[] OriginalData;
                lock (tiltStack)
                    OriginalData = tiltStack.GetHost(Intent.Read)[angleID];

                float[] ImageData = ResultData[p];
                for (int y = 0; y < size; y++)
                {
                    int PosY = (y + IntPosition.Y + DimsStack.Y) % DimsStack.Y;
                    for (int x = 0; x < size; x++)
                    {
                        int PosX = (x + IntPosition.X + DimsStack.X) % DimsStack.X;
                        ImageData[y * size + x] = OriginalData[PosY * DimsStack.X + PosX];
                    }
                }
            });
            if (normalize)
                GPU.NormParticles(Result.GetDevice(Intent.Read),
                                  Result.GetDevice(Intent.Write),
                                  Result.Dims.Slice(),
                                  (uint)(123 / CTF.PixelSize),     // FIX THE PARTICLE RADIUS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                  true,
                                  (uint)NParticles);
            //Result.WriteMRC($"d_paticleimages_{angleID:D3}.mrc");

            Result.ShiftSlices(Shifts);

            Image ResultFT = Result.AsFFT();
            Result.Dispose();

            return ResultFT;
        }

        public Image GetParticleSeriesFromMovies(Movie[] movies, Image[] movieData, int size, float3[] particleOrigins, float pixelSize, int planForw = 0)
        {
            int NParticles = particleOrigins.Length;
            Image Result = new Image(new int3(size, size, NParticles * NTilts), true, true);
            float[][] ResultData = Result.GetHost(Intent.Write);

            int PlanForw = planForw > 0 ? planForw : GPU.CreateFFTPlan(new int3(size, size, 1), (uint)NParticles);

            Image ParticleExtracts = new Image(IntPtr.Zero, new int3(size, size, NParticles));
            Image ParticleExtractsFT = new Image(IntPtr.Zero, new int3(size, size, NParticles), true, true);

            for (int t = 0; t < NTilts; t++)
            {
                float3 Scaling = new float3(1 / pixelSize, 1 / pixelSize, 1);
                float3 DimsInv = new float3(1f / movieData[t].Dims.X,
                                            1f / movieData[t].Dims.Y,
                                            1f / Math.Max(1, movieData[t].Dims.Z));

                float3[] ImagePositions = GetPositionsInOneTilt(particleOrigins, t);
                for (int p = 0; p < NParticles; p++)
                    ImagePositions[p] *= Scaling;       // Tilt image positions are returned in Angstroms initially

                float3[] MovieGridPositions = new float3[NParticles];
                for (int p = 0; p < NParticles; p++)
                    MovieGridPositions[p] = new float3(ImagePositions[p].X * DimsInv.X,
                                                       ImagePositions[p].Y * DimsInv.Y,
                                                       0);

                int3[] ExtractPositions = new int3[NParticles];
                float3[] ResidualShifts = new float3[NParticles];

                Image ParticleSumsFT = new Image(IntPtr.Zero, new int3(size, size, NParticles), true, true);
                ParticleSumsFT.Fill(0);

                for (int z = 0; z < movieData[t].Dims.Z; z++)
                {
                    for (int p = 0; p < NParticles; p++)
                        MovieGridPositions[p].Z = z * DimsInv.Z;

                    float2[] FrameShifts = movies[t].GetShiftFromPyramid(MovieGridPositions);
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3 Shifted = new float3(ImagePositions[p].X - FrameShifts[p].X / pixelSize,     // Don't forget, shifts are stored in Angstroms
                                                    ImagePositions[p].Y - FrameShifts[p].Y / pixelSize,
                                                    0);
                        ExtractPositions[p] = new int3(Shifted);
                        ResidualShifts[p] = new float3(ExtractPositions[p].X - Shifted.X + size / 2,
                                                       ExtractPositions[p].Y - Shifted.Y + size / 2,
                                                       0);

                        GPU.Extract(movieData[t].GetDeviceSlice(z, Intent.Read),
                                    ParticleExtracts.GetDevice(Intent.Write),
                                    movieData[t].Dims.Slice(),
                                    new int3(size, size, 1),
                                    Helper.ToInterleaved(ExtractPositions),
                                    false,
                                    (uint)NParticles);

                        GPU.FFT(ParticleExtracts.GetDevice(Intent.Read),
                                ParticleExtractsFT.GetDevice(Intent.Write),
                                ParticleExtracts.Dims.Slice(),
                                (uint)NParticles,
                                PlanForw);

                        ParticleExtractsFT.ShiftSlices(ResidualShifts);

                        ParticleSumsFT.Add(ParticleExtracts);
                    }
                }

                ParticleSumsFT.Multiply(1f / size / size / movieData[t].Dims.Z);

                float[][] ParticleSumsFTData = ParticleSumsFT.GetHost(Intent.Read);
                for (int p = 0; p < NParticles; p++)
                    ResultData[p * NTilts + t] = ParticleSumsFTData[p];

                ParticleSumsFT.Dispose();
                movieData[t].FreeDevice();
            }

            ParticleExtracts.Dispose();
            ParticleExtractsFT.Dispose();
            if (planForw <= 0)
                GPU.DestroyFFTPlan(PlanForw);

            return Result;
        }

        #endregion

        public override int[] GetRelevantImageSizes(int fullSize, float weightingThreshold)
        {
            int[] Result = new int[NTilts];

            float[][] AllWeights = new float[NTilts][];

            float GridStep = 1f / (NTilts - 1);
            for (int t = 0; t < NTilts; t++)
            {
                CTF CurrCTF = CTF.GetCopy();

                CurrCTF.Defocus = 0;
                CurrCTF.DefocusDelta = 0;
                CurrCTF.Cs = 0;
                CurrCTF.Amplitude = 1;

                if (GridDoseBfacs.Dimensions.Elements() <= 1)
                    CurrCTF.Bfactor = (decimal)-Dose[t] * 4;
                else
                    CurrCTF.Bfactor = (decimal)(GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep)) +
                                                Math.Abs(GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep))));

                AllWeights[t] = CurrCTF.Get1D(fullSize / 2, false);
            }

            int elementID = IndicesSortedDose[0];
            if (GridDoseBfacs.Dimensions.Elements() > 1)
                (elementID, _) = MathHelper.MaxElement(GridDoseBfacs.FlatValues);
            float[] LowerDoseWeights = AllWeights[elementID].ToList().ToArray();

            for (int t = 0; t < NTilts; t++)
            {
                for (int i = 0; i < LowerDoseWeights.Length; i++)
                    AllWeights[t][i] /= LowerDoseWeights[i];

                int MaxShell = 0;
                while (MaxShell < AllWeights[t].Length)
                {
                    if (AllWeights[t][MaxShell] < weightingThreshold)
                        break;
                    MaxShell++;
                }

                Result[t] = Math.Max(2, Math.Min(fullSize, MaxShell * 2));
            }

            return Result;
        }

        static Image[] _RawDataBuffers = null;
        static Image[] RawDataBuffers
        {
            get
            {
                if (_RawDataBuffers == null)
                    _RawDataBuffers = new Image[GPU.GetDeviceCount()];
                return _RawDataBuffers;
            }
        }
        static Image[][] _ScaledTiltBuffers = null;
        static Image[][] ScaledTiltBuffers
        {
            get
            {
                if (_ScaledTiltBuffers == null)
                    _ScaledTiltBuffers = new Image[GPU.GetDeviceCount()][];
                return _ScaledTiltBuffers;
            }
        }
        static Image[][] _ScaledTiltBuffersOdd = null;
        static Image[][] ScaledTiltBuffersOdd
        {
            get
            {
                if (_ScaledTiltBuffersOdd == null)
                    _ScaledTiltBuffersOdd = new Image[GPU.GetDeviceCount()][];
                return _ScaledTiltBuffersOdd;
            }
        }
        static Image[][] _ScaledTiltBuffersEven = null;
        static Image[][] ScaledTiltBuffersEven
        {
            get
            {
                if (_ScaledTiltBuffersEven == null)
                    _ScaledTiltBuffersEven = new Image[GPU.GetDeviceCount()][];
                return _ScaledTiltBuffersEven;
            }
        }

        public void LoadMovieData(ProcessingOptionsBase options, out Movie[] movies, out Image[] movieData, bool doOddEven, out Image[] movieDataOdd, out Image[] movieDataEven, bool useDenoised = false)
        {
            if (TiltMoviePaths.Length != NTilts)
                throw new Exception("A valid path is needed for each tilt.");

            if (options.EERGroupFrames > 0)
                HeaderEER.GroupNFrames = options.EERGroupFrames;

            movies = new Movie[NTilts];

            for (int t = 0; t < NTilts; t++)
                movies[t] = new Movie(IOPath.Combine(DataOrProcessingDirectoryName, TiltMoviePaths[t]));

            MapHeader Header = MapHeader.ReadFromFile(movies[0].DataPath);

            ImageDimensionsPhysical = new float2(Header.Dimensions.X, Header.Dimensions.Y) * (float)options.PixelSizeMean;

            int2 DimsScaled = new int2((int)Math.Round(Header.Dimensions.X / (float)options.DownsampleFactor / 2) * 2,
                                        (int)Math.Round(Header.Dimensions.Y / (float)options.DownsampleFactor / 2) * 2);

            SizeRoundingFactors = new float3(DimsScaled.X / (Header.Dimensions.X / (float)options.DownsampleFactor),
                                             DimsScaled.Y / (Header.Dimensions.Y / (float)options.DownsampleFactor),
                                             1);

            Header = MapHeader.ReadFromFile(useDenoised ? movies[0].AverageDenoisedPath : movies[0].AveragePath);
            if (Header.Dimensions.Z > 1)
                throw new Exception("This average has more than one layer.");

            bool DoScale = DimsScaled != new int2(Header.Dimensions);

            int PlanForw = 0, PlanBack = 0;
            if (DoScale)
            {
                PlanForw = GPU.CreateFFTPlan(Header.Dimensions.Slice(), 1);
                PlanBack = GPU.CreateIFFTPlan(new int3(DimsScaled), 1);
            }

            int CurrentDevice = GPU.GetDevice();

            #region Make sure reusable buffers are there and have correct dimensions

            if (RawDataBuffers[CurrentDevice] == null || RawDataBuffers[CurrentDevice].ElementsReal != Header.Dimensions.Elements())
            {
                if (RawDataBuffers[CurrentDevice] != null)
                    RawDataBuffers[CurrentDevice].Dispose();

                RawDataBuffers[CurrentDevice] = new Image(Header.Dimensions);
            }

            if (ScaledTiltBuffers[CurrentDevice] == null ||
                ScaledTiltBuffers[CurrentDevice].Length < NTilts ||
                ScaledTiltBuffers[CurrentDevice][0].ElementsReal != DimsScaled.Elements())
            {
                if (ScaledTiltBuffers[CurrentDevice] != null)
                    foreach (var item in ScaledTiltBuffers[CurrentDevice])
                        item.Dispose();

                ScaledTiltBuffers[CurrentDevice] = Helper.ArrayOfFunction(i => new Image(new int3(DimsScaled)), NTilts);
            }

            if (ScaledTiltBuffersOdd[CurrentDevice] == null ||
                ScaledTiltBuffersOdd[CurrentDevice].Length < NTilts ||
                ScaledTiltBuffersOdd[CurrentDevice][0].ElementsReal != DimsScaled.Elements())
            {
                if (ScaledTiltBuffersOdd[CurrentDevice] != null)
                {
                    foreach (var item in ScaledTiltBuffersOdd[CurrentDevice])
                        item.Dispose();
                    foreach (var item in ScaledTiltBuffersEven[CurrentDevice])
                        item.Dispose();
                }

                ScaledTiltBuffersOdd[CurrentDevice] = Helper.ArrayOfFunction(i => new Image(new int3(DimsScaled)), NTilts);
                ScaledTiltBuffersEven[CurrentDevice] = Helper.ArrayOfFunction(i => new Image(new int3(DimsScaled)), NTilts);
            }

            #endregion

            for (int t = 0; t < NTilts; t++)
            {
                IOHelper.ReadMapFloatPatient(50, 500,
                                             useDenoised ? movies[t].AverageDenoisedPath : movies[t].AveragePath,
                                             new int2(1),
                                             0,
                                             typeof(float),
                                             new[] { 0 },
                                             null,
                                             RawDataBuffers[CurrentDevice].GetHost(Intent.Write));

                if (DoScale)
                {
                    GPU.Scale(RawDataBuffers[CurrentDevice].GetDevice(Intent.Read),
                              ScaledTiltBuffers[CurrentDevice][t].GetDevice(Intent.Write),
                              Header.Dimensions,
                              new int3(DimsScaled),
                              1,
                              PlanForw,
                              PlanBack,
                              IntPtr.Zero,
                              IntPtr.Zero);

                    ScaledTiltBuffers[CurrentDevice][t].FreeDevice();
                }
                else
                {
                    Array.Copy(RawDataBuffers[CurrentDevice].GetHost(Intent.Read)[0], 0,
                               ScaledTiltBuffers[CurrentDevice][t].GetHost(Intent.Write)[0], 0,
                               (int)Header.Dimensions.Elements());
                }

                if (doOddEven)
                {
                    {
                        if (!File.Exists(movies[t].AverageOddPath))
                            throw new Exception("No odd/even frame averages found for this tilt series. Please re-process the movies with that export option checked.");

                        IOHelper.ReadMapFloatPatient(50, 500,
                                                     movies[t].AverageOddPath,
                                                     new int2(1),
                                                     0,
                                                     typeof(float),
                                                     new[] { 0 },
                                                     null,
                                                     RawDataBuffers[CurrentDevice].GetHost(Intent.Write));

                        if (DoScale)
                        {
                            GPU.Scale(RawDataBuffers[CurrentDevice].GetDevice(Intent.Read),
                                      ScaledTiltBuffersOdd[CurrentDevice][t].GetDevice(Intent.Write),
                                      Header.Dimensions,
                                      new int3(DimsScaled),
                                      1,
                                      PlanForw,
                                      PlanBack,
                                      IntPtr.Zero,
                                      IntPtr.Zero);

                            ScaledTiltBuffersOdd[CurrentDevice][t].FreeDevice();
                        }
                        else
                        {
                            Array.Copy(RawDataBuffers[CurrentDevice].GetHost(Intent.Read)[0], 0,
                                       ScaledTiltBuffersOdd[CurrentDevice][t].GetHost(Intent.Write)[0], 0,
                                       (int)Header.Dimensions.Elements());
                        }
                    }
                    {
                        if (!File.Exists(movies[t].AverageEvenPath))
                            throw new Exception("No odd/even frame averages found for this tilt series. Please re-process the movies with that export option checked.");

                        IOHelper.ReadMapFloatPatient(50, 500,
                                                     movies[t].AverageEvenPath,
                                                     new int2(1),
                                                     0,
                                                     typeof(float),
                                                     new[] { 0 },
                                                     null,
                                                     RawDataBuffers[CurrentDevice].GetHost(Intent.Write));

                        if (DoScale)
                        {
                            GPU.Scale(RawDataBuffers[CurrentDevice].GetDevice(Intent.Read),
                                      ScaledTiltBuffersEven[CurrentDevice][t].GetDevice(Intent.Write),
                                      Header.Dimensions,
                                      new int3(DimsScaled),
                                      1,
                                      PlanForw,
                                      PlanBack,
                                      IntPtr.Zero,
                                      IntPtr.Zero);

                            ScaledTiltBuffersEven[CurrentDevice][t].FreeDevice();
                        }
                        else
                        {
                            Array.Copy(RawDataBuffers[CurrentDevice].GetHost(Intent.Read)[0], 0,
                                       ScaledTiltBuffersEven[CurrentDevice][t].GetHost(Intent.Write)[0], 0,
                                       (int)Header.Dimensions.Elements());
                        }
                    }
                }
            }

            if (DoScale)
            {
                GPU.DestroyFFTPlan(PlanForw);
                GPU.DestroyFFTPlan(PlanBack);
            }

            movieData = ScaledTiltBuffers[CurrentDevice].Take(NTilts).ToArray();
            if (doOddEven)
            {
                movieDataOdd = ScaledTiltBuffersOdd[CurrentDevice].Take(NTilts).ToArray();
                movieDataEven = ScaledTiltBuffersEven[CurrentDevice].Take(NTilts).ToArray();
            }
            else
            {
                movieDataOdd = null;
                movieDataEven = null;
            }
        }

        public void LoadMovieSizes()
        {
            if (TiltMoviePaths.Length != NTilts)
                throw new Exception("A valid path is needed for each tilt.");

            Movie First = new Movie(IOPath.Combine(DataOrProcessingDirectoryName, TiltMoviePaths[0]));
            MapHeader Header = MapHeader.ReadFromFile(First.AveragePath);

            ImageDimensionsPhysical = new float2(Header.Dimensions.X, Header.Dimensions.Y) * Header.PixelSize.X;
        }

        static Image[] _RawMaskBuffers = null;
        static Image[] RawMaskBuffers
        {
            get
            {
                if (_RawMaskBuffers == null)
                    _RawMaskBuffers = new Image[GPU.GetDeviceCount()];
                return _RawMaskBuffers;
            }
        }
        static Image[][] _ScaledMaskBuffers = null;
        static Image[][] ScaledMaskBuffers
        {
            get
            {
                if (_ScaledMaskBuffers == null)
                    _ScaledMaskBuffers = new Image[GPU.GetDeviceCount()][];
                return _ScaledMaskBuffers;
            }
        }
        public void LoadMovieMasks(ProcessingOptionsBase options, out Image[] maskData)
        {
            Movie First = new Movie(IOPath.Combine(DataOrProcessingDirectoryName, TiltMoviePaths[0]));
            MapHeader Header = MapHeader.ReadFromFile(First.DataPath);

            int2 DimsScaled = new int2((int)Math.Round(Header.Dimensions.X / (float)options.DownsampleFactor / 2) * 2,
                                       (int)Math.Round(Header.Dimensions.Y / (float)options.DownsampleFactor / 2) * 2);

            int CurrentDevice = GPU.GetDevice();

            #region Make sure reusable buffers are there and have correct dimensions

            if (ScaledMaskBuffers[CurrentDevice] == null ||
                ScaledMaskBuffers[CurrentDevice].Length < NTilts ||
                ScaledMaskBuffers[CurrentDevice][0].ElementsReal != DimsScaled.Elements())
            {
                if (ScaledMaskBuffers[CurrentDevice] != null)
                    foreach (var item in ScaledMaskBuffers[CurrentDevice])
                        item.Dispose();

                ScaledMaskBuffers[CurrentDevice] = Helper.ArrayOfFunction(i => new Image(new int3(DimsScaled)), NTilts);
            }

            #endregion

            maskData = new Image[NTilts];

            for (int t = 0; t < NTilts; t++)
            {
                Movie M = new Movie(IOPath.Combine(DataOrProcessingDirectoryName, TiltMoviePaths[t]));
                string MaskPath = M.MaskPath;

                if (File.Exists(MaskPath))
                {
                    MapHeader MaskHeader = MapHeader.ReadFromFile(MaskPath);

                    if (RawMaskBuffers[CurrentDevice] == null || RawMaskBuffers[CurrentDevice].ElementsReal != MaskHeader.Dimensions.Elements())
                    {
                        if (RawMaskBuffers[CurrentDevice] != null)
                            RawMaskBuffers[CurrentDevice].Dispose();

                        RawMaskBuffers[CurrentDevice] = new Image(MaskHeader.Dimensions);
                    }


                    TiffNative.ReadTIFFPatient(50, 500, MaskPath, 0, true, RawMaskBuffers[CurrentDevice].GetHost(Intent.Write)[0]);

                    #region Rescale

                    GPU.Scale(RawMaskBuffers[CurrentDevice].GetDevice(Intent.Read),
                              ScaledMaskBuffers[CurrentDevice][t].GetDevice(Intent.Write),
                              MaskHeader.Dimensions,
                              new int3(DimsScaled),
                              1,
                              0,
                              0,
                              IntPtr.Zero,
                              IntPtr.Zero);

                    ScaledMaskBuffers[CurrentDevice][t].Binarize(0.7f);
                    ScaledMaskBuffers[CurrentDevice][t].FreeDevice();

                    #endregion

                    maskData[t] = ScaledMaskBuffers[CurrentDevice][t];
                }
            }
        }

        static int[][] DirtErasureLabelsBuffer = null;
        static Image[] DirtErasureMaskBuffer = null;
        public static void EraseDirt(Image tiltImage, Image tiltMask, float noiseScale = 0.1f)
        {
            if (tiltMask == null)
                return;

            float[] ImageData = tiltImage.GetHost(Intent.ReadWrite)[0];

            int CurrentDevice = GPU.GetDevice();

            #region Make sure reusable buffers are there and correctly sized
            
            if (DirtErasureLabelsBuffer == null || DirtErasureLabelsBuffer.Length != GPU.GetDeviceCount())
                DirtErasureLabelsBuffer = new int[GPU.GetDeviceCount()][];

            if (DirtErasureLabelsBuffer[CurrentDevice] == null || DirtErasureLabelsBuffer[CurrentDevice].Length != ImageData.Length)
                DirtErasureLabelsBuffer[CurrentDevice] = new int[ImageData.Length];
            
            if (DirtErasureMaskBuffer == null || DirtErasureMaskBuffer.Length != GPU.GetDeviceCount())
                DirtErasureMaskBuffer = new Image[GPU.GetDeviceCount()];

            if (DirtErasureMaskBuffer[CurrentDevice] == null || DirtErasureMaskBuffer[CurrentDevice].Dims != tiltMask.Dims)
            {
                if (DirtErasureMaskBuffer[CurrentDevice] != null)
                    DirtErasureMaskBuffer[CurrentDevice].Dispose();

                DirtErasureMaskBuffer[CurrentDevice] = new Image(tiltMask.Dims);
            }

            #endregion

            var Components = tiltMask.GetConnectedComponents(8, DirtErasureLabelsBuffer[CurrentDevice]);

            #region Inline Image.AsExpandedSmooth to use reusable buffer

            GPU.DistanceMapExact(tiltMask.GetDevice(Intent.Read), DirtErasureMaskBuffer[CurrentDevice].GetDevice(Intent.Write), tiltMask.Dims, 12);
            DirtErasureMaskBuffer[CurrentDevice].Multiply((float)Math.PI / 12f);
            DirtErasureMaskBuffer[CurrentDevice].Cos();
            DirtErasureMaskBuffer[CurrentDevice].Add(1);
            DirtErasureMaskBuffer[CurrentDevice].Multiply(0.5f);

            #endregion

            RandomNormal RandN = new RandomNormal((int)(MathHelper.Normalize(ImageData.Take(10000).ToArray()).Select(v => Math.Abs(v)).Sum() * 100));

            float[] MaskSmoothData = DirtErasureMaskBuffer[CurrentDevice].GetHost(Intent.Read)[0];
            DirtErasureMaskBuffer[CurrentDevice].FreeDevice();

            foreach (var component in Components)
            {
                if (component.NeighborhoodIndices.Length < 2)
                    continue;

                float[] NeighborhoodIntensities = Helper.IndexedSubset(ImageData, component.NeighborhoodIndices);
                float2 MeanStd = MathHelper.MeanAndStd(NeighborhoodIntensities);
                float ComponentMean = MeanStd.X;
                float ComponentStd = MeanStd.Y; 

                foreach (int id in component.ComponentIndices)
                    ImageData[id] = RandN.NextSingle(ComponentMean, ComponentStd * noiseScale);

                foreach (int id in component.NeighborhoodIndices)
                    ImageData[id] = MathHelper.Lerp(ImageData[id], RandN.NextSingle(ComponentMean, ComponentStd * noiseScale), MaskSmoothData[id]);
            }
        }

        public static void FreeDeviceBuffers()
        {
            foreach (var item in RawDataBuffers)
                item?.FreeDevice();
            foreach (var item in ScaledTiltBuffers)
                if (item != null)
                    foreach (var subitem in item)
                        subitem?.FreeDevice();
            foreach (var item in ScaledTiltBuffersOdd)
                if (item != null)
                    foreach (var subitem in item)
                        subitem?.FreeDevice();
            foreach (var item in ScaledTiltBuffersEven)
                if (item != null)
                    foreach (var subitem in item)
                        subitem?.FreeDevice();

            foreach (var item in RawMaskBuffers)
                item?.FreeDevice();
            foreach (var item in ScaledMaskBuffers)
                if (item != null)
                    foreach (var subitem in item)
                        subitem?.FreeDevice();

            foreach (var item in DirtErasureMaskBuffer)
                item?.FreeDevice();
        }

        #endregion

        #region Load/save meta

        public override void LoadMeta()
        {
            if (!File.Exists(XMLPath))
                return;

            try
            {
                using (Stream SettingsStream = File.OpenRead(XMLPath))
                {
                    XPathDocument Doc = new XPathDocument(SettingsStream);
                    XPathNavigator Reader = Doc.CreateNavigator();
                    Reader.MoveToRoot();
                    Reader.MoveToFirstChild();

                    #region Attributes

                    string DataDirectory = XMLHelper.LoadAttribute(Reader, "DataDirectory", "");
                    if (!string.IsNullOrEmpty(DataDirectory))
                        DataDirectoryName = DataDirectory;

                    AreAnglesInverted = XMLHelper.LoadAttribute(Reader, "AreAnglesInverted", AreAnglesInverted);
                    PlaneNormal = XMLHelper.LoadAttribute(Reader, "PlaneNormal", PlaneNormal);

                    GlobalBfactor = XMLHelper.LoadAttribute(Reader, "Bfactor", GlobalBfactor);
                    GlobalWeight = XMLHelper.LoadAttribute(Reader, "Weight", GlobalWeight);

                    MagnificationCorrection = XMLHelper.LoadAttribute(Reader, "MagnificationCorrection", MagnificationCorrection);

                    //_UnselectFilter = XMLHelper.LoadAttribute(Reader, "UnselectFilter", _UnselectFilter);
                    string UnselectManualString = XMLHelper.LoadAttribute(Reader, "UnselectManual", "null");
                    if (UnselectManualString != "null")
                        _UnselectManual = bool.Parse(UnselectManualString);
                    else
                        _UnselectManual = null;
                    CTFResolutionEstimate = XMLHelper.LoadAttribute(Reader, "CTFResolutionEstimate", CTFResolutionEstimate);

                    #endregion

                    #region Per-tilt propertries

                    {
                        XPathNavigator Nav = Reader.SelectSingleNode("//Angles");
                        if (Nav != null)
                            Angles = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    }

                    {
                        XPathNavigator Nav = Reader.SelectSingleNode("//Dose");
                        if (Nav != null)
                            Dose = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                        else
                            Dose = new float[Angles.Length];
                    }

                    {
                        XPathNavigator Nav = Reader.SelectSingleNode("//UseTilt");
                        if (Nav != null)
                            UseTilt = Nav.InnerXml.Split('\n').Select(v => bool.Parse(v)).ToArray();
                        else
                            UseTilt = Helper.ArrayOfConstant(true, Angles.Length);
                    }

                    {
                        XPathNavigator Nav = Reader.SelectSingleNode("//AxisAngle");
                        if (Nav != null)
                            TiltAxisAngles = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                        else
                            TiltAxisAngles = new float[Angles.Length];
                    }

                    {
                        XPathNavigator Nav = Reader.SelectSingleNode("//AxisOffsetX");
                        if (Nav != null)
                            TiltAxisOffsetX = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                        else
                            TiltAxisOffsetX = new float[Angles.Length];
                    }

                    {
                        XPathNavigator Nav = Reader.SelectSingleNode("//AxisOffsetY");
                        if (Nav != null)
                            TiltAxisOffsetY = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                        else
                            TiltAxisOffsetY = new float[Angles.Length];
                    }

                    {
                        XPathNavigator Nav = Reader.SelectSingleNode("//MoviePath");
                        if (Nav != null)
                            TiltMoviePaths = Nav.InnerXml.Split('\n').Select(v => v.Replace(" ", "").Replace("\r", "").Replace("\t", "")).ToArray();
                        else
                            TiltMoviePaths = new[] { "" };
                    }

                    {
                        XPathNavigator Nav = Reader.SelectSingleNode("//FOVFraction");
                        if (Nav != null)
                            FOVFraction = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                        else
                            FOVFraction = Helper.ArrayOfConstant(1f, Angles.Length);
                    }

                    #endregion

                    #region CTF fitting-related

                    {
                        TiltPS1D.Clear();
                        List<Tuple<int, float2[]>> TempPS1D = (from XPathNavigator NavPS1D in Reader.Select("//TiltPS1D")
                                                               let ID = int.Parse(NavPS1D.GetAttribute("ID", ""))
                                                               let NewPS1D = NavPS1D.InnerXml.Split(';').Select(v =>
                                                               {
                                                                   string[] Pair = v.Split('|');
                                                                   return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                                                               }).ToArray()
                                                               select new Tuple<int, float2[]>(ID, NewPS1D)).ToList();

                        TempPS1D.Sort((a, b) => a.Item1.CompareTo(b.Item1));
                        foreach (var ps1d in TempPS1D)
                            TiltPS1D.Add(ps1d.Item2);
                    }

                    {
                        TiltSimulatedScale.Clear();
                        List<Tuple<int, Cubic1D>> TempScale = (from XPathNavigator NavSimScale in Reader.Select("//TiltSimulatedScale")
                                                               let ID = int.Parse(NavSimScale.GetAttribute("ID", ""))
                                                               let NewScale = new Cubic1D(NavSimScale.InnerXml.Split(';').Select(v =>
                                                               {
                                                                   string[] Pair = v.Split('|');
                                                                   return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                                                               }).ToArray())
                                                               select new Tuple<int, Cubic1D>(ID, NewScale)).ToList();

                        TempScale.Sort((a, b) => a.Item1.CompareTo(b.Item1));
                        foreach (var scale in TempScale)
                            TiltSimulatedScale.Add(scale.Item2);
                    }

                    {
                        XPathNavigator NavPS1D = Reader.SelectSingleNode("//PS1D");
                        if (NavPS1D != null)
                            PS1D = NavPS1D.InnerXml.Split(';').Select(v =>
                            {
                                string[] Pair = v.Split('|');
                                return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                            }).ToArray();
                    }

                    {
                        XPathNavigator NavSimScale = Reader.SelectSingleNode("//SimulatedScale");
                        if (NavSimScale != null)
                            SimulatedScale = new Cubic1D(NavSimScale.InnerXml.Split(';').Select(v =>
                            {
                                string[] Pair = v.Split('|');
                                return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                            }).ToArray());
                    }

                    XPathNavigator NavCTF = Reader.SelectSingleNode("//CTF");
                    if (NavCTF != null)
                        CTF.ReadFromXML(NavCTF);

                    XPathNavigator NavOptionsCTF = Reader.SelectSingleNode("//OptionsCTF");
                    if (NavOptionsCTF != null)
                    {
                        ProcessingOptionsMovieCTF Temp = new ProcessingOptionsMovieCTF();
                        Temp.ReadFromXML(NavOptionsCTF);
                        OptionsCTF = Temp;
                    }

                    #endregion

                    #region Grids

                    XPathNavigator NavGridCTF = Reader.SelectSingleNode("//GridCTF");
                    if (NavGridCTF != null)
                        GridCTFDefocus = CubicGrid.Load(NavGridCTF);

                    XPathNavigator NavGridCTFDefocusDelta = Reader.SelectSingleNode("//GridCTFDefocusDelta");
                    if (NavGridCTFDefocusDelta != null)
                        GridCTFDefocusDelta = CubicGrid.Load(NavGridCTFDefocusDelta);

                    XPathNavigator NavGridCTFDefocusAngle = Reader.SelectSingleNode("//GridCTFDefocusAngle");
                    if (NavGridCTFDefocusAngle != null)
                        GridCTFDefocusAngle = CubicGrid.Load(NavGridCTFDefocusAngle);

                    XPathNavigator NavGridCTFPhase = Reader.SelectSingleNode("//GridCTFPhase");
                    if (NavGridCTFPhase != null)
                        GridCTFPhase = CubicGrid.Load(NavGridCTFPhase);

                    XPathNavigator NavMoveX = Reader.SelectSingleNode("//GridMovementX");
                    if (NavMoveX != null)
                        GridMovementX = CubicGrid.Load(NavMoveX);

                    XPathNavigator NavMoveY = Reader.SelectSingleNode("//GridMovementY");
                    if (NavMoveY != null)
                        GridMovementY = CubicGrid.Load(NavMoveY);

                    XPathNavigator NavVolumeWarpX = Reader.SelectSingleNode("//GridVolumeWarpX");
                    if (NavVolumeWarpX != null)
                        GridVolumeWarpX = LinearGrid4D.Load(NavVolumeWarpX);

                    XPathNavigator NavVolumeWarpY = Reader.SelectSingleNode("//GridVolumeWarpY");
                    if (NavVolumeWarpY != null)
                        GridVolumeWarpY = LinearGrid4D.Load(NavVolumeWarpY);

                    XPathNavigator NavVolumeWarpZ = Reader.SelectSingleNode("//GridVolumeWarpZ");
                    if (NavVolumeWarpZ != null)
                        GridVolumeWarpZ = LinearGrid4D.Load(NavVolumeWarpZ);

                    XPathNavigator NavAngleX = Reader.SelectSingleNode("//GridAngleX");
                    if (NavAngleX != null)
                        GridAngleX = CubicGrid.Load(NavAngleX);

                    XPathNavigator NavAngleY = Reader.SelectSingleNode("//GridAngleY");
                    if (NavAngleY != null)
                        GridAngleY = CubicGrid.Load(NavAngleY);

                    XPathNavigator NavAngleZ = Reader.SelectSingleNode("//GridAngleZ");
                    if (NavAngleZ != null)
                        GridAngleZ = CubicGrid.Load(NavAngleZ);

                    XPathNavigator NavDoseBfacs = Reader.SelectSingleNode("//GridDoseBfacs");
                    if (NavDoseBfacs != null)
                        GridDoseBfacs = CubicGrid.Load(NavDoseBfacs);

                    XPathNavigator NavDoseBfacsDelta = Reader.SelectSingleNode("//GridDoseBfacsDelta");
                    if (NavDoseBfacsDelta != null)
                        GridDoseBfacsDelta = CubicGrid.Load(NavDoseBfacsDelta);

                    XPathNavigator NavDoseBfacsAngle = Reader.SelectSingleNode("//GridDoseBfacsAngle");
                    if (NavDoseBfacsAngle != null)
                        GridDoseBfacsAngle = CubicGrid.Load(NavDoseBfacsAngle);

                    XPathNavigator NavDoseWeights = Reader.SelectSingleNode("//GridDoseWeights");
                    if (NavDoseWeights != null)
                        GridDoseWeights = CubicGrid.Load(NavDoseWeights);

                    XPathNavigator NavLocationBfacs = Reader.SelectSingleNode("//GridLocationBfacs");
                    if (NavLocationBfacs != null)
                        GridLocationBfacs = CubicGrid.Load(NavLocationBfacs);

                    XPathNavigator NavLocationWeights = Reader.SelectSingleNode("//GridLocationWeights");
                    if (NavLocationWeights != null)
                        GridLocationWeights = CubicGrid.Load(NavLocationWeights);

                    #endregion
                }
            }
            catch
            {
                return;
            }
        }

        public override void SaveMeta()
        {
            Directory.CreateDirectory(ProcessingDirectoryName);

            using (XmlTextWriter Writer = new XmlTextWriter(XMLPath, Encoding.UTF8))
            {
                Writer.Formatting = Formatting.Indented;
                Writer.IndentChar = '\t';
                Writer.Indentation = 1;
                Writer.WriteStartDocument();
                Writer.WriteStartElement("TiltSeries");

                #region Attributes

                Writer.WriteAttributeString("DataDirectory", DataDirectoryName ?? "");

                Writer.WriteAttributeString("AreAnglesInverted", AreAnglesInverted.ToString());
                Writer.WriteAttributeString("PlaneNormal", PlaneNormal.ToString());

                Writer.WriteAttributeString("Bfactor", GlobalBfactor.ToString(CultureInfo.InvariantCulture));
                Writer.WriteAttributeString("Weight", GlobalWeight.ToString(CultureInfo.InvariantCulture));

                Writer.WriteAttributeString("MagnificationCorrection", MagnificationCorrection.ToString());

                Writer.WriteAttributeString("UnselectFilter", UnselectFilter.ToString());
                Writer.WriteAttributeString("UnselectManual", UnselectManual.ToString());
                Writer.WriteAttributeString("CTFResolutionEstimate", CTFResolutionEstimate.ToString(CultureInfo.InvariantCulture));

                #endregion

                #region Per-tilt propertries

                Writer.WriteStartElement("Angles");
                Writer.WriteString(string.Join("\n", Angles.Select(v => v.ToString(CultureInfo.InvariantCulture))));
                Writer.WriteEndElement();

                Writer.WriteStartElement("Dose");
                Writer.WriteString(string.Join("\n", Dose.Select(v => v.ToString(CultureInfo.InvariantCulture))));
                Writer.WriteEndElement();

                Writer.WriteStartElement("UseTilt");
                Writer.WriteString(string.Join("\n", UseTilt.Select(v => v.ToString())));
                Writer.WriteEndElement();

                Writer.WriteStartElement("AxisAngle");
                Writer.WriteString(string.Join("\n", TiltAxisAngles.Select(v => v.ToString())));
                Writer.WriteEndElement();

                Writer.WriteStartElement("AxisOffsetX");
                Writer.WriteString(string.Join("\n", TiltAxisOffsetX.Select(v => v.ToString())));
                Writer.WriteEndElement();

                Writer.WriteStartElement("AxisOffsetY");
                Writer.WriteString(string.Join("\n", TiltAxisOffsetY.Select(v => v.ToString())));
                Writer.WriteEndElement();

                Writer.WriteStartElement("MoviePath");
                Writer.WriteString(string.Join("\n", TiltMoviePaths.Select(v => v.ToString())));
                Writer.WriteEndElement();

                Writer.WriteStartElement("FOVFraction");
                Writer.WriteString(string.Join("\n", FOVFraction.Select(v => v.ToString())));
                Writer.WriteEndElement();

                #endregion

                #region CTF fitting-related

                foreach (float2[] ps1d in TiltPS1D)
                {
                    Writer.WriteStartElement("TiltPS1D");
                    XMLHelper.WriteAttribute(Writer, "ID", TiltPS1D.IndexOf(ps1d));
                    Writer.WriteString(string.Join(";", ps1d.Select(v => v.X.ToString(CultureInfo.InvariantCulture) + "|" + v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                foreach (Cubic1D simulatedScale in TiltSimulatedScale)
                {
                    Writer.WriteStartElement("TiltSimulatedScale");
                    XMLHelper.WriteAttribute(Writer, "ID", TiltSimulatedScale.IndexOf(simulatedScale));
                    Writer.WriteString(string.Join(";",
                                                   simulatedScale.Data.Select(v => v.X.ToString(CultureInfo.InvariantCulture) +
                                                                                   "|" +
                                                                                   v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (PS1D != null)
                {
                    Writer.WriteStartElement("PS1D");
                    Writer.WriteString(string.Join(";", PS1D.Select(v => v.X.ToString(CultureInfo.InvariantCulture) + "|" + v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (SimulatedScale != null)
                {
                    Writer.WriteStartElement("SimulatedScale");
                    Writer.WriteString(string.Join(";",
                                                   SimulatedScale.Data.Select(v => v.X.ToString(CultureInfo.InvariantCulture) +
                                                                                    "|" +
                                                                                    v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (OptionsCTF != null)
                {
                    Writer.WriteStartElement("OptionsCTF");
                    OptionsCTF.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                Writer.WriteStartElement("CTF");
                CTF.WriteToXML(Writer);
                Writer.WriteEndElement();

                #endregion

                #region Grids

                Writer.WriteStartElement("GridCTF");
                GridCTFDefocus.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFDefocusDelta");
                GridCTFDefocusDelta.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFDefocusAngle");
                GridCTFDefocusAngle.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFPhase");
                GridCTFPhase.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridMovementX");
                GridMovementX.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridMovementY");
                GridMovementY.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridVolumeWarpX");
                GridVolumeWarpX.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridVolumeWarpY");
                GridVolumeWarpY.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridVolumeWarpZ");
                GridVolumeWarpZ.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridAngleX");
                GridAngleX.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridAngleY");
                GridAngleY.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridAngleZ");
                GridAngleZ.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridDoseBfacs");
                GridDoseBfacs.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridDoseBfacsDelta");
                GridDoseBfacsDelta.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridDoseBfacsAngle");
                GridDoseBfacsAngle.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridDoseWeights");
                GridDoseWeights.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridLocationBfacs");
                GridLocationBfacs.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridLocationWeights");
                GridLocationWeights.Save(Writer);
                Writer.WriteEndElement();

                #endregion

                Writer.WriteEndElement();
                Writer.WriteEndDocument();
            }
        }

        public override JsonNode ToMiniJson(string particleSuffix = null)
        {
            JsonNode Json = new JsonObject();

            // Path relative to processing folder, i.e. just the file name
            // Full path to data (including if it's in a nested folder) is stored in XML metadata
            Json["Path"] = Helper.PathToNameWithExtension(Path);

            // ProcessingStatus enum
            Json["ProcessingStatus"] = (int)ProcessingStatus;

            // Tilts
            Json["Tilts"] = JsonSerializer.SerializeToNode(TiltMoviePaths.Where((p, t) => UseTilt[t]).ToArray());

            // Angles
            Json["MinTilt"] = Angles.Min();
            Json["MaxTilt"] = Angles.Max();
            
            Json["MinAxis"] = TiltAxisAngles.Min();
            Json["MeanAxis"] = TiltAxisAngles.Mean();
            Json["MaxAxis"] = TiltAxisAngles.Max();
            
            // Shifts
            Json["MinShiftX"] = TiltAxisOffsetX.Select(Math.Abs).Min();
            Json["MeanShiftX"] = TiltAxisOffsetX.Select(Math.Abs).Average();
            Json["MaxShiftX"] = TiltAxisOffsetX.Select(Math.Abs).Max();
            Json["MinShiftY"] = TiltAxisOffsetY.Select(Math.Abs).Min();
            Json["MeanShiftY"] = TiltAxisOffsetY.Select(Math.Abs).Average();
            Json["MaxShiftY"] = TiltAxisOffsetY.Select(Math.Abs).Max();

            // CTF
            if (OptionsCTF != null)
            {
                if (GridCTFDefocus != null)
                {
                    var defoci = Enumerable.Range(0, NTilts).Select(i => GetTiltDefocus(i)).ToArray();
                    Json["MinDefocus"] = defoci.Min();
                    Json["MeanDefocus"] = defoci.Mean();
                    Json["MaxDefocus"] = defoci.Max();
                }

                Json["Astigmatism"] = Math.Abs(CTF.DefocusDelta);
                
                Json["MinPhase"] = GridCTFPhase.Values.Min();
                Json["MeanPhase"] = GridCTFPhase.Values.Mean();
                Json["MaxPhase"] = GridCTFPhase.Values.Max();
                
                Json["CtfResolution"] = CTFResolutionEstimate <= 0 ? null : MathF.Round((float)CTFResolutionEstimate, 2);

                Json["CtfInclination"] = MathF.Acos(PlaneNormal.Z) * Helper.ToDeg;
            }

            // Particle count for given suffix
            if (particleSuffix != null)
            {
                int ParticleCount = GetParticleCount(particleSuffix);
                Json["Particles"] = ParticleCount < 0 ? null : ParticleCount;
            }

            return Json;
        }

        #endregion

        #region Hashes

        public override string GetDataHash()
        {
            Movie First = new Movie(IOPath.Combine(DataOrProcessingDirectoryName, TiltMoviePaths[0]));
            FileInfo Info = new FileInfo(First.DataPath);

            byte[] DataBytes = new byte[Math.Min(1 << 19, Info.Length)];
            using (BinaryReader Reader = new BinaryReader(File.OpenRead(First.DataPath)))
            {
                Reader.Read(DataBytes, 0, DataBytes.Length);
            }

            DataBytes = Helper.Combine(Helper.ToBytes(RootName.ToCharArray()), DataBytes);

            return MathHelper.GetSHA1(DataBytes);
        }

        public override string GetProcessingHash()
        {
            List<byte[]> Arrays = new List<byte[]>();

            if (CTF != null)
            {
                Arrays.Add(Helper.ToBytes(new[]
                {
                    CTF.Amplitude,
                    CTF.Bfactor,
                    CTF.Cc,
                    CTF.Cs,
                    CTF.Defocus,
                    CTF.DefocusAngle,
                    CTF.DefocusDelta,
                    CTF.PhaseShift,
                    CTF.PixelSize,
                    CTF.PixelSizeAngle,
                    CTF.PixelSizeDeltaPercent,
                    CTF.Scale,
                    CTF.Voltage
                }));
                if (CTF.ZernikeCoeffsEven != null)
                    Arrays.Add(Helper.ToBytes(CTF.ZernikeCoeffsEven));
                if (CTF.ZernikeCoeffsOdd != null)
                    Arrays.Add(Helper.ToBytes(CTF.ZernikeCoeffsOdd));
            }
            #region Grids

            if (GridCTFDefocus != null)
            {
                Arrays.Add(GridCTFDefocus.Dimensions);
                Arrays.Add(Helper.ToBytes(GridCTFDefocus.FlatValues));
            }

            if (GridCTFPhase != null)
            {
                Arrays.Add(GridCTFPhase.Dimensions);
                Arrays.Add(Helper.ToBytes(GridCTFPhase.FlatValues));
            }

            if (GridMovementX != null)
            {
                Arrays.Add(GridMovementX.Dimensions);
                Arrays.Add(Helper.ToBytes(GridMovementX.FlatValues));
            }

            if (GridMovementY != null)
            {
                Arrays.Add(GridMovementY.Dimensions);
                Arrays.Add(Helper.ToBytes(GridMovementY.FlatValues));
            }

            if (GridVolumeWarpX != null)
            {
                Arrays.Add(GridVolumeWarpX.Dimensions);
                Arrays.Add(Helper.ToBytes(GridVolumeWarpX.Values));
            }

            if (GridVolumeWarpY != null)
            {
                Arrays.Add(GridVolumeWarpY.Dimensions);
                Arrays.Add(Helper.ToBytes(GridVolumeWarpY.Values));
            }

            if (GridVolumeWarpZ != null)
            {
                Arrays.Add(GridVolumeWarpZ.Dimensions);
                Arrays.Add(Helper.ToBytes(GridVolumeWarpZ.Values));
            }

            if (GridAngleX != null)
            {
                Arrays.Add(GridAngleX.Dimensions);
                Arrays.Add(Helper.ToBytes(GridAngleX.FlatValues));
            }

            if (GridAngleY != null)
            {
                Arrays.Add(GridAngleY.Dimensions);
                Arrays.Add(Helper.ToBytes(GridAngleY.FlatValues));
            }

            if (GridAngleZ != null)
            {
                Arrays.Add(GridAngleZ.Dimensions);
                Arrays.Add(Helper.ToBytes(GridAngleZ.FlatValues));
            }

            if (GridCTFDefocusAngle != null)
            {
                Arrays.Add(GridCTFDefocusAngle.Dimensions);
                Arrays.Add(Helper.ToBytes(GridCTFDefocusAngle.FlatValues));
            }

            if (GridCTFDefocusDelta != null)
            {
                Arrays.Add(GridCTFDefocusDelta.Dimensions);
                Arrays.Add(Helper.ToBytes(GridCTFDefocusDelta.FlatValues));
            }

            if (GridDoseBfacs != null)
            {
                Arrays.Add(GridDoseBfacs.Dimensions);
                Arrays.Add(Helper.ToBytes(GridDoseBfacs.FlatValues));
            }

            #endregion

            Arrays.Add(Helper.ToBytes(TiltAxisAngles));
            Arrays.Add(Helper.ToBytes(TiltAxisOffsetX));
            Arrays.Add(Helper.ToBytes(TiltAxisOffsetX));
            Arrays.Add(Helper.ToBytes(Angles));
            Arrays.Add(Helper.ToBytes(Dose));
            Arrays.Add(Helper.ToBytes(UseTilt));

            foreach (var moviePath in TiltMoviePaths)
            {
                Movie TiltMovie = new Movie(IOPath.Combine(DataOrProcessingDirectoryName, moviePath));
                Arrays.Add(Helper.ToBytes(TiltMovie.GetProcessingHash().ToCharArray()));
            }

            byte[] ArraysCombined = Arrays.SelectMany(a => a).ToArray();
            return MathHelper.GetSHA1(ArraysCombined);
        }

        #endregion
    }

    [Serializable]
    public abstract class TomoProcessingOptionsBase : ProcessingOptionsBase
    {
        public float3 DimensionsPhysical => Dimensions * (float)PixelSizeMean;

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((TomoProcessingOptionsBase)obj);
        }

        protected bool Equals(TomoProcessingOptionsBase other)
        {
            return base.Equals(other) &&
                   Dimensions == other.Dimensions;
        }

        public static bool operator ==(TomoProcessingOptionsBase left, TomoProcessingOptionsBase right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(TomoProcessingOptionsBase left, TomoProcessingOptionsBase right)
        {
            return !Equals(left, right);
        }
    }



    [Serializable]
    public class ProcessingOptionsTomoAretomo : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public int AlignZ { get; set; }

        [WarpSerializable]
        public decimal AxisAngle { get; set; }

        [WarpSerializable]
        public bool DoAxisSearch { get; set; }

        [WarpSerializable]
        public string Executable { get; set; }
        
        [WarpSerializable]
        public int[] NPatchesXY { get; set; }
    }
    
    [Serializable]
    public class ProcessingOptionsTomoEtomoPatch : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public decimal AxisAngle { get; set; }
        
        [WarpSerializable]
        public bool DoPatchTracking { get; set; }
        
        [WarpSerializable]
        public bool DoTiltAlign { get; set; }
        
        [WarpSerializable]
        public bool DoAxisAngleSearch { get; set; }
        
        [WarpSerializable]
        public decimal TiltStackAngPix { get; set; }
        
        [WarpSerializable]
        public decimal PatchSizeAngstroms { get; set; }
    }
    
    [Serializable]
    public class ProcessingOptionsTomoEtomoFiducials : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public decimal AxisAngle { get; set; }
        
        [WarpSerializable]
        public bool DoFiducialTracking { get; set; }
        
        [WarpSerializable]
        public decimal FiducialSizeNanometers { get; set; }
        
        [WarpSerializable]
        public bool DoTiltAlign { get; set; }
        
        [WarpSerializable]
        public bool DoAxisAngleSearch { get; set; }
        
        [WarpSerializable]
        public decimal TiltStackAngPix { get; set; }
        
        [WarpSerializable]
        public decimal TargetNBeads { get; set; }
    }
}
