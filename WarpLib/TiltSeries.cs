using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
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
using IOPath = System.IO.Path;

namespace Warp
{
    public class TiltSeries : Movie
    {
        #region Directories

        public string TiltStackDir => IOPath.Combine(ProcessingDirectoryName, "tiltstack", RootName);

        public string TiltStackPath => IOPath.Combine(TiltStackDir, RootName + ".st");

        public string AngleFilePath => IOPath.Combine(TiltStackDir, RootName + ".rawtlt");

        public string ReconstructionDir => IOPath.Combine(ProcessingDirectoryName, "reconstruction");

        public string ReconstructionDeconvDir => IOPath.Combine(ReconstructionDir, "deconv");

        public string ReconstructionOddDir => IOPath.Combine(ReconstructionDir, "odd");

        public string ReconstructionEvenDir => IOPath.Combine(ReconstructionDir, "even");

        public string ReconstructionCTFDir => IOPath.Combine(ReconstructionDir, "ctf");

        public string SubtomoDir => IOPath.Combine(ProcessingDirectoryName, "subtomo", RootName);

        public string ParticleSeriesDir => IOPath.Combine(ProcessingDirectoryName, "particleseries", RootName);

        public string WeightOptimizationDir => IOPath.Combine(ProcessingDirectoryName, "weightoptimization");

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
                throw new Exception("STAR file has no wrpDose or wrpTilt column.");

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

        #region Processing tasks

        #region CTF fitting

        public void ProcessCTFSimultaneous(ProcessingOptionsMovieCTF options)
        {
            IsProcessing = true;

            if (!Directory.Exists(PowerSpectrumDir))
                Directory.CreateDirectory(PowerSpectrumDir);

            int2 DimsFrame;
            {
                Movie FirstMovie = new Movie(IOPath.Combine(DataOrProcessingDirectoryName, TiltMoviePaths[0]));
                MapHeader HeaderMovie = MapHeader.ReadFromFile(FirstMovie.DataPath);
                DimsFrame = new int2(new float2(HeaderMovie.Dimensions.X, HeaderMovie.Dimensions.Y) / (float)options.DownsampleFactor + 1) / 2 * 2;
            }

            #region Dimensions and grids

            int NFrames = NTilts;
            int2 DimsImage = DimsFrame;
            int2 DimsRegionBig = new int2(1536);
            int2 DimsRegion = new int2(options.Window, options.Window);

            float OverlapFraction = 0.5f;
            int2 DimsPositionGrid;
            int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, new int2(DimsRegionBig.X, DimsRegionBig.Y), OverlapFraction, out DimsPositionGrid);
            float3[] PositionGridPhysical = PositionGrid.Select(v => new float3(v.X + DimsRegionBig.X / 2 - DimsImage.X / 2,
                                                                                v.Y + DimsRegionBig.Y / 2 - DimsImage.Y / 2,
                                                                                0) *
                                                                                (float)options.BinnedPixelSizeMean * 1e-4f).ToArray();
            int NPositions = (int)DimsPositionGrid.Elements();

            int3 CTFSpectraGrid = new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames);

            int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
            int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
            int NFreq = MaxFreqExclusive - MinFreqInclusive;

            #endregion

            #region Allocate memory

            // GPU
            Image CTFSpectra = new Image(IntPtr.Zero, new int3(DimsRegion.X, DimsRegion.X, (int)CTFSpectraGrid.Elements()), true);
            Image CTFMean = new Image(new int3(DimsRegion), true);
            Image CTFCoordsCart = new Image(new int3(DimsRegion), true, true);
            Image CTFCoordsPolarTrimmed = new Image(new int3(NFreq, DimsRegion.X, 1), false, true);

            // CPU
            float2[] GlobalPS1D = null;
            float[][] LocalPS1D = new float[NPositions * NFrames][];
            Cubic1D GlobalBackground = null, GlobalScale = null;
            CTF GlobalCTF = null;
            float2 GlobalPlaneAngle = new float2();

            #endregion

            #region Helper methods

            Func<float[], float[]> GetDefocusGrid = (defoci) =>
            {
                float[] Result = new float[NPositions * NFrames];

                for (int t = 0; t < NFrames; t++)
                {
                    float3 Normal = (Matrix3.RotateX(GlobalPlaneAngle.X * Helper.ToRad) * Matrix3.RotateY(GlobalPlaneAngle.Y * Helper.ToRad)) * new float3(0, 0, 1);
                    Normal = Matrix3.Euler(0, Angles[t] * (AreAnglesInverted ? -1 : 1) * Helper.ToRad, 0) * Normal;
                    Normal = Matrix3.Euler(0, 0, -TiltAxisAngles[t] * Helper.ToRad) * Normal;
                    for (int i = 0; i < NPositions; i++)
                        Result[t * NPositions + i] = defoci[t] - float3.Dot(Normal, PositionGridPhysical[i]) / Normal.Z;
                }

                return Result;
            };

            #region Background fitting methods

            Action UpdateBackgroundFit = () =>
            {
                float2[] ForPS1D = GlobalPS1D.Skip(Math.Max(5, MinFreqInclusive / 1)).ToArray();
                Cubic1D.FitCTF(ForPS1D,
                               GlobalCTF.Get1D(GlobalPS1D.Length, true, true).Skip(Math.Max(5, MinFreqInclusive / 1)).ToArray(),
                               GlobalCTF.GetZeros(),
                               GlobalCTF.GetPeaks(),
                               out GlobalBackground,
                               out GlobalScale);
            };

            Action<bool> UpdateRotationalAverage = keepbackground =>
            {
                float[] MeanData = CTFMean.GetHost(Intent.Read)[0];

                Image CTFMeanCorrected = new Image(new int3(DimsRegion), true);
                float[] MeanCorrectedData = CTFMeanCorrected.GetHost(Intent.Write)[0];

                // Subtract current background estimate from spectra, populate coords.
                Helper.ForEachElementFT(DimsRegion,
                                        (x, y, xx, yy, r, a) =>
                                        {
                                            int i = y * (DimsRegion.X / 2 + 1) + x;
                                            MeanCorrectedData[i] = MeanData[i] - GlobalBackground.Interp(r / DimsRegion.X);
                                        });

                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                GPU.CTFMakeAverage(CTFMeanCorrected.GetDevice(Intent.Read),
                                   CTFCoordsCart.GetDevice(Intent.Read),
                                   (uint)CTFMeanCorrected.DimsEffective.ElementsSlice(),
                                   (uint)DimsRegion.X,
                                   new[] { GlobalCTF.ToStruct() },
                                   GlobalCTF.ToStruct(),
                                   0,
                                   (uint)DimsRegion.X / 2,
                                   1,
                                   CTFAverage1D.GetDevice(Intent.Write));

                //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

                float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                float2[] ForPS1D = new float2[GlobalPS1D.Length];
                if (keepbackground)
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i] + GlobalBackground.Interp((float)i / DimsRegion.X));
                else
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                MathHelper.UnNaN(ForPS1D);

                GlobalPS1D = ForPS1D;

                CTFMeanCorrected.Dispose();
                CTFAverage1D.Dispose();
            };

            #endregion

            #endregion

            // Extract movie regions, create individual spectra in Cartesian coordinates and their mean.

            #region Create spectra

            int PlanForw = GPU.CreateFFTPlan(new int3(DimsRegionBig), (uint)NPositions);
            int PlanBack = GPU.CreateIFFTPlan(new int3(DimsRegion), (uint)NPositions);

            Movie[] TiltMovies;
            Image[] TiltMovieData;
            LoadMovieData(options, out TiltMovies, out TiltMovieData, false, out _, out _);

            for (int t = 0; t < NTilts; t++)
            {
                Image TiltMovieAverage = TiltMovieData[t];

                GPU.Normalize(TiltMovieAverage.GetDevice(Intent.Read),
                              TiltMovieAverage.GetDevice(Intent.Write),
                              (uint)TiltMovieAverage.ElementsReal,
                              1);

                Image MovieCTFMean = new Image(new int3(DimsRegion), true);

                GPU.CreateSpectra(TiltMovieAverage.GetDevice(Intent.Read),
                                  DimsImage,
                                  TiltMovieAverage.Dims.Z,
                                  PositionGrid,
                                  NPositions,
                                  DimsRegionBig,
                                  CTFSpectraGrid.Slice(),
                                  DimsRegion,
                                  CTFSpectra.GetDeviceSlice(t * (int)CTFSpectraGrid.ElementsSlice(), Intent.Write),
                                  MovieCTFMean.GetDevice(Intent.Write),
                                  PlanForw,
                                  PlanBack);

                CTFMean.Add(MovieCTFMean);

                MovieCTFMean.Dispose();
                TiltMovieAverage.FreeDevice();
            }

            GPU.DestroyFFTPlan(PlanBack);
            GPU.DestroyFFTPlan(PlanForw);

            CTFMean.Multiply(1f / NTilts);

            #endregion

            // Populate address arrays for later.

            #region Init addresses

            {
                float2[] CoordsData = new float2[CTFCoordsCart.ElementsSliceComplex];

                Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) => CoordsData[y * (DimsRegion.X / 2 + 1) + x] = new float2(r, a));
                CTFCoordsCart.UpdateHostWithComplex(new[] { CoordsData });

                CoordsData = new float2[NFreq * DimsRegion.X];
                Helper.ForEachElement(CTFCoordsPolarTrimmed.DimsSlice, (x, y) =>
                {
                    float Angle = (float)y / DimsRegion.X * (float)Math.PI;
                    float Ny = 1f / DimsRegion.X;
                    CoordsData[y * NFreq + x] = new float2((x + MinFreqInclusive) * Ny, Angle);
                });
                CTFCoordsPolarTrimmed.UpdateHostWithComplex(new[] { CoordsData });
            }

            #endregion

            #region Initial 1D spectra

            // Mean spectrum to fit background
            {
                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                GPU.CTFMakeAverage(CTFMean.GetDevice(Intent.Read),
                                   CTFCoordsCart.GetDevice(Intent.Read),
                                   (uint)CTFMean.ElementsSliceReal,
                                   (uint)DimsRegion.X,
                                   new[] { new CTF().ToStruct() },
                                   new CTF().ToStruct(),
                                   0,
                                   (uint)DimsRegion.X / 2,
                                   1,
                                   CTFAverage1D.GetDevice(Intent.Write));

                //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

                float[] CTFAverage1DData = CTFAverage1D.GetHost(Intent.Read)[0];
                float2[] ForPS1D = new float2[DimsRegion.X / 2];
                for (int i = 0; i < ForPS1D.Length; i++)
                    ForPS1D[i] = new float2((float)i / DimsRegion.X, (float)Math.Round(CTFAverage1DData[i], 4));
                GlobalPS1D = ForPS1D;

                CTFAverage1D.Dispose();
            }

            // Individual 1D spectra for initial grid search below
            {
                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                for (int s = 0; s < NPositions * NFrames; s++)
                {
                    GPU.CTFMakeAverage(CTFSpectra.GetDeviceSlice(s, Intent.Read),
                                       CTFCoordsCart.GetDevice(Intent.Read),
                                       (uint)CTFMean.ElementsSliceReal,
                                       (uint)DimsRegion.X,
                                       new[] { new CTF().ToStruct() },
                                       new CTF().ToStruct(),
                                       0,
                                       (uint)DimsRegion.X / 2,
                                       1,
                                       CTFAverage1D.GetDevice(Intent.Write));

                    //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

                    LocalPS1D[s] = CTFAverage1D.GetHostContinuousCopy();
                }

                CTFAverage1D.Dispose();
            }

            #endregion

            #region Do initial fit on mean 1D PS
            {
                float2[] ForPS1D = GlobalPS1D.Skip(MinFreqInclusive).Take(Math.Max(2, NFreq * 2 / 3)).ToArray();

                float[] CurrentBackground;

                // Get a very rough background spline fit with 3-5 nodes
                int NumNodes = Math.Max(3, (int)((options.RangeMax - options.RangeMin) * 5M * 2 / 3));
                GlobalBackground = Cubic1D.Fit(ForPS1D, NumNodes);

                CurrentBackground = GlobalBackground.Interp(ForPS1D.Select(p => p.X).ToArray());
                float[][] SubtractedLocal1D = new float[LocalPS1D.Length][];
                for (int s = 0; s < LocalPS1D.Length; s++)
                {
                    SubtractedLocal1D[s] = new float[NFreq * 2 / 3];
                    for (int f = 0; f < NFreq * 2 / 3; f++)
                        SubtractedLocal1D[s][f] = LocalPS1D[s][f + MinFreqInclusive] - CurrentBackground[f];
                }

                float[] GridDeltas = GetDefocusGrid(Helper.ArrayOfConstant(0f, NFrames));

                float ZMin = (float)options.ZMin;
                float ZMax = (float)options.ZMax;
                float PhaseMin = 0f;
                float PhaseMax = options.DoPhase ? 1f : 0f;

                float ZStep = Math.Max(0.01f, (ZMax - ZMin) / 200f);

                float BestZ = 0, BestPhase = 0, BestScore = -999;
                Parallel.For(0, (int)((ZMax - ZMin + ZStep - 1e-6f) / ZStep), zi =>
                {
                    float z = ZMin + zi * ZStep;

                    for (float p = PhaseMin; p <= PhaseMax; p += 0.01f)
                    {
                        float Score = 0;

                        for (int s = 0; s < NPositions * NFrames; s++)
                        {
                            CTF CurrentParams = new CTF
                            {
                                PixelSize = options.BinnedPixelSizeMean,

                                Defocus = (decimal)(z + GridDeltas[s]),
                                PhaseShift = (decimal)p,

                                Cs = options.Cs,
                                Voltage = options.Voltage,
                                Amplitude = options.Amplitude
                            };
                            float[] SimulatedCTF = CurrentParams.Get1D(GlobalPS1D.Length, true).Skip(MinFreqInclusive).Take(Math.Max(2, NFreq * 2 / 3)).ToArray();
                            MathHelper.NormalizeInPlace(SimulatedCTF);

                            Score += MathHelper.CrossCorrelate(SubtractedLocal1D[s], SimulatedCTF);
                        }

                        lock (ForPS1D)
                            if (Score > BestScore)
                            {
                                BestScore = Score;
                                BestZ = z;
                                BestPhase = p;
                            }
                    }
                });

                GlobalCTF = new CTF
                {
                    PixelSize = options.BinnedPixelSizeMean,

                    Defocus = (decimal)BestZ,
                    PhaseShift = (decimal)BestPhase,

                    Cs = options.Cs,
                    Voltage = options.Voltage,
                    Amplitude = options.Amplitude
                };

                //UpdateRotationalAverage(true);  // This doesn't have a nice background yet.

                // Scale everything to one common defocus value
                {
                    CTFStruct[] LocalParams = GridDeltas.Select((v, i) =>
                    {
                        CTF Local = GlobalCTF.GetCopy();
                        Local.Defocus += (decimal)v;
                        Local.Scale = (decimal)Math.Pow(1 - Math.Abs(Math.Sin(Angles[i / NPositions] * Helper.ToRad)), 2);

                        return Local.ToStruct();
                    }).ToArray();

                    Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                    CTF CTFAug = GlobalCTF.GetCopy();

                    GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                       CTFCoordsCart.GetDevice(Intent.Read),
                                       (uint)CTFSpectra.ElementsSliceReal,
                                       (uint)DimsRegion.X,
                                       LocalParams,
                                       CTFAug.ToStruct(),
                                       0,
                                       (uint)DimsRegion.X / 2,
                                       (uint)LocalParams.Length,
                                       CTFAverage1D.GetDevice(Intent.Write));

                    float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                    for (int i = 0; i < RotationalAverageData.Length; i++)
                        GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                    MathHelper.UnNaN(GlobalPS1D);

                    CTFAverage1D.Dispose();
                    CTFSpectra.FreeDevice();
                }

                UpdateBackgroundFit();          // Now get a reasonably nice background.

                #region For debug purposes, check what the background-subtracted average looks like at this point

                // Scale everything to one common defocus value
                if (false)
                {
                    Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                    float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                    // Construct background in Cartesian coordinates.
                    Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) =>
                    {
                        CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = GlobalBackground.Interp(r / DimsRegion.X);
                    });

                    CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                    CTFStruct[] LocalParams = GridDeltas.Select(v =>
                    {
                        CTF Local = GlobalCTF.GetCopy();
                        Local.Defocus += (decimal)v;

                        return Local.ToStruct();
                    }).ToArray();

                    Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                    CTF CTFAug = GlobalCTF.GetCopy();

                    GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                       CTFCoordsCart.GetDevice(Intent.Read),
                                       (uint)CTFSpectra.ElementsSliceReal,
                                       (uint)DimsRegion.X,
                                       LocalParams,
                                       CTFAug.ToStruct(),
                                       0,
                                       (uint)DimsRegion.X / 2,
                                       (uint)LocalParams.Length,
                                       CTFAverage1D.GetDevice(Intent.Write));

                    float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                    for (int i = 0; i < RotationalAverageData.Length; i++)
                        GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                    MathHelper.UnNaN(GlobalPS1D);

                    CTFSpectra.AddToSlices(CTFSpectraBackground);

                    CTFSpectraBackground.Dispose();
                    CTFAverage1D.Dispose();
                    CTFSpectra.FreeDevice();
                }

                #endregion
            }
            #endregion

            // Do BFGS optimization of defocus, astigmatism and phase shift,
            // using 2D simulation for comparison

            double[] DefociFromMovies = TiltMoviePaths.Select(p => (double)(new Movie(System.IO.Path.Combine(DataOrProcessingDirectoryName, p)).CTF.Defocus)).ToArray();

            double[] StartParams = new double[5];
            StartParams[0] = 0;
            StartParams[1] = 0;
            StartParams[2] = (double)GlobalCTF.DefocusDelta;
            StartParams[3] = (double)GlobalCTF.DefocusDelta;
            StartParams[4] = (double)GlobalCTF.DefocusAngle / 20 * Helper.ToRad;
            StartParams = Helper.Combine(StartParams,
                                         DefociFromMovies, //Helper.ArrayOfConstant((double)GlobalCTF.Defocus, NFrames),
                                         Helper.ArrayOfConstant((double)GlobalCTF.PhaseShift, Math.Max(1, NFrames / 3)));

            float3[] GridCoordsByAngle = Helper.ArrayOfFunction(i => new float3((float)i / (NFrames - 1), 0, 0), NFrames);
            float3[] GridCoordsByDose = Helper.ArrayOfFunction(i => new float3(Dose[i] / MathHelper.Max(Dose), 0, 0), NFrames);

            #region BFGS

            {
                // Second iteration will have a nicer background
                for (int opt = 0; opt < 1; opt++)
                {
                    if (opt > 0)
                        NFreq = Math.Min(NFreq + 10, DimsRegion.X / 2 - MinFreqInclusive - 1);

                    Image CTFSpectraPolarTrimmed = CTFSpectra.AsPolar((uint)MinFreqInclusive, (uint)(MinFreqInclusive + NFreq));
                    CTFSpectra.FreeDevice(); // This will only be needed again for the final PS1D.

                    #region Create background and scale

                    float[] CurrentScale = Helper.ArrayOfConstant(1f, GlobalPS1D.Length);// GlobalScale.Interp(GlobalPS1D.Select(p => p.X).ToArray());

                    Image CTFSpectraScale = new Image(new int3(NFreq, DimsRegion.X, 1));
                    float[] CTFSpectraScaleData = CTFSpectraScale.GetHost(Intent.Write)[0];

                    // Trim polar to relevant frequencies, and populate coordinates.
                    Parallel.For(0, DimsRegion.X, y =>
                    {
                        for (int x = 0; x < NFreq; x++)
                            CTFSpectraScaleData[y * NFreq + x] = CurrentScale[x + MinFreqInclusive];
                    });
                    //CTFSpectraScale.WriteMRC("ctfspectrascale.mrc");

                    // Background is just 1 line since we're in polar.
                    Image CurrentBackground = new Image(GlobalBackground.Interp(GlobalPS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray());

                    CTFSpectraPolarTrimmed.SubtractFromLines(CurrentBackground);
                    CurrentBackground.Dispose();

                    //CTFSpectraPolarTrimmed.WriteMRC("ctfspectrapolartrimmed.mrc");

                    #endregion

                    #region Eval and Gradient methods

                    // Helper method for getting CTFStructs for the entire spectra grid.
                    Func<double[], CTF, float[], float[], float[], CTFStruct[]> EvalGetCTF = (input, ctf, phaseValues, defocusValues, defocusDeltaValues) =>
                    {
                        CTF Local = ctf.GetCopy();
                        Local.DefocusAngle = (decimal)(input[4] * 20 / (Math.PI / 180));

                        CTFStruct LocalStruct = Local.ToStruct();
                        CTFStruct[] LocalParams = new CTFStruct[defocusValues.Length];
                        for (int f = 0; f < NFrames; f++)
                            for (int p = 0; p < NPositions; p++)
                            {
                                LocalParams[f * NPositions + p] = LocalStruct;
                                LocalParams[f * NPositions + p].Defocus = defocusValues[f * NPositions + p] * -1e-6f;
                                LocalParams[f * NPositions + p].DefocusDelta = defocusDeltaValues[f] * -1e-6f;
                                LocalParams[f * NPositions + p].PhaseShift = phaseValues[f] * (float)Math.PI;
                            }

                        return LocalParams;
                    };

                    Func<double[], double> Eval = input =>
                    {
                        GlobalPlaneAngle = new float2((float)input[0], (float)input[1]) * Helper.ToDeg;

                        CubicGrid TempGridPhase = new CubicGrid(new int3(Math.Max(1, NFrames / 3), 1, 1), input.Skip(5 + NFrames).Take(Math.Max(1, NFrames / 3)).Select(v => (float)v).ToArray());
                        CubicGrid TempGridDefocus = new CubicGrid(new int3(NFrames, 1, 1), input.Skip(5).Take(NFrames).Select(v => (float)v).ToArray());
                        CubicGrid TempGridDefocusDelta = new CubicGrid(new int3(1, 1, 1), new[] { (float)input[2] });

                        float[] PhaseValues = TempGridPhase.GetInterpolated(GridCoordsByDose);
                        float[] DefocusValues = GetDefocusGrid(TempGridDefocus.GetInterpolated(GridCoordsByAngle));
                        float[] DefocusDeltaValues = TempGridDefocusDelta.GetInterpolated(GridCoordsByDose);

                        CTFStruct[] LocalParams = EvalGetCTF(input, GlobalCTF, PhaseValues, DefocusValues, DefocusDeltaValues);

                        float[] Result = new float[LocalParams.Length];

                        GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                            CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                            CTFSpectraScale.GetDevice(Intent.Read),
                                            (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                            LocalParams,
                                            Result,
                                            (uint)LocalParams.Length);

                        float Score = Result.Sum();

                        if (float.IsNaN(Score) || float.IsInfinity(Score))
                            throw new Exception("Bad score.");

                        return Score;
                    };

                    Func<double[], double[]> Gradient = input =>
                    {
                        const float Step = 0.0025f;
                        double[] Result = new double[input.Length];

                        for (int i = 0; i < input.Length; i++)
                        {
                            if (!options.DoPhase && i >= 5 + NFrames)
                                continue;

                            double[] UpperInput = new double[input.Length];
                            input.CopyTo(UpperInput, 0);
                            UpperInput[i] += Step;
                            double UpperValue = Eval(UpperInput);

                            double[] LowerInput = new double[input.Length];
                            input.CopyTo(LowerInput, 0);
                            LowerInput[i] -= Step;
                            double LowerValue = Eval(LowerInput);

                            Result[i] = (UpperValue - LowerValue) / (2f * Step);
                        }

                        if (Result.Any(i => double.IsNaN(i) || double.IsInfinity(i)))
                            throw new Exception("Bad score.");

                        return Result;
                    };

                    #endregion

                    #region Do optimization

                    // StartParams are initialized above, before the optimization loop

                    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Gradient)
                    {
                        MaxIterations = 15
                    };
                    Optimizer.Maximize(StartParams);

                    #endregion

                    #region Retrieve parameters

                    GlobalCTF.Defocus = (decimal)MathHelper.Mean(Optimizer.Solution.Skip(5).Take(NFrames).Select(v => (float)v));
                    GlobalCTF.PhaseShift = (decimal)MathHelper.Mean(Optimizer.Solution.Skip(5 + NFrames).Take(Math.Max(1, NFrames / 3)).Select(v => (float)v));
                    GlobalCTF.DefocusDelta = (decimal)(Optimizer.Solution[2]) / 1;
                    GlobalCTF.DefocusAngle = (decimal)(Optimizer.Solution[4] * 20 * Helper.ToDeg);

                    if (GlobalCTF.DefocusDelta < 0)
                    {
                        GlobalCTF.DefocusAngle += 90;
                        GlobalCTF.DefocusDelta *= -1;
                    }
                    GlobalCTF.DefocusAngle = ((int)GlobalCTF.DefocusAngle + 180 * 99) % 180;

                    GlobalPlaneAngle = new float2((float)Optimizer.Solution[0],
                                                  (float)Optimizer.Solution[1]) * Helper.ToDeg;

                    {
                        CubicGrid TempGridPhase = new CubicGrid(new int3(Math.Max(1, NFrames / 3), 1, 1), StartParams.Skip(5 + NFrames).Take(Math.Max(1, NFrames / 3)).Select(v => (float)v).ToArray());
                        CubicGrid TempGridDefocusDelta = new CubicGrid(new int3(1, 1, 1), new[] { (float)GlobalCTF.DefocusDelta });
                        CubicGrid TempGridDefocus = new CubicGrid(new int3(NFrames, 1, 1), StartParams.Skip(5).Take(NFrames).Select(v => (float)v).ToArray());

                        GridCTFDefocus = new CubicGrid(new int3(1, 1, NTilts), TempGridDefocus.GetInterpolated(GridCoordsByAngle));
                        GridCTFDefocusDelta = new CubicGrid(new int3(1, 1, NTilts), TempGridDefocusDelta.GetInterpolated(GridCoordsByDose));
                        GridCTFDefocusAngle = new CubicGrid(new int3(1, 1, NTilts), Helper.ArrayOfConstant((float)GlobalCTF.DefocusAngle, NTilts));
                        GridCTFPhase = new CubicGrid(new int3(1, 1, NTilts), TempGridPhase.GetInterpolated(GridCoordsByDose));
                    }

                    #endregion

                    // Dispose GPU resources manually because GC can't be bothered to do it in time.
                    CTFSpectraPolarTrimmed.Dispose();
                    CTFSpectraScale.Dispose();

                    #region Get nicer envelope fit

                    // Scale everything to one common defocus value
                    {
                        float3[] GridCoords = Helper.ArrayOfFunction(i => new float3(0, 0, (float)i / (NFrames - 1)), NFrames);

                        float[] DefocusValues = GetDefocusGrid(GridCTFDefocus.GetInterpolated(GridCoords));
                        float[] DefocusDeltaValues = GridCTFDefocusDelta.GetInterpolated(GridCoords);
                        float[] DefocusAngleValues = GridCTFDefocusAngle.GetInterpolated(GridCoords);
                        float[] PhaseValues = GridCTFPhase.GetInterpolated(GridCoords);

                        CTFStruct[] LocalParams = new CTFStruct[DefocusValues.Length];
                        for (int f = 0; f < NFrames; f++)
                        {
                            for (int p = 0; p < NPositions; p++)
                            {
                                CTF Local = GlobalCTF.GetCopy();
                                Local.Defocus = (decimal)DefocusValues[f * NPositions + p];
                                Local.DefocusDelta = (decimal)DefocusDeltaValues[f];
                                Local.DefocusAngle = (decimal)DefocusAngleValues[f];
                                Local.PhaseShift = (decimal)PhaseValues[f];

                                LocalParams[f * NPositions + p] = Local.ToStruct();
                            }
                        }

                        Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                        CTF CTFAug = GlobalCTF.GetCopy();

                        GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                            CTFCoordsCart.GetDevice(Intent.Read),
                                            (uint)CTFSpectra.ElementsSliceReal,
                                            (uint)DimsRegion.X,
                                            LocalParams,
                                            CTFAug.ToStruct(),
                                            0,
                                            (uint)DimsRegion.X / 2,
                                            (uint)LocalParams.Length,
                                            CTFAverage1D.GetDevice(Intent.Write));

                        float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                        for (int i = 0; i < RotationalAverageData.Length; i++)
                            GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                        MathHelper.UnNaN(GlobalPS1D);

                        CTFAverage1D.Dispose();
                        CTFSpectra.FreeDevice();

                        UpdateBackgroundFit(); // Now get a nice background.
                    }

                    #endregion
                }
            }

            #endregion

            #region Create global, and per-tilt average spectra

            {
                TiltPS1D = new ObservableCollection<float2[]>();
                TiltSimulatedBackground = new ObservableCollection<Cubic1D>();
                TiltSimulatedScale = new ObservableCollection<Cubic1D>();
                Image AllPS2D = new Image(new int3(DimsRegion.X, DimsRegion.X / 2, NTilts));

                float3[] GridCoords = Helper.ArrayOfFunction(i => new float3(0, 0, (float)i / (NFrames - 1)), NFrames);

                float[] DefocusValues = GetDefocusGrid(GridCTFDefocus.GetInterpolated(GridCoords));
                float[] DefocusDeltaValues = GridCTFDefocusDelta.GetInterpolated(GridCoords);
                float[] DefocusAngleValues = GridCTFDefocusAngle.GetInterpolated(GridCoords);
                float[] PhaseValues = GridCTFPhase.GetInterpolated(GridCoords);

                // Scale everything to one common defocus value
                {
                    Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                    float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                    // Construct background in Cartesian coordinates.
                    Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) =>
                    {
                        CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = GlobalBackground.Interp(r / DimsRegion.X);
                    });

                    CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                    CTFStruct[] LocalParams = new CTFStruct[DefocusValues.Length];
                    for (int f = 0; f < NFrames; f++)
                    {
                        for (int p = 0; p < NPositions; p++)
                        {
                            CTF Local = GlobalCTF.GetCopy();
                            Local.Defocus = (decimal)DefocusValues[f * NPositions + p];
                            Local.DefocusDelta = (decimal)DefocusDeltaValues[f];
                            Local.DefocusAngle = (decimal)DefocusAngleValues[f];
                            Local.PhaseShift = (decimal)PhaseValues[f];
                            Local.Scale = (decimal)Math.Pow(1 - Math.Abs(Math.Sin(Angles[f] * Helper.ToRad)), 2);

                            LocalParams[f * NPositions + p] = Local.ToStruct();
                        }
                    }

                    Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                    CTF CTFAug = GlobalCTF.GetCopy();

                    {
                        GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                           CTFCoordsCart.GetDevice(Intent.Read),
                                           (uint)CTFSpectra.ElementsSliceReal,
                                           (uint)DimsRegion.X,
                                           LocalParams,
                                           CTFAug.ToStruct(),
                                           0,
                                           (uint)DimsRegion.X / 2,
                                           (uint)LocalParams.Length,
                                           CTFAverage1D.GetDevice(Intent.Write));

                        float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                        for (int i = 0; i < RotationalAverageData.Length; i++)
                            GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                        MathHelper.UnNaN(GlobalPS1D);

                        PS1D = GlobalPS1D.ToArray();
                    }

                    #region Now go through all tilts

                    for (int t = 0; t < NTilts; t++)
                    {
                        CTFAug.Defocus = (decimal)GridCTFDefocus.FlatValues[t];

                        GPU.CTFMakeAverage(CTFSpectra.GetDeviceSlice(t * NPositions, Intent.Read),
                                           CTFCoordsCart.GetDevice(Intent.Read),
                                           (uint)CTFSpectra.ElementsSliceReal,
                                           (uint)DimsRegion.X,
                                           LocalParams.Skip(t * NPositions).Take(NPositions).ToArray(),
                                           CTFAug.ToStruct(),
                                           0,
                                           (uint)DimsRegion.X / 2,
                                           (uint)NPositions,
                                           CTFAverage1D.GetDevice(Intent.Write));

                        float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                        for (int i = 0; i < RotationalAverageData.Length; i++)
                            GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                        MathHelper.UnNaN(GlobalPS1D);

                        TiltPS1D.Add(GlobalPS1D.ToArray());
                        TiltSimulatedBackground.Add(new Cubic1D(GlobalBackground.Data.ToArray()));
                        TiltSimulatedScale.Add(new Cubic1D(GlobalScale.Data.ToArray()));

                        #region Make 2D power spectrum for display

                        Image Sum2D = new Image(CTFSpectra.Dims.Slice(), true);
                        GPU.ReduceMean(CTFSpectra.GetDeviceSlice(t * NPositions, Intent.Read),
                                       Sum2D.GetDevice(Intent.Write),
                                       (uint)Sum2D.ElementsReal,
                                       (uint)NPositions,
                                       1);

                        float[] Sum2DData = Sum2D.GetHostContinuousCopy();
                        Sum2D.Dispose();

                        int3 DimsAverage = new int3(DimsRegion.X, DimsRegion.X / 2, 1);
                        float[] Average2DData = new float[DimsAverage.Elements()];
                        int DimHalf = DimsRegion.X / 2;

                        for (int y = 0; y < DimsAverage.Y; y++)
                        {
                            int yy = y * y;
                            for (int x = 0; x < DimHalf; x++)
                            {
                                int xx = x;
                                xx *= xx;
                                float r = (float)Math.Sqrt(xx + yy) / DimsRegion.X;
                                Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x + DimHalf] = Sum2DData[(DimsRegion.X - 1 - y) * (DimsRegion.X / 2 + 1) + x];
                            }

                            for (int x = 1; x < DimHalf; x++)
                            {
                                int xx = -(x - DimHalf);
                                float r = (float)Math.Sqrt(xx * xx + yy) / DimsRegion.X;
                                Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x] = Sum2DData[y * (DimsRegion.X / 2 + 1) + xx];
                            }
                        }

                        AllPS2D.GetHost(Intent.Write)[t] = Average2DData;

                        #endregion
                    }

                    #endregion

                    AllPS2D.WriteMRC(PowerSpectrumPath, true);

                    CTFSpectraBackground.Dispose();
                    CTFAverage1D.Dispose();
                    CTFSpectra.FreeDevice();
                }
            }

            #endregion

            CTF = GlobalCTF;
            SimulatedScale = GlobalScale;
            PlaneNormal = (Matrix3.RotateX(GlobalPlaneAngle.X * Helper.ToRad) * Matrix3.RotateY(GlobalPlaneAngle.Y * Helper.ToRad)) * new float3(0, 0, 1);

            #region Estimate fittable resolution

            {
                float[] Quality = CTF.EstimateQuality(PS1D.Select(p => p.Y).ToArray(),
                                                      SimulatedScale.Interp(PS1D.Select(p => p.X).ToArray()),
                                                      (float)options.RangeMin, 16);
                int FirstFreq = 0;
                while ((float.IsNaN(Quality[FirstFreq]) || Quality[FirstFreq] < 0.8f) && FirstFreq < Quality.Length - 1)
                    FirstFreq++;

                int LastFreq = FirstFreq;
                while (!float.IsNaN(Quality[LastFreq]) && Quality[LastFreq] > 0.3f && LastFreq < Quality.Length - 1)
                    LastFreq++;

                CTFResolutionEstimate = Math.Round(options.BinnedPixelSizeMean / ((decimal)LastFreq / options.Window), 1);
            }

            #endregion

            CTFSpectra.Dispose();
            CTFMean.Dispose();
            CTFCoordsCart.Dispose();
            CTFCoordsPolarTrimmed.Dispose();

            Simulated1D = GetSimulated1D();

            OptionsCTF = options;

            SaveMeta();

            IsProcessing = false;
            TiltCTFProcessed?.Invoke();
        }

        #endregion

        public void StackTilts(ProcessingOptionsTomoStack options)
        {
            Directory.CreateDirectory(TiltStackDir);

            Movie[] TiltMovies;
            Image[] TiltData;
            Image[] TiltMasks = null;
            LoadMovieData(options, out TiltMovies, out TiltData, false, out _, out _);
            if (options.ApplyMask)
                LoadMovieMasks(options, out TiltMasks);
            for (int z = 0; z < NTilts; z++)
            {
                if (options.ApplyMask)
                {
                    EraseDirt(TiltData[z], TiltMasks[z]);
                    TiltMasks[z]?.FreeDevice();
                }

                TiltData[z].FreeDevice();
            }

            var UsedTilts = TiltData.Where((d, i) => UseTilt[i]).ToArray();
            var UsedAngles = Angles.Where((d, i) => UseTilt[i]).ToArray();

            Image Stack = new Image(UsedTilts.Select(i => i.GetHost(Intent.Read)[0]).ToArray(), new int3(UsedTilts[0].Dims.X, UsedTilts[0].Dims.Y, UsedTilts.Length));
            Stack.WriteMRC(TiltStackPath, (float)options.BinnedPixelSizeMean, true);

            File.WriteAllLines(AngleFilePath, UsedAngles.Select(a => a.ToString("F2", CultureInfo.InvariantCulture)));
        }

        public void ImportAlignments(ProcessingOptionsTomoImportAlignments options)
        {
            string ResultsDir = string.IsNullOrEmpty(options.OverrideResultsDir) ? TiltStackDir : options.OverrideResultsDir;

            UseTilt = Helper.ArrayOfConstant(true, NTilts);

            #region Excluded tilts

            string CutviewsPath1 = IOPath.Combine(ResultsDir, RootName + "_cutviews0.rawtlt");
            string CutviewsPath2 = IOPath.Combine(ResultsDir, "../", RootName + "_cutviews0.rawtlt");
            string CutviewsPath3 = IOPath.Combine(ResultsDir, RootName + "_Imod", RootName + "_cutviews0.rawtlt");
            string CutviewsPath = null;
            try
            {
                CutviewsPath = (new[] { CutviewsPath3, CutviewsPath1, CutviewsPath2 }).First(s => File.Exists(s));
            }
            catch { }
            if (CutviewsPath != null)
            {
                List<float> CutAngles = File.ReadAllLines(CutviewsPath).Where(l => !string.IsNullOrEmpty(l)).Select(l => float.Parse(l, CultureInfo.InvariantCulture)).ToList();

                UseTilt = UseTilt.Select((v, t) => !CutAngles.Any(a => Math.Abs(a - Angles[t]) < 0.2)).ToArray();
            }

            #endregion

            int NValid = UseTilt.Count(v => v);

            #region Transforms

            // .xf
            {
                string[] Directories = {"", "../", $"{RootName}_Imod", $"{RootName}_aligned_Imod", $"../{RootName}_Imod" };
                string[] FileNames =
                {
                    $"{RootName}.xf",
                    $"{RootName.Replace(".mrc", "")}.xf",
                    $"{RootName}_st.xf",
                    $"{RootName.Replace(".mrc", "")}_st.xf"
                };
                string[] XfPaths = new string[Directories.Length * FileNames.Length];
                int idx;
                for (int i = 0; i < Directories.Length; i++)
                    for (int j = 0; j < FileNames.Length; j++)
                    {
                        idx = i * FileNames.Length + j;
                        XfPaths[idx] = IOPath.GetFullPath(IOPath.Combine(ResultsDir, Directories[i], FileNames[j]));
                    }

                if (Helper.IsDebug)
                {
                    Console.WriteLine("Possible XF file paths:");
                    foreach (string path in XfPaths)
                        Console.WriteLine($"{path}");
                }
                string XfPath = null;
                try
                {
                    XfPath = XfPaths.First(s => File.Exists(s));
                    if (Helper.IsDebug)
                        Console.WriteLine($"\nImporting 2D transforms from {XfPath}");
                }
                catch { }
                if (XfPath == null)
                    throw new Exception($"Could not find {RootName}.xf");

                string[] Lines = File.ReadAllLines(XfPath).Where(l => !string.IsNullOrEmpty(l)).ToArray();
                if (Lines.Length != NValid)
                    throw new Exception($"{NValid} active tilts in series, but {Lines.Length} lines in {XfPath}");

                for (int t = 0, iline = 0; t < NTilts; t++)
                {
                    if (!UseTilt[t])
                        continue;

                    string Line = Lines[iline];

                    string[] Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    float2 VecX = new float2(float.Parse(Parts[0], CultureInfo.InvariantCulture),
                                                float.Parse(Parts[2], CultureInfo.InvariantCulture));
                    float2 VecY = new float2(float.Parse(Parts[1], CultureInfo.InvariantCulture),
                                                float.Parse(Parts[3], CultureInfo.InvariantCulture));

                    Matrix3 Rotation = new Matrix3(VecX.X, VecX.Y, 0, VecY.X, VecY.Y, 0, 0, 0, 1);
                    float3 Euler = Matrix3.EulerFromMatrix(Rotation);

                    TiltAxisAngles[t] = Euler.Z * Helper.ToDeg;

                    //SortedAngle[i].Shift += VecX * float.Parse(Parts[4], CultureInfo.InvariantCulture) + VecY * float.Parse(Parts[5], CultureInfo.InvariantCulture);
                    float3 Shift = new float3(-float.Parse(Parts[4], CultureInfo.InvariantCulture), -float.Parse(Parts[5], CultureInfo.InvariantCulture), 0);
                    Shift = Rotation.Transposed() * Shift;

                    Shift *= (float)options.BinnedPixelSizeMean;

                    TiltAxisOffsetX[t] = Shift.X;
                    TiltAxisOffsetY[t] = Shift.Y;

                    iline++;
                }
            }

            // .tlt
            {
                string[] Directories = { "", "../", $"{RootName}_Imod", $"{RootName}_aligned_Imod", $"../{RootName}_Imod" };
                string[] FileNames =
                {
                    $"{RootName}.tlt",
                    $"{RootName.Replace(".mrc", "")}.tlt",
                    $"{RootName}_st.tlt",
                    $"{RootName.Replace(".mrc", "")}_st.tlt"
                };
                string[] TltPaths = new string[Directories.Length * FileNames.Length];
                int idx;
                for (int i = 0; i < Directories.Length; i++)
                    for (int j = 0; j < FileNames.Length; j++)
                    {
                        idx = i * FileNames.Length + j;
                        TltPaths[idx] = IOPath.GetFullPath(IOPath.Combine(ResultsDir, Directories[i], FileNames[j]));
                    }

                if (Helper.IsDebug)
                {
                    Console.WriteLine("Possible TLT file paths:");
                    foreach (string path in TltPaths)
                        Console.WriteLine($"{path}");
                }
                string TltPath = null;
                try
                {
                    TltPath = TltPaths.First(s => File.Exists(s));
                    if (Helper.IsDebug)
                        Console.WriteLine($"\nImporting tilt angles from {TltPath}");
                }
                catch { }
                if (TltPath == null)
                    throw new Exception($"Could not find {RootName}.xf");

                string[] Lines = File.ReadAllLines(TltPath).Where(l => !string.IsNullOrEmpty(l)).ToArray();

                if (Lines.Length == NValid)
                {
                    float[] ParsedTiltAngles = new float[NTilts];
                    for (int t = 0; t < NTilts; t++)
                    {
                        string Line = Lines[t];
                        ParsedTiltAngles[t] = float.Parse(Line, CultureInfo.InvariantCulture);
                    }

                    if (ParsedTiltAngles.All(angle => angle == 0))
                        throw new Exception($"all tilt angles are zero in {TltPath}");
                    else
                    {
                        for (int t = 0; t < NTilts; t++)
                        {
                            if (!UseTilt[t])
                                continue;
                            Angles[t] = ParsedTiltAngles[t];
                        }
                    }
                }
            }

            #endregion

            #region FOV fraction

            if (options.MinFOV > 0)
            {
                VolumeDimensionsPhysical = new float3((float)options.DimensionsPhysical.X, (float)options.DimensionsPhysical.Y, 1);
                LoadMovieSizes();

                int NSteps = 100;
                var Positions = new float3[NSteps * NSteps];
                for (int y = 0; y < NSteps; y++)
                {
                    float yy = VolumeDimensionsPhysical.Y * y / (NSteps - 1);
                    for (int x = 0; x < NSteps; x++)
                    {
                        float xx = VolumeDimensionsPhysical.X * x / (NSteps - 1);
                        Positions[y * NSteps + x] = new float3(xx, yy, 0);
                    }
                }

                float[] FOVFractions = new float[NTilts];

                for (int t = 0; t < NTilts; t++)
                {
                    if (!UseTilt[t])
                        continue;

                    float3[] ImagePositions = GetPositionsInOneTilt(Positions, t);
                    int NContained = 0;
                    foreach (var pos in ImagePositions)
                        if (pos.X >= 0 && pos.Y >= 0 &&
                            pos.X <= ImageDimensionsPhysical.X - 1 &&
                            pos.Y <= ImageDimensionsPhysical.Y - 1)
                            NContained++;

                    FOVFractions[t] = (float)NContained / ImagePositions.Length;
                }

                float FractionAt0 = Helper.ArrayOfFunction(i => FOVFractions[IndicesSortedAbsoluteAngle[i]], Math.Min(5, FOVFractions.Length)).Max();
                if (FractionAt0 > 0)
                    FOVFractions = FOVFractions.Select(v => v / FractionAt0).ToArray();

                UseTilt = UseTilt.Select((v, t) => v && FOVFractions[t] >= (float)options.MinFOV).ToArray();
            }

            #endregion
        }

        public void AlignToMaxL2(ProcessingOptionsTomoFullReconstruction options)
        {
            VolumeDimensionsPhysical = options.DimensionsPhysical;
            int SizeRegion = options.SubVolumeSize;

            #region Load and preprocess data

            Image[] TiltData;
            Image[] TiltMasks;
            LoadMovieData(options, out _, out TiltData, false, out _, out _, false);
            LoadMovieMasks(options, out TiltMasks);
            for (int z = 0; z < NTilts; z++)
            {
                EraseDirt(TiltData[z], TiltMasks[z]);
                TiltMasks[z]?.FreeDevice();

                TiltData[z].SubtractMeanGrid(new int2(1));
                TiltData[z] = TiltData[z].AsPaddedClamped(new int2(TiltData[z].Dims) * 2).AndDisposeParent();
                TiltData[z].MaskRectangularly((TiltData[z].Dims / 2).Slice(), MathF.Min(TiltData[z].Dims.X / 4, TiltData[z].Dims.Y / 4), false);
                //TiltData[z].WriteMRC("d_tiltdata.mrc", true);
                //TiltData[z].Bandpass(1f / (SizeRegion / 2), 1f, false, 1f / (SizeRegion / 2));
                TiltData[z].Bandpass(2f / (300f / (float)options.BinnedPixelSizeMean), 1f, false, 2f / (300f / (float)options.BinnedPixelSizeMean));
                TiltData[z] = TiltData[z].AsPadded(new int2(TiltData[z].Dims) / 2).AndDisposeParent();
                //TiltData[z].WriteMRC("d_tiltdatabp.mrc", true);

                GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                                TiltData[z].GetDevice(Intent.Write),
                                (uint)TiltData[z].ElementsReal,
                                1);

                if (options.Invert)
                    TiltData[z].Multiply(-1f);

                TiltData[z].FreeDevice();
            }

            int2 DimsImage = new int2(TiltData[0].Dims);
            int SizeReconstruction = SizeRegion;
            int SizeReconstructionPadded = SizeReconstruction * 2;

            #endregion

            float3[] TransformedAngles = GetAngleInAllTilts(VolumeDimensionsPhysical / 2).Select(a => Matrix3.EulerFromMatrix(Matrix3.Euler(a) * Matrix3.RotateX(0 * Helper.ToRad))).ToArray();

            #region Make CTF

            Image BlobCTF;
            Image Multiplicity;
            Image RecCTFs, RecCTFsAbs;
            {
                Image CTFCoords = CTF.GetCTFCoords(SizeReconstructionPadded, SizeReconstructionPadded);
                Image CTFs = GetCTFsForOneParticle(options, VolumeDimensionsPhysical * 0.5f, CTFCoords, null, true, false, false);
                Image CTFsAbs = CTFs.GetCopyGPU();
                CTFsAbs.Abs();
                RecCTFs = CTFs.GetCopyGPU();
                RecCTFsAbs = CTFsAbs.GetCopyGPU();

                //RecCTFs.Fill(1);
                //RecCTFsAbs.Fill(1);

                // CTF has to be converted to complex numbers with imag = 0, and weighted by itself
                Image CTFsComplex = new Image(CTFs.Dims, true, true);
                CTFsComplex.Fill(new float2(1f / (SizeReconstructionPadded * SizeReconstructionPadded), 0));
                CTFsComplex.Multiply(CTFs);
                CTFsComplex.Multiply(CTFs);

                //RandomNormal Random = new RandomNormal(123);
                //CTFsComplex.ShiftSlices(Helper.ArrayOfFunction(i => new float3(0, MathF.Abs(MathF.Sin(Angles[i] * Helper.ToRad)), 0) * 5f, NTilts));

                // Back-project and reconstruct
                Projector ProjCTF = new Projector(new int3(SizeReconstructionPadded), 1);
                Projector ProjCTFWeights = new Projector(new int3(SizeReconstructionPadded), 1);

                //ProjCTF.Weights.Fill(0.01f);

                ProjCTF.BackProject(CTFsComplex, CTFsAbs, TransformedAngles, MagnificationCorrection);

                CTFsAbs.Fill(1);
                ProjCTFWeights.BackProject(CTFsComplex, CTFsAbs, TransformedAngles, MagnificationCorrection);
                ProjCTFWeights.Weights.Min(1);
                ProjCTF.Data.Multiply(ProjCTFWeights.Weights);

                Multiplicity = ProjCTFWeights.Weights.GetCopyGPU();

                CTFsComplex.Dispose();
                ProjCTFWeights.Dispose();

                Image PSF = ProjCTF.Reconstruct(false, "C1", null, -1, -1, -1, 0).AsPadded(new int3(SizeReconstruction)).AndDisposeParent();
                PSF.WriteMRC("d_psf.mrc", true);
                //Console.WriteLine(PSF.GetHost(Intent.Read).SelectMany(v => v).Select(v => MathF.Min(0, v)).Sum());
                PSF.RemapToFT(true);
                ProjCTF.Dispose();

                BlobCTF = PSF.AsFFT(true).AndDisposeParent().
                              AsAmplitudes().AndDisposeParent();
                BlobCTF.Multiply(1f / (SizeRegion * SizeRegion));

                CTFs.Dispose();
                CTFsAbs.Dispose();
            }
            BlobCTF.WriteMRC("d_ctf.mrc", true);

            #endregion

            #region Make blobs

            int[] BlobSizes = { 40 };
            Image[] BlobVolumes = new Image[BlobSizes.Length];

            for (int i = 0; i < BlobSizes.Length; i++)
            {
                Image BlobVolume = new Image(new int3(SizeRegion));
                BlobVolume.Fill(1);
                BlobVolume.MaskSpherically(BlobSizes[i] / (float)options.BinnedPixelSizeMean, 3, true, true);

                float BlobSum = BlobVolume.GetHost(Intent.Read).SelectMany(v => v).Sum();
                //BlobVolume.Multiply(1f / BlobSum);

                BlobVolume.WriteMRC($"d_blob_{BlobSizes[i]}.mrc", true);

                BlobVolume = BlobVolume.AsFFT(true).AndDisposeParent();
                //BlobVolume.Multiply(BlobCTF);

                BlobVolumes[i] = BlobVolume;
            }

            #endregion

            int2 DimsPositionGrid;
            int3[] PositionGrid = Helper.GetEqualGridSpacing(new int2(new float2(VolumeDimensionsPhysical.X, VolumeDimensionsPhysical.Y) / (float)options.BinnedPixelSizeMean) - 64,
                                                               new int2(SizeRegion),
                                                               0.25f,
                                                               out DimsPositionGrid).Select(v => new int3(v.X + 32 + SizeRegion / 2,
                                                                                                          v.Y + 32 + SizeRegion / 2,
                                                                                                          0)).ToArray();
            float3[] PositionGridPhysical = PositionGrid.Select(v => new float3(v.X * (float)options.BinnedPixelSizeMean,
                                                                                v.Y * (float)options.BinnedPixelSizeMean,
                                                                                VolumeDimensionsPhysical.Z / 2)).ToArray();

            Image RegionMask = new Image(new int3(SizeRegion));
            RegionMask.Fill(1);
            //RegionMask.MaskRectangularly(new int3(SizeRegion / 2), SizeRegion / 4, true);
            RegionMask.MaskSpherically(SizeRegion / 2, SizeRegion / 4, true, false);
            RegionMask.WriteMRC("d_mask.mrc", true);

            Projector Reconstructor = new Projector(new int3(SizeReconstructionPadded), 1);
            Image Reconstruction = new Image(new int3(SizeReconstructionPadded));
            Image RecCropped = new Image(new int3(SizeReconstruction));

            int PlanForwParticles = GPU.CreateFFTPlan(new int3(SizeReconstructionPadded, SizeReconstructionPadded, 1), (uint)NTilts);
            int PlanForwRec, PlanBackRec, PlanCTFBack;
            Projector.GetPlans(Reconstructor.Dims, Reconstructor.Oversampling, out PlanForwRec, out PlanBackRec, out PlanCTFBack);

            Image ParticlesExtracted = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, NTilts));
            Image ParticlesExtractedFT = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, NTilts), true, true);

            Image RecSum = new Image(new int3(1));
            float[] h_RecSum = new float[1];

            GridMovementX = GridMovementX.Resize(new int3(1, 1, NTilts));
            GridMovementY = GridMovementY.Resize(GridMovementX.Dimensions);

            float2[] VecZ = new float2[NTilts];

            #region Parameter setting

            float[] OriX = GridMovementX.Values.ToArray();
            float[] OriY = GridMovementY.Values.ToArray();

            int OptimizedTilts = 1;

            Action<TiltSeries, double[]> SetParams = (series, input) =>
            {
                var ParamsX = input.Take((int)GridMovementX.Dimensions.Elements()).Select(v => (float)v).ToArray();
                var ParamsY = input.Skip((int)GridMovementX.Dimensions.Elements()).Take((int)GridMovementY.Dimensions.Elements()).Select(v => (float)v).ToArray();
                float2[] ParamsXY = Helper.Zip(ParamsX, ParamsY);
                //for (int i = 1; i <= NTilts / 2; i++)
                //{
                //    var PairXY = new List<float2>(2);
                //    var PairZ = new List<float2>(2);

                //    if (NTilts / 2 - i >= 0)
                //    {
                //        PairXY.Add(ParamsXY[NTilts / 2 - i]);
                //        PairZ.Add(VecZ[NTilts / 2 - i]);
                //    }
                //    if (NTilts / 2 + i < NTilts)
                //    {
                //        PairXY.Add(ParamsXY[NTilts / 2 + i]);
                //        PairZ.Add(VecZ[NTilts / 2 + i]);
                //    }
                //    float Length = MathF.Sqrt(PairZ.Select(v => v.LengthSq()).Sum());
                //    if (Length > 0)
                //    {
                //        float Dot = PairXY.Zip(PairZ, (v, z) => float2.Dot(v, z / Length)).Sum();

                //        if (NTilts / 2 - i >= 0)
                //            ParamsXY[NTilts / 2 - i] -= VecZ[NTilts / 2 - i] / Length * Dot;
                //        if (NTilts / 2 + i < NTilts)
                //            ParamsXY[NTilts / 2 + i] -= VecZ[NTilts / 2 + i] / Length * Dot;
                //    }
                //}
                if (OptimizedTilts <= 0)
                {
                    float Dot = ParamsXY.Zip(VecZ, (v, z) => float2.Dot(v, z)).Sum();
                    float Scale = MathHelper.FitScaleLeastSq(VecZ.SelectMany(v => new[] { v.X, v.Y }).ToArray(), ParamsXY.SelectMany(v => new[] { v.X, v.Y }).ToArray());
                    ParamsXY = ParamsXY.Zip(VecZ, (v, z) => v - z * Scale).ToArray();
                }

                series.GridMovementX = new CubicGrid(GridMovementX.Dimensions, ParamsXY.Zip(OriX, (v, o) => v.X + o).ToArray());
                series.GridMovementY = new CubicGrid(GridMovementY.Dimensions, ParamsXY.Zip(OriY, (v, o) => v.Y + o).ToArray());
            };

            #endregion

            #region Wiggle weights

            int NWiggleDifferentiable = GridMovementX.Values.Length +
                                        GridMovementY.Values.Length;
            (int[] indices, float2[] weights)[] AllWiggleWeights = new (int[] indices, float2[] weights)[NWiggleDifferentiable];

            {
                TiltSeries[] ParallelSeriesCopies = Helper.ArrayOfFunction(i => new TiltSeries(this.Path), 1);

                int NPositions = PositionGrid.Length;

                var ParticlePositionsOri = new float2[NPositions * NTilts];
                for (int p = 0; p < NPositions; p++)
                {
                    float3[] Positions = GetPositionInAllTilts(PositionGridPhysical[p]);
                    for (int t = 0; t < NTilts; t++)
                        ParticlePositionsOri[p * NTilts + t] = new float2(Positions[t].X, Positions[t].Y);
                }

                Helper.ForCPU(0, NWiggleDifferentiable, ParallelSeriesCopies.Length,
                (threadID) =>
                {
                    ParallelSeriesCopies[threadID].VolumeDimensionsPhysical = VolumeDimensionsPhysical;
                    ParallelSeriesCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                    ParallelSeriesCopies[threadID].SizeRoundingFactors = SizeRoundingFactors;
                },
                (iwiggle, threadID) =>
                {
                    double[] WiggleParams = new double[NWiggleDifferentiable];
                    WiggleParams[iwiggle] = 1;
                    SetParams(ParallelSeriesCopies[threadID], WiggleParams);

                    float2[] RawShifts = new float2[NPositions * NTilts];
                    for (int p = 0; p < NPositions; p++)
                    {
                        float3[] ParticlePositionsProjected = ParallelSeriesCopies[threadID].GetPositionInAllTilts(PositionGridPhysical[p]);

                        for (int t = 0; t < NTilts; t++)
                            RawShifts[p * NTilts + t] = new float2(ParticlePositionsProjected[t]) - ParticlePositionsOri[p * NTilts + t];
                    }

                    List<int> Indices = new List<int>();
                    List<float2> Weights = new List<float2>();
                    for (int i = 0; i < RawShifts.Length; i++)
                    {
                        if (RawShifts[i].LengthSq() > 1e-6f)
                        {
                            Indices.Add(i / NTilts);
                            Weights.Add(RawShifts[i]);

                            if (Math.Abs(RawShifts[i].X) > 1.5f)
                            {
                                throw new Exception("");
                            }
                        }
                    }

                    AllWiggleWeights[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                }, null);
            }

            #endregion

            Func<double[], float2, int, double[]> EvalIndividual = (input, shiftXY, shiftedTilt) =>
            {
                if (shiftXY != new float2(0))
                {
                    input = input.ToArray();
                    input[shiftedTilt] += shiftXY.X;
                    input[NTilts + shiftedTilt] += shiftXY.Y;
                }

                SetParams(this, input);

                double[] Result = new double[PositionGrid.Length];

                for (int ipos = 0; ipos < PositionGrid.Length; ipos++)
                {
                    var Pos = PositionGridPhysical[ipos];

                    Reconstructor.Data.Fill(0);
                    Reconstructor.Weights.Fill(0);

                    GetImagesForOneParticle(options, TiltData, SizeReconstructionPadded, Pos, PlanForwParticles, SizeReconstructionPadded - 36, 16, ParticlesExtracted, ParticlesExtractedFT);
                    //ParticlesExtractedFT.Fill(new float2(1f / (SizeReconstruction * SizeReconstruction), 0));
                    //ParticlesExtractedFT.ShiftSlices(Helper.ArrayOfFunction(t => new float3((float)(GridMovementX.Values[t] / (float)options.BinnedPixelSizeMean), 
                    //                                                                        (float)(GridMovementY.Values[t] / (float)options.BinnedPixelSizeMean), 0), NTilts));
                    //if (shiftXY != new float2(0))
                    //    ParticlesExtractedFT.ShiftSlices(Helper.ArrayOfFunction(t => t == shiftedTilt ? new float3(shiftXY) / (float)options.BinnedPixelSizeMean : new float3(0), NTilts));
                    ParticlesExtractedFT.Multiply(RecCTFs);
                    if (OptimizedTilts > 0)
                        ParticlesExtractedFT.Multiply(Helper.ArrayOfFunction(t => (t >= NTilts / 2 - OptimizedTilts && t <= NTilts / 2 + OptimizedTilts) ? 1f : 0, NTilts));

                    Reconstructor.BackProject(ParticlesExtractedFT, RecCTFsAbs, GetAngleInAllTilts(Pos), MagnificationCorrection);
                    Reconstructor.Data.Multiply(Multiplicity);
                    //Reconstructor.Weights.Fill(1);

                    Reconstructor.Reconstruct(Reconstruction.GetDevice(Intent.Write), false, "C1", null, PlanForwRec, PlanBackRec, PlanCTFBack, 0);
                    //Reconstruction.WriteMRC("d_reconstruction.mrc", true);

                    GPU.Pad(Reconstruction.GetDevice(Intent.Read),
                            RecCropped.GetDevice(Intent.Write),
                            new int3(SizeReconstructionPadded),
                            new int3(SizeReconstruction),
                            1);
                    //RecCropped.Max(0.1f);

                    RecCropped.Multiply(SizeReconstruction);
                    RecCropped.Abs();
                    RecCropped.Multiply(RecCropped);
                    RecCropped.Multiply(RecCropped);
                    RecCropped.Multiply(RegionMask);

                    GPU.Sum(RecCropped.GetDevice(Intent.Read), RecSum.GetDevice(Intent.Write), (uint)RecCropped.ElementsReal, 1);
                    GPU.CopyDeviceToHost(RecSum.GetDevice(Intent.Read), h_RecSum, 1);

                    Result[ipos] = h_RecSum[0] / RecCropped.ElementsReal * 100f / PositionGrid.Length;
                }

                return Result;
            };

            Func<double[], double> Eval = (input) =>
            {
                return EvalIndividual(input, new float2(0), 0).Sum();
            };

            Func<double[], double[]> Grad = (input) =>
            {
                double Delta = 0.5;
                double[] Result = new double[input.Length];

                //for (int i = 0; i < input.Length; i++)
                //{
                //    int itilt = (i / (int)GridMovementX.Dimensions.ElementsSlice()) % NTilts;
                //    if (itilt == NTilts / 2)
                //        continue;

                //    if (itilt != NTilts / 2 - OptimizedTilts && itilt != NTilts / 2 + OptimizedTilts)
                //        continue;

                //    double[] InputPlus = input.ToArray();
                //    InputPlus[i] += Delta;
                //    double ScorePlus = Eval(InputPlus);

                //    double[] InputMinus = input.ToArray();
                //    InputMinus[i] -= Delta;
                //    double ScoreMinus = Eval(InputMinus);

                //    Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                //}

                int NSlice = (int)GridMovementX.Dimensions.ElementsSlice();

                for (int i = 0; i < GridMovementX.Dimensions.Z; i++)
                {
                    int itilt = i;
                    if (itilt == NTilts / 2)
                        continue;

                    if (OptimizedTilts > 0)
                        if (itilt != NTilts / 2 - OptimizedTilts && itilt != NTilts / 2 + OptimizedTilts)
                            continue;

                    // Flipping the sign here because EvalIndividual image shift goes in the opposite direction of particle position change
                    double[] ScorePlusX = EvalIndividual(input, new float2(-(float)Delta, 0), itilt);
                    double[] ScoreMinusX = EvalIndividual(input, new float2((float)Delta, 0), itilt);
                    float[] GradsX = ScorePlusX.Zip(ScoreMinusX, (p, m) => (float)((p - m) / (Delta * 2))).ToArray();

                    double[] ScorePlusY = EvalIndividual(input, new float2(0, -(float)Delta), itilt);
                    double[] ScoreMinusY = EvalIndividual(input, new float2(0, (float)Delta), itilt);
                    float[] GradsY = ScorePlusY.Zip(ScoreMinusY, (p, m) => (float)((p - m) / (Delta * 2))).ToArray();

                    float2[] Grads = GradsX.Zip(GradsY, (x, y) => new float2(x, y)).ToArray();

                    for (int ielement = 0; ielement < NSlice; ielement++)
                    {
                        {
                            (int[] Elements, float2[] Weights) = AllWiggleWeights[i * NSlice + ielement];
                            float WeightedSum = 0;
                            for (int j = 0; j < Elements.Length; j++)
                                WeightedSum += float2.Dot(Grads[Elements[j]], Weights[j]);

                            Result[i * NSlice + ielement] = WeightedSum;
                        }

                        {
                            (int[] Elements, float2[] Weights) = AllWiggleWeights[GridMovementX.Values.Length + i * NSlice + ielement];
                            float WeightedSum = 0;
                            for (int j = 0; j < Elements.Length; j++)
                                WeightedSum += float2.Dot(Grads[Elements[j]], Weights[j]);

                            Result[GridMovementX.Values.Length + i * NSlice + ielement] = WeightedSum;
                        }
                    }
                }

                Console.WriteLine(Eval(input));

                return Result;
            };

            double[] StartValues = new double[GridMovementX.Dimensions.Elements() * 2];

            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartValues.Length, Eval, Grad);
            Optimizer.MaxLineSearch = 10;
            Optimizer.MaxIterations = 10;
            if (true)
                for (int i = 1; i <= NTilts / 2; i++)
                {
                    OptimizedTilts = i;

                    RecCTFsAbs?.Dispose();
                    RecCTFsAbs = RecCTFs.GetCopyGPU();
                    RecCTFsAbs.Multiply(Helper.ArrayOfFunction(t => (t >= NTilts / 2 - OptimizedTilts && t <= NTilts / 2 + OptimizedTilts) ? 1f : 0, NTilts));

                    {
                        Image OnesComplex = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, NTilts), true, true);
                        OnesComplex.Fill(new float2(1, 0));
                        Image Ones = OnesComplex.AsReal();
                        Ones.Multiply(Helper.ArrayOfFunction(t => (t >= NTilts / 2 - OptimizedTilts && t <= NTilts / 2 + OptimizedTilts) ? 1f : 0, NTilts));

                        Projector ProjCTFWeights = new Projector(new int3(SizeReconstructionPadded), 1);

                        ProjCTFWeights.BackProject(OnesComplex, Ones, TransformedAngles, MagnificationCorrection);
                        //ProjCTFWeights.Weights.Min(1);

                        Multiplicity?.Dispose();
                        Multiplicity = ProjCTFWeights.Weights.GetCopyGPU();
                        Multiplicity.WriteMRC($"d_multiplicity_{i:D2}.mrc", true);

                        ProjCTFWeights.Dispose();
                        Ones.Dispose();
                        OnesComplex.Dispose();
                    }

                    {
                        float2[] Pos0 = GetPositionInAllTilts(VolumeDimensionsPhysical * 0.5f).Select(v => new float2(v)).ToArray();
                        float2[] Pos1 = GetPositionInAllTilts(VolumeDimensionsPhysical * 0.5f + float3.UnitZ).Select(v => new float2(v)).ToArray();
                        float2[] Diff = Pos1.Zip(Pos0, (p1, p0) => p1 - p0).ToArray();
                        VecZ = new float2[NTilts];
                        for (int t = 0; t < NTilts; t++)
                            if (t == NTilts / 2 - OptimizedTilts || t == NTilts / 2 + OptimizedTilts)
                                VecZ[t] = Diff[t];
                        //VecZ = Diff;
                        float Length = MathF.Sqrt(VecZ.Select(v => v.LengthSq()).Sum());
                        VecZ = VecZ.Select(v => v / Length).ToArray();
                    }

                    Eval(StartValues);
                    Reconstruction.WriteMRC($"d_reconstruction_ori_{i:D2}.mrc", true);
                    Optimizer.Maximize(StartValues);
                    Reconstruction.WriteMRC($"d_reconstruction_ali_{i:D2}.mrc", true);

                    if (false)
                    {
                        var ParamsX = StartValues.Take((int)GridMovementX.Dimensions.Elements()).Select(v => (float)v).ToArray();
                        var ParamsY = StartValues.Skip((int)GridMovementX.Dimensions.Elements()).Take((int)GridMovementY.Dimensions.Elements()).Select(v => (float)v).ToArray();
                        float2[] ParamsXY = Helper.Zip(ParamsX, ParamsY);
                        float Dot = ParamsXY.Zip(VecZ, (v, z) => float2.Dot(v, z)).Sum();
                        float Scale = MathHelper.FitScaleLeastSq(VecZ.SelectMany(v => new[] { v.X, v.Y }).ToArray(), ParamsXY.SelectMany(v => new[] { v.X, v.Y }).ToArray());
                        ParamsXY = ParamsXY.Zip(VecZ, (v, z) => v - z * Scale).ToArray();
                        StartValues = Helper.Combine(ParamsXY.Select(v => (double)v.X).ToArray(), ParamsXY.Select(v => (double)v.Y).ToArray());
                    }

                    Console.WriteLine(Eval(StartValues));

                    int NSlice = (int)GridMovementX.Dimensions.ElementsSlice();

                    if (NTilts / 2 - OptimizedTilts - 1 >= 0)
                        for (int e = 0; e < NSlice; e++)
                        {
                            StartValues[(NTilts / 2 - OptimizedTilts - 1) * NSlice + e] = StartValues[(NTilts / 2 - OptimizedTilts) * NSlice + e];
                            StartValues[GridMovementX.Values.Length + (NTilts / 2 - OptimizedTilts - 1) * NSlice + e] = StartValues[GridMovementX.Values.Length + (NTilts / 2 - OptimizedTilts) * NSlice + e];
                        }
                    if (NTilts / 2 + OptimizedTilts + 1 < NTilts)
                    {
                        for (int e = 0; e < NSlice; e++)
                        {
                            StartValues[(NTilts / 2 + OptimizedTilts + 1) * NSlice + e] = StartValues[(NTilts / 2 + OptimizedTilts) * NSlice + e];
                            StartValues[GridMovementX.Values.Length + (NTilts / 2 + OptimizedTilts + 1) * NSlice + e] = StartValues[GridMovementX.Values.Length + (NTilts / 2 + OptimizedTilts) * NSlice + e];
                        }
                    }
                }

            if (true)
            {
                OptimizedTilts = -1;
                RecCTFsAbs?.Dispose();
                RecCTFsAbs = RecCTFs.GetCopyGPU();

                {
                    Image OnesComplex = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, NTilts), true, true);
                    OnesComplex.Fill(new float2(1, 0));
                    Image Ones = OnesComplex.AsReal();

                    Projector ProjCTFWeights = new Projector(new int3(SizeReconstructionPadded), 1);

                    ProjCTFWeights.BackProject(OnesComplex, Ones, TransformedAngles, MagnificationCorrection);
                    //ProjCTFWeights.Weights.Min(1);

                    Multiplicity?.Dispose();
                    Multiplicity = ProjCTFWeights.Weights.GetCopyGPU();
                    Multiplicity.WriteMRC($"d_multiplicity_global.mrc", true);

                    ProjCTFWeights.Dispose();
                    Ones.Dispose();
                    OnesComplex.Dispose();
                }

                {
                    float2[] Pos0 = GetPositionInAllTilts(VolumeDimensionsPhysical * 0.5f).Select(v => new float2(v)).ToArray();
                    float2[] Pos1 = GetPositionInAllTilts(VolumeDimensionsPhysical * 0.5f + float3.UnitZ).Select(v => new float2(v)).ToArray();
                    float2[] Diff = Pos1.Zip(Pos0, (p1, p0) => p1 - p0).ToArray();
                    VecZ = Diff;
                    VecZ[NTilts / 2] = new float2(0);
                    float Length = MathF.Sqrt(VecZ.Select(v => v.LengthSq()).Sum());
                    VecZ = VecZ.Select(v => v / Length).ToArray();
                }

                Optimizer.MaxLineSearch = 30;
                Optimizer.MaxIterations = 30;

                Eval(StartValues);
                Reconstruction.WriteMRC($"d_reconstruction_ori_global.mrc", true);
                Optimizer.Maximize(StartValues);
                Reconstruction.WriteMRC($"d_reconstruction_ali_global.mrc", true);

                {
                    var ParamsX = StartValues.Take((int)GridMovementX.Dimensions.Elements()).Select(v => (float)v).ToArray();
                    var ParamsY = StartValues.Skip((int)GridMovementX.Dimensions.Elements()).Take((int)GridMovementY.Dimensions.Elements()).Select(v => (float)v).ToArray();
                    float2[] ParamsXY = Helper.Zip(ParamsX, ParamsY);
                    float Dot = ParamsXY.Zip(VecZ, (v, z) => float2.Dot(v, z)).Sum();
                    float Scale = MathHelper.FitScaleLeastSq(VecZ.SelectMany(v => new[] { v.X, v.Y }).ToArray(), ParamsXY.SelectMany(v => new[] { v.X, v.Y }).ToArray());
                    ParamsXY = ParamsXY.Zip(VecZ, (v, z) => v - z * Scale).ToArray();
                    StartValues = Helper.Combine(ParamsXY.Select(v => (double)v.X).ToArray(), ParamsXY.Select(v => (double)v.Y).ToArray());
                }
            }
        }

        public void MatchFull(ProcessingOptionsTomoFullMatch options, Image template, Func<int3, float, string, bool> progressCallback)
        {
            bool IsCanceled = false;
            if (!Directory.Exists(MatchingDir))
                Directory.CreateDirectory(MatchingDir);

            string NameWithRes = RootName + $"_{options.BinnedPixelSizeMean:F2}Apx";

            float3[] HealpixAngles = Helper.GetHealpixAngles(options.HealpixOrder, options.Symmetry).Select(a => a * Helper.ToRad).ToArray();
            if (options.TiltRange >= 0)
            {
                float Limit = MathF.Sin((float)options.TiltRange * Helper.ToRad);
                HealpixAngles = HealpixAngles.Where(a => MathF.Abs(Matrix3.Euler(a).C3.Z) <= Limit).ToArray();
            }
            progressCallback?.Invoke(new int3(1), 0, $"Using {HealpixAngles.Length} orientations for matching");

            LoadMovieSizes();

            Image CorrVolume = null, AngleIDVolume = null;
            float[][] CorrData;
            float[][] AngleIDData;

            #region Dimensions

            int SizeSub = options.SubVolumeSize;
            int SizeParticle = (int)(options.TemplateDiameter / options.BinnedPixelSizeMean);
            int PeakDistance = (int)(options.PeakDistance / options.BinnedPixelSizeMean);

            int3 DimsVolumeScaled = new int3((int)Math.Round(options.DimensionsPhysical.X / (float)options.BinnedPixelSizeMean / 2) * 2,
                                                (int)Math.Round(options.DimensionsPhysical.Y / (float)options.BinnedPixelSizeMean / 2) * 2,
                                                (int)Math.Round(options.DimensionsPhysical.Z / (float)options.BinnedPixelSizeMean / 2) * 2);

            VolumeDimensionsPhysical = options.DimensionsPhysical;

            // Find optimal box size for matching
            {
                int BestSizeSub = 0;
                long BestVoxels = long.MaxValue;

                for (int testSizeSub = (SizeParticle * 2 + 31) / 32 * 32; testSizeSub <= options.SubVolumeSize; testSizeSub += 32)
                {
                    int TestSizeUseful = Math.Max(1, testSizeSub - SizeParticle * 2);
                    int3 TestGrid = (DimsVolumeScaled - SizeParticle + TestSizeUseful - 1) / TestSizeUseful;
                    long TestVoxels = TestGrid.Elements() * testSizeSub * testSizeSub * testSizeSub;

                    if (TestVoxels < BestVoxels)
                    {
                        BestVoxels = TestVoxels;
                        BestSizeSub = testSizeSub;
                    }
                }

                SizeSub = BestSizeSub;

                progressCallback?.Invoke(new int3(1), 0, $"Using {BestSizeSub} sub-volumes for matching, resulting in {((float)BestVoxels / DimsVolumeScaled.Elements() * 100 - 100):F0} % overhead");
            }

            int SizeSubPadded = SizeSub * 2;
            int SizeUseful = SizeSub - SizeParticle * 2;// Math.Min(SizeSub / 2, SizeSub - SizeParticle * 2);// Math.Min(SizeSub - SizeParticle, SizeSub / 2);

            int3 Grid = (DimsVolumeScaled - SizeParticle + SizeUseful - 1) / SizeUseful;
            List<float3> GridCoords = new List<float3>();
            for (int z = 0; z < Grid.Z; z++)
                for (int x = 0; x < Grid.X; x++)
                    for (int y = 0; y < Grid.Y; y++)
                        GridCoords.Add(new float3(x * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                                                    y * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                                                    z * SizeUseful + SizeUseful / 2 + SizeParticle / 2));

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), $"Using {Grid} sub-volumes");

            #endregion

            #region Get correlation and angles either by calculating them from scratch, or by loading precalculated volumes

            string CorrVolumePath = IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_corr.mrc");
            string AngleIDVolumePath = IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_angleid.tif");

            if (!File.Exists(IOPath.Combine(ReconstructionDir, NameWithRes + ".mrc")))
                throw new FileNotFoundException("A reconstruction at the desired resolution was not found.");

            Image TomoRec = null;

            if (!File.Exists(CorrVolumePath) || !options.ReuseCorrVolumes)
            {
                progressCallback?.Invoke(Grid, 0, "Loading...");

                TomoRec = Image.FromFile(IOPath.Combine(ReconstructionDir, NameWithRes + ".mrc"));

                CorrVolume = new Image(DimsVolumeScaled);
                CorrData = CorrVolume.GetHost(Intent.ReadWrite);

                AngleIDVolume = new Image(DimsVolumeScaled);
                AngleIDData = AngleIDVolume.GetHost(Intent.ReadWrite);

                float[] SpectrumWhitening = new float[128];

                if (options.WhitenSpectrum)
                {
                    Image CTFZero;
                    {
                        Projector Reconstructor = new Projector(new int3(256), 1);
                        Image OnesComplex = new Image(IntPtr.Zero, new int3(256, 256, 1), true, true);
                        OnesComplex.Fill(new float2(1, 0));
                        Image Ones = OnesComplex.AsReal();
                        Reconstructor.BackProject(OnesComplex,
                                                  Ones,
                                                  GetAnglesInOneTilt([VolumeDimensionsPhysical * 0.5f], [new float3(0)], IndicesSortedDose[0]),
                                                  Matrix2.Identity());
                        OnesComplex.Dispose();
                        Ones.Dispose();
                        Reconstructor.Weights.Fill(1);
                        CTFZero = Reconstructor.Reconstruct(true, "C1", null, -1, -1, -1, 0);
                        Reconstructor.Dispose();

                        CTFZero = CTFZero.AsScaledCTF(TomoRec.Dims).AndDisposeParent();
                    }

                    Image TomoAmps = TomoRec.GetCopyGPU();
                    TomoAmps.MaskRectangularly(TomoAmps.Dims - 64, 32, true);
                    TomoAmps = TomoAmps.AsFFT(true).AndDisposeParent().AsAmplitudes().AndDisposeParent();

                    int NBins = 128;// Math.Max(SizeSub / 2, TomoRec.Dims.Max() / 2);
                    double[] Sums = new double[NBins];
                    double[] Samples = new double[NBins];

                    float[][] TomoData = TomoAmps.GetHost(Intent.Read);
                    float[][] CTFData = CTFZero.GetHost(Intent.Read);
                    Helper.ForEachElementFT(TomoAmps.Dims, (x, y, z, xx, yy, zz) =>
                    {
                        float CTF = MathF.Abs(CTFData[z][y * (CTFZero.Dims.X / 2 + 1) + x]);
                        if (CTF < 1e-2f)
                            return;

                        float xnorm = (float)xx / TomoAmps.Dims.X * 2;
                        float ynorm = (float)yy / TomoAmps.Dims.Y * 2;
                        float znorm = (float)zz / TomoAmps.Dims.Z * 2;
                        float R = MathF.Sqrt(xnorm * xnorm + ynorm * ynorm + znorm * znorm);
                        if (R >= 1)
                            return;

                        R *= Sums.Length;
                        int ID = (int)R;
                        float W1 = R - ID;
                        float W0 = 1f - W1;

                        float Val = TomoData[z][y * (TomoAmps.Dims.X / 2 + 1) + x];
                        Val *= Val;

                        if (W0 > 0)
                        {
                            Sums[ID] += W0 * Val * CTF;
                            Samples[ID] += W0 * CTF;
                        }

                        if (ID < Sums.Length - 1 && W1 > 0)
                        {
                            Sums[ID + 1] += W1 * Val * CTF;
                            Samples[ID + 1] += W1 * CTF;
                        }
                    });

                    TomoAmps.Dispose();
                    CTFZero.Dispose();

                    for (int i = 0; i < Sums.Length; i++)
                        Sums[i] = Math.Sqrt(Sums[i] / Math.Max(1e-6, Samples[i]));

                    Sums[Sums.Length - 1] = Sums[Sums.Length - 3];
                    Sums[Sums.Length - 2] = Sums[Sums.Length - 3];

                    SpectrumWhitening = Sums.Select(v => 1 / MathF.Max(1e-10f, (float)v)).ToArray();
                    float Max = MathF.Max(1e-10f, SpectrumWhitening.Max());
                    SpectrumWhitening = SpectrumWhitening.Select(v => v / Max).ToArray();

                    TomoRec = TomoRec.AsSpectrumMultiplied(true, SpectrumWhitening).AndDisposeParent();
                    //TomoRec.WriteMRC("d_tomorec_whitened.mrc", true);
                }

                if (options.Lowpass < 0.999M)
                {
                    TomoRec.BandpassGauss(0, (float)options.Lowpass, true, (float)options.LowpassSigma);
                    //TomoRec.WriteMRC("d_tomorec_lowpass.mrc", true);
                }
                TomoRec.Bandpass(2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 1.5f, 2, true, 2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 0.5f);
                //TomoRec.WriteMRC("d_tomorec_highpass.mrc", true);

                #region Scale and pad/crop the template to the right size, create projector

                progressCallback?.Invoke(Grid, 0, "Preparing template...");

                Projector ProjectorReference, ProjectorMask, ProjectorRandom;
                Image TemplateMask;
                int TemplateMaskSum = 0;
                {
                    int SizeBinned = (int)Math.Round(template.Dims.X * (options.TemplatePixel / options.BinnedPixelSizeMean) / 2) * 2;

                    Image TemplateScaled = template.AsScaled(new int3(SizeBinned));
                    template.FreeDevice();

                    GPU.SphereMask(TemplateScaled.GetDevice(Intent.Read),
                                   TemplateScaled.GetDevice(Intent.Write),
                                   TemplateScaled.Dims,
                                   SizeParticle / 2,
                                   Math.Max(5, 20 / (float)options.BinnedPixelSizeMean),
                                   false,
                                   1);

                    float TemplateMax = TemplateScaled.GetHost(Intent.Read).Select(a => a.Max()).Max();
                    TemplateMask = TemplateScaled.GetCopyGPU();
                    TemplateMask.Binarize(TemplateMax * 0.2f);
                    TemplateMask = TemplateMask.AsDilatedMask(1, true).AndDisposeParent();
                    //TemplateMask.WriteMRC("d_templatemask.mrc", true);

                    TemplateScaled.Multiply(TemplateMask);
                    TemplateScaled.NormalizeWithinMask(TemplateMask, false);
                    //TemplateScaled.WriteMRC("d_template_norm.mrc", true);

                    #region Make phase-randomized template

                    if (false)
                    {
                        Random Rng = new Random(123);
                        RandomNormal RngN = new RandomNormal(123);
                        Image TemplateRandomFT = TemplateScaled.AsFFT(true);
                        TemplateRandomFT.TransformComplexValues(v =>
                        {
                            float Amp = v.Length() / TemplateRandomFT.Dims.Elements();
                            float Phase = Rng.NextSingle() * MathF.PI * 2;
                            return new float2(Amp * MathF.Cos(Phase), Amp * MathF.Sin(Phase));
                        });
                        TemplateRandomFT.Bandpass(0, 1, true, 0.01f);
                        //GPU.SymmetrizeFT(TemplateRandomFT.GetDevice(Intent.ReadWrite), TemplateRandomFT.Dims, options.Symmetry);
                        Image TemplateRandom = TemplateRandomFT.AsIFFT(true).AndDisposeParent();
                        TemplateRandom.TransformValues(v => RngN.NextSingle(0, 1));
                        TemplateRandomFT = TemplateRandom.AsFFT(true).AndDisposeParent();
                        GPU.SymmetrizeFT(TemplateRandomFT.GetDevice(Intent.ReadWrite), TemplateRandomFT.Dims, options.Symmetry);
                        TemplateRandom = TemplateRandomFT.AsIFFT(true).AndDisposeParent();
                        TemplateRandom.Multiply(TemplateMask);
                        TemplateRandom.WriteMRC("d_templaterandom.mrc", true);

                        {
                            Image TemplateAmps = TemplateScaled.AsFFT(true).AsAmplitudes().AndDisposeParent();
                            Image RandomAmps = TemplateRandom.AsFFT(true).AsAmplitudes().AndDisposeParent();
                            RandomAmps.Max(1e-16f);
                            Image RandomPhases = TemplateRandom.AsFFT(true);
                            RandomPhases.Divide(RandomAmps);
                            RandomAmps.Dispose();

                            RandomPhases.Multiply(TemplateAmps);
                            TemplateAmps.Dispose();

                            TemplateRandom = RandomPhases.AsIFFT(true).AndDisposeParent();
                            TemplateRandom.Multiply(TemplateMask);
                        }

                        TemplateRandom.NormalizeWithinMask(TemplateMask, true);
                        //TemplateRandom.WriteMRC("d_templaterandom_norm.mrc", true);

                        Image TemplateRandomPadded = TemplateRandom.AsPadded(new int3(SizeSub)).AndDisposeParent();

                        if (options.WhitenSpectrum)
                            TemplateRandomPadded = TemplateRandomPadded.AsSpectrumMultiplied(true, SpectrumWhitening).AndDisposeParent();

                        TemplateRandomPadded.Bandpass(2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 1.5f, 
                                                      2, true, 
                                                      2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 0.5f);
                    }

                    #endregion

                    Image TemplatePadded = TemplateScaled.AsPadded(new int3(SizeSub)).AndDisposeParent();
                    //TemplatePadded.WriteMRC("d_template.mrc", true);

                    Image TemplateMaskPadded = TemplateMask.AsPadded(new int3(SizeSub));

                    TemplateMaskSum = (int)TemplateMask.GetHost(Intent.Read).Select(a => a.Sum()).Sum();
                    TemplateMask.Multiply(1f / TemplateMaskSum);

                    if (options.WhitenSpectrum)
                    {
                        TemplatePadded = TemplatePadded.AsSpectrumMultiplied(true, SpectrumWhitening).AndDisposeParent();
                        //TemplatePadded.WriteMRC("d_template_whitened.mrc", true);
                    }

                    if (options.Lowpass < 0.999M)
                        TemplatePadded.BandpassGauss(0, (float)options.Lowpass, true, (float)options.LowpassSigma);

                    TemplatePadded.Bandpass(2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 1.5f, 2, true, 2 * (float)(options.BinnedPixelSizeMean / options.TemplateDiameter) * 0.5f);
                    //TemplatePadded.WriteMRC("d_template_highpass.mrc", true);

                    //TemplateRandomPadded = TemplateRandomPadded.AsSpectrumMultiplied(true, Sinc2).AndDisposeParent();
                    //TemplateRandomPadded.WriteMRC("d_templaterandom_filtered.mrc");

                    //new Star(TemplatePadded.AsAmplitudes1D(true, 1, 64), "wrpAmplitudes").Save("d_template_amplitudes.star");
                    //new Star(TemplateRandomPadded.AsAmplitudes1D(true, 1, 64), "wrpAmplitudes").Save("d_templaterandom_amplitudes.star");

                    ProjectorReference = new Projector(TemplatePadded, 2, true, 3);
                    TemplatePadded.Dispose();
                    ProjectorReference.PutTexturesOnDevice();

                    //ProjectorMask = new Projector(TemplateMaskPadded, 2, true, 3);
                    //TemplateMaskPadded.Dispose();
                    //ProjectorMask.PutTexturesOnDevice();

                    //ProjectorRandom = new Projector(TemplateRandomPadded, 2, true, 3);
                    //TemplateRandomPadded.Dispose();
                    //ProjectorRandom.PutTexturesOnDevice();
                }

                #endregion

                #region Preflight

                if (TomoRec.Dims != DimsVolumeScaled)
                    throw new DimensionMismatchException($"Tomogram dimensions ({TomoRec.Dims}) don't match expectation ({DimsVolumeScaled})");

                //if (options.WhitenSpectrum)
                //{
                //    progressCallback?.Invoke(Grid, 0, "Whitening tomogram spectrum...");

                //    TomoRec.WriteMRC("d_tomorec.mrc", true);
                //    TomoRec = TomoRec.AsSpectrumFlattened(true, 0.99f).AndDisposeParent();
                //    TomoRec.WriteMRC("d_tomorec_whitened.mrc", true);
                //}

                float[][] TomoRecData = TomoRec.GetHost(Intent.Read);

                int PlanForw, PlanBack, PlanForwCTF;
                Projector.GetPlans(new int3(SizeSub), 3, out PlanForw, out PlanBack, out PlanForwCTF);

                Image CTFCoords = CTF.GetCTFCoords(SizeSubPadded, SizeSubPadded);

                #endregion

                #region Match

                progressCallback?.Invoke(Grid, 0, "Matching...");

                int BatchSize = Grid.Y;
                float[] ProgressFraction = new float[1];
                for (int b = 0; b < GridCoords.Count; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, GridCoords.Count - b);

                    Image Subtomos = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch), true, true);

                    #region Create CTF for this column of subvolumes (X = const, Z = const)

                    Image SubtomoCTF;
                    {
                        Image CTFs = GetCTFsForOneParticle(options, GridCoords[b], CTFCoords, null, true, false, false);
                        //CTFs.Fill(1);
                        Image CTFsAbs = GetCTFsForOneParticle(options, GridCoords[b], CTFCoords, null, false, false, false);
                        CTFsAbs.Abs();

                        // CTF has to be converted to complex numbers with imag = 0, and weighted by itself

                        Image CTFsComplex = new Image(CTFs.Dims, true, true);
                        CTFsComplex.Fill(new float2(1, 0));
                        CTFsComplex.Multiply(CTFs);
                        CTFsComplex.Multiply(CTFs);
                        //if (b == 0)
                        //    CTFsComplex.AsAmplitudes().WriteMRC("d_ctfs.mrc", true);

                        // Back-project and reconstruct
                        Projector ProjCTF = new Projector(new int3(SizeSubPadded), 1);
                        Projector ProjCTFWeights = new Projector(new int3(SizeSubPadded), 1);

                        //ProjCTF.Weights.Fill(0.01f);

                        ProjCTF.BackProject(CTFsComplex, CTFsAbs, GetAngleInAllTilts(GridCoords[b]), MagnificationCorrection);

                        CTFsAbs.Fill(1);
                        ProjCTFWeights.BackProject(CTFsComplex, CTFsAbs, GetAngleInAllTilts(GridCoords[b]), MagnificationCorrection);
                        ProjCTFWeights.Weights.Min(1);
                        ProjCTF.Data.Multiply(ProjCTFWeights.Weights);
                        //ProjCTF.Weights.Fill(1);

                        CTFsComplex.Dispose();
                        ProjCTFWeights.Dispose();

                        Image PSF = ProjCTF.Reconstruct(false, "C1", null, -1, -1, -1, 0);
                        //PSF.WriteMRC("d_psf.mrc", true);
                        PSF.RemapToFT(true);
                        ProjCTF.Dispose();

                        SubtomoCTF = PSF.AsPadded(new int3(SizeSub), true).AndDisposeParent().
                                         AsFFT(true).AndDisposeParent().
                                         AsReal().AndDisposeParent();
                        SubtomoCTF.Multiply(1f / (SizeSubPadded * SizeSubPadded));

                        CTFs.Dispose();
                        CTFsAbs.Dispose();
                    }
                    //SubtomoCTF.WriteMRC("d_ctf.mrc", true);

                    #endregion

                    #region Extract subvolumes and store their FFTs

                    for (int st = 0; st < CurBatch; st++)
                    {
                        float[][] SubtomoData = new float[SizeSub][];

                        int XStart = (int)GridCoords[b + st].X - SizeSub / 2;
                        int YStart = (int)GridCoords[b + st].Y - SizeSub / 2;
                        int ZStart = (int)GridCoords[b + st].Z - SizeSub / 2;
                        for (int z = 0; z < SizeSub; z++)
                        {
                            SubtomoData[z] = new float[SizeSub * SizeSub];
                            int zz = ZStart + z;

                            for (int y = 0; y < SizeSub; y++)
                            {
                                int yy = YStart + y;
                                for (int x = 0; x < SizeSub; x++)
                                {
                                    int xx = XStart + x;
                                    if (xx >= 0 && xx < TomoRec.Dims.X && 
                                        yy >= 0 && yy < TomoRec.Dims.Y && 
                                        zz >= 0 && zz < TomoRec.Dims.Z)
                                        SubtomoData[z][y * SizeSub + x] = TomoRecData[zz][yy * TomoRec.Dims.X + xx];
                                    else
                                        SubtomoData[z][y * SizeSub + x] = 0;
                                }
                            }
                        }

                        Image Subtomo = new Image(SubtomoData, new int3(SizeSub));

                        // Re-use FFT plan created previously for CTF reconstruction since it has the right size
                        GPU.FFT(Subtomo.GetDevice(Intent.Read),
                                Subtomos.GetDeviceSlice(SizeSub * st, Intent.Write),
                                Subtomo.Dims,
                                1,
                                PlanForwCTF);

                        Subtomo.Dispose();
                    }
                    //Subtomos.Multiply(1f / (SizeSub * SizeSub * SizeSub));

                    #endregion

                    #region Perform correlation

                    //if (b == 0)
                    //    SubtomoCTF.WriteMRC16b("d_ctf.mrc", true);

                    Timer ProgressTimer = new Timer((a) =>
                        progressCallback?.Invoke(Grid, b + ProgressFraction[0] * CurBatch, "Matching..."), null, 1000, 1000);

                    Image BestCorrelation = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));
                    Image BestAngle = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));

                    GPU.CorrelateSubTomos(ProjectorReference.t_DataRe,
                                          ProjectorReference.t_DataIm,
                                          ProjectorReference.Oversampling,
                                          ProjectorReference.Data.Dims,
                                          Subtomos.GetDevice(Intent.Read),
                                          SubtomoCTF.GetDevice(Intent.Read),
                                          new int3(SizeSub),
                                          (uint)CurBatch,
                                          Helper.ToInterleaved(HealpixAngles),
                                          (uint)HealpixAngles.Length,
                                          (uint)options.BatchAngles,
                                          SizeParticle / 2,
                                          BestCorrelation.GetDevice(Intent.Write),
                                          BestAngle.GetDevice(Intent.Write),
                                          ProgressFraction);


                    //Image BestCorrelationRandom = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));
                    //Image BestAngleRandom = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));

                    //GPU.CorrelateSubTomos(ProjectorRandom.t_DataRe,
                    //                      ProjectorRandom.t_DataIm,
                    //                      ProjectorMask.t_DataRe,
                    //                      ProjectorMask.t_DataIm,
                    //                      ProjectorRandom.Oversampling,
                    //                      ProjectorRandom.Data.Dims,
                    //                      Subtomos.GetDevice(Intent.Read),
                    //                      SubtomoCTF.GetDevice(Intent.Read),
                    //                      new int3(SizeSub),
                    //                      (uint)CurBatch,
                    //                      Helper.ToInterleaved(HealpixAngles),
                    //                      (uint)HealpixAngles.Length,
                    //                      (uint)options.BatchAngles,
                    //                      SizeParticle / 2,
                    //                      BestCorrelationRandom.GetDevice(Intent.Write),
                    //                      BestAngleRandom.GetDevice(Intent.Write),
                    //                      IntPtr.Zero,
                    //                      ProgressFraction);

                    //BestCorrelation.WriteMRC($"d_bestcorr_{b:D2}.mrc", true);
                    //BestCorrelationRandom.WriteMRC($"d_bestcorr_random_{b:D2}.mrc", true);

                    //BestCorrelation.Subtract(BestCorrelationRandom);

                    #endregion

                    #region Put correlation values and best angle IDs back into the large volume

                    for (int st = 0; st < CurBatch; st++)
                    {
                        Image ThisCorrelation = new Image(BestCorrelation.GetDeviceSlice(SizeSub * st, Intent.Read), new int3(SizeSub));
                        Image CroppedCorrelation = ThisCorrelation.AsPadded(new int3(SizeUseful)).AndDisposeParent();

                        Image ThisAngle = new Image(BestAngle.GetDeviceSlice(SizeSub * st, Intent.Read), new int3(SizeSub));
                        Image CroppedAngle = ThisAngle.AsPadded(new int3(SizeUseful)).AndDisposeParent();

                        float[] SubCorr = CroppedCorrelation.GetHostContinuousCopy();
                        float[] SubAngle = CroppedAngle.GetHostContinuousCopy();
                        int3 Origin = new int3(GridCoords[b + st]) - SizeUseful / 2;
                        float Norm = 1f;// / (SizeSub * SizeSub * SizeSub * SizeSub);
                        for (int z = 0; z < SizeUseful; z++)
                        {
                            int zVol = Origin.Z + z;
                            if (zVol >= DimsVolumeScaled.Z - SizeParticle / 2)
                                continue;

                            for (int y = 0; y < SizeUseful; y++)
                            {
                                int yVol = Origin.Y + y;
                                if (yVol >= DimsVolumeScaled.Y - SizeParticle / 2)
                                    continue;

                                for (int x = 0; x < SizeUseful; x++)
                                {
                                    int xVol = Origin.X + x;
                                    if (xVol >= DimsVolumeScaled.X - SizeParticle / 2)
                                        continue;

                                    CorrData[zVol][yVol * DimsVolumeScaled.X + xVol] = SubCorr[(z * SizeUseful + y) * SizeUseful + x] * Norm;
                                    AngleIDData[zVol][yVol * DimsVolumeScaled.X + xVol] = SubAngle[(z * SizeUseful + y) * SizeUseful + x];
                                }
                            }
                        }

                        CroppedCorrelation.Dispose();
                        CroppedAngle.Dispose();
                    }

                    #endregion

                    Subtomos.Dispose();
                    SubtomoCTF.Dispose();

                    BestCorrelation.Dispose();
                    BestAngle.Dispose();
                    //BestCorrelationRandom.Dispose();
                    //BestAngleRandom.Dispose();

                    ProgressTimer.Dispose();
                    if (progressCallback != null)
                        IsCanceled = progressCallback(Grid, b + CurBatch, "Matching...");
                }

                #endregion

                #region Postflight

                GPU.DestroyFFTPlan(PlanForw);
                GPU.DestroyFFTPlan(PlanBack);
                GPU.DestroyFFTPlan(PlanForwCTF);

                CTFCoords.Dispose();
                ProjectorReference.Dispose();
                //ProjectorRandom.Dispose();
                //ProjectorMask.Dispose();

                #region Normalize by local standard deviation of TomoRec

                if (true)
                {
                    Image LocalStd = new Image(IntPtr.Zero, TomoRec.Dims);
                    GPU.LocalStd(TomoRec.GetDevice(Intent.Read),
                                 TomoRec.Dims,
                                 SizeParticle / 2,
                                 LocalStd.GetDevice(Intent.Write),
                                 IntPtr.Zero,
                                 0,
                                 0);

                    Image Center = LocalStd.AsPadded(LocalStd.Dims / 2);
                    float Median = Center.GetHostContinuousCopy().Median();
                    Center.Dispose();

                    LocalStd.Max(MathF.Max(1e-10f, Median));

                    //LocalStd.WriteMRC("d_localstd.mrc", true);

                    CorrVolume.Divide(LocalStd);

                    LocalStd.Dispose();
                }

                #endregion

                #region Normalize by background correlation std

                if (options.NormalizeScores)
                { 
                    Image Center = CorrVolume.AsPadded(CorrVolume.Dims / 2);
                    Center.Abs();
                    float[] Sorted = Center.GetHostContinuousCopy().OrderBy(v => v).ToArray();
                    float Percentile = Sorted[(int)(Sorted.Length * 0.68f)];
                    Center.Dispose();

                    CorrVolume.Multiply(1f / MathF.Max(1e-20f, Percentile));
                }

                #endregion

                #region Zero out correlation values not fully covered by desired number of tilts

                if (options.KeepOnlyFullVoxels)
                {
                    progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Trimming...");

                    float BinnedAngPix = (float)options.BinnedPixelSizeMean;
                    float Margin = (float)options.TemplateDiameter;

                    int Undersample = 4;
                    int3 DimsUndersampled = (DimsVolumeScaled + Undersample - 1) / Undersample;

                    float3[] ImagePositions = new float3[DimsUndersampled.ElementsSlice() * NTilts];
                    float3[] VolumePositions = new float3[DimsUndersampled.ElementsSlice()];
                    for (int y = 0; y < DimsUndersampled.Y; y++)
                        for (int x = 0; x < DimsUndersampled.X; x++)
                            VolumePositions[y * DimsUndersampled.X + x] = new float3((x + 0.5f) * Undersample * BinnedAngPix,
                                                                                     (y + 0.5f) * Undersample * BinnedAngPix, 
                                                                                     0);

                    float[][] OccupancyMask = Helper.ArrayOfFunction(z => Helper.ArrayOfConstant(1f, VolumePositions.Length), DimsUndersampled.Z);

                    float WidthNoMargin = ImageDimensionsPhysical.X - BinnedAngPix - Margin;
                    float HeightNoMargin = ImageDimensionsPhysical.Y - BinnedAngPix - Margin;

                    for ( int z = 0; z < DimsUndersampled.Z; z++)
                    {
                        float ZCoord = (z + 0.5f) * Undersample * BinnedAngPix;
                        for (int i = 0; i < VolumePositions.Length; i++)
                            VolumePositions[i].Z = ZCoord;

                        ImagePositions = GetPositionInAllTiltsNoLocalWarp(VolumePositions, ImagePositions);

                        for (int p = 0; p < VolumePositions.Length; p++)
                        {
                            for (int t = 0; t < NTilts; t++)
                            {
                                int i = p * NTilts + t;

                                if (UseTilt[t] &&
                                    (ImagePositions[i].X < Margin || ImagePositions[i].Y < Margin ||
                                    ImagePositions[i].X > WidthNoMargin ||
                                    ImagePositions[i].Y > HeightNoMargin))
                                {
                                    OccupancyMask[z][p] = 0;
                                    break;
                                }
                            }
                        }
                    }

                    CorrData = CorrVolume.GetHost(Intent.ReadWrite);
                    AngleIDData = AngleIDVolume.GetHost(Intent.ReadWrite);

                    for (int z = 0; z < DimsVolumeScaled.Z; z++)
                    {
                        int zz = z / Undersample;
                        for (int y = 0; y < DimsVolumeScaled.Y; y++)
                        {
                            int yy = y / Undersample;
                            for (int x = 0; x < DimsVolumeScaled.X; x++)
                            {
                                int xx = x / Undersample;
                                CorrData[z][y * DimsVolumeScaled.X + x] *= OccupancyMask[zz][yy * DimsUndersampled.X + xx];
                                AngleIDData[z][y * DimsVolumeScaled.X + x] *= OccupancyMask[zz][yy * DimsUndersampled.X + xx];
                            }
                        }
                    }
                }

                #endregion

                progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Saving global scores...");

                // Store correlation values and angle IDs for re-use later
                CorrVolume.WriteMRC16b(CorrVolumePath, (float)options.BinnedPixelSizeMean, true);
                AngleIDVolume.WriteTIFF(AngleIDVolumePath, (float)options.BinnedPixelSizeMean, typeof(float));

                #endregion
            }
            else
            {
                progressCallback?.Invoke(Grid, 0, "Loading...");

                TomoRec = Image.FromFile(IOPath.Combine(ReconstructionDir, NameWithRes + ".mrc"));

                if (!File.Exists(CorrVolumePath))
                    throw new FileNotFoundException("Pre-existing correlation volume not found.");

                if (!File.Exists(AngleIDVolumePath))
                    throw new FileNotFoundException("Pre-existing angle ID volume not found.");

                CorrVolume = Image.FromFile(CorrVolumePath);
                CorrData = CorrVolume.GetHost(Intent.Read);

                AngleIDVolume = Image.FromFile(AngleIDVolumePath);
                AngleIDData = AngleIDVolume.GetHost(Intent.Read);
            }

            //CorrImage?.Dispose();

            #endregion

            #region Get peak list that has at least NResults values

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Extracting best peaks...");

            int3[] InitialPeaks = new int3[0];
            {
                float Max = CorrVolume.GetHostContinuousCopy().Max();

                for (float s = Max * 0.9f; s > Max * 0.1f; s -= Max * 0.05f)
                {
                    float Threshold = s;
                    InitialPeaks = CorrVolume.GetLocalPeaks(PeakDistance, Threshold);

                    if (InitialPeaks.Length >= options.NResults)
                        break;
                }
            }

            #endregion

            #region Write out images for quickly assessing different thresholds for picking

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Preparing visualizations...");

            int TemplateThicknessPixels = (int)((float)options.TemplateDiameter / TomoRec.PixelSize);
            
            // extract projection over central slices of tomogram
            int ZThickness = Math.Max(1, (int)((float)options.TemplateDiameter / TomoRec.PixelSize));
            int ZCenter = (int)(TomoRec.Dims.Z / 2);
            int _ZMin = (int)(ZCenter - (int)((float)ZThickness / 2));
            int _ZMax = (int)(ZCenter + (int)((float)ZThickness / 2));
            Image TomogramSlice = TomoRec.AsRegion(
                origin: new int3(0, 0, _ZMin),
                dimensions: new int3(TomoRec.Dims.X, TomoRec.Dims.Y, ZThickness)
            ).AsReducedAlongZ().AndDisposeParent();
            
            // write images showing particle picks at different thresholds
            float[] AllPeakScores = new float[InitialPeaks.Count()];
            float[] Thresholds = { 3f, 4f, 5f, 6f, 7f, 8f, 9f };
            string PickingImageDirectory = IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_picks");
            Directory.CreateDirectory(PickingImageDirectory);

            float2 MeanStd;
            {
                Image CentralQuarter = TomogramSlice.AsPadded(new int2(TomogramSlice.Dims) / 2);
                MeanStd = MathHelper.MeanAndStd(CentralQuarter.GetHost(Intent.Read)[0]);
                CentralQuarter.Dispose();
            }
            float SliceMin = MeanStd.X - MeanStd.Y * 3;
            float SliceMax = MeanStd.X + MeanStd.Y * 3;
            TomogramSlice.TransformValues(v => (v - SliceMin) / (SliceMax - SliceMin) * 255);
            
            foreach (float threshold in Thresholds)
            {
                // get positions with score >= thresold
                for (int i = 0; i < InitialPeaks.Count(); i++)
                {
                    float3 Position = new float3(InitialPeaks[i]);
                    AllPeakScores[i] = CorrData[InitialPeaks[i].Z][InitialPeaks[i].Y * DimsVolumeScaled.X + InitialPeaks[i].X];
                }

                var filteredPositions = InitialPeaks.Zip(AllPeakScores, (position, score) => new { Position = position, Score = score })
                    .Where(item => item.Score >= threshold)
                    .Where(item => (item.Position.Z >= _ZMin && item.Position.Z <= _ZMax))
                    .Select(item => item.Position)
                    .ToArray();

                // write PNG with image and draw particle circles
                using (SKBitmap SliceImage = new SKBitmap(TomogramSlice.Dims.X, TomogramSlice.Dims.Y, SKColorType.Bgra8888, SKAlphaType.Opaque))
                {
                    float[] SliceData = TomogramSlice.GetHost(Intent.Read)[0];

                    for (int y = 0; y < TomogramSlice.Dims.Y; y++)
                    {
                        for (int x = 0; x < TomogramSlice.Dims.X; x++)
                        {
                            int i = y * TomogramSlice.Dims.X + x;
                            byte PixelValue = (byte)Math.Max(0, Math.Min(255, SliceData[(TomogramSlice.Dims.Y - 1 - y) * TomogramSlice.Dims.X + x]));
                            var color = new SKColor(PixelValue, PixelValue, PixelValue, 255); // Alpha is set to 255 for opaque
                            SliceImage.SetPixel(x, y, color);
                        }
                    }

                    using (SKCanvas canvas = new SKCanvas(SliceImage))
                    {
                        SKPaint paint = new SKPaint
                        {
                            Color = SKColors.Yellow,
                            IsAntialias = true,
                            Style = SKPaintStyle.Stroke, // Change to Fill for filled circles
                            StrokeWidth = 1.25f
                        };

                        foreach (var position in filteredPositions)
                        {
                            float radius = (((float)options.TemplateDiameter / 2f) * 1.0f) / TomoRec.PixelSize;
                            canvas.DrawCircle(position.X, TomogramSlice.Dims.Y - position.Y, radius: radius, paint);
                        }
                    }

                    string ThresholdedPicksImagePath = Helper.PathCombine(PickingImageDirectory, $"{NameWithRes}_{options.TemplateName}_threshold_{threshold}.png");
                    using (Stream s = File.Create(ThresholdedPicksImagePath))
                    { 
                        SliceImage.Encode(s, SKEncodedImageFormat.Png, 100);
                    }

                }
            }

            TomogramSlice.Dispose();

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Done...");

            #endregion

            TomoRec.Dispose();

            #region Write peak positions and angles into table

            Star TableOut = new Star(new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnCoordinateZ",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnMicrographName",
                "rlnAutopickFigureOfMerit"
            });

            {
                for (int n = 0; n < InitialPeaks.Length; n++)
                {
                    //float3 Position = RefinedPositions[n] / new float3(DimsVolumeCropped);
                    //float Score = RefinedScores[n];
                    //float3 Angle = RefinedAngles[n] * Helper.ToDeg;

                    float3 Position = new float3(InitialPeaks[n]);
                    float Score = CorrData[(int)Position.Z][(int)Position.Y * DimsVolumeScaled.X + (int)Position.X];
                    float3 Angle = HealpixAngles[(int)(AngleIDData[(int)Position.Z][(int)Position.Y * DimsVolumeScaled.X + (int)Position.X] + 0.5f)] * Helper.ToDeg;
                    Position /= new float3(DimsVolumeScaled);

                    TableOut.AddRow(new string[]
                    {
                        Position.X.ToString(CultureInfo.InvariantCulture),
                        Position.Y.ToString(CultureInfo.InvariantCulture),
                        Position.Z.ToString(CultureInfo.InvariantCulture),
                        Angle.X.ToString(CultureInfo.InvariantCulture),
                        Angle.Y.ToString(CultureInfo.InvariantCulture),
                        Angle.Z.ToString(CultureInfo.InvariantCulture),
                        RootName + ".tomostar",
                        Score.ToString(CultureInfo.InvariantCulture)
                    });
                }
            }

            CorrVolume?.Dispose();

            TableOut.Save(IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + ".star"));

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Done.");

            #endregion
        }

        public void ReconstructFull(ProcessingOptionsTomoFullReconstruction options, Func<int3, int, string, bool> progressCallback)
        {
            int GPUID = GPU.GetDevice();

            bool IsCanceled = false;
            string NameWithRes = RootName + $"_{options.BinnedPixelSizeMean:F2}Apx";

            Directory.CreateDirectory(ReconstructionDir);

            if (options.DoDeconv)
                Directory.CreateDirectory(ReconstructionDeconvDir);

            if (options.PrepareDenoising)
            {
                Directory.CreateDirectory(ReconstructionOddDir);
                Directory.CreateDirectory(ReconstructionEvenDir);
                Directory.CreateDirectory(ReconstructionCTFDir);
            }

            if (File.Exists(IOPath.Combine(ReconstructionDir, NameWithRes + ".mrc")) && !options.OverwriteFiles)
                return;

            #region Dimensions

            VolumeDimensionsPhysical = options.DimensionsPhysical;

            int3 DimsVolumeCropped = new int3((int)Math.Round(options.DimensionsPhysical.X / (float)options.BinnedPixelSizeMean / 2) * 2,
                                              (int)Math.Round(options.DimensionsPhysical.Y / (float)options.BinnedPixelSizeMean / 2) * 2,
                                              (int)Math.Round(options.DimensionsPhysical.Z / (float)options.BinnedPixelSizeMean / 2) * 2);
            int SizeSub = options.SubVolumeSize;
            int SizeSubPadded = (int)(SizeSub * options.SubVolumePadding) * 2;

            #endregion

            #region Establish reconstruction positions

            int3 Grid = (DimsVolumeCropped + SizeSub - 1) / SizeSub;
            List<float3> GridCoords = new List<float3>();
            for (int z = 0; z < Grid.Z; z++)
                for (int y = 0; y < Grid.Y; y++)
                    for (int x = 0; x < Grid.X; x++)
                        GridCoords.Add(new float3(x * SizeSub + SizeSub / 2,
                                                  y * SizeSub + SizeSub / 2,
                                                  z * SizeSub + SizeSub / 2));

            progressCallback?.Invoke(Grid, 0, "Loading...");

            #endregion

            #region Load and preprocess tilt series

            Movie[] TiltMovies;
            Image[] TiltData, TiltDataOdd, TiltDataEven;
            Image[] TiltMasks;
            LoadMovieData(options, out TiltMovies, out TiltData, options.PrepareDenoising && options.PrepareDenoisingFrames, out TiltDataOdd, out TiltDataEven);
            LoadMovieMasks(options, out TiltMasks);
            Image[][] TiltDataPreprocess = options.PrepareDenoising && options.PrepareDenoisingFrames ?
                                           new[] { TiltData, TiltDataEven, TiltDataOdd } :
                                           new[] { TiltData };
            for (int z = 0; z < NTilts; z++)
            {
                for (int idata = 0; idata < TiltDataPreprocess.Length; idata++)
                {
                    EraseDirt(TiltDataPreprocess[idata][z], TiltMasks[z]);
                    if (idata == TiltDataPreprocess.Length - 1)
                        TiltMasks[z]?.FreeDevice();

                    if (options.Normalize)
                    {
                        TiltDataPreprocess[idata][z].SubtractMeanGrid(new int2(1));
                        TiltDataPreprocess[idata][z].MaskRectangularly(new int3(new int2(TiltDataPreprocess[idata][z].Dims) - 32), 16, false);
                        TiltDataPreprocess[idata][z].Bandpass(1f / (SizeSub * (float)options.SubVolumePadding / 2), 1f, false, 0f);

                        GPU.Normalize(TiltDataPreprocess[idata][z].GetDevice(Intent.Read),
                                      TiltDataPreprocess[idata][z].GetDevice(Intent.Write),
                                      (uint)TiltDataPreprocess[idata][z].ElementsReal,
                                      1);
                    }

                    if (options.Invert)
                        TiltDataPreprocess[idata][z].Multiply(-1f);

                    TiltDataPreprocess[idata][z].FreeDevice();
                }
            }

            #endregion

            #region Memory and FFT plan allocation

            Image CTFCoords = CTF.GetCTFCoords(SizeSubPadded, SizeSubPadded);

            float[][] OutputRec = Helper.ArrayOfFunction(i => new float[DimsVolumeCropped.ElementsSlice()], DimsVolumeCropped.Z);
            float[][] OutputRecDeconv = null;
            float[][][] OutputRecHalves = null;
            if (options.PrepareDenoising)
            {
                OutputRecHalves = new[] { Helper.ArrayOfFunction(i => new float[DimsVolumeCropped.ElementsSlice()], DimsVolumeCropped.Z),
                                          Helper.ArrayOfFunction(i => new float[DimsVolumeCropped.ElementsSlice()], DimsVolumeCropped.Z)};
            }

            int NThreads = 1;

            int[] PlanForw = new int[NThreads], PlanBack = new int[NThreads], PlanForwCTF = new int[NThreads];
            for (int i = 0; i < NThreads; i++)
                Projector.GetPlans(new int3(SizeSubPadded), 1, out PlanForw[i], out PlanBack[i], out PlanForwCTF[i]);
            int[] PlanForwParticle = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeSubPadded, SizeSubPadded, 1), (uint)NTilts), NThreads);
            Projector[] Projectors = Helper.ArrayOfFunction(i => new Projector(new int3(SizeSubPadded), 1), NThreads);
            Projector[] Correctors = Helper.ArrayOfFunction(i => new Projector(new int3(SizeSubPadded), 1), NThreads);

            Image[] Subtomo = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded)), NThreads);
            Image[] SubtomoCropped = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub)), NThreads);

            Image[] Images = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts)), NThreads);
            Image[] ImagesFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true, true), NThreads);
            Image[] ImagesFTHalf = options.PrepareDenoising ? Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true, true), NThreads) : null;
            Image[] CTFs = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true), NThreads);
            Image[] CTFsHalf = options.PrepareDenoising ? Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true), NThreads) : null;
            Image[] Samples = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true), NThreads);
            foreach (var samples in Samples)
                samples.Fill(1);
            Image[] SamplesHalf = null;
            if (options.PrepareDenoising)
            {
                SamplesHalf = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true), NThreads);
                foreach (var samples in SamplesHalf)
                    samples.Fill(1);
            }

            #endregion

            #region Reconstruction

            int NDone = 0;

            Helper.ForCPU(0, GridCoords.Count, NThreads,
                threadID => GPU.SetDevice(GPUID),
                (p, threadID) =>
                {
                    if (IsCanceled)
                        return;

                    float3 CoordsPhysical = GridCoords[p] * (float)options.BinnedPixelSizeMean;

                    GetImagesForOneParticle(options, TiltData, SizeSubPadded, CoordsPhysical, PlanForwParticle[threadID], -1, 8, Images[threadID], ImagesFT[threadID]);
                    GetCTFsForOneParticle(options, CoordsPhysical, CTFCoords, null, true, false, false, CTFs[threadID]);

                    ImagesFT[threadID].Multiply(CTFs[threadID]);    // Weight and phase-flip image FTs

                    // We want final amplitudes in reconstruction to remain B-fac weighted. 
                    // Thus we will divide B-fac and CTF-weighted data by unweighted CTF, i.e. not by B-fac
                    GetCTFsForOneParticle(options, CoordsPhysical, CTFCoords, null, false, false, false, CTFs[threadID]);
                    CTFs[threadID].Abs();                           // No need for Wiener, just phase flipping

                    #region Normal reconstruction

                    {
                        Projectors[threadID].Data.Fill(0);
                        Projectors[threadID].Weights.Fill(0);

                        Correctors[threadID].Data.Fill(0);
                        Correctors[threadID].Weights.Fill(0);

                        Projectors[threadID].BackProject(ImagesFT[threadID], CTFs[threadID], GetAngleInAllTilts(CoordsPhysical), MagnificationCorrection);

                        Correctors[threadID].BackProject(ImagesFT[threadID], Samples[threadID], GetAngleInAllTilts(CoordsPhysical), MagnificationCorrection);

                        Correctors[threadID].Weights.Min(1);
                        Projectors[threadID].Data.Multiply(Correctors[threadID].Weights);
                        Projectors[threadID].Weights.Max(0.01f); 

                        Projectors[threadID].Reconstruct(Subtomo[threadID].GetDevice(Intent.Write), false, "C1", null, PlanForw[threadID], PlanBack[threadID], PlanForwCTF[threadID], 0);

                        GPU.Pad(Subtomo[threadID].GetDevice(Intent.Read),
                                SubtomoCropped[threadID].GetDevice(Intent.Write),
                                new int3(SizeSubPadded),
                                new int3(SizeSub),
                                1);

                        float[][] SubtomoData = SubtomoCropped[threadID].GetHost(Intent.Read);

                        int3 Origin = new int3(GridCoords[p]) - SizeSub / 2;
                        for (int z = 0; z < SizeSub; z++)
                        {
                            int zVol = Origin.Z + z;
                            if (zVol >= DimsVolumeCropped.Z)
                                continue;

                            for (int y = 0; y < SizeSub; y++)
                            {
                                int yVol = Origin.Y + y;
                                if (yVol >= DimsVolumeCropped.Y)
                                    continue;

                                for (int x = 0; x < SizeSub; x++)
                                {
                                    int xVol = Origin.X + x;
                                    if (xVol >= DimsVolumeCropped.X)
                                        continue;

                                    OutputRec[zVol][yVol * DimsVolumeCropped.X + xVol] = SubtomoData[z][y * SizeSub + x];
                                }
                            }
                        }
                    }

                    #endregion

                    #region Odd/even tilt reconstruction

                    if (options.PrepareDenoising)
                    {
                        for (int ihalf = 0; ihalf < 2; ihalf++)
                        {
                            if (options.PrepareDenoisingTilts)
                            {
                                GPU.CopyDeviceToDevice(ImagesFT[threadID].GetDevice(Intent.Read),
                                                       ImagesFTHalf[threadID].GetDevice(Intent.Write),
                                                       ImagesFT[threadID].ElementsReal);
                                GPU.CopyDeviceToDevice(CTFs[threadID].GetDevice(Intent.Read),
                                                       CTFsHalf[threadID].GetDevice(Intent.Write),
                                                       CTFs[threadID].ElementsReal);
                                GPU.CopyDeviceToDevice(Samples[threadID].GetDevice(Intent.Read),
                                                       SamplesHalf[threadID].GetDevice(Intent.Write),
                                                       Samples[threadID].ElementsReal);
                                ImagesFTHalf[threadID].Multiply(Helper.ArrayOfFunction(i => i % 2 == ihalf ? 1f : 0f, NTilts));
                                CTFsHalf[threadID].Multiply(Helper.ArrayOfFunction(i => i % 2 == ihalf ? 1f : 0f, NTilts));
                                SamplesHalf[threadID].Multiply(Helper.ArrayOfFunction(i => i % 2 == ihalf ? 1f : 0f, NTilts));
                            }
                            else
                            {
                                GetImagesForOneParticle(options, ihalf == 0 ? TiltDataOdd : TiltDataEven, SizeSubPadded, CoordsPhysical, PlanForwParticle[threadID], -1, 8, Images[threadID], ImagesFTHalf[threadID]);
                                GetCTFsForOneParticle(options, CoordsPhysical, CTFCoords, null, true, false, false, CTFsHalf[threadID]);

                                ImagesFTHalf[threadID].Multiply(CTFsHalf[threadID]);    // Weight and phase-flip image FTs
                                CTFsHalf[threadID].Abs();                               // No need for Wiener, just phase flipping
                            }

                            Projectors[threadID].Data.Fill(0);
                            Projectors[threadID].Weights.Fill(0);

                            Correctors[threadID].Weights.Fill(0);

                            Projectors[threadID].BackProject(ImagesFTHalf[threadID], CTFsHalf[threadID], GetAngleInAllTilts(CoordsPhysical), MagnificationCorrection);

                            Correctors[threadID].BackProject(ImagesFTHalf[threadID], SamplesHalf[threadID], GetAngleInAllTilts(CoordsPhysical), MagnificationCorrection);

                            Correctors[threadID].Weights.Min(1);
                            Projectors[threadID].Data.Multiply(Correctors[threadID].Weights);
                            Projectors[threadID].Weights.Max(0.01f);

                            Projectors[threadID].Reconstruct(Subtomo[threadID].GetDevice(Intent.Write), false, "C1", null, PlanForw[threadID], PlanBack[threadID], PlanForwCTF[threadID], 0);

                            GPU.Pad(Subtomo[threadID].GetDevice(Intent.Read),
                                    SubtomoCropped[threadID].GetDevice(Intent.Write),
                                    new int3(SizeSubPadded),
                                    new int3(SizeSub),
                                    1);

                            float[][] SubtomoData = SubtomoCropped[threadID].GetHost(Intent.Read);

                            int3 Origin = new int3(GridCoords[p]) - SizeSub / 2;
                            for (int z = 0; z < SizeSub; z++)
                            {
                                int zVol = Origin.Z + z;
                                if (zVol >= DimsVolumeCropped.Z)
                                    continue;

                                for (int y = 0; y < SizeSub; y++)
                                {
                                    int yVol = Origin.Y + y;
                                    if (yVol >= DimsVolumeCropped.Y)
                                        continue;

                                    for (int x = 0; x < SizeSub; x++)
                                    {
                                        int xVol = Origin.X + x;
                                        if (xVol >= DimsVolumeCropped.X)
                                            continue;

                                        OutputRecHalves[ihalf][zVol][yVol * DimsVolumeCropped.X + xVol] = SubtomoData[z][y * SizeSub + x];
                                    }
                                }
                            }
                        }
                    }

                    #endregion

                    lock (OutputRec)
                        if (progressCallback != null)
                            IsCanceled = progressCallback(Grid, ++NDone, "Reconstructing...");
                }, null);

            // Make 3D CTF for the center of the full tomogram. This can be used to train a deconvolving denoiser.
            // The 3D CTF shouldn't have a missing wedge, so fill everything else with 1s instead of 0s.
            if (options.PrepareDenoising)
            {
                //CTF Center = GetTiltCTF(IndicesSortedAbsoluteAngle[0]);
                //Center.PixelSize = options.BinnedPixelSizeMean;
                //Center.Bfactor = 0;
                //Center.Scale = 1;

                //int Dim = 64;
                //float[] CTF1D = Center.Get1D(Dim / 2, false);
                //Image MapCTF = new Image(new int3(Dim, Dim, Dim), true);
                //{
                //    float[][] ItemData = MapCTF.GetHost(Intent.Write);
                //    Helper.ForEachElementFT(new int3(Dim), (x, y, z, xx, yy, zz, r) =>
                //    {
                //        int r0 = (int)r;
                //        int r1 = r0 + 1;
                //        float v0 = r0 < CTF1D.Length ? CTF1D[r0] : 1;
                //        float v1 = r1 < CTF1D.Length ? CTF1D[r1] : 1;
                //        float v = MathHelper.Lerp(v0, v1, r - r0);

                //        ItemData[z][y * (Dim / 2 + 1) + x] = v;
                //    });
                //}

                //MapCTF.WriteMRC(ReconstructionCTFDir + NameWithRes + ".mrc", (float)options.BinnedPixelSizeMean, true);

                int Dim = 256;

                float3[] ParticlePositions = Helper.ArrayOfConstant(VolumeDimensionsPhysical / 2f, NTilts);
                Image CTFCoords64 = CTF.GetCTFCoords(Dim, Dim);

                Image CTF64 = GetCTFsForOneParticle(options, ParticlePositions, CTFCoords64, null, true, false, false, null);
                Image CTFUnweighted = GetCTFsForOneParticle(options, ParticlePositions, CTFCoords64, null, false, false, false, null);

                CTF64.Multiply(CTFUnweighted);

                Image CTFComplex = new Image(CTF64.Dims, true, true);
                CTFComplex.Fill(new float2(1, 0));
                CTFComplex.Multiply(CTF64);
                CTF64.Dispose();

                CTFUnweighted.Abs();

                Projector Reconstructor = new Projector(new int3(Dim), 1);
                Projector Corrector = new Projector(new int3(Dim), 1);

                Reconstructor.BackProject(CTFComplex, CTFUnweighted, GetAngleInAllTilts(ParticlePositions), MagnificationCorrection);

                CTFUnweighted.Fill(1);
                Corrector.BackProject(CTFComplex, CTFUnweighted, GetAngleInAllTilts(ParticlePositions), MagnificationCorrection);

                Corrector.Weights.Min(1);
                Reconstructor.Data.Multiply(Corrector.Weights);
                Corrector.Dispose();

                CTFComplex.Dispose();
                CTFUnweighted.Dispose();

                Reconstructor.Weights.Max(0.02f);

                Image CTF3D = Reconstructor.Reconstruct(true, "C1", null, 0, 0, 0, 0);
                Reconstructor.Dispose();

                CTF3D.WriteMRC16b(IOPath.Combine(ReconstructionCTFDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
                CTF3D.Dispose();
                CTFCoords64.Dispose();
            }

            #region Teardown

            for (int i = 0; i < NThreads; i++)
            {
                GPU.DestroyFFTPlan(PlanForw[i]);
                GPU.DestroyFFTPlan(PlanBack[i]);
                GPU.DestroyFFTPlan(PlanForwCTF[i]);
                GPU.DestroyFFTPlan(PlanForwParticle[i]);
                Projectors[i].Dispose();
                Correctors[i].Dispose();
                Subtomo[i].Dispose();
                SubtomoCropped[i].Dispose();
                Images[i].Dispose();
                ImagesFT[i].Dispose();
                CTFs[i].Dispose();
                Samples[i].Dispose();
                if (options.PrepareDenoising)
                {
                    ImagesFTHalf[i].Dispose();
                    CTFsHalf[i].Dispose();
                    SamplesHalf[i].Dispose();
                }
            }

            CTFCoords.Dispose();
            foreach (var image in TiltData)
                image.FreeDevice();
            foreach (var tiltMask in TiltMasks)
                tiltMask?.FreeDevice();

            #endregion

            if (IsCanceled)
                return;

            if (options.DoDeconv)
            {
                IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Deconvolving...");

                {
                    Image FullRec = new Image(OutputRec, DimsVolumeCropped);

                    Image FullRecFT = FullRec.AsFFT_CPU(true);
                    FullRec.Dispose();

                    CTF SubtomoCTF = CTF.GetCopy();
                    SubtomoCTF.Defocus = (decimal)GetTiltDefocus(NTilts / 2);
                    SubtomoCTF.PixelSize = options.BinnedPixelSizeMean;

                    GPU.DeconvolveCTF(FullRecFT.GetDevice(Intent.Read),
                                        FullRecFT.GetDevice(Intent.Write),
                                        FullRecFT.Dims,
                                        SubtomoCTF.ToStruct(),
                                        (float)options.DeconvStrength,
                                        (float)options.DeconvFalloff,
                                        (float)(options.BinnedPixelSizeMean * 2 / options.DeconvHighpass));

                    Image FullRecDeconv = FullRecFT.AsIFFT_CPU(true);
                    FullRecFT.Dispose();

                    OutputRecDeconv = FullRecDeconv.GetHost(Intent.Read);
                    FullRecDeconv.Dispose();
                }

                if (options.PrepareDenoising)
                {
                    for (int ihalf = 0; ihalf < 2; ihalf++)
                    {
                        Image FullRec = new Image(OutputRecHalves[ihalf], DimsVolumeCropped);

                        Image FullRecFT = FullRec.AsFFT_CPU(true);
                        FullRec.Dispose();

                        CTF SubtomoCTF = CTF.GetCopy();
                        SubtomoCTF.Defocus = (decimal)GetTiltDefocus(NTilts / 2);
                        SubtomoCTF.PixelSize = options.BinnedPixelSizeMean;

                        GPU.DeconvolveCTF(FullRecFT.GetDevice(Intent.Read),
                                            FullRecFT.GetDevice(Intent.Write),
                                            FullRecFT.Dims,
                                            SubtomoCTF.ToStruct(),
                                            (float)options.DeconvStrength,
                                            (float)options.DeconvFalloff,
                                            (float)(options.BinnedPixelSizeMean * 2 / options.DeconvHighpass));

                        Image FullRecDeconv = FullRecFT.AsIFFT_CPU(true);
                        FullRecFT.Dispose();

                        OutputRecHalves[ihalf] = FullRecDeconv.GetHost(Intent.Read);
                        FullRecDeconv.Dispose();
                    }
                }
            }

            if (options.KeepOnlyFullVoxels)
            {
                IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Trimming...");

                float BinnedAngPix = (float)options.BinnedPixelSizeMean;

                Parallel.For(0, DimsVolumeCropped.Z, z =>
                {
                    float3[] VolumePositions = new float3[DimsVolumeCropped.ElementsSlice()];
                    for (int y = 0; y < DimsVolumeCropped.Y; y++)
                        for (int x = 0; x < DimsVolumeCropped.X; x++)
                            VolumePositions[y * DimsVolumeCropped.X + x] = new float3(x * BinnedAngPix, y * BinnedAngPix, z * BinnedAngPix);

                    float3[] ImagePositions = GetPositionInAllTiltsNoLocalWarp(VolumePositions);

                    for (int i = 0; i < ImagePositions.Length; i++)
                    {
                        int ii = i / NTilts;
                        int t = i % NTilts;

                        if (ImagePositions[i].X < 0 || ImagePositions[i].Y < 0 ||
                            ImagePositions[i].X > ImageDimensionsPhysical.X - BinnedAngPix ||
                            ImagePositions[i].Y > ImageDimensionsPhysical.Y - BinnedAngPix)
                        {
                            OutputRec[z][ii] = 0;
                            if (options.DoDeconv)
                                OutputRecDeconv[z][ii] = 0;
                            if (options.PrepareDenoising)
                            {
                                OutputRecHalves[0][z][ii] = 0;
                                OutputRecHalves[1][z][ii] = 0;
                            }
                        }
                    }
                });
            }

            #endregion

            IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Writing...");

            Image OutputRecImage = new Image(OutputRec, DimsVolumeCropped);
            {
                Image OutputFlat = OutputRecImage.AsSliceXY(OutputRecImage.Dims.Z / 2);
                float2 MeanStd;
                {
                    Image CentralQuarter = OutputFlat.AsPadded(new int2(OutputFlat.Dims) / 2);
                    MeanStd = MathHelper.MeanAndStd(CentralQuarter.GetHost(Intent.Read)[0]);
                    CentralQuarter.Dispose();
                }
                float FlatMin = MeanStd.X - MeanStd.Y * 3;
                float FlatMax = MeanStd.X + MeanStd.Y * 3;
                OutputFlat.TransformValues(v => (v - FlatMin) / (FlatMax - FlatMin) * 255);

                OutputFlat.WritePNG(IOPath.Combine(ReconstructionDir, NameWithRes + ".png"));
                OutputFlat.Dispose();
            }
            OutputRecImage.WriteMRC16b(IOPath.Combine(ReconstructionDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
            OutputRecImage.Dispose();

            if (options.DoDeconv)
            {
                Image OutputRecDeconvImage = new Image(OutputRecDeconv, DimsVolumeCropped);
                OutputRecDeconvImage.WriteMRC16b(IOPath.Combine(ReconstructionDeconvDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
                OutputRecDeconvImage.Dispose();
            }

            if (options.PrepareDenoising)
            {
                Image OutputRecOddImage = new Image(OutputRecHalves[0], DimsVolumeCropped);
                OutputRecOddImage.WriteMRC16b(IOPath.Combine(ReconstructionOddDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
                OutputRecOddImage.Dispose();

                Image OutputRecEvenImage = new Image(OutputRecHalves[1], DimsVolumeCropped);
                OutputRecEvenImage.WriteMRC16b(IOPath.Combine(ReconstructionEvenDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
                OutputRecEvenImage.Dispose();
            }

            IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Done.");
        }

        public void ReconstructSubtomos(ProcessingOptionsTomoSubReconstruction options, float3[] positions, float3[] angles)
        {
            int GPUID = GPU.GetDevice();

            bool IsCanceled = false;
            if (options.UseCPU)
                Console.WriteLine("Using CPU");

            if (!Directory.Exists(SubtomoDir))
                Directory.CreateDirectory(SubtomoDir);

            #region Dimensions

            VolumeDimensionsPhysical = options.DimensionsPhysical;

            CTF MaxDefocusCTF = GetTiltCTF(IndicesSortedDose[0]);
            MaxDefocusCTF.PixelSize = options.BinnedPixelSizeMean;
            int MinimumBoxSize = (int)Math.Round(MaxDefocusCTF.GetAliasingFreeSize((float)options.BinnedPixelSizeMean * 2, (float)(options.ParticleDiameter / options.BinnedPixelSizeMean)) / 2f) * 2;

            int SizeSub = options.BoxSize;
            int SizeSubSuper = Math.Min(1024, Math.Max(SizeSub * 2, MinimumBoxSize));

            #endregion

            #region Load and preprocess tilt series

            Movie[] TiltMovies;
            Image[] TiltData;
            Image[] TiltMasks;
            LoadMovieData(options, out TiltMovies, out TiltData, false, out _, out _);
            LoadMovieMasks(options, out TiltMasks);
            for (int z = 0; z < NTilts; z++)
            {
                EraseDirt(TiltData[z], TiltMasks[z]);
                TiltMasks[z]?.FreeDevice();

                if (options.NormalizeInput)
                {
                    TiltData[z].SubtractMeanGrid(new int2(1));
                    TiltData[z].Bandpass(1f / SizeSub, 1f, false, 0f);

                    GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                                  TiltData[z].GetDevice(Intent.Write),
                                  (uint)TiltData[z].ElementsReal,
                                  1);
                }
                else
                {
                    TiltData[z].Bandpass(1f / SizeSub, 1f, false, 0f);
                }

                if (options.Invert)
                    TiltData[z].Multiply(-1f);

                TiltData[z].FreeDevice();

                //TiltData[z].Multiply(TiltMasks[z]);
            }

            #endregion

            #region Memory and FFT plan allocation

            Image CTFCoords = CTF.GetCTFCoords(SizeSubSuper, SizeSubSuper);

            int NThreads = 1;

            int[] PlanForwRec = new int[NThreads], PlanBackRec = new int[NThreads];
            if (!options.UseCPU)
                for (int i = 0; i < NThreads; i++)
                {
                    //Projector.GetPlans(new int3(SizeSubSuper), 1, out PlanForwRec[i], out PlanBackRec[i], out PlanForwCTF[i]);
                    PlanForwRec[i] = GPU.CreateFFTPlan(new int3(SizeSubSuper), 1);
                    PlanBackRec[i] = GPU.CreateIFFTPlan(new int3(SizeSubSuper), 1);
                }
            int[] PlanForwRecCropped = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeSub), 1), NThreads);
            int[] PlanForwParticle = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeSubSuper, SizeSubSuper, 1), (uint)NTilts), NThreads);

            Projector[] Projectors = Helper.ArrayOfFunction(i => new Projector(new int3(SizeSubSuper), 1), NThreads);
            Projector[] ProjectorsMultiplicity = Helper.ArrayOfFunction(i => new Projector(new int3(SizeSubSuper), 1), NThreads);

            Image[] VolumeCropped = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub)), NThreads);
            Image[] VolumeCTFCropped = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub), true), NThreads);

            Image[] Subtomo = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper)), NThreads);
            float[][] CPUBuffer = null;
            if (options.UseCPU)
                CPUBuffer = Helper.ArrayOfFunction(i => new float[new int3(SizeSubSuper).Elements()], NThreads);
            //Image[] SubtomoCTF = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper), true), NThreads);
            //Image[] SubtomoCTFComplex = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper), true, true), NThreads);
            Image[] SubtomoSparsityMask = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub), true), NThreads);
            Image[] Images = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts)), NThreads);
            Image[] ImagesFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true, true), NThreads);
            Image[] CTFs = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true), NThreads);
            Image[] CTFsAbs = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true), NThreads);
            Image[] CTFsUnweighted = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true), NThreads);
            Image[] CTFsComplex = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true, true), NThreads);

            Image[] SumAllParticles = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub)), NThreads);

            GPU.CheckGPUExceptions();

            #endregion

            float[] TiltWeights = new float[NTilts];
            if (options.DoLimitDose)
                for (int t = 0; t < Math.Min(NTilts, options.NTilts); t++)
                    TiltWeights[IndicesSortedDose[t]] = 1 * (UseTilt[t] ? 1 : 0);
            else
                TiltWeights = UseTilt.Select(v => v ? 1f : 0f).ToArray();

            Helper.ForCPU(0, positions.Length / NTilts, NThreads,
                threadID => GPU.SetDevice(GPUID),
                (p, threadID) =>
                {

                    if (IsCanceled)
                        return;

                    float3[] ParticlePositions = positions.Skip(p * NTilts).Take(NTilts).ToArray();
                    float3[] ParticleAngles = options.PrerotateParticles ? angles.Skip(p * NTilts).Take(NTilts).ToArray() : null;

                    #region Multiplicity

                    ProjectorsMultiplicity[threadID].Data.Fill(0);
                    ProjectorsMultiplicity[threadID].Weights.Fill(0);
                    CTFsComplex[threadID].Fill(new float2(1, 0));
                    CTFs[threadID].Fill(1);

                    ProjectorsMultiplicity[threadID].BackProject(CTFsComplex[threadID], CTFs[threadID], !options.PrerotateParticles ? GetAngleInAllTilts(ParticlePositions) : GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles), MagnificationCorrection);
                    ProjectorsMultiplicity[threadID].Weights.Min(1);

                    #endregion

                    Timing.Start("ExtractImageData");
                    GetImagesForOneParticle(options, TiltData, SizeSubSuper, ParticlePositions, PlanForwParticle[threadID], -1, 8, true, Images[threadID], ImagesFT[threadID]);
                    Timing.Finish("ExtractImageData");

                    Timing.Start("CreateRawCTF");
                    GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, null, true, false, false, CTFs[threadID]);
                    GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, null, false, false, false, CTFsUnweighted[threadID]);
                    Timing.Finish("CreateRawCTF");

                    if (options.DoLimitDose)
                        CTFs[threadID].Multiply(TiltWeights);

                    // Subtomo is (Image * CTFweighted) / abs(CTFunweighted)
                    // 3D CTF is (CTFweighted * CTFweighted) / abs(CTFweighted)

                    ImagesFT[threadID].Multiply(CTFs[threadID]);
                    //GPU.Abs(CTFs[threadID].GetDevice(Intent.Read),
                    //        CTFsAbs[threadID].GetDevice(Intent.Write),
                    //        CTFs[threadID].ElementsReal);

                    CTFsComplex[threadID].Fill(new float2(1, 0));
                    CTFsComplex[threadID].Multiply(CTFsUnweighted[threadID]);   // What the raw image is like: unweighted, unflipped
                    CTFsComplex[threadID].Multiply(CTFs[threadID]);             // Weight by the same CTF as raw image: weighted, unflipped

                    CTFsUnweighted[threadID].Abs();

                    #region Sub-tomo

                    Projectors[threadID].Data.Fill(0);
                    Projectors[threadID].Weights.Fill(0);

                    Timing.Start("ProjectImageData");
                    Projectors[threadID].BackProject(ImagesFT[threadID], CTFsUnweighted[threadID], !options.PrerotateParticles ? GetAngleInAllTilts(ParticlePositions) : GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles), MagnificationCorrection);
                    Timing.Finish("ProjectImageData");

                    //Projectors[threadID].Weights.Fill(1);
                    Projectors[threadID].Data.Multiply(ProjectorsMultiplicity[threadID].Weights);

                    Timing.Start("ReconstructSubtomo");
                    if (options.UseCPU)
                        Projectors[threadID].ReconstructCPU(Subtomo[threadID], CPUBuffer[threadID], false, "C1");
                    else
                        Projectors[threadID].Reconstruct(Subtomo[threadID].GetDevice(Intent.Write), false, "C1", null, PlanForwRec[threadID], PlanBackRec[threadID], PlanForwRec[threadID], 0);
                    Timing.Finish("ReconstructSubtomo");

                    GPU.Pad(Subtomo[threadID].GetDevice(Intent.Read),
                            VolumeCropped[threadID].GetDevice(Intent.Write),
                            new int3(SizeSubSuper),
                            new int3(SizeSub),
                            1);

                    if (options.NormalizeOutput)
                        GPU.NormParticles(VolumeCropped[threadID].GetDevice(Intent.Read),
                                          VolumeCropped[threadID].GetDevice(Intent.Write),
                                          new int3(SizeSub),
                                          (uint)Math.Round(options.ParticleDiameter / options.BinnedPixelSizeMean / 2),
                                          false,
                                          1);

                    SumAllParticles[threadID].Add(VolumeCropped[threadID]);

                    VolumeCropped[threadID].WriteMRC16b(IOPath.Combine(SubtomoDir, $"{RootName}{options.Suffix}_{p:D7}_{options.BinnedPixelSizeMean:F2}A.mrc"), (float)options.BinnedPixelSizeMean, true);

                    #endregion

                    #region CTF

                    // Back-project and reconstruct
                    Projectors[threadID].Data.Fill(0);
                    Projectors[threadID].Weights.Fill(0);

                    Projectors[threadID].BackProject(CTFsComplex[threadID], CTFsUnweighted[threadID], !options.PrerotateParticles ? GetAngleInAllTilts(ParticlePositions) : GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles), MagnificationCorrection);

                    //Projectors[threadID].Weights.Fill(1);
                    Projectors[threadID].Data.Multiply(ProjectorsMultiplicity[threadID].Weights);

                    Timing.Start("ReconstructCTF");
                    if (options.UseCPU)
                        Projectors[threadID].ReconstructCPU(Subtomo[threadID], CPUBuffer[threadID], false, "C1");
                    else
                        Projectors[threadID].Reconstruct(Subtomo[threadID].GetDevice(Intent.Write), false, "C1", null, PlanForwRec[threadID], PlanBackRec[threadID], PlanForwRec[threadID], 0);
                    Timing.Finish("ReconstructCTF");

                    Timing.Start("3DCTFCrop");
                    //SubtomoCTFComplex[threadID].Fill(new float2(1, 0));
                    //SubtomoCTFComplex[threadID].Multiply(SubtomoCTF[threadID]);
                    //GPU.IFFT(SubtomoCTFComplex[threadID].GetDevice(Intent.Read),
                    //         Subtomo[threadID].GetDevice(Intent.Write),
                    //         new int3(SizeSubSuper),
                    //         1,
                    //         PlanBackRec[threadID],
                    //         false);

                    GPU.Pad(Subtomo[threadID].GetDevice(Intent.Read),
                            VolumeCropped[threadID].GetDevice(Intent.Write),
                            new int3(SizeSubSuper),
                            new int3(SizeSub),
                            1);

                    GPU.FFT(VolumeCropped[threadID].GetDevice(Intent.Read),
                            Subtomo[threadID].GetDevice(Intent.Write),
                            new int3(SizeSub),
                            1,
                            PlanForwRecCropped[threadID]);

                    GPU.ShiftStackFT(Subtomo[threadID].GetDevice(Intent.Read),
                                     Subtomo[threadID].GetDevice(Intent.Write),
                                     new int3(SizeSub),
                                     new[] { SizeSub / 2f, SizeSub / 2f, SizeSub / 2f },
                                     1);

                    GPU.Real(Subtomo[threadID].GetDevice(Intent.Read),
                             VolumeCTFCropped[threadID].GetDevice(Intent.Write),
                             VolumeCTFCropped[threadID].ElementsReal);

                    VolumeCTFCropped[threadID].Multiply(1f / (SizeSubSuper * SizeSubSuper));
                    Timing.Finish("3DCTFCrop");

                    if (options.MakeSparse)
                    {
                        GPU.Abs(VolumeCTFCropped[threadID].GetDevice(Intent.Read),
                                SubtomoSparsityMask[threadID].GetDevice(Intent.Write),
                                VolumeCTFCropped[threadID].ElementsReal);
                        SubtomoSparsityMask[threadID].Binarize(0.01f);

                        VolumeCTFCropped[threadID].Multiply(SubtomoSparsityMask[threadID]);
                    }

                    VolumeCTFCropped[threadID].WriteMRC16b(IOPath.Combine(SubtomoDir, $"{RootName}{options.Suffix}_{p:D7}_ctf_{options.BinnedPixelSizeMean:F2}A.mrc"), (float)options.BinnedPixelSizeMean, true);

                    #endregion

                    //Console.WriteLine(SizeSubSuper);
                    //Timing.PrintMeasurements();
                }, null);

            // Write the sum of all particles
            {
                for (int i = 1; i < NThreads; i++)
                    SumAllParticles[0].Add(SumAllParticles[i]);
                SumAllParticles[0].Multiply(1f / Math.Max(1, positions.Length / NTilts));

                SumAllParticles[0].WriteMRC16b(IOPath.Combine(SubtomoDir, $"{RootName}{options.Suffix}_{options.BinnedPixelSizeMean:F2}A_average.mrc"), (float)options.BinnedPixelSizeMean, true);
            }

            #region Teardown

            for (int i = 0; i < NThreads; i++)
            {
                if (!options.UseCPU)
                {
                    GPU.DestroyFFTPlan(PlanForwRec[i]);
                    GPU.DestroyFFTPlan(PlanBackRec[i]);
                }
                //GPU.DestroyFFTPlan(PlanForwCTF[i]);
                GPU.DestroyFFTPlan(PlanForwParticle[i]);
                Projectors[i].Dispose();
                ProjectorsMultiplicity[i].Dispose();
                Subtomo[i].Dispose();
                //SubtomoCTF[i].Dispose();
                SubtomoSparsityMask[i].Dispose();
                Images[i].Dispose();
                ImagesFT[i].Dispose();
                CTFs[i].Dispose();
                CTFsAbs[i].Dispose();
                CTFsUnweighted[i].Dispose();
                CTFsComplex[i].Dispose();

                SumAllParticles[i].Dispose();

                GPU.DestroyFFTPlan(PlanForwRecCropped[i]);
                VolumeCropped[i].Dispose();
                VolumeCTFCropped[i].Dispose();
                //SubtomoCTFComplex[i].Dispose();
            }

            CTFCoords.Dispose();
            //CTFCoordsPadded.Dispose();
            foreach (var image in TiltData)
                image.FreeDevice();
            foreach (var tiltMask in TiltMasks)
                tiltMask?.FreeDevice();

            #endregion
        }

        public void ReconstructParticleSeries(ProcessingOptionsTomoSubReconstruction options, float3[] positions, float3[] angles, string tablePath, out Star tableOut)
        {
            bool IsCanceled = false;

            if (!Directory.Exists(ParticleSeriesDir))
                Directory.CreateDirectory(ParticleSeriesDir);

            #region Dimensions

            VolumeDimensionsPhysical = options.DimensionsPhysical;

            int SizeSub = options.BoxSize;

            #endregion

            #region Load and preprocess tilt series

            Movie[] TiltMovies;
            Image[] TiltData;
            Image[] TiltMasks;
            LoadMovieData(options, out TiltMovies, out TiltData, false, out _, out _);
            LoadMovieMasks(options, out TiltMasks);
            for (int z = 0; z < NTilts; z++)
            {
                EraseDirt(TiltData[z], TiltMasks[z]);
                TiltMasks[z]?.FreeDevice();

                if (options.NormalizeInput)
                {
                    TiltData[z].SubtractMeanGrid(new int2(1));
                    TiltData[z].Bandpass(1f / SizeSub, 1f, false, 0f);

                    GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                                  TiltData[z].GetDevice(Intent.Write),
                                  (uint)TiltData[z].ElementsReal,
                                  1);
                }
                else
                {
                    TiltData[z].Bandpass(1f / SizeSub, 1f, false, 0f);
                }

                if (options.Invert)
                    TiltData[z].Multiply(-1f);

                TiltData[z].FreeDevice();

                //TiltData[z].Multiply(TiltMasks[z]);
            }

            #endregion

            #region Memory and FFT plan allocation

            int PlanForwParticle = GPU.CreateFFTPlan(new int3(SizeSub, SizeSub, 1), (uint)NTilts);
            int PlanBackParticle = GPU.CreateIFFTPlan(new int3(SizeSub, SizeSub, 1), (uint)NTilts);

            #endregion

            #region Create STAR table

            tableOut = new Star(new string[]
            {
                "rlnTomoName",
                "rlnTomoParticleId",
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnCoordinateZ",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnTomoParticleName",
                "rlnOpticsGroup",
                "rlnImageName",
                "rlnOriginXAngst",
                "rlnOriginYAngst",
                "rlnOriginZAngst",
                "rlnTomoVisibleFrames"
            });

            #endregion

            Random Rand = new Random(Name.GetHashCode());

            CTF[] TiltCTFs = Helper.ArrayOfFunction(t => GetTiltCTF(t), NTilts);
            float PixelSize = (float)options.BinnedPixelSizeMean;

            int NParticles = positions.Length / NTilts;
            List<int> UsedTilts = options.DoLimitDose ? IndicesSortedDose.Take(options.NTilts).ToList() : IndicesSortedDose.ToList();
            UsedTilts.Sort();

            Image Images = new Image(new int3(SizeSub, SizeSub, NTilts));
            Image ImagesFT = new Image(new int3(SizeSub, SizeSub, NTilts), true, true);
            Image CTFs = new Image(new int3(SizeSub, SizeSub, NTilts), true, false);
            Image CTFCoords = CTF.GetCTFCoords(SizeSub, SizeSub);

            Image UsedParticles = new Image(new int3(SizeSub, SizeSub, UsedTilts.Count));
            float[][] UsedParticlesData = UsedParticles.GetHost(Intent.Write);

            Image SumAllParticles = new Image(new int3(SizeSub, SizeSub, NTilts));

            // Needed as long as we can't specify per-tilt B-factors in tomograms.star
            // B-factors are strictly tied to exposure, and aren't B-factors but Niko's formula
            Image RelionWeights = new Image(new int3(SizeSub, SizeSub, NTilts), true, false);
            {
                float[][] WeightsData = RelionWeights.GetHost(Intent.Write);
                float a = 0.245f;
                float b = -1.665f;
                float c = 2.81f;

                for (int t = 0; t < NTilts; t++)
                {
                    CTF TiltCTF = GetCTFParamsForOneTilt((float)options.BinnedPixelSizeMean,
                                                         new[] { 0f },
                                                         new[] { VolumeDimensionsPhysical * 0.5f },
                                                         t,
                                                         true)[0];

                    Helper.ForEachElementFT(new int2(SizeSub), (x, y, xx, yy) =>
                    {
                        float r = MathF.Sqrt(xx * xx + yy * yy);
                        float k = r / (SizeSub * (float)options.BinnedPixelSizeMean);
                        float d0 = a * MathF.Pow(k, b) + c;

                        WeightsData[t][y * (SizeSub / 2 + 1) + x] = MathF.Exp(-0.5f * Dose[t] / d0) * (float)TiltCTF.Scale;
                    });
                }
            }

            for (int p = 0; p < NParticles; p++)
            {
                float3[] ParticlePositions = positions.Skip(p * NTilts).Take(NTilts).ToArray();

                ImagesFT = GetImagesForOneParticle(options, TiltData, SizeSub, ParticlePositions, PlanForwParticle, -1, 0, false, Images, ImagesFT);
                GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, null, false, false, false, CTFs);

                ImagesFT.Multiply(CTFs);
                ImagesFT.Multiply(RelionWeights);

                GPU.IFFT(ImagesFT.GetDevice(Intent.Read),
                         Images.GetDevice(Intent.Write),
                         ImagesFT.Dims.Slice(),
                         (uint)ImagesFT.Dims.Z,
                         PlanBackParticle,
                         false);

                SumAllParticles.Add(Images);

                float[][] ImagesData = Images.GetHost(Intent.Read);
                for (int i = 0; i < UsedTilts.Count; i++)
                    Array.Copy(ImagesData[UsedTilts[i]], 0, UsedParticlesData[i], 0, ImagesData[0].Length);

                bool[] Visibility = GetPositionInAllTilts(positions.Skip(p * NTilts).Take(NTilts).ToArray()).Select(p =>
                {
                    return p.X > options.ParticleDiameter / 2 && p.X < ImageDimensionsPhysical.X - options.ParticleDiameter / 2 &&
                           p.Y > options.ParticleDiameter / 2 && p.Y < ImageDimensionsPhysical.Y - options.ParticleDiameter / 2;
                }).ToArray();
                for (int t = 0; t < NTilts; t++)
                    Visibility[t] = Visibility[t] && UseTilt[t];
                Visibility = Helper.IndexedSubset(Visibility, UsedTilts.ToArray());


                float3 Position0 = positions[p * NTilts + NTilts / 2] / (float)options.BinnedPixelSizeMean;
                float3 Angle0 = angles[p * NTilts + NTilts / 2];

                string SeriesPath = IOPath.Combine(ParticleSeriesDir, $"{RootName}{options.Suffix}_{options.BinnedPixelSizeMean:F2}A_{(p + 1):D6}.mrcs");
                string SeriesPathRelative = Helper.MakePathRelativeTo(SeriesPath, tablePath);

                tableOut.AddRow(new string[]
                {
                    RootName + ".tomostar",
                    (p + 1).ToString(),
                    Position0.X.ToString("F3", CultureInfo.InvariantCulture),
                    Position0.Y.ToString("F3", CultureInfo.InvariantCulture),
                    Position0.Z.ToString("F3", CultureInfo.InvariantCulture),
                    Angle0.X.ToString("F3", CultureInfo.InvariantCulture),
                    Angle0.Y.ToString("F3", CultureInfo.InvariantCulture),
                    Angle0.Z.ToString("F3", CultureInfo.InvariantCulture),
                    $"{RootName}/{p + 1}",
                    $"{RootName}",
                    SeriesPathRelative,
                    "0.0",
                    "0.0",
                    "0.0",
                    $"[{string.Join(',', Visibility.Select(v => v ? "1" : "0"))}]"
                });

                UsedParticles.WriteMRC16b(SeriesPath, (float)options.BinnedPixelSizeMean, true);

                if (IsCanceled)
                    break;
            }

            // Save the average of all particle stacks
            {
                SumAllParticles.Multiply(1f / Math.Max(1, NParticles));

                float[][] SumAllParticlesData = SumAllParticles.GetHost(Intent.Read);
                for (int i = 0; i < UsedTilts.Count; i++)
                    Array.Copy(SumAllParticlesData[UsedTilts[i]], 0, UsedParticlesData[i], 0, SumAllParticlesData[0].Length);

                string SumPath = IOPath.Combine(ParticleSeriesDir, $"{RootName}{options.Suffix}_{options.BinnedPixelSizeMean:F2}A_average.mrcs");
                UsedParticles.WriteMRC16b(SumPath, (float)options.BinnedPixelSizeMean, true);
            }

            #region Teardown

            RelionWeights.Dispose();
            SumAllParticles.Dispose();
            UsedParticles.Dispose();
            Images.Dispose();
            ImagesFT.Dispose();
            CTFs.Dispose();
            CTFCoords.Dispose();

            GPU.DestroyFFTPlan(PlanForwParticle);
            GPU.DestroyFFTPlan(PlanBackParticle);

            foreach (var image in TiltData)
                image.FreeDevice();
            foreach (var tiltMask in TiltMasks)
                tiltMask?.FreeDevice();

            #endregion
        }

        public void FigureOutNormalization(ProcessingOptionsTomoSubReconstruction options)
        {
            options.BinTimes = (decimal)Math.Log(30 / (double)options.PixelSizeMean, 2.0);
            
            VolumeDimensionsPhysical = options.DimensionsPhysical;

            //Movie[] TiltMovies;
            Image[] TiltData;
            Image[] TiltMasks;
            LoadMovieData(options, out _, out TiltData, false, out _, out _);
            LoadMovieMasks(options, out TiltMasks);
            for (int z = 0; z < NTilts; z++)
            {
                EraseDirt(TiltData[z], TiltMasks[z]);
                TiltMasks[z]?.FreeDevice();

                if (options.NormalizeInput)
                {
                    TiltData[z].SubtractMeanGrid(new int2(1));
                    TiltData[z].Bandpass(1f / 8f, 1f, false, 1f / 8f);

                    Image Cropped = TiltData[z].AsPadded(new int2(TiltData[z].Dims) / 2);
                    float2 MeanStd = MathHelper.MeanAndStd(Cropped.GetHost(Intent.Read)[0]);
                    Cropped.Dispose();

                    TiltData[z].Add(-MeanStd.X);
                    //if (MeanStd.Y > 0)
                    //    TiltData[z].Multiply(1 / MeanStd.Y);

                    Console.WriteLine(MeanStd.Y);

                    //GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                    //              TiltData[z].GetDevice(Intent.Write),
                    //              (uint)TiltData[z].ElementsReal,
                    //              1);
                }

                if (options.Invert)
                    TiltData[z].Multiply(-1f);

                TiltData[z].FreeDevice();

                //TiltData[z].Multiply(TiltMasks[z]);
            }

            Console.WriteLine("\n\n\n");

            int DimFull = Math.Min(TiltData[0].Dims.X, TiltData[0].Dims.Y);
            int DimCompare = DimFull / 4 * 2;

            Image CTFCoords = CTF.GetCTFCoords(DimFull, DimFull);
            Image CTFOri = new Image(new int3(DimFull, DimFull, NTilts), true, false);

            float3[] ParticlePositions = Helper.ArrayOfConstant(new float3(0.5f) * VolumeDimensionsPhysical, NTilts);
            float3[] ParticleAngles = GetAngleInAllTilts(ParticlePositions);

            Image ImagesOriFT = GetImagesForOneParticle(options, TiltData, DimFull, ParticlePositions, -1, -1, 8, true);
            GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, null, false, false, false, CTFOri);

            Image ImagesFlippedFT = ImagesOriFT.GetCopyGPU();
            Image ImagesWeightedFT = ImagesOriFT.GetCopyGPU();
            Image CTFWeighted = CTFOri.GetCopyGPU();
            Image CTFSign = CTFOri.GetCopyGPU();

            CTFSign.Sign();
            ImagesFlippedFT.Multiply(CTFSign);

            ImagesWeightedFT.Multiply(CTFSign);
            //CTFWeighted.Multiply(CTFOri);
            CTFWeighted.Abs();

            CTFSign.Dispose();

            int[] TiltIndicesSorted = IndicesSortedAbsoluteAngle;
            float[] Scales = new float[NTilts];

            for (int seriesLength = 0; seriesLength < NTilts; seriesLength++)
            {
                int[] CurrentIndices = TiltIndicesSorted.Take(Math.Max(1, seriesLength)).ToArray();
                float3[] CurrentAngles = Helper.IndexedSubset(ParticleAngles, CurrentIndices);

                Image StackWeightedFT = new Image(Helper.IndexedSubset(ImagesWeightedFT.GetHost(Intent.Read), CurrentIndices), new int3(DimFull, DimFull, CurrentIndices.Length), true, true);
                Image StackCTFWeighted = new Image(Helper.IndexedSubset(CTFWeighted.GetHost(Intent.Read), CurrentIndices), new int3(DimFull, DimFull, CurrentIndices.Length), true, false);

                Projector Reconstructor = new Projector(new int3(DimFull), 2);
                Reconstructor.BackProject(StackWeightedFT, StackCTFWeighted, CurrentAngles, Matrix2.Identity());
                Reconstructor.Weights.Max(1);
                Image VolFull = Reconstructor.Reconstruct(false, "C1", null, -1, -1, -1, 0);
                //VolFull.MaskSpherically(VolFull.Dims.X - 32, 16, true);
                VolFull.WriteMRC("d_volfull.mrc", true);

                StackWeightedFT.Fill(new float2(1, 0));
                StackCTFWeighted.Fill(1);
                Reconstructor.Data.Fill(0);
                Reconstructor.Weights.Fill(0);

                Reconstructor.BackProject(StackWeightedFT, StackCTFWeighted, CurrentAngles, Matrix2.Identity());
                Reconstructor.Weights.Max(1);
                Image PSF = Reconstructor.Reconstruct(false, "C1", null, -1, -1, -1, 0);
                PSF.WriteMRC("d_psf.mrc", true);

                Reconstructor.Dispose();

                Projector ProjectorCTF = new Projector(PSF, 2);
                Projector ProjectorVol = new Projector(VolFull, 2);

                Image NextFlippedFT = new Image(ImagesFlippedFT.GetHost(Intent.Read)[TiltIndicesSorted[seriesLength]], new int3(DimFull, DimFull, 1), true, true);
                Image WeightComplex = ProjectorCTF.Project(new int2(DimFull), new float3[] { ParticleAngles[TiltIndicesSorted[seriesLength]] });
                Image Weight = WeightComplex.AsAmplitudes().AndDisposeParent();
                Weight.WriteMRC("d_weight.mrc", true);

                NextFlippedFT.Multiply(Weight);
                Image NextFlipped = NextFlippedFT.AsIFFT(false, 0, true).AndDisposeParent();
                NextFlipped.MaskSpherically(NextFlipped.Dims.X / 2, 16, false, true);
                NextFlipped.WriteMRC("d_nextflipped.mrc", true);
                NextFlippedFT = NextFlipped.AsFFT().AndDisposeParent();

                Image RefFT = ProjectorVol.Project(new int2(DimFull), new float3[] { ParticleAngles[TiltIndicesSorted[seriesLength]] });
                Image Ref = RefFT.AsIFFT(false, 0, true).AndDisposeParent();
                Ref.MaskSpherically(Ref.Dims.X / 2, 16, false, true);
                Ref.WriteMRC("d_ref.mrc", true);
                RefFT = Ref.AsFFT().AndDisposeParent();

                ProjectorCTF.Dispose();
                ProjectorVol.Dispose();

                Func<double[], double> Eval = (input) =>
                {
                    Image NextCopy = NextFlippedFT.GetCopyGPU();
                    NextCopy.Multiply((float)input[0]);
                    NextCopy.Subtract(RefFT);
                    float Diff = NextCopy.GetHost(Intent.Read)[0].Skip(1).Select(v => v * v).Sum();
                    NextCopy.Dispose();

                    return Diff * 1000;
                };

                Func<double[], double[]> Grad = (input) =>
                {
                    double Score0 = Eval(input);
                    double Score1 = Eval(new double[] { input[0] + 1e-4 });
                    //Console.WriteLine(Score0);

                    return new double[] { (Score1 - Score0) / 1e-4 };
                };

                double[] StartParams = { 1.0 };

                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(1, Eval, Grad);
                Optimizer.Minimize(StartParams);

                Scales[TiltIndicesSorted[seriesLength]] = (float)StartParams[0];
                Console.WriteLine(StartParams[0]);

                RefFT.Dispose();
                WeightComplex.Dispose();
                NextFlippedFT.Dispose();
                PSF.Dispose();
                VolFull.Dispose();
                StackCTFWeighted.Dispose();
                StackWeightedFT.Dispose();
            }
        }

        public void AlignLocallyWithoutReferences(ProcessingOptionsTomoFullReconstruction options)
        {
            VolumeDimensionsPhysical = options.DimensionsPhysical;
            int SizeRegion = options.SubVolumeSize;

            #region Load and preprocess data

            Image[] TiltData;
            Image[] TiltMasks;
            LoadMovieData(options, out _, out TiltData, false, out _, out _);
            LoadMovieMasks(options, out TiltMasks);
            for (int z = 0; z < NTilts; z++)
            {
                EraseDirt(TiltData[z], TiltMasks[z]);
                TiltMasks[z]?.FreeDevice();

                TiltData[z].SubtractMeanGrid(new int2(1));
                //TiltData[z] = TiltData[z].AsPaddedClamped(new int2(TiltData[z].Dims) * 2).AndDisposeParent();
                //TiltData[z].MaskRectangularly((TiltData[z].Dims / 2).Slice(), MathF.Min(TiltData[z].Dims.X / 4, TiltData[z].Dims.Y / 4), false);
                //TiltData[z].WriteMRC("d_tiltdata.mrc", true);
                TiltData[z].Bandpass(1f / (SizeRegion / 2), 1f, false, 1f / (SizeRegion / 2));
                //TiltData[z] = TiltData[z].AsPadded(new int2(TiltData[z].Dims) / 2).AndDisposeParent();
                //TiltData[z].WriteMRC("d_tiltdatabp.mrc", true);

                GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                                TiltData[z].GetDevice(Intent.Write),
                                (uint)TiltData[z].ElementsReal,
                                1);

                if (options.Invert)
                    TiltData[z].Multiply(-1f);

                TiltData[z].FreeDevice();
            }

            int2 DimsImage = new int2(TiltData[0].Dims);
            int SizeReconstruction = Math.Max(DimsImage.X, DimsImage.Y);
            int SizeReconstructionPadded = SizeReconstruction * 2;

            #endregion

            int2 DimsPositionGrid;
            int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage - 64,
                                                               new int2(SizeRegion),
                                                               0.5f,
                                                               out DimsPositionGrid).Select(v => new int3(v.X + 32 + SizeRegion / 2,
                                                                                                          v.Y + 32 + SizeRegion / 2,
                                                                                                          0)).ToArray();
            float3[] PositionGridPhysical = PositionGrid.Select(v => new float3(v.X * (float)options.BinnedPixelSizeMean,
                                                                                v.Y * (float)options.BinnedPixelSizeMean,
                                                                                VolumeDimensionsPhysical.Z / 2)).ToArray();

            Image RegionMask = new Image(new int3(SizeRegion, SizeRegion, 1));
            RegionMask.Fill(1);
            RegionMask.MaskRectangularly(new int3(SizeRegion / 2, SizeRegion / 2, 1), SizeRegion / 4, false);
            RegionMask.WriteMRC("d_mask.mrc", true);

            GridMovementX = new CubicGrid(new int3(1, 1, NTilts));
            GridMovementY = new CubicGrid(new int3(1, 1, NTilts));

            GridVolumeWarpX = new LinearGrid4D(new int4(1));
            GridVolumeWarpY = new LinearGrid4D(new int4(1));
            GridVolumeWarpZ = new LinearGrid4D(new int4(1));

            Image Extracted1 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length)), 
                    Extracted2 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length)), 
                    Extracted3 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length));
            Image ExtractedFT1 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length), true, true), 
                    ExtractedFT2 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length), true, true), 
                    ExtractedFT3 = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length), true, true);
            int PlanForw = GPU.CreateFFTPlan(new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length);
            int PlanBack = GPU.CreateIFFTPlan(new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length);

            Image TiltDataFT = new Image(Helper.Combine(TiltData.Select(v => v.GetHost(Intent.Read)[0]).ToArray()), new int3(DimsImage.X, DimsImage.Y, NTilts)).AsFFT().AndDisposeParent();
            TiltDataFT.Multiply(1f / (DimsImage.Elements()));
            Image TiltDataFTFiltered = TiltDataFT.GetCopyGPU();
            Image TiltDataFiltered = new Image(new int3(DimsImage.X, DimsImage.Y, NTilts));
            int PlanBackTiltData = GPU.CreateIFFTPlan(TiltDataFiltered.Dims.Slice(), (uint)NTilts);

            Projector ProjectorCommonLine;
            {
                Image ZeroTilt = new Image(new int3(DimsImage.X));
                ZeroTilt.Fill(1); 
                ZeroTilt.MaskRectangularly(new int3(1, 1, SizeReconstruction), 0, true);
                ZeroTilt.MaskRectangularly(new int3(SizeReconstruction - 32, SizeReconstruction - 32, SizeReconstruction / 4), 16, true);

                ProjectorCommonLine = new Projector(ZeroTilt, 1);
                //Image Slice = ProjectorCommonLine.Project(new int2(SizeReconstruction), new[] { new float3(0, 3, -TiltAxisAngles[0]) * Helper.ToRad });
                //Slice.AsReal().WriteMRC("d_slicetest.mrc");
            }

            // Figure out global tilt angle offset
            if (true)
            {
                float[] OriAngles = Angles.ToArray();
                float[] OriAxisAngles = TiltAxisAngles.ToArray();

                Action<double[]> SetAngles = (input) =>
                {
                    for (int t = 0; t < NTilts; t++)
                        Angles[t] = OriAngles[t] + (float)input[0];

                    for (int t = 0; t < NTilts; t++)
                        TiltAxisAngles[t] = 84.05f;// OriAxisAngles[t] + (float)input[1];
                };

                Func<double[], double> Eval = (input) =>
                {
                    SetAngles(input);

                    double Result = 0;
                    bool FromScratch = true;

                    Image CommonLines = ProjectorCommonLine.Project(new int2(SizeReconstruction), TiltAxisAngles.Select(a => new float3(0, 3, -a) * Helper.ToRad).ToArray());
                    Image CommonLinesReal = CommonLines.AsReal().AndDisposeParent();
                    GPU.MultiplyComplexSlicesByScalar(TiltDataFT.GetDevice(Intent.Read),
                                                      CommonLinesReal.GetDevice(Intent.Read),
                                                      TiltDataFTFiltered.GetDevice(Intent.Write),
                                                      TiltDataFT.ElementsSliceComplex,
                                                      (uint)TiltDataFT.Dims.Z);
                    GPU.IFFT(TiltDataFTFiltered.GetDevice(Intent.Read),
                             TiltDataFiltered.GetDevice(Intent.Write),
                             TiltDataFiltered.Dims.Slice(),
                             (uint)TiltDataFiltered.Dims.Z,
                             PlanBackTiltData,
                             false);
                    TiltDataFiltered.Normalize();
                    CommonLinesReal.Dispose();
                    //TiltDataFiltered.WriteMRC("d_tiltdatafiltered.mrc", true);

                    for (int t = NTilts / 2 - 6; t <= NTilts / 2 + 6; t++)
                    {
                        if (FromScratch)
                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t - 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            GPU.Extract(TiltDataFiltered.GetDeviceSlice(t - 1, Intent.Read),
                                        Extracted1.GetDevice(Intent.Write),
                                        TiltData[t - 1].Dims,
                                        Extracted1.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Extracted1.GetDevice(Intent.Read), ExtractedFT1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT1.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT1.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        }
                        else
                        {
                            GPU.CopyDeviceToDevice(Extracted2.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), Extracted1.ElementsReal);
                        }

                        if (FromScratch)
                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            GPU.Extract(TiltDataFiltered.GetDeviceSlice(t, Intent.Read),
                                        Extracted2.GetDevice(Intent.Write),
                                        TiltData[t].Dims,
                                        Extracted2.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Extracted2.GetDevice(Intent.Read), ExtractedFT2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT2.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT2.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        }
                        else
                        {
                            GPU.CopyDeviceToDevice(Extracted3.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), Extracted2.ElementsReal);
                        }

                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t + 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            GPU.Extract(TiltDataFiltered.GetDeviceSlice(t + 1, Intent.Read),
                                        Extracted3.GetDevice(Intent.Write),
                                        TiltData[t + 1].Dims,
                                        Extracted1.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Extracted3.GetDevice(Intent.Read), ExtractedFT3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT3.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT3.GetDevice(Intent.Read), Extracted3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        }

                        Extracted1.Add(Extracted3);
                        Extracted1.Multiply(0.5f);

                        Extracted1.Multiply(Extracted2);
                        Extracted1.MultiplySlices(RegionMask);

                        Image Diff = Extracted1.AsSum3D();
                        Result += Diff.GetHost(Intent.Read)[0][0] * MathF.Pow(MathF.Cos(Angles[t] * Helper.ToRad), 1);
                        Diff.Dispose();

                        FromScratch = false;
                    }

                    return Result;
                };

                int OptIterations = 0;
                Func<double[], double[]> Grad = (input) =>
                {
                    double Delta = 0.1;
                    double[] Result = new double[input.Length];

                    if (OptIterations++ > 12)
                        return Result;

                    for (int i = 0; i < input.Length - 1; i++)
                    {
                        double[] InputPlus = input.ToArray();
                        InputPlus[i] += Delta;
                        double ScorePlus = Eval(InputPlus);

                        double[] InputMinus = input.ToArray();
                        InputMinus[i] -= Delta;
                        double ScoreMinus = Eval(InputMinus);

                        Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                    }

                    Console.WriteLine(Eval(input));

                    return Result;
                };

                double[] StartParams = new double[2];
                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                Optimizer.Maximize(StartParams);
            }

            // Patch Z position against raw patch CC
            if (false)
            {
                GridVolumeWarpX = new LinearGrid4D(new int4(1));
                GridVolumeWarpY = new LinearGrid4D(new int4(1));
                GridVolumeWarpZ = new LinearGrid4D(new int4(1));

                CubicGrid GridPatchZ = new CubicGrid(new int3(5, 5, 1));

                float[] OriWarping = Helper.ArrayOfConstant(VolumeDimensionsPhysical.Z / 2, GridPatchZ.Values.Length);

                Action<double[]> SetWarping = (input) =>
                {
                    float Mean = MathHelper.Mean(input.Select(v => (float)v));
                    GridPatchZ = new CubicGrid(GridPatchZ.Dimensions, input.Select((v, i) => OriWarping[i] + (float)v - Mean).ToArray());

                    float3[] InterpCoords = PositionGridPhysical.Select(v => new float3(v.X / VolumeDimensionsPhysical.X, v.Y / VolumeDimensionsPhysical.Y, 0.5f)).ToArray();
                    float[] InterpVals = GridPatchZ.GetInterpolated(InterpCoords);

                    for (int i = 0; i < PositionGridPhysical.Length; i++)
                        PositionGridPhysical[i].Z = InterpVals[i];
                };

                Func<double[], double> Eval = (input) =>
                {
                    SetWarping(input);

                    double Result = 0;
                    bool FromScratch = true;

                    for (int t = 1; t < NTilts - 1; t++)
                    {
                        if (FromScratch)
                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t - 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            GPU.Extract(TiltDataFiltered.GetDeviceSlice(t - 1, Intent.Read),
                                        Extracted1.GetDevice(Intent.Write),
                                        TiltData[t - 1].Dims,
                                        Extracted1.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Extracted1.GetDevice(Intent.Read), ExtractedFT1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT1.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT1.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        }
                        else
                        {
                            GPU.CopyDeviceToDevice(Extracted2.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), Extracted1.ElementsReal);
                        }

                        if (FromScratch)
                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            GPU.Extract(TiltDataFiltered.GetDeviceSlice(t, Intent.Read),
                                        Extracted2.GetDevice(Intent.Write),
                                        TiltData[t].Dims,
                                        Extracted2.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Extracted2.GetDevice(Intent.Read), ExtractedFT2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT2.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT2.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        }
                        else
                        {
                            GPU.CopyDeviceToDevice(Extracted3.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), Extracted2.ElementsReal);
                        }

                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t + 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            GPU.Extract(TiltDataFiltered.GetDeviceSlice(t + 1, Intent.Read),
                                        Extracted3.GetDevice(Intent.Write),
                                        TiltData[t + 1].Dims,
                                        Extracted1.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Extracted3.GetDevice(Intent.Read), ExtractedFT3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT3.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT3.GetDevice(Intent.Read), Extracted3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        }

                        Extracted1.Add(Extracted3);
                        Extracted1.Multiply(0.5f);

                        Extracted1.Multiply(Extracted2);
                        Extracted1.MultiplySlices(RegionMask);

                        Image Diff = Extracted1.AsSum3D();
                        Result += Diff.GetHost(Intent.Read)[0][0] * MathF.Pow(MathF.Cos(Angles[t] * Helper.ToRad), 1);
                        Diff.Dispose();

                        FromScratch = false;
                    }

                    return Result;
                };

                int OptIterations = 0;
                Func<double[], double[]> Grad = (input) =>
                {
                    double Delta = 0.1;
                    double[] Result = new double[input.Length];

                    if (OptIterations++ > 12)
                        return Result;

                    for (int i = 0; i < input.Length; i++)
                    {
                        double[] InputPlus = input.ToArray();
                        InputPlus[i] += Delta;
                        double ScorePlus = Eval(InputPlus);

                        double[] InputMinus = input.ToArray();
                        InputMinus[i] -= Delta;
                        double ScoreMinus = Eval(InputMinus);

                        Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                    }

                    Console.WriteLine(Eval(input));

                    return Result;
                };

                double[] StartParams = new double[OriWarping.Length];
                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                Optimizer.Maximize(StartParams);

                SetWarping(StartParams);
                Console.WriteLine(Eval(StartParams));

                new Image(PositionGridPhysical.Select(v => v.Z).ToArray(), new int3((int)Math.Sqrt(PositionGridPhysical.Length)).Slice()).WriteMRC("d_heightfield.mrc", true);
            }

            // Volume warp grid against raw patch CC
            if (false)
            {
                GridVolumeWarpX = new LinearGrid4D(new int4(3, 3, 1, 2));
                GridVolumeWarpY = new LinearGrid4D(GridVolumeWarpX.Dimensions);
                GridVolumeWarpZ = new LinearGrid4D(GridVolumeWarpX.Dimensions);

                float[] OriWarping = GridVolumeWarpZ.Values.Skip((int)GridVolumeWarpZ.Dimensions.ElementsSlice()).ToArray(); 

                Action<double[]> SetWarping = (input) =>
                {
                    float[] NewValues = new float[GridVolumeWarpZ.Values.Length];
                    float Mean = MathHelper.Mean(input.Select(v => (float)v));
                    for (int i = 0; i < GridVolumeWarpZ.Dimensions.Elements() - GridVolumeWarpZ.Dimensions.ElementsSlice(); i++)
                        NewValues[GridVolumeWarpZ.Dimensions.ElementsSlice() + i] = (float)input[i] - Mean;

                    GridVolumeWarpZ = new LinearGrid4D(GridVolumeWarpZ.Dimensions, NewValues);
                };

                Func<double[], double> Eval = (input) =>
                {
                    SetWarping(input);

                    double Result = 0;
                    bool FromScratch = true;

                    for (int t = 1; t < NTilts - 1; t++)
                    {
                        if (FromScratch)
                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t - 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            GPU.Extract(TiltData[t - 1].GetDevice(Intent.Read),
                                        Extracted1.GetDevice(Intent.Write),
                                        TiltData[t - 1].Dims,
                                        Extracted1.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Extracted1.GetDevice(Intent.Read), ExtractedFT1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT1.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT1.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        }
                        else
                        {
                            GPU.CopyDeviceToDevice(Extracted2.GetDevice(Intent.Read), Extracted1.GetDevice(Intent.Write), Extracted1.ElementsReal);
                        }

                        if (FromScratch)
                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                        Extracted2.GetDevice(Intent.Write),
                                        TiltData[t].Dims,
                                        Extracted2.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Extracted2.GetDevice(Intent.Read), ExtractedFT2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT2.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT2.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        }
                        else
                        {
                            GPU.CopyDeviceToDevice(Extracted3.GetDevice(Intent.Read), Extracted2.GetDevice(Intent.Write), Extracted2.ElementsReal);
                        }

                        {
                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t + 1).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            GPU.Extract(TiltData[t + 1].GetDevice(Intent.Read),
                                        Extracted3.GetDevice(Intent.Write),
                                        TiltData[t + 1].Dims,
                                        Extracted1.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Extracted3.GetDevice(Intent.Read), ExtractedFT3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT3.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT3.GetDevice(Intent.Read), Extracted3.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                        }

                        Extracted1.Add(Extracted3);
                        Extracted1.Multiply(0.5f);

                        Extracted1.Multiply(Extracted2);
                        Extracted1.MultiplySlices(RegionMask);

                        Image Diff = Extracted1.AsSum3D();
                        Result += Diff.GetHost(Intent.Read)[0][0] * MathF.Pow(MathF.Cos(Angles[t] * Helper.ToRad), 1);
                        Diff.Dispose();

                        FromScratch = false;
                    }

                    return Result;
                };

                int OptIterations = 0;
                Func<double[], double[]> Grad = (input) =>
                {
                    double Delta = 0.1;
                    double[] Result = new double[input.Length];

                    if (OptIterations++ > 12)
                        return Result;

                    for (int i = 0; i < input.Length; i++)
                    {
                        double[] InputPlus = input.ToArray();
                        InputPlus[i] += Delta;
                        double ScorePlus = Eval(InputPlus);

                        double[] InputMinus = input.ToArray();
                        InputMinus[i] -= Delta;
                        double ScoreMinus = Eval(InputMinus);

                        Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                    }

                    Console.WriteLine(Eval(input));

                    return Result;
                };

                double[] StartParams = new double[OriWarping.Length];
                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                Optimizer.Maximize(StartParams);

                SetWarping(StartParams);
            }

            // In-plane shift alignment
            if (true)
            {
                //CubicGrid GridWarp = new CubicGrid(new int3(3, 3, 1), new float[] { 0, 0, 0, 0, 10, 0, 0, 0, 0 });
                //{
                //    float3[] Coords = new float3[49];
                //    for (int y = 0; y < 7; y++)
                //        for (int x = 0; x < 7; x++)
                //            Coords[y * 7 + x] = new float3(x / 6f, y / 6f, 0);
                //    float[] Interpolated = GridWarp.GetInterpolated(Coords);

                //    //GridVolumeWarpZ = new LinearGrid4D(new int4(7, 7, 1, 1), Interpolated);
                //    GridVolumeWarpZ = new LinearGrid4D(new int4(1, 1, 1, 1), new[] { 10f });

                //    List<float2[]> TargetWarp = new List<float2[]>();
                //    for (int t = 0; t < NTilts; t++)
                //        TargetWarp.Add(GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => new float2(v.X, v.Y)).ToArray());

                //    float2 OffsetWarped = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, 0).Select(v => new float2(v.X, v.Y)).First();

                //    GridVolumeWarpZ = new LinearGrid4D(GridVolumeWarpX.Dimensions);

                //    float2 OffsetDefault = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, 0).Select(v => new float2(v.X, v.Y)).First();
                //    float2 Relative = OffsetWarped - OffsetDefault;

                //    Matrix3 TiltMatrix = Matrix3.Euler(0, Angles[0] * Helper.ToRad, -TiltAxisAngles[0] * Helper.ToRad);
                //    float3 Transformed = TiltMatrix * new float3(0, 0, 10);

                //    GridMovementX = new CubicGrid(new int3(1), new[] { -Transformed.X });
                //    GridMovementY = new CubicGrid(new int3(1), new[] { -Transformed.Y });
                //    OffsetWarped = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, 0).Select(v => new float2(v.X, v.Y)).First();

                //    GridMovementX = new CubicGrid(new int3(1), new[] { 0f });
                //    GridMovementY = new CubicGrid(new int3(1), new[] { 0f });
                //    OffsetDefault = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, 0).Select(v => new float2(v.X, v.Y)).First();

                //    Relative = OffsetWarped - OffsetDefault;
                //    //double[] StartParams

                //    //for (int t = 0; t < NTilts; t++)
                //    //{

                //    //}

                //}
                {
                }

                List<Image> CorrectedTilts = new List<Image>
                {
                    TiltData[IndicesSortedAbsoluteAngle[0]]
                };
                    List<int> TiltsProcessed = new List<int>()
                {
                    IndicesSortedAbsoluteAngle[0]
                };

                List<double> Scores = new List<double>();

                var FindClosestProcessedTilt = (int currentTilt) =>
                {
                    int Closest = TiltsProcessed.First();
                    float ClosestDist = MathF.Abs(Angles[currentTilt] - Angles[Closest]);

                    for (int i = 1; i < TiltsProcessed.Count; i++)
                    {
                        float Dist = MathF.Abs(Angles[currentTilt] - Angles[TiltsProcessed[i]]);
                        if (Dist < ClosestDist)
                        {
                            Closest = TiltsProcessed[i];
                            ClosestDist = Dist;
                        }
                    }

                    return Closest;
                };

                float[] FinalShiftsX = new float[GridMovementX.Values.Length];
                float[] FinalShiftsY = new float[GridMovementY.Values.Length];

                var MakeWarpedImage = (int t, Image warped) =>
                {
                    int2 DimsWarp = new int2(16);
                    float StepZ = 1f / Math.Max(NTilts - 1, 1);

                    float3[] InterpPoints = new float3[DimsWarp.Elements()];
                    for (int y = 0; y < DimsWarp.Y; y++)
                        for (int x = 0; x < DimsWarp.X; x++)
                            InterpPoints[y * DimsWarp.X + x] = new float3((float)x / (DimsWarp.X - 1), (float)y / (DimsWarp.Y - 1), t * StepZ);

                    float2[] WarpXY = Helper.Zip(GridMovementX.GetInterpolated(InterpPoints), GridMovementY.GetInterpolated(InterpPoints));
                    float[] WarpX = WarpXY.Select(v => v.X / (float)options.BinnedPixelSizeMean).ToArray();
                    float[] WarpY = WarpXY.Select(v => v.Y / (float)options.BinnedPixelSizeMean).ToArray();

                    Image TiltImagePrefiltered = TiltData[t].GetCopyGPU();
                    GPU.PrefilterForCubic(TiltImagePrefiltered.GetDevice(Intent.ReadWrite), TiltImagePrefiltered.Dims);

                    GPU.WarpImage(TiltImagePrefiltered.GetDevice(Intent.Read),
                                  warped.GetDevice(Intent.Write),
                                  DimsImage,
                                  WarpX,
                                  WarpY,
                                  DimsWarp,
                                  IntPtr.Zero);

                    TiltImagePrefiltered.Dispose();
                };

                for (int itilt = 1; itilt < NTilts; itilt++)
                {
                    int t = IndicesSortedAbsoluteAngle[itilt];
                    if (!UseTilt[t])
                        continue;

                    //if (t == 0 || t == NTilts - 1)
                    //    continue;

                    #region Make global reconstruction

                    Projector Reconstructor = new Projector(new int3(SizeReconstructionPadded), 1);
                    Projector Sampler = new Projector(new int3(SizeReconstructionPadded), 1);

                    Image CTFCoords = CTF.GetCTFCoords(SizeReconstructionPadded, SizeReconstructionPadded);
                    Image CTFExtracted = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, 1), true);

                    //int[] TwoClosest;
                    //{
                    //    List<int> AllTilts = Helper.ArrayOfSequence(0, NTilts, 1).ToList();
                    //    AllTilts.RemoveAll(v => !UseTilt[v] || v == t);
                    //    AllTilts.Sort((a, b) => Math.Abs(Angles[a] - Angles[t]).CompareTo(Math.Abs(Angles[b] - Angles[t])));
                    //    TwoClosest = AllTilts.Take(2).ToArray();
                    //}

                    for (int i = 0; i < itilt; i++)
                    {
                        int TiltID = IndicesSortedAbsoluteAngle[i];

                        if (i == itilt || !UseTilt[TiltID])// || (MathF.Sign(Angles[t]) != MathF.Sign(Angles[TiltID]) && i != 0))
                            continue;

                        float3 PositionInImage = GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, TiltID).First();
                        PositionInImage.X /= (float)options.BinnedPixelSizeMean;
                        PositionInImage.Y /= (float)options.BinnedPixelSizeMean;
                        int3 IntPosition = new int3(PositionInImage);
                        float2 Residual = new float2(-(PositionInImage.X - IntPosition.X), -(PositionInImage.Y - IntPosition.Y));
                        IntPosition.X -= DimsImage.X / 2;
                        IntPosition.Y -= DimsImage.Y / 2;
                        IntPosition.Z = 0;

                        Image Extracted = new Image(new int3(DimsImage));
                        GPU.Extract(CorrectedTilts[i].GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    new int3(DimsImage),
                                    new int3(DimsImage),
                                    Helper.ToInterleaved(new int3[] { IntPosition }),
                                    true,
                                    1);

                        Extracted.Multiply(1f / (SizeReconstructionPadded * SizeReconstructionPadded));
                        Image ExtractedPadded = Extracted.AsPadded(new int2(SizeReconstructionPadded)).AndDisposeParent();
                        ExtractedPadded.ShiftSlices(new[] { new float3(Residual.X - DimsImage.X, Residual.Y - DimsImage.Y, 0) });
                        Image ExtractedFT = ExtractedPadded.AsFFT().AndDisposeParent();

                        GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                          new[] { PositionInImage.Z },
                                          new[] { VolumeDimensionsPhysical / 2 },
                                          CTFCoords,
                                          null,
                                          TiltID,
                                          CTFExtracted,
                                          false);

                        ExtractedFT.Multiply(CTFExtracted);
                        CTFExtracted.Abs();

                        Reconstructor.BackProject(ExtractedFT,
                                                  CTFExtracted,
                                                  GetAnglesInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, new[] { new float3() }, TiltID),
                                                  Matrix2.Identity());

                        ExtractedFT.Fill(new float2(1, 0));
                        //ExtractedFT.Multiply(CTFExtracted);
                        CTFExtracted.Fill(1);

                        Sampler.BackProject(ExtractedFT,
                                            CTFExtracted,
                                            GetAnglesInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, new[] { new float3() }, TiltID),
                                            Matrix2.Identity());

                        ExtractedFT.Dispose();
                    }

                    int ClosestTilt = t;// FindClosestProcessedTilt(t);

                    Image Weights = Sampler.Weights.GetCopyGPU();
                    Weights.Min(1);

                    Reconstructor.Data.Multiply(Weights);
                    Sampler.Data.Multiply(Weights);
                    Weights.Dispose();

                    //Reconstructor.Weights.Fill(1);
                    //Sampler.Weights.Fill(1);
                    //Reconstructor.Weights.Max(1);
                    //Sampler.Weights.Max(1);

                    Image Reconstruction = Reconstructor.Reconstruct(false, "C1", null, -1, -1, -1, 0).AsPadded(new int3(SizeReconstruction)).AndDisposeParent();
                    Reconstructor.Dispose();
                    Reconstruction.MaskRectangularly(new int3(SizeReconstruction - 32, SizeReconstruction - 32, SizeReconstruction / 4), 16, true);
                    Reconstruction.WriteMRC("d_rec.mrc", true);

                    Image Samples = Sampler.Reconstruct(false, "C1", null, -1, -1, -1, 0).AsPadded(new int3(SizeReconstruction)).AndDisposeParent();
                    Sampler.Dispose();
                    Samples.MaskSpherically(SizeReconstruction - 32, 16, true);
                    Samples.MaskRectangularly(new int3(SizeReconstruction - 32, SizeReconstruction - 32, SizeReconstruction / 4), 16, true);
                    //Samples.WriteMRC("d_samples.mrc", true);

                    #endregion

                    #region Project average and filter currently missing tilt

                    Projector RecProjector = new Projector(Reconstruction, 2);
                    Projector SamplesProjector = new Projector(Samples, 1);
                    Reconstruction.Dispose();
                    Samples.Dispose();

                    Image NextTiltFull = RecProjector.ProjectToRealspace(new int2(SizeReconstruction), GetAnglesInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, new[] { new float3() }, ClosestTilt));
                    NextTiltFull.ShiftSlices(new float3[] { new float3(TiltAxisOffsetX[ClosestTilt], TiltAxisOffsetY[ClosestTilt], 0) / (float)options.BinnedPixelSizeMean });
                    NextTiltFull = NextTiltFull.AsPadded(DimsImage).AndDisposeParent();
                    NextTiltFull.Normalize();
                    RecProjector.Dispose();
                    NextTiltFull.WriteMRC($"d_nexttilt_{t:D2}.mrc", true);

                    Image NextTiltSamples = SamplesProjector.ProjectToRealspace(new int2(SizeReconstruction), GetAnglesInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, new[] { new float3() }, ClosestTilt));
                    NextTiltSamples = NextTiltSamples.AsPadded(DimsImage * 2).AndDisposeParent().AsFFT().AndDisposeParent().AsAmplitudes().AndDisposeParent();
                    NextTiltSamples.Multiply(1f / (DimsImage.Elements()));
                    SamplesProjector.Dispose();
                    //NextTiltSamples.WriteMRC("d_nexttiltsamples.mrc", true);

                    GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                      new[] { GetPositionsInOneTilt(new[] { VolumeDimensionsPhysical / 2 }, t).First().Z },
                                      new[] { VolumeDimensionsPhysical / 2 },
                                      CTFCoords,
                                      null,
                                      t,
                                      CTFExtracted,
                                      false);
                    CTFExtracted.Sign();

                    Image MissingTilt = TiltData[t].GetCopyGPU();
                    MissingTilt = MissingTilt.AsPaddedClamped(DimsImage * 2).AndDisposeParent();
                    MissingTilt.MaskRectangularly(new int3(DimsImage), DimsImage.X / 2, false);
                    MissingTilt = MissingTilt.AsFFT().AndDisposeParent();
                    MissingTilt.Multiply(CTFExtracted);
                    MissingTilt.Multiply(NextTiltSamples);
                    MissingTilt = MissingTilt.AsIFFT().AndDisposeParent().AsPadded(DimsImage).AndDisposeParent();
                    MissingTilt.Multiply(1f / (DimsImage.Elements()));
                    MissingTilt.Normalize();
                    MissingTilt.WriteMRC($"d_missingtilt_{t:D2}.mrc", true);

                    CTFExtracted.Dispose();
                    CTFCoords.Dispose();

                    #endregion

                    #region Make references from global projection

                    Image Refs;
                    {
                        float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, ClosestTilt).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                        int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                        float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                        IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                        Refs = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length));
                        GPU.Extract(NextTiltFull.GetDevice(Intent.Read),
                                    Refs.GetDevice(Intent.Write),
                                    NextTiltFull.Dims,
                                    Refs.Dims.Slice(),
                                    Helper.ToInterleaved(IntPositions),
                                    false,
                                    (uint)PositionGrid.Length);
                        Refs.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                        //Refs.Normalize();
                        //Refs.WriteMRC("d_refs.mrc", true);
                    }

                    #endregion

                    #region Perform optimization

                    {
                        float[] OriValuesX = new float[GridMovementX.Values.Length];// GridMovementX.Values.ToArray();
                        float[] OriValuesY = new float[GridMovementX.Values.Length];//GridMovementY.Values.ToArray();
                        int ParamsPerTilt = (int)GridMovementX.Dimensions.ElementsSlice();

                        Action<double[]> SetGrids = (input) =>
                        {
                            float[] NewValuesX = OriValuesX.ToArray();
                            float[] NewValuesY = OriValuesY.ToArray();
                            for (int i = 0; i < ParamsPerTilt; i++)
                            {
                                if (itilt > 10)
                                {
                                    if (i != ParamsPerTilt / 2)
                                    {
                                        NewValuesX[t * ParamsPerTilt + i] += (float)input[0 * 2 + 0];
                                        NewValuesY[t * ParamsPerTilt + i] += (float)input[0 * 2 + 1];
                                    }
                                    else
                                    {
                                        NewValuesX[t * ParamsPerTilt + i] += (float)input[1 * 2 + 0];
                                        NewValuesY[t * ParamsPerTilt + i] += (float)input[1 * 2 + 1];
                                    }
                                }
                                else
                                {
                                    NewValuesX[t * ParamsPerTilt + i] += (float)input[1 * 2 + 0];
                                    NewValuesY[t * ParamsPerTilt + i] += (float)input[1 * 2 + 1];
                                }
                            }

                            GridMovementX = new CubicGrid(GridMovementX.Dimensions, NewValuesX);
                            GridMovementY = new CubicGrid(GridMovementY.Dimensions, NewValuesY);
                        };

                        Func<double[], double> Eval = (input) =>
                        {
                            SetGrids(input);

                            float3[] PositionsInImage = GetPositionsInOneTilt(PositionGridPhysical, t).Select(v => v / (float)options.BinnedPixelSizeMean).ToArray();
                            int3[] IntPositions = PositionsInImage.Select(v => new int3(v)).ToArray();
                            float2[] Residuals = PositionsInImage.Select(v => new float2(-(v.X - (int)v.X), -(v.Y - (int)v.Y))).ToArray();
                            IntPositions = IntPositions.Select(v => new int3(v.X - SizeRegion / 2, v.Y - SizeRegion / 2, 0)).ToArray();

                            Image Raws = new Image(new int3(SizeRegion, SizeRegion, PositionGrid.Length));
                            GPU.Extract(MissingTilt.GetDevice(Intent.Read),
                                        Raws.GetDevice(Intent.Write),
                                        MissingTilt.Dims,
                                        Raws.Dims.Slice(),
                                        Helper.ToInterleaved(IntPositions),
                                        false,
                                        (uint)PositionGrid.Length);

                            GPU.FFT(Raws.GetDevice(Intent.Read), ExtractedFT1.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanForw);
                            ExtractedFT1.ShiftSlices(Residuals.Select(v => new float3(v.X, v.Y, 0)).ToArray());
                            GPU.IFFT(ExtractedFT1.GetDevice(Intent.Read), Raws.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)PositionGridPhysical.Length, PlanBack, true);
                            //Raws.Normalize();
                            //Raws.WriteMRC("d_raws.mrc", true);

                            Raws.Multiply(Refs);
                            //Raws.Multiply(Raws);
                            Raws.MultiplySlices(RegionMask);

                            Image Diff = Raws.AsSum3D().AndDisposeParent();
                            double Result = Diff.GetHost(Intent.Read)[0][0];
                            Diff.Dispose();

                            return Result;
                        };

                        int OptIterations = 0;
                        Func<double[], double[]> Grad = (input) =>
                        {
                            double Delta = 0.1;
                            double[] Result = new double[input.Length];

                            if (OptIterations++ > 8)
                                return Result;

                            for (int i = 0; i < input.Length; i++)
                            {
                                double[] InputPlus = input.ToArray();
                                InputPlus[i] += Delta;
                                double ScorePlus = Eval(InputPlus);

                                double[] InputMinus = input.ToArray();
                                InputMinus[i] -= Delta;
                                double ScoreMinus = Eval(InputMinus);

                                Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                            }

                            Console.WriteLine(Eval(input));

                            return Result;
                        };

                        double[] StartValues = new double[2 * 2];
                        BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartValues.Length, Eval, Grad);
                        Optimizer.Maximize(StartValues);

                        SetGrids(StartValues);
                        Scores.Add(Eval(StartValues));

                        Image CorrectedTilt = new Image(new int3(DimsImage));
                        MakeWarpedImage(t, CorrectedTilt);
                        CorrectedTilts.Add(CorrectedTilt);
                        TiltsProcessed.Add(t);

                        for (int i = 0; i < ParamsPerTilt; i++)
                        {
                            if (itilt > 10)
                            {
                                if (i != ParamsPerTilt / 2)
                                {
                                    FinalShiftsX[t * ParamsPerTilt + i] = OriValuesX[t * ParamsPerTilt + i] + (float)StartValues[0 * 2 + 0];
                                    FinalShiftsY[t * ParamsPerTilt + i] = OriValuesY[t * ParamsPerTilt + i] + (float)StartValues[0 * 2 + 1];
                                }
                                else
                                {
                                    FinalShiftsX[t * ParamsPerTilt + i] = OriValuesX[t * ParamsPerTilt + i] + (float)StartValues[1 * 2 + 0];
                                    FinalShiftsY[t * ParamsPerTilt + i] = OriValuesY[t * ParamsPerTilt + i] + (float)StartValues[1 * 2 + 1];
                                }
                            }
                            else
                            {
                                FinalShiftsX[t * ParamsPerTilt + i] = OriValuesX[t * ParamsPerTilt + i] + (float)StartValues[1 * 2 + 0];
                                FinalShiftsY[t * ParamsPerTilt + i] = OriValuesY[t * ParamsPerTilt + i] + (float)StartValues[1 * 2 + 1];
                            }
                        }

                        CorrectedTilt.WriteMRC($"d_corrected_{t:D2}.mrc", true);
                    }

                    #endregion

                    Refs.Dispose();
                    NextTiltFull.Dispose();
                    NextTiltSamples.Dispose();
                    MissingTilt.Dispose();

                    Console.WriteLine(GPU.GetFreeMemory(0) + " MB");
                }

                Console.WriteLine(Scores.Sum() / Scores.Count);

                GridMovementX = new CubicGrid(GridMovementX.Dimensions, FinalShiftsX);
                GridMovementY = new CubicGrid(GridMovementY.Dimensions, FinalShiftsY);
            }

            GPU.DestroyFFTPlan(PlanForw);
            GPU.DestroyFFTPlan(PlanBack);
            Extracted1.Dispose();
            Extracted2.Dispose();
            Extracted3.Dispose();
            ExtractedFT1.Dispose();
            ExtractedFT2.Dispose();
            ExtractedFT3.Dispose();

            foreach (var data in TiltData)
                data.Dispose();

            SaveMeta();
        }

        #endregion

        #region Multi-particle refinement

        public override void PerformMultiParticleRefinement(string workingDirectory,
                                                            ProcessingOptionsMPARefine optionsMPA,
                                                            Species[] allSpecies,
                                                            DataSource dataSource,
                                                            Image gainRef,
                                                            DefectModel defectMap,
                                                            Action<string> progressCallback)
        {
            int GPUID = GPU.GetDevice();
            HeaderEER.GroupNFrames = dataSource.EERGroupFrames;

            float BfactorWeightingThreshold = (float)optionsMPA.BFactorWeightingThreshold;

            //AreAnglesInverted = false;

            //MagnificationCorrection = new float3(1, 1, 0);

            if (CTF.ZernikeCoeffsOdd == null)
                CTF.ZernikeCoeffsOdd = new float[12];
            else if (CTF.ZernikeCoeffsOdd.Length < 12)
                CTF.ZernikeCoeffsOdd = Helper.Combine(CTF.ZernikeCoeffsOdd, new float[12 - CTF.ZernikeCoeffsOdd.Length]);

            if (CTF.ZernikeCoeffsEven == null)
                CTF.ZernikeCoeffsEven = new float[8];
            else if (CTF.ZernikeCoeffsEven.Length < 8)
                CTF.ZernikeCoeffsEven = Helper.Combine(CTF.ZernikeCoeffsEven, new float[8 - CTF.ZernikeCoeffsEven.Length]);

            #region Get particles belonging to this item; if there are not enough, abort

            string DataHash = GetDataHash();

            Dictionary<Species, Particle[]> SpeciesParticles = new Dictionary<Species, Particle[]>();
            foreach (var species in allSpecies)
                SpeciesParticles.Add(species, species.GetParticles(DataHash));

            if (SpeciesParticles.Select(p => p.Value.Length).Sum() < optionsMPA.MinParticlesPerItem)
                return;

            #endregion

            #region Figure out dimensions

            float SmallestAngPix = MathHelper.Min(allSpecies.Select(s => (float)s.PixelSize));
            float LargestBox = MathHelper.Max(allSpecies.Select(s => s.DiameterAngstrom)) * 2 / SmallestAngPix;

            float MinDose = MathHelper.Min(Dose), MaxDose = MathHelper.Max(Dose);
            float[] DoseInterpolationSteps = Dose.Select(d => (d - MinDose) / (MaxDose - MinDose)).ToArray();

            #endregion

            #region Load and preprocess tilt series

            progressCallback("Loading tilt series and masks...");

            decimal BinTimes = (decimal)Math.Log(SmallestAngPix / (float)dataSource.PixelSizeMean, 2.0);
            ProcessingOptionsTomoSubReconstruction OptionsDataLoad = new ProcessingOptionsTomoSubReconstruction()
            {
                PixelSize = dataSource.PixelSize,

                BinTimes = BinTimes,
                EERGroupFrames = dataSource.EERGroupFrames,
                GainPath = dataSource.GainPath,
                GainHash = "",
                GainFlipX = dataSource.GainFlipX,
                GainFlipY = dataSource.GainFlipY,
                GainTranspose = dataSource.GainTranspose,
                DefectsPath = dataSource.DefectsPath,
                DefectsHash = "",

                Dimensions = new float3((float)dataSource.DimensionsX,
                                        (float)dataSource.DimensionsY,
                                        (float)dataSource.DimensionsZ),

                Invert = true,
                NormalizeInput = true,
                NormalizeOutput = false,

                PrerotateParticles = true
            };

            VolumeDimensionsPhysical = OptionsDataLoad.DimensionsPhysical;

            Movie[] TiltMovies = null;
            Image[] TiltData = null;
            Image[] TiltMasks = null;

            Action LoadAndPreprocessTiltData = () =>
            {
                LoadMovieData(OptionsDataLoad, out TiltMovies, out TiltData, false, out _, out _);
                LoadMovieMasks(OptionsDataLoad, out TiltMasks);
                for (int z = 0; z < NTilts; z++)
                {
                    EraseDirt(TiltData[z], TiltMasks[z]);
                    TiltMasks[z]?.FreeDevice();

                    TiltData[z].SubtractMeanGrid(new int2(1));
                    TiltData[z].Bandpass(1f / LargestBox, 1f, false, 0f);

                    GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                                    TiltData[z].GetDevice(Intent.Write),
                                    (uint)TiltData[z].ElementsReal,
                                    1);

                    TiltData[z].Multiply(-1f);
                    //TiltData[z].Multiply(TiltMasks[z]);

                    //TiltData[z].FreeDevice();
                }
            };
            LoadAndPreprocessTiltData();

            Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after loading raw data of {Name}");

            #endregion

            #region Remove particles that are not contained in any of the tilt images

            foreach (var species in allSpecies)
            {
                if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                    continue;

                float Radius = species.DiameterAngstrom / 2;
                SpeciesParticles[species] = SpeciesParticles[species].Where(particle =>
                {
                    float3[] CoordinatesMoving = particle.GetCoordinateSeries(DoseInterpolationSteps);
                    float3[] ImagePositions = GetPositionInAllTilts(CoordinatesMoving);
                    bool AnyInside = false;

                    foreach (var p in ImagePositions)
                    {
                        float DistX = Math.Min(p.X, ImageDimensionsPhysical.X - p.X);
                        float DistY = Math.Min(p.Y, ImageDimensionsPhysical.Y - p.Y);
                        if (DistX >= Radius && DistY >= Radius)
                        {
                            AnyInside = true;
                            break;
                        }
                    }

                    return AnyInside;
                }).ToArray();
            }

            #endregion

            #region Compose optimization steps based on user's requests

            var OptimizationStepsWarp = new List<(WarpOptimizationTypes Type, int Iterations, string Name)>();
            {
                WarpOptimizationTypes TranslationComponents = 0;
                if (optionsMPA.DoImageWarp)
                    TranslationComponents |= WarpOptimizationTypes.ImageWarp;
                if (optionsMPA.DoVolumeWarp)
                    TranslationComponents |= WarpOptimizationTypes.VolumeWarp;

                if (TranslationComponents != 0)
                    OptimizationStepsWarp.Add((TranslationComponents, 10, "image & volume warping"));
            }
            {
                WarpOptimizationTypes AntisymComponents = 0;

                if (optionsMPA.DoZernike13)
                    AntisymComponents |= WarpOptimizationTypes.Zernike13;
                if (optionsMPA.DoZernike5)
                    AntisymComponents |= WarpOptimizationTypes.Zernike5;

                if (AntisymComponents != 0 && allSpecies.Any(s => s.ResolutionRefinement < (float)optionsMPA.MinimumCTFRefinementResolution))
                    OptimizationStepsWarp.Add((AntisymComponents, 10, "antisymmetrical aberrations"));
            }
            {
                if (optionsMPA.DoAxisAngles)
                    OptimizationStepsWarp.Add((WarpOptimizationTypes.AxisAngle, 6, "stage orientation"));
            }
            {
                WarpOptimizationTypes PoseComponents = 0;
                if (optionsMPA.DoParticlePoses)
                {
                    PoseComponents |= WarpOptimizationTypes.ParticlePosition;
                    PoseComponents |= WarpOptimizationTypes.ParticleAngle;
                }

                if (PoseComponents != 0)
                    OptimizationStepsWarp.Add((PoseComponents, 10, "particle poses"));
            }
            {
                if (optionsMPA.DoMagnification)
                    OptimizationStepsWarp.Add((WarpOptimizationTypes.Magnification, 4, "magnification"));
            }


            var OptimizationStepsCTF = new List<(CTFOptimizationTypes Type, int Iterations, string Name)>();
            {
                CTFOptimizationTypes DefocusComponents = 0;
                if (optionsMPA.DoDefocus)
                    DefocusComponents |= CTFOptimizationTypes.Defocus;
                if (optionsMPA.DoAstigmatismDelta)
                    DefocusComponents |= CTFOptimizationTypes.AstigmatismDelta;
                if (optionsMPA.DoAstigmatismAngle)
                    DefocusComponents |= CTFOptimizationTypes.AstigmatismAngle;
                if (optionsMPA.DoPhaseShift)
                    DefocusComponents |= CTFOptimizationTypes.PhaseShift;
                if (optionsMPA.DoCs)
                    DefocusComponents |= CTFOptimizationTypes.Cs;

                if (DefocusComponents != 0)
                    OptimizationStepsCTF.Add((DefocusComponents, 10, "CTF parameters"));

                CTFOptimizationTypes ZernikeComponents = 0;

                if (optionsMPA.DoZernike2)
                    ZernikeComponents |= CTFOptimizationTypes.Zernike2;
                if (optionsMPA.DoZernike4)
                    ZernikeComponents |= CTFOptimizationTypes.Zernike4;

                if (ZernikeComponents != 0)
                    OptimizationStepsCTF.Add((ZernikeComponents, 10, "symmetrical aberrations"));
            }

            #endregion

            Dictionary<Species, float[]> GoodParticleMasks = new Dictionary<Species, float[]>();

            if (optionsMPA.NIterations > 0)
            {
                #region Resize grids

                int AngleSpatialDim = 1;

                if (optionsMPA.DoAxisAngles)
                    if (GridAngleX == null || GridAngleX.Dimensions.X < AngleSpatialDim || GridAngleX.Dimensions.Z != NTilts)
                    {
                        GridAngleX = GridAngleX == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NTilts)) :
                                                          GridAngleX.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NTilts));
                        GridAngleY = GridAngleY == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NTilts)) :
                                                          GridAngleY.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NTilts));
                        GridAngleZ = GridAngleZ == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NTilts)) :
                                                          GridAngleZ.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NTilts));
                    }

                // Super crude way of figuring out how many parameters can be refined into the available particle signal
                //float OverallMass = 0;
                //foreach (var species in allSpecies)
                //    if (SpeciesParticles.ContainsKey(species))
                //        OverallMass += Math.Max((float)species.MolecularWeightkDa - 100, 0) * SpeciesParticles[species].Length;
                //float NParametersMax = OverallMass / 100 * 5;
                //NParametersMax -= GridAngleX.Values.Length * 3;
                //NParametersMax /= NTilts;
                //int MovementSpatialDim = Math.Min(5, Math.Max(1, (int)Math.Round(Math.Sqrt(NParametersMax))));
                int2 MovementSpatialDims = new int2(optionsMPA.ImageWarpWidth, optionsMPA.ImageWarpHeight);
                //MovementSpatialDim = 2;

                if (optionsMPA.DoImageWarp)
                    if (GridMovementX == null ||
                        GridMovementX.Dimensions.X != MovementSpatialDims.X ||
                        GridMovementX.Dimensions.Y != MovementSpatialDims.Y ||
                        GridMovementX.Dimensions.Z != NTilts)
                    {
                        int3 Dims = new int3(MovementSpatialDims.X, MovementSpatialDims.Y, NTilts);
                        GridMovementX = GridMovementX == null ? new CubicGrid(Dims) : GridMovementX.Resize(Dims);
                        GridMovementY = GridMovementY == null ? new CubicGrid(Dims) : GridMovementY.Resize(Dims);
                    }

                if (optionsMPA.DoVolumeWarp)
                {
                    int4 DimsVolumeWarp = new int4(optionsMPA.VolumeWarpWidth,
                                                   optionsMPA.VolumeWarpHeight,
                                                   optionsMPA.VolumeWarpDepth,
                                                   optionsMPA.VolumeWarpLength);
                    if (GridVolumeWarpX == null || GridVolumeWarpX.Dimensions != DimsVolumeWarp)
                    {
                        GridVolumeWarpX = GridVolumeWarpX == null ? new LinearGrid4D(DimsVolumeWarp) :
                                                                    GridVolumeWarpX.Resize(DimsVolumeWarp);
                        GridVolumeWarpY = GridVolumeWarpY == null ? new LinearGrid4D(DimsVolumeWarp) :
                                                                    GridVolumeWarpY.Resize(DimsVolumeWarp);
                        GridVolumeWarpZ = GridVolumeWarpZ == null ? new LinearGrid4D(DimsVolumeWarp) :
                                                                    GridVolumeWarpZ.Resize(DimsVolumeWarp);
                    }
                }

                #endregion

                #region Create species prerequisites and calculate spectral weights

                progressCallback("Calculating spectral weights...");

                Dictionary<Species, IntPtr[]> SpeciesParticleImages = new Dictionary<Species, IntPtr[]>();
                Dictionary<Species, IntPtr[]> SpeciesParticleQImages = new Dictionary<Species, IntPtr[]>();
                Dictionary<Species, float[]> SpeciesParticleDefoci = new Dictionary<Species, float[]>();
                Dictionary<Species, float[]> SpeciesContainmentMasks = new Dictionary<Species, float[]>();
                Dictionary<Species, float2[]> SpeciesParticleExtractedAt = new Dictionary<Species, float2[]>();
                Dictionary<Species, Image> SpeciesTiltWeights = new Dictionary<Species, Image>();
                Dictionary<Species, Image> SpeciesCTFWeights = new Dictionary<Species, Image>();
                Dictionary<Species, IntPtr> SpeciesParticleSubsets = new Dictionary<Species, IntPtr>();
                Dictionary<Species, (int Start, int End)> SpeciesParticleIDRanges = new Dictionary<Species, (int Start, int End)>();
                Dictionary<Species, int> SpeciesRefinementSize = new Dictionary<Species, int>();
                Dictionary<Species, int[]> SpeciesRelevantRefinementSizes = new Dictionary<Species, int[]>();
                Dictionary<Species, int> SpeciesCTFSuperresFactor = new Dictionary<Species, int>();

                Dictionary<Species, Image> CurrentWeightsDict = SpeciesTiltWeights;

                int NParticlesOverall = 0;

                float[][] AverageSpectrum1DAll = Helper.ArrayOfFunction(i => new float[128], NTilts);
                long[][] AverageSpectrum1DAllSamples = Helper.ArrayOfFunction(i => new long[128], NTilts);

                foreach (var species in allSpecies)
                {
                    if (SpeciesParticles[species].Length == 0)
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    SpeciesParticleIDRanges.Add(species, (NParticlesOverall, NParticlesOverall + NParticles));
                    NParticlesOverall += NParticles;

                    int Size = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / (float)OptionsDataLoad.BinnedPixelSizeMean);

                    int[] RelevantSizes = GetRelevantImageSizes(SizeFull, BfactorWeightingThreshold).Select(v => Math.Min(Size, v)).ToArray();

                    float Radius = species.DiameterAngstrom / 2;
                    float[] ContainmentMask = Helper.ArrayOfConstant(1f, NParticles * NTilts);

                    #region Extract particle images

                    //Image AverageRealspace = new Image(new int3(SizeFull, SizeFull, NTilts), true, true);
                    Image AverageAmplitudes = new Image(new int3(SizeFull, SizeFull, NTilts), true);
                    //Image ImagesRealspace = new Image(new int3(SizeFull, SizeFull, NTilts));
                    Image ImagesAmplitudes = new Image(new int3(SizeFull, SizeFull, NTilts), true);

                    Image ExtractResult = new Image(new int3(SizeFull, SizeFull, NTilts));
                    Image ExtractResultFT = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, NTilts), true, true);
                    //Image ExtractResultFTCropped = new Image(IntPtr.Zero, new int3(Size, Size, NTilts), true, true);

                    int[] PlanForw = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeFull, SizeFull, 1), (uint)NTilts), 1);

                    Helper.ForCPU(0, NParticles, 1, threadID => GPU.SetDevice(GPUID), (p, threadID) =>
                    {
                        float3[] Coords = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                        GetImagesForOneParticle(OptionsDataLoad,
                                                TiltData,
                                                SizeFull,
                                                Coords,
                                                PlanForw[threadID],
                                                ParticleDiameterPix,
                                                16,
                                                true,
                                                ExtractResult,
                                                ExtractResultFT);

                        float3[] ImageCoords = GetPositionInAllTilts(Coords);
                        for (int t = 0; t < NTilts; t++)
                        {
                            float3 Pos = ImageCoords[t];

                            float DistX = Math.Min(Pos.X, ImageDimensionsPhysical.X - Pos.X);
                            float DistY = Math.Min(Pos.Y, ImageDimensionsPhysical.Y - Pos.Y);
                            if (DistX < Radius || DistY < Radius)
                                ContainmentMask[p * NTilts + t] = 0;
                        }

                        //GPU.PadFT(ImagesFT.GetDevice(Intent.Read),
                        //          ExtractResultFTCropped.GetDevice(Intent.Write),
                        //          ImagesFT.Dims.Slice(),
                        //          ExtractResultFTCropped.Dims.Slice(),
                        //          (uint)NTilts);
                        //Image ImagesFTCropped = ImagesFT.AsPadded(new int2(Size));
                        //ImagesFT.Dispose();

                        GPU.Amplitudes(ExtractResultFT.GetDevice(Intent.Read),
                                       ImagesAmplitudes.GetDevice(Intent.Write),
                                       (uint)ExtractResultFT.ElementsComplex);
                        ImagesAmplitudes.Multiply(ImagesAmplitudes);
                        lock (AverageAmplitudes)
                            AverageAmplitudes.Add(ImagesAmplitudes);

                        //ImagesFTCropped.Multiply(Weights);

                        //lock (AverageRealspace)
                        //    AverageRealspace.Add(ExtractResultFT);

                        //ImagesFTCropped.Dispose();
                    }, null);

                    ExtractResult.Dispose();
                    ExtractResultFT.Dispose();
                    //ExtractResultFTCropped.Dispose();

                    ImagesAmplitudes.Dispose();

                    for (int i = 0; i < PlanForw.Length; i++)
                        GPU.DestroyFFTPlan(PlanForw[i]);

                    //AverageRealspace.Multiply(1f / NParticles);
                    //if (GPUID == 0)
                    //    AverageRealspace.AsIFFT().WriteMRC("d_avgreal.mrc", true);
                    //AverageRealspace.Dispose();

                    //ImagesRealspace.Dispose();

                    #endregion

                    #region Calculate spectra

                    //AverageRealspace.Multiply(1f / NParticles);
                    AverageAmplitudes.Multiply(1f / NParticles);
                    // if (GPUID == 0)
                    //     AverageAmplitudes.WriteMRC($"d_avgamps_{species.Name}.mrc", true);

                    float[][] Amps2D = AverageAmplitudes.GetHost(Intent.Read);

                    for (int t = 0; t < NTilts; t++)
                    {
                        Helper.ForEachElementFT(new int2(SizeFull), (x, y, xx, yy, r, angle) =>
                        {
                            int idx = (int)Math.Round(r / (SizeFull / 2) * AverageSpectrum1DAll[t].Length);
                            if (idx < AverageSpectrum1DAll[t].Length)
                            {
                                AverageSpectrum1DAll[t][idx] += Amps2D[t][y * (SizeFull / 2 + 1) + x] * NParticles;
                                AverageSpectrum1DAllSamples[t][idx] += NParticles;
                            }
                        });
                    }

                    AverageAmplitudes.Dispose();

                    #endregion

                    #region Defoci and extraction positions

                    float[] Defoci = new float[NParticles * NTilts];
                    float2[] ExtractedAt = new float2[NParticles * NTilts];

                    for (int p = 0; p < NParticles; p++)
                    {
                        float3[] Positions = GetPositionInAllTilts(Particles[p].GetCoordinateSeries(DoseInterpolationSteps));
                        for (int t = 0; t < NTilts; t++)
                        {
                            Defoci[p * NTilts + t] = Positions[t].Z;
                            ExtractedAt[p * NTilts + t] = new float2(Positions[t].X, Positions[t].Y);
                        }
                    }

                    #endregion

                    #region Subset indices

                    int[] Subsets = Particles.Select(p => p.RandomSubset).ToArray();
                    IntPtr SubsetsPtr = GPU.MallocDeviceFromHostInt(Subsets, Subsets.Length);

                    #endregion

                    #region CTF superres factor

                    CTF MaxDefocusCTF = GetTiltCTF(IndicesSortedDose[0]);
                    int MinimumBoxSize = Math.Max(species.HalfMap1Projector[GPUID].Dims.X, MaxDefocusCTF.GetAliasingFreeSize(species.ResolutionRefinement, species.DiameterAngstrom / (species.ResolutionRefinement / 2)));
                    float CTFSuperresFactor = (float)Math.Ceiling((float)MinimumBoxSize / species.HalfMap1Projector[GPUID].Dims.X);

                    #endregion

                    SpeciesParticleDefoci.Add(species, Defoci);
                    SpeciesContainmentMasks.Add(species, ContainmentMask);
                    SpeciesParticleExtractedAt.Add(species, ExtractedAt);
                    SpeciesParticleSubsets.Add(species, SubsetsPtr);
                    SpeciesRefinementSize.Add(species, Size);
                    SpeciesRelevantRefinementSizes.Add(species, RelevantSizes);
                    SpeciesCTFSuperresFactor.Add(species, (int)CTFSuperresFactor);

                    species.HalfMap1Projector[GPUID].PutTexturesOnDevice();
                    species.HalfMap2Projector[GPUID].PutTexturesOnDevice();
                }

                #region Calculate 1D PS averaged over all species and particles

                for (int t = 0; t < NTilts; t++)
                {
                    for (int i = 0; i < AverageSpectrum1DAll[t].Length; i++)
                        AverageSpectrum1DAll[t][i] /= Math.Max(1, AverageSpectrum1DAllSamples[t][i]);

                    float SpectrumMean = MathHelper.Mean(AverageSpectrum1DAll[t]);
                    for (int i = 0; i < AverageSpectrum1DAll[t].Length; i++)
                        AverageSpectrum1DAll[t][i] /= SpectrumMean;

                    for (int i = 0; i < AverageSpectrum1DAll[t].Length; i++)
                        if (AverageSpectrum1DAll[t][i] <= 0)
                        {
                            for (int j = 0; j < AverageSpectrum1DAll[t].Length; j++)
                            {
                                if (i - j >= 0 && AverageSpectrum1DAll[t][i - j] > 0)
                                {
                                    AverageSpectrum1DAll[t][i] = AverageSpectrum1DAll[t][i - j];
                                    break;
                                }

                                if (i + j < AverageSpectrum1DAll[t].Length && AverageSpectrum1DAll[t][i + j] > 0)
                                {
                                    AverageSpectrum1DAll[t][i] = AverageSpectrum1DAll[t][i + j];
                                    break;
                                }
                            }
                        }

                    if (AverageSpectrum1DAll[t].Any(v => v <= 0))
                        throw new Exception("The 1D amplitude spectrum contains zeros, which it really shouldn't! Can't proceed.");
                }

                #endregion

                #region Calculate weights

                foreach (var species in allSpecies)
                {
                    if (SpeciesParticles[species].Length == 0)
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;

                    int Size = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / (float)OptionsDataLoad.BinnedPixelSizeMean);

                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                    long ElementsSliceComplex = (Size / 2 + 1) * Size;

                    #region Dose weighting

                    ProcessingOptionsTomoSubReconstruction OptionsWeights = new ProcessingOptionsTomoSubReconstruction()
                    {
                        PixelSize = dataSource.PixelSize,

                        BinTimes = (decimal)Math.Log((species.ResolutionRefinement / 2) / (float)dataSource.PixelSizeMean, 2.0),

                        Dimensions = new float3((float)dataSource.DimensionsX,
                                                (float)dataSource.DimensionsY,
                                                (float)dataSource.DimensionsZ),

                        Invert = true,
                        NormalizeInput = true,
                        NormalizeOutput = false,

                        PrerotateParticles = true
                    };

                    Image CTFCoords = CTF.GetCTFCoords(Size, Size);
                    Image Weights = GetCTFsForOneParticle(OptionsWeights, VolumeDimensionsPhysical / 2, CTFCoords, null, true, true);
                    //Image VanillaWeights = Weights.GetCopy();
                    CTFCoords.Dispose();

                    #endregion

                    #region Divide weights by 1D PS, and create a 20 A high-passed version for CTF refinement

                    float[][] WeightsData = Weights.GetHost(Intent.ReadWrite);
                    for (int t = 0; t < NTilts; t++)
                    {
                        Helper.ForEachElementFT(new int2(Size), (x, y, xx, yy, r, angle) =>
                        {
                            if (r < Size / 2)
                            {
                                int idx = Math.Min(AverageSpectrum1DAll[t].Length - 1,
                                                   (int)Math.Round(r / (Size / 2) *
                                                                   (float)OptionsDataLoad.BinnedPixelSizeMean /
                                                                   (species.ResolutionRefinement / 2) *
                                                                   AverageSpectrum1DAll[t].Length));

                                WeightsData[t][y * (Size / 2 + 1) + x] /= AverageSpectrum1DAll[t][idx];
                            }
                            else
                            {
                                WeightsData[t][y * (Size / 2 + 1) + x] = 0;
                            }
                        });
                    }

                    //Weights.FreeDevice();
                    // if (GPUID == 0)
                    //     Weights.WriteMRC($"d_weights_{species.Name}.mrc", true);

                    Image WeightsRelevantlySized = new Image(new int3(Size, Size, NTilts), true);
                    for (int t = 0; t < NTilts; t++)
                        GPU.CropFTRealValued(Weights.GetDeviceSlice(t, Intent.Read),
                                            WeightsRelevantlySized.GetDeviceSlice(t, Intent.Write),
                                            Weights.Dims.Slice(),
                                            new int3(RelevantSizes[t]).Slice(),
                                            1);
                    // if (GPUID == 0)
                    //     WeightsRelevantlySized.WriteMRC($"d_weightsrelevant_{species.Name}.mrc", true);
                    Weights.Dispose();

                    Image CTFWeights = WeightsRelevantlySized.GetCopyGPU();
                    float[][] CTFWeightsData = CTFWeights.GetHost(Intent.ReadWrite);
                    for (int t = 0; t < CTFWeightsData.Length; t++)
                    {
                        int RelevantSize = RelevantSizes[t];
                        float R20 = Size * (species.ResolutionRefinement / 2 / 20f);
                        Helper.ForEachElementFT(new int2(RelevantSize), (x, y, xx, yy, r, angle) =>
                        {
                            float Weight = 1 - Math.Max(0, Math.Min(1, R20 - r));
                            CTFWeightsData[t][y * (RelevantSize / 2 + 1) + x] *= Weight;
                        });
                    }

                    CTFWeights.FreeDevice();
                    // if (GPUID == 0)
                    //     CTFWeights.WriteMRC($"d_ctfweights_{species.Name}.mrc", true);

                    #endregion

                    SpeciesCTFWeights.Add(species, CTFWeights);
                    SpeciesTiltWeights.Add(species, WeightsRelevantlySized);
                }

                #endregion

                // Remove original tilt image data from device, and dispose masks
                for (int t = 0; t < NTilts; t++)
                {
                    if (TiltMasks != null)
                        TiltMasks[t]?.FreeDevice();
                    //TiltData[t].FreeDevice();
                }

                Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after spectra estimation of {Name}");

                #endregion

                #region Tilt movie refinement

                if (optionsMPA.DoTiltMovies)
                {
                    Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB before movie refinement of {Name}");

                    Image StackForExport = null;
                    Image StackAverage = null;
                    Image AveragePlane = null;

                    for (int itilt = 0; itilt < NTilts; itilt++)
                    {
                        progressCallback($"Refining tilt movie {itilt + 1}/{NTilts}");

                        Movie tiltMovie = TiltMovies[itilt];
                        tiltMovie.NFrames = MapHeader.ReadFromFile(tiltMovie.DataPath).Dimensions.Z;

                        Image[] MovieData;
                        tiltMovie.LoadFrameData(OptionsDataLoad, gainRef, defectMap, out MovieData);

                        int3 StackDims = new int3(MovieData[0].Dims.X, MovieData[0].Dims.Y, MovieData.Length);
                        if (StackForExport == null || StackDims != StackForExport.Dims)
                        {
                            StackForExport?.Dispose();
                            StackForExport = new Image(IntPtr.Zero, StackDims);
                        }
                        for (int z = 0; z < MovieData.Length; z++)
                            GPU.CopyDeviceToDevice(MovieData[z].GetDevice(Intent.Read),
                                                   StackForExport.GetDeviceSlice(z, Intent.Write),
                                                   MovieData[z].ElementsReal);

                        if (StackAverage == null || StackAverage.Dims != StackForExport.Dims.Slice())
                        {
                            StackAverage?.Dispose();
                            StackAverage = new Image(IntPtr.Zero, StackForExport.Dims.Slice());
                            AveragePlane?.Dispose();
                            AveragePlane = new Image(IntPtr.Zero, StackForExport.Dims.Slice());
                        }
                        GPU.ReduceMean(StackForExport.GetDevice(Intent.Read),
                                       StackAverage.GetDevice(Intent.Write),
                                       (uint)StackAverage.ElementsReal,
                                       (uint)StackForExport.Dims.Z,
                                       1);
                        float[] AveragePlaneData = MathHelper.FitAndGeneratePlane(StackAverage.GetHost(Intent.Read)[0], new int2(StackAverage.Dims));
                        GPU.CopyHostToDevice(AveragePlaneData, AveragePlane.GetDevice(Intent.Write), AveragePlaneData.Length);

                        for (int z = 0; z < MovieData.Length; z++)
                        {
                            MovieData[z].Subtract(AveragePlane);
                            //MovieData[z].Bandpass(1f / LargestBox, 1f, false, 0f);

                            //MovieData[z].Multiply(-1f);
                            //MovieData[z].FreeDevice();
                        }

                        Dictionary<Species, Image> MovieSpeciesWeights = new Dictionary<Species, Image>();
                        foreach (var species in allSpecies)
                        {
                            if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                                continue;

                            Image Weights = new Image(IntPtr.Zero, new int3(SpeciesTiltWeights[species].Dims.X, SpeciesTiltWeights[species].Dims.Y, MovieData.Length), true);
                            for (int i = 0; i < MovieData.Length; i++)
                            {
                                GPU.CopyDeviceToDevice((species.ResolutionRefinement < 10 ? SpeciesCTFWeights : SpeciesTiltWeights)[species].GetDeviceSlice(itilt, Intent.Read),
                                                       Weights.GetDeviceSlice(i, Intent.Write),
                                                       Weights.ElementsSliceReal);
                            }
                            MovieSpeciesWeights.Add(species, Weights);
                        }

                        PerformMultiParticleRefinementOneTiltMovie(workingDirectory,
                                                                   optionsMPA,
                                                                   allSpecies,
                                                                   dataSource,
                                                                   tiltMovie,
                                                                   MovieData,
                                                                   itilt,
                                                                   SpeciesParticles,
                                                                   SpeciesParticleSubsets,
                                                                   SpeciesParticleIDRanges,
                                                                   SpeciesContainmentMasks,
                                                                   SpeciesRefinementSize,
                                                                   SpeciesRelevantRefinementSizes,
                                                                   MovieSpeciesWeights,
                                                                   SpeciesCTFSuperresFactor);

                        foreach (var pair in MovieSpeciesWeights)
                            pair.Value.Dispose();

                        foreach (var frame in MovieData)
                            frame.Dispose();

                        tiltMovie.ExportMovie(StackForExport, tiltMovie.OptionsMovieExport);

                        tiltMovie.SaveMeta();
                    }

                    StackForExport.Dispose();
                    StackAverage.Dispose();
                    AveragePlane.Dispose();

                    for (int t = 0; t < NTilts; t++)
                        TiltData[t].FreeDevice();

                    LoadAndPreprocessTiltData();

                    for (int t = 0; t < NTilts; t++)
                    {
                        if (TiltMasks != null)
                            TiltMasks[t]?.FreeDevice();
                        //TiltData[t].FreeDevice();
                    }

                    Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after movie refinement of {Name}");
                }

                #endregion

                #region Allocate pinned host memory for extracted particle images

                foreach (var species in allSpecies)
                {
                    int NParticles = SpeciesParticles[species].Length;
                    if (NParticles == 0)
                        continue;

                    int Size = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                    IntPtr[] ImagesFTPinned = Helper.ArrayOfFunction(t =>
                    {
                        long Footprint = (new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles;
                        if (optionsMPA.UseHostMemory)
                            return GPU.MallocHostPinned(Footprint);
                        else
                            return GPU.MallocDevice(Footprint);
                    }, NTilts);

                    IntPtr[] ImagesFTQPinned = null;
                    if (species.DoEwald)
                        ImagesFTQPinned = Helper.ArrayOfFunction(t => GPU.MallocDevice((new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles), NTilts);
                    GPU.CheckGPUExceptions();

                    SpeciesParticleImages.Add(species, ImagesFTPinned);
                    if (species.DoEwald)
                        SpeciesParticleQImages.Add(species, ImagesFTQPinned);
                }

                #endregion

                #region Helper functions

                Action<bool> ReextractPaddedParticles = (CorrectBeamTilt) =>
                {
                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                    int BatchSize = optionsMPA.BatchSize;

                    foreach (var species in allSpecies)
                    {
                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;

                        if (NParticles == 0 || !SpeciesCTFSuperresFactor.ContainsKey(species))
                            continue;

                        int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                        int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                        int[] SizesRelevant = SpeciesRelevantRefinementSizes[species];

                        float AngPixRefine = species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                        float2[] ExtractedAt = SpeciesParticleExtractedAt[species];

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);
                        Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefineSuper);
                        Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);
                        //GammaCorrection.WriteMRC("d_gamma.mrc", true);

                        Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize));
                        Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCroppedFTRelevantSize = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                        //Image Average = new Image(new int3(SizeRefine, SizeRefine, BatchSize));

                        int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                        int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                        if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                        {
                            Console.WriteLine($"SizeFullSuper = {SizeFullSuper}, BatchSize = {BatchSize}, free memory = {GPU.GetFreeMemory(GPUID)}");
                            throw new Exception("No FFT plans created!");
                        }

                        bool[] PQReverse = { species.EwaldReverse, !species.EwaldReverse };
                        IntPtr[][] PQStorage = species.DoEwald ? new[] { SpeciesParticleImages[species], SpeciesParticleQImages[species] } :
                                                                 new[] { SpeciesParticleImages[species] };

                        for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                        {
                            for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                            {
                                int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                                IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                                float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                                float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                                for (int t = 0; t < NTilts; t++)
                                {
                                    float3[] CoordinatesTilt = new float3[CurBatch];
                                    for (int p = 0; p < CurBatch; p++)
                                        CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];

                                    float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);

                                    float[] Defoci = new float[CurBatch];
                                    int3[] ExtractOrigins = new int3[CurBatch];
                                    float3[] ResidualShifts = new float3[BatchSize];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                        ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                        ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                        Defoci[p] = ImageCoords[p].Z;
                                        ExtractedAt[(batchStart + p) * NTilts + t] = new float2(ImageCoords[p]);
                                    }

                                    GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                                Extracted.GetDevice(Intent.Write),
                                                TiltData[t].Dims.Slice(),
                                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                                Helper.ToInterleaved(ExtractOrigins),
                                                true,
                                                (uint)CurBatch);

                                    GPU.FFT(Extracted.GetDevice(Intent.Read),
                                            ExtractedFT.GetDevice(Intent.Write),
                                            new int3(SizeFullSuper, SizeFullSuper, 1),
                                            (uint)CurBatch,
                                            PlanForwSuper);

                                    ExtractedFT.ShiftSlices(ResidualShifts);
                                    ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                    GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                               Extracted.GetDevice(Intent.Write),
                                               new int3(SizeFullSuper, SizeFullSuper, 1),
                                               new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                               (uint)CurBatch);

                                    if (CorrectBeamTilt)
                                        GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                           PhaseCorrection.GetDevice(Intent.Read),
                                                                           Extracted.GetDevice(Intent.Write),
                                                                           PhaseCorrection.ElementsSliceComplex,
                                                                           (uint)CurBatch);

                                    if (species.DoEwald)
                                    {
                                        GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, PQReverse[iewald], ExtractedCTF, true);

                                        GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                          ExtractedCTF.GetDevice(Intent.Read),
                                                                          ExtractedFT.GetDevice(Intent.Write),
                                                                          ExtractedCTF.ElementsComplex,
                                                                          1);
                                    }
                                    else
                                    {
                                        GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                        GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                                          ExtractedCTF.GetDevice(Intent.Read),
                                                                          ExtractedFT.GetDevice(Intent.Write),
                                                                          ExtractedCTF.ElementsComplex,
                                                                          1);
                                    }

                                    GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                             Extracted.GetDevice(Intent.Write),
                                             new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                             (uint)CurBatch,
                                             PlanBackSuper,
                                             false);

                                    GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                                    ExtractedCropped.GetDevice(Intent.Write),
                                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                    new int3(SizeRefine, SizeRefine, 1),
                                                    (uint)CurBatch);

                                    GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                                   ExtractedCropped.GetDevice(Intent.Write),
                                                   ExtractedCropped.Dims.Slice(),
                                                   ParticleDiameterPix / 2f,
                                                   16 * AngPixExtract / AngPixRefine,
                                                   true,
                                                   (uint)CurBatch);

                                    //Average.Add(ExtractedCropped);

                                    GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                            ExtractedCroppedFT.GetDevice(Intent.Write),
                                            new int3(SizeRefine, SizeRefine, 1),
                                            (uint)CurBatch,
                                            PlanForw);

                                    ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                    GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                               ExtractedCroppedFTRelevantSize.GetDevice(Intent.Write),
                                               new int3(SizeRefine).Slice(),
                                               new int3(SizesRelevant[t]).Slice(),
                                               (uint)CurBatch);

                                    GPU.CopyDeviceToHostPinned(ExtractedCroppedFTRelevantSize.GetDevice(Intent.Read),
                                                               new IntPtr((long)PQStorage[iewald][t] + (new int3(SizesRelevant[t]).Slice().ElementsFFT()) * 2 * batchStart * sizeof(float)),
                                                               (new int3(SizesRelevant[t]).Slice().ElementsFFT()) * 2 * CurBatch);
                                }
                            }
                        }

                        //Average.WriteMRC("d_average.mrc", true);
                        //Average.Dispose();

                        CoordsCTF.Dispose();
                        PhaseCorrection.Dispose();
                        GammaCorrection.Dispose();
                        Extracted.Dispose();
                        ExtractedFT.Dispose();
                        ExtractedCropped.Dispose();
                        ExtractedCroppedFT.Dispose();
                        ExtractedCroppedFTRelevantSize.Dispose();
                        ExtractedCTF.Dispose();

                        GPU.DestroyFFTPlan(PlanForwSuper);
                        GPU.DestroyFFTPlan(PlanBackSuper);
                        GPU.DestroyFFTPlan(PlanForw);
                    }

                    //foreach (var image in TiltData)
                    //    image.FreeDevice();

                    GPU.CheckGPUExceptions();
                };

                Func<float2[]> GetRawShifts = () =>
                {
                    float2[] Result = new float2[NParticlesOverall * NTilts];

                    foreach (var species in allSpecies)
                    {
                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        float SpeciesAngPix = species.ResolutionRefinement / 2;
                        if (NParticles == 0)
                            continue;

                        int Offset = SpeciesParticleIDRanges[species].Start;

                        float3[] ParticlePositions = new float3[NParticles * NTilts];
                        for (int p = 0; p < NParticles; p++)
                        {
                            float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                            for (int t = 0; t < NTilts; t++)
                                ParticlePositions[p * NTilts + t] = Positions[t];
                        }

                        float3[] ParticlePositionsProjected = GetPositionInAllTilts(ParticlePositions);
                        float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                        for (int p = 0; p < NParticles; p++)
                            for (int t = 0; t < NTilts; t++)
                                Result[(Offset + p) * NTilts + t] = (new float2(ParticlePositionsProjected[p * NTilts + t]) - ParticleExtractedAt[p * NTilts + t]);
                    }

                    return Result;
                };

                Func<float2, Species, float[]> GetRawCCSpecies = (shiftBias, Species) =>
                {
                    Particle[] Particles = SpeciesParticles[Species];

                    int NParticles = Particles.Length;
                    float AngPixRefine = Species.ResolutionRefinement / 2;

                    float[] SpeciesResult = new float[NParticles * NTilts * 3];
                    if (NParticles == 0)
                        return SpeciesResult;

                    float[] SpeciesResultQ = new float[NParticles * NTilts * 3];

                    float3[] ParticlePositions = new float3[NParticles * NTilts];
                    float3[] ParticleAngles = new float3[NParticles * NTilts];
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);
                        float3[] Angles = Particles[p].GetAngleSeries(DoseInterpolationSteps);

                        for (int t = 0; t < NTilts; t++)
                        {
                            ParticlePositions[p * NTilts + t] = Positions[t];
                            ParticleAngles[p * NTilts + t] = Angles[t];
                        }
                    }

                    float3[] ParticlePositionsProjected = GetPositionInAllTilts(ParticlePositions);
                    float3[] ParticleAnglesInTilts = GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles);

                    float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[Species];
                    float2[] ParticleShifts = new float2[NTilts * NParticles];
                    for (int p = 0; p < NParticles; p++)
                        for (int t = 0; t < NTilts; t++)
                            ParticleShifts[p * NTilts + t] = (new float2(ParticlePositionsProjected[p * NTilts + t]) - ParticleExtractedAt[p * NTilts + t] + shiftBias) / AngPixRefine;

                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[Species];

                    int SizeRefine = Species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;

                    Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                    Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NTilts), true, true);
                    for (int t = 0; t < NTilts; t++)
                        GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                    PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                                    PhaseCorrection.Dims.Slice(),
                                    new int3(RelevantSizes[t]).Slice(),
                                    1);

                    GPU.MultiParticleDiff(SpeciesResult,
                                            SpeciesParticleImages[Species],
                                            SizeRefine,
                                            RelevantSizes,
                                            Helper.ToInterleaved(ParticleShifts),
                                            Helper.ToInterleaved(ParticleAnglesInTilts),
                                            MagnificationCorrection.ToVec(),
                                            (Species.ResolutionRefinement < 8 ? SpeciesCTFWeights : SpeciesTiltWeights)[Species].GetDevice(Intent.Read),
                                            PhaseCorrectionAll.GetDevice(Intent.Read),
                                            Species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize) : 0,
                                            Species.CurrentMaxShellRefinement,
                                            new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                            new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                            Species.HalfMap1Projector[GPUID].Oversampling,
                                            Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                            SpeciesParticleSubsets[Species],
                                            NParticles,
                                            NTilts);

                    if (Species.DoEwald)
                        GPU.MultiParticleDiff(SpeciesResultQ,
                                                SpeciesParticleQImages[Species],
                                                SizeRefine,
                                                RelevantSizes,
                                                Helper.ToInterleaved(ParticleShifts),
                                                Helper.ToInterleaved(ParticleAnglesInTilts),
                                                MagnificationCorrection.ToVec(),
                                                (Species.ResolutionRefinement < 8 ? SpeciesCTFWeights : SpeciesTiltWeights)[Species].GetDevice(Intent.Read),
                                                PhaseCorrectionAll.GetDevice(Intent.Read),
                                                -CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize),
                                                Species.CurrentMaxShellRefinement,
                                                new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                                new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                                Species.HalfMap1Projector[GPUID].Oversampling,
                                                Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                                SpeciesParticleSubsets[Species],
                                                NParticles,
                                                NTilts);

                    PhaseCorrection.Dispose();
                    PhaseCorrectionAll.Dispose();

                    if (Species.DoEwald)
                        for (int i = 0; i < SpeciesResult.Length; i++)
                            SpeciesResult[i] += SpeciesResultQ[i];

                    float[] ContainmentMask = SpeciesContainmentMasks[Species];
                    for (int i = 0; i < NParticles * NTilts; i++)
                    {
                        SpeciesResult[i * 3 + 0] *= ContainmentMask[i];
                        SpeciesResult[i * 3 + 1] *= ContainmentMask[i];
                        SpeciesResult[i * 3 + 2] *= ContainmentMask[i];
                    }

                    return SpeciesResult;
                };

                Func<float2, float[]> GetRawCC = (shiftBias) =>
                {
                    float[] Result = new float[NParticlesOverall * NTilts * 3];

                    for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                    {
                        Species Species = allSpecies[ispecies];
                        Particle[] Particles = SpeciesParticles[Species];

                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        float[] SpeciesResult = GetRawCCSpecies(shiftBias, Species);

                        int Offset = SpeciesParticleIDRanges[Species].Start * NTilts * 3;
                        Array.Copy(SpeciesResult, 0, Result, Offset, SpeciesResult.Length);
                    }

                    GPU.CheckGPUExceptions();
                    //Console.WriteLine(GPU.GetFreeMemory(GPUID));

                    return Result;
                };

                Func<double[]> GetPerTiltCC = () =>
                {
                    double[] Result = new double[NTilts * 3];
                    float[] RawResult = GetRawCC(new float2(0));

                    for (int p = 0; p < NParticlesOverall; p++)
                        for (int t = 0; t < NTilts; t++)
                        {
                            Result[t * 3 + 0] += RawResult[(p * NTilts + t) * 3 + 0];
                            Result[t * 3 + 1] += RawResult[(p * NTilts + t) * 3 + 1];
                            Result[t * 3 + 2] += RawResult[(p * NTilts + t) * 3 + 2];
                        }

                    Result = Helper.ArrayOfFunction(t => Result[t * 3 + 0] / Math.Max(1e-10, Math.Sqrt(Result[t * 3 + 1] * Result[t * 3 + 2])) * 100 * NParticlesOverall, NTilts);

                    return Result;
                };

                Func<double[]> GetPerParticleCC = () =>
                {
                    double[] Result = new double[NParticlesOverall * 3];
                    float[] RawResult = GetRawCC(new float2(0));

                    for (int p = 0; p < NParticlesOverall; p++)
                        for (int t = 0; t < NTilts; t++)
                        {
                            Result[p * 3 + 0] += RawResult[(p * NTilts + t) * 3 + 0];
                            Result[p * 3 + 1] += RawResult[(p * NTilts + t) * 3 + 1];
                            Result[p * 3 + 2] += RawResult[(p * NTilts + t) * 3 + 2];
                        }

                    Result = Helper.ArrayOfFunction(p => Result[p * 3 + 0] / Math.Max(1e-10, Math.Sqrt(Result[p * 3 + 1] * Result[p * 3 + 2])) * 100 * NTilts, NParticlesOverall);

                    return Result;
                };

                Func<Species, double[]> GetPerParticleCCSpecies = (species) =>
                {
                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;

                    double[] Result = new double[NParticles * 3];
                    float[] RawResult = GetRawCCSpecies(new float2(0), species);

                    for (int p = 0; p < NParticles; p++)
                        for (int t = 0; t < NTilts; t++)
                        {
                            Result[p * 3 + 0] += RawResult[(p * NTilts + t) * 3 + 0];
                            Result[p * 3 + 1] += RawResult[(p * NTilts + t) * 3 + 1];
                            Result[p * 3 + 2] += RawResult[(p * NTilts + t) * 3 + 2];
                        }

                    Result = Helper.ArrayOfFunction(p => Result[p * 3 + 0] /
                                                         Math.Max(1e-10, Math.Sqrt(Result[p * 3 + 1] * Result[p * 3 + 2])) *
                                                         100 * NTilts, NParticles);

                    return Result;
                };

                Func<(float[] xp, float[] xm, float[] yp, float[] ym, float delta2)> GetRawShiftGradients = () =>
                {
                    float Delta = 0.1f;
                    float Delta2 = Delta * 2;

                    float[] h_ScoresXP = GetRawCC(float2.UnitX * Delta);
                    float[] h_ScoresXM = GetRawCC(-float2.UnitX * Delta);
                    float[] h_ScoresYP = GetRawCC(float2.UnitY * Delta);
                    float[] h_ScoresYM = GetRawCC(-float2.UnitY * Delta);

                    //for (int i = 0; i < Result.Length; i++)
                    //    Result[i] = new float2((h_ScoresXP[i] - h_ScoresXM[i]) / Delta2 * 100,
                    //                           (h_ScoresYP[i] - h_ScoresYM[i]) / Delta2 * 100);

                    return (h_ScoresXP, h_ScoresXM, h_ScoresYP, h_ScoresYM, Delta2);
                };

                #endregion

                #region BFGS prerequisites

                float2[] OriginalImageWarps = Helper.ArrayOfFunction(t => new float2(GridMovementX.Values[t], GridMovementY.Values[t]), GridMovementX.Values.Length);
                float3[] OriginalVolumeWarps = Helper.ArrayOfFunction(t => new float3(GridVolumeWarpX.Values[t], GridVolumeWarpY.Values[t], GridVolumeWarpZ.Values[t]), GridVolumeWarpX.Values.Length);

                float[] OriginalAngleX = GridAngleX.Values.ToArray();
                float[] OriginalAngleY = GridAngleY.Values.ToArray();
                float[] OriginalAngleZ = GridAngleZ.Values.ToArray();

                float4[] OriginalTiltCTFs = Helper.ArrayOfFunction(t => new float4(GridCTFDefocus.Values[t],
                                                                                   GridCTFDefocusDelta.Values[t],
                                                                                   GridCTFDefocusAngle.Values[t],
                                                                                   GridCTFPhase.Values[t]), NTilts);

                float[] OriginalParamsCTF =
                {
                    (float)CTF.Cs,
                };

                CTFOptimizationTypes[] CTFStepTypes =
                {
                    CTFOptimizationTypes.Defocus,
                    CTFOptimizationTypes.AstigmatismDelta,
                    CTFOptimizationTypes.AstigmatismAngle,
                    CTFOptimizationTypes.PhaseShift,
                    CTFOptimizationTypes.Zernike2,
                    CTFOptimizationTypes.Zernike2,
                    CTFOptimizationTypes.Zernike2,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.Cs,
                };

                float[] OriginalZernikeOdd = CTF.ZernikeCoeffsOdd.ToList().ToArray();
                float[] OriginalZernikeEven = CTF.ZernikeCoeffsEven.ToList().ToArray();

                //float2 OriginalBeamTilt = CTF.BeamTilt;
                Matrix2 OriginalMagnification = MagnificationCorrection.GetCopy();

                float3[][] OriginalParticlePositions = allSpecies.Select(s => Helper.Combine(SpeciesParticles[s].Select(p => p.Coordinates))).ToArray();
                float3[][] OriginalParticleAngles = allSpecies.Select(s => Helper.Combine(SpeciesParticles[s].Select(p => p.Angles))).ToArray();

                int BFGSIterations = 0;
                WarpOptimizationTypes CurrentOptimizationTypeWarp = 0;
                CTFOptimizationTypes CurrentOptimizationTypeCTF = 0;

                double[] InitialParametersWarp = new double[GridMovementX.Values.Length * 2 +
                                                            GridVolumeWarpX.Values.Length * 3 +
                                                            GridAngleX.Values.Length * 3 +
                                                            OriginalParticlePositions.Select(a => a.Length).Sum() * 3 +
                                                            OriginalParticleAngles.Select(a => a.Length).Sum() * 3 +
                                                            CTF.ZernikeCoeffsOdd.Length +
                                                            4];
                double[] InitialParametersDefocus = new double[NTilts * 4 +
                                                               CTF.ZernikeCoeffsEven.Length +
                                                               OriginalParamsCTF.Length];

                #endregion

                #region Set parameters from vector

                Action<double[], TiltSeries, bool> SetWarpFromVector = (input, series, setParticles) =>
                {
                    int Offset = 0;

                    float[] MovementXData = new float[GridMovementX.Values.Length];
                    float[] MovementYData = new float[GridMovementX.Values.Length];
                    for (int i = 0; i < MovementXData.Length; i++)
                    {
                        MovementXData[i] = OriginalImageWarps[i].X + (float)input[Offset + i];
                        MovementYData[i] = OriginalImageWarps[i].Y + (float)input[Offset + MovementXData.Length + i];
                    }
                    series.GridMovementX = new CubicGrid(GridMovementX.Dimensions, MovementXData);
                    series.GridMovementY = new CubicGrid(GridMovementY.Dimensions, MovementYData);

                    Offset += MovementXData.Length * 2;

                    float[] VolumeXData = new float[GridVolumeWarpX.Values.Length];
                    float[] VolumeYData = new float[GridVolumeWarpX.Values.Length];
                    float[] VolumeZData = new float[GridVolumeWarpX.Values.Length];
                    int GridVolumeSlice = (int)GridVolumeWarpX.Dimensions.ElementsSlice();
                    for (int i = 0; i < VolumeXData.Length; i++)
                    {
                        if (i < GridVolumeSlice)
                        {
                            VolumeXData[i] = OriginalVolumeWarps[i].X;
                            VolumeYData[i] = OriginalVolumeWarps[i].Y;
                            VolumeZData[i] = OriginalVolumeWarps[i].Z;
                        }
                        else
                        {
                            VolumeXData[i] = OriginalVolumeWarps[i].X + (float)input[Offset + i];
                            VolumeYData[i] = OriginalVolumeWarps[i].Y + (float)input[Offset + VolumeXData.Length + i];
                            VolumeZData[i] = OriginalVolumeWarps[i].Z + (float)input[Offset + VolumeXData.Length + VolumeYData.Length + i];
                        }
                    }
                    series.GridVolumeWarpX = new LinearGrid4D(GridVolumeWarpX.Dimensions, VolumeXData);
                    series.GridVolumeWarpY = new LinearGrid4D(GridVolumeWarpY.Dimensions, VolumeYData);
                    series.GridVolumeWarpZ = new LinearGrid4D(GridVolumeWarpZ.Dimensions, VolumeZData);

                    Offset += VolumeXData.Length * 3;

                    float[] AngleXData = new float[GridAngleX.Values.Length];
                    float[] AngleYData = new float[GridAngleY.Values.Length];
                    float[] AngleZData = new float[GridAngleZ.Values.Length];
                    for (int i = 0; i < AngleXData.Length; i++)
                    {
                        AngleXData[i] = OriginalAngleX[i] + (float)input[Offset + i];
                        AngleYData[i] = OriginalAngleY[i] + (float)input[Offset + AngleXData.Length + i];
                        AngleZData[i] = OriginalAngleZ[i] + (float)input[Offset + AngleXData.Length * 2 + i];
                    }
                    series.GridAngleX = new CubicGrid(GridAngleX.Dimensions, AngleXData);
                    series.GridAngleY = new CubicGrid(GridAngleY.Dimensions, AngleYData);
                    series.GridAngleZ = new CubicGrid(GridAngleZ.Dimensions, AngleZData);

                    Offset += AngleXData.Length * 3;

                    if (setParticles)
                    {
                        for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                        {
                            Particle[] Particles = SpeciesParticles[allSpecies[ispecies]];

                            int ResCoords = allSpecies[ispecies].TemporalResolutionMovement;

                            for (int p = 0; p < Particles.Length; p++)
                            {
                                for (int ic = 0; ic < ResCoords; ic++)
                                {
                                    Particles[p].Coordinates[ic] = OriginalParticlePositions[ispecies][p * ResCoords + ic] + new float3((float)input[Offset + (p * 6 + 0) * ResCoords + ic],
                                                                                                                                        (float)input[Offset + (p * 6 + 1) * ResCoords + ic],
                                                                                                                                        (float)input[Offset + (p * 6 + 2) * ResCoords + ic]);
                                    Particles[p].Angles[ic] = OriginalParticleAngles[ispecies][p * ResCoords + ic] + new float3((float)input[Offset + (p * 6 + 3) * ResCoords + ic],
                                                                                                                                (float)input[Offset + (p * 6 + 4) * ResCoords + ic],
                                                                                                                                (float)input[Offset + (p * 6 + 5) * ResCoords + ic]);
                                }
                            }

                            Offset += OriginalParticlePositions[ispecies].Length * 6;
                        }
                    }
                    else
                    {
                        Offset += OriginalParticlePositions.Select(a => a.Length).Sum() * 6;
                    }

                    //CTF.BeamTilt = OriginalBeamTilt + new float2((float)input[input.Length - 5],
                    //                                             (float)input[input.Length - 4]);

                    for (int icoeff = 0; icoeff < CTF.ZernikeCoeffsOdd.Length; icoeff++)
                        CTF.ZernikeCoeffsOdd[icoeff] = OriginalZernikeOdd[icoeff] + (float)input[Offset + icoeff];

                    Offset += CTF.ZernikeCoeffsOdd.Length;

                    MagnificationCorrection = OriginalMagnification + new Matrix2((float)input[input.Length - 4] / 100,
                                                                                  (float)input[input.Length - 3] / 100,
                                                                                  (float)input[input.Length - 2] / 100,
                                                                                  (float)input[input.Length - 1] / 100);

                    // MagnificationCorrection follows a different, weird convention.
                    // .x and .y define the X and Y axes of a scaling matrix, rotated by -.z
                    // Scaling .x up means the pixel size along that axis is smaller, thus a negative DeltaPercent
                    //CTF.PixelSizeDeltaPercent = -(decimal)(MagnificationCorrection.X - (MagnificationCorrection.X + MagnificationCorrection.Y) / 2);
                    //CTF.PixelSizeAngle = (decimal)(-MagnificationCorrection.Z * Helper.ToDeg);
                };

                Action<double[]> SetDefocusFromVector = input =>
                {
                    int Offset = 0;

                    float[] DefocusValues = Helper.ArrayOfFunction(t => OriginalTiltCTFs[t].X + (float)input[t * 4 + 0] * 0.1f, NTilts);
                    float[] AstigmatismValues = Helper.ArrayOfFunction(t => OriginalTiltCTFs[t].Y + (float)input[t * 4 + 1] * 0.1f, NTilts);
                    float[] AngleValues = Helper.ArrayOfFunction(t => OriginalTiltCTFs[t].Z + (float)input[t * 4 + 2] * 36, NTilts);
                    float[] PhaseValues = Helper.ArrayOfFunction(t => OriginalTiltCTFs[t].W + (float)input[t * 4 + 3] * 36, NTilts);

                    GridCTFDefocus = new CubicGrid(new int3(1, 1, NTilts), DefocusValues);
                    GridCTFDefocusDelta = new CubicGrid(new int3(1, 1, NTilts), AstigmatismValues);
                    GridCTFDefocusAngle = new CubicGrid(new int3(1, 1, NTilts), AngleValues);
                    GridCTFPhase = new CubicGrid(new int3(1, 1, NTilts), PhaseValues);

                    Offset += NTilts * 4;

                    {
                        float[] ValuesZernike = new float[CTF.ZernikeCoeffsEven.Length];
                        for (int i = 0; i < ValuesZernike.Length; i++)
                            ValuesZernike[i] = OriginalZernikeEven[i] + (float)input[Offset + i];

                        CTF.ZernikeCoeffsEven = ValuesZernike;
                        Offset += CTF.ZernikeCoeffsEven.Length;
                    }

                    CTF.Cs = (decimal)(OriginalParamsCTF[0] + input[input.Length - 1]);
                    //CTF.PixelSizeDeltaPercent = (decimal)(OriginalParamsCTF[1] + input[input.Length - 2] * 0.1f);
                    //CTF.PixelSizeAngle = (decimal)(OriginalParamsCTF[2] + input[input.Length - 1] * 36);
                };

                #endregion

                #region Wiggle weights

                progressCallback("Precomputing gradient weights...");

                int NWiggleDifferentiable = GridMovementX.Values.Length +
                                            GridMovementY.Values.Length +
                                            GridVolumeWarpX.Values.Length +
                                            GridVolumeWarpY.Values.Length +
                                            GridVolumeWarpZ.Values.Length;
                (int[] indices, float2[] weights)[] AllWiggleWeights = new (int[] indices, float2[] weights)[NWiggleDifferentiable];

                if (optionsMPA.DoImageWarp || optionsMPA.DoVolumeWarp)
                {
                    TiltSeries[] ParallelSeriesCopies = Helper.ArrayOfFunction(i => new TiltSeries(this.Path), 16);

                    Dictionary<Species, float3[]> AllParticlePositions = new Dictionary<Species, float3[]>();
                    foreach (var species in allSpecies)
                    {
                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        int Offset = SpeciesParticleIDRanges[species].Start;

                        float3[] ParticlePositions = new float3[NParticles * NTilts];
                        for (int p = 0; p < NParticles; p++)
                        {
                            float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                            for (int t = 0; t < NTilts; t++)
                                ParticlePositions[p * NTilts + t] = Positions[t];
                        }

                        AllParticlePositions.Add(species, ParticlePositions);
                    }

                    Helper.ForCPU(0, NWiggleDifferentiable, ParallelSeriesCopies.Length, (threadID) =>
                    {
                        ParallelSeriesCopies[threadID].VolumeDimensionsPhysical = VolumeDimensionsPhysical;
                        ParallelSeriesCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                        ParallelSeriesCopies[threadID].SizeRoundingFactors = SizeRoundingFactors;
                    },
                    (iwiggle, threadID) =>
                    {
                        double[] WiggleParams = new double[InitialParametersWarp.Length];
                        WiggleParams[iwiggle] = 1;
                        SetWarpFromVector(WiggleParams, ParallelSeriesCopies[threadID], false);

                        float2[] RawShifts = new float2[NParticlesOverall * NTilts];
                        foreach (var species in allSpecies)
                        {
                            Particle[] Particles = SpeciesParticles[species];
                            int NParticles = Particles.Length;
                            if (NParticles == 0)
                                continue;

                            float[] ContainmentMask = SpeciesContainmentMasks[species];
                            int Offset = SpeciesParticleIDRanges[species].Start;

                            float3[] ParticlePositions = AllParticlePositions[species];

                            float3[] ParticlePositionsProjected = ParallelSeriesCopies[threadID].GetPositionInAllTilts(ParticlePositions);
                            float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                            for (int p = 0; p < NParticles; p++)
                                for (int t = 0; t < NTilts; t++)
                                    RawShifts[(Offset + p) * NTilts + t] = (new float2(ParticlePositionsProjected[p * NTilts + t]) - ParticleExtractedAt[p * NTilts + t]) * ContainmentMask[p * NTilts + t];
                        }

                        List<int> Indices = new List<int>();
                        List<float2> Weights = new List<float2>();
                        for (int i = 0; i < RawShifts.Length; i++)
                        {
                            if (RawShifts[i].LengthSq() > 1e-6f)
                            {
                                Indices.Add(i);
                                Weights.Add(RawShifts[i]);

                                if (Math.Abs(RawShifts[i].X) > 1.5f)
                                {
                                    throw new Exception("");
                                }
                            }
                        }

                        AllWiggleWeights[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                    }, null);
                }

                #endregion

                double[] OldInput = new double[0];
                double[] OldGradient = new double[0];

                #region Loss and gradient functions for warping

                Func<double[], double> WarpEval = input =>
                {
                    SetWarpFromVector(input, this, true);

                    float[] RawCC = GetRawCC(new float2(0));
                    double SumAB = 0, SumA2 = 0, SumB2 = 0;
                    for (int p = 0; p < NParticlesOverall; p++)
                    {
                        for (int t = 0; t < NTilts; t++)
                        {
                            SumAB += RawCC[(p * NTilts + t) * 3 + 0];
                            SumA2 += RawCC[(p * NTilts + t) * 3 + 1];
                            SumB2 += RawCC[(p * NTilts + t) * 3 + 2];
                        }
                    }

                    double Score = SumAB / Math.Max(1e-10, Math.Sqrt(SumA2 * SumB2)) * NParticlesOverall * NTilts * 100;

                    //double[] TiltScores = GetPerTiltDiff2();
                    //double Score = TiltScores.Sum();

                    Console.WriteLine(Score);

                    return Score;
                };

                Func<double[], double[]> WarpGrad = input =>
                {
                    double Delta = 0.025;
                    double Delta2 = Delta * 2;

                    double[] Result = new double[input.Length];

                    if (BFGSIterations-- <= 0)
                        return Result;

                    if (MathHelper.AllEqual(input, OldInput))
                        return OldGradient;

                    int Offset = 0;

                    if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ImageWarp) != 0 || // GridMovementXY
                        (CurrentOptimizationTypeWarp & WarpOptimizationTypes.VolumeWarp) != 0)  // GridVolumeWarpXYZ
                    {
                        SetWarpFromVector(input, this, true);
                        (var XP, var XM, var YP, var YM, var Delta2Movement) = GetRawShiftGradients();

                        int NImageWarp = GridMovementX.Values.Length * 2;

                        if (true)
                        Parallel.For(0, AllWiggleWeights.Length, iwiggle =>
                        {
                            if (iwiggle < NImageWarp && (CurrentOptimizationTypeWarp & WarpOptimizationTypes.ImageWarp) == 0)
                                return;
                            if (iwiggle >= NImageWarp && (CurrentOptimizationTypeWarp & WarpOptimizationTypes.VolumeWarp) == 0)
                                return;

                            double SumGrad = 0;
                            double SumWeights = 0;
                            double SumWeightsGrad = 0;

                            int[] Indices = AllWiggleWeights[iwiggle].indices;
                            float2[] Weights = AllWiggleWeights[iwiggle].weights;

                            for (int i = 0; i < Indices.Length; i++)
                            {
                                int id = Indices[i];

                                SumWeights += Math.Abs(Weights[i].X) * Math.Sqrt(XP[id * 3 + 1] + XM[id * 3 + 1]) +
                                              Math.Abs(Weights[i].Y) * Math.Sqrt(YP[id * 3 + 1] + YM[id * 3 + 1]);
                                SumWeightsGrad += Math.Abs(Weights[i].X) + Math.Abs(Weights[i].Y);

                                double GradX = (XP[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(XP[id * 3 + 1] * XP[id * 3 + 2])) -
                                                XM[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(XM[id * 3 + 1] * XM[id * 3 + 2]))) / Delta2Movement;
                                double GradY = (YP[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(YP[id * 3 + 1] * YP[id * 3 + 2])) -
                                                YM[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(YM[id * 3 + 1] * YM[id * 3 + 2]))) / Delta2Movement;

                                SumGrad += Weights[i].X * Math.Sqrt(XP[id * 3 + 1] + XM[id * 3 + 1]) * GradX;
                                SumGrad += Weights[i].Y * Math.Sqrt(YP[id * 3 + 1] + YM[id * 3 + 1]) * GradY;
                            }

                            Result[Offset + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                        });

                        if (false)
                        for (int i = 0; i < NImageWarp; i++)
                        {
                            double[] InputPlus = input.ToArray();
                            InputPlus[Offset + i] += Delta;

                            double ScorePlus = WarpEval(InputPlus);

                            double[] InputMinus = input.ToArray();
                            InputMinus[Offset + i] -= Delta;

                            double ScoreMinus = WarpEval(InputMinus);

                            Result[Offset + i] = (ScorePlus - ScoreMinus) / Delta2;
                        }
                    }
                    Offset += AllWiggleWeights.Length;


                    if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.AxisAngle) != 0) // GridAngleX, Y and Z
                    {
                        int SliceElements = (int)GridAngleX.Dimensions.ElementsSlice();

                        for (int a = 0; a < 3; a++)
                        {
                            for (int i = 0; i < SliceElements; i++)
                            {
                                double[] InputPlus = input.ToArray();
                                for (int t = 0; t < NTilts; t++)
                                    InputPlus[Offset + t * SliceElements + i] += Delta;

                                SetWarpFromVector(InputPlus, this, true);
                                double[] ScoresPlus = GetPerTiltCC();

                                double[] InputMinus = input.ToArray();
                                for (int t = 0; t < NTilts; t++)
                                    InputMinus[Offset + t * SliceElements + i] -= Delta;

                                SetWarpFromVector(InputMinus, this, true);
                                double[] ScoresMinus = GetPerTiltCC();

                                for (int t = 0; t < NTilts; t++)
                                    Result[Offset + t * SliceElements + i] = (ScoresPlus[t] - ScoresMinus[t]) / Delta2;
                            }

                            Offset += GridAngleX.Values.Length;
                        }
                    }
                    else
                    {
                        Offset += GridAngleX.Values.Length * 3;
                    }


                    {
                        for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                        {
                            Species Species = allSpecies[ispecies];
                            Particle[] Particles = SpeciesParticles[Species];

                            int TemporalRes = allSpecies[ispecies].TemporalResolutionMovement;

                            if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ParticlePosition) != 0)
                                for (int iparam = 0; iparam < 3 * TemporalRes; iparam++)
                                {
                                    double[] InputPlus = input.ToArray();
                                    for (int p = 0; p < Particles.Length; p++)
                                        InputPlus[Offset + p * 6 * TemporalRes + iparam] += Delta;

                                    SetWarpFromVector(InputPlus, this, true);
                                    double[] ScoresPlus = GetPerParticleCCSpecies(Species);

                                    double[] InputMinus = input.ToArray();
                                    for (int p = 0; p < Particles.Length; p++)
                                        InputMinus[Offset + p * 6 * TemporalRes + iparam] -= Delta;

                                    SetWarpFromVector(InputMinus, this, true);
                                    double[] ScoresMinus = GetPerParticleCCSpecies(Species);

                                    for (int p = 0; p < Particles.Length; p++)
                                        Result[Offset + p * 6 * TemporalRes + iparam] = (ScoresPlus[p] - ScoresMinus[p]) / Delta2;
                                }

                            if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ParticleAngle) != 0)
                                for (int iparam = 3 * TemporalRes; iparam < 6 * TemporalRes; iparam++)
                                {
                                    double[] InputPlus = input.ToArray();
                                    for (int p = 0; p < Particles.Length; p++)
                                        InputPlus[Offset + p * 6 * TemporalRes + iparam] += Delta;

                                    SetWarpFromVector(InputPlus, this, true);
                                    double[] ScoresPlus = GetPerParticleCCSpecies(Species);

                                    double[] InputMinus = input.ToArray();
                                    for (int p = 0; p < Particles.Length; p++)
                                        InputMinus[Offset + p * 6 * TemporalRes + iparam] -= Delta;

                                    SetWarpFromVector(InputMinus, this, true);
                                    double[] ScoresMinus = GetPerParticleCCSpecies(Species);

                                    for (int p = 0; p < Particles.Length; p++)
                                        Result[Offset + p * 6 * TemporalRes + iparam] = (ScoresPlus[p] - ScoresMinus[p]) / Delta2;
                                }

                            Offset += OriginalParticlePositions[ispecies].Length * 6; // No * TemporalRes because it's already included in OriginalParticlePositions
                        }
                    }

                    if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.Zernike13) != 0)
                    {
                        for (int iparam = 0; iparam < Math.Min(6, CTF.ZernikeCoeffsOdd.Length); iparam++)
                        {
                            double[] InputPlus = input.ToArray();
                            InputPlus[Offset + iparam] += Delta;

                            //SetWarpFromVector(InputPlus, this, true);
                            double ScoresPlus = WarpEval(InputPlus);

                            double[] InputMinus = input.ToArray();
                            InputMinus[Offset + iparam] -= Delta;

                            //SetWarpFromVector(InputMinus, this, true);
                            double ScoresMinus = WarpEval(InputMinus);

                            Result[Offset + iparam] = (ScoresPlus - ScoresMinus) / Delta2;
                        }
                    }

                    if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.Zernike5) != 0)
                    {
                        for (int iparam = Math.Min(6, CTF.ZernikeCoeffsOdd.Length); iparam < Math.Min(12, CTF.ZernikeCoeffsOdd.Length); iparam++)
                        {
                            double[] InputPlus = input.ToArray();
                            InputPlus[Offset + iparam] += Delta;

                            //SetWarpFromVector(InputPlus, this, true);
                            double ScoresPlus = WarpEval(InputPlus);

                            double[] InputMinus = input.ToArray();
                            InputMinus[Offset + iparam] -= Delta;

                            //SetWarpFromVector(InputMinus, this, true);
                            double ScoresMinus = WarpEval(InputMinus);

                            Result[Offset + iparam] = (ScoresPlus - ScoresMinus) / Delta2;
                        }
                    }

                    Offset += CTF.ZernikeCoeffsOdd.Length;

                    if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.Magnification) != 0)
                    {
                        for (int iparam = 0; iparam < 4; iparam++)
                        {
                            double[] InputPlus = input.ToArray();
                            InputPlus[input.Length - 4 + iparam] += Delta;

                            //SetWarpFromVector(InputPlus, this, true);
                            double ScoresPlus = WarpEval(InputPlus);

                            double[] InputMinus = input.ToArray();
                            InputMinus[input.Length - 4 + iparam] -= Delta;

                            //SetWarpFromVector(InputMinus, this, true);
                            double ScoresMinus = WarpEval(InputMinus);

                            Result[input.Length - 4 + iparam] = (ScoresPlus - ScoresMinus) / Delta2;
                        }
                    }

                    OldInput = input.ToList().ToArray();
                    OldGradient = Result.ToList().ToArray();

                    return Result;
                };

                #endregion

                #region Loss and gradient functions for defocus

                Func<double[], double> DefocusEval = input =>
                {
                    SetDefocusFromVector(input);

                    double ScoreAB = 0, ScoreA2 = 0, ScoreB2 = 0;

                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                    int BatchSize = optionsMPA.BatchSize;
                    float[] ResultP = new float[BatchSize * 3];
                    float[] ResultQ = new float[BatchSize * 3];

                    foreach (var species in allSpecies)
                    {
                        if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                            continue;

                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                        int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];

                        float AngPixRefine = species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                        int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];
                        float[] ContainmentMask = SpeciesContainmentMasks[species];

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);   // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine

                        Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                        int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                        int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                        if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                        {
                            Console.WriteLine($"SizeFullSuper = {SizeFullSuper}, BatchSize = {BatchSize}, free memory = {GPU.GetFreeMemory(GPUID)}");
                            throw new Exception("No FFT plans created!");
                        }

                        Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                        Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NTilts), true, true);
                        for (int t = 0; t < NTilts; t++)
                            GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                       PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                                       PhaseCorrection.Dims.Slice(),
                                       new int3(RelevantSizes[t]).Slice(),
                                       1);

                        Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                        bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                        float[][] EwaldResults = { ResultP, ResultQ };

                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float[] BatchContainmentMask = ContainmentMask.Skip(batchStart * NTilts).Take(CurBatch * NTilts).ToArray();
                            float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                            float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                            for (int t = 0; t < NTilts; t++)
                            {
                                float3[] CoordinatesTilt = new float3[CurBatch];
                                float3[] AnglesTilt = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                                    AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                                }

                                float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                                float[] Defoci = new float[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p] = ImageCoords[p].Z;
                                }

                                for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                {
                                    GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                            Extracted.GetDevice(Intent.Write),
                                            TiltData[t].Dims.Slice(),
                                            new int3(SizeFullSuper, SizeFullSuper, 1),
                                            Helper.ToInterleaved(ExtractOrigins),
                                            true,
                                            (uint)CurBatch);

                                    GPU.FFT(Extracted.GetDevice(Intent.Read),
                                            ExtractedFT.GetDevice(Intent.Write),
                                            new int3(SizeFullSuper, SizeFullSuper, 1),
                                            (uint)CurBatch,
                                            PlanForwSuper);

                                    ExtractedFT.ShiftSlices(ResidualShifts);
                                    ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                    GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                               Extracted.GetDevice(Intent.Write),
                                               new int3(SizeFullSuper, SizeFullSuper, 1),
                                               new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                               (uint)CurBatch);

                                    if (species.DoEwald)
                                    {
                                        GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, EwaldReverse[iewald], ExtractedCTF, true);

                                        GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                           ExtractedCTF.GetDevice(Intent.Read),
                                                                           ExtractedFT.GetDevice(Intent.Write),
                                                                           ExtractedCTF.ElementsComplex,
                                                                           1);
                                    }
                                    else
                                    {
                                        GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                        GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                                          ExtractedCTF.GetDevice(Intent.Read),
                                                                          ExtractedFT.GetDevice(Intent.Write),
                                                                          ExtractedCTF.ElementsComplex,
                                                                          1);
                                    }

                                    GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                             Extracted.GetDevice(Intent.Write),
                                             new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                             (uint)CurBatch,
                                             PlanBackSuper,
                                             false);

                                    GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                                    ExtractedCropped.GetDevice(Intent.Write),
                                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                    new int3(SizeRefine, SizeRefine, 1),
                                                    (uint)CurBatch);

                                    GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                                   ExtractedCropped.GetDevice(Intent.Write),
                                                   ExtractedCropped.Dims.Slice(),
                                                   ParticleDiameterPix / 2f,
                                                   16 * AngPixExtract / AngPixRefine,
                                                   true,
                                                   (uint)CurBatch);

                                    GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                            ExtractedCroppedFT.GetDevice(Intent.Write),
                                            new int3(SizeRefine, SizeRefine, 1),
                                            (uint)CurBatch,
                                            PlanForw);

                                    ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                    GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                               ExtractedCropped.GetDevice(Intent.Write),
                                               new int3(SizeRefine).Slice(),
                                               new int3(RelevantSizes[t]).Slice(),
                                               (uint)CurBatch);


                                    GPU.MultiParticleDiff(EwaldResults[iewald],
                                                          new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                                          SizeRefine,
                                                          new[] { RelevantSizes[t] },
                                                          new float[CurBatch * 2],
                                                          Helper.ToInterleaved(ImageAngles),
                                                          MagnificationCorrection.ToVec(),
                                                          SpeciesCTFWeights[species].GetDeviceSlice(t, Intent.Read),
                                                          PhaseCorrectionAll.GetDeviceSlice(t, Intent.Read),
                                                          species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)species.PixelSize) * (iewald == 0 ? 1 : -1) : 0,
                                                          species.CurrentMaxShellRefinement,
                                                          new[] { species.HalfMap1Projector[GPUID].t_DataRe, species.HalfMap2Projector[GPUID].t_DataRe },
                                                          new[] { species.HalfMap1Projector[GPUID].t_DataIm, species.HalfMap2Projector[GPUID].t_DataIm },
                                                          species.HalfMap1Projector[GPUID].Oversampling,
                                                          species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                                          new IntPtr((long)SpeciesParticleSubsets[species] + batchStart * sizeof(int)),
                                                          CurBatch,
                                                          1);
                                }

                                for (int i = 0; i < CurBatch; i++)
                                {
                                    ScoreAB += (ResultP[i * 3 + 0] + ResultQ[i * 3 + 0]) * BatchContainmentMask[i * NTilts + t];
                                    ScoreA2 += (ResultP[i * 3 + 1] + ResultQ[i * 3 + 1]) * BatchContainmentMask[i * NTilts + t];
                                    ScoreB2 += (ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]) * BatchContainmentMask[i * NTilts + t];
                                }
                            }
                        }

                        PhaseCorrectionAll.Dispose();
                        PhaseCorrection.Dispose();
                        GammaCorrection.Dispose();

                        CoordsCTF.Dispose();
                        Extracted.Dispose();
                        ExtractedFT.Dispose();
                        ExtractedCropped.Dispose();
                        ExtractedCroppedFT.Dispose();
                        ExtractedCTF.Dispose();

                        GPU.DestroyFFTPlan(PlanForwSuper);
                        GPU.DestroyFFTPlan(PlanBackSuper);
                        GPU.DestroyFFTPlan(PlanForw);
                    }

                    //foreach (var image in TiltData)
                    //    image.FreeDevice();

                    double Score = ScoreAB / Math.Max(1e-10, Math.Sqrt(ScoreA2 * ScoreB2)) * NParticlesOverall * NTilts;
                    Score *= 100;

                    Console.WriteLine(Score);

                    return Score;
                };

                Func<double[], double[]> DefocusGrad = input =>
                {
                    double Delta = 0.001;
                    double Delta2 = Delta * 2;

                    double[] Deltas = { Delta, -Delta };

                    double[] Result = new double[input.Length];
                    double[] ScoresAB = new double[input.Length * 2];
                    double[] ScoresA2 = new double[input.Length * 2];
                    double[] ScoresB2 = new double[input.Length * 2];
                    int[] ScoresSamples = new int[input.Length * 2];

                    if (BFGSIterations-- <= 0)
                        return Result;

                    if (MathHelper.AllEqual(input, OldInput))
                        return OldGradient;

                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                    int BatchSize = 64;
                    float[] ResultP = new float[BatchSize * 3];
                    float[] ResultQ = new float[BatchSize * 3];

                    foreach (var species in allSpecies)
                    {
                        if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                            continue;

                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                        int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                        float AngPixRefine = species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                        int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                        float[] ContainmentMask = SpeciesContainmentMasks[species];

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);

                        Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedRefineSuper = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);
                        Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                        int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                        int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                        if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                        {
                            Console.WriteLine($"SizeFullSuper = {SizeFullSuper}, BatchSize = {BatchSize}, free memory = {GPU.GetFreeMemory(GPUID)}");
                            throw new Exception("No FFT plans created!");
                        }

                        Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                        Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NTilts), true, true);
                        for (int t = 0; t < NTilts; t++)
                            GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                        PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                                        PhaseCorrection.Dims.Slice(),
                                        new int3(RelevantSizes[t]).Slice(),
                                        1);

                        bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                        float[][] EwaldResults = { ResultP, ResultQ };

                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float[] BatchContainmentMask = ContainmentMask.Skip(batchStart * NTilts).Take(CurBatch * NTilts).ToArray();
                            float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                            float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                            for (int t = 0; t < NTilts; t++)
                            {
                                float3[] CoordinatesTilt = new float3[CurBatch];
                                float3[] AnglesTilt = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                                    AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                                }

                                float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                                float[] Defoci = new float[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p] = ImageCoords[p].Z;
                                }

                                GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                            Extracted.GetDevice(Intent.Write),
                                            TiltData[t].Dims.Slice(),
                                            new int3(SizeFullSuper, SizeFullSuper, 1),
                                            Helper.ToInterleaved(ExtractOrigins),
                                            true,
                                            (uint)CurBatch);

                                GPU.FFT(Extracted.GetDevice(Intent.Read),
                                        ExtractedFT.GetDevice(Intent.Write),
                                        new int3(SizeFullSuper, SizeFullSuper, 1),
                                        (uint)CurBatch,
                                        PlanForwSuper);

                                ExtractedFT.ShiftSlices(ResidualShifts);
                                ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                           Extracted.GetDevice(Intent.Write),
                                           new int3(SizeFullSuper, SizeFullSuper, 1),
                                           new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                           (uint)CurBatch);

                                for (int iparam = 0; iparam < CTFStepTypes.Length; iparam++)
                                {
                                    if ((CurrentOptimizationTypeCTF & CTFStepTypes[iparam]) == 0)
                                        continue;

                                    for (int idelta = 0; idelta < 2; idelta++)
                                    {
                                        double[] InputAltered = input.ToArray();
                                        if (iparam < 4)
                                            InputAltered[t * 4 + iparam] += Deltas[idelta];
                                        else
                                            InputAltered[input.Length - CTFStepTypes.Length + iparam] += Deltas[idelta];

                                        SetDefocusFromVector(InputAltered);

                                        ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                        for (int p = 0; p < CurBatch; p++)
                                            Defoci[p] = ImageCoords[p].Z;


                                        Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                                        for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                        {
                                            if (species.DoEwald)
                                            {
                                                GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, EwaldReverse[iewald], ExtractedCTF, true);

                                                GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                                  ExtractedCTF.GetDevice(Intent.Read),
                                                                                  ExtractedFT.GetDevice(Intent.Write),
                                                                                  ExtractedCTF.ElementsComplex,
                                                                                  1);
                                            }
                                            else
                                            {
                                                GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                                GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                                                  ExtractedCTF.GetDevice(Intent.Read),
                                                                                  ExtractedFT.GetDevice(Intent.Write),
                                                                                  ExtractedCTF.ElementsComplex,
                                                                                  1);
                                            }

                                            GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                                     ExtractedRefineSuper.GetDevice(Intent.Write),
                                                     new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                     (uint)CurBatch,
                                                     PlanBackSuper,
                                                     false);

                                            GPU.CropFTFull(ExtractedRefineSuper.GetDevice(Intent.Read),
                                                            ExtractedCropped.GetDevice(Intent.Write),
                                                            new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                            new int3(SizeRefine, SizeRefine, 1),
                                                            (uint)CurBatch);

                                            GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                                           ExtractedCropped.GetDevice(Intent.Write),
                                                           ExtractedCropped.Dims.Slice(),
                                                           ParticleDiameterPix / 2f,
                                                           16 * AngPixExtract / AngPixRefine,
                                                           true,
                                                           (uint)CurBatch);

                                            GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                                    ExtractedCroppedFT.GetDevice(Intent.Write),
                                                    new int3(SizeRefine, SizeRefine, 1),
                                                    (uint)CurBatch,
                                                    PlanForw);

                                            ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                            GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                                       ExtractedCropped.GetDevice(Intent.Write),
                                                       new int3(SizeRefine).Slice(),
                                                       new int3(RelevantSizes[t]).Slice(),
                                                       (uint)CurBatch);


                                            GPU.MultiParticleDiff(EwaldResults[iewald],
                                                                  new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                                                  SizeRefine,
                                                                  new[] { RelevantSizes[t] },
                                                                  new float[CurBatch * 2],
                                                                  Helper.ToInterleaved(ImageAngles),
                                                                  MagnificationCorrection.ToVec(),
                                                                  SpeciesCTFWeights[species].GetDeviceSlice(t, Intent.Read),
                                                                  PhaseCorrectionAll.GetDeviceSlice(t, Intent.Read),
                                                                  species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)species.PixelSize) * (iewald == 0 ? 1 : -1) : 0,
                                                                  species.CurrentMaxShellRefinement,
                                                                  new[] { species.HalfMap1Projector[GPUID].t_DataRe, species.HalfMap2Projector[GPUID].t_DataRe },
                                                                  new[] { species.HalfMap1Projector[GPUID].t_DataIm, species.HalfMap2Projector[GPUID].t_DataIm },
                                                                  species.HalfMap1Projector[GPUID].Oversampling,
                                                                  species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                                                  new IntPtr((long)SpeciesParticleSubsets[species] + batchStart * sizeof(int)),
                                                                  CurBatch,
                                                                  1);
                                        }

                                        GammaCorrection.Dispose();

                                        if (iparam < 4)
                                            for (int i = 0; i < CurBatch; i++)
                                            {
                                                ScoresAB[(t * 4 + iparam) * 2 + idelta] += (ResultP[i * 3 + 0] + ResultQ[i * 3 + 0]) * BatchContainmentMask[i * NTilts + t];
                                                ScoresA2[(t * 4 + iparam) * 2 + idelta] += (ResultP[i * 3 + 1] + ResultQ[i * 3 + 1]) * BatchContainmentMask[i * NTilts + t];
                                                ScoresB2[(t * 4 + iparam) * 2 + idelta] += (ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]) * BatchContainmentMask[i * NTilts + t];
                                                ScoresSamples[(t * 4 + iparam) * 2 + idelta]++;
                                            }
                                        else
                                            for (int i = 0; i < CurBatch; i++)
                                            {
                                                ScoresAB[(input.Length - CTFStepTypes.Length + iparam) * 2 + idelta] += (ResultP[i * 3 + 0] + ResultQ[i * 3 + 0]) * BatchContainmentMask[i * NTilts + t];
                                                ScoresA2[(input.Length - CTFStepTypes.Length + iparam) * 2 + idelta] += (ResultP[i * 3 + 1] + ResultQ[i * 3 + 1]) * BatchContainmentMask[i * NTilts + t];
                                                ScoresB2[(input.Length - CTFStepTypes.Length + iparam) * 2 + idelta] += (ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]) * BatchContainmentMask[i * NTilts + t];
                                                ScoresSamples[(input.Length - CTFStepTypes.Length + iparam) * 2 + idelta]++;
                                            }
                                    }
                                }
                            }
                        }

                        PhaseCorrectionAll.Dispose();
                        PhaseCorrection.Dispose();

                        CoordsCTF.Dispose();
                        Extracted.Dispose();
                        ExtractedFT.Dispose();
                        ExtractedRefineSuper.Dispose();
                        ExtractedCropped.Dispose();
                        ExtractedCroppedFT.Dispose();
                        ExtractedCTF.Dispose();

                        GPU.DestroyFFTPlan(PlanForwSuper);
                        GPU.DestroyFFTPlan(PlanBackSuper);
                        GPU.DestroyFFTPlan(PlanForw);
                    }

                    //foreach (var image in TiltData)
                    //    image.FreeDevice();

                    for (int i = 0; i < ScoresAB.Length; i++)
                        ScoresAB[i] = ScoresAB[i] / Math.Max(1e-10, Math.Sqrt(ScoresA2[i] * ScoresB2[i])) * ScoresSamples[i];

                    for (int i = 0; i < Result.Length; i++)
                        Result[i] = (ScoresAB[i * 2 + 0] - ScoresAB[i * 2 + 1]) / Delta2 * 100;

                    OldInput = input.ToList().ToArray();
                    OldGradient = Result.ToList().ToArray();

                    return Result;
                };

                #endregion

                #region Grid search for per-tilt defoci

                Func<double[], double[]> DefocusGridSearch = input =>
                {
                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                    int BatchSize = optionsMPA.BatchSize;
                    float[] ResultP = new float[BatchSize * 3];
                    float[] ResultQ = new float[BatchSize * 3];

                    List<float4>[] AllSearchValues = Helper.ArrayOfFunction(i => new List<float4>(), NTilts);
                    List<float4>[] CurrentSearchValues = Helper.ArrayOfFunction(i => new List<float4>(), NTilts);
                    decimal GridSearchDelta = 0.3M;
                    foreach (var list in CurrentSearchValues)
                    {
                        for (decimal d = -3M; d <= 3M; d += GridSearchDelta)
                            list.Add(new float4((float)d, 0, 0, 0));
                    }

                    for (int irefine = 0; irefine < 4; irefine++)
                    {
                        foreach (var species in allSpecies)
                        {
                            if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                                continue;

                            Particle[] Particles = SpeciesParticles[species];
                            int NParticles = Particles.Length;
                            if (NParticles == 0)
                                continue;

                            int SpeciesOffset = SpeciesParticleIDRanges[species].Start;

                            int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                            int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                            int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                            int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                            float AngPixRefine = species.ResolutionRefinement / 2;
                            int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                            int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                            float[] ContainmentMask = SpeciesContainmentMasks[species];

                            Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);

                            Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                            Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                            Image ExtractedRefineSuper = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);
                            Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                            Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                            Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                            Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                            Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NTilts), true, true);
                            for (int t = 0; t < NTilts; t++)
                                GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                            PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                                            PhaseCorrection.Dims.Slice(),
                                            new int3(RelevantSizes[t]).Slice(),
                                            1);

                            Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                            bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                            float[][] EwaldResults = { ResultP, ResultQ };

                            int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                            int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                            int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                            if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                            {
                                Console.WriteLine($"SizeFullSuper = {SizeFullSuper}, BatchSize = {BatchSize}, free memory = {GPU.GetFreeMemory(GPUID)}");
                                throw new Exception("No FFT plans created!");
                            }
                            GPU.CheckGPUExceptions();

                            for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                            {
                                int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                                IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                                float[] BatchContainmentMask = ContainmentMask.Skip(batchStart * NTilts).Take(CurBatch * NTilts).ToArray();
                                float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                                float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                                for (int t = 0; t < NTilts; t++)
                                {
                                    float3[] CoordinatesTilt = new float3[CurBatch];
                                    float3[] AnglesTilt = new float3[CurBatch];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                                        AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                                    }

                                    float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                    float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                                    float[] Defoci = new float[CurBatch];
                                    int3[] ExtractOrigins = new int3[CurBatch];
                                    float3[] ResidualShifts = new float3[BatchSize];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                        ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                        ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                        Defoci[p] = ImageCoords[p].Z;
                                    }

                                    GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                                Extracted.GetDevice(Intent.Write),
                                                TiltData[t].Dims.Slice(),
                                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                                Helper.ToInterleaved(ExtractOrigins),
                                                true,
                                                (uint)CurBatch);

                                    GPU.FFT(Extracted.GetDevice(Intent.Read),
                                            ExtractedFT.GetDevice(Intent.Write),
                                            new int3(SizeFullSuper, SizeFullSuper, 1),
                                            (uint)CurBatch,
                                            PlanForwSuper);

                                    ExtractedFT.ShiftSlices(ResidualShifts);
                                    ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                    GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                               Extracted.GetDevice(Intent.Write),
                                               new int3(SizeFullSuper, SizeFullSuper, 1),
                                               new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                               (uint)CurBatch);

                                    for (int idelta = 0; idelta < CurrentSearchValues[t].Count; idelta++)
                                    {
                                        double[] InputAltered = input.ToArray();
                                        InputAltered[t * 4 + 0] += CurrentSearchValues[t][idelta].X;

                                        SetDefocusFromVector(InputAltered);

                                        ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                        for (int p = 0; p < CurBatch; p++)
                                            Defoci[p] = ImageCoords[p].Z;

                                        for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                        {
                                            if (species.DoEwald)
                                            {
                                                GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, EwaldReverse[iewald], ExtractedCTF, true);

                                                GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                                    ExtractedCTF.GetDevice(Intent.Read),
                                                                                    ExtractedFT.GetDevice(Intent.Write),
                                                                                    ExtractedCTF.ElementsComplex,
                                                                                    1);
                                            }
                                            else
                                            {
                                                GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                                GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                                                    ExtractedCTF.GetDevice(Intent.Read),
                                                                                    ExtractedFT.GetDevice(Intent.Write),
                                                                                    ExtractedCTF.ElementsComplex,
                                                                                    1);
                                            }

                                            GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                                        ExtractedRefineSuper.GetDevice(Intent.Write),
                                                        new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                        (uint)CurBatch,
                                                        PlanBackSuper,
                                                        false);

                                            GPU.CropFTFull(ExtractedRefineSuper.GetDevice(Intent.Read),
                                                            ExtractedCropped.GetDevice(Intent.Write),
                                                            new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                            new int3(SizeRefine, SizeRefine, 1),
                                                            (uint)CurBatch);

                                            GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                                            ExtractedCropped.GetDevice(Intent.Write),
                                                            ExtractedCropped.Dims.Slice(),
                                                            ParticleDiameterPix / 2f,
                                                            16 * AngPixExtract / AngPixRefine,
                                                            true,
                                                            (uint)CurBatch);

                                            GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                                    ExtractedCroppedFT.GetDevice(Intent.Write),
                                                    new int3(SizeRefine, SizeRefine, 1),
                                                    (uint)CurBatch,
                                                    PlanForw);

                                            ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                            GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                                       ExtractedCropped.GetDevice(Intent.Write),
                                                       new int3(SizeRefine).Slice(),
                                                       new int3(RelevantSizes[t]).Slice(),
                                                       (uint)CurBatch);


                                            GPU.MultiParticleDiff(EwaldResults[iewald],
                                                                    new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                                                    SizeRefine,
                                                                    new[] { RelevantSizes[t] },
                                                                    new float[CurBatch * 2],
                                                                    Helper.ToInterleaved(ImageAngles),
                                                                    MagnificationCorrection.ToVec(),
                                                                    SpeciesCTFWeights[species].GetDeviceSlice(t, Intent.Read),
                                                                    PhaseCorrectionAll.GetDeviceSlice(t, Intent.Read),
                                                                    species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)species.PixelSize) * (iewald == 0 ? 1 : -1) : 0,
                                                                    species.CurrentMaxShellRefinement,
                                                                    new[] { species.HalfMap1Projector[GPUID].t_DataRe, species.HalfMap2Projector[GPUID].t_DataRe },
                                                                    new[] { species.HalfMap1Projector[GPUID].t_DataIm, species.HalfMap2Projector[GPUID].t_DataIm },
                                                                    species.HalfMap1Projector[GPUID].Oversampling,
                                                                    species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                                                    new IntPtr((long)SpeciesParticleSubsets[species] + batchStart * sizeof(int)),
                                                                    CurBatch,
                                                                    1);
                                        }

                                        for (int i = 0; i < CurBatch; i++)
                                            CurrentSearchValues[t][idelta] += new float4(0,
                                                                                         (ResultP[i * 3 + 0] + ResultQ[i * 3 + 0]) * BatchContainmentMask[i * NTilts + t],
                                                                                         (ResultP[i * 3 + 1] + ResultQ[i * 3 + 1]) * BatchContainmentMask[i * NTilts + t],
                                                                                         (ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]) * BatchContainmentMask[i * NTilts + t]);
                                    }
                                }
                            }

                            PhaseCorrectionAll.Dispose();
                            PhaseCorrection.Dispose();
                            GammaCorrection.Dispose();
                            CoordsCTF.Dispose();
                            Extracted.Dispose();
                            ExtractedFT.Dispose();
                            ExtractedRefineSuper.Dispose();
                            ExtractedCropped.Dispose();
                            ExtractedCroppedFT.Dispose();
                            ExtractedCTF.Dispose();

                            GPU.DestroyFFTPlan(PlanForwSuper);
                            GPU.DestroyFFTPlan(PlanBackSuper);
                            GPU.DestroyFFTPlan(PlanForw);
                        }

                        GridSearchDelta /= 2;
                        for (int t = 0; t < NTilts; t++)
                        {
                            CurrentSearchValues[t].Sort((a, b) => -((a.Y / Math.Max(1e-20, Math.Sqrt(a.Z * a.W))).CompareTo(b.Y / Math.Max(1e-20, Math.Sqrt(b.Z * b.W)))));
                            AllSearchValues[t].AddRange(CurrentSearchValues[t]);

                            List<float4> NewSearchValues = new List<float4>();
                            for (int j = 0; j < 2; j++)
                            {
                                NewSearchValues.Add(new float4(CurrentSearchValues[t][j].X + (float)GridSearchDelta, 0, 0, 0));
                                NewSearchValues.Add(new float4(CurrentSearchValues[t][j].X - (float)GridSearchDelta, 0, 0, 0));
                            }

                            CurrentSearchValues[t] = NewSearchValues;
                        }
                    }

                    for (int i = 0; i < NTilts; i++)
                    {
                        AllSearchValues[i].Sort((a, b) => -((a.Y / Math.Max(1e-10, Math.Sqrt(a.Z * a.W))).CompareTo(b.Y / Math.Max(1e-10, Math.Sqrt(b.Z * b.W)))));
                        input[i * 4 + 0] += AllSearchValues[i][0].X;
                    }

                    return input;
                };

                #endregion

                BroydenFletcherGoldfarbShanno OptimizerWarp = new BroydenFletcherGoldfarbShanno(InitialParametersWarp.Length, WarpEval, WarpGrad);
                BroydenFletcherGoldfarbShanno OptimizerDefocus = new BroydenFletcherGoldfarbShanno(InitialParametersDefocus.Length, DefocusEval, DefocusGrad);

                //WarpEval(InitialParametersWarp);

                bool NeedReextraction = true;

                for (int ioptim = 0; ioptim < optionsMPA.NIterations; ioptim++)
                {
                    foreach (var species in allSpecies)
                        species.CurrentMaxShellRefinement = (int)Math.Round(MathHelper.Lerp(optionsMPA.InitialResolutionPercent / 100f,
                                                                                            1f,
                                                                                            optionsMPA.NIterations == 1 ? 1 : ((float)ioptim / (optionsMPA.NIterations - 1))) *
                                                                            species.HalfMap1Projector[GPUID].Dims.X / 2);

                    if (NeedReextraction)
                    {
                        progressCallback($"Re-extracting particles for optimization iteration {ioptim + 1}/{optionsMPA.NIterations}");
                        ReextractPaddedParticles(false);
                    }
                    NeedReextraction = false;

                    foreach (var step in OptimizationStepsWarp)
                    {
                        progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, " + step.Name);

                        BFGSIterations = step.Iterations;
                        CurrentOptimizationTypeWarp = step.Type;
                        CurrentWeightsDict = SpeciesTiltWeights;

                        OptimizerWarp.Maximize(InitialParametersWarp);

                        OldInput = null;
                    }

                    if (allSpecies.Any(s => s.ResolutionRefinement < (float)optionsMPA.MinimumCTFRefinementResolution))
                    {
                        //ReextractPaddedParticles();
                        //WarpEval(InitialParametersWarp);

                        if (ioptim == 0 && optionsMPA.DoDefocusGridSearch)
                        {
                            progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, defocus grid search");

                            InitialParametersDefocus = DefocusGridSearch(InitialParametersDefocus);

                            NeedReextraction = true;
                        }

                        //CurrentWeightsDict = SpeciesFrameWeights;
                        //ReextractPaddedParticles();
                        //WarpEval(InitialParametersWarp);

                        foreach (var step in OptimizationStepsCTF)
                        {
                            progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, " + step.Name);

                            BFGSIterations = step.Iterations;
                            CurrentOptimizationTypeCTF = step.Type;
                            CurrentWeightsDict = SpeciesCTFWeights;

                            OptimizerDefocus.Maximize(InitialParametersDefocus);

                            OldInput = null;
                            NeedReextraction = true;
                        }

                        if (NeedReextraction && ioptim >= optionsMPA.NIterations - 1)
                        {
                            progressCallback($"Re-extracting particles after optimization iteration {ioptim + 1}/{optionsMPA.NIterations}");
                            ReextractPaddedParticles(false);
                        }
                        //NeedReextraction = false;
                    }
                }

                SetWarpFromVector(InitialParametersWarp, this, true);
                SetDefocusFromVector(InitialParametersDefocus);

                Console.WriteLine("Final score: ");
                WarpEval(InitialParametersWarp);

                Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after optimization of {Name}");

                #region Compute NCC for each particle to be able to take only N % of the best later

                {
                    double[] AllParticleScores = GetPerParticleCC();

                    foreach (var species in allSpecies)
                    {
                        Particle[] Particles = SpeciesParticles[species];

                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        double[] ParticleScores = Helper.Subset(AllParticleScores, SpeciesParticleIDRanges[species].Start, SpeciesParticleIDRanges[species].End);

                        List<int> IndicesSorted = Helper.ArrayOfSequence(0, NParticles, 1).ToList();
                        IndicesSorted.Sort((a, b) => ParticleScores[a].CompareTo(ParticleScores[b]));
                        int FirstGoodIndex = (int)(NParticles * 0.0);

                        float[] Mask = new float[NParticles];
                        for (int i = 0; i < NParticles; i++)
                            Mask[IndicesSorted[i]] = (i >= FirstGoodIndex ? 1f : 0f);

                        GoodParticleMasks.Add(species, Mask);
                    }
                }

                #endregion

                #region Compute FSC between refs and particles to estimate tilt and series weights

                if (true)
                {
                    progressCallback($"Calculating FRC between projections and particles for weight optimization");

                    int FSCLength = 128;
                    Image FSC = new Image(new int3(FSCLength, FSCLength, NTilts * 3), true);
                    Image FSCPerParticle = new Image(new int3(FSCLength / 2, NParticlesOverall * 3, 1));
                    //float[][] FSCData = FSC.GetHost(Intent.ReadWrite);
                    //float[][] FSCPerParticleData = FSCPerParticle.GetHost(Intent.ReadWrite);
                    Image PhaseResiduals = new Image(new int3(FSCLength, FSCLength, 2), true);

                    Star TableOut = new Star(new string[] { "wrpNormCoordinateX", "wrpNormCoordinateY", "wrpNormCoordinateZ" });

                    int BatchSize = optionsMPA.BatchSize;
                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                    for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                    {
                        Species Species = allSpecies[ispecies];
                        Particle[] Particles = SpeciesParticles[Species];

                        int NParticles = Particles.Length;
                        float SpeciesAngPix = Species.ResolutionRefinement / 2;
                        if (NParticles == 0)
                            continue;

                        int SpeciesOffset = SpeciesParticleIDRanges[Species].Start;

                        int SizeRefine = SpeciesRefinementSize[Species];
                        int[] RelevantSizes = SpeciesRelevantRefinementSizes[Species];

                        //Image CorrAB = new Image(new int3(SizeRefine, SizeRefine, NTilts), true);
                        //Image CorrA2 = new Image(new int3(SizeRefine, SizeRefine, NTilts), true);
                        //Image CorrB2 = new Image(new int3(SizeRefine, SizeRefine, NTilts), true);

                        float ScaleFactor = (float)Species.PixelSize * (FSCLength / 2 - 1) /
                                            (float)(Species.ResolutionRefinement / 2 * (SizeRefine / 2 - 1));

                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[Species];
                        int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[Species];
                        float AngPixRefine = Species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(Species.DiameterAngstrom / AngPixRefine);

                        {
                            Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);   // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine
                            Image CoordsCTFCropped = CTF.GetCTFCoords(SizeRefine, SizeRefine);

                            Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                            Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                            Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                            Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                            Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                            int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                            int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                            int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                            if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                                throw new Exception("No FFT plans created!");

                            Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefineSuper);
                            Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                            bool[] EwaldReverse = { Species.EwaldReverse, !Species.EwaldReverse };

                            for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                            {
                                int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                                IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                                float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                                float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                                for (int i = 0; i < CurBatch; i++)
                                {
                                    float3 Coords = CoordinatesMoving[i * NTilts];
                                    Coords /= VolumeDimensionsPhysical;
                                    TableOut.AddRow(new string[] { Coords.X.ToString(CultureInfo.InvariantCulture),
                                                                   Coords.Y.ToString(CultureInfo.InvariantCulture),
                                                                   Coords.Z.ToString(CultureInfo.InvariantCulture) });
                                }

                                for (int t = 0; t < NTilts; t++)
                                {
                                    float3[] CoordinatesTilt = new float3[CurBatch];
                                    float3[] AnglesTilt = new float3[CurBatch];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                                        AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                                    }

                                    float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                    float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                                    float[] Defoci = new float[CurBatch];
                                    int3[] ExtractOrigins = new int3[CurBatch];
                                    float3[] ResidualShifts = new float3[BatchSize];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                        ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                        ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                        Defoci[p] = ImageCoords[p].Z;
                                    }

                                    for (int iewald = 0; iewald < (Species.DoEwald ? 2 : 1); iewald++)
                                    {
                                        GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                                    Extracted.GetDevice(Intent.Write),
                                                    TiltData[t].Dims.Slice(),
                                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                                    Helper.ToInterleaved(ExtractOrigins),
                                                    true,
                                                    (uint)CurBatch);

                                        GPU.FFT(Extracted.GetDevice(Intent.Read),
                                                ExtractedFT.GetDevice(Intent.Write),
                                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                                (uint)CurBatch,
                                                PlanForwSuper);

                                        ExtractedFT.ShiftSlices(ResidualShifts);
                                        ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                        GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                                   Extracted.GetDevice(Intent.Write),
                                                   new int3(SizeFullSuper, SizeFullSuper, 1),
                                                   new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                   (uint)CurBatch);

                                        GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                           PhaseCorrection.GetDevice(Intent.Read),
                                                                           Extracted.GetDevice(Intent.Write),
                                                                           PhaseCorrection.ElementsSliceComplex,
                                                                           (uint)CurBatch);

                                        if (Species.DoEwald)
                                        {
                                            GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, EwaldReverse[iewald], ExtractedCTF, true);

                                            GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                              ExtractedCTF.GetDevice(Intent.Read),
                                                                              ExtractedFT.GetDevice(Intent.Write),
                                                                              ExtractedCTF.ElementsComplex,
                                                                              1);
                                        }
                                        else
                                        {
                                            GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                            GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                                              ExtractedCTF.GetDevice(Intent.Read),
                                                                              ExtractedFT.GetDevice(Intent.Write),
                                                                              ExtractedCTF.ElementsComplex,
                                                                              1);
                                        }

                                        GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                                 Extracted.GetDevice(Intent.Write),
                                                 new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                 (uint)CurBatch,
                                                 PlanBackSuper,
                                                 false);

                                        GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                                        ExtractedCropped.GetDevice(Intent.Write),
                                                        new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                        new int3(SizeRefine, SizeRefine, 1),
                                                        (uint)CurBatch);

                                        GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                                       ExtractedCropped.GetDevice(Intent.Write),
                                                       ExtractedCropped.Dims.Slice(),
                                                       ParticleDiameterPix / 2f,
                                                       16 * AngPixExtract / AngPixRefine,
                                                       true,
                                                       (uint)CurBatch);

                                        GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                                ExtractedCroppedFT.GetDevice(Intent.Write),
                                                new int3(SizeRefine, SizeRefine, 1),
                                                (uint)CurBatch,
                                                PlanForw);

                                        ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                        GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTFCropped, null, t, ExtractedCTF, true, true, true);


                                        GPU.MultiParticleCorr2D(FSC.GetDeviceSlice(t * 3, Intent.ReadWrite),
                                                                new IntPtr((long)FSCPerParticle.GetDevice(Intent.ReadWrite) + (SpeciesOffset + batchStart) * FSCPerParticle.Dims.X * 3 * sizeof(float)),
                                                                PhaseResiduals.GetDevice(Intent.ReadWrite),
                                                                FSCLength,
                                                                new IntPtr[] { ExtractedCroppedFT.GetDevice(Intent.Read) },
                                                                ExtractedCTF.GetDevice(Intent.Read),
                                                                SizeRefine,
                                                                ScaleFactor,
                                                                null,
                                                                new float[CurBatch * 2],
                                                                Helper.ToInterleaved(ImageAngles),
                                                                MagnificationCorrection.ToVec(),
                                                                Species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize) * (iewald == 0 ? 1 : -1) : 0,
                                                                new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                                                new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                                                Species.HalfMap1Projector[GPUID].Oversampling,
                                                                Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                                                new IntPtr((long)SpeciesParticleSubsets[Species] + batchStart * sizeof(int)),
                                                                CurBatch,
                                                                1);
                                    }
                                }
                            }

                            PhaseCorrection.Dispose();
                            GammaCorrection.Dispose();

                            CoordsCTFCropped.Dispose();
                            CoordsCTF.Dispose();
                            Extracted.Dispose();
                            ExtractedFT.Dispose();
                            ExtractedCropped.Dispose();
                            ExtractedCroppedFT.Dispose();
                            ExtractedCTF.Dispose();

                            GPU.DestroyFFTPlan(PlanForwSuper);
                            GPU.DestroyFFTPlan(PlanBackSuper);
                            GPU.DestroyFFTPlan(PlanForw);
                        }
                    }

                    FSC.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", RootName + "_fsc.mrc"), true);
                    FSC.Dispose();

                    FSCPerParticle.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", RootName + "_fscparticles.mrc"), true);
                    FSCPerParticle.Dispose();

                    PhaseResiduals.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", RootName + "_phaseresiduals.mrc"), true);
                    PhaseResiduals.Dispose();

                    TableOut.Save(System.IO.Path.Combine(workingDirectory, "..", RootName + "_fscparticles.star"));
                }

                #endregion

                #region Tear down

                foreach (var pair in SpeciesParticleImages)
                {
                    foreach (var ptr in SpeciesParticleImages[pair.Key])
                        if (optionsMPA.UseHostMemory)
                            GPU.FreeHostPinned(ptr);
                        else
                            GPU.FreeDevice(ptr);
                    if (pair.Key.DoEwald)
                        foreach (var ptr in SpeciesParticleQImages[pair.Key])
                            GPU.FreeDevice(ptr);
                    SpeciesCTFWeights[pair.Key].Dispose();
                    SpeciesTiltWeights[pair.Key].Dispose();
                    GPU.FreeDevice(SpeciesParticleSubsets[pair.Key]);

                    pair.Key.HalfMap1Projector[GPUID].FreeDevice();
                    pair.Key.HalfMap2Projector[GPUID].FreeDevice();
                }

                Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after optimization teardown of {Name}");

                #endregion
            }

            #region Update reconstructions with newly aligned particles

            GPU.SetDevice(GPUID);
            GPU.CheckGPUExceptions();

            progressCallback($"Extracting and back-projecting particles...");

            foreach (var species in allSpecies)
            {
                if (SpeciesParticles[species].Length == 0)
                    continue;

                Projector[] Reconstructions = { species.HalfMap1Reconstruction[GPUID], species.HalfMap2Reconstruction[GPUID] };

                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                int BatchSize = optionsMPA.BatchSize;

                CTF MaxDefocusCTF = GetTiltCTF(IndicesSortedDose[0]);
                float ExpectedResolution = Math.Max((float)OptionsDataLoad.BinnedPixelSizeMean * 2, (float)species.GlobalResolution * 0.8f);
                int ExpectedBoxSize = (int)(species.DiameterAngstrom / (ExpectedResolution / 2)) * 2;
                int MinimumBoxSize = Math.Max(ExpectedBoxSize, MaxDefocusCTF.GetAliasingFreeSize(ExpectedResolution, (float)(species.DiameterAngstrom / (ExpectedResolution / 2))));
                int CTFSuperresFactor = (int)Math.Ceiling((float)MinimumBoxSize / ExpectedBoxSize);

                int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                int SizeFullSuper = SizeFull * CTFSuperresFactor;

                float Radius = species.DiameterAngstrom / 2;

                Image CTFCoords = CTF.GetCTFCoords(SizeFullSuper, SizeFullSuper);
                //float2[] CTFCoordsData = CTFCoords.GetHostComplexCopy()[0];
                Image CTFCoordsP = CTF.GetCTFPCoords(SizeFullSuper, SizeFullSuper);
                float2[] CTFCoordsPData = CTFCoordsP.GetHostComplexCopy()[0];
                Image CTFCoordsCropped = CTF.GetCTFCoords(SizeFull, SizeFull);

                Image GammaCorrection = CTF.GetGammaCorrection(AngPixExtract, SizeFullSuper);

                //float[] PQSigns = new float[CTFCoordsData.Length];
                //CTF.PrecomputePQSigns(SizeFullSuper, 2, species.EwaldReverse, CTFCoordsData, CTFCoordsPData, PQSigns);

                Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixExtract, SizeFullSuper);

                Image IntermediateMaskAngles = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, 2), true);
                Image IntermediateFTCorr = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                Image IntermediateCTFP = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);

                Image MaskParticle = new Image(new int3(SizeFullSuper, SizeFullSuper, 1));
                MaskParticle.Fill(1);
                MaskParticle.MaskSpherically((float)(species.DiameterAngstrom + 6) / AngPixExtract, 3, false);
                MaskParticle.RemapToFT();

                Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize));
                Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize));
                Image ExtractedCroppedFTp = new Image(new int3(SizeFull, SizeFull, BatchSize), true, true);
                Image ExtractedCroppedFTq = new Image(new int3(SizeFull, SizeFull, BatchSize), true, true);

                Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true);
                Image ExtractedCTFCropped = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize), true);
                Image CTFWeights = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize), true);

                int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                int PlanForw = GPU.CreateFFTPlan(new int3(SizeFull, SizeFull, 1), (uint)BatchSize);

                if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                    throw new Exception("No FFT plans created!");

                GPU.CheckGPUExceptions();

                Particle[] AllParticles = SpeciesParticles[species];
                Particle[][] SubsetParticles = { AllParticles.Where(p => p.RandomSubset == 0).ToArray(),
                                                 AllParticles.Where(p => p.RandomSubset == 1).ToArray() };

                //Image CTFAvg = new Image(new int3(SizeFull, SizeFull, BatchSize), true);

                for (int isubset = 0; isubset < 2; isubset++)
                {
                    Particle[] Particles = SubsetParticles[isubset];
                    int NParticles = Particles.Length;
                    //NParticles = 1;

                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                        IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                        float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                        float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                        for (int t = 0; t < NTilts; t++)
                        {
                            float3[] CoordinatesTilt = new float3[CurBatch];
                            float3[] AnglesTilt = new float3[CurBatch];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                                AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                            }

                            float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                            float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                            float[] Defoci = new float[CurBatch];
                            int3[] ExtractOrigins = new int3[CurBatch];
                            float3[] ResidualShifts = new float3[BatchSize];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                Defoci[p] = ImageCoords[p].Z;
                            }

                            float[] ContainmentMask = Helper.ArrayOfConstant(1f, BatchSize);
                            for (int i = 0; i < ImageCoords.Length; i++)
                            {
                                float3 Pos = ImageCoords[i];

                                float DistX = Math.Min(Pos.X, ImageDimensionsPhysical.X - Pos.X);
                                float DistY = Math.Min(Pos.Y, ImageDimensionsPhysical.Y - Pos.Y);
                                if (DistX < Radius || DistY < Radius)
                                    ContainmentMask[i] = 0;
                            }

                            #region Image data

                            GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                        Extracted.GetDevice(Intent.Write),
                                        TiltData[t].Dims.Slice(),
                                        new int3(SizeFullSuper, SizeFullSuper, 1),
                                        Helper.ToInterleaved(ExtractOrigins),
                                        true,
                                        (uint)CurBatch);

                            GPU.FFT(Extracted.GetDevice(Intent.Read),
                                    ExtractedFT.GetDevice(Intent.Write),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    (uint)CurBatch,
                                    PlanForwSuper);

                            ExtractedFT.ShiftSlices(ResidualShifts);
                            ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                            GPU.MultiplyComplexSlicesByComplex(ExtractedFT.GetDevice(Intent.Read),
                                                               PhaseCorrection.GetDevice(Intent.Read),
                                                               ExtractedFT.GetDevice(Intent.Write),
                                                               PhaseCorrection.ElementsComplex,
                                                               (uint)CurBatch);

                            CTF[] CTFParams = GetCTFParamsForOneTilt(AngPixExtract, Defoci, ImageCoords, t, false, false, false);

                            CTF.ApplyPandQPrecomp(ExtractedFT,
                                                  CTFParams,
                                                  IntermediateFTCorr,
                                                  Extracted,
                                                  ExtractedCropped,
                                                  IntermediateCTFP,
                                                  CTFCoords,
                                                  GammaCorrection,
                                                  species.EwaldReverse,
                                                  null,
                                                  PlanForw,
                                                  PlanBackSuper,
                                                  ExtractedCroppedFTp,
                                                  ExtractedCroppedFTq);

                            GetCTFsForOneTilt(AngPixExtract, Defoci, ImageCoords, CTFCoordsCropped, null, t, CTFWeights, true, true, true);
                            CTFWeights.Multiply(ContainmentMask);

                            ExtractedCroppedFTp.Multiply(CTFWeights);
                            ExtractedCroppedFTq.Multiply(CTFWeights);

                            #endregion

                            #region CTF data

                            //float[][] ExtractedCTFData = ExtractedCTF.GetHost(Intent.Write);
                            //Parallel.For(0, CurBatch, i =>
                            //{
                            //    CTFParams[i].GetEwaldWeights(CTFCoordsData, species.DiameterAngstrom, ExtractedCTFData[i]);
                            //});
                            GPU.CreateCTFEwaldWeights(ExtractedCTF.GetDevice(Intent.Write),
                                                      CTFCoords.GetDevice(Intent.Read),
                                                      GammaCorrection.GetDevice(Intent.Read),
                                                      species.DiameterAngstrom,
                                                      (uint)CTFCoords.ElementsSliceComplex,
                                                      Helper.ArrayOfFunction(i => CTFParams[i].ToStruct(), CurBatch),
                                                      (uint)CurBatch);
                            ExtractedCTF.Multiply(ExtractedCTF);

                            ExtractedFT.Fill(new float2(1, 0));
                            ExtractedFT.Multiply(ExtractedCTF);

                            GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                     Extracted.GetDevice(Intent.Write),
                                     new int3(SizeFullSuper, SizeFullSuper, 1),
                                     (uint)CurBatch,
                                     PlanBackSuper,
                                     false);

                            GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                           ExtractedCropped.GetDevice(Intent.Write),
                                           new int3(SizeFullSuper, SizeFullSuper, 1),
                                           new int3(SizeFull, SizeFull, 1),
                                           (uint)CurBatch);

                            GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                    ExtractedFT.GetDevice(Intent.Write),
                                    new int3(SizeFull, SizeFull, 1),
                                    (uint)CurBatch,
                                    PlanForw);

                            GPU.Real(ExtractedFT.GetDevice(Intent.Read),
                                     ExtractedCTFCropped.GetDevice(Intent.Write),
                                     ExtractedCTFCropped.ElementsReal);

                            ExtractedCTFCropped.Multiply(1f / (SizeFull * SizeFull));
                            ExtractedCTFCropped.Multiply(CTFWeights);

                            #endregion
                            //ImageAngles = new[] { new float3(0, 0, 0) };
                            //ImageAngles = Helper.ArrayOfConstant(new float3(0, 0, 0), CurBatch);

                            Reconstructions[isubset].BackProject(ExtractedCroppedFTp, ExtractedCTFCropped, ImageAngles, MagnificationCorrection, CTFParams[0].GetEwaldRadius(SizeFull, (float)species.PixelSize));
                            Reconstructions[isubset].BackProject(ExtractedCroppedFTq, ExtractedCTFCropped, ImageAngles, MagnificationCorrection, -CTFParams[0].GetEwaldRadius(SizeFull, (float)species.PixelSize));
                        }
                    }
                }

                //CTFAvg.WriteMRC("d_ctfavg.mrc", true);

                //EmpiricalWeights.Dispose();

                CTFCoords.Dispose();
                CTFCoordsP.Dispose();
                CTFCoordsCropped.Dispose();
                GammaCorrection.Dispose();
                PhaseCorrection.Dispose();
                Extracted.Dispose();
                ExtractedFT.Dispose();
                ExtractedCropped.Dispose();
                ExtractedCroppedFTp.Dispose();
                ExtractedCroppedFTq.Dispose();
                ExtractedCTF.Dispose();
                ExtractedCTFCropped.Dispose();
                CTFWeights.Dispose();

                MaskParticle.Dispose();

                IntermediateMaskAngles.Dispose();
                IntermediateFTCorr.Dispose();
                IntermediateCTFP.Dispose();

                GPU.DestroyFFTPlan(PlanForwSuper);
                GPU.DestroyFFTPlan(PlanBackSuper);
                GPU.DestroyFFTPlan(PlanForw);

                species.HalfMap1Reconstruction[GPUID].FreeDevice();
                species.HalfMap2Reconstruction[GPUID].FreeDevice();
            }

            Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after backprojection of {Name}");

            #endregion

            for (int t = 0; t < NTilts; t++)
                TiltData[t]?.FreeDevice();

            Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after full refinement of {Name}");
        }

        public void PerformMultiParticleRefinementOneTiltMovie(string workingDirectory,
                                                               ProcessingOptionsMPARefine optionsMPA,
                                                               Species[] allSpecies,
                                                               DataSource dataSource,
                                                               Movie tiltMovie,
                                                               Image[] tiltMovieData,
                                                               int tiltID,
                                                               Dictionary<Species, Particle[]> SpeciesParticles,
                                                               Dictionary<Species, IntPtr> SpeciesParticleSubsets,
                                                               Dictionary<Species, (int Start, int End)> SpeciesParticleIDRanges,
                                                               Dictionary<Species, float[]> SpeciesContainmentMasks,
                                                               Dictionary<Species, int> SpeciesRefinementSize,
                                                               Dictionary<Species, int[]> SpeciesRelevantRefinementSizes,
                                                               Dictionary<Species, Image> SpeciesFrameWeights,
                                                               Dictionary<Species, int> SpeciesCTFSuperresFactor)
        {
            int GPUID = GPU.GetDevice();
            HeaderEER.GroupNFrames = dataSource.EERGroupFrames;
            NFrames = MapHeader.ReadFromFile(tiltMovie.DataPath).Dimensions.Z;
            //NFrames = 1;
            FractionFrames = 1;

            if (true)
            {
                #region Resize grids

                if (tiltMovie.PyramidShiftX == null || tiltMovie.PyramidShiftX.Count == 0 || tiltMovie.PyramidShiftX[0].Dimensions.Z != NFrames)
                {
                    tiltMovie.PyramidShiftX = new List<CubicGrid>();
                    tiltMovie.PyramidShiftY = new List<CubicGrid>();

                    //tiltMovie.PyramidShiftX.Add(new CubicGrid(new int3(1, 1, NFrames)));
                    tiltMovie.PyramidShiftX.Add(new CubicGrid(new int3(3, 3, 3)));

                    //tiltMovie.PyramidShiftY.Add(new CubicGrid(new int3(1, 1, NFrames)));
                    tiltMovie.PyramidShiftY.Add(new CubicGrid(new int3(3, 3, 3)));
                }

                #endregion

                #region Figure out dimensions

                tiltMovie.ImageDimensionsPhysical = new float2(new int2(MapHeader.ReadFromFile(tiltMovie.DataPath).Dimensions)) * (float)dataSource.PixelSizeMean;

                float MinDose = MathHelper.Min(Dose), MaxDose = MathHelper.Max(Dose);
                float TiltInterpolationCoord = (Dose[tiltID] - MinDose) / (MaxDose - MinDose);

                float SmallestAngPix = MathHelper.Min(allSpecies.Select(s => (float)s.PixelSize));
                float LargestBox = MathHelper.Max(allSpecies.Select(s => s.DiameterAngstrom)) * 2 / SmallestAngPix;

                decimal BinTimes = (decimal)Math.Log(SmallestAngPix / (float)dataSource.PixelSizeMean, 2.0);
                ProcessingOptionsTomoSubReconstruction OptionsDataLoad = new ProcessingOptionsTomoSubReconstruction()
                {
                    PixelSize = dataSource.PixelSize,

                    BinTimes = BinTimes,
                    GainPath = dataSource.GainPath,
                    GainHash = "",
                    GainFlipX = dataSource.GainFlipX,
                    GainFlipY = dataSource.GainFlipY,
                    GainTranspose = dataSource.GainTranspose,
                    DefectsPath = dataSource.DefectsPath,
                    DefectsHash = "",

                    Invert = true,
                    NormalizeInput = true,
                    NormalizeOutput = false,

                    PrerotateParticles = true
                };

                #endregion

                foreach (var frame in tiltMovieData)
                {
                    frame.Bandpass(1f / LargestBox, 1f, false, 0f);
                    frame.Multiply(-1);
                }

                #region Extract particles

                Dictionary<Species, IntPtr[]> SpeciesParticleImages = new Dictionary<Species, IntPtr[]>();
                Dictionary<Species, float2[]> SpeciesParticleExtractedAt = new Dictionary<Species, float2[]>();

                int NParticlesOverall = 0;

                foreach (var species in allSpecies)
                {
                    if (SpeciesParticles[species].Length == 0)
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    NParticlesOverall += NParticles;

                    int Size = SpeciesRelevantRefinementSizes[species][tiltID];// species.HalfMap1Projector[GPUID].Dims.X;
                    long ElementsSliceComplex = (Size / 2 + 1) * Size;

                    SpeciesParticleImages.Add(species, Helper.ArrayOfFunction(i =>
                    {
                        long Footprint = (new int3(Size).Slice().ElementsFFT()) * 2 * (long)NParticles;

                        if (optionsMPA.UseHostMemory)
                            return GPU.MallocHostPinned(Footprint);
                        else
                            return GPU.MallocDevice(Footprint);
                    }, NFrames));

                    SpeciesParticleExtractedAt.Add(species, new float2[NParticles * NFrames]);
                }

                #endregion

                #region Helper functions

                Action<bool> ReextractPaddedParticles = (CorrectBeamTilt) =>
                {
                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                    int BatchSize = optionsMPA.BatchSize;

                    foreach (var species in allSpecies)
                    {
                        if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                            continue;

                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;

                        int SizeRelevant = SpeciesRelevantRefinementSizes[species][tiltID];
                        int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                        int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                        float AngPixRefine = species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                        float2[] ExtractedAt = SpeciesParticleExtractedAt[species];

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);
                        Image BeamTiltCorrection = CTF.GetBeamTilt(SizeRefineSuper, SizeFullSuper);

                        Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize));
                        Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCroppedFTRelevantSize = new Image(IntPtr.Zero, new int3(SizeRelevant, SizeRelevant, BatchSize), true, true);
                        Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true);

                        Image CTFFrameWeights = tiltMovie.GetCTFsForOneParticle(OptionsDataLoad, new float3(0, 0, 0), CoordsCTF, null, true, true);

                        //Image SumAll = new Image(new int3(SizeRefine, SizeRefine, BatchSize));

                        int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                        int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);


                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float3[] CoordinatesMoving = BatchParticles.Select(p => p.GetCoordinatesAt(TiltInterpolationCoord)).ToArray();

                            float3[] CoordinatesTilt = GetPositionsInOneTilt(CoordinatesMoving, tiltID);

                            for (int f = 0; f < NFrames; f++)
                            {
                                float3[] ImageCoords = tiltMovie.GetPositionsInOneFrame(CoordinatesTilt, f);

                                float[] Defoci = new float[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p] = CoordinatesTilt[p].Z;
                                    ExtractedAt[(batchStart + p) * NFrames + f] = new float2(ImageCoords[p]);
                                }

                                GPU.Extract(tiltMovieData[f].GetDevice(Intent.Read),
                                            Extracted.GetDevice(Intent.Write),
                                            tiltMovieData[f].Dims.Slice(),
                                            new int3(SizeFullSuper, SizeFullSuper, 1),
                                            Helper.ToInterleaved(ExtractOrigins),
                                            true,
                                            (uint)CurBatch);

                                GPU.FFT(Extracted.GetDevice(Intent.Read),
                                        ExtractedFT.GetDevice(Intent.Write),
                                        new int3(SizeFullSuper, SizeFullSuper, 1),
                                        (uint)CurBatch,
                                        PlanForwSuper);

                                ExtractedFT.ShiftSlices(ResidualShifts);
                                ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                           Extracted.GetDevice(Intent.Write),
                                           new int3(SizeFullSuper, SizeFullSuper, 1),
                                           new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                           (uint)CurBatch);

                                GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, null, tiltID, ExtractedCTF, true);

                                if (CorrectBeamTilt)
                                    GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                       BeamTiltCorrection.GetDevice(Intent.Read),
                                                                       Extracted.GetDevice(Intent.Write),
                                                                       BeamTiltCorrection.ElementsSliceComplex,
                                                                       (uint)CurBatch);

                                GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                                  ExtractedCTF.GetDevice(Intent.Read),
                                                                  Extracted.GetDevice(Intent.Write),
                                                                  ExtractedCTF.ElementsReal,
                                                                  1);

                                GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                                  CTFFrameWeights.GetDeviceSlice(f, Intent.Read),
                                                                  ExtractedFT.GetDevice(Intent.Write),
                                                                  CTFFrameWeights.ElementsSliceReal,
                                                                  (uint)CurBatch);

                                GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                         Extracted.GetDevice(Intent.Write),
                                         new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                         (uint)CurBatch,
                                         PlanBackSuper,
                                         false);

                                GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                                ExtractedCropped.GetDevice(Intent.Write),
                                                new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                new int3(SizeRefine, SizeRefine, 1),
                                                (uint)CurBatch);

                                GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                               ExtractedCropped.GetDevice(Intent.Write),
                                               ExtractedCropped.Dims.Slice(),
                                               ParticleDiameterPix / 2f,
                                               16 * AngPixExtract / AngPixRefine,
                                               true,
                                               (uint)CurBatch);

                                //SumAll.Add(ExtractedCropped);

                                GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                        ExtractedCroppedFT.GetDevice(Intent.Write),
                                        new int3(SizeRefine, SizeRefine, 1),
                                        (uint)CurBatch,
                                        PlanForw);

                                ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                           ExtractedCroppedFTRelevantSize.GetDevice(Intent.Write),
                                           new int3(SizeRefine).Slice(),
                                           new int3(SizeRelevant).Slice(),
                                           (uint)CurBatch);

                                GPU.CopyDeviceToHostPinned(ExtractedCroppedFTRelevantSize.GetDevice(Intent.Read),
                                                           new IntPtr((long)SpeciesParticleImages[species][f] + (new int3(SizeRelevant).Slice().ElementsFFT()) * 2 * batchStart * sizeof(float)),
                                                           (new int3(SizeRelevant).Slice().ElementsFFT()) * 2 * CurBatch);
                            }
                        }

                        //SumAll.AsReducedAlongZ().WriteMRC("d_sumall.mrc", true);
                        //SumAll.Dispose();

                        CTFFrameWeights.Dispose();

                        CoordsCTF.Dispose();
                        Extracted.Dispose();
                        ExtractedFT.Dispose();
                        ExtractedCropped.Dispose();
                        ExtractedCroppedFT.Dispose();
                        ExtractedCroppedFTRelevantSize.Dispose();
                        ExtractedCTF.Dispose();

                        GPU.DestroyFFTPlan(PlanForwSuper);
                        GPU.DestroyFFTPlan(PlanBackSuper);
                        GPU.DestroyFFTPlan(PlanForw);
                    }
                };

                Func<float2, float[]> GetRawCC = (shiftBias) =>
                {
                    float[] Result = new float[NParticlesOverall * NFrames * 3];

                    for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                    {
                        Species Species = allSpecies[ispecies];
                        Particle[] Particles = SpeciesParticles[Species];

                        int NParticles = Particles.Length;
                        float SpeciesAngPix = Species.ResolutionRefinement / 2;
                        if (NParticles == 0)
                            continue;

                        float[] SpeciesResult = new float[NParticles * NFrames * 3];

                        float3[] ParticlePositions = new float3[NParticles * NFrames];
                        float3[] ParticleAngles = new float3[NParticles * NFrames];
                        for (int p = 0; p < NParticles; p++)
                        {
                            float3 Position = Particles[p].GetCoordinatesAt(TiltInterpolationCoord);
                            float3 Angles = Particles[p].GetAnglesAt(TiltInterpolationCoord);

                            for (int f = 0; f < NFrames; f++)
                            {
                                ParticlePositions[p * NFrames + f] = Position;
                                ParticleAngles[p * NFrames + f] = Angles;
                            }
                        }

                        float3[] ParticlePositionsTilt = GetPositionsInOneTilt(ParticlePositions, tiltID);

                        float3[] ParticlePositionsProjected = tiltMovie.GetPositionInAllFrames(ParticlePositionsTilt);
                        float3[] ParticleAnglesInFrames = GetAnglesInOneTilt(ParticlePositions, ParticleAngles, tiltID);

                        float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[Species];
                        float2[] ParticleShifts = new float2[NFrames * NParticles];
                        for (int p = 0; p < NParticles; p++)
                            for (int t = 0; t < NFrames; t++)
                                ParticleShifts[p * NFrames + t] = (new float2(ParticlePositionsProjected[p * NFrames + t]) - ParticleExtractedAt[p * NFrames + t] + shiftBias) / SpeciesAngPix;

                        int SizeRelevant = SpeciesRelevantRefinementSizes[Species][tiltID];
                        int SizeRefine = Species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;

                        Image PhaseCorrection = CTF.GetBeamTilt(SizeRefine, SizeFull);
                        float2[] BeamTilts = Helper.ArrayOfConstant(CTF.BeamTilt, NParticles);
                        Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                        for (int t = 0; t < NFrames; t++)
                            GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                        PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                                        PhaseCorrection.Dims.Slice(),
                                        new int3(SizeRelevant).Slice(),
                                        1);

                        GPU.MultiParticleDiff(SpeciesResult,
                                              SpeciesParticleImages[Species],
                                              SpeciesRefinementSize[Species],
                                              Helper.ArrayOfConstant(SizeRelevant, NFrames),
                                              Helper.ToInterleaved(ParticleShifts),
                                              Helper.ToInterleaved(ParticleAnglesInFrames),
                                              MagnificationCorrection.ToVec(),
                                              SpeciesFrameWeights[Species].GetDevice(Intent.ReadWrite),
                                              PhaseCorrectionAll.GetDevice(Intent.Read),
                                              0,
                                              Species.CurrentMaxShellRefinement,
                                              new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                              new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                              Species.HalfMap1Projector[GPUID].Oversampling,
                                              Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                              SpeciesParticleSubsets[Species],
                                              NParticles,
                                              NFrames);

                        PhaseCorrectionAll.Dispose();
                        PhaseCorrection.Dispose();

                        int Offset = SpeciesParticleIDRanges[Species].Start * NFrames * 3;
                        Array.Copy(SpeciesResult, 0, Result, Offset, SpeciesResult.Length);
                    }

                    return Result;
                };

                Func<(float[] xp, float[] xm, float[] yp, float[] ym, float delta2)> GetRawShiftGradients = () =>
                {
                    float Delta = 0.025f;
                    float Delta2 = Delta * 2;

                    float[] h_ScoresXP = GetRawCC(float2.UnitX * Delta);
                    float[] h_ScoresXM = GetRawCC(-float2.UnitX * Delta);
                    float[] h_ScoresYP = GetRawCC(float2.UnitY * Delta);
                    float[] h_ScoresYM = GetRawCC(-float2.UnitY * Delta);

                    //for (int i = 0; i < Result.Length; i++)
                    //    Result[i] = new float2((h_ScoresXP[i] - h_ScoresXM[i]) / Delta2 * 100,
                    //                           (h_ScoresYP[i] - h_ScoresYM[i]) / Delta2 * 100);

                    return (h_ScoresXP, h_ScoresXM, h_ScoresYP, h_ScoresYM, Delta2);
                };

                Func<double[]> GetPerFrameDiff2 = () =>
                {
                    double[] Result = new double[NFrames * 3];
                    float[] RawResult = GetRawCC(new float2(0));

                    for (int p = 0; p < NParticlesOverall; p++)
                        for (int f = 0; f < NFrames; f++)
                        {
                            Result[f * 3 + 0] += RawResult[(p * NFrames + f) * 3 + 0];
                            Result[f * 3 + 1] += RawResult[(p * NFrames + f) * 3 + 1];
                            Result[f * 3 + 2] += RawResult[(p * NFrames + f) * 3 + 2];
                        }

                    Result = Helper.ArrayOfFunction(t => Result[t * 3 + 0] / Math.Max(1e-10, Math.Sqrt(Result[t * 3 + 1] * Result[t * 3 + 2])) * 100 * NParticlesOverall, NFrames);

                    return Result;
                };

                #endregion

                ReextractPaddedParticles(false);

                float2[][] OriginalOffsets = Helper.ArrayOfFunction(p => Helper.ArrayOfFunction(t => new float2(tiltMovie.PyramidShiftX[p].Values[t],
                                                                                                                tiltMovie.PyramidShiftY[p].Values[t]),
                                                                                                tiltMovie.PyramidShiftX[p].Values.Length),
                                                                    tiltMovie.PyramidShiftX.Count);

                int BFGSIterations = 0;

                double[] InitialParametersWarp = new double[tiltMovie.PyramidShiftX.Select(g => g.Values.Length * 2).Sum()];

                #region Set parameters from vector

                Action<double[], Movie> SetWarpFromVector = (input, movie) =>
                {
                    int Offset = 0;

                    int3[] PyramidDimensions = tiltMovie.PyramidShiftX.Select(g => g.Dimensions).ToArray();

                    movie.PyramidShiftX.Clear();
                    movie.PyramidShiftY.Clear();

                    for (int p = 0; p < PyramidDimensions.Length; p++)
                    {
                        float[] MovementXData = new float[PyramidDimensions[p].Elements()];
                        float[] MovementYData = new float[PyramidDimensions[p].Elements()];
                        for (int i = 0; i < MovementXData.Length; i++)
                        {
                            MovementXData[i] = OriginalOffsets[p][i].X + (float)input[Offset + i];
                            MovementYData[i] = OriginalOffsets[p][i].Y + (float)input[Offset + MovementXData.Length + i];
                        }
                        movie.PyramidShiftX.Add(new CubicGrid(PyramidDimensions[p], MovementXData));
                        movie.PyramidShiftY.Add(new CubicGrid(PyramidDimensions[p], MovementYData));

                        Offset += MovementXData.Length * 2;
                    }
                };

                #endregion

                #region Wiggle weights

                int NWiggleDifferentiable = tiltMovie.PyramidShiftX.Select(g => g.Values.Length * 2).Sum();
                (int[] indices, float2[] weights)[] AllWiggleWeights = new (int[] indices, float2[] weights)[NWiggleDifferentiable];

                {
                    Movie[] ParallelMovieCopies = Helper.ArrayOfFunction(i => new Movie(tiltMovie.Path), 32);
                    Dictionary<Species, float3[]> SpeciesParticlePositions = new Dictionary<Species, float3[]>();
                    foreach (var species in allSpecies)
                    {
                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        float3[] ParticlePositions = new float3[NParticles * NFrames];
                        for (int p = 0; p < NParticles; p++)
                        {
                            float3 Position = Particles[p].GetCoordinatesAt(TiltInterpolationCoord);

                            for (int f = 0; f < NFrames; f++)
                                ParticlePositions[p * NFrames + f] = Position;
                        }

                        float3[] ParticlePositionsTilt = GetPositionsInOneTilt(ParticlePositions, tiltID);
                        SpeciesParticlePositions.Add(species, ParticlePositionsTilt);
                    }

                    Helper.ForCPU(0, NWiggleDifferentiable, ParallelMovieCopies.Length, (threadID) =>
                    {
                        ParallelMovieCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                        ParallelMovieCopies[threadID].NFrames = NFrames;
                        ParallelMovieCopies[threadID].FractionFrames = FractionFrames;
                    },
                    (iwiggle, threadID) =>
                    {
                        double[] WiggleParams = new double[InitialParametersWarp.Length];
                        WiggleParams[iwiggle] = 1;
                        SetWarpFromVector(WiggleParams, ParallelMovieCopies[threadID]);

                        float2[] RawShifts = new float2[NParticlesOverall * NFrames];
                        foreach (var species in allSpecies)
                        {
                            Particle[] Particles = SpeciesParticles[species];
                            int NParticles = Particles.Length;
                            if (NParticles == 0)
                                continue;

                            int Offset = SpeciesParticleIDRanges[species].Start;

                            float[] ContainmentMask = SpeciesContainmentMasks[species];

                            float3[] ParticlePositionsProjected = ParallelMovieCopies[threadID].GetPositionInAllFrames(SpeciesParticlePositions[species]);
                            float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                            for (int p = 0; p < NParticles; p++)
                                for (int f = 0; f < NFrames; f++)
                                    RawShifts[(Offset + p) * NFrames + f] = (new float2(ParticlePositionsProjected[p * NFrames + f]) - ParticleExtractedAt[p * NFrames + f]) * ContainmentMask[p * NTilts + tiltID];
                        }

                        List<int> Indices = new List<int>(RawShifts.Length / 5);
                        List<float2> Weights = new List<float2>(RawShifts.Length / 5);
                        for (int i = 0; i < RawShifts.Length; i++)
                        {
                            if (RawShifts[i].LengthSq() > 1e-6f)
                            {
                                Indices.Add(i);
                                Weights.Add(RawShifts[i]);

                                if (Math.Abs(RawShifts[i].X) > 1.05f)
                                    throw new Exception();
                            }
                        }

                        AllWiggleWeights[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                    }, null);
                }

                #endregion

                #region Loss and gradient functions for warping

                Func<double[], double> WarpEval = input =>
                {
                    SetWarpFromVector(input, tiltMovie);

                    double[] TiltScores = GetPerFrameDiff2();
                    double Score = TiltScores.Sum();

                    Console.WriteLine(Score);

                    return Score;
                };

                Func<double[], double[]> WarpGrad = input =>
                {
                    double[] Result = new double[input.Length];

                    if (++BFGSIterations >= 12)
                        return Result;

                    SetWarpFromVector(input, tiltMovie);
                    (var XP, var XM, var YP, var YM, var Delta2Movement) = GetRawShiftGradients();

                    Parallel.For(0, AllWiggleWeights.Length, iwiggle =>
                    {
                        double SumGrad = 0;
                        double SumWeights = 0;
                        double SumWeightsGrad = 0;

                        int[] Indices = AllWiggleWeights[iwiggle].indices;
                        float2[] Weights = AllWiggleWeights[iwiggle].weights;

                        for (int i = 0; i < Indices.Length; i++)
                        {
                            int id = Indices[i];

                            SumWeights += Math.Abs(Weights[i].X) * Math.Sqrt(XP[id * 3 + 1] + XM[id * 3 + 1]) +
                                          Math.Abs(Weights[i].Y) * Math.Sqrt(YP[id * 3 + 1] + YM[id * 3 + 1]);
                            SumWeightsGrad += Math.Abs(Weights[i].X) + Math.Abs(Weights[i].Y);

                            double GradX = (XP[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(XP[id * 3 + 1] * XP[id * 3 + 2])) -
                                            XM[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(XM[id * 3 + 1] * XM[id * 3 + 2]))) / Delta2Movement;
                            double GradY = (YP[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(YP[id * 3 + 1] * YP[id * 3 + 2])) -
                                            YM[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(YM[id * 3 + 1] * YM[id * 3 + 2]))) / Delta2Movement;

                            SumGrad += Weights[i].X * Math.Sqrt(XP[id * 3 + 1] + XM[id * 3 + 1]) * GradX;
                            SumGrad += Weights[i].Y * Math.Sqrt(YP[id * 3 + 1] + YM[id * 3 + 1]) * GradY;
                        }

                        Result[iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                    });

                    return Result;
                };

                #endregion


                foreach (var species in allSpecies)
                    species.CurrentMaxShellRefinement = species.HalfMap1Projector[GPUID].Dims.X / 2;

                BroydenFletcherGoldfarbShanno OptimizerWarp = new BroydenFletcherGoldfarbShanno(InitialParametersWarp.Length, WarpEval, WarpGrad);

                SetWarpFromVector(InitialParametersWarp, tiltMovie);

                BFGSIterations = 0;
                OptimizerWarp.Maximize(InitialParametersWarp);

                SetWarpFromVector(InitialParametersWarp, tiltMovie);

                #region Compute FSC between refs and particles to estimate frame and micrograph weights

                if (false)
                {
                    int FSCLength = 64;
                    Image FSC = new Image(new int3(FSCLength, FSCLength, NFrames * 3), true);
                    Image FSCPerParticle = new Image(new int3(FSCLength / 2, NParticlesOverall * 3, 1));
                    //float[][] FSCData = FSC.GetHost(Intent.ReadWrite);
                    Image PhaseResiduals = new Image(new int3(FSCLength, FSCLength, 2), true);

                    Star TableOut = new Star(new string[] { "wrpNormCoordinateX", "wrpNormCoordinateY", "wrpNormCoordinateZ" });

                    int BatchSize = optionsMPA.BatchSize;
                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                    for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                    {
                        Species Species = allSpecies[ispecies];
                        Particle[] Particles = SpeciesParticles[Species];

                        int NParticles = Particles.Length;
                        float SpeciesAngPix = Species.ResolutionRefinement / 2;
                        if (NParticles == 0)
                            continue;

                        int SpeciesOffset = SpeciesParticleIDRanges[Species].Start;

                        int SizeRefine = SpeciesRefinementSize[Species];
                        int[] RelevantSizes = SpeciesRelevantRefinementSizes[Species];

                        float ScaleFactor = (float)Species.PixelSize * (FSCLength / 2 - 1) /
                                            (float)(Species.ResolutionRefinement / 2 * (SizeRefine / 2 - 1));

                        {
                            int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[Species];
                            int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;
                            int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[Species];
                            float AngPixRefine = Species.ResolutionRefinement / 2;
                            int ParticleDiameterPix = (int)(Species.DiameterAngstrom / AngPixRefine);

                            Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);   // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine
                            Image CoordsCTFCropped = CTF.GetCTFCoords(SizeRefine, SizeRefine);

                            Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                            Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                            Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                            Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                            Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true);

                            int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                            int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                            int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                            if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                                throw new Exception("No FFT plans created!");

                            Image BeamTiltCorrection = CTF.GetBeamTilt(SizeRefineSuper, SizeFullSuper);

                            for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                            {
                                int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                                IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                                float3[] CoordinatesMoving = BatchParticles.Select(p => p.GetCoordinatesAt(TiltInterpolationCoord)).ToArray();
                                float3[] AnglesMoving = BatchParticles.Select(p => p.GetAnglesAt(TiltInterpolationCoord)).ToArray();

                                float3[] CoordinatesTilt = GetPositionsInOneTilt(CoordinatesMoving, tiltID);
                                float3[] ParticleAnglesInFrames = GetAnglesInOneTilt(CoordinatesMoving, AnglesMoving, tiltID);

                                for (int i = 0; i < CurBatch; i++)
                                {
                                    float3 Coords = new float3(CoordinatesMoving[i].X, CoordinatesMoving[i].Y, CoordinatesMoving[i].Z);
                                    Coords /= VolumeDimensionsPhysical;
                                    TableOut.AddRow(new string[] { Coords.X.ToString(CultureInfo.InvariantCulture),
                                                                   Coords.Y.ToString(CultureInfo.InvariantCulture),
                                                                   Coords.Z.ToString(CultureInfo.InvariantCulture)});
                                }

                                for (int f = 0; f < NFrames; f++)
                                {
                                    float3[] ImageCoords = tiltMovie.GetPositionsInOneFrame(CoordinatesTilt, f);

                                    float[] Defoci = new float[CurBatch];
                                    int3[] ExtractOrigins = new int3[CurBatch];
                                    float3[] ResidualShifts = new float3[BatchSize];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                        ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                        ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                        Defoci[p] = CoordinatesTilt[p].Z;
                                    }

                                    GPU.Extract(tiltMovieData[f].GetDevice(Intent.Read),
                                                Extracted.GetDevice(Intent.Write),
                                                tiltMovieData[f].Dims.Slice(),
                                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                                Helper.ToInterleaved(ExtractOrigins),
                                                true,
                                                (uint)CurBatch);

                                    GPU.FFT(Extracted.GetDevice(Intent.Read),
                                            ExtractedFT.GetDevice(Intent.Write),
                                            new int3(SizeFullSuper, SizeFullSuper, 1),
                                            (uint)CurBatch,
                                            PlanForwSuper);

                                    ExtractedFT.ShiftSlices(ResidualShifts);
                                    ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                    GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                               Extracted.GetDevice(Intent.Write),
                                               new int3(SizeFullSuper, SizeFullSuper, 1),
                                               new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                               (uint)CurBatch);

                                    GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, null, tiltID, ExtractedCTF, true);

                                    GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                       BeamTiltCorrection.GetDevice(Intent.Read),
                                                                       Extracted.GetDevice(Intent.Write),
                                                                       BeamTiltCorrection.ElementsSliceComplex,
                                                                       (uint)CurBatch);

                                    GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                                      ExtractedCTF.GetDevice(Intent.Read),
                                                                      ExtractedFT.GetDevice(Intent.Write),
                                                                      ExtractedCTF.ElementsReal,
                                                                      1);

                                    GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                             Extracted.GetDevice(Intent.Write),
                                             new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                             (uint)CurBatch,
                                             PlanBackSuper,
                                             false);

                                    GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                                    ExtractedCropped.GetDevice(Intent.Write),
                                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                                    new int3(SizeRefine, SizeRefine, 1),
                                                    (uint)CurBatch);

                                    GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                                   ExtractedCropped.GetDevice(Intent.Write),
                                                   ExtractedCropped.Dims.Slice(),
                                                   ParticleDiameterPix / 2f,
                                                   16 * AngPixExtract / AngPixRefine,
                                                   true,
                                                   (uint)CurBatch);

                                    //SumAll.Add(ExtractedCropped);

                                    GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                            ExtractedCroppedFT.GetDevice(Intent.Write),
                                            new int3(SizeRefine, SizeRefine, 1),
                                            (uint)CurBatch,
                                            PlanForw);

                                    ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                    GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTFCropped, null, tiltID, ExtractedCTF, true, true, true);


                                    //GPU.MultiParticleCorr2D(CorrAB.GetDeviceSlice(f, Intent.ReadWrite),
                                    //                        CorrA2.GetDeviceSlice(f, Intent.ReadWrite),
                                    //                        CorrB2.GetDeviceSlice(f, Intent.ReadWrite),
                                    //                        new IntPtr[] { ExtractedCroppedFT.GetDevice(Intent.Read) },
                                    //                        SizeRefine,
                                    //                        null,
                                    //                        new float[CurBatch * 2],
                                    //                        Helper.ToInterleaved(ParticleAnglesInFrames),
                                    //                        MagnificationCorrection * new float3(Species.HalfMap1Projector[GPUID].Oversampling,
                                    //                                                             Species.HalfMap1Projector[GPUID].Oversampling,
                                    //                                                             1),
                                    //                        new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                    //                        new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                    //                        Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                    //                        new IntPtr((long)SpeciesParticleSubsets[Species] + batchStart * sizeof(int)),
                                    //                        CurBatch,
                                    //                        1);


                                    GPU.MultiParticleCorr2D(FSC.GetDeviceSlice(f * 3, Intent.ReadWrite),
                                                            new IntPtr((long)FSCPerParticle.GetDevice(Intent.ReadWrite) + (SpeciesOffset + batchStart) * FSCPerParticle.Dims.X * 3 * sizeof(float)),
                                                            PhaseResiduals.GetDevice(Intent.ReadWrite),
                                                            FSCLength,
                                                            new IntPtr[] { ExtractedCroppedFT.GetDevice(Intent.Read) },
                                                            ExtractedCTF.GetDevice(Intent.Read),
                                                            SizeRefine,
                                                            ScaleFactor,
                                                            null,
                                                            new float[CurBatch * 2],
                                                            Helper.ToInterleaved(ParticleAnglesInFrames),
                                                            MagnificationCorrection.ToVec(),
                                                            0,
                                                            new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                                            new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                                            Species.HalfMap1Projector[GPUID].Oversampling,
                                                            Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                                            new IntPtr((long)SpeciesParticleSubsets[Species] + batchStart * sizeof(int)),
                                                            CurBatch,
                                                            1);
                                }
                            }

                            BeamTiltCorrection.Dispose();

                            CoordsCTFCropped.Dispose();
                            CoordsCTF.Dispose();
                            Extracted.Dispose();
                            ExtractedFT.Dispose();
                            ExtractedCropped.Dispose();
                            ExtractedCroppedFT.Dispose();
                            ExtractedCTF.Dispose();

                            GPU.DestroyFFTPlan(PlanForwSuper);
                            GPU.DestroyFFTPlan(PlanBackSuper);
                            GPU.DestroyFFTPlan(PlanForw);
                        }
                    }

                    FSC.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", $"{RootName}_tilt{tiltID:D3}_fsc.mrc"), true);
                    FSC.Dispose();

                    FSCPerParticle.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", $"{RootName}_tilt{tiltID:D3}_fscparticles.mrc"), true);
                    FSCPerParticle.Dispose();

                    PhaseResiduals.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", RootName + "_phaseresiduals.mrc"), true);
                    PhaseResiduals.Dispose();

                    TableOut.Save(System.IO.Path.Combine(workingDirectory, "..", $"{RootName}_tilt{tiltID:D3}_fscparticles.star"));
                }

                #endregion

                #region Tear down

                foreach (var pair in SpeciesParticleImages)
                {
                    foreach (var ptr in SpeciesParticleImages[pair.Key])
                        if (optionsMPA.UseHostMemory)
                            GPU.FreeHostPinned(ptr);
                        else
                            GPU.FreeDevice(ptr);
                }

                #endregion
            }
        }


        public override long MultiParticleRefinementCalculateHostMemory(ProcessingOptionsMPARefine optionsMPA,
                                                                        Species[] allSpecies,
                                                                        DataSource dataSource)
        {
            long Result = 0;

            string DataHash = GetDataHash();
            int GPUID = GPU.GetDevice();

            foreach (var species in allSpecies)
            {
                int NParticles = species.GetParticles(DataHash).Length;

                int Size = species.HalfMap1Projector[GPUID].Dims.X;
                int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;

                int[] RelevantSizes = GetRelevantImageSizes(SizeFull, (float)optionsMPA.BFactorWeightingThreshold).Select(v => Math.Min(Size, v)).ToArray();

                Result += Helper.ArrayOfFunction(t => (new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles * sizeof(float), NTilts).Sum();
            }

            return Result;
        }

        #endregion

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

        static int[][] DirtErasureLabelsBuffer = new int[GPU.GetDeviceCount()][];
        static Image[] DirtErasureMaskBuffer = new Image[GPU.GetDeviceCount()];
        public static void EraseDirt(Image tiltImage, Image tiltMask)
        {
            if (tiltMask == null)
                return;

            float[] ImageData = tiltImage.GetHost(Intent.ReadWrite)[0];

            int CurrentDevice = GPU.GetDevice();

            #region Make sure reusable buffers are there and correctly sized

            if (DirtErasureLabelsBuffer[CurrentDevice] == null || DirtErasureLabelsBuffer[CurrentDevice].Length != ImageData.Length)
                DirtErasureLabelsBuffer[CurrentDevice] = new int[ImageData.Length];

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

                foreach (int id in component.ComponentIndices)
                    ImageData[id] = RandN.NextSingle(MeanStd.X, MeanStd.Y * 0.1f);

                foreach (int id in component.NeighborhoodIndices)
                    ImageData[id] = MathHelper.Lerp(ImageData[id], RandN.NextSingle(MeanStd.X, MeanStd.Y * 0.1f), MaskSmoothData[id]);
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

            byte[] ArraysCombined = Helper.Combine(Arrays);
            return MathHelper.GetSHA1(ArraysCombined);
        }

        #endregion

        #region Experimental

        public Image SimulateTiltSeries(TomoProcessingOptionsBase options, int3 stackDimensions, float3[][] particleOrigins, float3[][] particleAngles, int[] nParticles, Projector[] references)
        {
            VolumeDimensionsPhysical = options.DimensionsPhysical;
            float BinnedPixelSize = (float)options.BinnedPixelSizeMean;

            Image SimulatedStack = new Image(stackDimensions);

            // Extract images, mask and resize them, create CTFs

            for (int iref = 0; iref < references.Length; iref++)
            {
                int Size = references[iref].Dims.X;
                int3 Dims = new int3(Size);

                Image CTFCoords = CTF.GetCTFCoords(Size, Size);

                #region For each particle, create CTFs and projections, and insert them into the simulated tilt series

                for (int p = 0; p < nParticles[iref]; p++)
                {
                    float3 ParticleCoords = particleOrigins[iref][p];

                    float3[] Positions = GetPositionInAllTilts(ParticleCoords);
                    for (int i = 0; i < Positions.Length; i++)
                        Positions[i] /= BinnedPixelSize;

                    float3[] Angles = GetParticleAngleInAllTilts(ParticleCoords, particleAngles[iref][p]);

                    Image ParticleCTFs = GetCTFsForOneParticle(options, ParticleCoords, CTFCoords, null);

                    // Make projections

                    float3[] ImageShifts = new float3[NTilts];

                    for (int t = 0; t < NTilts; t++)
                    {
                        ImageShifts[t] = new float3(Positions[t].X - (int)Positions[t].X, // +diff because we are shifting the projections into experimental data frame
                                                    Positions[t].Y - (int)Positions[t].Y,
                                                    Positions[t].Z - (int)Positions[t].Z);
                    }

                    Image ProjectionsFT = references[iref].Project(new int2(Size), Angles);

                    ProjectionsFT.ShiftSlices(ImageShifts);
                    ProjectionsFT.Multiply(ParticleCTFs);
                    ParticleCTFs.Dispose();

                    Image Projections = ProjectionsFT.AsIFFT().AndDisposeParent();
                    Projections.RemapFromFT();


                    // Insert projections into tilt series

                    for (int t = 0; t < NTilts; t++)
                    {
                        int2 IntPosition = new int2((int)Positions[t].X, (int)Positions[t].Y) - Size / 2;

                        float[] SimulatedData = SimulatedStack.GetHost(Intent.Write)[t];

                        float[] ImageData = Projections.GetHost(Intent.Read)[t];
                        for (int y = 0; y < Size; y++)
                        {
                            int PosY = y + IntPosition.Y;
                            if (PosY < 0 || PosY >= stackDimensions.Y)
                                continue;

                            for (int x = 0; x < Size; x++)
                            {
                                int PosX = x + IntPosition.X;
                                if (PosX < 0 || PosX >= stackDimensions.X)
                                    continue;

                                SimulatedData[PosY * SimulatedStack.Dims.X + PosX] += ImageData[y * Size + x];
                            }
                        }
                    }

                    Projections.Dispose();
                }

                #endregion

                CTFCoords.Dispose();
            }

            return SimulatedStack;
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
    public class ProcessingOptionsTomoStack : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public bool ApplyMask { get; set; }
    }

    [Serializable]
    public class ProcessingOptionsTomoImportAlignments : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public decimal MinFOV { get; set; }

        [WarpSerializable]
        public string OverrideResultsDir { get; set; }
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
    }


    [Serializable]
    public class ProcessingOptionsTomoFullReconstruction : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public bool OverwriteFiles { get; set; }

        [WarpSerializable]
        public bool Invert { get; set; }

        [WarpSerializable]
        public bool Normalize { get; set; }

        [WarpSerializable]
        public bool DoDeconv { get; set; }

        [WarpSerializable]
        public decimal DeconvStrength { get; set; }

        [WarpSerializable]
        public decimal DeconvFalloff { get; set; }

        [WarpSerializable]
        public decimal DeconvHighpass { get; set; }

        [WarpSerializable]
        public int SubVolumeSize { get; set; }

        [WarpSerializable]
        public decimal SubVolumePadding { get; set; }

        [WarpSerializable]
        public bool PrepareDenoising { get; set; }

        [WarpSerializable]
        public bool PrepareDenoisingFrames { get; set; }

        [WarpSerializable]
        public bool PrepareDenoisingTilts { get; set; }

        [WarpSerializable]
        public bool KeepOnlyFullVoxels { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsTomoFullReconstruction)obj);
        }

        protected bool Equals(ProcessingOptionsTomoFullReconstruction other)
        {
            return base.Equals(other) &&
                   Invert == other.Invert &&
                   Normalize == other.Normalize &&
                   DoDeconv == other.DoDeconv &&
                   DeconvStrength == other.DeconvStrength &&
                   DeconvFalloff == other.DeconvFalloff &&
                   DeconvHighpass == other.DeconvHighpass &&
                   SubVolumeSize == other.SubVolumeSize &&
                   SubVolumePadding == other.SubVolumePadding &&
                   PrepareDenoising == other.PrepareDenoising &&
                   PrepareDenoisingFrames == other.PrepareDenoisingFrames &&
                   PrepareDenoisingTilts == other.PrepareDenoisingTilts &&
                   KeepOnlyFullVoxels == other.KeepOnlyFullVoxels;
        }

        public static bool operator ==(ProcessingOptionsTomoFullReconstruction left, ProcessingOptionsTomoFullReconstruction right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsTomoFullReconstruction left, ProcessingOptionsTomoFullReconstruction right)
        {
            return !Equals(left, right);
        }
    }

    [Serializable]
    public class ProcessingOptionsTomoFullMatch : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public bool OverwriteFiles { get; set; }

        [WarpSerializable]
        public int SubVolumeSize { get; set; }

        [WarpSerializable]
        public string TemplateName { get; set; }

        [WarpSerializable]
        public decimal TemplatePixel { get; set; }

        [WarpSerializable]
        public decimal TemplateDiameter { get; set; }

        [WarpSerializable]
        public decimal PeakDistance { get; set; }

        [WarpSerializable]
        public decimal TemplateFraction { get; set; }

        [WarpSerializable]
        public bool KeepOnlyFullVoxels { get; set; }

        [WarpSerializable]
        public bool WhitenSpectrum { get; set; }

        [WarpSerializable]
        public decimal Lowpass { get; set; }

        [WarpSerializable]
        public decimal LowpassSigma { get; set; }

        [WarpSerializable]
        public string Symmetry { get; set; }

        [WarpSerializable]
        public int HealpixOrder { get; set; }

        [WarpSerializable]
        public decimal TiltRange { get; set; }

        [WarpSerializable]
        public int BatchAngles { get; set; }

        [WarpSerializable]
        public int Supersample { get; set; }

        [WarpSerializable]
        public int NResults { get; set; }

        [WarpSerializable]
        public bool NormalizeScores { get; set; }

        [WarpSerializable]
        public bool ReuseCorrVolumes { get; set; }
    }

    [Serializable]
    public class ProcessingOptionsTomoSubReconstruction : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public string Suffix { get; set; }
        [WarpSerializable]
        public int BoxSize { get; set; }
        [WarpSerializable]
        public int ParticleDiameter { get; set; }
        [WarpSerializable]
        public bool Invert { get; set; }
        [WarpSerializable]
        public bool NormalizeInput { get; set; }
        [WarpSerializable]
        public bool NormalizeOutput { get; set; }
        [WarpSerializable]
        public bool PrerotateParticles { get; set; }
        [WarpSerializable]
        public bool DoLimitDose { get; set; }
        [WarpSerializable]
        public int NTilts { get; set; }
        [WarpSerializable]
        public bool MakeSparse { get; set; }
        [WarpSerializable]
        public bool UseCPU { get; set; }
    }
}
