using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Accord;
using Accord.Math.Optimization;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;
using IOPath = System.IO.Path;

namespace Warp
{
    public partial class Movie : WarpBase
    {
        private static BenchmarkTimer[] CTFTimers = Helper.ArrayOfFunction(i => new BenchmarkTimer(i.ToString()), 8);
        private static BenchmarkTimer[] ShiftTimers = Helper.ArrayOfFunction(i => new BenchmarkTimer(i.ToString()), 8);
        private static BenchmarkTimer[] OutputTimers = Helper.ArrayOfFunction(i => new BenchmarkTimer(i.ToString()), 8);

        #region Paths and names
        
        private string _Path = "";
        public string Path
        {
            get { return _Path; }
            set
            {
                if (value != _Path)
                {
                    _Path = value;
                    OnPropertyChanged();
                }
            }
        }

        public string DataPath => string.IsNullOrEmpty(DataDirectoryName) ? Path : IOPath.Combine(DataDirectoryName, Name);

        public string Name => Helper.PathToNameWithExtension(Path);

        public string RootName => Helper.PathToName(Path);

        public string ProcessingDirectoryName
        {
            get
            {
                if (Path.Length == 0)
                    return "";

                return Helper.PathToFolder(Path);
            }
        }

        public string DataDirectoryName = "";

        public string DataOrProcessingDirectoryName => string.IsNullOrEmpty(DataDirectoryName) ? ProcessingDirectoryName : DataDirectoryName;

        public static readonly string PowerSpectrumDirName = "powerspectrum";
        public string PowerSpectrumDir => IOPath.Combine(ProcessingDirectoryName, PowerSpectrumDirName);
        
        public static readonly string MotionTrackDirName = "motion";
        public string MotionTrackDir => IOPath.Combine(ProcessingDirectoryName, MotionTrackDirName);
        
        public static readonly string AverageDirName = "average";
        public string AverageDir => IOPath.Combine(ProcessingDirectoryName, AverageDirName);
        
        public static string AverageOddDirName => IOPath.Combine(AverageDirName, "odd");
        public string AverageOddDir => IOPath.Combine(ProcessingDirectoryName, AverageOddDirName);
        
        public static string AverageEvenDirName => IOPath.Combine(AverageDirName, "even");
        public string AverageEvenDir => IOPath.Combine(ProcessingDirectoryName, AverageEvenDirName);
        
        public static string AverageDenoisedDirName => IOPath.Combine(AverageDirName, "denoised");
        public string AverageDenoisedDir => IOPath.Combine(ProcessingDirectoryName, AverageDenoisedDirName);
        
        public static readonly string DeconvolvedDirName = "deconv"; 
        public string DeconvolvedDir => IOPath.Combine(ProcessingDirectoryName, DeconvolvedDirName);
        
        public static readonly string DenoiseTrainingDirName = "denoising";
        public string DenoiseTrainingDir => IOPath.Combine(ProcessingDirectoryName, DenoiseTrainingDirName);
        
        public static string DenoiseTrainingOddDirName => IOPath.Combine(DenoiseTrainingDirName, "odd");
        public string DenoiseTrainingDirOdd => IOPath.Combine(ProcessingDirectoryName, DenoiseTrainingOddDirName);
        
        public static string DenoiseTrainingEvenDirName => IOPath.Combine(DenoiseTrainingDirName, "even");
        public string DenoiseTrainingDirEven => IOPath.Combine(ProcessingDirectoryName, DenoiseTrainingEvenDirName);
        
        public static string DenoiseTrainingCTFDirName => IOPath.Combine(DenoiseTrainingDirName, "ctf");
        public string DenoiseTrainingDirCTF => IOPath.Combine(ProcessingDirectoryName, DenoiseTrainingCTFDirName);
        
        public static string DenoiseTrainingModelName => IOPath.Combine(DenoiseTrainingDirName, "model.pt");
        public string DenoiseTrainingDirModel => IOPath.Combine(ProcessingDirectoryName, DenoiseTrainingModelName);
        
        public static readonly string MaskDirName = "mask";
        public string MaskDir => IOPath.Combine(ProcessingDirectoryName, MaskDirName);
        
        public static readonly string SegmentationDirName = "segmentation";
        public string SegmentationDir => IOPath.Combine(ProcessingDirectoryName, SegmentationDirName);
        
        public static string MembraneSegmentationDirName => IOPath.Combine(SegmentationDirName, "membranes");
        public string MembraneSegmentationDir => IOPath.Combine(ProcessingDirectoryName, MembraneSegmentationDirName);
        
        public static readonly string ParticlesDirName = "particles";
        public string ParticlesDir => IOPath.Combine(ProcessingDirectoryName, ParticlesDirName);
        
        public static string ParticlesDenoisingOddDirName => IOPath.Combine(ParticlesDirName, "odd");
        public string ParticlesDenoisingOddDir => IOPath.Combine(ProcessingDirectoryName, ParticlesDenoisingOddDirName);
        
        public static string ParticlesDenoisingEvenDirName => IOPath.Combine(ParticlesDirName, "even");
        public string ParticlesDenoisingEvenDir => IOPath.Combine(ProcessingDirectoryName, ParticlesDenoisingEvenDirName);
        
        public static readonly string MatchingDirName = "matching";
        public string MatchingDir => IOPath.Combine(ProcessingDirectoryName, MatchingDirName);
        
        public static readonly string ThumbnailsDirName = "thumbnails";
        public string ThumbnailsDir => IOPath.Combine(ProcessingDirectoryName, ThumbnailsDirName);
        
        public static string ToPowerSpectrumPath(string name) => IOPath.Combine(PowerSpectrumDirName, Helper.PathToName(name) + ".mrc");
        public string PowerSpectrumPath => IOPath.Combine(PowerSpectrumDir, RootName + ".mrc");
        
        public static string ToAveragePath(string name) => IOPath.Combine(AverageDirName, Helper.PathToName(name) + ".mrc");
        public string AveragePath => IOPath.Combine(AverageDir, RootName + ".mrc");
        
        public static string ToAverageOddPath(string name) => IOPath.Combine(AverageOddDirName, Helper.PathToName(name) + ".mrc");
        public string AverageOddPath => IOPath.Combine(AverageOddDir, RootName + ".mrc");
        
        public static string ToAverageEvenPath(string name) => IOPath.Combine(AverageEvenDirName, Helper.PathToName(name) + ".mrc");
        public string AverageEvenPath => IOPath.Combine(AverageEvenDir, RootName + ".mrc");
        
        public static string ToAverageDenoisedPath(string name) => IOPath.Combine(AverageDenoisedDirName, Helper.PathToName(name) + ".mrc");
        public string AverageDenoisedPath => IOPath.Combine(AverageDenoisedDir, RootName + ".mrc");
        
        public static string ToDeconvolvedPath(string name) => IOPath.Combine(DeconvolvedDirName, Helper.PathToName(name) + ".mrc");
        public string DeconvolvedPath => IOPath.Combine(DeconvolvedDir, RootName + ".mrc");
        
        public static string ToDenoiseTrainingOddPath(string name) => IOPath.Combine(DenoiseTrainingOddDirName, Helper.PathToName(name) + ".mrc");
        public string DenoiseTrainingOddPath => IOPath.Combine(DenoiseTrainingDirOdd, RootName + ".mrc");
        
        public static string ToDenoiseTrainingEvenPath(string name) => IOPath.Combine(DenoiseTrainingEvenDirName, Helper.PathToName(name) + ".mrc");
        public string DenoiseTrainingEvenPath => IOPath.Combine(DenoiseTrainingDirEven, RootName + ".mrc");
        
        public static string ToDenoiseTrainingCTFPath(string name) => IOPath.Combine(DenoiseTrainingCTFDirName, Helper.PathToName(name) + ".mrc");
        public string DenoiseTrainingCTFPath => IOPath.Combine(DenoiseTrainingDirCTF, RootName + ".mrc");
        
        public static string ToMaskPath(string name) => IOPath.Combine(MaskDirName, Helper.PathToName(name) + ".tif");
        public string MaskPath => IOPath.Combine(MaskDir, RootName + ".tif");
        
        public static string ToThumbnailsPath(string name) => IOPath.Combine(ThumbnailsDirName, Helper.PathToName(name) + ".png");
        public string ThumbnailsPath => IOPath.Combine(ThumbnailsDir, RootName + ".png");
        
        public static string ToMotionTracksPath(string name) => IOPath.Combine(AverageDirName, Helper.PathToName(name) + "_motion.json");
        public string MotionTracksPath => IOPath.Combine(AverageDir, RootName + "_motion.json");

        public string XMLName => RootName + ".xml";
        public static string ToXMLPath(string name) => Helper.PathToName(name) + ".xml";
        public string XMLPath => IOPath.Combine(ProcessingDirectoryName, XMLName);

        #endregion

        public float GlobalBfactor = 0;
        public float GlobalWeight = 1;

        #region Runtime dimensions
        // These must be populated before most operations, otherwise exceptions will be thrown.
        // Not an elegant solution, but it avoids passing them to a lot of methods.
        // Given in Angstrom.

        public float2 ImageDimensionsPhysical;

        public int NFrames = 1;
        public float FractionFrames = 1;

        #endregion

        #region Selection

        protected Nullable<bool> _UnselectManual = null;
        public Nullable<bool> UnselectManual
        {
            get { return _UnselectManual; }
            set
            {
                if (value != _UnselectManual)
                {
                    _UnselectManual = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                    SaveMeta();
                }
            }
        }

        protected bool _UnselectFilter = false;
        public bool UnselectFilter
        {
            get { return _UnselectFilter; }
            set
            {
                if (value != _UnselectFilter)
                {
                    _UnselectFilter = value;
                    OnPropertyChanged();
                    //SaveMeta();
                }
            }
        }

        public ProcessingStatus ProcessingStatus = ProcessingStatus.Unprocessed;

        #endregion

        #region Power spectrum and CTF

        private CTF _CTF = new CTF();
        public CTF CTF
        {
            get { return _CTF; }
            set
            {
                if (value != _CTF)
                {
                    _CTF = value;
                    OnPropertyChanged();
                }
            }
        }

        private Matrix2 _MagnificationCorrection = new Matrix2(1, 0, 0, 1);
        /// <summary>
        /// 
        /// </summary>
        public Matrix2 MagnificationCorrection
        {
            get { return _MagnificationCorrection; }
            set { if (value != _MagnificationCorrection) { _MagnificationCorrection = value; OnPropertyChanged(); } }
        }

        private float2[] _PS1D;
        public float2[] PS1D
        {
            get { return _PS1D; }
            set
            {
                if (value != _PS1D)
                {
                    _PS1D = value;
                    OnPropertyChanged();
                }
            }
        }

        private float2[] _Simulated1D;
        public float2[] Simulated1D
        {
            get { return _Simulated1D ?? (_Simulated1D = GetSimulated1D()); }
            set
            {
                if (value != _Simulated1D)
                {
                    _Simulated1D = value;
                    OnPropertyChanged();
                }
            }
        }

        protected float2[] GetSimulated1D()
        {
            if (PS1D == null || SimulatedScale == null)
                return null;

            float[] SimulatedCTF = CTF.Get1D(PS1D.Length, true);

            float2[] Result = new float2[PS1D.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = new float2(PS1D[i].X, SimulatedCTF[i] * SimulatedScale.Interp(PS1D[i].X));

            return Result;
        }

        private Cubic1D _SimulatedBackground;
        public Cubic1D SimulatedBackground
        {
            get { return _SimulatedBackground; }
            set
            {
                if (value != _SimulatedBackground)
                {
                    _SimulatedBackground = value;
                    OnPropertyChanged();
                }
            }
        }

        private Cubic1D _SimulatedScale = new Cubic1D(new[] { new float2(0, 1), new float2(1, 1) });
        public Cubic1D SimulatedScale
        {
            get { return _SimulatedScale; }
            set
            {
                if (value != _SimulatedScale)
                {
                    _SimulatedScale = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _CTFResolutionEstimate = 0;
        public decimal CTFResolutionEstimate
        {
            get { return _CTFResolutionEstimate; }
            set { if (value != _CTFResolutionEstimate) { _CTFResolutionEstimate = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Movement

        private decimal _MeanFrameMovement = 0;
        public decimal MeanFrameMovement
        {
            get { return _MeanFrameMovement; }
            set { if (value != _MeanFrameMovement) { _MeanFrameMovement = value; OnPropertyChanged(); } }
        }

        public bool HasLocalMovement => (GridLocalX != null && GridLocalX.FlatValues.Length > 1) || (PyramidShiftX != null && PyramidShiftX.Count > 1);
        public bool HasGlobalMovement => (GridMovementX != null && GridMovementX.FlatValues.Length > 1) || (PyramidShiftX != null && PyramidShiftX.Count > 0);

        public void SaveMotionTracks()
        {
            if (!HasGlobalMovement)
                return;
            
            // evaluate motion tracks on a grid and save as json
            int nx = GridLocalX.Dimensions.X, ny = GridLocalX.Dimensions.Y;
            float gx, gy; // grid coordinates [0, 1]
            
            // dictionary to store JSON data structure
            var motionTracks = new Dictionary<string, Dictionary<string, float[]>>();
            
            for (int x = 0; x < nx; x++)
            {
                gx = x / (float)Math.Max(1, nx - 1);
                for (int y = 0; y < ny; y++)
                {
                    gy = y / (float)Math.Max(1, ny - 1);
                    
                    // Get the motion track for this cell
                    var track = GetMotionTrack(new float2(gx, gy));
                    
                    // Initialize arrays for each cell
                    float[] vx = track.Select(v => v.X).ToArray();  // x motion values
                    float[] vy = track.Select(v => v.Y).ToArray();  // y motion values
                    
                    // Use row_column format for key
                    string cellKey = $"{x}_{y}";
                    motionTracks[cellKey] = new Dictionary<string, float[]>
                    {
                        { "x", vx },
                        { "y", vy }
                    };
                    
                }
            }
            
            // Serialize to JSON, creating directory first if necessary
            string json = JsonSerializer.Serialize(motionTracks);
            string directoryPath = IOPath.GetDirectoryName(MotionTracksPath);
            if (!string.IsNullOrEmpty(directoryPath))
                Directory.CreateDirectory(directoryPath);
            File.WriteAllText(MotionTracksPath, json);
        }

        #endregion

        #region Grids

        private CubicGrid _GridCTFDefocus = new CubicGrid(new int3(1));
        public CubicGrid GridCTFDefocus
        {
            get { return _GridCTFDefocus; }
            set
            {
                if (value != _GridCTFDefocus)
                {
                    _GridCTFDefocus = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridCTFDefocusDelta = new CubicGrid(new int3(1));
        public CubicGrid GridCTFDefocusDelta
        {
            get { return _GridCTFDefocusDelta; }
            set
            {
                if (value != _GridCTFDefocusDelta)
                {
                    _GridCTFDefocusDelta = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridCTFDefocusAngle = new CubicGrid(new int3(1));
        public CubicGrid GridCTFDefocusAngle
        {
            get { return _GridCTFDefocusAngle; }
            set
            {
                if (value != _GridCTFDefocusAngle)
                {
                    _GridCTFDefocusAngle = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridCTFCs = new CubicGrid(new int3(1));
        public CubicGrid GridCTFCs
        {
            get { return _GridCTFCs; }
            set { if (value != _GridCTFCs) { _GridCTFCs = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridCTFPhase = new CubicGrid(new int3(1));
        public CubicGrid GridCTFPhase
        {
            get { return _GridCTFPhase; }
            set
            {
                if (value != _GridCTFPhase)
                {
                    _GridCTFPhase = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridCTFDoming = new CubicGrid(new int3(1));
        public CubicGrid GridCTFDoming
        {
            get { return _GridCTFDoming; }
            set { if (value != _GridCTFDoming) { _GridCTFDoming = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridMovementX = new CubicGrid(new int3(1));
        public CubicGrid GridMovementX
        {
            get { return _GridMovementX; }
            set
            {
                if (value != _GridMovementX)
                {
                    _GridMovementX = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridMovementY = new CubicGrid(new int3(1));
        public CubicGrid GridMovementY
        {
            get { return _GridMovementY; }
            set
            {
                if (value != _GridMovementY)
                {
                    _GridMovementY = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridLocalX = new CubicGrid(new int3(1));
        public CubicGrid GridLocalX
        {
            get { return _GridLocalX; }
            set
            {
                if (value != _GridLocalX)
                {
                    _GridLocalX = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridLocalY = new CubicGrid(new int3(1));
        public CubicGrid GridLocalY
        {
            get { return _GridLocalY; }
            set
            {
                if (value != _GridLocalY)
                {
                    _GridLocalY = value;
                    OnPropertyChanged();
                }
            }
        }

        private List<CubicGrid> _PyramidShiftX = new List<CubicGrid>();
        public List<CubicGrid> PyramidShiftX
        {
            get { return _PyramidShiftX; }
            set
            {
                if (value != _PyramidShiftX)
                {
                    _PyramidShiftX = value;
                    OnPropertyChanged();
                }
            }
        }

        private List<CubicGrid> _PyramidShiftY = new List<CubicGrid>();
        public List<CubicGrid> PyramidShiftY
        {
            get { return _PyramidShiftY; }
            set
            {
                if (value != _PyramidShiftY)
                {
                    _PyramidShiftY = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridAngleX = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridAngleX
        {
            get { return _GridAngleX; }
            set { if (value != _GridAngleX) { _GridAngleX = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridAngleY = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridAngleY
        {
            get { return _GridAngleY; }
            set { if (value != _GridAngleY) { _GridAngleY = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridAngleZ = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridAngleZ
        {
            get { return _GridAngleZ; }
            set { if (value != _GridAngleZ) { _GridAngleZ = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridDoseBfacs = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridDoseBfacs
        {
            get { return _GridDoseBfacs; }
            set { if (value != _GridDoseBfacs) { _GridDoseBfacs = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridDoseBfacsDelta = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridDoseBfacsDelta
        {
            get { return _GridDoseBfacsDelta; }
            set { if (value != _GridDoseBfacsDelta) { _GridDoseBfacsDelta = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridDoseBfacsAngle = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridDoseBfacsAngle
        {
            get { return _GridDoseBfacsAngle; }
            set { if (value != _GridDoseBfacsAngle) { _GridDoseBfacsAngle = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridDoseWeights = new CubicGrid(new int3(1, 1, 1), new[] { 1f });
        public CubicGrid GridDoseWeights
        {
            get { return _GridDoseWeights; }
            set { if (value != _GridDoseWeights) { _GridDoseWeights = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridLocationBfacs = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridLocationBfacs
        {
            get { return _GridLocationBfacs; }
            set { if (value != _GridLocationBfacs) { _GridLocationBfacs = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridLocationWeights = new CubicGrid(new int3(1, 1, 1), new[] { 1f });
        public CubicGrid GridLocationWeights
        {
            get { return _GridLocationWeights; }
            set { if (value != _GridLocationWeights) { _GridLocationWeights = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Processing options

        private bool _IsProcessing = false;
        public bool IsProcessing
        {
            get { return _IsProcessing; }
            set
            {
                if (value != _IsProcessing)
                {
                    _IsProcessing = value;
                    if (value)
                        OnProcessingStarted();
                    else
                        OnProcessingFinished();
                    OnPropertyChanged();
                }
            }
        }

        private ProcessingOptionsMovieCTF _OptionsCTF = null;
        public ProcessingOptionsMovieCTF OptionsCTF
        {
            get { return _OptionsCTF; }
            set
            {
                if (value != _OptionsCTF)
                {
                    _OptionsCTF = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                    OnCTF1DChanged();
                    OnCTF2DChanged();
                    OnPS2DChanged();
                }
            }
        }

        private ProcessingOptionsMovieMovement _OptionsMovement = null;
        public ProcessingOptionsMovieMovement OptionsMovement
        {
            get { return _OptionsMovement; }
            set
            {
                if (value != _OptionsMovement)
                {
                    _OptionsMovement = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                    OnMovementChanged();
                }
            }
        }

        private ProcessingOptionsMovieExport _OptionsMovieExport = null;
        public ProcessingOptionsMovieExport OptionsMovieExport
        {
            get { return _OptionsMovieExport; }
            set
            {
                if (value != _OptionsMovieExport)
                {
                    _OptionsMovieExport = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                    OnAverageChanged();
                }
            }
        }

        private ProcessingOptionsParticleExport _OptionsParticlesExport = null;
        public ProcessingOptionsParticleExport OptionsParticlesExport
        {
            get { return _OptionsParticlesExport; }
            set { if (value != _OptionsParticlesExport) { _OptionsParticlesExport = value; OnPropertyChanged(); } }
        }

        private ProcessingOptionsBoxNet _OptionsBoxNet = null;
        public ProcessingOptionsBoxNet OptionsBoxNet
        {
            get { return _OptionsBoxNet; }
            set
            {
                if (value != _OptionsBoxNet)
                {
                    _OptionsBoxNet = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                }
            }
        }

        public bool AreOptionsConflicted()
        {
            bool Result = false;

            if (OptionsCTF != null && OptionsMovement != null)
                Result |= OptionsCTF != OptionsMovement;
            if (OptionsCTF != null && OptionsMovieExport != null)
                Result |= OptionsCTF != OptionsMovieExport;
            if (OptionsMovement != null && OptionsMovieExport != null)
                Result |= OptionsMovement != OptionsMovieExport;

            return Result;
        }

        #endregion

        #region Picking and particles

        public readonly Dictionary<string, decimal> PickingThresholds = new Dictionary<string, decimal>();
        private readonly Dictionary<string, int> ParticleCounts = new Dictionary<string, int>();

        public int GetParticleCount(string suffix)
        {
            if (string.IsNullOrEmpty(suffix))
                return -1;

            lock (ParticleCounts)
            {
                if (!ParticleCounts.ContainsKey(suffix))
                {
                    if (File.Exists(IOPath.Combine(MatchingDir, RootName + suffix + ".star")))
                        ParticleCounts.Add(suffix, Star.CountLines(IOPath.Combine(MatchingDir, RootName + suffix + ".star")));
                    else
                        ParticleCounts.Add(suffix, -1);
                }
            }

            return ParticleCounts[suffix];
        }

        public void UpdateParticleCount(string suffix, int count = -1)
        {
            if (string.IsNullOrEmpty(suffix))
                return;

            int Result = Math.Max(-1, count);
            if (count < 0)
                if (File.Exists(IOPath.Combine(MatchingDir, RootName + suffix + ".star")))
                    Result = Star.CountLines(IOPath.Combine(MatchingDir, RootName + suffix + ".star"));

            lock (ParticleCounts)
            {
                if (ParticleCounts.ContainsKey(suffix))
                    ParticleCounts[suffix] = Result;
                else
                    ParticleCounts.Add(suffix, Result);
            }
        }

        public void DiscoverParticleSuffixes(string[] fileNames = null)
        {
            ParticleCounts.Clear();

            if (fileNames != null)
            {
                string _RootName = RootName;

                foreach (var name in fileNames.Where(s => s.Contains(_RootName)).ToArray())
                {
                    string Suffix = Helper.PathToName(name);
                    Suffix = Suffix.Substring(RootName.Length);

                    if (!string.IsNullOrEmpty(Suffix))
                        UpdateParticleCount(Suffix);
                }
            }
            else
            {
                //if (Directory.Exists(MatchingDir))
                //{
                //    foreach (var file in Directory.EnumerateFiles(MatchingDir, RootName + "*.star"))
                //    {
                //        string Suffix = Helper.PathToName(file);
                //        Suffix = Suffix.Substring(RootName.Length);

                //        if (!string.IsNullOrEmpty(Suffix))
                //            UpdateParticleCount(Suffix);
                //    }
                //}
            }
        }

        public IEnumerable<string> GetParticlesSuffixes()
        {
            return ParticleCounts.Keys;
        }

        public bool HasAnyParticleSuffixes()
        {
            return ParticleCounts.Count > 0;
        }

        public bool HasParticleSuffix(string suffix)
        {
            return ParticleCounts.ContainsKey(suffix);
        }

        private decimal _MaskPercentage = -1;
        public decimal MaskPercentage
        {
            get { return _MaskPercentage; }
            set { if (value != _MaskPercentage) { _MaskPercentage = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Events

        public event EventHandler ProcessingStarted;
        private void OnProcessingStarted() => ProcessingStarted?.Invoke(this, null);

        public event EventHandler ProcessingChanged;
        private void OnProcessingChanged() => ProcessingChanged?.Invoke(this, null);

        public event EventHandler ProcessingFinished;
        private void OnProcessingFinished() => ProcessingFinished?.Invoke(this, null);

        public event EventHandler CTF1DChanged;
        private void OnCTF1DChanged() => CTF1DChanged?.Invoke(this, null);

        public event EventHandler CTF2DChanged;
        private void OnCTF2DChanged() => CTF2DChanged?.Invoke(this, null);

        public event EventHandler PS2DChanged;
        private void OnPS2DChanged() => PS2DChanged?.Invoke(this, null);

        public event EventHandler MovementChanged;
        private void OnMovementChanged() => MovementChanged?.Invoke(this, null);

        public event EventHandler AverageChanged;
        private void OnAverageChanged() => AverageChanged?.Invoke(this, null);

        public event EventHandler ParticlesChanged;
        public void OnParticlesChanged() => ParticlesChanged?.Invoke(this, null);

        #endregion

        public Movie(string path, string dataDirectoryName = null, string[] particleFileNames = null)
        {
            Path = path;

            LoadMeta();
            DiscoverParticleSuffixes(particleFileNames);

            if (dataDirectoryName != null)
                DataDirectoryName = dataDirectoryName;
        }

        #region Load/save meta

        public virtual void LoadMeta()
        {
            if (!File.Exists(XMLPath))
                return;

            try
            {
                byte[] XMLBytes = File.ReadAllBytes(XMLPath);

                using (Stream SettingsStream = new MemoryStream(XMLBytes))
                {
                    XPathDocument Doc = new XPathDocument(SettingsStream);
                    XPathNavigator Reader = Doc.CreateNavigator();
                    Reader.MoveToRoot();
                    Reader.MoveToFirstChild();

                    string DataDirectory = XMLHelper.LoadAttribute(Reader, "DataDirectory", "");
                    if (!string.IsNullOrEmpty(DataDirectory))
                        DataDirectoryName = DataDirectory;

                    //_UnselectFilter = XMLHelper.LoadAttribute(Reader, "UnselectFilter", _UnselectFilter);
                    string UnselectManualString = XMLHelper.LoadAttribute(Reader, "UnselectManual", "null");
                    if (UnselectManualString != "null")
                        _UnselectManual = bool.Parse(UnselectManualString);
                    else
                        _UnselectManual = null;
                    CTFResolutionEstimate = XMLHelper.LoadAttribute(Reader, "CTFResolutionEstimate", CTFResolutionEstimate);
                    MeanFrameMovement = XMLHelper.LoadAttribute(Reader, "MeanFrameMovement", MeanFrameMovement);
                    MaskPercentage = XMLHelper.LoadAttribute(Reader, "MaskPercentage", MaskPercentage);

                    GlobalBfactor = XMLHelper.LoadAttribute(Reader, "Bfactor", GlobalBfactor);
                    GlobalWeight = XMLHelper.LoadAttribute(Reader, "Weight", GlobalWeight);

                    MagnificationCorrection = XMLHelper.LoadAttribute(Reader, "MagnificationCorrection", MagnificationCorrection);

                    XPathNavigator NavPS1D = Reader.SelectSingleNode("//PS1D");
                    if (NavPS1D != null)
                        PS1D = NavPS1D.InnerXml.Split(';').Select(v =>
                        {
                            string[] Pair = v.Split('|');
                            return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                        }).ToArray();

                    XPathNavigator NavSimBackground = Reader.SelectSingleNode("//SimulatedBackground");
                    if (NavSimBackground != null)
                        _SimulatedBackground = new Cubic1D(NavSimBackground.InnerXml.Split(';').Select(v =>
                        {
                            string[] Pair = v.Split('|');
                            return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                        }).ToArray());

                    XPathNavigator NavSimScale = Reader.SelectSingleNode("//SimulatedScale");
                    if (NavSimScale != null)
                        _SimulatedScale = new Cubic1D(NavSimScale.InnerXml.Split(';').Select(v =>
                        {
                            string[] Pair = v.Split('|');
                            return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                        }).ToArray());

                    XPathNavigator NavCTF = Reader.SelectSingleNode("//CTF");
                    if (NavCTF != null)
                        CTF.ReadFromXML(NavCTF);

                    XPathNavigator NavGridCTF = Reader.SelectSingleNode("//GridCTF");
                    if (NavGridCTF != null)
                        GridCTFDefocus = CubicGrid.Load(NavGridCTF);

                    XPathNavigator NavGridCTFDefocusDelta = Reader.SelectSingleNode("//GridCTFDefocusDelta");
                    if (NavGridCTFDefocusDelta != null)
                        GridCTFDefocusDelta = CubicGrid.Load(NavGridCTFDefocusDelta);

                    XPathNavigator NavGridCTFDefocusAngle = Reader.SelectSingleNode("//GridCTFDefocusAngle");
                    if (NavGridCTFDefocusAngle != null)
                        GridCTFDefocusAngle = CubicGrid.Load(NavGridCTFDefocusAngle);

                    XPathNavigator NavGridCTFCs = Reader.SelectSingleNode("//GridCTFCs");
                    if (NavGridCTFCs != null)
                        GridCTFCs = CubicGrid.Load(NavGridCTFCs);

                    XPathNavigator NavGridCTFPhase = Reader.SelectSingleNode("//GridCTFPhase");
                    if (NavGridCTFPhase != null)
                        GridCTFPhase = CubicGrid.Load(NavGridCTFPhase);

                    XPathNavigator NavGridCTFDoming = Reader.SelectSingleNode("//GridCTFDoming");
                    if (NavGridCTFDoming != null)
                        GridCTFDoming = CubicGrid.Load(NavGridCTFDoming);

                    XPathNavigator NavMoveX = Reader.SelectSingleNode("//GridMovementX");
                    if (NavMoveX != null)
                        GridMovementX = CubicGrid.Load(NavMoveX);

                    XPathNavigator NavMoveY = Reader.SelectSingleNode("//GridMovementY");
                    if (NavMoveY != null)
                        GridMovementY = CubicGrid.Load(NavMoveY);

                    XPathNavigator NavLocalX = Reader.SelectSingleNode("//GridLocalMovementX");
                    if (NavLocalX != null)
                        GridLocalX = CubicGrid.Load(NavLocalX);

                    XPathNavigator NavLocalY = Reader.SelectSingleNode("//GridLocalMovementY");
                    if (NavLocalY != null)
                        GridLocalY = CubicGrid.Load(NavLocalY);

                    PyramidShiftX.Clear();
                    foreach (XPathNavigator NavShiftX in Reader.Select("//PyramidShiftX"))
                        PyramidShiftX.Add(CubicGrid.Load(NavShiftX));

                    PyramidShiftY.Clear();
                    foreach (XPathNavigator NavShiftY in Reader.Select("//PyramidShiftY"))
                        PyramidShiftY.Add(CubicGrid.Load(NavShiftY));

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

                    XPathNavigator NavOptionsCTF = Reader.SelectSingleNode("//OptionsCTF");
                    if (NavOptionsCTF != null)
                    {
                        ProcessingOptionsMovieCTF Temp = new ProcessingOptionsMovieCTF();
                        Temp.ReadFromXML(NavOptionsCTF);
                        OptionsCTF = Temp;
                    }

                    XPathNavigator NavOptionsMovement = Reader.SelectSingleNode("//OptionsMovement");
                    if (NavOptionsMovement != null)
                    {
                        ProcessingOptionsMovieMovement Temp = new ProcessingOptionsMovieMovement();
                        Temp.ReadFromXML(NavOptionsMovement);
                        OptionsMovement = Temp;
                    }

                    XPathNavigator NavOptionsBoxNet = Reader.SelectSingleNode("//OptionsBoxNet");
                    if (NavOptionsBoxNet != null)
                    {
                        ProcessingOptionsBoxNet Temp = new ProcessingOptionsBoxNet();
                        Temp.ReadFromXML(NavOptionsBoxNet);
                        OptionsBoxNet = Temp;
                    }

                    XPathNavigator NavOptionsExport = Reader.SelectSingleNode("//OptionsMovieExport");
                    if (NavOptionsExport != null)
                    {
                        ProcessingOptionsMovieExport Temp = new ProcessingOptionsMovieExport();
                        Temp.ReadFromXML(NavOptionsExport);
                        OptionsMovieExport = Temp;
                    }

                    XPathNavigator NavOptionsParticlesExport = Reader.SelectSingleNode("//OptionsParticlesExport");
                    if (NavOptionsParticlesExport != null)
                    {
                        ProcessingOptionsParticleExport Temp = new ProcessingOptionsParticleExport();
                        Temp.ReadFromXML(NavOptionsParticlesExport);
                        OptionsParticlesExport = Temp;
                    }

                    XPathNavigator NavPickingThresholds = Reader.SelectSingleNode("//PickingThresholds");
                    if (NavPickingThresholds != null)
                    {
                        PickingThresholds.Clear();

                        foreach (XPathNavigator nav in NavPickingThresholds.SelectChildren("Threshold", ""))
                            try
                            {
                                PickingThresholds.Add(nav.GetAttribute("Suffix", ""), decimal.Parse(nav.GetAttribute("Value", ""), CultureInfo.InvariantCulture));
                            }
                            catch { }
                    }
                }
            }
            catch
            {
                return;
            }
        }

        public virtual void SaveMeta()
        {
            Directory.CreateDirectory(ProcessingDirectoryName);

            using (XmlTextWriter Writer = new XmlTextWriter(XMLPath, Encoding.UTF8))
            {
                Writer.Formatting = Formatting.Indented;
                Writer.IndentChar = '\t';
                Writer.Indentation = 1;
                Writer.WriteStartDocument();
                Writer.WriteStartElement("Movie");

                Writer.WriteAttributeString("DataDirectory", DataDirectoryName ?? "");

                Writer.WriteAttributeString("UnselectFilter", UnselectFilter.ToString());
                Writer.WriteAttributeString("UnselectManual", UnselectManual != null ? UnselectManual.ToString() : "null");

                Writer.WriteAttributeString("CTFResolutionEstimate", CTFResolutionEstimate.ToString(CultureInfo.InvariantCulture));
                Writer.WriteAttributeString("MeanFrameMovement", MeanFrameMovement.ToString(CultureInfo.InvariantCulture));
                Writer.WriteAttributeString("MaskPercentage", MaskPercentage.ToString(CultureInfo.InvariantCulture));

                Writer.WriteAttributeString("Bfactor", GlobalBfactor.ToString(CultureInfo.InvariantCulture));
                Writer.WriteAttributeString("Weight", GlobalWeight.ToString(CultureInfo.InvariantCulture));

                Writer.WriteAttributeString("MagnificationCorrection", MagnificationCorrection.ToString());

                if (OptionsCTF != null)
                {
                    Writer.WriteStartElement("OptionsCTF");
                    OptionsCTF.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                if (PS1D != null)
                {
                    Writer.WriteStartElement("PS1D");
                    Writer.WriteString(string.Join(";", PS1D.Select(v => v.X.ToString(CultureInfo.InvariantCulture) + "|" + v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (SimulatedBackground != null)
                {
                    Writer.WriteStartElement("SimulatedBackground");
                    Writer.WriteString(string.Join(";",
                                                   _SimulatedBackground.Data.Select(v => v.X.ToString(CultureInfo.InvariantCulture) +
                                                                                         "|" +
                                                                                         v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (SimulatedScale != null)
                {
                    Writer.WriteStartElement("SimulatedScale");
                    Writer.WriteString(string.Join(";",
                                                   _SimulatedScale.Data.Select(v => v.X.ToString(CultureInfo.InvariantCulture) +
                                                                                    "|" +
                                                                                    v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                Writer.WriteStartElement("CTF");
                CTF.WriteToXML(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTF");
                GridCTFDefocus.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFDefocusDelta");
                GridCTFDefocusDelta.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFDefocusAngle");
                GridCTFDefocusAngle.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFCs");
                GridCTFCs.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFPhase");
                GridCTFPhase.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFDoming");
                GridCTFDoming.Save(Writer);
                Writer.WriteEndElement();

                if (OptionsMovement != null)
                {
                    Writer.WriteStartElement("OptionsMovement");
                    OptionsMovement.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                Writer.WriteStartElement("GridMovementX");
                GridMovementX.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridMovementY");
                GridMovementY.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridLocalMovementX");
                GridLocalX.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridLocalMovementY");
                GridLocalY.Save(Writer);
                Writer.WriteEndElement();

                foreach (var grid in PyramidShiftX)
                {
                    Writer.WriteStartElement("PyramidShiftX");
                    grid.Save(Writer);
                    Writer.WriteEndElement();
                }

                foreach (var grid in PyramidShiftY)
                {
                    Writer.WriteStartElement("PyramidShiftY");
                    grid.Save(Writer);
                    Writer.WriteEndElement();
                }

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

                if (OptionsBoxNet != null)
                {
                    Writer.WriteStartElement("OptionsBoxNet");
                    OptionsBoxNet.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                if (OptionsMovieExport != null)
                {
                    Writer.WriteStartElement("OptionsMovieExport");
                    OptionsMovieExport.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                if (OptionsParticlesExport != null)
                {
                    Writer.WriteStartElement("OptionsParticlesExport");
                    OptionsParticlesExport.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                if (PickingThresholds.Count > 0)
                {
                    Writer.WriteStartElement("PickingThresholds");
                    foreach (var pair in PickingThresholds)
                    {
                        Writer.WriteStartElement("Threshold");

                        XMLHelper.WriteAttribute(Writer, "Suffix", pair.Key);
                        XMLHelper.WriteAttribute(Writer, "Value", pair.Value);

                        Writer.WriteEndElement();
                    }
                    Writer.WriteEndElement();
                }

                Writer.WriteEndElement();
                Writer.WriteEndDocument();
            }
        }

        #endregion

        #region Hashes

        public virtual string GetDataHash()
        {
            FileInfo Info = new FileInfo(DataPath);
            byte[] DataBytes = new byte[Math.Min(1 << 19, Info.Length)];
            using (BinaryReader Reader = new BinaryReader(File.OpenRead(DataPath)))
            {
                Reader.Read(DataBytes, 0, DataBytes.Length);
            }

            DataBytes = Helper.Combine(Helper.ToBytes(RootName.ToCharArray()), DataBytes);

            return MathHelper.GetSHA1(DataBytes);
        }

        public virtual string GetProcessingHash()
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

            if (GridLocalX != null)
            {
                Arrays.Add(GridLocalX.Dimensions);
                Arrays.Add(Helper.ToBytes(GridLocalX.FlatValues));
            }

            if (GridLocalY != null)
            {
                Arrays.Add(GridLocalY.Dimensions);
                Arrays.Add(Helper.ToBytes(GridLocalY.FlatValues));
            }

            if (PyramidShiftX != null)
                foreach (var grid in PyramidShiftX)
                {
                    Arrays.Add(grid.Dimensions);
                    Arrays.Add(Helper.ToBytes(grid.FlatValues));
                }

            if (PyramidShiftY != null)
                foreach (var grid in PyramidShiftY)
                {
                    Arrays.Add(grid.Dimensions);
                    Arrays.Add(Helper.ToBytes(grid.FlatValues));
                }

            byte[] ArraysCombined = Helper.Combine(Arrays);
            return MathHelper.GetSHA1(ArraysCombined);
        }

        #endregion

        #region On-the-fly tasks

        public static Task WriteAverageAsync = null;
        private static Image ExportMovieCTFCoords = null;
        private static Image ExportMovieWiener = null;

        #endregion

        #region Helper functions

        #region GetPosition methods

        public float3[] GetPositionInAllFrames(float3 coords)
        {
            float3[] PerFrameCoords = Helper.ArrayOfConstant(coords, NFrames);

            return GetPositionInAllFrames(PerFrameCoords);
        }

        public float3[] GetPositionInAllFrames(float3[] coords)
        {
            float3[] Result = new float3[coords.Length];

            float GridStep = 1f / Math.Max(1, (NFrames - 1));

            float3[] GridCoords = new float3[coords.Length];
            float3[] GridCoordsFractional = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                int f = i % NFrames;

                GridCoords[i] = new float3(coords[i].X / ImageDimensionsPhysical.X, coords[i].Y / ImageDimensionsPhysical.Y, f * GridStep);
                GridCoordsFractional[i] = GridCoords[i];
                GridCoordsFractional[i].Z *= FractionFrames;
            }

            float[] GridGlobalXInterp = GridMovementX.GetInterpolatedNative(GridCoordsFractional);
            float[] GridGlobalYInterp = GridMovementY.GetInterpolatedNative(GridCoordsFractional);

            float[] GridLocalXInterp = GridLocalX.GetInterpolatedNative(GridCoordsFractional);
            float[] GridLocalYInterp = GridLocalY.GetInterpolatedNative(GridCoordsFractional);

            float[][] GridPyramidXInterp = new float[PyramidShiftX.Count][];
            float[][] GridPyramidYInterp = new float[PyramidShiftY.Count][];
            for (int p = 0; p < PyramidShiftX.Count; p++)
            {
                GridPyramidXInterp[p] = PyramidShiftX[p].GetInterpolatedNative(GridCoords);
                GridPyramidYInterp[p] = PyramidShiftY[p].GetInterpolatedNative(GridCoords);
            }

            float[] GridDefocusInterp = GridCTFDefocus.GetInterpolatedNative(GridCoords);
            float[] GridDomingInterp = GridCTFDoming.GetInterpolatedNative(GridCoords);

            for (int i = 0; i < coords.Length; i++)
            {
                float3 Transformed = coords[i];

                Transformed.X -= GridGlobalXInterp[i];
                Transformed.Y -= GridGlobalYInterp[i];

                Transformed.X -= GridLocalXInterp[i];
                Transformed.Y -= GridLocalYInterp[i];

                for (int p = 0; p < PyramidShiftX.Count; p++)
                {
                    Transformed.X -= GridPyramidXInterp[p][i];
                    Transformed.Y -= GridPyramidYInterp[p][i];
                }

                Transformed.Z = Transformed.Z * 1e-4f + GridDefocusInterp[i] + GridDomingInterp[i];


                Result[i] = Transformed;
            }

            return Result;
        }

        public float3[] GetPositionsInOneFrame(float3[] coords, int frameID)
        {
            float3[] Result = new float3[coords.Length];

            float GridStep = 1f / Math.Max(1, (NFrames - 1));

            float3[] GridCoords = new float3[coords.Length];
            float3[] GridCoordsFractional = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                GridCoords[i] = new float3(coords[i].X / ImageDimensionsPhysical.X, coords[i].Y / ImageDimensionsPhysical.Y, frameID * GridStep);
                GridCoordsFractional[i] = GridCoords[i];
                GridCoordsFractional[i].Z *= FractionFrames;
            }

            float[] GridGlobalXInterp = GridMovementX.GetInterpolatedNative(GridCoordsFractional);
            float[] GridGlobalYInterp = GridMovementY.GetInterpolatedNative(GridCoordsFractional);

            float[] GridLocalXInterp = GridLocalX.GetInterpolatedNative(GridCoordsFractional);
            float[] GridLocalYInterp = GridLocalY.GetInterpolatedNative(GridCoordsFractional);

            float[][] GridPyramidXInterp = new float[PyramidShiftX.Count][];
            float[][] GridPyramidYInterp = new float[PyramidShiftY.Count][];
            for (int p = 0; p < PyramidShiftX.Count; p++)
            {
                GridPyramidXInterp[p] = PyramidShiftX[p].GetInterpolatedNative(GridCoords);
                GridPyramidYInterp[p] = PyramidShiftY[p].GetInterpolatedNative(GridCoords);
            }

            float[] GridDefocusInterp = GridCTFDefocus.GetInterpolatedNative(GridCoords);
            float[] GridDomingInterp = GridCTFDoming.GetInterpolatedNative(GridCoords);

            for (int i = 0; i < coords.Length; i++)
            {
                float3 Transformed = coords[i];

                Transformed.X -= GridGlobalXInterp[i];
                Transformed.Y -= GridGlobalYInterp[i];

                Transformed.X -= GridLocalXInterp[i];
                Transformed.Y -= GridLocalYInterp[i];

                for (int p = 0; p < PyramidShiftX.Count; p++)
                {
                    Transformed.X -= GridPyramidXInterp[p][i];
                    Transformed.Y -= GridPyramidYInterp[p][i];
                }

                Transformed.Z = Transformed.Z * 1e-4f + GridDefocusInterp[i] + GridDomingInterp[i];


                Result[i] = Transformed;
            }

            return Result;
        }

        #endregion

        #region GetAngle methods

        public Matrix3[] GetParticleRotationMatrixInAllFrames(float3[] coords, float3[] angle)
        {
            Matrix3[] Result = new Matrix3[coords.Length];

            float GridStep = 1f / (NFrames - 1);

            float3[] GridCoords = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                int f = i % NFrames;
                GridCoords[i] = new float3(coords[i].X / ImageDimensionsPhysical.X, coords[i].Y / ImageDimensionsPhysical.X, f * GridStep);
            }

            float[] GridAngleXInterp = GridAngleX.GetInterpolatedNative(GridCoords);
            float[] GridAngleYInterp = GridAngleY.GetInterpolatedNative(GridCoords);
            float[] GridAngleZInterp = GridAngleZ.GetInterpolatedNative(GridCoords);

            for (int i = 0; i < coords.Length; i++)
            {
                Matrix3 ParticleMatrix = Matrix3.Euler(angle[i].X * Helper.ToRad,
                                                       angle[i].Y * Helper.ToRad,
                                                       angle[i].Z * Helper.ToRad);

                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleYInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleXInterp[i] * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * ParticleMatrix;

                Result[i] = Rotation;
            }

            return Result;
        }

        public float3[] GetParticleAngleInAllFrames(float3 coords, float3 angle)
        {
            float3[] PerTiltCoords = new float3[NFrames];
            float3[] PerTiltAngles = new float3[NFrames];
            for (int i = 0; i < NFrames; i++)
            {
                PerTiltCoords[i] = coords;
                PerTiltAngles[i] = angle;
            }

            return GetParticleAngleInAllFrames(PerTiltCoords, PerTiltAngles);
        }

        public float3[] GetParticleAngleInAllFrames(float3[] coords, float3[] angle)
        {
            float3[] Result = new float3[coords.Length];

            Matrix3[] Matrices = GetParticleRotationMatrixInAllFrames(coords, angle);

            for (int i = 0; i < Result.Length; i++)
                Result[i] = Matrix3.EulerFromMatrix(Matrices[i]);

            return Result;
        }

        public float3[] GetAnglesInOneFrame(float3[] coords, float3[] particleAngles, int frameID)
        {
            int NParticles = coords.Length;
            float3[] Result = new float3[NParticles];

            float GridStep = 1f / (NFrames - 1);

            for (int p = 0; p < NParticles; p++)
            {
                float3 GridCoords = new float3(coords[p].X / ImageDimensionsPhysical.X, coords[p].Y / ImageDimensionsPhysical.Y, frameID * GridStep);

                Matrix3 ParticleMatrix = Matrix3.Euler(particleAngles[p].X * Helper.ToRad,
                                                       particleAngles[p].Y * Helper.ToRad,
                                                       particleAngles[p].Z * Helper.ToRad);

                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZ.GetInterpolated(GridCoords) * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleY.GetInterpolated(GridCoords) * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleX.GetInterpolated(GridCoords) * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * ParticleMatrix;

                Result[p] = Matrix3.EulerFromMatrix(Rotation);
            }

            return Result;
        }

        #endregion

        #region GetImages methods

        public virtual Image GetImagesForOneParticle(ProcessingOptionsBase options, Image[] imageData, int size, float3 coords, int planForw = 0, int maskDiameter = -1, int maskEdge = 8, Image result = null, Image resultFT = null)
        {
            float3[] PerFrameCoords = Helper.ArrayOfConstant(coords, NFrames);

            return GetImagesForOneParticle(options, imageData, size, PerFrameCoords, planForw, maskDiameter, maskEdge, true, result, resultFT);
        }

        public virtual Image GetImagesForOneParticle(ProcessingOptionsBase options, Image[] imageData, int size, float3[] coordsMoving, int planForw = 0, int maskDiameter = -1, int maskEdge = 8, bool doDecenter = true, Image result = null, Image resultFT = null)
        {
            float3[] ImagePositions = GetPositionInAllFrames(coordsMoving);
            for (int t = 0; t < ImagePositions.Length; t++)
                ImagePositions[t] /= (float)options.BinnedPixelSizeMean;

            Image Result = result == null ? new Image(new int3(size, size, NFrames)) : result;
            float[][] ResultData = Result.GetHost(Intent.Write);
            float3[] Shifts = new float3[NFrames];

            int Decenter = doDecenter ? size / 2 : 0;

            for (int t = 0; t < NFrames; t++)
            {
                int3 DimsMovie = imageData[t].Dims;

                ImagePositions[t] -= size / 2;

                int2 IntPosition = new int2((int)ImagePositions[t].X, (int)ImagePositions[t].Y);
                float2 Residual = new float2(-(ImagePositions[t].X - IntPosition.X), -(ImagePositions[t].Y - IntPosition.Y));
                IntPosition.X += DimsMovie.X * 99;                                                                                   // In case it is negative, for the periodic boundaries modulo later
                IntPosition.Y += DimsMovie.Y * 99;
                Shifts[t] = new float3(Residual.X + Decenter, Residual.Y + Decenter, 0);                                             // Include an fftshift() for Fourier-space rotations later

                float[] OriginalData = imageData[t].GetHost(Intent.Read)[0];
                float[] ImageData = ResultData[t];

                unsafe
                {
                    fixed (float* OriginalDataPtr = OriginalData)
                    fixed (float* ImageDataPtr = ImageData)
                    {
                        for (int y = 0; y < size; y++)
                        {
                            int PosY = (y + IntPosition.Y) % DimsMovie.Y;
                            for (int x = 0; x < size; x++)
                            {
                                int PosX = (x + IntPosition.X) % DimsMovie.X;
                                ImageDataPtr[y * size + x] = OriginalDataPtr[PosY * DimsMovie.X + PosX];
                            }
                        }
                    }
                }
            };

            if (maskDiameter > 0)
                GPU.SphereMask(Result.GetDevice(Intent.Read),
                                Result.GetDevice(Intent.Write),
                                Result.Dims.Slice(),
                                maskDiameter / 2f,
                                maskEdge,
                                false,
                                (uint)Result.Dims.Z);

            Image ResultFT = resultFT == null ? new Image(IntPtr.Zero, new int3(size, size, NFrames), true, true) : resultFT;
            GPU.FFT(Result.GetDevice(Intent.Read),
                    ResultFT.GetDevice(Intent.Write),
                    Result.Dims,
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

        public virtual Image GetCTFsForOneParticle(ProcessingOptionsBase options, float3 coords, Image ctfCoords, Image gammaCorrection, bool weighted = true, bool weightsonly = false, bool useglobalweights = false, Image result = null)
        {
            float3[] PerFrameCoords = Helper.ArrayOfConstant(coords, NFrames);

            return GetCTFsForOneParticle(options, PerFrameCoords, ctfCoords, gammaCorrection, weighted, weightsonly, useglobalweights, result);
        }

        public virtual Image GetCTFsForOneParticle(ProcessingOptionsBase options, float3[] coordsMoving, Image ctfCoords, Image gammaCorrection, bool weighted = true, bool weightsonly = false, bool useglobalweights = false, Image result = null)
        {
            float3[] ImagePositions = GetPositionInAllFrames(coordsMoving);

            float GridStep = 1f / Math.Max(1, (NFrames - 1));

            CTFStruct[] Params = new CTFStruct[NFrames];
            for (int f = 0; f < NFrames; f++)
            {
                decimal Defocus = (decimal)ImagePositions[f].Z;

                CTF CurrCTF = CTF.GetCopy();
                CurrCTF.PixelSize = options.BinnedPixelSizeMean;
                if (!weightsonly)
                {
                    CurrCTF.Defocus = Defocus;
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
                    float3 InterpAt = new float3(coordsMoving[f].X / ImageDimensionsPhysical.X,
                                                 coordsMoving[f].Y / ImageDimensionsPhysical.Y,
                                                 f * GridStep);

                    CurrCTF.Bfactor = (decimal)GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, f * GridStep)) +
                                      (decimal)Math.Min(0, GridLocationBfacs.GetInterpolated(InterpAt));
                    CurrCTF.BfactorDelta = (decimal)GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, f * GridStep));
                    CurrCTF.BfactorAngle = (decimal)GridDoseBfacsAngle.GetInterpolated(new float3(0.5f, 0.5f, f * GridStep));

                    CurrCTF.Bfactor = Math.Min(CurrCTF.Bfactor, -Math.Abs(CurrCTF.BfactorDelta));

                    CurrCTF.Scale = (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, f * GridStep)) *
                                    (decimal)Math.Min(1, GridLocationWeights.GetInterpolated(InterpAt));

                    if (useglobalweights)
                    {
                        CurrCTF.Bfactor += (decimal)GlobalBfactor;
                        CurrCTF.Scale *= (decimal)GlobalWeight;
                    }
                }

                Params[f] = CurrCTF.ToStruct();
            }

            Image Result = new Image(IntPtr.Zero, new int3(ctfCoords.Dims.X, ctfCoords.Dims.Y, NFrames), true);
            GPU.CreateCTF(Result.GetDevice(Intent.Write),
                          ctfCoords.GetDevice(Intent.Read),
                          gammaCorrection == null ? IntPtr.Zero : gammaCorrection.GetDevice(Intent.Read),
                          (uint)ctfCoords.ElementsSliceComplex,
                          Params,
                          false,
                          (uint)NFrames);

            return Result;
        }

        public void GetCTFsForOneFrame(float pixelSize, float3[] defoci, float3[] coords, Image ctfCoords, Image gammaCorrection, int frameID, Image outSimulated, bool weighted = true, bool weightsonly = false, bool useglobalweights = false)
        {
            int NParticles = defoci.Length;
            CTFStruct[] Params = new CTFStruct[NParticles];

            float GridStep = 1f / Math.Max(1, NFrames - 1);

            CTF ProtoCTF = CTF.GetCopy();
            ProtoCTF.PixelSize = (decimal)pixelSize;
            if (!weightsonly)
            {

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
                Bfac = (decimal)GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));
                BfacDelta = (decimal)GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));
                BfacAngle = (decimal)GridDoseBfacsAngle.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));
                Weight = (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));

                //Bfac = Math.Min(Bfac, -Math.Abs(BfacDelta));

                //if (onlyanisoweights)
                //{
                //    Bfac = -Math.Abs(BfacDelta);
                //    Weight = 1;
                //}

                if (useglobalweights)// && !onlyanisoweights)
                {
                    Bfac += (decimal)GlobalBfactor;
                    Weight *= (decimal)GlobalWeight;
                }
            }

            float[] CsValues = GetCs(coords);

            for (int p = 0; p < NParticles; p++)
            {
                if (!weightsonly)
                {
                    ProtoCTF.Defocus = (decimal)defoci[p].X;
                    ProtoCTF.DefocusDelta = (decimal)defoci[p].Y;
                    ProtoCTF.DefocusAngle = (decimal)defoci[p].Z;
                    ProtoCTF.Cs = (decimal)CsValues[p];
                }

                if (weighted)
                {
                    ProtoCTF.Bfactor = Bfac;
                    ProtoCTF.BfactorDelta = BfacDelta;
                    ProtoCTF.BfactorAngle = BfacAngle;
                    ProtoCTF.Scale = Weight;

                    float3 InterpAt = new float3(coords[p].X / ImageDimensionsPhysical.X,
                                                 coords[p].Y / ImageDimensionsPhysical.Y,
                                                 0.5f);
                    ProtoCTF.Bfactor += (decimal)Math.Min(0, GridLocationBfacs.GetInterpolated(InterpAt));
                    ProtoCTF.Scale *= (decimal)Math.Min(1, GridLocationWeights.GetInterpolated(InterpAt));
                }

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

        public void GetComplexCTFsForOneFrame(float pixelSize, float3[] defoci, float3[] coords, Image ctfCoords, Image gammaCorrection, int frameID, bool reverse, Image outSimulated, bool weighted = true, bool weightsonly = false, bool useglobalweights = false)
        {
            int NParticles = defoci.Length;
            CTFStruct[] Params = new CTFStruct[NParticles];

            float GridStep = 1f / Math.Max(1, NFrames - 1);

            CTF ProtoCTF = CTF.GetCopy();
            ProtoCTF.PixelSize = (decimal)pixelSize;
            if (!weightsonly)
            {

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
                Bfac = (decimal)GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));
                BfacDelta = (decimal)GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));
                BfacAngle = (decimal)GridDoseBfacsAngle.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));
                Weight = (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));

                //Bfac = Math.Min(Bfac, -Math.Abs(BfacDelta));

                if (useglobalweights)
                {
                    Bfac += (decimal)GlobalBfactor;
                    Weight *= (decimal)GlobalWeight;
                }
            }

            float[] CsValues = GetCs(coords);

            for (int p = 0; p < NParticles; p++)
            {
                if (!weightsonly)
                {
                    ProtoCTF.Defocus = (decimal)defoci[p].X;
                    ProtoCTF.DefocusDelta = (decimal)defoci[p].Y;
                    ProtoCTF.DefocusAngle = (decimal)defoci[p].Z;
                    ProtoCTF.Cs = (decimal)CsValues[p];
                }

                if (weighted)
                {
                    ProtoCTF.Bfactor = Bfac;
                    ProtoCTF.BfactorDelta = BfacDelta;
                    ProtoCTF.BfactorAngle = BfacAngle;
                    ProtoCTF.Scale = Weight;

                    float3 InterpAt = new float3(coords[p].X / ImageDimensionsPhysical.X,
                                                 coords[p].Y / ImageDimensionsPhysical.Y,
                                                 0.5f);
                    ProtoCTF.Bfactor += (decimal)Math.Min(0, GridLocationBfacs.GetInterpolated(InterpAt));
                    ProtoCTF.Scale *= (decimal)Math.Min(1, GridLocationWeights.GetInterpolated(InterpAt));
                }

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

        public CTF[] GetCTFParamsForOneFrame(float pixelSize, float3[] defoci, float3[] coords, int frameID, bool weighted = true, bool weightsonly = false, bool useglobalweights = false)
        {
            int NParticles = defoci.Length;
            CTF[] Params = new CTF[NParticles];

            float GridStep = 1f / Math.Max(1, NFrames - 1);

            CTF ProtoCTF = CTF.GetCopy();
            ProtoCTF.PixelSize = (decimal)pixelSize;
            if (!weightsonly)
            {

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
                Bfac = (decimal)GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));
                BfacDelta = (decimal)GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));
                BfacAngle = (decimal)GridDoseBfacsAngle.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));
                Weight = (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, frameID * GridStep));

                if (useglobalweights)
                {
                    Bfac += (decimal)GlobalBfactor;
                    Weight *= (decimal)GlobalWeight;
                }
            }

            float[] CsValues = GetCs(coords);

            for (int p = 0; p < NParticles; p++)
            {
                if (!weightsonly)
                {
                    ProtoCTF.Defocus = (decimal)defoci[p].X;
                    ProtoCTF.DefocusDelta = (decimal)defoci[p].Y;
                    ProtoCTF.DefocusAngle = (decimal)defoci[p].Z;
                    ProtoCTF.Cs = (decimal)CsValues[p];
                }

                if (weighted)
                {
                    ProtoCTF.Bfactor = Bfac;
                    ProtoCTF.BfactorDelta = BfacDelta;
                    ProtoCTF.BfactorAngle = BfacAngle;
                    ProtoCTF.Scale = Weight;

                    float3 InterpAt = new float3(coords[p].X / ImageDimensionsPhysical.X,
                                                 coords[p].Y / ImageDimensionsPhysical.Y,
                                                 0.5f);
                    ProtoCTF.Bfactor += (decimal)Math.Min(0, GridLocationBfacs.GetInterpolated(InterpAt));
                    ProtoCTF.Scale *= (decimal)Math.Min(1, GridLocationWeights.GetInterpolated(InterpAt));
                }

                Params[p] = ProtoCTF.GetCopy();
            }

            return Params;
        }

        public float2[] GetAstigmatism(float3[] coords)
        {
            float3[] GridCoords = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
                GridCoords[i] = new float3(coords[i].X / ImageDimensionsPhysical.X, coords[i].Y / ImageDimensionsPhysical.Y, 0);

            float[] ValuesDelta = GridCTFDefocusDelta.GetInterpolated(GridCoords);
            float[] ValuesAngle = GridCTFDefocusAngle.GetInterpolated(GridCoords);

            return Helper.Zip(ValuesDelta, ValuesAngle);
        }

        public float[] GetCs(float3[] coords)
        {
            float3[] GridCoords = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
                GridCoords[i] = new float3(coords[i].X / ImageDimensionsPhysical.X, coords[i].Y / ImageDimensionsPhysical.Y, 0);

            float[] Values = GridCTFCs.GetInterpolated(GridCoords);
            for (int i = 0; i < Values.Length; i++)
                Values[i] += (float)CTF.Cs;

            return Values;
        }

        public Image GetMotionEnvelope(int size, float pixelSize, float2 position)
        {
            Image Result = new Image(new int3(size, size, NFrames), true);
            float[][] ResultData = Result.GetHost(Intent.Write);

            int Oversample = 20;
            float Step = 1f / (NFrames - 1);
            float StepOversampled = Step / (Oversample - 1);

            float2[] AllShifts = GetPositionInAllFrames(new float3(position)).Select(v => new float2(v.X, v.Y) / pixelSize).ToArray();
            Cubic1D SplineX = new Cubic1D(AllShifts.Select((v, i) => new float2((float)i / (NFrames - 1), v.X)).ToArray());
            Cubic1D SplineY = new Cubic1D(AllShifts.Select((v, i) => new float2((float)i / (NFrames - 1), v.Y)).ToArray());

            for (int f = 0; f < NFrames; f++)
            {
                float[] InterpPoints = Helper.ArrayOfFunction(i => (f - 0.5f) * Step + i * StepOversampled, Oversample);
                float3[] FrameTrack = Helper.Zip(SplineX.Interp(InterpPoints), SplineY.Interp(InterpPoints), new float[Oversample]);

                Image Point = new Image(IntPtr.Zero, new int3(size, size, Oversample), true, true);
                Point.Fill(new float2(1, 0));
                Point.ShiftSlices(FrameTrack);

                Image PointFlat = Point.AsReducedAlongZ();
                Point.Dispose();

                Image Amps = PointFlat.AsAmplitudes();
                PointFlat.Dispose();

                ResultData[f] = Amps.GetHost(Intent.Read)[0];
                Amps.Dispose();
            }

            return Result;
        }

        #endregion

        static float[][][] _RawLayers = null;
        static float[][][] RawLayers
        {
            get
            {
                if (_RawLayers == null)
                    _RawLayers = new float[GPU.GetDeviceCount()][][];
                return _RawLayers;
            }
        }
        public void LoadFrameData(ProcessingOptionsBase options, Image imageGain, DefectModel defectMap, out Image[] frameData, int redNFrames = -1)
        {
            HeaderEER.GroupNFrames = options.EERGroupFrames;

            MapHeader Header = MapHeader.ReadFromFile(DataPath, new int2(1), 0, typeof(float));

            string Extension = Helper.PathToExtension(DataPath).ToLower();
            bool IsTiff = Header.GetType() == typeof(HeaderTiff);
            bool IsEER = Header.GetType() == typeof(HeaderEER);

            if (imageGain != null)
                if (!IsEER)
                    if (Header.Dimensions.X != imageGain.Dims.X || Header.Dimensions.Y != imageGain.Dims.Y)
                        throw new Exception("Gain reference dimensions do not match image.");

            int EERSupersample = 1;
            if (imageGain != null && IsEER)
            {
                if (Header.Dimensions.X == imageGain.Dims.X)
                    EERSupersample = 1;
                else if (Header.Dimensions.X * 2 == imageGain.Dims.X)
                    EERSupersample = 2;
                else if (Header.Dimensions.X * 4 == imageGain.Dims.X)
                    EERSupersample = 3;
                else
                    throw new Exception("Invalid supersampling factor requested for EER based on gain reference dimensions");
            }
            int EERGroupFrames = 1;
            if (IsEER)
            {
                if (HeaderEER.GroupNFrames > 0)
                    EERGroupFrames = HeaderEER.GroupNFrames;
                else if (HeaderEER.GroupNFrames < 0)
                {
                    int NFrames = -HeaderEER.GroupNFrames;
                    EERGroupFrames = Header.Dimensions.Z / NFrames;
                }

                Header.Dimensions.Z /= EERGroupFrames;
            }

            HeaderEER.SuperResolution = EERSupersample;

            if (IsEER && imageGain != null)
            {
                Header.Dimensions.X = imageGain.Dims.X;
                Header.Dimensions.Y = imageGain.Dims.Y;
            }

            int NThreads = (IsTiff || IsEER) ? 6 : 2;
            int GPUThreads = 2;

            int CurrentDevice = GPU.GetDevice();

            if (RawLayers[CurrentDevice] == null ||
                RawLayers[CurrentDevice].Length != NThreads ||
                RawLayers[CurrentDevice][0].Length != Header.Dimensions.ElementsSlice())
                RawLayers[CurrentDevice] = Helper.ArrayOfFunction(i => new float[Header.Dimensions.ElementsSlice()], NThreads);

            Image[] GPULayers = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, Header.Dimensions.Slice()), GPUThreads);

            float ScaleFactor = 1 / (float)options.DownsampleFactor;

            int3 ScaledDims = new int3((int)Math.Round(Header.Dimensions.X * ScaleFactor) / 2 * 2,
                                       (int)Math.Round(Header.Dimensions.Y * ScaleFactor) / 2 * 2,
                                       Math.Min(redNFrames > 0 ? redNFrames : NFrames, Header.Dimensions.Z));

            Image[] FrameData = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, ScaledDims.Slice()), ScaledDims.Z);

            if (ScaleFactor == 1f)
            {
                object[] Locks = Helper.ArrayOfFunction(i => new object(), GPUThreads);

                Helper.ForCPU(0, FrameData.Length, NThreads, threadID => GPU.SetDevice(CurrentDevice), (z, threadID) =>
                {
                    if (IsTiff)
                        TiffNative.ReadTIFFPatient(50, 500, DataPath, z, true, RawLayers[CurrentDevice][threadID]);
                    else if (IsEER)
                        EERNative.ReadEERPatient(50, 500, DataPath, z * EERGroupFrames, Math.Min(((HeaderEER)Header).DimensionsUngrouped.Z, (z + 1) * EERGroupFrames), EERSupersample, RawLayers[CurrentDevice][threadID]);
                    else
                        IOHelper.ReadMapFloatPatient(50, 500,
                                                     DataPath,
                                                     new int2(1),
                                                     0,
                                                     typeof(float),
                                                     new[] { z },
                                                     null,
                                                     new[] { RawLayers[CurrentDevice][threadID] });

                    int GPUThreadID = threadID % GPUThreads;

                    lock (Locks[GPUThreadID])
                    {
                        GPU.CopyHostToDevice(RawLayers[CurrentDevice][threadID], GPULayers[GPUThreadID].GetDevice(Intent.Write), RawLayers[CurrentDevice][threadID].Length);

                        if (imageGain != null)
                        {
                            if (IsEER)
                                GPULayers[GPUThreadID].DivideSlices(imageGain);
                            else
                                GPULayers[GPUThreadID].MultiplySlices(imageGain);
                        }

                        GPU.Xray(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                 FrameData[z].GetDevice(Intent.Write),
                                 20f,
                                 new int2(Header.Dimensions),
                                 1);
                    }

                }, null);
            }
            else
            {
                int[] PlanForw = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(Header.Dimensions.Slice(), 1), GPUThreads);
                int[] PlanBack = Helper.ArrayOfFunction(i => GPU.CreateIFFTPlan(ScaledDims.Slice(), 1), GPUThreads);

                Image[] GPULayers2 = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, Header.Dimensions.Slice()), GPUThreads);

                Image[] GPULayersInputFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, Header.Dimensions.Slice(), true, true), GPUThreads);
                Image[] GPULayersOutputFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, ScaledDims.Slice(), true, true), GPUThreads);

                object[] Locks = Helper.ArrayOfFunction(i => new object(), GPUThreads);

                Helper.ForCPU(0, FrameData.Length, NThreads, threadID => GPU.SetDevice(CurrentDevice), (z, threadID) =>
                {
                    if (IsTiff)
                        TiffNative.ReadTIFFPatient(50, 500, Path, z, true, RawLayers[CurrentDevice][threadID]);
                    else if (IsEER)
                        EERNative.ReadEERPatient(50, 500, Path, z * EERGroupFrames, Math.Min(((HeaderEER)Header).DimensionsUngrouped.Z, (z + 1) * EERGroupFrames), EERSupersample, RawLayers[CurrentDevice][threadID]);
                    else
                        IOHelper.ReadMapFloatPatient(50, 500,
                                                     Path,
                                                     new int2(1),
                                                     0,
                                                     typeof(float),
                                                     new[] { z },
                                                     null,
                                                     new[] { RawLayers[CurrentDevice][threadID] });

                    int GPUThreadID = threadID % GPUThreads;

                    lock (Locks[GPUThreadID])
                    {
                        GPU.CopyHostToDevice(RawLayers[CurrentDevice][threadID], GPULayers[GPUThreadID].GetDevice(Intent.Write), RawLayers[CurrentDevice][threadID].Length);

                        if (imageGain != null)
                        {
                            if (IsEER)
                                GPULayers[GPUThreadID].DivideSlices(imageGain);
                            else
                                GPULayers[GPUThreadID].MultiplySlices(imageGain);
                        }

                        GPU.Xray(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                 GPULayers2[GPUThreadID].GetDevice(Intent.Write),
                                 20f,
                                 new int2(Header.Dimensions),
                                 1);

                        GPU.Scale(GPULayers2[GPUThreadID].GetDevice(Intent.Read),
                                  FrameData[z].GetDevice(Intent.Write),
                                  Header.Dimensions.Slice(),
                                  ScaledDims.Slice(),
                                  1,
                                  PlanForw[GPUThreadID],
                                  PlanBack[GPUThreadID],
                                  GPULayersInputFT[GPUThreadID].GetDevice(Intent.Write),
                                  GPULayersOutputFT[GPUThreadID].GetDevice(Intent.Write));
                    }

                }, null);

                for (int i = 0; i < GPUThreads; i++)
                {
                    GPU.DestroyFFTPlan(PlanForw[i]);
                    GPU.DestroyFFTPlan(PlanBack[i]);
                    GPULayersInputFT[i].Dispose();
                    GPULayersOutputFT[i].Dispose();
                    GPULayers2[i].Dispose();
                }
            }

            foreach (var item in GPULayers)
                item.Dispose();

            frameData = FrameData;
        }

        public void CreateThumbnail(int size, float stddevRange)
        {
            if (!File.Exists(AveragePath))
                return;

            Directory.CreateDirectory(ThumbnailsDir);

            Image Average = Image.FromFile(AveragePath);
            float ScaleFactor = (float)size / Math.Max(Average.Dims.X, Average.Dims.Y);
            int2 DimsScaled = new int2(new float2(Average.Dims.X, Average.Dims.Y) * ScaleFactor + 1) / 2 * 2;

            Image AverageScaled = Average.AsScaled(DimsScaled).AndDisposeParent();
            Image AverageCenter = AverageScaled.AsPadded(DimsScaled / 4 * 2);

            float2 MeanStd = MathHelper.MedianAndStd(AverageCenter.GetHost(Intent.Read)[0]);
            float Min = MeanStd.X;
            float Range = 0.5f / (MeanStd.Y * stddevRange);

            AverageCenter.Dispose();

            AverageScaled.TransformValues(v => ((v - Min) * Range + 0.5f) * 255);

            AverageScaled.WritePNG(ThumbnailsPath);
            AverageScaled.Dispose();
        }

        public float2[] GetMotionTrack(float2 position, int oversampleFactorAlongZ = 1, bool localOnly = false)
        {
            if (OptionsMovement == null || OptionsMovement.Dimensions.Z <= 1)
                return null;

            int NFrames = (int)OptionsMovement.Dimensions.Z;
            float2[] Result = new float2[NFrames * oversampleFactorAlongZ];

            float StepZ = 1f / Math.Max(NFrames * oversampleFactorAlongZ - 1, 1);
            for (int z = 0; z < NFrames * oversampleFactorAlongZ; z++)
                Result[z] = GetShiftFromPyramid(new float3(position.X, position.Y, z * StepZ), localOnly);

            return Result;
        }

        public float2 GetShiftFromPyramid(float3 coords, bool localOnly = false)
        {
            float2 Result = new float2(0, 0);

            Result.X = localOnly ? 0 : GridMovementX.GetInterpolated(coords);
            Result.Y = localOnly ? 0 : GridMovementY.GetInterpolated(coords);

            Result.X += GridLocalX.GetInterpolated(coords);
            Result.Y += GridLocalY.GetInterpolated(coords);

            for (int i = 0; i < PyramidShiftX.Count; i++)
            {
                Result.X += PyramidShiftX[i].GetInterpolated(coords);
                Result.Y += PyramidShiftY[i].GetInterpolated(coords);
            }

            return Result;
        }

        public float2[] GetShiftFromPyramid(float3[] coords, bool localOnly = false)
        {
            float2[] Result = new float2[coords.Length];

            if (!localOnly)
            {
                float[] X = GridMovementX.GetInterpolated(coords);
                float[] Y = GridMovementY.GetInterpolated(coords);
                for (int i = 0; i < Result.Length; i++)
                {
                    Result[i].X += X[i];
                    Result[i].Y += Y[i];
                }
            }

            {
                float[] X = GridLocalX.GetInterpolated(coords);
                float[] Y = GridLocalY.GetInterpolated(coords);
                for (int i = 0; i < Result.Length; i++)
                {
                    Result[i].X += X[i];
                    Result[i].Y += Y[i];
                }
            }

            for (int p = 0; p < PyramidShiftX.Count; p++)
            {
                float[] X = PyramidShiftX[p].GetInterpolated(coords);
                float[] Y = PyramidShiftY[p].GetInterpolated(coords);
                for (int i = 0; i < Result.Length; i++)
                {
                    Result[i].X += X[i];
                    Result[i].Y += Y[i];
                }
            }

            return Result;
        }

        public virtual int[] GetRelevantImageSizes(int fullSize, float weightingThreshold)
        {
            int[] Result = new int[NFrames];

            float[][] AllWeights = new float[NFrames][];

            float GridStep = 1f / (NFrames - 1);
            for (int f = 0; f < NFrames; f++)
            {
                CTF CurrCTF = CTF.GetCopy();

                CurrCTF.Defocus = 0;
                CurrCTF.DefocusDelta = 0;
                CurrCTF.Cs = 0;
                CurrCTF.Amplitude = 1;

                if (GridDoseBfacs.Dimensions.Elements() <= 1)
                    CurrCTF.Bfactor = (decimal)(-f * (float)OptionsMovieExport.DosePerAngstromFrame * 4);
                else
                    CurrCTF.Bfactor = (decimal)(GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, f * GridStep)) +
                                                Math.Abs(GridDoseBfacsDelta.GetInterpolated(new float3(0.5f, 0.5f, f * GridStep))));

                AllWeights[f] = CurrCTF.Get1D(fullSize / 2, false);
            }

            int elementID = 0;
            if (GridDoseBfacs.Dimensions.Elements() > 1)
                (elementID, _) = MathHelper.MaxElement(GridDoseBfacs.FlatValues);
            float[] LowerDoseWeights = AllWeights[elementID].ToList().ToArray();

            for (int t = 0; t < NFrames; t++)
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

        public virtual JsonNode ToMiniJson(string particleSuffix = null)
        {
            JsonNode Json = new JsonObject();

            // Path relative to processing folder, i.e. just the file name
            // Full path to data (including if it's in a nested folder) is stored in XML metadata
            Json["Path"] = Helper.PathToNameWithExtension(Path);

            // ProcessingStatus enum
            Json["Stat"] = (int)ProcessingStatus;

            // CTF
            {
                // Defocus
                Json["Def"] = CTF == null ? null : MathF.Round((float)CTF.Defocus, 4);

                // Phase shift
                Json["Phs"] = CTF == null ? null : MathF.Round((float)CTF.PhaseShift, 2);

                // Estimated resolution
                Json["Rsn"] = CTFResolutionEstimate <= 0 ? null : MathF.Round((float)CTFResolutionEstimate, 2);

                // Astigmatism plot X and Y
                Json["AsX"] = CTF == null ? null : MathF.Round(MathF.Cos((float)CTF.DefocusAngle * 2 * Helper.ToRad) * (float)CTF.DefocusDelta, 4);
                Json["AsY"] = CTF == null ? null : MathF.Round(MathF.Sin((float)CTF.DefocusAngle * 2 * Helper.ToRad) * (float)CTF.DefocusDelta, 4);
            }

            // Motion
            {
                Json["Mtn"] = OptionsMovement == null ? null : (double)MeanFrameMovement;
            }

            // 💩 percentage
            Json["Jnk"] = MaskPercentage < 0 ? null : MathF.Round((float)MaskPercentage, 1);

            // Particle count for given suffix
            if (particleSuffix != null)
            {
                int ParticleCount = GetParticleCount(particleSuffix);
                Json["Ptc"] = ParticleCount < 0 ? null : ParticleCount;
            }

            return Json;
        }

        #endregion
    }

    public enum WarpOptimizationTypes
    {
        ImageWarp = 1 << 0,
        VolumeWarp = 1 << 1,
        AxisAngle = 1 << 2,
        ParticlePosition = 1 << 3,
        ParticleAngle = 1 << 4,
        Magnification = 1 << 5,
        Zernike13 = 1 << 6,
        Zernike5 = 1 << 7,
        ParticleMag = 1 << 8
    }

    public enum CTFOptimizationTypes
    {
        Defocus = 1 << 0,
        AstigmatismDelta = 1 << 1,
        AstigmatismAngle = 1 << 2,
        PhaseShift = 1 << 3,
        Cs = 1 << 4,
        Doming = 1 << 5,
        Zernike2 = 1 << 6,
        Zernike4 = 1 << 7,
        DefocusGridSearch = 1 << 8,
        Distortion = 1 << 9
    }
    
    [Serializable]
    public class ProcessingOptionsTardisSegmentMembranes2D : ProcessingOptionsBase
    {
    }
}
