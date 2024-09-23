using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Accord.Math.Optimization;
using Warp.Headers;
using Warp.Tools;
using Accord;
using Warp.Sociology;
using IOPath = System.IO.Path;
using System.Text.Json.Nodes;
using System.Threading;

namespace Warp
{
    public class Movie : WarpBase
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

        public string PowerSpectrumDir => IOPath.Combine(ProcessingDirectoryName, "powerspectrum");
        public string MotionTrackDir => IOPath.Combine(ProcessingDirectoryName, "motion");
        public string AverageDir => IOPath.Combine(ProcessingDirectoryName, "average");
        public string AverageOddDir => IOPath.Combine(AverageDir, "odd");
        public string AverageEvenDir => IOPath.Combine(AverageDir, "even");
        public string AverageDenoisedDir => IOPath.Combine(AverageDir, "denoised");
        public string DeconvolvedDir => IOPath.Combine(ProcessingDirectoryName, "deconv");
        public string DenoiseTrainingDir => IOPath.Combine(ProcessingDirectoryName, "denoising");
        public string DenoiseTrainingDirOdd => IOPath.Combine(DenoiseTrainingDir, "odd");
        public string DenoiseTrainingDirEven => IOPath.Combine(DenoiseTrainingDir, "even");
        public string DenoiseTrainingDirCTF => IOPath.Combine(DenoiseTrainingDir, "ctf");
        public string DenoiseTrainingDirModel => IOPath.Combine(DenoiseTrainingDir, "model.pt");
        public string ShiftedStackDir => IOPath.Combine(ProcessingDirectoryName, "stack");
        public string MaskDir => IOPath.Combine(ProcessingDirectoryName, "mask");
        public string ParticlesDir => IOPath.Combine(ProcessingDirectoryName, "particles");
        public string ParticlesDenoisingOddDir => IOPath.Combine(ProcessingDirectoryName, "particles", "odd");
        public string ParticlesDenoisingEvenDir => IOPath.Combine(ProcessingDirectoryName, "particles", "even");
        public string ParticleTiltsDir => IOPath.Combine(ProcessingDirectoryName, "particletilts");
        public string ParticleCTFDir => IOPath.Combine(ProcessingDirectoryName, "particlectf");
        public string ParticleTiltsCTFDir => IOPath.Combine(ProcessingDirectoryName, "particletiltsctf");
        public string ParticleMoviesCTFDir => IOPath.Combine(ProcessingDirectoryName, "particlemoviesctf");
        public string MatchingDir => IOPath.Combine(ProcessingDirectoryName, "matching");
        public string ThumbnailsDir => IOPath.Combine(ProcessingDirectoryName, "thumbnails");

        public string PowerSpectrumPath => IOPath.Combine(PowerSpectrumDir, RootName + ".mrc");
        public string AveragePath => IOPath.Combine(AverageDir, RootName + ".mrc");
        public string AverageOddPath => IOPath.Combine(AverageOddDir, RootName + ".mrc");
        public string AverageEvenPath => IOPath.Combine(AverageEvenDir, RootName + ".mrc");
        public string AverageDenoisedPath => IOPath.Combine(AverageDenoisedDir, RootName + ".mrc");
        public string DeconvolvedPath => IOPath.Combine(DeconvolvedDir, RootName + ".mrc");
        public string DenoiseTrainingOddPath => IOPath.Combine(DenoiseTrainingDirOdd, RootName + ".mrc");
        public string DenoiseTrainingEvenPath => IOPath.Combine(DenoiseTrainingDirEven, RootName + ".mrc");
        public string DenoiseTrainingCTFPath => IOPath.Combine(DenoiseTrainingDirCTF, RootName + ".mrc");
        public string ShiftedStackPath => IOPath.Combine(ShiftedStackDir, RootName + "_movie.mrcs");
        public string MaskPath => IOPath.Combine(MaskDir, RootName + ".tif");
        public string ParticlesPath => IOPath.Combine(ParticlesDir, RootName + "_particles.mrcs");
        public string ParticleCTFPath => IOPath.Combine(ParticleCTFDir, RootName + "_particlesctf.mrcs");
        public string ParticleMoviesCTFPath => IOPath.Combine(ParticleMoviesCTFDir, RootName + "_particlemoviesctf.mrcs");
        public string ThumbnailsPath => IOPath.Combine(ThumbnailsDir, RootName + ".png");

        public string XMLName => RootName + ".xml";
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

        public virtual void ProcessCTF(Image originalStack, ProcessingOptionsMovieCTF options)
        {
            IsProcessing = true;

            if (!Directory.Exists(PowerSpectrumDir))
                Directory.CreateDirectory(PowerSpectrumDir);

            //CTF = new CTF();
            PS1D = null;
            _SimulatedBackground = null;
            _SimulatedScale = new Cubic1D(new[] { new float2(0, 1), new float2(1, 1) });

            #region Dimensions and grids

            int NFrames = options.UseMovieSum ? 1 : originalStack.Dims.Z;
            int2 DimsImage = new int2(originalStack.Dims);
            int2 DimsRegion = new int2(options.Window);
            int2 DimsRegionLarge = DimsRegion * 2;

            float OverlapFraction = 0.5f;
            int2 DimsPositionGrid;
            int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, DimsRegionLarge, OverlapFraction, out DimsPositionGrid);
            int NPositions = (int)DimsPositionGrid.Elements();

            // Auto grid dims
            if (options.GridDims.Elements() == 0)
            {
                float OverallDose = (float)(options.DosePerAngstromFrame < 0 ?
                                            -options.DosePerAngstromFrame :
                                            options.DosePerAngstromFrame * NFrames);

                int AutoZ = (int)MathF.Ceiling(Math.Max(1, OverallDose));
                int AutoX, AutoY;

                // For a FoV of 4000 Angstrom, use a grid side length of 5
                // Scale up linearly for larger FoV
                // Scale down linearly for lower dose than 30 e/A^2
                // At least 2x2 because we want a gradient even for low-dose tilt movies
                float ShortAngstrom = Math.Min(originalStack.Dims.X, originalStack.Dims.Y) * (float)options.BinnedPixelSizeMean;
                int ShortGrid = (int)Math.Max(2, MathF.Round(5f * Math.Min(1, OverallDose / 30f) * (ShortAngstrom / 4000f)));

                if (originalStack.Dims.X <= originalStack.Dims.Y)
                {
                    AutoX = ShortGrid;
                    AutoY = (int)MathF.Round(ShortGrid * (float)originalStack.Dims.Y / originalStack.Dims.X);
                }
                else
                {
                    AutoY = ShortGrid;
                    AutoX = (int)MathF.Round(ShortGrid * (float)originalStack.Dims.X / originalStack.Dims.Y);
                }

                options.GridDims = new int3(AutoX, AutoY, AutoZ);
            }

            int CTFGridX = Math.Min(DimsPositionGrid.X, options.GridDims.X);
            int CTFGridY = Math.Min(DimsPositionGrid.Y, options.GridDims.Y);
            int CTFGridZ = Math.Min(NFrames, options.GridDims.Z);
            GridCTFDefocus = new CubicGrid(new int3(CTFGridX, CTFGridY, CTFGridZ));
            GridCTFPhase = new CubicGrid(new int3(1, 1, CTFGridZ));

            bool CTFSpace = CTFGridX * CTFGridY > 1;
            bool CTFTime = CTFGridZ > 1;
            int3 CTFSpectraGrid = new int3(CTFSpace ? DimsPositionGrid.X : 1,
                                           CTFSpace ? DimsPositionGrid.Y : 1,
                                           CTFTime ? CTFGridZ : 1);

            int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
            int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
            int NFreq = MaxFreqExclusive - MinFreqInclusive;

            if (MaxFreqExclusive > DimsRegion.X / 2)
                throw new Exception("Max frequency to fit is higher than the Nyquist frequency");

            #endregion

            var Timer0 = CTFTimers[0].Start();
            #region Allocate GPU memory

            Image CTFSpectra = new Image(IntPtr.Zero, new int3(DimsRegion.X, DimsRegion.X, (int)CTFSpectraGrid.Elements()), true);
            Image CTFMean = new Image(IntPtr.Zero, new int3(DimsRegion), true);
            Image CTFCoordsCart = new Image(new int3(DimsRegion), true, true);
            Image CTFCoordsPolarTrimmed = new Image(new int3(NFreq, DimsRegion.X, 1), false, true);

            #endregion
            CTFTimers[0].Finish(Timer0);

            // Extract movie regions, create individual spectra in Cartesian coordinates and their mean.

            var Timer1 = CTFTimers[1].Start();
            #region Create spectra

            if (options.UseMovieSum)
            {
                Image StackAverage = null;
                if (!File.Exists(AveragePath))
                    StackAverage = originalStack.AsReducedAlongZ();
                else
                    StackAverage = Image.FromFile(AveragePath);

                originalStack?.FreeDevice();

                GPU.CreateSpectra(StackAverage.GetDevice(Intent.Read),
                                  DimsImage,
                                  1,
                                  PositionGrid,
                                  NPositions,
                                  DimsRegionLarge,
                                  CTFSpectraGrid,
                                  DimsRegion,
                                  CTFSpectra.GetDevice(Intent.Write),
                                  CTFMean.GetDevice(Intent.Write),
                                  0,
                                  0);

                StackAverage.Dispose();
            }
            else
            {
                GPU.CreateSpectra(originalStack.GetDevice(Intent.Read),
                                  DimsImage,
                                  NFrames,
                                  PositionGrid,
                                  NPositions,
                                  DimsRegionLarge,
                                  CTFSpectraGrid,
                                  DimsRegion,
                                  CTFSpectra.GetDevice(Intent.Write),
                                  CTFMean.GetDevice(Intent.Write),
                                  0,
                                  0);

                originalStack.FreeDevice(); // Won't need it in this method anymore.
            }

            //CTFSpectra.WriteMRC("d_spectra.mrc", true);
            #endregion
            CTFTimers[1].Finish(Timer1);

            // Populate address arrays for later.

            var Timer2 = CTFTimers[2].Start();
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
            CTFTimers[2].Finish(Timer2);

            // Retrieve average 1D spectrum from CTFMean (not corrected for astigmatism yet).

            var Timer3 = CTFTimers[3].Start();
            #region Initial 1D spectrum

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

                //CTFAverage1D.WriteMRC("d_CTFAverage1D.mrc");

                float[] CTFAverage1DData = CTFAverage1D.GetHost(Intent.Read)[0];
                float2[] ForPS1D = new float2[DimsRegion.X / 2];
                for (int i = 0; i < ForPS1D.Length; i++)
                    ForPS1D[i] = new float2((float)i / DimsRegion.X, (float)Math.Round(CTFAverage1DData[i], 4));
                _PS1D = ForPS1D;

                CTFAverage1D.Dispose();
            }

            #endregion
            CTFTimers[3].Finish(Timer3);

            var Timer4 = CTFTimers[4].Start();
            #region Background fitting methods

            Action UpdateBackgroundFit = () =>
            {
                float2[] ForPS1D = PS1D.Skip(Math.Max(5, MinFreqInclusive / 2)).ToArray();
                Cubic1D.FitCTF(ForPS1D,
                               CTF.Get1D(PS1D.Length, true, true).Skip(Math.Max(5, MinFreqInclusive / 2)).ToArray(),
                               CTF.GetZeros(),
                               CTF.GetPeaks(),
                               out _SimulatedBackground,
                               out _SimulatedScale);
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
                                            MeanCorrectedData[i] = MeanData[i] - _SimulatedBackground.Interp(r / DimsRegion.X);
                                        });

                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                GPU.CTFMakeAverage(CTFMeanCorrected.GetDevice(Intent.Read),
                                   CTFCoordsCart.GetDevice(Intent.Read),
                                   (uint)CTFMeanCorrected.DimsEffective.ElementsSlice(),
                                   (uint)DimsRegion.X,
                                   new[] { CTF.ToStruct() },
                                   CTF.ToStruct(),
                                   0,
                                   (uint)DimsRegion.X / 2,
                                   1,
                                   CTFAverage1D.GetDevice(Intent.Write));

                //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

                float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                float2[] ForPS1D = new float2[PS1D.Length];
                if (keepbackground)
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i] + _SimulatedBackground.Interp((float)i / DimsRegion.X));
                else
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                MathHelper.UnNaN(ForPS1D);

                _PS1D = ForPS1D;

                CTFMeanCorrected.Dispose();
                CTFAverage1D.Dispose();
            };

            #endregion

            // Fit background to currently best average (not corrected for astigmatism yet).
            {
                float2[] ForPS1D = PS1D.Skip(MinFreqInclusive).Take(Math.Max(2, NFreq)).ToArray();

                int NumNodes = Math.Max(3, (int)((options.RangeMax - options.RangeMin) * 5M));
                _SimulatedBackground = Cubic1D.Fit(ForPS1D, NumNodes); // This won't fit falloff and scale, because approx function is 0

                float[] CurrentBackground = _SimulatedBackground.Interp(PS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray();
                float[] Subtracted1D = Helper.ArrayOfFunction(i => ForPS1D[i].Y - CurrentBackground[i], ForPS1D.Length);
                MathHelper.NormalizeInPlace(Subtracted1D);

                float ZMin = (float)options.ZMin;
                float ZMax = (float)options.ZMax;
                float ZStep = (ZMax - ZMin) / 200f;

                float BestZ = 0, BestIceOffset = 0, BestPhase = 0, BestScore = -float.MaxValue;

                int NThreads = 8;
                float[] ZValues = Helper.ArrayOfFunction(i => i * 0.01f + ZMin, (int)((ZMax + 1e-5f - ZMin) / 0.01f));
                float[] MTBestZ = new float[NThreads];
                float[] MTBestPhase = new float[NThreads];
                float[] MTBestScore = Helper.ArrayOfConstant(-float.MaxValue, NThreads);

                Helper.ForCPU(0, ZValues.Length, NThreads, null, (i, threadID) =>
                {
                    float z = ZValues[i];

                    for (float p = 0; p <= (options.DoPhase ? 1f : 0f); p += 0.01f)
                    {
                        CTF CurrentParams = new CTF
                        {
                            PixelSize = options.BinnedPixelSizeMean,

                            Defocus = (decimal)z,
                            PhaseShift = (decimal)p,

                            Cs = options.Cs,
                            Voltage = options.Voltage,
                            Amplitude = options.Amplitude
                        };

                        float[] SimulatedCTF = Helper.Subset(CurrentParams.Get1D(PS1D.Length, true), MinFreqInclusive, MinFreqInclusive + NFreq);

                        MathHelper.NormalizeInPlace(SimulatedCTF);
                        float Score = MathHelper.CrossCorrelate(Subtracted1D, SimulatedCTF);

                        if (Score > MTBestScore[threadID])
                        {
                            MTBestScore[threadID] = Score;
                            MTBestZ[threadID] = z;
                            MTBestPhase[threadID] = p;
                        }
                    }
                }, null);

                for (int i = 0; i < NThreads; i++)
                {
                    if (MTBestScore[i] > BestScore)
                    {
                        BestScore = MTBestScore[i];
                        BestZ = MTBestZ[i];
                        BestPhase = MTBestPhase[i];
                    }
                }

                CTF = new CTF
                {
                    PixelSize = options.BinnedPixelSizeMean,

                    Defocus = (decimal)BestZ,
                    PhaseShift = (decimal)BestPhase,

                    Cs = options.Cs,
                    Voltage = options.Voltage,
                    Amplitude = options.Amplitude
                };

                UpdateRotationalAverage(true); // This doesn't have a nice background yet.
                UpdateBackgroundFit(); // Now get a reasonably nice background.
            }
            CTFTimers[4].Finish(Timer4);

            // Do BFGS optimization of defocus, astigmatism and phase shift,
            // using 2D simulation for comparison

            var Timer5 = CTFTimers[5].Start();
            #region BFGS

            GridCTFDefocus = new CubicGrid(GridCTFDefocus.Dimensions, (float)CTF.Defocus, (float)CTF.Defocus, Dimension.X);
            GridCTFPhase = new CubicGrid(GridCTFPhase.Dimensions, (float)CTF.PhaseShift, (float)CTF.PhaseShift, Dimension.X);

            {
                NFreq = MaxFreqExclusive - MinFreqInclusive;

                Image CTFSpectraPolarTrimmed = CTFSpectra.AsPolar((uint)MinFreqInclusive, (uint)(MinFreqInclusive + NFreq));
                CTFSpectra.FreeDevice(); // This will only be needed again for the final PS1D.

                #region Create background and scale

                float[] CurrentScale = _SimulatedScale.Interp(PS1D.Select(p => p.X).ToArray());

                Image CTFSpectraScale = new Image(new int3(NFreq, DimsRegion.X, 1));
                float[] CTFSpectraScaleData = CTFSpectraScale.GetHost(Intent.Write)[0];

                // Trim polar to relevant frequencies, and populate coordinates.
                Parallel.For(0, DimsRegion.X, y =>
                {
                    float Angle = ((float)y / DimsRegion.X + 0.5f) * (float)Math.PI;
                    for (int x = 0; x < NFreq; x++)
                        CTFSpectraScaleData[y * NFreq + x] = CurrentScale[x + MinFreqInclusive];
                });
                //CTFSpectraScale.WriteMRC("ctfspectrascale.mrc");

                // Background is just 1 line since we're in polar.
                Image CurrentBackground = new Image(_SimulatedBackground.Interp(PS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray());

                #endregion

                CTFSpectraPolarTrimmed.SubtractFromLines(CurrentBackground);
                CurrentBackground.Dispose();

                // Normalize background-subtracted spectra.
                GPU.Normalize(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                              CTFSpectraPolarTrimmed.GetDevice(Intent.Write),
                              (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                              (uint)CTFSpectraGrid.Elements());
                //CTFSpectraPolarTrimmed.WriteMRC("ctfspectrapolartrimmed.mrc");

                // Wiggle weights show how the defocus on the spectra grid is altered 
                // by changes in individual anchor points of the spline grid.
                // They are used later to compute the dScore/dDefocus values for each spectrum 
                // only once, and derive the values for each anchor point from them.
                float[][] WiggleWeights = GridCTFDefocus.GetWiggleWeights(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 1f / (CTFGridZ + 1)));
                float[][] WiggleWeightsPhase = GridCTFPhase.GetWiggleWeights(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 1f / (CTFGridZ + 1)));

                // Helper method for getting CTFStructs for the entire spectra grid.
                Func<double[], CTF, float[], float[], CTFStruct[]> EvalGetCTF = (input, ctf, defocusValues, phaseValues) =>
                {
                    decimal AlteredDelta = (decimal)input[input.Length - 2];
                    decimal AlteredAngle = (decimal)(input[input.Length - 1] * 20 * Helper.ToDeg);

                    CTF Local = ctf.GetCopy();
                    Local.DefocusDelta = AlteredDelta;
                    Local.DefocusAngle = AlteredAngle;

                    CTFStruct LocalStruct = Local.ToStruct();
                    CTFStruct[] LocalParams = new CTFStruct[defocusValues.Length];
                    for (int i = 0; i < LocalParams.Length; i++)
                    {
                        LocalParams[i] = LocalStruct;
                        LocalParams[i].Defocus = defocusValues[i] * -1e-6f;
                        LocalParams[i].PhaseShift = phaseValues[i] * (float)Math.PI;
                    }

                    return LocalParams;
                };

                // Simulate with adjusted CTF, compare to originals

                #region Eval and Gradient methods

                float BorderZ = 0.5f / CTFGridZ;

                Func<double[], double> Eval = input =>
                {
                    CubicGrid Altered = new CubicGrid(GridCTFDefocus.Dimensions, input.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v).ToArray());
                    float[] DefocusValues = Altered.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                    CubicGrid AlteredPhase = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v).ToArray());
                    float[] PhaseValues = AlteredPhase.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                    CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                    float IceIntensity = 1 / (1 + (float)Math.Exp(-input[input.Length - 3] * 10));
                    float2 IceStd = new float2((float)Math.Exp(input[input.Length - 5] * 10), (float)Math.Exp(input[input.Length - 4] * 10));
                    float IceOffset = (float)input[input.Length - 6] * (-1e4f);

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
                    const float Step = 0.005f;
                    double[] Result = new double[input.Length];

                    // In 0D grid case, just get gradient for all 4 parameters.
                    // In 1+D grid case, do simple gradient for ice ring, astigmatism, phase, ...
                    int StartComponent = input.Length - 2;
                    //int StartComponent = 0;
                    for (int i = StartComponent; i < input.Length; i++)
                    {
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

                    float IceIntensity = 1 / (1 + (float)Math.Exp(-input[input.Length - 3] * 10));
                    float2 IceStd = new float2((float)Math.Exp(input[input.Length - 5] * 10), (float)Math.Exp(input[input.Length - 4] * 10));
                    float IceOffset = (float)input[input.Length - 6] * (-1e4f);

                    float[] ResultPlus = new float[CTFSpectraGrid.Elements()];
                    float[] ResultMinus = new float[CTFSpectraGrid.Elements()];

                    // ... take shortcut for defoci, ...
                    {
                        CubicGrid AlteredPhase = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v).ToArray());
                        float[] PhaseValues = AlteredPhase.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                        {
                            CubicGrid AlteredPlus = new CubicGrid(GridCTFDefocus.Dimensions, input.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v + Step).ToArray());
                            float[] DefocusValues = AlteredPlus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                            CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                            GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                                CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                                CTFSpectraScale.GetDevice(Intent.Read),
                                                (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                                LocalParams,
                                                ResultPlus,
                                                (uint)LocalParams.Length);
                        }
                        {
                            CubicGrid AlteredMinus = new CubicGrid(GridCTFDefocus.Dimensions, input.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v - Step).ToArray());
                            float[] DefocusValues = AlteredMinus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                            CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                            GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                                CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                                CTFSpectraScale.GetDevice(Intent.Read),
                                                (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                                LocalParams,
                                                ResultMinus,
                                                (uint)LocalParams.Length);
                        }
                        float[] LocalGradients = new float[ResultPlus.Length];
                        for (int i = 0; i < LocalGradients.Length; i++)
                            LocalGradients[i] = ResultPlus[i] - ResultMinus[i];

                        // Now compute gradients per grid anchor point using the precomputed individual gradients and wiggle factors.
                        Parallel.For(0, GridCTFDefocus.Dimensions.Elements(), i => Result[i] = MathHelper.ReduceWeighted(LocalGradients, WiggleWeights[i]) / (2f * Step));
                    }

                    // ... and take shortcut for phases.
                    if (options.DoPhase)
                    {
                        CubicGrid AlteredPlus = new CubicGrid(GridCTFDefocus.Dimensions, input.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v).ToArray());
                        float[] DefocusValues = AlteredPlus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                        {
                            CubicGrid AlteredPhasePlus = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v + Step).ToArray());
                            float[] PhaseValues = AlteredPhasePlus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                            CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                            GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                                CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                                CTFSpectraScale.GetDevice(Intent.Read),
                                                (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                                LocalParams,
                                                ResultPlus,
                                                (uint)LocalParams.Length);
                        }
                        {
                            CubicGrid AlteredPhaseMinus = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v - Step).ToArray());
                            float[] PhaseValues = AlteredPhaseMinus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                            CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                            GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                                CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                                CTFSpectraScale.GetDevice(Intent.Read),
                                                (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                                LocalParams,
                                                ResultMinus,
                                                (uint)LocalParams.Length);
                        }
                        float[] LocalGradients = new float[ResultPlus.Length];
                        for (int i = 0; i < LocalGradients.Length; i++)
                            LocalGradients[i] = ResultPlus[i] - ResultMinus[i];

                        // Now compute gradients per grid anchor point using the precomputed individual gradients and wiggle factors.
                        Parallel.For(0, GridCTFPhase.Dimensions.Elements(), i => Result[i + GridCTFDefocus.Dimensions.Elements()] = MathHelper.ReduceWeighted(LocalGradients, WiggleWeightsPhase[i]) / (2f * Step));
                    }

                    foreach (var i in Result)
                        if (double.IsNaN(i) || double.IsInfinity(i))
                            throw new Exception("Bad score.");

                    return Result;
                };

                #endregion

                #region Optimize

                double[] StartParams = new double[GridCTFDefocus.Dimensions.Elements() + GridCTFPhase.Dimensions.Elements() + 6];
                for (int i = 0; i < GridCTFDefocus.Dimensions.Elements(); i++)
                    StartParams[i] = GridCTFDefocus.FlatValues[i];
                for (int i = 0; i < GridCTFPhase.Dimensions.Elements(); i++)
                    StartParams[i + GridCTFDefocus.Dimensions.Elements()] = GridCTFPhase.FlatValues[i];

                StartParams[StartParams.Length - 2] = (double)CTF.DefocusDelta;
                StartParams[StartParams.Length - 1] = (double)CTF.DefocusAngle / 20 * Helper.ToRad;

                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Gradient)
                {
                    MaxIterations = 15
                };
                Optimizer.Maximize(StartParams);

                #endregion

                #region Retrieve parameters

                CTF.Defocus = (decimal)MathHelper.Mean(Optimizer.Solution.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v));
                CTF.DefocusDelta = (decimal)Optimizer.Solution[StartParams.Length - 2];
                CTF.DefocusAngle = (decimal)(Optimizer.Solution[StartParams.Length - 1] * 20 * Helper.ToDeg);
                CTF.PhaseShift = (decimal)MathHelper.Mean(Optimizer.Solution.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v));

                if (CTF.DefocusDelta < 0)
                {
                    CTF.DefocusAngle += 90;
                    CTF.DefocusDelta *= -1;
                }
                CTF.DefocusAngle = ((int)CTF.DefocusAngle + 180 * 99) % 180;

                GridCTFDefocus = new CubicGrid(GridCTFDefocus.Dimensions, Optimizer.Solution.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v).ToArray());
                GridCTFPhase = new CubicGrid(GridCTFPhase.Dimensions, Optimizer.Solution.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v).ToArray());

                #endregion

                // Dispose GPU resources manually because GC can't be bothered to do it in time.
                CTFSpectraPolarTrimmed.Dispose();
                CTFSpectraScale.Dispose();

                #region Get nicer envelope fit

                {
                    if (!CTFSpace && !CTFTime)
                    {
                        UpdateRotationalAverage(true);
                    }
                    else
                    {
                        Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                        float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                        // Construct background in Cartesian coordinates.
                        Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) =>
                        {
                            CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = _SimulatedBackground.Interp(r / DimsRegion.X);
                        });

                        CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                        float[] DefocusValues = GridCTFDefocus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                        CTFStruct[] LocalParams = DefocusValues.Select(v =>
                        {
                            CTF Local = CTF.GetCopy();
                            Local.Defocus = (decimal)v + 0.0M;

                            return Local.ToStruct();
                        }).ToArray();

                        Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                        CTF CTFAug = CTF.GetCopy();
                        CTFAug.Defocus += 0.0M;
                        GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                           CTFCoordsCart.GetDevice(Intent.Read),
                                           (uint)CTFSpectra.ElementsSliceReal,
                                           (uint)DimsRegion.X,
                                           LocalParams,
                                           CTFAug.ToStruct(),
                                           0,
                                           (uint)DimsRegion.X / 2,
                                           (uint)CTFSpectraGrid.Elements(),
                                           CTFAverage1D.GetDevice(Intent.Write));

                        CTFSpectra.AddToSlices(CTFSpectraBackground);

                        float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                        float2[] ForPS1D = new float2[PS1D.Length];
                        for (int i = 0; i < ForPS1D.Length; i++)
                            ForPS1D[i] = new float2((float)i / DimsRegion.X, (float)Math.Round(RotationalAverageData[i], 4) + _SimulatedBackground.Interp((float)i / DimsRegion.X));
                        MathHelper.UnNaN(ForPS1D);
                        _PS1D = ForPS1D;

                        CTFSpectraBackground.Dispose();
                        CTFAverage1D.Dispose();
                        CTFSpectra.FreeDevice();
                    }

                    //CTF.Defocus = Math.Max(CTF.Defocus, 0);
                    UpdateBackgroundFit();
                }

                #endregion
            }

            #endregion
            CTFTimers[5].Finish(Timer5);

            // Subtract background from 2D average and write it to disk. 
            // This image is used for quick visualization purposes only.

            var Timer6 = CTFTimers[6].Start();
            #region PS2D update
            {
                int3 DimsAverage = new int3(DimsRegion.X, DimsRegion.X / 2, 1);
                float[] Average2DData = new float[DimsAverage.Elements()];
                float[] OriginalAverageData = CTFMean.GetHost(Intent.Read)[0];
                int DimHalf = DimsRegion.X / 2;

                for (int y = 0; y < DimsAverage.Y; y++)
                {
                    int yy = y * y;
                    for (int x = 0; x < DimHalf; x++)
                    {
                        int xx = x;
                        xx *= xx;
                        float r = (float)Math.Sqrt(xx + yy) / DimsRegion.X;
                        Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x + DimHalf] = OriginalAverageData[(DimsRegion.X - 1 - y) * (DimsRegion.X / 2 + 1) + x] - SimulatedBackground.Interp(r);
                    }

                    for (int x = 1; x < DimHalf; x++)
                    {
                        int xx = -(x - DimHalf);
                        float r = (float)Math.Sqrt(xx * xx + yy) / DimsRegion.X;
                        Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x] = OriginalAverageData[y * (DimsRegion.X / 2 + 1) + xx] - SimulatedBackground.Interp(r);
                    }
                }

                IOHelper.WriteMapFloat(PowerSpectrumPath,
                                       new HeaderMRC
                                       {
                                           Dimensions = DimsAverage,
                                           MinValue = MathHelper.Min(Average2DData),
                                           MaxValue = MathHelper.Max(Average2DData)
                                       },
                                       Average2DData);
            }
            #endregion
            CTFTimers[6].Finish(Timer6);

            var Timer7 = CTFTimers[7].Start();
            for (int i = 0; i < PS1D.Length; i++)
                PS1D[i].Y -= SimulatedBackground.Interp(PS1D[i].X);
            SimulatedBackground = new Cubic1D(SimulatedBackground.Data.Select(v => new float2(v.X, 0f)).ToArray());

            CTFSpectra.Dispose();
            CTFMean.Dispose();
            CTFCoordsCart.Dispose();
            CTFCoordsPolarTrimmed.Dispose();

            Simulated1D = GetSimulated1D();
            //CTFQuality = GetCTFQuality();

            #region Estimate fittable resolution
            {
                float[] Quality = CTF.EstimateQuality(PS1D.Select(p => p.Y).ToArray(), SimulatedScale.Interp(PS1D.Select(p => p.X).ToArray()), (float)options.RangeMin, 16);
                int FirstFreq = MinFreqInclusive + NFreq / 2;
                //while ((float.IsNaN(Quality[FirstFreq]) || Quality[FirstFreq] < 0.8f) && FirstFreq < Quality.Length - 1)
                //    FirstFreq++;

                int LastFreq = FirstFreq;
                while (!float.IsNaN(Quality[LastFreq]) && Quality[LastFreq] > 0.3f && LastFreq < Quality.Length - 1)
                    LastFreq++;

                CTFResolutionEstimate = Math.Round(options.BinnedPixelSizeMean / ((decimal)LastFreq / options.Window), 1);
            }
            #endregion

            OptionsCTF = options;

            SaveMeta();

            IsProcessing = false;
            CTFTimers[7].Finish(Timer7);

            //lock (CTFTimers)
            //{
            //    if (CTFTimers[0].NItems > 5)
            //        using (TextWriter Writer = File.CreateText("d_ctftimers.txt"))
            //            foreach (var timer in CTFTimers)
            //            {
            //                Debug.WriteLine(timer.Name + ": " + timer.GetAverageMilliseconds(100).ToString("F0"));
            //                Writer.WriteLine(timer.Name + ": " + timer.GetAverageMilliseconds(100).ToString("F0"));
            //            }
            //}
        }

        public void ProcessShift(Image originalStack, ProcessingOptionsMovieMovement options)
        {
            IsProcessing = true;

            // Deal with dimensions and grids.

            int NFrames = originalStack.Dims.Z;
            int2 DimsImage = new int2(originalStack.Dims);
            int2 DimsRegion = new int2(768, 768);

            float OverlapFraction = 0.5f;
            int2 DimsPositionGrid;
            int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, DimsRegion, OverlapFraction, out DimsPositionGrid);
            //PositionGrid = new[] { new int3(0, 0, 0) };
            //DimsPositionGrid = new int2(1, 1);
            int NPositions = PositionGrid.Length;

            // Auto grid dims
            if (options.GridDims.Elements() == 0)
            {
                float OverallDose = (float)(options.DosePerAngstromFrame < 0 ? 
                                            -options.DosePerAngstromFrame : 
                                            options.DosePerAngstromFrame * NFrames);

                int AutoZ = (int)MathF.Ceiling(Math.Max(1, OverallDose));
                int AutoX, AutoY;

                // For a FoV of 4000 Angstrom, use a grid side length of 5
                // Scale up linearly for larger FoV
                // Scale down linearly for lower dose than 30 e/A^2
                float ShortAngstrom = Math.Min(originalStack.Dims.X, originalStack.Dims.Y) * (float)options.BinnedPixelSizeMean;
                int ShortGrid = (int)Math.Max(1, MathF.Round(5f * Math.Min(1, OverallDose / 30f) * (ShortAngstrom / 4000f)));

                if (originalStack.Dims.X <= originalStack.Dims.Y)
                {
                    AutoX = ShortGrid;
                    AutoY = (int)MathF.Round(ShortGrid * (float)originalStack.Dims.Y / originalStack.Dims.X);
                }
                else
                {
                    AutoY = ShortGrid;
                    AutoX = (int)MathF.Round(ShortGrid * (float)originalStack.Dims.X / originalStack.Dims.Y);
                }

                options.GridDims = new int3(AutoX, AutoY, AutoZ);
            }

            int ShiftGridX = options.GridDims.X;
            int ShiftGridY = options.GridDims.Y;
            int ShiftGridZ = Math.Min(NFrames, options.GridDims.Z);
            GridMovementX = new CubicGrid(new int3(1, 1, ShiftGridZ));
            GridMovementY = new CubicGrid(new int3(1, 1, ShiftGridZ));

            int LocalGridX = Math.Min(DimsPositionGrid.X, options.GridDims.X);
            int LocalGridY = Math.Min(DimsPositionGrid.Y, options.GridDims.Y);
            int LocalGridZ = LocalGridX * LocalGridY <= 1 ? 1 : 4;//Math.Max(3, (int)Math.Ceiling(options.GridDims.Z / (float)(LocalGridX * LocalGridY)));
            GridLocalX = new CubicGrid(new int3(LocalGridX, LocalGridY, LocalGridZ));
            GridLocalY = new CubicGrid(new int3(LocalGridX, LocalGridY, LocalGridZ));

            PyramidShiftX = new List<CubicGrid>();
            PyramidShiftY = new List<CubicGrid>();

            int3 ShiftGrid = new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames);

            int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
            int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
            int NFreq = MaxFreqExclusive - MinFreqInclusive;

            int CentralFrame = NFrames / 2;

            int MaskExpansions = Math.Max(1, (int)Math.Ceiling(Math.Log(ShiftGridZ - 0.01, 3)));    // Each expansion doubles the temporal resolution
            int[] MaskSizes = new int[MaskExpansions];

            // Allocate memory and create all prerequisites:
            int MaskLength;
            Image ShiftFactors;
            Image Patches;
            Image PatchesAverage;
            Image Shifts;
            {
                List<long> Positions = new List<long>();
                List<float2> Factors = new List<float2>();
                List<float2> Freq = new List<float2>();
                int Min2 = MinFreqInclusive * MinFreqInclusive;
                int Max2 = MaxFreqExclusive * MaxFreqExclusive;
                float PixelSize = (float)options.BinnedPixelSizeMean;

                for (int y = 0; y < DimsRegion.Y; y++)
                {
                    int yy = y > DimsRegion.Y / 2 ? y - DimsRegion.Y : y;
                    for (int x = 0; x < DimsRegion.X / 2 + 1; x++)
                    {
                        int xx = x;
                        int r2 = xx * xx + yy * yy;
                        if (r2 >= Min2 && r2 < Max2)
                        {
                            Positions.Add(y * (DimsRegion.X / 2 + 1) + x);
                            Factors.Add(new float2((float)xx / DimsRegion.X * 2f * (float)Math.PI,
                                                   (float)yy / DimsRegion.Y * 2f * (float)Math.PI));

                            float Angle = (float)Math.Atan2(yy, xx);
                            float r = (float)Math.Sqrt(r2);
                            Freq.Add(new float2(r, Angle));
                        }
                    }
                }

                // Sort everyone by ascending distance from center.
                List<KeyValuePair<float, int>> FreqIndices = Freq.Select((v, i) => new KeyValuePair<float, int>(v.X, i)).ToList();
                FreqIndices.Sort((a, b) => a.Key.CompareTo(b.Key));
                int[] SortedIndices = FreqIndices.Select(v => v.Value).ToArray();

                Helper.Reorder(Positions, SortedIndices);
                Helper.Reorder(Factors, SortedIndices);
                Helper.Reorder(Freq, SortedIndices);

                float[] CTF2D = Helper.ArrayOfConstant(1f, Freq.Count);
                if (OptionsCTF != null)
                {
                    CTF CTFCopy = CTF.GetCopy();
                    CTFCopy.PixelSize = (decimal)PixelSize;
                    CTF2D = CTFCopy.Get2D(Freq.Select(v => new float2(v.X / DimsRegion.X, v.Y)).ToArray(), true);
                }

                long[] RelevantMask = Positions.ToArray();
                ShiftFactors = new Image(Helper.ToInterleaved(Factors.ToArray()));
                MaskLength = RelevantMask.Length;

                // Get mask sizes for different expansion steps.
                for (int i = 0; i < MaskExpansions; i++)
                {
                    float CurrentMaxFreq = MinFreqInclusive + (MaxFreqExclusive - MinFreqInclusive) / (float)MaskExpansions * (i + 1);
                    MaskSizes[i] = Freq.Count(v => v.X * v.X < CurrentMaxFreq * CurrentMaxFreq);
                }

                Patches = new Image(IntPtr.Zero, new int3(MaskLength, DimsPositionGrid.X * DimsPositionGrid.Y, NFrames), false, true, false);
                Image Sigma = new Image(IntPtr.Zero, new int3(DimsRegion), true);

                GPU.CreateShift(originalStack.GetDevice(Intent.Read),
                                DimsImage,
                                originalStack.Dims.Z,
                                PositionGrid,
                                PositionGrid.Length,
                                DimsRegion,
                                RelevantMask,
                                (uint)MaskLength,
                                Patches.GetDevice(Intent.Write),
                                Sigma.GetDevice(Intent.Write));

                //Sigma.WriteMRC("d_sigma.mrc", true);
                float AmpsMean = MathHelper.Mean(Sigma.GetHostContinuousCopy());
                //float[] Sigma1D = Sigma.AsAmplitudes1D(false);
                Sigma.Dispose();
                //Sigma1D[0] = Sigma1D[1];
                //float Sigma1DMean = MathHelper.Mean(Sigma1D);
                //Sigma1D = Sigma1D.Select(v => v / Sigma1DMean).ToArray();
                //Sigma1D = Sigma1D.Select(v => v > 0 ? 1 / v : 0).ToArray();
                //Sigma1D = Sigma1D.Select(v => 1 / Sigma1DMean).ToArray();

                float Bfac = (float)options.Bfactor * 0.25f;
                float[] BfacWeightsData = Freq.Select((v, i) =>
                {
                    float r2 = v.X / PixelSize / DimsRegion.X;
                    r2 *= r2;
                    return (float)Math.Exp(r2 * Bfac);// * CTF2D[i];// * Sigma1D[(int)Math.Round(v.X)];
                }).ToArray();
                Image BfacWeights = new Image(BfacWeightsData);

                Patches.MultiplyLines(BfacWeights);
                Patches.Multiply(1 / AmpsMean);
                BfacWeights.Dispose();

                originalStack.FreeDevice();
                PatchesAverage = new Image(IntPtr.Zero, new int3(MaskLength, NPositions, 1), false, true);
                Shifts = new Image(new float[NPositions * NFrames * 2]);
            }

            #region Fit movement

            var Timer0 = ShiftTimers[0].Start();
            {
                int MinXSteps = 1, MinYSteps = 1;
                int MinZSteps = Math.Min(NFrames, 3);
                int3 ExpansionGridSize = new int3(MinXSteps, MinYSteps, MinZSteps);
                int3 LocalGridSize = new int3(LocalGridX, LocalGridY, LocalGridZ);
                int LocalGridParams = (int)LocalGridSize.Elements() * 2;

                // Get wiggle weights for global and local shifts, will need latter in last iteration
                float[][] WiggleWeights = new CubicGrid(ExpansionGridSize).GetWiggleWeights(ShiftGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));
                float[][] WiggleWeightsLocal = new CubicGrid(LocalGridSize).GetWiggleWeights(ShiftGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                double[] StartParams = new double[ExpansionGridSize.Elements() * 2 + LocalGridParams];

                for (int m = 0; m < MaskExpansions; m++)
                {
                    bool LastIter = m == MaskExpansions - 1;
                    double[] LastAverage = null;

                    int ExpansionGridParams = (int)ExpansionGridSize.Elements() * 2;

                    #region Helper methods

                    Action<double[]> SetPositions = input =>
                    {
                        // Construct CubicGrids and get interpolated shift values.
                        CubicGrid AlteredGridX = new CubicGrid(ExpansionGridSize, input.Take(ExpansionGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                        CubicGrid AlteredGridY = new CubicGrid(ExpansionGridSize, input.Take(ExpansionGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());

                        float[] AlteredX = AlteredGridX.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                                                                              new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));
                        float[] AlteredY = AlteredGridY.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                                                                              new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                        // In last iteration, also model local motion
                        //if (LastIter)
                        {
                            CubicGrid AlteredGridLocalX = new CubicGrid(LocalGridSize, input.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                            CubicGrid AlteredGridLocalY = new CubicGrid(LocalGridSize, input.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());

                            float[] AlteredLocalX = AlteredGridLocalX.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                                                                                  new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));
                            float[] AlteredLocalY = AlteredGridLocalY.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                                                                                  new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                            for (int i = 0; i < AlteredX.Length; i++)
                            {
                                AlteredX[i] += AlteredLocalX[i];
                                AlteredY[i] += AlteredLocalY[i];
                            }
                        }

                        // Finally, set the shift values in the device array.
                        float[] ShiftData = Shifts.GetHost(Intent.Write)[0];
                        Parallel.For(0, AlteredX.Length, i =>
                        {
                            ShiftData[i * 2] = AlteredX[i];// - CenterFrameOffsets[i % NPositions].X;
                            ShiftData[i * 2 + 1] = AlteredY[i];// - CenterFrameOffsets[i % NPositions].Y;
                        });
                    };

                    Action<double[]> DoAverage = input =>
                    {
                        if (LastAverage == null || input.Where((t, i) => t != LastAverage[i]).Any())
                        {
                            SetPositions(input);
                            GPU.ShiftGetAverage(Patches.GetDevice(Intent.Read),
                                                PatchesAverage.GetDevice(Intent.Write),
                                                ShiftFactors.GetDevice(Intent.Read),
                                                (uint)MaskLength,
                                                (uint)MaskSizes[m],
                                                Shifts.GetDevice(Intent.Read),
                                                (uint)NPositions,
                                                (uint)NFrames);

                            if (LastAverage == null)
                                LastAverage = new double[input.Length];
                            Array.Copy(input, LastAverage, input.Length);
                        }
                    };

                    #endregion

                    #region Eval and gradient methods

                    Func<double[], double> Eval = input =>
                    {
                        DoAverage(input);
                        //SetPositions(input);

                        float[] Diff = new float[NPositions * NFrames];
                        GPU.ShiftGetDiff(Patches.GetDevice(Intent.Read),
                                         PatchesAverage.GetDevice(Intent.Read),
                                         ShiftFactors.GetDevice(Intent.Read),
                                         (uint)MaskLength,
                                         (uint)MaskSizes[m],
                                         Shifts.GetDevice(Intent.Read),
                                         Diff,
                                         (uint)NPositions,
                                         (uint)NFrames);

                        for (int i = 0; i < Diff.Length; i++)
                            Diff[i] = Diff[i];

                        //Debug.WriteLine(Diff.Sum());

                        return Diff.Sum();
                    };

                    Func<double[], double[]> Grad = input =>
                    {
                        DoAverage(input);
                        //SetPositions(input);

                        float[] GradX = new float[NPositions * NFrames], GradY = new float[NPositions * NFrames];

                        float[] Diff = new float[NPositions * NFrames * 2];
                        GPU.ShiftGetGrad(Patches.GetDevice(Intent.Read),
                                         PatchesAverage.GetDevice(Intent.Read),
                                         ShiftFactors.GetDevice(Intent.Read),
                                         (uint)MaskLength,
                                         (uint)MaskSizes[m],
                                         Shifts.GetDevice(Intent.Read),
                                         Diff,
                                         (uint)NPositions,
                                         (uint)NFrames);

                        for (int i = 0; i < GradX.Length; i++)
                        {
                            GradX[i] = Diff[i * 2];
                            GradY[i] = Diff[i * 2 + 1];
                        }

                        double[] Result = new double[input.Length];

                        Parallel.For(0, ExpansionGridParams / 2, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, i =>
                        //for (int i = 0; i < ExpansionGridParams / 2; i++)
                        {
                            Result[i * 2] = MathHelper.ReduceWeighted(GradX, WiggleWeights[i]);
                            Result[i * 2 + 1] = MathHelper.ReduceWeighted(GradY, WiggleWeights[i]);
                        });

                        //if (LastIter)
                        Parallel.For(0, LocalGridParams / 2, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, i =>
                        //for (int i = 0; i < LocalGridParams / 2; i++)
                        {
                            Result[ExpansionGridParams + i * 2] = MathHelper.ReduceWeighted(GradX, WiggleWeightsLocal[i]);
                            Result[ExpansionGridParams + i * 2 + 1] = MathHelper.ReduceWeighted(GradY, WiggleWeightsLocal[i]);
                        });

                        return Result;
                    };

                    #endregion

                    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                    Optimizer.MaxIterations = 10;
                    Optimizer.Minimize(StartParams);

                    // Anything should be quite centered anyway

                    //float MeanX = MathHelper.Mean(Optimizer.Solution.Where((v, i) => i % 2 == 0).Select(v => (float)v));
                    //float MeanY = MathHelper.Mean(Optimizer.Solution.Where((v, i) => i % 2 == 1).Select(v => (float)v));
                    //for (int i = 0; i < ExpansionGridSize.Elements(); i++)
                    //{
                    //    Optimizer.Solution[i * 2] -= MeanX;
                    //    Optimizer.Solution[i * 2 + 1] -= MeanY;
                    //}

                    // Store coarse values in grids.
                    GridMovementX = new CubicGrid(ExpansionGridSize, Optimizer.Solution.Take(ExpansionGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                    GridMovementY = new CubicGrid(ExpansionGridSize, Optimizer.Solution.Take(ExpansionGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());

                    //if (LastIter)
                    {
                        GridLocalX = new CubicGrid(LocalGridSize, Optimizer.Solution.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                        GridLocalY = new CubicGrid(LocalGridSize, Optimizer.Solution.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());
                    }

                    if (!LastIter)
                    {
                        // Refine sampling.
                        ExpansionGridSize = new int3(1, //(int)Math.Round((float)(ShiftGridX - MinXSteps) / (MaskExpansions - 1) * (m + 1) + MinXSteps),
                                                     1, //(int)Math.Round((float)(ShiftGridY - MinYSteps) / (MaskExpansions - 1) * (m + 1) + MinYSteps),
                                                     (int)Math.Round((float)Math.Min(ShiftGridZ, Math.Pow(3, m + 2))));
                        ExpansionGridParams = (int)ExpansionGridSize.Elements() * 2;

                        WiggleWeights = new CubicGrid(ExpansionGridSize).GetWiggleWeights(ShiftGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                        // Resize the grids to account for finer sampling.
                        GridMovementX = GridMovementX.Resize(ExpansionGridSize);
                        GridMovementY = GridMovementY.Resize(ExpansionGridSize);

                        // Construct start parameters for next optimization iteration.
                        StartParams = new double[ExpansionGridParams + LocalGridParams];
                        for (int i = 0; i < ExpansionGridParams / 2; i++)
                        {
                            StartParams[i * 2] = GridMovementX.FlatValues[i];
                            StartParams[i * 2 + 1] = GridMovementY.FlatValues[i];
                        }
                        for (int i = 0; i < LocalGridParams / 2; i++)
                        {
                            StartParams[ExpansionGridParams + i * 2] = GridLocalX.FlatValues[i];
                            StartParams[ExpansionGridParams + i * 2 + 1] = GridLocalY.FlatValues[i];
                        }
                        // Local shifts will be initialized with 0 for last iteration
                    }
                }
            }
            ShiftTimers[0].Finish(Timer0);
            #endregion

            // Center the global shifts
            {
                float2 AverageShift = new float2(MathHelper.Mean(GridMovementX.FlatValues),
                                                  MathHelper.Mean(GridMovementY.FlatValues));

                GridMovementX = new CubicGrid(GridMovementX.Dimensions, GridMovementX.FlatValues.Select(v => v - AverageShift.X).ToArray());
                GridMovementY = new CubicGrid(GridMovementY.Dimensions, GridMovementY.FlatValues.Select(v => v - AverageShift.Y).ToArray());
            }

            // Scale everything from (binned) pixels to Angstrom
            GridMovementX = new CubicGrid(GridMovementX.Dimensions, GridMovementX.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());
            GridMovementY = new CubicGrid(GridMovementY.Dimensions, GridMovementY.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());

            GridLocalX = new CubicGrid(GridLocalX.Dimensions, GridLocalX.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());
            GridLocalY = new CubicGrid(GridLocalY.Dimensions, GridLocalY.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());

            ShiftFactors.Dispose();
            Patches.Dispose();
            PatchesAverage.Dispose();
            Shifts.Dispose();

            OptionsMovement = options;

            // Calculate mean per-frame shift
            {
                float2[] Track = GetMotionTrack(new float2(0.5f, 0.5f), 1);
                float[] Diff = MathHelper.Diff(Track).Select(v => v.Length()).ToArray();
                MeanFrameMovement = (decimal)MathHelper.Mean(Diff.Take(Math.Max(1, Diff.Length / 3)));
            }

            SaveMeta();

            //lock (ShiftTimers)
            //{
            //    if (ShiftTimers[0].NItems > 0)
            //        foreach (var timer in ShiftTimers)
            //            Console.WriteLine(timer.Name + ": " + timer.GetAverageMilliseconds(1).ToString("F0"));
            //}

            IsProcessing = false;
        }

        public void MatchBoxNet2(BoxNetTorch[] networks, ProcessingOptionsBoxNet options, Func<int3, int, string, bool> progressCallback)
        {
            Stopwatch WatchPreflight = new Stopwatch();
            WatchPreflight.Start();

            IsProcessing = true;

            Directory.CreateDirectory(MatchingDir);

            Image Average = Image.FromFile(string.IsNullOrEmpty(options.OverrideImagePath) ? AveragePath : options.OverrideImagePath);
            float AveragePixelSize = Average.PixelSize;

            float PixelSizeBN = BoxNetTorch.PixelSize;
            int BatchSizeBN = networks[0].BatchSize;
            int2 DimsRegionBN = BoxNetTorch.DefaultDimensionsPredict;
            int2 DimsRegionValidBN = BoxNetTorch.DefaultDimensionsValidPredict;
            int BorderBN = (DimsRegionBN.X - DimsRegionValidBN.X) / 2;

            int2 DimsBN = (new int2(Average.Dims * AveragePixelSize / PixelSizeBN) + 1) / 2 * 2;

            //Image SoftMask = new Image(new int3(DimsBN.X * 2, DimsBN.Y * 2, 1));
            //SoftMask.TransformValues((x, y, z, v) =>
            //{
            //    float xx = (float)Math.Max(0, Math.Max(DimsBN.X / 2 - x, x - DimsBN.X * 3 / 2)) / (DimsBN.X / 2);
            //    float yy = (float)Math.Max(0, Math.Max(DimsBN.Y / 2 - y, y - DimsBN.Y * 3 / 2)) / (DimsBN.Y / 2);

            //    float r = Math.Min(1, (float)Math.Sqrt(xx * xx + yy * yy));
            //    float w = ((float)Math.Cos(r * Math.PI) + 1) * 0.5f;

            //    return w;
            //});

            Image AverageBN = Average.AsScaled(DimsBN).AndDisposeParent();
            //AverageBN.Normalize();

            AverageBN = AverageBN.AsPadded(DimsBN - 2).AndDisposeParent();
            Image AverageBNPadded = AverageBN.AsPaddedClamped(DimsBN * 2).AndDisposeParent();
            //AverageBNPadded.MultiplySlices(SoftMask);
            AverageBNPadded.MaskRectangularly(new int3(DimsBN), Math.Min(DimsBN.X, DimsBN.Y) / 2, false);
            AverageBNPadded.Bandpass(2f * PixelSizeBN / 500f, 1, false, 2f * PixelSizeBN / 500f);
            //SoftMask.Dispose();

            AverageBN = AverageBNPadded.AsPadded(DimsBN).AndDisposeParent();

            float2 MeanStd = new float2(0, 1);
            {
                Image AverageCenter = AverageBN.AsPadded(DimsBN / 2);
                MeanStd = MathHelper.MeanAndStd(AverageCenter.GetHost(Intent.Read)[0]);
                AverageCenter.Dispose();
            }

            MeanStd.Y = Math.Max(1e-16f, MeanStd.Y);
            //AverageBN.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
            AverageBN.Add(-MeanStd.X);
            AverageBN.Multiply(1f / MeanStd.Y * (options.PickingInvert ? -1 : 1));

            //GPU.Normalize(AverageBN.GetDevice(Intent.Read),
            //              AverageBN.GetDevice(Intent.Write),
            //              (uint)AverageBN.ElementsSliceReal,
            //              1);

            //if (options.PickingInvert)
            //    AverageBN.Multiply(-1f);

            int2 Margin = (BoxNetTorch.DefaultDimensionsPredict - BoxNetTorch.DefaultDimensionsValidPredict) / 2;
            DimsBN = DimsBN + Margin * 2;
            AverageBN = AverageBN.AsPadded(DimsBN).AndDisposeParent();

            float[] Predictions = new float[DimsBN.Elements()];
            float[] Mask = new float[DimsBN.Elements()];

            WatchPreflight.Stop();
            Debug.WriteLine("Preflight: " + WatchPreflight.ElapsedMilliseconds / 1000.0);

            {
                int2 DimsPositions = (DimsBN + DimsRegionValidBN - 1) / DimsRegionValidBN;
                float2 PositionStep = new float2(DimsBN - DimsRegionBN) / new float2(Math.Max(DimsPositions.X - 1, 1),
                                                                                     Math.Max(DimsPositions.Y - 1, 1));

                int NPositions = (int)DimsPositions.Elements();

                int3[] Positions = new int3[NPositions];
                for (int p = 0; p < NPositions; p++)
                {
                    int X = p % DimsPositions.X;
                    int Y = p / DimsPositions.X;
                    Positions[p] = new int3((int)(X * PositionStep.X + DimsRegionBN.X / 2),
                                            (int)(Y * PositionStep.Y + DimsRegionBN.Y / 2),
                                            0);
                }

                float[][] PredictionTiles = Helper.ArrayOfFunction(i => new float[DimsRegionBN.Elements()], NPositions);
                float[][] MaskTiles = Helper.ArrayOfFunction(i => new float[DimsRegionBN.Elements()], NPositions);

                int3[][] BatchedPositions = new int3[(NPositions + BatchSizeBN - 1) / BatchSizeBN][];
                for (int i = 0; i < BatchedPositions.Length; i++)
                {
                    int3[] BatchPositions = new int3[BatchSizeBN];
                    for (int j = 0; j < BatchSizeBN; j++)
                        BatchPositions[j] = Positions[Math.Min(NPositions - 1, i * BatchSizeBN + j)] - new int3(DimsRegionBN.X / 2, DimsRegionBN.Y / 2, 0);

                    BatchedPositions[i] = BatchPositions;
                }

                Image Extracted = new Image(IntPtr.Zero, new int3(DimsRegionBN.X, DimsRegionBN.Y, BatchSizeBN));
                int DeviceID = GPU.GetDevice();

                int BatchesDone = 0;
                float Threshold = (float)options.MinimumScore;

                lock (networks[0])
                {
                    Stopwatch WatchInference = new Stopwatch();
                    WatchInference.Start();

                    for (int b = 0; b < BatchedPositions.Length; b++)
                    {
                        #region Extract and normalize windows

                        GPU.Extract(AverageBN.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    AverageBN.Dims,
                                    new int3(DimsRegionBN),
                                    Helper.ToInterleaved(BatchedPositions[b]),
                                    true,
                                    (uint)BatchSizeBN);

                        #endregion

                        //Extracted[threadID].WriteMRC("d_extracted.mrc", true);

                        #region Predict

                        float[] BatchArgMax = null;
                        float[] BatchProbability = null;
                        networks[0].Predict(Extracted,
                                            out BatchArgMax,
                                            out BatchProbability);

                        //new Image(BatchArgMax.Select(v => (float)v).ToArray(), new int3(DimsRegionBN)).WriteMRC("d_labels.mrc", true);

                        int SliceElements = (int)DimsRegionBN.Elements();
                        for (int ib = 0; ib < BatchSizeBN; ib++)
                        {
                            if (b * BatchSizeBN + ib >= NPositions)
                                break;

                            for (int i = 0; i < SliceElements; i++)
                            {
                                int Label = (int)BatchArgMax[ib * SliceElements + i];
                                float Probability = BatchProbability[ib * SliceElements * 3 + Label * SliceElements + i];

                                PredictionTiles[b * BatchSizeBN + ib][i] = (Label == 1 && Probability >= Threshold ? Probability : 0);
                                MaskTiles[b * BatchSizeBN + ib][i] = Label == 2 && Probability >= 0.0f ? 1 : 0;
                            }
                        }

                        #endregion

                        progressCallback?.Invoke(new int3(NPositions, 1, 1), ++BatchesDone, "");
                    }

                    WatchInference.Stop();
                    Debug.WriteLine("Inference: " + WatchInference.ElapsedMilliseconds / 1000.0);
                }

                AverageBN.FreeDevice();
                Extracted.Dispose();

                Parallel.For(0, DimsBN.Y, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, y =>
                {
                    for (int x = 0; x < DimsBN.X; x++)
                    {
                        int ClosestX = (int)Math.Max(0, Math.Min(DimsPositions.X - 1, (int)(((float)x - DimsRegionBN.X / 2) / PositionStep.X + 0.5f)));
                        int ClosestY = (int)Math.Max(0, Math.Min(DimsPositions.Y - 1, (int)(((float)y - DimsRegionBN.Y / 2) / PositionStep.Y + 0.5f)));
                        int ClosestID = ClosestY * DimsPositions.X + ClosestX;

                        int3 Position = Positions[ClosestID];
                        int LocalX = Math.Max(0, Math.Min(DimsRegionBN.X - 1, x - Position.X + DimsRegionBN.X / 2));
                        int LocalY = Math.Max(0, Math.Min(DimsRegionBN.Y - 1, y - Position.Y + DimsRegionBN.Y / 2));

                        Predictions[y * DimsBN.X + x] = PredictionTiles[ClosestID][LocalY * DimsRegionBN.X + LocalX];
                        Mask[y * DimsBN.X + x] = MaskTiles[ClosestID][LocalY * DimsRegionBN.X + LocalX];
                    }
                });
            }


            AverageBN.Dispose();

            #region Rescale and save mask

            Image MaskImage = new Image(Mask, new int3(DimsBN));
            DimsBN -= Margin * 2;
            MaskImage = MaskImage.AsPadded(DimsBN).AndDisposeParent();

            // Get rid of all connected components in the mask that are too small
            {
                float[] MaskData = MaskImage.GetHost(Intent.ReadWrite)[0];

                List<List<int2>> Components = new List<List<int2>>();
                int[] PixelLabels = Helper.ArrayOfConstant(-1, MaskData.Length);

                for (int y = 0; y < DimsBN.Y; y++)
                {
                    for (int x = 0; x < DimsBN.X; x++)
                    {
                        int2 peak = new int2(x, y);

                        if (MaskData[DimsBN.ElementFromPosition(peak)] != 1 || PixelLabels[DimsBN.ElementFromPosition(peak)] >= 0)
                            continue;

                        List<int2> Component = new List<int2>() { peak };
                        int CN = Components.Count;

                        PixelLabels[DimsBN.ElementFromPosition(peak)] = CN;
                        Queue<int2> Expansion = new Queue<int2>(100);
                        Expansion.Enqueue(peak);

                        while (Expansion.Count > 0)
                        {
                            int2 pos = Expansion.Dequeue();
                            int PosElement = DimsBN.ElementFromPosition(pos);

                            if (pos.X > 0 && MaskData[PosElement - 1] == 1 && PixelLabels[PosElement - 1] < 0)
                            {
                                PixelLabels[PosElement - 1] = CN;
                                Component.Add(pos + new int2(-1, 0));
                                Expansion.Enqueue(pos + new int2(-1, 0));
                            }
                            if (pos.X < DimsBN.X - 1 && MaskData[PosElement + 1] == 1 && PixelLabels[PosElement + 1] < 0)
                            {
                                PixelLabels[PosElement + 1] = CN;
                                Component.Add(pos + new int2(1, 0));
                                Expansion.Enqueue(pos + new int2(1, 0));
                            }

                            if (pos.Y > 0 && MaskData[PosElement - DimsBN.X] == 1 && PixelLabels[PosElement - DimsBN.X] < 0)
                            {
                                PixelLabels[PosElement - DimsBN.X] = CN;
                                Component.Add(pos + new int2(0, -1));
                                Expansion.Enqueue(pos + new int2(0, -1));
                            }
                            if (pos.Y < DimsBN.Y - 1 && MaskData[PosElement + DimsBN.X] == 1 && PixelLabels[PosElement + DimsBN.X] < 0)
                            {
                                PixelLabels[PosElement + DimsBN.X] = CN;
                                Component.Add(pos + new int2(0, 1));
                                Expansion.Enqueue(pos + new int2(0, 1));
                            }
                        }

                        Components.Add(Component);
                    }
                }

                foreach (var component in Components)
                    if (component.Count < 20)
                        foreach (var pos in component)
                            MaskData[DimsBN.ElementFromPosition(pos)] = 0;

                MaskPercentage = (decimal)MaskData.Sum() / MaskData.Length * 100;
            }

            Image MaskImage8 = MaskImage.AsScaled(new int2(new float2(DimsBN) * BoxNetTorch.PixelSize / 8) / 2 * 2);
            MaskImage8.Binarize(0.5f);

            int MaxHitTestDistance = (int)((options.ExpectedDiameter / 2 + options.MinimumMaskDistance) / (decimal)BoxNetTorch.PixelSize) + 2;
            Image MaskDistance = MaskImage.AsDistanceMapExact(MaxHitTestDistance);
            MaskImage.Dispose();
            MaskDistance.Binarize(MaxHitTestDistance - 2);
            float[] MaskHitTest = MaskDistance.GetHostContinuousCopy();
            MaskDistance.Dispose();

            Directory.CreateDirectory(MaskDir);
            MaskImage8.WriteTIFF(MaskPath, 8, typeof(float));
            MaskImage8.Dispose();

            #endregion

            #region Find peaks

            Image PredictionsImage = new Image(Predictions, new int3(DimsBN + Margin * 2));
            PredictionsImage = PredictionsImage.AsPadded(DimsBN).AndDisposeParent();

            //Image PredictionsConvolved = PredictionsImage.AsConvolvedGaussian((float)options.ExpectedDiameter / PixelSizeBN / 6);
            //PredictionsConvolved.Multiply(PredictionsImage);
            //PredictionsImage.Dispose();

            //PredictionsImage.WriteMRC(MatchingDir + RootName + "_boxnet.mrc", PixelSizeBN, true);

            int3[] Peaks = PredictionsImage.GetLocalPeaks((int)((float)options.ExpectedDiameter / PixelSizeBN / 4 + 0.5f), 1e-6f);
            PredictionsImage.Dispose();

            int BorderDist = (int)((float)options.ExpectedDiameter / PixelSizeBN * 0.8f + 0.5f);
            Peaks = Peaks.Where(p => p.X > BorderDist && p.Y > BorderDist && p.X < DimsBN.X - BorderDist && p.Y < DimsBN.Y - BorderDist).ToArray();

            #endregion

            #region Label connected components and get centroids

            List<float2> Centroids;
            int[] Extents;
            {
                List<List<int2>> Components = new List<List<int2>>();
                int[] PixelLabels = Helper.ArrayOfConstant(-1, Predictions.Length);

                foreach (var peak in Peaks.Select(v => new int2(v)))
                {
                    if (PixelLabels[DimsBN.ElementFromPosition(peak)] >= 0)
                        continue;

                    List<int2> Component = new List<int2>() { peak };
                    int CN = Components.Count;

                    PixelLabels[DimsBN.ElementFromPosition(peak)] = CN;
                    Queue<int2> Expansion = new Queue<int2>(100);
                    Expansion.Enqueue(peak);

                    while (Expansion.Count > 0)
                    {
                        int2 pos = Expansion.Dequeue();
                        int PosElement = DimsBN.ElementFromPosition(pos);

                        if (pos.X > 0 && Predictions[PosElement - 1] > 0 && PixelLabels[PosElement - 1] < 0)
                        {
                            PixelLabels[PosElement - 1] = CN;
                            Component.Add(pos + new int2(-1, 0));
                            Expansion.Enqueue(pos + new int2(-1, 0));
                        }
                        if (pos.X < DimsBN.X - 1 && Predictions[PosElement + 1] > 0 && PixelLabels[PosElement + 1] < 0)
                        {
                            PixelLabels[PosElement + 1] = CN;
                            Component.Add(pos + new int2(1, 0));
                            Expansion.Enqueue(pos + new int2(1, 0));
                        }

                        if (pos.Y > 0 && Predictions[PosElement - DimsBN.X] > 0 && PixelLabels[PosElement - DimsBN.X] < 0)
                        {
                            PixelLabels[PosElement - DimsBN.X] = CN;
                            Component.Add(pos + new int2(0, -1));
                            Expansion.Enqueue(pos + new int2(0, -1));
                        }
                        if (pos.Y < DimsBN.Y - 1 && Predictions[PosElement + DimsBN.X] > 0 && PixelLabels[PosElement + DimsBN.X] < 0)
                        {
                            PixelLabels[PosElement + DimsBN.X] = CN;
                            Component.Add(pos + new int2(0, 1));
                            Expansion.Enqueue(pos + new int2(0, 1));
                        }
                    }

                    Components.Add(Component);
                }

                Centroids = Components.Select(c => MathHelper.Mean(c.Select(v => new float2(v)))).ToList();
                Extents = Components.Select(c => c.Count).ToArray();
            }

            List<int> ToDelete = new List<int>();

            // Hit test with crap mask
            for (int c1 = 0; c1 < Centroids.Count; c1++)
            {
                float2 P1 = Centroids[c1];
                if (MaskHitTest[(int)P1.Y * DimsBN.X + (int)P1.X] == 0)
                {
                    ToDelete.Add(c1);
                    continue;
                }
            }

            for (int c1 = 0; c1 < Centroids.Count - 1; c1++)
            {
                float2 P1 = Centroids[c1];

                for (int c2 = c1 + 1; c2 < Centroids.Count; c2++)
                {
                    if ((P1 - Centroids[c2]).Length() < (float)options.ExpectedDiameter / PixelSizeBN / 1.5f)
                    {
                        int D = Extents[c1] < Extents[c2] ? c1 : c2;

                        if (!ToDelete.Contains(D))
                            ToDelete.Add(D);
                    }
                }
            }

            ToDelete.Sort();
            for (int i = ToDelete.Count - 1; i >= 0; i--)
                Centroids.RemoveAt(ToDelete[i]);

            #endregion

            #region Write peak positions into table

            if (Centroids.Any())
            {
                Star TableOut = new Star(new string[]
                {
                    "rlnCoordinateX",
                    "rlnCoordinateY",
                    "rlnMicrographName",
                    "rlnAutopickFigureOfMerit"
                });

                {
                    foreach (float2 peak in Centroids)
                    {
                        float2 Position = peak * PixelSizeBN / AveragePixelSize;
                        float Score = Predictions[DimsBN.ElementFromPosition(new int2(peak))];

                        TableOut.AddRow(new string[]
                        {
                            Position.X.ToString(CultureInfo.InvariantCulture),
                            Position.Y.ToString(CultureInfo.InvariantCulture),
                            RootName + ".mrc",
                            Score.ToString(CultureInfo.InvariantCulture)
                        });
                    }
                }

                string StarSuffix = string.IsNullOrEmpty(options.OverrideStarSuffix)
                    ? Helper.PathToName(options.ModelName)
                    : options.OverrideStarSuffix;

                TableOut.Save(IOPath.Combine(MatchingDir, RootName + "_" + StarSuffix + ".star"));
                UpdateParticleCount("_" + StarSuffix);
            }

            #endregion

            OptionsBoxNet = options;
            SaveMeta();

            IsProcessing = false;
        }

        public static Task WriteAverageAsync = null;
        private static Image ExportMovieCTFCoords = null;
        private static Image ExportMovieWiener = null;
        public void ExportMovie(Image originalStack, ProcessingOptionsMovieExport options)
        {
            Debug.WriteLine(GPU.GetFreeMemory(GPU.GetDevice()));

            IsProcessing = true;

            #region Make sure all directories are there

            Directory.CreateDirectory(AverageDir);

            if (options.DoDenoise)
            {
                Directory.CreateDirectory(AverageOddDir);
                Directory.CreateDirectory(AverageEvenDir);
            }
            if (options.DoDenoiseDeconv)
            {
                Directory.CreateDirectory(DenoiseTrainingDirOdd);
                Directory.CreateDirectory(DenoiseTrainingDirEven);
                //Directory.CreateDirectory(DenoiseTrainingDirCTF);
            }

            if (options.DoStack)
                Directory.CreateDirectory(ShiftedStackDir);
            if (options.DoDeconv)
                Directory.CreateDirectory(DeconvolvedDir);

            #endregion

            #region Helper variables

            int3 Dims = originalStack.Dims;
            int FirstFrame = Math.Max(0, Math.Min(Dims.Z - 1, options.SkipFirstN));
            int LastFrameExclusive = Math.Min(Dims.Z, Dims.Z - options.SkipLastN);
            Dims.Z = LastFrameExclusive - FirstFrame;
            bool CanDenoise = Dims.Z > 1 && options.DoDenoise;
            bool CanDenoiseDeconv = Dims.Z > 1 && options.DoDenoiseDeconv;
            float DenoisingAngPix = Math.Max(NoiseNet2DTorch.PixelSize, (float)options.BinnedPixelSizeMean);     // Denoising always done at least at 5 A/px
            int2 DimsDenoise = new int2(new float2(Dims.X, Dims.Y) * (float)options.BinnedPixelSizeMean / DenoisingAngPix + 1) / 2 * 2;

            Task WriteDeconvAsync = null;
            Task WriteStackAsync = null;

            #endregion

            var Timer1 = OutputTimers[1].Start();

            #region Prepare spectral coordinates

            float PixelSize = (float)options.BinnedPixelSizeMean;
            if (ExportMovieCTFCoords == null || ExportMovieCTFCoords.Dims.Slice() != Dims.Slice())
            {
                if (ExportMovieCTFCoords != null)
                    ExportMovieCTFCoords.Dispose();
                ExportMovieCTFCoords = CTF.GetCTFCoordsParallel(new int2(Dims), new int2(Dims));
            }
            {
                if (options.DoDeconv)
                {
                    if (ExportMovieWiener == null || ExportMovieWiener.Dims.Slice() != Dims.Slice())
                    {
                        if (ExportMovieWiener != null)
                            ExportMovieWiener.Dispose();    

                        float2[] CTFCoordsData = new float2[Dims.Slice().ElementsFFT()];
                        Helper.ForEachElementFTParallel(new int2(Dims), (x, y, xx, yy) =>
                        {
                            float xs = xx / (float)Dims.X;
                            float ys = yy / (float)Dims.Y;
                            float r = (float)Math.Sqrt(xs * xs + ys * ys);
                            float angle = (float)Math.Atan2(yy, xx);
                            float CurrentPixelSize = PixelSize;

                            CTFCoordsData[y * (Dims.X / 2 + 1) + x] = new float2(r / CurrentPixelSize, angle);
                        });

                        float[] CTF2D = CTF.Get2DFromScaledCoords(CTFCoordsData, false);
                        float HighPassNyquist = PixelSize * 2 / 100;
                        float Strength = (float)Math.Pow(10, 3 * (double)options.DeconvolutionStrength);
                        float Falloff = (float)options.DeconvolutionFalloff * 100 / PixelSize;

                        Helper.ForEachElementFT(new int2(Dims), (x, y, xx, yy) =>
                        {
                            float xs = xx / (float)Dims.X * 2;
                            float ys = yy / (float)Dims.Y * 2;
                            float r = (float)Math.Sqrt(xs * xs + ys * ys);

                            float HighPass = 1 - (float)Math.Cos(Math.Min(1, r / HighPassNyquist) * Math.PI);
                            float SNR = (float)Math.Exp(-r * Falloff) * Strength * HighPass;
                            float CTFVal = CTF2D[y * (Dims.X / 2 + 1) + x];
                            CTF2D[y * (Dims.X / 2 + 1) + x] = CTFVal / (CTFVal * CTFVal + 1 / SNR);
                        });

                        ExportMovieWiener = new Image(CTF2D, Dims.Slice(), true);
                    }
                }
            }

            #endregion

            OutputTimers[1].Finish(Timer1);

            Debug.WriteLine(GPU.GetFreeMemory(GPU.GetDevice()));

            #region Timer 0
            var Timer0 = OutputTimers[0].Start();

            Image AverageFT = new Image(Dims.Slice(), true, true);
            Image AverageOddFT = (CanDenoise || CanDenoiseDeconv) ? new Image(Dims.Slice(), true, true) : null;
            Image AverageEvenFT = (CanDenoise || CanDenoiseDeconv) ? new Image(Dims.Slice(), true, true) : null;

            #region Warp, get FTs of all relevant frames, apply spectral filter, and add to average

            var Timer7 = OutputTimers[7].Start();

            int PlanForw = FFTPlanCache.GetFFTPlan(Dims.Slice(), 1); // GPU.CreateFFTPlan(Dims.Slice(), 1);

            IntPtr[] TempArray = { GPU.MallocArray(new int2(Dims * 1)) };
            Image[] Frame = { new Image(IntPtr.Zero, Dims.Slice()) };
            Image[] FrameFT = { new Image(IntPtr.Zero, Dims.Slice(), true, true) };
            Image[] FramePrefiltered = { new Image(IntPtr.Zero, Dims.Slice()) };
            Image[] DoseImage = { new Image(IntPtr.Zero, Dims.Slice(), true) };
            Image[] PS = { new Image(Dims.Slice(), true) };

            float StepZ = 1f / Math.Max(originalStack.Dims.Z - 1, 1);

            int DeviceID = GPU.GetDevice();

            OutputTimers[7].Finish(Timer7);

            Helper.ForCPU(0, Dims.Z, 1, threadID => GPU.SetDevice(DeviceID), (z, threadID) =>
            {
                #region Timer 6
                var Timer6 = OutputTimers[6].Start();

                int2 DimsWarp = new int2(16);
                float3[] InterpPoints = new float3[DimsWarp.Elements()];
                for (int y = 0; y < DimsWarp.Y; y++)
                    for (int x = 0; x < DimsWarp.X; x++)
                        InterpPoints[y * DimsWarp.X + x] = new float3((float)x / (DimsWarp.X - 1), (float)y / (DimsWarp.Y - 1), (z + FirstFrame) * StepZ);

                float2[] WarpXY = GetShiftFromPyramid(InterpPoints);
                float[] WarpX = WarpXY.Select(v => v.X / (float)options.BinnedPixelSizeMean).ToArray();
                float[] WarpY = WarpXY.Select(v => v.Y / (float)options.BinnedPixelSizeMean).ToArray();

                OutputTimers[6].Finish(Timer6);
                #endregion

                #region Timer 2
                var Timer2 = OutputTimers[2].Start();

                GPU.CopyDeviceToDevice(originalStack.GetDeviceSlice(z + FirstFrame, Intent.Read),
                                       FramePrefiltered[threadID].GetDevice(Intent.Write),
                                       Dims.ElementsSlice());

                GPU.PrefilterForCubic(FramePrefiltered[threadID].GetDevice(Intent.ReadWrite), Dims.Slice());

                GPU.WarpImage(FramePrefiltered[threadID].GetDevice(Intent.Read),
                              Frame[threadID].GetDevice(Intent.Write),
                              new int2(Dims),
                              WarpX,
                              WarpY,
                              DimsWarp,
                              TempArray[threadID]);

                PS[threadID].Fill(1f);
                int nframe = z + FirstFrame;

                OutputTimers[2].Finish(Timer2);
                #endregion

                #region Timer 5
                var Timer5 = OutputTimers[5].Start();

                // Apply dose weighting.
                {
                    float DosePerFrame = (float)options.DosePerAngstromFrame;
                    if (options.DosePerAngstromFrame < 0)
                        DosePerFrame = -(float)options.DosePerAngstromFrame / originalStack.Dims.Z;

                    CTF CTFBfac = new CTF()
                    {
                        PixelSize = (decimal)PixelSize,
                        Defocus = 0,
                        Amplitude = 1,
                        Cs = 0,
                        Cc = 0,
                        Bfactor = GridDoseBfacs.Values.Length <= 1 ?
                                  -(decimal)(DosePerFrame * (nframe + 0.5f) * 3) :
                                  (decimal)GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, (float)nframe / Math.Max(NFrames - 1, 1))),
                        Scale = GridDoseWeights.Values.Length <= 1 ?
                                1 :
                                (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, (float)nframe / Math.Max(NFrames - 1, 1)))
                    };
                    GPU.CreateCTF(DoseImage[threadID].GetDevice(Intent.Write),
                                  ExportMovieCTFCoords.GetDevice(Intent.Read),
                                  IntPtr.Zero,
                                  (uint)ExportMovieCTFCoords.ElementsSliceComplex,
                                  new[] { CTFBfac.ToStruct() },
                                  false,
                                  1);

                    PS[threadID].Multiply(DoseImage[threadID]);
                    //DoseImage.WriteMRC("dose.mrc");
                }
                //PS.WriteMRC("ps.mrc");

                OutputTimers[5].Finish(Timer5);
                #endregion

                #region Timer 3
                lock (Frame)
                {
                    var Timer3 = OutputTimers[3].Start();

                    GPU.FFT(Frame[threadID].GetDevice(Intent.Read),
                            FrameFT[threadID].GetDevice(Intent.Write),
                            Dims.Slice(),
                            1,
                            PlanForw);

                    GPU.MultiplyComplexSlicesByScalar(FrameFT[threadID].GetDevice(Intent.Read),
                                                      PS[threadID].GetDevice(Intent.Read),
                                                      FrameFT[threadID].GetDevice(Intent.Write),
                                                      PS[threadID].ElementsSliceReal,
                                                      1);

                    AverageFT.Add(FrameFT[threadID]);

                    if (CanDenoise || CanDenoiseDeconv)
                        (z % 2 != 0 ? AverageOddFT : AverageEvenFT).Add(FrameFT[threadID]); // Odd/even frame averages for denoising training data

                    OutputTimers[3].Finish(Timer3);
                }
                #endregion

            }, null);

            originalStack.FreeDevice();

            //GPU.DestroyFFTPlan(PlanForw);

            //Debug.WriteLine(GPU.GetFreeMemory(GPU.GetDevice()));

            #endregion

            OutputTimers[0].Finish(Timer0);
            #endregion

            foreach (var array in TempArray)
                GPU.FreeArray(array);
            foreach (var image in Frame)
                image.Dispose();
            foreach (var image in FrameFT)
                image.Dispose();
            foreach (var image in FramePrefiltered)
                image.Dispose();
            foreach (var image in DoseImage)
                image.Dispose();
            foreach (var image in PS)
                image.Dispose();

            //AverageFT.Divide(Weights);
            //AverageFT.WriteMRC("averageft.mrc");
            //Weights.WriteMRC("weights.mrc");
            //Weights.Dispose();

            #region Timer 4
            var Timer4 = OutputTimers[4].Start();

            Image Average = null, AverageOdd = null, AverageEven = null;
            if (options.DoAverage)
            {
                Average = AverageFT.AsIFFT(false, 0, true, true);

                // Previous division by weight sum brought values to stack average, multiply by number of frame to go back to sum
                Average.Multiply(Dims.Z);
                Average.FreeDevice();

                Image AverageDeconvOdd = null, AverageDeconvEven = null;

                if (CanDenoiseDeconv)
                {
                    Image AverageOddFTPadded;
                    Image AverageEvenFTPadded;
                    if (DimsDenoise != new int2(AverageOddFT.Dims))
                    {
                        AverageOddFTPadded = AverageOddFT.AsPadded(DimsDenoise);
                        AverageEvenFTPadded = AverageEvenFT.AsPadded(DimsDenoise);
                    }
                    else
                    {
                        AverageOddFTPadded = AverageOddFT.GetCopyGPU();
                        AverageEvenFTPadded = AverageEvenFT.GetCopyGPU();
                    }

                    AverageDeconvOdd = AverageOddFTPadded.AsIFFT(false, 0, true).AndDisposeParent();
                    AverageDeconvEven = AverageEvenFTPadded.AsIFFT(false, 0, true).AndDisposeParent();

                    if (OptionsCTF != null)
                    {
                        Image SoftMask = new Image(new int3(DimsDenoise.X * 2, DimsDenoise.Y * 2, 1));
                        SoftMask.TransformValues((x, y, z, v) =>
                        {
                            float xx = (float)Math.Max(0, Math.Max(DimsDenoise.X / 2 - x, x - DimsDenoise.X * 3 / 2)) / (DimsDenoise.X / 2);
                            float yy = (float)Math.Max(0, Math.Max(DimsDenoise.Y / 2 - y, y - DimsDenoise.Y * 3 / 2)) / (DimsDenoise.Y / 2);

                            float r = Math.Min(1, (float)Math.Sqrt(xx * xx + yy * yy));
                            float w = ((float)Math.Cos(r * Math.PI) + 1) * 0.5f;

                            return w;
                        });

                        AverageDeconvOdd = AverageDeconvOdd.AsPadded(DimsDenoise - 2).AndDisposeParent();
                        AverageDeconvEven = AverageDeconvEven.AsPadded(DimsDenoise - 2).AndDisposeParent();

                        Image OddPadded = AverageDeconvOdd.AsPaddedClamped(DimsDenoise * 2).AndDisposeParent();
                        Image EvenPadded = AverageDeconvEven.AsPaddedClamped(DimsDenoise * 2).AndDisposeParent();

                        OddPadded.MultiplySlices(SoftMask);
                        EvenPadded.MultiplySlices(SoftMask);

                        Image OddPaddedFT = OddPadded.AsFFT().AndDisposeParent();
                        Image EvenPaddedFT = EvenPadded.AsFFT().AndDisposeParent();

                        SoftMask.Dispose();

                        CTF DeconvCTF = CTF.GetCopy();
                        DeconvCTF.PixelSize = (decimal)DenoisingAngPix;

                        float HighpassNyquist = DenoisingAngPix * 2 / 500f;

                        GPU.DeconvolveCTF(OddPaddedFT.GetDevice(Intent.Read),
                                          OddPaddedFT.GetDevice(Intent.Write),
                                          OddPaddedFT.Dims,
                                          DeconvCTF.ToStruct(),
                                          0.85f,
                                          0.25f,
                                          HighpassNyquist);
                        GPU.DeconvolveCTF(EvenPaddedFT.GetDevice(Intent.Read),
                                          EvenPaddedFT.GetDevice(Intent.Write),
                                          EvenPaddedFT.Dims,
                                          DeconvCTF.ToStruct(),
                                          0.85f,
                                          0.25f,
                                          HighpassNyquist);

                        OddPadded = OddPaddedFT.AsIFFT(false, 0, true).AndDisposeParent();
                        EvenPadded = EvenPaddedFT.AsIFFT(false, 0, true).AndDisposeParent();

                        AverageDeconvOdd = OddPadded.AsPadded(DimsDenoise).AndDisposeParent();
                        AverageDeconvEven = EvenPadded.AsPadded(DimsDenoise).AndDisposeParent();

                        //AverageDeconvOdd.Multiply(1f / (3f / 4 * NFrames));
                        //AverageDeconvEven.Multiply(1f / (1f / 4 * NFrames));

                        //Image CTFCoords = CTF.GetCTFCoords(256, 256);
                        //CTF DeconvCTF = CTF.GetCopy();
                        //DeconvCTF.PixelSize = (decimal)DenoisingAngPix;

                        //Image CTFSimulated = new Image(new int3(256, 256, 1), true);
                        //GPU.CreateCTF(CTFSimulated.GetDevice(Intent.Write),
                        //              CTFCoords.GetDevice(Intent.Read),
                        //              IntPtr.Zero,
                        //              (uint)CTFCoords.ElementsComplex,
                        //              new[] { DeconvCTF.ToStruct() },
                        //              false,
                        //              1);

                        //CTFSimulated.WriteMRC(DenoiseTrainingCTFPath, true);
                        //CTFSimulated.Dispose();
                        //CTFCoords.Dispose();
                    }
                }

                if (CanDenoise)
                {
                    AverageOdd = AverageOddFT.AsIFFT(false, 0, true);
                    AverageEven = AverageEvenFT.AsIFFT(false, 0, true);
                }

                AverageOddFT?.Dispose();
                AverageEvenFT?.Dispose();

                // Write average async to disk
                WriteAverageAsync = new Task(() =>
                {
                    Average.WriteMRC16b(AveragePath, (float)options.BinnedPixelSizeMean, true);
                    Average.Dispose();

                    AverageOdd?.WriteMRC16b(AverageOddPath, (float)options.BinnedPixelSizeMean, true);
                    AverageOdd?.Dispose();
                    AverageEven?.WriteMRC16b(AverageEvenPath, (float)options.BinnedPixelSizeMean, true);
                    AverageEven?.Dispose();

                    AverageDeconvOdd?.WriteMRC16b(DenoiseTrainingOddPath, DenoisingAngPix, true);
                    AverageDeconvOdd?.Dispose();
                    AverageDeconvEven?.WriteMRC16b(DenoiseTrainingEvenPath, DenoisingAngPix, true);
                    AverageDeconvEven?.Dispose();

                    OnAverageChanged();
                });
                WriteAverageAsync.Start();
            }

            Image Deconvolved = null;
            if (options.DoDeconv)
            {
                AverageFT.Multiply(ExportMovieWiener);
                //ExportMovieWiener.Dispose();
                Deconvolved = AverageFT.AsIFFT(false, 0, true);
                Deconvolved.Multiply(Dims.Z);   // It's only the average at this point, needs to be sum

                // Write deconv async to disk
                WriteDeconvAsync = new Task(() =>
                {
                    Deconvolved.WriteMRC16b(DeconvolvedPath, (float)options.BinnedPixelSizeMean, true);
                    Deconvolved.Dispose();
                });
                WriteDeconvAsync.Start();
            }

            AverageFT.Dispose();
            //ExportMovieCTFCoords.Dispose();

            //Debug.WriteLine(GPU.GetFreeMemory(GPU.GetDevice()));

            // Wait for all async IO to finish
            WriteStackAsync?.Wait();
            //WriteDeconvAsync?.Wait();
            WriteAverageAsync?.Wait();

            OutputTimers[4].Finish(Timer4);
            #endregion

            OptionsMovieExport = options;
            SaveMeta();

            IsProcessing = false;

            lock (OutputTimers)
            {
                if (OutputTimers[0].NItems > 1)
                    foreach (var timer in OutputTimers)
                        Console.WriteLine(timer.Name + ": " + timer.GetAverageMilliseconds(100).ToString("F0"));
            }
        }

        public void ExportParticles(Image originalStack, float2[] positions, ProcessingOptionsParticleExport options)
        {
            IsProcessing = true;

            options.PreflipPhases &= OptionsCTF != null;

            #region Make sure all directories are there

            if (options.DoAverage && !Directory.Exists(ParticlesDir))
                Directory.CreateDirectory(ParticlesDir);

            if (options.DoDenoisingPairs)
            {
                Directory.CreateDirectory(ParticlesDenoisingOddDir);
                Directory.CreateDirectory(ParticlesDenoisingEvenDir);
            }

            #endregion

            #region Helper variables

            int NParticles = positions.Length;

            int3 DimsMovie = originalStack.Dims;
            int FirstFrame = Math.Max(0, Math.Min(DimsMovie.Z - 1, options.SkipFirstN));
            int LastFrameExclusive = Math.Min(DimsMovie.Z, DimsMovie.Z - options.SkipLastN);
            DimsMovie.Z = LastFrameExclusive - FirstFrame;

            int3 DimsParticle = new int3(options.BoxSize, options.BoxSize, NParticles);
            int3 DimsPreflip = DimsParticle.MultXY(2);
            int3 DimsExtraction = options.PreflipPhases ? DimsPreflip : DimsParticle;
            int3 DimsResampled = options.BoxSizeResample > 0 ? new int3(options.BoxSizeResample, options.BoxSizeResample, NParticles).MultXY(options.PreflipPhases ? 2 : 1) : DimsExtraction;

            Task WriteHalfAverageAsync = null;

            float PixelSize = (float)options.BinnedPixelSizeMean;

            #endregion

            #region Figure out where to extract, and how much to shift afterwards

            float3[] ParticleCenters = positions.Select(p => new float3(p) / (float)options.BinnedPixelSizeMean).ToArray(); // From Angstrom to binned pixels

            float3[][] ParticleOrigins = Helper.ArrayOfFunction(z =>
            {
                float Z = (z + FirstFrame) / (float)Math.Max(1, originalStack.Dims.Z - 1);
                return ParticleCenters.Select(p =>
                {
                    float2 LocalShift = GetShiftFromPyramid(new float3(p.X * PixelSize / options.Dimensions.X,                  // ParticleCenters are in binned pixels, Dimensions are in Angstrom
                                                                       p.Y * PixelSize / options.Dimensions.Y, Z)) / PixelSize; // Shifts are in physical Angstrom, convert to binned pixels
                    return new float3(p.X - LocalShift.X - DimsExtraction.X / 2, p.Y - LocalShift.Y - DimsExtraction.Y / 2, 0);
                }).ToArray();
            }, DimsMovie.Z);

            int3[][] ParticleIntegerOrigins = ParticleOrigins.Select(a => a.Select(p => new int3(p.Floor())).ToArray()).ToArray();
            float3[][] ParticleResidualShifts = Helper.ArrayOfFunction(z =>
                                                                       Helper.ArrayOfFunction(i =>
                                                                                              new float3(ParticleIntegerOrigins[z][i]) - ParticleOrigins[z][i],
                                                                                              NParticles),
                                                                       ParticleOrigins.Length);

            #endregion

            #region Pre-calc phase flipping and dose weighting

            Image CTFSign = null;
            Image CTFCoords = CTF.GetCTFCoords(DimsExtraction.X, DimsExtraction.X);
            Image GammaCorrection = CTF.GetGammaCorrection((float)options.BinnedPixelSizeMean, DimsExtraction.X);
            Image[] DoseWeights = null;

            if (options.PreflipPhases || options.DoDenoisingPairs)
            {
                CTFStruct[] Params = positions.Select(p =>
                {
                    CTF Local = CTF.GetCopy();
                    Local.Defocus = (decimal)GridCTFDefocus.GetInterpolated(new float3(p.X / options.Dimensions.X, p.Y / options.Dimensions.Y, 0));
                    return Local.ToStruct();
                }).ToArray();

                CTFSign = new Image(DimsExtraction, true);
                GPU.CreateCTF(CTFSign.GetDevice(Intent.Write),
                              CTFCoords.GetDevice(Intent.Read),
                              GammaCorrection.GetDevice(Intent.Read),
                              (uint)CTFCoords.ElementsSliceComplex,
                              Params,
                              false,
                              (uint)CTFSign.Dims.Z);

                GPU.Sign(CTFSign.GetDevice(Intent.Read),
                         CTFSign.GetDevice(Intent.Write),
                         CTFSign.ElementsReal);
            }

            if (options.DosePerAngstromFrame != 0)
            {
                float DosePerFrame = (float)options.DosePerAngstromFrame;
                if (options.DosePerAngstromFrame < 0)
                    DosePerFrame = -(float)options.DosePerAngstromFrame / originalStack.Dims.Z;

                DoseWeights = Helper.ArrayOfFunction(z =>
                {
                    Image Weights = new Image(IntPtr.Zero, DimsExtraction.Slice(), true);

                    CTF CTFBfac = new CTF
                    {
                        PixelSize = (decimal)PixelSize,
                        Defocus = 0,
                        Amplitude = 1,
                        Cs = 0,
                        Cc = 0,
                        Bfactor = GridDoseBfacs.Values.Length == 1 ? -(decimal)(DosePerFrame * (z + FirstFrame + 0.5f) * 4) : (decimal)GridDoseBfacs.Values[z + FirstFrame],
                        Scale = GridDoseWeights.Values.Length == 1 ? 1 : (decimal)GridDoseWeights.Values[z + FirstFrame]
                    };
                    GPU.CreateCTF(Weights.GetDevice(Intent.Write),
                                  CTFCoords.GetDevice(Intent.Read),
                                  IntPtr.Zero,
                                  (uint)CTFCoords.ElementsSliceComplex,
                                  new[] { CTFBfac.ToStruct() },
                                  false,
                                  1);

                    //if (z % 2 == 1)
                    //    Weights.Multiply(0);

                    return Weights;
                }, DimsMovie.Z);

                Image DoseWeightsSum = new Image(DimsExtraction.Slice(), true);
                foreach (var weights in DoseWeights)
                    DoseWeightsSum.Add(weights);
                DoseWeightsSum.Max(1e-6f);

                foreach (var weights in DoseWeights)
                    weights.Divide(DoseWeightsSum);

                DoseWeightsSum.Dispose();
            }

            GammaCorrection.Dispose();
            CTFCoords.Dispose();

            #endregion

            #region Make FFT plans and memory

            Image Extracted = new Image(IntPtr.Zero, DimsExtraction);
            Image ExtractedFT = new Image(IntPtr.Zero, DimsExtraction, true, true);
            Image ExtractedFTResampled = options.BoxSizeResample > 0 ? new Image(IntPtr.Zero, DimsResampled, true, true) : ExtractedFT;

            Image AverageFT = options.DoAverage ? new Image(DimsResampled, true, true) : null;
            Image[] AverageOddEvenFT = options.DoDenoisingPairs ? Helper.ArrayOfFunction(i => new Image(DimsResampled, true, true), 2) : null;

            int BatchSize = Math.Min(NParticles, 128);
            int PlanForw = GPU.CreateFFTPlan(DimsExtraction.Slice(), (uint)BatchSize);

            int3 DimsBuffer = new int3(Math.Max(DimsExtraction.X, DimsResampled.X), Math.Max(DimsExtraction.Y, DimsResampled.Y), BatchSize);
            Image BufferFFTSource = new Image(IntPtr.Zero, DimsBuffer);
            Image BufferFFTTarget = new Image(IntPtr.Zero, DimsBuffer, true, true);

            #endregion

            #region Extract and process everything

            for (int nframe = 0; nframe < DimsMovie.Z; nframe++)
            {
                GPU.Extract(originalStack.GetDeviceSlice(nframe + FirstFrame, Intent.Read),
                            Extracted.GetDevice(Intent.Write),
                            DimsMovie.Slice(),
                            DimsExtraction.Slice(),
                            Helper.ToInterleaved(ParticleIntegerOrigins[nframe]),
                            false,
                            (uint)NParticles);

                for (int b = 0; b < NParticles; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, NParticles - b);

                    GPU.CopyDeviceToDevice(Extracted.GetDeviceSlice(b, Intent.Read), 
                                           BufferFFTSource.GetDevice(Intent.Write), 
                                           CurBatch * DimsExtraction.ElementsSlice());

                    GPU.FFT(BufferFFTSource.GetDevice(Intent.Read),
                            BufferFFTTarget.GetDevice(Intent.Write),
                            DimsExtraction.Slice(),
                            (uint)BatchSize,
                            PlanForw);

                    GPU.CopyDeviceToDevice(BufferFFTTarget.GetDevice(Intent.Read),
                                           ExtractedFT.GetDeviceSlice(b, Intent.Write),
                                           CurBatch * DimsExtraction.ElementsFFTSlice() * 2);
                }

                if (CTFSign != null)
                    ExtractedFT.Multiply(CTFSign);

                if (options.DosePerAngstromFrame != 0)
                    ExtractedFT.MultiplySlices(DoseWeights[nframe]);

                ExtractedFT.ShiftSlices(ParticleResidualShifts[nframe]);

                if (options.BoxSizeResample > 0)
                    GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                               ExtractedFTResampled.GetDevice(Intent.Write),
                               DimsExtraction.Slice(),
                               DimsResampled.Slice(),
                               (uint)NParticles);

                if (options.DoAverage || options.DoDenoisingPairs)
                {
                    if (options.DoAverage)
                        AverageFT.Add(ExtractedFTResampled);

                    if (options.DoDenoisingPairs)
                        AverageOddEvenFT[nframe % 2].Add(ExtractedFTResampled);
                }
            }

            originalStack.FreeDevice();

            #region Clean up #1

            Extracted.Dispose();
            ExtractedFT.Dispose();
            if (options.BoxSizeResample > 0)
                ExtractedFTResampled.Dispose();
            GPU.DestroyFFTPlan(PlanForw);

            #endregion

            int PlanBack = GPU.CreateIFFTPlan(DimsResampled.Slice(), (uint)BatchSize);

            if (options.DoAverage)
            {
                Image Average = new Image(IntPtr.Zero, DimsResampled);

                for (int b = 0; b < NParticles; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, NParticles - b);

                    GPU.CopyDeviceToDevice(AverageFT.GetDeviceSlice(b, Intent.Read),
                                           BufferFFTTarget.GetDevice(Intent.Write),
                                           CurBatch * DimsResampled.ElementsFFTSlice() * 2);

                    GPU.IFFT(BufferFFTTarget.GetDevice(Intent.Read),
                            BufferFFTSource.GetDevice(Intent.Write),
                            DimsResampled.Slice(),
                            (uint)BatchSize,
                            PlanBack,
                            false);

                    GPU.CopyDeviceToDevice(BufferFFTSource.GetDevice(Intent.Read),
                                           Average.GetDeviceSlice(b, Intent.Write),
                                           CurBatch * DimsResampled.ElementsSlice());
                }

                AverageFT.Dispose();

                Image AverageCropped = Average.AsPadded(new int2(DimsParticle));
                Average.Dispose();

                #region Subtract background plane

                AverageCropped.FreeDevice();
                float[][] AverageData = AverageCropped.GetHost(Intent.ReadWrite);
                for (int p = 0; p < NParticles; p++)
                {
                    float[] ParticleData = AverageData[p];
                    float[] Background = MathHelper.FitAndGeneratePlane(ParticleData, new int2(AverageCropped.Dims));
                    for (int i = 0; i < ParticleData.Length; i++)
                        ParticleData[i] -= Background[i];
                }

                #endregion

                if (options.Normalize)
                    GPU.NormParticles(AverageCropped.GetDevice(Intent.Read),
                                      AverageCropped.GetDevice(Intent.Write),
                                      DimsParticle.Slice(),
                                      (uint)(options.Diameter / PixelSize / 2),
                                      options.Invert,
                                      (uint)NParticles);
                else if (options.Invert)
                    AverageCropped.Multiply(-1f);

                AverageCropped.FreeDevice();

                WriteHalfAverageAsync = new Task(() =>
                {
                    float PixelSizeResampled = (options.BoxSizeResample > 0 ? ((float)options.BoxSize / options.BoxSizeResample) : 1) * (float)options.BinnedPixelSizeMean;
                    AverageCropped.WriteMRC16b(IOPath.Combine(ParticlesDir, RootName + options.Suffix + ".mrcs"), PixelSizeResampled, true);
                    AverageCropped.Dispose();
                });
                WriteHalfAverageAsync.Start();
                GlobalTasks.Add(WriteHalfAverageAsync);
            }

            if (options.DoDenoisingPairs)
            {
                string[] OddEvenDir = { ParticlesDenoisingOddDir, ParticlesDenoisingEvenDir };

                for (int idenoise = 0; idenoise < 2; idenoise++)
                {
                    Image Average = new Image(IntPtr.Zero, AverageOddEvenFT[idenoise].Dims);// AverageOddEvenFT[idenoise].AsIFFT(false, PlanBack, true);

                    for (int b = 0; b < NParticles; b += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - b);

                        GPU.CopyDeviceToDevice(AverageOddEvenFT[idenoise].GetDeviceSlice(b, Intent.Read),
                                               BufferFFTTarget.GetDevice(Intent.Write),
                                               CurBatch * DimsResampled.ElementsFFTSlice() * 2);

                        GPU.IFFT(BufferFFTTarget.GetDevice(Intent.Read),
                                BufferFFTSource.GetDevice(Intent.Write),
                                DimsResampled.Slice(),
                                (uint)BatchSize,
                                PlanBack,
                                false);

                        GPU.CopyDeviceToDevice(BufferFFTSource.GetDevice(Intent.Read),
                                               Average.GetDeviceSlice(b, Intent.Write),
                                               CurBatch * DimsResampled.ElementsSlice());
                    }

                    AverageOddEvenFT[idenoise].Dispose();

                    Image AverageCropped = Average.AsPadded(new int2(DimsParticle));
                    Average.Dispose();

                    #region Subtract background plane

                    AverageCropped.FreeDevice();
                    float[][] AverageData = AverageCropped.GetHost(Intent.ReadWrite);
                    for (int p = 0; p < NParticles; p++)
                    {
                        float[] ParticleData = AverageData[p];
                        float[] Background = MathHelper.FitAndGeneratePlane(ParticleData, new int2(AverageCropped.Dims));
                        for (int i = 0; i < ParticleData.Length; i++)
                            ParticleData[i] -= Background[i];
                    }

                    #endregion

                    if (options.Normalize)
                        GPU.NormParticles(AverageCropped.GetDevice(Intent.Read),
                                          AverageCropped.GetDevice(Intent.Write),
                                          DimsParticle.Slice(),
                                          (uint)(options.Diameter / PixelSize / 2),
                                          options.Invert,
                                          (uint)NParticles);
                    else if (options.Invert)
                        AverageCropped.Multiply(-1f);

                    AverageCropped.FreeDevice();

                    WriteHalfAverageAsync = new Task(() =>
                    {
                        float PixelSizeResampled = (options.BoxSizeResample > 0 ? ((float)options.BoxSize / options.BoxSizeResample) : 1) * (float)options.BinnedPixelSizeMean;
                        AverageCropped.WriteMRC16b(IOPath.Combine(OddEvenDir[idenoise], RootName + options.Suffix + ".mrcs"), PixelSizeResampled, true);
                        AverageCropped.Dispose();
                    });
                    WriteHalfAverageAsync.Start();
                    GlobalTasks.Add(WriteHalfAverageAsync);
                }
            }

            #endregion

            #region Clean up #2

            GPU.DestroyFFTPlan(PlanBack);
            BufferFFTSource.Dispose();
            BufferFFTTarget.Dispose();

            CTFSign?.Dispose();
            if (DoseWeights != null)
                foreach (var doseWeight in DoseWeights)
                    doseWeight.Dispose();

            #endregion

            // Wait for all async IO to finish
            //WriteAverageAsync?.Wait();

            OptionsParticlesExport = options;
            SaveMeta();

            IsProcessing = false;
        }

        #endregion

        #region Offline tasks

        public void MatchFull(Image originalStack, ProcessingOptionsFullMatch options, Image template, Func<int3, int, string, bool> progressCallback)
        {
            bool IsCanceled = false;
            if (!Directory.Exists(MatchingDir))
                Directory.CreateDirectory(MatchingDir);

            string NameWithRes = RootName + $"_{options.BinnedPixelSizeMean:F2}Apx";

            float3[] HealpixAngles = Helper.GetHealpixAngles(options.HealpixOrder, options.Symmetry).Select(a => a * Helper.ToRad).ToArray();

            Image CorrImage = null;
            Image AngleImage = null;
            float[] CorrData;
            float[] AngleData;

            GPU.CheckGPUExceptions();

            #region Dimensions

            int SizeSub = options.SubPatchSize;
            int SizeSubPadded = SizeSub * 2;
            int SizeParticle = (int)(options.TemplateDiameter / options.BinnedPixelSizeMean);
            int SizeUseful = Math.Min(SizeSub / 2, SizeSub - SizeParticle * 2);// Math.Min(SizeSub - SizeParticle, SizeSub / 2);
            if (SizeUseful < 2)
                throw new DimensionMismatchException("Particle diameter is bigger than the box.");

            int3 DimsMicrographCropped = new int3(originalStack.Dims.X, originalStack.Dims.Y, 1);

            int3 Grid = (DimsMicrographCropped - SizeParticle + SizeUseful - 1) / SizeUseful;
            Grid.Z = 1;
            List<float3> GridCoords = new List<float3>();
            for (int y = 0; y < Grid.Y; y++)
                for (int x = 0; x < Grid.X; x++)
                    GridCoords.Add(new float3(x * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                                              y * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                                              0));

            int3 DimsExtraction = new int3(SizeSubPadded, SizeSubPadded, GridCoords.Count);
            int3 DimsParticle = new int3(SizeSub, SizeSub, GridCoords.Count);

            progressCallback?.Invoke(Grid, 0, "Preparing...");

            #endregion

            #region Figure out where to extract, and how much to shift afterwards

            float3[] ParticleCenters = GridCoords.Select(p => new float3(p)).ToArray();

            float3[][] ParticleOrigins = Helper.ArrayOfFunction(z =>
            {
                float Z = z / (float)Math.Max(1, originalStack.Dims.Z - 1);
                return ParticleCenters.Select(p =>
                {
                    float2 LocalShift = GetShiftFromPyramid(new float3(p.X / DimsMicrographCropped.X, p.Y / DimsMicrographCropped.Y, Z)) / (float)options.BinnedPixelSizeMean; // Shifts are in physical Angstrom, convert to binned pixels
                    return new float3(p.X - LocalShift.X - SizeSubPadded / 2, p.Y - LocalShift.Y - SizeSubPadded / 2, 0);
                }).ToArray();
            }, originalStack.Dims.Z);

            int3[][] ParticleIntegerOrigins = ParticleOrigins.Select(a => a.Select(p => new int3(p.Floor())).ToArray()).ToArray();
            float3[][] ParticleResidualShifts = Helper.ArrayOfFunction(z =>
                                                                           Helper.ArrayOfFunction(i =>
                                                                                                      new float3(ParticleIntegerOrigins[z][i]) - ParticleOrigins[z][i],
                                                                                                  GridCoords.Count),
                                                                       ParticleOrigins.Length);

            #endregion

            #region CTF, phase flipping & dose weighting

            Image CTFCoords = CTF.GetCTFCoords(SizeSub, (int)(SizeSub * options.DownsampleFactor));
            Image GammaCorrection = CTF.GetGammaCorrection((float)options.BinnedPixelSizeMean, SizeSub);
            Image CTFCoordsPadded = CTF.GetCTFCoords(SizeSub * 2, (int)(SizeSub * 2 * options.DownsampleFactor));
            Image[] DoseWeights = null;

            #region Dose

            if (options.DosePerAngstromFrame != 0)
            {
                float DosePerFrame = (float)options.DosePerAngstromFrame;
                if (options.DosePerAngstromFrame < 0)
                    DosePerFrame = -(float)options.DosePerAngstromFrame / originalStack.Dims.Z;

                float3 NikoConst = new float3(0.245f, -1.665f, 2.81f);
                Image CTFFreq = CTFCoordsPadded.AsReal();

                GPU.CheckGPUExceptions();

                DoseWeights = Helper.ArrayOfFunction(z =>
                {
                    Image Weights = new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, 1), true);

                    GPU.DoseWeighting(CTFFreq.GetDevice(Intent.Read),
                                      Weights.GetDevice(Intent.Write),
                                      (uint)Weights.ElementsSliceComplex,
                                      new[] { DosePerFrame * z, DosePerFrame * (z + 1) },
                                      NikoConst,
                                      options.Voltage > 250 ? 1 : 0.8f, // It's only defined for 300 and 200 kV, but let's not throw an exception
                                      1);

                    return Weights;
                }, originalStack.Dims.Z);

                GPU.CheckGPUExceptions();

                Image WeightSum = new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, 1), true);
                WeightSum.Fill(1e-15f);
                for (int nframe = 0; nframe < originalStack.Dims.Z; nframe++)
                    WeightSum.Add(DoseWeights[nframe]);
                WeightSum.Multiply(1f / originalStack.Dims.Z);

                GPU.CheckGPUExceptions();

                for (int nframe = 0; nframe < originalStack.Dims.Z; nframe++)
                {
                    DoseWeights[nframe].Divide(WeightSum);
                    //DoseWeights[nframe].WriteMRC($"d_doseweights_{GPU.GetDevice()}_{nframe}.mrc", true);
                }

                GPU.CheckGPUExceptions();

                WeightSum.Dispose();
                CTFFreq.Dispose();

                GPU.CheckGPUExceptions();
            }

            GPU.CheckGPUExceptions();

            #endregion

            #region Create CTF for themplate and padded phase flipping

            Image ExperimentalCTF = new Image(new int3(SizeSub, SizeSub, 1), true);
            Image ExperimentalCTFPadded = new Image(new int3(SizeSub * 2, SizeSub * 2, 1), true);
            CTF CTFParams = CTF.GetCopy();

            GPU.CreateCTF(ExperimentalCTF.GetDevice(Intent.Write),
                          CTFCoords.GetDevice(Intent.Read),
                          GammaCorrection.GetDevice(Intent.Read),
                          (uint)CTFCoords.ElementsComplex,
                          new[] { CTFParams.ToStruct() },
                          false,
                          1);
            ExperimentalCTF.Abs();

            GPU.CreateCTF(ExperimentalCTFPadded.GetDevice(Intent.Write),
                          CTFCoordsPadded.GetDevice(Intent.Read),
                          GammaCorrection.GetDevice(Intent.Read),
                          (uint)CTFCoordsPadded.ElementsComplex,
                          new[] { CTFParams.ToStruct() },
                          false,
                          1);
            ExperimentalCTFPadded.Sign();

            #endregion

            #endregion

            #region Whiten spectrum in images

            Image FlatteningFactors = new Image(new int3(SizeSub, SizeSub, 1), true);
            FlatteningFactors.Fill(1f);

            if (options.WhitenSpectrum)
            {
                progressCallback?.Invoke(Grid, 0, "Whitening spectral noise...");

                Image OriginalStackFlat = originalStack.AsSpectrumFlattened(false, 0.99f, 256);
                float[] AS1D = originalStack.AsAmplitudes1D(false, 0.99f, SizeSub / 2);
                originalStack.FreeDevice();
                originalStack = OriginalStackFlat;

                float[] FlatteningFactorsData = FlatteningFactors.GetHost(Intent.Write)[0];
                Helper.ForEachElementFT(new int2(FlatteningFactors.Dims), (x, y, xx, yy) =>
                {
                    int R = (int)((float)Math.Sqrt(xx * xx + yy * yy) / (SizeSub / 2) * AS1D.Length);
                    R = Math.Min(AS1D.Length - 1, R);

                    FlatteningFactorsData[y * (SizeSub / 2 + 1) + x] = AS1D[R] > 0 ? 1 / AS1D[R] : 0;
                });
            }

            ExperimentalCTF.MultiplySlices(FlatteningFactors);
            FlatteningFactors.Dispose();

            #endregion

            #region Extract and preprocess all patches

            Image AllPatchesFT;
            {
                Image AverageFT = new Image(DimsExtraction, true, true);
                Image Extracted = new Image(IntPtr.Zero, DimsExtraction);
                Image ExtractedFT = new Image(IntPtr.Zero, DimsExtraction, true, true);

                int PlanForw = GPU.CreateFFTPlan(DimsExtraction.Slice(), (uint)GridCoords.Count);
                int PlanBack = GPU.CreateIFFTPlan(DimsExtraction.Slice(), (uint)GridCoords.Count);

                for (int nframe = 0; nframe < originalStack.Dims.Z; nframe++)
                {
                    GPU.Extract(originalStack.GetDeviceSlice(nframe, Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                originalStack.Dims.Slice(),
                                DimsExtraction.Slice(),
                                Helper.ToInterleaved(ParticleIntegerOrigins[nframe]),
                                false,
                                (uint)GridCoords.Count);

                    GPU.FFT(Extracted.GetDevice(Intent.Read),
                            ExtractedFT.GetDevice(Intent.Write),
                            DimsExtraction.Slice(),
                            (uint)GridCoords.Count,
                            PlanForw);

                    ExtractedFT.MultiplySlices(ExperimentalCTFPadded);

                    ExtractedFT.ShiftSlices(ParticleResidualShifts[nframe]);

                    if (options.DosePerAngstromFrame != 0)
                        ExtractedFT.MultiplySlices(DoseWeights[nframe]);

                    AverageFT.Add(ExtractedFT);
                }

                Image Average = AverageFT.AsIFFT(false, PlanBack, true).AndDisposeParent();

                Image AllPatches = Average.AsPadded(new int2(DimsParticle));
                Average.Dispose();

                GPU.Normalize(AllPatches.GetDevice(Intent.Read),
                              AllPatches.GetDevice(Intent.Write),
                              (uint)AllPatches.ElementsSliceReal,
                              (uint)GridCoords.Count);
                if (options.Invert)
                    AllPatches.Multiply(-1f);

                AllPatchesFT = AllPatches.AsFFT();
                AllPatches.Dispose();

                GPU.DestroyFFTPlan(PlanBack);
                GPU.DestroyFFTPlan(PlanForw);
                Extracted.Dispose();
                ExtractedFT.Dispose();
            }

            #endregion

            originalStack.FreeDevice();

            #region Get correlation and angles

            //if (false)
            {
                #region Scale and pad/crop the template to the right size, create projector

                progressCallback?.Invoke(Grid, 0, "Preparing template...");

                Projector ProjectorReference;
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

                    Image TemplatePadded = TemplateScaled.AsPadded(new int3(SizeSub));
                    TemplateScaled.Dispose();

                    ProjectorReference = new Projector(TemplatePadded, 1, true, 2);
                    TemplatePadded.Dispose();
                    ProjectorReference.PutTexturesOnDevice();
                }

                #endregion

                CorrData = new float[DimsMicrographCropped.ElementsSlice()];
                AngleData = new float[DimsMicrographCropped.ElementsSlice()];

                progressCallback?.Invoke(Grid, 0, "Matching...");

                int BatchSize = 1;
                for (int b = 0; b < GridCoords.Count; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, GridCoords.Count - b);

                    #region Perform correlation

                    Image BestCorrelation = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, CurBatch));
                    Image BestAngle = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, CurBatch));

                    GPU.CorrelateSubTomos(ProjectorReference.t_DataRe,
                                          ProjectorReference.t_DataIm,
                                          ProjectorReference.Oversampling,
                                          ProjectorReference.Data.Dims,
                                          AllPatchesFT.GetDeviceSlice(b, Intent.Read),
                                          ExperimentalCTF.GetDevice(Intent.Read),
                                          new int3(SizeSub, SizeSub, 1),
                                          (uint)CurBatch,
                                          Helper.ToInterleaved(HealpixAngles),
                                          (uint)HealpixAngles.Length,
                                          (uint)128,
                                          SizeParticle / 2,
                                          BestCorrelation.GetDevice(Intent.Write),
                                          BestAngle.GetDevice(Intent.Write),
                                          null);

                    #endregion

                    #region Put correlation values and best angle IDs back into the large volume

                    for (int st = 0; st < CurBatch; st++)
                    {
                        Image ThisCorrelation = new Image(BestCorrelation.GetDeviceSlice(st, Intent.Read), new int3(SizeSub, SizeSub, 1));
                        Image CroppedCorrelation = ThisCorrelation.AsPadded(new int2(SizeUseful));

                        Image ThisAngle = new Image(BestAngle.GetDeviceSlice(st, Intent.Read), new int3(SizeSub, SizeSub, 1));
                        Image CroppedAngle = ThisAngle.AsPadded(new int2(SizeUseful));

                        float[] SubCorr = CroppedCorrelation.GetHostContinuousCopy();
                        float[] SubAngle = CroppedAngle.GetHostContinuousCopy();
                        int3 Origin = new int3(GridCoords[b + st]) - SizeUseful / 2;
                        for (int y = 0; y < SizeUseful; y++)
                        {
                            int yVol = Origin.Y + y;
                            if (yVol >= DimsMicrographCropped.Y - SizeParticle / 2)
                                continue;

                            for (int x = 0; x < SizeUseful; x++)
                            {
                                int xVol = Origin.X + x;
                                if (xVol >= DimsMicrographCropped.X - SizeParticle / 2)
                                    continue;

                                CorrData[yVol * DimsMicrographCropped.X + xVol] = SubCorr[y * SizeUseful + x];// / (SizeSub * SizeSub);
                                AngleData[yVol * DimsMicrographCropped.X + xVol] = SubAngle[y * SizeUseful + x];
                            }
                        }

                        CroppedCorrelation.Dispose();
                        ThisCorrelation.Dispose();
                        CroppedAngle.Dispose();
                        ThisAngle.Dispose();
                    }

                    #endregion

                    BestCorrelation.Dispose();
                    BestAngle.Dispose();

                    if (progressCallback != null)
                        IsCanceled = progressCallback(Grid, b + CurBatch, "Matching...");
                }

                #region Postflight

                CTFCoords.Dispose();
                GammaCorrection.Dispose();
                CTFCoordsPadded.Dispose();
                ExperimentalCTF.Dispose();
                ExperimentalCTFPadded.Dispose();
                AllPatchesFT.Dispose();
                ProjectorReference.Dispose();

                if (options.Supersample > 1)
                {
                    progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Looking for sub-pixel peaks...");

                    Image NormalSampled = new Image(CorrData, DimsMicrographCropped);
                    Image SuperSampled = new Image(NormalSampled.GetDevice(Intent.Read), NormalSampled.Dims);

                    GPU.SubpixelMax(NormalSampled.GetDevice(Intent.Read),
                                    SuperSampled.GetDevice(Intent.Write),
                                    NormalSampled.Dims,
                                    options.Supersample);

                    CorrData = SuperSampled.GetHost(Intent.Read)[0];

                    NormalSampled.Dispose();
                    SuperSampled.Dispose();
                }

                //if (options.KeepOnlyFullVoxels)
                {
                    progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Trimming...");

                    float Margin = (float)options.TemplateDiameter / (float)options.BinnedPixelSizeMean;

                    for (int y = 0; y < DimsMicrographCropped.Y; y++)
                        for (int x = 0; x < DimsMicrographCropped.X; x++)
                        {
                            if (x < Margin || y < Margin ||
                                x > DimsMicrographCropped.X - Margin ||
                                y > DimsMicrographCropped.Y - Margin)
                            {
                                CorrData[y * DimsMicrographCropped.X + x] = 0;
                            }
                        }
                }

                progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Saving global scores...");

                // Store correlation values and angle IDs for re-use later
                CorrImage = new Image(CorrData, DimsMicrographCropped);
                CorrImage.WriteMRC(IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + "_corr.mrc"), (float)options.BinnedPixelSizeMean, true);

                #endregion
            }

            #endregion

            #region Get peak list that has at most nPeaks values

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Extracting best peaks...");

            int3[] InitialPeaks = new int3[0];
            {
                float2 MeanAndStd = MathHelper.MeanAndStdNonZero(CorrImage.GetHostContinuousCopy());

                for (float s = 4; s > 0.5f; s -= 0.05f)
                {
                    float Threshold = MeanAndStd.X + MeanAndStd.Y * s;
                    InitialPeaks = CorrImage.GetLocalPeaks(SizeParticle * 2 / 3, Threshold);

                    if (InitialPeaks.Length >= options.NResults)
                        break;
                }
            }

            CorrImage.Dispose();

            #endregion

            #region Write peak positions and angles into table

            Star TableOut = new Star(new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnMicrographName",
                "rlnAutopickFigureOfMerit"
            });

            {
                for (int n = 0; n < Math.Min(InitialPeaks.Length, options.NResults); n++)
                {
                    float3 Position = new float3(InitialPeaks[n]) * (float)options.DownsampleFactor;
                    float Score = CorrData[DimsMicrographCropped.ElementFromPosition(InitialPeaks[n])];
                    float3 Angle = HealpixAngles[(int)AngleData[DimsMicrographCropped.ElementFromPosition(InitialPeaks[n])]] * Helper.ToDeg;

                    TableOut.AddRow(new string[]
                    {
                        Position.X.ToString(CultureInfo.InvariantCulture),
                        Position.Y.ToString(CultureInfo.InvariantCulture),
                        Angle.X.ToString(CultureInfo.InvariantCulture),
                        Angle.Y.ToString(CultureInfo.InvariantCulture),
                        Angle.Z.ToString(CultureInfo.InvariantCulture),
                        RootName + ".mrc",
                        Score.ToString(CultureInfo.InvariantCulture)
                    });
                }
            }

            TableOut.Save(IOPath.Combine(MatchingDir, NameWithRes + "_" + options.TemplateName + ".star"));
            UpdateParticleCount("_" + options.TemplateName);

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Done.");

            #endregion
        }

        #endregion

        #region Multi-particle refinement

        public virtual void PerformMultiParticleRefinement(string workingDirectory,
                                                           ProcessingOptionsMPARefine optionsMPA,
                                                           Species[] allSpecies,
                                                           DataSource dataSource,
                                                           Image gainRef,
                                                           DefectModel defectMap,
                                                           Action<string> progressCallback)
        {
            int GPUID = GPU.GetDevice();
            HeaderEER.GroupNFrames = dataSource.EERGroupFrames;

            NFrames = Math.Min(MapHeader.ReadFromFile(DataPath).Dimensions.Z, dataSource.FrameLimit);
            //NFrames = 1;
            FractionFrames = (float)NFrames / MapHeader.ReadFromFile(DataPath).Dimensions.Z;
            int RedNFrames = NFrames;

            float BfactorWeightingThreshold = (float)optionsMPA.BFactorWeightingThreshold;

            //MagnificationCorrection = new float3(1, 1, 0);
            //CTF.BeamTilt = new float2(0, 0);

            if (CTF.ZernikeCoeffsOdd == null)
                CTF.ZernikeCoeffsOdd = new float[12];
            else if (CTF.ZernikeCoeffsOdd.Length < 12)
                CTF.ZernikeCoeffsOdd = Helper.Combine(CTF.ZernikeCoeffsOdd, new float[12 - CTF.ZernikeCoeffsOdd.Length]);

            if (CTF.ZernikeCoeffsEven == null)
                CTF.ZernikeCoeffsEven = new float[8];
            else if (CTF.ZernikeCoeffsEven.Length < 8)
                CTF.ZernikeCoeffsEven = Helper.Combine(CTF.ZernikeCoeffsEven, new float[8 - CTF.ZernikeCoeffsEven.Length]);

            //GridLocationWeights = new CubicGrid(new int3(1), new float[] { 1f });
            //GridLocationBfacs = new CubicGrid(new int3(1), new float[] { 0f });

            //GridAngleX = new CubicGrid(new int3(1), new float[1]);
            //GridAngleY = new CubicGrid(new int3(1), new float[1]);
            //GridAngleZ = new CubicGrid(new int3(1), new float[1]);

            #region Get particles belonging to this item; if there are none, abort

            string DataHash = GetDataHash();

            Dictionary<Species, Particle[]> SpeciesParticles = new Dictionary<Species, Particle[]>();
            foreach (var species in allSpecies)
                SpeciesParticles.Add(species, species.GetParticles(DataHash));

            if (SpeciesParticles.Select(p => p.Value.Length).Sum() < optionsMPA.MinParticlesPerItem)
                return;

            //float2 ParticleCenter = MathHelper.Mean(SpeciesParticles[allSpecies[0]].Select(p => new float2(p.Coordinates[0].X, p.Coordinates[0].Y)));
            ////SpeciesParticles[allSpecies[0]] = SpeciesParticles[allSpecies[0]].Where(p => p.Coordinates[0].X >= ParticleCenter.X && p.Coordinates[0].Y <= ParticleCenter.Y).ToArray();
            //List<Particle> ParticlesSorted = SpeciesParticles[allSpecies[0]].ToList();
            //Dictionary<Particle, float> ParticleDistances = new Dictionary<Particle, float>();
            //foreach (var p in ParticlesSorted)
            //    ParticleDistances.Add(p, (ParticleCenter - new float2(p.Coordinates[0].X, p.Coordinates[0].Y)).Length());
            //ParticlesSorted.Sort((a, b) => ParticleDistances[a].CompareTo(ParticleDistances[b]));

            ////SpeciesParticles[allSpecies[0]] = ParticlesSorted.Skip(1).Take(1).ToArray();

            #endregion

            float4[] ParticleMags = Helper.ArrayOfConstant(new float4(0, 0, 0, 0), SpeciesParticles[allSpecies[0]].Length);

            #region Figure out dimensions

            {
                MapHeader Header = MapHeader.ReadFromFile(DataPath);
                ImageDimensionsPhysical = new float2(new int2(Header.Dimensions)) * (float)dataSource.PixelSizeMean;
                if (Header.GetType() == typeof(HeaderEER) && gainRef != null)
                    ImageDimensionsPhysical = new float2(new int2(gainRef.Dims)) * (float)dataSource.PixelSizeMean;
            }

            float SmallestAngPix = MathHelper.Min(allSpecies.Select(s => (float)s.PixelSize));
            float LargestBox = MathHelper.Max(allSpecies.Select(s => s.DiameterAngstrom)) * 2 / SmallestAngPix;

            float[] DoseInterpolationSteps = Helper.ArrayOfFunction(i => (float)i / Math.Max(1, NFrames - 1), NFrames);

            #endregion

            #region Load and preprocess frames

            progressCallback("Loading movie frame data...");

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
                                        1),

                Invert = true,
                NormalizeInput = true,
                NormalizeOutput = false,

                PrerotateParticles = true
            };

            Image[] FrameData;
            LoadFrameData(OptionsDataLoad, gainRef, defectMap, out FrameData);

            float2 AverageMeanStd;
            {
                Image Average = new Image(FrameData[0].Dims);
                foreach (var frame in FrameData)
                {
                    Average.Add(frame);
                    frame.FreeDevice();
                }
                Average.Multiply(1f / FrameData.Length);

                float[] MeanPlaneData = MathHelper.FitAndGeneratePlane(Average.GetHost(Intent.Read)[0], new int2(Average.Dims));
                Image MeanPlane = new Image(MeanPlaneData, Average.Dims);
                Image PaddedFrame = new Image(IntPtr.Zero, Average.Dims + new int3(2048, 2048, 1));

                Average.Fill(0f);

                foreach (var frame in FrameData)
                {
                    frame.Subtract(MeanPlane);

                    GPU.PadClamped(frame.GetDevice(Intent.Read),
                                   PaddedFrame.GetDevice(Intent.Write),
                                   frame.Dims,
                                   PaddedFrame.Dims,
                                   1);

                    PaddedFrame.Bandpass(1f / LargestBox, 1f, false, 0f);

                    GPU.Pad(PaddedFrame.GetDevice(Intent.Read),
                            frame.GetDevice(Intent.Write),
                            PaddedFrame.Dims,
                            frame.Dims,
                            1);

                    Average.Add(frame);

                    frame.FreeDevice();
                }

                AverageMeanStd = MathHelper.MeanAndStd(Average.GetHost(Intent.Read)[0]);
                Average.Dispose();
                MeanPlane.Dispose();
                PaddedFrame.Dispose();
            }

            for (int z = 0; z < FrameData.Length; z++)
            {
                FrameData[z].Add(-AverageMeanStd.X);
                FrameData[z].Multiply(-1f / AverageMeanStd.Y);

                FrameData[z].FreeDevice();
            }

            if (false)
            {
                Image Average = new Image(FrameData[0].Dims);
                foreach (var frame in FrameData)
                    Average.Add(frame);
                Average.Multiply(1f / FrameData.Length);

                if (GPUID == 0)
                    Average.WriteMRC("d_avg.mrc", true);
                Average.Dispose();
            }

            #endregion

            #region Compose optimization steps based on user's requests

            var OptimizationStepsWarp = new List<(WarpOptimizationTypes Type, int Iterations, string Name)>();
            {
                OptimizationStepsWarp.Add((WarpOptimizationTypes.ParticleMag, 5, "particle magnification"));

                if (optionsMPA.DoMagnification)
                {
                    OptimizationStepsWarp.Add((WarpOptimizationTypes.Magnification, 5, "magnification"));
                    //OptimizationStepsWarp.Add((WarpOptimizationTypes.Magnification | WarpOptimizationTypes.ParticlePosition | WarpOptimizationTypes.ParticleAngle, 10, "magnification"));
                }
                {
                    WarpOptimizationTypes TranslationComponents = 0;
                    if (optionsMPA.DoImageWarp)
                        TranslationComponents |= WarpOptimizationTypes.ImageWarp;

                    if (TranslationComponents != 0)
                        OptimizationStepsWarp.Add((TranslationComponents, 10, "image warping"));
                }
                {
                    WarpOptimizationTypes AntisymComponents = 0;

                    if (optionsMPA.DoZernike13)
                        AntisymComponents |= WarpOptimizationTypes.Zernike13;
                    if (optionsMPA.DoZernike5)
                        AntisymComponents |= WarpOptimizationTypes.Zernike5;

                    if (AntisymComponents != 0 && allSpecies.Any(s => s.ResolutionRefinement < (float)optionsMPA.MinimumCTFRefinementResolution))
                        OptimizationStepsWarp.Add((AntisymComponents | WarpOptimizationTypes.ParticlePosition, 20, "antisymmetrical aberrations"));
                }
                {
                    if (optionsMPA.DoDoming)
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
            }


            var OptimizationStepsCTF = new List<(CTFOptimizationTypes Type, int Iterations, string Name)>();
            {
                //if (optionsMPA.DoMagnification)
                //    OptimizationStepsCTF.Add((CTFOptimizationTypes.Distortion, 5, "Distortion"));

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
                //if (optionsMPA.DoMagnification)
                //    DefocusComponents |= CTFOptimizationTypes.Distortion;

                if (optionsMPA.DoDoming)
                    DefocusComponents |= CTFOptimizationTypes.Doming;

                if (DefocusComponents != 0)
                    OptimizationStepsCTF.Add((DefocusComponents, 10, "CTF parameters"));

                if (optionsMPA.DoCs)
                    OptimizationStepsCTF.Add((CTFOptimizationTypes.Cs, 10, "Cs"));

                CTFOptimizationTypes ZernikeComponents = 0;

                if (optionsMPA.DoZernike2)
                    ZernikeComponents |= CTFOptimizationTypes.Zernike2;
                if (optionsMPA.DoZernike4)
                    ZernikeComponents |= CTFOptimizationTypes.Zernike4;

                if (ZernikeComponents != 0)
                    OptimizationStepsCTF.Add((ZernikeComponents, 10, "symmetrical aberrations"));
            }

            #endregion

            if (optionsMPA.NIterations > 0)
            {
                #region Resize grids

                int2 MovementSpatialDims = new int2(optionsMPA.ImageWarpWidth, optionsMPA.ImageWarpHeight);

                if (optionsMPA.DoImageWarp)
                    if (PyramidShiftX.Count == 0 ||
                        PyramidShiftX[0].Dimensions.X != MovementSpatialDims.X ||
                        PyramidShiftX[0].Dimensions.Y != MovementSpatialDims.Y)
                    {
                        PyramidShiftX.Clear();
                        PyramidShiftY.Clear();

                        float OverallDose = (float)(dataSource.DosePerAngstromFrame < 0 ?
                                                    -dataSource.DosePerAngstromFrame :
                                                    dataSource.DosePerAngstromFrame * NFrames);

                        int NTemporal = (int)Math.Min(MathF.Ceiling(OverallDose), NFrames);

                        while (true)
                        {
                            PyramidShiftX.Add(new CubicGrid(new int3(MovementSpatialDims.X, MovementSpatialDims.Y, NTemporal)));
                            PyramidShiftY.Add(new CubicGrid(new int3(MovementSpatialDims.X, MovementSpatialDims.Y, NTemporal)));

                            MovementSpatialDims *= 2;
                            NTemporal = (NTemporal + 3) / 4;
                            if (NTemporal < 3)
                                break;
                        }
                    }

                int AngleSpatialDim = 3;

                if (optionsMPA.DoDoming)
                    if (GridAngleX == null || GridAngleX.Dimensions.X < AngleSpatialDim || GridAngleX.Dimensions.Z != NFrames)
                    {
                        GridAngleX = GridAngleX == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NFrames)) :
                                                          GridAngleX.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NFrames));
                        GridAngleY = GridAngleY == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NFrames)) :
                                                          GridAngleY.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NFrames));
                        GridAngleZ = GridAngleZ == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NFrames)) :
                                                          GridAngleZ.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NFrames));
                    }

                int DomingSpatialDim = 3;
                if (GridCTFDoming == null || GridCTFDoming.Dimensions.X < DomingSpatialDim || GridCTFDoming.Dimensions.Z != NFrames)
                {
                    GridCTFDoming = GridCTFDoming == null ? new CubicGrid(new int3(DomingSpatialDim, DomingSpatialDim, NFrames)) :
                                                            GridCTFDoming.Resize(new int3(DomingSpatialDim, DomingSpatialDim, NFrames));
                }

                if (GridDoseBfacs != null && GridDoseBfacs.Values.Length != NFrames)
                    GridDoseBfacs = new CubicGrid(new int3(1, 1, NFrames),
                                                  Helper.ArrayOfFunction(i => -(i + 0.5f) * (float)dataSource.DosePerAngstromFrame * 4, NFrames));

                //if (GridCTFDefocusDelta.Values.Length <= 1)
                {
                    //GridCTFDefocusDelta = new CubicGrid(new int3(3, 3, 1), Helper.ArrayOfConstant((float)CTF.DefocusDelta, 9));
                    //GridCTFDefocusAngle = new CubicGrid(new int3(3, 3, 1), Helper.ArrayOfConstant((float)CTF.DefocusAngle, 9));

                    if (GridCTFDefocusDelta == null || (GridCTFDefocusDelta.Dimensions.Elements() == 1 && GridCTFDefocusDelta.Values[0] == 0))
                    {
                        GridCTFDefocusDelta = new CubicGrid(new int3(3, 3, 1), Helper.ArrayOfConstant((float)CTF.DefocusDelta, 3 * 3));
                        GridCTFDefocusAngle = new CubicGrid(new int3(3, 3, 1), Helper.ArrayOfConstant((float)CTF.DefocusAngle, 3 * 3));
                    }
                    else
                    {
                        GridCTFDefocusDelta = GridCTFDefocusDelta.Resize(new int3(3, 3, 1));
                        GridCTFDefocusAngle = GridCTFDefocusAngle.Resize(new int3(3, 3, 1));
                    }
                }

                if (GridCTFCs == null || GridCTFCs.Dimensions.Elements() == 1)
                {
                    Console.WriteLine("Initialized Cs grid");
                    GridCTFCs = new CubicGrid(new int3(3, 3, 1));
                }

                #endregion

                #region Create species prerequisites and calculate spectral weights

                progressCallback("Extracting particles...");

                Dictionary<Species, IntPtr[]> SpeciesParticleImages = new Dictionary<Species, IntPtr[]>();
                Dictionary<Species, IntPtr[]> SpeciesParticleQImages = new Dictionary<Species, IntPtr[]>();
                Dictionary<Species, float[]> SpeciesParticleDefoci = new Dictionary<Species, float[]>();
                Dictionary<Species, float2[]> SpeciesParticleExtractedAt = new Dictionary<Species, float2[]>();
                Dictionary<Species, float[]> SpeciesParticleExtractedAtDefocus = new Dictionary<Species, float[]>();
                Dictionary<Species, Image> SpeciesFrameWeights = new Dictionary<Species, Image>();
                Dictionary<Species, Image> SpeciesCTFWeights = new Dictionary<Species, Image>();
                Dictionary<Species, IntPtr> SpeciesParticleSubsets = new Dictionary<Species, IntPtr>();
                Dictionary<Species, (int Start, int End)> SpeciesParticleIDRanges = new Dictionary<Species, (int Start, int End)>();
                Dictionary<Species, int> SpeciesRefinementSize = new Dictionary<Species, int>();
                Dictionary<Species, int[]> SpeciesRelevantRefinementSizes = new Dictionary<Species, int[]>();
                Dictionary<Species, int> SpeciesCTFSuperresFactor = new Dictionary<Species, int>();

                Dictionary<Species, Image> CurrentWeightsDict = SpeciesFrameWeights;

                float[] AverageSpectrum1DAll = new float[128];
                long[] AverageSpectrum1DAllSamples = new long[128];

                int NParticlesOverall = 0;

                Action AllocateParticleMemory = () =>
                {
                    foreach (var species in allSpecies)
                    {
                        if (SpeciesParticles[species].Length == 0)
                            continue;

                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;

                        int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                        IntPtr[] ImagesFTPinned = Helper.ArrayOfFunction(t =>
                        {
                            long Footprint = (new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles;
                            if (optionsMPA.UseHostMemory)
                                return GPU.MallocHostPinned(Footprint);
                            else
                                return GPU.MallocDevice(Footprint);
                        }, RedNFrames);

                        SpeciesParticleImages[species] = ImagesFTPinned;

                        IntPtr[] ImagesFTQPinned = null;
                        if (species.DoEwald)
                        {
                            ImagesFTQPinned = Helper.ArrayOfFunction(t => GPU.MallocDevice((new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles), RedNFrames);

                            SpeciesParticleQImages[species] = ImagesFTQPinned;
                        }
                    }
                };

                Action FreeParticleMemory = () =>
                {
                    foreach (var species in allSpecies)
                    {
                        if (SpeciesParticleImages.ContainsKey(species) && SpeciesParticleImages[species] != null)
                        {
                            if (optionsMPA.UseHostMemory)
                                foreach (var item in SpeciesParticleImages[species])
                                    GPU.FreeHostPinned(item);
                            else
                                foreach (var item in SpeciesParticleImages[species])
                                    GPU.FreeDevice(item);

                            SpeciesParticleImages[species] = null;

                            if (species.DoEwald)
                            {
                                foreach (var item in SpeciesParticleQImages[species])
                                    GPU.FreeDevice(item);

                                SpeciesParticleQImages[species] = null;
                            }
                        }
                    }
                };

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

                    int[] PlanForw = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeFull, SizeFull, 1), (uint)NFrames), 8);

                    int[] RelevantSizes = GetRelevantImageSizes(SizeFull, BfactorWeightingThreshold).Select(v => Math.Min(Size, v)).ToArray();

                    #region Extract particle images

                    Image AverageAmplitudes = new Image(new int3(SizeFull, SizeFull, 1), true);

                    {
                        Image[] AverageAmplitudesThreads = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, 1), true), PlanForw.Length);

                        Image[] Images = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, NFrames)), PlanForw.Length);
                        Image[] ImagesFT = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, NFrames), true, true), PlanForw.Length);
                        Image[] ReducedFT = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, 1), true, true), PlanForw.Length);
                        Image[] ImagesAmplitudes = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, 1), true), PlanForw.Length);

                        GPU.CheckGPUExceptions();

                        Helper.ForCPU(0, NParticles, PlanForw.Length, (threadID) => GPU.SetDevice(GPUID), (p, threadID) =>
                        {
                            GetImagesForOneParticle(OptionsDataLoad,
                                                    FrameData,
                                                    SizeFull,
                                                    Particles[p].GetCoordinateSeries(DoseInterpolationSteps),
                                                    PlanForw[threadID],
                                                    ParticleDiameterPix,
                                                    16,
                                                    true,
                                                    Images[threadID],
                                                    ImagesFT[threadID]);

                            GPU.ReduceMean(ImagesFT[threadID].GetDevice(Intent.Read),
                                            ReducedFT[threadID].GetDevice(Intent.Write),
                                            (uint)ImagesFT[threadID].ElementsSliceReal,
                                            (uint)ImagesFT[threadID].Dims.Z,
                                            1);

                            GPU.Amplitudes(ReducedFT[threadID].GetDevice(Intent.Read),
                                            ImagesAmplitudes[threadID].GetDevice(Intent.Write),
                                            (uint)ReducedFT[threadID].ElementsComplex);

                            ImagesAmplitudes[threadID].Multiply(ImagesAmplitudes[threadID]);

                            AverageAmplitudesThreads[threadID].Add(ImagesAmplitudes[threadID]);
                        }, null);

                        for (int i = 0; i < PlanForw.Length; i++)
                        {
                            AverageAmplitudes.Add(AverageAmplitudesThreads[i]);

                            AverageAmplitudesThreads[i].Dispose();
                            Images[i].Dispose();
                            ImagesFT[i].Dispose();
                            ReducedFT[i].Dispose();
                            ImagesAmplitudes[i].Dispose();
                        }
                    }

                    for (int i = 0; i < PlanForw.Length; i++)
                        GPU.DestroyFFTPlan(PlanForw[i]);

                    #endregion

                    #region Calculate spectrum

                    AverageAmplitudes.Multiply(1f / NParticles);
                    //if (GPUID == 0)
                    //    AverageAmplitudes.WriteMRC("d_avgamps.mrc", true);

                    float[] Amps1D = new float[Size / 2];
                    float[] Samples1D = new float[Size / 2];
                    float[][] Amps2D = AverageAmplitudes.GetHost(Intent.Read);

                    Helper.ForEachElementFT(new int2(SizeFull), (x, y, xx, yy, r, angle) =>
                    {
                        //int idx = (int)Math.Round(r);
                        //if (idx < Size / 2)
                        //{
                        //    Amps1D[idx] += Amps2D[0][y * (Size / 2 + 1) + x];
                        //    Samples1D[idx]++;
                        //}
                        int idx = (int)Math.Round(r / (SizeFull / 2) * AverageSpectrum1DAll.Length);
                        if (idx < AverageSpectrum1DAll.Length)
                        {
                            AverageSpectrum1DAll[idx] += Amps2D[0][y * (SizeFull / 2 + 1) + x] * NParticles;
                            AverageSpectrum1DAllSamples[idx] += NParticles;
                        }
                    });

                    //for (int i = 0; i < Amps1D.Length; i++)
                    //    Amps1D[i] = Amps1D[i] / Samples1D[i];

                    //float Amps1DMean = MathHelper.Mean(Amps1D);
                    //for (int i = 0; i < Amps1D.Length; i++)
                    //    Amps1D[i] = Amps1D[i] / Amps1DMean;

                    AverageAmplitudes.Dispose();

                    #endregion

                    #region Defoci and extraction positions

                    float[] Defoci = new float[NParticles * NFrames];
                    float2[] ExtractedAt = new float2[NParticles * NFrames];
                    float[] ExtractedAtDefocus = new float[NParticles * NFrames];

                    for (int p = 0; p < NParticles; p++)
                    {
                        float3[] Positions = GetPositionInAllFrames(Particles[p].GetCoordinateSeries(DoseInterpolationSteps));
                        for (int f = 0; f < NFrames; f++)
                        {
                            Defoci[p * NFrames + f] = Positions[f].Z;
                            ExtractedAt[p * NFrames + f] = new float2(Positions[f].X, Positions[f].Y);
                            ExtractedAtDefocus[p * NFrames + f] = Positions[f].Z;
                        }
                    }

                    #endregion

                    #region Subset indices

                    int[] Subsets = Particles.Select(p => p.RandomSubset).ToArray();
                    IntPtr SubsetsPtr = GPU.MallocDeviceFromHostInt(Subsets, Subsets.Length);

                    #endregion

                    #region CTF superres factor

                    CTF MaxDefocusCTF = CTF.GetCopy();
                    int MinimumBoxSize = Math.Max(species.HalfMap1Projector[GPUID].Dims.X, MaxDefocusCTF.GetAliasingFreeSize(species.ResolutionRefinement, ParticleDiameterPix));
                    float CTFSuperresFactor = (float)Math.Ceiling((float)MinimumBoxSize / species.HalfMap1Projector[GPUID].Dims.X) + 1e-3f;

                    #endregion

                    SpeciesParticleImages.Add(species, null);
                    if (species.DoEwald)
                        SpeciesParticleQImages.Add(species, null);
                    SpeciesParticleDefoci.Add(species, Defoci);
                    SpeciesParticleExtractedAt.Add(species, ExtractedAt);
                    SpeciesParticleExtractedAtDefocus.Add(species, ExtractedAtDefocus);
                    SpeciesParticleSubsets.Add(species, SubsetsPtr);
                    SpeciesRefinementSize.Add(species, Size);
                    SpeciesRelevantRefinementSizes.Add(species, RelevantSizes);
                    SpeciesCTFSuperresFactor.Add(species, (int)CTFSuperresFactor);

                    species.HalfMap1Projector[GPUID].PutTexturesOnDevice();
                    species.HalfMap2Projector[GPUID].PutTexturesOnDevice();
                }

                #region Calculate 1D PS averaged over all species and particles

                {
                    for (int i = 0; i < AverageSpectrum1DAll.Length; i++)
                        AverageSpectrum1DAll[i] /= Math.Max(1, AverageSpectrum1DAllSamples[i]);

                    float SpectrumMean = MathHelper.Mean(AverageSpectrum1DAll);
                    for (int i = 0; i < AverageSpectrum1DAll.Length; i++)
                        AverageSpectrum1DAll[i] /= SpectrumMean;

                    for (int i = 0; i < AverageSpectrum1DAll.Length; i++)
                        if (AverageSpectrum1DAll[i] <= 0)
                        {
                            for (int j = 0; j < AverageSpectrum1DAll.Length; j++)
                            {
                                if (i - j >= 0 && AverageSpectrum1DAll[i - j] > 0)
                                {
                                    AverageSpectrum1DAll[i] = AverageSpectrum1DAll[i - j];
                                    break;
                                }

                                if (i + j < AverageSpectrum1DAll.Length && AverageSpectrum1DAll[i + j] > 0)
                                {
                                    AverageSpectrum1DAll[i] = AverageSpectrum1DAll[i + j];
                                    break;
                                }
                            }
                        }

                    if (AverageSpectrum1DAll.Any(v => v <= 0))
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

                    Image CTFCoords = CTF.GetCTFCoords(Size, Size, CTF.Distortion);
                    Image Weights = GetCTFsForOneParticle(OptionsWeights, new float3(0.5f), CTFCoords, null, true, true);
                    //Weights.Min(1);
                    CTFCoords.Dispose();

                    #endregion

                    #region Divide weights by 1D PS, and create a 20 A high-passed version for CTF refinement

                    float[][] WeightsData = Weights.GetHost(Intent.ReadWrite);
                    for (int f = 0; f < NFrames; f++)
                        Helper.ForEachElementFT(new int2(Size), (x, y, xx, yy, r, angle) =>
                        {
                            if (r < Size / 2)
                            {
                                int idx = Math.Min(AverageSpectrum1DAll.Length - 1,
                                                    (int)Math.Round(r / (Size / 2) *
                                                                    (float)dataSource.PixelSizeMean /
                                                                    species.ResolutionRefinement *
                                                                    AverageSpectrum1DAll.Length));

                                WeightsData[f][y * (Size / 2 + 1) + x] /= AverageSpectrum1DAll[idx];
                            }
                            else
                            {
                                WeightsData[f][y * (Size / 2 + 1) + x] = 0;
                            }
                        });

                    //Weights.FreeDevice();
                    //if (GPUID == 0)
                    //    Weights.WriteMRC($"d_weights_{species.Name}.mrc", true);

                    Image WeightsRelevantlySized = new Image(new int3(Size, Size, NFrames), true);
                    for (int t = 0; t < NFrames; t++)
                        GPU.CropFTRealValued(Weights.GetDeviceSlice(t, Intent.Read),
                                            WeightsRelevantlySized.GetDeviceSlice(t, Intent.Write),
                                            Weights.Dims.Slice(),
                                            new int3(RelevantSizes[t]).Slice(),
                                            1);
                    //if (GPUID == 0)
                    //    WeightsRelevantlySized.WriteMRC($"d_weightsrelevant_{species.Name}.mrc", true);
                    Weights.Dispose();

                    Image CTFWeights = WeightsRelevantlySized.GetCopyGPU();
                    float[][] CTFWeightsData = CTFWeights.GetHost(Intent.ReadWrite);
                    for (int t = 0; t < CTFWeightsData.Length; t++)
                    {
                        int RelevantSize = RelevantSizes[t];
                        float R20 = Size * (species.ResolutionRefinement / 2 / 10f);
                        Helper.ForEachElementFT(new int2(RelevantSize), (x, y, xx, yy, r, angle) =>
                        {
                            float Weight = 1 - Math.Max(0, Math.Min(1, R20 - r));
                            CTFWeightsData[t][y * (RelevantSize / 2 + 1) + x] *= Weight;
                        });
                    }

                    CTFWeights.FreeDevice();
                    //if (GPUID == 0)
                    //    CTFWeights.WriteMRC($"d_ctfweights_{species.Name}.mrc", true);

                    #endregion

                    SpeciesCTFWeights.Add(species, CTFWeights);
                    SpeciesFrameWeights.Add(species, WeightsRelevantlySized);
                }

                #endregion

                // Remove original tilt image data from device, and dispose masks
                for (int f = 0; f < NFrames; f++)
                    FrameData[f].FreeDevice();

                #endregion

                #region Helper functions

                Action<bool> ReextractPaddedParticles = (CorrectBeamTilt) =>
                {
                    Stopwatch Watch = new Stopwatch();
                    Watch.Start();

                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                    foreach (var species in allSpecies)
                    {
                        if (!SpeciesParticles.ContainsKey(species))
                            continue;

                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        int SuperresFactor = SpeciesCTFSuperresFactor[species];
                        int BatchSize = optionsMPA.BatchSize;
                        BatchSize /= SuperresFactor * SuperresFactor;

                        int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                        int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                        int[] SizesRelevant = SpeciesRelevantRefinementSizes[species];

                        float AngPixRefine = species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                        float2[] ExtractedAt = SpeciesParticleExtractedAt[species];

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);
                        Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefineSuper);
                        Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

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

                                for (int f = 0; f < RedNFrames; f++)
                                {
                                    float3[] CoordinatesFrame = new float3[CurBatch];
                                    float3[] AnglesFrame = new float3[CurBatch];
                                    for (int p = 0; p < CurBatch; p++)
                                        CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];

                                    float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                    float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);

                                    float3[] Defoci = new float3[CurBatch];
                                    int3[] ExtractOrigins = new int3[CurBatch];
                                    float3[] ResidualShifts = new float3[BatchSize];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                        ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                        ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                        Defoci[p].X = ImageCoords[p].Z;
                                        Defoci[p].Y = Astigmatism[p].X;
                                        Defoci[p].Z = Astigmatism[p].Y;
                                        ExtractedAt[(batchStart + p) * NFrames + f] = new float2(ImageCoords[p]);
                                    }

                                    GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                                Extracted.GetDevice(Intent.Write),
                                                FrameData[f].Dims.Slice(),
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
                                        GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, PQReverse[iewald], ExtractedCTF, true);

                                        GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                           ExtractedCTF.GetDevice(Intent.Read),
                                                                           ExtractedFT.GetDevice(Intent.Write),
                                                                           ExtractedCTF.ElementsComplex,
                                                                           1);
                                    }
                                    else
                                    {
                                        GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                                   ParticleDiameterPix / 2f * 1.3f,
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
                                               ExtractedCroppedFTRelevantSize.GetDevice(Intent.Write),
                                               new int3(SizeRefine).Slice(),
                                               new int3(SizesRelevant[f]).Slice(),
                                               (uint)CurBatch);

                                    GPU.CopyDeviceToHostPinned(ExtractedCroppedFTRelevantSize.GetDevice(Intent.Read),
                                                               new IntPtr((long)PQStorage[iewald][f] + (new int3(SizesRelevant[f]).Slice().ElementsFFT()) * 2 * batchStart * sizeof(float)),
                                                               (new int3(SizesRelevant[f]).Slice().ElementsFFT()) * 2 * CurBatch);
                                }
                            }
                        }

                        CoordsCTF.Dispose();
                        GammaCorrection.Dispose();
                        PhaseCorrection.Dispose();
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

                    //foreach (var image in FrameData)
                    //    image.FreeDevice();

                    Watch.Stop();
                    Console.WriteLine($"Extracted particles in {Watch.ElapsedMilliseconds}");
                };

                Func<float2[]> GetRawShifts = () =>
                {
                    float2[] Result = new float2[NParticlesOverall * NFrames];

                    foreach (var species in allSpecies)
                    {
                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        float SpeciesAngPix = species.ResolutionRefinement / 2;
                        if (NParticles == 0)
                            continue;

                        int Offset = SpeciesParticleIDRanges[species].Start;

                        float3[] ParticlePositions = new float3[NParticles * NFrames];
                        for (int p = 0; p < NParticles; p++)
                        {
                            float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                            for (int f = 0; f < NFrames; f++)
                                ParticlePositions[p * NFrames + f] = Positions[f];
                        }

                        float3[] ParticlePositionsProjected = GetPositionInAllFrames(ParticlePositions);
                        float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                        for (int p = 0; p < NParticles; p++)
                            for (int f = 0; f < NFrames; f++)
                                Result[(Offset + p) * NFrames + f] = (new float2(ParticlePositionsProjected[p * NFrames + f]) - ParticleExtractedAt[p * NFrames + f]);
                    }

                    return Result;
                };

                Func<float2, Species, float[]> GetRawCCSpecies = (shiftBias, Species) =>
                {
                    Particle[] Particles = SpeciesParticles[Species];

                    int NParticles = Particles.Length;
                    float AngPixRefine = Species.ResolutionRefinement / 2;

                    if (NParticles == 0)
                        return new float[NParticles * NFrames * 3];

                    float[] SpeciesResult = new float[NParticles * RedNFrames * 3];
                    float[] SpeciesResultQ = new float[NParticles * RedNFrames * 3];

                    float3[] ParticlePositions = new float3[NParticles * NFrames];
                    float3[] ParticleAngles = new float3[NParticles * NFrames];
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);
                        float3[] Angles = Particles[p].GetAngleSeries(DoseInterpolationSteps);

                        for (int f = 0; f < NFrames; f++)
                        {
                            ParticlePositions[p * NFrames + f] = Positions[f];
                            ParticleAngles[p * NFrames + f] = Angles[f];// * Helper.ToRad;
                        }
                    }

                    float3[] ParticlePositionsProjected = GetPositionInAllFrames(ParticlePositions);
                    float3[] ParticleAnglesInFrames = GetParticleAngleInAllFrames(ParticlePositions, ParticleAngles); // ParticleAngles;

                    float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[Species];
                    float2[] ParticleShifts = new float2[NFrames * NParticles];
                    for (int p = 0; p < NParticles; p++)
                        for (int t = 0; t < NFrames; t++)
                            ParticleShifts[p * NFrames + t] = (new float2(ParticlePositionsProjected[p * NFrames + t]) - ParticleExtractedAt[p * NFrames + t] + shiftBias) / AngPixRefine;

                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[Species];

                    int SizeRefine = Species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;

                    Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                    Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                    for (int f = 0; f < NFrames; f++)
                        GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                    PhaseCorrectionAll.GetDeviceSlice(f, Intent.Write),
                                    PhaseCorrection.Dims.Slice(),
                                    new int3(RelevantSizes[f]).Slice(),
                                    1);

                    if (RedNFrames != NFrames)
                    {
                        float2[] RedParticleShifts = new float2[RedNFrames * NParticles];
                        for (int p = 0; p < NParticles; p++)
                            for (int f = 0; f < RedNFrames; f++)
                                RedParticleShifts[p * RedNFrames + f] = ParticleShifts[p * NFrames + f];

                        float3[] RedParticleAngles = new float3[RedNFrames * NParticles];
                        for (int p = 0; p < NParticles; p++)
                            for (int f = 0; f < RedNFrames; f++)
                                RedParticleAngles[p * RedNFrames + f] = ParticleAnglesInFrames[p * NFrames + f];

                        ParticleShifts = RedParticleShifts;
                        ParticleAnglesInFrames = RedParticleAngles;
                    }

                    GPU.MultiParticleDiff(SpeciesResult,
                                            SpeciesParticleImages[Species],
                                            SpeciesRefinementSize[Species],
                                            RelevantSizes,
                                            Helper.ToInterleaved(ParticleShifts),
                                            Helper.ToInterleaved(ParticleAnglesInFrames),
                                            MagnificationCorrection.ToVec(),
                                            (Species.ResolutionRefinement < 8 ? SpeciesCTFWeights : SpeciesFrameWeights)[Species].GetDevice(Intent.Read),
                                            PhaseCorrectionAll.GetDevice(Intent.Read),
                                            Species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize) : 0,
                                            Species.CurrentMaxShellRefinement,
                                            new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                            new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                            Species.HalfMap1Projector[GPUID].Oversampling,
                                            Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                            SpeciesParticleSubsets[Species],
                                            NParticles,
                                            RedNFrames);

                    if (Species.DoEwald)
                        GPU.MultiParticleDiff(SpeciesResultQ,
                                              SpeciesParticleQImages[Species],
                                              SpeciesRefinementSize[Species],
                                              RelevantSizes,
                                              Helper.ToInterleaved(ParticleShifts),
                                              Helper.ToInterleaved(ParticleAnglesInFrames),
                                              MagnificationCorrection.ToVec(),
                                              (Species.ResolutionRefinement < 8 ? SpeciesCTFWeights : SpeciesFrameWeights)[Species].GetDevice(Intent.Read),
                                              PhaseCorrectionAll.GetDevice(Intent.Read),
                                              -CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize),
                                              Species.CurrentMaxShellRefinement,
                                              new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                              new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                              Species.HalfMap1Projector[GPUID].Oversampling,
                                              Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                              SpeciesParticleSubsets[Species],
                                              NParticles,
                                              RedNFrames);

                    GPU.CheckGPUExceptions();

                    PhaseCorrection.Dispose();
                    PhaseCorrectionAll.Dispose();

                    if (Species.DoEwald)
                        for (int i = 0; i < SpeciesResult.Length; i++)
                            SpeciesResult[i] += SpeciesResultQ[i];

                    if (RedNFrames != NFrames)
                    {
                        float[] FullSpeciesResult = new float[NParticles * NFrames * 3];
                        for (int p = 0; p < NParticles; p++)
                            for (int f = 0; f < RedNFrames; f++)
                            {
                                FullSpeciesResult[(p * NFrames + f) * 3 + 0] = SpeciesResult[(p * RedNFrames + f) * 3 + 0];
                                FullSpeciesResult[(p * NFrames + f) * 3 + 1] = SpeciesResult[(p * RedNFrames + f) * 3 + 1];
                                FullSpeciesResult[(p * NFrames + f) * 3 + 2] = SpeciesResult[(p * RedNFrames + f) * 3 + 2];
                            }

                        SpeciesResult = FullSpeciesResult;
                    }

                    return SpeciesResult;
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

                        float[] SpeciesResult = GetRawCCSpecies(shiftBias, Species);

                        int Offset = SpeciesParticleIDRanges[Species].Start * NFrames * 3;
                        Array.Copy(SpeciesResult, 0, Result, Offset, SpeciesResult.Length);
                    }

                    return Result;
                };

                Func<double[]> GetPerFrameCC = () =>
                {
                    double[] Result = new double[NFrames * 3];
                    float[] RawResult = GetRawCC(new float2(0));

                    for (int p = 0; p < NParticlesOverall; p++)
                        for (int f = 0; f < RedNFrames; f++)
                        {
                            Result[f * 3 + 0] += RawResult[(p * NFrames + f) * 3 + 0];
                            Result[f * 3 + 1] += RawResult[(p * NFrames + f) * 3 + 1];
                            Result[f * 3 + 2] += RawResult[(p * NFrames + f) * 3 + 2];
                        }

                    Result = Helper.ArrayOfFunction(f => Result[f * 3 + 0] /
                                                         Math.Max(1e-10, Math.Sqrt(Result[f * 3 + 1] * Result[f * 3 + 2])) *
                                                         100 * NParticlesOverall,
                                                    NFrames);

                    return Result;
                };

                Func<double[]> GetPerParticleCC = () =>
                {
                    double[] Result = new double[NParticlesOverall * 3];
                    float[] RawResult = GetRawCC(new float2(0));

                    for (int p = 0; p < NParticlesOverall; p++)
                        for (int f = 0; f < RedNFrames; f++)
                        {
                            Result[p * 3 + 0] += RawResult[(p * NFrames + f) * 3 + 0];
                            Result[p * 3 + 1] += RawResult[(p * NFrames + f) * 3 + 1];
                            Result[p * 3 + 2] += RawResult[(p * NFrames + f) * 3 + 2];
                        }

                    Result = Helper.ArrayOfFunction(p => Result[p * 3 + 0] /
                                                         Math.Max(1e-10, Math.Sqrt(Result[p * 3 + 1] * Result[p * 3 + 2])) *
                                                         100 * NFrames, NParticlesOverall);

                    return Result;
                };

                Func<Species, double[]> GetPerParticleCCSpecies = (species) =>
                {
                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;

                    double[] Result = new double[NParticles * 3];
                    float[] RawResult = GetRawCC(new float2(0));

                    for (int p = 0; p < NParticles; p++)
                        for (int f = 0; f < RedNFrames; f++)
                        {
                            Result[p * 3 + 0] += RawResult[(p * NFrames + f) * 3 + 0];
                            Result[p * 3 + 1] += RawResult[(p * NFrames + f) * 3 + 1];
                            Result[p * 3 + 2] += RawResult[(p * NFrames + f) * 3 + 2];
                        }

                    Result = Helper.ArrayOfFunction(p => Result[p * 3 + 0] /
                                                         Math.Max(1e-10, Math.Sqrt(Result[p * 3 + 1] * Result[p * 3 + 2])) *
                                                         100 * NFrames, NParticles);

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
                    //{
                    //    Result[i] = new float2((h_ScoresXP[i] - h_ScoresXM[i]) / Delta2 * 100,
                    //                           (h_ScoresYP[i] - h_ScoresYM[i]) / Delta2 * 100);
                    //    //if (float.IsNaN(Result[i].X) || float.IsNaN(Result[i].Y))
                    //    //    throw new Exception();
                    //}

                    return (h_ScoresXP, h_ScoresXM, h_ScoresYP, h_ScoresYM, Delta2);
                };

                #endregion

                #region BFGS prerequisites

                float2[][] OriginalOffsets = Helper.ArrayOfFunction(p => Helper.ArrayOfFunction(t =>
                                                                             new float2(PyramidShiftX[p].Values[t],
                                                                                        PyramidShiftY[p].Values[t]),
                                                                             PyramidShiftX[p].Values.Length),
                                                                         PyramidShiftX.Count);

                float[] OriginalAngleX = GridAngleX.Values.ToArray();
                float[] OriginalAngleY = GridAngleY.Values.ToArray();
                float[] OriginalAngleZ = GridAngleZ.Values.ToArray();

                float4[] ParticleMagChanges = new float4[ParticleMags.Length];

                float[] OriginalParamsCTF =
                {
                    (float)CTF.PhaseShift,
                    CTF.Distortion.M11,
                    CTF.Distortion.M21,
                    CTF.Distortion.M12,
                    CTF.Distortion.M22
                };

                CTFOptimizationTypes[] CTFStepTypes =
                {
                    CTFOptimizationTypes.Defocus,
                    CTFOptimizationTypes.AstigmatismDelta,
                    CTFOptimizationTypes.AstigmatismAngle,
                    CTFOptimizationTypes.Cs,
                    CTFOptimizationTypes.Doming,
                    CTFOptimizationTypes.Zernike2,
                    CTFOptimizationTypes.Zernike2,
                    CTFOptimizationTypes.Zernike2,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.Zernike4,
                    CTFOptimizationTypes.PhaseShift,
                    CTFOptimizationTypes.Distortion,
                    CTFOptimizationTypes.Distortion,
                    CTFOptimizationTypes.Distortion,
                    CTFOptimizationTypes.Distortion
                };

                float[] OriginalDefocusDelta = GridCTFDefocusDelta.Values.ToList().ToArray();
                float[] OriginalDefocusAngle = GridCTFDefocusAngle.Values.ToList().ToArray();
                float[] OriginalCs = GridCTFCs.Values.ToList().ToArray();

                float[] OriginalDefocusDoming = GridCTFDoming.Values.ToList().ToArray();

                float[] OriginalZernikeOdd = CTF.ZernikeCoeffsOdd.ToList().ToArray();
                float[] OriginalZernikeEven = CTF.ZernikeCoeffsEven.ToList().ToArray();

                Matrix2 OriginalMagnification = MagnificationCorrection.GetCopy();

                float3[][] OriginalParticlePositions = allSpecies.Select(s => Helper.Combine(SpeciesParticles[s].Select(p => p.Coordinates))).ToArray();
                float3[][] OriginalParticleAngles = allSpecies.Select(s => Helper.Combine(SpeciesParticles[s].Select(p => p.Angles))).ToArray();

                int BFGSIterations = 0;
                WarpOptimizationTypes CurrentOptimizationTypeWarp = 0;
                CTFOptimizationTypes CurrentOptimizationTypeCTF = 0;

                double[] InitialParametersWarp = new double[PyramidShiftX.Select(g => g.Values.Length).Sum() * 2 +
                                                            GridAngleX.Values.Length * 3 +
                                                            OriginalParticlePositions.Select(a => a.Length).Sum() * 2 +
                                                            OriginalParticleAngles.Select(a => a.Length).Sum() * 3 +
                                                            CTF.ZernikeCoeffsOdd.Length +
                                                            9 * 4 +
                                                            4];
                double[] InitialParametersDefocus = new double[NParticlesOverall +
                                                               GridCTFDefocusDelta.Values.Length +
                                                               GridCTFDefocusAngle.Values.Length +
                                                               GridCTFCs.Values.Length +
                                                               GridCTFDoming.Values.Length +
                                                               CTF.ZernikeCoeffsEven.Length +
                                                               OriginalParamsCTF.Length];

                #endregion

                #region Set parameters from vector

                Action<double[], Movie, bool> SetWarpFromVector = (input, movie, setParticles) =>
                {
                    int Offset = 0;

                    int3[] PyramidDimensions = PyramidShiftX.Select(g => g.Dimensions).ToArray();

                    movie.PyramidShiftX.Clear();
                    movie.PyramidShiftY.Clear();

                    for (int p = 0; p < PyramidDimensions.Length; p++)
                    {
                        float[] MovementXData = new float[PyramidDimensions[p].Elements()];
                        float[] MovementYData = new float[PyramidDimensions[p].Elements()];
                        for (int i = 0; i < MovementXData.Length; i++)
                        {
                            MovementXData[i] = OriginalOffsets[p][i].X + (float)input[Offset + i * 2 + 0];
                            MovementYData[i] = OriginalOffsets[p][i].Y + (float)input[Offset + i * 2 + 1];
                        }
                        movie.PyramidShiftX.Add(new CubicGrid(PyramidDimensions[p], MovementXData));
                        movie.PyramidShiftY.Add(new CubicGrid(PyramidDimensions[p], MovementYData));

                        Offset += MovementXData.Length * 2;
                    }

                    float[] AngleXData = new float[GridAngleX.Values.Length];
                    float[] AngleYData = new float[GridAngleY.Values.Length];
                    float[] AngleZData = new float[GridAngleZ.Values.Length];
                    for (int i = 0; i < AngleXData.Length; i++)
                    {
                        AngleXData[i] = OriginalAngleX[i] + (float)input[Offset + i];
                        AngleYData[i] = OriginalAngleY[i] + (float)input[Offset + AngleXData.Length + i];
                        AngleZData[i] = OriginalAngleZ[i] + (float)input[Offset + AngleXData.Length * 2 + i];
                    }
                    movie.GridAngleX = new CubicGrid(GridAngleX.Dimensions, AngleXData);
                    movie.GridAngleY = new CubicGrid(GridAngleY.Dimensions, AngleYData);
                    movie.GridAngleZ = new CubicGrid(GridAngleZ.Dimensions, AngleZData);

                    Offset += AngleXData.Length * 3;

                    if (setParticles)
                    {
                        for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                        {
                            Particle[] Particles = SpeciesParticles[allSpecies[ispecies]];

                            int ResCoords = allSpecies[0].TemporalResolutionMovement;

                            for (int p = 0; p < Particles.Length; p++)
                            {
                                for (int ic = 0; ic < ResCoords; ic++)
                                {
                                    Particles[p].Coordinates[ic].X = OriginalParticlePositions[ispecies][p * ResCoords + ic].X + (float)input[Offset + (p * 5 + 0) * ResCoords + ic];
                                    Particles[p].Coordinates[ic].Y = OriginalParticlePositions[ispecies][p * ResCoords + ic].Y + (float)input[Offset + (p * 5 + 1) * ResCoords + ic];

                                    Particles[p].Angles[ic] = OriginalParticleAngles[ispecies][p * ResCoords + ic] + new float3((float)input[Offset + (p * 5 + 2) * ResCoords + ic],
                                                                                                                                (float)input[Offset + (p * 5 + 3) * ResCoords + ic],
                                                                                                                                (float)input[Offset + (p * 5 + 4) * ResCoords + ic]);
                                }
                            }

                            Offset += OriginalParticlePositions[ispecies].Length * 5;
                        }
                    }
                    else
                    {
                        Offset += OriginalParticlePositions.Select(a => a.Length).Sum() * 5;
                    }

                    for (int icoeff = 0; icoeff < CTF.ZernikeCoeffsOdd.Length; icoeff++)
                        CTF.ZernikeCoeffsOdd[icoeff] = OriginalZernikeOdd[icoeff] + (float)input[Offset + icoeff];

                    // In beam tilt, coeff 0 is coupled with 3, and 1 with 4
                    if (CTF.ZernikeCoeffsOdd.Length >= 5)
                    {
                        CTF.ZernikeCoeffsOdd[0] = 2 * CTF.ZernikeCoeffsOdd[3];
                        CTF.ZernikeCoeffsOdd[1] = 2 * CTF.ZernikeCoeffsOdd[4];
                    }

                    Offset += CTF.ZernikeCoeffsOdd.Length;

                    {
                        float4 MagCorrGlobal = MagnificationCorrection.ToVec();

                        CubicGrid Grid11 = new CubicGrid(new int3(3, 3, 1), input.Skip(Offset + 0).Take(9).Select(v => (float)v).ToArray());
                        CubicGrid Grid21 = new CubicGrid(new int3(3, 3, 1), input.Skip(Offset + 9).Take(9).Select(v => (float)v).ToArray());
                        CubicGrid Grid12 = new CubicGrid(new int3(3, 3, 1), input.Skip(Offset + 18).Take(9).Select(v => (float)v).ToArray());
                        CubicGrid Grid22 = new CubicGrid(new int3(3, 3, 1), input.Skip(Offset + 27).Take(9).Select(v => (float)v).ToArray());

                        float3[] Coords = SpeciesParticles[allSpecies[0]].Select(p => new float3(p.Coordinates[0].X / ImageDimensionsPhysical.X, p.Coordinates[0].Y / ImageDimensionsPhysical.Y, 0.5f)).ToArray();

                        float[] M11 = Grid11.GetInterpolated(Coords);
                        float[] M21 = Grid21.GetInterpolated(Coords);
                        float[] M12 = Grid12.GetInterpolated(Coords);
                        float[] M22 = Grid22.GetInterpolated(Coords);

                        for (int p = 0; p < ParticleMags.Length; p++)
                        {
                            ParticleMags[p] = MagCorrGlobal + new float4(M11[p], M21[p], M12[p], M22[p]) * 0.01f;
                        }

                        Offset += 9 * 4;
                    }

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

                Action<double[], Movie, bool> SetDefocusFromVector = (input, movie, setParticles) =>
                {
                    int Offset = 0;

                    if (setParticles)
                    {
                        for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                        {
                            Particle[] Particles = SpeciesParticles[allSpecies[ispecies]];

                            int ResCoords = allSpecies[ispecies].TemporalResolutionMovement;

                            for (int p = 0; p < Particles.Length; p++)
                            {
                                // Coords are in Angstrom and we want 0.1 * micrometer, thus * 1e3
                                for (int ic = 0; ic < ResCoords; ic++)
                                    Particles[p].Coordinates[ic].Z = OriginalParticlePositions[ispecies][p * ResCoords + ic].Z + (float)input[p + Offset] * 1e3f;
                            }
                            Offset += Particles.Length;
                        }
                    }
                    else
                    {
                        Offset += NParticlesOverall;
                    }

                    {
                        float[] ValuesDelta = new float[GridCTFDefocusDelta.Values.Length];
                        for (int i = 0; i < ValuesDelta.Length; i++)
                            ValuesDelta[i] = OriginalDefocusDelta[i] + (float)input[Offset + i] * 0.1f;

                        movie.GridCTFDefocusDelta = new CubicGrid(GridCTFDefocusDelta.Dimensions, ValuesDelta);
                        Offset += ValuesDelta.Length;
                    }

                    {
                        float[] ValuesAngle = new float[GridCTFDefocusAngle.Values.Length];
                        for (int i = 0; i < ValuesAngle.Length; i++)
                            ValuesAngle[i] = OriginalDefocusAngle[i] + (float)input[Offset + i] * 36;

                        movie.GridCTFDefocusAngle = new CubicGrid(GridCTFDefocusAngle.Dimensions, ValuesAngle);
                        Offset += ValuesAngle.Length;
                    }

                    {
                        float[] ValuesCs = new float[GridCTFCs.Values.Length];
                        for (int i = 0; i < ValuesCs.Length; i++)
                            ValuesCs[i] = OriginalCs[i] + (float)input[Offset + i] * 3;

                        movie.GridCTFCs = new CubicGrid(GridCTFCs.Dimensions, ValuesCs);
                        Offset += ValuesCs.Length;
                    }

                    {
                        float[] ValuesDoming = new float[GridCTFDoming.Values.Length];
                        for (int i = 0; i < ValuesDoming.Length; i++)
                            ValuesDoming[i] = OriginalDefocusDoming[i] + (float)input[Offset + i] * 0.1f;

                        movie.GridCTFDoming = new CubicGrid(GridCTFDoming.Dimensions, ValuesDoming);
                        Offset += ValuesDoming.Length;
                    }

                    {
                        float[] ValuesZernike = new float[CTF.ZernikeCoeffsEven.Length];
                        for (int i = 0; i < ValuesZernike.Length; i++)
                            ValuesZernike[i] = OriginalZernikeEven[i] + (float)input[Offset + i];

                        movie.CTF.ZernikeCoeffsEven = ValuesZernike;
                        Offset += CTF.ZernikeCoeffsEven.Length;
                    }

                    movie.CTF.PhaseShift = (decimal)(OriginalParamsCTF[0] + input[input.Length - 5]);

                    movie.CTF.Distortion = new Matrix2(OriginalParamsCTF[1] + (float)input[input.Length - 4] / 100,
                                                       OriginalParamsCTF[2] + (float)input[input.Length - 3] / 100,
                                                       OriginalParamsCTF[3] + (float)input[input.Length - 2] / 100,
                                                       OriginalParamsCTF[4] + (float)input[input.Length - 1] / 100);
                    movie.CTF.PixelSizeDeltaPercent = 0;
                };

                #endregion

                #region Wiggle weights

                progressCallback("Precomputing gradient weights...");

                int NWiggleDifferentiableWarp = PyramidShiftX.Select(g => g.Values.Length).Sum() * 2;
                (int[] indices, float2[] weights)[] AllWiggleWeightsWarp = new (int[] indices, float2[] weights)[NWiggleDifferentiableWarp];

                int NWiggleDifferentiableAstigmatism = GridCTFDefocusDelta.Values.Length;
                (int[] indices, float[] weights)[] AllWiggleWeightsAstigmatism = new (int[] indices, float[] weights)[NWiggleDifferentiableAstigmatism];

                int NWiggleDifferentiableDoming = GridCTFDoming.Values.Length;
                (int[] indices, float[] weights)[] AllWiggleWeightsDoming = new (int[] indices, float[] weights)[NWiggleDifferentiableDoming];

                //if ((optionsMPA.RefinedComponentsWarp & WarpOptimizationTypes.ImageWarp) != 0 ||
                //    (optionsMPA.RefinedComponentsCTF & CTFOptimizationTypes.AstigmatismDelta) != 0)
                {
                    Movie[] ParallelMovieCopies = Helper.ArrayOfFunction(i => new Movie(this.Path), 32);

                    Dictionary<Species, float3[]> SpeciesParticlePositions = new Dictionary<Species, float3[]>();
                    Dictionary<Species, float2[]> SpeciesParticleAstigmatism = new Dictionary<Species, float2[]>();
                    foreach (var species in allSpecies)
                    {
                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        float3[] ParticlePositions = new float3[NParticles * NFrames];
                        for (int p = 0; p < NParticles; p++)
                        {
                            float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                            for (int f = 0; f < NFrames; f++)
                                ParticlePositions[p * NFrames + f] = Positions[f];
                        }

                        SpeciesParticlePositions.Add(species, ParticlePositions);

                        SpeciesParticleAstigmatism.Add(species, GetAstigmatism(Particles.Select(p => p.Coordinates[0]).ToArray()));
                    }

                    #region Warp

                    if (optionsMPA.DoImageWarp || optionsMPA.DoZernike13 || optionsMPA.DoZernike5)
                        Helper.ForCPU(0, NWiggleDifferentiableWarp / 2, ParallelMovieCopies.Length, (threadID) =>
                        {
                            ParallelMovieCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                            ParallelMovieCopies[threadID].NFrames = NFrames;
                            ParallelMovieCopies[threadID].FractionFrames = FractionFrames;
                        },
                        (iwiggle, threadID) =>
                        {
                            double[] WiggleParams = new double[InitialParametersWarp.Length];
                            WiggleParams[iwiggle * 2] = 1;
                            SetWarpFromVector(WiggleParams, ParallelMovieCopies[threadID], false);

                            float2[] RawShifts = new float2[NParticlesOverall * NFrames];
                            foreach (var species in allSpecies)
                            {
                                Particle[] Particles = SpeciesParticles[species];
                                int NParticles = Particles.Length;
                                if (NParticles == 0)
                                    continue;

                                int Offset = SpeciesParticleIDRanges[species].Start;

                                float3[] ParticlePositionsProjected = ParallelMovieCopies[threadID].GetPositionInAllFrames(SpeciesParticlePositions[species]);
                                float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                                for (int p = 0; p < NParticles; p++)
                                    for (int f = 0; f < NFrames; f++)
                                        RawShifts[(Offset + p) * NFrames + f] = (new float2(ParticlePositionsProjected[p * NFrames + f]) - ParticleExtractedAt[p * NFrames + f]);
                            }

                            List<int> Indices = new List<int>(RawShifts.Length / 5);
                            List<float2> Weights = new List<float2>(RawShifts.Length / 5);
                            List<float2> WeightsY = new List<float2>(RawShifts.Length / 5);
                            for (int i = 0; i < RawShifts.Length; i++)
                            {
                                if (RawShifts[i].LengthSq() > 1e-6f)
                                {
                                    Indices.Add(i);
                                    Weights.Add(RawShifts[i]);
                                    WeightsY.Add(new float2(RawShifts[i].Y, RawShifts[i].X));

                                    if (Math.Abs(RawShifts[i].X) > 1.5f)
                                        throw new Exception();
                                }
                            }

                            AllWiggleWeightsWarp[iwiggle * 2 + 0] = (Indices.ToArray(), Weights.ToArray());
                            AllWiggleWeightsWarp[iwiggle * 2 + 1] = (Indices.ToArray(), WeightsY.ToArray());
                        }, null);

                    #endregion

                    #region Astigmatism

                    if (optionsMPA.DoAstigmatismDelta || optionsMPA.DoCs)
                        Helper.ForCPU(0, NWiggleDifferentiableAstigmatism, ParallelMovieCopies.Length, (threadID) =>
                        {
                            ParallelMovieCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                            ParallelMovieCopies[threadID].NFrames = NFrames;
                            ParallelMovieCopies[threadID].FractionFrames = FractionFrames;
                        },
                        (iwiggle, threadID) =>
                        {
                            double[] WiggleParams = new double[InitialParametersDefocus.Length];
                            WiggleParams[NParticlesOverall + iwiggle] = 10; // because it's weighted *0.1 later in SetDefocusFromVector
                            SetDefocusFromVector(WiggleParams, ParallelMovieCopies[threadID], false);

                            float2[] RawDiffs = new float2[NParticlesOverall];
                            foreach (var species in allSpecies)
                            {
                                Particle[] Particles = SpeciesParticles[species];
                                int NParticles = Particles.Length;
                                if (NParticles == 0)
                                    continue;

                                int Offset = SpeciesParticleIDRanges[species].Start;

                                float2[] ParticleAstigmatismAltered = ParallelMovieCopies[threadID].GetAstigmatism(Particles.Select(p => p.Coordinates[0]).ToArray());
                                float2[] ParticleAstigmatismOriginal = SpeciesParticleAstigmatism[species];

                                for (int p = 0; p < NParticles; p++)
                                    RawDiffs[Offset + p] = ParticleAstigmatismAltered[p] - ParticleAstigmatismOriginal[p];
                            }

                            List<int> Indices = new List<int>(RawDiffs.Length);
                            List<float> Weights = new List<float>(RawDiffs.Length);
                            for (int i = 0; i < RawDiffs.Length; i++)
                            {
                                Indices.Add(i);
                                Weights.Add(RawDiffs[i].X);
                            }

                            AllWiggleWeightsAstigmatism[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                        }, null);

                    #endregion

                    #region Doming

                    //if ((optionsMPA.RefinedComponentsCTF & CTFOptimizationTypes.Doming) != 0)
                    Helper.ForCPU(0, NWiggleDifferentiableDoming, ParallelMovieCopies.Length, (threadID) =>
                    {
                        ParallelMovieCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                        ParallelMovieCopies[threadID].NFrames = NFrames;
                        ParallelMovieCopies[threadID].FractionFrames = FractionFrames;
                    },
                    (iwiggle, threadID) =>
                    {
                        double[] WiggleParams = new double[InitialParametersDefocus.Length];
                        WiggleParams[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + iwiggle] = 10; // because it's weighted *0.1 later in SetDefocusFromVector
                        SetDefocusFromVector(WiggleParams, ParallelMovieCopies[threadID], false);

                        float[] RawDefoci = new float[NParticlesOverall * NFrames];
                        foreach (var species in allSpecies)
                        {
                            Particle[] Particles = SpeciesParticles[species];
                            int NParticles = Particles.Length;
                            if (NParticles == 0)
                                continue;

                            int Offset = SpeciesParticleIDRanges[species].Start;

                            float3[] ParticlePositionsProjected = ParallelMovieCopies[threadID].GetPositionInAllFrames(SpeciesParticlePositions[species]);
                            float[] ParticleExtractedAt = SpeciesParticleExtractedAtDefocus[species];

                            for (int p = 0; p < NParticles; p++)
                                for (int f = 0; f < NFrames; f++)
                                    RawDefoci[(Offset + p) * NFrames + f] = ParticlePositionsProjected[p * NFrames + f].Z - ParticleExtractedAt[p * NFrames + f];
                        }

                        List<int> Indices = new List<int>(RawDefoci.Length / NFrames);
                        List<float> Weights = new List<float>(RawDefoci.Length / NFrames);
                        for (int i = 0; i < RawDefoci.Length; i++)
                        {
                            if (Math.Abs(RawDefoci[i]) > 1e-6f)
                            {
                                Indices.Add(i);
                                Weights.Add(RawDefoci[i]);

                                if (Math.Abs(RawDefoci[i]) > 1.5f)
                                    throw new Exception();
                            }
                        }

                        AllWiggleWeightsDoming[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                    }, null);

                    #endregion
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
                        for (int f = 0; f < RedNFrames; f++)
                        {
                            SumAB += RawCC[(p * NFrames + f) * 3 + 0];
                            SumA2 += RawCC[(p * NFrames + f) * 3 + 1];
                            SumB2 += RawCC[(p * NFrames + f) * 3 + 2];
                        }
                    }

                    double Score = SumAB / Math.Max(1e-10, Math.Sqrt(SumA2 * SumB2)) * NParticlesOverall * NFrames * 100;

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

                    if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ImageWarp) != 0) // Image shift pyramids
                    {
                        SetWarpFromVector(input, this, true);
                        (var XP, var XM, var YP, var YM, var Delta2Movement) = GetRawShiftGradients();

                        Parallel.For(0, AllWiggleWeightsWarp.Length, iwiggle =>
                        {
                            double SumGrad = 0;
                            double SumWeights = 0;
                            double SumWeightsGrad = 0;

                            int[] Indices = AllWiggleWeightsWarp[iwiggle].indices;
                            float2[] Weights = AllWiggleWeightsWarp[iwiggle].weights;

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
                    }
                    Offset += AllWiggleWeightsWarp.Length;


                    if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.AxisAngle) != 0) // GridAngleX, Y and Z
                    {
                        int SliceElements = (int)GridAngleX.Dimensions.ElementsSlice();

                        for (int a = 0; a < 3; a++)
                        {
                            for (int i = 0; i < SliceElements; i++)
                            {
                                double[] InputPlus = input.ToArray();
                                for (int t = 0; t < NFrames; t++)
                                    InputPlus[Offset + t * SliceElements + i] += Delta;

                                SetWarpFromVector(InputPlus, this, true);
                                double[] ScoresPlus = GetPerFrameCC();

                                double[] InputMinus = input.ToArray();
                                for (int t = 0; t < NFrames; t++)
                                    InputMinus[Offset + t * SliceElements + i] -= Delta;

                                SetWarpFromVector(InputMinus, this, true);
                                double[] ScoresMinus = GetPerFrameCC();

                                for (int t = 0; t < NFrames; t++)
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
                                for (int iparam = 0; iparam < 2 * TemporalRes; iparam++)
                                {
                                    if (iparam % TemporalRes != 0)
                                        continue;

                                    double[] InputPlus = input.ToArray();
                                    for (int p = 0; p < Particles.Length; p++)
                                        InputPlus[Offset + p * 5 * TemporalRes + iparam] += Delta;

                                    SetWarpFromVector(InputPlus, this, true);
                                    double[] ScoresPlus = GetPerParticleCCSpecies(Species);

                                    double[] InputMinus = input.ToArray();
                                    for (int p = 0; p < Particles.Length; p++)
                                        InputMinus[Offset + p * 5 * TemporalRes + iparam] -= Delta;

                                    SetWarpFromVector(InputMinus, this, true);
                                    double[] ScoresMinus = GetPerParticleCCSpecies(Species);

                                    for (int p = 0; p < Particles.Length; p++)
                                        Result[Offset + p * 5 * TemporalRes + iparam] = (ScoresPlus[p] - ScoresMinus[p]) / Delta2;
                                }

                            if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ParticleAngle) != 0)
                                for (int iparam = 2 * TemporalRes; iparam < 5 * TemporalRes; iparam++)
                                {
                                    if (iparam % TemporalRes != 0)
                                        continue;

                                    double[] InputPlus = input.ToArray();
                                    for (int p = 0; p < Particles.Length; p++)
                                        InputPlus[Offset + p * 5 * TemporalRes + iparam] += Delta;

                                    SetWarpFromVector(InputPlus, this, true);
                                    double[] ScoresPlus = GetPerParticleCCSpecies(Species);

                                    double[] InputMinus = input.ToArray();
                                    for (int p = 0; p < Particles.Length; p++)
                                        InputMinus[Offset + p * 5 * TemporalRes + iparam] -= Delta;

                                    SetWarpFromVector(InputMinus, this, true);
                                    double[] ScoresMinus = GetPerParticleCCSpecies(Species);

                                    for (int p = 0; p < Particles.Length; p++)
                                        Result[Offset + p * 5 * TemporalRes + iparam] = (ScoresPlus[p] - ScoresMinus[p]) / Delta2;
                                }

                            Offset += OriginalParticlePositions[ispecies].Length * 5; // No * TemporalRes because it's already included in OriginalParticlePositions
                        }
                    }

                    if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.Zernike13) != 0)
                    {
                        // First 2 coeffs are coupled with 3 and 4 for beam tilt
                        for (int iparam = 2; iparam < Math.Min(6, CTF.ZernikeCoeffsOdd.Length); iparam++)
                        {
                            double[] InputPlus = input.ToArray();
                            InputPlus[Offset + iparam] += Delta;

                            double ScoresPlus = WarpEval(InputPlus);

                            double[] InputMinus = input.ToArray();
                            InputMinus[Offset + iparam] -= Delta;

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

                    if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ParticleMag) != 0)
                        for (int iparam = 0; iparam < 9 * 4; iparam++)
                        {
                            double[] InputPlus = input.ToArray();
                            InputPlus[Offset + iparam] += Delta;
                            double ScoresPlus = WarpEval(InputPlus);

                            double[] InputMinus = input.ToArray();
                            InputMinus[Offset + iparam] -= Delta;
                            double ScoresMinus = WarpEval(InputMinus);

                            Result[Offset + iparam] = (ScoresPlus - ScoresMinus) / Delta2;
                        }

                    Offset += 9 * 4;

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

                Func<double[], double> CTFEval = input =>
                {
                    SetDefocusFromVector(input, this, true);

                    double ScoreAB = 0, ScoreA2 = 0, ScoreB2 = 0;

                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                    foreach (var species in allSpecies)
                    {
                        if (!SpeciesParticles.ContainsKey(species))
                            continue;

                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        int SuperresFactor = SpeciesCTFSuperresFactor[species];
                        int BatchSize = optionsMPA.BatchSize;
                        BatchSize /= SuperresFactor * SuperresFactor;

                        float[] ResultP = new float[BatchSize * 3];
                        float[] ResultQ = new float[BatchSize * 3];

                        int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeRefineSuper = SizeRefine * SuperresFactor;
                        int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SuperresFactor;

                        float AngPixRefine = species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                        int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);   // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine

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
                        Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                        for (int f = 0; f < NFrames; f++)
                            GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                       PhaseCorrectionAll.GetDeviceSlice(f, Intent.Write),
                                       PhaseCorrection.Dims.Slice(),
                                       new int3(RelevantSizes[f]).Slice(),
                                       1);
                        PhaseCorrection.Dispose();

                        Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                        bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                        float[][] EwaldResults = { ResultP, ResultQ };

                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                            float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                            for (int f = 0; f < RedNFrames; f++)
                            {
                                float3[] CoordinatesFrame = new float3[CurBatch];
                                float3[] AnglesFrame = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                    AnglesFrame[p] = AnglesMoving[p * NFrames + f];// * Helper.ToRad;
                                }

                                float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f);// AnglesFrame;
                                float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);

                                float3[] Defoci = new float3[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p].X = ImageCoords[p].Z;
                                    Defoci[p].Y = Astigmatism[p].X;
                                    Defoci[p].Z = Astigmatism[p].Y;
                                }

                                for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                {
                                    GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                                Extracted.GetDevice(Intent.Write),
                                                FrameData[f].Dims.Slice(),
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
                                        GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, EwaldReverse[iewald], ExtractedCTF, true);

                                        GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                          ExtractedCTF.GetDevice(Intent.Read),
                                                                          ExtractedFT.GetDevice(Intent.Write),
                                                                          ExtractedCTF.ElementsComplex,
                                                                          1);
                                    }
                                    else
                                    {
                                        GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                                   ParticleDiameterPix / 2f * 1.3f,
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
                                               ExtractedCropped.GetDevice(Intent.Write),
                                               new int3(SizeRefine).Slice(),
                                               new int3(RelevantSizes[f]).Slice(),
                                               (uint)CurBatch);


                                    GPU.MultiParticleDiff(EwaldResults[iewald],
                                                          new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                                          SizeRefine,
                                                          new[] { RelevantSizes[f] },
                                                          new float[CurBatch * 2],
                                                          Helper.ToInterleaved(ImageAngles),
                                                          MagnificationCorrection.ToVec(),
                                                          SpeciesCTFWeights[species].GetDeviceSlice(f, Intent.Read),
                                                          PhaseCorrectionAll.GetDeviceSlice(f, Intent.Read),
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
                                    ScoreAB += ResultP[i * 3 + 0] + ResultQ[i * 3 + 0];
                                    ScoreA2 += ResultP[i * 3 + 1] + ResultQ[i * 3 + 1];
                                    ScoreB2 += ResultP[i * 3 + 2] + ResultQ[i * 3 + 2];
                                    //Debug.WriteLine(Result[i * 3 + 0]);
                                    //Debug.WriteLine(Result[i * 3 + 1]);
                                    //Debug.WriteLine(Result[i * 3 + 2]);
                                }
                            }
                        }

                        PhaseCorrectionAll.Dispose();
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

                    //foreach (var image in FrameData)
                    //    image.FreeDevice();

                    double Score = ScoreAB / Math.Max(1e-10, Math.Sqrt(ScoreA2 * ScoreB2)) * NParticlesOverall * NFrames;
                    Score *= 100;

                    Console.WriteLine(Score);

                    return Score;
                };

                Func<double[], double[]> CTFGrad = input =>
                {
                    //Stopwatch Watch = new Stopwatch();
                    //Watch.Start();

                    double Delta = 0.001;
                    double Delta2 = Delta * 2;

                    double[] Deltas = { Delta, -Delta };

                    double[] Result = new double[input.Length];
                    double[] ScoresAB = new double[input.Length * 2];
                    double[] ScoresA2 = new double[input.Length * 2];
                    double[] ScoresB2 = new double[input.Length * 2];
                    int[] ScoresSamples = new int[input.Length * 2];

                    float[][] PerParticleScoresAB = Helper.ArrayOfFunction(i => new float[(CTFStepTypes[i] & CTFOptimizationTypes.Doming) == 0 ? NParticlesOverall * 2 : NParticlesOverall * NFrames * 2], CTFStepTypes.Length);
                    float[][] PerParticleScoresA2 = Helper.ArrayOfFunction(i => new float[(CTFStepTypes[i] & CTFOptimizationTypes.Doming) == 0 ? NParticlesOverall * 2 : NParticlesOverall * NFrames * 2], CTFStepTypes.Length);
                    float[][] PerParticleScoresB2 = Helper.ArrayOfFunction(i => new float[(CTFStepTypes[i] & CTFOptimizationTypes.Doming) == 0 ? NParticlesOverall * 2 : NParticlesOverall * NFrames * 2], CTFStepTypes.Length);

                    if (BFGSIterations-- <= 0)
                        return Result;

                    if (MathHelper.AllEqual(input, OldInput))
                        return OldGradient;

                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                    foreach (var species in allSpecies)
                    {
                        if (!SpeciesParticles.ContainsKey(species) ||
                            species.ResolutionRefinement > (float)optionsMPA.MinimumCTFRefinementResolution)
                            continue;

                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        int SuperresFactor = SpeciesCTFSuperresFactor[species];
                        int BatchSize = optionsMPA.BatchSize;
                        BatchSize /= SuperresFactor * SuperresFactor;

                        float[] ResultP = new float[BatchSize * 3];
                        float[] ResultQ = new float[BatchSize * 3];

                        int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeRefineSuper = SizeRefine * SuperresFactor;
                        int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SuperresFactor;

                        int SpeciesOffset = SpeciesParticleIDRanges[species].Start;

                        float AngPixRefine = species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                        int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);

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
                        Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                        for (int t = 0; t < NFrames; t++)
                            GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                        PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                                        PhaseCorrection.Dims.Slice(),
                                        new int3(RelevantSizes[t]).Slice(),
                                        1);
                        PhaseCorrection.Dispose();

                        Image GammaCorrection = new Image(new int3(SizeRefineSuper, SizeRefineSuper, 1), true);

                        bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                        float[][] EwaldResults = { ResultP, ResultQ };

                        Stopwatch WatchZernike = new Stopwatch();

                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                            float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                            for (int f = 0; f < RedNFrames; f++)
                            {
                                float3[] CoordinatesFrame = new float3[CurBatch];
                                float3[] AnglesFrame = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                    AnglesFrame[p] = AnglesMoving[p * NFrames + f];// * Helper.ToRad;
                                }

                                float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f);// AnglesFrame;

                                float3[] Defoci = new float3[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                }

                                GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                        Extracted.GetDevice(Intent.Write),
                                        FrameData[f].Dims.Slice(),
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


                                SetDefocusFromVector(input, this, true);

                                //WatchZernike.Start();
                                GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper, GammaCorrection);
                                CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion, CoordsCTF);
                                //WatchZernike.Stop();

                                for (int iparam = 0; iparam < CTFStepTypes.Length; iparam++)
                                {
                                    if ((CurrentOptimizationTypeCTF & CTFStepTypes[iparam]) == 0)
                                        continue;

                                    for (int idelta = 0; idelta < 2; idelta++)
                                    {
                                        double[] InputAltered = input.ToArray();

                                        if (CTFStepTypes[iparam] == CTFOptimizationTypes.Defocus)
                                        {
                                            for (int i = 0; i < NParticles; i++)
                                                InputAltered[i] += Deltas[idelta];
                                        }
                                        else if (CTFStepTypes[iparam] == CTFOptimizationTypes.AstigmatismDelta)
                                        {
                                            for (int i = 0; i < GridCTFDefocusDelta.Values.Length; i++)
                                                InputAltered[NParticlesOverall + i] += Deltas[idelta];
                                        }
                                        else if (CTFStepTypes[iparam] == CTFOptimizationTypes.AstigmatismAngle)
                                        {
                                            for (int i = 0; i < GridCTFDefocusAngle.Values.Length; i++)
                                                InputAltered[NParticlesOverall + GridCTFDefocusDelta.Values.Length + i] += Deltas[idelta];
                                        }
                                        else if (CTFStepTypes[iparam] == CTFOptimizationTypes.Cs)
                                        {
                                            for (int i = 0; i < GridCTFCs.Values.Length; i++)
                                                InputAltered[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + i] += Deltas[idelta];
                                        }
                                        else if (CTFStepTypes[iparam] == CTFOptimizationTypes.Doming)
                                        {
                                            for (int i = 0; i < GridCTFDoming.Values.Length; i++)
                                                InputAltered[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + GridCTFCs.Values.Length + i] += Deltas[idelta];
                                        }
                                        else
                                        {
                                            InputAltered[InputAltered.Length - CTFStepTypes.Length + iparam] += Deltas[idelta];
                                        }

                                        SetDefocusFromVector(InputAltered, this, true);

                                        for (int p = 0; p < CurBatch; p++)
                                            CoordinatesFrame[p].Z = Particles[batchStart + p].GetSplineCoordinateZ().Interp(DoseInterpolationSteps[f]);

                                        ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                        float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);
                                        for (int p = 0; p < CurBatch; p++)
                                        {
                                            Defoci[p].X = ImageCoords[p].Z;
                                            Defoci[p].Y = Astigmatism[p].X;
                                            Defoci[p].Z = Astigmatism[p].Y;
                                        }

                                        //WatchZernike.Start();
                                        if ((CTFStepTypes[iparam] & CTFOptimizationTypes.Zernike2) != 0 || (CTFStepTypes[iparam] & CTFOptimizationTypes.Zernike4) != 0)
                                            GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper, GammaCorrection);
                                        //WatchZernike.Stop();

                                        if ((CTFStepTypes[iparam] & CTFOptimizationTypes.Distortion) != 0)
                                            CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion, CoordsCTF);

                                        for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                        {
                                            if (species.DoEwald)
                                            {
                                                GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, EwaldReverse[iewald], ExtractedCTF, true);

                                                GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                                  ExtractedCTF.GetDevice(Intent.Read),
                                                                                  ExtractedFT.GetDevice(Intent.Write),
                                                                                  ExtractedCTF.ElementsComplex,
                                                                                  1);
                                            }
                                            else
                                            {
                                                GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                                           ParticleDiameterPix / 2f * 1.3f,
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
                                                       new int3(RelevantSizes[f]).Slice(),
                                                       (uint)CurBatch);

                                            GPU.MultiParticleDiff(EwaldResults[iewald],
                                                                  new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                                                  SizeRefine,
                                                                  new[] { RelevantSizes[f] },
                                                                  new float[CurBatch * 2],
                                                                  Helper.ToInterleaved(ImageAngles),
                                                                  MagnificationCorrection.ToVec(),
                                                                  SpeciesCTFWeights[species].GetDeviceSlice(f, Intent.Read),
                                                                  PhaseCorrectionAll.GetDeviceSlice(f, Intent.Read),
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

                                        if ((CTFStepTypes[iparam] & CTFOptimizationTypes.Doming) == 0)
                                        {
                                            for (int i = 0; i < CurBatch; i++)
                                            {
                                                PerParticleScoresAB[iparam][(SpeciesOffset + batchStart + i) * 2 + idelta] += ResultP[i * 3 + 0] + ResultQ[i * 3 + 0];
                                                PerParticleScoresA2[iparam][(SpeciesOffset + batchStart + i) * 2 + idelta] += ResultP[i * 3 + 1] + ResultQ[i * 3 + 1];
                                                PerParticleScoresB2[iparam][(SpeciesOffset + batchStart + i) * 2 + idelta] += ResultP[i * 3 + 2] + ResultQ[i * 3 + 2];
                                            }
                                        }
                                        else
                                        {
                                            for (int i = 0; i < CurBatch; i++)
                                            {
                                                PerParticleScoresAB[iparam][((SpeciesOffset + batchStart + i) * NFrames + f) * 2 + idelta] = ResultP[i * 3 + 0] + ResultQ[i * 3 + 0];
                                                PerParticleScoresA2[iparam][((SpeciesOffset + batchStart + i) * NFrames + f) * 2 + idelta] = ResultP[i * 3 + 1] + ResultQ[i * 3 + 1];
                                                PerParticleScoresB2[iparam][((SpeciesOffset + batchStart + i) * NFrames + f) * 2 + idelta] = ResultP[i * 3 + 2] + ResultQ[i * 3 + 2];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        //Console.WriteLine($"Zernike: {WatchZernike.ElapsedMilliseconds}");

                        CoordsCTF.Dispose();
                        PhaseCorrectionAll.Dispose();
                        Extracted.Dispose();
                        ExtractedFT.Dispose();
                        ExtractedRefineSuper.Dispose();
                        ExtractedCropped.Dispose();
                        ExtractedCroppedFT.Dispose();
                        ExtractedCTF.Dispose();

                        GammaCorrection.Dispose();

                        GPU.DestroyFFTPlan(PlanForwSuper);
                        GPU.DestroyFFTPlan(PlanBackSuper);
                        GPU.DestroyFFTPlan(PlanForw);
                    }

                    for (int iparam = 0; iparam < CTFStepTypes.Length; iparam++)
                    {
                        if ((CurrentOptimizationTypeCTF & CTFStepTypes[iparam]) == 0)
                            continue;

                        if (CTFStepTypes[iparam] == CTFOptimizationTypes.Defocus)
                        {
                            for (int i = 0; i < NParticlesOverall; i++)
                            {
                                double ScorePlus = PerParticleScoresAB[iparam][i * 2 + 0] /
                                                   Math.Max(1e-10, Math.Sqrt(PerParticleScoresA2[iparam][i * 2 + 0] *
                                                                             PerParticleScoresB2[iparam][i * 2 + 0]));
                                double ScoreMinus = PerParticleScoresAB[iparam][i * 2 + 1] /
                                                    Math.Max(1e-10, Math.Sqrt(PerParticleScoresA2[iparam][i * 2 + 1] *
                                                                              PerParticleScoresB2[iparam][i * 2 + 1]));
                                Result[i] = (ScorePlus - ScoreMinus) / Delta2 * NFrames * 100;
                            }
                        }
                        else if (CTFStepTypes[iparam] == CTFOptimizationTypes.AstigmatismDelta)
                        {
                            for (int iwiggle = 0; iwiggle < NWiggleDifferentiableAstigmatism; iwiggle++)
                            {
                                double SumGrad = 0;
                                double SumWeights = 0;
                                double SumWeightsGrad = 0;

                                int[] Indices = AllWiggleWeightsAstigmatism[iwiggle].indices;
                                float[] Weights = AllWiggleWeightsAstigmatism[iwiggle].weights;

                                for (int i = 0; i < Indices.Length; i++)
                                {
                                    int id = Indices[i];
                                    float ABPlus = PerParticleScoresAB[iparam][id * 2 + 0];
                                    float ABMinus = PerParticleScoresAB[iparam][id * 2 + 1];
                                    float A2Plus = PerParticleScoresA2[iparam][id * 2 + 0];
                                    float A2Minus = PerParticleScoresA2[iparam][id * 2 + 1];
                                    float B2Plus = PerParticleScoresB2[iparam][id * 2 + 0];
                                    float B2Minus = PerParticleScoresB2[iparam][id * 2 + 1];

                                    SumWeights += Math.Abs(Weights[i]) * Math.Sqrt(A2Plus + A2Minus);
                                    SumWeightsGrad += Math.Abs(Weights[i]);

                                    double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                                    ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                                    SumGrad += Weights[i] * Math.Sqrt(A2Plus + A2Minus) * Grad;
                                }

                                Result[NParticlesOverall + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                            }
                        }
                        else if (CTFStepTypes[iparam] == CTFOptimizationTypes.AstigmatismAngle)
                        {
                            for (int iwiggle = 0; iwiggle < NWiggleDifferentiableAstigmatism; iwiggle++)
                            {
                                double SumGrad = 0;
                                double SumWeights = 0;
                                double SumWeightsGrad = 0;

                                int[] Indices = AllWiggleWeightsAstigmatism[iwiggle].indices;
                                float[] Weights = AllWiggleWeightsAstigmatism[iwiggle].weights;

                                for (int i = 0; i < Indices.Length; i++)
                                {
                                    int id = Indices[i];
                                    float ABPlus = PerParticleScoresAB[iparam][id * 2 + 0];
                                    float ABMinus = PerParticleScoresAB[iparam][id * 2 + 1];
                                    float A2Plus = PerParticleScoresA2[iparam][id * 2 + 0];
                                    float A2Minus = PerParticleScoresA2[iparam][id * 2 + 1];
                                    float B2Plus = PerParticleScoresB2[iparam][id * 2 + 0];
                                    float B2Minus = PerParticleScoresB2[iparam][id * 2 + 1];

                                    SumWeights += Math.Abs(Weights[i]) * Math.Sqrt(A2Plus + A2Minus);
                                    SumWeightsGrad += Math.Abs(Weights[i]);

                                    double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                                   ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                                    SumGrad += Weights[i] * Math.Sqrt(A2Plus + A2Minus) * Grad;
                                }

                                Result[NParticlesOverall + GridCTFDefocusDelta.Values.Length + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                            }
                        }
                        else if (CTFStepTypes[iparam] == CTFOptimizationTypes.Cs)
                        {
                            for (int iwiggle = 0; iwiggle < NWiggleDifferentiableAstigmatism; iwiggle++)
                            {
                                double SumGrad = 0;
                                double SumWeights = 0;
                                double SumWeightsGrad = 0;

                                int[] Indices = AllWiggleWeightsAstigmatism[iwiggle].indices;
                                float[] Weights = AllWiggleWeightsAstigmatism[iwiggle].weights;

                                for (int i = 0; i < Indices.Length; i++)
                                {
                                    int id = Indices[i];
                                    float ABPlus = PerParticleScoresAB[iparam][id * 2 + 0];
                                    float ABMinus = PerParticleScoresAB[iparam][id * 2 + 1];
                                    float A2Plus = PerParticleScoresA2[iparam][id * 2 + 0];
                                    float A2Minus = PerParticleScoresA2[iparam][id * 2 + 1];
                                    float B2Plus = PerParticleScoresB2[iparam][id * 2 + 0];
                                    float B2Minus = PerParticleScoresB2[iparam][id * 2 + 1];

                                    SumWeights += Math.Abs(Weights[i]) * Math.Sqrt(A2Plus + A2Minus);
                                    SumWeightsGrad += Math.Abs(Weights[i]);

                                    double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                                   ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                                    SumGrad += Weights[i] * Math.Sqrt(A2Plus + A2Minus) * Grad;
                                }

                                Result[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                            }
                        }
                        else if (CTFStepTypes[iparam] == CTFOptimizationTypes.Doming)
                        {
                            for (int iwiggle = 0; iwiggle < NWiggleDifferentiableDoming; iwiggle++)
                            {
                                double SumGrad = 0;
                                double SumWeights = 0;
                                double SumWeightsGrad = 0;

                                int[] Indices = AllWiggleWeightsDoming[iwiggle].indices;
                                float[] Weights = AllWiggleWeightsDoming[iwiggle].weights;

                                for (int i = 0; i < Indices.Length; i++)
                                {
                                    int id = Indices[i];
                                    float ABPlus = PerParticleScoresAB[iparam][id * 2 + 0];
                                    float ABMinus = PerParticleScoresAB[iparam][id * 2 + 1];
                                    float A2Plus = PerParticleScoresA2[iparam][id * 2 + 0];
                                    float A2Minus = PerParticleScoresA2[iparam][id * 2 + 1];
                                    float B2Plus = PerParticleScoresB2[iparam][id * 2 + 0];
                                    float B2Minus = PerParticleScoresB2[iparam][id * 2 + 1];

                                    SumWeights += Math.Abs(Weights[i]) * Math.Sqrt(A2Plus + A2Minus);
                                    SumWeightsGrad += Math.Abs(Weights[i]);

                                    double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                                    ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                                    SumGrad += Weights[i] * Math.Sqrt(A2Plus + A2Minus) * Grad;
                                }

                                Result[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + GridCTFCs.Values.Length + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                            }
                        }
                        else
                        {
                            double ABPlus = 0, ABMinus = 0;
                            double A2Plus = 0, A2Minus = 0;
                            double B2Plus = 0, B2Minus = 0;
                            for (int i = 0; i < NParticlesOverall; i++)
                            {
                                ABPlus += PerParticleScoresAB[iparam][i * 2 + 0];
                                ABMinus += PerParticleScoresAB[iparam][i * 2 + 1];

                                A2Plus += PerParticleScoresA2[iparam][i * 2 + 0];
                                A2Minus += PerParticleScoresA2[iparam][i * 2 + 1];

                                B2Plus += PerParticleScoresB2[iparam][i * 2 + 0];
                                B2Minus += PerParticleScoresB2[iparam][i * 2 + 1];
                            }

                            double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                           ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                            Result[Result.Length - CTFStepTypes.Length + iparam] = Grad * NParticlesOverall * NFrames * 100;
                        }
                    }

                    OldInput = input.ToList().ToArray();
                    OldGradient = Result.ToList().ToArray();

                    //Watch.Stop();
                    //Console.WriteLine(Watch.ElapsedMilliseconds);

                    return Result;
                };

                #endregion

                #region Grid search for per-particle defoci

                Func<double[], double[]> DefocusGridSearch = input =>
                {
                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                    foreach (var species in allSpecies)
                    {
                        if (!SpeciesParticles.ContainsKey(species))
                            continue;
                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        int SuperresFactor = SpeciesCTFSuperresFactor[species];
                        int BatchSize = optionsMPA.BatchSize;
                        BatchSize /= SuperresFactor * SuperresFactor;

                        float[] ResultP = new float[BatchSize * 3];
                        float[] ResultQ = new float[BatchSize * 3];

                        int SpeciesOffset = SpeciesParticleIDRanges[species].Start;

                        int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                        int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];

                        float AngPixRefine = species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                        int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);

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
                        Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                        for (int f = 0; f < NFrames; f++)
                            GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                        PhaseCorrectionAll.GetDeviceSlice(f, Intent.Write),
                                        PhaseCorrection.Dims.Slice(),
                                        new int3(RelevantSizes[f]).Slice(),
                                        1);
                        PhaseCorrection.Dispose();

                        Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                        bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                        float[][] EwaldResults = { ResultP, ResultQ };

                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                            float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                            List<float4>[] AllSearchValues = Helper.ArrayOfFunction(i => new List<float4>(), CurBatch);
                            List<float4>[] CurrentSearchValues = Helper.ArrayOfFunction(i => new List<float4>(), CurBatch);
                            decimal GridSearchDelta = 0.002M;
                            foreach (var list in CurrentSearchValues)
                                for (decimal d = -0.02M; d <= 0.02M; d += GridSearchDelta)
                                    list.Add(new float4((float)d, 0, 0, 0));
                            //for (decimal d = 0M; d <= 0M; d += GridSearchDelta)
                            //    list.Add(new float2((float)d, 0));

                            for (int irefine = 0; irefine < 4; irefine++)
                            {
                                for (int f = 0; f < RedNFrames; f++)
                                {
                                    float3[] CoordinatesFrame = new float3[CurBatch];
                                    float3[] AnglesFrame = new float3[CurBatch];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                        AnglesFrame[p] = AnglesMoving[p * NFrames + f];// * Helper.ToRad;
                                    }

                                    float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                    float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f);// AnglesFrame;

                                    float3[] Defoci = new float3[CurBatch];
                                    int3[] ExtractOrigins = new int3[CurBatch];
                                    float3[] ResidualShifts = new float3[BatchSize];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                        ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                        ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    }

                                    GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                                Extracted.GetDevice(Intent.Write),
                                                FrameData[f].Dims.Slice(),
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

                                    for (int idelta = 0; idelta < CurrentSearchValues[0].Count; idelta++)
                                    {
                                        double[] InputAltered = input.ToArray();
                                        for (int i = 0; i < CurBatch; i++)
                                            InputAltered[SpeciesOffset + batchStart + i] += CurrentSearchValues[i][idelta].X;

                                        SetDefocusFromVector(InputAltered, this, true);

                                        for (int p = 0; p < CurBatch; p++)
                                            CoordinatesFrame[p].Z = Particles[batchStart + p].GetSplineCoordinateZ().Interp(DoseInterpolationSteps[f]);

                                        ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                        float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);
                                        for (int p = 0; p < CurBatch; p++)
                                        {
                                            Defoci[p].X = ImageCoords[p].Z;
                                            Defoci[p].Y = Astigmatism[p].X;
                                            Defoci[p].Z = Astigmatism[p].Y;
                                        }

                                        for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                        {
                                            if (species.DoEwald)
                                            {
                                                GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, EwaldReverse[iewald], ExtractedCTF, true);

                                                GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                                    ExtractedCTF.GetDevice(Intent.Read),
                                                                                    ExtractedFT.GetDevice(Intent.Write),
                                                                                    ExtractedCTF.ElementsComplex,
                                                                                    1);
                                            }
                                            else
                                            {
                                                GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                                            ParticleDiameterPix / 2f * 1.3f,
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
                                                       ExtractedCropped.GetDevice(Intent.Write),
                                                       new int3(SizeRefine).Slice(),
                                                       new int3(RelevantSizes[f]).Slice(),
                                                       (uint)CurBatch);

                                            GPU.MultiParticleDiff(EwaldResults[iewald],
                                                                    new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                                                    SizeRefine,
                                                                    new[] { RelevantSizes[f] },
                                                                    new float[CurBatch * 2],
                                                                    Helper.ToInterleaved(ImageAngles),
                                                                    MagnificationCorrection.ToVec(),
                                                                    SpeciesCTFWeights[species].GetDeviceSlice(f, Intent.Read),
                                                                    PhaseCorrectionAll.GetDeviceSlice(f, Intent.Read),
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
                                            CurrentSearchValues[i][idelta] += new float4(0,
                                                                                         ResultP[i * 3 + 0] + ResultQ[i * 3 + 0],
                                                                                         ResultP[i * 3 + 1] + ResultQ[i * 3 + 1],
                                                                                         ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]);
                                    }
                                }

                                GridSearchDelta /= 2;
                                for (int i = 0; i < CurBatch; i++)
                                {
                                    CurrentSearchValues[i].Sort((a, b) => -((a.Y / Math.Max(1e-20, Math.Sqrt(a.Z * a.W))).CompareTo(b.Y / Math.Max(1e-20, Math.Sqrt(b.Z * b.W)))));
                                    AllSearchValues[i].AddRange(CurrentSearchValues[i]);

                                    List<float4> NewSearchValues = new List<float4>();
                                    for (int j = 0; j < 10; j++)
                                    {
                                        NewSearchValues.Add(new float4(CurrentSearchValues[i][j].X + (float)GridSearchDelta, 0, 0, 0));
                                        NewSearchValues.Add(new float4(CurrentSearchValues[i][j].X - (float)GridSearchDelta, 0, 0, 0));
                                    }

                                    CurrentSearchValues[i] = NewSearchValues;
                                }
                            }

                            for (int i = 0; i < CurBatch; i++)
                            {
                                AllSearchValues[i].Sort((a, b) => -((a.Y / Math.Max(1e-10, Math.Sqrt(a.Z * a.W))).CompareTo(b.Y / Math.Max(1e-10, Math.Sqrt(b.Z * b.W)))));
                                input[SpeciesOffset + batchStart + i] += AllSearchValues[i][0].X;
                            }
                        }

                        //SumAll.WriteMRC("d_sumall.mrc", true);

                        CoordsCTF.Dispose();
                        GammaCorrection.Dispose();
                        PhaseCorrectionAll.Dispose();
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

                    return input;
                };

                #endregion

                GPU.CheckGPUExceptions();

                BroydenFletcherGoldfarbShanno OptimizerWarp = new BroydenFletcherGoldfarbShanno(InitialParametersWarp.Length, WarpEval, WarpGrad);
                BroydenFletcherGoldfarbShanno OptimizerDefocus = new BroydenFletcherGoldfarbShanno(InitialParametersDefocus.Length, CTFEval, CTFGrad);

                bool NeedReextraction = true;

                SetWarpFromVector(InitialParametersWarp, this, true);
                SetDefocusFromVector(InitialParametersDefocus, this, true);

                for (int ioptim = 0; ioptim < optionsMPA.NIterations; ioptim++)
                {
                    foreach (var species in allSpecies)
                        species.CurrentMaxShellRefinement = (int)Math.Round(MathHelper.Lerp(optionsMPA.InitialResolutionPercent / 100f,
                                                                                            1f,
                                                                                            optionsMPA.NIterations == 1 ? 1 : ((float)ioptim / (optionsMPA.NIterations - 1))) *
                                                                            species.HalfMap1Projector[GPUID].Dims.X / 2);

                    foreach (var step in OptimizationStepsWarp)
                    {
                        if (NeedReextraction)
                        {
                            progressCallback($"Re-extracting particles for optimization iteration {ioptim + 1}/{optionsMPA.NIterations}");
                            AllocateParticleMemory();
                            ReextractPaddedParticles(false);

                            //WarpEval(InitialParametersWarp);
                            //CTFEval(InitialParametersDefocus);

                            GPU.CheckGPUExceptions();
                        }
                        NeedReextraction = false;

                        progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, " + step.Name);

                        BFGSIterations = step.Iterations;
                        CurrentOptimizationTypeWarp = step.Type;
                        CurrentWeightsDict = SpeciesCTFWeights;

                        OptimizerWarp.Maximize(InitialParametersWarp);

                        OldInput = null;

                        GPU.CheckGPUExceptions();
                    }

                    if (allSpecies.Any(s => s.ResolutionRefinement < (float)optionsMPA.MinimumCTFRefinementResolution))
                    {
                        //ReextractPaddedParticles();
                        //WarpEval(InitialParametersWarp);

                        if (ioptim == 0 && optionsMPA.DoDefocusGridSearch)
                        {
                            FreeParticleMemory();

                            CTFEval(InitialParametersDefocus);

                            progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, defocus grid search");

                            InitialParametersDefocus = DefocusGridSearch(InitialParametersDefocus);

                            NeedReextraction = true;

                            GPU.CheckGPUExceptions();
                        }

                        //CurrentWeightsDict = SpeciesFrameWeights;
                        //ReextractPaddedParticles();
                        //WarpEval(InitialParametersWarp);

                        foreach (var step in OptimizationStepsCTF)
                        {
                            FreeParticleMemory();

                            progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, " + step.Name);

                            BFGSIterations = step.Iterations;
                            CurrentOptimizationTypeCTF = step.Type;
                            CurrentWeightsDict = SpeciesCTFWeights;

                            OptimizerDefocus.Maximize(InitialParametersDefocus);

                            OldInput = null;
                            NeedReextraction = true;

                            GPU.CheckGPUExceptions();
                        }
                    }
                }

                FreeParticleMemory();

                SetWarpFromVector(InitialParametersWarp, this, true);
                SetDefocusFromVector(InitialParametersDefocus, this, true);

                #region Compute FSC between refs and particles to estimate frame and micrograph weights

                if (true)
                {
                    progressCallback($"Calculating FRC between projections and particles for weight optimization");

                    int FSCLength = 128;
                    Image FSC = new Image(new int3(FSCLength, FSCLength, RedNFrames * 3), true);
                    Image FSCPerParticle = new Image(new int3(FSCLength / 2, NParticlesOverall * 3, 1));
                    //float[][] FSCData = FSC.GetHost(Intent.ReadWrite);
                    //float[][] FSCPerParticleData = FSCPerParticle.GetHost(Intent.ReadWrite);
                    Image PhaseResiduals = new Image(new int3(FSCLength, FSCLength, 2), true);

                    Star TableOut = new Star(new string[] { "wrpNormCoordinateX", "wrpNormCoordinateY" });

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

                        //Image CorrAB = new Image(new int3(SizeRefine, SizeRefine, NFrames), true);
                        //Image CorrA2 = new Image(new int3(SizeRefine, SizeRefine, NFrames), true);
                        //Image CorrB2 = new Image(new int3(SizeRefine, SizeRefine, NFrames), true);

                        float ScaleFactor = (float)Species.PixelSize * (FSCLength / 2 - 1) /
                                            (float)(Species.ResolutionRefinement / 2 * (SizeRefine / 2 - 1));

                        {
                            int SuperresFactor = SpeciesCTFSuperresFactor[Species];
                            int BatchSize = optionsMPA.BatchSize;
                            BatchSize /= SuperresFactor * SuperresFactor;

                            int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[Species];
                            int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;
                            int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[Species];
                            float AngPixRefine = Species.ResolutionRefinement / 2;
                            int ParticleDiameterPix = (int)(Species.DiameterAngstrom / AngPixRefine);

                            Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);   // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine
                            Image CoordsCTFCropped = CTF.GetCTFCoords(SizeRefine, SizeRefine, CTF.Distortion);

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
                                    float2 Coords = new float2(CoordinatesMoving[i * NFrames].X, CoordinatesMoving[i * NFrames].Y);
                                    Coords /= ImageDimensionsPhysical;
                                    TableOut.AddRow(new string[] { Coords.X.ToString(CultureInfo.InvariantCulture),
                                                                   Coords.Y.ToString(CultureInfo.InvariantCulture) });
                                }

                                for (int f = 0; f < RedNFrames; f++)
                                {
                                    float3[] CoordinatesFrame = new float3[CurBatch];
                                    float3[] AnglesFrame = new float3[CurBatch];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                        AnglesFrame[p] = AnglesMoving[p * NFrames + f];
                                    }

                                    float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                    float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f);
                                    float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);

                                    float3[] Defoci = new float3[CurBatch];
                                    int3[] ExtractOrigins = new int3[CurBatch];
                                    float3[] ResidualShifts = new float3[BatchSize];
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                        ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                        ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                        Defoci[p].X = ImageCoords[p].Z;
                                        Defoci[p].Y = Astigmatism[p].X;
                                        Defoci[p].Z = Astigmatism[p].Y;
                                    }

                                    for (int iewald = 0; iewald < (Species.DoEwald ? 2 : 1); iewald++)
                                    {
                                        GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                                    Extracted.GetDevice(Intent.Write),
                                                    FrameData[f].Dims.Slice(),
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
                                            GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, EwaldReverse[iewald], ExtractedCTF, true);

                                            GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                                               ExtractedCTF.GetDevice(Intent.Read),
                                                                               ExtractedFT.GetDevice(Intent.Write),
                                                                               ExtractedCTF.ElementsComplex,
                                                                               1);
                                        }
                                        else
                                        {
                                            GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                                       ParticleDiameterPix / 2f * 1.3f,
                                                       16 * AngPixExtract / AngPixRefine,
                                                       true,
                                                       (uint)CurBatch);

                                        GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                                ExtractedCroppedFT.GetDevice(Intent.Write),
                                                new int3(SizeRefine, SizeRefine, 1),
                                                (uint)CurBatch,
                                                PlanForw);

                                        ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                        GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTFCropped, null, f, ExtractedCTF, true, true, true);

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
                    SpeciesCTFWeights[pair.Key].Dispose();
                    SpeciesFrameWeights[pair.Key].Dispose();
                    GPU.FreeDevice(SpeciesParticleSubsets[pair.Key]);

                    pair.Key.HalfMap1Projector[GPUID].FreeDevice();
                    pair.Key.HalfMap2Projector[GPUID].FreeDevice();
                }

                #endregion
            }

            GPU.CheckGPUExceptions();

            #region Update reconstructions with newly aligned particles

            progressCallback($"Extracting and back-projecting particles...");
            if (true)
                foreach (var species in allSpecies)
                {
                    if (SpeciesParticles[species].Length == 0)
                        continue;

                    Projector[] Reconstructions = { species.HalfMap1Reconstruction[GPUID], species.HalfMap2Reconstruction[GPUID] };

                    float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                    int BatchSize = optionsMPA.BatchSize;

                    CTF MaxDefocusCTF = CTF.GetCopy();
                    float ExpectedResolution = Math.Max((float)dataSource.PixelSizeMean * 2, (float)species.GlobalResolution * 0.8f);
                    int ExpectedBoxSize = (int)(species.DiameterAngstrom / (ExpectedResolution / 2)) * 2;
                    int MinimumBoxSize = Math.Max(ExpectedBoxSize, MaxDefocusCTF.GetAliasingFreeSize(ExpectedResolution, (float)(species.DiameterAngstrom / AngPixExtract)));
                    int CTFSuperresFactor = (int)Math.Ceiling((float)MinimumBoxSize / ExpectedBoxSize);

                    float AliasingFreeDiameter = MaxDefocusCTF.GetAliasingFreeSize(ExpectedResolution, (float)(species.DiameterAngstrom / AngPixExtract));

                    BatchSize /= CTFSuperresFactor * CTFSuperresFactor;
                    BatchSize = 1;

                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * CTFSuperresFactor;

                    Image CTFCoords = CTF.GetCTFCoords(SizeFullSuper, SizeFullSuper, CTF.Distortion);
                    float2[] CTFCoordsData = CTFCoords.GetHostComplexCopy()[0];
                    Image CTFCoordsP = CTF.GetCTFPCoords(SizeFullSuper, SizeFullSuper, CTF.Distortion);
                    float2[] CTFCoordsPData = CTFCoordsP.GetHostComplexCopy()[0];
                    Image CTFCoordsCropped = CTF.GetCTFCoords(SizeFull, SizeFull, CTF.Distortion);

                    Image GammaCorrection = CTF.GetGammaCorrection(AngPixExtract, SizeFullSuper);

                    GPU.CheckGPUExceptions();

                    float[] PQSigns = new float[CTFCoordsData.Length];
                    CTF.PrecomputePQSigns(SizeFullSuper, 2, species.EwaldReverse, CTFCoordsData, CTFCoordsPData, PQSigns);

                    GPU.CheckGPUExceptions();

                    Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixExtract, SizeFullSuper);

                    Image IntermediateMaskAngles = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, 2), true);
                    Image IntermediateFTCorr = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image IntermediateCTFP = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);

                    Image MaskParticle = new Image(new int3(SizeFullSuper, SizeFullSuper, 1));
                    MaskParticle.Fill(1);
                    MaskParticle.MaskSpherically((float)(species.DiameterAngstrom + 6) / AngPixExtract, 3, false);
                    MaskParticle.RemapToFT();

                    GPU.CheckGPUExceptions();

                    Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize));
                    Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize));
                    Image ExtractedCroppedFTp = new Image(new int3(SizeFull, SizeFull, BatchSize), true, true);
                    Image ExtractedCroppedFTq = new Image(new int3(SizeFull, SizeFull, BatchSize), true, true);

                    Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true);
                    Image ExtractedCTFCropped = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize), true);
                    Image CTFWeights = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize), true);

                    GPU.CheckGPUExceptions();

                    Image SourceWeight = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, 1), true);
                    {
                        CTF ProtoCTF = CTF.GetCopy();
                        ProtoCTF.Defocus = 0;
                        ProtoCTF.DefocusDelta = 0;
                        ProtoCTF.Cs = 0;
                        ProtoCTF.Amplitude = 1;

                        ProtoCTF.Scale = (decimal)dataSource.RelativeWeight;
                        ProtoCTF.Bfactor = (decimal)dataSource.RelativeBFactor.X;
                        ProtoCTF.BfactorDelta = (decimal)dataSource.RelativeBFactor.Y;
                        ProtoCTF.BfactorAngle = (decimal)dataSource.RelativeBFactor.Z;

                        GPU.CreateCTF(SourceWeight.GetDevice(Intent.Write),
                                      CTFCoordsCropped.GetDevice(Intent.Read),
                                      IntPtr.Zero,
                                      (uint)CTFCoordsCropped.ElementsSliceComplex,
                                      new[] { ProtoCTF.ToStruct() },
                                      false,
                                      1);
                    }

                    GPU.CheckGPUExceptions();

                    int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                    int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                    int PlanForw = GPU.CreateFFTPlan(new int3(SizeFull, SizeFull, 1), (uint)BatchSize);

                    if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                        throw new Exception("No FFT plans created!");

                    Particle[] ParticlesInside = SpeciesParticles[species].Where(p =>
                    {
                        float2 Center = new float2(p.Coordinates[0].X, p.Coordinates[0].Y);
                        float2 Rectangle = ImageDimensionsPhysical;
                        float FractionInside = MathHelper.CircleFractionInsideRectangle(Center, AliasingFreeDiameter / 2, new float2(0, 0), Rectangle);

                        return FractionInside > 0.0;
                    }).ToArray();

                    Console.WriteLine($"{ParticlesInside.Length} particles kept out of {SpeciesParticles[species].Length}, diameter = {AliasingFreeDiameter:F1}, area = {ImageDimensionsPhysical}");
                    Console.WriteLine($"Superres factor is {CTFSuperresFactor}");

                    Particle[][] SubsetParticles = { ParticlesInside.Where(p => p.RandomSubset == 0).ToArray(),
                                                     ParticlesInside.Where(p => p.RandomSubset == 1).ToArray() };

                    float4[][] SubsetMags = { ParticleMags.Where((m, p) => ParticlesInside[p].RandomSubset == 0).ToArray(),
                                              ParticleMags.Where((m, p) => ParticlesInside[p].RandomSubset == 1).ToArray()};

                    for (int isubset = 0; isubset < 2; isubset++)
                    {
                        Particle[] Particles = SubsetParticles[isubset];
                        int NParticles = Particles.Length;

                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                            float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                            for (int f = 0; f < RedNFrames; f++)
                            {
                                float3[] CoordinatesFrame = new float3[CurBatch];
                                float3[] AnglesFrame = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                    AnglesFrame[p] = AnglesMoving[p * NFrames + f];// * Helper.ToRad;
                                }

                                float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f);// AnglesFrame;
                                float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);

                                float3[] Defoci = new float3[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p].X = ImageCoords[p].Z;
                                    Defoci[p].Y = Astigmatism[p].X;
                                    Defoci[p].Z = Astigmatism[p].Y;
                                }

                                #region Image data

                                GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                            Extracted.GetDevice(Intent.Write),
                                            FrameData[f].Dims.Slice(),
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

                                CTF[] CTFParams = GetCTFParamsForOneFrame(AngPixExtract, Defoci, ImageCoords, f, false, false, false);

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

                                //CTF.ApplyPandQ(ExtractedFT,
                                //                CTFParams,
                                //                IntermediateFTCorr,
                                //                Extracted,
                                //                ExtractedCropped,
                                //                IntermediateCTFP,
                                //                IntermediateMaskAngles,
                                //                CTFCoordsData,
                                //                CTFCoordsPData,
                                //                MaskParticle,
                                //                PlanForw,
                                //                PlanBackSuper,
                                //                2,
                                //                species.EwaldReverse,
                                //                ExtractedCroppedFTp,
                                //                ExtractedCroppedFTq);

                                GetCTFsForOneFrame(AngPixExtract, Defoci, ImageCoords, CTFCoordsCropped, null, f, CTFWeights, true, true, true);
                                CTFWeights.MultiplySlices(SourceWeight);

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

                                // Try to correct motion-dampened amplitudes
                                //GetCTFsForOneFrame(AngPixExtract, Defoci, ImageCoords, CTFCoordsCropped, f, CTFWeights, true, true, true, true);
                                //GPU.MultiplySlices(ExtractedCTFCropped.GetDevice(Intent.Read),
                                //                    CTFWeights.GetDeviceSlice(f, Intent.Read),
                                //                    ExtractedCTFCropped.GetDevice(Intent.Write),
                                //                    CTFWeights.ElementsSliceReal,
                                //                    (uint)CurBatch);

                                #endregion

                                Reconstructions[isubset].BackProject(ExtractedCroppedFTp, ExtractedCTFCropped, ImageAngles, new Matrix2(SubsetMags[isubset][batchStart]), CTFParams[0].GetEwaldRadius(SizeFull, (float)species.PixelSize));
                                Reconstructions[isubset].BackProject(ExtractedCroppedFTq, ExtractedCTFCropped, ImageAngles, new Matrix2(SubsetMags[isubset][batchStart]), -CTFParams[0].GetEwaldRadius(SizeFull, (float)species.PixelSize));
                            }

                            GPU.CheckGPUExceptions();
                        }
                    }

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
                    SourceWeight.Dispose();

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

            for (int f = 0; f < FrameData.Length; f++)
                FrameData[f].Dispose();


            #endregion
        }

        public virtual long MultiParticleRefinementCalculateHostMemory(ProcessingOptionsMPARefine optionsMPA,
                                                                       Species[] allSpecies,
                                                                       DataSource dataSource)
        {
            long Result = 0;

            string DataHash = GetDataHash();
            int GPUID = GPU.GetDevice();

            NFrames = Math.Min(MapHeader.ReadFromFile(DataPath).Dimensions.Z, dataSource.FrameLimit);

            foreach (var species in allSpecies)
            {
                int NParticles = species.GetParticles(DataHash).Length;

                int Size = species.HalfMap1Projector[GPUID].Dims.X;
                int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;

                int[] RelevantSizes = GetRelevantImageSizes(SizeFull, (float)optionsMPA.BFactorWeightingThreshold).Select(v => Math.Min(Size, v)).ToArray();

                long OneSet = Helper.ArrayOfFunction(t => (new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles * sizeof(float), NFrames).Sum();
                if (species.DoEwald)
                    OneSet *= 2;

                Result += OneSet;
            }

            return Result;
        }

        public virtual long MultiParticleRefinementCalculateAvailableDeviceMemory(ProcessingOptionsMPARefine optionsMPA,
                                                                                  Species[] allSpecies,
                                                                                  DataSource dataSource)
        {

            string DataHash = GetDataHash();
            int GPUID = GPU.GetDevice();

            long Result = GPU.GetFreeMemory(GPUID);

            NFrames = Math.Min(MapHeader.ReadFromFile(DataPath).Dimensions.Z, dataSource.FrameLimit);

            foreach (var species in allSpecies)
            {
                species.HalfMap1Projector[GPUID].PutTexturesOnDevice();
                species.HalfMap2Projector[GPUID].PutTexturesOnDevice();
            }

            foreach (var species in allSpecies)
            {
                int NParticles = species.GetParticles(DataHash).Length;

                CTF MaxDefocusCTF = CTF.GetCopy();
                int MinimumBoxSize = Math.Max(species.HalfMap1Projector[GPUID].Dims.X, MaxDefocusCTF.GetAliasingFreeSize(species.ResolutionRefinement, (float)(species.DiameterAngstrom / species.ResolutionRefinement * 2)));
                int CTFSuperresFactor = (int)Math.Ceiling((float)MinimumBoxSize / species.HalfMap1Projector[GPUID].Dims.X);

                int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                int SizeRefineSuper = SizeRefine * CTFSuperresFactor;
                int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                int SizeFullSuper = species.HalfMap1Reconstruction[GPUID].Dims.X * CTFSuperresFactor;

                int BatchSize = optionsMPA.BatchSize;

                Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);
                Image PhaseCorrection = CTF.GetPhaseCorrection((float)species.PixelSize, SizeRefineSuper);

                Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize));
                Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                Image ExtractedCroppedFTRelevantSize = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true);

                int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                Result = Math.Min(Result, GPU.GetFreeMemory(GPUID));

                GPU.DestroyFFTPlan(PlanForw);
                GPU.DestroyFFTPlan(PlanBackSuper);
                GPU.DestroyFFTPlan(PlanForwSuper);

                ExtractedCTF.Dispose();
                ExtractedCroppedFTRelevantSize.Dispose();
                ExtractedCroppedFT.Dispose();
                ExtractedCropped.Dispose();
                ExtractedFT.Dispose();
                Extracted.Dispose();

                PhaseCorrection.Dispose();
                CoordsCTF.Dispose();
            }

            Result = Math.Max(0, Result - (1 << 30));   // Subtract 1 GB just in case

            return Result;
        }

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

        public float2[] GetMotionTrack(float2 position, int samples, bool localOnly = false)
        {
            if (OptionsMovement == null || OptionsMovement.Dimensions.Z <= 1)
                return null;

            int NFrames = (int)OptionsMovement.Dimensions.Z;
            float2[] Result = new float2[NFrames * samples];

            float StepZ = 1f / Math.Max(NFrames * samples - 1, 1);
            for (int z = 0; z < NFrames * samples; z++)
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

        public JsonNode ToMiniJson(string particleSuffix)
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
            Json["Mtn"] = MeanFrameMovement <= 0 ? null : MathF.Round((float)MeanFrameMovement, 2);

            // 💩 percentage
            Json["Jnk"] = MaskPercentage < 0 ? null : MathF.Round((float)MaskPercentage, 1);

            // Particle count for given suffix
            int ParticleCount = GetParticleCount(particleSuffix);
            Json["Ptc"] = ParticleCount < 0 ? null : ParticleCount;

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
    public class ProcessingOptionsMovieCTF : ProcessingOptionsBase
    {
        [WarpSerializable]
        public int Window { get; set; }
        [WarpSerializable]
        public decimal RangeMin { get; set; }
        [WarpSerializable]
        public decimal RangeMax { get; set; }
        [WarpSerializable]
        public int Voltage { get; set; }
        [WarpSerializable]
        public decimal Cs { get; set; }
        [WarpSerializable]
        public decimal Cc { get; set; }
        [WarpSerializable]
        public decimal Amplitude { get; set; }
        [WarpSerializable]
        public bool DoPhase { get; set; }
        [WarpSerializable]
        public bool UseMovieSum { get; set; }
        [WarpSerializable]
        public decimal ZMin { get; set; }
        [WarpSerializable]
        public decimal ZMax { get; set; }
        [WarpSerializable]
        public int3 GridDims { get; set; }
        [WarpSerializable]
        public decimal DosePerAngstromFrame { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsMovieCTF)obj);
        }

        protected bool Equals(ProcessingOptionsMovieCTF other)
        {
            return base.Equals(other) &&
                   Window == other.Window &&
                   RangeMin == other.RangeMin &&
                   RangeMax == other.RangeMax &&
                   Voltage == other.Voltage &&
                   Cs == other.Cs &&
                   Cc == other.Cc &&
                   Amplitude == other.Amplitude &&
                   DoPhase == other.DoPhase &&
                   UseMovieSum == other.UseMovieSum &&
                   ZMin == other.ZMin &&
                   ZMax == other.ZMax &&
                   GridDims == other.GridDims &&
                   DosePerAngstromFrame == other.DosePerAngstromFrame;
        }

        public static bool operator ==(ProcessingOptionsMovieCTF left, ProcessingOptionsMovieCTF right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsMovieCTF left, ProcessingOptionsMovieCTF right)
        {
            return !Equals(left, right);
        }
    }

    [Serializable]
    public class ProcessingOptionsMovieMovement : ProcessingOptionsBase
    {
        [WarpSerializable]
        public decimal RangeMin { get; set; }
        [WarpSerializable]
        public decimal RangeMax { get; set; }
        [WarpSerializable]
        public decimal Bfactor { get; set; }
        [WarpSerializable]
        public int3 GridDims { get; set; }
        [WarpSerializable]
        public decimal DosePerAngstromFrame { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsMovieMovement)obj);
        }

        protected bool Equals(ProcessingOptionsMovieMovement other)
        {
            return base.Equals(other) &&
                   RangeMin == other.RangeMin &&
                   RangeMax == other.RangeMax &&
                   Bfactor == other.Bfactor &&
                   GridDims == other.GridDims &&
                   DosePerAngstromFrame == other.DosePerAngstromFrame;
        }

        public static bool operator ==(ProcessingOptionsMovieMovement left, ProcessingOptionsMovieMovement right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsMovieMovement left, ProcessingOptionsMovieMovement right)
        {
            return !Equals(left, right);
        }
    }

    [Serializable]
    public class ProcessingOptionsMovieExport : ProcessingOptionsBase
    {
        [WarpSerializable]
        public bool DoAverage { get; set; }
        [WarpSerializable]
        public bool DoStack { get; set; }
        [WarpSerializable]
        public bool DoDeconv { get; set; }
        [WarpSerializable]
        public bool DoDenoise { get; set; }
        [WarpSerializable]
        public bool DoDenoiseDeconv { get; set; }
        [WarpSerializable]
        public decimal DeconvolutionStrength { get; set; }
        [WarpSerializable]
        public decimal DeconvolutionFalloff { get; set; }
        [WarpSerializable]
        public int StackGroupSize { get; set; }
        [WarpSerializable]
        public int SkipFirstN { get; set; }
        [WarpSerializable]
        public int SkipLastN { get; set; }
        [WarpSerializable]
        public decimal DosePerAngstromFrame { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsMovieExport)obj);
        }

        protected bool Equals(ProcessingOptionsMovieExport other)
        {
            return base.Equals(other) &&
                   DoAverage == other.DoAverage &&
                   DoStack == other.DoStack &&
                   DoDeconv == other.DoDeconv &&
                   DoDenoise == other.DoDenoise &&
                   DeconvolutionStrength == other.DeconvolutionStrength &&
                   DeconvolutionFalloff == other.DeconvolutionFalloff &&
                   StackGroupSize == other.StackGroupSize &&
                   SkipFirstN == other.SkipFirstN &&
                   SkipLastN == other.SkipLastN &&
                   DosePerAngstromFrame == other.DosePerAngstromFrame;
        }

        public static bool operator ==(ProcessingOptionsMovieExport left, ProcessingOptionsMovieExport right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsMovieExport left, ProcessingOptionsMovieExport right)
        {
            return !Equals(left, right);
        }
    }

    [Serializable]
    public class ProcessingOptionsParticleExport : ProcessingOptionsBase
    {
        [WarpSerializable]
        public string Suffix { get; set; }
        [WarpSerializable]
        public int BoxSize { get; set; }
        [WarpSerializable]
        public int BoxSizeResample { get; set; }
        [WarpSerializable]
        public int Diameter { get; set; }
        [WarpSerializable]
        public bool Invert { get; set; }
        [WarpSerializable]
        public bool Normalize { get; set; }
        [WarpSerializable]
        public bool DoAverage { get; set; }
        [WarpSerializable]
        public bool DoDenoisingPairs { get; set; }
        [WarpSerializable]
        public int StackGroupSize { get; set; }
        [WarpSerializable]
        public int SkipFirstN { get; set; }
        [WarpSerializable]
        public int SkipLastN { get; set; }
        [WarpSerializable]
        public decimal DosePerAngstromFrame { get; set; }
        [WarpSerializable]
        public int Voltage { get; set; }
        [WarpSerializable]
        public bool CorrectAnisotropy { get; set; }
        [WarpSerializable]
        public bool PreflipPhases { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsParticleExport)obj);
        }

        protected bool Equals(ProcessingOptionsParticleExport other)
        {
            return base.Equals(other) &&
                   Suffix == other.Suffix &&
                   BoxSize == other.BoxSize &&
                   BoxSizeResample == other.BoxSizeResample &&
                   Diameter == other.Diameter &&
                   Invert == other.Invert &&
                   Normalize == other.Normalize &&
                   DoAverage == other.DoAverage &&
                   DoDenoisingPairs == other.DoDenoisingPairs &&
                   StackGroupSize == other.StackGroupSize &&
                   SkipFirstN == other.SkipFirstN &&
                   SkipLastN == other.SkipLastN &&
                   DosePerAngstromFrame == other.DosePerAngstromFrame &&
                   Voltage == other.Voltage &&
                   CorrectAnisotropy == other.CorrectAnisotropy &&
                   PreflipPhases == other.PreflipPhases;
        }

        public static bool operator ==(ProcessingOptionsParticleExport left, ProcessingOptionsParticleExport right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsParticleExport left, ProcessingOptionsParticleExport right)
        {
            return !Equals(left, right);
        }
    }

    [Serializable]
    public class ProcessingOptionsFullMatch : ProcessingOptionsBase
    {
        [WarpSerializable]
        public bool OverwriteFiles { get; set; }

        [WarpSerializable]
        public bool Invert { get; set; }

        [WarpSerializable]
        public int SubPatchSize { get; set; }

        [WarpSerializable]
        public string TemplateName { get; set; }

        [WarpSerializable]
        public decimal TemplatePixel { get; set; }

        [WarpSerializable]
        public decimal TemplateDiameter { get; set; }

        [WarpSerializable]
        public decimal TemplateFraction { get; set; }

        [WarpSerializable]
        public bool WhitenSpectrum { get; set; }

        [WarpSerializable]
        public decimal DosePerAngstromFrame { get; set; }

        [WarpSerializable]
        public int Voltage { get; set; }

        [WarpSerializable]
        public string Symmetry { get; set; }

        [WarpSerializable]
        public int HealpixOrder { get; set; }

        [WarpSerializable]
        public int Supersample { get; set; }

        [WarpSerializable]
        public int NResults { get; set; }
    }

    [Serializable]
    public class ProcessingOptionsBoxNet : ProcessingOptionsBase
    {
        [WarpSerializable]
        public string ModelName { get; set; }

        [WarpSerializable]
        public bool OverwriteFiles { get; set; }

        [WarpSerializable]
        public bool PickingInvert { get; set; }

        [WarpSerializable]
        public decimal ExpectedDiameter { get; set; }

        [WarpSerializable]
        public decimal MinimumScore { get; set; }

        [WarpSerializable]
        public decimal MinimumMaskDistance { get; set; }

        [WarpSerializable]
        public bool ExportParticles { get; set; }

        [WarpSerializable]
        public int ExportBoxSize { get; set; }

        [WarpSerializable]
        public bool ExportInvert { get; set; }

        [WarpSerializable]
        public bool ExportNormalize { get; set; }

        [WarpSerializable]
        public string OverrideImagePath { get; set; }

        [WarpSerializable]
        public string OverrideStarSuffix { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsBoxNet)obj);
        }

        protected bool Equals(ProcessingOptionsBoxNet other)
        {
            return ModelName == other.ModelName &&
                   OverwriteFiles == other.OverwriteFiles &&
                   PickingInvert == other.PickingInvert &&
                   ExpectedDiameter == other.ExpectedDiameter &&
                   MinimumScore == other.MinimumScore &&
                   ExportParticles == other.ExportParticles &&
                   ExportBoxSize == other.ExportBoxSize &&
                   ExportInvert == other.ExportInvert &&
                   ExportNormalize == other.ExportNormalize &&
                   OverrideImagePath == other.OverrideImagePath;
        }

        public static bool operator ==(ProcessingOptionsBoxNet left, ProcessingOptionsBoxNet right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsBoxNet left, ProcessingOptionsBoxNet right)
        {
            return !Equals(left, right);
        }
    }

    [Serializable]
    public class ProcessingOptionsMPARefine : WarpBase
    {
        private int _NIterations = 3;
        [WarpSerializable]
        public int NIterations
        {
            get { return _NIterations; }
            set { if (value != _NIterations) { _NIterations = value; OnPropertyChanged(); } }
        }

        private decimal _BFactorWeightingThreshold = 0.25M;
        [WarpSerializable]
        public decimal BFactorWeightingThreshold
        {
            get { return _BFactorWeightingThreshold; }
            set { if (value != _BFactorWeightingThreshold) { _BFactorWeightingThreshold = value; OnPropertyChanged(); } }
        }

        private int _BatchSize = 16;
        [WarpSerializable]
        public int BatchSize
        {
            get { return _BatchSize; }
            set { if (value != _BatchSize) { _BatchSize = value; OnPropertyChanged(); } }
        }

        private int _InitialResolutionPercent = 80;
        [WarpSerializable]
        public int InitialResolutionPercent
        {
            get { return _InitialResolutionPercent; }
            set { if (value != _InitialResolutionPercent) { _InitialResolutionPercent = value; OnPropertyChanged(); } }
        }

        private int _MinParticlesPerItem = 10;
        [WarpSerializable]
        public int MinParticlesPerItem
        {
            get { return _MinParticlesPerItem; }
            set { if (value != _MinParticlesPerItem) { _MinParticlesPerItem = value; OnPropertyChanged(); } }
        }

        private bool _UseHostMemory = true;
        [WarpSerializable]
        public bool UseHostMemory
        {
            get { return _UseHostMemory; }
            set { if (value != _UseHostMemory) { _UseHostMemory = value; OnPropertyChanged(); } }
        }

        #region Geometry

        private bool _DoImageWarp = true;
        [WarpSerializable]
        public bool DoImageWarp
        {
            get { return _DoImageWarp; }
            set { if (value != _DoImageWarp) { _DoImageWarp = value; OnPropertyChanged(); } }
        }

        private int _ImageWarpWidth = 3;
        [WarpSerializable]
        public int ImageWarpWidth
        {
            get { return _ImageWarpWidth; }
            set { if (value != _ImageWarpWidth) { _ImageWarpWidth = value; OnPropertyChanged(); } }
        }

        private int _ImageWarpHeight = 3;
        [WarpSerializable]
        public int ImageWarpHeight
        {
            get { return _ImageWarpHeight; }
            set { if (value != _ImageWarpHeight) { _ImageWarpHeight = value; OnPropertyChanged(); } }
        }

        private bool _DoVolumeWarp = false;
        [WarpSerializable]
        public bool DoVolumeWarp
        {
            get { return _DoVolumeWarp; }
            set { if (value != _DoVolumeWarp) { _DoVolumeWarp = value; OnPropertyChanged(); } }
        }

        private int _VolumeWarpWidth = 3;
        [WarpSerializable]
        public int VolumeWarpWidth
        {
            get { return _VolumeWarpWidth; }
            set { if (value != _VolumeWarpWidth) { _VolumeWarpWidth = value; OnPropertyChanged(); } }
        }

        private int _VolumeWarpHeight = 3;
        [WarpSerializable]
        public int VolumeWarpHeight
        {
            get { return _VolumeWarpHeight; }
            set { if (value != _VolumeWarpHeight) { _VolumeWarpHeight = value; OnPropertyChanged(); } }
        }

        private int _VolumeWarpDepth = 2;
        [WarpSerializable]
        public int VolumeWarpDepth
        {
            get { return _VolumeWarpDepth; }
            set { if (value != _VolumeWarpDepth) { _VolumeWarpDepth = value; OnPropertyChanged(); } }
        }

        private int _VolumeWarpLength = 10;
        [WarpSerializable]
        public int VolumeWarpLength
        {
            get { return _VolumeWarpLength; }
            set { if (value != _VolumeWarpLength) { _VolumeWarpLength = value; OnPropertyChanged(); } }
        }

        private bool _DoAxisAngles = false;
        [WarpSerializable]
        public bool DoAxisAngles
        {
            get { return _DoAxisAngles; }
            set { if (value != _DoAxisAngles) { _DoAxisAngles = value; OnPropertyChanged(); } }
        }

        private bool _DoParticlePoses = true;
        [WarpSerializable]
        public bool DoParticlePoses
        {
            get { return _DoParticlePoses; }
            set { if (value != _DoParticlePoses) { _DoParticlePoses = value; OnPropertyChanged(); } }
        }

        private bool _DoMagnification = false;
        [WarpSerializable]
        public bool DoMagnification
        {
            get { return _DoMagnification; }
            set { if (value != _DoMagnification) { _DoMagnification = value; OnPropertyChanged(); } }
        }

        private bool _DoZernike13 = false;
        [WarpSerializable]
        public bool DoZernike13
        {
            get { return _DoZernike13; }
            set { if (value != _DoZernike13) { _DoZernike13 = value; OnPropertyChanged(); } }
        }

        private bool _DoZernike5 = false;
        [WarpSerializable]
        public bool DoZernike5
        {
            get { return _DoZernike5; }
            set { if (value != _DoZernike5) { _DoZernike5 = value; OnPropertyChanged(); } }
        }

        private decimal _GeometryHighPass = 20;
        [WarpSerializable]
        public decimal GeometryHighPass
        {
            get { return _GeometryHighPass; }
            set { if (value != _GeometryHighPass) { _GeometryHighPass = value; OnPropertyChanged(); } }
        }

        private bool _DoTiltMovies = false;
        [WarpSerializable]
        public bool DoTiltMovies
        {
            get { return _DoTiltMovies; }
            set { if (value != _DoTiltMovies) { _DoTiltMovies = value; OnPropertyChanged(); } }
        }

        #endregion

        #region CTF

        private bool _DoDefocus = false;
        [WarpSerializable]
        public bool DoDefocus
        {
            get { return _DoDefocus; }
            set { if (value != _DoDefocus) { _DoDefocus = value; OnPropertyChanged(); } }
        }

        private bool _DoAstigmatismDelta = false;
        [WarpSerializable]
        public bool DoAstigmatismDelta
        {
            get { return _DoAstigmatismDelta; }
            set { if (value != _DoAstigmatismDelta) { _DoAstigmatismDelta = value; OnPropertyChanged(); } }
        }

        private bool _DoAstigmatismAngle = false;
        [WarpSerializable]
        public bool DoAstigmatismAngle
        {
            get { return _DoAstigmatismAngle; }
            set { if (value != _DoAstigmatismAngle) { _DoAstigmatismAngle = value; OnPropertyChanged(); } }
        }

        private bool _DoPhaseShift = false;
        [WarpSerializable]
        public bool DoPhaseShift
        {
            get { return _DoPhaseShift; }
            set { if (value != _DoPhaseShift) { _DoPhaseShift = value; OnPropertyChanged(); } }
        }

        private bool _DoCs = false;
        [WarpSerializable]
        public bool DoCs
        {
            get { return _DoCs; }
            set { if (value != _DoCs) { _DoCs = value; OnPropertyChanged(); } }
        }

        private bool _DoDoming = false;
        [WarpSerializable]
        public bool DoDoming
        {
            get { return _DoDoming; }
            set { if (value != _DoDoming) { _DoDoming = value; OnPropertyChanged(); } }
        }

        private bool _DoZernike2 = false;
        [WarpSerializable]
        public bool DoZernike2
        {
            get { return _DoZernike2; }
            set { if (value != _DoZernike2) { _DoZernike2 = value; OnPropertyChanged(); } }
        }

        private bool _DoZernike4 = false;
        [WarpSerializable]
        public bool DoZernike4
        {
            get { return _DoZernike4; }
            set { if (value != _DoZernike4) { _DoZernike4 = value; OnPropertyChanged(); } }
        }

        private bool _DoDefocusGridSearch = false;
        [WarpSerializable]
        public bool DoDefocusGridSearch
        {
            get { return _DoDefocusGridSearch; }
            set { if (value != _DoDefocusGridSearch) { _DoDefocusGridSearch = value; OnPropertyChanged(); } }
        }

        private decimal _MinimumCTFRefinementResolution = 7;
        [WarpSerializable]
        public decimal MinimumCTFRefinementResolution
        {
            get { return _MinimumCTFRefinementResolution; }
            set { if (value != _MinimumCTFRefinementResolution) { _MinimumCTFRefinementResolution = value; OnPropertyChanged(); } }
        }

        private decimal _CTFHighPass = 20;
        [WarpSerializable]
        public decimal CTFHighPass
        {
            get { return _CTFHighPass; }
            set { if (value != _CTFHighPass) { _CTFHighPass = value; OnPropertyChanged(); } }
        }

        #endregion
    }
}
