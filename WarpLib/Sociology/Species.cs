using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using TorchSharp;
using Warp.Tools;

namespace Warp.Sociology
{
    public class Species : WarpBase
    {
        private Guid _GUID = Guid.NewGuid();
        [WarpSerializable]
        public Guid GUID
        {
            get { return _GUID; }
            set
            {
                if (value != _GUID)
                {
                    _GUID = value;
                    OnPropertyChanged();
                }
            }
        }

        private string _Version = "";
        [WarpSerializable]
        public string Version
        {
            get { return _Version; }
            set
            {
                if (value != _Version)
                {
                    _Version = value;
                    OnPropertyChanged();
                }
            }
        }

        private string _PreviousVersion = "";
        [WarpSerializable]
        public string PreviousVersion
        {
            get { return _PreviousVersion; }
            set
            {
                if (value != _PreviousVersion)
                {
                    _PreviousVersion = value;
                    OnPropertyChanged();
                }
            }
        }

        private string _Name = "New Species";
        [WarpSerializable]
        public string Name
        {
            get { return _Name; }
            set
            {
                if (value != _Name)
                {
                    _Name = value;
                    OnPropertyChanged();
                }
            }
        }

        public string NameSafe => Helper.RemoveInvalidChars(Name);

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

        public string FolderPath => Helper.PathToFolder(Path);

        public bool IsRemote
        {
            get
            {
                try
                {
                    return !new Uri(Path).IsFile;
                }
                catch
                {
                    return false;
                }
            }
        }

        private Species _Parent = null;
        public Species Parent
        {
            get { return _Parent; }
            set
            {
                if (value != _Parent)
                {
                    _Parent = value;
                    OnPropertyChanged();
                }
            }
        }

        private ObservableCollection<Species> _Children = new ObservableCollection<Species>();
        public ObservableCollection<Species> Children
        {
            get { return _Children; }
            set
            {
                if (value != _Children)
                {
                    _Children = value;
                    OnPropertyChanged();
                }
            }
        }

        public Species[] AllDescendants => Children.Count == 0 ? new[] { this } : Helper.Combine(Children.Select(c => c.AllDescendants));

        private List<Guid> ChildrenGUIDs = new List<Guid>();

        public Dictionary<Guid, string> UsedDataSources = new Dictionary<Guid, string>();

        #region Basic parameters

        private decimal _PixelSize = 1M;
        [WarpSerializable]
        public decimal PixelSize
        {
            get { return _PixelSize; }
            set { if (value != _PixelSize) { _PixelSize = value; OnPropertyChanged(); } }
        }

        private decimal _MolecularWeightkDa = 100;
        [WarpSerializable]
        public decimal MolecularWeightkDa
        {
            get { return _MolecularWeightkDa; }
            set { if (value != _MolecularWeightkDa) { _MolecularWeightkDa = value; OnPropertyChanged(); } }
        }

        private int _DiameterAngstrom = 50;
        [WarpSerializable]
        public int DiameterAngstrom
        {
            get { return _DiameterAngstrom; }
            set { if (value != _DiameterAngstrom) { _DiameterAngstrom = value; OnPropertyChanged(); } }
        }

        public int Size => (int)Math.Round(DiameterAngstrom / PixelSize) * 2;

        private string _Symmetry = "C1";
        [WarpSerializable]
        public string Symmetry
        {
            get { return _Symmetry; }
            set { if (value != _Symmetry) { _Symmetry = value; OnPropertyChanged(); } }
        }

        private int _HelicalUnits = 1;
        [WarpSerializable]
        public int HelicalUnits
        {
            get { return _HelicalUnits; }
            set { if (value != _HelicalUnits) { _HelicalUnits = value; OnPropertyChanged(); } }
        }

        private decimal _HelicalTwist = 0;
        [WarpSerializable]
        public decimal HelicalTwist
        {
            get { return _HelicalTwist; }
            set { if (value != _HelicalTwist) { _HelicalTwist = value; OnPropertyChanged(); } }
        }

        private decimal _HelicalRise = 0;
        [WarpSerializable]
        public decimal HelicalRise
        {
            get { return _HelicalRise; }
            set { if (value != _HelicalRise) { _HelicalRise = value; OnPropertyChanged(); } }
        }

        private int _HelicalHeight = 0;
        [WarpSerializable]
        public int HelicalHeight
        {
            get { return _HelicalHeight; }
            set { if (value != _HelicalHeight) { _HelicalHeight = value; OnPropertyChanged(); } }
        }

        private bool _DoEwald = false;
        [WarpSerializable]
        public bool DoEwald
        {
            get { return _DoEwald; }
            set { if (value != _DoEwald) { _DoEwald = value; OnPropertyChanged(); } }
        }

        private bool _EwaldReverse = true;
        [WarpSerializable]
        public bool EwaldReverse
        {
            get { return _EwaldReverse; }
            set { if (value != _EwaldReverse) { _EwaldReverse = value; OnPropertyChanged(); } }
        }

        private bool _UseForAlignment = true;
        [WarpSerializable]
        public bool UseForAlignment
        {
            get { return _UseForAlignment; }
            set { if (value != _UseForAlignment) { _UseForAlignment = value; OnPropertyChanged(); } }
        }

        private int _TemporalResolutionMovement = 1;
        [WarpSerializable]
        public int TemporalResolutionMovement
        {
            get { return _TemporalResolutionMovement; }
            set { if (value != _TemporalResolutionMovement) { _TemporalResolutionMovement = value; OnPropertyChanged(); } }
        }

        private int _TemporalResolutionRotation = 1;
        [WarpSerializable]
        public int TemporalResolutionRotation
        {
            get { return _TemporalResolutionRotation; }
            set { if (value != _TemporalResolutionRotation) { _TemporalResolutionRotation = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Particles

        public Particle[] Particles = new Particle[0];
        public int NParticles => Particles.Length;

        public Particle[] DescendantParticles => Children.Count == 0 ? Particles.ToArray() : Helper.Combine(Children.Select(c => c.DescendantParticles));

        private const string SuffixParticleFile = "_particles.star";
        public string NameParticleFile => NameSafe + SuffixParticleFile;
        public string PathParticleFile => Helper.PathCombine(FolderPath, NameParticleFile);

        private const string SuffixAngularDist = "_angdist.mrc";
        public string NameAngularDist => NameSafe + SuffixAngularDist;
        public string PathAngularDist => Helper.PathCombine(FolderPath, NameAngularDist);

        private Image _AngularDistribution = null;
        public Image AngularDistribution
        {
            get
            {
                if (_AngularDistribution == null)
                {
                    if (!File.Exists(PathAngularDist))
                        return null;
                    _AngularDistribution = Image.FromFile(PathAngularDist);
                }
                return _AngularDistribution;
            }
            set { if (value != _AngularDistribution) { _AngularDistribution = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Resolution

        private decimal _GlobalResolution = -1M;
        [WarpSerializable]
        public decimal GlobalResolution
        {
            get { return _GlobalResolution; }
            set { if (value != _GlobalResolution) { _GlobalResolution = value; OnPropertyChanged(); } }
        }

        private int _GlobalBFactor = 0;
        [WarpSerializable]
        public int GlobalBFactor
        {
            get { return _GlobalBFactor; }
            set { if (value != _GlobalBFactor) { _GlobalBFactor = value; OnPropertyChanged(); } }
        }

        private const string SuffixGlobalFSC = "_fsc.star";
        public string NameGlobalFSC => NameSafe + SuffixGlobalFSC;
        public string PathGlobalFSC => Helper.PathCombine(FolderPath, NameGlobalFSC);

        private float5[] _GlobalFSC = null;
        public float5[] GlobalFSC
        {
            get
            {
                if (_GlobalFSC == null)
                {
                    if (!File.Exists(PathGlobalFSC))
                        return null;
                    _GlobalFSC = Star.LoadFloat5(PathGlobalFSC);
                }
                return _GlobalFSC;
            }
            set { if (value != _GlobalFSC) { _GlobalFSC = value; OnPropertyChanged(); } }
        }
        public int SizeAtFSC05
        {
            get
            {
                int Shell = 0;
                while (++Shell < GlobalFSC.Length)
                {
                    if (GlobalFSC[Shell].W < 0.5f)
                    {
                        Shell--;
                        break;
                    }
                }

                return (Shell + 1) * 2;
            }
        }
        public float ResolutionAtFSC05
        {
            get
            {
                int Shell = 0;
                while (++Shell < GlobalFSC.Length)
                {
                    if (GlobalFSC[Shell].W < 0.5f)
                    {
                        Shell--;
                        break;
                    }
                }

                return GlobalFSC[Shell].X;
            }
        }

        private const string SuffixLocalResolution = "_localres.mrc";
        public string NameLocalResolution => NameSafe + SuffixLocalResolution;
        public string PathLocalResolution => Helper.PathCombine(FolderPath, NameLocalResolution);

        private Image _LocalResolution = null;
        public Image LocalResolution
        {
            get
            {
                if (_LocalResolution == null)
                {
                    if (!File.Exists(PathLocalResolution))
                        return null;
                    _LocalResolution = Image.FromFile(PathLocalResolution);
                }
                return _LocalResolution;
            }
            set { if (value != _LocalResolution) { _LocalResolution = value; OnPropertyChanged(); } }
        }
        public Image LocalResolutionAsync
        {
            get
            {
                if (_LocalResolution == null)
                {
                    if (!File.Exists(PathLocalResolution))
                        return null;
                    Task.Run(() =>
                    {
                        _LocalResolution = Image.FromFile(PathLocalResolution);
                        OnPropertyChanged(nameof(LocalResolution));
                    });
                    return null;
                }
                return _LocalResolution;
            }
        }

        private const string SuffixResolutionHistogram = "_localres.star";
        public string NameResolutionHistogram => NameSafe + SuffixResolutionHistogram;
        public string PathResolutionHistogram => Helper.PathCombine(FolderPath, NameResolutionHistogram);

        private float2[] _ResolutionHistogram = null;
        public float2[] ResolutionHistogram
        {
            get
            {
                if (_ResolutionHistogram == null)
                {
                    if (!File.Exists(PathResolutionHistogram))
                        return null;
                    _ResolutionHistogram = Star.LoadFloat2(PathResolutionHistogram);
                }
                return _ResolutionHistogram;
            }
            set { if (value != _ResolutionHistogram) { _ResolutionHistogram = value; OnPropertyChanged(); } }
        }

        private const string SuffixLocalBFactor = "_localbfac.mrc";
        public string NameLocalBFactor => NameSafe + SuffixLocalBFactor;
        public string PathLocalBFactor => Helper.PathCombine(FolderPath, NameLocalBFactor);

        private Image _LocalBFactor = null;
        public Image LocalBFactor
        {
            get
            {
                if (_LocalBFactor == null)
                {
                    if (!File.Exists(PathLocalBFactor))
                        return null;
                    _LocalBFactor = Image.FromFile(PathLocalBFactor);
                }

                return _LocalBFactor;
            }
            set { if (value != _LocalBFactor) { _LocalBFactor = value; OnPropertyChanged(); } }
        }
        public Image LocalBFactorAsync
        {
            get
            {
                if (_LocalBFactor == null)
                {
                    if (!File.Exists(PathLocalBFactor))
                        return null;
                    Task.Run(() =>
                    {
                        _LocalBFactor = Image.FromFile(PathLocalBFactor);
                        OnPropertyChanged(nameof(LocalBFactor));
                    });
                    return null;
                }
                return _LocalBFactor;
            }
        }

        private const string SuffixAnisoResolution = "_anisores.mrc";
        public string NameAnisoResolution => NameSafe + SuffixAnisoResolution;
        public string PathAnisoResolution => Helper.PathCombine(FolderPath, NameAnisoResolution);

        private Image _AnisoResolution = null;
        public Image AnisoResolution
        {
            get
            {
                if (_AnisoResolution == null)
                {
                    if (!File.Exists(PathAnisoResolution))
                        return null;
                    _AnisoResolution = Image.FromFile(PathAnisoResolution);
                }
                return _AnisoResolution;
            }
            set { if (value != _AnisoResolution) { _AnisoResolution = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Denoising

        private decimal _DenoisedAtResolution = -1M;
        [WarpSerializable]
        public decimal DenoisedAtResolution
        {
            get { return _DenoisedAtResolution; }
            set { if (value != _DenoisedAtResolution) { _DenoisedAtResolution = value; OnPropertyChanged(); } }
        }

        private bool _ApplyDenoising = true;
        [WarpSerializable]
        public bool ApplyDenoising
        {
            get { return _ApplyDenoising; }
            set { if (value != _ApplyDenoising) { _ApplyDenoising = value; OnPropertyChanged(); } }
        }

        private const string SuffixNoiseNet = "_noisenet.pt";
        public string NameNoiseNet => NameSafe + SuffixNoiseNet;
        public string PathNoiseNet => Helper.PathCombine(FolderPath, NameNoiseNet);

        private NoiseNet3DTorch Denoiser = null;

        #endregion

        #region Mask

        private const string SuffixMask = "_mask.mrc";
        public string NameMask => NameSafe + SuffixMask;
        public string PathMask => Helper.PathCombine(FolderPath, NameMask);

        private Image _Mask = null;
        public Image Mask
        {
            get
            {
                if (_Mask == null)
                {
                    if (!File.Exists(PathMask))
                        return null;
                    _Mask = Image.FromFile(PathMask);
                }
                return _Mask;
            }
            set { if (value != _Mask) { _Mask = value; OnPropertyChanged(); } }
        }
        public Image MaskAsync
        {
            get
            {
                if (_Mask == null)
                {
                    if (!File.Exists(PathMask))
                        return null;
                    Task.Run(() =>
                    {
                        _Mask = Image.FromFile(PathMask);
                        OnPropertyChanged(nameof(Mask));
                    });
                    return null;
                }
                return _Mask;
            }
        }

        private decimal _MaskAutoThreshold = 0.02M;
        [WarpSerializable]
        public decimal MaskAutoThreshold
        {
            get { return _MaskAutoThreshold; }
            set { if (value != _MaskAutoThreshold) { _MaskAutoThreshold = value; OnPropertyChanged(); } }
        }

        private decimal _MaskAutoResolution = 10;
        [WarpSerializable]
        public decimal MaskAutoResolution
        {
            get { return _MaskAutoResolution; }
            set { if (value != _MaskAutoResolution) { _MaskAutoResolution = value; OnPropertyChanged(); } }
        }

        private bool _MaskUseManual = true;
        [WarpSerializable]
        public bool MaskUseManual
        {
            get { return _MaskUseManual; }
            set { if (value != _MaskUseManual) { _MaskUseManual = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Maps

        private const string SuffixMapFiltered = "_filt.mrc";
        public string NameMapFiltered => NameSafe + SuffixMapFiltered;
        public string PathMapFiltered => Helper.PathCombine(FolderPath, NameMapFiltered);

        private Image _MapFiltered = null;
        public Image MapFiltered
        {
            get
            {
                if (_MapFiltered == null)
                {
                    if (!File.Exists(PathMapFiltered))
                        return null;
                    _MapFiltered = Image.FromFile(PathMapFiltered);
                }
                return _MapFiltered;
            }
            set { if (value != _MapFiltered) { _MapFiltered = value; OnPropertyChanged(); } }
        }
        public Image MapFilteredAsync
        {
            get
            {
                if (_MapFiltered == null)
                {
                    if (!File.Exists(PathMapFiltered))
                        return null;
                    Task.Run(() =>
                    {
                        _MapFiltered = Image.FromFile(PathMapFiltered);
                        OnPropertyChanged(nameof(MapFiltered));
                    });
                    return null;
                }
                return _MapFiltered;
            }
        }

        private const string SuffixMapFilteredSharpened = "_filtsharp.mrc";
        public string NameMapFilteredSharpened => NameSafe + SuffixMapFilteredSharpened;
        public string PathMapFilteredSharpened => Helper.PathCombine(FolderPath, NameMapFilteredSharpened);

        private Image _MapFilteredSharpened = null;
        public Image MapFilteredSharpened
        {
            get
            {
                if (_MapFilteredSharpened == null)
                {
                    if (!File.Exists(PathMapFilteredSharpened))
                        return null;
                    _MapFilteredSharpened = Image.FromFile(PathMapFilteredSharpened);
                }
                return _MapFilteredSharpened;
            }
            set { if (value != _MapFilteredSharpened) { _MapFilteredSharpened = value; OnPropertyChanged(); } }
        }
        public Image MapFilteredSharpenedAsync
        {
            get
            {
                if (_MapFilteredSharpened == null)
                {
                    if (!File.Exists(PathMapFilteredSharpened))
                        return null;
                    Task.Run(() =>
                    {
                        _MapFilteredSharpened = Image.FromFile(PathMapFilteredSharpened);
                        OnPropertyChanged(nameof(MapFilteredSharpened));
                    });
                    return null;
                }
                return _MapFilteredSharpened;
            }
        }

        private const string SuffixMapFilteredAnisotropic = "_filtanisotropic.mrc";
        public string NameMapFilteredAnisotropic => NameSafe + SuffixMapFilteredAnisotropic;
        public string PathMapFilteredAnisotropic => Helper.PathCombine(FolderPath, NameMapFilteredAnisotropic);

        private Image _MapFilteredAnisotropic = null;
        public Image MapFilteredAnisotropic
        {
            get
            {
                if (_MapFilteredAnisotropic == null)
                {
                    if (!File.Exists(PathMapFilteredAnisotropic))
                        return null;
                    _MapFilteredAnisotropic = Image.FromFile(PathMapFilteredAnisotropic);
                }
                return _MapFilteredAnisotropic;
            }
            set { if (value != _MapFilteredAnisotropic) { _MapFilteredAnisotropic = value; OnPropertyChanged(); } }
        }
        public Image MapFilteredAnisotropicAsync
        {
            get
            {
                if (_MapFilteredAnisotropic == null)
                {
                    if (!File.Exists(PathMapFilteredAnisotropic))
                        return null;
                    Task.Run(() =>
                    {
                        _MapFilteredAnisotropic = Image.FromFile(PathMapFilteredAnisotropic);
                        OnPropertyChanged(nameof(MapFilteredAnisotropic));
                    });
                    return null;
                }
                return _MapFilteredAnisotropic;
            }
        }

        private const string SuffixMapLocallyFiltered = "_filtlocal.mrc";
        public string NameMapLocallyFiltered => NameSafe + SuffixMapLocallyFiltered;
        public string PathMapLocallyFiltered => Helper.PathCombine(FolderPath, NameMapLocallyFiltered);

        private Image _MapLocallyFiltered = null;
        public Image MapLocallyFiltered
        {
            get
            {
                if (_MapLocallyFiltered == null)
                {
                    if (!File.Exists(PathMapLocallyFiltered))
                        return null;
                    _MapLocallyFiltered = Image.FromFile(PathMapLocallyFiltered);
                }
                return _MapLocallyFiltered;
            }
            set { if (value != _MapLocallyFiltered) { _MapLocallyFiltered = value; OnPropertyChanged(); } }
        }
        public Image MapLocallyFilteredAsync
        {
            get
            {
                if (_MapLocallyFiltered == null)
                {
                    if (!File.Exists(PathMapLocallyFiltered))
                        return null;
                    Task.Run(() =>
                    {
                        _MapLocallyFiltered = Image.FromFile(PathMapLocallyFiltered);
                        OnPropertyChanged(nameof(MapLocallyFiltered));
                    });
                    return null;
                }
                return _MapLocallyFiltered;
            }
        }

        private const string SuffixMapDenoised = "_denoised.mrc";
        public string NameMapDenoised => NameSafe + SuffixMapDenoised;
        public string PathMapDenoised => Helper.PathCombine(FolderPath, NameMapDenoised);

        private Image _MapDenoised = null;
        public Image MapDenoised
        {
            get
            {
                if (_MapDenoised == null)
                {
                    if (!File.Exists(PathMapDenoised))
                        return null;
                    _MapDenoised = Image.FromFile(PathMapDenoised);
                }
                return _MapDenoised;
            }
            set { if (value != _MapDenoised) { _MapDenoised = value; OnPropertyChanged(); } }
        }
        public Image MapDenoisedAsync
        {
            get
            {
                if (_MapDenoised == null)
                {
                    if (!File.Exists(PathMapDenoised))
                        return null;
                    Task.Run(() =>
                    {
                        _MapDenoised = Image.FromFile(PathMapDenoised);
                        OnPropertyChanged(nameof(MapDenoised));
                    });
                    return null;
                }
                return _MapDenoised;
            }
        }

        private const string SuffixHalfMap1 = "_half1.mrc";
        public string NameHalfMap1 => NameSafe + SuffixHalfMap1;
        public string PathHalfMap1 => Helper.PathCombine(FolderPath, NameHalfMap1);

        private Image _HalfMap1 = null;
        public Image HalfMap1
        {
            get
            {
                if (_HalfMap1 == null)
                {
                    if (!File.Exists(PathHalfMap1))
                        return null;
                    _HalfMap1 = Image.FromFile(PathHalfMap1);
                }
                return _HalfMap1;
            }
            set { if (value != _HalfMap1) { _HalfMap1 = value; OnPropertyChanged(); } }
        }
        public Image HalfMap1Async
        {
            get
            {
                if (_HalfMap1 == null)
                {
                    if (!File.Exists(PathHalfMap1))
                        return null;
                    Task.Run(() =>
                    {
                        _HalfMap1 = Image.FromFile(PathHalfMap1);
                        OnPropertyChanged(nameof(HalfMap1));
                    });
                    return null;
                }
                return _HalfMap1;
            }
        }

        private const string SuffixHalfMap2 = "_half2.mrc";
        private string NameHalfMap2 => NameSafe + SuffixHalfMap2;
        public string PathHalfMap2 => Helper.PathCombine(FolderPath, NameHalfMap2);

        private Image _HalfMap2 = null;
        public Image HalfMap2
        {
            get
            {
                if (_HalfMap2 == null)
                {
                    if (!File.Exists(PathHalfMap2))
                        return null;
                    _HalfMap2 = Image.FromFile(PathHalfMap2);
                }
                return _HalfMap2;
            }
            set { if (value != _HalfMap2) { _HalfMap2 = value; OnPropertyChanged(); } }
        }
        public Image HalfMap2Async
        {
            get
            {
                if (_HalfMap2 == null)
                {
                    if (!File.Exists(PathHalfMap2))
                        return null;
                    Task.Run(() =>
                    {
                        _HalfMap2 = Image.FromFile(PathHalfMap2);
                        OnPropertyChanged(nameof(HalfMap2));
                    });
                    return null;
                }
                return _HalfMap2;
            }
        }

        #endregion

        #region Refinement runtime

        #region Projectors

        private float _ResolutionRefinement = 1f;
        public float ResolutionRefinement
        {
            get { return _ResolutionRefinement; }
            set { if (value != _ResolutionRefinement) { _ResolutionRefinement = value; OnPropertyChanged(); } }
        }

        private int _CurrentMaxShellRefinement = 1;
        public int CurrentMaxShellRefinement
        {
            get { return _CurrentMaxShellRefinement; }
            set { if (value != _CurrentMaxShellRefinement) { _CurrentMaxShellRefinement = value; OnPropertyChanged(); } }
        }

        private Projector[] _HalfMap1Projector = null;
        public Projector[] HalfMap1Projector
        {
            get { return _HalfMap1Projector; }
            set { if (value != _HalfMap1Projector) { _HalfMap1Projector = value; OnPropertyChanged(); } }
        }

        private Projector[] _HalfMap2Projector = null;
        public Projector[] HalfMap2Projector
        {
            get { return _HalfMap2Projector; }
            set { if (value != _HalfMap2Projector) { _HalfMap2Projector = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Reconstructions

        private float _ResolutionReconstruction = 1f;
        public float ResolutionReconstruction
        {
            get { return _ResolutionReconstruction; }
            set { if (value != _ResolutionReconstruction) { _ResolutionReconstruction = value; OnPropertyChanged(); } }
        }

        private Projector[] _HalfMap1Reconstruction = null;
        public Projector[] HalfMap1Reconstruction
        {
            get { return _HalfMap1Reconstruction; }
            set { if (value != _HalfMap1Reconstruction) { _HalfMap1Reconstruction = value; OnPropertyChanged(); } }
        }

        private Projector[] _HalfMap2Reconstruction = null;
        public Projector[] HalfMap2Reconstruction
        {
            get { return _HalfMap2Reconstruction; }
            set { if (value != _HalfMap2Reconstruction) { _HalfMap2Reconstruction = value; OnPropertyChanged(); } }
        }

        #endregion

        #endregion

        private Species()
        {
            Children.CollectionChanged += Children_CollectionChanged;
        }

        private void Children_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            if (e.NewItems != null)
                foreach (var item in e.NewItems)
                    ((Species)item).Parent = this;

            if (e.OldItems != null)
                foreach (var item in e.OldItems)
                    ((Species)item).Parent = null;
        }

        public Species(Image halfMap1, Image halfMap2, Image mask) : this()
        {
            HalfMap1 = halfMap1.GetCopy();
            HalfMap2 = halfMap2.GetCopy();

            Mask = mask.GetCopy();
        }

        #region Particle operations

        public void AddParticles(Particle[] newParticles)
        {
            if (Particles != null)
                Particles = Helper.Combine(new[] { Particles, newParticles });
            else
                Particles = newParticles;
        }

        public void ReplaceParticles(Particle[] newParticles)
        {
            Particles = newParticles;
        }

        public Star ParticlesToStar()
        {
            if (Particles.Length == 0)
                return null;

            // Make sure each particle has desired temporal resolution
            //foreach (var particle in Particles)
            //{
            //    particle.ResampleCoordinates(TemporalResolutionMovement);
            //    particle.ResampleAngles(TemporalResolutionRotation);
            //}

            // Create columns
            List<string> ColumnNames = new List<string>();
            for (int i = 0; i < TemporalResolutionMovement; i++)
            {
                ColumnNames.Add($"wrpCoordinateX{i + 1}");
                ColumnNames.Add($"wrpCoordinateY{i + 1}");
                ColumnNames.Add($"wrpCoordinateZ{i + 1}");
            }
            for (int i = 0; i < TemporalResolutionRotation; i++)
            {
                ColumnNames.Add($"wrpAngleRot{i + 1}");
                ColumnNames.Add($"wrpAngleTilt{i + 1}");
                ColumnNames.Add($"wrpAnglePsi{i + 1}");
            }
            ColumnNames.Add("wrpRandomSubset");
            ColumnNames.Add("wrpSourceName");
            ColumnNames.Add("wrpSourceHash");

            Star TableOut = new Star(ColumnNames.ToArray());

            // Create rows
            foreach (var particle in Particles)
            {
                List<string> Row = new List<string>(ColumnNames.Count);

                for (int i = 0; i < TemporalResolutionMovement; i++)
                {
                    Row.Add(particle.Coordinates[i].X.ToString("F5", CultureInfo.InvariantCulture));
                    Row.Add(particle.Coordinates[i].Y.ToString("F5", CultureInfo.InvariantCulture));
                    Row.Add(particle.Coordinates[i].Z.ToString("F5", CultureInfo.InvariantCulture));
                }
                for (int i = 0; i < TemporalResolutionRotation; i++)
                {
                    Row.Add(particle.Angles[i].X.ToString("F5", CultureInfo.InvariantCulture));
                    Row.Add(particle.Angles[i].Y.ToString("F5", CultureInfo.InvariantCulture));
                    Row.Add(particle.Angles[i].Z.ToString("F5", CultureInfo.InvariantCulture));
                }
                Row.Add((particle.RandomSubset + 1).ToString(CultureInfo.InvariantCulture));
                Row.Add(particle.SourceName);
                Row.Add(particle.SourceHash);

                TableOut.AddRow(Row.ToArray());
            }

            return TableOut;
        }

        public Particle[] ParticlesFromStar(Star tableIn)
        {
            int MaxSamples = 0;
            while (tableIn.HasColumn($"wrpCoordinateX{MaxSamples + 1}"))
                MaxSamples++;

            string[] NamesCoordX = Helper.ArrayOfFunction(i => $"wrpCoordinateX{Math.Min(MaxSamples, i + 1)}", TemporalResolutionMovement);
            string[] NamesCoordY = Helper.ArrayOfFunction(i => $"wrpCoordinateY{Math.Min(MaxSamples, i + 1)}", TemporalResolutionMovement);
            string[] NamesCoordZ = Helper.ArrayOfFunction(i => $"wrpCoordinateZ{Math.Min(MaxSamples, i + 1)}", TemporalResolutionMovement);

            string[] NamesAngleRot = Helper.ArrayOfFunction(i => $"wrpAngleRot{Math.Min(MaxSamples, i + 1)}", TemporalResolutionRotation);
            string[] NamesAngleTilt = Helper.ArrayOfFunction(i => $"wrpAngleTilt{Math.Min(MaxSamples, i + 1)}", TemporalResolutionRotation);
            string[] NamesAnglePsi = Helper.ArrayOfFunction(i => $"wrpAnglePsi{Math.Min(MaxSamples, i + 1)}", TemporalResolutionRotation);

            float[][] ColumnsCoordX = NamesCoordX.Select(n => tableIn.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
            float[][] ColumnsCoordY = NamesCoordY.Select(n => tableIn.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
            float[][] ColumnsCoordZ = NamesCoordZ.Select(n => tableIn.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();

            float[][] ColumnsAngleRot = NamesAngleRot.Select(n => tableIn.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
            float[][] ColumnsAngleTilt = NamesAngleTilt.Select(n => tableIn.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
            float[][] ColumnsAnglePsi = NamesAnglePsi.Select(n => tableIn.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();

            int[] ColumnSubset = tableIn.GetColumn("wrpRandomSubset").Select(v => int.Parse(v) - 1).ToArray();

            string[] ColumnSourceName = tableIn.GetColumn("wrpSourceName");
            string[] ColumnSourceHash = tableIn.GetColumn("wrpSourceHash");

            Particle[] Result = new Particle[tableIn.RowCount];

            for (int p = 0; p < Result.Length; p++)
            {
                float3[] Coordinates = Helper.ArrayOfFunction(i => new float3(ColumnsCoordX[i][p],
                                                                              ColumnsCoordY[i][p],
                                                                              ColumnsCoordZ[i][p]), TemporalResolutionMovement);
                float3[] Angles = Helper.ArrayOfFunction(i => new float3(ColumnsAngleRot[i][p],
                                                                         ColumnsAngleTilt[i][p],
                                                                         ColumnsAnglePsi[i][p]), TemporalResolutionMovement);

                Result[p] = new Particle(Coordinates, Angles, ColumnSubset[p], ColumnSourceName[p], ColumnSourceHash[p]);
            }

            return Result;
        }

        public Star ParticlesToRelionStar(string micrographPrefix)
        {
            if (Particles.Length == 0)
                return null;

            Star TableOut = new Star(new[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnCoordinateZ",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnRandomSubset",
                "rlnMicrographName"
            });

            foreach (var particle in Particles)
            {
                float3 MeanCoordinates = particle.CoordinatesMean;
                float3 MeanAngles = particle.AnglesMean;
                string[] Row = new string[]
                {
                    MeanCoordinates.X.ToString("F5", CultureInfo.InvariantCulture),
                    MeanCoordinates.Y.ToString("F5", CultureInfo.InvariantCulture),
                    MeanCoordinates.Z.ToString("F5", CultureInfo.InvariantCulture),
                    MeanAngles.X.ToString("F5", CultureInfo.InvariantCulture),
                    MeanAngles.Y.ToString("F5", CultureInfo.InvariantCulture),
                    MeanAngles.Z.ToString("F5", CultureInfo.InvariantCulture),
                    (particle.RandomSubset + 1).ToString(CultureInfo.InvariantCulture),
                    micrographPrefix + particle.SourceName
                };

                TableOut.AddRow(Row);
            }

            return TableOut;
        }

        public Particle[] GetParticles(string sourceHash)
        {
            List<Particle> Result = new List<Particle>();

            foreach (var particle in Particles)
            {
                if (particle.SourceHash == sourceHash)
                    Result.Add(particle);
            }

            return Result.ToArray();
        }

        public static Particle[] ParticlesFromRelionStar(Star tableIn, float pixelSize, Dictionary<string, string> micrographHashes)
        {
            if (!tableIn.HasColumn("rlnCoordinateX") ||
                !tableIn.HasColumn("rlnCoordinateY") ||
                !tableIn.HasColumn("rlnAngleRot") ||
                !tableIn.HasColumn("rlnAngleTilt") ||
                !tableIn.HasColumn("rlnAnglePsi") ||
                !tableIn.HasColumn("rlnMicrographName"))
                throw new Exception("Particles need to have positions, angles, and source micrograph names.");

            int NParticles = tableIn.RowCount;
            bool IsTomogram = tableIn.HasColumn("rlnCoordinateZ");

            float[] CoordinatesX = tableIn.GetColumn("rlnCoordinateX").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * pixelSize).ToArray();
            float[] CoordinatesY = tableIn.GetColumn("rlnCoordinateY").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * pixelSize).ToArray();
            float[] CoordinatesZ = IsTomogram ? tableIn.GetColumn("rlnCoordinateZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * pixelSize).ToArray() : new float[NParticles];

            float[] OffsetsX = tableIn.HasColumn("rlnOffsetX") ? tableIn.GetColumn("rlnOffsetX").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * pixelSize).ToArray() : new float[NParticles];
            float[] OffsetsY = tableIn.HasColumn("rlnOffsetY") ? tableIn.GetColumn("rlnOffsetY").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * pixelSize).ToArray() : new float[NParticles];
            float[] OffsetsZ = tableIn.HasColumn("rlnOffsetZ") ? tableIn.GetColumn("rlnOffsetZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * pixelSize).ToArray() : new float[NParticles];

            float3[] Coordinates = Helper.ArrayOfFunction(p => new float3(CoordinatesX[p] - OffsetsX[p], CoordinatesY[p] - OffsetsY[p], CoordinatesZ[p] - OffsetsZ[p]), NParticles);

            float[] AnglesRot = tableIn.GetColumn("rlnAngleRot").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] AnglesTilt = tableIn.GetColumn("rlnAngleTilt").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] AnglesPsi = tableIn.GetColumn("rlnAnglePsi").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            float3[] Angles = Helper.ArrayOfFunction(p => new float3(AnglesRot[p], AnglesTilt[p], AnglesPsi[p]), NParticles);

            int[] Subsets = tableIn.HasColumn("rlnRandomSubset") ? tableIn.GetColumn("rlnRandomSubset").Select(v => int.Parse(v, CultureInfo.InvariantCulture) - 1).ToArray() : Helper.ArrayOfFunction(i => i % 2, NParticles);

            string[] MicrographNames = tableIn.GetColumn("rlnMicrographName").Select(v => v.Substring(v.LastIndexOf("/") + 1)).Select(v => v.Substring(0, v.LastIndexOf("."))).ToArray();
            string[] MicrographHashes = micrographHashes != null ? MicrographNames.Select(v => micrographHashes[v]).ToArray() : Helper.ArrayOfConstant("", NParticles);

            Particle[] Result = Helper.ArrayOfFunction(p => new Particle(new[] { Coordinates[p] }, new[] { Angles[p] }, Subsets[p], MicrographNames[p], MicrographHashes[p]), NParticles);
            return Result;
        }

        public static Dictionary<string, List<Particle>> GetParticlesGrouped(Particle[] particles)
        {
            Dictionary<string, List<Particle>> Groups = new Dictionary<string, List<Particle>>();

            foreach (var particle in particles)
            {
                string BestSource = !string.IsNullOrEmpty(particle.SourceHash) ? particle.SourceHash : particle.SourceName;
                if (!Groups.ContainsKey(BestSource))
                    Groups.Add(BestSource, new List<Particle>());
                Groups[BestSource].Add(particle);
            }

            return Groups;
        }

        public Dictionary<Particle, Particle> MatchParticlesWithinDistance(Particle[] particles, float matchDistance)
        {
            Dictionary<string, List<Particle>> OwnGroups = GetParticlesGrouped(Particles.ToArray());
            Dictionary<string, List<Particle>> OtherGroups = GetParticlesGrouped(particles);

            Dictionary<Particle, Particle> Mapping = new Dictionary<Particle, Particle>();

            foreach (var source in OtherGroups.Keys)
            {
                if (OwnGroups.ContainsKey(source))
                {
                    List<Particle> OwnParticles = OwnGroups[source];
                    List<Particle> OtherParticles = OtherGroups[source];

                    for (int i = 0; i < OtherParticles.Count; i++)
                    {
                        int BestID = -1;
                        float BestDistance = float.MaxValue;
                        for (int j = 0; j < OwnParticles.Count; j++)
                        {
                            float Distance = OtherParticles[i].DistanceSqFrom(OwnParticles[j]);
                            if (Distance < BestDistance)
                            {
                                BestDistance = Distance;
                                BestID = j;
                            }
                        }

                        if (BestID >= 0)
                        {
                            Mapping.Add(OtherParticles[i], OwnParticles[BestID]);
                            OwnParticles.RemoveAt(BestID);
                        }
                    }
                }
            }

            return Mapping;
        }

        public int UpdateParticleParameters(Particle[] particles, float matchDistance, bool updateCoordinates, bool updateAngles)
        {
            Dictionary<Particle, Particle> Mapping = MatchParticlesWithinDistance(particles, matchDistance);

            foreach (var pair in Mapping)
            {
                if (updateCoordinates)
                    pair.Value.Coordinates = pair.Key.Coordinates;
                if (updateAngles)
                    pair.Value.Angles = pair.Key.Angles;
            }

            return Mapping.Count;
        }

        public void ResampleParticleTemporalResolution(int movement, int angles)
        {
            Parallel.ForEach(Particles, particle =>
            {
                particle.ResampleCoordinates(movement);
                particle.ResampleAngles(angles);
            });

            TemporalResolutionMovement = movement;
            TemporalResolutionRotation = angles;
        }

        #endregion

        #region Processing

        public void CalculateResolutionAndFilter(float fixedResolution = -1, Action<string> progressCallback = null, int gpuID = -1)
        {
            if (gpuID < 0)
                gpuID = GPU.GetDeviceWithMostMemory();
            
            GPU.SetDevice(gpuID);

            //_HalfMap1 = Image.FromFile("d_half1.mrc");
            //Image Cropped1 = HalfMap1.AsPadded(new int3(400));
            //HalfMap1.Dispose();
            //_HalfMap1 = Cropped1;
            //_HalfMap2 = Image.FromFile("d_half2.mrc");
            //Image Cropped2 = HalfMap2.AsPadded(new int3(400));
            //HalfMap2.Dispose();
            //_HalfMap2 = Cropped2;

            progressCallback?.Invoke("1/5: Global FSC");

            #region If there is no resolution estimate yet, make one with a soft spherical mask

            if (GlobalResolution <= 0)
            {
                Image MaskSpherical = new Image(HalfMap1.Dims);
                MaskSpherical.Fill(1);
                MaskSpherical.MaskSpherically((float)DiameterAngstrom / (float)PixelSize, 16, true);

                float TempShellGlobal;
                float[] TempFSCUnmasked, TempFSCMasked, TempFSCRandomized, TempFSCCorrected;

                FSC.GetCorrectedFSC(HalfMap1,
                                    HalfMap2,
                                    MaskSpherical,
                                    0.143f,
                                    0.8f,
                                    out TempShellGlobal,
                                    out TempFSCUnmasked,
                                    out TempFSCMasked,
                                    out TempFSCRandomized,
                                    out TempFSCCorrected);

                GlobalResolution = (decimal)((float)PixelSize * HalfMap1.Dims.X / TempShellGlobal);

                MaskSpherical.Dispose();
            }

            #endregion

            if (Math.Abs(Mask.PixelSize - (float)PixelSize) > 0.02f)
            {
                int DimScaled = (int)Math.Round((Mask.PixelSize / (float)PixelSize) * Mask.Dims.X / 2) * 2;
                Image MaskScaled = Mask.AsScaled(new int3(DimScaled)).AsPadded(HalfMap1.Dims).AndDisposeParent();
                MaskScaled.Binarize(0.25f);

                MaskScaled.WriteMRC(PathMask, (float)PixelSize, true);
                MaskScaled.Dispose();
                _Mask.Dispose();
                _Mask = null;
            }

            float Softness = Math.Max(1, (float)GlobalResolution / (float)(PixelSize * 2));
            Image MaskSoft = FSC.MakeSoftMask(Mask, (int)(3 * Softness + 0.5f), (int)(3 * Softness + 0.5f));

            if (MaskSoft.Dims != HalfMap1.Dims)
            {
                Image MaskSoftPadded = MaskSoft.AsPadded(HalfMap1.Dims);
                MaskSoft.Dispose();
                MaskSoft = MaskSoftPadded;
            }

            _AnisoResolution?.Dispose();
            _MapFiltered?.Dispose();
            _MapFilteredSharpened?.Dispose();
            _MapLocallyFiltered?.Dispose();
            _LocalResolution?.Dispose();
            _LocalBFactor?.Dispose();

            Image MapSum = HalfMap1.GetCopyGPU();
            MapSum.Add(HalfMap2);
            MapSum.Multiply(0.5f);

            #region FSC

            float ShellGlobal;
            float[] FSCUnmasked, FSCMasked, FSCRandomized, FSCCorrected;
            if (fixedResolution <= 0)
            {
                FSC.GetCorrectedFSC(HalfMap1,
                                    HalfMap2,
                                    MaskSoft,
                                    0.143f,
                                    0.8f,
                                    out ShellGlobal,
                                    out FSCUnmasked,
                                    out FSCMasked,
                                    out FSCRandomized,
                                    out FSCCorrected);
            }
            else
            {
                ShellGlobal = MapSum.Dims.X * (float)PixelSize / fixedResolution;
                FSCUnmasked = FSCMasked = FSCRandomized = FSCCorrected = Helper.ArrayOfFunction(i => i <= ShellGlobal ? 1 : 1e-3f, MapSum.Dims.X / 2);
            }

            GlobalResolution = (decimal)((float)PixelSize * HalfMap1.Dims.X / ShellGlobal);
            GlobalFSC = Helper.ArrayOfFunction(i => new float5(HalfMap1.Dims.X * (float)PixelSize / i,
                                                               FSCUnmasked[i],
                                                               FSCRandomized[i],
                                                               FSCCorrected[i],
                                                               FSCMasked[i]),
                                               FSCCorrected.Length);

            progressCallback?.Invoke($"Global resolution is {GlobalResolution:F3}");

            //Console.WriteLine($"Global resolution is {GlobalResolution:F3}");
            //return;

            #endregion

            #region Anisotropic FSC

            progressCallback?.Invoke("2/5: Anisotropic FSC");

            if (fixedResolution <= 0)
                _AnisoResolution = FSC.GetAnisotropicFSC(HalfMap1,
                                                         HalfMap2,
                                                         MaskSoft,
                                                         (float)PixelSize,
                                                         0.143f,
                                                         2,
                                                         ShellGlobal);
            else
                _AnisoResolution = new Image(Helper.ArrayOfConstant(fixedResolution, 4), new int3(2, 2, 1));

            #endregion

            #region Weight & sharpen

            float BFacGlobal, BFacQuality;
            FSC.GetWeightedSharpened(MapSum,
                                     FSCCorrected,
                                     (float)PixelSize,
                                     fixedResolution <= 0 ? 10f : 0,
                                     (float)PixelSize * 2,
                                     (int)(ShellGlobal + 0.5f),
                                     out BFacGlobal,
                                     out BFacQuality,
                                     out _MapFiltered,
                                     out _MapFilteredSharpened);

            GlobalBFactor = (int)BFacGlobal;

            #endregion

            #region Filter map anisotropically

            if (PixelSize < 5)
            {
                Image MapSumMasked = MapSum.GetCopy();
                MapSumMasked.Multiply(MaskSoft);

                float[] Spectrum = MapSumMasked.AsAmplitudes1D(true, 1, MapSumMasked.Dims.X / 2);
                MapSumMasked.Dispose();

                int i10A = (int)((float)PixelSize * 2 / 10 * Spectrum.Length);
                float Amp10A = Spectrum[i10A];

                for (int i = 0; i < Spectrum.Length; i++)
                    Spectrum[i] = i < i10A ? 1 : (Amp10A / Spectrum[i]);

                Image MapSumFlat = MapSum.AsSpectrumMultiplied(true, Spectrum);

                _MapFilteredAnisotropic = FSC.FilterToAnisotropicResolution(MapSumFlat,
                                                                            _AnisoResolution,
                                                                            (float)PixelSize);
                MapSumFlat.Dispose();
            }
            else
            {

                _MapFilteredAnisotropic = FSC.FilterToAnisotropicResolution(MapSum,
                                                                            _AnisoResolution,
                                                                            (float)PixelSize);
            }

            #endregion

            #region Local resolution

            progressCallback?.Invoke("3/5: Local resolution");

            _LocalResolution = new Image(MapSum.Dims);
            _LocalBFactor = new Image(MapSum.Dims);

            // If fixed resolution is desired, locally filtered map will be identical with globally filtered one
            //if (false)
            if (fixedResolution <= 0)
            {
                int LocalResolutionWindow = Math.Max(30, Math.Min(50, (int)(GlobalResolution * 5 / PixelSize / 2 + 0.5M) * 2));

                _LocalResolution.Fill(LocalResolutionWindow * (float)PixelSize / 2);
                _LocalBFactor.Fill(BFacGlobal);

                if (LocalResolutionWindow <= HalfMap1.Dims.X)
                {
                    #region Mask maps and calculate local resolution as well as average FSC and amplitude curves for each resolution

                    Image Half1Masked = HalfMap1.GetCopyGPU();
                    HalfMap1.FreeDevice();
                    Half1Masked.Multiply(MaskSoft);
                    Image Half2Masked = HalfMap2.GetCopyGPU();
                    HalfMap2.FreeDevice();
                    Half2Masked.Multiply(MaskSoft);

                    int SpectrumLength = LocalResolutionWindow / 2;
                    int SpectrumOversampling = 2;
                    int NSpectra = SpectrumLength * SpectrumOversampling;

                    Image AverageFSC = new Image(new int3(SpectrumLength, 1, NSpectra));
                    Image AverageAmps = new Image(AverageFSC.Dims);
                    Image AverageSamples = new Image(new int3(NSpectra, 1, 1));
                    float[] GlobalLocalFSC = new float[LocalResolutionWindow / 2];

                    GPU.LocalFSC(Half1Masked.GetDevice(Intent.Read),
                                 Half2Masked.GetDevice(Intent.Read),
                                 MaskSoft.GetDevice(Intent.Read),
                                 HalfMap1.Dims,
                                 1,
                                 (float)PixelSize,
                                 _LocalResolution.GetDevice(Intent.Write),
                                 LocalResolutionWindow,
                                 0.143f,
                                 AverageFSC.GetDevice(Intent.ReadWrite),
                                 AverageAmps.GetDevice(Intent.ReadWrite),
                                 AverageSamples.GetDevice(Intent.ReadWrite),
                                 SpectrumOversampling,
                                 GlobalLocalFSC);

                    Half1Masked.Dispose();
                    Half2Masked.Dispose();
                    AverageFSC.FreeDevice();
                    AverageAmps.FreeDevice();
                    AverageSamples.FreeDevice();

                    #endregion

                    #region Figure out scaling factor to bring mean local resolution to global value

                    float LocalResScale;
                    {
                        Image MapSumAbs = MapSum.GetCopyGPU();
                        MapSumAbs.Abs();
                        MapSumAbs.Multiply(MaskSoft);
                        Image MapSumAbsConvolved = MapSumAbs.AsConvolvedRaisedCosine(0, LocalResolutionWindow / 2);
                        MapSumAbsConvolved.Multiply(MaskSoft);
                        MapSumAbs.Dispose();

                        float[] LocalResData = LocalResolution.GetHostContinuousCopy();
                        float[] MaskConvolvedData = MapSumAbsConvolved.GetHostContinuousCopy();
                        MapSumAbsConvolved.Dispose();

                        double WeightedSum = 0;
                        double Weights = 0;
                        for (int i = 0; i < LocalResData.Length; i++)
                        {
                            float Freq = 1 / LocalResData[i];

                            // No idea why local power * freq^2 produces good results, but it does!
                            float Weight = MaskConvolvedData[i] * Freq;
                            Weight *= Weight;

                            WeightedSum += Freq * Weight;
                            Weights += Weight;
                        }
                        WeightedSum /= Weights;
                        WeightedSum = 1 / WeightedSum;
                        LocalResScale = (float)GlobalResolution / (float)WeightedSum;
                        LocalResScale = 1;
                    }

                    #endregion

                    #region Build resolution-dependent B-factor model

                    #region Get average 1D amplitude spectra for each local resolution, weight by corresponding average local FSC curve, fit B-factors

                    List<float3> ResolutionBFacs = new List<float3>();
                    float[] AverageSamplesData = AverageSamples.GetHostContinuousCopy();

                    float[][] AverageFSCData = AverageFSC.GetHost(Intent.Read).Select((a, i) => a.Select(v => v / Math.Max(1e-10f, AverageSamplesData[i])).ToArray()).ToArray();
                    float[][] AverageAmpsData = AverageAmps.GetHost(Intent.Read);

                    float[][] FSCWeights = AverageFSCData.Select(a => a.Select(v => (float)Math.Sqrt(Math.Max(0, 2 * v / (1 + v)))).ToArray()).ToArray();
                    AverageAmpsData = AverageAmpsData.Select((a, i) => MathHelper.Mult(a, FSCWeights[i])).ToArray();

                    float[] ResInv = Helper.ArrayOfFunction(i => i / (float)(LocalResolutionWindow * PixelSize), SpectrumLength);

                    for (int i = 0; i < NSpectra; i++)
                    {
                        if (AverageSamplesData[i] < 100)
                            continue;

                        float ShellFirst = LocalResolutionWindow * (float)PixelSize / 10f;
                        float ShellLast = i / (float)SpectrumOversampling * LocalResScale;
                        if (ShellLast - ShellFirst + 1 < 2.5f)
                            continue;

                        float[] ShellWeights = Helper.ArrayOfFunction(s =>
                        {
                            float WeightFirst = Math.Max(0, Math.Min(1, 1 - (ShellFirst - s)));
                            float WeightLast = Math.Max(0, Math.Min(1, 1 - (s - ShellLast)));
                            return Math.Min(WeightFirst, WeightLast);
                        }, SpectrumLength);

                        float3[] Points = Helper.ArrayOfFunction(s => new float3(ResInv[s] * ResInv[s],
                                                                                 (float)Math.Log(Math.Max(1e-20f, AverageAmpsData[i][s])),
                                                                                 ShellWeights[s]), SpectrumLength);

                        float3 Fit = MathHelper.FitLineWeighted(Points.Skip(1).ToArray());
                        ResolutionBFacs.Add(new float3(ShellLast, -Fit.X * 4, AverageSamplesData[i]));
                    }

                    #endregion

                    #region Re-scale per-resolution B-factors to match global average, fit a * freq^b params to fit frequency vs. B-factor plot

                    float2 BFactorModel;
                    {
                        float WeightedMeanBFac = MathHelper.MeanWeighted(ResolutionBFacs.Select(v => v.Y).ToArray(), ResolutionBFacs.Select(v => v.Z * v.Z * v.X).ToArray());
                        //ResolutionBFacs = ResolutionBFacs.Select(v => new float3(v.X, v.Y * (float)-GlobalBFactor / WeightedMeanBFac, v.Z)).ToList();
                        //ResolutionBFacs = ResolutionBFacs.Select(v => new float3(v.X, v.Y, v.Z)).ToList();

                        float3[] BFacsLogLog = ResolutionBFacs.Select(v => new float3((float)Math.Log10(v.X), (float)Math.Log10(v.Y), v.Z)).ToArray();
                        float3 LineFit = MathHelper.FitLineWeighted(BFacsLogLog);

                        BFactorModel = new float2((float)Math.Pow(10, LineFit.Y), LineFit.X);
                    }

                    #endregion

                    #region Calculate filter ramps for each local resolution value

                    // Filter ramp consists of low-pass filter * FSC-based weighting (Rosenthal 2003) * B-factor correction
                    // Normalized by average ramp value within [0; low-pass shell] to prevent low-resolution regions having lower intensity

                    float[][] FilterRampsData = new float[NSpectra][];
                    for (int i = 0; i < NSpectra; i++)
                    {
                        float ShellLast = i / (float)SpectrumOversampling;
                        float[] ShellWeights = Helper.ArrayOfFunction(s => Math.Max(0, Math.Min(1, 1 - (s - ShellLast))), SpectrumLength);

                        float BFac = i == 0 ? 0 : BFactorModel.X * (float)Math.Pow(ShellLast, BFactorModel.Y);
                        float[] ShellSharps = ResInv.Select(r => (float)Math.Exp(Math.Min(50, r * r * BFac * 0.25))).ToArray();

                        FilterRampsData[i] = MathHelper.Mult(ShellWeights, ShellSharps);
                        float FilterSum = FilterRampsData[i].Sum() / Math.Max(1, ShellLast);

                        if (AverageSamplesData[i] > 10)
                            FilterRampsData[i] = MathHelper.Mult(FilterRampsData[i], FSCWeights[i]);
                        FilterRampsData[i] = FilterRampsData[i].Select(v => v / FilterSum).ToArray();
                    }
                    Image FilterRamps = new Image(FilterRampsData, new int3(SpectrumLength, 1, NSpectra));

                    #endregion

                    #endregion

                    #region Convolve local res with small Gaussian to get rid of local outliers
                    {
                        Image LocalResInv = new Image(IntPtr.Zero, LocalResolution.Dims);
                        LocalResInv.Fill((float)PixelSize * LocalResolutionWindow * LocalResScale); // Scale mean to global resolution in this step as well

                        LocalResInv.Divide(LocalResolution);
                        LocalResolution.Dispose();

                        Image LocalResInvSmooth = LocalResInv.AsConvolvedGaussian(1f, true);
                        LocalResInv.Dispose();

                        LocalResolution = new Image(IntPtr.Zero, LocalResInv.Dims);
                        LocalResolution.Fill((float)PixelSize * LocalResolutionWindow);
                        LocalResolution.Divide(LocalResInvSmooth);
                        LocalResInvSmooth.Dispose();
                    }
                    #endregion

                    #region Make local resolution histogram
                    {
                        int[] Bins = new int[LocalResolutionWindow / 2 * SpectrumOversampling];
                        float[] LocalResData = LocalResolution.GetHostContinuousCopy();
                        float[] SoftMaskData = MaskSoft.GetHostContinuousCopy();
                        float ScaleFactor = LocalResolutionWindow * (float)PixelSize * SpectrumOversampling;
                        for (int i = 0; i < LocalResData.Length; i++)
                        {
                            if (SoftMaskData[i] > 0.5f)
                                Bins[Math.Max(0, Math.Min(Bins.Length - 1, (int)(ScaleFactor / LocalResData[i] + 0.5f)))]++;
                        }

                        ResolutionHistogram = new float2[NSpectra];

                        for (int i = 0; i < NSpectra; i++)
                        {
                            float Shell = i / (float)SpectrumOversampling * LocalResScale;
                            float Resolution = LocalResolutionWindow * (float)PixelSize / Shell;
                            float Samples = Bins[i];

                            ResolutionHistogram[i] = new float2(Resolution, Samples);
                        }
                    }
                    #endregion

                    #region Finally, apply filter ramps (low-pass + B-factor) locally

                    MapLocallyFiltered = new Image(MapSum.Dims);
                    GPU.LocalFilter(MapSum.GetDevice(Intent.Read),
                                    MapLocallyFiltered.GetDevice(Intent.Write),
                                    MapSum.Dims,
                                    LocalResolution.GetDevice(Intent.Read),
                                    LocalResolutionWindow,
                                    (float)PixelSize,
                                    FilterRamps.GetDevice(Intent.Read),
                                    SpectrumOversampling);

                    //MapLocallyFiltered.WriteMRC("d_localfilt.mrc", (float)PixelSize, true);

                    #endregion
                }
            }
            else
            {
                _LocalResolution.Fill(fixedResolution);
                _LocalBFactor.Fill(0);
                MapLocallyFiltered = MapFilteredSharpened.GetCopyGPU();
            }

            #endregion

            #region Denoising

            progressCallback?.Invoke("4/5: Training denoising");

            HalfMap1.FreeDevice();
            HalfMap2.FreeDevice();
            MaskSoft.FreeDevice();

            bool TrainingFromScratch = DenoisedAtResolution <= 0;

            if (Denoiser == null)
                //Denoiser = new NoiseNet3DTorch(new int3(64), new[] { gpuID });
                Denoiser = new NoiseNet3DTorch(new int3(32), new[] { gpuID }, depth: 1, progressiveDepth: false, maxWidth: 64, batchSize: 64);
            if (!TrainingFromScratch)
                Denoiser.Load(PathNoiseNet);

            List<float> LocalResSorted = LocalResolution.GetHostContinuousCopy().ToList();
            //List<float> MaskData = Mask.GetHostContinuousCopy().ToList();
            //LocalResSorted = LocalResSorted.Where((v, i) => MaskData[i] > 0.5f).ToList();
            LocalResSorted.Sort();
            DenoisedAtResolution = Math.Max(PixelSize * 2, (decimal)LocalResSorted[(int)(0.005 * LocalResSorted.Count)]);
            DenoisedAtResolution = Math.Min(DenoisedAtResolution, GlobalResolution);
            if (fixedResolution > 0)
                DenoisedAtResolution = Math.Max(DenoisedAtResolution, (decimal)fixedResolution);

            HalfMap1.PixelSize = (float)PixelSize;
            HalfMap2.PixelSize = (float)PixelSize;

            (var ForDenoising1, var ForDenoising2, var ForDenoisingStats) = NoiseNet3DTorch.TrainOnVolumes(Denoiser,
                                                                                                          new[] { HalfMap1 },
                                                                                                          new[] { HalfMap2 },
                                                                                                          new[] { MaskSoft },
                                                                                                          (float)PixelSize,
                                                                                                          (float)DenoisedAtResolution,
                                                                                                          1.0f,
                                                                                                          false,
                                                                                                          true,
                                                                                                          TrainingFromScratch ? 3000 * 8 : 600 * 8,
                                                                                                          0,
                                                                                                          Denoiser.BatchSize,
                                                                                                          gpuID,
                                                                                                          s => progressCallback?.Invoke($"4/5: Training denoising: {s}"));

            GPU.SetDevice(gpuID);

            Denoiser.Save(PathNoiseNet);
            //Denoiser.Dispose();

            progressCallback?.Invoke("5/5: Applying denoising");

            MapDenoised = ForDenoising1[0];
            MapDenoised.Add(ForDenoising2[0]);
            ForDenoising2[0].Dispose();
            MapDenoised.Multiply(0.5f);

            NoiseNet3DTorch.Denoise(MapDenoised, new[] { Denoiser });
            //Denoiser.Dispose();
            //Denoiser = null;
            TorchSharp.Torch.CudaEmptyCache();
            //TFHelper.TF_FreeAllMemory();

            MapDenoised.TransformValues(v => v * ForDenoisingStats[0].Y + ForDenoisingStats[0].X);
            MapDenoised.FreeDevice();

            #endregion

            MapSum.Dispose();
            MaskSoft.Dispose();

            FreeDevice();
        }

        public void CalculateParticleStats()
        {
            _AngularDistribution?.Dispose();

            int2 DimsPlot = new int2(1024);
            float2[] UniqueAngles = Helper.GetHealpixRotTilt(3, "C1", 91f);

            int AngleOversample = 8;
            float fAngleOversample = AngleOversample;

            float2[] BinAngles = UniqueAngles.Select(v => new float2(v)).ToArray();
            float3[] BinVecs = BinAngles.Select(v => Matrix3.Euler(v.X * Helper.ToRad, v.Y * Helper.ToRad, 0).Transposed() * new float3(0, 0, 1)).ToArray();
            int[][] ClosestBin = Helper.ArrayOfFunction(i => new int[90 * AngleOversample + 1], 360 * AngleOversample + 1);
            Parallel.For(0, 90 * AngleOversample + 1, tilt =>
            {
                for (int rot = 0; rot < 360 * AngleOversample + 1; rot++)
                {
                    float3 AngleVec = Matrix3.Euler(rot / fAngleOversample * Helper.ToRad, tilt / fAngleOversample * Helper.ToRad, 0).Transposed() * new float3(0, 0, 1);

                    int ClosestID = 0;
                    float ClosestDist = float.MaxValue;
                    for (int i = 0; i < BinAngles.Length; i++)
                    {
                        float Dist = (BinVecs[i] - AngleVec).Length();
                        if (Dist < ClosestDist)
                        {
                            ClosestID = i;
                            ClosestDist = Dist;
                        }
                    }

                    ClosestBin[rot][tilt] = ClosestID;
                }
            });

            int[] AngularBins = new int[BinAngles.Length];

            float2[] ParticleAngles = DescendantParticles.Select(p => new float2(p.AnglesMean.X, p.AnglesMean.Y)).ToArray();

            ParticleAngles = ParticleAngles.Select(v => new float2(v.X < 0 ? v.X + 360 : v.X, v.Y)).ToArray();
            for (int p = 0; p < ParticleAngles.Length; p++)
            {
                float Rot = ParticleAngles[p].X;
                float Tilt = ParticleAngles[p].Y;
                if (Tilt > 90)
                {
                    Rot += 180;
                    Tilt = 180 - Tilt;
                }
                if (Rot > 360)
                    Rot -= 360;
                ParticleAngles[p] = new float2(Rot, Tilt);
            }

            foreach (var particleAngle in ParticleAngles)
            {
                int Rot = (int)Math.Max(0, Math.Min(360 * AngleOversample, Math.Round(particleAngle.X * AngleOversample)));
                int Tilt = (int)Math.Max(0, Math.Min(90 * AngleOversample, Math.Round(particleAngle.Y * AngleOversample)));
                AngularBins[ClosestBin[Rot][Tilt]]++;
            }

            float[] PlotData = new float[DimsPlot.Elements()];

            for (int y = 0; y < DimsPlot.Y; y++)
            {
                float yy = (y - DimsPlot.Y / 2f) / (DimsPlot.Y / 2f);

                for (int x = 0; x < DimsPlot.X; x++)
                {
                    float xx = (x - DimsPlot.X / 2f) / (DimsPlot.X / 2f);

                    float Rot = (float)Math.Atan2(yy, xx) * Helper.ToDeg;
                    if (Rot < 0)
                        Rot += 360;
                    Rot = Math.Max(0, Math.Min(360, Rot));

                    float Tilt = (float)Math.Sqrt(xx * xx + yy * yy) * 90;
                    if (Tilt > 90)
                        continue;

                    PlotData[y * DimsPlot.X + x] = AngularBins[ClosestBin[(int)Math.Round(Rot * fAngleOversample)][(int)Math.Round(Tilt * fAngleOversample)]];
                }
            }

            _AngularDistribution = new Image(PlotData, new int3(DimsPlot));
        }

        public void PrepareRefinementRequisites(bool makeRefs = true, int singleGPU = -1, string stagingSave = null, string stagingLoad = null)
        {
            string StagingDirectory = "";
            if (!string.IsNullOrEmpty(stagingSave))
                StagingDirectory = System.IO.Path.Join(stagingSave, GUID.ToString().Substring(0, 8));
            else if (!string.IsNullOrEmpty(stagingLoad))
                StagingDirectory = System.IO.Path.Join(stagingLoad, GUID.ToString().Substring(0, 8));

            if (!string.IsNullOrEmpty(stagingSave))
                Directory.CreateDirectory(StagingDirectory);

            int3 DimsCurrent = HalfMap1.Dims;
            int3 DimsDesired = new int3(Size);

            // Refinement resolution is at FSC = 0.143
            ResolutionRefinement = (float)GlobalResolution;
            int SizeRefinement = (int)Math.Round((float)PixelSize * 2 / ResolutionRefinement * DimsDesired.X / 2) * 2;
            ResolutionRefinement = DimsDesired.X / (float)SizeRefinement * (float)PixelSize * 2;    // Adjust resolution/pixel size to match multiple-of-2-sized box

            if (makeRefs)
            {
                Image HalfMap1Averaged = null, HalfMap2Averaged = null;

                if (string.IsNullOrEmpty(stagingLoad))
                {
                    //if (Math.Abs(Mask.PixelSize - (float)PixelSize) > 0.02f)
                    //{
                    //    int DimScaled = (int)Math.Round((Mask.PixelSize / (float)PixelSize) * Mask.Dims.X / 2) * 2;
                    //    Image Scaled = Mask.AsScaled(new int3(DimScaled));
                    //    Scaled.Binarize(0.5f);

                    //    Scaled.WriteMRC(PathMask, (float)PixelSize, true);
                    //    Scaled.Dispose();
                    //    _Mask.Dispose();
                    //    _Mask = null;
                    //}

                    Image HalfMap1Padded = HalfMap1;
                    Image HalfMap2Padded = HalfMap2;
                    Image MaskPadded = Mask;

                    // Pad maps if size isn't 2 * particle diameter
                    if (DimsDesired != HalfMap1.Dims)
                    {
                        HalfMap1Padded = HalfMap1.AsPadded(DimsDesired);
                        HalfMap1.FreeDevice();
                    }
                    if (DimsDesired != HalfMap2.Dims)
                    {
                        HalfMap2Padded = HalfMap2.AsPadded(DimsDesired);
                        HalfMap2.FreeDevice();
                    }
                    if (DimsDesired != Mask.Dims)
                    {
                        MaskPadded = Mask.AsPadded(DimsDesired);
                        Mask.FreeDevice();
                    }

                    #region Denoise half-maps

                    HalfMap1Padded.FreeDevice();
                    HalfMap2Padded.FreeDevice();
                    MaskPadded.FreeDevice();

                    if (ApplyDenoising)
                    {
                        if (true)
                        {
                            if (Denoiser == null)
                                Denoiser = new NoiseNet3DTorch(new int3(32), new[] { singleGPU < 0 ? GPU.GetDeviceWithMostMemory() : singleGPU }, depth: 1, progressiveDepth: false, maxWidth: 64, batchSize: 64);
                            Denoiser.Load(PathNoiseNet);

                            HalfMap1Padded.PixelSize = (float)PixelSize;
                            HalfMap2Padded.PixelSize = (float)PixelSize;

                            // Not actually training, just preparing input data the same way it was for training
                            (var ForDenoising1, var ForDenoising2, var ForDenoisingStats) = NoiseNet3DTorch.TrainOnVolumes(Denoiser,
                                                                                                                          new[] { HalfMap1Padded },
                                                                                                                          new[] { HalfMap2Padded },
                                                                                                                          new[] { MaskPadded },
                                                                                                                          (float)PixelSize,
                                                                                                                          (float)DenoisedAtResolution,
                                                                                                                          1.0f,
                                                                                                                          false,
                                                                                                                          false,
                                                                                                                          0,
                                                                                                                          0,
                                                                                                                          Denoiser.BatchSize,
                                                                                                                          singleGPU < 0 ? ((GPU.GetDeviceWithMostMemory() + 1) % GPU.GetDeviceCount()) : singleGPU,
                                                                                                                          null);

                            GPU.SetDevice(singleGPU < 0 ? 0 : singleGPU);

                            using (new InferenceMode(true))
                            {
                                NoiseNet3DTorch.Denoise(ForDenoising1[0], new[] { Denoiser });
                                NoiseNet3DTorch.Denoise(ForDenoising2[0], new[] { Denoiser });
                            }

                            TorchSharp.Torch.CudaEmptyCache();

                            GPU.CheckGPUExceptions();

                            GPU.SetDevice(Math.Max(0, singleGPU));

                            ForDenoising1[0].TransformValues(v => v * ForDenoisingStats[0].Y + ForDenoisingStats[0].X);
                            HalfMap1Padded.FreeDevice();
                            HalfMap1Padded = ForDenoising1[0];
                            //HalfMap1Padded.WriteMRC($"d_denoised_half1_{Name}_{GPU.GetDevice()}.mrc", true);

                            ForDenoising2[0].TransformValues(v => v * ForDenoisingStats[0].Y + ForDenoisingStats[0].X);
                            HalfMap2Padded.FreeDevice();
                            HalfMap2Padded = ForDenoising2[0];
                            //HalfMap2Padded.WriteMRC($"d_denoised_half2_{Name}_{GPU.GetDevice()}.mrc", true);
                        }
                        else
                        {
                            HalfMap1Padded = Image.FromFile($"d_denoised_half1_{Name}_{GPU.GetDevice()}.mrc");
                            HalfMap2Padded = Image.FromFile($"d_denoised_half2_{Name}_{GPU.GetDevice()}.mrc");
                        }
                    }

                    #endregion

                    GPU.SetDevice(singleGPU < 0 ? 0 : singleGPU);

                    // Scale down to refinement resolution
                    Image HalfMap1Scaled = HalfMap1Padded.AsScaled(new int3(SizeRefinement));
                    HalfMap1Padded.FreeDevice();
                    Image HalfMap2Scaled = HalfMap2Padded.AsScaled(new int3(SizeRefinement));
                    HalfMap2Padded.FreeDevice();

                    GPU.CheckGPUExceptions();

                    if (!ApplyDenoising)
                    {
                        // Filter maps according to SSNR
                        // Weight by the masked, uncorrected FSC to avoid the artifacts near randomization limit
                        float[] SpectralWeighting = GlobalFSC.Select(v => Math.Max(0, v.V)).Take(SizeRefinement / 2).ToArray();
                        Image HalfMap1SSNRFiltered = HalfMap1Scaled.AsSpectrumMultiplied(true, SpectralWeighting);
                        //HalfMap1SSNRFiltered.WriteMRC("d_filt1.mrc", true);
                        HalfMap1Scaled.FreeDevice();
                        Image HalfMap2SSNRFiltered = HalfMap2Scaled.AsSpectrumMultiplied(true, SpectralWeighting);
                        //HalfMap2SSNRFiltered.WriteMRC("d_filt2.mrc", true);
                        HalfMap2Scaled.FreeDevice();
                    }

                    // Average halves up to 40 A or where FSC falls below 0.998, whatever is higher
                    int Shell40A = (int)Math.Round((float)PixelSize / 40 * Size);
                    int ShellHighFSC = 0;
                    //while (GlobalFSC[ShellHighFSC].W > 0.998f && ShellHighFSC < Size / 2 - 2) ShellHighFSC++;
                    FSC.AverageLowFrequencies(HalfMap1Scaled, HalfMap2Scaled, Math.Max(Shell40A, ShellHighFSC), out HalfMap1Averaged, out HalfMap2Averaged);
                    HalfMap1Scaled.Dispose();
                    HalfMap2Scaled.Dispose();

                    GPU.CheckGPUExceptions();

                    // Scale mask to refinement size and make it soft
                    Image MaskScaled = MaskPadded.AsScaled(new int3(SizeRefinement));
                    MaskPadded.FreeDevice();
                    MaskScaled.Binarize(0.25f);
                    Image MaskSoft = FSC.MakeSoftMask(MaskScaled, 3, 6);
                    MaskScaled.Dispose();

                    MaskSoft.MaskSpherically(MaskSoft.Dims.X / 2, Math.Max(2, Math.Min(16, MaskSoft.Dims.X / 8f)), true);

                    GPU.CheckGPUExceptions();

                    // Multiply half-maps with mask
                    HalfMap1Averaged.Multiply(MaskSoft);
                    // HalfMap1Averaged.WriteMRC($"d_reference_half1_{Name}_{singleGPU}.mrc", true);
                    if (!string.IsNullOrEmpty(stagingSave))
                        HalfMap1Averaged.WriteMRC(System.IO.Path.Join(StagingDirectory, "half1.mrc"), true);

                    HalfMap2Averaged.Multiply(MaskSoft);
                    // HalfMap2Averaged.WriteMRC($"d_reference_half2_{Name}_{singleGPU}.mrc", true);
                    if (!string.IsNullOrEmpty(stagingSave))
                        HalfMap2Averaged.WriteMRC(System.IO.Path.Join(StagingDirectory, "half2.mrc"), true);

                    if (!string.IsNullOrEmpty(stagingSave))
                        MaskSoft.WriteMRC(System.IO.Path.Join(StagingDirectory, "mask.mrc"), true);

                    MaskSoft.Dispose();

                    GPU.CheckGPUExceptions();
                }
                else
                {
                    HalfMap1Averaged = Image.FromFile(System.IO.Path.Join(StagingDirectory, "half1.mrc"));
                    HalfMap2Averaged = Image.FromFile(System.IO.Path.Join(StagingDirectory, "half2.mrc"));
                }

                // Create reference projectors for refinement
                HalfMap1Projector = Helper.ArrayOfFunction(i => (singleGPU < 0 || singleGPU == i) ? new Projector(HalfMap1Averaged, 1) : null, GPU.GetDeviceCount());
                HalfMap1Averaged.Dispose();
                HalfMap2Projector = Helper.ArrayOfFunction(i => (singleGPU < 0 || singleGPU == i) ? new Projector(HalfMap2Averaged, 1) : null, GPU.GetDeviceCount());
                HalfMap2Averaged.Dispose();

                GPU.CheckGPUExceptions();
            }

            // Initialize reconstructions at full Nyquist
            ResolutionReconstruction = (float)PixelSize * 2;
            int SizeReconstruction = DimsDesired.X;

            HalfMap1Reconstruction = Helper.ArrayOfFunction(i => (singleGPU < 0 || singleGPU == i) ? new Projector(new int3(SizeReconstruction), 1) : null, GPU.GetDeviceCount());
            HalfMap2Reconstruction = Helper.ArrayOfFunction(i => (singleGPU < 0 || singleGPU == i) ? new Projector(new int3(SizeReconstruction), 1) : null, GPU.GetDeviceCount());
        }

        public void GatherRefinementProgress(string[] folders, int gpuID = -1)
        {
            if (gpuID >= 0)
                GPU.SetDevice(gpuID);

            string SpeciesID = GUID.ToString().Substring(0, 8);

            Projector Rec1 = HalfMap1Reconstruction[0];
            Projector Rec2 = HalfMap2Reconstruction[0];

            Particle[] OriginalParticles = Particles.ToList().ToArray();
            Particle[] FinalParticles = Particles;

            foreach (var folder in folders)
            {
                Projector[] Recs = { Rec1, Rec2 };

                Helper.ForCPU(0, 2, 2, null, (ihalf, threadID) =>
                {
                    if (gpuID >= 0)
                        GPU.SetDevice(gpuID);

                    foreach (var path in Directory.EnumerateFiles(folder, $"{SpeciesID}_half{ihalf + 1}_*.mrc"))
                    {
                        Console.WriteLine("Gathering from " + path);
                        Projector Saved = Projector.FromFile(path);

                        Recs[ihalf].Data.Add(Saved.Data);
                        Recs[ihalf].Weights.Add(Saved.Weights);

                        Saved.Dispose();
                    }
                }, null);

                Particle[] UpdatedParticles = ParticlesFromStar(new Star(System.IO.Path.Combine(folder, $"{SpeciesID}_particles.star")));

                for (int p = 0; p < OriginalParticles.Length; p++)
                    if (OriginalParticles[p].Coordinates.Where((v, i) => v != UpdatedParticles[p].Coordinates[i]).Count() > 0 ||
                        OriginalParticles[p].Angles.Where((v, i) => v != UpdatedParticles[p].Angles[i]).Count() > 0)
                        FinalParticles[p] = UpdatedParticles[p];
            }

            Rec1.FreeDevice();
            Rec2.FreeDevice();
        }

        public void FinishRefinement(int gpuID = -1)
        {
            if (gpuID >= 0)
                GPU.SetDevice(gpuID);

            //HalfMap1Reconstruction.Data.Fill(new float2(1, 0));
            //HalfMap1Reconstruction.Weights.Fill(1);

            for (int i = 0; i < HalfMap1Reconstruction.Length; i++)
            {
                HalfMap1Reconstruction[i]?.FreeDevice();
                HalfMap2Reconstruction[i]?.FreeDevice();

                if (HalfMap1Projector != null)
                {
                    HalfMap1Projector[i]?.Dispose();
                    HalfMap2Projector[i]?.Dispose();
                }
            }

            for (int i = 1; i < HalfMap1Reconstruction.Length; i++)
            {
                if (HalfMap1Reconstruction[i] == null)
                    continue;

                HalfMap1Reconstruction[0].Data.Add(HalfMap1Reconstruction[i].Data);
                HalfMap1Reconstruction[0].Weights.Add(HalfMap1Reconstruction[i].Weights);
                HalfMap1Reconstruction[i].Dispose();
                HalfMap1Reconstruction[0].FreeDevice();

                HalfMap2Reconstruction[0].Data.Add(HalfMap2Reconstruction[i].Data);
                HalfMap2Reconstruction[0].Weights.Add(HalfMap2Reconstruction[i].Weights);
                HalfMap2Reconstruction[i].Dispose();
                HalfMap2Reconstruction[0].FreeDevice();
            }

            //HalfMap1Reconstruction[0].Data = new Image(Helper.Zip(Image.FromFile($"d_half1_re_{Name}.mrc").GetHostContinuousCopy(), Image.FromFile($"d_half1_im_{Name}.mrc").GetHostContinuousCopy()), HalfMap1Reconstruction[0].DimsOversampled, true);
            //HalfMap2Reconstruction[0].Data = new Image(Helper.Zip(Image.FromFile($"d_half2_re_{Name}.mrc").GetHostContinuousCopy(), Image.FromFile($"d_half2_im_{Name}.mrc").GetHostContinuousCopy()), HalfMap1Reconstruction[0].DimsOversampled, true);

            //HalfMap1Reconstruction[0].Weights = new Image(Image.FromFile($"d_half1_weights_{Name}.mrc").GetHostContinuousCopy(), HalfMap1Reconstruction[0].DimsOversampled, true);
            //HalfMap2Reconstruction[0].Weights = new Image(Image.FromFile($"d_half2_weights_{Name}.mrc").GetHostContinuousCopy(), HalfMap1Reconstruction[0].DimsOversampled, true);

            //HalfMap1Reconstruction[0].Data.AsReal().WriteMRC($"d_half1_re_{Name}.mrc");
            //HalfMap1Reconstruction[0].Data.AsImaginary().WriteMRC($"d_half1_im_{Name}.mrc");
            //HalfMap1Reconstruction[0].Weights.WriteMRC($"d_half1_weights_{Name}.mrc");

            //HalfMap2Reconstruction[0].Data.AsReal().WriteMRC($"d_half2_re_{Name}.mrc");
            //HalfMap2Reconstruction[0].Data.AsImaginary().WriteMRC($"d_half2_im_{Name}.mrc");
            //HalfMap2Reconstruction[0].Weights.WriteMRC($"d_half2_weights_{Name}.mrc");

            SaveParticles();

            HelicalSymmetry Helical = null;
            if (HelicalUnits > 1)
                Helical = new HelicalSymmetry(HelicalUnits, (float)HelicalTwist, (float)(HelicalRise / PixelSize));

            Console.WriteLine("Reconstructing half-map 1...");
            HalfMap1 = HalfMap1Reconstruction[0].Reconstruct(false, Symmetry, Helical, -1, -1, -1, 0, true);
            //HalfMap1Reconstruction[0].Data.AsReal().WriteMRC($"d_half1_re_{Name}.mrc");
            //HalfMap1Reconstruction[0].Weights.WriteMRC($"d_half1_weights_{Name}.mrc");
            HalfMap1Reconstruction[0].Dispose();
            HalfMap1Reconstruction[0] = null;
            HalfMap1.Bandpass(0, (float)(HalfMap1.Dims.X / 2 - 2) / (HalfMap1.Dims.X / 2), true);
            HalfMap1.MaskSpherically(HalfMap1.Dims.X - 32, 15, true);
            if (Helical != null)
                HalfMap1.MaskRectangularly(new int3(HalfMap1.Dims.X, HalfMap1.Dims.Y, (int)(HelicalHeight / PixelSize)), 8, true);
            HalfMap1.FreeDevice();
            if (Helper.IsDebug)
                HalfMap1.WriteMRC($"d_half1_{Name}.mrc", true);

            Console.WriteLine("Reconstructing half-map 2...");
            HalfMap2 = HalfMap2Reconstruction[0].Reconstruct(false, Symmetry, Helical, -1, -1, -1, 0, true);
            //HalfMap2Reconstruction[0].Data.AsReal().WriteMRC($"d_half2_re_{Name}.mrc");
            //HalfMap2Reconstruction[0].Weights.WriteMRC($"d_half2_weights_{Name}.mrc");
            HalfMap2Reconstruction[0].Dispose();
            HalfMap2Reconstruction[0] = null;
            HalfMap2.Bandpass(0, (float)(HalfMap2.Dims.X / 2 - 2) / (HalfMap2.Dims.X / 2), true);
            HalfMap2.MaskSpherically(HalfMap2.Dims.X - 32, 15, true);
            if (Helical != null)
                HalfMap2.MaskRectangularly(new int3(HalfMap2.Dims.X, HalfMap2.Dims.Y, (int)(HelicalHeight / PixelSize)), 8, true);
            HalfMap1.FreeDevice();
            if (Helper.IsDebug)
                HalfMap2.WriteMRC($"d_half2_{Name}.mrc", true);

            Console.WriteLine("Finalizing map...");
            CalculateResolutionAndFilter(-1, s => Console.WriteLine(s), gpuID);
        }

        #endregion

        #region File and directory bookkeeping

        public void DropMaps()
        {
            _AnisoResolution?.Dispose();
            _AnisoResolution = null;

            _LocalResolution?.Dispose();
            _LocalResolution = null;

            _LocalBFactor?.Dispose();
            _LocalBFactor = null;

            _Mask?.Dispose();
            _Mask = null;

            _HalfMap1?.Dispose();
            _HalfMap1 = null;

            _HalfMap2?.Dispose();
            _HalfMap2 = null;

            _MapFiltered?.Dispose();
            _MapFiltered = null;

            _MapFilteredSharpened?.Dispose();
            _MapFilteredSharpened = null;

            _MapFilteredAnisotropic?.Dispose();
            _MapFilteredAnisotropic = null;

            _MapLocallyFiltered?.Dispose();
            _MapLocallyFiltered = null;

            _AngularDistribution?.Dispose();
            _AngularDistribution = null;

            _MapDenoised?.Dispose();
            _MapDenoised = null;
        }

        public void FreeDevice()
        {
            _AnisoResolution?.FreeDevice();
            _LocalResolution?.FreeDevice();
            _LocalBFactor?.FreeDevice();
            _Mask?.FreeDevice();
            _HalfMap1?.FreeDevice();
            _HalfMap2?.FreeDevice();
            _MapFiltered?.FreeDevice();
            _MapFilteredSharpened?.FreeDevice();
            _MapFilteredAnisotropic?.FreeDevice();
            _MapLocallyFiltered?.FreeDevice();
            _AngularDistribution?.FreeDevice();
            _MapDenoised?.FreeDevice();
        }

        #endregion

        #region Saving

        public void Commit()
        {
            PreviousVersion = Version;
            ComputeVersionHash();

            if (PreviousVersion != null && Version == PreviousVersion)
                return;

            string VersionFolderPath = Helper.PathCombine(FolderPath, "versions", Version);
            Directory.CreateDirectory(VersionFolderPath);

            string FileName = Helper.PathToNameWithExtension(Path);
            string OriginalFolderPath = FolderPath;

            if (File.Exists(PathNoiseNet))
            {
                if (Denoiser == null)
                    Denoiser = new NoiseNet3DTorch(new int3(32), new[] { GPU.GetDeviceWithMostMemory() }, depth: 1, progressiveDepth: false, maxWidth: 64, batchSize: 64);
                Denoiser.Load(PathNoiseNet);
            }

            Path = Helper.PathCombine(VersionFolderPath, FileName);
            try
            {
                Save();
            }
            catch { }

            Path = Helper.PathCombine(OriginalFolderPath, FileName);
            Save();

            //Denoiser?.Dispose();
            //Denoiser = null;
            TorchSharp.Torch.CudaEmptyCache();
            //TFHelper.TF_FreeAllMemory();
        }

        public void Save()
        {
            XmlTextWriter Writer = new XmlTextWriter(File.Create(Path), Encoding.UTF8);
            Writer.Formatting = Formatting.Indented;
            Writer.IndentChar = '\t';
            Writer.Indentation = 1;
            Writer.WriteStartDocument();
            Writer.WriteStartElement("Species");

            WriteToXML(Writer);

            Writer.WriteStartElement("UsedDataSources");
            foreach (var source in UsedDataSources)
            {
                Writer.WriteStartElement("DataSource");
                XMLHelper.WriteAttribute(Writer, "GUID", source.Key.ToString());
                XMLHelper.WriteAttribute(Writer, "Version", source.Value);
                Writer.WriteEndElement();
            }
            Writer.WriteEndElement();

            Writer.WriteEndElement();
            Writer.WriteEndDocument();
            Writer.Flush();
            Writer.Close();

            SaveParticles();
            SaveMaps();
        }

        public void SaveMaps()
        {
            _HalfMap1?.WriteMRC(PathHalfMap1, (float)PixelSize, true);
            _HalfMap2?.WriteMRC(PathHalfMap2, (float)PixelSize, true);

            _Mask?.WriteMRC(PathMask, (float)PixelSize, true);

            _MapFiltered?.WriteMRC(PathMapFiltered, (float)PixelSize, true);
            _MapFilteredSharpened?.WriteMRC(PathMapFilteredSharpened, (float)PixelSize, true);
            _MapFilteredAnisotropic?.WriteMRC(PathMapFilteredAnisotropic, (float)PixelSize, true);
            _MapLocallyFiltered?.WriteMRC(PathMapLocallyFiltered, (float)PixelSize, true);
            _MapDenoised?.WriteMRC(PathMapDenoised, (float)PixelSize, true);

            _AnisoResolution?.WriteMRC(PathAnisoResolution, true);
            _LocalResolution?.WriteMRC(PathLocalResolution, (float)PixelSize, true);
            _LocalBFactor?.WriteMRC(PathLocalBFactor, (float)PixelSize, true);

            _AngularDistribution?.WriteMRC(PathAngularDist, true);

            // Save curves to STAR tables
            if (_GlobalFSC != null)
                new Star(_GlobalFSC, "wrpResolution", "wrpFSCUnmasked", "wrpFSCRandomized", "wrpFSCCorrected", "wrpFSCMasked").Save(PathGlobalFSC);
            if (_ResolutionHistogram != null)
                new Star(_ResolutionHistogram, "wrpResolution", "wrpSamples").Save(PathResolutionHistogram);

            Denoiser?.Save(PathNoiseNet);
        }

        public void SaveParticles()
        {
            ParticlesToStar()?.Save(PathParticleFile);
        }

        public void ComputeVersionHash()
        {
            StringBuilder Builder = new StringBuilder();

            foreach (var source in UsedDataSources)
            {
                Builder.Append(source.Key);
                Builder.Append(source.Value);
            }

            Builder.Append(MathHelper.GetSHA1(Helper.ToBytes(Helper.ToInterleaved(Helper.Combine(DescendantParticles.Select(p => p.Coordinates))))));
            Builder.Append(MathHelper.GetSHA1(Helper.ToBytes(Helper.ToInterleaved(Helper.Combine(DescendantParticles.Select(p => p.Angles))))));

            Builder.Append(MathHelper.GetSHA1(Helper.ToBytes(HalfMap1.GetHostContinuousCopy())));
            Builder.Append(MathHelper.GetSHA1(Helper.ToBytes(HalfMap2.GetHostContinuousCopy())));
            Builder.Append(MathHelper.GetSHA1(Helper.ToBytes(Mask.GetHostContinuousCopy())));

            Version = MathHelper.GetSHA1(Helper.ToBytes(Builder.ToString().ToCharArray())).Substring(0, 8);
        }

        #endregion

        #region Load XML

        public void Load(string path)
        {
            Path = path;

            if (!IsRemote)
            {
                using (Stream SettingsStream = File.OpenRead(path))
                {
                    XPathDocument Doc = new XPathDocument(SettingsStream);
                    XPathNavigator Reader = Doc.CreateNavigator();
                    Reader.MoveToRoot();

                    Reader.MoveToRoot();
                    Reader.MoveToChild("Species", "");

                    ReadFromXML(Reader);

                    UsedDataSources = new Dictionary<Guid, string>();
                    foreach (XPathNavigator nav in Reader.Select("UsedDataSources/DataSource"))
                    {
                        Guid SourceGUID = Guid.Parse(nav.GetAttribute("GUID", ""));
                        string SourceVersion = nav.GetAttribute("Version", "");

                        UsedDataSources.Add(SourceGUID, SourceVersion);
                    }
                }

                Particles = ParticlesFromStar(new Star(PathParticleFile));
            }
        }

        public void ResolveChildren(IEnumerable<Species> species)
        {
            Children.Clear();

            foreach (var guid in ChildrenGUIDs)
            {
                Species Match = species.First(s => s.GUID == guid);
                if (Match != null)
                    Children.Add(Match);
                else
                    throw new Exception($"Child with GUID = {guid} could not be resolved.");
            }
        }

        #endregion

        #region Static

        public static Species FromFile(string path)
        {
            Species Loaded = new Species();
            Loaded.Load(path);

            return Loaded;
        }

        #endregion
    }
}
