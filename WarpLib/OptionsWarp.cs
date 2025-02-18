using Newtonsoft.Json;
using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Text;
using System.Xml;
using System.Xml.XPath;
using Warp.Tools;
using ProcessingOptionsTomoFullReconstruction = Warp.TiltSeries.ProcessingOptionsTomoFullReconstruction;
using ProcessingOptionsTomoFullMatch = Warp.TiltSeries.ProcessingOptionsTomoFullMatch;
using ProcessingOptionsTomoSubReconstruction = Warp.TiltSeries.ProcessingOptionsTomoSubReconstruction;


using ProcessingOptionsMovieExport = Warp.Movie.ProcessingOptionsMovieExport;
using ProcessingOptionsMovieMovement = Warp.Movie.ProcessingOptionsMovieMovement;
using ProcessingOptionsMovieCTF = Warp.Movie.ProcessingOptionsMovieCTF;
using ProcessingOptionsParticleExport = Warp.Movie.ProcessingOptionsParticleExport;
using ProcessingOptionsFullMatch = Warp.Movie.ProcessingOptionsFullMatch;
using ProcessingOptionsBoxNet = Warp.Movie.ProcessingOptionsBoxNet;

namespace Warp
{
    [JsonObject(MemberSerialization.OptIn)]
    public class OptionsWarp : WarpBase
    {
        public ObservableCollection<string> _InputDatTypes = new ObservableCollection<string>
        {
            "int8", "int16", "int32", "int64", "float32", "float64"
        };
        public ObservableCollection<string> InputDatTypes
        {
            get { return _InputDatTypes; }
        } 
        
        #region Pixel size

        #endregion

        #region Things to process

        private bool _ProcessCTF = true;
        [WarpSerializable]
        [JsonProperty]
        public bool ProcessCTF
        {
            get { return _ProcessCTF; }
            set { if (value != _ProcessCTF) { _ProcessCTF = value; OnPropertyChanged(); } }
        }

        private bool _ProcessMovement = true;
        [WarpSerializable]
        [JsonProperty]
        public bool ProcessMovement
        {
            get { return _ProcessMovement; }
            set { if (value != _ProcessMovement) { _ProcessMovement = value; OnPropertyChanged(); } }
        }

        private bool _ProcessPicking = false;
        [WarpSerializable]
        [JsonProperty]
        public bool ProcessPicking
        {
            get { return _ProcessPicking; }
            set { if (value != _ProcessPicking) { _ProcessPicking = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Sub-categories

        private OptionsImport _Import = new OptionsImport();
        [JsonProperty]
        public OptionsImport Import
        {
            get { return _Import; }
            set { if (value != _Import) { _Import = value; OnPropertyChanged(); } }
        }

        private OptionsCTF _CTF = new OptionsCTF();
        [JsonProperty]
        public OptionsCTF CTF
        {
            get { return _CTF; }
            set { if (value != _CTF) { _CTF = value; OnPropertyChanged(); } }
        }

        private OptionsMovement _Movement = new OptionsMovement();
        [JsonProperty]
        public OptionsMovement Movement
        {
            get { return _Movement; }
            set { if (value != _Movement) { _Movement = value; OnPropertyChanged(); } }
        }

        private OptionsGrids _Grids = new OptionsGrids();
        [JsonProperty]
        public OptionsGrids Grids
        {
            get { return _Grids; }
            set { if (value != _Grids) { _Grids = value; OnPropertyChanged(); } }
        }

        private OptionsPicking _Picking = new OptionsPicking();
        [JsonProperty]
        public OptionsPicking Picking
        {
            get { return _Picking; }
            set { if (value != _Picking) { _Picking = value; OnPropertyChanged(); } }
        }

        private OptionsTomo _Tomo = new OptionsTomo();
        [JsonProperty]
        public OptionsTomo Tomo
        {
            get { return _Tomo; }
            set { if (value != _Tomo) { _Tomo = value; OnPropertyChanged(); } }
        }

        private OptionsExport _Export = new OptionsExport();
        [JsonProperty]
        public OptionsExport Export
        {
            get { return _Export; }
            set { if (value != _Export) { _Export = value; OnPropertyChanged(); } }
        }

        private OptionsTasks _Tasks = new OptionsTasks();
        [JsonProperty]
        public OptionsTasks Tasks
        {
            get { return _Tasks; }
            set { if (value != _Tasks) { _Tasks = value; OnPropertyChanged(); } }
        }

        private OptionsFilter _Filter = new OptionsFilter();
        [JsonProperty]
        public OptionsFilter Filter
        {
            get { return _Filter; }
            set { if (value != _Filter) { _Filter = value; OnPropertyChanged(); } }
        }

        #endregion

        public OptionsWarp()
        {
            Import.PropertyChanged += SubOptions_PropertyChanged;
            CTF.PropertyChanged += SubOptions_PropertyChanged;
            Movement.PropertyChanged += SubOptions_PropertyChanged;
            Grids.PropertyChanged += SubOptions_PropertyChanged;
            Picking.PropertyChanged += SubOptions_PropertyChanged;
            Tomo.PropertyChanged += SubOptions_PropertyChanged;
            Export.PropertyChanged += SubOptions_PropertyChanged;
            Tasks.PropertyChanged += SubOptions_PropertyChanged;
            Filter.PropertyChanged += SubOptions_PropertyChanged;
        }

        private void SubOptions_PropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (sender == Import)
                OnPropertyChanged("Import." + e.PropertyName);
            else if (sender == CTF)
                OnPropertyChanged("CTF." + e.PropertyName);
            else if (sender == Movement)
                OnPropertyChanged("Movement." + e.PropertyName);
            else if (sender == Grids)
                OnPropertyChanged("Grids." + e.PropertyName);
            else if (sender == Tomo)
                OnPropertyChanged("Tomo." + e.PropertyName);
            else if (sender == Picking)
                OnPropertyChanged("Picking." + e.PropertyName);
            else if (sender == Export)
                OnPropertyChanged("Export." + e.PropertyName);
            else if (sender == Tasks)
                OnPropertyChanged("Tasks." + e.PropertyName);
            else if (sender == Filter)
                OnPropertyChanged("Filter." + e.PropertyName);
        }

        public void Save(string path)
        {
            XmlTextWriter Writer = new XmlTextWriter(File.Create(path), Encoding.UTF8);
            Writer.Formatting = System.Xml.Formatting.Indented;
            Writer.IndentChar = '\t';
            Writer.Indentation = 1;
            Writer.WriteStartDocument();
            Writer.WriteStartElement("Settings");
            
            WriteToXML(Writer);

            Writer.WriteStartElement("Import");
            Import.WriteToXML(Writer);
            Writer.WriteEndElement();

            Writer.WriteStartElement("CTF");
            CTF.WriteToXML(Writer);
            Writer.WriteEndElement();

            Writer.WriteStartElement("Movement");
            Movement.WriteToXML(Writer);
            Writer.WriteEndElement();

            Writer.WriteStartElement("Grids");
            Grids.WriteToXML(Writer);
            Writer.WriteEndElement();

            Writer.WriteStartElement("Tomo");
            Tomo.WriteToXML(Writer);
            Writer.WriteEndElement();

            Writer.WriteStartElement("Picking");
            Picking.WriteToXML(Writer);
            Writer.WriteEndElement();

            Writer.WriteStartElement("Export");
            Export.WriteToXML(Writer);
            Writer.WriteEndElement();

            Writer.WriteStartElement("Tasks");
            Tasks.WriteToXML(Writer);
            Writer.WriteEndElement();

            Writer.WriteStartElement("Filter");
            Filter.WriteToXML(Writer);
            Writer.WriteEndElement();

            Writer.WriteEndElement();
            Writer.WriteEndDocument();
            Writer.Flush();
            Writer.Close();
        }

        public void Load(string path)
        {
            try
            {
                using (Stream SettingsStream = File.OpenRead(path))
                {
                    XPathDocument Doc = new XPathDocument(SettingsStream);
                    XPathNavigator Reader = Doc.CreateNavigator();
                    Reader.MoveToRoot();

                    Reader.MoveToRoot();
                    Reader.MoveToChild("Settings", "");

                    ReadFromXML(Reader);

                    // Legacy support for when there was PixelSizeX/Y/Angle in Options root level
                    Import.PixelSize = XMLHelper.LoadParamNode(Reader, "PixelSizeX", Import.PixelSize);

                    Import.ReadFromXML(Reader.SelectSingleNode("Import"));
                    CTF.ReadFromXML(Reader.SelectSingleNode("CTF"));
                    Movement.ReadFromXML(Reader.SelectSingleNode("Movement"));
                    Grids.ReadFromXML(Reader.SelectSingleNode("Grids"));
                    Tomo.ReadFromXML(Reader.SelectSingleNode("Tomo"));
                    Picking.ReadFromXML(Reader.SelectSingleNode("Picking"));
                    Export.ReadFromXML(Reader.SelectSingleNode("Export"));
                    Tasks.ReadFromXML(Reader.SelectSingleNode("Tasks"));
                    Filter.ReadFromXML(Reader.SelectSingleNode("Filter"));

                    // Legacy support for when RangeMin and Max were given as fractions of Nyquist
                    if (CTF.RangeMin <= 1)
                    {
                        CTF.RangeMin = Import.BinnedPixelSize * 2 / CTF.RangeMin;
                        CTF.RangeMax = Import.BinnedPixelSize * 2 / CTF.RangeMax;
                    }

                    if (Movement.RangeMin <= 1)
                    {
                        Movement.RangeMin = Import.BinnedPixelSize * 2 / Movement.RangeMin;
                        Movement.RangeMax = Import.BinnedPixelSize * 2 / Movement.RangeMax;
                    }

                    //Import.RecalcBinnedPixelSize();
                }
            }
            catch (FileNotFoundException ex)
            {
                Console.WriteLine("File not found: " + ex.Message);
                Environment.Exit(1);
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred: " + ex.Message);
            }
        }

        #region 2D processing settings creation and adoption

        public ProcessingOptionsBase FillProcessingBase(ProcessingOptionsBase options)
        {
            options.PixelSize = Import.PixelSize;
            options.BinTimes = Import.BinTimes;
            options.EERGroupFrames = Import.ExtensionEER ? Import.EERGroupFrames : 0;
            options.GainPath = Import.CorrectGain ? Import.GainPath : "";
            options.GainHash = Import.CorrectGain ? Import.GainReferenceHash : "";
            options.DefectsPath = Import.CorrectDefects ? Import.DefectsPath : "";
            options.DefectsHash = Import.CorrectDefects ? Import.DefectMapHash : "";
            options.GainFlipX = Import.GainFlipX;
            options.GainFlipY = Import.GainFlipY;
            options.GainTranspose = Import.GainTranspose;

            return options;
        }

        public ProcessingOptionsMovieCTF GetProcessingMovieCTF()
        {
            return (ProcessingOptionsMovieCTF)FillProcessingBase(
            new ProcessingOptionsMovieCTF
            {
                Window = CTF.Window,
                RangeMin = Import.BinnedPixelSize * 2 / CTF.RangeMin,
                RangeMax = Import.BinnedPixelSize * 2 / CTF.RangeMax,
                Voltage = CTF.Voltage,
                Cs = CTF.Cs,
                Cc = CTF.Cc,
                Amplitude = CTF.Amplitude,
                DoPhase = CTF.DoPhase,
                UseMovieSum = CTF.UseMovieSum,
                ZMin = CTF.ZMin,
                ZMax = CTF.ZMax,
                GridDims = new int3(Grids.CTFX, Grids.CTFY, Grids.CTFZ),
                DosePerAngstromFrame = Import.DosePerAngstromFrame
            });
        }

        public void Adopt(ProcessingOptionsMovieCTF options)
        {
            Import.PixelSize = options.PixelSize;
            Import.BinTimes = options.BinTimes;
            Import.GainPath = options.GainPath;
            Import.CorrectGain = !string.IsNullOrEmpty(options.GainPath);
            Import.GainFlipX = options.GainFlipX;
            Import.GainFlipY = options.GainFlipY;
            Import.GainTranspose = options.GainTranspose;
            Import.DefectsPath = options.DefectsPath;
            Import.CorrectDefects = !string.IsNullOrEmpty(options.DefectsPath);

            CTF.Window = options.Window;
            CTF.RangeMin = Import.BinnedPixelSize * 2 / options.RangeMin;
            CTF.RangeMax = Import.BinnedPixelSize * 2 / options.RangeMax;
            CTF.Voltage = options.Voltage;
            CTF.Cs = options.Cs;
            CTF.Cc = options.Cc;
            CTF.Amplitude = options.Amplitude;
            CTF.DoPhase = options.DoPhase;
            CTF.UseMovieSum = options.UseMovieSum;
            CTF.ZMin = options.ZMin;
            CTF.ZMax = options.ZMax;

            Grids.CTFX = options.GridDims.X;
            Grids.CTFY = options.GridDims.Y;
            Grids.CTFZ = options.GridDims.Z;
        }

        public ProcessingOptionsMovieMovement GetProcessingMovieMovement()
        {
            return (ProcessingOptionsMovieMovement)FillProcessingBase(
            new ProcessingOptionsMovieMovement
            {
                RangeMin = Import.BinnedPixelSize * 2 / Movement.RangeMin,
                RangeMax = Import.BinnedPixelSize * 2 / Movement.RangeMax,
                Bfactor = Movement.Bfactor,
                GridDims = new int3(Grids.MovementX, Grids.MovementY, Grids.MovementZ),
                DosePerAngstromFrame = Import.DosePerAngstromFrame
            });
        }

        public void Adopt(ProcessingOptionsMovieMovement options)
        {
            Import.PixelSize = options.PixelSize;
            Import.BinTimes = options.BinTimes;
            Import.GainPath = options.GainPath;
            Import.CorrectGain = !string.IsNullOrEmpty(options.GainPath);
            Import.GainFlipX = options.GainFlipX;
            Import.GainFlipY = options.GainFlipY;
            Import.GainTranspose = options.GainTranspose;
            Import.DefectsPath = options.DefectsPath;
            Import.CorrectDefects = !string.IsNullOrEmpty(options.DefectsPath);
            Movement.RangeMin = Import.BinnedPixelSize * 2 / options.RangeMin;
            Movement.RangeMax = Import.BinnedPixelSize * 2 / options.RangeMax;
            Movement.Bfactor = options.Bfactor;
            Grids.MovementX = options.GridDims.X;
            Grids.MovementY = options.GridDims.Y;
            Grids.MovementZ = options.GridDims.Z;
        }

        public ProcessingOptionsMovieExport GetProcessingMovieExport()
        {
            return (ProcessingOptionsMovieExport)FillProcessingBase(
            new ProcessingOptionsMovieExport
            {
                DosePerAngstromFrame = Import.DosePerAngstromFrame,

                DoAverage = true, //Export.DoAverage,
                DoStack = Export.DoStack,
                DoDeconv = Export.DoDeconvolve,
                DoDenoise = Export.DoDenoise,
                DeconvolutionStrength = Export.DeconvolutionStrength,
                DeconvolutionFalloff = Export.DeconvolutionFalloff,
                StackGroupSize = Export.StackGroupSize,
                SkipFirstN = Export.SkipFirstN,
                SkipLastN = Export.SkipLastN
            });
        }

        public void Adopt(ProcessingOptionsMovieExport options)
        {
            Import.PixelSize = options.PixelSize;
            Import.BinTimes = options.BinTimes;
            Import.GainPath = options.GainPath;
            Import.CorrectGain = !string.IsNullOrEmpty(options.GainPath);
            Import.GainFlipX = options.GainFlipX;
            Import.GainFlipY = options.GainFlipY;
            Import.GainTranspose = options.GainTranspose;
            Import.DefectsPath = options.DefectsPath;
            Import.CorrectDefects = !string.IsNullOrEmpty(options.DefectsPath);

            Import.DosePerAngstromFrame = options.DosePerAngstromFrame;

            Export.DoAverage = options.DoAverage;
            Export.DoStack = options.DoStack;
            Export.DoDeconvolve = options.DoDeconv;
            Export.DeconvolutionStrength = options.DeconvolutionStrength;
            Export.DeconvolutionFalloff = options.DeconvolutionFalloff;
            Export.StackGroupSize = options.StackGroupSize;
            Export.SkipFirstN = options.SkipFirstN;
            Export.SkipLastN = options.SkipLastN;
        }

        public ProcessingOptionsParticleExport GetProcessingParticleExport()
        {
            decimal BinTimes = (decimal)Math.Log((double)(Tasks.Export2DPixel / Import.PixelSize), 2.0);

            var Result = (ProcessingOptionsParticleExport)FillProcessingBase(
            new ProcessingOptionsParticleExport
            {
                DosePerAngstromFrame = Import.DosePerAngstromFrame,

                BoxSize = Picking.BoxSize,
                BoxSizeResample = Picking.DoResample ? Picking.BoxSizeResample : -1,
                Diameter = Picking.Diameter,
                Invert = Picking.DataStyle == "cryo",
                Normalize = true,

                DoAverage = true,
                DoDenoisingPairs = false,
                StackGroupSize = 1,
                SkipFirstN = Export.SkipFirstN,
                SkipLastN = Export.SkipLastN,

                Voltage = CTF.Voltage
            });

            Result.BinTimes = BinTimes;

            return Result;
        }

        public ProcessingOptionsParticleExport GetProcessingParticleExportTask()
        {
            decimal BinTimes = (decimal)Math.Log((double)(Tasks.Export2DPixel / Import.PixelSize), 2.0);

            var Result = (ProcessingOptionsParticleExport)FillProcessingBase(
            new ProcessingOptionsParticleExport
            {
                Suffix = Tasks.OutputSuffix,
                
                DosePerAngstromFrame = Import.DosePerAngstromFrame,

                DoAverage = Tasks.Export2DDoAverages,
                DoDenoisingPairs = Tasks.Export2DDoDenoisingPairs,
                StackGroupSize = Export.StackGroupSize,
                SkipFirstN = Export.SkipFirstN,
                SkipLastN = Export.SkipLastN,

                Voltage = CTF.Voltage
            });

            Result.BinTimes = BinTimes;

            return Result;
        }

        public ProcessingOptionsFullMatch GetProcessingFullMatch()
        {
            decimal BinTimes = (decimal)Math.Log((double)(Tasks.TomoFullReconstructPixel / Import.PixelSize), 2.0);

            var Result = (ProcessingOptionsFullMatch)FillProcessingBase(
            new ProcessingOptionsFullMatch
            {
                DosePerAngstromFrame = Import.DosePerAngstromFrame,
                Voltage = CTF.Voltage,

                TemplatePixel = Tasks.TomoMatchTemplatePixel,
                TemplateDiameter = Tasks.TomoMatchTemplateDiameter,
                TemplateFraction = Tasks.TomoMatchTemplateFraction,

                SubPatchSize = 384,
                Symmetry = Tasks.TomoMatchSymmetry,
                HealpixOrder = (int)Tasks.TomoMatchHealpixOrder,

                Supersample = 5,
                
                NResults = (int)Tasks.TomoMatchNResults,

                Invert = Tasks.InputInvert,
                WhitenSpectrum = Tasks.TomoMatchWhitenSpectrum
            });

            Result.BinTimes = BinTimes;

            return Result;
        }

        public ProcessingOptionsBoxNet GetProcessingBoxNet()
        {
            return (ProcessingOptionsBoxNet)FillProcessingBase(
            new ProcessingOptionsBoxNet
            {
                OverwriteFiles = true,
                OverrideImagePath = "",

                ModelName = Picking.ModelPath,

                PickingInvert = Picking.DataStyle != "cryo",
                ExpectedDiameter = Picking.Diameter,
                MinimumScore = Picking.MinimumScore,
                MinimumMaskDistance = Picking.MinimumMaskDistance,

                ExportParticles = Picking.DoExport,
                ExportBoxSize = Picking.BoxSize,
                ExportInvert = Picking.Invert,
                ExportNormalize = Picking.Normalize
            });
        }
        
        public ProcessingOptionsTardisSegmentMembranes2D GetProcessingTardisSegmentMembranes2D()
        {
            return (ProcessingOptionsTardisSegmentMembranes2D)FillProcessingBase(
                new ProcessingOptionsTardisSegmentMembranes2D()
                {
                    // add options here later if required
                });
        }

        #endregion

        #region Tomo processing settings creation

        public TomoProcessingOptionsBase FillTomoProcessingBase(TomoProcessingOptionsBase options)
        {
            options.Dimensions = new float3((float)Tomo.DimensionsX,
                                    (float)Tomo.DimensionsY,
                                    (float)Tomo.DimensionsZ);

            return (TomoProcessingOptionsBase)FillProcessingBase(options);

        }

        public ProcessingOptionsTomoFullReconstruction GetProcessingTomoFullReconstruction()
        {
            decimal BinTimes = (decimal)Math.Log((double)(Tasks.TomoFullReconstructPixel / Import.PixelSize), 2.0);

            var Result = (ProcessingOptionsTomoFullReconstruction)FillProcessingBase(
            new ProcessingOptionsTomoFullReconstruction
            {
                PixelSize = Import.PixelSize,
                BinTimes = BinTimes,
                EERGroupFrames = Import.ExtensionEER ? Import.EERGroupFrames : 0,
                GainPath = Import.CorrectGain ? Import.GainPath : "",
                GainHash = Import.CorrectGain ? Import.GainReferenceHash : "",
                DefectsPath = Import.CorrectDefects ? Import.DefectsPath : "",
                DefectsHash = Import.CorrectDefects ? Import.DefectMapHash : "",
                GainFlipX = Import.GainFlipX,
                GainFlipY = Import.GainFlipY,
                GainTranspose = Import.GainTranspose,

                Dimensions = new float3((float)Tomo.DimensionsX,
                                        (float)Tomo.DimensionsY,
                                        (float)Tomo.DimensionsZ),

                DoDeconv = Tasks.TomoFullReconstructDoDeconv,
                DeconvStrength = Tasks.TomoFullReconstructDeconvStrength,
                DeconvFalloff = Tasks.TomoFullReconstructDeconvFalloff,
                DeconvHighpass = Tasks.TomoFullReconstructDeconvHighpass,

                Invert = Tasks.InputInvert,
                Normalize = Tasks.InputNormalize,
                SubVolumeSize = 64,
                SubVolumePadding = 2,

                PrepareDenoising = Tasks.TomoFullReconstructPrepareDenoising,
                PrepareDenoisingFrames = Tasks.TomoFullReconstructDenoisingFrames,
                PrepareDenoisingTilts = Tasks.TomoFullReconstructDenoisingTilts,

                KeepOnlyFullVoxels = Tasks.TomoFullReconstructOnlyFullVoxels
            });

            Result.BinTimes = BinTimes;

            return Result;
        }

        public ProcessingOptionsTomoFullMatch GetProcessingTomoFullMatch()
        {
            decimal BinTimes = (decimal)Math.Log((double)(Tasks.TomoFullReconstructPixel / Import.PixelSize), 2.0);

            var Result = (ProcessingOptionsTomoFullMatch)FillProcessingBase(
            new ProcessingOptionsTomoFullMatch
            {
                PixelSize = Import.PixelSize,
                BinTimes = BinTimes,
                EERGroupFrames = Import.ExtensionEER ? Import.EERGroupFrames : 0,
                GainPath = Import.CorrectGain ? Import.GainPath : "",
                GainHash = Import.CorrectGain ? Import.GainReferenceHash : "",
                DefectsPath = Import.CorrectDefects ? Import.DefectsPath : "",
                DefectsHash = Import.CorrectDefects ? Import.DefectMapHash : "",
                GainFlipX = Import.GainFlipX,
                GainFlipY = Import.GainFlipY,
                GainTranspose = Import.GainTranspose,

                Dimensions = new float3((float)Tomo.DimensionsX,
                                        (float)Tomo.DimensionsY,
                                        (float)Tomo.DimensionsZ),
                
                TemplatePixel = Tasks.TomoMatchTemplatePixel,
                TemplateDiameter = Tasks.TomoMatchTemplateDiameter,
                PeakDistance = Tasks.TomoMatchPeakDistance,
                TemplateFraction = Tasks.TomoMatchTemplateFraction,
                
                SubVolumeSize = 192,
                Symmetry = Tasks.TomoMatchSymmetry,
                HealpixOrder = (int)Tasks.TomoMatchHealpixOrder,
                TiltRange = -1,
                BatchAngles = Tasks.TomoMatchBatchAngles,

                Supersample = 1,

                MaxMissingTilts = 2,
                NResults = (int)Tasks.TomoMatchNResults,

                ReuseCorrVolumes = Tasks.ReuseCorrVolumes,

                WhitenSpectrum = Tasks.TomoMatchWhitenSpectrum,
                Lowpass = 1.0M,
                LowpassSigma = 0M
            });

            Result.BinTimes = BinTimes;

            return Result;
        }

        public ProcessingOptionsTomoSubReconstruction GetProcessingTomoSubReconstruction()
        {
            decimal BinTimes = (decimal)Math.Log((double)(Tasks.TomoSubReconstructPixel / Import.PixelSize), 2.0);

            var Result = (ProcessingOptionsTomoSubReconstruction)FillProcessingBase(
            new ProcessingOptionsTomoSubReconstruction
            {
                PixelSize = Import.PixelSize,
                BinTimes = BinTimes,
                EERGroupFrames = Import.ExtensionEER ? Import.EERGroupFrames : 0,
                GainPath = Import.CorrectGain ? Import.GainPath : "",
                GainHash = Import.CorrectGain ? Import.GainReferenceHash : "",
                DefectsPath = Import.CorrectDefects ? Import.DefectsPath : "",
                DefectsHash = Import.CorrectDefects ? Import.DefectMapHash : "",
                GainFlipX = Import.GainFlipX,
                GainFlipY = Import.GainFlipY,
                GainTranspose = Import.GainTranspose,

                Dimensions = new float3((float)Tomo.DimensionsX,
                                        (float)Tomo.DimensionsY,
                                        (float)Tomo.DimensionsZ),

                Suffix = "",

                BoxSize = (int)Tasks.TomoSubReconstructBox,
                ParticleDiameter = (int)Tasks.TomoSubReconstructDiameter,

                Invert = Tasks.InputInvert,
                NormalizeInput = Tasks.InputNormalize,
                NormalizeOutput = Tasks.OutputNormalize,

                PrerotateParticles = Tasks.TomoSubReconstructPrerotated,
                DoLimitDose = Tasks.TomoSubReconstructDoLimitDose,
                NTilts = Tasks.TomoSubReconstructNTilts,

                MakeSparse = Tasks.TomoSubReconstructMakeSparse,

                UseCPU = Tasks.UseCPU
            });

            Result.BinTimes = BinTimes;

            return Result;
        }

        #endregion

        public ProcessingStatus GetMovieProcessingStatus(Movie movie, bool considerFilter = true)
        {
            ProcessingOptionsMovieCTF OptionsCTF = GetProcessingMovieCTF();
            ProcessingOptionsMovieMovement OptionsMovement = GetProcessingMovieMovement();
            ProcessingOptionsBoxNet OptionsBoxNet = GetProcessingBoxNet();
            ProcessingOptionsMovieExport OptionsExport = GetProcessingMovieExport(); 

            bool DoCTF = ProcessCTF;
            bool DoMovement = ProcessMovement && movie.GetType() == typeof(Movie);
            bool DoBoxNet = ProcessPicking && movie.GetType() == typeof(Movie);
            bool DoExport = (OptionsExport.DoAverage || OptionsExport.DoStack || OptionsExport.DoDeconv || OptionsExport.DoDenoise) && (movie is Movie);

            ProcessingStatus Status = ProcessingStatus.Processed;

            if (movie.UnselectManual != null && (bool)movie.UnselectManual)
            {
                Status = ProcessingStatus.LeaveOut;
            }
            else if (movie.OptionsCTF == null && movie.OptionsMovement == null && movie.OptionsMovieExport == null)
            {
                Status = ProcessingStatus.Unprocessed;
            }
            else
            {
                if (DoCTF && (movie.OptionsCTF == null || movie.OptionsCTF != OptionsCTF))
                    Status = ProcessingStatus.Outdated;
                else if (DoMovement && (movie.OptionsMovement == null || movie.OptionsMovement != OptionsMovement))
                    Status = ProcessingStatus.Outdated;
                else if (DoBoxNet && (movie.OptionsBoxNet == null || movie.OptionsBoxNet != OptionsBoxNet))
                    Status = ProcessingStatus.Outdated;
                else if (DoExport && (movie.OptionsMovieExport == null || movie.OptionsMovieExport != OptionsExport))
                    Status = ProcessingStatus.Outdated;
            }

            if (Status == ProcessingStatus.Processed && movie.UnselectFilter && movie.UnselectManual == null && considerFilter)
                Status = ProcessingStatus.FilteredOut;

            return Status;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class OptionsImport : WarpBase
    {
        private string _DataFolder = "";
        [WarpSerializable]
        [JsonProperty]
        public string DataFolder
        {
            get { return _DataFolder; }
            set
            {
                if (value != _DataFolder)
                {
                    _DataFolder = value;
                    OnPropertyChanged();
                }
            }
        }

        private string _ProcessingFolder = "";
        [WarpSerializable]
        [JsonProperty]
        public string ProcessingFolder
        {
            get { return _ProcessingFolder; }
            set { if (value != _ProcessingFolder) { _ProcessingFolder = value; OnPropertyChanged(); } }
        }

        public string ProcessingOrDataFolder => string.IsNullOrEmpty(ProcessingFolder) ? DataFolder : ProcessingFolder;

        private bool _DoRecursiveSearch = false;
        [WarpSerializable]
        [JsonProperty]
        public bool DoRecursiveSearch
        {
            get { return _DoRecursiveSearch; }
            set { if (value != _DoRecursiveSearch) { _DoRecursiveSearch = value; OnPropertyChanged(); } }
        }

        private string _Extension = "*.mrc";
        [WarpSerializable]
        [JsonProperty]
        public string Extension
        {
            get { return _Extension; }
            set
            {
                if (value != _Extension)
                {
                    _Extension = value;
                    OnPropertyChanged();

                    OnPropertyChanged("ExtensionMRC");
                    OnPropertyChanged("ExtensionMRCS");
                    OnPropertyChanged("ExtensionEM");
                    OnPropertyChanged("ExtensionTIFF");
                    OnPropertyChanged("ExtensionTIFFF");
                    OnPropertyChanged("ExtensionEER");
                    OnPropertyChanged("ExtensionDAT");
                    OnPropertyChanged("ExtensionTomoSTAR");
                }
            }
        }
        
        public bool ExtensionMRC
        {
            get { return Extension == "*.mrc"; }
            set
            {
                if (value != (Extension == "*.mrc"))
                {
                    if (value)
                        Extension = "*.mrc";
                    OnPropertyChanged();
                }
            }
        }
        
        public bool ExtensionMRCS
        {
            get { return Extension == "*.mrcs"; }
            set
            {
                if (value != (Extension == "*.mrcs"))
                {
                    if (value)
                        Extension = "*.mrcs";
                    OnPropertyChanged();
                }
            }
        }
        
        public bool ExtensionEM
        {
            get { return Extension == "*.em"; }
            set
            {
                if (value != (Extension == "*.em"))
                {
                    if (value)
                        Extension = "*.em";
                    OnPropertyChanged();
                }
            }
        }
        
        public bool ExtensionTIFF
        {
            get { return Extension == "*.tif"; }
            set
            {
                if (value != (Extension == "*.tif"))
                {
                    if (value)
                        Extension = "*.tif";
                    OnPropertyChanged();
                }
            }
        }

        public bool ExtensionTIFFF
        {
            get { return Extension == "*.tiff"; }
            set
            {
                if (value != (Extension == "*.tiff"))
                {
                    if (value)
                        Extension = "*.tiff";
                    OnPropertyChanged();
                }
            }
        }

        public bool ExtensionEER
        {
            get { return Extension == "*.eer"; }
            set
            {
                if (value != (Extension == "*.eer"))
                {
                    if (value)
                        Extension = "*.eer";
                    OnPropertyChanged();
                }
            }
        }

        public bool ExtensionTomoSTAR
        {
            get { return Extension == "*.tomostar"; }
            set
            {
                if (value != (Extension == "*.tomostar"))
                {
                    if (value)
                        Extension = "*.tomostar";
                    OnPropertyChanged();
                }
            }
        }
        
        public bool ExtensionDAT
        {
            get { return Extension == "*.dat"; }
            set
            {
                if (value != (Extension == "*.dat"))
                {
                    if (value)
                        Extension = "*.dat";
                    OnPropertyChanged();
                }
            }
        }

        private int _HeaderlessWidth = 7676;
        [WarpSerializable]
        [JsonProperty]
        public int HeaderlessWidth
        {
            get { return _HeaderlessWidth; }
            set { if (value != _HeaderlessWidth) { _HeaderlessWidth = value; OnPropertyChanged(); } }
        }

        private int _HeaderlessHeight = 7420;
        [WarpSerializable]
        [JsonProperty]
        public int HeaderlessHeight
        {
            get { return _HeaderlessHeight; }
            set { if (value != _HeaderlessHeight) { _HeaderlessHeight = value; OnPropertyChanged(); } }
        }

        private string _HeaderlessType = "int8";
        [WarpSerializable]
        [JsonProperty]
        public string HeaderlessType
        {
            get { return _HeaderlessType; }
            set { if (value != _HeaderlessType) { _HeaderlessType = value; OnPropertyChanged(); } }
        }

        private long _HeaderlessOffset = 0;
        [WarpSerializable]
        [JsonProperty]
        public long HeaderlessOffset
        {
            get { return _HeaderlessOffset; }
            set { if (value != _HeaderlessOffset) { _HeaderlessOffset = value; OnPropertyChanged(); } }
        }

        private decimal _PixelSize = 1.35M;
        [WarpSerializable]
        [JsonProperty]
        public decimal PixelSize
        {
            get { return _PixelSize; }
            set
            {
                if (value != _PixelSize)
                {
                    _PixelSize = value;
                    OnPropertyChanged();
                    RecalcBinnedPixelSize();
                }
            }
        }

        private decimal _BinTimes = 0;
        [WarpSerializable]
        [JsonProperty]
        public decimal BinTimes
        {
            get { return _BinTimes; }
            set
            {
                if (value != _BinTimes)
                {
                    _BinTimes = value;
                    OnPropertyChanged();
                    RecalcBinnedPixelSize();
                }
            }
        }

        private decimal _BinnedPixelSize = 1M;
        public decimal BinnedPixelSize
        {
            get { return _BinnedPixelSize; }
            set { if (value != _BinnedPixelSize) { _BinnedPixelSize = value; OnPropertyChanged(); } }
        }

        private void RecalcBinnedPixelSize()
        {
            BinnedPixelSize = PixelSize * (decimal)Math.Pow(2.0, (double)BinTimes);
        }

        private string _GainPath = "";
        [WarpSerializable]
        [JsonProperty]
        public string GainPath
        {
            get { return _GainPath; }
            set
            {
                if (value != _GainPath)
                {
                    _GainPath = value;
                    OnPropertyChanged();
                }
            }
        }

        private string _DefectsPath = "";
        [WarpSerializable]
        [JsonProperty]
        public string DefectsPath
        {
            get { return _DefectsPath; }
            set { if (value != _DefectsPath) { _DefectsPath = value; OnPropertyChanged(); } }
        }

        private bool _GainFlipX = false;
        [WarpSerializable]
        [JsonProperty]
        public bool GainFlipX
        {
            get { return _GainFlipX; }
            set { if (value != _GainFlipX) { _GainFlipX = value; OnPropertyChanged(); } }
        }

        private bool _GainFlipY = false;
        [WarpSerializable]
        [JsonProperty]
        public bool GainFlipY
        {
            get { return _GainFlipY; }
            set { if (value != _GainFlipY) { _GainFlipY = value; OnPropertyChanged(); } }
        }

        private bool _GainTranspose = false;
        [WarpSerializable]
        [JsonProperty]
        public bool GainTranspose
        {
            get { return _GainTranspose; }
            set { if (value != _GainTranspose) { _GainTranspose = value; OnPropertyChanged(); } }
        }

        private bool _CorrectGain = false;
        [WarpSerializable]
        [JsonProperty]
        public bool CorrectGain
        {
            get { return _CorrectGain; }
            set
            {
                if (value != _CorrectGain)
                {
                    _CorrectGain = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _CorrectDefects = false;
        [WarpSerializable]
        [JsonProperty]
        public bool CorrectDefects
        {
            get { return _CorrectDefects; }
            set { if (value != _CorrectDefects) { _CorrectDefects = value; OnPropertyChanged(); } }
        }

        private string _GainReferenceHash = "";
        public string GainReferenceHash
        {
            get { return _GainReferenceHash; }
            set { if (value != _GainReferenceHash) { _GainReferenceHash = value; OnPropertyChanged(); } }
        }

        private string _DefectMapHash = "";
        public string DefectMapHash
        {
            get { return _DefectMapHash; }
            set { if (value != _DefectMapHash) { _DefectMapHash = value; OnPropertyChanged(); } }
        }

        private decimal _DosePerAngstromFrame = 0;
        [WarpSerializable]
        [JsonProperty]
        public decimal DosePerAngstromFrame
        {
            get { return _DosePerAngstromFrame; }
            set { if (value != _DosePerAngstromFrame) { _DosePerAngstromFrame = value; OnPropertyChanged(); } }
        }

        private int _EERGroupFrames = 10;
        [WarpSerializable]
        [JsonProperty]
        public int EERGroupFrames
        {
            get { return _EERGroupFrames; }
            set { if (value != _EERGroupFrames) { _EERGroupFrames = value; OnPropertyChanged(); } }
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class OptionsCTF : WarpBase
    {
        private int _Window = 512;
        [WarpSerializable]
        [JsonProperty]
        public int Window
        {
            get { return _Window; }
            set
            {
                if (value != _Window)
                {
                    _Window = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _RangeMin = 30M;
        [WarpSerializable]
        [JsonProperty]
        public decimal RangeMin
        {
            get { return _RangeMin; }
            set
            {
                if (value != _RangeMin)
                {
                    _RangeMin = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _RangeMax = 4M;
        [WarpSerializable]
        [JsonProperty]
        public decimal RangeMax
        {
            get { return _RangeMax; }
            set
            {
                if (value != _RangeMax)
                {
                    _RangeMax = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _MinQuality = 0.8M;
        [WarpSerializable]
        [JsonProperty]
        public decimal MinQuality
        {
            get { return _MinQuality; }
            set
            {
                if (value != _MinQuality)
                {
                    _MinQuality = value;
                    OnPropertyChanged();
                }
            }
        }

        private int _Voltage = 300;
        [WarpSerializable]
        [JsonProperty]
        public int Voltage
        {
            get { return _Voltage; }
            set
            {
                if (value != _Voltage)
                {
                    _Voltage = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _Cs = 2.7M;
        [WarpSerializable]
        [JsonProperty]
        public decimal Cs
        {
            get { return _Cs; }
            set
            {
                if (value != _Cs)
                {
                    _Cs = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _Cc = 2.7M;
        [WarpSerializable]
        public decimal Cc
        {
            get { return _Cc; }
            set { if (value != _Cc) { _Cc = value; OnPropertyChanged(); } }
        }

        private decimal _Amplitude = 0.07M;
        [WarpSerializable]
        [JsonProperty]
        public decimal Amplitude
        {
            get { return _Amplitude; }
            set
            {
                if (value != _Amplitude)
                {
                    _Amplitude = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _IllAperture = 30;
        [WarpSerializable]
        public decimal IllAperture
        {
            get { return _IllAperture; }
            set { if (value != _IllAperture) { _IllAperture = value; OnPropertyChanged(); } }
        }

        private decimal _DeltaE = 0.7M;
        [WarpSerializable]
        public decimal DeltaE
        {
            get { return _DeltaE; }
            set { if (value != _DeltaE) { _DeltaE = value; OnPropertyChanged(); } }
        }

        private decimal _Thickness = 0;
        [WarpSerializable]
        public decimal Thickness
        {
            get { return _Thickness; }
            set { if (value != _Thickness) { _Thickness = value; OnPropertyChanged(); } }
        }

        private bool _DoPhase = true;
        [WarpSerializable]
        [JsonProperty]
        public bool DoPhase
        {
            get { return _DoPhase; }
            set { if (value != _DoPhase) { _DoPhase = value; OnPropertyChanged(); } }
        }

        //private bool _DoIce = false;
        //[WarpSerializable]
        //public bool DoIce
        //{
        //    get { return _DoIce; }
        //    set { if (value != _DoIce) { _DoIce = value; OnPropertyChanged(); } }
        //}

        private bool _UseMovieSum = false;
        [WarpSerializable]
        [JsonProperty]
        public bool UseMovieSum
        {
            get { return _UseMovieSum; }
            set { if (value != _UseMovieSum) { _UseMovieSum = value; OnPropertyChanged(); } }
        }

        private decimal _ZMin = 0M;
        [WarpSerializable]
        [JsonProperty]
        public decimal ZMin
        {
            get { return _ZMin; }
            set
            {
                if (value != _ZMin)
                {
                    _ZMin = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _ZMax = 5M;
        [WarpSerializable]
        [JsonProperty]
        public decimal ZMax
        {
            get { return _ZMax; }
            set
            {
                if (value != _ZMax)
                {
                    _ZMax = value;
                    OnPropertyChanged();
                }
            }
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class OptionsMovement : WarpBase
    {
        private decimal _RangeMin = 500M;
        [WarpSerializable]
        [JsonProperty]
        public decimal RangeMin
        {
            get { return _RangeMin; }
            set
            {
                if (value != _RangeMin)
                {
                    _RangeMin = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _RangeMax = 8M;
        [WarpSerializable]
        [JsonProperty]
        public decimal RangeMax
        {
            get { return _RangeMax; }
            set
            {
                if (value != _RangeMax)
                {
                    _RangeMax = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _Bfactor = -500;
        [WarpSerializable]
        [JsonProperty]
        public decimal Bfactor
        {
            get { return _Bfactor; }
            set { if (value != _Bfactor) { _Bfactor = value; OnPropertyChanged(); } }
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class OptionsGrids : WarpBase
    {
        private int _CTFX = 5;
        [WarpSerializable]
        [JsonProperty]
        public int CTFX
        {
            get { return _CTFX; }
            set { if (value != _CTFX) { _CTFX = value; OnPropertyChanged(); } }
        }

        private int _CTFY = 5;
        [WarpSerializable]
        [JsonProperty]
        public int CTFY
        {
            get { return _CTFY; }
            set { if (value != _CTFY) { _CTFY = value; OnPropertyChanged(); } }
        }

        private int _CTFZ = 1;
        [WarpSerializable]
        [JsonProperty]
        public int CTFZ
        {
            get { return _CTFZ; }
            set { if (value != _CTFZ) { _CTFZ = value; OnPropertyChanged(); } }
        }

        private int _MovementX = 5;
        [WarpSerializable]
        [JsonProperty]
        public int MovementX
        {
            get { return _MovementX; }
            set { if (value != _MovementX) { _MovementX = value; OnPropertyChanged(); } }
        }

        private int _MovementY = 5;
        [WarpSerializable]
        [JsonProperty]
        public int MovementY
        {
            get { return _MovementY; }
            set { if (value != _MovementY) { _MovementY = value; OnPropertyChanged(); } }
        }

        private int _MovementZ = 20;
        [WarpSerializable]
        [JsonProperty]
        public int MovementZ
        {
            get { return _MovementZ; }
            set { if (value != _MovementZ) { _MovementZ = value; OnPropertyChanged(); } }
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class OptionsPicking : WarpBase
    {
        private string _ModelPath = "";
        [WarpSerializable]
        [JsonProperty]
        public string ModelPath
        {
            get { return _ModelPath; }
            set { if (value != _ModelPath) { _ModelPath = value; OnPropertyChanged(); } }
        }

        private string _DataStyle = "cryo";
        [WarpSerializable]
        [JsonProperty]
        public string DataStyle
        {
            get { return _DataStyle; }
            set { if (value != _DataStyle) { _DataStyle = value; OnPropertyChanged(); } }
        }

        private int _Diameter = 200;
        [WarpSerializable]
        [JsonProperty]
        public int Diameter
        {
            get { return _Diameter; }
            set { if (value != _Diameter) { _Diameter = value; OnPropertyChanged(); } }
        }

        private decimal _MinimumScore = 0.95M;
        [WarpSerializable]
        [JsonProperty]
        public decimal MinimumScore
        {
            get { return _MinimumScore; }
            set { if (value != _MinimumScore) { _MinimumScore = value; OnPropertyChanged(); } }
        }

        private decimal _MinimumMaskDistance = 0;
        [WarpSerializable]
        [JsonProperty]
        public decimal MinimumMaskDistance
        {
            get { return _MinimumMaskDistance; }
            set { if (value != _MinimumMaskDistance) { _MinimumMaskDistance = value; OnPropertyChanged(); } }
        }

        private bool _DoExport = false;
        [WarpSerializable]
        [JsonProperty]
        public bool DoExport
        {
            get { return _DoExport; }
            set { if (value != _DoExport) { _DoExport = value; OnPropertyChanged(); } }
        }

        private int _BoxSize = 128;
        [WarpSerializable]
        [JsonProperty]
        public int BoxSize
        {
            get { return _BoxSize; }
            set { if (value != _BoxSize) { _BoxSize = value; OnPropertyChanged(); } }
        }

        private bool _DoResample = false;
        [WarpSerializable]
        [JsonProperty]
        public bool DoResample
        {
            get { return _DoResample; }
            set { if (value != _DoResample) { _DoResample = value; OnPropertyChanged(); } }
        }

        private int _BoxSizeResample = 128;
        [WarpSerializable]
        [JsonProperty]
        public int BoxSizeResample
        {
            get { return _BoxSizeResample; }
            set { if (value != _BoxSizeResample) { _BoxSizeResample = value; OnPropertyChanged(); } }
        }

        private int _BoxMiniSize = -1;
        [WarpSerializable]
        [JsonProperty]
        public int BoxMiniSize
        {
            get { return _BoxMiniSize; }
            set { if (value != _BoxMiniSize) { _BoxMiniSize = value; OnPropertyChanged(); } }
        }

        private bool _Invert = true;
        [WarpSerializable]
        [JsonProperty]
        public bool Invert
        {
            get { return _Invert; }
            set { if (value != _Invert) { _Invert = value; OnPropertyChanged(); } }
        }

        private bool _Normalize = true;
        [WarpSerializable]
        [JsonProperty]
        public bool Normalize
        {
            get { return _Normalize; }
            set { if (value != _Normalize) { _Normalize = value; OnPropertyChanged(); } }
        }

        private bool _DoRunningWindow = true;
        [WarpSerializable]
        [JsonProperty]
        public bool DoRunningWindow
        {
            get { return _DoRunningWindow; }
            set { if (value != _DoRunningWindow) { _DoRunningWindow = value; OnPropertyChanged(); } }
        }

        private int _RunningWindowLength = 10000;
        [WarpSerializable]
        [JsonProperty]
        public int RunningWindowLength
        {
            get { return _RunningWindowLength; }
            set { if (value != _RunningWindowLength) { _RunningWindowLength = value; OnPropertyChanged(); } }
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class OptionsTomo : WarpBase
    {
        private decimal _DimensionsX = 3712;
        [WarpSerializable]
        [JsonProperty]
        public decimal DimensionsX
        {
            get { return _DimensionsX; }
            set { if (value != _DimensionsX) { _DimensionsX = value; OnPropertyChanged(); } }
        }

        private decimal _DimensionsY = 3712;
        [WarpSerializable]
        [JsonProperty]
        public decimal DimensionsY
        {
            get { return _DimensionsY; }
            set { if (value != _DimensionsY) { _DimensionsY = value; OnPropertyChanged(); } }
        }

        private decimal _DimensionsZ = 1400;
        [WarpSerializable]
        [JsonProperty]
        public decimal DimensionsZ
        {
            get { return _DimensionsZ; }
            set { if (value != _DimensionsZ) { _DimensionsZ = value; OnPropertyChanged(); } }
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class OptionsExport : WarpBase
    {
        private bool _DoAverage = true;
        [WarpSerializable]
        [JsonProperty]
        public bool DoAverage
        {
            get { return _DoAverage; }
            set
            {
                if (value != _DoAverage)
                {
                    _DoAverage = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _DoStack = false;
        [WarpSerializable]
        [JsonProperty]
        public bool DoStack
        {
            get { return _DoStack; }
            set
            {
                if (value != _DoStack)
                {
                    _DoStack = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _DoDeconvolve = false;
        [WarpSerializable]
        public bool DoDeconvolve
        {
            get { return _DoDeconvolve; }
            set { if (value != _DoDeconvolve) { _DoDeconvolve = value; OnPropertyChanged(); } }
        }

        private bool _DoDenoise = false;
        [WarpSerializable]
        public bool DoDenoise
        {
            get { return _DoDenoise; }
            set { if (value != _DoDenoise) { _DoDenoise = value; OnPropertyChanged(); } }
        }

        private decimal _DeconvolutionStrength = 1;
        [WarpSerializable]
        public decimal DeconvolutionStrength
        {
            get { return _DeconvolutionStrength; }
            set { if (value != _DeconvolutionStrength) { _DeconvolutionStrength = value; OnPropertyChanged(); } }
        }

        private decimal _DeconvolutionFalloff = 1;
        [WarpSerializable]
        public decimal DeconvolutionFalloff
        {
            get { return _DeconvolutionFalloff; }
            set { if (value != _DeconvolutionFalloff) { _DeconvolutionFalloff = value; OnPropertyChanged(); } }
        }

        private int _StackGroupSize = 1;
        [WarpSerializable]
        [JsonProperty]
        public int StackGroupSize
        {
            get { return _StackGroupSize; }
            set
            {
                if (value != _StackGroupSize)
                {
                    _StackGroupSize = value;
                    OnPropertyChanged();
                }
            }
        }

        private int _SkipFirstN = 0;
        [WarpSerializable]
        [JsonProperty]
        public int SkipFirstN
        {
            get { return _SkipFirstN; }
            set { if (value != _SkipFirstN) { _SkipFirstN = value; OnPropertyChanged(); } }
        }

        private int _SkipLastN = 0;
        [WarpSerializable]
        [JsonProperty]
        public int SkipLastN
        {
            get { return _SkipLastN; }
            set { if (value != _SkipLastN) { _SkipLastN = value; OnPropertyChanged(); } }
        }
    }

    public class OptionsTasks : WarpBase
    {
        #region Common

        private bool _UseRelativePaths = true;
        [WarpSerializable]
        public bool UseRelativePaths
        {
            get { return _UseRelativePaths; }
            set { if (value != _UseRelativePaths) { _UseRelativePaths = value; OnPropertyChanged(); } }
        }

        private bool _UseCPU = false;
        [WarpSerializable]
        public bool UseCPU
        {
            get { return _UseCPU; }
            set { if (value != _UseCPU) { _UseCPU = value; OnPropertyChanged(); } }
        }

        private bool _IncludeFilteredOut = false;
        [WarpSerializable]
        public bool IncludeFilteredOut
        {
            get { return _IncludeFilteredOut; }
            set { if (value != _IncludeFilteredOut) { _IncludeFilteredOut = value; OnPropertyChanged(); } }
        }

        private bool _IncludeUnselected = false;
        [WarpSerializable]
        public bool IncludeUnselected
        {
            get { return _IncludeUnselected; }
            set { if (value != _IncludeUnselected) { _IncludeUnselected = value; OnPropertyChanged(); } }
        }

        private bool _InputOnePerItem = false;
        [WarpSerializable]
        public bool InputOnePerItem
        {
            get { return _InputOnePerItem; }
            set { if (value != _InputOnePerItem) { _InputOnePerItem = value; OnPropertyChanged(); } }
        }

        private decimal _InputPixelSize = 1;
        [WarpSerializable]
        public decimal InputPixelSize
        {
            get { return _InputPixelSize; }
            set { if (value != _InputPixelSize) { _InputPixelSize = value; OnPropertyChanged(); } }
        }

        private decimal _InputShiftPixelSize = 1;
        [WarpSerializable]
        public decimal InputShiftPixelSize
        {
            get { return _InputShiftPixelSize; }
            set { if (value != _InputShiftPixelSize) { _InputShiftPixelSize = value; OnPropertyChanged(); } }
        }

        private decimal _OutputPixelSize = 1;
        [WarpSerializable]
        public decimal OutputPixelSize
        {
            get { return _OutputPixelSize; }
            set { if (value != _OutputPixelSize) { _OutputPixelSize = value; OnPropertyChanged(); } }
        }

        private string _OutputSuffix = "";
        [WarpSerializable]
        public string OutputSuffix
        {
            get { return _OutputSuffix; }
            set { if (value != _OutputSuffix) { _OutputSuffix = value; OnPropertyChanged(); } }
        }

        private bool _InputInvert = true;
        [WarpSerializable]
        public bool InputInvert
        {
            get { return _InputInvert; }
            set { if (value != _InputInvert) { _InputInvert = value; OnPropertyChanged(); } }
        }

        private bool _InputNormalize = true;
        [WarpSerializable]
        public bool InputNormalize
        {
            get { return _InputNormalize; }
            set { if (value != _InputNormalize) { _InputNormalize = value; OnPropertyChanged(); } }
        }

        private bool _InputFlipX = false;
        [WarpSerializable]
        public bool InputFlipX
        {
            get { return _InputFlipX; }
            set { if (value != _InputFlipX) { _InputFlipX = value; OnPropertyChanged(); } }
        }

        private bool _InputFlipY = false;
        [WarpSerializable]
        public bool InputFlipY
        {
            get { return _InputFlipY; }
            set { if (value != _InputFlipY) { _InputFlipY = value; OnPropertyChanged(); } }
        }

        private bool _OutputNormalize = true;
        [WarpSerializable]
        public bool OutputNormalize
        {
            get { return _OutputNormalize; }
            set { if (value != _OutputNormalize) { _OutputNormalize = value; OnPropertyChanged(); } }
        }

        #endregion

        #region 2D

        private bool _MicListMakePolishing = false;
        [WarpSerializable]
        public bool MicListMakePolishing
        {
            get { return _MicListMakePolishing; }
            set { if (value != _MicListMakePolishing) { _MicListMakePolishing = value; OnPropertyChanged(); } }
        }

        private bool _AdjustDefocusSkipExcluded = true;
        [WarpSerializable]
        public bool AdjustDefocusSkipExcluded
        {
            get { return _AdjustDefocusSkipExcluded; }
            set { if (value != _AdjustDefocusSkipExcluded) { _AdjustDefocusSkipExcluded = value; OnPropertyChanged(); } }
        }

        private bool _AdjustDefocusDeleteExcluded = false;
        [WarpSerializable]
        public bool AdjustDefocusDeleteExcluded
        {
            get { return _AdjustDefocusDeleteExcluded; }
            set { if (value != _AdjustDefocusDeleteExcluded) { _AdjustDefocusDeleteExcluded = value; OnPropertyChanged(); } }
        }

        private decimal _Export2DPixel = 1M;
        [WarpSerializable]
        public decimal Export2DPixel
        {
            get { return _Export2DPixel; }
            set { if (value != _Export2DPixel) { _Export2DPixel = value; OnPropertyChanged(); } }
        }

        private decimal _Export2DBoxSize = 128;
        [WarpSerializable]
        public decimal Export2DBoxSize
        {
            get { return _Export2DBoxSize; }
            set { if (value != _Export2DBoxSize) { _Export2DBoxSize = value; OnPropertyChanged(); } }
        }

        private decimal _Export2DParticleDiameter = 100;
        [WarpSerializable]
        public decimal Export2DParticleDiameter
        {
            get { return _Export2DParticleDiameter; }
            set { if (value != _Export2DParticleDiameter) { _Export2DParticleDiameter = value; OnPropertyChanged(); } }
        }

        private bool _Export2DDoAverages = true;
        [WarpSerializable]
        public bool Export2DDoAverages
        {
            get { return _Export2DDoAverages; }
            set { if (value != _Export2DDoAverages) { _Export2DDoAverages = value; OnPropertyChanged(); } }
        }

        private bool _Export2DDoMovies = false;
        [WarpSerializable]
        public bool Export2DDoMovies
        {
            get { return _Export2DDoMovies; }
            set { if (value != _Export2DDoMovies) { _Export2DDoMovies = value; OnPropertyChanged(); } }
        }

        private bool _Export2DDoOnlyTable = false;
        [WarpSerializable]
        public bool Export2DDoOnlyTable
        {
            get { return _Export2DDoOnlyTable; }
            set { if (value != _Export2DDoOnlyTable) { _Export2DDoOnlyTable = value; OnPropertyChanged(); } }
        }

        private bool _Export2DDoDenoisingPairs = false;
        [WarpSerializable]
        public bool Export2DDoDenoisingPairs
        {
            get { return _Export2DDoDenoisingPairs; }
            set { if (value != _Export2DDoDenoisingPairs) { _Export2DDoDenoisingPairs = value; OnPropertyChanged(); } }
        }

        private bool _Export2DPreflip = false;
        [WarpSerializable]
        public bool Export2DPreflip
        {
            get { return _Export2DPreflip; }
            set { if (value != _Export2DPreflip) { _Export2DPreflip = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Tomo

        #region Full reconstruction

        private decimal _TomoFullReconstructPixel = 1M;
        [WarpSerializable]
        public decimal TomoFullReconstructPixel
        {
            get { return _TomoFullReconstructPixel; }
            set { if (value != _TomoFullReconstructPixel) { _TomoFullReconstructPixel = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructDoDeconv = false;
        [WarpSerializable]
        public bool TomoFullReconstructDoDeconv
        {
            get { return _TomoFullReconstructDoDeconv; }
            set { if (value != _TomoFullReconstructDoDeconv) { _TomoFullReconstructDoDeconv = value; OnPropertyChanged(); } }
        }

        private decimal _TomoFullReconstructDeconvStrength = 1M;
        [WarpSerializable]
        public decimal TomoFullReconstructDeconvStrength
        {
            get { return _TomoFullReconstructDeconvStrength; }
            set { if (value != _TomoFullReconstructDeconvStrength) { _TomoFullReconstructDeconvStrength = value; OnPropertyChanged(); } }
        }

        private decimal _TomoFullReconstructDeconvFalloff = 1M;
        [WarpSerializable]
        public decimal TomoFullReconstructDeconvFalloff
        {
            get { return _TomoFullReconstructDeconvFalloff; }
            set { if (value != _TomoFullReconstructDeconvFalloff) { _TomoFullReconstructDeconvFalloff = value; OnPropertyChanged(); } }
        }

        private decimal _TomoFullReconstructDeconvHighpass = 300;
        [WarpSerializable]
        public decimal TomoFullReconstructDeconvHighpass
        {
            get { return _TomoFullReconstructDeconvHighpass; }
            set { if (value != _TomoFullReconstructDeconvHighpass) { _TomoFullReconstructDeconvHighpass = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructInvert = true;
        [WarpSerializable]
        public bool TomoFullReconstructInvert
        {
            get { return _TomoFullReconstructInvert; }
            set { if (value != _TomoFullReconstructInvert) { _TomoFullReconstructInvert = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructNormalize = true;
        [WarpSerializable]
        public bool TomoFullReconstructNormalize
        {
            get { return _TomoFullReconstructNormalize; }
            set { if (value != _TomoFullReconstructNormalize) { _TomoFullReconstructNormalize = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructPrepareDenoising = false;
        [WarpSerializable]
        public bool TomoFullReconstructPrepareDenoising
        {
            get { return _TomoFullReconstructPrepareDenoising; }
            set { if (value != _TomoFullReconstructPrepareDenoising) { _TomoFullReconstructPrepareDenoising = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructDenoisingFrames = true;
        [WarpSerializable]
        public bool TomoFullReconstructDenoisingFrames
        {
            get { return _TomoFullReconstructDenoisingFrames; }
            set { if (value != _TomoFullReconstructDenoisingFrames) { _TomoFullReconstructDenoisingFrames = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructDenoisingTilts = false;
        [WarpSerializable]
        public bool TomoFullReconstructDenoisingTilts
        {
            get { return _TomoFullReconstructDenoisingTilts; }
            set { if (value != _TomoFullReconstructDenoisingTilts) { _TomoFullReconstructDenoisingTilts = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructOnlyFullVoxels = false;
        [WarpSerializable]
        public bool TomoFullReconstructOnlyFullVoxels
        {
            get { return _TomoFullReconstructOnlyFullVoxels; }
            set { if (value != _TomoFullReconstructOnlyFullVoxels) { _TomoFullReconstructOnlyFullVoxels = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Sub reconstruction

        private bool _TomoSubReconstructNormalizedCoords = false;
        [WarpSerializable]
        public bool TomoSubReconstructNormalizedCoords
        {
            get { return _TomoSubReconstructNormalizedCoords; }
            set { if (value != _TomoSubReconstructNormalizedCoords) { _TomoSubReconstructNormalizedCoords = value; OnPropertyChanged(); } }
        }

        private decimal _TomoSubReconstructPixel = 1M;
        [WarpSerializable]
        public decimal TomoSubReconstructPixel
        {
            get { return _TomoSubReconstructPixel; }
            set { if (value != _TomoSubReconstructPixel) { _TomoSubReconstructPixel = value; OnPropertyChanged(); } }
        }

        private decimal _TomoSubReconstructBox = 128;
        [WarpSerializable]
        public decimal TomoSubReconstructBox
        {
            get { return _TomoSubReconstructBox; }
            set { if (value != _TomoSubReconstructBox) { _TomoSubReconstructBox = value; OnPropertyChanged(); } }
        }

        private decimal _TomoSubReconstructDiameter = 100;
        [WarpSerializable]
        public decimal TomoSubReconstructDiameter
        {
            get { return _TomoSubReconstructDiameter; }
            set { if (value != _TomoSubReconstructDiameter) { _TomoSubReconstructDiameter = value; OnPropertyChanged(); } }
        }

        private bool _TomoSubReconstructVolume = true;
        [WarpSerializable]
        public bool TomoSubReconstructVolume
        {
            get { return _TomoSubReconstructVolume; }
            set { if (value != _TomoSubReconstructVolume) { _TomoSubReconstructVolume = value; OnPropertyChanged(); } }
        }

        private bool _TomoSubReconstructSeries = false;
        [WarpSerializable]
        public bool TomoSubReconstructSeries
        {
            get { return _TomoSubReconstructSeries; }
            set { if (value != _TomoSubReconstructSeries) { _TomoSubReconstructSeries = value; OnPropertyChanged(); } }
        }

        private decimal _TomoSubReconstructShiftX = 0M;
        [WarpSerializable]
        public decimal TomoSubReconstructShiftX
        {
            get { return _TomoSubReconstructShiftX; }
            set { if (value != _TomoSubReconstructShiftX) { _TomoSubReconstructShiftX = value; OnPropertyChanged(); } }
        }

        private decimal _TomoSubReconstructShiftY = 0M;
        [WarpSerializable]
        public decimal TomoSubReconstructShiftY
        {
            get { return _TomoSubReconstructShiftY; }
            set { if (value != _TomoSubReconstructShiftY) { _TomoSubReconstructShiftY = value; OnPropertyChanged(); } }
        }

        private decimal _TomoSubReconstructShiftZ = 0M;
        [WarpSerializable]
        public decimal TomoSubReconstructShiftZ
        {
            get { return _TomoSubReconstructShiftZ; }
            set { if (value != _TomoSubReconstructShiftZ) { _TomoSubReconstructShiftZ = value; OnPropertyChanged(); } }
        }

        private bool _TomoSubReconstructPrerotated = false;
        [WarpSerializable]
        public bool TomoSubReconstructPrerotated
        {
            get { return _TomoSubReconstructPrerotated; }
            set { if (value != _TomoSubReconstructPrerotated) { _TomoSubReconstructPrerotated = value; OnPropertyChanged(); } }
        }

        private bool _TomoSubReconstructDoLimitDose = false;
        [WarpSerializable]
        public bool TomoSubReconstructDoLimitDose
        {
            get { return _TomoSubReconstructDoLimitDose; }
            set { if (value != _TomoSubReconstructDoLimitDose) { _TomoSubReconstructDoLimitDose = value; OnPropertyChanged(); } }
        }

        private int _TomoSubReconstructNTilts = 1;
        [WarpSerializable]
        public int TomoSubReconstructNTilts
        {
            get { return _TomoSubReconstructNTilts; }
            set { if (value != _TomoSubReconstructNTilts) { _TomoSubReconstructNTilts = value; OnPropertyChanged(); } }
        }

        private bool _TomoSubReconstructMakeSparse = true;
        [WarpSerializable]
        public bool TomoSubReconstructMakeSparse
        {
            get { return _TomoSubReconstructMakeSparse; }
            set { if (value != _TomoSubReconstructMakeSparse) { _TomoSubReconstructMakeSparse = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Template matching

        private decimal _TomoMatchTemplatePixel = 1M;
        [WarpSerializable]
        public decimal TomoMatchTemplatePixel
        {
            get { return _TomoMatchTemplatePixel; }
            set { if (value != _TomoMatchTemplatePixel) { _TomoMatchTemplatePixel = value; OnPropertyChanged(); } }
        }

        private decimal _TomoMatchTemplateDiameter = 100;
        [WarpSerializable]
        public decimal TomoMatchTemplateDiameter
        {
            get { return _TomoMatchTemplateDiameter; }
            set
            {
                if (value != _TomoMatchTemplateDiameter)
                {
                    _TomoMatchTemplateDiameter = value;
                    OnPropertyChanged();
                    TomoUpdateMatchRecommendation();
                }
            }
        }

        private decimal _TomoMatchPeakDistance = 100;
        [WarpSerializable]
        public decimal TomoMatchPeakDistance
        {
            get { return _TomoMatchPeakDistance; }
            set { if (value != _TomoMatchPeakDistance) { _TomoMatchPeakDistance = value; OnPropertyChanged(); } }
        }

        private decimal _TomoMatchTemplateFraction = 100M;
        [WarpSerializable]
        public decimal TomoMatchTemplateFraction
        {
            get { return _TomoMatchTemplateFraction; }
            set { if (value != _TomoMatchTemplateFraction) { _TomoMatchTemplateFraction = value; OnPropertyChanged(); } }
        }

        private bool _TomoMatchWhitenSpectrum = true;
        [WarpSerializable]
        public bool TomoMatchWhitenSpectrum
        {
            get { return _TomoMatchWhitenSpectrum; }
            set { if (value != _TomoMatchWhitenSpectrum) { _TomoMatchWhitenSpectrum = value; OnPropertyChanged(); } }
        }

        private decimal _TomoMatchHealpixOrder = 1;
        [WarpSerializable]
        public decimal TomoMatchHealpixOrder
        {
            get { return _TomoMatchHealpixOrder; }
            set
            {
                if (value != _TomoMatchHealpixOrder)
                {
                    _TomoMatchHealpixOrder = value;
                    OnPropertyChanged();

                    TomoMatchHealpixAngle = Math.Round(60M / (decimal)Math.Pow(2, (double)value), 3);
                }
            }
        }

        private int _TomoMatchBatchAngles = 32;
        [WarpSerializable]
        public int TomoMatchBatchAngles
        {
            get { return _TomoMatchBatchAngles; }
            set { if (value != _TomoMatchBatchAngles) { _TomoMatchBatchAngles = value; OnPropertyChanged(); } }
        }

        private decimal _TomoMatchHealpixAngle = 30;
        public decimal TomoMatchHealpixAngle
        {
            get { return _TomoMatchHealpixAngle; }
            set
            {
                if (value != _TomoMatchHealpixAngle)
                {
                    _TomoMatchHealpixAngle = value;
                    OnPropertyChanged();
                    TomoUpdateMatchRecommendation();
                }
            }
        }

        private string _TomoMatchSymmetry = "C1";
        [WarpSerializable]
        public string TomoMatchSymmetry
        {
            get { return _TomoMatchSymmetry; }
            set { if (value != _TomoMatchSymmetry) { _TomoMatchSymmetry = value; OnPropertyChanged(); } }
        }

        private decimal _TomoMatchRecommendedAngPix = 25.88M;
        public decimal TomoMatchRecommendedAngPix
        {
            get { return _TomoMatchRecommendedAngPix; }
            set { if (value != _TomoMatchRecommendedAngPix) { _TomoMatchRecommendedAngPix = value; OnPropertyChanged(); } }
        }

        private void TomoUpdateMatchRecommendation()
        {
            float2 AngularSampling = new float2((float)Math.Sin((float)TomoMatchHealpixAngle * Helper.ToRad),
                                                1f - (float)Math.Cos((float)TomoMatchHealpixAngle * Helper.ToRad));
            decimal AtLeast = TomoMatchTemplateDiameter / 2 * (decimal)AngularSampling.Length();
            AtLeast = Math.Round(AtLeast, 2);
            TomoMatchRecommendedAngPix = AtLeast;
        }

        private decimal _TomoMatchNResults = 1000;
        [WarpSerializable]
        public decimal TomoMatchNResults
        {
            get { return _TomoMatchNResults; }
            set { if (value != _TomoMatchNResults) { _TomoMatchNResults = value; OnPropertyChanged(); } }
        }

        private bool _ReuseCorrVolumes = false;
        [WarpSerializable]
        public bool ReuseCorrVolumes
        {
            get { return _ReuseCorrVolumes; }
            set { if (value != _ReuseCorrVolumes) { _ReuseCorrVolumes = value; OnPropertyChanged(); } }
        }

        #endregion

        #endregion
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class OptionsFilter : WarpBase
    {
        private decimal _AstigmatismMax = 3;
        [WarpSerializable]
        [JsonProperty]
        public decimal AstigmatismMax
        {
            get { return _AstigmatismMax; }
            set { if (value != _AstigmatismMax) { _AstigmatismMax = value; OnPropertyChanged(); } }
        }

        private decimal _DefocusMin = 0;
        [WarpSerializable]
        [JsonProperty]
        public decimal DefocusMin
        {
            get { return _DefocusMin; }
            set { if (value != _DefocusMin) { _DefocusMin = value; OnPropertyChanged(); } }
        }

        private decimal _DefocusMax = 5;
        [WarpSerializable]
        [JsonProperty]
        public decimal DefocusMax
        {
            get { return _DefocusMax; }
            set { if (value != _DefocusMax) { _DefocusMax = value; OnPropertyChanged(); } }
        }

        private decimal _PhaseMin = 0;
        [WarpSerializable]
        [JsonProperty]
        public decimal PhaseMin
        {
            get { return _PhaseMin; }
            set { if (value != _PhaseMin) { _PhaseMin = value; OnPropertyChanged(); } }
        }

        private decimal _PhaseMax = 1;
        [WarpSerializable]
        [JsonProperty]
        public decimal PhaseMax
        {
            get { return _PhaseMax; }
            set { if (value != _PhaseMax) { _PhaseMax = value; OnPropertyChanged(); } }
        }

        private decimal _ResolutionMax = 5;
        [WarpSerializable]
        [JsonProperty]
        public decimal ResolutionMax
        {
            get { return _ResolutionMax; }
            set { if (value != _ResolutionMax) { _ResolutionMax = value; OnPropertyChanged(); } }
        }

        private decimal _MotionMax = 5;
        [WarpSerializable]
        [JsonProperty]
        public decimal MotionMax
        {
            get { return _MotionMax; }
            set { if (value != _MotionMax) { _MotionMax = value; OnPropertyChanged(); } }
        }

        private string _ParticlesSuffix = "";
        [WarpSerializable]
        [JsonProperty]
        public string ParticlesSuffix
        {
            get { return _ParticlesSuffix; }
            set { if (value != _ParticlesSuffix) { _ParticlesSuffix = value; OnPropertyChanged(); } }
        }

        private int _ParticlesMin = 1;
        [WarpSerializable]
        [JsonProperty]
        public int ParticlesMin
        {
            get { return _ParticlesMin; }
            set { if (value != _ParticlesMin) { _ParticlesMin = value; OnPropertyChanged(); } }
        }

        private decimal _MaskPercentage = 10;
        [WarpSerializable]
        [JsonProperty]
        public decimal MaskPercentage
        {
            get { return _MaskPercentage; }
            set { if (value != _MaskPercentage) { _MaskPercentage = value; OnPropertyChanged(); } }
        }
    }

    public enum ProcessingStatus
    {
        Processed = 1,
        Outdated = 2,
        Unprocessed = 3,
        FilteredOut = 4,
        LeaveOut = 5
    }
}