using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;
using BitMiracle.LibTiff.Classic;
using MahApps.Metro.Controls.Dialogs;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;
using Orientation = System.Windows.Controls.Orientation;
using Path = System.Windows.Shapes.Path;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for MicrographDisplay.xaml
    /// </summary>
    public partial class MicrographDisplay : UserControl
    {
        bool UpdatesPaused = false;
        bool IsDraggingImage = false;
        bool IsDraggingParticle = false;
        bool IsDraggingBrush = false;
        Particle DraggedParticle = null;
        Point DraggingStartPoint;
        bool BlinkyOn = false;
        DispatcherTimer BlinkyTimer;

        bool ShowedNoiseNetWarning = false;

        #region Dependency properties

        public Movie Movie
        {
            get { return (Movie)GetValue(MovieProperty); }
            set { SetValue(MovieProperty, value); }
        }
        public static readonly DependencyProperty MovieProperty = DependencyProperty.Register("Movie", typeof(Movie), typeof(MicrographDisplay), new PropertyMetadata(null, (sender, e) => ((MicrographDisplay)sender).DispatchMovieChanged(sender, e)));

        public decimal Zoom
        {
            get { return (decimal)GetValue(ZoomProperty); }
            set { SetValue(ZoomProperty, value); }
        }
        public static readonly DependencyProperty ZoomProperty = DependencyProperty.Register("Zoom", typeof(decimal), typeof(MicrographDisplay), new PropertyMetadata(0.25M, (sender, e) => ((MicrographDisplay)sender).DisplaySettingsChanged(sender, e)));
        private double ScaleFactor => (double)Zoom;

        public decimal IntensityRange
        {
            get { return (decimal)GetValue(IntensityRangeProperty); }
            set { SetValue(IntensityRangeProperty, value); }
        }
        public static readonly DependencyProperty IntensityRangeProperty = DependencyProperty.Register("IntensityRange", typeof(decimal), typeof(MicrographDisplay), new PropertyMetadata(2.5M, (sender, e) => ((MicrographDisplay)sender).DisplaySettingsChanged(sender, e)));

        public int SliceID
        {
            get { return (int)GetValue(SliceIDProperty); }
            set { SetValue(SliceIDProperty, value); }
        }
        public static readonly DependencyProperty SliceIDProperty = DependencyProperty.Register("SliceID", typeof(int), typeof(MicrographDisplay), new PropertyMetadata(1, (sender, e) => ((MicrographDisplay)sender).DisplaySettingsChanged(sender, e)));

        public bool DeconvEnabled
        {
            get { return (bool)GetValue(DeconvEnabledProperty); }
            set { SetValue(DeconvEnabledProperty, value); }
        }
        public static readonly DependencyProperty DeconvEnabledProperty = DependencyProperty.Register("DeconvEnabled", typeof(bool), typeof(MicrographDisplay), new PropertyMetadata(false, (sender, e) => ((MicrographDisplay)sender).DisplaySettingsChanged(sender, e)));
        
        public string DeconvMode
        {
            get { return (string)GetValue(DeconvModeProperty); }
            set { SetValue(DeconvModeProperty, value); }
        }
        public static readonly DependencyProperty DeconvModeProperty = DependencyProperty.Register("DeconvMode", typeof(string), typeof(MicrographDisplay), new PropertyMetadata("Denoise", (sender, e) => ((MicrographDisplay)sender).DisplaySettingsChanged(sender, e)));

        public decimal DeconvStrength
        {
            get { return (decimal)GetValue(DeconvStrengthProperty); }
            set { SetValue(DeconvStrengthProperty, value); }
        }
        public static readonly DependencyProperty DeconvStrengthProperty = DependencyProperty.Register("DeconvStrength", typeof(decimal), typeof(MicrographDisplay), new PropertyMetadata(1M, (sender, e) => ((MicrographDisplay)sender).DisplaySettingsChanged(sender, e)));

        public decimal DeconvFalloff
        {
            get { return (decimal)GetValue(DeconvFalloffProperty); }
            set { SetValue(DeconvFalloffProperty, value); }
        }
        public static readonly DependencyProperty DeconvFalloffProperty = DependencyProperty.Register("DeconvFalloff", typeof(decimal), typeof(MicrographDisplay), new PropertyMetadata(1M, (sender, e) => ((MicrographDisplay)sender).DisplaySettingsChanged(sender, e)));

        public int DeconvHighpass
        {
            get { return (int)GetValue(DeconvHighpassProperty); }
            set { SetValue(DeconvHighpassProperty, value); }
        }
        public static readonly DependencyProperty DeconvHighpassProperty = DependencyProperty.Register("DeconvHighpass", typeof(int), typeof(MicrographDisplay), new PropertyMetadata(100, (sender, e) => ((MicrographDisplay)sender).DisplaySettingsChanged(sender, e)));

        public bool TrackShow
        {
            get { return (bool)GetValue(TrackShowProperty); }
            set { SetValue(TrackShowProperty, value); }
        }
        public static readonly DependencyProperty TrackShowProperty = DependencyProperty.Register("TrackShow", typeof(bool), typeof(MicrographDisplay), new PropertyMetadata(false, (sender, e) => ((MicrographDisplay)sender).TrackSettingsChanged(sender, e)));

        public bool TrackLocalOnly
        {
            get { return (bool)GetValue(TrackLocalOnlyProperty); }
            set { SetValue(TrackLocalOnlyProperty, value); }
        }
        public static readonly DependencyProperty TrackLocalOnlyProperty = DependencyProperty.Register("TrackLocalOnly", typeof(bool), typeof(MicrographDisplay), new PropertyMetadata(false, (sender, e) => ((MicrographDisplay)sender).TrackSettingsChanged(sender, e)));

        public decimal TrackScale
        {
            get { return (decimal)GetValue(TrackScaleProperty); }
            set { SetValue(TrackScaleProperty, value); }
        }
        public static readonly DependencyProperty TrackScaleProperty = DependencyProperty.Register("TrackScale", typeof(decimal), typeof(MicrographDisplay), new PropertyMetadata(20M, (sender, e) => ((MicrographDisplay)sender).TrackSettingsChanged(sender, e)));

        public int TrackGridX
        {
            get { return (int)GetValue(TrackGridXProperty); }
            set { SetValue(TrackGridXProperty, value); }
        }
        public static readonly DependencyProperty TrackGridXProperty = DependencyProperty.Register("TrackGridX", typeof(int), typeof(MicrographDisplay), new PropertyMetadata(8, (sender, e) => ((MicrographDisplay)sender).TrackSettingsChanged(sender, e)));

        public int TrackGridY
        {
            get { return (int)GetValue(TrackGridYProperty); }
            set { SetValue(TrackGridYProperty, value); }
        }
        public static readonly DependencyProperty TrackGridYProperty = DependencyProperty.Register("TrackGridY", typeof(int), typeof(MicrographDisplay), new PropertyMetadata(8, (sender, e) => ((MicrographDisplay)sender).TrackSettingsChanged(sender, e)));

        public bool ElevationShow
        {
            get { return (bool)GetValue(ElevationShowProperty); }
            set { SetValue(ElevationShowProperty, value); }
        }
        public static readonly DependencyProperty ElevationShowProperty = DependencyProperty.Register("ElevationShow", typeof(bool), typeof(MicrographDisplay), new PropertyMetadata(false, (sender, e) => ((MicrographDisplay)sender).ElevationSettingsChanged(sender, e)));

        public bool ParticlesShow
        {
            get { return (bool)GetValue(ParticlesShowProperty); }
            set { SetValue(ParticlesShowProperty, value); }
        }
        public static readonly DependencyProperty ParticlesShowProperty = DependencyProperty.Register("ParticlesShow", typeof(bool), typeof(MicrographDisplay), new PropertyMetadata(false, (sender, e) => ((MicrographDisplay)sender).ParticlesSettingsChanged(sender, e)));
        
        public string ParticlesSuffix
        {
            get { return (string)GetValue(ParticlesSuffixProperty); }
            set { SetValue(ParticlesSuffixProperty, value); }
        }
        public static readonly DependencyProperty ParticlesSuffixProperty = DependencyProperty.Register("ParticlesSuffix", typeof(string), typeof(MicrographDisplay), new PropertyMetadata(null, (sender, e) => ((MicrographDisplay)sender).ParticlesSuffixChanged(sender, e)));

        public int ParticlesDiameter
        {
            get { return (int)GetValue(ParticlesDiameterProperty); }
            set { SetValue(ParticlesDiameterProperty, value); }
        }
        public static readonly DependencyProperty ParticlesDiameterProperty = DependencyProperty.Register("ParticlesDiameter", typeof(int), typeof(MicrographDisplay), new PropertyMetadata(100, (sender, e) => ((MicrographDisplay)sender).ParticlesSettingsChanged(sender, e)));
        
        public Color ParticleCircleBrush
        {
            get { return (Color)GetValue(ParticleCircleBrushProperty); }
            set { SetValue(ParticleCircleBrushProperty, value); }
        }
        public static readonly DependencyProperty ParticleCircleBrushProperty = DependencyProperty.Register("ParticleCircleBrush", typeof(Color), typeof(MicrographDisplay), new PropertyMetadata(Color.FromArgb(255, 255, 240, 0), (sender, e) => ((MicrographDisplay)sender).ParticlesSettingsChanged(sender, e)));
        
        public string ParticleCircleStyle
        {
            get { return (string)GetValue(ParticleCircleStyleProperty); }
            set { SetValue(ParticleCircleStyleProperty, value); }
        }
        public static readonly DependencyProperty ParticleCircleStyleProperty = DependencyProperty.Register("ParticleCircleStyle", typeof(string), typeof(MicrographDisplay), new PropertyMetadata("Circles", (sender, e) => ((MicrographDisplay)sender).ParticlesSettingsChanged(sender, e)));

        public decimal ParticlesThreshold
        {
            get { return (decimal)GetValue(ParticlesThresholdProperty); }
            set { SetValue(ParticlesThresholdProperty, value); }
        }
        public static readonly DependencyProperty ParticlesThresholdProperty = DependencyProperty.Register("ParticlesThreshold", typeof(decimal), typeof(MicrographDisplay), new PropertyMetadata(0M, (sender, e) => ((MicrographDisplay)sender).ParticlesSettingsChanged(sender, e)));

        public int ParticlesCount
        {
            get { return (int)GetValue(ParticlesCountProperty); }
            set { SetValue(ParticlesCountProperty, value); }
        }
        public static readonly DependencyProperty ParticlesCountProperty = DependencyProperty.Register("ParticlesCount", typeof(int), typeof(MicrographDisplay), new PropertyMetadata(0));

        public bool ParticlesShowFOM
        {
            get { return (bool)GetValue(ParticlesShowFOMProperty); }
            set { SetValue(ParticlesShowFOMProperty, value); }
        }
        public static readonly DependencyProperty ParticlesShowFOMProperty = DependencyProperty.Register("ParticlesShowFOM", typeof(bool), typeof(MicrographDisplay), new PropertyMetadata(false, (sender, e) => ((MicrographDisplay)sender).ParticlesSettingsChanged(sender, e)));
        
        public bool ParticlesBlinky
        {
            get { return (bool)GetValue(ParticlesBlinkyProperty); }
            set { SetValue(ParticlesBlinkyProperty, value); }
        }
        public static readonly DependencyProperty ParticlesBlinkyProperty = DependencyProperty.Register("ParticlesBlinky", typeof(bool), typeof(MicrographDisplay), new PropertyMetadata(false, (sender, e) => ((MicrographDisplay)sender).ParticlesBlinkyChanged(sender, e)));

        public bool MaskShow
        {
            get { return (bool)GetValue(MaskShowProperty); }
            set { SetValue(MaskShowProperty, value); }
        }
        public static readonly DependencyProperty MaskShowProperty = DependencyProperty.Register("MaskShow", typeof(bool), typeof(MicrographDisplay), new PropertyMetadata(false, (sender, e) => ((MicrographDisplay)sender).MaskSettingsChanged(sender, e)));

        public bool BrushActive
        {
            get { return (bool)GetValue(BrushActiveProperty); }
            set { SetValue(BrushActiveProperty, value); }
        }
        public static readonly DependencyProperty BrushActiveProperty = DependencyProperty.Register("BrushActive", typeof(bool), typeof(MicrographDisplay), new PropertyMetadata(false));
        
        public int BrushDiameter
        {
            get { return (int)GetValue(BrushDiameterProperty); }
            set { SetValue(BrushDiameterProperty, value); }
        }
        public static readonly DependencyProperty BrushDiameterProperty = DependencyProperty.Register("BrushDiameter", typeof(int), typeof(MicrographDisplay), new PropertyMetadata(300));
        
        #endregion

        private float2 _VisualsCenterOffset = new float2();
        public float2 VisualsCenterOffset
        {
            get { return _VisualsCenterOffset; }
            set
            {
                if (value != _VisualsCenterOffset)
                {
                    _VisualsCenterOffset = value;
                    PositionVisuals();
                }
            }
        }

        float PixelSize = 1;

        int PlanForw = -1;
        int2 DimsPlanForw = new int2(-1);
        int PlanBack = -1;
        int2 DimsPlanBack = new int2(-1);
        int PlanForw3A = -1;
        int2 DimsPlanForw3A = new int2(-1);
        int PlanBack3A = -1;
        int2 DimsPlanBack3A = new int2(-1);

        Image MicrographFT = null;
        Image MicrographDenoised = null;
        Movie MicrographOwner = null;
        int MicrographOwnerSlice = -1;

        private BoxNetTorch[] BoxNetworks = null;
        private string BoxNetworksModelPath = "";

        private NoiseNet2DTorch[] NoiseNetworks = null;
        private string NoiseNetworksModelDir = "";

        readonly List<Particle> Particles = new List<Particle>();
        readonly List<Particle> ParticlesBad = new List<Particle>();

        private int2 MicrographDims = new int2(1);
        private int2 MicrographDimsDenoise = new int2(1);
        private int2 MaskDims = new int2(1);
        private byte[] MaskData = null;
        private byte[] MaskImageBytes = null;

        private readonly SemaphoreSlim UpdateImageSync = new SemaphoreSlim(1);
        bool IsUpdatingImage = false;

        public MicrographDisplay()
        {
            InitializeComponent();

            ImageDisplay.PreviewMouseWheel += ImageDisplay_MouseWheel;
            ImageDisplay.MouseLeave += MicrographDisplay_MouseLeave;
            SizeChanged += MicrographDisplay_SizeChanged;

            BlinkyTimer = new DispatcherTimer(new TimeSpan(0, 0, 0, 0, 800), DispatcherPriority.Normal, (a, b) =>
            {
                BlinkyOn = !BlinkyOn;
                UpdateParticles();
            }, Dispatcher);
            BlinkyTimer.Stop();
        }

        #region Data context and settings changes

        private void DispatchMovieChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            Dispatcher.Invoke(() => MovieChanged(sender, e));
        }

        private async void MovieChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            while (IsUpdatingImage)
                await Task.Delay(50);

            #region Attach and detach event handlers

            if (e.OldValue != null)
            {
                if (e.OldValue.GetType() == typeof(Movie))
                {
                    Movie E = (Movie)e.OldValue;

                    E.ProcessingChanged -= Movie_ProcessingChanged;
                    E.AverageChanged -= Movie_AverageChanged;

                    if (File.Exists(E.AveragePath))
                        RenderToThumbnail(E.ThumbnailsPath, 384);
                }
                else if (e.OldValue.GetType() == typeof(TiltSeries))
                {
                    TiltSeries E = (TiltSeries)e.OldValue;
                }
            }

            if (e.NewValue != null)
            {
                if (e.NewValue.GetType() == typeof(Movie))
                {
                    Movie E = (Movie)e.NewValue;

                    E.ProcessingChanged += Movie_ProcessingChanged;
                    E.AverageChanged += Movie_AverageChanged;
                }
                else if (e.NewValue.GetType() == typeof(TiltSeries))
                {
                    TiltSeries E = (TiltSeries)e.NewValue;
                }
            }

            #endregion

            #region Figure out visibility of various elements

            IsDraggingImage = false;
            IsDraggingBrush = false;
            PanelImageSettings.Visibility = Visibility.Collapsed;
            PanelTrackSettings.Visibility = Visibility.Collapsed;
            PanelElevationSettings.Visibility = Visibility.Collapsed;
            PanelParticleSettings.Visibility = Visibility.Collapsed;

            if (e.NewValue != null)
            {
                GridNoMovie.Visibility = Visibility.Hidden;

                if (e.NewValue.GetType() == typeof (Movie) && ((Movie)e.NewValue).OptionsMovement == null) // No estimate available yet for movie.
                {
                    CanvasTrack.Visibility = Visibility.Hidden;
                    CanvasTrackMouse.Visibility = Visibility.Hidden;
                    GridNotProcessed.Visibility = Visibility.Visible;
                }
                else // Estimate data available, or it's a tilt series.
                {
                    CanvasTrack.Visibility = Visibility.Visible;
                    CanvasTrackMouse.Visibility = Visibility.Visible;
                    GridNotProcessed.Visibility = Visibility.Hidden;

                    if (e.NewValue.GetType() == typeof (TiltSeries)) // If tilt series, set range for the tilt slider according to header
                    {
                        TiltSeries E = (TiltSeries)e.NewValue;

                        SliderSliceID.MaxValue = E.NTilts;
                        UpdatesPaused = true;
                        SliceID = E.IndicesSortedDose[0] + 1;
                        UpdatesPaused = false;

                        PanelDeconvOptions.Visibility = Visibility.Collapsed;
                        SliderSliceID.Visibility = Visibility.Visible;
                    }
                    else
                    {
                        Movie E = (Movie)e.NewValue;
                        
                        SliderSliceID.Visibility = Visibility.Collapsed;
                    }
                }

                Dispatcher.Invoke(() =>
                {
                    UpdateImage();
                    UpdateElevation();
                    UpdateTrackGrid();
                    UpdateTrackMouse(null);
                }, DispatcherPriority.Background);
            }
            else // No movie selected.
            {
                GridNoMovie.Visibility = Visibility.Visible;
                GridNotProcessed.Visibility = Visibility.Hidden;
                ImageDisplay.Visibility = Visibility.Hidden;
                CanvasTrack.Visibility = Visibility.Hidden;
                CanvasTrackMouse.Visibility = Visibility.Hidden;
            }

            #endregion

            #region Deal with particles and mask

            string[] AvailableSuffixes = DiscoverValidSuffixes();
            PopulateValidSuffixes();

            if (AvailableSuffixes.Length > 0)
            {
                PanelParticleSettings.Visibility = Visibility.Visible;

                if (ParticlesSuffix == null)
                    ParticlesSuffix = AvailableSuffixes[0];
                else if (!AvailableSuffixes.Contains(ParticlesSuffix))
                    ParticlesSuffix = AvailableSuffixes[0];
            }

            if (e.NewValue != null && e.NewValue.GetType() == typeof(Movie))
            {
                if (File.Exists(((Movie)e.NewValue).AveragePath))
                {
                    MapHeader Header = MapHeader.ReadFromFile(((Movie)e.NewValue).AveragePath);
                    PixelSize = Header.PixelSize.X;
                    MicrographDims = new int2(Header.Dimensions);
                    MicrographDimsDenoise = new int2(new float2(MicrographDims) * PixelSize / NoiseNet2DTorch.PixelSize + 1) / 2 * 2;

                    float2 MicrographDimsPhysical = new float2(MicrographDims) * PixelSize;
                    MaskDims = new int2(MicrographDimsPhysical / 8f + 0.5f) / 2 * 2;

                    // Load mask from disk if it's there
                    
                }

                LoadParticles();

                if (ParticlesSuffix != null && Movie.PickingThresholds.ContainsKey(ParticlesSuffix))
                    ParticlesThreshold = Movie.PickingThresholds[ParticlesSuffix];
                else if (Particles.Count > 0)
                    ParticlesThreshold = (decimal)Particles.Select(p => p.FOM).Min();
            }
            else
            {
                Particles.Clear();
                ParticlesBad.Clear();
            }
                
            ReloadMask();

            UpdateParticles();
            UpdateMask();

            #endregion
        }

        private void DisplaySettingsChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args)
        {
            if (UpdatesPaused)
                return;

            Dispatcher.Invoke(() =>
            {
                UpdateImage();
                UpdateElevation();
                UpdateTrackGrid();
                UpdateTrackMouse(null);
                UpdateMask();
            });
        }

        private void TrackSettingsChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args)
        {
            if (UpdatesPaused)
                return;

            UpdateTrackGrid();
            UpdateTrackMouse(null);
        }

        private void ElevationSettingsChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args)
        {
            ImageElevation.Visibility = ElevationShow ? Visibility.Visible : Visibility.Collapsed;
        }

        private void Movie_ProcessingChanged(object sender, EventArgs e)
        {
            DispatchUpdateImage();
            DispatchUpdateElevation();

            Dispatcher.InvokeAsync(() =>
            {
                UpdateTrackGrid();
                UpdateTrackMouse(null);
            });
        }

        private void Movie_AverageChanged(object sender, EventArgs e)
        {
            MicrographOwner = null;

            MicrographFT?.Dispose();
            MicrographFT = null;

            MicrographDenoised?.Dispose();
            MicrographDenoised = null;

            DispatchUpdateImage();
        }

        #endregion

        #region Image display elements

        private void MicrographDisplay_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            PositionVisuals();
        }

        private void DispatchUpdateImage()
        {
            Dispatcher.Invoke(() => UpdateImage());
        }

        public async void UpdateImage()
        {
            while (IsUpdatingImage)
                await Task.Delay(50);

            if (IsUpdatingImage)
                Debug.WriteLine("I shouldn't be here.");

            if (Movie == null)
                return;

            string AveragePath = Movie.AveragePath;

            if (Movie.GetType() == typeof(Movie) && !File.Exists(AveragePath))
            {
                ImageDisplay.Visibility = Visibility.Collapsed;
                CanvasTrack.Visibility = Visibility.Collapsed;
                GridNotProcessed.Visibility = Visibility.Visible;
                return;
            }

            //UpdateImageSync.Wait();
            IsUpdatingImage = true;

            ImageSource AverageImage = null;
            GridNotProcessed.Visibility = Visibility.Hidden;
            //ImageDisplay.Visibility = Visibility.Collapsed;
            CanvasTrack.Visibility = Visibility.Collapsed;
            CanvasTrackMouse.Visibility = Visibility.Collapsed;
            ProgressImage.Visibility = Visibility.Visible;

            PanelVisuals.Effect = new BlurEffect() { Radius = 16 };

            PanelImageSettings.Visibility = Visibility.Visible;

            // Warn once about low memory in case denoising is to be performed
            if (!ShowedNoiseNetWarning && DeconvMode == "Denoise" && DeconvEnabled)
            {
                ShowedNoiseNetWarning = true;

                if (GPU.GetFreeMemory(0) < 5500)
                {
                    var Result = await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Careful there!",
                                                                                                     "You are about to activate denoising. This feature will consume ca. 2.5 GB of GPU memory\n" +
                                                                                                     "permanently until you restart Warp. It looks like your GPU might not have enough memory\n" +
                                                                                                     "to still be able to do all the other cool things. A likely symptom will be completely black\n" +
                                                                                                     "movie averages, or Warp just crashing.\n\n" +
                                                                                                     "Would you like to disable denoising and do deconvolution instead?",
                                                                                                     MessageDialogStyle.AffirmativeAndNegative,
                                                                                                     new MetroDialogSettings()
                                                                                                     {
                                                                                                         AffirmativeButtonText = "Yes, deconvolve instead",
                                                                                                         NegativeButtonText = "No, proceed with denoising"
                                                                                                     });
                    
                    if (Result == MessageDialogResult.Affirmative)
                    {
                        IsUpdatingImage = false;
                        DeconvMode = "Deconvolve";
                        return;
                    }
                }
            }

            PanelDeconvCanonicalOptions.Visibility = DeconvMode == "Deconvolve" ? Visibility.Visible : Visibility.Collapsed;
            PanelDeconvDenoisingOptions.Visibility = DeconvMode == "Denoise" ? Visibility.Visible : Visibility.Collapsed;

            //await Task.Run(() =>
            {
                Image Average = null;
                bool CanDeconvolve = Movie.GetType() == typeof (Movie) && Movie.OptionsCTF != null;
                bool ShouldDeconvolve = CanDeconvolve && DeconvEnabled;

                PanelDeconvOptions.Visibility = CanDeconvolve ? Visibility.Visible : Visibility.Collapsed;

                #region Process precached FT of the image, or load original from disk if no processing is needed

                if (Zoom != 1 || ShouldDeconvolve)
                {
                    Movie _Movie = Movie;
                    double _ScaleFactor = ScaleFactor;
                    decimal _Zoom = Zoom;
                    decimal _DeconvStrength = DeconvStrength;
                    decimal _DeconvFalloff = DeconvFalloff;
                    decimal _DeconvHighpass = DeconvHighpass;

                    if (!(ShouldDeconvolve && DeconvMode == "Denoise"))     // Normal deconvolution
                    {
                        Average = await Task.Run(() =>
                        {
                            Image AverageFT;
                            if (_Movie.GetType() == typeof(Movie))
                                AverageFT = GetMicrographFT(_Movie);
                            else
                                AverageFT = new Image(new int3(64, 64, 1), true, true);

                            Image AverageFTCropped = AverageFT;
                            if (_Zoom < 1)
                            {
                                int2 DimsZoomed = AverageFT.DimsSlice * (float)_ScaleFactor / 2 * 2;
                                AverageFTCropped = AverageFT.AsPadded(DimsZoomed);
                            }

                            Image AverageFTCroppedDeconv = AverageFTCropped;
                            if (ShouldDeconvolve)
                            {
                                CTF DeconvCTF = _Movie.CTF.GetCopy();
                                DeconvCTF.PixelSize = (decimal)PixelSize;
                                DeconvCTF.PixelSize /= (decimal)Math.Min(1, _ScaleFactor);

                                float HighpassNyquist = (float)(DeconvCTF.PixelSize * 2 / _DeconvHighpass);

                                AverageFTCroppedDeconv = new Image(IntPtr.Zero, AverageFTCropped.Dims, true, true);
                                GPU.DeconvolveCTF(AverageFTCropped.GetDevice(Intent.Read),
                                                  AverageFTCroppedDeconv.GetDevice(Intent.Write),
                                                  AverageFTCropped.Dims,
                                                  DeconvCTF.ToStruct(),
                                                  (float)_DeconvStrength,
                                                  (float)_DeconvFalloff,
                                                  HighpassNyquist);
                            }

                            Image Result =  AverageFTCroppedDeconv.AsIFFT(false, GetPlanBack(AverageFTCroppedDeconv.DimsSlice), true, true);

                            if (_Zoom < 1)
                                AverageFTCropped.Dispose();
                            if (ShouldDeconvolve)
                                AverageFTCroppedDeconv.Dispose();

                            return Result;
                        });
                    }
                    else        // Deconvolution + denoising
                    {
                        Average = await Task.Run(async () =>
                        {
                            int2 DimsZoomed = MicrographDims * (float)Math.Min(1, _ScaleFactor) / 2 * 2;

                            try
                            {
                                Image Denoised = await GetMicrographDenoised(_Movie);
                                return Denoised.AsScaled(DimsZoomed, GetPlanForw3A(new int2(Denoised.Dims)), GetPlanBack(DimsZoomed));
                            }
                            catch (Exception exc)
                            {
                                await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                                    "Something went wrong while trying to denoise this image. Here are the details:\n\n" +
                                                                                                    exc.ToString() +
                                                                                                    "\n\nIf you keep getting this message, your system might not have enough memory\n" +
                                                                                                    "to process data and apply denoising at the same time. Please report the issue at\n" +
                                                                                                    "https://groups.google.com/forum/#!forum/warp-em");

                                return new Image(new int3(DimsZoomed));
                            }
                        });
                    }
                }
                else
                {
                    if (Movie.GetType() == typeof(Movie))
                        Average = await Task.Run(() => Image.FromFile(AveragePath));
                    else
                        Average = new Image(new int3(64, 64, 1)); // Image.FromFile(Movie.Path, SliceID - 1);
                }

                Average.Multiply(1e-6f);

                #endregion

                #region Create a BitmapSource from the raw image data

                float[] Data = Average.GetHost(Intent.Read)[0];
                Average.Dispose();

                int Width = Average.Dims.X;
                int Height = Average.Dims.Y;
                int Elements = Width * Height;

                double Mean = 0, StdMinus = 0, StdPlus = 0;
                int SamplesMinus = 0, SamplesPlus = 0;

                unsafe
                {
                    fixed (float* DataPtr = Data)
                    {
                        float* DataP = DataPtr;

                        int Samples = 0;
                        for (int y = Height / 3; y < Height * 2 / 3; y++)
                            for (int x = Width / 3; x < Width * 2 / 3; x++)
                            {
                                Mean += DataPtr[y * Width + x];
                                Samples++;
                            }
                        Mean /= Samples;

                        for (int y = Height / 4; y < Height * 3 / 4; y++)
                            for (int x = Width / 4; x < Width * 3 / 4; x++)
                            {
                                double Val = DataPtr[y * Width + x];

                                Val -= Mean;
                                if (Val >= 0)
                                {
                                    StdPlus += Val * Val;
                                    SamplesPlus++;
                                }
                                else
                                {
                                    StdMinus += Val * Val;
                                    SamplesMinus++;
                                }
                            }

                        StdMinus = (float)Math.Sqrt(StdMinus / SamplesMinus);
                        StdPlus = (float)Math.Sqrt(StdPlus / SamplesPlus);

                        if (StdPlus == 0f)
                        {
                            AverageImage = BitmapSource.Create(Width, Height, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, new byte[Data.Length], Width);
                        }
                        else
                        {
                            float Min = (float)(Mean - (float)IntensityRange * StdMinus);
                            float Range = (float)((float)IntensityRange * (StdMinus + StdPlus));

                            byte[] DataBytes = new byte[Data.Length];
                            fixed (byte* DataBytesPtr = DataBytes)
                            {
                                DataP = DataPtr;
                                byte* DataBytesP = DataBytesPtr;
                                Parallel.For(0, Height, y =>
                                {
                                    for (int x = 0; x < Width; x++)
                                        DataBytesP[(Height - 1 - y) * Width + x] = (byte)(Math.Max(Math.Min(1f, (DataP[y * Width + x] - Min) / Range), 0f) * 255f);
                                });
                            }

                            AverageImage = BitmapSource.Create(Width, Height, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, DataBytes, Width);
                            AverageImage.Freeze();
                        }
                    }
                }

                #endregion
            }//);

            PanelVisuals.Effect = null;

            ProgressImage.Visibility = Visibility.Hidden;
            ImageDisplay.Source = AverageImage;
            ImageDisplay.Visibility = Visibility.Visible;
            CanvasTrack.Visibility = Visibility.Visible;
            CanvasTrackMouse.Visibility = Visibility.Visible;

            //UpdateImageSync.Release();
            IsUpdatingImage = false;
            Debug.WriteLine("Released UpdateImage semaphore.");

            PositionVisuals();
        }

        private void DispatchUpdateElevation()
        {
            Dispatcher.Invoke(() => UpdateElevation());
        }

        public async void UpdateElevation()
        {
            if (Movie == null)
                return;

            ImageSource ElevationImage = null;

            if (Movie.GetType() == typeof(Movie) && Movie.GridCTFDefocus != null && Movie.GridCTFDefocus.Dimensions.Elements() > 1)
            {
                int2 DimsElevation = new int2(16);
                float3[] ElevationPositions = Helper.Combine(Helper.ArrayOfFunction(y => Helper.ArrayOfFunction(x => new float3(x / (float)(DimsElevation.X - 1),
                                                                                                                                1 - y / (float)(DimsElevation.Y - 1),
                                                                                                                                0.5f),
                                                                                                                DimsElevation.X),
                                                                                    DimsElevation.Y));
                float[] ElevationValues = Movie.GridCTFDefocus.GetInterpolated(ElevationPositions);

                //new Image(ElevationValues, new int3(DimsElevation)).WriteMRC("d_elevation.mrc");

                float MinValue = MathHelper.Min(ElevationValues);
                float MaxValue = MathHelper.Max(ElevationValues);
                float Range = MaxValue - MinValue;
                int ZeroR = Colors.DeepPink.R, ZeroG = Colors.DeepPink.G, ZeroB = Colors.DeepPink.B;
                int OneR = Colors.DeepSkyBlue.R - ZeroR, OneG = Colors.DeepSkyBlue.G - ZeroG, OneB = Colors.DeepSkyBlue.B - ZeroB;
                byte Alpha = 50;

                byte[] ElevationBytes = new byte[DimsElevation.Elements() * 4];
                for (int i = 0; i < ElevationValues.Length; i++)
                {
                    ElevationBytes[i * 4 + 3] = Alpha;
                    ElevationBytes[i * 4 + 2] = (byte)Math.Max(0, Math.Min(255, (ElevationValues[i] - MinValue) / Range * OneR + ZeroR));
                    ElevationBytes[i * 4 + 1] = (byte)Math.Max(0, Math.Min(255, (ElevationValues[i] - MinValue) / Range * OneG + ZeroG));
                    ElevationBytes[i * 4 + 0] = (byte)Math.Max(0, Math.Min(255, (ElevationValues[i] - MinValue) / Range * OneB + ZeroB));
                }

                ElevationImage = BitmapSource.Create(DimsElevation.X, DimsElevation.Y, 96, 96, PixelFormats.Bgra32, null, ElevationBytes, DimsElevation.X * 4);

                TextElevationMin.Text = MinValue.ToString("F3") + " μm";
                TextElevationMax.Text = MaxValue.ToString("F3") + " μm";

                PanelElevationSettings.Visibility = Visibility.Visible;
            }
            else
            {
                PanelElevationSettings.Visibility = Visibility.Collapsed;
            }

            ImageElevation.Source = ElevationImage;
        }

        public void PositionVisuals()
        {
            if (ImageDisplay.Source != null)
            {
                float ClampedScaleFactor = (float)Math.Max(1, ScaleFactor);

                ImageDisplay.Width = ImageDisplay.Source.Width * ClampedScaleFactor;
                ImageDisplay.Height = ImageDisplay.Source.Height * ClampedScaleFactor;

                ImageMask.Width = ImageDisplay.Width;
                ImageMask.Height = ImageDisplay.Height;

                CanvasTrack.Width = ImageDisplay.Source.Width * ClampedScaleFactor;
                CanvasTrack.Height = ImageDisplay.Source.Height * ClampedScaleFactor;

                CanvasTrackMouse.Width = ImageDisplay.Source.Width * ClampedScaleFactor;
                CanvasTrackMouse.Height = ImageDisplay.Source.Height * ClampedScaleFactor;

                float2 ScaledCenterOffset = VisualsCenterOffset * (float)ScaleFactor;
                float2 CanvasOrigin = (new float2((float)ScrollViewDisplay.ActualWidth, (float)ScrollViewDisplay.ActualHeight) -
                                       new float2((float)ImageDisplay.Width, (float)ImageDisplay.Height)) * 0.5f -
                                      ScaledCenterOffset;

                Canvas.SetLeft(PanelVisuals, CanvasOrigin.X);
                Canvas.SetTop(PanelVisuals, CanvasOrigin.Y);
            }
        }

        private void UpdateTrackGrid()
        {
            CanvasTrack.Children.Clear();

            if (ImageDisplay.Source == null || Movie == null || (Movie.GetType() == typeof (Movie) && Movie.OptionsMovement == null) || Movie.GetType() == typeof(TiltSeries))
            {
                PanelTrackSettings.Visibility = Visibility.Collapsed;
                return;
            }

            PanelTrackSettings.Visibility = Visibility.Visible;

            if (!TrackShow)
                return;

            double PixelSize = 1;
            if (Movie.GetType() == typeof(Movie) && Movie.OptionsMovieExport != null)
                PixelSize = (double)Movie.OptionsMovieExport.BinnedPixelSizeMean;

            double Scale = ScaleFactor * (double)TrackScale / PixelSize;

            Movie TempMovie = Movie;
            int2 GridDims = new int2(TrackGridX, TrackGridY);
            bool LocalOnly = TrackLocalOnly;

            Parallel.For(0, GridDims.Y, y =>
            {
                for (int x = 0; x < GridDims.X; x++)
                {
                    float2 NormalizedPosition = new float2((x + 0.5f) * 1f / GridDims.X, (y + 0.5f) * 1f / GridDims.Y);

                    float2[] TrackData = TempMovie.GetMotionTrack(NormalizedPosition, 1, LocalOnly);
                    if (TrackData == null)
                        continue;
                    IEnumerable<Point> TrackPoints = TrackData.Select(v => new Point((-v.X + TrackData[0].X) * Scale, -(-v.Y + TrackData[0].Y) * Scale));

                    // Construct path.
                    CanvasTrack.Dispatcher.InvokeAsync(() =>
                    {
                        Path TrackPath = new Path()
                        {
                            Stroke = new SolidColorBrush(Colors.White),
                            StrokeThickness = 2.0,
                            StrokeLineJoin = PenLineJoin.Bevel,
                            IsHitTestVisible = false
                        };
                        PolyLineSegment PlotSegment = new PolyLineSegment(TrackPoints, true);
                        PathFigure PlotFigure = new PathFigure
                        {
                            Segments = new PathSegmentCollection { PlotSegment },
                            StartPoint = TrackPoints.First()
                        };
                        TrackPath.Data = new PathGeometry { Figures = new PathFigureCollection { PlotFigure } };

                        CanvasTrack.Children.Add(TrackPath);
                        Canvas.SetLeft(TrackPath, NormalizedPosition.X * CanvasTrack.Width);
                        Canvas.SetTop(TrackPath, (1 - NormalizedPosition.Y) * CanvasTrack.Height);
                    });
                }
            });
        }

        private void UpdateTrackMouse(MouseEventArgs e)
        {
            CanvasTrackMouse.Children.Clear();

            if (BrushActive)
                return;

            if (e == null)
                return;
            if (!TrackShow || ImageDisplay.Source == null || Movie == null || (Movie.GetType() == typeof(Movie) && Movie.OptionsMovement == null) || !ImageDisplay.IsMouseOver)
                return;

            double PixelSize = 1;
            if (Movie.GetType() == typeof (Movie) && Movie.OptionsMovieExport != null)
                PixelSize = (double)Movie.OptionsMovieExport.BinnedPixelSizeMean;

            double Scale = ScaleFactor * (double)TrackScale / PixelSize;

            bool LocalOnly = TrackLocalOnly;

            // Get motion track at mouse position.
            {
                Point MousePos = e.GetPosition(CanvasTrackMouse);
                float2 NormalizedPosition = new float2((float)(MousePos.X / CanvasTrackMouse.Width),
                                                       (float)(1 - MousePos.Y / CanvasTrackMouse.Height));
                float2[] TrackData = Movie.GetMotionTrack(NormalizedPosition, 1, LocalOnly);
                if (TrackData == null)
                    return;
                Point[] TrackPoints = TrackData.Select(v => new Point((-v.X + TrackData[0].X) * Scale, -(-v.Y + TrackData[0].Y) * Scale)).ToArray();

                // Construct path.
                Path TrackPath = new Path()
                {
                    Stroke = new SolidColorBrush(Colors.DeepSkyBlue),
                    StrokeThickness = 2.5,
                    StrokeLineJoin = PenLineJoin.Round
                };
                PolyLineSegment PlotSegment = new PolyLineSegment(TrackPoints, true);
                PathFigure PlotFigure = new PathFigure
                {
                    Segments = new PathSegmentCollection { PlotSegment },
                    StartPoint = TrackPoints[0]
                };
                TrackPath.Data = new PathGeometry { Figures = new PathFigureCollection { PlotFigure } };

                var TrackShadow = new DropShadowEffect
                {
                    Opacity = 2,
                    Color = Colors.Black,
                    BlurRadius = 6,
                    ShadowDepth = 0,
                    RenderingBias = RenderingBias.Quality
                };
                TrackPath.Effect = TrackShadow;

                //TrackPath.PreviewMouseWheel += ImageDisplay_MouseWheel;
                //TrackPath.PreviewMouseMove += ImageDisplay_OnPreviewMouseMove;
                TrackPath.IsHitTestVisible = false;

                CanvasTrackMouse.Children.Add(TrackPath);
                Canvas.SetLeft(TrackPath, MousePos.X);
                Canvas.SetTop(TrackPath, MousePos.Y);

                for (int z = 0; z < TrackPoints.Length / 1 + 1; z++)
                {
                    Point DotPosition = TrackPoints[Math.Min(TrackPoints.Length - 1, z * 1)];
                    Ellipse Dot = new Ellipse()
                    {
                        Width = 4,
                        Height = 4,
                        Fill = new SolidColorBrush(Colors.DeepSkyBlue),
                        StrokeThickness = 0,
                        Effect = TrackShadow
                    };
                    //Dot.PreviewMouseWheel += ImageDisplay_MouseWheel;
                    //Dot.PreviewMouseMove += ImageDisplay_OnPreviewMouseMove;
                    Dot.IsHitTestVisible = false;

                    CanvasTrackMouse.Children.Add(Dot);
                    Canvas.SetLeft(Dot, MousePos.X + DotPosition.X - 2.0);
                    Canvas.SetTop(Dot, MousePos.Y + DotPosition.Y - 2.0);
                }
            }
        }

        #endregion

        #region Mouse interaction

        private void ImageDisplay_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (ImageDisplay.Source != null)
            {
                if (ImageDisplay.IsMouseOver && BrushActive && (Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl)))
                {
                    BrushDiameter = Math.Max(8, BrushDiameter + Math.Sign(e.Delta) * 10);
                    UpdateBrushEllipse(e);
                }
                else
                {
                    decimal NewZoom = Math.Max(0.05M, Math.Min(8, Zoom + Math.Sign(e.Delta) * 0.05M));
                    float NewScale = (float)NewZoom;
                    if (NewZoom == Zoom)
                        return;

                    float ScaleChange = NewScale / (float)ScaleFactor;

                    float2 ImageCenter = new float2((float)ImageDisplay.Width, (float)ImageDisplay.Height) / 2 + VisualsCenterOffset * (float)ScaleFactor;
                    Point PointAbsolute = e.GetPosition(ImageDisplay);
                    float2 RelativeToCenter = (new float2((float)PointAbsolute.X, (float)PointAbsolute.Y) - ImageCenter);
                    float2 ExpectedShift = RelativeToCenter * ScaleChange - RelativeToCenter;

                    Zoom = NewZoom;

                    VisualsCenterOffset += ExpectedShift / NewScale;

                    UpdateParticles();
                }
            }
        }

        private void ImageDisplay_OnPreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (!IsDraggingImage)
            {
                UpdateTrackMouse(e);
                UpdateBrushEllipse(e);
            }

            if (IsDraggingImage)
            {
                Point NewDraggingPoint = e.GetPosition(this);
                VisualsCenterOffset -= new float2((float)(NewDraggingPoint.X - DraggingStartPoint.X),
                                                  (float)(NewDraggingPoint.Y - DraggingStartPoint.Y)) / (float)ScaleFactor;
                DraggingStartPoint = NewDraggingPoint;
            }
            else if (IsDraggingParticle && DraggedParticle != null)
            {
                Point NewDraggingPoint = e.GetPosition(this);
                Vector Delta = NewDraggingPoint - DraggingStartPoint;
                DraggingStartPoint = NewDraggingPoint;

                Delta /= ScaleFactor;
                Delta.Y *= -1;

                DraggedParticle.Coordinates = new[] { DraggedParticle.CoordinatesMean + new float3((float)Delta.X, (float)Delta.Y, 0) };
                SaveParticles();

                UpdateParticles();
            }
            else if (BrushActive && IsDraggingBrush)
            {
                float2 Norm = new float2(MaskDims) / new float2((float)ImageDisplay.ActualWidth, (float)ImageDisplay.ActualHeight);

                float2 PaintStart = new float2((float)DraggingStartPoint.X, (float)(ImageDisplay.ActualHeight - 1 - DraggingStartPoint.Y)) * Norm;
                Point DraggingEndPoint = e.GetPosition(ImageDisplay);
                float2 PaintEnd = new float2((float)DraggingEndPoint.X, (float)(ImageDisplay.ActualHeight - 1 - DraggingEndPoint.Y)) * Norm;

                DraggingStartPoint = DraggingEndPoint;

                if (PaintStart == PaintEnd)
                    PaintEnd.X -= 1e-4f;

                PaintMaskLine(PaintStart, PaintEnd, BrushDiameter / 2f / 8f, Mouse.LeftButton == MouseButtonState.Pressed ? (byte)1 : (byte)0);
            }
            else if (Mouse.RightButton == MouseButtonState.Pressed)
            {
                Point PointInvY = e.GetPosition(CanvasParticles);
                PointInvY.Y = CanvasParticles.ActualHeight - 1 - PointInvY.Y;
                Tuple<Particle, double> ClosestParticle = GetClosestParticle(PointInvY);

                if (ClosestParticle.Item1 != null)
                {
                    Particles.Remove(ClosestParticle.Item1);
                    ParticlesBad.Add(ClosestParticle.Item1);
                    SaveParticles();

                    if (ParticlesBad.Count >= 1)
                        PopulateValidSuffixes();

                    UpdateParticles();
                }
            }
        }

        private void MicrographDisplay_MouseLeave(object sender, MouseEventArgs e)
        {
            CanvasTrackMouse.Children.Clear();
        }

        private void ImageDisplay_OnMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (!BrushActive || Mouse.MiddleButton == MouseButtonState.Pressed)
            {
                Point PointInvY = e.GetPosition(CanvasParticles);
                PointInvY.Y = CanvasParticles.ActualHeight - 1 - PointInvY.Y;
                Tuple<Particle, double> ClosestParticle = GetClosestParticle(PointInvY);

                if (Mouse.LeftButton == MouseButtonState.Pressed || Mouse.MiddleButton == MouseButtonState.Pressed)
                {
                    if (Mouse.LeftButton == MouseButtonState.Pressed && ImageDisplay.IsMouseOver)
                    {
                        if (!CanEditParticles() && File.Exists(Movie.AveragePath)) // Create a .star for manual picking
                        {
                            if (!Directory.Exists(Movie.MatchingDir))
                                Directory.CreateDirectory(Movie.MatchingDir);
                            ParticlesSuffix = "_manual";
                            SaveParticles();

                            ParticlesShow = true;
                            ParticlesThreshold = 0;

                            string[] AvailableSuffixes = DiscoverValidSuffixes();
                            PopulateValidSuffixes();

                            if (AvailableSuffixes.Length > 0)
                            {
                                PanelParticleSettings.Visibility = Visibility.Visible;

                                if (ParticlesSuffix == null)
                                    ParticlesSuffix = AvailableSuffixes[0];
                                else if (!AvailableSuffixes.Contains(ParticlesSuffix))
                                    ParticlesSuffix = AvailableSuffixes[0];
                            }

                            if (File.Exists(Movie.AveragePath))
                            {
                                MapHeader Header = MapHeader.ReadFromFile(Movie.AveragePath);
                                PixelSize = Header.PixelSize.X;
                            }
                        }

                        if (ParticlesShow)
                        {
                            if (ClosestParticle.Item1 == null)
                            {
                                Particle NewParticle = new Particle(new[] { new float3((float)(PointInvY.X / ScaleFactor), (float)(PointInvY.Y / ScaleFactor), 0) },
                                                                    new[] { new float3(0) },
                                                                    Particles.Count % 2,
                                                                    "",
                                                                    "",
                                                                    Particles.Count > 0 ? MathHelper.Max(Particles.Select(p => p.FOM)) : 0);
                                Particles.Add(NewParticle);
                                SaveParticles();

                                ClosestParticle = new Tuple<Particle, double>(NewParticle, 0);

                                UpdateParticles();
                            }

                            IsDraggingParticle = true;
                            DraggedParticle = ClosestParticle.Item1;
                        }
                    }
                    else if (Mouse.MiddleButton == MouseButtonState.Pressed || (Mouse.LeftButton == MouseButtonState.Pressed && !ParticlesShow))
                    {
                        IsDraggingImage = true;
                    }

                    DraggingStartPoint = e.GetPosition(this);
                    e.Handled = true;
                }
                else if (Mouse.RightButton == MouseButtonState.Pressed)
                {
                    if (ParticlesShow && ClosestParticle.Item1 != null)
                    {
                        Particles.Remove(ClosestParticle.Item1);
                        if (!Keyboard.IsKeyDown(Key.LeftShift) && !Keyboard.IsKeyDown(Key.RightShift))
                            ParticlesBad.Add(ClosestParticle.Item1);
                        SaveParticles();

                        UpdateParticles();
                    }
                }
            }
            else if (Mouse.LeftButton == MouseButtonState.Pressed || Mouse.RightButton == MouseButtonState.Pressed)
            {
                DraggingStartPoint = e.GetPosition(ImageDisplay);
                IsDraggingBrush = true;
                e.Handled = true;

                float2 Norm = new float2(MaskDims) / new float2((float)ImageDisplay.ActualWidth, (float)ImageDisplay.ActualHeight);

                float2 PaintStart = new float2((float)DraggingStartPoint.X, (float)(ImageDisplay.ActualHeight - 1 - DraggingStartPoint.Y)) * Norm;
                Point DraggingEndPoint = e.GetPosition(ImageDisplay);
                float2 PaintEnd = new float2((float)DraggingEndPoint.X, (float)(ImageDisplay.ActualHeight - 1 - DraggingEndPoint.Y)) * Norm;

                if (PaintStart == PaintEnd)
                    PaintEnd.X -= 1e-4f;

                PaintMaskLine(PaintStart, PaintEnd, BrushDiameter / 2f / 8f, Mouse.LeftButton == MouseButtonState.Pressed ? (byte)1 : (byte)0);
            }
        }

        private void ImageDisplay_OnMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (IsDraggingBrush)
            {
                float2 Norm = new float2(MaskDims) / new float2((float)ImageDisplay.ActualWidth, (float)ImageDisplay.ActualHeight);

                float2 PaintStart = new float2((float)DraggingStartPoint.X, (float)(ImageDisplay.ActualHeight - 1 - DraggingStartPoint.Y)) * Norm;
                Point DraggingEndPoint = e.GetPosition(ImageDisplay);
                float2 PaintEnd = new float2((float)DraggingEndPoint.X, (float)(ImageDisplay.ActualHeight - 1 - DraggingEndPoint.Y)) * Norm;

                if (PaintStart == PaintEnd)
                    PaintEnd.X -= 1e-4f;

                PaintMaskLine(PaintStart, PaintEnd, BrushDiameter / 2f / 8f, e.ChangedButton == MouseButton.Left ? (byte)1 : (byte)0);

                SaveMask();
            }

            IsDraggingBrush = false;
            IsDraggingImage = false;
            IsDraggingParticle = false;
            DraggedParticle = null;
            e.Handled = true;
        }

        #endregion

        #region Particles

        private void ParticlesSuffixChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args)
        {
            LoadParticles();

            if (Movie.PickingThresholds.ContainsKey(ParticlesSuffix))
                ParticlesThreshold = Movie.PickingThresholds[ParticlesSuffix];
            else if (Particles.Count > 0)
                ParticlesThreshold = (decimal)Particles.Select(p => p.FOM).Min();

            UpdateParticles();
        }

        private void ParticlesSettingsChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args)
        {
            UpdateParticles();
        }

        private void ParticlesBlinkyChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args)
        {
            if (BlinkyTimer.IsEnabled && !ParticlesBlinky)
            {
                BlinkyTimer.Stop();
                UpdateParticles();
            }
            else if (!BlinkyTimer.IsEnabled && ParticlesBlinky)
            {
                UpdateParticles();
                BlinkyTimer.Start();
                BlinkyOn = false;
            }
        }

        private void LoadParticles()
        {
            Particles.Clear();
            ParticlesBad.Clear();

            if (!CanEditParticles())
                return;

            float DownsampleFactor = 1;
            //if (Movie.OptionsMovieExport != null)
            //    DownsampleFactor = (float)Movie.OptionsMovieExport.DownsampleFactor;

            {
                Star TableIn = new Star(System.IO.Path.Combine(Movie.MatchingDir, Movie.RootName + ParticlesSuffix + ".star"));
                if (!TableIn.HasColumn("rlnCoordinateX") || !TableIn.HasColumn("rlnCoordinateY"))
                    return;

                float[] ColumnCoordX = TableIn.GetColumn("rlnCoordinateX").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                float[] ColumnCoordY = TableIn.GetColumn("rlnCoordinateY").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

                float[] ColumnAngleRot = TableIn.HasColumn("rlnAngleRot") ? TableIn.GetColumn("rlnAngleRot").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : null;
                float[] ColumnAngleTilt = TableIn.HasColumn("rlnAngleTilt") ? TableIn.GetColumn("rlnAngleTilt").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : null;
                float[] ColumnAnglePsi = TableIn.HasColumn("rlnAnglePsi") ? TableIn.GetColumn("rlnAnglePsi").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : null;

                float[] ColumnScore = TableIn.HasColumn("rlnAutopickFigureOfMerit") ? TableIn.GetColumn("rlnAutopickFigureOfMerit").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : null;

                for (int r = 0; r < TableIn.RowCount; r++)
                {
                    float3 Position = new float3(ColumnCoordX[r], ColumnCoordY[r], 0) / DownsampleFactor;
                    float3 Angle = new float3(ColumnAngleRot?[r] ?? 0,
                                              ColumnAngleTilt?[r] ?? 0,
                                              ColumnAnglePsi?[r] ?? 0);
                    float Score = ColumnScore?[r] ?? 0;

                    Particles.Add(new Particle(new[] { Position }, new[] { Angle }, Particles.Count % 2, "", "", Score));
                }
            }

            if (!ParticlesSuffix.Contains("falsepositives") && File.Exists(System.IO.Path.Combine(Movie.MatchingDir, Movie.RootName + ParticlesSuffix + "_falsepositives.star")))
            {
                Star TableIn = new Star(System.IO.Path.Combine(Movie.MatchingDir, Movie.RootName + ParticlesSuffix + "_falsepositives.star"));
                if (!TableIn.HasColumn("rlnCoordinateX") || !TableIn.HasColumn("rlnCoordinateY"))
                    return;

                float[] ColumnCoordX = TableIn.GetColumn("rlnCoordinateX").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                float[] ColumnCoordY = TableIn.GetColumn("rlnCoordinateY").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

                float[] ColumnAngleRot = TableIn.HasColumn("rlnAngleRot") ? TableIn.GetColumn("rlnAngleRot").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : null;
                float[] ColumnAngleTilt = TableIn.HasColumn("rlnAngleTilt") ? TableIn.GetColumn("rlnAngleTilt").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : null;
                float[] ColumnAnglePsi = TableIn.HasColumn("rlnAnglePsi") ? TableIn.GetColumn("rlnAnglePsi").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : null;

                float[] ColumnScore = TableIn.HasColumn("rlnAutopickFigureOfMerit") ? TableIn.GetColumn("rlnAutopickFigureOfMerit").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : null;

                for (int r = 0; r < TableIn.RowCount; r++)
                {
                    float3 Position = new float3(ColumnCoordX[r], ColumnCoordY[r], 0) / DownsampleFactor;
                    float3 Angle = new float3(ColumnAngleRot?[r] ?? 0,
                                              ColumnAngleTilt?[r] ?? 0,
                                              ColumnAnglePsi?[r] ?? 0);
                    float Score = ColumnScore?[r] ?? 0;

                    ParticlesBad.Add(new Particle(new[] { Position }, new[] { Angle }, Particles.Count % 2, "", "", Score));
                }
            }
        }

        private void SaveParticles()
        {
            if (!(Movie != null && Movie.GetType() == typeof(Movie) && !string.IsNullOrEmpty(ParticlesSuffix)))
                return;

            float DownsampleFactor = 1;
            //if (Movie.OptionsMovieExport != null)
            //    DownsampleFactor = (float)Movie.OptionsMovieExport.DownsampleFactor;

            {
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

                foreach (var particle in Particles)
                {
                    var Row = new string[]
                    {
                        (particle.CoordinatesMean.X * DownsampleFactor).ToString(CultureInfo.InvariantCulture),
                        (particle.CoordinatesMean.Y * DownsampleFactor).ToString(CultureInfo.InvariantCulture),
                        particle.AnglesMean.X.ToString(CultureInfo.InvariantCulture),
                        particle.AnglesMean.Y.ToString(CultureInfo.InvariantCulture),
                        particle.AnglesMean.Z.ToString(CultureInfo.InvariantCulture),
                        Movie.RootName + ".mrc",
                        particle.FOM.ToString(CultureInfo.InvariantCulture)
                    };

                    TableOut.AddRow(Row);
                }

                TableOut.Save(System.IO.Path.Combine(Movie.MatchingDir, Movie.RootName + ParticlesSuffix + ".star"));

                Movie.UpdateParticleCount(ParticlesSuffix, Particles.Count);
            }

            if (!ParticlesSuffix.Contains("falsepositives") && ParticlesBad.Count > 0)
            {
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

                foreach (var particle in ParticlesBad)
                {
                    string[] Row = new string[]
                    {
                        (particle.CoordinatesMean.X * DownsampleFactor).ToString(CultureInfo.InvariantCulture),
                        (particle.CoordinatesMean.Y * DownsampleFactor).ToString(CultureInfo.InvariantCulture),
                        particle.AnglesMean.X.ToString(CultureInfo.InvariantCulture),
                        particle.AnglesMean.Y.ToString(CultureInfo.InvariantCulture),
                        particle.AnglesMean.Z.ToString(CultureInfo.InvariantCulture),
                        Movie.RootName + ".mrc",
                        particle.FOM.ToString(CultureInfo.InvariantCulture)
                    };

                    TableOut.AddRow(Row);
                }

                TableOut.Save(System.IO.Path.Combine(Movie.MatchingDir, Movie.RootName + ParticlesSuffix + "_falsepositives.star"));

                Movie.UpdateParticleCount(ParticlesSuffix + "_falsepositives", ParticlesBad.Count);
            }

            Movie.OnParticlesChanged();
        }

        private bool CanEditParticles()
        {
            return Movie != null && Movie.GetType() == typeof(Movie) && !string.IsNullOrEmpty(ParticlesSuffix) && File.Exists(System.IO.Path.Combine(Movie.MatchingDir, Movie.RootName + ParticlesSuffix + ".star"));
        }

        private string[] DiscoverValidSuffixes()
        {
            List<string> Result = new List<string>();

            if (Movie != null && Movie.GetType() == typeof(Movie) && Directory.Exists(Movie.MatchingDir))
                foreach (var file in Directory.EnumerateFiles(Movie.MatchingDir, Movie.RootName + "*.star"))
                {
                    string Suffix = file.Substring(file.LastIndexOf(Movie.RootName) + Movie.RootName.Length);
                    Suffix = Suffix.Substring(0, Suffix.Length - ".star".Length);
                    Result.Add(Suffix);
                }

            return Result.ToArray();
        }

        private void PopulateValidSuffixes()
        {
            string[] AvailableSuffixes = DiscoverValidSuffixes();
            MenuParticlesSuffix.Items.Clear();
            foreach (var suffix in AvailableSuffixes)
                MenuParticlesSuffix.Items.Add(suffix);
        }

        private Tuple<Particle, double> GetClosestParticle(Point position)
        {
            Particle ClosestParticle = null;
            double ClosestParticleDistance = double.MaxValue;
            if (ParticlesShow && Particles != null)
            {
                foreach (var particle in Particles)
                {
                    if (particle.FOM < (float)ParticlesThreshold)
                        continue;

                    Point ParticlePosition = new Point(particle.CoordinatesMean.X * ScaleFactor, particle.CoordinatesMean.Y * ScaleFactor);
                    double Distance = (ParticlePosition - position).Length;
                    if (Distance < ClosestParticleDistance)
                    {
                        ClosestParticleDistance = Distance;
                        ClosestParticle = particle;
                    }
                }
            }

            if (ClosestParticleDistance > (double)ParticlesDiameter / 2 / PixelSize * ScaleFactor)
                ClosestParticle = null;

            return new Tuple<Particle, double>(ClosestParticle, ClosestParticleDistance);
        }

        private void UpdateParticles()
        {
            CanvasParticles.Children.Clear();
            //ImageDisplay.ToolTip = "MIDDLE: Pan\nSCROLL: Zoom";

            if (BlinkyTimer.IsEnabled && !BlinkyOn)
                return;

            ParticlesCount = Particles?.Count(p => p.FOM >= (float)ParticlesThreshold) ?? 0;

            if (!ParticlesShow || Particles == null || Particles.Count == 0 || Movie == null || Movie.GetType() != typeof(Movie) || !File.Exists(Movie.AveragePath))
                return;

            //ImageDisplay.ToolTip = "LEFT: Add/move particle\nMIDDLE: Pan view\nSCROLL: Zoom\nRIGHT: Delete particle";

            float MinFOM = Particles.Where(p => p.FOM >= (float)ParticlesThreshold).Select(p => p.FOM).Min();
            float MaxFOM = Particles.Where(p => p.FOM >= (float)ParticlesThreshold).Select(p => p.FOM).Max();
            float RangeFOM = Math.Max(1e-6f, MaxFOM - MinFOM);
            double PreferredFOMHeight = (double)ParticlesDiameter / PixelSize * ScaleFactor / 3;
            float3 Color1 = new float3(20 / 360f, 1, 0.5f);
            float3 Color0 = new float3(180 / 360f, 1, 0.5f);
            byte Alpha = 255;
            DropShadowEffect TextShadow = new DropShadowEffect() { BlurRadius = 4, Color = Colors.Black, Opacity = 1, ShadowDepth = 0 };

            SolidColorBrush CircleBrush = new SolidColorBrush(ParticleCircleBrush);
            CircleBrush.Freeze();

            bool AreParticlesCircles = ParticleCircleStyle == "Circles";

            foreach (var particle in Particles)
            {
                if (particle.FOM < (float)ParticlesThreshold)
                    continue;

                float3 Position = particle.CoordinatesMean;

                if (AreParticlesCircles)
                {
                    Ellipse ParticleCircle = new Ellipse
                    {
                        Width = Math.Max(1, (double)ParticlesDiameter / PixelSize * ScaleFactor),
                        Height = Math.Max(1, (double)ParticlesDiameter / PixelSize * ScaleFactor),
                        Stroke = CircleBrush,
                        StrokeThickness = 2,
                        IsHitTestVisible = false
                    };

                    CanvasParticles.Children.Add(ParticleCircle);
                    Canvas.SetLeft(ParticleCircle, Position.X * ScaleFactor - ParticleCircle.Width / 2);
                    Canvas.SetBottom(ParticleCircle, Position.Y * ScaleFactor - ParticleCircle.Height / 2);
                }
                else
                {
                    Ellipse ParticleCircle = new Ellipse
                    {
                        Width = 6,
                        Height = 6,
                        Fill = CircleBrush,
                        StrokeThickness = 0,
                        IsHitTestVisible = false
                    };

                    CanvasParticles.Children.Add(ParticleCircle);
                    Canvas.SetLeft(ParticleCircle, Position.X * ScaleFactor - 3);
                    Canvas.SetBottom(ParticleCircle, Position.Y * ScaleFactor - 3);
                }

                if (ParticlesShowFOM)
                {
                    float3 ColorFloat = float3.HSL2RGB(float3.Lerp(Color0, Color1, (particle.FOM - MinFOM) / RangeFOM)) * 255f;
                    Size TextSize = MeasureString(particle.FOM.ToString("F2", CultureInfo.InvariantCulture), PreferredFOMHeight);
                    Color TextColor = Color.FromArgb(Alpha,
                                                     (byte)Math.Max(0, Math.Min(255, ColorFloat.X)),
                                                     (byte)Math.Max(0, Math.Min(255, ColorFloat.Y)),
                                                     (byte)Math.Max(0, Math.Min(255, ColorFloat.Z)));
                    Brush TextBrush = new SolidColorBrush(TextColor);
                    TextBrush.Freeze();

                    TextBlock FOMText = new TextBlock()
                    {
                        FontSize = PreferredFOMHeight,
                        Text = particle.FOM.ToString("F2", CultureInfo.InvariantCulture),
                        FontWeight = FontWeights.Bold,
                        Foreground = TextBrush,
                        Effect = TextShadow
                    };

                    CanvasParticles.Children.Add(FOMText);
                    Canvas.SetLeft(FOMText, Position.X * ScaleFactor - TextSize.Width / 2);
                    Canvas.SetBottom(FOMText, Position.Y * ScaleFactor - TextSize.Height / 2);
                }
            }
        }

        private void MenuParticlesSuffix_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            
        }

        private void SliderParticlesThreshold_OnValueChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (CanEditParticles())
            {
                if (!Movie.PickingThresholds.ContainsKey(ParticlesSuffix))
                    Movie.PickingThresholds.Add(ParticlesSuffix, 0);
                Movie.PickingThresholds[ParticlesSuffix] = ParticlesThreshold;

                Movie.SaveMeta();
            }
        }

        private async void ButtonParticlesThresholdApplyToAll_OnClick(object sender, RoutedEventArgs e)
        {
            if (!CanEditParticles())
                return;

            var DialogResult = await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Apply threshold to all micrographs",
                                                                                                   $"The current threshold ({ParticlesThreshold}) will be applied to all micrographs.\nThis step will overwrite any previously saved thresholds.\nWould you like to continue?",
                                                                                                   MessageDialogStyle.AffirmativeAndNegative);

            if (DialogResult == MessageDialogResult.Affirmative)
            {
                var ProgressDialog = await ((MainWindow)Application.Current.MainWindow).ShowProgressAsync("Applying picking thresholds...", "");

                Movie[] ImmutableItems = ((MainWindow)Application.Current.MainWindow).FileDiscoverer.GetImmutableFiles();
                ProgressDialog.Maximum = ImmutableItems.Length;

                string CurrentSuffix = ParticlesSuffix;
                decimal CurrentThreshold = ParticlesThreshold;

                await Task.Run(() =>
                {
                    int p = 0;
                    foreach (var item in ImmutableItems)
                    {
                        if (item.GetType() == typeof(Movie))
                        {
                            if (!item.PickingThresholds.ContainsKey(CurrentSuffix))
                                item.PickingThresholds.Add(CurrentSuffix, 0);
                            item.PickingThresholds[CurrentSuffix] = CurrentThreshold;
                            item.SaveMeta();
                        }
                        Dispatcher.Invoke(() => ProgressDialog.SetProgress(++p));
                    }
                });

                await ProgressDialog.CloseAsync();
            }
        }

        private void SliderParticlesThreshold_OnMouseEnter(object sender, MouseEventArgs e)
        {
            ParticlesShowFOM = true;
        }

        private void SliderParticlesThreshold_OnMouseLeave(object sender, MouseEventArgs e)
        {
            ParticlesShowFOM = false;
        }

        private void PanelParticleCircleColor_OnMouseDown(object sender, MouseButtonEventArgs e)
        {
            PopupParticleCircleColor.PlacementTarget = PanelParticleCircleColor;
            PopupParticleCircleColor.IsOpen = true;
        }

        #endregion

        #region BoxNet and NoiseNet

        private async void ButtonBoxNetPick_OnClick(object sender, RoutedEventArgs e)
        {
            if (Movie == null || !File.Exists(Movie.AveragePath))
                return;

            ProcessingOptionsBoxNet Options = WarpRuntime.MainWindow.Options.GetProcessingBoxNet();
            Movie _Movie = Movie;

            string ModelPath = ((MainWindow)Application.Current.MainWindow).LocatePickingModel(Options.ModelName);// @"G:\boxnetv2\boxnet_model_trained\BoxNet_20180514_000415";
            if (string.IsNullOrEmpty(ModelPath))
            {
                await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", $"Model directory \"{Options.ModelName}\" does not exist.");
                return;
            }

            var Dialog = await ((MainWindow)Application.Current.MainWindow).ShowProgressAsync($"Running micrograph through {Options.ModelName}...", "");
            Dialog.SetIndeterminate();

            await Task.Run(async () =>
            {
                if (BoxNetworks == null || ModelPath != BoxNetworksModelPath)
                {
                    Dialog.SetMessage($"Loading {Options.ModelName} model...");

                    DropBoxNetworks();
                    PopulateBoxNetworks(ModelPath);

                    Dialog.SetMessage($"Running micrograph through {Options.ModelName}...");
                }

                _Movie.MatchBoxNet2(BoxNetworks, Options, (size, value, name) =>
                {
                    Dispatcher.Invoke(() =>
                    {
                        Dialog.Maximum = size.Elements();
                        Dialog.SetProgress(value);
                    });

                    return false;
                });
            });

            await Dialog.CloseAsync();

            string[] AvailableSuffixes = DiscoverValidSuffixes();
            MenuParticlesSuffix.Items.Clear();
            foreach (var suffix in AvailableSuffixes)
                MenuParticlesSuffix.Items.Add(suffix);

            ParticlesSuffix = "_" + Helper.PathToNameWithExtension(Options.ModelName);
            PanelParticleSettings.Visibility = Visibility.Visible;

            LoadParticles();

            if (Particles.Count > 0)
                ParticlesThreshold = (decimal)Particles.Select(p => p.FOM).Min();
            ParticlesShow = true;
            UpdateParticles();

            ReloadMask();
            MaskShow = true;
            UpdateMask();

            WarpRuntime.MainWindow.UpdateStatsAll();
        }

        private void PopulateBoxNetworks(string modelPath)
        {
            FreeOnDevice();
            Image.FreeDeviceAll();

            BoxNetworks = new[] { new BoxNetTorch(BoxNetTorch.DefaultDimensionsPredict, new float[] { 1f, 1f, 1f }, new int[] { GPU.GetDeviceWithMostMemory() }, BoxNetTorch.DefaultBatchPredict) };
            foreach (var net in BoxNetworks)
                net.Load(modelPath);

            BoxNetworksModelPath = modelPath;
        }

        public void DropBoxNetworks()
        {
            if (BoxNetworks == null)
                return;

            foreach (var network in BoxNetworks)
                network.Dispose();
            BoxNetworks = null;
            BoxNetworksModelPath = "";
        }

        private async Task<NoiseNet2DTorch[]> PopulateNoiseNetworks(string modelDir)
        {
            ProgressDialogController Progress = null;
            await Dispatcher.Invoke(async () =>
            {
                Progress = await ((MainWindow)Application.Current.MainWindow).ShowProgressAsync("Loading NoiseNet model...", "");
                Progress.SetIndeterminate();
            });

            FreeOnDevice();
            Image.FreeDeviceAll();

            await Task.Run(() =>
            {
                NoiseNetworks = new[] { new NoiseNet2DTorch(new int2(256), new[] { GPU.GetDeviceWithMostMemory() }, 1) };
                foreach (var net in NoiseNetworks)
                    net.Load(modelDir);

                NoiseNetworksModelDir = modelDir;
            });

            await Dispatcher.Invoke(async () => await Progress.CloseAsync());

            return NoiseNetworks;
        }

        public void DropNoiseNetworks()
        {
            if (NoiseNetworks == null)
                return;

            foreach (var network in NoiseNetworks)
                network.Dispose();
            NoiseNetworks = null;
            NoiseNetworksModelDir = "";
        }

        public void UpdateBoxNetName(string name)
        {
            ButtonBoxNetPick.Content = ("Pick with " + name).ToUpper();
        }

        private async void ButtonDeconvRetrain_Click(object sender, RoutedEventArgs e)
        {
            Movie[] Movies = ((MainWindow)Application.Current.MainWindow).FileDiscoverer.GetImmutableFiles();

            List<Movie> WithExamples = new List<Movie>();
            foreach (var movie in Movies)
                if (File.Exists(movie.DenoiseTrainingOddPath) && File.Exists(movie.DenoiseTrainingEvenPath))
                    WithExamples.Add(movie);

            //WithExamples = WithExamples.Take(128).ToList();

            bool Proceed = false;

            if (WithExamples.Count < 10)
            {
                await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Not enough training examples found", 
                                                                                    "You need training data from at least 10 movies for retraining, although more is recommended.\n" +
                                                                                    "Training data are generated automatically as you process movies with 'Average' activated in\n" +
                                                                                    "the Output options. Examples cannot be generated from input data with fewer than 2 frames.", 
                                                                                    MessageDialogStyle.Affirmative);
            }
            else
            {
                var Result = await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Ready to start retraining", 
                                                                                                 $"Retraining will use examples from {WithExamples.Count} movies. Proceed?", 
                                                                                                 MessageDialogStyle.AffirmativeAndNegative);
                if (Result == MessageDialogResult.Affirmative)
                    Proceed = true;
            }

            if (!Proceed)
                return;

            var ProgressDialog = await ((MainWindow)Application.Current.MainWindow).ShowProgressAsync("Preparing training examples...", "");

            await Task.Run(() =>
            {
                int Dim = 256;
                int BatchSize = 16;

                DropNoiseNetworks();

                GPU.SetDevice(0);

                List<Image> MapsEven = new List<Image>();
                List<Image> MapsOdd = new List<Image>();

                Dispatcher.Invoke(() => ProgressDialog.SetProgress(0));

                int Loaded = 0;
                foreach (Movie movie in WithExamples)
                {
                    Image OddImage = Image.FromFile(movie.DenoiseTrainingOddPath);
                    float2 StdMean;
                    {
                        Image OddImageCenter = OddImage.AsPadded(new int2(OddImage.Dims) / 2);
                        StdMean = MathHelper.MeanAndStd(OddImageCenter.GetHost(Intent.Read)[0]);
                        OddImageCenter.Dispose();
                    }

                    if (StdMean.X == 0 || StdMean.Y == 0)
                        continue;

                    {
                        if (OddImage.GetHost(Intent.Read)[0].Any(v => float.IsNaN(v) || float.IsInfinity(v)))
                            continue;
                        OddImage.TransformValues(v => Math.Max(-5, Math.Min((v) / StdMean.Y, 5)));
                        //OddImage.TransformValues(v => (v - StdMean.X) / StdMean.Y);

                        GPU.PrefilterForCubic(OddImage.GetDevice(Intent.ReadWrite), OddImage.Dims);
                        OddImage.FreeDevice();
                    }
                    
                    Image EvenImage = Image.FromFile(movie.DenoiseTrainingEvenPath);
                    float2 OtherStdMean;
                    {
                        Image EvenImageCenter = EvenImage.AsPadded(new int2(OddImage.Dims) / 2);
                        OtherStdMean = MathHelper.MeanAndStd(EvenImageCenter.GetHost(Intent.Read)[0]);
                        EvenImageCenter.Dispose();
                    }
                    {
                        if (EvenImage.GetHost(Intent.Read)[0].Any(v => float.IsNaN(v) || float.IsInfinity(v)))
                            continue;
                        EvenImage.TransformValues(v => Math.Max(-5, Math.Min((v) / StdMean.Y, 5)));
                        //EvenImage.TransformValues(v => (v - OtherStdMean.X) / OtherStdMean.Y);

                        GPU.PrefilterForCubic(EvenImage.GetDevice(Intent.ReadWrite), EvenImage.Dims);
                        EvenImage.FreeDevice();
                    }

                    MapsOdd.Add(OddImage);
                    MapsEven.Add(EvenImage);

                    Loaded++;
                    Dispatcher.Invoke(() => ProgressDialog.SetProgress(Loaded / (float)WithExamples.Count));
                }

                Dispatcher.Invoke(() =>
                {
                    ProgressDialog.SetIndeterminate();
                    ProgressDialog.SetTitle("Loading default model...");
                });

                FreeOnDevice();
                Image.FreeDeviceAll();

                TorchSharp.Torch.SetSeed(123);
                NoiseNet2DTorch TrainModel = new NoiseNet2DTorch(new int2(Dim), new[] { GPU.GetDeviceWithMostMemory() }, BatchSize);
                //TrainModel.Load("NoiseNet2D_default.pt");

                GPU.SetDevice(0);

                Random Rand = new Random(123);

                int NMaps = MapsOdd.Count;
                int NMapsPerEpoch = Math.Min(32, NMaps);
                int MapSamples = BatchSize;
                int NEpochs = 1000;

                (ulong texture, ulong array)[] TexturesOdd = MapsOdd.Select(m =>
                {
                    ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                    GPU.CreateTexture3D(m.GetDevice(Intent.Read), m.Dims, Texture, TextureArray, true);
                    m.FreeDevice();

                    return (Texture[0], TextureArray[0]);
                }).ToArray();

                (ulong texture, ulong array)[] TexturesEven = MapsEven.Select(m =>
                {
                    ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                    GPU.CreateTexture3D(m.GetDevice(Intent.Read), m.Dims, Texture, TextureArray, true);
                    m.FreeDevice();

                    return (Texture[0], TextureArray[0]);
                }).ToArray();

                Image[] ExtractedOdd = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, MapSamples)), NMapsPerEpoch);
                Image[] ExtractedEven = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, MapSamples)), NMapsPerEpoch);

                Image[] ExtractedOddRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, MapSamples)), NMapsPerEpoch);
                Image[] ExtractedEvenRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, MapSamples)), NMapsPerEpoch);

                Dispatcher.Invoke(() =>
                {
                    ProgressDialog.SetProgress(0);
                    ProgressDialog.SetTitle("Retraining...");
                });

                for (int iepoch = 0; iepoch < NEpochs; iepoch++)
                {
                    int[] ShuffledMapIDs = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMaps, 1), NMapsPerEpoch, Rand.Next(9999));

                    for (int m = 0; m < NMapsPerEpoch; m++)
                    {
                        int MapID = ShuffledMapIDs[m];

                        Image MapOdd = MapsOdd[MapID];
                        Image MapEven = MapsEven[MapID];

                        int3 DimsMap = MapEven.Dims;

                        int3 Margin = new int3((int)(Dim / 2));
                        Margin.Z = 0;
                        float3[] Position = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * (DimsMap.X - Margin.X * 2) + Margin.X,
                                                                                   (float)Rand.NextDouble() * (DimsMap.Y - Margin.Y * 2) + Margin.Y,
                                                                                   0), MapSamples);

                        float3[] Angle = Helper.ArrayOfFunction(i => new float3(0,
                                                                                0,
                                                                                (float)Rand.NextDouble() * 360) * Helper.ToRad, MapSamples);

                        //float[] StdFudge = Helper.ArrayOfFunction(i => (float)Rand.NextDouble() * 0.8f + 0.6f, MapSamples);
                        //float MeanFudge = (float)Rand.NextDouble() * 0.5f - 0.25f;

                        {
                            //ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                            //GPU.CreateTexture3D(MapEven.GetDevice(Intent.Read), MapEven.Dims, Texture, TextureArray, true);
                            //MapEven.FreeDevice();

                            GPU.Rotate3DExtractAt(TexturesEven[MapID].texture,
                                                  MapEven.Dims,
                                                  ExtractedOdd[m].GetDevice(Intent.Write),
                                                  new int3(Dim, Dim, 1),
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedOdd[m].Multiply(StdFudge);
                            //ExtractedOdd[m].Add(MeanFudge);

                            //ExtractedSource[MapID].WriteMRC("d_extractedsource.mrc", true);

                            //GPU.DestroyTexture(Texture[0], TextureArray[0]);
                        }

                        {
                            //ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                            //GPU.CreateTexture3D(MapOdd.GetDevice(Intent.Read), MapOdd.Dims, Texture, TextureArray, true);
                            //MapOdd.FreeDevice();

                            GPU.Rotate3DExtractAt(TexturesOdd[MapID].texture,
                                                  MapOdd.Dims,
                                                  ExtractedEven[m].GetDevice(Intent.Write),
                                                  new int3(Dim, Dim, 1),
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedEven[m].Multiply(StdFudge);
                            //ExtractedEven[m].Add(MeanFudge);

                            //ExtractedTarget.WriteMRC("d_extractedtarget.mrc", true);

                            //GPU.DestroyTexture(Texture[0], TextureArray[0]);
                        }

                        //ExtractedOdd[m].Add(ExtractedEven[m]);
                        //ExtractedOdd[m].Multiply(0.5f);

                        //MapEven.FreeDevice();
                        //MapOdd.FreeDevice();
                    }

                    // Shuffle individual examples between batches so each batch doesn't source from only one map
                    for (int b = 0; b < MapSamples; b++)
                    {
                        int[] Order = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMapsPerEpoch, 1), NMapsPerEpoch, Rand.Next(9999999));
                        for (int i = 0; i < Order.Length; i++)
                        {
                            GPU.CopyDeviceToDevice(ExtractedOdd[i].GetDeviceSlice(b, Intent.Read),
                                                   ExtractedOddRand[Order[i]].GetDeviceSlice(b, Intent.Write),
                                                   Dim * Dim);
                            GPU.CopyDeviceToDevice(ExtractedEven[i].GetDeviceSlice(b, Intent.Read),
                                                   ExtractedEvenRand[Order[i]].GetDeviceSlice(b, Intent.Write),
                                                   Dim * Dim);
                        }
                    }

                    List<float> AllLosses = new List<float>();
                    float[] Loss = null;

                    //for (int s = 0; s < MapSamples; s++)
                    {
                        float CurrentLearningRate = (float)Math.Exp(MathHelper.Lerp((float)Math.Log(1e-4), 
                                                                                    (float)Math.Log(1e-5), 
                                                                                    iepoch / (float)NEpochs));

                        for (int m = 0; m < ShuffledMapIDs.Length; m++)
                        {
                            int MapID = m;

                            bool Twist = true;// Rand.Next(2) == 0;
                            Image Prediction;

                            TrainModel.Train((Twist ? ExtractedOddRand : ExtractedEvenRand)[MapID],
                                             (Twist ? ExtractedEvenRand : ExtractedOddRand)[MapID],
                                             CurrentLearningRate,
                                             out Prediction,
                                             out Loss);

                            Dispatcher.Invoke(() => ProgressDialog.SetProgress((iepoch * NMapsPerEpoch + m + 1) / (float)(NEpochs * NMapsPerEpoch)));
                        }
                    }
                }

                Dispatcher.Invoke(() =>
                {
                    ProgressDialog.SetIndeterminate();
                    ProgressDialog.SetTitle("Saving new model...");
                });

                TrainModel.Save(WithExamples[0].DenoiseTrainingDirModel);
                NoiseNetworksModelDir = "";
                TrainModel.Dispose();

                IEnumerable<Image>[] AllGPUResources = { MapsEven, MapsOdd, 
                                                         ExtractedEven, ExtractedOdd, 
                                                         ExtractedEvenRand, ExtractedOddRand };

                foreach (var collection in AllGPUResources)
                    foreach (var item in collection)
                        item.Dispose();

                foreach (var item in TexturesEven)
                    GPU.DestroyTexture(item.texture, item.array);
                foreach (var item in TexturesOdd)
                    GPU.DestroyTexture(item.texture, item.array);
            });

            Image.FreeDeviceAll();
            FreeOnDevice();
            DropNoiseNetworks();
            DropBoxNetworks();
            TorchSharp.Torch.CudaEmptyCache();

            await ProgressDialog.CloseAsync();

            MicrographOwner = null;
            DispatchUpdateImage();
        }

        #endregion

        #region Mask

        private void ReloadMask()
        {
            if (Movie != null && File.Exists(Movie.AveragePath))
            {
                if (File.Exists(Movie.MaskPath))
                {
                    Image MaskImage = Image.FromFile(Movie.MaskPath);
                    MaskDims = MaskImage.DimsSlice;
                    MaskData = new byte[MaskDims.Elements()];
                    float[] LoadedMaskData = MaskImage.GetHost(Intent.Read)[0];
                    for (int i = 0; i < MaskData.Length; i++)
                        MaskData[i] = LoadedMaskData[i] > 0 ? (byte)1 : (byte)0;
                    MaskImage.Dispose();
                }
                else
                {
                    MaskData = new byte[MaskDims.Elements()];
                }
            }
            else
            {
                MaskDims = new int2(1);
                MaskData = new byte[1];
            }

            // Fill mask image with the base color, so only alpha needs to be set later
            MaskImageBytes = new byte[MaskDims.Elements() * 4];
            for (int i = 0; i < MaskData.Length; i++)
            {
                MaskImageBytes[i * 4 + 0] = 227; // 135; // Colors.Orange.B;
                MaskImageBytes[i * 4 + 1] = 72; // 43; // Colors.Orange.G;
                MaskImageBytes[i * 4 + 2] = 177; // 105; // Colors.Orange.R;
                MaskImageBytes[i * 4 + 3] = 255;
            }
        }

        private void MaskSettingsChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args)
        {
            UpdateMask();
        }

        private void UpdateMask()
        {
            if (!MaskShow || MaskData == null || MaskData.Length == 1)
            {
                ImageMask.Visibility = Visibility.Collapsed;
                return;
            }

            ImageMask.Visibility = Visibility.Visible;

            for (int y = 0; y < MaskDims.Y; y++)
            {
                int yy = MaskDims.Y - 1 - y;
                for (int x = 0; x < MaskDims.X; x++)
                {
                    MaskImageBytes[(yy * MaskDims.X + x) * 4 + 3] = MaskData[y * MaskDims.X + x] > 0 ? (byte)100 : (byte)0;
                }
            }

            ImageSource MaskImageSource = BitmapSource.Create(MaskDims.X, MaskDims.Y, 96, 96, PixelFormats.Bgra32, null, MaskImageBytes, MaskDims.X * 4);
            MaskImageSource.Freeze();

            ImageMask.Source = MaskImageSource;
            //ImageMask.Width = ImageDisplay.ActualWidth;
            //ImageMask.Height = ImageDisplay.ActualHeight;
        }

        private void PaintMaskLine(float2 start, float2 finish, float radius, byte value)
        {
            float2 RectStart = new float2(Math.Max(0, Math.Min(start.X - radius, finish.X - radius)),
                                          Math.Max(0, Math.Min(start.Y - radius, finish.Y - radius)));
            float2 RectEnd = new float2(Math.Min(MaskDims.X - 1, Math.Max(start.X + radius, finish.X + radius)),
                                        Math.Min(MaskDims.Y - 1, Math.Max(start.Y + radius, finish.Y + radius)));
            float SegmentLength = (finish - start).Length();
            float2 Direction = (finish - start) / Math.Max(1e-6f, SegmentLength);
            float RadiusSq = radius * radius;

            for (int y = (int)RectStart.Y; y <= RectEnd.Y; y++)
            {
                for (int x = (int)RectStart.X; x <= RectEnd.X; x++)
                {
                    float2 P = new float2(x, y);
                    float T = Math.Max(0, Math.Min(SegmentLength, float2.Dot(P - start, Direction)));
                    float2 ClosestPoint = start + Direction * T;
                    float DistSq = (P - ClosestPoint).LengthSq();
                    if (DistSq < RadiusSq)
                        MaskData[y * MaskDims.X + x] = value;
                }
            }

            UpdateMask();
        }

        private void ButtonMaskPaint_OnClick(object sender, RoutedEventArgs e)
        {
            if (!BrushActive)
            {
                BrushActive = true;
                MaskShow = true;
                ButtonMaskPaint.Foreground = new SolidColorBrush(Colors.Red);
                ButtonMaskPaint.FontWeight = FontWeights.Black;
            }
            else
            {
                BrushActive = false;
                ButtonMaskPaint.Foreground = new SolidColorBrush(Colors.CornflowerBlue);
                ButtonMaskPaint.FontWeight = FontWeights.Medium;
            }
        }

        private void UpdateBrushEllipse(MouseEventArgs e)
        {
            if (!ImageDisplay.IsMouseOver || !BrushActive)
            {
                EllipseBrush.Visibility = Visibility.Hidden;
                return;
            }

            EllipseBrush.Visibility = Visibility.Visible;

            Point MousePos = e.GetPosition(ImageDisplay);
            double ScaledRadius = BrushDiameter / 2.0 / PixelSize * ScaleFactor;

            EllipseBrush.Width = ScaledRadius * 2;
            EllipseBrush.Height = ScaledRadius * 2;

            Canvas.SetLeft(EllipseBrush, MousePos.X - ScaledRadius);
            Canvas.SetTop(EllipseBrush, MousePos.Y - ScaledRadius);
        }

        private void SaveMask()
        {
            try
            {
                Directory.CreateDirectory(Movie.MaskDir);

                Tiff output = Tiff.Open(Movie.MaskPath, "w");

                const int samplesPerPixel = 1;
                const int bitsPerSample = 8;

                output.SetField(TiffTag.IMAGEWIDTH, MaskDims.X / samplesPerPixel);
                output.SetField(TiffTag.IMAGELENGTH, MaskDims.Y);
                output.SetField(TiffTag.SAMPLESPERPIXEL, samplesPerPixel);
                output.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample);
                output.SetField(TiffTag.ORIENTATION, BitMiracle.LibTiff.Classic.Orientation.BOTLEFT);
                output.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);

                output.SetField(TiffTag.COMPRESSION, Compression.LZW);

                output.SetField(TiffTag.ROWSPERSTRIP, output.DefaultStripSize(0));
                output.SetField(TiffTag.XRESOLUTION, 100.0);
                output.SetField(TiffTag.YRESOLUTION, 100.0);
                output.SetField(TiffTag.RESOLUTIONUNIT, ResUnit.INCH);

                // specify that it's a page within the multipage file
                //output.SetField(TiffTag.SUBFILETYPE, FileType.PAGE);
                // specify the page number
                //output.SetField(TiffTag.PAGENUMBER, 0, 1);

                for (int j = 0; j < MaskDims.Y; j++)
                {
                    int jflip = MaskDims.Y - 1 - j;
                    output.WriteScanline(Helper.Subset(MaskData, jflip * MaskDims.X, (jflip + 1) * MaskDims.X), j);
                }

                output.WriteDirectory();
                output.FlushData();

                output.Dispose();
                
                Movie.MaskPercentage = (decimal)MaskData.Select(v => (int)v).Sum() / MaskData.Length * 100;
                Movie.SaveMeta();
            }
            catch { }
        }

        #endregion

        #region Helpers

        int GetPlanForw(int2 dims)
        {
            if (dims == DimsPlanForw && PlanForw > 0)
                return PlanForw;

            if (PlanForw > 0)
                GPU.DestroyFFTPlan(PlanForw);

            PlanForw = GPU.CreateFFTPlan(new int3(dims), 1);
            DimsPlanForw = dims;

            return PlanForw;
        }

        int GetPlanBack(int2 dims)
        {
            if (dims == DimsPlanBack && PlanBack > 0)
                return PlanBack;

            if (PlanBack > 0)
                GPU.DestroyFFTPlan(PlanBack);

            PlanBack = GPU.CreateIFFTPlan(new int3(dims), 1);
            DimsPlanBack = dims;

            return PlanBack;
        }

        int GetPlanForw3A(int2 dims)
        {
            if (dims == DimsPlanForw3A && PlanForw3A > 0)
                return PlanForw3A;

            if (PlanForw3A > 0)
                GPU.DestroyFFTPlan(PlanForw3A);

            PlanForw3A = GPU.CreateFFTPlan(new int3(dims), 1);
            DimsPlanForw3A = dims;

            return PlanForw3A;
        }

        int GetPlanBack3A(int2 dims)
        {
            if (dims == DimsPlanBack3A && PlanBack3A > 0)
                return PlanBack3A;

            if (PlanBack3A > 0)
                GPU.DestroyFFTPlan(PlanBack3A);

            PlanBack3A = GPU.CreateIFFTPlan(new int3(dims), 1);
            DimsPlanBack3A = dims;

            return PlanBack3A;
        }

        Image GetMicrographFT(Movie owner)
        {
            if (owner == null)
                return null;

            if (owner.GetType() == typeof (Movie))
            {
                if (MicrographFT != null && MicrographOwner == owner)
                    return MicrographFT;

                MicrographFT?.Dispose();
                MicrographDenoised?.Dispose();
                MicrographDenoised = null;

                if (!File.Exists(owner.AveragePath))
                    return null;

                Image Micrograph = Image.FromFile(owner.AveragePath);

                bool _DeconvEnabled = false;
                bool _DenoiseEnabled = false;
                Dispatcher.Invoke(() => _DeconvEnabled = DeconvEnabled);
                Dispatcher.Invoke(() => _DenoiseEnabled = DeconvMode == "Denoise");
                if (_DeconvEnabled && !_DenoiseEnabled)
                {
                    float BPScaling = 8f / Micrograph.PixelSize;
                    int2 DimsOri = new int2(Micrograph.Dims);

                    Micrograph.Normalize();
                    Image MicrographPadded = Micrograph.AsPadded(DimsOri - 8).AndDisposeParent().AsPaddedClamped(new int2(Micrograph.Dims) * 2).AndDisposeParent();
                    MicrographPadded.MaskRectangularly(new int3(DimsOri), DimsOri.X / 2, false);
                    MicrographPadded.Bandpass(2f / 32f / BPScaling, 1, false, 2f / 32f / BPScaling);
                    Micrograph = MicrographPadded.AsPadded(DimsOri).AndDisposeParent();
                }

                MicrographFT = Micrograph.AsFFT(false, GetPlanForw(Micrograph.DimsSlice));
                MicrographOwner = owner;
                Micrograph.Dispose();

                return MicrographFT;
            }
            else if (owner.GetType() == typeof(TiltSeries))
            {
                TiltSeries Owner = owner as TiltSeries;

                if (MicrographFT != null && MicrographOwner == owner && MicrographOwnerSlice == SliceID)
                    return MicrographFT;

                MicrographFT?.Dispose();

                if (SliceID < 1 || SliceID > Owner.NTilts)
                    return null;

                if (!File.Exists(Owner.Path))
                    return null;

                return null;
                //Image Micrograph = Image.FromFile(Owner.Path, SliceID - 1);
                //MicrographFT = Micrograph.AsFFT(false, GetPlanForw(Micrograph.DimsSlice));
                //MicrographOwner = Owner;
                //MicrographOwnerSlice = SliceID;
                //Micrograph.Dispose();

                //return MicrographFT;
            }

            return null;
        }

        async Task<Image> GetMicrographDenoised(Movie owner)
        {
            if (owner == null)
                return null;

            if (owner.GetType() == typeof(Movie))
            {
                if (MicrographDenoised != null && MicrographOwner == owner)
                    return MicrographDenoised;

                if (!File.Exists(owner.AveragePath))
                    return null;

                Image ImageFT = GetMicrographFT(owner);
                if (ImageFT == null)
                    return null;


                string ModelDir = owner.DenoiseTrainingDirModel;
                if (!File.Exists(ModelDir))
                    ModelDir = "NoiseNet2D_default.pt";

                if (NoiseNetworks == null || NoiseNetworksModelDir != ModelDir)
                {
                    DropNoiseNetworks();
                    await PopulateNoiseNetworks(ModelDir);
                }

                MicrographDenoised?.Dispose();

                Image ImageCroppedFT = ImageFT.AsPadded(MicrographDimsDenoise);
                Image ImageCropped = ImageCroppedFT.AsIFFT(false, 0, true).AndDisposeParent();
                ImageCropped = ImageCropped.AsPadded(MicrographDimsDenoise - 2).AndDisposeParent();
                ImageCropped = ImageCropped.AsPaddedClamped(MicrographDimsDenoise * 2).AndDisposeParent();

                ImageCroppedFT = ImageCropped.AsFFT().AndDisposeParent();

                CTF CTFDenoise = owner.CTF.GetCopy();
                CTFDenoise.PixelSize = (decimal)NoiseNet2DTorch.PixelSize;
                float HighpassNyquist = NoiseNet2DTorch.PixelSize * 2 / 500f;
                GPU.DeconvolveCTF(ImageCroppedFT.GetDevice(Intent.Read),
                                  ImageCroppedFT.GetDevice(Intent.Write),
                                  ImageCroppedFT.Dims,
                                  CTFDenoise.ToStruct(),
                                  0.85f,
                                  0.25f,
                                  HighpassNyquist);

                MicrographDenoised = ImageCroppedFT.AsIFFT(false, 0, true).AndDisposeParent();
                MicrographDenoised = MicrographDenoised.AsPadded(MicrographDimsDenoise).AndDisposeParent();

                float2 MeanStd;
                {
                    Image MicrographCenter = MicrographDenoised.AsPadded(MicrographDimsDenoise / 2);
                    MeanStd = MathHelper.MeanAndStd(MicrographCenter.GetHost(Intent.Read)[0]);
                    MicrographCenter.Dispose();
                }

                //MicrographDenoised.TransformValues(v => MathF.Sqrt(MathF.Abs(v - MeanStd.X)) * Math.Sign(v - MeanStd.X));
                //{
                //    Image MicrographCenter = MicrographDenoised.AsPadded(MicrographDimsDenoise / 2);
                //    MeanStd = MathHelper.MeanAndStd(MicrographCenter.GetHost(Intent.Read)[0]);
                //    MicrographCenter.Dispose();
                //}

                //{
                //    Image OriginalCropped = MicrographDenoised.AsPadded(new int2(MicrographDenoised.Dims) - 16).AndDisposeParent();

                //    Image OriginalPadded = OriginalCropped.AsPaddedClamped(new int2(MicrographDenoised.Dims) * 2).AndDisposeParent();
                //    OriginalPadded.Bandpass(2f / 128, 1, false, 1f / 128);

                //    MicrographDenoised = OriginalPadded.AsPadded(new int2(OriginalPadded.Dims) / 2).AndDisposeParent();
                //}

                //MicrographDenoised.SubtractMeanGrid(new int2(4, 4));

                //GPU.Normalize(MicrographDenoised.GetDevice(Intent.Read),
                //              MicrographDenoised.GetDevice(Intent.Write),
                //              (uint)MicrographDenoised.ElementsReal,
                //              1);

                MicrographDenoised.TransformValues(v => Math.Max(-5, Math.Min((v - MeanStd.X) / MeanStd.Y, 5)));

                //MicrographDenoised.Multiply(1f / 1.14f);

                NoiseNet2DTorch.Denoise(MicrographDenoised, NoiseNetworks);
                
                return MicrographDenoised;
            }
            else if (owner.GetType() == typeof(TiltSeries))
            {
                return null;
            }

            return null;
        }

        public void FreeOnDevice()
        {
            MicrographFT?.FreeDevice();
            MicrographFT = null;

            MicrographDenoised?.FreeDevice();
            MicrographDenoised = null;

            MicrographOwner = null;
            MicrographOwnerSlice = -1;

            if (PlanForw > 0)
                GPU.DestroyFFTPlan(PlanForw);
            PlanForw = -1;
            DimsPlanForw = new int2(-1);

            if (PlanBack > 0)
                GPU.DestroyFFTPlan(PlanBack);
            PlanBack = -1;
            DimsPlanBack = new int2(-1);

            if (PlanForw3A > 0)
                GPU.DestroyFFTPlan(PlanForw3A);
            PlanForw3A = -1;
            DimsPlanForw3A = new int2(-1);

            if (PlanBack3A > 0)
                GPU.DestroyFFTPlan(PlanBack3A);
            PlanBack3A = -1;
            DimsPlanBack3A = new int2(-1);
        }

        private Size MeasureString(string candidate, double size)
        {
            FormattedText Test = new FormattedText(candidate,
                                                   CultureInfo.CurrentUICulture,
                                                   FlowDirection.LeftToRight,
                                                   new Typeface(FontFamily, FontStyle, FontWeights.Bold, FontStretch),
                                                   size,
                                                   Brushes.Black);

            return new Size(Test.Width, Test.Height);
        }

        private void RenderToThumbnail(string path, int size)
        {
            try
            {
                double ScaleFactor = size / Math.Max(ImageDisplay.ActualWidth, ImageDisplay.ActualHeight);
                int ScaledWidth = (int)(ImageDisplay.ActualWidth * ScaleFactor + 0.5);
                int ScaledHeight = (int)(ImageDisplay.ActualHeight * ScaleFactor + 0.5);


                DrawingVisual Visual = new DrawingVisual();
                using (DrawingContext context = Visual.RenderOpen())
                {
                    VisualBrush brush = new VisualBrush(PanelVisuals);
                    context.DrawRectangle(brush,
                                          null,
                                          new Rect(new Point(), new Size(ImageDisplay.ActualWidth, ImageDisplay.ActualHeight)));
                }

                RenderOptions.SetBitmapScalingMode(Visual, BitmapScalingMode.Fant);

                Visual.Transform = new ScaleTransform(ScaledWidth / ImageDisplay.ActualWidth,
                                                      ScaledHeight / ImageDisplay.ActualHeight);

                RenderTargetBitmap Image = new RenderTargetBitmap(ScaledWidth, ScaledHeight, 96, 96, PixelFormats.Pbgra32);
                RenderOptions.SetBitmapScalingMode(Image, BitmapScalingMode.Fant);
                Image.Render(Visual);

                var Encoder = new PngBitmapEncoder();
                Encoder.Frames.Add(BitmapFrame.Create(Image));

                Directory.CreateDirectory(path.Substring(0, Math.Max(path.LastIndexOf("\\"), path.LastIndexOf("/"))));
                using (Stream stream = File.Create(path))
                    Encoder.Save(stream);
            }
            catch { }
        }

        public void SetProcessingMode(bool enabled)
        {
            if (enabled)
            {
                ButtonBoxNetPick.IsEnabled = false;
                ButtonDeconvRetrain.IsEnabled = false;
            }
            else
            {
                ButtonBoxNetPick.IsEnabled = true;
                ButtonDeconvRetrain.IsEnabled = true;
            }
        }

        #endregion
    }
}
