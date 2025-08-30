using M;
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
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Warp;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;
using WorkerWrapper = Warp.Workers.WorkerWrapper;

namespace M.Controls
{
    /// <summary>
    /// Interaction logic for DialogParticleExport3D.xaml
    /// </summary>
    public partial class DialogTomoParticleExport : UserControl
    {
        #region Dependency properties

        public decimal OutputAngPix
        {
            get { return (decimal)GetValue(OutputAngPixProperty); }
            set { SetValue(OutputAngPixProperty, value); }
        }
        public static readonly DependencyProperty OutputAngPixProperty =
            DependencyProperty.Register("OutputAngPix", typeof(decimal), typeof(DialogTomoParticleExport), new PropertyMetadata(1M));
        
        public int BoxSize
        {
            get { return (int)GetValue(BoxSizeProperty); }
            set { SetValue(BoxSizeProperty, value); }
        }
        public static readonly DependencyProperty BoxSizeProperty =
            DependencyProperty.Register("BoxSize", typeof(int), typeof(DialogTomoParticleExport), new PropertyMetadata(128));

        public int Diameter
        {
            get { return (int)GetValue(DiameterProperty); }
            set { SetValue(DiameterProperty, value); }
        }
        public static readonly DependencyProperty DiameterProperty =
            DependencyProperty.Register("Diameter", typeof(int), typeof(DialogTomoParticleExport), new PropertyMetadata(100));

        public bool ReconstructVolume
        {
            get { return (bool)GetValue(ReconstructVolumeProperty); }
            set { SetValue(ReconstructVolumeProperty, value); }
        }
        public static readonly DependencyProperty ReconstructVolumeProperty =
            DependencyProperty.Register("ReconstructVolume", typeof(bool), typeof(DialogTomoParticleExport), new PropertyMetadata(true));

        public bool ReconstructSeries
        {
            get { return (bool)GetValue(ReconstructSeriesProperty); }
            set { SetValue(ReconstructSeriesProperty, value); }
        }
        public static readonly DependencyProperty ReconstructSeriesProperty =
            DependencyProperty.Register("ReconstructSeries", typeof(bool), typeof(DialogTomoParticleExport), new PropertyMetadata(false));

        public bool ApplyShift
        {
            get { return (bool)GetValue(ApplyShiftProperty); }
            set { SetValue(ApplyShiftProperty, value); }
        }
        public static readonly DependencyProperty ApplyShiftProperty =
            DependencyProperty.Register("ApplyShift", typeof(bool), typeof(DialogTomoParticleExport), new PropertyMetadata(false));

        public decimal ShiftX
        {
            get { return (decimal)GetValue(ShiftXProperty); }
            set { SetValue(ShiftXProperty, value); }
        }
        public static readonly DependencyProperty ShiftXProperty =
            DependencyProperty.Register("ShiftX", typeof(decimal), typeof(DialogTomoParticleExport), new PropertyMetadata(0M));

        public decimal ShiftY
        {
            get { return (decimal)GetValue(ShiftYProperty); }
            set { SetValue(ShiftYProperty, value); }
        }
        public static readonly DependencyProperty ShiftYProperty =
            DependencyProperty.Register("ShiftY", typeof(decimal), typeof(DialogTomoParticleExport), new PropertyMetadata(0M));

        public decimal ShiftZ
        {
            get { return (decimal)GetValue(ShiftZProperty); }
            set { SetValue(ShiftZProperty, value); }
        }
        public static readonly DependencyProperty ShiftZProperty =
            DependencyProperty.Register("ShiftZ", typeof(decimal), typeof(DialogTomoParticleExport), new PropertyMetadata(0M));

        public bool ReconstructPrerotated
        {
            get { return (bool)GetValue(ReconstructPrerotatedProperty); }
            set { SetValue(ReconstructPrerotatedProperty, value); }
        }
        public static readonly DependencyProperty ReconstructPrerotatedProperty =
            DependencyProperty.Register("ReconstructPrerotated", typeof(bool), typeof(DialogTomoParticleExport), new PropertyMetadata(false));

        public bool ReconstructDoLimitDose
        {
            get { return (bool)GetValue(ReconstructDoLimitDoseProperty); }
            set { SetValue(ReconstructDoLimitDoseProperty, value); }
        }
        public static readonly DependencyProperty ReconstructDoLimitDoseProperty =
            DependencyProperty.Register("ReconstructDoLimitDose", typeof(bool), typeof(DialogTomoParticleExport), new PropertyMetadata(false));

        public int ReconstructNTilts
        {
            get { return (int)GetValue(ReconstructNTiltsProperty); }
            set { SetValue(ReconstructNTiltsProperty, value); }
        }
        public static readonly DependencyProperty ReconstructNTiltsProperty =
            DependencyProperty.Register("ReconstructNTilts", typeof(int), typeof(DialogTomoParticleExport), new PropertyMetadata(0));

        public bool InputInvert
        {
            get { return (bool)GetValue(InputInvertProperty); }
            set { SetValue(InputInvertProperty, value); }
        }
        public static readonly DependencyProperty InputInvertProperty =
            DependencyProperty.Register("InputInvert", typeof(bool), typeof(DialogTomoParticleExport), new PropertyMetadata(true));

        public bool InputNormalize
        {
            get { return (bool)GetValue(InputNormalizeProperty); }
            set { SetValue(InputNormalizeProperty, value); }
        }
        public static readonly DependencyProperty InputNormalizeProperty =
            DependencyProperty.Register("InputNormalize", typeof(bool), typeof(DialogTomoParticleExport), new PropertyMetadata(true));

        public bool OutputNormalize
        {
            get { return (bool)GetValue(OutputNormalizeProperty); }
            set { SetValue(OutputNormalizeProperty, value); }
        }
        public static readonly DependencyProperty OutputNormalizeProperty =
            DependencyProperty.Register("OutputNormalize", typeof(bool), typeof(DialogTomoParticleExport), new PropertyMetadata(true));

        public bool ReconstructMakeSparse
        {
            get { return (bool)GetValue(ReconstructMakeSparseProperty); }
            set { SetValue(ReconstructMakeSparseProperty, value); }
        }
        public static readonly DependencyProperty ReconstructMakeSparseProperty =
            DependencyProperty.Register("ReconstructMakeSparse", typeof(bool), typeof(DialogTomoParticleExport), new PropertyMetadata(false));

        #endregion

        public string ExportPath;
        public event Action Close;

        DataSource[] Sources;
        Species Species;

        List<UIElement> DisableWhileProcessing;

        bool IsCanceled = false;

        List<float> Timings = new List<float>();

        public DialogTomoParticleExport(Population population, Species species)
        {
            InitializeComponent();

            OutputAngPix = species.PixelSize;
            Diameter = species.DiameterAngstrom;
            BoxSize = species.Size;

            DataContext = this;

            Sources = population.Sources.ToArray();
            Species = species;

            DisableWhileProcessing = new List<UIElement>
            {
                SliderBoxSize,
                SliderParticleDiameter,
                CheckInvert,
                CheckInputNormalize,
                CheckOutputNormalize,
            };
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonWrite_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.SaveFileDialog SaveDialog = new System.Windows.Forms.SaveFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultSave = SaveDialog.ShowDialog();

            if (ResultSave.ToString() == "OK")
            {
                ExportPath = SaveDialog.FileName;
            }
            else
            {
                return;
            }

            decimal _OutputAngPix = OutputAngPix;
            int _BoxSize = BoxSize;
            int _Diameter = Diameter;

            bool _ReconstructVolume = ReconstructVolume;
            bool _ReconstructSeries = ReconstructSeries;

            bool _ApplyShift = ApplyShift;
            decimal _ShiftX = ShiftX;
            decimal _ShiftY = ShiftY;
            decimal _ShiftZ = ShiftZ;

            bool _ReconstructPrerotated = ReconstructPrerotated;
            bool _ReconstructDoLimitDose = ReconstructDoLimitDose;
            int _ReconstructNTilts = ReconstructNTilts;

            bool _InputInvert = InputInvert;
            bool _InputNormalize = InputNormalize;
            bool _OutputNormalize = OutputNormalize;
            bool _ReconstructMakeSparse = ReconstructMakeSparse;

            if (!_ReconstructVolume)
                _ReconstructPrerotated = true;

            float3 AdditionalShiftAngstrom = new float3((float)_ShiftX,
                                                        (float)_ShiftY,
                                                        (float)_ShiftZ);

            ProgressWrite.Visibility = Visibility.Visible;
            ProgressWrite.IsIndeterminate = true;
            PanelButtons.Visibility = Visibility.Collapsed;
            PanelRemaining.Visibility = Visibility.Visible;

            foreach (var element in DisableWhileProcessing)
                element.IsEnabled = false;

            MainWindow MainWindow = (MainWindow)Application.Current.MainWindow;

            await Task.Run(() =>
            {
                List<Star> AllSourceTables = new List<Star>();
                string[] AllSourceHashes = Helper.GetUniqueElements(Species.Particles.Select(p => p.SourceHash)).ToArray();
                AllSourceHashes = AllSourceHashes.Where(h => Sources.Any(s => s.Files.ContainsKey(h))).ToArray();

                foreach (var source in Sources)
                {
                    if (!source.IsTiltSeries)
                        continue;

                    #region Get all movies that can potentially be used

                    List<string> ValidSourceHashes = AllSourceHashes.Where(h => source.Files.ContainsKey(h)).ToList();
                    List<string> ValidSourcePaths = ValidSourceHashes.Select(k => System.IO.Path.Combine(source.FolderPath, source.Files[k])).ToList();
                    List<string> ValidMovieNames = ValidSourcePaths.Select(p => Helper.PathToName(p)).ToList();
                    List<TiltSeries> ValidSeries = ValidSourcePaths.Select(p => new TiltSeries(p)).ToList();

                    List<Particle> Particles = Species.Particles.Where(p => ValidSourceHashes.Contains(p.SourceHash)).ToList();
                    Particles.Sort((a, b) => a.SourceHash.CompareTo(b.SourceHash));

                    #endregion

                    if (IsCanceled)
                        return;

                    #region Create worker processes

                    int NDevices = GPU.GetDeviceCount();
                    List<int> UsedDevices = MainWindow.GetDeviceList();
                    List<int> UsedDeviceProcesses = Helper.Combine(Helper.ArrayOfFunction(i => UsedDevices.Select(d => d + i * NDevices).ToArray(), 1)).ToList();

                    WorkerWrapper[] Workers = new WorkerWrapper[GPU.GetDeviceCount() * 1];
                    foreach (var gpuID in UsedDeviceProcesses)
                    {
                        Workers[gpuID] = new WorkerWrapper(gpuID, false, false);
                        Workers[gpuID].SetHeaderlessParams(new int2(1, 1),
                                                           0,
                                                           "float32");

                        Workers[gpuID].LoadGainRef(source.GainPath,
                                                   source.GainFlipX,
                                                   source.GainFlipY,
                                                   source.GainTranspose,
                                                   source.DefectsPath);
                    }

                    #endregion

                    List<Star> SourceTables = new List<Star>();

                    {                        
                        Dispatcher.Invoke(() => ProgressWrite.MaxValue = AllSourceHashes.Length);
                        
                        Helper.ForEachGPU(ValidSeries, (series, gpuID) =>
                        {
                            if (IsCanceled)
                                return;

                            Stopwatch ItemTime = new Stopwatch();
                            ItemTime.Start();

                            string SeriesHash = series.GetDataHash();

                            Star TableOut = new Star(new string[] { "rlnMagnification",
                                                                    "rlnDetectorPixelSize",
                                                                    "rlnCoordinateX",
                                                                    "rlnCoordinateY",
                                                                    "rlnCoordinateZ",
                                                                    "rlnAngleRot",
                                                                    "rlnAngleTilt",
                                                                    "rlnAnglePsi",
                                                                    "rlnImageName",
                                                                    "rlnCtfImage",
                                                                    "rlnRandomSubset",
                                                                    "rlnPixelSize",
                                                                    "rlnVoltage",
                                                                    "rlnSphericalAberration",
                                                                    "rlnMicrographName"
                            });
                            
                            ProcessingOptionsTomoSubReconstruction ExportOptions = new ProcessingOptionsTomoSubReconstruction()
                            {
                                PixelSize = source.PixelSize,

                                BinTimes = (decimal)Math.Log((double)(_OutputAngPix / source.PixelSizeMean), 2.0),
                                GainPath = source.GainPath,
                                DefectsPath = source.DefectsPath,
                                GainFlipX = source.GainFlipX,
                                GainFlipY = source.GainFlipY,
                                GainTranspose = source.GainTranspose,

                                Dimensions = new float3((float)source.DimensionsX,
                                                        (float)source.DimensionsY,
                                                        (float)source.DimensionsZ),

                                Suffix = "_" + Species.NameSafe,

                                BoxSize = _BoxSize,
                                ParticleDiameter = _Diameter,

                                Invert = _InputInvert,
                                NormalizeInput = _InputNormalize,
                                NormalizeOutput = _OutputNormalize,

                                PrerotateParticles = _ReconstructPrerotated,
                                DoLimitDose = _ReconstructDoLimitDose,
                                NTilts = Math.Min(series.NTilts, Math.Min(source.FrameLimit, _ReconstructNTilts)),

                                MakeSparse = _ReconstructMakeSparse
                            };

                            Particle[] SeriesParticles = Particles.Where(p => p.SourceHash == SeriesHash).ToArray();
                            int NParticles = SeriesParticles.Length;

                            #region Process particle positions and angles

                            float3[] Positions = new float3[NParticles * series.NTilts];
                            float3[] Angles = new float3[NParticles * series.NTilts];
                            int[] Subsets = SeriesParticles.Select(p => p.RandomSubset + 1).ToArray();
                            float MinDose = MathHelper.Min(series.Dose);
                            float MaxDose = MathHelper.Max(series.Dose);
                            float[] InterpolationSteps = Helper.ArrayOfFunction(i => (series.Dose[i] - MinDose) / (MaxDose - MinDose), series.NTilts);

                            for (int p = 0; p < NParticles; p++)
                            {
                                float3[] ParticlePositions = SeriesParticles[p].GetCoordinateSeries(InterpolationSteps);
                                float3[] ParticleAngles = SeriesParticles[p].GetAngleSeries(InterpolationSteps);

                                if (_ApplyShift)
                                {
                                    Matrix3 R0 = Matrix3.Euler(ParticleAngles[0] * Helper.ToRad);
                                    float3 RotatedShift = R0 * AdditionalShiftAngstrom;

                                    for (int t = 0; t < ParticlePositions.Length; t++)
                                        ParticlePositions[t] += RotatedShift;
                                }

                                if (!_ReconstructPrerotated)
                                {
                                    Matrix3 R0I = Matrix3.Euler(ParticleAngles[0] * Helper.ToRad).Transposed();

                                    for (int t = 0; t < ParticleAngles.Length; t++)
                                        ParticleAngles[t] = Matrix3.EulerFromMatrix(R0I * Matrix3.Euler(ParticleAngles[t] * Helper.ToRad)) * Helper.ToDeg;
                                }

                                for (int t = 0; t < series.NTilts; t++)
                                {
                                    Positions[p * series.NTilts + t] = ParticlePositions[t];
                                    Angles[p * series.NTilts + t] = ParticleAngles[t];
                                }

                                string PathSubtomo = System.IO.Path.Combine(series.SubtomoDir, $"{series.RootName}{ExportOptions.Suffix}_{p:D7}_{ExportOptions.BinnedPixelSizeMean:F2}A.mrc");
                                string PathCTF = System.IO.Path.Combine(series.SubtomoDir, $"{series.RootName}{ExportOptions.Suffix}_{p:D7}_ctf_{ExportOptions.BinnedPixelSizeMean:F2}A.mrc");

                                Uri UriStar = new Uri(ExportPath);
                                PathSubtomo = UriStar.MakeRelativeUri(new Uri(PathSubtomo)).ToString();
                                PathCTF = UriStar.MakeRelativeUri(new Uri(PathCTF)).ToString();

                                TableOut.AddRow(new string[]
                                {
                                    "10000.0",
                                    ExportOptions.BinnedPixelSizeMean.ToString("F5", CultureInfo.InvariantCulture),
                                    (ParticlePositions[0].X / (float)ExportOptions.BinnedPixelSizeMean).ToString("F5", CultureInfo.InvariantCulture),
                                    (ParticlePositions[0].Y / (float)ExportOptions.BinnedPixelSizeMean).ToString("F5", CultureInfo.InvariantCulture),
                                    (ParticlePositions[0].Z / (float)ExportOptions.BinnedPixelSizeMean).ToString("F5", CultureInfo.InvariantCulture),
                                    (_ReconstructPrerotated ? 0 : SeriesParticles[p].Angles[0].X).ToString("F5", CultureInfo.InvariantCulture),
                                    (_ReconstructPrerotated ? 0 : SeriesParticles[p].Angles[0].Y).ToString("F5", CultureInfo.InvariantCulture),
                                    (_ReconstructPrerotated ? 0 : SeriesParticles[p].Angles[0].Z).ToString("F5", CultureInfo.InvariantCulture),
                                    PathSubtomo,
                                    PathCTF,
                                    Subsets[p].ToString(),
                                    ExportOptions.BinnedPixelSizeMean.ToString("F5", CultureInfo.InvariantCulture),
                                    series.CTF.Voltage.ToString("F1", CultureInfo.InvariantCulture),
                                    series.CTF.Cs.ToString("F3", CultureInfo.InvariantCulture),
                                    series.Name
                                });
                            }

                            #endregion
                                
                            #region Finally, reconstruct the actual sub-tomos

                            if (_ReconstructVolume)
                            {
                                Workers[gpuID].TomoExportParticleSubtomos(series.DataPath, ExportOptions, Positions, Angles);
                            }
                            else
                            {
                                series.ReconstructParticleSeries(ExportOptions, Positions, Angles, ExportPath, out TableOut);
                            }

                            lock (AllSourceTables)
                                AllSourceTables.Add(TableOut);

                            #endregion

                            #region Add this micrograph's table to global collection, update remaining time estimate

                            lock (AllSourceTables)
                            {
                                Timings.Add(ItemTime.ElapsedMilliseconds / (float)UsedDeviceProcesses.Count);

                                int MsRemaining = (int)(MathHelper.Mean(Timings) * (AllSourceHashes.Length - AllSourceTables.Count));
                                TimeSpan SpanRemaining = new TimeSpan(0, 0, 0, 0, MsRemaining);

                                Dispatcher.Invoke(() => TextRemaining.Text = SpanRemaining.ToString((int)SpanRemaining.TotalHours > 0 ? @"hh\:mm\:ss" : @"mm\:ss"));

                                Dispatcher.Invoke(() =>
                                {
                                    ProgressWrite.IsIndeterminate = false;
                                    ProgressWrite.Value = AllSourceTables.Count;
                                });
                            }

                        #endregion

                        }, 1, UsedDeviceProcesses);
                    }

                    Thread.Sleep(10000);    // Writing out particles is async, so if workers are killed immediately they may not write out everything

                    foreach (var worker in Workers)
                        worker?.Dispose();

                    if (IsCanceled)
                        return;
                }

                if (AllSourceTables.Count > 0)
                    (new Star(AllSourceTables.ToArray())).Save(ExportPath);
            });

            Close?.Invoke();
        }

        private void ButtonAbort_OnClick(object sender, RoutedEventArgs e)
        {
            ButtonAbort.IsEnabled = false;
            IsCanceled = true;
        }
    }
}
