using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
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
using MahApps.Metro.Controls;
using MahApps.Metro.Controls.Dialogs;
using Warp.Sociology;
using Warp.Tools;
using Warp;
using Image = Warp.Image;

namespace M.Controls.Sociology.Dialogs
{
    /// <summary>
    /// Interaction logic for DialogCreateNewSpecies.xaml
    /// </summary>
    public partial class DialogCreateNewSpecies : UserControl
    {
        public event Action Finish;
        public event Action Close;

        private Population Population;

        NewSpeciesWizzardState _WizzardState = NewSpeciesWizzardState.Particles;
        private NewSpeciesWizzardState WizzardState
        {
            get
            {
                return _WizzardState;
            }
            set
            {
                if (_WizzardState != value)
                {
                    _WizzardState = value;

                    int Tab = (int)value;

                    for (int i = 0; i < Tabs.Length; i++)
                        Headers[i].Foreground = i == Tab ? Brushes.Black : i < Tab ? Brushes.DimGray : Brushes.LightGray;

                    Tabs[Tab].Visibility = Visibility.Visible;
                    PresenterTabs.Content = Tabs[Tab];
                    RevalidateTab();
                }
            }
        }

        private Grid[] Tabs;
        private TextBlock[] Headers;
        private Func<bool>[] TabValidators;

        #region Properties general parameters

        public string SpeciesName
        {
            get { return (string)GetValue(SpeciesNameProperty); }
            set { SetValue(SpeciesNameProperty, value); }
        }
        public static readonly DependencyProperty SpeciesNameProperty = DependencyProperty.Register("SpeciesName", typeof(string), typeof(DialogCreateNewSpecies), new PropertyMetadata("New Species", (a, b) => ((DialogCreateNewSpecies)a).UpdateParameters()));

        public int SpeciesDiameter
        {
            get { return (int)GetValue(SpeciesDiameterProperty); }
            set { SetValue(SpeciesDiameterProperty, value); }
        }
        public static readonly DependencyProperty SpeciesDiameterProperty = DependencyProperty.Register("SpeciesDiameter", typeof(int), typeof(DialogCreateNewSpecies), new PropertyMetadata(100, (a, b) => ((DialogCreateNewSpecies)a).UpdateParameters()));

        public decimal SpeciesWeight
        {
            get { return (decimal)GetValue(SpeciesWeightProperty); }
            set { SetValue(SpeciesWeightProperty, value); }
        }
        public static readonly DependencyProperty SpeciesWeightProperty = DependencyProperty.Register("SpeciesWeight", typeof(decimal), typeof(DialogCreateNewSpecies), new PropertyMetadata(200M, (a, b) => ((DialogCreateNewSpecies)a).UpdateParameters()));

        public string SpeciesSymmetry
        {
            get { return (string)GetValue(SpeciesSymmetryProperty); }
            set { SetValue(SpeciesSymmetryProperty, value); }
        }
        public static readonly DependencyProperty SpeciesSymmetryProperty = DependencyProperty.Register("SpeciesSymmetry", typeof(string), typeof(DialogCreateNewSpecies), new PropertyMetadata("C1", (a, b) => ((DialogCreateNewSpecies)a).UpdateParameters()));

        public int TemporalResMov
        {
            get { return (int)GetValue(TemporalResMovProperty); }
            set { SetValue(TemporalResMovProperty, value); }
        }
        public static readonly DependencyProperty TemporalResMovProperty = DependencyProperty.Register("TemporalResMov", typeof(int), typeof(DialogCreateNewSpecies), new PropertyMetadata(1, (a, b) => ((DialogCreateNewSpecies)a).UpdateParameters()));

        #endregion

        #region Properties half-maps

        public string PathHalfmap1 = "", PathHalfmap2 = "";
        public Image Halfmap1Final, Halfmap2Final;
        private Image Halfmap1, Halfmap2;

        public decimal HalfmapPixelSize
        {
            get { return (decimal)GetValue(HalfmapPixelSizeProperty); }
            set { SetValue(HalfmapPixelSizeProperty, value); }
        }
        public static readonly DependencyProperty HalfmapPixelSizeProperty = DependencyProperty.Register("HalfmapPixelSize", typeof(decimal), typeof(DialogCreateNewSpecies), new PropertyMetadata(1M, (a, b) => ((DialogCreateNewSpecies)a).UpdateHalfmapsResolution()));

        public int HalfmapLowpass
        {
            get { return (int)GetValue(HalfmapLowpassProperty); }
            set { SetValue(HalfmapLowpassProperty, value); }
        }
        public static readonly DependencyProperty HalfmapLowpassProperty = DependencyProperty.Register("HalfmapLowpass", typeof(int), typeof(DialogCreateNewSpecies), new PropertyMetadata(1, (a, b) => ((DialogCreateNewSpecies)a).UpdateHalfmapsResolution()));

        #endregion

        #region Properties mask

        public string PathMask = "";
        public Image MaskFinal;
        private Image Mask;

        public decimal MaskLowpass
        {
            get { return (decimal)GetValue(MaskLowpassProperty); }
            set { SetValue(MaskLowpassProperty, value); }
        }
        public static readonly DependencyProperty MaskLowpassProperty = DependencyProperty.Register("MaskLowpass", typeof(decimal), typeof(DialogCreateNewSpecies), new PropertyMetadata(1M, (a, b) => ((DialogCreateNewSpecies)a).UpdateMaskResolution()));

        public decimal MaskThreshold
        {
            get { return (decimal)GetValue(MaskThresholdProperty); }
            set { SetValue(MaskThresholdProperty, value); }
        }
        public static readonly DependencyProperty MaskThresholdProperty = DependencyProperty.Register("MaskThreshold", typeof(decimal), typeof(DialogCreateNewSpecies), new PropertyMetadata(0.02M, (a, b) => ((DialogCreateNewSpecies)a).UpdateMaskResolution()));

        #endregion

        #region Properties particles

        public Star TableWarp, TableRelion;
        private string PathWarp, PathRelion;

        public decimal ParticleCoordinatesPixel
        {
            get { return (decimal)GetValue(ParticleCoordinatesPixelProperty); }
            set { SetValue(ParticleCoordinatesPixelProperty, value); }
        }
        public static readonly DependencyProperty ParticleCoordinatesPixelProperty = DependencyProperty.Register("ParticleCoordinatesPixel", typeof(decimal), typeof(DialogCreateNewSpecies), new PropertyMetadata(1M, (sender, value) => ((DialogCreateNewSpecies)sender).UpdateParticles()));

        public decimal ParticleShiftsPixel
        {
            get { return (decimal)GetValue(ParticleShiftsPixelProperty); }
            set { SetValue(ParticleShiftsPixelProperty, value); }
        }
        public static readonly DependencyProperty ParticleShiftsPixelProperty = DependencyProperty.Register("ParticleShiftsPixel", typeof(decimal), typeof(DialogCreateNewSpecies), new PropertyMetadata(1M, (sender, value) => ((DialogCreateNewSpecies)sender).UpdateParticles()));

        private bool[] UseSource;
        private DataSource[] ValidSources;
        public DataSource[] UsedSources => ValidSources.Where((s, i) => UseSource[i]).ToArray();

        private int ParticlesMatched, ParticlesUnmatched;

        public Particle[] ParticlesFinal;

        #endregion

        public DialogCreateNewSpecies(Population population)
        {
            InitializeComponent();

            DataContext = this;
            
            RendererMask.Camera.SurfaceThreshold = 0.5M;

            Population = population;

            Tabs = new[] { TabParameters, TabHalfmaps, TabMask, TabParticles };
            Headers = new[] { HeaderParameters, HeaderHalfmaps, HeaderMask, HeaderParticles };
            TabValidators = new Func<bool>[] { () => ValidateParameters(), () => ValidateHalfmaps(), () => ValidateMask(), () => ValidateParticles() };

            foreach (var tab in Tabs)
                GridTabs.Children.Remove(tab);

            WizzardState = NewSpeciesWizzardState.Parameters;

            ValidSources = Population.Sources.Where(s => !s.IsRemote).ToArray();
            UseSource = Helper.ArrayOfConstant(true, ValidSources.Length);
            for (int i = 0; i < ValidSources.Length; i++)
            {
                CheckBox CheckSource = new CheckBox
                {
                    Content = ValidSources[i].Name,
                    FontSize = 18,
                    IsChecked = true
                };
                int s = i;
                CheckSource.Click += (source, e) =>
                {
                    UseSource[s] = (bool)((CheckBox)source).IsChecked;
                    UpdateParticles();
                };
                PanelSources.Children.Add(CheckSource);
            }
        }

        void RevalidateTab()
        {
            int Tab = (int)WizzardState;
            if (TabValidators[Tab] != null)
            {
                if (TabValidators[Tab]())
                {
                    ButtonNext.IsEnabled = true;
                    ButtonNext.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#7ec730"));
                }
                else
                {
                    ButtonNext.IsEnabled = false;
                    ButtonNext.Foreground = Brushes.Gray;
                }
            }
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            DisposeFull();
            Close?.Invoke();
        }

        private void ButtonPrevious_OnClick(object sender, RoutedEventArgs e)
        {
            //PresenterTabs.Transition = TransitionType.Down;
            ButtonNext.Content = "NEXT";

            if (WizzardState == NewSpeciesWizzardState.Halfmaps)
            {
                WizzardState = NewSpeciesWizzardState.Parameters;
                ButtonPrevious.Visibility = Visibility.Collapsed;
            }
            else if (WizzardState == NewSpeciesWizzardState.Mask)
            {
                WizzardState = NewSpeciesWizzardState.Halfmaps;
            }
            else if (WizzardState == NewSpeciesWizzardState.Particles)
            {
                WizzardState = NewSpeciesWizzardState.Mask;
            }
        }

        private void ButtonNext_OnClick(object sender, RoutedEventArgs e)
        {
            //PresenterTabs.Transition = TransitionType.Up;
            ButtonPrevious.Visibility = Visibility.Visible;

            if (WizzardState == NewSpeciesWizzardState.Parameters)
            {
                WizzardState = NewSpeciesWizzardState.Halfmaps;
            }
            else if (WizzardState == NewSpeciesWizzardState.Halfmaps)
            {
                WizzardState = NewSpeciesWizzardState.Mask;
                UpdateMask();
            }
            else if (WizzardState == NewSpeciesWizzardState.Mask)
            {
                WizzardState = NewSpeciesWizzardState.Particles;
                ButtonNext.Content = "FINISH";
            }
            else if (WizzardState == NewSpeciesWizzardState.Particles)
            {
                BeforeFinish();
            }
        }

        async void BeforeFinish()
        {
            List<string> Messages = new List<string>();

            if (Halfmap1Final == null)
                Messages.Add("No half-map 1.");
            if (Halfmap2Final == null)
                Messages.Add("No half-map 2.");
            if (Halfmap1Final != null && Halfmap2Final != null && Halfmap1Final.Dims != Halfmap2Final.Dims)
                Messages.Add("Half-map dimensions don't match.");

            if (MaskFinal == null)
                Messages.Add("No mask.");
            if (Halfmap1Final != null && MaskFinal != null && Halfmap1Final.Dims != MaskFinal.Dims)
                Messages.Add("Mask and half-map dimensions don't match.");

            if (SpeciesName.ToLower() == "new species")
                Messages.Add("Species name left at default value.");

            if (ParticlesFinal == null || ParticlesFinal.Length == 0)
                Messages.Add("No particles.");

            if (Messages.Count > 0)
            {
                await((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", "There are some problems:\n\n" + string.Join("\n", Messages));
                return;
            }

            Dispose();
            Finish?.Invoke();
        }

        #region Parameters

        void UpdateParameters()
        {
            RevalidateTab();
        }

        bool ValidateParameters()
        {
            return SpeciesName.ToLower() != "new species";
        }

        #endregion

        #region Half-maps

        async void UpdateHalfmaps()
        {
            if (!IsInitialized)
                return;
            
            ProgressHalfmaps.Visibility = Visibility.Visible;

            bool DoLowpass = (bool)RadioHalfmap2Lowpass.IsChecked;
            float LowpassNyquist = (float)HalfmapPixelSize * 2 / HalfmapLowpass;

            await Task.Run(() =>
            {
                Halfmap1Final?.Dispose();
                Halfmap1Final = null;
                Halfmap2Final?.Dispose();
                Halfmap2Final = null;

                if (Halfmap1 != null)
                {
                    Halfmap1Final = Halfmap1.GetCopyGPU();

                    if (DoLowpass)
                    {
                        Halfmap1Final.Bandpass(0, LowpassNyquist, true);
                        Halfmap2Final = Halfmap1Final.GetCopyGPU();
                    }
                }

                if (Halfmap2 != null && !DoLowpass)
                {
                    Halfmap2Final = Halfmap2.GetCopyGPU();
                }

                Dispatcher.Invoke(() =>
                {
                    RendererHalfmap1.Volume = Halfmap1Final;
                    RendererHalfmap2.Volume = Halfmap2Final;
                });
            });

            ProgressHalfmaps.Visibility = Visibility.Hidden;
            RevalidateTab();
        }

        void UpdateHalfmapsResolution()
        {
            if ((bool)RadioHalfmap2Lowpass.IsChecked)
                UpdateHalfmaps();

            RevalidateTab();
        }

        bool ValidateHalfmaps()
        {
            return Halfmap1Final != null && Halfmap2Final != null;
        }

        private async void ButtonHalfmap1Path_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "MRC file|*.mrc",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Halfmap1 = Image.FromFile(Dialog.FileName);

                if ((bool)RadioHalfmap2File.IsChecked && Halfmap2 != null && Halfmap1.Dims != Halfmap2.Dims)
                {
                    Halfmap1 = null;
                    await((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", "Half-map dimensions don't match.");
                    return;
                }

                float2 MeanStd = MathHelper.MeanAndStd(Halfmap1.GetHostContinuousCopy());
                RendererHalfmap1.Camera.SurfaceThreshold = (decimal)(MeanStd.X + 3 * MeanStd.Y);
                RendererHalfmap2.Camera.SurfaceThreshold = (decimal)(MeanStd.X + 3 * MeanStd.Y);
                MaskThreshold = (decimal)(MeanStd.X + 3 * MeanStd.Y);

                HalfmapPixelSize = (decimal)Halfmap1.PixelSize;

                PathHalfmap1 = Dialog.FileName;
                ButtonHalfmap1Path.Content = Helper.ShortenString(PathHalfmap1, 55);

                UpdateHalfmaps();
            }
        }

        private async void ButtonHalfmap2Path_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "MRC file|*.mrc",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Halfmap2 = Image.FromFile(Dialog.FileName);

                if (Halfmap1 != null && Halfmap1.Dims != Halfmap2.Dims)
                {
                    Halfmap2 = null;
                    await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", "Half-map dimensions don't match.");
                    return;
                }

                PathHalfmap2 = Dialog.FileName;
                ButtonHalfmap2Path.Content = Helper.ShortenString(PathHalfmap2, 55);

                UpdateHalfmaps();
            }
        }

        private async void RadioHalfmap2File_OnChecked(object sender, RoutedEventArgs e)
        {
            if ((bool)RadioHalfmap2File.IsChecked && Halfmap1 != null && Halfmap2 != null && Halfmap1.Dims != Halfmap2.Dims)
                await((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", "Half-map dimensions don't match.");

            UpdateHalfmaps();
        }

        #endregion

        #region Mask

        async void UpdateMask()
        {
            if (!IsInitialized)
                return;

            bool DoNew = (bool)RadioMaskNew.IsChecked;
            bool UseHalfmap2 = (bool)RadioHalfmap2File.IsChecked && Halfmap2 != null;
            float Threshold = (float)MaskThreshold;
            float LowpassNyquist = (float)HalfmapPixelSize * 2 / (float)MaskLowpass;

            ProgressMask.Visibility = Visibility.Visible;

            await Task.Run(() =>
            {
                MaskFinal?.Dispose();
                MaskFinal = null;

                if (!DoNew && Mask != null)
                {
                    MaskFinal = Mask.GetCopyGPU();
                    MaskFinal.Binarize(0.99f);
                }
                else if (DoNew)
                {
                    MaskFinal = Halfmap1.GetCopyGPU();
                    if (UseHalfmap2)
                    {
                        MaskFinal.Add(Halfmap2);
                        MaskFinal.Multiply(0.5f);
                    }

                    MaskFinal.Bandpass(0, LowpassNyquist, true);
                    MaskFinal.Binarize(Threshold);
                }

                Dispatcher.Invoke(() =>
                {
                    RendererMask.Volume = MaskFinal;
                });
            });

            ProgressMask.Visibility = Visibility.Hidden;
            RevalidateTab();
        }

        void UpdateMaskResolution()
        {
            if ((bool)RadioMaskNew.IsChecked)
                UpdateMask();

            RevalidateTab();
        }

        bool ValidateMask()
        {
            return MaskFinal != null && MaskFinal.Dims == Halfmap1.Dims;
        }

        private async void ButtonMaskPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "MRC file|*.mrc",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Mask = Image.FromFile(Dialog.FileName);

                if ((bool)RadioMaskFile.IsChecked && Halfmap1 != null && Halfmap1.Dims != Mask.Dims)
                {
                    Mask = null;
                    await((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", "Mask dimensions don't match half-maps.");
                    return;
                }

                PathMask = Dialog.FileName;
                ButtonMaskPath.Content = Helper.ShortenString(PathMask, 55);

                UpdateMask();
            }
        }

        private void RadioMaskFile_OnChecked(object sender, RoutedEventArgs e)
        {
            UpdateMask();
        }

        #endregion

        #region Particles

        bool PauseParticleUpdates = false;

        async void UpdateParticles()
        {
            if (!IsInitialized || PauseParticleUpdates)
                return;

            bool UseWarp = (bool)RadioParticlesWarp.IsChecked;
            float AngPixCoords = (float)ParticleCoordinatesPixel;
            float AngPixShifts = (float)ParticleShiftsPixel;
            int ResMov = TemporalResMov;
            int ResRot = TemporalResMov;

            TextParticlesError.Visibility = Visibility.Collapsed;
            TextParticlesError.Text = "";
            ProgressParticles.Visibility = Visibility.Visible;

            ParticlesFinal = null;

            await Task.Run(() =>
            {
                if (UseWarp && TableWarp != null)
                {
                    #region Figure out missing sources

                    Dictionary<string, int> ParticleHashes = new Dictionary<string, int>();
                    foreach (var hash in TableWarp.GetColumn("wrpSourceHash"))
                    {
                        if (!ParticleHashes.ContainsKey(hash))
                            ParticleHashes.Add(hash, 0);
                        ParticleHashes[hash]++;
                    }

                    HashSet<string> AvailableHashes = new HashSet<string>(Helper.Combine(ValidSources.Select(s => s.Files.Keys.ToArray())));
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
                        ParticlesFinal[p].ResampleCoordinates(ResMov);
                        ParticlesFinal[p].ResampleAngles(ResRot);
                    }

                    #endregion
                }
                else if (!UseWarp && TableRelion != null)
                {
                    string MicrographColName = TableRelion.HasColumn("rlnMicrographName") ? "rlnMicrographName" : "rlnTomoName";

                    #region Figure out missing and ambigous sources

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
                        int Possibilities = UsedSources.Count(source => source.Files.Values.Contains(imageName));

                        if (Possibilities == 0)
                            NamesNotFound.Add(imageName);
                        else if (Possibilities > 1)
                            NamesAmbiguous.Add(imageName);
                        else
                            NamesGood.Add(imageName);
                    }

                    if (NamesAmbiguous.Count > 0)
                    {
                        Dispatcher.Invoke(() =>
                        {
                            TextParticlesError.Text = $"{NamesAmbiguous.Count} image names are ambiguous between selected data sources!";
                            TextParticlesError.Visibility = Visibility.Visible;
                        });
                    }

                    ParticlesUnmatched = NamesNotFound.Sum(h => ParticleImageNames[h]);
                    ParticlesMatched = TableRelion.RowCount - ParticlesUnmatched;

                    #endregion

                    #region Create particles

                    Dictionary<string, string> ReverseMapping = new Dictionary<string, string>();
                    foreach (var source in UsedSources)
                        foreach (var pair in source.Files)
                            if (NamesGood.Contains(pair.Value))
                                ReverseMapping.Add(pair.Value, pair.Key);

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
                        particle.ResampleCoordinates(ResMov);
                        particle.ResampleAngles(ResRot);
                    }

                    #endregion
                }
                else
                {
                    ParticlesMatched = 0;
                    ParticlesUnmatched = 0;
                }
            });

            ProgressParticles.Visibility = Visibility.Hidden;

            TextParticlesResult.Text = $"{ParticlesMatched}/{ParticlesMatched + ParticlesUnmatched} particles matched to available data sources";

            RevalidateTab();
        }

        bool ValidateParticles()
        {
            return (((bool)RadioParticlesRelion.IsChecked) && string.IsNullOrEmpty(TextParticlesError.Text) && ParticlesMatched > 0) ||
                   (((bool)RadioParticlesWarp.IsChecked) && TableWarp != null && TableWarp.RowCount > 0);
        }

        private async void ButtonParticlesWarpPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR file|*.star",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                TableWarp = new Star(Dialog.FileName);

                if (!TableWarp.HasColumn("wrpCoordinateX1") ||
                    !TableWarp.HasColumn("wrpCoordinateY1") ||
                    !TableWarp.HasColumn("wrpAngleRot1") ||
                    !TableWarp.HasColumn("wrpAngleTilt1") ||
                    !TableWarp.HasColumn("wrpAnglePsi1") ||
                    !TableWarp.HasColumn("wrpSourceHash"))
                {
                    TableWarp = null;
                    await((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", "Table does not contain all essential columns (coordinates, angles, source hash).");
                    return;
                }

                PathWarp = Dialog.FileName;
                ButtonParticlesWarpPath.Content = Helper.ShortenString(PathWarp, 55);

                UpdateParticles();
            }
        }

        private async void ButtonParticlesRelionPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR file|*_data.star",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                bool Is3;
                (TableRelion, Is3) = Star.LoadRelion3Particles(Dialog.FileName);

                string MicrographColName = TableRelion.HasColumn("rlnMicrographName") ? "rlnMicrographName" : "rlnTomoName";

                if (!TableRelion.HasColumn("rlnCoordinateX") ||
                    !TableRelion.HasColumn("rlnCoordinateY") ||
                    !TableRelion.HasColumn("rlnAngleRot") ||
                    !TableRelion.HasColumn("rlnAngleTilt") ||
                    !TableRelion.HasColumn("rlnAnglePsi") ||
                    !TableRelion.HasColumn(MicrographColName))
                {
                    TableRelion = null;
                    await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", "Table does not contain all essential columns (coordinates, angles, micrograph/tomo name).");
                    return;
                }

                if (TableRelion.HasColumn("rlnDetectorPixelSize") && TableRelion.HasColumn("rlnMagnification"))
                {
                    try
                    {
                        decimal DetectorPixel = decimal.Parse(TableRelion.GetRowValue(0, "rlnDetectorPixelSize"), CultureInfo.InvariantCulture) * 1e4M;
                        decimal Mag = decimal.Parse(TableRelion.GetRowValue(0, "rlnMagnification"), CultureInfo.InvariantCulture);

                        PauseParticleUpdates = true;
                        {
                            ParticleCoordinatesPixel = DetectorPixel / Mag;
                            if (!Is3)
                                ParticleShiftsPixel = DetectorPixel / Mag;
                            else
                                ParticleShiftsPixel = 1;
                        }
                        PauseParticleUpdates = false;
                    }
                    catch { }
                }
                else if (TableRelion.HasColumn("rlnImagePixelSize"))
                {
                    decimal PixelSize = decimal.Parse(TableRelion.GetRowValue(0, "rlnImagePixelSize"), CultureInfo.InvariantCulture);
                    ParticleCoordinatesPixel = PixelSize;
                    ParticleShiftsPixel = 1;
                }

                int NameIndex = TableRelion.GetColumnID(MicrographColName);
                for (int r = 0; r < TableRelion.RowCount; r++)
                    TableRelion.SetRowValue(r, NameIndex, Helper.PathToNameWithExtension(TableRelion.GetRowValue(r, NameIndex)));

                PathRelion = Dialog.FileName;
                ButtonParticlesRelionPath.Content = Helper.ShortenString(PathRelion, 55);

                UpdateParticles();
            }
        }

        private void RadioParticlesWarp_OnChecked(object sender, RoutedEventArgs e)
        {
            UpdateParticles();
        }

        #endregion

        public void Dispose()
        {
            Halfmap1?.Dispose();
            Halfmap2?.Dispose();
            Mask?.Dispose();
        }

        public void DisposeFull()
        {
            Dispose();

            Halfmap1Final?.Dispose();
            Halfmap2Final?.Dispose();
            MaskFinal?.Dispose();
        }
    }

    enum NewSpeciesWizzardState
    {
        Parameters = 0,
        Halfmaps = 1,
        Mask = 2,
        Particles = 3
    }
}
