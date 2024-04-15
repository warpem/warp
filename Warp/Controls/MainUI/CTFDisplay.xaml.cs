using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
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
using LiveCharts;
using LiveCharts.Defaults;
using Warp.Headers;
using Warp.Tools;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for CTFDisplay.xaml
    /// </summary>
    public partial class CTFDisplay : UserControl
    {
        public Movie Movie
        {
            get { return (Movie)GetValue(MovieProperty); }
            set { SetValue(MovieProperty, value); }
        }
        public static readonly DependencyProperty MovieProperty = DependencyProperty.Register("Movie", typeof(Movie), typeof(CTFDisplay), new PropertyMetadata(null, (sender, e) => ((CTFDisplay)sender).MovieChanged(sender, e)));

        public decimal FittingRangeMin
        {
            get { return (decimal)GetValue(FittingRangeMinProperty); }
            set { SetValue(FittingRangeMinProperty, value); }
        }
        public static readonly DependencyProperty FittingRangeMinProperty =
            DependencyProperty.Register("FittingRangeMin", typeof(decimal), typeof(CTFDisplay), new PropertyMetadata(0M, (sender, e) => ((CTFDisplay)sender).FittingRangeChanged(sender, e)));

        public decimal FittingRangeMax
        {
            get { return (decimal)GetValue(FittingRangeMaxProperty); }
            set { SetValue(FittingRangeMaxProperty, value); }
        }
        public static readonly DependencyProperty FittingRangeMaxProperty =
            DependencyProperty.Register("FittingRangeMax", typeof(decimal), typeof(CTFDisplay), new PropertyMetadata(0M, (sender, e) => ((CTFDisplay)sender).FittingRangeChanged(sender, e)));

        private List<TiltDialItem> TiltKnobs = new List<TiltDialItem>();
        private int _TiltID = 0;
        private int TiltID
        {
            get { return _TiltID; }
            set
            {
                if (value != _TiltID)
                {
                    _TiltID = value;
                    UpdateTiltDial();
                    DispatchMovie_PS2DChanged(this, null);
                    DispatchMovie_CTF1DChanged(this, null);
                    DispatchUpdateTiltInfo();
                    AdjustGridVisibility();
                }
            }
        }

        private bool _ShowSeriesAverage = false;
        private bool ShowSeriesAverage
        {
            get
            {
                return _ShowSeriesAverage;
            }
            set
            {
                if (value != _ShowSeriesAverage)
                {
                    _ShowSeriesAverage = value;
                    ButtonUseAverage.Content = value ? "SHOW TILT AVERAGE" : "SHOW SERIES AVERAGE";
                    Movie_CTF1DChanged(null, null);
                }
            }
        }

        public CTFDisplay()
        {
            InitializeComponent();
            SizeChanged += CTFDisplay_SizeChanged;
        }

        private void CTFDisplay_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            ImagePS2D.Clip = new EllipseGeometry(new Point(ActualWidth / 2, 0), ActualWidth / 2, ActualHeight / 2);
            ImageSimulated2D.Clip = new EllipseGeometry(new Point(ActualWidth / 2, ActualHeight / 2), ActualWidth / 2, ActualHeight / 2);
            ComposeTomoDial();
        }

        private void MovieChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            if (e.OldValue != null)
            {
                Movie E = (Movie)e.OldValue;

                E.CTF1DChanged -= DispatchMovie_CTF1DChanged;
                E.CTF2DChanged -= DispatchMovie_CTF2DChanged;
                E.PS2DChanged -= DispatchMovie_PS2DChanged;

                if (E.GetType() == typeof(TiltSeries))
                {
                    ((TiltSeries)E).TiltCTFProcessed -= TiltSeries_TiltCTFProcessed;
                }
            }

            if (e.NewValue != null)
            {
                Movie E = (Movie)e.NewValue;

                E.CTF1DChanged += DispatchMovie_CTF1DChanged;
                E.CTF2DChanged += DispatchMovie_CTF2DChanged;
                E.PS2DChanged += DispatchMovie_PS2DChanged;

                if (E.GetType() == typeof (TiltSeries))
                {
                    TiltID = ((TiltSeries)E).IndicesSortedDose[0];
                    ((TiltSeries)E).TiltCTFProcessed += TiltSeries_TiltCTFProcessed;
                }
            }

            // Draw the tilt dial
            ComposeTomoDial();

            Movie_CTF1DChanged(null, null);
            Movie_CTF2DChanged(null, null);
            Movie_PS2DChanged(null, null);
            FittingRangeChanged(null, new DependencyPropertyChangedEventArgs());
            DispatchUpdateTiltInfo();

            AdjustGridVisibility();
        }

        private void TiltSeries_TiltCTFProcessed()
        {
            Dispatcher.Invoke(() => UpdateTiltDial());
            DispatchUpdateTiltInfo();
        }

        private void AdjustGridVisibility()
        {
            Dispatcher.Invoke(() =>
            {
                if (Movie != null)
                {
                    if (Movie.OptionsCTF == null) // No estimate available yet.
                    {
                        GridDisplay.Visibility = Visibility.Hidden;
                        GridNoMovie.Visibility = Visibility.Hidden;
                        GridNotProcessed.Visibility = Visibility.Visible;
                    }
                    else // Estimate data available.
                    {
                        GridDisplay.Visibility = Visibility.Visible;
                        GridNoMovie.Visibility = Visibility.Hidden;
                        GridNotProcessed.Visibility = Visibility.Hidden;
                    }

                    if (Movie.GetType() == typeof (TiltSeries))
                    {
                        CanvasDial.Visibility = Visibility.Visible;
                        ButtonUseAverage.Visibility = Visibility.Visible;

                        GridParamsMovie.Visibility = Visibility.Hidden;
                        GridParamsTiltSeries.Visibility = Visibility.Visible;
                    }
                    else
                    {
                        CanvasDial.Visibility = Visibility.Hidden;
                        ButtonUseAverage.Visibility = Visibility.Hidden;

                        GridParamsMovie.Visibility = Visibility.Visible;
                        GridParamsTiltSeries.Visibility = Visibility.Hidden;
                    }
                }
                else // No movie selected.
                {
                    GridDisplay.Visibility = Visibility.Hidden;
                    GridNoMovie.Visibility = Visibility.Visible;
                    GridNotProcessed.Visibility = Visibility.Hidden;
                    CanvasDial.Visibility = Visibility.Hidden;
                }
            });
        }

        private void DispatchMovie_PS2DChanged(object sender, EventArgs e)
        {
            Dispatcher.InvokeAsync(() => Movie_PS2DChanged(sender, e));
        }

        private async void Movie_PS2DChanged(object sender, EventArgs e)
        {
            try
            {
                AdjustGridVisibility();
                ImagePS2D.Visibility = Visibility.Hidden;
                if (Movie == null)
                    return;

                if (Movie.OptionsCTF == null || !File.Exists(Movie.PowerSpectrumPath))
                    return;

                // Check if 2D PS file has the PS for the currently selected tilt
                if (Movie.GetType() == typeof(TiltSeries))
                {
                    MapHeader Header = MapHeader.ReadFromFile(Movie.PowerSpectrumPath);
                    if (Header.Dimensions.Z <= TiltID)
                        return;
                }

                ProgressPS2D.Visibility = Visibility.Visible;
                Movie movie = Movie;

                //await Task.Delay(1000);
                await Task.Run(() =>
                {
                    ImageSource PS2D;

                    unsafe
                    {
                        int Slice = -1;
                        if (movie.GetType() == typeof(TiltSeries))
                            Slice = TiltID;

                        MapHeader Header = MapHeader.ReadFromFile(movie.PowerSpectrumPath);
                        float[] Data = Image.FromFile(movie.PowerSpectrumPath, Slice).GetHostContinuousCopy();

                        int Width = Header.Dimensions.X;
                        int HalfWidth = Width / 2;
                        int Height = Header.Dimensions.Y; // The usual x / 2 + 1

                        int RadiusMin2 = (int)(movie.OptionsCTF.RangeMin * HalfWidth);
                        RadiusMin2 *= RadiusMin2;
                        int RadiusMax2 = (int)(movie.OptionsCTF.RangeMax * HalfWidth);
                        RadiusMax2 *= RadiusMax2;

                        double Sum1 = 0;
                        double Sum2 = 0;
                        int Samples = 0;

                        fixed (float* DataPtr = Data)
                        {
                            float* DataP = DataPtr;
                            for (int y = 0; y < Height; y++)
                            {
                                for (int x = 0; x < Width; x++)
                                {
                                    int XCentered = x - HalfWidth;
                                    int YCentered = Height - 1 - y;
                                    int Radius2 = XCentered * XCentered + YCentered * YCentered;
                                    if (Radius2 >= RadiusMin2 && Radius2 <= RadiusMax2)
                                    {
                                        Sum1 += *DataP;
                                        Sum2 += (*DataP) * (*DataP);
                                        Samples++;
                                    }

                                    DataP++;
                                }
                            }

                            float Mean = (float)(Sum1 / Samples);
                            float Std = (float)(Math.Sqrt(Samples * Sum2 - Sum1 * Sum1) / Samples);
                            float ValueMin = Mean - 1.5f * Std;
                            float ValueMax = Mean + 3.0f * Std;

                            float Range = ValueMax - ValueMin;
                            if (Range <= 0f)
                            {
                                PS2D = BitmapSource.Create(Width, Height, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, new byte[Data.Length], Width);
                            }
                            else
                            {
                                byte[] DataBytes = new byte[Data.Length];
                                fixed (byte* DataBytesPtr = DataBytes)
                                {
                                    for (int y = 0; y < Height; y++)
                                    for (int x = 0; x < Width; x++)
                                        DataBytesPtr[(Height - 1 - y) * Width + x] = (byte)(Math.Max(Math.Min(1f, (DataPtr[y * Width + x] - ValueMin) / Range), 0f) * 255f);
                                }

                                PS2D = BitmapSource.Create(Width, Height, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, DataBytes, Width);
                                PS2D.Freeze();
                            }

                            Dispatcher.Invoke(() =>
                            {
                                ImagePS2D.Source = PS2D;
                                ImagePS2D.Visibility = Visibility.Visible;
                            });
                        }
                    }
                });

                ProgressPS2D.Visibility = Visibility.Hidden;
            }
            catch { }
        }

        private void DispatchMovie_CTF2DChanged(object sender, EventArgs e)
        {
            Dispatcher.InvokeAsync(() => Movie_CTF2DChanged(sender, e));
        }

        private async void Movie_CTF2DChanged(object sender, EventArgs e)
        {
            try
            {
                AdjustGridVisibility();
                ImageSimulated2D.Visibility = Visibility.Hidden;

                if (Movie == null)
                    return;

                if (Movie.OptionsCTF == null || Movie.CTF == null)
                    return;

                this.Width = Movie.OptionsCTF.Window;
                this.Height = Movie.OptionsCTF.Window;
                ProgressCTF2D.Visibility = Visibility.Visible;

                int Width = Movie.OptionsCTF.Window / 2;
                CTF MovieCTF = Movie.CTF;

                //await Task.Delay(1000);
                await Task.Run(() =>
                {
                    ImageSource Simulated2D;

                    unsafe
                    {
                        float2[] SimCoords = new float2[Width * Width];
                        fixed (float2* SimCoordsPtr = SimCoords)
                        {
                            float2* SimCoordsP = SimCoordsPtr;
                            for (int y = 0; y < Width; y++)
                            {
                                int ycoord = Width - 1 - y;
                                int ycoord2 = ycoord * ycoord;
                                for (int x = 0; x < Width; x++)
                                {
                                    int xcoord = x - Width;
                                    *SimCoordsP++ = new float2((float)Math.Sqrt(xcoord * xcoord + ycoord2) / (Width * 2), (float)Math.Atan2(ycoord, xcoord));
                                }
                            }
                        }
                        float[] Sim2D = MovieCTF.Get2D(SimCoords, true, true, true);
                        byte[] Sim2DBytes = new byte[Sim2D.Length];
                        fixed (byte* Sim2DBytesPtr = Sim2DBytes)
                        fixed (float* Sim2DPtr = Sim2D)
                        {
                            byte* Sim2DBytesP = Sim2DBytesPtr;
                            float* Sim2DP = Sim2DPtr;
                            for (int i = 0; i < Width * Width; i++)
                                *Sim2DBytesP++ = (byte)(*Sim2DP++ * 128f + 127f);
                        }

                        Simulated2D = BitmapSource.Create(Width, Width, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, Sim2DBytes, Width);
                        Simulated2D.Freeze();
                    }

                    Dispatcher.Invoke(() =>
                    {
                        ImageSimulated2D.Source = Simulated2D;
                        ImageSimulated2D.Visibility = Visibility.Visible;
                    });
                });

                ProgressCTF2D.Visibility = Visibility.Hidden;
            }
            catch
            {
            }
        }

        private void DispatchMovie_CTF1DChanged(object sender, EventArgs e)
        {
            Dispatcher.InvokeAsync(() => Movie_CTF1DChanged(sender, e));
        }

        private async void Movie_CTF1DChanged(object sender, EventArgs e)
        {
            try
            {
                AdjustGridVisibility();
                if (Movie == null)
                    return;

                TiltSeries Series = Movie as TiltSeries;

                bool NoDataAvailable = false;
                if ((Movie.GetType() == typeof(Movie) || ShowSeriesAverage) && (Movie.PS1D == null || Movie.Simulated1D == null))
                    NoDataAvailable = true;
                else if (Series != null && (Series.TiltPS1D.Count <= TiltID || Series.TiltPS1D[TiltID] == null))
                    NoDataAvailable = true;

                if (NoDataAvailable)
                {
                    Plot1DSeriesExperimental.Values = null;
                    Plot1DSeriesSimulated.Values = null;
                    Plot1DSeriesQuality.Values = null;

                    return;
                }

                //ProgressCTF1D.Visibility = Visibility.Visible;
                Movie movie = Movie;
                decimal fittingRangeMin = movie.CTF.PixelSize * 2 / FittingRangeMin;

                //await Task.Delay(1000);
                await Task.Run(() =>
                {

                    float2[] ExperimentalData = (Series == null || _ShowSeriesAverage) ? movie.PS1D : Series.TiltPS1D[TiltID];
                    float2[] SimulatedData = (Series == null || _ShowSeriesAverage) ? movie.Simulated1D : Series.GetTiltSimulated1D(TiltID);
                    Cubic1D ScaleData = (Series == null || _ShowSeriesAverage) ? movie.SimulatedScale : Series.TiltSimulatedScale[TiltID];

                    int N = ExperimentalData.Length * 2;

                    ChartValues<ObservablePoint> ExperimentalValues = new ChartValues<ObservablePoint>(ExperimentalData.Select(p => new ObservablePoint(p.X * N, p.Y)));
                    ChartValues<ObservablePoint> SimulatedValues = new ChartValues<ObservablePoint>(SimulatedData.Select(p => new ObservablePoint(p.X * N, p.Y)));

                    CTF CTF = (Series == null || _ShowSeriesAverage) ? movie.CTF : Series.GetTiltCTF(TiltID);
                    float[] Quality = CTF.EstimateQuality(ExperimentalData.Select(p => p.Y).ToArray(), ScaleData.Interp(ExperimentalData.Select(p => p.X).ToArray()), (float)fittingRangeMin, 16);

                    Dispatcher.Invoke(() =>
                    {
                        Plot1DSeriesExperimental.Values = ExperimentalValues;
                        Plot1DSeriesSimulated.Values = SimulatedValues;
                        Plot1DSeriesQuality.Values = new ChartValues<double>(Quality.Select(v => (double)Math.Max(0, v)));

                    });
                });

                AdjustYAxis();
                FittingRangeChanged(null, new DependencyPropertyChangedEventArgs());

                //ProgressCTF1D.Visibility = Visibility.Hidden;
            }
            catch
            {
            }
        }

        private void AdjustYAxis()
        {
            if (Movie == null)
                return;

            TiltSeries Series = Movie as TiltSeries;

            if ((Movie.GetType() == typeof (Movie) || ShowSeriesAverage) && (Movie.PS1D == null || Movie.Simulated1D == null))
                return;
            if (Series != null && (Series.TiltPS1D.Count <= TiltID || Series.TiltPS1D[TiltID] == null))
                return;

            float2[] ExperimentalData = (Series == null || _ShowSeriesAverage) ? Movie.PS1D : Series.TiltPS1D[TiltID];
            float2[] SimulatedData = (Series == null || _ShowSeriesAverage) ? Movie.Simulated1D : Series.GetTiltSimulated1D(TiltID);

            decimal fittingRangeMin = Movie.CTF.PixelSize * 2 / FittingRangeMin;
            decimal fittingRangeMax = Movie.CTF.PixelSize * 2 / FittingRangeMax;

            int MinN = (int)(ExperimentalData.Length * fittingRangeMin);
            int N = (int)(ExperimentalData.Length * (fittingRangeMax - fittingRangeMin));

            IEnumerable<float> RelevantExperimental = ExperimentalData.Select(p => p.Y).Skip(MinN).Take(N);
            IEnumerable<float> RelevantSimulated = SimulatedData.Select(p => p.Y).Skip(MinN).Take(N);

            float MinExperimental = MathHelper.Min(RelevantExperimental);
            float MaxExperimental = MathHelper.Max(RelevantExperimental);
            float MinSimulated = MathHelper.Min(RelevantSimulated);
            float MaxSimulated = MathHelper.Max(RelevantSimulated);

            Plot1DAxisY.MinValue = Math.Min(MinExperimental, MinSimulated);
            Plot1DAxisY.MaxValue = Math.Max(MaxExperimental, MaxSimulated) * 1.25f;

            Plot1DAxisY.LabelFormatter = val => val.ToString("F3", CultureInfo.InvariantCulture);
            Plot1DAxisYQuality.LabelFormatter = val => val.ToString("F3", CultureInfo.InvariantCulture);
            Plot1DAxisX.LabelFormatter = val => (Movie.OptionsCTF.Window / val * (double)Movie.OptionsCTF.BinnedPixelSizeMean).ToString("F2") + " Å";
        }

        private void ComposeTomoDial()
        {
            CanvasDial.Children.Clear();
            TiltKnobs.Clear();

            if (Movie == null || Movie.GetType() != typeof (TiltSeries)) return;

            TiltSeries Series = (TiltSeries)Movie;

            for (int i = 0; i < Series.NTilts; i++)
            {
                TiltDialItem Knob = new TiltDialItem();

                Knob.TiltID = i;
                Knob.Angle = Series.Angles[i];
                Knob.Dose = Series.Dose[i];
                Knob.DoProcess = Series.UseTilt[i];

                Knob.Width = 160;
                Knob.Height = 20;

                TransformGroup TGroup = new TransformGroup();
                TGroup.Children.Add(new TranslateTransform(-80, -10));
                TGroup.Children.Add(new RotateTransform(-Knob.Angle));
                Knob.RenderTransform = TGroup;

                TiltKnobs.Add(Knob);
                CanvasDial.Children.Add(Knob);

                double Radius = ActualWidth / 2;
                Canvas.SetLeft(Knob, Math.Cos((Knob.Angle + 180) * Helper.ToRad) * (Radius + 85) + Radius);
                Canvas.SetTop(Knob, -Math.Sin((Knob.Angle + 180) * Helper.ToRad) * (Radius + 85) + Radius);

                Knob.SelectionChanged += (o, args) => TiltID = ((TiltDialItem)o).TiltID;
                Knob.DoProcessChanged += (o, args) =>
                {
                    Series.UseTilt[Knob.TiltID] = (bool)args.NewValue;
                    Series.SaveMeta();
                    UpdateTiltDial();
                };
                Knob.MouseWheel += TiltDial_MouseWheel;
            }

            UpdateTiltDial();
        }

        private void UpdateTiltDial()
        {
            TiltSeries Series = Movie as TiltSeries;

            foreach (var Knob in TiltKnobs)
            {
                Knob.IsSelected = Knob.TiltID == TiltID;

                Brush KnobBrush;
                if (!Knob.DoProcess)
                    KnobBrush = Brushes.DarkGray;
                else if (Series != null && Series.TiltPS1D.Count > Knob.TiltID && Series.TiltPS1D[Knob.TiltID] != null)
                    KnobBrush = Brushes.Green;
                else
                    KnobBrush = Brushes.Red;
                Knob.KnobBrush = KnobBrush;
            }
        }

        private void DispatchUpdateTiltInfo()
        {
            Dispatcher.Invoke(() => UpdateTiltInfo());
        }

        private void UpdateTiltInfo()
        {
            if (Movie != null && Movie.GetType() == typeof(TiltSeries))
            {
                TiltSeries Series = Movie as TiltSeries;

                TextTiltDefocus.Value = Series.GetTiltDefocus(TiltID).ToString("F3", CultureInfo.InvariantCulture);
                TextTiltDefocusDelta.Value = Series.GetTiltDefocusDelta(TiltID).ToString("F3", CultureInfo.InvariantCulture);
                TextTiltDefocusAngle.Value = Series.GetTiltDefocusAngle(TiltID).ToString("F1", CultureInfo.InvariantCulture);
                TextTiltPhase.Value = Series.GetTiltPhase(TiltID).ToString("F2", CultureInfo.InvariantCulture);
                TextTiltResEstimate.Value = Series.CTFResolutionEstimate.ToString("F2", CultureInfo.InvariantCulture);

                float NormalAngle1 = (float)Math.Atan2(Series.PlaneNormal.Y, Series.PlaneNormal.X) * Helper.ToDeg;
                float NormalAngle2 = (float)Math.Asin(new float2(Series.PlaneNormal.X, Series.PlaneNormal.Y).Length()) * Helper.ToDeg;

                TransformPlaneInclination.Angle = -NormalAngle1;
                TextPlaneInclination.Value = NormalAngle2.ToString("F1", CultureInfo.InvariantCulture);

                TextAngleMismatch.Visibility = Series.AreAnglesInverted ? Visibility.Visible : Visibility.Hidden;
            }
        }

        private void TiltDial_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            TiltSeries Series = Movie as TiltSeries;
            if (Series != null)
            {
                List<int> SortedAngle = Series.IndicesSortedAngle.ToList();
                int CurrentPos = SortedAngle.IndexOf(TiltID);
                int Delta = -Math.Sign(e.Delta);
                TiltID = SortedAngle[Math.Max(0, Math.Min(SortedAngle.Count - 1, CurrentPos + Delta))];
            }
        }

        private void FittingRangeChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            Plot1DAxisXSectionConsider.Visibility = Visibility.Hidden;

            if (Movie == null)
                return;

            TiltSeries Series = Movie as TiltSeries;

            if (Movie.GetType() == typeof(Movie) && (Movie.PS1D == null || Movie.Simulated1D == null))
                return;
            if (Series != null && (Series.TiltPS1D.Count <= TiltID || Series.TiltPS1D[TiltID] == null))
                return;

            Plot1DAxisXSectionConsider.Visibility = Visibility.Visible;

            float2[] ExperimentalData = Series == null ? Movie.PS1D : Series.TiltPS1D[TiltID];

            decimal fittingRangeMin = Movie.CTF.PixelSize * 2 / FittingRangeMin;
            decimal fittingRangeMax = Movie.CTF.PixelSize * 2 / FittingRangeMax;

            int N = ExperimentalData.Length;
            Plot1DAxisXSectionConsider.Value = N * (double)fittingRangeMin;
            Plot1DAxisXSectionConsider.SectionWidth = N * (double)(fittingRangeMax - fittingRangeMin);
        }

        private void ButtonUseAverage_OnClick(object sender, RoutedEventArgs e)
        {
            ShowSeriesAverage = !ShowSeriesAverage;
        }
    }
}
