using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
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
using System.Windows.Threading;
using Warp.Tools;
using Brushes = System.Windows.Media.Brushes;
using Color = System.Windows.Media.Color;
using Pen = System.Windows.Media.Pen;
using Size = System.Windows.Size;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for SingleAxisScatter.xaml
    /// </summary>
    public partial class SingleAxisScatter : UserControl
    {
        #region Dependency properties

        public ObservableCollection<SingleAxisPoint> Points
        {
            get { return (ObservableCollection<SingleAxisPoint>)GetValue(PointsProperty); }
            set { SetValue(PointsProperty, value); }
        }
        public static readonly DependencyProperty PointsProperty = DependencyProperty.Register("Points", typeof(ObservableCollection<SingleAxisPoint>), typeof(SingleAxisScatter), new PropertyMetadata(null, (a, b) => ((SingleAxisScatter)a).Render()));

        public double AxisMin
        {
            get { return (double)GetValue(AxisMinProperty); }
            set { SetValue(AxisMinProperty, value); }
        }
        public static readonly DependencyProperty AxisMinProperty = DependencyProperty.Register("AxisMin", typeof(double), typeof(SingleAxisScatter), new PropertyMetadata(double.NaN, (a,b) => ((SingleAxisScatter)a).Render()));

        public double AxisMax
        {
            get { return (double)GetValue(AxisMaxProperty); }
            set { SetValue(AxisMaxProperty, value); }
        }
        public static readonly DependencyProperty AxisMaxProperty = DependencyProperty.Register("AxisMax", typeof(double), typeof(SingleAxisScatter), new PropertyMetadata(double.NaN, (a, b) => ((SingleAxisScatter)a).Render()));

        public double RangeHighlightMin
        {
            get { return (double)GetValue(RangeHighlightMinProperty); }
            set { SetValue(RangeHighlightMinProperty, value); }
        }
        public static readonly DependencyProperty RangeHighlightMinProperty = DependencyProperty.Register("RangeHighlightMin", typeof(double), typeof(SingleAxisScatter), new PropertyMetadata(0.0, (a, b) => ((SingleAxisScatter)a).AdjustRangeHighlight()));

        public double RangeHighlightMax
        {
            get { return (double)GetValue(RangeHighlightMaxProperty); }
            set { SetValue(RangeHighlightMaxProperty, value); }
        }
        public static readonly DependencyProperty RangeHighlightMaxProperty = DependencyProperty.Register("RangeHighlightMax", typeof(double), typeof(SingleAxisScatter), new PropertyMetadata(1.0, (a, b) => ((SingleAxisScatter)a).AdjustRangeHighlight()));

        public double PointRadius
        {
            get { return (double)GetValue(PointRadiusProperty); }
            set { SetValue(PointRadiusProperty, value); }
        }
        public static readonly DependencyProperty PointRadiusProperty = DependencyProperty.Register("PointRadius", typeof(double), typeof(SingleAxisScatter), new PropertyMetadata(2.5, (a, b) => ((SingleAxisScatter)a).Render()));

        public double Zoom
        {
            get { return (double)GetValue(ZoomProperty); }
            set { SetValue(ZoomProperty, value); }
        }
        public static readonly DependencyProperty ZoomProperty = DependencyProperty.Register("Zoom", typeof(double), typeof(SingleAxisScatter), new PropertyMetadata(1.0, (a, b) => ((SingleAxisScatter)a).Render()));

        public ObservableCollection<Color> PointColors
        {
            get { return (ObservableCollection<Color>)GetValue(PointColorsProperty); }
            set { SetValue(PointColorsProperty, value); }
        }
        public static readonly DependencyProperty PointColorsProperty = DependencyProperty.Register("PointColors", typeof(ObservableCollection<Color>), typeof(SingleAxisScatter), new PropertyMetadata(null, (a, b) => ((SingleAxisScatter)a).Render()));
        
        public int HighlightID
        {
            get { return (int)GetValue(HighlightIDProperty); }
            set { SetValue(HighlightIDProperty, value); }
        }
        public static readonly DependencyProperty HighlightIDProperty = DependencyProperty.Register("HighlightID", typeof(int), typeof(SingleAxisScatter), new PropertyMetadata(-1, (a, b) => ((SingleAxisScatter)a).OnHighlightIDChanged()));

        #endregion

        public event Action<Movie> PointClicked;
        public event EventHandler HighlightIDChanged;

        double ValueMin;
        double ValueMax;
        double StepX;
        double OffsetX;
        double StepY;
        System.Windows.Point[] PointCenters;

        public SingleAxisScatter()
        {
            InitializeComponent();

            Points = new ObservableCollection<SingleAxisPoint>();
            PointColors = new ObservableCollection<Color>();

            SizeChanged += (sender, args) => Render();
        }

        protected override Size MeasureOverride(Size constraint)
        {
            return new Size(1, 1);
        }

        public void Render()
        {
            try
            {
                double CanvasWidth = (ActualWidth - 24) * Zoom;
                double CanvasHeight = Height - 14;

                CanvasPlot.Children.Clear();

                if (CanvasWidth <= 0 || CanvasHeight <= 0)
                    return;

                #region Determine axis range

                ValueMin = double.MaxValue;
                ValueMax = -double.MaxValue;

                if (!double.IsNaN(AxisMin))
                    ValueMin = AxisMin;
                else if (Points != null && Points.Count > 0)
                    foreach (var point in Points)
                    {
                        if (!double.IsNaN(point.Value))
                            ValueMin = Math.Min(ValueMin, point.Value);
                    }
                else
                    ValueMin = 0;

                if (!double.IsNaN(AxisMax))
                    ValueMax = AxisMax;
                else if (Points != null && Points.Count > 0 && Points.Any(v => !double.IsNaN(v.Value)))
                    foreach (var point in Points)
                    {
                        if (!double.IsNaN(point.Value))
                            ValueMax = Math.Max(ValueMax, point.Value);
                    }
                else
                    ValueMax = 1;

                double Range = ValueMax - ValueMin;

                if (ValueMin == ValueMax) // Range = 0, probably only one point
                {
                    ValueMax += 1;
                    ValueMin -= 1;
                }
                else // Make sure there are a few pixels left to draw the points at the extremes
                {
                    if (double.IsNaN(AxisMax))
                        ValueMax += Range / CanvasHeight * PointRadius;
                    if (double.IsNaN(AxisMin))
                        ValueMin -= Range / CanvasHeight * PointRadius;
                }

                Range = ValueMax - ValueMin;

                Dispatcher.Invoke(() =>
                {
                    string FloatFormat = "F0";
                    if (Math.Max(ValueMax, ValueMin) < 100)
                        FloatFormat = "F1";
                    if (Math.Max(ValueMax, ValueMin) < 10)
                        FloatFormat = "F2";

                    TextLineBottom.Text = ValueMin.ToString(FloatFormat, CultureInfo.InvariantCulture);
                    TextLineCenter.Text = ((ValueMin + ValueMax) * 0.5).ToString(FloatFormat, CultureInfo.InvariantCulture);
                    TextLineTop.Text = ValueMax.ToString(FloatFormat, CultureInfo.InvariantCulture);
                });

                #endregion

                #region Adjust range highlight

                double RangeHighlightClampedMin = Math.Max(ValueMin, RangeHighlightMin);
                double RangeHighlightClampedMax = Math.Min(ValueMax, RangeHighlightMax);

                PanelRangeHighlight.Margin = new Thickness(0, 7 + (ValueMax - RangeHighlightClampedMax) / Range * CanvasHeight, 0, 0);
                PanelRangeHighlight.Height = Math.Max(0, RangeHighlightClampedMax - RangeHighlightClampedMin) / Range * CanvasHeight;

                #endregion

                if (Range < 0 || Points == null || Points.Count == 0)
                {
                    ImagePlot.Source = null;
                    return;
                }

                float[] HistogramBins = new float[50];

                DrawingGroup DGroup = new DrawingGroup();
                using (DrawingContext DContext = DGroup.Open())
                {
                    DContext.PushClip(new RectangleGeometry(new Rect(new Size(CanvasWidth, CanvasHeight))));

                    Pen OutlinePen = new Pen(Brushes.Transparent, 0);
                    OutlinePen.Freeze();

                    SolidColorBrush BackgroundBrush = new SolidColorBrush(Colors.Transparent);
                    BackgroundBrush.Freeze();
                    DContext.DrawRectangle(BackgroundBrush, OutlinePen, new Rect(new Size(CanvasWidth, CanvasHeight)));

                    SolidColorBrush[] ColorBrushes = PointColors.Count > 0
                                                         ? PointColors.Select(c =>
                                                         {
                                                             SolidColorBrush Brush = new SolidColorBrush(Color.FromArgb(255, c.R, c.G, c.B));
                                                             Brush.Freeze();
                                                             return Brush;
                                                         }).ToArray()
                                                         : new[] { new SolidColorBrush(Color.FromArgb(150, Colors.DeepSkyBlue.R, Colors.DeepSkyBlue.G, Colors.DeepSkyBlue.B)) };

                    StepX = (CanvasWidth - PointRadius * 2) / Points.Count;
                    OffsetX = StepX / 2 + PointRadius;
                    StepY = CanvasHeight / Range;

                    PointCenters = new System.Windows.Point[Points.Count];

                    for (int i = 0; i < Points.Count; i++)
                    {
                        if (double.IsNaN(Points[i].Value))
                            continue;

                        double X = i * StepX + OffsetX;
                        double Y = (ValueMax - Points[i].Value) * StepY;

                        DContext.DrawEllipse(ColorBrushes[Points[i].ColorID], OutlinePen, new System.Windows.Point(X, Y), PointRadius, PointRadius);

                        PointCenters[i] = new System.Windows.Point(X, Y);

                        HistogramBins[Math.Max(0, Math.Min(HistogramBins.Length - 1, (int)((Points[i].Value - ValueMin) / Range * (HistogramBins.Length - 1) + 0.5)))]++;
                    }
                }

                DrawingImage Plot = new DrawingImage(DGroup);
                Plot.Freeze();
                Dispatcher.Invoke(() => ImagePlot.Source = Plot);

                Dispatcher.Invoke(() =>
                {
                    float[] HistogramConv = new float[HistogramBins.Length];
                    float[] ConvKernel = { 0.11f, 0.37f, 0.78f, 1f, 0.78f, 0.37f, 0.11f };
                    for (int i = 0; i < HistogramBins.Length; i++)
                    {
                        float Sum = 0;
                        float Samples = 0;
                        for (int j = 0; j < ConvKernel.Length; j++)
                        {
                            int ij = i - 3 + j;
                            if (ij < 0 || ij >= HistogramBins.Length)
                                continue;
                            Sum += ConvKernel[j] * HistogramBins[ij];
                            Samples += ConvKernel[j];
                        }
                        HistogramConv[i] = Sum / Samples;
                    }

                    float HistogramMax = MathHelper.Max(HistogramConv);
                    if (HistogramMax > 0)
                        HistogramConv = HistogramConv.Select(v => v / HistogramMax * 16).ToArray();

                    Polygon HistogramPolygon = new Polygon()
                    {
                        Stroke = Brushes.Transparent,
                        Fill = Brushes.Gray,
                        Opacity = 0.15
                    };
                    PointCollection HistogramPoints = new PointCollection(HistogramConv.Length);

                    HistogramPoints.Add(new System.Windows.Point(16, 0));
                    HistogramPoints.Add(new System.Windows.Point(16, CanvasHeight));
                    for (int i = 0; i < HistogramConv.Length; i++)
                    {
                        double X = 15 - HistogramConv[i];
                        double Y = CanvasHeight - (i / (float)(HistogramConv.Length - 1) * CanvasHeight);
                        HistogramPoints.Add(new System.Windows.Point(X, Y));
                    }

                    HistogramPolygon.Points = HistogramPoints;

                    CanvasHistogram.Children.Clear();
                    CanvasHistogram.Children.Add(HistogramPolygon);
                    Canvas.SetRight(HistogramPolygon, 0);
                });
            }
            catch
            {
            }
        }

        private void AdjustRangeHighlight()
        {
            double Range = ValueMax - ValueMin;
            double CanvasHeight = Height - 14;

            if (CanvasHeight <= 0 || Range <= 0)
                return;

            double RangeHighlightClampedMin = Math.Max(ValueMin, RangeHighlightMin);
            double RangeHighlightClampedMax = Math.Min(ValueMax, RangeHighlightMax);

            PanelRangeHighlight.Margin = new Thickness(0, 7 + (ValueMax - RangeHighlightClampedMax) / Range * CanvasHeight, 0, 0);
            PanelRangeHighlight.Height = Math.Max(0, RangeHighlightClampedMax - RangeHighlightClampedMin) / Range * CanvasHeight;
        }

        private void Main_OnPreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            Zoom = Math.Max(1, Math.Min(8, Zoom + Math.Sign(e.Delta) * 0.5));
            Dispatcher.InvokeAsync(() => ImagePlot_OnMouseMove(sender, e), DispatcherPriority.ApplicationIdle);
        }

        private void ImagePlot_OnMouseMove(object sender, MouseEventArgs e)
        {
            if (Zoom > 1)
            {
                System.Windows.Point ViewportPoint = e.GetPosition(ScrollViewerPlot);
                double ScrollFraction = ViewportPoint.X / ScrollViewerPlot.ActualWidth;
                ScrollViewerPlot.ScrollToHorizontalOffset(ScrollFraction * ScrollViewerPlot.ScrollableWidth);
            }

            if (Points == null || Points.Count == 0 || PointCenters == null || PointCenters.Length == 0)
            {
                HighlightID = -1;
                return;
            }

            System.Windows.Point MousePoint = e.GetPosition(ImagePlot);
            int ClosestID = GetClosestPoint(MousePoint);
            List<int> Neighbors = Helper.ArrayOfSequence(Math.Max(0, ClosestID - 5), Math.Min(PointCenters.Length, ClosestID + 5), 1).ToList();
            Neighbors.Sort((a, b) => (PointCenters[a] - MousePoint).Length.CompareTo((PointCenters[b] - MousePoint).Length));
            ClosestID = Neighbors.First();
            System.Windows.Point Closest = PointCenters[ClosestID];

            double Dist = (Closest - MousePoint).Length;
            if (Dist > PointRadius * Zoom * 2)
            {
                if (ImagePlot.ToolTip != null)
                    ((ToolTip)ImagePlot.ToolTip).IsOpen = false;
                //ImagePlot.ToolTip = null;
                HighlightID = -1;
                return;
            }

            HighlightID = ClosestID;

            if (ImagePlot.ToolTip != null && ImagePlot.ToolTip.GetType() == typeof(ToolTip))
            {
                ((ToolTip)ImagePlot.ToolTip).Content = (ClosestID + 1) + ": " +
                                                       (Points[ClosestID].Context != null ? Points[ClosestID].Context.RootName + ", " : "") +
                                                       Points[ClosestID].Value.ToString("F3", CultureInfo.InvariantCulture);
                ((ToolTip)ImagePlot.ToolTip).IsOpen = false;
                ((ToolTip)ImagePlot.ToolTip).IsOpen = true;
            }
        }

        private int GetClosestPoint(System.Windows.Point point)
        {
            return Math.Max(0, Math.Min(Points.Count - 1, (int)Math.Round((point.X - OffsetX) / StepX)));
        }

        private void ImagePlot_OnMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (Points == null || Points.Count == 0 || PointCenters == null || PointCenters.Length == 0)
                return;

            System.Windows.Point MousePoint = e.GetPosition(ImagePlot);
            int ClosestID = GetClosestPoint(MousePoint);
            List<int> Neighbors = Helper.ArrayOfSequence(Math.Max(0, ClosestID - 5), Math.Min(PointCenters.Length, ClosestID + 5), 1).ToList();
            Neighbors.Sort((a, b) => (PointCenters[a] - MousePoint).Length.CompareTo((PointCenters[b] - MousePoint).Length));
            ClosestID = Neighbors.First();
            System.Windows.Point Closest = PointCenters[ClosestID];

            double Dist = (Closest - MousePoint).Length;
            if (Dist > PointRadius * Zoom * 2)
                return;

            Dispatcher.InvokeAsync(() =>
            {
                if (ImagePlot.ToolTip != null)
                    ((ToolTip)ImagePlot.ToolTip).IsOpen = false;
            });

            PointClicked?.Invoke(Points[ClosestID].Context);
        }

        private void ImagePlot_OnMouseLeave(object sender, MouseEventArgs e)
        {
            Dispatcher.InvokeAsync(() =>
            {
                if (ImagePlot.ToolTip != null)
                    ((ToolTip)ImagePlot.ToolTip).IsOpen = false;
            });
        }

        private void OnHighlightIDChanged()
        {
            HighlightIDChanged?.Invoke(this, null);

            try
            {
                CanvasPlot.Children.Clear();

                if (Points == null ||
                    HighlightID < 0 ||
                    HighlightID >= Points.Count ||
                    double.IsNaN(Points[HighlightID].Value))
                    return;

                Ellipse PointOutline = new Ellipse
                {
                    Width = PointRadius * 2 + 4,
                    Height = PointRadius * 2 + 4,
                    Stroke = Brushes.Gray,
                    StrokeThickness = 3,
                    IsHitTestVisible = false
                };
                CanvasPlot.Children.Add(PointOutline);
                Canvas.SetLeft(PointOutline, PointCenters[HighlightID].X - PointRadius - 2);
                Canvas.SetTop(PointOutline, PointCenters[HighlightID].Y - PointRadius - 2);

                if (Zoom > 1 && !IsMouseOver)
                {
                    double ScrollFraction = PointCenters[HighlightID].X / ImagePlot.ActualWidth;
                    ScrollViewerPlot.ScrollToHorizontalOffset(ScrollFraction * ScrollViewerPlot.ScrollableWidth);
                }
            }
            catch { }
        }
    }

    public struct SingleAxisPoint
    {
        public double Value;
        public int ColorID;
        public Movie Context;

        public SingleAxisPoint(double value, int colorID, Movie context)
        {
            Value = value;
            ColorID = colorID;
            Context = context;
        }
    }
}
