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
    public partial class DualAxisScatter : UserControl
    {
        #region Dependency properties

        public ObservableCollection<DualAxisPoint> Points
        {
            get { return (ObservableCollection<DualAxisPoint>)GetValue(PointsProperty); }
            set { SetValue(PointsProperty, value); }
        }
        public static readonly DependencyProperty PointsProperty = DependencyProperty.Register("Points", typeof(ObservableCollection<DualAxisPoint>), typeof(DualAxisScatter), new PropertyMetadata(null, (a, b) => ((DualAxisScatter)a).Render()));

        public double AxisMax
        {
            get { return (double)GetValue(AxisMaxProperty); }
            set { SetValue(AxisMaxProperty, value); }
        }
        public static readonly DependencyProperty AxisMaxProperty = DependencyProperty.Register("AxisMax", typeof(double), typeof(DualAxisScatter), new PropertyMetadata(double.NaN, (a, b) => ((DualAxisScatter)a).Render()));

        public double PointRadius
        {
            get { return (double)GetValue(PointRadiusProperty); }
            set { SetValue(PointRadiusProperty, value); }
        }
        public static readonly DependencyProperty PointRadiusProperty = DependencyProperty.Register("PointRadius", typeof(double), typeof(DualAxisScatter), new PropertyMetadata(2.5, (a, b) => ((DualAxisScatter)a).Render()));

        public double Zoom
        {
            get { return (double)GetValue(ZoomProperty); }
            set { SetValue(ZoomProperty, value); }
        }
        public static readonly DependencyProperty ZoomProperty = DependencyProperty.Register("Zoom", typeof(double), typeof(DualAxisScatter), new PropertyMetadata(1.0, (a, b) => ((DualAxisScatter)a).Render()));

        public ObservableCollection<Color> PointColors
        {
            get { return (ObservableCollection<Color>)GetValue(PointColorsProperty); }
            set { SetValue(PointColorsProperty, value); }
        }
        public static readonly DependencyProperty PointColorsProperty = DependencyProperty.Register("PointColors", typeof(ObservableCollection<Color>), typeof(DualAxisScatter), new PropertyMetadata(null, (a, b) => ((DualAxisScatter)a).Render()));

        public int HighlightID
        {
            get { return (int)GetValue(HighlightIDProperty); }
            set { SetValue(HighlightIDProperty, value); }
        }
        public static readonly DependencyProperty HighlightIDProperty = DependencyProperty.Register("HighlightID", typeof(int), typeof(DualAxisScatter), new PropertyMetadata(-1, (a, b) => ((DualAxisScatter)a).OnHighlightIDChanged()));

        #endregion

        public event Action<Movie> PointClicked;
        public event EventHandler HighlightIDChanged;

        double ValueMin;
        double ValueMax;
        double StepX;
        double OffsetX;
        double StepY;
        System.Windows.Point[] PointCenters;

        public DualAxisScatter()
        {
            InitializeComponent();

            Points = new ObservableCollection<DualAxisPoint>();
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
                double CanvasWidth = ActualWidth;
                double CanvasHeight = ActualHeight;
                double FullRange = AxisMax * 2;

                CanvasPlot.Children.Clear();

                if (double.IsNaN(AxisMax))
                    return;

                if (CanvasWidth <= 0 || CanvasHeight <= 0)
                    return;

                if (Points == null || Points.Count == 0)
                {
                    ImagePlot.Source = null;
                    return;
                }

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
                                                             SolidColorBrush Brush = new SolidColorBrush(Color.FromArgb(120, c.R, c.G, c.B));
                                                             Brush.Freeze();
                                                             return Brush;
                                                         }).ToArray()
                                                         : new[] { new SolidColorBrush(Color.FromArgb(150, Colors.DeepSkyBlue.R, Colors.DeepSkyBlue.G, Colors.DeepSkyBlue.B)) };
                    
                    PointCenters = new System.Windows.Point[Points.Count];

                    for (int i = 0; i < Points.Count; i++)
                    {
                        if (double.IsNaN(Points[i].X) || double.IsNaN(Points[i].Y))
                            continue;

                        //if (Math.Abs(Points[i].X) > AxisMax || Math.Abs(Points[i].Y) > AxisMax)
                        //    continue;

                        double X = Points[i].X / FullRange * CanvasWidth + CanvasWidth / 2;
                        double Y = Points[i].Y / FullRange * CanvasHeight + CanvasHeight / 2;

                        DContext.DrawEllipse(ColorBrushes[Points[i].ColorID], OutlinePen, new System.Windows.Point(X, Y), PointRadius, PointRadius);

                        PointCenters[i] = new System.Windows.Point(X, Y);
                    }
                }

                DrawingImage Plot = new DrawingImage(DGroup);
                Plot.Freeze();
                Dispatcher.Invoke(() => ImagePlot.Source = Plot);
            }
            catch
            {
            }
        }

        private void ImagePlot_OnMouseMove(object sender, MouseEventArgs e)
        {
            if (Points == null || Points.Count == 0 || PointCenters == null || PointCenters.Length == 0)
            {
                HighlightID = -1;
                return;
            }

            System.Windows.Point MousePoint = e.GetPosition(ImagePlot);
            int ClosestID = GetClosestPoint(MousePoint);
            System.Windows.Point Closest = PointCenters[ClosestID];

            double Dist = (Closest - MousePoint).Length;
            if (Dist > PointRadius * 2)
            {
                if (ImagePlot.ToolTip != null)
                    ((ToolTip)ImagePlot.ToolTip).IsOpen = false;
                ImagePlot.ToolTip = null;
                HighlightID = -1;
                return;
            }

            HighlightID = ClosestID;

            if (ImagePlot.ToolTip == null || ImagePlot.ToolTip.GetType() != typeof(ToolTip))
                ImagePlot.ToolTip = new ToolTip();
            ((ToolTip)ImagePlot.ToolTip).Content = (ClosestID + 1) + ": " +
                                                   (Points[ClosestID].Context != null ? Points[ClosestID].Context.RootName + ", " : "") +
                                                   Points[ClosestID].Label;
            ((ToolTip)ImagePlot.ToolTip).IsOpen = false;
            ((ToolTip)ImagePlot.ToolTip).IsOpen = true;
        }

        private int GetClosestPoint(System.Windows.Point point)
        {
            if (PointCenters == null || PointCenters.Length == 0)
                return 0;

            double X = point.X;
            double Y = point.Y;

            double ClosestDistance = double.MaxValue;
            int ClosestID = 0;
            for (int i = 0; i < PointCenters.Length; i++)
            {
                double XX = PointCenters[i].X - X;
                XX *= XX;
                double YY = PointCenters[i].Y - Y;
                YY *= YY;
                double Distance = XX + YY;
                if (Distance <= ClosestDistance)
                {
                    ClosestDistance = Distance;
                    ClosestID = i;
                }
            }

            return ClosestID;
        }

        private void ImagePlot_OnMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (Points == null || Points.Count == 0 || PointCenters == null || PointCenters.Length == 0)
                return;

            System.Windows.Point MousePoint = e.GetPosition(ImagePlot);
            int ClosestID = GetClosestPoint(MousePoint);
            System.Windows.Point Closest = PointCenters[ClosestID];

            double Dist = (Closest - MousePoint).Length;
            if (Dist > PointRadius * 2)
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

            CanvasPlot.Children.Clear();

            if (Points == null ||
                HighlightID < 0 ||
                HighlightID >= Points.Count ||
                double.IsNaN(Points[HighlightID].X) ||
                double.IsNaN(Points[HighlightID].Y))
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
        }
    }

    public struct DualAxisPoint
    {
        public double X, Y;
        public int ColorID;
        public Movie Context;
        public string Label;

        public DualAxisPoint(double x, double y, int colorID, Movie context, string label)
        {
            X = x;
            Y = y;
            ColorID = colorID;
            Context = context;
            Label = label;
        }
    }
}
