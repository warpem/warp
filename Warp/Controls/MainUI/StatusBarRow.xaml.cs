using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for StatusBarRow.xaml
    /// </summary>
    public partial class StatusBarRow : UserControl
    {
        public List<Movie> RowItems
        {
            get { return (List<Movie>)GetValue(ItemsProperty); }
            set { SetValue(ItemsProperty, value); }
        }
        public static readonly DependencyProperty ItemsProperty = DependencyProperty.Register("RowItems", typeof(List<Movie>), typeof(StatusBarRow), new PropertyMetadata(new List<Movie>(), (sender, e) => ((StatusBarRow)sender).ItemsChanged(sender, e)));

        public List<Movie> AllItems;

        public Movie ActiveItem
        {
            get { return (Movie)GetValue(ActiveItemProperty); }
            set { SetValue(ActiveItemProperty, value); }
        }
        public static readonly DependencyProperty ActiveItemProperty = DependencyProperty.Register("ActiveItem", typeof(Movie), typeof(StatusBarRow), new PropertyMetadata(null, (sender, e) => ((StatusBarRow)sender).OnActiveItemChanged(sender, e)));
        
        public double ItemWidth
        {
            get { return (double)GetValue(ItemWidthProperty); }
            set { SetValue(ItemWidthProperty, value); }
        }
        public static readonly DependencyProperty ItemWidthProperty = DependencyProperty.Register("ItemWidth", typeof(double), typeof(StatusBarRow), new PropertyMetadata(4.0, (sender, e) => ((StatusBarRow)sender).UpdateElements()));
        
        public bool IsMinimized
        {
            get { return (bool)GetValue(IsMinimizedProperty); }
            set { SetValue(IsMinimizedProperty, value); }
        }
        public static readonly DependencyProperty IsMinimizedProperty = DependencyProperty.Register("IsMinimized", typeof(bool), typeof(StatusBarRow), new PropertyMetadata(false, (sender, e) =>
        {
            ((StatusBarRow)sender).UpdateTracker();
            ((StatusBarRow)sender).UpdateHighlightTracker();
        }));
        
        public Movie HighlightItem
        {
            get { return (Movie)GetValue(HighlightItemProperty); }
            set { SetValue(HighlightItemProperty, value); }
        }
        public static readonly DependencyProperty HighlightItemProperty = DependencyProperty.Register("HighlightItem", typeof(Movie), typeof(StatusBarRow), new PropertyMetadata(null, (sender, e) => ((StatusBarRow)sender).OnHighlightItemChanged(sender, e)));



        public event Action ActiveItemStatusChanged;
        public event Action<Movie> ActiveItemChanged;
        public event Action<Movie> HighlightItemChanged;

        private static SolidColorBrush BrushProcessed = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#cc77cd77"));
        private static SolidColorBrush BrushOutdated = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ccffc456"));
        private static SolidColorBrush BrushFilteredOut = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#cc7edfff"));
        private static SolidColorBrush BrushUnprocessed = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ccff7d7d"));
        private static SolidColorBrush BrushDeselected = new SolidColorBrush(Colors.LightGray);

        private static SolidColorBrush BrushProcessedOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ff75f575"));
        private static SolidColorBrush BrushOutdatedOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ffffc456"));
        private static SolidColorBrush BrushFilteredOutOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ff7edfff"));
        private static SolidColorBrush BrushUnprocessedOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ffff7d7d"));


        public StatusBarRow()
        {
            InitializeComponent();

            SizeChanged += (sender, e) =>
            {
                if (e.NewSize.Width != e.PreviousSize.Width)
                    UpdateElements();
            };
        }

        private void ItemsChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            UpdateElements();
        }

        private void OnActiveItemChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ActiveItemChanged?.Invoke(ActiveItem);
            UpdateTracker();
        }

        private void OnHighlightItemChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            HighlightItemChanged?.Invoke(HighlightItem);
            UpdateHighlightTracker();
        }

        private void Item_ProcessingChanged(object sender, EventArgs e)
        {
            DispatchUpdateElements();
        }

        private void DispatchUpdateElements()
        {
            Dispatcher.Invoke(() => UpdateElements());
        }

        public void UpdateElements()
        {
            PanelSegments.Children.Clear();

            ProcessingOptionsMovieCTF OptionsCTF = WarpRuntime.MainWindow.Options.GetProcessingMovieCTF();
            ProcessingOptionsMovieMovement OptionsMovement = WarpRuntime.MainWindow.Options.GetProcessingMovieMovement();
            ProcessingOptionsBoxNet OptionsBoxNet = WarpRuntime.MainWindow.Options.GetProcessingBoxNet();
            ProcessingOptionsMovieExport OptionsExport = WarpRuntime.MainWindow.Options.GetProcessingMovieExport();

            List<Tuple<Movie, ProcessingStatus>> WithStatus = new List<Tuple<Movie, ProcessingStatus>>(RowItems.Count);
            foreach (var item in RowItems)
                WithStatus.Add(new Tuple<Movie, ProcessingStatus>(item, StatusBar.GetMovieProcessingStatus(item, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, WarpRuntime.MainWindow.Options)));


            if (RowItems.Count == 0 || AllItems == null || ActualWidth == 0 || ItemWidth == 0) // Hide unnecessary elements.
            {
                HideElements();
            }
            else
            {
                ShowElements();

                // Create colored navigation bar.

                ProcessingStatus CurrentStatus = WithStatus[0].Item2;
                int CurrentSteps = 0;
                double OverallOffsetX = 0;

                foreach (var movie in WithStatus)
                {
                    if (movie.Item2 != CurrentStatus)
                    {
                        Rectangle Segment = new Rectangle
                        {
                            Width = CurrentSteps * ItemWidth + 0.3,
                            Height = 12,
                            Fill = StatusBar.StatusToBrush(CurrentStatus),
                            SnapsToDevicePixels = false
                        };
                        PanelSegments.Children.Add(Segment);
                        Canvas.SetLeft(Segment, OverallOffsetX);

                        CurrentStatus = movie.Item2;
                        OverallOffsetX += CurrentSteps * ItemWidth;
                        CurrentSteps = 0;
                    }

                    CurrentSteps++;
                }
                if (CurrentSteps > 0)
                {
                    Rectangle Segment = new Rectangle
                    {
                        Width = CurrentSteps * ItemWidth,
                        Height = 12,
                        Fill = StatusBar.StatusToBrush(CurrentStatus),
                        SnapsToDevicePixels = false
                    };
                    PanelSegments.Children.Add(Segment);
                    Canvas.SetLeft(Segment, OverallOffsetX);
                }

                UpdateTracker();
            }
        }

        private void UpdateTracker()
        {
            if (ActiveItem != null && RowItems.Count > 0 && AllItems != null && RowItems.IndexOf(ActiveItem) >= 0)
            {
                if (!IsMinimized)
                {
                    PathPosition.Visibility = Visibility.Visible;
                    PathPositionInverted.Visibility = Visibility.Collapsed;
                    CheckCurrentName.Visibility = Visibility.Visible;
                    TextCurrentName.Visibility = Visibility.Visible;
                }
                else
                {
                    PathPosition.Visibility = Visibility.Collapsed;
                    PathPositionInverted.Visibility = Visibility.Visible;
                    CheckCurrentName.Visibility = Visibility.Collapsed;
                    TextCurrentName.Visibility = Visibility.Collapsed;
                }

                int PositionIndex = RowItems.IndexOf(ActiveItem);

                double IdealOffset = (PositionIndex + 0.5) * ItemWidth;
                PathPosition.Margin =
                    new Thickness(Math.Max(0, Math.Min(IdealOffset - PathPosition.ActualWidth / 2, ActualWidth - PathPosition.ActualWidth)),
                                  0, 0, 0);
                PathPositionInverted.Margin =
                    new Thickness(Math.Max(0, Math.Min(IdealOffset - PathPosition.ActualWidth / 2, ActualWidth - PathPosition.ActualWidth)),
                                  0, 0, 0);

                string TextString = $"{ActiveItem.RootName} ({AllItems.IndexOf(ActiveItem) + 1}/{AllItems.Count})";
                Size TextSize = MeasureString(TextString);
                CheckCurrentName.Margin =
                    new Thickness(Math.Max(0, Math.Min(IdealOffset - TextSize.Width / 2, ActualWidth - TextSize.Width) - 24),
                                  6, 0, 7);
                TextCurrentName.Text = TextString;

                ProcessingOptionsMovieCTF OptionsCTF = WarpRuntime.MainWindow.Options.GetProcessingMovieCTF();
                ProcessingOptionsMovieMovement OptionsMovement = WarpRuntime.MainWindow.Options.GetProcessingMovieMovement();
                ProcessingOptionsBoxNet OptionsBoxNet = WarpRuntime.MainWindow.Options.GetProcessingBoxNet();
                ProcessingOptionsMovieExport OptionsExport = WarpRuntime.MainWindow.Options.GetProcessingMovieExport();

                PenCurrentName.Brush = StatusBar.StatusToBrush(StatusBar.GetMovieProcessingStatus(ActiveItem, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, WarpRuntime.MainWindow.Options));
                //PenCurrentName.Brush.Opacity = 0.3;
            }
            else
            {
                PathPosition.Visibility = Visibility.Collapsed;
                PathPositionInverted.Visibility = Visibility.Collapsed;
                CheckCurrentName.Visibility = Visibility.Collapsed;
                TextCurrentName.Visibility = Visibility.Collapsed;
            }
        }

        private void UpdateHighlightTracker()
        {
            if (HighlightItem != null && RowItems.Count > 0 && AllItems != null && RowItems.IndexOf(HighlightItem) >= 0)
            {
                PathPosition.Opacity = 0.3;
                CheckCurrentName.Opacity = 0.5;
                BlurCurrentName.Radius = 6;

                PathHighlightPosition.Visibility = Visibility.Visible;
                CheckHighlightCurrentName.Visibility = Visibility.Visible;

                int PositionIndex = RowItems.IndexOf(HighlightItem);

                double IdealOffset = (PositionIndex + 0.5) * ItemWidth;
                PathHighlightPosition.Margin = new Thickness(Math.Max(0, Math.Min(IdealOffset - PathHighlightPosition.ActualWidth / 2, ActualWidth - PathHighlightPosition.ActualWidth)),
                                                             0, 0, 0);

                string TextString = $"{HighlightItem.RootName} ({AllItems.IndexOf(HighlightItem) + 1}/{AllItems.Count})";
                Size TextSize = MeasureString(TextString);
                CheckHighlightCurrentName.Margin = new Thickness(Math.Max(0, Math.Min(IdealOffset - TextSize.Width / 2, ActualWidth - TextSize.Width) - 24),
                                                        6, 0, 7);
                TextHighlightCurrentName.Text = TextString;
                CheckHighlightCurrentName.IsChecked = HighlightItem.UnselectManual != null ? !(bool)HighlightItem.UnselectManual : (bool?)null;

                ProcessingOptionsMovieCTF OptionsCTF = WarpRuntime.MainWindow.Options.GetProcessingMovieCTF();
                ProcessingOptionsMovieMovement OptionsMovement = WarpRuntime.MainWindow.Options.GetProcessingMovieMovement();
                ProcessingOptionsBoxNet OptionsBoxNet = WarpRuntime.MainWindow.Options.GetProcessingBoxNet();
                ProcessingOptionsMovieExport OptionsExport = WarpRuntime.MainWindow.Options.GetProcessingMovieExport();

                PenHighlightCurrentName.Brush = StatusBar.StatusToBrush(StatusBar.GetMovieProcessingStatus(HighlightItem, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, WarpRuntime.MainWindow.Options));

                PanelSegmentHighlight.Children.Clear();
                Rectangle Segment = new Rectangle
                {
                    Width = ItemWidth,
                    Height = 12,
                    Fill = new SolidColorBrush(Colors.Transparent),
                    Stroke = PathHighlightPosition.Fill,
                    StrokeThickness = 1,
                    SnapsToDevicePixels = false
                };
                PanelSegmentHighlight.Children.Add(Segment);
                Canvas.SetLeft(Segment, PositionIndex * ItemWidth);

                if (File.Exists(HighlightItem.ThumbnailsPath))
                {
                    ImageSource Image = new BitmapImage(new Uri(HighlightItem.ThumbnailsPath));
                    ImageThumbnail.Source = Image;

                    PopupThumbnail.HorizontalOffset = Mouse.GetPosition(PanelSegments).X - Image.Width / 2;
                    ImageThumbnail.Visibility = Visibility.Visible;
                }
                else
                {
                    PopupThumbnail.HorizontalOffset = Mouse.GetPosition(PanelSegments).X;
                    ImageThumbnail.Visibility = Visibility.Collapsed;
                }

                #region Figure out processing status

                bool DoCTF = WarpRuntime.MainWindow.Options.ProcessCTF;
                bool DoMovement = WarpRuntime.MainWindow.Options.ProcessMovement;
                bool DoPicking = WarpRuntime.MainWindow.Options.ProcessPicking;
                bool DoExport = OptionsExport.DoAverage || OptionsExport.DoStack || OptionsExport.DoDeconv || DoPicking;

                bool NeedsNewCTF = OptionsCTF != HighlightItem.OptionsCTF;
                bool NeedsNewMotion = OptionsMovement != HighlightItem.OptionsMovement;
                bool NeedsNewPicking = (OptionsBoxNet != HighlightItem.OptionsBoxNet ||
                                        NeedsNewMotion);
                bool NeedsNewExport = (DoMovement && NeedsNewMotion) ||
                                       OptionsExport != HighlightItem.OptionsMovieExport ||
                                       (OptionsExport.DoDeconv && NeedsNewCTF && DoCTF);

                if (HighlightItem.OptionsCTF != null)
                    IndicatorCTF.Foreground = NeedsNewCTF ? (DoCTF ? BrushOutdatedOpaque : BrushDeselected) : BrushProcessedOpaque;
                else if (DoCTF)
                    IndicatorCTF.Foreground = BrushUnprocessedOpaque;
                else
                    IndicatorCTF.Foreground = BrushDeselected;

                if (HighlightItem.OptionsMovement != null)
                    IndicatorMotion.Foreground = NeedsNewMotion ? (DoMovement ? BrushOutdatedOpaque : BrushDeselected) : BrushProcessedOpaque;
                else if (DoMovement && HighlightItem.GetType() == typeof(Movie))
                    IndicatorMotion.Foreground = BrushUnprocessedOpaque;
                else
                    IndicatorMotion.Foreground = BrushDeselected;

                if (HighlightItem.OptionsBoxNet != null)
                    IndicatorPicking.Foreground = NeedsNewPicking ? (DoPicking ? BrushOutdatedOpaque : BrushDeselected) : BrushProcessedOpaque;
                else if (DoPicking && HighlightItem.GetType() == typeof(Movie))
                    IndicatorPicking.Foreground = BrushUnprocessedOpaque;
                else
                    IndicatorPicking.Foreground = BrushDeselected;

                if (HighlightItem.OptionsMovieExport != null)
                    IndicatorExport.Foreground = NeedsNewExport ? (DoExport ? BrushOutdatedOpaque : BrushDeselected) : BrushProcessedOpaque;
                else if (DoExport && HighlightItem.GetType() == typeof(Movie))
                    IndicatorExport.Foreground = BrushUnprocessedOpaque;
                else
                    IndicatorExport.Foreground = BrushDeselected;

                #endregion

                PopupThumbnail.PlacementTarget = PanelSegments;
                PopupThumbnail.VerticalOffset = -12;
                PopupThumbnail.IsOpen = true;
            }
            else
            {
                PathPosition.Opacity = 1;
                CheckCurrentName.Opacity = 1;
                BlurCurrentName.Radius = 0;

                PathHighlightPosition.Visibility = Visibility.Collapsed;
                CheckHighlightCurrentName.Visibility = Visibility.Collapsed;

                PanelSegmentHighlight.Children.Clear();

                PopupThumbnail.IsOpen = false;
            }
        }

        private void ShowElements()
        {
            PanelSegments.Visibility = Visibility.Visible;
            PathPosition.Visibility = Visibility.Visible;
            TextCurrentName.Visibility = Visibility.Visible;
            CheckCurrentName.Visibility = Visibility.Visible;
        }

        private void HideElements()
        {
            PanelSegments.Visibility = Visibility.Hidden;
            PathPosition.Visibility = Visibility.Hidden;
            TextCurrentName.Visibility = Visibility.Hidden;
            CheckCurrentName.Visibility = Visibility.Hidden;
        }

        private void MainGrid_OnMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (RowItems.Count == 0)
                return;

            int NewIndex = (int)((float)e.GetPosition(this).X / ItemWidth);
            NewIndex = Math.Max(0, Math.Min(NewIndex, RowItems.Count - 1));

            ActiveItem = RowItems[NewIndex];
        }

        private void PanelSegments_OnMouseMove(object sender, MouseEventArgs e)
        {
            if (RowItems == null || RowItems.Count == 0 || CheckCurrentName.IsMouseOver)
            {
                HighlightItem = null;
            }
            else
            {
                int ID = (int)(e.GetPosition(PanelSegments).X / ItemWidth);
                HighlightItem = RowItems[Math.Max(0, Math.Min(RowItems.Count - 1, ID))];
            }
        }

        private void PanelSegments_OnMouseLeave(object sender, MouseEventArgs e)
        {
            HighlightItem = null;
            UpdateHighlightTracker();
        }

        private void CheckCurrentName_OnClick(object sender, RoutedEventArgs e)
        {
            ActiveItemStatusChanged?.Invoke();
        }

        private Size MeasureString(string candidate)
        {
            FormattedText Test = new FormattedText(candidate,
                                                   CultureInfo.CurrentUICulture,
                                                   FlowDirection.LeftToRight,
                                                   new Typeface(TextCurrentName.FontFamily, TextCurrentName.FontStyle, TextCurrentName.FontWeight, TextCurrentName.FontStretch),
                                                   TextCurrentName.FontSize,
                                                   Brushes.Black);

            return new Size(Test.Width, Test.Height);
        }
    }
}
