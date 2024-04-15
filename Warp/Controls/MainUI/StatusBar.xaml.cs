using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
//using System.Runtime.Remoting.Channels;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for StatusBar.xaml
    /// </summary>
    public partial class StatusBar : UserControl
    {
        public ObservableCollection<Movie> Items
        {
            get { return (ObservableCollection<Movie>)GetValue(ItemsProperty); }
            set { SetValue(ItemsProperty, value); }
        }
        public static readonly DependencyProperty ItemsProperty = DependencyProperty.Register("Items", typeof (ObservableCollection<Movie>), typeof (StatusBar), new PropertyMetadata(new ObservableCollection<Movie>(), (sender, e) => ((StatusBar)sender).ItemsChanged(sender, e)));

        private List<Movie> FilteredItems = new List<Movie>();

        public Movie ActiveItem
        {
            get { return (Movie)GetValue(ActiveItemProperty); }
            set { SetValue(ActiveItemProperty, value); }
        }
        public static readonly DependencyProperty ActiveItemProperty = DependencyProperty.Register("ActiveItem", typeof(Movie), typeof(StatusBar), 
            new PropertyMetadata(null, (sender, e) => ((StatusBar)sender).ActiveItemChanged(sender, e)));

        public bool IsFilterEnabled
        {
            get { return (bool)GetValue(IsFilterEnabledProperty); }
            set { SetValue(IsFilterEnabledProperty, value); }
        }
        public static readonly DependencyProperty IsFilterEnabledProperty = DependencyProperty.Register("IsFilterEnabled", typeof(bool), typeof(StatusBar), new PropertyMetadata(true, (sender, e) => ((StatusBar)sender).IsFilterEnabledChanged(sender, e)));

        public string FilterSearchPattern
        {
            get { return (string)GetValue(FilterSearchPatternProperty); }
            set { SetValue(FilterSearchPatternProperty, value); }
        }
        public static readonly DependencyProperty FilterSearchPatternProperty = DependencyProperty.Register("FilterSearchPattern", typeof(string), typeof(StatusBar), new PropertyMetadata("", (sender, e) => ((StatusBar)sender).FilterParameterChanged(sender, e)));
        
        public bool FilterIncludeProcessed
        {
            get { return (bool)GetValue(FilterIncludeProcessedProperty); }
            set { SetValue(FilterIncludeProcessedProperty, value); }
        }
        public static readonly DependencyProperty FilterIncludeProcessedProperty = DependencyProperty.Register("FilterIncludeProcessed", typeof(bool), typeof(StatusBar), new PropertyMetadata(true, (sender, e) => ((StatusBar)sender).FilterParameterChanged(sender, e)));

        public bool FilterIncludeOutdated
        {
            get { return (bool)GetValue(FilterIncludeOutdatedProperty); }
            set { SetValue(FilterIncludeOutdatedProperty, value); }
        }
        public static readonly DependencyProperty FilterIncludeOutdatedProperty = DependencyProperty.Register("FilterIncludeOutdated", typeof(bool), typeof(StatusBar), new PropertyMetadata(true, (sender, e) => ((StatusBar)sender).FilterParameterChanged(sender, e)));

        public bool FilterIncludeUnprocessed
        {
            get { return (bool)GetValue(FilterIncludeUnprocessedProperty); }
            set { SetValue(FilterIncludeUnprocessedProperty, value); }
        }
        public static readonly DependencyProperty FilterIncludeUnprocessedProperty = DependencyProperty.Register("FilterIncludeUnprocessed", typeof(bool), typeof(StatusBar), new PropertyMetadata(true, (sender, e) => ((StatusBar)sender).FilterParameterChanged(sender, e)));

        public bool FilterIncludeFilteredOut
        {
            get { return (bool)GetValue(FilterIncludeFilteredOutProperty); }
            set { SetValue(FilterIncludeFilteredOutProperty, value); }
        }
        public static readonly DependencyProperty FilterIncludeFilteredOutProperty = DependencyProperty.Register("FilterIncludeFilteredOut", typeof(bool), typeof(StatusBar), new PropertyMetadata(true, (sender, e) => ((StatusBar)sender).FilterParameterChanged(sender, e)));

        public bool FilterIncludeDeselected
        {
            get { return (bool)GetValue(FilterIncludeDeselectedProperty); }
            set { SetValue(FilterIncludeDeselectedProperty, value); }
        }
        public static readonly DependencyProperty FilterIncludeDeselectedProperty = DependencyProperty.Register("FilterIncludeDeselected", typeof(bool), typeof(StatusBar), new PropertyMetadata(true, (sender, e) => ((StatusBar)sender).FilterParameterChanged(sender, e)));

        public event Action ActiveItemStatusChanged;

        private List<StatusBarRow> RowControls = new List<StatusBarRow>();

        private static SolidColorBrush BrushProcessed = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#cc77cd77"));
        private static SolidColorBrush BrushOutdated = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ccffc456"));
        private static SolidColorBrush BrushFilteredOut = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#cc7edfff"));
        private static SolidColorBrush BrushUnprocessed = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ccff7d7d"));
        private static SolidColorBrush BrushDeselected = new SolidColorBrush(Colors.LightGray);

        private static SolidColorBrush BrushProcessedOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ff75f575"));
        private static SolidColorBrush BrushOutdatedOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ffffc456"));
        private static SolidColorBrush BrushFilteredOutOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ff7edfff"));
        private static SolidColorBrush BrushUnprocessedOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ffff7d7d"));

        public StatusBar()
        {
            SizeChanged += (sender, e) =>
            {
                if (e.NewSize.Width != e.PreviousSize.Width)
                    UpdateElements();
            };
            IsVisibleChanged += (sender, e) => 
            {
                if (IsVisible)
                    UpdateElements();
            };
            InitializeComponent();

            ApplyFilter();
            UpdateElements();
        }

        private void ItemsChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ObservableCollection<Movie> OldValue = e.OldValue as ObservableCollection<Movie>;
            if (OldValue != null)
            {
                OldValue.CollectionChanged -= Movies_CollectionChanged;
                foreach (var item in OldValue)
                    item.ProcessingChanged -= Item_ProcessingChanged;
            }
            ObservableCollection<Movie> NewValue = e.NewValue as ObservableCollection<Movie>;
            if (NewValue != null)
            {
                NewValue.CollectionChanged += Movies_CollectionChanged;
                foreach (var item in NewValue)
                    item.ProcessingChanged += Item_ProcessingChanged;

                if (!NewValue.Contains(ActiveItem)) // Switched to a different folder
                    ActiveItem = null;
            }

            ApplyFilter();
            UpdateElements();
        }

        private void ActiveItemChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            if (ActiveItem != null && !FilteredItems.Contains(ActiveItem))
                IsFilterEnabled = false;

            foreach (var row in RowControls)
                row.ActiveItem = (Movie)e.NewValue;
        }

        private void Movies_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            if (e.OldItems != null)
                foreach (var item in e.OldItems.Cast<Movie>())
                    item.ProcessingChanged -= Item_ProcessingChanged;

            if (e.NewItems != null)
                foreach (var item in e.NewItems.Cast<Movie>())
                    item.ProcessingChanged += Item_ProcessingChanged;

            ApplyFilter();
            UpdateElements();
        }

        private void Item_ProcessingChanged(object sender, EventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                ApplyFilter();
                UpdateElements();
            });
        }

        public void UpdateElements()
        {
            PanelRows.Children.Clear();

            foreach (var row in RowControls)
            {
                row.ActiveItemStatusChanged -= RowActiveItemStatusChanged;
                row.ActiveItemChanged -= RowActiveItemChanged;
                row.HighlightItemChanged -= RowHighlightItemChanged;
            }
            RowControls.Clear();

            ProcessingOptionsMovieCTF OptionsCTF = WarpRuntime.MainWindow.Options.GetProcessingMovieCTF();
            ProcessingOptionsMovieMovement OptionsMovement = WarpRuntime.MainWindow.Options.GetProcessingMovieMovement();
            ProcessingOptionsBoxNet OptionsBoxNet = WarpRuntime.MainWindow.Options.GetProcessingBoxNet();
            ProcessingOptionsMovieExport OptionsExport = WarpRuntime.MainWindow.Options.GetProcessingMovieExport();

            List<Tuple<Movie, ProcessingStatus>> WithStatus = new List<Tuple<Movie, ProcessingStatus>>(FilteredItems.Count);
            foreach (var item in FilteredItems)
                WithStatus.Add(new Tuple<Movie, ProcessingStatus>(item, GetMovieProcessingStatus(item, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, WarpRuntime.MainWindow.Options)));

            // Update number of processed movies.
            int NProcessed = WithStatus.Sum(m => m.Item2 == ProcessingStatus.Processed || m.Item2 == ProcessingStatus.FilteredOut ? 1 : 0);
            int NProcessable = WithStatus.Sum(m => m.Item2 != ProcessingStatus.LeaveOut ? 1 : 0);
            TextNumberProcessed.Text = $"Processed {NProcessed}/{NProcessable}.";
            
            if (FilteredItems.Count == 0 || ActualWidth == 0) // Hide unnecessary elements.
            {
                HideElements();
            }
            else
            {
                ShowElements();

                // Create colored navigation bar.
                double StepLength = Math.Max(2, ActualWidth / FilteredItems.Count);
                int ItemsPerRow = (int)(ActualWidth / StepLength + 1e-3);
                int NRows = (FilteredItems.Count + ItemsPerRow - 1) / ItemsPerRow;

                for (int r = 0; r < NRows; r++)
                {
                    List<Movie> RowSubset = FilteredItems.Skip(r * ItemsPerRow).Take(ItemsPerRow).ToList();
                    StatusBarRow Row = new StatusBarRow();
                    Row.HorizontalAlignment = HorizontalAlignment.Stretch;
                    Row.ItemWidth = StepLength;
                    Row.AllItems = FilteredItems;
                    Row.RowItems = RowSubset;
                    Row.ActiveItem = ActiveItem;
                    Row.ActiveItemStatusChanged += RowActiveItemStatusChanged;
                    Row.ActiveItemChanged += RowActiveItemChanged;
                    Row.HighlightItemChanged += RowHighlightItemChanged;

                    PanelRows.Children.Add(Row);
                    RowControls.Add(Row);
                }
            }
        }

        private void RowActiveItemStatusChanged()
        {
            ActiveItemStatusChanged?.Invoke();
        }

        private void RowActiveItemChanged(Movie item)
        {
            ActiveItem = item;
        }

        private void RowHighlightItemChanged(Movie item)
        {
            if (item == null)
            {
                foreach (var row in RowControls)
                    row.IsMinimized = false;
            }
            else
            {
                foreach (var row in RowControls)
                    row.IsMinimized = !row.RowItems.Contains(item);
            }
        }

        private void ShowElements()
        {
            PanelRows.Visibility = Visibility.Visible;
        }

        private void HideElements()
        {
            PanelRows.Visibility = Visibility.Hidden;
        }

        public static Brush StatusToBrush(ProcessingStatus status)
        {
            if (status == ProcessingStatus.Processed)
                return BrushProcessed.CloneCurrentValue();
            else if (status == ProcessingStatus.Outdated)
                return BrushOutdated.CloneCurrentValue();
            else if (status == ProcessingStatus.Unprocessed)
                return BrushUnprocessed.CloneCurrentValue();
            else if (status == ProcessingStatus.FilteredOut)
                return BrushFilteredOut.CloneCurrentValue();
            else
                return BrushDeselected.CloneCurrentValue();
        }

        private void MainGrid_OnPreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            MoveToOtherItem(-Math.Sign(e.Delta));
        }

        public void MoveToOtherItem(int delta)
        {
            if (FilteredItems.Count == 0)
                return;

            if (ActiveItem == null || !FilteredItems.Contains(ActiveItem))
            {
                ActiveItem = FilteredItems[0];
                return;
            }
            else
            {
                int NewIndex = FilteredItems.IndexOf(ActiveItem) + delta;
                ActiveItem = FilteredItems[Math.Max(0, Math.Min(NewIndex, FilteredItems.Count - 1))];
            }
        }

        public void ShowProgressBar()
        {
            ProgressDiscovery.Visibility = Visibility.Visible;
        }

        public void HideProgressBar()
        {
            ProgressDiscovery.Visibility = Visibility.Hidden;
        }

        public static ProcessingStatus GetMovieProcessingStatus(Movie movie, ProcessingOptionsMovieCTF optionsCTF, ProcessingOptionsMovieMovement optionsMovement, ProcessingOptionsBoxNet optionsBoxNet, ProcessingOptionsMovieExport optionsExport, OptionsWarp options, bool considerFilter = true)
        {
            bool DoCTF = options.ProcessCTF;
            bool DoMovement = options.ProcessMovement && movie.GetType() == typeof(Movie);
            bool DoBoxNet = options.ProcessPicking && movie.GetType() == typeof(Movie);
            bool DoExport = (optionsExport.DoAverage || optionsExport.DoStack || optionsExport.DoDeconv) && movie.GetType() == typeof(Movie);

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
                if (DoCTF && (movie.OptionsCTF == null || movie.OptionsCTF != optionsCTF))
                    Status = ProcessingStatus.Outdated;
                else if (DoMovement && (movie.OptionsMovement == null || movie.OptionsMovement != optionsMovement))
                    Status = ProcessingStatus.Outdated;
                else if (DoBoxNet && (movie.OptionsBoxNet == null || movie.OptionsBoxNet != optionsBoxNet))
                    Status = ProcessingStatus.Outdated;
                else if (DoExport && (movie.OptionsMovieExport == null || movie.OptionsMovieExport != optionsExport))
                    Status = ProcessingStatus.Outdated;
            }

            if (Status == ProcessingStatus.Processed && movie.UnselectFilter && movie.UnselectManual == null && considerFilter)
                Status = ProcessingStatus.FilteredOut;

            return Status;
        }

        private void ButtonGlass_Click(object sender, RoutedEventArgs e)
        {
            IsFilterEnabled = !IsFilterEnabled;
        }

        private void IsFilterEnabledChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            if (IsFilterEnabled)
            {
                GlassPathColor1.Color = Colors.DeepSkyBlue;
                GlassPathColor2.Color = Colors.DeepPink;
                PanelFilterOptions.Visibility = Visibility.Visible;
            }
            else
            {
                GlassPathColor1.Color = Colors.Black;
                GlassPathColor2.Color = Colors.Black;
                PanelFilterOptions.Visibility = Visibility.Collapsed;
            }

            ApplyFilter();
            UpdateElements();
        }

        private void FilterParameterChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ApplyFilter();
            UpdateElements();
        }

        public void ApplyFilter()
        {
            if (!IsFilterEnabled)
            {
                FilteredItems = Items.ToList();
            }
            else
            {
                List<Movie> Result = Items.ToList();

                if (!string.IsNullOrEmpty(FilterSearchPattern))
                {
                    try
                    {
                        Regex Expression = new Regex(FilterSearchPattern);
                        Result = Result.Where(m => Expression.IsMatch(m.Name)).ToList();
                    }
                    catch { }
                }

                if (!FilterIncludeProcessed || !FilterIncludeOutdated || !FilterIncludeUnprocessed || !FilterIncludeFilteredOut || !FilterIncludeDeselected)
                {
                    ProcessingOptionsMovieCTF OptionsCTF = WarpRuntime.MainWindow.Options.GetProcessingMovieCTF();
                    ProcessingOptionsMovieMovement OptionsMovement = WarpRuntime.MainWindow.Options.GetProcessingMovieMovement();
                    ProcessingOptionsBoxNet OptionsBoxNet = WarpRuntime.MainWindow.Options.GetProcessingBoxNet();
                    ProcessingOptionsMovieExport OptionsExport = WarpRuntime.MainWindow.Options.GetProcessingMovieExport();

                    List<Tuple<Movie, ProcessingStatus>> WithStatus = new List<Tuple<Movie, ProcessingStatus>>(Items.Count);
                    foreach (var item in Result)
                        WithStatus.Add(new Tuple<Movie, ProcessingStatus>(item, GetMovieProcessingStatus(item, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, WarpRuntime.MainWindow.Options)));

                    Result = WithStatus.Where(t =>
                    {
                        return (t.Item2 == ProcessingStatus.Processed && FilterIncludeProcessed) ||
                               (t.Item2 == ProcessingStatus.Outdated && FilterIncludeOutdated) ||
                               (t.Item2 == ProcessingStatus.Unprocessed && FilterIncludeUnprocessed) ||
                               (t.Item2 == ProcessingStatus.FilteredOut && FilterIncludeFilteredOut) ||
                               (t.Item2 == ProcessingStatus.LeaveOut && FilterIncludeDeselected);
                    }).Select(t => t.Item1).ToList();
                }

                FilteredItems = Result;
            }
        }
    }

    [ValueConversion(typeof(bool), typeof(bool))]
    public class InverseBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            //if (targetType != typeof(bool))
            //    throw new InvalidOperationException("The target must be a boolean");

            return !(bool?)value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return !(bool?)value;
        }
    }
}
