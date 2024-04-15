using System;
using System.Collections.Generic;
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
using Warp.Tools;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for DialogTomoReconstruction.xaml
    /// </summary>
    public partial class DialogTomoReconstruction : UserControl
    {
        private int NParallel = 1;
        private int3[] GridSizes;
        private int[] GridProgress;
        private string[] GridNames;
        private ReconstructionMiniature[] GridControls;
        private TextBlock[] GridLabels;

        private TiltSeries[] Series;
        private OptionsWarp Options;

        public event Action Close;

        bool IsCanceled = false;

        public DialogTomoReconstruction(TiltSeries[] series, OptionsWarp options)
        {
            InitializeComponent();

            Series = series;
            Options = options;

            DataContext = Options;

            bool CanUseFrames = false;
            if (series.Length > 0)
            {
                Movie FirstMovie = new Movie(Path.Combine(series[0].DataOrProcessingDirectoryName, series[0].TiltMoviePaths[0]));
                CanUseFrames = File.Exists(FirstMovie.AverageEvenPath) && File.Exists(FirstMovie.AverageOddPath);
            }

            if (!CanUseFrames)
            {
                Options.Tasks.TomoFullReconstructDenoisingFrames = false;
                Options.Tasks.TomoFullReconstructDenoisingTilts = true;
                RadioDenoisingFrames.IsEnabled = false;
            }
        }

        public void SetGridSize(int n, int3 size)
        {
            if (n >= NParallel)
                return;

            GridSizes[n] = size;
            GridControls[n].GridSize = size;
        }

        public void SetGridProgress(int n, int value)
        {
            if (n >= NParallel)
                return;

            GridProgress[n] = value;
            GridControls[n].Value = value;
        }

        public void SetGridName(int n, string name)
        {
            if (n >= NParallel)
                return;

            GridNames[n] = name;
            GridLabels[n].Text = name;
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonReconstruct_OnClick(object sender, RoutedEventArgs e)
        {
            bool Filter = (bool)CheckFilter.IsChecked;
            bool Manual = (bool)CheckManual.IsChecked;

            #region Get all movies that can potentially be used

            List<TiltSeries> ValidSeries = Series.Where(v =>
            {
                if (!Filter && v.UnselectFilter && v.UnselectManual == null)
                    return false;
                if (!Manual && v.UnselectManual != null && (bool)v.UnselectManual)
                    return false;
                if (v.OptionsCTF == null)
                    return false;
                return true;
            }).ToList();

            if (ValidSeries.Count == 0)
            {
                Close?.Invoke();
                return;
            }

            #endregion

            #region Set up progress displays

            NParallel = Math.Min(ValidSeries.Count, WarpRuntime.MainWindow.GetDeviceList().Count());
            GridSizes = Helper.ArrayOfConstant(new int3(1), NParallel);
            GridProgress = new int[NParallel];
            GridNames = new string[NParallel];

            GridControls = Helper.ArrayOfFunction(i => new ReconstructionMiniature { Width = 256 }, NParallel);
            GridLabels = Helper.ArrayOfFunction(i => new TextBlock { FontSize = 18, Margin = new Thickness(0, 15, 0, 0), HorizontalAlignment = HorizontalAlignment.Center }, NParallel);

            for (int n = 0; n < NParallel; n++)
            {
                StackPanel SP = new StackPanel { Orientation = Orientation.Vertical, Margin = new Thickness(15, 0, 15, 0) };
                SP.Children.Add(GridControls[n]);
                SP.Children.Add(GridLabels[n]);

                PanelGrids.Children.Add(SP);
            }

            #endregion

            PanelConfiguration.Visibility = Visibility.Collapsed;
            PanelProgress.Visibility = Visibility.Visible;

            int Completed = 0;
            ProgressOverall.Maximum = ValidSeries.Count;

            await Task.Run(() =>
            {
                Helper.ForEachGPU(ValidSeries, (item, gpuID) =>
                {
                    if (IsCanceled)
                        return true;    // This cancels the iterator

                    ProcessingOptionsTomoFullReconstruction SeriesOptions = Options.GetProcessingTomoFullReconstruction();

                    item.ReconstructFull(SeriesOptions, (size, value, name) =>
                    {
                        Dispatcher.Invoke(() =>
                        {
                            SetGridSize(gpuID, size);
                            SetGridProgress(gpuID, value);
                            SetGridName(gpuID, name);
                        });

                        return IsCanceled;
                    });

                    Dispatcher.Invoke(() => ProgressOverall.Value = ++Completed);

                    return false;   // No need to cancel GPU ForEach iterator
                }, 1, WarpRuntime.MainWindow.GetDeviceList());
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
