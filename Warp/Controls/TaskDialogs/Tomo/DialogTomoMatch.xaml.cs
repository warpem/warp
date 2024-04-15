using System;
using System.Collections.Generic;
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
using Warp.Headers;
using Warp.Tools;

namespace Warp.Controls.TaskDialogs.Tomo
{
    /// <summary>
    /// Interaction logic for DialogTomoMatch.xaml
    /// </summary>
    public partial class DialogTomoMatch : UserControl
    {
        private int NParallel = 1;
        private int3[] GridSizes;
        private int[] GridProgress;
        private string[] GridNames;
        private ReconstructionMiniature[] GridControls;
        private TextBlock[] GridLabels;

        private TiltSeries[] Series;
        private string PathTemplate;
        private OptionsWarp Options;

        public event Action Close;

        bool IsCanceled = false;

        public DialogTomoMatch(TiltSeries[] series, string pathTemplate, OptionsWarp options)
        {
            InitializeComponent();

            Series = series;
            PathTemplate = pathTemplate;
            Options = options;

            DataContext = Options;

            TextTemplateName.Value = Helper.PathToNameWithExtension(pathTemplate);
            MapHeader Header = MapHeader.ReadFromFile(pathTemplate);
            if (Math.Abs(Header.PixelSize.X - 1f) > 1e-6f)
                Options.Tasks.TomoMatchTemplatePixel = (decimal)Header.PixelSize.X;
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

            NParallel = Math.Min(ValidSeries.Count, GPU.GetDeviceCount());
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
                #region Load template and make copies for all devices

                Image[] Template = new Image[NParallel];
                {
                    Template[0] = Image.FromFile(PathTemplate);

                    Template[0].PixelSize = (float)Options.Tasks.TomoMatchTemplatePixel;
                    for (int i = 0; i < Template.Length; i++)
                        Template[i] = Template[0].GetCopy();
                }

                #endregion

                Helper.ForEachGPU(ValidSeries, (item, gpuID) =>
                {
                    if (IsCanceled)
                        return true;    // This cancels the iterator

                    ProcessingOptionsTomoFullMatch SeriesOptions = Options.GetProcessingTomoFullMatch();
                    SeriesOptions.TemplateName = Helper.PathToName(PathTemplate);
                    SeriesOptions.NormalizeScores = true;

                    item.MatchFull(SeriesOptions, Template[gpuID], (size, value, name) =>
                    {
                        Dispatcher.Invoke(() =>
                        {
                            SetGridSize(gpuID, size);
                            SetGridProgress(gpuID, (int)(value + 0.5f));
                            SetGridName(gpuID, name);
                        });

                        return IsCanceled;
                    });

                    Dispatcher.Invoke(() => ProgressOverall.Value = ++Completed);

                    return false;   // No need to cancel GPU ForEach iterator
                }, 1, Helper.ArrayOfSequence(0, NParallel, 1).ToList());
            });

            Close?.Invoke();
        }

        private void ButtonAbort_OnClick(object sender, RoutedEventArgs e)
        {
            ButtonAbort.IsEnabled = false;
            ButtonAbort.Content = "CANCELING...";
            IsCanceled = true;
        }
    }
}
