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
using System.Windows.Shapes;
using Warp.Controls.ProgressIndicators;
using Warp.Headers;
using Warp.Tools;

namespace Warp.Controls.TaskDialogs.TwoD
{
    /// <summary>
    /// Interaction logic for Dialog2DMatch.xaml
    /// </summary>
    public partial class Dialog2DMatch : UserControl
    {
        private int NParallel = 1;
        private int3[] GridSizes;
        private int[] GridProgress;
        private string[] GridNames;
        private FlatGridMiniature[] GridControls;
        private TextBlock[] GridLabels;

        private Movie[] Movies;
        private string PathTemplate;
        private OptionsWarp Options;

        public event Action Close;

        bool IsCanceled = false;

        public Dialog2DMatch(Movie[] movies, string pathTemplate, OptionsWarp options)
        {
            InitializeComponent();

            Movies = movies;
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

        private async void ButtonMatch_OnClick(object sender, RoutedEventArgs e)
        {
            bool Filter = (bool)CheckFilter.IsChecked;
            bool Manual = (bool)CheckManual.IsChecked;

            #region Get all movies that can potentially be used

            List<Movie> ValidMovies = Movies.Where(v =>
            {
                if (!Filter && v.UnselectFilter && v.UnselectManual == null)
                    return false;
                if (!Manual && v.UnselectManual != null && (bool)v.UnselectManual)
                    return false;
                if (v.OptionsCTF == null)
                    return false;
                return true;
            }).ToList();

            if (ValidMovies.Count == 0)
            {
                Close?.Invoke();
                return;
            }

            #endregion

            #region Set up progress displays

            NParallel = Math.Min(ValidMovies.Count, GPU.GetDeviceCount());
            GridSizes = Helper.ArrayOfConstant(new int3(1), NParallel);
            GridProgress = new int[NParallel];
            GridNames = new string[NParallel];

            GridControls = Helper.ArrayOfFunction(i => new FlatGridMiniature() { Width = 256 }, NParallel);
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
            ProgressOverall.Maximum = ValidMovies.Count;

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

                #region Load gain reference if needed

                Image[] ImageGain = new Image[NParallel];
                DefectModel[] DefectMap = new DefectModel[NParallel];
                if (!string.IsNullOrEmpty(Options.Import.GainPath) && Options.Import.CorrectGain && File.Exists(Options.Import.GainPath))
                    for (int d = 0; d < NParallel; d++)
                        ImageGain[d] = MainWindow.LoadAndPrepareGainReference();
                if (!string.IsNullOrEmpty(Options.Import.DefectsPath) && Options.Import.CorrectDefects && File.Exists(Options.Import.DefectsPath))
                    for (int d = 0; d < NParallel; d++)
                        DefectMap[d] = MainWindow.LoadAndPrepareDefectMap();

                #endregion

                object SyncDummy = new object();

                Helper.ForEachGPU(ValidMovies, (item, gpuID) =>
                {
                    if (IsCanceled)
                        return true;    // This cancels the iterator

                    Dispatcher.Invoke(() =>
                    {
                        SetGridSize(gpuID, new int3(1, 1, 1));
                        SetGridProgress(gpuID, 0);
                        SetGridName(gpuID, "Loading data...");
                    });

                    ProcessingOptionsFullMatch MatchOptions = Options.GetProcessingFullMatch();
                    MatchOptions.TemplateName = Helper.PathToName(PathTemplate);

                    #region Load and prepare original movie

                    Image OriginalStack = null;
                    decimal ScaleFactor = 1M / MatchOptions.DownsampleFactor;

                    MapHeader OriginalHeader = MapHeader.ReadFromFile(item.DataPath);

                    MainWindow.LoadAndPrepareHeaderAndMap(item.DataPath, ImageGain[gpuID], DefectMap[gpuID], ScaleFactor, out OriginalHeader, out OriginalStack);

                    if (IsCanceled)
                    {
                        OriginalStack?.Dispose();

                        return true;
                    }

                    #endregion

                    item.MatchFull(OriginalStack, MatchOptions, Template[gpuID], (size, value, name) =>
                    {
                        Dispatcher.Invoke(() =>
                        {
                            SetGridSize(gpuID, size);
                            SetGridProgress(gpuID, value);
                            SetGridName(gpuID, name);
                        });

                        return IsCanceled;
                    });

                    OriginalStack?.Dispose();

                    Dispatcher.Invoke(() => ProgressOverall.Value = ++Completed);

                    return false;   // No need to cancel GPU ForEach iterator
                }, 1);
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
