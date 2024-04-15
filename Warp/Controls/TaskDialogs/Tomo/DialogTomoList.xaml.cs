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
using Warp.Tools;

namespace Warp.Controls.TaskDialogs.Tomo
{
    /// <summary>
    /// Interaction logic for DialogTomogramList.xaml
    /// </summary>
    public partial class DialogTomoList : UserControl
    {
        public TiltSeries[] Series;
        public string ExportPath;
        public OptionsWarp Options;
        public event Action Close;

        public DialogTomoList(TiltSeries[] series, OptionsWarp options)
        {
            InitializeComponent();

            Series = series;
            Options = options;

            DataContext = Options;
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonWrite_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.SaveFileDialog FileDialog = new System.Windows.Forms.SaveFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult Result = FileDialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                ExportPath = FileDialog.FileName;
            }
            else
            {
                return;
            }

            bool Relative = (bool)CheckRelative.IsChecked;
            bool Filter = (bool)CheckFilter.IsChecked;
            bool Manual = (bool)CheckManual.IsChecked;

            ProgressWrite.Visibility = Visibility.Visible;
            PanelButtons.Visibility = Visibility.Hidden;

            await Task.Run(() =>
            {
                List<TiltSeries> ValidSeries = Series.Where(v =>
                {
                    if (!Filter && v.UnselectFilter && v.UnselectManual == null)
                        return false;
                    if (!Manual && v.UnselectManual != null && (bool)v.UnselectManual)
                        return false;
                    return true;
                }).ToList();

                string PathPrefix = "";
                if (ValidSeries.Count > 0 && Relative)
                {
                    Uri UriStar = new Uri(ExportPath);
                    PathPrefix = UriStar.MakeRelativeUri(new Uri(ValidSeries[0].Path)).ToString();

                    PathPrefix = PathPrefix.Substring(0, PathPrefix.IndexOf(Helper.PathToNameWithExtension(PathPrefix)));
                }

                Dispatcher.Invoke(() => ProgressWrite.Maximum = ValidSeries.Count);

                bool IncludeCTF = ValidSeries.Any(v => v.OptionsCTF != null);

                Star TableOut = new Star(new[] { "rlnMicrographName" });
                if (IncludeCTF)
                {
                    TableOut.AddColumn("rlnCtfMaxResolution");
                }

                int r = 0;
                foreach (var series in ValidSeries)
                {
                    List<string> Row = new List<string> { PathPrefix + series.Name };

                    if (IncludeCTF)
                    {
                        if (series.OptionsCTF != null)
                            Row.Add(series.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture));
                        else
                            Row.Add("999");
                    }

                    TableOut.AddRow(Row.ToArray());

                    Dispatcher.Invoke(() => ProgressWrite.Value = ++r);
                }

                TableOut.Save(ExportPath);
            });

            Close?.Invoke();
        }
    }
}
