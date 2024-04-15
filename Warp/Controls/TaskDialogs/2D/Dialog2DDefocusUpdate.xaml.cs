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

namespace Warp.Controls
{
    public partial class Dialog2DDefocusUpdate : UserControl
    {
        public Movie[] Movies;
        public string ImportPath, ExportPath;
        public OptionsWarp Options;
        public event Action Close;

        public Dialog2DDefocusUpdate(Movie[] movies, string importPath, OptionsWarp options)
        {
            InitializeComponent();
            DataContext = this;
            
            Movies = movies;
            ImportPath = importPath;
            Options = options;

            if (movies.Length > 0)
                Options.Tasks.InputPixelSize = movies.First(m => m.OptionsCTF != null).OptionsCTF.BinnedPixelSizeMean;

            DataContext = Options;
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonWrite_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.SaveFileDialog SaveDialog = new System.Windows.Forms.SaveFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultSave = SaveDialog.ShowDialog();

            if (ResultSave.ToString() == "OK")
            {
                ExportPath = SaveDialog.FileName;
            }
            else
            {
                return;
            }

            bool Skip = (bool)RadioSkip.IsChecked;
            bool Delete = (bool)RadioDelete.IsChecked;

            float AngPix = (float)Options.Tasks.InputPixelSize;

            bool Filter = (bool)CheckFilter.IsChecked;
            bool Manual = (bool)CheckManual.IsChecked;

            ProgressWrite.Visibility = Visibility.Visible;
            PanelButtons.Visibility = Visibility.Hidden;

            Dispatcher.Invoke(() => ProgressWrite.IsIndeterminate = true);

            await Task.Run(() =>
            {
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
                List<string> ValidMovieNames = ValidMovies.Select(m => m.RootName).ToList();

                #endregion

                #region Read table and intersect its micrograph set with valid movies

                Star TableIn = new Star(ImportPath);
                if (!TableIn.HasColumn("rlnMicrographName"))
                    throw new Exception("Couldn't find rlnMicrographName column.");
                if (!TableIn.HasColumn("rlnCoordinateX"))
                    throw new Exception("Couldn't find rlnCoordinateX column.");
                if (!TableIn.HasColumn("rlnCoordinateY"))
                    throw new Exception("Couldn't find rlnCoordinateY column.");

                Dictionary<string, List<int>> Groups = new Dictionary<string, List<int>>();
                {
                    string[] ColumnMicNames = TableIn.GetColumn("rlnMicrographName");
                    for (int r = 0; r < ColumnMicNames.Length; r++)
                    {
                        if (!Groups.ContainsKey(ColumnMicNames[r]))
                            Groups.Add(ColumnMicNames[r], new List<int>());
                        Groups[ColumnMicNames[r]].Add(r);
                    }
                    Groups = Groups.ToDictionary(group => Helper.PathToName(group.Key), group => group.Value);

                    Groups = Groups.Where(@group => ValidMovieNames.Contains(@group.Key)).ToDictionary(@group => @group.Key, @group => @group.Value);
                }

                bool[] RowsIncluded = new bool[TableIn.RowCount];
                foreach (var @group in Groups)
                    foreach (var r in group.Value)
                        RowsIncluded[r] = true;
                List<int> RowsNotIncluded = new List<int>();
                for (int r = 0; r < RowsIncluded.Length; r++)
                    if (!RowsIncluded[r])
                        RowsNotIncluded.Add(r);

                #endregion

                #region Make sure all columns are there
                
                if (!TableIn.HasColumn("rlnVoltage"))
                    TableIn.AddColumn("rlnVoltage", "300.0");

                if (!TableIn.HasColumn("rlnSphericalAberration"))
                    TableIn.AddColumn("rlnSphericalAberration", "2.7");

                if (!TableIn.HasColumn("rlnAmplitudeContrast"))
                    TableIn.AddColumn("rlnAmplitudeContrast", "0.07");

                if (!TableIn.HasColumn("rlnPhaseShift"))
                    TableIn.AddColumn("rlnPhaseShift", "0.0");

                if (!TableIn.HasColumn("rlnDefocusU"))
                    TableIn.AddColumn("rlnDefocusU", "0.0");

                if (!TableIn.HasColumn("rlnDefocusV"))
                    TableIn.AddColumn("rlnDefocusV", "0.0");

                if (!TableIn.HasColumn("rlnDefocusAngle"))
                    TableIn.AddColumn("rlnDefocusAngle", "0.0");

                if (!TableIn.HasColumn("rlnCtfMaxResolution"))
                    TableIn.AddColumn("rlnCtfMaxResolution", "999.0");

                #endregion

                Dispatcher.Invoke(() => ProgressWrite.Maximum = ValidMovies.Count);

                {
                    int i = 0;
                    float[] PosX = TableIn.GetColumn("rlnCoordinateX").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    float[] PosY = TableIn.GetColumn("rlnCoordinateY").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    foreach (var movie in ValidMovies)
                    {
                        if (Groups.ContainsKey(movie.RootName))
                        {
                            List<int> GroupRows = Groups[movie.RootName];

                            float Astigmatism = (float)movie.CTF.DefocusDelta / 2;
                            float PhaseShift = movie.OptionsCTF.DoPhase ? movie.GridCTFPhase.GetInterpolated(new float3(0.5f)) * 180 : 0;

                            foreach (var r in GroupRows)
                            {
                                float3 Position = new float3(PosX[r] * AngPix / movie.OptionsCTF.Dimensions.X,
                                                             PosY[r] * AngPix / movie.OptionsCTF.Dimensions.Y,
                                                             0.5f);
                                float LocalDefocus = movie.GridCTFDefocus.GetInterpolated(Position);

                                TableIn.SetRowValue(r, "rlnDefocusU", ((LocalDefocus + Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture));
                                TableIn.SetRowValue(r, "rlnDefocusV", ((LocalDefocus - Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture));
                                TableIn.SetRowValue(r, "rlnDefocusAngle", movie.CTF.DefocusAngle.ToString("F1", CultureInfo.InvariantCulture));

                                TableIn.SetRowValue(r, "rlnVoltage", movie.CTF.Voltage.ToString("F1", CultureInfo.InvariantCulture));
                                TableIn.SetRowValue(r, "rlnSphericalAberration", movie.CTF.Cs.ToString("F4", CultureInfo.InvariantCulture));
                                TableIn.SetRowValue(r, "rlnAmplitudeContrast", movie.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture));
                                TableIn.SetRowValue(r, "rlnPhaseShift", PhaseShift.ToString("F1", CultureInfo.InvariantCulture));
                                TableIn.SetRowValue(r, "rlnCtfMaxResolution", movie.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture));
                            }
                        }

                        Dispatcher.Invoke(() =>
                        {

                            ProgressWrite.IsIndeterminate = false;
                            ProgressWrite.Value = ++i;
                        });
                    }
                }

                Dispatcher.Invoke(() => ProgressWrite.IsIndeterminate = true);

                if (Delete)
                    TableIn.RemoveRows(RowsNotIncluded.ToArray());

                TableIn.Save(ExportPath);
            });

            Close?.Invoke();
        }
    }
}
