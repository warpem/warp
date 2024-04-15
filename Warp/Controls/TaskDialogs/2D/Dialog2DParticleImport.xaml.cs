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
using MahApps.Metro.Controls.Dialogs;
using Warp.Headers;
using Warp.Tools;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for Dialog2DParticleImport.xaml
    /// </summary>
    public partial class Dialog2DParticleImport : UserControl
    {
        public Movie[] Movies;
        public string ImportPath;
        public OptionsWarp Options;
        public event Action Close;

        public Dialog2DParticleImport(Movie[] movies, string importPath, OptionsWarp options)
        {
            InitializeComponent();

            Movies = movies;
            Options = options;
            ImportPath = importPath;

            DataContext = Options;

            TextSuffix.Text = Helper.PathToName(importPath);

            Star TableIn = new Star(importPath, "", null, 1);
            if (TableIn.RowCount > 0 && TableIn.HasColumn("rlnMagnification") && TableIn.HasColumn("rlnDetectorPixelSize"))
            {
                Options.Tasks.InputPixelSize = (decimal)(TableIn.GetRowValueFloat(0, "rlnDetectorPixelSize") / TableIn.GetRowValueFloat(0, "rlnMagnification")) * 1e4M;
                Options.Tasks.InputShiftPixelSize = Options.Tasks.InputPixelSize;
            }
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonWrite_OnClick(object sender, RoutedEventArgs e)
        {
            string Suffix = TextSuffix.Text;
            float ScaleFactorCoords = (float)(Options.Tasks.InputPixelSize / Options.Import.BinnedPixelSize);
            float ScaleFactorShifts = (float)(Options.Tasks.InputShiftPixelSize / Options.Import.BinnedPixelSize);

            ProgressWrite.Visibility = Visibility.Visible;
            PanelButtons.Visibility = Visibility.Hidden;

            this.IsEnabled = false;

            await Task.Run(async () =>
            {
                Star TableIn = new Star(ImportPath);

                #region Make sure everything is there

                if (TableIn.RowCount == 0 || !TableIn.HasColumn("rlnCoordinateX") || !TableIn.HasColumn("rlnCoordinateY") || !TableIn.HasColumn("rlnMicrographName"))
                {
                    await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                        "Please make sure the table contains data with rlnCoordinateX, rlnCoordinateY and rlnMicrographName columns.");
                    Close?.Invoke();
                    return;
                }

                if (Movies.Length == 0)
                {
                    Close?.Invoke();
                    return;
                }

                #endregion

                Directory.CreateDirectory(Movies[0].MatchingDir);

                Dispatcher.Invoke(() => ProgressWrite.Maximum = Movies.Length);

                int IndexMicName = TableIn.GetColumnID("rlnMicrographName");
                TableIn.ModifyAllValuesInColumn("rlnMicrographName", v => Helper.PathToName(v));

                int IndexCoordX = TableIn.GetColumnID("rlnCoordinateX");
                int IndexCoordY = TableIn.GetColumnID("rlnCoordinateY");

                #region Subtract origin from coordinates

                int IndexOriginX = TableIn.GetColumnID("rlnOriginX");
                if (IndexOriginX >= 0)
                {
                    TableIn.ModifyAllValuesInColumn("rlnCoordinateX", (v, r) => (float.Parse(v) * ScaleFactorCoords - float.Parse(TableIn.GetRowValue(r, IndexOriginX)) * ScaleFactorShifts).ToString("F4"));
                    TableIn.SetColumn("rlnOriginX", Helper.ArrayOfConstant("0.0", TableIn.RowCount));
                }
                else
                {
                    TableIn.ModifyAllValuesInColumn("rlnCoordinateX", v => (float.Parse(v) * ScaleFactorCoords).ToString("F4"));
                }
                int IndexOriginY = TableIn.GetColumnID("rlnOriginY");
                if (IndexOriginY >= 0)
                {
                    TableIn.ModifyAllValuesInColumn("rlnCoordinateY", (v, r) => (float.Parse(v) * ScaleFactorCoords - float.Parse(TableIn.GetRowValue(r, IndexOriginY)) * ScaleFactorShifts).ToString("F4"));
                    TableIn.SetColumn("rlnOriginY", Helper.ArrayOfConstant("0.0", TableIn.RowCount));
                }
                else
                {
                    TableIn.ModifyAllValuesInColumn("rlnCoordinateY", v => (float.Parse(v) * ScaleFactorCoords).ToString("F4"));
                }

                #endregion

                var GroupedRows = TableIn.GetAllRows().GroupBy(row => row[IndexMicName]).ToDictionary(g => g.Key, g => g.ToList());

                foreach (var movie in Movies)
                {
                    if (GroupedRows.ContainsKey(movie.RootName))
                    {
                        int3 Dims = MapHeader.ReadFromFile(movie.DataPath,
                                                           new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                           Options.Import.HeaderlessOffset,
                                                           ImageFormatsHelper.StringToType(Options.Import.HeaderlessType)).Dimensions;
                        float3 BinnedDims = new float3(Dims) * (float)Options.Import.PixelSize / (float)Options.Import.BinnedPixelSize;

                        List<string[]> Rows = GroupedRows[movie.RootName];
                        foreach (var row in Rows)
                        {
                            row[IndexMicName] = movie.Name;

                            float CoordX = float.Parse(row[IndexCoordX]);
                            if (Options.Tasks.InputFlipX)
                                CoordX = BinnedDims.X - CoordX;

                            float CoordY = float.Parse(row[IndexCoordY]);
                            if (Options.Tasks.InputFlipY)
                                CoordY = BinnedDims.Y - CoordY;

                            row[IndexCoordX] = CoordX.ToString("F4");
                            row[IndexCoordY] = CoordY.ToString("F4");
                        }

                        Star TableOut = new Star(TableIn.GetColumnNames());
                        foreach (var row in Rows)
                            TableOut.AddRow(row);

                        TableOut.Save(System.IO.Path.Combine(movie.MatchingDir, movie.RootName + "_" + Suffix + ".star"));

                        movie.UpdateParticleCount("_" + Suffix);
                    }

                    Dispatcher.Invoke(() => ProgressWrite.Value++);
                }
            });

            Close?.Invoke();
        }
    }
}
