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
using Warp.Headers;
using Warp.Tools;

namespace Warp.Controls
{
    public partial class Dialog2DList : UserControl
    {
        public Movie[] Movies;
        public string ExportPath;
        public OptionsWarp Options;
        public event Action Close;

        public Dialog2DList(Movie[] movies, OptionsWarp options)
        {
            InitializeComponent();

            Movies = movies;
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
                List<Movie> ValidMovies = Movies.Where(v =>
                {
                    if (!Filter && v.UnselectFilter && v.UnselectManual == null)
                        return false;
                    if (!Manual && v.UnselectManual != null && (bool)v.UnselectManual)
                        return false;
                    if (Options.Tasks.MicListMakePolishing && (v.OptionsMovement == null || v.GridMovementX.Dimensions.Z <= 1))
                        return false;
                    return true;
                }).ToList();

                string PathPrefix = "", PathPrefixAverage = "", GainPath = "";
                if (ValidMovies.Count > 0 && Relative)
                {
                    Uri UriStar = new Uri(ExportPath);

                    PathPrefix = Helper.PathToFolder(UriStar.MakeRelativeUri(new Uri(ValidMovies[0].Path)).ToString());
                    if (PathPrefix == ValidMovies[0].Name)
                        PathPrefix = "";
                    PathPrefixAverage = Helper.PathToFolder(UriStar.MakeRelativeUri(new Uri(ValidMovies[0].AveragePath)).ToString());

                    if (!string.IsNullOrEmpty(Options.Import.GainPath) && Options.Import.CorrectGain)
                    {
                        // Since RELION can't read DM4 or flip/transpose the gain, save a transformed version as MRC.

                        string TransformedGainPath = Helper.PathToFolder(Options.Import.GainPath) + "gain/transformed.mrc";
                        Directory.CreateDirectory(Helper.PathToFolder(TransformedGainPath));

                        Image TransformedGain = MainWindow.LoadAndPrepareGainReference();
                        TransformedGain.WriteMRC(TransformedGainPath, (float)Options.Import.PixelSize, true);
                        TransformedGain.Dispose();

                        GainPath = UriStar.MakeRelativeUri(new Uri(TransformedGainPath)).ToString();
                    }
                }

                Dispatcher.Invoke(() => ProgressWrite.Maximum = ValidMovies.Count);

                bool IncludeCTF = ValidMovies.Any(v => v.OptionsCTF != null);
                bool IncludePolishing = ValidMovies.Any(v => v.OptionsMovement != null) && Options.Tasks.MicListMakePolishing;

                Star TableOut = new Star(new[] { "rlnMicrographName" });
                if (IncludeCTF)
                {
                    TableOut.AddColumn("rlnMagnification");
                    TableOut.AddColumn("rlnDetectorPixelSize");
                    TableOut.AddColumn("rlnVoltage");
                    TableOut.AddColumn("rlnSphericalAberration");
                    TableOut.AddColumn("rlnAmplitudeContrast");
                    TableOut.AddColumn("rlnPhaseShift");
                    TableOut.AddColumn("rlnDefocusU");
                    TableOut.AddColumn("rlnDefocusV");
                    TableOut.AddColumn("rlnDefocusAngle");
                    TableOut.AddColumn("rlnCtfMaxResolution");
                    TableOut.AddColumn("rlnBeamTiltX");
                    TableOut.AddColumn("rlnBeamTiltY");
                }
                if (IncludePolishing)
                    TableOut.AddColumn("rlnMicrographMetadata");

                List<Action> TablesToWrite = new List<Action>();

                if (IncludePolishing && ValidMovies.Count > 0)
                    Directory.CreateDirectory(ValidMovies[0].MotionTrackDir);

                int r = 0;
                Parallel.ForEach(ValidMovies, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, movie =>
                {
                    List<string> Row = new List<string>() { PathPrefixAverage + movie.RootName + ".mrc" };

                    if (movie.OptionsCTF != null)
                        Row.AddRange(new[]
                        {
                            (1e4M / movie.OptionsCTF.BinnedPixelSizeMean).ToString("F1", CultureInfo.InvariantCulture),
                            "1.0",
                            movie.CTF.Voltage.ToString("F1", CultureInfo.InvariantCulture),
                            movie.CTF.Cs.ToString("F4", CultureInfo.InvariantCulture),
                            movie.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture),
                            (movie.CTF.PhaseShift * 180M).ToString("F1", CultureInfo.InvariantCulture),
                            ((movie.CTF.Defocus + movie.CTF.DefocusDelta / 2) * 1e4M).ToString("F1", CultureInfo.InvariantCulture),
                            ((movie.CTF.Defocus - movie.CTF.DefocusDelta / 2) * 1e4M).ToString("F1", CultureInfo.InvariantCulture),
                            movie.CTF.DefocusAngle.ToString("F1", CultureInfo.InvariantCulture),
                            movie.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture),
                            movie.CTF.BeamTilt.X.ToString("F2", CultureInfo.InvariantCulture),
                            movie.CTF.BeamTilt.Y.ToString("F2", CultureInfo.InvariantCulture)
                         });
                    else
                        Row.AddRange(Helper.ArrayOfFunction(i => "0.0", 12));

                    if (IncludePolishing)
                        Row.Add(System.IO.Path.Combine(movie.MotionTrackDir, movie.RootName + ".star"));

                    lock (TableOut)
                        TableOut.AddRow(Row.ToArray());

                    if (IncludePolishing)
                    {
                        MapHeader Header = MapHeader.ReadFromFile(movie.DataPath);

                        List<string> HeaderNames = new List<string>()
                         {
                            "rlnImageSizeX",
                            "rlnImageSizeY",
                            "rlnImageSizeZ",
                            "rlnMicrographMovieName",
                            "rlnMicrographBinning",
                            "rlnMicrographOriginalPixelSize",
                            "rlnMicrographDoseRate",
                            "rlnMicrographPreExposure",
                            "rlnVoltage",
                            "rlnMicrographStartFrame",
                            "rlnMotionModelVersion"
                         };

                        List<string> HeaderValues = new List<string>()
                         {
                            Header.Dimensions.X.ToString(),
                            Header.Dimensions.Y.ToString(),
                            Header.Dimensions.Z.ToString(),
                            PathPrefix + movie.Name,
                            Math.Pow(2.0, (double)Options.Import.BinTimes).ToString("F5"),
                            Options.Import.PixelSize.ToString("F5"),
                            Options.Import.DosePerAngstromFrame.ToString("F5"),
                            "0",
                            Options.CTF.Voltage.ToString(),
                            "1",
                            "0"
                         };

                        if (!string.IsNullOrEmpty(GainPath))
                        {
                            HeaderNames.Add("rlnMicrographGainName");
                            HeaderValues.Add(GainPath);
                        }

                        StarParameters ParamsTable = new StarParameters(HeaderNames.ToArray(), HeaderValues.ToArray());

                        float2[] MotionTrack = movie.GetMotionTrack(new float2(0.5f, 0.5f), 1).Select(v => v / (float)Options.Import.PixelSize).ToArray();
                        Star TrackTable = new Star(new[]
                        {
                            Helper.ArrayOfSequence(1, MotionTrack.Length + 1, 1).Select(v => v.ToString()).ToArray(),
                            MotionTrack.Select(v => (-v.X).ToString("F5")).ToArray(),
                            MotionTrack.Select(v => (-v.Y).ToString("F5")).ToArray()
                         },
                        new[]
                        {
                            "rlnMicrographFrameNumber",
                            "rlnMicrographShiftX",
                            "rlnMicrographShiftY"
                        });

                        Star.SaveMultitable(System.IO.Path.Combine(movie.MotionTrackDir, movie.RootName + ".star"), new Dictionary<string, Star>()
                         {
                            { "general", ParamsTable },
                            { "global_shift", TrackTable },
                         });
                    }

                    lock (this)
                    {
                        r++;
                        if (r % 10 == 0 || r == ValidMovies.Count)
                            Dispatcher.Invoke(() => ProgressWrite.Value = r);
                    }
                });

                TableOut.Save(ExportPath);
            });

            Close?.Invoke();
        }
    }
}
