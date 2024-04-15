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
using MahApps.Metro.Controls.Dialogs;
using Warp.Headers;
using Warp.Tools;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for BoxNetExport.xaml
    /// </summary>
    public partial class BoxNetExport : UserControl
    {
        public string ModelName;
        public OptionsWarp Options;
        public event Action Close;

        private string SuffixPositive, SuffixFalsePositive, SuffixUncertain;
        private string FolderPositive, FolderFalsePositive, FolderUncertain;

        public string NewName;

        public BoxNetExport(OptionsWarp options)
        {
            InitializeComponent();

            Options = options;
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonSuffixPositive_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                Movie[] Movies = WarpRuntime.MainWindow.FileDiscoverer.GetImmutableFiles();

                bool FoundMatchingSuffix = false;
                string StarName = Helper.PathToName(OpenDialog.FileName);
                foreach (var item in Movies)
                {
                    if (StarName.Contains(item.RootName))
                    {
                        FoundMatchingSuffix = true;
                        SuffixPositive = StarName.Substring(item.RootName.Length);
                        FolderPositive = Helper.PathToFolder(OpenDialog.FileName);
                        ButtonSuffixPositiveText.Text = "*" + SuffixPositive;
                        ButtonSuffixPositiveText.ToolTip = FolderPositive + "*" + SuffixPositive;
                        break;
                    }
                }

                if (!FoundMatchingSuffix)
                {
                    await WarpRuntime.MainWindow.ShowMessageAsync("Oopsie", "STAR file could not be matched to any of the movies to determine the suffix.");
                    return;
                }
            }
        }

        private async void ButtonSuffixFalsePositive_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                Movie[] Movies = WarpRuntime.MainWindow.FileDiscoverer.GetImmutableFiles();

                bool FoundMatchingSuffix = false;
                string StarName = Helper.PathToName(OpenDialog.FileName);
                foreach (var item in Movies)
                {
                    if (StarName.Contains(item.RootName))
                    {
                        FoundMatchingSuffix = true;
                        SuffixFalsePositive = StarName.Substring(item.RootName.Length);
                        FolderFalsePositive = Helper.PathToFolder(OpenDialog.FileName);
                        ButtonSuffixFalsePositiveText.Text = "*" + SuffixFalsePositive;
                        ButtonSuffixFalsePositiveText.ToolTip = FolderFalsePositive + "*" + SuffixFalsePositive;
                        break;
                    }
                }

                if (!FoundMatchingSuffix)
                {
                    await WarpRuntime.MainWindow.ShowMessageAsync("Oopsie", "STAR file could not be matched to any of the movies to determine the suffix.");
                    return;
                }
            }
        }

        private async void ButtonSuffixUncertain_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                Movie[] Movies = WarpRuntime.MainWindow.FileDiscoverer.GetImmutableFiles();

                bool FoundMatchingSuffix = false;
                string StarName = Helper.PathToName(OpenDialog.FileName);
                foreach (var item in Movies)
                {
                    if (StarName.Contains(item.RootName))
                    {
                        FoundMatchingSuffix = true;
                        SuffixUncertain = StarName.Substring(item.RootName.Length);
                        FolderUncertain = Helper.PathToFolder(OpenDialog.FileName);
                        ButtonSuffixUncertainText.Text = "*" + SuffixUncertain;
                        ButtonSuffixUncertainText.ToolTip = FolderUncertain + "*" + SuffixUncertain;
                        break;
                    }
                }

                if (!FoundMatchingSuffix)
                {
                    await WarpRuntime.MainWindow.ShowMessageAsync("Oopsie", "STAR file could not be matched to any of the movies to determine the suffix.");
                    return;
                }
            }
        }

        private async void ButtonExport_OnClick(object sender, RoutedEventArgs e)
        {
            float Diameter = (float)SliderDiameter.Value;

            string ErrorString = "";

            //if (string.IsNullOrEmpty(NewName))
            //    ErrorString += "No name specified.\n";

            if (string.IsNullOrEmpty(SuffixPositive))
                ErrorString += "No positive examples selected.\n";

            if (!string.IsNullOrEmpty(ErrorString))
            {
                await WarpRuntime.MainWindow.ShowMessageAsync("Oopsie", ErrorString);
                return;
            }

            bool IsNegative = (bool)CheckNegative.IsChecked;

            Movie[] Movies = WarpRuntime.MainWindow.FileDiscoverer.GetImmutableFiles();
            List<Movie> ValidMovies = new List<Movie>();

            foreach (var movie in Movies)
            {
                if (movie.UnselectManual != null && (bool)movie.UnselectManual)
                    continue;

                if (!File.Exists(movie.AveragePath))
                    continue;

                bool HasExamples = File.Exists(FolderPositive + movie.RootName + SuffixPositive + ".star");
                if (!HasExamples)
                    continue;

                ValidMovies.Add(movie);
            }

            if (ValidMovies.Count == 0)
            {
                await WarpRuntime.MainWindow.ShowMessageAsync("Oopsie", "No movie averages could be found to create training examples. Please process the movies first to create the averages.");
                return;
            }

            string SavePath = "";

            System.Windows.Forms.SaveFileDialog SaveDialog = new System.Windows.Forms.SaveFileDialog
            {
                Filter = "TIFF Files|*.tif"
            };
            System.Windows.Forms.DialogResult ResultSave = SaveDialog.ShowDialog();

            if (ResultSave.ToString() == "OK")
            {
                SavePath = SaveDialog.FileName;
            }
            else
            {
                return;
            }

            var ProgressDialog = await WarpRuntime.MainWindow.ShowProgressAsync("Preparing examples...", "");

            await Task.Run(async () =>
            {
                ProgressDialog.Maximum = ValidMovies.Count;

                List<Image> AllAveragesBN = new List<Image>();
                List<Image> AllLabelsBN = new List<Image>();
                List<Image> AllCertainBN = new List<Image>();

                int MoviesDone = 0;
                foreach (var movie in ValidMovies)
                {
                    MapHeader Header = MapHeader.ReadFromFile(movie.AveragePath);
                    float PixelSize = Header.PixelSize.X;

                    #region Load positions, and possibly move on to next movie

                    List<float2> PosPositive = new List<float2>();
                    List<float2> PosFalse = new List<float2>();
                    List<float2> PosUncertain = new List<float2>();

                    if (File.Exists(FolderPositive + movie.RootName + SuffixPositive + ".star"))
                        PosPositive.AddRange(Star.LoadFloat2(FolderPositive + movie.RootName + SuffixPositive + ".star",
                                                             "rlnCoordinateX",
                                                             "rlnCoordinateY").Select(v => v * PixelSize / BoxNetTorch.PixelSize));
                    if (PosPositive.Count == 0)
                        continue;

                    if (File.Exists(FolderFalsePositive + movie.RootName + SuffixFalsePositive + ".star"))
                        PosFalse.AddRange(Star.LoadFloat2(FolderFalsePositive + movie.RootName + SuffixFalsePositive + ".star",
                                                              "rlnCoordinateX",
                                                              "rlnCoordinateY").Select(v => v * PixelSize / BoxNetTorch.PixelSize));

                    if (File.Exists(FolderUncertain + movie.RootName + SuffixUncertain + ".star"))
                        PosUncertain.AddRange(Star.LoadFloat2(FolderUncertain + movie.RootName + SuffixUncertain + ".star",
                                                              "rlnCoordinateX",
                                                              "rlnCoordinateY").Select(v => v * PixelSize / BoxNetTorch.PixelSize));
                    
                    #endregion

                    Image Average = Image.FromFile(movie.AveragePath);
                    int2 Dims = new int2(Average.Dims);

                    Image Mask = null;
                    if (File.Exists(movie.MaskPath))
                        Mask = Image.FromFile(movie.MaskPath);

                    float RadiusParticle = Math.Max(1, Diameter / 2 / BoxNetTorch.PixelSize);
                    float RadiusPeak = Math.Max(1.5f, Diameter / 2 / BoxNetTorch.PixelSize / 4);
                    float RadiusFalse = Math.Max(1, Diameter / 2 / BoxNetTorch.PixelSize);
                    float RadiusUncertain = Math.Max(1, Diameter / 2 / BoxNetTorch.PixelSize);

                    #region Rescale everything and allocate memory

                    int2 DimsBN = new int2(new float2(Dims) * PixelSize / BoxNetTorch.PixelSize + 0.5f) / 2 * 2;
                    Image AverageBN = Average.AsScaled(DimsBN);
                    Average.Dispose();

                    if (IsNegative)
                        AverageBN.Multiply(-1f);

                    GPU.Normalize(AverageBN.GetDevice(Intent.Read),
                                  AverageBN.GetDevice(Intent.Write),
                                  (uint)AverageBN.ElementsSliceReal,
                                  1);

                    Image MaskBN = null;
                    if (Mask != null)
                    {
                        MaskBN = Mask.AsScaled(DimsBN);
                        Mask.Dispose();
                    }

                    Image LabelsBN = new Image(new int3(DimsBN));
                    Image CertainBN = new Image(new int3(DimsBN));
                    CertainBN.Fill(1f);

                    #endregion

                    #region Paint all positive and uncertain peaks

                    for (int i = 0; i < 3; i++)
                    {
                        var positions = (new[] { PosPositive, PosFalse, PosUncertain })[i];
                        float R = (new[] { RadiusPeak, RadiusFalse, RadiusUncertain })[i];
                        float R2 = R * R;
                        float Label = (new[] { 1, 4, 0 })[i];
                        float[] ImageData = (new[] { LabelsBN.GetHost(Intent.ReadWrite)[0], CertainBN.GetHost(Intent.ReadWrite)[0], CertainBN.GetHost(Intent.ReadWrite)[0] })[i];

                        foreach (var pos in positions)
                        {
                            int2 Min = new int2(Math.Max(0, (int)(pos.X - R)), Math.Max(0, (int)(pos.Y - R)));
                            int2 Max = new int2(Math.Min(DimsBN.X - 1, (int)(pos.X + R)), Math.Min(DimsBN.Y - 1, (int)(pos.Y + R)));

                            for (int y = Min.Y; y <= Max.Y; y++)
                            {
                                float yy = y - pos.Y;
                                yy *= yy;
                                for (int x = Min.X; x <= Max.X; x++)
                                {
                                    float xx = x - pos.X;
                                    xx *= xx;

                                    float r2 = xx + yy;
                                    if (r2 <= R2)
                                        ImageData[y * DimsBN.X + x] = Label;
                                }
                            }
                        }
                    }

                    #endregion

                    #region Add junk mask if there is one

                    if (MaskBN != null)
                    {
                        float[] LabelsBNData = LabelsBN.GetHost(Intent.ReadWrite)[0];
                        float[] MaskBNData = MaskBN.GetHost(Intent.Read)[0];
                        for (int i = 0; i < LabelsBNData.Length; i++)
                            if (MaskBNData[i] > 0.5f)
                                LabelsBNData[i] = 2;
                    }

                    #endregion

                    #region Clean up

                    MaskBN?.Dispose();

                    AllAveragesBN.Add(AverageBN);
                    AverageBN.FreeDevice();

                    AllLabelsBN.Add(LabelsBN);
                    LabelsBN.FreeDevice();

                    AllCertainBN.Add(CertainBN);
                    CertainBN.FreeDevice();

                    #endregion

                    ProgressDialog.SetProgress(++MoviesDone);
                }

                #region Figure out smallest dimensions that contain everything

                int2 DimsCommon = new int2(1);
                foreach (var image in AllAveragesBN)
                {
                    DimsCommon.X = Math.Max(DimsCommon.X, image.Dims.X);
                    DimsCommon.Y = Math.Max(DimsCommon.Y, image.Dims.Y);
                }

                #endregion

                #region Put everything in one stack and save

                Image Everything = new Image(new int3(DimsCommon.X, DimsCommon.Y, AllAveragesBN.Count * 3));
                float[][] EverythingData = Everything.GetHost(Intent.ReadWrite);

                for (int i = 0; i < AllAveragesBN.Count; i++)
                {
                    if (AllAveragesBN[i].Dims == new int3(DimsCommon)) // No padding needed
                    {
                        EverythingData[i * 3 + 0] = AllAveragesBN[i].GetHost(Intent.Read)[0];
                        EverythingData[i * 3 + 1] = AllLabelsBN[i].GetHost(Intent.Read)[0];
                        EverythingData[i * 3 + 2] = AllCertainBN[i].GetHost(Intent.Read)[0];
                    }
                    else // Padding needed
                    {
                        {
                            Image Padded = AllAveragesBN[i].AsPadded(DimsCommon);
                            AllAveragesBN[i].Dispose();
                            EverythingData[i * 3 + 0] = Padded.GetHost(Intent.Read)[0];
                            Padded.FreeDevice();
                        }
                        {
                            Image Padded = AllLabelsBN[i].AsPadded(DimsCommon);
                            AllLabelsBN[i].Dispose();
                            EverythingData[i * 3 + 1] = Padded.GetHost(Intent.Read)[0];
                            Padded.FreeDevice();
                        }
                        {
                            Image Padded = AllCertainBN[i].AsPadded(DimsCommon);
                            AllCertainBN[i].Dispose();
                            EverythingData[i * 3 + 2] = Padded.GetHost(Intent.Read)[0];
                            Padded.FreeDevice();
                        }
                    }
                }

                Everything.WriteTIFF(SavePath, BoxNetTorch.PixelSize, typeof(float));

                #endregion

                await ProgressDialog.CloseAsync();
            });

            Close?.Invoke();
        }
    }
}
