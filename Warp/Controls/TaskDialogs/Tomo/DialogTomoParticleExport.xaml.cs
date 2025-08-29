using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
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
using Warp.Workers;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for DialogParticleExport3D.xaml
    /// </summary>
    public partial class DialogTomoParticleExport : UserControl
    {
        public TiltSeries[] Series;
        public string ImportPath, ExportPath;
        public OptionsWarp Options;
        public event Action Close;

        string InputSuffix = "";
        string InputFolder = "";

        List<UIElement> DisableWhileProcessing;

        bool IsCanceled = false;

        List<float> Timings = new List<float>();

        public DialogTomoParticleExport(TiltSeries[] series, string importPath, OptionsWarp options)
        {
            InitializeComponent();

            Options = options;

            if (series.Any(m => m.OptionsCTF != null))
            {
                Options.Tasks.InputPixelSize = series.First(m => m.OptionsCTF != null).OptionsCTF.BinnedPixelSizeMean;
            }
            else
            {
                Options.Tasks.Export2DPreflip = false;
                CheckPreflip.IsEnabled = false;
            }

            Series = series;
            ImportPath = importPath;

            DataContext = Options;

            DisableWhileProcessing = new List<UIElement>
            {
                SliderImportAngPix,
                SliderBoxSize,
                SliderParticleDiameter,
                CheckInvert,
                CheckNormalize,
                CheckPreflip,
                CheckRelative,
                CheckFilter,
                CheckManual
            };

            #region Check if using different input files for each item makes sense

            bool FoundMatchingPrefix = false;
            string ImportName = Helper.PathToNameWithExtension(importPath);
            foreach (var item in Series)
            {
                if (ImportName.Contains(item.RootName + "_"))
                {
                    FoundMatchingPrefix = true;
                    InputSuffix = ImportName.Substring(item.RootName.Length);
                    break;
                }
            }

            if (FoundMatchingPrefix)
            {
                CheckOneInputPerItem.IsEnabled = true;
                CheckOneInputPerItem.ToolTip = $"{InputSuffix} will be used as suffix.";
                Options.Tasks.InputOnePerItem = true;

                FileInfo Info = new FileInfo(importPath);
                InputFolder = Info.DirectoryName;
                if (InputFolder.Last() != '/' && InputFolder.Last() != '\\')
                    InputFolder += "/";
            }
            else
            {
                CheckOneInputPerItem.IsEnabled = false;
                CheckOneInputPerItem.ToolTip = "No matching suffix found.";
                Options.Tasks.InputOnePerItem = false;
            }

            #endregion
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

            bool Invert = (bool)CheckInvert.IsChecked;
            bool Normalize = (bool)CheckNormalize.IsChecked;
            bool Preflip = (bool)CheckPreflip.IsChecked;

            bool Relative = (bool)CheckRelative.IsChecked;

            bool Filter = (bool)CheckFilter.IsChecked;
            bool Manual = (bool)CheckManual.IsChecked;

            int BoxSize = (int)Options.Tasks.Export2DBoxSize;
            int NormDiameter = (int)Options.Tasks.Export2DParticleDiameter;

            bool DoVolumes = (bool)RadioVolume.IsChecked;
            //if (!DoVolumes)
            //    Options.Tasks.TomoSubReconstructPrerotated = true;

            bool MakeSparse = (bool)CheckSparse.IsChecked;

            float3 AdditionalShiftAngstrom = (bool)CheckShiftParticles.IsChecked ? 
                                             new float3((float)SliderShiftParticlesX.Value,
                                                        (float)SliderShiftParticlesY.Value,
                                                        (float)SliderShiftParticlesZ.Value) : 
                                             new float3(0);

            ProgressWrite.Visibility = Visibility.Visible;
            ProgressWrite.IsIndeterminate = true;
            PanelButtons.Visibility = Visibility.Collapsed;
            PanelRemaining.Visibility = Visibility.Visible;

            foreach (var element in DisableWhileProcessing)
                element.IsEnabled = false;

            await Task.Run(() =>
            {
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
                List<string> ValidMovieNames = ValidSeries.Select(m => m.RootName).ToList();

                #endregion

                #region Read table and intersect its micrograph set with valid movies

                Star TableIn;

                if (Options.Tasks.InputOnePerItem)
                {
                    List<Star> Tables = new List<Star>();
                    foreach (var item in Series)
                    {
                        string StarPath = InputFolder + item.RootName + InputSuffix;
                        if (File.Exists(StarPath))
                        {
                            Star TableItem = new Star(StarPath);
                            if (!TableItem.HasColumn("rlnMicrographName"))
                                TableItem.AddColumn("rlnMicrographName", item.Name);
                            else
                                TableItem.SetColumn("rlnMicrographName", Helper.ArrayOfConstant(item.Name, TableItem.RowCount));

                            Tables.Add(TableItem);
                        }
                    }

                    TableIn = new Star(Tables.ToArray());
                }
                else
                {
                    TableIn = new Star(ImportPath);
                }

                if (!TableIn.HasColumn("rlnMicrographName"))
                    throw new Exception("Couldn't find rlnMicrographName column.");
                if (!TableIn.HasColumn("rlnCoordinateX"))
                    throw new Exception("Couldn't find rlnCoordinateX column.");
                if (!TableIn.HasColumn("rlnCoordinateY"))
                    throw new Exception("Couldn't find rlnCoordinateY column.");
                if (!TableIn.HasColumn("rlnCoordinateZ"))
                    throw new Exception("Couldn't find rlnCoordinateZ column.");

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

                    Groups = Groups.Where(group => ValidMovieNames.Any(n => group.Key == n)).ToDictionary(group => group.Key, group => group.Value);
                }

                bool[] RowsIncluded = new bool[TableIn.RowCount];
                foreach (var group in Groups)
                foreach (var r in group.Value)
                    RowsIncluded[r] = true;
                List<int> RowsNotIncluded = new List<int>();
                for (int r = 0; r < RowsIncluded.Length; r++)
                    if (!RowsIncluded[r])
                        RowsNotIncluded.Add(r);

                ValidSeries = ValidSeries.Where(v => Groups.Any(n => n.Key == v.RootName)).ToList();

                if (ValidSeries.Count == 0)     // Exit if there is nothing to export, otherwise errors will be thrown below
                    return;

                #endregion

                #region Make sure all columns are there

                if (!TableIn.HasColumn("rlnMagnification"))
                    TableIn.AddColumn("rlnMagnification", "10000.0");
                else
                    TableIn.SetColumn("rlnMagnification", Helper.ArrayOfConstant("10000.0", TableIn.RowCount));

                if (!TableIn.HasColumn("rlnDetectorPixelSize"))
                    TableIn.AddColumn("rlnDetectorPixelSize", Options.Tasks.TomoSubReconstructPixel.ToString("F5", CultureInfo.InvariantCulture));
                else
                    TableIn.SetColumn("rlnDetectorPixelSize", Helper.ArrayOfConstant(Options.Tasks.TomoSubReconstructPixel.ToString("F5", CultureInfo.InvariantCulture), TableIn.RowCount));

                if (!TableIn.HasColumn("rlnCtfMaxResolution"))
                    TableIn.AddColumn("rlnCtfMaxResolution", "999.0");

                if (!TableIn.HasColumn("rlnImageName"))
                    TableIn.AddColumn("rlnImageName", "None");

                if (!TableIn.HasColumn("rlnCtfImage"))
                    TableIn.AddColumn("rlnCtfImage", "None");

                List<Star> SeriesTablesOut = new List<Star>();

                #endregion
                
                if (IsCanceled)
                    return;

                #region Create worker processes

                int NDevices = GPU.GetDeviceCount();
                List<int> UsedDevices = WarpRuntime.MainWindow.GetDeviceList();
                List<int> UsedDeviceProcesses = new List<int>() { 0 };// Helper.Combine(Helper.ArrayOfFunction(i => UsedDevices.Select(d => d + i * NDevices).ToArray(), MainWindow.GlobalOptions.ProcessesPerDevice)).ToList();

                WorkerWrapper[] Workers = new WorkerWrapper[GPU.GetDeviceCount() * MainWindow.GlobalOptions.ProcessesPerDevice];
                foreach (var gpuID in UsedDeviceProcesses)
                {
                    Workers[gpuID] = new WorkerWrapper(gpuID, silent: false, attachDebugger: false);
                    Workers[gpuID].SetHeaderlessParams(new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                        Options.Import.HeaderlessOffset,
                                                        Options.Import.HeaderlessType);

                    Workers[gpuID].LoadGainRef(Options.Import.CorrectGain ? Options.Import.GainPath : "",
                                               Options.Import.GainFlipX,
                                               Options.Import.GainFlipY,
                                               Options.Import.GainTranspose,
                                               Options.Import.CorrectDefects ? Options.Import.DefectsPath : "");
                }

                #endregion

                Star TableOut = null;
                {
                    Dictionary<string, Star> MicrographTables = new Dictionary<string, Star>();

                    #region Get coordinates and angles

                    float[] PosX = TableIn.GetColumn("rlnCoordinateX").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    float[] PosY = TableIn.GetColumn("rlnCoordinateY").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    float[] PosZ = TableIn.GetColumn("rlnCoordinateZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    float[] ShiftX = TableIn.HasColumn("rlnOriginX") ? TableIn.GetColumn("rlnOriginX").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[TableIn.RowCount];
                    float[] ShiftY = TableIn.HasColumn("rlnOriginY") ? TableIn.GetColumn("rlnOriginY").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[TableIn.RowCount];
                    float[] ShiftZ = TableIn.HasColumn("rlnOriginZ") ? TableIn.GetColumn("rlnOriginZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[TableIn.RowCount];
                    

                    if (Options.Tasks.TomoSubReconstructNormalizedCoords)
                    {
                        for (int r = 0; r < TableIn.RowCount; r++)
                        {
                            PosX[r] *= (float)Options.Tomo.DimensionsX * (float)Options.Import.PixelSize;
                            PosY[r] *= (float)Options.Tomo.DimensionsY * (float)Options.Import.PixelSize;
                            PosZ[r] *= (float)Options.Tomo.DimensionsZ * (float)Options.Import.PixelSize;
                        }
                    }
                    else
                    {
                        for (int r = 0; r < TableIn.RowCount; r++)
                        {
                            PosX[r] = (PosX[r] - ShiftX[r]) * (float)Options.Tasks.InputPixelSize;
                            PosY[r] = (PosY[r] - ShiftY[r]) * (float)Options.Tasks.InputPixelSize;
                            PosZ[r] = (PosZ[r] - ShiftZ[r]) * (float)Options.Tasks.InputPixelSize;
                        }
                    }

                    float[] AngleRot = TableIn.HasColumn("rlnAngleRot") && Options.Tasks.TomoSubReconstructPrerotated ? TableIn.GetColumn("rlnAngleRot").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[TableIn.RowCount];
                    float[] AngleTilt = TableIn.HasColumn("rlnAngleTilt") && Options.Tasks.TomoSubReconstructPrerotated ? TableIn.GetColumn("rlnAngleTilt").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[TableIn.RowCount];
                    float[] AnglePsi = TableIn.HasColumn("rlnAnglePsi") && Options.Tasks.TomoSubReconstructPrerotated ? TableIn.GetColumn("rlnAnglePsi").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[TableIn.RowCount];

                    if (TableIn.HasColumn("rlnOriginX"))
                        TableIn.RemoveColumn("rlnOriginX");
                    if (TableIn.HasColumn("rlnOriginY"))
                        TableIn.RemoveColumn("rlnOriginY");
                    if (TableIn.HasColumn("rlnOriginZ"))
                        TableIn.RemoveColumn("rlnOriginZ");

                    if (AdditionalShiftAngstrom.Length() > 0)
                    {
                        for (int r = 0; r < TableIn.RowCount; r++)
                        {
                            Matrix3 R = Matrix3.Euler(AngleRot[r] * Helper.ToRad,
                                                      AngleTilt[r] * Helper.ToRad,
                                                      AnglePsi[r] * Helper.ToRad);
                            float3 RotatedShift = R * AdditionalShiftAngstrom;

                            PosX[r] += RotatedShift.X;
                            PosY[r] += RotatedShift.Y;
                            PosZ[r] += RotatedShift.Z;
                        }
                    }

                    if (Options.Tasks.TomoSubReconstructPrerotated)
                    {
                        if (TableIn.HasColumn("rlnAngleRot"))
                        {
                            TableIn.RemoveColumn("rlnAngleRot");
                            TableIn.AddColumn("rlnAngleRot", "0");
                        }
                        if (TableIn.HasColumn("rlnAngleTilt"))
                        {
                            TableIn.RemoveColumn("rlnAngleTilt");
                            TableIn.AddColumn("rlnAngleTilt", "0");
                        }
                        if (TableIn.HasColumn("rlnAnglePsi"))
                        {
                            TableIn.RemoveColumn("rlnAnglePsi");
                            TableIn.AddColumn("rlnAnglePsi", "0");
                        }
                    }

                    #endregion

                    Dispatcher.Invoke(() => ProgressWrite.MaxValue = ValidSeries.Count);


                    //Dispatcher.Invoke(() =>
                    //{
                    //    ProgressWrite.IsIndeterminate = true;
                    //    TextRemaining.Text = "?:??";
                    //});

                    Helper.ForEachGPU(ValidSeries, (series, gpuID) =>
                    {
                        if (IsCanceled)
                            return;

                        Stopwatch ItemTime = new Stopwatch();
                        ItemTime.Start();
                        
                        ProcessingOptionsTomoSubReconstruction ExportOptions = Options.GetProcessingTomoSubReconstruction();

                        #region Update row values

                        List<int> GroupRows = Groups.First(n => n.Key.Contains(series.RootName)).Value;

                        int pi = 0;
                        foreach (var r in GroupRows)
                        {
                            TableIn.SetRowValue(r, "rlnCtfMaxResolution", series.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture));

                            TableIn.SetRowValue(r, "rlnCoordinateX", (PosX[r] / (float)ExportOptions.BinnedPixelSizeMean).ToString("F3", CultureInfo.InvariantCulture));
                            TableIn.SetRowValue(r, "rlnCoordinateY", (PosY[r] / (float)ExportOptions.BinnedPixelSizeMean).ToString("F3", CultureInfo.InvariantCulture));
                            TableIn.SetRowValue(r, "rlnCoordinateZ", (PosZ[r] / (float)ExportOptions.BinnedPixelSizeMean).ToString("F3", CultureInfo.InvariantCulture));

                            #region Figure out relative or absolute path to sub-tomo and its CTF

                            string PathSubtomo = System.IO.Path.Combine(series.SubtomoDir, $"{series.RootName}{ExportOptions.Suffix}_{pi:D7}_{ExportOptions.BinnedPixelSizeMean:F2}A.mrc");
                            string PathCTF = //MakeSparse ?
                                            //(series.SubtomoDir + $"{series.RootName}_{pi:D7}_ctf_{ExportOptions.BinnedPixelSizeMean:F2}A.tif") :
                                            System.IO.Path.Combine(series.SubtomoDir, $"{series.RootName}{ExportOptions.Suffix}_{pi:D7}_ctf_{ExportOptions.BinnedPixelSizeMean:F2}A.mrc");
                            if (Relative)
                            {
                                Uri UriStar = new Uri(ExportPath);
                                PathSubtomo = UriStar.MakeRelativeUri(new Uri(PathSubtomo)).ToString();
                                PathCTF = UriStar.MakeRelativeUri(new Uri(PathCTF)).ToString();
                            }

                            #endregion

                            TableIn.SetRowValue(r, "rlnImageName", PathSubtomo);
                            TableIn.SetRowValue(r, "rlnCtfImage", PathCTF);

                            pi++;
                        }

                        #endregion

                        #region Populate micrograph table with rows for all exported particles

                        #endregion

                        #region Finally, reconstruct the actual sub-tomos

                        float3[] TomoPositions = Helper.Combine(GroupRows.Select(r => Helper.ArrayOfConstant(new float3(PosX[r], PosY[r], PosZ[r]), series.NTilts)).ToArray());
                        float3[] TomoAngles = Helper.Combine(GroupRows.Select(r => Helper.ArrayOfConstant(new float3(AngleRot[r], AngleTilt[r], AnglePsi[r]), series.NTilts)).ToArray());

                        if (DoVolumes)
                        {
                            Star MicrographTable = new Star(TableIn.GetColumnNames());
                        
                            foreach (var r in GroupRows)
                                MicrographTable.AddRow(TableIn.GetRow(r).ToArray());

                            if (!MicrographTable.HasColumn("rlnPixelSize"))
                                MicrographTable.AddColumn("rlnPixelSize", ExportOptions.BinnedPixelSizeMean.ToString("F5", CultureInfo.InvariantCulture));
                            if (!MicrographTable.HasColumn("rlnVoltage"))
                                MicrographTable.AddColumn("rlnVoltage", series.CTF.Voltage.ToString("F3", CultureInfo.InvariantCulture));
                            if (!MicrographTable.HasColumn("rlnSphericalAberration"))
                                MicrographTable.AddColumn("rlnSphericalAberration", series.CTF.Cs.ToString("F3", CultureInfo.InvariantCulture));

                            Workers[gpuID].TomoExportParticleSubtomos(series.Path, ExportOptions, TomoPositions, TomoAngles);
                            //series.ReconstructSubtomos(ExportOptions, TomoPositions, TomoAngles);

                            lock (MicrographTables)
                                MicrographTables.Add(series.RootName, MicrographTable);
                        }
                        else
                        {
                            Star SeriesTable;
                            Random Rand = new Random(123);
                            int[] Subsets = Helper.ArrayOfFunction(i => Rand.Next(1, 3), GroupRows.Count);
                            series.ReconstructParticleSeries(ExportOptions, TomoPositions, TomoAngles, ExportPath, out SeriesTable);

                            lock (MicrographTables)
                                MicrographTables.Add(series.RootName, SeriesTable);
                        }

                        #endregion

                        #region Add this micrograph's table to global collection, update remaining time estimate

                        lock (MicrographTables)
                        {
                            Timings.Add(ItemTime.ElapsedMilliseconds / (float)UsedDeviceProcesses.Count);

                            int MsRemaining = (int)(MathHelper.Mean(Timings) * (ValidSeries.Count - MicrographTables.Count));
                            TimeSpan SpanRemaining = new TimeSpan(0, 0, 0, 0, MsRemaining);

                            Dispatcher.Invoke(() => TextRemaining.Text = SpanRemaining.ToString((int)SpanRemaining.TotalHours > 0 ? @"hh\:mm\:ss" : @"mm\:ss"));

                            Dispatcher.Invoke(() =>
                            {
                                ProgressWrite.IsIndeterminate = false;
                                ProgressWrite.Value = MicrographTables.Count;
                            });
                        }

                        #endregion

                    }, 1, UsedDeviceProcesses);

                    if (MicrographTables.Count > 0)
                        TableOut = new Star(MicrographTables.Values.ToArray());
                }

                GlobalTasks.WaitAll();    // Writing out particles is async, so if workers are killed immediately they may not write out everything

                foreach (var worker in Workers)
                    worker?.Dispose();

                if (IsCanceled)
                    return;

                if (DoVolumes)
                {
                    TableOut.Save(ExportPath);
                }
                else
                {
                    ProcessingOptionsTomoSubReconstruction ExportOptions = Options.GetProcessingTomoSubReconstruction();
                    Dictionary<string, int> OpticsNumbering = new Dictionary<string, int>();

                    #region Construct optics group table, combine it with particles, and save resulting multi-table

                    Star Table2DMode = new StarParameters(new[] { "rlnTomoSubTomosAre2DStacks" }, new[] { "1" });
                    Star TableOptics = new Star(new[]
                    {
                        "rlnOpticsGroup",
                        "rlnOpticsGroupName",
                        "rlnSphericalAberration",
                        "rlnVoltage",
                        "rlnTomoTiltSeriesPixelSize",
                        "rlnCtfDataAreCtfPremultiplied",
                        "rlnImageDimensionality",
                        "rlnTomoSubtomogramBinning",
                        "rlnImagePixelSize",
                        "rlnImageSize",
                        "rlnAmplitudeContrast"
                    });
                    foreach (var series in ValidSeries)
                    {
                        OpticsNumbering.Add(series.RootName, TableOptics.RowCount + 1);

                        TableOptics.AddRow(new string[]
                        {
                            (TableOptics.RowCount + 1).ToString(),
                            $"opticsGroup{TableOptics.RowCount + 1}",
                            series.CTF.Cs.ToString("F3", CultureInfo.InvariantCulture),
                            series.CTF.Voltage.ToString("F3", CultureInfo.InvariantCulture),
                            Options.Import.PixelSize.ToString("F5", CultureInfo.InvariantCulture),
                            "1",
                            "2",
                            ExportOptions.DownsampleFactor.ToString("F5", CultureInfo.InvariantCulture),
                            ExportOptions.BinnedPixelSizeMean.ToString("F5", CultureInfo.InvariantCulture),
                            ExportOptions.BoxSize.ToString(),
                            series.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture)
                        });
                    }

                    TableOut.ModifyAllValuesInColumn("rlnOpticsGroup", s => OpticsNumbering[s].ToString());

                    Star.SaveMultitable(ExportPath, new Dictionary<string, Star>()
                    {
                        { "general", Table2DMode },
                        { "optics", TableOptics },
                        { "particles", TableOut }
                    });

                    #endregion

                    #region Construct tilt series table with projection matrices

                    Star TableGeneral = new Star(new string[]
                    {
                        "rlnTomoName",
                        "rlnTomoTiltSeriesName",
                        "rlnTomoFrameCount",
                        "rlnTomoSizeX",
                        "rlnTomoSizeY",
                        "rlnTomoSizeZ",
                        "rlnTomoHand",
                        "rlnOpticsGroupName",
                        "rlnTomoTiltSeriesPixelSize",
                        "rlnVoltage",
                        "rlnSphericalAberration",
                        "rlnAmplitudeContrast",
                        "rlnTomoImportFractionalDose"
                    });

                    Dictionary<string, Star> SeriesTables = new Dictionary<string, Star>() { { "global", TableGeneral } };

                    foreach (var series in ValidSeries)
                    {
                        List<int> UsedTilts = ExportOptions.DoLimitDose ? series.IndicesSortedDose.Take(ExportOptions.NTilts).ToList() : series.IndicesSortedDose.ToList();
                        float TiltDose = series.Dose[UsedTilts[1]] - series.Dose[UsedTilts[0]];
                        UsedTilts.Sort();

                        series.VolumeDimensionsPhysical = ExportOptions.DimensionsPhysical;

                        TableGeneral.AddRow(new string[]
                        {
                            series.RootName,
                            "dummy.mrc", //series.RootName + ".mrc",
                            UsedTilts.Count.ToString(),
                            Options.Tomo.DimensionsX.ToString(),
                            Options.Tomo.DimensionsY.ToString(),
                            Options.Tomo.DimensionsZ.ToString(),
                            "-1.0",
                            $"opticsGroup{OpticsNumbering[series.RootName]}",
                            Options.Import.PixelSize.ToString("F5", CultureInfo.InvariantCulture),
                            series.CTF.Voltage.ToString("F3", CultureInfo.InvariantCulture),
                            series.CTF.Cs.ToString("F3", CultureInfo.InvariantCulture),
                            series.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture),
                            TiltDose.ToString("F3", CultureInfo.InvariantCulture)
                        });

                        Star SeriesTable = new Star(new string[]
                        {
                            "rlnTomoProjX",
                            "rlnTomoProjY",
                            "rlnTomoProjZ",
                            "rlnTomoProjW",
                            "rlnDefocusU",
                            "rlnDefocusV",
                            "rlnDefocusAngle",
                            "rlnCtfScalefactor",
                            "rlnMicrographPreExposure"
                        });

                        float3[] TiltAngles = series.GetAngleInAllTilts(series.VolumeDimensionsPhysical * 0.5f);

                        foreach (var i in UsedTilts)
                        {
                            Matrix3 M = Matrix3.Euler(TiltAngles[i]);
                            float3 ImageCoords = series.GetPositionsInOneTilt(new[] { series.VolumeDimensionsPhysical * 0.5f }, i).First();
                            CTF TiltCTF = series.GetCTFParamsForOneTilt((float)Options.Import.PixelSize, new[] { ImageCoords.Z }, new[] { ImageCoords }, i, true).First();

                            SeriesTable.AddRow(new string[]
                            {
                                $"[{M.M11},{M.M12},{M.M13},0]",
                                $"[{M.M21},{M.M22},{M.M23},0]",
                                $"[{M.M31},{M.M32},{M.M33},0]",
                                "[0,0,0,1]",
                                ((TiltCTF.Defocus + TiltCTF.DefocusDelta / 2) * 1e4M).ToString("F1", CultureInfo.InvariantCulture),
                                ((TiltCTF.Defocus - TiltCTF.DefocusDelta / 2) * 1e4M).ToString("F1", CultureInfo.InvariantCulture),
                                TiltCTF.DefocusAngle.ToString("F3", CultureInfo.InvariantCulture),
                                TiltCTF.Scale.ToString("F3", CultureInfo.InvariantCulture),
                                series.Dose[i].ToString("F3", CultureInfo.InvariantCulture)
                            });
                        }

                        SeriesTables.Add(series.RootName, SeriesTable);
                    }

                    Star.SaveMultitable(System.IO.Path.Combine(Helper.PathToFolder(ExportPath), Helper.PathToName(ExportPath) + "_tomograms.star"), SeriesTables);

                    #endregion
                }
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
