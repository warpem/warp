using MahApps.Metro.Controls.Dialogs;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using Warp.Headers;
using Warp.Tools;

namespace Warp.Controls.TaskDialogs.Tomo
{
    /// <summary>
    /// Interaction logic for DialogTomoImportImod.xaml
    /// </summary>
    public partial class DialogTomoImportImod : UserControl
    {
        public string PathMdoc
        {
            get { return (string)GetValue(PathMdocProperty); }
            set { SetValue(PathMdocProperty, value); }
        }
        public static readonly DependencyProperty PathMdocProperty = DependencyProperty.Register("PathMdoc", typeof(string), typeof(DialogTomoImportImod), new PropertyMetadata("", (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public string PathImod
        {
            get { return (string)GetValue(PathImodProperty); }
            set { SetValue(PathImodProperty, value); }
        }
        public static readonly DependencyProperty PathImodProperty = DependencyProperty.Register("PathImod", typeof(string), typeof(DialogTomoImportImod), new PropertyMetadata("", (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public string PathMovie
        {
            get { return (string)GetValue(PathMovieProperty); }
            set { SetValue(PathMovieProperty, value); }
        }
        public static readonly DependencyProperty PathMovieProperty = DependencyProperty.Register("PathMovie", typeof(string), typeof(DialogTomoImportImod), new PropertyMetadata("", (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));
        
        public string Suffixes
        {
            get { return (string)GetValue(SuffixesProperty); }
            set { SetValue(SuffixesProperty, value); }
        }
        public static readonly DependencyProperty SuffixesProperty = DependencyProperty.Register("Suffixes", typeof(string), typeof(DialogTomoImportImod), new PropertyMetadata("", (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));
               
        public bool DontInvertTilts
        {
            get { return (bool)GetValue(DontInvertTiltsProperty); }
            set { SetValue(DontInvertTiltsProperty, value); }
        }
        public static readonly DependencyProperty DontInvertTiltsProperty = DependencyProperty.Register("DontInvertTilts", typeof(bool), typeof(DialogTomoImportImod), new PropertyMetadata(false, (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public decimal PixelSize
        {
            get { return (decimal)GetValue(PixelSizeProperty); }
            set { SetValue(PixelSizeProperty, value); }
        }
        public static readonly DependencyProperty PixelSizeProperty = DependencyProperty.Register("PixelSize", typeof(decimal), typeof(DialogTomoImportImod), new PropertyMetadata(1M, (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public decimal Dose
        {
            get { return (decimal)GetValue(DoseProperty); }
            set { SetValue(DoseProperty, value); }
        }
        public static readonly DependencyProperty DoseProperty = DependencyProperty.Register("Dose", typeof(decimal), typeof(DialogTomoImportImod), new PropertyMetadata(0.01M, (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public ObservableCollection<ParsedEntry> ParsedEntries
        {
            get { return (ObservableCollection<ParsedEntry>)GetValue(ParsedEntriesProperty); }
            set { SetValue(ParsedEntriesProperty, value); }
        }
        public static readonly DependencyProperty ParsedEntriesProperty = DependencyProperty.Register("ParsedEntries", typeof(ObservableCollection<ParsedEntry>), typeof(DialogTomoImportImod), new PropertyMetadata(null));
        

        public OptionsWarp Options;
        public event Action Close;

        public DialogTomoImportImod(OptionsWarp options)
        {
            InitializeComponent();

            Options = options;

            PixelSize = Options.Import.BinnedPixelSize;

            ParsedEntries = new ObservableCollection<ParsedEntry>();

            DataContext = this;
        }

        private async void Reevaluate()
        {
            ObservableCollection<ParsedEntry> OldEntries = ParsedEntries;
            ParsedEntries = new ObservableCollection<ParsedEntry>();

            PanelProgressIndicator.Visibility = Visibility.Visible;

            string _PathMdoc = PathMdoc;
            string _PathImod = PathImod;
            string _PathMovie = PathMovie;
            string _Suffixes = Suffixes;
            bool _DontInvertTilts = DontInvertTilts;
            decimal _PixelSize = PixelSize;
            decimal _Dose = Dose;
            var _ParsedEntries = ParsedEntries;

            StringBuilder AllExceptions = new StringBuilder();

            await Task.Run(() =>
            {
                try
                {
                    List<string> SuffixList = _Suffixes.Replace(" ", "").Split(new[] { ',', ';', '|' }, StringSplitOptions.RemoveEmptyEntries).ToList();
                    SuffixList.Add("");
                    SuffixList = SuffixList.Select(s => s + ".mdoc").ToList();

                    if (string.IsNullOrEmpty(_PathMdoc))
                        throw new Exception("No folder specified for .mdoc files!");

                    Dictionary<string, List<string>> MdocTuples = new Dictionary<string, List<string>>();
                    foreach (var filepath in Directory.EnumerateFiles(_PathMdoc, "*.mdoc"))
                    {
                        string FileName = Helper.PathToNameWithExtension(filepath);
                        if (FileName[0] == '.')
                            continue;

                        string Root = FileName;
                        foreach (var suffix in SuffixList)
                            if (FileName.Contains(suffix))
                            {
                                Root = FileName.Substring(0, FileName.IndexOf(suffix));
                                break;
                            }

                        Root = Root.Replace(".mrc", "");
                        //Root = Helper.PathToName(Root);

                        if (!MdocTuples.ContainsKey(Root))
                            MdocTuples.Add(Root, new List<string>());

                        MdocTuples[Root].Add(FileName);
                    }

                    if (MdocTuples.Count == 0)
                        throw new Exception($"No .mdoc files found in {_PathMdoc}.");
                                       
                    foreach (var mdocNames in MdocTuples)
                    {
                        try
                        {
                            float AxisAngle = 0;
                            List<MdocEntry> Entries = new List<MdocEntry>();
                            bool FoundTime = false;

                            foreach (var mdocName in mdocNames.Value)
                            {
                                using (TextReader Reader = new StreamReader(File.OpenRead(Path.Combine(_PathMdoc, mdocName))))
                                {
                                    string Line;
                                    while ((Line = Reader.ReadLine()) != null)
                                    {
                                        if (Line.Contains("Tilt axis angle = "))
                                        {
                                            string Suffix = Line.Substring(Line.IndexOf("Tilt axis angle = ") + "Tilt axis angle = ".Length);
                                            Suffix = Suffix.Substring(0, Suffix.IndexOf(","));

                                            AxisAngle = float.Parse(Suffix, CultureInfo.InvariantCulture);
                                            continue;
                                        }

                                        if (Line.Length < 7 || Line.Substring(1, 6) != "ZValue")
                                            continue;

                                        MdocEntry NewEntry = new MdocEntry();

                                        {
                                            string[] Parts = Line.Split(new[] { " = " }, StringSplitOptions.RemoveEmptyEntries);
                                            if (Parts[0] == "[ZValue")
                                                NewEntry.ZValue = int.Parse(Parts[1].Replace("]", ""));
                                        }

                                        while ((Line = Reader.ReadLine()) != null)
                                        {
                                            string[] Parts = Line.Split(new[] { " = " }, StringSplitOptions.RemoveEmptyEntries);
                                            if (Parts.Length < 2)
                                                break;

                                            if (Parts[0] == "TiltAngle")
                                                NewEntry.TiltAngle = (float)Math.Round(float.Parse(Parts[1], CultureInfo.InvariantCulture), 1);
                                            else if (Parts[0] == "ExposureDose")
                                                NewEntry.Dose = float.Parse(Parts[1], CultureInfo.InvariantCulture);
                                            else if (Parts[0] == "SubFramePath")
                                                NewEntry.Name = Parts[1].Substring(Parts[1].LastIndexOf("\\") + 1);
                                            else if (Parts[0] == "DateTime")
                                            {
                                                try
                                                {
                                                    try
                                                    {
                                                        NewEntry.Time = DateTime.ParseExact(Parts[1], "dd-MMM-yy  HH:mm:ss", CultureInfo.InvariantCulture);
                                                    }
                                                    catch
                                                    {
                                                        NewEntry.Time = DateTime.ParseExact(Parts[1], "dd-MMM-yyyy  HH:mm:ss", CultureInfo.InvariantCulture);
                                                    }

                                                    FoundTime = true;
                                                }
                                                catch
                                                {
                                                    FoundTime = false;
                                                }
                                            }
                                        }

                                        if (mdocNames.Value.Count == 1)
                                            Entries.RemoveAll(v => v.ZValue == NewEntry.ZValue);

                                        Entries.Add(NewEntry);
                                    }
                                }
                            }

                            List<MdocEntry> SortedTime = new List<MdocEntry>(Entries);
                            SortedTime.Sort((a, b) => a.Time.CompareTo(b.Time));

                            // Do running dose
                            float Accumulated = 0;
                            foreach (var entry in SortedTime)
                            {
                                Accumulated += entry.Dose;
                                entry.Dose = Accumulated;
                            }

                            // In case mdoc doesn't tell anything about the dose, use default value
                            if (_Dose > 0)
                                for (int i = 0; i < SortedTime.Count; i++)
                                {
                                    SortedTime[i].Dose = (i + 0.5f) * (float)_Dose;
                                    Accumulated += (float)_Dose;
                                }

                            // Sort entires by angle and time (accumulated dose)
                            List<MdocEntry> SortedAngle = new List<MdocEntry>(Entries);
                            SortedAngle.Sort((a, b) => a.TiltAngle.CompareTo(b.TiltAngle));
                            // Sometimes, there will be 2 0-tilts at the beginning of plus and minus series. 
                            // Sort them according to dose, considering in which order plus and minus were acquired
                            float DoseMinus = SortedAngle.Take(SortedAngle.Count / 2).Select(v => v.Dose).Sum();
                            float DosePlus = SortedAngle.Skip(SortedAngle.Count / 2).Take(SortedAngle.Count / 2).Select(v => v.Dose).Sum();
                            int OrderCorrection = DoseMinus < DosePlus ? 1 : -1;
                            SortedAngle.Sort((a, b) => a.TiltAngle.CompareTo(b.TiltAngle) != 0 ? a.TiltAngle.CompareTo(b.TiltAngle) : a.Dose.CompareTo(b.Dose) * OrderCorrection);

                            if (string.IsNullOrEmpty(_PathMovie))
                                throw new Exception("No folder specified for the original movies.");

                            SortedAngle.RemoveAll(v => IsMovieExcluded(Path.Combine(_PathMovie, v.Name)));
                            //SortedAngle.RemoveAll(v => Math.Abs(v.TiltAngle) > 50);

                            if (SortedAngle.Count == 0)
                                throw new Exception($"No movies found for {mdocNames.Key}, or none have been processed in Warp yet.");

                            string CutviewsPath1 = Path.Combine(_PathImod, mdocNames.Key, mdocNames.Key + "_cutviews0.rawtlt");
                            string CutviewsPath2 = Path.Combine(_PathImod, mdocNames.Key + "_cutviews0.rawtlt");
                            string CutviewsPath = File.Exists(CutviewsPath1) ? CutviewsPath1 : (File.Exists(CutviewsPath2) ? CutviewsPath2 : null);
                            if (CutviewsPath != null)
                            {
                                List<float> CutAngles = new List<float>();
                                using (TextReader Reader = File.OpenText(CutviewsPath))
                                {
                                    string Line;
                                    while ((Line = Reader.ReadLine()) != null)
                                        try
                                        {
                                            CutAngles.Add(float.Parse(Line, CultureInfo.InvariantCulture));
                                        }
                                        catch { }
                                }

                                SortedAngle.RemoveAll(v => CutAngles.Any(a => Math.Abs(a - v.TiltAngle) < 0.2));
                            }

                            string XfPath1 = Path.Combine(_PathImod, mdocNames.Key, mdocNames.Key + ".xf");
                            string XfPath2 = Path.Combine(_PathImod, mdocNames.Key + ".xf");
                            string XfPath3 = Path.Combine(_PathImod, mdocNames.Key, mdocNames.Key + "_Imod", mdocNames.Key + ".xf");
                            string XfPath = null;
                            try
                            {
                                XfPath = (new[] { XfPath3, XfPath1, XfPath2, XfPath3 }).First(s => File.Exists(s));
                            }
                            catch { }
                            bool HasAlignmentData = false;
                            int Pretilt = 0;
                            if (XfPath != null)
                            {
                                HasAlignmentData = true;
                                using (TextReader Reader = new StreamReader(File.OpenRead(XfPath)))
                                {
                                    string Line;
                                    for (int i = 0; i < SortedAngle.Count; i++)
                                    {
                                        Line = Reader.ReadLine();
                                        if (Line == null)
                                            throw new Exception("There are fewer lines in the .xf file than there are tilts. Please deactivate the same tilts in Warp.");
                                        string[] Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                                        float2 VecX = new float2(float.Parse(Parts[0], CultureInfo.InvariantCulture),
                                                                 float.Parse(Parts[2], CultureInfo.InvariantCulture));
                                        float2 VecY = new float2(float.Parse(Parts[1], CultureInfo.InvariantCulture),
                                                                 float.Parse(Parts[3], CultureInfo.InvariantCulture));

                                        Matrix3 Rotation = new Matrix3(VecX.X, VecX.Y, 0, VecY.X, VecY.Y, 0, 0, 0, 1);
                                        float3 Euler = Matrix3.EulerFromMatrix(Rotation);

                                        SortedAngle[i].AxisAngle = Euler.Z * Helper.ToDeg;

                                        //SortedAngle[i].Shift += VecX * float.Parse(Parts[4], CultureInfo.InvariantCulture) + VecY * float.Parse(Parts[5], CultureInfo.InvariantCulture);
                                        float3 Shift = new float3(-float.Parse(Parts[4], CultureInfo.InvariantCulture), -float.Parse(Parts[5], CultureInfo.InvariantCulture), 0);
                                        Shift = Rotation.Transposed() * Shift;

                                        SortedAngle[i].Shift += new float2(Shift);
                                    }
                                }

                                string AlignLogPath = Path.Combine(_PathImod, mdocNames.Key, "align_patchtracking.log");
                                if (File.Exists(AlignLogPath))
                                {
                                    List<float4> Markers = new List<float4>();
                                    using (TextReader Reader = new StreamReader(File.OpenRead(AlignLogPath)))
                                    {
                                        string Line;
                                        while ((Line = Reader.ReadLine()) != null)
                                        {
                                            if (Line.Contains("3-D point coordinates (with centroid zero)"))
                                            {
                                                Reader.ReadLine();

                                                while ((Line = Reader.ReadLine()) != null && Line.Length > 5)
                                                {
                                                    string[] Parts = Line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                                                    Markers.Add(new float4(float.Parse(Parts[1], CultureInfo.InvariantCulture),
                                                                           float.Parse(Parts[2], CultureInfo.InvariantCulture),
                                                                           float.Parse(Parts[3], CultureInfo.InvariantCulture),
                                                                           1 / float.Parse(Parts[6], CultureInfo.InvariantCulture)));
                                                }

                                                break;
                                            }
                                        }
                                    }

                                    if (Markers.Count >= 3)
                                    {
                                        float3 Normal = MathHelper.FitPlaneWeighted(Markers.ToArray());
                                        float Angle = MathF.Asin(Normal.X) * Helper.ToDeg;
                                        Pretilt = (int)Math.Round(Angle);
                                    }
                                }
                            }

                            string SolutionPath = Path.Combine(_PathImod, mdocNames.Key, "taSolution.log");
                            if (File.Exists(SolutionPath))
                            {
                                try
                                {
                                    using (TextReader Reader = new StreamReader(File.OpenRead(SolutionPath)))
                                    {
                                        string Line;
                                        while ((Line = Reader.ReadLine()) != null)
                                        {
                                            if (Line.ToLower().Contains("view") && Line.ToLower().Contains("rotation") && Line.ToLower().Contains("mag"))
                                                break;
                                        }

                                        for (int i = 0; i < SortedAngle.Count; i++)
                                        {
                                            Line = Reader.ReadLine();
                                            string[] Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                                            //SortedAngle[i].AxisAngle = float.Parse(Parts[1], CultureInfo.InvariantCulture);
                                            SortedAngle[i].TiltAngle += float.Parse(Parts[3], CultureInfo.InvariantCulture);
                                        }
                                    }
                                }
                                catch { }
                            }

                            Star Table = new Star(new[]
                            {
                                "wrpMovieName",
                                "wrpAngleTilt",
                                "wrpAxisAngle",
                                "wrpAxisOffsetX",
                                "wrpAxisOffsetY",
                                "wrpDose"
                            });

                            //Image Stack = CreateStacks ? new Image(new int3(Entries[0].Micrograph.Dims.X, Entries[0].Micrograph.Dims.Y, Entries.Count)) : null;
                            //float[][] StackData = Stack?.GetHost(Intent.Write);

                            for (int i = 0; i < SortedAngle.Count; i++)
                            {
                                //if (CreateStacks)
                                //    StackData[i] = SortedAngle[i].Micrograph.GetHost(Intent.Read)[0];

                                string PathToMovie = Path.Combine(_PathMovie, SortedAngle[i].Name);

                                Uri UriMovie = new Uri(Options.Import.DataFolder);
                                string MovieRelativePath = UriMovie.MakeRelativeUri(new Uri(PathToMovie)).ToString();

                                Table.AddRow(new string[]
                                {
                                    MovieRelativePath,
                                    (SortedAngle[i].TiltAngle * (_DontInvertTilts ? 1 : -1)).ToString(CultureInfo.InvariantCulture),
                                    SortedAngle[i].AxisAngle.ToString(CultureInfo.InvariantCulture),
                                    (SortedAngle[i].Shift.X * (float)_PixelSize).ToString(CultureInfo.InvariantCulture),
                                    (SortedAngle[i].Shift.Y * (float)_PixelSize).ToString(CultureInfo.InvariantCulture),
                                    SortedAngle[i].Dose.ToString(CultureInfo.InvariantCulture)
                                });
                            }

                            //Table.Save(Path.Combine(Options.Import.Folder, mdocNames.Key + ".tomostar"));
                            Dispatcher.Invoke(() =>
                            {
                                ParsedEntries.Add(new ParsedEntry()
                                {
                                    DoImport = OldEntries.Any(v => v.Name == mdocNames.Key) ? OldEntries.First(v => v.Name == mdocNames.Key).DoImport : true,
                                    Name = mdocNames.Key,
                                    NTilts = Table.RowCount,
                                    Dose = (int)Math.Round(SortedAngle.Select(v => v.Dose).Max()),
                                    Aligned = HasAlignmentData,
                                    Pretilt = Pretilt,
                                    Table = Table,
                                    TiltAngles = SortedAngle.Select(e => e.TiltAngle).ToArray(),
                                    Rotation = AxisAngle
                                });
                            });

                            // Write out tilt series into IMOD folder + its individual subfolder
                            //if (CreateStacks)
                            //{
                            //    Console.WriteLine("Writing tilt series.");
                            //    Directory.CreateDirectory(DataRoot + ImodFolder + mdocNames.Key);
                            //    Stack.WriteMRC(DataRoot + ImodFolder + mdocNames.Key + "\\" + mdocNames.Key + ".st");
                            //}
                        }
                        catch (Exception exc)
                        {
                            AllExceptions.AppendLine(exc.Message);
                            AllExceptions.AppendLine("");
                        }
                    }
                }
                catch (Exception exc)
                {
                    AllExceptions.AppendLine(exc.Message);
                    AllExceptions.AppendLine("");
                }
            });

            ButtonWrite.IsEnabled = true;
            ButtonCreateStacks.IsEnabled = true;

            PanelProgressIndicator.Visibility = Visibility.Hidden;

            PanelImportResults.Visibility = Visibility.Visible;
            ListParsedEntries.Visibility = Visibility.Visible;
            TextErrors.Visibility = Visibility.Collapsed;

            if (AllExceptions.Length > 0)
            {
                TextErrors.Text = AllExceptions.ToString();
                TextErrors.Visibility = Visibility.Visible;
            }
        }

        private void ButtonMdocPath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                SelectedPath = Options.Import.DataFolder
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                    Dialog.SelectedPath += '\\';

                PathMdoc = Dialog.SelectedPath;
                ButtonMdocPathText.Text = PathMdoc;
            }
        }

        private void ButtonImodPath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                SelectedPath = Options.Import.DataFolder
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                    Dialog.SelectedPath += '\\';

                PathImod = Dialog.SelectedPath;
                ButtonImodPathText.Text = PathImod;
            }
        }

        private void ButtonMoviePath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                SelectedPath = Options.Import.DataFolder
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                    Dialog.SelectedPath += '\\';

                PathMovie = Dialog.SelectedPath;
                ButtonMoviePathText.Text = PathMovie;
            }
        }

        private async void ButtonWrite_Click(object sender, RoutedEventArgs e)
        {
            List<string> Duplicates = new List<string>();

            foreach (var entry in ParsedEntries.Where(v => v.DoImport))
            {
                string TablePath = Path.Combine(Options.Import.DataFolder, entry.Name + ".tomostar");
                entry.Table.Save(TablePath);

                if (File.Exists(Path.Combine(Options.Import.ProcessingOrDataFolder, entry.Name + ".xml")))
                    Duplicates.Add(Path.Combine(Options.Import.ProcessingOrDataFolder, entry.Name));
            }

            if (Duplicates.Count > 0)
            {
                var Result = await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Found old processing results!",
                                                                                                 $"It looks like you already used Warp to process earlier versions of some of these TomoSTAR files.\n" +
                                                                                                 $"If you write out new TomoSTAR files now, the old XML metadata won't be updated until you delete\n" +
                                                                                                 $"them manually, reload the items in Warp, and reprocess them.\n\n" +
                                                                                                 $"Would you like to automatically delete the XML metadata and reload the items now?",
                                                                                                 MessageDialogStyle.AffirmativeAndNegative);
                if (Result == MessageDialogResult.Affirmative)
                {
                    foreach (var item in Duplicates)
                        File.Delete(item + ".xml");

                    WarpRuntime.MainWindow.AdjustInput();
                }
            }

            Close?.Invoke();
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonCreateStacks_Click(object sender, RoutedEventArgs e)
        {
            var ProgressDialog = await WarpRuntime.MainWindow.ShowProgressAsync("Writing data...", "");

            ObservableCollection<ParsedEntry> _ParsedEntries = new ObservableCollection<ParsedEntry>(ParsedEntries.Where(v => v.DoImport));

            string _PathMovie = PathMovie;

            await Task.Run(() =>
            {
                Image StackBuffer = null;
                int MaxTilts = 0;

                int idone = 0;
                foreach (var entry in _ParsedEntries)
                {
                    Dispatcher.Invoke(() => ProgressDialog.SetTitle($"Writing stack for {entry.Name}..."));

                    Directory.CreateDirectory(Path.Combine(Options.Import.ProcessingOrDataFolder, "imod", entry.Name));

                    Movie[] Movies = Helper.ArrayOfFunction(i => new Movie(Path.Combine(_PathMovie, Path.GetFileName(entry.Table.GetRowValue(i, "wrpMovieName")))), entry.Table.RowCount);

                    MapHeader Header = MapHeader.ReadFromFile(Movies[0].AveragePath);
                    int3 StackDims = new int3(Header.Dimensions.X, Header.Dimensions.Y, Movies.Length);

                    if (StackBuffer == null)
                    {
                        StackBuffer = new Image(StackDims);
                    }
                    else if (StackBuffer.Dims != StackDims)
                    {
                        if (MaxTilts < StackDims.Z || StackBuffer.Dims.Slice() != StackDims.Slice())
                        {
                            StackBuffer.Dispose();
                            StackBuffer = new Image(StackDims);
                        }
                        else
                        {
                            StackBuffer.Dims = StackDims;
                        }
                    }

                    MaxTilts = StackDims.Z;

                    float[][] StackData = StackBuffer.GetHost(Intent.Write);

                    for (int i = 0; i < Movies.Length; i++)
                        IOHelper.ReadMapFloatPatient(50, 500,
                                                     Movies[i].AveragePath,
                                                     new int2(1),
                                                     0,
                                                     typeof(float),
                                                     new[] { 0 },
                                                     null,
                                                     new[] { StackData[i] });

                    //HeaderMRC StackHeader = new HeaderMRC();
                    //StackHeader.ImodTilt = entry.TiltAngles;
                    //StackHeader.ImodRotation = entry.Rotation;
                    //StackHeader.PixelSize = Header.PixelSize;

                    StackBuffer.WriteMRC(Path.Combine(Options.Import.ProcessingOrDataFolder, "imod", entry.Name, entry.Name + ".st"), Header.PixelSize.X, true, null);

                    using (TextWriter Writer = File.CreateText(Path.Combine(Options.Import.ProcessingOrDataFolder, "imod", entry.Name, entry.Name + ".rawtlt")))
                    {
                        foreach (var angle in entry.TiltAngles)
                            Writer.WriteLine(angle.ToString(CultureInfo.InvariantCulture));
                    }

                    //HeaderMRC H = (HeaderMRC)MapHeader.ReadFromFile(Path.Combine(Options.Import.Folder, "imod", entry.Name, entry.Name + ".st"));

                    Dispatcher.Invoke(() => ProgressDialog.SetProgress(++idone / (double)_ParsedEntries.Count));
                }
            });

            await ProgressDialog.CloseAsync();

            Close?.Invoke();
        }

        Dictionary<string, bool> MovieExcludedStatus = new Dictionary<string, bool>();

        private bool IsMovieExcluded(string path)
        {
            if (!MovieExcludedStatus.ContainsKey(path))
            {
                bool Status = false;

                string PathXml = path.Substring(0, path.LastIndexOf('.')) + ".xml";
                if (!File.Exists(PathXml))// || !File.Exists(path)) // Checking path breaks with separate data and processing dirs
                    Status = true;

                Movie M = new Movie(path);
                if (M.UnselectManual != null && (bool)M.UnselectManual)
                    Status = true;

                if (!File.Exists(M.AveragePath))
                    Status = true;

                MovieExcludedStatus.Add(path, Status);
            }

            return MovieExcludedStatus[path];
        }
    }
}
