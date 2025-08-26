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
using MahApps.Metro.Controls;
using MahApps.Metro.Controls.Dialogs;
using Warp.Sociology;
using Warp.Tools;
using Warp;
using Image = Warp.Image;

namespace M.Controls.Sociology.Dialogs
{
    /// <summary>
    /// Interaction logic for DialogSpeciesAddParticles.xaml
    /// </summary>
    public partial class DialogSpeciesParticleSets : UserControl
    {
        public event Action Add;
        public event Action Close;

        public int TemporalResMov
        {
            get { return (int)GetValue(TemporalResMovProperty); }
            set { SetValue(TemporalResMovProperty, value); }
        }
        public static readonly DependencyProperty TemporalResMovProperty = DependencyProperty.Register("TemporalResMov", typeof(int), typeof(DialogSpeciesParticleSets), new PropertyMetadata(1));

        #region Properties particles

        public Star TableWarp, TableRelion;
        private string PathWarp, PathRelion;

        public decimal ParticleCoordinatesPixel
        {
            get { return (decimal)GetValue(ParticleCoordinatesPixelProperty); }
            set { SetValue(ParticleCoordinatesPixelProperty, value); }
        }
        public static readonly DependencyProperty ParticleCoordinatesPixelProperty = DependencyProperty.Register("ParticleCoordinatesPixel", typeof(decimal), typeof(DialogSpeciesParticleSets), new PropertyMetadata(1M, (sender, value) => ((DialogSpeciesParticleSets)sender).UpdateParticles()));

        public decimal ParticleShiftsPixel
        {
            get { return (decimal)GetValue(ParticleShiftsPixelProperty); }
            set { SetValue(ParticleShiftsPixelProperty, value); }
        }
        public static readonly DependencyProperty ParticleShiftsPixelProperty = DependencyProperty.Register("ParticleShiftsPixel", typeof(decimal), typeof(DialogSpeciesParticleSets), new PropertyMetadata(1M, (sender, value) => ((DialogSpeciesParticleSets)sender).UpdateParticles()));

        private bool[] UseSource;
        private DataSource[] ValidSources;
        public DataSource[] UsedSources => ValidSources.Where((s, i) => UseSource[i]).ToArray();

        private int ParticlesMatched, ParticlesUnmatched;

        private Particle[] ParticlesOld;
        private Particle[] ParticlesNew;

        public Particle[] ParticlesFinal;

        #endregion

        public decimal ToleranceDistance
        {
            get { return (decimal)GetValue(ToleranceDistanceProperty); }
            set { SetValue(ToleranceDistanceProperty, value); }
        }
        public static readonly DependencyProperty ToleranceDistanceProperty = DependencyProperty.Register("ToleranceDistance", typeof(decimal), typeof(DialogSpeciesParticleSets), new PropertyMetadata(10M, (sender, value) => ((DialogSpeciesParticleSets)sender).UpdateSets()));
        
        float[] ClosestDistanceOld;
        float[] ClosestDistanceNew;

        int NBins = 100;
        float MaxDistance = -1;

        public DialogSpeciesParticleSets(Population population, Species species)
        {
            InitializeComponent();

            DataContext = this;

            TemporalResMov = species.TemporalResolutionMovement;
            ParticlesOld = species.Particles;

            ValidSources = population.Sources.Where(s => !s.IsRemote).ToArray();
            UseSource = Helper.ArrayOfConstant(true, ValidSources.Length);
            for (int i = 0; i < ValidSources.Length; i++)
            {
                CheckBox CheckSource = new CheckBox
                {
                    Content = ValidSources[i].Name,
                    FontSize = 18,
                    IsChecked = true
                };
                int s = i;
                CheckSource.Click += (source, e) =>
                {
                    UseSource[s] = (bool)((CheckBox)source).IsChecked;
                    UpdateParticles();
                };
                PanelSources.Children.Add(CheckSource);
            }

            UpdateParticles();
            Revalidate();
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private void ButtonAdd_OnClick(object sender, RoutedEventArgs e)
        {
            Add?.Invoke();
        }

        void Revalidate()
        {
            if (ValidateParticles() && ValidateSets())
            {
                ButtonAdd.Visibility = Visibility.Visible;
            }
            else
            {
                ButtonAdd.Visibility = Visibility.Hidden;
            }
        }

        bool PauseParticleUpdates = false;
        bool PauseSetUpdates = false;

        async void UpdateParticles()
        {
            if (!IsInitialized || PauseParticleUpdates)
                return;


            bool UseWarp = (bool)RadioParticlesWarp.IsChecked;
            float AngPixCoords = (float)ParticleCoordinatesPixel;
            float AngPixShifts = (float)ParticleShiftsPixel;
            int ResMov = TemporalResMov;
            int ResRot = TemporalResMov;

            TextParticlesError.Visibility = Visibility.Collapsed;
            ProgressParticles.Visibility = Visibility.Visible;

            ParticlesNew = null;

            ClosestDistanceOld = null;
            ClosestDistanceNew = null;

            await Task.Run(() =>
            {
                if (UseWarp && TableWarp != null)
                {
                    #region Figure out missing sources

                    Dictionary<string, int> ParticleHashes = new Dictionary<string, int>();
                    foreach (var hash in TableWarp.GetColumn("wrpSourceHash"))
                    {
                        if (!ParticleHashes.ContainsKey(hash))
                            ParticleHashes.Add(hash, 0);
                        ParticleHashes[hash]++;
                    }

                    HashSet<string> AvailableHashes = new HashSet<string>(Helper.Combine(ValidSources.Select(s => s.Files.Keys.ToArray()).ToArray()));
                    List<string> HashesNotFound = ParticleHashes.Keys.Where(hash => !AvailableHashes.Contains(hash)).ToList();

                    ParticlesUnmatched = HashesNotFound.Sum(h => ParticleHashes[h]);
                    ParticlesMatched = TableWarp.RowCount - ParticlesUnmatched;

                    #endregion

                    #region Create particles

                    int TableResMov = 1, TableResRot = 1;
                    string[] PrefixesMov = { "wrpCoordinateX", "wrpCoordinateY", "wrpCoordinateZ" };
                    string[] PrefixesRot = { "wrpAngleRot", "wrpAngleTilt", "wrpAnglePsi" };

                    while (true)
                    {
                        if (PrefixesMov.Any(p => !TableWarp.HasColumn(p + (TableResMov + 1).ToString())))
                            break;
                        TableResMov++;
                    }
                    while (true)
                    {
                        if (PrefixesRot.Any(p => !TableWarp.HasColumn(p + (TableResRot + 1).ToString())))
                            break;
                        TableResRot++;
                    }

                    string[] NamesCoordX = Helper.ArrayOfFunction(i => $"wrpCoordinateX{i + 1}", TableResMov);
                    string[] NamesCoordY = Helper.ArrayOfFunction(i => $"wrpCoordinateY{i + 1}", TableResMov);
                    string[] NamesCoordZ = Helper.ArrayOfFunction(i => $"wrpCoordinateZ{i + 1}", TableResMov);

                    string[] NamesAngleRot = Helper.ArrayOfFunction(i => $"wrpAngleRot{i + 1}", TableResRot);
                    string[] NamesAngleTilt = Helper.ArrayOfFunction(i => $"wrpAngleTilt{i + 1}", TableResRot);
                    string[] NamesAnglePsi = Helper.ArrayOfFunction(i => $"wrpAnglePsi{i + 1}", TableResRot);

                    float[][] ColumnsCoordX = NamesCoordX.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
                    float[][] ColumnsCoordY = NamesCoordY.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
                    float[][] ColumnsCoordZ = NamesCoordZ.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();

                    float[][] ColumnsAngleRot = NamesAngleRot.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
                    float[][] ColumnsAngleTilt = NamesAngleTilt.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
                    float[][] ColumnsAnglePsi = NamesAnglePsi.Select(n => TableWarp.GetColumn(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();

                    int[] ColumnSubset = TableWarp.GetColumn("wrpRandomSubset").Select(v => int.Parse(v) - 1).ToArray();

                    string[] ColumnSourceName = TableWarp.GetColumn("wrpSourceName");
                    string[] ColumnSourceHash = TableWarp.GetColumn("wrpSourceHash");

                    ParticlesNew = new Particle[TableWarp.RowCount];

                    for (int p = 0; p < ParticlesNew.Length; p++)
                    {
                        float3[] Coordinates = Helper.ArrayOfFunction(i => new float3(ColumnsCoordX[i][p],
                                                                                      ColumnsCoordY[i][p],
                                                                                      ColumnsCoordZ[i][p]), TableResMov);
                        float3[] Angles = Helper.ArrayOfFunction(i => new float3(ColumnsAngleRot[i][p],
                                                                                 ColumnsAngleTilt[i][p],
                                                                                 ColumnsAnglePsi[i][p]), TableResRot);

                        ParticlesNew[p] = new Particle(Coordinates, Angles, ColumnSubset[p], ColumnSourceName[p], ColumnSourceHash[p]);
                        ParticlesNew[p].ResampleCoordinates(ResMov);
                        ParticlesNew[p].ResampleAngles(ResRot);
                    }

                    #endregion
                }
                else if (!UseWarp && TableRelion != null)
                {
                    #region Figure out missing and ambigous sources

                    Dictionary<string, int> ParticleImageNames = new Dictionary<string, int>();
                    foreach (var imageName in TableRelion.GetColumn("rlnMicrographName"))
                    {
                        if (!ParticleImageNames.ContainsKey(imageName))
                            ParticleImageNames.Add(imageName, 0);
                        ParticleImageNames[imageName]++;
                    }

                    List<string> NamesNotFound = new List<string>();
                    List<string> NamesAmbiguous = new List<string>();
                    HashSet<string> NamesGood = new HashSet<string>();
                    foreach (var imageName in ParticleImageNames.Keys)
                    {
                        int Possibilities = UsedSources.Count(source => source.Files.Values.Contains(imageName));

                        if (Possibilities == 0)
                            NamesNotFound.Add(imageName);
                        else if (Possibilities > 1)
                            NamesAmbiguous.Add(imageName);
                        else
                            NamesGood.Add(imageName);
                    }

                    if (NamesAmbiguous.Count > 0)
                    {
                        Dispatcher.Invoke(() =>
                        {
                            TextParticlesError.Text = $"{NamesAmbiguous.Count} image names are ambiguous between selected data sources!";
                            TextParticlesError.Visibility = Visibility.Visible;
                        });
                    }

                    ParticlesUnmatched = NamesNotFound.Sum(h => ParticleImageNames[h]);
                    ParticlesMatched = TableRelion.RowCount - ParticlesUnmatched;

                    #endregion

                    #region Create particles

                    Dictionary<string, string> ReverseMapping = new Dictionary<string, string>();
                    foreach (var source in UsedSources)
                        foreach (var pair in source.Files)
                            if (NamesGood.Contains(pair.Value))
                                ReverseMapping.Add(pair.Value, pair.Key);

                    List<int> ValidRows = new List<int>(TableRelion.RowCount);
                    string[] ColumnMicNames = TableRelion.GetColumn("rlnMicrographName");
                    for (int r = 0; r < ColumnMicNames.Length; r++)
                        if (ReverseMapping.ContainsKey(ColumnMicNames[r]))
                            ValidRows.Add(r);
                    Star CleanRelion = TableRelion.CreateSubset(ValidRows);

                    int NParticles = CleanRelion.RowCount;
                    bool IsTomogram = CleanRelion.HasColumn("rlnCoordinateZ");

                    float[] CoordinatesX = CleanRelion.GetColumn("rlnCoordinateX").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixCoords).ToArray();
                    float[] CoordinatesY = CleanRelion.GetColumn("rlnCoordinateY").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixCoords).ToArray();
                    float[] CoordinatesZ = IsTomogram ? CleanRelion.GetColumn("rlnCoordinateZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixCoords).ToArray() : new float[NParticles];

                    float[] OffsetsX = new float[NParticles];
                    float[] OffsetsY = new float[NParticles];
                    float[] OffsetsZ = new float[NParticles];

                    if (CleanRelion.HasColumn("rlnOriginX"))
                    {
                        OffsetsX = CleanRelion.HasColumn("rlnOriginX") ? CleanRelion.GetColumn("rlnOriginX").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixShifts).ToArray() : OffsetsX;
                        OffsetsY = CleanRelion.HasColumn("rlnOriginY") ? CleanRelion.GetColumn("rlnOriginY").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixShifts).ToArray() : OffsetsY;
                        OffsetsZ = CleanRelion.HasColumn("rlnOriginZ") ? CleanRelion.GetColumn("rlnOriginZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture) * AngPixShifts).ToArray() : OffsetsZ;
                    }
                    else if (CleanRelion.HasColumn("rlnOriginXAngst"))
                    {
                        OffsetsX = CleanRelion.HasColumn("rlnOriginXAngst") ? CleanRelion.GetColumn("rlnOriginXAngst").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : OffsetsX;
                        OffsetsY = CleanRelion.HasColumn("rlnOriginYAngst") ? CleanRelion.GetColumn("rlnOriginYAngst").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : OffsetsY;
                        OffsetsZ = CleanRelion.HasColumn("rlnOriginZAngst") ? CleanRelion.GetColumn("rlnOriginZAngst").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : OffsetsZ;
                    }

                    float3[] Coordinates = Helper.ArrayOfFunction(p => new float3(CoordinatesX[p] - OffsetsX[p], CoordinatesY[p] - OffsetsY[p], CoordinatesZ[p] - OffsetsZ[p]), NParticles);

                    float[] AnglesRot = CleanRelion.GetColumn("rlnAngleRot").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    float[] AnglesTilt = CleanRelion.GetColumn("rlnAngleTilt").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    float[] AnglesPsi = CleanRelion.GetColumn("rlnAnglePsi").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

                    float3[] Angles = Helper.ArrayOfFunction(p => new float3(AnglesRot[p], AnglesTilt[p], AnglesPsi[p]), NParticles);

                    int[] Subsets = CleanRelion.HasColumn("rlnRandomSubset") ? CleanRelion.GetColumn("rlnRandomSubset").Select(v => int.Parse(v, CultureInfo.InvariantCulture) - 1).ToArray() : Helper.ArrayOfFunction(i => i % 2, NParticles);

                    string[] MicrographNames = CleanRelion.GetColumn("rlnMicrographName").ToArray();
                    string[] MicrographHashes = MicrographNames.Select(v => ReverseMapping[v]).ToArray();

                    ParticlesNew = Helper.ArrayOfFunction(p => new Particle(new[] { Coordinates[p] }, new[] { Angles[p] }, Subsets[p], MicrographNames[p], MicrographHashes[p]), NParticles);
                    foreach (var particle in ParticlesNew)
                    {
                        particle.ResampleCoordinates(ResMov);
                        particle.ResampleAngles(ResRot);
                    }

                    #endregion
                }
                else
                {
                    ParticlesMatched = 0;
                    ParticlesUnmatched = 0;
                }

                #region Match particles between old and new set, compute histogram

                if (ParticlesNew != null && ParticlesNew.Length > 0)
                {
                    bool[] MatchFoundOld = new bool[ParticlesOld.Length];
                    bool[] MatchFoundNew = new bool[ParticlesNew.Length];
                    ClosestDistanceOld = Helper.ArrayOfConstant(-1f, ParticlesOld.Length);
                    ClosestDistanceNew = Helper.ArrayOfConstant(-1f, ParticlesNew.Length);

                    Dictionary<string, (List<Particle>, List<int>)> GroupedOld = new Dictionary<string, (List<Particle>, List<int>)>();
                    Dictionary<string, (List<Particle>, List<int>)> GroupedNew = new Dictionary<string, (List<Particle>, List<int>)>();

                    for (int i = 0; i < ParticlesOld.Length; i++)
                    {
                        string Hash = ParticlesOld[i].SourceHash;
                        if (!GroupedOld.ContainsKey(Hash))
                            GroupedOld.Add(Hash, (new List<Particle>(), new List<int>()));
                        GroupedOld[Hash].Item1.Add(ParticlesOld[i]);
                        GroupedOld[Hash].Item2.Add(i);
                    }

                    for (int i = 0; i < ParticlesNew.Length; i++)
                    {
                        string Hash = ParticlesNew[i].SourceHash;
                        if (!GroupedNew.ContainsKey(Hash))
                            GroupedNew.Add(Hash, (new List<Particle>(), new List<int>()));
                        GroupedNew[Hash].Item1.Add(ParticlesNew[i]);
                        GroupedNew[Hash].Item2.Add(i);
                    }

                    Parallel.ForEach(GroupedNew, pair =>
                    {
                        if (!GroupedOld.ContainsKey(pair.Key))
                            return;

                        List<Particle> Old = GroupedOld[pair.Key].Item1;
                        List<int> OldIDs = GroupedOld[pair.Key].Item2;
                        List<Particle> New = pair.Value.Item1;
                        List<int> NewIDs = pair.Value.Item2;

                        float[][] DistanceMatrix = new float[Old.Count][];

                        for (int i1 = 0; i1 < Old.Count; i1++)
                        {
                            float BestDistance = float.MaxValue;
                            int BestID = -1;
                            float3 P1 = Old[i1].Coordinates[0];

                            DistanceMatrix[i1] = new float[New.Count];

                            for (int i2 = 0; i2 < New.Count; i2++)
                            {
                                float3 P2 = New[i2].Coordinates[0];
                                float Distance2 = (P2 - P1).Length();

                                DistanceMatrix[i1][i2] = Distance2;

                                //if (Distance2 < BestDistance)
                                //{
                                //    BestDistance = Distance2;
                                //    BestID = OldIDs[i2];
                                //}
                            }

                            //BestDistance = (float)Math.Sqrt(BestDistance);

                            //if (BestID >= 0)
                            //{
                            //    ClosestDistanceOld[BestID] = BestDistance;
                            //    ClosestDistanceNew[NewIDs[i1]] = BestDistance;
                            //}
                        }

                        HungarianAlgorithm Bla = new HungarianAlgorithm(DistanceMatrix);
                        int[] Matching = Bla.execute();

                        for (int i1 = 0; i1 < Matching.Length; i1++)
                        {
                            if (Matching[i1] < 0)
                                continue;

                            int i2 = Matching[i1];
                            ClosestDistanceOld[OldIDs[i1]] = DistanceMatrix[i1][i2];
                            ClosestDistanceNew[NewIDs[i2]] = DistanceMatrix[i1][i2];
                        }
                    });

                    IEnumerable<float> ValidDistances = ClosestDistanceNew.Where(v => v >= 0);
                    MaxDistance = Math.Max(1, MathHelper.Max(ValidDistances));

                    float[] HistogramBins = new float[NBins];
                    foreach (var d in ValidDistances)
                        HistogramBins[(int)Math.Round(d / MaxDistance * (NBins - 1))]++;

                    double HistogramWidth = 400;
                    //Dispatcher.Invoke(() => HistogramWidth = PanelDistanceOptions.ActualWidth - 1);

                    float MaxBin = Math.Max(1, MathHelper.Max(HistogramBins));
                    HistogramBins = HistogramBins.Select(v => v / MaxBin).ToArray();
                    
                    PauseSetUpdates = true;

                    Dispatcher.Invoke(() =>
                    {
                        PointCollection HistogramPoints = new PointCollection();

                        HistogramPoints.Add(new Point(0, 30));
                        for (int i = 0; i < NBins; i++)
                        {
                            double X = (double)i / (NBins - 1) * HistogramWidth;
                            double Y = (1 - HistogramBins[i]) * 30;
                            HistogramPoints.Add(new Point(X, Y));
                        }
                        HistogramPoints.Add(new Point(HistogramWidth, 30));

                        SliderToleranceDistance.MaxValue = (decimal)MaxDistance;
                        ToleranceDistance = Math.Min((decimal)MaxDistance, ToleranceDistance);

                        PolygonHistogramGreen.Points = HistogramPoints;
                        PolygonHistogramGray.Points = HistogramPoints;

                        TextMaxDistance.Text = $"{MaxDistance:F1} Å";
                    });

                    PauseSetUpdates = false;
                }

                #endregion
            });

            ProgressParticles.Visibility = Visibility.Hidden;

            TextParticlesResult.Text = $"{ParticlesMatched}/{ParticlesMatched + ParticlesUnmatched} new particles matched to available data sources";

            UpdateSets();

            Revalidate();
        }

        void UpdateSets()
        {
            if (PauseSetUpdates)
                return;

            PanelDistanceOptions.Visibility = Visibility.Collapsed;
                        
            if (ParticlesOld == null || ParticlesNew == null || 
                ClosestDistanceOld == null || ClosestDistanceNew == null)
                return;

            PanelDistanceOptions.Visibility = Visibility.Visible;

            IEnumerable<float> ValidDistances = ClosestDistanceNew.Where(v => v >= 0 && v <= (float)ToleranceDistance + 1e-6f);

            double FractionGreen = Math.Max(0, Math.Min(1, (float)ToleranceDistance / MaxDistance));
            
            double HistogramWidth = 400;

            ClipHistogramGreen.Rect = new Rect(new Size(HistogramWidth * FractionGreen, 30));
            ClipHistogramGray.Rect = new Rect(new Point(HistogramWidth * FractionGreen, 0), new Size(HistogramWidth * (1 - FractionGreen), 30));

            TextExclusiveOld.Text = (ParticlesOld.Length - ValidDistances.Count()).ToString();
            TextIntersected.Text = (ValidDistances.Count()).ToString();
            TextExclusiveNew.Text = (ParticlesNew.Length - ValidDistances.Count()).ToString();

            List<Particle> ParticlesFinalList = new List<Particle>(ParticlesOld.Length);
            bool IncludeOld = (bool)CheckSetOld.IsChecked;
            bool IncludeIntersection = (bool)CheckSetBoth.IsChecked;
            bool IncludeNew = (bool)CheckSetNew.IsChecked;

            for (int i = 0; i < ParticlesOld.Length; i++)
                if (((ClosestDistanceOld[i] < 0 || ClosestDistanceOld[i] > (float)ToleranceDistance + 1e-6f) && IncludeOld) || 
                    (ClosestDistanceOld[i] >= 0 && ClosestDistanceOld[i] <= (float)ToleranceDistance + 1e-6f && IncludeIntersection))
                    ParticlesFinalList.Add(ParticlesOld[i]);

            for (int i = 0; i < ParticlesNew.Length; i++)
                if ((ClosestDistanceNew[i] < 0 || ClosestDistanceNew[i] > (float)ToleranceDistance + 1e-6f) && IncludeNew)
                    ParticlesFinalList.Add(ParticlesNew[i]);

            ParticlesFinalList.Sort((a, b) => a.SourceHash.CompareTo(b.SourceHash));
            ParticlesFinal = ParticlesFinalList.ToArray();
        }

        bool ValidateParticles()
        {
            return (((bool)RadioParticlesRelion.IsChecked) && string.IsNullOrEmpty(TextParticlesError.Text) && ParticlesMatched > 0) ||
                   (((bool)RadioParticlesWarp.IsChecked) && TableWarp != null && TableWarp.RowCount > 0);
        }

        bool ValidateSets()
        {
            if (ParticlesOld == null || ParticlesNew == null || ClosestDistanceNew == null || ParticlesFinal == null)
                return false;

            return ParticlesFinal.Length > 0;
        }

        private void CheckSet_Checked(object sender, RoutedEventArgs e)
        {
            UpdateSets();
            Revalidate();
        }

        private async void ButtonParticlesWarpPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR file|*.star",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                TableWarp = new Star(Dialog.FileName);

                if (!TableWarp.HasColumn("wrpCoordinateX1") ||
                    !TableWarp.HasColumn("wrpCoordinateY1") ||
                    !TableWarp.HasColumn("wrpAngleRot1") ||
                    !TableWarp.HasColumn("wrpAngleTilt1") ||
                    !TableWarp.HasColumn("wrpAnglePsi1") ||
                    !TableWarp.HasColumn("wrpSourceHash"))
                {
                    TableWarp = null;
                    await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", "Table does not contain all essential columns (coordinates, angles, source hash).");
                    return;
                }

                PathWarp = Dialog.FileName;
                ButtonParticlesWarpPath.Content = Helper.ShortenString(PathWarp, 55);

                UpdateParticles();
            }
        }

        private async void ButtonParticlesRelionPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR file|*_data.star",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                bool Is3;
                (TableRelion, Is3) = Star.LoadRelion3Particles(Dialog.FileName);

                if (!TableRelion.HasColumn("rlnCoordinateX") ||
                    !TableRelion.HasColumn("rlnCoordinateY") ||
                    !TableRelion.HasColumn("rlnAngleRot") ||
                    !TableRelion.HasColumn("rlnAngleTilt") ||
                    !TableRelion.HasColumn("rlnAnglePsi") ||
                    !TableRelion.HasColumn("rlnMicrographName"))
                {
                    TableRelion = null;
                    await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie", "Table does not contain all essential columns (coordinates, angles, micrograph name).");
                    return;
                }

                if (TableRelion.HasColumn("rlnDetectorPixelSize") && TableRelion.HasColumn("rlnMagnification"))
                {
                    try
                    {
                        decimal DetectorPixel = decimal.Parse(TableRelion.GetRowValue(0, "rlnDetectorPixelSize"), CultureInfo.InvariantCulture) * 1e4M;
                        decimal Mag = decimal.Parse(TableRelion.GetRowValue(0, "rlnMagnification"), CultureInfo.InvariantCulture);

                        PauseParticleUpdates = true;
                        {
                            ParticleCoordinatesPixel = DetectorPixel / Mag;
                            if (!Is3)
                            ParticleShiftsPixel = DetectorPixel / Mag;
                        }
                        PauseParticleUpdates = false;
                    }
                    catch { }
                }
                else if (TableRelion.HasColumn("rlnImagePixelSize"))
                {
                    decimal PixelSize = decimal.Parse(TableRelion.GetRowValue(0, "rlnImagePixelSize"), CultureInfo.InvariantCulture);
                    ParticleCoordinatesPixel = PixelSize;
                    ParticleShiftsPixel = 1;
                }

                int NameIndex = TableRelion.GetColumnID("rlnMicrographName");
                for (int r = 0; r < TableRelion.RowCount; r++)
                    TableRelion.SetRowValue(r, NameIndex, Helper.PathToNameWithExtension(TableRelion.GetRowValue(r, NameIndex)));

                PathRelion = Dialog.FileName;
                ButtonParticlesRelionPath.Content = Helper.ShortenString(PathRelion, 55);

                UpdateParticles();
            }
        }

        private void RadioParticlesWarp_OnChecked(object sender, RoutedEventArgs e)
        {
            UpdateParticles();
        }
    }
}
