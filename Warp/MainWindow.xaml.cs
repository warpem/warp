using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using ControlzEx.Standard;
using ControlzEx.Theming;
using LiveCharts;
using LiveCharts.Defaults;
using MahApps.Metro.Controls.Dialogs;
using Microsoft.AspNetCore.Hosting;
using Warp.Controls;
using Warp.Controls.TaskDialogs.Tomo;
using Warp.Controls.TaskDialogs.TwoD;
using Warp.Headers;
using Warp.Tools;
using Path = System.IO.Path;

namespace Warp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : MahApps.Metro.Controls.MetroWindow
    {
        private const string DefaultGlobalOptionsName = "global.settings";
        public static GlobalOptions GlobalOptions = new GlobalOptions();

        #region MAIN WINDOW

        private CheckBox[] CheckboxesGPUStats;
        private int[] BaselinesGPUStats;
        private DispatcherTimer TimerGPUStats;
        private DispatcherTimer TimerCheckUpdates;

        private IWebHost RESTHost;

        public MainWindow()
        {
            WarpRuntime.MainWindow = this;
            Options = new OptionsWarp();

            #region Make sure everything is OK with GPUs
            try
            {
                if (GPU.GetDeviceCount() <= 0)
                    throw new Exception();
            }
            catch (Exception exc)
            {
                MessageBox.Show("No CUDA devices found, or couldn't load NativeAcceleration.dll due to missing dependencies, shutting down.\n\n" +
                                "First things to check:\n" +
                                "-At least one GPU with Maxwell (GeForce 9xx, Quadro Mxxxx, Tesla Mxx) or later architecture available?\n" +
                                "-Latest GPU driver installed?\n" +
                                "-VC++ 2022 redistributable installed?\n" +
                                "-Any bundled libraries missing? (reinstall Warp to be sure)\n" +
                                "\n" +
                                "If none of this works, please report the issue in https://groups.google.com/forum/#!forum/warp-em");
                Close();
            }

            GPU.SetDevice(0);
            #endregion

            DataContext = Options;

            InitializeComponent();

            PlotStatsAstigmatism.DataContext = this;
            PlotStatsDefocus.DataContext = this;
            PlotStatsPhase.DataContext = this;
            PlotStatsResolution.DataContext = this;
            PlotStatsMotion.DataContext = this;
            PlotStatsParticles.DataContext = this;
            PlotStatsMaskPercentage.DataContext = this;

            ProcessingStatusBar.DataContext = this;
            CTFDisplayControl.DataContext = this;
            MicrographDisplayControl.DataContext = this;

            #region Options events

            Options.PropertyChanged += Options_PropertyChanged;
            Options.Import.PropertyChanged += OptionsImport_PropertyChanged;
            Options.CTF.PropertyChanged += OptionsCTF_PropertyChanged;
            Options.Movement.PropertyChanged += OptionsMovement_PropertyChanged;
            Options.Grids.PropertyChanged += OptionsGrids_PropertyChanged;
            Options.Tomo.PropertyChanged += OptionsTomo_PropertyChanged;
            Options.Picking.PropertyChanged += OptionsPicking_PropertyChanged;
            Options.Export.PropertyChanged += OptionsExport_PropertyChanged;
            Options.Tasks.PropertyChanged += OptionsTasks_PropertyChanged;
            Options.Filter.PropertyChanged += OptionsFilter_PropertyChanged;

            #endregion

            Closing += MainWindow_Closing;

            #region GPU statistics

            CheckboxesGPUStats = Helper.ArrayOfFunction(i =>
                                                        {
                                                            CheckBox NewCheckBox = new CheckBox
                                                            {
                                                                Foreground = Brushes.Black,
                                                                Margin = new Thickness(10, 0, 10, 0),
                                                                IsChecked = true,
                                                                Opacity = 1.0,
                                                                Focusable = false
                                                            };
                                                            NewCheckBox.MouseEnter += (a, b) => NewCheckBox.Opacity = 1.0;
                                                            NewCheckBox.MouseLeave += (a, b) => NewCheckBox.Opacity = 1.0;

                                                            return NewCheckBox;
                                                        },
                                                        GPU.GetDeviceCount());
            foreach (var checkBox in CheckboxesGPUStats)
                PanelGPUStats.Children.Add(checkBox);
            BaselinesGPUStats = Helper.ArrayOfFunction(i => (int)GPU.GetFreeMemory(i), GPU.GetDeviceCount());

            TimerGPUStats = new DispatcherTimer(new TimeSpan(0, 0, 0, 0, 200), DispatcherPriority.Background, (a, b) =>
            {
                for (int i = 0; i < CheckboxesGPUStats.Length; i++)
                {
                    int CurrentMemory = (int)GPU.GetFreeMemory(i);
                    IntPtr NamePtr = GPU.GetDeviceName(i);
                    string Name = Marshal.PtrToStringAnsi(NamePtr);
                    CPU.HostFree(NamePtr);
                    CheckboxesGPUStats[i].Content = $"#{i}, {Name}: {((float)CurrentMemory / 1024).ToString("f1")} GB";
                }
            }, Dispatcher);

            #endregion

            #region Control set definitions

            DisableWhenPreprocessing = new List<UIElement>
            {
                GridOptionsIO,
                GridOptionsIOTomo,
                GridOptionsPreprocessing,
                GridOptionsCTF,
                GridOptionsMovement,
                GridOptionsGrids,
                GridOptionsPicking,
                GridOptionsPostprocessing,
                ButtonOptionsSave,
                ButtonOptionsLoad,
                ButtonOptionsAdopt,
                ButtonProcessOneItemCTF,
                ButtonProcessOneItemTiltHandedness,
                SwitchProcessCTF,
                SwitchProcessMovement,
                SwitchProcessPicking,
                PanelOverviewTasks2D,
                PanelOverviewTasks3D
            };
            DisableWhenPreprocessing.AddRange(CheckboxesGPUStats);

            HideWhen2D = new List<UIElement>
            {
                GridOptionsIOTomo,
                PanelOverviewTasks3D,
                ButtonProcessOneItemTiltHandedness
            };
            foreach (var element in HideWhen2D)
                element.Visibility = Visibility.Collapsed;

            HideWhenTomo = new List<UIElement>
            {
                ButtonTasksAdjustDefocus,
                ButtonTasksExportParticles,
                //CheckCTFDoIce,
                PanelProcessMovement,
                GridOptionsPicking,
                GridMovement,
                GridOptionsMovement,
                LabelModelsHeader,
                GridOptionsGrids,
                GridOptionsPicking,
                PanelProcessPicking,
                LabelOutputHeader,
                GridOptionsPostprocessing,
                GridOptionsIO2D,
                PanelOverviewTasks2D
            };

            HideWhenNoActiveItem = new List<UIElement>
            {
                ButtonProcessOneItemCTF
            };

            #endregion

            #region File discoverer

            FileDiscoverer = new FileDiscoverer();
            FileDiscoverer.FilesChanged += FileDiscoverer_FilesChanged;
            FileDiscoverer.IncubationStarted += FileDiscoverer_IncubationStarted;
            FileDiscoverer.IncubationEnded += FileDiscoverer_IncubationEnded;

            #endregion

            ProcessingStatusBar.ActiveItemStatusChanged += UpdateStatsStatus;

            #region Load settings

            // Load settings from previous session
            if (File.Exists(DefaultOptionsName))
                Options.Load(DefaultOptionsName);

            OptionsAutoSave = true;
            OptionsLookForFolderOptions = true;

            if (File.Exists(DefaultGlobalOptionsName))
                GlobalOptions.Load(DefaultGlobalOptionsName);

            UpdateStatsAll();

            #endregion

            #region Disable devices if needed



            #endregion

            #region Show prompt on first run

            //if (GlobalOptions.ShowTiffReminder)
            //    Dispatcher.InvokeAsync(async () =>
            //    {
            //        var DialogResult = await this.ShowMessageAsync("Careful there!",
            //                                                       "As of v1.0.6, Warp handles TIFF files differently. Find out more at http://www.warpem.com/warp/?page_id=361.\n" +
            //                                                       "Go there now?",
            //                                                       MessageDialogStyle.AffirmativeAndNegative,
            //                                                       new MetroDialogSettings
            //                                                       {
            //                                                           AffirmativeButtonText = "Yes",
            //                                                           NegativeButtonText = "No"
            //                                                       });
            //        if (DialogResult == MessageDialogResult.Affirmative)
            //        {
            //            Process.Start("http://www.warpem.com/warp/?page_id=361");
            //        }
            //        else
            //        {
            //            GlobalOptions.ShowTiffReminder = false;
            //            GlobalOptions.Save(DefaultGlobalOptionsName);
            //        }
            //    }, DispatcherPriority.ApplicationIdle);

            //if (!GlobalOptions.PromptShown)
            //    Dispatcher.InvokeAsync(async () =>
            //    {
            //        CustomDialog Dialog = new CustomDialog();
            //        Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            //        FirstRunPrompt DialogContent = new FirstRunPrompt();
            //        DialogContent.Close += () =>
            //        {
            //            this.HideMetroDialogAsync(Dialog);
            //            GlobalOptions.PromptShown = true;
            //            GlobalOptions.AllowCollection = (bool)DialogContent.CheckAgree.IsChecked;

            //            GlobalOptions.Save(DefaultGlobalOptionsName);

            //            GlobalOptions.LogEnvironment();
            //        };
            //        Dialog.Content = DialogContent;

            //        this.ShowMetroDialogAsync(Dialog);
            //    }, DispatcherPriority.ApplicationIdle);

            #endregion

            #region Start listening to Web API calls if desired

            if (GlobalOptions.APIPort > 0)
            {
                try
                {
                    RESTHost = new WebHostBuilder()
                                   .UseContentRoot(Directory.GetCurrentDirectory())
                                   .UseKestrel()
                                   .UseStartup<Startup>()
                                   //.UseUrls($"http://localhost:{GlobalOptions.APIPort}")
                                   .Build();

                    RESTHost.RunAsync();
                }
                catch (Exception exc)
                {
                    this.ShowMessageAsync("Oops, there was a problem starting the web API service",
                                            exc.ToString(),
                                            MessageDialogStyle.Affirmative);
                }
            }

            #endregion

            #region Test stuff

            {
                //WorkerWrapper Worker = new WorkerWrapper(1);
                //Worker.Dispose();
            }

            #endregion
        }

        private void MainWindow_Closing(object sender, CancelEventArgs e)
        {
            try
            {
                RESTHost?.StopAsync();
                SaveDefaultSettings();
                FileDiscoverer.Shutdown();
            }
            catch (Exception)
            {
                // ignored
            }
        }
        
        private void ButtonUpdateAvailable_OnClick(object sender, RoutedEventArgs e)
        {
            Process.Start("http://www.warpem.com/warp/?page_id=65");
        }

        private void SwitchDayNight_OnClick(object sender, RoutedEventArgs e)
        {
            if (ThemeManager.Current.DetectTheme(Application.Current).Name == "Light.Cyan")
            {
                ThemeManager.Current.ChangeTheme(Application.Current, "Dark.Cyan");

                this.GlowBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#304160"));
                //this.WindowTitleBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#304160"));

                SwitchDayNight.Content = "🦇";
            }
            else
            {
                ThemeManager.Current.ChangeTheme(Application.Current, "Light.Cyan");

                this.GlowBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#41b1e1"));
                //this.WindowTitleBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#41b1e1"));

                SwitchDayNight.Content = "🔆";
            }

            GridCTF.UpdateLines();
            GridMovement.UpdateLines();
        }

        #region Hot keys

        public ActionCommand HotKeyLeft
        {
            get
            {
                return new ActionCommand(() =>
                {
                    if (TabProcessingCTF.IsSelected || TabProcessingMovement.IsSelected || TabProcessingCTFAndMovement.IsSelected)
                        ProcessingStatusBar.MoveToOtherItem(-1);
                });
            }
        }

        public ActionCommand HotKeyRight
        {
            get
            {
                return new ActionCommand(() =>
                {
                    if (TabProcessingCTF.IsSelected || TabProcessingMovement.IsSelected || TabProcessingCTFAndMovement.IsSelected)
                        ProcessingStatusBar.MoveToOtherItem(1);
                });
            }
        }

        public ActionCommand HotKeyW
        {
            get
            {
                return new ActionCommand(() =>
                {
                    TabProcessingOverview.IsSelected = true;
                });
            }
        }

        public ActionCommand HotKeyE
        {
            get
            {
                return new ActionCommand(() =>
                {
                    if (IsPreprocessingCollapsed)
                        TabProcessingCTFAndMovement.IsSelected = true;
                    else
                        TabProcessingCTF.IsSelected = true;
                });
            }
        }

        public ActionCommand HotKeyR
        {
            get 
            {
                return new ActionCommand(() =>
                {
                    if (IsPreprocessingCollapsed)
                        TabProcessingCTFAndMovement.IsSelected = true;
                    else
                        TabProcessingMovement.IsSelected = true;
                });
            }
        }

        public ActionCommand HotKeyF
        {
            get
            {
                return new ActionCommand(() =>
                {
                    if (TabProcessingCTF.IsSelected || TabProcessingMovement.IsSelected || TabProcessingCTFAndMovement.IsSelected)
                        if (DisplayedMovie != null)
                        {
                            if (DisplayedMovie.UnselectManual == null || !(bool)DisplayedMovie.UnselectManual)
                                DisplayedMovie.UnselectManual = true;
                            else
                                DisplayedMovie.UnselectManual = false;
                            //ProcessingStatusBar.UpdateElements();
                        }
                });
            }
        }

        #endregion

        #endregion

        #region Options

        #region Helper variables

        const string DefaultOptionsName = "previous.settings";


        private OptionsWarp _Options;
        public OptionsWarp Options
        {
            get 
            { 
                return _Options; 
            }
            set
            {
                _Options = value;
                SetValue(OptionsProperty, value);
            }
        }
        public static readonly DependencyProperty OptionsProperty = DependencyProperty.Register("Options", typeof(OptionsWarp), typeof(MainWindow), new PropertyMetadata(null));

        bool OptionsAutoSave = false;
        public bool OptionsLookForFolderOptions = false;

        public Movie DisplayedMovie
        {
            get { return (Movie)GetValue(DisplayedMovieProperty); }
            set { SetValue(DisplayedMovieProperty, value); }
        }
        public static readonly DependencyProperty DisplayedMovieProperty = DependencyProperty.Register("DisplayedMovie", typeof(Movie), typeof(MainWindow), new PropertyMetadata(null));

        public int OverviewPlotHighlightID
        {
            get { return (int)GetValue(OverviewPlotHighlightIDProperty); }
            set { SetValue(OverviewPlotHighlightIDProperty, value); }
        }
        public static readonly DependencyProperty OverviewPlotHighlightIDProperty = DependencyProperty.Register("OverviewPlotHighlightID", typeof(int), typeof(MainWindow), new PropertyMetadata(-1));


        #endregion

        private async void Options_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (new[] { "Import.DataFolder", "Import.ProcessingFolder", "Import.DoRecursiveSearch" }.Any(v => e.PropertyName == v))
            {
                if (!string.IsNullOrEmpty(Options.Import.ProcessingFolder) && !IOHelper.CheckFolderPermission(Options.Import.ProcessingFolder))
                {
                    Options.Import.DataFolder = "";
                    return;
                }

                ButtonInputPathText.Text = Options.Import.DataFolder == "" ? "Select folder..." : Options.Import.DataFolder;
                ButtonInputPathText.ToolTip = Options.Import.DataFolder == "" ? "" : Options.Import.DataFolder;

                ButtonInputProcessingPathText.Text = Options.Import.ProcessingFolder == "" ? "Select folder..." : Options.Import.ProcessingFolder;
                ButtonInputProcessingPathText.ToolTip = Options.Import.ProcessingFolder == "" ? "" : Options.Import.ProcessingFolder;

                if (OptionsLookForFolderOptions)
                {
                    OptionsLookForFolderOptions = false;

                    if (File.Exists(Path.Combine(Options.Import.ProcessingOrDataFolder, DefaultOptionsName)))
                    {
                        var MessageResult = await this.ShowMessageAsync("Options File Found in Folder",
                                                                        "A file with options from a previous Warp session was found in this folder. Load it?",
                                                                        MessageDialogStyle.AffirmativeAndNegative,
                                                                        new MetroDialogSettings
                                                                        {
                                                                            AffirmativeButtonText = "Yes",
                                                                            NegativeButtonText = "No"
                                                                        });

                        if (MessageResult == MessageDialogResult.Affirmative)
                        {
                            if (string.IsNullOrEmpty(Options.Import.ProcessingFolder))
                            {
                                string SelectedFolder = Options.Import.DataFolder;

                                Options.Load(Path.Combine(Options.Import.DataFolder, DefaultOptionsName));

                                Options.Import.DataFolder = SelectedFolder;
                            }
                            else
                            {
                                string SelectedFolder = Options.Import.ProcessingFolder;

                                Options.Load(Path.Combine(Options.Import.ProcessingFolder, DefaultOptionsName));

                                Options.Import.ProcessingFolder = SelectedFolder;
                            }
                        }
                    }

                    OptionsLookForFolderOptions = true;
                }

                AdjustInput();
                TomoAdjustInterface();
            }
            else if (e.PropertyName == "Import.Extension")
            {
                AdjustInput();
                TomoAdjustInterface();
            }
            else if (e.PropertyName == "Import.GainPath")
            {
                if (!File.Exists(Options.Import.GainPath))
                {
                    Options.Import.GainPath = "";
                    return;
                }

                Options.Import.GainReferenceHash = MathHelper.GetSHA1(Options.Import.GainPath, 1 << 20);
                ButtonGainPathText.Text = Options.Import.GainPath == "" ? "Select gain reference..." : Options.Import.GainPath;
                ButtonGainPathText.ToolTip = Options.Import.GainPath == "" ? null : Options.Import.GainPath;
            }
            else if (e.PropertyName == "Import.DefectsPath")
            {
                if (!File.Exists(Options.Import.DefectsPath))
                {
                    Options.Import.DefectsPath = "";
                    return;
                }

                Options.Import.DefectMapHash = MathHelper.GetSHA1(Options.Import.DefectsPath, 1 << 20);
                ButtonDefectsPathText.Text = Options.Import.DefectsPath == "" ? "Select defect map..." : Options.Import.DefectsPath;
                ButtonDefectsPathText.ToolTip = Options.Import.DefectsPath == "" ? null : Options.Import.DefectsPath;
            }
            else if (e.PropertyName == "CTF.Window")
            {
                CTFDisplayControl.Width = CTFDisplayControl.Height = Math.Min(1024, Options.CTF.Window);
            }
            else if (e.PropertyName == "CTF.DoPhase")
            {
                UpdateFilterRanges();
            }
            else if (e.PropertyName == "Import.PixelSize")
            {
                UpdateFilterRanges();

                if (OptionsAutoSave)
                {
                    Options.Tasks.TomoFullReconstructPixel = Math.Max(Options.Import.PixelSize, Options.Tasks.TomoFullReconstructPixel);
                    Options.Tasks.TomoSubReconstructPixel = Math.Max(Options.Import.PixelSize, Options.Tasks.TomoSubReconstructPixel);
                }
            }
            else if (e.PropertyName == "ProcessCTF")
            {
                if (!Options.ProcessCTF)
                {
                    GridCTF.Opacity = 0.5;
                    PanelGridCTFParams.Opacity = 0.5;
                }
                else
                {
                    GridCTF.Opacity = 1;
                    PanelGridCTFParams.Opacity = 1;
                }

                UpdateFilterRanges();
                SaveDefaultSettings();
            }
            else if (e.PropertyName == "ProcessMovement")
            {
                if (!Options.ProcessMovement)
                {
                    GridMovement.Opacity = 0.5;
                    PanelGridMovementParams.Opacity = 0.5;
                }
                else
                {
                    GridMovement.Opacity = 1;
                    PanelGridMovementParams.Opacity = 1;
                }

                UpdateFilterRanges();
                SaveDefaultSettings();
            }
            else if (e.PropertyName == "Runtime.DisplayedMovie")
            {
                foreach (var element in HideWhenNoActiveItem)
                    element.Visibility = DisplayedMovie == null ? Visibility.Collapsed : Visibility.Visible;
            }
            else if (e.PropertyName == "Picking.ModelPath")
            {
                if (string.IsNullOrEmpty(LocatePickingModel(Options.Picking.ModelPath)))
                {
                    Options.Picking.ModelPath = "";
                    //return;
                }
                ButtonPickingModelNameText.Text = Options.Picking.ModelPath == "" ? "Select BoxNet model..." : Options.Picking.ModelPath;
                MicrographDisplayControl.UpdateBoxNetName(Options.Picking.ModelPath);
            }

            if (OptionsAutoSave && !e.PropertyName.StartsWith("Tasks"))
            {
                Dispatcher.Invoke(() =>
                {
                    ProcessingStatusBar.UpdateElements();
                    UpdateStatsStatus();
                    UpdateButtonOptionsAdopt();
                });
            }
        }

        private void OptionsImport_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsCTF_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsMovement_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsGrids_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsTomo_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsPicking_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsExport_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsTasks_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsFilter_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            UpdateFilterRanges();
            UpdateFilterResult();
            UpdateStatsStatus();
            SaveDefaultSettings();
        }

        private void OptionsAdvanced_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsRuntime_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == "DisplayedMovie")
            {
                UpdateButtonOptionsAdopt();
            }
        }

        private void SaveDefaultSettings()
        {
            if (OptionsAutoSave)
            {
                try
                {
                    Options.Save(DefaultOptionsName);
                    GlobalOptions.Save(DefaultGlobalOptionsName);
                } catch { }

                if (Options.Import.ProcessingOrDataFolder != "")
                    try
                    {
                        Options.Save(Options.Import.ProcessingOrDataFolder + DefaultOptionsName);
                    } catch { }
            }
        }

        #endregion

        #region File Discoverer

        public readonly FileDiscoverer FileDiscoverer;

        private void FileDiscoverer_FilesChanged()
        {
            Movie[] ImmutableItems = null;
            Helper.Time("FileDiscoverer.GetImmutableFiles", () => ImmutableItems = FileDiscoverer.GetImmutableFiles());

            Dispatcher.InvokeAsync(() =>
            {
                ProcessingStatusBar.Items = new ObservableCollection<Movie>(ImmutableItems);
                if (DisplayedMovie == null && ImmutableItems.Length > 0)
                    DisplayedMovie = ImmutableItems[0];
            });

            Helper.Time("FileDiscoverer.UpdateStatsAll", () => UpdateStatsAll());
        }

        private void FileDiscoverer_IncubationStarted()
        {
            
        }

        private void FileDiscoverer_IncubationEnded()
        {
            
        }

        #endregion

        #region TAB: RAW DATA

        #region Helper variables

        public static bool IsPreprocessing = false;
        public static bool IsStoppingPreprocessing = false;
        static Task PreprocessingTask = null;

        bool IsPreprocessingCollapsed = false;
        int PreprocessingWidth = 450;

        readonly List<UIElement> DisableWhenPreprocessing;
        readonly List<UIElement> HideWhen2D, HideWhenTomo;
        readonly List<UIElement> HideWhenNoActiveItem;

        #endregion

        #region Left menu panel

        private void ButtonInputPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                SelectedPath = Options.Import.DataFolder
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                //if (string.IsNullOrEmpty(Options.Import.ProcessingFolder) && !IOHelper.CheckFolderPermission(Dialog.SelectedPath))
                //{
                //    MessageBox.Show("Don't have permission to access the selected folder.");
                //    return;
                //}

                if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                    Dialog.SelectedPath += '\\';

                OptionsAutoSave = false;
                Options.Import.DataFolder = Dialog.SelectedPath;
                OptionsAutoSave = true;
            }
        }

        private void ButtonInputRemoveProcessingPath_OnClick(object sender, RoutedEventArgs e)
        {
            OptionsAutoSave = false;

            Options.Import.ProcessingFolder = "";
            Options.Import.DoRecursiveSearch = false;

            OptionsAutoSave = true;
        }

        private void ButtonInputProcessingPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                SelectedPath = Options.Import.ProcessingFolder
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                if (!IOHelper.CheckFolderPermission(Dialog.SelectedPath))
                {
                    MessageBox.Show("Don't have permission to access the selected folder.");
                    return;
                }

                if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                    Dialog.SelectedPath += '\\';

                OptionsAutoSave = false;
                Options.Import.ProcessingFolder = Dialog.SelectedPath;
                OptionsAutoSave = true;
            }
        }

        private void ButtonInputExtension_OnClick(object sender, RoutedEventArgs e)
        {
            PopupInputExtension.IsOpen = true;
        }

        private void ButtonGainPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "Image Files|*.dm4;*.mrc;*.em",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Options.Import.GainPath = Dialog.FileName;
                Options.Import.CorrectGain = true;
            }
        }

        private async void ButtonDefectPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "Image Files|*.dm4;*.mrc;*.em;*.tif;*.tiff|Rectangle list|*.txt",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                string Extension = Helper.PathToExtension(Dialog.FileName).ToLower();

                if (Extension != ".txt")
                {
                    Options.Import.DefectsPath = Dialog.FileName;
                    Options.Import.CorrectDefects = true;
                }
                else
                {
                    if (!File.Exists(Options.Import.GainPath) || !Options.Import.CorrectGain)
                    {
                        await Dispatcher.InvokeAsync(async () =>
                        {
                            await this.ShowMessageAsync("Oopsie",
                                                        "Please select and activate a gain reference before a defect map\n" +
                                                        "can be created from a rectangle list file");
                        });
                        return;
                    }

                    try
                    {
                        int3 GainDims = (MapHeader.ReadFromFile(Options.Import.GainPath)).Dimensions;
                        List<int4> Rectangles = new List<int4>();
                        using (TextReader Reader = File.OpenText(Dialog.FileName))
                        {
                            string Line;
                            while ((Line = Reader.ReadLine()) != null)
                            {
                                string[] Parts = Line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                                if (Parts.Length == 4)
                                    Rectangles.Add(new int4(int.Parse(Parts[0]),
                                                            int.Parse(Parts[1]),
                                                            int.Parse(Parts[2]),
                                                            int.Parse(Parts[3])));
                            }
                        }

                        if (Rectangles.Count == 0)
                            throw new Exception("No valid rectangle definitions could be found.");

                        Image DefectsMap = new Image(GainDims);
                        float[] DefectsData = DefectsMap.GetHost(Intent.ReadWrite)[0];
                        foreach (var rect in Rectangles)
                        {
                            for (int y = 0; y < rect.W; y++)
                            {
                                int yy = rect.Y + y;
                                for (int x = 0; x < rect.Z; x++)
                                {
                                    int xx = rect.X + x;
                                    if (xx < 0 || xx >= GainDims.X || yy < 0 || yy >= GainDims.Y)
                                        throw new Exception($"Rectangle exceeded image dimensions: {xx}, {yy} (0-based)");

                                    DefectsData[yy * GainDims.X + xx] = 1;
                                }
                            }
                        }

                        Directory.CreateDirectory(Path.Combine(Options.Import.ProcessingOrDataFolder, "defectmap"));
                        DefectsMap.WriteTIFF(Path.Combine(Options.Import.ProcessingOrDataFolder, "defectmap", "defects.tif"), 1, typeof(float));
                        DefectsMap.Dispose();

                        Options.Import.DefectsPath = Path.Combine(Options.Import.ProcessingOrDataFolder, "defectmap", "defects.tif");
                        Options.Import.CorrectDefects = true;
                    }
                    catch (Exception exc)
                    {
                        await Dispatcher.InvokeAsync(async () =>
                        {
                            await this.ShowMessageAsync("Oopsie",
                                                        "Couldn't create defect map because:\n" +
                                                        exc.ToString());
                        });
                        return;
                    }


                }
            }
        }

        private void ButtonOptionsSave_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.SaveFileDialog Dialog = new System.Windows.Forms.SaveFileDialog
            {
                Filter = "Setting Files|*.settings"
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();
            if (Result == System.Windows.Forms.DialogResult.OK)
            {
                Options.Save(Dialog.FileName);
            }
        }

        private void ButtonOptionsLoad_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "Setting Files|*.settings",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();
            if (Result == System.Windows.Forms.DialogResult.OK)
            {
                Options.Load(Dialog.FileName);
            }
        }

        #region Options adoption

        private void ButtonOptionsAdopt_OnClick(object sender, RoutedEventArgs e)
        {
            if (DisplayedMovie != null)
            {
                ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
                ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
                ProcessingOptionsBoxNet OptionsBoxNet = Options.GetProcessingBoxNet();
                ProcessingOptionsMovieExport OptionsExport = Options.GetProcessingMovieExport();

                ProcessingStatus Status = StatusBar.GetMovieProcessingStatus(DisplayedMovie, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, Options);

                if (Status == ProcessingStatus.Outdated)
                {
                    ButtonOptionsAdopt.Visibility = Visibility.Visible;
                    OptionsAutoSave = false;

                    if (DisplayedMovie.OptionsCTF != null)
                        Options.Adopt(DisplayedMovie.OptionsCTF);
                    if (DisplayedMovie.OptionsMovement != null)
                        Options.Adopt(DisplayedMovie.OptionsMovement);
                    if (DisplayedMovie.OptionsMovieExport != null)
                        Options.Adopt(DisplayedMovie.OptionsMovieExport);

                    OptionsAutoSave = true;
                    SaveDefaultSettings();
                    ProcessingStatusBar.UpdateElements();
                    UpdateStatsStatus();

                    bool IsConflicted = DisplayedMovie.AreOptionsConflicted();
                    if (IsConflicted)
                        this.ShowMessageAsync("Don't panic, but...", "... the input options for individual processing steps have conflicting parameters. Please reprocess the data with uniform input options.");
                }
            }

            ButtonOptionsAdopt.Visibility = Visibility.Hidden;
        }

        private void UpdateButtonOptionsAdopt()
        {
            ButtonOptionsAdopt.Visibility = Visibility.Hidden;

            if (DisplayedMovie != null)
            {
                ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
                ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
                ProcessingOptionsBoxNet OptionsBoxNet = Options.GetProcessingBoxNet();
                ProcessingOptionsMovieExport OptionsExport = Options.GetProcessingMovieExport();

                ProcessingStatus Status = StatusBar.GetMovieProcessingStatus(DisplayedMovie, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, Options);

                if (Status == ProcessingStatus.Outdated)
                    ButtonOptionsAdopt.Visibility = Visibility.Visible;
            }
        }

        #endregion

        #region Picking
        
        private void ButtonPickingModelName_OnClick(object sender, RoutedEventArgs e)
        {
            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            BoxNetSelect DialogContent = new BoxNetSelect(Options.Picking.ModelPath, Options);
            DialogContent.Close += () =>
            {
                Options.Picking.ModelPath = DialogContent.ModelName;
                this.HideMetroDialogAsync(Dialog);
            };
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        public string LocatePickingModel(string name)
        {
            if (string.IsNullOrEmpty(name))
                return null;

            name += ".pt";

            if (File.Exists(name))
            {
                return name;
            }
            else if (File.Exists(System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet3models", name)))
            {
                return System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet3models", name);
            }

            return null;
        }

        #endregion

        #endregion

        public void StartProcessing()
        {
            if (!IsPreprocessing && !IsStoppingPreprocessing)
                Dispatcher.InvokeAsync(() => ButtonStartProcessing_OnClick(null, null));
        }

        public void StopProcessing()
        {
            if (IsPreprocessing && !IsStoppingPreprocessing)
                Dispatcher.InvokeAsync(() => ButtonStartProcessing_OnClick(null, null));
        }

        private async void ButtonStartProcessing_OnClick(object sender, RoutedEventArgs e)
        {
            if (!IsPreprocessing)
            {
                foreach (var item in DisableWhenPreprocessing)
                    item.IsEnabled = false;
                MicrographDisplayControl.SetProcessingMode(true);

                ButtonStartProcessing.Content = "STOP PROCESSING";
                ButtonStartProcessing.Foreground = Brushes.Red;
                IsPreprocessing = true;

                bool IsTomo = Options.Import.ExtensionTomoSTAR;

                PreprocessingTask = Task.Run(async () =>
                {
                    int NDevices = GPU.GetDeviceCount();
                    List<int> UsedDevices = GetDeviceList();
                    List<int> UsedDeviceProcesses = Helper.Combine(Helper.ArrayOfFunction(i => UsedDevices.Select(d => d + i * NDevices).ToArray(), GlobalOptions.ProcessesPerDevice)).ToList();

                    #region Check if options are compatible

                    {
                        string ErrorMessage = "";
                    }

                    #endregion

                    #region Load gain reference if needed

                    Image ImageGain = null;
                    DefectModel DefectMap = null;
                    if (!string.IsNullOrEmpty(Options.Import.GainPath) && Options.Import.CorrectGain && File.Exists(Options.Import.GainPath))
                        try
                        {
                            ImageGain = LoadAndPrepareGainReference();
                        }
                        catch (Exception exc)
                        {
                            ImageGain?.Dispose();

                            await Dispatcher.InvokeAsync(async () =>
                            {
                                await this.ShowMessageAsync("Oopsie",
                                                            "Something went wrong when trying to load the gain reference.\n\n" +
                                                            "The exception raised is:\n" + exc);

                                ButtonStartProcessing_OnClick(sender, e);
                            });

                            return;
                        }
                    if (!string.IsNullOrEmpty(Options.Import.DefectsPath) && Options.Import.CorrectDefects && File.Exists(Options.Import.DefectsPath))
                        try
                        {
                            DefectMap = LoadAndPrepareDefectMap();

                            if (ImageGain != null && new int2(ImageGain.Dims) != DefectMap.Dims)
                                throw new Exception("Defect map and gain reference dimensions don't match.");
                        }
                        catch (Exception exc)
                        {
                            DefectMap?.Dispose();

                            await Dispatcher.InvokeAsync(async () =>
                            {
                                await this.ShowMessageAsync("Oopsie",
                                                            "Something went wrong when trying to load the defect map.\n\n" +
                                                            "The exception raised is:\n" + exc);

                                ButtonStartProcessing_OnClick(sender, e);
                            });

                            return;
                        }

                    #endregion

                    #region Load BoxNet model if needed

                    BoxNetTorch[] BoxNetworks = new BoxNetTorch[NDevices];
                    object[] BoxNetLocks = Helper.ArrayOfFunction(i => new object(), NDevices);

                    if (!IsTomo && Options.ProcessPicking)
                    {
                        ProgressDialogController ProgressDialog = null;

                        Image.FreeDeviceAll();

                        try
                        {
                            await Dispatcher.Invoke(async () => ProgressDialog = await this.ShowProgressAsync($"Loading {Options.Picking.ModelPath} model...", ""));
                            ProgressDialog.SetIndeterminate();

                            if (string.IsNullOrEmpty(Options.Picking.ModelPath) || LocatePickingModel(Options.Picking.ModelPath) == null)
                                throw new Exception("No BoxNet model selected. Please use the options panel to select a model.");

                            MicrographDisplayControl.DropBoxNetworks();
                            foreach (var d in UsedDevices)
                            {
                                BoxNetworks[d] = new BoxNetTorch(BoxNetTorch.DefaultDimensionsPredict, new float[] { 1f, 1f, 1f }, new int[] { d }, BoxNetTorch.DefaultBatchPredict);
                                BoxNetworks[d].Load(LocatePickingModel(Options.Picking.ModelPath));
                            }
                        }
                        catch (Exception exc)
                        {
                            await Dispatcher.Invoke(async () =>
                            {
                                await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                                    "There was an error loading the specified BoxNet model for picking.\n\n" +
                                                                                                    "The exception raised is:\n" + exc);

                                ButtonStartProcessing_OnClick(sender, e);
                            });

                            ImageGain?.Dispose();
                            DefectMap?.Dispose();

                            await ProgressDialog.CloseAsync();

                            return;
                        }

                        await ProgressDialog.CloseAsync();
                    }

                    #endregion
                    
                    #region Load or create STAR table for BoxNet output, if needed

                    string BoxNetSuffix = Helper.PathToName(Options.Picking.ModelPath);

                    Star TableBoxNetAll = null;
                    string PathBoxNetAll = Path.Combine(Options.Import.ProcessingOrDataFolder, "allparticles_" + BoxNetSuffix + ".star");
                    string PathBoxNetAllSubset = Path.Combine(Options.Import.ProcessingOrDataFolder, "allparticles_last" + Options.Picking.RunningWindowLength + "_" + BoxNetSuffix + ".star");
                    string PathBoxNetFiltered = Path.Combine(Options.Import.ProcessingOrDataFolder, "goodparticles_" + BoxNetSuffix + ".star");
                    string PathBoxNetFilteredSubset = Path.Combine(Options.Import.ProcessingOrDataFolder, "goodparticles_last" + Options.Picking.RunningWindowLength + "_" + BoxNetSuffix + ".star");
                    object TableBoxNetAllWriteLock = new object();
                    int TableBoxNetConcurrent = 0;

                    // Switch filter suffix to the one used in current processing
                    //if (Options.ProcessPicking)
                    //    Dispatcher.Invoke(() => Options.Filter.ParticlesSuffix = "_" + BoxNetSuffix);

                    Dictionary<Movie, List<string[]>> AllMovieParticleRows = new Dictionary<Movie, List<string[]>>();

                    if (!IsTomo && Options.ProcessPicking && Options.Picking.DoExport && !string.IsNullOrEmpty(Options.Picking.ModelPath))
                    {
                        Movie[] TempMovies = FileDiscoverer.GetImmutableFiles();

                        if (File.Exists(PathBoxNetAll))
                        {
                            ProgressDialogController ProgressDialog = null;
                            await Dispatcher.Invoke(async () => ProgressDialog = await this.ShowProgressAsync($"Loading particle metadata from previous run...", ""));
                            ProgressDialog.SetIndeterminate();

                            TableBoxNetAll = new Star(PathBoxNetAll);

                            Dictionary<string, Movie> NameMapping = new Dictionary<string, Movie>();
                            string[] ColumnMicName = TableBoxNetAll.GetColumn("rlnMicrographName");
                            for (int r = 0; r < ColumnMicName.Length; r++)
                            {
                                if (!NameMapping.ContainsKey(ColumnMicName[r]))
                                {
                                    var Movie = TempMovies.Where(m => ColumnMicName[r].Contains(m.Name));
                                    if (Movie.Count() != 1)
                                        continue;

                                    NameMapping.Add(ColumnMicName[r], Movie.First());
                                    AllMovieParticleRows.Add(Movie.First(), new List<string[]>());
                                }

                                AllMovieParticleRows[NameMapping[ColumnMicName[r]]].Add(TableBoxNetAll.GetRow(r));
                            }

                            await ProgressDialog.CloseAsync();
                        }
                        else
                        {
                            TableBoxNetAll = new Star(new string[] { });
                        }

                        #region Make sure all columns are there

                        if (!TableBoxNetAll.HasColumn("rlnCoordinateX"))
                            TableBoxNetAll.AddColumn("rlnCoordinateX", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnCoordinateY"))
                            TableBoxNetAll.AddColumn("rlnCoordinateY", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnMagnification"))
                            TableBoxNetAll.AddColumn("rlnMagnification", "10000.0");
                        else
                            TableBoxNetAll.SetColumn("rlnMagnification", Helper.ArrayOfConstant("10000.0", TableBoxNetAll.RowCount));

                        if (!TableBoxNetAll.HasColumn("rlnDetectorPixelSize"))
                            TableBoxNetAll.AddColumn("rlnDetectorPixelSize", Options.Import.BinnedPixelSize.ToString("F5", CultureInfo.InvariantCulture));
                        else
                            TableBoxNetAll.SetColumn("rlnDetectorPixelSize", Helper.ArrayOfConstant(Options.Import.BinnedPixelSize.ToString("F5", CultureInfo.InvariantCulture), TableBoxNetAll.RowCount));

                        if (!TableBoxNetAll.HasColumn("rlnVoltage"))
                            TableBoxNetAll.AddColumn("rlnVoltage", Options.CTF.Voltage.ToString("F1", CultureInfo.InvariantCulture));

                        if (!TableBoxNetAll.HasColumn("rlnSphericalAberration"))
                            TableBoxNetAll.AddColumn("rlnSphericalAberration", "2.7");

                        if (!TableBoxNetAll.HasColumn("rlnAmplitudeContrast"))
                            TableBoxNetAll.AddColumn("rlnAmplitudeContrast", "0.07");

                        if (!TableBoxNetAll.HasColumn("rlnPhaseShift"))
                            TableBoxNetAll.AddColumn("rlnPhaseShift", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnDefocusU"))
                            TableBoxNetAll.AddColumn("rlnDefocusU", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnDefocusV"))
                            TableBoxNetAll.AddColumn("rlnDefocusV", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnDefocusAngle"))
                            TableBoxNetAll.AddColumn("rlnDefocusAngle", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnCtfMaxResolution"))
                            TableBoxNetAll.AddColumn("rlnCtfMaxResolution", "999.0");

                        if (!TableBoxNetAll.HasColumn("rlnImageName"))
                            TableBoxNetAll.AddColumn("rlnImageName", "None");

                        if (!TableBoxNetAll.HasColumn("rlnMicrographName"))
                            TableBoxNetAll.AddColumn("rlnMicrographName", "None");

                        #endregion

                        #region Repair

                        var RepairMovies = TempMovies.Where(m => !AllMovieParticleRows.ContainsKey(m) && m.OptionsBoxNet != null && File.Exists(Path.Combine(m.MatchingDir, m.RootName + "_" + BoxNetSuffix + ".star"))).ToList();
                        if (RepairMovies.Count() > 0)
                        {
                            ProgressDialogController ProgressDialog = null;
                            await Dispatcher.Invoke(async () => ProgressDialog = await this.ShowProgressAsync($"Repairing particle metadata...", ""));

                            int NRepaired = 0;
                            foreach (var item in RepairMovies)
                            {
                                float2[] Positions = Star.LoadFloat2(Path.Combine(item.MatchingDir, item.RootName + "_" + BoxNetSuffix + ".star"),
                                                                     "rlnCoordinateX",
                                                                     "rlnCoordinateY");

                                float[] Defoci = new float[Positions.Length];
                                if (item.GridCTFDefocus != null)
                                    Defoci = item.GridCTFDefocus.GetInterpolated(Positions.Select(v => new float3(v.X / (item.OptionsBoxNet.Dimensions.X / (float)item.OptionsBoxNet.BinnedPixelSizeMean),
                                                                                                           v.Y / (item.OptionsBoxNet.Dimensions.Y / (float)item.OptionsBoxNet.BinnedPixelSizeMean),
                                                                                                           0.5f)).ToArray());
                                float Astigmatism = (float)item.CTF.DefocusDelta / 2;
                                float PhaseShift = item.GridCTFPhase.GetInterpolated(new float3(0.5f)) * 180;

                                List<string[]> NewRows = new List<string[]>();
                                for (int r = 0; r < Positions.Length; r++)
                                {
                                    string[] Row = Helper.ArrayOfConstant("0", TableBoxNetAll.ColumnCount);

                                    Row[TableBoxNetAll.GetColumnID("rlnMagnification")] = "10000.0";
                                    Row[TableBoxNetAll.GetColumnID("rlnDetectorPixelSize")] = item.OptionsBoxNet.BinnedPixelSizeMean.ToString("F5", CultureInfo.InvariantCulture);

                                    Row[TableBoxNetAll.GetColumnID("rlnDefocusU")] = ((Defoci[r] + Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnDefocusV")] = ((Defoci[r] - Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnDefocusAngle")] = item.CTF.DefocusAngle.ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnVoltage")] = item.CTF.Voltage.ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnSphericalAberration")] = item.CTF.Cs.ToString("F4", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnAmplitudeContrast")] = item.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnPhaseShift")] = PhaseShift.ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnCtfMaxResolution")] = item.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnCoordinateX")] = Positions[r].X.ToString("F2", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnCoordinateY")] = Positions[r].Y.ToString("F2", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnImageName")] = (r + 1).ToString("D7") + "@particles/" + item.RootName + "_" + BoxNetSuffix + ".mrcs";
                                    Row[TableBoxNetAll.GetColumnID("rlnMicrographName")] = item.Name;

                                    NewRows.Add(Row);
                                }

                                AllMovieParticleRows.Add(item, NewRows);

                                NRepaired++;
                                Dispatcher.Invoke(() => ProgressDialog.SetProgress((float)NRepaired / RepairMovies.Count));
                            }

                            await ProgressDialog.CloseAsync();
                        }

                        #endregion
                    }

                    #endregion

                    #region Spawn workers and let them load gain refs

                    WorkerWrapper[] Workers = new WorkerWrapper[GPU.GetDeviceCount() * GlobalOptions.ProcessesPerDevice];
                    Parallel.ForEach(UsedDeviceProcesses, gpuID =>
                    {
                        Workers[gpuID] = new WorkerWrapper(gpuID);
                        Workers[gpuID].SetHeaderlessParams(new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                           Options.Import.HeaderlessOffset,
                                                           Options.Import.HeaderlessType);

                        if ((!string.IsNullOrEmpty(Options.Import.GainPath) || !string.IsNullOrEmpty(Options.Import.DefectsPath)) &&
                            (Options.Import.CorrectGain || Options.Import.CorrectDefects))
                            Workers[gpuID].LoadGainRef(Options.Import.CorrectGain ? Options.Import.GainPath : "",
                                                       Options.Import.GainFlipX,
                                                       Options.Import.GainFlipY,
                                                       Options.Import.GainTranspose,
                                                       Options.Import.CorrectDefects ? Options.Import.DefectsPath : "");
                        else
                            Workers[gpuID].LoadGainRef("", false, false, false, "");
                    });

                    bool CheckedGainDims = ImageGain == null;

                    #endregion

                    while (true)
                    {
                        if (!IsPreprocessing)
                            break;

                        #region Figure out what needs preprocessing

                        Movie[] ImmutableItems = FileDiscoverer.GetImmutableFiles();
                        List<Movie> NeedProcessing = new List<Movie>();

                        ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
                        ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
                        ProcessingOptionsMovieExport OptionsExport = Options.GetProcessingMovieExport();
                        ProcessingOptionsBoxNet OptionsBoxNet = Options.GetProcessingBoxNet();

                        bool DoCTF = Options.ProcessCTF;
                        bool DoMovement = Options.ProcessMovement;
                        bool DoPicking = Options.ProcessPicking;

                        foreach (var item in ImmutableItems)
                        {
                            ProcessingStatus Status = StatusBar.GetMovieProcessingStatus(item, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, Options, false);

                            if (Status == ProcessingStatus.Outdated || Status == ProcessingStatus.Unprocessed)
                                NeedProcessing.Add(item);
                        }

                        #endregion

                        if (NeedProcessing.Count == 0)
                        {
                            await Task.Delay(20);
                            continue;
                        }

                        #region Make sure gain dims match those of first image to be processed

                        if (!CheckedGainDims)
                        {
                            string ItemPath;

                            if (NeedProcessing[0].GetType() == typeof(Movie))
                                ItemPath = NeedProcessing[0].DataPath;
                            else
                                ItemPath = Path.Combine(((TiltSeries)NeedProcessing[0]).DataOrProcessingDirectoryName, ((TiltSeries)NeedProcessing[0]).TiltMoviePaths[0]);

                            MapHeader Header = MapHeader.ReadFromFilePatient(50, 500,
                                                                             ItemPath,
                                                                             new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                                             Options.Import.HeaderlessOffset,
                                                                             ImageFormatsHelper.StringToType(Options.Import.HeaderlessType));

                            if (Helper.PathToExtension(ItemPath).ToLower() != ".eer")
                                if (Header.Dimensions.X != ImageGain.Dims.X || Header.Dimensions.Y != ImageGain.Dims.Y)
                                {
                                    ImageGain.Dispose();
                                    DefectMap?.Dispose();

                                    foreach (var worker in Workers)
                                        worker?.Dispose();

                                    await Dispatcher.InvokeAsync(async () =>
                                    {
                                        await this.ShowMessageAsync("Oopsie", "Image dimensions do not match those of the gain reference. Maybe it needs to be rotated or transposed?");

                                        ButtonStartProcessing_OnClick(sender, e);
                                    });

                                    break;
                                }

                            CheckedGainDims = true;
                        }

                        #endregion

                        Dispatcher.Invoke(() =>
                        {
                            ProcessingStatusBar.ShowProgressBar();
                            StatsProgressIndicator.Visibility = Visibility.Visible;
                        });

                        #region Perform preprocessing on all available GPUs

                        Helper.ForEachGPU(NeedProcessing, (item, gpuID) =>
                        {
                            if (!IsPreprocessing)
                                return true;    // This cancels the iterator

                            Image OriginalStack = null;

                            try
                            {
                                item.SaveMeta();

                                var TimerOverall = BenchmarkAllProcessing.Start();

                                ProcessingOptionsMovieCTF CurrentOptionsCTF = Options.GetProcessingMovieCTF();
                                ProcessingOptionsMovieMovement CurrentOptionsMovement = Options.GetProcessingMovieMovement();
                                ProcessingOptionsBoxNet CurrentOptionsBoxNet = Options.GetProcessingBoxNet();
                                ProcessingOptionsMovieExport CurrentOptionsExport = Options.GetProcessingMovieExport();

                                bool DoExport = OptionsExport.DoAverage || OptionsExport.DoStack || OptionsExport.DoDeconv || (DoPicking && !File.Exists(item.AveragePath));

                                bool NeedsNewCTF = CurrentOptionsCTF != item.OptionsCTF && DoCTF;
                                bool NeedsNewMotion = CurrentOptionsMovement != item.OptionsMovement && DoMovement;
                                bool NeedsNewPicking = DoPicking &&
                                                       (CurrentOptionsBoxNet != item.OptionsBoxNet ||
                                                        NeedsNewMotion);
                                bool NeedsNewExport = DoExport &&
                                                      (NeedsNewMotion ||
                                                       CurrentOptionsExport != item.OptionsMovieExport ||
                                                       (CurrentOptionsExport.DoDeconv && NeedsNewCTF));

                                bool NeedsMoreDenoisingExamples = !Directory.Exists(item.DenoiseTrainingDirOdd) || 
                                                                 Directory.EnumerateFiles(item.DenoiseTrainingDirOdd, "*.mrc").Count() < 256;   // Having more than 128 examples is a waste of space
                                bool DoesDenoisingExampleExist = File.Exists(item.DenoiseTrainingOddPath);
                                bool NeedsDenoisingExample = NeedsMoreDenoisingExamples || (DoesDenoisingExampleExist && (NeedsNewCTF || NeedsNewExport));
                                CurrentOptionsExport.DoDenoiseDeconv = NeedsDenoisingExample;

                                MapHeader OriginalHeader = null;
                                decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                                bool NeedStack = NeedsNewCTF ||
                                                 NeedsNewMotion ||
                                                 NeedsNewExport ||
                                                 (NeedsNewPicking && CurrentOptionsBoxNet.ExportParticles);

                                if (!IsTomo)
                                {
                                    //Debug.WriteLine(GPU.GetDevice() + " loading...");
                                    var TimerRead = BenchmarkRead.Start();

                                    LoadAndPrepareHeaderAndMap(item.DataPath, ImageGain, DefectMap, ScaleFactor, out OriginalHeader, out OriginalStack, false);
                                    if (NeedStack)
                                        Workers[gpuID].LoadStack(item.DataPath, ScaleFactor, CurrentOptionsExport.EERGroupFrames);

                                    BenchmarkRead.Finish(TimerRead);
                                    //Debug.WriteLine(GPU.GetDevice() + " loaded.");
                                }

                                // Store original dimensions in Angstrom
                                if (!IsTomo)
                                {
                                    CurrentOptionsCTF.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.Import.PixelSize);
                                    CurrentOptionsMovement.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.Import.PixelSize);
                                    CurrentOptionsBoxNet.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.Import.PixelSize);
                                    CurrentOptionsExport.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.Import.PixelSize);
                                }
                                else
                                {
                                    ((TiltSeries)item).LoadMovieSizes();

                                    float3 StackDims = new float3(((TiltSeries)item).ImageDimensionsPhysical, ((TiltSeries)item).NTilts);
                                    CurrentOptionsCTF.Dimensions = StackDims;
                                    CurrentOptionsMovement.Dimensions = StackDims;
                                    CurrentOptionsExport.Dimensions = StackDims;
                                }
                                
                                //Debug.WriteLine(GPU.GetDevice() + " processing...");

                                if (!IsPreprocessing)
                                {
                                    OriginalStack?.Dispose();
                                    return true;
                                } // These checks are needed to abort the processing faster

                                if (DoMovement && NeedsNewMotion && !IsTomo)
                                {
                                    var TimerMotion = BenchmarkMotion.Start();

                                    Workers[gpuID].MovieProcessMovement(item.Path, CurrentOptionsMovement);
                                    item.LoadMeta();
                                    //item.ProcessShift(OriginalStack, CurrentOptionsMovement);

                                    BenchmarkMotion.Finish(TimerMotion);
                                }
                                if (!IsPreprocessing)
                                {
                                    OriginalStack?.Dispose();
                                    return true;
                                }

                                if (DoCTF && NeedsNewCTF)
                                {
                                    var TimerCTF = BenchmarkCTF.Start();

                                    if (!IsTomo)
                                    {
                                        Workers[gpuID].MovieProcessCTF(item.Path, CurrentOptionsCTF);
                                        item.LoadMeta();
                                    }
                                    else
                                    {
                                        Workers[gpuID].TomoProcessCTF(item.Path, CurrentOptionsCTF);
                                        item.LoadMeta();
                                    }

                                    BenchmarkCTF.Finish(TimerCTF);
                                }
                                if (!IsPreprocessing)
                                {
                                    OriginalStack?.Dispose();
                                    return true;
                                }

                                if (DoExport && NeedsNewExport && !IsTomo)
                                {
                                    var TimerOutput = BenchmarkOutput.Start();

                                    Workers[gpuID].MovieExportMovie(item.Path, CurrentOptionsExport);
                                    item.LoadMeta();
                                    //item.ExportMovie(OriginalStack, CurrentOptionsExport);

                                    BenchmarkOutput.Finish(TimerOutput);
                                }

                                if (!File.Exists(item.ThumbnailsPath))
                                    item.CreateThumbnail(384, 2.5f);

                                if (DoPicking && NeedsNewPicking && !IsTomo)
                                {
                                    var TimerPicking = BenchmarkPicking.Start();

                                    MapHeader AverageHeader = MapHeader.ReadFromFilePatient(50, 500, item.AveragePath, new int2(1), 0, typeof(float));

                                    item.MatchBoxNet2(new[] { BoxNetworks[gpuID % NDevices] }, CurrentOptionsBoxNet, null);

                                    #region Export particles if needed

                                    if (CurrentOptionsBoxNet.ExportParticles)
                                    {
                                        float2[] Positions = Star.LoadFloat2(Path.Combine(item.MatchingDir, item.RootName + "_" + BoxNetSuffix + ".star"),
                                                                             "rlnCoordinateX",
                                                                             "rlnCoordinateY").Select(v => v * AverageHeader.PixelSize.X).ToArray();

                                        ProcessingOptionsParticleExport ParticleOptions = new ProcessingOptionsParticleExport
                                        {
                                            Suffix = "_" + BoxNetSuffix,

                                            BoxSize = CurrentOptionsBoxNet.ExportBoxSize,
                                            Diameter = (int)CurrentOptionsBoxNet.ExpectedDiameter,
                                            Invert = CurrentOptionsBoxNet.ExportInvert,
                                            Normalize = CurrentOptionsBoxNet.ExportNormalize,
                                            CorrectAnisotropy = false,

                                            PixelSize = CurrentOptionsBoxNet.PixelSize,
                                            Dimensions = CurrentOptionsBoxNet.Dimensions,

                                            BinTimes = CurrentOptionsBoxNet.BinTimes,
                                            GainPath = CurrentOptionsBoxNet.GainPath,
                                            DosePerAngstromFrame = Options.Import.DosePerAngstromFrame,

                                            DoAverage = true,
                                            StackGroupSize = 1,
                                            SkipFirstN = Options.Export.SkipFirstN,
                                            SkipLastN = Options.Export.SkipLastN,

                                            Voltage = Options.CTF.Voltage
                                        };

                                        if (Positions.Length > 0)
                                        {
                                            Workers[gpuID].MovieExportParticles(item.Path, ParticleOptions, Positions);
                                            item.LoadMeta();
                                            //item.ExportParticles(OriginalStack, Positions, ParticleOptions);
                                        }

                                        OriginalStack?.Dispose();
                                        //Debug.WriteLine(GPU.GetDevice() + " processed.");

                                        float[] Defoci = new float[Positions.Length];
                                        if (item.GridCTFDefocus != null)
                                            Defoci = item.GridCTFDefocus.GetInterpolated(Positions.Select(v => new float3(v.X / CurrentOptionsBoxNet.Dimensions.X,
                                                                                                                   v.Y / CurrentOptionsBoxNet.Dimensions.Y,
                                                                                                                   0.5f)).ToArray());
                                        float Astigmatism = (float)item.CTF.DefocusDelta / 2;
                                        float PhaseShift = item.GridCTFPhase.GetInterpolated(new float3(0.5f)) * 180;

                                        List<string[]> NewRows = new List<string[]>();
                                        for (int r = 0; r < Positions.Length; r++)
                                        {
                                            string[] Row = Helper.ArrayOfConstant("0", TableBoxNetAll.ColumnCount);

                                            Row[TableBoxNetAll.GetColumnID("rlnMagnification")] = "10000.0";
                                            Row[TableBoxNetAll.GetColumnID("rlnDetectorPixelSize")] = Options.Import.BinnedPixelSize.ToString("F5", CultureInfo.InvariantCulture);

                                            Row[TableBoxNetAll.GetColumnID("rlnDefocusU")] = ((Defoci[r] + Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnDefocusV")] = ((Defoci[r] - Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnDefocusAngle")] = item.CTF.DefocusAngle.ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnVoltage")] = item.CTF.Voltage.ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnSphericalAberration")] = item.CTF.Cs.ToString("F4", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnAmplitudeContrast")] = item.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnPhaseShift")] = PhaseShift.ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnCtfMaxResolution")] = item.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnCoordinateX")] = (Positions[r].X / (float)CurrentOptionsBoxNet.BinnedPixelSizeMean).ToString("F2", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnCoordinateY")] = (Positions[r].Y / (float)CurrentOptionsBoxNet.BinnedPixelSizeMean).ToString("F2", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnImageName")] = (r + 1).ToString("D7") + "@particles/" + item.RootName + "_" + BoxNetSuffix + ".mrcs";
                                            Row[TableBoxNetAll.GetColumnID("rlnMicrographName")] = item.Name;

                                            NewRows.Add(Row);
                                        }

                                        List<string[]> RowsAll = new List<string[]>();
                                        List<string[]> RowsGood = new List<string[]>();

                                        lock (AllMovieParticleRows)
                                        {
                                            if (!AllMovieParticleRows.ContainsKey(item))
                                                AllMovieParticleRows.Add(item, NewRows);
                                            else
                                                AllMovieParticleRows[item] = NewRows;

                                            foreach (var pair in AllMovieParticleRows)
                                            {
                                                RowsAll.AddRange(pair.Value);
                                                if (!(pair.Key.UnselectFilter || (pair.Key.UnselectManual != null && pair.Key.UnselectManual.Value)))
                                                    RowsGood.AddRange(pair.Value);
                                            }
                                        }

                                        if (TableBoxNetConcurrent == 0)
                                        {
                                            lock (TableBoxNetAllWriteLock)
                                                TableBoxNetConcurrent++;

                                            Task.Run(() =>
                                            {
                                                Star TempTableAll = new Star(TableBoxNetAll.GetColumnNames());
                                                TempTableAll.AddRow(RowsAll);

                                                bool SuccessAll = false;
                                                while (!SuccessAll)
                                                {
                                                    try
                                                    {
                                                        TempTableAll.Save(PathBoxNetAll + "_" + item.RootName);
                                                        lock (TableBoxNetAllWriteLock)
                                                        {
                                                            if (File.Exists(PathBoxNetAll))
                                                                File.Delete(PathBoxNetAll);
                                                            File.Move(PathBoxNetAll + "_" + item.RootName, PathBoxNetAll);

                                                            if (Options.Picking.DoRunningWindow && TempTableAll.RowCount > 0)
                                                            {
                                                                TempTableAll.CreateSubset(Helper.ArrayOfSequence(Math.Max(0, TempTableAll.RowCount - Options.Picking.RunningWindowLength), 
                                                                                                                 TempTableAll.RowCount - 1, 
                                                                                                                 1)).Save(PathBoxNetAllSubset);
                                                            }
                                                        }
                                                        SuccessAll = true;
                                                    }
                                                    catch { }
                                                }

                                                Star TempTableGood = new Star(TableBoxNetAll.GetColumnNames());
                                                TempTableGood.AddRow(RowsGood);

                                                bool SuccessGood = false;
                                                while (!SuccessGood)
                                                {
                                                    try
                                                    {
                                                        TempTableGood.Save(PathBoxNetFiltered + "_" + item.RootName);
                                                        lock (TableBoxNetAllWriteLock)
                                                        {
                                                            if (File.Exists(PathBoxNetFiltered))
                                                                File.Delete(PathBoxNetFiltered);
                                                            File.Move(PathBoxNetFiltered + "_" + item.RootName, PathBoxNetFiltered);

                                                            if (Options.Picking.DoRunningWindow && TempTableGood.RowCount > 0)
                                                            {
                                                                TempTableGood.CreateSubset(Helper.ArrayOfSequence(Math.Max(0, TempTableGood.RowCount - Options.Picking.RunningWindowLength),
                                                                                                                  TempTableGood.RowCount - 1,
                                                                                                                  1)).Save(PathBoxNetFilteredSubset);
                                                            }
                                                        }
                                                        SuccessGood = true;
                                                    }
                                                    catch { }
                                                }

                                                lock (TableBoxNetAllWriteLock)
                                                    TableBoxNetConcurrent--;
                                            });
                                        }
                                    }
                                    else
                                    {
                                        OriginalStack?.Dispose();
                                        Debug.WriteLine(GPU.GetDevice() + " processed.");
                                    }

                                    #endregion

                                    BenchmarkPicking.Finish(TimerPicking);
                                }
                                else
                                {
                                    OriginalStack?.Dispose();
                                    Debug.WriteLine(GPU.GetDevice() + " processed.");
                                }


                                Dispatcher.Invoke(() =>
                                {
                                    if (DisplayedMovie == item)
                                        UpdateButtonOptionsAdopt();

                                    ProcessingStatusBar.ApplyFilter();
                                    ProcessingStatusBar.UpdateElements();
                                });

                                BenchmarkAllProcessing.Finish(TimerOverall);

                                UpdateStatsAll();

                                return false; // No need to cancel GPU ForEach iterator
                            }
                            catch (Exception exc)
                            {
                                OriginalStack?.Dispose();

                                item.UnselectManual = true;
                                UpdateStatsAll();

                                Dispatcher.Invoke(() =>
                                {
                                    ProcessingStatusBar.ApplyFilter();
                                    ProcessingStatusBar.UpdateElements();
                                });

                                return false;
                            }
                        }, 1, UsedDeviceProcesses);


                        Dispatcher.Invoke(() =>
                        {
                            UpdateStatsAll();
                            ProcessingStatusBar.UpdateElements();
                        });

                        #endregion

                        Dispatcher.Invoke(() =>
                        {
                            ProcessingStatusBar.HideProgressBar();
                            StatsProgressIndicator.Visibility = Visibility.Hidden;
                        });
                    }

                    ImageGain?.Dispose();
                    DefectMap?.Dispose();

                    Parallel.ForEach(Workers, worker =>
                    {
                        worker?.Dispose();
                    });

                    foreach (int d in UsedDevices)
                        BoxNetworks[d]?.Dispose();

                    #region Make sure all particle tables are written out in their most recent form

                    if (Options.ProcessPicking && Options.Picking.DoExport && !string.IsNullOrEmpty(Options.Picking.ModelPath))
                    {
                        ProgressDialogController ProgressDialog = null;
                        await Dispatcher.Invoke(async () => ProgressDialog = await this.ShowProgressAsync($"Waiting for the last particle files to be written out...", ""));
                        ProgressDialog.SetIndeterminate();

                        List<string[]> RowsAll = new List<string[]>();
                        List<string[]> RowsGood = new List<string[]>();

                        lock (AllMovieParticleRows)
                        {
                            foreach (var pair in AllMovieParticleRows)
                            {
                                RowsAll.AddRange(pair.Value);
                                if (!(pair.Key.UnselectFilter || (pair.Key.UnselectManual != null && pair.Key.UnselectManual.Value)))
                                    RowsGood.AddRange(pair.Value);
                            }
                        }

                        while (TableBoxNetConcurrent > 0)
                            Thread.Sleep(50);
                        
                        Star TempTableAll = new Star(TableBoxNetAll.GetColumnNames());
                        TempTableAll.AddRow(RowsAll);

                        bool SuccessAll = false;
                        while (!SuccessAll)
                        {
                            try
                            {
                                TempTableAll.Save(PathBoxNetAll + "_temp");
                                lock (TableBoxNetAllWriteLock)
                                {
                                    if (File.Exists(PathBoxNetAll))
                                        File.Delete(PathBoxNetAll);
                                    File.Move(PathBoxNetAll + "_temp", PathBoxNetAll);
                                }
                                SuccessAll = true;
                            }
                            catch { }
                        }

                        Star TempTableGood = new Star(TableBoxNetAll.GetColumnNames());
                        TempTableGood.AddRow(RowsGood);

                        bool SuccessGood = false;
                        while (!SuccessGood)
                        {
                            try
                            {
                                TempTableGood.Save(PathBoxNetFiltered + "_temp");
                                lock (TableBoxNetAllWriteLock)
                                {
                                    if (File.Exists(PathBoxNetFiltered))
                                        File.Delete(PathBoxNetFiltered);
                                    File.Move(PathBoxNetFiltered + "_temp", PathBoxNetFiltered);
                                }
                                SuccessGood = true;
                            }
                            catch { }
                        }

                        await ProgressDialog.CloseAsync();
                    }

                    #endregion
                });
            }
            else
            {
                IsStoppingPreprocessing = true;

                ButtonStartProcessing.IsEnabled = false;
                ButtonStartProcessing.Content = "STOPPING...";

                IsPreprocessing = false;
                if (PreprocessingTask != null)
                {
                    await PreprocessingTask;
                    PreprocessingTask = null;
                }

                foreach (var item in DisableWhenPreprocessing)
                    item.IsEnabled = true;
                MicrographDisplayControl.SetProcessingMode(false);
                
                #region Timers

                BenchmarkAllProcessing.Clear();
                BenchmarkRead.Clear();
                BenchmarkCTF.Clear();
                BenchmarkMotion.Clear();
                BenchmarkPicking.Clear();
                BenchmarkOutput.Clear();

                #endregion

                UpdateStatsAll();

                ButtonStartProcessing.IsEnabled = true;
                ButtonStartProcessing.Content = "START PROCESSING";
                ButtonStartProcessing.Foreground = new LinearGradientBrush(Colors.DeepSkyBlue, Colors.DeepPink, 0);

                IsStoppingPreprocessing = false;
            }
        }

        private void ButtonCollapseMainPreprocessing_OnClick(object sender, RoutedEventArgs e)
        {
            if (!IsPreprocessingCollapsed)
            {
                PreprocessingWidth = (int)ColumnMainPreprocessing.Width.Value;

                ButtonCollapseMainPreprocessing.Content = "▶";
                ButtonCollapseMainPreprocessing.ToolTip = "Expand";
                GridMainPreprocessing.Visibility = Visibility.Collapsed;
                ColumnMainPreprocessing.Width = new GridLength(0);

                bool NeedsTabSwitch = TabsProcessingView.SelectedItem == TabProcessingCTF || TabsProcessingView.SelectedItem == TabProcessingMovement;

                // Reorder controls
                GridCTFDisplay.Children.Clear();
                GridMicrographDisplay.Children.Clear();

                GridMergedCTFAndMovement.Children.Add(CTFDisplayControl);
                GridMergedCTFAndMovement.Children.Add(PanelButtonsOneTimeCTF);
                GridMergedCTFAndMovement.Children.Add(MicrographDisplayControl);
                Grid.SetColumn(MicrographDisplayControl, 2);

                // Hide and show tabs
                TabProcessingCTFAndMovement.Visibility = Visibility.Visible;
                TabProcessingCTF.Visibility = Visibility.Collapsed;
                TabProcessingMovement.Visibility = Visibility.Collapsed;

                if (NeedsTabSwitch)
                    TabsProcessingView.SelectedItem = TabProcessingCTFAndMovement;
            }
            else
            {
                ButtonCollapseMainPreprocessing.Content = "◀";
                ButtonCollapseMainPreprocessing.ToolTip = "Collapse";
                GridMainPreprocessing.Visibility = Visibility.Visible;
                ColumnMainPreprocessing.Width = new GridLength(PreprocessingWidth);

                bool NeedsTabSwitch = TabsProcessingView.SelectedItem == TabProcessingCTFAndMovement;

                // Reorder controls
                GridMergedCTFAndMovement.Children.Clear();

                GridCTFDisplay.Children.Add(CTFDisplayControl);
                GridCTFDisplay.Children.Add(PanelButtonsOneTimeCTF);
                GridMicrographDisplay.Children.Add(MicrographDisplayControl);
                Grid.SetColumn(MicrographDisplayControl, 0);

                // Hide and show tabs
                TabProcessingCTFAndMovement.Visibility = Visibility.Collapsed;
                TabProcessingCTF.Visibility = Visibility.Visible;
                TabProcessingMovement.Visibility = Visibility.Visible;

                if (NeedsTabSwitch)
                    TabsProcessingView.SelectedItem = TabProcessingCTF;
            }

            IsPreprocessingCollapsed = !IsPreprocessingCollapsed;
        }

        #region L2 TAB: OVERVIEW

        #region Statistics and filters

        #region Variables

        private int StatsAstigmatismZoom = 1;

        private BenchmarkTimer BenchmarkRead = new BenchmarkTimer("File read");
        private BenchmarkTimer BenchmarkCTF = new BenchmarkTimer("CTF");
        private BenchmarkTimer BenchmarkMotion = new BenchmarkTimer("Motion");
        private BenchmarkTimer BenchmarkPicking = new BenchmarkTimer("Picking");
        private BenchmarkTimer BenchmarkOutput = new BenchmarkTimer("Output");

        private BenchmarkTimer BenchmarkAllProcessing = new BenchmarkTimer("All processing");

        #endregion

        public void UpdateStatsAll()
        {
            UpdateFilterRanges();
            UpdateFilterResult();
            UpdateStatsAstigmatismPlot();
            UpdateStatsStatus();
            UpdateFilterSuffixMenu();
            UpdateBenchmarkTimes();
        }

        private void UpdateStatsStatus()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();

            bool HaveCTF = Options.ProcessCTF || Items.Any(v => v.OptionsCTF != null && v.CTF != null);
            bool HavePhase = Options.CTF.DoPhase || Items.Any(v => v.OptionsCTF != null && v.OptionsCTF.DoPhase);
            bool HaveMovement = Options.ProcessMovement || Items.Any(v => v.OptionsMovement != null);
            bool HaveParticles = Items.Any(m => m.HasParticleSuffix(Options.Filter.ParticlesSuffix));

            ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
            ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
            ProcessingOptionsBoxNet OptionsBoxNet = Options.GetProcessingBoxNet();
            ProcessingOptionsMovieExport OptionsExport = Options.GetProcessingMovieExport();

            int[] ColorIDs = new int[Items.Length];
            int NProcessed = 0, NOutdated = 0, NUnprocessed = 0, NFilteredOut = 0, NUnselected = 0;
            for (int i = 0; i < Items.Length; i++)
            {
                ProcessingStatus Status = StatusBar.GetMovieProcessingStatus(Items[i], OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, Options);
                Items[i].ProcessingStatus = Status;

                int ID = 0;
                switch (Status)
                {
                    case ProcessingStatus.Processed:
                        ID = 0;
                        NProcessed++;
                        break;
                    case ProcessingStatus.Outdated:
                        ID = 1;
                        NOutdated++;
                        break;
                    case ProcessingStatus.Unprocessed:
                        ID = 2;
                        NUnprocessed++;
                        break;
                    case ProcessingStatus.FilteredOut:
                        ID = 3;
                        NFilteredOut++;
                        break;
                    case ProcessingStatus.LeaveOut:
                        ID = 4;
                        NUnselected++;
                        break;
                }
                ColorIDs[i] = ID;
            }

            Dispatcher.InvokeAsync(() =>
            {
                StatsSeriesStatusProcessed.Visibility = NProcessed == 0 ? Visibility.Collapsed : Visibility.Visible;
                StatsSeriesStatusOutdated.Visibility = NOutdated == 0 ? Visibility.Collapsed : Visibility.Visible;
                StatsSeriesStatusUnprocessed.Visibility = NUnprocessed == 0 ? Visibility.Collapsed : Visibility.Visible;
                StatsSeriesStatusUnfiltered.Visibility = NFilteredOut == 0 ? Visibility.Collapsed : Visibility.Visible;
                StatsSeriesStatusUnselected.Visibility = NUnselected == 0 ? Visibility.Collapsed : Visibility.Visible;

                StatsSeriesStatusProcessed.Values = new ChartValues<ObservableValue> { new ObservableValue(NProcessed) };
                StatsSeriesStatusOutdated.Values = new ChartValues<ObservableValue> { new ObservableValue(NOutdated) };
                StatsSeriesStatusUnprocessed.Values = new ChartValues<ObservableValue> { new ObservableValue(NUnprocessed) };
                StatsSeriesStatusUnfiltered.Values = new ChartValues<ObservableValue> { new ObservableValue(NFilteredOut) };
                StatsSeriesStatusUnselected.Values = new ChartValues<ObservableValue> { new ObservableValue(NUnselected) };
            });

            if (HaveCTF)
            {
                #region Defocus

                double[] DefocusValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    if (Items[i].OptionsCTF != null && Items[i].CTF != null)
                        DefocusValues[i] = (double)Items[i].CTF.Defocus;
                    else
                        DefocusValues[i] = double.NaN;

                SingleAxisPoint[] DefocusPlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    DefocusPlotValues[i] = new SingleAxisPoint(DefocusValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() => PlotStatsDefocus.Points = new ObservableCollection<SingleAxisPoint>(DefocusPlotValues));

                #endregion

                #region Phase

                if (HavePhase)
                {
                    double[] PhaseValues = new double[Items.Length];
                    for (int i = 0; i < Items.Length; i++)
                        if (Items[i].OptionsCTF != null && Items[i].CTF != null)
                            PhaseValues[i] = (double)Items[i].CTF.PhaseShift;
                        else
                            PhaseValues[i] = double.NaN;

                    SingleAxisPoint[] PhasePlotValues = new SingleAxisPoint[Items.Length];
                    for (int i = 0; i < Items.Length; i++)
                        PhasePlotValues[i] = new SingleAxisPoint(PhaseValues[i], ColorIDs[i], Items[i]);

                    Dispatcher.InvokeAsync(() => PlotStatsPhase.Points = new ObservableCollection<SingleAxisPoint>(PhasePlotValues));
                }
                else
                    Dispatcher.InvokeAsync(() => PlotStatsPhase.Points = null);

                #endregion

                #region Resolution

                double[] ResolutionValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    if (Items[i].CTFResolutionEstimate > 0)
                        ResolutionValues[i] = (double)Items[i].CTFResolutionEstimate;
                    else
                        ResolutionValues[i] = double.NaN;

                SingleAxisPoint[] ResolutionPlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    ResolutionPlotValues[i] = new SingleAxisPoint(ResolutionValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() => PlotStatsResolution.Points = new ObservableCollection<SingleAxisPoint>(ResolutionPlotValues));

                #endregion
            }
            else
            {
                Dispatcher.InvokeAsync(() =>
                {
                    //StatsSeriesAstigmatism0.Values = new ChartValues<ObservablePoint>();
                    PlotStatsDefocus.Points = null;
                    PlotStatsPhase.Points = null;
                    PlotStatsResolution.Points = null;
                });
            }

            if (HaveMovement)
            {
                double[] MovementValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    if (Items[i].MeanFrameMovement > 0)
                        MovementValues[i] = (double)Items[i].MeanFrameMovement;
                    else
                        MovementValues[i] = double.NaN;

                SingleAxisPoint[] MovementPlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    MovementPlotValues[i] = new SingleAxisPoint(MovementValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() => PlotStatsMotion.Points = new ObservableCollection<SingleAxisPoint>(MovementPlotValues));
            }
            else
            {
                Dispatcher.InvokeAsync(() => PlotStatsMotion.Points = null);
            }

            if (HaveParticles)
            {
                int CountSum = 0, CountFilteredSum = 0;
                double[] ParticleValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                {
                    int Count = Items[i].GetParticleCount(Options.Filter.ParticlesSuffix);
                    if (Count >= 0)
                    {
                        ParticleValues[i] = Count;
                        CountSum += Count;

                        if (!(Items[i].UnselectFilter || (Items[i].UnselectManual != null && Items[i].UnselectManual.Value)))
                            CountFilteredSum += Count;
                    }
                    else
                        ParticleValues[i] = double.NaN;
                }

                SingleAxisPoint[] ParticlePlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    ParticlePlotValues[i] = new SingleAxisPoint(ParticleValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() =>
                {
                    PlotStatsParticles.Points = new ObservableCollection<SingleAxisPoint>(ParticlePlotValues);
                    TextStatsParticlesOverall.Value = CountSum.ToString();
                    TextStatsParticlesFiltered.Value = CountFilteredSum.ToString();
                });
            }
            else
            {
                Dispatcher.InvokeAsync(() => PlotStatsParticles.Points = null);
            }

            {
                double[] MaskPercentageValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    if (Items[i].MaskPercentage >= 0)
                        MaskPercentageValues[i] = (double)Items[i].MaskPercentage;
                    else
                        MaskPercentageValues[i] = double.NaN;

                SingleAxisPoint[] MaskPercentagePlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    MaskPercentagePlotValues[i] = new SingleAxisPoint(MaskPercentageValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() => PlotStatsMaskPercentage.Points = new ObservableCollection<SingleAxisPoint>(MaskPercentagePlotValues));
            }

            try
            {
                JsonArray ItemsJson = new JsonArray(Items.Select(m => m.ToMiniJson(Options.Filter.ParticlesSuffix)).ToArray());
                File.WriteAllText(Path.Join(Options.Import.ProcessingOrDataFolder, "items.json"), ItemsJson.ToJsonString(new JsonSerializerOptions() { WriteIndented = true }));
            }
            catch { }
        }

        private void UpdateStatsAstigmatismPlot()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();

            bool HaveCTF = Options.ProcessCTF || Items.Any(v => v.OptionsCTF != null && v.CTF != null);

            if (HaveCTF)
            {
                #region Astigmatism

                DualAxisPoint[] AstigmatismPoints = new DualAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                {
                    Movie item = Items[i];
                    DualAxisPoint P = new DualAxisPoint();
                    P.Context = item;
                    P.ColorID = i * 4 / Items.Length;
                    if (item.OptionsCTF != null && item.CTF != null)
                    {
                        P.X = Math.Round(Math.Cos((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta, 4);
                        P.Y = Math.Round(Math.Sin((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta, 4);
                        P.Label = item.CTF.DefocusDelta.ToString("F4");
                    }
                    else
                        P.Label = "";

                    AstigmatismPoints[i] = P;
                }

                Dispatcher.InvokeAsync(() =>
                {
                    PlotStatsAstigmatism.Points = new ObservableCollection<DualAxisPoint>(AstigmatismPoints);
                });

                #endregion
            }
            else
            {
                Dispatcher.InvokeAsync(() =>
                {
                    PlotStatsAstigmatism.Points = new ObservableCollection<DualAxisPoint>();
                });
            }
        }

        float2 FilterAstigmatismMean = new float2(0);
        float FilterAstigmatismStd = 0.1f;

        private void UpdateFilterRanges()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();
            Movie[] ItemsWithCTF = Items.Where(v => v.OptionsCTF != null && v.CTF != null).ToArray();
            Movie[] ItemsWithMovement = Items.Where(v => v.OptionsMovement != null).ToArray();

            #region Astigmatism (includes adjusting the plot elements)

            float2 AstigmatismMean = new float2();
            float AstigmatismStd = 0.1f;
            float AstigmatismMax = 0.4f;

            // Get all items with valid CTF information
            List<float2> AstigmatismPoints = new List<float2>(ItemsWithCTF.Length);
            foreach (var item in ItemsWithCTF)
                AstigmatismPoints.Add(new float2((float)Math.Cos((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta,
                                                    (float)Math.Sin((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta));

            // Calculate mean and stddev of all points in Cartesian coords
            if (AstigmatismPoints.Count > 0)
            {
                AstigmatismMean = new float2();
                AstigmatismMax = 0;
                foreach (var point in AstigmatismPoints)
                {
                    AstigmatismMean += point;
                    AstigmatismMax = Math.Max(AstigmatismMax, point.LengthSq());
                }
                AstigmatismMax = (float)Math.Sqrt(AstigmatismMax);
                AstigmatismMean /= AstigmatismPoints.Count;

                AstigmatismStd = 0;
                foreach (var point in AstigmatismPoints)
                    AstigmatismStd += (point - AstigmatismMean).LengthSq();
                AstigmatismStd = (float)Math.Max(1e-4, Math.Sqrt(AstigmatismStd / AstigmatismPoints.Count));
            }

            AstigmatismMax = Math.Max(1e-4f, (float)Math.Ceiling(AstigmatismMax * 20) / 20);

            // Set the labels for outer and inner circle
            Dispatcher.InvokeAsync(() =>
            {
                StatsAstigmatismLabelOuter.Value = (AstigmatismMax / StatsAstigmatismZoom).ToString("F3", CultureInfo.InvariantCulture);
                StatsAstigmatismLabelInner.Value = (AstigmatismMax / StatsAstigmatismZoom / 2).ToString("F3", CultureInfo.InvariantCulture);

                // Adjust plot axes
                
                PlotStatsAstigmatism.AxisMax = AstigmatismMax / StatsAstigmatismZoom;

                // Scale and position the valid range ellipse
                StatsAstigmatismEllipseSigma.Width = AstigmatismStd * StatsAstigmatismZoom * (float)Options.Filter.AstigmatismMax / AstigmatismMax * 256;
                StatsAstigmatismEllipseSigma.Height = AstigmatismStd * StatsAstigmatismZoom * (float)Options.Filter.AstigmatismMax / AstigmatismMax * 256;
                Canvas.SetLeft(StatsAstigmatismEllipseSigma, AstigmatismMean.X / AstigmatismMax * 128 * StatsAstigmatismZoom + 128 - StatsAstigmatismEllipseSigma.Width / 2);
                Canvas.SetTop(StatsAstigmatismEllipseSigma, AstigmatismMean.Y / AstigmatismMax * 128 * StatsAstigmatismZoom + 128 - StatsAstigmatismEllipseSigma.Height / 2);
            });

            lock (Options)
            {
                FilterAstigmatismMean = AstigmatismMean;
                FilterAstigmatismStd = AstigmatismStd;
            }

            #endregion

            bool HaveCTF = Options.ProcessCTF || ItemsWithCTF.Length > 0;
            bool HavePhase = Options.CTF.DoPhase || ItemsWithCTF.Any(v => v.OptionsCTF.DoPhase);
            bool HaveMovement = Options.ProcessMovement || ItemsWithMovement.Length > 0;
            bool HaveParticles = Items.Any(m => m.HasAnyParticleSuffixes());

            Dispatcher.InvokeAsync(() =>
            {
                PanelStatsAstigmatism.Visibility = HaveCTF ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsDefocus.Visibility = HaveCTF ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsPhase.Visibility = HaveCTF && HavePhase ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsResolution.Visibility = HaveCTF ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsMotion.Visibility = HaveMovement ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsParticles.Visibility = HaveParticles ? Visibility.Visible : Visibility.Collapsed;
            });
        }

        private void UpdateFilterResult()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();

            float2 AstigmatismMean;
            float AstigmatismStd;
            lock (Options)
            {
                AstigmatismMean = FilterAstigmatismMean;
                AstigmatismStd = FilterAstigmatismStd;
            }

            foreach (var item in Items)
            {
                bool FilterStatus = true;

                if (item.OptionsCTF != null)
                {
                    FilterStatus &= item.CTF.Defocus >= Options.Filter.DefocusMin && item.CTF.Defocus <= Options.Filter.DefocusMax;
                    float AstigmatismDeviation = (new float2((float)Math.Cos((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta,
                                                             (float)Math.Sin((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta) - AstigmatismMean).Length() / AstigmatismStd;
                    FilterStatus &= AstigmatismDeviation <= (float)Options.Filter.AstigmatismMax;

                    FilterStatus &= item.CTFResolutionEstimate <= Options.Filter.ResolutionMax;

                    if (Options.CTF.DoPhase)
                        FilterStatus &= item.CTF.PhaseShift >= Options.Filter.PhaseMin && item.CTF.PhaseShift <= Options.Filter.PhaseMax;
                }

                if (item.OptionsMovement != null)
                {
                    FilterStatus &= item.MeanFrameMovement <= Options.Filter.MotionMax;
                }

                if (item.HasAnyParticleSuffixes())
                {
                    int Count = item.GetParticleCount(Options.Filter.ParticlesSuffix);
                    if (Count >= 0)
                        FilterStatus &= Count >= Options.Filter.ParticlesMin;
                }

                FilterStatus &= item.MaskPercentage <= Options.Filter.MaskPercentage;

                item.UnselectFilter = !FilterStatus;
            }

            // Calculate average CTF
            Task.Run(() =>
            {
                try
                {
                    CTF[] AllCTFs = Items.Where(m => m.OptionsCTF != null && !m.UnselectFilter).Select(m => m.CTF.GetCopy()).ToArray();
                    decimal PixelSize = Options.Import.BinnedPixelSize;

                    Dispatcher.Invoke(() => StatsDefocusAverageCTFFrequencyLabel.Text = $"1/{PixelSize:F1} Å");

                    float[] AverageCTFValues = new float[192];
                    foreach (var ctf in AllCTFs)
                    {
                        ctf.PixelSize = PixelSize;
                        float[] Simulated = ctf.Get1D(AverageCTFValues.Length, true);

                        for (int i = 0; i < Simulated.Length; i++)
                            AverageCTFValues[i] += Simulated[i];
                    }

                    if (AllCTFs.Length > 1)
                        for (int i = 0; i < AverageCTFValues.Length; i++)
                            AverageCTFValues[i] /= AllCTFs.Length;

                    float MinAverage = MathHelper.Min(AverageCTFValues);

                    Dispatcher.Invoke(() =>
                    {
                        IEnumerable<Point> TrackPoints = AverageCTFValues.Select((v, i) => new Point(i, 24 - 1 - (24 * v)));

                        System.Windows.Shapes.Path TrackPath = new System.Windows.Shapes.Path()
                        {
                            Stroke = StatsDefocusAverageCTFFrequencyLabel.Foreground,
                            StrokeThickness = 1,
                            StrokeLineJoin = PenLineJoin.Bevel,
                            IsHitTestVisible = false
                        };
                        PolyLineSegment PlotSegment = new PolyLineSegment(TrackPoints, true);
                        PathFigure PlotFigure = new PathFigure
                        {
                            Segments = new PathSegmentCollection { PlotSegment },
                            StartPoint = TrackPoints.First()
                        };
                        TrackPath.Data = new PathGeometry { Figures = new PathFigureCollection { PlotFigure } };

                        StatsDefocusAverageCTFCanvas.Children.Clear();
                        StatsDefocusAverageCTFCanvas.Children.Add(TrackPath);
                        Canvas.SetBottom(TrackPath, 24 * MinAverage);
                    });
                }
                catch { }
            });
        }

        public void UpdateFilterSuffixMenu()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();
            List<string> Suffixes = new List<string>();

            foreach (var movie in Items)
                foreach (var suffix in movie.GetParticlesSuffixes())
                    if (!Suffixes.Contains(suffix))
                        Suffixes.Add(suffix);

            Suffixes.Sort();
            Dispatcher.InvokeAsync(() =>
            {
                MenuParticlesSuffix.Items.Clear();
                foreach (var suffix in Suffixes)
                    MenuParticlesSuffix.Items.Add(suffix);

                if ((string.IsNullOrEmpty(Options.Filter.ParticlesSuffix) || !Suffixes.Contains(Options.Filter.ParticlesSuffix))
                    && Suffixes.Count > 0)
                    Options.Filter.ParticlesSuffix = Suffixes[0];
            });
        }

        public void UpdateBenchmarkTimes()
        {
            Dispatcher.Invoke(() =>
            {
                StatsBenchmarkOverall.Text = "";

                if (BenchmarkAllProcessing.NItems < 5)
                    return;

                int NMeasurements = Math.Min(BenchmarkAllProcessing.NItems, 100);

                StatsBenchmarkOverall.Text = ((int)Math.Round(BenchmarkAllProcessing.GetPerSecondConcurrent(NMeasurements) * 3600)) + " / h";

                StatsBenchmarkInput.Text = BenchmarkRead.NItems > 0 ? (BenchmarkRead.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
                StatsBenchmarkCTF.Text = BenchmarkCTF.NItems > 0 ? (BenchmarkCTF.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
                StatsBenchmarkMotion.Text = BenchmarkMotion.NItems > 0 ? (BenchmarkMotion.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
                StatsBenchmarkPicking.Text = BenchmarkPicking.NItems > 0 ? (BenchmarkPicking.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
                StatsBenchmarkOutput.Text = BenchmarkOutput.NItems > 0 ? (BenchmarkOutput.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
            });
        }

        #region GUI events

        private void PlotStatsAstigmatism_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingCTF.IsSelected = true;
            });
        }

        private void PlotStatsDefocus_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;
            
            DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingCTF.IsSelected = true;
            });
        }

        private void PlotStatsPhase_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingCTF.IsSelected = true;
            });
        }

        private void PlotStatsResolution_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingCTF.IsSelected = true;
            });
        }

        private void PlotStatsMotion_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingMovement.IsSelected = true;
            });
        }

        private void PlotStatsParticles_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingMovement.IsSelected = true;
            });
        }

        private void PlotStatsMaskPercentage_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingMovement.IsSelected = true;
            });
        }

        private void StatsAstigmatismBackground_OnMouseWheel(object sender, MouseWheelEventArgs e)
        {
            StatsAstigmatismZoom = Math.Max(1, StatsAstigmatismZoom + Math.Sign(e.Delta));
            UpdateFilterRanges();
        }

        private void MenuParticlesSuffix_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            UpdateFilterSuffixMenu();
        }
        
        private void StatsBenchmarkOverall_OnMouseEnter(object sender, MouseEventArgs e)
        {
            StatsBenchmarkDetails.Visibility = Visibility.Visible;
        }

        private void StatsBenchmarkDetails_OnMouseLeave(object sender, MouseEventArgs e)
        {
            StatsBenchmarkDetails.Visibility = Visibility.Hidden;
        }

        #endregion

        #endregion

        #region Task button events (micrograph, particle export, adjustment etc.)

        #region 2D

        private void ButtonTasksExportMicrographs_OnClick(object sender, RoutedEventArgs e)
        {
            Movie[] Movies = FileDiscoverer.GetImmutableFiles();

            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            Dialog2DList DialogContent = new Dialog2DList(Movies, Options);
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        private void ButtonTasksAdjustDefocus_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star",
                Title = "Select a file with particle metadata, e.g. a run_data.star file from RELION's refinement"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                Movie[] Movies = FileDiscoverer.GetImmutableFiles();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                Dialog2DDefocusUpdate DialogContent = new Dialog2DDefocusUpdate(Movies, OpenDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        private void ButtonTasksExportParticles_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star",
                Title = "Select a file with particle coordinates for either the entire data set or one movie"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                Movie[] Movies = FileDiscoverer.GetImmutableFiles();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                Dialog2DParticleExport DialogContent = new Dialog2DParticleExport(Movies, OpenDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        private void ButtonTasksImportParticles_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog FileDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star",
                Title = "Select a file with particle coordinates for either the entire data set or one movie"
            };
            System.Windows.Forms.DialogResult Result = FileDialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Movie[] Movies = FileDiscoverer.GetImmutableFiles();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                Dialog2DParticleImport DialogContent = new Dialog2DParticleImport(Movies, FileDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        private void ButtonTasksMatch_OnClick(object sender, RoutedEventArgs e)
        {
            if (Options.Import.ExtensionTomoSTAR)   // This is not for tomo
                return;
            System.Windows.Forms.OpenFileDialog FileDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "MRC Volumes|*.mrc",
                Multiselect = false,
                Title = "Select template volume"
            };
            System.Windows.Forms.DialogResult Result = FileDialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Movie[] ImmutableItems = FileDiscoverer.GetImmutableFiles();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                Dialog2DMatch DialogContent = new Dialog2DMatch(ImmutableItems, FileDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        private void ButtonTasksExportBoxNet_OnClick(object sender, RoutedEventArgs e)
        {
            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            BoxNetExport DialogContent = new BoxNetExport(Options);
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        #endregion

        #region Tomo
               
        private void ButtonTasksImportImod_Click(object sender, RoutedEventArgs e)
        {
            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogTomoImportImod DialogContent = new DialogTomoImportImod(Options);
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        private void ButtonTasksExportTomograms_OnClick(object sender, RoutedEventArgs e)
        {
            TiltSeries[] Series = FileDiscoverer.GetImmutableFiles().Cast<TiltSeries>().ToArray();

            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogTomoList DialogContent = new DialogTomoList(Series, Options);
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        private void ButtonTasksReconstructTomograms_OnClick(object sender, RoutedEventArgs e)
        {
            if (!Options.Import.ExtensionTomoSTAR)
                return;

            TiltSeries[] ImmutableItems = FileDiscoverer.GetImmutableFiles().Cast<TiltSeries>().ToArray();

            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogTomoReconstruction DialogContent = new DialogTomoReconstruction(ImmutableItems, Options);
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        private void ButtonTasksReconstructSubtomograms_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star",
                Title = "Select file with particles coordinates for either the entire data set or one tomogram"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                TiltSeries[] Series = FileDiscoverer.GetImmutableFiles().Cast<TiltSeries>().ToArray();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                DialogTomoParticleExport DialogContent = new DialogTomoParticleExport(Series, OpenDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        private void ButtonTasksMatchTomograms_OnClick(object sender, RoutedEventArgs e)
        {
            if (!Options.Import.ExtensionTomoSTAR)
                return;
            System.Windows.Forms.OpenFileDialog FileDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "MRC Volumes|*.mrc",
                Multiselect = false,
                Title = "Select template volume"
            };
            System.Windows.Forms.DialogResult Result = FileDialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                TiltSeries[] ImmutableItems = FileDiscoverer.GetImmutableFiles().Cast<TiltSeries>().ToArray();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                DialogTomoMatch DialogContent = new DialogTomoMatch(ImmutableItems, FileDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        #endregion

        #endregion

        private void TomoAdjustInterface()
        {
            if (Options.Import.ExtensionTomoSTAR)            // Tomo interface
            {
                foreach (var element in HideWhen2D)
                    element.Visibility = Visibility.Visible;
                foreach (var element in HideWhenTomo)
                    element.Visibility = Visibility.Collapsed;

                Options.ProcessMovement = false;
                Options.Export.DoAverage = false;
                Options.Export.DoDeconvolve = false;
                Options.Export.DoStack = false;
            }
            else                                        // 2D interface
            {
                foreach (var element in HideWhen2D)
                    element.Visibility = Visibility.Collapsed;
                foreach (var element in HideWhenTomo)
                    element.Visibility = Visibility.Visible;

                if (Options.Import.ExtensionEER)
                    GridOptionsFrameGrouping.Visibility = Visibility.Visible;
                else
                    GridOptionsFrameGrouping.Visibility = Visibility.Collapsed;
            }
        }

        #endregion

        #region L2 TAB: CTF

        private async void ButtonProcessOneItemCTF_OnClick(object sender, RoutedEventArgs e)
        {
            if (DisplayedMovie == null)
                return;

            Stopwatch Watch = new Stopwatch();
            Watch.Start();

            Movie Item = DisplayedMovie;

            var Dialog = await this.ShowProgressAsync("Please wait...", $"Processing CTF for {Item.Name}...");
            Dialog.SetIndeterminate();

            await Task.Run(async () =>
            {
                Image ImageGain = null;
                DefectModel DefectMap = null;
                Image OriginalStack = null;

                HeaderEER.GroupNFrames = Options.Import.EERGroupFrames;

                try
                {
                    #region Get gain ref if needed

                    if (!string.IsNullOrEmpty(Options.Import.GainPath) && Options.Import.CorrectGain && File.Exists(Options.Import.GainPath))
                        ImageGain = LoadAndPrepareGainReference();

                    if (!string.IsNullOrEmpty(Options.Import.DefectsPath) && Options.Import.CorrectDefects && File.Exists(Options.Import.DefectsPath))
                        DefectMap = LoadAndPrepareDefectMap();

                    if (ImageGain != null && DefectMap != null)
                        if (ImageGain.Dims.X != DefectMap.Dims.X || ImageGain.Dims.Y != DefectMap.Dims.Y)
                            throw new Exception("Gain reference and defect map dimensions don't match");

                    #endregion

                    bool IsTomo = Item.GetType() == typeof(TiltSeries);

                    #region Load movie

                    MapHeader OriginalHeader = null;
                    decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                    if (!IsTomo)
                        LoadAndPrepareHeaderAndMap(Item.DataPath, ImageGain, DefectMap, ScaleFactor, out OriginalHeader, out OriginalStack);

                    #endregion

                    Watch.Stop();
                    Debug.WriteLine(Watch.ElapsedMilliseconds / 1e3);

                    ProcessingOptionsMovieCTF CurrentOptionsCTF = Options.GetProcessingMovieCTF();

                    // Store original dimensions in Angstrom
                    if (!IsTomo)
                    {
                        CurrentOptionsCTF.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.Import.PixelSize);
                    }
                    else
                    {
                        ((TiltSeries)Item).LoadMovieSizes();

                        float3 StackDims = new float3(((TiltSeries)Item).ImageDimensionsPhysical, ((TiltSeries)Item).NTilts);
                        CurrentOptionsCTF.Dimensions = StackDims;
                    }

                    if (Item.GetType() == typeof(Movie))
                        Item.ProcessCTF(OriginalStack, CurrentOptionsCTF);
                    else
                        ((TiltSeries)Item).ProcessCTFSimultaneous(CurrentOptionsCTF);

                    Dispatcher.Invoke(() =>
                    {
                        UpdateButtonOptionsAdopt();

                        ProcessingStatusBar.UpdateElements();
                    });

                    UpdateStatsAll();

                    OriginalStack?.Dispose();
                    ImageGain?.Dispose();
                    DefectMap?.Dispose();

                    await Dialog.CloseAsync();
                }
                catch (Exception exc)
                {
                    ImageGain?.Dispose();
                    DefectMap?.Dispose();
                    OriginalStack?.Dispose();

                    await Dispatcher.Invoke(async () =>
                    {
                        if (Dialog.IsOpen)
                            await Dialog.CloseAsync();

                        Item.UnselectManual = true;

                        await this.ShowMessageAsync("Oopsie", "An error occurred while fitting the CTF. Likely reasons include:\n\n" +
                                                              "– Insufficient read/write permissions in this folder.\n" +
                                                              "– Too low defocus to yield more than one CTF peak in the processing range.\n" +
                                                              "– Mismatch in gain reference and image dimensions.\n\n" +
                                                              "The exception raised is:\n" + exc.ToString());
                    });
                }
            });
        }
        

        private async void ButtonProcessOneItemTiltHandedness_Click(object sender, RoutedEventArgs e)
        {
            if (DisplayedMovie == null)
                return;

            Stopwatch Watch = new Stopwatch();
            Watch.Start();

            TiltSeries Series = (TiltSeries)DisplayedMovie;

            var Dialog = await this.ShowProgressAsync("Please wait...", $"Loading tilt movies and estimating gradients...");
            Dialog.SetIndeterminate();

            await Task.Run(async () =>
            {
                try
                {
                    Movie[] TiltMovies = Series.TiltMoviePaths.Select(s => new Movie(Path.Combine(Series.DataOrProcessingDirectoryName, s))).ToArray();

                    if (TiltMovies.Any(m => m.GridCTFDefocus.Values.Length < 2))
                        throw new Exception("One or more tilt movies don't have local defocus information.\n" +
                                            "Please run CTF estimation on all individual tilt movies with a 2x2 spatial resolution grid.");

                    Series.VolumeDimensionsPhysical = new float3((float)Options.Tomo.DimensionsX,
                                                                 (float)Options.Tomo.DimensionsY,
                                                                 (float)Options.Tomo.DimensionsZ) * (float)Options.Import.PixelSize;
                    Series.ImageDimensionsPhysical = new float2(Series.VolumeDimensionsPhysical.X, Series.VolumeDimensionsPhysical.Y);

                    float[] GradientsEstimated = new float[Series.NTilts];
                    float[] GradientsAssumed = new float[Series.NTilts];

                    float3[] Points = 
                    {
                        new float3(0, Series.VolumeDimensionsPhysical.Y / 2, Series.VolumeDimensionsPhysical.Z / 2),
                        new float3(Series.VolumeDimensionsPhysical.X, Series.VolumeDimensionsPhysical.Y / 2, Series.VolumeDimensionsPhysical.Z / 2)
                    };

                    float3[] Projected0 = Series.GetPositionInAllTilts(Points[0]).Select(v => v / new float3(Series.ImageDimensionsPhysical.X, Series.ImageDimensionsPhysical.Y, 1)).ToArray();
                    float3[] Projected1 = Series.GetPositionInAllTilts(Points[1]).Select(v => v / new float3(Series.ImageDimensionsPhysical.X, Series.ImageDimensionsPhysical.Y, 1)).ToArray();

                    for (int t = 0; t < Series.NTilts; t++)
                    {
                        float Interp0 = TiltMovies[t].GridCTFDefocus.GetInterpolated(new float3(Projected0[t].X, Projected0[0].Y, 0.5f));
                        float Interp1 = TiltMovies[t].GridCTFDefocus.GetInterpolated(new float3(Projected1[t].X, Projected1[0].Y, 0.5f));
                        GradientsEstimated[t] = Interp1 - Interp0;

                        GradientsAssumed[t] = Projected1[t].Z - Projected0[t].Z;
                    }

                    if (GradientsEstimated.Length > 1)
                    {
                        GradientsEstimated = MathHelper.Normalize(GradientsEstimated);
                        GradientsAssumed = MathHelper.Normalize(GradientsAssumed);
                    }
                    else
                    {
                        GradientsEstimated[0] = Math.Sign(GradientsEstimated[0]);
                        GradientsAssumed[0] = Math.Sign(GradientsAssumed[0]);
                    }

                    float Correlation = MathHelper.DotProduct(GradientsEstimated, GradientsAssumed) / GradientsEstimated.Length;

                    if (Dialog.IsOpen)
                        await Dialog.CloseAsync();

                    if (Correlation > 0)
                        await Dispatcher.Invoke(async () =>
                        {
                            await this.ShowMessageAsync("", $"It looks like the angles are in accord with the estimated defocus gradients. Correlation = {Correlation:F2}");
                        });
                    else
                    {
                        bool DoFlip = false;

                        await Dispatcher.Invoke(async () =>
                        {
                            var Result = await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("You're in the Upside Down!",
                                                                                                             $"It looks like the defocus handedness should be flipped. Correlation = {Correlation:F2}\n" +
                                                                                                             "Would you like to flip it for all tilt series currently loaded?\n" +
                                                                                                             "You should probably repeat CTF estimation after flipping.",
                                                                                                             MessageDialogStyle.AffirmativeAndNegative);
                            if (Result == MessageDialogResult.Affirmative)
                                DoFlip = true;
                        });

                        if (DoFlip)
                        {
                            await Dispatcher.Invoke(async () => Dialog = await this.ShowProgressAsync("Please wait...", $"Saving tilt series metadata..."));

                            TiltSeries[] AllSeries = FileDiscoverer.GetImmutableFiles().Select(m => (TiltSeries)m).ToArray();
                            Dispatcher.Invoke(() => Dialog.Maximum = AllSeries.Length);

                            for (int i = 0; i < AllSeries.Length; i++)
                            {
                                AllSeries[i].AreAnglesInverted = !AllSeries[i].AreAnglesInverted;
                                AllSeries[i].SaveMeta();

                                Dialog.SetProgress(i + 1);
                            }
                        }
                    }
                }
                catch (Exception exc)
                {
                    await Dispatcher.Invoke(async () =>
                    {
                        if (Dialog.IsOpen)
                            await Dialog.CloseAsync();

                        await this.ShowMessageAsync("Oopsie", exc.ToString());
                    });
                }
            });

            if (Dialog.IsOpen)
                await Dialog.CloseAsync();
        }

        #endregion

        private void TabsProcessingView_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ProcessingStatusBar == null)
                return;

            if (TabsProcessingView.SelectedItem == TabProcessingOverview)
                ProcessingStatusBar.Visibility = Visibility.Collapsed;
            else
                ProcessingStatusBar.Visibility = Visibility.Visible;
        }

        #endregion
        
        #region Helper methods

        public void AdjustInput()
        {
            FileDiscoverer.ChangePath(Options.Import.DataFolder, Options.Import.ProcessingFolder, Options.Import.Extension, Options.Import.DoRecursiveSearch && !string.IsNullOrEmpty(Options.Import.ProcessingFolder));
        }

        public static Image LoadAndPrepareGainReference()
        {
            OptionsWarp Options = WarpRuntime.MainWindow.Options;

            Image Gain = Image.FromFilePatient(50, 500,
                                               Options.Import.GainPath,
                                               new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                               (int)Options.Import.HeaderlessOffset,
                                               ImageFormatsHelper.StringToType(Options.Import.HeaderlessType));

            float Mean = MathHelper.Mean(Gain.GetHost(Intent.Read)[0]);
            if (Mean == 0)
                Mean = 1;
            Gain.TransformValues(v => v == 0 ? 1 : v / Mean);

            if (Options.Import.GainFlipX)
                Gain = Gain.AsFlippedX();
            if (Options.Import.GainFlipY)
                Gain = Gain.AsFlippedY();
            if (Options.Import.GainTranspose)
                Gain = Gain.AsTransposed();

            return Gain;
        }

        public static DefectModel LoadAndPrepareDefectMap()
        {
            OptionsWarp Options = WarpRuntime.MainWindow.Options;

            Image Defects = Image.FromFilePatient(50, 500,
                                                  Options.Import.DefectsPath,
                                                  new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                  (int)Options.Import.HeaderlessOffset,
                                                  ImageFormatsHelper.StringToType(Options.Import.HeaderlessType));

            if (Options.Import.GainFlipX)
                Defects = Defects.AsFlippedX();
            if (Options.Import.GainFlipY)
                Defects = Defects.AsFlippedY();
            if (Options.Import.GainTranspose)
                Defects = Defects.AsTransposed();

            DefectModel Model = new DefectModel(Defects, 4);
            Defects.Dispose();

            return Model;
        }

        public static void LoadAndPrepareHeaderAndMap(string path, Image imageGain, DefectModel defectMap, decimal scaleFactor, out MapHeader header, out Image stack, bool needStack = true, int maxThreads = 8)
        {
            OptionsWarp Options = WarpRuntime.MainWindow.Options;

            HeaderEER.GroupNFrames = Options.Import.EERGroupFrames;

            header = MapHeader.ReadFromFilePatient(50, 500,
                                                   path,
                                                   new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                   Options.Import.HeaderlessOffset,
                                                   ImageFormatsHelper.StringToType(Options.Import.HeaderlessType));

            string Extension = Helper.PathToExtension(path).ToLower();
            bool IsTiff = header.GetType() == typeof(HeaderTiff);
            bool IsEER = header.GetType() == typeof(HeaderEER);

            if (imageGain != null)
                if (!IsEER)
                    if (header.Dimensions.X != imageGain.Dims.X || header.Dimensions.Y != imageGain.Dims.Y)
                        throw new Exception("Gain reference dimensions do not match image.");

            int EERSupersample = 1;
            if (imageGain != null && IsEER)
            {
                if (header.Dimensions.X == imageGain.Dims.X)
                    EERSupersample = 1;
                else if (header.Dimensions.X * 2 == imageGain.Dims.X)
                    EERSupersample = 2;
                else if (header.Dimensions.X * 4 == imageGain.Dims.X)
                    EERSupersample = 3;
                else
                    throw new Exception("Invalid supersampling factor requested for EER based on gain reference dimensions");
            }
            int EERGroupFrames = 1;
            if (IsEER)
            {
                if (HeaderEER.GroupNFrames > 0)
                    EERGroupFrames = HeaderEER.GroupNFrames;
                else if (HeaderEER.GroupNFrames < 0)
                {
                    int NFrames = -HeaderEER.GroupNFrames;
                    EERGroupFrames = header.Dimensions.Z / NFrames;
                }

                header.Dimensions.Z /= EERGroupFrames;
            }

            HeaderEER.SuperResolution = EERSupersample;

            if (IsEER && imageGain != null)
            {
                header.Dimensions.X = imageGain.Dims.X;
                header.Dimensions.Y = imageGain.Dims.Y;
            }
            MapHeader Header = header;

            int NThreads = (IsTiff || IsEER) ? 6 : 2;

            int CurrentDevice = GPU.GetDevice();

            if (needStack)
            {
                byte[] TiffBytes = null;
                if (IsTiff)
                {
                    MemoryStream Stream = new MemoryStream();
                    using (Stream BigBufferStream = IOHelper.OpenWithBigBuffer(path))
                        BigBufferStream.CopyTo(Stream);
                    TiffBytes = Stream.GetBuffer();
                }

                if (scaleFactor == 1M)
                {
                    stack = new Image(header.Dimensions);
                    float[][] OriginalStackData = stack.GetHost(Intent.Write);

                    Helper.ForCPU(0, header.Dimensions.Z, NThreads, threadID => GPU.SetDevice(CurrentDevice), (z, threadID) =>
                    {
                        Image Layer = null;
                        MemoryStream TiffStream = TiffBytes != null ? new MemoryStream(TiffBytes) : null;

                        if (!IsEER)
                            Layer = Image.FromFilePatient(50, 500,
                                                        path,
                                                        new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                        (int)Options.Import.HeaderlessOffset,
                                                        ImageFormatsHelper.StringToType(Options.Import.HeaderlessType),
                                                        z,
                                                        TiffStream);
                        else
                        {
                            Layer = new Image(Header.Dimensions.Slice());
                            EERNative.ReadEERPatient(50, 500,
                                                     path, z * EERGroupFrames, Math.Min(((HeaderEER)Header).DimensionsUngrouped.Z, (z + 1) * EERGroupFrames), EERSupersample, Layer.GetHost(Intent.Write)[0]);
                        }

                        lock (OriginalStackData)
                        {
                            if (imageGain != null)
                            {
                                if (IsEER)
                                    Layer.DivideSlices(imageGain);
                                else
                                    Layer.MultiplySlices(imageGain);
                            }

                            if (defectMap != null)
                            {
                                Image LayerCopy = Layer.GetCopyGPU();
                                defectMap.Correct(LayerCopy, Layer);
                                LayerCopy.Dispose();
                            }

                            Layer.Xray(20f);

                            OriginalStackData[z] = Layer.GetHost(Intent.Read)[0];
                            Layer.Dispose();
                        }

                    }, null);
                }
                else
                {
                    int3 ScaledDims = new int3((int)Math.Round(header.Dimensions.X * scaleFactor) / 2 * 2,
                                               (int)Math.Round(header.Dimensions.Y * scaleFactor) / 2 * 2,
                                               header.Dimensions.Z);

                    stack = new Image(ScaledDims);
                    float[][] OriginalStackData = stack.GetHost(Intent.Write);

                    int PlanForw = GPU.CreateFFTPlan(header.Dimensions.Slice(), 1);
                    int PlanBack = GPU.CreateIFFTPlan(ScaledDims.Slice(), 1);

                    Helper.ForCPU(0, ScaledDims.Z, NThreads, threadID => GPU.SetDevice(CurrentDevice), (z, threadID) =>
                    {
                        Image Layer = null;
                        MemoryStream TiffStream = TiffBytes != null ? new MemoryStream(TiffBytes) : null;

                        if (!IsEER)
                            Layer = Image.FromFilePatient(50, 500,
                                                        path,
                                                        new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                        (int)Options.Import.HeaderlessOffset,
                                                        ImageFormatsHelper.StringToType(Options.Import.HeaderlessType),
                                                        z,
                                                        TiffStream);
                        else
                        {
                            Layer = new Image(Header.Dimensions.Slice());
                            EERNative.ReadEERPatient(50, 500,
                                path, z * EERGroupFrames, Math.Min(((HeaderEER)Header).DimensionsUngrouped.Z, (z + 1) * EERGroupFrames), EERSupersample, Layer.GetHost(Intent.Write)[0]);
                        }

                        Image ScaledLayer = null;
                        lock (OriginalStackData)
                        {
                            if (imageGain != null)
                            {
                                if (IsEER)
                                    Layer.DivideSlices(imageGain);
                                else
                                    Layer.MultiplySlices(imageGain);
                            }

                            if (defectMap != null)
                            {
                                Image LayerCopy = Layer.GetCopyGPU();
                                defectMap.Correct(LayerCopy, Layer);
                                LayerCopy.Dispose();
                            }

                            Layer.Xray(20f);

                            ScaledLayer = Layer.AsScaled(new int2(ScaledDims), PlanForw, PlanBack);
                            Layer.Dispose();
                        }

                        OriginalStackData[z] = ScaledLayer.GetHost(Intent.Read)[0];
                        ScaledLayer.Dispose();

                    }, null);

                    GPU.DestroyFFTPlan(PlanForw);
                    GPU.DestroyFFTPlan(PlanBack);
                }
            }
            else
            {
                stack = null;
            }
        }

        public List<int> GetDeviceList()
        {
            List<int> Devices = new List<int>();
            Dispatcher.Invoke(() =>
            {
                for (int i = 0; i < CheckboxesGPUStats.Length; i++)
                    if ((bool)CheckboxesGPUStats[i].IsChecked)
                        Devices.Add(i);
            });

            return Devices;
        }

        #endregion

        
    }

    public static class WarpRuntime
    {
        public static MainWindow MainWindow;
    }

    public class NyquistAngstromConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (value == null || WarpRuntime.MainWindow == null || WarpRuntime.MainWindow.Options == null)
                return null;

            return WarpRuntime.MainWindow.Options.Import.BinnedPixelSize * 2 / (decimal)value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (value == null || WarpRuntime.MainWindow == null || WarpRuntime.MainWindow.Options == null)
                return null;

            return WarpRuntime.MainWindow.Options.Import.BinnedPixelSize * 2 / (decimal)value;
        }
    }

    public class ActionCommand : ICommand
    {
        private readonly Action _action;

        public ActionCommand(Action action)
        {
            _action = action;
        }

        public void Execute(object parameter)
        {
            _action();
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public event EventHandler CanExecuteChanged;
    }
}
