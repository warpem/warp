using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
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
using System.Windows.Threading;
using Warp;
using Warp.Headers;
using Warp.Tools;
using Image = Warp.Image;
using MathHelper = Warp.Tools.MathHelper;

namespace Cube
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : MahApps.Metro.Controls.MetroWindow
    {
        public static Options Options = new Options();
        bool FreezeUpdates = false;

        ObservableCollection<Particle> Particles = new ObservableCollection<Particle>();
        ObservableCollection<Particle> DeletedParticles = new ObservableCollection<Particle>();

        List<Particle>[] SliceXYParticles = null;
        List<Particle>[] SliceZYParticles = null;
        List<Particle>[] SliceXZParticles = null;
        
        readonly SolidColorBrush ParticleBrush, ActiveParticleBrush;

        float PixelSize = 1f;
        Image Tomogram;
        float[] TomogramContinuous;

        int2 DragStart = new int2(0, 0);
        bool IsDragging = false;
        Particle DraggingParticle = null;

        IEnumerable<Particle> GoodParticles => Particles.Where(p => p.Score >= (float)Options.ParticleScoreMin && p.Score <= (float)Options.ParticleScoreMax);
        IEnumerable<Particle> BadParticles
        {
            get
            {
                List<Particle> Result = new List<Particle>();
                Result.AddRange(Particles.Where(p => p.Score < (float)Options.ParticleScoreMin || p.Score > (float)Options.ParticleScoreMax));
                Result.AddRange(DeletedParticles);

                return Result;
            }
        }

        public MainWindow()
        {
            InitializeComponent();

            GridDisplay.Visibility = Visibility.Collapsed;

            DataContext = Options;
            Options.PropertyChanged += Options_PropertyChanged;

            PreviewMouseUp += MainWindow_PreviewMouseUp;

            Particles.CollectionChanged += Particles_CollectionChanged;

            if (File.Exists("previous.settings"))
                Options.Load("previous.settings");

            ParticleBrush = new SolidColorBrush(Colors.Lime);
            ParticleBrush.Freeze();
            ActiveParticleBrush = new SolidColorBrush(Colors.Fuchsia);
            ActiveParticleBrush.Freeze();
        }

        private void MainWindow_PreviewMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Middle)
                IsDragging = false;
        }

        private void MainWindow_OnPreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Delete)
            {
                if (Options.ActiveParticle != null)
                {
                    int OldIndex = Particles.IndexOf(Options.ActiveParticle);
                    FreezeUpdates = true;
                    DeletedParticles.Add(Options.ActiveParticle);
                    Particles.Remove(Options.ActiveParticle);
                    FreezeUpdates = false;

                    if (OldIndex > 0)
                        Options.ActiveParticle = Particles[OldIndex - 1];
                    else if (OldIndex < Particles.Count)
                        Options.ActiveParticle = Particles[OldIndex];
                    else
                        Options.ActiveParticle = null;

                    if (Options.ActiveParticle != null)
                        CenterOn(new int3(Options.ActiveParticle.Position));
                }

                e.Handled = true;
            }
            else if (e.Key == Key.Left || e.Key == Key.Right)
            {
                if (Particles.Count < 1)
                    return;

                if (Options.ActiveParticle == null)
                {
                    Options.ActiveParticle = Particles[0];
                }
                else
                {
                    int Delta = e.Key == Key.Left ? -1 : 1;
                    int NewIndex = (Particles.IndexOf(Options.ActiveParticle) + Delta + Particles.Count) % Particles.Count;
                    Options.ActiveParticle = Particles[NewIndex];
                }

                CenterOn(new int3(Options.ActiveParticle.Position));

                e.Handled = true;
            }
        }

        private void Options_PropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == "PathTomogram")
            {
                ButtonTomogramPathText.Text = Options.PathTomogram != "" ? Options.PathTomogram : "Select tomogram...";
            }

            if (FreezeUpdates)
                return;

            if (e.PropertyName == "InputLowpass")
            {
                UpdateTomogramPlanes();
                UpdateParticlePlanes();
            }
            else if (e.PropertyName == "InputAverageSlices")
            {
                UpdateTomogramPlanes();
                UpdateParticlePlanes();
                UpdateCrosshairs();
            }
            else if (e.PropertyName == "DisplayIntensityMin" || e.PropertyName == "DisplayIntensityMax")
            {
                UpdateTomogramPlanes();
                UpdateParticlePlanes();
            }
            else if (e.PropertyName == "ZoomLevel")
            {
                UpdateTomogramPlanes();
                UpdateView();
                UpdateBoxes();
            }
            else if (e.PropertyName == "PlaneX")
            {
                UpdateTomogramZY();
                UpdateView();
                UpdateBoxesZY();
            }
            else if (e.PropertyName == "PlaneY")
            {
                UpdateTomogramXZ();
                UpdateView();
                UpdateBoxesXZ();
            }
            else if (e.PropertyName == "PlaneZ")
            {
                UpdateTomogramXY();
                UpdateView();
                UpdateBoxesXY();
            }
            else if (e.PropertyName == "ViewX" || e.PropertyName == "ViewY" || e.PropertyName == "ViewZ")
            {
                UpdateView();
            }
            else if (e.PropertyName == "MouseX" || e.PropertyName == "MouseY" || e.PropertyName == "MouseZ")
            {
                UpdateCrosshairs();
                MousePositionChanged();
            }
            else if (e.PropertyName == "ParticlePlaneX")
            {
                if (Options.ActiveParticle != null)
                    MoveParticle(Options.ActiveParticle, new float3(Options.ParticlePlaneX,
                                                                  Options.ActiveParticle.Position.Y,
                                                                  Options.ActiveParticle.Position.Z));
            }
            else if (e.PropertyName == "ParticlePlaneY")
            {
                if (Options.ActiveParticle != null)
                    MoveParticle(Options.ActiveParticle, new float3(Options.ActiveParticle.Position.X,
                                                                    Options.ParticlePlaneY,
                                                                    Options.ActiveParticle.Position.Z));
            }
            else if (e.PropertyName == "ParticlePlaneZ")
            {
                if (Options.ActiveParticle != null)
                    MoveParticle(Options.ActiveParticle, new float3(Options.ActiveParticle.Position.X,
                                                                    Options.ActiveParticle.Position.Y,
                                                                    Options.ParticlePlaneZ));
            }
            else if (e.PropertyName == "BoxSize")
            {
                UpdateBoxes();
                UpdateParticlePlanes();
            }
            else if (e.PropertyName == "ActiveParticle")
            {
                ActiveParticleChanged();
            }
            else if (e.PropertyName == "CentralBlob" || e.PropertyName == "IsosurfaceThreshold")
            {
            }
            else if (e.PropertyName == "ParticleScoreMin" || e.PropertyName == "ParticleScoreMax")
            {
                Options.NParticles = GoodParticles.Count();
                UpdateBoxes();
            }
        }

        private void CanvasXY_OnSizeChanged(object sender, SizeChangedEventArgs e)
        {
            UpdateView();
        }

        private void MainWindow_OnClosing(object sender, CancelEventArgs e)
        {
            try
            {
                Options.Save("previous.settings");
            }
            catch (Exception)
            {
                // ignored
            }
        }

        private void Particles_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            if (SliceXYParticles == null || SliceZYParticles == null || SliceXZParticles == null)
                return;

            Options.NParticles = GoodParticles.Count();

            if (e.OldItems != null)
                foreach (var oldItem in e.OldItems)
                {
                    Particle OldParticle = (Particle)oldItem;

                    if (SliceXYParticles[(int)OldParticle.Position.Z].Contains(OldParticle))
                        SliceXYParticles[(int)OldParticle.Position.Z].Remove(OldParticle);
                    if (SliceZYParticles[(int)OldParticle.Position.X].Contains(OldParticle))
                        SliceZYParticles[(int)OldParticle.Position.X].Remove(OldParticle);
                    if (SliceXZParticles[(int)OldParticle.Position.Y].Contains(OldParticle))
                        SliceXZParticles[(int)OldParticle.Position.Y].Remove(OldParticle);

                    OldParticle.PropertyChanged -= Particle_PropertyChanged;
                }

            if (e.NewItems != null)
                foreach (var newItem in e.NewItems)
                {
                    Particle NewParticle = (Particle)newItem;

                    SliceXYParticles[(int)NewParticle.Position.Z].Add(NewParticle);
                    SliceZYParticles[(int)NewParticle.Position.X].Add(NewParticle);
                    SliceXZParticles[(int)NewParticle.Position.Y].Add(NewParticle);

                    NewParticle.PropertyChanged += Particle_PropertyChanged;
                }

            if (e.Action == NotifyCollectionChangedAction.Reset)
            {
                foreach (var slice in SliceXYParticles)
                    slice.Clear();
                foreach (var slice in SliceZYParticles)
                    slice.Clear();
                foreach (var slice in SliceXZParticles)
                    slice.Clear();
            }

            if (!FreezeUpdates)
                UpdateBoxes();
        }

        private void Particle_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (sender == Options.ActiveParticle)
            {
                if (e.PropertyName == "Position")
                {
                    Options.ParticlePlaneX = (int)Options.ActiveParticle.Position.X;
                    Options.ParticlePlaneY = (int)Options.ActiveParticle.Position.Y;
                    Options.ParticlePlaneZ = (int)Options.ActiveParticle.Position.Z;
                }
            }
        }

        private void LoadTomogram(string path)
        {
            if (!File.Exists(path))
                return;

            //GridDisplay.Visibility = Visibility.Collapsed;

            HeaderMRC Header = (HeaderMRC)MapHeader.ReadFromFile(path);
            PixelSize = Header.PixelSize.X;

            Tomogram = Image.FromFile(path);

            TomogramContinuous = Tomogram.GetHostContinuousCopy();

            FreezeUpdates = true;

            Options.PlaneX = Tomogram.Dims.X / 2;
            Options.PlaneY = Tomogram.Dims.Y / 2;
            Options.PlaneZ = Tomogram.Dims.Z / 2;

            Options.ViewX = Tomogram.Dims.X / 2;
            Options.ViewY = Tomogram.Dims.Y / 2;
            Options.ViewZ = Tomogram.Dims.Z / 2;

            Particles.Clear();
            DeletedParticles.Clear();
            Options.ActiveParticle = null;

            SliceXYParticles = new List<Particle>[Tomogram.Dims.Z];
            for (int i = 0; i < SliceXYParticles.Length; i++)
                SliceXYParticles[i] = new List<Particle>();
            SliceZYParticles = new List<Particle>[Tomogram.Dims.X];
            for (int i = 0; i < SliceZYParticles.Length; i++)
                SliceZYParticles[i] = new List<Particle>();
            SliceXZParticles = new List<Particle>[Tomogram.Dims.Y];
            for (int i = 0; i < SliceXZParticles.Length; i++)
                SliceXZParticles[i] = new List<Particle>();

            Dispatcher.Invoke(() =>
            {
                SliderPlaneX.MaxValue = Tomogram.Dims.X - 1;
                SliderPlaneY.MaxValue = Tomogram.Dims.Y - 1;
                SliderPlaneZ.MaxValue = Tomogram.Dims.Z - 1;

                SliderParticleX.MaxValue = Tomogram.Dims.X - 1;
                SliderParticleY.MaxValue = Tomogram.Dims.Y - 1;
                SliderParticleZ.MaxValue = Tomogram.Dims.Z - 1;

                FreezeUpdates = false;

                UpdateTomogramPlanes();
                UpdateView();
                UpdateBoxes();

                GridDisplay.Visibility = Visibility.Visible;
            });
        }

        #region Tomogram planes

        private void UpdateTomogramPlanes()
        {
            if (Tomogram == null)
                return;

            Task TaskXY = new Task(UpdateTomogramXY);
            Task TaskZY = new Task(UpdateTomogramZY);
            Task TaskXZ = new Task(UpdateTomogramXZ);

            TaskXY.Start();
            TaskZY.Start();
            TaskXZ.Start();

            TaskXY.Wait();
            TaskZY.Wait();
            TaskXZ.Wait();
        }

        private void UpdateTomogramXY()
        {
            if (Tomogram == null)
                return;

            int PlaneElements = Tomogram.Dims.X * Tomogram.Dims.Y;
            float[] Data = new float[PlaneElements];

            unsafe
            {
                fixed (float* DataPtr = Data)
                fixed (float* FilteredPtr = TomogramContinuous)
                {
                    int FirstSlice = Math.Max(0, Options.PlaneZ - Options.InputAverageSlices / 2);
                    int LastSlice = Math.Min(Tomogram.Dims.Z, FirstSlice + Options.InputAverageSlices);
                    for (int s = FirstSlice; s < LastSlice; s++)
                    {
                        float* DataP = DataPtr;
                        float* FilteredP = FilteredPtr + s * PlaneElements; // Offset to the right Z
                        for (int i = 0; i < PlaneElements; i++)
                            *DataP++ += *FilteredP++;
                    }
                }
            }

            ImageSource Result = GetImage(Data, new int2(Tomogram.Dims.X, Tomogram.Dims.Y));
            Result.Freeze();
            Dispatcher.InvokeAsync(() => ImageXY.Source = Result);
        }

        private void UpdateTomogramZY()
        {
            if (Tomogram == null)
                return;

            int PlaneElements = Tomogram.Dims.Z * Tomogram.Dims.Y;
            int XYElements = Tomogram.Dims.X * Tomogram.Dims.Y;
            float[] Data = new float[PlaneElements];

            unsafe
            {
                fixed (float* DataPtr = Data)
                fixed (float* FilteredPtr = TomogramContinuous)
                {
                    int FirstSlice = Math.Max(0, Options.PlaneX - Options.InputAverageSlices / 2);
                    int LastSlice = Math.Min(Tomogram.Dims.X, FirstSlice + Options.InputAverageSlices);
                    for (int s = FirstSlice; s < LastSlice; s++)
                    {
                        float* DataP = DataPtr;
                        for (int y = 0; y < Tomogram.Dims.Y; y++)
                        {
                            float* FilteredP = FilteredPtr + y * Tomogram.Dims.X + s;
                            for (int z = 0; z < Tomogram.Dims.Z; z++)
                                *DataP++ += FilteredP[z * XYElements];
                        }
                    }
                }
            }

            ImageSource Result = GetImage(Data, new int2(Tomogram.Dims.Z, Tomogram.Dims.Y));
            Result.Freeze();
            Dispatcher.InvokeAsync(() => ImageZY.Source = Result);
        }

        private void UpdateTomogramXZ()
        {
            if (Tomogram == null)
                return;

            int PlaneElements = Tomogram.Dims.X * Tomogram.Dims.Z;
            int XYElements = Tomogram.Dims.X * Tomogram.Dims.Y;
            float[] Data = new float[PlaneElements];

            unsafe
            {
                fixed (float* DataPtr = Data)
                fixed (float* FilteredPtr = TomogramContinuous)
                {
                    int FirstSlice = Math.Max(0, Options.PlaneY - Options.InputAverageSlices / 2);
                    int LastSlice = Math.Min(Tomogram.Dims.Y, FirstSlice + Options.InputAverageSlices);
                    for (int s = FirstSlice; s < LastSlice; s++)
                    {
                        float* DataP = DataPtr;
                        for (int z = 0; z < Tomogram.Dims.Z; z++)
                        {
                            float* FilteredP = FilteredPtr + z * XYElements + s * Tomogram.Dims.X;
                            for (int x = 0; x < Tomogram.Dims.X; x++)
                                *DataP++ += FilteredP[x];
                        }
                    }
                }
            }

            ImageSource Result = GetImage(Data, new int2(Tomogram.Dims.X, Tomogram.Dims.Z));
            Result.Freeze();
            Dispatcher.InvokeAsync(() => ImageXZ.Source = Result);
        }

        #endregion

        #region Particle planes

        private void UpdateParticlePlanes()
        {
            UpdateParticleXY();
            UpdateParticleZY();
            UpdateParticleXZ();
        }

        private void UpdateParticleXY()
        {
            if (Options.ActiveParticle == null || Tomogram == null)
                return;

            int BoxSize = Options.BoxSize;
            int BoxSizeHalf = BoxSize / 2;
            int PlaneElements = BoxSize * BoxSize;
            int XYElements = Tomogram.Dims.X * Tomogram.Dims.Y;
            float[] Data = new float[PlaneElements];

            int3 Pos = new int3(Options.ActiveParticle.Position);

            unsafe
            {
                fixed (float* DataPtr = Data)
                fixed (float* FilteredPtr = TomogramContinuous)
                {
                    int FirstSlice = Pos.Z - Options.InputAverageSlices / 2;
                    int LastSlice = FirstSlice + Options.InputAverageSlices;
                    FirstSlice = Math.Max(0, FirstSlice);
                    LastSlice = Math.Min(Tomogram.Dims.Z, LastSlice);

                    for (int s = FirstSlice; s < LastSlice; s++)
                    {
                        float* DataP = DataPtr;
                        float* FilteredP = FilteredPtr + s * XYElements;
                        for (int y = Pos.Y - BoxSizeHalf; y < Pos.Y + BoxSizeHalf; y++)
                        {
                            int yy = Math.Max(0, Math.Min(y, Tomogram.Dims.Y - 1));

                            for (int x = Pos.X - BoxSizeHalf; x < Pos.X + BoxSizeHalf; x++)
                            {
                                int xx = Math.Max(0, Math.Min(x, Tomogram.Dims.X - 1));
                                *DataP++ += FilteredP[yy * Tomogram.Dims.X + xx];
                            }
                        }
                    }
                }
            }

            ImageSource Result = GetImage(Data, new int2(BoxSize, BoxSize));
            Result.Freeze();
            Dispatcher.InvokeAsync(() => ImageParticleXY.Source = Result);
        }

        private void UpdateParticleZY()
        {
            if (Options.ActiveParticle == null || Tomogram == null)
                return;

            int BoxSize = Options.BoxSize;
            int BoxSizeHalf = BoxSize / 2;
            int PlaneElements = BoxSize * BoxSize;
            int XYElements = Tomogram.Dims.X * Tomogram.Dims.Y;
            float[] Data = new float[PlaneElements];

            int3 Pos = new int3(Options.ActiveParticle.Position);

            unsafe
            {
                fixed (float* DataPtr = Data)
                fixed (float* FilteredPtr = TomogramContinuous)
                {
                    int FirstSlice = Pos.X - Options.InputAverageSlices / 2;
                    int LastSlice = FirstSlice + Options.InputAverageSlices;
                    FirstSlice = Math.Max(0, FirstSlice);
                    LastSlice = Math.Min(Tomogram.Dims.X, LastSlice);

                    for (int s = FirstSlice; s < LastSlice; s++)
                    {
                        float* DataP = DataPtr;
                        for (int y = Pos.Y - BoxSizeHalf; y < Pos.Y + BoxSizeHalf; y++)
                        {
                            int yy = Math.Max(0, Math.Min(y, Tomogram.Dims.Y - 1));
                            float* FilteredP = FilteredPtr + yy * Tomogram.Dims.X + s;

                            for (int z = Pos.Z - BoxSizeHalf; z < Pos.Z + BoxSizeHalf; z++)
                            {
                                int zz = Math.Max(0, Math.Min(z, Tomogram.Dims.Z - 1));
                                *DataP++ += FilteredP[zz * XYElements];
                            }
                        }
                    }
                }
            }

            ImageSource Result = GetImage(Data, new int2(BoxSize, BoxSize));
            Result.Freeze();
            Dispatcher.InvokeAsync(() => ImageParticleZY.Source = Result);
        }

        private void UpdateParticleXZ()
        {
            if (Options.ActiveParticle == null || Tomogram == null)
                return;

            int BoxSize = Options.BoxSize;
            int BoxSizeHalf = BoxSize / 2;
            int PlaneElements = BoxSize * BoxSize;
            int XYElements = Tomogram.Dims.X * Tomogram.Dims.Y;
            float[] Data = new float[PlaneElements];

            int3 Pos = new int3(Options.ActiveParticle.Position);

            unsafe
            {
                fixed (float* DataPtr = Data)
                fixed (float* FilteredPtr = TomogramContinuous)
                {
                    int FirstSlice = Pos.Y - Options.InputAverageSlices / 2;
                    int LastSlice = FirstSlice + Options.InputAverageSlices;
                    FirstSlice = Math.Max(0, FirstSlice);
                    LastSlice = Math.Min(Tomogram.Dims.Y, LastSlice);

                    for (int s = FirstSlice; s < LastSlice; s++)
                    {
                        float* DataP = DataPtr;
                        float* FilteredP = FilteredPtr + s * Tomogram.Dims.X;
                        for (int z = Pos.Z - BoxSizeHalf; z < Pos.Z + BoxSizeHalf; z++)
                        {
                            int zz = Math.Max(0, Math.Min(z, Tomogram.Dims.Z - 1));

                            for (int x = Pos.X - BoxSizeHalf; x < Pos.X + BoxSizeHalf; x++)
                            {
                                int xx = Math.Max(0, Math.Min(x, Tomogram.Dims.X - 1));
                                *DataP++ += FilteredP[zz * XYElements + xx];
                            }
                        }
                    }
                }
            }

            ImageSource Result = GetImage(Data, new int2(BoxSize, BoxSize));
            Result.Freeze();
            Dispatcher.InvokeAsync(() => ImageParticleXZ.Source = Result);
        }

        #endregion

        #region View & overlays

        private void UpdateView()
        {
            // XY
            {
                float2 CanvasCenter = new float2((float)CanvasXY.ActualWidth / 2, (float)CanvasXY.ActualHeight / 2);
                float2 ImageCenter = new float2((float)(Options.ViewX * Options.ZoomLevel), (float)(Options.ViewY * Options.ZoomLevel));
                float2 Delta = CanvasCenter - ImageCenter;

                Canvas.SetLeft(ImageXY, Math.Round(Delta.X));
                Canvas.SetBottom(ImageXY, Math.Round(Delta.Y));

                Canvas.SetLeft(CanvasOverlayXY, Math.Round(Delta.X));
                Canvas.SetBottom(CanvasOverlayXY, Math.Round(Delta.Y));

                Canvas.SetLeft(CanvasParticlesXY, Math.Round(Delta.X));
                Canvas.SetBottom(CanvasParticlesXY, Math.Round(Delta.Y));
            }
            // ZY
            {
                float2 CanvasCenter = new float2((float)CanvasZY.ActualWidth / 2, (float)CanvasZY.ActualHeight / 2);
                float2 ImageCenter = new float2((float)(Options.ViewZ * Options.ZoomLevel), (float)(Options.ViewY * Options.ZoomLevel));
                float2 Delta = CanvasCenter - ImageCenter;

                Canvas.SetLeft(ImageZY, Math.Round(Delta.X));
                Canvas.SetBottom(ImageZY, Math.Round(Delta.Y));

                Canvas.SetLeft(CanvasOverlayZY, Math.Round(Delta.X));
                Canvas.SetBottom(CanvasOverlayZY, Math.Round(Delta.Y));

                Canvas.SetLeft(CanvasParticlesZY, Math.Round(Delta.X));
                Canvas.SetBottom(CanvasParticlesZY, Math.Round(Delta.Y));
            }
            // XZ
            {
                float2 CanvasCenter = new float2((float)CanvasXZ.ActualWidth / 2, (float)CanvasXZ.ActualHeight / 2);
                float2 ImageCenter = new float2((float)(Options.ViewX * Options.ZoomLevel), (float)(Options.ViewZ * Options.ZoomLevel));
                float2 Delta = CanvasCenter - ImageCenter;

                Canvas.SetLeft(ImageXZ, Math.Round(Delta.X));
                Canvas.SetBottom(ImageXZ, Math.Round(Delta.Y));

                Canvas.SetLeft(CanvasOverlayXZ, Math.Round(Delta.X));
                Canvas.SetBottom(CanvasOverlayXZ, Math.Round(Delta.Y));

                Canvas.SetLeft(CanvasParticlesXZ, Math.Round(Delta.X));
                Canvas.SetBottom(CanvasParticlesXZ, Math.Round(Delta.Y));
            }
        }

        private void UpdateCrosshairs()
        {
            // XY
            {
                CrosshairXYX.X1 = 0;
                CrosshairXYX.X2 = 0;
                CrosshairXYX.Y1 = 0;
                CrosshairXYX.Y2 = ImageXY.ActualHeight;

                //CrosshairXYX.StrokeThickness = Math.Max(1, Options.InputAverageSlices * (double)Options.ZoomLevel);
                CrosshairXYX.Opacity = CrosshairXYX.StrokeThickness == 1 ? 0.8 : 0.2;
                Canvas.SetBottom(CrosshairXYX, 0);
                Canvas.SetLeft(CrosshairXYX, (int)((Options.MouseX) * Options.ZoomLevel));

                CrosshairXYY.X1 = 0;
                CrosshairXYY.X2 = ImageXY.ActualWidth;
                CrosshairXYY.Y1 = 0;
                CrosshairXYY.Y2 = 0;

                //CrosshairXYY.StrokeThickness = Math.Max(1, Options.InputAverageSlices * (double)Options.ZoomLevel);
                CrosshairXYY.Opacity = CrosshairXYX.StrokeThickness == 1 ? 0.8 : 0.2;
                Canvas.SetBottom(CrosshairXYY, (int)((Options.MouseY) * Options.ZoomLevel));
                Canvas.SetLeft(CrosshairXYY, 0);
            }
            // ZY
            {
                CrosshairZYZ.X1 = 0;
                CrosshairZYZ.X2 = 0;
                CrosshairZYZ.Y1 = 0;
                CrosshairZYZ.Y2 = ImageZY.ActualHeight;

                //CrosshairZYZ.StrokeThickness = Math.Max(1, Options.InputAverageSlices * (double)Options.ZoomLevel);
                CrosshairZYZ.Opacity = CrosshairXYX.StrokeThickness == 1 ? 0.8 : 0.2;
                Canvas.SetBottom(CrosshairZYZ, 0);
                Canvas.SetLeft(CrosshairZYZ, (int)((Options.MouseZ) * Options.ZoomLevel));

                CrosshairZYY.X1 = 0;
                CrosshairZYY.X2 = ImageZY.ActualWidth;
                CrosshairZYY.Y1 = 0;
                CrosshairZYY.Y2 = 0;

                //CrosshairZYY.StrokeThickness = Math.Max(1, Options.InputAverageSlices * (double)Options.ZoomLevel);
                CrosshairZYY.Opacity = CrosshairXYX.StrokeThickness == 1 ? 0.8 : 0.2;
                Canvas.SetBottom(CrosshairZYY, (int)((Options.MouseY) * Options.ZoomLevel));
                Canvas.SetLeft(CrosshairZYY, 0);
            }
            // XZ
            {
                CrosshairXZX.X1 = 0;
                CrosshairXZX.X2 = 0;
                CrosshairXZX.Y1 = 0;
                CrosshairXZX.Y2 = ImageXZ.ActualHeight;

                //CrosshairXZX.StrokeThickness = Math.Max(1, Options.InputAverageSlices * (double)Options.ZoomLevel);
                CrosshairXZX.Opacity = CrosshairXZX.StrokeThickness == 1 ? 0.8 : 0.2;
                Canvas.SetBottom(CrosshairXZX, 0);
                Canvas.SetLeft(CrosshairXZX, (int)((Options.MouseX) * Options.ZoomLevel));

                CrosshairXZZ.X1 = 0;
                CrosshairXZZ.X2 = ImageXZ.ActualWidth;
                CrosshairXZZ.Y1 = 0;
                CrosshairXZZ.Y2 = 0;

                //CrosshairXZZ.StrokeThickness = Math.Max(1, Options.InputAverageSlices * (double)Options.ZoomLevel);
                CrosshairXZZ.Opacity = CrosshairXZZ.StrokeThickness == 1 ? 0.8 : 0.2;
                Canvas.SetBottom(CrosshairXZZ, (int)((Options.MouseZ) * Options.ZoomLevel));
                Canvas.SetLeft(CrosshairXZZ, 0);
            }
        }

        private void UpdateBoxes()
        {
            UpdateBoxesXY();
            UpdateBoxesZY();
            UpdateBoxesXZ();
        }

        private void UpdateBoxesXY()
        {
            if (SliceXYParticles != null)
            {
                List<Particle> VisibleParticles = new List<Particle>();

                int FirstSlice = Options.PlaneZ - Options.BoxSize / 2 + 1;
                int LastSlice = FirstSlice + Options.BoxSize - 1;

                for (int s = Math.Max(0, FirstSlice); s < Math.Min(LastSlice, Tomogram.Dims.Z); s++)
                    VisibleParticles.AddRange(SliceXYParticles[s]);

                CanvasParticlesXY.Children.Clear();

                double BoxRadius = Options.BoxSize / 2;

                CanvasParticlesXY.Visibility = Visibility.Collapsed;

                foreach (var part in VisibleParticles.Where(p => p.Score >= (float)Options.ParticleScoreMin && p.Score <= (float)Options.ParticleScoreMax))
                {
                    double Dist = (Options.PlaneZ - part.Position.Z) / BoxRadius;
                    double Angle = Math.Asin(Dist);
                    double R = Math.Cos(Angle) * BoxRadius;

                    Ellipse Circle = new Ellipse()
                    {
                        Width = R * 2 * (double)Options.ZoomLevel,
                        Height = R * 2 * (double)Options.ZoomLevel,
                        Stroke = part.IsSelected ? ActiveParticleBrush : ParticleBrush,
                        StrokeThickness = 1,
                        Opacity = 1.0 - Math.Abs(Dist * 0.0),
                        IsHitTestVisible = false
                    };

                    Canvas.SetLeft(Circle, (part.Position.X - R) * (double)Options.ZoomLevel);
                    Canvas.SetBottom(Circle, (part.Position.Y - R) * (double)Options.ZoomLevel);
                    CanvasParticlesXY.Children.Add(Circle);
                    part.BoxXY = Circle;
                }

                CanvasParticlesXY.Visibility = Visibility.Visible;
            }
        }

        private void UpdateBoxesZY()
        {
            if (SliceZYParticles != null)
            {
                List<Particle> VisibleParticles = new List<Particle>();

                int FirstSlice = Options.PlaneX - Options.BoxSize / 2 + 1;
                int LastSlice = FirstSlice + Options.BoxSize - 1;

                for (int s = Math.Max(0, FirstSlice); s < Math.Min(LastSlice, Tomogram.Dims.X); s++)
                    VisibleParticles.AddRange(SliceZYParticles[s]);

                CanvasParticlesZY.Children.Clear();

                double BoxRadius = Options.BoxSize / 2;

                CanvasParticlesZY.Visibility = Visibility.Collapsed;

                foreach (var part in VisibleParticles.Where(p => p.Score >= (float)Options.ParticleScoreMin && p.Score <= (float)Options.ParticleScoreMax))
                {
                    double Dist = (Options.PlaneX - part.Position.X) / BoxRadius;
                    double Angle = Math.Asin(Dist);
                    double R = Math.Cos(Angle) * BoxRadius;

                    Ellipse Circle = new Ellipse()
                    {
                        Width = R * 2 * (double)Options.ZoomLevel,
                        Height = R * 2 * (double)Options.ZoomLevel,
                        Stroke = part.IsSelected ? ActiveParticleBrush : ParticleBrush,
                        StrokeThickness = 1,
                        Opacity = 1.0 - Math.Abs(Dist * 0.0),
                        IsHitTestVisible = false
                    };

                    Canvas.SetLeft(Circle, (part.Position.Z - R) * (double)Options.ZoomLevel);
                    Canvas.SetBottom(Circle, (part.Position.Y - R) * (double)Options.ZoomLevel);
                    CanvasParticlesZY.Children.Add(Circle);
                    part.BoxZY = Circle;
                }

                CanvasParticlesZY.Visibility = Visibility.Visible;
            }
        }

        private void UpdateBoxesXZ()
        {
            if (SliceXZParticles != null)
            {
                List<Particle> VisibleParticles = new List<Particle>();

                int FirstSlice = Options.PlaneY - Options.BoxSize / 2 + 1;
                int LastSlice = FirstSlice + Options.BoxSize - 1;

                for (int s = Math.Max(0, FirstSlice); s < Math.Min(LastSlice, Tomogram.Dims.Y); s++)
                    VisibleParticles.AddRange(SliceXZParticles[s]);

                CanvasParticlesXZ.Children.Clear();

                double BoxRadius = Options.BoxSize / 2;

                CanvasParticlesXZ.Visibility = Visibility.Collapsed;

                foreach (var part in VisibleParticles.Where(p => p.Score >= (float)Options.ParticleScoreMin && p.Score <= (float)Options.ParticleScoreMax))
                {
                    double Dist = (Options.PlaneY - part.Position.Y) / BoxRadius;
                    double Angle = Math.Asin(Dist);
                    double R = Math.Cos(Angle) * BoxRadius;

                    Ellipse Circle = new Ellipse()
                    {
                        Width = R * 2 * (double)Options.ZoomLevel,
                        Height = R * 2 * (double)Options.ZoomLevel,
                        Stroke = part.IsSelected ? ActiveParticleBrush : ParticleBrush,
                        StrokeThickness = 1,
                        Opacity = 1.0 - Math.Abs(Dist * 0.0),
                        IsHitTestVisible = false
                    };

                    Canvas.SetLeft(Circle, (part.Position.X - R) * (double)Options.ZoomLevel);
                    Canvas.SetBottom(Circle, (part.Position.Z - R) * (double)Options.ZoomLevel);
                    CanvasParticlesXZ.Children.Add(Circle);
                    part.BoxXZ = Circle;
                }

                CanvasParticlesXZ.Visibility = Visibility.Visible;
            }
        }

        #endregion

        private void UpdateParticleScores()
        {
            float ScoreMin = MathHelper.Min(Particles.Select(p => p.Score));
            float ScoreMax = MathHelper.Max(Particles.Select(p => p.Score));
            float ScoreSpread = ScoreMax - ScoreMin;

            SliderParticleScoreMin.MinValue = (decimal)ScoreMin;
            SliderParticleScoreMin.MaxValue = (decimal)ScoreMax;

            SliderParticleScoreMax.MinValue = (decimal)ScoreMin;
            SliderParticleScoreMax.MaxValue = (decimal)ScoreMax;

            Options.ParticleScoreMin = (decimal)ScoreMin;
            Options.ParticleScoreMax = (decimal)ScoreMax;

            if (ScoreSpread == 0)
            {
                SliderParticleScoreMin.StepSize = 0;
                SliderParticleScoreMax.StepSize = 0;
            }
            else
            {
                float Order = (float)Math.Floor(Math.Log10(ScoreMax));
                float StepSize = (float)Math.Pow(10, Order - 3);

                SliderParticleScoreMin.StepSize = (decimal)StepSize;
                SliderParticleScoreMax.StepSize = (decimal)StepSize;
            }

            UpdateBoxes();

            Options.NParticles = GoodParticles.Count();
        }

        private void ButtonTomogramPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "MRC Volume|*.mrc",
                Multiselect = false,
                InitialDirectory = !string.IsNullOrWhiteSpace(Options.PathTomogram) ? 
                                   System.IO.Path.GetDirectoryName(Options.PathTomogram) : 
                                   null
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                try
                {
                    Options.PathTomogram = Dialog.FileName;

                    HeaderMRC Header = (HeaderMRC)MapHeader.ReadFromFile(Dialog.FileName);
                    long Elements = (long)Header.Dimensions.X * Header.Dimensions.Y * Header.Dimensions.Z;
                    if (Elements > int.MaxValue)
                        throw new Exception("Volumes with more than 2^31-1 elements are not supported, please scale it down.");

                    GridProcessingOverlay.Visibility = Visibility.Visible;
                    TextProcessingMessage.Text = "Loading " + Options.PathTomogram + "...";

                    Thread LoadingThread = new Thread(() =>
                    {
                        try
                        {
                            LoadTomogram(Options.PathTomogram);
                        }
                        catch (Exception exc)
                        {
                        }
                        finally
                        {
                            GridProcessingOverlay.Dispatcher.Invoke(() => GridProcessingOverlay.Visibility = Visibility.Hidden);
                        }
                    });

                    LoadingThread.Start();
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Couldn't read volume: " + ex.Message);
                }
            }
        }

        #region Mouse

        private void ImageZY_OnMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (Tomogram == null)
                return;

            int Delta = Math.Sign(e.Delta);

            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift))
            {
                decimal Factor = Delta > 0 ? 2M : 0.5M;
                Options.ZoomLevel = Math.Min(4M, Math.Max(0.125M, Options.ZoomLevel * Factor));
            }
            else
            {
                Options.PlaneX = Math.Max(0, Math.Min(Tomogram.Dims.X - 1, Options.PlaneX + Delta));
                Options.MouseX = Options.PlaneX;
            }
        }

        private void ImageXY_OnMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (Tomogram == null)
                return;

            int Delta = Math.Sign(e.Delta);

            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift))
            {
                decimal Factor = Delta > 0 ? 2M : 0.5M;
                Options.ZoomLevel = Math.Min(4M, Math.Max(0.125M, Options.ZoomLevel * Factor));
            }
            else
            {
                Options.PlaneZ = Math.Max(0, Math.Min(Tomogram.Dims.Z - 1, Options.PlaneZ + Delta));
                Options.MouseZ = Options.PlaneZ;
            }
        }

        private void ImageXZ_OnMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (Tomogram == null)
                return;

            int Delta = Math.Sign(e.Delta);

            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift))
            {
                decimal Factor = Delta > 0 ? 2M : 0.5M;
                Options.ZoomLevel = Math.Min(4M, Math.Max(0.125M, Options.ZoomLevel * Factor));
            }
            else
            {
                Options.PlaneY = Math.Max(0, Math.Min(Tomogram.Dims.Y - 1, Options.PlaneY + Delta));
                Options.MouseY = Options.PlaneY;
            }
        }

        private void Image_OnMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Middle)
            {
                IsDragging = true;
                DragStart = new int2((int)e.GetPosition(this).X, (int)e.GetPosition(this).Y);
            }
            else
            {
                Options.PlaneX = Options.MouseX;
                Options.PlaneY = Options.MouseY;
                Options.PlaneZ = Options.MouseZ;

                if (e.ChangedButton == MouseButton.Left || e.ChangedButton == MouseButton.Right)
                {
                    if (SliceXYParticles != null && SliceZYParticles != null && SliceXZParticles != null)
                    {
                        List<Particle> VisibleParticles = new List<Particle>();

                        if (sender == ImageXY)
                        {
                            int FirstSlice = Options.PlaneZ - Options.BoxSize / 2;
                            int LastSlice = FirstSlice + Options.BoxSize;

                            for (int s = Math.Max(0, FirstSlice); s < Math.Min(LastSlice, Tomogram.Dims.Z); s++)
                                VisibleParticles.AddRange(SliceXYParticles[s]);
                        }
                        else if (sender == ImageZY)
                        {
                            int FirstSlice = Options.PlaneX - Options.BoxSize / 2;
                            int LastSlice = FirstSlice + Options.BoxSize;

                            for (int s = Math.Max(0, FirstSlice); s < Math.Min(LastSlice, Tomogram.Dims.X); s++)
                                VisibleParticles.AddRange(SliceZYParticles[s]);
                        }
                        else if (sender == ImageXZ)
                        {
                            int FirstSlice = Options.PlaneY - Options.BoxSize / 2;
                            int LastSlice = FirstSlice + Options.BoxSize;

                            for (int s = Math.Max(0, FirstSlice); s < Math.Min(LastSlice, Tomogram.Dims.Y); s++)
                                VisibleParticles.AddRange(SliceXZParticles[s]);
                        }

                        float BoxRadius = Options.BoxSize / 2;
                        List<Tuple<float, Particle>> Candidates = new List<Tuple<float, Particle>>();

                        foreach (var part in VisibleParticles)
                        {
                            float Dist = (part.Position - new float3(Options.MouseX, Options.MouseY, Options.MouseZ)).Length();
                            if (Dist <= BoxRadius)
                                Candidates.Add(new Tuple<float, Particle>(Dist, part));
                        }

                        if (Candidates.Count > 0)
                        {
                            Candidates.Sort((a, b) => a.Item1.CompareTo(b.Item1));
                            Particle ClickedParticle = Candidates[0].Item2;

                            if (e.ChangedButton == MouseButton.Left)
                            {
                                if (GoodParticles.Contains(ClickedParticle))
                                {
                                    Options.ActiveParticle = ClickedParticle;
                                    DraggingParticle = ClickedParticle;
                                }
                                else
                                {
                                    ClickedParticle.Score = (float)Options.ParticleScoreMax;
                                    DeletedParticles.Remove(ClickedParticle);
                                    Particles.Add(ClickedParticle);
                                }
                            }
                            else if (e.ChangedButton == MouseButton.Right)
                            {
                                int OldIndex = Particles.IndexOf(ClickedParticle);
                                FreezeUpdates = true;
                                Particles.Remove(ClickedParticle);
                                DeletedParticles.Add(ClickedParticle);
                                FreezeUpdates = false;

                                if (ClickedParticle == Options.ActiveParticle)
                                    Options.ActiveParticle = null;

                                UpdateBoxes();
                            }
                        }
                    }
                }
            }
        }

        private void Image_OnMouseLeave(object sender, MouseEventArgs e)
        {
            IsDragging = false;
        }

        private void ImageZY_OnMouseMove(object sender, MouseEventArgs e)
        {
            if (IsDragging)
            {
                int2 NewPosition = new int2((int)e.GetPosition(this).X, (int)e.GetPosition(this).Y);
                int2 Delta = NewPosition - DragStart;
                Delta.Y *= -1;

                Options.ViewZ -= Delta.X / Options.ZoomLevel;
                Options.ViewY -= Delta.Y / Options.ZoomLevel;

                DragStart = NewPosition;
            }

            Options.MouseZ = (int)Math.Round(e.GetPosition(ImageZY).X / (double)Options.ZoomLevel);
            Options.MouseY = (int)Math.Round((ImageZY.ActualHeight - 1 - e.GetPosition(ImageZY).Y) / (double)Options.ZoomLevel);
            Options.MouseX = Options.PlaneX;
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                Options.PlaneZ = Options.MouseZ;
                Options.PlaneY = Options.MouseY;
            }
        }

        private void ImageXY_OnMouseMove(object sender, MouseEventArgs e)
        {
            if (IsDragging)
            {
                int2 NewPosition = new int2((int)e.GetPosition(this).X, (int)e.GetPosition(this).Y);
                int2 Delta = NewPosition - DragStart;
                Delta.Y *= -1;

                Options.ViewX -= Delta.X / Options.ZoomLevel;
                Options.ViewY -= Delta.Y / Options.ZoomLevel;

                DragStart = NewPosition;
            }

            Options.MouseX = (int)Math.Round(e.GetPosition(ImageXY).X / (double)Options.ZoomLevel);
            Options.MouseY = (int)Math.Round((ImageXY.ActualHeight - 1 - e.GetPosition(ImageXY).Y) / (double)Options.ZoomLevel);
            Options.MouseZ = Options.PlaneZ;
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                Options.PlaneX = Options.MouseX;
                Options.PlaneY = Options.MouseY;
            }

            if (!IsDragging && e.RightButton == MouseButtonState.Pressed)
            {
                if (SliceXYParticles != null && SliceZYParticles != null && SliceXZParticles != null)
                {
                    List<Particle> VisibleParticles = new List<Particle>();

                    if (sender == ImageXY)
                    {
                        int FirstSlice = Options.MouseZ - Options.BoxSize / 2;
                        int LastSlice = FirstSlice + Options.BoxSize;

                        for (int s = Math.Max(0, FirstSlice); s < Math.Min(LastSlice, Tomogram.Dims.Z); s++)
                            VisibleParticles.AddRange(SliceXYParticles[s]);
                    }
                    else if (sender == ImageZY)
                    {
                        int FirstSlice = Options.MouseX - Options.BoxSize / 2;
                        int LastSlice = FirstSlice + Options.BoxSize;

                        for (int s = Math.Max(0, FirstSlice); s < Math.Min(LastSlice, Tomogram.Dims.X); s++)
                            VisibleParticles.AddRange(SliceZYParticles[s]);
                    }
                    else if (sender == ImageXZ)
                    {
                        int FirstSlice = Options.MouseY - Options.BoxSize / 2;
                        int LastSlice = FirstSlice + Options.BoxSize;

                        for (int s = Math.Max(0, FirstSlice); s < Math.Min(LastSlice, Tomogram.Dims.Y); s++)
                            VisibleParticles.AddRange(SliceXZParticles[s]);
                    }

                    float BoxRadius = Options.BoxSize / 2;
                    List<Tuple<float, Particle>> Candidates = new List<Tuple<float, Particle>>();

                    foreach (var part in VisibleParticles)
                    {
                        float Dist = (part.Position - new float3(Options.MouseX, Options.MouseY, Options.MouseZ)).Length();
                        if (Dist <= BoxRadius)
                            Candidates.Add(new Tuple<float, Particle>(Dist, part));
                    }

                    if (Candidates.Count > 0)
                    {
                        Particle DeletedParticle = Candidates[0].Item2;

                        int OldIndex = Particles.IndexOf(DeletedParticle);
                        FreezeUpdates = true;
                        Particles.Remove(DeletedParticle);
                        DeletedParticles.Add(DeletedParticle);
                        FreezeUpdates = false;

                        if (DeletedParticle == Options.ActiveParticle)
                            Options.ActiveParticle = null;

                        CanvasParticlesXY.Children.Remove(DeletedParticle.BoxXY);
                        CanvasParticlesZY.Children.Remove(DeletedParticle.BoxZY);
                        CanvasParticlesXZ.Children.Remove(DeletedParticle.BoxXZ);
                    }
                }
            }
        }

        private void ImageXZ_OnMouseMove(object sender, MouseEventArgs e)
        {
            if (IsDragging)
            {
                int2 NewPosition = new int2((int)e.GetPosition(this).X, (int)e.GetPosition(this).Y);
                int2 Delta = NewPosition - DragStart;
                Delta.Y *= -1;

                Options.ViewX -= Delta.X / Options.ZoomLevel;
                Options.ViewZ -= Delta.Y / Options.ZoomLevel;

                DragStart = NewPosition;
            }

            Options.MouseX = (int)Math.Round(e.GetPosition(ImageXZ).X / (double)Options.ZoomLevel);
            Options.MouseZ = (int)Math.Round((ImageXZ.ActualHeight - 1 - e.GetPosition(ImageXZ).Y) / (double)Options.ZoomLevel);
            Options.MouseY = Options.PlaneY;
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                Options.PlaneX = Options.MouseX;
                Options.PlaneZ = Options.MouseZ;
            }
        }

        private void ImageZY_OnMouseEnter(object sender, MouseEventArgs e)
        {
            //Options.ViewX = Options.PlaneX;
        }

        private void ImageXY_OnMouseEnter(object sender, MouseEventArgs e)
        {
            //Options.ViewZ = Options.PlaneZ;
        }

        private void ImageXZ_OnMouseEnter(object sender, MouseEventArgs e)
        {
            //Options.ViewY = Options.PlaneY;
        }

        private void Image_OnMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left)
            {
                if (DraggingParticle == null && !Keyboard.IsKeyDown(Key.LeftShift) && !Keyboard.IsKeyDown(Key.RightShift))
                {
                    Particle NewParticle = new Particle(new float3(Options.MouseX, Options.MouseY, Options.MouseZ), new float3(0, 0, 0));

                    // Freeze so boxes aren't redrawn twice upon setting new particle as active
                    FreezeUpdates = true;
                    Particles.Add(NewParticle);
                    FreezeUpdates = false;

                    Options.ActiveParticle = NewParticle;
                }
                else
                {
                    DraggingParticle = null;
                }
            }
        }

        #endregion

        #region Helpers

        private void ActiveParticleChanged()
        {
            UpdateBoxes();
            UpdateParticlePlanes();

            if (Options.ActiveParticle != null)
            {
                Options.ParticlePlaneX = (int)Options.ActiveParticle.Position.X;
                Options.ParticlePlaneY = (int)Options.ActiveParticle.Position.Y;
                Options.ParticlePlaneZ = (int)Options.ActiveParticle.Position.Z;
            }
        }

        private void MousePositionChanged()
        {
            if (DraggingParticle != null && Mouse.LeftButton == MouseButtonState.Pressed)
                MoveParticle(DraggingParticle, new float3(Options.MouseX, Options.MouseY, Options.MouseZ));
        }

        private void MoveParticle(Particle particle, float3 newPosition)
        {
            if (particle.Position != newPosition)
            {
                FreezeUpdates = true;

                int OldIndex = Particles.IndexOf(particle);
                Particles.Remove(particle);
                particle.Position = newPosition;
                Particles.Insert(OldIndex, particle);

                if (Options.ActiveParticle == particle)
                {
                    Options.ParticlePlaneX = (int)Options.ActiveParticle.Position.X;
                    Options.ParticlePlaneY = (int)Options.ActiveParticle.Position.Y;
                    Options.ParticlePlaneZ = (int)Options.ActiveParticle.Position.Z;

                    UpdateParticlePlanes();
                }

                FreezeUpdates = false;

                UpdateBoxes();
            }
        }

        private void CenterOn(int3 position)
        {
            Options.PlaneX = position.X;
            Options.PlaneY = position.Y;
            Options.PlaneZ = position.Z;

            /*Options.ViewX = position.X;
            Options.ViewY = position.Y;
            Options.ViewZ = position.Z;*/
        }

        private ImageSource GetImage(float[] data, int2 dims)
        {
            float NyquistFraction = 2f * PixelSize / (float)Options.InputLowpass;

            Image Filtered = new Image(data, new int3(dims.X, dims.Y, 1));
            if (NyquistFraction < 1f)
                Filtered.Bandpass(0, NyquistFraction, false);

            /*if (Options.ZoomLevel != 1)
            {
                int2 DimsScaled = new int2((int)(dims.X * Options.ZoomLevel), (int)(dims.Y * Options.ZoomLevel));
                Image Scaled = Filtered.AsScaledMassive(DimsScaled);

                data = Scaled.GetHostContinuousCopy();
                dims = DimsScaled;

                Scaled.Dispose();
            }*/
            else
            {
                data = Filtered.GetHostContinuousCopy();
            }

            Filtered.Dispose();

            float2 Stats = MathHelper.MeanAndStd(data);
            float MinVal = Stats.X + Stats.Y * (float)Options.DisplayIntensityMin;
            float MaxVal = Stats.X + Stats.Y * (float)Options.DisplayIntensityMax;
            float Range = MaxVal - MinVal;

            byte[] DataBytes = new byte[data.Length];

            unsafe
            {
                fixed (float* DataPtr = data)
                fixed (byte* BytePtr = DataBytes)
                {
                    for (int y = 0; y < dims.Y; y++)
                    {
                        int yy = dims.Y - y - 1;
                        float* DataP = DataPtr + y * dims.X;
                        byte* ByteP = BytePtr + yy * dims.X;

                        for (int x = 0; x < dims.X; x++)
                            ByteP[x] = (byte)(Math.Max(0, Math.Min(1, (DataP[x] - MinVal) / Range)) * 255);
                    }
                }
            }

            ImageSource Result = BitmapSource.Create(dims.X, dims.Y, 96 / (double)Options.ZoomLevel, 96 / (double)Options.ZoomLevel, PixelFormats.Indexed8, BitmapPalettes.Gray256, DataBytes, dims.X);
            return Result;
        }

        #endregion

        private void ButtonPointsImport_OnClick(object sender, RoutedEventArgs e)
        {
            if (Tomogram == null)
            {
                MessageBox.Show("This will not work without a tomogram loaded.");
                return;
            }

            CoordsImportWindow ImportWindow = new CoordsImportWindow();
            ImportWindow.DataContext = Options;
            ImportWindow.ParentWindow = this;
            ImportWindow.Owner = this;

            ImportWindow.ShowDialog();
        }

        public void PointsImport()
        {
            if (Tomogram == null)
            {
                MessageBox.Show("This will not work without a tomogram loaded.");
                return;
            }

            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog();
            Dialog.InitialDirectory = !string.IsNullOrWhiteSpace(Options.PathParticles) ?
                                      System.IO.Path.GetDirectoryName(Options.PathParticles) :
                                      null;
            Dialog.Filter = "STAR File|*.star|Text File|*.txt";
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                FreezeUpdates = true;
                Particles.Clear();
                DeletedParticles.Clear();

                Options.PathParticles = Dialog.FileName;

                float3 ImportScale = new float3((float)Tomogram.Dims.X / Options.ImportVolumeWidth,
                                                (float)Tomogram.Dims.Y / Options.ImportVolumeHeight,
                                                (float)Tomogram.Dims.Z / Options.ImportVolumeDepth);

                try
                {
                    FileInfo Info = new FileInfo(Dialog.FileName);
                    if (Info.Extension.ToLower().Replace(".", "") == "txt")
                    {
                        using (TextReader Reader = new StreamReader(File.OpenRead(Dialog.FileName)))
                        {
                            string Line;
                            while ((Line = Reader.ReadLine()) != null)
                            {
                                string[] Parts = Line.Split(new[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);

                                if (Parts.Length < 3)
                                {
                                    Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                                    if (Parts.Length < 3)
                                        continue;
                                }

                                if (Parts.Length != 3 && Parts.Length != 6)
                                    throw new Exception("Tab-delimited text file must have either 3 (XYZ) or 6 (XYZ Rot Tilt Psi) columns.");

                                float X = 0, Y = 0, Z = 0, Rot = 0, Tilt = 0, Psi = 0;
                                if (Parts.Length >= 3)
                                {
                                    X = float.Parse(Parts[0]);
                                    Y = float.Parse(Parts[1]);
                                    Z = float.Parse(Parts[2]);
                                }
                                if (Parts.Length == 6)
                                {
                                    Rot = float.Parse(Parts[3]);
                                    Tilt = float.Parse(Parts[4]);
                                    Psi = float.Parse(Parts[5]);
                                }

                                //Z -= 57;

                                if (Options.ImportInvertX)
                                    X = Options.ImportVolumeWidth - X - 1;
                                if (Options.ImportInvertY)
                                    Y = Options.ImportVolumeHeight - Y - 1;
                                if (Options.ImportInvertZ)
                                    Z = Options.ImportVolumeDepth - Z - 1;

                                X *= ImportScale.X;
                                Y *= ImportScale.Y;
                                Z *= ImportScale.Z;

                                if (X < 0 || X >= Tomogram.Dims.X || Y < 0 || Y >= Tomogram.Dims.Y || Z < 0 || Z >= Tomogram.Dims.Z)
                                    continue;

                                /*X = Math.Max(0, Math.Min(X, Tomogram.Dims.X - 1));
                                Y = Math.Max(0, Math.Min(Y, Tomogram.Dims.Y - 1));
                                Z = Math.Max(0, Math.Min(Z, Tomogram.Dims.Z - 1));*/

                                Particle NewParticle = new Particle(new float3(X, Y, Z), new float3(Rot, Tilt, Psi));
                                Particles.Add(NewParticle);
                            }
                        }
                    }
                    else if (Info.Extension.ToLower().Replace(".", "") == "star")
                    {
                        Star Table = new Star(Dialog.FileName);
                        string[] ColumnX = Table.GetColumn("rlnCoordinateX");
                        string[] ColumnY = Table.GetColumn("rlnCoordinateY");
                        string[] ColumnZ = Table.GetColumn("rlnCoordinateZ");

                        string[] ColumnShiftX = Table.GetColumn("rlnOriginX");
                        string[] ColumnShiftY = Table.GetColumn("rlnOriginY");
                        string[] ColumnShiftZ = Table.GetColumn("rlnOriginZ");

                        string[] ColumnRot = Table.GetColumn("rlnAngleRot");
                        string[] ColumnTilt = Table.GetColumn("rlnAngleTilt");
                        string[] ColumnPsi = Table.GetColumn("rlnAnglePsi");

                        string[] ColumnTomoName = Table.GetColumn("rlnMicrographName");

                        string[] ColumnScore = Table.GetColumn("rlnAutopickFigureOfMerit");

                        string TomoRootName = Helper.PathToName(Options.PathTomogram);

                        for (int i = 0; i < Table.RowCount; i++)
                        {
                            //if (ColumnTomoName != null)
                            //    if (!ColumnTomoName[i].Contains(TomoRootName))
                            //        continue;

                            float X = 0, Y = 0, Z = 0, Rot = 0, Tilt = 0, Psi = 0;

                            if (ColumnX != null)
                                X = float.Parse(ColumnX[i]);
                            if (ColumnY != null)
                                Y = float.Parse(ColumnY[i]);
                            if (ColumnZ != null)
                                Z = float.Parse(ColumnZ[i]);

                            if (ColumnShiftX != null)
                                X -= float.Parse(ColumnShiftX[i]);
                            if (ColumnShiftY != null)
                                Y -= float.Parse(ColumnShiftY[i]);
                            if (ColumnShiftZ != null)
                                Z -= float.Parse(ColumnShiftZ[i]);

                            if (ColumnRot != null)
                                Rot = float.Parse(ColumnRot[i]);
                            if (ColumnTilt != null)
                                Tilt = float.Parse(ColumnTilt[i]);
                            if (ColumnPsi != null)
                                Psi = float.Parse(ColumnPsi[i]);

                            if (Options.ImportInvertX)
                                X = Options.ImportVolumeWidth - X - 1;
                            if (Options.ImportInvertY)
                                Y = Options.ImportVolumeHeight - Y - 1;
                            if (Options.ImportInvertZ)
                                Z = Options.ImportVolumeDepth - Z - 1;

                            X *= ImportScale.X;
                            Y *= ImportScale.Y;
                            Z *= ImportScale.Z;

                            if (X < 0 || X >= Tomogram.Dims.X || Y < 0 || Y >= Tomogram.Dims.Y || Z < 0 || Z >= Tomogram.Dims.Z)
                                continue;

                            /*X = Math.Max(0, Math.Min(X, Tomogram.Dims.X - 1));
                            Y = Math.Max(0, Math.Min(Y, Tomogram.Dims.Y - 1));
                            Z = Math.Max(0, Math.Min(Z, Tomogram.Dims.Z - 1));*/

                            Particle NewParticle = new Particle(new float3(X, Y, Z), new float3(Rot, Tilt, Psi));

                            if (ColumnScore != null)
                                NewParticle.Score = float.Parse(ColumnScore[i], CultureInfo.InvariantCulture);

                            Particles.Add(NewParticle);
                        }
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Couldn't parse file: " + ex.Message);
                }

                UpdateParticleScores();

                FreezeUpdates = false;
                UpdateBoxes();
            }
        }

        private void ButtonPointsExport_OnClick(object sender, RoutedEventArgs e)
        {
            if (Tomogram == null)
            {
                MessageBox.Show("This will not work without a tomogram loaded.");
                return;
            }

            CoordsExportWindow ExportWindow = new CoordsExportWindow();
            ExportWindow.DataContext = Options;
            ExportWindow.ParentWindow = this;
            ExportWindow.Owner = this;

            ExportWindow.ShowDialog();
        }

        private void ExportToStar(IEnumerable<Particle> particles, string path)
        {
            FileInfo Info = new FileInfo(Options.PathTomogram);
            string MicName = Info.Name;

            float3 ExportScale = new float3((float)Options.ExportVolumeWidth / Tomogram.Dims.X,
                                            (float)Options.ExportVolumeHeight / Tomogram.Dims.Y,
                                            (float)Options.ExportVolumeDepth / Tomogram.Dims.Z);

            Star Table = new Star(new[]
            {
                    "rlnCoordinateX",
                    "rlnCoordinateY",
                    "rlnCoordinateZ",
                    "rlnOriginX",
                    "rlnOriginY",
                    "rlnOriginZ",
                    "rlnAngleRot",
                    "rlnAngleTilt",
                    "rlnAnglePsi",
                    "rlnMicrographName"
                });

            foreach (var particle in particles)
            {
                float3 Scaled = particle.Position * ExportScale;
                if (Options.ExportInvertX)
                    Scaled.X = Options.ExportVolumeWidth - Scaled.X - 1;
                if (Options.ExportInvertY)
                    Scaled.Y = Options.ExportVolumeHeight - Scaled.Y - 1;
                if (Options.ExportInvertZ)
                    Scaled.Z = Options.ExportVolumeDepth - Scaled.Z - 1;

                Table.AddRow(new string[]
                    {
                        Scaled.X.ToString(CultureInfo.InvariantCulture),
                        Scaled.Y.ToString(CultureInfo.InvariantCulture),
                        Scaled.Z.ToString(CultureInfo.InvariantCulture),
                        "0",
                        "0",
                        "0",
                        particle.Angle.X.ToString(CultureInfo.InvariantCulture),
                        particle.Angle.Y.ToString(CultureInfo.InvariantCulture),
                        particle.Angle.Z.ToString(CultureInfo.InvariantCulture),
                        MicName
                    });
            }

            Table.Save(path);
        }

        public void PointsExport()
        {
            if (Tomogram == null)
            {
                MessageBox.Show("This will not work without a tomogram loaded.");
                return;
            }

            System.Windows.Forms.SaveFileDialog Dialog = new System.Windows.Forms.SaveFileDialog();
            Dialog.Filter = "STAR File|*.star";
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                ExportToStar(GoodParticles, Dialog.FileName);
                ExportToStar(BadParticles, Dialog.FileName + ".bad");
            }
        }
    }
}
