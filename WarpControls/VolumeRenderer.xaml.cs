using System;
using System.Collections.Generic;
using System.Drawing;
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
using Color = System.Windows.Media.Color;

namespace Warp
{
    /// <summary>
    /// Interaction logic for VolumeRenderer.xaml
    /// </summary>
    public partial class VolumeRenderer : UserControl
    {
        bool RenderingEnabled = true;
        bool IsMouseDraggingRotate = false;
        bool IsMouseDraggingPan = false;
        float2 LastMousePos;

        public Image Volume
        {
            get { return (Image)GetValue(VolumeProperty); }
            set { SetValue(VolumeProperty, value); }
        }
        public static readonly DependencyProperty VolumeProperty = DependencyProperty.Register("Volume", typeof (Image), typeof (VolumeRenderer), new PropertyMetadata(null, (o, args) =>
        {
            ((VolumeRenderer)o).SetAutoThreshold();
            ((VolumeRenderer)o).RenderDataChanged(args);
        }));

        public Image Coloring
        {
            get { return (Image)GetValue(ColoringProperty); }
            set { SetValue(ColoringProperty, value); }
        }
        public static readonly DependencyProperty ColoringProperty = DependencyProperty.Register("Coloring", typeof(Image), typeof(VolumeRenderer), new PropertyMetadata(null, (o, args) => ((VolumeRenderer)o).RenderDataChanged(args)));

        private ulong t_Volume = 0, a_Volume = 0;
        private ulong t_Coloring = 0, a_Coloring = 0;

        public float3[] ColorMap
        {
            get { return (float3[])GetValue(ColorMapProperty); }
            set { SetValue(ColorMapProperty, value); }
        }
        public static readonly DependencyProperty ColorMapProperty = DependencyProperty.Register("ColorMap", typeof(float3[]), typeof(VolumeRenderer), new PropertyMetadata(null, (o, args) => ((VolumeRenderer)o).RenderDataChanged(args)));

        private Image ColorMapImage;
        
        public VolumeRendererCamera Camera
        {
            get { return (VolumeRendererCamera)GetValue(CameraProperty); }
            set { SetValue(CameraProperty, value); }
        }
        public static readonly DependencyProperty CameraProperty = DependencyProperty.Register("Camera", typeof(VolumeRendererCamera), typeof(VolumeRenderer), new PropertyMetadata());

        int Supersampling = 4;

        public int SupersamplingLive
        {
            get { return (int)GetValue(SupersamplingLiveProperty); }
            set { SetValue(SupersamplingLiveProperty, value); }
        }
        public static readonly DependencyProperty SupersamplingLiveProperty = DependencyProperty.Register("SupersamplingLive", typeof(int), typeof(VolumeRenderer), new PropertyMetadata(2));

        public int SupersamplingStill
        {
            get { return (int)GetValue(SupersamplingStillProperty); }
            set { SetValue(SupersamplingStillProperty, value); }
        }
        public static readonly DependencyProperty SupersamplingStillProperty = DependencyProperty.Register("SupersamplingStill", typeof(int), typeof(VolumeRenderer), new PropertyMetadata(4));

        public event ViewportMouseEvent PreviewViewportMouseMove;
        public event ViewportMouseButtonEvent PreviewViewportMouseDown, PreviewViewportMouseUp;

        private int2 DimsImage;
        private byte[] BGRA;
        private byte[] HitTest;
        private float[] IntersectionPoints;

        public VolumeRenderer()
        {
            InitializeComponent();

            Camera = new VolumeRendererCamera();
            PopupControls.DataContext = this;
            Camera.PropertyChanged += Camera_PropertyChanged;
            SizeChanged += VolumeRenderer_SizeChanged;
        }

        private void VolumeRenderer_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            UpdateRendering();
        }

        private void RenderDataChanged(DependencyPropertyChangedEventArgs args)
        {
            FreeOnDevice();
            UpdateRendering();
        }

        private void RenderSettingChanged(DependencyPropertyChangedEventArgs args)
        {
            UpdateRendering();
        }

        private void CameraChanged(DependencyPropertyChangedEventArgs args)
        {
            if (args.OldValue != null)
                ((VolumeRendererCamera)args.OldValue).PropertyChanged -= Camera_PropertyChanged;

            if (args.NewValue != null)
                ((VolumeRendererCamera)args.NewValue).PropertyChanged += Camera_PropertyChanged;

            UpdateRendering();
        }

        private void Camera_PropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            UpdateRendering();
        }

        private void PutOnDevice()
        {
            if (!IsOnDevice())
            {
                FreeOnDevice();

                if (Volume != null)
                {
                    Image VolumePrefiltered = Volume.GetCopyGPU();
                    GPU.PrefilterForCubic(VolumePrefiltered.GetDevice(Intent.ReadWrite), VolumePrefiltered.Dims);
                    ulong[] T = new ulong[1], A = new ulong[1];
                    GPU.CreateTexture3D(VolumePrefiltered.GetDevice(Intent.Read), VolumePrefiltered.Dims, T, A, true);
                    VolumePrefiltered.Dispose();

                    t_Volume = T[0];
                    a_Volume = A[0];
                }

                if (Coloring != null)
                {
                    ulong[] T = new ulong[1], A = new ulong[1];
                    GPU.CreateTexture3D(Coloring.GetDevice(Intent.Read), Coloring.Dims, T, A, true);

                    t_Coloring = T[0];
                    a_Coloring = A[0];
                }

                if (ColorMap != null)
                    ColorMapImage = new Image(Helper.ToInterleaved(ColorMap));
                else
                    ColorMapImage = new Image(Helper.ArrayOfConstant(1f, 6));
            }
        }

        public void FreeOnDevice()
        {
            if (t_Volume > 0)
            {
                GPU.DestroyTexture(t_Volume, a_Volume);
                t_Volume = 0;
                a_Volume = 0;
            }

            if (t_Coloring > 0)
            {
                GPU.DestroyTexture(t_Coloring, a_Coloring);
                t_Coloring = 0;
                a_Coloring = 0;
            }

            ColorMapImage?.FreeDevice();
        }

        private bool IsOnDevice()
        {
            return t_Volume > 0;
        }

        public void RenderVolume(int2 dimsImage, int supersampling)
        {
            if (BGRA == null || BGRA.Length != dimsImage.Elements() * 4)
                BGRA = new byte[dimsImage.Elements() * 4];
            if (HitTest == null || HitTest.Length != dimsImage.Elements())
                HitTest = new byte[dimsImage.Elements()];
            if (IntersectionPoints == null || IntersectionPoints.Length != dimsImage.Elements() * 3)
                IntersectionPoints = new float[dimsImage.Elements() * 3];

            if (Volume != null)
            {
                Matrix4 M = Matrix4.Euler((float)Camera.AngleRot * Helper.ToRad,
                                          (float)Camera.AngleTilt * Helper.ToRad,
                                          (float)Camera.AnglePsi * Helper.ToRad).Transposed();
                

                float3 CameraVec = M * new float3(-(float)Camera.PanX, -(float)Camera.PanY, Volume.Dims.X);
                float3 PixelX = M * new float3(1, 0, 0) / supersampling / (float)Camera.Zoom;
                float3 PixelY = M * new float3(0, -1, 0) / supersampling / (float)Camera.Zoom;
                float3 View = M * new float3(0, 0, -1);

                GPU.RenderVolume(t_Volume,
                                 Volume.Dims,
                                 Camera.ShowSurface ? (float)Camera.SurfaceThreshold : 1e35f,
                                 Camera.ShowColoring ? t_Coloring : 0,
                                 Coloring?.Dims ?? new int3(1),
                                 dimsImage,
                                 CameraVec,
                                 PixelX,
                                 PixelY,
                                 View,
                                 ColorMapImage.GetDevice(Intent.Read),
                                 ColorMapImage.Dims.X / 3,
                                 new float2((float)Camera.ColoringRangeMin, (float)Camera.ColoringRangeMax),
                                 new float2((float)Camera.ShadingRangeMin, (float)Camera.ShadingRangeMax),
                                 new float3(Camera.IntensityColor.X / 255f, Camera.IntensityColor.Y / 255f, Camera.IntensityColor.Z / 255f),
                                 Camera.ShowIntensity ? new float2((float)Camera.IntensityRangeMin, (float)Camera.IntensityRangeMax) : new float2(1e35f, 1e36f),
                                 IntersectionPoints,
                                 HitTest,
                                 BGRA);
            }
            else
            {
                for (int i = 0; i < BGRA.Length; i++)
                {
                    BGRA[i] = 1;
                }
                BGRA = Helper.ArrayOfConstant((byte)1, (int)dimsImage.Elements() * 4);
            }
        }

        public void SetVolumeFrom(Image data)
        {
            if (Volume == null || Volume.Dims != data.Dims)
            {
                Volume?.Dispose();
                FreeOnDevice();
                Volume = data.GetCopyGPU();
            }
            else
            {
                GPU.CopyDeviceToDevice(data.GetDevice(Intent.Read),
                                       Volume.GetDevice(Intent.Write),
                                       data.ElementsReal);
            }
        }

        public void SetAutoThreshold()
        {
            if (Volume != null && Camera != null)
            {
                float2 MeanStd = MathHelper.MeanAndStd(Volume.GetHostContinuousCopy());
                Camera.SurfaceThreshold = (decimal)(MeanStd.X + MeanStd.Y * 4);
            }
        }

        public void UpdateRendering()
        {
            if (!RenderingEnabled)
                return;

            if (ActualWidth <= 1 || ActualHeight <= 1)
                return;

            bool WasOnDevice = IsOnDevice();
            if (!WasOnDevice)
                PutOnDevice();

            DimsImage = new int2(Math.Max(1, (int)ActualWidth), Math.Max(1, (int)ActualHeight)) * Supersampling;

            RenderVolume(DimsImage, Supersampling);

            BitmapSource Rendering = BitmapSource.Create(DimsImage.X, DimsImage.Y, 96, 96, PixelFormats.Bgra32, null, BGRA, DimsImage.X * 4);
            Rendering.Freeze();

            Dispatcher.Invoke(() => ImageDisplay.Source = Rendering);

            if (!WasOnDevice)
                FreeOnDevice();
        }

        private void ImageDisplay_OnMouseDown(object sender, MouseButtonEventArgs e)
        {
            float2 Viewport2DPixels, Viewport2DWorld;
            bool HitTestResult;
            float3 Viewport3DWorld;
            GetMouseStats(e.GetPosition(ImageDisplay), out Viewport2DPixels, out Viewport2DWorld, out HitTestResult, out Viewport3DWorld);

            PreviewViewportMouseDown?.Invoke(Viewport2DPixels, Viewport2DWorld, HitTestResult, Viewport3DWorld, e);
            if (e.Handled)
                return;

            PutOnDevice();
            if (e.ChangedButton == MouseButton.Left)
            {
                IsMouseDraggingRotate = true;
                Supersampling = SupersamplingLive;
            }
            if (e.ChangedButton == MouseButton.Middle)
            {
                IsMouseDraggingPan = true;
                Supersampling = SupersamplingLive;
            }

            LastMousePos = new float2((float)e.GetPosition(ImageDisplay).X, (float)e.GetPosition(ImageDisplay).Y);
        }

        private void ImageDisplay_OnMouseUp(object sender, MouseButtonEventArgs e)
        {
            float2 Viewport2DPixels, Viewport2DWorld;
            bool HitTestResult;
            float3 Viewport3DWorld;
            GetMouseStats(e.GetPosition(ImageDisplay), out Viewport2DPixels, out Viewport2DWorld, out HitTestResult, out Viewport3DWorld);

            PreviewViewportMouseUp?.Invoke(Viewport2DPixels, Viewport2DWorld, HitTestResult, Viewport3DWorld, e);
            if (e.Handled)
                return;

            if (e.ChangedButton == MouseButton.Left)
                IsMouseDraggingRotate = false;
            if (e.ChangedButton == MouseButton.Middle)
                IsMouseDraggingPan = false;
            
            Supersampling = SupersamplingStill;
            UpdateRendering();
        }

        private void ImageDisplay_OnMouseMove(object sender, MouseEventArgs e)
        {
            float2 Viewport2DPixels, Viewport2DWorld;
            bool HitTestResult;
            float3 Viewport3DWorld;
            GetMouseStats(e.GetPosition(ImageDisplay), out Viewport2DPixels, out Viewport2DWorld, out HitTestResult, out Viewport3DWorld);

            PreviewViewportMouseMove?.Invoke(Viewport2DPixels, Viewport2DWorld, HitTestResult, Viewport3DWorld, e);
            if (e.Handled)
                return;

            PutOnDevice();

            float2 NewMousePos = new float2((float)e.GetPosition(ImageDisplay).X, (float)e.GetPosition(ImageDisplay).Y);
            float2 MouseDelta = NewMousePos - LastMousePos;
            LastMousePos = NewMousePos;

            if (IsMouseDraggingRotate || IsMouseDraggingPan)
            {
                RenderingEnabled = false;
            }

            if (IsMouseDraggingRotate)
            {
                float3 Axis = new float3(MouseDelta.Y, MouseDelta.X, 0);
                Matrix3 MAxis = Matrix3.RotateAxis(Axis, -(float)Math.Asin(Math.Min(1, Math.Max(-1, Axis.Length() / (Volume != null ? Volume.Dims.X / 4 : 1) / Math.Max(1, (float)Camera.Zoom)))));

                Matrix3 MEuler = Matrix3.Euler((float)Camera.AngleRot * Helper.ToRad,
                                               (float)Camera.AngleTilt * Helper.ToRad,
                                               (float)Camera.AnglePsi * Helper.ToRad).Transposed();

                float3 NewAngles = Matrix3.EulerFromMatrix((MEuler * MAxis).Transposed()) * Helper.ToDeg;
                while (NewAngles.X < -360)
                    NewAngles.X += 360;
                while (NewAngles.X > 360)
                    NewAngles.X -= 360;
                while (NewAngles.Y < -360)
                    NewAngles.Y += 360;
                while (NewAngles.Y > 360)
                    NewAngles.Y -= 360;
                while (NewAngles.Z < -360)
                    NewAngles.Z += 360;
                while (NewAngles.Z > 360)
                    NewAngles.Z -= 360;

                Camera.AngleRot = (decimal)NewAngles.X;
                Camera.AngleTilt = (decimal)NewAngles.Y;
                Camera.AnglePsi = (decimal)NewAngles.Z;
            }

            if (IsMouseDraggingPan)
            {
                Camera.PanX += (decimal)MouseDelta.X / Camera.Zoom;
                Camera.PanY -= (decimal)MouseDelta.Y / Camera.Zoom;
            }


            if (IsMouseDraggingRotate || IsMouseDraggingPan)
            {
                RenderingEnabled = true;
                UpdateRendering();
            }
        }

        private void ImageDisplay_OnMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (e.Delta > 0)
                Camera.Zoom = Math.Min(100, Camera.Zoom * 1.2M);
            else if (e.Delta < 0)
                Camera.Zoom = Math.Max(0.01M, Camera.Zoom / 1.2M);
        }

        private void VolumeRenderer_OnMouseEnter(object sender, MouseEventArgs e)
        {
            PopupControls.IsOpen = true;
            PutOnDevice();
        }

        private void VolumeRenderer_OnMouseLeave(object sender, MouseEventArgs e)
        {
            if (!IsMouseOver && !PopupControls.IsMouseOver)
            {
                PopupControls.IsOpen = false;
                FreeOnDevice();
            }
        }

        private void GetMouseStats(System.Windows.Point point, out float2 viewport2DPixels, out float2 viewport2DWorld, out bool hitTest, out float3 viewport3DWorld)
        {
            float2 Position = new float2((float)point.X, (float)point.Y);
            float2 ViewportCenter = new float2((float)ActualWidth, (float)ActualHeight) * 0.5f;
            
            viewport2DPixels = Position;
            viewport2DWorld = (Position - ViewportCenter) / (float)Camera.Zoom + new float2((float)Camera.PanX, (float)Camera.PanY);

            int2 PositionSuper = new int2(Position * Supersampling + 0.5f);
            int IndexSuper = DimsImage.ElementFromPosition(PositionSuper);
            if (HitTest != null && IndexSuper < HitTest.Length)
            {
                hitTest = HitTest[IndexSuper] > 0;
                if (hitTest)
                    viewport3DWorld = new float3(IntersectionPoints[IndexSuper * 3 + 0],
                                                 IntersectionPoints[IndexSuper * 3 + 1],
                                                 IntersectionPoints[IndexSuper * 3 + 2]);
                else
                    viewport3DWorld = new float3();
            }
            else
            {
                hitTest = false;
                viewport3DWorld = new float3();
            }
        }
    }

    public class VolumeRendererCamera : WarpBase
    {
        private decimal _AngleRot = 0;
        [WarpSerializable]
        public decimal AngleRot
        {
            get { return _AngleRot; }
            set { if (value != _AngleRot) { _AngleRot = value; OnPropertyChanged(); } }
        }

        private decimal _AngleTilt = 0;
        [WarpSerializable]
        public decimal AngleTilt
        {
            get { return _AngleTilt; }
            set { if (value != _AngleTilt) { _AngleTilt = value; OnPropertyChanged(); } }
        }

        private decimal _AnglePsi = 0;
        [WarpSerializable]
        public decimal AnglePsi
        {
            get { return _AnglePsi; }
            set { if (value != _AnglePsi) { _AnglePsi = value; OnPropertyChanged(); } }
        }

        private decimal _PanX = 0;
        [WarpSerializable]
        public decimal PanX
        {
            get { return _PanX; }
            set { if (value != _PanX) { _PanX = value; OnPropertyChanged(); } }
        }

        private decimal _PanY = 0;
        [WarpSerializable]
        public decimal PanY
        {
            get { return _PanY; }
            set { if (value != _PanY) { _PanY = value; OnPropertyChanged(); } }
        }

        private decimal _Zoom = 1M;
        [WarpSerializable]
        public decimal Zoom
        {
            get { return _Zoom; }
            set { if (value != _Zoom) { _Zoom = value; OnPropertyChanged(); } }
        }

        private decimal _SurfaceThreshold = 0.02M;
        [WarpSerializable]
        public decimal SurfaceThreshold
        {
            get { return _SurfaceThreshold; }
            set { if (value != _SurfaceThreshold) { _SurfaceThreshold = value; OnPropertyChanged(); } }
        }

        private bool _ShowSurface = true;
        [WarpSerializable]
        public bool ShowSurface
        {
            get { return _ShowSurface; }
            set { if (value != _ShowSurface) { _ShowSurface = value; OnPropertyChanged(); } }
        }

        private decimal _ShadingRangeMin = 0;
        [WarpSerializable]
        public decimal ShadingRangeMin
        {
            get { return _ShadingRangeMin; }
            set { if (value != _ShadingRangeMin) { _ShadingRangeMin = value; OnPropertyChanged(); } }
        }
        [WarpSerializable]

        private decimal _ShadingRangeMax = 1M;
        [WarpSerializable]
        public decimal ShadingRangeMax
        {
            get { return _ShadingRangeMax; }
            set { if (value != _ShadingRangeMax) { _ShadingRangeMax = value; OnPropertyChanged(); } }
        }

        private bool _ShowColoring = false;
        [WarpSerializable]
        public bool ShowColoring
        {
            get { return _ShowColoring; }
            set { if (value != _ShowColoring) { _ShowColoring = value; OnPropertyChanged(); } }
        }

        private decimal _ColoringRangeMin = 0;
        [WarpSerializable]
        public decimal ColoringRangeMin
        {
            get { return _ColoringRangeMin; }
            set { if (value != _ColoringRangeMin) { _ColoringRangeMin = value; OnPropertyChanged(); } }
        }

        private decimal _ColoringRangeMax = 1M;
        [WarpSerializable]
        public decimal ColoringRangeMax
        {
            get { return _ColoringRangeMax; }
            set { if (value != _ColoringRangeMax) { _ColoringRangeMax = value; OnPropertyChanged(); } }
        }

        private bool _ShowIntensity = false;
        [WarpSerializable]
        public bool ShowIntensity
        {
            get { return _ShowIntensity; }
            set { if (value != _ShowIntensity) { _ShowIntensity = value; OnPropertyChanged(); } }
        }

        private int3 _IntensityColor = new int3(0);
        [WarpSerializable]
        public int3 IntensityColor
        {
            get { return _IntensityColor; }
            set { if (value != _IntensityColor) { _IntensityColor = value; OnPropertyChanged(); } }
        }

        private decimal _IntensityRangeMin = 0;
        [WarpSerializable]
        public decimal IntensityRangeMin
        {
            get { return _IntensityRangeMin; }
            set { if (value != _IntensityRangeMin) { _IntensityRangeMin = value; OnPropertyChanged(); } }
        }

        private decimal _IntensityRangeMax = 1M;
        [WarpSerializable]
        public decimal IntensityRangeMax
        {
            get { return _IntensityRangeMax; }
            set { if (value != _IntensityRangeMax) { _IntensityRangeMax = value; OnPropertyChanged(); } }
        }
    }

    public delegate void ViewportMouseEvent(float2 viewport2DPixels, float2 viewport2DWorld, bool hitTest, float3 viewport3DWorld, MouseEventArgs e);
    public delegate void ViewportMouseButtonEvent(float2 viewport2DPixels, float2 viewport2DWorld, bool hitTest, float3 viewport3DWorld, MouseButtonEventArgs e);
}
