using System;
using System.Collections.Generic;
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

namespace Warp
{
    /// <summary>
    /// Interaction logic for SingleSlice.xaml
    /// </summary>
    public partial class SingleSliceRenderer : UserControl
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
        public static readonly DependencyProperty VolumeProperty = DependencyProperty.Register("Volume", typeof(Image), typeof(SingleSliceRenderer), new PropertyMetadata(null, (o, args) =>
        {
            ((SingleSliceRenderer)o).SetAutoThreshold();
            ((SingleSliceRenderer)o).RenderDataChanged(args);
        }));

        public int PositionX
        {
            get { return (int)GetValue(PositionXProperty); }
            set { SetValue(PositionXProperty, value); }
        }
        public static readonly DependencyProperty PositionXProperty = DependencyProperty.Register("PositionX", typeof(int), typeof(SingleSliceRenderer), new PropertyMetadata(0, (o, args) =>
        {
            ((SingleSliceRenderer)o).RenderSettingChanged(args);
        }));

        public int PositionY
        {
            get { return (int)GetValue(PositionYProperty); }
            set { SetValue(PositionYProperty, value); }
        }
        public static readonly DependencyProperty PositionYProperty = DependencyProperty.Register("PositionY", typeof(int), typeof(SingleSliceRenderer), new PropertyMetadata(0, (o, args) =>
        {
            ((SingleSliceRenderer)o).RenderSettingChanged(args);
        }));

        public int PositionZ
        {
            get { return (int)GetValue(PositionZProperty); }
            set { SetValue(PositionZProperty, value); }
        }
        public static readonly DependencyProperty PositionZProperty = DependencyProperty.Register("PositionZ", typeof(int), typeof(SingleSliceRenderer), new PropertyMetadata(0, (o, args) =>
        {
            ((SingleSliceRenderer)o).RenderSettingChanged(args);
        }));

        public decimal AngleRot
        {
            get { return (decimal)GetValue(AngleRotProperty); }
            set { SetValue(AngleRotProperty, value); }
        }
        public static readonly DependencyProperty AngleRotProperty = DependencyProperty.Register("AngleRot", typeof(decimal), typeof(SingleSliceRenderer), new PropertyMetadata(0M, (o, args) =>
            {
                ((SingleSliceRenderer)o).RenderSettingChanged(args);
            }));

        public decimal AngleTilt
        {
            get { return (decimal)GetValue(AngleTiltProperty); }
            set { SetValue(AngleTiltProperty, value); }
        }
        public static readonly DependencyProperty AngleTiltProperty = DependencyProperty.Register("AngleTilt", typeof(decimal), typeof(SingleSliceRenderer), new PropertyMetadata(0M, (o, args) =>
        {
            ((SingleSliceRenderer)o).RenderSettingChanged(args);
        }));

        public decimal AnglePsi
        {
            get { return (decimal)GetValue(AnglePsiProperty); }
            set { SetValue(AnglePsiProperty, value); }
        }
        public static readonly DependencyProperty AnglePsiProperty = DependencyProperty.Register("AnglePsi", typeof(decimal), typeof(SingleSliceRenderer), new PropertyMetadata(0M, (o, args) =>
        {
            ((SingleSliceRenderer)o).RenderSettingChanged(args);
        }));

        public decimal RotatedX
        {
            get { return (decimal)GetValue(RotatedXProperty); }
            set { SetValue(RotatedXProperty, value); }
        }
        public static readonly DependencyProperty RotatedXProperty = DependencyProperty.Register("RotatedX", typeof(decimal), typeof(SingleSliceRenderer), new PropertyMetadata(0M));

        public decimal RotatedY
        {
            get { return (decimal)GetValue(RotatedYProperty); }
            set { SetValue(RotatedYProperty, value); }
        }
        public static readonly DependencyProperty RotatedYProperty = DependencyProperty.Register("RotatedY", typeof(decimal), typeof(SingleSliceRenderer), new PropertyMetadata(0M));

        public decimal RotatedZ
        {
            get { return (decimal)GetValue(RotatedZProperty); }
            set { SetValue(RotatedZProperty, value); }
        }
        public static readonly DependencyProperty RotatedZProperty = DependencyProperty.Register("RotatedZ", typeof(decimal), typeof(SingleSliceRenderer), new PropertyMetadata(0M));

        public decimal ThresholdLower
        {
            get { return (decimal)GetValue(ThresholdLowerProperty); }
            set { SetValue(ThresholdLowerProperty, value); }
        }
        public static readonly DependencyProperty ThresholdLowerProperty = DependencyProperty.Register("ThresholdLower", typeof(decimal), typeof(SingleSliceRenderer), new PropertyMetadata(0M, (o, args) =>
        {
            ((SingleSliceRenderer)o).RenderSettingChanged(args);
        }));

        public decimal ThresholdUpper
        {
            get { return (decimal)GetValue(ThresholdUpperProperty); }
            set { SetValue(ThresholdUpperProperty, value); }
        }
        public static readonly DependencyProperty ThresholdUpperProperty = DependencyProperty.Register("ThresholdUpper", typeof(decimal), typeof(SingleSliceRenderer), new PropertyMetadata(1M, (o, args) =>
        {
            ((SingleSliceRenderer)o).RenderSettingChanged(args);
        }));

        public bool PointerShow
        {
            get { return (bool)GetValue(PointerShowProperty); }
            set { SetValue(PointerShowProperty, value); }
        }
        public static readonly DependencyProperty PointerShowProperty = DependencyProperty.Register("PointerShow", typeof(bool), typeof(SingleSliceRenderer), new PropertyMetadata(false, (o, args) =>
        {
            ((SingleSliceRenderer)o).RenderSettingChanged(args);
        }));

        public ColorScale ColorScale
        {
            get { return (ColorScale)GetValue(ColorScaleProperty); }
            set { SetValue(ColorScaleProperty, value); }
        }
        public static readonly DependencyProperty ColorScaleProperty = DependencyProperty.Register("ColorScale", typeof(ColorScale), typeof(SingleSliceRenderer), new PropertyMetadata(null, (o, args) =>
        {
            ((SingleSliceRenderer)o).RenderSettingChanged(args);
        }));


        private int2 DimsImage;

        public SingleSliceRenderer()
        {
            InitializeComponent();

            DataContext = this;
            PopupControls.DataContext = this;

            SizeChanged += SingleSliceRenderer_SizeChanged;
        }

        private void SingleSliceRenderer_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            UpdateRendering();
        }

        private void SingleSliceRenderer_OnMouseEnter(object sender, MouseEventArgs e)
        {
            PopupControls.IsOpen = true;
        }

        private void SingleSliceRenderer_OnMouseLeave(object sender, MouseEventArgs e)
        {
            if (!IsMouseOver && !PopupControls.IsMouseOver)
            {
                PopupControls.IsOpen = false;
            }
        }

        private void RenderDataChanged(DependencyPropertyChangedEventArgs args)
        {
            UpdateRendering();
        }

        private void RenderSettingChanged(DependencyPropertyChangedEventArgs args)
        {
            UpdateRendering();
        }

        public void SetVolumeFrom(Image data)
        {
            if (Volume == null || Volume.Dims != data.Dims)
            {
                Volume?.Dispose();
                Volume = data.GetCopyGPU();

                SliderPositionX.MaxValue = data.Dims.X - 1;
                SliderPositionY.MaxValue = data.Dims.Y - 1;
                SliderPositionZ.MaxValue = data.Dims.Z - 1;

                PositionX = data.Dims.X / 2;
                PositionY = data.Dims.Y / 2;
                PositionZ = data.Dims.Z / 2;
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
            if (Volume != null)
            {
                float[] Data = Volume.GetHostContinuousCopy();
                ThresholdLower = (decimal)MathHelper.Min(Data);
                ThresholdUpper = (decimal)MathHelper.Max(Data);
            }
        }

        private Image GetSlice(int oversample)
        {
            if (Volume == null)
                return new Image(new int3(1));

            int2 Dims = new int2(Math.Max(Volume.Dims.X, Math.Max(Volume.Dims.Y, Volume.Dims.Z)));

            Matrix3 M = Matrix3.Euler((float)AngleRot * Helper.ToRad,
                                      (float)AngleTilt * Helper.ToRad,
                                      (float)AnglePsi * Helper.ToRad).Transposed();

            float3 ImageCenter = new float3(Dims.X / 2, Dims.Y / 2, Volume.Dims.Z / 2);
            float3 VolumeCenter = new float3(Volume.Dims / 2);
            
            Image Slice = new Image(new int3(Dims));
            float[] SliceData = Slice.GetHost(Intent.Write)[0];

            for (int y = 0; y < Dims.Y; y++)
            {
                for (int x = 0; x < Dims.X; x++)
                {
                    float3 Pos = M * (new float3(x, y, PositionZ) - ImageCenter) + VolumeCenter;
                    SliceData[y * Dims.X + x] = Volume.GetInterpolatedValue(Pos);
                }
            }

            if (oversample <= 1)
                return Slice;

            Image SliceOversampled = Slice.AsScaled(Dims * oversample);
            Slice.Dispose();

            return SliceOversampled;
        }

        private ImageSource DataToBitmap(float[] data, int2 dims)
        {
            ImageSource Bitmap;

            float Min = (float)ThresholdLower;
            float Range = (float)(ThresholdUpper - ThresholdLower);
            if (Range == 0)
                Range = 1e-5f;

            ColorScale Palette = ColorScale == null ? 
                                 new ColorScale(new[] { new float4(0, 0, 0, 1), new float4(1, 1, 1, 1) }) : 
                                 ColorScale;

            byte[] DataBytes = new byte[data.Length * 4];
            for (int y = 0; y < dims.Y; y++)
                for (int x = 0; x < dims.X; x++)
                {
                    float V = (data[y * dims.X + x] - Min) / Range;
                    float4 C = Palette.GetColor(V) * 255;

                    DataBytes[((dims.Y - 1 - y) * dims.X + x) * 4 + 3] = (byte)C.W;
                    DataBytes[((dims.Y - 1 - y) * dims.X + x) * 4 + 2] = (byte)C.X;
                    DataBytes[((dims.Y - 1 - y) * dims.X + x) * 4 + 1] = (byte)C.Y;
                    DataBytes[((dims.Y - 1 - y) * dims.X + x) * 4 + 0] = (byte)C.Z;
                }

            Bitmap = BitmapSource.Create(dims.X, dims.Y, 96, 96, PixelFormats.Bgra32, null, DataBytes, dims.X * 4);
            Bitmap.Freeze();

            return Bitmap;
        }

        public void UpdateRendering()
        {
            if (Volume == null || !RenderingEnabled)
                return;

            if (ActualWidth <= 1 || ActualHeight <= 1)
                return;

            int3 Position = new int3(PositionX, PositionY, PositionZ);

            Image Slice = GetSlice(2);
            ImageSource Bitmap = DataToBitmap(Slice.GetHost(Intent.Read)[0], new int2(Slice.Dims));

            Slice.Dispose();

            ImageViewerXY.Source = Bitmap;

            #region Update pointer positions

            Canvas.SetLeft(PointerXY, (PositionX / (double)Volume.Dims.X) * ImageViewerXY.ActualWidth - 4);
            Canvas.SetTop(PointerXY, ImageViewerXY.ActualHeight - 1 - (PositionY / (double)Volume.Dims.Y) * ImageViewerXY.ActualHeight - 4);

            PointerXY.Visibility = PointerShow ? Visibility.Visible : Visibility.Hidden;

            #endregion

            {
                int2 Dims = new int2(Math.Max(Volume.Dims.X, Math.Max(Volume.Dims.Y, Volume.Dims.Z)));

                Matrix3 M = Matrix3.Euler((float)AngleRot * Helper.ToRad,
                                          (float)AngleTilt * Helper.ToRad,
                                          (float)AnglePsi * Helper.ToRad).Transposed();

                float3 ImageCenter = new float3(Dims.X / 2, Dims.Y / 2, Volume.Dims.Z / 2);
                float3 VolumeCenter = new float3(Volume.Dims / 2);

                float3 Rotated = M * (new float3(PositionX, PositionY, PositionZ) - ImageCenter) + VolumeCenter;

                RotatedX = (decimal)Rotated.X;
                RotatedY = (decimal)Rotated.Y;
                RotatedZ = (decimal)Rotated.Z;
            }
        }

        private void ImageViewerXY_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (Volume == null)
                return;

            Point Pos = e.GetPosition((IInputElement)sender);

            Pos.X /= ImageViewerXY.ActualWidth;
            Pos.Y /= ImageViewerXY.ActualHeight;

            RenderingEnabled = false;

            PositionX = (int)(Math.Max(0, Math.Min(1, Pos.X)) * Volume.Dims.X);
            PositionY = Volume.Dims.Y - 1 - (int)(Math.Max(0, Math.Min(1, Pos.Y)) * Volume.Dims.Y);

            RenderingEnabled = true;
            UpdateRendering();
        }

        private void ImageViewerXY_MouseMove(object sender, MouseEventArgs e)
        {
            if (Mouse.LeftButton == MouseButtonState.Pressed)
                ImageViewerXY_MouseDown(sender, new MouseButtonEventArgs(e.MouseDevice, e.Timestamp, MouseButton.Left));
        }

        private void ImageViewerXY_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (Volume == null)
                return;

            PositionZ = Math.Max(0, Math.Min(Volume.Dims.Z - 1, PositionZ - Math.Sign(e.Delta)));
        }

        private void PresetXY_Click(object sender, RoutedEventArgs e)
        {
            RenderingEnabled = false;

            AngleRot = 0;
            AngleTilt = 0;
            AnglePsi = 0;

            RenderingEnabled = true;
            UpdateRendering();
        }

        private void PresetXZ_Click(object sender, RoutedEventArgs e)
        {
            RenderingEnabled = false;

            AngleRot = -90;
            AngleTilt = 90;
            AnglePsi = 90;

            RenderingEnabled = true;
            UpdateRendering();
        }

        private void PresetZY_Click(object sender, RoutedEventArgs e)
        {
            RenderingEnabled = false;

            AngleRot = 0;
            AngleTilt = -90;
            AnglePsi = 0;

            RenderingEnabled = true;
            UpdateRendering();
        }
    }
}
