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
    /// Interaction logic for SliceRenderer.xaml
    /// </summary>
    public partial class MultiSliceRenderer : UserControl
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
        public static readonly DependencyProperty VolumeProperty = DependencyProperty.Register("Volume", typeof(Image), typeof(MultiSliceRenderer), new PropertyMetadata(null, (o, args) =>
        {
            ((MultiSliceRenderer)o).SetAutoThreshold();
            ((MultiSliceRenderer)o).RenderDataChanged(args);
        }));

        public int PositionX
        {
            get { return (int)GetValue(PositionXProperty); }
            set { SetValue(PositionXProperty, value); }
        }
        public static readonly DependencyProperty PositionXProperty = DependencyProperty.Register("PositionX", typeof(int), typeof(MultiSliceRenderer), new PropertyMetadata(0, (o, args) =>
        {
            ((MultiSliceRenderer)o).RenderSettingChanged(args);
        }));
               
        public int PositionY
        {
            get { return (int)GetValue(PositionYProperty); }
            set { SetValue(PositionYProperty, value); }
        }
        public static readonly DependencyProperty PositionYProperty = DependencyProperty.Register("PositionY", typeof(int), typeof(MultiSliceRenderer), new PropertyMetadata(0, (o, args) =>
        {
            ((MultiSliceRenderer)o).RenderSettingChanged(args);
        }));
        
        public int PositionZ
        {
            get { return (int)GetValue(PositionZProperty); }
            set { SetValue(PositionZProperty, value); }
        }
        public static readonly DependencyProperty PositionZProperty = DependencyProperty.Register("PositionZ", typeof(int), typeof(MultiSliceRenderer), new PropertyMetadata(0, (o, args) =>
        {
            ((MultiSliceRenderer)o).RenderSettingChanged(args);
        }));

        public decimal ThresholdLower
        {
            get { return (decimal)GetValue(ThresholdLowerProperty); }
            set { SetValue(ThresholdLowerProperty, value); }
        }
        public static readonly DependencyProperty ThresholdLowerProperty = DependencyProperty.Register("ThresholdLower", typeof(decimal), typeof(MultiSliceRenderer), new PropertyMetadata(0M, (o, args) =>
        {
            ((MultiSliceRenderer)o).RenderSettingChanged(args);
        }));

        public decimal ThresholdUpper
        {
            get { return (decimal)GetValue(ThresholdUpperProperty); }
            set { SetValue(ThresholdUpperProperty, value); }
        }
        public static readonly DependencyProperty ThresholdUpperProperty = DependencyProperty.Register("ThresholdUpper", typeof(decimal), typeof(MultiSliceRenderer), new PropertyMetadata(1M, (o, args) =>
        {
            ((MultiSliceRenderer)o).RenderSettingChanged(args);
        }));

        public bool PointerShow
        {
            get { return (bool)GetValue(PointerShowProperty); }
            set { SetValue(PointerShowProperty, value); }
        }
        public static readonly DependencyProperty PointerShowProperty = DependencyProperty.Register("PointerShow", typeof(bool), typeof(MultiSliceRenderer), new PropertyMetadata(false, (o, args) =>
        {
            ((MultiSliceRenderer)o).RenderSettingChanged(args);
        }));

        public ColorScale ColorScale
        {
            get { return (ColorScale)GetValue(ColorScaleProperty); }
            set { SetValue(ColorScaleProperty, value); }
        }
        public static readonly DependencyProperty ColorScaleProperty = DependencyProperty.Register("ColorScale", typeof(ColorScale), typeof(MultiSliceRenderer), new PropertyMetadata(null, (o, args) =>
        {
            ((MultiSliceRenderer)o).RenderSettingChanged(args);
        }));


        private int2 DimsImage;

        public MultiSliceRenderer()
        {
            InitializeComponent();

            DataContext = this;

            SizeChanged += SliceRenderer_SizeChanged;
        }

        private void SliceRenderer_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            UpdateRendering();
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

        private Image GetSlice(int dim1, int dim2, int position, int oversample)
        {
            if (Volume == null)
                return new Image(new int3(1));

            int2 Dims = new int2(Volume.Dims.Dimension(dim1), Volume.Dims.Dimension(dim2));

            int dim3 = 3 - dim1 - dim2;

            Image Slice = new Image(new int3(Dims));
            float[] SliceData = Slice.GetHost(Intent.Write)[0];
            float[][] VolumeData = Volume.GetHost(Intent.Read);

            int3 Vec1 = dim1 == 0 ? int3.UnitX : (dim1 == 1 ? int3.UnitY : int3.UnitZ);
            int3 Vec2 = dim2 == 0 ? int3.UnitX : (dim2 == 1 ? int3.UnitY : int3.UnitZ);

            int3 PositionOffset = (dim3 == 0 ? int3.UnitX : (dim3 == 1 ? int3.UnitY : int3.UnitZ)) * Math.Max(0, Math.Min(position, Volume.Dims.Dimension(dim3) - 1));

            for (int y = 0; y < Dims.Y; y++)
            {
                for (int x = 0; x < Dims.X; x++)
                {
                    int3 Pos = Vec1 * x + Vec2 * y + PositionOffset;
                    SliceData[y * Dims.X + x] = VolumeData[Pos.Z][Pos.Y * Volume.Dims.X + Pos.X];
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

            int3[] Axes =
            {
                new int3(0, 2, 1),
                new int3(0, 1, 2),
                new int3(2, 1, 0)
            };

            System.Windows.Controls.Image[] Viewers = { ImageViewerXZ, ImageViewerXY, ImageViewerZY };

            int3 Position = new int3(PositionX, PositionY, PositionZ);

            for (int i = 0; i < Axes.Length; i++)
            {
                Image Slice = GetSlice(Axes[i].X, Axes[i].Y, Position.Dimension(Axes[i].Z), 2);
                ImageSource Bitmap = DataToBitmap(Slice.GetHost(Intent.Read)[0], new int2(Slice.Dims));

                Slice.Dispose();

                Viewers[i].Source = Bitmap;
            }

            #region Update pointer positions

            Canvas.SetRight(PointerXZ, ImageViewerXZ.ActualWidth - 1 - (PositionX / (double)Volume.Dims.X) * ImageViewerXZ.ActualWidth - 4);
            Canvas.SetBottom(PointerXZ, (PositionZ / (double)Volume.Dims.Z) * ImageViewerXZ.ActualHeight - 4);

            Canvas.SetRight(PointerXY, ImageViewerXY.ActualWidth - 1 - (PositionX / (double)Volume.Dims.X) * ImageViewerXY.ActualWidth - 4);
            Canvas.SetTop(PointerXY, ImageViewerXY.ActualHeight - 1 - (PositionY / (double)Volume.Dims.Y) * ImageViewerXY.ActualHeight - 4);

            Canvas.SetLeft(PointerZY, (PositionZ / (double)Volume.Dims.Z) * ImageViewerZY.ActualWidth - 4);
            Canvas.SetTop(PointerZY, ImageViewerZY.ActualHeight - 1 - (PositionY / (double)Volume.Dims.Y) * ImageViewerZY.ActualHeight - 4);

            PointerXZ.Visibility = PointerShow ? Visibility.Visible : Visibility.Hidden;
            PointerXY.Visibility = PointerShow ? Visibility.Visible : Visibility.Hidden;
            PointerZY.Visibility = PointerShow ? Visibility.Visible : Visibility.Hidden;

            #endregion
        }

        private void ImageViewerXZ_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (Volume == null)
                return;

            Point Pos = e.GetPosition((IInputElement)sender);

            Pos.X /= ImageViewerXZ.ActualWidth;
            Pos.Y /= ImageViewerXZ.ActualHeight;

            RenderingEnabled = false;

            PositionX = (int)(Math.Max(0, Math.Min(1, Pos.X)) * Volume.Dims.X);
            PositionZ = Volume.Dims.Z - 1 - (int)(Math.Max(0, Math.Min(1, Pos.Y)) * Volume.Dims.Z);

            RenderingEnabled = true;
            UpdateRendering();
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

        private void ImageViewerZY_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (Volume == null)
                return;

            Point Pos = e.GetPosition((IInputElement)sender);

            Pos.X /= ImageViewerZY.ActualWidth;
            Pos.Y /= ImageViewerZY.ActualHeight;

            RenderingEnabled = false;

            PositionZ = (int)(Math.Max(0, Math.Min(1, Pos.X)) * Volume.Dims.Z);
            PositionY = Volume.Dims.Y - 1 - (int)(Math.Max(0, Math.Min(1, Pos.Y)) * Volume.Dims.Y);

            RenderingEnabled = true;
            UpdateRendering();
        }

        private void ImageViewerXZ_MouseMove(object sender, MouseEventArgs e)
        {
            if (Mouse.LeftButton == MouseButtonState.Pressed)
                ImageViewerXZ_MouseDown(sender, new MouseButtonEventArgs(e.MouseDevice, e.Timestamp, MouseButton.Left));
        }

        private void ImageViewerXY_MouseMove(object sender, MouseEventArgs e)
        {
            if (Mouse.LeftButton == MouseButtonState.Pressed)
                ImageViewerXY_MouseDown(sender, new MouseButtonEventArgs(e.MouseDevice, e.Timestamp, MouseButton.Left));
        }

        private void ImageViewerZY_MouseMove(object sender, MouseEventArgs e)
        {
            if (Mouse.LeftButton == MouseButtonState.Pressed)
                ImageViewerZY_MouseDown(sender, new MouseButtonEventArgs(e.MouseDevice, e.Timestamp, MouseButton.Left));
        }

        private void ImageViewerXZ_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (Volume == null)
                return;

            PositionY = Math.Max(0, Math.Min(Volume.Dims.Y - 1, PositionY - Math.Sign(e.Delta)));
        }

        private void ImageViewerXY_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (Volume == null)
                return;

            PositionZ = Math.Max(0, Math.Min(Volume.Dims.Z - 1, PositionZ - Math.Sign(e.Delta)));
        }

        private void ImageViewerZY_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (Volume == null)
                return;

            PositionX = Math.Max(0, Math.Min(Volume.Dims.X - 1, PositionX - Math.Sign(e.Delta)));
        }
    }
}
