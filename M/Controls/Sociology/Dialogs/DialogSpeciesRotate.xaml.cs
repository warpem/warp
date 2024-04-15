using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Warp;
using Warp.Sociology;
using Warp.Tools;

namespace M.Controls.Sociology.Dialogs
{
    /// <summary>
    /// Interaction logic for DialogSpeciesRotate.xaml
    /// </summary>
    public partial class DialogSpeciesRotate : System.Windows.Controls.UserControl
    {
        public event Action Rotate;
        public event Action Close;

        public decimal AngleRot
        {
            get { return (decimal)GetValue(AngleRotProperty); }
            set { SetValue(AngleRotProperty, value); }
        }
        public static readonly DependencyProperty AngleRotProperty = DependencyProperty.Register("AngleRot", typeof(decimal), typeof(DialogSpeciesRotate), new PropertyMetadata(0M, (sender, args) => ((DialogSpeciesRotate)sender).UpdateMatrix()));

        public decimal AngleTilt
        {
            get { return (decimal)GetValue(AngleTiltProperty); }
            set { SetValue(AngleTiltProperty, value); }
        }
        public static readonly DependencyProperty AngleTiltProperty = DependencyProperty.Register("AngleTilt", typeof(decimal), typeof(DialogSpeciesRotate), new PropertyMetadata(0M, (sender, args) => ((DialogSpeciesRotate)sender).UpdateMatrix()));

        public decimal AnglePsi
        {
            get { return (decimal)GetValue(AnglePsiProperty); }
            set { SetValue(AnglePsiProperty, value); }
        }
        public static readonly DependencyProperty AnglePsiProperty = DependencyProperty.Register("AnglePsi", typeof(decimal), typeof(DialogSpeciesRotate), new PropertyMetadata(0M, (sender, args) => ((DialogSpeciesRotate)sender).UpdateMatrix()));


        private Particle[] ParticlesOld;
        public Particle[] ParticlesFinal;

        private Projector Proj;

        public DialogSpeciesRotate(Population population, Species species, Image volume)
        {
            InitializeComponent();

            DataContext = this;

            ParticlesOld = species.Particles;

            Task.Run(() =>
            {
                Image VolumeScaled = volume.AsScaled(new int3(128)).AndFreeParent();
                Proj = new Projector(VolumeScaled, 2);
                VolumeScaled.Dispose();

                Dispatcher.Invoke(() =>
                {
                    ProgressRenderer.Visibility = Visibility.Collapsed;
                    RendererSlices.Visibility = Visibility.Visible;
                    UpdateMatrix();
                });
            });
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private void ButtonExpand_OnClick(object sender, RoutedEventArgs e)
        {
            List<Particle> RotatedParticles = new List<Particle>();
            Matrix3 R = Matrix3.Euler(new float3((float)AngleRot, (float)AngleTilt, (float)AnglePsi) * Helper.ToRad).Transposed();

            foreach (var p in ParticlesOld)
            {
                float3[] AnglesNew = p.Angles.Select(a => Matrix3.EulerFromMatrix(Matrix3.Euler(a * Helper.ToRad) * R) * Helper.ToDeg).ToArray();

                Particle Rotated = p.GetCopy();
                Rotated.Angles = AnglesNew;

                RotatedParticles.Add(Rotated);
            }

            ParticlesFinal = RotatedParticles.ToArray();

            Proj.Dispose();
            RendererSlices.Volume.Dispose();

            Rotate?.Invoke();
        }

        private void UpdateMatrix()
        {
            float3 Angles = new float3((float)AngleRot, (float)AngleTilt, (float)AnglePsi) * Helper.ToRad;
            Matrix3 R = Matrix3.Euler(Angles);

            TextM11.Text = R.M11.ToString("F3", CultureInfo.InvariantCulture);
            TextM21.Text = R.M21.ToString("F3", CultureInfo.InvariantCulture);
            TextM31.Text = R.M31.ToString("F3", CultureInfo.InvariantCulture);

            TextM12.Text = R.M12.ToString("F3", CultureInfo.InvariantCulture);
            TextM22.Text = R.M22.ToString("F3", CultureInfo.InvariantCulture);
            TextM32.Text = R.M32.ToString("F3", CultureInfo.InvariantCulture);

            TextM13.Text = R.M13.ToString("F3", CultureInfo.InvariantCulture);
            TextM23.Text = R.M23.ToString("F3", CultureInfo.InvariantCulture);
            TextM33.Text = R.M33.ToString("F3", CultureInfo.InvariantCulture);

            if (Proj == null)
                return;

            Image RotatedFT = Proj.Project(new int3(128), new[] { Angles });
            Image Rotated = RotatedFT.AsIFFT(true).AndDisposeParent();
            Rotated.RemapFromFT(true);
            RendererSlices.SetVolumeFrom(Rotated);
            RendererSlices.UpdateRendering();
            Rotated.Dispose();
        }
    }
}
