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
using Warp.Sociology;
using Warp.Tools;

namespace M.Controls.Sociology.Dialogs
{
    /// <summary>
    /// Interaction logic for DialogSpeciesShift.xaml
    /// </summary>
    public partial class DialogSpeciesShift : UserControl
    {
        public event Action Shift;
        public event Action Close;

        public decimal ShiftX
        {
            get { return (decimal)GetValue(ShiftXProperty); }
            set { SetValue(ShiftXProperty, value); }
        }
        public static readonly DependencyProperty ShiftXProperty = DependencyProperty.Register("ShiftX", typeof(decimal), typeof(DialogSpeciesShift), new PropertyMetadata(0M));

        public decimal ShiftY
        {
            get { return (decimal)GetValue(ShiftYProperty); }
            set { SetValue(ShiftYProperty, value); }
        }
        public static readonly DependencyProperty ShiftYProperty = DependencyProperty.Register("ShiftY", typeof(decimal), typeof(DialogSpeciesShift), new PropertyMetadata(0M));

        public decimal ShiftZ
        {
            get { return (decimal)GetValue(ShiftZProperty); }
            set { SetValue(ShiftZProperty, value); }
        }
        public static readonly DependencyProperty ShiftZProperty = DependencyProperty.Register("ShiftZ", typeof(decimal), typeof(DialogSpeciesShift), new PropertyMetadata(0M));



        private Particle[] ParticlesOld;
        public Particle[] ParticlesFinal;

        public DialogSpeciesShift(Population population, Species species)
        {
            InitializeComponent();

            DataContext = this;

            ParticlesOld = species.Particles;
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private void ButtonExpand_OnClick(object sender, RoutedEventArgs e)
        {
            List<Particle> ShiftedParticles = new List<Particle>();
            float3 AdditionalShiftAngstrom = new float3((float)ShiftX, (float)ShiftY, (float)ShiftZ);

            foreach (var p in ParticlesOld)
            {
                Matrix3 R0 = Matrix3.Euler(p.Angles[0] * Helper.ToRad);
                float3 RotatedShift = R0 * AdditionalShiftAngstrom;

                Particle ShiftedParticle = p.GetCopy();

                for (int t = 0; t < p.Coordinates.Length; t++)
                    ShiftedParticle.Coordinates[t] += RotatedShift;

                ShiftedParticles.Add(ShiftedParticle);
            }

            ParticlesFinal = ShiftedParticles.ToArray();

            Shift?.Invoke();
        }
    }
}
