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
    public partial class DialogSpeciesSymExpand : UserControl
    {
        public event Action Expand;
        public event Action Close;

        public string SpeciesSymmetry
        {
            get { return (string)GetValue(SpeciesSymmetryProperty); }
            set { SetValue(SpeciesSymmetryProperty, value); }
        }
        public static readonly DependencyProperty SpeciesSymmetryProperty = DependencyProperty.Register("SpeciesSymmetry", typeof(string), typeof(DialogSpeciesSymExpand), new PropertyMetadata("C1"));
               
        public string SpeciesSymmetryRemaining
        {
            get { return (string)GetValue(SpeciesSymmetryRemainingProperty); }
            set { SetValue(SpeciesSymmetryRemainingProperty, value); }
        }
        public static readonly DependencyProperty SpeciesSymmetryRemainingProperty = DependencyProperty.Register("SpeciesSymmetryRemaining", typeof(string), typeof(DialogSpeciesSymExpand), new PropertyMetadata("C1"));

        private Particle[] ParticlesOld;
        public Particle[] ParticlesFinal;

        public DialogSpeciesSymExpand(Population population, Species species)
        {
            InitializeComponent();

            DataContext = this;

            SpeciesSymmetry = species.Symmetry;
            ParticlesOld = species.Particles;
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private void ButtonExpand_OnClick(object sender, RoutedEventArgs e)
        {
            Symmetry Sym = new Symmetry(SpeciesSymmetry);
            Matrix3[] SymMats = Sym.GetRotationMatrices();

            List<Particle> ExpandedParticles = new List<Particle>();
            Matrix3[] Angles = new Matrix3[ParticlesOld[0].Angles.Length];

            foreach (var p in ParticlesOld)
            {
                for (int i = 0; i < Angles.Length; i++)
                    Angles[i] = Matrix3.Euler(p.Angles[i] * Helper.ToRad);

                foreach (var m in SymMats)
                {
                    float3[] AnglesNew = Angles.Select(a => Matrix3.EulerFromMatrix(a * m) * Helper.ToDeg).ToArray();
                    Particle RotatedParticle = p.GetCopy();
                    RotatedParticle.Angles = AnglesNew;

                    ExpandedParticles.Add(RotatedParticle);
                }
            }

            ParticlesFinal = ExpandedParticles.ToArray();

            Expand?.Invoke();
        }
    }
}
