using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    /// Interaction logic for SymmetrySelector.xaml
    /// </summary>
    public partial class SymmetrySelector : UserControl
    {
        public string Value
        {
            get { return (string)GetValue(ValueProperty); }
            set { SetValue(ValueProperty, value); }
        }
        public static readonly DependencyProperty ValueProperty = DependencyProperty.Register("Value", typeof(string), typeof(SymmetrySelector), new PropertyMetadata("C1", (o, args) => ((SymmetrySelector)o).OnValueChanged()));

        public decimal Multiplicity
        {
            get { return (decimal)GetValue(MultiplicityProperty); }
            set { SetValue(MultiplicityProperty, value); }
        }
        public static readonly DependencyProperty MultiplicityProperty = DependencyProperty.Register("Multiplicity", typeof(decimal), typeof(SymmetrySelector), new PropertyMetadata(1M, (o, args) => ((SymmetrySelector)o).OnMultiplicityChanged()));

        public event DependencyPropertyChangedEventHandler ValueChanged;

        private string[] Symmetries;

        public SymmetrySelector()
        {
            InitializeComponent();
            SliderMultiplicity.DataContext = this;
            
            Symmetries = typeof(SymmetryTypes).GetEnumNames();

            foreach (var symmetry in Symmetries)
                ComboGroups.Items.Add(symmetry);

            OnValueChanged();
            ComboGroups_OnSelectionChanged(this, null);
        }

        void OnValueChanged()
        {
            Symmetry Sym = new Symmetry(Value);

            Multiplicity = Sym.Multiplicity;
            ComboGroups.SelectedIndex = (int)Sym.Type;

            ValueChanged?.Invoke(this, new DependencyPropertyChangedEventArgs(ValueProperty, Value, Value));
        }

        void OnMultiplicityChanged()
        {
            Symmetry Sym = new Symmetry(Value);
            Sym.Multiplicity = (int)Multiplicity;

            Value = Sym.ToString();
        }

        private void ComboGroups_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            SymmetryTypes SymType = Symmetry.ParseType((string)ComboGroups.SelectedItem);

            SliderMultiplicity.Visibility = Symmetry.HasMultiplicity(SymType) ? Visibility.Visible : Visibility.Collapsed;
            if (SymType == SymmetryTypes.In || SymType == SymmetryTypes.InH)
            {
                SliderMultiplicity.MaxValue = 5;
                Multiplicity = Math.Min(5, Multiplicity);
            }
            else
            {
                SliderMultiplicity.MaxValue = 99;
            }

            Value = new Symmetry(SymType, (int)Multiplicity).ToString();
        }
    }
}
