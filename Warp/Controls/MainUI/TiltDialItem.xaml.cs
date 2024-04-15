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

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for TiltDialItem.xaml
    /// </summary>
    public partial class TiltDialItem : UserControl
    {
        public int TiltID
        {
            get { return (int)GetValue(TiltIDProperty); }
            set { SetValue(TiltIDProperty, value); }
        }
        public static readonly DependencyProperty TiltIDProperty = DependencyProperty.Register("TiltID", typeof(int), typeof(TiltDialItem), new PropertyMetadata(0));

        public bool IsSelected
        {
            get { return (bool)GetValue(IsSelectedProperty); }
            set { SetValue(IsSelectedProperty, value); }
        }
        public static readonly DependencyProperty IsSelectedProperty = DependencyProperty.Register("IsSelected", typeof(bool), typeof(TiltDialItem), new PropertyMetadata(true, (o, args) =>
        {
            TiltDialItem O = (TiltDialItem)o;
            bool NewVal = (bool)args.NewValue;
            O.EllipseKnob.Stroke = NewVal ? Brushes.Black : Brushes.Transparent;
            O.TextAngle.Visibility = NewVal ? Visibility.Visible : Visibility.Hidden;
            O.TextDose.Visibility = NewVal ? Visibility.Visible : Visibility.Hidden;
            O.CheckDoProcess.Visibility = NewVal ? Visibility.Visible : Visibility.Hidden;
        }));

        public bool DoProcess
        {
            get { return (bool)GetValue(DoProcessProperty); }
            set { SetValue(DoProcessProperty, value); }
        }
        public static readonly DependencyProperty DoProcessProperty = DependencyProperty.Register("DoProcess", typeof(bool), typeof(TiltDialItem), new PropertyMetadata(true, (o, args) =>
        {
            TiltDialItem O = (TiltDialItem)o;
            O.DoProcessChanged?.Invoke(o, args);
        }));

        public float Angle
        {
            get { return (float)GetValue(AngleProperty); }
            set { SetValue(AngleProperty, value); }
        }
        public static readonly DependencyProperty AngleProperty = DependencyProperty.Register("Angle", typeof(float), typeof(TiltDialItem), new PropertyMetadata(-9999f, (o, args) =>
        {
            TiltDialItem O = (TiltDialItem)o;
            float NewVal = (float)args.NewValue;
            string AngleString = (NewVal > 0 ? "+" : "") + NewVal.ToString("F1") + " °, ";
            O.TextAngle.Text = AngleString;
        }));

        public event PropertyChangedCallback SelectionChanged;
        public event PropertyChangedCallback DoProcessChanged;

        public float Dose
        {
            get { return (float)GetValue(DoseProperty); }
            set { SetValue(DoseProperty, value); }
        }
        public static readonly DependencyProperty DoseProperty = DependencyProperty.Register("Dose", typeof(float), typeof(TiltDialItem), new PropertyMetadata(0f, (o, args) =>
        {
            TiltDialItem O = (TiltDialItem)o;
            float NewVal = (float)args.NewValue;
            string DoseString = NewVal.ToString("F1") + " e⁻/Å²";
            O.TextDose.Text = DoseString;
        }));

        public Brush KnobBrush
        {
            get { return (Brush)GetValue(KnobBrushProperty); }
            set { SetValue(KnobBrushProperty, value); }
        }
        public static readonly DependencyProperty KnobBrushProperty = DependencyProperty.Register("KnobBrush", typeof(Brush), typeof(TiltDialItem), new PropertyMetadata(Brushes.Green));

        public TiltDialItem()
        {
            InitializeComponent();
        }

        private void TiltDialItem_OnMouseEnter(object sender, MouseEventArgs e)
        {
            if (!IsSelected)
            {
                EllipseKnob.Stroke = Brushes.Gray;
                TextAngle.Visibility = Visibility.Visible;
                TextDose.Visibility = Visibility.Visible;
                CheckDoProcess.Visibility = Visibility.Visible;
            }
        }

        private void TiltDialItem_OnMouseLeave(object sender, MouseEventArgs e)
        {
            if (!IsSelected)
            {
                EllipseKnob.Stroke = Brushes.Transparent;
                TextAngle.Visibility = Visibility.Hidden;
                TextDose.Visibility = Visibility.Hidden;
                CheckDoProcess.Visibility = Visibility.Hidden;
            }
        }

        private void OnMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (!IsSelected)
            {
                IsSelected = true;
                SelectionChanged?.Invoke(this, new DependencyPropertyChangedEventArgs(IsSelectedProperty, false, true));
            }
        }
    }
}
