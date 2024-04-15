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

namespace Warp
{
    /// <summary>
    /// Interaction logic for OldschoolBar.xaml
    /// </summary>
    public partial class OldschoolBar : UserControl
    {
        public bool IsIndeterminate
        {
            get { return (bool)GetValue(IsIndeterminateProperty); }
            set { SetValue(IsIndeterminateProperty, value); }
        }
        public static readonly DependencyProperty IsIndeterminateProperty = DependencyProperty.Register("IsIndeterminate", typeof(bool), typeof(OldschoolBar), new PropertyMetadata(false, (o, args) => ((OldschoolBar)o).RenderMouse()));

        public int Value
        {
            get { return (int)GetValue(ValueProperty); }
            set { SetValue(ValueProperty, value); }
        }
        public static readonly DependencyProperty ValueProperty = DependencyProperty.Register("Value", typeof(int), typeof(OldschoolBar), new PropertyMetadata(0, (o, args) => ((OldschoolBar)o).RenderMouse()));

        public int MaxValue
        {
            get { return (int)GetValue(MaxValueProperty); }
            set { SetValue(MaxValueProperty, value); }
        }
        public static readonly DependencyProperty MaxValueProperty = DependencyProperty.Register("MaxValue", typeof(int), typeof(OldschoolBar), new PropertyMetadata(1, (o, args) => ((OldschoolBar)o).RenderMouse()));

        public OldschoolBar()
        {
            InitializeComponent();

            SizeChanged += OldschoolBar_SizeChanged;

            string[] FoodSelection = "🍏 🍐 🍌 🍉 🍇 🍓 🍈 🍒 🍍 🌽 🍯 🍞 🍗 🍔 🍟 🌭 🍕 🌮 🍨 🍦 🍰 🎂 🍭 🍩 🍺 🍷 🍸 🍹 ☕️".Split(new [] {' '}, StringSplitOptions.RemoveEmptyEntries);
            Random Rand = new Random();
            string FoodSymbol = FoodSelection[Rand.Next(FoodSelection.Length)];
            TextFood.Text = FoodSymbol;

            RenderMouse();
        }

        private void OldschoolBar_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            RenderMouse();
        }

        private void RenderMouse()
        {
            if (IsIndeterminate)
            {
                BarIndeterminate.Visibility = Visibility.Visible;
                RectangleTail.Visibility = Visibility.Collapsed;
                ImageMouse.Visibility = Visibility.Collapsed;
                TextFood.Visibility = Visibility.Hidden;
            }
            else
            {
                BarIndeterminate.Visibility = Visibility.Collapsed;
                RectangleTail.Visibility = Visibility.Visible;
                ImageMouse.Visibility = Visibility.Visible;
                TextFood.Visibility = Visibility.Visible;

                double ProgressFraction = (double)Value / MaxValue;

                ImageMouse.Margin = new Thickness(ProgressFraction * (this.ActualWidth - ImageMouse.ActualWidth), 0, 0, 0);
                RectangleTail.Width = ProgressFraction * (this.ActualWidth - ImageMouse.ActualWidth);// + ImageMouse.ActualWidth / 2;
                VisualBrushTrail.Viewport = new Rect(0, 0, 10 / RectangleTail.Width, 1.01);
                Canvas.SetLeft(TextFood, this.ActualWidth);
                Canvas.SetTop(TextFood, (this.ActualHeight - TextFood.ActualHeight) * 0.5);
            }
        }
    }
}
