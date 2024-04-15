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

namespace Sparta
{
    /// <summary>
    /// Interaction logic for FormattedTextBlock.xaml
    /// </summary>
    public partial class FormattedTextBlock : UserControl
    {
        public string Value
        {
            get { return (string)GetValue(ValueProperty); }
            set { SetValue(ValueProperty, value); }
        }
        public static readonly DependencyProperty ValueProperty =
            DependencyProperty.Register("Value", typeof(string), typeof(FormattedTextBlock), new PropertyMetadata(""));

        public string TextFormat
        {
            get { return (string)GetValue(TextFormatProperty); }
            set { SetValue(TextFormatProperty, value); }
        }
        public static readonly DependencyProperty TextFormatProperty =
            DependencyProperty.Register("TextFormat", typeof(string), typeof(FormattedTextBlock), new PropertyMetadata("{0}"));

        public string ToolTipValue
        {
            get { return (string)GetValue(ToolTipValueProperty); }
            set { SetValue(ToolTipValueProperty, value); }
        }
        public static readonly DependencyProperty ToolTipValueProperty =
            DependencyProperty.Register("ToolTipValue", typeof(string), typeof(FormattedTextBlock), new PropertyMetadata(null));

        public string ToolTipFormat
        {
            get { return (string)GetValue(ToolTipFormatProperty); }
            set { SetValue(ToolTipFormatProperty, value); }
        }
        public static readonly DependencyProperty ToolTipFormatProperty =
            DependencyProperty.Register("ToolTipFormat", typeof(string), typeof(FormattedTextBlock), new PropertyMetadata("{0}"));

        public FormattedTextBlock()
        {
            InitializeComponent();
        }
    }
}
