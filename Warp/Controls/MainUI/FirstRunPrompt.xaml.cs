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
    /// Interaction logic for AnalyticsPrompt.xaml
    /// </summary>
    public partial class FirstRunPrompt : UserControl
    {
        public event Action Close;

        public FirstRunPrompt()
        {
            InitializeComponent();
        }

        private void ButtonClose_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private void Hyperlink_OnClick(object sender, RoutedEventArgs e)
        {
            System.Diagnostics.Process.Start("http://www.warpem.com/warp/?page_id=51");
        }
    }
}
