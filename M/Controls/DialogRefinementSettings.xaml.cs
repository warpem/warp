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

namespace M.Controls
{
    /// <summary>
    /// Interaction logic for DialogRefinementSettings.xaml
    /// </summary>
    public partial class DialogRefinementSettings : UserControl
    {
        public event Action StartRefinement;
        public event Action Close;

        public DialogRefinementSettings()
        {
            InitializeComponent();
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private void ButtonRefine_Click(object sender, RoutedEventArgs e)
        {
            StartRefinement?.Invoke();
        }
    }
}
