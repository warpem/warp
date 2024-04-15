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

namespace M.Controls.Sociology.Dialogs
{
    /// <summary>
    /// Interaction logic for DialogCreateSourceFromSettings.xaml
    /// </summary>
    public partial class DialogCreateSourceFromSettings : UserControl
    {
        public event Action Create;
        public event Action Close;

        public DialogCreateSourceFromSettings()
        {
            InitializeComponent();
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private void ButtonCreate_OnClick(object sender, RoutedEventArgs e)
        {
            Create?.Invoke();
        }
    }
}
