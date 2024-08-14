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
using System.Windows.Shapes;

namespace Cube
{
    /// <summary>
    /// Interaction logic for CoordsExportWindow.xaml
    /// </summary>
    public partial class CoordsExportWindow
    {
        public MainWindow ParentWindow;

        public CoordsExportWindow()
        {
            InitializeComponent();
        }

        private void ButtonExportCoords_OnClick(object sender, RoutedEventArgs e)
        {
            Close();
            ParentWindow.PointsExport();
        }
    }
}
