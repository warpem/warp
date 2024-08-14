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
    /// Interaction logic for CoordsImportWindow.xaml
    /// </summary>
    public partial class CoordsImportWindow
    {
        public MainWindow ParentWindow;

        public CoordsImportWindow()
        {
            InitializeComponent();
        }

        private void ButtonImportCoords_OnClick(object sender, RoutedEventArgs e)
        {
            Close();
            ParentWindow.PointsImport();
        }
    }
}
