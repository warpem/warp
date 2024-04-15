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
    /// Interaction logic for CreatePopulation.xaml
    /// </summary>
    public partial class DialogCreatePopulation : UserControl
    {
        public event Action Create;
        public event Action Close;

        public DialogCreatePopulation()
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

        private void ButtonChangeFolder_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog();
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                if (!IOHelper.CheckFolderPermission(Dialog.SelectedPath))
                {
                    MessageBox.Show("Don't have permission to access the selected folder.");
                    return;
                }

                if (Dialog.SelectedPath.Last() != '\\')
                    Dialog.SelectedPath += '\\';

                ButtonChangeFolder.Content = Dialog.SelectedPath;
                ButtonCreate.Visibility = Visibility.Visible;
            }
        }
    }
}
