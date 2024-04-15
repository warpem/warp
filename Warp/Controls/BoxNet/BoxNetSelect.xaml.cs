using System;
using System.Collections.Generic;
using System.IO;
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
using MahApps.Metro.Controls.Dialogs;
using Warp.Tools;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for BoxNetSelect.xaml
    /// </summary>
    public partial class BoxNetSelect : UserControl
    {
        private string PreviousModelName;

        public string ModelName;
        public OptionsWarp Options;
        public event Action Close;

        public BoxNetSelect(string previousModelName, OptionsWarp options)
        {
            InitializeComponent();

            PreviousModelName = previousModelName;
            Options = options;

            string ModelFolder = System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet3models");

            if (Directory.Exists(ModelFolder))
                foreach (var modelPath in Directory.EnumerateFiles(ModelFolder, "*.pt"))
                {
                    string ModelName = Helper.PathToName(modelPath);

                    ListLocals.Items.Add(new ListBoxItem { Content = ModelName, ToolTip = modelPath });
                    if (ModelName == PreviousModelName)
                        ListLocals.SelectedIndex = ListLocals.Items.Count - 1;
                }

            if (ListLocals.Items.IsEmpty)
            {
                ButtonRetrain.Visibility = Visibility.Collapsed;
                ButtonSelect.Visibility = Visibility.Collapsed;
            }

            ListLocals_OnSelectionChanged(this, null);
        }

        private void ButtonRetrain_OnClick(object sender, RoutedEventArgs e)
        {
            MainWindow Win = (MainWindow)Application.Current.MainWindow;

            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            BoxNetTrain DialogContent = new BoxNetTrain(ModelName, Options);
            DialogContent.Close += async () =>
            {
                await Win.HideMetroDialogAsync(Dialog);
                if (string.IsNullOrEmpty(DialogContent.NewName))
                    return;

                ModelName = DialogContent.NewName;

                ListLocals.Items.Clear();

                string ModelFolder = System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet3models");

                if (Directory.Exists(ModelFolder))
                    foreach (var modelPath in Directory.EnumerateFiles(ModelFolder, "*.pt"))
                    {
                        string ModelName = Helper.PathToName(modelPath);

                        ListLocals.Items.Add(new ListBoxItem { Content = ModelName, ToolTip = modelPath });
                        if (ModelName == PreviousModelName)
                            ListLocals.SelectedIndex = ListLocals.Items.Count - 1;
                    }

                ListLocals_OnSelectionChanged(this, null);
            };
            Dialog.Content = DialogContent;

            Win.ShowMetroDialogAsync(Dialog);
        }

        private void ButtonSelect_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            ModelName = PreviousModelName;
            Close?.Invoke();
        }

        private void ListLocals_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ListLocals.SelectedItem == null)
            {
                ButtonRetrain.Visibility = Visibility.Collapsed;
                ButtonSelect.Visibility = Visibility.Collapsed;
            }
            else
            {
                ButtonRetrain.Visibility = Visibility.Visible;
                ButtonSelect.Visibility = Visibility.Visible;
                ModelName = (string)((ListBoxItem)ListLocals.SelectedItem).Content;
            }
        }

        private void ButtonBrowseRepository_OnClick(object sender, RoutedEventArgs e)
        {
            System.Diagnostics.Process.Start("http://boxnet.warpem.com/models/");
        }
    }
}
