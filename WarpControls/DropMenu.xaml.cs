using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
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
    /// Interaction logic for DropMenu.xaml
    /// </summary>
    public partial class DropMenu : UserControl
    {
        public ObservableCollection<string> Items
        {
            get { return (ObservableCollection<string>)GetValue(ItemsProperty); }
            set { SetValue(ItemsProperty, value); }
        }
        public static readonly DependencyProperty ItemsProperty = DependencyProperty.Register("Items", typeof(ObservableCollection<string>), typeof(DropMenu), new PropertyMetadata(null));

        public string SelectedItem
        {
            get { return (string)GetValue(SelectedItemProperty); }
            set { SetValue(SelectedItemProperty, value); }
        }
        public static readonly DependencyProperty SelectedItemProperty = DependencyProperty.Register("SelectedItem", typeof(string), typeof(DropMenu), new PropertyMetadata("", (sender, e) => ((DropMenu)sender).OnSelectionChanged()));

        public int SelectedIndex
        {
            get { return Items.IndexOf(SelectedItem); }
            set { ListItems.SelectedIndex = value; }
        }

        public event SelectionChangedEventHandler SelectionChanged;

        public DropMenu()
        {
            Items = new ObservableCollection<string>();

            InitializeComponent();
            TextSelection.DataContext = this;
            ListItems.DataContext = this;

            ListItems.PreviewMouseUp += (a, b) => PopupMenu.IsOpen = false;

            Items.CollectionChanged += Items_CollectionChanged;
            ListItems.SelectionChanged += ListItems_SelectionChanged;
        }

        private void ListItems_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ListItems.SelectedIndex >= 0 && ListItems.SelectedItem != null)
                SelectedItem = (string)ListItems.SelectedItem;
        }

        private void Items_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        {
            if (Items.Count > 0 && SelectedItem == "")
                SelectedItem = Items[0];
        }

        private void OnSelectionChanged()
        {
            if (SelectedItem != "" && Items.Contains(SelectedItem))
                ListItems.SelectedItem = SelectedItem;
            else
                ListItems.SelectedItem = null;

            SelectionChanged?.Invoke(this, null);
        }

        private void TextSelection_OnMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (!PopupMenu.IsOpen)
            {
                PopupMenu.IsOpen = true;
                ListItems.Focus();
            }
        }
    }
}
