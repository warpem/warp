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
    /// Interaction logic for EditableTextBlock.xaml
    /// </summary>
    public partial class EditableTextBlock : UserControl
    {
        string PreviousValue = "";
        bool CanTrigger = true;

        public string Text
        {
            get { return (string)GetValue(TextProperty); }
            set { SetValue(TextProperty, value); }
        }
        public static readonly DependencyProperty TextProperty =
            DependencyProperty.Register("Text", typeof(string), typeof(EditableTextBlock), new PropertyMetadata(""));

        public bool IsNumeric
        {
            get { return (bool)GetValue(IsNumericProperty); }
            set { SetValue(IsNumericProperty, value); }
        }
        public static readonly DependencyProperty IsNumericProperty =
            DependencyProperty.Register("IsNumeric", typeof(bool), typeof(EditableTextBlock), new PropertyMetadata(false));

        public bool IsManuallyTriggered
        {
            get { return (bool)GetValue(IsManuallyTriggeredProperty); }
            set { SetValue(IsManuallyTriggeredProperty, value); }
        }
        public static readonly DependencyProperty IsManuallyTriggeredProperty =
            DependencyProperty.Register("IsManuallyTriggered", typeof(bool), typeof(EditableTextBlock), new PropertyMetadata(false));

        public event Action EditCompleted;

        public EditableTextBlock()
        {
            InitializeComponent();
        }

        private void UserControl_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (IsManuallyTriggered)
                return;

            if (CanTrigger)
                TriggerEdit();
        }

        private void EditText_LostFocus(object sender, RoutedEventArgs e)
        {
            EditCompleted?.Invoke();

            EditText.Visibility = Visibility.Collapsed;
            CanTrigger = true;
        }

        private void EditText_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
                Keyboard.Focus(this);
        }

        public void TriggerEdit()
        {
            CanTrigger = false;
            PreviousValue = Text;

            EditText.Visibility = Visibility.Visible;
            EditText.FontSize = FontSize;
            Dispatcher.BeginInvoke(System.Windows.Threading.DispatcherPriority.ApplicationIdle, new Action(() => { EditText.Focus(); }));
            EditText.SelectAll();
        }
    }
}
