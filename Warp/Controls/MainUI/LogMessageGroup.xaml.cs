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

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for LogMessageGroup.xaml
    /// </summary>
    public partial class LogMessageGroup : UserControl
    {
        private string _GroupName = "";
        public string GroupName
        {
            get { return _GroupName; }
            set
            {
                if (value != _GroupName)
                {
                    _GroupName = value;
                    TextGroupName.Text = value;
                }
            }
        }

        private IEnumerable<LogMessage> _Messages = null;
        public IEnumerable<LogMessage> Messages
        {
            get { return _Messages; }
            set
            {
                if (value != _Messages)
                {
                    _Messages = value;
                    UpdateMessages();
                }
            }
        }

        public LogMessageGroup()
        {
            InitializeComponent();
        }

        void UpdateMessages()
        {
            PanelMessages.Children.Clear();

            foreach (var message in Messages)
            {
                LogMessageDisplay Display = new LogMessageDisplay()
                {
                    DataContext = message
                };
                PanelMessages.Children.Add(Display);
            }
        }
    }
}
