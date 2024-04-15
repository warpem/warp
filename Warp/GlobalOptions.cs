using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;

namespace Warp
{
    public class GlobalOptions : WarpBase
    {
        private Guid Secret = Guid.Parse("5527e951-beab-46d3-ba75-73ea94d1a9df");

        private bool _PromptShown = false;
        [WarpSerializable]
        public bool PromptShown
        {
            get { return _PromptShown; }
            set
            {
                if (value != _PromptShown)
                {
                    _PromptShown = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _AllowCollection = false;
        [WarpSerializable]
        public bool AllowCollection
        {
            get { return _AllowCollection; }
            set
            {
                if (value != _AllowCollection)
                {
                    _AllowCollection = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _ShowBoxNetReminder = true;
        [WarpSerializable]
        public bool ShowBoxNetReminder
        {
            get { return _ShowBoxNetReminder; }
            set { if (value != _ShowBoxNetReminder) { _ShowBoxNetReminder = value; OnPropertyChanged(); } }
        }

        private bool _CheckForUpdates = true;
        [WarpSerializable]
        public bool CheckForUpdates
        {
            get { return _CheckForUpdates; }
            set { if (value != _CheckForUpdates) { _CheckForUpdates = value; OnPropertyChanged(); } }
        }

        private bool _ShowTiffReminder = true;
        [WarpSerializable]
        public bool ShowTiffReminder
        {
            get { return _ShowTiffReminder; }
            set { if (value != _ShowTiffReminder) { _ShowTiffReminder = value; OnPropertyChanged(); } }
        }

        private int _ProcessesPerDevice = 1;
        [WarpSerializable]
        public int ProcessesPerDevice
        {
            get { return _ProcessesPerDevice; }
            set { if (value != _ProcessesPerDevice) { _ProcessesPerDevice = value; OnPropertyChanged(); } }
        }

        private string _ExcludeDevices = "";
        [WarpSerializable]
        public string ExcludeDevices
        {
            get { return _ExcludeDevices; }
            set { if (value != _ExcludeDevices) { _ExcludeDevices = value; OnPropertyChanged(); } }
        }

        private int _APIPort = -1;
        [WarpSerializable]
        public int APIPort
        {
            get { return _APIPort; }
            set { if (value != _APIPort) { _APIPort = value; OnPropertyChanged(); } }
        }

        public void Load(string path)
        {
            try
            {
                using (Stream SettingsStream = File.OpenRead(path))
                {
                    XPathDocument Doc = new XPathDocument(SettingsStream);
                    XPathNavigator Reader = Doc.CreateNavigator();
                    Reader.MoveToRoot();

                    Reader.MoveToRoot();
                    Reader.MoveToChild("Settings", "");

                    ReadFromXML(Reader);
                }
            }
            catch { }
        }

        public void Save(string path)
        {
            XmlTextWriter Writer = new XmlTextWriter(File.Create(path), Encoding.Unicode);
            Writer.Formatting = Formatting.Indented;
            Writer.IndentChar = '\t';
            Writer.Indentation = 1;
            Writer.WriteStartDocument();
            Writer.WriteStartElement("Settings");

            WriteToXML(Writer);

            Writer.WriteEndElement();
            Writer.WriteEndDocument();
            Writer.Flush();
            Writer.Close();
        }
    }
}
