using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Warp;
using Warp.Tools;

namespace M
{
    public class Options : WarpBase
    {
        private OptionsRuntime _Runtime = new OptionsRuntime();
        public OptionsRuntime Runtime
        {
            get { return _Runtime; }
            set { if (value != _Runtime) { _Runtime = value; OnPropertyChanged(); } }
        }

        #region Runtime

        public MainWindow MainWindow;

        public void UpdateGPUStats()
        {
            int NDevices = GPU.GetDeviceCount();
            string[] Stats = new string[NDevices];
            for (int i = 0; i < NDevices; i++)
                Stats[i] = "GPU" + i + ": " + GPU.GetFreeMemory(i) + " MB";
            Runtime.GPUStats = string.Join(", ", Stats);
        }

        #endregion

        public Options()
        {
        }

        private void SubOptions_PropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
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
    }

    public class OptionsRuntime : WarpBase
    {
        private int _DeviceCount = 0;
        public int DeviceCount
        {
            get { return _DeviceCount; }
            set { if (value != _DeviceCount) { _DeviceCount = value; OnPropertyChanged(); } }
        }

        private string _GPUStats = "";
        public string GPUStats
        {
            get { return _GPUStats; }
            set { if (value != _GPUStats) { _GPUStats = value; OnPropertyChanged(); } }
        }
    }
}