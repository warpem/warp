using System.IO;
using System.Text;
using System.Xml;
using System.Xml.XPath;
using Warp;
using Warp.Tools;

namespace Cube
{
    public class Options : WarpBase
    {
        private string _PathTomogram = "";
        public string PathTomogram
        {
            get { return _PathTomogram; }
            set { if (value != _PathTomogram) { _PathTomogram = value; OnPropertyChanged(); } }
        }

        private decimal _InputLowpass = 20;
        public decimal InputLowpass
        {
            get { return _InputLowpass; }
            set { if (value != _InputLowpass) { _InputLowpass = value; OnPropertyChanged(); } }
        }

        private int _InputAverageSlices = 1;
        public int InputAverageSlices
        {
            get { return _InputAverageSlices; }
            set { if (value != _InputAverageSlices) { _InputAverageSlices = value; OnPropertyChanged(); } }
        }

        private decimal _DisplayIntensityMin = -2;
        public decimal DisplayIntensityMin
        {
            get { return _DisplayIntensityMin; }
            set { if (value != _DisplayIntensityMin) { _DisplayIntensityMin = value; OnPropertyChanged(); } }
        }

        private decimal _DisplayIntensityMax = 2;
        public decimal DisplayIntensityMax
        {
            get { return _DisplayIntensityMax; }
            set { if (value != _DisplayIntensityMax) { _DisplayIntensityMax = value; OnPropertyChanged(); } }
        }

        private decimal _ZoomLevel = 1;
        public decimal ZoomLevel
        {
            get { return _ZoomLevel; }
            set { if (value != _ZoomLevel) { _ZoomLevel = value; OnPropertyChanged(); } }
        }

        private int _NParticles = 0;
        public int NParticles
        {
            get { return _NParticles; }
            set { if (value != _NParticles) { _NParticles = value; OnPropertyChanged(); } }
        }

        private int _BoxSize = 16;
        public int BoxSize
        {
            get { return _BoxSize; }
            set { if (value != _BoxSize) { _BoxSize = value; OnPropertyChanged(); } }
        }

        private int _PlaneX = 0;
        public int PlaneX
        {
            get { return _PlaneX; }
            set { if (value != _PlaneX) { _PlaneX = value; OnPropertyChanged(); } }
        }

        private int _PlaneY = 0;
        public int PlaneY
        {
            get { return _PlaneY; }
            set { if (value != _PlaneY) { _PlaneY = value; OnPropertyChanged(); } }
        }

        private int _PlaneZ = 0;
        public int PlaneZ
        {
            get { return _PlaneZ; }
            set { if (value != _PlaneZ) { _PlaneZ = value; OnPropertyChanged(); } }
        }

        private int _ParticlePlaneX = 0;
        public int ParticlePlaneX
        {
            get { return _ParticlePlaneX; }
            set { if (value != _ParticlePlaneX) { _ParticlePlaneX = value; OnPropertyChanged(); } }
        }

        private int _ParticlePlaneY = 0;
        public int ParticlePlaneY
        {
            get { return _ParticlePlaneY; }
            set { if (value != _ParticlePlaneY) { _ParticlePlaneY = value; OnPropertyChanged(); } }
        }

        private int _ParticlePlaneZ = 0;
        public int ParticlePlaneZ
        {
            get { return _ParticlePlaneZ; }
            set { if (value != _ParticlePlaneZ) { _ParticlePlaneZ = value; OnPropertyChanged(); } }
        }

        private decimal _ViewX = 0;
        public decimal ViewX
        {
            get { return _ViewX; }
            set { if (value != _ViewX) { _ViewX = value; OnPropertyChanged(); } }
        }

        private decimal _ViewY = 0;
        public decimal ViewY
        {
            get { return _ViewY; }
            set { if (value != _ViewY) { _ViewY = value; OnPropertyChanged(); } }
        }

        private decimal _ViewZ = 0;
        public decimal ViewZ
        {
            get { return _ViewZ; }
            set { if (value != _ViewZ) { _ViewZ = value; OnPropertyChanged(); } }
        }

        private int _MouseX = 0;
        public int MouseX
        {
            get { return _MouseX; }
            set { if (value != _MouseX) { _MouseX = value; OnPropertyChanged(); } }
        }

        private int _MouseY = 0;
        public int MouseY
        {
            get { return _MouseY; }
            set { if (value != _MouseY) { _MouseY = value; OnPropertyChanged(); } }
        }

        private int _MouseZ = 0;
        public int MouseZ
        {
            get { return _MouseZ; }
            set { if (value != _MouseZ) { _MouseZ = value; OnPropertyChanged(); } }
        }

        private Particle _ActiveParticle = null;
        public Particle ActiveParticle
        {
            get { return _ActiveParticle; }
            set
            {
                if (value != _ActiveParticle)
                {
                    if (_ActiveParticle != null)
                        _ActiveParticle.IsSelected = false;

                    _ActiveParticle = value;

                    if (_ActiveParticle != null)
                        _ActiveParticle.IsSelected = true;

                    OnPropertyChanged();
                }
            }
        }

        private bool _CentralBlob = true;
        public bool CentralBlob
        {
            get { return _CentralBlob; }
            set { if (value != _CentralBlob) { _CentralBlob = value; OnPropertyChanged(); } }
        }

        private int _ImportVolumeWidth = 1;
        public int ImportVolumeWidth
        {
            get { return _ImportVolumeWidth; }
            set { if (value != _ImportVolumeWidth) { _ImportVolumeWidth = value; OnPropertyChanged(); } }
        }

        private int _ImportVolumeHeight = 1;
        public int ImportVolumeHeight
        {
            get { return _ImportVolumeHeight; }
            set { if (value != _ImportVolumeHeight) { _ImportVolumeHeight = value; OnPropertyChanged(); } }
        }

        private int _ImportVolumeDepth = 1;
        public int ImportVolumeDepth
        {
            get { return _ImportVolumeDepth; }
            set { if (value != _ImportVolumeDepth) { _ImportVolumeDepth = value; OnPropertyChanged(); } }
        }

        private bool _ImportInvertX = false;
        public bool ImportInvertX
        {
            get { return _ImportInvertX; }
            set { if (value != _ImportInvertX) { _ImportInvertX = value; OnPropertyChanged(); } }
        }

        private bool _ImportInvertY = false;
        public bool ImportInvertY
        {
            get { return _ImportInvertY; }
            set { if (value != _ImportInvertY) { _ImportInvertY = value; OnPropertyChanged(); } }
        }

        private bool _ImportInvertZ = false;
        public bool ImportInvertZ
        {
            get { return _ImportInvertZ; }
            set { if (value != _ImportInvertZ) { _ImportInvertZ = value; OnPropertyChanged(); } }
        }

        private int _ExportVolumeWidth = 1000;
        public int ExportVolumeWidth
        {
            get { return _ExportVolumeWidth; }
            set { if (value != _ExportVolumeWidth) { _ExportVolumeWidth = value; OnPropertyChanged(); } }
        }

        private int _ExportVolumeHeight = 1000;
        public int ExportVolumeHeight
        {
            get { return _ExportVolumeHeight; }
            set { if (value != _ExportVolumeHeight) { _ExportVolumeHeight = value; OnPropertyChanged(); } }
        }

        private int _ExportVolumeDepth = 300;
        public int ExportVolumeDepth
        {
            get { return _ExportVolumeDepth; }
            set { if (value != _ExportVolumeDepth) { _ExportVolumeDepth = value; OnPropertyChanged(); } }
        }

        private bool _ExportInvertX = false;
        public bool ExportInvertX
        {
            get { return _ExportInvertX; }
            set { if (value != _ExportInvertX) { _ExportInvertX = value; OnPropertyChanged(); } }
        }

        private bool _ExportInvertY = false;
        public bool ExportInvertY
        {
            get { return _ExportInvertY; }
            set { if (value != _ExportInvertY) { _ExportInvertY = value; OnPropertyChanged(); } }
        }

        private bool _ExportInvertZ = false;
        public bool ExportInvertZ
        {
            get { return _ExportInvertZ; }
            set { if (value != _ExportInvertZ) { _ExportInvertZ = value; OnPropertyChanged(); } }
        }

        private string _PathParticles = "";
        public string PathParticles
        {
            get { return _PathParticles; }
            set { if (value != _PathParticles) { _PathParticles = value; OnPropertyChanged(); } }
        }

        private decimal _ParticleScoreMin = 0;
        public decimal ParticleScoreMin
        {
            get { return _ParticleScoreMin; }
            set { if (value != _ParticleScoreMin) { _ParticleScoreMin = value; OnPropertyChanged(); } }
        }

        private decimal _ParticleScoreMax = 1;
        public decimal ParticleScoreMax
        {
            get { return _ParticleScoreMax; }
            set { if (value != _ParticleScoreMax) { _ParticleScoreMax = value; OnPropertyChanged(); } }
        }

        public void Save(string path)
        {
            XmlTextWriter Writer = new XmlTextWriter(File.Create(path), Encoding.Unicode);
            Writer.Formatting = Formatting.Indented;
            Writer.IndentChar = '\t';
            Writer.Indentation = 1;
            Writer.WriteStartDocument();
            Writer.WriteStartElement("Settings");

            XMLHelper.WriteParamNode(Writer, "InputLowpass", InputLowpass);
            XMLHelper.WriteParamNode(Writer, "InputAverageSlices", InputAverageSlices);
            XMLHelper.WriteParamNode(Writer, "DisplayIntensityMin", DisplayIntensityMin);
            XMLHelper.WriteParamNode(Writer, "DisplayIntensityMax", DisplayIntensityMax);
            XMLHelper.WriteParamNode(Writer, "BoxSize", BoxSize);
            XMLHelper.WriteParamNode(Writer, "CentralBlob", CentralBlob);

            XMLHelper.WriteParamNode(Writer, "ImportVolumeWidth", ImportVolumeWidth);
            XMLHelper.WriteParamNode(Writer, "ImportVolumeHeight", ImportVolumeHeight);
            XMLHelper.WriteParamNode(Writer, "ImportVolumeDepth", ImportVolumeDepth);

            XMLHelper.WriteParamNode(Writer, "ExportVolumeWidth", ExportVolumeWidth);
            XMLHelper.WriteParamNode(Writer, "ExportVolumeHeight", ExportVolumeHeight);
            XMLHelper.WriteParamNode(Writer, "ExportVolumeDepth", ExportVolumeDepth);

            Writer.WriteEndElement();
            Writer.WriteEndDocument();
            Writer.Flush();
            Writer.Close();
        }

        public void Load(string path)
        {
            using (Stream SettingsStream = File.OpenRead(path))
            {
                XPathDocument Doc = new XPathDocument(SettingsStream);
                XPathNavigator Reader = Doc.CreateNavigator();
                Reader.MoveToRoot();

                InputLowpass = XMLHelper.LoadParamNode(Reader, "InputLowpass", InputLowpass);
                InputAverageSlices = XMLHelper.LoadParamNode(Reader, "InputAverageSlices", InputAverageSlices);
                DisplayIntensityMin = XMLHelper.LoadParamNode(Reader, "DisplayIntensityMin", DisplayIntensityMin);
                DisplayIntensityMax = XMLHelper.LoadParamNode(Reader, "DisplayIntensityMax", DisplayIntensityMax);
                BoxSize = XMLHelper.LoadParamNode(Reader, "BoxSize", BoxSize);
                CentralBlob = XMLHelper.LoadParamNode(Reader, "CentralBlob", CentralBlob);

                ImportVolumeWidth = XMLHelper.LoadParamNode(Reader, "ImportVolumeWidth", ImportVolumeWidth);
                ImportVolumeHeight = XMLHelper.LoadParamNode(Reader, "ImportVolumeHeight", ImportVolumeHeight);
                ImportVolumeDepth = XMLHelper.LoadParamNode(Reader, "ImportVolumeDepth", ImportVolumeDepth);

                ExportVolumeWidth = XMLHelper.LoadParamNode(Reader, "ExportVolumeWidth", ExportVolumeWidth);
                ExportVolumeHeight = XMLHelper.LoadParamNode(Reader, "ExportVolumeHeight", ExportVolumeHeight);
                ExportVolumeDepth = XMLHelper.LoadParamNode(Reader, "ExportVolumeDepth", ExportVolumeDepth);
            }
        }
    }
}