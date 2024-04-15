using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Warp.Tools;

namespace Warp.Sociology
{
    public class Population : WarpBase
    {
        private bool DoAutosave = true;

        #region Paths and names

        private string _Path = "";
        public string Path
        {
            get { return _Path; }
            set
            {
                if (value != _Path)
                {
                    _Path = value;
                    OnPropertyChanged();
                }
            }
        }

        public string FolderPath => Helper.PathToFolder(Path);

        private string _Name = "";
        [WarpSerializable]
        public string Name
        {
            get { return _Name; }
            set { if (value != _Name) { _Name = value; OnPropertyChanged(); } }
        }

        public string ParticlesDir => Helper.PathCombine(FolderPath, "particles");
        public string ParticleTiltsDir => Helper.PathCombine(FolderPath, "particletilts");
        public string ParticleMoviesDir => Helper.PathCombine(FolderPath, "particlemovies");
        public string SpeciesDir => Helper.PathCombine(FolderPath, "species");

        #endregion

        private ObservableCollection<Species> _Species = new ObservableCollection<Species>();
        public ObservableCollection<Species> Species
        {
            get { return _Species; }
            set { if (value != _Species) { _Species = value; OnPropertyChanged(); } }
        }

        private ObservableCollection<DataSource> _Sources = new ObservableCollection<DataSource>();
        public ObservableCollection<DataSource> Sources
        {
            get { return _Sources; }
            set { if (value != _Sources) { _Sources = value; OnPropertyChanged(); } }
        }

        private ProcessingOptionsMPARefine _LastRefinementOptions = new ProcessingOptionsMPARefine();
        public ProcessingOptionsMPARefine LastRefinementOptions
        {
            get { return _LastRefinementOptions; }
            set { if (value != _LastRefinementOptions) { _LastRefinementOptions = value; OnPropertyChanged(); } }
        }

        public Population(string path)
        {
            path = Helper.PathCombine(Environment.CurrentDirectory, path);

            Species.CollectionChanged += Species_CollectionChanged;
            Sources.CollectionChanged += Sources_CollectionChanged;
            PropertyChanged += Population_PropertyChanged;

            if (File.Exists(path))  // Load
                Load(path);
            else
                Path = path;        // Create
        }

        private void Species_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        {
            if (DoAutosave)
                Save();
        }

        private void Sources_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        {
            if (DoAutosave)
                Save();
        }

        private void Population_PropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (DoAutosave)
                Save();
        }

        public void Load(string path)
        {
            DoAutosave = false;

            Path = path;

            using (Stream SettingsStream = File.OpenRead(path))
            {
                XPathDocument Doc = new XPathDocument(SettingsStream);
                XPathNavigator Reader = Doc.CreateNavigator();
                Reader.MoveToRoot();
                Reader.MoveToChild("Population", "");

                ReadFromXML(Reader);

                List<Species> AllSpecies = new List<Species>();
                foreach (XPathNavigator nav in Reader.Select("Species/Species"))
                {
                    Guid SpeciesGUID = Guid.Parse(nav.GetAttribute("GUID", ""));
                    string SpeciesPath = nav.GetAttribute("Path", "");

                    Species LoadedSpecies = Sociology.Species.FromFile(System.IO.Path.Combine(FolderPath, SpeciesPath));
                    if (LoadedSpecies.GUID != SpeciesGUID)
                        throw new Exception("Stored GUID does not match that of the species.");

                    AllSpecies.Add(LoadedSpecies);
                }
                foreach (var species in AllSpecies)
                    species.ResolveChildren(AllSpecies);

                Species.Clear();
                foreach (var toplevel in AllSpecies.Where(s => s.Parent == null))
                    Species.Add(toplevel);

                foreach (XPathNavigator nav in Reader.Select("Sources/Source"))
                {
                    string Path = nav.GetAttribute("Path", "");
                    Guid SourceGUID = Guid.Parse(nav.GetAttribute("GUID", ""));

                    DataSource LoadedSource = DataSource.FromFile(System.IO.Path.Combine(FolderPath, Path));
                    if (SourceGUID != LoadedSource.GUID)
                        throw new Exception("Stored GUID does not match that of the data source.");

                    Sources.Add(LoadedSource);
                }

                XPathNavigator NavLastRefinementOptions = Reader.SelectSingleNode("//LastRefinementOptions");
                if (NavLastRefinementOptions != null)
                {
                    ProcessingOptionsMPARefine Temp = new ProcessingOptionsMPARefine();
                    Temp.ReadFromXML(NavLastRefinementOptions);
                    LastRefinementOptions = Temp;
                }
            }

            DoAutosave = true;
        }

        public void Save()
        {
            XmlTextWriter Writer = new XmlTextWriter(File.Create(Path), Encoding.Unicode);
            Writer.Formatting = Formatting.Indented;
            Writer.IndentChar = '\t';
            Writer.Indentation = 1;
            Writer.WriteStartDocument();
            Writer.WriteStartElement("Population");

            WriteToXML(Writer);

            Species[] AllSpecies = Helper.Combine(Species.Select(s => s.AllDescendants));
            Writer.WriteStartElement("Species");
            foreach (var species in AllSpecies)
            {
                Writer.WriteStartElement("Species");
                XMLHelper.WriteAttribute(Writer, "GUID", species.GUID.ToString());
                XMLHelper.WriteAttribute(Writer, "Path", Helper.MakePathRelativeTo(species.Path, Path));
                Writer.WriteEndElement();
            }
            Writer.WriteEndElement();

            Writer.WriteStartElement("Sources");
            foreach (var source in Sources)
            {
                Writer.WriteStartElement("Source");
                XMLHelper.WriteAttribute(Writer, "GUID", source.GUID.ToString());
                XMLHelper.WriteAttribute(Writer, "Path", Helper.MakePathRelativeTo(source.Path, Path));
                Writer.WriteEndElement();
            }
            Writer.WriteEndElement();
                       
            if (LastRefinementOptions != null)
            {
                Writer.WriteStartElement("LastRefinementOptions");
                LastRefinementOptions.WriteToXML(Writer);
                Writer.WriteEndElement();
            }

            Writer.WriteEndElement();
            Writer.WriteEndDocument();
            Writer.Flush();
            Writer.Close();
        }

        public void SaveRefinementProgress(string folder)
        {
            foreach (var species in Species)
            {
                string SpeciesID = species.GUID.ToString().Substring(0, 8);

                for (int i = 0; i < species.HalfMap1Reconstruction.Length; i++)
                {
                    species.HalfMap1Reconstruction[i]?.WriteMRC(System.IO.Path.Combine(folder, $"{SpeciesID}_half1_{i}.mrc"));
                    species.HalfMap2Reconstruction[i]?.WriteMRC(System.IO.Path.Combine(folder, $"{SpeciesID}_half2_{i}.mrc"));
                }
                species.ParticlesToStar().Save(System.IO.Path.Combine(folder, $"{SpeciesID}_particles.star"));
            }
        }

        public void GatherRefinementProgress(string[] folders)
        {
            foreach (var species in Species)
            {
                string SpeciesID = species.GUID.ToString().Substring(0, 8);

                Projector Rec1 = species.HalfMap1Reconstruction[0];
                Projector Rec2 = species.HalfMap2Reconstruction[0];

                Particle[] OriginalParticles = species.Particles.ToList().ToArray();
                Particle[] FinalParticles = species.Particles;

                foreach (var folder in folders)
                {
                    foreach (var path in Directory.EnumerateFiles(folder, $"{SpeciesID}_half1_*.mrc"))
                    {
                        Projector Saved = Projector.FromFile(path);

                        Rec1.Data.Add(Saved.Data);
                        Rec1.Weights.Add(Saved.Weights);

                        Saved.Dispose();
                    }

                    foreach (var path in Directory.EnumerateFiles(folder, $"{SpeciesID}_half2_*.mrc"))
                    {
                        Projector Saved = Projector.FromFile(path);

                        Rec2.Data.Add(Saved.Data);
                        Rec2.Weights.Add(Saved.Weights);

                        Saved.Dispose();
                    }

                    Particle[] UpdatedParticles = species.ParticlesFromStar(new Star(System.IO.Path.Combine(folder, $"{SpeciesID}_particles.star")));

                    for (int p = 0; p < OriginalParticles.Length; p++)
                        if (OriginalParticles[p].Coordinates.Where((v, i) => v != UpdatedParticles[p].Coordinates[i]).Count() > 0 ||
                            OriginalParticles[p].Angles.Where((v, i) => v != UpdatedParticles[p].Angles[i]).Count() > 0)
                            FinalParticles[p] = UpdatedParticles[p];
                }

                Rec1.FreeDevice();
                Rec2.FreeDevice();
            }
        }
    }
}
