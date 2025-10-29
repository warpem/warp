using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools
{
    public class Mdoc
    {
        public string Path;
        public List<MdocEntry> Entries = new List<MdocEntry>();

        public Mdoc(string path, IEnumerable<MdocEntry> entries)
        {
            Entries.AddRange(entries);
        }

        public static Mdoc FromFile(string[] paths, float defaultDose = 0)
        {
            float AxisAngle = 0;
            List<MdocEntry> Entries = new List<MdocEntry>();
            bool FoundTime = false;

            foreach (var mdocPath in paths)
            {
                using (TextReader Reader = new StreamReader(File.OpenRead(mdocPath)))
                {
                    string Line;
                    while ((Line = Reader.ReadLine()) != null)
                    {
                        if (Line.Contains("Tilt axis angle = "))
                        {
                            string Suffix = Line.Substring(Line.IndexOf("Tilt axis angle = ") + "Tilt axis angle = ".Length);
                            Suffix = Suffix.Substring(0, Suffix.IndexOf(","));

                            AxisAngle = float.Parse(Suffix, CultureInfo.InvariantCulture);
                            continue;
                        }

                        if (Line.Length < 7 || Line.Substring(0, 7) != "[ZValue")
                            continue;

                        MdocEntry NewEntry = new MdocEntry();

                        {
                            string[] Parts = Line.Split(new[] { " = " }, StringSplitOptions.RemoveEmptyEntries);
                            if (Parts[0] == "[ZValue")
                                NewEntry.ZValue = int.Parse(Parts[1].Replace("]", ""));
                        }

                        while ((Line = Reader.ReadLine()) != null)
                        {
                            string[] Parts = Line.Split(new[] { " = " }, StringSplitOptions.RemoveEmptyEntries);
                            if (Parts.Length < 2)
                                break;

                            if (Parts[0] == "TiltAngle")
                                NewEntry.TiltAngle = (float)Math.Round(float.Parse(Parts[1], CultureInfo.InvariantCulture), 1);
                            else if (Parts[0] == "ExposureDose")
                                NewEntry.Dose = float.Parse(Parts[1], CultureInfo.InvariantCulture);
                            else if (Parts[0] == "SubFramePath")
                                NewEntry.Name = Parts[1].Substring(Parts[1].LastIndexOf("\\") + 1);
                            else if (Parts[0] == "DateTime")
                            {
                                NewEntry.Time = DateTime.ParseExact(Parts[1], "dd-MMM-yy  HH:mm:ss", CultureInfo.InvariantCulture);
                                FoundTime = true;
                            }
                        }

                        //if (mdocNames.Value.Count == 1)
                        //    Entries.RemoveAll(v => v.ZValue == NewEntry.ZValue);

                        Entries.Add(NewEntry);
                    }
                }
            }

            List<MdocEntry> SortedTime = new List<MdocEntry>(Entries);
            SortedTime.Sort((a, b) => a.Time.CompareTo(b.Time));

            // Do running dose
            float Accumulated = 0;
            foreach (var entry in SortedTime)
            {
                Accumulated += entry.Dose;
                entry.Dose = Accumulated;
            }

            // In case mdoc doesn't tell anything about the dose, use default value
            if (defaultDose > 0)
                for (int i = 0; i < SortedTime.Count; i++)
                {
                    SortedTime[i].Dose = (i + 0.5f) * defaultDose;
                    Accumulated += defaultDose;
                }

            // Sort entires by angle and time (accumulated dose)
            List<MdocEntry> SortedAngle = new List<MdocEntry>(Entries);
            SortedAngle.Sort((a, b) => a.TiltAngle.CompareTo(b.TiltAngle));
            // Sometimes, there will be 2 0-tilts at the beginning of plus and minus series. 
            // Sort them according to dose, considering in which order plus and minus were acquired
            float DoseMinus = SortedAngle.Take(SortedAngle.Count / 2).Select(v => v.Dose).Sum();
            float DosePlus = SortedAngle.Skip(SortedAngle.Count / 2).Take(SortedAngle.Count / 2).Select(v => v.Dose).Sum();
            int OrderCorrection = DoseMinus < DosePlus ? 1 : -1;
            SortedAngle.Sort((a, b) => a.TiltAngle.CompareTo(b.TiltAngle) != 0 ? a.TiltAngle.CompareTo(b.TiltAngle) : (a.Dose.CompareTo(b.Dose) * OrderCorrection));

            return new Mdoc(paths[0], SortedAngle);
        }

        public static void UpdateWithXf(string path, Mdoc entriesSortedAngle)
        {
            using (TextReader Reader = new StreamReader(File.OpenRead(path)))
            {
                string Line;
                for (int i = 0; i < entriesSortedAngle.Entries.Count; i++)
                {
                    Line = Reader.ReadLine();
                    string[] Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    float2 VecX = new float2(float.Parse(Parts[0], CultureInfo.InvariantCulture),
                                             float.Parse(Parts[2], CultureInfo.InvariantCulture));
                    float2 VecY = new float2(float.Parse(Parts[1], CultureInfo.InvariantCulture),
                                             float.Parse(Parts[3], CultureInfo.InvariantCulture));

                    Matrix3 Rotation = new Matrix3(VecX.X, VecX.Y, 0, VecY.X, VecY.Y, 0, 0, 0, 1);
                    float3 Euler = Matrix3.EulerFromMatrix(Rotation);

                    entriesSortedAngle.Entries[i].AxisAngle = Euler.Z * Helper.ToDeg;

                    //SortedAngle[i].Shift += VecX * float.Parse(Parts[4], CultureInfo.InvariantCulture) + VecY * float.Parse(Parts[5], CultureInfo.InvariantCulture);
                    float3 Shift = new float3(-float.Parse(Parts[4], CultureInfo.InvariantCulture), -float.Parse(Parts[5], CultureInfo.InvariantCulture), 0);
                    Shift = Rotation.Transposed() * Shift;

                    entriesSortedAngle.Entries[i].Shift += new float2(Shift);
                }
            }
        }
    }


    public class MdocEntry
    {
        public int ZValue;
        public string Name;
        public float TiltAngle;
        public float AxisAngle;
        public float Dose;
        public DateTime Time;
        public Image Micrograph;
        public float2 Shift;
        public float MaxTranslation;
        public float AverageIntensity;
        public float MaskedFraction;
    }

    public class ParsedEntry : WarpBase
    {
        private bool _DoImport = true;
        public bool DoImport
        {
            get { return _DoImport; }
            set { if (value != _DoImport) { _DoImport = value; OnPropertyChanged(); } }
        }

        private string _Name = "";
        public string Name
        {
            get { return _Name; }
            set { if (value != _Name) { _Name = value; OnPropertyChanged(); } }
        }

        private int _NTilts = 0;
        public int NTilts
        {
            get { return _NTilts; }
            set { if (value != _NTilts) { _NTilts = value; OnPropertyChanged(); } }
        }

        private float[] _TiltAngles = null;
        public float[] TiltAngles
        {
            get { return _TiltAngles; }
            set { if (value != _TiltAngles) { _TiltAngles = value; OnPropertyChanged(); } }
        }

        private float _Rotation = 0;
        public float Rotation
        {
            get { return _Rotation; }
            set { if (value != _Rotation) { _Rotation = value; OnPropertyChanged(); } }
        }

        private int _Dose = 0;
        public int Dose
        {
            get { return _Dose; }
            set { if (value != _Dose) { _Dose = value; OnPropertyChanged(); } }
        }

        private bool _Aligned = false;
        public bool Aligned
        {
            get { return _Aligned; }
            set { if (value != _Aligned) { _Aligned = value; OnPropertyChanged(); } }
        }

        private int _Pretilt = 0;
        public int Pretilt
        {
            get { return _Pretilt; }
            set { if (value != _Pretilt) { _Pretilt = value; OnPropertyChanged(); } }
        }

        public Star Table;
    }
}
