using System;
using System.IO;
using System.Xml;
using Warp;
using Xunit;

namespace Tests;

public class TiltSeriesMetadataExtensionsTests
{
    [Fact]
    public void NamespacedCalibrationSurvivesNativeMetadataEdit()
    {
        string folder = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "warp-extensions-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            string tomo = System.IO.Path.Combine(folder, "TS.tomostar");
            File.WriteAllText(tomo, "data_\nloop_\n_wrpDose #1\n_wrpAngleTilt #2\n_wrpMovieName #3\n10 -30 a.mrc\n0 0 b.mrc\n20 30 c.mrc\n");
            var initial = new TiltSeries(tomo);
            initial.SaveMeta();
            var doc = new XmlDocument();
            doc.Load(initial.XMLPath);
            var beam = doc.CreateElement("beam", "Refinement", "urn:beam:refinement:1");
            beam.SetAttribute("schemaVersion", "1");
            beam.SetAttribute("normalization", "none");
            var calibration = doc.CreateElement("beam", "Calibration", "urn:beam:refinement:1");
            calibration.InnerText = "{\"half_gains\":[[1,0.7],[1,0.8]]}";
            beam.AppendChild(calibration);
            doc.DocumentElement.AppendChild(beam);
            var opaque = doc.CreateElement("other", "Future", "urn:another:extension");
            opaque.SetAttribute("keep", "yes");
            doc.DocumentElement.AppendChild(opaque);
            doc.Save(initial.XMLPath);

            var loaded = new TiltSeries(tomo);
            Assert.Equal(2, loaded.MetadataExtensions.Length);
            loaded.Angles[0] += 1;
            loaded.SaveMeta();
            doc.Load(loaded.XMLPath);
            var namespaces = new XmlNamespaceManager(doc.NameTable);
            namespaces.AddNamespace("beam", "urn:beam:refinement:1");
            namespaces.AddNamespace("other", "urn:another:extension");
            Assert.Equal(calibration.InnerText,
                doc.SelectSingleNode("/TiltSeries/beam:Refinement/beam:Calibration", namespaces).InnerText);
            Assert.Equal("yes", doc.SelectSingleNode("/TiltSeries/other:Future", namespaces).Attributes["keep"].Value);
            var reloaded = new TiltSeries(tomo);
            Assert.Equal(2, reloaded.MetadataExtensions.Length);
            Assert.Equal(-29f, reloaded.Angles[0]);
        }
        finally { Directory.Delete(folder, true); }
    }
}
