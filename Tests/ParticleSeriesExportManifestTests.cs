using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class ParticleSeriesExportManifestTests
{
    [Fact]
    public void RawManifestPreservesExactWorkerGeometryAndSnapshot()
    {
        string folder = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "warp-manifest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            string tomo = System.IO.Path.Combine(folder, "TS.tomostar");
            File.WriteAllText(tomo, "data_\nloop_\n_wrpDose #1\n_wrpAngleTilt #2\n_wrpMovieName #3\n10 -30 a.mrc\n0 0 b.mrc\n20 30 c.mrc\n");
            var ts = new TiltSeries(tomo)
            {
                ImageDimensionsPhysical = new float2(123, 234),
                VolumeDimensionsPhysical = new float3(345, 456, 567),
                SizeRoundingFactors = new float3(1.001f, 0.999f, 1)
            };
            ts.SaveMeta();
            byte[] original = File.ReadAllBytes(ts.XMLPath);
            var options = new ProcessingOptionsTomoSubReconstruction
            {
                DontPremultiply = true, NormalizeInput = false, Invert = true,
                DoLimitDose = true, NTilts = 2, BoxSize = 8, PixelSize = 1, BinTimes = 1
            };
            Assert.Equal(new[] { 0, 1 }, ParticleSeriesExportManifest.GetExportTiltIndices(ts, options));
            string tempStar = System.IO.Path.Combine(folder, "TS_temp.star");
            ParticleSeriesExportManifest.Capture(ts, options, tempStar);
            string outputFolder = System.IO.Path.Combine(folder, "export");
            Directory.CreateDirectory(outputFolder);
            string output = System.IO.Path.Combine(outputFolder, "particles.star");
            File.WriteAllText(output, "data_particles\n");
            // Later XML edits cannot alter the immutable worker snapshot.
            File.AppendAllText(ts.XMLPath, "\n");
            ParticleSeriesExportManifest.WriteCombined(output,
                new[] { ParticleSeriesExportManifest.SidecarPath(tempStar) }, options);
            var manifest = JsonNode.Parse(File.ReadAllText(System.IO.Path.ChangeExtension(output, ".beam.json")));
            Assert.Equal("none", manifest["normalization"].GetValue<string>());
            Assert.True(manifest["contrast_inverted"].GetValue<bool>());
            Assert.Equal(2.0, manifest["pixel_size_angstrom"].GetValue<double>());
            Assert.Equal(Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(output))).ToLowerInvariant(),
                manifest["star_sha256"].GetValue<string>());
            var record = manifest["series"]["TS.tomostar"];
            Assert.Equal(1.001f, record["size_rounding_factors"][0].GetValue<float>());
            Assert.Equal(new[] { 0, 1 }, record["tilt_indices"].AsArray().Select(v => v.GetValue<int>()).ToArray());
            string snapshot = System.IO.Path.Combine(outputFolder, record["xml"].GetValue<string>());
            Assert.Equal(original, File.ReadAllBytes(snapshot));
        }
        finally { Directory.Delete(folder, true); }
    }
}
