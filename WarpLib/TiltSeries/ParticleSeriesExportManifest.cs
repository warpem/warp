using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Warp;

/// <summary>Immutable extraction geometry for downstream raw-series refinement.</summary>
public static class ParticleSeriesExportManifest
{
    static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public static string SidecarPath(string particleTablePath) => particleTablePath + ".beam.json";

    public static int[] GetExportTiltIndices(TiltSeries series, ProcessingOptionsTomoSubReconstruction options)
    {
        int[] sorted = series.IndicesSortedDose;
        int count = options.DoLimitDose ? Math.Min(options.NTilts, sorted.Length) : sorted.Length;
        int[] indices = new int[count];
        Array.Copy(sorted, indices, count);
        Array.Sort(indices);
        return indices;
    }

    /// <summary>Called by the worker while exact image-rescaling factors are still in memory.</summary>
    public static void Capture(TiltSeries series, ProcessingOptionsTomoSubReconstruction options,
                               string particleTablePath)
    {
        if (!options.DontPremultiply || string.IsNullOrEmpty(particleTablePath))
            return;
        byte[] xml = File.ReadAllBytes(series.XMLPath);
        string hash = Convert.ToHexString(SHA256.HashData(xml)).ToLowerInvariant();
        string snapshot = System.IO.Path.GetFullPath(System.IO.Path.Combine(
            System.IO.Path.GetDirectoryName(particleTablePath), series.RootName + "-" + hash + ".xml"));
        if (!File.Exists(snapshot))
            File.WriteAllBytes(snapshot, xml);
        var record = new
        {
            series_name = series.RootName + ".tomostar",
            xml = snapshot,
            xml_sha256 = hash,
            tilt_indices = GetExportTiltIndices(series, options),
            image_dimensions_angstrom = new[] { series.ImageDimensionsPhysical.X, series.ImageDimensionsPhysical.Y },
            volume_dimensions_angstrom = new[] { series.VolumeDimensionsPhysical.X, series.VolumeDimensionsPhysical.Y, series.VolumeDimensionsPhysical.Z },
            size_rounding_factors = new[] { series.SizeRoundingFactors.X, series.SizeRoundingFactors.Y, series.SizeRoundingFactors.Z }
        };
        WriteAtomically(SidecarPath(particleTablePath), JsonSerializer.Serialize(record, JsonOptions));
    }

    /// <summary>Publish the manifest after the final particle STAR, including its integrity hash.</summary>
    public static void WriteCombined(string particleStarPath, IEnumerable<string> snapshots,
                                     ProcessingOptionsTomoSubReconstruction options)
    {
        string root = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(particleStarPath));
        string snapshotDirectory = System.IO.Path.Combine(root,
            System.IO.Path.GetFileNameWithoutExtension(particleStarPath) + ".warp_metadata");
        Directory.CreateDirectory(snapshotDirectory);
        var series = new JsonObject();
        foreach (string path in snapshots)
        {
            JsonObject record = JsonNode.Parse(File.ReadAllText(path)).AsObject();
            string name = record["series_name"].GetValue<string>();
            record.Remove("series_name");
            string xmlPath = record["xml"].GetValue<string>();
            byte[] xml = File.ReadAllBytes(xmlPath);
            string hash = Convert.ToHexString(SHA256.HashData(xml)).ToLowerInvariant();
            if (hash != record["xml_sha256"].GetValue<string>())
                throw new InvalidDataException("Particle export XML changed before manifest publication: " + xmlPath);
            string target = System.IO.Path.Combine(snapshotDirectory, System.IO.Path.GetFileName(xmlPath));
            if (!File.Exists(target))
                File.WriteAllBytes(target, xml);
            record["xml"] = System.IO.Path.GetRelativePath(root, target);
            series.Add(name, record);
        }
        var manifest = new JsonObject
        {
            ["schema_version"] = 1,
            ["normalization"] = options.NormalizeInput ? "per_tilt_input_std" : "none",
            ["contrast_inverted"] = options.Invert,
            ["ctf_premultiplied"] = false,
            ["stack_layout"] = "compact_visible",
            ["highpass_fourier_pixels"] = 1,
            ["box_size"] = options.BoxSize,
            ["pixel_size_angstrom"] = (double)options.BinnedPixelSizeMean,
            ["source_pixel_size_angstrom"] = (double)options.PixelSizeMean,
            ["star_sha256"] = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(particleStarPath))).ToLowerInvariant(),
            ["series"] = series
        };
        WriteAtomically(System.IO.Path.ChangeExtension(particleStarPath, ".beam.json"), manifest.ToJsonString(JsonOptions));
    }

    static void WriteAtomically(string path, string json)
    {
        string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            File.WriteAllText(temporary, json + Environment.NewLine);
            File.Move(temporary, path, true);
        }
        finally
        {
            if (File.Exists(temporary))
                File.Delete(temporary);
        }
    }
}
