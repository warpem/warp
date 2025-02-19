using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Accord.Math.Optimization;
using Warp;
using Warp.Tools;
using IOPath = System.IO.Path;

namespace Warp;

public partial class Movie
{
    public void SubtractMembranes(ProcessingOptionsSubtractMembranes options)
    {
        if (!File.Exists(AveragePath) ||
            !File.Exists(MembraneControlPointsPath) ||
            !File.Exists(MembraneProfilesPath))
        {
            throw new FileNotFoundException("One or more input files do not exist.");
        }

        Image ImageRaw = Image.FromFile(AveragePath);
        Dictionary<string, Star> ControlPoints = Star.FromMultitable(MembraneControlPointsPath, new string[] { "path" });
        Dictionary<string, Star> MembraneProfiles = Star.FromMultitable(MembraneProfilesPath, new string[] { "profile" });

        Image AllMembranes = new Image(ImageRaw.Dims);
        float[] AllMembranesData = AllMembranes.GetHost(Intent.ReadWrite)[0];

        int MembraneCount = ControlPoints.Count;
        int MembraneIndex = 0;

        foreach (var key in ControlPoints.Keys)
        {
            Console.WriteLine($"Subtracting membrane {MembraneIndex + 1} of {MembraneCount}");
            Star ControlPointTable = ControlPoints[key];
            Star ProfileTable = MembraneProfiles[key];

            // Convert control points to float2 list
            List<float2> ControlPointsList = ControlPointTable.GetFloat2("wrpControlPointXAngst", "wrpControlPointYAngst").ToList();

            bool IsClosed = ControlPointsList.First() == ControlPointsList.Last();
            SplinePath2D Spline = new SplinePath2D(ControlPointsList.ToArray(), IsClosed);

            float2[] Points = Spline.GetInterpolated(Helper.ArrayOfFunction(i => (float)i / (ControlPointsList.Count - 1), ControlPointsList.Count));

            int RecDim = ProfileTable.RowCount;
            float[] RecData = ProfileTable.GetFloat("membrane_profile");

            foreach (var p in Points)
            {
                int x = (int)MathF.Round(p.X);
                int y = (int)MathF.Round(p.Y);
                if (x < 0 || x >= ImageRaw.Dims.X || y < 0 || y >= ImageRaw.Dims.Y) continue;

                float Coord = (x % RecDim) / (float)RecDim * RecData.Length;
                int Coord0 = (int)Coord;
                int Coord1 = Math.Min(RecData.Length - 1, Coord0 + 1);
                float Weight1 = Coord - Coord0;
                float Weight0 = 1 - Weight1;

                float Val = RecData[Coord0] * Weight0 + RecData[Coord1] * Weight1;
                AllMembranesData[y * ImageRaw.Dims.X + x] += Val;
            }

            MembraneIndex++;
        }

        ImageRaw.Subtract(AllMembranes);
        AllMembranes.Dispose();

        List<SplinePath2D> membraneSplines = ControlPoints.Values.Select(cp => 
            new SplinePath2D(cp.GetFloat2("wrpControlPointXAngst", "wrpControlPointYAngst").ToArray(), 
                cp.GetFloat2("wrpControlPointXAngst", "wrpControlPointYAngst").First() == 
                cp.GetFloat2("wrpControlPointXAngst", "wrpControlPointYAngst").Last())).ToList();

        SubtractMembranesHelper.AddBlendedNoise(ImageRaw, membraneSplines, options.NoiseStdDevScale);

        string outputPath = $"d_subtracted_{IOPath.GetFileNameWithoutExtension(AveragePath)}.mrc";
        ImageRaw.WriteMRC(outputPath, true);
        ImageRaw.Dispose();

        Console.WriteLine("Membrane subtraction complete.");
    }
}

[Serializable]
public class ProcessingOptionsSubtractMembranes : ProcessingOptionsBase
{
    [WarpSerializable] public float NoiseStdDevScale { get; set; }
}

public static class SubtractMembranesHelper
{
    public static void AddBlendedNoise(Image image, List<SplinePath2D> membraneSplines, float noiseScale)
    {
        // Create an image to store traced membrane regions
        Image MembraneMask = new Image(image.Dims);
        MembraneMask.Fill(0f);
        float[] MembraneMaskData = MembraneMask.GetHost(Intent.ReadWrite)[0];

        // Rasterize membrane splines into the mask
        foreach (var spline in membraneSplines)
        {
            float2[] Points = spline.GetInterpolated(Helper.ArrayOfFunction(i => (float)i / (spline.Points.Count - 1), spline.Points.Count));
            foreach (var p in Points)
            {
                int x = (int)MathF.Round(p.X);
                int y = (int)MathF.Round(p.Y);
                if (x >= 0 && x < image.Dims.X && y >= 0 && y < image.Dims.Y)
                {
                    MembraneMaskData[y * image.Dims.X + x] = 1f;
                }
            }
        }

        // Compute distance map (soft mask)
        Image DistanceMap = MembraneMask.AsDistanceMapExact(60, false);  // Soft blending over ~60 pixels
        DistanceMap.TransformValues(v => 1f - (MathF.Cos(v / 60f * MathF.PI) * 0.5f + 0.5f)); // Cosine mask

        // Generate noise with same statistics as the image
        float2 MeanStd = MathHelper.MeanAndStd(image.GetHost(Intent.Read)[0]);
        RandomNormal RandN = new RandomNormal(123);
        Image Noise = new Image(image.Dims);
        Noise.TransformValues(v => RandN.NextSingle(MeanStd.X, MeanStd.Y * noiseScale));

        // Blend noise using the soft mask
        Noise.Subtract(image);
        Noise.Multiply(DistanceMap);
        image.Add(Noise);

        // Cleanup
        MembraneMask.Dispose();
        DistanceMap.Dispose();
        Noise.Dispose();
    }
}