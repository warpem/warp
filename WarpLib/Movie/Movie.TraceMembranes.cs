using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Accord;
using Accord.Math.Optimization;
using Warp;
using Warp.Tools;
using IOPath = System.IO.Path;

namespace Warp
{
    public partial class Movie
    {
        public void TraceMembranes(ProcessingOptionsTraceMembranes options)
        {
            List<Image> toDispose = new();
            try
            {
                #region read inputs

                // Validate file inputs
                if (!File.Exists(AveragePath) || !File.Exists(MembraneSegmentationPath))
                    throw new FileNotFoundException("Required input files not found");

                // Load and prepare images
                Image MaskRaw = Image.FromFile(MembraneSegmentationPath);
                toDispose.Add(MaskRaw);
                Image ImageRaw = Image.FromFile(AveragePath);
                toDispose.Add(ImageRaw);

                #endregion

                #region lowpass

                ImageRaw.SubtractMeanGrid(new int2(1));
                Image ImageLowpass = ImageRaw.GetCopyGPU();
                toDispose.Add(ImageLowpass);

                float pixelSize = ImageRaw.PixelSize;
                ImageRaw.Bandpass(pixelSize * 2 / (float)options.HighResolutionLimit,
                    1,
                    false,
                    pixelSize * 2 / (float)options.RolloffWidth);

                ImageLowpass.Bandpass(pixelSize * 2 / (float)options.HighResolutionLimit,
                    pixelSize * 2 / (float)options.LowResolutionLimit,
                    false,
                    pixelSize * 2 / (float)options.RolloffWidth);

                #endregion lowpass

                #region find connected components

                var Components = MaskRaw.GetConnectedComponents()
                    .Where(c => c.ComponentIndices.Length >= options.MinimumComponentPixels)
                    .ToArray();

                if (Components.Length == 0)
                    throw new Exception("No valid membrane components found");

                #endregion

                #region calculate distance map

                // Create Images for inverse mask and distance map
                Image MaskInv = new Image(IntPtr.Zero, MaskRaw.Dims);
                Image DistanceMap = new Image(IntPtr.Zero, MaskRaw.Dims);
                toDispose.Add(MaskInv);
                toDispose.Add(DistanceMap);

                MaskInv.Fill(1f);
                MaskInv.Subtract(MaskRaw);

                // Calculate exact distance map
                GPU.DistanceMapExact(MaskInv.GetDevice(Intent.Read),
                    DistanceMap.GetDevice(Intent.Write),
                    MaskRaw.Dims,
                    20);

                float[] MaskInvData = MaskInv.GetHost(Intent.ReadWrite)[0];
                float[] DistanceData = DistanceMap.GetHost(Intent.Read)[0];

                MaskInv.Fill(0f);

                #endregion

                #region trace ridges

                // Trace ridges as points at which the local gradient peaks in at least one direction
                for (int i = MaskRaw.Dims.X + 1; i < DistanceData.Length - MaskRaw.Dims.X - 1; i++)
                {
                    float Center = DistanceData[i];

                    // Only consider points with significant thickness (>2 pixels)
                    if (Center < 2)
                        continue;

                    // Check for 1-pixel thick ridge - peak in any of 4 directions
                    if (DistanceData[i - 1] < Center && DistanceData[i + 1] < Center)
                    {
                        MaskInvData[i] = 1;
                        continue;
                    }

                    if (DistanceData[i - MaskRaw.Dims.X] < Center && DistanceData[i + MaskRaw.Dims.X] < Center)
                    {
                        MaskInvData[i] = 1;
                        continue;
                    }

                    if (DistanceData[i - 1 - MaskRaw.Dims.X] < Center && DistanceData[i + 1 + MaskRaw.Dims.X] < Center)
                    {
                        MaskInvData[i] = 1;
                        continue;
                    }

                    if (DistanceData[i - 1 + MaskRaw.Dims.X] < Center && DistanceData[i + 1 - MaskRaw.Dims.X] < Center)
                    {
                        MaskInvData[i] = 1;
                        continue;
                    }

                    // Handle 2-pixel thick ridges by checking for flat gradients
                    if (DistanceData[i - 1] == DistanceData[i + 1])
                    {
                        if (DistanceData[i - MaskRaw.Dims.X] < Center && DistanceData[i + MaskRaw.Dims.X] <= Center)
                        {
                            MaskInvData[i] = 1;
                            continue;
                        }
                    }

                    if (DistanceData[i - MaskRaw.Dims.X] == DistanceData[i + MaskRaw.Dims.X])
                    {
                        if (DistanceData[i - 1] < Center && DistanceData[i + 1] <= Center)
                        {
                            MaskInvData[i] = 1;
                            continue;
                        }
                    }

                    if (DistanceData[i - 1 - MaskRaw.Dims.X] == DistanceData[i + 1 + MaskRaw.Dims.X])
                    {
                        if (DistanceData[i - 1 + MaskRaw.Dims.X] < Center && DistanceData[i + 1 - MaskRaw.Dims.X] <= Center)
                        {
                            MaskInvData[i] = 1;
                            continue;
                        }
                    }

                    if (DistanceData[i - 1 + MaskRaw.Dims.X] == DistanceData[i + 1 - MaskRaw.Dims.X])
                    {
                        if (DistanceData[i - 1 - MaskRaw.Dims.X] < Center && DistanceData[i + 1 + MaskRaw.Dims.X] <= Center)
                        {
                            MaskInvData[i] = 1;
                            continue;
                        }
                    }
                }

                // Change 45deg stairs from pattern "0 1, 1 1" to "0 1, 1 0"
                for (int i = MaskRaw.Dims.X + 1; i < DistanceData.Length - MaskRaw.Dims.X - 1; i++)
                {
                    if (MaskInvData[i] != 1)
                        continue;

                    if (MaskInvData[i + 1] == 1 && MaskInvData[i + MaskRaw.Dims.X] == 1)
                    {
                        MaskInvData[i] = 0;
                    }
                    else if (MaskInvData[i + 1] == 1 && MaskInvData[i - MaskRaw.Dims.X] == 1)
                    {
                        MaskInvData[i] = 0;
                    }
                    else if (MaskInvData[i - 1] == 1 && MaskInvData[i + MaskRaw.Dims.X] == 1)
                    {
                        MaskInvData[i] = 0;
                    }
                    else if (MaskInvData[i - 1] == 1 && MaskInvData[i - MaskRaw.Dims.X] == 1)
                    {
                        MaskInvData[i] = 0;
                    }
                }

                // Fill 1-pixel gaps in horizontal and vertical lines
                for (int i = MaskRaw.Dims.X + 1; i < DistanceData.Length - MaskRaw.Dims.X - 1; i++)
                {
                    if (MaskInvData[i] != 0)
                        continue;

                    int[] Neighbors = new[]
                    {
                        i - 1,
                        i + 1,
                        i - MaskRaw.Dims.X,
                        i + MaskRaw.Dims.X,
                        i - MaskRaw.Dims.X - 1,
                        i - MaskRaw.Dims.X + 1,
                        i + MaskRaw.Dims.X - 1,
                        i + MaskRaw.Dims.X + 1
                    };

                    if (Neighbors.Count(n => MaskInvData[n] == 1) == 2)
                    {
                        if (MaskInvData[i - 1] == 1 && MaskInvData[i + 1] == 1)
                        {
                            MaskInvData[i] = 1;
                        }
                        else if (MaskInvData[i - MaskRaw.Dims.X] == 1 && MaskInvData[i + MaskRaw.Dims.X] == 1)
                        {
                            MaskInvData[i] = 1;
                        }
                    }
                }

                // Remove ridge pixels in gaps that are too wide to be a single bilayer
                for (int i = 0; i < DistanceData.Length; i++)
                    if (DistanceData[i] > 4)
                        MaskInvData[i] = 0;

                DistanceMap.Dispose();

                // Find connected components with at least MinimumComponentPixels pixels
                Components = MaskInv.GetConnectedComponents()
                    .Where(c => c.ComponentIndices.Length >= options.MinimumComponentPixels)
                    .ToArray();

                #endregion

                Dictionary<string, Star> PathTables = new();

                // Process each component
                for (int ic = 0; ic < Components.Length; ic++)
                {
                    Console.WriteLine($"Refining membrane {ic + 1} of {Components.Length}");
                    List<int> IDsRemain = new List<int>(Components[ic].ComponentIndices);

                    #region Junctions

                    // Find all junction points that have more than 2 branches
                    List<int> Junctions = new List<int>();
                    foreach (int id in IDsRemain)
                    {
                        int[] Neighbors = new int[]
                        {
                            id - 1,
                            id + 1,
                            id - MaskRaw.Dims.X,
                            id + MaskRaw.Dims.X,
                            id - MaskRaw.Dims.X - 1,
                            id - MaskRaw.Dims.X + 1,
                            id + MaskRaw.Dims.X - 1,
                            id + MaskRaw.Dims.X + 1
                        };

                        int NeighborCount = 0;
                        foreach (int n in Neighbors)
                            if (IDsRemain.Contains(n))
                                NeighborCount++;

                        if (NeighborCount > 2)
                            Junctions.Add(id);
                    }

                    // Process each junction - keep only 2 longest branches
                    foreach (var junction in Junctions)
                    {
                        int[] Neighbors = new int[]
                        {
                            junction - 1,
                            junction + 1,
                            junction - MaskRaw.Dims.X,
                            junction + MaskRaw.Dims.X,
                            junction - MaskRaw.Dims.X - 1,
                            junction - MaskRaw.Dims.X + 1,
                            junction + MaskRaw.Dims.X - 1,
                            junction + MaskRaw.Dims.X + 1
                        };

                        List<List<int>> Branches = new List<List<int>>();
                        List<int> IDsRemainNoJunc = new List<int>(IDsRemain);
                        IDsRemainNoJunc.Remove(junction);

                        foreach (var neighbor in Neighbors)
                            if (IDsRemainNoJunc.Contains(neighbor))
                                Branches.Add(TraceMembranesHelper.TraceLine(IDsRemainNoJunc, new(), IDsRemainNoJunc.IndexOf(neighbor), 1, MaskRaw.Dims.X));

                        // Sort branches by length and remove all but the longest two
                        Branches.Sort((a, b) => a.Count.CompareTo(b.Count));
                        foreach (var branch in Branches.Take(Branches.Count - 2))
                            IDsRemain.RemoveAll(i => branch.Contains(i));
                    }

                    // Find endpoints 
                    List<int> Endpoints = new List<int>();
                    foreach (int id in IDsRemain)
                    {
                        int[] Neighbors = new int[]
                        {
                            id - 1,
                            id + 1,
                            id - MaskRaw.Dims.X,
                            id + MaskRaw.Dims.X,
                            id - MaskRaw.Dims.X - 1,
                            id - MaskRaw.Dims.X + 1,
                            id + MaskRaw.Dims.X - 1,
                            id + MaskRaw.Dims.X + 1
                        };

                        int NeighborCount = 0;
                        foreach (int n in Neighbors)
                            if (IDsRemain.Contains(n))
                                NeighborCount++;

                        if (NeighborCount == 1)
                            Endpoints.Add(id);
                    }

                    if (Endpoints.Count > 2)
                    {
                        Console.WriteLine($"Found a path with more than 2 endpoints in {IOPath.GetFileName(MembraneSegmentationPath)}");
                        continue;
                    }

                    // Trace the final path
                    List<int> FinalPath;
                    if (Endpoints.Count == 2)
                    {
                        // Open path - trace from one endpoint to the other
                        FinalPath = TraceMembranesHelper.TraceLine(IDsRemain, new(), IDsRemain.IndexOf(Endpoints[0]), 1, MaskRaw.Dims.X);
                    }
                    else
                    {
                        // Closed path - trace from arbitrary point and close the loop
                        FinalPath = TraceMembranesHelper.TraceLine(IDsRemain, new(), 0, 1, MaskRaw.Dims.X);
                        FinalPath.Add(FinalPath.First()); // Close the loop
                    }

                    // Convert path to line segments
                    List<int2> LineSegments = new();
                    for (int i = 0; i < FinalPath.Count - 1; i++)
                        LineSegments.Add(new int2(FinalPath[i], FinalPath[i + 1]));

                    #endregion

                    #region Spline Fitting

// Create scaled trace image and get image data
                    Image TraceScaled = new Image(ImageRaw.Dims);
                    float[] ImageData = ImageRaw.GetHost(Intent.ReadWrite)[0];
                    float[] ImageLowpassData = ImageLowpass.GetHost(Intent.ReadWrite)[0];
                    toDispose.Add(TraceScaled);

// Calculate scale factor between mask and image dimensions
                    float2 ScaleFactor = new float2(ImageRaw.Dims.X / (float)MaskRaw.Dims.X,
                        ImageRaw.Dims.Y / (float)MaskRaw.Dims.Y);

// Convert line segments to points in image coordinates
                    List<float2> Points = new List<float2>();
                    foreach (var line in LineSegments)
                        Points.Add(new float2(line.X % MaskRaw.Dims.X, line.X / MaskRaw.Dims.X) * ScaleFactor);
                    Points.Add(new float2(LineSegments.Last().Y % MaskRaw.Dims.X,
                        LineSegments.Last().Y / MaskRaw.Dims.X) * ScaleFactor);

                    bool IsClosed = Points.First() == Points.Last();

                    TraceScaled.Fill(0);
                    float[] TraceScaledData = TraceScaled.GetHost(Intent.ReadWrite)[0];

// Calculate number of control points based on spacing
                    float PointSpacing = (float)options.SplinePointSpacing;
                    int NControlPoints = (int)MathF.Ceiling(Points.Count / PointSpacing) + 1;
                    if (IsClosed)
                        NControlPoints = Math.Max(NControlPoints, 3);

// Fit spline to points
                    SplinePath2D Spline = SplinePath2D.Fit(Points.ToArray(), IsClosed, NControlPoints);
                    if (Spline.IsClockwise())
                        Spline = Spline.AsReversed();

                    float2[] ControlPoints = Spline.Points.ToArray();
                    float2[] Normals = Spline.GetControlPointNormals();

// Create intensity spline
                    SplinePath1D IntensitySpline = new SplinePath1D(Helper.ArrayOfConstant(1f, 4), IsClosed);

// Get interpolated points and intensity scale factors
                    Points = Spline.GetInterpolated(Helper.ArrayOfFunction(i => (float)i / (Points.Count - 1),
                        Points.Count)).ToList();
                    List<float> ScaleFactors = IntensitySpline.GetInterpolated(
                        Helper.ArrayOfFunction(i => (float)i / (Points.Count - 1), Points.Count)).ToList();

                    #endregion

                    #region Profile Reconstruction

// Calculate membrane parameters in pixels
                    int MaxDistance = (int)MathF.Round((float)options.MembraneHalfWidth / ImageRaw.PixelSize);
                    float SoftEdge = (float)options.MembraneEdgeSoftness / ImageRaw.PixelSize;

// Create distance map from traced points
                    DistanceMap = TraceScaled.AsDistanceMapExact(MaxDistance);
                    float[] DistanceMapData = DistanceMap.GetHost(Intent.Read)[0];

// Initialize lists for membrane data
                    List<float> MembraneRefVals = new();
                    List<int2> MembranePixels = new();
                    List<int> MembraneClosestPoints = new();
                    List<float> MembraneSegmentLengths = new();
                    List<float2> MembraneTangents = new();
                    List<float2> MembraneNormals = new();
                    List<float> MembraneWeights = new();

// Populate per-pixel helper data
                    for (int i = 0; i < DistanceMapData.Length; i++)
                    {
                        if (DistanceMapData[i] < MaxDistance)
                        {
                            int2 iPoint = new int2(i % DistanceMap.Dims.X, i / DistanceMap.Dims.X);

                            MembraneRefVals.Add(ImageLowpassData[iPoint.Y * ImageRaw.Dims.X + iPoint.X]);
                            MembranePixels.Add(iPoint);

                            float2 Point = new(iPoint);

                            // Find closest point from Points
                            List<float> PointDistances = Points.Select(p => (p - Point).LengthSq()).ToList();
                            List<int> ClosestIDs = Helper.ArrayOfSequence(0, Points.Count, 1).ToList();
                            ClosestIDs.Sort((a, b) => PointDistances[a].CompareTo(PointDistances[b]));

                            // Handle special case where points are identical
                            if (Points[ClosestIDs[0]] == Points[ClosestIDs[1]])
                                ClosestIDs.RemoveAt(1);

                            int ID0 = ClosestIDs[0] < ClosestIDs[1] ? ClosestIDs[0] : ClosestIDs[1];
                            int ID1 = ClosestIDs[0] < ClosestIDs[1] ? ClosestIDs[1] : ClosestIDs[0];

                            float2 ClosestPoint0 = Points[ID0];
                            float2 ClosestPoint1 = Points[ID1];

                            MembraneClosestPoints.Add(ID0);
                            MembraneSegmentLengths.Add((ClosestPoint1 - ClosestPoint0).Length());

                            float2 Tangent = (ClosestPoint1 - ClosestPoint0).Normalized();
                            MembraneTangents.Add(Tangent);

                            float2 Normal = new(Tangent.Y, -Tangent.X);
                            MembraneNormals.Add(Normal);
                        }
                    }

                    DistanceMap.Dispose();

                    #endregion

                    #region Membrane Refinement

// Define membrane coordinate calculation
                    Func<int, float> GetMembraneCoord = (i) =>
                    {
                        float2 Location = new float2(MembranePixels[i]);
                        float2 ClosestPoint = Points[MembraneClosestPoints[i]];

                        float2 Delta = Location - ClosestPoint;
                        float LineCoord = Math.Clamp(float2.Dot(Delta, MembraneTangents[i]),
                            0,
                            MembraneSegmentLengths[i]);
                        ClosestPoint = ClosestPoint + MembraneTangents[i] * LineCoord;
                        Delta = Location - ClosestPoint;

                        float Coord = MathF.Sign(float2.Dot(Delta, MembraneNormals[i])) *
                            Delta.Length() + MaxDistance;
                        return Coord;
                    };

// Calculate membrane weights for soft edges
                    for (int i = 0; i < MembranePixels.Count; i++)
                    {
                        float Coord = (MathF.Abs(GetMembraneCoord(i) - MaxDistance) -
                                       (MaxDistance - SoftEdge)) / SoftEdge;
                        float Weight = MathF.Cos(Math.Clamp(Coord, 0, 1) * MathF.PI) * 0.5f + 0.5f;
                        MembraneWeights.Add(Weight);
                    }

// Setup profile reconstruction dimensions and data
                    int RecDim = MaxDistance * 2;
                    float[] RecData = new float[RecDim];
                    float[] RecWeights = new float[RecDim];
                    double[] OptimizationInput = new double[Spline.Points.Count + IntensitySpline.Points.Count];
                    double[] OptimizationFallback = OptimizationInput.ToArray();

// Perform membrane refinement iterations
                    for (int iiter = 0; iiter < options.RefinementIterations; iiter++)
                    {
                        // Progress?.Report($"Refining membrane {ic + 1} of {Components.Length}");

                        RecData = new float[RecDim];
                        RecWeights = new float[RecDim];

                        // Reconstruct membrane profile
                        for (int i = 0; i < MembranePixels.Count; i++)
                        {
                            float Coord = GetMembraneCoord(i);
                            float Val = ImageLowpassData[MembranePixels[i].Y * ImageRaw.Dims.X +
                                                         MembranePixels[i].X];

                            Coord = Math.Clamp(Coord, 0, RecDim - 1);
                            int Coord0 = (int)Coord;
                            int Coord1 = Math.Min(RecDim - 1, Coord0 + 1);
                            float Weight1 = (Coord - Coord0);
                            float Weight0 = 1 - Weight1;

                            RecData[Coord0] += Val * Weight0;
                            RecData[Coord1] += Val * Weight1;
                            RecWeights[Coord0] += Weight0;
                            RecWeights[Coord1] += Weight1;
                        }

                        for (int i = 0; i < RecDim; i++)
                            RecData[i] /= Math.Max(1e-16f, RecWeights[i]);

                        // Define optimization functions
                        Func<double[], double> Eval = (input) =>
                        {
                            // Update control points
                            float2[] NewControlPoints = new float2[ControlPoints.Length];
                            for (int i = 0; i < ControlPoints.Length; i++)
                                NewControlPoints[i] = ControlPoints[i] + Normals[i] * (float)input[i];

                            if (IsClosed)
                                NewControlPoints[NewControlPoints.Length - 1] = NewControlPoints[0];

                            Spline = new SplinePath2D(NewControlPoints, IsClosed);
                            Points = Spline.GetInterpolated(Helper.ArrayOfFunction(
                                i => (float)i / (Points.Count - 1), Points.Count)).ToList();

                            // Update intensities
                            float[] NewIntensityControlPoints = new float[IntensitySpline.Points.Count];
                            for (int i = 0; i < NewIntensityControlPoints.Length; i++)
                                NewIntensityControlPoints[i] = MathF.Exp((float)input[NewControlPoints.Length + i]);

                            if (IsClosed)
                                NewIntensityControlPoints[NewIntensityControlPoints.Length - 1] =
                                    NewIntensityControlPoints[0];

                            IntensitySpline = new SplinePath1D(NewIntensityControlPoints, IsClosed);
                            ScaleFactors = IntensitySpline.GetInterpolated(Helper.ArrayOfFunction(
                                i => (float)i / (Points.Count - 1), Points.Count)).ToList();

                            // Calculate RMSD
                            double Result = 0;
                            for (int i = 0; i < MembranePixels.Count; i++)
                            {
                                float Coord = GetMembraneCoord(i);
                                Coord = Math.Clamp(Coord, 0, RecDim - 1);

                                int Coord0 = (int)Coord;
                                int Coord1 = Math.Min(RecDim - 1, Coord0 + 1);
                                float Weight1 = (Coord - Coord0);
                                float Weight0 = 1 - Weight1;

                                float Val = RecData[Coord0] * Weight0 + RecData[Coord1] * Weight1;
                                float Intensity = ScaleFactors[MembraneClosestPoints[i]];
                                Val *= Intensity;

                                Result += (MembraneRefVals[i] - Val) * (MembraneRefVals[i] - Val) *
                                          MembraneWeights[i];
                            }

                            return Math.Sqrt(Result / MembranePixels.Count);
                        };

                        // Run optimization
                        try
                        {
                            BroydenFletcherGoldfarbShanno Optimizer =
                                new BroydenFletcherGoldfarbShanno(OptimizationInput.Length, Eval, null);
                            Optimizer.MaxIterations = 10;
                            Optimizer.MaxLineSearch = 5;
                            Optimizer.Minimize(OptimizationInput);
                            OptimizationFallback = OptimizationInput.ToArray();
                        }
                        catch(Exception e)
                        {
                            OptimizationInput = OptimizationFallback.ToArray();
                            Eval(OptimizationInput);
                        }
                    }

                    #endregion

                    #region Save Results

                    // Update control points with final optimization results
                    for (int i = 0; i < ControlPoints.Length; i++)
                        ControlPoints[i] = ControlPoints[i] + Normals[i] * (float)OptimizationInput[i];
                    if (IsClosed)
                        ControlPoints[ControlPoints.Length - 1] = ControlPoints[0];

                    // Create and add path table
                    PathTables.Add($"path{ic:D3}", new Star(
                        ControlPoints.Select(v => v * ImageRaw.PixelSize).ToArray(),
                        "wrpControlPointXAngst",
                        "wrpControlPointYAngst"));

                    // Save final control points
                    Directory.CreateDirectory(MembraneModelsDir);
                    Star.SaveMultitable(MembraneControlPointsPath, PathTables);

                    #endregion
                }
            }
            finally
            {
                // Clean up all allocated images
                foreach (var image in toDispose)
                    image.Dispose();
            }
        }
    }
}


[Serializable]
public class ProcessingOptionsTraceMembranes : ProcessingOptionsBase
{
    [WarpSerializable] public decimal HighResolutionLimit { get; set; } = 300;
    [WarpSerializable] public decimal LowResolutionLimit { get; set; } = 20;
    [WarpSerializable] public decimal RolloffWidth { get; set; } = 600;
    [WarpSerializable] public decimal MembraneHalfWidth { get; set; } = 60;
    [WarpSerializable] public decimal MembraneEdgeSoftness { get; set; } = 30;
    [WarpSerializable] public decimal SplinePointSpacing { get; set; } = 15;
    [WarpSerializable] public int RefinementIterations { get; set; } = 2;
    [WarpSerializable] public int MinimumComponentPixels { get; set; } = 20;
}


public static class TraceMembranesHelper
{
    public static List<int> TraceLine(List<int> Indices, List<int> Visited, int StartIndex, int Direction, int DimsX)
    {
        List<int> Result = new List<int>();
        int Current = Indices[StartIndex];
        Result.Add(Current);
        Visited.Add(Current);

        while(true)
        {
            int[] Neighbors = new int[]
            {
                Current - 1,
                Current + 1,
                Current - DimsX,
                Current + DimsX,
                Current - DimsX - 1,
                Current - DimsX + 1,
                Current + DimsX - 1,
                Current + DimsX + 1
            };

            bool Found = false;
            foreach (int n in Neighbors)
            {
                if (Indices.Contains(n) && !Visited.Contains(n))
                {
                    Current = n;
                    Result.Add(Current);
                    Visited.Add(Current);
                    Found = true;
                    break;
                }
            }

            if (!Found)
                break;
        }

        if (Direction < 0)
            Result.Reverse();

        return Result;
    }
}