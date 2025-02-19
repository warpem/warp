using System;
using System.Collections.Generic;
using System.Linq;
using Accord.Math.Optimization;
using Warp;
using Warp.Tools;

namespace Warp
{
    public partial class Movie
    {
        public void TraceMembranes()
        {
            float maxDistance = 64;
            float softEdge = 32;
            
            // Load images
            Image maskRaw = Image.FromFile(MembraneSegmentationPath);
            Image imageRaw = Image.FromFile(AveragePath);
            float angPixMic = imageRaw.PixelSize;

            // Preprocess raw image
            imageRaw = TraceMembranesHelper.PreprocessImage(imageRaw);
            Image imageLowpass = TraceMembranesHelper.CreateLowpassImage(imageRaw, angPixMic);

            // Create and process distance map
            var maskInv = TraceMembranesHelper.CreateInvertedMask(maskRaw);
            var distanceMap = TraceMembranesHelper.CreateDistanceMap(maskInv);
            var ridgePoints = TraceMembranesHelper.TraceRidges(distanceMap, new int2(maskRaw.Dims));

            // Trace membranes
            var connectedComponents = TraceMembranesHelper.FindConnectedComponents(ridgePoints, new int2(maskRaw.Dims));
            var membraneLines = TraceMembranesHelper.ProcessComponents(connectedComponents, new int2(maskRaw.Dims));

            // Refine membranes
            Dictionary<string, Star> PathTables = new();
            List<MembraneProfile> MembraneProfiles = new();

            for (int ilines = 0; ilines < membraneLines.Count; ilines++)
            {
                var (paths, profile) = TraceMembranesHelper.RefineMembraneCoordinates(
                    membraneLines[ilines],
                    imageRaw,
                    imageLowpass,
                    angPixMic,
                    maxDistance,
                    softEdge
                );

                PathTables.Add($"path{ilines:D3}", paths);
                MembraneProfiles.Add(profile);
            }

            // Save results
            Star.SaveMultitable(MembraneControlPointsPath, PathTables);
            TraceMembranesHelper.SaveMembraneProfiles(MembraneProfilesPath, MembraneProfiles, angPixMic);

            // Cleanup
            maskRaw.Dispose();
            imageRaw.Dispose();
            imageLowpass.Dispose();
        }
    }
}

public static class TraceMembranesHelper
{
    public static Image PreprocessImage(Image raw)
    {
        raw.SubtractMeanGrid(new int2(1));
        raw = raw.AsPadded(raw.DimsSlice * 2).AndDisposeParent();

        // Apply bandpass filter
        float angPix = raw.PixelSize;
        raw.Bandpass(angPix * 2 / 300f, 1, false, angPix * 2 / 600f);

        return raw;
    }

    public static Image CreateLowpassImage(Image raw, float angPix)
    {
        Image lowpass = raw.GetCopyGPU();
        lowpass.Bandpass(angPix * 2 / 300f, angPix * 2 / 20f, false, angPix * 2 / 600f);
        return lowpass;
    }

    public static Image CreateInvertedMask(Image mask)
    {
        Image inverted = new Image(IntPtr.Zero, mask.Dims);
        inverted.Fill(1f);
        inverted.Subtract(mask);
        return inverted;
    }

    public static Image CreateDistanceMap(Image maskInv)
    {
        Image distMap = new Image(IntPtr.Zero, maskInv.Dims);
        GPU.DistanceMapExact(maskInv.GetDevice(Intent.Read),
            distMap.GetDevice(Intent.Write),
            maskInv.Dims,
            20);
        return distMap;
    }

    public static List<int> TraceRidges(Image distanceMap, int2 dims)
    {
        List<int> ridgePoints = new List<int>();
        float[] distData = distanceMap.GetHost(Intent.Read)[0];

        // Trace ridges as points at which the local gradient peaks
        for (int i = dims.X + 1; i < distData.Length - dims.X - 1; i++)
        {
            float center = distData[i];

            if (center < 2) // At least 32 Angstrom thick
                continue;

            // Check for ridge points in multiple directions
            if (IsRidgePoint(distData, i, dims, center))
                ridgePoints.Add(i);
        }

        return ridgePoints;
    }

    public static bool IsRidgePoint(float[] distData, int index, int2 dims, float center)
    {
        // Simple case: 1-pixel thick ridge
        if (distData[index - 1] < center && distData[index + 1] < center)
            return true;

        if (distData[index - dims.X] < center && distData[index + dims.X] < center)
            return true;

        // Diagonal directions
        if (distData[index - 1 - dims.X] < center && distData[index + 1 + dims.X] < center)
            return true;

        if (distData[index - 1 + dims.X] < center && distData[index + 1 - dims.X] < center)
            return true;

        // Handle 2-pixel thick ridges
        return Handle2PixelRidge(distData, index, dims, center);
    }

    public static bool Handle2PixelRidge(float[] distData, int index, int2 dims, float center)
    {
        // Check for 2-pixel thick ridges in various directions
        if (distData[index - 1] == distData[index + 1])
        {
            if (distData[index - dims.X] < center && distData[index + dims.X] <= center)
                return true;
        }

        if (distData[index - dims.X] == distData[index + dims.X])
        {
            if (distData[index - 1] < center && distData[index + 1] <= center)
                return true;
        }

        // Diagonal cases
        if (distData[index - 1 - dims.X] == distData[index + 1 + dims.X])
        {
            if (distData[index - 1 + dims.X] < center && distData[index + 1 - dims.X] <= center)
                return true;
        }

        if (distData[index - 1 + dims.X] == distData[index + 1 - dims.X])
        {
            if (distData[index - 1 - dims.X] < center && distData[index + 1 + dims.X] <= center)
                return true;
        }

        return false;
    }

    public static List<ComponentInfo> FindConnectedComponents(List<int> points, int2 dims)
    {
        var components = new List<ComponentInfo>();
        HashSet<int> remainingPoints = new HashSet<int>(points);
        int minComponentSize = 20;

        while(remainingPoints.Count > 0)
        {
            var component = new ComponentInfo();
            var startPoint = remainingPoints.First();
            TraceComponent(startPoint, remainingPoints, component, dims);

            if (component.PointIdx.Count >= minComponentSize)
                components.Add(component);
        }

        return components;
    }

    public static void TraceComponent(int startPoint,
        HashSet<int> remainingPoints,
        ComponentInfo component,
        int2 dims)
    {
        Queue<int> queue = new Queue<int>();
        queue.Enqueue(startPoint);
        remainingPoints.Remove(startPoint);
        component.PointIdx.Add(startPoint);

        while(queue.Count > 0)
        {
            int current = queue.Dequeue();
            foreach (int neighbor in GetNeighbors(current, dims))
            {
                if (remainingPoints.Contains(neighbor))
                {
                    queue.Enqueue(neighbor);
                    remainingPoints.Remove(neighbor);
                    component.PointIdx.Add(neighbor);
                }
            }
        }
    }

    public static IEnumerable<int> GetNeighbors(int point, int2 dims)
    {
        return new[]
        {
            point - 1,
            point + 1,
            point - dims.X,
            point + dims.X,
            point - dims.X - 1,
            point - dims.X + 1,
            point + dims.X - 1,
            point + dims.X + 1
        };
    }

    public static (Star paths, MembraneProfile profile) RefineMembraneCoordinates(
        List<int2> lineSegments,
        Image imageRaw,
        Image imageLowpass,
        float angPixMic,
        float maxDistance,
        float softEdge)
    {
        // Convert line segments to continuous path
        var pathSegment = CreatePathSegment(lineSegments, new int2(imageRaw.Dims));

        // Fit spline to path
        var spline = FitSplineToPath(pathSegment);

        // Create profile
        var profile = CreateMembraneProfile(
            spline,
            imageRaw,
            imageLowpass,
            maxDistance,
            softEdge
        );

        // Optimize path coordinates
        var optimizedPoints = OptimizePathCoordinates(
            spline,
            profile,
            imageRaw,
            imageLowpass
        );

        // Create STAR table with results
        var paths = new Star(optimizedPoints.Select(p => p * angPixMic).ToArray(),
            "wrpControlPointXAngst",
            "wrpControlPointYAngst");

        return (paths, profile);
    }

    public static float2[] OptimizePathCoordinates(SplinePath2D spline,
        MembraneProfile profile,
        Image imageRaw,
        Image imageLowpass)
    {
        float2[] controlPoints = spline.Points.ToArray();
        float2[] normals = spline.GetControlPointNormals();
        bool isClosed = spline.IsClosed;

        // Create intensity spline for scaling
        SplinePath1D intensitySpline = new SplinePath1D(Helper.ArrayOfConstant(1f, 4), isClosed);
        List<float> scaleFactors = intensitySpline.GetInterpolated(
            Helper.ArrayOfFunction(i => (float)i / (controlPoints.Length - 1),
                controlPoints.Length)).ToList();

        // Setup optimization input vector:
        // First part: Control point displacements along their normals
        // Second part: Log of intensity scale factors
        double[] optimizationInput = new double[controlPoints.Length + intensitySpline.Points.Count];
        double[] optimizationFallback = optimizationInput.ToArray();

        // Define evaluation function for optimization
        Func<double[], double> eval = (input) =>
        {
            // Update control points based on input displacements
            float2[] newControlPoints = new float2[controlPoints.Length];
            for (int i = 0; i < controlPoints.Length; i++)
                newControlPoints[i] = controlPoints[i] + normals[i] * (float)input[i];

            if (isClosed)
                newControlPoints[newControlPoints.Length - 1] = newControlPoints[0];

            // Create new spline with updated control points
            SplinePath2D newSpline = new SplinePath2D(newControlPoints, isClosed);

            // Update intensity control points
            float[] newIntensityControlPoints = new float[intensitySpline.Points.Count];
            for (int i = 0; i < newIntensityControlPoints.Length; i++)
                newIntensityControlPoints[i] = MathF.Exp((float)input[newControlPoints.Length + i]);

            if (isClosed)
                newIntensityControlPoints[newIntensityControlPoints.Length - 1] = newIntensityControlPoints[0];

            // Create new intensity spline and get interpolated values
            SplinePath1D newIntensitySpline = new SplinePath1D(newIntensityControlPoints, isClosed);
            List<float> newScaleFactors = newIntensitySpline.GetInterpolated(
                Helper.ArrayOfFunction(i => (float)i / (controlPoints.Length - 1),
                    controlPoints.Length)).ToList();

            // Calculate error between model and experimental data
            float[] rawData = imageRaw.GetHost(Intent.Read)[0];
            float[] lowpassData = imageLowpass.GetHost(Intent.Read)[0];

            double totalError = 0;
            int pointCount = 0;

            // Sample points along the membrane
            var t = Helper.ArrayOfFunction(i => (float)i / (controlPoints.Length * 10), controlPoints.Length * 10);
            List<float2> pathPoints = newSpline.GetInterpolated(t).ToList();
            List<float2> normals = newSpline.GetNormals(t).ToList();

            for (int ip = 0; ip < pathPoints.Count; ip++)
            {
                float2 point = pathPoints[ip];
                float2 normal = normals[ip];
                float scale = newScaleFactors[ip * controlPoints.Length / pathPoints.Count];

                // Sample points perpendicular to membrane
                for (float d = -profile.Dimension / 2; d <= profile.Dimension / 2; d++)
                {
                    float2 samplePoint = point + normal * d;
                    int x = (int)MathF.Round(samplePoint.X);
                    int y = (int)MathF.Round(samplePoint.Y);

                    if (x < 0 || x >= imageRaw.Dims.X || y < 0 || y >= imageRaw.Dims.Y)
                        continue;

                    int index = y * imageRaw.Dims.X + x;
                    float experimental = lowpassData[index];

                    // Get profile value and weight
                    int profileIndex = (int)MathF.Round(d + profile.Dimension / 2);
                    if (profileIndex < 0 || profileIndex >= profile.Dimension)
                        continue;

                    float model = profile.ProfileData[profileIndex] * scale;
                    float weight = profile.WeightData[profileIndex];

                    // Add weighted squared error
                    double error = (experimental - model) * (experimental - model) * weight;
                    totalError += error;
                    pointCount++;
                }
            }

            return Math.Sqrt(totalError / Math.Max(1, pointCount));
        };

        // Define gradient function for optimization
        Func<double[], double[]> grad = (input) =>
        {
            double[] result = new double[input.Length];

            // Calculate gradient numerically
            for (int i = 0; i < input.Length - (isClosed ? 1 : 0); i++)
            {
                double[] inputPlus = input.ToArray();
                inputPlus[i] += 1e-3;
                double[] inputMinus = input.ToArray();
                inputMinus[i] -= 1e-3;

                result[i] = (eval(inputPlus) - eval(inputMinus)) / 2e-3;
            }

            return result;
        };

        // Run optimization
        try
        {
            BroydenFletcherGoldfarbShanno optimizer = new BroydenFletcherGoldfarbShanno(optimizationInput.Length, eval, grad);
            optimizer.MaxIterations = 10;
            optimizer.MaxLineSearch = 5;
            optimizer.Minimize(optimizationInput);
            optimizationFallback = optimizationInput.ToArray();
        }
        catch(Exception e)
        {
            Console.WriteLine($"Optimization failed: {e.Message}");
            optimizationInput = optimizationFallback.ToArray();
        }

        // Create final control points with optimized positions
        float2[] finalControlPoints = new float2[controlPoints.Length];
        for (int i = 0; i < controlPoints.Length; i++)
            finalControlPoints[i] = controlPoints[i] + normals[i] * (float)optimizationInput[i];

        if (isClosed)
            finalControlPoints[finalControlPoints.Length - 1] = finalControlPoints[0];

        return finalControlPoints;
    }

    public static PathSegment CreatePathSegment(List<int2> lineSegments, int2 dims)
    {
        var segment = new PathSegment();

        // Convert line segments to points
        foreach (var line in lineSegments)
        {
            segment.Points.Add(new float2(line.X % dims.X, line.X / dims.X));
        }

        // Check if path is closed
        segment.IsClosed = (segment.Points.First() - segment.Points.Last()).LengthSq() < 1e-5f;

        return segment;
    }

    public static SplinePath2D FitSplineToPath(PathSegment path)
    {
        float pointSpacing = 15;
        int nControlPoints = (int)MathF.Ceiling(path.Points.Count / pointSpacing) + 1;

        if (path.IsClosed)
            nControlPoints = Math.Max(nControlPoints, 3);

        return SplinePath2D.Fit(path.Points.ToArray(),
            path.IsClosed,
            nControlPoints);
    }

    public static MembraneProfile CreateMembraneProfile(
        SplinePath2D spline,
        Image imageRaw,
        Image imageLowpass,
        float maxDistance,
        float softEdge
    )
    {
        int profileDim = (int)(maxDistance * 2);
        var profile = new MembraneProfile(profileDim);

        // Get image data
        float[] rawData = imageRaw.GetHost(Intent.Read)[0];
        float[] lowpassData = imageLowpass.GetHost(Intent.Read)[0];

        // Get interpolated points along the spline
        List<float2> points = spline.GetInterpolated(
            Helper.ArrayOfFunction(i => (float)i / (spline.Points.Count - 1),
                spline.Points.Count)).ToList();

        // Calculate membrane pixels and their properties
        List<int2> membranePixels = new List<int2>();
        List<float> membraneRefVals = new List<float>();
        List<int> membraneClosestPoints = new List<int>();
        List<float> membraneSegmentLengths = new List<float>();
        List<float2> membraneTangents = new List<float2>();
        List<float2> membraneNormals = new List<float2>();
        List<float> membraneWeights = new List<float>();

        // For each pixel in potential membrane region
        for (int y = 0; y < imageRaw.Dims.Y; y++)
        {
            for (int x = 0; x < imageRaw.Dims.X; x++)
            {
                float2 pixel = new float2(x, y);

                // Find closest spline points
                List<float> pointDistances = points.Select(p => (p - pixel).LengthSq()).ToList();
                List<int> closestIDs = Helper.ArrayOfSequence(0, points.Count, 1).ToList();
                closestIDs.Sort((a, b) => pointDistances[a].CompareTo(pointDistances[b]));

                // Only process pixels within maxDistance
                if (MathF.Sqrt(pointDistances[closestIDs[0]]) > maxDistance)
                    continue;

                // Get reference intensity from lowpass image
                membraneRefVals.Add(lowpassData[y * imageRaw.Dims.X + x]);
                membranePixels.Add(new int2(x, y));

                // Calculate tangent and normal
                int id0 = Math.Min(closestIDs[0], closestIDs[1]);
                int id1 = Math.Max(closestIDs[0], closestIDs[1]);
                float2 point0 = points[id0];
                float2 point1 = points[id1];

                membraneClosestPoints.Add(id0);
                membraneSegmentLengths.Add((point1 - point0).Length());

                float2 tangent = (point1 - point0).Normalized();
                membraneTangents.Add(tangent);

                float2 normal = new float2(tangent.Y, -tangent.X);
                membraneNormals.Add(normal);
            }
        }

        // Calculate membrane coordinates and weights
        for (int i = 0; i < membranePixels.Count; i++)
        {
            float2 location = new float2(membranePixels[i]);
            float2 closestPoint = points[membraneClosestPoints[i]];

            // Calculate position along membrane
            float2 delta = location - closestPoint;
            float lineCoord = Math.Clamp(
                float2.Dot(delta, membraneTangents[i]),
                0,
                membraneSegmentLengths[i]);

            closestPoint += membraneTangents[i] * lineCoord;
            delta = location - closestPoint;

            // Calculate coordinate in membrane reference frame
            float coord = MathF.Sign(float2.Dot(delta, membraneNormals[i])) * delta.Length() + maxDistance;
            coord = Math.Clamp(coord, 0, profileDim - 1);

            // Calculate weight based on distance from membrane
            float distFromEdge = (MathF.Abs(coord - maxDistance) - (maxDistance - softEdge)) / softEdge;
            float weight = MathF.Cos(Math.Clamp(distFromEdge, 0, 1) * MathF.PI) * 0.5f + 0.5f;

            // Add to profile
            int coord0 = (int)coord;
            int coord1 = Math.Min(profileDim - 1, coord0 + 1);
            float weight1 = (coord - coord0);
            float weight0 = 1 - weight1;

            profile.ProfileData[coord0] += membraneRefVals[i] * weight0 * weight;
            profile.ProfileData[coord1] += membraneRefVals[i] * weight1 * weight;
            profile.WeightData[coord0] += weight0 * weight;
            profile.WeightData[coord1] += weight1 * weight;
        }

        // Normalize profile
        for (int i = 0; i < profileDim; i++)
        {
            if (profile.WeightData[i] > 1e-16f)
            {
                profile.ProfileData[i] /= profile.WeightData[i];
            }
        }

        return profile;
    }

    public static List<List<int2>> ProcessComponents(List<ComponentInfo> components, int2 dims)
    {
        List<List<int2>> componentLines = new List<List<int2>>();

        foreach (var component in components)
        {
            List<int> remainingIds = new List<int>(component.PointIdx);

            #region Find Junctions

            // Find all junction points that have more than 2 branches
            List<int> junctions = new List<int>();
            foreach (int id in remainingIds)
            {
                int[] neighbors = new int[]
                {
                    id - 1,
                    id + 1,
                    id - dims.X,
                    id + dims.X,
                    id - dims.X - 1,
                    id - dims.X + 1,
                    id + dims.X - 1,
                    id + dims.X + 1
                };

                int neighborCount = 0;
                foreach (int n in neighbors)
                    if (remainingIds.Contains(n))
                        neighborCount++;

                if (neighborCount > 2)
                    junctions.Add(id);
            }

            // Process each junction and keep only the two longest branches
            foreach (var junction in junctions)
            {
                int[] neighbors = new int[]
                {
                    junction - 1,
                    junction + 1,
                    junction - dims.X,
                    junction + dims.X,
                    junction - dims.X - 1,
                    junction - dims.X + 1,
                    junction + dims.X - 1,
                    junction + dims.X + 1
                };

                List<List<int>> branches = new List<List<int>>();
                List<int> idsWithoutJunction = new List<int>(remainingIds);
                idsWithoutJunction.Remove(junction);

                foreach (var neighbor in neighbors)
                    if (idsWithoutJunction.Contains(neighbor))
                        branches.Add(TraceLine(idsWithoutJunction, new(), idsWithoutJunction.IndexOf(neighbor), 1, dims.X));

                // Sort branches by length and remove all but the two longest
                branches.Sort((a, b) => b.Count.CompareTo(a.Count));
                foreach (var branch in branches.Skip(2))
                    remainingIds.RemoveAll(i => branch.Contains(i));
            }

            #endregion

            #region Find Endpoints

            // If path isn't closed, it should have 2 endpoints
            List<int> endpoints = new List<int>();
            foreach (int id in remainingIds)
            {
                int[] neighbors = new int[]
                {
                    id - 1,
                    id + 1,
                    id - dims.X,
                    id + dims.X,
                    id - dims.X - 1,
                    id - dims.X + 1,
                    id + dims.X - 1,
                    id + dims.X + 1
                };

                int neighborCount = 0;
                foreach (int n in neighbors)
                    if (remainingIds.Contains(n))
                        neighborCount++;

                if (neighborCount == 1)
                    endpoints.Add(id);
            }

            if (endpoints.Count > 2)
            {
                Console.WriteLine($"Warning: Found a path with more than 2 endpoints, skipping component");
                continue;
            }

            #endregion

            #region Create Final Path

            List<int> finalPath;
            if (endpoints.Count == 2)
            {
                // Open path with two endpoints
                finalPath = TraceLine(remainingIds, new(), remainingIds.IndexOf(endpoints[0]), 1, dims.X);
            }
            else
            {
                // Closed path with no endpoints
                finalPath = TraceLine(remainingIds, new(), 0, 1, dims.X);
                finalPath.Add(finalPath[0]); // Close the loop by adding first point again
            }

            #endregion

            // Convert path to line segments
            List<int2> lineSegments = new List<int2>();
            for (int i = 0; i < finalPath.Count - 1; i++)
                lineSegments.Add(new int2(finalPath[i], finalPath[i + 1]));

            componentLines.Add(lineSegments);
        }

        return componentLines;
    }

    public static List<int> TraceLine(List<int> points, List<int> visited, int currentIndex, int depth, int width)
    {
        if (currentIndex < 0 || currentIndex >= points.Count)
            return visited;

        int currentPoint = points[currentIndex];
        visited.Add(currentPoint);

        // Get all neighbors
        int[] neighbors = new int[]
        {
            currentPoint - 1,
            currentPoint + 1,
            currentPoint - width,
            currentPoint + width,
            currentPoint - width - 1,
            currentPoint - width + 1,
            currentPoint + width - 1,
            currentPoint + width + 1
        };

        // Find next unvisited neighbor
        foreach (int neighbor in neighbors)
        {
            int nextIndex = points.IndexOf(neighbor);
            if (nextIndex >= 0 && !visited.Contains(neighbor))
            {
                return TraceLine(points, visited, nextIndex, depth + 1, width);
            }
        }

        return visited;
    }

    public static void SaveMembraneProfiles(string filename, List<(MembraneProfile Profile, PathSegment Path)> membranes, float angPixMic)
    {
        Dictionary<string, Star> tables = new Dictionary<string, Star>();

        for (int i = 0; i < membranes.Count; i++)
        {
            var (profile, path) = membranes[i];

            // Convert control points to float2[] and scale to Angstroms
            float2[] controlPoints = new float2[path.ControlPoints.Count];
            for (int j = 0; j < path.ControlPoints.Count; j++)
            {
                controlPoints[j] = path.ControlPoints[j] * angPixMic;
            }

            // Use the correct Star constructor that takes float2[]
            tables.Add($"path{i:D3}", new Star(controlPoints, "wrpControlPointXAngst", "wrpControlPointYAngst"));
        }

        Star.SaveMultitable(filename, tables);
    }
}


public class MembraneProfile
{
    public float[] ProfileData { get; set; }
    public float[] WeightData { get; set; }
    public float ScaleFactors { get; set; }
    public int Dimension { get; set; }

    public MembraneProfile(int dimension)
    {
        Dimension = dimension;
        ProfileData = new float[dimension];
        WeightData = new float[dimension];
        ScaleFactors = 1.0f;
    }
}

public class PathSegment
{
    public List<float2> Points { get; set; }
    public bool IsClosed { get; set; }
    public List<float2> ControlPoints { get; set; }
    public List<float2> Normals { get; set; }

    public PathSegment()
    {
        Points = new List<float2>();
        ControlPoints = new List<float2>();
        Normals = new List<float2>();
        IsClosed = false;
    }
}

public class ComponentInfo
{
    public List<int> PointIdx { get; private set; }
    public bool IsClosed { get; set; }

    // Statistics about the component
    public int MinX { get; private set; }
    public int MaxX { get; private set; }
    public int MinY { get; private set; }
    public int MaxY { get; private set; }
    public float CenterX { get; private set; }
    public float CenterY { get; private set; }

    public ComponentInfo()
    {
        PointIdx = new List<int>();
        IsClosed = false;
        MinX = int.MaxValue;
        MaxX = int.MinValue;
        MinY = int.MaxValue;
        MaxY = int.MinValue;
    }

    public void AddPoint(int index, int2 dims)
    {
        PointIdx.Add(index);

        // Calculate X,Y coordinates from the linear index
        int x = index % dims.X;
        int y = index / dims.X;

        // Update bounds
        MinX = Math.Min(MinX, x);
        MaxX = Math.Max(MaxX, x);
        MinY = Math.Min(MinY, y);
        MaxY = Math.Max(MaxY, y);

        // Update center
        CenterX = (MinX + MaxX) / 2.0f;
        CenterY = (MinY + MaxY) / 2.0f;
    }

    public int Width => MaxX - MinX + 1;
    public int Height => MaxY - MinY + 1;
    public int BoundingArea => Width * Height;
    public int Size => PointIdx.Count;
    public float Density => PointIdx.Count / (float)BoundingArea;

    public bool TouchesEdge(int2 imageDims)
    {
        return MinX == 0 || MaxX == imageDims.X - 1 ||
               MinY == 0 || MaxY == imageDims.Y - 1;
    }

    public List<int2> GetLocalCoordinates(int2 dims)
    {
        List<int2> localPoints = new List<int2>();
        foreach (int index in PointIdx)
        {
            int x = index % dims.X - MinX;
            int y = index / dims.X - MinY;
            localPoints.Add(new int2(x, y));
        }

        return localPoints;
    }

    // Create a binary mask of just this component
    public float[] CreateMask(int2 dims)
    {
        float[] mask = new float[dims.X * dims.Y];
        foreach (int index in PointIdx)
        {
            mask[index] = 1.0f;
        }

        return mask;
    }

    public bool IsConnectedTo(ComponentInfo other, int2 dims)
    {
        foreach (int point in PointIdx)
        {
            // Get all 8 neighbors
            int[] neighbors = new[]
            {
                point - 1, // Left
                point + 1, // Right
                point - dims.X, // Up
                point + dims.X, // Down
                point - dims.X - 1, // Up-Left
                point - dims.X + 1, // Up-Right
                point + dims.X - 1, // Down-Left
                point + dims.X + 1 // Down-Right
            };

            // Check if any neighbor is in the other component
            if (neighbors.Any(n => other.PointIdx.Contains(n)))
                return true;
        }

        return false;
    }

    public float MinimumDistanceTo(ComponentInfo other, int2 dims)
    {
        float minDist = float.MaxValue;

        foreach (int p1 in PointIdx)
        {
            int x1 = p1 % dims.X;
            int y1 = p1 / dims.X;

            foreach (int p2 in other.PointIdx)
            {
                int x2 = p2 % dims.X;
                int y2 = p2 / dims.X;

                float dx = x2 - x1;
                float dy = y2 - y1;
                float dist = MathF.Sqrt(dx * dx + dy * dy);

                minDist = Math.Min(minDist, dist);
            }
        }

        return minDist;
    }
}