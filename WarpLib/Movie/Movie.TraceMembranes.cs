using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Accord.Math.Optimization;
using Warp;
using Warp.Tools;

namespace Warp
{
    public partial class Movie
    {
        public void TraceMembranes(ProcessingOptionsTraceMembranes options)
        {
            Console.WriteLine("Starting membrane tracing...");

            // Load necessary input files
            Image maskRaw = Image.FromFile(AveragePath);
            Image imageRaw = Image.FromFile(MembraneSegmentationPath);

            // Apply bandpass filtering
            Image imageLowpass = imageRaw.GetCopyGPU();
            imageLowpass.Bandpass(options.BandpassLow, options.BandpassHigh, false);

            // Skeletonize the mask
            Image ridgeMask = TraceMembranesHelper.Skeletonize(maskRaw, 2.0f);

            // Detect connected components (membranes)
            var components = TraceMembranesHelper.FindConnectedComponents(ridgeMask);
            ridgeMask.Dispose();

            int totalMembranes = components.Count;

            if (totalMembranes == 0)
                throw new Exception("No valid membrane components detected.");

            List<SplinePath2D> splines = new List<SplinePath2D>();

            for (int i = 0; i < totalMembranes; i++)
            {
                Console.WriteLine($"Tracing membrane {i + 1} of {totalMembranes} (0%)");

                var component = components[i];
                var spline = TraceMembranesHelper.FitSplineToComponent(component);

                Console.WriteLine($"Tracing membrane {i + 1} of {totalMembranes} (50%)");


                spline = TraceMembranesHelper.RefineSplineControlPoints(imageLowpass, spline, 10);
                Console.WriteLine($"Refining membrane {i + 1} of {totalMembranes} (100%)");

                splines.Add(spline);
            }

            // Save output
            TraceMembranesHelper.SaveControlPoints(MembraneControlPointsPath, splines);
            Console.WriteLine("Membrane tracing completed.");

            // Cleanup
            imageRaw.Dispose();
            imageLowpass.Dispose();
            maskRaw.Dispose();
        }
    }

    [Serializable]
    public class ProcessingOptionsTraceMembranes : ProcessingOptionsBase
    {
        // Placeholder properties for future use
        [WarpSerializable] public int MinComponentSize { get; set; } = 20; // px
        [WarpSerializable] public float BandpassLow { get; set; } = 0.002f;
        [WarpSerializable] public float BandpassHigh { get; set; } = 0.05f;
    }
}

public static class TraceMembranesHelper
{
    public static Image ComputeDistanceMap(Image binaryMask)
    {
        int width = binaryMask.Dims.X;
        int height = binaryMask.Dims.Y;

        Image maskInv = new Image(IntPtr.Zero, new int3(width, height, 1));
        Image distanceMap = new Image(IntPtr.Zero, new int3(width, height, 1));

        maskInv.Fill(1f);
        maskInv.Subtract(binaryMask);

        GPU.DistanceMapExact(maskInv.GetDevice(Intent.Read), distanceMap.GetDevice(Intent.Write), new int3(width, height, 1), 20);

        maskInv.Dispose();
        return distanceMap;
    }

    public static Image Skeletonize(Image mask, float minDistanceThreshold)
    {
        Image distanceMap = ComputeDistanceMap(mask);
        
        int width = distanceMap.Dims.X;
        int height = distanceMap.Dims.Y;
        Image ridgeMask = new Image(distanceMap.Dims);

        float[] distanceData = distanceMap.GetHost(Intent.Read)[0];
        float[] ridgeData = ridgeMask.GetHost(Intent.ReadWrite)[0];

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int index = y * width + x;
                float centerValue = distanceData[index];

                if (centerValue < minDistanceThreshold)
                    continue;

                // Check local maxima in principal directions
                bool isRidge =
                    (distanceData[index - 1] < centerValue && distanceData[index + 1] < centerValue) || // Horizontal
                    (distanceData[index - width] < centerValue && distanceData[index + width] < centerValue) || // Vertical
                    (distanceData[index - width - 1] < centerValue && distanceData[index + width + 1] < centerValue) || // Diagonal 1
                    (distanceData[index - width + 1] < centerValue && distanceData[index + width - 1] < centerValue); // Diagonal 2

                if (isRidge)
                    ridgeData[index] = 1.0f;
            }
        }

        // Process junctions: Keep only the two longest branches
        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int index = y * width + x;
                if (ridgeData[index] != 1.0f)
                    continue;

                int[] neighbors =
                {
                    index - 1, index + 1, index - width, index + width,
                    index - width - 1, index - width + 1, index + width - 1, index + width + 1
                };

                List<int> connected = new List<int>();
                foreach (int neighbor in neighbors)
                {
                    if (ridgeData[neighbor] == 1.0f)
                        connected.Add(neighbor);
                }

                if (connected.Count <= 2)
                    continue;

                // Trace each branch and determine its length
                List<(int start, int length)> branches = new List<(int, int)>();
                foreach (int start in connected)
                {
                    HashSet<int> visited = new HashSet<int>();
                    int length = TraceBranch(start, ridgeData, width, height, visited);
                    branches.Add((start, length));
                }

                // Keep only the two longest branches
                branches.Sort((a, b) => b.length.CompareTo(a.length));
                HashSet<int> keep = new HashSet<int> { branches[0].start, branches[1].start };

                foreach (int start in connected)
                {
                    if (!keep.Contains(start))
                        ridgeData[start] = 0.0f;
                }
            }
        }
        distanceMap.Dispose();
        return ridgeMask;
    }

    private static int TraceBranch(int start, float[] ridgeData, int width, int height, HashSet<int> visited)
    {
        Queue<int> queue = new Queue<int>();
        queue.Enqueue(start);
        int length = 0;

        while(queue.Count > 0)
        {
            int index = queue.Dequeue();
            if (visited.Contains(index) || ridgeData[index] != 1.0f)
                continue;

            visited.Add(index);
            length++;

            int[] neighbors =
            {
                index - 1, index + 1, index - width, index + width,
                index - width - 1, index - width + 1, index + width - 1, index + width + 1
            };

            foreach (int neighbor in neighbors)
            {
                if (!visited.Contains(neighbor) && ridgeData[neighbor] == 1.0f)
                    queue.Enqueue(neighbor);
            }
        }

        return length;
    }

    public static List<List<(int x, int y)>> FindConnectedComponents(Image ridgeMask)
    {
        int width = ridgeMask.Dims.X;
        int height = ridgeMask.Dims.Y;
        float[] data = ridgeMask.GetHost(Intent.Read)[0];

        bool[,] visited = new bool[height, width];
        List<List<(int x, int y)>> components = new();

        // Neighbor offsets for 8-connectivity
        int[] dx = [-1, 0, 1, -1, 1, -1, 0, 1];
        int[] dy = [-1, -1, -1, 0, 0, 1, 1, 1];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (data[y * width + x] == 1 && !visited[y, x])
                {
                    List<(int x, int y)> component = new();
                    Queue<(int x, int y)> queue = new();

                    queue.Enqueue((x, y));
                    visited[y, x] = true;

                    while(queue.Count > 0)
                    {
                        (int cx, int cy) = queue.Dequeue();
                        component.Add((cx, cy));

                        for (int i = 0; i < 8; i++)
                        {
                            int nx = cx + dx[i];
                            int ny = cy + dy[i];

                            if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                                data[ny * width + nx] == 1 && !visited[ny, nx])
                            {
                                queue.Enqueue((nx, ny));
                                visited[ny, nx] = true;
                            }
                        }
                    }

                    components.Add(component);
                }
            }
        }

        return components;
    }

    public static SplinePath2D FitSplineToComponent(List<(int x, int y)> componentPixels)
    {
        if (componentPixels == null || componentPixels.Count < 3)
            throw new ArgumentException("Component must have at least three pixels to fit a spline.");

        // Convert pixel coordinates to float2 points for spline fitting
        List<float2> points = componentPixels.Select(p => new float2(p.x, p.y)).ToList();

        // Check if the component forms a closed loop
        bool isClosed = points.First().Equals(points.Last());

        // Determine the number of control points based on component length
        int pointSpacing = 15;
        int numControlPoints = (int)MathF.Ceiling(points.Count / (float)pointSpacing) + 1;
        if (isClosed)
            numControlPoints = Math.Max(numControlPoints, 3);

        // Fit a spline to the extracted points
        SplinePath2D spline = SplinePath2D.Fit(points.ToArray(), isClosed, numControlPoints);

        // Ensure correct orientation (clockwise convention)
        if (spline.IsClockwise())
            spline = spline.AsReversed();

        return spline;
    }

    public static float[] ReconstructMembraneProfile(Image lowpassImage, SplinePath2D spline)
    {
        int maxDistance = 60; // Assumed membrane width in pixels
        int recDim = maxDistance * 2;
        float[] recData = new float[recDim];
        float[] recWeights = new float[recDim];

        // Compute distance map along the spline
        Image traceImage = new Image(lowpassImage.Dims);
        float[] traceData = traceImage.GetHost(Intent.ReadWrite)[0];

        List<float2> points = spline.GetInterpolated(Helper.ArrayOfFunction(i => (float)i / (spline.Points.Count - 1), spline.Points.Count)).ToList();

        // Rasterize spline onto the image
        for (int i = 0; i < points.Count - 1; i++)
        {
            float2 p0 = points[i];
            float2 p1 = points[i + 1];

            int x0 = (int)MathF.Round(p0.X);
            int y0 = (int)MathF.Round(p0.Y);
            int x1 = (int)MathF.Round(p1.X);
            int y1 = (int)MathF.Round(p1.Y);

            if (x0 < 0 || x0 >= traceImage.Dims.X || y0 < 0 || y0 >= traceImage.Dims.Y) continue;
            if (x1 < 0 || x1 >= traceImage.Dims.X || y1 < 0 || y1 >= traceImage.Dims.Y) continue;

            int dx = Math.Abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
            int dy = -Math.Abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
            int err = dx + dy, e2;

            while(true)
            {
                traceData[y0 * traceImage.Dims.X + x0] = 1;
                if (x0 == x1 && y0 == y1) break;
                e2 = 2 * err;
                if (e2 >= dy)
                {
                    err += dy;
                    x0 += sx;
                }

                if (e2 <= dx)
                {
                    err += dx;
                    y0 += sy;
                }
            }
        }

        Image distanceMap = traceImage.AsDistanceMapExact(maxDistance);
        float[] distanceMapData = distanceMap.GetHost(Intent.Read)[0];

        // Compute membrane profile by accumulating intensities
        for (int i = 0; i < distanceMapData.Length; i++)
        {
            if (distanceMapData[i] < maxDistance)
            {
                float coord = distanceMapData[i] + maxDistance;
                coord = Math.Clamp(coord, 0, recDim - 1);
                int coord0 = (int)coord;
                int coord1 = Math.Min(recDim - 1, coord0 + 1);
                float weight1 = coord - coord0;
                float weight0 = 1 - weight1;

                float val = lowpassImage.GetHost(Intent.Read)[0][i];

                recData[coord0] += val * weight0;
                recData[coord1] += val * weight1;
                recWeights[coord0] += weight0;
                recWeights[coord1] += weight1;
            }
        }

        for (int i = 0; i < recDim; i++)
            recData[i] /= Math.Max(1e-6f, recWeights[i]); // Avoid division by zero

        traceImage.Dispose();
        distanceMap.Dispose();

        return recData;
    }

    public static SplinePath2D RefineSplineControlPoints(Image lowpassImage, SplinePath2D spline, int iterations)
    {
        if (spline == null || spline.Points == null || spline.Points.Count < 3)
            throw new ArgumentException("Spline must have at least three control points for refinement.");

        float2[] controlPoints = spline.Points.ToArray();
        float2[] normals = spline.GetControlPointNormals();

        int maxDistance = 60;
        float softEdge = 30;

        double[] optimizationInput = new double[controlPoints.Length];
        double[] optimizationFallback = optimizationInput.ToArray();

        Func<double[], double> Eval = (input) =>
        {
            float2[] newControlPoints = new float2[controlPoints.Length];
            for (int i = 0; i < controlPoints.Length; i++)
                newControlPoints[i] = controlPoints[i] + normals[i] * (float)input[i];

            if (spline.IsClosed)
                newControlPoints[newControlPoints.Length - 1] = newControlPoints[0];

            SplinePath2D newSpline = new SplinePath2D(newControlPoints, spline.IsClosed);
            List<float2> points = newSpline.GetInterpolated(
                Helper.ArrayOfFunction(i => (float)i / (controlPoints.Length - 1), controlPoints.Length)
            ).ToList();

            float[] imageData = lowpassImage.GetHost(Intent.Read)[0];
            float2 imageDims = new float2(lowpassImage.Dims.X, lowpassImage.Dims.Y);

            double error = 0;
            foreach (var point in points)
            {
                int x = Math.Clamp((int)MathF.Round(point.X), 0, (int)imageDims.X - 1);
                int y = Math.Clamp((int)MathF.Round(point.Y), 0, (int)imageDims.Y - 1);
                float value = imageData[y * lowpassImage.Dims.X + x];

                error += value * value;
            }

            return Math.Sqrt(error / points.Count);
        };

        Func<double[], double[]> Grad = (input) =>
        {
            double[] gradient = new double[input.Length];
            for (int i = 0; i < input.Length - (spline.IsClosed ? 1 : 0); i++)
            {
                double[] inputPlus = input.ToArray();
                inputPlus[i] += 1e-3;
                double[] inputMinus = input.ToArray();
                inputMinus[i] -= 1e-3;

                gradient[i] = (Eval(inputPlus) - Eval(inputMinus)) / 2e-3;
            }

            return gradient;
        };

        for (int iter = 0; iter < iterations; iter++)
        {
            try
            {
                BroydenFletcherGoldfarbShanno optimizer = new BroydenFletcherGoldfarbShanno(optimizationInput.Length, Eval, Grad);
                optimizer.MaxIterations = 10;
                optimizer.MaxLineSearch = 5;
                optimizer.Minimize(optimizationInput);
                optimizationFallback = optimizationInput.ToArray();
            }
            catch(Exception e)
            {
                Console.WriteLine($"Optimization failed at iteration {iter}: {e.Message}");
                optimizationInput = optimizationFallback.ToArray();
                Eval(optimizationInput);
            }
        }

        for (int i = 0; i < controlPoints.Length; i++)
            controlPoints[i] += normals[i] * (float)optimizationInput[i];

        if (spline.IsClosed)
            controlPoints[controlPoints.Length - 1] = controlPoints[0];

        return new SplinePath2D(controlPoints, spline.IsClosed);
    }


    public static void SaveControlPoints(string path, List<SplinePath2D> splines)
    {
        using (StreamWriter writer = new StreamWriter(path))
        {
            foreach (var (spline, index) in splines.Select((s, i) => (s, i)))
            {
                writer.WriteLine($"data_path{index:D3}");
                writer.WriteLine();
                writer.WriteLine("loop_");
                writer.WriteLine("_wrpControlPointXAngst");
                writer.WriteLine("_wrpControlPointYAngst");

                foreach (var point in spline.Points)
                {
                    writer.WriteLine($"{point.X.ToString(CultureInfo.InvariantCulture)} {point.Y.ToString(CultureInfo.InvariantCulture)}");
                }

                writer.WriteLine();
            }
        }
    }
}