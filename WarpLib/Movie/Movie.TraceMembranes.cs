using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
                // Validate file inputs
                if (!File.Exists(AveragePath) || !File.Exists(MembraneSegmentationPath))
                    throw new FileNotFoundException("Required input files not found");

                // Load image and mask
                Image average = Image.FromFile(AveragePath);
                Image maskRaw = Image.FromFile(MembraneSegmentationPath);
                toDispose.Add(maskRaw);

                float angPixMic = average.PixelSize;
                float angPixMask = maskRaw.PixelSize;

                #region preprocessing

                Console.WriteLine("preprocessing...");
                // early exit if no membranes found
                var components = maskRaw.GetConnectedComponents().ToArray();
                if (components.Length == 0)
                    throw new Exception("No membranes found");

                // Mean subtraction
                average.SubtractMeanGrid(new int2(1));

                // 2x padding
                average = average.AsPadded(average.DimsSlice * 2).AndDisposeParent();
                toDispose.Add(average);

                // filtering
                average.Bandpass(
                    nyquistLow: angPixMic * 2 / 300f, // 1/300 Å
                    nyquistHigh: 1f, // nyquist
                    nyquistsoftedge: angPixMic * 2 / 600f, // 1/600 Å
                    isVolume: false
                );


                Image averageLowpass20 = average.GetCopyGPU();
                averageLowpass20.Bandpass(
                    nyquistLow: angPixMic * 2 / 300f, // 1/300 Å
                    nyquistHigh: angPixMic * 2 / 20f, // 1/20 Å
                    nyquistsoftedge: angPixMic * 2 / 600f, // 1/600 Å
                    isVolume: false
                );

                Image averageLowpass50 = average.GetCopyGPU();
                averageLowpass50.Bandpass(
                    nyquistLow: angPixMic * 2 / 300f, // 1/300 Å
                    nyquistHigh: angPixMic * 2 / 50f, // 1/50 Å
                    nyquistsoftedge: angPixMic * 2 / 600f, // 1/600 Å
                    isVolume: true
                );

                // Reducing back to original size and add to list for disposal
                averageLowpass20 = averageLowpass20.AsPadded(averageLowpass20.DimsSlice / 2).AndDisposeParent();
                averageLowpass50 = averageLowpass50.AsPadded(averageLowpass50.DimsSlice / 2).AndDisposeParent();
                toDispose.Add(averageLowpass20);
                toDispose.Add(averageLowpass50);
                
                Console.WriteLine("Finished preprocessing");

                #endregion

                // trace 1px thick ridges through each membrane, prune any extra branches in the resulting skeleton
                var skeleton = TraceMembranesHelper.Skeletonize(maskRaw);
                toDispose.Add(skeleton);
                TraceMembranesHelper.PruneBranchesInPlace(skeleton);
                Console.WriteLine("Skeletonized input image and pruned short branches");

                // find each individual 1px thick membrane in preprocessed skeleton
                components = skeleton.GetConnectedComponents();
                Console.WriteLine($"Found {components.Length} connected components");
                components = components
                    .Where(c => c.ComponentIndices.Length >= options.MinimumComponentPixels)
                    .ToArray();
                Console.WriteLine($"{components.Length} membranes left after filtering (component must have at least {options.MinimumComponentPixels} pixels)");

                // create output dir
                Directory.CreateDirectory(MembraneModelsDir);

                // refine each membrane one by one
                for (int ic = 0; ic < components.Length; ic++)
                {
                    Console.WriteLine($"Refining membrane {ic + 1} of {components.Length}");

                    // Find pixel indices for this membrane
                    List<int> componentIndices = new List<int>(components[ic].ComponentIndices);
                    // Fit control points of a 2D spline path to these pixel positions
                    SplinePath2D initialPath = TraceMembranesHelper.FitInitialSpline(
                        componentIndices: componentIndices,
                        skeleton: skeleton,
                        pixelSizeMask: angPixMask,
                        pixelSizeAverage: angPixMic,
                        controlPointSpacingAngst: (float)options.SplinePointSpacing
                    );
                    Console.WriteLine("Initial path fit to skeletonized membrane");

                    // Optimize control points and intensities based on image data, first at low res
                    Console.WriteLine("Modelling membrane at low resolution (low pass 50Å)");
                    var (profile1D, refinedPath, intensitySpline) = TraceMembranesHelper.RefineMembrane(
                        initialSpline: initialPath,
                        image: averageLowpass50,
                        maxDistanceAngst: 100f,
                        softEdgeWidthAngst: 30f,
                        refinementIterations: 1
                    );

                    // then at higher res
                    Console.WriteLine("Modelling membrane at higher resolution (low pass 20Å)");
                    (profile1D, refinedPath, intensitySpline) = TraceMembranesHelper.RefineMembrane(
                        initialSpline: refinedPath,
                        image: averageLowpass20,
                        maxDistanceAngst: 60f,
                        softEdgeWidthAngst: 30f,
                        refinementIterations: options.RefinementIterations
                    );

                    #region write output file
                    
                    // Create a table with path control point coordinates
                    var controlPointsTable = new Star(
                        values: refinedPath.Points.ToArray(),
                        nameColumn1: "wrpPathControlPointXAngst",
                        nameColumn2: "wrpPathControlPointYAngst"
                    );

                    // Create a table containing membrane intensity spline control points information
                    var intensitySplineTable = new Star(
                        values: intensitySpline.Points.ToArray(),
                        nameColumn1: "wrpIntensitySplineControlPoints"
                    );

                    // create a table containing 1d profile data
                    var profileTable = new Star(
                        values: profile1D,
                        nameColumn1: "wrpMembraneProfile"
                    );

                    // add tables to output
                    var outputTables = new Dictionary<string, Star>();
                    outputTables.Add(key: "profile1d", value: profileTable);
                    outputTables.Add(key: "path", value: controlPointsTable);
                    outputTables.Add(key: "intensity", value: intensitySplineTable);

                    // write output file
                    string outputFile = IOPath.Combine(MembraneModelsDir, $"{RootName}_membrane{ic:D3}.star");
                    Star.SaveMultitable(outputFile, outputTables);
                    Console.WriteLine($"Membrane model saved to {outputFile}");

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
    [WarpSerializable] public decimal SplinePointSpacing { get; set; } = 200; // angstroms
    [WarpSerializable] public int RefinementIterations { get; set; } = 2;
    [WarpSerializable] public int MinimumComponentPixels { get; set; } = 20;
}


public static class TraceMembranesHelper
{
    public static (float[] profile1D, SplinePath2D path, SplinePath1D intensitySpline) RefineMembrane(
        SplinePath2D initialSpline,
        Image image,
        float maxDistanceAngst,
        float softEdgeWidthAngst,
        int refinementIterations)
    {
        // get image data
        float[] imageData = image.GetHost(Intent.Read)[0];

        // convert distances from angstroms to pixels
        int maxDistance = (int)(maxDistanceAngst / image.PixelSize);
        float softEdgeWidth = softEdgeWidthAngst / image.PixelSize;

        // Get control points and normals
        float2[] controlPoints = initialSpline.Points.ToArray();
        float2[] normals = initialSpline.GetControlPointNormals();

        // sample points along spline
        int nPoints = (int)initialSpline.EstimatedLength;
        float[] tPoints = Helper.ArrayOfFunction(i => (float)i / nPoints, n: nPoints);
        float2[] splinePoints = initialSpline.GetInterpolated(tPoints);

        // find pixels up to maximum distance away from membrane
        int2[] membranePixels = FindMembranePixels(
            path: initialSpline, imageDims: image.Dims, maxDistancePx: maxDistance
        );
        int nMembranePixels = membranePixels.Length;
        Console.WriteLine($"{nMembranePixels}px in membrane");

        // allocate arrays for cached per-pixel data
        float[] membraneRefVals = new float[nMembranePixels];
        int[] membraneClosestPointIdx = new int[nMembranePixels];
        float[] membraneSegmentLengths = new float[nMembranePixels];
        float2[] membraneTangents = new float2[nMembranePixels];
        float2[] membraneNormals = new float2[nMembranePixels];
        float[] membraneWeights = new float[nMembranePixels];
        bool[] excludePixel = new bool[nMembranePixels];
        
        Console.WriteLine("populating cache with per-membrane-pixel data used in optimization...");
        // cache a bunch of data per-pixel in membrane
        for (int p = 0; p < nMembranePixels; p++)
        {
            // grab pixel coords and linear index
            int2 iPixel = membranePixels[p];
            int i = iPixel.Y * image.Dims.X + iPixel.X;

            // calculate per pixel data
            float2 pixel = new float2(iPixel);
            var (idx0, idx1) = FindClosestLineSegment(queryPoint: pixel, pathPoints: splinePoints);
            var (p0, p1) = (splinePoints[idx0], splinePoints[idx1]);
            float length = (p1 - p0).Length();
            float2 tangent = (p1 - p0).Normalized();
            float2 normal = new float2(tangent.Y, -tangent.X);
            float signedDistanceFromMembrane = GetSignedDistanceFromMembrane(
                pixelLocation: pixel,
                closestPointOnSpline: p0,
                tangent: tangent,
                normal: normal,
                segmentLength: length
            );
            float weight = CalculateWeight(
                distanceFromSpline: MathF.Abs(signedDistanceFromMembrane),
                maxDistance: maxDistance,
                softEdgeWidth: softEdgeWidth
            );
            float projectedLength = float2.Dot(pixel - p0, tangent);

            membraneRefVals[p] = imageData[i];
            membraneClosestPointIdx[p] = idx0;
            membraneSegmentLengths[p] = length;
            membraneTangents[p] = tangent;
            membraneNormals[p] = normal;
            membraneWeights[p] = weight;
            excludePixel[p] = projectedLength < 0 || projectedLength > length;
        }
        Console.WriteLine("cache populated");


        // Create intensity spline (initially flat)
        SplinePath1D intensitySpline = new SplinePath1D(
            points: Helper.ArrayOfConstant(1f, initialSpline.IsClosed ? 3 : 4),
            isClosed: initialSpline.IsClosed
        );

        // Initialize optimization parameters
        double[] optimizationInput = new double[controlPoints.Length + intensitySpline.Points.Count];
        double[] optimizationFallback = optimizationInput.ToArray();
        int recDim = maxDistance * 2;

        // Pre-calculate 1D profile once per iteration
        float[] recData = new float[recDim];
        float[] recWeights = new float[recDim];
        
        // Perform refinement iterations
        for (int iter = 0; iter < refinementIterations; iter++)
        {
            recData = new float[recDim];
            recWeights = new float[recDim];
            
            for (int p = 0; p < membranePixels.Length; p++)
            {
                // get image pixel position and linear index
                int2 iPixel = membranePixels[p];
                float2 pixel = new float2(iPixel);
                int i = iPixel.Y * image.Dims.X + iPixel.X;

                // early exit if pixel not within line segment
                // temporarily switched off due to artifacts
                // if (excludePixel[p])
                //     continue;

                // get image value for current pixel
                float val = imageData[i];

                // get coord in 1D reconstruction
                float coord = GetSignedDistanceFromMembrane(
                    pixelLocation: pixel,
                    closestPointOnSpline: splinePoints[membraneClosestPointIdx[p]],
                    tangent: membraneTangents[p],
                    normal: membraneNormals[p],
                    segmentLength: membraneSegmentLengths[p]
                );
                coord += maxDistance;
                coord = Math.Clamp(coord, 0, recDim - 1);

                // reconstruct 1d profile with linear interpolation
                int floor = (int)coord;
                int ceil = Math.Min(recDim - 1, floor + 1);
                float weight1 = (coord - floor);
                float weight0 = 1 - weight1;

                recData[floor] += val * weight0;
                recData[ceil] += val * weight1;
                recWeights[floor] += weight0;
                recWeights[ceil] += weight1;
            }

            for (int i = 0; i < recDim; i++)
                recData[i] /= Math.Max(1e-16f, recWeights[i]);

            // Define optimization function
            Func<double[], double> eval = (input) =>
            {
                // Update spline control points
                float2[] newControlPoints = new float2[controlPoints.Length];
                for (int i = 0; i < controlPoints.Length; i++)
                    newControlPoints[i] = controlPoints[i] + normals[i] * (float)input[i];

                if (initialSpline.IsClosed)
                    newControlPoints[newControlPoints.Length - 1] = newControlPoints[0];

                // Recreate spline with new control points
                SplinePath2D spline = new SplinePath2D(newControlPoints, initialSpline.IsClosed);
                splinePoints = spline.GetInterpolated(tPoints);

                // Update intensity values
                float[] newIntensityControlPoints = new float[intensitySpline.Points.Count];
                for (int i = 0; i < newIntensityControlPoints.Length; i++)
                    newIntensityControlPoints[i] = MathF.Exp((float)input[newControlPoints.Length + i]);

                if (initialSpline.IsClosed)
                    newIntensityControlPoints[newIntensityControlPoints.Length - 1] = newIntensityControlPoints[0];

                // Recreate intensity spline
                SplinePath1D newIntensitySpline = new SplinePath1D(newIntensityControlPoints, initialSpline.IsClosed);
                List<float> scaleFactors = newIntensitySpline.GetInterpolated(tPoints).ToList();

                // Calculate error
                double result = 0;
                for (int i = 0; i < membranePixels.Length; i++)
                {
                    float coord = GetSignedDistanceFromMembrane(
                        new float2(membranePixels[i]),
                        splinePoints[membraneClosestPointIdx[i]],
                        membraneTangents[i],
                        membraneNormals[i],
                        membraneSegmentLengths[i]
                    );
                    coord += maxDistance;
                    coord = Math.Clamp(coord, 0, recDim - 1);
                    int coord0 = (int)coord;
                    int coord1 = Math.Min(recDim - 1, coord0 + 1);
                    float weight1 = (coord - coord0);
                    float weight0 = 1 - weight1;

                    float val = recData[coord0] * weight0 + recData[coord1] * weight1;
                    float intensity = scaleFactors[membraneClosestPointIdx[i]];
                    val *= intensity;

                    result += (membraneRefVals[i] - val) * (membraneRefVals[i] - val) * membraneWeights[i];
                }

                return Math.Sqrt(result / membranePixels.Length);
            };

            int gradIteration = 0;

            // Calculate gradient for optimization
            Func<double[], double[]> grad = (input) =>
            {
                double[] result = new double[input.Length];

                for (int i = 0; i < input.Length - (initialSpline.IsClosed ? 1 : 0); i++)
                {
                    double[] inputPlus = input.ToArray();
                    inputPlus[i] += 1e-3;
                    double[] inputMinus = input.ToArray();
                    inputMinus[i] -= 1e-3;

                    result[i] = (eval(inputPlus) - eval(inputMinus)) / 2e-3;
                }

                Console.WriteLine(eval(input));

                return result;
            };

            // Perform optimization
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
                Console.WriteLine($"Optimization iteration {iter + 1} failed: {e.Message}");
                optimizationInput = optimizationFallback.ToArray();
                eval(optimizationInput);
            }
        }

        // Update the control points with the optimized values
        for (int i = 0; i < controlPoints.Length; i++)
            controlPoints[i] += normals[i] * (float)optimizationInput[i];

        if (initialSpline.IsClosed)
            controlPoints[controlPoints.Length - 1] = controlPoints[0];

        // Apply spline updates
        SplinePath2D finalSpline = new SplinePath2D(controlPoints, initialSpline.IsClosed);

        // Get final intensity values
        float[] finalIntensityControls = new float[intensitySpline.Points.Count];
        for (int i = 0; i < finalIntensityControls.Length; i++)
            finalIntensityControls[i] = MathF.Exp((float)optimizationInput[controlPoints.Length + i]);

        if (initialSpline.IsClosed)
            finalIntensityControls[finalIntensityControls.Length - 1] = finalIntensityControls[0];

        SplinePath1D finalIntensitySpline = new SplinePath1D(finalIntensityControls, initialSpline.IsClosed);
        
        return (recData, finalSpline, finalIntensitySpline);
    }

    public static Image Skeletonize(Image image)
    {
        // Create a copy to avoid modifying the original
        Image skeleton = new Image(image.Dims);

        // Get data from the images
        float[] imageData = image.GetHost(Intent.Read)[0];
        float[] skeletonData = skeleton.GetHost(Intent.ReadWrite)[0];

        int width = image.Dims.X;
        int height = image.Dims.Y;

        // Initialize skeleton with binary input image (0 for background, 1 for foreground)
        for (int i = 0; i < imageData.Length; i++)
        {
            skeletonData[i] = imageData[i] > 0 ? 1.0f : 0.0f;
        }

        // Array to track points marked for deletion
        bool[] pointsToDelete = new bool[imageData.Length];

        // Initialize lookup tables for Zhang algorithm
        // These contain the precalculated deletion decisions based on neighborhood patterns
        bool[] G123_LUT = InitializeG123LUT();
        bool[] G123P_LUT = InitializeG123PLUT();

        // Continue until no more changes
        bool hasChanged;
        do
        {
            hasChanged = false;

            // Reset points to delete
            Array.Clear(pointsToDelete, 0, pointsToDelete.Length);

            // First subiteration with G123_LUT
            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    int i = y * width + x;

                    // Skip background pixels
                    if (skeletonData[i] == 0) continue;

                    // Get 8-connected neighborhood
                    int p1 = skeletonData[i - 1] > 0 ? 1 : 0; // Left
                    int p2 = skeletonData[i - 1 - width] > 0 ? 1 : 0; // Top-left
                    int p3 = skeletonData[i - width] > 0 ? 1 : 0; // Top
                    int p4 = skeletonData[i + 1 - width] > 0 ? 1 : 0; // Top-right
                    int p5 = skeletonData[i + 1] > 0 ? 1 : 0; // Right
                    int p6 = skeletonData[i + 1 + width] > 0 ? 1 : 0; // Bottom-right
                    int p7 = skeletonData[i + width] > 0 ? 1 : 0; // Bottom
                    int p8 = skeletonData[i - 1 + width] > 0 ? 1 : 0; // Bottom-left

                    // Calculate lookup index by combining all 8 neighbors
                    // The format matches Zhang's paper: P2 P3 P4 P1 P0 P5 P8 P7 P6
                    // where P0 is assumed to be 1 (center pixel)
                    int index = (p2 << 0) | (p3 << 1) | (p4 << 2) |
                                (p1 << 3) | (1 << 4) | (p5 << 5) |
                                (p8 << 6) | (p7 << 7) | (p6 << 8);

                    // Determine if this pixel should be deleted based on LUT
                    if (G123_LUT[index])
                    {
                        pointsToDelete[i] = true;
                        hasChanged = true;
                    }
                }
            }

            // Apply first subiteration deletion
            for (int i = 0; i < skeletonData.Length; i++)
            {
                if (pointsToDelete[i])
                {
                    skeletonData[i] = 0;
                }
            }

            // Reset points to delete for second subiteration
            Array.Clear(pointsToDelete, 0, pointsToDelete.Length);

            // Second subiteration with G123P_LUT
            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    int i = y * width + x;

                    // Skip background pixels
                    if (skeletonData[i] == 0) continue;

                    // Get 8-connected neighborhood
                    int p1 = skeletonData[i - 1] > 0 ? 1 : 0; // Left
                    int p2 = skeletonData[i - 1 - width] > 0 ? 1 : 0; // Top-left
                    int p3 = skeletonData[i - width] > 0 ? 1 : 0; // Top
                    int p4 = skeletonData[i + 1 - width] > 0 ? 1 : 0; // Top-right
                    int p5 = skeletonData[i + 1] > 0 ? 1 : 0; // Right
                    int p6 = skeletonData[i + 1 + width] > 0 ? 1 : 0; // Bottom-right
                    int p7 = skeletonData[i + width] > 0 ? 1 : 0; // Bottom
                    int p8 = skeletonData[i - 1 + width] > 0 ? 1 : 0; // Bottom-left

                    // Calculate lookup index
                    int index = (p2 << 0) | (p3 << 1) | (p4 << 2) |
                                (p1 << 3) | (1 << 4) | (p5 << 5) |
                                (p8 << 6) | (p7 << 7) | (p6 << 8);

                    // Determine if this pixel should be deleted based on LUT
                    if (G123P_LUT[index])
                    {
                        pointsToDelete[i] = true;
                        hasChanged = true;
                    }
                }
            }

            // Apply second subiteration deletion
            for (int i = 0; i < skeletonData.Length; i++)
            {
                if (pointsToDelete[i])
                {
                    skeletonData[i] = 0;
                }
            }
        } while(hasChanged);

        // Change 45deg stairs from pattern "0 1, 1 1" to "0 1, 1 0"
        for (int i = width + 1; i < skeletonData.Length - width - 1; i++)
        {
            if (skeletonData[i] != 1)
                continue;

            if (skeletonData[i + 1] == 1 && skeletonData[i + width] == 1)
            {
                skeletonData[i] = 0;
            }
            else if (skeletonData[i + 1] == 1 && skeletonData[i - width] == 1)
            {
                skeletonData[i] = 0;
            }
            else if (skeletonData[i - 1] == 1 && skeletonData[i + width] == 1)
            {
                skeletonData[i] = 0;
            }
            else if (skeletonData[i - 1] == 1 && skeletonData[i - width] == 1)
            {
                skeletonData[i] = 0;
            }
        }

        // set values to 0 at image borders
        // Top and bottom rows
        for (int x = 0; x < width; x++)
        {
            // Top row
            skeletonData[x] = 0;

            // Bottom row
            skeletonData[(height - 1) * width + x] = 0;
        }

        // Left and right columns (excluding the corners which were already handled)
        for (int y = 1; y < height - 1; y++)
        {
            // Left column
            skeletonData[y * width] = 0;

            // Right column
            skeletonData[y * width + (width - 1)] = 0;
        }


        return skeleton;
    }

    public static SplinePath2D FitInitialSpline(
        List<int> componentIndices,
        Image skeleton,
        float pixelSizeMask,
        float pixelSizeAverage,
        float controlPointSpacingAngst)
    {
        // Calculate control point spacing in pixels
        float controlPointSpacing = controlPointSpacingAngst / pixelSizeMask;

        // Fit spline to component pixels
        SplinePath2D spline = ComponentToSpline(componentIndices, skeleton, controlPointSpacing);

        // Rescale control points to unbinned image pixel space
        var rescaledControlPoints = spline.Points.ToArray();
        for (int i = 0; i < rescaledControlPoints.Length; i++)
            rescaledControlPoints[i] *= (pixelSizeMask / pixelSizeAverage);

        // Create new spline with rescaled control points
        return new SplinePath2D(rescaledControlPoints, spline.IsClosed);
    }

    public static Image RasterizeSpline(SplinePath2D spline, int3 imageDims)
    {
        // Interpolate points along the path for precise membrane reconstruction
        float estimatedLength = spline.EstimatedLength;
        float[] t = Helper.ArrayOfFunction(i => (float)i / (int)estimatedLength, n: (int)estimatedLength);
        List<float2> points = spline.GetInterpolated(t).ToList();

        // Create working image for tracing
        Image RasterizedPath = new Image(imageDims);
        float[] TraceData = RasterizedPath.GetHost(Intent.ReadWrite)[0];

        // Rasterize the spline path for distance calculation
        for (int ip = 0; ip < points.Count - 1; ip++)
        {
            float2 P0 = points[ip];
            float2 P1 = points[ip + 1];

            int x0 = (int)MathF.Round(P0.X);
            int y0 = (int)MathF.Round(P0.Y);
            int x1 = (int)MathF.Round(P1.X);
            int y1 = (int)MathF.Round(P1.Y);

            if (x0 < 0 || x0 >= RasterizedPath.Dims.X || y0 < 0 || y0 >= RasterizedPath.Dims.Y)
                continue;
            if (x1 < 0 || x1 >= RasterizedPath.Dims.X || y1 < 0 || y1 >= RasterizedPath.Dims.Y)
                continue;

            // Bresenham's line algorithm
            int dx = Math.Abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
            int dy = -Math.Abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
            int err = dx + dy, e2;

            while(true)
            {
                TraceData[y0 * RasterizedPath.Dims.X + x0] = 1;
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

        return RasterizedPath;
    }

    // coordinate system is membrane is 0
    public static float GetSignedDistanceFromMembrane(
        float2 pixelLocation,
        float2 closestPointOnSpline,
        float2 tangent,
        float2 normal,
        float segmentLength)
    {
        // update closest point to be the closest point on the line segment
        float2 delta = pixelLocation - closestPointOnSpline;
        float lineCoord = Math.Clamp(float2.Dot(delta, tangent), 0, segmentLength);
        closestPointOnSpline += (tangent * lineCoord);

        // Calculate signed 1D membrane coordinate, 0 means on the membrane
        delta = pixelLocation - closestPointOnSpline;
        float dist1D = delta.Length();
        int sign = MathF.Sign(float2.Dot(delta, normal));
        return sign * dist1D;
    }

    public static Image SplineToCoarseDistanceMap(SplinePath2D spline, int3 imageDims, int maxDistance)
    {
        Image RasterizedPath = RasterizeSpline(spline, imageDims: imageDims);
        Image DistanceMap = RasterizedPath
            .AsDistanceMapExact(maxDistance: maxDistance, isVolume: false)
            .AndDisposeParent();
        return DistanceMap;
    }

    private static bool[] InitializeG123LUT()
    {
        bool[] lut = new bool[512];

        // Here we'll implement the specific lookup table from Zhang's algorithm
        // The values are based on three conditions:
        // 1. Number of nonzero neighbors between 2 and 6
        // 2. Exactly one 0-1 transition in the ordered sequence p1,p2,...,p8,p1
        // 3. At least one of p1, p3, p5 is 0
        // 4. At least one of p3, p5, p7 is 0

        for (int i = 0; i < 512; i++)
        {
            // Skip if center pixel is not set (this should never happen in our loop above)
            if ((i & (1 << 4)) == 0)
                continue;

            // Extract individual neighbor values using bitmasks
            int p1 = (i >> 3) & 1;
            int p2 = (i >> 0) & 1;
            int p3 = (i >> 1) & 1;
            int p4 = (i >> 2) & 1;
            int p5 = (i >> 5) & 1;
            int p6 = (i >> 8) & 1;
            int p7 = (i >> 7) & 1;
            int p8 = (i >> 6) & 1;

            // Condition 1: Count non-zero neighbors
            int neighborCount = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
            if (neighborCount < 2 || neighborCount > 6)
                continue;

            // Condition 2: Count 0-1 transitions in sequence p1,p2,...,p8,p1
            int transitions = 0;
            transitions += (p1 == 0 && p2 == 1) ? 1 : 0;
            transitions += (p2 == 0 && p3 == 1) ? 1 : 0;
            transitions += (p3 == 0 && p4 == 1) ? 1 : 0;
            transitions += (p4 == 0 && p5 == 1) ? 1 : 0;
            transitions += (p5 == 0 && p6 == 1) ? 1 : 0;
            transitions += (p6 == 0 && p7 == 1) ? 1 : 0;
            transitions += (p7 == 0 && p8 == 1) ? 1 : 0;
            transitions += (p8 == 0 && p1 == 1) ? 1 : 0;

            if (transitions != 1)
                continue;

            // Conditions 3 & 4: For first subiteration
            if (p1 == 0 || p3 == 0 || p5 == 0)
                if (p3 == 0 || p5 == 0 || p7 == 0)
                    lut[i] = true;
        }

        return lut;
    }

    private static bool[] InitializeG123PLUT()
    {
        bool[] lut = new bool[512];

        // Similar to first LUT but with different conditions 3 and 4:
        // 3. At least one of p1, p3, p7 is 0
        // 4. At least one of p1, p5, p7 is 0

        for (int i = 0; i < 512; i++)
        {
            // Skip if center pixel is not set
            if ((i & (1 << 4)) == 0)
                continue;

            // Extract individual neighbor values
            int p1 = (i >> 3) & 1;
            int p2 = (i >> 0) & 1;
            int p3 = (i >> 1) & 1;
            int p4 = (i >> 2) & 1;
            int p5 = (i >> 5) & 1;
            int p6 = (i >> 8) & 1;
            int p7 = (i >> 7) & 1;
            int p8 = (i >> 6) & 1;

            // Condition 1: Count non-zero neighbors
            int neighborCount = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
            if (neighborCount < 2 || neighborCount > 6)
                continue;

            // Condition 2: Count 0-1 transitions
            int transitions = 0;
            transitions += (p1 == 0 && p2 == 1) ? 1 : 0;
            transitions += (p2 == 0 && p3 == 1) ? 1 : 0;
            transitions += (p3 == 0 && p4 == 1) ? 1 : 0;
            transitions += (p4 == 0 && p5 == 1) ? 1 : 0;
            transitions += (p5 == 0 && p6 == 1) ? 1 : 0;
            transitions += (p6 == 0 && p7 == 1) ? 1 : 0;
            transitions += (p7 == 0 && p8 == 1) ? 1 : 0;
            transitions += (p8 == 0 && p1 == 1) ? 1 : 0;

            if (transitions != 1)
                continue;

            // Conditions 3 & 4: For second subiteration
            if (p1 == 0 || p3 == 0 || p7 == 0)
                if (p1 == 0 || p5 == 0 || p7 == 0)
                    lut[i] = true;
        }

        return lut;
    }

    public static void PruneBranchesInPlace(Image skeleton)
    {
        // Get data from the skeleton image
        float[] skeletonData = skeleton.GetHost(Intent.ReadWrite)[0];

        int width = skeleton.Dims.X;
        int height = skeleton.Dims.Y;

        // Find all junction points and endpoints
        List<int> junctions = new List<int>();
        List<int> endpoints = new List<int>();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;

                // Skip background pixels
                if (skeletonData[i] == 0) continue;

                // Get 8-connected neighborhood
                int[] neighbors = new int[8];
                neighbors[0] = skeletonData[i - 1] > 0 ? 1 : 0; // Left
                neighbors[1] = skeletonData[i - 1 - width] > 0 ? 1 : 0; // Top-left
                neighbors[2] = skeletonData[i - width] > 0 ? 1 : 0; // Top
                neighbors[3] = skeletonData[i + 1 - width] > 0 ? 1 : 0; // Top-right
                neighbors[4] = skeletonData[i + 1] > 0 ? 1 : 0; // Right
                neighbors[5] = skeletonData[i + 1 + width] > 0 ? 1 : 0; // Bottom-right
                neighbors[6] = skeletonData[i + width] > 0 ? 1 : 0; // Bottom
                neighbors[7] = skeletonData[i - 1 + width] > 0 ? 1 : 0; // Bottom-left

                // Count neighbors
                int neighborCount = 0;
                for (int n = 0; n < 8; n++)
                {
                    neighborCount += neighbors[n];
                }

                // Junction points have more than 2 neighbors
                if (neighborCount > 2)
                {
                    junctions.Add(i);
                }
                // Endpoints have exactly 1 neighbor
                else if (neighborCount == 1)
                {
                    endpoints.Add(i);
                }
            }
        }

        // Process each junction to keep only the two longest branches
        foreach (int junction in junctions)
        {
            // Get all branches starting from this junction
            List<List<int>> branches = FindBranchesFromJunction(skeletonData, junction, width);

            // Sort branches by length (longest first)
            branches.Sort((a, b) => b.Count.CompareTo(a.Count));

            // Remove all but the two longest branches
            for (int i = 2; i < branches.Count; i++)
            {
                foreach (int pixel in branches[i])
                {
                    // Don't remove the junction point itself
                    if (pixel != junction)
                    {
                        skeletonData[pixel] = 0;
                    }
                }
            }
        }
    }

    private static List<List<int>> FindBranchesFromJunction(float[] imageData, int junctionPixel, int width)
    {
        List<List<int>> branches = new List<List<int>>();

        // Find initial neighbors of the junction
        int[] offsets = new int[]
        {
            -1, // Left
            -width - 1, // TopLeft
            -width, // Top
            -width + 1, // TopRight
            1, // Right
            width + 1, // BottomRight
            width, // Bottom
            width - 1 // BottomLeft
        };

        // Create a temporary copy of the image data to work with
        float[] tempData = new float[imageData.Length];
        Array.Copy(imageData, tempData, imageData.Length);

        // Mark the junction as visited
        tempData[junctionPixel] = 2;

        // Find all initial branch pixels (neighbors of the junction)
        List<int> initialBranchPixels = new List<int>();
        foreach (int offset in offsets)
        {
            int neighborPixel = junctionPixel + offset;

            // No bounds checking needed as guaranteed by caller

            // Check if this is a valid branch pixel
            if (tempData[neighborPixel] == 1)
            {
                initialBranchPixels.Add(neighborPixel);
            }
        }

        // Trace each branch starting from the initial branch pixels
        foreach (int startPixel in initialBranchPixels)
        {
            // Start a new branch with the junction point and initial branch pixel
            List<int> branch = new List<int>();
            branch.Add(junctionPixel);
            branch.Add(startPixel);

            // Mark the initial branch pixel as visited
            tempData[startPixel] = 2;

            int currentPixel = startPixel;
            bool continueTracing = true;

            while(continueTracing)
            {
                // Find the next pixel in the branch
                bool foundNext = false;
                foreach (int offset in offsets)
                {
                    int nextPixel = currentPixel + offset;

                    // No bounds checking needed as guaranteed by caller

                    // Check if this is a valid next pixel (not visited and part of the skeleton)
                    if (tempData[nextPixel] == 1)
                    {
                        branch.Add(nextPixel);
                        tempData[nextPixel] = 2; // Mark as visited
                        currentPixel = nextPixel;
                        foundNext = true;

                        // Check if this is a junction (has more than 2 neighbors)
                        int neighborCount = 0;
                        foreach (int neighborOffset in offsets)
                        {
                            int neighbor = nextPixel + neighborOffset;
                            if (tempData[neighbor] > 0)
                            {
                                neighborCount++;
                            }
                        }

                        // Stop if we've reached another junction or an endpoint
                        if (neighborCount > 2 || neighborCount == 1)
                        {
                            continueTracing = false;
                        }

                        break;
                    }
                }

                // Stop if we can't find a next pixel
                if (!foundNext)
                {
                    continueTracing = false;
                }
            }

            branches.Add(branch);
        }

        return branches;
    }

    public static List<int> FindEndpoints(Image skeleton, List<int> componentIndices)
    {
        List<int> endpoints = new List<int>();
        float[] skeletonData = skeleton.GetHost(Intent.ReadWrite)[0];
        int width = skeleton.Dims.X;

        foreach (int idx in componentIndices)
        {
            // Explicitly calculate all 8 neighbor indices
            int[] neighborIndices = new int[8];
            neighborIndices[0] = idx - width - 1; // Top-left
            neighborIndices[1] = idx - width; // Top
            neighborIndices[2] = idx - width + 1; // Top-right
            neighborIndices[3] = idx - 1; // Left
            neighborIndices[4] = idx + 1; // Right
            neighborIndices[5] = idx + width - 1; // Bottom-left
            neighborIndices[6] = idx + width; // Bottom
            neighborIndices[7] = idx + width + 1; // Bottom-right

            // Count neighboring skeleton pixels
            int neighborCount = 0;
            for (int i = 0; i < 8; i++)
            {
                int idxLinear = neighborIndices[i];
                if (idxLinear < 0 || idxLinear > skeletonData.Length)
                    continue;
                if (skeletonData[neighborIndices[i]] == 1)
                    neighborCount++;
            }

            // Endpoint has only one neighbor
            if (neighborCount == 1)
                endpoints.Add(idx);
        }

        return endpoints;
    }

    public static Image MakeComponentImage(List<int> componentIndices, int3 dims)
    {
        Image image = new Image(dims);
        float[] imageData = image.GetHost(Intent.ReadWrite)[0];

        foreach (int idx in componentIndices)
            imageData[idx] = 1;

        return image;
    }

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

    public static SplinePath2D ComponentToSpline(List<int> componentIndices, Image skeleton, float controlPointSpacing)
    {
        List<int> endpoints = FindEndpoints(skeleton, componentIndices);

        bool pathIsClosed = false;
        if (endpoints.Count == 0)
            pathIsClosed = true;
        else if (endpoints.Count == 2)
            pathIsClosed = false;
        else
            throw new Exception("found more than two endpoints after branch pruning...?");

        // Determine if membrane is a closed loop or has endpoints
        List<int> FinalPath;

        if (pathIsClosed)
        {
            FinalPath = TraceLine(componentIndices, new List<int>(), 0, 1, skeleton.Dims.X);
            FinalPath.Add(FinalPath.First());
        }
        else
        {
            FinalPath = TraceLine(
                componentIndices,
                new List<int>(),
                componentIndices.IndexOf(endpoints[0]),
                1,
                skeleton.Dims.X
            );
        }

        // Convert to 2D coordinates
        List<float2> Points = new List<float2>();
        foreach (int index in FinalPath)
            Points.Add(new float2(index % skeleton.Dims.X, index / skeleton.Dims.X));

        // Sample control points for the spline
        int NControlPoints = (int)MathF.Ceiling(Points.Count / controlPointSpacing) + 1;
        if (pathIsClosed)
            NControlPoints = Math.Max(NControlPoints, 3);

        // Fit spline to the points
        SplinePath2D spline = SplinePath2D.Fit(Points.ToArray(), pathIsClosed, NControlPoints);

        // Ensure consistent orientation (counterclockwise if closed)
        if (pathIsClosed && spline.IsClockwise())
            spline = spline.AsReversed();

        return spline;
    }

    // returns a tuple of indices for the two closest points
    public static (int, int) FindClosestLineSegment(float2 queryPoint, float2[] pathPoints)
    {
        if (pathPoints == null || pathPoints.Length < 2)
            throw new ArgumentException("Path must contain at least 2 points", nameof(pathPoints));

        // Find closest point on the path
        int closestPointIdx = 0;
        float minDist = float.MaxValue;

        for (int j = 0; j < pathPoints.Length; j++)
        {
            float dist = (pathPoints[j] - queryPoint).Length();
            if (dist < minDist)
            {
                minDist = dist;
                closestPointIdx = j;
            }
        }

        // Get next closest point
        int nextIdx = (closestPointIdx + 1) % pathPoints.Length;
        int prevIdx = (closestPointIdx - 1 + pathPoints.Length) % pathPoints.Length;

        float nextDist = (pathPoints[nextIdx] - queryPoint).Length();
        float prevDist = (pathPoints[prevIdx] - queryPoint).Length();

        int nextClosestPointIdx = nextDist < prevDist ? nextIdx : prevIdx;

        // Ensure correct order so tangents have consistent direction
        int id0 = Math.Min(closestPointIdx, nextClosestPointIdx);
        int id1 = Math.Max(closestPointIdx, nextClosestPointIdx);

        return (id0, id1);
    }

    public static int2[] FindMembranePixels(SplinePath2D path, int3 imageDims, int maxDistancePx)
    {
        // list to store set of membrane pixels
        List<int2> membranePixels = new List<int2>();
        
        // find pixels within distance
        using (Image distanceMap = SplineToCoarseDistanceMap(path, imageDims, maxDistancePx))
        {
            float[] distanceMapData = distanceMap.GetHost(Intent.Read)[0];

            for (int i = 0; i < distanceMapData.Length; i++)
            {
                // early exit
                if (distanceMapData[i] >= maxDistancePx)
                    continue;

                // calculate 2d indices into image array
                int x = i % imageDims.X;
                int y = i / imageDims.X;
                membranePixels.Add(new int2(x, y));
            }
        }
        return membranePixels.ToArray();
    }

    public static float CalculateWeight(float distanceFromSpline, float maxDistance, float softEdgeWidth)
    {
        float distanceFromEdge = maxDistance - distanceFromSpline;

        // 1 if outside soft edge regiojn
        if (distanceFromEdge > softEdgeWidth)
            return 1f;

        // cosine falloff over softEdgeWidth
        float fractionalDistanceIntoSoftEdge = (softEdgeWidth - distanceFromEdge) / softEdgeWidth;
        float weight = MathF.Cos(fractionalDistanceIntoSoftEdge * MathF.PI) * 0.5f + 0.5f;
        return weight;
    }
}