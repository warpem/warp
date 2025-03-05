using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
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
                // Validate file inputs
                if (!File.Exists(AveragePath) || !File.Exists(MembraneSegmentationPath))
                    throw new FileNotFoundException("Required input files not found");

                // Load image and mask
                Image average = Image.FromFile(AveragePath);
                Image maskRaw = Image.FromFile(MembraneSegmentationPath);
                // Image mask = maskRaw.AsScaled(new int3(4096, 4096, 1));
                // mask.Binarize(0.5f);
                // mask.WriteMRC16b("ab_maskfull.mrc");

                float angPixMic = average.PixelSize;
                float angPixMask = maskRaw.PixelSize;

                toDispose.Add(average);
                toDispose.Add(maskRaw);

                #region preprocessing

                // // rescale mask to 8A/px
                // float targetAngPixMask = 8;
                // float scaleFactor = angPixMask / targetAngPixMask;
                // int3 dimsMaskScaled = new int3((int)Math.Round(maskRaw.Dims.X * scaleFactor / 2) * 2,
                //     (int)Math.Round(maskRaw.Dims.Y * scaleFactor / 2) * 2,
                //     1);
                // Image MaskRescaled = maskRaw.AsScaled(newDims: dimsMaskScaled);
                // MaskRescaled.Binarize(0.5f);

                // early exit if no membranes found
                var components = maskRaw.GetConnectedComponents().ToArray();
                if (components.Length == 0)
                    throw new Exception("No membranes found");

                float lowPassFrequency = angPixMic * 2 / 300f; // ~1/300 Å
                float highPassFrequencyRaw = 1f; // Nyquist frequency
                float highPassFrequencyLowpass = angPixMic * 2 / 20f; // ~1/20 Å
                float rolloffWidth = angPixMic * 2 / 600f; // ~1/600 Å

                // Mean subtraction
                average.SubtractMeanGrid(new int2(1));

                // 2x padding
                average = average.AsPadded(average.DimsSlice * 2).AndDisposeParent();
                Image averageLowpass = average.GetCopyGPU();

                // Bandpass filtering
                average.Bandpass(lowPassFrequency, highPassFrequencyRaw, false, rolloffWidth);
                averageLowpass.Bandpass(lowPassFrequency, highPassFrequencyLowpass, false, rolloffWidth);

                // Reducing back to original size
                average = average.AsPadded(average.DimsSlice / 2).AndDisposeParent();
                averageLowpass = averageLowpass.AsPadded(averageLowpass.DimsSlice / 2).AndDisposeParent();

                #endregion

                // trace 1px thick ridges through each membrane, prune any extra branches in the resulting skeleton
                var skeleton = TraceMembranesHelper.Skeletonize(maskRaw);
                TraceMembranesHelper.PruneBranchesInPlace(skeleton);
                Image skeletonScaled = skeleton.AsScaled(new int3(4096, 4096, 1));
                skeletonScaled.Binarize(0.5f);
                skeletonScaled.WriteMRC16b("ab_skeleton_pruned.mrc");

                // find each individual 1px thick membrane in preprocessed skeleton
                components = skeleton.GetConnectedComponents()
                    .Where(c => c.ComponentIndices.Length >= options.MinimumComponentPixels)
                    .ToArray();

                // process each membrane one by one
                Dictionary<string, Star> outputTables = new();

                for (int ic = 0; ic < components.Length; ic++)
                {
                    Console.WriteLine($"Refining membrane {ic + 1} of {components.Length}");

                    // Extract pixel indices for this membrane
                    List<int> componentIndices = new List<int>(components[ic].ComponentIndices);

                    // Fit control points of a 2D spline path to these pixel positions
                    SplinePath2D initialPath = TraceMembranesHelper.FitInitialSpline(
                        componentIndices: componentIndices,
                        skeleton: skeleton,
                        pixelSizeMask: angPixMask,
                        pixelSizeAverage: angPixMic,
                        controlPointSpacingAngst: (float)options.SplinePointSpacing
                    );

                    // Optimize control points and intensities based on image data
                    var (optimizedPath, intensitySpline) = TraceMembranesHelper.OptimizeMembrane(
                        initialSpline: initialPath,
                        image: averageLowpass,
                        maxDistanceAngst: 60f, // consider pixels up to this distance from the membrane
                        softEdgeWidthAngst: 30f,
                        refinementIterations: 2
                    );

                    #region Save Results

                    // Create a table with control point coordinates
                    // outputTables.Add($"path{ic:D3}", new Star(
                    //     controlPoints.Select(v => v * angPixMic).ToArray(),
                    //     "wrpControlPointXAngst",
                    //     "wrpControlPointYAngst"));

                    // Store membrane intensity information
                    // PathTables[$"path{ic:D3}"].AddColumn("wrpIntensity",
                    //     FinalIntensityControls.Take(controlPoints.Length).Select(v => v.ToString()).ToArray());

                    // // Store metadata about the membrane
                    // PathTables[$"path{ic:D3}"].SetValueFloat("wrpIsClosed", IsClosed ? 1.0f : 0.0f);
                    // PathTables[$"path{ic:D3}"].SetValueFloat("wrpPixelSize", pixelSize);
                    // PathTables[$"path{ic:D3}"].SetValueFloat("wrpMembraneHalfWidth", (float)options.MembraneHalfWidth);

                    // Subtract this membrane from the original image
                    // for (int i = 0; i < MembranePixels.Count; i++)
                    // {
                    //     float Coord = GetMembraneCoord(i);
                    //     Coord = Math.Clamp(Coord, 0, RecDim - 1);
                    //
                    //     int Coord0 = (int)Coord;
                    //     int Coord1 = Math.Min(RecDim - 1, Coord0 + 1);
                    //     float Weight1 = (Coord - Coord0);
                    //     float Weight0 = 1 - Weight1;
                    //
                    //     float Val = RecData[Coord0] * Weight0 + RecData[Coord1] * Weight1;
                    //     float Intensity = ScaleFactors[MembraneClosestPoints[i]];
                    //     Val *= Intensity;
                    //
                    //     int pixelIndex = MembranePixels[i].Y * ImageRaw.Dims.X + MembranePixels[i].X;
                    //     ImageRaw.GetHost(Intent.ReadWrite)[0][pixelIndex] -= Val * MembraneWeights[i];
                    // }

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
    [WarpSerializable] public decimal SplinePointSpacing { get; set; } = 200; // angstroms
    [WarpSerializable] public int RefinementIterations { get; set; } = 2;
    [WarpSerializable] public int MinimumComponentPixels { get; set; } = 50;
}


public static class TraceMembranesHelper
{
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
        } while (hasChanged);

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
            rescaledControlPoints[i] = rescaledControlPoints[i] * (pixelSizeMask / pixelSizeAverage);

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

            while (true)
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

    public static (SplinePath2D Spline, SplinePath1D IntensitySpline) OptimizeMembrane(
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
        Console.WriteLine($"maxdist px: {maxDistance}");
        float softEdgeWidth = softEdgeWidthAngst / image.PixelSize;

        // Get control points and normals
        float2[] controlPoints = initialSpline.Points.ToArray();
        float2[] normals = initialSpline.GetControlPointNormals();

        // sample points along spline
        int nPoints = (int)initialSpline.EstimatedLength;
        float[] tPoints = Helper.ArrayOfFunction(i => (float)i / nPoints, n: nPoints);
        float2[] points = initialSpline.GetInterpolated(tPoints);

        // find pixels up to maximum distance away from membrane
        int2[] membranePixels = FindMembranePixels(
            spline: initialSpline, imageDims: image.Dims, maxDistancePx: maxDistance
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

        // calculate cached per-pixel data
        for (int p = 0; p < nMembranePixels; p++)
        {
            int2 iPixel = membranePixels[p];
            float2 pixel = new float2(iPixel);
            var (idx0, idx1) = FindClosestLineSegment(queryPoint: pixel, pathPoints: points);
            var (p0, p1) = (points[idx0], points[idx1]);
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
            membraneClosestPointIdx[p] = idx0;
            membraneSegmentLengths[p] = length;
            membraneTangents[p] = tangent;
            membraneNormals[p] = normal;
            membraneWeights[p] = weight;
        }

        // Image weightsImage = new Image(image.Dims);
        // weightsImage.Fill(0f);
        // float[] weightsImageData = weightsImage.GetHost(Intent.ReadWrite)[0];
        // for (int p = 0; p < membranePixels.Length; p++)
        // {
        //     int2 pixel = membranePixels[p];
        //     int i = pixel.Y * image.Dims.X + pixel.X;
        //     weightsImageData[i] = membraneWeights[p];
        // }
        // weightsImage.WriteMRC16b("ab_weights.mrc");
        // Console.WriteLine("ab_weights.mrc");

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

            int2 iPixel;
            float2 pixel;

            for (int p = 0; p < membranePixels.Length; p++)
            {
                // get image value for current pixel
                iPixel = membranePixels[p];
                pixel = new float2(iPixel);
                int i = iPixel.Y * image.Dims.X + iPixel.X;
                float val = imageData[i];

                // get coord in 1D reconstruction
                float coord = GetSignedDistanceFromMembrane(
                    pixelLocation: pixel,
                    closestPointOnSpline: points[membraneClosestPointIdx[p]],
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
            
            // Make a diagnostic image of the 1D profile for comparison with python
            Image profile2d = new Image(new int3(recDim, 200, 1));
            float[] profileData = profile2d.GetHost(Intent.ReadWrite)[0];
            for (int x = 0; x < recDim; x++)
            {
                for (int y = 0; y < 200; y++)
                {
                    profileData[y * recDim + x] = recData[x];
                }
            }
            profile2d.WriteMRC16b("ab_profile_2d.mrc");

            // Make a diagnostic image of the initial reconstruction of this membrane from the 1d profile
            Image initialMembraneImage = new Image(image.Dims);
            initialMembraneImage.Fill(0f);
            float[] initialMembraneImageData = initialMembraneImage.GetHost(Intent.ReadWrite)[0];

            for (int p = 0; p < membranePixels.Length; p++)
            {
                iPixel = membranePixels[p];
                float coord = GetSignedDistanceFromMembrane(
                    pixelLocation: new float2(iPixel),
                    closestPointOnSpline: points[membraneClosestPointIdx[p]],
                    tangent: membraneTangents[p],
                    normal: membraneNormals[p],
                    segmentLength: membraneSegmentLengths[p]
                );
                coord += maxDistance;
                coord = Math.Clamp(coord, 0, recDim - 1);

                int coord0 = (int)coord;
                int coord1 = Math.Min(recDim - 1, coord0 + 1);
                float weight1 = (coord - coord0);
                float weight0 = 1 - weight1;

                float val = recData[coord0] * weight0 + recData[coord1] * weight1;
                int i = iPixel.Y * initialMembraneImage.Dims.X + iPixel.X;
                initialMembraneImageData[i] = val;
            }

            string outputPath = $"ab_initialmembrane.mrc";
            initialMembraneImage.WriteMRC(outputPath);
            Console.WriteLine(outputPath);

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
                points = spline.GetInterpolated(tPoints);

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
                        points[membraneClosestPointIdx[i]],
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

                return result;
            };

            // Perform optimization
            try
            {
                BroydenFletcherGoldfarbShanno optimizer =
                    new BroydenFletcherGoldfarbShanno(optimizationInput.Length, eval, grad);
                optimizer.MaxIterations = 10;
                optimizer.MaxLineSearch = 5;
                optimizer.Minimize(optimizationInput);
                optimizationFallback = optimizationInput.ToArray();
            }
            catch (Exception e)
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
        
        // Make a diagnostic image of the initial reconstruction of this membrane from the 1d profile
        Image finalMembraneImage = new Image(image.Dims);
        finalMembraneImage.Fill(0f);
        float[] finalMembraneImageData = finalMembraneImage.GetHost(Intent.ReadWrite)[0];

        for (int p = 0; p < membranePixels.Length; p++)
        {
            int2 iPixel = membranePixels[p];
            float coord = GetSignedDistanceFromMembrane(
                pixelLocation: new float2(iPixel),
                closestPointOnSpline: points[membraneClosestPointIdx[p]],
                tangent: membraneTangents[p],
                normal: membraneNormals[p],
                segmentLength: membraneSegmentLengths[p]
            );
            coord += maxDistance;
            coord = Math.Clamp(coord, 0, recDim - 1);

            int coord0 = (int)coord;
            int coord1 = Math.Min(recDim - 1, coord0 + 1);
            float weight1 = (coord - coord0);
            float weight0 = 1 - weight1;

            float val = recData[coord0] * weight0 + recData[coord1] * weight1;
            int i = iPixel.Y * finalMembraneImage.Dims.X + iPixel.X;
            finalMembraneImageData[i] = val;
        }

        string outputPathfinal = $"ab_finalmembrane.mrc";
        finalMembraneImage.WriteMRC(outputPathfinal);
        Console.WriteLine(outputPathfinal);

        return (finalSpline, finalIntensitySpline);
    }

    public static Image SplineToCoarseDistanceMap(SplinePath2D spline, int3 imageDims, int maxDistance)
    {
        Image RasterizedPath = RasterizeSpline(spline, imageDims: imageDims);
        Image DistanceMap = RasterizedPath.AsDistanceMapExact(maxDistance: maxDistance, isVolume: false);
        RasterizedPath.Dispose();
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

    public static Image PruneBranchesInPlace(Image skeleton)
    {
        // Get data from the skeleton image
        float[] skeletonData = skeleton.GetHost(Intent.ReadWrite)[0];

        int width = skeleton.Dims.X;
        int height = skeleton.Dims.Y;

        // Find all junction points and endpoints
        List<int> junctions = new List<int>();
        List<int> endpoints = new List<int>();

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
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

        // Trace from each endpoint to find branches...
        foreach (int endpoint in endpoints)
        {
            List<int> branch = TraceFromEndpoint(skeletonData, endpoint, width);

            // Find if this branch connects to a junction
            int lastPixel = branch[branch.Count - 1];
            bool connectsToJunction = junctions.Contains(lastPixel);

            // Remove branch if it connects to a junction
            if (connectsToJunction)
            {
                // Remove all pixels in branch except the junction point
                for (int i = 0; i < branch.Count - 1; i++)
                {
                    skeletonData[branch[i]] = 0;
                }
            }
        }

        return skeleton;
    }

    private static List<int> TraceFromEndpoint(float[] imageData, int startPixel, int width)
    {
        List<int> branch = new List<int>();
        branch.Add(startPixel);

        int currentPixel = startPixel;
        bool continueTracing = true;

        while (continueTracing)
        {
            // Find the next pixel in the branch
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

            bool foundNext = false;
            foreach (int offset in offsets)
            {
                int nextPixel = currentPixel + offset;

                // Make sure nextPixel is within bounds
                if (nextPixel < 0 || nextPixel >= imageData.Length) continue;

                // Check if this is a valid next pixel
                if (imageData[nextPixel] > 0 && !branch.Contains(nextPixel))
                {
                    branch.Add(nextPixel);
                    currentPixel = nextPixel;
                    foundNext = true;

                    // Check if this is a junction (has more than 2 neighbors)
                    int neighborCount = 0;
                    foreach (int neighborOffset in offsets)
                    {
                        int neighbor = nextPixel + neighborOffset;
                        if (neighbor >= 0 && neighbor < imageData.Length && imageData[neighbor] > 0)
                        {
                            neighborCount++;
                        }
                    }

                    // Stop if we've reached a junction or another endpoint
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

        return branch;
    }

    public static List<int> FindEndpoints(Image skeleton, List<int> componentIndices)
    {
        List<int> endpoints = new List<int>();
        float[] skeletonData = skeleton.GetHost(Intent.ReadWrite)[0];
        skeleton.WriteMRC16b("ab_skeleton_endpoints.mrc");
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

        while (true)
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

    public static int2[] FindMembranePixels(SplinePath2D spline, int3 imageDims, int maxDistancePx)
    {
        List<int2> membranePixels = new List<int2>();
        Image distanceMap = SplineToCoarseDistanceMap(spline, imageDims, maxDistancePx);
        distanceMap.WriteMRC16b("ab_dist_to_mem.mrc");
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

        distanceMap.Dispose();
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


//
//                     #region Profile Reconstruction
//
// // Calculate membrane parameters in pixels
//                     int MaxDistance = (int)MathF.Round((float)options.MembraneHalfWidth / ImageRaw.PixelSize);
//                     float SoftEdge = (float)options.MembraneEdgeSoftness / ImageRaw.PixelSize;
//
// // Create distance map from traced points
//                     DistanceMap = TraceScaled.AsDistanceMapExact(MaxDistance);
//                     float[] DistanceMapData = DistanceMap.GetHost(Intent.Read)[0];
//
// // Initialize lists for membrane data
//                     List<float> MembraneRefVals = new();
//                     List<int2> MembranePixels = new();
//                     List<int> MembraneClosestPoints = new();
//                     List<float> MembraneSegmentLengths = new();
//                     List<float2> MembraneTangents = new();
//                     List<float2> MembraneNormals = new();
//                     List<float> MembraneWeights = new();
//
// // Populate per-pixel helper data
//                     for (int i = 0; i < DistanceMapData.Length; i++)
//                     {
//                         if (DistanceMapData[i] < MaxDistance)
//                         {
//                             int2 iPoint = new int2(i % DistanceMap.Dims.X, i / DistanceMap.Dims.X);
//
//                             MembraneRefVals.Add(ImageLowpassData[iPoint.Y * ImageRaw.Dims.X + iPoint.X]);
//                             MembranePixels.Add(iPoint);
//
//                             float2 Point = new(iPoint);
//
//                             // Find closest point from Points
//                             List<float> PointDistances = Points.Select(p => (p - Point).LengthSq()).ToList();
//                             List<int> ClosestIDs = Helper.ArrayOfSequence(0, Points.Count, 1).ToList();
//                             ClosestIDs.Sort((a, b) => PointDistances[a].CompareTo(PointDistances[b]));
//
//                             // Handle special case where points are identical
//                             if (Points[ClosestIDs[0]] == Points[ClosestIDs[1]])
//                                 ClosestIDs.RemoveAt(1);
//
//                             int ID0 = ClosestIDs[0] < ClosestIDs[1] ? ClosestIDs[0] : ClosestIDs[1];
//                             int ID1 = ClosestIDs[0] < ClosestIDs[1] ? ClosestIDs[1] : ClosestIDs[0];
//
//                             float2 ClosestPoint0 = Points[ID0];
//                             float2 ClosestPoint1 = Points[ID1];
//
//                             MembraneClosestPoints.Add(ID0);
//                             MembraneSegmentLengths.Add((ClosestPoint1 - ClosestPoint0).Length());
//
//                             float2 Tangent = (ClosestPoint1 - ClosestPoint0).Normalized();
//                             MembraneTangents.Add(Tangent);
//
//                             float2 Normal = new(Tangent.Y, -Tangent.X);
//                             MembraneNormals.Add(Normal);
//                         }
//                     }
//
//                     DistanceMap.Dispose();
//
//                     #endregion
//
//                     #region Membrane Refinement
//
// // Define membrane coordinate calculation
//                     Func<int, float> GetMembraneCoord = (i) =>
//                     {
//                         float2 Location = new float2(MembranePixels[i]);
//                         float2 ClosestPoint = Points[MembraneClosestPoints[i]];
//
//                         float2 Delta = Location - ClosestPoint;
//                         float LineCoord = Math.Clamp(float2.Dot(Delta, MembraneTangents[i]),
//                             0,
//                             MembraneSegmentLengths[i]);
//                         ClosestPoint = ClosestPoint + MembraneTangents[i] * LineCoord;
//                         Delta = Location - ClosestPoint;
//
//                         float Coord = MathF.Sign(float2.Dot(Delta, MembraneNormals[i])) *
//                             Delta.Length() + MaxDistance;
//                         return Coord;
//                     };
//
// // Calculate membrane weights for soft edges
//                     for (int i = 0; i < MembranePixels.Count; i++)
//                     {
//                         float Coord = (MathF.Abs(GetMembraneCoord(i) - MaxDistance) -
//                                        (MaxDistance - SoftEdge)) / SoftEdge;
//                         float Weight = MathF.Cos(Math.Clamp(Coord, 0, 1) * MathF.PI) * 0.5f + 0.5f;
//                         MembraneWeights.Add(Weight);
//                     }
//
// // Setup profile reconstruction dimensions and data
//                     int RecDim = MaxDistance * 2;
//                     float[] RecData = new float[RecDim];
//                     float[] RecWeights = new float[RecDim];
//                     double[] OptimizationInput = new double[Spline.Points.Count + IntensitySpline.Points.Count];
//                     double[] OptimizationFallback = OptimizationInput.ToArray();
//
// // Perform membrane refinement iterations
//                     for (int iiter = 0; iiter < options.RefinementIterations; iiter++)
//                     {
//                         // Progress?.Report($"Refining membrane {ic + 1} of {Components.Length}");
//
//                         RecData = new float[RecDim];
//                         RecWeights = new float[RecDim];
//
//                         // Reconstruct membrane profile
//                         for (int i = 0; i < MembranePixels.Count; i++)
//                         {
//                             float Coord = GetMembraneCoord(i);
//                             float Val = ImageLowpassData[MembranePixels[i].Y * ImageRaw.Dims.X +
//                                                          MembranePixels[i].X];
//
//                             Coord = Math.Clamp(Coord, 0, RecDim - 1);
//                             int Coord0 = (int)Coord;
//                             int Coord1 = Math.Min(RecDim - 1, Coord0 + 1);
//                             float Weight1 = (Coord - Coord0);
//                             float Weight0 = 1 - Weight1;
//
//                             RecData[Coord0] += Val * Weight0;
//                             RecData[Coord1] += Val * Weight1;
//                             RecWeights[Coord0] += Weight0;
//                             RecWeights[Coord1] += Weight1;
//                         }
//
//                         for (int i = 0; i < RecDim; i++)
//                             RecData[i] /= Math.Max(1e-16f, RecWeights[i]);
//
//                         // Define optimization functions
//                         Func<double[], double> Eval = (input) =>
//                         {
//                             // Update control points
//                             float2[] NewControlPoints = new float2[ControlPoints.Length];
//                             for (int i = 0; i < ControlPoints.Length; i++)
//                                 NewControlPoints[i] = ControlPoints[i] + Normals[i] * (float)input[i];
//
//                             if (IsClosed)
//                                 NewControlPoints[NewControlPoints.Length - 1] = NewControlPoints[0];
//
//                             Spline = new SplinePath2D(NewControlPoints, IsClosed);
//                             Points = Spline.GetInterpolated(Helper.ArrayOfFunction(
//                                 i => (float)i / (Points.Count - 1), Points.Count)).ToList();
//
//                             // Update intensities
//                             float[] NewIntensityControlPoints = new float[IntensitySpline.Points.Count];
//                             for (int i = 0; i < NewIntensityControlPoints.Length; i++)
//                                 NewIntensityControlPoints[i] = MathF.Exp((float)input[NewControlPoints.Length + i]);
//
//                             if (IsClosed)
//                                 NewIntensityControlPoints[NewIntensityControlPoints.Length - 1] =
//                                     NewIntensityControlPoints[0];
//
//                             IntensitySpline = new SplinePath1D(NewIntensityControlPoints, IsClosed);
//                             ScaleFactors = IntensitySpline.GetInterpolated(Helper.ArrayOfFunction(
//                                 i => (float)i / (Points.Count - 1), Points.Count)).ToList();
//
//                             // Calculate RMSD
//                             double Result = 0;
//                             for (int i = 0; i < MembranePixels.Count; i++)
//                             {
//                                 float Coord = GetMembraneCoord(i);
//                                 Coord = Math.Clamp(Coord, 0, RecDim - 1);
//
//                                 int Coord0 = (int)Coord;
//                                 int Coord1 = Math.Min(RecDim - 1, Coord0 + 1);
//                                 float Weight1 = (Coord - Coord0);
//                                 float Weight0 = 1 - Weight1;
//
//                                 float Val = RecData[Coord0] * Weight0 + RecData[Coord1] * Weight1;
//                                 float Intensity = ScaleFactors[MembraneClosestPoints[i]];
//                                 Val *= Intensity;
//
//                                 Result += (MembraneRefVals[i] - Val) * (MembraneRefVals[i] - Val) *
//                                           MembraneWeights[i];
//                             }
//
//                             return Math.Sqrt(Result / MembranePixels.Count);
//                         };
//
//                         // Run optimization
//                         try
//                         {
//                             BroydenFletcherGoldfarbShanno Optimizer =
//                                 new BroydenFletcherGoldfarbShanno(OptimizationInput.Length, Eval, null);
//                             Optimizer.MaxIterations = 10;
//                             Optimizer.MaxLineSearch = 5;
//                             Optimizer.Minimize(OptimizationInput);
//                             OptimizationFallback = OptimizationInput.ToArray();
//                         }
//                         catch (Exception e)
//                         {
//                             OptimizationInput = OptimizationFallback.ToArray();
//                             Eval(OptimizationInput);
//                         }
//                     }
//
//                     #endregion
//
//                     #region Save Results
//
//                     // Update control points with final optimization results
//                     for (int i = 0; i < ControlPoints.Length; i++)
//                         ControlPoints[i] = ControlPoints[i] + Normals[i] * (float)OptimizationInput[i];
//                     if (IsClosed)
//                         ControlPoints[ControlPoints.Length - 1] = ControlPoints[0];
//
//                     // Create and add path table
//                     PathTables.Add($"path{ic:D3}", new Star(
//                         ControlPoints.Select(v => v * ImageRaw.PixelSize).ToArray(),
//                         "wrpControlPointXAngst",
//                         "wrpControlPointYAngst"));
//
//                     // Save final control points
//                     Directory.CreateDirectory(MembraneModelsDir);
//                     Star.SaveMultitable(MembraneControlPointsPath, PathTables);
//
//                     #endregion