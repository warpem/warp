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
        List<Image> toDispose = new();
        try
        {
            // Validate required input files exist
            if (!File.Exists(AveragePath))
                throw new FileNotFoundException($"Average image file not found at {AveragePath}");

            var membraneStarFiles = Directory.GetFiles(MembraneModelsDir).Where(
                file => IOPath.GetFileName(file).StartsWith(RootName + "_membrane") && 
                IOPath.GetFileName(file).EndsWith(".star")
            );
            int nMembranes = membraneStarFiles.Count();
            if (nMembranes == 0)
                throw new FileNotFoundException($"no membrane files found in {MembraneModelsDir}");
            
            // load image data
            Image average = Image.FromFile(AveragePath);
            toDispose.Add(average);
            
            // preprocess
            average.SubtractMeanGrid(new int2(1));
            average = average.AsPadded(average.DimsSlice * 2).AndDisposeParent();
            average.Bandpass(
                nyquistLow: average.PixelSize * 2 / 300f, // 1/300 Å
                nyquistHigh: 1f, // nyquist
                nyquistsoftedge: average.PixelSize * 2 / 600f,  // 1/600 Å
                isVolume: false
            );
            average = average.AsPadded(average.DimsSlice / 2).AndDisposeParent();
            
            // make Image for all membranes
            Image allMembranes = new Image(average.Dims);
            float[] allMembranesData = allMembranes.GetHost(Intent.ReadWrite)[0];
            toDispose.Add(allMembranes);
            
            // make Image for all membrane traces (used for calculating regions around each membrane for adding noise)
            Image allTraces = new Image(average.Dims);
            float[] allTracesData = allTraces.GetHost(Intent.ReadWrite)[0];
            toDispose.Add(allTraces);
            
            // reconstruct each membrane
            for (int ic = 0; ic < nMembranes; ic++)
            {
                Console.WriteLine($"Reconstructing membrane {ic + 1} of {nMembranes}");

                var (profile1D, path, intensitySpline) = SubtractMembranesHelper.LoadMembrane(movie: this, index: ic);
                
                // get max distance away from membrane to reconstruct from profile data
                int maxDistance = profile1D.Length / 2;

                // sample points along spline
                int nPoints = (int)path.EstimatedLength;
                float[] tPoints = Helper.ArrayOfFunction(i => (float)i / nPoints, n: nPoints);
                float2[] points = path.GetInterpolated(tPoints);
                
                // find pixels which are close to current membrane
                int2[] membranePixels = TraceMembranesHelper.FindMembranePixels(
                    path: path, imageDims: average.Dims, maxDistancePx: maxDistance
                );
                int nMembranePixels = membranePixels.Length;
                int2 iPixel;
                
                // allocate arrays for cached per-pixel data
                int[] membraneClosestPointIdx = new int[nMembranePixels];
                float[] membraneSegmentLengths = new float[nMembranePixels];
                float2[] membraneTangents = new float2[nMembranePixels];
                float2[] membraneNormals = new float2[nMembranePixels];
                float[] membraneWeights = new float[nMembranePixels];
                bool[] excludePixel = new bool[nMembranePixels];
                
                // calculate cached per-pixel data
                for (int p = 0; p < nMembranePixels; p++)
                {
                    // grab pixel coords and calculate per pixel data
                    iPixel = membranePixels[p];
                    float2 pixel = new float2(iPixel);
                    var (idx0, idx1) = TraceMembranesHelper.FindClosestLineSegment(queryPoint: pixel, pathPoints: points);
                    var (p0, p1) = (points[idx0], points[idx1]);
                    float length = (p1 - p0).Length();
                    float2 tangent = (p1 - p0).Normalized();
                    float2 normal = new float2(tangent.Y, -tangent.X);
                    float signedDistanceFromMembrane = TraceMembranesHelper.GetSignedDistanceFromMembrane(
                        pixelLocation: pixel,
                        closestPointOnSpline: p0,
                        tangent: tangent,
                        normal: normal,
                        segmentLength: length
                    );
                    float weight = TraceMembranesHelper.CalculateWeight(
                        distanceFromSpline: MathF.Abs(signedDistanceFromMembrane),
                        maxDistance: maxDistance,
                        softEdgeWidth: (int)(30f / average.PixelSize)
                    );
                    float projectedLength = float2.Dot(pixel - p0, tangent);
                    
                    membraneClosestPointIdx[p] = idx0;
                    membraneSegmentLengths[p] = length;
                    membraneTangents[p] = tangent;
                    membraneNormals[p] = normal;
                    membraneWeights[p] = weight;
                    excludePixel[p] = projectedLength < 0 || projectedLength > length;
                }

                int recDim = profile1D.Length;
                
                // reconstruct the membrane
                for (int p = 0; p < membranePixels.Length; p++)
                {
                    iPixel = membranePixels[p];
                    float2 pixel = new float2(iPixel);
                    
                    // early exit if pixel is not 'within' line segment
                    // temporarily switched off because of artifacts
                    // if (excludePixel[p])
                    //     continue;
                    
                    float coord = TraceMembranesHelper.GetSignedDistanceFromMembrane(
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

                    float val = profile1D[coord0] * weight0 + profile1D[coord1] * weight1;
                    int i = iPixel.Y * average.Dims.X + iPixel.X;
                    allMembranesData[i] += val * membraneWeights[p];
                }
                
                // Rasterize lines so we can compute the distance map
                for (int ip = 0; ip < points.Length - 1; ip++)
                {
                    float2 P0 = points[ip];
                    float2 P1 = points[ip + 1];

                    int x0 = (int)MathF.Round(P0.X);
                    int y0 = (int)MathF.Round(P0.Y);
                    int x1 = (int)MathF.Round(P1.X);
                    int y1 = (int)MathF.Round(P1.Y);

                    if (x0 < 0 || x0 > allTraces.Dims.X - 1 || y0 < 0 || y0 > allTraces.Dims.Y - 1)
                        continue;
                    if (x1 < 0 || x1 > allTraces.Dims.X - 1 || y1 < 0 || y1 > allTraces.Dims.Y - 1)
                        continue;

                    int dx = Math.Abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
                    int dy = -Math.Abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
                    int err = dx + dy, e2; // error value e_xy

                    while (true)
                    {
                        allTracesData[y0 * allTraces.Dims.X + x0] = 1;
                        if (x0 == x1 && y0 == y1) break;
                        e2 = 2 * err;
                        if (e2 >= dy) { err += dy; x0 += sx; }
                        if (e2 <= dx) { err += dx; y0 += sy; }
                    }
                }
                
            }
            // write out all membranes image (for diagnostics)
            allMembranes.WriteMRC16b(MembraneReconstructionPath);
            
            // Subtract membranes
            Directory.CreateDirectory(AverageMembraneSubtractedDir);
            allMembranes.Multiply((float)options.MembraneSubtractionFactor);
            average.Subtract(allMembranes);
            average.WriteMRC16b(AverageMembraneSubtractedPath);
        }
        finally
        {
            // Clean up resources
            foreach (var image in toDispose)
                image.Dispose();
        }
    }
}

[Serializable]
public class ProcessingOptionsSubtractMembranes : ProcessingOptionsBase
{
    [WarpSerializable] public decimal MembraneSubtractionFactor { get; set; } = 0.75M;
}

public static class SubtractMembranesHelper
{
    public static (float[] profile1D, SplinePath2D path, SplinePath1D intensitySpline) LoadMembrane(Movie movie, int index)
    {
        // load file
        string filename = IOPath.Combine(movie.MembraneModelsDir, $"{movie.RootName}_membrane{index:D3}.star");
        Dictionary<string, Star> membraneData = Star.FromMultitable(
            path: filename,
            names: new[]
            {
                "profile1d",
                "path",
                "intensity"
            }
        );
        
        // extract data
        float[] profile1D = membraneData["profile1d"].GetFloat("wrpMembraneProfile");
        float2[] pathControlPoints = membraneData["path"]
            .GetFloat2("wrpPathControlPointXAngst", "wrpPathControlPointYAngst");
        float[] intensitySplineControlPoints = membraneData["intensity"]
            .GetFloat("wrpIntensitySplineControlPoints");
        
        // is spline closed? yes if last control point matches first
        bool isClosed = pathControlPoints[0] == pathControlPoints[^1];
        
        // construct splines
        SplinePath2D path = new SplinePath2D(points: pathControlPoints, isClosed: isClosed);
        SplinePath1D intensitySpline = new SplinePath1D(points: intensitySplineControlPoints, isClosed: isClosed);


        return (profile1D, path, intensitySpline);
    }
}