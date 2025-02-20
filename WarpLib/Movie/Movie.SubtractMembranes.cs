// using System;
// using System.Collections.Generic;
// using System.Globalization;
// using System.IO;
// using System.Linq;
// using Accord.Math.Optimization;
// using Warp;
// using Warp.Tools;
// using IOPath = System.IO.Path;
//
// namespace Warp;
//
// public partial class Movie
// {
//     public void SubtractMembranes(ProcessingOptionsSubtractMembranes options)
//     {
//         List<Image> toDispose = new();
//         try
//         {
//             // Validate required input files exist
//             if (!File.Exists(AveragePath))
//                 throw new FileNotFoundException($"Average image file not found at {AveragePath}");
//
//             if (!File.Exists(MembraneControlPointsPath))
//                 throw new FileNotFoundException($"Membrane control points file not found at {MembraneControlPointsPath}");
//
//             if (!File.Exists(MembraneProfilesPath))
//                 throw new FileNotFoundException($"Membrane profiles file not found at {MembraneProfilesPath}");
//
//             // Load input data
//             Image ImageRaw = Image.FromFile(AveragePath);
//             toDispose.Add(ImageRaw);
//
//             var ControlPoints = Star.Load(MembraneControlPointsPath);
//             var Profiles = Star.Load(MembraneProfilesPath);
//
//             // Process each membrane
//             int membraneCount = ControlPoints.Count;
//             for (int i = 0; i < membraneCount; i++)
//             {
//                 // Progress?.Report($"Subtracting membrane {i + 1} of {membraneCount}");
//
//                 // TODO: Implement membrane subtraction logic
//                 // 1. Get control points for current membrane
//                 // 2. Get corresponding profile data
//                 // 3. Apply profile subtraction
//                 // 4. Add noise if specified by options.NoiseStdDevScale
//             }
//
//             // Save output
//             string outputPath = IOPath.Combine(
//                 IOPath.GetDirectoryName(AveragePath),
//                 IOPath.GetFileName(AveragePath)
//             );
//             ImageRaw.WriteMRC16b(outputPath);
//         }
//         finally
//         {
//             // Clean up resources
//             foreach (var image in toDispose)
//                 image.Dispose();
//         }
//     }
// }
//
// [Serializable]
// public class ProcessingOptionsSubtractMembranes : ProcessingOptionsBase
// {
//     [WarpSerializable] public decimal NoiseStdDevScale { get; set; } = 0.25M;
// }
//
// public static class SubtractMembranesHelper
// {
// }