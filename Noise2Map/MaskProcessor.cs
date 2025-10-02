using System;
using System.IO;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Handles mask loading and processing
    /// </summary>
    public static class MaskProcessor
    {
        /// <summary>
        /// Loads and processes the mask, calculating crop box and bounds
        /// </summary>
        /// <param name="context">Processing context to update with mask information</param>
        /// <param name="options">Configuration options</param>
        public static void LoadMask(ProcessingContext context, Options options)
        {
            if (string.IsNullOrEmpty(options.MaskPath))
            {
                Console.WriteLine("No mask specified.\n");
                return;
            }

            Console.Write("Loading mask... ");

            context.Mask = Image.FromFile(Path.Combine(context.WorkingDirectory, options.MaskPath));

            if (!options.DontKeepDimensions)
            {
                context.BoundsMin = context.Mask.Dims;
                context.BoundsMax = new int3(0);
            }

            CalculateBounds(context, options);

            if (context.CropBox.X < 2)
                throw new Exception("Mask does not seem to contain any non-zero values.");

            context.CropBox += 64;

            context.CropBox = new int3(
                Math.Min(context.CropBox.X, context.Mask.Dims.X),
                Math.Min(context.CropBox.Y, context.Mask.Dims.Y),
                Math.Min(context.CropBox.Z, context.Mask.Dims.Z)
            );

            Console.WriteLine("done.\n");
        }

        /// <summary>
        /// Calculates crop box and bounds from the mask
        /// </summary>
        private static void CalculateBounds(ProcessingContext context, Options options)
        {
            int3 cropBox = context.CropBox;
            int3 boundsMin = context.BoundsMin;
            int3 boundsMax = context.BoundsMax;

            context.Mask.TransformValues((x, y, z, v) =>
            {
                if (v > 1e-3f)
                {
                    cropBox = new int3(
                        Math.Max(cropBox.X, Math.Abs(x - context.Mask.Dims.X / 2) * 2),
                        Math.Max(cropBox.Y, Math.Abs(y - context.Mask.Dims.Y / 2) * 2),
                        Math.Max(cropBox.Z, Math.Abs(z - context.Mask.Dims.Z / 2) * 2)
                    );

                    if (!options.DontKeepDimensions)
                    {
                        boundsMin = int3.Min(boundsMin, new int3(x, y, z));
                        boundsMax = int3.Max(boundsMax, new int3(x, y, z));
                    }
                }

                return v;
            });

            context.CropBox = cropBox;
            context.BoundsMin = boundsMin;
            context.BoundsMax = boundsMax;
        }
    }
}