using System;
using System.Collections.Generic;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Holds all shared state and data structures used throughout the Noise2Map pipeline
    /// </summary>
    public class ProcessingContext
    {
        // Rotating map pool (for memory-efficient training)
        public RotatingMapPool MapPool { get; set; }

        // Mask and bounds
        public Image Mask { get; set; }
        public int3 CropBox { get; set; } = new int3(-1);
        public int3 BoundsMin { get; set; } = new int3(0);
        public int3 BoundsMax { get; set; } = new int3(10000);

        // Dimensions
        public int3 TrainingDims { get; set; }
        public bool IsTomo { get; set; }

        // Directories
        public string WorkingDirectory { get; set; }
        public string ProgramFolder { get; set; }

        public void Dispose()
        {
            Mask?.Dispose();
            MapPool?.Dispose();
        }
    }
}