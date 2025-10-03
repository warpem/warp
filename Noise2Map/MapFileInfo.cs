using System;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Stores metadata about a map pair without loading the actual image data
    /// </summary>
    public class MapFileInfo
    {
        public string Map1Path { get; set; }
        public string Map2Path { get; set; }
        public string MapCombinedPath { get; set; }
        public string MapName { get; set; }

        // Preprocessing metadata
        public float PixelSize { get; set; }
        public float2 MeanStd { get; set; }
        public int3 OriginalDims { get; set; }
        public int3 CropBox { get; set; }

        // For spectral flattening
        public float[] SpectrumMultipliers { get; set; }

        // CTF path
        public string CTFPath { get; set; }

        // Denoising metadata
        public bool IsForDenoising { get; set; }
    }
}
