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
        // Maps and textures (legacy - for when all maps are loaded into memory)
        public List<Image> Maps1 { get; set; } = new List<Image>();
        public List<Image> Maps2 { get; set; } = new List<Image>();
        public List<Image> MapCTFs { get; set; } = new List<Image>();
        public List<ulong[]> Textures1 { get; set; } = new List<ulong[]>();
        public List<ulong[]> Textures2 { get; set; } = new List<ulong[]>();

        // Rotating map pool (for memory-efficient training on large datasets)
        public RotatingMapPool MapPool { get; set; }

        // Denoising data
        public List<Image> MapsForDenoising { get; set; } = new List<Image>();
        public List<Image> MapsForDenoising2 { get; set; } = new List<Image>();
        public List<string> NamesForDenoising { get; set; } = new List<string>();
        public List<int3> DimensionsForDenoising { get; set; } = new List<int3>();
        public List<int3> OriginalBoxForDenoising { get; set; } = new List<int3>();
        public List<float2> MeanStdForDenoising { get; set; } = new List<float2>();
        public List<float> PixelSizeForDenoising { get; set; } = new List<float>();

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

            foreach (var map in Maps1)
                map?.Dispose();
            foreach (var map in Maps2)
                map?.Dispose();
            foreach (var map in MapCTFs)
                map?.Dispose();
            foreach (var map in MapsForDenoising)
                map?.Dispose();
            foreach (var map in MapsForDenoising2)
                map?.Dispose();
        }
    }
}