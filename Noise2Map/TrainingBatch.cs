using System;
using Warp;

namespace Noise2Map
{
    /// <summary>
    /// Container for a prepared training batch (single map with samples shuffled across multiple maps)
    /// </summary>
    public class TrainingBatch : IDisposable
    {
        public Image ExtractedSource { get; set; }
        public Image ExtractedTarget { get; set; }
        public Image ExtractedCTF { get; set; }

        public void Dispose()
        {
            ExtractedSource?.Dispose();
            ExtractedTarget?.Dispose();
            ExtractedCTF?.Dispose();
        }
    }
}
