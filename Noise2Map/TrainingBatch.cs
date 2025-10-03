using System;
using Warp;

namespace Noise2Map
{
    /// <summary>
    /// Encapsulates a batch of prepared and shuffled training data ready for consumption by the training thread
    /// </summary>
    public class TrainingBatch : IDisposable
    {
        public int[] ShuffledMapIDs { get; set; }
        public Image[] ExtractedSourceRand { get; set; }
        public Image[] ExtractedTargetRand { get; set; }
        public Image[] ExtractedCTFRand { get; set; }

        public void Dispose()
        {
            DisposeImageArray(ExtractedSourceRand);
            DisposeImageArray(ExtractedTargetRand);
            DisposeImageArray(ExtractedCTFRand);
        }

        private void DisposeImageArray(Image[] images)
        {
            if (images != null)
            {
                foreach (var img in images)
                {
                    img?.Dispose();
                }
            }
        }
    }
}
