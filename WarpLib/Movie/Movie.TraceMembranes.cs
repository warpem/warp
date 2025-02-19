using System;
using Warp;

namespace Warp
{
    public partial class Movie
    {
        public void TraceMembranes()
        {
        }

        public void SubtractMembranes()
        {
            
        }
    }
    
    [Serializable]
    public class ProcessingOptionsTraceMembranes : ProcessingOptionsBase
    {
        // Placeholder properties for future use
        [WarpSerializable] public int MinComponentSize { get; set; } = 20;  // px
        [WarpSerializable] public float BandpassLow { get; set; } = 0.002f;
        [WarpSerializable] public float BandpassHigh { get; set; } = 0.05f;
        [WarpSerializable] public bool EnableSplineRefinement { get; set; } = true;
    }
}

public static class TraceMembranesHelper
{
    public static Image HelperMethod1(Image raw)
    {
        return new Image();
    }
}