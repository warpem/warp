using System;
using System.Collections.Generic;

namespace Warp.Tools;

/// <summary>
/// cuFFT plans for one thread's work alone, made on demand and destroyed with this object. Work that runs beside the
/// worker's main thread uses these instead of <see cref="FFTPlanCache"/>, whose plans are shared: one cuFFT plan must
/// not run on two threads at once, and the cache may destroy a plan another thread is still using. The transforms
/// are the same as with the cache's plans.
/// </summary>
public sealed class PrivateFftPlans : IDisposable
{
    readonly Dictionary<(int3 Dims, bool Forward), int> plans = new();

    /// <summary>A single-slice real-to-complex plan of <paramref name="dims"/>.</summary>
    public int Forward(int3 dims) => Get(dims, true);

    /// <summary>A single-slice complex-to-real plan of <paramref name="dims"/>.</summary>
    public int Inverse(int3 dims) => Get(dims, false);

    int Get(int3 dims, bool forward)
    {
        ObjectDisposedException.ThrowIf(plans == null, this);
        if (!plans.TryGetValue((dims, forward), out int plan))
            plans[(dims, forward)] = plan = forward ? GPU.CreateFFTPlan(dims, 1) : GPU.CreateIFFTPlan(dims, 1);
        return plan;
    }

    public void Dispose()
    {
        foreach (int plan in plans.Values) GPU.DestroyFFTPlan(plan);
        plans.Clear();
    }
}
