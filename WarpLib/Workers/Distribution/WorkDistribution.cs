using Microsoft.Extensions.Logging;

namespace Warp.Workers.Distribution;

/// <summary>
/// Static service locator for global access to the work distribution system.
/// Provides a shared WorkDistributor instance accessible from anywhere in the application.
/// </summary>
public static class WorkDistribution
{
    /// <summary>
    /// The shared WorkDistributor instance.
    /// </summary>
    public static readonly WorkDistributor Instance = new WorkDistributor();
}