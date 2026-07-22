using System;
using System.Collections.Generic;
using Warp.Tools;
using ZLinq;

namespace Warp;

public static class RelionParticleSeriesExport
{
    public static int[] GetVisibleTiltIndices(IReadOnlyList<int> usedTilts,
                                              IReadOnlyList<bool> visibility)
    {
        if (usedTilts.Count != visibility.Count)
            throw new ArgumentException("Used-tilt and visibility counts must match.");

        return usedTilts.Where((_, i) => visibility[i]).ToArray();
    }

    public static int3 GetVirtualTomogramDimensions(float3 dimensionsPhysical,
                                                     float particlePixelSize)
    {
        if (particlePixelSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(particlePixelSize));

        return new int3(
            Math.Max(1, (int)Math.Round(dimensionsPhysical.X / particlePixelSize)),
            Math.Max(1, (int)Math.Round(dimensionsPhysical.Y / particlePixelSize)),
            Math.Max(1, (int)Math.Round(dimensionsPhysical.Z / particlePixelSize)));
    }

    public static float GetRelionHand(bool areAnglesInverted)
    {
        return areAnglesInverted ? 1f : -1f;
    }

    public static decimal GetPhaseShiftDegrees(decimal phaseShiftPi)
    {
        return phaseShiftPi * 180M;
    }
}
