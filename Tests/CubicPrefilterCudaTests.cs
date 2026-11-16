using System;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CubicPrefilterCudaTests
{
    /// <summary>
    /// The tiled (rows) and batched (columns) prefilter kernels run each line's recursion with the same operations in
    /// the same order as the one-thread-per-line kernels, so the coefficients must be identical, not merely close.
    /// Sizes cover partial tiles and batches, lines shorter than the 12-sample initialization horizon, and a 4k frame.
    /// </summary>
    [TemplateMatchCudaFact]
    public void FasterPrefilterMatchesTheOriginalBitForBit()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            foreach (var (w, h) in new[] { (4096, 4096), (75, 64), (33, 1000), (1000, 33), (5, 7), (32, 32), (17, 129) })
            {
                var random = new Random(w * 31 + h);
                float[] data = Enumerable.Range(0, w * h).Select(_ => (float)(random.NextDouble() * 200 - 50)).ToArray();
                using var fast = new Image(data, new int3(w, h, 1));
                using var reference = new Image(data, new int3(w, h, 1));
                GPU.PrefilterForCubic(fast.GetDevice(Intent.ReadWrite), fast.Dims);
                GPU.PrefilterForCubicReference2D(reference.GetDevice(Intent.ReadWrite), reference.Dims);
                float[] a = fast.GetHost(Intent.Read)[0], b = reference.GetHost(Intent.Read)[0];
                int mismatches = 0, first = -1;
                for (int i = 0; i < a.Length; i++)
                    if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) { mismatches++; if (first < 0) first = i; }
                Assert.True(mismatches == 0, $"{w}x{h}: {mismatches} coefficients differ, first at {first}: {(first >= 0 ? a[first] : 0):G9} vs {(first >= 0 ? b[first] : 0):G9}");
                // And the filter did something.
                Assert.True(a.Zip(data, (x, y) => Math.Abs(x - y)).Max() > 1);
            }
        }
    }
}
