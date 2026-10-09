using System;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public sealed class CtfCudaFactAttribute : FactAttribute
{
    public CtfCudaFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("WARP_RUN_CUDA_TESTS") != "1")
            Skip = "Set WARP_RUN_CUDA_TESTS=1 on a CUDA host.";
    }
}

public class CtfPowerSpectrumCudaTests
{
    [CtfCudaFact]
    public void FrameGroupingPreservesAllPowerAndUsesFrameCoordinates()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var image = new Image(new int3(128, 128, 7));
            var rng = new Random(732);
            var pixels = image.GetHost(Intent.Write);
            for (int f = 0; f < pixels.Length; f++) for (int i = 0; i < pixels[f].Length; i++) pixels[f][i] = (float)(rng.NextDouble() - .5) * (f + 1);
            var options = new ProcessingOptionsMovieCTF { Window = 64, PixelSize = 4, RangeMin = .15M, RangeMax = .8M, ZMin = .1M, ZMax = .8M, Voltage = 300, Cs = 2.7M, Amplitude = .07M };
            var pooled = CtfPowerSpectrum.Extract(image, options);
            var grouped = CtfPowerSpectrum.Extract(image, options, 3);
            int patches = pooled.Observations.Count;
            Assert.Equal(patches * 3, grouped.Observations.Count);
            Assert.Equal(.5f / 6, grouped.Observations[0].Position.Z, 6);
            Assert.Equal(2.5f / 6, grouped.Observations[patches].Position.Z, 6);
            Assert.Equal(5f / 6, grouped.Observations[2 * patches].Position.Z, 6);
            for (int p = 0; p < patches; p++)
            {
                var a = pooled.Observations[p].Spectrum.Samples;
                for (int i = 0; i < a.Length; i++)
                {
                    double power = 0, count = 0;
                    for (int group = 0; group < 3; group++)
                    {
                        var b = grouped.Observations[group * patches + p].Spectrum.Samples[i];
                        count += b.Count; power += b.Count * b.Power;
                    }
                    Assert.InRange(Math.Abs(count - a[i].Count), 0, 1e-10);
                    Assert.InRange(Math.Abs(power / count - a[i].Power) / Math.Max(1, a[i].Power), 0, 2e-6);
                }
            }
        }
    }
}
