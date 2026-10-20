using System;
using System.Reflection;
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
    static int ApertureSize(int2 dimensions, ProcessingOptionsMovieCTF options) =>
        (int)typeof(CtfPowerSpectrum).GetMethod("ApertureSize", BindingFlags.Static | BindingFlags.NonPublic)
            .Invoke(null, new object[] { dimensions, options });

    [Fact]
    public void ApertureProtectsPhysicalSupportAtSmallPixelsAndRespectsImageBounds()
    {
        var options = new ProcessingOptionsMovieCTF
            { Window = 512, PixelSize = .834M, RangeMin = .0556M, RangeMax = .2085M, ZMin = 1, ZMax = 8, Voltage = 300 };
        Assert.Equal(1536, ApertureSize(new int2(5760, 4092), options));
        Assert.Equal(1000, ApertureSize(new int2(1400, 1001), options));
        options.PixelSize = 1.7005M; options.ZMax = 7;
        options.RangeMin = 2 * options.PixelSize / 30; options.RangeMax = 2 * options.PixelSize / 4;
        Assert.Equal(512, ApertureSize(new int2(3838, 3710), options));
        options.PixelSize = 1.5M; options.ZMax = 6;
        options.RangeMin = 2 * options.PixelSize / 30; options.RangeMax = 2 * options.PixelSize / 4;
        Assert.Equal(512, ApertureSize(new int2(2048), options));
        options.PixelSize = 2.75M;
        options.RangeMin = 5.5M / 30;
        options.RangeMax = 5.5M / 8;
        Assert.Equal(512, ApertureSize(new int2(2048), options));
    }

    [CtfCudaFact]
    public void EnlargedApertureKeepsRequestedDisplayAndDiagnosticGeometry()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var image = new Image(new int3(768, 768, 1));
            var random = new Random(143);
            var pixels = image.GetHost(Intent.Write)[0];
            for (int i = 0; i < pixels.Length; i++) pixels[i] = (float)random.NextDouble();
            var options = new ProcessingOptionsMovieCTF
                { Window = 128, PixelSize = 1, RangeMin = 2M / 30, RangeMax = .25M, ZMin = .5M, ZMax = 2, Voltage = 300, Cs = 2.7M, Amplitude = .07M };
            var automatic = CtfPowerSpectrum.Extract(image, options);
            Assert.True(automatic.AperturePixels > options.Window);
            Assert.Equal(128 * 64, automatic.Display.Length);
            using var diagnostics = new CtfPowerSpectrum.Extractor(new int2(image.Dims), options, fullSpectrum: true);
            Assert.Equal(automatic.AperturePixels, diagnostics.AperturePixels);
            options.Window = automatic.AperturePixels;
            var explicitWindow = CtfPowerSpectrum.Extract(image, options);
            Assert.Equal(explicitWindow.PositionGrid, automatic.PositionGrid);
            Assert.Equal(explicitWindow.FourierSize, automatic.FourierSize);
            for (int p = 0; p < automatic.Observations.Count; p++)
            {
                Assert.Equal(explicitWindow.Observations[p].Position, automatic.Observations[p].Position);
                Assert.Equal(explicitWindow.Observations[p].Spectrum.Samples, automatic.Observations[p].Spectrum.Samples);
            }
        }
    }

    [CtfCudaFact]
    public void MovieFramesArePooledIntoSpatialSpectra()
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
            using var extractor=new CtfPowerSpectrum.Extractor(new int2(128),options);
            var separate=new CtfPowerSpectrum.Extraction[pixels.Length];
            for(int f=0;f<pixels.Length;f++)
            {
                using var frame=new Image(new[]{pixels[f]},new int3(128,128,1));
                separate[f]=extractor.Extract(frame,group:f);
                Assert.All(separate[f].Observations,o=>Assert.Equal(f,o.Group));
            }
            int patches = pooled.Observations.Count;
            Assert.Equal(pooled.PositionGrid.Elements(),patches);
            Assert.All(pooled.Observations,o=>{Assert.Equal(0,o.Group);Assert.Equal(.5f,o.Position.Z);});
            for (int p = 0; p < patches; p++)
            {
                var a = pooled.Observations[p].Spectrum.Samples;
                for (int i = 0; i < a.Length; i++)
                {
                    double power = 0, count = 0;
                    for (int frame = 0; frame < pixels.Length; frame++)
                    {
                        var b = separate[frame].Observations[p].Spectrum.Samples[i];
                        count += b.Count; power += b.Count * b.Power;
                    }
                    Assert.InRange(Math.Abs(count - a[i].Count), 0, 1e-10);
                    Assert.InRange(Math.Abs(power / count - a[i].Power) / Math.Max(1, a[i].Power), 0, 2e-6);
                }
            }
        }
    }
}
