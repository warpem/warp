using System;
using System.Linq;
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
    public void SmallGridFrameSumsMatchFineGridPeriodograms()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var image = new Image(new int3(512, 512, 5));
            var random = new Random(911);
            var frames = image.GetHost(Intent.Write);
            for (int f = 0; f < frames.Length; f++)
                for (int y = 0; y < 512; y++) for (int x = 0; x < 512; x++)
                    frames[f][y * 512 + x] = (float)(random.NextDouble() - .5 + .3 * Math.Sin(.31 * x + .17 * y + f) + .002 * x);
            var options = new ProcessingOptionsMovieCTF
                { Window = 128, PixelSize = 1, RangeMin = 2M / 30, RangeMax = .5M, ZMin = .3M, ZMax = 1, Voltage = 300, Cs = 2.7M, Amplitude = .07M };
            var dims = new int2(image.Dims);
            var toggle = typeof(CtfPowerSpectrum).GetField("UseSmallGrid", BindingFlags.Static | BindingFlags.NonPublic);
            CtfPowerSpectrum.Extraction direct, directFull;
            toggle.SetValue(null, false);
            try
            {
                using var fitDirect = new CtfPowerSpectrum.Extractor(dims, options);
                using var fullDirect = new CtfPowerSpectrum.Extractor(dims, options, fullSpectrum: true);
                direct = fitDirect.Extract(image); directFull = fullDirect.Extract(image);
            }
            finally { toggle.SetValue(null, true); }
            using var fit = new CtfPowerSpectrum.Extractor(dims, options);
            using var full = new CtfPowerSpectrum.Extractor(dims, options, fullSpectrum: true);
            var summed = fit.Extract(image);
            // Both grids are finer than twice the window, so both take the small-grid path by default.
            Assert.Equal(128, direct.AperturePixels);
            Assert.True(direct.FourierSize > 2 * direct.AperturePixels && directFull.FourierSize > direct.FourierSize);
            Assert.True(full.CanShareFrameSums(fit));
            var shared = full.Extract(image, framesSummedBy: fit);
            image.GetDevice(Intent.Read);
            Assert.True(image.IsDeviceCurrent);
            using var onDevice = new CtfPowerSpectrum.Extractor(dims, options);
            var resident = onDevice.Extract(image);
            AssertSameSpectra(direct, summed);
            AssertSameSpectra(directFull, shared);
            AssertSameSpectra(summed, resident);

            // Frame sums are only shared for the image they were read from, unchanged since.
            using var other = new Image(image.Dims);
            Assert.Throws<InvalidOperationException>(() => full.Extract(other, framesSummedBy: fit));
            Assert.NotNull(full.Extract(image, framesSummedBy: onDevice));
            image.GetHost(Intent.ReadWrite);
            Assert.Throws<InvalidOperationException>(() => full.Extract(image, framesSummedBy: onDevice));
        }
    }

    [CtfCudaFact]
    public void OwnStreamDiagnosticReadIsIdenticalWhileTheDefaultStreamWorks()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var image = new Image(new int3(512, 512, 4));
            var random = new Random(27);
            var frames = image.GetHost(Intent.Write);
            for (int f = 0; f < frames.Length; f++)
                for (int i = 0; i < frames[f].Length; i++) frames[f][i] = (float)(random.NextDouble() - .5 + .2 * Math.Sin(.23 * (i % 512) + .11 * (i / 512) + f));
            var options = new ProcessingOptionsMovieCTF
                { Window = 128, PixelSize = 1, RangeMin = 2M / 30, RangeMax = .5M, ZMin = .3M, ZMax = 1, Voltage = 300, Cs = 2.7M, Amplitude = .07M };
            var dims = new int2(image.Dims);
            using var fit = new CtfPowerSpectrum.Extractor(dims, options);
            fit.Extract(image);
            using var full = new CtfPowerSpectrum.Extractor(dims, options, fullSpectrum: true);
            var expected = full.Extract(image, framesSummedBy: fit);
            // As in ProcessCTF: another thread reads the diagnostics on its own stream while this one keeps the
            // default stream busy with transforms of the same frames.
            CtfPowerSpectrum.Extractor concurrent = null;
            var read = System.Threading.Tasks.Task.Run(() =>
            {
                GPU.SetDevice(0);
                concurrent = new CtfPowerSpectrum.Extractor(dims, options, fullSpectrum: true);
                concurrent.UseOwnStream();
                return concurrent.Extract(image, framesSummedBy: fit);
            });
            using (var copy = image.GetCopyGPU())
                for (int i = 0; i < 20; i++) using (var ft = copy.AsFFT()) { }
            var actual = read.GetAwaiter().GetResult();
            try
            {
                Assert.Equal(expected.Observations.Count, actual.Observations.Count);
                for (int p = 0; p < expected.Observations.Count; p++)
                {
                    var a = expected.Observations[p].Spectrum.Samples; var b = actual.Observations[p].Spectrum.Samples;
                    for (int i = 0; i < a.Length; i++) Assert.Equal(a[i].Power, b[i].Power);
                }
                Assert.Equal(expected.Display, actual.Display);
            }
            finally { concurrent?.Dispose(); }
        }
    }

    static void AssertSameSpectra(CtfPowerSpectrum.Extraction expected, CtfPowerSpectrum.Extraction actual)
    {
        Assert.Equal(expected.Observations.Count, actual.Observations.Count);
        Assert.Equal(expected.FourierSize, actual.FourierSize);
        for (int p = 0; p < expected.Observations.Count; p++)
        {
            var a = expected.Observations[p].Spectrum.Samples; var b = actual.Observations[p].Spectrum.Samples;
            Assert.Equal(a.Length, b.Length);
            // FP32 transforms round relative to the strongest Fourier components: errors are bounded by a small fraction
            // of the peak everywhere, and relative to the value wherever the power is not far below the peak.
            double peak = a.Max(s => s.Power), absolute = 0, relative = 0;
            for (int i = 0; i < a.Length; i++)
            {
                Assert.Equal(a[i].Count, b[i].Count);
                double error = Math.Abs(a[i].Power - b[i].Power);
                absolute = Math.Max(absolute, error / peak);
                if (a[i].Power >= 1e-3 * peak) relative = Math.Max(relative, error / a[i].Power);
            }
            Assert.True(absolute < 1e-6 && relative < 1e-4, $"patch {p}: error {absolute:E2} of peak, {relative:E2} of value");
        }
        double displayPeak = expected.Display.Max(v => Math.Abs(v)), displayAbsolute = 0, displayRelative = 0;
        for (int i = 0; i < expected.Display.Length; i++)
        {
            double error = Math.Abs(expected.Display[i] - actual.Display[i]);
            displayAbsolute = Math.Max(displayAbsolute, error / displayPeak);
            if (Math.Abs(expected.Display[i]) >= 1e-3 * displayPeak) displayRelative = Math.Max(displayRelative, error / Math.Abs(expected.Display[i]));
        }
        Assert.True(displayAbsolute < 1e-6 && displayRelative < 1e-4, $"display: error {displayAbsolute:E2} of peak, {displayRelative:E2} of value");
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
