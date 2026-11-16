using System;
using System.Collections.Generic;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfSignalTests
{
    const double Pixel = .934, Voltage = 300, Cs = 2.7, Amplitude = .07, ZMin = .3, ZMax = 4;
    const int Fft = 1024, Sectors = 24;

    /// <summary>A pooled fitting-band spectrum as the extractor lays it out: radial bins in 24 angular sectors.
    /// Log power has Gaussian noise that shrinks with the pixels in a bin, as averaged periodograms do.</summary>
    static CtfSpectrumFit.Sample[] Spectrum(double defocus, double ax, double ay, double contrast, int seed, Func<double, double> modulation = null)
    {
        double lambda = CtfSpectrumFit.Wavelength(Voltage), kd = Math.PI * lambda * 1e4, kc = -.5 * Math.PI * Cs * 1e7 * lambda * lambda * lambda;
        var random = new Random(seed); var samples = new List<CtfSpectrumFit.Sample>();
        for (int r = (int)(Fft * Pixel / 25); r <= (int)(Fft * Pixel / 4.2); r++)
            for (int s = 0; s < Sectors; s++)
            {
                double q = r / (Fft * Pixel), q2 = q * q, angle = (s + .5) * Math.PI / Sectors, count = Math.PI * r / Sectors;
                double astigX = q2 * Math.Cos(2 * angle), astigY = q2 * Math.Sin(2 * angle);
                double gamma = kd * (q2 * defocus + astigX * ax + astigY * ay) + kc * q2 * q2 + Math.Asin(Amplitude);
                double background = Math.Exp(-12 * q) + .3, rings = contrast * Math.Exp(-60 * q2) * (modulation?.Invoke(q2) ?? 1) * Math.Pow(Math.Sin(gamma), 2);
                double gaussian = Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
                samples.Add(new(q2, q2 * q2, astigX, astigY, background * (1 + rings) * Math.Exp(gaussian / Math.Sqrt(20 * count)), count));
            }
        return samples.ToArray();
    }

    static CTF Fit(double defocus, double ax, double ay) => new()
    {
        PixelSize = (decimal)Pixel, Voltage = (decimal)Voltage, Cs = (decimal)Cs, Amplitude = (decimal)Amplitude, Defocus = (decimal)defocus,
        DefocusDelta = (decimal)(2 * Math.Sqrt(ax * ax + ay * ay)), DefocusAngle = (decimal)(.5 * Math.Atan2(ay, ax) * 180 / Math.PI)
    };

    [Theory]
    [InlineData(1.9, .03, -.02)]
    [InlineData(1.2, .25, .1)]    // strong astigmatism: decoys may choose astigmatism too, yet the true fit stands out
    public void ThonRingsScoreFarAboveTheThreshold(double defocus, double ax, double ay)
    {
        var samples = Spectrum(defocus, ax, ay, .15, 1);
        double right = CtfSignal.Score(samples, Fit(defocus, ax, ay), ZMin, ZMax), wrong = CtfSignal.Score(samples, Fit(defocus + .7, ax, ay), ZMin, ZMax);
        Assert.True(right > 2 * CtfSignal.DefaultThreshold, $"right {right}, wrong {wrong}");
        // The same rings do not vouch for a fit at another defocus.
        Assert.True(wrong < CtfSignal.DefaultThreshold, $"right {right}, wrong {wrong}");
    }

    [Fact]
    public void ThickIceIsScoredWithItsSlabModulation()
    {
        // 1000–2000 Å of ice: the slab modulation's first zero falls between 9 and 6 Å, and ring contrast flips beyond it.
        // Scored with that modulation, as the fit models it, such spectra score higher than with plain sin²γ.
        double kd = Math.PI * CtfSpectrumFit.Wavelength(Voltage) * 1e4, with = 0, without = 0;
        foreach (double thickness in new[] { .1, .15, .2 })
            foreach (int seed in new[] { 1, 2 })
            {
                double Slab(double q2) => CtfSlabModel.ModulationValue(kd * q2, thickness * thickness, 0, 0);
                var samples = Spectrum(1.9, .03, -.02, .15, seed, Slab);
                double score = CtfSignal.Score(samples, Fit(1.9, .03, -.02), ZMin, ZMax, samples.Select(s => Slab(s.Q2)).ToArray());
                Assert.True(score > CtfSignal.DefaultThreshold, $"{thickness * 1e4:F0} Å, seed {seed}: {score}");
                with += score; without += CtfSignal.Score(samples, Fit(1.9, .03, -.02), ZMin, ZMax);
            }
        Assert.True(with > without + 4, $"with {with}, without {without}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void AFitToNoiseScoresBelowTheThreshold(int seed)
    {
        // Without rings, choose the defocus and astigmatism that match the noise best, as a fit does.
        var samples = Spectrum(1.9, 0, 0, 0, seed);
        var spectrum = CtfSignal.Spectrum.Create(samples, Fit(1, 0, 0));
        var steps = new[] { -.45, -.27, -.09, .09, .27, .45 };
        var best = Best(spectrum, (from i in Enumerable.Range(0, 741) from x in steps from y in steps select (ZMin + i * .005, x, y)).ToArray());
        for (int round = 0; round < 3; round++)
        {
            var b = best;
            best = Best(spectrum, (from d in Enumerable.Range(-8, 17) from x in Enumerable.Range(-5, 11) from y in Enumerable.Range(-5, 11)
                select (b.Defocus + d * .0005, Math.Clamp(b.X + x * .02, -.5, .5), Math.Clamp(b.Y + y * .02, -.5, .5))).ToArray());
        }
        double score = CtfSignal.Score(samples, Fit(best.Defocus, best.X, best.Y), ZMin, ZMax);
        Assert.True(score < CtfSignal.DefaultThreshold - 1, $"score {score}");
    }

    static (double Defocus, double X, double Y) Best(CtfSignal.Spectrum spectrum, (double, double, double)[] candidates)
    {
        var values = new double[candidates.Length];
        System.Threading.Tasks.Parallel.For(0, candidates.Length, i => values[i] = spectrum.Correlation(candidates[i].Item1, candidates[i].Item2, candidates[i].Item3));
        return candidates[Array.IndexOf(values, values.Max())];
    }

    [Fact]
    public void TooNarrowADefocusRangeLeavesTooFewDecoys()
    {
        var samples = Spectrum(1.9, 0, 0, .05, 1);
        Assert.True(double.IsNaN(CtfSignal.Score(samples, Fit(1.9, 0, 0), 1.6, 2.2)));
        Assert.False(double.IsNaN(CtfSignal.Score(samples, Fit(1.9, 0, 0), 1, 2.8)));
    }

    [Fact]
    public void AstigmatismMatchesTheFitsParametrization()
    {
        var options = new ProcessingOptionsMovieCTF { Voltage = 300, Cs = 2.7m, Amplitude = .07m };
        var ctf = CtfFitEngine.MakeCtf(options, 1.9, .12, -.05, 0);
        var (x, y) = CtfSignal.Astigmatism(ctf);
        Assert.Equal(.12, x, 6);
        Assert.Equal(-.05, y, 6);
    }

    [Fact]
    public void PoolingPatchesAveragesTheirPower()
    {
        var a = Spectrum(1.9, 0, 0, .05, 1); var b = Spectrum(1.9, 0, 0, .05, 2);
        var pooled = a.Select((s, i) => s with { Power = (s.Power + b[i].Power) / 2 }).ToArray();
        var patches = new[] { a, b }.Select(s => new CtfPowerSpectrum.Observation(new CtfSpectrumFit(s, Voltage, Cs, Amplitude), new float3(.5f), 0)).ToArray();
        Assert.Equal(CtfSignal.Score(pooled, Fit(1.9, 0, 0), ZMin, ZMax), CtfSignal.Score(patches, Fit(1.9, 0, 0), ZMin, ZMax), 9);
    }
}
