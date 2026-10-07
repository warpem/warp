using System;
using System.Collections.Generic;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfSpectrumFitTests
{
    static CtfSpectrumFit.Sample[] Spectrum(double defocus, double ax, double ay, double phase)
    {
        var result = new List<CtfSpectrumFit.Sample>();
        double lambda = CtfSpectrumFit.Wavelength(300);
        for (int r = 18; r < 110; r++) for (int sector = 0; sector < 12; sector++)
            {
                double q = r / 768.0, q2 = q * q, angle = (sector + .37) * Math.PI / 12;
                double u = q2 * Math.Cos(2 * angle), v = q2 * Math.Sin(2 * angle);
                double gamma = Math.PI * lambda * 1e4 * (q2 * defocus + u * ax + v * ay) - .5 * Math.PI * 2.7e7 * Math.Pow(lambda, 3) * q2 * q2 + Math.Asin(.07) + phase;
                double power = 3 + 7 * q + (1 + 2 * q) * Math.Pow(Math.Sin(gamma), 2);
                result.Add(new(q2, q2 * q2, u, v, power, 30));
            }
        return result.ToArray();
    }
    [Fact]
    public void ProfiledGradientIncludesAstigmatismPhaseAndNuisanceRefitting()
    {
        var fit = new CtfSpectrumFit(Spectrum(2.3, .04, -.025, .2), 300, 2.7, .07);
        double[] x = { 2.307, .037, -.021, .22 };
        var e = fit.Evaluate(x[0], x[1], x[2], x[3]);
        for (int i = 0; i < 4; i++)
        {
            const double h = 1e-6;
            x[i] += h; double upper = fit.Evaluate(x[0], x[1], x[2], x[3]).Loss;
            x[i] -= 2 * h; double lower = fit.Evaluate(x[0], x[1], x[2], x[3]).Loss; x[i] += h;
            double fd = (upper - lower) / (2 * h);
            Assert.True(Math.Abs(fd - e.Gradient[i]) < 1e-4 * Math.Max(1, Math.Abs(fd)), $"parameter {i}: {e.Gradient[i]} vs {fd}");
        }
    }
    [Fact]
    public void JointFitRecoversCtfWithoutFixingBackgroundOrEnvelope()
    {
        var fit = new CtfSpectrumFit(Spectrum(2.3, .04, -.025, .2), 300, 2.7, .07);
        var result = CtfFitOptimizer.Minimize(x => { var e = fit.Evaluate(x[0], x[1], x[2], x[3]); return (e.Loss, e.Gradient); },
            new[] { 2.31, .035, -.02, .23 }, new[] { .02, .02, .02, .1 }, new[] { .5, -.5, -.5, 0.0 }, new[] { 5.0, .5, .5, Math.PI });
        double[] truth = { 2.3, .04, -.025, .2 };
        for (int i = 0; i < 4; i++) Assert.InRange(Math.Abs(result.Parameters[i] - truth[i]), 0, 2e-4);
    }
    [Fact]
    public void PowerModelUsesWarpsDefocusAndPhaseConvention()
    {
        var ctf = new CTF { Voltage = 300, Cs = 2.7M, Amplitude = .07M, Defocus = 2.3M, PhaseShift = .2M };
        double q = .083, lambda = CtfSpectrumFit.Wavelength(300);
        double gamma = Math.PI * lambda * 23000 * q * q - .5 * Math.PI * 2.7e7 * Math.Pow(lambda, 3) * Math.Pow(q, 4) + Math.Asin(.07) + .2 * Math.PI;
        Assert.InRange(Math.Abs(Math.Pow(ctf.Get1DDouble(q, false, true, true), 2) - Math.Pow(Math.Sin(gamma), 2)), 0, 1e-10);
    }

    [Fact]
    public void AstigmatismOutputUsesFullPrincipalDefocusDifference()
    {
        var options = new ProcessingOptionsMovieCTF { PixelSize = 1.5M, Voltage = 300, Cs = 2.7M, Amplitude = .07M };
        var ctf = CtfFitEngine.MakeCtf(options, 2.3, .04, -.025, .2);
        var samples = Spectrum(2.3, .04, -.025, .2);
        var fit = new CtfSpectrumFit(samples, 300, 2.7, .07);
        var model = fit.Evaluate(2.3, .04, -.025, .2, true).Model;
        for (int i = 0; i < samples.Length; i += 17)
        {
            var v = samples[i]; double angle = .5 * Math.Atan2(v.AstigY, v.AstigX);
            var amplitude = ctf.Get2D(new[] { new float2((float)(Math.Sqrt(v.Q2) * (double)ctf.PixelSize), (float)angle) }, false, true, true)[0];
            Assert.InRange(Math.Abs(amplitude * amplitude - model[i]), 0, 2e-5);
        }
    }

    [Fact]
    public void TiltPlaneAndGridChainRuleMatchesFiniteDifferences()
    {
        var geometry = new CtfFitGeometry(new[] { .2, .8 }, new[] { .7, .3 }, .12, -.08,
            Matrix3.Euler(0, 0, .42f) * Matrix3.Euler(0, .91f, 0));
        double[] p = { 2.1, 2.4, .035, -.021, .17, .23, .035, -.044 };
        var truth = geometry.Evaluate(p);
        var spectrum = new CtfSpectrumFit(Spectrum(truth.Defocus + .004, .04, -.025, truth.Phase + .02), 300, 2.7, .07);
        var e = spectrum.Evaluate(truth.Defocus, p[2], p[3], truth.Phase);
        var gradient = new double[p.Length]; geometry.Accumulate(gradient, e.Gradient, truth.SlopeX, truth.SlopeY);
        double Loss() { var q = geometry.Evaluate(p); return spectrum.Evaluate(q.Defocus, p[2], p[3], q.Phase).Loss; }
        for (int j = 0; j < p.Length; j++)
        {
            const double h = 1e-6; p[j] += h; double a = Loss(); p[j] -= 2 * h; double b = Loss(); p[j] += h;
            double fd = (a - b) / (2 * h);
            Assert.InRange(Math.Abs(fd - gradient[j]) / Math.Max(1, Math.Abs(fd)), 0, 1e-4);
        }
    }

    [Fact]
    public void CubicGridWeightsUseActualPatchAndFrameCoordinates()
    {
        var dims = new int3(3, 2, 4); var positions = new[] { new float3(.13f, .28f, .125f), new float3(.87f, .72f, .875f) };
        var values = new float[dims.Elements()]; for (int i = 0; i < values.Length; i++) values[i] = (float)Math.Sin(i);
        using var grid = new CubicGrid(dims, values);
        var actual = grid.GetInterpolated(positions); var weights = CtfFitGeometry.GridWeights(dims, positions);
        for (int i = 0; i < positions.Length; i++)
        {
            double sum = 0, unity = 0; for (int j = 0; j < values.Length; j++) { sum += weights[i][j] * values[j]; unity += weights[i][j]; }
            Assert.InRange(Math.Abs(sum - actual[i]), 0, 2e-6); Assert.InRange(Math.Abs(unity - 1), 0, 2e-6);
        }
    }

    [Theory]
    [InlineData(.7, 0.0, .035)]
    [InlineData(2.3, .37, .035)]
    [InlineData(2.3, .37, .12)]
    [InlineData(4.8, 0.0, .035)]
    public void GlobalSearchAndJointFitRecoverNoisyCtf(double df, double phase, double astigX)
        => CheckGlobalRecovery(df, phase, astigX, false);

    internal static void CheckGlobalRecovery(double df, double phase, double astigX, bool gpu)
    {
        var options = new ProcessingOptionsMovieCTF { PixelSize = 1.5M, Voltage = 300, Cs = 2.7M, Amplitude = .07M, ZMin = .5M, ZMax = 6M, DoPhase = phase > 0 };
        var rng = new Random(103);
        var ctf = CtfFitEngine.MakeCtf(options, df, astigX, -.025, phase);
        var samples = new List<CtfSpectrumFit.Sample>();
        for (int r = 20; r < 160; r++) for (int a = 0; a < 24; a++)
            {
                double q = r / 768.0, q2 = q * q, angle = (a + .5) * Math.PI / 24;
                double signed = ctf.Get2D(new[] { new float2((float)(q * 1.5), (float)angle) }, false, true, true)[0];
                double mean = 2 + 5 * q + Math.Exp(-30 * q2) * signed * signed;
                double noise = Math.Sqrt(-2 * Math.Log(rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
                samples.Add(new(q2, q2 * q2, q2 * Math.Cos(2 * angle), q2 * Math.Sin(2 * angle), Math.Max(.01, mean * (1 + noise / 30)), 900));
            }
        var spectrum = new CtfSpectrumFit(samples.ToArray(), 300, 2.7, .07);
        var records = new[] { new CtfPowerSpectrum.Observation(spectrum, new float3(.5f), 0) };
        var cpuSeed = CtfCpuReference.Initialize(records, new double[1], options);
        var seed = gpu ? CtfFitEngine.Initialize(records, new double[1], options) : cpuSeed;
        if (gpu) Assert.InRange(Math.Abs(seed.Defocus - cpuSeed.Defocus), 0, .002);
        var geometry = new[] { new CtfFitGeometry(new[] { 1.0 }, new[] { 1.0 }) };
        var initial = new[] { seed.Defocus, 0.0, 0.0, seed.Phase };
        var fit = gpu ? CtfFitEngine.Refine(records, geometry, initial, options) : CtfCpuReference.Refine(records, geometry, initial, options);
        Assert.InRange(Math.Abs(fit.Parameters[0] - df), 0, .005);
        Assert.InRange(Math.Abs(fit.Parameters[1] - astigX), 0, .003);
        Assert.InRange(Math.Abs(fit.Parameters[2] + .025), 0, .003);
        Assert.InRange(Math.Abs(fit.Parameters[3] - phase), 0, .04);
    }

    [Fact]
    public void ReweightedProfileGradientRemainsAnalytic()
    {
        var samples = Spectrum(2.3, .04, -.025, .2);
        for (int i = 0; i < samples.Length; i += 31) samples[i] = samples[i] with { Power = samples[i].Power * 5 };
        var spectrum = new CtfSpectrumFit(samples, 300, 2.7, .07);
        spectrum.Reweight(2.3, .04, -.025, .2);
        const double h = 1e-6;
        var e = spectrum.Evaluate(2.302, .04, -.025, .2);
        double fd = (spectrum.Evaluate(2.302 + h, .04, -.025, .2).Loss - spectrum.Evaluate(2.302 - h, .04, -.025, .2).Loss) / (2 * h);
        Assert.InRange(Math.Abs(e.Gradient[0] - fd) / Math.Max(1, Math.Abs(fd)), 0, 1e-4);
    }

    [Fact]
    public void RobustPassesRejectStrongNarrowSpectrumContamination()
    {
        var samples = Spectrum(2.3, .04, -.025, .2);
        for (int i = 0; i < samples.Length; i++)
        {
            double q = Math.Sqrt(samples[i].Q2);
            samples[i] = samples[i] with { Power = samples[i].Power + 40 * Math.Exp(-.5 * Math.Pow((q - .083) / .001, 2)), Count = 500 };
        }
        var spectrum = new CtfSpectrumFit(samples, 300, 2.7, .07);
        double[] initial = { 2.305, .04, -.025, .2 };
        var ordinary = CtfFitOptimizer.Minimize(p => { var e = spectrum.Evaluate(p[0], p[1], p[2], p[3]); return (e.Loss, e.Gradient); }, initial,
            new[] { .02, .02, .02, .1 }, new[] { 1.0, -.5, -.5, 0 }, new[] { 4.0, .5, .5, Math.PI });
        var options = new ProcessingOptionsMovieCTF { ZMin = 1, ZMax = 4, DoPhase = true };
        var robust = CtfCpuReference.Refine(new[] { new CtfPowerSpectrum.Observation(spectrum, new float3(.5f), 0) },
            new[] { new CtfFitGeometry(new[] { 1.0 }, new[] { 1.0 }) }, initial, options);
        double before = Math.Abs(ordinary.Parameters[0] - 2.3), after = Math.Abs(robust.Parameters[0] - 2.3);
        Assert.True(after < before, $"Robust {after} vs ordinary {before}");
        Assert.InRange(after, 0, .005);
    }

    [Fact]
    public void PositiveEnvelopeRejectsAnticorrelatedRings()
    {
        var spectrum = new CtfSpectrumFit(Spectrum(2.3, .04, -.025, .2), 300, 2.7, .07);
        var correct = spectrum.Evaluate(2.3, .04, -.025, .2);
        var flipped = spectrum.Evaluate(2.3, .04, -.025, .2 + Math.PI / 2, true);
        Assert.True(flipped.Loss > correct.Loss + 1);
        Assert.All(flipped.Envelope, v => Assert.True(v >= 0));
    }
}
