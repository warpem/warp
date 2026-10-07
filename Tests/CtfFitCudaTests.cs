using System;
using System.Collections.Generic;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfFitCudaTests
{
    static CtfSpectrumFit Spectrum(int variant)
    {
        var samples = new List<CtfSpectrumFit.Sample>();
        var rng = new Random(117 + variant);
        var options = new ProcessingOptionsMovieCTF { PixelSize = 1.5M, Voltage = 300, Cs = 2.7M, Amplitude = .07M };
        var ctf = CtfFitEngine.MakeCtf(options, 2.3 + variant * .013, .04, -.025, .2);
        for (int r = 18; r < 130; r++) for (int a = 0; a < 12; a++)
            {
                double q = r / 768.0, q2 = q * q, angle = (a + .3) * Math.PI / 12;
                double amplitude = ctf.Get2D(new[] { new float2((float)(q * 1.5), (float)angle) }, false, true, true)[0];
                double power = 3 + 5 * q + (1 + 2 * q) * amplitude * amplitude + .2 * (rng.NextDouble() - .5);
                if (variant == 1 && r == 64) power += 20;
                samples.Add(new(q2, q2 * q2, q2 * Math.Cos(2 * angle), q2 * Math.Sin(2 * angle), power, 35 + variant * 10));
            }
        return new(samples.ToArray(), 300, 2.7, .07);
    }
    // FP32 trigonometry and profiled solves need looser arithmetic parity than the CPU
    // reference; parameter recovery below is independently bounded to sub-Å defocus.
    static void Close(double expected, double actual, double relative = 1e-4)
        => Assert.True(Math.Abs(actual - expected) <= relative * Math.Max(1, Math.Abs(expected)), $"Expected {expected:G17}, actual {actual:G17}, relative tolerance {relative:G6}");

    [CtfCudaFact]
    public void BatchedTiltSeedsMatchIndependentSearchesAndUnequalMovieFrameGroups()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var options = new ProcessingOptionsMovieCTF { Voltage = 300, Cs = 2.7M, Amplitude = .07M, ZMin = 1, ZMax = 4, DoPhase = true };
            var groups = Enumerable.Range(0, 3).Select(g => Enumerable.Range(0, 3).Select(i =>
            {
                var samples = Spectrum(g).Samples.Select(s => s with { Count = s.Count * (i == 1 ? 3 : 2) }).ToArray();
                return new CtfPowerSpectrum.Observation(new CtfSpectrumFit(samples, 300, 2.7, .07), new float3(.5f), i);
            }).ToArray()).ToArray();
            var offsets = groups.Select(g => new double[g.Length]).ToArray();
            var batched = CtfFitEngine.InitializeMany(groups, offsets, options);
            for (int g = 0; g < groups.Length; g++)
            {
                var single = CtfFitEngine.Initialize(groups[g], offsets[g], options);
                Close(single.Defocus, batched[g].Defocus, 1e-7);
                Close(single.Phase, batched[g].Phase, 1e-7);
                var cpu = CtfCpuReference.Initialize(groups[g], offsets[g], options);
                Assert.InRange(Math.Abs(cpu.Defocus-batched[g].Defocus), 0, .002);
            }
        }
    }

    [CtfCudaFact]
    public void CoarseSearchMatchesIndependentCpuScoresWithGeometryOffsetsAndPhase()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var spectra = new[] { Spectrum(0), Spectrum(1), Spectrum(2) };
            using var batch = new CtfThinGpuBatch(spectra);
            double[] offsets = { -.17, 0, .23 };
            var trials = new List<double>();
            for (double df = .1; df < 15; df += .071)
                for (int phase = 0; phase < 12; phase++) { trials.Add(df); trials.Add(phase * Math.PI / 12); }
            var actual = batch.Search(trials.ToArray(), offsets);
            for (int i = 0; i < trials.Count / 2; i++) for (int k = 0; k < spectra.Length; k++)
                Close(spectra[k].QuickScore(trials[2*i] + offsets[k], trials[2*i+1]), actual[i*spectra.Length+k], 3e-4);
            // Recreating a batch after IRLS must preserve current weights, while rebuilding baseline variance weights.
            double[] poses = { 2.3, .04, -.025, .2, 2.313, .04, -.025, .2, 2.326, .04, -.025, .2 };
            batch.Evaluate(poses, true);
            batch.SynchronizeWeights();
            using var resumed = new CtfThinGpuBatch(spectra);
            var expected = (double[])batch.Evaluate(poses).Clone();
            var restored = resumed.Evaluate(poses);
            for (int i = 0; i < expected.Length; i++) Close(expected[i], restored[i], 1e-6);
        }
    }

    [CtfCudaFact]
    public void GlobalGpuInitializationAndRefinementRecoverNoisyCtfAcrossSearchRange()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            foreach (var truth in new[] { (.7, 0.0, .035), (2.3, .37, .035), (2.3, .37, .12), (4.8, 0.0, .035) })
                CtfSpectrumFitTests.CheckGlobalRecovery(truth.Item1, truth.Item2, truth.Item3, true);
        }
    }

    [CtfCudaFact]
    public void ProfiledScoresGradientsAndRobustWeightsMatchCpuIncludingActiveBounds()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var spectra = new[] { Spectrum(0), Spectrum(1), Spectrum(2) };
            using var batch = new CtfThinGpuBatch(spectra);
            double[] poses = { 2.307, .037, -.021, .22, 2.318, .04, -.023, .21, 2.326, .04, -.025, .2 + Math.PI / 2 };
            for (int pass = 0; pass < 3; pass++)
            {
                double[] actual = (double[])batch.Evaluate(poses).Clone();
                for (int i = 0; i < spectra.Length; i++)
                {
                    var expected = spectra[i].Evaluate(poses[4 * i], poses[4 * i + 1], poses[4 * i + 2], poses[4 * i + 3]);
                    Close(expected.Loss, actual[6 * i]);
                    for (int j = 0; j < 4; j++) Close(expected.Gradient[j], actual[6 * i + 1 + j], 5e-3);
                }
                // Differentiate the GPU objective itself, including the constrained nuisance refit.
                for (int j = 0; j < 4; j++)
                {
                    // Resolve finite differences above FP32 rounding while keeping phase changes small.
                    const double h = 2e-4;
                    poses[j] += h; double plus = batch.Evaluate(poses)[0];
                    poses[j] -= 2 * h; double minus = batch.Evaluate(poses)[0]; poses[j] += h;
                    Close((plus - minus) / (2 * h), actual[1 + j], 1e-2);
                }
                actual = (double[])batch.Evaluate(poses, true).Clone();
                for (int i = 0; i < spectra.Length; i++)
                    Close(spectra[i].Reweight(poses[4 * i], poses[4 * i + 1], poses[4 * i + 2], poses[4 * i + 3]), actual[6 * i + 5], 2e-5);
            }
            batch.SynchronizeWeights();
            var final = batch.Evaluate(poses);
            for (int i = 0; i < spectra.Length; i++)
                Close(spectra[i].Evaluate(poses[4 * i], poses[4 * i + 1], poses[4 * i + 2], poses[4 * i + 3]).Loss, final[6 * i]);
        }
    }

    [CtfCudaFact]
    public void CpuSteeredGpuOptimizationRecoversSameSolutionAsCpuReference()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var options = new ProcessingOptionsMovieCTF { ZMin = 1, ZMax = 4, DoPhase = true };
            var geometry = new[] { new CtfFitGeometry(new[] { 1.0 }, new[] { 1.0 }) };
            var initial = new[] { 2.31, .035, -.02, .23 };
            CtfPowerSpectrum.Observation[] Records() => new[] { new CtfPowerSpectrum.Observation(Spectrum(0), new float3(.5f), 0) };
            var cpu = CtfCpuReference.Refine(Records(), geometry, initial, options);
            var gpu = CtfThinReference.Refine(Records(), geometry, initial, options);
            for (int i = 0; i < 3; i++) Assert.InRange(Math.Abs(cpu.Parameters[i] - gpu.Parameters[i]) * 1e4, 0, .5);
            Assert.InRange(Math.Abs(cpu.Parameters[3] - gpu.Parameters[3]), 0, 1e-3);
            Close(cpu.Loss, gpu.Loss, 1e-4);
        }
    }


    [CtfCudaFact]
    public void WeakAndLowDefocusSpectraRemainFiniteAndAgreeWithCpu()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            foreach (double df in new[] { .1, .3, .7, 6.0, 15.0 })
            {
                var samples = Spectrum(0).Samples.Select(s => s with
                {
                    Power = 5 + 2 * Math.Sqrt(s.Q2) + .01 * Math.Sin(2 * Math.PI * CtfSpectrumFit.Wavelength(300) * 1e4 * s.Q2 * df)
                }).ToArray();
                var spectrum = new CtfSpectrumFit(samples, 300, 2.7, .07);
                using var batch = new CtfThinGpuBatch(new[] { spectrum });
                var cpu = spectrum.Evaluate(df, 0, 0, 0);
                var gpu = batch.Evaluate(new[] { df, 0.0, 0.0, 0.0 });
                Assert.All(gpu, v => Assert.True(double.IsFinite(v), $"Nonfinite objective at {df} µm"));
                Close(cpu.Loss, gpu[0]);
            }
        }
    }

    [CtfCudaFact]
    public void DiagnosticsReuseAcceptedGpuSplinesAndMatchCpuRefits()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var options = new ProcessingOptionsMovieCTF { PixelSize = 1.5M, Voltage = 300, Cs = 2.7M, Amplitude = .07M, ZMin = 1, ZMax = 4, DoPhase = true };
            var records = new[] { new CtfPowerSpectrum.Observation(Spectrum(0), new float3(.5f), 0), new CtfPowerSpectrum.Observation(Spectrum(1), new float3(.5f), 1) };
            var geometry = new[] { new CtfFitGeometry(new[] { 1.0, 0.0 }, new[] { 1.0 }), new CtfFitGeometry(new[] { 0.0, 1.0 }, new[] { 1.0 }) };
            var fit = CtfThinReference.Refine(records, geometry, new[] { 2.31, 2.32, .035, -.02, .23 }, options);
            var p = fit.Parameters;
            var references = Enumerable.Range(0, 2).Select(i => CtfFitEngine.MakeCtf(options, p[i], p[2], p[3], p[4])).ToArray();
            var global = CtfFitEngine.MakeCtf(options, (p[0] + p[1]) * .5, p[2], p[3], p[4]);
            const int window = 256, fft = 512;
            var display = new[] { Enumerable.Repeat(100f, window * window / 2).ToArray(), Enumerable.Repeat(100f, window * window / 2).ToArray() };
            var result = CtfFitDiagnostics.Create(records, geometry, fit, new[] { 0, 1 }, references, global, fft, window, display);
            Assert.Equal(fft/2,result.Global.Quality.Length);
            foreach(var diagnostic in result.Groups.Append(result.Global))
                Assert.Equal(CtfFitDiagnostics.EstimateResolution(diagnostic.Quality, diagnostic == result.Global ? global : references[Array.IndexOf(result.Groups,diagnostic)]),diagnostic.Resolution);
            var expectedGlobalSum = new double[fft / 2]; var expectedGlobalWeight = new double[fft / 2];
            double kd = Math.PI * CtfSpectrumFit.Wavelength(300) * 1e4, kc = -.5 * Math.PI * 2.7 * 1e7 * Math.Pow(CtfSpectrumFit.Wavelength(300), 3);
            void Accumulate(double[] sum, double[] weights, CtfSpectrumFit.Sample sample, double bg, double env, double scale, double df, CTF reference)
            {
                if (env < 1e-8) return;
                double target = kd * (sample.Q2 * df + sample.AstigX * p[2] + sample.AstigY * p[3]) + kc * sample.Q4 + p[4] - (double)reference.PhaseShift * Math.PI;
                double q2 = sample.Q2, refDf = (double)reference.Defocus;
                for (int k = 0; k < 8; k++) q2 -= (kd * refDf * q2 + kc * q2 * q2 - target) / (kd * refDf + 2 * kc * q2);
                if (!(q2 > 0)) return;
                double r = Math.Sqrt(q2) * 1.5 * fft; int b = (int)r;
                if (b < 0 || b >= sum.Length - 1) return;
                double w = sample.Count * env * env, v = (sample.Power / scale - bg) / env, f = r - b;
                sum[b] += w * v * (1 - f); weights[b] += w * (1 - f); sum[b + 1] += w * v * f; weights[b + 1] += w * f;
            }
            for (int i = 0; i < 2; i++)
            {
                var spectrum = records[i].Spectrum;
                var e = spectrum.Evaluate(p[i], p[2], p[3], p[4], true);
                var sum = new double[fft / 2]; var weight = new double[fft / 2];
                var bg = new double[window / 2]; var env = new double[window / 2]; var count = new double[window / 2];
                for (int j = 0; j < spectrum.Samples.Length; j++)
                {
                    var sample = spectrum.Samples[j];
                    Accumulate(sum, weight, sample, e.Background[j], e.Envelope[j], spectrum.PowerScale, p[i], references[i]);
                    Accumulate(expectedGlobalSum, expectedGlobalWeight, sample, e.Background[j], e.Envelope[j], spectrum.PowerScale, p[i], global);
                    int b = (int)(Math.Sqrt(sample.Q2) * 1.5 * window);
                    if (b < bg.Length) { bg[b] += e.Background[j] * spectrum.PowerScale * sample.Count; env[b] += e.Envelope[j] * spectrum.PowerScale * sample.Count; count[b] += sample.Count; }
                }
                for (int b = 0; b < sum.Length; b++) Close(weight[b] > 0 ? sum[b] / weight[b] : 0, result.Groups[i].Spectrum[b].Y, 3e-3);
                for (int b = 0; b < bg.Length; b++) if (count[b] > 0) { bg[b] /= count[b]; env[b] /= count[b]; }
                double floor = env.Max() * 1e-3;
                for (int y = 0; y < window / 2; y++) for (int x = 0; x < window; x++)
                    {
                        int xx = x - window / 2, yy = window / 2 - 1 - y, b = (int)Math.Sqrt(xx * xx + yy * yy);
                        double expected = b < bg.Length && count[b] > 0 && env[b] > floor ? (100 - bg[b]) / env[b] : 0;
                        Close(expected, display[i][y * window + x], 3e-3);
                    }
            }
            for (int b = 0; b < expectedGlobalSum.Length; b++) Close(expectedGlobalWeight[b] > 0 ? expectedGlobalSum[b] / expectedGlobalWeight[b] : 0, result.Global.Spectrum[b].Y, 3e-3);
            // A nearly absent high-frequency envelope must not amplify residuals into huge plot spikes.
            var weakCoefficients = (float[])fit.Coefficients.Clone();
            int stride = weakCoefficients.Length / records.Length, knots = stride / 2;
            for (int i = 0; i < records.Length; i++)
            {
                Array.Fill(weakCoefficients, 1e-7f, i * stride + knots, knots);
                weakCoefficients[i * stride + knots] = 1;
            }
            var weak = CtfFitDiagnostics.Create(records, geometry, fit with { Coefficients = weakCoefficients }, new[] { 0, 1 }, references, global,
                fft, window, new[] { new float[window * window / 2], new float[window * window / 2] });
            for (int b = 100; b < 125; b++) Assert.Equal(0, weak.Global.Spectrum[b].Y);
            // Movie groups combine into one reference/display rather than one display per frame group.
            var movie = CtfFitDiagnostics.Create(records, geometry, fit, new[] { 0, 0 }, new[] { global }, global, fft, window, new[] { new float[window * window / 2] });
            Assert.Same(movie.Groups[0], movie.Global);
            for (int b = 0; b < expectedGlobalSum.Length; b++) Close(result.Global.Spectrum[b].Y, movie.Global.Spectrum[b].Y, 1e-4);
        }
    }

    [CtfCudaFact]
    public void CudaExtractionMatchesCpuWindowingAndFourierBinning()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            const int window = 64, fft = 128;
            using var image = new Image(new int3(window, window, 1));
            var rng = new Random(918);
            float[] source = image.GetHost(Intent.Write)[0];
            for (int i = 0; i < source.Length; i++) source[i] = (float)(rng.NextDouble() * 3 + 30);
            var options = new ProcessingOptionsMovieCTF { Window = window, PixelSize = 4, RangeMin = .15M, RangeMax = .8M, ZMin = .1M, ZMax = .8M, Voltage = 300, Cs = 2.7M, Amplitude = .07M };
            var result = CtfPowerSpectrum.Extract(image, options);
            Assert.Equal(fft, result.FourierSize);
            float[] hann = Enumerable.Range(0, window).Select(x => (float)(.5 - .5 * Math.Cos(2 * Math.PI * (x + .5) / window))).ToArray();
            double mean = 0, norm = 0;
            for (int y = 0; y < window; y++) for (int x = 0; x < window; x++) { double w = hann[x] * hann[y]; mean += source[y * window + x] * w; norm += w; }
            mean /= norm;
            using var padded = new Image(new int3(fft, fft, 1));
            float[] pixels = padded.GetHost(Intent.Write)[0];
            for (int y = 0; y < window; y++) for (int x = 0; x < window; x++) pixels[(y + 32) * fft + x + 32] = (float)((source[y * window + x] - mean) * hann[x] * hann[y]);
            using var fourier = padded.AsFFT();
            float[] values = fourier.GetHost(Intent.Read)[0];
            var powers = new double[24 * fft / 2]; var counts = new int[powers.Length];
            for (int y = 0; y < fft; y++) for (int x = 0; x <= fft / 2; x++)
                {
                    int yy = y <= fft / 2 ? y : y - fft;
                    if (x == 0 && yy < 0) continue;
                    double r = Math.Sqrt(x * x + yy * yy), q = r / (fft * 4.0), angle = Math.Atan2(yy, x);
                    if (q < .15 / 8 || q >= .8 / 8 || r >= fft / 2) continue;
                    if (angle < 0) angle += Math.PI;
                    int b = Math.Min(23, (int)(angle * 24 / Math.PI)) * (fft / 2) + (int)r, i = y * (fft / 2 + 1) + x;
                    powers[b] += (double)values[2 * i] * values[2 * i] + (double)values[2 * i + 1] * values[2 * i + 1]; counts[b]++;
                }
            int sample = 0;
            for (int b = 0; b < powers.Length; b++) if (counts[b] > 0)
                    Close(powers[b] / counts[b], result.Observations[0].Spectrum.Samples[sample++].Power, 1e-5);
            Assert.Equal(sample, result.Observations[0].Spectrum.Samples.Length);
            source = image.GetHost(Intent.Write)[0]; source[window * window / 2] = float.NaN;
            Assert.Throws<InvalidOperationException>(() => CtfPowerSpectrum.Extract(image, options));
        }
    }
}
