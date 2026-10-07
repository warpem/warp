using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Warp;
using Warp.Tools;
using Xunit;
using Complex = System.Numerics.Complex;

namespace Tests;

public sealed class TemplateMatchCudaTheoryAttribute : TheoryAttribute
{
    public TemplateMatchCudaTheoryAttribute()
    {
        if (Environment.GetEnvironmentVariable("WARP_RUN_CUDA_TESTS") != "1")
            Skip = "Set WARP_RUN_CUDA_TESTS=1 on a CUDA host with rebuilt NativeAcceleration.";
    }
}

/// <summary>Opt-in tests of the native batch ABI, independent of managed batch orchestration.</summary>
public class TemplateMatchBatchCudaTests
{
    [DllImport("NativeAcceleration", EntryPoint = "TemplateMatchRefineBatchBfgs", CallingConvention = CallingConvention.Cdecl)]
    private static extern int RefineBatchBfgs(ulong textureRe, ulong textureIm,
        int dim, int box, int views, int particles, int hypotheses,
        IntPtr data, IntPtr ctf, IntPtr quadrature, IntPtr inverseNoise, IntPtr phaseRadii,
        [In] float[] geometry, [In] float[] bounds, [In] float[] symmetry, int symmetryCount,
        [In, Out] float[] poses, [In, Out] int[] seedIds,
        float pixel, float cutoff, float diameter, int maxIterations, float mergeDistance, float mergeAngle,
        [Out] double[] summary, [Out] int[] diagnostics, [Out] double[] tiltStats);

    private delegate int BatchRefiner(ulong textureRe, ulong textureIm,
        int dim, int box, int views, int particles, int hypotheses,
        IntPtr data, IntPtr ctf, IntPtr quadrature, IntPtr inverseNoise, IntPtr phaseRadii,
        float[] geometry, float[] bounds, float[] symmetry, int symmetryCount,
        float[] poses, int[] seedIds, float pixel, float cutoff, float diameter,
        int maxIterations, float mergeDistance, float mergeAngle,
        double[] summary, int[] diagnostics, double[] tiltStats);

    [TemplateMatchCudaFact]
    public void ThirtyTwoStartsMatchScalarFrozenGeometryAcrossParticlesAndViews()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(particles: 2);
            const int hypotheses = 32;
            float[] poses = Starts(2, hypotheses);
            Result result = fixture.Run(poses, Seeds(2 * hypotheses), 0);
            for (int particle = 0; particle < 2; particle++)
                for (int h = 0; h < hypotheses; h++)
                {
                    int lane = particle * hypotheses + h;
                    double[] scalar = fixture.Scalar(particle, poses.AsSpan(lane * 12, 12).ToArray());
                    double cross = 0, power = 0;
                    for (int t = 0; t < Fixture.Views; t++)
                    {
                        Near(result.TiltStats[(lane * Fixture.Views + t) * 2], scalar[t * 14], fixture.ScoreTolerance, "batch/scalar tilt C");
                        Near(result.TiltStats[(lane * Fixture.Views + t) * 2 + 1], scalar[t * 14 + 1], fixture.ScoreTolerance, "batch/scalar tilt P");
                        cross += scalar[t * 14];
                        power += scalar[t * 14 + 1];
                    }
                    Near(result.Summary[lane * 4], cross, fixture.ScoreTolerance, "shared cross");
                    Near(result.Summary[lane * 4 + 1], power, fixture.ScoreTolerance, "shared power");
                    Near(result.Summary[lane * 4 + 2], cross / Math.Sqrt(power), fixture.ScoreTolerance, "initial signed Z");
                    Assert.Equal(lane, result.Seeds[lane]);
                    Assert.Equal(-1, result.Diagnostics[lane * 4 + 3]);
                    for (int j = 0; j < 12; j++) Near(result.Poses[lane * 12 + j], poses[lane * 12 + j], 1e-7, "zero-step pose");
                }
            // The second particle deliberately has one unusable view. It must contribute no
            // evidence without disabling the other views or borrowing the first particle's mask.
            for (int h = 0; h < hypotheses; h++)
            {
                int offset = ((hypotheses + h) * Fixture.Views + 2) * 2;
                Assert.Equal(0, result.TiltStats[offset]);
                Assert.Equal(0, result.TiltStats[offset + 1]);
            }
        }
    }

    [TemplateMatchCudaFact]
    public void OptimizerLanesAreIndependentOfBatchOrderInactiveAndInvalidSlots()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(particles: 2);
            const int hypotheses = 32, iterations = 12;
            float[] poses = Starts(2, hypotheses);
            int[] seeds = Seeds(2 * hypotheses);
            foreach (int lane in new[] { 30, 62 }) poses[lane * 12] = float.NaN;
            foreach (int lane in new[] { 31, 63 })
            {
                seeds[lane] = -1;
                for (int j = 0; j < 12; j++) poses[lane * 12 + j] = float.NaN;
            }
            Result oneStep = fixture.Run(poses, seeds, 1);
            bool acceptedAny = false;
            for (int p = 0; p < 2; p++)
                for (int h = 0; h < 30; h++)
                {
                    int lane = p * hypotheses + h;
                    double translation2 = 0, rotationTrace = 0;
                    for (int j = 0; j < 3; j++)
                    {
                        double delta = oneStep.Poses[lane * 12 + j] - poses[lane * 12 + j];
                        translation2 += delta * delta;
                    }
                    for (int j = 3; j < 12; j++) rotationTrace += (double)oneStep.Poses[lane * 12 + j] * poses[lane * 12 + j];
                    double angle = Math.Acos(Math.Clamp((rotationTrace - 1) / 2, -1, 1));
                    Assert.True(Math.Sqrt(translation2) <= 2 * Fixture.Pixel + 1e-5, "A single physical translation step is capped at two stage pixels.");
                    Assert.True(angle <= 5 * Math.PI / 180 + 1e-5, "A single geodesic rotation step is capped at five degrees.");
                    Assert.InRange(oneStep.Diagnostics[lane * 4], 0, 1);
                    acceptedAny |= oneStep.Diagnostics[lane * 4] == 1;
                }
            Assert.True(acceptedAny, "The step-cap fixture must exercise an accepted update.");
            Result batch = fixture.Run(poses, seeds, iterations);
            for (int particle = 0; particle < 2; particle++)
            {
                int first = particle * hypotheses;
                foreach (int h in new[] { 0, 9, 29 })
                {
                    int lane = first + h;
                    Result alone = fixture.Run(poses.AsSpan(lane * 12, 12).ToArray(), new[] { seeds[lane] }, iterations, particle, 1);
                    CompareLane(batch, lane, alone, 0);
                    Assert.True(batch.Score(lane) >= batch.Summary[lane * 4 + 2] - 2e-5, "Accepted steps must preserve signed Z.");
                    AssertProperRotation(batch.Poses.AsSpan(lane * 12 + 3, 9).ToArray());
                    AssertFinalScore(fixture, batch, lane, particle);
                }
                Assert.Equal(-1, batch.Seeds[first + 30]);
                Assert.Equal(4, batch.Diagnostics[(first + 30) * 4 + 2]);
                Assert.Equal(-1, batch.Seeds[first + 31]);
                Assert.Equal(3, batch.Diagnostics[(first + 31) * 4 + 2]);
                for (int h = 30; h < 32; h++)
                {
                    Assert.Equal(-2, batch.Diagnostics[(first + h) * 4 + 3]);
                    for (int j = 0; j < Fixture.Views * 2; j++) Assert.Equal(0, batch.TiltStats[(first + h) * Fixture.Views * 2 + j]);
                }
            }
            float[] reversed = new float[poses.Length];
            int[] reversedSeeds = new int[seeds.Length];
            for (int p = 0; p < 2; p++)
                for (int h = 0; h < hypotheses; h++)
                {
                    int source = p * hypotheses + h, target = p * hypotheses + hypotheses - 1 - h;
                    Array.Copy(poses, source * 12, reversed, target * 12, 12);
                    reversedSeeds[target] = seeds[source];
                }
            Result permuted = fixture.Run(reversed, reversedSeeds, iterations);
            for (int p = 0; p < 2; p++)
                for (int h = 0; h < 30; h++) CompareLane(batch, p * hypotheses + h, permuted, p * hypotheses + hypotheses - 1 - h);
        }
    }

    [TemplateMatchCudaFact]
    public void NegativeSignedZStartsCanAscendInsteadOfClampingTheirGradientToZero()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            // A constant Fourier template makes rotation irrelevant and leaves an exact,
            // smooth translation objective. Negative data gives a negative initial Z.
            using var fixture = new Fixture(1, constantTemplate: true, amplitude: -1);
            float[] poses = new float[32 * 12];
            for (int h = 0; h < 32; h++) WritePose(poses, h, new float3(.3f + .012f * h, -.1f, .08f), new Matrix3());
            Result result = fixture.Run(poses, Seeds(32), 24);
            for (int h = 0; h < 32; h++)
            {
                Assert.True(result.Summary[h * 4 + 2] < -.1, "The fixture must actually start with negative Z.");
                Assert.True(result.Score(h) > result.Summary[h * 4 + 2] + .01, "Signed-Z refinement must escape a nonstationary negative start.");
                Assert.True(result.Diagnostics[h * 4] > 0, "At least one step must be accepted.");
            }
            AssertFinalScore(fixture, result, 0, 0);
        }
    }

    [TemplateMatchCudaFact]
    public void SymmetryMergeUsesRightObjectActionAndPreservesDistinctTranslations()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(1, constantTemplate: true);
            Matrix3 basis = Matrix3.Euler(.31f, .72f, -.23f);
            float[] poses = new float[32 * 12];
            int[] seeds = Enumerable.Repeat(-1, 32).ToArray();
            WritePose(poses, 0, new float3(0), basis);
            WritePose(poses, 1, new float3(.025f, 0, 0), basis * Matrix3.RotateZ(MathF.PI / 2));
            WritePose(poses, 2, new float3(.035f, 0, 0), basis);
            WritePose(poses, 3, new float3(.8f, 0, 0), basis * Matrix3.RotateZ(MathF.PI / 2));
            WritePose(poses, 4, new float3(.045f, 0, 0), basis * Matrix3.RotateZ(MathF.PI / 4));
            for (int h = 0; h < 5; h++) seeds[h] = 100 + h;
            Result c1 = fixture.Run(poses, seeds, 0, symmetry: Pack(new Matrix3()), mergeDistance: .1f, mergeAngle: .02f);
            Result octahedral = fixture.Run(poses, seeds, 0, symmetry: Octahedral(), mergeDistance: .1f, mergeAngle: .02f);
            Assert.Equal(new[] { 100, 101, 103, 104 }, c1.Seeds.Where(s => s >= 0).ToArray());
            Assert.Equal(new[] { 100, 103, 104 }, octahedral.Seeds.Where(s => s >= 0).ToArray());
            Assert.Equal(0, c1.Diagnostics[2 * 4 + 3]);
            Assert.Equal(0, octahedral.Diagnostics[1 * 4 + 3]);
            Assert.Equal(0, octahedral.Diagnostics[2 * 4 + 3]);
            // Merged hypotheses retain their own final statistics for diagnostics.
            Near(octahedral.Score(1), c1.Score(1), 1e-7, "merge preserves discarded score");
            Assert.True(octahedral.Score(0) > octahedral.Score(1));
            Assert.Equal(-1, octahedral.Diagnostics[3 * 4 + 3]);
        }
    }

    [TemplateMatchCudaFact]
    public void TranslationBoundsAreTotalBoundsIncludingFixedCoordinates()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(1, constantTemplate: true, truePosition: new float3(1.1f, 0, 0));
            float[] poses = new float[32 * 12];
            for (int h = 0; h < 32; h++) WritePose(poses, h, new float3(-.3f + .01f * h, 0, 0), new Matrix3());
            Result result = fixture.Run(poses, Seeds(32), 30, bounds: new[] { -.4f, 0, 0, .4f, 0, 0 });
            for (int h = 0; h < 32; h++)
            {
                Assert.InRange(result.Poses[h * 12], -.400001f, .400001f);
                Assert.InRange(result.Poses[h * 12], .30f, .400001f);
                Assert.Equal(0, result.Poses[h * 12 + 1]);
                Assert.Equal(0, result.Poses[h * 12 + 2]);
                Assert.True(result.Score(h) > result.Summary[h * 4 + 2] + .01);
            }
            AssertFinalScore(fixture, result, 0, 0);
        }
    }

    [TemplateMatchCudaFact]
    public void NonzeroAngularRefinementImprovesThirtyTwoStartsAndReportsFinalPoseScore()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(1);
            float3 position = new(.25f, -.18f, .12f);
            Matrix3 rotation = Matrix3.Euler(.21f, .54f, -.16f);
            float[] poses = new float[32 * 12], truth = new float[12];
            WritePose(truth, 0, position, rotation);
            for (int h = 0; h < 32; h++)
            {
                float direction = h % 2 == 0 ? 1 : -1;
                Matrix3 offset = Matrix3.RotateX(direction * (.10f + .002f * h)) *
                    Matrix3.RotateY(.055f * MathF.Sin(h + .3f)) * Matrix3.RotateZ(.07f * MathF.Cos(.7f * h));
                WritePose(poses, h, position, rotation * offset);
            }
            // Fix translations to remove the possibility of fitting orientation error with
            // motion. Recovery and score parity are separate assertions with separate tolerances.
            Result result = fixture.Run(poses, Seeds(32), 120,
                bounds: new[] { position.X, position.Y, position.Z, position.X, position.Y, position.Z });
            double[] oracle = fixture.Scalar(0, truth);
            double trueCross = Enumerable.Range(0, Fixture.Views).Sum(t => oracle[t * 14]);
            double truePower = Enumerable.Range(0, Fixture.Views).Sum(t => oracle[t * 14 + 1]);
            double trueZ = trueCross / Math.Sqrt(truePower);
            var errors = new List<double>();
            int improvedAngles = 0;
            for (int h = 0; h < 32; h++)
            {
                double initialError = RotationError(poses.AsSpan(h * 12 + 3, 9), truth.AsSpan(3, 9));
                double finalError = RotationError(result.Poses.AsSpan(h * 12 + 3, 9), truth.AsSpan(3, 9));
                Assert.True(initialError > 5 * Math.PI / 180, "The fixture must start far enough from the true orientation.");
                Assert.True(result.Score(h) >= result.Summary[h * 4 + 2] - 2e-5, "Angular refinement must preserve signed Z.");
                Assert.True(result.Score(h) <= trueZ + fixture.ScoreTolerance * (1 + Math.Abs(trueZ)), "The exact matched pose bounds the score.");
                if (finalError < initialError / 2) improvedAngles++;
                errors.Add(finalError);
                AssertProperRotation(result.Poses.AsSpan(h * 12 + 3, 9).ToArray());
                for (int j = 0; j < 3; j++) Near(result.Poses[h * 12 + j], truth[j], 1e-7, "fixed position during angular refinement");
                AssertFinalScore(fixture, result, h, 0);
            }
            errors.Sort();
            Assert.True(improvedAngles >= 24, $"Only {improvedAngles}/32 starts halved their angular error.");
            Assert.True(errors[0] < 2 * Math.PI / 180, $"Best angular error is {errors[0] * 180 / Math.PI:R} degrees.");
            Assert.True(errors[16] < 4 * Math.PI / 180, $"Median angular error is {errors[16] * 180 / Math.PI:R} degrees.");
        }
    }

    [TemplateMatchCudaFact]
    public void ZeroModelPowerRejectsOtherwiseValidStarts()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(1, zeroModel: true);
            Result result = fixture.Run(Starts(1, 32), Seeds(32), 20);
            for (int h = 0; h < 32; h++)
            {
                Assert.Equal(-1, result.Seeds[h]);
                Assert.Equal(4, result.Diagnostics[h * 4 + 2]);
                Assert.Equal(-2, result.Diagnostics[h * 4 + 3]);
                Assert.Equal(0, result.Diagnostics[h * 4]);
            }
            Assert.All(result.Summary, value => Assert.Equal(0, value));
            Assert.All(result.TiltStats, value => Assert.Equal(0, value));
        }
    }

    [TemplateMatchCudaFact]
    public void Fp32MergeResolvesAnglesNearTheProductionThreshold()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(1, constantTemplate: true);
            const float threshold = .0004f; // Approximately 0.023 degrees.
            Matrix3 basis = Matrix3.Euler(.31f, .72f, -.23f);
            float[] poses = new float[3 * 12];
            WritePose(poses, 0, new float3(0), basis);
            WritePose(poses, 1, new float3(0), basis * Matrix3.RotateZ(threshold * .5f));
            WritePose(poses, 2, new float3(0), basis * Matrix3.RotateZ(threshold * 2));
            Result result = fixture.Run(poses, Seeds(3), 0, mergeDistance: .025f, mergeAngle: threshold);
            Assert.Equal(2, result.Seeds.Count(seed => seed >= 0));
            Assert.Equal(1, result.Seeds.Take(2).Count(seed => seed >= 0));
            Assert.Equal(2, result.Seeds[2]);
        }
    }

    [TemplateMatchCudaTheory]
    [InlineData(-.08)]
    [InlineData(.2)]
    public void EnvelopeSpectraMatchScalarScoresAndRecoverJointParameters(double b)
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(2, envelopeB: b);
            float[] poses = new float[24];
            for (int p = 0; p < 2; p++)
                WritePose(poses, p, new float3(.25f, -.18f, .12f), Matrix3.Euler(.21f + p * .04f, .54f, -.16f));
            var spectra = fixture.Envelopes(poses, new[] { 0, 1 });
            for (int p = 0; p < 2; p++)
            {
                double[] scalar = fixture.Scalar(p, poses.AsSpan(p * 12, 12).ToArray());
                Near(spectra[p].Cross.Sum(v => (double)v), Enumerable.Range(0, Fixture.Views).Sum(t => scalar[t * 14]), 3e-5, "envelope/scalar C");
                Near(spectra[p].Power.Sum(v => (double)v), Enumerable.Range(0, Fixture.Views).Sum(t => scalar[t * 14 + 1]), 3e-5, "envelope/scalar P");
                var fit = TemplateMatchEnvelope.Fit(spectra[p], 20, -.15, .4);
                Near(fit.B, b, .001, "joint B recovery from independent Fourier data");
                Near(fit.Amplitude, 1.3 + .2 * p, .003, "joint amplitude recovery");
                // Evaluate a nonzero B independently by applying the envelope to
                // each CTF Fourier sample, without binning or the new GPU kernel.
                double trialB = b * .8;
                double[] shaped = fixture.Scalar(p, poses.AsSpan(p * 12, 12).ToArray(), trialB);
                double c = 0, power = 0;
                for (int i = 0; i < spectra[p].Power.Length; i++)
                {
                    double q2 = (double)i * spectra[p].MaximumFrequencySquared / (spectra[p].Power.Length - 1);
                    double e = Math.Exp(-trialB * (q2 - 20) / 4);
                    c += spectra[p].Cross[i] * e; power += spectra[p].Power[i] * e * e;
                }
                Near(c, Enumerable.Range(0, Fixture.Views).Sum(t => shaped[t * 14]), 3e-5, "nonzero-B C");
                Near(power, Enumerable.Range(0, Fixture.Views).Sum(t => shaped[t * 14 + 1]), 3e-5, "nonzero-B P");
            }
            // Inactive particles ignore their pose buffers; their output is zero.
            Array.Fill(poses, float.NaN, 12, 12);
            var inactive = fixture.Envelopes(poses, new[] { 0, -1 });
            Assert.All(inactive[1].Cross, v => Assert.Equal(0, v));
            Assert.All(inactive[1].Power, v => Assert.Equal(0, v));
        }
    }

    [TemplateMatchCudaFact]
    public void EnvelopeHighpassIsAppliedBeforeBinningWithStrictBoundaryAndIgnoresExcludedData()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(2, envelopeB: .2);
            float[] poses = new float[24];
            for (int p = 0; p < 2; p++)
                WritePose(poses, p, new float3(.25f, -.18f, .12f), Matrix3.Euler(.21f + p * .04f, .54f, -.16f));
            // 1.13f is exactly the fixture's x=1,y=0 physical q². Putting
            // NaNs at and below the boundary detects both <= errors and reads
            // of excluded low-frequency data, even in a straddling bin.
            const float cutoff = 1.13f;
            var before = fixture.Envelopes(poses, new[] { 0, 1 }, cutoff);
            fixture.PoisonDataAtOrBelow(cutoff);
            var after = fixture.Envelopes(poses, new[] { 0, 1 }, cutoff);
            for (int p = 0; p < 2; p++)
            {
                var scalar = fixture.Scalar(p, poses.AsSpan(p * 12, 12).ToArray(), minimumQ2: cutoff);
                Near(after[p].Cross.Sum(v => (double)v), Enumerable.Range(0, Fixture.Views).Sum(t => scalar[t * 14]), 3e-5, "strict band cross");
                Near(after[p].Power.Sum(v => (double)v), Enumerable.Range(0, Fixture.Views).Sum(t => scalar[t * 14 + 1]), 3e-5, "strict band power");
                Near(after[p].Cross.Sum(v => (double)v), before[p].Cross.Sum(v => (double)v), 3e-5, "excluded data invariance");
                var fit = TemplateMatchEnvelope.Fit(after[p], 20, -.15, .4);
                Near(fit.B, .2, .001, "band B recovery");
                Near(fit.Amplitude, 1.3 + .2 * p, .003, "band amplitude recovery");
            }
        }
    }

    [TemplateMatchCudaFact]
    public void PerTiltEnvelopeSpectraPreserveTiltIdentityAndSumToAggregate()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new Fixture(2, envelopeB: .2);
            float[] poses = new float[24];
            for (int p = 0; p < 2; p++)
                WritePose(poses, p, new float3(.25f, -.18f, .12f), Matrix3.Euler(.21f + p * .04f, .54f, -.16f));
            const float minimum = 1.13f;
            var aggregate = fixture.Envelopes(poses, new[] { 0, 1 }, minimum);
            var byTilt = fixture.Envelopes(poses, new[] { 0, 1 }, minimum, true);
            for (int p = 0; p < 2; p++)
            {
                var scalar = fixture.Scalar(p, poses.AsSpan(p * 12, 12).ToArray(), minimumQ2: minimum);
                for (int t = 0; t < Fixture.Views; t++)
                {
                    Near(byTilt[p * Fixture.Views + t].Cross.Sum(v => (double)v), scalar[t * 14], 3e-5, "per-tilt scalar C");
                    Near(byTilt[p * Fixture.Views + t].Power.Sum(v => (double)v), scalar[t * 14 + 1], 3e-5, "per-tilt scalar P");
                }
                for (int i = 0; i < aggregate[p].Power.Length; i++)
                {
                    Near(Enumerable.Range(0, Fixture.Views).Sum(t => (double)byTilt[p * Fixture.Views + t].Cross[i]), aggregate[p].Cross[i], 3e-5, "summed-bin C");
                    Near(Enumerable.Range(0, Fixture.Views).Sum(t => (double)byTilt[p * Fixture.Views + t].Power[i]), aggregate[p].Power[i], 3e-5, "summed-bin P");
                }
            }
            Array.Fill(poses, float.NaN, 12, 12);
            var inactive = fixture.Envelopes(poses, new[] { 0, -1 }, minimum, true);
            for (int t = 0; t < Fixture.Views; t++)
            {
                Assert.All(inactive[Fixture.Views + t].Cross, v => Assert.Equal(0, v));
                Assert.All(inactive[Fixture.Views + t].Power, v => Assert.Equal(0, v));
            }
        }
    }

    private sealed class Result
    {
        public float[] Poses;
        public int[] Seeds, Diagnostics;
        public double[] Summary, TiltStats;
        public double Score(int lane) => Summary[lane * 4] / Math.Sqrt(Summary[lane * 4 + 1]);
    }

    private sealed class Fixture : IDisposable
    {
        public const int Box = 16, Dim = 35, Views = 3;
        public const float Pixel = 1.7f, Cutoff = 6.7f;
        private const int Elements = Box * (Box / 2 + 1);
        private readonly int particles;
        private readonly BatchRefiner refine;
        public double ScoreTolerance { get; }
        private readonly float[] geometry;
        private readonly List<IntPtr> allocations = new();
        private readonly ulong[] textures = new ulong[2], arrays = new ulong[2];
        private readonly IntPtr data, ctf, quad, weights, radii;
        private readonly float[] hostCtf, hostQuad, hostRadii, hostWeights;

        public Fixture(int particles, bool constantTemplate = false, float amplitude = 1.3f, float3? truePosition = null, bool zeroModel = false, double envelopeB = 0)
        {
            this.particles = particles;
            refine = RefineBatchBfgs;
            // BFGS reduces its sufficient statistics
            // in FP32; the scalar score oracle continues to use FP64 reductions.
            ScoreTolerance = 3e-5;
            geometry = new float[particles * Views * 18];
            float[] volume = new float[(Dim / 2 + 1) * Dim * Dim * 2];
            for (int iz = 0; iz < Dim; iz++)
                for (int iy = 0; iy < Dim; iy++)
                    for (int ix = 0; ix <= Dim / 2; ix++)
                    {
                        int y = iy <= Dim / 2 ? iy : iy - Dim, z = iz <= Dim / 2 ? iz : iz - Dim;
                        int index = ((iz * Dim + iy) * (Dim / 2 + 1) + ix) * 2;
                        double envelope = Math.Exp(-.006 * (ix * ix + 1.4 * y * y + 1.8 * z * z));
                        double phase = .31 * ix + .17 * y - .23 * z;
                        volume[index] = constantTemplate ? 1 : (float)(envelope * (1 + .13 * Math.Cos(phase)));
                        volume[index + 1] = constantTemplate ? 0 : (float)(.2 * envelope * Math.Sin(phase));
                    }
            GPU.CreateTexture3DComplex(Upload(volume), new int3(Dim / 2 + 1, Dim, Dim), textures, arrays, false);
            float[] observed = new float[particles * Views * Elements * 2];
            float[] baseCtf = new float[particles * Views * Elements], quadrature = new float[baseCtf.Length];
            float[] inverseNoise = new float[baseCtf.Length], phaseRadii = new float[baseCtf.Length];
            for (int p = 0; p < particles; p++)
                for (int t = 0; t < Views; t++)
                {
                    int offset = (p * Views + t) * 18;
                    float tilt = -.65f + .6f * t + .08f * p;
                    float[] matrix = Pack((Matrix3.RotateZ(.07f + .03f * p) * Matrix3.RotateY(tilt)).Transposed());
                    for (int j = 0; j < 9; j++) geometry[offset + j] = matrix[j] * 2;
                    // A nonorthogonal in-plane map catches assuming G is only a rotation.
                    if (!constantTemplate) { geometry[offset] *= 1.025f; geometry[offset + 3] += .035f; }
                    float[] shifts = { MathF.Cos(tilt), .035f, -.06f, 1, MathF.Sin(tilt), .08f };
                    for (int j = 0; j < shifts.Length; j++) shifts[j] /= Pixel;
                    Array.Copy(shifts, 0, geometry, offset + 9, 6);
                    if (!constantTemplate)
                    {
                        geometry[offset + 15] = .0012f * (t + 1);
                        geometry[offset + 16] = -.0007f;
                        geometry[offset + 17] = .0018f;
                    }
                    float[] truth = new float[12];
                    WritePose(truth, 0, truePosition ?? (constantTemplate ? new float3(0) : new float3(.25f, -.18f, .12f)),
                        constantTemplate ? new Matrix3() : Matrix3.Euler(.21f + p * .04f, .54f, -.16f));
                    FrozenMapping(geometry.AsSpan(offset, 18), truth, out float[] projection, out float[] shift, out float beta);
                    for (int row = 0; row < Box; row++)
                        for (int x = 0; x <= Box / 2; x++)
                        {
                            int y = row <= Box / 2 ? row : row - Box;
                            int local = row * (Box / 2 + 1) + x, id = (p * Views + t) * Elements + local;
                            baseCtf[id] = constantTemplate ? 1 : .8f * MathF.Sin(.17f * local + .31f * t + .1f * p);
                            quadrature[id] = constantTemplate ? 0 : -.8f * MathF.Cos(.17f * local + .31f * t + .1f * p);
                            if (zeroModel) baseCtf[id] = quadrature[id] = 0;
                            inverseNoise[id] = p == 1 && t == 2 ? 0 : .7f + .2f * p + .1f * t + .13f * MathF.Cos(.09f * local);
                            phaseRadii[id] = 1.13f * x * x + .83f * y * y + .07f * x * y;
                            double phase = beta * phaseRadii[id];
                            double transfer = baseCtf[id] * Math.Cos(phase) + quadrature[id] * Math.Sin(phase);
                            Complex value = CpuProject(volume, projection, shift, x, y) * transfer * (amplitude + (constantTemplate ? 0 : .2 * p));
                            value *= Math.Exp(-envelopeB * (phaseRadii[id] - 20) / 4);
                            observed[id * 2] = (float)value.Real;
                            observed[id * 2 + 1] = (float)value.Imaginary;
                        }
                }
            hostCtf = baseCtf; hostQuad = quadrature; hostRadii = phaseRadii; hostWeights = inverseNoise;
            data = Upload(observed); ctf = Upload(baseCtf); quad = Upload(quadrature);
            weights = Upload(inverseNoise); radii = Upload(phaseRadii);
        }

        private IntPtr Upload(float[] values)
        {
            IntPtr pointer = GPU.MallocDeviceFromHost(values, values.Length);
            allocations.Add(pointer);
            return pointer;
        }

        public Result Run(float[] inputPoses, int[] seeds, int iterations, int firstParticle = 0, int? count = null,
            float[] symmetry = null, float mergeDistance = 0, float mergeAngle = 0, float[] bounds = null)
        {
            int particleCount = count ?? particles, hypotheses = seeds.Length / particleCount;
            Assert.Equal(particleCount * hypotheses * 12, inputPoses.Length);
            var result = new Result { Poses = (float[])inputPoses.Clone(), Seeds = (int[])seeds.Clone(),
                Summary = new double[seeds.Length * 4], Diagnostics = new int[seeds.Length * 4],
                TiltStats = new double[seeds.Length * Views * 2] };
            symmetry ??= Pack(new Matrix3());
            bounds ??= Enumerable.Range(0, particleCount).SelectMany(_ => new[] { -3f, -3f, -3f, 3f, 3f, 3f }).ToArray();
            int offset = firstParticle * Views * Elements * sizeof(float);
            Assert.Equal(0, refine(textures[0], textures[1], Dim, Box, Views, particleCount, hypotheses,
                IntPtr.Add(data, offset * 2), IntPtr.Add(ctf, offset), IntPtr.Add(quad, offset),
                IntPtr.Add(weights, offset), IntPtr.Add(radii, offset),
                geometry.AsSpan(firstParticle * Views * 18, particleCount * Views * 18).ToArray(), bounds,
                symmetry, symmetry.Length / 9, result.Poses, result.Seeds,
                Pixel, Cutoff, 16, iterations, mergeDistance, mergeAngle, result.Summary, result.Diagnostics, result.TiltStats));
            return result;
        }

        public void PoisonDataAtOrBelow(float minimumQ2)
        {
            float[] values = new float[hostRadii.Length * 2];
            GPU.CopyDeviceToHost(data, values, values.Length);
            for (int i = 0; i < hostRadii.Length; i++)
                if (hostRadii[i] <= minimumQ2) values[2 * i] = values[2 * i + 1] = float.NaN;
            GPU.CopyHostToDevice(values, data, values.Length);
        }

        public TemplateMatchEnvelopeSpectrum[] Envelopes(float[] poses, int[] active, float minimumQ2 = 0, bool byTilt = false)
        {
            int bins = TemplateMatchEnvelope.SpectrumBins;
            float maximum = hostRadii.Max();
            int count = particles * (byTilt ? Views : 1);
            float[] spectra = new float[count * 2 * bins];
            int status = byTilt ? GPU.TemplateMatchEnvelopeSpectraByTilt(textures[0], textures[1], Dim, Box, Views, particles,
                data, ctf, quad, weights, radii, geometry, poses, active, Cutoff, bins, minimumQ2, maximum, spectra)
                : GPU.TemplateMatchEnvelopeSpectra(textures[0], textures[1], Dim, Box, Views, particles,
                data, ctf, quad, weights, radii, geometry, poses, active, Cutoff, bins, minimumQ2, maximum, spectra);
            Assert.Equal(0, status);
            return Enumerable.Range(0, count).Select(p => new TemplateMatchEnvelopeSpectrum(maximum,
                spectra.AsSpan(p * 2 * bins, bins).ToArray(), spectra.AsSpan((p * 2 + 1) * bins, bins).ToArray(), minimumQ2)).ToArray();
        }

        public double[] Scalar(int particle, float[] pose, double envelopeB = 0, float minimumQ2 = 0)
        {
            int offset = particle * Views * Elements * sizeof(float);
            IntPtr context = IntPtr.Zero;
            try
            {
                IntPtr evalCtf = ctf, evalQuad = quad, evalWeights = weights;
                if (minimumQ2 > 0)
                    evalWeights = Upload(hostWeights.Select((value, i) => hostRadii[i] > minimumQ2 ? value : 0).ToArray());
                if (envelopeB != 0)
                {
                    float[] shapedCtf = new float[hostCtf.Length], shapedQuad = new float[hostQuad.Length];
                    for (int i = 0; i < shapedCtf.Length; i++)
                    {
                        double e = Math.Exp(-envelopeB * (hostRadii[i] - 20) / 4);
                        shapedCtf[i] = (float)(hostCtf[i] * e); shapedQuad[i] = (float)(hostQuad[i] * e);
                    }
                    evalCtf = Upload(shapedCtf); evalQuad = Upload(shapedQuad);
                }
                Assert.Equal(0, GPU.TemplateMatchRefineCreate(textures[0], textures[1], Dim, Box, Views,
                    IntPtr.Add(data, offset * 2), IntPtr.Add(evalCtf, offset), IntPtr.Add(evalQuad, offset),
                    IntPtr.Add(evalWeights, offset), IntPtr.Add(radii, offset), out context));
                float[] matrices = new float[Views * 9], shifts = new float[Views * 2], beta = new float[Views];
                for (int t = 0; t < Views; t++)
                {
                    FrozenMapping(geometry.AsSpan((particle * Views + t) * 18, 18), pose, out float[] m, out float[] s, out beta[t]);
                    Array.Copy(m, 0, matrices, t * 9, 9); Array.Copy(s, 0, shifts, t * 2, 2);
                }
                double[] output = new double[Views * 14];
                Assert.Equal(0, GPU.TemplateMatchRefineEvaluate(context, matrices, new float[Views * 54], shifts,
                    new float[Views * 12], beta, new float[Views * 6], Cutoff, output));
                return output;
            }
            finally { if (context != IntPtr.Zero) GPU.TemplateMatchRefineDestroy(context); }
        }

        public void Dispose()
        {
            for (int i = 0; i < 2; i++) if (textures[i] != 0) GPU.DestroyTexture(textures[i], arrays[i]);
            foreach (IntPtr pointer in allocations) GPU.FreeDevice(pointer);
        }
    }

    // Independently materialize R^T G, physical-to-pixel shifts and the depth phase.
    private static void FrozenMapping(ReadOnlySpan<float> geometry, ReadOnlySpan<float> pose,
        out float[] matrix, out float[] shift, out float beta)
    {
        matrix = new float[9]; shift = new float[2]; beta = 0;
        for (int column = 0; column < 3; column++)
            for (int row = 0; row < 3; row++)
                for (int k = 0; k < 3; k++) matrix[row + 3 * column] += pose[3 + k + 3 * row] * geometry[k + 3 * column];
        for (int k = 0; k < 3; k++)
        {
            shift[0] += geometry[9 + k * 2] * pose[k];
            shift[1] += geometry[10 + k * 2] * pose[k];
            beta += geometry[15 + k] * pose[k];
        }
    }

    // Host tensor-product interpolation is independent of both native scorers.
    private static Complex CpuProject(float[] volume, float[] matrix, float[] shift, int kx, int ky)
    {
        double x = (double)matrix[0] * kx + (double)matrix[3] * ky;
        double y = (double)matrix[1] * kx + (double)matrix[4] * ky;
        double z = (double)matrix[2] * kx + (double)matrix[5] * ky;
        bool conjugate = x < 0;
        if (conjugate) { x = -x; y = -y; z = -z; }
        if (x >= Fixture.Dim / 2) return Complex.Zero;
        int ix = (int)Math.Floor(x), iy = (int)Math.Floor(y), iz = (int)Math.Floor(z);
        double fx = x - ix, fy = y - iy, fz = z - iz;
        Complex value = Complex.Zero;
        for (int dz = 0; dz < 2; dz++)
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                {
                    int wy = ((iy + dy) % Fixture.Dim + Fixture.Dim) % Fixture.Dim;
                    int wz = ((iz + dz) % Fixture.Dim + Fixture.Dim) % Fixture.Dim;
                    int index = ((wz * Fixture.Dim + wy) * (Fixture.Dim / 2 + 1) + ix + dx) * 2;
                    double weight = (dx == 0 ? 1 - fx : fx) * (dy == 0 ? 1 - fy : fy) * (dz == 0 ? 1 - fz : fz);
                    value += new Complex(volume[index], volume[index + 1]) * weight;
                }
        if (conjugate) value = Complex.Conjugate(value);
        double phase = -2 * Math.PI * ((double)kx * shift[0] + (double)ky * shift[1]) / Fixture.Box;
        return value * Complex.FromPolarCoordinates(1, phase);
    }

    private static float[] Starts(int particles, int hypotheses)
    {
        float[] result = new float[particles * hypotheses * 12];
        for (int p = 0; p < particles; p++)
            for (int h = 0; h < hypotheses; h++)
                WritePose(result, p * hypotheses + h,
                    new float3(-.6f + .041f * h, .37f * MathF.Sin(h + .2f), -.3f + .019f * h),
                    Matrix3.Euler(.11f + .013f * h, .43f + .004f * h + p * .06f, -.29f + .006f * h));
        return result;
    }

    private static int[] Seeds(int count) => Enumerable.Range(0, count).ToArray();
    private static void WritePose(float[] target, int lane, float3 position, Matrix3 rotation)
    {
        target[lane * 12] = position.X; target[lane * 12 + 1] = position.Y; target[lane * 12 + 2] = position.Z;
        Array.Copy(Pack(rotation), 0, target, lane * 12 + 3, 9);
    }
    private static float[] Pack(Matrix3 m) => new[] { m.M11, m.M21, m.M31, m.M12, m.M22, m.M32, m.M13, m.M23, m.M33 };

    // All 24 proper signed permutation matrices; no native symmetry helper is used.
    private static float[] Octahedral()
    {
        var result = new List<float>();
        foreach (int[] permutation in new[] { new[] { 0, 1, 2 }, new[] { 0, 2, 1 }, new[] { 1, 0, 2 }, new[] { 1, 2, 0 }, new[] { 2, 0, 1 }, new[] { 2, 1, 0 } })
            for (int signs = 0; signs < 8; signs++)
            {
                float[] matrix = new float[9];
                for (int c = 0; c < 3; c++) matrix[permutation[c] + 3 * c] = (signs & (1 << c)) == 0 ? 1 : -1;
                if (Determinant(matrix) > 0) result.AddRange(matrix);
            }
        Assert.Equal(24 * 9, result.Count);
        return result.ToArray();
    }
    private static double Determinant(float[] m) => m[0] * (m[4] * m[8] - m[7] * m[5]) -
        m[3] * (m[1] * m[8] - m[7] * m[2]) + m[6] * (m[1] * m[5] - m[4] * m[2]);
    private static void AssertProperRotation(float[] m)
    {
        Near(Determinant(m), 1, 1e-4, "rotation determinant");
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
            {
                double dot = 0;
                for (int k = 0; k < 3; k++) dot += m[k + 3 * a] * m[k + 3 * b];
                Near(dot, a == b ? 1 : 0, 1e-4, "rotation orthogonality");
        }
    }
    private static double RotationError(ReadOnlySpan<float> first, ReadOnlySpan<float> second)
    {
        double trace = 0;
        for (int j = 0; j < 9; j++) trace += (double)first[j] * second[j];
        return Math.Acos(Math.Clamp((trace - 1) / 2, -1, 1));
    }
    private static void AssertFinalScore(Fixture fixture, Result result, int lane, int particle)
    {
        double[] scalar = fixture.Scalar(particle, result.Poses.AsSpan(lane * 12, 12).ToArray());
        double cross = 0, power = 0;
        for (int t = 0; t < Fixture.Views; t++)
        {
            cross += scalar[t * 14];
            power += scalar[t * 14 + 1];
            Near(result.TiltStats[(lane * Fixture.Views + t) * 2], scalar[t * 14], fixture.ScoreTolerance, "final-pose scalar tilt C");
            Near(result.TiltStats[(lane * Fixture.Views + t) * 2 + 1], scalar[t * 14 + 1], fixture.ScoreTolerance, "final-pose scalar tilt P");
        }
        Near(result.Summary[lane * 4], cross, fixture.ScoreTolerance, "final-pose scalar C");
        Near(result.Summary[lane * 4 + 1], power, fixture.ScoreTolerance, "final-pose scalar P");
        Near(result.Score(lane), cross / Math.Sqrt(power), fixture.ScoreTolerance, "final-pose scalar signed Z");
    }
    private static void CompareLane(Result a, int ia, Result b, int ib)
    {
        Assert.Equal(a.Seeds[ia], b.Seeds[ib]);
        for (int j = 0; j < 12; j++) Near(a.Poses[ia * 12 + j], b.Poses[ib * 12 + j], 2e-5, "independent pose");
        for (int j = 0; j < 4; j++) Near(a.Summary[ia * 4 + j], b.Summary[ib * 4 + j], 2e-5, "independent sufficient statistics");
        for (int j = 0; j < 3; j++) Assert.Equal(a.Diagnostics[ia * 4 + j], b.Diagnostics[ib * 4 + j]);
        for (int j = 0; j < Fixture.Views * 2; j++) Near(a.TiltStats[ia * Fixture.Views * 2 + j], b.TiltStats[ib * Fixture.Views * 2 + j], 2e-5, "independent tilt statistics");
    }
    private static void Near(double actual, double expected, double tolerance, string label)
    {
        Assert.True(double.IsFinite(actual) && Math.Abs(actual - expected) <= tolerance * (1 + Math.Abs(expected)),
            $"{label}: {actual:R} != {expected:R}");
    }
}
