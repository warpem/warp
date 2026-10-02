using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

/// <summary>Small synthetic integration tests of the actual managed candidate refinement entry point.</summary>
public class TemplateMatchPipelineCudaTests
{
    [TemplateMatchCudaFact]
    public void TemplateMaskPreservesTheSpecifiedParticleRadiusAtBothStages()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            MethodInfo method = typeof(TiltSeries).GetMethod("MaskMatchTemplate", BindingFlags.Static | BindingFlags.NonPublic);
            Assert.NotNull(method);
            foreach (bool coarse in new[] { true, false })
            {
                float pixel = coarse ? 10 : 5;
                using Image volume = new(new int3(64));
                volume.Fill(1);
                method.Invoke(null, new object[] { volume, 130f, pixel, coarse });
                float[] slice = volume.GetHost(Intent.Read)[32];
                // The old extra division by two attenuated the ferritin shell around 50–65 A.
                Assert.InRange(slice[32 * 64 + 32 + (int)(60 / pixel)], 0.99999f, 1.00001f);
                float edge = coarse ? 50 : 20;
                for (int x = 0; x <= 24; x++)
                {
                    double fraction = Math.Clamp((x * pixel - 65) / edge, 0, 1);
                    double expected = 0.5 * (1 + Math.Cos(Math.PI * fraction));
                    Assert.True(Math.Abs(slice[32 * 64 + 32 + x] - expected) < 2e-6);
                }
            }
        }
    }

    [TemplateMatchCudaFact]
    public void KnownPoseMatchesIndependentForwardModelAndSharedNoiseScaling()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new PipelineFixture(addNoise: false);
            Start[] starts = { new(fixture.TruePosition, fixture.TrueRotation, 17) };
            // Zero steps intentionally isolates scoring from optimizer accuracy. The private entry point
            // accepts this even though the command line requires positive refinement iteration counts.
            Solution[] result = Refine(fixture, starts, 0, out int usable);
            Assert.Equal(PipelineFixture.Views, usable);
            Assert.Single(result);
            Assert.Equal(0, result[0].Iterations);
            Near(result[0].Amplitude, PipelineFixture.Amplitude, 0.003, "known-pose amplitude");
            double power = 0;
            for (int t = 0; t < PipelineFixture.Views; t++)
            {
                Near(result[0].Statistics[t * 14], PipelineFixture.Amplitude * fixture.OraclePower[t], 0.003, "tilt cross");
                Near(result[0].Statistics[t * 14 + 1], fixture.OraclePower[t], 0.003, "tilt model power");
                power += fixture.OraclePower[t];
            }
            double expectedZ = PipelineFixture.Amplitude * Math.Sqrt(power);
            Near(result[0].Z, expectedZ, 0.003, "known-pose Z");
            Near(result[0].Gain, 0.5 * expectedZ * expectedZ, 0.006, "known-pose profile gain");

            // Four times the inverse variance doubles Z and quadruples gain, while one common
            // profiled amplitude stays unchanged. This also exercises a second scorer context.
            foreach (float[] tiltWeights in fixture.NoiseWeights)
                for (int i = 0; i < tiltWeights.Length; i++) tiltWeights[i] *= 4;
            Solution scaled = Assert.Single(Refine(fixture, starts, 0, out _));
            Near(scaled.Amplitude, result[0].Amplitude, 1e-5, "noise scaling amplitude");
            Near(scaled.Z, 2 * result[0].Z, 1e-5, "noise scaling Z");
            Near(scaled.Gain, 4 * result[0].Gain, 1e-5, "noise scaling gain");
        }
    }

    [TemplateMatchCudaFact]
    public void NearbyStartsRecoverKnownPoseIndependentlyWithFixedPatchNoise()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new PipelineFixture(addNoise: true);
            Start[] starts =
            {
                new(fixture.Anchor + new float3(-0.8f, 0.5f, -0.6f),
                    fixture.TrueRotation * Matrix3.RotateX(0.08f) * Matrix3.RotateY(-0.06f) * Matrix3.RotateZ(0.05f), 11),
                new(fixture.Anchor + new float3(1.2f, -0.8f, 0.4f),
                    fixture.TrueRotation * Matrix3.RotateX(-0.06f) * Matrix3.RotateY(0.08f) * Matrix3.RotateZ(-0.04f), 22)
            };
            Solution[] initial = Refine(fixture, starts, 0, out _);
            Solution[] refined = Refine(fixture, starts, 90, out int usable);
            Assert.Equal(PipelineFixture.Views, usable);
            Assert.Equal(2, refined.Length);
            Assert.Equal(11, refined[0].StartIndex);
            Assert.Equal(22, refined[1].StartIndex);
            for (int i = 0; i < refined.Length; i++)
            {
                Assert.True(refined[i].Z >= initial[i].Z - 1e-5, "Refinement must not lower its starting score.");
                Assert.InRange(refined[i].Iterations, 1, 90);
                Assert.All(refined[i].Statistics, value => Assert.True(double.IsFinite(value)));
            }
            Solution best = refined[0].Z >= refined[1].Z ? refined[0] : refined[1];
            // Recovery tolerances are deliberately looser than scoring equivalence: the optimizer
            // sees single-precision image geometry, trilinear Fourier interpolation and seeded noise.
            Assert.True((best.Position - fixture.TruePosition).Length() < 0.6f,
                $"Position error is {(best.Position - fixture.TruePosition).Length():R} Angstrom.");
            Assert.True(RotationError(best.Rotation, fixture.TrueRotation) < 2 * Math.PI / 180,
                $"Rotation error is {RotationError(best.Rotation, fixture.TrueRotation) * 180 / Math.PI:R} degrees.");
            Near(best.Amplitude, PipelineFixture.Amplitude, 0.02, "recovered amplitude");

            // Running the second start alone must reproduce its result in a batch. This detects
            // optimizer state, extraction or sufficient statistics leaking from the previous start.
            Solution alone = Assert.Single(Refine(fixture, new[] { starts[1] }, 90, out _));
            Assert.True((alone.Position - refined[1].Position).Length() < 1e-3f);
            Assert.True(RotationError(alone.Rotation, refined[1].Rotation) < 0.001);
            Near(alone.Z, refined[1].Z, 1e-6, "independent-start score");
        }
    }

    [TemplateMatchCudaFact]
    public void EnvelopeDiagnosticUsesManagedPhysicalFrequenciesAndPreservesPoseAndScore()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using var fixture = new PipelineFixture(addNoise: false);
            Start[] starts = { new(fixture.TruePosition, fixture.TrueRotation, 17) };
            Solution plain = Assert.Single(Refine(fixture, starts, 0, out _));
            Solution diagnostic = Assert.Single(Refine(fixture, starts, 0, out _, true, 0));
            Assert.Equal(plain.Position, diagnostic.Position);
            Assert.Equal(plain.Z, diagnostic.Z);
            Assert.NotNull(diagnostic.Envelope);
            var spectrum = diagnostic.Envelope;
            Assert.InRange(spectrum.MaximumFrequencySquared, .02f, .08f);
            double cross = 0, power = 0;
            for (int t = 0; t < PipelineFixture.Views; t++)
            { cross += plain.Statistics[t * 14]; power += plain.Statistics[t * 14 + 1]; }
            double spectrumC = 0, spectrumP = 0;
            foreach (float v in spectrum.Cross) spectrumC += v;
            foreach (float v in spectrum.Power) spectrumP += v;
            Near(spectrumC, cross, 3e-5, "managed envelope cross");
            Near(spectrumP, power, 3e-5, "managed envelope power");
            var fit = TemplateMatchEnvelope.Fit(spectrum, spectrum.MeanFrequencySquared());
            Assert.InRange(Math.Abs(fit.B), 0, 10);
            Near(fit.Amplitude, PipelineFixture.Amplitude, .005, "joint physical amplitude");
            // Production default selects q > 1/30 A, without modifying the
            // optimizer's scores or pose. The synthetic data have B=0.
            Solution band = Assert.Single(Refine(fixture, starts, 0, out _, true, exportTilts: true));
            Assert.Equal(plain.Z, band.Z);
            Assert.Equal(plain.Position, band.Position);
            Assert.Equal(1.0 / 900, band.Envelope.MinimumFrequencySquared);
            Assert.True(band.Envelope.MeanFrequencySquared() > 1.0 / 900);
            double step = band.Envelope.MaximumFrequencySquared / (band.Envelope.Power.Length - 1.0);
            for (int i = 0; (i + 1) * step <= band.Envelope.MinimumFrequencySquared; i++)
                Assert.Equal(0, band.Envelope.Power[i]);
            var bandFit = TemplateMatchEnvelope.Fit(band.Envelope, band.Envelope.MeanFrequencySquared());
            Assert.InRange(Math.Abs(bandFit.B), 0, 10);
            Near(bandFit.Amplitude, PipelineFixture.Amplitude, .005, "band physical amplitude");
            Assert.True(bandFit.GainAtZeroB < plain.Gain);
            Assert.Equal(PipelineFixture.Views, band.TiltEnvelopes.Length);
            for (int i = 0; i < band.Envelope.Power.Length; i++)
            {
                Near(band.TiltEnvelopes.Sum(s => (double)s.Cross[i]), band.Envelope.Cross[i], 3e-5, "managed tilt C sum");
                Near(band.TiltEnvelopes.Sum(s => (double)s.Power[i]), band.Envelope.Power[i], 3e-5, "managed tilt P sum");
            }
        }
    }

    private sealed record Start(float3 Position, Matrix3 Rotation, int Index);
    private sealed class Solution
    {
        public float3 Position;
        public Matrix3 Rotation;
        public double Z, Gain, Amplitude;
        public int StartIndex, Iterations;
        public double[] Statistics;
        public TemplateMatchEnvelopeSpectrum Envelope;
        public TemplateMatchEnvelopeSpectrum[] TiltEnvelopes;
    }

    private static Solution[] Refine(PipelineFixture fixture, Start[] starts, int iterations, out int usableTilts, bool fitEnvelope = false, decimal fitHighpass = 30, bool exportTilts = false)
    {
        Type type = typeof(TiltSeries).GetNestedType("MatchSolution", BindingFlags.NonPublic);
        Assert.NotNull(type);
        var list = (IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(type));
        foreach (Start start in starts)
        {
            object item = Activator.CreateInstance(type, nonPublic: true);
            type.GetField("Position").SetValue(item, start.Position);
            type.GetField("Rotation").SetValue(item, start.Rotation);
            type.GetField("StartIndex").SetValue(item, start.Index);
            list.Add(item);
        }
        MethodInfo method = typeof(TiltSeries).GetMethod("RefineMatchCandidate", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        var options = new ProcessingOptionsTomoFullMatch { RefineIterations = iterations, Lowpass = 0.8M, RefineFitBfactor = fitEnvelope, RefineFitHighpass = fitHighpass, RefineExportTiltSpectra = exportTilts };
        object[] args = { options, fixture.Anchor, list, fixture.Tilts, new Image[PipelineFixture.Views],
            fixture.NoiseWeights, fixture.Projector, fixture.CtfCoordinates, PipelineFixture.Box, PipelineFixture.Pixel, 7f, 0 };
        var returned = (IList)method.Invoke(fixture.Series, args);
        usableTilts = (int)args[^1];
        var result = new Solution[returned.Count];
        for (int i = 0; i < result.Length; i++)
        {
            object item = returned[i];
            T Read<T>(string field) => (T)type.GetField(field).GetValue(item);
            result[i] = new Solution
            {
                Position = Read<float3>("Position"), Rotation = Read<Matrix3>("Rotation"),
                Z = Read<double>("Z"), Gain = Read<double>("Gain"), Amplitude = Read<double>("Amplitude"),
                StartIndex = Read<int>("StartIndex"), Iterations = Read<int>("Iterations"),
                Statistics = Read<double[]>("TiltStatistics"), Envelope = Read<TemplateMatchEnvelopeSpectrum>("EnvelopeSpectrum"),
                TiltEnvelopes = Read<TemplateMatchEnvelopeSpectrum[]>("TiltEnvelopeSpectra")
            };
        }
        return result;
    }

    private sealed class PipelineFixture : IDisposable
    {
        public const int Box = 48, Views = 5;
        public const float Pixel = 3, Amplitude = 1.7f;
        private readonly string directory = Path.Combine(Path.GetTempPath(), "warp_match_pipeline_" + Guid.NewGuid().ToString("N"));
        private readonly List<CubicGrid> grids = new();
        public TiltSeries Series { get; }
        public Image[] Tilts { get; } = new Image[Views];
        public float[][] NoiseWeights { get; } = new float[Views][];
        public double[] OraclePower { get; } = new double[Views];
        public Projector Projector { get; }
        public Image CtfCoordinates { get; }
        public float3 Anchor { get; } = new(94.35f, 113.8f, 61.2f);
        public float3 TruePosition => Anchor + new float3(2.7f, -2.1f, 1.8f);
        public Matrix3 TrueRotation { get; } = Matrix3.Euler(0.31f, 0.77f, -0.28f);

        public PipelineFixture(bool addNoise)
        {
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, "synthetic.tomostar");
            File.WriteAllText(path, "data_\n\nloop_\n_wrpMovieName #1\n_wrpAngleTilt #2\n_wrpDose #3\n" +
                "tilt0.mrc -50 1\ntilt1.mrc -25 3\ntilt2.mrc 0 5\ntilt3.mrc 25 7\ntilt4.mrc 50 9\n");
            Series = new TiltSeries(path)
            {
                VolumeDimensionsPhysical = new float3(180, 240, 120),
                ImageDimensionsPhysical = new float2(128 * Pixel),
                LevelAngleX = 2.5f, LevelAngleY = -1.75f,
                TiltAxisAngles = new[] { 7f, 7f, 7f, 7f, 7f },
                TiltAxisOffsetX = new[] { 1.2f, -0.7f, 0.8f, -1.1f, 0.4f },
                TiltAxisOffsetY = new[] { -0.6f, 0.9f, -1.3f, 0.3f, 0.7f },
                CTF = new CTF { PixelSize = (decimal)Pixel, Voltage = 300, Cs = 2.7M, Amplitude = 0.07M,
                    PixelSizeDeltaPercent = 0.025M, PixelSizeAngle = 23 }
            };
            Series.GridCTFDefocus = Grid(new int3(1, 1, Views), 0.35f);
            Series.GridCTFDefocusDelta = Grid(new int3(1, 1, Views), 0.065f);
            Series.GridCTFDefocusAngle = Grid(new int3(1, 1, Views), 31f);
            Series.GridCTFPhase = Grid(new int3(1, 1, Views), 0.11f);
            using Image volume = CreateAsymmetricVolume();
            Projector = new Projector(volume, 2, true);
            CtfCoordinates = CTF.GetCTFCoords(Box, Box, Series.MagnificationCorrection);
            var random = new Random(48103);
            int elements = Box * (Box / 2 + 1);
            for (int t = 0; t < Views; t++)
            {
                float3 anchorInImage = Series.GetPositionsInOneTilt(new[] { Anchor }, t)[0];
                float3 trueInImage = Series.GetPositionsInOneTilt(new[] { TruePosition }, t)[0];
                int originX = (int)Math.Floor(anchorInImage.X / Pixel - Box / 2f);
                int originY = (int)Math.Floor(anchorInImage.Y / Pixel - Box / 2f);
                float3 shift = new(trueInImage.X / Pixel - originX, trueInImage.Y / Pixel - originY, 0);
                float3 angles = Series.GetAnglesInOneTilt(new[] { TruePosition },
                    new[] { Matrix3.EulerFromMatrix(TrueRotation) * Helper.ToDeg }, t)[0];
                // Independent existing forward API: do not synthesize with TemplateMatchRefineEvaluate.
                using Image projection = Projector.Project(new int2(Box), new[] { angles }, new[] { shift }, new[] { 1f });
                using Image ctf = new(new int3(Box, Box, 1), true);
                CTF parameters = Series.GetCTFParamsForOneTilt(Pixel, new[] { trueInImage.Z }, new[] { TruePosition }, t)[0];
                GPU.CreateCTF(ctf.GetDevice(Intent.Write), CtfCoordinates.GetDevice(Intent.Read), IntPtr.Zero,
                    (uint)elements, new[] { parameters.ToStruct() }, false, 1);
                projection.Multiply(ctf);
                float[] fourier = projection.GetHost(Intent.Read)[0];
                NoiseWeights[t] = new float[elements];
                for (int y = 0; y < Box; y++)
                    for (int x = 0; x <= Box / 2; x++)
                    {
                        int ky = y <= Box / 2 ? y : y - Box;
                        int index = y * (Box / 2 + 1) + x;
                        float weight = 0.8f + 0.15f * t + 0.2f * (x * x + ky * ky) / (Box * Box);
                        NoiseWeights[t][index] = weight;
                        if (!Independent(x, ky)) continue;
                        OraclePower[t] += weight * ((double)fourier[2 * index] * fourier[2 * index] +
                                                   (double)fourier[2 * index + 1] * fourier[2 * index + 1]);
                    }
                projection.Multiply(Amplitude);
                // Production uses FFT(patch) / Box^2, so synthesize with an unnormalized inverse FFT.
                using Image patch = projection.AsIFFT(normalize: false);
                float[] patchValues = patch.GetHost(Intent.Read)[0];
                float maximum = 0;
                foreach (float value in patchValues) maximum = Math.Max(maximum, Math.Abs(value));
                double sigma = addNoise ? maximum * 0.001 : 0;
                Tilts[t] = new Image(new int3(128, 128, 1));
                float[] tiltValues = Tilts[t].GetHost(Intent.Write)[0];
                Array.Clear(tiltValues);
                for (int y = 0; y < Box; y++)
                    for (int x = 0; x < Box; x++)
                    {
                        double noise = sigma == 0 ? 0 : sigma * Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) *
                            Math.Cos(2 * Math.PI * random.NextDouble());
                        tiltValues[(originY + y) * 128 + originX + x] = patchValues[y * Box + x] + (float)noise;
                    }
            }
        }

        private CubicGrid Grid(int3 dims, float value)
        {
            var values = new float[dims.Elements()];
            Array.Fill(values, value);
            var grid = new CubicGrid(dims, values);
            grids.Add(grid);
            return grid;
        }

        private static Image CreateAsymmetricVolume()
        {
            Image volume = new(new int3(Box));
            float[][] values = volume.GetHost(Intent.Write);
            (double X, double Y, double Z, double Sigma, double Weight)[] blobs =
            {
                (-3.2, -1.4, -1.7, 1.1, 1.0), (2.7, 1.8, -0.4, 1.25, 0.7),
                (-0.9, 3.1, 2.6, 0.95, 1.3), (1.6, -3.1, 3.3, 1.35, 0.5)
            };
            for (int z = 0; z < Box; z++)
                for (int y = 0; y < Box; y++)
                    for (int x = 0; x < Box; x++)
                    {
                        double value = 0;
                        foreach (var blob in blobs)
                        {
                            double dx = x - Box / 2 - blob.X, dy = y - Box / 2 - blob.Y, dz = z - Box / 2 - blob.Z;
                            value += blob.Weight * Math.Exp(-(dx * dx + dy * dy + dz * dz) / (2 * blob.Sigma * blob.Sigma));
                        }
                        values[z][y * Box + x] = (float)value;
                    }
            return volume;
        }

        private static bool Independent(int x, int y) => !(x == 0 && y <= 0) && x < Box / 2 &&
            Math.Abs(y) < Box / 2 && x * x + y * y <= Math.Pow(Box * 0.5 * 0.8, 2);

        public void Dispose()
        {
            foreach (Image tilt in Tilts) tilt?.Dispose();
            Projector?.Dispose();
            CtfCoordinates?.Dispose();
            foreach (CubicGrid grid in grids) grid.Dispose();
            Directory.Delete(directory, true);
        }
    }

    private static double RotationError(Matrix3 actual, Matrix3 expected)
    {
        Matrix3 difference = actual * expected.Transposed();
        return Math.Acos(Math.Clamp(((double)difference.M11 + difference.M22 + difference.M33 - 1) / 2, -1, 1));
    }
    private static void Near(double actual, double expected, double relative, string label) =>
        Assert.True(double.IsFinite(actual) && Math.Abs(actual - expected) <= 1e-9 + relative * Math.Abs(expected),
            $"{label}: actual {actual:R}, expected {expected:R}");
}
