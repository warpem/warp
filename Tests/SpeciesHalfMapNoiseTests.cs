using System;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class SpeciesHalfMapNoiseTests
{
    [CtfCudaFact]
    public void LowpassedHalfMapsKeepLowFrequenciesAndLoseSpuriousMaskedFsc()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using Image map = RandomMap(64);
            float cutoff = 2 * .79f / 10;
            map.Bandpass(0, cutoff, true, .05f);
            CheckNoiseFloor(map, cutoff);
        }
    }

    [CtfCudaFact]
    public void UpsampledHalfMapsKeepLowFrequenciesAndLoseSpuriousMaskedFsc()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using Image original = RandomMap(32);
            using Image upsampled = original.AsScaled(new int3(64));
            CheckNoiseFloor(upsampled, (float)original.Dims.X / upsampled.Dims.X);
        }
    }

    [CtfCudaFact]
    public void NoiseFloorScalesWithMapDensity()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using Image map = RandomMap(32);
            using Image scaled = map.GetCopy();
            scaled.Multiply(1e-6f);
            map.AddHighFrequencyNoise(.5f, seed: 123);
            scaled.AddHighFrequencyNoise(.5f, seed: 123);
            float[][] expected = map.GetHost(Intent.Read);
            float[][] actual = scaled.GetHost(Intent.Read);
            for (int z = 0; z < map.Dims.Z; z++)
                for (int i = 0; i < expected[z].Length; i++)
                    Assert.InRange(Math.Abs(actual[z][i] / 1e-6f - expected[z][i]), 0, 3e-6f);
        }
    }

    private static Image RandomMap(int size)
    {
        Image map = new Image(new int3(size));
        RandomNormal random = new RandomNormal(321);
        map.TransformValues(_ => random.NextSingle(0, 1));
        return map;
    }

    private static void CheckNoiseFloor(Image map, float cutoff)
    {
        using Image half1 = map.GetCopy();
        using Image half2 = map.GetCopy();
        half1.AddHighFrequencyNoise(cutoff, seed: 123);
        half2.AddHighFrequencyNoise(cutoff, seed: 456);

        using (Image before = map.AsFFT(true))
        using (Image after = half1.AsFFT(true))
        {
            float[][] a = before.GetHost(Intent.Read), b = after.GetHost(Intent.Read);
            double signalPower = 0, errorPower = 0;
            VisitBand(map.Dims.X, 0, cutoff - .06f, (z, i) =>
            {
                signalPower += (double)a[z][i] * a[z][i] + (double)a[z][i + 1] * a[z][i + 1];
                errorPower += Math.Pow(a[z][i] - b[z][i], 2) + Math.Pow(a[z][i + 1] - b[z][i + 1], 2);
            });
            Assert.True(errorPower / signalPower < 1e-10, "Noise must preserve the retained low frequencies.");
        }

        // Identical band-limited maps are deliberately the worst case: masking
        // alone leaks shared signal into empty shells, where FSC would be 1.
        half1.MaskSpherically(32, 8, true);
        half2.MaskSpherically(32, 8, true);
        using Image ft1 = half1.AsFFT(true);
        using Image ft2 = half2.AsFFT(true);
        float[][] f1 = ft1.GetHost(Intent.Read), f2 = ft2.GetHost(Intent.Read);
        double cross = 0, power1 = 0, power2 = 0;
        // Upsampling 32 -> 64 retains a Fourier cube with corners out to
        // sqrt(3) / 2; measure beyond that support as well as the filter cutoff.
        VisitBand(map.Dims.X, .9f, 1, (z, i) =>
        {
            cross += (double)f1[z][i] * f2[z][i] + (double)f1[z][i + 1] * f2[z][i + 1];
            power1 += (double)f1[z][i] * f1[z][i] + (double)f1[z][i + 1] * f1[z][i + 1];
            power2 += (double)f2[z][i] * f2[z][i] + (double)f2[z][i + 1] * f2[z][i + 1];
        });
        Assert.True(power1 > 0 && power2 > 0, "Empty shells must contain noise.");
        double fsc = cross / Math.Sqrt(power1 * power2);
        Assert.InRange(fsc, -.1, .143);
    }

    private static void VisitBand(int size, float low, float high, Action<int, int> visit)
    {
        for (int z = 0; z < size; z++)
            for (int y = 0; y < size; y++)
                for (int x = 0; x <= size / 2; x++)
                {
                    float fy = Math.Min(y, size - y) / (size / 2f);
                    float fz = Math.Min(z, size - z) / (size / 2f);
                    float fx = x / (size / 2f);
                    float r = MathF.Sqrt(fx * fx + fy * fy + fz * fz);
                    if (r >= low && r < high)
                        visit(z, (y * (size / 2 + 1) + x) * 2);
                }
    }
}
