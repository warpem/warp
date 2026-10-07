using System;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class TemplateMatchPreparationCudaTests
{
    [TemplateMatchCudaFact]
    public void RefinementProjectorsPreserveAnalyticFourierValuesAcrossBoxSizes()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            foreach (int size in new[] { 48, 64, 100, 160 })
                foreach (bool straightToGpu in new[] { false, true })
                {
                    using Image volume = new(new int3(size));
                    float[] center = volume.GetHost(Intent.Write)[size / 2];
                    center[size / 2 * size + size / 2] = size * size;
                    center[size / 2 * size + size / 2 + 1] = size * size * .5f;
                    double x = Math.PI / (2 * size), corrected = .5 * Math.Pow(x / Math.Sin(x), 2);
                    double re = 1 + corrected * Math.Cos(2 * Math.PI / size), im = -corrected * Math.Sin(2 * Math.PI / size);
                    using Projector projector = new(volume, 2, straightToGpu);
                    if (!straightToGpu)
                    {
                        float[] row = projector.Data.GetHost(Intent.Read)[0];
                        Assert.InRange(row[0], 1 + corrected - 1e-5, 1 + corrected + 1e-5);
                        Assert.InRange(row[4], re - 1e-5, re + 1e-5);
                        Assert.InRange(row[5], im - 1e-5, im + 1e-5);
                    }
                    using Image projection = projector.Project(new int2(size), [new float3(0)]);
                    float[] projected = projection.GetHost(Intent.Read)[0];
                    Assert.InRange(projected[0], 1 + corrected - 1e-5, 1 + corrected + 1e-5);
                    Assert.InRange(projected[2], re - 1e-5, re + 1e-5);
                    Assert.InRange(projected[3], im - 1e-5, im + 1e-5);
                }
        }
    }
    private static readonly float[] Identity = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    [TemplateMatchCudaFact]
    public void HybridCountsOnlyUsableOverlappingViews()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using Image weights = new(new int3(8, 8, 3), true);
            var data = weights.GetHost(Intent.Write);
            Array.Fill(data[0], 1f / 3); Array.Fill(data[1], 1f / 3); Array.Clear(data[2]);
            float[] rotations = new float[27]; Array.Copy(Identity, rotations, 9); Array.Copy(Identity, 0, rotations, 9, 9);
            GPU.MatchHybridWeights(weights.GetDevice(Intent.ReadWrite), 8, 1, 3, rotations, new[] { 4f, 4f, 4f }, 3, 400);
            data = weights.GetHost(Intent.Read);
            for (int t = 0; t < 2; t++)
            {
                Assert.Equal(0, data[t][0]);
                foreach (float w in data[t].Skip(1)) Assert.InRange(w, .249999f, .250001f);
            }
            Assert.All(data[2], w => Assert.Equal(0, w));
            // Orthogonal tilt planes overlap only along their common line.
            float[] orthogonal = { 0, 0, -1, 0, 1, 0, 1, 0, 0 };
            Array.Copy(orthogonal, 0, rotations, 9, 9);
            data = weights.GetHost(Intent.Write);
            Array.Fill(data[0], 1f / 3); Array.Fill(data[1], 1f / 3); Array.Clear(data[2]);
            GPU.MatchHybridWeights(weights.GetDevice(Intent.ReadWrite), 8, 1, 3, rotations, new[] { 4f, 4f, 4f }, 3, 400);
            data = weights.GetHost(Intent.Read);
            for (int t = 0; t < 2; t++)
            {
                Assert.InRange(data[t][1], .333332f, .333334f);
                Assert.InRange(data[t][5], .249999f, .250001f);
            }
        }
    }
    [TemplateMatchCudaFact]
    public void TransferSumsSquaredCtfAndIndependentNoiseOnTheSlice()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using Image ctfs = new(new int3(8, 8, 2), true); ctfs.Fill(2);
            using Image transfer = new(new int3(8), true);
            GPU.MatchTransfer(ctfs.GetDevice(Intent.Read), transfer.GetDevice(Intent.Write), 8, 2,
                Identity.Concat(Identity).ToArray(), new[] { .25f, .5f });
            var data = transfer.GetHost(Intent.Read);
            Assert.Equal(0, data[0][0]);
            foreach (float v in data[0].Skip(1)) Assert.InRange(v, 2.99999f, 3.00001f);
            foreach (var slice in data.Skip(1)) Assert.All(slice, v => Assert.Equal(0, v));
        }
    }
    [TemplateMatchCudaFact]
    public void DefocusBinInterpolationPreservesAnAffineImageAndVoxelOrigin()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using Image image = new(new int3(8, 8, 1));
            float[] d = image.GetHost(Intent.Write)[0];
            for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) d[y * 8 + x] = 2 * x + 3 * y;
            using Image volume = new(new int3(4)); volume.Fill(0);
            using Image geometry = new(new int3(6, 2, 2));
            var g = geometry.GetHost(Intent.Write);
            for (int z = 0; z < 2; z++) for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++)
                    { int i = (y * 2 + x) * 3; g[z][i] = 1.25f + 4 * x; g[z][i + 1] = 1.5f + 4 * y; g[z][i + 2] = .8f * z; }
            for (int bin = 0; bin < 2; bin++) GPU.MatchBackproject(image.GetDevice(Intent.Read), new int2(8),
                volume.GetDevice(Intent.ReadWrite), new int3(4), geometry.GetDevice(Intent.Read), new int3(2), 4, bin, 1);
            var output = volume.GetHost(Intent.Read);
            for (int z = 0; z < 4; z++) for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
                        Assert.InRange(output[z][y * 4 + x], 7 + 2 * x + 3 * y - 1e-4f, 7 + 2 * x + 3 * y + 1e-4f);
        }
    }
    [TemplateMatchCudaFact]
    public void SpectrumResamplingPreservesHermitianSymmetryAndConstants()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            using Image input = new(new int3(12, 10, 8), true); input.Fill(3);
            using Image output = new(new int3(6), true);
            GPU.MatchSpectrumResample(input.GetDevice(Intent.Read), input.Dims, output.GetDevice(Intent.Write), 6);
            foreach (var slice in output.GetHost(Intent.Read)) Assert.All(slice, v => Assert.InRange(v, 2.99999f, 3.00001f));
        }
    }
}
