using System;
using System.Collections.Generic;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class TemplateMatchTopKCudaTests
{
    [TemplateMatchCudaFact]
    public void LocalPeaksResolvePlateausAndIgnoreNonfiniteValues()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            int3 dims = new int3(7, 5, 3);
            using (Image plateau = new Image(Enumerable.Repeat(2f, (int)dims.Elements()).ToArray(), dims))
            {
                int3[] peaks = plateau.GetLocalPeaks(2, -float.MaxValue);
                Assert.Single(peaks);
                Assert.Equal(new int3(0), peaks[0]);
            }

            dims = new int3(7, 3, 1);
            float[] values = Enumerable.Repeat(float.NegativeInfinity, (int)dims.Elements()).ToArray();
            values[1 * dims.X + 1] = 5;
            values[1 * dims.X + 2] = 5; // Equal adjacent candidates: lower linear ID wins.
            values[1 * dims.X + 5] = 5; // Outside suppression radius: a distinct equal peak survives.
            values[1] = float.PositiveInfinity; // Invalid neighbor must not suppress the finite peak below it.
            values[5] = float.NaN;
            using (Image candidates = new Image(values, dims))
            {
                int3[] peaks = candidates.GetLocalPeaks(1, -float.MaxValue);
                Assert.Equal(new[] { new int3(1, 1, 0), new int3(5, 1, 0) }, peaks);
            }

            using (Image invalid = new Image(new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity }, new int3(3, 1, 1)))
                Assert.Empty(invalid.GetLocalPeaks(1, -float.MaxValue));
        }
    }

    [TemplateMatchCudaFact]
    public void GatherConvertsRankMajorVolumesAndRejectsBoundaryCoordinates()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            int3 dims = new int3(4, 3, 2);
            const int elements = 24, topK = 3;
            float[] scores = new float[elements * topK], angles = new float[elements * topK];
            for (int rank = 0; rank < topK; rank++)
                for (int voxel = 0; voxel < elements; voxel++)
                {
                    scores[rank * elements + voxel] = 1000 - rank * 100 + voxel;
                    angles[rank * elements + voxel] = rank * 40 + voxel;
                }
            int3[] positions =
            {
                new int3(0, 0, 0), new int3(3, 2, 1), new int3(2, 1, 0),
                new int3(-1, 0, 0), new int3(4, 0, 0), new int3(0, -1, 0),
                new int3(0, 3, 0), new int3(0, 0, -1), new int3(0, 0, 2),
                new int3(0, 0, 0)
            };
            IntPtr deviceScores = GPU.MallocDeviceFromHost(scores, scores.Length);
            IntPtr deviceAngles = GPU.MallocDeviceFromHost(angles, angles.Length);
            try
            {
                float[] packedScores = new float[positions.Length * topK], packedAngles = new float[positions.Length * topK];
                Assert.Equal(0, GPU.GatherTemplateMatchTopK(deviceScores, deviceAngles, dims,
                    positions, positions.Length, topK, packedScores, packedAngles));
                for (int p = 0; p < positions.Length; p++)
                {
                    int3 position = positions[p];
                    bool valid = position.X >= 0 && position.X < dims.X && position.Y >= 0 &&
                                 position.Y < dims.Y && position.Z >= 0 && position.Z < dims.Z;
                    int voxel = (position.Z * dims.Y + position.Y) * dims.X + position.X;
                    for (int rank = 0; rank < topK; rank++)
                    {
                        Assert.Equal(valid ? scores[rank * elements + voxel] : float.NegativeInfinity, packedScores[p * topK + rank]);
                        Assert.Equal(valid ? angles[rank * elements + voxel] : -1f, packedAngles[p * topK + rank]);
                    }
                }
            }
            finally
            {
                GPU.FreeDevice(deviceAngles);
                GPU.FreeDevice(deviceScores);
            }
        }
    }

    [TemplateMatchCudaFact]
    public void BatchedTopKMatchesExhaustiveSingleAnglesAndLegacyRankZero()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            const int box = 8, projectorDim = 19, count = 5, topK = 7;
            const int elements = box * box * box, fourierElements = (box / 2 + 1) * box * box;
            int3 dims = new int3(box), projectorDims = new int3(projectorDim);
            var allocations = new List<IntPtr>();
            var textures = new ulong[2];
            var textureArrays = new ulong[2];
            IntPtr Upload(float[] values)
            {
                IntPtr pointer = GPU.MallocDeviceFromHost(values, values.Length);
                allocations.Add(pointer);
                return pointer;
            }
            float[] Read(IntPtr pointer, int length)
            {
                float[] values = new float[length];
                GPU.CopyDeviceToHost(pointer, values, length);
                return values;
            }
            try
            {
                float[] volume = new float[(projectorDim / 2 + 1) * projectorDim * projectorDim * 2];
                for (int i = 0; i < volume.Length; i++) volume[i] = MathF.Sin(i * .131f) + .3f * MathF.Cos(i * .317f);
                GPU.CreateTexture3DComplex(Upload(volume), new int3(projectorDim / 2 + 1, projectorDim, projectorDim),
                    textures, textureArrays, false); // Match Projector's point-sampled textures.
                float[] data = new float[fourierElements * 2];
                for (int i = 0; i < data.Length; i++) data[i] = MathF.Sin(i * .071f) + .4f * MathF.Cos(i * .137f);
                IntPtr deviceData = Upload(data), ctf = Upload(Enumerable.Repeat(1f, fourierElements).ToArray());
                IntPtr topScores = Upload(new float[elements * topK]), topAngles = Upload(new float[elements * topK]);
                IntPtr oneScores = Upload(new float[elements]), oneAngles = Upload(new float[elements]);
                float[] angles = { .13f, .41f, -.27f, -.33f, .82f, .19f, .47f, 1.13f, -.51f,
                                   -.61f, .63f, .37f, .29f, 1.47f, -.73f };
                float[] progress = new float[1];

                // Five angles with batch size two exercises the partial final batch.
                GPU.CorrelateLargeVolumeTopK(textures[0], textures[1], 2, projectorDims,
                    deviceData, ctf, dims, angles, count, 2, 3, topK, topScores, topAngles, progress);
                float[] scores = Read(topScores, elements * topK), ids = Read(topAngles, elements * topK);
                Assert.Equal(1f, progress[0]);
                GPU.CorrelateLargeVolumeTopK(textures[0], textures[1], 2, projectorDims,
                    deviceData, ctf, dims, angles, count, 2, 3, 1, oneScores, oneAngles, progress);
                float[] singleBestScores = Read(oneScores, elements), singleBestAngles = Read(oneAngles, elements);
                for (int voxel = 0; voxel < elements; voxel++)
                {
                    Near(scores[voxel], singleBestScores[voxel]);
                    Assert.Equal(ids[voxel], singleBestAngles[voxel]);
                }

                // Template normalization must be invariant to a common transfer scale, including
                // whitened transfers far below the historical absolute 0.01 CTF cutoff.
                GPU.CopyHostToDevice(Enumerable.Repeat(1e-5f, fourierElements).ToArray(), ctf, fourierElements);
                GPU.CorrelateLargeVolumeTopK(textures[0], textures[1], 2, projectorDims,
                    deviceData, ctf, dims, angles, count, 2, 3, 1, oneScores, oneAngles, progress);
                float[] scaledScores = Read(oneScores, elements);
                for (int voxel = 0; voxel < elements; voxel++) Near(singleBestScores[voxel], scaledScores[voxel]);
                GPU.CopyHostToDevice(Enumerable.Repeat(1f, fourierElements).ToArray(), ctf, fourierElements);

                float[][] exhaustive = new float[count][];
                for (int angle = 0; angle < count; angle++)
                {
                    float[] singleAngle = angles.Skip(angle * 3).Take(3).ToArray();
                    GPU.CorrelateLargeVolumeTopK(textures[0], textures[1], 2, projectorDims,
                        deviceData, ctf, dims, singleAngle, 1, 1, 3, 1, oneScores, oneAngles, progress);
                    exhaustive[angle] = Read(oneScores, elements);
                }
                for (int voxel = 0; voxel < elements; voxel++)
                {
                    int[] expected = Enumerable.Range(0, count).OrderByDescending(a => exhaustive[a][voxel]).ThenBy(a => a).ToArray();
                    var retained = new HashSet<int>();
                    for (int rank = 0; rank < count; rank++)
                    {
                        int angle = (int)ids[rank * elements + voxel];
                        Assert.InRange(angle, 0, count - 1);
                        Assert.True(retained.Add(angle));
                        Near(scores[rank * elements + voxel], exhaustive[angle][voxel]);
                        Near(scores[rank * elements + voxel], exhaustive[expected[rank]][voxel]);
                        if (rank > 0) Assert.True(scores[(rank - 1) * elements + voxel] >= scores[rank * elements + voxel]);
                    }
                    for (int rank = count; rank < topK; rank++)
                    {
                        Assert.Equal(float.NegativeInfinity, scores[rank * elements + voxel]);
                        Assert.Equal(-1f, ids[rank * elements + voxel]);
                    }
                }

                // Zero data gives exact tied scores and verifies ascending angle IDs
                // across batches, as well as reinitialization of reused output buffers.
                GPU.CopyHostToDevice(new float[data.Length], deviceData, data.Length);
                GPU.CorrelateLargeVolumeTopK(textures[0], textures[1], 2, projectorDims,
                    deviceData, ctf, dims, angles, count, 2, 3, topK, topScores, topAngles, progress);
                scores = Read(topScores, elements * topK);
                ids = Read(topAngles, elements * topK);
                for (int rank = 0; rank < count; rank++)
                    for (int voxel = 0; voxel < elements; voxel++)
                    {
                        Assert.Equal(0f, scores[rank * elements + voxel]);
                        Assert.Equal((float)rank, ids[rank * elements + voxel]);
                    }
            }
            finally
            {
                for (int i = 0; i < textures.Length; i++)
                    if (textures[i] != 0) GPU.DestroyTexture(textures[i], textureArrays[i]);
                foreach (IntPtr pointer in allocations) GPU.FreeDevice(pointer);
            }
        }
    }

    private static void Near(float actual, float expected)
    {
        Assert.True(float.IsFinite(actual) && Math.Abs(actual - expected) <= 2e-4f * (1 + Math.Abs(expected)),
            $"{actual:R} != {expected:R}");
    }
}
