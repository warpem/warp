using System;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

public class TemplateMatchingTests
{
    [Fact]
    public void PoolingUsesNeighboursAndKeepsTranslationDistinctStarts()
    {
        int3 dims = new int3(3);
        var scores = Volume(3, dims, float.NegativeInfinity);
        var ids = Volume(3, dims, -1);
        Set(scores, ids, 0, new int3(1), dims, 10, 0);
        Set(scores, ids, 1, new int3(1), dims, 9, 1);
        Set(scores, ids, 2, new int3(1), dims, 7, 0); // Duplicate angle and translation.
        Set(scores, ids, 0, new int3(2, 1, 1), dims, 12, 2);
        Set(scores, ids, 1, new int3(2, 1, 1), dims, 8, 0); // Same angle, distinct translation.
        float3[] angles = { new float3(0), new float3(0.1f, 0.2f, 0.3f), new float3(0.4f) };

        var starts = TemplateMatching.GatherStarts(new int3(1), dims, 2.5f, scores, ids, angles, 20);

        Assert.Equal(new[] { 12f, 10f, 9f, 8f }, starts.Select(s => s.ProposalScore));
        Assert.Equal(new[] { 2, 0, 1, 0 }, starts.Select(s => s.AngleId));
        Assert.Equal(5, starts[0].Position.X);
        Assert.Equal(2.5f, starts[0].Position.Y);
        Assert.Equal(0.1f, starts[2].Angles.X);
        Assert.Equal(0.2f, starts[2].Angles.Y);
        Assert.Equal(0.3f, starts[2].Angles.Z);
        Assert.Equal(1, starts[2].Rank);
        Assert.Equal(4, starts.Length);
    }

    [Fact]
    public void TiesPreferCentralVoxelThenRankAndHaveDeterministicOrder()
    {
        int3 dims = new int3(3);
        var scores = Volume(2, dims, float.NegativeInfinity);
        var ids = Volume(2, dims, -1);
        Set(scores, ids, 0, new int3(1), dims, 10, 2);
        Set(scores, ids, 1, new int3(1), dims, 10, 1);
        Set(scores, ids, 0, new int3(0, 1, 1), dims, 10, 0);
        Set(scores, ids, 0, new int3(2, 1, 1), dims, 10, 0);

        var starts = TemplateMatching.GatherStarts(new int3(1), dims, 1, scores, ids, new float3[3], 3);

        Assert.Equal(new[] { 2, 1, 0 }, starts.Select(s => s.AngleId));
        Assert.Equal(new[] { 1, 1, 0 }, starts.Select(s => s.SourceVoxel.X));
    }

    [Fact]
    public void CornerPoolingOmitsOutOfBoundsAndDiagonalVoxels()
    {
        int3 dims = new int3(2);
        var scores = Volume(1, dims, 1);
        var ids = Volume(1, dims, 0);

        var starts = TemplateMatching.GatherStarts(new int3(0), dims, 1, scores, ids, new float3[1], 20);

        Assert.Equal(4, starts.Length);
        Assert.All(starts, s => Assert.InRange(s.SourceVoxel.X + s.SourceVoxel.Y + s.SourceVoxel.Z, 0, 1));
        Assert.Equal(new int3(0), starts[0].SourceVoxel);
    }

    [Fact]
    public void PoolingSkipsNonfiniteScoresAndInvalidAnglesWithoutRoundingIds()
    {
        int3 dims = new int3(1);
        var scores = Volume(8, dims, 5);
        var ids = Volume(8, dims, -1);
        float[] badAndGoodIds = { float.NaN, float.PositiveInfinity, -1, 3, 0.5f, 0, 1, 2 };
        for (int rank = 0; rank < ids.Length; rank++) ids[rank][0][0] = badAndGoodIds[rank];
        scores[5][0][0] = float.NaN;
        float3[] angles = { new float3(0), new float3(1), new float3(float.NaN) };

        var starts = TemplateMatching.GatherStarts(new int3(0), dims, 1, scores, ids, angles, 20);

        Assert.Single(starts);
        Assert.Equal(1, starts[0].AngleId);
        Assert.Equal(6, starts[0].Rank);
    }

    [Fact]
    public void SparsePoolingMatchesVolumePooling()
    {
        int3 dims = new int3(3);
        int3[] voxels = { new int3(1), new int3(0, 1, 1), new int3(2, 1, 1),
                          new int3(1, 0, 1), new int3(1, 2, 1), new int3(1, 1, 0), new int3(1, 1, 2) };
        var scores = Volume(2, dims, float.NegativeInfinity);
        var ids = Volume(2, dims, -1);
        var sparseScores = new float[14];
        var sparseIds = new float[14];
        for (int p = 0; p < voxels.Length; p++)
            for (int rank = 0; rank < 2; rank++)
            {
                sparseScores[p * 2 + rank] = 10 - p - rank;
                sparseIds[p * 2 + rank] = (p + rank) % 3;
                Set(scores, ids, rank, voxels[p], dims, sparseScores[p * 2 + rank], sparseIds[p * 2 + rank]);
            }
        sparseIds[5] = ids[1][1][5] = float.NaN;

        var fromVolume = TemplateMatching.GatherStarts(new int3(1), dims, 2, scores, ids, new float3[3], 10);
        var fromSparse = TemplateMatching.GatherStarts(voxels, sparseScores, sparseIds, 2, 2, new float3[3], 10);

        Assert.Equal(fromVolume.Select(Key), fromSparse.Select(Key));
    }

    private static (int X, int Y, int Z, int Angle, int Rank, float Score) Key(TemplateMatchStart start) =>
        (start.SourceVoxel.X, start.SourceVoxel.Y, start.SourceVoxel.Z, start.AngleId, start.Rank, start.ProposalScore);

    private static float[][][] Volume(int ranks, int3 dims, float initial)
    {
        var result = new float[ranks][][];
        for (int rank = 0; rank < ranks; rank++)
        {
            result[rank] = new float[dims.Z][];
            for (int z = 0; z < dims.Z; z++) result[rank][z] = Enumerable.Repeat(initial, dims.X * dims.Y).ToArray();
        }
        return result;
    }

    private static void Set(float[][][] scores, float[][][] ids, int rank, int3 voxel, int3 dims, float score, float id)
    {
        scores[rank][voxel.Z][voxel.Y * dims.X + voxel.X] = score;
        ids[rank][voxel.Z][voxel.Y * dims.X + voxel.X] = id;
    }
}
