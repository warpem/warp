using System;
using System.Collections.Generic;

namespace Warp.Tools
{
    /// <summary>A proposal in physical position units and Euler angles in radians.</summary>
    public sealed class TemplateMatchStart
    {
        public float3 Position { get; }
        public float3 Angles { get; }
        public float ProposalScore { get; }
        public int AngleId { get; }
        public int Rank { get; }
        public int3 SourceVoxel { get; }

        internal TemplateMatchStart(int3 voxel, float pixelSize, float3 angles, float score, int angleId, int rank)
        {
            Position = new float3(voxel) * pixelSize;
            Angles = angles;
            ProposalScore = score;
            AngleId = angleId;
            Rank = rank;
            SourceVoxel = voxel;
        }
    }

    public enum TemplateMatchTerminationReason
    {
        GradientTolerance,
        IterationLimit,
        LineSearchStalled,
        StepTolerance,
        InsufficientTilts,
        InvalidPose
    }

    /// <summary>Managed proposal pooling with deterministic ordering; no native or GPU calls.</summary>
    public static class TemplateMatching
    {
        /// <summary>
        /// Pool the peak voxel and its six face neighbours from rank-major [rank][z][y * X + x] arrays.
        /// Positions use the voxel origin convention, voxel * pixelSize. Invalid scores, angle IDs and angles
        /// are ignored. Ties prefer the peak, then lower rank, then z/y/x and angle ID.
        /// Deduplication deliberately only removes the same angle ID at the same voxel: adjacent translations
        /// remain independent starts. Symmetry-equivalent but differently numbered angles remain separate;
        /// Warp's symmetry matrix lookup requires native code and is not used by this managed helper.
        /// </summary>
        public static TemplateMatchStart[] GatherStarts(int3 peak, int3 dims, float pixelSize,
                                                       float[][][] scores, float[][][] angleIds,
                                                       float3[] angles, int maxStarts)
        {
            if (dims.X <= 0 || dims.Y <= 0 || dims.Z <= 0 || (long)dims.X * dims.Y > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(dims));
            if (!float.IsFinite(pixelSize) || pixelSize <= 0 ||
                (double)Math.Max(dims.X - 1, Math.Max(dims.Y - 1, dims.Z - 1)) * pixelSize > float.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(pixelSize));
            if (!Inside(peak, dims))
                throw new ArgumentOutOfRangeException(nameof(peak));
            if (scores == null) throw new ArgumentNullException(nameof(scores));
            if (angleIds == null) throw new ArgumentNullException(nameof(angleIds));
            if (angles == null) throw new ArgumentNullException(nameof(angles));
            if (scores.Length != angleIds.Length)
                throw new ArgumentException("Scores and angle IDs must contain the same number of ranks.");
            if (maxStarts < 0) throw new ArgumentOutOfRangeException(nameof(maxStarts));
            if (maxStarts == 0) return Array.Empty<TemplateMatchStart>();

            int3[] offsets = { new int3(0), new int3(-1, 0, 0), new int3(1, 0, 0),
                               new int3(0, -1, 0), new int3(0, 1, 0), new int3(0, 0, -1), new int3(0, 0, 1) };
            var candidates = new List<TemplateMatchStart>();
            for (int rank = 0; rank < scores.Length; rank++)
            {
                if (scores[rank] == null || angleIds[rank] == null ||
                    scores[rank].Length != dims.Z || angleIds[rank].Length != dims.Z)
                    throw new ArgumentException("Every rank must have one slice per z coordinate.");

                foreach (int3 offset in offsets)
                {
                    int3 voxel = new int3(peak.X + offset.X, peak.Y + offset.Y, peak.Z + offset.Z);
                    if (!Inside(voxel, dims)) continue;
                    float[] scoreSlice = scores[rank][voxel.Z];
                    float[] angleSlice = angleIds[rank][voxel.Z];
                    if (scoreSlice == null || angleSlice == null ||
                        scoreSlice.Length != dims.X * dims.Y || angleSlice.Length != dims.X * dims.Y)
                        throw new ArgumentException("Every used slice must have X * Y scores and angle IDs.");

                    int index = voxel.Y * dims.X + voxel.X;
                    float score = scoreSlice[index];
                    float id = angleSlice[index];
                    if (!float.IsFinite(score) || !float.IsFinite(id) || id < 0 ||
                        (double)id >= angles.Length || id != MathF.Truncate(id)) continue;
                    int angleId = (int)id;
                    float3 angle = angles[angleId];
                    if (!float.IsFinite(angle.X) || !float.IsFinite(angle.Y) || !float.IsFinite(angle.Z)) continue;
                    candidates.Add(new TemplateMatchStart(voxel, pixelSize, angle, score, angleId, rank));
                }
            }

            return SelectStarts(candidates, peak, maxStarts);
        }

        /// <summary>
        /// Pool sparsely gathered proposals in position-major, then rank order. sourceVoxels[0] is the
        /// central peak for tie breaking; callers omit out-of-bounds neighbours before gathering.
        /// Uses the same conservative deduplication as the rank-major volume overload.
        /// </summary>
        public static TemplateMatchStart[] GatherStarts(int3[] sourceVoxels, float[] scores, float[] angleIds,
                                                       int topK, float pixelSize, float3[] angles, int maxStarts)
        {
            if (sourceVoxels == null) throw new ArgumentNullException(nameof(sourceVoxels));
            if (scores == null) throw new ArgumentNullException(nameof(scores));
            if (angleIds == null) throw new ArgumentNullException(nameof(angleIds));
            if (angles == null) throw new ArgumentNullException(nameof(angles));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
            if (maxStarts < 0) throw new ArgumentOutOfRangeException(nameof(maxStarts));
            if (!float.IsFinite(pixelSize) || pixelSize <= 0) throw new ArgumentOutOfRangeException(nameof(pixelSize));
            if ((long)sourceVoxels.Length * topK != scores.Length || scores.Length != angleIds.Length)
                throw new ArgumentException("Sparse scores and angle IDs must contain sourceVoxels.Length * topK entries.");
            if (sourceVoxels.Length == 0 || maxStarts == 0) return Array.Empty<TemplateMatchStart>();

            var candidates = new List<TemplateMatchStart>();
            for (int position = 0; position < sourceVoxels.Length; position++)
            {
                int3 voxel = sourceVoxels[position];
                float3 physical = new float3(voxel) * pixelSize;
                if (!float.IsFinite(physical.X) || !float.IsFinite(physical.Y) || !float.IsFinite(physical.Z))
                    throw new ArgumentOutOfRangeException(nameof(sourceVoxels), "Physical positions must be finite.");
                for (int rank = 0; rank < topK; rank++)
                {
                    int index = position * topK + rank;
                    float score = scores[index];
                    float id = angleIds[index];
                    if (!float.IsFinite(score) || !float.IsFinite(id) || id < 0 ||
                        (double)id >= angles.Length || id != MathF.Truncate(id)) continue;
                    int angleId = (int)id;
                    float3 angle = angles[angleId];
                    if (!float.IsFinite(angle.X) || !float.IsFinite(angle.Y) || !float.IsFinite(angle.Z)) continue;
                    candidates.Add(new TemplateMatchStart(voxel, pixelSize, angle, score, angleId, rank));
                }
            }
            return SelectStarts(candidates, sourceVoxels[0], maxStarts);
        }

        private static TemplateMatchStart[] SelectStarts(List<TemplateMatchStart> candidates, int3 peak, int maxStarts)
        {
            candidates.Sort((a, b) =>
            {
                int order = b.ProposalScore.CompareTo(a.ProposalScore);
                if (order != 0) return order;
                order = DistanceSquared(a.SourceVoxel, peak).CompareTo(DistanceSquared(b.SourceVoxel, peak));
                if (order != 0) return order;
                order = a.Rank.CompareTo(b.Rank);
                if (order != 0) return order;
                order = a.SourceVoxel.Z.CompareTo(b.SourceVoxel.Z);
                if (order != 0) return order;
                order = a.SourceVoxel.Y.CompareTo(b.SourceVoxel.Y);
                if (order != 0) return order;
                order = a.SourceVoxel.X.CompareTo(b.SourceVoxel.X);
                return order != 0 ? order : a.AngleId.CompareTo(b.AngleId);
            });

            var result = new List<TemplateMatchStart>(Math.Min(maxStarts, candidates.Count));
            var seen = new HashSet<(int Angle, int X, int Y, int Z)>();
            foreach (TemplateMatchStart candidate in candidates)
            {
                int3 voxel = candidate.SourceVoxel;
                if (!seen.Add((candidate.AngleId, voxel.X, voxel.Y, voxel.Z))) continue;
                result.Add(candidate);
                if (result.Count == maxStarts) break;
            }
            return result.ToArray();
        }

        private static bool Inside(int3 p, int3 dims) => p.X >= 0 && p.Y >= 0 && p.Z >= 0 && p.X < dims.X && p.Y < dims.Y && p.Z < dims.Z;
        private static double DistanceSquared(int3 a, int3 b) => ((double)a.X - b.X) * ((double)a.X - b.X) + ((double)a.Y - b.Y) * ((double)a.Y - b.Y) + ((double)a.Z - b.Z) * ((double)a.Z - b.Z);
    }
}
