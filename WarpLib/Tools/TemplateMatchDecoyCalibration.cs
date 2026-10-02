using System;
using System.Collections.Generic;

namespace Warp.Tools
{
    /// <summary>
    /// Empirical false-output counts from equal-exposure, complete decoy searches.
    /// These are diagnostics until decoy exchangeability has been validated;
    /// they are neither posterior probabilities nor calibrated FDR estimates.
    /// </summary>
    public sealed class TemplateMatchDecoyCount
    {
        public int[] CountsBySearch { get; }
        public int SearchCount => CountsBySearch.Length;
        public long TotalExceedances { get; }
        public double MeanCount => (double)TotalExceedances / SearchCount;
        public double OneCountResolution => 1.0 / SearchCount;
        public bool TailUnresolved => TotalExceedances == 0;

        internal TemplateMatchDecoyCount(int[] counts, long total)
        {
            CountsBySearch = counts;
            TotalExceedances = total;
        }
    }

    public static class TemplateMatchDecoyCalibration
    {
        /// <summary>
        /// Counts final decoy outputs at least as large as the target score.
        /// Each supplied array must come from one successful full search of the
        /// same tilt series with the same proposal/refinement/output limits.
        /// Empty successful searches contribute zero counts and one exposure.
        /// Missing or failed searches must never be represented by empty arrays.
        /// </summary>
        public static TemplateMatchDecoyCount Count(float targetScore, IReadOnlyList<float[]> decoySearches)
        {
            if (!float.IsFinite(targetScore))
                throw new ArgumentOutOfRangeException(nameof(targetScore), "Target score must be finite.");
            if (decoySearches == null || decoySearches.Count == 0)
                throw new ArgumentException("At least one completed decoy search is required.", nameof(decoySearches));

            int[] counts = new int[decoySearches.Count];
            long total = 0;
            for (int search = 0; search < decoySearches.Count; search++)
            {
                float[] scores = decoySearches[search] ??
                    throw new ArgumentException("A missing decoy search cannot be counted as zero output.", nameof(decoySearches));
                foreach (float score in scores)
                {
                    if (!float.IsFinite(score))
                        throw new ArgumentException("Decoy scores must be finite.", nameof(decoySearches));
                    if (score >= targetScore)
                        counts[search]++;
                }
                total += counts[search];
            }
            return new TemplateMatchDecoyCount(counts, total);
        }
    }
}
