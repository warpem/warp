using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public static class ClusterTrieste
    {
        public static Tuple<T, int, float, float>[] Cluster<T>(T[] items,
                                                               Func<T, T, float> distFunc,
                                                               Func<float, float> weightFunc,
                                                               float cutoffDist,
                                                               float cutoffRatio,
                                                               float cutoffPercentile = -1,
                                                               int k = -1,
                                                               float haloDist = -1)
        {
            if (cutoffPercentile > 0)
            {
                List<float> Cutoffs = new List<float>();

                Parallel.For(0, items.Length, i1 =>
                {
                    float[] Distances = new float[items.Length - 1];

                    for (int i2 = 0; i2 < items.Length; i2++)
                    {
                        if (i1 == i2)
                            continue;

                        Distances[i2 < i1 ? i2 : i2 - 1] = distFunc(items[i1], items[i2]);
                    }

                    List<float> DistancesList = new List<float>(Distances);
                    DistancesList.Sort();

                    lock (Cutoffs)
                        Cutoffs.Add(DistancesList[(int)(Distances.Length * (cutoffPercentile * 0.01f))]);
                });

                cutoffDist = MathHelper.Median(Cutoffs);
            }

            float[] LocalDensity = new float[items.Length];
            float[] DistanceToSuperior = new float[items.Length];
            int[] SuperiorIndex = new int[items.Length];
            float[] Ratio = new float[items.Length];
            int[] ClusterID = new int[items.Length].Select(v => -1).ToArray();

            // Calculate local density
            {
                Parallel.For(0, items.Length, i1 =>
                {
                    float DensitySum = 0;

                    for (int i2 = 0; i2 < items.Length; i2++)
                    {
                        if (i1 == i2)
                            continue;

                        float Dist = distFunc(items[i1], items[i2]);
                        //MaxDistance = Math.Max(MaxDistance, Dist);
                        if (weightFunc == null && Dist <= cutoffDist)
                            DensitySum++;
                        else
                            DensitySum += weightFunc(Dist / cutoffDist);
                    }

                    LocalDensity[i1] = DensitySum;
                });
            }

            // Find nearest superior neighbor
            {
                Parallel.For(0, items.Length, i1 =>
                {
                    int BestIndex = -1;
                    float BestDensity = LocalDensity[i1];
                    float BestDistance = float.MaxValue;

                    for (int i2 = 0; i2 < items.Length; i2++)
                        if (LocalDensity[i2] > BestDensity)
                        {
                            float Dist = distFunc(items[i1], items[i2]);
                            if (Dist < BestDistance)
                            {
                                BestIndex = i2;
                                BestDistance = Dist;
                            }
                        }

                    DistanceToSuperior[i1] = BestDistance;
                    SuperiorIndex[i1] = BestIndex;
                    Ratio[i1] = DistanceToSuperior[i1] * LocalDensity[i1];
                });

                float MaxDistance = MathHelper.Max(DistanceToSuperior.Where(v => v < float.MaxValue));
                for (int i = 0; i < items.Length; i++)
                    if (SuperiorIndex[i] < 0)
                        DistanceToSuperior[i] = MaxDistance;
            }

            // Find right ratio cutoff to get right number of clusters
            if (k > 0)
            {
                List<float> RatioList = new List<float>(Ratio);
                RatioList.Sort();
                cutoffRatio = RatioList[RatioList.Count - k];
            }

            // Assign cluster IDs to peaks
            int nk = 0;
            {

                for (int i = 0; i < items.Length; i++)
                    if (Ratio[i] >= cutoffRatio)
                        ClusterID[i] = nk++;
            }

            // Assign cluster IDs to everyone else
            {
                for (int i = 0; i < items.Length; i++)
                    ClusterID[i] = GetSuperiorClusterID(i, SuperiorIndex, ClusterID);
            }

            if (haloDist > 0)
            {
                bool[] IsBorder = new bool[items.Length];
                Parallel.For(0, items.Length, i1 =>
                {
                    for (int i2 = 0; i2 < items.Length; i2++)
                        if (ClusterID[i1] != ClusterID[i2] && distFunc(items[i1], items[i2]) < haloDist)
                        {
                            IsBorder[i1] = true;
                            break;
                        }
                });

                List<float>[] BorderValues = new List<float>[nk].Select(v => new List<float>()).ToArray();

                for (int i = 0; i < items.Length; i++)
                    if (IsBorder[i])
                        BorderValues[ClusterID[i]].Add(LocalDensity[i]);

                float[] BorderDensity = BorderValues.Select(v => v.Count > 0 ? MathHelper.Max(v) : 0).ToArray();

                for (int i = 0; i < items.Length; i++)
                    if (LocalDensity[i] < BorderDensity[ClusterID[i]])
                        ClusterID[i] = -1;
            }

            Tuple<T, int, float, float>[] Result = new Tuple<T, int, float, float>[items.Length];

            for (int i = 0; i < items.Length; i++)
                Result[i] = new Tuple<T, int, float, float>(items[i], ClusterID[i], DistanceToSuperior[i], LocalDensity[i]);

            return Result;
        }

        private static int GetSuperiorClusterID(int i, int[] neighborIndex, int[] cluster)
        {
            if (cluster[i] > 0 || neighborIndex[i] < 0)
                return cluster[i];

            int C = GetSuperiorClusterID(neighborIndex[i], neighborIndex, cluster);
            cluster[neighborIndex[i]] = C;

            return C;
        }
    }
}
