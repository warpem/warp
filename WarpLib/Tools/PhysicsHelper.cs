using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using Accord.Math.Optimization;

namespace Warp.Tools
{
    public static class PhysicsHelper
    {
        public static float3[] FillWithEquidistantPoints(Image mask, int n, out float R)
        {
            float3 MaskCenter = mask.AsCenterOfMass();
            float[] MaskData = mask.GetHostContinuousCopy();
            int3 Dims = mask.Dims;
            
            float3[] BestSolution = null;

            float a = 0, b = Dims.X / 2;
            R = (a + b) / 2;
            float3 Offset = new float3(0, 0, 0);

            for (int o = 0; o < 2; o++)
            {
                for (int i = 0; i < 10; i++)
                {
                    R = (a + b) / 2;

                    float Root3 = (float)Math.Sqrt(3);
                    float ZTerm = (float)(2 * Math.Sqrt(6) / 3);
                    float SpacingX = R * 2;
                    float SpacingY = Root3 * R;
                    float SpacingZ = ZTerm * R;
                    int3 DimsSphere = new int3(Math.Min(512, (int)Math.Ceiling(Dims.X / SpacingX)),
                                               Math.Min(512, (int)Math.Ceiling(Dims.Y / SpacingX)),
                                               Math.Min(512, (int)Math.Ceiling(Dims.Z / SpacingX)));
                    BestSolution = new float3[DimsSphere.Elements()];

                    for (int z = 0; z < DimsSphere.Z; z++)
                    {
                        for (int y = 0; y < DimsSphere.Y; y++)
                        {
                            for (int x = 0; x < DimsSphere.X; x++)
                            {
                                BestSolution[DimsSphere.ElementFromPosition(x, y, z)] = new float3(2 * x + (y + z) % 2,
                                                                                                   Root3 * (y + 1 / 3f * (z % 2)),
                                                                                                   ZTerm * z) * R + Offset;
                            }
                        }
                    }

                    List<float3> InsideMask = BestSolution.Where(p =>
                    {
                        int3 ip = new int3(p);
                        if (ip.X >= 0 && ip.X < Dims.X && ip.Y >= 0 && ip.Y < Dims.Y && ip.Z >= 0 && ip.Z < Dims.Z)
                            return MaskData[Dims.ElementFromPosition(new int3(p))] == 1;
                        return false;
                    }).ToList();
                    BestSolution = InsideMask.ToArray();

                    if (BestSolution.Length == n)
                        break;
                    else if (BestSolution.Length < n)
                        b = R;
                    else
                        a = R;
                }

                float3 CenterOfPoints = MathHelper.Mean(BestSolution);
                Offset = MaskCenter - CenterOfPoints;

                a = 0.8f * R;
                b = 1.2f * R;
            }

            BestSolution = BestSolution.Select(v => v + Offset).ToArray();

            return BestSolution;
        }

        public static Image PartitionMask(Image mask, float3[] centroids)
        {
            float[] MaskData = mask.GetHostContinuousCopy();
            int[] Partitions = new int[MaskData.Length];
            int3 Dims = mask.Dims;

            Parallel.For(0, Dims.Z, z =>
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        if (MaskData[Dims.ElementFromPosition(x, y, z)] != 1)
                            Partitions[Dims.ElementFromPosition(x, y, z)] = -1;
                        else
                        {
                            float3 Pos = new float3(x, y, z);
                            float BestDist = float.MaxValue;
                            int BestCentroid = 0;

                            for (int c = 0; c < centroids.Length; c++)
                            {
                                float Dist = (Pos - centroids[c]).LengthSq();
                                if (Dist < BestDist)
                                {
                                    BestDist = Dist;
                                    BestCentroid = c;
                                }
                            }

                            Partitions[Dims.ElementFromPosition(x, y, z)] = BestCentroid;
                        }
                    }
                }
            });

            return new Image(Partitions.Select(v => (float)v).ToArray(), Dims);
        }

        public static double[][] GetHessian(float3[] points, float[] weights, float radius)
        {
            int N = points.Length;
            float Radius2 = radius * radius;

            double[] Adjacency = new double[N * N];
            double[] EdgeLengths = new double[N * N];
            for (int p1 = 0; p1 < N; p1++)
            {
                for (int p2 = 0; p2 <= p1; p2++)
                {
                    double DistSq = (points[p1] - points[p2]).LengthSq();
                    EdgeLengths[p1 * N + p2] = DistSq;
                    EdgeLengths[p2 * N + p1] = DistSq;

                    if (DistSq <= Radius2 * 4)
                    {
                        Adjacency[p1 * N + p2] = Math.Exp(-DistSq / Radius2);
                        Adjacency[p2 * N + p1] = Math.Exp(-DistSq / Radius2);
                    }
                }
            }

            float[] StartVec = Helper.ToInterleaved(points);
            double[][] Hessian = new double[StartVec.Length][];
            for (int i = 0; i < Hessian.Length; i++)
                Hessian[i] = new double[StartVec.Length];

            double[] WeightsSqrt = weights.Select(v => Math.Sqrt(v)).ToArray();

            Parallel.For(0, N, p1 =>
            {
                // Off-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    double[] R = { points[p1].X - points[p2].X, points[p1].Y - points[p2].Y, points[p1].Z - points[p2].Z };
                    double RSq = p1 == p2 ? 1 : EdgeLengths[p1 * N + p2];

                    for (int m = 0; m < 3; m++)
                        for (int n = 0; n < 3; n++)
                            Hessian[p1 * 3 + m][p2 * 3 + n] = -1 * Adjacency[p1 * N + p2] * R[m] * R[n] / RSq;
                }

                // On-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    double[] R = { points[p1].X - points[p2].X, points[p1].Y - points[p2].Y, points[p1].Z - points[p2].Z };
                    double RSq = p1 == p2 ? 1 : EdgeLengths[p1 * N + p2];

                    for (int m = 0; m < 3; m++)
                        for (int n = 0; n < 3; n++)
                            Hessian[p1 * 3 + m][p1 * 3 + n] += Adjacency[p1 * N + p2] * R[m] * R[n] / RSq;
                }
            });

            for (int p1 = 0; p1 < N; p1++)
            {
                for (int p2 = 0; p2 < N; p2++)
                {
                    double W = WeightsSqrt[p1] * WeightsSqrt[p2];

                    for (int m = 0; m < 3; m++)
                        for (int n = 0; n < 3; n++)
                            Hessian[p1 * 3 + m][p2 * 3 + n] *= W;
                }
            }

            return Hessian;
        }

        public static double[][] GetHessian2(float2[] points, float[] weights, float radius)
        {
            int N = points.Length;
            float Radius2 = radius * radius;

            double[] Adjacency = new double[N * N];
            double[] EdgeLengths = new double[N * N];
            for (int p1 = 0; p1 < N; p1++)
            {
                for (int p2 = 0; p2 <= p1; p2++)
                {
                    double DistSq = (points[p1] - points[p2]).LengthSq();
                    EdgeLengths[p1 * N + p2] = DistSq;
                    EdgeLengths[p2 * N + p1] = DistSq;

                    if (DistSq <= Radius2 * 4)
                    {
                        Adjacency[p1 * N + p2] = Math.Exp(-DistSq / Radius2);
                        Adjacency[p2 * N + p1] = Math.Exp(-DistSq / Radius2);
                    }
                }
            }

            float[] StartVec = Helper.ToInterleaved(points);
            double[][] Hessian = new double[StartVec.Length][];
            for (int i = 0; i < Hessian.Length; i++)
                Hessian[i] = new double[StartVec.Length];

            double[] WeightsSqrt = weights.Select(v => Math.Sqrt(v)).ToArray();

            Parallel.For(0, N, p1 =>
            {
                // Off-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    double[] R = { points[p1].X - points[p2].X, points[p1].Y - points[p2].Y };
                    double RSq = p1 == p2 ? 1 : EdgeLengths[p1 * N + p2];

                    for (int m = 0; m < 2; m++)
                        for (int n = 0; n < 2; n++)
                            Hessian[p1 * 2 + m][p2 * 2 + n] = -1 * Adjacency[p1 * N + p2] * R[m] * R[n] / RSq;
                }

                // On-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    double[] R = { points[p1].X - points[p2].X, points[p1].Y - points[p2].Y };
                    double RSq = p1 == p2 ? 1 : EdgeLengths[p1 * N + p2];

                    for (int m = 0; m < 2; m++)
                        for (int n = 0; n < 2; n++)
                            Hessian[p1 * 2 + m][p1 * 2 + n] += Adjacency[p1 * N + p2] * R[m] * R[n] / RSq;
                }
            });

            for (int p1 = 0; p1 < N; p1++)
            {
                for (int p2 = 0; p2 < N; p2++)
                {
                    double W = WeightsSqrt[p1] * WeightsSqrt[p2];

                    for (int m = 0; m < 2; m++)
                        for (int n = 0; n < 2; n++)
                            Hessian[p1 * 2 + m][p2 * 2 + n] *= W;
                }
            }

            return Hessian;
        }

        public static double[][] GetHessian1(float[] points, float[] weights, float radius)
        {
            int N = points.Length;
            float Radius2 = radius * radius;

            double[] Adjacency = new double[N * N];
            double[] EdgeLengths = new double[N * N];
            for (int p1 = 0; p1 < N; p1++)
            {
                for (int p2 = 0; p2 <= p1; p2++)
                {
                    double DistSq = (points[p1] - points[p2]) * (points[p1] - points[p2]);
                    EdgeLengths[p1 * N + p2] = DistSq;
                    EdgeLengths[p2 * N + p1] = DistSq;

                    if (DistSq <= Radius2 * 4)
                    {
                        Adjacency[p1 * N + p2] = Math.Exp(-DistSq / Radius2);
                        Adjacency[p2 * N + p1] = Math.Exp(-DistSq / Radius2);
                    }
                }
            }

            float[] StartVec = points.ToArray();
            double[][] Hessian = new double[StartVec.Length][];
            for (int i = 0; i < Hessian.Length; i++)
                Hessian[i] = new double[StartVec.Length];

            double[] WeightsSqrt = weights.Select(v => Math.Sqrt(v)).ToArray();

            Parallel.For(0, N, p1 =>
            {
                // Off-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    double[] R = { points[p1] - points[p2] };
                    double RSq = p1 == p2 ? 1 : EdgeLengths[p1 * N + p2];

                    Hessian[p1][p2] = -1 * Adjacency[p1 * N + p2] * R[0] * R[0] / RSq;
                }

                // On-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    double[] R = { points[p1] - points[p2] };
                    double RSq = p1 == p2 ? 1 : EdgeLengths[p1 * N + p2];

                    Hessian[p1][p1] += Adjacency[p1 * N + p2] * R[0] * R[0] / RSq;
                }
            });

            for (int p1 = 0; p1 < N; p1++)
            {
                for (int p2 = 0; p2 < N; p2++)
                {
                    double W = WeightsSqrt[p1] * WeightsSqrt[p2];

                    Hessian[p1][p2] *= W;
                }
            }

            return Hessian;
        }

        public static float[][] GetHessianSingle(float3[] points, float[] weights, float radius)
        {
            int N = points.Length;
            float Radius2 = radius * radius;
            float Radius24 = Radius2 * 4;

            float[] Adjacency = new float[N * N];
            float[] EdgeLengths = new float[N * N];
            for (int p1 = 0; p1 < N; p1++)
            {
                for (int p2 = 0; p2 <= p1; p2++)
                {
                    float DistSq = (points[p1] - points[p2]).LengthSq();
                    EdgeLengths[p1 * N + p2] = DistSq;
                    EdgeLengths[p2 * N + p1] = DistSq;

                    if (DistSq <= Radius24)
                    {
                        Adjacency[p1 * N + p2] = (float)Math.Exp(-DistSq / Radius2);
                        Adjacency[p2 * N + p1] = (float)Math.Exp(-DistSq / Radius2);
                    }
                }
            }

            float[] StartVec = Helper.ToInterleaved(points);
            float[][] Hessian = Helper.ArrayOfFunction(i => new float[StartVec.Length], StartVec.Length);

            float[] WeightsSqrt = weights.Select(v => (float)Math.Sqrt(v)).ToArray();

            Parallel.For(0, N, p1 =>
            {
                // Off-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    float[] R = { points[p1].X - points[p2].X, points[p1].Y - points[p2].Y, points[p1].Z - points[p2].Z };
                    float RSq = p1 == p2 ? 1 : EdgeLengths[p1 * N + p2];

                    for (int m = 0; m < 3; m++)
                        for (int n = 0; n < 3; n++)
                            Hessian[p1 * 3 + m][p2 * 3 + n] = -1 * Adjacency[p1 * N + p2] * R[m] * R[n] / RSq;
                }

                // On-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    float[] R = { points[p1].X - points[p2].X, points[p1].Y - points[p2].Y, points[p1].Z - points[p2].Z };
                    float RSq = p1 == p2 ? 1 : EdgeLengths[p1 * N + p2];

                    for (int m = 0; m < 3; m++)
                        for (int n = 0; n < 3; n++)
                            Hessian[p1 * 3 + m][p1 * 3 + n] += Adjacency[p1 * N + p2] * R[m] * R[n] / RSq;
                }
            });

            for (int p1 = 0; p1 < N; p1++)
            {
                for (int p2 = 0; p2 < N; p2++)
                {
                    float W = WeightsSqrt[p1] * WeightsSqrt[p2];

                    for (int m = 0; m < 3; m++)
                        for (int n = 0; n < 3; n++)
                            Hessian[p1 * 3 + m][p2 * 3 + n] *= W;
                }
            }

            return Hessian;
        }

        public static float[][] IterativePCA(float[][] data, int ncomponents)
        {
            int N = data[0].Length;
            int L = data.Length;

            float[] Mean = new float[N];
            for (int n = 0; n < N; n++)
            {
                for (int l = 0; l < L; l++)
                    Mean[n] += data[l][n];

                Mean[n] /= L;
            }

            for (int n = 0; n < N; n++)
                for (int l = 0; l < L; l++)
                    data[l][n] -= Mean[n];

            List<float[]> Components = new List<float[]>();
            float[][] FinalScores = new float[ncomponents][];
            RandomNormal Rand = new RandomNormal(123);

            float[][] DataWork = new float[L][];
            for (int l = 0; l < L; l++)
                DataWork[l] = data[l].ToArray();

            for (int icomp = 0; icomp < ncomponents; icomp++)
            {
                float[] Component = MathHelper.NormalizeL2(Helper.ArrayOfFunction(i => Rand.NextSingle(0, 1), N));
                //float[] Component = MathHelper.NormalizeL2(Helper.ArrayOfConstant(1f, N));
                //float[] Component = MathHelper.NormalizeL2(Helper.ArrayOfFunction(i => (float)i - N / 2, N));

                for (int iiter = 0; iiter < 30; iiter++)
                {
                    //Component = MathHelper.NormalizeL2(Component);

                    float[] Scores = new float[L];
                    for (int l = 0; l < L; l++)
                        Scores[l] = MathHelper.DotProduct(Component, DataWork[l]);
                    
                    FinalScores[icomp] = Scores;

                    Component = new float[N];

                    for (int l = 0; l < L; l++)
                        for (int n = 0; n < N; n++)
                            Component[n] += DataWork[l][n] * Scores[l];

                    if (icomp > 0)
                    {
                        for (int icompprev = 0; icompprev < icomp; icompprev++)
                        {
                            float Dot = MathHelper.DotProduct(Components[icompprev], Component);
                            float[] Proj = Components[icompprev].Select(v => v * Dot).ToArray();
                            for (int n = 0; n < N; n++)
                                Component[n] -= Proj[n];
                        }
                    }

                    Component = MathHelper.NormalizeL2(Component);

                    for (int l = 0; l < L; l++)
                        Scores[l] = MathHelper.DotProduct(Component, DataWork[l]);

                    FinalScores[icomp] = Scores;
                }

                Components.Add(Component);

                for (int l = 0; l < L; l++)
                    for (int n = 0; n < N; n++)
                        DataWork[l][n] -= Component[n] * FinalScores[icomp][l];
            }

            for (int icomp = 0; icomp < ncomponents; icomp++)
                for (int l = 0; l < L; l++)
                    FinalScores[icomp][l] = MathHelper.DotProduct(Components[icomp], data[l]);

            float[] Significance = FinalScores.Select(a => a.Select(v => v * v).Sum()).ToArray();

            return Components.ToArray();
        }
        public static void GetSparseHessian(float3[] points, float[] weights, float radius, out int[] matrixI, out int[] matrixJ, out double[] matrixValues)
        {
            int N = points.Length;
            float Radius2 = radius * radius;

            double[][] Adjacency = new double[N][];
            double[][] EdgeLengths = new double[N][];
            for (int p1 = 0; p1 < N; p1++)
            {
                Adjacency[p1] = new double[N];
                EdgeLengths[p1] = new double[N];

                for (int p2 = 0; p2 <= p1; p2++)
                {
                    double DistSq = (points[p1] - points[p2]).LengthSq();
                    EdgeLengths[p1][p2] = DistSq;
                    EdgeLengths[p2][p1] = DistSq;

                    if (DistSq <= Radius2 * 4)
                    {
                        Adjacency[p1][p2] = Math.Exp(-DistSq / Radius2);
                        Adjacency[p2][p1] = Math.Exp(-DistSq / Radius2);
                    }
                }
            }

            List<double>[] SparseColumns = new List<double>[N * 3];
            List<int>[] SparseIndices = new List<int>[N * 3];
            for (int n = 0; n < N * 3; n++)
            {
                SparseColumns[n] = new List<double>(N * 3 / 100);
                SparseIndices[n] = new List<int>(N * 3 / 100);
            }

            Parallel.For(0, N, p1 =>
            {
                // Off-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    double AdjValue = Adjacency[p1][p2];
                    if (AdjValue == 0)
                        continue;

                    double[] R = { points[p1].X - points[p2].X, points[p1].Y - points[p2].Y, points[p1].Z - points[p2].Z };
                    double RSq = p1 == p2 ? 1 : EdgeLengths[p1][p2];

                    for (int m = 0; m < 3; m++)
                        for (int n = 0; n < 3; n++)
                        {
                            int Row = p2 * 3 + n;
                            SparseColumns[p1 * 3 + m].Add(-1 * AdjValue * R[m] * R[n] / RSq);
                            SparseIndices[p1 * 3 + m].Add(Row);
                        }
                }
            });

            Parallel.For(0, N, p1 =>
            {
                // On-diagonal
                for (int p2 = 0; p2 < N; p2++)
                {
                    double AdjValue = Adjacency[p1][p2];
                    if (AdjValue == 0)
                        continue;

                    double[] R = { points[p1].X - points[p2].X, points[p1].Y - points[p2].Y, points[p1].Z - points[p2].Z };
                    double RSq = p1 == p2 ? 1 : EdgeLengths[p1][p2];

                    for (int m = 0; m < 3; m++)
                        for (int n = 0; n < 3; n++)
                        {
                            int Row = p1 * 3 + n;
                            int SparseIndex = 0;
                            while (SparseIndices[p1 * 3 + m][SparseIndex] != Row) SparseIndex++;

                            SparseColumns[p1 * 3 + m][SparseIndex] += AdjValue * R[m] * R[n] / RSq;
                        }
                }
            });

            double[] WeightsSqrt = weights.Select(v => Math.Sqrt(v)).ToArray();

            for (int p1m = 0; p1m < N * 3; p1m++)
            {
                for (int sparseIndex = 0; sparseIndex < SparseIndices[p1m].Count; sparseIndex++)
                {
                    int p2n = SparseIndices[p1m][sparseIndex];
                    double W = WeightsSqrt[p1m / 3] * WeightsSqrt[p2n / 3];

                    SparseColumns[p1m][sparseIndex] *= W;
                }
            }

            int NSparse = SparseIndices.Select(l => l.Count).Sum();
            matrixI = new int[NSparse];
            matrixJ = new int[NSparse];
            matrixValues = new double[NSparse];

            for (int p1 = 0, i = 0; p1 < N * 3; p1++)
            {
                for (int sparseIndex = 0; sparseIndex < SparseIndices[p1].Count; sparseIndex++, i++)
                {
                    int p2 = SparseIndices[p1][sparseIndex];

                    matrixI[i] = p1;
                    matrixJ[i] = p2;
                    matrixValues[i] = SparseColumns[p1][sparseIndex];
                }
            }
        }

        public static Image GetVolumeFromAtoms(float3[] atoms, int3 dims, float sigma, float[] weights = null, float blobRadius = 1.0f, float blobAlpha = 6f)
        {
            if (weights == null)
                weights = atoms.Select(v => 1f).ToArray();

            float BlobRadius = blobRadius * (sigma / 0.5f);
            float BlobAlpha = blobAlpha;
            int BlobOrder = 0;
            KaiserTable KaiserT = new KaiserTable(1000, BlobRadius, BlobAlpha, BlobOrder);

            int SigmaExtent = (int)Math.Ceiling(BlobRadius);
            float[] VolumeData = new float[dims.Elements()];
            sigma = 2 * sigma * sigma;

            float[][] ParallelVolumes = new float[10][];
            for (int i = 0; i < ParallelVolumes.Length; i++)
                ParallelVolumes[i] = new float[VolumeData.Length];

            Parallel.For(0, 10, p =>
            {
                for (int a = p; a < atoms.Length; a += 10)
                {
                    float3 atom = atoms[a];
                    float weight = weights[a];

                    int3 IAtom = new int3(atom.Round());
                    int StartZ = Math.Max(0, IAtom.Z - SigmaExtent);
                    int EndZ = Math.Min(dims.Z - 1, IAtom.Z + SigmaExtent);

                    for (int z = StartZ; z <= EndZ; z++)
                    {
                        int StartY = Math.Max(0, IAtom.Y - SigmaExtent);
                        int EndY = Math.Min(dims.Y - 1, IAtom.Y + SigmaExtent);

                        for (int y = StartY; y <= EndY; y++)
                        {
                            int StartX = Math.Max(0, IAtom.X - SigmaExtent);
                            int EndX = Math.Min(dims.X - 1, IAtom.X + SigmaExtent);

                            for (int x = StartX; x <= EndX; x++)  
                            {
                                float3 Pos = new float3(x, y, z);
                                //ParallelVolumes[p][(z * dims.Y + y) * dims.X + x] += (float)Math.Exp(-(Pos - atom).LengthSq() / sigma) * weight;
                                ParallelVolumes[p][(z * dims.Y + y) * dims.X + x] += KaiserT.GetValue((Pos - atom).Length()) * weight;
                            }
                        }
                    }
                }
            });

            for (int p = 0; p < ParallelVolumes.Length; p++)
                for (int i = 0; i < VolumeData.Length; i++)
                    VolumeData[i] += ParallelVolumes[p][i];

            return new Image(VolumeData, dims);
        }

        public static float[] MatchVolumeIntensities(float3[] atoms, Image volume, Image mask, float radius, out float sigma, out float correlation, float blobRadius = 1.0f, float blobAlpha = 6f)
        {
            int3 Dims = volume.Dims;

            float[] MaskData = mask.GetHostContinuousCopy();
            List<int> MaskIndices = new List<int>();
            for (int i = 0; i < MaskData.Length; i++)
                if (MaskData[i] == 1)
                    MaskIndices.Add(i);

            int3 DimsUpscaled = Dims * 2;
            DimsUpscaled.Z = DimsUpscaled.Z > 2 ? DimsUpscaled.Z : 1;
            Image VolumeUpscaled;
            if (DimsUpscaled.Z == 1)
                VolumeUpscaled = volume.AsScaled(new int2(DimsUpscaled));
            else
                VolumeUpscaled = volume.AsScaled(DimsUpscaled);
            volume.FreeDevice();
            VolumeUpscaled.FreeDevice();

            float[] VolumeData = volume.GetHostContinuousCopy();
            float[] VolumeDataUpscaled = VolumeUpscaled.GetHostContinuousCopy();
            float[] VolumeSparse = new float[MaskIndices.Count];
            for (int i = 0; i < MaskIndices.Count; i++)
                VolumeSparse[i] = VolumeData[MaskIndices[i]];
            MathHelper.NormalizeInPlace(VolumeSparse);

            Func<float, float[]> GetIntensities = s =>
            {
                float[] Result = new float[atoms.Length];
                
                float3[] Normalized = atoms.Select(a => a * 2 * new float3(1f / (DimsUpscaled.X - 1), 1f / (DimsUpscaled.Y - 1), 1f / (DimsUpscaled.Z - 1))).ToArray();
                CubicGrid Grid = new CubicGrid(DimsUpscaled, VolumeDataUpscaled);

                Result = Grid.GetInterpolatedNative(Normalized);

                return Result;
            };

            Func<double[], double> Eval = input =>
            {
                float[] CurIntensities = GetIntensities((float)Math.Log(input[0]));
                Image Simulated = GetVolumeFromAtoms(atoms, volume.Dims, (float)Math.Log(input[0]), CurIntensities, blobRadius, blobAlpha);
                float[] SimulatedData = Simulated.GetHostContinuousCopy();
                float[] SimulatedSparse = new float[MaskIndices.Count];
                for (int i = 0; i < MaskIndices.Count; i++)
                    SimulatedSparse[i] = SimulatedData[MaskIndices[i]];
                MathHelper.NormalizeInPlace(SimulatedSparse);

                float[] Corr = MathHelper.Mult(SimulatedSparse, VolumeSparse);

                return Corr.Sum() / Corr.Length;
            };

            sigma = radius * 1f;
            float[] FinalIntensities = GetIntensities(sigma);
            correlation = (float)Eval(new[] { Math.Exp(sigma) });

            return FinalIntensities;
        }

        public static Image SimulateElectronEvents(Image map, float eventsPerPixel, float backgroundFraction = 0.5f, int seed = 123)
        {
            List<float> MapSorted = map.GetHostContinuousCopy().ToList();
            MapSorted.Sort();

            float Min = MapSorted[(int)(0.02 * MapSorted.Count)];
            float Max = MapSorted[(int)(0.98 * MapSorted.Count)];
            float Range = Max - Min;

            int Bins = 1024;
            int BinsBackground = (int)(backgroundFraction * Bins);
            int BinsSignal = Bins - BinsBackground;

            float[] MapData = map.GetHostContinuousCopy();
            List<int> Probabilities = new List<int>(MapData.Length * Bins);

            for (int i = 0; i < MapData.Length; i++)
            {
                int Probability = Math.Max(0, Math.Min(BinsSignal, (int)((MapData[i] - Min) / Range * BinsSignal))) + BinsBackground;
                for (int j = 0; j < Probability; j++)
                    Probabilities.Add(i);
            }

            Image Simulation = new Image(map.Dims);
            float[] SimulationData = Simulation.GetHost(Intent.Write)[0];
            Random Rand = new Random(seed);
            RandomNormal RandN = new RandomNormal(seed);

            int NElectrons = (int)(eventsPerPixel * MapData.Length);
            for (int e = 0; e < NElectrons; e++)
            {
                int i = Probabilities[Rand.Next(Probabilities.Count)];
                SimulationData[i] += RandN.NextSingle(1, 0.3f);
            }

            return Simulation;
        }

        public static void GetAtomsFromPDB(string path, 
                                           bool centered, 
                                           out float3[] points, 
                                           out string[] atomNames,
                                           out string[] elements, 
                                           out float[] intensities, 
                                           out string[] residueNames, 
                                           out int[] residueIds, 
                                           out char[] chains, 
                                           out string[] textRows)
        {
            List<float3> Points = new List<float3>();
            List<string> AtomNames = new List<string>();
            List<string> Elements = new List<string>();
            List<float> Intensities = new List<float>();
            List<string> ResidueNames = new List<string>();
            List<int> ResidueIDs = new List<int>();
            List<char> Chains = new List<char>();
            List<string> Rows = new List<string>();

            using (TextReader Reader = File.OpenText(path))
            {
                List<char> ValidChains = new List<char>();
                bool ParsedChains = false;

                string Line = null;
                while ((Line = Reader.ReadLine()) != null)
                {
                    while (!ParsedChains && Line.Substring(0, 10) == "REMARK 350")
                    {
                        if (Line.Contains(" CHAINS: "))
                        {
                            char[] ValidChain = Line.Substring(Line.LastIndexOf(" CHAINS: ") + " CHAINS: ".Length).Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Select(s => s.Trim()[0]).ToArray();
                            ValidChains.AddRange(Chains);
                        }

                        if (Line.Contains("BIOMOLECULE: 2"))
                        {
                            ParsedChains = true;
                            break;
                        }

                        Line = Reader.ReadLine();
                    }

                    if (Line.Length < 4)
                        continue;

                    if (Line.Substring(0, 4) == "ATOM")
                    {
                        ParsedChains = true;

                        char Chain = Line[21];
                        if (ValidChains.Count > 0 && !ValidChains.Contains(Chain))
                            continue;

                        string ResidueName = Line.Substring(17, 3).Trim();
                        int ResidueID = int.Parse(Line.Substring(22, 4));

                        float3 AtomPos = new float3(float.Parse(Line.Substring(30, 8)),
                                                    float.Parse(Line.Substring(38, 8)),
                                                    float.Parse(Line.Substring(46, 8)));

                        string Element = Line.Substring(76, 2).Trim();
                        float Mass = 1;
                        switch (Element[0])
                        {
                            case 'H':
                                Mass = 1.00794f;
                                break;
                            case 'C':
                                Mass = 12.0107f;
                                break;
                            case 'N':
                                Mass = 14.0067f;
                                break;
                            case 'O':
                                Mass = 15.9994f;
                                break;
                            case 'P':
                                Mass = 30.973762f;
                                break;
                        }

                        Points.Add(AtomPos);
                        AtomNames.Add(Line.Substring(12, 3).Trim());
                        Elements.Add(Element);
                        Intensities.Add(Mass);
                        ResidueNames.Add(ResidueName);
                        ResidueIDs.Add(ResidueID);
                        Chains.Add(Chain);
                        Rows.Add(Line);
                    }
                }
            }

            points = Points.ToArray();
            atomNames = AtomNames.ToArray();
            elements = Elements.ToArray();
            intensities = Intensities.ToArray();
            residueNames = ResidueNames.ToArray();
            residueIds = ResidueIDs.ToArray();
            chains = Chains.ToArray();
            textRows = Rows.ToArray();

            if (centered)
            {
                float3 Center = MathHelper.MeanWeighted(points, intensities);
                points = points.Select(v => v - Center).ToArray();

                for (int r = 0; r < Rows.Count; r++)
                {
                    string X = points[r].X.ToString("F2", CultureInfo.InvariantCulture);
                    while (X.Length < 8)
                        X = " " + X;
                    string Y = points[r].Y.ToString("F2", CultureInfo.InvariantCulture);
                    while (Y.Length < 8)
                        Y = " " + Y;
                    string Z = points[r].Z.ToString("F2", CultureInfo.InvariantCulture);
                    while (Z.Length < 8)
                        Z = " " + Z;

                    string L = textRows[r];
                    textRows[r] = L.Substring(0, 30) + X + Y + Z + L.Substring(54);
                }
            }
        }
    }

    class WeightedPoint
    {
        public float3 Position;
        public float Weight;

        public WeightedPoint(float3 position, float weight)
        {
            Position = position;
            Weight = weight;
        }
    }
}
