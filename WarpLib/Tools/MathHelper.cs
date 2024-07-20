using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Accord;
using MathNet.Numerics.Statistics;

namespace Warp.Tools
{
    public static class MathHelper
    {
        public static float Sinc(float v)
        {
            if (MathF.Abs(v) < 1e-8f)
                return 1;

            return MathF.Sin(v * MathF.PI) / (v * MathF.PI);
        }
        public static float Sinc2(float v)
        {
            if (MathF.Abs(v) < 1e-8f)
                return 1;

            float Result = MathF.Sin(v * MathF.PI) / (v * MathF.PI);
            return Result * Result;
        }

        public static float Mean(IEnumerable<float> data)
        {
            double Sum = data.Sum(i => i);
            return (float)Sum / data.Count();
        }

        public static float2 Mean(IEnumerable<float2> data)
        {
            float2 Sum = new float2(0, 0);
            foreach (var p in data)
                Sum += p;

            return Sum / data.Count();
        }

        public static float3 Mean(IEnumerable<float3> data)
        {
            float3 Sum = new float3(0, 0, 0);
            foreach (var p in data)
                Sum += p;

            return Sum / data.Count();
        }

        public static float4 Mean(IEnumerable<float4> data)
        {
            float4 Sum = new float4(0, 0, 0, 0);
            foreach (var p in data)
                Sum += p;

            return Sum / data.Count();
        }

        public static float5 Mean(IEnumerable<float5> data)
        {
            float5 Sum = new float5(0, 0, 0, 0, 0);
            foreach (var p in data)
                Sum += p;

            return Sum / data.Count();
        }

        public static float Median(float[] data)
        {
            Array.Sort(data);
            return data[data.Length / 2];
        }

        public static float2 GeometricMedian(IEnumerable<float2> points, float tolerance = 1e-5f, int maxIterations = 100)
        {
            // Initial guess as the centroid
            float2 Guess = Mean(points);

            for (int i = 0; i < maxIterations; i++)
            {
                float2 NewGuess = new float2(0, 0);
                float WeightSum = 0;

                foreach (float2 point in points)
                {
                    float Distance = (Guess - point).Length();
                    if (Distance > 0)
                    {
                        float Weight = 1 / Distance;
                        NewGuess += point * Weight;
                        WeightSum += Weight;
                    }
                }
                
                // Avoid division by zero which happens if all points are identical
                // in this case, the arithmetic mean (Guess) is the geometric mean
                if (WeightSum == 0)
                {
                    return Guess;
                }

                NewGuess /= WeightSum;

                if ((NewGuess - Guess).Length() < tolerance)
                    return NewGuess;

                Guess = NewGuess;
            }

            return Guess;
        }

        public static float[] MovingWindowMedian(float[] data, int windowSize)
        {
            float[] Filtered = new float[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                float[] Window = Helper.Subset(data, Math.Max(0, i - windowSize / 2), Math.Min(data.Length, i + windowSize / 2 + 1));
                Filtered[i] = Median(Window);
            }

            return Filtered;
        }

        public static float StdDev(IEnumerable<float> data)
        {
            double Sum = 0f, Sum2 = 0f;
            foreach (var i in data)
            {
                Sum += i;
                Sum2 += (double)i * i;
            }

            return MathF.Sqrt((float)Math.Max(0, data.Count() * Sum2 - Sum * Sum)) / data.Count();
        }

        public static float2 MeanAndStd(IEnumerable<float> data)
        {
            double Sum = 0f, Sum2 = 0f;
            foreach (var i in data)
            {
                Sum += i;
                Sum2 += i * i;
            }

            if (Sum == Sum2)
                return new float2(0, 0);

            return new float2((float)Sum / data.Count(), MathF.Sqrt((float)Math.Max(0, data.Count() * Sum2 - Sum * Sum)) / data.Count());
        }

        public static float2 MeanAndStdNonZero(IEnumerable<float> data)
        {
            double Sum = 0f, Sum2 = 0f;
            long Samples = 0;

            foreach (var i in data)
            {
                if (i == 0)
                    continue;

                Sum += i;
                Sum2 += i * i;
                Samples++;
            }

            return new float2((float)Sum / Samples, MathF.Sqrt((float)Math.Max(0, Samples * Sum2 - Sum * Sum)) / Samples);
        }


        public static float2 MeanAndStdNonMask(IEnumerable<float> data)
        {
            double Sum = 0f, Sum2 = 0f;
            long Samples = 0;
            // We use a simplistic heuristic to guess whether the image is masked
            // If the first two values are equal, we assume the image is masked
            // If the image is masked, we ignore the values of pixels that are masked
            if (NearlyEqual(data.ElementAt(0), data.ElementAt(1)))
            {
                float IgnoreMe = data.ElementAt(0);
                foreach (var i in data)
                {
                    if (NearlyEqual(i,IgnoreMe,3))
                        continue;

                    Sum += i;
                    Sum2 += i * i;
                    Samples++;
                }
            }
            else
            {
                foreach (var i in data)
                {
                    if (i == 0.0f)
                        continue;

                    Sum += i;
                    Sum2 += i * i;
                    Samples++;
                }
            }

            return new float2((float)Sum / Samples, MathF.Sqrt((float)Math.Max(0, Samples * Sum2 - Sum * Sum)) / Samples);
        }

        public static float2 MedianAndStd(IEnumerable<float> data)
        {
            double Median = data.Median();
            double SqSum = 0;

            foreach (var v in data)
            {
                double Diff = v - Median;
                SqSum += Diff * Diff;
            }

            return new float2((float)Median, (float)Math.Sqrt(SqSum / data.Count()));
        }

        public static float2 MedianAndPercentileDiff(IEnumerable<float> data, decimal percentile)
        {
            float Median = data.Median();
            var Diff = data.Select(v => MathF.Abs(v - Median));
            float PercentileDiff = Percentile(Diff, percentile);

            return new float2(Median, PercentileDiff);
        }

        public static bool NearlyEqual(float a, float b, int tolerance_significant_digits = 6)
        {
            // Special cases
            if (float.IsInfinity(a) || float.IsNaN(a) || float.IsInfinity(b) || float.IsNaN(b))
            {
                return false;
            }

            // If a or b is zero
            if (a == 0 || b == 0)
            {
                // It's nearly equal if the absolute difference is less than the tolerance
                return MathF.Abs(a - b) < MathF.Pow(10, -tolerance_significant_digits);
            }

            // Calculate the absolute value of log10 of a and b
            float log10a = MathF.Log10(MathF.Abs(a));
            float log10b = MathF.Log10(MathF.Abs(b));

            // If the two values are more than 10x apart, they are not equal!
            if (MathF.Abs(log10a - log10b) > 1.0f)
            {
                return false;
            }

            // Calculate the difference relative to the maximum absolute value
            float diff = MathF.Abs(a - b);
            float maxAbsVal = MathF.Max(MathF.Abs(a), MathF.Abs(b));

            // Check if the relative difference is less than the tolerance
            return diff / maxAbsVal <= MathF.Pow(10, -tolerance_significant_digits);
        }


        public static float[] Normalize(float[] data)
        {
            double Sum = 0f, Sum2 = 0f;
            foreach (var i in data)
            {
                Sum += i;
                Sum2 += i * i;
            }

            float Std = MathF.Sqrt((float)Math.Max(0, data.Length * Sum2 - Sum * Sum)) / data.Count();
            float Avg = (float) Sum / data.Length;

            float[] Result = new float[data.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = (data[i] - Avg) / Std;

            return Result;
        }

        public static void NormalizeInPlace(float[] data)
        {
            unsafe
            {
                fixed (float* DataPtr = data)
                {
                    float* DataP = DataPtr;
                    double Sum = 0f, Sum2 = 0f;
                    for (int i = 0; i < data.Length; i++)
                    {
                        double Val = *DataP++;
                        Sum += Val;
                        Sum2 += Val * Val;
                    }

                    float Std = MathF.Sqrt((float)Math.Max(0, data.Length * Sum2 - Sum * Sum)) / data.Length;
                    float Avg = (float)Sum / data.Length;

                    DataP = DataPtr;
                    for (int i = 0; i < data.Length; i++)
                    {
                        *DataP = (*DataP - Avg) / Std;
                        DataP++;
                    }
                }
            }
        }

        public static float L2(float[] data)
        {
            double Sum = 0;
            for (int i = 0; i < data.Length; i++)
                Sum += (double)data[i] * (double)data[i];
            Sum = Math.Sqrt(Sum);

            return (float)Sum;
        }

        public static float[] NormalizeL2(float[] data)
        {
            float Sum = L2(data);
            float[] Result = new float[data.Length];

            for (int i = 0; i < data.Length; i++)
                Result[i] = data[i] / Sum;

            return Result;
        }

        public static void NormalizeL2InPlace(float[] data)
        {
            float Sum = L2(data);

            for (int i = 0; i < data.Length; i++)
                data[i] /= Sum;
        }

        public static int[] Histogram(IEnumerable<float> data, int nbins, float min = float.NaN, float max = float.NaN)
        {
            if (float.IsNaN(min))
                min = Min(data);
            if (float.IsNaN(max))
                max = Max(data);

            float Range = max - min;

            int[] HistogramBins = new int[nbins];

            foreach (float v in data)
                HistogramBins[Math.Max(0, Math.Min(nbins - 1, (int)((v - min) / Range * (nbins - 1) + 0.5)))]++;

            return HistogramBins;
        }

        public static int[] Histogram2D(float[] data1, float[] data2, int nbins, float min1 = float.NaN, float max1 = float.NaN, float min2 = float.NaN, float max2 = float.NaN)
        {
            if (float.IsNaN(min1))
                min1 = Min(data1);
            if (float.IsNaN(max1))
                max1 = Max(data1);
            if (float.IsNaN(min2))
                min2 = Min(data2);
            if (float.IsNaN(max2))
                max2 = Max(data2);

            float Range1 = max1 - min1;
            float Range2 = max2 - min2;

            int[] HistogramBins = new int[nbins * nbins];

            for (int i = 0; i < data1.Length; i++)
            {
                float v1 = data1[i];
                int id1 = Math.Max(0, Math.Min(nbins - 1, (int)((v1 - min1) / Range1 * (nbins - 1) + 0.5)));

                float v2 = data2[i];
                int id2 = Math.Max(0, Math.Min(nbins - 1, (int)((v2 - min2) / Range2 * (nbins - 1) + 0.5)));

                HistogramBins[id2 * nbins + id1]++;
            }

            return HistogramBins;
        }

        public static float CrossCorrelate(float[] data1, float[] data2)
        {
            float Sum = 0;
            unsafe
            {
                fixed (float* Data1Ptr = data1)
                fixed (float* Data2Ptr = data2)
                {
                    float* Data1P = Data1Ptr;
                    float* Data2P = Data2Ptr;

                    for (int i = 0; i < data1.Length; i++)
                        Sum += *Data1P++ * *Data2P++;
                }
            }

            return Sum / data1.Length;
        }

        public static float CrossCorrelateNormalized(float[] data1, float[] data2)
        {
            return CrossCorrelate(Normalize(data1), Normalize(data2));
        }

        public static float Min(IEnumerable<float> data)
        {
            float Min = float.MaxValue;
            return data.Aggregate(Min, (start, i) => Math.Min(start, i));
        }

        public static float[] Min(float[] data, float val)
        {
            float[] Result = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
                Result[i] = Math.Min(data[i], val);

            return Result;
        }

        public static float[] Min(float[] data1, float[] data2)
        {
            if (data1.Length != data2.Length)
                throw new DimensionMismatchException();

            float[] Result = new float[data1.Length];
            for (int i = 0; i < data1.Length; i++)
                Result[i] = Math.Min(data1[i], data2[i]);

            return Result;
        }

        public static int Min(IEnumerable<int> data)
        {
            int Min = int.MaxValue;
            return data.Aggregate(Min, (start, i) => Math.Min(start, i));
        }

        public static float Max(IEnumerable<float> data)
        {
            float Max = -float.MaxValue;
            return data.Aggregate(Max, (start, i) => Math.Max(start, i));
        }

        public static float2 Max(IEnumerable<float2> data)
        {
            float2 Max = new float2(-float.MaxValue);
            foreach (var val in data)
            {
                Max.X = Math.Max(Max.X, val.X);
                Max.Y = Math.Max(Max.Y, val.Y);
            }

            return Max;
        }

        public static float3 Max(IEnumerable<float3> data)
        {
            float3 Max = new float3(-float.MaxValue);
            foreach (var val in data)
            {
                Max.X = Math.Max(Max.X, val.X);
                Max.Y = Math.Max(Max.Y, val.Y);
                Max.Z = Math.Max(Max.Z, val.Z);
            }

            return Max;
        }

        public static (int id, float value) MaxElement(float[] data)
        {
            int MaxID = 0;
            float Max = float.MinValue;

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] > Max)
                {
                    MaxID = i;
                    Max = data[i];
                }
            }

            return (MaxID, Max);
        }

        public static float[] Max(float[] data, float val)
        {
            float[] Result = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
                Result[i] = Math.Max(data[i], val);

            return Result;
        }

        public static float[] Max(float[] data1, float[] data2)
        {
            if (data1.Length != data2.Length)
                throw new DimensionMismatchException();

            float[] Result = new float[data1.Length];
            for (int i = 0; i < data1.Length; i++)
                Result[i] = Math.Max(data1[i], data2[i]);

            return Result;
        }

        public static double[] Max(double[] data1, double[] data2)
        {
            if (data1.Length != data2.Length)
                throw new DimensionMismatchException();

            double[] Result = new double[data1.Length];
            for (int i = 0; i < data1.Length; i++)
                Result[i] = Math.Max(data1[i], data2[i]);

            return Result;
        }

        public static int Max(IEnumerable<int> data)
        {
            int Max = -int.MaxValue;
            return data.Aggregate(Max, (start, i) => Math.Max(start, i));
        }

        public static float[] Plus(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] + data2[i];

            return Result;
        }

        public static float[] Minus(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] - data2[i];

            return Result;
        }

        public static float[] Mult(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] * data2[i];

            return Result;
        }

        public static float[] Div(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] / data2[i];

            return Result;
        }

        public static float[] Subtract(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] - data2[i];

            return Result;
        }

        public static float2[] Subtract(float2[] data1, float2[] data2)
        {
            float2[] Result = new float2[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] - data2[i];

            return Result;
        }

        public static float3[] Subtract(float3[] data1, float3[] data2)
        {
            float3[] Result = new float3[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] - data2[i];

            return Result;
        }

        public static float[] Add(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] + data2[i];

            return Result;
        }

        public static float2[] Add(float2[] data1, float2[] data2)
        {
            float2[] Result = new float2[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] + data2[i];

            return Result;
        }

        public static float3[] Add(float3[] data1, float3[] data2)
        {
            float3[] Result = new float3[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] + data2[i];

            return Result;
        }

        public static float[] Diff(float[] data)
        {
            float[] D = new float[data.Length - 1];
            for (int i = 0; i < data.Length - 1; i++)
                D[i] = data[i + 1] - data[i];

            return D;
        }

        public static double[] Diff(double[] data)
        {
            double[] D = new double[data.Length - 1];
            for (int i = 0; i < data.Length - 1; i++)
                D[i] = data[i + 1] - data[i];

            return D;
        }

        public static float2[] Diff(float2[] data)
        {
            float2[] D = new float2[data.Length - 1];
            for (int i = 0; i < data.Length - 1; i++)
                D[i] = data[i + 1] - data[i];

            return D;
        }

        public static bool AllEqual(double[] a1, double[] a2)
        {
            if (a1 == null || a2 == null || a1.Length != a2.Length)
                return false;

            for (int i = 0; i < a1.Length; i++)
            {
                if (a1[i] != a2[i])
                    return false;
            }

            return true;
        }

        public static bool AllEqual(float[] a1, float[] a2)
        {
            if (a1 == null || a2 == null || a1.Length != a2.Length)
                return false;

            for (int i = 0; i < a1.Length; i++)
            {
                if (a1[i] != a2[i])
                    return false;
            }

            return true;
        }

        public static float DotProduct(float[] data1, float[] data2)
        {
            double Sum = 0;
            for (int i = 0; i < data1.Length; i++)
                Sum += data1[i] * data2[i];

            return (float)Sum;
        }

        public static int NextMultipleOf(int value, int factor)
        {
            return ((value + factor - 1) / factor) * factor;
        }

        public static float ReduceWeighted(float[] data, float[] weights)
        {
            float Sum = 0f;
            float Weightsum = 0f;
            unsafe
            {
                fixed (float* dataPtr = data)
                fixed (float* weightsPtr = weights)
                {
                    float* dataP = dataPtr;
                    float* weightsP = weightsPtr;

                    for (int i = 0; i < data.Length; i++)
                    {
                        Sum += *dataP++ * *weightsP;
                        Weightsum += *weightsP++;
                    }
                }
            }

            return Sum;
        }

        public static float ReduceWeightedSparse(float[] data, (int index, float weight)[] weights)
        {
            float Sum = 0f;
            float Weightsum = 0f;
            unsafe
            {
                fixed (float* dataPtr = data)
                {
                    float* dataP = dataPtr;

                    for (int i = 0; i < weights.Length; i++)
                    {
                        int index = weights[i].index;
                        float weight = weights[i].weight;
                        Sum += dataPtr[index] * weight;
                        Weightsum += weight;
                    }
                }
            }

            return Sum;
        }

        public static float MeanWeighted(float[] data, float[] weights)
        {
            float Sum = 0f;
            float Weightsum = 0f;
            unsafe
            {
                fixed (float* dataPtr = data)
                fixed (float* weightsPtr = weights)
                {
                    float* dataP = dataPtr;
                    float* weightsP = weightsPtr;

                    for (int i = 0; i < data.Length; i++)
                    {
                        Sum += *dataP++ * *weightsP;
                        Weightsum += *weightsP++;
                    }
                }
            }

            if (Math.Abs(Weightsum) > 1e-6f)
                return Sum / Weightsum;
            else
                return 0;
        }

        public static float3 MeanWeighted(float3[] data, float[] weights)
        {
            float3 Sum = new float3();
            float Weightsum = 0f;
            unsafe
            {
                fixed (float3* dataPtr = data)
                fixed (float* weightsPtr = weights)
                {
                    float3* dataP = dataPtr;
                    float* weightsP = weightsPtr;

                    for (int i = 0; i < data.Length; i++)
                    {
                        Sum += *dataP++ * *weightsP;
                        Weightsum += *weightsP++;
                    }
                }
            }

            if (Math.Abs(Weightsum) > 1e-6f)
                return Sum / Weightsum;
            else
                return new float3();
        }

        public static float[] MeanWeighted(float[][] data, float[] weights)
        {
            float[] Mean = new float[data[0].Length];
            float WeightSum = weights.Sum();

            for (int i = 0; i < data.Length; i++)
                for (int j = 0; j < Mean.Length; j++)
                    Mean[j] += data[i][j];

            for (int i = 0; i < Mean.Length; i++)
                Mean[i] /= WeightSum;

            return Mean;
        }

        public static void UnNaN(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
                if (float.IsNaN(data[i]))
                    data[i] = 0;
        }

        public static void UnNaN(float2[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                if (float.IsNaN(data[i].X))
                    data[i].X = 0;
                if (float.IsNaN(data[i].Y))
                    data[i].Y = 0;
            }
        }

        public static float ResidualFraction(float value)
        {
            return value - (int)value;
        }

        public static T Percentile<T>(IEnumerable<T> data, decimal percentile)
        {
            List<T> Sorted = new List<T>(data);
            Sorted.Sort();

            return Sorted[(int)((Sorted.Count - 1) * (percentile / 100M))];
        }

        public static T Median<T>(IEnumerable<T> data)
        {
            return Percentile(data, 50);
        }

        public static float[] WithinNStd(float[] data, float nstd)
        {
            float Mean = MathHelper.Mean(data);
            float Std = StdDev(data) * nstd;

            return data.Where(t => Math.Abs(t - Mean) <= Std).ToArray();
        }

        public static float[] WithinNStdFromMedian(float[] data, float nstd)
        {
            float Mean = Median(data);
            float Std = StdDev(data) * nstd;

            List<float> Result = data.Where(t => Math.Abs(t - Mean) <= Std).ToList();

            return Result.ToArray();
        }

        public static int[] WithinNStdFromMedianIndices(float[] data, float nstd)
        {
            float Mean = Median(data);
            float Std = StdDev(data) * nstd;

            List<int> Result = new List<int>();

            for (int i = 0; i < data.Length; i++)
                if (Math.Abs(data[i] - Mean) <= Std)
                    Result.Add(i);

            return Result.ToArray();
        }

        public static float[] TakeNLowest(float[] data, int N, out int[] indices)
        {
            N = Math.Min(N, data.Length);
            float[] Result = new float[N];
            List<int> ResultIndices = new List<int>();

            for (int n = 0; n < N; n++)
            {
                float LowestVal = float.MaxValue;
                int LowestIndex = 0;

                for (int i = 0; i < data.Length; i++)
                    if (data[i] < LowestVal && !ResultIndices.Contains(i))
                    {
                        LowestVal = data[i];
                        LowestIndex = i;
                    }

                Result[n] = LowestVal;
                ResultIndices.Add(LowestIndex);
            }

            indices = ResultIndices.ToArray();
            return Result;
        }

        public static double[] TakeNLowest(double[] data, int N, out int[] indices)
        {
            N = Math.Min(N, data.Length);
            double[] Result = new double[N];
            List<int> ResultIndices = new List<int>();

            for (int n = 0; n < N; n++)
            {
                double LowestVal = double.MaxValue;
                int LowestIndex = 0;

                for (int i = 0; i < data.Length; i++)
                    if (data[i] < LowestVal && !ResultIndices.Contains(i))
                    {
                        LowestVal = data[i];
                        LowestIndex = i;
                    }

                Result[n] = LowestVal;
                ResultIndices.Add(LowestIndex);
            }

            indices = ResultIndices.ToArray();
            return Result;
        }

        public static float[] TakeNHighest(float[] data, int N, out int[] indices)
        {
            N = Math.Min(N, data.Length);
            float[] Result = new float[N];
            List<int> ResultIndices = new List<int>();

            for (int n = 0; n < N; n++)
            {
                float HighestVal = -float.MaxValue;
                int HighestIndex = 0;

                for (int i = 0; i < data.Length; i++)
                    if (data[i] > HighestVal && !ResultIndices.Contains(i))
                    {
                        HighestVal = data[i];
                        HighestIndex = i;
                    }

                Result[n] = HighestVal;
                ResultIndices.Add(HighestIndex);
            }

            indices = ResultIndices.ToArray();
            return Result;
        }

        public static double[] TakeNHighest(double[] data, int N, out int[] indices)
        {
            N = Math.Min(N, data.Length);
            double[] Result = new double[N];
            List<int> ResultIndices = new List<int>();

            for (int n = 0; n < N; n++)
            {
                double HighestVal = -double.MaxValue;
                int HighestIndex = 0;

                for (int i = 0; i < data.Length; i++)
                    if (data[i] > HighestVal && !ResultIndices.Contains(i))
                    {
                        HighestVal = data[i];
                        HighestIndex = i;
                    }

                Result[n] = HighestVal;
                ResultIndices.Add(HighestIndex);
            }

            indices = ResultIndices.ToArray();
            return Result;
        }

        public static float[] TakeAllBelow(float[] data, float threshold, out int[] indices)
        {
            List<float> Result = new List<float>();
            List<int> ResultIndices = new List<int>();

            for (int i = 0; i < data.Length; i++)
                if (data[i] < threshold)
                {
                    Result.Add(data[i]);
                    ResultIndices.Add(i);
                }

            indices = ResultIndices.ToArray();
            return Result.ToArray();
        }

        public static float[] TakeAllAbove(float[] data, float threshold, out int[] indices)
        {
            List<float> Result = new List<float>();
            List<int> ResultIndices = new List<int>();

            for (int i = 0; i < data.Length; i++)
                if (data[i] > threshold)
                {
                    Result.Add(data[i]);
                    ResultIndices.Add(i);
                }

            indices = ResultIndices.ToArray();
            return Result.ToArray();
        }

        public static float Lerp(float a, float b, float x)
        {
            return a + (b - a) * x;
        }

        public static float3 FitPlane(float3[] points)
        {
            double D = 0;
            double E = 0;
            double F = 0;
            double G = 0;
            double H = 0;
            double I = 0;
            double J = 0;
            double K = 0;
            double L = 0;
            double W2 = 0;
            double error = 0;
            double denom = 0;

            for (int i = 0; i < points.Length; i++)
            {
                D += points[i].X * points[i].X;
                E += points[i].X * points[i].Y;
                F += points[i].X;
                G += points[i].Y * points[i].Y;
                H += points[i].Y;
                I += 1;
                J += points[i].X * points[i].Z;
                K += points[i].Y * points[i].Z;
                L += points[i].Z;
            }

            denom = F * F * G - 2 * E * F * H + D * H * H + E * E * I - D * G * I;

            // X axis slope
            double plane_a = (H * H * J - G * I * J + E * I * K + F * G * L - H * (F * K + E * L)) / denom;
            // Y axis slope
            double plane_b = (E * I * J + F * F * K - D * I * K + D * H * L - F * (H * J + E * L)) / denom;
            // Z axis intercept
            double plane_c = (F * G * J - E * H * J - E * F * K + D * H * K + E * E * L - D * G * L) / denom;

            return new float3((float)plane_a, (float)plane_b, (float)plane_c);
        }

        public static float3 FitPlaneWeighted(float4[] points)
        {
            double D = 0;
            double E = 0;
            double F = 0;
            double G = 0;
            double H = 0;
            double I = 0;
            double J = 0;
            double K = 0;
            double L = 0;
            double W2 = 0;
            double error = 0;
            double denom = 0;

            for (int i = 0; i < points.Length; i++)
            {
                W2 = points[i].W * points[i].W;
                D += points[i].X * points[i].X * W2;
                E += points[i].X * points[i].Y * W2;
                F += points[i].X * W2;
                G += points[i].Y * points[i].Y * W2;
                H += points[i].Y * W2;
                I += 1 * W2;
                J += points[i].X * points[i].Z * W2;
                K += points[i].Y * points[i].Z * W2;
                L += points[i].Z * W2;
            }

            denom = F * F * G - 2 * E * F * H + D * H * H + E * E * I - D * G * I;

            // X axis slope
            double plane_a = (H * H * J - G * I * J + E * I * K + F * G * L - H * (F * K + E * L)) / denom;
            // Y axis slope
            double plane_b = (E * I * J + F * F * K - D * I * K + D * H * L - F * (H * J + E * L)) / denom;
            // Z axis intercept
            double plane_c = (F * G * J - E * H * J - E * F * K + D * H * K + E * E * L - D * G * L) / denom;

            return new float3((float)plane_a, (float)plane_b, (float)plane_c);
        }

        public static float3 FitPlane(float[] intensities, int2 dims)
        {
            double D = 0;
            double E = 0;
            double F = 0;
            double G = 0;
            double H = 0;
            double I = 0;
            double J = 0;
            double K = 0;
            double L = 0;
            double denom = 0;

            for (int y = 0; y < dims.Y; y++)
            {
                for (int x = 0; x < dims.X; x++)
                {
                    float3 Point = new float3(x, y, intensities[y * dims.X + x]);
                    D += Point.X * Point.X;
                    E += Point.X * Point.Y;
                    F += Point.X;
                    G += Point.Y * Point.Y;
                    H += Point.Y;
                    I += 1;
                    J += Point.X * Point.Z;
                    K += Point.Y * Point.Z;
                    L += Point.Z;
                }
            }

            denom = F * F * G - 2 * E * F * H + D * H * H + E * E * I - D * G * I;

            // X axis slope
            double plane_a = (H * H * J - G * I * J + E * I * K + F * G * L - H * (F * K + E * L)) / denom;
            // Y axis slope
            double plane_b = (E * I * J + F * F * K - D * I * K + D * H * L - F * (H * J + E * L)) / denom;
            // Z axis intercept
            double plane_c = (F * G * J - E * H * J - E * F * K + D * H * K + E * E * L - D * G * L) / denom;

            return new float3((float)plane_a, (float)plane_b, (float)plane_c);
        }

        public static float[] FitAndGeneratePlane(float[] intensities, int2 dims)
        {
            float3 Plane = FitPlane(intensities, dims);

            float[] Result = new float[dims.Elements()];

            for (int y = 0; y < dims.Y; y++)
            {
                for (int x = 0; x < dims.X; x++)
                {
                    Result[y * dims.X + x] = x * Plane.X + y * Plane.Y + Plane.Z;
                }
            }

            return Result;
        }

        public static void FitAndSubtractPlane(float[] intensities, int2 dims)
        {
            float3 Plane = FitPlane(intensities, dims);

            for (int y = 0; y < dims.Y; y++)
            {
                for (int x = 0; x < dims.X; x++)
                {
                    intensities[y * dims.X + x] -= x * Plane.X + y * Plane.Y + Plane.Z;
                }
            }
        }

        public static void FitAndSubtractGrid(float[] intensities, int2 dims, int2 gridDims)
        {
            float2 GridSpacing = new float2(dims) / new float2(gridDims + 1);
            float2[] GridCentroids = Helper.Combine(Helper.ArrayOfFunction(y => 
                                                    Helper.ArrayOfFunction(x => new float2((x + 1) * GridSpacing.X, 
                                                                                           (y + 1) * GridSpacing.Y), 
                                                    gridDims.X), gridDims.Y));
            float3[] Planes = new float3[gridDims.Elements()];

            Parallel.For(0, gridDims.Elements(), ci =>
            {
                float2 Centroid = GridCentroids[ci];

                int2 PositionStart = int2.Max(new int2(Centroid - GridSpacing), 0);
                int2 PositionEnd = int2.Min(new int2(Centroid + GridSpacing + 1), dims);
                int2 PatchDims = PositionEnd - PositionStart;
                float3[] Points = new float3[PatchDims.Elements()];

                for (int y = 0; y < PatchDims.Y; y++)
                    for (int x = 0; x < PatchDims.X; x++)
                        Points[y * PatchDims.X + x] = new float3(x - GridSpacing.X,
                                                                    y - GridSpacing.Y,
                                                                    intensities[(y + PositionStart.Y) * dims.X + x + PositionStart.X]);

                Planes[ci] = FitPlane(Points);
            });

            float[] FittedGrid = new float[dims.Elements()];

            Parallel.For(0, dims.Y, y =>
            {
                for (int x = 0; x < dims.X; x++)
                {
                    int2 Centroid0 = int2.Min(int2.Max(new int2((new float2(x, y) - GridSpacing) / GridSpacing), 0), gridDims - 1);
                    int CentroidID00 = gridDims.ElementFromPosition(Centroid0);
                    int CentroidID01 = gridDims.ElementFromPosition(int2.Min(Centroid0 + new int2(1, 0), gridDims - 1));
                    int CentroidID10 = gridDims.ElementFromPosition(int2.Min(Centroid0 + new int2(0, 1), gridDims - 1));
                    int CentroidID11 = gridDims.ElementFromPosition(int2.Min(Centroid0 + new int2(1, 1), gridDims - 1));

                    float2 Centroid00 = GridCentroids[CentroidID00];
                    float2 Centroid01 = GridCentroids[CentroidID01];
                    float2 Centroid10 = GridCentroids[CentroidID10];
                    float2 Centroid11 = GridCentroids[CentroidID11];

                    float Interp00 = (x - Centroid00.X) * Planes[CentroidID00].X + (y - Centroid00.Y) * Planes[CentroidID00].Y + Planes[CentroidID00].Z;
                    float Interp01 = (x - Centroid01.X) * Planes[CentroidID01].X + (y - Centroid01.Y) * Planes[CentroidID01].Y + Planes[CentroidID01].Z;
                    float Interp10 = (x - Centroid10.X) * Planes[CentroidID10].X + (y - Centroid10.Y) * Planes[CentroidID10].Y + Planes[CentroidID10].Z;
                    float Interp11 = (x - Centroid11.X) * Planes[CentroidID11].X + (y - Centroid11.Y) * Planes[CentroidID11].Y + Planes[CentroidID11].Z;

                    float fX = Math.Max(0, Math.Min(1, (x - Centroid00.X) / GridSpacing.X));
                    float fY = Math.Max(0, Math.Min(1, (y - Centroid00.Y) / GridSpacing.Y));

                    float Interp0 = Interp00 * (1 - fX) + Interp01 * fX;
                    float Interp1 = Interp10 * (1 - fX) + Interp11 * fX;

                    float Interp = Interp0 * (1 - fY) + Interp1 * fY;

                    FittedGrid[y * dims.X + x] = Interp;
                }
            });

            for (int i = 0; i < intensities.Length; i++)
                intensities[i] -= FittedGrid[i];
        }

        public static string GetSHA1(byte[] data)
        {
            using (SHA1 hasher = new SHA1CryptoServiceProvider())
            {
                byte[] HashBytes = hasher.ComputeHash(data);

                return Convert.ToBase64String(HashBytes).Replace("=", "").Replace("/", "-").Replace("+", "_");
            }
        }

        public static string GetSHA1(string path, long maxBytes)
        {
            using (BinaryReader reader = new BinaryReader(File.OpenRead(path)))
            {
                maxBytes = Math.Min(maxBytes, reader.BaseStream.Length);
                byte[] Bytes = reader.ReadBytes((int)maxBytes);

                return GetSHA1(Bytes);
            }
        }

        public static float3 FitLineWeighted(float3[] points)
        {
            float ss_xy = 0;
            float ss_xx = 0;
            float ss_yy = 0;
            float ave_x = 0;
            float ave_y = 0;
            float sum_w = 0;
            for (int i = 0; i < points.Length; i++)
            {
                ave_x += points[i].Z * points[i].X;
                ave_y += points[i].Z * points[i].Y;
                sum_w += points[i].Z;
                ss_xx += points[i].Z * points[i].X * points[i].X;
                ss_yy += points[i].Z * points[i].Y * points[i].Y;
                ss_xy += points[i].Z * points[i].X * points[i].Y;
            }
            ave_x /= sum_w;
            ave_y /= sum_w;
            ss_xx -= sum_w * ave_x * ave_x;
            ss_yy -= sum_w * ave_y * ave_y;
            ss_xy -= sum_w * ave_x * ave_y;

            float Slope = 0;
            float Intercept = 0;
            float Quality = 0;

            if (ss_xx > 0)
            {
                Slope = ss_xy / ss_xx;
                Intercept = ave_y - Slope * ave_x;
                Quality = ss_xy * ss_xy / (ss_xx * ss_yy);
            }

            return new float3(Slope, Intercept, Quality);
        }

        public static float FitScaleLeastSq(float[] source, float[] target)
        {
            if (source.All(v => v == 0))
                return 0;

            double Diff09 = 0;
            double Diff10 = 0;
            double Diff11 = 0;

            for (int i = 0; i < source.Length; i++)
            {
                double S = source[i];
                double T = target[i];

                double Diff = (S * 0.99) - T;
                Diff09 += Diff * Diff;

                Diff = S - T;
                Diff10 += Diff * Diff;

                Diff = (S * 1.01) - T;
                Diff11 += Diff * Diff;
            }

            double x1 = 0.99, x2 = 1.0, x3 = 1.01;
            double y1 = Diff09 / source.Length,
                   y2 = Diff10 / source.Length,
                   y3 = Diff11 / source.Length;

            double Denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
            double A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / Denom;
            double B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / Denom;
            double C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / Denom;

            double xv = -B / (2 * A);

            return (float)xv;
        }

        public static float Gauss(float x, float mu, float sigma)
        {
            sigma = -1f / (sigma * sigma * 2);
            x -= mu;

            return MathF.Exp(x * x * sigma);
        }

        public static float[] GetGaussianKernel1D(int extent, float sigma, bool normalized)
        {
            float[] Kernel = new float[extent * 2 + 1];

            sigma = -1f / (sigma * sigma * 2);
            float GaussianSum = 0;

            for (int i = 0; i < Kernel.Length; i++)
            {
                int ii = i - extent;
                ii *= ii;

                float G = MathF.Exp(ii * sigma);

                GaussianSum += G;
                Kernel[i] = G;
            }

            if (normalized)
                for (int i = 0; i < Kernel.Length; i++)
                    Kernel[i] /= GaussianSum;

            return Kernel;
        }

        public static float[] ConvolveWithKernel1D(float[] data, float[] kernel)
        {
            float[] Convolved = new float[data.Length];
            int Extent = kernel.Length / 2;

            for (int i1 = 0; i1 < data.Length; i1++)
            {
                float Sum = 0;
                float Weights = 0;

                for (int ik = 0; ik < kernel.Length; ik++)
                {
                    int i2 = i1 + ik - Extent;
                    if (i2 < 0 || i2 >= data.Length)
                        continue;

                    Sum += data[i2] * kernel[ik];
                    Weights += kernel[ik];
                }

                if (Weights != 0)
                    Convolved[i1] = Sum / Weights;
            }

            return Convolved;
        }

        public static float CircleFractionInsideRectangle(float2 center, float radius, float2 topleft, float2 bottomright, int steps = 100)
        {
            int Inside = 0;
            float AngleStep = 360f / steps * Helper.ToRad;

            for (int i = 0; i < steps; i++)
            {
                float2 Point = center + new float2(MathF.Cos(i * AngleStep), MathF.Sin(i * AngleStep)) * radius;
                if (Point > topleft && Point < bottomright)
                    Inside++;
            }

            return (float)Inside / steps;
        }

        public static float2? FindIntersection(float2 origin1, float2 direction1, float2 origin2, float2 direction2)
        {
            float x1 = origin1.X, y1 = origin1.Y;
            float x2 = x1 + direction1.X, y2 = y1 + direction1.Y;
            float x3 = origin2.X, y3 = origin2.Y;
            float x4 = x3 + direction2.X, y4 = y3 + direction2.Y;

            float den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
            if (den == 0) return null; // Parallel rays

            float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
            float u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den;

            // Check if intersection is within the ray segments
            if (t >= 0 && u >= 0)
            {
                float2 intersection = new float2(x1 + t * (x2 - x1), y1 + t * (y2 - y1));
                return intersection;
            }

            return null; // Intersection is outside of the rays
        }

        public static int DrawBinomial(int numberOfTrials, float successProbability, Random rng)
        {
            int Successes = 0;

            for (int i = 0; i < numberOfTrials; i++)
                if (rng.NextSingle() < successProbability)
                    Successes++;

            return Successes;
        }
    }

    public class KaiserTable
    {
        public float[] Values;
        public float Sampling;
        public float Radius;

        public KaiserTable(int samples, float radius, float alpha, int order)
        {
            Sampling = radius / samples;
            Values = new float[samples];
            Radius = radius;

            for (int i = 0; i < samples; i++)
                Values[i] = CPU.KaiserBessel(i * Sampling, radius, alpha, order);
        }

        public float GetValue(float r)
        {
            int Sample = (int)(r / Sampling);
            if (Sample >= Values.Length)
                return 0;

            return Values[Sample];
        }
    }

    public class KaiserFTTable
    {
        public float[] Values;
        public float Sampling;
        public float Radius;

        public KaiserFTTable(int samples, float radius, float alpha, int order)
        {
            Sampling = 0.5f / samples;
            Values = new float[samples];
            Radius = radius;

            for (int i = 0; i < samples; i++)
                Values[i] = CPU.KaiserBessel(i * Sampling, radius, alpha, order);
        }

        public float GetValue(float r)
        {
            int Sample = (int)(r / Sampling);
            if (Sample >= Values.Length)
                return 0;

            return Values[Sample];
        }
    }
}
