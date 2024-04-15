using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using Accord.Math.Optimization;
using Warp.Tools;

namespace Warp
{
    /// <summary>
    /// Implements Piecewise Cubic Hermite Interpolation exactly as in Matlab.
    /// </summary>
    public class Cubic1D
    {
        public readonly float2[] Data;
        readonly float[] Breaks;
        readonly float4[] Coefficients;

        public Cubic1D(float2[] data)
        {
            // Sort points to go strictly from left to right.
            List<float2> DataList = data.ToList();
            DataList.Sort((p1, p2) => p1.X.CompareTo(p2.X));
            data = DataList.ToArray();

            Data = data;
            Breaks = data.Select(i => i.X).ToArray();
            Coefficients = new float4[data.Length - 1];

            float[] h = MathHelper.Diff(data.Select(i => i.X).ToArray());
            float[] del = MathHelper.Div(MathHelper.Diff(data.Select(i => i.Y).ToArray()), h);
            float[] slopes = GetPCHIPSlopes(data, del);

            float[] dzzdx = new float[del.Length];
            for (int i = 0; i < dzzdx.Length; i++)
                dzzdx[i] = (del[i] - slopes[i]) / h[i];

            float[] dzdxdx = new float[del.Length];
            for (int i = 0; i < dzdxdx.Length; i++)
                dzdxdx[i] = (slopes[i + 1] - del[i]) / h[i];

            for (int i = 0; i < Coefficients.Length; i++)
                Coefficients[i] = new float4((dzdxdx[i] - dzzdx[i]) / h[i],
                                             2f * dzzdx[i] - dzdxdx[i],
                                             slopes[i],
                                             data[i].Y);
        }

        public float[] Interp(float[] x)
        {
            if (Data.Length == 1)
                return Helper.ArrayOfConstant(Data[0].Y, x.Length);

            float[] y = new float[x.Length];

            float[] b = Breaks;
            float4[] c = Coefficients;

            int[] indices = new int[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] < b[1])
                    indices[i] = 0;
                else if (x[i] >= b[b.Length - 2])
                    indices[i] = b.Length - 2;
                else
                    for (int j = 2; j < b.Length - 1; j++)
                        if (x[i] < b[j])
                        {
                            indices[i] = j - 1;
                            break;
                        }
            }

            float[] xs = new float[x.Length];
            for (int i = 0; i < xs.Length; i++)
                xs[i] = x[i] - b[indices[i]];

            for (int i = 0; i < x.Length; i++)
            {
                int index = indices[i];
                float v = c[index].X;
                v = xs[i] * v + c[index].Y;
                v = xs[i] * v + c[index].Z;
                v = xs[i] * v + c[index].W;

                y[i] = v;
            }

            return y;
        }

        public float Interp(float x)
        {
            if (Data.Length == 1)
                return Data[0].Y;

            float[] b = Breaks;
            float4[] c = Coefficients;

            int index = 0;

            if (x < b[1])
                index = 0;
            else if (x >= b[b.Length - 2])
                index = b.Length - 2;
            else
                for (int j = 2; j < b.Length - 1; j++)
                    if (x < b[j])
                    {
                        index = j - 1;
                        break;
                    }

            float xs = x - b[index];
            
            float v = c[index].X;
            v = xs * v + c[index].Y;
            v = xs * v + c[index].Z;
            v = xs * v + c[index].W;

            float y = v;

            return y;
        }

        private static float[] GetPCHIPSlopes(float2[] data, float[] del)
        {
            if (data.Length == 1)
                return new[] { 0f, 0f };

            if (data.Length == 2)
                return new[] { del[0], del[0] };   // Do only linear

            float[] d = new float[data.Length];
            float[] h = MathHelper.Diff(data.Select(i => i.X).ToArray());
            for (int k = 0; k < del.Length - 1; k++)
            {
                if (del[k] * del[k + 1] <= 0f)
                    continue;

                float hs = h[k] + h[k + 1];
                float w1 = (h[k] + hs) / (3f * hs);
                float w2 = (hs + h[k + 1]) / (3f * hs);
                float dmax = Math.Max(Math.Abs(del[k]), Math.Abs(del[k + 1]));
                float dmin = Math.Min(Math.Abs(del[k]), Math.Abs(del[k + 1]));
                d[k + 1] = dmin / (w1 * (del[k] / dmax) + w2 * (del[k + 1] / dmax));
            }

            d[0] = ((2f * h[0] + h[1]) * del[0] - h[0] * del[1]) / (h[0] + h[1]);
            if (Math.Sign(d[0]) != Math.Sign(del[0]))
                d[0] = 0;
            else if (Math.Sign(del[0]) != Math.Sign(del[1]) && Math.Abs(d[0]) > Math.Abs(3f * del[0]))
                d[0] = 3f * del[0];

            int n = d.Length - 1;
            d[n] = ((2 * h[n - 1] + h[n - 2]) * del[n - 1] - h[n - 1] * del[n - 2]) / (h[n - 1] + h[n - 2]);
            if (Math.Sign(d[n]) != Math.Sign(del[n - 1]))
                d[n] = 0;
            else if (Math.Sign(del[n - 1]) != Math.Sign(del[n - 2]) && Math.Abs(d[n]) > Math.Abs(3f * del[n - 1]))
                d[n] = 3f * del[n - 1];

            return d;
        }

        public static Cubic1D Fit(float2[] data, int numnodes)
        {
            List<float2> Fitted = new List<float2>();

            int Extent = (int)Math.Max((float)data.Length / numnodes, 1);
            for (int n = 0; n < numnodes; n++)
            {
                float Sum = 0;
                int Samples = 0;
                int Start = Math.Min(data.Length - 1, n * (data.Length / numnodes));
                int Finish = Math.Min(data.Length, Start + Extent);

                for (int i = Start; i < Finish; i++, Samples++)
                    Sum += data[i].Y;

                Fitted.Add(new float2(data[Start + Extent / 2].X, Sum / Samples));
            }

            return new Cubic1D(Fitted.ToArray());
        }

        public static void FitCTF(float2[] data, float[] simulation, float[] zeros, float[] peaks, out Cubic1D background, out Cubic1D scale)
        {
            if (zeros.Length < 2 || peaks.Length < 1)
            {
                background = new Cubic1D(new[] { new float2(0, 0), new float2(1, 0) });
                scale = new Cubic1D(new[] { new float2(0, 1), new float2(1, 1) });

                return;
            }

            float MinX = MathHelper.Min(data.Select(p => p.X)), MaxX = MathHelper.Max(data.Select(p => p.X)), ScaleX = 1f / (MaxX - MinX);

            peaks = peaks.Where(v => v >= MinX && v <= MaxX).Where((v, i) => i % 1 == 0).ToArray();
            zeros = zeros.Where(v => v >= MinX && v <= MaxX).Where((v, i) => i % 1 == 0).ToArray();

            List<float2> Peaks = new List<float2>(), Zeros = new List<float2>();
            foreach (var zero in zeros)
            {
                int Pos = (int)((zero - MinX) * ScaleX * data.Length);
                int First = Math.Max(0, Pos - 1);
                int Last = Math.Min(data.Length - 1, Pos + 1);
                float MinVal = data[First].Y;
                for (int i = First; i < Last; i++)
                    MinVal = Math.Min(MinVal, data[i].Y);
                Zeros.Add(new float2(zero, MinVal));
            }
            float[] Background = (new Cubic1D(Zeros.ToArray())).Interp(data.Select(v => v.X).ToArray());
            float2[] DataSubtracted = Helper.ArrayOfFunction(i => new float2(data[i].X, data[i].Y - Background[i]), Background.Length);

            for (int z = 0; z < Zeros.Count; z++)
            {
                float2 Zero = Zeros[z];

                int Pos = (int)((Zero.X - MinX) * ScaleX * DataSubtracted.Length);
                int First = Math.Max(0, Pos - 1);
                int Last = Math.Min(DataSubtracted.Length, Pos + 1);
                float MinVal = DataSubtracted[First].Y;
                for (int i = First; i < Last; i++)
                    MinVal = Math.Min(MinVal, DataSubtracted[i].Y);

                Zeros[z] = new float2(Zero.X, Zero.Y + MinVal);
            }
            Background = (new Cubic1D(Zeros.ToArray())).Interp(data.Select(v => v.X).ToArray());
            DataSubtracted = Helper.ArrayOfFunction(i => new float2(data[i].X, data[i].Y - Background[i]), Background.Length);

            float GlobalMax = 0;
            foreach (var peak in peaks)
            {
                int Pos = (int)((peak - MinX) * ScaleX * DataSubtracted.Length);
                int First = Math.Max(0, Pos - 1);
                int Last = Math.Min(DataSubtracted.Length, Pos + 1);
                float MaxVal = GlobalMax * 0.05f;// DataSubtracted[First].Y;
                for (int i = First; i < Last; i++)
                    MaxVal = Math.Max(MaxVal, DataSubtracted[i].Y);
                Peaks.Add(new float2(peak, MaxVal));
                GlobalMax = Math.Max(MaxVal, GlobalMax);
            }

            background = Zeros.Count > 1 ? new Cubic1D(Zeros.ToArray()) : new Cubic1D(new[] { new float2(0, 0), new float2(1, 0) });
            scale = Peaks.Count > 1 ? new Cubic1D(Peaks.ToArray()) : new Cubic1D(new[] { new float2(0, 1), new float2(1, 1) });

            return;

            int EveryNth = 1;
            float[] ZerosX = new float[Zeros.Count / EveryNth];
            for (int i = 0; i < ZerosX.Length; i++)
                ZerosX[i] = Zeros[i * EveryNth].X;

            float[] PeaksX = new float[Peaks.Count / EveryNth];
            for (int i = 0; i < PeaksX.Length; i++)
                PeaksX[i] = Peaks[i * EveryNth].X;

            float[] DataX = data.Select(v => v.X).ToArray();
            float[] DataY = data.Select(v => v.Y).ToArray();

            Func<double[], double> Eval = (input) =>
            {
                Cubic1D SplineBackground = new Cubic1D(Helper.ArrayOfFunction(i => new float2(ZerosX[i], (float)Math.Exp(input[i])), ZerosX.Length));
                Cubic1D SplineScale = new Cubic1D(Helper.ArrayOfFunction(i => new float2(PeaksX[i], (float)Math.Exp(input[i + ZerosX.Length])), PeaksX.Length));

                float[] ContinuumBackground = SplineBackground.Interp(DataX);
                float[] ContinuumScale = SplineScale.Interp(DataX);

                float[] Diff = Helper.ArrayOfFunction(i => ContinuumBackground[i] + ContinuumScale[i] * simulation[i] - DataY[i], ContinuumBackground.Length);
                float DiffSq = 0;
                for (int i = 0; i < Diff.Length; i++)
                    DiffSq += Diff[i] * Diff[i];

                return DiffSq;
            };

            Func<double[], double[]> Grad = (input) =>
            {
                double CurrentValue = Eval(input);
                double[] Result = new double[input.Length];

                Parallel.For(0, input.Length, i =>
                {
                    double Delta = 1e-5;
                    double[] InputPlus = input.ToArray();
                    InputPlus[i] += Delta;

                    Result[i] = (Eval(InputPlus) - CurrentValue) / Delta;
                });

                return Result;
            };

            double[] ResampledBackground = background.Interp(ZerosX).Select(v => Math.Log(Math.Max(v, 1e-5))).ToArray();
            double[] ResampledScale = scale.Interp(PeaksX).Select(v => Math.Log(Math.Max(v, 1e-5))).ToArray();
            double[] StartParams = Helper.Combine(ResampledBackground, ResampledScale);

            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
            Optimizer.MaxIterations = 10;

            Optimizer.Minimize(StartParams);

            background = new Cubic1D(Helper.ArrayOfFunction(i => new float2(ZerosX[i], (float)Math.Exp(StartParams[i])), ZerosX.Length));
            scale = new Cubic1D(Helper.ArrayOfFunction(i => new float2(PeaksX[i], (float)Math.Exp(StartParams[i + ZerosX.Length])), PeaksX.Length));
        }
    }
}
