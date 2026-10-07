using System;
using System.Collections.Generic;
using System.Linq;
using Warp.Tools;
using ZLinq;

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

    }
}
