using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using Warp.Tools;

namespace Warp
{
    public abstract class Cubic1DShort
    {
        public static Cubic1DShort GetInterpolator(float2[] data)
        {
            if (data.Length == 4)
                return new Cubic1D4(data);
            if (data.Length == 3)
                return new Cubic1D3(data);

            return new Cubic1D2(data);
        }

        public abstract float Interp(float x);
    }

    public class Cubic1D4 : Cubic1DShort
    {
        public readonly float2[] Data;
        readonly float[] Breaks;
        readonly float4[] Coefficients;

        public Cubic1D4(float2[] data)
        {
            Data = data;
            Breaks = new[] { data[0].X, data[1].X, data[2].X, data[3].X };

            Coefficients = new float4[3];

            float[] h = { data[1].X - data[0].X, data[2].X - data[1].X, data[3].X - data[2].X };
            float[] del = { (data[1].Y - data[0].Y) / h[0], (data[2].Y - data[1].Y) / h[1], (data[3].Y - data[2].Y) / h[2] };
            float[] slopes = GetPCHIPSlopes(data, del);

            float[] dzzdx = new float[3];
            for (int i = 0; i < 3; i++)
                dzzdx[i] = (del[i] - slopes[i]) / h[i];

            float[] dzdxdx = new float[3];
            for (int i = 0; i < 3; i++)
                dzdxdx[i] = (slopes[i + 1] - del[i]) / h[i];

            for (int i = 0; i < 3; i++)
                Coefficients[i] = new float4((dzdxdx[i] - dzzdx[i]) / h[i],
                                             2f * dzzdx[i] - dzdxdx[i],
                                             slopes[i],
                                             data[i].Y);
        }

        public override float Interp(float x)
        {
            float[] b = Breaks;
            float4[] c = Coefficients;

            int index = 0;

            if (x < b[1])
                index = 0;
            else if (x >= b[2])
                index = 2;
            else
                for (int j = 2; j < 3; j++)
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
            float[] d = new float[4];
            float[] h = { data[1].X - data[0].X, data[2].X - data[1].X, data[3].X - data[2].X };
            for (int k = 0; k < 2; k++)
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

            int n = 4 - 1;
            d[n] = ((2 * h[n - 1] + h[n - 2]) * del[n - 1] - h[n - 1] * del[n - 2]) / (h[n - 1] + h[n - 2]);
            if (Math.Sign(d[n]) != Math.Sign(del[n - 1]))
                d[n] = 0;
            else if (Math.Sign(del[n - 1]) != Math.Sign(del[n - 2]) && Math.Abs(d[n]) > Math.Abs(3f * del[n - 1]))
                d[n] = 3f * del[n - 1];

            return d;
        }
    }

    public class Cubic1D3 : Cubic1DShort
    {
        public readonly float2[] Data;
        readonly float[] Breaks;
        readonly float4[] Coefficients;

        public Cubic1D3(float2[] data)
        {
            Data = data;
            Breaks = new[] { data[0].X, data[1].X, data[2].X };

            Coefficients = new float4[2];

            float[] h = { data[1].X - data[0].X, data[2].X - data[1].X };
            float[] del = { (data[1].Y - data[0].Y) / h[0], (data[2].Y - data[1].Y) / h[1] };
            float[] slopes = GetPCHIPSlopes(data, del);

            float[] dzzdx = new float[2];
            for (int i = 0; i < 2; i++)
                dzzdx[i] = (del[i] - slopes[i]) / h[i];

            float[] dzdxdx = new float[2];
            for (int i = 0; i < 2; i++)
                dzdxdx[i] = (slopes[i + 1] - del[i]) / h[i];

            for (int i = 0; i < 2; i++)
                Coefficients[i] = new float4((dzdxdx[i] - dzzdx[i]) / h[i],
                                             2f * dzzdx[i] - dzdxdx[i],
                                             slopes[i],
                                             data[i].Y);
        }

        public override float Interp(float x)
        {
            float[] b = Breaks;
            float4[] c = Coefficients;

            int index = 0;

            if (x < b[1])
                index = 0;
            else if (x >= b[1])
                index = 1;

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
            float[] d = new float[3];
            float[] h = { data[1].X - data[0].X, data[2].X - data[1].X };
            for (int k = 0; k < 1; k++)
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

            int n = 3 - 1;
            d[n] = ((2 * h[n - 1] + h[n - 2]) * del[n - 1] - h[n - 1] * del[n - 2]) / (h[n - 1] + h[n - 2]);
            if (Math.Sign(d[n]) != Math.Sign(del[n - 1]))
                d[n] = 0;
            else if (Math.Sign(del[n - 1]) != Math.Sign(del[n - 2]) && Math.Abs(d[n]) > Math.Abs(3f * del[n - 1]))
                d[n] = 3f * del[n - 1];

            return d;
        }
    }

    public class Cubic1D2 : Cubic1DShort
    {
        public readonly float2[] Data;
        readonly float[] Breaks;
        readonly float4[] Coefficients;

        public Cubic1D2(float2[] data)
        {
            Data = data;
            Breaks = new[] { data[0].X, data[1].X };

            Coefficients = new float4[1];

            float h = data[1].X - data[0].X;
            float del = (data[1].Y - data[0].Y) / h;
            float[] slopes = { del, del };

            float[] dzzdx = new float[1];
            for (int i = 0; i < 1; i++)
                dzzdx[i] = (del - slopes[i]) / h;

            float[] dzdxdx = new float[1];
            for (int i = 0; i < 1; i++)
                dzdxdx[i] = (slopes[i + 1] - del) / h;

            for (int i = 0; i < 1; i++)
                Coefficients[i] = new float4((dzdxdx[i] - dzzdx[i]) / h,
                                             2f * dzzdx[i] - dzdxdx[i],
                                             slopes[i],
                                             data[i].Y);
        }

        public override float Interp(float x)
        {
            float[] b = Breaks;
            float4[] c = Coefficients;

            int index = 0;

            float xs = x - b[index];

            float v = c[index].X;
            v = xs * v + c[index].Y;
            v = xs * v + c[index].Z;
            v = xs * v + c[index].W;

            float y = v;

            return y;
        }
    }
}
