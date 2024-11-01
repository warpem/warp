using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public class ColorScale
    {
        int QuantizationSteps;
        int4[] Stops;

        public ColorScale(float4[] stops, int quantizationSteps)
        {
            QuantizationSteps = quantizationSteps;
            Stops = new int4[QuantizationSteps];

            for (int i = 0; i < QuantizationSteps; i++)
                Stops[i] = GetColor(stops, (float)i / (QuantizationSteps - 1));
        }

        private int4 GetColor(float4[] stops, float v)
        {
            v *= stops.Length - 1;

            float4 C1 = stops[Math.Min(stops.Length - 1, Math.Max((int)v, 0))];
            float4 C2 = stops[Math.Min(stops.Length - 1, Math.Max((int)v + 1, 0))];

            float4 Interp = float4.Lerp(C1, C2, v - (int)v);
            return new int4((int)Math.Clamp(Interp.X * 255, 0, 255),
                            (int)Math.Clamp(Interp.Y * 255, 0, 255),
                            (int)Math.Clamp(Interp.Z * 255, 0, 255),
                            (int)Math.Clamp(Interp.W * 255, 0, 255));
        }

        public int4 GetColor(float v) => Stops[(int)(Math.Clamp(v, 0, 1) * (QuantizationSteps - 1))];
    }
}