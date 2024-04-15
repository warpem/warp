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
        float4[] Stops;

        public ColorScale(float4[] stops)
        {
            Stops = stops;
        }

        public float4 GetColor(float v)
        {
            v *= Stops.Length - 1;

            float4 C1 = Stops[Math.Min(Stops.Length - 1, Math.Max((int)v, 0))];
            float4 C2 = Stops[Math.Min(Stops.Length - 1, Math.Max((int)v + 1, 0))];

            return float4.Lerp(C1, C2, v - (int)v);
        }
    }
}
