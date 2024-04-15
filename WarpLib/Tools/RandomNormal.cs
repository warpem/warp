using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public class RandomNormal
    {
        Random Generator;

        float X1, X2;
        bool Call = false;

        public RandomNormal(int seed = 123)
        {
            Generator = new Random(seed);
        }

        public float NextSingle(float mu, float sigma)
        {
            double U1, U2, W;

            if (sigma == 0)
                return mu;

            if (Call)
            {
                Call = !Call;
                return mu + sigma * X2;
            }

            do
            {
                U1 = Generator.NextDouble() * 2 - 1;
                U2 = Generator.NextDouble() * 2 - 1;
                W = Math.Pow(U1, 2) + Math.Pow(U2, 2);
            } while (W >= 1 || W == 0);

            double mult = Math.Sqrt((-2 * Math.Log(W)) / W);
            X1 = (float)(U1 * mult);
            X2 = (float)(U2 * mult);

            Call = !Call;

            return mu + sigma * X1;
        }
    }
}
