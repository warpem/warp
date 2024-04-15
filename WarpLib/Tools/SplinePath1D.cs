using Accord.Math.Optimization;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public class SplinePath1D : IDisposable
    {
        private IntPtr EinsplineX = IntPtr.Zero;

        private List<float> PointsExtended;

        public bool IsClosed { get; set; }

        public ReadOnlyCollection<float> Points { get; }

        public SplinePath1D(float[] points, bool isClosed)
        {
            Points = new ReadOnlyCollection<float>(points);
            PointsExtended = new(points);

            IsClosed = isClosed;

            if (isClosed)
                for (int i = 0; i < 2; i++)
                {
                    PointsExtended.Add(points[i + 1]);
                    PointsExtended.Insert(0, points[points.Length - 2 - i]);
                }

            EinsplineX = CPU.CreateEinspline1(PointsExtended.ToArray(), PointsExtended.Count, 0);
        }

        public float[] GetInterpolated(float[] t)
        {
            if (IsClosed)
            {
                float[] tScaled = new float[t.Length];
                float Scale = (Points.Count - 1) / (float)(PointsExtended.Count - 1);
                float Offset = 2f / (Points.Count - 1);

                for (int i = 0; i < t.Length; i++)
                    tScaled[i] = (t[i] + Offset) * Scale;

                t = tScaled;
            }

            float[] ResultX = new float[t.Length];

            CPU.EvalEinspline1(EinsplineX, t, t.Length, ResultX);

            return ResultX;
        }

        public SplinePath1D AsReversed()
        {
            float[] PointsReversed = Points.ToArray();
            Array.Reverse(PointsReversed);

            return new SplinePath1D(PointsReversed, IsClosed);
        }

        ~SplinePath1D()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (EinsplineX != IntPtr.Zero)
            {
                CPU.DestroyEinspline(EinsplineX);
                EinsplineX = IntPtr.Zero;
            }
        }

        public static SplinePath1D Fit(float[] points, bool isClosed, int nControlPoints)
        {
            float[] ControlPoints = new float[nControlPoints];
            float[] t = Helper.ArrayOfFunction(i => (float)i / (points.Length - 1), points.Length);

            float PointSpacing = points.Length / (float)(nControlPoints - 1);
            for (int ip = 0; ip < nControlPoints; ip++)
                ControlPoints[ip] = points[Math.Min(points.Length - 1, (int)(ip * PointSpacing))];

            int NIter = 0;

            Func<double[], double> Eval = (input) =>
            {
                float[] NewPoints = new float[nControlPoints];
                for (int ip = 0; ip < nControlPoints; ip++)
                    NewPoints[ip] = (float)input[ip];

                if (isClosed)
                    NewPoints[NewPoints.Length - 1] = NewPoints[0];

                using (var NewSpline = new SplinePath1D(NewPoints, isClosed))
                {
                    float[] NewPointsInterpolated = NewSpline.GetInterpolated(t);
                    return MathF.Sqrt(MathHelper.Subtract(NewPointsInterpolated, points).Select(v => v * v).Sum() / input.Length);
                }
            };

            Func<double[], double[]> Grad = (input) =>
            {
                double[] Result = new double[input.Length];

                //if (++NIter > 20)
                //    return Result;

                for (int i = 0; i < (isClosed ? input.Length - 2 : input.Length); i++)
                {
                    double[] InputPlus = input.ToArray();
                    InputPlus[i] += 1e-3;
                    double[] InputMinus = input.ToArray();
                    InputMinus[i] -= 1e-3;

                    Result[i] = (Eval(InputPlus) - Eval(InputMinus)) / 2e-3;
                }

                return Result;
            };

            double[] StartValues = ControlPoints.Select(v => (double)v).ToArray();
            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartValues.Length, Eval, Grad);

            Optimizer.Minimize(StartValues);

            {
                float[] NewPoints = new float[nControlPoints];
                for (int ip = 0; ip < nControlPoints; ip++)
                    NewPoints[ip] = (float)StartValues[ip];

                if (isClosed)
                    NewPoints[NewPoints.Length - 1] = NewPoints[0];

                return new SplinePath1D(NewPoints, isClosed);
            }
        }
    }
}
