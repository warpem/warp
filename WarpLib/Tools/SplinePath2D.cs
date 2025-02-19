using Accord.Math.Optimization;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public class SplinePath2D : IDisposable
    {
        private IntPtr EinsplineX = IntPtr.Zero;
        private IntPtr EinsplineY = IntPtr.Zero;

        private List<float2> PointsExtended;

        public bool IsClosed { get; set; }

        public ReadOnlyCollection<float2> Points { get; }

        public SplinePath2D(float2[] points, bool isClosed)
        {
            Points = new ReadOnlyCollection<float2>(points);
            PointsExtended = new(points);

            IsClosed = isClosed;

            if (isClosed)
                for (int i = 0; i < 2; i++)
                {
                    PointsExtended.Add(points[i + 1]);
                    PointsExtended.Insert(0, points[points.Length - 2 - i]);
                }

            EinsplineX = CPU.CreateEinspline1(PointsExtended.Select(p => p.X).ToArray(), PointsExtended.Count, 0);
            EinsplineY = CPU.CreateEinspline1(PointsExtended.Select(p => p.Y).ToArray(), PointsExtended.Count, 0);
        }

        public float2[] GetInterpolated(float[] t)
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
            float[] ResultY = new float[t.Length];

            CPU.EvalEinspline1(EinsplineX, t, t.Length, ResultX);
            CPU.EvalEinspline1(EinsplineY, t, t.Length, ResultY);

            return Helper.Zip(ResultX, ResultY);
        }
        
        public float2[] GetNormals(float[] t)
        {
            // Get points slightly before and after each t value
            float2[] resultPlus = GetInterpolated(t.Select(v => v + 1e-3f).ToArray());
            float2[] resultMinus = GetInterpolated(t.Select(v => v - 1e-3f).ToArray());

            // Calculate normals for each point
            float2[] normals = new float2[t.Length];
            for (int i = 0; i < t.Length; i++)
            {
                float2 tangent = (resultPlus[i] - resultMinus[i]).Normalized();
                normals[i] = new float2(tangent.Y, -tangent.X);
            }

            return normals;
        }

        public float2[] GetControlPointNormals()
        {
            float[] t = Helper.ArrayOfFunction(i => (float)i / (Points.Count - 1), Points.Count);


            float2[] ResultPlus = GetInterpolated(t.Select(v => v + 1e-3f).ToArray());
            float2[] ResultMinus = GetInterpolated(t.Select(v => v - 1e-3f).ToArray());

            float2[] Tangents = MathHelper.Subtract(ResultPlus, ResultMinus).Select(v => v.Normalized()).ToArray();
            float2[] Normals = Tangents.Select(v => new float2(v.Y, -v.X)).ToArray();

            return Normals;
        }

        public SplinePath2D AsReversed()
        {
            float2[] PointsReversed = Points.ToArray();
            Array.Reverse(PointsReversed);

            return new SplinePath2D(PointsReversed, IsClosed);
        }

        /// <summary>
        /// A path is considered clockwise if the normals at the control points point inwards.
        /// The normals point to the right of the direction of the path.
        /// </summary>
        /// <returns></returns>
        public bool IsClockwise()
        {
            float2[] Normals = GetControlPointNormals();

            int Sum = 0;
            for (int i = 0; i < Normals.Length - (IsClosed ? 2 : 1); i++)
            {
                float2? IntersectionPoint = MathHelper.FindIntersection(Points[i], Normals[i], Points[i + 1], Normals[i + 1]);
                if (IntersectionPoint.HasValue)
                    Sum++;
                else
                    Sum--;
            }

            return Sum > 0;
        }

        ~SplinePath2D()
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
            if (EinsplineY != IntPtr.Zero)
            {
                CPU.DestroyEinspline(EinsplineY);
                EinsplineY = IntPtr.Zero;
            }
        }

        public static SplinePath2D Fit(float2[] points, bool isClosed, int nControlPoints)
        {
            float2[] ControlPoints = new float2[nControlPoints];
            float[] t = Helper.ArrayOfFunction(i => (float)i / (points.Length - 1), points.Length);

            float PointSpacing = points.Length / (float)(nControlPoints - 1);
            for (int ip = 0; ip < nControlPoints; ip++)
                ControlPoints[ip] = points[Math.Min(points.Length - 1, (int)(ip * PointSpacing))];

            int NIter = 0;

            Func<double[], double> Eval = (input) =>
            {
                float2[] NewPoints = new float2[nControlPoints];
                for (int ip = 0; ip < nControlPoints; ip++)
                    NewPoints[ip] = new float2((float)input[ip * 2], (float)input[ip * 2 + 1]);

                if (isClosed)
                    NewPoints[NewPoints.Length - 1] = NewPoints[0];

                using (var NewSpline = new SplinePath2D(NewPoints, isClosed))
                {
                    float2[] NewPointsInterpolated = NewSpline.GetInterpolated(t);
                    return MathF.Sqrt(MathHelper.Subtract(NewPointsInterpolated, points).Select(v => v.LengthSq()).Sum());
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

            double[] StartValues = Helper.ToInterleaved(ControlPoints).Select(v => (double)v).ToArray();
            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartValues.Length, Eval, Grad);
            Optimizer.MaxIterations = 10;
            Optimizer.MaxLineSearch = 5;

            Optimizer.Minimize(StartValues);

            {
                float2[] NewPoints = new float2[nControlPoints];
                for (int ip = 0; ip < nControlPoints; ip++)
                    NewPoints[ip] = new float2((float)StartValues[ip * 2], (float)StartValues[ip * 2 + 1]);

                if (isClosed)
                    NewPoints[NewPoints.Length - 1] = NewPoints[0];

                return new SplinePath2D(NewPoints, isClosed);
            }
        }
    }
}
