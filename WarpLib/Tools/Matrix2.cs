using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public class Matrix2
    {
        public float M11, M21;
        public float M12, M22;

        public float2 C1 => new float2(M11, M21);
        public float2 C2 => new float2(M12, M22);

        public float2 R1 => new float2(M11, M12);
        public float2 R2 => new float2(M21, M22);

        public Matrix2()
        {
            M11 = 1;
            M21 = 0;

            M12 = 0;
            M22 = 1;
        }

        public Matrix2(float m11, float m21,
                       float m12, float m22)
        {
            M11 = m11;
            M21 = m21;

            M12 = m12;
            M22 = m22;
        }

        public Matrix2(float[,] m)
        {
            M11 = m[0, 0];
            M21 = m[1, 0];

            M12 = m[0, 1];
            M22 = m[1, 1];
        }

        public Matrix2(float[] m)
        {
            M11 = m[0];
            M21 = m[1];

            M12 = m[2];
            M22 = m[3];
        }

        public Matrix2(Matrix3 m)
        {
            M11 = m.M11;
            M21 = m.M21;

            M12 = m.M12;
            M22 = m.M22;
        }

        public Matrix2(Matrix4 m)
        {
            M11 = m.M11;
            M21 = m.M21;

            M12 = m.M12;
            M22 = m.M22;
        }

        public Matrix2(float4 v)
        {
            M11 = v.X;
            M21 = v.Y;

            M12 = v.Z;
            M22 = v.W;
        }

        public Matrix2 GetCopy()
        {
            return new Matrix2(M11, M21, M12, M22);
        }

        public Matrix2 Transposed()
        {
            return new Matrix2(M11, M12,
                               M21, M22);
        }

        public Matrix2 NormalizedColumns()
        {
            return FromColumns(C1.Normalized(), C2.Normalized());
        }

        public float[] ToArray()
        {
            return new[] { M11, M21, M12, M22 };
        }

        public float4 ToVec()
        {
            return new float4(M11, M21, M12, M22);
        }

        public float[,] ToMultidimArray()
        {
            float[,] Result = { { M11, M12 }, { M21, M22 } };
            return Result;
        }

        public string ToMatlabString()
        {
            return $"[{M11}, {M12}; {M21}, {M22};]";
        }

        public override string ToString()
        {
            return $"{M11.ToString(CultureInfo.InvariantCulture)}, {M21.ToString(CultureInfo.InvariantCulture)}, {M12.ToString(CultureInfo.InvariantCulture)}, {M22.ToString(CultureInfo.InvariantCulture)}";
        }

        public static Matrix2 FromColumns(float2 c1, float2 c2)
        {
            return new Matrix2(c1.X, c1.Y,
                               c2.X, c2.Y);
        }

        public static Matrix2 Zero()
        {
            return new Matrix2(0, 0, 0, 0);
        }

        public static Matrix2 Identity()
        { 
            return new Matrix2(1, 0, 0, 1); 
        }

        public static Matrix2 Scale(float scaleX, float scaleY)
        {
            return new Matrix2(scaleX, 0,
                               0, scaleY);
        }

        public static Matrix2 RotateZ(float angle)
        {
            float c = MathF.Cos(angle);
            float s = MathF.Sin(angle);

            return new Matrix2(c, s, -s, c);
        }

        public static Matrix2 operator +(Matrix2 o1, Matrix2 o2)
        {
            return new Matrix2(o1.M11 + o2.M11, o1.M21 + o2.M21,
                               o1.M12 + o2.M12, o1.M22 + o2.M22);
        }

        public static Matrix2 operator -(Matrix2 o1, Matrix2 o2)
        {
            return new Matrix2(o1.M11 - o2.M11, o1.M21 - o2.M21,
                               o1.M12 - o2.M12, o1.M22 - o2.M22);
        }

        public static Matrix2 operator *(Matrix2 o1, Matrix2 o2)
        {
            return new Matrix2(o1.M11 * o2.M11 + o1.M12 * o2.M21, o1.M21 * o2.M11 + o1.M22 * o2.M21,
                               o1.M11 * o2.M12 + o1.M12 * o2.M22, o1.M21 * o2.M12 + o1.M22 * o2.M22);
        }

        public static float2 operator *(Matrix2 o1, float2 o2)
        {
            return new float2(o1.M11 * o2.X + o1.M12 * o2.Y,
                              o1.M21 * o2.X + o1.M22 * o2.Y);
        }

        public static Matrix2 operator *(Matrix2 o1, float o2)
        {
            return new Matrix2(o1.M11 * o2, o1.M21 * o2,
                               o1.M12 * o2, o1.M22 * o2);
        }

        public static Matrix2 operator /(Matrix2 o1, float o2)
        {
            return new Matrix2(o1.M11 / o2, o1.M21 / o2,
                               o1.M12 / o2, o1.M22 / o2);
        }
    }
}
