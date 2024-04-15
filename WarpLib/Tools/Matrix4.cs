using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    public struct Matrix4
    {
        public float M11, M21, M31, M41;
        public float M12, M22, M32, M42;
        public float M13, M23, M33, M43;
        public float M14, M24, M34, M44;

        public float4 C1 => new float4(M11, M21, M31, M41);
        public float4 C2 => new float4(M12, M22, M32, M42);
        public float4 C3 => new float4(M13, M23, M33, M43);
        public float4 C4 => new float4(M14, M24, M34, M44);

        public float4 R1 => new float4(M11, M12, M13, M14);
        public float4 R2 => new float4(M21, M22, M23, M24);
        public float4 R3 => new float4(M31, M32, M33, M34);
        public float4 R4 => new float4(M41, M42, M43, M44);

        public Matrix4()
        {
            M11 = 1;
            M21 = 0;
            M31 = 0;
            M41 = 0;

            M12 = 0;
            M22 = 1;
            M32 = 0;
            M42 = 0;

            M13 = 0;
            M23 = 0;
            M33 = 1;
            M43 = 0;

            M14 = 0;
            M24 = 0;
            M34 = 0;
            M44 = 1;
        }

        public Matrix4(float m11, float m21, float m31, float m41,
                       float m12, float m22, float m32, float m42,
                       float m13, float m23, float m33, float m43,
                       float m14, float m24, float m34, float m44)
        {
            M11 = m11;
            M21 = m21;
            M31 = m31;
            M41 = m41;

            M12 = m12;
            M22 = m22;
            M32 = m32;
            M42 = m42;

            M13 = m13;
            M23 = m23;
            M33 = m33;
            M43 = m43;

            M14 = m14;
            M24 = m24;
            M34 = m34;
            M44 = m44;
        }

        public Matrix4(Matrix3 m)
        {
            M11 = m.M11;
            M21 = m.M21;
            M31 = m.M31;
            M41 = 0;

            M12 = m.M12;
            M22 = m.M22;
            M32 = m.M32;
            M42 = 0;

            M13 = m.M13;
            M23 = m.M23;
            M33 = m.M33;
            M43 = 0;

            M14 = 0;
            M24 = 0;
            M34 = 0;
            M44 = 1;
        }

        public Matrix4 Transposed()
        {
            return new Matrix4(M11, M12, M13, M14,
                               M21, M22, M23, M24,
                               M31, M32, M33, M34,
                               M41, M42, M43, M44);
        }

        public Matrix4 NormalizedColumns()
        {
            return FromColumns(C1.Normalized(), C2.Normalized(), C3.Normalized(), C4.Normalized());
        }

        public string ToMatlabString()
        {
            return $"[{M11}, {M12}, {M13}, {M14}; {M21}, {M22}, {M23}, {M24}; {M31}, {M32}, {M33}, {M34}; {M41}, {M42}, {M43}, {M44}]";
        }

        public static Matrix4 FromColumns(float4 c1, float4 c2, float4 c3, float4 c4)
        {
            return new Matrix4(c1.X, c1.Y, c1.Z, c1.W,
                               c2.X, c2.Y, c2.Z, c2.W,
                               c3.X, c3.Y, c3.Z, c3.W,
                               c4.X, c4.Y, c4.Z, c4.W);
        }

        public static Matrix4 Zero()
        {
            return new Matrix4(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        public static Matrix4 Scale(float scaleX, float scaleY, float scaleZ)
        {
            return new Matrix4(scaleX, 0, 0, 0,
                               0, scaleY, 0, 0,
                               0, 0, scaleZ, 0,
                               0, 0, 0,      1);
        }

        public static Matrix4 RotateX(float angle)
        {
            float c = MathF.Cos(angle);
            float s = MathF.Sin(angle);

            return new Matrix4(1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1);
        }

        public static Matrix4 RotateY(float angle)
        {
            float c = MathF.Cos(angle);
            float s = MathF.Sin(angle);

            return new Matrix4(c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1);
        }

        public static Matrix4 RotateZ(float angle)
        {
            float c = MathF.Cos(angle);
            float s = MathF.Sin(angle);

            return new Matrix4(c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
        }

        public static Matrix4 RotateAxis(float3 axis, float angle)
        {
            float3 r = axis.Normalized();
            float CT = MathF.Cos(angle);
            float ST = MathF.Sin(angle);

            return new Matrix4(CT + (1 - CT) * r.X * r.X,       (1 - CT) * r.X * r.Y + r.Z * ST, (1 - CT) * r.X * r.Z - r.Y * ST, 0,
                               (1 - CT) * r.X * r.Y - r.Z * ST, CT + (1 - CT) * r.Y * r.Y,       (1 - CT) * r.Y * r.Z + r.X * ST, 0,
                               (1 - CT) * r.X * r.Z + r.Y * ST, (1 - CT) * r.Y * r.Z - r.X * ST, CT + (1 - CT) * r.Z * r.Z,       0,
                               0,                               0,                               0,                               1);
        }

        public static Matrix4 Euler(float rot, float tilt, float psi)
        {
            float ca, sa, cb, sb, cg, sg;
            float cc, cs, sc, ss;

            ca = MathF.Cos(rot);
            cb = MathF.Cos(tilt);
            cg = MathF.Cos(psi);
            sa = MathF.Sin(rot);
            sb = MathF.Sin(tilt);
            sg = MathF.Sin(psi);
            cc = cb * ca;
            cs = cb * sa;
            sc = sb * ca;
            ss = sb * sa;

            Matrix4 A = new Matrix4();
            A.M11 = cg * cc - sg * sa;
            A.M12 = cg * cs + sg * ca;
            A.M13 = -cg * sb;
            A.M21 = -sg * cc - cg * sa;
            A.M22 = -sg * cs + cg * ca;
            A.M23 = sg * sb;
            A.M31 = sc;
            A.M32 = ss;
            A.M33 = cb;

            return A;
        }

        public static Matrix4 Translate(float x, float y, float z)
        {
            Matrix4 Result = new Matrix4();
            Result.M14 = x;
            Result.M24 = y;
            Result.M34 = z;

            return Result;
        }

        public static Matrix4 Translate(float3 v)
        {
            return Translate(v.X, v.Y, v.Z);
        }

        public static float3 EulerFromMatrix(Matrix4 a)
        {
            float alpha, beta, gamma;
            float abs_sb, sign_sb;

            abs_sb = MathF.Sqrt(a.M13 * a.M13 + a.M23 * a.M23);
            if (abs_sb > 16 * 1.192092896e-07f)
            {
                gamma = MathF.Atan2(a.M23, -a.M13);
                alpha = MathF.Atan2(a.M32, a.M31);
                if (MathF.Abs(MathF.Sin(gamma)) < 1.192092896e-07f)
                    sign_sb = MathF.Sign(-a.M13 / MathF.Cos(gamma));
                else
                    sign_sb = (MathF.Sin(gamma) > 0) ? MathF.Sign(a.M23) : -MathF.Sign(a.M23);
                beta = MathF.Atan2(sign_sb * abs_sb, a.M33);
            }
            else
            {
                if (MathF.Sign(a.M33) > 0)
                {
                    // Let's consider the matrix as a rotation around Z
                    alpha = 0;
                    beta = 0;
                    gamma = MathF.Atan2(-a.M21, a.M11);
                }
                else
                {
                    alpha = 0;
                    beta = MathF.PI;
                    gamma = MathF.Atan2(a.M21, -a.M11);
                }
            }

            return new float3(alpha, beta, gamma);
        }

        public static Matrix4 operator +(Matrix4 o1, Matrix4 o2)
        {
            return new Matrix4(o1.M11 + o2.M11, o1.M21 + o2.M21, o1.M31 + o2.M31, o1.M41 + o2.M41,
                               o1.M12 + o2.M12, o1.M22 + o2.M22, o1.M32 + o2.M32, o1.M42 + o2.M42,
                               o1.M13 + o2.M13, o1.M23 + o2.M23, o1.M33 + o2.M33, o1.M43 + o2.M43,
                               o1.M14 + o2.M14, o1.M24 + o2.M24, o1.M34 + o2.M34, o1.M44 + o2.M44);
        }

        public static Matrix4 operator -(Matrix4 o1, Matrix4 o2)
        {
            return new Matrix4(o1.M11 - o2.M11, o1.M21 - o2.M21, o1.M31 - o2.M31, o1.M41 - o2.M41,
                               o1.M12 - o2.M12, o1.M22 - o2.M22, o1.M32 - o2.M32, o1.M42 - o2.M42,
                               o1.M13 - o2.M13, o1.M23 - o2.M23, o1.M33 - o2.M33, o1.M43 - o2.M43,
                               o1.M14 - o2.M14, o1.M24 - o2.M24, o1.M34 - o2.M34, o1.M44 - o2.M44);
        }

        public static Matrix4 operator *(Matrix4 o1, Matrix4 o2)
        {
            return new Matrix4(o1.M11 * o2.M11 + o1.M12 * o2.M21 + o1.M13 * o2.M31 + o1.M14 * o2.M41, o1.M21 * o2.M11 + o1.M22 * o2.M21 + o1.M23 * o2.M31 + o1.M24 * o2.M41, o1.M31 * o2.M11 + o1.M32 * o2.M21 + o1.M33 * o2.M31 + o1.M34 * o2.M41, o1.M41 * o2.M11 + o1.M42 * o2.M21 + o1.M43 * o2.M31 + o1.M44 * o2.M41,
                               o1.M11 * o2.M12 + o1.M12 * o2.M22 + o1.M13 * o2.M32 + o1.M14 * o2.M42, o1.M21 * o2.M12 + o1.M22 * o2.M22 + o1.M23 * o2.M32 + o1.M24 * o2.M42, o1.M31 * o2.M12 + o1.M32 * o2.M22 + o1.M33 * o2.M32 + o1.M34 * o2.M42, o1.M41 * o2.M12 + o1.M42 * o2.M22 + o1.M43 * o2.M32 + o1.M44 * o2.M42,
                               o1.M11 * o2.M13 + o1.M12 * o2.M23 + o1.M13 * o2.M33 + o1.M14 * o2.M43, o1.M21 * o2.M13 + o1.M22 * o2.M23 + o1.M23 * o2.M33 + o1.M24 * o2.M43, o1.M31 * o2.M13 + o1.M32 * o2.M23 + o1.M33 * o2.M33 + o1.M34 * o2.M43, o1.M41 * o2.M13 + o1.M42 * o2.M23 + o1.M43 * o2.M33 + o1.M44 * o2.M43,
                               o1.M11 * o2.M14 + o1.M12 * o2.M24 + o1.M13 * o2.M34 + o1.M14 * o2.M44, o1.M21 * o2.M14 + o1.M22 * o2.M24 + o1.M23 * o2.M34 + o1.M24 * o2.M44, o1.M31 * o2.M14 + o1.M32 * o2.M24 + o1.M33 * o2.M34 + o1.M34 * o2.M44, o1.M41 * o2.M14 + o1.M42 * o2.M24 + o1.M43 * o2.M34 + o1.M44 * o2.M44);
        }

        public static float4 operator *(Matrix4 o1, float4 o2)
        {
            return new float4(o1.M11 * o2.X + o1.M12 * o2.Y + o1.M13 * o2.Z + o1.M14 * o2.W,
                              o1.M21 * o2.X + o1.M22 * o2.Y + o1.M23 * o2.Z + o1.M24 * o2.W,
                              o1.M31 * o2.X + o1.M32 * o2.Y + o1.M33 * o2.Z + o1.M34 * o2.W,
                              o1.M41 * o2.X + o1.M42 * o2.Y + o1.M43 * o2.Z + o1.M44 * o2.W);
        }

        public static float3 operator *(Matrix4 o1, float3 o2)
        {
            return new float3(o1.M11 * o2.X + o1.M12 * o2.Y + o1.M13 * o2.Z + o1.M14,
                              o1.M21 * o2.X + o1.M22 * o2.Y + o1.M23 * o2.Z + o1.M24,
                              o1.M31 * o2.X + o1.M32 * o2.Y + o1.M33 * o2.Z + o1.M34);
        }

        public static Matrix4 operator /(Matrix4 o1, float o2)
        {
            return new Matrix4(o1.M11 / o2, o1.M21 / o2, o1.M31 / o2, o1.M41 / o2,
                               o1.M12 / o2, o1.M22 / o2, o1.M32 / o2, o1.M42 / o2,
                               o1.M13 / o2, o1.M23 / o2, o1.M33 / o2, o1.M43 / o2,
                               o1.M14 / o2, o1.M24 / o2, o1.M34 / o2, o1.M44 / o2);
        }
    }
}
