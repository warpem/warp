using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Accord;

namespace Warp.Tools
{
    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    [JsonConverter(typeof(Float5Converter))]
    public struct float5
    {
        public float X, Y, Z, W, V;

        public float5(float x, float y, float z, float w, float v)
        {
            X = x;
            Y = y;
            Z = z;
            W = w;
            V = v;
        }

        public float5(float v)
        {
            X = v;
            Y = v;
            Z = v;
            W = v;
            V = v;
        }

        public float5(byte[] value)
        {
            X = BitConverter.ToSingle(value, 0);
            Y = BitConverter.ToSingle(value, sizeof(float));
            Z = BitConverter.ToSingle(value, 2 * sizeof(float));
            W = BitConverter.ToSingle(value, 3 * sizeof(float));
            V = BitConverter.ToSingle(value, 4 * sizeof(float));
        }

        public static implicit operator byte[] (float5 value)
        {
            return Helper.Combine(new[]
            {
                BitConverter.GetBytes(value.X),
                BitConverter.GetBytes(value.Y),
                BitConverter.GetBytes(value.Z),
                BitConverter.GetBytes(value.W),
                BitConverter.GetBytes(value.V)
            });
        }

        public float Length()
        {
            return MathF.Sqrt(X * X + Y * Y + Z * Z + W * W + V * V);
        }

        public float LengthSq()
        {
            return X * X + Y * Y + Z * Z + W * W + V * V;
        }

        public float5 Normalized()
        {
            return this / Length();
        }

        public static readonly float5 UnitX = new float5(1, 0, 0, 0, 0);
        public static readonly float5 UnitY = new float5(0, 1, 0, 0, 0);
        public static readonly float5 UnitZ = new float5(0, 0, 1, 0, 0);
        public static readonly float5 UnitW = new float5(0, 0, 0, 1, 0);
        public static readonly float5 UnitV = new float5(0, 0, 0, 0, 1);

        public override bool Equals(Object obj)
        {
            return obj is float5 && this == (float5)obj;
        }

        public static bool operator ==(float5 o1, float5 o2)
        {
            return o1.X == o2.X && o1.Y == o2.Y && o1.Z == o2.Z && o1.W == o2.W && o1.V == o2.V;
        }

        public static bool operator !=(float5 o1, float5 o2)
        {
            return !(o1 == o2);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(X, Y, Z, W, V);
        }

        public static float5 operator +(float5 o1, float5 o2)
        {
            return new float5(o1.X + o2.X, o1.Y + o2.Y, o1.Z + o2.Z, o1.W + o2.W, o1.V + o2.V);
        }

        public static float5 operator -(float5 o1, float5 o2)
        {
            return new float5(o1.X - o2.X, o1.Y - o2.Y, o1.Z - o2.Z, o1.W - o2.W, o1.V - o2.V);
        }

        public static float5 operator -(float5 o1)
        {
            return new float5(-o1.X, -o1.Y, -o1.Z, -o1.W, -o1.V);
        }

        public static float5 operator *(float5 o1, float5 o2)
        {
            return new float5(o1.X * o2.X, o1.Y * o2.Y, o1.Z * o2.Z, o1.W * o2.W, o1.V * o2.V);
        }

        public static float5 operator /(float5 o1, float5 o2)
        {
            return new float5(o1.X / o2.X, o1.Y / o2.Y, o1.Z / o2.Z, o1.W / o2.W, o1.V / o2.V);
        }

        public static float5 operator +(float5 o1, float o2)
        {
            return new float5(o1.X + o2, o1.Y + o2, o1.Z + o2, o1.W + o2, o1.V + o2);
        }

        public static float5 operator -(float5 o1, float o2)
        {
            return new float5(o1.X - o2, o1.Y - o2, o1.Z - o2, o1.W - o2, o1.V - o2);
        }

        public static float5 operator *(float5 o1, float o2)
        {
            return new float5(o1.X * o2, o1.Y * o2, o1.Z * o2, o1.W * o2, o1.V * o2);
        }

        public static float5 operator /(float5 o1, float o2)
        {
            return new float5(o1.X / o2, o1.Y / o2, o1.Z / o2, o1.W / o2, o1.V / o2);
        }

        public override string ToString()
        {
            return $"{X}, {Y}, {Z}, {W}, {V}";
        }

        public static float RMSD(float5[] points)
        {
            float5 Mean = MathHelper.Mean(points);
            float Result = 0;

            for (int i = 0; i < points.Length; i++)
                Result += (points[i] - Mean).LengthSq();
            Result = MathF.Sqrt(Result / points.Length);

            return Result;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    [JsonConverter(typeof(Float4Converter))]
    public struct float4
    {
        public float X, Y, Z, W;

        public float4(float x, float y, float z, float w)
        {
            X = x;
            Y = y;
            Z = z;
            W = w;
        }

        public float4(float v)
        {
            X = v;
            Y = v;
            Z = v;
            W = v;
        }

        public float4(int4 v)
        {
            X = v.X;
            Y = v.Y;
            Z = v.Z;
            W = v.W;
        }

        public float4(byte[] value)
        {
            X = BitConverter.ToSingle(value, 0);
            Y = BitConverter.ToSingle(value, sizeof(float));
            Z = BitConverter.ToSingle(value, 2 * sizeof(float));
            W = BitConverter.ToSingle(value, 3 * sizeof(float));
        }

        public static implicit operator byte[] (float4 value)
        {
            return Helper.Combine(new[]
            {
                BitConverter.GetBytes(value.X),
                BitConverter.GetBytes(value.Y),
                BitConverter.GetBytes(value.Z),
                BitConverter.GetBytes(value.W)
            });
        }

        public float Length()
        {
            return MathF.Sqrt(X * X + Y * Y + Z * Z + W * W);
        }

        public float LengthSq()
        {
            return X * X + Y * Y + Z * Z + W * W;
        }

        public float4 Normalized()
        {
            return this / Length();
        }

        public static readonly float4 UnitX = new float4(1, 0, 0, 0);
        public static readonly float4 UnitY = new float4(0, 1, 0, 0);
        public static readonly float4 UnitZ = new float4(0, 0, 1, 0);
        public static readonly float4 UnitW = new float4(0, 0, 0, 1);

        public override bool Equals(Object obj)
        {
            return obj is float4 && this == (float4)obj;
        }

        public static bool operator ==(float4 o1, float4 o2)
        {
            return o1.X == o2.X && o1.Y == o2.Y && o1.Z == o2.Z && o1.W == o2.W;
        }

        public static bool operator !=(float4 o1, float4 o2)
        {
            return !(o1 == o2);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(X, Y, Z, W);
        }

        public static float4 operator +(float4 o1, float4 o2)
        {
            return new float4(o1.X + o2.X, o1.Y + o2.Y, o1.Z + o2.Z, o1.W + o2.W);
        }

        public static float4 operator -(float4 o1, float4 o2)
        {
            return new float4(o1.X - o2.X, o1.Y - o2.Y, o1.Z - o2.Z, o1.W - o2.W);
        }

        public static float4 operator -(float4 o1)
        {
            return new float4(-o1.X, -o1.Y, -o1.Z, -o1.W);
        }

        public static float4 operator *(float4 o1, float4 o2)
        {
            return new float4(o1.X * o2.X, o1.Y * o2.Y, o1.Z * o2.Z, o1.W * o2.W);
        }

        public static float4 operator /(float4 o1, float4 o2)
        {
            return new float4(o1.X / o2.X, o1.Y / o2.Y, o1.Z / o2.Z, o1.W / o2.W);
        }

        public static float4 operator +(float4 o1, float o2)
        {
            return new float4(o1.X + o2, o1.Y + o2, o1.Z + o2, o1.W + o2);
        }

        public static float4 operator -(float4 o1, float o2)
        {
            return new float4(o1.X - o2, o1.Y - o2, o1.Z - o2, o1.W - o2);
        }

        public static float4 operator *(float4 o1, float o2)
        {
            return new float4(o1.X * o2, o1.Y * o2, o1.Z * o2, o1.W * o2);
        }

        public static float4 operator /(float4 o1, float o2)
        {
            return new float4(o1.X / o2, o1.Y / o2, o1.Z / o2, o1.W / o2);
        }

        public override string ToString()
        {
            return $"{X}, {Y}, {Z}, {W}";
        }

        public float[] ToArray()
        {
            return new float[] { X, Y, Z, W };
        }

        public static float4 Lerp(float4 a, float4 b, float weight)
        {
            return a + (b - a) * weight;
        }

        public static float RMSD(float4[] points)
        {
            float4 Mean = MathHelper.Mean(points);
            float Result = 0;

            for (int i = 0; i < points.Length; i++)
                Result += (points[i] - Mean).LengthSq();
            Result = MathF.Sqrt(Result / points.Length);

            return Result;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    [JsonConverter(typeof(Float3Converter))]
    public struct float3
    {
        public float X, Y, Z;

        public float3(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public float3(float v)
        {
            X = v;
            Y = v;
            Z = v;
        }

        public float3(float2 v)
        {
            X = v.X;
            Y = v.Y;
            Z = 0f;
        }

        public float3(float2 v12, float v3)
        {
            X = v12.X;
            Y = v12.Y;
            Z = v3;
        }

        public float3(int3 v)
        {
            X = v.X;
            Y = v.Y;
            Z = v.Z;
        }

        public float3(byte[] value)
        {
            X = BitConverter.ToSingle(value, 0);
            Y = BitConverter.ToSingle(value, sizeof(float));
            Z = BitConverter.ToSingle(value, 2 * sizeof(float));
        }

        public static implicit operator byte[] (float3 value)
        {
            return Helper.Combine(new[]
            {
                BitConverter.GetBytes(value.X),
                BitConverter.GetBytes(value.Y),
                BitConverter.GetBytes(value.Z)
            });
        }

        public float Length()
        {
            return MathF.Sqrt(X * X + Y * Y + Z * Z);
        }

        public float LengthSq()
        {
            return X * X + Y * Y + Z * Z;
        }

        public float3 Floor()
        {
            return new float3(MathF.Floor(X), MathF.Floor(Y), MathF.Floor(Z));
        }

        public float3 Ceil()
        {
            return new float3(MathF.Ceiling(X), MathF.Ceiling(Y), MathF.Ceiling(Z));
        }

        public float3 Round()
        {
            return new float3(MathF.Round(X), MathF.Round(Y), MathF.Round(Z));
        }

        public float3 Normalized()
        {
            return this / Length();
        }

        public static float Dot(float3 a, float3 b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }

        public static float3 Cross(float3 a, float3 b)
        {
            return new float3(a.Y * b.Z - a.Z * b.Y,
                              a.Z * b.X - a.X * b.Z,
                              a.X * b.Y - a.Y * b.X);
        }

        public static readonly float3 UnitX = new float3(1, 0, 0);
        public static readonly float3 UnitY = new float3(0, 1, 0);
        public static readonly float3 UnitZ = new float3(0, 0, 1);
        public static readonly float3 UnitXY = new float3(1, 1, 0);

        public override bool Equals(Object obj)
        {
            return obj is float3 && this == (float3)obj;
        }

        public static bool operator ==(float3 o1, float3 o2)
        {
            return o1.X == o2.X && o1.Y == o2.Y && o1.Z == o2.Z;
        }

        public static bool operator !=(float3 o1, float3 o2)
        {
            return !(o1 == o2);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(X, Y, Z);
        }

        public static bool operator >(float3 o1, float3 o2)
        {
            return o1.X > o2.X && o1.Y > o2.Y && o1.Z > o2.Z;
        }

        public static bool operator <(float3 o1, float3 o2)
        {
            return o1.X < o2.X && o1.Y < o2.Y && o1.Z < o2.Z;
        }

        public override string ToString()
        {
            return X + ", " + Y + ", " + Z;
        }

        public float[] ToArray()
        {
            return new float[] { X, Y, Z };
        }

        public static float3 operator +(float3 o1, float3 o2)
        {
            return new float3(o1.X + o2.X, o1.Y + o2.Y, o1.Z + o2.Z);
        }

        public static float3 operator -(float3 o1, float3 o2)
        {
            return new float3(o1.X - o2.X, o1.Y - o2.Y, o1.Z - o2.Z);
        }

        public static float3 operator -(float3 o1)
        {
            return new float3(-o1.X, -o1.Y, -o1.Z);
        }

        public static float3 operator *(float3 o1, float3 o2)
        {
            return new float3(o1.X * o2.X, o1.Y * o2.Y, o1.Z * o2.Z);
        }

        public static float3 operator /(float3 o1, float3 o2)
        {
            return new float3(o1.X / o2.X, o1.Y / o2.Y, o1.Z / o2.Z);
        }

        public static float3 operator +(float3 o1, float o2)
        {
            return new float3(o1.X + o2, o1.Y + o2, o1.Z + o2);
        }

        public static float3 operator -(float3 o1, float o2)
        {
            return new float3(o1.X - o2, o1.Y - o2, o1.Z - o2);
        }

        public static float3 operator *(float3 o1, float o2)
        {
            return new float3(o1.X * o2, o1.Y * o2, o1.Z * o2);
        }

        public static float3 operator /(float3 o1, float o2)
        {
            return new float3(o1.X / o2, o1.Y / o2, o1.Z / o2);
        }

        public static float3[] Add(float3[] o1, float3[] o2)
        {
            if (o1.Length != o2.Length)
                throw new DimensionMismatchException();

            float3[] Result = new float3[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] + o2[i];

            return Result;
        }

        public static float3[] Add(float3[] o1, float3 o2)
        {
            float3[] Result = new float3[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] + o2;

            return Result;
        }

        public static float3[] Subtract(float3[] o1, float3[] o2)
        {
            if (o1.Length != o2.Length)
                throw new DimensionMismatchException();

            float3[] Result = new float3[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] - o2[i];

            return Result;
        }

        public static float3[] Subtract(float3[] o1, float3 o2)
        {
            float3[] Result = new float3[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] - o2;

            return Result;
        }

        public static float3[] MultiplyScalar(float3[] o1, float o2)
        {
            float3[] Result = new float3[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] * o2;

            return Result;
        }

        // Given H,S,L in range of 0-1
        // Returns a Color (RGB struct) in range of 0-255
        public static float3 HSL2RGB(float3 hsl)
        {
            float v;
            float r, g, b;

            r = hsl.Z;   // default to gray
            g = hsl.Z;
            b = hsl.Z;
            v = (hsl.Z <= 0.5f) ? (hsl.Z * (1.0f + hsl.Y)) : (hsl.Z + hsl.Y - hsl.Z * hsl.Y);
            if (v > 0)
            {
                float m;
                float sv;
                int sextant;
                float fract, vsf, mid1, mid2;

                m = hsl.Z + hsl.Z - v;
                sv = (v - m) / v;
                hsl.X *= 6.0f;
                sextant = (int)hsl.X;
                fract = hsl.X - sextant;
                vsf = v * sv * fract;
                mid1 = m + vsf;
                mid2 = v - vsf;
                switch (sextant)
                {
                    case 0:
                        r = v;
                        g = mid1;
                        b = m;
                        break;
                    case 1:
                        r = mid2;
                        g = v;
                        b = m;
                        break;
                    case 2:
                        r = m;
                        g = v;
                        b = mid1;
                        break;
                    case 3:
                        r = m;
                        g = mid2;
                        b = v;
                        break;
                    case 4:
                        r = mid1;
                        g = m;
                        b = v;
                        break;
                    case 5:
                        r = v;
                        g = m;
                        b = mid2;
                        break;
                }
            }

            return new float3(r, g, b);
        }

        public static float3 RGB2HSL(float3 rgb)
        {
            float h, s, l;

            float r = rgb.X;
            float g = rgb.Y;
            float b = rgb.Z;
            float v;
            float m;
            float vm;
            float r2, g2, b2;

            h = 0; // default to black
            s = 0;
            l = 0;
            v = Math.Max(r, g);
            v = Math.Max(v, b);
            m = Math.Min(r, g);
            m = Math.Min(m, b);
            l = (m + v) / 2.0f;
            if (l <= 0.0)
            {
                return new float3(h, s, l);
            }
            vm = v - m;
            s = vm;
            if (s > 0.0f)
            {
                s /= (l <= 0.5f) ? (v + m) : (2.0f - v - m);
            }
            else
            {
                return new float3(h, s, l);
            }
            r2 = (v - r) / vm;
            g2 = (v - g) / vm;
            b2 = (v - b) / vm;
            if (r == v)
            {
                h = (g == m ? 5.0f + b2 : 1.0f - g2);
            }
            else if (g == v)
            {
                h = (b == m ? 1.0f + r2 : 3.0f - b2);
            }
            else
            {
                h = (r == m ? 3.0f + g2 : 5.0f - r2);
            }
            h /= 6.0f;

            return new float3(h, s, l);
        }

        public static float3 Lerp(float3 a, float3 b, float weight)
        {
            return a + (b - a) * weight;
        }

        public static float3 LerpHSL(float3 a, float3 b, float weight)
        {
            float2 Ha = new float2(MathF.Cos(a.X * MathF.PI * 2), MathF.Sin(a.X * MathF.PI * 2));
            float2 Hb = new float2(MathF.Cos(b.X * MathF.PI * 2), MathF.Sin(b.X * MathF.PI * 2));

            float2 Hvec = Ha + (Hb - Ha) * weight;
            float H = MathF.Atan2(Hvec.Y, Hvec.X) / (2 * MathF.PI);
            if (H < 0)
                H = 1 + H;

            return new float3(H, a.Y + (b.Y - a.Y) * weight, a.Z + (b.Z - a.Z) * weight);
        }

        public static float RMSD(float3[] points)
        {
            float3 Mean = MathHelper.Mean(points);
            float Result = 0;

            for (int i = 0; i < points.Length; i++)
                Result += (points[i] - Mean).LengthSq();
            Result = MathF.Sqrt(Result / points.Length);

            return Result;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    [JsonConverter(typeof(Float2Converter))]
    public struct float2
    {
        public float X, Y;

        public float2(float x, float y)
        {
            X = x;
            Y = y;
        }

        public float2(float v)
        {
            X = v;
            Y = v;
        }

        public float2(int2 v)
        {
            X = v.X;
            Y = v.Y;
        }

        public float2(byte[] value)
        {
            X = BitConverter.ToSingle(value, 0);
            Y = BitConverter.ToSingle(value, sizeof(float));
        }

        public float2(float3 v)
        {
            X = v.X;
            Y = v.Y;
        }

        public static implicit operator byte[] (float2 value)
        {
            return Helper.Combine(new[]
            {
                BitConverter.GetBytes(value.X),
                BitConverter.GetBytes(value.Y)
            });
        }

        public float Length()
        {
            return MathF.Sqrt(X * X + Y * Y);
        }

        public float LengthSq()
        {
            return X * X + Y * Y;
        }

        public float2 Floor()
        {
            return new float2(MathF.Floor(X), MathF.Floor(Y));
        }

        public float2 Ceil()
        {
            return new float2(MathF.Ceiling(X), MathF.Ceiling(Y));
        }

        public float2 Normalized()
        {
            return this / Length();
        }

        public static float Dot(float2 a, float2 b)
        {
            return a.X * b.X + a.Y * b.Y;
        }

        public static readonly float2 UnitX = new float2(1, 0);
        public static readonly float2 UnitY = new float2(0, 1);

        public override bool Equals(Object obj)
        {
            return obj is float2 && this == (float2)obj;
        }

        public static bool operator ==(float2 o1, float2 o2)
        {
            return o1.X == o2.X && o1.Y == o2.Y;
        }

        public static bool operator !=(float2 o1, float2 o2)
        {
            return !(o1 == o2);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(X, Y);
        }

        public static bool operator >(float2 o1, float2 o2)
        {
            return o1.X > o2.X && o1.Y > o2.Y;
        }

        public static bool operator <(float2 o1, float2 o2)
        {
            return o1.X < o2.X && o1.Y < o2.Y;
        }

        public override string ToString()
        {
            return X + ", " + Y;
        }

        public float[] ToArray()
        {
            return new float[] { X, Y };
        }

        public static float2 operator +(float2 o1, float2 o2)
        {
            return new float2(o1.X + o2.X, o1.Y + o2.Y);
        }

        public static float2 operator -(float2 o1, float2 o2)
        {
            return new float2(o1.X - o2.X, o1.Y - o2.Y);
        }

        public static float2 operator -(float2 o1)
        {
            return new float2(-o1.X, -o1.Y);
        }

        public static float2 operator *(float2 o1, float2 o2)
        {
            return new float2(o1.X * o2.X, o1.Y * o2.Y);
        }

        public static float2 operator /(float2 o1, float2 o2)
        {
            return new float2(o1.X / o2.X, o1.Y / o2.Y);
        }

        public static float2 operator +(float2 o1, float o2)
        {
            return new float2(o1.X + o2, o1.Y + o2);
        }

        public static float2 operator -(float2 o1, float o2)
        {
            return new float2(o1.X - o2, o1.Y - o2);
        }

        public static float2 operator *(float2 o1, float o2)
        {
            return new float2(o1.X * o2, o1.Y * o2);
        }

        public static float2 operator /(float2 o1, float o2)
        {
            return new float2(o1.X / o2, o1.Y / o2);
        }

        public static float2[] Add(float2[] o1, float2[] o2)
        {
            if (o1.Length != o2.Length)
                throw new DimensionMismatchException();

            float2[] Result = new float2[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] + o2[i];

            return Result;
        }

        public static float2[] Add(float2[] o1, float2 o2)
        {
            float2[] Result = new float2[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] + o2;

            return Result;
        }

        public static float2[] Subtract(float2[] o1, float2[] o2)
        {
            if (o1.Length != o2.Length)
                throw new DimensionMismatchException();

            float2[] Result = new float2[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] - o2[i];

            return Result;
        }

        public static float2[] Subtract(float2[] o1, float2 o2)
        {
            float2[] Result = new float2[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] - o2;

            return Result;
        }

        public static float2[] MultiplyScalar(float2[] o1, float o2)
        {
            float2[] Result = new float2[o1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = o1[i] * o2;

            return Result;
        }
        
        public static float RMSD(float2[] points)
        {
            float2 Mean = MathHelper.Mean(points);
            float Result = 0;

            for (int i = 0; i < points.Length; i++)
                Result += (points[i] - Mean).LengthSq();
            Result = MathF.Sqrt(Result / points.Length);

            return Result;
        }
    }

    public class Float2Converter : JsonConverter<float2>
    {
        public override float2 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            float x = default;
            float y = default;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    return new float2 { X = x, Y = y };

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    string propertyName = reader.GetString();
                    reader.Read();
                    switch (propertyName)
                    {
                        case "X":
                            x = reader.GetSingle();
                            break;
                        case "Y":
                            y = reader.GetSingle();
                            break;
                        default:
                            throw new JsonException();
                    }
                }
            }

            throw new JsonException();
        }

        public override void Write(Utf8JsonWriter writer, float2 value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("X", value.X);
            writer.WriteNumber("Y", value.Y);
            writer.WriteEndObject();
        }
    }

    public class Float3Converter : JsonConverter<float3>
    {
        public override float3 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            float x = default;
            float y = default;
            float z = default;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    return new float3 { X = x, Y = y, Z = z };

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    string propertyName = reader.GetString();
                    reader.Read();
                    switch (propertyName)
                    {
                        case "X":
                            x = reader.GetSingle();
                            break;
                        case "Y":
                            y = reader.GetSingle();
                            break;
                        case "Z":
                            z = reader.GetSingle();
                            break;
                        default:
                            throw new JsonException();
                    }
                }
            }

            throw new JsonException();
        }

        public override void Write(Utf8JsonWriter writer, float3 value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("X", value.X);
            writer.WriteNumber("Y", value.Y);
            writer.WriteNumber("Z", value.Z);
            writer.WriteEndObject();
        }
    }

    public class Float4Converter : JsonConverter<float4>
    {
        public override float4 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            float x = default;
            float y = default;
            float z = default;
            float w = default;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    return new float4 { X = x, Y = y, Z = z, W = w };

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    string propertyName = reader.GetString();
                    reader.Read();
                    switch (propertyName)
                    {
                        case "X":
                            x = reader.GetSingle();
                            break;
                        case "Y":
                            y = reader.GetSingle();
                            break;
                        case "Z":
                            z = reader.GetSingle();
                            break;
                        case "W":
                            w = reader.GetSingle();
                            break;
                        default:
                            throw new JsonException();
                    }
                }
            }

            throw new JsonException();
        }

        public override void Write(Utf8JsonWriter writer, float4 value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("X", value.X);
            writer.WriteNumber("Y", value.Y);
            writer.WriteNumber("Z", value.Z);
            writer.WriteNumber("W", value.W);
            writer.WriteEndObject();
        }
    }

    public class Float5Converter : JsonConverter<float5>
    {
        public override float5 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            float x = default;
            float y = default;
            float z = default;
            float w = default;
            float v = default;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    return new float5 { X = x, Y = y, Z = z, W = w, V = v };

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    string propertyName = reader.GetString();
                    reader.Read();
                    switch (propertyName)
                    {
                        case "X":
                            x = reader.GetSingle();
                            break;
                        case "Y":
                            y = reader.GetSingle();
                            break;
                        case "Z":
                            z = reader.GetSingle();
                            break;
                        case "W":
                            w = reader.GetSingle();
                            break;
                        case "V":
                            v = reader.GetSingle();
                            break;
                        default:
                            throw new JsonException();
                    }
                }
            }

            throw new JsonException();
        }

        public override void Write(Utf8JsonWriter writer, float5 value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("X", value.X);
            writer.WriteNumber("Y", value.Y);
            writer.WriteNumber("Z", value.Z);
            writer.WriteNumber("W", value.W);
            writer.WriteNumber("V", value.V);
            writer.WriteEndObject();
        }
    }
}
