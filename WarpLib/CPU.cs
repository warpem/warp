using System;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security;
using Warp.Tools;

namespace Warp
{
    [SuppressUnmanagedCodeSecurity]
    public static class CPU
    {
        private static bool? _NativeAvailable = null;
        public static bool NativeAvailable
        {
            get
            {
                if (_NativeAvailable == null)
                {
                    try
                    {
                        IntPtr Test = HostMalloc(1 << 10);
                        HostFree(Test);

                        _NativeAvailable = true;
                    }
                    catch (DllNotFoundException)
                    {
                        _NativeAvailable = false;
                    }
                }

                return (bool)_NativeAvailable;
            }
        }

        [DllImport("NativeAcceleration", EntryPoint = "HostMalloc")]
        public static extern IntPtr HostMalloc(long elements);

        [DllImport("NativeAcceleration", EntryPoint = "HostFree")]
        public static extern void HostFree(IntPtr h_pointer);

        [DllImport("NativeAcceleration", EntryPoint = "InitProjector")]
        public static extern void InitProjector(int3 dims, int oversampling, float[] data, IntPtr initialized, int projdim);

        [DllImport("NativeAcceleration", EntryPoint = "BackprojectorReconstruct")]
        public static extern void BackprojectorReconstruct(int3 dimsori, int oversampling, IntPtr d_data, IntPtr d_weights, string c_symmetry, bool do_reconstruct_ctf, float[] h_reconstruction);

        [DllImport("NativeAcceleration", EntryPoint = "GetAnglesCount")]
        public static extern int GetAnglesCount(int healpixorder, string c_symmetry = "C1", float limittilt = -91);

        [DllImport("NativeAcceleration", EntryPoint = "GetAngles")]
        public static extern void GetAngles(float[] h_angles, int healpixorder, string c_symmetry = "C1", float limittilt = -91);

        [DllImport("NativeAcceleration", EntryPoint = "SymmetryGetNumberOfMatrices")]
        public static extern int SymmetryGetNumberOfMatrices(string c_symmetry);

        [DllImport("NativeAcceleration", EntryPoint = "SymmetryGetMatrices")]
        public static extern void SymmetryGetMatrices(string c_symmetry, float[] h_matrices);

        [DllImport("NativeAcceleration", EntryPoint = "OptimizeWeights")]
        public static extern void OptimizeWeights(int nrecs,
                                                  float[] h_recft,
                                                  float[] h_recweights,
                                                  float[] h_r2,
                                                  int elements,
                                                  int[] h_subsets,
                                                  float[] h_bfacs,
                                                  float[] h_weightfactors,
                                                  float[] h_recsum1,
                                                  float[] h_recsum2,
                                                  float[] h_weightsum1,
                                                  float[] h_weightsum2);

        [DllImport("NativeAcceleration", EntryPoint = "SparseEigen")]
        public static extern void SparseEigen(int[] sparsei,
                                              int[] sparsej,
                                              double[] sparsevalues,
                                              int nsparse,
                                              int sidelength,
                                              int nvectors,
                                              double[] eigenvectors,
                                              double[] eigenvalues);

        // Bessel.cpp:
        [DllImport("NativeAcceleration", EntryPoint = "KaiserBessel")]
        public static extern float KaiserBessel(float r, float a, float alpha, int m);

        [DllImport("NativeAcceleration", EntryPoint = "KaiserBesselFT")]
        public static extern float KaiserBesselFT(float w, float a, float alpha, int m);

        [DllImport("NativeAcceleration", EntryPoint = "KaiserBesselProj")]
        public static extern float KaiserBesselProj(float r, float a, float alpha, int m);

        // Einspline.cpp:

        [DllImport("NativeAcceleration", EntryPoint = "CreateEinspline3")]
        public static extern IntPtr CreateEinspline3(float[] h_values, int3 dims, float3 margins);

        [DllImport("NativeAcceleration", EntryPoint = "CreateEinspline2")]
        public static extern IntPtr CreateEinspline2(float[] h_values, int2 dims, float2 margins);

        [DllImport("NativeAcceleration", EntryPoint = "CreateEinspline1")]
        public static extern IntPtr CreateEinspline1(float[] h_values, int dims, float margins);

        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline3")]
        public static extern void EvalEinspline3(IntPtr spline, float3[] h_pos, int npos, float[] h_output);
        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline3")]
        public static extern void EvalEinspline3(IntPtr spline, ref float3 h_pos, int npos, ref float h_output);

        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline2XY")]
        public static extern void EvalEinspline2XY(IntPtr spline, float3[] h_pos, int npos, float[] h_output);
        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline2XY")]
        public static extern void EvalEinspline2XY(IntPtr spline, ref float3 h_pos, int npos, ref float h_output);

        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline2XZ")]
        public static extern void EvalEinspline2XZ(IntPtr spline, float3[] h_pos, int npos, float[] h_output);
        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline2XZ")]
        public static extern void EvalEinspline2XZ(IntPtr spline, ref float3 h_pos, int npos, ref float h_output);

        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline2YZ")]
        public static extern void EvalEinspline2YZ(IntPtr spline, float3[] h_pos, int npos, float[] h_output);
        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline2YZ")]
        public static extern void EvalEinspline2YZ(IntPtr spline, ref float3 h_pos, int npos, ref float h_output);

        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline1")]
        public static extern void EvalEinspline1(IntPtr spline, float[] h_pos, int npos, float[] h_output);

        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline1X")]
        public static extern void EvalEinspline1X(IntPtr spline, float3[] h_pos, int npos, float[] h_output);
        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline1X")]
        public static extern void EvalEinspline1X(IntPtr spline, ref float3 h_pos, int npos, ref float h_output);

        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline1Y")]
        public static extern void EvalEinspline1Y(IntPtr spline, float3[] h_pos, int npos, float[] h_output);
        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline1Y")]
        public static extern void EvalEinspline1Y(IntPtr spline, ref float3 h_pos, int npos, ref float h_output);

        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline1Z")]
        public static extern void EvalEinspline1Z(IntPtr spline, float3[] h_pos, int npos, float[] h_output);
        [DllImport("NativeAcceleration", EntryPoint = "EvalEinspline1Z")]
        public static extern void EvalEinspline1Z(IntPtr spline, ref float3 h_pos, int npos, ref float h_output);

        [DllImport("NativeAcceleration", EntryPoint = "DestroyEinspline")]
        public static extern void DestroyEinspline(IntPtr spline);

        [DllImport("NativeAcceleration", EntryPoint = "EvalLinear4Batch")]
        public static extern void EvalLinear4Batch(int4 dims, float[] values, float4[] h_pos, int npos, float[] h_output);

        [DllImport("NativeAcceleration", EntryPoint = "EvalLinear4Batch")]
        public static extern void EvalLinear4Batch(int4 dims, float[] values, ref float4 h_pos, int npos, ref float h_output);

        [DllImport("NativeAcceleration", EntryPoint = "EvalLinear4")]
        public static extern float EvalLinear4(int4 dims, float[] values, float4 coords);

        // FFT.cpp:

        [DllImport("NativeAcceleration", EntryPoint = "FFT_CPU")]
        public static extern void FFT_CPU(float[] data, float[] result, int3 dims, int nthreads);

        [DllImport("NativeAcceleration", EntryPoint = "IFFT_CPU")]
        public static extern void IFFT_CPU(float[] data, float[] result, int3 dims, int nthreads);

        // Float16.cpp:

        [DllImport("NativeAcceleration", EntryPoint = "FloatToHalfAVX2")]
        public static extern unsafe void FloatToHalfAVX2(float* src, ushort * dst, long count);

        [DllImport("NativeAcceleration", EntryPoint = "HalfToFloatAVX2")]
        public static extern unsafe void HalfToFloatAVX2(ushort* src, float* dst, long count);

        [DllImport("NativeAcceleration", EntryPoint = "FloatToHalfScalars")]
        public static extern unsafe void FloatToHalfScalars(float* src, ushort* dst, long count);

        [DllImport("NativeAcceleration", EntryPoint = "HalfToFloatScalars")]
        public static extern unsafe void HalfToFloatScalars(ushort* src, float* dst, long count);

        public static unsafe void FloatToHalfManaged(float* src, ushort* dst, long count)
        {
            uint* isrc = (uint*)src;
            for (long i = 0; i < count; i++)
            {
                uint original = isrc[i];

                uint sign = (original & 0x80000000u) >> 31; // 1 bit
                uint exponent = (original & 0x7f800000u) >> 23; // 10 bits
                uint mantissa = (original & 0x007fffffu); // 23 bits

                ushort res = (ushort)(sign << 15); // first make a signed zero

                if (exponent == 0)
                {
                    // Do nothing. Subnormal numbers will be signed zero.
                    dst[i] = res;
                    continue;
                }
                else if (exponent == 255) // Inf, -Inf, NaN 
                {
                    res |= 31 << 10; // exponent is 31
                    res |= (ushort)(mantissa >> 13); // fractional is truncated from 23 bits to 10 bits

                    dst[i] = res;
                    continue;
                }

                mantissa += 1 << 12; // add 1 to 13th bit to round.
                if ((mantissa & (1 << 23)) != 0) // carry up
                    exponent++;

                if (exponent > 127 + 15) // Overflow: don't create INF but truncate to MAX.
                {
                    res |= 30 << 10; // maximum exponent 30 (= +15)
                    res |= (ushort)0x03ffu; // 10 bits of 1s
                }
                else if (exponent < 127 - 14) // Underflow
                {
                }
                else
                {
                    res |= (ushort)(((exponent + 15 - 127) & 0x1f) << 10);
                    res |= (ushort)(mantissa >> 13); // fractional is truncated from 23 bits to 10 bits
                }

                dst[i] = res;
            }
        }

        public static unsafe void HalfToFloatManaged(ushort* src, float* dst, long count)
        {
            for (long i = 0; i < count; i++)
            {
                uint sign = (src[i] & 0x8000u) >> 15; // 1 bit
                uint exponent = (src[i] & 0x7c00u) >> 10; // 5 bits
                uint mantissa = src[i] & 0x03ffu; // 10 bits

                uint res = sign << 31;

                if (exponent == 0)
                {
                }
                else if (exponent == 31) // Inf, -Inf, NaN
                {
                    res |= 255 << 23; // exponent is 255
                    res |= mantissa << 13; // keep fractional by expanding from 10 bits to 23 bits
                }
                else // normal numbers
                {
                    res |= (exponent + 127 - 15) << 23; // shift the offset
                    res |= mantissa << 13; // keep fractional by expanding from 10 bits to 23 bits
                }

                dst[i] = *((float*)&res);
            }
        }

        public static unsafe void FloatToHalf(float* src, ushort* dst, long count)
        {
            if (NativeAvailable)
            {
                if (Avx2.IsSupported)
                    FloatToHalfAVX2(src, dst, count);
                else
                    FloatToHalfScalars(src, dst, count);
            }
            else
                FloatToHalfManaged(src, dst, count);
        }

        public static unsafe void HalfToFloat(ushort* src, float* dst, long count)
        {
            if (NativeAvailable)
            {
                if (Avx2.IsSupported)
                    HalfToFloatAVX2(src, dst, count);
                else
                    HalfToFloatScalars(src, dst, count);
            }
            else
                HalfToFloatManaged(src, dst, count);
        }

        // FSC.cpp:

        [DllImport("NativeAcceleration", EntryPoint = "ConicalFSC")]
        public static extern void ConicalFSC(float[] volume1ft, float[] volume2ft, int3 dims, float[] directions, int ndirections, float anglestep, int minshell, float threshold, float particlefraction, float[] result);
    }
}
