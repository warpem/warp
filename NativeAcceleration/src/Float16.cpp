#include "include/Functions.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Scalar conversions originally adapted from RELION's float16.h (GPL2).
// IEEE binary16 conversion with the same round-to-nearest, ties-to-even
// behavior as F16C. This also keeps scalar tails and non-x86 CPUs consistent.
// NaNs retain their sign and representable payload and are quieted like F16C.
uint16_t FloatToHalfScalar(float f)
{
    uint32_t src;
    memcpy(&src, &f, sizeof(src));
    const uint16_t sign = (src >> 16) & 0x8000u;
    const uint32_t exponent = (src >> 23) & 0xffu;
    const uint32_t mantissa = src & 0x007fffffu;

    if (exponent == 255)
        return sign | 0x7c00u | (mantissa ? (mantissa >> 13) | 0x0200u : 0);

    if (exponent >= 143)
        return sign | 0x7c00u; // Overflow rounds to infinity.

    if (exponent < 102)
        return sign; // Smaller than half of the minimum binary16 subnormal.

    if (exponent <= 112)
    {
        // Restore the implicit leading bit before rounding to a subnormal.
        const uint32_t significand = mantissa | 0x00800000u;
        const uint32_t shift = 126 - exponent; // 14..24
        uint32_t result = significand >> shift;
        const uint32_t remainder = significand & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (result & 1)))
            ++result;
        // A carry to 0x0400 is the smallest normal binary16 value.
        return sign | result;
    }

    uint32_t result = ((exponent - 112) << 10) | (mantissa >> 13);
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1)))
        ++result; // Carry can advance the exponent, including to infinity.
    return sign | result;
}

float HalfToFloatScalar(uint16_t h)
{
    const uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x03ffu;
    uint32_t result = sign;

    if (exponent == 0)
    {
        if (mantissa != 0)
        {
            exponent = 113;
            while ((mantissa & 0x0400u) == 0)
            {
                mantissa <<= 1;
                --exponent;
            }
            result |= (exponent << 23) | ((mantissa & 0x03ffu) << 13);
        }
    }
    else if (exponent == 31)
    {
        result |= 0x7f800000u | (mantissa << 13);
        if (mantissa != 0)
            result |= 0x00400000u; // Quiet a signaling NaN.
    }
    else
        result |= ((exponent + 112) << 23) | (mantissa << 13);

    float value;
    memcpy(&value, &result, sizeof(value));
    return value;
}

__declspec(dllexport) void __stdcall FloatToHalfAVX2(const float* src, uint16_t* dst, size_t count)
{
    size_t i = 0;

#if defined(__AVX2__)
    if (count >= 8)
        for (i = 0; i <= count - 8; i += 8) 
        {
            __m256 src_vec = _mm256_loadu_ps(src + i);
            __m128i dst_vec = _mm256_cvtps_ph(src_vec, 0);
            _mm_storeu_si128((__m128i*)(dst + i), dst_vec);
        }

#endif
    for (; i < count; i++)
        dst[i] = FloatToHalfScalar(src[i]);

}

__declspec(dllexport) void __stdcall HalfToFloatAVX2(const uint16_t* src, float* dst, size_t count)
{
    size_t i = 0;

#if defined(__AVX2__)
    if (count >= 8)
        for (i = 0; i <= count - 8; i += 8)
        {
            __m128i src_vec = _mm_loadu_si128((const __m128i*)(src + i));
            __m256 dst_vec = _mm256_cvtph_ps(src_vec);
            _mm256_storeu_ps(dst + i, dst_vec);
        }

#endif
    for (; i < count; i++)
        dst[i] = HalfToFloatScalar(src[i]);
}

__declspec(dllexport) void __stdcall FloatToHalfScalars(const float* src, uint16_t* dst, size_t count)
{
    for (size_t i = 0; i < count; i++)
        dst[i] = FloatToHalfScalar(src[i]);
}

__declspec(dllexport) void __stdcall HalfToFloatScalars(const uint16_t* src, float* dst, size_t count)
{
    for (size_t i = 0; i < count; i++)
        dst[i] = HalfToFloatScalar(src[i]);
}