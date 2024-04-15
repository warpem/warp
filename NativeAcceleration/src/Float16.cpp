#include "include/Functions.h"
#include <immintrin.h>

// Scalar conversion methods slightly adapted from RELION's float16.h, which is GPL2

uint16_t FloatToHalfScalar(float f) 
{
    uint32_t src = *reinterpret_cast<uint32_t*>(&f);

    uint32_t sign = (src & 0x80000000u) >> 31; // 1 bit
    uint32_t exponent = (src & 0x7f800000u) >> 23; // 10 bits
    uint32_t mantissa = (src & 0x007fffffu); // 23 bits

    uint16_t res = sign << 15; // first make a signed zero

    if (exponent == 0)
    {
        // Do nothing. Subnormal numbers will be signed zero.
        return res;
    }
    else if (exponent == 255) // Inf, -Inf, NaN 
    {
        res |= 31 << 10; // exponent is 31
        res |= mantissa >> 13; // fractional is truncated from 23 bits to 10 bits
        return res;
    }

    mantissa += 1 << 12; // add 1 to 13th bit to round.
    if (mantissa & (1 << 23)) // carry up
        exponent++;

    if (exponent > 127 + 15) // Overflow: don't create INF but truncate to MAX.
    {
        res |= 30 << 10; // maximum exponent 30 (= +15)
        res |= 0x03ffu; // 10 bits of 1s
        return res;
    }
    else if (exponent < 127 - 14) // Underflow
    {
        return res; // TODO: generate subnormali numbers instead of returning a signed zero
    }
    else
    {
        res |= ((exponent + 15 - 127) & 0x1f) << 10;
        res |= mantissa >> 13; // fractional is truncated from 23 bits to 10 bits
        return res;
    }
}

float HalfToFloatScalar(uint16_t h) 
{
    uint32_t sign = (h & 0x8000u) >> 15; // 1 bit
    uint32_t exponent = (h & 0x7c00u) >> 10; // 5 bits
    uint32_t mantissa = h & 0x03ffu; // 10 bits

    uint32_t res = sign << 31;

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
    return *reinterpret_cast<float*>(&res);
}

__declspec(dllexport) void __stdcall FloatToHalfAVX2(const float* src, uint16_t* dst, size_t count)
{
    size_t i = 0;

    if (count >= 8)
        for (i = 0; i <= count - 8; i += 8) 
        {
            __m256 src_vec = _mm256_loadu_ps(src + i);
            __m128i dst_vec = _mm256_cvtps_ph(src_vec, 0);
            _mm_storeu_si128((__m128i*)(dst + i), dst_vec);
        }

    for (; i < count; i++)
        dst[i] = FloatToHalfScalar(src[i]);

}

__declspec(dllexport) void __stdcall HalfToFloatAVX2(const uint16_t* src, float* dst, size_t count)
{
    size_t i = 0;

    if (count >= 8)
        for (i = 0; i <= count - 8; i += 8)
        {
            __m128i src_vec = _mm_loadu_si128((const __m128i*)(src + i));
            __m256 dst_vec = _mm256_cvtph_ps(src_vec);
            _mm256_storeu_ps(dst + i, dst_vec);
        }

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