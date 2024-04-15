#ifndef AVX_HELPER_H
#define AVX_HELPER_H

#include "immintrin.h"
#include "complex.h"

namespace relion
{

#ifndef FLOAT_PRECISION

	inline __m256d _avx_complex_mul_scalar_2(Complex* c, double* s)
	{
		__m256d __c = _mm256_load_pd((double*)c);
		__m256d __s = _mm256_setr_pd(s[0], s[0], s[1], s[1]);

		return _mm256_mul_pd(__c, __s);
	}

	inline __m256d _avx_complex_mul_complex_2(Complex* c1, Complex* c2)
	{
		__m256d __c1 = _mm256_load_pd((double*)c1);
		__m256d __c2 = _mm256_load_pd((double*)c2);
		__m256d __neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

		__m256d __c3 = _mm256_mul_pd(__c1, __c2);

		__c2 = _mm256_permute_pd(__c2, 0x5);
		__c2 = _mm256_mul_pd(__c2, __neg);

		__m256d __c4 = _mm256_mul_pd(__c1, __c2);

		return _mm256_hsub_pd(__c3, __c4);
	}

#define LIN_INTERP_AVX(l, r, a) _mm256_add_pd(l, _mm256_mul_pd(_mm256_sub_pd(r, l), a))

#else

	inline __m256 _avx_complex_mul_scalar_4(Complex* c, float* s)
	{
		__m256 __c = _mm256_load_ps((float*)c);
		__m256 __s = _mm256_setr_ps(s[0], s[0], s[1], s[1], s[2], s[2], s[3], s[3]);

		return _mm256_mul_ps(__c, __s);
	}

#define LIN_INTERP_AVX(l, r, a) _mm_add_ps(l, _mm_mul_ps(_mm_sub_ps(r, l), a))

#endif

#endif
}