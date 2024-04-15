#ifndef PREREQUISITES_CUH
#define PREREQUISITES_CUH

//#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#define tmin(a, b) (((a) < (b)) ? (a) : (b))
#define tmax(a, b) (((a) > (b)) ? (a) : (b))

#include "helper_math.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdexcept>

namespace gtom
{
#define GTOM_TESTING
	//#define GTOM_DOUBLE

#ifdef GTOM_DOUBLE
	typedef double tfloat;
	typedef cufftDoubleComplex tcomplex;
#define IS_TFLOAT_DOUBLE true
#define cmul cuCmul
#define cconj cuConj
#else
	typedef float tfloat;
	typedef cufftComplex tcomplex;
#define IS_TFLOAT_DOUBLE false
#define cmul cuCmulf
#define cconj cuConjf
#endif

	typedef unsigned char uchar;
	typedef cudaTextureObject_t cudaTex;

	struct tfloat2
	{
		tfloat x;
		tfloat y;

		__host__ __device__ tfloat2() : x((tfloat)0), y((tfloat)0) {}
		__host__ __device__ tfloat2(tfloat val) : x(val), y(val) {}
		__host__ __device__ tfloat2(tfloat x, tfloat y) : x(x), y(y) {}
		__host__ __device__ tfloat2(float2 val) : x(val.x), y(val.y) {}
	};

	struct tfloat3
	{
		tfloat x;
		tfloat y;
		tfloat z;

		__host__ __device__ tfloat3() : x((tfloat)0), y((tfloat)0), z((tfloat)0) {}
		__host__ __device__ tfloat3(tfloat x, tfloat y, tfloat z) : x(x), y(y), z(z) {}
		__host__ __device__ tfloat3(int x, int y, int z) : x((tfloat)x), y((tfloat)y), z((tfloat)z) {}
		__host__ __device__ tfloat3(tfloat val) : x(val), y(val), z(val) {}
		__host__ __device__ tfloat3(float3 val) : x(val.x), y(val.y), z(val.z) {}
	};

	struct tfloat4
	{
		tfloat x;
		tfloat y;
		tfloat z;
		tfloat w;

		__host__ __device__ tfloat4() : x((tfloat)0), y((tfloat)0), z((tfloat)0), w((tfloat)0) {}
		__host__ __device__ tfloat4(tfloat val) : x(val), y(val), z(val), w(val) {}
		__host__ __device__ tfloat4(tfloat x, tfloat y, tfloat z, tfloat w) : x(x), y(y), z(z), w(w) {}
	};

	struct tfloat5
	{
		tfloat x;
		tfloat y;
		tfloat z;
		tfloat w;
		tfloat v;

		__host__ __device__ tfloat5() : x((tfloat)0), y((tfloat)0), z((tfloat)0), w((tfloat)0), v((tfloat)0) {}
		__host__ __device__ tfloat5(tfloat val) : x(val), y(val), z(val), w(val), v(val) {}
		__host__ __device__ tfloat5(tfloat x, tfloat y, tfloat z, tfloat w, tfloat v) : x(x), y(y), z(z), w(w), v(v) {}
	};

	inline int2 toInt2(int x, int y)
	{
		int2 value = { x, y };
		return value;
	}

	inline int2 toInt2(int3 dims)
	{
		int2 value = { dims.x, dims.y };
		return value;
	}

	inline int2 toInt2FFT(int2 val)
	{
		int2 value = { val.x / 2 + 1, val.y };
		return value;
	}

	inline int2 toInt2FFT(int3 val)
	{
		int2 value = { val.x / 2 + 1, val.y };
		return value;
	}

	inline uint2 toUint2(uint x, uint y)
	{
		uint2 value = { x, y };
		return value;
	}

	inline uint2 toUint2(int2 o)
	{
		uint2 value = { (uint)o.x, (uint)o.y };
		return value;
	}

	inline int3 toInt3(int x, int y, int z)
	{
		int3 value = { x, y, z };
		return value;
	}

	inline int3 toInt3FFT(int3 val)
	{
		int3 value = { val.x / 2 + 1, val.y, val.z };
		return value;
	}

	inline int3 toInt3FFT(int2 val)
	{
		int3 value = { val.x / 2 + 1, val.y, 1 };
		return value;
	}

	inline uint3 toUint3(uint x, uint y, uint z)
	{
		uint3 value = { x, y, z };
		return value;
	}

	inline uint3 toUint3(int x, int y, int z)
	{
		uint3 value = { (uint)x, (uint)y, (uint)z };
		return value;
	}

	inline uint3 toUint3(int3 o)
	{
		uint3 value = { (uint)o.x, (uint)o.y, (uint)o.z };
		return value;
	}

	inline ushort3 toShort3(int x, int y, int z)
	{
		ushort3 value = { (ushort)x, (ushort)y, (ushort)z };
		return value;
	}

	inline int3 toInt3(int2 val)
	{
		int3 value = { val.x, val.y, 1 };
		return value;
	}

	struct imgstats5
	{
		tfloat mean;
		tfloat min;
		tfloat max;
		tfloat stddev;
		tfloat var;

		imgstats5(tfloat mean, tfloat min, tfloat max, tfloat stddev, tfloat var) : mean(mean), min(min), max(max), stddev(stddev), var(var) {}
		imgstats5() : mean(0), min(0), max(0), stddev(0), var(0) {}
	};

	template <class T1, class T2> struct tuple2
	{
		T1 t1;
		T2 t2;

		__host__ __device__ tuple2(T1 t1, T2 t2) : t1(t1), t2(t2) {}
		__host__ __device__ tuple2() {}
	};

#ifdef GTOM_DOUBLE
#define PI 3.1415926535897932384626433832795
#define PI2 6.283185307179586476925286766559
#define PIHALF 1.5707963267948966192313216916398
#else
#define PI 3.1415926535897932384626433832795f
#define PI2 6.283185307179586476925286766559f
#define PIHALF 1.5707963267948966192313216916398f
#endif
#define ToRad(x) ((tfloat)(x) / (tfloat)180 * PI)
#define ToDeg(x) ((tfloat)(x) / PI * (tfloat)180)

#define getOffset(x, y, stride) ((y) * (stride) + (x))
#define getOffset3(x, y, z, stridex, stridey) (((z) * (stridey) + (y)) * (stridex) + (x))
#define DimensionCount(dims) (3 - tmax(2 - tmax((dims).z, 1), 0) - tmax(2 - tmax((dims).y, 1), 0) - tmax(2 - tmax((dims).x, 1), 0))
#define NextMultipleOf(value, base) (((value) + (base) - 1) / (base) * (base))
#define ElementsFFT1(dims) ((dims) / 2 + 1)
#define Elements2(dims) ((dims).x * (dims).y)
#define ElementsFFT2(dims) (ElementsFFT1((dims).x) * (dims).y)
#define Elements(dims) (Elements2(dims) * (dims).z)
#define ElementsFFT(dims) (ElementsFFT1((dims).x) * (dims).y * (dims).z)
#define FFTShift(x, dim) (((x) + (dim) / 2) % (dim))
#define IFFTShift(x, dim) (((x) + ((dim) + 1) / 2) % (dim))

#define crossp(a, b) tfloat3((a).y * (b).z - (a).z * (b).y, (a).z * (b).x - (a).x * (b).z, (a).x * (b).y - (a).y - (b).x)
#define dotp(a, b) ((a).x * (b).x + (a).y * (b).y + (a).z * (b).z)
#define dotp2(a, b) ((a).x * (b).x + (a).y * (b).y)

template <typename T> __host__ __device__ int sgn(T val) 
{
	return (T(0) < val) - (val < T(0));
}


	enum T_INTERP_MODE
	{
		T_INTERP_LINEAR = 1,
		T_INTERP_CUBIC = 2,
		T_INTERP_FOURIER = 3,
		T_INTERP_SINC = 4
	};

	/**
	 * \brief Executes a call and prints the time needed for execution.
	 * \param[in] call	The call to be executed
	 */
#ifdef GTOM_TESTING
#define CUDA_MEASURE_TIME(call) \
				{ \
				float time = 0.0f; \
				cudaEvent_t start, stop; \
				cudaEventCreate(&start); \
				cudaEventCreate(&stop); \
				cudaEventRecord(start); \
				call; \
				cudaDeviceSynchronize(); \
				cudaEventRecord(stop); \
				cudaEventSynchronize(stop); \
				cudaEventElapsedTime(&time, start, stop); \
				printf("Kernel in %s executed in %f ms.\n", __FILE__, time); \
				}
#else
#define CUDA_MEASURE_TIME(call) call
#endif

	// Process has done x out of n rounds,
	// and we want a bar of width w and resolution r.
	static inline void progressbar(int x, int n, int w)
	{
		// Calculuate the ratio of complete-to-incomplete.
		float ratio = x / (float)n;
		int c = ratio * w;

		// Show the percentage complete.
		printf("%3d%% [", (int)(ratio * 100));

		// Show the load bar.
		for (int i = 0; i < c; i++)
			printf("=");

		for (int i = c; i < w; i++)
			printf(" ");

		// ANSI Control codes to go back to the
		// previous line and clear it.
		printf("]\n\033[F\033[J");
	}
}
#endif