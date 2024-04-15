#include "include/Functions.h"
#include "einspline/bspline.h"

using namespace gtom;

__declspec(dllexport) void* __stdcall CreateEinspline3(float* h_values, int3 dims, float3 margins)
{
	Ugrid gridx, gridy, gridz;
	gridx.start = margins.x; gridx.end = 1 - margins.x; gridx.num = dims.x;
	gridy.start = margins.y; gridy.end = 1 - margins.y; gridy.num = dims.y;
	gridz.start = margins.z; gridz.end = 1 - margins.z; gridz.num = dims.z;

	BCtype_s bcx, bcy, bcz;
	bcx.lCode = bcx.rCode = NATURAL;
	bcy.lCode = bcy.rCode = NATURAL;
	bcz.lCode = bcz.rCode = NATURAL;

	UBspline_3d_s* spline = create_UBspline_3d_s(gridz, gridy, gridx, bcz, bcy, bcx, h_values);

	return spline;
}

__declspec(dllexport) void* __stdcall CreateEinspline2(float* h_values, int2 dims, float2 margins)
{
	Ugrid gridx, gridy;
	gridx.start = margins.x; gridx.end = 1 - margins.x; gridx.num = dims.x;
	gridy.start = margins.y; gridy.end = 1 - margins.y; gridy.num = dims.y;

	BCtype_s bcx, bcy;
	bcx.lCode = bcx.rCode = NATURAL;
	bcy.lCode = bcy.rCode = NATURAL;

	UBspline_2d_s* spline = create_UBspline_2d_s(gridy, gridx, bcy, bcx, h_values);

	return spline;
}

__declspec(dllexport) void* __stdcall CreateEinspline1(float* h_values, int dims, float margins)
{
	Ugrid gridx, gridy;
	gridx.start = margins; gridx.end = 1 - margins; gridx.num = dims;

	BCtype_s bcx;
	bcx.lCode = bcx.rCode = NATURAL;

	UBspline_1d_s* spline = create_UBspline_1d_s(gridx, bcx, h_values);

	return spline;
}

__declspec(dllexport) void __stdcall EvalEinspline3(void* spline, float3* h_pos, int npos, float* h_output)
{
	//#pragma omp parallel for
#pragma loop(hint_parallel(4))
#pragma loop(ivdep)
	for (int i = 0; i < npos; i++)
		eval_UBspline_3d_s((UBspline_3d_s*)spline, h_pos[i].z, h_pos[i].y, h_pos[i].x, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline2XY(void* spline, float3* h_pos, int npos, float* h_output)
{
	//#pragma omp parallel for
#pragma loop(hint_parallel(4))
#pragma loop(ivdep)
	for (int i = 0; i < npos; i++)
		eval_UBspline_2d_s((UBspline_2d_s*)spline, h_pos[i].y, h_pos[i].x, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline2XZ(void* spline, float3* h_pos, int npos, float* h_output)
{
	//#pragma omp parallel for
#pragma loop(hint_parallel(4))
#pragma loop(ivdep)
	for (int i = 0; i < npos; i++)
		eval_UBspline_2d_s((UBspline_2d_s*)spline, h_pos[i].z, h_pos[i].x, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline2YZ(void* spline, float3* h_pos, int npos, float* h_output)
{
	//#pragma omp parallel for
#pragma loop(hint_parallel(4))
#pragma loop(ivdep)
	for (int i = 0; i < npos; i++)
		eval_UBspline_2d_s((UBspline_2d_s*)spline, h_pos[i].z, h_pos[i].y, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline1(void* spline, float* h_pos, int npos, float* h_output)
{
	//#pragma omp parallel for
#pragma loop(hint_parallel(4))
#pragma loop(ivdep)
	for (int i = 0; i < npos; i++)
		eval_UBspline_1d_s((UBspline_1d_s*)spline, h_pos[i], h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline1X(void* spline, float3* h_pos, int npos, float* h_output)
{
	//#pragma omp parallel for
#pragma loop(hint_parallel(4))
#pragma loop(ivdep)
	for (int i = 0; i < npos; i++)
		eval_UBspline_1d_s((UBspline_1d_s*)spline, h_pos[i].x, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline1Y(void* spline, float3* h_pos, int npos, float* h_output)
{
	//#pragma omp parallel for
#pragma loop(hint_parallel(4))
#pragma loop(ivdep)
	for (int i = 0; i < npos; i++)
		eval_UBspline_1d_s((UBspline_1d_s*)spline, h_pos[i].y, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline1Z(void* spline, float3* h_pos, int npos, float* h_output)
{
	//#pragma omp parallel for
	#pragma loop(hint_parallel(4))
	#pragma loop(ivdep)
	for (int i = 0; i < npos; i++)
		eval_UBspline_1d_s((UBspline_1d_s*)spline, h_pos[i].z, h_output + i);
}

__declspec(dllexport) void __stdcall DestroyEinspline(void* spline)
{
	destroy_Bspline(spline);
}

//#include <immintrin.h>
//inline __m256 lerp(__m256 a, __m256 b, __m256 t) 
//{
//	return _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(b, a), t), a);
//}
//
//__declspec(dllexport) void __stdcall EvalLinear4Batch(const int4 dims, const float* values, const float4* h_pos, const int npos, float* h_output)
//{
//	__m256 dimsfx = _mm256_set1_ps((float)dims.x - 1);
//	__m256 dimsfy = _mm256_set1_ps((float)dims.y - 1);
//	__m256 dimsfz = _mm256_set1_ps((float)dims.z - 1);
//	__m256 dimsfw = _mm256_set1_ps((float)dims.w - 1);
//
//	__m256i dimsix = _mm256_set1_epi32(dims.x - 1);
//	__m256i dimsiy = _mm256_set1_epi32(dims.y - 1);
//	__m256i dimsiz = _mm256_set1_epi32(dims.z - 1);
//	__m256i dimsiw = _mm256_set1_epi32(dims.w - 1);
//
//	__m256i zero = _mm256_setzero_si256();
//	__m256i one = _mm256_set1_epi32(1);
//
//	const float* h_posf = (const float*)h_pos;
//
//	const int dimxyz = dims.x * dims.y * dims.z;
//	const int dimxy = dims.x * dims.y;
//
//	for (int i = 0; i < 0; i += 8)
//	{
//		__m256 coordsx = _mm256_set_ps(h_posf[(i + 7) * 4], h_posf[(i + 6) * 4], h_posf[(i + 5) * 4], h_posf[(i + 4) * 4],
//									   h_posf[(i + 3) * 4], h_posf[(i + 2) * 4], h_posf[(i + 1) * 4], h_posf[i * 4]);
//		__m256 coordsy = _mm256_set_ps(h_posf[(i + 7) * 4 + 1], h_posf[(i + 6) * 4 + 1], h_posf[(i + 5) * 4 + 1], h_posf[(i + 4) * 4 + 1],
//									   h_posf[(i + 3) * 4 + 1], h_posf[(i + 2) * 4 + 1], h_posf[(i + 1) * 4 + 1], h_posf[i * 4 + 1]);
//		__m256 coordsz = _mm256_set_ps(h_posf[(i + 7) * 4 + 2], h_posf[(i + 6) * 4 + 2], h_posf[(i + 5) * 4 + 2], h_posf[(i + 4) * 4 + 2],
//									   h_posf[(i + 3) * 4 + 2], h_posf[(i + 2) * 4 + 2], h_posf[(i + 1) * 4 + 2], h_posf[i * 4 + 2]);
//		__m256 coordsw = _mm256_set_ps(h_posf[(i + 7) * 4 + 3], h_posf[(i + 6) * 4 + 3], h_posf[(i + 5) * 4 + 3], h_posf[(i + 4) * 4 + 3],
//									   h_posf[(i + 3) * 4 + 3], h_posf[(i + 2) * 4 + 3], h_posf[(i + 1) * 4 + 3], h_posf[i * 4 + 3]);
//
//		coordsx = _mm256_mul_ps(coordsx, dimsfx);
//		coordsy = _mm256_mul_ps(coordsy, dimsfy);
//		coordsz = _mm256_mul_ps(coordsz, dimsfz);
//		coordsw = _mm256_mul_ps(coordsw, dimsfw);
//
//		__m256i Pos0x = _mm256_max_epi32(zero, _mm256_min_epi32(_mm256_cvttps_epi32(coordsx), dimsix));
//		__m256i Pos0y = _mm256_max_epi32(zero, _mm256_min_epi32(_mm256_cvttps_epi32(coordsy), dimsiy));
//		__m256i Pos0z = _mm256_max_epi32(zero, _mm256_min_epi32(_mm256_cvttps_epi32(coordsz), dimsiz));
//		__m256i Pos0w = _mm256_max_epi32(zero, _mm256_min_epi32(_mm256_cvttps_epi32(coordsw), dimsiw));
//
//		__m256i Pos1x = _mm256_min_epi32(_mm256_add_epi32(Pos0x, one), dimsix);
//		__m256i Pos1y = _mm256_min_epi32(_mm256_add_epi32(Pos0y, one), dimsiy);
//		__m256i Pos1z = _mm256_min_epi32(_mm256_add_epi32(Pos0z, one), dimsiz);
//		__m256i Pos1w = _mm256_min_epi32(_mm256_add_epi32(Pos0w, one), dimsiw);
//
//		__m256i index_base0w = _mm256_mullo_epi32(Pos0w, _mm256_set1_epi32(dimxyz));
//		__m256i index_base1w = _mm256_mullo_epi32(Pos1w, _mm256_set1_epi32(dimxyz));
//		__m256i index_base0z = _mm256_mullo_epi32(Pos0z, _mm256_set1_epi32(dimxy));
//		__m256i index_base1z = _mm256_mullo_epi32(Pos1z, _mm256_set1_epi32(dimxy));
//		__m256i index_base0y = _mm256_mullo_epi32(Pos0y, _mm256_set1_epi32(dims.x));
//		__m256i index_base1y = _mm256_mullo_epi32(Pos1y, _mm256_set1_epi32(dims.x));
//
//		__m256 C0000 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base0w, _mm256_add_epi32(index_base0z, _mm256_add_epi32(index_base0y, Pos0x))), 4);
//		__m256 C0001 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base0w, _mm256_add_epi32(index_base0z, _mm256_add_epi32(index_base0y, Pos1x))), 4);
//		__m256 C0010 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base0w, _mm256_add_epi32(index_base0z, _mm256_add_epi32(index_base1y, Pos0x))), 4);
//		__m256 C0011 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base0w, _mm256_add_epi32(index_base0z, _mm256_add_epi32(index_base1y, Pos1x))), 4);
//		__m256 C0100 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base0w, _mm256_add_epi32(index_base1z, _mm256_add_epi32(index_base0y, Pos0x))), 4);
//		__m256 C0101 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base0w, _mm256_add_epi32(index_base1z, _mm256_add_epi32(index_base0y, Pos1x))), 4);
//		__m256 C0110 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base0w, _mm256_add_epi32(index_base1z, _mm256_add_epi32(index_base1y, Pos0x))), 4);
//		__m256 C0111 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base0w, _mm256_add_epi32(index_base1z, _mm256_add_epi32(index_base1y, Pos1x))), 4);
//
//		__m256 C1000 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base1w, _mm256_add_epi32(index_base0z, _mm256_add_epi32(index_base0y, Pos0x))), 4);
//		__m256 C1001 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base1w, _mm256_add_epi32(index_base0z, _mm256_add_epi32(index_base0y, Pos1x))), 4);
//		__m256 C1010 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base1w, _mm256_add_epi32(index_base0z, _mm256_add_epi32(index_base1y, Pos0x))), 4);
//		__m256 C1011 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base1w, _mm256_add_epi32(index_base0z, _mm256_add_epi32(index_base1y, Pos1x))), 4);
//		__m256 C1100 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base1w, _mm256_add_epi32(index_base1z, _mm256_add_epi32(index_base0y, Pos0x))), 4);
//		__m256 C1101 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base1w, _mm256_add_epi32(index_base1z, _mm256_add_epi32(index_base0y, Pos1x))), 4);
//		__m256 C1110 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base1w, _mm256_add_epi32(index_base1z, _mm256_add_epi32(index_base1y, Pos0x))), 4);
//		__m256 C1111 = _mm256_i32gather_ps(values, _mm256_add_epi32(index_base1w, _mm256_add_epi32(index_base1z, _mm256_add_epi32(index_base1y, Pos1x))), 4);
//
//		__m256 fracx = _mm256_sub_ps(coordsx, _mm256_cvtepi32_ps(Pos0x));
//		__m256 fracy = _mm256_sub_ps(coordsy, _mm256_cvtepi32_ps(Pos0y));
//		__m256 fracz = _mm256_sub_ps(coordsz, _mm256_cvtepi32_ps(Pos0z));
//		__m256 fracw = _mm256_sub_ps(coordsw, _mm256_cvtepi32_ps(Pos0w));
//
//		__m256 C000 = lerp(C0000, C0001, fracx);
//		__m256 C001 = lerp(C0010, C0011, fracx);
//		__m256 C010 = lerp(C0100, C0101, fracx);
//		__m256 C011 = lerp(C0110, C0111, fracx);
//		__m256 C100 = lerp(C1000, C1001, fracx);
//		__m256 C101 = lerp(C1010, C1011, fracx);
//		__m256 C110 = lerp(C1100, C1101, fracx);
//		__m256 C111 = lerp(C1110, C1111, fracx);
//
//		__m256 C00 = lerp(C000, C001, fracy);
//		__m256 C01 = lerp(C010, C011, fracy);
//		__m256 C10 = lerp(C100, C101, fracy);
//		__m256 C11 = lerp(C110, C111, fracy);
//
//		__m256 C0 = lerp(C00, C01, fracz);
//		__m256 C1 = lerp(C10, C11, fracz);
//
//		_mm256_store_ps(&h_output[i], lerp(C0, C1, fracw));
//	}
//
//	int scalarstart = 0;// (npos / 8) * 8;
//	for (int i = scalarstart; i < npos; i++)
//		h_output[i] = EvalLinear4(dims, values, h_pos[i]);
//}

__declspec(dllexport) void __stdcall EvalLinear4Batch(const int4 dims, const float* __restrict values, const float4* __restrict h_pos, const int npos, float* __restrict h_output)
{
	const float dimsfx = (float)dims.x - 1;
	const float dimsfy = (float)dims.y - 1;
	const float dimsfz = (float)dims.z - 1;
	const float dimsfw = (float)dims.w - 1;

	const float* __restrict h_posf = (const float*)h_pos;

	const int dimxyz = dims.x * dims.y * dims.z;
	const int dimxy = dims.x * dims.y;

	#pragma loop(hint_parallel(4))
	#pragma loop(ivdep)
	for (int i = 0; i < npos; i++)
	{
		const float coordsx = h_posf[i * 4] * dimsfx;
		const float coordsy = h_posf[i * 4 + 1] * dimsfy;
		const float coordsz = h_posf[i * 4 + 2] * dimsfz;
		const float coordsw = h_posf[i * 4 + 3] * dimsfw;

		const int Pos0x = max(0, min((int)coordsx, dims.x - 1));
		const int Pos0y = max(0, min((int)coordsy, dims.y - 1));
		const int Pos0z = max(0, min((int)coordsz, dims.z - 1));
		const int Pos0w = max(0, min((int)coordsw, dims.w - 1));

		const int Pos1x = min(Pos0x + 1, dims.x - 1);
		const int Pos1y = min(Pos0y + 1, dims.y - 1);
		const int Pos1z = min(Pos0z + 1, dims.z - 1);
		const int Pos1w = min(Pos0w + 1, dims.w - 1);

		const float fracx = coordsx - Pos0x;
		const float fracy = coordsy - Pos0y;
		const float fracz = coordsz - Pos0z;
		const float fracw = coordsw - Pos0w;

		const int index_base0w = Pos0w * dimxyz;
		const int index_base1w = Pos1w * dimxyz;
		const int index_base0z = Pos0z * dimxy;
		const int index_base1z = Pos1z * dimxy;
		const int index_base0y = Pos0y * dims.x;
		const int index_base1y = Pos1y * dims.x;

		float C0000 = values[index_base0w + index_base0z + index_base0y + Pos0x];
		float C0001 = values[index_base0w + index_base0z + index_base0y + Pos1x];
		float C0010 = values[index_base0w + index_base0z + index_base1y + Pos0x];
		float C0011 = values[index_base0w + index_base0z + index_base1y + Pos1x];
		float C0100 = values[index_base0w + index_base1z + index_base0y + Pos0x];
		float C0101 = values[index_base0w + index_base1z + index_base0y + Pos1x];
		float C0110 = values[index_base0w + index_base1z + index_base1y + Pos0x];
		float C0111 = values[index_base0w + index_base1z + index_base1y + Pos1x];

		float C1000 = values[index_base1w + index_base0z + index_base0y + Pos0x];
		float C1001 = values[index_base1w + index_base0z + index_base0y + Pos1x];
		float C1010 = values[index_base1w + index_base0z + index_base1y + Pos0x];
		float C1011 = values[index_base1w + index_base0z + index_base1y + Pos1x];
		float C1100 = values[index_base1w + index_base1z + index_base0y + Pos0x];
		float C1101 = values[index_base1w + index_base1z + index_base0y + Pos1x];
		float C1110 = values[index_base1w + index_base1z + index_base1y + Pos0x];
		float C1111 = values[index_base1w + index_base1z + index_base1y + Pos1x];

		float C000 = lerp(C0000, C0001, fracx);
		float C001 = lerp(C0010, C0011, fracx);
		float C010 = lerp(C0100, C0101, fracx);
		float C011 = lerp(C0110, C0111, fracx);

		float C100 = lerp(C1000, C1001, fracx);
		float C101 = lerp(C1010, C1011, fracx);
		float C110 = lerp(C1100, C1101, fracx);
		float C111 = lerp(C1110, C1111, fracx);


		float C00 = lerp(C000, C001, fracy);
		float C01 = lerp(C010, C011, fracy);

		float C10 = lerp(C100, C101, fracy);
		float C11 = lerp(C110, C111, fracy);


		float C0 = lerp(C00, C01, fracz);

		float C1 = lerp(C10, C11, fracz);

		h_output[i] = lerp(C0, C1, fracw);
	}
}

__declspec(dllexport) float __stdcall EvalLinear4(const int4 dims, const float* values, float4 coords)
{
	float4 dimsf = make_float4(dims);
	coords *= dimsf - 1;

	const int4 Pos0 = make_int4(max(0, min((int)coords.x, dims.x - 1)),
		max(0, min((int)coords.y, dims.y - 1)),
		max(0, min((int)coords.z, dims.z - 1)),
		max(0, min((int)coords.w, dims.w - 1)));

	const int4 Pos1 = make_int4(min(Pos0.x + 1, dims.x - 1),
		min(Pos0.y + 1, dims.y - 1),
		min(Pos0.z + 1, dims.z - 1),
		min(Pos0.w + 1, dims.w - 1));

	coords -= make_float4(Pos0);

	float C0000 = values[(((Pos0.w * dims.z + Pos0.z) * dims.y + Pos0.y) * dims.x + Pos0.x)];
	float C0001 = values[(((Pos0.w * dims.z + Pos0.z) * dims.y + Pos0.y) * dims.x + Pos1.x)];
	float C0010 = values[(((Pos0.w * dims.z + Pos0.z) * dims.y + Pos1.y) * dims.x + Pos0.x)];
	float C0011 = values[(((Pos0.w * dims.z + Pos0.z) * dims.y + Pos1.y) * dims.x + Pos1.x)];
	float C0100 = values[(((Pos0.w * dims.z + Pos1.z) * dims.y + Pos0.y) * dims.x + Pos0.x)];
	float C0101 = values[(((Pos0.w * dims.z + Pos1.z) * dims.y + Pos0.y) * dims.x + Pos1.x)];
	float C0110 = values[(((Pos0.w * dims.z + Pos1.z) * dims.y + Pos1.y) * dims.x + Pos0.x)];
	float C0111 = values[(((Pos0.w * dims.z + Pos1.z) * dims.y + Pos1.y) * dims.x + Pos1.x)];

	float C1000 = values[(((Pos1.w * dims.z + Pos0.z) * dims.y + Pos0.y) * dims.x + Pos0.x)];
	float C1001 = values[(((Pos1.w * dims.z + Pos0.z) * dims.y + Pos0.y) * dims.x + Pos1.x)];
	float C1010 = values[(((Pos1.w * dims.z + Pos0.z) * dims.y + Pos1.y) * dims.x + Pos0.x)];
	float C1011 = values[(((Pos1.w * dims.z + Pos0.z) * dims.y + Pos1.y) * dims.x + Pos1.x)];
	float C1100 = values[(((Pos1.w * dims.z + Pos1.z) * dims.y + Pos0.y) * dims.x + Pos0.x)];
	float C1101 = values[(((Pos1.w * dims.z + Pos1.z) * dims.y + Pos0.y) * dims.x + Pos1.x)];
	float C1110 = values[(((Pos1.w * dims.z + Pos1.z) * dims.y + Pos1.y) * dims.x + Pos0.x)];
	float C1111 = values[(((Pos1.w * dims.z + Pos1.z) * dims.y + Pos1.y) * dims.x + Pos1.x)];

	float C000 = lerp(C0000, C0001, coords.x);
	float C001 = lerp(C0010, C0011, coords.x);
	float C010 = lerp(C0100, C0101, coords.x);
	float C011 = lerp(C0110, C0111, coords.x);

	float C100 = lerp(C1000, C1001, coords.x);
	float C101 = lerp(C1010, C1011, coords.x);
	float C110 = lerp(C1100, C1101, coords.x);
	float C111 = lerp(C1110, C1111, coords.x);


	float C00 = lerp(C000, C001, coords.y);
	float C01 = lerp(C010, C011, coords.y);

	float C10 = lerp(C100, C101, coords.y);
	float C11 = lerp(C110, C111, coords.y);


	float C0 = lerp(C00, C01, coords.z);

	float C1 = lerp(C10, C11, coords.z);


	return lerp(C0, C1, coords.w);
}