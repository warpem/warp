#ifndef DEVICE_FUNCTIONS_CUH
#define DEVICE_FUNCTIONS_CUH

#define GLM_FORCE_RADIANS
#define GLM_FORCE_INLINE
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"

namespace gtom
{
	inline __device__ float sinc(float x)
	{
		if (abs(x) <= 1e-4f)
			return 1.0f;
		else
			return sin(x * PI) / (x * PI);
	}

	inline __device__ double sinc(double x)
	{
		if (abs(x) <= 1e-8)
			return 1.0;
		else
			return sin(x * 3.1415926535897932384626433832795) / (x * 3.1415926535897932384626433832795);
	}

	inline __device__ glm::mat2 d_Matrix2Rotation(tfloat angle)
	{
		double c = cos(angle);
		double s = sin(angle);

		return glm::mat2(c, s, -s, c);
	}

	inline __device__ glm::mat3 d_Matrix3Euler(float3 angles)
	{
		float alpha = angles.x;
		float beta = angles.y;
		float gamma = angles.z;

		float ca, sa, cb, sb, cg, sg;
		float cc, cs, sc, ss;

		ca = cos(alpha);
		cb = cos(beta);
		cg = cos(gamma);
		sa = sin(alpha);
		sb = sin(beta);
		sg = sin(gamma);
		cc = cb * ca;
		cs = cb * sa;
		sc = sb * ca;
		ss = sb * sa;

		return glm::mat3(cg * cc - sg * sa, -sg * cc - cg * sa, sc,
						 cg * cs + sg * ca, -sg * cs + cg * ca, ss,
								  -cg * sb,			   sg * sb, cb);
	}

	inline __device__ glm::mat3 d_Matrix3RotationX(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);

		return glm::mat3(1, 0, 0, 0, c, s, 0, -s, c);
	}

	inline __device__ glm::mat3 d_Matrix3RotationY(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);

		return glm::mat3(c, 0, -s, 0, 1, 0, s, 0, c);
	}

	inline __device__ glm::mat3 d_Matrix3RotationZ(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);

		return glm::mat3(c, s, 0, -s, c, 0, 0, 0, 1);
	}

	inline __device__ float d_Bilinear(float* d_data, int2 dims, float x, float y)
	{
		x = tmax(0, tmin(x, dims.x - 1));
		y = tmax(0, tmin(y, dims.y - 1));

		int x0 = (int)x;
		x -= x0;
		int x1 = tmin(x0 + 1, dims.x - 1);

		int y0 = (int)y;
		y -= y0;
		int y1 = tmin(y0 + 1, dims.y - 1);

		float d00 = d_data[y0 * dims.x + x0];
		float d01 = d_data[y0 * dims.x + x1];
		float d10 = d_data[y1 * dims.x + x0];
		float d11 = d_data[y1 * dims.x + x1];

		float d0 = lerp(d00, d01, x);
		float d1 = lerp(d10, d11, x);

		return lerp(d0, d1, y);
	}

	inline __device__ float2 d_GetProjectionSlice(cudaTex t_volumeRe, cudaTex t_volumeIm, int dim, glm::vec3 pos, glm::mat3 rotation, float2 shift, int dimproj)
	{
		float shiftfactor = -(shift.x * pos.x + shift.y * pos.y) / dimproj * (tfloat)PI2;
		float2 shiftmultiplicator = make_cuComplex(cos(shiftfactor), sin(shiftfactor));

		float2 val;

		pos = pos * rotation;	// vector * matrix uses the transposed version, which is exactly what is needed here

		// Only asymmetric half is stored
		float is_neg_x = 1.0f;
		if (pos.x < -1e-5f)
		{
			// Get complex conjugated hermitian symmetry pair
			pos.x = abs(pos.x);
			pos.y = -pos.y;
			pos.z = -pos.z;
			is_neg_x = -1.0f;
		}

		// Trilinear interpolation (with physical coords)
		float x0 = floor(pos.x + 1e-5f);
		pos.x -= x0;
		x0 += 0.5f;
		float x1 = x0 + 1.0f;

		float y0 = floor(pos.y);
		pos.y -= y0;
		float y1 = y0 + 1;
		if (y0 < 0)
			y0 += dim;
		y0 += 0.5f;
		if (y1 < 0)
			y1 += dim;
		y1 += 0.5f;

		float z0 = floor(pos.z);
		pos.z -= z0;
		float z1 = z0 + 1;
		if (z0 < 0)
			z0 += dim;
		z0 += 0.5f;
		if (z1 < 0)
			z1 += dim;
		z1 += 0.5f;

		float2 d000 = make_cuComplex(tex3D<float>(t_volumeRe, x0, y0, z0), tex3D<float>(t_volumeIm, x0, y0, z0));
		float2 d001 = make_cuComplex(tex3D<float>(t_volumeRe, x1, y0, z0), tex3D<float>(t_volumeIm, x1, y0, z0));
		float2 d010 = make_cuComplex(tex3D<float>(t_volumeRe, x0, y1, z0), tex3D<float>(t_volumeIm, x0, y1, z0));
		float2 d011 = make_cuComplex(tex3D<float>(t_volumeRe, x1, y1, z0), tex3D<float>(t_volumeIm, x1, y1, z0));
		float2 d100 = make_cuComplex(tex3D<float>(t_volumeRe, x0, y0, z1), tex3D<float>(t_volumeIm, x0, y0, z1));
		float2 d101 = make_cuComplex(tex3D<float>(t_volumeRe, x1, y0, z1), tex3D<float>(t_volumeIm, x1, y0, z1));
		float2 d110 = make_cuComplex(tex3D<float>(t_volumeRe, x0, y1, z1), tex3D<float>(t_volumeIm, x0, y1, z1));
		float2 d111 = make_cuComplex(tex3D<float>(t_volumeRe, x1, y1, z1), tex3D<float>(t_volumeIm, x1, y1, z1));

		float2 dx00 = lerp(d000, d001, pos.x);
		float2 dx01 = lerp(d010, d011, pos.x);
		float2 dx10 = lerp(d100, d101, pos.x);
		float2 dx11 = lerp(d110, d111, pos.x);

		float2 dxy0 = lerp(dx00, dx01, pos.y);
		float2 dxy1 = lerp(dx10, dx11, pos.y);

		val = lerp(dxy0, dxy1, pos.z);

		val.y *= is_neg_x;

		return cmul(val, shiftmultiplicator);
	}

	inline __device__ float2 d_GetProjectionSliceFrom2D(cudaTex t_slicesRe, cudaTex t_slicesIm, int dim, glm::vec2 pos, glm::mat2 rotation, int slice)
	{
		pos = pos * rotation;	// vector * matrix uses the transposed version, which is exactly what is needed here

		// Only asymmetric half is stored
		float is_neg_x = 1.0f;
		if (pos.x < -1e-5f)
		{
			// Get complex conjugated hermitian symmetry pair
			pos.x = abs(pos.x);
			pos.y = -pos.y;
			is_neg_x = -1.0f;
		}

		// Trilinear interpolation (with physical coords)
		float x0 = floor(pos.x + 1e-5f);
		pos.x -= x0;
		x0 += 0.5f;
		float x1 = x0 + 1.0f;

		float y0 = floor(pos.y);
		pos.y -= y0;
		float y1 = y0 + 1;
		if (y0 < 0)
			y0 += dim;
		y0 += 0.5f;
		if (y1 < 0)
			y1 += dim;
		y1 += 0.5f;

		y0 += dim * slice;
		y1 += dim * slice;

		float2 d000 = make_cuComplex(tex2D<float>(t_slicesRe, x0, y0), tex2D<float>(t_slicesIm, x0, y0));
		float2 d001 = make_cuComplex(tex2D<float>(t_slicesRe, x1, y0), tex2D<float>(t_slicesIm, x1, y0));
		float2 d010 = make_cuComplex(tex2D<float>(t_slicesRe, x0, y1), tex2D<float>(t_slicesIm, x0, y1));
		float2 d011 = make_cuComplex(tex2D<float>(t_slicesRe, x1, y1), tex2D<float>(t_slicesIm, x1, y1));

		float2 dx00 = lerp(d000, d001, pos.x);
		float2 dx01 = lerp(d010, d011, pos.x);

		float2 val = lerp(dx00, dx01, pos.y);

		val.y *= is_neg_x;

		return val;
	}
}
#endif