#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/glm/gtc/matrix_transform.hpp"
#include "gtom/include/Angles.cuh"

namespace gtom
{
	glm::mat4 Matrix4Euler(tfloat3 angles)
	{
		/*float phi = angles.x;
		float theta = angles.y;
		float psi = angles.z;

		return Matrix4RotationZ(psi) * Matrix4RotationY(theta) * Matrix4RotationZ(phi);*/
		float alpha = angles.x;
		float beta = angles.y;
		float gamma = angles.z;

		/*return Matrix3RotationZ(psi) * Matrix3RotationY(theta) * Matrix3RotationZ(phi);*/
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

		return glm::mat4(cg * cc - sg * sa, -sg * cc - cg * sa, sc, 0,
						 cg * cs + sg * ca, -sg * cs + cg * ca, ss, 0,
						 -cg * sb, sg * sb, cb, 0,
						 0, 0, 0, 1);
	}

	glm::mat3 Matrix3Euler(tfloat3 angles)
	{
		/*float psi = angles.x;
		float theta = angles.y;
		float phi = angles.z;

		return Matrix3RotationZ(psi) * Matrix3RotationY(theta) * Matrix3RotationZ(phi);*/

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
						 -cg * sb, sg * sb, cb);
	}

	glm::mat4 Matrix4EulerLegacy(tfloat2 angles)
	{
		float phi = PI / 2.0f - angles.x;
		float psi = angles.x - PI / 2.0f;
		float theta = angles.y;

		return glm::transpose(Matrix4Euler(tfloat3(phi, theta, psi)));
	}

	tfloat3 EulerFromMatrix(glm::mat4 m)
	{
		// In glm, m[x][y] is element at column x, row y

		float phi = 0.0f, theta = 0.0f, psi = 0.0f;
		float abssintheta = sqrt(m[2][0] * m[2][0] + m[2][1] * m[2][1]);
		if (abssintheta > 0.00001f)
		{
			psi = PI - atan2(m[1][2], m[0][2]);
			phi = PI - atan2(m[2][1], -m[2][0]);
			float s;
			if (sin(phi) == 0.0f)
				s = -m[2][0] / cos(phi) >= 0.0f ? 1.0f : -1.0f;
			else
				s = m[2][1] / sin(phi) >= 0.0f ? 1.0f : -1.0f;
			theta = atan2(s * abssintheta, m[2][2]);
		}
		else
		{
			psi = 0.0f;
			if (m[2][2] > 0.0f)
			{
				theta = 0.0f;
				phi = atan2(m[0][1], m[0][0]);
			}
			else
			{
				theta = PI;
				phi = atan2(m[0][1], m[0][0]);
			}
		}

		return tfloat3(phi, theta, psi);
	}

	tfloat3 EulerFromMatrix(glm::mat3 m)
	{
		// In glm, m[x][y] is element at column x, row y

		float phi = 0.0f, theta = 0.0f, psi = 0.0f;
		float abssintheta = sqrt(m[2][0] * m[2][0] + m[2][1] * m[2][1]);
		if (abssintheta > 0.00001f)
		{
			psi = PI - atan2(m[1][2], m[0][2]);
			phi = PI - atan2(m[2][1], -m[2][0]);
			float s;
			if (sin(phi) == 0.0f)
				s = -m[2][0] / cos(phi) >= 0.0f ? 1.0f : -1.0f;
			else
				s = m[2][1] / sin(phi) >= 0.0f ? 1.0f : -1.0f;
			theta = atan2(s * abssintheta, m[2][2]);
		}
		else
		{
			psi = 0.0f;
			if (m[2][2] > 0.0f)
			{
				theta = 0.0f;
				phi = atan2(m[0][1], m[0][0]);
			}
			else
			{
				theta = PI;
				phi = atan2(m[0][1], m[0][0]);
			}
		}

		return tfloat3(phi, theta, psi);
	}

	tfloat3 EulerInverse(tfloat3 angles)
	{
		return tfloat3(-angles.z, -angles.y, -angles.x);
	}

	float EulerCompare(tfloat3 angles1, tfloat3 angles2)
	{
		glm::mat3 m1 = Matrix3Euler(angles1);
		glm::mat3 m2 = Matrix3Euler(angles2);

		glm::vec3 v1 = glm::normalize(m1 * glm::vec3(1, 1, 1));
		glm::vec3 v2 = glm::normalize(m2 * glm::vec3(1, 1, 1));

		return glm::dot(v1, v2);
	}

	glm::vec3 ViewVectorFromPolar(tfloat3 polar)
	{
		glm::mat3 mforvec = Matrix3RotationZ(polar.z) * Matrix3RotationY(polar.y) * Matrix3RotationX(polar.x);
		glm::vec3 v = mforvec * glm::vec3(0, 0, 1);

		return v;
	}

	glm::mat3 Matrix3PolarViewVector(tfloat3 polar, tfloat alpha)
	{
		glm::vec3 v = ViewVectorFromPolar(polar);
		tfloat sa = sin(alpha), ca = cos(alpha);

		glm::mat3 rotation = glm::mat3(ca + v.x * v.x * (1.0f - ca),
			v.y * v.x * (1.0f - ca) + v.z * sa,
			v.z * v.x * (1.0f - ca) - v.y * sa,

			v.x * v.y * (1.0f - ca) - v.z * sa,
			ca + v.y * v.y * (1.0f - ca),
			v.z * v.y * (1.0f - ca) + v.x * sa,

			v.x * v.z * (1.0f - ca) + v.y * sa,
			v.y * v.z * (1.0f - ca) - v.x * sa,
			ca + v.z * v.z * (1.0f - ca));

		return rotation;
	}

	glm::mat4 Matrix4PolarViewVector(tfloat3 vector, tfloat alpha)
	{
		glm::mat3 mforvec = Matrix3RotationZ(vector.z) * Matrix3RotationY(vector.y) * Matrix3RotationX(vector.x);
		glm::vec3 v = mforvec * glm::vec3(0, 0, 1);
		tfloat sa = sin(alpha), ca = cos(alpha);

		glm::mat4 rotation = glm::mat4(ca + v.x * v.x * (1.0f - ca),
			v.y * v.x * (1.0f - ca) + v.z * sa,
			v.z * v.x * (1.0f - ca) - v.y * sa,
			0.0f,

			v.x * v.y * (1.0f - ca) - v.z * sa,
			ca + v.y * v.y * (1.0f - ca),
			v.z * v.y * (1.0f - ca) + v.x * sa,
			0.0f,

			v.x * v.z * (1.0f - ca) + v.y * sa,
			v.y * v.z * (1.0f - ca) - v.x * sa,
			ca + v.z * v.z * (1.0f - ca),
			0.0f,

			0.0f,
			0.0f,
			0.0f,
			1.0f);

		return rotation;
	}

	tfloat3 EulerFromViewVector(glm::vec3 view)
	{
		tfloat theta = acos(view.z);
		tfloat phi = atan2(view.y, view.x);

		return tfloat3(phi, theta, (tfloat)0);
	}

	tfloat3 EulerFromPolarViewVector(tfloat3 polar, tfloat alpha)
	{
		glm::vec3 view = ViewVectorFromPolar(polar);
		tfloat3 euler = EulerFromViewVector(view);
		euler.z = alpha;

		return euler;
	}

	tfloat3 PolarViewVectorFromEuler(tfloat3 euler)
	{
		return tfloat3(0.0f, euler.y, euler.z);
	}

	glm::mat4 Matrix4Translation(tfloat3 translation)
	{
		return glm::mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, translation.x, translation.y, translation.z, 1);
	}

	glm::mat4 Matrix4Scale(tfloat3 scale)
	{
		return glm::mat4(scale.x, 0, 0, 0, 0, scale.y, 0, 0, 0, 0, scale.z, 0, 0, 0, 0, 1);
	}

	glm::mat4 Matrix4RotationX(tfloat angle)
	{
		double c = cos(angle);
		double s = sin(angle);

		return glm::mat4(1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1);
	}

	glm::mat4 Matrix4RotationY(tfloat angle)
	{
		double c = cos(angle);
		double s = sin(angle);

		return glm::mat4(c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1);
	}

	glm::mat4 Matrix4RotationZ(tfloat angle)
	{
		double c = cos(angle);
		double s = sin(angle);

		return glm::mat4(c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	}

	glm::mat3 Matrix3Translation(tfloat2 translation)
	{
		return glm::mat3(1, 0, 0, 0, 1, 0, translation.x, translation.y, 1);
	}

	glm::mat3 Matrix3Scale(tfloat3 scale)
	{
		return glm::mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
	}

	glm::mat3 Matrix3RotationX(tfloat angle)
	{
		double c = cos(angle);
		double s = sin(angle);

		return glm::mat3(1, 0, 0, 0, c, s, 0, -s, c);
	}

	glm::mat3 Matrix3RotationY(tfloat angle)
	{
		double c = cos(angle);
		double s = sin(angle);

		return glm::mat3(c, 0, -s, 0, 1, 0, s, 0, c);
	}

	glm::mat3 Matrix3RotationZ(tfloat angle)
	{
		double c = cos(angle);
		double s = sin(angle);

		return glm::mat3(c, s, 0, -s, c, 0, 0, 0, 1);
	}

	glm::mat3 Matrix3RotationCustom(tfloat3 axis, tfloat angle)
	{
		double c = cos(angle);
		double c1 = 1.0 - c;
		double s = sin(angle);

		return glm::mat3(
			c + axis.x + axis.x * c1,
			axis.y * axis.x * c1 + axis.z * s,
			axis.z * axis.x * c1 - axis.y * s,

			axis.x * axis.y * c1 - axis.z * s,
			c + axis.y * axis.y * c1,
			axis.z * axis.y * c1 + axis.x * s,

			axis.x * axis.z * c1 + axis.y * s,
			axis.y * axis.z * c1 - axis.x * s,
			c + axis.z * axis.z * c1
			);
	}

	glm::mat3 Matrix3RotationInPlaneAxis(tfloat axisangle, tfloat rotationangle)
	{
		return Matrix3RotationCustom(tfloat3(cos(axisangle), sin(axisangle), 0.0), rotationangle);
	}

	glm::mat2 Matrix2Scale(tfloat2 scale)
	{
		return glm::mat2(scale.x, 0, 0, scale.y);
	}

	glm::mat2 Matrix2Rotation(tfloat angle)
	{
		double c = cos(angle);
		double s = sin(angle);

		return glm::mat2(c, s, -s, c);
	}
}