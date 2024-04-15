#include "Prerequisites.cuh"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_INLINE
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"

#ifndef ANGLES_CUH
#define ANGLES_CUH

namespace gtom
{
	glm::mat4 Matrix4Euler(tfloat3 angles);
	glm::mat4 Matrix4EulerLegacy(tfloat2 angles);
	glm::mat3 Matrix3Euler(tfloat3 angles);

	tfloat3 EulerFromMatrix(glm::mat4 m);
	tfloat3 EulerFromMatrix(glm::mat3 m);

	tfloat3 EulerInverse(tfloat3 angles);
	tfloat EulerCompare(tfloat3 angles1, tfloat3 angles2);

	glm::vec3 ViewVectorFromPolar(tfloat3 polar);
	glm::mat3 Matrix3PolarViewVector(tfloat3 vector, tfloat alpha);
	glm::mat4 Matrix4PolarViewVector(tfloat3 vector, tfloat alpha);
	tfloat3 EulerFromViewVector(glm::vec3 view);
	tfloat3 EulerFromPolarViewVector(tfloat3 polar, tfloat alpha);
	tfloat3 PolarViewVectorFromEuler(tfloat3 euler);

	glm::mat4 Matrix4Translation(tfloat3 translation);
	glm::mat4 Matrix4Scale(tfloat3 scale);
	glm::mat4 Matrix4RotationX(tfloat angle);
	glm::mat4 Matrix4RotationY(tfloat angle);
	glm::mat4 Matrix4RotationZ(tfloat angle);

	glm::mat3 Matrix3Scale(tfloat3 scale);
	glm::mat3 Matrix3RotationX(tfloat angle);
	glm::mat3 Matrix3RotationY(tfloat angle);
	glm::mat3 Matrix3RotationZ(tfloat angle);
	glm::mat3 Matrix3RotationCustom(tfloat3 axis, tfloat angle);
	glm::mat3 Matrix3RotationInPlaneAxis(tfloat axisangle, tfloat rotationangle);

	glm::mat3 Matrix3Translation(tfloat2 translation);
	glm::mat2 Matrix2Scale(tfloat2 scale);
	glm::mat2 Matrix2Rotation(tfloat angle);

	tfloat3* GetEqualAngularSpacing(tfloat2 phirange, tfloat2 thetarange, tfloat2 psirange, tfloat increment, int &numangles);
	std::vector<float3> GetEqualAngularSpacing(float2 phirange, float2 thetarange, float2 psirange, float increment);
}
#endif