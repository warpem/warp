#include "cufft.h"
#include "Angles.cuh"
#include "Prerequisites.cuh"

#ifndef PROJECTION_CUH
#define PROJECTION_CUH

namespace gtom
{
	//////////////
	//Projection//
	//////////////

	//Backward.cu:
	void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, bool outputzerocentered, int batch);

	//Forward.cu:
	void d_ProjForward(tfloat* d_volume, tfloat* d_volumepsf, int3 dimsvolume, tfloat* d_projections, tfloat* d_projectionspsf, tfloat3* h_angles, tfloat2* h_shifts, T_INTERP_MODE mode, int batch);
	void d_ProjForward(tcomplex* d_volumeft, tfloat* d_volumepsf, int3 dimsvolume, tcomplex* d_projectionsft, tfloat* d_projectionspsf, tfloat3* h_angles, tfloat2* h_shifts, T_INTERP_MODE mode, bool outputzerocentered, int batch);
	void d_ProjForwardRaytrace(tfloat* d_volume, int3 dimsvolume, tfloat3 volumeoffset, tfloat* d_projections, int2 dimsproj, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int supersample, int batch);

	//PseudoAtoms.cu:
	void d_ProjNMAPseudoAtoms(float3* d_positions,
								float* d_intensities,
								uint natoms,
								int3 dimsvol,
								float sigma,
								uint kernelextent,
								float* h_blobvalues,
								float blobsampling,
								uint nblobvalues,
								float3* d_normalmodes,
								float* d_normalmodefactors,
								uint nmodes,
								float3* h_angles,
								float2* h_offsets,
								float scale,
								float* d_proj,
								int2 dimsproj,
								uint batch);

	void d_ProjSoftPseudoAtoms(float3* d_positions,
								float* d_intensities,
								uint natoms,
								int3 dimsvol,
								float sigma,
								uint kernelextent,
								float3* d_coarsedeltas,
								float* d_coarseweights,
								int* d_coarseneighbors,
								uint ncoarseneighbors,
								uint ncoarse,
								float3* h_angles,
								float2* h_offsets,
								float scale,
								float* d_proj,
								int2 dimsproj,
								uint batch);

	//RaySum.cu:
	void d_RaySum(cudaTex t_volume, glm::vec3* d_start, glm::vec3* d_finish, tfloat* d_sums, T_INTERP_MODE mode, uint supersample, uint batch);
}
#endif