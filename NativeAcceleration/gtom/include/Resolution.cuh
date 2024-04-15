#include "cufft.h"
#include "Prerequisites.cuh"

#ifndef RESOLUTION_CUH
#define RESOLUTION_CUH

namespace gtom
{
	//////////////
	//Resolution//
	//////////////

	enum T_FSC_MODE
	{
		T_FSC_THRESHOLD = 0,
		T_FSC_FIRSTMIN = 1
	};

	//FSC.cu:
	void d_FSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_curve, int maxradius, cufftHandle* plan, int batch = 1);
	void d_FSC(tcomplex* d_volumeft1,
		tcomplex* d_volumeft2,
		int3 dimsvolume,
		tfloat* d_curve,
		int maxradius,
		tfloat* d_outnumerators = NULL,
		tfloat* d_outdenominators1 = NULL,
		tfloat* d_outdenominators2 = NULL,
		int batch = 1);

	//LocalFilter.cu:
	void d_LocalFilter(tfloat* d_input,
						tfloat* d_filtered,
						int3 dimsvolume,
						tfloat* d_resolution,
						int windowsize,
						tfloat angpix,
						tfloat* d_filterramps,
						int rampsoversample);

	//LocalFSC.cu:
	void d_LocalFSC(tfloat* d_volume1,
					tfloat* d_volume2,
					tfloat* d_volumemask,
					int3 dimsvolume,
					tfloat* d_resolution,
					int windowsize,
					int spacing,
					tfloat fscthreshold,
					tfloat angpix,
					tfloat* d_avgfsc,
					tfloat* d_avgamps,
					tfloat* d_avgsamples,
					int avgoversample,
					tfloat* h_globalfsc);

	//LocalFSCBfac.cu:
	void d_LocalFSCBfac(tfloat* d_volume1,
						tfloat* d_volume2,
						int3 dimsvolume,
						tfloat* d_resolution,
						tfloat* d_bfactors,
						tfloat* d_corrected,
						tfloat* d_unsharpened,
						int windowsize,
						tfloat fscthreshold,
						bool dolocalbfac,
						tfloat globalbfac,
						tfloat minresbfac,
						tfloat angpix,
						tfloat minbfac,
						tfloat bfacbias,
						tfloat mtfslope,
						bool doanisotropy,
						bool dofilterhalfmaps);

	//AnisotropicFSC:
	void d_AnisotropicFSC(tcomplex* d_volumeft1,
							tcomplex* d_volumeft2,
							int3 dimsvolume,
							tfloat* d_curve,
							int maxradius,
							float3 direction,
							float coneangle,
							tfloat* d_outnumerators = NULL,
							tfloat* d_outdenominators1 = NULL,
							tfloat* d_outdenominators2 = NULL,
							int batch = 1);
	void d_AnisotropicFSCMap(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_map, int2 anglesteps, int maxradius, T_FSC_MODE fscmode, tfloat threshold, cufftHandle* plan, int batch);

	//LocalAnisotropicFSC>
	void d_LocalAnisotropicFSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, uint nvolumes, std::vector<float3> v_directions, float coneangle, tfloat* d_resolution, int windowsize, int maxradius, tfloat threshold);
}
#endif