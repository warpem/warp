#include "cufft.h"
#include "Prerequisites.cuh"
#include "Angles.cuh"

#ifndef TRANSFORMATION_CUH
#define TRANSFORMATION_CUH


namespace gtom
{
	//////////////////
	//Transformation//
	//////////////////

	//Bin.cu:
	void d_Bin(tfloat* d_input, tfloat* d_output, int3 dims, int bincount, int batch = 1);

	//Combined.cu:
	void d_ScaleRotateShift2D(tfloat* d_input, tfloat* d_output, int2 dims, tfloat2* h_scales, tfloat* h_angles, tfloat2* h_shifts, T_INTERP_MODE mode, bool outputzerocentered, int batch);
	void d_ScaleRotateShiftCubic2D(cudaTex t_input, tfloat* d_output, int2 dims, tfloat2 scale, tfloat angle, tfloat2 shift, bool outputzerocentered);

	//Coordinates.cu:
	void d_Cart2Polar(tfloat* d_input, tfloat* d_output, int2 dims, T_INTERP_MODE mode, uint innerradius = 0, uint exclusiveouterradius = 0, int batch = 1);
	int2 GetCart2PolarSize(int2 dims);
	void d_Cart2PolarFFT(tfloat* d_input, tfloat* d_output, int2 dims, T_INTERP_MODE mode, uint innerradius = 0, uint exclusiveouterradius = 0, int batch = 1);
	int2 GetCart2PolarFFTSize(int2 dims);
	uint GetCart2PolarNonredundantSize(int2 dims);
	uint GetCart2PolarNonredundantSize(int2 dims, int maskinner, int maskouter);
	float2* GetPolarNonredundantCoords(int2 dims);
	float2* GetPolarNonredundantCoords(int2 dims, int maskinner, int maskouter);
	uint GetCart2PolarFFTNonredundantSize(int2 dims);
	uint GetCart2PolarFFTNonredundantSize(int2 dims, int maskinner, int maskouter);
	float2* GetPolarFFTNonredundantCoords(int2 dims);
	float2* GetPolarFFTNonredundantCoords(int2 dims, int maskinner, int maskouter);

	//FitMagAnisotropy.cu:
	void d_FitMagAnisotropy(tfloat* h_image, int2 dimsimage, float compareradius, float maxdistortion, float distortionstep, float anglestep, float &bestdistortion, float &bestangle);

	//FFTLines.cu:
	void d_FFTLines(tcomplex* d_input, tcomplex* d_output, int2 dims, T_INTERP_MODE mode, int anglesteps, int linewidth, int batch);

	//MagAnisotropy.cu:
	void d_MagAnisotropyCorrect(tfloat* d_image, int2 dimsimage, tfloat* d_scaledimage, int2 dimsscaled, float majorpixel, float minorpixel, float majorangle, uint supersample, uint batch = 1);

	//Rotation.cu:
	void d_Rotate2D(tfloat* d_input, tfloat* d_output, int2 dims, tfloat* angles, T_INTERP_MODE mode, bool isoutputzerocentered, uint batch = 1);
	void d_Rotate2D(cudaTex* t_input, tfloat* d_output, int2 dims, tfloat* h_angles, T_INTERP_MODE mode, bool isoutputzerocentered, uint batch = 1);
	void d_Rotate3D(tfloat* d_volume, tfloat* d_output, int3 dims, tfloat3* h_angles, uint nangles, T_INTERP_MODE mode, bool outputzerocentered);
	void d_Rotate3D(cudaTex t_volume, tfloat* d_output, int3 dims, tfloat3* h_angles, uint nangles, T_INTERP_MODE mode, bool outputzerocentered);
    void d_Rotate3DExtractAt(cudaTex t_volume, int3 dimsvolume, tfloat* d_proj, int3 dimsproj, tfloat3* h_angles, tfloat3* h_positions, T_INTERP_MODE mode, uint batch);
    void d_Rotate3DExtractAt(cudaTex t_volume, int3 dimsvolume, tfloat* d_proj, int3 dimsproj, glm::mat3* d_matrices, tfloat3* d_positions, T_INTERP_MODE mode, uint batch);
	void d_Rotate2DFT(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat* angles, tfloat maxfreq, T_INTERP_MODE mode, bool isoutputzerocentered, int batch = 1);
	void d_Rotate2DFT(cudaTex t_inputRe, cudaTex t_inputIm, tcomplex* d_output, int3 dims, tfloat angle, tfloat maxfreq, T_INTERP_MODE mode, bool isoutputzerocentered);
	void d_Rotate3DFT(tcomplex* d_volume, tcomplex* d_output, int3 dims, tfloat3* h_angles, int nangles, T_INTERP_MODE mode, bool outputzerocentered);
	void d_Rotate3DFT(cudaTex t_Re, cudaTex t_Im, tcomplex* d_output, int3 dims, tfloat3* h_angles, int nangles, T_INTERP_MODE mode, bool outputzerocentered);
	void d_Rotate3DFT(tfloat* d_volume, tfloat* d_output, int3 dims, tfloat3* h_angles, int nangles, T_INTERP_MODE mode, bool outputzerocentered);
	void d_Rotate3DFT(cudaTex t_volume, tfloat* d_output, int3 dims, tfloat3* h_angles, int nangles, T_INTERP_MODE mode, bool outputzerocentered);

	//Shift.cu:
	void d_Shift(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* h_delta, cufftHandle* planforw = NULL, cufftHandle* planback = NULL, tcomplex* d_sharedintermediate = NULL, int batch = 1);
	void d_Shift(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* h_delta, bool iszerocentered = false, int batch = 1);
	void d_MotionBlur(tfloat* d_output, int3 dims, float3* h_shifts, uint nshifts, bool iszerocentered, uint batch = 1);

	//Scale.cu:
	void d_Scale(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, T_INTERP_MODE mode, cufftHandle* planforw = NULL, cufftHandle* planback = NULL, int batch = 1, tcomplex* d_inputfft = NULL, tcomplex* d_outputfft = NULL);

	//Warp2D.cu:
	void d_Warp2D(tfloat* d_image, int2 dimsimage, tfloat2* d_grid, int2 dimsgrid, tfloat* d_output, uint batch = 1);
	void d_Warp2D(cudaTex* dt_image, int2 dimsimage, cudaTex* dt_gridx, cudaTex* dt_gridy, int2 dimsgrid, tfloat* d_output, uint batch = 1);
	void d_Warp2D(cudaTex dt_image, int2 dimsimage, cudaTex dt_gridx, cudaTex dt_gridy, int2 dimsgrid, tfloat* d_output);
}
#endif