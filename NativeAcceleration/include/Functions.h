#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "gtom/include/GTOM.cuh"

using namespace std;

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>

#ifndef _WIN32
    #define dllexport
    #define __declspec(x)
    #define __stdcall
#endif

// Bessel.cpp:
extern "C" __declspec(dllexport) float __stdcall KaiserBessel(float r, float a, float alpha, int m);
extern "C" __declspec(dllexport) float __stdcall KaiserBesselFT(float w, float a, float alpha, int m);
extern "C" __declspec(dllexport) float __stdcall KaiserBesselProj(float r, float a, float alpha, int m);

// Correlation.cpp:

extern "C" __declspec(dllexport) void CorrelateSubTomos(unsigned long long t_projectordataRe,
                                                        unsigned long long t_projectordataIm,
                                                        float projectoroversample,
                                                        int3 dimsprojector,
                                                        float2* d_experimentalft,
                                                        float* d_ctf,
                                                        int3 dimsvolume,
                                                        uint nvolumes,
                                                        float3* h_angles,
                                                        uint nangles,
                                                        uint batchangles,
                                                        float maskradius,
                                                        float* d_bestcorrelation,
                                                        int* d_bestangle,
                                                        float* h_progressfraction);

extern "C" __declspec(dllexport) int* LocalPeaks(float* d_input, int* h_peaksnum, int3 dims, int localextent, float threshold);
extern "C" __declspec(dllexport) void SubpixelMax(float* d_input, float* d_output, int3 dims, int subpixsteps);

extern "C" __declspec(dllexport) void PeakOne2D(float* d_input, float3* d_positions, float* d_values, int2 dims, int2 dimsregion, bool subtractcenter, int batch);

extern "C" __declspec(dllexport) void CorrelateRealspace(float* d_image1, float* d_image2, int3 dims, float* d_mask, float* d_corr, uint batch);

// CTF.cu:
extern "C" __declspec(dllexport) void CreateSpectra(float* d_frame,
													int2 dimsframe,
													int nframes,
													int3* h_origins,
													int norigins,
													int2 dimsregion,
													int3 ctfgrid,
                                                    int2 dimsregionscaled,
													float* d_outputall,
													float* d_outputmean,
                                                    cufftHandle planforw,
                                                    cufftHandle planback);

extern "C" __declspec(dllexport) void CTFMakeAverage(float* d_ps, 
													 float2* d_pscoords, 
													 uint length, 
													 uint sidelength, 
													 gtom::CTFParams* h_sourceparams, 
													 gtom::CTFParams targetparams, 
													 uint minbin, 
													 uint maxbin, 
													 uint batch, 
													 float* d_output);

extern "C" __declspec(dllexport) void CTFCompareToSim(float* d_ps, 
                                                      float2* d_pscoords, 
                                                      float* d_scale, 
                                                      uint length, 
                                                      gtom::CTFParams* h_sourceparams, 
                                                      float* h_scores, 
                                                      uint batch);

// CubicGPU.cu:
extern "C" __declspec(dllexport) void __stdcall CubicGPUInterpIrregular(unsigned long long t_input, 
                                                                        int3 dimsgrid,
                                                                        float3* h_positions, 
                                                                        int npositions, 
                                                                        float* h_output);

// einspline.cpp:
extern "C" __declspec(dllexport) void* __stdcall CreateEinspline3(float* h_values, int3 dims, float3 margins);
extern "C" __declspec(dllexport) void* __stdcall CreateEinspline2(float* h_values, int2 dims, float2 margins);
extern "C" __declspec(dllexport) void* __stdcall CreateEinspline1(float* h_values, int dims, float margins);
extern "C" __declspec(dllexport) void __stdcall EvalEinspline3(void* spline, float3* h_pos, int npos, float* h_output);
extern "C" __declspec(dllexport) void __stdcall EvalEinspline2XY(void* spline, float3* h_pos, int npos, float* h_output);
extern "C" __declspec(dllexport) void __stdcall EvalEinspline2XZ(void* spline, float3* h_pos, int npos, float* h_output);
extern "C" __declspec(dllexport) void __stdcall EvalEinspline2YZ(void* spline, float3* h_pos, int npos, float* h_output);
extern "C" __declspec(dllexport) void __stdcall EvalEinspline1(void* spline, float* h_pos, int npos, float* h_output);
extern "C" __declspec(dllexport) void __stdcall EvalEinspline1X(void* spline, float3* h_pos, int npos, float* h_output);
extern "C" __declspec(dllexport) void __stdcall EvalEinspline1Y(void* spline, float3* h_pos, int npos, float* h_output);
extern "C" __declspec(dllexport) void __stdcall EvalEinspline1Z(void* spline, float3* h_pos, int npos, float* h_output);
extern "C" __declspec(dllexport) void __stdcall DestroyEinspline(void* spline);
extern "C" __declspec(dllexport) void __stdcall EvalLinear4Batch(const int4 dims, const float* values, const float4 * h_pos, const int npos, float* h_output);
extern "C" __declspec(dllexport) float __stdcall EvalLinear4(const int4 dims, const float* values, float4 coords);

// ParticleMultibody.cu:
extern "C" __declspec(dllexport) void ParticleMultibodyGetDiff(float2* d_experimental,
                                                                float2* d_reference,
                                                                float2* d_shiftfactors,
                                                                float* d_ctf,
                                                                float* d_invsigma2,
                                                                int2 dims,
                                                                float2* h_shifts,
                                                                float* h_diff,
                                                                uint nparticles,
                                                                uint nbodies);

extern "C" __declspec(dllexport) void ParticleMultibodyProject(unsigned long long* h_textureRe, 
                                                                unsigned long long* h_textureIm,
                                                                int3 dimsvolume, 
                                                                float2* d_proj, 
                                                                int2 dimsproj, 
                                                                float3* h_angles, 
                                                                float2* h_shifts, 
                                                                float* h_globalweights, 
                                                                float supersample, 
                                                                uint nbodies, 
                                                                uint batch);

// ParticleNMA.cu:
extern "C" __declspec(dllexport) void ParticleNMAGetDiff(float2* d_experimental,
                                                        float2* d_reference,
                                                        float* d_ctf,
                                                        float* d_invsigma2,
                                                        int2 dims,
                                                        float* h_diff,
                                                        uint nparticles);

extern "C" __declspec(dllexport) void ParticleNMAGetMeanDisplacement(float3* d_normalmodes, 
                                                                    uint natoms, 
                                                                    float* h_modefactors, 
                                                                    uint nmodes, 
                                                                    uint nmodels, 
                                                                    float* h_displacement);

extern "C" __declspec(dllexport) void ParticleNMAGetRigidTransform(float3* d_positions, 
                                                                    float3* d_normalmodes, 
                                                                    uint natoms, 
                                                                    float* h_modefactors, 
                                                                    uint nmodes, 
                                                                    uint nmodels, 
                                                                    float3* h_centertrans, 
                                                                    float* h_rotmat);

// Angles.cpp:
extern "C" __declspec(dllexport) int __stdcall GetAnglesCount(int healpixorder, char* c_symmetry, float limittilt);
extern "C" __declspec(dllexport) void __stdcall GetAngles(float3* h_angles, int healpixorder, char* c_symmetry, float limittilt);

// Backprojection.cu:
extern "C" __declspec(dllexport) void __stdcall ProjBackwardOne(float* d_volume, int3 dimsvolume, float* d_image, int2 dimsimage, float* h_angles, float* h_offsets, bool outputzerocentered, bool sliceonly, bool halfonly, int batch);

// BoxNet2.cu:
extern "C" __declspec(dllexport) void __stdcall BoxNet2Augment(float* d_inputmic,
                                                                float* d_inputlabel,
                                                                int2 dimsinput,
                                                                float* d_outputmic,
                                                                float* d_outputlabel,
                                                                int2 dimsoutput,
                                                                float2* h_offsets,
                                                                float* h_rotations,
                                                                float3* h_scales,
															    float offsetmean,
															    float offsetscale,
                                                                float noisestddev,
                                                                int seed,
                                                                bool channelsfirst,
                                                                uint batch);

// C2DNet.cu:
extern "C" __declspec(dllexport) void C2DNetAlign(float2 * d_refs,
                                                int dimprojector,
                                                float supersample,
                                                float2 * d_dataft,
                                                float* d_data,
                                                float* d_ctf,
                                                int dimdata,
                                                float3 * d_initposes,
                                                int nposesperdata,
                                                int initshell,
                                                int maxshell,
                                                int niters,
                                                float anglestep,
                                                float shiftstep,
                                                int ntop,
                                                int batch,
                                                float* d_aligneddata,
                                                float* d_alignedctf);

// CubeNet.cu:
extern "C" __declspec(dllexport) void CubeNetAugment(float* d_inputvol,
													float* d_inputlabel,
													int3 dimsinput,
													float* d_outputmic,
													float* d_outputlabel,
													int nclasses,
													int3 dimsoutput,
													float3* h_offsets,
													float3* h_rotations,
													float3* h_scales,
													float noisestddev,
													int seed,
													uint batch);

// Deconv.cu:
extern "C" __declspec(dllexport) void DeconvolveCTF(float2* d_inputft, float2* d_outputft, int3 dims, gtom::CTFParams ctfparams, float strength, float falloff, float highpassnyquist);

// Defects.cu:
extern "C" __declspec(dllexport) void CorrectDefects(float* d_input, float* d_output, int3 * d_locations, int* d_neighbors, int ndefects);

// Device.cpp:

extern "C" __declspec(dllexport) int __stdcall GetDeviceCount();
extern "C" __declspec(dllexport) void __stdcall SetDevice(int device);
extern "C" __declspec(dllexport) int __stdcall GetDevice();
extern "C" __declspec(dllexport) long long __stdcall GetFreeMemory(int device);
extern "C" __declspec(dllexport) long long __stdcall GetTotalMemory(int device);
extern "C" __declspec(dllexport) char* __stdcall GetDeviceName(int device);
extern "C" __declspec(dllexport) void __stdcall DeviceSynchronize();

// DFT.cu:
extern "C" __declspec(dllexport) void BenchmarkCUFFT(float* d_input, float2 * d_output, cufftHandle planforw, int elements, int batch, int repeats);
extern "C" __declspec(dllexport) void BenchmarkDFT(float* d_input, float2 * d_sincos, float2* d_output, int elements, int batch, int repeats);
extern "C" __declspec(dllexport) void BenchmarkBLASFT(void* cublas, float2 * d_input, float2 * d_sincos, float2 * d_output, int elements, int batch, int repeats);

// FFT.cpp:
extern "C" __declspec(dllexport) void __stdcall FFT_CPU(float* data, float* result, int3 dims, int nthreads);
extern "C" __declspec(dllexport) void __stdcall IFFT_CPU(float* data, float* result, int3 dims, int nthreads);

// Float16.cpp:

extern "C" __declspec(dllexport) void __stdcall FloatToHalfAVX2(const float* src, uint16_t * dst, size_t count);
extern "C" __declspec(dllexport) void __stdcall HalfToFloatAVX2(const uint16_t * src, float* dst, size_t count);
extern "C" __declspec(dllexport) void __stdcall FloatToHalfScalars(const float* src, uint16_t * dst, size_t count);
extern "C" __declspec(dllexport) void __stdcall HalfToFloatScalars(const uint16_t * src, float* dst, size_t count);

// FSC.cpp:
extern "C" __declspec(dllexport) void __stdcall ConicalFSC(float2* volume1ft, 
															float2* volume2ft, 
															int3 dims, 
															float3* directions, 
															int ndirections, 
															float anglestep, 
															int minshell, 
															float threshold, 
															float particlefraction, 
															float* result);

// Helical.cu:
extern "C" __declspec(dllexport) void __stdcall HelicalSymmetrize(unsigned long long tcpf_volume,
																  float* d_output,
																  int3 dims,
																  float twist,
																  float rise,
																  float maxz,
																  float maxr);

// Memory.cpp:

extern "C" __declspec(dllexport) float* __stdcall MallocDevice(long long elements);
extern "C" __declspec(dllexport) float* __stdcall MallocDeviceFromHost(float* h_data, long long elements);
extern "C" __declspec(dllexport) int* __stdcall MallocDeviceFromHostInt(int* h_data, long long elements);
extern "C" __declspec(dllexport) void* __stdcall MallocDeviceHalf(long long elements);
extern "C" __declspec(dllexport) void* __stdcall MallocDeviceHalfFromHost(float* h_data, long long elements);

extern "C" __declspec(dllexport) void __stdcall FreeDevice(void* d_data);

extern "C" __declspec(dllexport) void __stdcall CopyDeviceToHost(float* d_source, float* h_dest, long long elements);
extern "C" __declspec(dllexport) void __stdcall CopyDeviceToHostPinned(float* d_source, float* hp_dest, long long elements);
extern "C" __declspec(dllexport) void __stdcall CopyDeviceHalfToHost(half* d_source, float* h_dest, long long elements);
extern "C" __declspec(dllexport) void __stdcall CopyDeviceToDevice(float* d_source, float* d_dest, long long elements);
extern "C" __declspec(dllexport) void __stdcall CopyDeviceHalfToDeviceHalf(half* d_source, half* d_dest, long long elements);
extern "C" __declspec(dllexport) void __stdcall CopyHostToDevice(float* h_source, float* d_dest, long long elements);
extern "C" __declspec(dllexport) void __stdcall CopyHostToHost(float* h_source, float* h_dest, long long elements);
extern "C" __declspec(dllexport) void __stdcall CopyHostPinnedToDevice(float* hp_source, float* d_dest, long long elements);
extern "C" __declspec(dllexport) void __stdcall CopyHostToDeviceHalf(float* h_source, half* d_dest, long long elements);

extern "C" __declspec(dllexport) void __stdcall SingleToHalf(float* d_source, half* d_dest, long long elements);
extern "C" __declspec(dllexport) void __stdcall HalfToSingle(half* d_source, float* d_dest, long long elements);

extern "C" __declspec(dllexport) float* __stdcall MallocHostPinned(long long elements);
extern "C" __declspec(dllexport) void __stdcall FreeHostPinned(void* hp_data);

extern "C" __declspec(dllexport) void* __stdcall HostMalloc(long long elements);
extern "C" __declspec(dllexport) void __stdcall HostFree(void* h_pointer);

extern "C" __declspec(dllexport) void __stdcall DeviceReset();

extern "C" __declspec(dllexport) cudaArray_t MallocArray(int2 dims);
extern "C" __declspec(dllexport) void CopyDeviceToArray(float* d_input, cudaArray_t a_output, int2 dims);
extern "C" __declspec(dllexport) void FreeArray(cudaArray_t a_input);

// TiffNative.cpp:

extern "C" __declspec(dllexport) void ReadTIFF(const char* path, int layer, bool flipy, float* h_result);

// EER.cpp:

extern "C" __declspec(dllexport) void ReadEERCombinedFrame(const char* path, int firstFrameInclusive, int lastFrameExclusive, int eer_upsampling, float* h_result);

// Post.cu:

extern "C" __declspec(dllexport) void GetMotionFilter(float* d_output, 
														int3 dims, 
														float3* h_shifts, 
														uint nshifts, 
														uint batch);

extern "C" __declspec(dllexport) void CorrectMagAnisotropy(float* d_image, 
                                                            int2 dimsimage, 
                                                            float* d_scaled, 
                                                            int2 dimsscaled, 
                                                            float majorpixel, 
                                                            float minorpixel, 
                                                            float majorangle, 
                                                            uint supersample, 
                                                            uint batch);

extern "C" __declspec(dllexport) void DoseWeighting(float* d_freq,
                                                    float* d_output,
                                                    uint length,
                                                    float2* h_doserange,
                                                    float3 nikoconst,
                                                    float voltagescaling,
                                                    uint batch);

extern "C" __declspec(dllexport) void NormParticles(float* d_input, float* d_output, int3 dims, uint particleradius, bool flipsign, uint batch);

// Shift.cu:

extern "C" __declspec(dllexport) void CreateShift(float* d_frame,
													int2 dimsframe,
													int nframes,
													int3* h_origins,
													int norigins,
													int2 dimsregion,
													size_t* h_mask,
													uint masklength,
                                                    float2* d_outputall,
													float* d_sigma);

extern "C" __declspec(dllexport) void ShiftGetAverage(float2* d_individual,
                                                        float2* d_average,
                                                        float2* d_shiftfactors,
														uint length,
														uint probelength,
														float2* d_shifts,
														uint nspectra,
														uint nframes);

extern "C" __declspec(dllexport) void ShiftGetDiff(float2* d_individual,
                                                    float2* d_average,
                                                    float2* d_shiftfactors,
													uint length,
													uint probelength,
													float2* d_shifts,
													float* h_diff,
													uint npositions,
													uint nframes);

extern "C" __declspec(dllexport) void ShiftGetGrad(float2* d_individual,
                                                    float2* d_average,
                                                    float2* d_shiftfactors,
													uint length,
													uint probelength,
													float2* d_shifts,
													float2* h_grad,
													uint npositions,
													uint nframes);

extern "C" __declspec(dllexport) void CreateMotionBlur(float* d_output, 
                                                       int3 dims, 
                                                       float* h_shifts, 
                                                       uint nshifts, 
                                                       uint batch);

// ParticleSoftBody.cu:
extern "C" __declspec(dllexport) void ParticleSoftBodyDeform(float3* d_initialpositions,
                                                            float3* d_finalpositions,
                                                            uint npositions,
                                                            int* d_neighborids,
                                                            int* d_neighboroffsets,
                                                            float* d_edgelengths,
                                                            int* d_connectedactuator,
                                                            float3* d_actuatordeltas,
                                                            uint nactuators,
                                                            uint niterations,
                                                            uint nrelaxations,
                                                            uint batch);

// Projector.cpp:
extern "C" __declspec(dllexport) void InitProjector(int3 dims, int oversampling, float* data, float* initialized, int projdim);
extern "C" __declspec(dllexport) void BackprojectorReconstruct(int3 dimsori, int oversampling, float2* d_data, float* d_weights, char* c_symmetry, bool do_reconstruct_ctf, float* h_reconstruction);
extern "C" __declspec(dllexport) void __stdcall BackprojectorReconstructGPU(int3 dimsori,
																			int3 dimspadded,
																			int oversampling,
																			float2* d_dataft,
																			float* d_weights,
																			char* c_symmetry,
																			int helix_units,
																			float helix_twist,
																			float helix_rise,
																			bool do_reconstruct_ctf,
																			float* d_result,
																			cufftHandle pre_planforw,
																			cufftHandle pre_planback,
																			cufftHandle pre_planforwctf,
																			int griddingiterations,
																			int nvolumes);

// Raycast.cu:
extern "C" __declspec(dllexport) void RenderVolume(unsigned long long t_intensities,
                                                    int3 dimsvolume,
                                                    float surfacethreshold,
                                                    unsigned long long t_coloring,
                                                    int3 dimscoloring,
                                                    int2 dimsimage,
                                                    float3 camera,
                                                    float3 pixelx,
                                                    float3 pixely,
                                                    float3 view,
                                                    float3* d_colormap,
                                                    int ncolors,
                                                    float2 colorrange,
                                                    float2 shadingrange,
                                                    float3 intensitycolor,
                                                    float2 intensityrange,
                                                    float3* d_intersectionpoints,
                                                    char* h_hittest,
                                                    uchar4* h_bgra);

// RealspaceProjection.cu:
extern "C" __declspec(dllexport) void RealspaceProjectForward(float* d_volume,
																int3 dimsvolume,
																float* d_projections,
																int2 dimsproj,
																float supersample,
																float3* h_angles,
																int batch);

extern "C" __declspec(dllexport) void RealspaceProjectBackward(float* d_volume,
																int3 dimsvolume,
																float* d_projections,
																int2 dimsproj,
																float supersample,
																float3* h_angles,
																bool normalizesamples,
																int batch);

// Symmetry.cpp:
extern "C" __declspec(dllexport) int SymmetryGetNumberOfMatrices(char* c_symmetry);
extern "C" __declspec(dllexport) void SymmetryGetMatrices(char* c_symmetry, float* h_matrices);
extern "C" __declspec(dllexport) void SymmetrizeFT(float2* d_data, int3 dims, char* c_symmetry);

// MPARefine.cu:
extern "C" __declspec(dllexport) void MultiParticleDiff(float3* h_result,
														float2** hp_experimental,
														int dimdata,
														int* h_relevantdims,
														float2* h_shifts,
														float3* h_angles,
														float4 magnification,
														float* d_weights,
														float2* d_phasecorrection,
                                                        float ewaldradius,
                                                        int maxshell,
														unsigned long long* h_volumeRe,
														unsigned long long* h_volumeIm,
                                                        float supersample,
														int dimprojector,
														int* d_subsets,
														int nparticles,
														int ntilts);

extern "C" __declspec(dllexport) void MultiParticleSimulate(float* d_result,
															int2 dimsmic,
															int dimdata,
															float2* h_positions,
															float2* h_shifts,
															float3* h_angles,
															float* h_defoci,
															float* d_weights,
															gtom::CTFParams* h_ctfparams,
															unsigned long long* h_volumeRe,
															unsigned long long* h_volumeIm,
															int dimprojector,
															int* d_subsets,
															int nparticles,
															int ntilts);

extern "C" __declspec(dllexport) void MultiParticleCorr2D(float* d_result2d,
                                                            float* d_result1dparticles,
                                                            float* d_resultphaseresiduals,
                                                            int dimresult,
                                                            float2** hp_experimental,
                                                            float* d_weights,
                                                            int dimdata,
                                                            float scalingfactor,
                                                            int* h_relevantdims,
                                                            float2* h_shifts,
                                                            float3* h_angles,
                                                            float4 magnification,
                                                            float ewaldradius,
                                                            unsigned long long* h_volumeRe,
                                                            unsigned long long* h_volumeIm,
                                                            float supersample,
                                                            int dimprojector,
                                                            int* d_subsets,
                                                            int nparticles,
                                                            int ntilts);

extern "C" __declspec(dllexport) void MultiParticleResidual(float2* d_result,
															float2** hp_experimental,
															int dimdata,
															float2* h_shifts,
															float3* h_angles,
															float4 magnification,
															float* d_weights,
															unsigned long long* h_volumeRe,
															unsigned long long* h_volumeIm,
															int dimprojector,
															int* d_subsets,
															int nparticles,
															int ntilts);

extern "C" __declspec(dllexport) void MultiParticleSumAmplitudes(float* hp_result,
																int dimdata,
																float3* h_angles,
																unsigned long long t_volumeRe,
																unsigned long long t_volumeIm,
																int dimprojector,
																int nparticles);

// PCA.cu:

extern "C" __declspec(dllexport) void PCALeastSq(float* h_result,
											     float2* d_experimental,
											     float* d_ctf,
											     float* d_spectral,
											     int dimdata,
                                                 float rmax,
											     float2* h_shifts,
											     float3* h_angles,
											     float4 magnification,
											     unsigned long long t_volumeRe,
											     unsigned long long t_volumeIm,
											     int dimprojector,
											     int nparticles,
											     int ntilts);


// Tools.cu:

extern "C" __declspec(dllexport) void Extract(float* d_input,
												float* d_output,
												int3 dims,
												int3 dimsregion,
												int3* h_origins,
                                                bool zeropad,
												uint batch);


extern "C" __declspec(dllexport) void ExtractMultisource(void** h_inputs, 
                                                         float* d_output, 
                                                         int3 dims, 
                                                         int3 dimsregion, 
                                                         int3* h_origins, 
                                                         int nsources, 
                                                         uint batch);

extern "C" __declspec(dllexport) void ExtractHalf(float* d_input,
													float* d_output,
													int3 dims,
													int3 dimsregion,
													int3* h_origins,
													uint batch);

extern "C" __declspec(dllexport) void ReduceMean(float* d_input, 
													float* d_output, 
													uint vectorlength, 
													uint nvectors, 
													uint batch);

extern "C" __declspec(dllexport) void ReduceMeanHalf(half* d_input, half* d_output, uint vectorlength, uint nvectors, uint batch);

extern "C" __declspec(dllexport) void ReduceAdd(float* d_input, float* d_output, uint vectorlength, uint nvectors, uint batch);

extern "C" __declspec(dllexport) void ReduceMax(float* d_input, float* d_output, uint vectorlength, uint nvectors, uint batch);

extern "C" __declspec(dllexport) void Normalize(float* d_ps,
												float* d_output,
												uint length,
												uint batch);

extern "C" __declspec(dllexport) void NormalizeMasked(float* d_ps, 
                                                      float* d_output, 
                                                      float* d_mask, 
                                                      uint length, 
                                                      uint batch);

extern "C" __declspec(dllexport) void SphereMask(float* d_input, 
												 float* d_output, 
												 int3 dims, 
												 float radius, 
												 float sigma, 
												 bool decentered, 
												 uint batch);

extern "C" __declspec(dllexport) void CreateCTF(float* d_output,
												float2* d_coords, 
                                                float* d_gammacorrection,
												uint length,
												gtom::CTFParams* h_params,
												bool amplitudesquared,
												uint batch);


extern "C" __declspec(dllexport) void CreateCTFComplex(float* d_output, 
                                                        float2* d_coords, 
                                                        float* d_gammacorrection,
                                                        uint length, 
                                                        gtom::CTFParams* h_params, 
                                                        bool reverse, 
                                                        uint batch);

extern "C" __declspec(dllexport) void CreateCTFEwaldWeights(float* d_output, 
                                                            float2* d_coords, 
                                                            float* d_gammacorrection, 
                                                            float particlediameter, 
                                                            uint length, 
                                                            gtom::CTFParams* h_params, 
                                                            uint batch);

extern "C" __declspec(dllexport) void Resize(float* d_input,
											int3 dimsinput,
											float* d_output,
											int3 dimsoutput,
											uint batch);

extern "C" __declspec(dllexport) void ShiftStack(float* d_input,
												float* d_output,
												int3 dims,
												float* h_shifts,
												uint batch);
extern "C" __declspec(dllexport) void ShiftStackFT(float* d_input, 
                                                    float* d_output, 
                                                    int3 dims, 
                                                    float* h_shifts, 
                                                    uint batch);

extern "C" __declspec(dllexport) void ShiftStackMassive(float* d_input,
                                                        float* d_output,
                                                        int3 dims,
                                                        float* h_shifts,
                                                        uint batch);

extern "C" __declspec(dllexport) void FFT(float* d_input, float2* d_output, int3 dims, uint batch, cufftHandle plan);

extern "C" __declspec(dllexport) void IFFT(float2* d_input, float* d_output, int3 dims, uint batch, cufftHandle plan, bool normalize);

extern "C" __declspec(dllexport) void Pad(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch);

extern "C" __declspec(dllexport) void PadClamped(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch);

extern "C" __declspec(dllexport) void PadFT(float2* d_input, float2* d_output, int3 olddims, int3 newdims, uint batch);

extern "C" __declspec(dllexport) void PadFTRealValued(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch);

extern "C" __declspec(dllexport) void PadFTFull(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch);

extern "C" __declspec(dllexport) void CropFT(float2* d_input, float2* d_output, int3 olddims, int3 newdims, uint batch);

extern "C" __declspec(dllexport) void CropFTRealValued(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch);

extern "C" __declspec(dllexport) void CropFTFull(float* d_input, float* d_output, int3 olddims, int3 newdims, uint batch);

extern "C" __declspec(dllexport) void RemapToFTComplex(float2* d_input, float2* d_output, int3 dims, uint batch);

extern "C" __declspec(dllexport) void RemapToFTFloat(float* d_input, float* d_output, int3 dims, uint batch);

extern "C" __declspec(dllexport) void RemapFromFTComplex(float2* d_input, float2* d_output, int3 dims, uint batch);

extern "C" __declspec(dllexport) void RemapFromFTFloat(float* d_input, float* d_output, int3 dims, uint batch);

extern "C" __declspec(dllexport) void RemapFullToFTFloat(float* d_input, float* d_output, int3 dims, uint batch);

extern "C" __declspec(dllexport) void RemapFullFromFTFloat(float* d_input, float* d_output, int3 dims, uint batch);

extern "C" __declspec(dllexport) void Cart2Polar(float* d_input, float* d_output, int2 dims, uint innerradius, uint exclusiveouterradius, uint batch);

extern "C" __declspec(dllexport) void Cart2PolarFFT(float* d_input, float* d_output, int2 dims, uint innerradius, uint exclusiveouterradius, uint batch);

extern "C" __declspec(dllexport) void Xray(float* d_input, float* d_output, float ndevs, int2 dims, uint batch);

extern "C" __declspec(dllexport) void Sum(float* d_input, float* d_output, uint length, uint batch);

extern "C" __declspec(dllexport) void Abs(float* d_input, float* d_output, size_t length);

extern "C" __declspec(dllexport) void Amplitudes(float2* d_input, float* d_output, size_t length);

extern "C" __declspec(dllexport) void Sign(float* d_input, float* d_output, size_t length);

extern "C" __declspec(dllexport) void Sqrt(float* d_input, float* d_output, size_t length);

extern "C" __declspec(dllexport) void Cos(float* d_input, float* d_output, size_t length);

extern "C" __declspec(dllexport) void Sin(float* d_input, float* d_output, size_t length);

extern "C" __declspec(dllexport) void AddScalar(float* d_input, float summand, float* d_output, size_t elements);

extern "C" __declspec(dllexport) void AddToSlices(float* d_input, float* d_summands, float* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void SubtractFromSlices(float* d_input, float* d_subtrahends, float* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void MultiplySlices(float* d_input, float* d_multiplicators, float* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void DivideSlices(float* d_input, float* d_divisors, float* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void AddToSlicesHalf(half* d_input, half* d_summands, half* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void SubtractFromSlicesHalf(half* d_input, half* d_subtrahends, half* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void MultiplySlicesHalf(half* d_input, half* d_multiplicators, half* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void MultiplyComplexSlicesByScalar(float2* d_input, float* d_multiplicators, float2* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void MultiplyComplexSlicesByComplex(float2* d_input, float2* d_multiplicators, float2* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void MultiplyComplexSlicesByComplexConj(float2* d_input, float2* d_multiplicators, float2* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void DivideComplexSlicesByScalar(float2* d_input, float* d_divisors, float2* d_output, size_t sliceelements, uint slices);

extern "C" __declspec(dllexport) void MultiplyByScalar(float* d_input, float* d_output, float multiplicator, size_t elements);

extern "C" __declspec(dllexport) void MultiplyByScalars(float* d_input, float* d_output, float* d_multiplicators, size_t elements, uint batch);

extern "C" __declspec(dllexport) void Scale(float* d_input, float* d_output, int3 dimsinput, int3 dimsoutput, uint batch, int planforw, int planback, float2* d_inputfft, float2* d_outputfft);

extern "C" __declspec(dllexport) void ProjectForward(float2* d_inputft, float2* d_outputft, int3 dimsinput, int2 dimsoutput, float3* h_angles, float supersample, uint batch);

extern "C" __declspec(dllexport) void ProjectForwardShifted(float2* d_inputft, float2* d_outputft, int3 dimsinput, int2 dimsoutput, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch);

extern "C" __declspec(dllexport) void ProjectForward3D(float2* d_inputft, float2* d_outputft, int3 dimsinput, int3 dimsoutput, float3* h_angles, float supersample, uint batch);

extern "C" __declspec(dllexport) void ProjectForward3DShifted(float2* d_inputft, float2* d_outputft, int3 dimsinput, int3 dimsoutput, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch);

extern "C" __declspec(dllexport) void ProjectForwardTex(unsigned long long t_inputRe, unsigned long long t_inputIm, float2* d_outputft, int3 dimsinput, int2 dimsoutput, float3* h_angles, float supersample, uint batch);

extern "C" __declspec(dllexport) void ProjectForwardShiftedTex(unsigned long long t_inputRe, unsigned long long t_inputIm, float2* d_outputft, int3 dimsinput, int2 dimsoutput, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch);

extern "C" __declspec(dllexport) void ProjectForward3DTex(unsigned long long t_inputRe, unsigned long long t_inputIm, float2* d_outputft, int3 dimsinput, int3 dimsoutput, float3* h_angles, float supersample, uint batch);

extern "C" __declspec(dllexport) void ProjectForward3DShiftedTex(unsigned long long t_inputRe, unsigned long long t_inputIm, float2* d_outputft, int3 dimsinput, int3 dimsoutput, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch);

extern "C" __declspec(dllexport) void ProjectBackward(float2* d_volumeft, float* d_volumeweights, int3 dimsvolume, float2* d_projft, float* d_projweights, int2 dimsproj, int rmax, float3* h_angles, int* h_ivolume, float4 magnification, float ewaldradius, float supersample, bool outputdecentered, bool squareinterpweights, uint batch);

extern "C" __declspec(dllexport) void ProjectBackwardShifted(float2* d_volumeft, float* d_volumeweights, int3 dimsvolume, float2* d_projft, float* d_projweights, int2 dimsproj, int rmax, float3* h_angles, float3* h_shifts, float* h_globalweights, float supersample, uint batch);

extern "C" __declspec(dllexport) void Bandpass(float* d_input, float* d_output, int3 dims, float nyquistlow, float nyquisthigh, float nyquistsoftedge, uint batch);

extern "C" __declspec(dllexport) void FourierBandpass(float2* d_input, int3 dims, float nyquistlow, float nyquisthigh, float nyquistsoftedge, uint batch);

extern "C" __declspec(dllexport) void BandpassGauss(float* d_input, float* d_output, int3 dims, float nyquistlow, float nyquisthigh, float sigma, uint batch);

extern "C" __declspec(dllexport) void FourierBandpassGauss(float2* d_input, int3 dims, float nyquistlow, float nyquisthigh, float sigma, uint batch);

extern "C" __declspec(dllexport) void BandpassButter(float* d_input, float* d_output, int3 dims, float nyquistlow, float nyquisthigh, int order, uint batch);

extern "C" __declspec(dllexport) void FourierBandpassButter(float2 * d_input, int3 dims, float nyquistlow, float nyquisthigh, int order, uint batch);

extern "C" __declspec(dllexport) void LocalStd(float* d_map, int3 dimsmap, float localradius, float* d_std, float* d_mean, cufftHandle planforw, cufftHandle planback);

extern "C" __declspec(dllexport) void Rotate2D(float* d_input, float* d_output, int2 dims, float* h_angles, int oversample, uint batch);

extern "C" __declspec(dllexport) void ShiftAndRotate2D(float* d_input, float* d_output, int2 dims, float2* h_shifts, float* h_angles, uint batch);

extern "C" __declspec(dllexport) void MinScalar(float* d_input, float* d_output, float value, uint elements);

extern "C" __declspec(dllexport) void MaxScalar(float* d_input, float* d_output, float value, uint elements);

extern "C" __declspec(dllexport) void Min(float* d_input1, float* d_input2, float* d_output, uint elements);

extern "C" __declspec(dllexport) void Max(float* d_input1, float* d_input2, float* d_output, uint elements);

extern "C" __declspec(dllexport) int CreateFFTPlan(int3 dims, uint batch);

extern "C" __declspec(dllexport) int CreateIFFTPlan(int3 dims, uint batch);

extern "C" __declspec(dllexport) void DestroyFFTPlan(cufftHandle plan);

extern "C" __declspec(dllexport) void* CreateBLAS();

extern "C" __declspec(dllexport) void DestroyBLAS(void* handle);

extern "C" __declspec(dllexport) void LocalRes(float* d_volume1,
                                                float* d_volume2,
                                                int3 dims,
                                                float pixelsize,
                                                float* d_filtered,
                                                float* d_filteredsharp,
                                                float* d_localres,
                                                float* d_localbfac,
                                                int windowsize,
                                                float fscthreshold,
                                                bool dolocalbfactor,
                                                float minresbfactor,
                                                float globalbfactor,
                                                float mtfslope,
                                                bool doanisotropy,
                                                bool dofilterhalfmaps);

extern "C" __declspec(dllexport) void LocalFSC(float* d_volume1,
												float* d_volume2,
												float* d_volumemask,
												int3 dims,
												int spacing,
												float pixelsize,
												float* d_localres,
												int windowsize,
												float fscthreshold,
												float* d_avgfsc,
												float* d_avgamps,
												float* d_avgsamples,
												int avgoversample,
												float* h_globalfsc);

extern "C" __declspec(dllexport) void LocalFilter(float* d_input,
													float* d_filtered,
													int3 dims,
													float* d_localres,
													int windowsize,
													float angpix,
													float* d_filterramps,
													int rampsoversample);

extern "C" __declspec(dllexport) void ProjectNMAPseudoAtoms(float3* d_positions,
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

extern "C" __declspec(dllexport) void ProjectSoftPseudoAtoms(float3* d_positions,
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

extern "C" __declspec(dllexport) void DistanceMap(float* d_input, float* d_output, int3 dims, int maxiterations);
extern "C" __declspec(dllexport) void DistanceMapExact(float* d_input, float* d_output, int3 dims, int maxdistance);

extern "C" __declspec(dllexport) void PrefilterForCubic(float* d_data, int3 dims);
extern "C" __declspec(dllexport) void CreateTexture3D(float* d_data, int3 dims, unsigned long long* h_textureid, unsigned long long* h_arrayid, bool linearfiltering);
extern "C" __declspec(dllexport) void CreateTexture3DComplex(float2* d_data, int3 dims, unsigned long long* h_textureid, unsigned long long* h_arrayid, bool linearfiltering);
extern "C" __declspec(dllexport) void DestroyTexture(unsigned long long textureid, unsigned long long arrayid);

extern "C" __declspec(dllexport) void ValueFill(float* d_input, size_t elements, float value);
extern "C" __declspec(dllexport) void ValueFillComplex(float2* d_input, size_t elements, float2 value);

extern "C" __declspec(dllexport) void Real(float2* d_input, float* d_output, size_t elements);
extern "C" __declspec(dllexport) void Imag(float2* d_input, float* d_output, size_t elements);

extern "C" __declspec(dllexport) int PeekLastCUDAError();

extern "C" __declspec(dllexport) void DistortImages(float* d_input, int2 dimsinput, float* d_output, int2 dimsoutput, float2* h_offsets, float* h_rotations, float3* h_scales, float noisestddev, int seed, uint batch);
extern "C" __declspec(dllexport) void DistortImagesAffine(float* d_input, int2 dimsinput, float* d_output, int2 dimsoutput, float2* h_offsets, float4* h_distortions, uint batch);
extern "C" __declspec(dllexport) void WarpImage(float* d_input, float* d_output, int2 dims, float* h_warpx, float* h_warpy, int2 dimswarp, cudaArray_t a_input);

extern "C" __declspec(dllexport) void Rotate3DExtractAt(unsigned long long t_volume, int3 dimsvolume, float* d_proj, int3 dimsproj, float3* h_angles, float3* h_positions, uint batch);

extern "C" __declspec(dllexport) void BackProjectTomo(float2* d_volumeft, int3 dimsvolume, float2* d_projft, float* d_projweights, int3 dimsproj, uint rmax, float3* h_angles, uint batch);

extern "C" __declspec(dllexport) void Repeat(float* d_input, float* d_output, uint elements, uint ncopies, int batch);


// WeightOptimization.cpp:
extern "C" __declspec(dllexport) void OptimizeWeights(int nrecs,
                                                        float* h_recft, 
                                                        float* h_recweights, 
                                                        float* h_r2, 
                                                        int elements, 
                                                        int* h_subsets, 
                                                        float* h_bfacs, 
                                                        float* h_weightfactors, 
                                                        float* h_recsum1, 
                                                        float* h_recsum2, 
                                                        float* h_weightsum1, 
                                                        float* h_weightsum2);

#endif
