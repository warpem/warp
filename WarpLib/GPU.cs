using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security;
using Warp.Tools;

namespace Warp
{
    [SuppressUnmanagedCodeSecurity]
    public static class GPU
    {
        public static readonly object Sync = new object();

        public static event Action MemoryChanged;

        public static void OnMemoryChanged()
        {
            //MemoryChanged?.Invoke();
        }

        // Memory.cpp:

        [DllImport("NativeAcceleration", EntryPoint = "GetDeviceCount")]
        public static extern int GetDeviceCount();

        [DllImport("NativeAcceleration", EntryPoint = "SetDevice")]
        public static extern void SetDevice(int id);

        [DllImport("NativeAcceleration", EntryPoint = "GetDevice")]
        public static extern int GetDevice();

        [DllImport("NativeAcceleration", EntryPoint = "GetFreeMemory")]
        public static extern long GetFreeMemory(int device);

        [DllImport("NativeAcceleration", EntryPoint = "GetTotalMemory")]
        public static extern long GetTotalMemory(int device);

        [DllImport("NativeAcceleration", EntryPoint = "GetDeviceName")]
        public static extern IntPtr GetDeviceName(int device);

        [DllImport("NativeAcceleration", EntryPoint = "DeviceSynchronize")]
        public static extern void DeviceSynchronize();

        [DllImport("NativeAcceleration", EntryPoint = "MallocDevice")]
        public static extern IntPtr MallocDevice(long elements);

        [DllImport("NativeAcceleration", EntryPoint = "MallocDeviceFromHost")]
        public static extern IntPtr MallocDeviceFromHost(float[] h_data, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "MallocDeviceFromHostInt")]
        public static extern IntPtr MallocDeviceFromHostInt(int[] h_data, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "MallocDeviceHalf")]
        public static extern IntPtr MallocDeviceHalf(long elements);

        [DllImport("NativeAcceleration", EntryPoint = "MallocDeviceHalfFromHost")]
        public static extern IntPtr MallocDeviceHalfFromHost(float[] h_data, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "FreeDevice")]
        public static extern void FreeDevice(IntPtr d_data);

        [DllImport("NativeAcceleration", EntryPoint = "CopyDeviceToHost")]
        public static extern void CopyDeviceToHost(IntPtr d_source, float[] h_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "CopyDeviceToHostPinned")]
        public static extern void CopyDeviceToHostPinned(IntPtr d_source, IntPtr hp_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "CopyDeviceHalfToHost")]
        public static extern void CopyDeviceHalfToHost(IntPtr d_source, float[] h_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "CopyDeviceToDevice")]
        public static extern void CopyDeviceToDevice(IntPtr d_source, IntPtr d_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "CopyDeviceHalfToDeviceHalf")]
        public static extern void CopyDeviceHalfToDeviceHalf(IntPtr d_source, IntPtr d_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "CopyHostToDevice")]
        public static extern void CopyHostToDevice(float[] h_source, IntPtr d_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "CopyHostToHost")]
        public static extern void CopyHostToHost(float[] h_source, IntPtr h_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "CopyHostToHost")]
        public static extern void CopyHostToHost(IntPtr h_source, float[] h_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "CopyHostToDevice")]
        public static extern void CopyHostPinnedToDevice(IntPtr hp_source, IntPtr d_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "CopyHostToDeviceHalf")]
        public static extern void CopyHostToDeviceHalf(float[] h_source, IntPtr d_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "SingleToHalf")]
        public static extern void SingleToHalf(IntPtr d_source, IntPtr d_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "HalfToSingle")]
        public static extern void HalfToSingle(IntPtr d_source, IntPtr d_dest, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "MallocHostPinned")]
        public static extern IntPtr MallocHostPinned(long elements);

        [DllImport("NativeAcceleration", EntryPoint = "FreeHostPinned")]
        public static extern void FreeHostPinned(IntPtr hp_data);

        [DllImport("NativeAcceleration", EntryPoint = "DeviceReset")]
        public static extern void DeviceReset();

        [DllImport("NativeAcceleration", EntryPoint = "MallocArray")]
        public static extern IntPtr MallocArray(int2 dims);

        [DllImport("NativeAcceleration", EntryPoint = "CopyDeviceToArray")]
        public static extern void CopyDeviceToArray(IntPtr d_input, IntPtr a_output, int2 dims);

        [DllImport("NativeAcceleration", EntryPoint = "FreeArray")]
        public static extern void FreeArray(IntPtr a_input);

        // Backprojection.cu:
        [DllImport("NativeAcceleration", EntryPoint = "ProjBackwardOne")]
        public static extern void ProjBackwardOne(IntPtr d_volume, int3 dimsvolume, IntPtr d_image, int2 dimsimage, float[] h_angles, float[] h_offsets, bool outputzerocentered, bool sliceonly, bool halfonly, int batch);

        // BoxNet2.cu:
        [DllImport("NativeAcceleration", EntryPoint = "BoxNet2Augment")]
        public static extern void BoxNet2Augment(IntPtr d_inputmic,
                                                 IntPtr d_inputlabel,
                                                 int2 dimsinput,
                                                 IntPtr d_outputmic,
                                                 IntPtr d_outputlabel,
                                                 int2 dimsoutput,
                                                 float[] h_offsets,
                                                 float[] h_rotations,
                                                 float[] h_scales,
                                                 float offsetmean,
                                                 float offsetscale,
                                                 float noisestddev,
                                                 int seed,
                                                 bool channelsfirst,
                                                 uint batch);

        // C2DNet.cu:
        [DllImport("NativeAcceleration", EntryPoint = "C2DNetAlign")]
        public static extern void C2DNetAlign(IntPtr d_refs,
                                               int dimprojector,
                                               float supersample,
                                               IntPtr d_dataft,
                                               IntPtr d_data,
                                               IntPtr d_ctf,
                                               int dimdata,
                                               IntPtr d_initposes,
                                               int nposesperdata,
                                               int initshell,
                                               int maxshell,
                                               int niters,
                                               float anglestep,
                                               float shiftstep,
                                               int ntop,
                                               int batch,
                                               IntPtr d_aligneddata,
                                               IntPtr d_alignedctf);

        // CubeNet.cu:
        [DllImport("NativeAcceleration", EntryPoint = "CubeNetAugment")]
        public static extern void CubeNetAugment(IntPtr d_inputvol,
                                                IntPtr d_inputlabel,
                                                int3 dimsinput,
                                                IntPtr d_outputmic,
                                                IntPtr d_outputlabel,
                                                int nclasses,
                                                int3 dimsoutput,
                                                float[] h_offsets,
                                                float[] h_rotations,
                                                float[] h_scales,
                                                float noisestddev,
                                                int seed,
                                                uint batch);

        // Correlation.cpp:
        [DllImport("NativeAcceleration", EntryPoint = "CorrelateSubTomos")]
        public static extern void CorrelateSubTomos(ulong t_projectordataRe,
                                                    ulong t_projectordataIm,
                                                    float projectoroversample,
                                                    int3 dimsprojector,
                                                    IntPtr d_experimentalft,
                                                    IntPtr d_ctf,
                                                    int3 dimsvolume,
                                                    uint nvolumes,
                                                    float[] h_angles,
                                                    uint nangles,
                                                    uint batchangles,
                                                    float maskradius,
                                                    IntPtr d_bestcorrelation,
                                                    IntPtr d_bestangle,
                                                    float[] h_progressfraction);

        [DllImport("NativeAcceleration", EntryPoint = "LocalPeaks")]
        public static extern IntPtr LocalPeaks(IntPtr d_input, int[] h_peaksnum, int3 dims, int localextent, float threshold);

        [DllImport("NativeAcceleration", EntryPoint = "SubpixelMax")]
        public static extern void SubpixelMax(IntPtr d_input, IntPtr d_output, int3 dims, int subpixsteps);

        [DllImport("NativeAcceleration", EntryPoint = "PeakOne2D")]
        public static extern void PeakOne2D(IntPtr d_input, IntPtr d_positions, IntPtr d_values, int2 dims, int2 dimsregion, bool subtractcenter, int batch);

        [DllImport("NativeAcceleration", EntryPoint = "CorrelateRealspace")]
        public static extern void CorrelateRealspace(IntPtr d_image1, IntPtr d_image2, int3 dims, IntPtr d_mask, IntPtr d_corr, uint batch);

        // CTF.cu:

        [DllImport("NativeAcceleration", EntryPoint = "CreateSpectra")]
        public static extern void CreateSpectra(IntPtr d_frame,
                                                int2 dimsframe,
                                                int nframes,
                                                int3[] h_origins,
                                                int norigins,
                                                int2 dimsregion,
                                                int3 ctfgrid,
                                                int2 dimsregionscaled,
                                                IntPtr d_outputall,
                                                IntPtr d_outputmean,
                                                int planforw,
                                                int planback);

        [DllImport("NativeAcceleration", EntryPoint = "CTFMakeAverage")]
        public static extern void CTFMakeAverage(IntPtr d_ps,
                                                 IntPtr d_pscoords,
                                                 uint length,
                                                 uint sidelength,
                                                 CTFStruct[] h_sourceparams,
                                                 CTFStruct targetparams,
                                                 uint minbin,
                                                 uint maxbin,
                                                 uint batch,
                                                 IntPtr d_output);

        [DllImport("NativeAcceleration", EntryPoint = "CTFCompareToSim")]
        public static extern void CTFCompareToSim(IntPtr d_ps,
                                                  IntPtr d_pscoords,
                                                  IntPtr d_scale,
                                                  uint length,
                                                  CTFStruct[] h_sourceparams,
                                                  float[] h_scores,
                                                  uint batch);

        // Deconv.cu:
        [DllImport("NativeAcceleration", EntryPoint = "DeconvolveCTF")]
        public static extern void DeconvolveCTF(IntPtr d_inputft, IntPtr d_outputft, int3 dims, CTFStruct ctfparams, float strength, float falloff, float highpassnyquist);

        // Defects.cu:
        [DllImport("NativeAcceleration", EntryPoint = "CorrectDefects")]
        public static extern void CorrectDefects(IntPtr d_input, IntPtr d_output, IntPtr d_locations, IntPtr d_neighbors, int ndefects);

        // DFT.cu:
        [DllImport("NativeAcceleration", EntryPoint = "BenchmarkCUFFT")]
        public static extern void BenchmarkCUFFT(IntPtr d_input, IntPtr d_output, int planforw, int elements, int batch, int repeats);

        [DllImport("NativeAcceleration", EntryPoint = "BenchmarkDFT")]
        public static extern void BenchmarkDFT(IntPtr d_input, IntPtr d_sincos, IntPtr d_output, int elements, int batch, int repeats);

        [DllImport("NativeAcceleration", EntryPoint = "BenchmarkBLASFT")]
        public static extern void BenchmarkBLASFT(IntPtr cublas, IntPtr d_input, IntPtr d_sincos, IntPtr d_output, int elements, int batch, int repeats);


        // ParticleMultiBody.cu:

        [DllImport("NativeAcceleration", EntryPoint = "ParticleMultibodyGetDiff")]
        public static extern void ParticleMultibodyGetDiff(IntPtr d_experimental,
                                                           IntPtr d_reference,
                                                           IntPtr d_shiftfactors,
                                                           IntPtr d_ctf,
                                                           IntPtr d_invsigma2,
                                                           int2 dims,
                                                           float[] h_shifts,
                                                           float[] h_diff,
                                                           uint nparticles,
                                                           uint nbodies);

        [DllImport("NativeAcceleration", EntryPoint = "ParticleMultibodyProject")]
        public static extern void ParticleMultibodyProject(ulong[] h_textureRe,
                                                           ulong[] h_textureIm,
                                                           int3 dimsvolume,
                                                           IntPtr d_proj,
                                                           int2 dimsproj,
                                                           float[] h_angles,
                                                           float[] h_shifts,
                                                           float[] h_globalweights,
                                                           float supersample,
                                                           uint nbodies,
                                                           uint batch);


        // ParticleNMA.cu:

        [DllImport("NativeAcceleration", EntryPoint = "ParticleNMAGetDiff")]
        public static extern void ParticleNMAGetDiff(IntPtr d_experimental,
                                                     IntPtr d_reference,
                                                     IntPtr d_ctf,
                                                     IntPtr d_invsigma2,
                                                     int2 dims,
                                                     float[] h_diff,
                                                     uint nparticles);

        [DllImport("NativeAcceleration", EntryPoint = "ParticleNMAGetMeanDisplacement")]
        public static extern void ParticleNMAGetMeanDisplacement(IntPtr d_normalmodes,
                                                                 uint natoms,
                                                                 float[] h_modefactors,
                                                                 uint nmodes,
                                                                 uint nmodels,
                                                                 float[] h_displacement);

        [DllImport("NativeAcceleration", EntryPoint = "ParticleNMAGetRigidTransform")]
        public static extern void ParticleNMAGetRigidTransform(IntPtr d_positions,
                                                               IntPtr d_normalmodes,
                                                               uint natoms,
                                                               float[] h_modefactors,
                                                               uint nmodes,
                                                               uint nmodels,
                                                               float[] h_centertrans,
                                                               float[] h_rotmat);


        // Post.cu:

        [DllImport("NativeAcceleration", EntryPoint = "DoseWeighting")]
        public static extern void DoseWeighting(IntPtr d_freq, IntPtr d_output, uint length, float[] h_doserange, float3 nikoconst, float voltagescaling, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "CorrectMagAnisotropy")]
        public static extern void CorrectMagAnisotropy(IntPtr d_image,
                                                       int2 dimsimage,
                                                       IntPtr d_scaled,
                                                       int2 dimsscaled,
                                                       float majorpixel,
                                                       float minorpixel,
                                                       float majorangle,
                                                       uint supersample,
                                                       uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "NormParticles")]
        public static extern void NormParticles(IntPtr d_input, IntPtr d_output, int3 dims, uint particleradius, bool flipsign, uint batch);


        // Raycast.cu:

        [DllImport("NativeAcceleration", EntryPoint = "RenderVolume")]
        public static extern void RenderVolume(ulong t_intensities,
                                               int3 dimsvolume,
                                               float surfacethreshold,
                                               ulong t_coloring,
                                               int3 dimscoloring,
                                               int2 dimsimage,
                                               float3 camera,
                                               float3 pixelx,
                                               float3 pixely,
                                               float3 view,
                                               IntPtr d_colormap,
                                               int ncolors,
                                               float2 colorrange,
                                               float2 shadingrange,
                                               float3 intensitycolor,
                                               float2 intensityrange,
                                               float[] h_intersectionpoints,
                                               byte[] h_hittest,
                                               byte[] h_bgra);

        // RealspaceProjections.cu:

        [DllImport("NativeAcceleration", EntryPoint = "RealspaceProjectForward")]
        public static extern void RealspaceProjectForward(IntPtr d_volume,
                                                          int3 dimsvolume,
                                                          IntPtr d_projections,
                                                          int2 dimsproj,
                                                          float supersample,
                                                          float[] h_angles,
                                                          int batch);

        [DllImport("NativeAcceleration", EntryPoint = "RealspaceProjectBackward")]
        public static extern void RealspaceProjectBackward(IntPtr d_volume,
                                                           int3 dimsvolume,
                                                           IntPtr d_projections,
                                                           int2 dimsproj,
                                                           float supersample,
                                                           float[] h_angles,
                                                           bool normalizesamples,
                                                           int batch);

        // Shift.cu:

        [DllImport("NativeAcceleration", EntryPoint = "CreateShift")]
        public static extern void CreateShift(IntPtr d_frame,
                                              int2 dimsframe,
                                              int nframes,
                                              int3[] h_origins,
                                              int norigins,
                                              int2 dimsregion,
                                              long[] h_mask,
                                              uint masklength,
                                              IntPtr d_outputall,
                                              IntPtr d_sigma);

        [DllImport("NativeAcceleration", EntryPoint = "ShiftGetAverage")]
        public static extern void ShiftGetAverage(IntPtr d_individual,
                                                  IntPtr d_average,
                                                  IntPtr d_shiftfactors,
                                                  uint length,
                                                  uint probelength,
                                                  IntPtr d_shifts,
                                                  uint nspectra,
                                                  uint nframes);

        [DllImport("NativeAcceleration", EntryPoint = "ShiftGetDiff")]
        public static extern void ShiftGetDiff(IntPtr d_individual,
                                               IntPtr d_average,
                                               IntPtr d_shiftfactors,
                                               uint length,
                                               uint probelength,
                                               IntPtr d_shifts,
                                               float[] h_diff,
                                               uint npositions,
                                               uint nframes);

        [DllImport("NativeAcceleration", EntryPoint = "ShiftGetGrad")]
        public static extern void ShiftGetGrad(IntPtr d_individual,
                                               IntPtr d_average,
                                               IntPtr d_shiftfactors,
                                               uint length,
                                               uint probelength,
                                               IntPtr d_shifts,
                                               float[] h_grad,
                                               uint npositions,
                                               uint nframes);

        [DllImport("NativeAcceleration", EntryPoint = "CreateMotionBlur")]
        public static extern void CreateMotionBlur(IntPtr d_output, int3 dims, float[] h_shifts, uint nshifts, uint batch);


        // ParticleSoftBody.cu:

        [DllImport("NativeAcceleration", EntryPoint = "ParticleSoftBodyDeform")]
        public static extern void ParticleSoftBodyDeform(IntPtr d_initialpositions,
                                                         IntPtr d_finalpositions,
                                                         uint npositions,
                                                         IntPtr d_neighborids,
                                                         IntPtr d_neighboroffsets,
                                                         IntPtr d_edgelengths,
                                                         IntPtr d_connectedactuator,
                                                         IntPtr d_actuatordeltas,
                                                         uint nactuators,
                                                         uint niterations,
                                                         uint nrelaxations,
                                                         uint batch);

        // MPARefine.cu:
        [DllImport("NativeAcceleration", EntryPoint = "MultiParticleDiff")]
        public static extern void MultiParticleDiff(float[] h_result,
                                                    IntPtr[] hp_experimental,
                                                    int dimdata,
                                                    int[] relevantdims,
                                                    float[] h_shifts,
                                                    float[] h_angles,
                                                    float4 c_magnification,
                                                    IntPtr d_weights,
                                                    IntPtr d_phasecorrection,
                                                    float ewaldradius,
                                                    int maxshell,
                                                    ulong[] h_volumeRe,
                                                    ulong[] h_volumeIm,
                                                    float supersample,
                                                    int dimprojector,
                                                    IntPtr d_subsets,
                                                    int nparticles,
                                                    int ntilts);

        [DllImport("NativeAcceleration", EntryPoint = "MultiParticleSimulate")]
        public static extern void MultiParticleSimulate(IntPtr d_result,
                                                        int2 dimsmic,
                                                        int dimdata,
                                                        float[] h_positions,
                                                        float[] h_shifts,
                                                        float[] h_angles,
                                                        float[] h_defoci,
                                                        IntPtr d_weights,
                                                        CTFStruct[] h_ctfparams,
                                                        ulong[] h_volumeRe,
                                                        ulong[] h_volumeIm,
                                                        int dimprojector,
                                                        IntPtr d_subsets,
                                                        int nparticles,
                                                        int ntilts);

        [DllImport("NativeAcceleration", EntryPoint = "MultiParticleCorr2D")]
        public static extern void MultiParticleCorr2D(IntPtr d_result2d,
                                                      IntPtr d_result1dparticles,
                                                      IntPtr d_resultphaseresiduals,
                                                      int dimresult,
                                                      IntPtr[] hp_experimental,
                                                      IntPtr d_weights,
                                                      int dimdata,
                                                      float scalingfactor,
                                                      int[] relevantdims,
                                                      float[] h_shifts,
                                                      float[] h_angles,
                                                      float4 c_magnification,
                                                      float ewaldradius,
                                                      ulong[] h_volumeRe,
                                                      ulong[] h_volumeIm,
                                                      float supersample,
                                                      int dimprojector,
                                                      IntPtr d_subsets,
                                                      int nparticles,
                                                      int ntilts);

        [DllImport("NativeAcceleration", EntryPoint = "MultiParticleResidual")]
        public static extern void MultiParticleResidual(IntPtr d_result,
                                                        IntPtr[] hp_experimental,
                                                        int dimdata,
                                                        float[] h_shifts,
                                                        float[] h_angles,
                                                        float4 magnification,
                                                        IntPtr d_weights,
                                                        ulong[] h_volumeRe,
                                                        ulong[] h_volumeIm,
                                                        int dimprojector,
                                                        IntPtr d_subsets,
                                                        int nparticles,
                                                        int ntilts);

        [DllImport("NativeAcceleration", EntryPoint = "MultiParticleSumAmplitudes")]
        public static extern void MultiParticleSumAmplitudes(float[] h_result,
                                                             int dimdata,
                                                             float[] h_angles,
                                                             ulong t_volumeRe,
                                                             ulong t_volumeIm,
                                                             int dimprojector,
                                                             int nparticles);

        // PCA.cu:
        [DllImport("NativeAcceleration", EntryPoint = "PCALeastSq")]
        public static extern void PCALeastSq(float[] h_result,
                                             IntPtr d_experimental,
                                             IntPtr d_ctf,
                                             IntPtr d_spectral,
                                             int dimdata,
                                             float rmax,
                                             float[] h_shifts,
                                             float[] h_angles,
                                             float4 magnification,
                                             ulong t_volumeRe,
                                             ulong t_volumeIm,
                                             int dimprojector,
                                             int nparticles,
                                             int ntilts);

        // Tools.cu:

        [DllImport("NativeAcceleration", EntryPoint = "FFT")]
        public static extern void FFT(IntPtr d_input, IntPtr d_output, int3 dims, uint batch, int plan);

        [DllImport("NativeAcceleration", EntryPoint = "IFFT")]
        public static extern void IFFT(IntPtr d_input, IntPtr d_output, int3 dims, uint batch, int plan, bool normalize);

        [DllImport("NativeAcceleration", EntryPoint = "Pad")]
        public static extern void Pad(IntPtr d_input, IntPtr d_output, int3 olddims, int3 newdims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "PadClamped")]
        public static extern void PadClamped(IntPtr d_input, IntPtr d_output, int3 olddims, int3 newdims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "PadFT")]
        public static extern void PadFT(IntPtr d_input, IntPtr d_output, int3 olddims, int3 newdims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "PadFTRealValued")]
        public static extern void PadFTRealValued(IntPtr d_input, IntPtr d_output, int3 olddims, int3 newdims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "PadFTFull")]
        public static extern void PadFTFull(IntPtr d_input, IntPtr d_output, int3 olddims, int3 newdims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "CropFT")]
        public static extern void CropFT(IntPtr d_input, IntPtr d_output, int3 olddims, int3 newdims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "CropFTRealValued")]
        public static extern void CropFTRealValued(IntPtr d_input, IntPtr d_output, int3 olddims, int3 newdims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "CropFTFull")]
        public static extern void CropFTFull(IntPtr d_input, IntPtr d_output, int3 olddims, int3 newdims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "RemapToFTComplex")]
        public static extern void RemapToFTComplex(IntPtr d_input, IntPtr d_output, int3 dims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "RemapToFTFloat")]
        public static extern void RemapToFTFloat(IntPtr d_input, IntPtr d_output, int3 dims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "RemapFromFTComplex")]
        public static extern void RemapFromFTComplex(IntPtr d_input, IntPtr d_output, int3 dims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "RemapFromFTFloat")]
        public static extern void RemapFromFTFloat(IntPtr d_input, IntPtr d_output, int3 dims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "RemapFullToFTFloat")]
        public static extern void RemapFullToFTFloat(IntPtr d_input, IntPtr d_output, int3 dims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "RemapFullFromFTFloat")]
        public static extern void RemapFullFromFTFloat(IntPtr d_input, IntPtr d_output, int3 dims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Extract")]
        public static extern void Extract(IntPtr d_input,
                                          IntPtr d_output,
                                          int3 dims,
                                          int3 dimsregion,
                                          int[] h_origins,
                                          bool zeropad,
                                          uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ExtractMultisource")]
        public static extern void ExtractMultisource(IntPtr[] h_inputs,
                                          IntPtr d_output,
                                          int3 dims,
                                          int3 dimsregion,
                                          int[] h_origins,
                                          int nsources,
                                          uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ExtractHalf")]
        public static extern void ExtractHalf(IntPtr d_input,
                                              IntPtr d_output,
                                              int3 dims,
                                              int3 dimsregion,
                                              int[] h_origins,
                                              uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ReduceMean")]
        public static extern void ReduceMean(IntPtr d_input, IntPtr d_output, uint vectorlength, uint nvectors, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ReduceMeanHalf")]
        public static extern void ReduceMeanHalf(IntPtr d_input, IntPtr d_output, uint vectorlength, uint nvectors, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ReduceAdd")]
        public static extern void ReduceAdd(IntPtr d_input, IntPtr d_output, uint vectorlength, uint nvectors, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ReduceMax")]
        public static extern void ReduceMax(IntPtr d_input, IntPtr d_output, uint vectorlength, uint nvectors, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Normalize")]
        public static extern void Normalize(IntPtr d_ps,
                                               IntPtr d_output,
                                               uint length,
                                               uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "NormalizeMasked")]
        public static extern void NormalizeMasked(IntPtr d_ps,
                                                  IntPtr d_output,
                                                  IntPtr d_mask,
                                                  uint length,
                                                  uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "SphereMask")]
        public static extern void SphereMask(IntPtr d_input, IntPtr d_output, int3 dims, float radius, float sigma, bool decentered, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "CreateCTF")]
        public static extern void CreateCTF(IntPtr d_output,
                                            IntPtr d_coords,
                                            IntPtr d_gammacorrection,
                                            uint length,
                                            CTFStruct[] h_params,
                                            bool amplitudesquared,
                                            uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "CreateCTFComplex")]
        public static extern void CreateCTFComplex(IntPtr d_output,
                                                    IntPtr d_coords,
                                                    IntPtr d_gammacorrection,
                                                    uint length,
                                                    CTFStruct[] h_params,
                                                    bool reverse,
                                                    uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "CreateCTFEwaldWeights")]
        public static extern void CreateCTFEwaldWeights(IntPtr d_output,
                                            IntPtr d_coords,
                                            IntPtr d_gammacorrection,
                                            float particlediameter,
                                            uint length,
                                            CTFStruct[] h_params,
                                            uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Resize")]
        public static extern void Resize(IntPtr d_input,
                                         int3 dimsinput,
                                         IntPtr d_output,
                                         int3 dimsoutput,
                                         uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ShiftStack")]
        public static extern void ShiftStack(IntPtr d_input,
                                             IntPtr d_output,
                                             int3 dims,
                                             float[] h_shifts,
                                             uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ShiftStackFT")]
        public static extern void ShiftStackFT(IntPtr d_input,
                                               IntPtr d_output,
                                               int3 dims,
                                               float[] h_shifts,
                                               uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ShiftStackMassive")]
        public static extern void ShiftStackMassive(IntPtr d_input,
                                                    IntPtr d_output,
                                                    int3 dims,
                                                    float[] h_shifts,
                                                    uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Cart2Polar")]
        public static extern void Cart2Polar(IntPtr d_input, IntPtr d_output, int2 dims, uint innerradius, uint exclusiveouterradius, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Cart2PolarFFT")]
        public static extern void Cart2PolarFFT(IntPtr d_input, IntPtr d_output, int2 dims, uint innerradius, uint exclusiveouterradius, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Xray")]
        public static extern void Xray(IntPtr d_input, IntPtr d_output, float ndevs, int2 dims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Sum")]
        public static extern void Sum(IntPtr d_input, IntPtr d_output, uint length, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Abs")]
        public static extern void Abs(IntPtr d_input, IntPtr d_output, long length);

        [DllImport("NativeAcceleration", EntryPoint = "Amplitudes")]
        public static extern void Amplitudes(IntPtr d_input, IntPtr d_output, long length);

        [DllImport("NativeAcceleration", EntryPoint = "Sign")]
        public static extern void Sign(IntPtr d_input, IntPtr d_output, long length);

        [DllImport("NativeAcceleration", EntryPoint = "Sqrt")]
        public static extern void Sqrt(IntPtr d_input, IntPtr d_output, long length);

        [DllImport("NativeAcceleration", EntryPoint = "Cos")]
        public static extern void Cos(IntPtr d_input, IntPtr d_output, long length);

        [DllImport("NativeAcceleration", EntryPoint = "Sin")]
        public static extern void Sin(IntPtr d_input, IntPtr d_output, long length);

        [DllImport("NativeAcceleration", EntryPoint = "AddScalar")]
        public static extern void AddScalar(IntPtr d_input, float summand, IntPtr d_output, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "AddToSlices")]
        public static extern void AddToSlices(IntPtr d_input, IntPtr d_summands, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "SubtractFromSlices")]
        public static extern void SubtractFromSlices(IntPtr d_input, IntPtr d_subtrahends, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "MultiplySlices")]
        public static extern void MultiplySlices(IntPtr d_input, IntPtr d_multiplicators, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "DivideSlices")]
        public static extern void DivideSlices(IntPtr d_input, IntPtr d_divisors, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "AddToSlicesHalf")]
        public static extern void AddToSlicesHalf(IntPtr d_input, IntPtr d_summands, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "SubtractFromSlicesHalf")]
        public static extern void SubtractFromSlicesHalf(IntPtr d_input, IntPtr d_subtrahends, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "MultiplySlicesHalf")]
        public static extern void MultiplySlicesHalf(IntPtr d_input, IntPtr d_multiplicators, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "MultiplyComplexSlicesByScalar")]
        public static extern void MultiplyComplexSlicesByScalar(IntPtr d_input, IntPtr d_multiplicators, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "MultiplyComplexSlicesByComplex")]
        public static extern void MultiplyComplexSlicesByComplex(IntPtr d_input, IntPtr d_multiplicators, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "MultiplyComplexSlicesByComplexConj")]
        public static extern void MultiplyComplexSlicesByComplexConj(IntPtr d_input, IntPtr d_multiplicators, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "DivideComplexSlicesByScalar")]
        public static extern void DivideComplexSlicesByScalar(IntPtr d_input, IntPtr d_divisors, IntPtr d_output, long sliceelements, uint slices);

        [DllImport("NativeAcceleration", EntryPoint = "MultiplyByScalar")]
        public static extern void MultiplyByScalar(IntPtr d_input, IntPtr d_output, float multiplicator, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "MultiplyByScalars")]
        public static extern void MultiplyByScalars(IntPtr d_input, IntPtr d_output, IntPtr d_multiplicators, long elements, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Scale")]
        public static extern void Scale(IntPtr d_input, IntPtr d_output, int3 dimsinput, int3 dimsoutput, uint batch, int planforw, int planback, IntPtr d_inputfft, IntPtr d_outputfft);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectForward")]
        public static extern void ProjectForward(IntPtr d_inputft, IntPtr d_outputft, int3 dimsinput, int2 dimsoutput, float[] h_angles, float supersample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectForwardShifted")]
        public static extern void ProjectForwardShifted(IntPtr d_inputft, IntPtr d_outputft, int3 dimsinput, int2 dimsoutput, float[] h_angles, float[] h_shifts, float[] h_globalweights, float supersample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectForward3D")]
        public static extern void ProjectForward3D(IntPtr d_inputft, IntPtr d_outputft, int3 dimsinput, int3 dimsoutput, float[] h_angles, float supersample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectForward3DShifted")]
        public static extern void ProjectForward3DShifted(IntPtr d_inputft, IntPtr d_outputft, int3 dimsinput, int3 dimsoutput, float[] h_angles, float[] h_shifts, float[] h_globalweights, float supersample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectForwardTex")]
        public static extern void ProjectForwardTex(ulong t_inputRe, ulong t_inputIm, IntPtr d_outputft, int3 dimsinput, int2 dimsoutput, float[] h_angles, float supersample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectForwardShiftedTex")]
        public static extern void ProjectForwardShiftedTex(ulong t_inputRe, ulong t_inputIm, IntPtr d_outputft, int3 dimsinput, int2 dimsoutput, float[] h_angles, float[] h_shifts, float[] h_globalweights, float supersample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectForward3DTex")]
        public static extern void ProjectForward3DTex(ulong t_inputRe, ulong t_inputIm, IntPtr d_outputft, int3 dimsinput, int3 dimsoutput, float[] h_angles, float supersample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectForward3DShiftedTex")]
        public static extern void ProjectForward3DShiftedTex(ulong t_inputRe, ulong t_inputIm, IntPtr d_outputft, int3 dimsinput, int3 dimsoutput, float[] h_angles, float[] h_shifts, float[] h_globalweights, float supersample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectBackward")]
        public static extern void ProjectBackward(IntPtr d_volumeft, IntPtr d_volumeweights, int3 dimsvolume, IntPtr d_projft, IntPtr d_projweights, int2 dimsproj, int rmax, float[] h_angles, int[] ivolume, float4 magnification, float ewaldradius, float supersample, bool outputdecentered, bool squareinputweights, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectBackwardShifted")]
        public static extern void ProjectBackwardShifted(IntPtr d_volumeft, IntPtr d_volumeweights, int3 dimsvolume, IntPtr d_projft, IntPtr d_projweights, int2 dimsproj, int rmax, float[] h_angles, float[] h_shifts, float[] h_globalweights, float supersample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "BackprojectorReconstructGPU")]
        public static extern void BackprojectorReconstructGPU(int3 dimsori, 
                                                              int3 dimspadded, 
                                                              int oversampling, 
                                                              IntPtr d_dataft, 
                                                              IntPtr d_weights, 
                                                              string c_symmetry, 
                                                              int helix_units, 
                                                              float helix_twist, 
                                                              float helix_rise, 
                                                              bool do_reconstruct_ctf, 
                                                              IntPtr d_result, 
                                                              int pre_planforw = -1, 
                                                              int pre_planback = -1, 
                                                              int pre_planforwctf = -1, 
                                                              int griddingiterations = 10, 
                                                              int nvolumes = 1);

        [DllImport("NativeAcceleration", EntryPoint = "Bandpass")]
        public static extern void Bandpass(IntPtr d_input, IntPtr d_output, int3 dims, float nyquistlow, float nyquisthigh, float nyquistsoftedge, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "FourierBandpass")]
        public static extern void FourierBandpass(IntPtr d_inputft, int3 dims, float nyquistlow, float nyquisthigh, float nyquistsoftedge, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "BandpassGauss")]
        public static extern void BandpassGauss(IntPtr d_input, IntPtr d_output, int3 dims, float nyquistlow, float nyquisthigh, float nyquistsigma, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "FourierBandpassGauss")]
        public static extern void FourierBandpassGauss(IntPtr d_inputft, int3 dims, float nyquistlow, float nyquisthigh, float nyquistsigma, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "BandpassButter")]
        public static extern void BandpassButter(IntPtr d_input, IntPtr d_output, int3 dims, float nyquistlow, float nyquisthigh, int order, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "FourierBandpassButter")]
        public static extern void FourierBandpassButter(IntPtr d_inputft, int3 dims, float nyquistlow, float nyquisthigh, int order, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "LocalStd")]
        public static extern void LocalStd(IntPtr d_map, int3 dimsmap, float localradius, IntPtr d_std, IntPtr d_mean, int planforw, int planback);

        [DllImport("NativeAcceleration", EntryPoint = "Rotate2D")]
        public static extern void Rotate2D(IntPtr d_input, IntPtr d_output, int2 dims, float[] h_angles, int oversample, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ShiftAndRotate2D")]
        public static extern void ShiftAndRotate2D(IntPtr d_input, IntPtr d_output, int2 dims, float[] h_shifts, float[] h_angles, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "MinScalar")]
        public static extern void MinScalar(IntPtr d_input, IntPtr d_output, float value, uint elements);

        [DllImport("NativeAcceleration", EntryPoint = "MaxScalar")]
        public static extern void MaxScalar(IntPtr d_input, IntPtr d_output, float value, uint elements);

        [DllImport("NativeAcceleration", EntryPoint = "Min")]
        public static extern void Min(IntPtr d_input1, IntPtr d_input2, IntPtr d_output, uint elements);

        [DllImport("NativeAcceleration", EntryPoint = "Max")]
        public static extern void Max(IntPtr d_input1, IntPtr d_input2, IntPtr d_output, uint elements);

        [DllImport("NativeAcceleration", EntryPoint = "CreateFFTPlan")]
        public static extern int CreateFFTPlan(int3 dims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "CreateIFFTPlan")]
        public static extern int CreateIFFTPlan(int3 dims, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "DestroyFFTPlan")]
        public static extern void DestroyFFTPlan(int plan);

        [DllImport("NativeAcceleration", EntryPoint = "CreateBLAS")]
        public static extern IntPtr CreateBLAS();

        [DllImport("NativeAcceleration", EntryPoint = "DestroyBLAS")]
        public static extern void DestroyBLAS(IntPtr handle);

        [DllImport("NativeAcceleration", EntryPoint = "LocalRes")]
        public static extern void LocalRes(IntPtr d_volume1,
                                           IntPtr d_volume2,
                                           int3 dims,
                                           float pixelsize,
                                           IntPtr d_filtered,
                                           IntPtr d_filteredsharp,
                                           IntPtr d_localres,
                                           IntPtr d_localbfac,
                                           int windowsize,
                                           float fscthreshold,
                                           bool dolocalbfactor,
                                           float minresbfactor,
                                           float globalbfactor,
                                           float mtfslope,
                                           bool doanisotropy,
                                           bool dofilterhalfmaps);

        [DllImport("NativeAcceleration", EntryPoint = "LocalFSC")]
        public static extern void LocalFSC(IntPtr d_volume1,
                                           IntPtr d_volume2,
                                           IntPtr d_volumemask,
                                           int3 dims,
                                           int spacing,
                                           float pixelsize,
                                           IntPtr d_localres,
                                           int windowsize,
                                           float fscthreshold,
                                           IntPtr d_avgfsc,
                                           IntPtr d_avgamps,
                                           IntPtr d_avgsamples,
                                           int avgoversample,
                                           float[] h_globalfsc);

        [DllImport("NativeAcceleration", EntryPoint = "LocalFilter")]
        public static extern void LocalFilter(IntPtr d_input,
                                              IntPtr d_filtered,
                                              int3 dims,
                                              IntPtr d_localres,
                                              int windowsize,
                                              float angpix,
                                              IntPtr d_filterramps,
                                              int rampsoversample);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectNMAPseudoAtoms")]
        public static extern void ProjectNMAPseudoAtoms(IntPtr d_positions,
                                                        IntPtr d_intensities,
                                                        uint natoms,
                                                        int3 dimsvol,
                                                        float sigma,
                                                        uint kernelextent,
                                                        float[] h_blobvalues,
                                                        float blobsampling,
                                                        uint nblobvalues,
                                                        IntPtr d_normalmodes,
                                                        IntPtr d_normalmodefactors,
                                                        uint nmodes,
                                                        float[] h_angles,
                                                        float[] h_offsets,
                                                        float scale,
                                                        IntPtr d_proj,
                                                        int2 dimsproj,
                                                        uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "ProjectSoftPseudoAtoms")]
        public static extern void ProjectSoftPseudoAtoms(IntPtr d_positions,
                                                         IntPtr d_intensities,
                                                         uint natoms,
                                                         int3 dimsvol,
                                                         float sigma,
                                                         uint kernelextent,
                                                         IntPtr d_coarsedeltas,
                                                         IntPtr d_coarseweights,
                                                         IntPtr d_coarseneighbors,
                                                         uint ncoarseneighbors,
                                                         uint ncoarse,
                                                         float[] h_angles,
                                                         float[] h_offsets,
                                                         float scale,
                                                         IntPtr d_proj,
                                                         int2 dimsproj,
                                                         uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "DistanceMap")]
        public static extern void DistanceMap(IntPtr d_input, IntPtr d_output, int3 dims, int maxiterations);

        [DllImport("NativeAcceleration", EntryPoint = "DistanceMapExact")]
        public static extern void DistanceMapExact(IntPtr d_input, IntPtr d_output, int3 dims, int maxdistance);

        [DllImport("NativeAcceleration", EntryPoint = "PrefilterForCubic")]
        public static extern void PrefilterForCubic(IntPtr d_data, int3 dims);

        [DllImport("NativeAcceleration", EntryPoint = "CreateTexture3D")]
        public static extern void CreateTexture3D(IntPtr d_data, int3 dims, ulong[] h_textureid, ulong[] h_arrayid, bool linearfiltering);

        [DllImport("NativeAcceleration", EntryPoint = "CreateTexture3DComplex")]
        public static extern void CreateTexture3DComplex(IntPtr d_data, int3 dims, ulong[] h_textureid, ulong[] h_arrayid, bool linearfiltering);

        [DllImport("NativeAcceleration", EntryPoint = "DestroyTexture")]
        public static extern void DestroyTexture(ulong textureid, ulong arrayid);

        [DllImport("NativeAcceleration", EntryPoint = "ValueFill")]
        public static extern void ValueFill(IntPtr d_input, long elements, float value);

        [DllImport("NativeAcceleration", EntryPoint = "ValueFillComplex")]
        public static extern void ValueFillComplex(IntPtr d_input, long elements, float2 value);

        [DllImport("NativeAcceleration", EntryPoint = "Real")]
        public static extern void Real(IntPtr d_input, IntPtr d_output, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "Imag")]
        public static extern void Imag(IntPtr d_input, IntPtr d_output, long elements);

        [DllImport("NativeAcceleration", EntryPoint = "SymmetrizeFT")]
        public static extern void SymmetrizeFT(IntPtr d_data, int3 dims, string c_symmetry);

        [DllImport("NativeAcceleration", EntryPoint = "DistortImages")]
        public static extern void DistortImages(IntPtr d_input, int2 dimsinput, IntPtr d_output, int2 dimsoutput, float[] h_offsets, float[] h_rotations, float[] h_scales, float noisestddev, int seed, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "DistortImagesAffine")]
        public static extern void DistortImagesAffine(IntPtr d_input, int2 dimsinput, IntPtr d_output, int2 dimsoutput, float[] h_offsets, float[] h_distortions, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "WarpImage")]
        public static extern void WarpImage(IntPtr d_input, IntPtr d_output, int2 dims, float[] h_warpx, float[] h_warpy, int2 dimswarp, IntPtr a_input);

        [DllImport("NativeAcceleration", EntryPoint = "Rotate3DExtractAt")]
        public static extern void Rotate3DExtractAt(ulong t_volume, int3 dimsvolume, IntPtr d_proj, int3 dimsproj, float[] h_angles, float[] h_positions, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "BackProjectTomo")]
        public static extern void BackProjectTomo(IntPtr d_volumeft, int3 dimsvolume, IntPtr d_projft, IntPtr d_projweights, int3 dimsproj, uint rmax, float[] h_angles, uint batch);

        [DllImport("NativeAcceleration", EntryPoint = "Repeat")]
        public static extern void Repeat(IntPtr d_input, IntPtr d_output, uint elements, uint ncopies, int batch);



        [DllImport("NativeAcceleration", EntryPoint = "PeekLastCUDAError")]
        public static extern int PeekLastCUDAError();

        public static int GetDeviceWithMostMemory()
        {
            int[] SortedIndices = Helper.AsSortedIndices(Helper.ArrayOfFunction(i => GetTotalMemory(i), GetDeviceCount()), (a, b) => a.CompareTo(b));
            return SortedIndices[SortedIndices.Length - 1];
        }

        public static void CheckGPUExceptions()
        {
            int Result = PeekLastCUDAError();
            if (Result != 0)
            {
                Dictionary<int, string> ErrorNames = new Dictionary<int, string>()
                {
                    { 1, "cudaErrorMissingConfiguration" },
                    { 2, "cudaErrorMemoryAllocation" },
                    { 3, "cudaErrorInitializationError" },
                    { 4, "cudaErrorLaunchFailure" },
                    { 5, "cudaErrorPriorLaunchFailure" },
                    { 6, "cudaErrorLaunchTimeout" },
                    { 7, "cudaErrorLaunchOutOfResources" },
                    { 8, "cudaErrorInvalidDeviceFunction" },
                    { 9, "cudaErrorInvalidConfiguration" },
                    { 10, "cudaErrorInvalidDevice" },
                    { 11, "cudaErrorInvalidValue" },
                    { 12, "cudaErrorInvalidPitchValue" },
                    { 13, "cudaErrorInvalidSymbol" },
                    { 14, "cudaErrorMapBufferObjectFailed" },
                    { 15, "cudaErrorUnmapBufferObjectFailed" },
                    { 16, "cudaErrorInvalidHostPointer" },
                    { 17, "cudaErrorInvalidDevicePointer" },
                    { 18, "cudaErrorInvalidTexture" },
                    { 19, "cudaErrorInvalidTextureBinding" },
                    { 20, "cudaErrorInvalidChannelDescriptor" },
                    { 21, "cudaErrorInvalidMemcpyDirection" },
                    { 22, "cudaErrorAddressOfConstant" },
                    { 23, "cudaErrorTextureFetchFailed" },
                    { 24, "cudaErrorTextureNotBound" },
                    { 25, "cudaErrorSynchronizationError" },
                    { 26, "cudaErrorInvalidFilterSetting" },
                    { 27, "cudaErrorInvalidNormSetting" },
                    { 28, "cudaErrorMixedDeviceExecution" },
                    { 29, "cudaErrorCudartUnloading" },
                    { 30, "cudaErrorUnknown" },
                    { 31, "cudaErrorNotYetImplemented" },
                    { 32, "cudaErrorMemoryValueTooLarge" },
                    { 33, "cudaErrorInvalidResourceHandle" },
                    { 34, "cudaErrorNotReady" },
                    { 35, "cudaErrorInsufficientDriver" },
                    { 36, "cudaErrorSetOnActiveProcess" },
                    { 37, "cudaErrorInvalidSurface" },
                    { 38, "cudaErrorNoDevice" },
                    { 39, "cudaErrorECCUncorrectable" },
                    { 40, "cudaErrorSharedObjectSymbolNotFound" },
                    { 41, "cudaErrorSharedObjectInitFailed" },
                    { 42, "cudaErrorUnsupportedLimit" },
                    { 43, "cudaErrorDuplicateVariableName" },
                    { 44, "cudaErrorDuplicateTextureName" },
                    { 45, "cudaErrorDuplicateSurfaceName" },
                    { 46, "cudaErrorDevicesUnavailable" },
                    { 47, "cudaErrorInvalidKernelImage" },
                    { 48, "cudaErrorNoKernelImageForDevice" },
                    { 49, "cudaErrorIncompatibleDriverContext" },
                    { 50, "cudaErrorPeerAccessAlreadyEnabled" },
                    { 51, "cudaErrorPeerAccessNotEnabled" },
                    { 54, "cudaErrorDeviceAlreadyInUse" },
                    { 55, "cudaErrorProfilerDisabled" },
                    { 56, "cudaErrorProfilerNotInitialized" },
                    { 57, "cudaErrorProfilerAlreadyStarted" },
                    { 58, "cudaErrorProfilerAlreadyStopped" },
                    { 59, "cudaErrorAssert" },
                    { 60, "cudaErrorTooManyPeers" },
                    { 61, "cudaErrorHostMemoryAlreadyRegistered" },
                    { 62, "cudaErrorHostMemoryNotRegistered" },
                    { 63, "cudaErrorOperatingSystem" },
                    { 64, "cudaErrorPeerAccessUnsupported" },
                    { 65, "cudaErrorLaunchMaxDepthExceeded" },
                    { 66, "cudaErrorLaunchFileScopedTex" },
                    { 67, "cudaErrorLaunchFileScopedSurf" },
                    { 68, "cudaErrorSyncDepthExceeded" },
                    { 69, "cudaErrorLaunchPendingCountExceeded" },
                    { 70, "cudaErrorNotPermitted" },
                    { 71, "cudaErrorNotSupported" },
                    { 72, "cudaErrorHardwareStackError" },
                    { 73, "cudaErrorIllegalInstruction" },
                    { 74, "cudaErrorMisalignedAddress" },
                    { 75, "cudaErrorInvalidAddressSpace" },
                    { 76, "cudaErrorInvalidPc" },
                    { 77, "cudaErrorIllegalAddress" },
                    { 78, "cudaErrorInvalidPtx" },
                    { 79, "cudaErrorInvalidGraphicsContext" }
                };

                if (ErrorNames.ContainsKey(Result))
                    throw new Exception(ErrorNames[Result]);
                else
                    throw new Exception("cudaError with unknown code");
            }
        }
    }

    public class DeviceToken
    {
        public int ID;

        public DeviceToken(int id)
        {
            ID = id;
        }
    }
}
