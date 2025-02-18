using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;

namespace Warp;

public partial class TiltSeries
{
    public override void PerformMultiParticleRefinement(string workingDirectory,
        ProcessingOptionsMPARefine optionsMPA,
        Species[] allSpecies,
        DataSource dataSource,
        Image gainRef,
        DefectModel defectMap,
        Action<string> progressCallback)
    {
        int GPUID = GPU.GetDevice();
        HeaderEER.GroupNFrames = dataSource.EERGroupFrames;

        float BfactorWeightingThreshold = (float)optionsMPA.BFactorWeightingThreshold;

        //AreAnglesInverted = false;

        //MagnificationCorrection = new float3(1, 1, 0);

        if (CTF.ZernikeCoeffsOdd == null)
            CTF.ZernikeCoeffsOdd = new float[12];
        else if (CTF.ZernikeCoeffsOdd.Length < 12)
            CTF.ZernikeCoeffsOdd = Helper.Combine(CTF.ZernikeCoeffsOdd, new float[12 - CTF.ZernikeCoeffsOdd.Length]);

        if (CTF.ZernikeCoeffsEven == null)
            CTF.ZernikeCoeffsEven = new float[8];
        else if (CTF.ZernikeCoeffsEven.Length < 8)
            CTF.ZernikeCoeffsEven = Helper.Combine(CTF.ZernikeCoeffsEven, new float[8 - CTF.ZernikeCoeffsEven.Length]);

        #region Get particles belonging to this item; if there are not enough, abort

        string DataHash = GetDataHash();

        Dictionary<Species, Particle[]> SpeciesParticles = new Dictionary<Species, Particle[]>();
        foreach (var species in allSpecies)
            SpeciesParticles.Add(species, species.GetParticles(DataHash));

        if (SpeciesParticles.Select(p => p.Value.Length).Sum() < optionsMPA.MinParticlesPerItem)
            return;

        #endregion

        #region Figure out dimensions

        float SmallestAngPix = MathHelper.Min(allSpecies.Select(s => (float)s.PixelSize));
        float LargestBox = MathHelper.Max(allSpecies.Select(s => s.DiameterAngstrom)) * 2 / SmallestAngPix;

        float MinDose = MathHelper.Min(Dose), MaxDose = MathHelper.Max(Dose);
        float[] DoseInterpolationSteps = Dose.Select(d => (d - MinDose) / (MaxDose - MinDose)).ToArray();

        #endregion

        #region Load and preprocess tilt series

        progressCallback("Loading tilt series and masks...");

        decimal BinTimes = (decimal)Math.Log(SmallestAngPix / (float)dataSource.PixelSizeMean, 2.0);
        ProcessingOptionsTomoSubReconstruction OptionsDataLoad = new ProcessingOptionsTomoSubReconstruction()
        {
            PixelSize = dataSource.PixelSize,

            BinTimes = BinTimes,
            EERGroupFrames = dataSource.EERGroupFrames,
            GainPath = dataSource.GainPath,
            GainHash = "",
            GainFlipX = dataSource.GainFlipX,
            GainFlipY = dataSource.GainFlipY,
            GainTranspose = dataSource.GainTranspose,
            DefectsPath = dataSource.DefectsPath,
            DefectsHash = "",

            Dimensions = new float3((float)dataSource.DimensionsX,
                (float)dataSource.DimensionsY,
                (float)dataSource.DimensionsZ),

            Invert = true,
            NormalizeInput = true,
            NormalizeOutput = false,

            PrerotateParticles = true
        };

        VolumeDimensionsPhysical = OptionsDataLoad.DimensionsPhysical;

        Movie[] TiltMovies = null;
        Image[] TiltData = null;
        Image[] TiltMasks = null;

        Action LoadAndPreprocessTiltData = () =>
        {
            LoadMovieData(OptionsDataLoad, out TiltMovies, out TiltData, false, out _, out _);
            LoadMovieMasks(OptionsDataLoad, out TiltMasks);
            for (int z = 0; z < NTilts; z++)
            {
                EraseDirt(TiltData[z], TiltMasks[z]);
                TiltMasks[z]?.FreeDevice();

                TiltData[z].SubtractMeanGrid(new int2(1));
                TiltData[z].Bandpass(1f / LargestBox, 1f, false, 0f);

                GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                    TiltData[z].GetDevice(Intent.Write),
                    (uint)TiltData[z].ElementsReal,
                    1);

                TiltData[z].Multiply(-1f);
                //TiltData[z].Multiply(TiltMasks[z]);

                //TiltData[z].FreeDevice();
            }
        };
        LoadAndPreprocessTiltData();

        Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after loading raw data of {Name}");

        #endregion

        #region Remove particles that are not contained in any of the tilt images

        foreach (var species in allSpecies)
        {
            if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                continue;

            float Radius = species.DiameterAngstrom / 2;
            SpeciesParticles[species] = SpeciesParticles[species].Where(particle =>
            {
                float3[] CoordinatesMoving = particle.GetCoordinateSeries(DoseInterpolationSteps);
                float3[] ImagePositions = GetPositionInAllTilts(CoordinatesMoving);
                bool AnyInside = false;

                foreach (var p in ImagePositions)
                {
                    float DistX = Math.Min(p.X, ImageDimensionsPhysical.X - p.X);
                    float DistY = Math.Min(p.Y, ImageDimensionsPhysical.Y - p.Y);
                    if (DistX >= Radius && DistY >= Radius)
                    {
                        AnyInside = true;
                        break;
                    }
                }

                return AnyInside;
            }).ToArray();
        }

        #endregion

        #region Compose optimization steps based on user's requests

        var OptimizationStepsWarp = new List<(WarpOptimizationTypes Type, int Iterations, string Name)>();
        {
            WarpOptimizationTypes TranslationComponents = 0;
            if (optionsMPA.DoImageWarp)
                TranslationComponents |= WarpOptimizationTypes.ImageWarp;
            if (optionsMPA.DoVolumeWarp)
                TranslationComponents |= WarpOptimizationTypes.VolumeWarp;

            if (TranslationComponents != 0)
                OptimizationStepsWarp.Add((TranslationComponents, 10, "image & volume warping"));
        }
        {
            WarpOptimizationTypes AntisymComponents = 0;

            if (optionsMPA.DoZernike13)
                AntisymComponents |= WarpOptimizationTypes.Zernike13;
            if (optionsMPA.DoZernike5)
                AntisymComponents |= WarpOptimizationTypes.Zernike5;

            if (AntisymComponents != 0 && allSpecies.Any(s => s.ResolutionRefinement < (float)optionsMPA.MinimumCTFRefinementResolution))
                OptimizationStepsWarp.Add((AntisymComponents, 10, "antisymmetrical aberrations"));
        }
        {
            if (optionsMPA.DoAxisAngles)
                OptimizationStepsWarp.Add((WarpOptimizationTypes.AxisAngle, 6, "stage orientation"));
        }
        {
            WarpOptimizationTypes PoseComponents = 0;
            if (optionsMPA.DoParticlePoses)
            {
                PoseComponents |= WarpOptimizationTypes.ParticlePosition;
                PoseComponents |= WarpOptimizationTypes.ParticleAngle;
            }

            if (PoseComponents != 0)
                OptimizationStepsWarp.Add((PoseComponents, 10, "particle poses"));
        }
        {
            if (optionsMPA.DoMagnification)
                OptimizationStepsWarp.Add((WarpOptimizationTypes.Magnification, 4, "magnification"));
        }


        var OptimizationStepsCTF = new List<(CTFOptimizationTypes Type, int Iterations, string Name)>();
        {
            CTFOptimizationTypes DefocusComponents = 0;
            if (optionsMPA.DoDefocus)
                DefocusComponents |= CTFOptimizationTypes.Defocus;
            if (optionsMPA.DoAstigmatismDelta)
                DefocusComponents |= CTFOptimizationTypes.AstigmatismDelta;
            if (optionsMPA.DoAstigmatismAngle)
                DefocusComponents |= CTFOptimizationTypes.AstigmatismAngle;
            if (optionsMPA.DoPhaseShift)
                DefocusComponents |= CTFOptimizationTypes.PhaseShift;
            if (optionsMPA.DoCs)
                DefocusComponents |= CTFOptimizationTypes.Cs;

            if (DefocusComponents != 0)
                OptimizationStepsCTF.Add((DefocusComponents, 10, "CTF parameters"));

            CTFOptimizationTypes ZernikeComponents = 0;

            if (optionsMPA.DoZernike2)
                ZernikeComponents |= CTFOptimizationTypes.Zernike2;
            if (optionsMPA.DoZernike4)
                ZernikeComponents |= CTFOptimizationTypes.Zernike4;

            if (ZernikeComponents != 0)
                OptimizationStepsCTF.Add((ZernikeComponents, 10, "symmetrical aberrations"));
        }

        #endregion

        Dictionary<Species, float[]> GoodParticleMasks = new Dictionary<Species, float[]>();

        if (optionsMPA.NIterations > 0)
        {
            #region Resize grids

            int AngleSpatialDim = 1;

            if (optionsMPA.DoAxisAngles)
                if (GridAngleX == null || GridAngleX.Dimensions.X < AngleSpatialDim || GridAngleX.Dimensions.Z != NTilts)
                {
                    GridAngleX = GridAngleX == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NTilts)) : GridAngleX.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NTilts));
                    GridAngleY = GridAngleY == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NTilts)) : GridAngleY.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NTilts));
                    GridAngleZ = GridAngleZ == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NTilts)) : GridAngleZ.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NTilts));
                }

            // Super crude way of figuring out how many parameters can be refined into the available particle signal
            //float OverallMass = 0;
            //foreach (var species in allSpecies)
            //    if (SpeciesParticles.ContainsKey(species))
            //        OverallMass += Math.Max((float)species.MolecularWeightkDa - 100, 0) * SpeciesParticles[species].Length;
            //float NParametersMax = OverallMass / 100 * 5;
            //NParametersMax -= GridAngleX.Values.Length * 3;
            //NParametersMax /= NTilts;
            //int MovementSpatialDim = Math.Min(5, Math.Max(1, (int)Math.Round(Math.Sqrt(NParametersMax))));
            int2 MovementSpatialDims = new int2(optionsMPA.ImageWarpWidth, optionsMPA.ImageWarpHeight);
            //MovementSpatialDim = 2;

            if (optionsMPA.DoImageWarp)
                if (GridMovementX == null ||
                    GridMovementX.Dimensions.X != MovementSpatialDims.X ||
                    GridMovementX.Dimensions.Y != MovementSpatialDims.Y ||
                    GridMovementX.Dimensions.Z != NTilts)
                {
                    int3 Dims = new int3(MovementSpatialDims.X, MovementSpatialDims.Y, NTilts);
                    GridMovementX = GridMovementX == null ? new CubicGrid(Dims) : GridMovementX.Resize(Dims);
                    GridMovementY = GridMovementY == null ? new CubicGrid(Dims) : GridMovementY.Resize(Dims);
                }

            if (optionsMPA.DoVolumeWarp)
            {
                int4 DimsVolumeWarp = new int4(optionsMPA.VolumeWarpWidth,
                    optionsMPA.VolumeWarpHeight,
                    optionsMPA.VolumeWarpDepth,
                    optionsMPA.VolumeWarpLength);
                if (GridVolumeWarpX == null || GridVolumeWarpX.Dimensions != DimsVolumeWarp)
                {
                    GridVolumeWarpX = GridVolumeWarpX == null ? new LinearGrid4D(DimsVolumeWarp) : GridVolumeWarpX.Resize(DimsVolumeWarp);
                    GridVolumeWarpY = GridVolumeWarpY == null ? new LinearGrid4D(DimsVolumeWarp) : GridVolumeWarpY.Resize(DimsVolumeWarp);
                    GridVolumeWarpZ = GridVolumeWarpZ == null ? new LinearGrid4D(DimsVolumeWarp) : GridVolumeWarpZ.Resize(DimsVolumeWarp);
                }
            }

            #endregion

            #region Create species prerequisites and calculate spectral weights

            progressCallback("Calculating spectral weights...");

            Dictionary<Species, IntPtr[]> SpeciesParticleImages = new Dictionary<Species, IntPtr[]>();
            Dictionary<Species, IntPtr[]> SpeciesParticleQImages = new Dictionary<Species, IntPtr[]>();
            Dictionary<Species, float[]> SpeciesParticleDefoci = new Dictionary<Species, float[]>();
            Dictionary<Species, float[]> SpeciesContainmentMasks = new Dictionary<Species, float[]>();
            Dictionary<Species, float2[]> SpeciesParticleExtractedAt = new Dictionary<Species, float2[]>();
            Dictionary<Species, Image> SpeciesTiltWeights = new Dictionary<Species, Image>();
            Dictionary<Species, Image> SpeciesCTFWeights = new Dictionary<Species, Image>();
            Dictionary<Species, IntPtr> SpeciesParticleSubsets = new Dictionary<Species, IntPtr>();
            Dictionary<Species, (int Start, int End)> SpeciesParticleIDRanges = new Dictionary<Species, (int Start, int End)>();
            Dictionary<Species, int> SpeciesRefinementSize = new Dictionary<Species, int>();
            Dictionary<Species, int[]> SpeciesRelevantRefinementSizes = new Dictionary<Species, int[]>();
            Dictionary<Species, int> SpeciesCTFSuperresFactor = new Dictionary<Species, int>();

            Dictionary<Species, Image> CurrentWeightsDict = SpeciesTiltWeights;

            int NParticlesOverall = 0;

            float[][] AverageSpectrum1DAll = Helper.ArrayOfFunction(i => new float[128], NTilts);
            long[][] AverageSpectrum1DAllSamples = Helper.ArrayOfFunction(i => new long[128], NTilts);

            foreach (var species in allSpecies)
            {
                if (SpeciesParticles[species].Length == 0)
                    continue;

                Particle[] Particles = SpeciesParticles[species];
                int NParticles = Particles.Length;
                SpeciesParticleIDRanges.Add(species, (NParticlesOverall, NParticlesOverall + NParticles));
                NParticlesOverall += NParticles;

                int Size = species.HalfMap1Projector[GPUID].Dims.X;
                int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                int ParticleDiameterPix = (int)(species.DiameterAngstrom / (float)OptionsDataLoad.BinnedPixelSizeMean);

                int[] RelevantSizes = GetRelevantImageSizes(SizeFull, BfactorWeightingThreshold).Select(v => Math.Min(Size, v)).ToArray();

                float Radius = species.DiameterAngstrom / 2;
                float[] ContainmentMask = Helper.ArrayOfConstant(1f, NParticles * NTilts);

                #region Extract particle images

                //Image AverageRealspace = new Image(new int3(SizeFull, SizeFull, NTilts), true, true);
                Image AverageAmplitudes = new Image(new int3(SizeFull, SizeFull, NTilts), true);
                //Image ImagesRealspace = new Image(new int3(SizeFull, SizeFull, NTilts));
                Image ImagesAmplitudes = new Image(new int3(SizeFull, SizeFull, NTilts), true);

                Image ExtractResult = new Image(new int3(SizeFull, SizeFull, NTilts));
                Image ExtractResultFT = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, NTilts), true, true);
                //Image ExtractResultFTCropped = new Image(IntPtr.Zero, new int3(Size, Size, NTilts), true, true);

                int[] PlanForw = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeFull, SizeFull, 1), (uint)NTilts), 1);

                Helper.ForCPU(0, NParticles, 1, threadID => GPU.SetDevice(GPUID), (p, threadID) =>
                {
                    float3[] Coords = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                    GetImagesForOneParticle(OptionsDataLoad,
                        TiltData,
                        SizeFull,
                        Coords,
                        PlanForw[threadID],
                        ParticleDiameterPix,
                        16,
                        true,
                        ExtractResult,
                        ExtractResultFT);

                    float3[] ImageCoords = GetPositionInAllTilts(Coords);
                    for (int t = 0; t < NTilts; t++)
                    {
                        float3 Pos = ImageCoords[t];

                        float DistX = Math.Min(Pos.X, ImageDimensionsPhysical.X - Pos.X);
                        float DistY = Math.Min(Pos.Y, ImageDimensionsPhysical.Y - Pos.Y);
                        if (DistX < Radius || DistY < Radius)
                            ContainmentMask[p * NTilts + t] = 0;
                    }

                    //GPU.PadFT(ImagesFT.GetDevice(Intent.Read),
                    //          ExtractResultFTCropped.GetDevice(Intent.Write),
                    //          ImagesFT.Dims.Slice(),
                    //          ExtractResultFTCropped.Dims.Slice(),
                    //          (uint)NTilts);
                    //Image ImagesFTCropped = ImagesFT.AsPadded(new int2(Size));
                    //ImagesFT.Dispose();

                    GPU.Amplitudes(ExtractResultFT.GetDevice(Intent.Read),
                        ImagesAmplitudes.GetDevice(Intent.Write),
                        (uint)ExtractResultFT.ElementsComplex);
                    ImagesAmplitudes.Multiply(ImagesAmplitudes);
                    lock(AverageAmplitudes)
                        AverageAmplitudes.Add(ImagesAmplitudes);

                    //ImagesFTCropped.Multiply(Weights);

                    //lock (AverageRealspace)
                    //    AverageRealspace.Add(ExtractResultFT);

                    //ImagesFTCropped.Dispose();
                }, null);

                ExtractResult.Dispose();
                ExtractResultFT.Dispose();
                //ExtractResultFTCropped.Dispose();

                ImagesAmplitudes.Dispose();

                for (int i = 0; i < PlanForw.Length; i++)
                    GPU.DestroyFFTPlan(PlanForw[i]);

                //AverageRealspace.Multiply(1f / NParticles);
                //if (GPUID == 0)
                //    AverageRealspace.AsIFFT().WriteMRC("d_avgreal.mrc", true);
                //AverageRealspace.Dispose();

                //ImagesRealspace.Dispose();

                #endregion

                #region Calculate spectra

                //AverageRealspace.Multiply(1f / NParticles);
                AverageAmplitudes.Multiply(1f / NParticles);
                // if (GPUID == 0)
                //     AverageAmplitudes.WriteMRC($"d_avgamps_{species.Name}.mrc", true);

                float[][] Amps2D = AverageAmplitudes.GetHost(Intent.Read);

                for (int t = 0; t < NTilts; t++)
                {
                    Helper.ForEachElementFT(new int2(SizeFull), (x, y, xx, yy, r, angle) =>
                    {
                        int idx = (int)Math.Round(r / (SizeFull / 2) * AverageSpectrum1DAll[t].Length);
                        if (idx < AverageSpectrum1DAll[t].Length)
                        {
                            AverageSpectrum1DAll[t][idx] += Amps2D[t][y * (SizeFull / 2 + 1) + x] * NParticles;
                            AverageSpectrum1DAllSamples[t][idx] += NParticles;
                        }
                    });
                }

                AverageAmplitudes.Dispose();

                #endregion

                #region Defoci and extraction positions

                float[] Defoci = new float[NParticles * NTilts];
                float2[] ExtractedAt = new float2[NParticles * NTilts];

                for (int p = 0; p < NParticles; p++)
                {
                    float3[] Positions = GetPositionInAllTilts(Particles[p].GetCoordinateSeries(DoseInterpolationSteps));
                    for (int t = 0; t < NTilts; t++)
                    {
                        Defoci[p * NTilts + t] = Positions[t].Z;
                        ExtractedAt[p * NTilts + t] = new float2(Positions[t].X, Positions[t].Y);
                    }
                }

                #endregion

                #region Subset indices

                int[] Subsets = Particles.Select(p => p.RandomSubset).ToArray();
                IntPtr SubsetsPtr = GPU.MallocDeviceFromHostInt(Subsets, Subsets.Length);

                #endregion

                #region CTF superres factor

                CTF MaxDefocusCTF = GetTiltCTF(IndicesSortedDose[0]);
                int MinimumBoxSize = Math.Max(species.HalfMap1Projector[GPUID].Dims.X, MaxDefocusCTF.GetAliasingFreeSize(species.ResolutionRefinement, species.DiameterAngstrom / (species.ResolutionRefinement / 2)));
                float CTFSuperresFactor = (float)Math.Ceiling((float)MinimumBoxSize / species.HalfMap1Projector[GPUID].Dims.X);

                #endregion

                SpeciesParticleDefoci.Add(species, Defoci);
                SpeciesContainmentMasks.Add(species, ContainmentMask);
                SpeciesParticleExtractedAt.Add(species, ExtractedAt);
                SpeciesParticleSubsets.Add(species, SubsetsPtr);
                SpeciesRefinementSize.Add(species, Size);
                SpeciesRelevantRefinementSizes.Add(species, RelevantSizes);
                SpeciesCTFSuperresFactor.Add(species, (int)CTFSuperresFactor);

                species.HalfMap1Projector[GPUID].PutTexturesOnDevice();
                species.HalfMap2Projector[GPUID].PutTexturesOnDevice();
            }

            #region Calculate 1D PS averaged over all species and particles

            for (int t = 0; t < NTilts; t++)
            {
                for (int i = 0; i < AverageSpectrum1DAll[t].Length; i++)
                    AverageSpectrum1DAll[t][i] /= Math.Max(1, AverageSpectrum1DAllSamples[t][i]);

                float SpectrumMean = MathHelper.Mean(AverageSpectrum1DAll[t]);
                for (int i = 0; i < AverageSpectrum1DAll[t].Length; i++)
                    AverageSpectrum1DAll[t][i] /= SpectrumMean;

                for (int i = 0; i < AverageSpectrum1DAll[t].Length; i++)
                    if (AverageSpectrum1DAll[t][i] <= 0)
                    {
                        for (int j = 0; j < AverageSpectrum1DAll[t].Length; j++)
                        {
                            if (i - j >= 0 && AverageSpectrum1DAll[t][i - j] > 0)
                            {
                                AverageSpectrum1DAll[t][i] = AverageSpectrum1DAll[t][i - j];
                                break;
                            }

                            if (i + j < AverageSpectrum1DAll[t].Length && AverageSpectrum1DAll[t][i + j] > 0)
                            {
                                AverageSpectrum1DAll[t][i] = AverageSpectrum1DAll[t][i + j];
                                break;
                            }
                        }
                    }

                if (AverageSpectrum1DAll[t].Any(v => v <= 0))
                    throw new Exception("The 1D amplitude spectrum contains zeros, which it really shouldn't! Can't proceed.");
            }

            #endregion

            #region Calculate weights

            foreach (var species in allSpecies)
            {
                if (SpeciesParticles[species].Length == 0)
                    continue;

                Particle[] Particles = SpeciesParticles[species];
                int NParticles = Particles.Length;

                int Size = species.HalfMap1Projector[GPUID].Dims.X;
                int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                int ParticleDiameterPix = (int)(species.DiameterAngstrom / (float)OptionsDataLoad.BinnedPixelSizeMean);

                int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                long ElementsSliceComplex = (Size / 2 + 1) * Size;

                #region Dose weighting

                ProcessingOptionsTomoSubReconstruction OptionsWeights = new ProcessingOptionsTomoSubReconstruction()
                {
                    PixelSize = dataSource.PixelSize,

                    BinTimes = (decimal)Math.Log((species.ResolutionRefinement / 2) / (float)dataSource.PixelSizeMean, 2.0),

                    Dimensions = new float3((float)dataSource.DimensionsX,
                        (float)dataSource.DimensionsY,
                        (float)dataSource.DimensionsZ),

                    Invert = true,
                    NormalizeInput = true,
                    NormalizeOutput = false,

                    PrerotateParticles = true
                };

                Image CTFCoords = CTF.GetCTFCoords(Size, Size);
                Image Weights = GetCTFsForOneParticle(OptionsWeights, VolumeDimensionsPhysical / 2, CTFCoords, null, true, true);
                //Image VanillaWeights = Weights.GetCopy();
                CTFCoords.Dispose();

                #endregion

                #region Divide weights by 1D PS, and create a 20 A high-passed version for CTF refinement

                float[][] WeightsData = Weights.GetHost(Intent.ReadWrite);
                for (int t = 0; t < NTilts; t++)
                {
                    Helper.ForEachElementFT(new int2(Size), (x, y, xx, yy, r, angle) =>
                    {
                        if (r < Size / 2)
                        {
                            int idx = Math.Min(AverageSpectrum1DAll[t].Length - 1,
                                (int)Math.Round(r / (Size / 2) *
                                                (float)OptionsDataLoad.BinnedPixelSizeMean /
                                                (species.ResolutionRefinement / 2) *
                                                AverageSpectrum1DAll[t].Length));

                            WeightsData[t][y * (Size / 2 + 1) + x] /= AverageSpectrum1DAll[t][idx];
                        }
                        else
                        {
                            WeightsData[t][y * (Size / 2 + 1) + x] = 0;
                        }
                    });
                }

                //Weights.FreeDevice();
                // if (GPUID == 0)
                //     Weights.WriteMRC($"d_weights_{species.Name}.mrc", true);

                Image WeightsRelevantlySized = new Image(new int3(Size, Size, NTilts), true);
                for (int t = 0; t < NTilts; t++)
                    GPU.CropFTRealValued(Weights.GetDeviceSlice(t, Intent.Read),
                        WeightsRelevantlySized.GetDeviceSlice(t, Intent.Write),
                        Weights.Dims.Slice(),
                        new int3(RelevantSizes[t]).Slice(),
                        1);
                // if (GPUID == 0)
                //     WeightsRelevantlySized.WriteMRC($"d_weightsrelevant_{species.Name}.mrc", true);
                Weights.Dispose();

                Image CTFWeights = WeightsRelevantlySized.GetCopyGPU();
                float[][] CTFWeightsData = CTFWeights.GetHost(Intent.ReadWrite);
                for (int t = 0; t < CTFWeightsData.Length; t++)
                {
                    int RelevantSize = RelevantSizes[t];
                    float R20 = Size * (species.ResolutionRefinement / 2 / 20f);
                    Helper.ForEachElementFT(new int2(RelevantSize), (x, y, xx, yy, r, angle) =>
                    {
                        float Weight = 1 - Math.Max(0, Math.Min(1, R20 - r));
                        CTFWeightsData[t][y * (RelevantSize / 2 + 1) + x] *= Weight;
                    });
                }

                CTFWeights.FreeDevice();
                // if (GPUID == 0)
                //     CTFWeights.WriteMRC($"d_ctfweights_{species.Name}.mrc", true);

                #endregion

                SpeciesCTFWeights.Add(species, CTFWeights);
                SpeciesTiltWeights.Add(species, WeightsRelevantlySized);
            }

            #endregion

            // Remove original tilt image data from device, and dispose masks
            for (int t = 0; t < NTilts; t++)
            {
                if (TiltMasks != null)
                    TiltMasks[t]?.FreeDevice();
                //TiltData[t].FreeDevice();
            }

            Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after spectra estimation of {Name}");

            #endregion

            #region Tilt movie refinement

            if (optionsMPA.DoTiltMovies)
            {
                Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB before movie refinement of {Name}");

                Image StackForExport = null;
                Image StackAverage = null;
                Image AveragePlane = null;

                for (int itilt = 0; itilt < NTilts; itilt++)
                {
                    progressCallback($"Refining tilt movie {itilt + 1}/{NTilts}");

                    Movie tiltMovie = TiltMovies[itilt];
                    tiltMovie.NFrames = MapHeader.ReadFromFile(tiltMovie.DataPath).Dimensions.Z;

                    Image[] MovieData;
                    tiltMovie.LoadFrameData(OptionsDataLoad, gainRef, defectMap, out MovieData);

                    int3 StackDims = new int3(MovieData[0].Dims.X, MovieData[0].Dims.Y, MovieData.Length);
                    if (StackForExport == null || StackDims != StackForExport.Dims)
                    {
                        StackForExport?.Dispose();
                        StackForExport = new Image(IntPtr.Zero, StackDims);
                    }

                    for (int z = 0; z < MovieData.Length; z++)
                        GPU.CopyDeviceToDevice(MovieData[z].GetDevice(Intent.Read),
                            StackForExport.GetDeviceSlice(z, Intent.Write),
                            MovieData[z].ElementsReal);

                    if (StackAverage == null || StackAverage.Dims != StackForExport.Dims.Slice())
                    {
                        StackAverage?.Dispose();
                        StackAverage = new Image(IntPtr.Zero, StackForExport.Dims.Slice());
                        AveragePlane?.Dispose();
                        AveragePlane = new Image(IntPtr.Zero, StackForExport.Dims.Slice());
                    }

                    GPU.ReduceMean(StackForExport.GetDevice(Intent.Read),
                        StackAverage.GetDevice(Intent.Write),
                        (uint)StackAverage.ElementsReal,
                        (uint)StackForExport.Dims.Z,
                        1);
                    float[] AveragePlaneData = MathHelper.FitAndGeneratePlane(StackAverage.GetHost(Intent.Read)[0], new int2(StackAverage.Dims));
                    GPU.CopyHostToDevice(AveragePlaneData, AveragePlane.GetDevice(Intent.Write), AveragePlaneData.Length);

                    for (int z = 0; z < MovieData.Length; z++)
                    {
                        MovieData[z].Subtract(AveragePlane);
                        //MovieData[z].Bandpass(1f / LargestBox, 1f, false, 0f);

                        //MovieData[z].Multiply(-1f);
                        //MovieData[z].FreeDevice();
                    }

                    Dictionary<Species, Image> MovieSpeciesWeights = new Dictionary<Species, Image>();
                    foreach (var species in allSpecies)
                    {
                        if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                            continue;

                        Image Weights = new Image(IntPtr.Zero, new int3(SpeciesTiltWeights[species].Dims.X, SpeciesTiltWeights[species].Dims.Y, MovieData.Length), true);
                        for (int i = 0; i < MovieData.Length; i++)
                        {
                            GPU.CopyDeviceToDevice((species.ResolutionRefinement < 10 ? SpeciesCTFWeights : SpeciesTiltWeights)[species].GetDeviceSlice(itilt, Intent.Read),
                                Weights.GetDeviceSlice(i, Intent.Write),
                                Weights.ElementsSliceReal);
                        }

                        MovieSpeciesWeights.Add(species, Weights);
                    }

                    PerformMultiParticleRefinementOneTiltMovie(workingDirectory,
                        optionsMPA,
                        allSpecies,
                        dataSource,
                        tiltMovie,
                        MovieData,
                        itilt,
                        SpeciesParticles,
                        SpeciesParticleSubsets,
                        SpeciesParticleIDRanges,
                        SpeciesContainmentMasks,
                        SpeciesRefinementSize,
                        SpeciesRelevantRefinementSizes,
                        MovieSpeciesWeights,
                        SpeciesCTFSuperresFactor);

                    foreach (var pair in MovieSpeciesWeights)
                        pair.Value.Dispose();

                    foreach (var frame in MovieData)
                        frame.Dispose();

                    tiltMovie.ExportMovie(StackForExport, tiltMovie.OptionsMovieExport);

                    tiltMovie.SaveMeta();
                }

                StackForExport.Dispose();
                StackAverage.Dispose();
                AveragePlane.Dispose();

                for (int t = 0; t < NTilts; t++)
                    TiltData[t].FreeDevice();

                LoadAndPreprocessTiltData();

                for (int t = 0; t < NTilts; t++)
                {
                    if (TiltMasks != null)
                        TiltMasks[t]?.FreeDevice();
                    //TiltData[t].FreeDevice();
                }

                Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after movie refinement of {Name}");
            }

            #endregion

            #region Allocate pinned host memory for extracted particle images

            foreach (var species in allSpecies)
            {
                int NParticles = SpeciesParticles[species].Length;
                if (NParticles == 0)
                    continue;

                int Size = species.HalfMap1Projector[GPUID].Dims.X;
                int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                IntPtr[] ImagesFTPinned = Helper.ArrayOfFunction(t =>
                {
                    long Footprint = (new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles;
                    if (optionsMPA.UseHostMemory)
                        return GPU.MallocHostPinned(Footprint);
                    else
                        return GPU.MallocDevice(Footprint);
                }, NTilts);

                IntPtr[] ImagesFTQPinned = null;
                if (species.DoEwald)
                    ImagesFTQPinned = Helper.ArrayOfFunction(t => GPU.MallocDevice((new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles), NTilts);
                GPU.CheckGPUExceptions();

                SpeciesParticleImages.Add(species, ImagesFTPinned);
                if (species.DoEwald)
                    SpeciesParticleQImages.Add(species, ImagesFTQPinned);
            }

            #endregion

            #region Helper functions

            Action<bool> ReextractPaddedParticles = (CorrectBeamTilt) =>
            {
                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                int BatchSize = optionsMPA.BatchSize;

                foreach (var species in allSpecies)
                {
                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;

                    if (NParticles == 0 || !SpeciesCTFSuperresFactor.ContainsKey(species))
                        continue;

                    int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                    int[] SizesRelevant = SpeciesRelevantRefinementSizes[species];

                    float AngPixRefine = species.ResolutionRefinement / 2;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                    float2[] ExtractedAt = SpeciesParticleExtractedAt[species];

                    Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);
                    Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefineSuper);
                    Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);
                    //GammaCorrection.WriteMRC("d_gamma.mrc", true);

                    Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize));
                    Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                    Image ExtractedCroppedFTRelevantSize = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                    Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                    //Image Average = new Image(new int3(SizeRefine, SizeRefine, BatchSize));

                    int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                    int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                    int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                    if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                    {
                        Console.WriteLine($"SizeFullSuper = {SizeFullSuper}, BatchSize = {BatchSize}, free memory = {GPU.GetFreeMemory(GPUID)}");
                        throw new Exception("No FFT plans created!");
                    }

                    bool[] PQReverse = { species.EwaldReverse, !species.EwaldReverse };
                    IntPtr[][] PQStorage = species.DoEwald ? new[] { SpeciesParticleImages[species], SpeciesParticleQImages[species] } : new[] { SpeciesParticleImages[species] };

                    for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                    {
                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                            float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                            for (int t = 0; t < NTilts; t++)
                            {
                                float3[] CoordinatesTilt = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                    CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];

                                float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);

                                float[] Defoci = new float[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p] = ImageCoords[p].Z;
                                    ExtractedAt[(batchStart + p) * NTilts + t] = new float2(ImageCoords[p]);
                                }

                                GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    TiltData[t].Dims.Slice(),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    Helper.ToInterleaved(ExtractOrigins),
                                    true,
                                    (uint)CurBatch);

                                GPU.FFT(Extracted.GetDevice(Intent.Read),
                                    ExtractedFT.GetDevice(Intent.Write),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    (uint)CurBatch,
                                    PlanForwSuper);

                                ExtractedFT.ShiftSlices(ResidualShifts);
                                ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    (uint)CurBatch);

                                if (CorrectBeamTilt)
                                    GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                        PhaseCorrection.GetDevice(Intent.Read),
                                        Extracted.GetDevice(Intent.Write),
                                        PhaseCorrection.ElementsSliceComplex,
                                        (uint)CurBatch);

                                if (species.DoEwald)
                                {
                                    GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, PQReverse[iewald], ExtractedCTF, true);

                                    GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                        ExtractedCTF.GetDevice(Intent.Read),
                                        ExtractedFT.GetDevice(Intent.Write),
                                        ExtractedCTF.ElementsComplex,
                                        1);
                                }
                                else
                                {
                                    GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                    GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                        ExtractedCTF.GetDevice(Intent.Read),
                                        ExtractedFT.GetDevice(Intent.Write),
                                        ExtractedCTF.ElementsComplex,
                                        1);
                                }

                                GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    (uint)CurBatch,
                                    PlanBackSuper,
                                    false);

                                GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                    ExtractedCropped.GetDevice(Intent.Write),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    new int3(SizeRefine, SizeRefine, 1),
                                    (uint)CurBatch);

                                GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                    ExtractedCropped.GetDevice(Intent.Write),
                                    ExtractedCropped.Dims.Slice(),
                                    ParticleDiameterPix / 2f,
                                    16 * AngPixExtract / AngPixRefine,
                                    true,
                                    (uint)CurBatch);

                                //Average.Add(ExtractedCropped);

                                GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                    ExtractedCroppedFT.GetDevice(Intent.Write),
                                    new int3(SizeRefine, SizeRefine, 1),
                                    (uint)CurBatch,
                                    PlanForw);

                                ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                    ExtractedCroppedFTRelevantSize.GetDevice(Intent.Write),
                                    new int3(SizeRefine).Slice(),
                                    new int3(SizesRelevant[t]).Slice(),
                                    (uint)CurBatch);

                                GPU.CopyDeviceToHostPinned(ExtractedCroppedFTRelevantSize.GetDevice(Intent.Read),
                                    new IntPtr((long)PQStorage[iewald][t] + (new int3(SizesRelevant[t]).Slice().ElementsFFT()) * 2 * batchStart * sizeof(float)),
                                    (new int3(SizesRelevant[t]).Slice().ElementsFFT()) * 2 * CurBatch);
                            }
                        }
                    }

                    //Average.WriteMRC("d_average.mrc", true);
                    //Average.Dispose();

                    CoordsCTF.Dispose();
                    PhaseCorrection.Dispose();
                    GammaCorrection.Dispose();
                    Extracted.Dispose();
                    ExtractedFT.Dispose();
                    ExtractedCropped.Dispose();
                    ExtractedCroppedFT.Dispose();
                    ExtractedCroppedFTRelevantSize.Dispose();
                    ExtractedCTF.Dispose();

                    GPU.DestroyFFTPlan(PlanForwSuper);
                    GPU.DestroyFFTPlan(PlanBackSuper);
                    GPU.DestroyFFTPlan(PlanForw);
                }

                //foreach (var image in TiltData)
                //    image.FreeDevice();

                GPU.CheckGPUExceptions();
            };

            Func<float2[]> GetRawShifts = () =>
            {
                float2[] Result = new float2[NParticlesOverall * NTilts];

                foreach (var species in allSpecies)
                {
                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    float SpeciesAngPix = species.ResolutionRefinement / 2;
                    if (NParticles == 0)
                        continue;

                    int Offset = SpeciesParticleIDRanges[species].Start;

                    float3[] ParticlePositions = new float3[NParticles * NTilts];
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                        for (int t = 0; t < NTilts; t++)
                            ParticlePositions[p * NTilts + t] = Positions[t];
                    }

                    float3[] ParticlePositionsProjected = GetPositionInAllTilts(ParticlePositions);
                    float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                    for (int p = 0; p < NParticles; p++)
                    for (int t = 0; t < NTilts; t++)
                        Result[(Offset + p) * NTilts + t] = (new float2(ParticlePositionsProjected[p * NTilts + t]) - ParticleExtractedAt[p * NTilts + t]);
                }

                return Result;
            };

            Func<float2, Species, float[]> GetRawCCSpecies = (shiftBias, Species) =>
            {
                Particle[] Particles = SpeciesParticles[Species];

                int NParticles = Particles.Length;
                float AngPixRefine = Species.ResolutionRefinement / 2;

                float[] SpeciesResult = new float[NParticles * NTilts * 3];
                if (NParticles == 0)
                    return SpeciesResult;

                float[] SpeciesResultQ = new float[NParticles * NTilts * 3];

                float3[] ParticlePositions = new float3[NParticles * NTilts];
                float3[] ParticleAngles = new float3[NParticles * NTilts];
                for (int p = 0; p < NParticles; p++)
                {
                    float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);
                    float3[] Angles = Particles[p].GetAngleSeries(DoseInterpolationSteps);

                    for (int t = 0; t < NTilts; t++)
                    {
                        ParticlePositions[p * NTilts + t] = Positions[t];
                        ParticleAngles[p * NTilts + t] = Angles[t];
                    }
                }

                float3[] ParticlePositionsProjected = GetPositionInAllTilts(ParticlePositions);
                float3[] ParticleAnglesInTilts = GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles);

                float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[Species];
                float2[] ParticleShifts = new float2[NTilts * NParticles];
                for (int p = 0; p < NParticles; p++)
                for (int t = 0; t < NTilts; t++)
                    ParticleShifts[p * NTilts + t] = (new float2(ParticlePositionsProjected[p * NTilts + t]) - ParticleExtractedAt[p * NTilts + t] + shiftBias) / AngPixRefine;

                int[] RelevantSizes = SpeciesRelevantRefinementSizes[Species];

                int SizeRefine = Species.HalfMap1Projector[GPUID].Dims.X;
                int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;

                Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NTilts), true, true);
                for (int t = 0; t < NTilts; t++)
                    GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                        PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                        PhaseCorrection.Dims.Slice(),
                        new int3(RelevantSizes[t]).Slice(),
                        1);

                GPU.MultiParticleDiff(SpeciesResult,
                    SpeciesParticleImages[Species],
                    SizeRefine,
                    RelevantSizes,
                    Helper.ToInterleaved(ParticleShifts),
                    Helper.ToInterleaved(ParticleAnglesInTilts),
                    MagnificationCorrection.ToVec(),
                    (Species.ResolutionRefinement < 8 ? SpeciesCTFWeights : SpeciesTiltWeights)[Species].GetDevice(Intent.Read),
                    PhaseCorrectionAll.GetDevice(Intent.Read),
                    Species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize) : 0,
                    Species.CurrentMaxShellRefinement,
                    new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                    new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                    Species.HalfMap1Projector[GPUID].Oversampling,
                    Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                    SpeciesParticleSubsets[Species],
                    NParticles,
                    NTilts);

                if (Species.DoEwald)
                    GPU.MultiParticleDiff(SpeciesResultQ,
                        SpeciesParticleQImages[Species],
                        SizeRefine,
                        RelevantSizes,
                        Helper.ToInterleaved(ParticleShifts),
                        Helper.ToInterleaved(ParticleAnglesInTilts),
                        MagnificationCorrection.ToVec(),
                        (Species.ResolutionRefinement < 8 ? SpeciesCTFWeights : SpeciesTiltWeights)[Species].GetDevice(Intent.Read),
                        PhaseCorrectionAll.GetDevice(Intent.Read),
                        -CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize),
                        Species.CurrentMaxShellRefinement,
                        new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                        new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                        Species.HalfMap1Projector[GPUID].Oversampling,
                        Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                        SpeciesParticleSubsets[Species],
                        NParticles,
                        NTilts);

                PhaseCorrection.Dispose();
                PhaseCorrectionAll.Dispose();

                if (Species.DoEwald)
                    for (int i = 0; i < SpeciesResult.Length; i++)
                        SpeciesResult[i] += SpeciesResultQ[i];

                float[] ContainmentMask = SpeciesContainmentMasks[Species];
                for (int i = 0; i < NParticles * NTilts; i++)
                {
                    SpeciesResult[i * 3 + 0] *= ContainmentMask[i];
                    SpeciesResult[i * 3 + 1] *= ContainmentMask[i];
                    SpeciesResult[i * 3 + 2] *= ContainmentMask[i];
                }

                return SpeciesResult;
            };

            Func<float2, float[]> GetRawCC = (shiftBias) =>
            {
                float[] Result = new float[NParticlesOverall * NTilts * 3];

                for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                {
                    Species Species = allSpecies[ispecies];
                    Particle[] Particles = SpeciesParticles[Species];

                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    float[] SpeciesResult = GetRawCCSpecies(shiftBias, Species);

                    int Offset = SpeciesParticleIDRanges[Species].Start * NTilts * 3;
                    Array.Copy(SpeciesResult, 0, Result, Offset, SpeciesResult.Length);
                }

                GPU.CheckGPUExceptions();
                //Console.WriteLine(GPU.GetFreeMemory(GPUID));

                return Result;
            };

            Func<double[]> GetPerTiltCC = () =>
            {
                double[] Result = new double[NTilts * 3];
                float[] RawResult = GetRawCC(new float2(0));

                for (int p = 0; p < NParticlesOverall; p++)
                for (int t = 0; t < NTilts; t++)
                {
                    Result[t * 3 + 0] += RawResult[(p * NTilts + t) * 3 + 0];
                    Result[t * 3 + 1] += RawResult[(p * NTilts + t) * 3 + 1];
                    Result[t * 3 + 2] += RawResult[(p * NTilts + t) * 3 + 2];
                }

                Result = Helper.ArrayOfFunction(t => Result[t * 3 + 0] / Math.Max(1e-10, Math.Sqrt(Result[t * 3 + 1] * Result[t * 3 + 2])) * 100 * NParticlesOverall, NTilts);

                return Result;
            };

            Func<double[]> GetPerParticleCC = () =>
            {
                double[] Result = new double[NParticlesOverall * 3];
                float[] RawResult = GetRawCC(new float2(0));

                for (int p = 0; p < NParticlesOverall; p++)
                for (int t = 0; t < NTilts; t++)
                {
                    Result[p * 3 + 0] += RawResult[(p * NTilts + t) * 3 + 0];
                    Result[p * 3 + 1] += RawResult[(p * NTilts + t) * 3 + 1];
                    Result[p * 3 + 2] += RawResult[(p * NTilts + t) * 3 + 2];
                }

                Result = Helper.ArrayOfFunction(p => Result[p * 3 + 0] / Math.Max(1e-10, Math.Sqrt(Result[p * 3 + 1] * Result[p * 3 + 2])) * 100 * NTilts, NParticlesOverall);

                return Result;
            };

            Func<Species, double[]> GetPerParticleCCSpecies = (species) =>
            {
                Particle[] Particles = SpeciesParticles[species];
                int NParticles = Particles.Length;

                double[] Result = new double[NParticles * 3];
                float[] RawResult = GetRawCCSpecies(new float2(0), species);

                for (int p = 0; p < NParticles; p++)
                for (int t = 0; t < NTilts; t++)
                {
                    Result[p * 3 + 0] += RawResult[(p * NTilts + t) * 3 + 0];
                    Result[p * 3 + 1] += RawResult[(p * NTilts + t) * 3 + 1];
                    Result[p * 3 + 2] += RawResult[(p * NTilts + t) * 3 + 2];
                }

                Result = Helper.ArrayOfFunction(p => Result[p * 3 + 0] /
                                                     Math.Max(1e-10, Math.Sqrt(Result[p * 3 + 1] * Result[p * 3 + 2])) *
                                                     100 * NTilts, NParticles);

                return Result;
            };

            Func<(float[] xp, float[] xm, float[] yp, float[] ym, float delta2)> GetRawShiftGradients = () =>
            {
                float Delta = 0.1f;
                float Delta2 = Delta * 2;

                float[] h_ScoresXP = GetRawCC(float2.UnitX * Delta);
                float[] h_ScoresXM = GetRawCC(-float2.UnitX * Delta);
                float[] h_ScoresYP = GetRawCC(float2.UnitY * Delta);
                float[] h_ScoresYM = GetRawCC(-float2.UnitY * Delta);

                //for (int i = 0; i < Result.Length; i++)
                //    Result[i] = new float2((h_ScoresXP[i] - h_ScoresXM[i]) / Delta2 * 100,
                //                           (h_ScoresYP[i] - h_ScoresYM[i]) / Delta2 * 100);

                return (h_ScoresXP, h_ScoresXM, h_ScoresYP, h_ScoresYM, Delta2);
            };

            #endregion

            #region BFGS prerequisites

            float2[] OriginalImageWarps = Helper.ArrayOfFunction(t => new float2(GridMovementX.Values[t], GridMovementY.Values[t]), GridMovementX.Values.Length);
            float3[] OriginalVolumeWarps = Helper.ArrayOfFunction(t => new float3(GridVolumeWarpX.Values[t], GridVolumeWarpY.Values[t], GridVolumeWarpZ.Values[t]), GridVolumeWarpX.Values.Length);

            float[] OriginalAngleX = GridAngleX.Values.ToArray();
            float[] OriginalAngleY = GridAngleY.Values.ToArray();
            float[] OriginalAngleZ = GridAngleZ.Values.ToArray();

            float4[] OriginalTiltCTFs = Helper.ArrayOfFunction(t => new float4(GridCTFDefocus.Values[t],
                GridCTFDefocusDelta.Values[t],
                GridCTFDefocusAngle.Values[t],
                GridCTFPhase.Values[t]), NTilts);

            float[] OriginalParamsCTF =
            {
                (float)CTF.Cs,
            };

            CTFOptimizationTypes[] CTFStepTypes =
            {
                CTFOptimizationTypes.Defocus,
                CTFOptimizationTypes.AstigmatismDelta,
                CTFOptimizationTypes.AstigmatismAngle,
                CTFOptimizationTypes.PhaseShift,
                CTFOptimizationTypes.Zernike2,
                CTFOptimizationTypes.Zernike2,
                CTFOptimizationTypes.Zernike2,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.Cs,
            };

            float[] OriginalZernikeOdd = CTF.ZernikeCoeffsOdd.ToList().ToArray();
            float[] OriginalZernikeEven = CTF.ZernikeCoeffsEven.ToList().ToArray();

            //float2 OriginalBeamTilt = CTF.BeamTilt;
            Matrix2 OriginalMagnification = MagnificationCorrection.GetCopy();

            float3[][] OriginalParticlePositions = allSpecies.Select(s => Helper.Combine(SpeciesParticles[s].Select(p => p.Coordinates))).ToArray();
            float3[][] OriginalParticleAngles = allSpecies.Select(s => Helper.Combine(SpeciesParticles[s].Select(p => p.Angles))).ToArray();

            int BFGSIterations = 0;
            WarpOptimizationTypes CurrentOptimizationTypeWarp = 0;
            CTFOptimizationTypes CurrentOptimizationTypeCTF = 0;

            double[] InitialParametersWarp = new double[GridMovementX.Values.Length * 2 +
                                                        GridVolumeWarpX.Values.Length * 3 +
                                                        GridAngleX.Values.Length * 3 +
                                                        OriginalParticlePositions.Select(a => a.Length).Sum() * 3 +
                                                        OriginalParticleAngles.Select(a => a.Length).Sum() * 3 +
                                                        CTF.ZernikeCoeffsOdd.Length +
                                                        4];
            double[] InitialParametersDefocus = new double[NTilts * 4 +
                                                           CTF.ZernikeCoeffsEven.Length +
                                                           OriginalParamsCTF.Length];

            #endregion

            #region Set parameters from vector

            Action<double[], TiltSeries, bool> SetWarpFromVector = (input, series, setParticles) =>
            {
                int Offset = 0;

                float[] MovementXData = new float[GridMovementX.Values.Length];
                float[] MovementYData = new float[GridMovementX.Values.Length];
                for (int i = 0; i < MovementXData.Length; i++)
                {
                    MovementXData[i] = OriginalImageWarps[i].X + (float)input[Offset + i];
                    MovementYData[i] = OriginalImageWarps[i].Y + (float)input[Offset + MovementXData.Length + i];
                }

                series.GridMovementX = new CubicGrid(GridMovementX.Dimensions, MovementXData);
                series.GridMovementY = new CubicGrid(GridMovementY.Dimensions, MovementYData);

                Offset += MovementXData.Length * 2;

                float[] VolumeXData = new float[GridVolumeWarpX.Values.Length];
                float[] VolumeYData = new float[GridVolumeWarpX.Values.Length];
                float[] VolumeZData = new float[GridVolumeWarpX.Values.Length];
                int GridVolumeSlice = (int)GridVolumeWarpX.Dimensions.ElementsSlice();
                for (int i = 0; i < VolumeXData.Length; i++)
                {
                    if (i < GridVolumeSlice)
                    {
                        VolumeXData[i] = OriginalVolumeWarps[i].X;
                        VolumeYData[i] = OriginalVolumeWarps[i].Y;
                        VolumeZData[i] = OriginalVolumeWarps[i].Z;
                    }
                    else
                    {
                        VolumeXData[i] = OriginalVolumeWarps[i].X + (float)input[Offset + i];
                        VolumeYData[i] = OriginalVolumeWarps[i].Y + (float)input[Offset + VolumeXData.Length + i];
                        VolumeZData[i] = OriginalVolumeWarps[i].Z + (float)input[Offset + VolumeXData.Length + VolumeYData.Length + i];
                    }
                }

                series.GridVolumeWarpX = new LinearGrid4D(GridVolumeWarpX.Dimensions, VolumeXData);
                series.GridVolumeWarpY = new LinearGrid4D(GridVolumeWarpY.Dimensions, VolumeYData);
                series.GridVolumeWarpZ = new LinearGrid4D(GridVolumeWarpZ.Dimensions, VolumeZData);

                Offset += VolumeXData.Length * 3;

                float[] AngleXData = new float[GridAngleX.Values.Length];
                float[] AngleYData = new float[GridAngleY.Values.Length];
                float[] AngleZData = new float[GridAngleZ.Values.Length];
                for (int i = 0; i < AngleXData.Length; i++)
                {
                    AngleXData[i] = OriginalAngleX[i] + (float)input[Offset + i];
                    AngleYData[i] = OriginalAngleY[i] + (float)input[Offset + AngleXData.Length + i];
                    AngleZData[i] = OriginalAngleZ[i] + (float)input[Offset + AngleXData.Length * 2 + i];
                }

                series.GridAngleX = new CubicGrid(GridAngleX.Dimensions, AngleXData);
                series.GridAngleY = new CubicGrid(GridAngleY.Dimensions, AngleYData);
                series.GridAngleZ = new CubicGrid(GridAngleZ.Dimensions, AngleZData);

                Offset += AngleXData.Length * 3;

                if (setParticles)
                {
                    for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                    {
                        Particle[] Particles = SpeciesParticles[allSpecies[ispecies]];

                        int ResCoords = allSpecies[ispecies].TemporalResolutionMovement;

                        for (int p = 0; p < Particles.Length; p++)
                        {
                            for (int ic = 0; ic < ResCoords; ic++)
                            {
                                Particles[p].Coordinates[ic] = OriginalParticlePositions[ispecies][p * ResCoords + ic] + new float3((float)input[Offset + (p * 6 + 0) * ResCoords + ic],
                                    (float)input[Offset + (p * 6 + 1) * ResCoords + ic],
                                    (float)input[Offset + (p * 6 + 2) * ResCoords + ic]);
                                Particles[p].Angles[ic] = OriginalParticleAngles[ispecies][p * ResCoords + ic] + new float3((float)input[Offset + (p * 6 + 3) * ResCoords + ic],
                                    (float)input[Offset + (p * 6 + 4) * ResCoords + ic],
                                    (float)input[Offset + (p * 6 + 5) * ResCoords + ic]);
                            }
                        }

                        Offset += OriginalParticlePositions[ispecies].Length * 6;
                    }
                }
                else
                {
                    Offset += OriginalParticlePositions.Select(a => a.Length).Sum() * 6;
                }

                //CTF.BeamTilt = OriginalBeamTilt + new float2((float)input[input.Length - 5],
                //                                             (float)input[input.Length - 4]);

                for (int icoeff = 0; icoeff < CTF.ZernikeCoeffsOdd.Length; icoeff++)
                    CTF.ZernikeCoeffsOdd[icoeff] = OriginalZernikeOdd[icoeff] + (float)input[Offset + icoeff];

                Offset += CTF.ZernikeCoeffsOdd.Length;

                MagnificationCorrection = OriginalMagnification + new Matrix2((float)input[input.Length - 4] / 100,
                    (float)input[input.Length - 3] / 100,
                    (float)input[input.Length - 2] / 100,
                    (float)input[input.Length - 1] / 100);

                // MagnificationCorrection follows a different, weird convention.
                // .x and .y define the X and Y axes of a scaling matrix, rotated by -.z
                // Scaling .x up means the pixel size along that axis is smaller, thus a negative DeltaPercent
                //CTF.PixelSizeDeltaPercent = -(decimal)(MagnificationCorrection.X - (MagnificationCorrection.X + MagnificationCorrection.Y) / 2);
                //CTF.PixelSizeAngle = (decimal)(-MagnificationCorrection.Z * Helper.ToDeg);
            };

            Action<double[]> SetDefocusFromVector = input =>
            {
                int Offset = 0;

                float[] DefocusValues = Helper.ArrayOfFunction(t => OriginalTiltCTFs[t].X + (float)input[t * 4 + 0] * 0.1f, NTilts);
                float[] AstigmatismValues = Helper.ArrayOfFunction(t => OriginalTiltCTFs[t].Y + (float)input[t * 4 + 1] * 0.1f, NTilts);
                float[] AngleValues = Helper.ArrayOfFunction(t => OriginalTiltCTFs[t].Z + (float)input[t * 4 + 2] * 36, NTilts);
                float[] PhaseValues = Helper.ArrayOfFunction(t => OriginalTiltCTFs[t].W + (float)input[t * 4 + 3] * 36, NTilts);

                GridCTFDefocus = new CubicGrid(new int3(1, 1, NTilts), DefocusValues);
                GridCTFDefocusDelta = new CubicGrid(new int3(1, 1, NTilts), AstigmatismValues);
                GridCTFDefocusAngle = new CubicGrid(new int3(1, 1, NTilts), AngleValues);
                GridCTFPhase = new CubicGrid(new int3(1, 1, NTilts), PhaseValues);

                Offset += NTilts * 4;

                {
                    float[] ValuesZernike = new float[CTF.ZernikeCoeffsEven.Length];
                    for (int i = 0; i < ValuesZernike.Length; i++)
                        ValuesZernike[i] = OriginalZernikeEven[i] + (float)input[Offset + i];

                    CTF.ZernikeCoeffsEven = ValuesZernike;
                    Offset += CTF.ZernikeCoeffsEven.Length;
                }

                CTF.Cs = (decimal)(OriginalParamsCTF[0] + input[input.Length - 1]);
                //CTF.PixelSizeDeltaPercent = (decimal)(OriginalParamsCTF[1] + input[input.Length - 2] * 0.1f);
                //CTF.PixelSizeAngle = (decimal)(OriginalParamsCTF[2] + input[input.Length - 1] * 36);
            };

            #endregion

            #region Wiggle weights

            progressCallback("Precomputing gradient weights...");

            int NWiggleDifferentiable = GridMovementX.Values.Length +
                                        GridMovementY.Values.Length +
                                        GridVolumeWarpX.Values.Length +
                                        GridVolumeWarpY.Values.Length +
                                        GridVolumeWarpZ.Values.Length;
            (int[] indices, float2[] weights)[] AllWiggleWeights = new (int[] indices, float2[] weights)[NWiggleDifferentiable];

            if (optionsMPA.DoImageWarp || optionsMPA.DoVolumeWarp)
            {
                TiltSeries[] ParallelSeriesCopies = Helper.ArrayOfFunction(i => new TiltSeries(this.Path), 16);

                Dictionary<Species, float3[]> AllParticlePositions = new Dictionary<Species, float3[]>();
                foreach (var species in allSpecies)
                {
                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    int Offset = SpeciesParticleIDRanges[species].Start;

                    float3[] ParticlePositions = new float3[NParticles * NTilts];
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                        for (int t = 0; t < NTilts; t++)
                            ParticlePositions[p * NTilts + t] = Positions[t];
                    }

                    AllParticlePositions.Add(species, ParticlePositions);
                }

                Helper.ForCPU(0, NWiggleDifferentiable, ParallelSeriesCopies.Length, (threadID) =>
                    {
                        ParallelSeriesCopies[threadID].VolumeDimensionsPhysical = VolumeDimensionsPhysical;
                        ParallelSeriesCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                        ParallelSeriesCopies[threadID].SizeRoundingFactors = SizeRoundingFactors;
                    },
                    (iwiggle, threadID) =>
                    {
                        double[] WiggleParams = new double[InitialParametersWarp.Length];
                        WiggleParams[iwiggle] = 1;
                        SetWarpFromVector(WiggleParams, ParallelSeriesCopies[threadID], false);

                        float2[] RawShifts = new float2[NParticlesOverall * NTilts];
                        foreach (var species in allSpecies)
                        {
                            Particle[] Particles = SpeciesParticles[species];
                            int NParticles = Particles.Length;
                            if (NParticles == 0)
                                continue;

                            float[] ContainmentMask = SpeciesContainmentMasks[species];
                            int Offset = SpeciesParticleIDRanges[species].Start;

                            float3[] ParticlePositions = AllParticlePositions[species];

                            float3[] ParticlePositionsProjected = ParallelSeriesCopies[threadID].GetPositionInAllTilts(ParticlePositions);
                            float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                            for (int p = 0; p < NParticles; p++)
                            for (int t = 0; t < NTilts; t++)
                                RawShifts[(Offset + p) * NTilts + t] = (new float2(ParticlePositionsProjected[p * NTilts + t]) - ParticleExtractedAt[p * NTilts + t]) * ContainmentMask[p * NTilts + t];
                        }

                        List<int> Indices = new List<int>();
                        List<float2> Weights = new List<float2>();
                        for (int i = 0; i < RawShifts.Length; i++)
                        {
                            if (RawShifts[i].LengthSq() > 1e-6f)
                            {
                                Indices.Add(i);
                                Weights.Add(RawShifts[i]);

                                if (Math.Abs(RawShifts[i].X) > 1.5f)
                                {
                                    throw new Exception("");
                                }
                            }
                        }

                        AllWiggleWeights[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                    }, null);
            }

            #endregion

            double[] OldInput = new double[0];
            double[] OldGradient = new double[0];

            #region Loss and gradient functions for warping

            Func<double[], double> WarpEval = input =>
            {
                SetWarpFromVector(input, this, true);

                float[] RawCC = GetRawCC(new float2(0));
                double SumAB = 0, SumA2 = 0, SumB2 = 0;
                for (int p = 0; p < NParticlesOverall; p++)
                {
                    for (int t = 0; t < NTilts; t++)
                    {
                        SumAB += RawCC[(p * NTilts + t) * 3 + 0];
                        SumA2 += RawCC[(p * NTilts + t) * 3 + 1];
                        SumB2 += RawCC[(p * NTilts + t) * 3 + 2];
                    }
                }

                double Score = SumAB / Math.Max(1e-10, Math.Sqrt(SumA2 * SumB2)) * NParticlesOverall * NTilts * 100;

                //double[] TiltScores = GetPerTiltDiff2();
                //double Score = TiltScores.Sum();

                Console.WriteLine(Score);

                return Score;
            };

            Func<double[], double[]> WarpGrad = input =>
            {
                double Delta = 0.025;
                double Delta2 = Delta * 2;

                double[] Result = new double[input.Length];

                if (BFGSIterations-- <= 0)
                    return Result;

                if (MathHelper.AllEqual(input, OldInput))
                    return OldGradient;

                int Offset = 0;

                if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ImageWarp) != 0 || // GridMovementXY
                    (CurrentOptimizationTypeWarp & WarpOptimizationTypes.VolumeWarp) != 0) // GridVolumeWarpXYZ
                {
                    SetWarpFromVector(input, this, true);
                    (var XP, var XM, var YP, var YM, var Delta2Movement) = GetRawShiftGradients();

                    int NImageWarp = GridMovementX.Values.Length * 2;

                    if (true)
                        Parallel.For(0, AllWiggleWeights.Length, iwiggle =>
                        {
                            if (iwiggle < NImageWarp && (CurrentOptimizationTypeWarp & WarpOptimizationTypes.ImageWarp) == 0)
                                return;
                            if (iwiggle >= NImageWarp && (CurrentOptimizationTypeWarp & WarpOptimizationTypes.VolumeWarp) == 0)
                                return;

                            double SumGrad = 0;
                            double SumWeights = 0;
                            double SumWeightsGrad = 0;

                            int[] Indices = AllWiggleWeights[iwiggle].indices;
                            float2[] Weights = AllWiggleWeights[iwiggle].weights;

                            for (int i = 0; i < Indices.Length; i++)
                            {
                                int id = Indices[i];

                                SumWeights += Math.Abs(Weights[i].X) * Math.Sqrt(XP[id * 3 + 1] + XM[id * 3 + 1]) +
                                              Math.Abs(Weights[i].Y) * Math.Sqrt(YP[id * 3 + 1] + YM[id * 3 + 1]);
                                SumWeightsGrad += Math.Abs(Weights[i].X) + Math.Abs(Weights[i].Y);

                                double GradX = (XP[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(XP[id * 3 + 1] * XP[id * 3 + 2])) -
                                                XM[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(XM[id * 3 + 1] * XM[id * 3 + 2]))) / Delta2Movement;
                                double GradY = (YP[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(YP[id * 3 + 1] * YP[id * 3 + 2])) -
                                                YM[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(YM[id * 3 + 1] * YM[id * 3 + 2]))) / Delta2Movement;

                                SumGrad += Weights[i].X * Math.Sqrt(XP[id * 3 + 1] + XM[id * 3 + 1]) * GradX;
                                SumGrad += Weights[i].Y * Math.Sqrt(YP[id * 3 + 1] + YM[id * 3 + 1]) * GradY;
                            }

                            Result[Offset + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                        });

                    if (false)
                        for (int i = 0; i < NImageWarp; i++)
                        {
                            double[] InputPlus = input.ToArray();
                            InputPlus[Offset + i] += Delta;

                            double ScorePlus = WarpEval(InputPlus);

                            double[] InputMinus = input.ToArray();
                            InputMinus[Offset + i] -= Delta;

                            double ScoreMinus = WarpEval(InputMinus);

                            Result[Offset + i] = (ScorePlus - ScoreMinus) / Delta2;
                        }
                }

                Offset += AllWiggleWeights.Length;


                if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.AxisAngle) != 0) // GridAngleX, Y and Z
                {
                    int SliceElements = (int)GridAngleX.Dimensions.ElementsSlice();

                    for (int a = 0; a < 3; a++)
                    {
                        for (int i = 0; i < SliceElements; i++)
                        {
                            double[] InputPlus = input.ToArray();
                            for (int t = 0; t < NTilts; t++)
                                InputPlus[Offset + t * SliceElements + i] += Delta;

                            SetWarpFromVector(InputPlus, this, true);
                            double[] ScoresPlus = GetPerTiltCC();

                            double[] InputMinus = input.ToArray();
                            for (int t = 0; t < NTilts; t++)
                                InputMinus[Offset + t * SliceElements + i] -= Delta;

                            SetWarpFromVector(InputMinus, this, true);
                            double[] ScoresMinus = GetPerTiltCC();

                            for (int t = 0; t < NTilts; t++)
                                Result[Offset + t * SliceElements + i] = (ScoresPlus[t] - ScoresMinus[t]) / Delta2;
                        }

                        Offset += GridAngleX.Values.Length;
                    }
                }
                else
                {
                    Offset += GridAngleX.Values.Length * 3;
                }


                {
                    for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                    {
                        Species Species = allSpecies[ispecies];
                        Particle[] Particles = SpeciesParticles[Species];

                        int TemporalRes = allSpecies[ispecies].TemporalResolutionMovement;

                        if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ParticlePosition) != 0)
                            for (int iparam = 0; iparam < 3 * TemporalRes; iparam++)
                            {
                                double[] InputPlus = input.ToArray();
                                for (int p = 0; p < Particles.Length; p++)
                                    InputPlus[Offset + p * 6 * TemporalRes + iparam] += Delta;

                                SetWarpFromVector(InputPlus, this, true);
                                double[] ScoresPlus = GetPerParticleCCSpecies(Species);

                                double[] InputMinus = input.ToArray();
                                for (int p = 0; p < Particles.Length; p++)
                                    InputMinus[Offset + p * 6 * TemporalRes + iparam] -= Delta;

                                SetWarpFromVector(InputMinus, this, true);
                                double[] ScoresMinus = GetPerParticleCCSpecies(Species);

                                for (int p = 0; p < Particles.Length; p++)
                                    Result[Offset + p * 6 * TemporalRes + iparam] = (ScoresPlus[p] - ScoresMinus[p]) / Delta2;
                            }

                        if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ParticleAngle) != 0)
                            for (int iparam = 3 * TemporalRes; iparam < 6 * TemporalRes; iparam++)
                            {
                                double[] InputPlus = input.ToArray();
                                for (int p = 0; p < Particles.Length; p++)
                                    InputPlus[Offset + p * 6 * TemporalRes + iparam] += Delta;

                                SetWarpFromVector(InputPlus, this, true);
                                double[] ScoresPlus = GetPerParticleCCSpecies(Species);

                                double[] InputMinus = input.ToArray();
                                for (int p = 0; p < Particles.Length; p++)
                                    InputMinus[Offset + p * 6 * TemporalRes + iparam] -= Delta;

                                SetWarpFromVector(InputMinus, this, true);
                                double[] ScoresMinus = GetPerParticleCCSpecies(Species);

                                for (int p = 0; p < Particles.Length; p++)
                                    Result[Offset + p * 6 * TemporalRes + iparam] = (ScoresPlus[p] - ScoresMinus[p]) / Delta2;
                            }

                        Offset += OriginalParticlePositions[ispecies].Length * 6; // No * TemporalRes because it's already included in OriginalParticlePositions
                    }
                }

                if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.Zernike13) != 0)
                {
                    for (int iparam = 0; iparam < Math.Min(6, CTF.ZernikeCoeffsOdd.Length); iparam++)
                    {
                        double[] InputPlus = input.ToArray();
                        InputPlus[Offset + iparam] += Delta;

                        //SetWarpFromVector(InputPlus, this, true);
                        double ScoresPlus = WarpEval(InputPlus);

                        double[] InputMinus = input.ToArray();
                        InputMinus[Offset + iparam] -= Delta;

                        //SetWarpFromVector(InputMinus, this, true);
                        double ScoresMinus = WarpEval(InputMinus);

                        Result[Offset + iparam] = (ScoresPlus - ScoresMinus) / Delta2;
                    }
                }

                if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.Zernike5) != 0)
                {
                    for (int iparam = Math.Min(6, CTF.ZernikeCoeffsOdd.Length); iparam < Math.Min(12, CTF.ZernikeCoeffsOdd.Length); iparam++)
                    {
                        double[] InputPlus = input.ToArray();
                        InputPlus[Offset + iparam] += Delta;

                        //SetWarpFromVector(InputPlus, this, true);
                        double ScoresPlus = WarpEval(InputPlus);

                        double[] InputMinus = input.ToArray();
                        InputMinus[Offset + iparam] -= Delta;

                        //SetWarpFromVector(InputMinus, this, true);
                        double ScoresMinus = WarpEval(InputMinus);

                        Result[Offset + iparam] = (ScoresPlus - ScoresMinus) / Delta2;
                    }
                }

                Offset += CTF.ZernikeCoeffsOdd.Length;

                if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.Magnification) != 0)
                {
                    for (int iparam = 0; iparam < 4; iparam++)
                    {
                        double[] InputPlus = input.ToArray();
                        InputPlus[input.Length - 4 + iparam] += Delta;

                        //SetWarpFromVector(InputPlus, this, true);
                        double ScoresPlus = WarpEval(InputPlus);

                        double[] InputMinus = input.ToArray();
                        InputMinus[input.Length - 4 + iparam] -= Delta;

                        //SetWarpFromVector(InputMinus, this, true);
                        double ScoresMinus = WarpEval(InputMinus);

                        Result[input.Length - 4 + iparam] = (ScoresPlus - ScoresMinus) / Delta2;
                    }
                }

                OldInput = input.ToList().ToArray();
                OldGradient = Result.ToList().ToArray();

                return Result;
            };

            #endregion

            #region Loss and gradient functions for defocus

            Func<double[], double> DefocusEval = input =>
            {
                SetDefocusFromVector(input);

                double ScoreAB = 0, ScoreA2 = 0, ScoreB2 = 0;

                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                int BatchSize = optionsMPA.BatchSize;
                float[] ResultP = new float[BatchSize * 3];
                float[] ResultQ = new float[BatchSize * 3];

                foreach (var species in allSpecies)
                {
                    if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];

                    float AngPixRefine = species.ResolutionRefinement / 2;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];
                    float[] ContainmentMask = SpeciesContainmentMasks[species];

                    Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper); // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine

                    Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                    Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                    Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                    int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                    int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                    int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                    if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                    {
                        Console.WriteLine($"SizeFullSuper = {SizeFullSuper}, BatchSize = {BatchSize}, free memory = {GPU.GetFreeMemory(GPUID)}");
                        throw new Exception("No FFT plans created!");
                    }

                    Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                    Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NTilts), true, true);
                    for (int t = 0; t < NTilts; t++)
                        GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                            PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                            PhaseCorrection.Dims.Slice(),
                            new int3(RelevantSizes[t]).Slice(),
                            1);

                    Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                    bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                    float[][] EwaldResults = { ResultP, ResultQ };

                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                        IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                        float[] BatchContainmentMask = ContainmentMask.Skip(batchStart * NTilts).Take(CurBatch * NTilts).ToArray();
                        float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                        float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                        for (int t = 0; t < NTilts; t++)
                        {
                            float3[] CoordinatesTilt = new float3[CurBatch];
                            float3[] AnglesTilt = new float3[CurBatch];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                                AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                            }

                            float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                            float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                            float[] Defoci = new float[CurBatch];
                            int3[] ExtractOrigins = new int3[CurBatch];
                            float3[] ResidualShifts = new float3[BatchSize];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                Defoci[p] = ImageCoords[p].Z;
                            }

                            for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                            {
                                GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    TiltData[t].Dims.Slice(),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    Helper.ToInterleaved(ExtractOrigins),
                                    true,
                                    (uint)CurBatch);

                                GPU.FFT(Extracted.GetDevice(Intent.Read),
                                    ExtractedFT.GetDevice(Intent.Write),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    (uint)CurBatch,
                                    PlanForwSuper);

                                ExtractedFT.ShiftSlices(ResidualShifts);
                                ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    (uint)CurBatch);

                                if (species.DoEwald)
                                {
                                    GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, EwaldReverse[iewald], ExtractedCTF, true);

                                    GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                        ExtractedCTF.GetDevice(Intent.Read),
                                        ExtractedFT.GetDevice(Intent.Write),
                                        ExtractedCTF.ElementsComplex,
                                        1);
                                }
                                else
                                {
                                    GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                    GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                        ExtractedCTF.GetDevice(Intent.Read),
                                        ExtractedFT.GetDevice(Intent.Write),
                                        ExtractedCTF.ElementsComplex,
                                        1);
                                }

                                GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    (uint)CurBatch,
                                    PlanBackSuper,
                                    false);

                                GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                    ExtractedCropped.GetDevice(Intent.Write),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    new int3(SizeRefine, SizeRefine, 1),
                                    (uint)CurBatch);

                                GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                    ExtractedCropped.GetDevice(Intent.Write),
                                    ExtractedCropped.Dims.Slice(),
                                    ParticleDiameterPix / 2f,
                                    16 * AngPixExtract / AngPixRefine,
                                    true,
                                    (uint)CurBatch);

                                GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                    ExtractedCroppedFT.GetDevice(Intent.Write),
                                    new int3(SizeRefine, SizeRefine, 1),
                                    (uint)CurBatch,
                                    PlanForw);

                                ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                    ExtractedCropped.GetDevice(Intent.Write),
                                    new int3(SizeRefine).Slice(),
                                    new int3(RelevantSizes[t]).Slice(),
                                    (uint)CurBatch);


                                GPU.MultiParticleDiff(EwaldResults[iewald],
                                    new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                    SizeRefine,
                                    new[] { RelevantSizes[t] },
                                    new float[CurBatch * 2],
                                    Helper.ToInterleaved(ImageAngles),
                                    MagnificationCorrection.ToVec(),
                                    SpeciesCTFWeights[species].GetDeviceSlice(t, Intent.Read),
                                    PhaseCorrectionAll.GetDeviceSlice(t, Intent.Read),
                                    species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)species.PixelSize) * (iewald == 0 ? 1 : -1) : 0,
                                    species.CurrentMaxShellRefinement,
                                    new[] { species.HalfMap1Projector[GPUID].t_DataRe, species.HalfMap2Projector[GPUID].t_DataRe },
                                    new[] { species.HalfMap1Projector[GPUID].t_DataIm, species.HalfMap2Projector[GPUID].t_DataIm },
                                    species.HalfMap1Projector[GPUID].Oversampling,
                                    species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                    new IntPtr((long)SpeciesParticleSubsets[species] + batchStart * sizeof(int)),
                                    CurBatch,
                                    1);
                            }

                            for (int i = 0; i < CurBatch; i++)
                            {
                                ScoreAB += (ResultP[i * 3 + 0] + ResultQ[i * 3 + 0]) * BatchContainmentMask[i * NTilts + t];
                                ScoreA2 += (ResultP[i * 3 + 1] + ResultQ[i * 3 + 1]) * BatchContainmentMask[i * NTilts + t];
                                ScoreB2 += (ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]) * BatchContainmentMask[i * NTilts + t];
                            }
                        }
                    }

                    PhaseCorrectionAll.Dispose();
                    PhaseCorrection.Dispose();
                    GammaCorrection.Dispose();

                    CoordsCTF.Dispose();
                    Extracted.Dispose();
                    ExtractedFT.Dispose();
                    ExtractedCropped.Dispose();
                    ExtractedCroppedFT.Dispose();
                    ExtractedCTF.Dispose();

                    GPU.DestroyFFTPlan(PlanForwSuper);
                    GPU.DestroyFFTPlan(PlanBackSuper);
                    GPU.DestroyFFTPlan(PlanForw);
                }

                //foreach (var image in TiltData)
                //    image.FreeDevice();

                double Score = ScoreAB / Math.Max(1e-10, Math.Sqrt(ScoreA2 * ScoreB2)) * NParticlesOverall * NTilts;
                Score *= 100;

                Console.WriteLine(Score);

                return Score;
            };

            Func<double[], double[]> DefocusGrad = input =>
            {
                double Delta = 0.001;
                double Delta2 = Delta * 2;

                double[] Deltas = { Delta, -Delta };

                double[] Result = new double[input.Length];
                double[] ScoresAB = new double[input.Length * 2];
                double[] ScoresA2 = new double[input.Length * 2];
                double[] ScoresB2 = new double[input.Length * 2];
                int[] ScoresSamples = new int[input.Length * 2];

                if (BFGSIterations-- <= 0)
                    return Result;

                if (MathHelper.AllEqual(input, OldInput))
                    return OldGradient;

                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                int BatchSize = 64;
                float[] ResultP = new float[BatchSize * 3];
                float[] ResultQ = new float[BatchSize * 3];

                foreach (var species in allSpecies)
                {
                    if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                    float AngPixRefine = species.ResolutionRefinement / 2;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                    float[] ContainmentMask = SpeciesContainmentMasks[species];

                    Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);

                    Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image ExtractedRefineSuper = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);
                    Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                    Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                    Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                    int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                    int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                    int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                    if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                    {
                        Console.WriteLine($"SizeFullSuper = {SizeFullSuper}, BatchSize = {BatchSize}, free memory = {GPU.GetFreeMemory(GPUID)}");
                        throw new Exception("No FFT plans created!");
                    }

                    Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                    Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NTilts), true, true);
                    for (int t = 0; t < NTilts; t++)
                        GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                            PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                            PhaseCorrection.Dims.Slice(),
                            new int3(RelevantSizes[t]).Slice(),
                            1);

                    bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                    float[][] EwaldResults = { ResultP, ResultQ };

                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                        IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                        float[] BatchContainmentMask = ContainmentMask.Skip(batchStart * NTilts).Take(CurBatch * NTilts).ToArray();
                        float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                        float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                        for (int t = 0; t < NTilts; t++)
                        {
                            float3[] CoordinatesTilt = new float3[CurBatch];
                            float3[] AnglesTilt = new float3[CurBatch];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                                AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                            }

                            float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                            float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                            float[] Defoci = new float[CurBatch];
                            int3[] ExtractOrigins = new int3[CurBatch];
                            float3[] ResidualShifts = new float3[BatchSize];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                Defoci[p] = ImageCoords[p].Z;
                            }

                            GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                TiltData[t].Dims.Slice(),
                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                Helper.ToInterleaved(ExtractOrigins),
                                true,
                                (uint)CurBatch);

                            GPU.FFT(Extracted.GetDevice(Intent.Read),
                                ExtractedFT.GetDevice(Intent.Write),
                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                (uint)CurBatch,
                                PlanForwSuper);

                            ExtractedFT.ShiftSlices(ResidualShifts);
                            ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                            GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                (uint)CurBatch);

                            for (int iparam = 0; iparam < CTFStepTypes.Length; iparam++)
                            {
                                if ((CurrentOptimizationTypeCTF & CTFStepTypes[iparam]) == 0)
                                    continue;

                                for (int idelta = 0; idelta < 2; idelta++)
                                {
                                    double[] InputAltered = input.ToArray();
                                    if (iparam < 4)
                                        InputAltered[t * 4 + iparam] += Deltas[idelta];
                                    else
                                        InputAltered[input.Length - CTFStepTypes.Length + iparam] += Deltas[idelta];

                                    SetDefocusFromVector(InputAltered);

                                    ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                    for (int p = 0; p < CurBatch; p++)
                                        Defoci[p] = ImageCoords[p].Z;


                                    Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                                    for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                    {
                                        if (species.DoEwald)
                                        {
                                            GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, EwaldReverse[iewald], ExtractedCTF, true);

                                            GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                ExtractedCTF.GetDevice(Intent.Read),
                                                ExtractedFT.GetDevice(Intent.Write),
                                                ExtractedCTF.ElementsComplex,
                                                1);
                                        }
                                        else
                                        {
                                            GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                            GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                ExtractedCTF.GetDevice(Intent.Read),
                                                ExtractedFT.GetDevice(Intent.Write),
                                                ExtractedCTF.ElementsComplex,
                                                1);
                                        }

                                        GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                            ExtractedRefineSuper.GetDevice(Intent.Write),
                                            new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                            (uint)CurBatch,
                                            PlanBackSuper,
                                            false);

                                        GPU.CropFTFull(ExtractedRefineSuper.GetDevice(Intent.Read),
                                            ExtractedCropped.GetDevice(Intent.Write),
                                            new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                            new int3(SizeRefine, SizeRefine, 1),
                                            (uint)CurBatch);

                                        GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                            ExtractedCropped.GetDevice(Intent.Write),
                                            ExtractedCropped.Dims.Slice(),
                                            ParticleDiameterPix / 2f,
                                            16 * AngPixExtract / AngPixRefine,
                                            true,
                                            (uint)CurBatch);

                                        GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                            ExtractedCroppedFT.GetDevice(Intent.Write),
                                            new int3(SizeRefine, SizeRefine, 1),
                                            (uint)CurBatch,
                                            PlanForw);

                                        ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                        GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                            ExtractedCropped.GetDevice(Intent.Write),
                                            new int3(SizeRefine).Slice(),
                                            new int3(RelevantSizes[t]).Slice(),
                                            (uint)CurBatch);


                                        GPU.MultiParticleDiff(EwaldResults[iewald],
                                            new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                            SizeRefine,
                                            new[] { RelevantSizes[t] },
                                            new float[CurBatch * 2],
                                            Helper.ToInterleaved(ImageAngles),
                                            MagnificationCorrection.ToVec(),
                                            SpeciesCTFWeights[species].GetDeviceSlice(t, Intent.Read),
                                            PhaseCorrectionAll.GetDeviceSlice(t, Intent.Read),
                                            species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)species.PixelSize) * (iewald == 0 ? 1 : -1) : 0,
                                            species.CurrentMaxShellRefinement,
                                            new[] { species.HalfMap1Projector[GPUID].t_DataRe, species.HalfMap2Projector[GPUID].t_DataRe },
                                            new[] { species.HalfMap1Projector[GPUID].t_DataIm, species.HalfMap2Projector[GPUID].t_DataIm },
                                            species.HalfMap1Projector[GPUID].Oversampling,
                                            species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                            new IntPtr((long)SpeciesParticleSubsets[species] + batchStart * sizeof(int)),
                                            CurBatch,
                                            1);
                                    }

                                    GammaCorrection.Dispose();

                                    if (iparam < 4)
                                        for (int i = 0; i < CurBatch; i++)
                                        {
                                            ScoresAB[(t * 4 + iparam) * 2 + idelta] += (ResultP[i * 3 + 0] + ResultQ[i * 3 + 0]) * BatchContainmentMask[i * NTilts + t];
                                            ScoresA2[(t * 4 + iparam) * 2 + idelta] += (ResultP[i * 3 + 1] + ResultQ[i * 3 + 1]) * BatchContainmentMask[i * NTilts + t];
                                            ScoresB2[(t * 4 + iparam) * 2 + idelta] += (ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]) * BatchContainmentMask[i * NTilts + t];
                                            ScoresSamples[(t * 4 + iparam) * 2 + idelta]++;
                                        }
                                    else
                                        for (int i = 0; i < CurBatch; i++)
                                        {
                                            ScoresAB[(input.Length - CTFStepTypes.Length + iparam) * 2 + idelta] += (ResultP[i * 3 + 0] + ResultQ[i * 3 + 0]) * BatchContainmentMask[i * NTilts + t];
                                            ScoresA2[(input.Length - CTFStepTypes.Length + iparam) * 2 + idelta] += (ResultP[i * 3 + 1] + ResultQ[i * 3 + 1]) * BatchContainmentMask[i * NTilts + t];
                                            ScoresB2[(input.Length - CTFStepTypes.Length + iparam) * 2 + idelta] += (ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]) * BatchContainmentMask[i * NTilts + t];
                                            ScoresSamples[(input.Length - CTFStepTypes.Length + iparam) * 2 + idelta]++;
                                        }
                                }
                            }
                        }
                    }

                    PhaseCorrectionAll.Dispose();
                    PhaseCorrection.Dispose();

                    CoordsCTF.Dispose();
                    Extracted.Dispose();
                    ExtractedFT.Dispose();
                    ExtractedRefineSuper.Dispose();
                    ExtractedCropped.Dispose();
                    ExtractedCroppedFT.Dispose();
                    ExtractedCTF.Dispose();

                    GPU.DestroyFFTPlan(PlanForwSuper);
                    GPU.DestroyFFTPlan(PlanBackSuper);
                    GPU.DestroyFFTPlan(PlanForw);
                }

                //foreach (var image in TiltData)
                //    image.FreeDevice();

                for (int i = 0; i < ScoresAB.Length; i++)
                    ScoresAB[i] = ScoresAB[i] / Math.Max(1e-10, Math.Sqrt(ScoresA2[i] * ScoresB2[i])) * ScoresSamples[i];

                for (int i = 0; i < Result.Length; i++)
                    Result[i] = (ScoresAB[i * 2 + 0] - ScoresAB[i * 2 + 1]) / Delta2 * 100;

                OldInput = input.ToList().ToArray();
                OldGradient = Result.ToList().ToArray();

                return Result;
            };

            #endregion

            #region Grid search for per-tilt defoci

            Func<double[], double[]> DefocusGridSearch = input =>
            {
                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                int BatchSize = optionsMPA.BatchSize;
                float[] ResultP = new float[BatchSize * 3];
                float[] ResultQ = new float[BatchSize * 3];

                List<float4>[] AllSearchValues = Helper.ArrayOfFunction(i => new List<float4>(), NTilts);
                List<float4>[] CurrentSearchValues = Helper.ArrayOfFunction(i => new List<float4>(), NTilts);
                decimal GridSearchDelta = 0.3M;
                foreach (var list in CurrentSearchValues)
                {
                    for (decimal d = -3M; d <= 3M; d += GridSearchDelta)
                        list.Add(new float4((float)d, 0, 0, 0));
                }

                for (int irefine = 0; irefine < 4; irefine++)
                {
                    foreach (var species in allSpecies)
                    {
                        if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                            continue;

                        Particle[] Particles = SpeciesParticles[species];
                        int NParticles = Particles.Length;
                        if (NParticles == 0)
                            continue;

                        int SpeciesOffset = SpeciesParticleIDRanges[species].Start;

                        int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                        int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                        float AngPixRefine = species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                        int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                        float[] ContainmentMask = SpeciesContainmentMasks[species];

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);

                        Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedRefineSuper = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);
                        Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                        Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                        Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NTilts), true, true);
                        for (int t = 0; t < NTilts; t++)
                            GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                                PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                                PhaseCorrection.Dims.Slice(),
                                new int3(RelevantSizes[t]).Slice(),
                                1);

                        Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                        bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                        float[][] EwaldResults = { ResultP, ResultQ };

                        int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                        int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                        if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                        {
                            Console.WriteLine($"SizeFullSuper = {SizeFullSuper}, BatchSize = {BatchSize}, free memory = {GPU.GetFreeMemory(GPUID)}");
                            throw new Exception("No FFT plans created!");
                        }

                        GPU.CheckGPUExceptions();

                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float[] BatchContainmentMask = ContainmentMask.Skip(batchStart * NTilts).Take(CurBatch * NTilts).ToArray();
                            float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                            float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                            for (int t = 0; t < NTilts; t++)
                            {
                                float3[] CoordinatesTilt = new float3[CurBatch];
                                float3[] AnglesTilt = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                                    AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                                }

                                float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                                float[] Defoci = new float[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p] = ImageCoords[p].Z;
                                }

                                GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    TiltData[t].Dims.Slice(),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    Helper.ToInterleaved(ExtractOrigins),
                                    true,
                                    (uint)CurBatch);

                                GPU.FFT(Extracted.GetDevice(Intent.Read),
                                    ExtractedFT.GetDevice(Intent.Write),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    (uint)CurBatch,
                                    PlanForwSuper);

                                ExtractedFT.ShiftSlices(ResidualShifts);
                                ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    (uint)CurBatch);

                                for (int idelta = 0; idelta < CurrentSearchValues[t].Count; idelta++)
                                {
                                    double[] InputAltered = input.ToArray();
                                    InputAltered[t * 4 + 0] += CurrentSearchValues[t][idelta].X;

                                    SetDefocusFromVector(InputAltered);

                                    ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                    for (int p = 0; p < CurBatch; p++)
                                        Defoci[p] = ImageCoords[p].Z;

                                    for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                    {
                                        if (species.DoEwald)
                                        {
                                            GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, EwaldReverse[iewald], ExtractedCTF, true);

                                            GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                ExtractedCTF.GetDevice(Intent.Read),
                                                ExtractedFT.GetDevice(Intent.Write),
                                                ExtractedCTF.ElementsComplex,
                                                1);
                                        }
                                        else
                                        {
                                            GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                            GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                                ExtractedCTF.GetDevice(Intent.Read),
                                                ExtractedFT.GetDevice(Intent.Write),
                                                ExtractedCTF.ElementsComplex,
                                                1);
                                        }

                                        GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                            ExtractedRefineSuper.GetDevice(Intent.Write),
                                            new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                            (uint)CurBatch,
                                            PlanBackSuper,
                                            false);

                                        GPU.CropFTFull(ExtractedRefineSuper.GetDevice(Intent.Read),
                                            ExtractedCropped.GetDevice(Intent.Write),
                                            new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                            new int3(SizeRefine, SizeRefine, 1),
                                            (uint)CurBatch);

                                        GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                            ExtractedCropped.GetDevice(Intent.Write),
                                            ExtractedCropped.Dims.Slice(),
                                            ParticleDiameterPix / 2f,
                                            16 * AngPixExtract / AngPixRefine,
                                            true,
                                            (uint)CurBatch);

                                        GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                            ExtractedCroppedFT.GetDevice(Intent.Write),
                                            new int3(SizeRefine, SizeRefine, 1),
                                            (uint)CurBatch,
                                            PlanForw);

                                        ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                        GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                            ExtractedCropped.GetDevice(Intent.Write),
                                            new int3(SizeRefine).Slice(),
                                            new int3(RelevantSizes[t]).Slice(),
                                            (uint)CurBatch);


                                        GPU.MultiParticleDiff(EwaldResults[iewald],
                                            new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                            SizeRefine,
                                            new[] { RelevantSizes[t] },
                                            new float[CurBatch * 2],
                                            Helper.ToInterleaved(ImageAngles),
                                            MagnificationCorrection.ToVec(),
                                            SpeciesCTFWeights[species].GetDeviceSlice(t, Intent.Read),
                                            PhaseCorrectionAll.GetDeviceSlice(t, Intent.Read),
                                            species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)species.PixelSize) * (iewald == 0 ? 1 : -1) : 0,
                                            species.CurrentMaxShellRefinement,
                                            new[] { species.HalfMap1Projector[GPUID].t_DataRe, species.HalfMap2Projector[GPUID].t_DataRe },
                                            new[] { species.HalfMap1Projector[GPUID].t_DataIm, species.HalfMap2Projector[GPUID].t_DataIm },
                                            species.HalfMap1Projector[GPUID].Oversampling,
                                            species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                            new IntPtr((long)SpeciesParticleSubsets[species] + batchStart * sizeof(int)),
                                            CurBatch,
                                            1);
                                    }

                                    for (int i = 0; i < CurBatch; i++)
                                        CurrentSearchValues[t][idelta] += new float4(0,
                                            (ResultP[i * 3 + 0] + ResultQ[i * 3 + 0]) * BatchContainmentMask[i * NTilts + t],
                                            (ResultP[i * 3 + 1] + ResultQ[i * 3 + 1]) * BatchContainmentMask[i * NTilts + t],
                                            (ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]) * BatchContainmentMask[i * NTilts + t]);
                                }
                            }
                        }

                        PhaseCorrectionAll.Dispose();
                        PhaseCorrection.Dispose();
                        GammaCorrection.Dispose();
                        CoordsCTF.Dispose();
                        Extracted.Dispose();
                        ExtractedFT.Dispose();
                        ExtractedRefineSuper.Dispose();
                        ExtractedCropped.Dispose();
                        ExtractedCroppedFT.Dispose();
                        ExtractedCTF.Dispose();

                        GPU.DestroyFFTPlan(PlanForwSuper);
                        GPU.DestroyFFTPlan(PlanBackSuper);
                        GPU.DestroyFFTPlan(PlanForw);
                    }

                    GridSearchDelta /= 2;
                    for (int t = 0; t < NTilts; t++)
                    {
                        CurrentSearchValues[t].Sort((a, b) => -((a.Y / Math.Max(1e-20, Math.Sqrt(a.Z * a.W))).CompareTo(b.Y / Math.Max(1e-20, Math.Sqrt(b.Z * b.W)))));
                        AllSearchValues[t].AddRange(CurrentSearchValues[t]);

                        List<float4> NewSearchValues = new List<float4>();
                        for (int j = 0; j < 2; j++)
                        {
                            NewSearchValues.Add(new float4(CurrentSearchValues[t][j].X + (float)GridSearchDelta, 0, 0, 0));
                            NewSearchValues.Add(new float4(CurrentSearchValues[t][j].X - (float)GridSearchDelta, 0, 0, 0));
                        }

                        CurrentSearchValues[t] = NewSearchValues;
                    }
                }

                for (int i = 0; i < NTilts; i++)
                {
                    AllSearchValues[i].Sort((a, b) => -((a.Y / Math.Max(1e-10, Math.Sqrt(a.Z * a.W))).CompareTo(b.Y / Math.Max(1e-10, Math.Sqrt(b.Z * b.W)))));
                    input[i * 4 + 0] += AllSearchValues[i][0].X;
                }

                return input;
            };

            #endregion

            BroydenFletcherGoldfarbShanno OptimizerWarp = new BroydenFletcherGoldfarbShanno(InitialParametersWarp.Length, WarpEval, WarpGrad);
            BroydenFletcherGoldfarbShanno OptimizerDefocus = new BroydenFletcherGoldfarbShanno(InitialParametersDefocus.Length, DefocusEval, DefocusGrad);

            //WarpEval(InitialParametersWarp);

            bool NeedReextraction = true;

            for (int ioptim = 0; ioptim < optionsMPA.NIterations; ioptim++)
            {
                foreach (var species in allSpecies)
                    species.CurrentMaxShellRefinement = (int)Math.Round(MathHelper.Lerp(optionsMPA.InitialResolutionPercent / 100f,
                            1f,
                            optionsMPA.NIterations == 1 ? 1 : ((float)ioptim / (optionsMPA.NIterations - 1))) *
                        species.HalfMap1Projector[GPUID].Dims.X / 2);

                if (NeedReextraction)
                {
                    progressCallback($"Re-extracting particles for optimization iteration {ioptim + 1}/{optionsMPA.NIterations}");
                    ReextractPaddedParticles(false);
                }

                NeedReextraction = false;

                foreach (var step in OptimizationStepsWarp)
                {
                    progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, " + step.Name);

                    BFGSIterations = step.Iterations;
                    CurrentOptimizationTypeWarp = step.Type;
                    CurrentWeightsDict = SpeciesTiltWeights;

                    OptimizerWarp.Maximize(InitialParametersWarp);

                    OldInput = null;
                }

                if (allSpecies.Any(s => s.ResolutionRefinement < (float)optionsMPA.MinimumCTFRefinementResolution))
                {
                    //ReextractPaddedParticles();
                    //WarpEval(InitialParametersWarp);

                    if (ioptim == 0 && optionsMPA.DoDefocusGridSearch)
                    {
                        progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, defocus grid search");

                        InitialParametersDefocus = DefocusGridSearch(InitialParametersDefocus);

                        NeedReextraction = true;
                    }

                    //CurrentWeightsDict = SpeciesFrameWeights;
                    //ReextractPaddedParticles();
                    //WarpEval(InitialParametersWarp);

                    foreach (var step in OptimizationStepsCTF)
                    {
                        progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, " + step.Name);

                        BFGSIterations = step.Iterations;
                        CurrentOptimizationTypeCTF = step.Type;
                        CurrentWeightsDict = SpeciesCTFWeights;

                        OptimizerDefocus.Maximize(InitialParametersDefocus);

                        OldInput = null;
                        NeedReextraction = true;
                    }

                    if (NeedReextraction && ioptim >= optionsMPA.NIterations - 1)
                    {
                        progressCallback($"Re-extracting particles after optimization iteration {ioptim + 1}/{optionsMPA.NIterations}");
                        ReextractPaddedParticles(false);
                    }
                    //NeedReextraction = false;
                }
            }

            SetWarpFromVector(InitialParametersWarp, this, true);
            SetDefocusFromVector(InitialParametersDefocus);

            Console.WriteLine("Final score: ");
            WarpEval(InitialParametersWarp);

            Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after optimization of {Name}");

            #region Compute NCC for each particle to be able to take only N % of the best later

            {
                double[] AllParticleScores = GetPerParticleCC();

                foreach (var species in allSpecies)
                {
                    Particle[] Particles = SpeciesParticles[species];

                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    double[] ParticleScores = Helper.Subset(AllParticleScores, SpeciesParticleIDRanges[species].Start, SpeciesParticleIDRanges[species].End);

                    List<int> IndicesSorted = Helper.ArrayOfSequence(0, NParticles, 1).ToList();
                    IndicesSorted.Sort((a, b) => ParticleScores[a].CompareTo(ParticleScores[b]));
                    int FirstGoodIndex = (int)(NParticles * 0.0);

                    float[] Mask = new float[NParticles];
                    for (int i = 0; i < NParticles; i++)
                        Mask[IndicesSorted[i]] = (i >= FirstGoodIndex ? 1f : 0f);

                    GoodParticleMasks.Add(species, Mask);
                }
            }

            #endregion

            #region Compute FSC between refs and particles to estimate tilt and series weights

            if (true)
            {
                progressCallback($"Calculating FRC between projections and particles for weight optimization");

                int FSCLength = 128;
                Image FSC = new Image(new int3(FSCLength, FSCLength, NTilts * 3), true);
                Image FSCPerParticle = new Image(new int3(FSCLength / 2, NParticlesOverall * 3, 1));
                //float[][] FSCData = FSC.GetHost(Intent.ReadWrite);
                //float[][] FSCPerParticleData = FSCPerParticle.GetHost(Intent.ReadWrite);
                Image PhaseResiduals = new Image(new int3(FSCLength, FSCLength, 2), true);

                Star TableOut = new Star(new string[] { "wrpNormCoordinateX", "wrpNormCoordinateY", "wrpNormCoordinateZ" });

                int BatchSize = optionsMPA.BatchSize;
                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                {
                    Species Species = allSpecies[ispecies];
                    Particle[] Particles = SpeciesParticles[Species];

                    int NParticles = Particles.Length;
                    float SpeciesAngPix = Species.ResolutionRefinement / 2;
                    if (NParticles == 0)
                        continue;

                    int SpeciesOffset = SpeciesParticleIDRanges[Species].Start;

                    int SizeRefine = SpeciesRefinementSize[Species];
                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[Species];

                    //Image CorrAB = new Image(new int3(SizeRefine, SizeRefine, NTilts), true);
                    //Image CorrA2 = new Image(new int3(SizeRefine, SizeRefine, NTilts), true);
                    //Image CorrB2 = new Image(new int3(SizeRefine, SizeRefine, NTilts), true);

                    float ScaleFactor = (float)Species.PixelSize * (FSCLength / 2 - 1) /
                                        (float)(Species.ResolutionRefinement / 2 * (SizeRefine / 2 - 1));

                    int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[Species];
                    int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[Species];
                    float AngPixRefine = Species.ResolutionRefinement / 2;
                    int ParticleDiameterPix = (int)(Species.DiameterAngstrom / AngPixRefine);

                    {
                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper); // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine
                        Image CoordsCTFCropped = CTF.GetCTFCoords(SizeRefine, SizeRefine);

                        Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true, true);

                        int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                        int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                        if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                            throw new Exception("No FFT plans created!");

                        Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefineSuper);
                        Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                        bool[] EwaldReverse = { Species.EwaldReverse, !Species.EwaldReverse };

                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                            float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                            for (int i = 0; i < CurBatch; i++)
                            {
                                float3 Coords = CoordinatesMoving[i * NTilts];
                                Coords /= VolumeDimensionsPhysical;
                                TableOut.AddRow(new string[]
                                {
                                    Coords.X.ToString(CultureInfo.InvariantCulture),
                                    Coords.Y.ToString(CultureInfo.InvariantCulture),
                                    Coords.Z.ToString(CultureInfo.InvariantCulture)
                                });
                            }

                            for (int t = 0; t < NTilts; t++)
                            {
                                float3[] CoordinatesTilt = new float3[CurBatch];
                                float3[] AnglesTilt = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                                    AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                                }

                                float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                                float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                                float[] Defoci = new float[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p] = ImageCoords[p].Z;
                                }

                                for (int iewald = 0; iewald < (Species.DoEwald ? 2 : 1); iewald++)
                                {
                                    GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                                        Extracted.GetDevice(Intent.Write),
                                        TiltData[t].Dims.Slice(),
                                        new int3(SizeFullSuper, SizeFullSuper, 1),
                                        Helper.ToInterleaved(ExtractOrigins),
                                        true,
                                        (uint)CurBatch);

                                    GPU.FFT(Extracted.GetDevice(Intent.Read),
                                        ExtractedFT.GetDevice(Intent.Write),
                                        new int3(SizeFullSuper, SizeFullSuper, 1),
                                        (uint)CurBatch,
                                        PlanForwSuper);

                                    ExtractedFT.ShiftSlices(ResidualShifts);
                                    ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                    GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                        Extracted.GetDevice(Intent.Write),
                                        new int3(SizeFullSuper, SizeFullSuper, 1),
                                        new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                        (uint)CurBatch);

                                    GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                        PhaseCorrection.GetDevice(Intent.Read),
                                        Extracted.GetDevice(Intent.Write),
                                        PhaseCorrection.ElementsSliceComplex,
                                        (uint)CurBatch);

                                    if (Species.DoEwald)
                                    {
                                        GetComplexCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, EwaldReverse[iewald], ExtractedCTF, true);

                                        GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                            ExtractedCTF.GetDevice(Intent.Read),
                                            ExtractedFT.GetDevice(Intent.Write),
                                            ExtractedCTF.ElementsComplex,
                                            1);
                                    }
                                    else
                                    {
                                        GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, GammaCorrection, t, ExtractedCTF, true);

                                        GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                            ExtractedCTF.GetDevice(Intent.Read),
                                            ExtractedFT.GetDevice(Intent.Write),
                                            ExtractedCTF.ElementsComplex,
                                            1);
                                    }

                                    GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                        Extracted.GetDevice(Intent.Write),
                                        new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                        (uint)CurBatch,
                                        PlanBackSuper,
                                        false);

                                    GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                        ExtractedCropped.GetDevice(Intent.Write),
                                        new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                        new int3(SizeRefine, SizeRefine, 1),
                                        (uint)CurBatch);

                                    GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                        ExtractedCropped.GetDevice(Intent.Write),
                                        ExtractedCropped.Dims.Slice(),
                                        ParticleDiameterPix / 2f,
                                        16 * AngPixExtract / AngPixRefine,
                                        true,
                                        (uint)CurBatch);

                                    GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                        ExtractedCroppedFT.GetDevice(Intent.Write),
                                        new int3(SizeRefine, SizeRefine, 1),
                                        (uint)CurBatch,
                                        PlanForw);

                                    ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                    GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTFCropped, null, t, ExtractedCTF, true, true, true);


                                    GPU.MultiParticleCorr2D(FSC.GetDeviceSlice(t * 3, Intent.ReadWrite),
                                        new IntPtr((long)FSCPerParticle.GetDevice(Intent.ReadWrite) + (SpeciesOffset + batchStart) * FSCPerParticle.Dims.X * 3 * sizeof(float)),
                                        PhaseResiduals.GetDevice(Intent.ReadWrite),
                                        FSCLength,
                                        new IntPtr[] { ExtractedCroppedFT.GetDevice(Intent.Read) },
                                        ExtractedCTF.GetDevice(Intent.Read),
                                        SizeRefine,
                                        ScaleFactor,
                                        null,
                                        new float[CurBatch * 2],
                                        Helper.ToInterleaved(ImageAngles),
                                        MagnificationCorrection.ToVec(),
                                        Species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize) * (iewald == 0 ? 1 : -1) : 0,
                                        new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                        new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                        Species.HalfMap1Projector[GPUID].Oversampling,
                                        Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                        new IntPtr((long)SpeciesParticleSubsets[Species] + batchStart * sizeof(int)),
                                        CurBatch,
                                        1);
                                }
                            }
                        }

                        PhaseCorrection.Dispose();
                        GammaCorrection.Dispose();

                        CoordsCTFCropped.Dispose();
                        CoordsCTF.Dispose();
                        Extracted.Dispose();
                        ExtractedFT.Dispose();
                        ExtractedCropped.Dispose();
                        ExtractedCroppedFT.Dispose();
                        ExtractedCTF.Dispose();

                        GPU.DestroyFFTPlan(PlanForwSuper);
                        GPU.DestroyFFTPlan(PlanBackSuper);
                        GPU.DestroyFFTPlan(PlanForw);
                    }
                }

                FSC.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", RootName + "_fsc.mrc"), true);
                FSC.Dispose();

                FSCPerParticle.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", RootName + "_fscparticles.mrc"), true);
                FSCPerParticle.Dispose();

                PhaseResiduals.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", RootName + "_phaseresiduals.mrc"), true);
                PhaseResiduals.Dispose();

                TableOut.Save(System.IO.Path.Combine(workingDirectory, "..", RootName + "_fscparticles.star"));
            }

            #endregion

            #region Tear down

            foreach (var pair in SpeciesParticleImages)
            {
                foreach (var ptr in SpeciesParticleImages[pair.Key])
                    if (optionsMPA.UseHostMemory)
                        GPU.FreeHostPinned(ptr);
                    else
                        GPU.FreeDevice(ptr);
                if (pair.Key.DoEwald)
                    foreach (var ptr in SpeciesParticleQImages[pair.Key])
                        GPU.FreeDevice(ptr);
                SpeciesCTFWeights[pair.Key].Dispose();
                SpeciesTiltWeights[pair.Key].Dispose();
                GPU.FreeDevice(SpeciesParticleSubsets[pair.Key]);

                pair.Key.HalfMap1Projector[GPUID].FreeDevice();
                pair.Key.HalfMap2Projector[GPUID].FreeDevice();
            }

            Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after optimization teardown of {Name}");

            #endregion
        }

        #region Update reconstructions with newly aligned particles

        GPU.SetDevice(GPUID);
        GPU.CheckGPUExceptions();

        progressCallback($"Extracting and back-projecting particles...");

        foreach (var species in allSpecies)
        {
            if (SpeciesParticles[species].Length == 0)
                continue;

            Projector[] Reconstructions = { species.HalfMap1Reconstruction[GPUID], species.HalfMap2Reconstruction[GPUID] };

            float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
            int BatchSize = optionsMPA.BatchSize;

            CTF MaxDefocusCTF = GetTiltCTF(IndicesSortedDose[0]);
            float ExpectedResolution = Math.Max((float)OptionsDataLoad.BinnedPixelSizeMean * 2, (float)species.GlobalResolution * 0.8f);
            int ExpectedBoxSize = (int)(species.DiameterAngstrom / (ExpectedResolution / 2)) * 2;
            int MinimumBoxSize = Math.Max(ExpectedBoxSize, MaxDefocusCTF.GetAliasingFreeSize(ExpectedResolution, (float)(species.DiameterAngstrom / (ExpectedResolution / 2))));
            int CTFSuperresFactor = (int)Math.Ceiling((float)MinimumBoxSize / ExpectedBoxSize);

            int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
            int SizeFullSuper = SizeFull * CTFSuperresFactor;

            float Radius = species.DiameterAngstrom / 2;

            Image CTFCoords = CTF.GetCTFCoords(SizeFullSuper, SizeFullSuper);
            //float2[] CTFCoordsData = CTFCoords.GetHostComplexCopy()[0];
            Image CTFCoordsP = CTF.GetCTFPCoords(SizeFullSuper, SizeFullSuper);
            float2[] CTFCoordsPData = CTFCoordsP.GetHostComplexCopy()[0];
            Image CTFCoordsCropped = CTF.GetCTFCoords(SizeFull, SizeFull);

            Image GammaCorrection = CTF.GetGammaCorrection(AngPixExtract, SizeFullSuper);

            //float[] PQSigns = new float[CTFCoordsData.Length];
            //CTF.PrecomputePQSigns(SizeFullSuper, 2, species.EwaldReverse, CTFCoordsData, CTFCoordsPData, PQSigns);

            Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixExtract, SizeFullSuper);

            Image IntermediateMaskAngles = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, 2), true);
            Image IntermediateFTCorr = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
            Image IntermediateCTFP = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);

            Image MaskParticle = new Image(new int3(SizeFullSuper, SizeFullSuper, 1));
            MaskParticle.Fill(1);
            MaskParticle.MaskSpherically((float)(species.DiameterAngstrom + 6) / AngPixExtract, 3, false);
            MaskParticle.RemapToFT();

            Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize));
            Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
            Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize));
            Image ExtractedCroppedFTp = new Image(new int3(SizeFull, SizeFull, BatchSize), true, true);
            Image ExtractedCroppedFTq = new Image(new int3(SizeFull, SizeFull, BatchSize), true, true);

            Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true);
            Image ExtractedCTFCropped = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize), true);
            Image CTFWeights = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize), true);

            int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
            int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
            int PlanForw = GPU.CreateFFTPlan(new int3(SizeFull, SizeFull, 1), (uint)BatchSize);

            if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                throw new Exception("No FFT plans created!");

            GPU.CheckGPUExceptions();

            Particle[] AllParticles = SpeciesParticles[species];
            Particle[][] SubsetParticles =
            {
                AllParticles.Where(p => p.RandomSubset == 0).ToArray(),
                AllParticles.Where(p => p.RandomSubset == 1).ToArray()
            };

            //Image CTFAvg = new Image(new int3(SizeFull, SizeFull, BatchSize), true);

            for (int isubset = 0; isubset < 2; isubset++)
            {
                Particle[] Particles = SubsetParticles[isubset];
                int NParticles = Particles.Length;
                //NParticles = 1;

                for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                    IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                    float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                    float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                    for (int t = 0; t < NTilts; t++)
                    {
                        float3[] CoordinatesTilt = new float3[CurBatch];
                        float3[] AnglesTilt = new float3[CurBatch];
                        for (int p = 0; p < CurBatch; p++)
                        {
                            CoordinatesTilt[p] = CoordinatesMoving[p * NTilts + t];
                            AnglesTilt[p] = AnglesMoving[p * NTilts + t];
                        }

                        float3[] ImageCoords = GetPositionsInOneTilt(CoordinatesTilt, t);
                        float3[] ImageAngles = GetAnglesInOneTilt(CoordinatesTilt, AnglesTilt, t);

                        float[] Defoci = new float[CurBatch];
                        int3[] ExtractOrigins = new int3[CurBatch];
                        float3[] ResidualShifts = new float3[BatchSize];
                        for (int p = 0; p < CurBatch; p++)
                        {
                            float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                            ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                            ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                            Defoci[p] = ImageCoords[p].Z;
                        }

                        float[] ContainmentMask = Helper.ArrayOfConstant(1f, BatchSize);
                        for (int i = 0; i < ImageCoords.Length; i++)
                        {
                            float3 Pos = ImageCoords[i];

                            float DistX = Math.Min(Pos.X, ImageDimensionsPhysical.X - Pos.X);
                            float DistY = Math.Min(Pos.Y, ImageDimensionsPhysical.Y - Pos.Y);
                            if (DistX < Radius || DistY < Radius)
                                ContainmentMask[i] = 0;
                        }

                        #region Image data

                        GPU.Extract(TiltData[t].GetDevice(Intent.Read),
                            Extracted.GetDevice(Intent.Write),
                            TiltData[t].Dims.Slice(),
                            new int3(SizeFullSuper, SizeFullSuper, 1),
                            Helper.ToInterleaved(ExtractOrigins),
                            true,
                            (uint)CurBatch);

                        GPU.FFT(Extracted.GetDevice(Intent.Read),
                            ExtractedFT.GetDevice(Intent.Write),
                            new int3(SizeFullSuper, SizeFullSuper, 1),
                            (uint)CurBatch,
                            PlanForwSuper);

                        ExtractedFT.ShiftSlices(ResidualShifts);
                        ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                        GPU.MultiplyComplexSlicesByComplex(ExtractedFT.GetDevice(Intent.Read),
                            PhaseCorrection.GetDevice(Intent.Read),
                            ExtractedFT.GetDevice(Intent.Write),
                            PhaseCorrection.ElementsComplex,
                            (uint)CurBatch);

                        CTF[] CTFParams = GetCTFParamsForOneTilt(AngPixExtract, Defoci, ImageCoords, t, false, false, false);

                        CTF.ApplyPandQPrecomp(ExtractedFT,
                            CTFParams,
                            IntermediateFTCorr,
                            Extracted,
                            ExtractedCropped,
                            IntermediateCTFP,
                            CTFCoords,
                            GammaCorrection,
                            species.EwaldReverse,
                            null,
                            PlanForw,
                            PlanBackSuper,
                            ExtractedCroppedFTp,
                            ExtractedCroppedFTq);

                        GetCTFsForOneTilt(AngPixExtract, Defoci, ImageCoords, CTFCoordsCropped, null, t, CTFWeights, true, true, true);
                        CTFWeights.Multiply(ContainmentMask);

                        ExtractedCroppedFTp.Multiply(CTFWeights);
                        ExtractedCroppedFTq.Multiply(CTFWeights);

                        #endregion

                        #region CTF data

                        //float[][] ExtractedCTFData = ExtractedCTF.GetHost(Intent.Write);
                        //Parallel.For(0, CurBatch, i =>
                        //{
                        //    CTFParams[i].GetEwaldWeights(CTFCoordsData, species.DiameterAngstrom, ExtractedCTFData[i]);
                        //});
                        GPU.CreateCTFEwaldWeights(ExtractedCTF.GetDevice(Intent.Write),
                            CTFCoords.GetDevice(Intent.Read),
                            GammaCorrection.GetDevice(Intent.Read),
                            species.DiameterAngstrom,
                            (uint)CTFCoords.ElementsSliceComplex,
                            Helper.ArrayOfFunction(i => CTFParams[i].ToStruct(), CurBatch),
                            (uint)CurBatch);
                        ExtractedCTF.Multiply(ExtractedCTF);

                        ExtractedFT.Fill(new float2(1, 0));
                        ExtractedFT.Multiply(ExtractedCTF);

                        GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                            Extracted.GetDevice(Intent.Write),
                            new int3(SizeFullSuper, SizeFullSuper, 1),
                            (uint)CurBatch,
                            PlanBackSuper,
                            false);

                        GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                            ExtractedCropped.GetDevice(Intent.Write),
                            new int3(SizeFullSuper, SizeFullSuper, 1),
                            new int3(SizeFull, SizeFull, 1),
                            (uint)CurBatch);

                        GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                            ExtractedFT.GetDevice(Intent.Write),
                            new int3(SizeFull, SizeFull, 1),
                            (uint)CurBatch,
                            PlanForw);

                        GPU.Real(ExtractedFT.GetDevice(Intent.Read),
                            ExtractedCTFCropped.GetDevice(Intent.Write),
                            ExtractedCTFCropped.ElementsReal);

                        ExtractedCTFCropped.Multiply(1f / (SizeFull * SizeFull));
                        ExtractedCTFCropped.Multiply(CTFWeights);

                        #endregion

                        //ImageAngles = new[] { new float3(0, 0, 0) };
                        //ImageAngles = Helper.ArrayOfConstant(new float3(0, 0, 0), CurBatch);

                        Reconstructions[isubset].BackProject(ExtractedCroppedFTp, ExtractedCTFCropped, ImageAngles, MagnificationCorrection, CTFParams[0].GetEwaldRadius(SizeFull, (float)species.PixelSize));
                        Reconstructions[isubset].BackProject(ExtractedCroppedFTq, ExtractedCTFCropped, ImageAngles, MagnificationCorrection, -CTFParams[0].GetEwaldRadius(SizeFull, (float)species.PixelSize));
                    }
                }
            }

            //CTFAvg.WriteMRC("d_ctfavg.mrc", true);

            //EmpiricalWeights.Dispose();

            CTFCoords.Dispose();
            CTFCoordsP.Dispose();
            CTFCoordsCropped.Dispose();
            GammaCorrection.Dispose();
            PhaseCorrection.Dispose();
            Extracted.Dispose();
            ExtractedFT.Dispose();
            ExtractedCropped.Dispose();
            ExtractedCroppedFTp.Dispose();
            ExtractedCroppedFTq.Dispose();
            ExtractedCTF.Dispose();
            ExtractedCTFCropped.Dispose();
            CTFWeights.Dispose();

            MaskParticle.Dispose();

            IntermediateMaskAngles.Dispose();
            IntermediateFTCorr.Dispose();
            IntermediateCTFP.Dispose();

            GPU.DestroyFFTPlan(PlanForwSuper);
            GPU.DestroyFFTPlan(PlanBackSuper);
            GPU.DestroyFFTPlan(PlanForw);

            species.HalfMap1Reconstruction[GPUID].FreeDevice();
            species.HalfMap2Reconstruction[GPUID].FreeDevice();
        }

        Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after backprojection of {Name}");

        #endregion

        for (int t = 0; t < NTilts; t++)
            TiltData[t]?.FreeDevice();

        Console.WriteLine($"{GPU.GetFreeMemory(GPUID)} MB after full refinement of {Name}");
    }

    public void PerformMultiParticleRefinementOneTiltMovie(string workingDirectory,
        ProcessingOptionsMPARefine optionsMPA,
        Species[] allSpecies,
        DataSource dataSource,
        Movie tiltMovie,
        Image[] tiltMovieData,
        int tiltID,
        Dictionary<Species, Particle[]> SpeciesParticles,
        Dictionary<Species, IntPtr> SpeciesParticleSubsets,
        Dictionary<Species, (int Start, int End)> SpeciesParticleIDRanges,
        Dictionary<Species, float[]> SpeciesContainmentMasks,
        Dictionary<Species, int> SpeciesRefinementSize,
        Dictionary<Species, int[]> SpeciesRelevantRefinementSizes,
        Dictionary<Species, Image> SpeciesFrameWeights,
        Dictionary<Species, int> SpeciesCTFSuperresFactor)
    {
        int GPUID = GPU.GetDevice();
        HeaderEER.GroupNFrames = dataSource.EERGroupFrames;
        NFrames = MapHeader.ReadFromFile(tiltMovie.DataPath).Dimensions.Z;
        //NFrames = 1;
        FractionFrames = 1;

        if (true)
        {
            #region Resize grids

            if (tiltMovie.PyramidShiftX == null || tiltMovie.PyramidShiftX.Count == 0 || tiltMovie.PyramidShiftX[0].Dimensions.Z != NFrames)
            {
                tiltMovie.PyramidShiftX = new List<CubicGrid>();
                tiltMovie.PyramidShiftY = new List<CubicGrid>();

                //tiltMovie.PyramidShiftX.Add(new CubicGrid(new int3(1, 1, NFrames)));
                tiltMovie.PyramidShiftX.Add(new CubicGrid(new int3(3, 3, 3)));

                //tiltMovie.PyramidShiftY.Add(new CubicGrid(new int3(1, 1, NFrames)));
                tiltMovie.PyramidShiftY.Add(new CubicGrid(new int3(3, 3, 3)));
            }

            #endregion

            #region Figure out dimensions

            tiltMovie.ImageDimensionsPhysical = new float2(new int2(MapHeader.ReadFromFile(tiltMovie.DataPath).Dimensions)) * (float)dataSource.PixelSizeMean;

            float MinDose = MathHelper.Min(Dose), MaxDose = MathHelper.Max(Dose);
            float TiltInterpolationCoord = (Dose[tiltID] - MinDose) / (MaxDose - MinDose);

            float SmallestAngPix = MathHelper.Min(allSpecies.Select(s => (float)s.PixelSize));
            float LargestBox = MathHelper.Max(allSpecies.Select(s => s.DiameterAngstrom)) * 2 / SmallestAngPix;

            decimal BinTimes = (decimal)Math.Log(SmallestAngPix / (float)dataSource.PixelSizeMean, 2.0);
            ProcessingOptionsTomoSubReconstruction OptionsDataLoad = new ProcessingOptionsTomoSubReconstruction()
            {
                PixelSize = dataSource.PixelSize,

                BinTimes = BinTimes,
                GainPath = dataSource.GainPath,
                GainHash = "",
                GainFlipX = dataSource.GainFlipX,
                GainFlipY = dataSource.GainFlipY,
                GainTranspose = dataSource.GainTranspose,
                DefectsPath = dataSource.DefectsPath,
                DefectsHash = "",

                Invert = true,
                NormalizeInput = true,
                NormalizeOutput = false,

                PrerotateParticles = true
            };

            #endregion

            foreach (var frame in tiltMovieData)
            {
                frame.Bandpass(1f / LargestBox, 1f, false, 0f);
                frame.Multiply(-1);
            }

            #region Extract particles

            Dictionary<Species, IntPtr[]> SpeciesParticleImages = new Dictionary<Species, IntPtr[]>();
            Dictionary<Species, float2[]> SpeciesParticleExtractedAt = new Dictionary<Species, float2[]>();

            int NParticlesOverall = 0;

            foreach (var species in allSpecies)
            {
                if (SpeciesParticles[species].Length == 0)
                    continue;

                Particle[] Particles = SpeciesParticles[species];
                int NParticles = Particles.Length;
                NParticlesOverall += NParticles;

                int Size = SpeciesRelevantRefinementSizes[species][tiltID]; // species.HalfMap1Projector[GPUID].Dims.X;
                long ElementsSliceComplex = (Size / 2 + 1) * Size;

                SpeciesParticleImages.Add(species, Helper.ArrayOfFunction(i =>
                {
                    long Footprint = (new int3(Size).Slice().ElementsFFT()) * 2 * (long)NParticles;

                    if (optionsMPA.UseHostMemory)
                        return GPU.MallocHostPinned(Footprint);
                    else
                        return GPU.MallocDevice(Footprint);
                }, NFrames));

                SpeciesParticleExtractedAt.Add(species, new float2[NParticles * NFrames]);
            }

            #endregion

            #region Helper functions

            Action<bool> ReextractPaddedParticles = (CorrectBeamTilt) =>
            {
                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                int BatchSize = optionsMPA.BatchSize;

                foreach (var species in allSpecies)
                {
                    if (!SpeciesParticles.ContainsKey(species) || SpeciesParticles[species].Length == 0)
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;

                    int SizeRelevant = SpeciesRelevantRefinementSizes[species][tiltID];
                    int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                    float AngPixRefine = species.ResolutionRefinement / 2;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                    float2[] ExtractedAt = SpeciesParticleExtractedAt[species];

                    Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper);
                    Image BeamTiltCorrection = CTF.GetBeamTilt(SizeRefineSuper, SizeFullSuper);

                    Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                    Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize));
                    Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                    Image ExtractedCroppedFTRelevantSize = new Image(IntPtr.Zero, new int3(SizeRelevant, SizeRelevant, BatchSize), true, true);
                    Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true);

                    Image CTFFrameWeights = tiltMovie.GetCTFsForOneParticle(OptionsDataLoad, new float3(0, 0, 0), CoordsCTF, null, true, true);

                    //Image SumAll = new Image(new int3(SizeRefine, SizeRefine, BatchSize));

                    int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                    int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                    int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);


                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                        IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                        float3[] CoordinatesMoving = BatchParticles.Select(p => p.GetCoordinatesAt(TiltInterpolationCoord)).ToArray();

                        float3[] CoordinatesTilt = GetPositionsInOneTilt(CoordinatesMoving, tiltID);

                        for (int f = 0; f < NFrames; f++)
                        {
                            float3[] ImageCoords = tiltMovie.GetPositionsInOneFrame(CoordinatesTilt, f);

                            float[] Defoci = new float[CurBatch];
                            int3[] ExtractOrigins = new int3[CurBatch];
                            float3[] ResidualShifts = new float3[BatchSize];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                Defoci[p] = CoordinatesTilt[p].Z;
                                ExtractedAt[(batchStart + p) * NFrames + f] = new float2(ImageCoords[p]);
                            }

                            GPU.Extract(tiltMovieData[f].GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                tiltMovieData[f].Dims.Slice(),
                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                Helper.ToInterleaved(ExtractOrigins),
                                true,
                                (uint)CurBatch);

                            GPU.FFT(Extracted.GetDevice(Intent.Read),
                                ExtractedFT.GetDevice(Intent.Write),
                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                (uint)CurBatch,
                                PlanForwSuper);

                            ExtractedFT.ShiftSlices(ResidualShifts);
                            ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                            GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                new int3(SizeFullSuper, SizeFullSuper, 1),
                                new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                (uint)CurBatch);

                            GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, null, tiltID, ExtractedCTF, true);

                            if (CorrectBeamTilt)
                                GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                    BeamTiltCorrection.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    BeamTiltCorrection.ElementsSliceComplex,
                                    (uint)CurBatch);

                            GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                ExtractedCTF.GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                ExtractedCTF.ElementsReal,
                                1);

                            GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                CTFFrameWeights.GetDeviceSlice(f, Intent.Read),
                                ExtractedFT.GetDevice(Intent.Write),
                                CTFFrameWeights.ElementsSliceReal,
                                (uint)CurBatch);

                            GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                (uint)CurBatch,
                                PlanBackSuper,
                                false);

                            GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                ExtractedCropped.GetDevice(Intent.Write),
                                new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                new int3(SizeRefine, SizeRefine, 1),
                                (uint)CurBatch);

                            GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                ExtractedCropped.GetDevice(Intent.Write),
                                ExtractedCropped.Dims.Slice(),
                                ParticleDiameterPix / 2f,
                                16 * AngPixExtract / AngPixRefine,
                                true,
                                (uint)CurBatch);

                            //SumAll.Add(ExtractedCropped);

                            GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                ExtractedCroppedFT.GetDevice(Intent.Write),
                                new int3(SizeRefine, SizeRefine, 1),
                                (uint)CurBatch,
                                PlanForw);

                            ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                            GPU.CropFT(ExtractedCroppedFT.GetDevice(Intent.Read),
                                ExtractedCroppedFTRelevantSize.GetDevice(Intent.Write),
                                new int3(SizeRefine).Slice(),
                                new int3(SizeRelevant).Slice(),
                                (uint)CurBatch);

                            GPU.CopyDeviceToHostPinned(ExtractedCroppedFTRelevantSize.GetDevice(Intent.Read),
                                new IntPtr((long)SpeciesParticleImages[species][f] + (new int3(SizeRelevant).Slice().ElementsFFT()) * 2 * batchStart * sizeof(float)),
                                (new int3(SizeRelevant).Slice().ElementsFFT()) * 2 * CurBatch);
                        }
                    }

                    //SumAll.AsReducedAlongZ().WriteMRC("d_sumall.mrc", true);
                    //SumAll.Dispose();

                    CTFFrameWeights.Dispose();

                    CoordsCTF.Dispose();
                    Extracted.Dispose();
                    ExtractedFT.Dispose();
                    ExtractedCropped.Dispose();
                    ExtractedCroppedFT.Dispose();
                    ExtractedCroppedFTRelevantSize.Dispose();
                    ExtractedCTF.Dispose();

                    GPU.DestroyFFTPlan(PlanForwSuper);
                    GPU.DestroyFFTPlan(PlanBackSuper);
                    GPU.DestroyFFTPlan(PlanForw);
                }
            };

            Func<float2, float[]> GetRawCC = (shiftBias) =>
            {
                float[] Result = new float[NParticlesOverall * NFrames * 3];

                for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                {
                    Species Species = allSpecies[ispecies];
                    Particle[] Particles = SpeciesParticles[Species];

                    int NParticles = Particles.Length;
                    float SpeciesAngPix = Species.ResolutionRefinement / 2;
                    if (NParticles == 0)
                        continue;

                    float[] SpeciesResult = new float[NParticles * NFrames * 3];

                    float3[] ParticlePositions = new float3[NParticles * NFrames];
                    float3[] ParticleAngles = new float3[NParticles * NFrames];
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3 Position = Particles[p].GetCoordinatesAt(TiltInterpolationCoord);
                        float3 Angles = Particles[p].GetAnglesAt(TiltInterpolationCoord);

                        for (int f = 0; f < NFrames; f++)
                        {
                            ParticlePositions[p * NFrames + f] = Position;
                            ParticleAngles[p * NFrames + f] = Angles;
                        }
                    }

                    float3[] ParticlePositionsTilt = GetPositionsInOneTilt(ParticlePositions, tiltID);

                    float3[] ParticlePositionsProjected = tiltMovie.GetPositionInAllFrames(ParticlePositionsTilt);
                    float3[] ParticleAnglesInFrames = GetAnglesInOneTilt(ParticlePositions, ParticleAngles, tiltID);

                    float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[Species];
                    float2[] ParticleShifts = new float2[NFrames * NParticles];
                    for (int p = 0; p < NParticles; p++)
                    for (int t = 0; t < NFrames; t++)
                        ParticleShifts[p * NFrames + t] = (new float2(ParticlePositionsProjected[p * NFrames + t]) - ParticleExtractedAt[p * NFrames + t] + shiftBias) / SpeciesAngPix;

                    int SizeRelevant = SpeciesRelevantRefinementSizes[Species][tiltID];
                    int SizeRefine = Species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;

                    Image PhaseCorrection = CTF.GetBeamTilt(SizeRefine, SizeFull);
                    float2[] BeamTilts = Helper.ArrayOfConstant(CTF.BeamTilt, NParticles);
                    Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                    for (int t = 0; t < NFrames; t++)
                        GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                            PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                            PhaseCorrection.Dims.Slice(),
                            new int3(SizeRelevant).Slice(),
                            1);

                    GPU.MultiParticleDiff(SpeciesResult,
                        SpeciesParticleImages[Species],
                        SpeciesRefinementSize[Species],
                        Helper.ArrayOfConstant(SizeRelevant, NFrames),
                        Helper.ToInterleaved(ParticleShifts),
                        Helper.ToInterleaved(ParticleAnglesInFrames),
                        MagnificationCorrection.ToVec(),
                        SpeciesFrameWeights[Species].GetDevice(Intent.ReadWrite),
                        PhaseCorrectionAll.GetDevice(Intent.Read),
                        0,
                        Species.CurrentMaxShellRefinement,
                        new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                        new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                        Species.HalfMap1Projector[GPUID].Oversampling,
                        Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                        SpeciesParticleSubsets[Species],
                        NParticles,
                        NFrames);

                    PhaseCorrectionAll.Dispose();
                    PhaseCorrection.Dispose();

                    int Offset = SpeciesParticleIDRanges[Species].Start * NFrames * 3;
                    Array.Copy(SpeciesResult, 0, Result, Offset, SpeciesResult.Length);
                }

                return Result;
            };

            Func<(float[] xp, float[] xm, float[] yp, float[] ym, float delta2)> GetRawShiftGradients = () =>
            {
                float Delta = 0.025f;
                float Delta2 = Delta * 2;

                float[] h_ScoresXP = GetRawCC(float2.UnitX * Delta);
                float[] h_ScoresXM = GetRawCC(-float2.UnitX * Delta);
                float[] h_ScoresYP = GetRawCC(float2.UnitY * Delta);
                float[] h_ScoresYM = GetRawCC(-float2.UnitY * Delta);

                //for (int i = 0; i < Result.Length; i++)
                //    Result[i] = new float2((h_ScoresXP[i] - h_ScoresXM[i]) / Delta2 * 100,
                //                           (h_ScoresYP[i] - h_ScoresYM[i]) / Delta2 * 100);

                return (h_ScoresXP, h_ScoresXM, h_ScoresYP, h_ScoresYM, Delta2);
            };

            Func<double[]> GetPerFrameDiff2 = () =>
            {
                double[] Result = new double[NFrames * 3];
                float[] RawResult = GetRawCC(new float2(0));

                for (int p = 0; p < NParticlesOverall; p++)
                for (int f = 0; f < NFrames; f++)
                {
                    Result[f * 3 + 0] += RawResult[(p * NFrames + f) * 3 + 0];
                    Result[f * 3 + 1] += RawResult[(p * NFrames + f) * 3 + 1];
                    Result[f * 3 + 2] += RawResult[(p * NFrames + f) * 3 + 2];
                }

                Result = Helper.ArrayOfFunction(t => Result[t * 3 + 0] / Math.Max(1e-10, Math.Sqrt(Result[t * 3 + 1] * Result[t * 3 + 2])) * 100 * NParticlesOverall, NFrames);

                return Result;
            };

            #endregion

            ReextractPaddedParticles(false);

            float2[][] OriginalOffsets = Helper.ArrayOfFunction(p => Helper.ArrayOfFunction(t => new float2(tiltMovie.PyramidShiftX[p].Values[t],
                        tiltMovie.PyramidShiftY[p].Values[t]),
                    tiltMovie.PyramidShiftX[p].Values.Length),
                tiltMovie.PyramidShiftX.Count);

            int BFGSIterations = 0;

            double[] InitialParametersWarp = new double[tiltMovie.PyramidShiftX.Select(g => g.Values.Length * 2).Sum()];

            #region Set parameters from vector

            Action<double[], Movie> SetWarpFromVector = (input, movie) =>
            {
                int Offset = 0;

                int3[] PyramidDimensions = tiltMovie.PyramidShiftX.Select(g => g.Dimensions).ToArray();

                movie.PyramidShiftX.Clear();
                movie.PyramidShiftY.Clear();

                for (int p = 0; p < PyramidDimensions.Length; p++)
                {
                    float[] MovementXData = new float[PyramidDimensions[p].Elements()];
                    float[] MovementYData = new float[PyramidDimensions[p].Elements()];
                    for (int i = 0; i < MovementXData.Length; i++)
                    {
                        MovementXData[i] = OriginalOffsets[p][i].X + (float)input[Offset + i];
                        MovementYData[i] = OriginalOffsets[p][i].Y + (float)input[Offset + MovementXData.Length + i];
                    }

                    movie.PyramidShiftX.Add(new CubicGrid(PyramidDimensions[p], MovementXData));
                    movie.PyramidShiftY.Add(new CubicGrid(PyramidDimensions[p], MovementYData));

                    Offset += MovementXData.Length * 2;
                }
            };

            #endregion

            #region Wiggle weights

            int NWiggleDifferentiable = tiltMovie.PyramidShiftX.Select(g => g.Values.Length * 2).Sum();
            (int[] indices, float2[] weights)[] AllWiggleWeights = new (int[] indices, float2[] weights)[NWiggleDifferentiable];

            {
                Movie[] ParallelMovieCopies = Helper.ArrayOfFunction(i => new Movie(tiltMovie.Path), 32);
                Dictionary<Species, float3[]> SpeciesParticlePositions = new Dictionary<Species, float3[]>();
                foreach (var species in allSpecies)
                {
                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    float3[] ParticlePositions = new float3[NParticles * NFrames];
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3 Position = Particles[p].GetCoordinatesAt(TiltInterpolationCoord);

                        for (int f = 0; f < NFrames; f++)
                            ParticlePositions[p * NFrames + f] = Position;
                    }

                    float3[] ParticlePositionsTilt = GetPositionsInOneTilt(ParticlePositions, tiltID);
                    SpeciesParticlePositions.Add(species, ParticlePositionsTilt);
                }

                Helper.ForCPU(0, NWiggleDifferentiable, ParallelMovieCopies.Length, (threadID) =>
                    {
                        ParallelMovieCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                        ParallelMovieCopies[threadID].NFrames = NFrames;
                        ParallelMovieCopies[threadID].FractionFrames = FractionFrames;
                    },
                    (iwiggle, threadID) =>
                    {
                        double[] WiggleParams = new double[InitialParametersWarp.Length];
                        WiggleParams[iwiggle] = 1;
                        SetWarpFromVector(WiggleParams, ParallelMovieCopies[threadID]);

                        float2[] RawShifts = new float2[NParticlesOverall * NFrames];
                        foreach (var species in allSpecies)
                        {
                            Particle[] Particles = SpeciesParticles[species];
                            int NParticles = Particles.Length;
                            if (NParticles == 0)
                                continue;

                            int Offset = SpeciesParticleIDRanges[species].Start;

                            float[] ContainmentMask = SpeciesContainmentMasks[species];

                            float3[] ParticlePositionsProjected = ParallelMovieCopies[threadID].GetPositionInAllFrames(SpeciesParticlePositions[species]);
                            float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                            for (int p = 0; p < NParticles; p++)
                            for (int f = 0; f < NFrames; f++)
                                RawShifts[(Offset + p) * NFrames + f] = (new float2(ParticlePositionsProjected[p * NFrames + f]) - ParticleExtractedAt[p * NFrames + f]) * ContainmentMask[p * NTilts + tiltID];
                        }

                        List<int> Indices = new List<int>(RawShifts.Length / 5);
                        List<float2> Weights = new List<float2>(RawShifts.Length / 5);
                        for (int i = 0; i < RawShifts.Length; i++)
                        {
                            if (RawShifts[i].LengthSq() > 1e-6f)
                            {
                                Indices.Add(i);
                                Weights.Add(RawShifts[i]);

                                if (Math.Abs(RawShifts[i].X) > 1.05f)
                                    throw new Exception();
                            }
                        }

                        AllWiggleWeights[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                    }, null);
            }

            #endregion

            #region Loss and gradient functions for warping

            Func<double[], double> WarpEval = input =>
            {
                SetWarpFromVector(input, tiltMovie);

                double[] TiltScores = GetPerFrameDiff2();
                double Score = TiltScores.Sum();

                Console.WriteLine(Score);

                return Score;
            };

            Func<double[], double[]> WarpGrad = input =>
            {
                double[] Result = new double[input.Length];

                if (++BFGSIterations >= 12)
                    return Result;

                SetWarpFromVector(input, tiltMovie);
                (var XP, var XM, var YP, var YM, var Delta2Movement) = GetRawShiftGradients();

                Parallel.For(0, AllWiggleWeights.Length, iwiggle =>
                {
                    double SumGrad = 0;
                    double SumWeights = 0;
                    double SumWeightsGrad = 0;

                    int[] Indices = AllWiggleWeights[iwiggle].indices;
                    float2[] Weights = AllWiggleWeights[iwiggle].weights;

                    for (int i = 0; i < Indices.Length; i++)
                    {
                        int id = Indices[i];

                        SumWeights += Math.Abs(Weights[i].X) * Math.Sqrt(XP[id * 3 + 1] + XM[id * 3 + 1]) +
                                      Math.Abs(Weights[i].Y) * Math.Sqrt(YP[id * 3 + 1] + YM[id * 3 + 1]);
                        SumWeightsGrad += Math.Abs(Weights[i].X) + Math.Abs(Weights[i].Y);

                        double GradX = (XP[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(XP[id * 3 + 1] * XP[id * 3 + 2])) -
                                        XM[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(XM[id * 3 + 1] * XM[id * 3 + 2]))) / Delta2Movement;
                        double GradY = (YP[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(YP[id * 3 + 1] * YP[id * 3 + 2])) -
                                        YM[id * 3 + 0] / Math.Max(1e-15, Math.Sqrt(YM[id * 3 + 1] * YM[id * 3 + 2]))) / Delta2Movement;

                        SumGrad += Weights[i].X * Math.Sqrt(XP[id * 3 + 1] + XM[id * 3 + 1]) * GradX;
                        SumGrad += Weights[i].Y * Math.Sqrt(YP[id * 3 + 1] + YM[id * 3 + 1]) * GradY;
                    }

                    Result[iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                });

                return Result;
            };

            #endregion


            foreach (var species in allSpecies)
                species.CurrentMaxShellRefinement = species.HalfMap1Projector[GPUID].Dims.X / 2;

            BroydenFletcherGoldfarbShanno OptimizerWarp = new BroydenFletcherGoldfarbShanno(InitialParametersWarp.Length, WarpEval, WarpGrad);

            SetWarpFromVector(InitialParametersWarp, tiltMovie);

            BFGSIterations = 0;
            OptimizerWarp.Maximize(InitialParametersWarp);

            SetWarpFromVector(InitialParametersWarp, tiltMovie);

            #region Compute FSC between refs and particles to estimate frame and micrograph weights

            if (false)
            {
                int FSCLength = 64;
                Image FSC = new Image(new int3(FSCLength, FSCLength, NFrames * 3), true);
                Image FSCPerParticle = new Image(new int3(FSCLength / 2, NParticlesOverall * 3, 1));
                //float[][] FSCData = FSC.GetHost(Intent.ReadWrite);
                Image PhaseResiduals = new Image(new int3(FSCLength, FSCLength, 2), true);

                Star TableOut = new Star(new string[] { "wrpNormCoordinateX", "wrpNormCoordinateY", "wrpNormCoordinateZ" });

                int BatchSize = optionsMPA.BatchSize;
                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                {
                    Species Species = allSpecies[ispecies];
                    Particle[] Particles = SpeciesParticles[Species];

                    int NParticles = Particles.Length;
                    float SpeciesAngPix = Species.ResolutionRefinement / 2;
                    if (NParticles == 0)
                        continue;

                    int SpeciesOffset = SpeciesParticleIDRanges[Species].Start;

                    int SizeRefine = SpeciesRefinementSize[Species];
                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[Species];

                    float ScaleFactor = (float)Species.PixelSize * (FSCLength / 2 - 1) /
                                        (float)(Species.ResolutionRefinement / 2 * (SizeRefine / 2 - 1));

                    {
                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[Species];
                        int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[Species];
                        float AngPixRefine = Species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(Species.DiameterAngstrom / AngPixRefine);

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper); // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine
                        Image CoordsCTFCropped = CTF.GetCTFCoords(SizeRefine, SizeRefine);

                        Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                        Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
                        Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true);

                        int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                        int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
                        int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

                        if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                            throw new Exception("No FFT plans created!");

                        Image BeamTiltCorrection = CTF.GetBeamTilt(SizeRefineSuper, SizeFullSuper);

                        for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                            IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                            float3[] CoordinatesMoving = BatchParticles.Select(p => p.GetCoordinatesAt(TiltInterpolationCoord)).ToArray();
                            float3[] AnglesMoving = BatchParticles.Select(p => p.GetAnglesAt(TiltInterpolationCoord)).ToArray();

                            float3[] CoordinatesTilt = GetPositionsInOneTilt(CoordinatesMoving, tiltID);
                            float3[] ParticleAnglesInFrames = GetAnglesInOneTilt(CoordinatesMoving, AnglesMoving, tiltID);

                            for (int i = 0; i < CurBatch; i++)
                            {
                                float3 Coords = new float3(CoordinatesMoving[i].X, CoordinatesMoving[i].Y, CoordinatesMoving[i].Z);
                                Coords /= VolumeDimensionsPhysical;
                                TableOut.AddRow(new string[]
                                {
                                    Coords.X.ToString(CultureInfo.InvariantCulture),
                                    Coords.Y.ToString(CultureInfo.InvariantCulture),
                                    Coords.Z.ToString(CultureInfo.InvariantCulture)
                                });
                            }

                            for (int f = 0; f < NFrames; f++)
                            {
                                float3[] ImageCoords = tiltMovie.GetPositionsInOneFrame(CoordinatesTilt, f);

                                float[] Defoci = new float[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p] = CoordinatesTilt[p].Z;
                                }

                                GPU.Extract(tiltMovieData[f].GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    tiltMovieData[f].Dims.Slice(),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    Helper.ToInterleaved(ExtractOrigins),
                                    true,
                                    (uint)CurBatch);

                                GPU.FFT(Extracted.GetDevice(Intent.Read),
                                    ExtractedFT.GetDevice(Intent.Write),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    (uint)CurBatch,
                                    PlanForwSuper);

                                ExtractedFT.ShiftSlices(ResidualShifts);
                                ExtractedFT.Multiply(1f / (SizeFullSuper * SizeFullSuper));

                                GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    new int3(SizeFullSuper, SizeFullSuper, 1),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    (uint)CurBatch);

                                GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTF, null, tiltID, ExtractedCTF, true);

                                GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                    BeamTiltCorrection.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    BeamTiltCorrection.ElementsSliceComplex,
                                    (uint)CurBatch);

                                GPU.MultiplyComplexSlicesByScalar(Extracted.GetDevice(Intent.Read),
                                    ExtractedCTF.GetDevice(Intent.Read),
                                    ExtractedFT.GetDevice(Intent.Write),
                                    ExtractedCTF.ElementsReal,
                                    1);

                                GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    (uint)CurBatch,
                                    PlanBackSuper,
                                    false);

                                GPU.CropFTFull(Extracted.GetDevice(Intent.Read),
                                    ExtractedCropped.GetDevice(Intent.Write),
                                    new int3(SizeRefineSuper, SizeRefineSuper, 1),
                                    new int3(SizeRefine, SizeRefine, 1),
                                    (uint)CurBatch);

                                GPU.SphereMask(ExtractedCropped.GetDevice(Intent.Read),
                                    ExtractedCropped.GetDevice(Intent.Write),
                                    ExtractedCropped.Dims.Slice(),
                                    ParticleDiameterPix / 2f,
                                    16 * AngPixExtract / AngPixRefine,
                                    true,
                                    (uint)CurBatch);

                                //SumAll.Add(ExtractedCropped);

                                GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                    ExtractedCroppedFT.GetDevice(Intent.Write),
                                    new int3(SizeRefine, SizeRefine, 1),
                                    (uint)CurBatch,
                                    PlanForw);

                                ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                GetCTFsForOneTilt(AngPixRefine, Defoci, CoordinatesTilt, CoordsCTFCropped, null, tiltID, ExtractedCTF, true, true, true);


                                //GPU.MultiParticleCorr2D(CorrAB.GetDeviceSlice(f, Intent.ReadWrite),
                                //                        CorrA2.GetDeviceSlice(f, Intent.ReadWrite),
                                //                        CorrB2.GetDeviceSlice(f, Intent.ReadWrite),
                                //                        new IntPtr[] { ExtractedCroppedFT.GetDevice(Intent.Read) },
                                //                        SizeRefine,
                                //                        null,
                                //                        new float[CurBatch * 2],
                                //                        Helper.ToInterleaved(ParticleAnglesInFrames),
                                //                        MagnificationCorrection * new float3(Species.HalfMap1Projector[GPUID].Oversampling,
                                //                                                             Species.HalfMap1Projector[GPUID].Oversampling,
                                //                                                             1),
                                //                        new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                //                        new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                //                        Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                //                        new IntPtr((long)SpeciesParticleSubsets[Species] + batchStart * sizeof(int)),
                                //                        CurBatch,
                                //                        1);


                                GPU.MultiParticleCorr2D(FSC.GetDeviceSlice(f * 3, Intent.ReadWrite),
                                    new IntPtr((long)FSCPerParticle.GetDevice(Intent.ReadWrite) + (SpeciesOffset + batchStart) * FSCPerParticle.Dims.X * 3 * sizeof(float)),
                                    PhaseResiduals.GetDevice(Intent.ReadWrite),
                                    FSCLength,
                                    new IntPtr[] { ExtractedCroppedFT.GetDevice(Intent.Read) },
                                    ExtractedCTF.GetDevice(Intent.Read),
                                    SizeRefine,
                                    ScaleFactor,
                                    null,
                                    new float[CurBatch * 2],
                                    Helper.ToInterleaved(ParticleAnglesInFrames),
                                    MagnificationCorrection.ToVec(),
                                    0,
                                    new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                                    new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                                    Species.HalfMap1Projector[GPUID].Oversampling,
                                    Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                                    new IntPtr((long)SpeciesParticleSubsets[Species] + batchStart * sizeof(int)),
                                    CurBatch,
                                    1);
                            }
                        }

                        BeamTiltCorrection.Dispose();

                        CoordsCTFCropped.Dispose();
                        CoordsCTF.Dispose();
                        Extracted.Dispose();
                        ExtractedFT.Dispose();
                        ExtractedCropped.Dispose();
                        ExtractedCroppedFT.Dispose();
                        ExtractedCTF.Dispose();

                        GPU.DestroyFFTPlan(PlanForwSuper);
                        GPU.DestroyFFTPlan(PlanBackSuper);
                        GPU.DestroyFFTPlan(PlanForw);
                    }
                }

                FSC.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", $"{RootName}_tilt{tiltID:D3}_fsc.mrc"), true);
                FSC.Dispose();

                FSCPerParticle.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", $"{RootName}_tilt{tiltID:D3}_fscparticles.mrc"), true);
                FSCPerParticle.Dispose();

                PhaseResiduals.WriteMRC(System.IO.Path.Combine(workingDirectory, "..", RootName + "_phaseresiduals.mrc"), true);
                PhaseResiduals.Dispose();

                TableOut.Save(System.IO.Path.Combine(workingDirectory, "..", $"{RootName}_tilt{tiltID:D3}_fscparticles.star"));
            }

            #endregion

            #region Tear down

            foreach (var pair in SpeciesParticleImages)
            {
                foreach (var ptr in SpeciesParticleImages[pair.Key])
                    if (optionsMPA.UseHostMemory)
                        GPU.FreeHostPinned(ptr);
                    else
                        GPU.FreeDevice(ptr);
            }

            #endregion
        }
    }

    public override long MultiParticleRefinementCalculateHostMemory(ProcessingOptionsMPARefine optionsMPA,
        Species[] allSpecies,
        DataSource dataSource)
    {
        long Result = 0;

        string DataHash = GetDataHash();
        int GPUID = GPU.GetDevice();

        foreach (var species in allSpecies)
        {
            int NParticles = species.GetParticles(DataHash).Length;

            int Size = species.HalfMap1Projector[GPUID].Dims.X;
            int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;

            int[] RelevantSizes = GetRelevantImageSizes(SizeFull, (float)optionsMPA.BFactorWeightingThreshold).Select(v => Math.Min(Size, v)).ToArray();

            Result += Helper.ArrayOfFunction(t => (new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles * sizeof(float), NTilts).Sum();
        }

        return Result;
    }
}